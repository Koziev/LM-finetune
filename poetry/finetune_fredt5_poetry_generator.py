"""
Тренировка модели генерации стихов на претрейненной модели FRED T5 XL
"""
import logging
import os
import json
import io
import random
import itertools
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import argparse

import numpy as np
import tqdm
import sklearn.model_selection
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import HfArgumentParser
from pynvml import *


proj_dir = os.path.expanduser('~/polygon/text_generator')


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


def load_samples(dataset_path, tokenizer):
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            try:
                input_text = '<LM>' + sample['prompt_text']

                # Вариант с генерацией обычного текста
                output_text = sample['output_text']
                output_text = '\n'.join(output_text.split('\n')[:4])

                input_tokens = tokenizer.encode(input_text, add_special_tokens=False, truncation=True, max_length=512)
                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                samples.append({'input_tokens': input_tokens, 'output_tokens': output_tokens,
                                'input_text': input_text, 'output_text': output_text})

            except Exception as ex:
                print(ex)

    return samples


class FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_input_len = 0
        self.max_output_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]

        for sample in samples:
            input_ids = sample['input_tokens']
            output_ids = sample['output_tokens'] + [self.eos_token_id]
            self.samples.append((input_ids, output_ids))
            self.max_input_len = max(self.max_input_len, len(input_ids))
            self.max_output_len = max(self.max_output_len, len(output_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        input_ids, output_ids = self.samples[index]

        input_npad = self.max_input_len - len(input_ids)
        attention_mask = [1]*len(input_ids) + [0]*input_npad
        input_ids = input_ids + input_npad * [self.pad_token_id]

        output_npad = self.max_output_len - len(output_ids)
        labels = output_ids + output_npad * [-100]

        return {'input_ids': torch.LongTensor(input_ids),
                'attention_mask': attention_mask,
                'labels': torch.LongTensor(labels),
                }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='ai-forever/FRED-T5-1.7B',
        metadata={"help": "The model checkpoint for weights initialization."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default=os.path.join(proj_dir, 'tmp', os.path.join(proj_dir, 'tmp', 'long_poems_gpt_dataset.jsonl')),
        metadata={"help": "Путь к датасету со стихами"}
    )


class MyPrinterCallback(TrainerCallback):
    def __init__(self, filepath):
        self.wrt = open(filepath, 'w')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'epoch' in logs and 'loss' in logs:
                self.wrt.write('{}\t{}\n'.format(logs['epoch'], logs['loss']))
                self.wrt.flush()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.optim:
        training_args.optim = "adafactor"

    if not training_args.output_dir:
        training_args.output_dir = os.path.join(proj_dir, 'tmp', os.path.join(proj_dir, 'tmp', 't5_poetry_generator'))

    verbose = training_args.local_rank in (-1, 0)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Удаляем старые логи tensorboard
    tensorboard_dir = os.path.join(training_args.output_dir, 'runs')
    if os.path.exists(tensorboard_dir):
        logger.info('Removing "%s"', tensorboard_dir)
        shutil.rmtree(tensorboard_dir)

    device = training_args.device
    logging.info('device={}'.format(device))

    pretrained_model_name = model_args.model_name_or_path

    logger.info('Loading pretrained model "%s"', pretrained_model_name)
    if 'FRED-T5' in pretrained_model_name:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(pretrained_model_name)
    else:
        tokenizer = transformers.T5Tokenizer.from_pretrained(pretrained_model_name)

    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
    model.to(device)
    print_gpu_utilization()

    logger.info('\nLoading dataset "%s"...', data_args.dataset_path)
    train_samples = load_samples(data_args.dataset_path, tokenizer)
    logger.info('Train samples: %d', len(train_samples))

    train_dataset = FinetuneDataset(train_samples, tokenizer)
    # test_dataset = FinetuneDataset(test_samples, tokenizer)

    printer = MyPrinterCallback(os.path.join(proj_dir, 'tmp', 'finetune_fredt5_poetry_generator.loss.log'))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        # compute_metrics=compute_metrics,
        callbacks=[printer]  #[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    logger.info('Start training...')
    train_result = trainer.train()

    logger.info(f'Saving the model and tokenizer')
    trainer.save_model(output_dir=training_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info('All done :)')
