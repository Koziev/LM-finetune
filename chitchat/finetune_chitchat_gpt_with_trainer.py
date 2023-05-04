"""
Тренировка модели болталки Axioma.
Эксперимент с файнтюном: токены истории диалога не включаем в backprop, присваивая соответствующим целям (labels) значение -100
"""
import logging
import os
import json
import sys
import io
import random
import itertools
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tqdm
import sklearn.model_selection
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import HfArgumentParser
from pynvml import *


proj_dir = os.path.expanduser('~/polygon/chatbot')


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


def load_samples(dataset_path, tokenizer):
    samples = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        for sample in tqdm.tqdm(data, desc='Loading samples', total=len(data)):
            try:
                lines = []
                for i, msg in enumerate(sample):
                    if 0 == (i % 2):
                        lines.append('человек: ' + msg)
                    else:
                        lines.append('чатбот: ' + msg)

                text = '\n'.join(lines)
                tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                if len(tokens) < 512:
                    samples.append({'tokens': tokens, 'text': text})
                else:
                    lines0 = list(lines)

                    lines = lines[:-1]
                    while len(lines) > 1:
                        text = '\n'.join(lines)
                        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                        if len(tokens) < 512:
                            samples.append({'tokens': tokens, 'text': text})
                            break
                        else:
                            lines = lines[:-1]


            except Exception as ex:
                print(ex)

    return samples


class FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]

        for sample in samples:
            input_ids = [self.bos_token_id] + sample['tokens'] + [self.eos_token_id]
            labels = input_ids
            attention_map = [1] * len(labels)
            self.samples.append((input_ids, labels, attention_map))
            self.max_len = max(self.max_len, len(input_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        input_ids, labels, attention_map = self.samples[index]
        npad = self.max_len - len(input_ids)
        input_ids = input_ids + npad * [self.pad_token_id]
        labels = labels + [-100] * npad
        attention_mask = attention_map + [0] * npad
        return {'input_ids': torch.LongTensor(input_ids),
                'labels': torch.LongTensor(labels),
                'attention_mask': torch.LongTensor(attention_mask)}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='sberbank-ai/rugpt3medium_based_on_gpt2',
        metadata={"help": "The model checkpoint for weights initialization."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default=os.path.join(proj_dir, 'tmp', 'axioma_dialogues.solid.json'),
        metadata={"help": "Путь к датасету со диалогами"}
    )


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.output_dir:
        training_args.output_dir = os.path.join(proj_dir, 'tmp', 'rugpt_chitchat')

    rank0 = training_args.local_rank in (-1, 0)

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
    if rank0:
        tensorboard_dir = os.path.join(training_args.output_dir, 'runs')
        if os.path.exists(tensorboard_dir):
            logger.info('Removing "%s"', tensorboard_dir)
            shutil.rmtree(tensorboard_dir)

    pretrained_model_name = model_args.model_name_or_path

    print('Loading pretrained model "{}"...'.format(pretrained_model_name))
    if 'xglm' in pretrained_model_name.lower():
        tokenizer = transformers.XGLMTokenizer.from_pretrained(pretrained_model_name)
        model = transformers.XGLMForCausalLM.from_pretrained(pretrained_model_name)
    elif 'bloom' in pretrained_model_name:
        tokenizer = transformers.BloomTokenizer.from_pretrained(pretrained_model_name)
        model = transformers.BloomForCausalLM.from_pretrained(pretrained_model_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name)

    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    if rank0:
        print_gpu_utilization()

    print('\nTokenizer:')
    for token in '<s> </s> <pad>'.split():
        print('token "{}" id={}'.format(token, tokenizer.encode(token, add_special_tokens=False)))

    print('\nLoading dataset...')
    train_samples = load_samples(data_args.dataset_path, tokenizer)
    print('Train samples: {}'.format(len(train_samples)))

    train_dataset = FinetuneDataset(train_samples, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        # compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    logger.info('Start training...')
    try:
        train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    except KeyboardInterrupt as ex:
        print('!!! CTRL+C !!!')

    logger.info(f'Saving the model and tokenizer to "%s"', training_args.output_dir)
    trainer.save_model(output_dir=training_args.output_dir)
    #model.save_pretrained(training_args.output_dir)
    #tokenizer.save_pretrained(training_args.output_dir)

    logger.info('All done :)')
