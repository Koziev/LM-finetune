"""
Эксперимент с файнтюном модели LLaMa на стихах.
"""

import glob
import logging
import os
import json
import io
import random
import itertools
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
from pathlib import Path

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
import deepspeed

proj_dir = os.path.expanduser('~/polygon/text_generator')


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    # logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def pad_sequence(sequence, pad_id, max_len):
    l = len(sequence)
    if l < max_len:
        return sequence + [pad_id] * (max_len - l)
    else:
        return sequence


def load_samples(data_args, tokenizer):
    samples = []
    with open(data_args.dataset_path, 'r') as f:
        for sample_str in f:
            sample = json.loads(sample_str)
            prompt = sample['prompt_text']
            if prompt:
                if data_args.output_syllables:
                    # Вариант с генерацией цепочки слогов
                    lines = []
                    for line in sample['output'].split('<nl>'):
                        line = line.strip()
                        tokens = line.split(' ')
                        tokens = tokens[::-1]
                        line = ' '.join(tokens)
                        line = line.replace(' | ', '|')
                        line = line.replace(' ', '\u2010')
                        line = line.replace('|', ' ')
                        lines.append(line)
                    output_text = '\n'.join(lines)
                else:
                    output_text = sample['output_text']

                    # 29.04.2023 ограничим 2 первым катренами
                    output_text = '\n\n'.join(output_text.split('\n\n')[:2])

                input_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                samples.append((input_tokens, output_tokens, prompt, output_text))

                if data_args.max_samples > 0 and len(samples) >= data_args.max_samples:
                    break

    return samples


class FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        assert (len(tokenizer.encode('#', add_special_tokens=False)) == 1)
        self.sep_token_id = tokenizer.encode('#', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.pad_token_id

        for src_ids, output_ids, src_text, output_text in samples:
            input_ids = [self.bos_token_id] + src_ids + [self.sep_token_id] + output_ids + [self.eos_token_id]

            # Токены затравки дают label=-100
            labels = [-100] + [-100] * len(src_ids) + [-100] + output_ids + [self.eos_token_id]

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
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='decapoda-research/llama-7b-hf',
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Load model in 8-bit"},
    )


@dataclass
class DataSetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default=os.path.join(proj_dir, 'tmp', os.path.join(proj_dir, 'tmp', 'лирика.jsonl')),
        metadata={"help": "Путь к датасету со стихами"}
    )

    output_syllables: Optional[bool] = field(
        default=False,
        metadata={"help": "Силлабо-тоническое представление выходного текста"}
    )

    max_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "Максимальное кол-во сэмплов, считываемых из датасета"}
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
    parser = HfArgumentParser((ModelArguments, DataSetArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Удаляем старые логи tensorboard
    if training_args.local_rank in (-1, 0):
        for f in glob.glob(training_args.output_dir + '/*'):
            if os.path.isfile(f):
                os.remove(f)

        tensorboard_dir = os.path.join(training_args.output_dir, 'runs')
        if os.path.exists(tensorboard_dir):
            logger.info('Removing "%s"', tensorboard_dir)
            shutil.rmtree(tensorboard_dir)

    logger.info('Loading pretrained model "%s"', model_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                              # load_in_8bit=model_args.load_in_8bit,
                                                              # device_map="auto"
                                                              )
    #model.half()
    #model.to(device)

    if training_args.local_rank in (0, -1):
        print('=' * 80)
        print_gpu_utilization()

        if training_args.deepspeed:
            print('=' * 30 + 'ZeRo 2' + '=' * 30)
            deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=1,
                                                                                                num_nodes=1,
                                                                                                additional_buffer_factor=1.5)
        print('=' * 80)

    # ------------------------- ТОКЕНИЗАТОР ----------------------------------
    logger.info('Loading tokenizer "%s"', model_args.model_name_or_path)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '</s>'})

    for t in ['#', '<s>', '</s>', '<pad>']:
        logger.debug('Tokenizer: token=%s ==> %s', t, str(tokenizer.encode(t, add_special_tokens=False)))

    tokenizer.save_pretrained(training_args.output_dir)

    logger.info('Loading dataset "%s"', data_args.dataset_path)
    train_samples = load_samples(data_args, tokenizer)
    logger.info('Training set: %d samples', len(train_samples))

    train_dataset = FinetuneDataset(train_samples, tokenizer)

    printer = MyPrinterCallback(os.path.join(proj_dir, 'tmp', 'finetune_llama.loss.log'))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        callbacks=[printer]
    )

    logger.info('Start training...')
    train_result = trainer.train()

    #trainer.save_model(output_dir=training_args.output_dir)
    if training_args.local_rank in (0, -1):
        logger.info(f'Saving the model and tokenizer')
        model.save_pretrained(save_directory=training_args.output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    logger.info('All done :)')
