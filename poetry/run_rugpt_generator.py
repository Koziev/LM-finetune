import os
import json
import sys
import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from generative_poetry.whitespace_normalization import normalize_whitespaces


class RugptGenerator:
    def __init__(self, model_path, generation_config):
        self.model_path = model_path
        self.generation_config = generation_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        model_name_or_path = os.path.expanduser(self.model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def generate_output(self, context, num_return_sequences):
        encoded_prompt = self.tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        pad_token_id = self.tokenizer.encode('<pad>', add_special_tokens=False)[0]
        #end_token_id = self.tokenizer.encode('</s>', add_special_tokens=False)[0]

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            pad_token_id=pad_token_id,
            **self.generation_config
        )

        stop_token = '</s>'

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            if stop_token in text:
                text = text[: text.find(stop_token)]

            text = text[text.index('#')+1:].strip()
            text = text.replace('\u2010', '').replace('\u0301', '')
            text = normalize_whitespaces(text)
            generated_sequences.add(text)

        return list(generated_sequences)


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/text_generator')

    parser = argparse.ArgumentParser(description='Отладочный консольный генератор пирожков')
    parser.add_argument('--model_path', type=str, default=os.path.join(proj_dir, 'tmp', 'verses_rugpt.all'))
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--penalty_alpha', type=float, default=None)
    parser.add_argument('--epsilon_cutoff', type=float, default=0.0)
    parser.add_argument('--eta_cutoff', type=float, default=0.0)
    parser.add_argument('--diversity_penalty', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=None)
    parser.add_argument('--encoder_repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--renormalize_logits', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.9, help='Температура сэмплинга')
    parser.add_argument('--top_p', type=float, default=0.6, help='top-p')
    parser.add_argument('--top_k', type=int, default=0, help='top-k')
    parser.add_argument('--typical_p', type=float, default=None, help='typical-p')
    args = parser.parse_args()

    generation_args = {'max_length': args.max_length,
                       'num_return_sequences': args.num_return_sequences,
                       'do_sample': args.do_sample,
                       'num_beams': args.num_beams,
                       'num_beam_groups': args.num_beam_groups,
                       'penalty_alpha': args.penalty_alpha,
                       'epsilon_cutoff': args.epsilon_cutoff,
                       'eta_cutoff': args.eta_cutoff,
                       'diversity_penalty': args.diversity_penalty,
                       'repetition_penalty': args.repetition_penalty,
                       'encoder_repetition_penalty': args.encoder_repetition_penalty,
                       'length_penalty': args.length_penalty,
                       'no_repeat_ngram_size': args.no_repeat_ngram_size,
                       'renormalize_logits': args.renormalize_logits,
                       'temperature': args.temperature,
                       'top_p': args.top_p,
                       'top_k': args.top_k,
                       'typical_p': args.typical_p,
                       }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    poem_generator = RugptGenerator(args.model_path, generation_args)
    poem_generator.load()

    while True:
        prompt = input(':> ').strip()
        if prompt:
            seed = prompt + '#'
            px = poem_generator.generate_output(seed, num_return_sequences=10)
            print('-'*80)
            for ipoem, p in enumerate(px, start=1):
                print('='*30 + ' POEM #{} '.format(ipoem) + '='*30)
                print(p)
            print('-'*80)
