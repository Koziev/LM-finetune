import os
import json
import sys
import argparse

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from peft import PeftModel, PeftConfig


class RugptGenerator:
    def __init__(self, model_path, temperature, top_p):
        self.model_path = os.path.expanduser(model_path)
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
        #self.model = PeftModel.from_pretrained(self.model_path)

        peft_model_id = self.model_path
        config = PeftConfig.from_pretrained(peft_model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        self.model = model.to(self.device)
        #self.model.eval()

    def generate_output(self, context, num_return_sequences):
        length = 200

        encoded_prompt = self.tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        pad_token_id = self.tokenizer.encode('<pad>', add_special_tokens=False)[0]
        #end_token_id = self.tokenizer.encode('</s>', add_special_tokens=False)[0]

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            #end_token_id=end_token_id,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p
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

            generated_sequences.add(text)

        return list(generated_sequences)


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/text_generator')

    parser = argparse.ArgumentParser(description='Отладочный консольный генератор стихов на базе rugpt13B+LoRa')
    parser.add_argument('--model_path', type=str, default=os.path.join(proj_dir, 'tmp', 'verses_rugpt13b_lora_domain=lyrycs_syllables=1'))
    parser.add_argument('--temperature', type=float, default=1.0, help='Температура сэмплинга')
    parser.add_argument('--top_p', type=float, default=0.8, help='top-p')
    parser.add_argument('--top_k', type=int, default=0, help='top-k')
    parser.add_argument('--typical_p', type=float, default=0.0, help='typical-p')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    poem_generator = RugptGenerator(model_path=args.model_path, temperature=args.temperature, top_p=args.top_p)
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
