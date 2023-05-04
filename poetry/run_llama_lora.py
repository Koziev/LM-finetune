"""
Инференс стихов из отфайнтюненной модели LLaMa+LoRa (см. finetune_llama_lora.py)
"""

import transformers
import torch
from peft import PeftModel, PeftConfig


model_path = "/home/jovyan/polygon/text_generator/tmp/verses_model=llama7b_lora_domain=lyrics_syllables=0"

print('Loading LLaMa tokenizer "{}"...'.format(model_path))
tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)

peft_model_id = model_path
print('Loading peft model "{}"...'.format(peft_model_id))
config = PeftConfig.from_pretrained(peft_model_id)

print('Loading backbone LLaMa "{}"...'.format(config.base_model_name_or_path))
model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

device = torch.device("cuda")
model.to(device)

while True:
    seed = input(':> ')
    prompt = '<s>' + seed + '#'

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    #print('DEBUG@26 encoded_prompt=', encoded_prompt)
    encoded_prompt = encoded_prompt.to(device)

    pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]
    # end_token_id = self.tokenizer.encode('</s>', add_special_tokens=False)[0]

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=1.0,
        top_p=0.80,
        max_length=300,
        num_return_sequences=5,
    )

    stop_token = '</s>'

    generated_sequences = set()
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        #print('DEBUG@46 ==> ', generated_sequence)

        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        if stop_token in text:
            text = text[: text.find(stop_token)]

        text = text[text.index('#') + 1:].strip()
        text = text.replace('\u2010', '').replace('\u0301', '')
        print('='*80)
        print(text)
