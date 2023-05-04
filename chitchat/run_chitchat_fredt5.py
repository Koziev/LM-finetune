import os
import argparse

import torch
import transformers
from transformers import T5Config


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/chatbot')

    parser = argparse.ArgumentParser(description='Консольная интерактивная проверка модели читчата')
    parser.add_argument('--model', type=str, default=os.path.join(proj_dir, 'tmp', 'fredt5_chitchat'), help='Путь к каталогу с файлами модели')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_dir = args.model
    print(f'Loading model "{model_dir}"...')
    t5_config = T5Config.from_pretrained(model_dir)

    if 'FRED-T5' in t5_config.name_or_path:
        t5_tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_dir)
    else:
        t5_tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

    t5_model = transformers.T5ForConditionalGeneration.from_pretrained(model_dir)
    t5_model.to(device)
    t5_model.eval()

    while True:
        print('-'*80)
        dialog = []
        while True:
            msg = input('H:> ').strip()
            if len(msg) == 0:
                break

            msg = msg[0].upper() + msg[1:]

            dialog.append('человек: ' + msg)

            #prompt = '<LM>'+'\n'.join(dialog)
            prompt = '<SC1>' + '\n'.join(dialog) + '\nчатбот: <extra_id_0>'

            input_ids = t5_tokenizer(prompt, return_tensors='pt').input_ids
            out_ids = t5_model.generate(input_ids=input_ids.to(device),
                                        max_length=200,
                                        eos_token_id=t5_tokenizer.eos_token_id,
                                        early_stopping=True,
                                        do_sample=True,
                                        temperature=1.0,
                                        top_k=0,
                                        top_p=0.85)

            t5_output = t5_tokenizer.decode(out_ids[0][1:])
            if '</s>' in t5_output:
                t5_output = t5_output[:t5_output.find('</s>')].strip()

            t5_output = t5_output.replace('<extra_id_0>', '').strip()

            print('B:> {}'.format(t5_output))
            dialog.append('чатбот: ' + t5_output)
