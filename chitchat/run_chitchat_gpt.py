import os.path

import torch
import transformers


class Chitchat(object):
    def __init__(self, device, models_dir):
        model_name = os.path.join(models_dir, 'rugpt_chitchat')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def reply(self, history, num_return_sequences):
        prompt = '<s>' + '\n'.join(history) + '\nчатбот:'
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        output_sequences = self.model.generate(input_ids=encoded_prompt,
                                               max_length=len(prompt) + 120,
                                               temperature=0.90,
                                               typical_p=None,
                                               top_k=0,
                                               top_p=0.8,
                                               do_sample=True,
                                               num_return_sequences=num_return_sequences,
                                               pad_token_id=self.tokenizer.pad_token_id)

        replies = []

        for o in output_sequences:
            reply = self.tokenizer.decode(o.tolist(), clean_up_tokenization_spaces=True)
            reply = reply[len(prompt):]  # отсекаем затравку
            reply = reply[: reply.find('</s>')]

            if '\nчеловек:' in reply:
                reply = reply[:reply.index('\nчеловек:')]

            reply = reply.strip()

            if reply not in replies:  # только уникальные реплики, сохраняем порядок выдачи
                replies.append(reply)

        return replies


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = os.path.expanduser('~/polygon/chatbot/tmp')

    chitchat = Chitchat(device, models_dir)

    while True:
        dialog = []
        while True:
            msg = input('H:> ').strip()
            if msg:
                dialog.append('человек: ' + msg)
                reply = chitchat.reply(dialog, num_return_sequences=1)[0]
                print(f'B:> {reply}')
                dialog.append('чатбот: ' + reply)
            else:
                dialog = []
                print('-'*100)


