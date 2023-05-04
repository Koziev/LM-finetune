# LM-finetune

В этом репе я собрал свои текущие рабочие скрипты для файнтюна языковых моделей (rugpt, LLaMa, FRED T5) средствами transformers.
В случае больших моделей (7B и 13B) используются варианты а) deepspeed б) LoRa.

В коде нет ничего нового и особо умного, просто базовый пайплайн в рамках рекомендаций для transformers.Trainer.

## ГЕНЕРАТОР СТИХОВ

### Генератор стихов на базе модели LLaMa 7B и 13B

Код: [finetune_llama.py](./poetry/finetune_llama.py)

Используется deepspeed, что позволяет тюнить модели на 40Гб гпушках. Судя по отчету deepspeed'а, возможен
также файнтюн на V100 с 32Гб. Обратите внимание, что требуется очень много обычной RAM, более 240 Гб,
чтобы deepspeed выгружал туда тензоры.

Запуск файнтюна на 4 GPU:

```
python -m torch.distributed.launch --nproc_per_node=4 finetune_llama.py \
--dataset_path ~/polygon/text_generator/tmp/лирика.jsonl \
--max_samples 10000 \
--output_syllables 0 \
--model_name_or_path decapoda-research/llama-7b-hf \
--output_dir ~/polygon/text_generator/tmp/verses_model=llama7b_domain=lyrics_syllables=0 \
--overwrite_output_dir 1 \
--per_device_train_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--bf16 1 \
--fp16 0 \
--gradient_checkpointing 0 \
--gradient_accumulation_step 8 \
--do_train 1 \
--do_eval 0 \
--report_to tensorboard \
--evaluation_strategy no \
--logging_strategy steps \
--logging_steps 10 \
--save_strategy no \
--deepspeed 13b_deepspeed_config.json
```

Файл с конфигурацией deepspeed'а: [13b_deepspeed_config.json](./poetry/13b_deepspeed_config.json)

Код инференса: [run_llama.py](./poetry/run_llama.py). Disclaimer: этот код инференсит только на 80Гб A100.

### Генератор стихов на базе LLaMa 7B и 13B с использованием библиотеки PEFT (метод LoRa)

Код файнтюна: [finetune_llama_lora.py](./poetry/finetune_llama_lora.py)

Запуск файнтюна на 2 ГПУ:

```
python -m torch.distributed.run --nproc_per_node=2 finetune_llama_lora.py \
--dataset_path ~/polygon/text_generator/tmp/лирика.jsonl \
--max_samples 10000 \
--output_syllables 0 \
--model_name_or_path decapoda-research/llama-7b-hf \
--output_dir ~/polygon/text_generator/tmp/verses_model=llama7b_lora_domain=lyrics_syllables=0 \
--overwrite_output_dir 1 \
--per_device_train_batch_size 1 \
--learning_rate 1e-4 \
--num_train_epochs 1 \
--bf16 0 \
--fp16 0 \
--gradient_checkpointing 0 \
--gradient_accumulation_step 8 \
--do_train 1 \
--do_eval 0 \
--report_to tensorboard \
--evaluation_strategy no \
--logging_strategy steps \
--logging_steps 200 \
--save_strategy no \
```

Код инференса: [run_llama_lora.py](./poetry/run_llama_lora.py)

### Генератор стихов на базе FRED T5 XL

Код для файнтюна: [finetune_fredt5_poetry_generator.py](./poetry/finetune_fredt5_poetry_generator.py)

Запуск файнтюна на 2 ГПУ:

```
python -m torch.distributed.run --nproc_per_node=2 finetune_fredt5_poetry_generator.py \
 --model_name_or_path ai-forever/FRED-T5-1.7B \
 --dataset_path ~/polygon/text_generator/tmp/all_verses.jsonl \
 --prompt prompt_text \
 --optim "adafactor" \
 --learning_rate 1e-3 \
 --lr_scheduler_type constant \
 --per_device_train_batch_size 8 \
 --gradient_checkpointing 0 \
 --gradient_accumulation_steps 4 \
 --num_train_epochs 1 \
 --report_to tensorboard \
 --logging_strategy steps \
 --logging_steps 100 \
 --output_dir ~/polygon/text_generator/tmp/verses_fredt5 \
 --save_strategy no
```

Запуск инференса: [run_fredt5_poetry_generator.py](./poetry/run_fredt5_poetry_generator.py)


### Генератор стихов на базе моделей rugpt (кроме rugpt13B)

Код для файнтюна: [finetune_rugpt_with_prompt_masking.py](./poetry/finetune_rugpt_with_prompt_masking.py)

Запуск на 2 ГПУ, базовая модель rugpt3large_based_on_gpt2:

```
python -m torch.distributed.run --nproc_per_node=2 finetune_rugpt_with_prompt_masking.py \
--dataset_path ~/polygon/text_generator/tmp/лирика.jsonl \
--output_syllables 1 \
--model_name_or_path sberbank-ai/rugpt3large_based_on_gpt2 \
--output_dir ~/polygon/text_generator/tmp/verses_model=rugpt_large_domain=lyrics_syllables=1 \
--overwrite_output_dir 1 \
--per_device_train_batch_size 8 \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--fp16 1 \
--gradient_checkpointing 0 \
--gradient_accumulation_step 8 \
--do_train 1 \
--do_eval 0 \
--report_to tensorboard \
--evaluation_strategy no \
--logging_strategy steps \
--logging_steps 200 \
--save_strategy no
```

Инференс: [run_rugpt_generator.py](./poetry/run_rugpt_generator.py)


## ЧИТЧАТ

### Файнтюн читчата на базе модели FRED T5 XL 1.7B

Особенность подхода: вместо префикса <LM> для входной последовательности ставится селектор денойзера `<SC1>`,
и добавляется токен `<extra_id_0>` в том месте (конец диалога), где находится генерируемая реплика.

Код: [finetune_chitchat_fredt5_with_trainer.py](./chitchat/finetune_chitchat_fredt5_with_trainer.py).

Пример запуска на 1 ГПУ:

```
python finetune_chitchat_fredt5_with_trainer.py \
 --dataset_path axioma_dialogues.json \
 --optim "adafactor" \
 --learning_rate 1e-4 \
 --lr_scheduler_type constant \
 --per_gpu_train_batch_size 6 \
 --gradient_checkpointing 0 \
 --gradient_accumulation_steps 8 \
 --num_train_epochs 1 \
 --report_to tensorboard \
 --logging_strategy steps \
 --logging_steps 500 \
 --output_dir ~/polygon/chatbot/tmp/fredt5_chitchat \
 --save_strategy no
```

Датасет для этой модели: [axioma_dialogues.json](./chitchat/axioma_dialogues.json) сделан из русскоязычной части [датасета проекта OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1).
Каждая ответная реплика вместе с предшествующим контекстом образует отдельный сэмпл для seq2seq модели. Реплики человека и чатбота отмечаются
метками `<h>` и `<и>` соответственно. Для файнтюна они преобразуются в префиксы `человек:` и `чатбот:`.

После файнтюна запустить генерацию можно с помощью кода [run_chitchat_fredt5.py](./chitchat/run_chitchat_fredt5.py).

### Файнтюн читчата на базе модели sberbank-ai/rugpt3medium_based_on_gpt2

Также подходит для других моделей семейства rugpt.

Код [finetune_chitchat_gpt_with_trainer.py](./chitchat/finetune_chitchat_gpt_with_trainer.py).

Датасет: [axioma_dialogues.solid.json](./chitchat/axioma_dialogues.solid.json)

Запуск файнтюна на 1 GPU:

```
python finetune_chitchat_gpt_with_trainer.py \
 --model_name_or_path sberbank-ai/rugpt3medium_based_on_gpt2 \
 --learning_rate 1e-5 \
 --lr_scheduler_type constant \
 --per_gpu_train_batch_size 2 \
 --gradient_checkpointing 0 \
 --gradient_accumulation_steps 8 \
 --num_train_epochs 1 \
 --report_to tensorboard \
 --logging_strategy steps \
 --logging_steps 100 \
 --output_dir ~/polygon/chatbot/tmp/rugpt_chitchat \
 --save_strategy no
```

Код инференса: [run_chitchat_gpt.py](./chitchat/run_chitchat_gpt.py).










