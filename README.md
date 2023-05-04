# LM-finetune

Код для файнтюна LM (rugpt, LLaMa, FRED T5) средствами transformers + deepspeed + LoRa

## Файнтюн читчата на базе модели FRED T5 XL 1.7B

Особенность подхода: вместо префикса <LM> для входной последовательности ставится селектор денойзера <SC1>,
и добавляется токен <extra_id_0> в том месте (конец диалога), где находится генерируемая реплика.

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

## Файнтюн читчата на базе модели sberbank-ai/rugpt3medium_based_on_gpt2

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










