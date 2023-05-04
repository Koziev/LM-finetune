python finetune_chitchat_gpt_with_trainer.py \
--model_name_or_path sberbank-ai/rugpt3medium_based_on_gpt2 \
--learning_rate 1e-5 \
--lr_scheduler_type constant \
--per_gpu_train_batch_size 16 \
--gradient_checkpointing 0 \
--gradient_accumulation_steps 8 \
--num_train_epochs 1 \
--report_to tensorboard \
--logging_strategy steps \
--logging_steps 100 \
--output_dir ~/polygon/chatbot/tmp/rugpt_chitchat \
--save_strategy no

