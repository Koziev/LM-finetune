#python -m torch.distributed.run --nproc_per_node=2 finetune_rugpt13b_lora.py \
#python finetune_rugpt13b_lora.py \

python -m torch.distributed.run --nproc_per_node=2 finetune_rugpt13b_lora.py \
--dataset ~/polygon/text_generator/tmp/лирика.jsonl \
--output_syllables 1 \
--model_name_or_path ai-forever/rugpt13b \
--output_dir ~/polygon/text_generator/tmp/verses_rugpt13b_lora_domain=lyrycs_syllables=1 \
--overwrite_output_dir 1 \
--per_device_train_batch_size 3 \
--learning_rate 1e-4 \
--lr_scheduler_type constant \
--gradient_accumulation_steps 8 \
--num_train_epochs 1 \
--do_train 1 \
--do_eval 0 \
--report_to tensorboard \
--evaluation_strategy no \
--logging_strategy steps \
--logging_steps 100 \
--save_strategy no
