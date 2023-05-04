#--max_samples 50000 \

python -m torch.distributed.launch --nproc_per_node=4 finetune_rugpt13b.py \
--dataset ~/polygon/text_generator/tmp/лирика.jsonl \
--output_syllables 0 \
--model_name_or_path ai-forever/rugpt13b \
--output_dir ~/polygon/text_generator/tmp/verses_rugpt13b_domain=lyrycs_syllables=0 \
--overwrite_output_dir 1 \
--per_device_train_batch_size 4 \
--learning_rate 5e-6 \
--lr_scheduler_type constant \
--gradient_accumulation_steps 16 \
--num_train_epochs 1 \
--do_train 1 \
--do_eval 0 \
--fp16 1 \
--bf16 0 \
--report_to tensorboard \
--evaluation_strategy no \
--logging_strategy steps \
--logging_steps 5 \
--save_strategy no \
--deepspeed 13b_deepspeed_config.json
