#python finetune_rugpt_with_prompt_masking.py \

#python -m torch.distributed.run --nproc_per_node=2 finetune_rugpt_with_prompt_masking.py \


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
