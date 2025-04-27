#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /workspace/SmolVLM-Finetune/llava_dataset.json \
    --image_folder /workspace/SmolVLM-Finetune/dataset_images \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_connector True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/testing \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --connector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 1
