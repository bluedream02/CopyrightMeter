#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python concept_inversion.py \
        --pretrained_model_name_or_path="/path/stable-diffusion-1-5" \
        --train_data_dir=$DATA_DIR \
        --learnable_property="person" \
        --placeholder_token="<person-style>" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=1000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=1000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=25 \
        --esd_checkpoint=$ESD_CKPT \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention
       