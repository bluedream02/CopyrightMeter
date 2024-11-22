#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --train_data_dir=$DATA_DTR \
  --caption_column="text" --dataloader_num_workers=1 \
  --resolution=512 --random_flip --train_batch_size=4 \
  --num_train_epochs=50 --checkpointing_steps=200 \
  --learning_rate=1e-04 --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="$PROMPT"