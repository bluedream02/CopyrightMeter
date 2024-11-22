export MODEL_PATH="/data/xunaen/Text-to-image-Copyright/stable-diffusion-1-5"
export INSTANCE_DIR="add_one/"
export CLASS_DIR="data/class-painting"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/add_one/"
python train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a painting in the style of sks" \
  --class_prompt="a photo of painting" \
  --inference_prompt="a painting in the style of sks" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
