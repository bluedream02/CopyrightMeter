export EXPERIMENT_NAME="new"
export MODEL_PATH="/data/xunaen/Text-to-image-Copyright/stable-diffusion-1-5"
export CLEAN_TRAIN_DIR="data/A" 
export CLEAN_ADV_DIR="data/B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/ADVERSARIAL"
export CLASS_DIR="data/class-painting"


# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

python attacks/aspl.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
  --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
  --instance_prompt="a painting in the style of sks" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --class_prompt="a photo of painting" \
  --output_dir=$OUTPUT_DIR \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 


# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/DREAMBOOTH"

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
  --checkpointing_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8