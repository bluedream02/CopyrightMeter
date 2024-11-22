python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="$DATA_DIR" \
  --learnable_property="object" \
  --placeholder_token="[V]" \
  --initializer_token="person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="$TEXTUAL_OUTPUT_DIR" \
  --validation_prompt="A photo of [V]"	\
  --num_validation_images=4	\
  --validation_steps=100	\
  --checkpointing_steps=1000