export MODEL_NAME="/path/Text-to-image-Copyright/stable-diffusion-1-5"

python concept_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="person" \
        --placeholder_token="<person-style>" --initializer_token="person" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=5000 \
        --learning_rate=5.0e-03 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --save_as_full_pipeline \
        --checkpointing_steps=5000 \
        --output_dir=$OUTPUT_DIR \
        --num_train_images=25 \
        --ac_checkpoint=$ESD_CKPT \
        --mixed_precision="fp16" \
        --enable_xformers_memory_efficient_attention
       