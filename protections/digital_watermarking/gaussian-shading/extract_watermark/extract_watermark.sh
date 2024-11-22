


python extract.py \
    --model_id "/path/model/stable-diffusion-1-5" \
    --single_image_path "" \
    --image_directory_path $IMAGE_DIR \
    --key_hex "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7" \
    --nonce_hex "14192f43863f6bad1bf54b7832697389" \
    --original_message_hex "6c746865726f0000000000000000000000000000000000000000000000000000" \
    --num_inference_steps 50 \
    --scheduler "DDIM" \
    --is_traverse_subdirectories 0