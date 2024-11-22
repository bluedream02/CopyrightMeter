export MODEL_PATH="./DIA_coated_blur_lora" \
export SAVE_PATH="./DIA_coated_blur_gen/" \

CUDA_VISIBLE_DEVICES=2 python generate.py --model_path $MODEL_PATH --save_path  $SAVE_PATH