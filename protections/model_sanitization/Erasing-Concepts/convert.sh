export CUDA_VISIBLE_DEVICES=1
python convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path “training-output-path” \
    --original_config_file "model config .yaml file" \
    --dump_path "convert-output-path"