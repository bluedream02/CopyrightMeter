CUDA_VISIBLE_DEVICES=0 
python scripts/txt2img.py \
    --prompt "$ARTIST" \
    --ckpt /path/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt \
    --config configs/stable-diffusion/v2-inference-v.yaml \
    --H 768 \
    --W 768 \
    --device cuda \
    --key_hex "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7" \
    --nonce_hex "" \
    --message "lthero" \
    --steps 50 \
    --outdir $OUTPUT_DIR \
    --n_iter 30 \
    --n_samples 1