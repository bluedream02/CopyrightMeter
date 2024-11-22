python eval-scripts/generate-images.py \
    --model_name "./models/$DATASET/$ARTIST.pt" \
    --prompts_path "./data/artist.csv" \
    --save_path "output_images/$DATASET/$ARTIST/" \
    --num_samples 1 \
    --ddim_steps 50 \
    --ddst "$DATASET" \
    --atst "$ARTIST"