python train-scripts/train_erase.py \
    --concepts "$ARTIST" \
    --guided_concepts "$TYPE" \
    --device '1' \
    --concept_type "$TYPE" \   # art / object
    --dataset "$DATASET"