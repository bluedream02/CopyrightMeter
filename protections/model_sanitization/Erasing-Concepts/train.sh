#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train-esd.py --prompt "vangogh" --train_method "full" \