#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python tree_ring.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 100 