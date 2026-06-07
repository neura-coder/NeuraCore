#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 src/train.py \
    --data_path data/train.json \
    --save_dir checkpoints_add \
    --batch_size 4 \
    --epochs 10 \
    --num_layers 6 \
    --gradient_accumulation 2