#!/bin/bash

cd dino

python main_dino.py --arch vit_small \
    --batch_size_per_gpu 16 \
    --patch_size 8 \
    --data_path ./datasets/imagenet \
    --output_dir logs \
    --norm_last_layer False \
    --finetune ./checkpoints/dino_deitsmall8_pretrain_full_checkpoint.pth \
    --sample_divisions \
    --saveckp_freq 5 \
    --epochs 15 \
    --lr 1e-6 \
    --local_crops_number 10