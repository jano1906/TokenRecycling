#!/bin/bash

cd deit

python run_with_submitit.py --memfs_imnet \
        --checkpoint_every 10 --sample_divisions \
        --finetune ./checkpoints/deit_3_small_224_1k.pth \
        --output_dir deit_finetune_wsampling_imagenet1k \
        --model deit_small_patch16_LS \
        --data-path ./datasets/imagenet --batch 256 \
        --lr 1e-6 --min-lr 1e-6 --epochs 100 --weight-decay 0.05 \
        --sched cosine --input-size 224 --eval-crop-ratio 1.0 \
        --reprob 0.0 --nodes 1 --ngpus 8 --smoothing 0.0 \
        --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw \
        --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 \
        --unscale-lr --repeated-aug --bce-loss \
        --color-jitter 0.3 --ThreeAugment