#!/bin/bash

cd deit

model_ckpt=./checkpoints/deit_3_small_224_1k.pth
ouput_dir=out
model=deit_small_patch16_LS
sample_divisions="--sample_divisions"


# finetune cifar10
data_set=CIFAR10
data_path=./datasets/cifar10
epochs=100

python main.py $sample_divisions \
    --resume $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment

# finetune cifar100
data_set=CIFAR100
data_path=./datasets/cifar100
epochs=100

python main.py $sample_divisions \
    --resume $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment

# finetune flowers
data_set=FLOWERS
data_path=./datasets/flowers
epochs=1000

python main.py $sample_divisions \
    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment
