#!/bin/bash

cd AME

ckpt="./checkpoints/mae_pretrain_vit_base.pth"

root_dir=train_big_multi_random
dss=(0 1)
archs=(mae_vit_base_patch16_dec512d8b)
gss=(2)
modes=(5)

for dsi in ${dss[@]}; do
    imh=224
    imw=224
    num_glimpses=12
    if [[ $dsi -eq "0" ]]; then
        task_name="Sun360Classification"
        dir="sun360"
        imh=128
        imw=256
        num_glimpses=8
    elif [[ $dsi -eq "1" ]]; then
        task_name="Cifar100Classification"
        dir="cifar100"
    elif [[ $dsi -eq "2" ]]; then
        task_name="Cifar10Classification"
        dir="cifar10"
    else
        echo "wrong dataset value, expected 0,1,2,3 but got" $dsi
        exit 1
    fi
for arch in ${archs[@]}; do
for gs in ${gss[@]}; do
for mdi in ${modes[@]}; do
    if [[ $mdi -eq "0" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        rl="--no-rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "1" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "2" ]]; then
        sp="RandomClsMae"
        sd="--sample-divisions"
        rl="--no-rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "3" ]]; then
        sp="RandomClsMae"
        sd="--sample-divisions"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "4" ]]; then
        sp="AttentionClsMae"
        sd="--sample-divisions"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "5" ]]; then
        sp="RandomClsMae"
        sd="--sample-divisions"
        rl="--rec-loss"
        ss="--no-single-step"
    elif [[ $mdi -eq "6" ]]; then
        sp="AttentionClsMae"
        sd="--sample-divisions"
        rl="--rec-loss"
        ss="--no-single-step"
    
    else
        echo "wrong mode value, expected 0,1,2, but got" $mdi
        exit 1
    fi
            
    python train.py \
        $task_name \
        $sp \
        --data-dir=$HOME/datasets/$dir \
        --pretrained-mae-path=$ckpt \
        --tensorboard \
        --vit-arch $arch \
        --glimpse-size $gs \
        $sd \
        $rl \
        $ss \
        --lr 1e-5 \
        --output-dir $root_dir/$dir \
        --image-size $imh $imw \
        --num-glimpses $num_glimpses

done
done
done
done
done
