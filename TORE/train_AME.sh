#!/bin/bash

cd AME

ckpt="./checkpoints/mae_pretrain_vit_base.pth"

root_dir=tore_sun224   # name of an experiment
dss=(7)                                             # choose a dataset
archs=(mae_vit_base_patch16_dec128d4h1b)            # choose architecture
gss=(2)                                             # choose glimpse size
modes=(3)                                           # choose training mode


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
    elif [[ $dsi -eq "3" ]]; then
        task_name="Sun360Reconstruction"
        dir="sun360"
        imh=128
        imw=256
        num_glimpses=8
    elif [[ $dsi -eq "4" ]]; then
        task_name="CarsClassification"
        dir="cars"
    elif [[ $dsi -eq "5" ]]; then
        task_name="FlowersClassification"
        dir="flowers"
    elif [[ $dsi -eq "6" ]]; then
        task_name="Food101Classification"
        dir="food"
    elif [[ $dsi -eq "7" ]]; then
        task_name="Sun360Classification"
        dir="sun360"
    else
        exit 1
    fi
for arch in ${archs[@]}; do
for gs in ${gss[@]}; do
for mdi in ${modes[@]}; do
    k="--K 0"
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
    elif [[ $mdi -eq "7" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        rl="--rec-loss"
        ss="--no-single-step"
    elif [[ $mdi -eq "8" ]]; then
        sp="AttentionClsMae"
        sd="--no-sample-divisions"
        rl="--rec-loss"
        ss="--no-single-step"
    elif [[ $mdi -eq "9" ]]; then
        sp="RandomMae"
        sd="--sample-divisions"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "10" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 0"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "11" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 1"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "12" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 2"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "13" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 3"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "14" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 4"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "15" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 5"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "16" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 6"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "17" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 7"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "18" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 8"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "19" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 9"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "20" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 10"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "21" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 11"
        rl="--rec-loss"
        ss="--single-step"
    elif [[ $mdi -eq "22" ]]; then
        sp="RandomClsMae"
        sd="--no-sample-divisions"
        k="--K 12"
        rl="--rec-loss"
        ss="--single-step"
    
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
        $k \
        --lr 1e-5 \
        --output-dir $root_dir/$dir \
        --image-size $imh $imw \
        --num-glimpses $num_glimpses \

done
done
done
done
done
