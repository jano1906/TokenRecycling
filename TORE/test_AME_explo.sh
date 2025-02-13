#!/bin/bash

cd AME

root_dir=/home/jano1906/git/TORE/AME/aa_results_ours
dss=(0 1)
sequential_accuracy="--sequential-accuracy"
center_first="" #"--center-first"
arch=AttentionClsMae

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
    else
        echo "wrong dataset value, expected 0,1,2,3 but got" $dsi
        exit 1
    fi
for run in $root_dir/$dir/logs/*; do
for gs in {2..2}; do
for K in {0..0}; do
for force_K in {0..12}; do

for ckpt in $root_dir/$dir/checkpoints/$(basename $run)/*; do

python predict.py \
    $task_name \
    $arch \
    --K $K \
    --data-dir=$HOME/datasets/$dir \
    --load-model-path="" \
    --eval-ckpt $ckpt \
    --ddp \
    --test \
    --glimpse-size $gs \
    --eval-run $run \
    --image-size $imh $imw \
    --num-glimpses $num_glimpses \
    --output_dir $root_dir/$dir/results \
    --force-K-prediction $force_K \
    $sequential_accuracy
    $center_first

done
done
done
done
done
done