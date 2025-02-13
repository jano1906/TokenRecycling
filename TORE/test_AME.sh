#!/bin/bash

cd AME

root_dirs=(/home/jano1906/git/TORE/AME/ame_sun224)
archs=(1 0)
dss=(7)
sequential_accuracy="--sequential-accuracy"
center_first="" #"--center-first"
for root_dir in ${root_dirs[@]}; do
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
        echo "wrong dataset value, expected 0,1,2,3 but got" $dsi
        exit 1
    fi
for run in $root_dir/$dir/logs/*; do
echo $run
for gs in {2..2}; do
for K in {0..0}; do
for archi in ${archs[@]}; do
    if [[ $archi -eq "0" ]]; then
        arch=RandomClsMae
        force_K=""
    elif [[ $archi -eq "1" ]]; then
        arch=AttentionClsMae
        force_K=""
    elif [[ $archi -eq "2" ]]; then
        arch=AttentionClsMae
        force_K="--force-K-prediction 0"
    elif [[ $archi -eq "3" ]]; then
        arch=AttentionMae
        force_K=""
    else
        exit 1
    fi

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
    $force_K \
    $sequential_accuracy
    $center_first

done
done
done
done
done
done
done