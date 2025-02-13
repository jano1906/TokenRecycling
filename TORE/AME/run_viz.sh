#!/bin/bash

arch=mae_vit_base_patch16_dec128d4h1b
mp=/home/jano1906/git/TORE/AME/ablation_decoder_size/sun360/checkpoints/2024-02-25_20:53:42-john-tree/epoch=57-step=25984.ckpt
dd=sun360

if [[ $dd == "sun360" ]]; then
ds=Sun360Classification
ng=8
imh=128
imw=256
elif [[ $dd == "cifar100" ]]; then
ds=Cifar100Classification
ng=12
imh=224
imw=224
fi
ks=(0)
for k in ${ks[@]}; do

python3 predict.py $ds AttentionClsMae \
    --data-dir ~/datasets/$dd \
    --vit-arch $arch \
    --pretrained-mae-path="" \
    --glimpse-size 2 \
    --no-sample-divisions \
    --load-model-path=$mp \
    --K $k \
    --image-size $imh $imw \
    --num-glimpses $ng \
    --output_dir trash \
    --visualization-path viz/expl_viz \
    --eval-batch-size 16 \
    #--sequential-accuracy \
    #--avg-glimpse-path viz/avg_glimipse__${dd}_arch${arch}_K${k}.png \

done
    #--vit-arch mae_vit_base_patch16_dec128d4h1b \
    #--load-model-path=/home/jano1906/git/TORE/AME/ablation_decoder_size/sun360/checkpoints/2024-02-25_20:53:42-john-tree/epoch=57-step=25984.ckpt \