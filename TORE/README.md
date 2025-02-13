# TORE
TOken REcycling for efficient sequential inference

## Structure

We bundle 3 repositories, namely [DeiT](https://github.com/facebookresearch/deit), [AME](https://github.com/apardyl/AME), and [DINO](https://github.com/facebookresearch/dino) in the corresponding folders.

The code for training and evaluation of Active Visual Exploration (AVE) models is contained in AME directory. Training and evaluation code for Sequential Classification and Transfer Learning is in the deit directory. Training of DINO is implemented in the dino directory, while evaluation of DINO happens in deit directory.


## Training

To train models used in AVE run `train_AME.sh`.

To finetune DeiT with TORE on imagenet run `train_tore_deit_finetune100ep.sh`.

To finetune DeiT with TORE on (CIFAR10/CIFAR100/FLOWERS102) run `train_tore_deit_transfer.sh`.

To train TORE + DeiT on imagenet end to end run `train_tore_deit_e2e.sh`.

To finetune DINO with TORE on imagenet run `train_tore_dino_finetune15ep.sh`


## Evaluation

To evaluate models used in AVE run `test_AME.sh`.

To evaluate DeiT on Sequential Classification and other tasks run `test_deit.sh`

To evaluate DINO run `deit/eval_knn.py` and `deit/eval_video_segmentation.py`