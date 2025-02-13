import math

import torch
import argparse
import json
import os

from sklearn.manifold import TSNE
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import utils
from timm.utils import accuracy
import numpy as np
import models
import models_v2
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch import nn
import models_dino
import cv2

from datasets import FewExamplesDataset, build_dataset
from mask_const import sample_masks, get_division_masks_for_model, DIVISION_IDS, DIVISION_MASKS, DIVISION_SPECS_56_56, \
    division_masks_from_spec
from functools import partial
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as TT
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('Script containing tasks with inference only', add_help=False)
    parser.add_argument('--batch-size', default=4, type=int)
    # Model parameters
    parser.add_argument('--model', default='vit_small_8', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=1., type=float, help="Crop ratio for evaluation")

    parser.add_argument('--data-path', default=f'~/datasets/imagenet/ILSVRC/Data/CLS-LOC/', type=str,
                        help='dataset path')
    parser.add_argument('--data-split', default='val', choices=['train', 'val'],
                        type=str)
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path where to save')

    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('--evaluate', nargs='*', default=[])
    parser.add_argument('--extract', nargs='*', default=[])
    parser.add_argument('--count_flops', action="store_true")

    parser.add_argument('--group_by_class', action="store_true")
    parser.add_argument('--random_masks', action="store_true")

    return parser


@torch.no_grad()
def extract(data_loader, model, device, KMs, random_masks):
    # switch to evaluation mode
    model.eval()
    division_masks = division_masks_from_spec(DIVISION_SPECS_56_56)

    ret = {
        f"{k}_{m}": {"features": [], "targets": []}
        for k, m in KMs
    }

    _, input_tensor = next(iter(data_loader))

    # We need to reorder the images to [batch, channel, width, height]
    # The array of loaded images is [batch, height, width, channel]

    with torch.cuda.amp.autocast():
        for k, m in KMs:
            if random_masks:
                masks = sample_masks(division_masks, m)
            else:
                masks = division_masks[m][0]
            input_tensor = input_tensor.to(device, non_blocking=True)
            features = model.get_intermediate_layers_forward_afterK(input_tensor, K=k, masks=masks, n=1, keep_token_order=True)[0]
            features = features[:, 1:, :]
            ret[f"{k}_{m}"]["features"].append(features)

    return ret


def PCA_path_tokens_rgb(features, output_dir):
    for kM in features.keys():
        bsz, L, feat_dim = features[kM]['features'][0].shape
        patch_tokens = features[kM]['features'][0].reshape([bsz, feat_dim, -1])

        patch_h = math.isqrt(L)

        total_features = patch_tokens.reshape(bsz * patch_h * patch_h, feat_dim) #4(*H*w, 1024)

        total_features = total_features.detach().cpu()

        pca = PCA(n_components=3)
        pca.fit(total_features)
        pca_features = pca.transform(total_features)

        pca_features_bg = pca_features[:, 0] < 0.45  # from first histogram
        pca_features_fg = ~pca_features_bg

        pca.fit(total_features[pca_features_fg])
        pca_features_left = pca.transform(total_features[pca_features_fg])

        for i in range(3):
            # min_max scaling
            pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (
                        pca_features_left[:, i].max() - pca_features_left[:, i].min())

        pca_features_rgb = pca_features.copy()
        # for black background
        pca_features_rgb[pca_features_bg] = 0
        # new scaled foreground features
        pca_features_rgb[pca_features_fg] = pca_features_left

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(bsz, patch_h, patch_h, 3)

        fig = plt.figure(figsize=(10, 10))

        for i in range(pca_features_rgb.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features_rgb[i])
            plt.axis('off')  # Turn off axis labels
            fig.savefig(f"{output_dir}/output_3_rgb_{kM}.png")
        print('Saved PCA matching')


def PCA_path_tokens_foreground_seg(features, patch_size=16):
    feat_dim = 384
    patch_h = 448 // patch_size
    patch_w = 448 // patch_size

    for kM in features.keys():
        patch_tokens = features[kM]['features'][0].reshape([4, feat_dim, -1])

        total_features = patch_tokens.reshape(4 * patch_h * patch_w, feat_dim) #4(*H*w, 1024)

        pca = PCA(n_components=3)
        pca.fit(total_features)
        pca_features = pca.transform(total_features)

        pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                             (pca_features[:, 0].max() - pca_features[:, 0].min())

        fig = plt.figure(figsize=(10, 10))

        for i in range(pca_features.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features[i * patch_h * patch_w: (i + 1) * patch_h * patch_w, 0].reshape(patch_h, patch_w))
            fig.savefig(f"output_3_{kM}.png")

in1k_classes = {
    'aircraft': 403,
    'airliner': 404,
    'arch': 873
}

def tsne(features, k, classes_to_render):
    import cne
    y = features['targets']
    features = features['features']
    x = nn.functional.normalize(torch.tensor(features), dim=1, p=2).numpy()

    import random
    random_numbers = [random.randint(0, 999) for _ in range(50)]

    #mask = torch.zeros(targets.shape).to(bool)
    #for i in range(0, 10):
    #    mask_i = torch.tensor((targets == i))
    #    mask = torch.logical_or(mask, mask_i)
    #y = targets[mask]
    #x = features[mask]


    print('Generating t-sne... for ', k)

    #embedder_ncvis = cne.CNE(loss_mode="nce",
    #                         k=15,
    #                         optimizer="adam",
    #                         parametric=True,
    #                         print_freq_epoch=10)
    #tsne = embedder_ncvis.fit_transform(x)
    tsne = TSNE(n_components=2, learning_rate='auto', init = 'random').fit_transform(x)

    #plt.style.use('dark_background')

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(*tsne.T, c=y, alpha=1.0, s=12, cmap="tab10", edgecolor="none")
    #title_text = plt.title(f"$\kappa$={k}", fontsize=30)
    fig.suptitle(f"$\kappa$={k}", fontsize=30)  # Adjust y as needed

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    #title_text.set_position('top')
    plt.gca().set_aspect("equal")
    plt.axis("off")
    fig.savefig(f"dino_tsne_{k}.png")



def extract_patches_k16(data_loader, model, device, random_masks, *args, **kwargs):
    KMs = [[k, 16] for k in range(9)]
    return extract(data_loader, model, device, KMs=KMs, random_masks=random_masks)


def main_setup(args):
    if args.checkpoint is None:
        print("[WARNING] --checkpoint is None")

    output_dir = Path(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=args.output_dir == "debug")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=(224, 224)
    )
    model = model.to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['teacher']
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Loaded checkpoint: ", msg)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    args.data_set = 'FEW'
    dataset, _ = build_dataset(is_train=True, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        drop_last=False
    )

    return model, data_loader



def main(args):
    model, data_loader = main_setup(args)

    ret_dict = extract_patches_k16(data_loader, model, args.device, random_masks=False)
    #for k in [0, 4]:
    #    features = np.load(f"/data/pwojcik/deit/dino_cifar_val_224_2/extract_k16/{k}_16.npz")
    #    tsne(features, k, in1k_classes)

    #PCA_path_tokens_seg(ret_dict)
    PCA_path_tokens_rgb(ret_dict, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

