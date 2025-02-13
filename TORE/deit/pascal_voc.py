import json
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from PIL import Image
import xml.etree.ElementTree as ET

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.transforms import PILToTensor

import models_v2
import models_dino
from pathlib import Path

from timm.models import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import utils
from mask_const import get_division_masks_for_model


def parse_annotation_and_mask(annotation_path, masks_dir, img_size):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_path = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    masks = []
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'name': obj_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

        mask_file = os.path.join(masks_dir, image_path.replace('.jpg', '.png'))
        mask = Image.open(mask_file)
        mask = pth_transforms.Compose([pth_transforms.transforms.Resize(img_size),
                                       pth_transforms.transforms.ToTensor()])(mask)
        int_image = (mask * 255.0).to(torch.uint8)
        masks.append(int_image)
        #masks.append(PILToTensor()(mask))

    return {
        'image_path': image_path,
        'width': width,
        'height': height,
        'objects': objects,
        'masks': masks
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--model', default='deit_small_patch16_LS', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = model.to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        #checkpoint = torch.load(args.checkpoint, map_location='cpu')['teacher']
        #url = "https://dl.fbaipublicfiles.com/dino/" + "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        #pretrained_dict = torch.hub.load_state_dict_from_url(url=url)
        #pretrained_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
        #checkpoint = torch.load(args.checkpoint, map_location='cpu')
        utils.interpolate_pos_embed(model, checkpoint['model'])
        msg = model.load_state_dict(checkpoint['model'])
        print("Loaded checkpoint: ", msg)
        #utils.interpolate_pos_embed(model, checkpoint['model'])
        #msg = model.load_state_dict(checkpoint['model'])
        #print("Loaded checkpoint: ", msg)
    # open image

    dataset_dir = '/data/pwojcik/VOCdevkit/VOC2012/'
    validation_image_set_file = os.path.join(dataset_dir, 'ImageSets/Segmentation/val.txt')
    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    masks_dir = os.path.join(dataset_dir, 'SegmentationClass')

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    with open(validation_image_set_file, 'r') as f:
        validation_image_list = f.read().strip().split('\n')

    torch.set_printoptions(threshold=10000)
    np.set_printoptions(threshold=np.inf)
    jaccards = []
    for K in range(len(model.blocks)+1):
        print(f"Validation for: {K}")
        jacs = []
        for image_id in validation_image_list:
            annotation_file = os.path.join(annotations_dir, image_id + '.xml')
            annotation_info = parse_annotation_and_mask(annotation_file, masks_dir, args.image_size)

            image_path = os.path.join(dataset_dir, 'JPEGImages', annotation_info['image_path'])
            #print(f'Image Path: {image_path}')
            #print(f'Width: {annotation_info["width"]}, Height: {annotation_info["height"]}')

            # make the image divisible by the patch size
            img = transform(Image.open(image_path))

            w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // args.patch_size
            h_featmap = img.shape[-1] // args.patch_size

            division_masks = get_division_masks_for_model(model)
            with torch.cuda.amp.autocast():
                masks = division_masks[16][0]
                img = img.to(device, non_blocking=True)
                _ = model.comp_forward_afterK(img, K=K, masks=masks)#, keep_token_order=True)
            #attentions = model.get_last_selfattention(img.to(device))
            attentions = model.last_attn#[11]
            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            if args.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                    0].cpu().numpy()

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                0].cpu().numpy()

            objs = annotation_info['objects']
            mask = annotation_info['masks'][0]
            w, h = mask.shape[1] - mask.shape[1] % args.patch_size, mask.shape[2] - mask.shape[2] % args.patch_size
            mask = mask[:, :w, :h]
            #print(mask)
            unique = np.unique(mask).tolist()[1:-1]
            #if len(np.unique(mask).tolist()[1:-1]) > 0:
            #    unique = np.unique(mask).tolist()[1:-1]
            #else:
            #    unique = np.unique(mask).tolist()[:-1]
            if len(unique) == 0:
                continue
            #assert len(objs) == len(unique)
            jac = 0
            for o in unique:
                masko = mask == o
                intersection = masko * th_attn
                intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
                union = (masko + th_attn) > 0
                union = torch.sum(torch.sum(union, dim=-1), dim=-1)
                jaco = intersection / union
                jac += max(jaco)
            jac /= len(unique)
            jacs.append(jac.item())
        J = sum(jacs) / len(jacs)
        print("Jaccard:", J)
        jaccards.append(J)
    print(jaccards)


