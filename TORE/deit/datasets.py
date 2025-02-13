# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets import StanfordCars, Flowers102, VisionDataset
from torch.utils.data import Dataset, DataLoader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image


class FewExamplesDataset(VisionDataset):
    def __init__(self, root, transform=None, train=True):
        super(FewExamplesDataset, self).__init__(root, transform=transform)
        self.image_paths = os.path.join(root, f'{"train" if train else "test"}')
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.file_list = [filename for filename in os.listdir(self.image_paths) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_paths, self.file_list[idx])
        orig_image = Image.open(img_path)

        image = self.transform(orig_image)

        orig_image = self.to_tensor(orig_image)
        orig_image = transforms.Resize((image.shape[1], image.shape[2]))(orig_image)

        return orig_image, image

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


from typing import List, Dict, Tuple
class PatchedImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")  
        class_to_idx = {cls_name: int(cls_name) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def build_dataset(is_train, args):
    if args.data_set == 'FEW':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),  # Adjust the image size as needed
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        dataset = FewExamplesDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = -1
    else:
        transform = build_transform(is_train, args)
    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'IMNET2':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = PatchedImageFolder(root, transform=transform)
        nb_classes = len(dataset.classes)
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = ImageFolder(root, transform=transform)
        nb_classes = len(dataset.classes)
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CARS':
        dataset = StanfordCars(args.data_path, split="train" if is_train else "test", transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'FLOWERS':
        dataset = Flowers102(args.data_path, split="train" if is_train else "test", transform=transform)
        nb_classes = 102

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
