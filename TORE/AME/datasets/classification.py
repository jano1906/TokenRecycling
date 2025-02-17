import abc
import os
import sys
from collections import Counter
from typing import Optional

import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import ImageNet, CIFAR100, CIFAR10, ImageFolder, Food101

from datasets.base import BaseDataModule
from datasets.utils import get_default_img_transform, get_default_aug_img_transform


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.data = file_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]

    def class_stats(self):
        return [v for k, v in sorted(Counter(self.labels).items())]


class BaseClassificationDataModule(BaseDataModule, abc.ABC):
    cls_num_classes = 0

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.inst_num_classes = None

    @property
    def num_classes(self):
        if self.inst_num_classes is not None:
            return self.inst_num_classes
        return self.cls_num_classes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        #print(f'Train class statistics:', self.train_dataset.class_stats(), file=sys.stderr)
        return super().train_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        #print(f'Test class statistics:', self.test_dataset.class_stats(), file=sys.stderr)
        return super().test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        #print(f'Val class statistics:', self.val_dataset.class_stats(), file=sys.stderr)
        return super().val_dataloader()

class Sun360Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 26

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'meta/sun360-dataset26-random.txt')) as f:
            file_list = f.readlines()
        labels = [str.join('/', p.split('/')[:2]) for p in file_list]
        classes = {name: idx for idx, name in enumerate(sorted(set(labels)))}
        labels = [classes[x] for x in labels]
        file_list = [os.path.join(self.data_dir, p.strip()) for p in file_list]
        val_list = file_list[:len(file_list) // 10]
        val_labels = labels[:len(file_list) // 10]
        train_list = file_list[len(file_list) // 10:]
        train_labels = labels[len(file_list) // 10:]

        if stage == 'fit':
            self.train_dataset = ClassificationDataset(file_list=train_list, label_list=train_labels,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ClassificationDataset(file_list=val_list, label_list=val_labels,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class Cifar100Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 100

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = CIFAR100(root=self.data_dir, train=True,
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.no_aug else
                                                   get_default_aug_img_transform(self.image_size, scale=False))
            
            self.val_dataset = CIFAR100(root=self.data_dir, train=False,
                                                 transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()

class Cifar10Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 10

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = CIFAR10(root=self.data_dir, train=True,
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.no_aug else
                                                   get_default_aug_img_transform(self.image_size, scale=False))
            
            self.val_dataset = CIFAR10(root=self.data_dir, train=False,
                                                 transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class Food101Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 101

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = Food101(root=self.data_dir, split="train",
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.no_aug else
                                                   get_default_aug_img_transform(self.image_size, scale=False),
                                                   download=True)
            
            self.val_dataset = Food101(root=self.data_dir, split="test",
                                                 transform=get_default_img_transform(self.image_size), download=True)
        else:
            raise NotImplemented()


class FlowersClassification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 102

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = ImageFolder(root=os.path.join(self.data_dir, 'train'),
                                          transform=
                                          get_default_img_transform(self.image_size)
                                          if self.no_aug else
                                          get_default_aug_img_transform(self.image_size, scale=False))

            self.val_dataset = ImageFolder(root=os.path.join(self.data_dir, 'test'),
                                        transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()
        
class ImageNetWithStats(ImageNet):
    def class_stats(self):
        return [v for k, v in sorted(Counter(self.targets).items())]


class ImageNet1kClassification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 1000

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit':
            self.train_dataset = ImageNetWithStats(root=self.data_dir, split='train',
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.no_aug else
                                                   get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ImageNetWithStats(root=self.data_dir, split='val',
                                                 transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class CarsClassification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 196

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = ImageFolder(root=os.path.join(self.data_dir, 'train'),
                                          transform=
                                          get_default_img_transform(self.image_size)
                                          if self.no_aug else
                                          get_default_aug_img_transform(self.image_size, scale=False))

            self.val_dataset = ImageFolder(root=os.path.join(self.data_dir, 'test'),
                                        transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        data = torch.load(file_name)
        self.data = data['latents']
        self.labels = data['targets']

        print(self.class_stats())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0), self.labels[index]

    def class_stats(self):
        return [v for k, v in sorted(Counter(x.item() for x in self.labels).items())]


class EmbedClassification(BaseClassificationDataModule):
    has_test_data = False

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.inst_num_classes = args.num_classes

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        parent_parser = super().add_argparse_args(parent_parser, **kwargs)
        parser = parent_parser.add_argument_group(EmbedClassification.__name__)
        parser.add_argument('--num-classes',
                            help='number of classes',
                            type=int,
                            default=26)
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = EmbedDataset(file_name=os.path.join(self.data_dir, 'embeds_train.pck'))
            self.val_dataset = EmbedDataset(file_name=os.path.join(self.data_dir, 'embeds_val.pck'))
        else:
            raise NotImplemented()
