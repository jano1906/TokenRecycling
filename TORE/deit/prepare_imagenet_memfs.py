import os

import torch
from torchvision.datasets.imagenet import ARCHIVE_META, ImageNet

from utils import is_main_process


def prepare_imagenet(tar_path):
    memfs_path = os.environ['MEMFS']
    print('using memfs at:', memfs_path)
    destination_path = os.path.join(memfs_path, 'imagenet')

    if is_main_process():
        os.mkdir(destination_path)
        for tar_file, _ in ARCHIVE_META.values():
            os.symlink(os.path.join(tar_path, tar_file), os.path.join(destination_path, tar_file))
        print('unpacking train set')
        ImageNet(destination_path, split='train')
        print('unpacking val set')
        ImageNet(destination_path, split='val')
        print('done unpacking imagenet at:', destination_path)
    else:
        print('waiting for imagenet preproc')

    torch.distributed.barrier()
    print('done imagenet preproc')
    return destination_path
