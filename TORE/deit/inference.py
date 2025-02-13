import torch
import argparse
import json
import os
from timm.models import create_model
from pathlib import Path
from tqdm import tqdm
import utils
from timm.utils import accuracy
import numpy as np
import models
import models_v2
import models_dino
from mask_const import sample_masks, get_division_masks_for_model, DIVISION_IDS, DIVISION_MASKS
from functools import partial
from fvcore.nn import FlopCountAnalysis
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser('Script containing tasks with inference only', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    # Model parameters
    parser.add_argument('--model', default='vit_small_8', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=1., type=float, help="Crop ratio for evaluation")

    parser.add_argument('--data-path', default=f'~/datasets/imagenet/ILSVRC/Data/CLS-LOC/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'CIFAR10', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--data-split', default='val', choices=['train', 'val', 'test'],
                        type=str)
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path where to save')
    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument('--device', default="cuda", type=str)
    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--evaluate', nargs='*', default=[])
    parser.add_argument('--extract', nargs='*', default=[])
    parser.add_argument('--count_flops', action="store_true")
    
    parser.add_argument('--group_by_class', action="store_true")
    parser.add_argument('--random_masks', action="store_true")
    
    return parser


# -------------------- GENERIC ---------------------

@torch.no_grad()
def evaluate(data_loader, model, device, KMs, random_masks, seq: bool=False, group_by_class: bool=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()
    division_masks = get_division_masks_for_model(model)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if group_by_class:
            classes = set(target.cpu().numpy())
            grouped = {c: [images[target == c].clone(), torch.full(((target == c).sum().item(),), c, device=target.device)] for c in classes}
        else:
            classes = [0]
            grouped = {0: [images, target]}
        for c, [images, target] in grouped.items():
            # compute output
            with torch.cuda.amp.autocast():
                outputs = [
                    [[k, m], model(images, K=k, seq=seq, masks=sample_masks(division_masks, m) if random_masks else division_masks[m][0])]
                    for k, m in KMs
                ]
            accuracies = [
                [[k, m, i], accuracy(out, target)[0]]
                for [[k, m], outs] in outputs
                for i, out in (enumerate(outs) if seq else [[0, outs]])
            ]

            batch_size = images.shape[0]
            for [[k, m, i], acc] in accuracies:
                name = ''
                if group_by_class:
                    name += f'cls{c}_'
                name += 'acc1'
                if [k, m] != [0, 1]:
                    name += f"_K{k}_M{m}"
                    if seq:
                        name += f"_i{i}"
                metric_logger.meters[name].update(acc.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def extract(data_loader, model, device, KMs, random_masks, seq: bool=False):
    # switch to evaluation mode
    model.eval()
    division_masks = get_division_masks_for_model(model)
    
    ret = {
        f"{k}_{m}": {"features" : [], "targets": []}
        for k, m in KMs
    }

    for images, targets in tqdm(data_loader, "extracting features "):
        images = images.to(device, non_blocking=True)
        targets = targets.cpu().numpy()
        with torch.cuda.amp.autocast():
            for k, m in KMs:
                if random_masks:
                    masks = sample_masks(division_masks, m)
                else:
                    masks = division_masks[m][0]
                features = model(images, K=k, masks=masks).cpu().numpy()
                ret[f"{k}_{m}"]["features"].append(features)
                ret[f"{k}_{m}"]["targets"].append(targets)

    for k, m in KMs:
        ret[f"{k}_{m}"]["features"] = np.concatenate(ret[f"{k}_{m}"]["features"])
        ret[f"{k}_{m}"]["targets"] = np.concatenate(ret[f"{k}_{m}"]["targets"])
    return ret

def np_dict_to_dir_tree(root, dic):
    if set(dic.keys()) == set(["features", "targets"]):
        np.savez_compressed(root, **dic)
        return
    os.makedirs(root, exist_ok=True)
    for k in dic:
        np_dict_to_dir_tree(os.path.join(root, str(k)), dic[k])

# -------------------- SPECIALIZED ---------------------

def evaluate_km(data_loader, model, device, group_by_class, random_masks, *args, **kwargs):
    KMs = [[k, m] for m in get_division_masks_for_model(model).keys() for k in range(len(model.blocks)+1)]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class, random_masks=random_masks)

def evaluate_01_816(data_loader, model, device, group_by_class, random_masks, *args, **kwargs):
    KMs = [[0, 1], [8, 16]]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class, random_masks=random_masks)

def evaluate_k16_seq(data_loader, model, device, group_by_class, random_masks, *args, **kwargs):
    KMs = [[k, 16] for k in range(len(model.blocks)+1)]
    return evaluate(data_loader, model, device, KMs=KMs, seq=True, group_by_class=group_by_class, random_masks=random_masks)

def extract_k16(data_loader, model, device, random_masks, *args, **kwargs):
    KMs = [[k, 16] for k in range(len(model.blocks)+1)]
    return extract(data_loader, model, device, KMs=KMs, random_masks=random_masks)

def extract_01(data_loader, model, device, random_masks, *args, **kwargs):
    return extract(data_loader, model, device, KMs=[[0, 1]], random_masks=random_masks)


# -------------------- FLOPS ---------------------

def count_flops(create_model_fn, img_size):
    IMG = torch.zeros(1, 3, img_size, img_size)
    division_masks = DIVISION_MASKS[14][16][0]
    division_ids = DIVISION_IDS[14][16][0]
    imgs = []
    for divm in division_masks:
        divm = np.expand_dims(divm, [0, 1]).repeat(3, axis=1).repeat(16, axis=2).repeat(16, axis=3)
        divm = np.expand_dims(divm, axis=0)
        H, W = divm.sum(axis=3).max(), divm.sum(axis=4).max()
        imgs.append(IMG[divm].reshape(1, 3, H, W))

    with torch.no_grad():
        flops = {}
        for k in tqdm(range(len(create_model_fn().blocks)), f"K: "):
            flops[k] = []
            model = create_model_fn()
            model.comp_next_init()
            cache = model._comp_next_cache
            for i, [img, id] in enumerate(zip(imgs, division_ids)):
                model = create_model_fn()
                model.forward = partial(model.comp_next, K=k, ids=id)
                model.eval()
                model._comp_next_cache = cache
                flops[k].append(FlopCountAnalysis(model, img).total())
                cache = model._comp_next_cache
                assert len(cache["xs_feats"]) == i+1, len(cache["xs_feats"])
                assert cache["i"] == i+1, cache["i"]
    return flops


from datasets import transforms, datasets, ImageFolder, PatchedImageFolder, INatDataset, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def default_transform_dataset(is_train, args):
    resize_im = args.input_size > 32
    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
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

    return dataset, nb_classes


def main_setup(args):
    if args.checkpoint is None:
        print("[WARNING] --checkpoint is None")

    output_dir = Path(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=args.output_dir == "debug")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.distributed:
        utils.init_distributed_mode(args)
    dataset, nb_classes = default_transform_dataset(is_train=args.data_split == "train", args=args)
    with open(os.path.join(output_dir, "class_to_idx.json"), "a") as f:
        f.write(json.dumps(dataset.class_to_idx))
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
        img_size=(args.input_size, args.input_size)
    )
    model = model.to(args.device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['teacher']
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Loaded checkpoint: ", msg)

    return model, data_loader, nb_classes, output_dir


def main(args):
    model, data_loader, nb_classes, output_dir = main_setup(args)
    if args.checkpoint is not None:
        for task_name in args.evaluate:
            task = globals()[task_name]
            test_stats = task(
                data_loader=data_loader,
                model=model,
                device=args.device,
                group_by_class=args.group_by_class,
                random_masks=args.random_masks)
            with open(os.path.join(output_dir, task_name + ".txt"), "w") as f:
                f.write(json.dumps(test_stats))

        for task_name in args.extract:
            assert not args.distributed
            task = globals()[task_name]
            test_stats = task(
                data_loader=data_loader,
                model=model,
                device=args.device,
                group_by_class=args.group_by_class,
                random_masks=args.random_masks)
            np_dict_to_dir_tree(os.path.join(output_dir, task_name), test_stats)
    
    if args.count_flops:
        create_model_fn = partial(create_model,
            args.model,
            pretrained=False,
            num_classes=nb_classes,
            img_size=args.input_size

        )
        flops = count_flops(create_model_fn, args.input_size)
        flat_flops = [(str(k), i, v / 1e9) for k, l in flops.items() for i, v in enumerate(l)]
        df = pd.DataFrame(flat_flops, columns=["K", "i", "GFLOPs"])
        pd.DataFrame.to_csv(df, os.path.join(output_dir, "count_flops.csv"))


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
