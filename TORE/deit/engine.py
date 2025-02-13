# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import random
from typing import Iterable, Optional, Callable, Dict

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from mask_const import sample_masks, get_division_masks_for_model


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, ext_logger: Optional[Callable[[Dict, int], None]] = None,
                    sample_divisions = False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            K, M = 0, 1
            if sample_divisions:
                division_masks = get_division_masks_for_model(model)
                K = random.randint(0, len(model.blocks))
                M = random.choice(list(division_masks.keys()))
                masks = sample_masks(division_masks, M)
            
            outputs = model(samples, K=K, masks=masks if M != 1 else None)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if ext_logger is not None:
        ext_logger({
            'Train/Loss': metric_logger.loss.global_avg,
            'Train/LR': metric_logger.lr.global_avg
        }, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch=None, ext_logger: Optional[Callable[[Dict, int], None]] = None, KMs=[[0,1]]):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    division_masks = get_division_masks_for_model(model)


    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = [
                [[k, m], model(images, K=k, masks=sample_masks(division_masks, m))]
                for k, m in KMs
            ]
        
        accuracies = [
            [[k, m], accuracy(outs, target)[0]]
            for [[k, m], outs] in outputs
        ]

        batch_size = images.shape[0]
        for [[k, m], acc] in accuracies:
            name = 'acc1'
            if [k, m] != [0, 1]:
                name += f"_K{k}_M{m}"
            metric_logger.meters[name].update(acc.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if ext_logger is not None:  
        dic = {f'Eval/{k}': v.global_avg for k, v in metric_logger.meters.items()}
        ext_logger(dic, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
