import os
import sys
import argparse

import torch
from torch import nn
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == '__main__':
    temperature = 0.07
    knns = []
    for kappa in range(0, 13):
        train_path = f"./dino_in1k_train/extract_k16/{kappa}_16.npz"
        test_path = f"./dino_in1k_val/extract_k16/{kappa}_16.npz"

        train_features = torch.tensor(np.load(train_path)['features']).cuda()
        train_labels = torch.tensor(np.load(train_path)['targets']).cuda()

        test_features = torch.tensor(np.load(test_path)['features']).cuda()
        test_labels = torch.tensor(np.load(test_path)['targets']).cuda()
        top1, top5 = knn_classifier(train_features, train_labels,
                                    test_features, test_labels, 20, temperature)
        knns.append((top1, top5))
        print(f"{kappa} {20}-NN classifier result: Top1: {top1}, Top5: {top5}")
    print(knns)