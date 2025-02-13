from functools import partial
from typing import Any, Dict

import torch
import torchmetrics
from torch import nn

from architectures.base import BaseArchitecture
from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.base import BaseDataModule
from datasets.classification import BaseClassificationDataModule, EmbedClassification
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD
from architectures.mae import MaskedAutoencoderViT
from architectures.efficientformer import EfficientViTAutoencoder

class ClassificationMae(BaseGlimpseMae):

    def __init__(self, args, datamodule):
        super().__init__(args, datamodule)
        assert isinstance(datamodule, BaseClassificationDataModule)
        self.num_classes = datamodule.num_classes
        if self.sequential_predictions:
            for i in range(self.num_glimpses):
                self.define_metric(f'accuracy_{i}',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.num_classes,
                                   average='micro'))

        self.define_metric('accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.num_classes,
                                   average='micro'))
        
        self.imagenet_mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
        self.imagenet_std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)

        self.define_metric('rmse_overall', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('rmse_pred', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('rmse_masked', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('tina', torchmetrics.MeanMetric)

        self.head = nn.Sequential(
            nn.Linear(self.mae.encoder_num_features, self.num_classes),
        )

        self.classifications = []

        self.criterion = nn.CrossEntropyLoss()

    def forward_one(self, x, mask_indices, mask, glimpses, new_mask_list, new_mask_ids_list, K=None, single_step=False, initial_step=False) -> Dict[str, torch.Tensor]:
        out = super().forward_one(x, mask_indices, mask, glimpses, new_mask_list, new_mask_ids_list, K, single_step, initial_step)
        if isinstance(self.mae, MaskedAutoencoderViT):
            latent = out['latent'][:, 0, :]  # get cls token
        elif isinstance(self.mae, EfficientViTAutoencoder):
            latent = out['latent'].mean(dim=1) # avgpool
        else:
            assert False
        out['classification'] = self.head(latent)
        if self.sequential_predictions and not initial_step:
            self.classifications.append(out["classification"])
        
        return out

    def calculate_loss_one(self, out, batch):
        if self.rec_loss:
            rec_loss = self.mae.forward_loss(batch[0], out['out'], out['mask'] if self.masked_loss else None)
        else:
            rec_loss = 0
        cls_loss = self.criterion(out['classification'], batch[1])
        return rec_loss * 0.1 + cls_loss

    def __rev_normalize(self, img):
        if self.imagenet_mean.device != img.device:
            self.imagenet_mean = self.imagenet_mean.to(img.device)
        if self.imagenet_std.device != img.device:
            self.imagenet_std = self.imagenet_std.to(img.device)
        
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        if self.sequential_predictions:
            for i, y in enumerate(self.classifications):
                self.log_metric(mode, f'accuracy_{i}', y, batch[1])
            self.classifications = []

        if self.rec_loss:
            with torch.no_grad():
                if out["mask"].shape[0] == 1:
                    out["mask"] = out["mask"].repeat(batch[0].shape[0], 1)
                reconstructed = self.mae.reconstruct(out['out'], batch[0], out['mask'])
                reconstructed = self.__rev_normalize(reconstructed)
                pred = self.mae.unpatchify(out['out'])
                pred = self.__rev_normalize(pred)
                target = self.__rev_normalize(batch[0])
                mask_neg = ~out['mask']

                self.log_metric(mode, 'rmse_overall', reconstructed, target)
                self.log_metric(mode, 'rmse_pred', pred, target)
                self.log_metric(mode, 'rmse_masked', self.mae.patchify(reconstructed)[mask_neg, :],
                                self.mae.patchify(target)[mask_neg, :])
                tina_metric = torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, 1)), [0, 1, 2])
                self.log_metric(mode, 'tina', tina_metric)

        self.log_metric(mode, 'accuracy', out['classification'], batch[1])


class RandomClsMae(ClassificationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardClsMae(ClassificationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionClsMae(ClassificationMae):
    glimpse_selector_class = AttentionGlimpseSelector


class EmbedClassifier(BaseArchitecture):
    def __init__(self, args: Any, datamodule: BaseDataModule):
        super().__init__(args, datamodule)
        assert isinstance(datamodule, EmbedClassification)
        self.num_classes = datamodule.num_classes

        self.cls = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.define_metric('accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy,
                                   num_classes=self.num_classes,
                                   average='micro'))

    def forward(self, batch) -> Any:
        x = batch[0]
        target = batch[1]

        x = x[:, 0, 0, :]
        x = self.cls(x)

        loss = self.criterion(x, target)

        return {"out": x, "loss": loss}

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        self.log_metric(mode, 'accuracy', out['out'], batch[1])
