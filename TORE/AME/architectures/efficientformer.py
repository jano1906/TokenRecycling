# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from architectures.mae_utils import get_2d_sincos_pos_embed, Block

from timm.models.efficientformer_v2 import efficientformerv2_l

class EfficientViTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = 224
        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder = efficientformerv2_l(pretrained=True)
        self.encoder_num_features = self.encoder.num_features

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        decoder_embed_dim = 128
        num_patches = 196
        decoder_num_heads = 4
        mlp_ratio = 4
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        decoder_depth = 1
        patch_size = 16
        self.patch_size = [patch_size, patch_size]
        self.grid_size = [14,14]
        out_chans = 3
        norm_pix_loss = False
        self.decoder_embed = nn.Linear(384, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        c = imgs.shape[1]
        p = self.patch_size[0]
        assert imgs.shape[2] % p == 0
        assert imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size[0]
        h = self.grid_size[0]
        w = self.grid_size[1]
        c = int(x.shape[2] / p ** 2)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def comp_forward_afterK(self, x, K, masks):
        assert K == 0
        mask = sum(masks)
        x = x * mask.repeat_interleave(self.patch_size[0], dim=-2).repeat_interleave(self.patch_size[1], dim=-1)
        x = self.encoder.forward_features(x)
        x = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        x = x.flatten(-2, -1)
        x = x.transpose(-1, -2)
        return x
    
    def comp_next_init(self, K):
        self._comp_next_cache = {
            "mask": None
        }
        pass   
    
    def forward_encoder(self, x, patch_indices):
        assert patch_indices.flatten().shape == (0,), patch_indices.flatten().shape
        return self.comp_forward_afterK(x, 0, [torch.zeros(1,14,14, device = x.device)])


    def forward_encoder_next(self, x, ids):
        if self._comp_next_cache["mask"] is None:
            self._comp_next_cache["mask"] = torch.zeros(x.shape[0], 1, *self.grid_size, device=x.device)
        mask = self._comp_next_cache["mask"].flatten(1)
        mask.scatter_(-1, ids.to(device=mask.device), torch.ones_like(mask, device = mask.device))
        mask = mask.reshape(mask.shape[0], 1, *self.grid_size)
        self._comp_next_cache["mask"] = mask

        return self.comp_forward_afterK(x, 0, [mask])

    def forward_decoder(self, x, mask, patch_indices):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        
        mask = mask.unsqueeze(-1)
        x = x * mask + self.mask_token * (~mask)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if mask is not None:
            mask_neg = ~mask
            loss = (loss * mask_neg).sum() / mask_neg.sum()  # mean loss on removed patches
        else:
            loss = loss.sum() / pred.shape[1]  # mean loss on all patches

        return loss

    def reconstruct(self, pred, target, mask: torch.Tensor):
        with torch.no_grad():
            pred_img = pred.detach().clone()
            pred_img[mask, :] = self.patchify(target)[mask, :]
            pred_img = self.unpatchify(pred_img)
            return pred_img

    @torch.no_grad()
    def segmentation_output(self, pred):
        pred = self.unpatchify(pred)
        return torch.argmax(pred, dim=1)

    @property
    def last_attn(self):
        return torch.stack([block.attn.last_attn for block in self.decoder_blocks], dim=0)

def efficient_vit_base_patch16_dec128d4h1b(**kwargs):
    model = EfficientViTAutoencoder()
    return model
