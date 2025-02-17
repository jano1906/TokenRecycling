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
from timm.models.vision_transformer import PatchEmbed

from architectures.mae_utils import get_2d_sincos_pos_embed, Block

from architectures.efficientformer import efficient_vit_base_patch16_dec128d4h1b


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.encoder_num_features = embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=False)
        self.patch_size = [patch_size, patch_size]
        num_patches = self.patch_embed.num_patches
        self.grid_size = (
            self.patch_embed.img_size[0] // self.patch_embed.patch_size[0],
            self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
        h = self.grid_size[0]
        w = self.grid_size[1]
        c = int(x.shape[2] / p ** 2)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def comp_next_init(self, K):
        self._comp_next_cache = {
            "cls_mean": torch.zeros_like(self.cls_token),
            "xs_feats": [],
            "i": 0,
            "K": K,
        }   
    
    def comp_next_deinit(self):
        del self._comp_next_cache
    
    def split_input(self, x, masks, include_cls=True):
        assert masks is not None
        assert len(masks[0].shape) == 3, f"{masks[0].shape}"
        masks = [mask.flatten(-2,-1) for mask in masks]
        if masks[0].shape[1] == x.shape[1]:
            assert not include_cls
        elif masks[0].shape[1] == x.shape[1]-1:
            if include_cls:
                masks = [torch.cat([torch.ones(mask.shape[0], 1, dtype=bool, device=mask.device), mask], dim=1) for mask in masks]
            else:
                masks = [torch.cat([torch.zeros(mask.shape[0], 1, dtype=bool, device=mask.device), mask], dim=1) for mask in masks]
        else:
            assert False, f"{masks[0].shape}, {x.shape}"
    
        xs = []
        for mask in masks:
            if mask.shape[0] == 1:
                xs.append(x[:, mask[0, :]].reshape(x.shape[0], -1, x.shape[-1]))
            else:
                xs.append(x[mask].reshape(x.shape[0], -1, x.shape[-1]))
            
        return xs

    def prepare_tokens(self, x, ids=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        if ids is not None:
            ids_ = ids.unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1])
            pe_ = self.pos_embed[:, 1:].expand(ids.shape[0], -1, -1)
            x = x + torch.gather(pe_, 1, ids_)
            
        else:
            x = x + self.pos_embed[:, 1:]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x

    def ___prepare_tokens(self, x, ids=None):
        # embed patches
        x = self.patch_embed(x)
        N, L, D = x.shape  # batch, length, dim
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x = x.gather(1, ids.unsqueeze(2).repeat(1, 1, x.shape[2])).reshape(N, -1, D)
        # Calculate pad_mask
        sorted_indices, indices = torch.sort(ids, dim=1)
        is_overlap = sorted_indices[:, :-1] == sorted_indices[:, 1:]
        is_overlap = torch.cat((torch.full_like(sorted_indices[:, :1], fill_value=False), is_overlap), dim=1)
        pad_mask = torch.gather(is_overlap, 1, indices)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, pad_mask

    def forward_encoder_next(self, x, ids):
        assert hasattr(self, "_comp_next_cache"), "Call 'comp_next_init' before calling 'comp_next'!"
        x = self.prepare_tokens(x, ids=ids)
        K = self._comp_next_cache["K"]
        for blk in self.blocks[:K]:
            x = blk(x)

        cls_mean = self._comp_next_cache["cls_mean"]
        i = self._comp_next_cache["i"]

        self._comp_next_cache["cls_mean"] = cls_mean * i/(i+1) + x[:,[0], :] / (i+1)
        self._comp_next_cache["xs_feats"].append(x[:,1:,:])
        self._comp_next_cache["i"] += 1
        cls_mean, xs_feats = self._comp_next_cache["cls_mean"], self._comp_next_cache["xs_feats"]
        x = torch.cat([cls_mean] + xs_feats, dim=1)
            
        for blk in self.blocks[K:]:
            x = blk(x)
            
        x = self.norm(x)
        return x

    def comp_forward_afterK(self, x, K, masks):
        B = x.shape[0]
        x = self.prepare_tokens(x)


        def subencoder(x):
            for blk in self.blocks[:K]:
                x = blk(x)
            return x
        
        if K > 0 or masks is not None:
            xs = self.split_input(x, masks)

            if all(x.shape[1] == xs[0].shape[1] for x in xs):
                xs = subencoder(torch.cat(xs, dim=0))
                xs = xs.reshape(xs.shape[0]//B, B, *xs.shape[1:])
                
                xs_cls = xs[:, :, 0, :]
                xs_feats = xs[:, :, 1:, :]
                xs_feats = xs_feats.transpose(0,1)
                xs_feats = xs_feats.flatten(1,2)
                x = torch.cat([xs_cls.mean(dim=0).unsqueeze(1), xs_feats], dim=1)
            else:
                xs = [subencoder(x) for x in xs]

                xs_cls = torch.stack([x[:, [0], :] for x in xs])
                xs_feats = [x[:, 1:, :] for x in xs]
                x = torch.cat([xs_cls.mean(dim=0)] + xs_feats, dim=1)
        else:
            x = subencoder(x)
        
        for blk in self.blocks[K:]:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_encoder(self, x, patch_indices):
        # embed patches
        x = self.patch_embed(x)
        N, L, D = x.shape  # batch, length, dim
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x = x.gather(1, patch_indices.unsqueeze(2).repeat(1, 1, x.shape[2])).reshape(N, -1, D)
        # Calculate pad_mask
        #sorted_indices, indices = torch.sort(patch_indices, dim=1)
        #is_overlap = sorted_indices[:, :-1] == sorted_indices[:, 1:]
        #is_overlap = torch.cat((torch.full_like(sorted_indices[:, :1], fill_value=False), is_overlap), dim=1)
        #pad_mask = torch.gather(is_overlap, 1, indices)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, mask, patch_indices):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        x_ = self.mask_token.repeat(x.shape[0], mask.shape[1], 1)
        patch_indices = patch_indices.squeeze(0)
        if len(patch_indices.shape) == 1:
            if patch_indices.shape[0] == 0:
                patch_indices = torch.empty(x.shape[0], 0, device=patch_indices.device)
            else:
                patch_indices = patch_indices.unsqueeze(0).expand(x.shape[0], -1)
        elif len(patch_indices.shape) == 2:
            pass
        else:
            assert False
        
        x_.scatter_(1, patch_indices.unsqueeze(2).repeat(1, 1, x_.shape[2]), x[:, 1:, :])
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

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


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d16h1b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8h1b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec256d8h1b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=256, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec128d4h1b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
