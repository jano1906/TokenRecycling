from timm.models.efficientformer import EfficientFormer, Stem4, EfficientFormerStage, Downsample, MetaBlock2d, LayerScale
import torch
from torch import nn
from timm.layers import DropPath, trunc_normal_, to_2tuple, Mlp
from typing import Dict


class Convert(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2).flatten(1,3)
        return x



class Attention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim=384,
            key_dim=32,
            num_heads=8,
            attn_ratio=4,
            resolution=7
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio

        self.qkv = nn.Linear(dim, self.key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        resolution = to_2tuple(resolution)
        pos = torch.stack(torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        self.register_buffer('attention_bias_idxs', rel_pos)
        self.attention_bias_cache = {}  # per-device attention_biases cache (data-parallel compat)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device, ids) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            assert False, self.attention_bias_idxs #TODO

            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x, ids):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.val_dim], dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn # TODO: + self.get_attention_biases(x.device, ids)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)
        return x

class MetaBlock1d(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            proj_drop=0.,
            drop_path=0.,
            layer_scale_init_value=1e-5
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x, ids):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x), ids)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class EfficientFormerStage(nn.Module):

    def __init__(
            self,
            dim,
            dim_out,
            depth,
            downsample=True,
            num_vit=1,
            pool_size=3,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_cl=nn.LayerNorm,
            proj_drop=.0,
            drop_path=0.,
            layer_scale_init_value=1e-5,
):
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = Downsample(in_chs=dim, out_chs=dim_out, norm_layer=norm_layer)
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        if num_vit and num_vit >= depth:
            blocks.append(Convert())

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit and num_vit > remain_idx:
                blocks.append(
                    MetaBlock1d(
                        dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer_cl,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    ))
            else:
                blocks.append(
                    MetaBlock2d(
                        dim,
                        pool_size=pool_size,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    ))
                if num_vit and num_vit == remain_idx:

                    blocks.append(Convert())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, ids):
        B,N,C,H,W = x.shape
        x = x.flatten(0,1)
        x = self.downsample(x)
        x = x.reshape(B,N,*x.shape[1:])
        
        #print(f"after {self.downsample}", x.shape)
        for blk in self.blocks:
            if isinstance(blk, MetaBlock2d) and len(x.shape) == 5:
                x = x.flatten(0,1)
            elif not isinstance(blk, MetaBlock2d) and len(x.shape) == 4:
                x = x.reshape(B,N,*x.shape[1:])
        #    print(type(blk), x.shape)
            if isinstance(blk, MetaBlock1d):
                x = blk(x, ids)
            else:
                x = blk(x)
        if len(x.shape) == 4:
            x = x.reshape(B,N,*x.shape[1:])
            
        #print(f"after {["1d" if isinstance(b, MetaBlock1d) else "2d" for b in self.blocks]} ", x.shape)
        return x

class EfficientFormer(nn.Module):

    def __init__(
            self,
            depths,
            embed_dims=None,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            downsamples=None,
            num_vit=0,
            mlp_ratios=4,
            pool_size=3,
            layer_scale_init_value=1e-5,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_cl=nn.LayerNorm,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool

        self.stem = Stem4(in_chans, embed_dims[0], norm_layer=norm_layer)
        prev_dim = embed_dims[0]

        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        stages = []
        for i in range(len(depths)):
            stage = EfficientFormerStage(
                prev_dim,
                embed_dims[i],
                depths[i],
                downsample=downsamples[i],
                num_vit=num_vit if i == 3 else 0,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                norm_layer_cl=norm_layer_cl,
                norm_layer=norm_layer,
                proj_drop=proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            prev_dim = embed_dims[i]
            stages.append(stage)

        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = embed_dims[-1]
        self.norm = norm_layer_cl(self.num_features)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # assuming model is always distilled (valid for current checkpoints, will split def if that changes)
        self.head_dist = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.apply(self._init_weights)

    # init for classification
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters() if 'attention_biases' in k}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x, ids):
        B, N, C, H, W = x.shape
        x = x.flatten(0,1)
        x = self.stem(x)
        x = x.reshape(B,N,*x.shape[1:])
        #print("after stem ", x.shape)
        for stage in self.stages:
            x = stage(x, ids)
        #print("after stages ", x.shape)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward(self, x, ids):
        x = self.forward_features(x, ids)
        x = self.forward_head(x)
        return x













EfficientFormer_width = {
    'l1': (48, 96, 224, 448),
    'l3': (64, 128, 320, 512),
    'l7': (96, 192, 384, 768),
}

EfficientFormer_depth = {
    'l1': (3, 2, 6, 4),
    'l3': (4, 4, 12, 6),
    'l7': (6, 6, 18, 8),
}



def unstructured_efficient_l7():
    model_args = dict(
        depths=EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer_width['l7'],
        num_vit=8,
    )
    model = EfficientFormer(**model_args)
    return model

def unstructured_efficient_l3():
    model_args = dict(
        depths=EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer_width['l3'],
        num_vit=8,
    )
    model = EfficientFormer(**model_args)
    return model

def unstructured_efficient_l1():
    model_args = dict(
        depths=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        num_vit=1,
    )
    model = EfficientFormer(**model_args)
    return model

#input = torch.ones(1,3,224,224)
#input = torch.ones(1, 4,3,32,32)
#model(input, [[1,2,3,4]])