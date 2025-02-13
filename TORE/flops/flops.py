from fvcore.nn import FlopCountAnalysis
import torch
import json
from torch import nn
from typing import *
from argparse import ArgumentParser
import math

from unstructured_efficientformer import unstructured_efficient_l7, unstructured_efficient_l3, unstructured_efficient_l1
from timm.models.swin_transformer import swin_base_patch4_window7_224 as swin_base,\
                                         swin_tiny_patch4_window7_224 as swin_tiny
from timm.models.efficientformer_v2 import efficientformerv2_l as efficient_v2_l, efficientformerv2_s2
from timm.models.efficientformer import efficientformer_l7, efficientformer_l3, efficientformer_l1
from timm.models.vision_transformer import vit_base_patch16_224 as vit_base,\
                                           vit_small_patch16_224 as vit_small
from timm.models.pvt_v2 import pvt_v2_b5, pvt_v2_b2, pvt_v2_b4

from timm.models.vision_transformer import Block, Mlp, LayerType, init_weights_vit_timm, get_init_weights_vit, _load_weights
from functools import partial
from timm.layers import to_2tuple, PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module





class UnstructuredPatchEmbed(nn.Module):
    """ Image tokens to Patch Embedding
    """
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
            **kwargs
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        #self.proj = nn.Linear(in_chans * self.patch_size[0] * self.patch_size[1], embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.flatten(0,1)
        x = self.proj(x)
        x = x.reshape(B,N,*x.shape[1:])
        x = x.permute(0,1,3,4,2)
        x = x.flatten(1,3)
        x = self.norm(x)
        return x


class UnstructuredVisionTransformer(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = UnstructuredPatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            **kwargs,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

        self.reset_tore_cache()

    def reset_tore_cache(self):
        self._tore_cache = {
            "cls_mean": torch.zeros_like(self.cls_token),
            "xs_feats": [],
            "i": 0,
            "k": None,
        }


    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor, ids) -> torch.Tensor:
        B,N,C = x.shape
        assert not self.dynamic_img_size
        assert len(ids) == x.shape[1], [len(ids), x.shape]

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            pos_embed = self.pos_embed[:, ids]
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            offset = len(to_cat)
            ids = list(range(offset)) + [i+offset for i in ids]
            pos_embed = self.pos_embed[:, ids]
            
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor, ids, k, use_cache) -> torch.Tensor:
        if self._tore_cache["k"] is None:
            self._tore_cache["k"] = k

        assert self._tore_cache["k"] == k
        
        x = self.patch_embed(x)
        x = self._pos_embed(x, ids)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        assert not self.grad_checkpointing

        for blk in self.blocks[:k]:
            x = blk(x)

        if use_cache:
            cls_mean = self._tore_cache["cls_mean"]
            i = self._tore_cache["i"]

            self._tore_cache["cls_mean"] = cls_mean * i/(i+1) + x[:,[0], :] / (i+1)
            self._tore_cache["xs_feats"].append(x[:,1:,:])
            self._tore_cache["i"] += 1
            cls_mean, xs_feats = self._tore_cache["cls_mean"], self._tore_cache["xs_feats"]
            x = torch.cat([cls_mean] + xs_feats, dim=1)

        for blk in self.blocks[k:]:
            x = blk(x)
    
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, ids, k=0, use_cache=False) -> torch.Tensor:
        x = self.forward_features(x, ids, k, use_cache)
        x = self.forward_head(x)
        return x

def unstructured_vit_large():
    model = UnstructuredVisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    return model

def unstructured_vit_base():
    model = UnstructuredVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return model

def unstructured_vit_small():
    model = UnstructuredVisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return model












MODELS = {
    "unstructured_vit_large": unstructured_vit_large,
    "unstructured_vit_base": unstructured_vit_base,
    "unstructured_vit_small": unstructured_vit_small,
    "tore_vit_base": unstructured_vit_base,
    "tore_vit_small": unstructured_vit_small,
    "vit_base": vit_base,
    "vit_small": vit_small,
    "swin_base": swin_base,
    "swin_tiny": swin_tiny,
    "pvt_v2_b2": pvt_v2_b2,
    "pvt_v2_b4": pvt_v2_b4,
    "pvt_v2_b5": pvt_v2_b5,
    "efficient_v2_l": efficient_v2_l,
    "efficient_l7": efficientformer_l7,
    "efficient_l3": efficientformer_l3,
    "efficient_l1": efficientformer_l1,
    "unstructured_efficient_l7": unstructured_efficient_l7,
    "unstructured_efficient_l3": unstructured_efficient_l3,
    "unstructured_efficient_l1": unstructured_efficient_l1,
    }

for k in range(13):
    MODELS[f"tore_k{k}_vit_small"] = unstructured_vit_small
    MODELS[f"tore_k{k}_vit_base"] = unstructured_vit_base

def decoder_factory(d, b, h):
    def Decoder():
        decoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=h)
        transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=b)
        return transformer_decoder
    return Decoder

DECODERS = {
    "decoder_512d8b16h": decoder_factory(512, 8, 16),
    "decoder_512d1b16h": decoder_factory(512, 1, 16),
    "decoder_256d1b8h": decoder_factory(256, 1, 8),
    "decoder_128d1b4h": decoder_factory(128, 1, 4),
    "decoder_064d1b2h": decoder_factory(64, 1, 2),
    "decoder_032d1b1h": decoder_factory(32, 1, 1),
}

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str)
    parser.add_argument("--decoders", nargs="+", type=str)
    parser.add_argument("--glimpse_size", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=128)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--n_glimpses", type=int, default=8)
    return parser

def count_flops(Model, input):
    model = Model()
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input).total()
    return flops, model

def main(args):
    ret = []
    input = torch.zeros(1,3,args.input_h,args.input_w)
    to_iter = {}
    if args.models:
        if args.models[0] == "all":
            to_iter.update(MODELS)
        else:
            to_iter.update({m: MODELS[m] for m in args.models})

    if args.decoders:
        if args.decoders[0] == "all":
            to_iter.update(DECODERS)
        else:
            to_iter.update({m: DECODERS[m] for m in args.decoders})
    

    for model_name in to_iter:
        Model = to_iter[model_name]
        if model_name.startswith("unstructured"):
            flops = []
            for n in range(1, args.n_glimpses):
                x = torch.zeros(1,n,3,args.glimpse_size*16, args.glimpse_size*16)
                f, _ = count_flops(Model, (x, list(range((args.glimpse_size**2) *n))))
                flops.append(f)
        elif model_name.startswith("tore"):
            flops = []
            ids = list(range(args.glimpse_size ** 2))
            x = torch.zeros(1, 1, 3, args.glimpse_size*16,args.glimpse_size*16)
            k = int(model_name.split("_")[1][1:])
            def Model_factory(tore_cache):
                def _Model():
                    model = Model()
                    if tore_cache is not None:
                        model._tore_cache = tore_cache
                    return model
                return _Model
            tore_cache = None
            for n in range(26):
                f, m = count_flops(Model_factory(tore_cache), (x, ids, k, True))
                tore_cache = m._tore_cache
                flops.append(f)
        elif model_name.startswith("decoder"):
            dim = model_name.split("_")[1][:3]
            dim = int(dim)
            x = torch.zeros(1, 196, dim)
            flops, _ = count_flops(Model, x)
        
        else: flops, _ = count_flops(Model, input)

        ret.append({"model_name": model_name, "flops": flops, "glimpse_size": args.glimpse_size, "image_size": (args.input_h, args.input_w)})

    return ret

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ret = main(args)
    for rec in ret:
        print(json.dumps(rec))
