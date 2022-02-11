"""
"Learning Features from Parameter-Free Layers"
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

# This code is written upon https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import OrderedDict

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Mlp, trunc_normal_, Attention
from timm.models.registry import register_model
from timm.models.layers import DropPath


class EffLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3, bias=True, kernel_size=3,
                 proj_drop=0.):
        super().__init__()

        self.qkv = nn.Linear(dim, dim * mlp_ratio, bias=bias)
        self.proj = nn.Linear(mlp_ratio * dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                 padding=kernel_size // 2, stride=1)
        self.actv = nn.GELU()

    def forward(self, x):
        x = self.qkv(x)
        x = self.actv(x)
        x = rearrange(x, 'b (h w) c -> b c h w',
                      h=int(x.shape[1] ** (1 / 2)), w=int(x.shape[1] ** (1 / 2)))
        x = self.pool(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.actv(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='max',
                 kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn_type == 'max':
            self.attn = EffLayer(dim, mlp_ratio=3, kernel_size=kernel_size,
                                 bias=True, proj_drop=drop)
        elif attn_type == 'attn':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop)

        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EffTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 representation_size=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 attn_type='max'):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                depth)]  # stochastic depth decay rule
        attn_types = ['attn' if i % 2 == 0 else attn_type for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, attn_type=attn_types[i]) for i in
            range(depth)])
        self.norm = norm_layer(embed_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict(
                [('fc', nn.Linear(embed_dim, representation_size)),
                 ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features,
                              num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.gap(rearrange(x, 'b n c -> b c n')).flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def vit_ti_max(pretrained=False, **kwargs):
    model = EffTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                           mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           attn_type='max', **kwargs)

    return model


@register_model
def vit_s_max(pretrained=False, **kwargs):
    model = EffTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                           mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           attn_type='max', **kwargs)

    return model


@register_model
def vit_b_max(pretrained=False, **kwargs):
    model = EffTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                           mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           attn_type='max', **kwargs)

    return model


if __name__ == '__main__':
    model = vit_s_max().eval()  # .cuda()

    input = torch.randn(1, 3, 224, 224)  # .cuda()
    output = model(input)

    loss = output.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')
