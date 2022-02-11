"""
"Learning Features from Parameter-Free Layers"
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

# This code is written upon https://github.com/naver-ai/pit/blob/master/pit.py

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from vit_pf import Block as eff_transformer_block


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None,
                 attn_type='attn'):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        attn_types = ['attn' if i % 2 == 0 else attn_type for i in range(depth)]
        self.blocks = nn.ModuleList([
            eff_transformer_block(dim=embed_dim, num_heads=heads,
                                  mlp_ratio=mlp_ratio, qkv_bias=True,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=drop_path_prob[i],
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                  attn_type=attn_types[i]) for i in
            range(depth)])

    def forward(self, x):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):
        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride,
                 base_dims, depth, heads, mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0,
                 attn_type='attn'):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True)
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio, drop_rate, attn_drop_rate,
                            drop_path_prob, attn_type=attn_type)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2)
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def change_resolution(self, h, w):
        self.pos_embed = nn.Parameter(
            F.interpolate(self.pos_embed.data, (h, w), mode='bicubic'),
            requires_grad=True
        )

    def forward_features(self, x):
        x = self.patch_embed(x)

        if x.shape[2:4] == self.pos_embed.shape[2:4]:
            pos_embed = self.pos_embed
        else:
            pos_embed = F.interpolate(self.pos_embed, x.shape[2:4],
                                      mode='bicubic')

        x = self.pos_drop(x + pos_embed)

        features = []

        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            features.append(x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        features.append(x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = self.gap(rearrange(x, 'b n c -> b c n')).flatten(start_dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def pit_b_max(pretrained=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        attn_type='max',
        **kwargs
    )

    return model


@register_model
def pit_s_max(pretrained=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        attn_type='max',
        **kwargs
    )

    return model


@register_model
def pit_ti_max(pretrained=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        attn_type='max',
        **kwargs
    )

    return model


if __name__ == '__main__':
    model = pit_s_max().eval()  # .cuda()

    input = torch.randn(1, 3, 224, 224)  # .cuda()
    output = model(input)

    loss = output.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')
