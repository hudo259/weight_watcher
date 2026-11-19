# Source: CycleMLP paper (SHI Labs), https://github.com/ShoufaChen/CycleMLP
# License: MIT, Copyright (c) 2012 Shoufa Chen (see LICENSES/MIT.txt)
# Modified: 2026 by Thomas Hudcovic. removed mmseg/mmdet registration blocks entirely.
#         Use fork_feat=True directly, the _feat classes were just mmseg wrappers around that flag.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from timm.layers.helpers import to_2tuple

# Building blocks

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CycleFC(nn.Module):
    """
    Cycle Fully-Connected layer implemented via deformable conv with
    fixed cyclic offsets, the spatial mixing operation in CycleMLP.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.dilation     = _pair(dilation)
        self.groups       = groups

        # 1x1 weight, spatial mixing comes from the cyclic offset, not kernel
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, 1, 1)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (
                    (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
                )
            else:
                offset[0, 2 * i + 0, 0, 0] = (
                    (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                )
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        return deform_conv2d_tv(
            x, self.offset.expand(B, -1, H, W),
            self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
        )


class CycleMLP(nn.Module):
    """Token mixer: horizontal CycleFC + vertical CycleFC + channel MLP, gated."""
    def __init__(self, dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c    = nn.Linear(dim, dim, bias=qkv_bias)
        self.sfc_h    = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w    = CycleFC(dim, dim, (3, 1), 1, 0)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = (self.reweight(a)
             .reshape(B, C, 3)
             .permute(2, 0, 1)
             .softmax(dim=0)
             .unsqueeze(2).unsqueeze(2))

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CycleBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = mlp_fn(dim, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)
        self.skip_lam  = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride     = to_2tuple(stride)
        padding    = to_2tuple(padding)
        self.proj  = nn.Conv2d(in_chans, embed_dim,
                               kernel_size=patch_size, stride=stride, padding=padding)
        self.norm  = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size=2):
        super().__init__()
        assert patch_size == 2
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, H, W, C) channel-last
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False,
                 attn_drop=0., drop_path_rate=0., skip_lam=1.0,
                 mlp_fn=CycleMLP, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (drop_path_rate
                     * (block_idx + sum(layers[:index]))
                     / (sum(layers) - 1))
        blocks.append(CycleBlock(
            dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            attn_drop=attn_drop, drop_path=block_dpr,
            skip_lam=skip_lam, mlp_fn=mlp_fn,
        ))
    return nn.Sequential(*blocks)


# CycleNet backbone
class CycleNet(nn.Module):
    """
    CycleMLP Network.

    Args:
        layers:      Blocks per stage.
        embed_dims:  Channel widths per stage.
        transitions: Whether to downsample between stages.
        mlp_ratios:  FFN expansion per stage.
        fork_feat:   If True, return 4 spatial feature maps instead of logits.
                     This is the mode used for dense prediction (segmentation).
                     Pass fork_feat=True instead of using the mmseg _feat classes.
        num_classes: Used only when fork_feat=False.
    """
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dims=None, transitions=None,
                 mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False):
        super().__init__()
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7, stride=4, padding=2,
            in_chans=3, embed_dim=embed_dims[0],
        )

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i], i, layers,
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(Downsample(embed_dims[i], embed_dims[i + 1]))

        self.network = nn.ModuleList(network)

        if fork_feat:
            # out_indices are the stage block indices in network
            # (every other entry, stages at 0, 2, 4, 6)
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                self.add_module(
                    f'norm{i_layer}', norm_layer(embed_dims[i_emb])
                )
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = (nn.Linear(embed_dims[-1], num_classes)
                         if num_classes > 0 else nn.Identity())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)   # (B,C,H,W) -> (B,H,W,C)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm = getattr(self, f'norm{idx}')
                # Permute to (B,C,H,W) for UperNet compatibility
                outs.append(norm(x).permute(0, 3, 1, 2).contiguous())

        if self.fork_feat:
            return outs   # list of 4 (B,C,H,W) tensors

        B, H, W, C = x.shape
        return x.reshape(B, -1, C)

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)

        if self.fork_feat:
            return x   # list of 4 feature maps

        x = self.norm(x)
        return self.head(x.mean(1))


# Variant constructors

def cyclemlp_b1(**kwargs):
    return CycleNet(
        layers=[2, 2, 4, 2], embed_dims=[64, 128, 320, 512],
        patch_size=7, transitions=[True, True, True, True],
        mlp_ratios=[4, 4, 4, 4], mlp_fn=CycleMLP, **kwargs,
    )

def cyclemlp_b2(**kwargs):
    return CycleNet(
        layers=[2, 3, 10, 3], embed_dims=[64, 128, 320, 512],
        patch_size=7, transitions=[True, True, True, True],
        mlp_ratios=[4, 4, 4, 4], mlp_fn=CycleMLP, **kwargs,
    )

def cyclemlp_b4(**kwargs):
    return CycleNet(
        layers=[3, 8, 27, 3], embed_dims=[64, 128, 320, 512],
        patch_size=7, transitions=[True, True, True, True],
        mlp_ratios=[8, 8, 4, 4], mlp_fn=CycleMLP, **kwargs,
    )

def cyclemlp_b5(**kwargs):
    return CycleNet(
        layers=[3, 4, 24, 3], embed_dims=[96, 192, 384, 768],
        patch_size=7, transitions=[True, True, True, True],
        mlp_ratios=[4, 4, 4, 4], mlp_fn=CycleMLP, **kwargs,
    )

# Channel widths per variant
DIMS = {
    "b1": (64, 128, 320, 512),
    "b2": (64, 128, 320, 512),
    "b4": (64, 128, 320, 512),
    "b5": (96, 192, 384, 768),
}

CONSTRUCTORS = {
    "b1": cyclemlp_b1,
    "b2": cyclemlp_b2,
    "b4": cyclemlp_b4,
    "b5": cyclemlp_b5,
}
