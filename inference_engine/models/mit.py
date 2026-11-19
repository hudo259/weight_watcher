# Source:   https://github.com/lucidrains/segformer-pytorch
# License:  MIT, Copyright (c) 2021 Phil Wang (see LICENSES/MIT.txt)
# Modified: 2026 by Thomas Hudcovic. Original code unchanged. Added the MiTBackbone wrapper at the end of the file for UperNet.

from math import sqrt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# Helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# Building blocks

class DsConv2d(nn.Module):
    """Depthwise-separable conv."""
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                      groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std  = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn   = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with sequence reduction."""
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale  = (dim // heads) ** -0.5
        self.heads  = heads
        self.to_q   = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv  = nn.Conv2d(dim, dim * 2, reduction_ratio,
                                 stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w  = x.shape[-2:]
        heads = self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads),
            (q, k, v),
        )
        sim  = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b i j, b j d -> b i d', attn, v)
        out  = rearrange(out, '(b h) (x y) c -> b (h c) x y',
                         h=heads, x=h, y=w)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super().__init__()
        hidden = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            DsConv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# MiT encoder
class MiT(nn.Module):
    """
    Mix Transformer encoder (SegFormer backbone).

    Produces 4 feature maps at [H/4, H/8, H/16, H/32].

    Args:
        channels:        Input image channels (3).
        dims:            Channel widths per stage.
        heads:           Attention heads per stage.
        ff_expansion:    FFN expansion ratio per stage.
        reduction_ratio: Sequence reduction ratio per stage.
        num_layers:      Number of transformer blocks per stage.
    """
    def __init__(self, *, channels, dims, heads, ff_expansion,
                 reduction_ratio, num_layers):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims       = (channels, *dims)
        dim_pairs  = list(zip(dims[:-1], dims[1:]))
        self.stages = nn.ModuleList()

        for (dim_in, dim_out), (kernel, stride, padding), n_layers, ff_exp, n_heads, red_ratio in zip(
            dim_pairs, stage_kernel_stride_pad,
            num_layers, ff_expansion, heads, reduction_ratio,
        ):
            get_overlap = nn.Unfold(kernel, stride=stride, padding=padding)
            embed       = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)
            layers      = nn.ModuleList()
            for _ in range(n_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(
                        dim=dim_out, heads=n_heads, reduction_ratio=red_ratio,
                    )),
                    PreNorm(dim_out, MixFeedForward(
                        dim=dim_out, expansion_factor=ff_exp,
                    )),
                ]))
            self.stages.append(nn.ModuleList([get_overlap, embed, layers]))

    def forward(self, x, return_layer_outputs=False):
        h, w          = x.shape[-2:]
        layer_outputs = []

        for (get_overlap, embed, layers) in self.stages:
            x           = get_overlap(x)
            num_patches = x.shape[-1]
            ratio       = int(sqrt((h * w) / num_patches))
            x           = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
            x           = embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x
            layer_outputs.append(x)

        return layer_outputs if return_layer_outputs else x


# Variant constructors

def mit_b0(**kwargs):
    return MiT(channels=3, dims=(32, 64, 160, 256),
                heads=(1, 2, 5, 8), ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1), num_layers=(2, 2, 2, 2), **kwargs)

def mit_b1(**kwargs):
    return MiT(channels=3, dims=(64, 128, 320, 512),
                heads=(1, 2, 5, 8), ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1), num_layers=(2, 2, 2, 2), **kwargs)

def mit_b2(**kwargs):
    return MiT(channels=3, dims=(64, 128, 320, 512),
                heads=(1, 2, 5, 8), ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1), num_layers=(3, 4, 6, 3), **kwargs)

def mit_b4(**kwargs):
    return MiT(channels=3, dims=(64, 128, 320, 512),
                heads=(1, 2, 5, 8), ff_expansion=(8, 8, 4, 4),
                reduction_ratio=(8, 4, 2, 1), num_layers=(3, 8, 27, 3), **kwargs)

# Channel widths per variant
DIMS = {
    "b0": (32,  64,  160, 256),
    "b1": (64,  128, 320, 512),
    "b2": (64,  128, 320, 512),
    "b4": (64,  128, 320, 512),
}

CONSTRUCTORS = {"b0": mit_b0, "b1": mit_b1, "b2": mit_b2, "b4": mit_b4}

# Backbone wrapper for UperNet
class MiTBackbone(nn.Module):
    """
    Wraps MiT so it always returns 4 (B, C, H, W) feature maps.
    Used as backbone in UperNet.
    """
    def __init__(self, mit: MiT):
        super().__init__()
        self.mit = mit

    def forward(self, x):
        return self.mit(x, return_layer_outputs=True)
