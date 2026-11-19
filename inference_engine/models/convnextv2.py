# Source: Meta Platforms, Inc., https://github.com/facebookresearch/ConvNeXt-V2
# License: MIT, Copyright (c) Meta Platforms, Inc. and affiliates (see LICENSES/MIT.txt)
# Modified: 2026 by Thomas Hudcovic. Added forward_multiscale(), which returns the 4 stage feature maps for the UperNet decoder

import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first and channels_last."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class GRN(nn.Module):
    """Global Response Normalization."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class Block(nn.Module):
    """ConvNeXt V2 block: depthwise conv + norm + 2× pointwise linear + GRN."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv   = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = LayerNorm(dim, eps=1e-6)
        self.pwconv1  = nn.Linear(dim, 4 * dim)
        self.act      = nn.GELU()
        self.grn      = GRN(4 * dim)
        self.pwconv2  = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)

class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2.

    Args:
        in_chans:        Input channels.
        num_classes:     Classification head output size.
        depths:          Number of blocks per stage.
        dims:            Channel widths per stage.
        drop_path_rate:  Stochastic depth rate.
        head_init_scale: Scale factor for head init.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.dims   = dims

        # Stem + 3 downsampling layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        # 4 stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dims[i], dp_rates[cur + j]) for j in range(depths[i])]
            ))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # --- Original classification path (unchanged) ---
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))

    # CHANGE: added for UperNet. Returns 4 spatial feature maps
    def forward_multiscale(self, x):
        """
        Returns [(B,C0,H/4,W/4), (B,C1,H/8,W/8),
                 (B,C2,H/16,W/16), (B,C3,H/32,W/32)].
        Does NOT apply the final norm or GAP, spatial info is preserved.
        """
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features


# --- Variant constructors ---

def convnextv2_tiny(**kwargs):
    return ConvNeXtV2(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)

def convnextv2_base(**kwargs):
    return ConvNeXtV2(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), **kwargs)

def convnextv2_large(**kwargs):
    return ConvNeXtV2(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536), **kwargs)

# Channel widths per variant. Needed by UperNet to set feature_channels
DIMS = {
    "tiny":  (96, 192, 384, 768),
    "base":  (128, 256, 512, 1024),
    "large": (192, 384, 768, 1536),
}
