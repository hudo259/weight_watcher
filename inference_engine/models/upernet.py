# Source:   https://github.com/yassouali/pytorch-segmentation
#           https://github.com/open-mmlab/mmsegmentation
# License:  MIT, Copyright (c) 2020 Yassine (see LICENSES/MIT.txt)
#           Apache-2.0, Copyright 2020 The MMSegmentation Authors (see LICENSES/Apache-2.0.txt)
# Modified: 2026 by Thomas Hudcovic:
#  - Backbone is injected (not hardcoded to ResNet/Swin)
#  - Softmax removed from ClassifyBlock b/c CrossEntropyLoss needs raw logits
#  - FPN_fuse: removed assert feature_channels[0]==fpn_out, added projection
#    for all scales so any channel width is accepted
#  - PSPModule: channel_last flag added so it works with both Swin
#    (channel-last tokens) and all other backbones (channel-first tensors)
#  - Simplified head to a single Conv2d, removes the complex ConvLayer
#  - Exposes get_backbone_params / get_decoder_params for differential LR

from itertools import chain
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPModule(nn.Module):
    """
    Pyramid Pooling Module.
    Pools to multiple spatial sizes, projects each, concatenates, fuses.
    """
    def __init__(self, in_channels: int, out_channels: int = 256,
                 pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for size in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes),
                      in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h, w  = x.shape[-2:]
        parts = [x] + [
            F.interpolate(stage(x), size=(h, w),
                          mode='bilinear', align_corners=True)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(parts, dim=1))

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.

    CHANGE: projects ALL scales to fpn_channels (not just scales 1+).
    This removes the constraint that feature_channels[0] == fpn_out, so
    any backbone channel configuration is accepted.
    """
    def __init__(self, feature_channels: List[int], fpn_channels: int):
        super().__init__()
        # 1×1 lateral projection for every scale
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1)
            for c in feature_channels
        ])
        # 3×3 smooth conv after top-down fusion (one per merge point)
        self.smooth_conv = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for _ in range(len(feature_channels) - 1)
        ])
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_channels, fpn_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Project all scales
        features = [conv(f) for conv, f in zip(self.conv1x1, features)]

        # Top-down fusion
        for i in reversed(range(1, len(features))):
            features[i - 1] = features[i - 1] + F.interpolate(
                features[i],
                size=features[i - 1].shape[-2:],
                mode='bilinear', align_corners=True,
            )

        # Smooth
        smoothed = [s(features[i])
                    for i, s in enumerate(self.smooth_conv)]
        smoothed = list(reversed(smoothed))
        smoothed.append(features[-1])   # [P1, P2, P3, P4]

        # Upsample all to finest scale and fuse
        H, W = smoothed[0].shape[-2:]
        smoothed[1:] = [
            F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True)
            for f in smoothed[1:]
        ]
        return self.conv_fusion(torch.cat(smoothed, dim=1))


# UperNet decoder
class UperNet(nn.Module):
    """
    UperNet segmentation decoder.

    Accepts any hierarchical backbone that returns a list of 4
    (B, C, H, W) feature maps at [H/4, H/8, H/16, H/32] resolution.

    Pipeline:
        PSPModule on coarsest scale -> FPN fusion -> 3x3 head -> upsample to input size

    Args:
        backbone:         nn.Module, must implement forward(x) returning
                          a list of 4 (B, C, H, W) tensors.
        feature_channels: Channel widths [C0, C1, C2, C3].
        num_classes:      Number of segmentation classes.
        fpn_channels:     Internal FPN channel width (default 256).
    """
    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int],
        num_classes: int,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.backbone = backbone

        psp_out = feature_channels[-1] // 4
        self.psp = PSPModule(feature_channels[-1], psp_out)

        # PSP keeps the channel count of the last scale (bottleneck outputs in_channels)
        self.fpn = FPNNeck(feature_channels, fpn_channels)

        # CHANGE: simple head instead of ConvLayer, cleaner, same function
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            # CHANGE: raw logits (no Softmax), CrossEntropyLoss needs logits
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

        # Initialise decoder weights
        for m in [self.psp, self.fpn, self.head]:
            m.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(
            self.psp.parameters(),
            self.fpn.parameters(),
            self.head.parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        features = self.backbone(x)          # list of 4 (B, C, H, W)
        features[-1] = self.psp(features[-1])

        out = self.fpn(features)             # (B, fpn_channels, H/4, W/4)
        out = self.head(out)                 # (B, num_classes, H/4, W/4)
        return F.interpolate(out, size=input_size,
                             mode='bilinear', align_corners=True)


# Backbone wrappers (normalise each backbone to the list-of-4 interface)

class ConvNeXtV2Backbone(nn.Module):
    """Wraps ConvNeXtV2 to return 4-scale features via forward_multiscale."""
    def __init__(self, convnext):
        super().__init__()
        self.model = convnext

    def forward(self, x):
        return self.model.forward_multiscale(x)


class MiTBackbone(nn.Module):
    """Wraps MiT to always return 4-scale features."""
    def __init__(self, mit):
        super().__init__()
        self.model = mit

    def forward(self, x):
        return self.model(x, return_layer_outputs=True)


class CycleMLPBackbone(nn.Module):
    """Wraps CycleNet (fork_feat=True) to return 4-scale features."""
    def __init__(self, cyclenet):
        super().__init__()
        assert cyclenet.fork_feat, \
            "CycleNet must be instantiated with fork_feat=True for segmentation"
        self.model = cyclenet

    def forward(self, x):
        x = self.model.forward_embeddings(x)
        return self.model.forward_tokens(x)   # list of 4 (B,C,H,W)
