"""
Thin classifier wrappers for each backbone.

Each wraps the backbone in a nn.Module that:
  1. Calls the backbone's forward pass
  2. Applies global average pooling to get a (B, C) vector
  3. Applies a linear classification head

All expose get_backbone_params() / get_decoder_params() for differential LR
in the Trainer optimizer.

ConvNeXtV2 and CycleNet already have built-in classification heads but we
wrap them anyway for a consistent interface and differential LR support.
"""

import torch
import torch.nn as nn
from itertools import chain

class ConvNeXtV2Classifier(nn.Module):
    """
    ConvNeXtV2 classifier.
    Uses the built-in head, wraps for differential LR interface.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone   # ConvNeXtV2 instance

    def get_backbone_params(self):
        return (p for n, p in self.backbone.named_parameters()
                if not n.startswith('head'))

    def get_decoder_params(self):
        return self.backbone.head.parameters()

    def forward(self, x):
        return self.backbone(x)

class MiTClassifier(nn.Module):
    """
    MiT encoder + GAP + linear head for classification.
    MiT has no built-in classification head, we add one.
    """
    def __init__(self, backbone, num_classes: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone   # MiT instance
        self.norm     = nn.LayerNorm(embed_dim)
        self.head     = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.norm.parameters(), self.head.parameters())

    def forward(self, x):
        feat = self.backbone(x)          # (B, C, H, W) last stage output
        feat = feat.mean([-2, -1])       # GAP -> (B, C)
        feat = self.norm(feat)
        return self.head(feat)

class CycleNetClassifier(nn.Module):
    """
    CycleNet classifier (fork_feat=False).
    Uses the built-in head, wraps for differential LR interface.
    """
    def __init__(self, backbone):
        super().__init__()
        assert not backbone.fork_feat, \
            "CycleNet must be instantiated with fork_feat=False for classification"
        self.backbone = backbone

    def get_backbone_params(self):
        return (p for n, p in self.backbone.named_parameters()
                if not n.startswith('head'))

    def get_decoder_params(self):
        return self.backbone.head.parameters()

    def forward(self, x):
        return self.backbone(x)
