"""
Custom loss classes referenced by loss YAML configs via _target_.

Loss YAMLs:
  configs/loss/cross_entropy_classify.yaml:    torch.nn.CrossEntropyLoss
  configs/loss/cross_entropy_segment.yaml:     torch.nn.CrossEntropyLoss
  configs/loss/dice_bce.yaml:                  training.losses.DiceBCELoss
  ... etc.

torch.nn.CrossEntropyLoss is referenced directly so no wrapper is needed.
DiceBCELoss lives here because it is a custom class with no upstream home.
"""

import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
    0.5 * CrossEntropy + 0.5 * smp.DiceLoss for binary segmentation.
    Used for ISIC 2018 (small dataset, binary, class imbalance).

    Args:
        ignore_index: Pixel value excluded from loss (default 255).
        smooth:       Dice smoothing constant (default 1e-4).
    """
    def __init__(self, ignore_index: int = 255, smooth: float = 1e-4):
        super().__init__()
        from segmentation_models_pytorch.losses import DiceLoss
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index,
                             smooth=smooth)
        self.bce  = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)
