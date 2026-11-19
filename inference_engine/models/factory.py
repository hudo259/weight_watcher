"""
Single entry point: build_model(cfg) -> nn.Module.

Handles all 6 combinations:
  backbone:  convnextv2 | mit | cyclemlp
  task:      classification | segmentation
"""

import torch.nn as nn

from models.convnextv2 import DIMS as CNX_DIMS
from models.convnextv2 import convnextv2_tiny, convnextv2_base, convnextv2_large
from models.mit import DIMS as MIT_DIMS, CONSTRUCTORS as MIT_CONSTRUCTORS
from models.cycle_mlp import DIMS as CML_DIMS, CONSTRUCTORS as CML_CONSTRUCTORS
from models.classifiers import (
    ConvNeXtV2Classifier, MiTClassifier, CycleNetClassifier,
)
from models.upernet import (
    UperNet,
    ConvNeXtV2Backbone, MiTBackbone, CycleMLPBackbone,
)

# ConvNeXtV2 variant constructors
CNX_CONSTRUCTORS = {
    "tiny":  convnextv2_tiny,
    "base":  convnextv2_base,
    "large": convnextv2_large,
}


def build_model(backbone: str, variant: str, task: str,
                n_classes: int, fpn_ch: int = 256) -> nn.Module:
    """
    Build and return a model.

    Args:
        backbone:  "convnextv2" | "mit" | "cyclemlp"
        variant:   size string, e.g. "base", "b2"
        task:      "classification" | "segmentation"
        n_classes: number of output classes
        fpn_ch:    UperNet FPN channel width (segmentation only)
    """
    backbone = backbone.lower()
    variant  = variant.lower()
    task     = task.lower()

    if task == "classification":
        return _build_classifier(backbone, variant, n_classes)
    elif task == "segmentation":
        return _build_segmenter(backbone, variant, n_classes, fpn_ch)
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _build_classifier(backbone: str, variant: str, num_classes: int) -> nn.Module:
    if backbone == "convnextv2":
        if variant not in CNX_CONSTRUCTORS:
            raise ValueError(f"ConvNeXtV2 variant {variant!r} not in {list(CNX_CONSTRUCTORS)}")
        model = CNX_CONSTRUCTORS[variant](num_classes=num_classes)
        return ConvNeXtV2Classifier(model)

    elif backbone == "mit":
        if variant not in MIT_CONSTRUCTORS:
            raise ValueError(f"MiT variant {variant!r} not in {list(MIT_CONSTRUCTORS)}")
        model     = MIT_CONSTRUCTORS[variant]()
        embed_dim = MIT_DIMS[variant][-1]   # last stage channels
        return MiTClassifier(model, num_classes, embed_dim)

    elif backbone == "cyclemlp":
        if variant not in CML_CONSTRUCTORS:
            raise ValueError(f"CycleMLP variant {variant!r} not in {list(CML_CONSTRUCTORS)}")
        model = CML_CONSTRUCTORS[variant](num_classes=num_classes, fork_feat=False)
        return CycleNetClassifier(model)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")


def _build_segmenter(backbone: str, variant: str,
                     num_classes: int, fpn_channels: int) -> nn.Module:
    if backbone == "convnextv2":
        if variant not in CNX_CONSTRUCTORS:
            raise ValueError(f"ConvNeXtV2 variant {variant!r} not in {list(CNX_CONSTRUCTORS)}")
        model    = CNX_CONSTRUCTORS[variant]()    # no classification head needed
        dims     = CNX_DIMS[variant]
        wrapped  = ConvNeXtV2Backbone(model)
        return UperNet(wrapped, list(dims), num_classes, fpn_channels)

    elif backbone == "mit":
        if variant not in MIT_CONSTRUCTORS:
            raise ValueError(f"MiT variant {variant!r} not in {list(MIT_CONSTRUCTORS)}")
        model   = MIT_CONSTRUCTORS[variant]()
        dims    = MIT_DIMS[variant]
        wrapped = MiTBackbone(model)
        return UperNet(wrapped, list(dims), num_classes, fpn_channels)

    elif backbone == "cyclemlp":
        if variant not in CML_CONSTRUCTORS:
            raise ValueError(f"CycleMLP variant {variant!r} not in {list(CML_CONSTRUCTORS)}")
        model   = CML_CONSTRUCTORS[variant](fork_feat=True)
        dims    = CML_DIMS[variant]
        wrapped = CycleMLPBackbone(model)
        return UperNet(wrapped, list(dims), num_classes, fpn_channels)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")
