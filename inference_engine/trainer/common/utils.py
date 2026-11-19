"""
Minimal _target_ instantiation

Handles the same _target_ convention used in the YAML configs:

    _target_: albumentations.Normalize
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

Works with OmegaConf DictConfig or plain dicts.
Nested _target_ objects are resolved recursively.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any
import torch

from omegaconf import DictConfig, OmegaConf

def _import(target: str) -> Any:
    """Import a dotted class path, e.g. 'albumentations.Normalize'."""
    module_path, _, class_name = target.rpartition(".")
    module = import_module(module_path)
    return getattr(module, class_name)


def instantiate(cfg: Any, **kwargs) -> Any:
    """
    Instantiate an object from a config with a _target_ key.

    Args:
        cfg:    OmegaConf DictConfig or plain dict with '_target_' key.
        kwargs: Additional keyword arguments merged into cfg (overrides).

    Returns:
        The constructed object.

    Example config:
        _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std:  [0.229, 0.224, 0.225]

    Lists are handled so albumentations.Compose works:
        _target_: albumentations.Compose
        transforms:
          - _target_: albumentations.HorizontalFlip
            p: 0.5
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg, dict):
        return cfg

    cfg = dict(cfg)
    cfg.update(kwargs)

    target = cfg.pop("_target_")
    cls    = _import(target)

    # Recursively instantiate any nested _target_ values
    resolved = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and "_target_" in v:
            resolved[k] = instantiate(v)
        elif isinstance(v, list):
            resolved[k] = [
                instantiate(item) if isinstance(item, dict) and "_target_" in item
                else item
                for item in v
            ]
        else:
            resolved[k] = v

    return cls(**resolved)

def load_pretrained(model: torch.nn.Module, path: str) -> None:
    """
    Load model weights from a pretrained checkpoint for fine-tuning.

    Unlike --resume, this:
      - loads model weights only (no optimizer/scheduler/epoch state)
      - uses strict=False so mismatched keys are skipped silently
      - works for both same-task (wrong head size) and cross-task
        (classification -> segmentation) transfer

    The checkpoint can be:
      - a training checkpoint saved by CheckpointManager
        (dict with 'model_state' key)
      - a raw state_dict
      - a timm/torchvision style checkpoint
        (dict with 'model' or 'state_dict' key)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # Unwrap common checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state")   # our CheckpointManager format
            or ckpt.get("model")      # timm format
            or ckpt.get("state_dict") # torchvision format
            or ckpt                   # raw state_dict
        )
    else:
        state_dict = ckpt


    # Remove keys that don't match in shape, typically the classification head.
    # These will be randomly initialized in the new model, which is correct
    # behaviour when fine-tuning on a different number of classes.
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Only report keys that were skipped due to shape mismatch, not just absence
    shape_mismatch = [k for k in state_dict if k in model_state
                      and state_dict[k].shape != model_state[k].shape]

    # Summarise what was loaded
    total     = len(state_dict)
    n_missing = len(missing)
    n_unexpected = len(unexpected)
    n_loaded  = total - n_unexpected

    print(f"[Pretrained] Loaded {n_loaded}/{total} keys from {path}")
    if n_missing:
        print(f"[Pretrained] {n_missing} missing keys (e.g. new head will train from scratch): {missing[:3]} {'...' if n_missing > 3 else ''}")
    if n_unexpected:
        print(f"[Pretrained] {n_unexpected} unexpected keys (ignored): {unexpected[:3]}{'...' if n_unexpected > 3 else ''}")
    if shape_mismatch:
        print(f"[Pretrained] Skipped (shape mismatch, reinitialised): {shape_mismatch}")
