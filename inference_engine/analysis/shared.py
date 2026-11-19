"""Loaders and parsers shared by the analysis scripts."""

import argparse
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from projections.common.layer_decomposition import LayerDecomposition


# Keys under which trainers have stored the state dict, tried in order.
STATE_DICT_KEYS = ("model_state", "state_dict", "model", "model_state_dict")


class CkptMode(Enum):
    STRICT = "strict"
    MATCH_SHAPES = "match_shapes"  # drop params whose shape differs, e.g. a num_classes head


def load_decompositions(path: str) -> dict[str, LayerDecomposition]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {k: LayerDecomposition.from_dict(v) for k, v in raw.items()}


def load_projections(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["projections"]


def load_points_3d(path: str) -> dict[str, np.ndarray]:
    return {
        k: v["points_3d"].cpu().float().numpy()
        for k, v in load_projections(path).items()
    }


def _match_shapes(model: nn.Module, state: dict) -> dict:
    model_sd = model.state_dict()
    return {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}


def load_checkpoint(model: nn.Module, path: str, device: str, mode: CkptMode) -> None:
    print("Loading checkpoint...")
    print(f"Checkpoint mode: {str(mode)}")
    print(f"Checkpoint path: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Unwrap the first known container key, else treat ckpt as a raw state dict
    state = ckpt
    for key in STATE_DICT_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break

    if mode == CkptMode.STRICT:
        model.load_state_dict(state)
        return

    matched = _match_shapes(model, state)

    if not matched: #Empty dicts evaluate to False
        raise ValueError(f"No compatible model tensors found in checkpoint: {path}")

    skipped = state.keys() - matched.keys()
    if skipped:
        print(f"Skipped {len(skipped)} checkpoint keys:")
        for key in sorted(skipped):
            print(f"  {key}")

    result = model.load_state_dict(matched, strict=False)

    if result.missing_keys:
        print(f"Missing {len(result.missing_keys)} model keys:")
        for key in result.missing_keys:
            print(f"  {key}")


def parse_run(s: str, fields: tuple[str, ...], n_optional: int = 0) -> list:
    """Split a comma-separated --runs entry into len(fields) parts.

    The last n_optional fields may be omitted and come back as None, e.g.
    parse_run("mit,seg,pca,p.pt", ("model", "task", "method", "proj", "decomp"), 1)
    -> ["mit", "seg", "pca", "p.pt", None]
    The last field keeps any further commas.
    """
    n_max = len(fields)
    parts = s.split(",", n_max - 1)

    if not n_max - n_optional <= len(parts) <= n_max:
        raise argparse.ArgumentTypeError(
            f"Each --runs entry must be '{','.join(fields)}', got: {s!r}"
        )

    return parts + [None] * (n_max - len(parts))


def primary_metric(task: str, metrics: dict[str, float]) -> float:
    """
    Pick the headline metric from the Evaluator's metrics dict.

    trainer.build_metrics() names them exactly:
        classification -> "accuracy" (+ "top5", optional "macro_f1")
        segmentation   -> "miou"     (+ "pixel_acc")
    We also tolerate MetricCollection prefixing (e.g. "val_accuracy") by
    substring-matching as a fallback.
    """
    want = "accuracy" if task == "classification" else "miou"

    # Exact match first
    if want in metrics:
        return metrics[want]

    # Substring match (handles any collection prefix/suffix)
    for k, v in metrics.items():
        if want in k.lower():
            return v

    # Fallback: first metric value, with a loud warning
    k0 = next(iter(metrics))
    print(f"WARNING: '{want}' not found among {list(metrics)}, using '{k0}'.")
    return metrics[k0]
