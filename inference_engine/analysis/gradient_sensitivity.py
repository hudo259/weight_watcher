"""
Fast alternative to sensitivity.py using gradient-based sensitivity
instead of perturbation-based sensitivity.

Mathematical argument
------------------
The PCA inverse transform is:
    w_c = z_c @ V_k.T + mu + residuals + init_weights

Since V_k is fixed and linear, by the chain rule:
    d_loss/d_z_c = d_loss/d_w_c @ V_k         [k-vector per unit]

The sensitivity score for unit c is:
    s_c = ||d_loss/d_z_c||_2 = ||d_loss/d_w_c @ V_k||_2

This measures how much a small perturbation in z_c changes the loss. 
Identical in spirit to the perturbation approach but computed exactly
in one pass rather than approximated via N random directions.

Relationship to perturbation sensitivity
-----------------------------------------
For a random unit direction d in R^k:
    sensitivity_perturbation ≈ epsilon * |d_loss/d_z_c . d|

Averaged over many random directions:
    E[|d_loss/d_z_c . d|] = (1 / sqrt(k)) * ||d_loss/d_z_c||_2

So gradient sensitivity is proportional to perturbation sensitivity
in expectation — they rank units in the same order.

Output
------
    sensitivity.csv              - per-unit gradient sensitivity scores
    sensitivity_summary.csv      - mean +/- std per layer
    sensitivity_heatmap.pdf      - heatmap [layer x unit]
    sensitivity_by_layer_type.pdf
"""

import argparse
import os
from math import ceil
from multiprocessing import cpu_count
import sys
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as v2
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydoc import locate

OmegaConf.register_new_resolver("locate", locate, replace=True)

# This script lives in inference_engine/analysis/ but imports from
# inference_engine/ (main, models, inference, projections). Add the parent
# directory to sys.path so those imports resolve regardless of the current
# working directory the script is launched from.
_ANALYSIS_DIR = Path(__file__).resolve().parent            # inference_engine/analysis
_ENGINE = _ANALYSIS_DIR.parent                             # inference_engine
for _p in (_ENGINE, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.factory import build_model
from projections.common.layer_decomposition import LayerDecomposition
from trainer.common.datasets import build_loaders
from shared import CkptMode, load_checkpoint, load_decompositions, load_projections


def build_anchor_loader(args) -> DataLoader:

    if args.val_transforms:
        cfg    = OmegaConf.load(args.val_transforms)
        val_tf = v2.Compose([instantiate(t) for t in cfg.transforms])
    else:
        val_tf = v2.Compose([
            v2.ToImage(),
            v2.Resize((args.image_size, args.image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    args.task = args.subcommand if hasattr(args, "subcommand") else args.task
    if args.num_workers is None:
        args.num_workers = ceil(cpu_count() / 2)

    _, val_loader = build_loaders(args, train_tf=val_tf, val_tf=val_tf)

    n = len(val_loader.dataset)
    anchor_size = min(args.anchor_size, n)
    if anchor_size < n:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(n, size=anchor_size, replace=False).tolist()
        subset  = Subset(val_loader.dataset, indices)
        val_loader = DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        print(f"  Anchor set: {anchor_size}/{n} images sampled (seed={args.seed})")
    return val_loader


def compute_gradient_sensitivity(
    model:          nn.Module,
    decompositions: dict[str, LayerDecomposition],
    projections:    dict[str, dict],
    loader:         DataLoader,
    loss_fn:        nn.Module,
    task:           str,
    device:         str,
) -> list[dict]:
    """
    Compute gradient-based sensitivity for all units in all layers.

    Steps:
    1. Register hooks to accumulate squared gradients w.r.t. each
       layer's weight parameter across all anchor batches.
    2. Run one forward+backward pass per batch.
    3. For each layer, project the weight gradient through V_k to
       get the gradient in z-space: g_z = g_w @ V_k  [C, k]
    4. Sensitivity = ||g_z_c||_2 per unit c.

    Returns list of row dicts compatible with sensitivity.py's CSV format.
    """
    model.eval()
    model.to(device)

    layer_keys = list(decompositions.keys())

    # Build name -> parameter mapping for all decomposed layers
    param_map: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        # Match layer key to parameter name (layer_key is the param name)
        if name in decompositions:
            param_map[name] = param
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    print(f"  Gradient sensitivity: {len(param_map)} layers with grad enabled")
    print(f"  Running {len(loader)} batches on anchor set...")

    # Accumulate sum of squared gradients per parameter across batches
    grad_accum: dict[str, torch.Tensor] = {
        key: torch.zeros_like(param_map[key], device=device)
        for key in param_map
    }
    n_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        images  = images.to(device)
        targets = targets.to(device)

        model.zero_grad()
        logits = model(images)

        # Loss from the YAML-configured criterion (matches training objective)
        loss = loss_fn(logits, targets)

        loss.backward()

        # Accumulate squared gradients
        for key, param in param_map.items():
            if param.grad is not None:
                grad_accum[key] += param.grad.detach()

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"    Batch {n_batches}/{len(loader)}", flush=True)

    print(f"  Processed {n_batches} batches. Computing per-unit sensitivities...")

    # Re-enable all grads
    for param in model.parameters():
        param.requires_grad_(True)

    rows = []

    for depth_idx, layer_key in enumerate(layer_keys):
        decomp = decompositions[layer_key]
        C      = decomp.num_components

        if layer_key not in grad_accum:
            # Layer not in param_map (no grad) — zero sensitivity
            for unit_idx in range(C):
                rows.append({
                    "layer_key":          layer_key,
                    "layer_type":         decomp.layer_type,
                    "layer_depth":        depth_idx,
                    "unit_idx":           unit_idx,
                    "sensitivity":        0.0,
                    "ev_scale":           1.0,
                    "effective_epsilon":  0.0,
                    "method":             "gradient",
                    "model":              "",
                    "task":               task,
                })
            continue

        # Mean raw gradient per weight element (averaged over batches)
        g_raw = grad_accum[layer_key] / max(n_batches, 1)  # same shape as param

        # Flatten to [C, d] matching the decomposition convention
        t  = decomp.layer_type
        os = decomp.original_shape

        if t in ("conv2d", "conv1d", "depthwise_conv"):
            # g_sq shape: [C, *os] — flatten per-unit
            g_flat = g_raw.view(C, -1)                     # [C, d]
        elif t in ("linear", "attention_query", "attention_key",
                   "attention_value", "attention_output",
                   "attention_fused_qkv"):
            if len(g_raw.shape) == 4:
                g_flat = g_raw.squeeze(-1).squeeze(-1)     # [C, d]
            else:
                g_flat = g_raw                             # [C, d]
        elif t in ("attention_mha_head", "attention_mha_head_fused_qkv"):
            num_heads, head_dim, in_dim = os
            g_flat = g_raw.view(C, -1)
        else:
            g_flat = g_raw.view(C, -1)

        g_flat = g_flat.float().cpu()

        # Project through V_k: g_z_c = g_w_c @ V_k  [C, k]
        # This gives the gradient in the 3D embedding space
        pca = decomp.decomposer
        V_k = pca._eigenvectors  # [d, k]
        if V_k is None:
            print(f"  [WARN] No eigenvectors for {layer_key}, skipping")
            continue

        V_k = V_k.float().cpu()    # [d, k]

        # g_z [C, k]: gradient of loss w.r.t. z_c for each unit
        g_z = g_flat @ V_k        # [C, k]

        # Sensitivity = L2 norm of gradient in z-space
        sensitivity_scores = g_z.norm(dim=1).numpy()  # [C]

        # EV scale (same as perturbation script, for consistency)
        ev = getattr(pca, "explained_variance", None)
        if ev is not None:
            mean_ev  = float(ev.mean().item())
            ev_scale = float(np.sqrt(max(mean_ev, 1e-6)))
        else:
            mean_ev  = float("nan")
            ev_scale = 1.0

        for unit_idx in range(C):
            rows.append({
                "layer_key":         layer_key,
                "layer_type":        decomp.layer_type,
                "layer_depth":       depth_idx,
                "unit_idx":          unit_idx,
                "sensitivity":       float(sensitivity_scores[unit_idx]),
                "ev_scale":          ev_scale,
                "effective_epsilon": 0.0,   # N/A for gradient method
                "method":            "gradient",
                "model":             "",
                "task":              task,
            })

        print(
            f"  Layer [{depth_idx:3d}/{len(layer_keys)-1}] "
            f"{layer_key[:55]:<55}  C={C}  "
            f"mean_sens={sensitivity_scores.mean():.5f}  "
            f"max_sens={sensitivity_scores.max():.5f}"
        )

    return rows


def plot_by_layer_type(df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for lt, grp in df.groupby("layer_type"):
        ax.boxplot(grp["sensitivity"].values,
                   positions=[list(df["layer_type"].unique()).index(lt)],
                   labels=[lt], patch_artist=True)
    ax.set_ylabel("||∂L/∂z_c||")
    ax.set_title("Gradient sensitivity by layer type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_by_layer_type.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Gradient-based sensitivity analysis (fast alternative to perturbation)"
    )
    ap.add_argument("task", choices=["classification", "segmentation"])
    ap.add_argument("--backbone",    required=True,
                    choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant",     required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--dataset",     required=True,
                    choices=["imagenet100", "isic2019",
                             "voc2012", "voc2012_sbd", "isic2018"])
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--decomposers", required=True)
    ap.add_argument("--projection",  required=True)
    ap.add_argument("--val_transforms", default=None)
    ap.add_argument("--loss_config",    required=True,
                    help="Path to loss YAML config (same as train.py)")
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument("--anchor_size", type=int, default=200)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--image_size",  type=int, default=512)
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--seed",        type=int, default=108)
    ap.add_argument("--model_name",  default=None,
                    help="Model name for CSV column (default: backbone_variant)")
    ap.add_argument("--out_dir",     default="./results/sensitivity")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_name = args.model_name or f"{args.backbone}_{args.variant}"

    # Build model
    print("Building model...")
    model = build_model(
        backbone=args.backbone,
        variant=args.variant,
        task=args.task,
        n_classes=args.num_classes,
    ).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device, CkptMode.STRICT)

    # Load decompositions
    print("Loading decompositions...")
    decompositions = load_decompositions(args.decomposers)
    projections    = load_projections(args.projection)
    print(f"  {len(decompositions)} layers loaded")

    # Build anchor set
    print("Building anchor set...")
    args.subcommand = args.task
    loader = build_anchor_loader(args)

    # Build loss from YAML config via instantiate (same pattern as train.py)

    loss_cfg = OmegaConf.load(args.loss_config)
    loss_cfg.ignore_index = args.ignore_index
    loss_fn = instantiate(loss_cfg)

    # Class weights for ISIC2019 — computed from the dataset, same as training
    if args.dataset == "isic2019":
        base_ds = loader.dataset
        # Unwrap Subset to reach the dataset with compute_class_weights
        if hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        if hasattr(base_ds, "compute_class_weights"):
            class_weights = base_ds.compute_class_weights()
            loss_fn.register_buffer("weight", class_weights.to(args.device))
            print(f"[Loss]  ISIC2019 class weights: "
                  f"{[f'{w:.2f}' for w in class_weights.tolist()]}")

    loss_fn = loss_fn.to(args.device)
    print(f"[Loss]  Using {type(loss_fn).__name__} from {args.loss_config}")

    # Run gradient sensitivity
    print(f"\nRunning gradient sensitivity on {args.anchor_size} anchor images...")
    rows = compute_gradient_sensitivity(
        model=model,
        decompositions=decompositions,
        projections=projections,
        loader=loader,
        loss_fn=loss_fn,
        task=args.task,
        device=args.device,
    )

    # Add model/task columns
    for r in rows:
        r["model"] = model_name
        r["task"]  = f"{args.task}_{args.dataset}" 

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw CSV: {csv_path}")

    # Summary
    summary = (
        df.groupby(["model", "task", "layer_key", "layer_type", "layer_depth"])
        ["sensitivity"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "sensitivity_mean", "std": "sensitivity_std",
                         "min": "sensitivity_min",  "max": "sensitivity_max",
                         "count": "n_units"})
    )
    summary_path = os.path.join(args.out_dir, "sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    print("\nTop 10 most sensitive units:")
    top = df.nlargest(10, "sensitivity")[
        ["layer_key", "layer_type", "unit_idx", "sensitivity"]
    ]
    print(top.to_string(index=False))

    # Plots
    plot_by_layer_type(df, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
