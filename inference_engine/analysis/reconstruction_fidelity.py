"""
Measures how faithfully each decomposition method can reconstruct the
original model weights, and how much accuracy drops after reconstruction.

Two measurements per (model, task, method):
  1. Weight MSE: ||W_original - W_reconstructed||² per layer, averaged
     over all components.
  2. Accuracy drop: baseline_acc - reconstructed_acc, where the
     reconstructed model has ALL decomposed layers replaced by their
     inverse_transform output.

For PCA the reconstruction is lossless when residuals are stored (the
residuals are loaded from the projection file). For UMAP and FourierKPCA
the reconstruction is approximate.

Output
------
  results/reconstruction_fidelity/
      reconstruction_fidelity.csv (per-layer MSE per method)
      reconstruction_fidelity_summary.csv (per-method aggregate + accuracy)
      barplot_mse.pdf
      barplot_accuracy_drop.pdf
"""

import argparse
import copy
import os
from math import ceil
from multiprocessing import cpu_count
import sys
from pathlib import Path

# This script lives in inference_engine/analysis/ but imports from
# inference_engine/ (main, models, inference, projections). Add the parent
# directory to sys.path so those imports resolve regardless of the current
# working directory the script is launched from.
_ANALYSIS_DIR = Path(__file__).resolve().parent            # inference_engine/analysis
_ENGINE = _ANALYSIS_DIR.parent                             # inference_engine
for _p in (_ENGINE, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.factory import build_model
from inference.weight_reconstruction import WeightReconstructor
from trainer.common.datasets import build_loaders
from trainer.common.trainer import build_metrics, unpack_batch
from shared import CkptMode, load_checkpoint, load_decompositions, load_projections, parse_run

RUN_FIELDS = ("method", "decomposers_path", "projection_path")


# Inference helpers
def build_val_loader(args) -> DataLoader:
    import torchvision.transforms.v2 as v2

    val_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize((args.image_size, args.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ns = argparse.Namespace(
        task=args.task,
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        ignore_index=getattr(args, "ignore_index", -100),
    )
    _, val_loader = build_loaders(ns, train_tf=val_tf, val_tf=val_tf)
    return val_loader


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    dataset_name: str,
    num_classes: int,
    ignore_index: int,
    device: str,
) -> dict[str, float]:
    """Run one full eval pass, return dict of metric_name -> value."""
    dev = torch.device(device)
    model.eval()
    metrics = build_metrics(
        task=task,
        num_classes=num_classes,
        dataset_name=dataset_name,
        ignore_index=ignore_index,
        device=str(dev),
    )
    metrics.reset()
    for batch in loader:
        images, targets = unpack_batch(batch, task, dev)
        preds = model(images)
        metrics.update(preds, targets)
    return {k: v.item() for k, v in metrics.compute().items()}


# Weight MSE computation
def compute_weight_mse(
    model_original: nn.Module,
    reconstructor: WeightReconstructor,
    device: str,
) -> dict[str, float]:
    """
    For each decomposed layer, compare original weight vs reconstructed weight.
    Returns layer_key -> MSE (mean over all elements).
    """
    dev = torch.device(device)
    orig_sd = model_original.state_dict()
    mse_per_layer = {}

    for key in reconstructor.layer_keys:
        orig_w = orig_sd[key].float().to(dev)
        recon_w = reconstructor.reconstruct_layer(key).to(dev)
        # Both should be the same shape
        if orig_w.shape != recon_w.shape:
            print(f"  WARNING: shape mismatch for {key}: "
                  f"{orig_w.shape} vs {recon_w.shape}")
            mse_per_layer[key] = float("nan")
            continue
        mse = ((orig_w - recon_w) ** 2).mean().item()
        mse_per_layer[key] = mse

    return mse_per_layer


# Main analysis loop
def analyze_run(
    method: str,
    decomposers_path: str,
    projection_path: str,
    model_original: nn.Module,
    baseline_metrics: dict[str, float],
    loader: DataLoader,
    args,
) -> tuple[list[dict], dict]:
    """
    Returns:
        layer_rows: list of per-layer MSE dicts
        summary_row: dict with method-level accuracy drop
    """
    print(f"\n  Method: {method}")
    decompositions = load_decompositions(decomposers_path)
    projections    = load_projections(projection_path)

    # Build reconstructor (makes a copy of coordinates — does not modify model)
    reconstructor = WeightReconstructor(
        decompositions=decompositions,
        projections=projections,
        device=args.device,
    )

    # Step 1: Weight MSE per layer
    mse_per_layer = compute_weight_mse(model_original, reconstructor, args.device)
    layer_rows = []
    for key, mse in mse_per_layer.items():
        layer_rows.append({
            "method":     method,
            "layer_key":  key,
            "layer_type": decompositions[key].layer_type,
            "n_units":    decompositions[key].num_components,
            "weight_mse": mse,
        })

    mean_mse = np.nanmean([r["weight_mse"] for r in layer_rows])
    print(f"    Mean weight MSE across layers: {mean_mse:.6e}")

    # Step 2: Accuracy after reconstruction
    # Deep-copy the model so we don't permanently alter it
    model_recon = copy.deepcopy(model_original)
    model_recon.to(args.device)
    reconstructor_copy = WeightReconstructor(
        decompositions=decompositions,
        projections=projections,
        device=args.device,
    )
    reconstructor_copy.reconstruct_and_load(model_recon)

    recon_metrics = run_inference(
        model=model_recon,
        loader=loader,
        task=args.task,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        ignore_index=getattr(args, "ignore_index", -100),
        device=args.device,
    )

    # Compute accuracy drops for all reported metrics
    drops = {
        f"drop_{k}": baseline_metrics.get(k, float("nan")) - v
        for k, v in recon_metrics.items()
    }

    print(f"    Baseline: {baseline_metrics}")
    print(f"    Recon:    {recon_metrics}")
    print(f"    Drops:    {drops}")

    summary_row = {
        "method":      method,
        "mean_weight_mse": mean_mse,
        **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
        **{f"recon_{k}":    v for k, v in recon_metrics.items()},
        **drops,
    }
    return layer_rows, summary_row


# Plotting
def plot_mse(layer_df: pd.DataFrame, out_dir: str) -> None:
    methods = layer_df["method"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(methods)
    x = np.arange(layer_df["layer_key"].nunique())
    layer_keys = layer_df["layer_key"].unique()

    for i, method in enumerate(methods):
        sub = layer_df[layer_df["method"] == method].set_index("layer_key")
        vals = [sub.loc[k, "weight_mse"] if k in sub.index else float("nan")
                for k in layer_keys]
        ax.bar(x + i * width, vals, width=width, label=method)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(
        [k.split(".")[-2] if "." in k else k for k in layer_keys],
        rotation=90, fontsize=6
    )
    ax.set_ylabel("Weight MSE")
    ax.set_title("Weight reconstruction MSE per layer")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "barplot_mse.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_drop(summary_df: pd.DataFrame, out_dir: str) -> None:
    drop_cols = [c for c in summary_df.columns if c.startswith("drop_")]
    if not drop_cols:
        return

    fig, axes = plt.subplots(1, len(drop_cols), figsize=(5 * len(drop_cols), 4),
                             squeeze=False)
    for ax, col in zip(axes[0], drop_cols):
        ax.bar(summary_df["method"], summary_df[col])
        ax.set_title(col.replace("drop_", "Δ "))
        ax.set_ylabel("Accuracy drop (baseline − reconstructed)")
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "barplot_accuracy_drop.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def main():
    ap = argparse.ArgumentParser(description="Reconstruction fidelity analysis")

    # Model
    ap.add_argument("--backbone",    required=True,
                    choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant",     required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--task",        required=True,
                    choices=["classification", "segmentation"])

    # Data
    ap.add_argument("--dataset",     required=True,
                    choices=["imagenet100", "isic2019", "voc2012", "isic2018"])
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--ignore_index", type=int, default=-100)

    # Checkpoint
    ap.add_argument("--checkpoint",  required=True)

    # Decomposition runs
    ap.add_argument(
        "--runs", nargs="+", required=True,
        metavar="METHOD,DECOMPOSERS_PATH,PROJECTION_PATH",
    )

    # Runtime
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--image_size",  type=int, default=224)
    ap.add_argument("--out_dir",     default="./results/reconstruction_fidelity")
    args = ap.parse_args()

    if args.num_workers is None:
        args.num_workers = ceil(cpu_count() / 2)

    os.makedirs(args.out_dir, exist_ok=True)

    # Build model and load checkpoint once
    print("Building model and loading checkpoint...")
    model = build_model(
        backbone=args.backbone,
        variant=args.variant,
        task=args.task,
        n_classes=args.num_classes,
    ).to(args.device)
    model.eval()
    load_checkpoint(model, args.checkpoint, args.device, CkptMode.MATCH_SHAPES)

    # Build val loader once
    print("Building val loader...")
    loader = build_val_loader(args)

    # Baseline accuracy with original weights
    print("Running baseline inference...")
    baseline_metrics = run_inference(
        model=model,
        loader=loader,
        task=args.task,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        device=args.device,
    )
    print(f"Baseline metrics: {baseline_metrics}")

    all_layer_rows = []
    summary_rows   = []

    for run_str in args.runs:
        method, decomposers_path, projection_path = parse_run(run_str, RUN_FIELDS)
        layer_rows, summary_row = analyze_run(
            method=method,
            decomposers_path=decomposers_path,
            projection_path=projection_path,
            model_original=model,
            baseline_metrics=baseline_metrics,
            loader=loader,
            args=args,
        )
        all_layer_rows.extend(layer_rows)
        summary_rows.append(summary_row)

    layer_df   = pd.DataFrame(all_layer_rows)
    summary_df = pd.DataFrame(summary_rows)

    layer_path   = os.path.join(args.out_dir, "reconstruction_fidelity.csv")
    summary_path = os.path.join(args.out_dir, "reconstruction_fidelity_summary.csv")
    layer_df.to_csv(layer_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {layer_path}")
    print(f"Saved: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    plot_mse(layer_df, args.out_dir)
    plot_accuracy_drop(summary_df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
