"""
Measures how sensitive model behavior is to perturbations of each
semantic unit in the 3D decomposition space.

For each unit c in each decomposed layer L:
  1. Perturb its 3D coordinates: z'[c] = z[c] + epsilon * direction
     where direction is a random unit vector in R^k (typically R^3).
  2. Reconstruct the perturbed layer weights via inverse_transform.
  3. Load the perturbed weights into the model (all other layers unchanged).
  4. Run inference on the anchor set, measure KL(baseline || perturbed).
  5. sensitivity[c] = mean over N directions of perturbed_KL.

We use KL(baseline || perturbed) as the sensitivity score rather than
discrete task-metric drop, for three reasons:
  (a) KL is continuous and label-independent, giving lower-variance
      per-unit estimates from a small anchor set (~200 images);
  (b) KL captures behavioral change directly, matching the paper's
      hypothesis ("geometric position predicts functional sensitivity");
  (c) baseline KL = 0 by construction, so sensitivity is just the
      perturbed KL value (no subtraction needed).

For binary segmentation (1-channel sigmoid output, e.g. ISIC2018) we use
Bernoulli KL; for multi-class output we use categorical KL. The branch is
selected automatically based on output channel count.

NOTE: This script only perturbs ONE layer at a time and restores it before
moving to the next unit. The baseline logits are cached once at the start.

Output
------
  results/sensitivity/
      sensitivity.csv              - per-unit sensitivity scores (mean KL)
      sensitivity_summary.csv      - mean +/- std per layer
      sensitivity_heatmap.pdf      - heatmap [layer x unit], sorted by depth
      sensitivity_by_layer_type.pdf
"""

import argparse
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.factory import build_model
from inference.weight_reconstruction import WeightReconstructor
from trainer.common.datasets import build_loaders
from trainer.common.trainer import unpack_batch
from shared import CkptMode, load_checkpoint, load_decompositions, load_projections

from torchvision.transforms import v2
from omegaconf import OmegaConf
from pydoc import locate
from hydra.utils import instantiate


def _build_transforms(cfg):
    return v2.Compose([instantiate(t) for t in cfg.transforms])


def build_val_loader(args) -> DataLoader:
    OmegaConf.register_new_resolver("locate", locate, replace=True)
    val_cfg = OmegaConf.load(args.val_transforms)
    val_tf = _build_transforms(val_cfg)

    ns = argparse.Namespace(
        task=args.task,
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_index=getattr(args, "ignore_index", -100),
    )
    _, val_loader = build_loaders(ns, train_tf=val_tf, val_tf=val_tf)
    return val_loader


def build_anchor_loader(
    val_loader: DataLoader,
    anchor_size: int,
    seed: int,
) -> DataLoader:
    """
    Randomly subsample anchor_size images from the val dataset.
    Using a small anchor set (100-200 images) instead of the full val set
    reduces sensitivity runtime by ~65x with negligible impact on relative
    KL measurements between perturbed and baseline models.

    IMPORTANT: shuffle=False so that the cached baseline logits line up
    with subsequent perturbed-model passes by batch index.
    """
    from torch.utils.data import Subset
    dataset = val_loader.dataset
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if anchor_size >= n:
        print(f"  anchor_size={anchor_size} >= dataset size={n}, using full val set")
        return val_loader

    indices = rng.choice(n, size=anchor_size, replace=False).tolist()
    subset = Subset(dataset, indices)

    anchor_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=val_loader.batch_size,
        num_workers=val_loader.num_workers,
        shuffle=False,
        pin_memory=val_loader.pin_memory,
        persistent_workers=val_loader.num_workers > 0,
    )

    print(f"  Anchor set: {anchor_size}/{n} images sampled (seed={seed})")
    return anchor_loader


# KL divergence helpers
def kl_per_sample(
    logits_b: torch.Tensor,
    logits_p: torch.Tensor,
) -> torch.Tensor:
    """
    KL(baseline || perturbed), per sample (classification) or per pixel
    (segmentation). Auto-handles binary (1-channel) vs multi-class
    (>1 channel) outputs.

    Args:
        logits_b: baseline logits, shape [B, C] or [B, C, H, W]
        logits_p: perturbed logits, same shape

    Returns:
        kl: [B] for classification, [B, H, W] for segmentation.
    """
    if logits_p.shape[1] == 1:
        # Bernoulli KL per element. Compute log p(1), log p(0) directly
        # via logsigmoid to keep numerics tight on saturated logits.
        log_p1 = F.logsigmoid(logits_p)
        log_p0 = F.logsigmoid(-logits_p)
        log_b1 = F.logsigmoid(logits_b)
        log_b0 = F.logsigmoid(-logits_b)
        kl = log_b1.exp() * (log_b1 - log_p1) + log_b0.exp() * (log_b0 - log_p0)
        return kl.squeeze(1)
    else:
        # Categorical KL per sample/pixel, sum over class dim.
        log_p = F.log_softmax(logits_p, dim=1)
        log_b = F.log_softmax(logits_b, dim=1)
        return (log_b.exp() * (log_b - log_p)).sum(dim=1)


@torch.no_grad()
def cache_baseline_logits(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: str,
    fp16: bool = False,
) -> list[torch.Tensor]:
    """
    One-shot pass over the anchor set: store model output logits per batch
    on CPU. Used to compute KL(baseline || perturbed) on every subsequent
    perturbation pass without re-running the unperturbed model.

    NOTE: The loader MUST be deterministic (shuffle=False)
    Recommended: set fp16=True to halve cache memory. KL is robust to fp16 precision in this range.
    """
    dev = torch.device(device)
    model.eval()
    cache: list[torch.Tensor] = []
    for batch in loader:
        images, _ = unpack_batch(batch, task, dev)
        logits = model(images)

        if fp16:
            logits = logits.half()
        cache.append(logits.cpu())

    return cache


# Inference (KL against cached baseline)
@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    baseline_logits_cache: list[torch.Tensor],
    task: str,
    ignore_index: int,
    device: str,
) -> float:
    """
    Run one forward pass with the (currently perturbed) model and return
    the mean KL(baseline || perturbed).

    For segmentation, KL is averaged over valid pixels (ignore_index
    excluded). For classification, averaged over batch samples.

    Logits are clamped to [-30, 30] before KL to prevent single-sample
    blow-up from saturated outputs (e.g. confidently wrong predictions
    where the perturbed distribution assigns ~0 probability to a class
    the baseline assigns non-trivial mass).
    """
    dev = torch.device(device)
    model.eval()
    total_kl = 0.0
    total_n = 0
    for i, batch in enumerate(loader):
        images, targets = unpack_batch(batch, task, dev)
        logits_p = model(images).float().clamp(-30, 30)
        logits_b = baseline_logits_cache[i].to(dev).float().clamp(-30, 30)
        kl = kl_per_sample(logits_b, logits_p)  # [B] cls, [B,H,W] seg

        if task == "segmentation":
            mask = (targets != ignore_index)
            if mask.any():
                total_kl += kl[mask].sum().item()
                total_n += int(mask.sum().item())
        else:
            total_kl += kl.sum().item()
            total_n += kl.numel()

    return total_kl / max(total_n, 1)


# Sensitivity computation
def perturb_unit_and_measure(
    reconstructor: WeightReconstructor,
    model: nn.Module,
    layer_key: str,
    unit_idx: int,
    epsilon: float,
    n_directions: int,
    baseline_logits_cache: list[torch.Tensor],
    loader: DataLoader,
    task: str,
    ignore_index: int,
    device: str,
    rng: np.random.Generator,
    ev_scale: float = 1.0,
) -> float:
    """
    Perturb unit_idx in layer_key N times, measure mean KL(baseline || perturbed).
    Returns float sensitivity score (higher = more sensitive unit; baseline
    KL is exactly 0 so no subtraction is needed).

    ev_scale: multiply epsilon by this factor before applying perturbation.
    Set to sqrt(mean_ev) of the layer to normalise perturbation magnitude
    across layer types with different explained variance (EV). This ensures
    a given epsilon produces a comparable weight-space displacement
    regardless of how much variance the k-D subspace captures for that layer.
    """
    k = reconstructor._current_Z[layer_key].shape[1]
    original_z = reconstructor._current_Z[layer_key][unit_idx].clone()
    effective_epsilon = epsilon * ev_scale

    kl_values = []
    for _ in range(n_directions):
        # Random unit vector in k-dimensional decomposition space
        direction = torch.from_numpy(
            rng.standard_normal(k).astype(np.float32)
        )
        direction = direction / (direction.norm() + 1e-8)

        reconstructor._current_Z[layer_key][unit_idx] = (
            original_z + effective_epsilon * direction
        )

        with torch.no_grad():
            weight_tensor = reconstructor.reconstruct_layer(layer_key)
            model.state_dict()[layer_key].copy_(weight_tensor)

        perturbed_kl = run_inference(
            model=model,
            loader=loader,
            baseline_logits_cache=baseline_logits_cache,
            task=task,
            ignore_index=ignore_index,
            device=device,
        )
        kl_values.append(perturbed_kl)

    # Restore original coordinates and weights
    reconstructor._current_Z[layer_key][unit_idx] = original_z
    with torch.no_grad():
        weight_tensor = reconstructor.reconstruct_layer(layer_key)
        model.state_dict()[layer_key].copy_(weight_tensor)

    return float(np.mean(kl_values))


def run_sensitivity(
    model: nn.Module,
    reconstructor: WeightReconstructor,
    loader: DataLoader,
    baseline_logits_cache: list[torch.Tensor],
    args,
) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    layer_keys = reconstructor.layer_keys

    total_units = sum(
        reconstructor._decompositions[k].num_components
        for k in layer_keys
    )
    done = 0

    for depth_idx, layer_key in enumerate(layer_keys):
        decomp = reconstructor._decompositions[layer_key]
        C = decomp.num_components

        # Compute ev_scale = sqrt(mean_ev) for this layer.
        # Normalises epsilon so a unit perturbation in k-D space produces
        # comparable weight-space displacement across layer types regardless
        # of how much variance the k-D subspace captures.
        ev = getattr(decomp.decomposer, "explained_variance", None)
        if ev is not None:
            mean_ev = float(ev.mean().item())
            ev_scale = float(np.sqrt(mean_ev))
            mean_ev_str = f"{mean_ev:.3f}"
        else:
            ev_scale = 1.0   # fallback for UMAP / FourierKPCA
            mean_ev_str = "n/a"
        effective_eps = args.epsilon * ev_scale

        print(
            f"\n  Layer [{depth_idx:3d}/{len(layer_keys)-1}] "
            f"{layer_key[:55]:<55}  C={C}  "
            f"mean_ev={mean_ev_str}  "
            f"eff_eps={effective_eps:.4f}"
        )

        for unit_idx in range(C):
            sensitivity = perturb_unit_and_measure(
                reconstructor=reconstructor,
                model=model,
                layer_key=layer_key,
                unit_idx=unit_idx,
                epsilon=args.epsilon,
                n_directions=args.n_directions,
                baseline_logits_cache=baseline_logits_cache,
                loader=loader,
                task=args.task,
                ignore_index=getattr(args, "ignore_index", -100),
                device=args.device,
                rng=rng,
                ev_scale=ev_scale,
            )
            rows.append({
                "layer_key":   layer_key,
                "layer_type":  decomp.layer_type,
                "layer_depth": depth_idx,
                "unit_idx":    unit_idx,
                "sensitivity": sensitivity,
                "ev_scale":    ev_scale,
                "effective_epsilon": effective_eps,
            })
            done += 1
            if unit_idx % max(1, C // 5) == 0:
                pct = 100 * done / total_units
                print(
                    f"    unit {unit_idx:4d}/{C}  "
                    f"sensitivity={sensitivity:.5f}  "
                    f"[{pct:.1f}% overall]"
                )

    return rows


def plot_sensitivity_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap of sensitivity scores.
    Y-axis = layer depth, X-axis = unit index (up to max_units per layer).
    Color = sensitivity (mean KL).

    Note: This function has been generated with the help of Claude
    """
    layer_depths = sorted(df["layer_depth"].unique())
    max_units = df.groupby("layer_depth")["unit_idx"].max().max() + 1
    max_show = min(max_units, 256)   # cap width for readability

    grid = np.full((len(layer_depths), max_show), np.nan)
    for row_idx, depth in enumerate(layer_depths):
        sub = df[df["layer_depth"] == depth].sort_values("unit_idx")
        for _, r in sub.iterrows():
            if r["unit_idx"] < max_show:
                grid[row_idx, int(r["unit_idx"])] = r["sensitivity"]

    fig, ax = plt.subplots(
        figsize=(max(10, max_show // 16), max(4, len(layer_depths) // 4))
    )

    # Sensitivity (KL) is non-negative, so use a sequential colormap.
    vmax = np.nanpercentile(grid, 99)
    im = ax.imshow(
        grid, aspect="auto", cmap="viridis",
        vmin=0, vmax=vmax, interpolation="nearest"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"Sensitivity (mean KL, eps={df.attrs.get('epsilon','?')})"
    )
    ax.set_xlabel("Unit index")
    ax.set_ylabel("Layer depth")
    ax.set_title("Sensitivity map - per-unit KL under random perturbation")

    keys = (
        df.drop_duplicates("layer_depth")
          .sort_values("layer_depth")["layer_key"]
          .tolist()
    )
    step = max(1, len(layer_depths) // 10)
    ax.set_yticks(range(0, len(layer_depths), step))
    ax.set_yticklabels(
        [keys[i].split(".")[-2] if "." in keys[i] else keys[i]
         for i in range(0, len(layer_depths), step)],
        fontsize=7,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_heatmap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensitivity_by_layer_type(df: pd.DataFrame, out_dir: str) -> None:
    """
    Box plot: sensitivity distribution per layer type.

    Note: This function has been generated with the help of Claude
    """
    types = df["layer_type"].unique()
    data = [df[df["layer_type"] == t]["sensitivity"].values for t in types]
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(types)), 4))
    ax.boxplot(data, labels=types, patch_artist=True)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sensitivity (mean KL)")
    ax.set_title("Sensitivity distribution by layer type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_by_layer_type.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Sensitivity analysis via KL(baseline || perturbed).")

    # Model
    ap.add_argument("--backbone",    required=True,
                    choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant",     required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--task",        required=True,
                    choices=["classification", "segmentation"])

    # Data
    ap.add_argument("--dataset",      required=True,
                    choices=["imagenet100", "isic2019", "voc2012_sbd", "isic2018"])
    ap.add_argument("--data_root",    required=True)
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument(
        "--val_transforms", required=True,
        help="Path to val transforms YAML config, "
             "e.g. configs/transform/imagenet100_val.yaml",
    )

    # Checkpoint + decomposition
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--decomposers", required=True,
                    help="Path to *_decomposers.pt")
    ap.add_argument("--projection",  required=True,
                    help="Path to *_projected.pt")

    # Perturbation
    ap.add_argument("--epsilon",      type=float, default=0.5,
                    help="Perturbation magnitude in k-D decomposition space")
    ap.add_argument("--n_directions", type=int,   default=10,
                    help="Number of random perturbation directions per unit")

    # Runtime
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument(
        "--anchor_size", type=int, default=None,
        help="Number of val images to use as anchor set for each "
             "perturbation inference pass. None = full val set. Setting "
             "e.g. 200 reduces runtime by ~65x with minimal impact on "
             "relative sensitivity scores.",
    )
    ap.add_argument(
        "--cache_fp16", action="store_true",
        help="Store cached baseline logits in fp16 to halve CPU RAM use. "
             "Recommended for high-resolution segmentation (e.g. VOC2012). "
             "KL is robust to fp16 precision in this range.",
    )
    ap.add_argument("--out_dir", default="./results/sensitivity")

    # Subset of layers to analyse (optional - useful for speed)
    ap.add_argument(
        "--layer_filter", default=None,
        help="Comma-separated substring(s) to select layers, "
             "e.g. 'stage_3,stage_4'. None = all layers.",
    )
    args = ap.parse_args()

    if args.num_workers is None:
        args.num_workers = ceil(cpu_count() / 2)

    os.makedirs(args.out_dir, exist_ok=True)

    # Build model
    print("Building model...")
    model = build_model(
        backbone=args.backbone,
        variant=args.variant,
        task=args.task,
        n_classes=args.num_classes,
    ).to(args.device)
    model.eval()
    load_checkpoint(model, args.checkpoint, args.device, CkptMode.MATCH_SHAPES)

    # Build val loader
    print("Building val loader...")
    loader = build_val_loader(args)

    # Optionally subsample to a smaller anchor set for faster sensitivity
    # inference. Required to be deterministic (shuffle=False) so the
    # cached baseline logits line up with subsequent passes by batch index.
    if args.anchor_size is not None:
        print(f"Building anchor set ({args.anchor_size} images)...")
        loader = build_anchor_loader(loader, args.anchor_size, args.seed)

    # Load decomposition
    decompositions = load_decompositions(args.decomposers)
    projections = load_projections(args.projection)

    # Optional layer filter
    if args.layer_filter:
        substrings = [s.strip() for s in args.layer_filter.split(",")]
        decompositions = {
            k: v for k, v in decompositions.items()
            if any(sub in k for sub in substrings)
        }
        projections = {k: v for k, v in projections.items() if k in decompositions}
        print(f"  Layer filter applied: {len(decompositions)} layers selected")

    reconstructor = WeightReconstructor(
        decompositions=decompositions,
        projections=projections,
        device=args.device,
    )
    print("  [OK] Reconstructor built", flush=True)

    # Sanity probe: report output shape so user can verify the binary
    # vs multi-class KL branch on the first run.
    with torch.no_grad():
        item = loader.dataset[0]
        probe_batch = next(iter(loader))
        probe_images, _ = unpack_batch(
            probe_batch, args.task, torch.device(args.device)
        )
        probe_logits = model(probe_images[:1])
        print(f"Model output shape (probe): {tuple(probe_logits.shape)}")
        if probe_logits.shape[1] == 1:
            print("  -> 1-channel output detected: using Bernoulli KL.")
        else:
            print(f"  -> {probe_logits.shape[1]}-channel output: "
                  f"using categorical KL.")

    # Cache baseline logits (one forward pass over the anchor set).
    print("\nCaching baseline logits...")
    baseline_logits_cache = cache_baseline_logits(
        model=model,
        loader=loader,
        task=args.task,
        device=args.device,
        fp16=args.cache_fp16,
    )
    cache_mem_mb = sum(t.element_size() * t.nelement()
                       for t in baseline_logits_cache) / (1024 ** 2)
    print(f"  Cached {len(baseline_logits_cache)} batches "
          f"({cache_mem_mb:.1f} MB CPU RAM"
          f"{', fp16' if args.cache_fp16 else ', fp32'})")

    # Sanity check: baseline-vs-itself KL must be ~0.
    baseline_self_kl = run_inference(
        model=model,
        loader=loader,
        baseline_logits_cache=baseline_logits_cache,
        task=args.task,
        ignore_index=getattr(args, "ignore_index", -100),
        device=args.device,
    )
    print(f"  Baseline-vs-self KL (should be ~0): {baseline_self_kl:.2e}")
    if baseline_self_kl > 1e-3:
        print("  WARNING: baseline-vs-self KL is non-negligible. "
              "Check loader determinism (shuffle=False) and dropout/BN "
              "are in eval mode.")

    # Sensitivity loop
    total_units = sum(d.num_components for d in decompositions.values())
    total_inferences = total_units * args.n_directions
    n_anchor = args.anchor_size if args.anchor_size is not None else "full val"
    print(f"\nRunning sensitivity: {total_units} units x "
          f"{args.n_directions} directions = {total_inferences} inference passes")
    print(f"  Anchor set: {n_anchor} images per inference pass")
    print("  (Use --anchor_size 200 to speed up; --layer_filter to subset layers)\n")

    rows = run_sensitivity(
        model=model,
        reconstructor=reconstructor,
        loader=loader,
        baseline_logits_cache=baseline_logits_cache,
        args=args,
    )

    df = pd.DataFrame(rows)
    df.attrs["epsilon"] = args.epsilon

    csv_path = os.path.join(args.out_dir, "sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw CSV: {csv_path}")

    summary = (
        df.groupby(["layer_key", "layer_depth", "layer_type"])["sensitivity"]
        .agg(["mean", "std", "max"])
        .reset_index()
        .rename(columns={"mean": "sens_mean", "std": "sens_std", "max": "sens_max"})
        .sort_values("layer_depth")
    )
    summary_path = os.path.join(args.out_dir, "sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Top 10 most sensitive units
    print("\nTop 10 most sensitive units:")
    top = df.nlargest(10, "sensitivity")[
        ["layer_key", "unit_idx", "sensitivity"]
    ]
    print(top.to_string(index=False))

    plot_sensitivity_heatmap(df, args.out_dir)
    plot_sensitivity_by_layer_type(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
