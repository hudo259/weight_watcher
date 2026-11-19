"""
Summarises pairwise L2 distances between semantic unit embeddings in
projected.pt files, per layer and model. Useful for choosing a sensible
epsilon for sensitivity.py perturbations.

Output
------
    pairwise_distances.csv (per-layer stats)
    pairwise_distances_summary.csv (per (model, layer_type) aggregation)
    pairwise_distances_vs_epsilon.pdf (plot of distance distributions with epsilon reference lines)
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Distance computation
def pairwise_stats(pts: np.ndarray) -> dict:
    """
    Compute pairwise L2 distance stats for [C, k] points.
    Returns dict with min, max, mean, std, median of upper-triangle distances.
    Skips layers with fewer than 3 units.
    """
    C = pts.shape[0]
    if C < 3:
        return None

    # Compute upper triangle pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]   # [C, C, k]
    dist = np.linalg.norm(diff, axis=-1)        # [C, C]
    idx  = np.triu_indices(C, k=1)
    d    = dist[idx]

    return {
        "n_units":      C,
        "n_pairs":      len(d),
        "dist_min":     float(d.min()),
        "dist_max":     float(d.max()),
        "dist_mean":    float(d.mean()),
        "dist_std":     float(d.std()),
        "dist_median":  float(np.median(d)),
        "dist_p10":     float(np.percentile(d, 10)),
        "dist_p90":     float(np.percentile(d, 90)),
    }


def load_projected(path: str) -> tuple[str, dict]:
    """
    Load a projected.pt file. Returns (model_name, projections_dict).
    model_name is inferred from the filename.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    model_name = os.path.basename(path).replace("_projected.pt", "")
    return model_name, data["projections"]


def infer_layer_type(key: str) -> str:
    """Infer layer type from layer key string."""
    k = key.lower()
    if "depthwise" in k or "dw" in k or ("conv" in k and "depth" in k):
        return "depthwise_conv"
    if "attn" in k or "attention" in k:
        if "query" in k or ".q." in k:
            return "attention_query"
        if "output" in k or ".o." in k or "proj" in k:
            return "attention_output"
        return "attention"
    if "conv" in k:
        return "conv2d"
    if "linear" in k or "fc" in k or "mlp" in k or "pwconv" in k:
        return "linear"
    return "other"


def analyze_file(path: str) -> list[dict]:
    """Process one projected.pt file, return list of per-layer row dicts."""
    model_name, projections = load_projected(path)
    rows = []

    layer_keys = list(projections.keys())
    for depth_idx, key in enumerate(layer_keys):
        pts = projections[key]["points_3d"].float().cpu().numpy()
        stats = pairwise_stats(pts)
        if stats is None:
            continue

        rows.append({
            "model":       model_name,
            "layer_key":   key,
            "layer_type":  infer_layer_type(key),
            "layer_depth": depth_idx,
            **stats,
        })

    return rows


def plot_distances_vs_epsilon(
    df: pd.DataFrame,
    out_dir: str,
    epsilons: list[float] = [0.1, 0.3, 0.5, 1.0],
) -> None:
    """
    For each model: boxplot of dist_mean per layer type,
    with vertical reference lines at each epsilon value.
    Helps choose a sensible epsilon for sensitivity.py.

    NOTE: this function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY = {}
        def display(raw, mapping): return mapping.get(raw, raw)

    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention":        "Attention",
        "other":            "Other",
    }

    models   = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.5 * n_models, 5.0),
        squeeze=False,
    )

    for col_idx, model in enumerate(models):
        ax       = axes[0][col_idx]
        model_df = df[df["model"] == model]

        layer_types = sorted(model_df["layer_type"].unique())
        data   = [model_df[model_df["layer_type"] == lt]["dist_mean"].values
                  for lt in layer_types]
        labels = [LT_LABELS.get(lt, lt) for lt in layer_types]

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5})

        # Color boxes by layer type index
        cmap = plt.get_cmap("tab10")
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap(i))
            patch.set_alpha(0.7)

        # Epsilon reference lines
        colors_eps = ["green", "orange", "red", "darkred"]
        for eps, c in zip(epsilons, colors_eps):
            ax.axhline(eps, color=c, linewidth=1.0, linestyle="--",
                       alpha=0.8, label=f"ε={eps}")

        ax.set_title(display(model, MODEL_DISPLAY), fontsize=9)
        ax.set_ylabel("Mean pairwise distance", fontsize=8)
        ax.set_xlabel("Layer type", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx == n_models - 1:
            ax.legend(fontsize=7, frameon=False, title="epsilon",
                      title_fontsize=7)

    fig.suptitle(
        "Pairwise unit distances vs epsilon reference lines\n"
        "(epsilon should be small relative to mean distance for subtle perturbations,\n"
        " or comparable to mean distance for strong perturbations)",
        fontsize=9,
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "pairwise_distances_vs_epsilon.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Summarise pairwise distances in projected.pt files")
    ap.add_argument(
        "--projections_root", default=None,
        help="Root folder — all *_projected.pt files found recursively")
    ap.add_argument(
        "--pattern", default="*",
        help="Glob pattern for subfolders under projections_root (default: *)")
    ap.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit list of projected.pt file paths (overrides --projections_root)")
    ap.add_argument(
        "--epsilons", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0],
        help="Epsilon reference values to draw on the plot")
    ap.add_argument("--out_dir", default="./results/pairwise_distances")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect files
    if args.files:
        paths = args.files
    elif args.projections_root:
        pattern = os.path.join(
            args.projections_root, args.pattern, "*_projected.pt"
        )
        paths = sorted(glob.glob(pattern, recursive=True))
        if not paths:
            # Also try recursive glob
            pattern = os.path.join(
                args.projections_root, "**", "*_projected.pt"
            )
            paths = sorted(glob.glob(pattern, recursive=True))
    else:
        ap.error("Provide either --projections_root or --files")

    if not paths:
        print("No projected.pt files found.")
        return

    print(f"Found {len(paths)} projected.pt files.")

    all_rows = []
    for path in paths:
        print(f"  Processing: {os.path.basename(path)}")
        rows = analyze_file(path)
        all_rows.extend(rows)
        if rows:
            means = [r["dist_mean"] for r in rows]
            print(f"    {len(rows)} layers | "
                  f"dist_mean: min={min(means):.4f} "
                  f"mean={np.mean(means):.4f} "
                  f"max={max(means):.4f}")

    if not all_rows:
        print("No rows collected.")
        return

    df = pd.DataFrame(all_rows)

    # Raw CSV
    csv_path = os.path.join(args.out_dir, "pairwise_distances.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary per (model, layer_type)
    summary = (
        df.groupby(["model", "layer_type"])[
            ["dist_min", "dist_max", "dist_mean",
             "dist_std", "dist_median", "dist_p10", "dist_p90"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary_path = os.path.join(args.out_dir, "pairwise_distances_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\nPer (model, layer_type) mean pairwise distance:")
    pivot = df.groupby(["model", "layer_type"])["dist_mean"].mean().unstack()
    print(pivot.round(4).to_string())

    print(f"\nGlobal dist_mean: {df['dist_mean'].mean():.4f} "
          f"(range {df['dist_mean'].min():.4f}–{df['dist_mean'].max():.4f})")
    print(f"Current epsilon=0.5 is "
          f"{'LARGER' if 0.5 > df['dist_mean'].mean() else 'smaller'} "
          f"than global mean pairwise distance.")

    plot_distances_vs_epsilon(df, args.out_dir, args.epsilons)
    print("\nDone.")


if __name__ == "__main__":
    main()
