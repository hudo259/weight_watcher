#!/usr/bin/env python3
"""
Reads trajectory_geometry.csv (raw per-unit file from trajectory_geometry.py)
and produces ratio_displacement_summary.pdf — the main paper figure showing
directionality ratio vs. normalised displacement, one subplot per model.

Color  = layer type (consistent across subplots).
Marker = task type: circle for classification, triangle for segmentation.

Usage
-----
python trajectory_geometry_paper_plot.py \\
    --csv  results/trajectory_geometry/trajectory_geometry.csv \\
    --out  figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from plot_style import (
    LAYER_COLORS as LT_COLORS,
    LAYER_LABELS as LT_LABELS,
    MODEL_LABELS,
    MODEL_ORDER as MODELS,
)

# ── Publication rc settings (matching sensitivity_plots.py) ──────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.usetex":        False,
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     9,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.7",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.75,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


# Task ordering — canonical for task name detection
CLS_TASKS = {"classification_imagenet100", "classification_isic2019"}
SEG_TASKS = {"segmentation_isic2018", "segmentation_voc2012_sbd"}

def _is_segmentation(task: str) -> bool:
    return "segmentation" in task.lower()

TASK_DISPLAY = {
    "classification_imagenet100":  "ImageNet-100",
    "classification_isic2019":     "ISIC2019",
    "segmentation_isic2018":       "ISIC2018",
    "segmentation_voc2012_sbd":    "VOC2012",
}

# Marker: circle = classification, triangle = segmentation
def _marker(task: str) -> str:
    return "^" if _is_segmentation(task) else "o"

_MARKER_SIZE = 22   # scatter s= value


def _quadrant_labels(ax: plt.Axes) -> None:
    """Add the four descriptive corner labels."""
    kw = dict(fontsize=5, color="black", transform=ax.transAxes)
    ax.text(0.02, 0.97, "directional, small \u03b4",
            va="top",    ha="left",  **kw)
    ax.text(0.98, 0.97, "directional, large \u03b4",
            va="top",    ha="right", **kw)
    ax.text(0.02, 0.03, "oscillatory, small \u03b4",
            va="bottom", ha="left",  **kw)
    ax.text(0.98, 0.03, "oscillatory, large \u03b4",
            va="bottom", ha="right", **kw)


def plot_ratio_displacement_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Main paper figure: 1 row × 3 subplots (one per model).
    Each point = one (task, layer_depth, layer_type) layer-level mean.
    """
    models_present = [m for m in MODELS if m in df["model"].values]

    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(7.2, 3.0),
        squeeze=False,
        gridspec_kw=dict(wspace=0.30),
    )

    # Collect handles for the shared legend
    lt_handles:  dict[str, mpatches.Patch] = {}
    cls_handle:  mlines.Line2D | None = None
    seg_handle:  mlines.Line2D | None = None

    for col_idx, model in enumerate(models_present):
        ax       = axes[0][col_idx]
        model_df = df[df["model"] == model]

        # Aggregate to per-(task, layer_depth, layer_type) means
        agg = (
            model_df
            .groupby(["task", "layer_depth", "layer_type"])[
                ["displacement_norm", "ratio"]
            ]
            .mean()
            .reset_index()
        )

        # Median displacement — used as the vertical reference line
        x_mid = float(agg["displacement_norm"].median())

        # Reference lines
        ax.axhline(0.5,   color="0.75", linewidth=0.6, linestyle="--", zorder=1)
        ax.axvline(x_mid, color="0.75", linewidth=0.6, linestyle="--", zorder=1)

        # Scatter
        for lt, lt_sub in agg.groupby("layer_type"):
            color = LT_COLORS.get(lt, "0.5")
            label = LT_LABELS.get(lt, lt)

            for task, task_sub in lt_sub.groupby("task"):
                marker = _marker(task)
                ax.scatter(
                    task_sub["displacement_norm"],
                    task_sub["ratio"],
                    c=[color],
                    marker=marker,
                    s=_MARKER_SIZE,
                    alpha=0.70,
                    linewidths=0,
                    zorder=3,
                )

            # Layer-type legend proxy
            if label not in lt_handles:
                lt_handles[label] = mpatches.Patch(
                    facecolor=color, label=label,
                    linewidth=0, alpha=0.85,
                )

        # Task-type marker legend proxies (only build once)
        if cls_handle is None:
            cls_handle = mlines.Line2D(
                [], [], color="0.25", marker="o", markersize=5,
                linestyle="none", label="Classification",
            )
        if seg_handle is None:
            seg_handle = mlines.Line2D(
                [], [], color="0.25", marker="^", markersize=5,
                linestyle="none", label="Segmentation",
            )

        _quadrant_labels(ax)

        ax.set_title(MODEL_LABELS.get(model, model), pad=5)
        ax.set_xlabel("Displacement (norm.)")
        if col_idx == 0:
            ax.set_ylabel("Directionality ratio")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.grid(axis="both", linewidth=0.30, alpha=0.40, color="0.8", zorder=0)
        ax.set_axisbelow(True)

    # ── Shared legend: layer types | separator | task markers ────────────────
    # Preserve canonical layer-type order
    ordered_lt = [LT_LABELS[lt] for lt in LT_LABELS if LT_LABELS[lt] in lt_handles]
    lt_h = [lt_handles[k] for k in ordered_lt]

    separator = mpatches.Patch(color="none", label=" ")

    fig.legend(
        lt_h + [separator, cls_handle, seg_handle],
        ordered_lt + [" ", "Classification", "Segmentation"],
        loc="upper center",
        ncol=len(lt_h) + 3,
        bbox_to_anchor=(0.5, 1.06),
        frameon=True,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.9,
        fontsize=7.5,
    )

    path = out_dir / "ratio_displacement_summary.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Reworked ratio_displacement_summary in publication style."
    )
    ap.add_argument("--csv",  required=True,
                    help="Path to trajectory_geometry.csv")
    ap.add_argument("--out",  default=".",
                    help="Output directory")
    args = ap.parse_args()

    df      = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ratio_displacement_summary...")
    plot_ratio_displacement_summary(df, out_dir)
    print(f"\nDone. Figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()