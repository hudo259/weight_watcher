#!/usr/bin/env python3
"""
Sensitivity analysis figures for the paper.

Produces three PDF figures:
  sensitivity_classification.pdf  – ImageNet-100 vs ISIC2019, 3 model subplots
  sensitivity_segmentation.pdf    – VOC2012 vs ISIC2018, 3 model subplots
  sensitivity_combined.pdf        – 3-row × 2-col grid, cls left / seg right,
                                    y-axes synced per row for direct comparison

Usage:
    python sensitivity_plots.py \\
        --cls  results/classification_summary.csv \\
        --seg  results/segmentation_summary.csv \\
        --out  figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from plot_style import (
    DATASET_COLORS as COLORS,
    DATASET_HATCH as HATCH,
    DATASET_EDGE as EDGE,
    DATASET_LABELS,
    LAYER_LABELS,
    MODEL_LABELS,
    MODEL_ORDER as MODELS,
    LAYER_ORDER,
)

# ── Publication rc settings ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.usetex":        False,   # flip to True if full LaTeX is installed
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
    "pdf.fonttype":       42,     # Type-1 / TrueType embeds for camera-ready
    "ps.fonttype":        42,
})


# ── Y-axis formatter: 3 sig-figs, switch to 1e-notation below 0.001 ──────────
def _yfmt(val, _pos):
    if val == 0:
        return "0"
    if abs(val) >= 0.01:
        return f"{val:.3f}"
    # Scientific notation with 1 sig-fig mantissa
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10 ** exp)
    return rf"${mant:.1f}\!\times\!10^{{{exp}}}$" if plt.rcParams["text.usetex"] \
           else f"{mant:.1f}e{exp}"


# ── Core bar-group drawing ────────────────────────────────────────────────────
def draw_bars(ax: plt.Axes, sub: pd.DataFrame, datasets: list[str]) -> None:
    """
    Draw a grouped bar chart on *ax* for one model and the given datasets.

    sub      : rows for one model; columns include
               <dataset>_mean_sens and <dataset>_std_sens
    datasets : list of dataset column prefixes, drawn left→right
    """
    # Keep only layer types present in this model's data, in canonical order
    present = [lt for lt in LAYER_ORDER if lt in sub["layer_type"].values]
    sub = sub.set_index("layer_type").loc[present].reset_index()

    n_lt = len(present)
    n_ds = len(datasets)
    bar_w = min(0.18, 0.7 / n_ds)           # narrower if more datasets
    x = np.arange(n_lt)
    offsets = np.linspace(
        -(n_ds - 1) / 2 * bar_w,
         (n_ds - 1) / 2 * bar_w,
        n_ds,
    )

    for ds, off in zip(datasets, offsets):
        means = sub[f"{ds}_mean_sens"].values
        stds  = sub[f"{ds}_std_sens"].values
        # Clip the lower error bar at zero (std can exceed mean for heavy tails)
        yerr_lo = np.minimum(stds, means)
        yerr_hi = stds
        ax.bar(
            x + off, means,
            width=bar_w,
            color=COLORS[ds],
            edgecolor=EDGE[ds],
            hatch=HATCH[ds],
            linewidth=0.5,
            label=DATASET_LABELS[ds],
            yerr=[yerr_lo, yerr_hi],
            error_kw=dict(
                ecolor="0.25",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
            ),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LAYER_LABELS.get(lt, lt) for lt in present],
        rotation=35, ha="right", rotation_mode="anchor",
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_yfmt))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.35, alpha=0.55, color="0.7", zorder=0)
    ax.set_axisbelow(True)


def _shared_legend(fig: plt.Figure, axes, all_datasets: list[str]) -> None:
    """Place a single deduplicated legend above the figure."""
    # Collect handles from the first axis that has them
    seen: dict[str, matplotlib.artist.Artist] = {}
    for ax in np.array(axes).flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    # Preserve insertion order (= dataset order)
    ordered = [seen[DATASET_LABELS[ds]] for ds in all_datasets
               if DATASET_LABELS[ds] in seen]
    ordered_labels = [DATASET_LABELS[ds] for ds in all_datasets
                      if DATASET_LABELS[ds] in seen]
    fig.legend(
        ordered, ordered_labels,
        loc="upper center",
        ncol=len(all_datasets),
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.2,
    )


# ── Figure 1: Classification ──────────────────────────────────────────────────
def plot_classification(cls_df: pd.DataFrame, out_dir: Path) -> Path:
    datasets = ["imagenet100", "isic2019"]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.5), sharey=False,
        gridspec_kw=dict(wspace=0.40),
    )
    for ax, model in zip(axes, MODELS):
        sub = cls_df[cls_df["model"] == model].copy()
        draw_bars(ax, sub, datasets)
        ax.set_title(MODEL_LABELS[model], pad=5)
    axes[0].set_ylabel("Mean sensitivity $s_c$")
    _shared_legend(fig, axes, datasets)
    path = out_dir / "sensitivity_classification.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 2: Segmentation ────────────────────────────────────────────────────
def plot_segmentation(seg_df: pd.DataFrame, out_dir: Path) -> Path:
    datasets = ["voc2012_sbd", "isic2018"]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.5), sharey=False,
        gridspec_kw=dict(wspace=0.40),
    )
    for ax, model in zip(axes, MODELS):
        sub = seg_df[seg_df["model"] == model].copy()
        draw_bars(ax, sub, datasets)
        ax.set_title(MODEL_LABELS[model], pad=5)
    axes[0].set_ylabel("Mean sensitivity $s_c$")
    _shared_legend(fig, axes, datasets)
    path = out_dir / "sensitivity_segmentation.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 3: Combined 3×2 grid (cls / seg per model) ────────────────────────
def plot_combined(cls_df: pd.DataFrame, seg_df: pd.DataFrame,
                  out_dir: Path) -> Path:
    """
    3 rows (models) × 2 columns (classification | segmentation).
    Y-axes are synced per row so the task-type comparison is direct.
    This is the recommended main-paper figure.
    """
    cls_ds = ["imagenet100", "isic2019"]
    seg_ds = ["voc2012_sbd", "isic2018"]

    fig, axes = plt.subplots(
        3, 2, figsize=(7.0, 5.8),
        gridspec_kw=dict(wspace=0.35, hspace=0.60),
    )

    for row_i, model in enumerate(MODELS):
        ax_c, ax_s = axes[row_i, 0], axes[row_i, 1]

        draw_bars(ax_c, cls_df[cls_df["model"] == model].copy(), cls_ds)
        draw_bars(ax_s, seg_df[seg_df["model"] == model].copy(), seg_ds)

        # Column headers only on the first row
        if row_i == 0:
            ax_c.set_title("Classification", pad=5, fontsize=9)
            ax_s.set_title("Segmentation",   pad=5, fontsize=9)

        # Row label (model name) on the left y-axis
        ax_c.set_ylabel(MODEL_LABELS[model] + "\n$s_c$", fontsize=7.5)

        # Sync y-limits across the row so cls and seg use the same scale
        y_top = max(ax_c.get_ylim()[1], ax_s.get_ylim()[1])
        ax_c.set_ylim(0, y_top * 1.05)
        ax_s.set_ylim(0, y_top * 1.05)

    _shared_legend(fig, axes, cls_ds + seg_ds)
    path = out_dir / "sensitivity_combined.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 4: Task-effect delta (segmentation − classification, global means) ─
def plot_task_delta(cls_df: pd.DataFrame, seg_df: pd.DataFrame,
                    out_dir: Path) -> Path:
    """
    Per-(model, layer_type): plot mean_seg − mean_cls (averaged across
    datasets within each task type) as a horizontal lollipop chart.
    Highlights how much more functionally critical units become under
    segmentation training.
    """
    # Average across datasets within each task
    cls_mean = cls_df.copy()
    seg_mean = seg_df.copy()

    cls_cols = [c for c in cls_df.columns if c.endswith("_mean_sens")]
    seg_cols = [c for c in seg_df.columns if c.endswith("_mean_sens")]

    cls_mean["mean_cls"] = cls_df[cls_cols].mean(axis=1)
    seg_mean["mean_seg"] = seg_df[seg_cols].mean(axis=1)

    merged = cls_mean[["model", "layer_type", "mean_cls"]].merge(
        seg_mean[["model", "layer_type", "mean_seg"]],
        on=["model", "layer_type"],
    )
    merged["delta"] = merged["mean_seg"] - merged["mean_cls"]

    # Label: "Model / LayerType"
    merged["label"] = merged.apply(
        lambda r: f"{MODEL_LABELS[r.model].split()[0]}  {LAYER_LABELS.get(r.layer_type, r.layer_type)}",
        axis=1,
    )
    # Sort by delta descending
    merged = merged.sort_values("delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(merged) + 0.6))
    y = np.arange(len(merged))
    colors = ["#D55E00" if d > 0 else "#0072B2" for d in merged["delta"]]

    ax.barh(y, merged["delta"], height=0.55,
            color=colors, edgecolor="none", zorder=3)
    ax.scatter(merged["delta"], y, color=[
        "#9A3E00" if d > 0 else "#004E7C" for d in merged["delta"]
    ], s=20, zorder=4, linewidths=0)

    ax.axvline(0, color="0.3", linewidth=0.75, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"], fontsize=7.5)
    ax.set_xlabel("$\Delta s_c$ (segmentation − classification)", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.grid(axis="x", linewidth=0.35, alpha=0.55, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    path = out_dir / "sensitivity_task_delta.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate sensitivity paper figures.")
    ap.add_argument("--cls", required=True,
                    help="Path to classification_summary.csv")
    ap.add_argument("--seg", required=True,
                    help="Path to segmentation_summary.csv")
    ap.add_argument("--out", default=".",
                    help="Output directory (default: current directory)")
    args = ap.parse_args()

    cls_df = pd.read_csv(args.cls)
    seg_df = pd.read_csv(args.seg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sensitivity figures...")
    plot_classification(cls_df, out_dir)
    plot_segmentation(seg_df, out_dir)
    plot_combined(cls_df, seg_df, out_dir)
    plot_task_delta(cls_df, seg_df, out_dir)
    print(f"\nAll figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()


# ── Figure 5: All 4 datasets, 4 bars per layer type, 1×3 compact grid ─────────
def plot_all_tasks(cls_df: pd.DataFrame, seg_df: pd.DataFrame,
                   out_dir: Path) -> Path:
    """
    1 row × 3 subplots (one per model). Each subplot shows 4 bars per
    layer type — (ImageNet-100, ISIC2019) | gap | (VOC2012, ISIC2018) —
    so classification and segmentation are directly comparable within the
    same panel.  Y-axes are independent per model.

    This is the most compact summary figure: all tasks, all models,
    in a single full-width row.
    """
    import matplotlib.patches as mpatches

    df = cls_df.merge(seg_df, on=["model", "layer_type"])

    MEAN_COLS = {ds: f"{ds}_mean_sens" for ds in DATASETS}
    STD_COLS  = {ds: f"{ds}_std_sens"  for ds in DATASETS}

    def _draw(ax, sub):
        present = [lt for lt in LAYER_ORDER if lt in sub["layer_type"].values]
        sub = sub.set_index("layer_type").loc[present].reset_index()
        bar_w = 0.14
        gap   = 0.06
        c     = (bar_w + gap) / 2
        offsets = [-c - bar_w / 2, -c + bar_w / 2, c - bar_w / 2, c + bar_w / 2]
        x = np.arange(len(present))
        for ds, off in zip(DATASETS, offsets):
            means = sub[MEAN_COLS[ds]].values
            stds  = sub[STD_COLS[ds]].values
            ax.bar(x + off, means, width=bar_w,
                   color=COLORS[ds], edgecolor=EDGE[ds],
                   hatch=HATCH[ds], linewidth=0.45,
                   label=DATASET_LABELS[ds],
                   yerr=[np.minimum(stds, means), stds],
                   error_kw=dict(ecolor="0.25", elinewidth=0.7,
                                 capsize=1.8, capthick=0.7),
                   zorder=3)
        for xi in x[1:]:
            ax.axvline(xi - 0.5, color="0.88", linewidth=0.5, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS.get(lt, lt) for lt in present],
                           rotation=30, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linewidth=0.35, alpha=0.5, color="0.75", zorder=0)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False,
                             gridspec_kw=dict(wspace=0.40))
    for ax, model in zip(axes, MODELS):
        _draw(ax, df[df["model"] == model].copy())
        ax.set_title(MODEL_LABELS[model], pad=5)
    axes[0].set_ylabel("Mean sensitivity $s_c$")

    # Grouped legend: "Classification ›" | 2 handles | "Segmentation ›" | 2 handles
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h

    cls_title = mpatches.Patch(color="none", label="Classification \u203a")
    seg_title = mpatches.Patch(color="none", label="Segmentation \u203a")
    legend_handles = [cls_title, seen["ImageNet-100"], seen["ISIC2019"],
                      seg_title, seen["VOC2012"],       seen["ISIC2018"]]
    legend_labels  = ["Classification \u203a", "ImageNet-100", "ISIC2019",
                      "Segmentation \u203a",   "VOC2012",       "ISIC2018"]
    leg = fig.legend(legend_handles, legend_labels,
                     loc="upper center", ncol=6,
                     bbox_to_anchor=(0.5, 1.10),
                     frameon=True, handlelength=1.3,
                     handletextpad=0.4, columnspacing=0.8,
                     fontsize=7.5)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        if "\u203a" in text.get_text():
            text.set_style("italic")
            text.set_color("0.4")
            handle.set_visible(False)

    path = out_dir / "sensitivity_all_tasks.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path