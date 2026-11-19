#!/usr/bin/env python3
"""
Intervention figure: metric degradation vs. perturbation magnitude.

Reads the merged intervention_results CSV (all models and datasets in one file)
and plots how much the task metric drops as the perturbation magnitude epsilon
grows, separately for the three conditions:

    geometric  - perturb the top-decile units by sensitivity s_c
    random     - perturb a random unit subset of the same size
    residual   - perturb the SAME units, but in the discarded residual subspace

Reading the figure: the vertical gap between the geometric and random lines is
the effect. It is clearest at small-to-moderate epsilon; at large epsilon both
approach total collapse and the gap compresses, which is why the paper reports
epsilon = 2.

Expected CSV columns:
    task, model, dataset, epsilon, condition, metric_mean, metric_std,
    drop_from_baseline, baseline

Merge the per-run files first, e.g. in DuckDB:
    COPY (SELECT * FROM read_csv_auto('intervention/*/intervention_results.csv',
          union_by_name=true)) TO 'intervention_ALL.csv' (HEADER, DELIMITER ',');

Note: This file has been generated with the help of Claude
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import pandas as pd
from plot_style import (
    MODEL_LABELS,
    MODEL_ORDER,
)

# Publication rc settings (matching the other paper figures) 
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

# Distinct from the layer-type palette used elsewhere; geo is the "hero" line.
COND_STYLE = {
    "geometric": {"color": "#D55E00", "marker": "o", "ls": "-",
                  "label": "geometric (top-10% by $s_c$)"},
    "random":    {"color": "#0072B2", "marker": "o", "ls": "--",
                  "label": "random"},
    "residual":  {"color": "#009E73", "marker": "o", "ls": ":",
                  "label": "residual subspace"},
}
COND_ORDER = ["geometric", "random", "residual"]


DATASET_LABELS = {
    "imagenet100":  "ImageNet-100",
    "isic2019":     "ISIC2019",
    "isic2018":     "ISIC2018",
    "voc2012_sbd":  "VOC2012",
}

# Line style per dataset, used when several datasets share one panel
DATASET_LS = {
    "imagenet100":  "-",
    "isic2019":     "--",
    "isic2018":     ":",
    "voc2012_sbd":  "-.",
}


def _metric_label(task: str) -> str:
    return r"$\Delta$ accuracy" if "classif" in str(task).lower() else r"$\Delta$ mIoU"


def _panel(ax, sub, ks, show_errorbars=True, dataset_ls=None):
    """Draw the three condition lines for one (model, dataset) subset."""
    for cond in COND_ORDER:
        c_sub = sub[sub["condition"] == cond].sort_values("epsilon")
        if c_sub.empty:
            continue
        st = COND_STYLE[cond]
        ls = dataset_ls if dataset_ls else st["ls"]
        if show_errorbars and "metric_std" in c_sub.columns:
            ax.errorbar(c_sub["epsilon"], c_sub["drop_from_baseline"],
                        yerr=c_sub["metric_std"],
                        color=st["color"], marker=st["marker"], linestyle=ls,
                        linewidth=1.4, markersize=2,
                        capsize=2, elinewidth=0.7, capthick=0.7,
                        alpha=0.9, zorder=4)
        else:
            ax.plot(c_sub["epsilon"], c_sub["drop_from_baseline"],
                    color=st["color"], marker=st["marker"], linestyle=ls,
                    linewidth=1.4, markersize=2, alpha=0.9, zorder=4)

    ax.axhline(0.0, color="0.5", linewidth=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.get_xaxis().set_minor_locator(mticker.NullLocator())
    ax.grid(axis="both", linewidth=0.3, alpha=0.4, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)


def _legend(fig, extra_handles=None, extra_labels=None, ncol=None):
    handles = [mlines.Line2D([], [], color=COND_STYLE[c]["color"],
                             linestyle=COND_STYLE[c]["ls"],
                             linewidth=1.6, markersize=2)
               for c in COND_ORDER]
    labels = [COND_STYLE[c]["label"] for c in COND_ORDER]
    if extra_handles:
        handles += [mlines.Line2D([], [], color="none")] + extra_handles
        labels += [" "] + extra_labels
    fig.legend(handles, labels, loc="upper center",
               ncol=ncol or min(len(labels), 7),
               bbox_to_anchor=(0.5, 1.14), frameon=True,
               handlelength=2.0, handletextpad=0.5, columnspacing=1.0)


def plot_combined(df, out_dir, datasets=None):
    """One panel per model; datasets distinguished by line style."""
    models = [m for m in MODEL_ORDER if m in df["model"].values] or \
             sorted(df["model"].unique())
    ds_present = [d for d in DATASET_LS if d in df["dataset"].values]
    if datasets:
        ds_present = [d for d in ds_present if d in datasets]

    ks = sorted(df["epsilon"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.7),
                             squeeze=False, sharey=False,
                             gridspec_kw=dict(wspace=0.28))

    for col, model in enumerate(models):
        ax = axes[0][col]
        for ds in ds_present:
            sub = df[(df["model"] == model) & (df["dataset"] == ds)]
            if sub.empty:
                continue
            _panel(ax, sub, ks, show_errorbars=(len(ds_present) == 1),
                   dataset_ls=DATASET_LS.get(ds) if len(ds_present) > 1 else None)
        ax.set_title(MODEL_LABELS.get(model, model), pad=5)
        ax.set_xlabel(r"Perturbation magnitude $\varepsilon$")
        if col == 0:
            ax.set_ylabel("Metric drop from baseline")

    extra_h, extra_l = None, None
    if len(ds_present) > 1:
        extra_h = [mlines.Line2D([], [], color="0.35",
                                 linestyle=DATASET_LS[d], linewidth=1.3)
                   for d in ds_present]
        extra_l = [DATASET_LABELS.get(d, d) for d in ds_present]
    _legend(fig, extra_h, extra_l)

    path = out_dir / "intervention_epsilon_sweep.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_per_dataset(df, out_dir):
    """One figure per dataset; one panel per model. Cleaner, no line-style overload."""
    ks = sorted(df["epsilon"].unique())
    paths = []
    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        models = [m for m in MODEL_ORDER if m in ds_df["model"].values] or \
                 sorted(ds_df["model"].unique())
        if not models:
            continue

        fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.7),
                                 squeeze=False, sharey=False,
                                 gridspec_kw=dict(wspace=0.28))
        for col, model in enumerate(models):
            ax = axes[0][col]
            sub = ds_df[ds_df["model"] == model]
            _panel(ax, sub, ks, show_errorbars=True)
            ax.set_title(MODEL_LABELS.get(model, model), pad=5)
            ax.set_xlabel(r"Perturbation magnitude $\varepsilon$")
            if col == 0:
                task = sub["task"].iloc[0] if not sub.empty else ""
                ax.set_ylabel(_metric_label(task) + " from baseline")

        _legend(fig)
        safe = str(ds).replace("/", "_")
        path = out_dir / f"intervention_epsilon_{safe}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(
        description="Plot intervention degradation vs. perturbation magnitude")
    ap.add_argument("--csv", required=True,
                    help="Merged intervention results CSV (all models/datasets)")
    ap.add_argument("--out", default=".", help="Output directory")
    ap.add_argument("--per-dataset", action="store_true",
                    help="Also write one figure per dataset")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict the combined figure to these datasets")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these models")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.models:
        df = df[df["model"].isin(args.models)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows | models: {sorted(df['model'].unique())}")
    print(f"  datasets: {sorted(df['dataset'].unique())}")
    print(f"  epsilon:  {sorted(df['epsilon'].unique())}")
    print(f"  conditions: {sorted(df['condition'].unique())}")

    plot_combined(df, out_dir, datasets=args.datasets)
    if args.per_dataset:
        plot_per_dataset(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
