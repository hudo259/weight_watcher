#!/usr/bin/env python3
"""
k-sweep figure: displacement-sensitivity correlation vs. embedding dimension.

Reads ksweep_ALL.csv and shows that the SIGN of the displacement-sensitivity
correlation persists (and typically strengthens) as the embedding dimension k
grows. This directly answers the concern that the sign reversal in projection
layers could be an artifact of discarding variance at k=3.

How to read it: lines below zero that move DOWNWARD with growing k are the
projection layers whose negative correlation strengthens; lines above zero
moving UPWARD are the spatial-mixing convolutions. Were the effect a k=3
truncation artifact, all lines would instead converge toward zero.

Colour = layer type (consistent with the other paper figures)
Panel  = model
Line   = one task (dataset), so within-panel consistency is visible

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
    LAYER_COLORS as LT_COLORS,
    LAYER_LABELS as LT_LABELS,
    LAYER_ORDER as LT_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
)

# Publication rc settings
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


# Line style per task, so the four datasets are distinguishable within a panel
TASK_STYLES = {
    "classification_imagenet100": ("-",  "ImageNet-100"),
    "classification_isic2019":    ("--", "ISIC2019"),
    "segmentation_isic2018":      (":",  "ISIC2018"),
    "segmentation_voc2012_sbd":   ("-.", "VOC2012"),
}


def _task_style(task):
    """Return (linestyle, label) for a task, tolerating naming variations."""
    if task in TASK_STYLES:
        return TASK_STYLES[task]
    t = str(task).lower()
    for key, val in TASK_STYLES.items():
        ds = key.split("_", 1)[1]
        if ds in t or t in ds:
            return val
    return ("-", str(task))


def plot_ksweep(df, out_dir, mode="task_lines"):
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    if not models:
        models = sorted(df["model"].unique())

    fig, axes = plt.subplots(
        1, len(models), figsize=(7.2, 2.7), squeeze=False, sharey=True,
        gridspec_kw=dict(wspace=0.12),
    )

    ks = sorted(df["k"].unique())
    lt_present = []

    for col, model in enumerate(models):
        ax = axes[0][col]
        sub = df[df["model"] == model]

        # zero line: the reference the reviewer concern is about
        ax.axhline(0.0, color="0", linewidth=1.0, zorder=2)

        for lt in [l for l in LT_ORDER if l in sub["layer_type"].values]:
            if lt not in lt_present:
                lt_present.append(lt)
            colour = LT_COLORS.get(lt, "0.5")
            lt_sub = sub[sub["layer_type"] == lt]

            if mode == "task_mean":
                agg = lt_sub.groupby("k")["rho_disp_sens"].mean().reindex(ks)
                ax.plot(ks, agg.values, color=colour, linewidth=1.6,
                        marker="o", markersize=3, zorder=4)
            else:
                for task, t_sub in lt_sub.groupby("task"):
                    ls, _ = _task_style(task)
                    t_sub = t_sub.sort_values("k")
                    ax.plot(t_sub["k"], t_sub["rho_disp_sens"],
                            color=colour, linewidth=1.1, linestyle=ls,
                            marker="o", markersize=2, alpha=0.85, zorder=4)

        ax.set_title(MODEL_LABELS.get(model, model), pad=5)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        # suppress log minor ticks (4x10^0 etc.) that collide with the k labels
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.get_xaxis().set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("Embedding dimension $k$")
        if col == 0:
            ax.set_ylabel(r"$\rho$(displacement, sensitivity)")
        ax.grid(axis="both", linewidth=0.3, alpha=0.4, color="0.8", zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.06)

    # --- Legend: layer-type colours, plus task line styles if shown ----------
    ordered_lts = [l for l in LT_ORDER if l in lt_present]
    handles = [mlines.Line2D([], [], color=LT_COLORS.get(lt, "0.5"),
                             linewidth=1.8) for lt in ordered_lts]
    labels = [LT_LABELS.get(lt, lt) for lt in ordered_lts]

    if mode != "task_mean":
        tasks_present = sorted(df["task"].unique())
        if tasks_present:
            handles.append(mlines.Line2D([], [], color="none"))
            labels.append(" ")
        for task in tasks_present:
            ls, lab = _task_style(task)
            handles.append(mlines.Line2D([], [], color="0.35",
                                         linestyle=ls, linewidth=1.2))
            labels.append(lab)

    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(labels), 9), bbox_to_anchor=(0.5, 1.16),
               frameon=True, handlelength=1.8, handletextpad=0.5, columnspacing=1.0)

    suffix = "_mean" if mode == "task_mean" else ""
    path = out_dir / f"ksweep_correlation{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def plot_ksweep_per_task(df, out_dir):
    """One figure per task, so each plot shows only that dataset's lines."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    if not models:
        models = sorted(df["model"].unique())

    ks = sorted(df["k"].unique())
    paths = []

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]          # <-- filter by task
        if task_df.empty:
            continue

        fig, axes = plt.subplots(                  # <-- new figure per task
            1, len(models), figsize=(7.2, 2.7), squeeze=False, sharey=True,
            gridspec_kw=dict(wspace=0.12),
        )
        lt_present = []                            # <-- reset per task

        for col, model in enumerate(models):
            ax = axes[0][col]
            sub = task_df[task_df["model"] == model]

            ax.axhline(0.0, color="0", linewidth=1.0, zorder=2)

            for lt in [l for l in LT_ORDER if l in sub["layer_type"].values]:
                if lt not in lt_present:
                    lt_present.append(lt)
                lt_sub = sub[sub["layer_type"] == lt].sort_values("k")
                ax.plot(lt_sub["k"], lt_sub["rho_disp_sens"],
                        color=LT_COLORS.get(lt, "0.5"), linewidth=1.4,
                        linestyle="-", marker="o", markersize=2,
                        alpha=0.9, zorder=4)

            ax.set_title(MODEL_LABELS.get(model, model), pad=5)
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
            ax.get_xaxis().set_minor_locator(mticker.NullLocator())
            ax.set_xlabel("Embedding dimension $k$")
            if col == 0:
                ax.set_ylabel(r"$\rho$(displacement, sensitivity)")
            ax.grid(axis="both", linewidth=0.3, alpha=0.4, color="0.8", zorder=0)
            ax.set_axisbelow(True)
            ax.margins(x=0.06)

        ordered_lts = [l for l in LT_ORDER if l in lt_present]
        handles = [mlines.Line2D([], [], color=LT_COLORS.get(lt, "0.5"),
                                 linewidth=1.8) for lt in ordered_lts]
        labels = [LT_LABELS.get(lt, lt) for lt in ordered_lts]
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(labels), 9), bbox_to_anchor=(0.5, 1.14),
                   frameon=True, handlelength=1.8, handletextpad=0.5,
                   columnspacing=1.0)

        # readable filename: strip the task prefix, keep the dataset
        safe = str(task).replace("/", "_")
        path = out_dir / f"ksweep_correlation_{safe}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)                              # <-- inside loop, but after save
        print(f"  Saved: {path}")
        paths.append(path)

    return paths                                    # <-- after the loop


def main():
    ap = argparse.ArgumentParser(
        description="Plot displacement-sensitivity correlation vs. k")
    ap.add_argument("--csv", required=True, help="Path to ksweep_ALL.csv")
    ap.add_argument("--out", default=".", help="Output directory")
    ap.add_argument("--mode", default="task_lines",
                    choices=["task_lines", "task_mean", "per_task"],
                    help="One line per task (default) or averaged over tasks (task_mean) or one plot per task (per_task)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model names")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.models:
        df = df[df["model"].isin(args.models)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows | models: {sorted(df['model'].unique())} "
          f"| k: {sorted(df['k'].unique())}")

    if args.mode == "per_task":
        plot_ksweep_per_task(df, out_dir)
    else:
        plot_ksweep(df, out_dir, mode=args.mode)
    print("Done.")


if __name__ == "__main__":
    main()
