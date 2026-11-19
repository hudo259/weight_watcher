#!/usr/bin/env python3
"""
Reads variance_explained_summary.csv (from variance_explained.py)
and produces compact_bar_layer_type_by_task.pdf — the main paper figure
showing mean explained variance per layer type, with one bar per dataset,
grouped by classification / segmentation.

Usage
-----
python variance_explained_paper_plot.py \\
    --csv  results/variance_explained/variance_explained_summary.csv \\
    --out  figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from plot_style import (
    DATASET_COLORS as COLORS,
    DATASET_HATCH as HATCH,
    DATASET_EDGE as EDGE,
    DATASET_LABELS,
    LAYER_LABELS,
    LAYER_ORDER,
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

# ── Okabe-Ito task palette — identical to sensitivity_plots.py ────────────────
# Blues = classification, oranges = segmentation
TASK_KEY = {
    "classification_imagenet100":  "imagenet100",
    "classification_isic2019":     "isic2019",
    "segmentation_isic2018":       "isic2018",
    "segmentation_voc2012_sbd":    "voc2012_sbd",
}


# Dataset draw order: classification pair first, then segmentation pair
DATASETS_ORDERED = ["imagenet100", "isic2019", "voc2012_sbd", "isic2018"]


MODEL_LABELS = {
    # Raw names (from decomposers)
    "convnextv2_base":  "ConvNeXt V2 Base",
    "cyclemlp_b2":      "CycleMLP B2",
    "mit_b2":           "MiT B2",
    # Display names (variance_explained.py applies display() before saving CSV)
    "ConvNeXt V2 Base": "ConvNeXt V2 Base",
    "CycleMLP B2":      "CycleMLP B2",
    "MiT B2":           "MiT B2",
}

# Canonical display order — models are matched regardless of whether
# the CSV uses raw or display names.
_MODEL_ORDER = ["ConvNeXt V2 Base", "CycleMLP B2", "MiT B2"]


def _normalise_task(raw: str) -> str | None:
    """Map a raw task string to its canonical dataset key, or None."""
    # Try the lookup table first
    if raw in TASK_KEY:
        return TASK_KEY[raw]
    # Fallback: substring match
    for ds_key, ds_label in DATASET_LABELS.items():
        if ds_key in raw or ds_label.lower() in raw.lower():
            return ds_key
    return None


def plot_compact_bar_layer_type_by_task(
    df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """
    1 row × 3 subplots (one per model).
    X-axis  = layer type.
    Grouped bars = one per dataset, ordered (ImageNet-100, ISIC2019, VOC2012, ISIC2018).
    Color   = dataset (Okabe-Ito, matching sensitivity figures).
    Y-axis  = Mean explained variance (top-3 PCA components), 0–1.
    """
    # Normalise task column to dataset keys
    df = df.copy()
    df["dataset_key"] = df["task"].map(_normalise_task)
    df = df[df["dataset_key"].notna()].copy()

    # Only datasets that are actually present
    present_datasets = [d for d in DATASETS_ORDERED if d in df["dataset_key"].values]
    n_ds = len(present_datasets)

    raw_models = list(df["model"].unique())
    def _model_sort_key(m):
        display_name = MODEL_LABELS.get(m, m)
        try:
            return _MODEL_ORDER.index(display_name)
        except ValueError:
            return len(_MODEL_ORDER)
    models_present = sorted(raw_models, key=_model_sort_key)

    bar_w = min(0.16, 0.72 / n_ds)
    gap   = 0.06   # gap between cls and seg pairs
    c     = (bar_w + gap) / 2
    # Offsets: cls0, cls1 | gap | seg0, seg1
    if n_ds == 4:
        offsets = [-c - bar_w / 2, -c + bar_w / 2, c - bar_w / 2, c + bar_w / 2]
    else:
        offsets = list(np.linspace(
            -(n_ds - 1) / 2 * bar_w,
             (n_ds - 1) / 2 * bar_w,
            n_ds,
        ))

    fig, axes = plt.subplots(
        1, len(models_present),
        figsize=(7.2, 2.8),
        sharey=True,
        squeeze=False,
        gridspec_kw=dict(wspace=0.22),
    )

    for col_idx, model in enumerate(models_present):
        ax  = axes[0][col_idx]
        sub = df[df["model"] == model]

        # Layer types present in canonical order
        present_lts = [lt for lt in LAYER_ORDER
                       if lt in sub["layer_type"].values]
        x = np.arange(len(present_lts))

        for ds, off in zip(present_datasets, offsets):
            means, lo_errs, hi_errs = [], [], []
            for lt in present_lts:
                row = sub[
                    (sub["dataset_key"] == ds) &
                    (sub["layer_type"]   == lt)
                ]
                if row.empty:
                    means.append(0.0); lo_errs.append(0.0); hi_errs.append(0.0)
                else:
                    m = float(row["ev_mean"].iloc[0])
                    s = float(row["ev_std"].iloc[0]) if "ev_std" in row.columns else 0.0
                    means.append(m)
                    lo_errs.append(min(s, m))  # don't go below 0
                    hi_errs.append(s)

            ax.bar(
                x + off, means,
                width=bar_w,
                color=COLORS[ds],
                edgecolor=EDGE[ds],
                hatch=HATCH[ds],
                linewidth=0.45,
                label=DATASET_LABELS[ds],
                yerr=[lo_errs, hi_errs],
                error_kw=dict(
                    ecolor="0.25",
                    elinewidth=0.7,
                    capsize=2.0,
                    capthick=0.7,
                ),
                zorder=3,
            )

        # Subtle group dividers between layer types
        for xi in x[1:]:
            ax.axvline(xi - 0.5, color="0.88", linewidth=0.5, zorder=1)

        # Reference line at 0.5
        ax.axhline(0.5, color="0.55", linewidth=0.65, linestyle="--",
                   alpha=0.7, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LAYER_LABELS.get(lt, lt) for lt in present_lts],
            rotation=30, ha="right", rotation_mode="anchor",
        )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.set_title(MODEL_LABELS.get(model, model), pad=5)
        ax.grid(axis="y", linewidth=0.35, alpha=0.50, color="0.75", zorder=0)
        ax.set_axisbelow(True)

    axes[0][0].set_ylabel("Mean EV (top-3 PCA)")

    # ── Grouped legend: "Classification ›" | 2 handles | "Segmentation ›" | 2 ─
    seen: dict[str, matplotlib.artist.Artist] = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h

    cls_title = mpatches.Patch(color="none", label="Classification \u203a")
    seg_title = mpatches.Patch(color="none", label="Segmentation \u203a")

    cls_ds_in = [d for d in ["imagenet100", "isic2019"] if DATASET_LABELS[d] in seen]
    seg_ds_in = [d for d in ["voc2012_sbd", "isic2018"] if DATASET_LABELS[d] in seen]

    legend_handles = (
        [cls_title] + [seen[DATASET_LABELS[d]] for d in cls_ds_in] +
        [seg_title] + [seen[DATASET_LABELS[d]] for d in seg_ds_in]
    )
    legend_labels = (
        ["Classification \u203a"] + [DATASET_LABELS[d] for d in cls_ds_in] +
        ["Segmentation \u203a"]   + [DATASET_LABELS[d] for d in seg_ds_in]
    )

    leg = fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.10),
        frameon=True,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.8,
        fontsize=7.5,
    )
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        if "\u203a" in text.get_text():
            text.set_style("italic")
            text.set_color("0.4")
            handle.set_visible(False)

    path = out_dir / "compact_bar_layer_type_by_task.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Reworked compact_bar_layer_type_by_task in publication style."
    )
    ap.add_argument("--csv",  required=True,
                    help="Path to variance_explained_summary.csv")
    ap.add_argument("--out",  default=".",
                    help="Output directory")
    args = ap.parse_args()

    df      = pd.read_csv(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating compact_bar_layer_type_by_task...")
    plot_compact_bar_layer_type_by_task(df, out_dir)
    print(f"\nDone. Figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()