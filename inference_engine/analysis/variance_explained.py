"""
For each layer of each model, report the fraction of trajectory variance
captured by the top k=3 PCA / FourierKPCA components per semantic unit
(filter, neuron, attention head).

UMAP does not have an explained-variance property and is skipped here.
Use reconstruction_fidelity.py for a method-agnostic quality measure.

Inputs
------
One or more decomposers files (*_decomposers.pt), each produced by
projection.py for a single (model, task, method) combination.

Output
------
  results/variance_explained/
      variance_explained.csv: per-unit rows
      variance_explained_summary.csv: (mean +- std per (model, task, layer_type))
      grouped_bar_layer_type.pdf
      boxplot_by_architecture.pdf
      boxplot_by_layer_depth.pdf
      scatter_ev_vs_d.pdf
"""

import argparse
import os
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from projections.common.layer_decomposition import LayerDecomposition
from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
from shared import load_decompositions, parse_run

RUN_FIELDS = ("model", "task", "method", "path")


def collect_variance(decompositions: dict[str, LayerDecomposition]) -> list[dict]:
    """
    For each layer that has PCA or FourierKPCA decomposer, collect
    per-unit explained variance [C] and return a list of row dicts.
    """
    rows = []
    for layer_key, decomp in decompositions.items():
        ev = getattr(decomp.decomposer, "explained_variance", None)
        if ev is None:
            continue
        ev_np = ev.cpu().numpy()
        layer_depth = _layer_depth_index(layer_key, decompositions)
        d = int(np.prod(decomp.original_shape))
        for unit_idx, val in enumerate(ev_np):
            rows.append({
                "layer_key":          layer_key,
                "layer_type":         decomp.layer_type,
                "layer_depth":        layer_depth,
                "unit_idx":           unit_idx,
                "d":                  d,
                "explained_variance": float(val),
            })
    return rows


def _layer_depth_index(key: str, decompositions: dict) -> int:
    """Return the position of this layer key in the ordered layer list."""
    keys = list(decompositions.keys())
    return keys.index(key) if key in keys else 0


def plot_grouped_bar_layer_type(df: pd.DataFrame, out_dir: str) -> None:
    """
    One subplot column per model, one subplot row per task.
    Within each subplot: grouped bars by layer_type showing mean EV ± std.
    Color encodes layer_type consistently across all subplots.

    This is the figure that justifies k=3 and shows the architecture-specific
    layer-type pattern (depthwise >> linear, segmentation > classification).

    Note: this function has been generated with the help of Claude
    """
    models = sorted(df["model"].unique())
    tasks  = sorted(df["task"].unique())
    layer_types = sorted(df["layer_type"].unique())

    # Consistent color map for layer types
    cmap   = plt.get_cmap("tab10")
    lt_colors = {lt: cmap(i) for i, lt in enumerate(layer_types)}

    # Friendly display names for layer types
    lt_labels = {
        "conv2d":            "Conv2D",
        "depthwise_conv":    "Depthwise Conv",
        "linear":            "Linear",
        "attention_query":   "Attn Query",
        "attention_output":  "Attn Output",
        "attention_kv":      "Attn KV",
    }

    n_rows = len(tasks)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        "Mean explained variance (top-3 PCA components) by layer type",
        fontsize=11, y=1.01,
    )

    for row_idx, task in enumerate(tasks):
        for col_idx, model in enumerate(models):
            ax  = axes[row_idx][col_idx]
            sub = df[(df["model"] == model) & (df["task"] == task)]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Compute mean ± std per layer type for this (model, task)
            agg = (
                sub.groupby("layer_type")["explained_variance"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            present_types = agg["layer_type"].tolist()
            x      = np.arange(len(present_types))
            means  = agg["mean"].values
            stds   = agg["std"].values
            colors = [lt_colors[lt] for lt in present_types]
            xlbls  = [lt_labels.get(lt, lt) for lt in present_types]

            bars = ax.bar(
                x, means,
                yerr=stds,
                color=colors,
                width=0.65,
                capsize=4,
                error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.7},
                alpha=0.85,
            )

            # Reference line at 0.5
            ax.axhline(0.5, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.6, label="0.5")

            ax.set_xticks(x)
            ax.set_xticklabels(xlbls, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.set_ylabel("Mean EV (top-3)", fontsize=8)

            # Title: model on top row, task on left column
            title_parts = []
            if row_idx == 0:
                title_parts.append(display(model, MODEL_DISPLAY))
            if col_idx == 0:
                title_parts.append(display(task, DATASET_DISPLAY))
            if title_parts:
                ax.set_title(" / ".join(title_parts), fontsize=9)
            else:
                ax.set_title(task.replace("_", " "), fontsize=8)

            # Annotate bar tops with mean value
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + 0.03, 0.98),
                    f"{mean:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="black",
                )

    # Shared legend for layer types
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=lt_colors[lt], label=lt_labels.get(lt, lt), alpha=0.85)
        for lt in layer_types
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(layer_types),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "grouped_bar_layer_type.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_compact_by_task(df: pd.DataFrame, out_dir: str) -> None:
    """
    Compact paper figure: one subplot per model (3 side by side).
    X-axis = layer type, grouped bars = one bar per task.
    Color encodes task.

    Directly shows segmentation > classification pattern within each
    layer type for each model, in a space-efficient 1x3 layout.

    Note: this function has been generated with the help of Claude
    """
    models     = sorted(df["model"].unique())
    tasks      = sorted(df["task"].unique())
    layer_types = sorted(df["layer_type"].unique())

    # Task colors
    task_cmap   = plt.get_cmap("Set2")
    task_colors = {t: task_cmap(i) for i, t in enumerate(tasks)}

    # Friendly short task labels
    task_labels = {t: display(t, DATASET_DISPLAY) for t in tasks}
    
    # Friendly layer type labels
    lt_labels = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.5 * n_models, 4.0),
        sharey=True,
        squeeze=False,
    )

    for col_idx, model in enumerate(models):
        ax  = axes[0][col_idx]
        sub = df[df["model"] == model]

        # Layer types present for this model
        present_lts = sorted(sub["layer_type"].unique())
        n_lt        = len(present_lts)
        n_tasks     = len(tasks)
        bar_width   = 0.75 / n_tasks
        x_base      = np.arange(n_lt)

        for t_idx, task in enumerate(tasks):
            task_sub = sub[sub["task"] == task]
            means, stds = [], []
            for lt in present_lts:
                lt_grp = task_sub[task_sub["layer_type"] == lt]["explained_variance"]
                if lt_grp.empty:
                    means.append(0.0)
                    stds.append(0.0)
                else:
                    means.append(lt_grp.mean())
                    stds.append(lt_grp.std())

            offset = (t_idx - n_tasks / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_base + offset,
                means,
                yerr=stds,
                width=bar_width,
                color=task_colors[task],
                label=task_labels[task],
                alpha=0.85,
                capsize=3,
                error_kw={"linewidth": 0.7, "ecolor": "black", "alpha": 0.6},
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [lt_labels.get(lt, lt) for lt in present_lts],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(display(model, MODEL_DISPLAY), fontsize=9)
        ax.axhline(0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax.set_ylabel("Mean EV (top-3 PCA)", fontsize=8)

    # Shared legend for tasks
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=task_colors[t], alpha=0.85,
                       label=task_labels[t])
        for t in tasks
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(tasks),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )

    fig.suptitle(
        "Explained variance by layer type and task",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "compact_bar_layer_type_by_task.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def boxplot_by_architecture(df: pd.DataFrame, out_dir: str) -> None:
    """
        One box per (model, task) combination, pooling all layers and units.

        Note: this function has been generated with the help of Claude
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df.groupby(["model", "task"])
    labels, data = [], []
    for (model, task), grp in groups:
        labels.append(f"{model}\n{task}")
        data.append(grp["explained_variance"].values)

    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.set_ylabel("Explained variance (top-3 components)")
    ax.set_title("Trajectory intrinsic dimensionality by architecture and task")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="0.90 threshold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "boxplot_by_architecture.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def boxplot_by_layer_depth(df: pd.DataFrame, out_dir: str) -> None:
    """
    One subplot per model. X-axis = layer depth bin, Y = explained variance.
    Shows how explained variance changes with depth.

    Note: this function has been generated with the help of Claude
    """
    models = df["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        n_layers = sub["layer_depth"].nunique()
        bin_size = max(1, n_layers // 8)
        sub = sub.copy()
        sub["depth_bin"] = (sub["layer_depth"] // bin_size) * bin_size

        groups = sub.groupby("depth_bin")
        labels = [str(k) for k in sorted(groups.groups.keys())]
        data   = [groups.get_group(k)["explained_variance"].values
                  for k in sorted(groups.groups.keys())]

        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(model)
        ax.set_xlabel("Layer depth (binned)")
        ax.set_ylabel("Explained variance")
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Explained variance vs network depth")
    plt.tight_layout()
    path = os.path.join(out_dir, "boxplot_by_layer_depth.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_ev_d_confound(df: pd.DataFrame, out_dir: str) -> None:
    """
    Confound check: Spearman rank correlation between d (flattened weight
    vector size) and explained_variance, computed per model and globally.

    A low / non-significant correlation rules out d as a confounder,
    i.e. the EV differences we observe across layer types are not simply
    a consequence of units having different parameter counts.
    """
    from scipy.stats import spearmanr

    rows = []
    print("\n --- EV vs d confound check (Spearman ρ) ---")

    # Per model
    for model, grp in df.groupby("model"):
        r, p = spearmanr(grp["d"], grp["explained_variance"])
        n    = len(grp)
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {display(model, MODEL_DISPLAY):<30}  ρ={r:+.3f}  p={p:.2e}  n={n}  {sig}")
        rows.append({"scope": model, "spearman_r": r, "p_value": p, "n": n})

    # Within each (model, layer_type) — d is constant so correlation is undefined
    for (model, lt), grp in df.groupby(["model", "layer_type"]):
        if grp["d"].nunique() < 2:
            rows.append({
                "scope":      f"{model} / {lt}",
                "spearman_r": float("nan"),
                "p_value":    float("nan"),
                "n":          len(grp),
            })
            continue
        r, p = spearmanr(grp["d"], grp["explained_variance"])
        rows.append({
            "scope":      f"{model} / {lt}",
            "spearman_r": r,
            "p_value":    p,
            "n":          len(grp),
        })

    # Global
    r_all, p_all = spearmanr(df["d"], df["explained_variance"])
    sig = "***" if p_all < 0.001 else ("**" if p_all < 0.01 else ("*" if p_all < 0.05 else "ns"))
    print(f"  {'Global (all models)':<30}  ρ={r_all:+.3f}  p={p_all:.2e}  n={len(df)}  {sig}")
    print("─────────────────────────────────────────────────────────────")
    rows.append({"scope": "global", "spearman_r": r_all, "p_value": p_all, "n": len(df)})

    conf_df   = pd.DataFrame(rows)
    conf_path = os.path.join(out_dir, "ev_d_confound_check.csv")
    conf_df.to_csv(conf_path, index=False)
    print(f"  Saved: {conf_path}")


def main():
    ap = argparse.ArgumentParser(description="Variance explained analysis")
    ap.add_argument(
        "--runs", nargs="+", required=True, metavar="MODEL,TASK,METHOD,PATH",
        help="Each entry: model_name,task,method,decomposers_path"
    )
    ap.add_argument("--out_dir", default="./results/variance_explained")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for run_str in args.runs:
        model, task, method, path = parse_run(run_str, RUN_FIELDS)
        if method.lower() == "umap":
            print(f"  Skipping {model}/{task}/umap — no explained variance for UMAP")
            continue

        print(f"Loading {model}/{task}/{method}: {path}")
        decompositions = load_decompositions(path)
        rows = collect_variance(decompositions)

        for r in rows:
            r["model"]  = model
            r["task"]   = task
            r["method"] = method

        all_rows.extend(rows)
        n_layers = len(decompositions)
        mean_ev  = np.mean([r["explained_variance"] for r in rows])
        print(f"  {n_layers} layers, {len(rows)} units, mean EV = {mean_ev:.4f}")

    if not all_rows:
        print("No rows collected — check your --runs arguments.")
        return

    df = pd.DataFrame(all_rows)
    df["model"] = df["model"].map(lambda x: display(x, MODEL_DISPLAY))
    df["task"]  = df["task"].map(lambda x: display(x, DATASET_DISPLAY))

    # Save raw
    csv_path = os.path.join(args.out_dir, "variance_explained.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw CSV: {csv_path}")

    # Summary: mean +- std per (model, task, method, layer_type)
    summary = (
        df.groupby(["model", "task", "method", "layer_type"])["explained_variance"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "ev_mean", "std": "ev_std", "count": "n_units"})
    )

    #summary["model"] = summary["model"].map(lambda x: display(x, MODEL_DISPLAY))
    #summary["task"]  = summary["task"].map(lambda x: display(x, DATASET_DISPLAY))
    summary_path = os.path.join(args.out_dir, "variance_explained_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary CSV: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    # Plots
    plot_grouped_bar_layer_type(df, args.out_dir)   # main paper figure — full grid
    plot_compact_by_task(df, args.out_dir)              # compact version
    boxplot_by_architecture(df, args.out_dir)
    boxplot_by_layer_depth(df, args.out_dir)
    compute_ev_d_confound(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
