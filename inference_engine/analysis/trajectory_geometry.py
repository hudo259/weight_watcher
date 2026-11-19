"""
Analyses the geometry of per-unit training trajectories in the 3D
decomposition space.

Three measurements per unit c in layer L:
  path_length[c]    = sum_t || z[c, t+1] - z[c, t] ||₂
                      Total distance travelled — large = unit changed a lot.
  displacement[c]   = || z[c, T-1] - z[c, 0] ||₂
                      Net displacement — small displacement with large path
                      length means the unit oscillated and returned.
  velocity[c, t]    = || z[c, t+1] - z[c, t] ||₂
                      Step size at each checkpoint — lets you see when
                      each layer "froze" during training.

Requires decomposers to have been produced with --save_trajectory.

Optionally computes transfer displacement: how far each unit moves when
fine-tuned on a downstream task, measured in the shared ImageNet-100
decomposition space. Requires --finetune_projections.

Output
------
  results/trajectory_geometry/
      trajectory_geometry.csv (per-unit path_length + displacement)
      trajectory_geometry_summary.csv (mean +- std per (model, task, layer_type))
      transfer_displacement.csv (per-unit transfer displacement (optional))
      transfer_displacement_summary.csv (mean per (model, downstream_task, layer_depth))
      velocity_heatmap_{model}_{task}.pdf (heatmap [epoch x unit] per run)
      transfer_depth_profile_{model}.pdf (mean transfer displacement vs layer depth)
      boxplot_path_length_norm.pdf
      boxplot_displacement_norm.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from shared import load_decompositions, load_points_3d, parse_run

RUN_FIELDS = ("model", "task", "method", "decomposers_path")

# Loading


def parse_finetune(s: str):
    """Parse 'model,downstream_task,method,projected_path'."""
    parts = s.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Each --finetune_projections entry must be "
            "'model,downstream_task,method,projected_path'"
        )
    return parts


# Geometry computations
def compute_geometry(traj: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        traj: [C, T, k] trajectory tensor
    Returns:
        path_length:  [C]    total distance each unit travels
        displacement: [C]    net distance from start to end
        velocity:     [C, T-1] step size at each checkpoint transition
    """
    C, T, k = traj.shape
    steps = traj[:, 1:, :] - traj[:, :-1, :]   # [C, T-1, k]
    velocity = steps.norm(dim=-1)               # [C, T-1]

    path_length  = velocity.sum(dim=-1)         # [C]
    displacement = (traj[:, -1, :] - traj[:, 0, :]).norm(dim=-1)  # [C]

    return (
        path_length.numpy(),
        displacement.numpy(),
        velocity.numpy(),
    )


# Freeze time
def compute_freeze_time(
    velocity: np.ndarray,
    threshold_frac: float = 0.05,
) -> np.ndarray:
    """
    For each unit, find the first checkpoint step where velocity drops
    below threshold_frac * peak_velocity and stays below it for the
    remainder of training.

    Args:
        velocity:       [C, T-1] step velocities per unit per checkpoint step
        threshold_frac: fraction of peak velocity below which a unit is
                        considered "frozen". Default 0.05 = 5% of peak.

    Returns:
        freeze_step: [C] int array. Value = step index (0-indexed into T-1
                     steps) at which each unit froze. Value = T-1 if the
                     unit never froze (never dropped below threshold).

    Definition of "froze": the last step where velocity > threshold, plus 1.
    If a unit's velocity is always above the threshold it never froze (= T-1).
    If a unit's velocity is always below the threshold it froze at step 0.
    """
    C, T_minus_1 = velocity.shape
    freeze_steps = np.full(C, T_minus_1, dtype=np.float32)

    for c in range(C):
        vel_c  = velocity[c]
        peak   = vel_c.max()
        thresh = threshold_frac * peak
        if peak < 1e-10:
            # Unit never moved — froze at step 0
            freeze_steps[c] = 0.0
            continue
        # Find last step where velocity exceeds threshold
        above = np.where(vel_c > thresh)[0]
        if len(above) == 0:
            freeze_steps[c] = 0.0
        else:
            freeze_steps[c] = float(above[-1] + 1)   # first step it's "done"

    return freeze_steps


def plot_freeze_time_vs_depth(
    df: "pd.DataFrame",
    out_dir: str,
) -> None:
    """
    Stacked panel plot: one figure per model, one panel per task (stacked
    vertically), all sharing the same x-axis (layer depth).

    Each panel shows mean freeze time (fraction of training) vs layer depth,
    with one line per layer type. A value near 0 means the layer froze early;
    near 1 means it kept moving until the end of training.

    Uses display_names.py for model/task/layer_type labels if available.

    Note: This function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY, DATASET_DISPLAY = {}, {}
        def display(raw, mapping): return mapping.get(raw, raw)

    # Friendly layer type labels
    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    # Consistent layer type colors across all panels
    all_lts   = sorted(df["layer_type"].unique())
    lt_cmap   = plt.get_cmap("tab10")
    lt_colors = {lt: lt_cmap(i) for i, lt in enumerate(all_lts)}

    # Task display order
    TASK_ORDER = [
        "classification_imagenet100",
        "classification_isic2019",
        "segmentation_isic2018",
        "segmentation_voc2012_sbd",
    ]

    models = sorted(df["model"].unique())

    for model in models:
        model_df = df[df["model"] == model]
        tasks_present = [t for t in TASK_ORDER if t in model_df["task"].unique()]
        # Any tasks not in TASK_ORDER appended at end
        tasks_present += [t for t in sorted(model_df["task"].unique())
                          if t not in tasks_present]
        n_tasks = len(tasks_present)
        if n_tasks == 0:
            continue

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(7, 2.2 * n_tasks))
        gs  = gridspec.GridSpec(
            n_tasks, 1,
            hspace=0.06,
            figure=fig,
        )
        axes = [fig.add_subplot(gs[i]) for i in range(n_tasks)]
        for ax in axes[1:]:
            ax.sharex(axes[0])

        legend_handles = {}   # collect for shared legend
        all_profiles   = []   # collect to compute global y range for this model

        # First pass: compute all profiles and global y range
        for task in tasks_present:
            grp = model_df[model_df["task"] == task].copy()
            if "n_checkpoints" in grp.columns:
                grp["freeze_frac"] = (
                    grp["freeze_step"] /
                    (grp["n_checkpoints"] - 1).clip(lower=1)
                )
            else:
                grp["freeze_frac"] = grp["freeze_step"]
            profile = (
                grp.groupby(["layer_depth", "layer_type"])["freeze_frac"]
                .mean()
                .reset_index()
                .sort_values("layer_depth")
            )
            all_profiles.append((task, profile))

        # Adaptive y range across all tasks for this model
        all_vals = np.concatenate([p["freeze_frac"].values for _, p in all_profiles])
        y_min = max(0.0, float(all_vals.min()) - 0.05)
        y_max = min(1.0, float(all_vals.max()) + 0.05)
        # Generate 3 evenly spaced yticks within the range
        y_ticks = np.linspace(y_min, y_max, 3)

        # Second pass: draw panels
        for row_idx, (task, profile) in enumerate(all_profiles):
            ax = axes[row_idx]

            for lt, sub in profile.groupby("layer_type"):
                color = lt_colors.get(lt, "gray")
                label = LT_LABELS.get(lt, lt)
                line, = ax.plot(
                    sub["layer_depth"], sub["freeze_frac"],
                    marker="o", markersize=2.5,
                    linewidth=1.2,
                    color=color,
                    label=label,
                )
                legend_handles[label] = line

            ax.axhline(0.5, color="gray", linewidth=0.5,
                       linestyle="--", alpha=0.6)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{v:.2f}" for v in y_ticks], fontsize=7)

            # Task label as y-axis title on right side
            task_label = display(task, DATASET_DISPLAY)
            ax.set_ylabel(task_label, fontsize=8, rotation=0,
                          ha="left", va="center", labelpad=4)
            ax.yaxis.set_label_position("right")

            # Hide x tick labels on all but bottom panel
            if row_idx < n_tasks - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Layer index (depth)", fontsize=9)

            ax.tick_params(axis="x", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Title just above top panel — use axes position not fig y=1.01
        axes[0].set_title(
            f"{display(model, MODEL_DISPLAY)} — freeze time vs layer depth",
            fontsize=10, pad=4,
        )

        # Shared legend just below the bottom panel, tight
        if legend_handles:
            bottom_ax_pos = axes[-1].get_position()
            fig.legend(
                handles=list(legend_handles.values()),
                labels=list(legend_handles.keys()),
                loc="upper center",
                ncol=len(legend_handles),
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, bottom_ax_pos.y0 - 0.01),
                bbox_transform=fig.transFigure,
            )

        plt.tight_layout()
        safe_model = model.replace(" ", "_")
        p = os.path.join(out_dir, f"freeze_time_vs_depth_{safe_model}.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

# Analysis
def analyze_run(model, task, method, decomposers_path, freeze_threshold: float = 0.05) -> tuple[list[dict], dict]:
    print(f"\nLoading {model}/{task}/{method}: {decomposers_path}")
    decompositions = load_decompositions(decomposers_path)

    unit_rows   = []
    layer_velocities = {}  # layer_key → [C, T-1] numpy

    for depth_idx, (key, decomp) in enumerate(decompositions.items()):
        traj = decomp.trajectory_points_3d   # [C, T, k] or None
        if traj is None:
            print(f"  WARNING: no trajectory for {key} — "
                  f"re-run projection.py with --save_trajectory")
            continue

        pl, disp, vel = compute_geometry(traj)
        C = traj.shape[0]
        T = traj.shape[1]

        # d = original flattened component dimensionality before decomposition.
        # Normalization factor: sqrt(d) prevents confounding of cross-architecture
        # comparisons by different per-unit parameter counts.
        d      = int(np.prod(decomp.original_shape))
        d_sqrt = np.sqrt(max(d, 1))

        layer_velocities[key] = vel   # [C, T-1]

        # Freeze time: first checkpoint step where velocity drops below
        # freeze_threshold_frac * peak_velocity for each unit.
        freeze_steps = compute_freeze_time(vel, threshold_frac=freeze_threshold)

        for unit_idx in range(C):
            unit_rows.append({
                "model":              model,
                "task":               task,
                "method":             method,
                "layer_key":          key,
                "layer_type":         decomp.layer_type,
                "layer_depth":        depth_idx,
                "n_checkpoints":      T,
                "unit_idx":           unit_idx,
                "d":                  d,
                "path_length":        float(pl[unit_idx]),
                "path_length_norm":   float(pl[unit_idx] / d_sqrt),
                "displacement":       float(disp[unit_idx]),
                "displacement_norm":  float(disp[unit_idx] / d_sqrt),
                "ratio":              float(disp[unit_idx] / (pl[unit_idx] + 1e-8)),
                "freeze_step":        float(freeze_steps[unit_idx]),
            })

        mean_freeze = freeze_steps.mean()
        print(f"  [{depth_idx:3d}] {key[:50]:<50}  "
              f"C={C}  T={T}  d={d}  "
              f"mean_path_norm={pl.mean()/d_sqrt:.4f}  "
              f"mean_freeze={mean_freeze:.1f}/{T-1}")

    return unit_rows, layer_velocities

# Velocity heatmap
def plot_ratio_displacement_scatter(df: "pd.DataFrame", out_dir: str) -> None:
    """
    For each model: one subplot per layer type, scatter of
    displacement_norm (x) vs ratio (y), points colored by task.

    Four quadrants reveal distinct unit behaviors:
      - High ratio, high displacement: directional + large learning
      - High ratio, low displacement:  directional + conservative
      - Low ratio,  high displacement: oscillatory + active
      - Low ratio,  low displacement:  barely moved (frozen/redundant)

    Note: This function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY, DATASET_DISPLAY = {}, {}
        def display(raw, mapping): return mapping.get(raw, raw)

    TASK_ORDER = [
        "classification_imagenet100",
        "classification_isic2019",
        "segmentation_isic2018",
        "segmentation_voc2012_sbd",
    ]

    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    # Task colors — consistent across all figures
    all_tasks   = [t for t in TASK_ORDER if t in df["task"].unique()]
    all_tasks  += [t for t in sorted(df["task"].unique()) if t not in all_tasks]
    task_cmap   = plt.get_cmap("Set2")
    task_colors = {t: task_cmap(i) for i, t in enumerate(all_tasks)}

    models = sorted(df["model"].unique())

    for model in models:
        model_df     = df[df["model"] == model]
        layer_types  = sorted(model_df["layer_type"].unique())
        n_lt         = len(layer_types)

        # Use per-layer mean so we plot layer-level aggregates, not raw units
        # (too many points otherwise). Group by (task, layer_depth, layer_type).
        agg = (
            model_df.groupby(["task", "layer_depth", "layer_type"])
            [["displacement_norm", "ratio"]]
            .mean()
            .reset_index()
        )

        n_cols = min(n_lt, 3)
        n_rows = int(np.ceil(n_lt / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.0 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )

        legend_handles = {}

        for idx, lt in enumerate(layer_types):
            ax      = axes[idx // n_cols][idx % n_cols]
            lt_data = agg[agg["layer_type"] == lt]

            for task in all_tasks:
                task_data = lt_data[lt_data["task"] == task]
                if task_data.empty:
                    continue
                label  = display(task, DATASET_DISPLAY)
                color  = task_colors[task]
                sc = ax.scatter(
                    task_data["displacement_norm"],
                    task_data["ratio"],
                    c=[color],
                    s=18,
                    alpha=0.75,
                    label=label,
                    edgecolors="none",
                )
                legend_handles[label] = sc

            # Quadrant reference lines
            ax.axhline(0.5, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.5)
            x_mid = float(agg["displacement_norm"].median())
            ax.axvline(x_mid, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.5)

            ax.set_title(LT_LABELS.get(lt, lt), fontsize=9)
            ax.set_xlabel("Displacement (norm.)", fontsize=8)
            ax.set_ylabel("Directionality ratio", fontsize=8)
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused subplots
        for idx in range(n_lt, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        # Shared legend
        if legend_handles:
            fig.legend(
                handles=list(legend_handles.values()),
                labels=list(legend_handles.keys()),
                loc="lower center",
                ncol=len(legend_handles),
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, -0.04),
            )

        fig.suptitle(
            f"{display(model, MODEL_DISPLAY)} — directionality vs displacement",
            fontsize=10,
        )
        plt.tight_layout()
        safe_model = model.replace(" ", "_")
        p = os.path.join(out_dir, f"ratio_displacement_{safe_model}.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


def plot_ratio_displacement_compact(df: "pd.DataFrame", out_dir: str) -> None:
    """
    Compact version: one figure per model, 2x2 grid of tasks.
    Each panel: all layer types as colored dots (displacement_norm vs ratio).
    Color = layer type, consistent across panels.
    Directly shows segmentation vs classification contrast and
    layer-type clustering within each task.

    Note: This function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY, DATASET_DISPLAY = {}, {}
        def display(raw, mapping): return mapping.get(raw, raw)

    TASK_ORDER = [
        "classification_imagenet100",
        "classification_isic2019",
        "segmentation_isic2018",
        "segmentation_voc2012_sbd",
    ]

    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    # Layer type colors — consistent across all figures
    all_lts   = sorted(df["layer_type"].unique())
    lt_cmap   = plt.get_cmap("tab10")
    lt_colors = {lt: lt_cmap(i) for i, lt in enumerate(all_lts)}

    models = sorted(df["model"].unique())

    for model in models:
        model_df = df[df["model"] == model]

        # Layer-level means
        agg = (
            model_df.groupby(["task", "layer_depth", "layer_type"])
            [["displacement_norm", "ratio"]]
            .mean()
            .reset_index()
        )

        tasks_present = [t for t in TASK_ORDER if t in agg["task"].unique()]
        tasks_present += [t for t in sorted(agg["task"].unique())
                          if t not in tasks_present]
        n_tasks = len(tasks_present)

        # Layout: 2 columns, ceil(n_tasks/2) rows
        n_cols = 2
        n_rows = int(np.ceil(n_tasks / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.2 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )

        legend_handles = {}

        for idx, task in enumerate(tasks_present):
            ax       = axes[idx // n_cols][idx % n_cols]
            task_agg = agg[agg["task"] == task]

            for lt in sorted(task_agg["layer_type"].unique()):
                lt_data = task_agg[task_agg["layer_type"] == lt]
                label   = LT_LABELS.get(lt, lt)
                color   = lt_colors[lt]
                sc = ax.scatter(
                    lt_data["displacement_norm"],
                    lt_data["ratio"],
                    c=[color],
                    s=22,
                    alpha=0.80,
                    label=label,
                    edgecolors="none",
                )
                legend_handles[label] = sc

            # Quadrant reference lines
            ax.axhline(0.5, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.5)
            x_mid = float(agg["displacement_norm"].median())
            ax.axvline(x_mid, color="gray", linewidth=0.6,
                       linestyle="--", alpha=0.5)

            # Quadrant annotations
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.text(0.02, 0.97, "directional, small Δ",
                    transform=ax.transAxes, fontsize=6,
                    va="top", ha="left", color="black", alpha=0.7)
            ax.text(0.98, 0.97, "directional, large Δ",
                    transform=ax.transAxes, fontsize=6,
                    va="top", ha="right", color="black", alpha=0.7)
            ax.text(0.02, 0.03, "oscillatory, small Δ",
                    transform=ax.transAxes, fontsize=6,
                    va="bottom", ha="left", color="black", alpha=0.7)
            ax.text(0.98, 0.03, "oscillatory, large Δ",
                    transform=ax.transAxes, fontsize=6,
                    va="bottom", ha="right", color="black", alpha=0.7)

            task_label = display(task, DATASET_DISPLAY)
            ax.set_title(task_label, fontsize=9)
            ax.set_xlabel("Displacement (norm.)", fontsize=8)
            ax.set_ylabel("Directionality ratio", fontsize=8)
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused subplots
        for idx in range(n_tasks, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        # Shared legend
        if legend_handles:
            fig.legend(
                handles=list(legend_handles.values()),
                labels=list(legend_handles.keys()),
                loc="lower center",
                ncol=len(legend_handles),
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, -0.04),
            )

        fig.suptitle(
            f"{display(model, MODEL_DISPLAY)}: directionality vs displacement",
            fontsize=10,
        )
        plt.tight_layout()
        safe_model = model.replace(" ", "_")
        p = os.path.join(out_dir, f"ratio_displacement_compact_{safe_model}.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


def plot_ratio_displacement_summary(df: "pd.DataFrame", out_dir: str) -> None:
    """
    Single-figure summary: 3 subplots side by side, one per model.
    Scatter of displacement_norm (x) vs ratio (y).
    Color  = layer type (consistent with other figures).
    Marker = task type: circle for classification, triangle for segmentation.

    This is the main paper figure, one compact view showing both the
    layer-type clustering and the segmentation vs classification contrast.

    Note: This function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY, DATASET_DISPLAY = {}, {}
        def display(raw, mapping): return mapping.get(raw, raw)

    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    # Layer type colors
    all_lts   = sorted(df["layer_type"].unique())
    lt_cmap   = plt.get_cmap("tab10")
    lt_colors = {lt: lt_cmap(i) for i, lt in enumerate(all_lts)}

    # Marker per task type
    def task_marker(task: str) -> str:
        return "x" if "segmentation" in task else "o"

    models = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.5 * n_models, 4.0),
        squeeze=False,
    )

    lt_legend_handles  = {}
    cls_handle  = None
    seg_handle  = None

    for col_idx, model in enumerate(models):
        ax       = axes[0][col_idx]
        model_df = df[df["model"] == model]

        # Layer-level means
        agg = (
            model_df.groupby(["task", "layer_depth", "layer_type"])
            [["displacement_norm", "ratio"]]
            .mean()
            .reset_index()
        )

        for lt in sorted(agg["layer_type"].unique()):
            lt_data = agg[agg["layer_type"] == lt]
            color   = lt_colors[lt]
            label   = LT_LABELS.get(lt, lt)

            for task in sorted(lt_data["task"].unique()):
                task_data = lt_data[lt_data["task"] == task]
                marker    = task_marker(task)
                sc = ax.scatter(
                    task_data["displacement_norm"],
                    task_data["ratio"],
                    c=[color],
                    marker=marker,
                    s=20,
                    alpha=0.60,
                    edgecolors="none",
                )
                # Collect legend handles
                if label not in lt_legend_handles:
                    lt_legend_handles[label] = ax.scatter(
                        [], [], c=[color], marker="o", s=25,
                        label=label, edgecolors="none",
                    )
                if "segmentation" in task and seg_handle is None:
                    seg_handle = ax.scatter(
                        [], [], c="black", marker="x", s=25,
                        label="Segmentation", edgecolors="none",
                    )
                if "classification" in task and cls_handle is None:
                    cls_handle = ax.scatter(
                        [], [], c="black", marker="o", s=25,
                        label="Classification", edgecolors="none",
                    )

        # Quadrant lines
        ax.axhline(0.5, color="gray", linewidth=0.6,
                   linestyle="--", alpha=0.5)
        x_mid = float(agg["displacement_norm"].median())
        ax.axvline(x_mid, color="gray", linewidth=0.6,
                   linestyle="--", alpha=0.5)

        ax.text(0.02, 0.97, "directional, small Δ",
                transform=ax.transAxes, fontsize=6,
                va="top", ha="left", color="black", alpha=0.7)
        ax.text(0.98, 0.97, "directional, large Δ",
                transform=ax.transAxes, fontsize=6,
                va="top", ha="right", color="black", alpha=0.7)
        ax.text(0.02, 0.03, "oscillatory, small Δ",
                transform=ax.transAxes, fontsize=6,
                va="bottom", ha="left", color="black", alpha=0.7)
        ax.text(0.98, 0.03, "oscillatory, large Δ",
                transform=ax.transAxes, fontsize=6,
                va="bottom", ha="right", color="black", alpha=0.7)

        ax.set_title(display(model, MODEL_DISPLAY), fontsize=9)
        ax.set_xlabel("Displacement (norm.)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Directionality ratio", fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Two-section legend: layer types + task type markers
    type_handles  = list(lt_legend_handles.values())
    shape_handles = [h for h in [cls_handle, seg_handle] if h is not None]

    from matplotlib.lines import Line2D
    separator = Line2D([], [], color="none", label=" ")

    fig.legend(
        handles=type_handles + [separator] + shape_handles,
        loc="lower center",
        ncol=len(type_handles) + 1 + len(shape_handles),
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Directionality vs displacement across architectures and tasks",
        fontsize=10,
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "ratio_displacement_summary.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_ratio_displacement_aggregated(df: "pd.DataFrame", out_dir: str) -> None:
    """
    Main paper figure — maximally aggregated version.
    One figure, 3 subplots side by side (one per model).

    Each point = one (task, layer_type) combination, i.e. the mean across
    ALL layers and units of that type for that task.
    Marker size = number of layers of that type (log-scaled).
    Color  = layer type.
    Marker = task type: 'o' classification, 'x' segmentation.

    Uncluttered: at most ~8-10 points per subplot.

    Note: this function has been generated with the help of Claude
    """
    try:
        from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
    except ImportError:
        MODEL_DISPLAY, DATASET_DISPLAY = {}, {}
        def display(raw, mapping): return mapping.get(raw, raw)

    LT_LABELS = {
        "conv2d":           "Conv2D",
        "depthwise_conv":   "DW Conv",
        "linear":           "Linear",
        "attention_query":  "Attn-Q",
        "attention_output": "Attn-O",
        "attention_kv":     "Attn-KV",
    }

    all_lts   = sorted(df["layer_type"].unique())
    lt_cmap   = plt.get_cmap("tab10")
    lt_colors = {lt: lt_cmap(i) for i, lt in enumerate(all_lts)}

    def task_marker(task: str) -> str:
        return "x" if "segmentation" in task else "o"

    models   = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.5 * n_models, 4.2),
        squeeze=False,
    )

    lt_legend_handles  = {}
    cls_handle  = None
    seg_handle  = None

    for col_idx, model in enumerate(models):
        ax       = axes[0][col_idx]
        model_df = df[df["model"] == model]

        # Aggregate to (task, layer_type) mean — one point per combination
        agg = (
            model_df.groupby(["task", "layer_type"])
            .agg(
                displacement_norm=("displacement_norm", "mean"),
                ratio=("ratio", "mean"),
                n_layers=("layer_depth", "nunique"),
            )
            .reset_index()
        )

        # Marker size: log-scaled by number of layers, range 40-200
        n_min = agg["n_layers"].min()
        n_max = agg["n_layers"].max()
        def scale_size(n):
            if n_max == n_min:
                return 100
            return 40 + 160 * (np.log1p(n - n_min) / np.log1p(n_max - n_min))

        for _, row in agg.iterrows():
            lt     = row["layer_type"]
            task   = row["task"]
            color  = lt_colors[lt]
            marker = task_marker(task)
            size   = scale_size(row["n_layers"])
            label  = LT_LABELS.get(lt, lt)

            ax.scatter(
                row["displacement_norm"],
                row["ratio"],
                c=[color],
                marker=marker,
                s=size,
                alpha=0.90,
                edgecolors="black",
                linewidths=0.4,
            )

            # Build legend handles once
            if label not in lt_legend_handles:
                lt_legend_handles[label] = ax.scatter(
                    [], [], c=[color], marker="o", s=60,
                    label=label, edgecolors="black", linewidths=0.4,
                )
            if "segmentation" in task and seg_handle is None:
                seg_handle = ax.scatter(
                    [], [], c="black", marker="x", s=60,
                    label="Segmentation", edgecolors="black", linewidths=0.4,
                )
            if "classification" in task and cls_handle is None:
                cls_handle = ax.scatter(
                    [], [], c="black", marker="o", s=60,
                    label="Classification", edgecolors="black", linewidths=0.4,
                )

        # Quadrant lines
        ax.axhline(0.5, color="gray", linewidth=0.6,
                   linestyle="--", alpha=0.5)
        x_mid = float(agg["displacement_norm"].median())
        ax.axvline(x_mid, color="gray", linewidth=0.6,
                   linestyle="--", alpha=0.5)

        ax.text(0.02, 0.97, "directional, small Δ",
                transform=ax.transAxes, fontsize=6,
                va="top", ha="left", color="black", alpha=0.7)
        ax.text(0.98, 0.97, "directional, large Δ",
                transform=ax.transAxes, fontsize=6,
                va="top", ha="right", color="black", alpha=0.7)
        ax.text(0.02, 0.03, "oscillatory, small Δ",
                transform=ax.transAxes, fontsize=6,
                va="bottom", ha="left", color="black", alpha=0.7)
        ax.text(0.98, 0.03, "oscillatory, large Δ",
                transform=ax.transAxes, fontsize=6,
                va="bottom", ha="right", color="black", alpha=0.7)

        ax.set_title(display(model, MODEL_DISPLAY), fontsize=9)
        ax.set_xlabel("Displacement (norm.)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Directionality ratio", fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Two-section legend
    from matplotlib.lines import Line2D
    separator    = Line2D([], [], color="none", label=" ")
    type_handles = list(lt_legend_handles.values())
    shape_handles = [h for h in [cls_handle, seg_handle] if h is not None]

    fig.legend(
        handles=type_handles + [separator] + shape_handles,
        loc="lower center",
        ncol=len(type_handles) + 1 + len(shape_handles),
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Directionality vs displacement aggregated by layer type and task",
        fontsize=10,
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "ratio_displacement_aggregated.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# Box plots
def boxplot_metric(df: pd.DataFrame, metric: str, out_dir: str) -> None:
    """
    Note: This function has been generated with the help of Claude
    """
    groups = df.groupby(["model", "task"])
    labels, data = [], []
    for (model, task), grp in groups:
        labels.append(f"{model}\n{task}")
        data.append(grp[metric].dropna().values)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels)), 5))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} distribution by run")
    plt.tight_layout()
    path = os.path.join(out_dir, f"boxplot_{metric}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

# Transfer displacement
def compute_transfer_displacement(
    decompositions: dict,
    finetune_projs: dict,
    model: str,
    downstream_task: str,
    method: str,
) -> list:
    """
    For each unit, compute ||z_finetune - z_pretrain|| in the shared
    ImageNet-100 decomposition space.

    pretrain_z = trajectory_points_3d[:, -1, :] (final ImageNet-100 checkpoint)
    finetune_z = finetune_projs[layer_key] (fine-tuned checkpoint projected into the same decomposition space)

    Both must use the same fitted decomposer so coordinate spaces are comparable.
    """
    rows = []
    for depth_idx, (key, decomp) in enumerate(decompositions.items()):
        if key not in finetune_projs:
            print(f"  WARNING: {key} not in finetune projections! Skipping...")
            continue
        traj = decomp.trajectory_points_3d
        if traj is None:
            print(f"  WARNING: no trajectory for {key}! Skipping transfer disp...")
            continue

        pretrain_z = traj[:, -1, :].numpy() # [C, k]
        finetune_z = finetune_projs[key] # [C, k]

        if pretrain_z.shape != finetune_z.shape:
            print(f"  WARNING: shape mismatch for {key}: {pretrain_z.shape} vs {finetune_z.shape}! Skipping...")
            continue

        transfer_disp = np.linalg.norm(finetune_z - pretrain_z, axis=-1)  # [C]
        d      = int(np.prod(decomp.original_shape))
        d_sqrt = np.sqrt(max(d, 1))
        C      = pretrain_z.shape[0]

        for unit_idx in range(C):
            rows.append({
                "model":                      model,
                "downstream_task":            downstream_task,
                "method":                     method,
                "layer_key":                  key,
                "layer_type":                 decomp.layer_type,
                "layer_depth":                depth_idx,
                "unit_idx":                   unit_idx,
                "d":                          d,
                "transfer_displacement":      float(transfer_disp[unit_idx]),
                "transfer_displacement_norm": float(transfer_disp[unit_idx] / d_sqrt),
            })

        print(f"  [{depth_idx:3d}] {key[:50]:<50}  "
              f"mean={transfer_disp.mean():.4f}  "
              f"norm={transfer_disp.mean()/d_sqrt:.4f}")
    return rows

def plot_transfer_depth_profile(df: "pd.DataFrame", out_dir: str) -> None:
    """
    Mean transfer_displacement_norm vs layer depth, one line per downstream
    task. Tests the 'early layers transfer, late layers adapt' hypothesis.

    Note: this function has been generated with the help of Claude
    """
    for model, sub in df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for task, tsub in sub.groupby("downstream_task"):
            profile = (
                tsub.groupby("layer_depth")["transfer_displacement_norm"]
                .mean()
                .reset_index()
                .sort_values("layer_depth")
            )
            ax.plot(profile["layer_depth"], profile["transfer_displacement_norm"],
                    marker="o", markersize=3, label=task)
        ax.set_xlabel("Layer depth")
        ax.set_ylabel("Mean transfer displacement / sqrt(d)")
        ax.set_title(
            f"{model} -- transfer displacement vs layer depth\n"
            f"(small = layer preserved under fine-tuning, large = layer adapted)"
        )
        ax.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"transfer_depth_profile_{model}.pdf")
        fig.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")

def main():
    ap = argparse.ArgumentParser(description="Trajectory geometry analysis")
    ap.add_argument(
        "--runs", nargs="+", required=True,
        metavar="MODEL,TASK,METHOD,DECOMPOSERS_PATH",)
    ap.add_argument("--out_dir", default="./results/trajectory_geometry")
    ap.add_argument(
        "--freeze_threshold", type=float, default=0.05,
        help="Fraction of peak velocity below which a unit is considered frozen "
             "(default: 0.05 = 5%% of peak velocity).")
    ap.add_argument(
        "--finetune_projections", nargs="*", default=[],
        metavar="MODEL,DOWNSTREAM_TASK,METHOD,PROJECTED_PATH",
        help="Optional. Compute transfer displacement for each entry. "
             "PROJECTED_PATH must be a *_projected.pt produced by running "
             "projection.py with the ImageNet-100 decomposers on the fine-tuned checkpoint.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Cache decompositions keyed by (model, method) for transfer displacement reuse
    run_decompositions: dict = {}
    all_unit_rows = []

    for run_str in args.runs:
        model, task, method, path = parse_run(run_str, RUN_FIELDS)
        unit_rows, layer_velocities = analyze_run(
            model, task, method, path,
            freeze_threshold=args.freeze_threshold,
        )

        all_unit_rows.extend(unit_rows)
        run_decompositions[(model, method)] = load_decompositions(path)

    if not all_unit_rows:
        print("No rows — did you run projection.py with --save_trajectory?")
        return

    df = pd.DataFrame(all_unit_rows)

    csv_path = os.path.join(args.out_dir, "trajectory_geometry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw CSV: {csv_path}")

    summary = (
        df.groupby(["model", "task", "method", "layer_type"])[
            ["path_length", "path_length_norm",
             "displacement", "displacement_norm", "ratio", "freeze_step"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary_path = os.path.join(args.out_dir, "trajectory_geometry_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    # Normalized columns for cross-architecture comparison, avoids confounding
    # by different per-unit parameter counts (d). Raw columns also saved.
    boxplot_metric(df, "path_length_norm",  args.out_dir)
    boxplot_metric(df, "displacement_norm", args.out_dir)
    boxplot_metric(df, "ratio",             args.out_dir)
    boxplot_metric(df, "path_length",       args.out_dir)
    boxplot_metric(df, "displacement",      args.out_dir)

    # Transfer displacement analysis (optional)
    # Freeze time depth profile
    if "freeze_step" in df.columns:
        plot_freeze_time_vs_depth(df, args.out_dir)

    plot_ratio_displacement_scatter(df, args.out_dir)
    plot_ratio_displacement_compact(df, args.out_dir)
    plot_ratio_displacement_summary(df, args.out_dir)
    plot_ratio_displacement_aggregated(df, args.out_dir)

    if args.finetune_projections:
        all_transfer_rows = []

        for ft_str in args.finetune_projections:
            model, downstream_task, method, ft_path = parse_finetune(ft_str)
            key = (model, method)

            if key not in run_decompositions:
                print(f"\nWARNING: no decompositions loaded for {model}/{method}. Make sure a matching --runs entry exists.")
                continue

            print(f"\nComputing transfer displacement: {model} pretrain -> {downstream_task} ({method})")
            finetune_projs = load_points_3d(ft_path)
            transfer_rows  = compute_transfer_displacement(
                decompositions=run_decompositions[key],
                finetune_projs=finetune_projs,
                model=model,
                downstream_task=downstream_task,
                method=method,
            )
            all_transfer_rows.extend(transfer_rows)

        if all_transfer_rows:
            tdf = pd.DataFrame(all_transfer_rows)

            t_csv = os.path.join(args.out_dir, "transfer_displacement.csv")
            tdf.to_csv(t_csv, index=False)
            print(f"\nSaved: {t_csv}")

            t_summary = (
                tdf.groupby(
                    ["model", "downstream_task", "method", "layer_depth", "layer_type"]
                )[["transfer_displacement", "transfer_displacement_norm"]]
                .agg(["mean", "std"])
                .reset_index()
            )
            t_summary.columns = ["_".join(c).strip("_") for c in t_summary.columns]
            t_sum_csv = os.path.join(args.out_dir, "transfer_displacement_summary.csv")
            t_summary.to_csv(t_sum_csv, index=False)
            print(f"Saved: {t_sum_csv}")

            plot_transfer_depth_profile(tdf, args.out_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()
