"""
Cross-referencing cluster analysis with sensitivity. If geometrically isolated units are more sensitive to perturbation, we can establish 
a direct link between weight space geometry and functional importance. 
This is what elevates the paper from "here are some geometric observations" to "geometry predicts function." 
Concretely: after running both scripts, compute Spearman correlation between each unit's silhouette score (from cluster analysis) 
and its sensitivity score. The same is done for path length vs sensitivity. 

The central question: do units that are geometrically distinctive also tend
to be functionally important?

Three correlations are computed per (model, task, method):
  1. Silhouette score (cluster isolation) vs sensitivity
     Hypothesis: isolated units (high silhouette) are more sensitive ->
     no nearby neighbours to compensate if the unit is moved.
  2. Path length norm vs sensitivity
     Hypothesis: units that moved a lot during training are more sensitive ->
     they ended up in a specific location that the model depends on.
  3. Displacement norm vs sensitivity
     Hypothesis: units with large net displacement from init are more
     sensitive -> they drifted far from their initial configuration to
     fulfil a specific role.

All correlations use Spearman rank correlation

Inputs
------
The three CSV files produced by the other analysis scripts:
  - results/cluster_analysis/cluster_analysis.csv
  - results/trajectory_geometry/trajectory_geometry.csv
  - results/sensitivity/sensitivity.csv

NOTE: All three must have been produced using the same decomposers file so that
layer_key and unit_idx are consistent across CSVs.

Output
------
  results/geometry_function_correlation/
      geometry_function_correlation.csv (per-unit joined table)
      geometry_function_correlation_stats.csv (Spearman r + p per (model, task))
      scatter_silhouette_vs_sensitivity.pdf
      scatter_pathlength_vs_sensitivity.pdf
      scatter_displacement_vs_sensitivity.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Correlation helpers
def spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman r and p-value, returning (nan, nan) if too few valid pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


# Scatter plot helper
def scatter_geo_vs_sensitivity(
    df: pd.DataFrame,
    geo_col: str,
    geo_label: str,
    out_dir: str,
) -> None:
    """
    One subplot per (model, task) combination.
    X = geometric metric, Y = sensitivity.
    Points coloured by layer_type.

    Note: This function has been generated with the help of Claude
    """
    runs    = df.groupby(["model", "task"])
    n_runs  = len(runs)
    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(5 * n_runs, 4),
        squeeze=False,
    )

    layer_types = df["layer_type"].unique()
    cmap        = plt.cm.tab10
    type_colors = {t: cmap(i / max(len(layer_types) - 1, 1))
                   for i, t in enumerate(layer_types)}

    for col_idx, ((model, task), grp) in enumerate(runs):
        ax = axes[0][col_idx]
        for lt, sub in grp.groupby("layer_type"):
            valid = sub[[geo_col, "sensitivity"]].dropna()
            ax.scatter(
                valid[geo_col], valid["sensitivity"],
                label=lt, color=type_colors[lt],
                s=6, alpha=0.5,
            )

        # Spearman annotation
        r, p = spearman_safe(grp[geo_col], grp["sensitivity"])
        ax.set_title(f"{model}/{task}\nr={r:.3f}  p={p:.3f}", fontsize=9)
        ax.set_xlabel(geo_label)
        ax.set_ylabel("Sensitivity (gradient norm)")
        ax.axhline(0, color="black", linewidth=0.5)

        if col_idx == 0:
            ax.legend(fontsize=7, markerscale=2)

    fname = f"scatter_{geo_col}_vs_sensitivity.pdf"
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# Main
def main():
    ap = argparse.ArgumentParser(
        description="Cross-reference geometry metrics with functional sensitivity"
    )
    ap.add_argument("--cluster_csv",     required=True,
                    help="cluster_analysis.csv from cluster_analysis.py")
    ap.add_argument("--trajectory_csv",  required=True,
                    help="trajectory_geometry.csv from trajectory_geometry.py")
    ap.add_argument("--sensitivity_csv", required=True,
                    help="sensitivity.csv from sensitivity.py")
    ap.add_argument("--out_dir", default="./results/geometry_function_correlation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load CSVs ----
    print("Loading CSVs...")
    cluster_df     = pd.read_csv(args.cluster_csv)
    trajectory_df  = pd.read_csv(args.trajectory_csv)
    sensitivity_df = pd.read_csv(args.sensitivity_csv)

    print(f"  cluster:     {len(cluster_df)} rows")
    print(f"  trajectory:  {len(trajectory_df)} rows")
    print(f"  sensitivity: {len(sensitivity_df)} rows")

    # ---- Select columns needed from each CSV ----
    cluster_cols = [
        "model", "task", "method", "layer_key", "layer_type",
        "layer_depth", "unit_idx",
        "silhouette_kmeans", "silhouette_kmeans_unit", "silhouette_dbscan",
        "n_clusters_dbscan",
    ]
    trajectory_cols = [
        "model", "task", "method", "layer_key", "layer_type",
        "layer_depth", "unit_idx",
        "path_length_norm", "displacement_norm", "ratio", "d",
    ]
    sensitivity_cols = [
        "model", "task", "layer_key", "layer_type", "layer_depth",
        "unit_idx", "sensitivity",
    ]

    # Keep only columns that actually exist (graceful degradation)
    cluster_df    = cluster_df[[c for c in cluster_cols    if c in cluster_df.columns]]
    trajectory_df = trajectory_df[[c for c in trajectory_cols if c in trajectory_df.columns]]
    sensitivity_df = sensitivity_df[[c for c in sensitivity_cols if c in sensitivity_df.columns]]

    if "model" in sensitivity_df.columns and "task" in sensitivity_df.columns:
        sens_model = sensitivity_df["model"].iloc[0]
        sens_task  = sensitivity_df["task"].iloc[0]
        print(f"  Filtering to model={sens_model}, task={sens_task}")
        cluster_df    = cluster_df[
            (cluster_df["model"] == sens_model) &
            (cluster_df["task"]  == sens_task)
        ]
        trajectory_df = trajectory_df[
            (trajectory_df["model"] == sens_model) &
            (trajectory_df["task"]  == sens_task)
        ]
        # Dynamically include model and task in the join keys to avoid duplicates
        join_keys = ["model", "task", "layer_key", "unit_idx"]
    else:
        # sensitivity_df has no model/task columns. it's per-run (one model, one task).
        # Join on layer_key + unit_idx only; model/task come from cluster/trajectory.
        join_keys = ["layer_key", "unit_idx"]


    df = cluster_df.merge(
        trajectory_df.drop(columns=["layer_type", "layer_depth"], errors="ignore"),
        on=["model", "task", "method", "layer_key", "unit_idx"],
        how="inner",
    )
    df = df.merge(
        sensitivity_df,
        on=join_keys,
        how="inner",
        suffixes=("", "_sens"),
    )

    if df.empty:
        print(
            "\nWARNING: joined dataframe is empty.\n"
            "Check that cluster_analysis.py, trajectory_geometry.py, and\n"
            "sensitivity.py were all run with the same decomposers file,\n"
            "so that layer_key and unit_idx are consistent across CSVs."
        )
        return

    print(f"\nJoined table: {len(df)} rows across "
          f"{df['layer_key'].nunique()} layers")

    # Save joined table
    joined_path = os.path.join(args.out_dir, "geometry_function_correlation.csv")
    df.to_csv(joined_path, index=False)
    print(f"Saved joined CSV: {joined_path}")

    # Compute Spearman correlations per (model, task, method)
    geo_metrics = [
        ("silhouette_kmeans",      "K-Means silhouette (layer mean)"),
        ("silhouette_kmeans_unit", "K-Means silhouette (per unit)"),
        ("silhouette_dbscan",      "DBSCAN silhouette"),
        ("path_length_norm",       "Path length / sqrt(d)"),
        ("displacement_norm",      "Displacement / sqrt(d)"),
        ("ratio",                  "Displacement / path length (directionality)"),
    ]

    stat_rows = []
    for (model, task, method), grp in df.groupby(["model", "task", "method"]):
        row = {"model": model, "task": task, "method": method,
               "n_units": len(grp)}
        for col, label in geo_metrics:
            if col not in grp.columns:
                row[f"r_{col}"] = float("nan")
                row[f"p_{col}"] = float("nan")
                continue
            r, p = spearman_safe(grp[col], grp["sensitivity"])
            row[f"r_{col}"] = r
            row[f"p_{col}"] = p
            print(f"  {model}/{task}/{method}  {col:<25}  r={r:.3f}  p={p:.4f}")
        stat_rows.append(row)

    stats_df = pd.DataFrame(stat_rows)
    stats_path = os.path.join(args.out_dir, "geometry_function_correlation_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved stats CSV: {stats_path}")
    print("\nCorrelation summary:")
    print(stats_df.to_string(index=False))

    # Scatter plots
    for col, label in geo_metrics:
        if col not in df.columns:
            continue
        scatter_geo_vs_sensitivity(df, col, label, args.out_dir)

    # Per-layer-type Spearman correlations
    layertype_rows = []
    for (model, task, method, layer_type), grp in df.groupby(
            ["model", "task", "method", "layer_type"]):
        row = {
            "model":      model,
            "task":       task,
            "method":     method,
            "layer_type": layer_type,
            "n_units":    len(grp),
        }
        for col, label in geo_metrics:
            if col not in grp.columns:
                row[f"r_{col}"] = float("nan")
                row[f"p_{col}"] = float("nan")
                continue
            r, p = spearman_safe(grp[col], grp["sensitivity"])
            row[f"r_{col}"] = r
            row[f"p_{col}"] = p
        layertype_rows.append(row)

    layertype_df = pd.DataFrame(layertype_rows)
    layertype_path = os.path.join(
        args.out_dir, "geometry_function_correlation_by_layer_type.csv"
    )
    layertype_df.to_csv(layertype_path, index=False)
    print(f"\nSaved per-layer-type stats: {layertype_path}")
    print(layertype_df.to_string(index=False))


    print("\nDone.")


if __name__ == "__main__":
    main()
