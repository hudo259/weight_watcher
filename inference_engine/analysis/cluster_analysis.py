"""
Examines how semantic units (filters, neurons, attention heads) self-organise
in the 3D decomposition space at the final checkpoint.

Two clustering algorithms:
  - DBSCAN:  discovers clusters of arbitrary shape; noise points labelled -1.
  - K-Means: fixed number of clusters k; we sweep k=2..k_max and report the
             silhouette score for each k, then pick the best.

Measurements per layer:
  - n_clusters_dbscan, n_noise_dbscan, silhouette_dbscan
  - best_k_kmeans, silhouette_kmeans

Output
------
  results/cluster_analysis/
      cluster_analysis.csv: per-layer stats
      cluster_analysis_summary.csv: mean per (model, task, layer_type)
      scatter_{model}_{task}_{layer_depth}.pdf: 2D scatter (first 2 dims of 3D)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from shared import load_points_3d, parse_run

# decomposers_path is optional but required for layer_type in the output CSV
# and for geometry_function_correlation.py.
RUN_FIELDS = ("model", "task", "method", "projection_path", "decomposers_path")

# Loading


def load_layer_types(decomposers_path: str) -> dict[str, str]:
    """Load layer_key -> layer_type from a *_decomposers.pt file."""
    raw = torch.load(decomposers_path, map_location="cpu", weights_only=False)
    return {k: v["layer_type"] for k, v in raw.items()}


# Clustering
def run_dbscan(
    pts: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple:
    """
    Returns (n_clusters, n_noise, silhouette).
    silhouette = nan if fewer than 2 non-noise clusters.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    sil = float("nan")
    if n_clusters >= 2:
        # silhouette_score only on non-noise points
        mask = labels != -1
        if mask.sum() >= 2:
            sil = silhouette_score(pts[mask], labels[mask])

    return n_clusters, int(n_noise), sil


def run_kmeans_sweep(
    pts: np.ndarray,
    k_max: int,
    random_state: int = 42,
) -> tuple:
    """
    Sweep k from 2 to k_max, return (best_k, best_silhouette).
    """
    C = pts.shape[0]
    k_max = min(k_max, C - 1)
    if k_max < 2:
        return 1, float("nan")

    best_k   = 2
    best_sil = -1.0

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(pts)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(pts, labels)
        if sil > best_sil:
            best_sil = sil
            best_k   = k

    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    best_labels = km_best.fit_predict(pts)
    unit_sils = silhouette_samples(pts, best_labels)  # [C]
    return best_k, best_sil, unit_sils


# Analysis
def analyze_run(
    model: str,
    task: str,
    method: str,
    projection_path: str,
    dbscan_eps: float,
    dbscan_min: int,
    kmeans_kmax: int,
    scatter_max_layers: int,
    out_dir: str,
    decomposers_path: str = None,
) -> list[dict]:
    """
    Returns one row per UNIT (not per layer) so that results can be joined
    with sensitivity.csv and trajectory_geometry.csv on (layer_key, unit_idx).

    Each row contains:
      - layer-level cluster stats (silhouette, n_clusters) — same for all
        units in the same layer
      - per-unit KMeans cluster assignment (kmeans_cluster)
      - layer_type (from decomposers_path if provided, else "unknown")
    """
    print(f"\nLoading {model}/{task}/{method}: {projection_path}")
    layer_projs = load_points_3d(projection_path)

    # Load layer_type mapping if decomposers file is provided
    layer_types: dict[str, str] = {}
    if decomposers_path is not None:
        layer_types = load_layer_types(decomposers_path)
        print(f"  Loaded layer_type for {len(layer_types)} layers from decomposers")
    else:
        print("  WARNING: no decomposers_path provided — layer_type will be 'unknown'")
        print("  Add decomposers_path as 5th field in --runs for full output.")

    unit_rows = []   # one row per unit — required for join with sensitivity

    for depth_idx, (key, pts) in enumerate(layer_projs.items()):
        C = pts.shape[0]
        layer_type = layer_types.get(key, "unknown")

        if C < 4:
            print(f"  [{depth_idx:3d}] {key[:55]}: skipped (C={C} < 4)")
            continue

        # Standardise before clustering so eps is scale-invariant across layers
        pts_scaled = StandardScaler().fit_transform(pts)

        # DBSCAN — layer-level stats
        n_cl_db, n_noise, sil_db = run_dbscan(pts_scaled, dbscan_eps, dbscan_min)

        # KMeans sweep — find best k
        best_k, sil_km, unit_sils = run_kmeans_sweep(pts_scaled, kmeans_kmax)

        # Fit final KMeans with best_k to get per-unit cluster assignments
        actual_k = max(2, best_k)
        km_final = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
        unit_cluster_labels = km_final.fit_predict(pts_scaled)   # [C]

        # DBSCAN per-unit labels for cluster assignment
        db_final = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min).fit(pts_scaled)
        dbscan_labels = db_final.labels_   # [C], -1 = noise

        print(f"  [{depth_idx:3d}] {key[:50]:<50}  C={C}  "
              f"DBSCAN: n_cl={n_cl_db} noise={n_noise} sil={sil_db:.3f}  "
              f"KMeans: k={best_k} sil={sil_km:.3f}")

        # One row per unit
        for unit_idx in range(C):
            unit_rows.append({
                "model":              model,
                "task":               task,
                "method":             method,
                "layer_key":          key,
                "layer_type":         layer_type,
                "layer_depth":        depth_idx,
                "unit_idx":           unit_idx,
                # Layer-level stats (same for all units in this layer)
                "n_units":            C,
                "n_clusters_dbscan":  n_cl_db,
                "n_noise_dbscan":     n_noise,
                "silhouette_dbscan":  sil_db,
                "best_k_kmeans":      best_k,
                "silhouette_kmeans":  sil_km,
                "silhouette_kmeans_unit": float(unit_sils[unit_idx]),
                # Per-unit cluster assignments
                "kmeans_cluster":     int(unit_cluster_labels[unit_idx]),
                "dbscan_cluster":     int(dbscan_labels[unit_idx]),
            })

        # Scatter plot for first N layers
        if depth_idx < scatter_max_layers:
            _scatter(pts, key, model, task, depth_idx, best_k, out_dir)

    return unit_rows


def _scatter(
    pts: np.ndarray,
    key: str,
    model: str,
    task: str,
    depth_idx: int,
    best_k: int,
    out_dir: str,
) -> None:
    """2-D scatter of first two PCA dimensions, colored by K-Means cluster."""
    km = KMeans(n_clusters=max(2, best_k), n_init=10, random_state=42)
    labels = km.fit_predict(pts)

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10",
                         s=8, alpha=0.7)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    short_key = key.split(".")[-2] if "." in key else key
    ax.set_title(f"{model}/{task} — layer {depth_idx}: {short_key}")
    plt.colorbar(scatter, ax=ax, label="K-Means cluster")
    plt.tight_layout()

    fname = f"scatter_{model}_{task}_{depth_idx:03d}.pdf"
    path  = os.path.join(out_dir, fname)
    fig.savefig(path)
    plt.close(fig)


# Summary plots
def plot_silhouette_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: mean silhouette per (model, task) for both methods."""
    groups = df.groupby(["model", "task"])
    labels = [f"{m}\n{t}" for (m, t), _ in groups]
    sil_db = [grp["silhouette_dbscan"].mean() for _, grp in groups]
    sil_km = [grp["silhouette_kmeans"].mean() for _, grp in groups]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 4))
    ax.bar(x - w / 2, sil_db, width=w, label="DBSCAN")
    ax.bar(x + w / 2, sil_km, width=w, label="K-Means (best k)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Cluster quality by architecture and task")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "silhouette_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def main():
    ap = argparse.ArgumentParser(description="Cluster analysis of point clouds")
    ap.add_argument(
        "--runs", nargs="+", required=True,
        metavar="MODEL,TASK,METHOD,PROJECTION_PATH",
    )
    ap.add_argument("--dbscan_eps",       type=float, default=0.5,
                    help="DBSCAN neighbourhood radius (in standardised space)")
    ap.add_argument("--dbscan_min",       type=int,   default=5,
                    help="DBSCAN minimum samples per cluster")
    ap.add_argument("--kmeans_kmax",      type=int,   default=10,
                    help="Maximum k to sweep for K-Means")
    ap.add_argument("--scatter_max_layers", type=int, default=5,
                    help="Number of layers to generate scatter plots for")
    ap.add_argument("--out_dir",          default="./results/cluster_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for run_str in args.runs:
        model, task, method, proj_path, decomp_path = parse_run(run_str, RUN_FIELDS, n_optional=1)
        rows = analyze_run(
            model=model,
            task=task,
            method=method,
            projection_path=proj_path,
            dbscan_eps=args.dbscan_eps,
            dbscan_min=args.dbscan_min,
            kmeans_kmax=args.kmeans_kmax,
            scatter_max_layers=args.scatter_max_layers,
            out_dir=args.out_dir,
            decomposers_path=decomp_path,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No rows collected.")
        return

    df = pd.DataFrame(all_rows)

    csv_path = os.path.join(args.out_dir, "cluster_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw CSV: {csv_path}")

    # Summary is now at layer level — aggregate per-unit rows back to layer stats
    summary = (
        df.groupby(["model", "task", "method", "layer_key", "layer_type", "layer_depth"])[
            ["silhouette_dbscan", "silhouette_kmeans",
             "n_clusters_dbscan", "best_k_kmeans"]
        ]
        .first()   # all units in a layer share the same layer-level stats
        .reset_index()
    )
    summary_path = os.path.join(args.out_dir, "cluster_analysis_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_silhouette_comparison(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
