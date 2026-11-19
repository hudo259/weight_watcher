"""
For each layer, computes pairwise unit distances under PCA, UMAP and
FourierKPCA projections, then measures how consistently the three methods
rank inter-unit distances via Spearman rank correlation.

A high correlation means the three methods agree on the geometry of the
weight space. The structure is intrinsic to the data, not an artifact of
the decomposition method.

Inputs
------
For each (model, task) you need a PCA path, a KPCA path, and one or more
labelled UMAP paths. Each projection file is the *_projected.pt output of
projection.py and contains:
    {"projections": {layer_key: {"points_3d": Tensor[C, k], ...}}}

Each --runs entry is:
    model,task,pca_path,kpca_path[,label:umap_path ...]

  - pca_path and kpca_path are positional (fields 3 and 4).
  - Every subsequent field is a labelled UMAP path: label:path
    where label is a short string identifying the UMAP variant
    (e.g. "n5", "0-1_15").  At least one UMAP entry is required.

Output
------
  results/method_consistency/
      method_consistency.csv (Spearman r per (model, task, umap_variant, layer))
      method_consistency_summary.csv (mean +- std per (model, task, umap_variant))
      heatmap_spearman_<variant>.pdf (one heatmap pdf per UMAP variant)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from matplotlib.patches import Patch
from display_names import MODEL_DISPLAY, DATASET_DISPLAY, display
from shared import load_points_3d, parse_run as split_run

# Last field keeps its commas: one or more label:umap_path entries
RUN_FIELDS = ("model", "task", "pca_path", "kpca_path", "label:umap_path[,...]")

def load_effective_neighbors(decomp_path: str) -> dict[str, int]:
    """Returns layer_key -> effective_n_neighbors from UMAPDecomposer."""
    data = torch.load(decomp_path, map_location="cpu", weights_only=False)
    result = {}
    for key, layer_decomp in data.items():
        # Handle both LayerDecomposition objects and raw dicts
        if isinstance(layer_decomp, dict):
            decomposer_data = layer_decomp.get("decomposer", {})
            if isinstance(decomposer_data, dict):
                n = decomposer_data.get("effective_n_neighbors",
                    decomposer_data.get("n_neighbors", None))
            else:
                n = getattr(decomposer_data, "_effective_n_neighbors",
                    getattr(decomposer_data, "_n_neighbors", None))
        else:
            decomposer = layer_decomp.decomposer
            n = getattr(decomposer, "_effective_n_neighbors",
                getattr(decomposer, "_n_neighbors", None))
        result[key] = n
    return result

def parse_run(s: str):
    """
    Parse one --runs entry.

    Format:
        model,task,pca_path,kpca_path,label1:umap_path1[,label2:umap_path2,...]

    Returns:
        model       str
        task        str
        pca_path    str
        kpca_path   str
        umap_runs   list of (label, path) tuples, at least one entry
    """
    model, task, pca_path, kpca_path, umap_fields = split_run(s, RUN_FIELDS)

    umap_runs = []
    for field in umap_fields.split(","):
        if ":" not in field:
            raise argparse.ArgumentTypeError(
                f"UMAP entries must be 'label:path' or 'label:proj_path:decomp_path', got: {field!r}"
            )

        subparts = field.split(":", 2)
        label = subparts[0]
        proj_path = subparts[1]
        decomp_path = subparts[2] if len(subparts) == 3 else None
        umap_runs.append((label, proj_path, decomp_path))

    return model, task, pca_path, kpca_path, umap_runs


# Distance / correlation helpers
def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Upper-triangle pairwise L2 distances for [C, k] points.
    Returns a flat array of C*(C-1)/2 distances.
    """
    diff = points[:, None, :] - points[None, :, :]   # [C, C, k]
    dist = np.linalg.norm(diff, axis=-1)              # [C, C]
    idx  = np.triu_indices(points.shape[0], k=1)
    return dist[idx]


def spearman_between(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Spearman rank correlation between pairwise distance vectors."""
    d_a = pairwise_distances(pts_a)
    d_b = pairwise_distances(pts_b)
    if len(d_a) < 3:
        return float("nan")

    r, _ = spearmanr(d_a, d_b)
    return float(r)


def analyze_run(
    model:     str,
    task:      str,
    pca_path:  str,
    kpca_path: str,
    umap_runs: list[tuple[str, str, str | None]],
) -> list[dict]:
    """
    For each UMAP variant and each layer, compute:
        r_pca_umap   Spearman(PCA, UMAP)
        r_pca_kpca   Spearman(PCA, KPCA)
        r_umap_kpca  Spearman(UMAP, KPCA)

    Returns a list of row dicts with a 'umap_variant' column so all
    variants land in the same CSV and can be compared directly.
    """
    print(f"\n  Loading PCA  : {pca_path}")
    pca_projs  = load_points_3d(pca_path)
    print(f"  Loading KPCA : {kpca_path}")
    kpca_projs = load_points_3d(kpca_path)

    # PCA key order defines canonical layer depth
    pca_key_order = list(pca_projs.keys())

    all_rows = []

    for umap_label, umap_path, decomp_path in umap_runs:
        print(f"\n  UMAP variant '{umap_label}': {umap_path}")
        umap_projs = load_points_3d(umap_path)

        common_keys = set(pca_projs) & set(umap_projs) & set(kpca_projs)
        if not common_keys:
            print(f"    WARNING: no common layer keys for {model}/{task}/{umap_label}")
            continue

        eff_neighbors = load_effective_neighbors(decomp_path) if decomp_path else {}

        layer_keys = [k for k in pca_key_order if k in common_keys]

        for depth_idx, key in enumerate(layer_keys):
            pca_pts  = pca_projs[key]
            umap_pts = umap_projs[key]
            kpca_pts = kpca_projs[key]

            if pca_pts.shape[0] < 4:
                continue

            r_pca_umap  = spearman_between(pca_pts,  umap_pts)
            r_pca_kpca  = spearman_between(pca_pts,  kpca_pts)
            r_umap_kpca = spearman_between(umap_pts, kpca_pts)

            all_rows.append({
                "model":        model,
                "task":         task,
                "umap_variant": umap_label,
                "layer_key":    key,
                "layer_depth":  depth_idx,
                "n_units":      pca_pts.shape[0],
                "r_pca_umap":   r_pca_umap,
                "r_pca_kpca":   r_pca_kpca,
                "r_umap_kpca":  r_umap_kpca,
                "effective_n_neighbors": eff_neighbors.get(key, None)
            })

            print(f"    [{depth_idx:3d}] {key[:55]:<55}  "
                  f"PCA↔UMAP={r_pca_umap:.3f}  "
                  f"PCA↔KPCA={r_pca_kpca:.3f}  "
                  f"UMAP↔KPCA={r_umap_kpca:.3f}")

    return all_rows


def plot_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    """
    For each UMAP variant, produce a PDF with one row of subplots per
    (model, task) and three columns: PCA <-> UMAP, PCA <-> KPCA, UMAP <-> KPCA.
    Layers where the adaptive n_neighbors cap kicked in are shown in grey.

    Note: This function has been generated with the help of Claude
    """
    col_specs = [
        ("r_pca_umap",  "PCA - UMAP"),
        ("r_pca_kpca",  "PCA - KPCA"),
        ("r_umap_kpca", "UMAP - KPCA"),
    ]
    # Only UMAP-involved pairs are affected by adaptive n_neighbors
    umap_cols = {"r_pca_umap", "r_umap_kpca"}

    cmap = plt.get_cmap("RdYlGn")
    has_eff_neighbors = "effective_n_neighbors" in df.columns

    for variant, vdf in df.groupby("umap_variant"):
        # Configured n_neighbors = max effective across all layers in this variant
        # (layers that weren't capped will have the full value)
        max_val = vdf["effective_n_neighbors"].max()
        configured_n = (
            int(max_val)if has_eff_neighbors and not pd.isna(max_val) else None
        )

        runs   = list(vdf.groupby(["model", "task"]))
        n_runs = len(runs)
        n_cols = len(col_specs)

        fig, axes = plt.subplots(
            n_runs, n_cols,
            figsize=(5 * n_cols, 4 * n_runs),
            squeeze=False,
        )
        fig.suptitle(f"UMAP variant: {variant}", fontsize=11, y=1.01)

        for row_idx, ((model, task), grp) in enumerate(runs):
            grp_sorted = grp.sort_values("layer_depth")
            vals_dict  = {col: grp_sorted[col].values for col, _ in col_specs}

            # Boolean mask: True where adaptive cap reduced n_neighbors
            if has_eff_neighbors and configured_n is not None:
                adapted = grp_sorted["effective_n_neighbors"].fillna(configured_n).values < configured_n
            else:
                adapted = np.zeros(len(grp_sorted), dtype=bool)

            any_adapted = adapted.any()

            for col_idx, (col, label) in enumerate(col_specs):
                ax   = axes[row_idx][col_idx]
                vals = vals_dict[col]

                # Grey only for UMAP-involved columns; PCA↔KPCA is unaffected
                if col in umap_cols and any_adapted:
                    bar_colors = np.where(
                        adapted[:, None],
                        np.array([[0.75, 0.75, 0.75, 1.0]]),
                        cmap(np.clip(vals, 0, 1))
                    )
                else:
                    bar_colors = cmap(np.clip(vals, 0, 1))

                ax.bar(range(len(vals)), vals, color=bar_colors)
                ax.set_ylim(0, 1)
                ax.axhline(0.9, color="green", linewidth=0.8,
                           linestyle="--", alpha=0.6)
                ax.set_title(f"{display(model, MODEL_DISPLAY)}\n{display(task, DATASET_DISPLAY)}\n{label}", fontsize=9)
                ax.set_xlabel("Layer depth")
                ax.set_ylabel("Spearman $\\rho$")

                if col in umap_cols and any_adapted:
                    ax.legend(
                        handles=[
                            Patch(facecolor=cmap(0.9), label="full n_neighbors"),
                            Patch(facecolor="silver",  label="adaptive n_neighbors"),
                        ],
                        fontsize=7, loc="lower right",
                    )

        plt.tight_layout()
        safe_label = variant.replace("/", "-").replace(" ", "_")
        path = os.path.join(out_dir, f"heatmap_spearman_{safe_label}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

def main():
    ap = argparse.ArgumentParser(
        description="Method consistency analysis"
    )
    ap.add_argument(
        "--runs", nargs="+", required=True,
        metavar="MODEL,TASK,PCA_PATH,KPCA_PATH,LABEL:UMAP_PATH[,...]",
        help=(
            "One entry per (model, task). Format: "
            "model,task,pca_path,kpca_path,label1:umap_path1[,label2:umap_path2,...]"
        ),
    )
    ap.add_argument("--out_dir", default="./results/method_consistency")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for run_str in args.runs:
        model, task, pca_path, kpca_path, umap_runs = parse_run(run_str)
        print(f"\n{'='*60}")
        print(f"  {display(model, MODEL_DISPLAY)} / {display(task, DATASET_DISPLAY)}: {len(umap_runs)} UMAP variant(s)")
        print(f"{'='*60}")
        rows = analyze_run(model, task, pca_path, kpca_path, umap_runs)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows collected — check your paths.")
        return

    df = pd.DataFrame(all_rows)
    df["model"] = df["model"].map(lambda x: display(x, MODEL_DISPLAY))
    df["task"]  = df["task"].map(lambda x: display(x, DATASET_DISPLAY))

    # Raw CSV
    csv_path = os.path.join(args.out_dir, "method_consistency.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw CSV : {csv_path}")

    # Summary: mean +- std per (model, task, umap_variant)
    summary = (
        df.groupby(["model", "task", "umap_variant"])[
            ["r_pca_umap", "r_pca_kpca", "r_umap_kpca"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary_path = os.path.join(args.out_dir, "method_consistency_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary : {summary_path}")
    print("\nSummary (mean r_pca_umap per variant):")
    print(
        summary[["model", "task", "umap_variant", "r_pca_umap_mean"]]
        .sort_values(["model", "task", "r_pca_umap_mean"], ascending=[True, True, False])
        .to_string(index=False)
    )

    plot_heatmaps(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
