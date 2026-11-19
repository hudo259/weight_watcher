"""
For every layer, re-runs the shared-basis SVD keeping the FULL eigenvalue
spectrum (projection.py / pca.py normally truncate to k), and reports how many
principal components are needed to reach a target cumulative explained
variance (default 90%).
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# This script lives in inference_engine/analysis/ but imports from
# inference_engine/ (main, models, inference, projections). Add the parent
# directory to sys.path so those imports resolve regardless of the current
# working directory the script is launched from.
_ANALYSIS_DIR = Path(__file__).resolve().parent          # inference_engine/analysis
_ENGINE = _ANALYSIS_DIR.parent                            # inference_engine
for _p in (_ENGINE, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from projections.common.templates import expand_templates
from projections.common.layer_decomposition import (
    _flatten_weight,
    _load_trajectories,
)
from projections.projection import _register_resolvers, get_state_dict_key


def _full_spectrum_ev(trajectories: torch.Tensor) -> np.ndarray:
    """
    Given [C, T, d] RAW weight trajectories, replicate pca.py's fit() exactly
    but keep ALL eigenvalues. Returns the cumulative explained-variance curve
    (length = min(C*T, d)), i.e. cumsum(eigenvalues)/total_var.

    Mirrors pca.py:
        init   = traj[:,0,:]
        delta  = traj - init
        X_all  = delta.reshape(C*T, d)
        mean   = X_all.mean(0);  X_c = X_all - mean
        S      = svdvals(X_c)
        eig    = S^2 / (C*T - 1)
    """
    C, T, d = trajectories.shape
    init = trajectories[:, 0, :]
    delta = trajectories - init.unsqueeze(1)
    X_all = delta.reshape(C * T, d)
    mean = X_all.mean(dim=0)
    X_c = X_all - mean

    # Singular values only (cheaper than full SVD)
    S = torch.linalg.svdvals(X_c)                     # [min(C*T, d)]
    eig = (S ** 2) / max(C * T - 1, 1)
    total = eig.sum().clamp(min=1e-12)
    cum = torch.cumsum(eig, dim=0) / total
    return cum.cpu().numpy()


def _k_for_threshold(cum_ev: np.ndarray, thr: float) -> int:
    """Smallest number of components whose cumulative EV >= thr."""
    idx = np.searchsorted(cum_ev, thr) + 1
    return int(min(idx, len(cum_ev)))


def main():
    ap = argparse.ArgumentParser(description="Per-layer scree / EV analysis")
    ap.add_argument("-i", "--input_dir", required=True,
                    help="Directory of training checkpoints (same as projection.py)")
    ap.add_argument("-c", "--config", required=True,
                    help="Decomposition config YAML (same as projection.py)")
    ap.add_argument("--model_name", required=True,
                    help="e.g. convnextv2_base")
    ap.add_argument("--task", required=True,
                    help="e.g. classification_imagenet100")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95],
                    help="Cumulative-EV thresholds to report (default 0.90 0.95)")
    ap.add_argument("--k_ref", type=int, default=3,
                    help="Reference k to report EV at (default 3, the paper's k)")
    ap.add_argument("-o", "--out_dir", default="analysis/results/scree")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda for the SVD (cpu is fine)")
    args = ap.parse_args()

    _register_resolvers()
    os.makedirs(args.out_dir, exist_ok=True)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    layer_mapping = expand_templates(config)

    # Enumerate checkpoints exactly like projection.py
    root_dir = Path(args.input_dir)
    checkpoint_paths = sorted({
        p.resolve(): p
        for p in itertools.chain.from_iterable(
            root_dir.rglob(pat) for pat in ("*.pt", "*.pth")
        )
    }.values(), key=lambda p: p.name)
    checkpoint_paths = [str(p) for p in checkpoint_paths]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints in {args.input_dir}")
    state_dict_key = get_state_dict_key(checkpoint_paths[0])

    print(f"[scree] {args.model_name} / {args.task}: "
          f"{len(layer_mapping)} layers, {len(checkpoint_paths)} checkpoints")

    rows: list[dict] = []
    for layer_key, info in layer_mapping.items():
        # layer_mapping values are bare strings, or dicts only when a per-layer
        # k / num_heads override was set (see templates.expand_templates).
        layer_type = info["type"] if isinstance(info, dict) else info
        num_heads  = info.get("num_heads") if isinstance(info, dict) else None

        flatten_fn = lambda w, lt=layer_type, nh=num_heads: _flatten_weight(
            w.float(), lt, nh
        )
        try:
            trajectories, _ = _load_trajectories(
                checkpoint_paths, layer_key, state_dict_key, flatten_fn, args.device
            )
        except KeyError:
            print(f"  [skip] {layer_key} not in checkpoints")
            continue

        C, T, d = trajectories.shape
        cum_ev = _full_spectrum_ev(trajectories)

        row = {
            "model": args.model_name,
            "task": args.task,
            "layer_key": layer_key,
            "layer_type": layer_type,
            "C": C, "T": T, "d": d,
            "n_components_available": len(cum_ev),
            f"ev_at_k{args.k_ref}": float(cum_ev[min(args.k_ref, len(cum_ev)) - 1]),
        }
        for thr in args.thresholds:
            row[f"k_at_{int(thr*100)}"] = _k_for_threshold(cum_ev, thr)
        rows.append(row)
        print(f"  {layer_type:<16} {layer_key:<45} "
              f"d={d:<5} EV@{args.k_ref}={row[f'ev_at_k{args.k_ref}']:.3f} "
              f"k@90={row.get('k_at_90','-')}")

    df = pd.DataFrame(rows)
    per_layer_path = os.path.join(
        args.out_dir, f"scree_{args.model_name}_{args.task}.csv"
    )
    df.to_csv(per_layer_path, index=False)
    print(f"\n[scree] Saved per-layer: {per_layer_path}")

    # Aggregate per layer type
    agg_cols = {
        f"ev_at_k{args.k_ref}": ["mean"],
    }
    for thr in args.thresholds:
        agg_cols[f"k_at_{int(thr*100)}"] = ["median", "mean", "max"]

    agg = df.groupby(["model", "task", "layer_type"]).agg(agg_cols)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["n_layers"] = (
        df.groupby(["model", "task", "layer_type"]).size().values
    )
    agg_path = os.path.join(
        args.out_dir, f"scree_summary_{args.model_name}_{args.task}.csv"
    )
    agg.to_csv(agg_path, index=False)
    print(f"[scree] Saved summary:   {agg_path}\n")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
