"""
The paper's redundancy interpretation says: in PROJECTION layers (linear, fused
key-value), units that moved far from initialisation drift into densely
populated regions of the embedding, where many similar units can compensate for
them, so they are functionally REDUNDANT. Units that barely moved sit in
isolated positions and are IRREPLACEABLE.

That interpretation makes a falsifiable prediction:

    Pruning high-displacement units in projection layers should damage the
    model LESS than pruning random units, while pruning geometrically
    ISOLATED (high-silhouette) units should damage it MORE.

This script tests that prediction directly, converting an interpretation
inferred from two correlations into a demonstrated capability.

For a given prune ratio q, we zero out q% of the units per layer, selected by
five different criteria, reconstruct the weights, load them into the live model,
and measure the drop in the task metric (accuracy / mIoU):

  high_disp : top-q by normalised displacement   -> predicted LEAST damage
  low_disp  : bottom-q by normalised displacement-> predicted more damage
  high_sil  : top-q by per-unit silhouette       -> predicted MOST damage
  low_sens  : bottom-q by sensitivity s_c        -> standard pruning baseline
  random    : random q%                          -> chance baseline

Pruning is STRUCTURED and PER-LAYER: within each layer we zero whole semantic
units (a full conv filter / linear-layer row), matching how the paper defines
units and keeping the comparison layer-type separable.
"""

import argparse
import copy
import os
import re
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

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models.factory import build_model
from inference.evaluator import Evaluator
from inference.weight_reconstruction import WeightReconstructor
from shared import primary_metric

# Reuse the exact loaders/helpers from main.py so data + checkpoints are
# handled identically to the rest of the pipeline.
from main import (
    _set_seeds,
    _load_checkpoint,
    _load_decompositions,
    _build_val_loader,
)


def _norm(s: str) -> str:
    """Strip non-alphanumerics and lowercase."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _filter_model_task(df: pd.DataFrame, model_name: str, dataset: str) -> pd.DataFrame:
    """Filter a multi-model/multi-task CSV down to this run (bidirectional match)."""
    if df is None:
        return None
    out = df
    if "model" in out.columns:
        out = out[out["model"].apply(_norm) == _norm(model_name)]
    if "task" in out.columns and dataset:
        dn = _norm(dataset)
        out = out[out["task"].apply(lambda t: dn in _norm(t) or _norm(t) in dn)]
    if "method" in out.columns and (out["method"] == "pca").any():
        out = out[out["method"] == "pca"]
    return out


# The pruning core
# ----------------
CONDITIONS = ("high_disp", "low_disp", "high_sil", "low_sens", "random", "depth_high", "depth_low", "gm_final", "fpgm_final", "norm_final")


class PruningRunner:
    def __init__(self, model, reconstructor: WeightReconstructor,
                 evaluator: Evaluator, task: str, device: str):
        self._model = model
        self._recon = reconstructor
        self._eval = evaluator
        self._task = task
        self._device = device

        # Baseline = the fully reconstructed (un-pruned) model, so the measured
        # drop isolates the PRUNING effect from any reconstruction error.
        self._recon.reset()
        self._recon.reconstruct_and_load(self._model)
        self._clean_state = copy.deepcopy(self._model.state_dict())

    def _restore_clean(self) -> None:
        self._model.load_state_dict(self._clean_state)

    def baseline_metric(self) -> float:
        self._restore_clean()
        return primary_metric(self._task, self._eval.run().metrics)

    def _select_units(
        self,
        layer_key: str,
        condition: str,
        q: float,
        scores: dict[str, pd.DataFrame],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the unit indices to PRUNE in this layer for the condition."""
        C = self._recon._current_Z[layer_key].shape[0]
        n = max(1, int(round(q * C)))

        if condition == "random":
            return rng.choice(C, size=n, replace=False)

        spec = {
            "high_disp": ("trajectory", "displacement_norm", False),  # descending
            "low_disp":  ("trajectory", "displacement_norm", True),   # ascending
            "high_sil":  ("cluster",    "silhouette_kmeans_unit", False),
            "low_sens":  ("sensitivity", "sensitivity", True),
            "depth_high": ("fpgm", "depth_disp", False), # highest depth = most central
            "depth_low": ("fpgm", "depth_disp", True), # lowest depth = most peripheral 
            "gm_final": ("fpgm", "gm_final", True), # smallest distance = most central 
            "fpgm_final": ("fpgm", "fpgm_final", True), # smallest sum-of-distances = most central
            "norm_final": ("fpgm", "norm_final", True), # smallest norm = magnitude pruning
        }[condition]
        src, col, ascending = spec

        df = scores.get(src)
        if df is None or col not in df.columns:
            return np.array([], dtype=np.int64)

        sub = df[df["layer_key"] == layer_key]
        sub = sub[["unit_idx", col]].dropna()
        if len(sub) < n:
            return np.array([], dtype=np.int64)

        order = sub.sort_values(col, ascending=ascending)
        idx = order["unit_idx"].to_numpy()[:n].astype(np.int64)
        return idx[idx < C]   # guard against index drift

    # the actual prune + evaluate 
    def run_condition(
        self,
        condition: str,
        q: float,
        scores: dict[str, pd.DataFrame],
        layer_types: list[str] | None,
        seed: int,
    ) -> dict[str, float]:
        """
        Zero out q% of units per (eligible) layer, load into the model, evaluate.
        Returns {"metric": ..., "n_pruned": ..., "n_layers": ...}
        """
        rng = np.random.default_rng(seed)
        self._restore_clean()
        self._recon.reset()

        state_dict = self._model.state_dict()
        total_pruned, n_layers = 0, 0

        for layer_key in self._recon.layer_keys:
            decomp = self._recon._decompositions[layer_key]
            lt = decomp.layer_type
            if layer_types and lt not in layer_types:
                continue   # layer type not under test -> leave untouched

            idx = self._select_units(layer_key, condition, q, scores, rng)
            if len(idx) == 0:
                continue

            # Reconstruct this layer's weights, then ZERO the selected units.
            # _reshape_to_original always puts the unit index on dim 0, for
            # every layer type, so weight[idx] = 0 removes whole units.
            weight = self._recon.reconstruct_layer(layer_key).clone()
            weight[torch.as_tensor(idx, dtype=torch.long, device=weight.device)] = 0.0
            state_dict[layer_key].copy_(weight)

            total_pruned += len(idx)
            n_layers += 1

        if n_layers == 0:
            metric = float("nan")
            total_pruned = 0
        else:
            # Any layer we did NOT prune keeps its clean reconstructed weights
            # (already in state_dict from _restore_clean).
            metric = primary_metric(self._task, self._eval.run().metrics)

        return {"metric": metric, "n_pruned": total_pruned, "n_layers": n_layers}


def main():
    ap = argparse.ArgumentParser(description="Geometry-guided pruning comparison")
    ap.add_argument("task", choices=["classification", "segmentation"])
    ap.add_argument("--backbone", required=True, choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["imagenet100", "isic2019", "voc2012_sbd", "isic2018"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--decomposers_path", required=True)
    ap.add_argument("--projection_path", required=True)
    ap.add_argument("--val_transforms", default=None)
    ap.add_argument("--loss_config", default=None)

    # Score sources for the selection criteria
    ap.add_argument("--sensitivity_csv", default=None,
                    help="sensitivity.csv (per-unit s_c) for low_sens")
    ap.add_argument("--trajectory_csv", default=None,
                    help="trajectory_geometry.csv (displacement_norm) for high/low_disp")
    ap.add_argument("--cluster_csv", default=None,
                    help="cluster_analysis.csv (silhouette_kmeans_unit) for high_sil")
    ap.add_argument("--fpgm_csv", default=None,
                    help="fpgm_per_unit.csv for depth_disp, gm_final, fpgm_final, norm_final")

    ap.add_argument("--layer_types", nargs="*", default=None,
                    help="Restrict pruning to these layer types (e.g. linear). "
                         "Default: all decomposed layers. The redundancy prediction "
                         "is specific to projection layers, so run 'linear' and "
                         "'conv2d' separately to contrast them.")
    ap.add_argument("--prune_ratios", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="Seeds (only the 'random' condition varies; others are "
                         "deterministic, so 1 seed suffices for them)")
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--eval_size", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_dir", default="analysis/results/pruning")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _set_seeds(args.seed)
    args.subcommand = args.task
    args.anchor_size = args.eval_size
    args.feature_layer = None
    model_name = f"{args.backbone}_{args.variant}"

    print("[pruning] Building model...")
    model = build_model(backbone=args.backbone, variant=args.variant,
                        task=args.task, n_classes=args.num_classes).to(args.device)
    model.eval()
    _load_checkpoint(model, args.checkpoint_path, args.device)

    val_loader = _build_val_loader(args)
    decompositions, projections = _load_decompositions(
        args.decomposers_path, args.projection_path)
    reconstructor = WeightReconstructor(decompositions=decompositions,
                                        projections=projections, device=args.device)

    loss_fn = None
    if args.loss_config:
        loss_fn = instantiate(OmegaConf.load(args.loss_config)).to(args.device)

    evaluator = Evaluator(model=model, dataloader=val_loader, task=args.task,
                          dataset_name=args.dataset, num_classes=args.num_classes,
                          loss_fn=loss_fn, amp=args.amp,
                          ignore_index=args.ignore_index)

    # Load the score sources, filtered to this model+dataset
    scores: dict[str, pd.DataFrame] = {}
    for key, path in (("sensitivity", args.sensitivity_csv),
                      ("trajectory", args.trajectory_csv),
                      ("cluster", args.cluster_csv),
                      ("fpgm", args.fpgm_csv)):
        if path and os.path.exists(path):
            df = _filter_model_task(pd.read_csv(path), model_name, args.dataset)
            scores[key] = df
            print(f"[pruning] {key:<12} CSV: {len(df)} rows after model/task filter")
        else:
            scores[key] = None
            print(f"[pruning] {key:<12} CSV: MISSING — dependent conditions skipped")

    runner = PruningRunner(model=model, reconstructor=reconstructor,
                           evaluator=evaluator, task=args.task, device=args.device)

    baseline = runner.baseline_metric()
    print(f"\n[pruning] Baseline {args.task} metric: {baseline:.4f}")
    print(f"[pruning] Pruning layer types: "
          f"{args.layer_types if args.layer_types else 'ALL'}\n")

    rows: list[dict] = []
    for q in args.prune_ratios:
        for cond in CONDITIONS:
            # Deterministic conditions need only one seed; random needs several.
            n_seeds = args.n_seeds if cond == "random" else 1
            vals, pruned, nlay = [], 0, 0
            for s in range(n_seeds):
                out = runner.run_condition(
                    condition=cond, q=q, scores=scores,
                    layer_types=args.layer_types, seed=args.seed + 1000 * s)
                vals.append(out["metric"])
                pruned, nlay = out["n_pruned"], out["n_layers"]
            if nlay == 0:
                print(f"  q={q:<5} {cond:<10}  [skipped — no eligible units]")
                continue
            mean_m, std_m = float(np.mean(vals)), float(np.std(vals))
            rows.append({
                "task": args.task, "model": model_name, "dataset": args.dataset,
                "layer_types": ",".join(args.layer_types) if args.layer_types else "all",
                "prune_ratio": q, "condition": cond,
                "metric_mean": mean_m, "metric_std": std_m,
                "drop_from_baseline": baseline - mean_m,
                "baseline": baseline, "n_units_pruned": pruned, "n_layers": nlay,
            })
            print(f"  q={q:<5} {cond:<10}  metric={mean_m:.4f}±{std_m:.4f}  "
                  f"drop={baseline - mean_m:+.4f}  ({pruned} units / {nlay} layers)")

    if not rows:
        print("\n[pruning] No results — check that the score CSVs cover these layers.")
        return

    df = pd.DataFrame(rows)
    lt_tag = "_".join(args.layer_types) if args.layer_types else "all"
    out_csv = os.path.join(
        args.out_dir, f"pruning_{model_name}_{args.task}_{args.dataset}_{lt_tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[pruning] Saved: {out_csv}")

    # The prediction test 
    print("\n[pruning] PREDICTION TEST (redundancy interpretation):")
    print("  expect  high_disp < random < high_sil   (drop from baseline)")
    for q in sorted(df.prune_ratio.unique()):
        sub = df[df.prune_ratio == q].set_index("condition")["drop_from_baseline"]
        hd = sub.get("high_disp", float("nan"))
        rd = sub.get("random",    float("nan"))
        hs = sub.get("high_sil",  float("nan"))
        verdict = ("SUPPORTED" if (hd < rd < hs) else
                   "partial"   if (hd < rd or rd < hs) else "NOT supported")
        print(f"  q={q:<5} high_disp={hd:+.4f}  random={rd:+.4f}  "
              f"high_sil={hs:+.4f}   -> {verdict}")


if __name__ == "__main__":
    main()
