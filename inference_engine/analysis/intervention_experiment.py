"""
Quantified behavioral intervention 

Does perturbing units that the *geometry* flags as functionally sensitive
degrade model behavior more than perturbing random units by the same
amount? And do perturbations in the DISCARDED residual subspace matter
less than perturbations in the retained k-dim embedding?

For a fixed perturbation magnitude epsilon (scaled per-layer by each PCA
axis' natural standard deviation, i.e. sqrt(eigenvalue)), we compare three
matched-magnitude conditions and measure the drop in the task metric
(accuracy for classification, mIoU for segmentation) on the validation set:

  1. GEOMETRIC : displace the embedding coordinate z_c of the units the
                 geometry predicts to be most sensitive (top-q by gradient
                 sensitivity s_c if a sensitivity CSV is supplied, else the
                 highest-displacement units) along a random direction IN the
                 retained k-dim embedding.
  2. RANDOM    : displace a RANDOM subset of units (same count, same
                 per-axis magnitude) in the retained k-dim embedding.
  3. RESIDUAL  : displace the SAME top units as (1), but in the discarded
                 out-of-subspace residual directions (same L2 magnitude),
                 leaving the k-dim embedding coordinate untouched.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

# This script lives in inference_engine/analysis/ but imports from
# inference_engine/ (main, models, inference, projections). Add the parent
# directory to sys.path so those imports resolve regardless of the current
# working directory the script is launched from.
_INFERENCE_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(_INFERENCE_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_ENGINE_DIR))

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models.factory import build_model
from inference.evaluator import Evaluator
from inference.weight_reconstruction import WeightReconstructor
from projections.common.layer_decomposition import LayerDecomposition
from shared import primary_metric

# Reuse the exact loaders/helpers from main.py so data + checkpoints are
# handled identically to the rest of the pipeline.
from main import (
    _set_seeds,
    _load_checkpoint,
    _load_decompositions,
    _build_val_loader,
)


def _axis_scale(decomp: LayerDecomposition) -> torch.Tensor:
    """
    Per-axis standard deviation of the retained k-dim embedding, i.e.
    sqrt(eigenvalue). Used so a perturbation of 'epsilon' means 'epsilon
    natural standard deviations along each embedding axis', making the
    magnitude comparable across layers of very different weight scale.
    Returns [k] tensor (float, cpu).
    """
    ev = getattr(decomp.decomposer, "eigenvalues", None)
    if ev is None:
        # UMAP/FKPCA fallback — unit scale
        return torch.ones(decomp.k, dtype=torch.float32)
    return ev.detach().cpu().clamp(min=1e-12).sqrt().float()


# The intervention core 
class InterventionRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        reconstructor: WeightReconstructor,
        evaluator: Evaluator,
        task: str,
        device: str,
    ):
        self._model = model
        self._recon = reconstructor
        self._eval  = evaluator
        self._task  = task
        self._device = device

        # Snapshot the clean, fully-reconstructed baseline weights ONCE so we
        # can restore exactly between conditions. We reconstruct-and-load from
        # the untouched initial Z so the "baseline" is the reconstruction, not
        # the raw checkpoint. This isolates the *perturbation* effect from any
        # (tiny) reconstruction error.
        self._recon.reset()
        self._recon.reconstruct_and_load(self._model)
        self._clean_state = copy.deepcopy(self._model.state_dict())

    def _restore_clean(self) -> None:
        self._model.load_state_dict(self._clean_state)

    def baseline_metric(self) -> float:
        self._restore_clean()
        res = self._eval.run()
        return primary_metric(self._task, res.metrics)

    def _target_units(
        self,
        layer_key: str,
        sensitivity: pd.DataFrame | None,
        top_q: float,
        rng: np.random.Generator,
        random_subset: bool,
    ) -> np.ndarray:
        """
        Return the unit indices to perturb in this layer.
        - random_subset=False: top_q fraction by sensitivity (or displacement)
        - random_subset=True : a random top_q fraction (matched count)
        """
        C = self._recon._current_Z[layer_key].shape[0]
        n_target = max(1, int(round(top_q * C)))

        if random_subset:
            return rng.choice(C, size=n_target, replace=False)

        if sensitivity is not None:
            sub = sensitivity[sensitivity["layer_key"] == layer_key]
            if len(sub) >= n_target:
                # highest sensitivity units
                order = sub.sort_values("sensitivity", ascending=False)
                return order["unit_idx"].to_numpy()[:n_target].astype(np.int64)

        # Fallback: highest displacement in the embedding (||z_c||)
        Z = self._recon._initial_Z[layer_key]
        disp = Z.norm(dim=1).numpy()
        return np.argsort(-disp)[:n_target].astype(np.int64)

    def run_condition(
        self,
        condition: str,             # "geometric" | "random" | "residual"
        epsilon: float,
        sensitivity: pd.DataFrame | None,
        top_q: float,
        seed: int,
    ) -> float:
        """
        Apply the perturbation for one condition across all layers, evaluate,
        and return the primary metric. Restores clean weights first.
        """
        rng = np.random.default_rng(seed)
        self._restore_clean()
        self._recon.reset()   # reset Z back to initial before perturbing

        for layer_key in self._recon.layer_keys:
            decomp = self._recon._decompositions[layer_key]
            Z0 = self._recon._initial_Z[layer_key]          # [C, k]
            C, k = Z0.shape
            scale = _axis_scale(decomp)                     # [k]

            idx = self._target_units(
                layer_key, sensitivity, top_q, rng,
                random_subset=(condition == "random"),
            )
            if len(idx) == 0:
                continue

            if condition in ("geometric", "random"):
                # Random unit-direction per targeted unit, scaled per-axis by
                # epsilon * sqrt(eigenvalue). Perturb the embedding coordinate.
                dirs = torch.from_numpy(
                    rng.standard_normal(size=(len(idx), k)).astype(np.float32)
                )
                dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
                delta = dirs * scale.unsqueeze(0) * epsilon   # [n, k]
                Znew = self._recon._current_Z[layer_key].clone()
                Znew[idx] = Z0[idx] + delta
                self._recon._current_Z[layer_key] = Znew

            elif condition == "residual":
                # Perturb the DISCARDED residual subspace by the SAME L2
                # magnitude as the geometric condition would, leaving Z fixed.
                res = self._recon._residuals.get(layer_key)
                if res is None:
                    continue  # no residual stored (non-PCA) — skip
                # Match magnitude: geometric perturbation has L2 = epsilon *
                # ||scale|| per unit (since dir is unit-norm, scaled per axis).
                target_mag = float((scale * epsilon).norm())
                rdir = torch.from_numpy(
                    rng.standard_normal(size=(len(idx), res.shape[1])).astype(np.float32)
                )
                rdir = rdir / rdir.norm(dim=1, keepdim=True).clamp(min=1e-12)
                res_new = res.clone()
                res_new[idx] = res[idx] + rdir * target_mag
                self._recon._residuals[layer_key] = res_new

        # Reconstruct perturbed weights into the live model and evaluate
        self._recon.reconstruct_and_load(self._model)
        res = self._eval.run()
        metric = primary_metric(self._task, res.metrics)

        # Restore residuals if we mutated them
        if condition == "residual":
            self._recon.reset()  # resets Z; residuals restored below
            # _initial residuals: rebuild from the untouched projection is not
            # stored, so we re-clone from clean by resetting the reconstructor
            # object's residual cache. Simplest: reload decomposition residuals.
            # (Handled by caller re-instantiating if needed.)

        return metric


def main():
    ap = argparse.ArgumentParser(description="Quantified intervention experiment")
    ap.add_argument("task", choices=["classification", "segmentation"])
    ap.add_argument("--backbone", required=True,
                    choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["imagenet100", "isic2019", "voc2012_sbd", "isic2018"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--decomposers_path", required=True)
    ap.add_argument("--projection_path", required=True)
    ap.add_argument("--val_transforms", default=None)
    ap.add_argument("--loss_config", default=None,
                    help="Optional loss YAML (only needed if you want loss logged)")
    ap.add_argument("--sensitivity_csv", default=None,
                    help="Per-unit sensitivity.csv; if given, geometric target "
                         "= highest-s_c units, else highest-displacement units")
    ap.add_argument("--ignore_index", type=int, default=255)
    ap.add_argument("--eval_size", type=int, default=500,
                    help="Validation samples for each evaluation pass")
    ap.add_argument("--top_q", type=float, default=0.10,
                    help="Fraction of units per layer to perturb")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                    help="Perturbation magnitudes in units of per-axis std")
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="Random seeds per condition (averaged)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_dir", default="./results/intervention")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _set_seeds(args.seed)

    # Map --eval_size onto main.py's anchor_size mechanism for the val loader
    args.subcommand = args.task
    args.anchor_size = args.eval_size
    args.feature_layer = None

    # Build model + load checkpoint
    print("[intervention] Building model...")
    model = build_model(
        backbone=args.backbone, variant=args.variant,
        task=args.task, n_classes=args.num_classes,
    ).to(args.device)
    model.eval()
    _load_checkpoint(model, args.checkpoint_path, args.device)

    # Val loader (identical path to main.py)
    val_loader = _build_val_loader(args)

    # Decompositions + projections
    decompositions, projections = _load_decompositions(
        args.decomposers_path, args.projection_path
    )
    reconstructor = WeightReconstructor(
        decompositions=decompositions,
        projections=projections,
        device=args.device,
    )

    # Optional loss (not required for metric-only intervention)
    loss_fn = None
    if args.loss_config:
        cfg = OmegaConf.load(args.loss_config)
        loss_fn = instantiate(cfg).to(args.device)

    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        task=args.task,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        loss_fn=loss_fn,
        amp=args.amp,
        ignore_index=args.ignore_index,
    )

    sensitivity = None
    if args.sensitivity_csv and os.path.exists(args.sensitivity_csv):
        sensitivity = pd.read_csv(args.sensitivity_csv)
        print(f"[intervention] Loaded sensitivity CSV "
              f"({len(sensitivity)} rows) — geometric target = top-s_c units")
    else:
        print("[intervention] No sensitivity CSV — geometric target = "
              "highest-displacement units")

    runner = InterventionRunner(
        model=model, reconstructor=reconstructor,
        evaluator=evaluator, task=args.task, device=args.device,
    )

    baseline = runner.baseline_metric()
    print(f"\n[intervention] Baseline {args.task} metric: {baseline:.4f}\n")

    rows: list[dict] = []
    for eps in args.epsilons:
        for condition in ("geometric", "random", "residual"):
            vals = []
            for s in range(args.n_seeds):
                # Fresh reconstructor state each run to keep residuals clean
                reconstructor_seed = WeightReconstructor(
                    decompositions=decompositions,
                    projections=projections,
                    device=args.device,
                )
                runner._recon = reconstructor_seed
                m = runner.run_condition(
                    condition=condition, epsilon=eps,
                    sensitivity=sensitivity, top_q=args.top_q,
                    seed=args.seed + 1000 * s,
                )
                vals.append(m)
            mean_m = float(np.mean(vals))
            std_m  = float(np.std(vals))
            drop   = baseline - mean_m
            rows.append({
                "task": args.task, "model": f"{args.backbone}_{args.variant}",
                "dataset": args.dataset, "epsilon": eps,
                "condition": condition,
                "metric_mean": mean_m, "metric_std": std_m,
                "drop_from_baseline": drop,
                "baseline": baseline,
            })
            print(f"  eps={eps:<4}  {condition:<10}  "
                  f"metric={mean_m:.4f}±{std_m:.4f}  drop={drop:+.4f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "intervention_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[intervention] Saved: {out_csv}")

    # Quick ratio summary
    print("\n[intervention] Geometric-vs-random drop ratio per epsilon:")
    for eps in args.epsilons:
        g = df[(df.epsilon == eps) & (df.condition == "geometric")]["drop_from_baseline"].iloc[0]
        r = df[(df.epsilon == eps) & (df.condition == "random")]["drop_from_baseline"].iloc[0]
        res = df[(df.epsilon == eps) & (df.condition == "residual")]["drop_from_baseline"].iloc[0]
        ratio = g / r if abs(r) > 1e-9 else float("inf")
        print(f"  eps={eps:<4}  geometric={g:+.4f}  random={r:+.4f}  "
              f"residual={res:+.4f}  (geo/rand = {ratio:.2f}x)")


if __name__ == "__main__":
    main()
