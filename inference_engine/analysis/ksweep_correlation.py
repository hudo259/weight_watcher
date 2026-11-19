"""
The displacement->sensitivity sign-reversal in linear/projection layers might be an artifact of taking the
sensitivity gradient w.r.t. a 3-D coordinate that discards most of the
trajectory variance in exactly those layers.

For each k in the sweep we recompute s_c(k) per unit, then recompute the
per-(layer_type) Spearman correlation rho( delta_c , s_c(k) ) and report
whether the negative sign in linear/projection layers persists.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import spearmanr

# This script lives in inference_engine/analysis/ but imports from
# inference_engine/ (main, models, inference, projections). Add the parent
# directory to sys.path so those imports resolve regardless of the current
# working directory the script is launched from.
_ANALYSIS_DIR = Path(__file__).resolve().parent          # inference_engine/analysis
_ENGINE = _ANALYSIS_DIR.parent                            # inference_engine
for _p in (_ENGINE, _ANALYSIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.factory import build_model
from projections.common.templates import expand_templates
from projections.common.layer_decomposition import (
    _flatten_weight, _load_trajectories,
)
from projections.projection import _register_resolvers, get_state_dict_key

# Reuse the sensitivity machinery
from gradient_sensitivity import build_anchor_loader
from shared import CkptMode, load_checkpoint


# Fit full-basis PCA per layer (up to k_max) and cache eigenvectors
def fit_full_basis(
    trajectories: torch.Tensor, k_max: int
) -> dict[str, torch.Tensor]:
    """
    Replicate pca.py.fit() but keep k_max eigenvectors. Returns the pieces the
    sensitivity projection needs. trajectories: [C, T, d] raw weights.
    """
    C, T, d = trajectories.shape
    init = trajectories[:, 0, :]
    delta = trajectories - init.unsqueeze(1)
    X_all = delta.reshape(C * T, d)
    mean = X_all.mean(dim=0)
    X_c = X_all - mean

    # full_matrices=False -> Vh is [min(CT,d), d]
    _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    kmax = min(k_max, Vh.shape[0])
    V = Vh[:kmax].T                      # [d, kmax]  (nested: V[:, :k] is top-k)
    eig = (S[:kmax] ** 2) / max(C * T - 1, 1)
    return {"V": V.float(), "eig": eig.float(), "d": d, "C": C}


# One backward pass -> raw weight-gradient per layer
def accumulate_raw_gradients(
    model: nn.Module,
    layer_keys: list[str],
    loader,
    loss_fn: nn.Module,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Accumulate the mean raw weight gradient dL/dW for each decomposed layer
    over the anchor set. k-independent. Mirrors
    gradient_sensitivity.compute_gradient_sensitivity's accumulation.
    """
    model.eval().to(device)
    param_map: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if name in layer_keys:
            param_map[name] = param
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    grad_accum = {k: torch.zeros_like(param_map[k], device=device)
                  for k in param_map}
    n_batches = 0
    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()

        for k, p in param_map.items():
            if p.grad is not None:
                grad_accum[k] += p.grad.detach()

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"    batch {n_batches}/{len(loader)}", flush=True)

    for p in model.parameters():
        p.requires_grad_(True)
    return {k: v / max(n_batches, 1) for k, v in grad_accum.items()}


def _flatten_grad(g_raw: torch.Tensor, layer_type: str, C: int,
                  num_heads: int | None = None) -> torch.Tensor:
    """
    Flatten a raw weight-gradient to [C, d] matching the PCA/_flatten_weight
    convention so that g_flat @ V_k is well-defined. C is the number of
    semantic units the PCA basis was fit on (rows for conv/linear, heads for
    mha_head).
    """
    if layer_type == "depthwise_conv":
        # weight is [C,1,kH,kW]; _flatten_weight squeezes dim 1
        return g_raw.squeeze(1).view(C, -1)

    if layer_type in ("conv2d", "conv1d"):
        return g_raw.view(C, -1)

    if layer_type in ("linear", "attention_query", "attention_key",
                      "attention_value", "attention_output",
                      "attention_fused_qkv"):
        if g_raw.dim() == 4:
            g_raw = g_raw.squeeze(-1).squeeze(-1)
        return g_raw

    if layer_type in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        if g_raw.dim() == 4:
            g_raw = g_raw.squeeze(-1).squeeze(-1)

        total_out, in_dim = g_raw.shape
        return g_raw.view(num_heads, (total_out // num_heads) * in_dim)
    return g_raw.view(C, -1)


def main():
    ap = argparse.ArgumentParser(description="k-sweep displacement-sensitivity correlation")
    ap.add_argument("task", choices=["classification", "segmentation"])
    ap.add_argument("--backbone", required=True, choices=["convnextv2", "mit", "cyclemlp"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint_path", required=True,
                    help="Final model checkpoint (for the gradient)")
    ap.add_argument("--input_dir", required=True,
                    help="Training checkpoint dir (for the trajectory SVD)")
    ap.add_argument("--config", required=True,
                    help="Decomposition config YAML (layer templates)")
    ap.add_argument("--val_transforms", default=None)
    ap.add_argument("--loss_config", required=True)
    ap.add_argument("--trajectory_csv", required=True,
                    help="trajectory_geometry.csv — source of displacement_norm")
    ap.add_argument("--k_values", type=int, nargs="+", default=[3, 5, 10, 20])
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument("--anchor_size", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="analysis/results/k_sweep")
    args = ap.parse_args()

    _register_resolvers()
    os.makedirs(args.out_dir, exist_ok=True)
    args.subcommand = args.task
    model_name = f"{args.backbone}_{args.variant}"
    task_full = f"{args.task}_{args.dataset}"
    k_max = max(args.k_values)

    # Model + anchor loader + loss
    print("[k-sweep] Building model + anchor loader...")
    model = build_model(backbone=args.backbone, variant=args.variant,
                        task=args.task, n_classes=args.num_classes).to(args.device)
    load_checkpoint(model, args.checkpoint_path, args.device, CkptMode.STRICT)
    loader = build_anchor_loader(args)
    loss_cfg = OmegaConf.load(args.loss_config)
    loss_fn = instantiate(loss_cfg).to(args.device)

    # Layer templates + checkpoints for the SVD
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    layer_mapping = expand_templates(config)
    layer_keys = list(layer_mapping.keys())

    root_dir = Path(args.input_dir)
    ckpts = sorted({p.resolve(): p for p in itertools.chain.from_iterable(
        root_dir.rglob(pat) for pat in ("*.pt", "*.pth"))}.values(),
        key=lambda p: p.name)
    ckpts = [str(p) for p in ckpts]
    sdk = get_state_dict_key(ckpts[0])

    # One backward pass: raw gradients
    print("[k-sweep] Accumulating raw weight gradients (one pass)...")
    raw_grads = accumulate_raw_gradients(model, layer_keys, loader, loss_fn, args.device)

    # Per layer: fit full basis, cache flattened gradient
    print("[k-sweep] Fitting full PCA bases (up to k_max=%d)..." % k_max)
    per_layer = {}   # layer_key -> {V[d,kmax], g_flat[C,d], layer_type}
    for layer_key, info in layer_mapping.items():
        if layer_key not in raw_grads:
            continue

        # bare string, or dict only for per-layer overrides (templates.py)
        layer_type = info["type"] if isinstance(info, dict) else info
        nh = info.get("num_heads") if isinstance(info, dict) else None
        flatten_fn = lambda w, lt=layer_type, n=nh: _flatten_weight(w.float(), lt, n)
        try:
            traj, _ = _load_trajectories(ckpts, layer_key, sdk, flatten_fn, "cpu")
        except KeyError:
            continue

        basis = fit_full_basis(traj, k_max)
        g_flat = _flatten_grad(raw_grads[layer_key].float().cpu(),
                               layer_type, basis["C"], nh)
        per_layer[layer_key] = {
            "V": basis["V"], "g_flat": g_flat, "layer_type": layer_type,
        }
        print(f"  {layer_type:<16} {layer_key:<45} d={basis['d']}")

    # Displacement from trajectory CSV (k-independent)
    traj_df = pd.read_csv(args.trajectory_csv)
    # filter to this model+task (bidirectional norm like layout.py)
    def _norm(s): return "".join(c for c in str(s).lower() if c.isalnum())
    tdf = traj_df[traj_df["model"].apply(_norm) == _norm(model_name)]
    dn = _norm(args.dataset)
    tdf = tdf[tdf["task"].apply(lambda t: dn in _norm(t) or _norm(t) in dn)]
    disp_lookup = tdf.set_index(["layer_key", "unit_idx"])["displacement_norm"]

    # Sweep k: recompute s_c(k), join with displacement, correlate
    all_rows = []
    for k in args.k_values:
        per_unit = []

        for layer_key, d in per_layer.items():
            Vk = d["V"][:, :k]                     # [d, k]  nested slice
            g_z = d["g_flat"] @ Vk                 # [C, k]
            s = g_z.norm(dim=1).numpy()            # [C]

            for unit_idx in range(len(s)):
                disp = disp_lookup.get((layer_key, unit_idx), np.nan)
                per_unit.append({
                    "layer_key": layer_key,
                    "layer_type": d["layer_type"],
                    "unit_idx": unit_idx,
                    "k": k,
                    "sensitivity": float(s[unit_idx]),
                    "displacement_norm": float(disp),
                })
        pud = pd.DataFrame(per_unit)

        # Per-layer-type Spearman rho(displacement, sensitivity) at this k
        for lt, sub in pud.groupby("layer_type"):
            m = sub[["displacement_norm", "sensitivity"]].dropna()

            if len(m) < 5:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(m["displacement_norm"], m["sensitivity"])

            all_rows.append({
                "model": model_name, "task": task_full,
                "layer_type": lt, "k": k, "n_units": len(m),
                "rho_disp_sens": float(rho), "p": float(p),
            })

    res = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.out_dir, f"ksweep_{model_name}_{task_full}.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n[k-sweep] Saved: {out_csv}\n")

    # Pivot for readability: layer_type x k
    pivot = res.pivot_table(index="layer_type", columns="k",
                            values="rho_disp_sens")
    print("Spearman rho(displacement, sensitivity) by layer type x k:")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Highlight the rebuttal-relevant layers
    print("\nSign-reversal check (linear / projection layers):")
    for lt in pivot.index:
        if "linear" in lt or "attention" in lt or lt == "conv2d":
            vals = pivot.loc[lt]
            signs = "".join("-" if v < 0 else "+" for v in vals if not np.isnan(v))
            print(f"  {lt:<20} rho across k = "
                  f"{[f'{v:+.3f}' for v in vals]}  signs=[{signs}]")


if __name__ == "__main__":
    main()
