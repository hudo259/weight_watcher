"""
Layer decomposition across training checkpoints.

Because just loading all layers is unfeasible for the hardware we have here,
layers are processed in chunks. For each chunk:
  1. Single pass through all T checkpoints, extracting only the chunk's
     layer weights. Peak RAM = chunk_size x max_layer_size.
  2. Parallel fitting (pure computation, no I/O).
  3. Free the chunk before loading the next.

This keeps RAM proportional to chunk_size rather than to the total model size, while reducing disk reads by roughly a factor of N_layers / chunk_size.

Layer-level shared basis:

All three decomposition methods (PCA, FourierKPCA, UMAP) fit a SINGLE shared k-dimensional basis per layer across all C semantic units.
This ensures that cross-unit distances, clustering, and geometric comparisons in the resulting 3D point cloud are directly meaningful.
Per-unit bases (C separate decompositions) would place each unit in an incompatible coordinate system, invalidating any inter-unit analysis.
"""

from __future__ import annotations

import gc
import torch
from dataclasses import dataclass
from typing import Any, Callable

from projections.decompositions.base_decomposition import BaseDecomposer
from projections.decompositions.pca import PCADecomposer
from projections.decompositions.umap import UMAPDecomposer
from projections.decompositions.fourier_kernel_pca import FourierKPCADecomposer


# Decomposer factory
def build_decomposer(method: str, k: int, **kwargs) -> BaseDecomposer:
    if method == "pca":
        return PCADecomposer(k=k)
    elif method == "umap":
        return UMAPDecomposer(k=k, **kwargs)
    elif method == "fourier_kpca":
        return FourierKPCADecomposer(k=k, **kwargs)
    else:
        raise ValueError(
            f"Unknown decomposition method: {method!r}. "
            "Choose from: pca, umap, fourier_kpca"
        )


def load_decomposer(data: dict) -> BaseDecomposer:
    method = data.get("method")
    if method == "pca":
        return PCADecomposer.from_dict(data)
    elif method == "umap":
        return UMAPDecomposer.from_dict(data)
    elif method == "fourier_kpca":
        return FourierKPCADecomposer.from_dict(data)
    else:
        raise ValueError(f"Unknown method in saved dict: {method!r}")


@dataclass
class LayerDecomposition:
    """Fitted decomposition for one layer."""
    layer_key:            str
    layer_type:           str
    method:               str
    decomposer:           BaseDecomposer
    num_components:       int
    original_shape:       tuple[int, ...]
    trajectory_points_3d: torch.Tensor | None = None   # [C, T, k]
    checkpoint_paths:     list[str] | None    = None

    @property
    def k(self) -> int:
        return self.decomposer.k

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory_points_3d is not None

    def to_dict(self) -> dict:
        d = {
            "layer_key":      self.layer_key,
            "layer_type":     self.layer_type,
            "method":         self.method,
            "decomposer":     self.decomposer.to_dict(),
            "num_components": self.num_components,
            "original_shape": self.original_shape,
        }
        if self.trajectory_points_3d is not None:
            d["trajectory_points_3d"] = self.trajectory_points_3d.cpu()
        if self.checkpoint_paths is not None:
            d["checkpoint_paths"] = self.checkpoint_paths
        return d

    @staticmethod
    def from_dict(data: dict) -> "LayerDecomposition":
        return LayerDecomposition(
            layer_key=data["layer_key"],
            layer_type=data["layer_type"],
            method=data["method"],
            decomposer=load_decomposer(data["decomposer"]),
            num_components=data["num_components"],
            original_shape=tuple(data["original_shape"]),
            trajectory_points_3d=data.get("trajectory_points_3d"),
            checkpoint_paths=data.get("checkpoint_paths"),
        )


def _flatten_weight(
    weight: torch.Tensor,
    layer_type: str,
    num_heads: int | None = None,
) -> torch.Tensor:
    """Reshape a weight tensor to [C, d]."""
    if layer_type in ("conv2d", "conv1d"):
        return weight.view(weight.shape[0], -1)

    elif layer_type == "depthwise_conv":
        return weight.squeeze(1).view(weight.shape[0], -1)

    elif layer_type in (
        "linear",
        "attention_query", "attention_key",
        "attention_value", "attention_output",
        "attention_fused_qkv",
    ):
        if weight.dim() == 4:
            weight = weight.squeeze(-1).squeeze(-1)
        return weight

    elif layer_type in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        if weight.dim() == 4:
            weight = weight.squeeze(-1).squeeze(-1)
        if num_heads is None:
            raise ValueError(
                f"layer_type='{layer_type}' requires num_heads"
            )
        total_out, in_dim = weight.shape
        if total_out % num_heads != 0:
            raise ValueError(
                f"Weight first dim ({total_out}) not divisible by "
                f"num_heads ({num_heads})"
            )
        head_dim = total_out // num_heads
        return weight.view(num_heads, head_dim * in_dim)

    else:
        raise ValueError(
            f"Unknown layer type: {layer_type!r}. "
            "See projections/templates.py for supported types."
        )


def _original_shape(
    weight_shape: torch.Size,
    layer_type: str,
    num_heads: int | None,
) -> tuple[int, ...]:
    if layer_type in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        total_out = weight_shape[0]
        in_dim    = weight_shape[1]
        head_dim  = total_out // num_heads
        return (num_heads, head_dim, in_dim)
    else:
        return tuple(weight_shape[1:])


# Trajectory loading helpers
def _get_state(path: str, state_dict_key: str | None,
               device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt[state_dict_key] if state_dict_key else ckpt


def _load_weight(state: dict, layer_key: str) -> torch.Tensor:
    w = state[layer_key].float()
    if not torch.isfinite(w).all():
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w


def _load_trajectories(
    checkpoint_paths: list[str],
    layer_key: str,
    state_dict_key: str | None,
    flatten_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, Any]:
    """
    Load one layer's raw weights across all T checkpoints.

    Returns:
        trajectories: [C, T, d]  raw weight values (NOT differences)
        weight_shape: shape of the raw weight tensor
    """
    first_state  = _get_state(checkpoint_paths[0], state_dict_key, device)
    first_weight = _load_weight(first_state, layer_key)
    first_flat   = flatten_fn(first_weight)
    C, d         = first_flat.shape
    T            = len(checkpoint_paths)

    trajectories = torch.zeros(C, T, d, dtype=torch.float32, device=device)
    trajectories[:, 0, :] = first_flat

    for t, path in enumerate(checkpoint_paths[1:], start=1):
        sd = _get_state(path, state_dict_key, device)
        trajectories[:, t, :] = flatten_fn(_load_weight(sd, layer_key))

    return trajectories, first_weight.shape


def _project_trajectory(
    trajectories: torch.Tensor,
    decomposer: BaseDecomposer,
) -> torch.Tensor:
    """
    Project all T trajectory snapshots for all C units.

    Args:
        trajectories: [C, T, d]  raw weights
        decomposer:   fitted, transform([C, d]) -> [C, k]

    Returns:
        [C, T, k]  on CPU
    """
    C, T, _ = trajectories.shape
    traj_3d = torch.zeros(C, T, decomposer.k)
    for t in range(T):
        z_t, _ = decomposer.transform(trajectories[:, t, :])
        traj_3d[:, t, :] = z_t.detach().cpu()
    return traj_3d


# Chunked single-pass loader
def _load_chunk_trajectories(
    checkpoint_paths: list[str],
    chunk_mapping: dict[str, Any],
    state_dict_key: str | None,
    load_device: str = "cpu",
) -> dict[str, tuple[torch.Tensor, Any]]:
    """
    Single pass through all T checkpoints, loading only the layers in
    chunk_mapping.

    Peak RAM = len(chunk_mapping) x max_layer_trajectory_size.

    Returns:
        {layer_key: (trajectory [C, T, d], raw_weight_shape)}
    """
    T = len(checkpoint_paths)

    # Inspect first checkpoint to get shapes and pre-allocate
    first_sd = _get_state(checkpoint_paths[0], state_dict_key, "cpu")

    trajectories: dict[str, tuple[torch.Tensor, Any]] = {}
    for layer_key, layer_info in chunk_mapping.items():
        layer_type = layer_info["type"] if isinstance(layer_info, dict) else layer_info
        num_heads  = layer_info.get("num_heads") if isinstance(layer_info, dict) else None

        if layer_key not in first_sd:
            print(f"  WARNING: {layer_key!r} not found in checkpoint! Skipping...")
            continue

        w    = _load_weight(first_sd, layer_key)
        flat = _flatten_weight(w, layer_type, num_heads)
        C, d = flat.shape

        traj = torch.zeros(C, T, d, dtype=torch.float32, device=load_device)
        traj[:, 0, :] = flat.to(load_device)
        trajectories[layer_key] = (traj, w.shape)

    del first_sd

    # Single pass through remaining checkpoints
    for t, path in enumerate(checkpoint_paths[1:], start=1):
        sd = _get_state(path, state_dict_key, "cpu")
        for layer_key, layer_info in chunk_mapping.items():
            if layer_key not in trajectories:
                continue
            layer_type = layer_info["type"] if isinstance(layer_info, dict) else layer_info
            num_heads  = layer_info.get("num_heads") if isinstance(layer_info, dict) else None
            w    = _load_weight(sd, layer_key)
            flat = _flatten_weight(w, layer_type, num_heads)
            trajectories[layer_key][0][:, t, :] = flat.to(load_device)
        del sd

    return trajectories


def _estimate_chunk_gb(
    chunk_mapping: dict[str, Any],
    T: int,
    first_checkpoint: str,
    state_dict_key: str | None,
) -> float:
    """Estimate peak RAM (GB) for one chunk of trajectory tensors."""
    try:
        sd = _get_state(first_checkpoint, state_dict_key, "cpu")
        total = 0
        for layer_key, layer_info in chunk_mapping.items():
            if layer_key not in sd:
                continue
            layer_type = layer_info["type"] if isinstance(layer_info, dict) else layer_info
            num_heads  = layer_info.get("num_heads") if isinstance(layer_info, dict) else None
            flat = _flatten_weight(sd[layer_key].float(), layer_type, num_heads)
            C, d = flat.shape
            total += C * T * d * 4
        del sd
        return total / 1024 ** 3
    except Exception:
        return 0.0

# Worker: fit one layer
def fit_layer(
    layer_key:       str,
    trajectory:      torch.Tensor,    # [C, T, d] raw weights
    weight_shape:    Any,
    layer_type:      str,
    k:               int,
    method:          str = "pca",
    method_kwargs:   dict | None = None,
    save_trajectory: bool = False,
) -> tuple[str, LayerDecomposition]:
    """
    Fit a decomposer on a pre-loaded trajectory tensor.

    Receives raw weight trajectories [C, T, d]. The decomposer is
    responsible for computing difference matrices internally and fitting
    a layer-level shared basis.

    No I/O, pure computation. Called by Pool.starmap in decompose_all_layers
    or sequentially for num_workers=1.
    """
    num_heads = None
    if layer_type in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        num_heads = trajectory.shape[0]

    C, T, d = trajectory.shape

    decomposer = build_decomposer(method, k=k, **(method_kwargs or {}))
    print(f"  [{method.upper()}] {layer_key}: "
          f"fitting {C} units (T={T}, d={d})...")
    decomposer.fit(trajectory)

    traj_3d = None
    if save_trajectory:
        traj_3d = _project_trajectory(trajectory, decomposer)

    return layer_key, LayerDecomposition(
        layer_key=layer_key,
        layer_type=layer_type,
        method=method,
        decomposer=decomposer,
        num_components=C,
        original_shape=_original_shape(weight_shape, layer_type, num_heads),
        trajectory_points_3d=traj_3d,
        checkpoint_paths=None,
    )


# Main entry point: chunked decomposition
def decompose_all_layers(
    checkpoint_paths: list[str],
    layer_mapping:    dict[str, Any],
    method:           str,
    k:                int,
    state_dict_key:   str | None = None,
    method_kwargs:    dict | None = None,
    save_trajectory:  bool = False,
    num_workers:      int = 1,
    chunk_size:       int | None = None,
    device:           str = "cpu",
) -> dict[str, LayerDecomposition]:
    """
    Decompose all layers using chunked single-pass loading.

    For each chunk of chunk_size layers:
      1. Single pass through all T checkpoints extracting only those layers.
         Peak RAM = chunk_size x max_layer_trajectory_size.
      2. Fit decomposers in parallel (num_workers workers per chunk).
      3. Free the chunk before loading the next.

    Args:
        checkpoint_paths: T paths in chronological order.
        layer_mapping:    {layer_key: layer_info} from expand_templates().
        method:           "pca" | "umap" | "fourier_kpca"
        k:                Output dimensionality.
        state_dict_key:   Key to unwrap from each checkpoint dict.
        method_kwargs:    Extra kwargs forwarded to build_decomposer.
        save_trajectory:  Store [C, T, k] in each LayerDecomposition.
        num_workers:      Workers per chunk. UMAP forced to 1.
        chunk_size:       Layers per chunk. Defaults to num_workers.
                          Set higher than num_workers when using a single
                          CUDA worker to amortize I/O passes.
        device:           Passed through for compatibility; trajectories are
                          always loaded on CPU to avoid VRAM pressure.
                          Pass "cuda" or "cuda:N" to use GPU SVD inside PCA.

    Returns:
        {layer_key: LayerDecomposition}
    """
    # UMAP fits one model on all C units, no multiprocessing needed
    if method == "umap" and num_workers > 1:
        print("  Warning: UMAP uses a single fit, forcing num_workers=1")
        num_workers = 1

    if chunk_size is None:
        chunk_size = max(num_workers, 1)

    load_device = "cpu"   # always load trajectories to RAM
    fit_device  = device  # PCA SVD can run on cuda if requested

    layer_keys = list(layer_mapping.keys())
    n_chunks   = (len(layer_keys) + chunk_size - 1) // chunk_size
    chunks     = [
        layer_keys[i: i + chunk_size]
        for i in range(0, len(layer_keys), chunk_size)
    ]

    T = len(checkpoint_paths)
    print(f"  {len(layer_keys)} layers -> {n_chunks} chunks of ≤{chunk_size} "
          f"({T} checkpoints, {num_workers} worker(s) per chunk)")

    # FourierKPCA and UMAP are numpy-based, the GPU does nothing for them
    if method in ("fourier_kpca", "umap") and fit_device != "cpu":
        print(f"  Note: {method} is numpy-based; fit_device set to cpu")
        fit_device = "cpu"

    all_decompositions: dict[str, LayerDecomposition] = {}

    for chunk_idx, chunk_keys in enumerate(chunks):
        chunk_mapping = {k: layer_mapping[k] for k in chunk_keys}
        chunk_gb = _estimate_chunk_gb(
            chunk_mapping, T, checkpoint_paths[0], state_dict_key
        )
        print(f"\n Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_keys)} layers, ~{chunk_gb:.1f} GB)")

        # Phase 1: single-pass load (no I/O after this)
        chunk_trajectories = _load_chunk_trajectories(
            checkpoint_paths=checkpoint_paths,
            chunk_mapping=chunk_mapping,
            state_dict_key=state_dict_key,
            load_device=load_device,
        )

        # Phase 2: parallel fitting
        jobs = [
            (
                layer_key,
                traj.to(fit_device) if fit_device != "cpu" else traj,
                weight_shape,
                chunk_mapping[layer_key]["type"]
                    if isinstance(chunk_mapping[layer_key], dict)
                    else chunk_mapping[layer_key],
                k,
                method,
                method_kwargs,
                save_trajectory,
            )
            for layer_key in chunk_keys
            if layer_key in chunk_trajectories
            for traj, weight_shape in [chunk_trajectories[layer_key]]
        ]

        if num_workers > 1:
            # Use spawn to avoid CUDA re-initialization in forked workers
            ctx = torch.multiprocessing.get_context("spawn")
            with ctx.Pool(num_workers) as pool:
                results = pool.starmap(fit_layer, jobs)
        else:
            results = []
            for job in jobs:
                results.append(fit_layer(*job))

        all_decompositions.update(dict(results))

        del chunk_trajectories
        gc.collect()

    return all_decompositions

# Single-layer entry point (backward compatibility)
def decompose_layer(
    checkpoint_paths: list[str],
    layer_key: str,
    layer_type: str,
    k: int,
    device: str,
    state_dict_key: str | None = None,
    num_heads: int | None = None,
    method: str = "pca",
    method_kwargs: dict | None = None,
    save_trajectory: bool = False,
) -> tuple[str, LayerDecomposition]:
    """
    Original single-layer entry point. Kept for backward compatibility.
    For batch processing, use decompose_all_layers() instead.
    """
    flatten_fn = lambda w: _flatten_weight(w.float(), layer_type, num_heads)
    trajectories, weight_shape = _load_trajectories(
        checkpoint_paths, layer_key, state_dict_key, flatten_fn, device
    )

    return fit_layer(
        layer_key=layer_key,
        trajectory=trajectories,
        weight_shape=weight_shape,
        layer_type=layer_type,
        k=k,
        method=method,
        method_kwargs=method_kwargs,
        save_trajectory=save_trajectory,
    )

# Project a single checkpoint into the decomposition space
def project_model(
    checkpoint_path: str,
    decomp: LayerDecomposition,
    device: str = "cpu",
    state_dict_key: str | None = None,
) -> tuple[str, dict]:
    """
    Project one layer from a single checkpoint into the decomposition space.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd   = ckpt[state_dict_key] if state_dict_key else ckpt

    layer_key = decomp.layer_key
    if layer_key not in sd:
        raise KeyError(
            f"Layer '{layer_key}' not found in checkpoint '{checkpoint_path}'"
        )

    weight = _load_weight(sd, layer_key)

    num_heads = None
    if decomp.layer_type in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        num_heads = decomp.original_shape[0]

    weight_flat = _flatten_weight(weight, decomp.layer_type, num_heads)

    if weight_flat.shape[0] != decomp.num_components:
        raise ValueError(
            f"Component count mismatch for '{layer_key}': "
            f"checkpoint has {weight_flat.shape[0]}, "
            f"decomposition expects {decomp.num_components}"
        )

    return_residuals = isinstance(decomp.decomposer, PCADecomposer)
    points_3d, residuals = decomp.decomposer.transform(
        weight_flat, return_residuals=return_residuals
    )

    return layer_key, {
        "points_3d": points_3d.cpu(),
        "residuals": residuals.cpu() if residuals is not None else None,
    }
