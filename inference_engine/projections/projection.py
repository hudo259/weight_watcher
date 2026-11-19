"""
Project neural network weight trajectories into k-dimensional point clouds.

Call like this:
python projection.py <method> -i <checkpoint_dir> -m <best.pth> -c <config.yaml> -o <output_dir>

Methods:  pca  |  umap  |  fourier_kpca

Examples:
  # PCA with CUDA SVD, 8 workers, 16 layers per chunk
  python projection.py pca -i ./ckpts/epochs -m ./ckpts/best/best.pth \\
      -c configs/decomposition/convnextv2_base_classification.yaml \\
      -o ./projections/convnextv2_imagenet100_pca \\
      -k 3 --save_trajectory --num_workers 1 --chunk_size 16 --device cuda

  # FourierKPCA (numpy-based, always CPU)
  python projection.py fourier_kpca ... --kpca_n_rff_features 1024 \\
      --kpca_regularization 0.001 --num_workers 1 --chunk_size 2

  # UMAP (single fit per layer, num_workers forced to 1)
  python projection.py umap ... --num_workers 1 --chunk_size 16

Notes:
  - Checkpoints in --input_dir are sorted by filename; best.pth must NOT be
    in the same directory (use a best/ subdirectory and symlink epoch files).
  - FourierKPCA and UMAP always run on CPU regardless of --device.
  - PCA with --device cuda uses torch.linalg.svd on GPU; set --num_workers 1
    to avoid CUDA context conflicts in forked workers.
  - For UMAP, use --num_workers 1 (deadlocks with joblib multiprocessing).
"""

import colorsys
import itertools
import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import torch
from omegaconf import OmegaConf

from projections.common.templates import expand_templates
from projections.common.layer_decomposition import (
    decompose_all_layers,
    project_model,
)


# OmegaConf resolvers
def _register_resolvers():
    def range_resolver(start, stop=None, step=None):
        if stop is None:   return list(range(int(start)))
        if step is None:   return list(range(int(start), int(stop)))
        return list(range(int(start), int(stop), int(step)))

    def mul_resolver(a, b):    return int(a) * int(b)
    def list_resolver(*args):  return list(args)
    def concat_resolver(*args): return "".join(str(a) for a in args)
    def env_resolver(var, default=None): return os.getenv(var, default)

    for name, fn in [
        ("range",  range_resolver),
        ("mul",    mul_resolver),
        ("list",   list_resolver),
        ("concat", concat_resolver),
        ("env",    env_resolver),
    ]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass


# Helpers
def get_state_dict_key(checkpoint_path: str) -> str | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model_state", "state_dict", "model", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return key
    return None


def _make_palette(n: int,
                  exclude_hue: float = 50 / 360,
                  exclude_radius: float = 29.0) -> list:
    start = (exclude_hue + exclude_radius / 360.0) % 1.0
    span  = 1.0 - 2 * (exclude_radius / 360.0)
    return [
        list(colorsys.hsv_to_rgb((start + (i / n) * span) % 1.0, 0.75, 0.95))
        for i in range(n)
    ]


def _build_method_kwargs(args) -> dict:
    if args.subcommand == "umap":
        return {
            "n_neighbors": args.umap_n_neighbors,
            "min_dist":    args.umap_min_dist,
        }
    if args.subcommand == "fourier_kpca":
        kwargs = {"n_rff_features": args.kpca_n_rff_features}
        if args.kpca_gamma is not None:
            kwargs["gamma"] = args.kpca_gamma
        if args.kpca_regularization > 0:
            kwargs["regularization"] = args.kpca_regularization
        return kwargs
    return {}


# CLI argument parsers
def _pca_parser(p):
    return p


def _umap_parser(p):
    p.add_argument("--umap_n_neighbors", type=int,   default=15)
    p.add_argument("--umap_min_dist",    type=float, default=0.1)
    return p


def _fourier_kpca_parser(p):
    p.add_argument("--kpca_n_rff_features",  type=int,   default=1024,
                   help="Random Fourier features D (default 1024). "
                        "Higher D = better kernel approx, more memory.")
    p.add_argument("--kpca_gamma",           type=float, default=None,
                   help="RBF kernel gamma. None = auto-estimate.")
    p.add_argument("--kpca_regularization",  type=float, default=0.0,
                   help="Ridge regularization (default 0). "
                        "Set > 0 (e.g. 1e-3) when d > D.")
    return p


def _base_parser() -> ArgumentParser:
    p = ArgumentParser(add_help=False)

    # Required
    p.add_argument("-i", "--input_dir",       required=True,
                   help="Directory containing training checkpoints (*.pt, *.pth). "
                        "Must contain ONLY epoch checkpoints, keep best.pth "
                        "in a separate subdirectory.")
    p.add_argument("-m", "--model_checkpoint", required=True,
                   help="Final model checkpoint to project (e.g. best/best.pth)")
    p.add_argument("-c", "--config",           required=True,
                   help="Decomposition config YAML")
    p.add_argument("-o", "--output_dir",       default="./output")

    # Decomposition
    p.add_argument("-k", "--num_components", type=int, default=3,
                   help="Output dimensionality k (default: 3)")

    # Compute
    p.add_argument("-w", "--num_workers", type=int, default=1,
                   help="Parallel workers per chunk. "
                        "Use 1 for UMAP and when --device cuda. "
                        "Use N>1 for CPU PCA only.")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Layers per loading chunk (default: num_workers). "
                        "Set higher than num_workers for single-worker CUDA PCA "
                        "to reduce I/O passes. E.g. --num_workers 1 --chunk_size 16 "
                        "--device cuda.")
    p.add_argument("-d", "--device", default="cpu",
                   help="Device for computation. "
                        "Accepts 'cpu', 'cuda', or 'cuda:N' for a specific GPU. "
                        "FourierKPCA and UMAP always use CPU regardless. "
                        "PCA benefits from 'cuda' for GPU-accelerated SVD "
                        "(use --num_workers 1 to avoid CUDA fork issues).")

    # Trajectory
    p.add_argument("--save_trajectory", action="store_true",
                   help="Store [C, T, k] trajectory coordinates per layer. "
                        "Required for trajectory_geometry.py. "
                        "Not recommended for UMAP (very slow).")

    return p

def main() -> None:
    _register_resolvers()

    root_p = ArgumentParser(
        description="Decompose model weight trajectories to k-dimensional point clouds."
    )
    sub_p = root_p.add_subparsers(dest="subcommand", required=True)
    base  = _base_parser()

    pca_p = sub_p.add_parser("pca", parents=[base], conflict_handler="resolve")
    _pca_parser(pca_p)

    umap_p = sub_p.add_parser("umap", parents=[base], conflict_handler="resolve")
    _umap_parser(umap_p)

    kpca_p = sub_p.add_parser("fourier_kpca", parents=[base], conflict_handler="resolve")
    _fourier_kpca_parser(kpca_p)

    args = root_p.parse_args()

    # Validate device string
    if args.device not in ("cpu",) and not args.device.startswith("cuda"):
        root_p.error(
            f"Invalid --device {args.device!r}. "
            "Use 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:1')."
        )

    # Load config 
    print("=" * 70)
    print("Loading config...")
    print("=" * 70)

    config        = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    layer_mapping = expand_templates(config)

    print(f"  Model:  {config.get('name', 'Unnamed')}")
    print(f"  Layers: {len(layer_mapping)}")
    print()

    # Find checkpoints
    print("=" * 70)
    print("Finding checkpoints...")
    print("=" * 70)

    root_dir = Path(args.input_dir)
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.input_dir}")

    checkpoint_paths = sorted({
        p.resolve(): p
        for p in itertools.chain.from_iterable(
            root_dir.rglob(pat) for pat in ("*.pt", "*.pth")
        )
    }.values(), key=lambda p: p.name)
    checkpoint_paths = [str(p) for p in checkpoint_paths]

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No *.pt / *.pth files found in {args.input_dir}"
        )

    print(f"  Found {len(checkpoint_paths)} checkpoints")
    print(f"  First: {checkpoint_paths[0]}")
    print(f"  Last:  {checkpoint_paths[-1]}")

    state_dict_key = get_state_dict_key(checkpoint_paths[0])
    print(f"  State dict key: {state_dict_key!r}")
    print()

    # Fit decomposers
    print("=" * 70)
    print(f"Fitting {args.subcommand.upper()} decomposers...")
    print("=" * 70)

    method_kwargs = _build_method_kwargs(args)
    if method_kwargs:
        print(f"  Hyperparameters: {method_kwargs}")

    decompositions = decompose_all_layers(
        checkpoint_paths=checkpoint_paths,
        layer_mapping=layer_mapping,
        method=args.subcommand,
        k=args.num_components,
        state_dict_key=state_dict_key,
        method_kwargs=method_kwargs,
        save_trajectory=args.save_trajectory,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    total_components = sum(d.num_components for d in decompositions.values())
    print(f"\n  Done. {len(decompositions)} layers, "
          f"{total_components:,} components")
    print()

    # Project final model checkpoint
    print("=" * 70)
    print("Projecting final model...")
    print("=" * 70)

    if not Path(args.model_checkpoint).exists():
        raise FileNotFoundError(f"Not found: {args.model_checkpoint}")

    model_sdk = get_state_dict_key(args.model_checkpoint)

    projection_jobs = [
        (args.model_checkpoint, decomp, "cpu", model_sdk)
        for decomp in decompositions.values()
    ]
    print(f"  Projecting {len(projection_jobs)} layers...")

    # Always run project_model sequentially on CPU, it's fast enough (single checkpoint)
    proj_results = [project_model(*job) for job in projection_jobs]
    projections  = dict(proj_results)

    # Attach trajectory coordinates
    if args.save_trajectory:
        for layer_key, decomp in decompositions.items():
            if decomp.has_trajectory:
                projections[layer_key]["trajectory_3d"]    = decomp.trajectory_points_3d
                projections[layer_key]["checkpoint_paths"] = decomp.checkpoint_paths

    print("  Done.")
    print()

    # Save
    print("=" * 70)
    print("Saving...")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = _make_palette(len(decompositions))
    for i, layer_key in enumerate(decompositions.keys()):
        projections[layer_key]["color"] = torch.tensor(
            palette[i], dtype=torch.float32
        )

    model_name = config.get("name", "model").replace(" ", "_").lower()

    proj_data = {
        "version": "1.0",
        "metadata": {
            "model_name":               config.get("name", "Unnamed"),
            "architecture":             config.get("architecture", "Unknown"),
            "method":                   args.subcommand,
            "k":                        args.num_components,
            "has_residuals":            True,
            "has_trajectory":           args.save_trajectory,
            "total_layers":             len(projections),
            "total_components":         total_components,
            "model_checkpoint":         args.model_checkpoint,
            "num_training_checkpoints": len(checkpoint_paths),
        },
        "projections": projections,
    }
    proj_path = output_dir / f"{model_name}_projected.pt"
    torch.save(proj_data, proj_path)
    print(f"  Projections:  {proj_path}")

    decomp_data = {
        layer_key: decomp.to_dict()
        for layer_key, decomp in decompositions.items()
    }
    decomp_path = output_dir / f"{model_name}_decomposers.pt"
    torch.save(decomp_data, decomp_path)
    print(f"  Decomposers:  {decomp_path}")

    proj_mb   = proj_path.stat().st_size   / 1024 ** 2
    decomp_mb = decomp_path.stat().st_size / 1024 ** 2

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Model:        {config.get('name', 'Unnamed')}")
    print(f"  Method:       {args.subcommand}")
    print(f"  k:            {args.num_components}")
    print(f"  Layers:       {len(decompositions)}")
    print(f"  Components:   {total_components:,}")
    print(f"  Trajectory:   {'yes' if args.save_trajectory else 'no'}")
    print(f"  Projections:  {proj_mb:.1f} MB")
    print(f"  Decomposers:  {decomp_mb:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
