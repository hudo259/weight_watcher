"""
Use like this:
-----
    Subcommands: classification | segmentation

  # Classification
  python main.py classification \\
      --backbone convnextv2 --variant tiny --num_classes 100 \\
      --checkpoint_path checkpoints/best.pth \\
      --decomposers_path projections/model_decomposers.pt \\
      --projection_path  projections/model_projected.pt \\
      --img_dirs data/imagenet100/val

  # Segmentation (via UperNet)
  python main.py segmentation \\
      --backbone mit --variant b2 --num_classes 21 \\
      --checkpoint_path checkpoints/best.pth \\
      --decomposers_path projections/model_decomposers.pt \\
      --projection_path  projections/model_projected.pt \\
      --img_dirs data/voc/val/images \\
      --mask_dirs data/voc/val/masks
"""

import argparse
import os
import signal
import sys
from math import ceil
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torchvision.transforms.v2 as v2
from omegaconf import OmegaConf
from pydoc import locate
from hydra.utils import instantiate

from models.factory import build_model
from inference.inference_engine import InferenceEngine
from projections.common.layer_decomposition import LayerDecomposition

# Use the same datasets as training val/test split becomes the anchor data.
# build_loaders returns (train_loader, val_loader); we use the val_loader.
from trainer.common.datasets import build_loaders

OmegaConf.register_new_resolver("locate", locate, replace=True)

def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)

    # IPC
    p.add_argument("-a", "--address", default="tcp://*:9999")

    # Model
    p.add_argument("--backbone",    required=True,
                   choices=["convnextv2", "mit", "cyclemlp"],
                   help="Backbone architecture")
    p.add_argument("--variant",     required=True,
                   help="Model variant, e.g. 'tiny', 'b2'")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Number of output classes")
    p.add_argument("--checkpoint_path", required=True,
                   help="Path to model checkpoint (*.pth saved by trainer)")

    # Decomposition files (from projection.py)
    p.add_argument("--decomposers_path", required=True,
                   help="Path to *_decomposers.pt from projection.py")
    p.add_argument("--projection_path",  required=True,
                   help="Path to *_projected.pt from projection.py")

    # Dataset
    p.add_argument("--dataset",     required=True,
                   choices=["imagenet100", "isic2019", "voc2012_sbd", "isic2018"],
                   help="Dataset name (same values as training)")
    p.add_argument("--data_root",   required=True,
                   help="Root directory of the dataset (same as training --data_root)")
    p.add_argument("--anchor_size", type=int, default=100,
               help="Number of validation samples to use as anchor set (default: 100)")
    p.add_argument("--val_transforms", default=None,
               help="Path to val transforms YAML config, "
                    "e.g. ../configs/transform/imagenet100_val.yaml. "
                    "If not provided, falls back to hardcoded resize+normalize.")

    # Runtime
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=108)
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size",  type=int, default=512)

    # Optional
    p.add_argument("--feature_layer",   default=None,
                   help="Layer name to extract spatial features from")
    p.add_argument("--pca_target_dims", type=int, default=-1,
                   help="PCA-reduce features before sending (-1 = off) (For Feature Space projection)")

    # CSV files for layouts
    p.add_argument("--trajectory_geometry_csv", default=None)
    p.add_argument("--sensitivity_csv",         default=None)
    p.add_argument("--variance_explained_csv",  default=None)

    return p


def _add_classification_args(p: argparse.ArgumentParser):
    # No extra args needed, dataset + data_root in base are sufficient
    return p


def _add_segmentation_args(p: argparse.ArgumentParser):
    p.add_argument("--ignore_index", type=int, default=255, help="ignore pixel value (for VOC2012 and ISIC 2018)")
    return p

# Helpers

def _set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(model: torch.nn.Module, path: str, device: str) -> None:
    """Load a checkpoint saved by our Trainer (key: 'model_state')."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Support our trainer's format and common alternatives
    for key in ("model_state", "state_dict", "model", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            model.load_state_dict(ckpt[key])
            print(f"[main] Loaded checkpoint '{path}' (key='{key}')")
            return
    # Checkpoint IS the state dict
    model.load_state_dict(ckpt)
    print(f"[main] Loaded checkpoint '{path}' (raw state dict)")


def _load_decompositions(
    decomposers_path: str,
    projection_path: str,
) -> tuple[dict, dict]:
    """
    Load fitted decomposers and initial 3-D projections.

    Returns:
        decompositions: layer_key -> LayerDecomposition
        projections:    layer_key -> {"points_3d": Tensor, "residuals": ..., "color": Tensor}
    """
    # Decomposers file: layer_key -> dict from LayerDecomposition.to_dict()
    decomp_raw     = torch.load(decomposers_path, map_location="cpu", weights_only=False)
    decompositions = {k: LayerDecomposition.from_dict(v) for k, v in decomp_raw.items()}

    # Projection file: {"version": ..., "metadata": ..., "projections": {...}}
    proj_data   = torch.load(projection_path, map_location="cpu", weights_only=False)
    projections = proj_data["projections"]

    missing = set(decompositions) - set(projections)
    if missing:
        raise ValueError(
            f"{len(missing)} layer(s) in decomposers file missing from projection file: "
            f"{list(missing)[:5]} ..."
        )
    return decompositions, projections

def _anchor_indices(dataset, n: int) -> list[int]:
    """
    Pick n samples spread evenly over classes, so concept scoring has
    positives for every class. ImageFolder sorts by class, so the first
    n samples would cover only n / samples_per_class classes.
    e.g. n=300, 100 classes -> 3 samples per class
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return list(range(n))   # no per-image class labels, e.g. segmentation

    # Sort by (rank within class, class): first sample of every class,
    # then the second of every class, ...
    rank: dict[int, int] = {}
    order = []
    for idx, t in enumerate(targets):
        r = rank.get(t, 0)
        rank[t] = r + 1
        order.append((r, t, idx))

    order.sort()
    return [idx for _, _, idx in order[:n]]    

def _build_transforms(cfg) -> v2.Compose:
    """
    Build a torchvision v2 pipeline from a transforms YAML config.
    """
    return v2.Compose([instantiate(t) for t in cfg.transforms])

def _build_val_loader(args) -> DataLoader:

    if args.val_transforms:
        from omegaconf import OmegaConf
        cfg    = OmegaConf.load(args.val_transforms)
        val_tf = _build_transforms(cfg)
        print(f"[main] Val transforms loaded from {args.val_transforms}")
    else:
        val_tf = v2.Compose([
            v2.ToImage(),
            v2.Resize((args.image_size, args.image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print("[main] Val transforms: hardcoded resize+normalize fallback")

    args.task = args.subcommand
    if args.num_workers is None:
        args.num_workers = ceil(cpu_count() / 2)

    _, val_loader = build_loaders(args, train_tf=val_tf, val_tf=val_tf)

    if args.anchor_size > 0 and args.anchor_size < len(val_loader.dataset):
        #indices    = list(range(args.anchor_size))
        indices = _anchor_indices(val_loader.dataset, args.anchor_size)
        subset = Subset(val_loader.dataset, indices)
        val_loader = DataLoader(
            subset,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
        )
        print(f"[main] Anchor set limited to {args.anchor_size} samples")

    return val_loader


def build_engine(args) -> InferenceEngine:
    _set_seeds(args.seed)

    # Build model using our factory
    task = args.subcommand
    model = build_model(
        backbone=args.backbone,
        variant=args.variant,
        task=task,
        n_classes=args.num_classes,
    ).to(args.device)
    model.eval()

    _load_checkpoint(model, args.checkpoint_path, args.device)

    dataloader = _build_val_loader(args)

    decompositions, projections = _load_decompositions(
        args.decomposers_path, args.projection_path
    )

    ignore_index = getattr(args, "ignore_index", -100)

    return InferenceEngine(
        model=model,
        model_name=args.backbone + "_" + args.variant,
        dataloader=dataloader,
        decompositions=decompositions,
        projections=projections,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
        device=args.device,
        amp=args.amp,
        address=args.address,
        task=task,
        feature_layer=args.feature_layer,
        ignore_index=ignore_index,
        pca_target_dims=getattr(args, "pca_target_dims", -1),
        trajectory_geometry_csv=args.trajectory_geometry_csv,
        variance_explained_csv=args.variance_explained_csv,
        sensitivity_csv=args.sensitivity_csv
    )

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    root = argparse.ArgumentParser(description="Inference engine backend")
    sub  = root.add_subparsers(dest="subcommand", required=True)
    base = _base_parser()

    cls_p = sub.add_parser("classification", parents=[base],
                            conflict_handler="resolve")
    _add_classification_args(cls_p)

    seg_p = sub.add_parser("segmentation", parents=[base],
                            conflict_handler="resolve")
    _add_segmentation_args(seg_p)

    args   = root.parse_args()
    engine = build_engine(args)

    def _shutdown(sig, frame):
        print("\n[main] Shutting down...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not engine.start():
        print("[main] Failed to start engine")
        sys.exit(1)

    try:
        engine.run()
    except Exception as e:
        import traceback
        print(f"[main] Fatal error: {e}")
        traceback.print_exc()
        engine.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
