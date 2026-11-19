#!/usr/bin/env python3
"""
Subcommands: classification | segmentation
 
Each subcommand takes dataset arguments (paths, number of classes, etc.)
via argparse. Component configs (model architecture, optimizer, transforms,
loss function) are passed as paths to YAML files and loaded with OmegaConf.
 
Transforms
----------
YAML configs list torchvision.transforms.v2 ops by _target_. They should
include augmentation ops and a final Normalize, but NOT ToImage or ToDtype.
_build_transforms inserts those automatically:
 
    ToImage()  <- prepended (PIL -> uint8 tv_tensors.Image)
    <all ops from YAML in order>
    ToDtype(torch.float32, scale=True), inserted just before Normalize
    Normalize(mean, std)
 
This means YAML configs never need to care about tensor conversion.
 
RandAugment is listed directly in the YAML as a regular transform op. It must
come before Normalize (i.e. while the image is still uint8), which is the
natural position since it appears before Normalize in any sensible pipeline.
 
Mixup / CutMix are batch-level and controlled via CLI flags --mixup / --cutmix
(classification only).
 
Call like this:
-----
Classification:
    python train.py classification \\
        --data_root        data/imagenet100 \\
        --dataset          imagenet100 \\
        --num_classes      100 \\
        --model_config     configs/model/cyclemlp_b2.yaml \\
        --optimizer_config configs/optimizer/adamw.yaml \\
        --train_transforms configs/transforms/classify_imagenet_train.yaml \\
        --val_transforms   configs/transforms/classify_imagenet_val.yaml \\
        --loss_config      configs/loss/cross_entropy.yaml \\
        --epochs           100 \\
        --batch_size       32 \\
        --mixup            0.8 \\
        --cutmix           1.0
 
Segmentation:
    python train.py segmentation \\
        --data_root        data/voc2012 \\
        --dataset          voc2012 \\
        --num_classes      21 \\
        --ignore_index     255 \\
        --model_config     configs/model/mit_b2.yaml \\
        --optimizer_config configs/optimizer/adamw.yaml \\
        --train_transforms configs/transforms/seg_voc_train.yaml \\
        --val_transforms   configs/transforms/seg_voc_val.yaml \\
        --loss_config      configs/loss/cross_entropy.yaml \\
        --epochs           120 \\
        --batch_size       8 \\
        --grad_accum       4
 
Resume from checkpoint:
    python train.py classification ... --resume checkpoints/best.pth
"""

import argparse
import random
from math import ceil
from multiprocessing import cpu_count
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from pydoc import locate

from hydra.utils import instantiate
from trainer.common.datasets import build_loaders
from models.factory import build_model
from trainer.common.trainer import Trainer
from trainer.common.utils import load_pretrained
import torchvision
from torchvision.transforms import v2
        
# Argument parsers

def _base_parser() -> argparse.ArgumentParser:
    """Arguments shared by both subcommands."""
    p = argparse.ArgumentParser(add_help=False)

    # Dataset
    p.add_argument("--data_root",   required=True,
                   help="Root directory of the dataset")
    p.add_argument("--dataset",     required=True,
                   choices=["imagenet100", "isic2019", "voc2012", "voc2012_sbd", "isic2018"],
                   help="Dataset name")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--num_workers", type=int,
                   default=ceil(cpu_count() / 2))

    # Component configs (paths to YAML files)
    p.add_argument("--model_config",     required=True,
                   help="Path to model YAML config")
    p.add_argument("--optimizer_config", required=True,
                   help="Path to optimizer YAML config")
    p.add_argument("--train_transforms", required=True,
                   help="Path to training transforms YAML config")
    p.add_argument("--val_transforms",   required=True,
                   help="Path to validation transforms YAML config")
    p.add_argument("--loss_config",      required=True,
                   help="Path to loss YAML config")

    # Training loop
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--grad_accum",   type=int,   default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--amp",          action="store_true",
                   help="Enable automatic mixed precision")
    p.add_argument("--clip_grad",    type=float, default=1.0,
                   help="Gradient clipping norm (0 = disabled)")
    p.add_argument("--eval_every",   type=int,   default=1,
                   help="Run validation every N epochs")

    # Checkpointing
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--save_every",     type=int, default=10,
                   help="Save a periodic checkpoint every N epochs")
    p.add_argument("--keep_top_k",     type=int, default=3,
                   help="Keep top-K best checkpoints")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--pretrained", default=None, 
                   help="Path to pretrained checkpoint for fine-tuning."
                        "Loads model weights only, optimizer and scheduler start fresh. "
                        "Mismatched keys are skipped automatically.")

    # Misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed",   type=int, default=42)

    return p

def _add_segmentation_args(p: argparse.ArgumentParser):
    p.add_argument("--ignore_index", type=int, default=255,
                   help="Mask pixel value to ignore in loss and metrics")

    return p

def _add_classification_args(p: argparse.ArgumentParser):
    p.add_argument("--ignore_index", type=int, default=-100)
    p.add_argument("--mixup",  type=float, default=0.0,
               help="Mixup alpha (0 = disabled)")
    p.add_argument("--cutmix", type=float, default=0.0,
               help="CutMix alpha (0 = disabled)")

    return p

# Config loading and component construction

def _load_configs(args) -> dict:
    return {
        "model":            OmegaConf.load(args.model_config),
        "optimizer":        OmegaConf.load(args.optimizer_config),
        "train_transforms": OmegaConf.load(args.train_transforms),
        "val_transforms":   OmegaConf.load(args.val_transforms),
        "loss":             OmegaConf.load(args.loss_config),
    }

def _build_transforms(cfg) -> torchvision.transforms.v2.Compose:
    """
    Build a torchvision v2 pipeline from a transforms YAML config.
    """
    return v2.Compose([instantiate(t) for t in cfg.transforms])
 
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    root = argparse.ArgumentParser(description="Train segmentation/classification models")
    sub  = root.add_subparsers(dest="task", required=True)
    base = _base_parser()

    seg_p = sub.add_parser("segmentation",   parents=[base],
                            conflict_handler="resolve")
    _add_segmentation_args(seg_p)

    cls_p = sub.add_parser("classification", parents=[base],
                            conflict_handler="resolve")
    _add_classification_args(cls_p)

    args = root.parse_args()
    set_seed(args.seed)

    # Load component configs
    OmegaConf.register_new_resolver("locate", locate)
    configs = _load_configs(args)

    # Build transforms
    train_tf = _build_transforms(configs["train_transforms"])
    val_tf   = _build_transforms(configs["val_transforms"])

    # Build model
    model = build_model(
        backbone  = configs["model"].backbone,
        variant   = configs["model"].variant,
        task      = args.task,
        n_classes = args.num_classes,
        fpn_ch    = int(configs["model"].get("fpn_channels", 256)),
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] {configs['model'].backbone}-{configs['model'].variant}, {n_params:.1f}M parameters")
    
    # Load pretrained weights for fine-tuning (if they are given)
    if args.pretrained:
        load_pretrained(model, args.pretrained)

    # Build data loaders
    train_loader, val_loader = build_loaders(args, train_tf, val_tf)
    print(f"[Data]  {args.dataset}  train={len(train_loader.dataset)}"
          f"  val={len(val_loader.dataset)}")

    # Class weights for ISIC 2019
    class_weights = None
    if args.dataset == "isic2019":
        class_weights = train_loader.dataset.compute_class_weights()
        print(f"[Loss]  class weights: "
              f"{[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Build loss from YAML config via instantiate
    # ignore_index and label_smoothing live in the loss YAML.
    # class_weights is computed from dataset after construction.
    configs["loss"].ignore_index = args.ignore_index
    loss_fn = instantiate(configs["loss"])

    if class_weights is not None:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        loss_fn.register_buffer("weight", class_weights.to(device))

    # Build optimizer
    opt_cfg   = configs["optimizer"]
    optimizer = _build_optimizer(opt_cfg, model)

    # Save complete run configuration to checkpoint_dir/run_config.json
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": vars(args),
        "model":            OmegaConf.to_container(configs["model"],            resolve=True),
        "optimizer":        OmegaConf.to_container(configs["optimizer"],        resolve=True),
        "loss":             OmegaConf.to_container(configs["loss"],             resolve=True),
        "train_transforms": OmegaConf.to_container(configs["train_transforms"], resolve=True),
        "val_transforms":   OmegaConf.to_container(configs["val_transforms"],   resolve=True),
    }
    config_path = Path(args.checkpoint_dir) / "run_config.json"

    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=lambda obj: str(obj))
    print(f"[Config] Saved to {config_path}")

    # Train
    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        opt_cfg      = opt_cfg,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        args         = args,
    )
    trainer.fit()

def _build_optimizer(opt_cfg, model):
    from trainer.common.trainer import build_optimizer
    return build_optimizer(opt_cfg, model)

if __name__ == "__main__":
    main()
