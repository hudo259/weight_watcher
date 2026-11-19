from __future__ import annotations

import math
import shutil
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, SequentialLR, LinearLR
)
from torch.utils.data import DataLoader
from trainer.common.datasets import MixupCutmix

# Keys in the optimizer YAML that are not optimizer constructor args.
# Read by the Trainer for scheduling but must be stripped before instantiate.
_TRAINER_ONLY_KEYS = frozenset({
    "backbone_lr_scale", "warmup_epochs", "min_lr", "scheduler",
})

def build_optimizer(opt_cfg: DictConfig, model: nn.Module):
    from omegaconf import OmegaConf
    from pydoc import locate

    lr    = float(opt_cfg.lr)
    scale = float(opt_cfg.get("backbone_lr_scale", 0.1))

    if (hasattr(model, "get_backbone_params") and
            hasattr(model, "get_decoder_params")):
        params = [
            {"params": list(model.get_backbone_params()),
             "lr": lr * scale, "name": "backbone"},
            {"params": list(model.get_decoder_params()),
             "lr": lr, "name": "decoder"},
        ]
    else:
        params = list(model.parameters())

    opt_dict = {
        k: v for k, v in OmegaConf.to_container(opt_cfg, resolve=True).items()
        if k not in _TRAINER_ONLY_KEYS and k != "_target_"
    }

    target = locate(opt_cfg._target_)
    return target(params=params, **opt_dict)


# Utilities
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.
        self._n   = 0

    def update(self, val: float, n: int = 1):
        self._sum += val * n
        self._n   += n

    @property
    def avg(self) -> float:
        return self._sum / self._n if self._n else 0.


class CheckpointManager:
    def __init__(self, save_dir: str, keep_top_k: int = 3):
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self._best: list[tuple[float, Path]] = []

    def save(self, state: dict, epoch: int, metric: float,
             periodic: bool = False) -> Path:
        if periodic:
            p = self.save_dir / f"epoch_{epoch:04d}.pth"
            torch.save(state, p)
            return p

        p = self.save_dir / f"best_ep{epoch:04d}_{metric:.4f}.pth"
        torch.save(state, p)
        self._best.append((metric, p))
        self._best.sort(key=lambda x: x[0], reverse=True)

        while len(self._best) > self.keep_top_k:
            _, old = self._best.pop()
            if old.exists():
                old.unlink()

        best_link = self.save_dir / "best.pth"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        try:
            best_link.symlink_to(self._best[0][1].name)
        except OSError:
            shutil.copy2(self._best[0][1], best_link)

        return p

    @property
    def best_metric(self) -> float | None:
        return self._best[0][0] if self._best else None
    
def build_metrics(
    task:         str,
    num_classes:  int,
    dataset_name: str,
    ignore_index: int = -100,
    device:       str = "cpu",
) -> "torchmetrics.MetricCollection":
    """
    Build the standard TorchMetrics collection for a task/dataset combination.

    Extracted from Trainer._build_metrics so the Evaluator can use identical
    metrics, ensuring consistent numbers across training and inference.

    Args:
        task:         "classification" or "segmentation"
        num_classes:  Number of output classes
        dataset_name: Dataset name (lowercase), used to select ISIC-specific metrics
        ignore_index: Label value to ignore in segmentation metrics
        device:       Move metrics to this device after construction
    """
    n   = num_classes
    ign = ignore_index

    if task == "classification":
        m = {
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=n, top_k=1),
            "top5": torchmetrics.Accuracy(
                task="multiclass", num_classes=n, top_k=min(5, n)),
        }
        if "isic" in dataset_name:
            m["macro_f1"] = torchmetrics.F1Score(
                task="multiclass", num_classes=n, average="macro")
    else:
        m = {
            "miou": torchmetrics.JaccardIndex(
                task="multiclass", num_classes=n,
                ignore_index=ign, average="macro"),
            "pixel_acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=n,
                ignore_index=ign, multidim_average="global"),
        }

    return torchmetrics.MetricCollection(m).to(device)

def unpack_batch(
    batch,
    task:   str,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Unpack a batch from either a tuple-based or dict-based dataset.

    Tuple  (image, target) (torchvision ImageFolder, VOCSegmentation,
                                        ISICSegDataset, ISICClassifyDataset)
    Dict   {"image": ..., "label"/"masks": ...} (inference_components datasets)

    Returns (images, targets) both on device, targets as long.
    """
    if isinstance(batch, (tuple, list)):
        images, targets = batch[0], batch[1]
    else:
        images  = batch["image"]
        targets = batch["label"] if task == "classification" else batch["masks"]
    return images.to(device), targets.to(device).long()

class Trainer:
    def __init__(
        self,
        model:        nn.Module,
        optimizer,
        opt_cfg:      DictConfig,
        loss_fn:      nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        args:         Namespace,
    ):
        self.args         = args
        self.task         = args.task
        self.dataset_name = args.dataset.lower()

        self.device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )
        self.model   = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) \
                       if isinstance(loss_fn, nn.Module) else loss_fn

        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self._num_classes  = args.num_classes
        self._ignore_index = args.ignore_index
        self._accum        = args.grad_accum
        self._steps_per_epoch = math.ceil(len(train_loader) / self._accum)

        self.scaler = torch.GradScaler(
            device=self.device.type,
            enabled=(args.amp and self.device.type == "cuda"),
        )

        self.scheduler, self._warmup_steps = self._build_schedulers(opt_cfg)
        self.val_metrics = self._build_metrics().to(self.device)

        # After building metrics, add:
        self.mixup_cutmix = None
        if self.task == "classification":
            mixup_alpha  = getattr(args, "mixup",  0.0)
            cutmix_alpha = getattr(args, "cutmix", 0.0)
            if mixup_alpha > 0 or cutmix_alpha > 0:
                self.mixup_cutmix = MixupCutmix(
                    mixup_alpha  = mixup_alpha,
                    cutmix_alpha = cutmix_alpha,
                    num_classes  = self._num_classes,
                )
                print(f"[Augmentation] Mixup α={mixup_alpha}  CutMix α={cutmix_alpha}")

        self.ckpt = CheckpointManager(args.checkpoint_dir,
                                      keep_top_k=args.keep_top_k)

        self.start_epoch = 0
        self.global_step = 0

        # Log to checkpoint_dir/tb_logs
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = str(Path(args.checkpoint_dir) / "tb_logs")
        self.writer = SummaryWriter(log_dir=tb_dir)
        print(f"[TensorBoard] {tb_dir}")

        # Log hyperparameters to TensorBoard
        flat_hparams = {k: str(v) for k, v in vars(args).items()}
        self.writer.add_hparams(flat_hparams, metric_dict={})

        if args.resume:
            self._load_checkpoint(args.resume)

    def _build_metrics(self) -> torchmetrics.MetricCollection:
        return build_metrics(
            task=self.task,
            num_classes=self._num_classes,
            dataset_name=self.dataset_name,
            ignore_index=self._ignore_index,
            device=str(self.device),
        )

    def fit(self):
        args = self.args
        eff  = args.batch_size * self._accum
        print(f"\n{'='*60}")
        print(f"  task      : {self.task}")
        print(f"  dataset   : {self.dataset_name}")
        print(f"  device    : {self.device}")
        print(f"  epochs    : {args.epochs}")
        print(f"  batch     : {args.batch_size} × {self._accum} = {eff} eff.")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, args.epochs):
            t0      = time.time()
            train_m = self._train_epoch(epoch)
            val_m   = {}

            if (epoch + 1) % args.eval_every == 0:
                val_m = self._val_epoch()

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_m, val_m, elapsed)

            monitor = val_m.get("val/accuracy",
                       val_m.get("val/miou",
                       val_m.get("val/dice", -1.0))) if val_m else -1.0
            
            self._scheduler_step(epoch, monitor if val_m else None)

            state = self._state(epoch, monitor)
            if val_m:
                self.ckpt.save(state, epoch, monitor)
            if (epoch + 1) % args.save_every == 0:
                self.ckpt.save(state, epoch, monitor, periodic=True)

        print("\nTraining complete.")
        if self.ckpt.best_metric is not None:
            print(f"Best val metric: {self.ckpt.best_metric:.4f}")
        self.writer.close()

    # Train loop
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_m = AverageMeter()
        acc_m  = AverageMeter()

        self.optimizer.zero_grad()

        for idx, batch in enumerate(self.train_loader):
            images, targets = self._unpack(batch)
            is_last = idx == len(self.train_loader) - 1

            if self.mixup_cutmix:
                images, targets = self.mixup_cutmix(images, targets)

            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss   = self.loss_fn(logits, targets) / self._accum

            self.scaler.scale(loss).backward()
            loss_m.update(loss.item() * self._accum, images.size(0))

            if (idx + 1) % self._accum == 0 or is_last:
                if self.args.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                if isinstance(self.scheduler, (SequentialLR, OneCycleLR)):
                    self.scheduler.step()

                with torch.no_grad():
                    hard_targets = targets.argmax(1) if targets.dim() == 2 else targets
                    preds = logits.argmax(1)
                    if self.task == "segmentation":
                        valid = targets != self._ignore_index
                        acc_m.update(
                            ((preds == hard_targets) & valid).sum().item()
                            / valid.sum().clamp(1).item(),
                            images.size(0),
                        )
                    else:
                        acc_m.update(
                            (preds == hard_targets).float().mean().item(),
                            images.size(0),
                        )

                    lr = self.optimizer.param_groups[0]["lr"]
                    mk = "acc" if self.task == "classification" else "px_acc"
                    epoch_step = self.global_step - (epoch * self._steps_per_epoch)
                    print(f"  [ep {epoch+1} step {epoch_step}/{self._steps_per_epoch}] "
                        f"loss={loss_m.avg:.4f}  {mk}={acc_m.avg:.4f}  "
                        f"lr={lr:.2e}"
                    )
                    self.writer.add_scalar("train/loss", loss_m.avg, self.global_step)
                    self.writer.add_scalar(f"train/{mk}", acc_m.avg, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        mk = "accuracy" if self.task == "classification" else "pixel_acc"
        return {"train/loss": loss_m.avg, f"train/{mk}": acc_m.avg}

    # Val loop
    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()
        loss_m = AverageMeter()

        for batch in self.val_loader:
            images, targets = self._unpack(batch)
            with torch.autocast(device_type=self.device.type,
                                    enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss   = self.loss_fn(logits, targets)

            loss_m.update(loss.item(), images.size(0))
            self.val_metrics.update(logits, targets)

        computed = {f"val/{k}": v.item()
                    for k, v in self.val_metrics.compute().items()}
        computed["val/loss"] = loss_m.avg
        return computed

    # Helpers

    def _unpack(self, batch):
        return unpack_batch(batch, self.task, self.device)

    def _build_schedulers(self, opt_cfg):
        warmup_epochs = int(opt_cfg.get("warmup_epochs", 5))
        min_lr        = float(opt_cfg.get("min_lr", 1e-6))
        warmup_steps  = warmup_epochs * self._steps_per_epoch
        total_steps   = self.args.epochs * self._steps_per_epoch
        cosine_steps  = max(total_steps - warmup_steps, 1)

        sched_name = opt_cfg.get("scheduler", "cosine").lower()

        if sched_name == "cosine":
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=cosine_steps, eta_min=min_lr,
            )
            if warmup_steps > 0:
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                sched = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                sched = cosine
            return sched, warmup_steps

        elif sched_name == "onecycle":
            #OneCycleLR owns its own warmup (linear up, cosine down). So no separate warmup scheduler is needed.
            #max_lr defaults to the optimizer's base LR. Override via YAML with 'onecycle_max_lr'.
            lr = float(opt_cfg.lr)
            sched = OneCycleLR(
                self.optimizer,
                max_lr=float(opt_cfg.get("onecycle_max_lr", lr)),
                total_steps=total_steps,
                pct_start=float(opt_cfg.get("onecycle_pct_start", 0.3)),
                div_factor=float(opt_cfg.get("onecycle_div_factor", 25.0)),
                final_div_factor=float(opt_cfg.get("onecycle_final_div_factor", 1e4)),
                anneal_strategy=str(opt_cfg.get("onecycle_anneal_strategy", "cos")),
            )
            return sched, 0

        elif sched_name == "none":
            return None, 0

        else:
            raise ValueError(f"Unknown scheduler: {sched_name!r}")

    def _scheduler_step(self, epoch: int, monitor: float | None):
        if self.scheduler is None or isinstance(
                self.scheduler, (OneCycleLR, SequentialLR)): 
            return
        else:
            self.scheduler.step()

    def _log_epoch(self, epoch, train_m, val_m, elapsed):
        parts  = [f"Epoch {epoch+1:4d}/{self.args.epochs}  ({elapsed:.1f}s)"]
        parts += [f"{k}={v:.4f}" for k, v in train_m.items()]
        parts += [f"{k}={v:.4f}" for k, v in val_m.items()]
        print("  " + "  ".join(parts))
        for k, v in {**train_m, **val_m}.items():
            self.writer.add_scalar(k, v, epoch + 1)
        for pg in self.optimizer.param_groups:
            if "name" in pg:
                self.writer.add_scalar(f"lr/{pg['name']}", pg["lr"], epoch + 1)

    def _state(self, epoch: int, metric: float) -> dict:
        return {
            "epoch":           epoch + 1,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "best_metric":     metric,
        }

    def _load_checkpoint(self, path: str):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = ckpt.get("epoch", 0)
        print(f"  -> epoch {self.start_epoch}, "
              f"best={ckpt.get('best_metric', 0):.4f}")
