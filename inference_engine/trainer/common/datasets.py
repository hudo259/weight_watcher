from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets as tvd
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

"""
Minimal custom datasets for ISIC data that is not in torchvision.

ISICSegDataset: ISIC 2018 Task 1, binary lesion segmentation.
ISICClassifyDataset: ISIC 2019, 8-class skin lesion classification.
                     Thin wrapper around ImageFolder (expects folder-per-class layout).

For ISIC 2019, reorganise the downloaded data into ImageFolder structure once:
    isic2019/
    ├── train/
    │   ├── MEL/  ├── NV/  ├── BCC/  ├── AK/
    │   ├── BKL/  ├── DF/  ├── VASC/ └── SCC/
    └── val/
        └── (same structure)

For ISIC 2018 Task 1:
    isic2018/
    ├── train/
    │   ├── images/   ISIC_*.jpg
    │   └── masks/    ISIC_*_segmentation.png
    └── val/
        ├── images/
        └── masks/

Class weights for ISIC 2019 (pre-computed, approximate for B2 split):
    [3.8, 0.3, 2.5, 3.5, 1.5, 6.0, 7.0, 5.0]  (NV dominates at roughly 67%)
    These are passed to CrossEntropyLoss(weight=...) in the trainer.
"""
# ISIC 2018: binary segmentation
class ISICSegDataset(Dataset):
    """
    ISIC 2018 Task 1, binary lesion segmentation.

    Returns (image_tensor, mask_tensor) where mask is 0/1 long tensor.

    The `transforms` argument should be a _SegTransformAdapter wrapping an
    albumentations Compose pipeline (wired up automatically by build_loaders).
    It receives (PIL image, PIL mask) and returns (CHW float tensor, HW long tensor).

    Binarisation (0/255 -> 0/1) is applied here after the transform so it is
    always done regardless of whether a transform is present.
    """
    def __init__(self, root: str, split: str = "train", transforms=None):
        self.transforms = transforms
        img_dir  = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"

        self.images = sorted(img_dir.glob("*.jpg")) + \
                      sorted(img_dir.glob("*.png"))
        self.masks  = sorted(mask_dir.glob("*.png"))

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Image/mask count mismatch in {root}/{split}: "
                f"{len(self.images)} images vs {len(self.masks)} masks"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask  = Image.open(self.masks[idx]).convert("L")

        if self.transforms is not None:
            # _SegTransformAdapter handles PIL -> (float tensor, long tensor)
            image, mask = self.transforms(image, mask)
        else:
            # Bare fallback: no augmentation, just convert to tensors
            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)
            mask  = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Binarise: ISIC masks are stored as 0/255, convert to class 0/1
        mask = (mask > 0).long()

        return image, mask


# ISIC 2019: classification
class ISICClassifyDataset(Dataset):
    """
    ISIC 2019 classification, thin wrapper around ImageFolder.
    Expects folder-per-class layout (see module docstring).
    Returns (image_tensor, label_int) tuples.
    """
    def __init__(self, root: str, split: str = "train", transform=None):
        self._ds = ImageFolder(
            root=os.path.join(root, split),
            transform=transform,
        )
        self.classes     = self._ds.classes
        self.class_to_idx = self._ds.class_to_idx

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]   # (tensor, int)

    def compute_class_weights(self) -> torch.Tensor:
        # we do inverse frequency balancing here
        labels  = [label for _, label in self._ds.samples]
        counts  = np.bincount(labels, minlength=len(self.classes)).astype(float)

        N = counts.sum()
        K = len(self.classes)
        weights = N / (counts * K) # w_c = N / (N_c * K)

        return torch.tensor(weights, dtype=torch.float32)

class _SegTransformAdapter:
    """
    Adapts a torchvision v2.Compose pipeline for paired (PIL image, PIL mask) inputs.
 
    Both VOCSegmentation and ISICSegDataset call transforms(PIL_image, PIL_mask).
    torchvision v2 transforms accept multiple inputs when the second is wrapped
    as tv_tensors.Mask:
      - Spatial ops (RandomResizedCrop, RandomHorizontalFlip, Resize, ...) apply
        to both, using NEAREST interpolation for the mask automatically.
      - Colour / normalize ops only apply to the image.
 
    Classification datasets (ImageFolder, ISICClassifyDataset) pass a single PIL
    image to transform(), which v2.Compose handles natively, no adapter needed.
    """
    def __init__(self, transform):
        self.transform = transform
 
    def __call__(self, image: Image.Image, mask: Image.Image):
        # Wrap mask as tv_tensors.Mask so v2 spatial transforms track it
        mask_t = tv_tensors.Mask(
            torch.as_tensor(np.array(mask, dtype=np.int64))  # (H, W)
        )
 
        # image is PIL; v2.ToImage() inside the pipeline converts it to a
        # uint8 tv_tensors.Image. Passing both together lets v2 apply the
        # same random parameters to image and mask in lock-step.
        image_out, mask_out = self.transform(image, mask_t)
 
        # Strip the tv_tensors wrapper; squeeze channel dim if present
        mask_out = torch.as_tensor(mask_out)
        if mask_out.ndim == 3:
            mask_out = mask_out.squeeze(0)
 
        return image_out, mask_out.long()
 


# Main entry point
def build_loaders(args, train_tf, val_tf):
    """
    Args:
        args:     Parsed argparse namespace. Must have:
                    args.dataset, args.data_root, args.batch_size,
                    args.num_workers, args.task
        train_tf: Instantiated torchvision v2.Compose pipeline.
        val_tf:   Same for validation.
 
    Returns:
        (train_loader, val_loader)
    """
    name = args.dataset.lower()
 
    if args.task == "segmentation":
        # Wrap: VOCSegmentation and ISICSegDataset pass (PIL, PIL) to transforms.
        # The adapter converts the mask to tv_tensors.Mask so v2 spatial ops
        # are applied consistently to both image and mask.
        train_tf = _SegTransformAdapter(train_tf)
        val_tf   = _SegTransformAdapter(val_tf)
    # Classification: v2.Compose accepts a single PIL image directly.
    # ImageFolder and ISICClassifyDataset already call transform(PIL), so no
    # adapter is needed.
 
    if name == "imagenet100":
        return _imagenet100(args, train_tf, val_tf)
    elif name == "isic2019":
        return _isic2019(args, train_tf, val_tf)
    elif name == "voc2012":
        return _voc2012(args, train_tf, val_tf)
    elif name == "voc2012_sbd":
        return _voc2012_sbd(args, train_tf, val_tf)
    elif name == "isic2018":
        return _isic2018(args, train_tf, val_tf)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


# Per-dataset constructors


def _imagenet100(args, train_tf, val_tf):
    train_ds = tvd.ImageFolder(f"{args.data_root}/train", transform=train_tf)
    val_ds   = tvd.ImageFolder(f"{args.data_root}/val",   transform=val_tf)
    return _make_loaders(train_ds, val_ds, args)
 
 
def _isic2019(args, train_tf, val_tf):
    train_ds = ISICClassifyDataset(args.data_root, split="train", transform=train_tf)
    val_ds   = ISICClassifyDataset(args.data_root, split="val",   transform=val_tf)
    #train_ds.class_weights = train_ds.compute_class_weights()
    return _make_loaders(train_ds, val_ds, args)
 
 
def _voc2012(args, train_tf, val_tf):
    train_ds = tvd.VOCSegmentation(
        root=args.data_root, year="2012", image_set="train",
        download=False, transforms=train_tf,
    )
    val_ds = tvd.VOCSegmentation(
        root=args.data_root, year="2012", image_set="val",
        download=False, transforms=val_tf,
    )
    return _make_loaders(train_ds, val_ds, args)

def _voc2012_sbd(args, train_tf, val_tf):
    """
    Augmented VOC 2012 training set: SBD train_noval (8,498 images) combined
    with standard VOC 2012 train (1,464 images) = ~10,582 total.

    Expects args.data_root to be the parent of both VOCdevkit/ and
    benchmark_RELEASE/, e.g.:

        data_root/
        ├── VOCdevkit/
        │   └── VOC2012/
        └── benchmark_RELEASE/
            └── dataset/

    Val set is always standard VOC 2012 val. No data leakage, SBD
    train_noval explicitly excludes VOC 2012 val images.
    """
    from torchvision.datasets import SBDataset
    from torch.utils.data import ConcatDataset

    sbd_root = os.path.join(args.data_root, "benchmark_RELEASE", "dataset")

    sbd_ds = SBDataset(
        root=sbd_root,
        image_set="train_noval",   # excludes VOC 2012 val, no leakage
        mode="segmentation",        # default is 'boundaries', must override
        download=False,
        transforms=train_tf,
    )
    voc_train_ds = tvd.VOCSegmentation(
        root=args.data_root, year="2012", image_set="train",
        download=False, transforms=train_tf,
    )
    train_ds = ConcatDataset([sbd_ds, voc_train_ds])
    print(f"  VOC 2012 augmented: {len(sbd_ds)} SBD + "
          f"{len(voc_train_ds)} VOC train = {len(train_ds)} total")

    val_ds = tvd.VOCSegmentation(
        root=args.data_root, year="2012", image_set="val",
        download=False, transforms=val_tf,
    )
    return _make_loaders(train_ds, val_ds, args)

 
def _isic2018(args, train_tf, val_tf):
    train_ds = ISICSegDataset(args.data_root, split="train", transforms=train_tf)
    val_ds   = ISICSegDataset(args.data_root, split="val",   transforms=val_tf)
    return _make_loaders(train_ds, val_ds, args)
 
 


# Shared loader builder
def _make_loaders(train_ds, val_ds, args):
    nw = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True,
        drop_last=True, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        drop_last=False, persistent_workers=nw > 0,
    )
    return train_loader, val_loader

class MixupCutmix:
    """
    Applies Mixup, CutMix, or neither to a batch of images and labels.
    Both can be active simultaneously, one is chosen randomly per batch.

    Args:
        mixup_alpha:   Beta distribution alpha for Mixup (0 = disabled).
        cutmix_alpha:  Beta distribution alpha for CutMix (0 = disabled).
        switch_prob:   Probability of using CutMix when both are enabled.
        num_classes:   Number of classes, needed for one-hot label encoding.
    """
    def __init__(self, mixup_alpha: float, cutmix_alpha: float,
                 switch_prob: float = 0.5, num_classes: int = 1000):
        try:
            from torchvision.transforms.v2 import MixUp, CutMix
            self._transforms: list = []

            if mixup_alpha > 0:
                self._transforms.append(MixUp(alpha=mixup_alpha,
                                               num_classes=num_classes))
            if cutmix_alpha > 0:
                self._transforms.append(CutMix(alpha=cutmix_alpha,
                                                num_classes=num_classes))
                
            self._switch_prob = switch_prob
            self._enabled = len(self._transforms) > 0

        except ImportError:
            print("[MixupCutmix] torchvision.transforms.v2 not available. "
                  "Mixup/CutMix disabled. Upgrade to torchvision >= 0.15.")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return len(self._transforms) > 0

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images: (B, C, H, W) float tensor
            labels: (B,) long tensor, converted to soft one-hot on return
 
        Returns:
            images: (B, C, H, W), mixed images
            labels: (B, num_classes), soft one-hot labels
        """
        if not self.enabled:
            return images, labels
 
        if len(self._transforms) == 2:
            tf = self._transforms[1] if torch.rand(1).item() < self._switch_prob \
                 else self._transforms[0]
        else:
            tf = self._transforms[0]
 
        return tf(images, labels)
