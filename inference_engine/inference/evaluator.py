"""
Runs eval passes over the dataloader and builds result payloads for the renderer. 
Supports classification and segmentation tasks.

Deliberately simple forward pass: model(batch["image"].to(device)).
Does not use dataset.applicator(), model.input_device, or time_step.
Those were diffusion-model conventions not applicable to our architectures.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ipc.iipc import ImageBatchMsg, ImageBatchType, ImageFormat, ImageMsg
from trainer.common.trainer import build_metrics, unpack_batch

# ImageNet normalization parameters, used to denormalize tensors for display.
# Training datasets apply these; we must invert before sending to the renderer.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _to_uint8_rgb(images: torch.Tensor) -> np.ndarray:
    """
    Convert a [B, C, H, W] float tensor to [B, H, W, C] uint8 numpy array.

    Handles two cases:
      - ImageNet-normalized tensors (roughly in [-2.1, 2.6]), denormalized first.
      - [0, 1] range tensors, scaled directly.

    Heuristic: if any value is outside [-1.5, 1.5] we assume ImageNet
    normalization was applied and denormalize. This covers all practical
    training pipelines without requiring an explicit flag.
    """
    imgs = images.float().cpu()
    if imgs.min() < -1.5 or imgs.max() > 1.5:
        # Denormalize from ImageNet stats
        imgs = imgs * _IMAGENET_STD + _IMAGENET_MEAN
    imgs = imgs.clamp(0.0, 1.0)
    imgs = (imgs * 255).byte()              # [B, C, H, W] uint8
    return imgs.permute(0, 2, 3, 1).numpy()  # [B, H, W, C]


@dataclass
class EvalResult:
    """All outputs from a single evaluation pass."""
    loss:     float
    metrics:  dict[str, float]
    features: torch.Tensor    # [N, D] pooled activations or logits
    images:   ImageBatchMsg   # PREDICTIONS batch
    labels:   torch.Tensor    # [N] one integer label per sample


class Evaluator:
    """Runs eval passes and builds result payloads."""

    def __init__(
        self,
        model:             nn.Module,
        dataloader:        DataLoader,
        task:              Literal["classification", "segmentation"],
        dataset_name:      str,
        num_classes:       int,
        loss_fn:           nn.Module | None = None,
        amp:               bool = False,
        ignore_index:      int = -100,
        feature_layer:     str | None = None,
        feature_buffer_fn: Callable[[], torch.Tensor | None] | None = None,
    ):
        self._model              = model
        self._dataloader         = dataloader
        self._task               = task
        self._loss_fn            = loss_fn
        self._amp                = amp
        self._feature_layer      = feature_layer
        self._get_feature_buffer = feature_buffer_fn or (lambda: None)
        self._num_classes        = num_classes

        # Use the same TorchMetrics as the Trainer for consistency
        device = self._device()
        self._metrics = build_metrics(
            task=task,
            num_classes=num_classes,
            dataset_name=dataset_name,
            ignore_index=ignore_index,
            device=str(device),
        )

        self._raw_image_batch: ImageBatchMsg | None = None

        self._annotate = (
            self._annotate_segmentations
            if task == "segmentation"
            else self._annotate_classifications
        )

    @property
    def num_classes(self):
        return self._num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self._device().type, enabled=self._amp):
            return self._model(images)

    # main eval loop
    def run(self) -> EvalResult:
        """Run a full evaluation pass and return all results."""
        device = self._device()

        self._metrics.reset()
        loss_sum     = 0.0
        n_samples    = 0
        all_features: list[torch.Tensor] = []
        all_labels:   list[torch.Tensor] = []
        all_images:   list[ImageMsg]     = []
        index_offset  = 0

        with torch.no_grad():
            for batch in self._dataloader:
                images, targets = unpack_batch(batch, self._task, device)
                with torch.autocast(device_type=device.type, enabled=self._amp):
                    predictions = self._model(images)
                B = images.size(0)

                blended = self._annotate(
                    images.cpu(), predictions, index_offset=index_offset
                )
                all_images.extend(blended)
                index_offset += B

                all_features.append(self._pool_features(predictions))
                all_labels.append(self._extract_labels(targets))

                if self._loss_fn is not None:
                    loss_sum += self._loss_fn(predictions, targets).item() * B
                n_samples += B

                self._metrics.update(predictions, targets)

        avg_loss    = loss_sum / max(n_samples, 1)
        avg_metrics = {k: v.item() for k, v in self._metrics.compute().items()}
        features    = torch.cat(all_features, dim=0).float()
        labels      = torch.cat(all_labels,   dim=0)

        print(f"[Evaluator] loss={avg_loss:.4f}  metrics={avg_metrics}")

        return EvalResult(
            loss=avg_loss,
            metrics=avg_metrics,
            features=features,
            images=ImageBatchMsg(
                batch_type=ImageBatchType.PREDICTIONS, images=all_images
            ),
            labels=labels,
        )

    def build_raw_images(self) -> ImageBatchMsg:
        """
        Build (and cache) the raw anchor image batch.

        Handles both tuple and dict batch formats, and denormalizes
        ImageNet-normalized tensors before display (see _to_uint8_rgb).
        """
        if self._raw_image_batch is not None:
            print("[Evaluator] Resending cached raw images")
            return self._raw_image_batch

        images = []
        idx    = 0
        for batch in self._dataloader:
            # Extract image tensor regardless of batch format
            if isinstance(batch, (list, tuple)):
                img_tensor = batch[0]
            else:
                img_tensor = batch["image"]

            imgs = _to_uint8_rgb(img_tensor)   # [B, H, W, C] uint8

            for img in imgs:
                images.append(ImageMsg(
                    image_id=idx, image=img,
                    format=ImageFormat.RGB, name=f"raw_{idx}",
                ))
                idx += 1

        self._raw_image_batch = ImageBatchMsg(
            batch_type=ImageBatchType.RAW_IMAGES, images=images
        )
        print(f"[Evaluator] Built {idx} raw anchor images")
        return self._raw_image_batch

    def get_input_resolution(self) -> tuple[int, int]:
        """Return (height, width) of anchor images."""
        if self._raw_image_batch and self._raw_image_batch.images:
            img = self._raw_image_batch.images[0].image
            return img.shape[0], img.shape[1]
        for batch in self._dataloader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
            if imgs.dim() == 4:
                return imgs.shape[2], imgs.shape[3]
            break
        return 512, 512

    def _device(self) -> torch.device:
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # feature pooling
    def _pool_features(self, predictions: torch.Tensor) -> torch.Tensor:
        """Pool spatial features or return logits for the feature viewer."""
        feat = self._get_feature_buffer() if self._feature_layer else predictions
        if feat is None:
            feat = predictions
        if feat.dim() == 4:
            feat = feat.mean(dim=(-2, -1))   # [B, C, H, W] -> [B, C]
        return feat.detach().cpu()

    # label extraction
    def _extract_labels(self, targets: torch.Tensor) -> torch.Tensor:
        if self._task == "classification":
            return targets.cpu().long()
        else:
            # Mode of the flattened mask = dominant class per image
            return targets.cpu().long().flatten(1).mode(dim=1).values

    # image annotation

    def _annotate_segmentations(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        index_offset: int = 0,
    ) -> list[ImageMsg]:
        class_map = predictions.argmax(dim=1).cpu().numpy().astype(np.uint8)
        return [
            ImageMsg(
                image_id=index_offset + i,
                image=class_map[i],
                format=ImageFormat.GRAYSCALE,
                name=f"sample_{index_offset + i}",
            )
            for i in range(len(class_map))
        ]

    def _annotate_classifications(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        index_offset: int = 0,
    ) -> list[ImageMsg]:
        probs     = predictions.softmax(dim=1).detach().cpu().numpy()
        class_ids = probs.argmax(axis=1)
        imgs_rgb  = _to_uint8_rgb(images)   # [B, H, W, 3] uint8, properly denormalized

        results = []
        for i in range(len(imgs_rgb)):
            img  = imgs_rgb[i].copy()
            cls  = class_ids[i]
            conf = probs[i, cls]
            H, W = img.shape[:2]

            label     = f"cls {cls}  {conf:.2f}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            scale     = max(0.4, W / 500.0)
            thickness = max(1, int(scale * 1.5))
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

            # Text bottom-left, with a dark background bar
            x  = 8
            y  = H - baseline - 6
            cv2.rectangle(img,
                        (x - 2, y - th - baseline - 2),
                        (x + tw + 2, H),
                        (0, 0, 0), cv2.FILLED)
            cv2.putText(img, label, (x, y), font, scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

            results.append(ImageMsg(
                image_id=index_offset + i,
                image=img,
                format=ImageFormat.RGB,
                name=f"sample_{index_offset + i}",
            ))
        return results
