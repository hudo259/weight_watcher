"""
Computes spatial activation maps for a set of selected filters and returns
an ImageBatchMsg(ACTIVATION_MAP) ready to send to the renderer.

Uses HookManager.temp_spatial_hooks() so hooks are always cleaned up.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ipc.iipc import ImageBatchMsg, ImageBatchType, ImageFormat, ImageMsg
from projections.common.layer_decomposition import LayerDecomposition
from inference.hook_manager import HookManager


# Layer types that produce spatially meaningful activations
_SPATIAL_LAYER_TYPES = frozenset({
    "conv2d", "depthwise_conv",
    "attention_query", "attention_key", "attention_value",
    "attention_output", "attention_fused_qkv",
    "attention_mha_head", "attention_mha_head_fused_qkv",
})
# conv1d outputs [B, C, L]: 1-D, not directly overlayable; excluded for now


class ActivationMapper:
    """Produces per-image spatial activation maps for selected filters."""

    def __init__(self, hook_manager: HookManager):
        self._hooks = hook_manager

    def compute(
        self,
        model:            nn.Module,
        dataloader:       DataLoader,
        forward_fn, # callable(images), output ignored, hooks capture activations
        by_layer:         dict[int, list[int]],
        layer_keys:       list[str],
        pca_dict:         dict[str, LayerDecomposition],
        aggregation_mode: int,             # 0 = average, 1 = maximum
        input_h:          int,
        input_w:          int,
    ) -> ImageBatchMsg | None:
        """
        Parameters
        ----------
        by_layer         : {cloud_idx: [filter_indices]}
        aggregation_mode : 0 = average across filters/layers, 1 = maximum
        input_h, input_w : target resolution for upsampled maps

        Returns
        -------
        ImageBatchMsg(ACTIVATION_MAP) or None if no valid maps were produced.
        """
        by_layer = self._filter_spatial(by_layer, layer_keys, pca_dict)
        if not by_layer:
            print("[ActivationMapper] All selected layers are linear, no spatial maps available")
            return None

        raw_maps = self._collect_activations(
            model, dataloader, forward_fn, by_layer, layer_keys
        )
        layer_maps = self._aggregate_and_upsample(
            raw_maps, by_layer, layer_keys, pca_dict,
            aggregation_mode, input_h, input_w
        )
        if not layer_maps:
            print("[ActivationMapper] No valid spatial maps produced")
            return None

        encoded = self._combine_and_encode(layer_maps, aggregation_mode)
        return self._build_batch(encoded)

    # Pipeline stages

    @staticmethod
    def _filter_spatial(
        by_layer:   dict[int, list[int]],
        layer_keys: list[str],
        pca_dict:   dict[str, LayerDecomposition],
    ) -> dict[int, list[int]]:
        return {
            ci: fi for ci, fi in by_layer.items()
            if pca_dict[layer_keys[ci]].layer_type in _SPATIAL_LAYER_TYPES
        }

    def _collect_activations(
        self,
        model:       nn.Module,
        dataloader:  DataLoader,
        forward_fn,
        by_layer:    dict[int, list[int]],
        layer_keys:  list[str],
    ) -> dict[int, torch.Tensor]:
        """
        Run one forward pass with temp hooks and return per-cloud activation
        tensors: {cloud_idx -> [N_total, C, H, W] or [N_total, T, D]}.
        """
        with self._hooks.temp_spatial_hooks(model, by_layer, layer_keys) as buffers:
            with torch.no_grad():
                for batch in dataloader:
                    images = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
                    forward_fn(images.to(next(model.parameters()).device))

        # Concatenate per-cloud lists
        return {
            ci: torch.cat(chunks, dim=0)
            for ci, chunks in buffers.items()
            if chunks
        }

    @staticmethod
    def _aggregate_and_upsample(
        raw_maps:         dict[int, torch.Tensor],
        by_layer:         dict[int, list[int]],
        layer_keys:       list[str],
        pca_dict:         dict[str, LayerDecomposition],
        aggregation_mode: int,
        input_h:          int,
        input_w:          int,
    ) -> list[np.ndarray]:
        """
        For each layer:
          1. Select the requested filter indices.
          2. Aggregate along the filter dim (avg or max) -> [N, H_layer, W_layer].
          3. Upsample to (input_h, input_w).

        Returns a list of [N, input_h, input_w] float32 arrays — one per layer.
        """
        layer_maps: list[np.ndarray] = []

        for cloud_idx, filter_indices in by_layer.items():
            layer_key  = layer_keys[cloud_idx]
            layer_type = pca_dict[layer_key].layer_type

            if cloud_idx not in raw_maps:
                # Hook registration failed for this module (HookManager printed
                # the reason above). No activation data to aggregate.
                print(f"[ActivationMapper] '{layer_key}': no activation data "
                    f"collected (hook registration failed)! Skipping...")
                continue
            acts = raw_maps[cloud_idx]   # [N, C, H, W]  or  [N, T, D]

            if acts.dim() == 4:
                selected = acts[:, filter_indices, :, :]   # [N, F, H, W]
                fmap = (selected.mean(dim=1) if aggregation_mode == 0
                        else selected.max(dim=1).values)   # [N, H, W]
            elif acts.dim() == 3:
                # MiT layers use attention_query / attention_output types but no CLS token
                HAS_CLS_TOKEN = frozenset()   # NOTE add ViT types here when supported

                N, T, D = acts.shape
                if layer_type in HAS_CLS_TOKEN:
                    tokens  = acts[:, 1:, :] # drop CLS token
                    T_sq    = T - 1
                else:
                    tokens = acts
                    T_sq = T
                H_p     = int(T_sq ** 0.5)
                W_p     = T_sq // H_p
                if H_p * W_p != T_sq:
                    print(f"[ActivationMapper] {layer_key}: {T_sq} tokens can't "
                          f"be reshaped to a rectangle! Skipping...")
                    continue
                selected = tokens[:, :, filter_indices]     # [N, T-1, F]
                agg = (selected.mean(dim=2) if aggregation_mode == 0
                       else selected.max(dim=2).values)     # [N, T-1]
                fmap = agg.reshape(N, H_p, W_p)

            else:
                continue

            fmap_np = fmap.numpy().astype(np.float32)

            # Upsample each sample to input resolution
            upsampled = np.stack([
                cv2.resize(fmap_np[i], (input_w, input_h),
                           interpolation=cv2.INTER_LINEAR)
                for i in range(len(fmap_np))
            ])  # [N, input_h, input_w]

            layer_maps.append(upsampled)

        return layer_maps

    @staticmethod
    def _combine_and_encode(
        layer_maps:       list[np.ndarray],
        aggregation_mode: int,
    ) -> np.ndarray:
        """
        Combine [L, N, H, W] maps across layers, normalise globally to [0,1],
        and encode to uint8.

        Encoding contract with HeatmapImageMode (image_modes.cpp):
        The fragment shader reads the R8 texture as float r in [0,1], then
        remaps to [-1,1] via:  heat = r * 2.0 - 1.0
        and applies the hardcoded diverging colormap:
            heat = -1  (uint8=0): blue (minimum activation)
            heat =  0  (uint8=127): white (mid-range)
            heat = +1  (uint8=255): red (maximum activation)

        NOTE: this is separate from the user-selectable colormaps (Viridis,
        Inferno, Coolwarm) used for concept scores on the point cloud, those
        go through a different path entirely (the heatmap VBO in scene.cpp).
        """
        stacked = np.stack(layer_maps, axis=0)   # [L, N, H, W]
        combined = (stacked.mean(axis=0) if aggregation_mode == 0
                    else stacked.max(axis=0))     # [N, H, W]

        combined = combined.astype(np.float32)
        abs_max = np.abs(combined).max()
        if abs_max > 1e-8:
            combined = combined / abs_max # maps to [-1, 1], zero stays at zero
        # Encode [-1, 1] -> [0, 255] to match the shader's r * 2.0 - 1.0 decode
        return ((combined + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _build_batch(encoded: np.ndarray) -> ImageBatchMsg:
        images = [
            ImageMsg(
                image_id=i,
                image=encoded[i],
                format=ImageFormat.GRAYSCALE,
                name=f"act_s{i}",
            )
            for i in range(len(encoded))
        ]
        return ImageBatchMsg(batch_type=ImageBatchType.ACTIVATION_MAP, images=images)
