"""
Owns all decomposition state and knows how to:
  - apply point updates / full-cloud messages to _current_Z
  - reconstruct weight tensors from decomposition coordinates
  - load them into a model's state dict
  - build the PointCloudMsg and LayerMetaMsg for the renderer

Updated to use LayerDecomposition (supports PCA, UMAP, FourierKPCA)
instead of the old LayerPCA + pca_reconstruct_batch path.
"""

import numpy as np
import torch
import torch.nn as nn

from ipc.iipc import LayerMetaMsg, PointCloudMsg, TransformedPointsMsg
from projections.common.layer_decomposition import LayerDecomposition
from projections.common.layout import LAYER_TYPE_COLORS

# Weight shape restoration
def _reshape_to_original(
    weight_flat: torch.Tensor,
    decomp: LayerDecomposition,
) -> torch.Tensor:
    """
    Reshape a flat [C, d] weight tensor back to the original parameter shape.

    Uses the layer_type and original_shape stored in LayerDecomposition,
    which follow the same conventions as the old LayerPCA.
    """
    C  = decomp.num_components
    t  = decomp.layer_type
    os = decomp.original_shape

    if t in ("conv2d", "conv1d", "depthwise_conv"):
        return weight_flat.view(C, *os)

    elif t in (
        "linear",
        "attention_query", "attention_key",
        "attention_value", "attention_output",
    ):
        # os = (in_features,) for Linear; (in_features, 1, 1) for Conv2d 1×1
        if len(os) == 3:
            return weight_flat.unsqueeze(-1).unsqueeze(-1)
        return weight_flat

    elif t == "attention_fused_qkv":
        if len(os) == 3:
            return weight_flat.unsqueeze(-1).unsqueeze(-1)
        return weight_flat

    elif t in ("attention_mha_head", "attention_mha_head_fused_qkv"):
        # os = (num_heads, head_dim, in_dim)
        num_heads, head_dim, in_dim = os
        return weight_flat.view(num_heads * head_dim, in_dim)

    else:
        raise ValueError(f"Unknown layer type: {t!r}")


class WeightReconstructor:
    """Manages decomposition coordinates and weight reconstruction."""

    def __init__(
        self,
        decompositions: dict[str, LayerDecomposition],
        projections:    dict[str, dict],
        device:         str = "cuda:0",
    ):
        self._decompositions = decompositions
        self._device         = device
        self._layer_keys: list[str] = list(decompositions.keys())

        missing = set(decompositions) - set(projections)
        if missing:
            raise ValueError(
                f"{len(missing)} layer(s) in decomposers file missing from "
                f"projection file: {list(missing)[:5]} ..."
            )

        # Current k-dimensional coordinates per layer: [C, k]  (CPU)
        self._current_Z: dict[str, torch.Tensor] = {
            key: projections[key]["points_3d"].clone().float()
            for key in self._layer_keys
        }

        # Initial coordinates for reset on renderer reconnect
        self._initial_Z: dict[str, torch.Tensor] = {
            key: projections[key]["points_3d"].clone().float()
            for key in self._layer_keys
        }

        # Residuals for reconstruction tensor for PCA, None otherwise.
        # For UMAP and FourierKPCA the info dict is not stored in the
        # projection file (return_residuals=False in project_model), so
        # reconstruction for those methods is always approximate from Z.
        self._residuals: dict[str, torch.Tensor | None] = {
            key: (
                projections[key]["residuals"].clone().float()
                if isinstance(projections[key].get("residuals"), torch.Tensor)
                else None
            )
            for key in self._layer_keys
        }

        #self._layer_colors: np.ndarray | None = self._build_layer_colors(projections)
        #Uncomment this (and comment the one above) for per-layer-type colors
        self._layer_colors = self._build_layer_type_colors(decompositions)


    # Properties

    @property
    def layer_keys(self) -> list[str]:
        return self._layer_keys

    @property
    def layer_colors(self) -> np.ndarray | None:
        return self._layer_colors
    
    @property
    def layer_types(self) -> dict[str, str]:
        return {key: decomp.layer_type for key, decomp in self._decompositions.items()}

    # Point update methods

    def apply_point_updates(self, msg: TransformedPointsMsg) -> None:
        """Apply a partial update (one layer) from the renderer's gizmo transform."""
        if msg.cloud_id >= len(self._layer_keys):
            print(f"[WeightReconstructor] Ignoring update for unknown layer "
                  f"{msg.cloud_id}")
            return
        layer_key               = self._layer_keys[msg.cloud_id]
        component_indices       = msg.indices.astype(np.int64)
        positions               = torch.from_numpy(msg.positions)
        self._current_Z[layer_key][component_indices] = positions

    def apply_full_cloud(self, msg: PointCloudMsg) -> None:
        """Replace all coordinates from a full PointCloudMsg."""
        points = torch.from_numpy(msg.points)   # [total_components, k]
        offset = 0
        for key in self._layer_keys:
            C = self._decompositions[key].num_components
            self._current_Z[key] = points[offset:offset + C].clone()
            offset += C

    def reset(self) -> None:
        """Restore initial coordinates (called on renderer reconnect)."""
        for key in self._layer_keys:
            self._current_Z[key] = self._initial_Z[key].clone()

    # Weight reconstruction

    def reconstruct_layer(self, layer_key: str) -> torch.Tensor:
        """
        Reconstruct a single layer's weight tensor from its current coordinates.

        Uses the layer's fitted decomposer (PCA / UMAP / FourierKPCA).
        Residuals are included when available (PCA only). They represent the
        component of the original weight orthogonal to the k-dim subspace, and
        are kept fixed during interactive manipulation.
        """
        decomp    = self._decompositions[layer_key]
        Z         = self._current_Z[layer_key].to(self._device)
        residuals = self._residuals.get(layer_key)
        if residuals is not None:
            residuals = residuals.to(self._device)

        # Reconstruct [C, d] via the decomposer's inverse_transform
        weight_flat = decomp.decomposer.inverse_transform(Z, residuals)
        weight_flat = weight_flat.to(self._device)

        return _reshape_to_original(weight_flat, decomp)

    def reconstruct_and_load(self, model: nn.Module) -> None:
        """Reconstruct every decomposed layer and patch them into the model."""
        state_dict = model.state_dict()
        for key in self._layer_keys:
            state_dict[key].copy_(self.reconstruct_layer(key))

    # IPC message builders

    def build_point_cloud(self) -> PointCloudMsg:
        """Flatten all current layer projections into a single PointCloudMsg."""
        all_points = np.concatenate(
            [self._current_Z[key].numpy() for key in self._layer_keys],
            axis=0,
        ).astype(np.float32)

        if all_points.shape[1] != 3:
            raise ValueError(
                f"Renderer expects 3-D points but decomposition has "
                f"k={all_points.shape[1]}. Re-run projection.py with -k 3."
            )
        total = all_points.shape[0]
        print(f"[WeightReconstructor] Point cloud: {total:,} points, "
              f"{len(self._layer_keys)} layers")
        return PointCloudMsg(cloud_id=0, points=all_points)

    def build_layer_meta(self, num_classes: int = 0) -> LayerMetaMsg:
        """Build the LayerMetaMsg to send before the point cloud."""
        colors = (self._layer_colors.flatten().tolist()
                  if self._layer_colors is not None else None)
        return LayerMetaMsg.from_lists(
            cloud_id=0,
            layer_keys=self._layer_keys,
            point_counts=[len(self._current_Z[k]) for k in self._layer_keys],
            colors=colors,
            layer_types=[self._decompositions[k].layer_type for k in self._layer_keys],
            num_classes=num_classes,
        )

    # Internal helpers

    @staticmethod
    def _build_layer_colors(projections: dict) -> np.ndarray | None:
        colors = []
        for key, proj in projections.items():
            color = proj.get("color")
            if color is None:
                return None
            colors.append(color.numpy() if hasattr(color, "numpy") else color)
        return np.stack(colors, axis=0).astype(np.float32)
    
    @staticmethod
    def _build_layer_type_colors(decompositions: dict) -> np.ndarray:
        colors = []
        for decomp in decompositions.values():
            lt = decomp.layer_type
            colors.append(LAYER_TYPE_COLORS.get(lt, (0.5, 0.5, 0.5)))
        return np.array(colors, dtype=np.float32)
