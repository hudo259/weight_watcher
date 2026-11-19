from __future__ import annotations

import pickle
import warnings
import torch
import numpy as np
import math

from projections.decompositions.base_decomposition import BaseDecomposer

warnings.filterwarnings(
    "ignore",
    message="n_jobs value .* overridden",
    category=UserWarning,
)


class UMAPDecomposer(BaseDecomposer):
    """
    Layer-level UMAP fitted on final displacement vectors.

    Pipeline:
        fit([C, T, d] raw weights):
            1. Compute ΔW_c^(T) = W_c^(T) - W_c^(1) (final displacements)
            2. Fit ONE umap.UMAP on [C, d] displacement vectors

        transform([C, d] raw weights) -> Z [C, k]:
            \\DeltaX   = X - init_weights
            Z    = umap.transform(ΔX)

        inverse_transform(Z [C, k]) -> X [C, d]:
            \\DeltaX_hat = umap.inverse_transform(Z)
            X_hat  = ΔX_hat + init_weights (approximate raw weight recon)

    One UMAP fit on [C, d] final displacements.
    Shared coordinate system: all units comparable.
    """

    def __init__(
        self,
        k: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self._k            = k
        self._n_neighbors  = n_neighbors
        self._min_dist     = min_dist
        self._metric       = metric
        self._random_state = random_state
        self._model        = None
        self._init_weights: torch.Tensor | None = None    # [C, d]
        self._C:            int = 0
        self._effective_n_neighbors: int = 0

    def fit(self, trajectories: torch.Tensor) -> None:
        """
        Args:
            trajectories: [C, T, d]  RAW weight vectors.
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required. Install with: pip install umap-learn"
            )

        C, T, d = trajectories.shape
        self._C = C

        # Store initialization weights
        self._init_weights = trajectories[:, 0, :].clone()    # [C, d]

        # Final displacement ΔW_c^(T) = W_c^(T) - W_c^(1)
        final_delta = (trajectories[:, -1, :] -
                       self._init_weights).float().cpu().numpy()  # [C, d]

        #exclude up to C-1
        #effective_neighbors = min(self._n_neighbors, C - 1)

        #exclude
        effective_neighbors = max(2, min(self._n_neighbors, math.ceil(0.33 * C), C-1))
        self._effective_n_neighbors = effective_neighbors
        if effective_neighbors < self._n_neighbors:
            print(
                f"  [UMAP] Warning: C={C} units < n_neighbors="
                f"{self._n_neighbors}. Using n_neighbors={effective_neighbors}."
            )

        print(f"  [UMAP] Fitting one model on {C} displacement vectors (d={d})...")
        self._model = umap.UMAP(
            n_components=self._k,
            n_neighbors=effective_neighbors,
            min_dist=self._min_dist,
            metric=self._metric,
            random_state=self._random_state,
        )
        self._model.fit(final_delta)
        print("  [UMAP] Done.")

    def transform(
        self,
        X: torch.Tensor,
        return_residuals: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            X: [C, d]  RAW weight vectors

        Returns:
            Z:         [C, k]
            residuals: always None (UMAP has no linear residual concept)
        """
        self._check_fitted()
        delta_np = (X.float().cpu() -
                    self._init_weights).numpy()               # [C, d]
        Z_np = self._model.transform(delta_np)               # [C, k]
        return torch.tensor(Z_np, dtype=torch.float32), None

    def inverse_transform(
        self,
        Z: torch.Tensor,
        residuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Approximate inverse via barycentric interpolation.

        Args:
            Z: [C, k]

        Returns:
            X_hat: [C, d]  approximate RAW weight reconstruction
        """
        self._check_fitted()
        Z_np = Z.float().cpu().numpy()
        try:
            delta_hat = self._model.inverse_transform(Z_np)  # [C, d]
        except AttributeError:
            raise RuntimeError(
                "UMAP inverse_transform not available. "
                "Upgrade to umap-learn >= 0.5."
            )
        delta_tensor = torch.tensor(delta_hat, dtype=torch.float32)
        return delta_tensor + self._init_weights             # [C, d] raw weights

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def k(self) -> int:
        return self._k

    @property
    def n_components(self) -> int:
        return self._C

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "method":        "umap",
            "k":             self._k,
            "n_neighbors":   self._n_neighbors,
            "effective_n_neighbors": self._effective_n_neighbors,
            "min_dist":      self._min_dist,
            "metric":        self._metric,
            "random_state":  self._random_state,
            "C":             self._C,
            "init_weights":  self._init_weights.cpu(),
            "model_bytes":   pickle.dumps(self._model),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UMAPDecomposer":
        obj = cls(
            k=data["k"],
            n_neighbors=data["n_neighbors"],
            min_dist=data["min_dist"],
            metric=data["metric"],
            random_state=data["random_state"],
        )
        obj._C            = data.get("C", 0)
        obj._init_weights = data.get("init_weights")
        obj._effective_n_neighbors = data.get("effective_n_neighbors", data["n_neighbors"])
        if "model_bytes" in data:
            obj._model = pickle.loads(data["model_bytes"])
        elif "models_bytes" in data:
            raise RuntimeError(
                "Old per-unit UMAP format detected. Please rerun projection.py."
            )
        return obj

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "UMAPDecomposer must be fit() before transform()."
            )
