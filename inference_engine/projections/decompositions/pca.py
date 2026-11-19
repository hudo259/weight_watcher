from __future__ import annotations

from projections.decompositions.base_decomposition import BaseDecomposer

import torch


class PCADecomposer(BaseDecomposer):
    """
    Layer-level linear PCA fitted on difference matrices.

    Pipeline:
        fit([C, T, d] raw weights):
            1. Compute \\DeltaW_c^(t) = W_c^(t) - W_c^(1)  [difference matrices]
            2. Stack all C x T difference vectors -> X_\\ell [C*T, d]
            3. Fit ONE shared SVD -> V_k^(\\ell) [d, k]

        transform([C, d] raw weights) -> Z [C, k]:
            \\DeltaX = X - init_weights
            Z  = (\\DeltaX - \\mu) @ V_k

        inverse_transform(Z [C, k], residuals) -> X [C, d]:
            \\DeltaX_hat = Z @ V_k.T + \\mu + residuals
            X_hat  = \\DeltaX_hat + init_weights (exact reconstruction)

    The shared basis V_k^(\\ell) places all C units in the same coordinate
    system, making cross-unit distances and clustering meaningful.
    init_weights = W_c^(1) is stored per unit to enable projection of any
    new checkpoint into the same space.
    """

    def __init__(self, k: int = 3):
        self._k = k
        self._init_weights:    torch.Tensor | None = None   # [C, d]
        self._mean:            torch.Tensor | None = None   # [d]
        self._eigenvectors:    torch.Tensor | None = None   # [d, k]
        self._eigenvalues:     torch.Tensor | None = None   # [k]
        self._explained_var_layer: torch.Tensor | None = None
        self._explained_var_unit:  torch.Tensor | None = None  # [C]

    def fit(self, trajectories: torch.Tensor) -> None:
        """
        Args:
            trajectories: [C, T, d]  RAW weight vectors (not pre-differenced).
                          Differences are computed internally here.
        """
        C, T, d = trajectories.shape
        if T < self._k:
            raise ValueError(
                f"PCADecomposer: T={T} checkpoints < k={self._k}."
            )

        # Store initialization weights W_c^(1
        self._init_weights = trajectories[:, 0, :].clone()     # [C, d]

        # Compute difference matrices ΔW_c^(t) = W_c^(t) - W_c^(1)
        delta = trajectories - self._init_weights.unsqueeze(1) # [C, T, d]
        # Note: delta[:, 0, :] is all zeros by construction (t=1 reference)

        # Stack all (unit, time) difference vectors
        X_all = delta.reshape(C * T, d)                        # [C*T, d]
        self._mean = X_all.mean(dim=0)                         # [d]
        X_c   = X_all - self._mean                             # [C*T, d]

        # Single shared SVD
        _, S, Vt = torch.linalg.svd(X_c, full_matrices=False)
        self._eigenvectors = Vt[:self._k].T                    # [d, k]

        eigenvalues = (S ** 2) / max(C * T - 1, 1)
        total_var   = eigenvalues.sum()
        self._eigenvalues  = eigenvalues[:self._k]
        self._explained_var_layer = (
            self._eigenvalues.sum() / total_var.clamp(min=1e-12)
        )

        # Per-unit explained variance: fraction of each unit's trajectory
        # variance captured by the shared layer-level basis
        delta_c = delta - self._mean                            # [C, T, d]
        proj    = delta_c @ self._eigenvectors                  # [C, T, k]
        unit_total = (delta_c ** 2).sum(dim=(1, 2))             # [C]
        unit_proj  = (proj    ** 2).sum(dim=(1, 2))             # [C]
        self._explained_var_unit = (
            unit_proj / unit_total.clamp(min=1e-12)
        )                                                       # [C]

    def transform(
        self,
        X: torch.Tensor,
        return_residuals: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            X:                [C, d]  RAW weight vectors
            return_residuals: if True return out-of-subspace residual
                              (in difference space) for lossless reconstruction

        Returns:
            Z:         [C, k]
            residuals: [C, d] in difference space, or None
        """
        self._check_fitted()
        device = X.device
        init = self._init_weights.to(device)                   # [C, d]
        mean = self._mean.to(device)                           # [d]
        V    = self._eigenvectors.to(device)                   # [d, k]

        delta = X - init                                        # [C, d] ΔW
        delta_c = delta - mean                                  # [C, d] centered
        Z = delta_c @ V                                         # [C, k]

        residuals = None
        if return_residuals:
            residuals = delta_c - Z @ V.T                      # [C, d] in Δ-space

        return Z, residuals

    def inverse_transform(
        self,
        Z: torch.Tensor,
        residuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Z:         [C, k]
            residuals: [C, d] in difference space (from transform()) or None

        Returns:
            X_hat: [C, d]  RAW reconstructed weights (lossless with residuals)
        """
        self._check_fitted()
        device = Z.device
        init = self._init_weights.to(device)                   # [C, d]
        mean = self._mean.to(device)
        V    = self._eigenvectors.to(device)

        delta_hat = Z @ V.T + mean                             # [C, d] in Δ-space
        if residuals is not None:
            delta_hat = delta_hat + residuals.to(device)

        return delta_hat + init                                 # [C, d] raw weights

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    @property
    def k(self) -> int:
        return self._k

    @property
    def n_components(self) -> int:
        return (self._explained_var_unit.shape[0]
                if self._explained_var_unit is not None else 0)

    @property
    def explained_variance(self) -> torch.Tensor | None:
        """Per-unit explained variance [C]."""
        return self._explained_var_unit

    @property
    def explained_variance_layer(self) -> torch.Tensor | None:
        return self._explained_var_layer

    @property
    def eigenvalues(self) -> torch.Tensor | None:
        return self._eigenvalues

    # Serialisation
    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "method":            "pca",
            "k":                 self._k,
            "init_weights":      self._init_weights.cpu(),
            "mean":              self._mean.cpu(),
            "eigenvectors":      self._eigenvectors.cpu(),
            "eigenvalues":       self._eigenvalues.cpu(),
            "explained_var":     self._explained_var_unit.cpu(),
            "explained_var_layer": self._explained_var_layer.cpu(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PCADecomposer":
        obj = cls(k=data["k"])
        obj._init_weights       = data["init_weights"]
        obj._mean               = data["mean"]
        obj._eigenvectors       = data["eigenvectors"]
        obj._eigenvalues        = data["eigenvalues"]
        obj._explained_var_unit = data["explained_var"]
        obj._explained_var_layer = data.get(
            "explained_var_layer",
            data["explained_var"].mean(),
        )
        return obj

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "PCADecomposer must be fit() before transform()."
            )
