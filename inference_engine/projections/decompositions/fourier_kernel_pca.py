from __future__ import annotations

import torch
import numpy as np

from projections.decompositions.base_decomposition import BaseDecomposer


class _Cos:
    def transform(self, angle):
        main_component = angle % (2 * np.pi)
        k = angle // (2 * np.pi)
        p = np.where((main_component > np.pi), 1, 0)
        return np.cos(main_component), {'k': k * 2 * np.pi, 'p': p}

    def invert_transform(self, fx, info):
        fx_bounded = np.minimum(np.maximum(fx, -1), 1)
        angle = np.where(info['p'],
                         2 * np.pi - np.arccos(fx_bounded),
                         np.arccos(fx_bounded))
        return angle + info['k']


class _AffineTransform:
    def __init__(self, A, b, regularization=0):
        self.A = A
        self.b = b
        self.regularization = regularization
        self.m, self.n = A.shape
        if self.m < self.n and regularization == 0:
            raise ValueError(
                'Only works when A raises the dimension of the input vector. '
                'Set regularization > 0.'
            )
        m, = b.shape
        if self.m != m:
            raise ValueError('b has the wrong dimension')
        if regularization > 0:
            self.A_pinv = np.linalg.inv(
                self.A.T @ self.A + regularization * np.eye(self.n)
            ) @ self.A.T
        else:
            import scipy.linalg as linalg
            self.u, self.s, self.vh = linalg.svd(
                A, full_matrices=False, compute_uv=True
            )

    def transform(self, x):
        return x @ self.A.T + self.b, None

    def invert_transform(self, fx, _info):
        y = fx - self.b
        if self.regularization > 0:
            x = y @ self.A_pinv.T
        else:
            x = ((1 / self.s) * (y @ self.u)) @ self.vh
        return x


class _RBFSampler:
    def __init__(self, n_features, gamma=1.0, n_components=100,
                 random_state=None, regularization=0):
        rng = np.random.RandomState(random_state)
        random_weights_ = np.sqrt(2 * gamma) * rng.normal(
            size=(n_components, n_features)
        )
        random_offset_ = rng.uniform(0, 2 * np.pi, size=n_components)
        self.aff = _AffineTransform(random_weights_, random_offset_,
                                    regularization)
        self.cos = _Cos()
        self.n_features    = n_features
        self.gamma         = gamma
        self.n_components  = n_components
        self.random_state  = random_state
        self.regularization = regularization

    def transform(self, x):
        x1, _   = self.aff.transform(x)
        z, info = self.cos.transform(x1)
        return np.sqrt(2.0) / np.sqrt(self.n_components) * z, info

    def invert_transform(self, f_x, info):
        f_x_rescaled = np.sqrt(self.n_components) / np.sqrt(2.0) * f_x
        f_x1 = self.cos.invert_transform(f_x_rescaled, info)
        x    = self.aff.invert_transform(f_x1, None)
        return x


class FourierKPCADecomposer(BaseDecomposer):
    """
    Layer-level Invertible Kernel PCA via Random Fourier Features fitted on
    difference matrices.

    Pipeline:
        fit([C, T, d] raw weights):
            1. Compute \\DeltaW_c^(t) = W_c^(t) - W_c^(1)
            2. Map [C*T, d] differences -> [C*T, D] via RBF feature map
            3. Fit ONE shared PCA in feature space -> shared mean [D], evecs [D,k]

        transform([C, d] raw weights) -> Z [C, k]:
            \\DeltaX       = X - init_weights
            phi, info = rbf.transform(\\DeltaX)
            Z         = (phi - mean_feat) @ evecs

        inverse_transform(Z, info) -> X [C, d]:
            phi_hat  = Z @ evecs.T + mean_feat
            delta_hat = rbf.invert_transform(phi_hat, info)
            X_hat    = delta_hat + init_weights (reconstructed raw weights)
    """

    def __init__(
        self,
        k: int = 3,
        n_rff_features: int = 1024,
        gamma: float | None = None,
        random_state: int = 42,
        regularization: float = 0.0,
    ):
        self._k              = k
        self._n_rff_features = n_rff_features
        self._gamma          = gamma
        self._random_state   = random_state
        self._regularization = regularization

        self.rbf:              _RBFSampler | None  = None
        self._init_weights:    torch.Tensor | None = None   # [C, d]
        self._mean_feat:       torch.Tensor | None = None   # [D]
        self._evecs:           torch.Tensor | None = None   # [D, k]
        self._evals:           torch.Tensor | None = None   # [k]
        self._expl_var_layer:  torch.Tensor | None = None
        self._expl_var_unit:   torch.Tensor | None = None   # [C]
        self._C:               int = 0

    @staticmethod
    def _estimate_gamma(delta_np: np.ndarray) -> float:
        sample = delta_np.reshape(-1, delta_np.shape[-1])
        if len(sample) > 2000:
            idx = np.random.choice(len(sample), 2000, replace=False)
            sample = sample[idx]
        sq = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
        med = np.median(sq[sq > 0]) + 1e-8
        return 1.0 / (2.0 * med)

    def fit(self, trajectories: torch.Tensor) -> None:
        """
        Args:
            trajectories: [C, T, d]  RAW weight vectors.
        """
        C, T, d = trajectories.shape
        self._C = C

        # Store initialization weights
        self._init_weights = trajectories[:, 0, :].clone()     # [C, d]

        # Compute difference matrices
        delta = trajectories - self._init_weights.unsqueeze(1) # [C, T, d]
        delta_np = delta.float().cpu().numpy()                  # [C, T, d]

        gamma = (self._gamma if self._gamma is not None
                 else self._estimate_gamma(delta_np))

        D = self._n_rff_features
        self.rbf = _RBFSampler(
            n_features=d,
            gamma=gamma,
            n_components=D,
            random_state=self._random_state,
            regularization=self._regularization,
        )

        # Map ALL C*T difference vectors through RBF feature map 
        delta_flat = delta_np.reshape(C * T, d)                # [C*T, d]
        phi_flat, _ = self.rbf.transform(delta_flat)           # [C*T, D]
        phi_flat = phi_flat.astype(np.float32)

        # Fit ONE shared PCA in feature space
        phi = torch.from_numpy(phi_flat)                       # [C*T, D]
        mean_feat = phi.mean(dim=0)                            # [D]
        phi_c     = phi - mean_feat                            # [C*T, D]

        _, S, Vt = torch.linalg.svd(phi_c, full_matrices=False)
        self._mean_feat = mean_feat
        self._evecs     = Vt[:self._k].T                       # [D, k]

        eigenvalues = (S ** 2) / max(C * T - 1, 1)
        total_var   = eigenvalues.sum()
        self._evals = eigenvalues[:self._k]
        self._expl_var_layer = (
            self._evals.sum() / total_var.clamp(min=1e-12)
        )

        # Per-unit explained variance in feature space
        phi_units = phi.reshape(C, T, D)
        phi_u_c   = phi_units - mean_feat
        proj_u    = phi_u_c @ self._evecs
        unit_total = (phi_u_c ** 2).sum(dim=(1, 2))
        unit_proj  = (proj_u  ** 2).sum(dim=(1, 2))
        self._expl_var_unit = (
            unit_proj / unit_total.clamp(min=1e-12)
        )

    def transform(
        self,
        X: torch.Tensor,
        return_residuals: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            X: [C, d]  RAW weight vectors
            return_residuals: if True return cos branch info for exact inverse

        Returns:
            Z: [C, k]
            info: cos branch dict or None
        """
        self._check_fitted()
        init = self._init_weights                              # [C, d]

        # Compute difference then apply RBF feature map
        delta_np = (X.float().cpu() - init).numpy()           # [C, d]
        phi_np, info = self.rbf.transform(delta_np)           # [C, D], info
        phi   = torch.from_numpy(phi_np.astype(np.float32))
        phi_c = phi - self._mean_feat                         # [C, D]

        Z = phi_c @ self._evecs                               # [C, k]

        return Z, (info if return_residuals else None)

    def inverse_transform(
        self,
        Z: torch.Tensor,
        residuals=None,                                       # cos branch info
    ) -> torch.Tensor:
        """
        Args:
            Z: [C, k]
            residuals: cos branch info dict from transform(return_residuals=True)

        Returns:
            X_hat: [C, d]  RAW reconstructed weights
        """
        self._check_fitted()
        info = residuals

        # Reverse PCA in feature space -> delta_hat [C, d]
        phi_hat_c = Z @ self._evecs.T                         # [C, D]
        phi_hat   = (phi_hat_c + self._mean_feat).numpy()

        if info is not None:
            delta_hat = self.rbf.invert_transform(phi_hat, info)
        else:
            f_x_rescaled = (np.sqrt(self.rbf.n_components) /
                            np.sqrt(2.0)) * phi_hat
            f_x1      = np.arccos(np.clip(f_x_rescaled, -1, 1))
            delta_hat = self.rbf.aff.invert_transform(f_x1, None)

        # Add back initialization weights to get raw weight reconstruction
        delta_tensor = torch.from_numpy(delta_hat.astype(np.float32))
        return delta_tensor + self._init_weights              # [C, d]

    @property
    def is_fitted(self) -> bool:
        return self.rbf is not None and self._evecs is not None

    @property
    def k(self) -> int:
        return self._k

    @property
    def n_components(self) -> int:
        return self._C

    @property
    def explained_variance(self) -> torch.Tensor | None:
        return self._expl_var_unit

    @property
    def explained_variance_layer(self) -> torch.Tensor | None:
        return self._expl_var_layer

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "method":          "fourier_kpca",
            "k":               self._k,
            "n_rff_features":  self._n_rff_features,
            "gamma":           self._gamma,
            "random_state":    self._random_state,
            "regularization":  self._regularization,
            "C":               self._C,
            "init_weights":    self._init_weights.cpu(),
            "random_weights_": torch.from_numpy(self.rbf.aff.A),
            "random_offset_":  torch.from_numpy(self.rbf.aff.b),
            "rbf_gamma":       self.rbf.gamma,
            "mean_feat":       self._mean_feat.cpu(),
            "evecs":           self._evecs.cpu(),
            "evals":           self._evals.cpu(),
            "expl_var":        self._expl_var_unit.cpu(),
            "expl_var_layer":  self._expl_var_layer.cpu(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FourierKPCADecomposer":
        obj = cls(
            k=data["k"],
            n_rff_features=data["n_rff_features"],
            gamma=data.get("gamma"),
            random_state=data["random_state"],
            regularization=data.get("regularization", 0.0),
        )
        random_weights_ = data["random_weights_"].numpy()
        random_offset_  = data["random_offset_"].numpy()

        obj.rbf = _RBFSampler.__new__(_RBFSampler)
        obj.rbf.aff = _AffineTransform(
            random_weights_, random_offset_, obj._regularization
        )
        obj.rbf.cos           = _Cos()
        obj.rbf.n_features    = random_weights_.shape[1]
        obj.rbf.gamma         = data["rbf_gamma"]
        obj.rbf.n_components  = random_weights_.shape[0]
        obj.rbf.random_state  = data["random_state"]
        obj.rbf.regularization = obj._regularization

        obj._C            = data.get("C", data["expl_var"].shape[0])
        obj._init_weights = data["init_weights"]
        obj._mean_feat    = data["mean_feat"]
        obj._evecs        = data["evecs"]
        obj._evals        = data["evals"]
        obj._expl_var_unit  = data["expl_var"]
        obj._expl_var_layer = data.get(
            "expl_var_layer", data["expl_var"].mean()
        )
        return obj

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "FourierKPCADecomposer must be fit() before transform()."
            )
