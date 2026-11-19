from __future__ import annotations
 
from abc import ABC, abstractmethod
 
import torch

class BaseDecomposer(ABC):
    """
    Common interface for all decomposition methods.
 
    All methods operate on batches of components simultaneously where
    possible. The unit of decomposition is one component (e.g. one
    convolutional filter, one attention head) whose weight vector has
    dimension d and whose training trajectory spans T checkpoints.
 
    fit() receives all components for one layer at once:
        [num_components, T, d]
 
    transform() and inverse_transform() also operate on all components
    simultaneously:
        [num_components, d] -> [num_components, k]
    """
 
    @abstractmethod
    def fit(self, trajectories: torch.Tensor) -> None:
        """
        Learn the decomposition from a training trajectory.
 
        Args:
            trajectories: Float32 tensor of shape [num_components, T, d]
                          where T = number of training checkpoints and
                          d = flattened component dimensionality.
        """
 
    @abstractmethod
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project components into the k-dimensional learned space.
 
        Args:
            X: [num_components, d], one weight vector per component
 
        Returns:
            Z: [num_components, k]
        """
 
    @abstractmethod
    def inverse_transform(
        self,
        Z: torch.Tensor,
        residuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct weight vectors from the k-dimensional embedding.
 
        Args:
            Z:         [num_components, k]
            residuals: [num_components, d]  optional exact-reconstruction
                       residuals (from transform with return_residuals=True)
 
        Returns:
            X_hat: [num_components, d]
        """
 
    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize fitted state to a plain dict (torch-saveable)."""
 
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseDecomposer":
        """Reconstruct from a saved dict."""
 
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True once fit() has been called."""
 
    @property
    @abstractmethod
    def k(self) -> int:
        """Output dimensionality."""
 
    @property
    @abstractmethod
    def n_components(self) -> int:
        """Number of weight components this decomposer was fit on."""
