import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from sklearn.decomposition import PCA


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors with optional boundary weighting."""

    def __init__(
        self,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
        bc_weight: float = 1.0,
        boundary_threshold: float = 0.1,
        use_boundary_weighting: bool = False,
    ):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.bc_weight = bc_weight
        self.boundary_threshold = boundary_threshold
        self.use_boundary_weighting = use_boundary_weighting
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels: Optional[int] = None
        self.out_channels: Optional[int] = None
        self.shape_: Optional[tuple] = None
        self.output_shape_: Optional[tuple] = None

    def fit(self, samples: np.ndarray, solution: np.ndarray) -> "PCAReducerMultiChannel":
        # Handle both 3D (irregular grid) and 4D (regular grid) inputs
        if samples.ndim == 3:
            # Irregular grid: (num_samples, C_in, N) -> (num_samples, C_in, N, 1)
            samples = samples[:, :, :, np.newaxis]
        if samples.ndim != 4:
            raise ValueError(f"Expected 3D (num_samples, C_in, N) or 4D (num_samples, C_in, N, M), got {samples.shape}")
        
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)

        # Handle both 3D and 4D solution inputs
        if solution.ndim == 2:
            # (num_samples, N) -> (num_samples, 1, N, 1) for irregular grid
            solution = solution[:, np.newaxis, :, np.newaxis]
        elif solution.ndim == 3:
            # (num_samples, C_out, N) -> (num_samples, C_out, N, 1) for irregular grid
            solution = solution[:, :, :, np.newaxis]
        elif solution.ndim == 4:
            pass  # Already 4D
        else:
            raise ValueError(
                f"Expected 2D (num_samples, N), 3D (num_samples, C_out, N), or 4D (num_samples, C_out, N, M), got {solution.shape}"
            )
        
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError(f"Input and output spatial dimensions must match: input ({n}, {m}) vs output ({n_out}, {m_out})")
        self.out_channels = c_out
        self.output_shape_ = (c_out, n, m)
        y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        if self.use_boundary_weighting:
            boundary_weights = self._compute_boundary_weights(n, m)
            y_weighted = y_out * boundary_weights.reshape(1, -1)
        else:
            y_weighted = y_out

        self.pca_in = PCA(
            n_components=None,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(x_in)
        self.pca_out = PCA(
            n_components=None,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(y_weighted)

        return self

    def _compute_boundary_weights(self, n: int, m: int) -> np.ndarray:
        weights = np.ones((n, m), dtype=np.float32)
        x_coords = np.linspace(0, 1, n)
        y_coords = np.linspace(0, 1, m)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")
        dist_to_boundary = np.minimum(
            np.minimum(grid_x, 1 - grid_x),
            np.minimum(grid_y, 1 - grid_y),
        )
        boundary_mask = dist_to_boundary < self.boundary_threshold
        weights[boundary_mask] = self.bc_weight
        return weights.flatten()

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    @property
    def input_components(self) -> np.ndarray:
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_
