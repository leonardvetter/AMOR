import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============= PCAReducer Class =============
@dataclass
class PCAConfig:
    svd_solver: str = 'auto'
    whiten: bool = False
    random_state: Optional[int] = None

class PCAReducer:
    """
    PCA-based reduced-basis helper for (num_samples, N, N) fields.
    - Fits separate PCA models for inputs and outputs.
    - Projects to first k components or components covering a variance cutoff j.
    - Reconstructs outputs from reduced coefficients.
    """
    def __init__(self, cfg: PCAConfig = PCAConfig()):
        self.cfg = cfg
        self.pca_in:  Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.shape_ = None  # (N, N)

    def fit(self, samples: np.ndarray, solution: np.ndarray):
        """
        samples, solution: (num_samples, N, N).
        """
        assert samples.shape == solution.shape and samples.ndim == 3, "Use (num_samples, N, N)"
        num_samples, N, M = samples.shape
        self.shape_ = (N, M)

        X_in  = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        Y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        self.pca_in  = PCA(n_components=None, svd_solver=self.cfg.svd_solver,
                           whiten=self.cfg.whiten, random_state=self.cfg.random_state).fit(X_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.cfg.svd_solver,
                           whiten=self.cfg.whiten, random_state=self.cfg.random_state).fit(Y_out)
        return self

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99) -> tuple[int, int]:
        """Return (k_in, k_out) meeting given variance cutoffs."""
        assert self.pca_in and self.pca_out
        k_in  = self._k_from_cutoff(self.pca_in.explained_variance_ratio_,  j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    def transform_inputs(self, samples: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Return input PCA coefficients (num_samples, k)."""
        assert self.pca_in
        X = samples.reshape(samples.shape[0], -1).astype(np.float32, copy=False)
        Z = self.pca_in.transform(X)
        return Z[:, :k] if k is not None else Z

    def transform_outputs(self, solution: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Return output PCA coefficients (num_samples, k)."""
        assert self.pca_out
        Y = solution.reshape(solution.shape[0], -1).astype(np.float32, copy=False)
        Z = self.pca_out.transform(Y)
        return Z[:, :k] if k is not None else Z

    def inverse_outputs(self, coeffs: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct outputs (num_samples, N, N) from output PCA coeffs.
        If k is provided, only the first k components are used.
        """
        assert self.pca_out and self.shape_ is not None
        if k is None:
            Y_flat = self.pca_out.inverse_transform(coeffs)
        else:
            comps = self.pca_out.components_[:k, :]
            mean  = self.pca_out.mean_
            Y_flat = coeffs @ comps + mean
        return Y_flat.reshape((coeffs.shape[0], *self.shape_))

    def plot_explained_variance(self):
        """Plot explained variance ratios for input and output PCA."""
        assert self.pca_in and self.pca_out
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(self.pca_in.explained_variance_ratio_, marker='o', linewidth=1)
        axes[0].set_title("Input PCA — explained variance ratio")
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("EVR")
        axes[0].grid(True)
        axes[1].plot(self.pca_out.explained_variance_ratio_, marker='o', linewidth=1)
        axes[1].set_title("Output PCA — explained variance ratio")
        axes[1].set_xlabel("Component")
        axes[1].set_ylabel("EVR")
        axes[1].grid(True)
        plt.show()

    @property
    def input_components(self) -> np.ndarray:
        """(n_in_components, D) principal directions for inputs."""
        assert self.pca_in
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        """(n_out_components, D) principal directions for outputs."""
        assert self.pca_out
        return self.pca_out.components_