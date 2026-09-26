"""AMOR spatial bases (one module per family).

Used by PCAReducer.fit_alternative_basis / train_*_basis scripts.
"""

from .common import BASIS_TYPES, channelize_rows, flatten_samples, orthonormal_rows, pca_like_from_rows
from .laplacian import laplacian_eigen_rows
from .polynomial import polynomial_rows
from .dmd import dmd_mode_rows
from .random_init import random_orthonormal_rows

__all__ = [
    "BASIS_TYPES",
    "channelize_rows",
    "flatten_samples",
    "orthonormal_rows",
    "pca_like_from_rows",
    "laplacian_eigen_rows",
    "polynomial_rows",
    "dmd_mode_rows",
    "random_orthonormal_rows",
]
