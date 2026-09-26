"""Shared helpers for AMOR bases."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

BASIS_TYPES = (
    "pca",
    "dmd_hybrid",
    "chebyshev",
    "legendre",
    "jacobi",
    "laplacian",
)


def flatten_samples(samples: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    if samples.ndim != 4:
        raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
    num_samples, c_in, n, m = samples.shape
    x = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
    return x, num_samples, c_in, n, m


def orthonormal_rows(candidates: np.ndarray, k: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError(f"Expected 2D candidates, got {candidates.shape}")
    q, _ = np.linalg.qr(candidates.T, mode="reduced")
    rows = q.T.astype(np.float32)
    return rows[:k]


def pca_like_from_rows(mean: np.ndarray, components: np.ndarray, n_samples: int) -> PCA:
    """Build sklearn-PCA-compatible object from mean + row basis vectors."""
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    n_comp = int(components.shape[0])
    ev = np.ones(n_comp, dtype=np.float64)
    pca = PCA(n_components=n_comp)
    pca.mean_ = mean
    pca.components_ = components
    pca.explained_variance_ = ev
    pca.explained_variance_ratio_ = ev / max(float(ev.sum()), 1e-30)
    pca.n_samples_ = int(n_samples)
    pca.n_features_in_ = int(mean.shape[0])
    # sklearn>=1.x leaves n_components_ unset until fit(); set explicitly for row-built bases
    pca.n_components_ = n_comp
    return pca


def channelize_rows(rows_1ch: np.ndarray, c: int, k: int) -> np.ndarray:
    if c == 1:
        return rows_1ch[:k].astype(np.float32)
    p, s = rows_1ch.shape
    d = c * s
    rows = []
    for j in range(p):
        for ch in range(c):
            r = np.zeros(d, dtype=np.float32)
            r[ch * s:(ch + 1) * s] = rows_1ch[j]
            rows.append(r)
            if len(rows) >= k:
                return np.stack(rows, axis=0)
    return np.stack(rows, axis=0)
