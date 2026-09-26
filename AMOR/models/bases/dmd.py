"""Exact DMD spatial modes (hybrid DMD basis)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .common import orthonormal_rows


def dmd_mode_rows(
    snapshots: np.ndarray,
    snapshots_next: np.ndarray,
    k: int,
    svd_rank: Optional[int] = None,
) -> np.ndarray:
    x = np.asarray(snapshots, dtype=np.float64).T
    xp = np.asarray(snapshots_next, dtype=np.float64).T
    if x.shape != xp.shape:
        raise ValueError(f"DMD snapshot shape mismatch: {x.shape} vs {xp.shape}")
    rank = min(x.shape) if svd_rank is None else min(svd_rank, min(x.shape))
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    rank = min(rank, u.shape[1])
    ur = u[:, :rank]
    sr = s[:rank]
    v = vh[:rank, :].T
    a_tilde = ur.T @ xp @ v @ np.diag(1.0 / sr)
    evals, w = np.linalg.eig(a_tilde)
    modes = ur @ w
    order = np.argsort(np.abs(evals))[::-1]
    mode_rows = modes[:, order].T.real
    return orthonormal_rows(mode_rows, k)
