"""Laplacian eigenfunction spatial basis on an N×M grid."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .random_init import random_orthonormal_rows


def _grid_1d_laplacian(n: int, periodic: bool) -> sp.csr_matrix:
    if periodic:
        diag = np.full(n, 2.0, dtype=np.float64)
        off = np.full(n, -1.0, dtype=np.float64)
        l_mat = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="lil")
        l_mat[0, n - 1] = -1.0
        l_mat[n - 1, 0] = -1.0
    else:
        diag = np.ones(n, dtype=np.float64)
        diag[1:-1] = 2.0
        off = -np.ones(n - 1, dtype=np.float64)
        l_mat = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr")
    return l_mat.tocsr()


def laplacian_eigen_rows(n: int, m: int, k: int, periodic: bool) -> np.ndarray:
    lx = _grid_1d_laplacian(n, periodic)
    ly = _grid_1d_laplacian(m, periodic)
    l2 = sp.kron(lx, sp.eye(m, format="csr")) + sp.kron(sp.eye(n, format="csr"), ly)
    n_eigs = min(k, l2.shape[0] - 2)
    _, evecs = spla.eigsh(l2, k=n_eigs, which="SM")
    rows = evecs.T.astype(np.float32)
    if rows.shape[0] < k:
        pad = random_orthonormal_rows(n * m, k - rows.shape[0], random_state=0)
        rows = np.concatenate([rows, pad], axis=0)
    return rows[:k]
