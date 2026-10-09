"""P1 tetrahedral mass matrices (consistent sparse and optional lumped diagonal)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def tet_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    t = nodes[elements]
    v1 = t[:, 1] - t[:, 0]
    v2 = t[:, 2] - t[:, 0]
    v3 = t[:, 3] - t[:, 0]
    return np.abs(np.einsum("ij,ij->i", np.cross(v1, v2), v3)) / 6.0


def consistent_nodal_mass(nodes: np.ndarray, elements: np.ndarray) -> sp.csr_matrix:
    """Consistent P1 tet mass M_ij = ∫ φ_i φ_j dx (sparse SPD, not assumed diagonal).

    Reference element: M_e = (V/20) * (1 + δ_ij) on the 4 local dofs.
    Vectorized assembly (no Python per-element loops).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    n = nodes.shape[0]
    vols = tet_volumes(nodes, elements)
    ok = vols > 0.0
    if not np.any(ok):
        return sp.eye(n, format="csr", dtype=np.float64)
    elems = elements[ok]
    a = (vols[ok] / 20.0).astype(np.float64)  # (E,)
    # Local pattern (1+δ_ij): diag scale 2, off-diag 1 → 16 entries / tet
    ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    ii = ii.ravel()  # 16
    jj = jj.ravel()
    scale = np.where(ii == jj, 2.0, 1.0).astype(np.float64)  # (16,)
    # rows/cols: (E, 16)
    rows = elems[:, ii].reshape(-1)
    cols = elems[:, jj].reshape(-1)
    data = (a[:, None] * scale[None, :]).reshape(-1)
    M = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    diag = M.diagonal()
    if np.any(diag <= 0.0):
        fill = float(np.min(diag[diag > 0.0])) if np.any(diag > 0.0) else 1.0
        dfix = np.where(diag <= 0.0, fill, 0.0)
        M = M + sp.diags(dfix, format="csr")
    return M


def lumped_nodal_mass(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Row-sum lumping of the consistent P1 mass (diagonal vector). Prefer
    ``consistent_nodal_mass`` + sparse ops when an inner product is needed."""
    M = consistent_nodal_mass(nodes, elements)
    m = np.asarray(M.sum(axis=1)).reshape(-1)
    if np.any(m <= 0.0):
        pos = m[m > 0.0]
        fill = float(np.min(pos)) if pos.size else 1.0
        m[m <= 0.0] = fill
    return m


def mass_diag_stacked(mass_nodes: np.ndarray, n_channels: int, n: int, m: int) -> np.ndarray:
    """Flatten-order diagonal weights for (C, N, M) layout (legacy / lumped only)."""
    base = np.asarray(mass_nodes, dtype=np.float64).reshape(n * m)
    return np.tile(base, n_channels)


def mass_kron_channels(M: sp.spmatrix, n_channels: int) -> sp.csr_matrix:
    """Block-diag(M, …, M) for C stacked channels in C-major flatten order."""
    if n_channels == 1:
        return M.tocsr()
    return sp.block_diag([M] * int(n_channels), format="csr")


def apply_mass(M: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    """y = M x for x shaped (D,) or (B, D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return M @ x
    if x.ndim == 2:
        return (M @ x.T).T
    raise ValueError(f"expected 1d/2d, got {x.shape}")
