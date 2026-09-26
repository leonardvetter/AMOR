"""Lumped P1 tetrahedral nodal mass (diagonal M weights)."""

from __future__ import annotations

import numpy as np


def lumped_nodal_mass(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n = nodes.shape[0]
    m = np.zeros(n, dtype=np.float64)
    t = nodes[elements]
    v1 = t[:, 1] - t[:, 0]
    v2 = t[:, 2] - t[:, 0]
    v3 = t[:, 3] - t[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(v1, v2), v3)) / 6.0
    contrib = vol / 4.0
    for j in range(4):
        np.add.at(m, elements[:, j], contrib)
    if np.any(m <= 0.0):
        pos = m[m > 0.0]
        fill = float(np.min(pos)) if pos.size else 1.0
        m[m <= 0.0] = fill
    return m


def mass_diag_stacked(mass_nodes: np.ndarray, n_channels: int, n: int, m: int) -> np.ndarray:
    """Flatten-order weights for (C, N, M) with C-order reshape to (C*N*M,)."""
    base = mass_nodes.reshape(n * m)
    return np.tile(base, n_channels).astype(np.float64)
