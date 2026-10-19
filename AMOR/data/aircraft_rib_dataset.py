"""Load bcfield NPZ and build nodal field tensors (E, load_xyz) -> u_xyz."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def split_indices(
    n: int, seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same split as train_amor_aircraft_rib.py --dataset bcfield (seed, 70/20/10)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(train_frac * n))
    n_val = max(0, min(n - n_train, int(val_frac * n)))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def normalize_per_channel(
    x: np.ndarray, y: np.ndarray, tr_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Per-channel z-score from train indices only; shape (1,C,1) stats."""
    x_mu = x[tr_idx].mean(axis=(0, 2), keepdims=True)
    x_std = x[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    y_mu = y[tr_idx].mean(axis=(0, 2), keepdims=True)
    y_std = y[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    stats = {
        "x_mu": x_mu.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mu": y_mu.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
    return (x - x_mu) / x_std, (y - y_mu) / y_std, stats


_MESH_KEYS = frozenset(
    {
        "nodes",
        "elements",
        "boundary_node_idx",
        "boundary_facet_tris",
        "boundary_facet_area",
        "dirichlet_tris",
        "neumann_tris",
        "completed_count",
        "n_samples_requested",
    }
)


def load_bcfield_dataset(path: str, *, use_solve_ok_only: bool = False) -> Tuple[Dict[str, np.ndarray], int]:
    d0 = np.load(path, allow_pickle=True)
    n_alloc = int(np.asarray(d0["u"]).shape[0]) if "u" in d0.files else 0
    if "completed_count" in d0.files:
        n_use = int(np.asarray(d0["completed_count"]).reshape(-1)[0])
    else:
        n_use = n_alloc
    n_req = (
        int(np.asarray(d0["n_samples_requested"]).reshape(-1)[0])
        if "n_samples_requested" in d0.files
        else n_alloc
    )
    n_use = min(n_use, n_req)

    def _take(k: str, arr: np.ndarray, sel: np.ndarray | slice) -> np.ndarray:
        if k in _MESH_KEYS or arr.ndim == 0:
            return arr
        if arr.shape and arr.shape[0] in (n_alloc, n_req):
            return arr[sel]
        return arr

    if use_solve_ok_only and "solve_ok" in d0.files:
        ok = np.asarray(d0["solve_ok"], dtype=bool).reshape(-1)[:n_use]
        idx = np.where(ok)[0]
        d = {k: _take(k, np.asarray(d0[k]), idx) for k in d0.files}
        n_use = int(idx.shape[0])
    else:
        # traction_facet and u_magnitude are not used by training and are tens of GB.
        skip = {"traction_facet", "u_magnitude"}
        d = {k: _take(k, np.asarray(d0[k]), slice(0, n_use)) for k in d0.files if k not in skip}
    return d, n_use


def scatter_boundary_load_field(
    load_field_boundary: np.ndarray,
    boundary_node_idx: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Map (nb,3) boundary load vectors to nodal (N,3) on full mesh (zeros elsewhere)."""
    nb = int(boundary_node_idx.shape[0])
    if load_field_boundary.shape[0] != nb:
        raise ValueError(
            f"load_field_boundary rows {load_field_boundary.shape[0]} != boundary nodes {nb}"
        )
    full = np.zeros((n_nodes, 3), dtype=np.float32)
    bni = np.asarray(boundary_node_idx, dtype=np.int64).reshape(-1)
    lf = np.asarray(load_field_boundary, dtype=np.float32).reshape(nb, 3)
    full[bni] = lf
    return full


def scatter_boundary_load_magnitude(
    load_field_boundary: np.ndarray,
    boundary_node_idx: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Nodal |load| from boundary load_field_boundary (for plots / diagnostics)."""
    full = scatter_boundary_load_field(load_field_boundary, boundary_node_idx, n_nodes)
    return np.linalg.norm(full, axis=1).astype(np.float32)


def build_field_tensors(d: Dict[str, np.ndarray], n_use: int) -> Tuple[np.ndarray, np.ndarray]:
    if "E_nodal" not in d:
        raise ValueError("Dataset missing required key E_nodal")
    if "load_field_boundary" not in d or "boundary_node_idx" not in d:
        raise ValueError("Dataset missing load_field_boundary or boundary_node_idx")
    if "u" not in d:
        raise ValueError("Dataset missing u (nodal displacement, S,N,3)")

    e = np.asarray(d["E_nodal"], dtype=np.float32)[:n_use]
    u = np.asarray(d["u"], dtype=np.float32)[:n_use]
    if u.ndim != 3 or u.shape[2] != 3:
        raise ValueError(f"Expected u shape (S,N,3), got {u.shape}")
    bni = np.asarray(d["boundary_node_idx"], dtype=np.int64)
    n_nodes = int(e.shape[1])
    load_xyz = np.zeros((n_use, 3, n_nodes), dtype=np.float32)
    for i in range(n_use):
        load_xyz[i] = scatter_boundary_load_field(
            np.asarray(d["load_field_boundary"][i], dtype=np.float32),
            bni,
            n_nodes,
        ).T  # (3,N)
    x = np.concatenate([e[:, None, :], load_xyz], axis=1)  # (S,4,N): E + load_x,y,z
    y = np.transpose(u, (0, 2, 1)).astype(np.float32)  # (S,3,N): u_x,u_y,u_z
    return x, y
