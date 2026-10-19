"""Geo-FNO airfoil / elasticity data: loaders, published splits, lumped node mass.

Both problems map the mesh coordinates (2 input channels, one value per node, fixed node order)
to a scalar field: airfoil C-mesh (221 x 51 nodes) -> Mach number, elasticity point cloud
(972 nodes) -> stress. Fields are returned as (n_samples, C, H, W) so the repo's AMOR /
PCAReducerMultiChannelMass can be used unchanged; the elasticity cloud is stored as (972, 1).

Splits are the published ones (Li et al. 2023, 1000 train / 200 test). Verified against the
Geo-FNO scripts (elasticity.py, airfoil.py): airfoil test = samples [1000, 1200), elasticity test = the
last 200 of 2000. Samples outside train/test are unused (no validation set, no early stopping).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

N_TRAIN, N_TEST = 1000, 200
AIRFOIL_SHAPE = (221, 51)


def load_airfoil(data_dir: Path):
    """X (N,2,221,51) = (x, y) node coordinates, Y (N,1,221,51) = Mach number (Q channel 4)."""
    d = Path(data_dir) / "Airfoil_dataset"
    x, y, q = (np.load(d / f) for f in ("NACA_Cylinder_X.npy", "NACA_Cylinder_Y.npy", "NACA_Cylinder_Q.npy"))
    X = np.stack([x, y], axis=1).astype(np.float32)
    Y = q[:, 4:5].astype(np.float32)
    return X, Y


def load_elasticity(data_dir: Path):
    """X (N,2,972,1) = (x, y) node coordinates of each sample's mesh, Y (N,1,972,1) = stress."""
    d = Path(data_dir) / "Elasticity_dataset"
    xy = np.load(d / "Random_UnitCell_XY_10.npy")  # (972, 2, N)
    sigma = np.load(d / "Random_UnitCell_sigma_10.npy")  # (972, N)
    X = xy.transpose(2, 1, 0)[..., None].astype(np.float32)  # (N, 2, 972, 1)
    Y = sigma.T[:, None, :, None].astype(np.float32)  # (N, 1, 972, 1)
    return X, Y


def published_split(name: str, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    train = np.arange(N_TRAIN)
    test = np.arange(N_TRAIN, N_TRAIN + N_TEST) if name == "airfoil" else np.arange(n_total - N_TEST, n_total)
    return train, test


def load(name: str, data_dir: Path):
    X, Y = load_airfoil(data_dir) if name == "airfoil" else load_elasticity(data_dir)
    return X, Y, *published_split(name, len(X))


# ----------------------------------------------------------------- lumped node mass
def quad_mesh_node_mass(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lumped (dual-cell) node areas of a structured quad mesh given as (H, W) coordinate arrays.

    Each quad's area (shoelace) is split equally among its four corners; node masses sum to the mesh area.
    """
    x0, x1, x2, x3 = x[:-1, :-1], x[:-1, 1:], x[1:, 1:], x[1:, :-1]
    y0, y1, y2, y3 = y[:-1, :-1], y[:-1, 1:], y[1:, 1:], y[1:, :-1]
    area = 0.5 * np.abs((x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3))
    m = np.zeros(x.shape)
    for r, c in [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)),
                 (slice(1, None), slice(1, None)), (slice(1, None), slice(0, -1))]:
        m[r, c] += area / 4.0
    return m.reshape(-1)


def point_cloud_node_mass(points: np.ndarray, max_edge_quantile: float = 0.99) -> np.ndarray:
    """Lumped P1 node areas of a 2D point cloud: Delaunay triangles contribute area/3 to each vertex.

    Triangles whose longest edge exceeds the given quantile are dropped: Delaunay spans holes and
    concavities with slivers that would put spurious mass on their boundary.
    """
    pts = np.asarray(points, dtype=np.float64)
    simp = Delaunay(pts).simplices
    p = pts[simp]  # (n_tri, 3, 2)
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    edge = np.max([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)], axis=0)
    keep = edge <= np.quantile(edge, max_edge_quantile)
    m = np.zeros(len(pts))
    for v in range(3):
        np.add.at(m, simp[keep, v], area[keep] / 3.0)
    pos = m[m > 0]
    m[m <= 0] = pos.min() if len(pos) else 1.0  # nodes left without any triangle
    return m


def node_mass(name: str, X_train: np.ndarray) -> np.ndarray:
    """Lumped node mass on the mean training mesh (the basis is one fixed object, so one fixed M)."""
    mean_mesh = X_train.mean(axis=0)  # (2, H, W) physical coordinates
    if name == "airfoil":
        return np.maximum(quad_mesh_node_mass(mean_mesh[0], mean_mesh[1]), 1e-12)
    return point_cloud_node_mass(mean_mesh[:, :, 0].T)
