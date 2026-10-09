"""Mesh / irregular-grid spatial bases for AMOR (graph Laplacian + scattered polynomials)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev, legendre
from scipy.special import eval_jacobi
from sklearn.decomposition import PCA

from .PCAReducer import (
    PCAReducerMultiChannel,
    _channelize_rows,
    _flatten_samples,
    _orthonormal_rows,
    _pca_like_from_rows,
    _random_orthonormal_rows,
)


def _normalize_nodes(nodes: np.ndarray) -> np.ndarray:
    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return (2.0 * (nodes - lo) / span - 1.0).astype(np.float64)


def _poly_1d(family: str, deg: int, xs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if family == "chebyshev":
        return chebyshev.Chebyshev.basis(deg)(xs)
    if family == "legendre":
        return legendre.Legendre.basis(deg)(xs)
    if family == "jacobi":
        return eval_jacobi(deg, alpha, beta, xs).astype(np.float64)
    raise ValueError(f"Unknown polynomial family {family!r}")


def graph_laplacian_from_tets(nodes: np.ndarray, elements: np.ndarray) -> sp.csr_matrix:
    """Symmetric graph Laplacian from tet mesh (uniform edge weights)."""
    n = len(nodes)
    rows, cols, vals = [], [], []
    for tet in np.asarray(elements, dtype=np.int64):
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = int(tet[a]), int(tet[b])
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([1.0, 1.0])
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return sp.diags(deg) - adj


def mesh_laplacian_eigen_rows(
    nodes: np.ndarray,
    elements: np.ndarray,
    k: int,
    mass_nodes: np.ndarray | None = None,
    mass_matrix=None,
) -> np.ndarray:
    """Lowest-frequency mesh Laplacian modes as rows (k, N).

    Prefer ``mass_matrix`` (sparse SPD) or consistent P1 from the mesh.
    Solves ``L φ = λ M φ`` (M-orthonormal) or the Euclidean problem when no mass.
    """
    from .mass.mesh_mass import consistent_nodal_mass

    L = graph_laplacian_from_tets(nodes, elements)
    n_eigs = min(k, L.shape[0] - 2)
    M = None
    if mass_matrix is not None:
        M = sp.csr_matrix(mass_matrix, dtype=np.float64)
    elif mass_nodes is not None:
        # Consistent P1 only (mass_nodes is a legacy presence flag).
        M = consistent_nodal_mass(nodes, elements)
        _ = mass_nodes
    if M is not None:
        if M.shape[0] != L.shape[0]:
            raise ValueError(f"mass shape {M.shape} != n_nodes {L.shape[0]}")
        _, evecs = spla.eigsh(L, k=n_eigs, M=M, which="SM")
    else:
        _, evecs = spla.eigsh(L, k=n_eigs, which="SM")
    rows = evecs.T.astype(np.float32)
    if rows.shape[0] < k:
        pad = _random_orthonormal_rows(len(nodes), k - rows.shape[0], random_state=0)
        rows = np.concatenate([rows, pad], axis=0)
    return rows[:k]


def mesh_scattered_poly_rows(
    nodes: np.ndarray,
    k: int,
    family: str,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Total-degree 3D tensor polynomials evaluated at mesh nodes, orthonormal rows (k, N)."""
    xyz = _normalize_nodes(nodes)
    max_deg = int(np.ceil(k ** (1.0 / 3.0))) + 2
    modes = []
    for total in range(max_deg + 1):
        for i in range(total + 1):
            for j in range(total + 1 - i):
                l = total - i - j
                fx = _poly_1d(family, i, xyz[:, 0], alpha, beta)
                fy = _poly_1d(family, j, xyz[:, 1], alpha, beta)
                fz = _poly_1d(family, l, xyz[:, 2], alpha, beta)
                modes.append((fx * fy * fz).astype(np.float32))
                if len(modes) >= k:
                    break
            if len(modes) >= k:
                break
        if len(modes) >= k:
            break
    mat = np.stack(modes, axis=0)
    return _orthonormal_rows(mat, k)


def _fit_mesh_side(
    matrix: np.ndarray,
    mean: np.ndarray,
    basis_type: str,
    k_max: int,
    n_samples: int,
    random_state,
    nodes: np.ndarray,
    elements: np.ndarray,
    n_channels: int,
    jacobi_alpha: float,
    jacobi_beta: float,
    mass_nodes: np.ndarray | None = None,
) -> PCA:
    n_nodes = nodes.shape[0]
    if basis_type == "pca":
        solver = "randomized" if k_max < min(matrix.shape) else "auto"
        return PCA(n_components=k_max, svd_solver=solver, random_state=random_state).fit(matrix)
    p_needed = int(np.ceil(k_max / max(1, n_channels)))
    if basis_type == "laplacian":
        rows_1 = mesh_laplacian_eigen_rows(
            nodes, elements, p_needed, mass_nodes=mass_nodes,
        )
    elif basis_type in ("chebyshev", "legendre", "jacobi"):
        rows_1 = mesh_scattered_poly_rows(nodes, p_needed, basis_type, jacobi_alpha, jacobi_beta)
    else:
        raise ValueError(f"Unknown mesh basis_type={basis_type!r}")
    rows = _channelize_rows(rows_1, n_channels, k_max)
    return _pca_like_from_rows(mean.astype(np.float64), rows, n_samples)


def fit_mesh_alternative_basis(
    samples: np.ndarray,
    solution: np.ndarray,
    basis_type: str,
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    random_state=None,
    jacobi_alpha: float = 1.0,
    jacobi_beta: float = 1.0,
    n_components_in: int | None = None,
    n_components_out: int | None = None,
) -> PCAReducerMultiChannel:
    """Fit mesh bases on (S, C, N, 1) tensors."""
    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input/output spatial dims must match")
    y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

    rb = PCAReducerMultiChannel(random_state=random_state)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)

    x_mean = x_in.mean(axis=0)
    y_mean = y_out.mean(axis=0)

    def _cap(k, feat):
        hi = min(num_samples, feat)
        if k is None:
            return hi
        return int(max(1, min(k, hi)))

    k_in = _cap(n_components_in, x_in.shape[1])
    k_out = _cap(n_components_out, y_out.shape[1])

    rb.pca_in = _fit_mesh_side(
        x_in, x_mean, basis_type, k_in, num_samples, random_state,
        nodes, elements, c_in, jacobi_alpha, jacobi_beta,
    )
    rb.pca_out = _fit_mesh_side(
        y_out, y_mean, basis_type, k_out, num_samples, random_state,
        nodes, elements, c_out, jacobi_alpha, jacobi_beta,
    )
    rb.basis_type = basis_type
    rb.output_on_increment = False
    return rb


def fit_mesh_alternative_basis_mass(
    samples: np.ndarray,
    solution: np.ndarray,
    basis_type: str,
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    mass_nodes: np.ndarray,
    random_state=None,
    jacobi_alpha: float = 1.0,
    jacobi_beta: float = 1.0,
    n_components_in: int | None = None,
    n_components_out: int | None = None,
):
    """Mesh alternative bases compatible with PCAReducerMultiChannelMass."""
    import scipy.sparse as sp
    from .mass.mesh_mass import consistent_nodal_mass, mass_kron_channels
    from .PCAReducer import PCAReducerMultiChannelMass, pca_like_from_rows_mass

    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input/output spatial dims must match")
    y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

    M_nodes = consistent_nodal_mass(nodes, elements)
    mass_nodes = np.asarray(M_nodes.diagonal(), dtype=np.float64)
    M_in = mass_kron_channels(M_nodes, c_in)
    M_out = mass_kron_channels(M_nodes, c_out)

    rb = PCAReducerMultiChannelMass(random_state=random_state)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)
    rb.mass_nodes_ = mass_nodes.astype(np.float32)
    rb.mass_in_ = M_in
    rb.mass_out_ = M_out
    rb.mass_diag_in_ = np.asarray(M_in.sum(axis=1)).reshape(-1).astype(np.float32)
    rb.mass_diag_out_ = np.asarray(M_out.sum(axis=1)).reshape(-1).astype(np.float32)

    def _cap(k, feat):
        hi = min(num_samples, feat)
        if k is None:
            return hi
        return int(max(1, min(k, hi)))

    k_in = _cap(n_components_in, x_in.shape[1])
    k_out = _cap(n_components_out, y_out.shape[1])

    if basis_type == "pca":
        return PCAReducerMultiChannelMass(
            svd_solver="randomized", random_state=random_state,
        ).fit(samples, solution, k_in=k_in, k_out=max(k_in, k_out),
              nodes=nodes, elements=elements)

    # Laplacian GEVP uses consistent P1 mass; polynomials remassed via Gram below.
    mass_for_lap = mass_nodes if basis_type == "laplacian" else None
    rb.pca_in = _fit_mesh_side(
        x_in, x_in.mean(0), basis_type, k_in, num_samples, random_state,
        nodes, elements, c_in, jacobi_alpha, jacobi_beta,
        mass_nodes=mass_for_lap,
    )
    rb.pca_out = _fit_mesh_side(
        y_out, y_out.mean(0), basis_type, k_out, num_samples, random_state,
        nodes, elements, c_out, jacobi_alpha, jacobi_beta,
        mass_nodes=mass_for_lap,
    )
    rb.pca_in = pca_like_from_rows_mass(
        x_in.mean(0), rb.pca_in.components_, M_in, num_samples,
    )
    rb.pca_out = pca_like_from_rows_mass(
        y_out.mean(0), rb.pca_out.components_, M_out, num_samples,
    )
    rb.basis_type = basis_type
    return rb
