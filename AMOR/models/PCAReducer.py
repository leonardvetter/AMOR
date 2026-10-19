"""PCA reducers for multi-channel grid and mesh fields, including mass-weighted PCA.

Paths to arrays are supplied by the caller. This module does not open a default data directory.
"""
import json
import os
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

from .bases import (
    BASIS_TYPES,
    channelize_rows,
    dmd_mode_rows,
    flatten_samples,
    laplacian_eigen_rows,
    orthonormal_rows,
    pca_like_from_rows,
    polynomial_rows,
    random_orthonormal_rows,
)
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import eval_jacobi
from numpy.polynomial import chebyshev, legendre



# Back-compat aliases (bases live in models/bases/)
_flatten_samples = flatten_samples
_orthonormal_rows = orthonormal_rows
_pca_like_from_rows = pca_like_from_rows
_channelize_rows = channelize_rows
_dmd_mode_rows = dmd_mode_rows
_laplacian_eigen_rows = laplacian_eigen_rows
_polynomial_rows = polynomial_rows
_random_orthonormal_rows = random_orthonormal_rows

class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors with optional boundary weighting."""

    def __init__(
        self,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
        bc_weight: float = 1.0,
        boundary_threshold: float = 0.1,
        use_boundary_weighting: bool = False,
    ):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.bc_weight = bc_weight
        self.boundary_threshold = boundary_threshold
        self.use_boundary_weighting = use_boundary_weighting
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels: Optional[int] = None
        self.out_channels: Optional[int] = None
        self.shape_: Optional[tuple] = None
        self.output_shape_: Optional[tuple] = None
        self.basis_type: Optional[str] = None
        self.output_on_increment: bool = False

    def fit(
        self,
        samples: np.ndarray,
        solution: np.ndarray,
        n_components_in: Optional[int] = None,
        n_components_out: Optional[int] = None,
    ) -> "PCAReducerMultiChannel":
        """Fit input/output PCA bases.

        n_components_in / n_components_out: if given, fit and keep ONLY that many
        leading modes instead of the full basis. The AMOR/PCANet only ever use
        the k_in/k_out leading modes, so truncating here avoids computing and
        storing the full (D x D) basis (huge for fine grids), cutting fit time
        and memory. Pass k_in/k_out to save exactly the modes the model needs.
        """
        if samples.ndim < 3:
            raise ValueError(f"Expected (num_samples, C_in, *spatial), got {samples.shape}")
        num_samples, c_in = samples.shape[0], samples.shape[1]
        spatial = tuple(int(x) for x in samples.shape[2:])
        self.n_channels = c_in
        self.shape_ = spatial
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)

        # (S, *spatial) has no channel axis -> promote to (S, 1, *spatial).
        # Do NOT use 2+len(spatial): that also matches multi-channel (S, C, *spatial).
        if solution.ndim == 1 + len(spatial):
            solution = solution.reshape(num_samples, 1, *spatial)
        if solution.ndim != 2 + len(spatial):
            raise ValueError(
                f"Expected (num_samples, C_out, *spatial) or (num_samples, *spatial), got {solution.shape}"
            )
        _, c_out = solution.shape[0], solution.shape[1]
        if tuple(int(x) for x in solution.shape[2:]) != spatial:
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        self.output_shape_ = (c_out, *spatial)
        y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        if self.use_boundary_weighting:
            if len(spatial) != 2:
                raise ValueError("use_boundary_weighting requires 2D spatial grids")
            n, m = spatial
            boundary_weights = self._compute_boundary_weights(n, m)
            y_weighted = y_out * boundary_weights.reshape(1, -1)
        else:
            y_weighted = y_out

        # Cap requested modes by the feasible number of components (n_samples, n_features).
        def _cap(k, n_samp, n_feat):
            if k is None:
                return None
            return int(max(1, min(k, n_samp, n_feat)))

        k_in_fit = _cap(n_components_in, x_in.shape[0], x_in.shape[1])
        k_out_fit = _cap(n_components_out, y_weighted.shape[0], y_weighted.shape[1])
        # randomized solver is required (and far cheaper) when truncating a fine grid.
        in_solver = "randomized" if (k_in_fit is not None and self.svd_solver == "auto") else self.svd_solver
        out_solver = "randomized" if (k_out_fit is not None and self.svd_solver == "auto") else self.svd_solver

        self.pca_in = PCA(
            n_components=k_in_fit,
            svd_solver=in_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(x_in)
        self.pca_out = PCA(
            n_components=k_out_fit,
            svd_solver=out_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(y_weighted)

        return self

    def fit_spatial(
        self,
        samples: np.ndarray,
        solution: np.ndarray,
        n_components_in: Optional[int] = None,
        n_components_out: Optional[int] = None,
    ) -> "PCAReducerMultiChannel":
        """Fit shared spatial PCA (size N·M), stacking channels as extra samples."""
        if samples.ndim < 3:
            raise ValueError(f"Expected (num_samples, C_in, *spatial), got {samples.shape}")
        num_samples, c_in = samples.shape[0], samples.shape[1]
        spatial = tuple(int(x) for x in samples.shape[2:])
        spat = int(np.prod(spatial))
        self.n_channels = c_in
        self.shape_ = spatial
        x_in = samples.reshape(num_samples, c_in, spat)
        x_in = np.ascontiguousarray(
            x_in.transpose(1, 0, 2).reshape(num_samples * c_in, spat), dtype=np.float32
        )

        if solution.ndim == 1 + len(spatial):
            solution = solution.reshape(num_samples, 1, *spatial)
        if solution.ndim != 2 + len(spatial):
            raise ValueError(
                f"Expected (num_samples, C_out, *spatial) or (num_samples, *spatial), got {solution.shape}"
            )
        _, c_out = solution.shape[0], solution.shape[1]
        if tuple(int(x) for x in solution.shape[2:]) != spatial:
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        self.output_shape_ = (c_out, *spatial)
        y_out = solution.reshape(num_samples, c_out, spat)
        y_out = np.ascontiguousarray(
            y_out.transpose(1, 0, 2).reshape(num_samples * c_out, spat), dtype=np.float32
        )

        def _cap(k, n_samp, n_feat):
            if k is None:
                return None
            return int(max(1, min(k, n_samp, n_feat)))

        k_in_fit = _cap(n_components_in, x_in.shape[0], x_in.shape[1])
        k_out_fit = _cap(n_components_out, y_out.shape[0], y_out.shape[1])
        in_solver = "randomized" if (k_in_fit is not None and self.svd_solver == "auto") else self.svd_solver
        out_solver = "randomized" if (k_out_fit is not None and self.svd_solver == "auto") else self.svd_solver

        self.pca_in = PCA(
            n_components=k_in_fit,
            svd_solver=in_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(x_in)
        self.pca_out = PCA(
            n_components=k_out_fit,
            svd_solver=out_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(y_out)
        return self

    def _compute_boundary_weights(self, n: int, m: int) -> np.ndarray:
        weights = np.ones((n, m), dtype=np.float32)
        x_coords = np.linspace(0, 1, n)
        y_coords = np.linspace(0, 1, m)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")
        dist_to_boundary = np.minimum(
            np.minimum(grid_x, 1 - grid_x),
            np.minimum(grid_y, 1 - grid_y),
        )
        boundary_mask = dist_to_boundary < self.boundary_threshold
        weights[boundary_mask] = self.bc_weight
        return weights.flatten()

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    def truncate(self, k_in: int, k_out: int) -> "PCAReducerMultiChannel":
        """Keep only the first k_in / k_out PCA modes; discard the rest in place."""
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before truncate")
        k_in = int(min(k_in, self.pca_in.components_.shape[0]))
        k_out = int(min(k_out, self.pca_out.components_.shape[0]))
        if k_in < 1 or k_out < 1:
            raise ValueError(f"k_in={k_in}, k_out={k_out} must be >= 1")
        self.pca_in.components_ = np.ascontiguousarray(self.pca_in.components_[:k_in])
        self.pca_in.explained_variance_ratio_ = self.pca_in.explained_variance_ratio_[:k_in]
        self.pca_in.n_components_ = k_in
        self.pca_out.components_ = np.ascontiguousarray(self.pca_out.components_[:k_out])
        self.pca_out.explained_variance_ratio_ = self.pca_out.explained_variance_ratio_[:k_out]
        self.pca_out.n_components_ = k_out
        return self

    @staticmethod
    def _pca_from_arrays(
        mean: np.ndarray,
        components: np.ndarray,
        *,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
        n_samples: int,
    ) -> PCA:
        mean = np.asarray(mean, dtype=np.float64)
        components = np.asarray(components, dtype=np.float64)
        pca = PCA(n_components=components.shape[0])
        pca.mean_ = mean
        pca.components_ = components
        pca.explained_variance_ = np.asarray(explained_variance, dtype=np.float64)
        pca.explained_variance_ratio_ = np.asarray(explained_variance_ratio, dtype=np.float64)
        pca.n_components_ = components.shape[0]
        pca.n_features_in_ = mean.shape[0]
        pca.n_samples_ = n_samples
        return pca

    @property
    def input_components(self) -> np.ndarray:
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_

    def save_training_bundle(
        self,
        path: str,
        *,
        normalization: dict,
        film_norm: dict,
        meta: dict,
    ) -> None:
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before save_training_bundle")
        out_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(
            out_path,
            n_channels=np.array([self.n_channels], dtype=np.int64),
            out_channels=np.array([self.out_channels], dtype=np.int64),
            shape_=np.asarray(self.shape_, dtype=np.int64),
            pca_in_mean=self.pca_in.mean_.astype(np.float64),
            pca_in_components=self.pca_in.components_.astype(np.float64),
            pca_in_evr=self.pca_in.explained_variance_ratio_.astype(np.float64),
            pca_in_ev=np.asarray(self.pca_in.explained_variance_, dtype=np.float64),
            pca_out_mean=self.pca_out.mean_.astype(np.float64),
            pca_out_components=self.pca_out.components_.astype(np.float64),
            pca_out_evr=self.pca_out.explained_variance_ratio_.astype(np.float64),
            pca_out_ev=np.asarray(self.pca_out.explained_variance_, dtype=np.float64),
            x_mu=np.asarray(normalization["x_mu"], dtype=np.float32),
            x_std=np.asarray(normalization["x_std"], dtype=np.float32),
            y_mu=np.asarray(normalization["y_mu"], dtype=np.float32),
            y_std=np.asarray(normalization["y_std"], dtype=np.float32),
            nu_mu=np.array([float(film_norm["nu_mu"])], dtype=np.float64),
            nu_std=np.array([float(film_norm["nu_std"])], dtype=np.float64),
            case_mu=np.array([float(film_norm["case_mu"])], dtype=np.float64),
            case_std=np.array([float(film_norm["case_std"])], dtype=np.float64),
            meta_json=np.array([json.dumps(meta)], dtype=object),
        )

    @classmethod
    def load_training_bundle(cls, path: str):
        data = np.load(path, allow_pickle=True)
        rb = cls()
        rb.n_channels = int(np.asarray(data["n_channels"]).reshape(-1)[0])
        rb.out_channels = int(np.asarray(data["out_channels"]).reshape(-1)[0])
        shape = tuple(int(x) for x in np.asarray(data["shape_"]).reshape(-1))
        rb.shape_ = shape
        rb.output_shape_ = (rb.out_channels, *shape)
        meta = json.loads(str(np.asarray(data["meta_json"]).reshape(-1)[0]))
        n_train = int(meta["n_train"])

        def _load_pca(prefix: str) -> PCA:
            ev = data[f"{prefix}_ev"] if f"{prefix}_ev" in data.files else None
            if ev is None:
                ev = np.ones(int(data[f"{prefix}_components"].shape[0]), dtype=np.float64)
            return cls._pca_from_arrays(
                np.asarray(data[f"{prefix}_mean"]),
                np.asarray(data[f"{prefix}_components"]),
                explained_variance=np.asarray(ev, dtype=np.float64),
                explained_variance_ratio=np.asarray(data[f"{prefix}_evr"], dtype=np.float64),
                n_samples=n_train,
            )

        rb.pca_in = _load_pca("pca_in")
        rb.pca_out = _load_pca("pca_out")
        stats = {
            "x_mu": np.asarray(data["x_mu"]),
            "x_std": np.asarray(data["x_std"]),
            "y_mu": np.asarray(data["y_mu"]),
            "y_std": np.asarray(data["y_std"]),
        }
        film_norm = {
            "nu_mu": float(np.asarray(data["nu_mu"]).reshape(-1)[0]),
            "nu_std": float(np.asarray(data["nu_std"]).reshape(-1)[0]),
            "case_mu": float(np.asarray(data["case_mu"]).reshape(-1)[0]),
            "case_std": float(np.asarray(data["case_std"]).reshape(-1)[0]),
        }
        return rb, stats, film_norm, meta


# BASIS_TYPES and spatial-basis builders: see models/bases/

def _fit_side_pca(
    matrix: np.ndarray,
    mean: np.ndarray,
    basis_type: str,
    k_max: int,
    n_samples: int,
    random_state: Optional[int],
    *,
    dmd_snapshots_next: Optional[np.ndarray] = None,
    grid_shape: tuple[int, int],
    laplacian_periodic: bool,
    jacobi_alpha: float,
    jacobi_beta: float,
    dmd_svd_rank: Optional[int] = None,
    n_channels: int = 1,
) -> PCA:
    n, m = grid_shape
    d = matrix.shape[1]
    spatial = n * m

    if basis_type == "pca":
        # Only build/store the k_max leading modes (randomized when truncating a fine grid).
        solver = "randomized" if k_max < min(matrix.shape) else "auto"
        return PCA(n_components=k_max, svd_solver=solver, random_state=random_state).fit(matrix)

    if d != n_channels * spatial:
        raise ValueError(
            f"{basis_type} basis expects flatten length n_channels*n*m="
            f"{n_channels}*{spatial}={n_channels*spatial}, got {d}"
        )

    # Build enough single-channel spatial modes, then lift to the c-channel field.
    p_needed = int(np.ceil(k_max / max(1, n_channels)))
    if basis_type in ("chebyshev", "legendre", "jacobi"):
        rows_1ch = _polynomial_rows(n, m, p_needed, basis_type, jacobi_alpha, jacobi_beta)
    elif basis_type == "laplacian":
        rows_1ch = _laplacian_eigen_rows(n, m, p_needed, laplacian_periodic)
    else:
        raise ValueError(f"Unknown basis_type={basis_type!r}; expected one of {BASIS_TYPES}")

    rows = _channelize_rows(rows_1ch, n_channels, k_max)
    return _pca_like_from_rows(mean.astype(np.float64), rows, n_samples)


def fit_alternative_basis(
    samples: np.ndarray,
    solution: np.ndarray,
    basis_type: str = "pca",
    *,
    dmd_input_next: Optional[np.ndarray] = None,
    dmd_output_next: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    laplacian_periodic: bool = True,
    jacobi_alpha: float = 1.0,
    jacobi_beta: float = 1.0,
    dmd_svd_rank: Optional[int] = None,
    bc_weight: float = 1.0,
    boundary_threshold: float = 0.1,
    use_boundary_weighting: bool = False,
    n_components_in: Optional[int] = None,
    n_components_out: Optional[int] = None,
) -> PCAReducerMultiChannel:
    """Fit PCAReducerMultiChannel with non-PCA spatial bases (AMOR-compatible).

    n_components_in / n_components_out cap how many modes are built and stored so
    only the k_in / k_out modes the model consumes are materialized (no full DxD
    basis, no CPU-heavy eig on thousands of modes).
    """
    if basis_type not in BASIS_TYPES:
        raise ValueError(f"basis_type must be one of {BASIS_TYPES}, got {basis_type!r}")

    rb = PCAReducerMultiChannel(
        random_state=random_state,
        bc_weight=bc_weight,
        boundary_threshold=boundary_threshold,
        use_boundary_weighting=use_boundary_weighting,
    )

    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    if solution.ndim != 4:
        raise ValueError(f"Expected solution (N, C, N, N), got {solution.shape}")
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input and output spatial dimensions must match")
    y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)

    if use_boundary_weighting:
        weights = rb._compute_boundary_weights(n, m)
        y_fit = y_out * weights.reshape(1, -1)
    else:
        y_fit = y_out

    x_mean = x_in.mean(axis=0)
    y_mean = y_fit.mean(axis=0)

    def _cap(k, feat):
        hi = min(num_samples, feat)
        if k is None:
            return hi
        return int(max(1, min(k, hi)))

    k_in_max = _cap(n_components_in, x_in.shape[1])
    k_out_max = _cap(n_components_out, y_fit.shape[1])

    if basis_type == "pca":
        in_solver = "randomized" if k_in_max < min(x_in.shape) else "auto"
        out_solver = "randomized" if k_out_max < min(y_fit.shape) else "auto"
        rb.pca_in = PCA(n_components=k_in_max, svd_solver=in_solver, random_state=random_state).fit(x_in)
        rb.pca_out = PCA(n_components=k_out_max, svd_solver=out_solver, random_state=random_state).fit(y_fit)
        rb.basis_type = "pca"
        return rb

    rb.pca_in = _fit_side_pca(
        x_in,
        x_mean,
        basis_type,
        k_in_max,
        num_samples,
        random_state,
        dmd_snapshots_next=dmd_input_next,
        grid_shape=(n, m),
        laplacian_periodic=laplacian_periodic,
        jacobi_alpha=jacobi_alpha,
        jacobi_beta=jacobi_beta,
        dmd_svd_rank=dmd_svd_rank,
        n_channels=c_in,
    )
    rb.pca_out = _fit_side_pca(
        y_fit,
        y_mean,
        basis_type,
        k_out_max,
        num_samples,
        random_state,
        dmd_snapshots_next=dmd_output_next,
        grid_shape=(n, m),
        laplacian_periodic=laplacian_periodic,
        jacobi_alpha=jacobi_alpha,
        jacobi_beta=jacobi_beta,
        dmd_svd_rank=dmd_svd_rank,
        n_channels=c_out,
    )
    rb.basis_type = basis_type
    rb.output_on_increment = True
    return rb


def fit_random_init_basis(
    samples: np.ndarray,
    solution: np.ndarray,
    *,
    random_state: Optional[int] = None,
    bc_weight: float = 1.0,
    boundary_threshold: float = 0.1,
    use_boundary_weighting: bool = False,
    n_components_in: Optional[int] = None,
    n_components_out: Optional[int] = None,
) -> PCAReducerMultiChannel:
    """Random orthonormal spatial init for trainable learned AMOR bases."""
    rb = PCAReducerMultiChannel(
        random_state=random_state,
        bc_weight=bc_weight,
        boundary_threshold=boundary_threshold,
        use_boundary_weighting=use_boundary_weighting,
    )
    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    if solution.ndim != 4:
        raise ValueError(f"Expected solution (N, C, N, N), got {solution.shape}")
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input and output spatial dimensions must match")
    y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)
    if use_boundary_weighting:
        weights = rb._compute_boundary_weights(n, m)
        y_fit = y_out * weights.reshape(1, -1)
    else:
        y_fit = y_out
    x_mean = x_in.mean(axis=0)
    y_mean = y_fit.mean(axis=0)

    def _cap(k, feat):
        hi = min(num_samples, feat)
        if k is None:
            return hi
        return int(max(1, min(k, hi)))

    k_in_max = _cap(n_components_in, x_in.shape[1])
    k_out_max = _cap(n_components_out, y_fit.shape[1])
    rows_in = _random_orthonormal_rows(x_in.shape[1], k_in_max, random_state)
    rows_out = _random_orthonormal_rows(y_fit.shape[1], k_out_max, random_state)
    rb.pca_in = _pca_like_from_rows(x_mean.astype(np.float64), rows_in, num_samples)
    rb.pca_out = _pca_like_from_rows(y_mean.astype(np.float64), rows_out, num_samples)
    rb.output_on_increment = True
    return rb


def fit_dmd_hybrid_basis(
    samples: np.ndarray,
    solution: np.ndarray,
    *,
    dmd_output_next: np.ndarray,
    random_state: Optional[int] = None,
    dmd_svd_rank: Optional[int] = None,
    bc_weight: float = 1.0,
    boundary_threshold: float = 0.1,
    use_boundary_weighting: bool = False,
) -> PCAReducerMultiChannel:
    """PCA input on u(t); DMD output on ordered Δu pairs."""
    rb = PCAReducerMultiChannel(
        random_state=random_state,
        bc_weight=bc_weight,
        boundary_threshold=boundary_threshold,
        use_boundary_weighting=use_boundary_weighting,
    )
    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input and output spatial dimensions must match")

    delta = solution.reshape(num_samples, -1).astype(np.float32, copy=False) - x_in
    y_mean = delta.mean(axis=0)
    k_out_max = min(num_samples, delta.shape[1])

    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)
    rb.pca_in = PCA(n_components=None, svd_solver="auto", random_state=random_state).fit(x_in)
    rank = dmd_svd_rank if dmd_svd_rank is not None else k_out_max
    rows = _dmd_mode_rows(delta, dmd_output_next, k_out_max, svd_rank=rank)
    rb.pca_out = _pca_like_from_rows(y_mean.astype(np.float64), rows, num_samples)
    rb.basis_type = "dmd_hybrid"
    rb.output_on_increment = True
    return rb


def kolmogorov_dmd_aligned_pairs(
    data_path: str,
    train_traj_indices: np.ndarray,
    xy_mean: float,
    xy_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordered DMD pairs with equal length (T-2 steps per trajectory).
    Input: u(t) -> u(t+1). Output increments: Δu(t), Δu(t+1).
    Returns x, x_next, d, d_next each (N, D).
    """
    data = np.load(data_path, mmap_mode="r")
    x_list, xn_list, d_list, dn_list = [], [], [], []
    scale = float(xy_std) + 1e-8
    for traj_idx in train_traj_indices:
        traj = np.array(data[int(traj_idx)], dtype=np.float32)
        for t in range(traj.shape[0] - 2):
            u0 = (traj[t] - xy_mean) / scale
            u1 = (traj[t + 1] - xy_mean) / scale
            u2 = (traj[t + 2] - xy_mean) / scale
            x_list.append(u0.reshape(-1))
            xn_list.append(u1.reshape(-1))
            d_list.append((u1 - u0).reshape(-1))
            dn_list.append((u2 - u1).reshape(-1))
    if not x_list:
        raise ValueError("No DMD pairs: trajectories too short")
    return (
        np.stack(x_list, axis=0),
        np.stack(xn_list, axis=0),
        np.stack(d_list, axis=0),
        np.stack(dn_list, axis=0),
    )


def kolmogorov_train_trajectory_indices(n_trajectories: int, seed: int = 42) -> np.ndarray:
    """Trajectory-level 70% train indices (same seed as pair shuffle script)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trajectories)
    n_train = int(0.7 * n_trajectories)
    return perm[:n_train]

# =============================================================================
# Mass-weighted PCA (rib meshes). Euclidean PCAReducerMultiChannel is above.
# =============================================================================

def _mass_diag_stacked(*args, **kwargs):
    """Lazy import so Euclidean PCAReducer users need not put models/mass on path."""
    from .mass.mesh_mass import mass_diag_stacked as _impl
    return _impl(*args, **kwargs)


# Alias used by mass-PCA code below (was: from mesh_mass import mass_diag_stacked)
mass_diag_stacked = _mass_diag_stacked

class _MassPCAView:
    def __init__(
        self,
        mean_: np.ndarray,
        components_: np.ndarray,
        n_components_: int,
        explained_variance_ratio_: np.ndarray,
        encoder_components_: Optional[np.ndarray] = None,
    ):
        self.mean_ = mean_
        self.components_ = components_
        self.n_components_ = n_components_
        self.explained_variance_ratio_ = explained_variance_ratio_
        # Optional dual encoder rows E (k, D); default encode uses Φᵀ M via mass path.
        self.encoder_components_ = encoder_components_


class PCAReducerMultiChannelMass:
    def __init__(
        self,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
    ):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in: Optional[_MassPCAView] = None
        self.pca_out: Optional[_MassPCAView] = None
        self.n_channels: Optional[int] = None
        self.out_channels: Optional[int] = None
        self.shape_: Optional[tuple] = None
        self.output_shape_: Optional[tuple] = None
        self.mass_diag_in_: Optional[np.ndarray] = None
        self.mass_diag_out_: Optional[np.ndarray] = None
        self.mass_nodes_: Optional[np.ndarray] = None
        self.mass_in_ = None  # scipy.sparse CSR (full stacked), or None
        self.mass_out_ = None

    @staticmethod
    def _fit_mass_pca_sparse(
        X: np.ndarray,
        M,
        *,
        n_components: int,
    ) -> _MassPCAView:
        """Snapshot POD in the M-inner product (M any SPD sparse/dense, not assumed diag)."""
        from .mass.mesh_mass import apply_mass

        n_s, n_f = X.shape
        k = int(min(n_components, n_s, n_f))
        if k < 1:
            raise ValueError(f"n_components={n_components} invalid for data shape {X.shape}")
        mu = X.mean(axis=0)
        Xc = np.asarray(X - mu, dtype=np.float64)
        # G_ij = x_i^T M x_j. Chunked so M X is not a second full copy of X.
        G = np.empty((n_s, n_s), dtype=np.float64)
        chunk = 128
        for i0 in range(0, n_s, chunk):
            sl = slice(i0, min(i0 + chunk, n_s))
            MXi = apply_mass(M, Xc[sl])
            G[sl] = MXi @ Xc.T
        G = 0.5 * (G + G.T)
        evals, evecs = np.linalg.eigh(G)
        idx = np.argsort(evals)[::-1][:k]
        lam = np.maximum(evals[idx], 1e-30)
        U = evecs[:, idx]
        # Φ = Xc^T U / sqrt(λ) ⇒ Φ^T M Φ = I (not M Xc^T U — that would bake M twice).
        Phi = (Xc.T @ U) / np.sqrt(lam)[None, :]
        components = Phi.T.astype(np.float32)
        total = float(np.maximum(evals, 0.0).sum())
        evr = (np.maximum(evals[idx], 0.0) / max(total, 1e-30)).astype(np.float32)
        return _MassPCAView(
            mean_=mu.astype(np.float32),
            components_=components,
            n_components_=int(components.shape[0]),
            explained_variance_ratio_=evr,
        )

    def fit(
        self,
        samples: np.ndarray,
        solution: np.ndarray,
        mass_nodes: np.ndarray = None,
        *,
        k_in: int,
        k_out: int,
        mass_matrix=None,
        nodes: np.ndarray = None,
        elements: np.ndarray = None,
    ) -> "PCAReducerMultiChannelMass":
        """Fit M-PCA with consistent sparse P1 mass (no diagonal-M assumption).

        Provide ``mass_matrix`` (CSR) or ``nodes``+``elements`` to assemble it.
        ``mass_nodes`` is ignored (legacy kwarg kept for call-site compat).
        """
        import scipy.sparse as sp
        from .mass.mesh_mass import (
            apply_mass,
            consistent_nodal_mass,
            mass_kron_channels,
        )

        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, M), got {samples.shape}")
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        x_in = samples.reshape(num_samples, -1).astype(np.float64, copy=False)

        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected solution 4D, got {solution.shape}")
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        self.output_shape_ = (c_out, n, m)
        y_out = solution.reshape(num_samples, -1).astype(np.float64, copy=False)

        n_nodes = n * m
        if mass_matrix is not None:
            M_nodes = sp.csr_matrix(mass_matrix, dtype=np.float64)
            if M_nodes.shape != (n_nodes, n_nodes):
                raise ValueError(f"mass_matrix shape {M_nodes.shape} != {(n_nodes, n_nodes)}")
        elif nodes is not None and elements is not None:
            M_nodes = consistent_nodal_mass(nodes, elements)
            if M_nodes.shape[0] != n_nodes:
                raise ValueError("consistent mass size != N*M")
        else:
            raise ValueError(
                "fit requires mass_matrix or nodes+elements "
                "(consistent P1; lumped mass_nodes fallback removed)"
            )
        _ = mass_nodes  # legacy unused

        self.mass_nodes_ = np.asarray(M_nodes.diagonal(), dtype=np.float32)
        self.mass_in_ = mass_kron_channels(M_nodes, c_in)
        self.mass_out_ = mass_kron_channels(M_nodes, c_out)
        # Legacy diag buffers (row sums) for callers that still read them.
        self.mass_diag_in_ = np.asarray(self.mass_in_.sum(axis=1)).reshape(-1).astype(np.float32)
        self.mass_diag_out_ = np.asarray(self.mass_out_.sum(axis=1)).reshape(-1).astype(np.float32)

        self.pca_in = self._fit_mass_pca_sparse(x_in, self.mass_in_, n_components=int(k_in))
        self.pca_out = self._fit_mass_pca_sparse(y_out, self.mass_out_, n_components=int(k_out))
        # M-adjoint encoders so share_bases / dual encode match z = Φᵀ M (x−μ).
        from .dis_dual import m_adjoint_encoder_rows

        self.pca_in.encoder_components_ = m_adjoint_encoder_rows(
            self.pca_in.components_, self.mass_in_
        ).astype(np.float32)
        self.pca_out.encoder_components_ = m_adjoint_encoder_rows(
            self.pca_out.components_, self.mass_out_
        ).astype(np.float32)
        return self

    def encoder_mass_diag(self, *, use_input_basis: bool, dim: int) -> np.ndarray:
        """Deprecated name: returns row-sum lumping of M (compat). Prefer ``mass_matrix``."""
        if self.mass_diag_in_ is None:
            raise ValueError("PCAReducerMultiChannelMass not fitted")
        base = self.mass_diag_in_ if use_input_basis else self.mass_diag_out_
        if base.shape[0] < dim:
            return np.pad(base, (0, dim - base.shape[0]), constant_values=1.0)
        return base[:dim]

    def mass_matrix(self, *, use_input_basis: bool):
        M = self.mass_in_ if use_input_basis else self.mass_out_
        if M is None:
            raise ValueError("PCAReducerMultiChannelMass has no sparse mass_matrix")
        return M

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    @property
    def input_components(self) -> np.ndarray:
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_


def pca_like_from_rows_mass(
    mean: np.ndarray,
    rows: np.ndarray,
    mass_flat_or_matrix,
    n_samples: int,
) -> _MassPCAView:
    """Build mass-PCA view from mean + row basis (DIS / mesh alternative bases).

    ``mass_flat_or_matrix`` may be a sparse/dense SPD mass matrix or a 1-d
    lumped diagonal (stored as sparse diag). Modes are M-orthonormalized via
    the Gram ``Φ M Φᵀ`` — no diagonal-M assumption in the orthonormalization.
    """
    import scipy.sparse as sp
    from .dis_dual import m_orthonormalize_rows_general

    mean = np.asarray(mean, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    M = mass_flat_or_matrix
    if not sp.issparse(M) and np.ndim(M) == 1:
        M = sp.diags(np.asarray(M, dtype=np.float64).reshape(-1), format="csr")
    components = m_orthonormalize_rows_general(rows, M).astype(np.float32)
    n_comp = int(components.shape[0])
    evr = np.ones(n_comp, dtype=np.float32) / max(1, n_comp)
    return _MassPCAView(
        mean_=mean.astype(np.float32),
        components_=components,
        n_components_=n_comp,
        explained_variance_ratio_=evr,
    )


def fit_random_init_basis_mass(
    samples: np.ndarray,
    solution: np.ndarray,
    mass_nodes: np.ndarray = None,
    *,
    random_state=None,
    n_components_in: int | None = None,
    n_components_out: int | None = None,
    nodes: np.ndarray = None,
    elements: np.ndarray = None,
    mass_matrix=None,
) -> PCAReducerMultiChannelMass:
    """Random M-orthonormal init for trainable AMOR bases on rib meshes."""
    import scipy.sparse as sp
    from .mass.mesh_mass import consistent_nodal_mass, mass_kron_channels

    x_in, num_samples, c_in, n, m = _flatten_samples(samples)
    if solution.ndim == 3:
        solution = solution[:, None, :, :]
    _, c_out, n_out, m_out = solution.shape
    if (n_out, m_out) != (n, m):
        raise ValueError("Input/output spatial dims must match")
    y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

    n_nodes = n * m
    if mass_matrix is not None:
        M_nodes = sp.csr_matrix(mass_matrix, dtype=np.float64)
    elif nodes is not None and elements is not None:
        M_nodes = consistent_nodal_mass(nodes, elements)
    else:
        raise ValueError(
            "fit_random_init_basis_mass needs mass_matrix or nodes+elements "
            "(consistent P1; lumped mass_nodes fallback removed)"
        )
    _ = mass_nodes
    if M_nodes.shape[0] != n_nodes:
        raise ValueError(f"mass size {M_nodes.shape[0]} != N*M={n_nodes}")

    rb = PCAReducerMultiChannelMass(random_state=random_state)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, m)
    rb.output_shape_ = (c_out, n, m)
    rb.mass_nodes_ = np.asarray(M_nodes.diagonal(), dtype=np.float32)
    rb.mass_in_ = mass_kron_channels(M_nodes, c_in)
    rb.mass_out_ = mass_kron_channels(M_nodes, c_out)
    rb.mass_diag_in_ = np.asarray(rb.mass_in_.sum(axis=1)).reshape(-1).astype(np.float32)
    rb.mass_diag_out_ = np.asarray(rb.mass_out_.sum(axis=1)).reshape(-1).astype(np.float32)

    k_in = int(min(n_components_in or x_in.shape[1], x_in.shape[1]))
    k_out = int(min(n_components_out or y_out.shape[1], y_out.shape[1]))
    rows_in = _random_orthonormal_rows(x_in.shape[1], k_in, random_state=random_state)
    rows_out = _random_orthonormal_rows(y_out.shape[1], k_out, random_state=random_state)
    rb.pca_in = pca_like_from_rows_mass(x_in.mean(0), rows_in, rb.mass_in_, num_samples)
    rb.pca_out = pca_like_from_rows_mass(y_out.mean(0), rows_out, rb.mass_out_, num_samples)
    return rb

