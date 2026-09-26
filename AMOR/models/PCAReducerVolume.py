"""PCA reducer for gridded tensor fields (N, C, *spatial) with arbitrary spatial rank."""

from __future__ import annotations

import gc
import heapq
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .bases import (
    channelize_rows as _channelize_rows,
    orthonormal_rows as _orthonormal_rows,
    pca_like_from_rows as _pca_like_from_rows,
    random_orthonormal_rows as _random_orthonormal_rows,
)
from .bases.laplacian import _grid_1d_laplacian
def _gram_evr_from_memmap(mat: np.memmap, row_block: int = 32) -> np.ndarray:
    """Explained-variance ratios from sample-space Gram matrix (no components built)."""
    n, _p = mat.shape
    mean = np.zeros(_p, dtype=np.float64)
    for i in range(n):
        mean += mat[i]
    mean /= n

    gram = np.zeros((n, n), dtype=np.float64)
    for i0 in range(0, n, row_block):
        i1 = min(i0 + row_block, n)
        xi = np.asarray(mat[i0:i1], dtype=np.float64) - mean
        for j0 in range(0, n, row_block):
            j1 = min(j0 + row_block, n)
            xj = np.asarray(mat[j0:j1], dtype=np.float64) - mean
            gram[i0:i1, j0:j1] = xi @ xj.T

    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 1e-12]
    return eigvals / eigvals.sum()


def k_at_variance_cutoff(evr: np.ndarray, j: float) -> int:
    return int(np.searchsorted(np.cumsum(evr), j) + 1)


def _pca_fit_memmap(
    mat: np.memmap,
    variance_target: float,
    min_components: int | None = None,
    n_components_exact: int | None = None,
    row_block: int = 32,
) -> PCA:
    """Sample-space PCA for (n_samples, n_features) memmap when n_samples << n_features."""
    n, p = mat.shape
    # n_samples << n_features (sample-space PCA), so the whole block fits in RAM as
    # float32 (n*p*4 bytes). Load once and do Gram + reconstruction as single BLAS
    # matmuls: O(k*n*p) done in ONE data pass, not k passes (row_block kept only for
    # signature compatibility). This is the difference between minutes and >4h at k~700.
    X = np.array(mat, dtype=np.float32, copy=True)          # (n, p)
    mean = X.mean(axis=0)                                    # (p,)
    X -= mean                                                # center in place

    # float32 GEMM into a tiny (n,n) Gram, then promote for the eigensolve. Avoids
    # 47 GB float64 copies of X; float32 Gram is ample for the leading eigenvectors.
    gram = (X @ X.T).astype(np.float64)                     # (n, n)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-12
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    evr = eigvals / eigvals.sum()
    if n_components_exact is not None:
        k = min(int(n_components_exact), len(evr))
    else:
        k = min(int(np.searchsorted(np.cumsum(evr), variance_target) + 1), len(evr))
        if min_components is not None:
            k = max(k, min(min_components, len(evr)))

    # components[i] = (eigvec_i . X_centered) / sqrt(eigval_i); all k at once.
    comp_vecs = eigvecs[:, :k].astype(np.float32)           # (n, k)
    scale = np.sqrt(eigvals[:k]).astype(np.float32)         # (k,)
    components = (comp_vecs.T @ X) / scale[:, None]         # (k, p) single matmul
    components = np.ascontiguousarray(components, dtype=np.float32)

    pca = PCA(n_components=k)
    pca.mean_ = mean.astype(np.float32)
    pca.components_ = components
    pca.explained_variance_ratio_ = evr[:k]
    pca.explained_variance_ = eigvals[:k] / max(n - 1, 1)
    pca.singular_values_ = np.sqrt(eigvals[:k])
    pca.n_components_ = k
    pca.n_features_in_ = p
    return pca


class PCAReducerVolume:
    """
    Fit PCA on flattened gridded fields (N, C, *spatial).

    Exposes ``shape_ = spatial_dims`` so AMOR uses native 2D/3D/ND layout;
    PCA encode/decode flatten C * prod(spatial) internally.
    """

    def __init__(
        self,
        svd_solver: str = "randomized",
        whiten: bool = False,
        random_state: Optional[int] = None,
    ):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels: Optional[int] = None
        self.out_channels: Optional[int] = None
        self.volume_shape_: Optional[Tuple[int, ...]] = None
        self.shape_: Optional[Tuple[int, ...]] = None
        self.output_shape_: Optional[Tuple[int, ...]] = None

    def fit(self, samples: np.ndarray, solution: np.ndarray) -> "PCAReducerVolume":
        if samples.ndim < 4:
            raise ValueError(f"Expected (N, C_in, *spatial), got {samples.shape}")
        num_samples, c_in = samples.shape[0], samples.shape[1]
        spatial = tuple(int(x) for x in samples.shape[2:])
        self.n_channels = c_in
        self.volume_shape_ = spatial
        self.shape_ = spatial
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)

        if solution.ndim < 4:
            raise ValueError(f"Expected (N, C_out, *spatial), got {solution.shape}")
        _, c_out = solution.shape[0], solution.shape[1]
        if tuple(solution.shape[2:]) != spatial:
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        self.output_shape_ = (c_out, *spatial)
        y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        self.pca_in = PCA(
            n_components=min(num_samples, x_in.shape[1]),
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(x_in)
        self.pca_out = PCA(
            n_components=min(num_samples, y_out.shape[1]),
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(y_out)
        return self

    def fit_from_mmap(
        self,
        x_mmap: np.ndarray,
        y_mmap: np.ndarray,
        indices: Sequence[int],
        variance_target: float = 0.99,
        variance_target_out: float | None = None,
        min_components_in: int | None = None,
        min_components_out: int | None = None,
        n_components_in: int | None = None,
        n_components_out: int | None = None,
        mmap_dir: str | Path | None = None,
    ) -> "PCAReducerVolume":
        """Row-by-row memmap matrices; PCA keeps variance_target or exact/min mode counts."""
        if variance_target_out is None:
            variance_target_out = variance_target
        idx = np.asarray(indices, dtype=np.int64)
        n_samples = len(idx)
        if n_samples == 0:
            raise ValueError("No training indices for PCA fit")

        x0 = np.asarray(x_mmap[int(idx[0])], dtype=np.float32)
        y0 = np.asarray(y_mmap[int(idx[0])], dtype=np.float32)
        if x0.ndim < 3 or y0.ndim < 3:
            raise ValueError(f"Expected (C, *spatial) samples, got x={x0.shape} y={y0.shape}")

        c_in, *spatial = (int(x) for x in x0.shape)
        spatial_t = tuple(spatial)
        c_out = int(y0.shape[0])
        if tuple(int(x) for x in y0.shape[1:]) != spatial_t:
            raise ValueError("Input and output spatial dimensions must match")

        n_in = c_in * int(np.prod(spatial_t))
        n_out = c_out * int(np.prod(spatial_t))
        mmap_dir = Path(mmap_dir or os.environ.get("TMPDIR", "/tmp"))
        mmap_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()

        def _fill_memmap(path: Path, arr_mmap: np.ndarray, n_feat: int) -> np.memmap:
            mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_samples, n_feat))
            for row, sample_idx in enumerate(idx):
                mm[row] = np.asarray(arr_mmap[int(sample_idx)], dtype=np.float32).reshape(-1)
            mm.flush()
            return mm

        x_path = mmap_dir / f"ldc3d_pca_x_{pid}.dat"
        x_mat = _fill_memmap(x_path, x_mmap, n_in)
        self.pca_in = _pca_fit_memmap(
            x_mat, variance_target, min_components=min_components_in, n_components_exact=n_components_in
        )
        del x_mat
        x_path.unlink(missing_ok=True)
        gc.collect()

        y_path = mmap_dir / f"ldc3d_pca_y_{pid}.dat"
        y_mat = _fill_memmap(y_path, y_mmap, n_out)
        self.pca_out = _pca_fit_memmap(
            y_mat, variance_target_out, min_components=min_components_out, n_components_exact=n_components_out
        )
        del y_mat
        y_path.unlink(missing_ok=True)
        gc.collect()

        self.n_channels = c_in
        self.out_channels = c_out
        self.volume_shape_ = spatial_t
        self.shape_ = spatial_t
        self.output_shape_ = (c_out, *spatial_t)
        return self

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducerVolume must be fitted first")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    @staticmethod
    def modes_at_cutoffs(evr: np.ndarray, cutoffs: Sequence[float]) -> dict[float, int]:
        return {float(j): k_at_variance_cutoff(evr, float(j)) for j in cutoffs}

    def truncate(self, k_in: int, k_out: int) -> "PCAReducerVolume":
        """Keep only the first k_in / k_out PCA modes; discard the rest in place."""
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducerVolume must be fitted before truncate")
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

    def save(self, path: str | Path) -> None:
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducerVolume must be fitted before save")
        path = Path(path)
        np.savez_compressed(
            path,
            n_channels=self.n_channels,
            out_channels=self.out_channels,
            volume_shape=np.asarray(self.volume_shape_, dtype=np.int64),
            pca_in_mean=self.pca_in.mean_,
            pca_in_components=self.pca_in.components_,
            pca_in_evr=self.pca_in.explained_variance_ratio_,
            pca_out_mean=self.pca_out.mean_,
            pca_out_components=self.pca_out.components_,
            pca_out_evr=self.pca_out.explained_variance_ratio_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCAReducerVolume":
        data = np.load(path)
        rb = cls()
        rb.n_channels = int(data["n_channels"])
        rb.out_channels = int(data["out_channels"])
        spatial = tuple(int(x) for x in data["volume_shape"])
        rb.volume_shape_ = spatial
        rb.shape_ = spatial
        rb.output_shape_ = (rb.out_channels, *spatial)

        def _pca_from_arrays(mean, components, evr):
            pca = PCA(n_components=components.shape[0])
            pca.mean_ = mean
            pca.components_ = components
            pca.explained_variance_ratio_ = evr
            pca.n_components_ = components.shape[0]
            pca.n_features_in_ = mean.shape[0]
            return pca

        rb.pca_in = _pca_from_arrays(
            data["pca_in_mean"], data["pca_in_components"], data["pca_in_evr"]
        )
        rb.pca_out = _pca_from_arrays(
            data["pca_out_mean"], data["pca_out_components"], data["pca_out_evr"]
        )
        return rb


def _laplacian_eigen_rows_nd(
    shape: Tuple[int, ...],
    k: int,
    periodic: bool,
) -> np.ndarray:
    """Lowest-k eigenmodes of the separable ND grid Laplacian (rows, prod(shape))."""
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    if ndim < 2:
        raise ValueError(f"Need spatial ndim >= 2, got {shape}")
    evals_1d = []
    evecs_1d = []
    for n in shape:
        l1 = _grid_1d_laplacian(n, periodic).toarray()
        w, v = np.linalg.eigh(l1)
        evals_1d.append(w.astype(np.float64))
        evecs_1d.append(v.astype(np.float64))  # columns = modes

    # Priority queue of multi-indices by total eigenvalue.
    heap: list[tuple[float, tuple[int, ...]]] = []
    zero = tuple(0 for _ in shape)
    heapq.heappush(heap, (sum(evals_1d[d][0] for d in range(ndim)), zero))
    seen = {zero}
    rows = []
    spatial = int(np.prod(shape))
    while heap and len(rows) < k:
        _lam, idx = heapq.heappop(heap)
        mode = np.ones(1, dtype=np.float64)
        for d, i in enumerate(idx):
            mode = np.kron(mode, evecs_1d[d][:, i])
        rows.append(mode.astype(np.float32))
        for d in range(ndim):
            nxt = list(idx)
            nxt[d] += 1
            if nxt[d] >= shape[d]:
                continue
            t = tuple(nxt)
            if t in seen:
                continue
            seen.add(t)
            heapq.heappush(heap, (sum(evals_1d[ax][t[ax]] for ax in range(ndim)), t))
    stacked = np.stack(rows, axis=0)
    if stacked.shape[0] < k:
        pad = _random_orthonormal_rows(spatial, k - stacked.shape[0], random_state=0)
        stacked = np.concatenate([stacked, pad], axis=0)
    return _orthonormal_rows(stacked, k)


def _mean_from_mmap(arr_mmap: np.ndarray, indices: Sequence[int], n_feat: int) -> np.ndarray:
    mean = np.zeros(n_feat, dtype=np.float64)
    for sample_idx in indices:
        mean += np.asarray(arr_mmap[int(sample_idx)], dtype=np.float64).reshape(-1)
    mean /= max(len(indices), 1)
    return mean.astype(np.float32)


def fit_volume_laplacian_basis(
    x_mmap: np.ndarray,
    y_mmap: np.ndarray,
    indices: Sequence[int],
    k_in: int,
    k_out: int,
    *,
    laplacian_periodic: bool = False,
    random_state: Optional[int] = None,
) -> PCAReducerVolume:
    """Build AMOR-compatible volume reducer from ND grid Laplacian eigenmodes."""
    del random_state  # deterministic geometry basis
    idx = list(indices)
    x0 = np.asarray(x_mmap[int(idx[0])], dtype=np.float32)
    y0 = np.asarray(y_mmap[int(idx[0])], dtype=np.float32)
    c_in, *spatial = (int(x) for x in x0.shape)
    spatial_t = tuple(spatial)
    c_out = int(y0.shape[0])
    if tuple(int(x) for x in y0.shape[1:]) != spatial_t:
        raise ValueError("Input and output spatial dimensions must match")

    n_in = c_in * int(np.prod(spatial_t))
    n_out = c_out * int(np.prod(spatial_t))
    x_mean = _mean_from_mmap(x_mmap, idx, n_in)
    y_mean = _mean_from_mmap(y_mmap, idx, n_out)

    p_in = int(np.ceil(k_in / max(1, c_in)))
    p_out = int(np.ceil(k_out / max(1, c_out)))
    rows_in_1 = _laplacian_eigen_rows_nd(spatial_t, p_in, laplacian_periodic)
    rows_out_1 = _laplacian_eigen_rows_nd(spatial_t, p_out, laplacian_periodic)
    rows_in = _channelize_rows(rows_in_1, c_in, k_in)
    rows_out = _channelize_rows(rows_out_1, c_out, k_out)

    rb = PCAReducerVolume(random_state=None)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.volume_shape_ = spatial_t
    rb.shape_ = spatial_t
    rb.output_shape_ = (c_out, *spatial_t)
    rb.pca_in = _pca_like_from_rows(x_mean.astype(np.float64), rows_in, len(idx))
    rb.pca_out = _pca_like_from_rows(y_mean.astype(np.float64), rows_out, len(idx))
    rb.basis_type = "laplacian"
    return rb


def fit_volume_random_init_basis(
    x_mmap: np.ndarray,
    y_mmap: np.ndarray,
    indices: Sequence[int],
    k_in: int,
    k_out: int,
    *,
    random_state: Optional[int] = None,
) -> PCAReducerVolume:
    """Random orthonormal volume rows + data means (learned-basis init)."""
    idx = list(indices)
    x0 = np.asarray(x_mmap[int(idx[0])], dtype=np.float32)
    y0 = np.asarray(y_mmap[int(idx[0])], dtype=np.float32)
    c_in, *spatial = (int(x) for x in x0.shape)
    spatial_t = tuple(spatial)
    c_out = int(y0.shape[0])
    n_in = c_in * int(np.prod(spatial_t))
    n_out = c_out * int(np.prod(spatial_t))
    x_mean = _mean_from_mmap(x_mmap, idx, n_in)
    y_mean = _mean_from_mmap(y_mmap, idx, n_out)
    rows_in = _random_orthonormal_rows(n_in, k_in, random_state)
    rows_out = _random_orthonormal_rows(n_out, k_out, random_state)
    rb = PCAReducerVolume(random_state=random_state)
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.volume_shape_ = spatial_t
    rb.shape_ = spatial_t
    rb.output_shape_ = (c_out, *spatial_t)
    rb.pca_in = _pca_like_from_rows(x_mean.astype(np.float64), rows_in, len(idx))
    rb.pca_out = _pca_like_from_rows(y_mean.astype(np.float64), rows_out, len(idx))
    rb.basis_type = "random_init"
    return rb
