"""Memory-efficient Roseberry MC probe stack + thin SVD for DIS rows."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd


def resolve_n_states(n_available: int, n_states: int | None) -> int:
    """n_states<=0 or None → use all available samples."""
    if n_states is None or int(n_states) <= 0:
        return int(n_available)
    return min(int(n_states), int(n_available))


def svd_rows_from_probes(
    probes: np.ndarray,
    k: int,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """probes (n_tot, d) → (rows (k,d), eigvals≈σ²). Uses randomized SVD.

    Accepts memmap; avoids full RAM materialization when possible.
    """
    if probes.ndim != 2:
        raise ValueError(f"probes must be 2d, got {probes.shape}")
    n_tot, d = probes.shape
    k_eff = min(k, n_tot, d)
    if k_eff < 1:
        raise ValueError("k_eff < 1")
    # Keep memmap / float32 as-is; only upcast small in-memory float64 arrays.
    if isinstance(probes, np.memmap) or probes.dtype == np.float32:
        x = probes
    elif probes.nbytes > 2 * 1024**3:
        x = np.asarray(probes, dtype=np.float32)
    else:
        x = np.asarray(probes, dtype=np.float64)
    n_oversamples = min(20, max(2, n_tot - k_eff))
    _, s, vt = randomized_svd(
        x,
        n_components=k_eff,
        n_oversamples=n_oversamples,
        random_state=random_state,
    )
    return vt.astype(np.float64), (s.astype(np.float64) ** 2)


class ProbeMemmapStack:
    """Append probe rows to float32 memmap; SVD without holding float64 list."""

    def __init__(self, path: str | Path, n_tot: int, d: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_tot = int(n_tot)
        self.d = int(d)
        self._i = 0
        self.mm = np.lib.format.open_memmap(
            str(self.path), mode="w+", dtype=np.float32, shape=(self.n_tot, self.d)
        )

    def append(self, row: np.ndarray):
        if self._i >= self.n_tot:
            raise IndexError("ProbeMemmapStack full")
        self.mm[self._i] = np.asarray(row, dtype=np.float32).reshape(-1)
        self._i += 1

    def finalize_svd(self, k: int, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
        if self._i != self.n_tot:
            raise RuntimeError(f"expected {self.n_tot} rows, got {self._i}")
        self.mm.flush()
        rows, ev = svd_rows_from_probes(self.mm, k, random_state=random_state)
        return rows, ev

    def close(self, *, delete: bool = True):
        del self.mm
        if delete and self.path.is_file():
            try:
                os.remove(self.path)
            except OSError:
                pass
