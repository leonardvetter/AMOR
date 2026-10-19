"""Chebyshev / Legendre / Jacobi tensor-product spatial bases."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev, legendre
from scipy.special import eval_jacobi

def _lowdin_rows(candidates: np.ndarray, k: int) -> np.ndarray:
    """Symmetric orthonormalization, orth(Ξ) = Ξ (Ξᵀ Ξ)^{-1/2}.

    Euclidean QR on a nested total-degree flag collapses Jacobi, Legendre, and
    Chebyshev to one basis up to sign. Löwdin keeps the raw Gram matrix, so the
    three families stay distinct. This is the paper's orth_M with M = I.
    """
    rows = np.asarray(candidates[:k], dtype=np.float64)
    gram = rows @ rows.T
    evals, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
    inv_sqrt = evecs * (1.0 / np.sqrt(np.clip(evals, 1e-12, None)))
    whitened = (inv_sqrt @ evecs.T) @ rows
    return whitened.astype(np.float32)


def _poly_basis_1d(family: str, deg: int, xs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if family == "chebyshev":
        return chebyshev.Chebyshev.basis(deg)(xs)
    if family == "legendre":
        return legendre.Legendre.basis(deg)(xs)
    if family == "jacobi":
        return eval_jacobi(deg, alpha, beta, xs).astype(np.float64)
    raise ValueError(f"Unknown polynomial family: {family}")


def polynomial_rows(n: int, m: int, k: int, family: str, alpha: float, beta: float) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, m, dtype=np.float64)
    max_deg = int(np.ceil((np.sqrt(8.0 * k + 1.0) - 3.0) / 2.0)) + 1
    modes = []
    for total in range(max_deg + 1):
        for i in range(total + 1):
            j = total - i
            fx = _poly_basis_1d(family, i, xs, alpha, beta)
            fy = _poly_basis_1d(family, j, ys, alpha, beta)
            modes.append(np.kron(fx, fy).astype(np.float32))
            if len(modes) >= k:
                break
        if len(modes) >= k:
            break
    return _lowdin_rows(np.stack(modes, axis=0), k)
