"""Dual-encoder DIS: physical decoder Φ + offline encoder E with E Φ ≈ I.

Input GEVP (Zahm option 2): Σ_in Φ = C^{-1} Φ Λ, Φᵀ C^{-1} Φ = I.
MC path yields ψ with Φ = C^{1/2} ψ. Decoder rows = Φ.
Encoder E_in = Λ^{-1} (Σ_in Φ)ᵀ from averaged Jacobian products
(equivalently Φᵀ C^{-1} when Φ are exact GEVP eigenvectors).

Output: Σ_out M Φ_out = Φ_out Λ_out with Φ_outᵀ M Φ_out = I; M-adjoint
encoder E_out = Φ_outᵀ M (M = I on structured grids).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def phi_from_psi(psi: np.ndarray, cov) -> np.ndarray:
    """Φ = C^{1/2} ψ (rows). ``cov`` needs ``apply_sqrt``."""
    psi = np.asarray(psi, dtype=np.float64)
    return np.stack([cov.apply_sqrt(r) for r in psi], axis=0)


def encoder_from_cinv(phi: np.ndarray, cov) -> np.ndarray:
    """E = Φᵀ C^{-1} as rows: row i = C^{-1} φ_i."""
    phi = np.asarray(phi, dtype=np.float64)
    return np.stack([cov.apply_cinv(r) for r in phi], axis=0)


def encoder_from_sigma_phi(
    sigma_phi: np.ndarray,
    lam: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """E = Λ^{-1} (Σ Φ)ᵀ with row convention: rows = (ΣΦ) / λ."""
    sp = np.asarray(sigma_phi, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    if sp.shape[0] != lam.shape[0]:
        raise ValueError(f"sigma_phi rows {sp.shape[0]} != lam {lam.shape[0]}")
    scale = 1.0 / np.maximum(lam, eps)
    return sp * scale[:, None]


def assert_dual(E: np.ndarray, Phi: np.ndarray, *, atol: float = 1e-3, name: str = "E@Phi") -> float:
    """Return ||E Φᵀ - I||_F (row convention: both (k, D))."""
    E = np.asarray(E, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    g = E @ Phi.T
    err = float(np.linalg.norm(g - np.eye(g.shape[0]), ord="fro"))
    if err > atol:
        print(f"WARNING {name}: ||E Phi^T - I||_F = {err:.3e} > {atol:g}", flush=True)
    else:
        print(f"{name}: ||E Phi^T - I||_F = {err:.3e}", flush=True)
    return err


def average_sigma_phi_rows(
    phi_rows: np.ndarray,
    apply_jtj: Callable[[np.ndarray], np.ndarray],
    *,
    n_report: int = 8,
) -> np.ndarray:
    """For each mode φ, ``apply_jtj(φ)`` returns averaged (J^* J φ)."""
    phi = np.asarray(phi_rows, dtype=np.float64)
    out = np.zeros_like(phi)
    k = phi.shape[0]
    for j in range(k):
        out[j] = np.asarray(apply_jtj(phi[j]), dtype=np.float64).reshape(-1)
        if (j + 1) % n_report == 0 or (j + 1) == k:
            print(f"  ΣΦ mode {j + 1}/{k}", flush=True)
    return out


def build_dual_encoder_from_sigma(
    phi_rows: np.ndarray,
    lam: np.ndarray,
    apply_jtj: Callable[[np.ndarray], np.ndarray],
    *,
    atol: float = 1e-2,
    name: str = "DIS_in E@Phi",
) -> np.ndarray:
    """E = Λ^{-1}(Σ Φ)ᵀ via averaged J^*J products; assert dual property."""
    sigma_phi = average_sigma_phi_rows(phi_rows, apply_jtj)
    E = encoder_from_sigma_phi(sigma_phi, lam)
    assert_dual(E, phi_rows, atol=atol, name=name)
    return E


def m_orthonormalize_rows(rows: np.ndarray, mass_diag_or_none=None) -> np.ndarray:
    """M-orthonormalize rows. ``None`` → Euclidean; 1-d → sparse diag; else general M."""
    rows = np.asarray(rows, dtype=np.float64)
    if mass_diag_or_none is None:
        q, _ = np.linalg.qr(rows.T, mode="reduced")
        return q.T
    import scipy.sparse as sp

    M = mass_diag_or_none
    if not sp.issparse(M) and np.ndim(M) == 1:
        w = np.asarray(M, dtype=np.float64).reshape(-1)
        if w.shape[0] != rows.shape[1]:
            raise ValueError("mass size mismatch")
        M = sp.diags(np.maximum(w, 1e-30), format="csr")
    return m_orthonormalize_rows_general(rows, M)


def m_orthonormalize_rows_general(rows: np.ndarray, M) -> np.ndarray:
    """Rows Φ with Φ M Φᵀ = I for general SPD sparse/dense M (not assumed diagonal)."""
    import scipy.sparse as sp

    rows = np.asarray(rows, dtype=np.float64)
    if not sp.issparse(M):
        M = sp.csr_matrix(np.asarray(M, dtype=np.float64))
    else:
        M = M.tocsr()
    if M.shape[0] != rows.shape[1] or M.shape[1] != rows.shape[1]:
        raise ValueError(f"M shape {M.shape} incompatible with rows {rows.shape}")
    MPhi = M @ rows.T
    G = rows @ MPhi
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    evals = np.maximum(evals, 1e-30)
    scale = 1.0 / np.sqrt(evals)
    Ginvih = (evecs * scale[None, :]) @ evecs.T
    return Ginvih @ rows


def m_adjoint_encoder_rows(phi_rows: np.ndarray, M) -> np.ndarray:
    """E = Φᵀ M as rows: encode z = E (x−μ) = Φᵀ M (x−μ)."""
    import scipy.sparse as sp
    from .mass.mesh_mass import apply_mass

    phi = np.asarray(phi_rows, dtype=np.float64)
    if not sp.issparse(M) and np.ndim(M) == 1:
        M = sp.diags(np.asarray(M, dtype=np.float64).reshape(-1), format="csr")
    return apply_mass(M, phi)


class HippylibExactPrior:
    """Exact BiLaplacian prior on FEM parameter DoFs (C^{-1}=R; C^{1/2}=A^{-1} M^{1/2})."""

    def __init__(self, prior):
        self.prior = prior

    def apply_cinv(self, x_dof: np.ndarray) -> np.ndarray:
        import dolfin as dl

        x = np.asarray(x_dof, dtype=np.float64).reshape(-1)
        v = dl.Vector()
        self.prior.init_vector(v, 0)
        v.set_local(np.ascontiguousarray(x))
        v.apply("")
        out = dl.Vector()
        self.prior.init_vector(out, 0)
        self.prior.R.mult(v, out)
        return np.asarray(out.get_local(), dtype=np.float64).copy()

    def apply_sqrt(self, x_dof: np.ndarray) -> np.ndarray:
        """C^{1/2} x ≈ A^{-1} (M^{1/2} x) with lumped M^{1/2} from diag(M)."""
        import dolfin as dl

        x = np.asarray(x_dof, dtype=np.float64).reshape(-1)
        if not (hasattr(self.prior, "Asolver") and hasattr(self.prior, "M")):
            # Fallback: Rsolver gives C x; not a true square root — callers should
            # prefer grid LowRankPriorCov for Φ = C^{1/2} ψ when Asolver missing.
            v = dl.Vector()
            self.prior.init_vector(v, 0)
            v.set_local(np.ascontiguousarray(x))
            v.apply("")
            y = dl.Vector()
            self.prior.init_vector(y, 0)
            self.prior.Rsolver.solve(y, v)
            return np.asarray(y.get_local(), dtype=np.float64).copy()
        try:
            mdiag = np.asarray(self.prior.M.get_diagonal().get_local(), dtype=np.float64)
        except Exception:
            mdiag = np.ones_like(x)
        w = np.sqrt(np.maximum(mdiag, 1e-30)) * x
        rhs = dl.Vector()
        self.prior.init_vector(rhs, 0)
        rhs.set_local(np.ascontiguousarray(w))
        rhs.apply("")
        y = dl.Vector()
        self.prior.init_vector(y, 0)
        self.prior.Asolver.solve(y, rhs)
        return np.asarray(y.get_local(), dtype=np.float64).copy()


class GridMappedExactPrior:
    """Apply exact hippylib prior ops on extraction-grid vectors via DoF map.

    Used for Helmholtz DIS when ``--prior exact``: Φ = C^{1/2} ψ and
    E = Φᵀ C^{-1} on the same grid used by AMOR.
    """

    def __init__(self, prior, *, Vh_param, box, nx, grid_to_parameter_vector, extract_to_grid, vector2Function):
        self._exact = HippylibExactPrior(prior)
        self.Vh_param = Vh_param
        self.box = box
        self.nx = int(nx)
        self._grid_to_param = grid_to_parameter_vector
        self._extract = extract_to_grid
        self._v2f = vector2Function
        self.rank = self.nx * self.nx  # full rank (exact)

    def _to_dof(self, x_grid: np.ndarray):
        g = np.asarray(x_grid, dtype=np.float64).reshape(self.nx, self.nx)
        return self._grid_to_param(self.Vh_param, g, self.box)

    def _from_dof(self, v_dof) -> np.ndarray:
        f = self._v2f(v_dof, self.Vh_param)
        return np.asarray(self._extract(f, self.box, self.nx, self.nx), dtype=np.float64).ravel()

    def apply_cinv(self, x_grid: np.ndarray) -> np.ndarray:
        import dolfin as dl

        v = self._to_dof(x_grid)
        y_loc = self._exact.apply_cinv(v.get_local())
        out = dl.Vector()
        self._exact.prior.init_vector(out, 0)
        out.set_local(np.ascontiguousarray(y_loc))
        out.apply("")
        return self._from_dof(out)

    def apply_sqrt(self, x_grid: np.ndarray) -> np.ndarray:
        import dolfin as dl

        v = self._to_dof(x_grid)
        y_loc = self._exact.apply_sqrt(v.get_local())
        out = dl.Vector()
        self._exact.prior.init_vector(out, 0)
        out.set_local(np.ascontiguousarray(y_loc))
        out.apply("")
        return self._from_dof(out)
