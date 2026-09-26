"""Roseberry DIS Jacobian for generate_nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.py (FiPy Grid2D, D=exp(m))."""

from __future__ import annotations

import math

import numpy as np
from fipy import CellVariable, DiffusionTerm, Grid2D


def _fipy_forward(
    f_flat: np.ndarray,
    m_flat: np.ndarray,
    mesh,
    D_cv: CellVariable,
    f_cv: CellVariable,
    u_var: CellVariable,
) -> np.ndarray:
    D_cv.setValue(np.exp(np.asarray(m_flat, dtype=np.float64).reshape(-1)))
    f_cv.setValue(np.asarray(f_flat, dtype=np.float64).reshape(-1))
    (DiffusionTerm(coeff=D_cv, var=u_var) == -f_cv).solve(var=u_var)
    return np.asarray(u_var.value, dtype=np.float64).reshape(-1).copy()


def _fipy_linear_solve(
    m_flat: np.ndarray,
    rhs_flat: np.ndarray,
    mesh,
    D_cv: CellVariable,
    du_var: CellVariable,
) -> np.ndarray:
    D_cv.setValue(np.exp(np.asarray(m_flat, dtype=np.float64).reshape(-1)))
    rhs_cv = CellVariable(mesh=mesh, value=np.asarray(rhs_flat, dtype=np.float64).reshape(-1))
    du_var.setValue(0.0)
    (DiffusionTerm(coeff=D_cv, var=du_var) == rhs_cv).solve(var=du_var)
    return np.asarray(du_var.value, dtype=np.float64).reshape(-1).copy()


def _grad_m_fipy(
    u_flat: np.ndarray,
    p_flat: np.ndarray,
    m_flat: np.ndarray,
    mesh,
) -> np.ndarray:
    """Adjoint grad w.r.t. m at fixed (u, p): D * (grad u · grad p) on cells."""
    u_cv = CellVariable(mesh=mesh, value=np.asarray(u_flat, dtype=np.float64).reshape(-1))
    u_cv.constrain(0.0, where=mesh.exteriorFaces)
    p_cv = CellVariable(mesh=mesh, value=np.asarray(p_flat, dtype=np.float64).reshape(-1))
    p_cv.constrain(0.0, where=mesh.exteriorFaces)
    D_cv = CellVariable(mesh=mesh, value=np.exp(np.asarray(m_flat, dtype=np.float64).reshape(-1)))
    gu = u_cv.grad
    gp = p_cv.grad
    return np.asarray((D_cv * (gu[0] * gp[0] + gu[1] * gp[1])).value, dtype=np.float64).reshape(-1).copy()


class NonlinearDiffusionJacobian:
    """J of FiPy map (f, m) -> u from generate_nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.py."""

    def __init__(self, n: int, L: float = 1.0, *, fd_eps: float = 1e-5):
        self.n = n
        self.L = L
        self.dx = L / n
        self.d = n * n
        self.d_in = 2 * self.d
        self.fd_eps = fd_eps
        self.mesh = Grid2D(nx=n, ny=n, dx=self.dx, dy=self.dx)
        self._D_cv = CellVariable(mesh=self.mesh, value=1.0)
        self._f_cv = CellVariable(mesh=self.mesh, value=0.0)
        self._u = CellVariable(mesh=self.mesh, value=0.0)
        self._u.constrain(0.0, where=self.mesh.exteriorFaces)
        self._du = CellVariable(mesh=self.mesh, value=0.0)
        self._du.constrain(0.0, where=self.mesh.exteriorFaces)
        self._f_flat: np.ndarray | None = None
        self._m_flat: np.ndarray | None = None
        self._u_flat: np.ndarray | None = None

    def _forward(self, f_flat: np.ndarray, m_flat: np.ndarray) -> np.ndarray:
        return _fipy_forward(f_flat, m_flat, self.mesh, self._D_cv, self._f_cv, self._u)

    def _solve(self, m_flat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return _fipy_linear_solve(m_flat, rhs, self.mesh, self._D_cv, self._du)

    def set_state(self, f: np.ndarray, m: np.ndarray):
        self._f_flat = np.asarray(f, dtype=np.float64).reshape(-1)
        self._m_flat = np.asarray(m, dtype=np.float64).reshape(-1)
        self._u_flat = self._forward(self._f_flat, self._m_flat)

    def Jv(self, vf: np.ndarray, vm: np.ndarray) -> np.ndarray:
        assert self._f_flat is not None and self._m_flat is not None
        vf = np.asarray(vf, dtype=np.float64).reshape(-1)
        vm = np.asarray(vm, dtype=np.float64).reshape(-1)
        eps = self.fd_eps
        up = self._forward(self._f_flat + eps * vf, self._m_flat + eps * vm)
        um = self._forward(self._f_flat - eps * vf, self._m_flat - eps * vm)
        return (up - um) / (2.0 * eps)

    def Jtw(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._m_flat is not None and self._u_flat is not None
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        p = self._solve(self._m_flat, w)
        grad_f = -p.copy()
        grad_m = _grad_m_fipy(self._u_flat, p, self._m_flat, self.mesh)
        return grad_f, grad_m

    def jv_forward_fd(self, vf: np.ndarray, vm: np.ndarray, eps: float | None = None) -> np.ndarray:
        return self.Jv(vf, vm) if eps is None or abs(eps - self.fd_eps) < 1e-15 else self._jv_fd(vf, vm, eps)

    def _jv_fd(self, vf: np.ndarray, vm: np.ndarray, eps: float) -> np.ndarray:
        assert self._f_flat is not None and self._m_flat is not None
        vf = np.asarray(vf, dtype=np.float64).reshape(-1)
        vm = np.asarray(vm, dtype=np.float64).reshape(-1)
        up = self._forward(self._f_flat + eps * vf, self._m_flat + eps * vm)
        um = self._forward(self._f_flat - eps * vf, self._m_flat - eps * vm)
        return (up - um) / (2.0 * eps)


def collect_dis_probes(
    jac: NonlinearDiffusionJacobian,
    f: np.ndarray,
    m: np.ndarray,
    n_probes: int,
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
    prior_cov=None,
    x_std_flat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    jac.set_state(f, m)
    a_list, b_list = [], []
    x_std = None if x_std_flat is None else np.asarray(x_std_flat, dtype=np.float64).reshape(-1)
    for _ in range(n_probes):
        g = rng.standard_normal(jac.d)
        gf, gm = jac.Jtw(g)
        a = np.concatenate([gf, gm])
        if x_std is not None:
            a = a * x_std  # physical J^T → z-score adjoint
        if prior_cov is not None:
            a = prior_cov.apply_sqrt(a)
        a_list.append(a * scale)
        vf = rng.standard_normal(jac.d)
        vm = rng.standard_normal(jac.d)
        b_list.append(jac.Jv(vf, vm) * scale)
    return np.stack(a_list, 0), np.stack(b_list, 0)


def fit_dis_rows_from_data(
    f_samples: np.ndarray,
    m_samples: np.ndarray,
    k_in: int,
    k_out: int,
    n_states: int,
    n_probes: int,
    seed: int,
    *,
    L: float = 1.0,
    memmap_dir: str | None = None,
    prior_cov=None,
    x_std_flat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n_states<=0 → every sample in f_samples (no subsample).

    If prior_cov given: probe with C^{1/2} on (optionally z-scored) input adjoints;
    SVD → rows_in = ψ (Euclidean-orthonormal in whitened space).

    AMOR then whitens inputs with C^{-1/2} before lift and uses Euclidean encode on ψ.
    Equivalence: z = ψ^T C^{-1/2}(x-μ) = φ^T C^{-1}(x-μ) with φ = C^{1/2}ψ (Zahm option 2).
    """
    import os

    from .dis_mc_svd import ProbeMemmapStack, resolve_n_states

    n = int(round(np.sqrt(f_samples.shape[1])))
    jac = NonlinearDiffusionJacobian(n, L=L)
    rng = np.random.default_rng(seed)
    s = resolve_n_states(len(f_samples), n_states)
    if s == len(f_samples):
        idx = np.arange(len(f_samples))
    else:
        idx = rng.choice(len(f_samples), size=s, replace=False)
    n_tot = s * n_probes
    scale = 1.0 / math.sqrt(max(1, n_tot))
    tmp = memmap_dir or os.environ.get("DIS_MEMMAP_DIR", "/tmp")
    os.makedirs(tmp, exist_ok=True)
    tag = f"nonlin_dis_{os.getpid()}"
    stack_in = ProbeMemmapStack(os.path.join(tmp, f"{tag}_A.npy"), n_tot, 2 * jac.d)
    stack_out = ProbeMemmapStack(os.path.join(tmp, f"{tag}_B.npy"), n_tot, jac.d)
    try:
        for si, i in enumerate(idx):
            a, b = collect_dis_probes(
                jac,
                f_samples[i],
                m_samples[i].reshape(n, n),
                n_probes,
                rng,
                scale=scale,
                prior_cov=prior_cov,
                x_std_flat=x_std_flat,
            )
            for r in range(n_probes):
                stack_in.append(a[r])
                stack_out.append(b[r])
            if (si + 1) % 8 == 0 or (si + 1) == s:
                print(f"  DIS sample {si+1}/{s}", flush=True)
        rows_w, ev_in = stack_in.finalize_svd(k_in, random_state=int(rng.integers(0, 2**31 - 1)))
        rows_out, ev_out = stack_out.finalize_svd(k_out, random_state=int(rng.integers(0, 2**31 - 1)))
    finally:
        stack_in.close(delete=True)
        stack_out.close(delete=True)
    # Keep ψ (whitened Euclidean); runtime whitens x via C^{-1/2}.
    return rows_w, ev_in, rows_out, ev_out
