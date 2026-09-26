"""
3D linear elastostatics on a uniform voxel grid for neural-operator training.

Operator map:
    input  X = E(x)          float32 (n_samples, N, N, N)
    output Y = (ux,uy,uz)    float32 (n_samples, 3, N, N, N)

PDE (isotropic linear elasticity, fixed Poisson ratio nu):
    -div(sigma(u; E)) = 0 in (0,1)^3
    sigma = lambda(E) tr(eps) I + 2 mu(E) eps
    eps = sym(grad u)
    mu = E / (2(1+nu)),  lambda = E*nu / ((1+nu)(1-2*nu))

Boundary conditions (fixed for all samples):
    u = 0 on bottom face z=0
    traction t = (0, 0, -t0) on top face z=1
    traction-free on lateral faces

E(x) drawn from a low-dimensional truncated KL / spectral GRF on log-E:
    E = clip(E0 * exp(sigma * g)),  g = sum_{m=1}^{n_kl} xi_m * sqrt(lam_m) * phi_m

Discretization: trilinear hex FEM on an N^3 nodal grid (N-1)^3 cells.
With fixed nu, element stiffness Ke = E_elem * Ke0(nu); Ke0 precomputed once.

Solver:
    N small  -> scipy sparse direct (UMFPACK/SuperLU)
    N large  -> CG + ILU(0)/Jacobi (scipy); optional PETSc CG+GAMG if --solver petsc

Dependencies: numpy, scipy (optional petsc4py).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _atomic_savez(path: str, **arrays) -> None:
    """Write NPZ via temp file + os.replace so a crash never leaves a corrupt out.

    Note: np.savez_compressed appends '.npz' unless the path already ends with it.
    """
    path = os.path.abspath(path)
    tmp = path + f".tmp.{os.getpid()}.npz"
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass



# --------------------------------------------------------------------------- #
# Low-dimensional random field for log-E
# --------------------------------------------------------------------------- #
def _kl_basis(N: int, n_kl: int, length_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Product-cosine KL basis on the N^3 nodal grid.

    Returns
    -------
    lam : (n_kl,) eigenvalues (descending)
    phi : (n_kl, N, N, N) orthonormal modes (L2 on the grid, mean-zero)
    """
    h = 1.0 / max(N - 1, 1)
    xs = np.linspace(0.0, 1.0, N)
    # Candidate wave-numbers (non-negative integers); skip (0,0,0)
    max_k = max(4, int(np.ceil(n_kl ** (1.0 / 3.0)) + 4))
    cands = []
    for kx in range(0, max_k + 1):
        for ky in range(0, max_k + 1):
            for kz in range(0, max_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                k2 = float(kx * kx + ky * ky + kz * kz)
                # Squared-exp eigenvalues on unit cube
                lam = float(np.exp(-0.5 * (length_scale ** 2) * (np.pi ** 2) * k2))
                cands.append((lam, kx, ky, kz))
    cands.sort(key=lambda t: -t[0])
    cands = cands[:n_kl]

    phi = np.empty((n_kl, N, N, N), dtype=np.float64)
    lam = np.empty(n_kl, dtype=np.float64)
    for m, (lm, kx, ky, kz) in enumerate(cands):
        fx = np.cos(np.pi * kx * xs) if kx > 0 else np.ones(N)
        fy = np.cos(np.pi * ky * xs) if ky > 0 else np.ones(N)
        fz = np.cos(np.pi * kz * xs) if kz > 0 else np.ones(N)
        mode = np.einsum("i,j,k->ijk", fx, fy, fz)
        mode = mode - mode.mean()
        # Unit mean-square (pointwise O(1)), not Euclidean unit — else amp ~ N^{-3/2}
        rms = float(np.sqrt(np.mean(mode ** 2)))
        if rms < 1e-15:
            mode = np.zeros_like(mode)
            rms = 1.0
        phi[m] = mode / rms
        lam[m] = lm
    # Normalize eigenvalue scale so Var(g) ~ 1 when xi~N(0,1)
    s = float(lam.sum())
    if s > 0:
        lam = lam / s
    return lam, phi


def sample_E_field(
    n_samples: int,
    N: int,
    *,
    n_kl: int = 8,
    length_scale: float = 0.35,
    E0: float = 1.0,
    sigma: float = 0.5,
    E_min: float = 0.2,
    E_max: float = 5.0,
    rng: Optional[np.random.Generator] = None,
    basis: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Sample n_samples low-dim lognormal E fields on N^3 nodes."""
    if rng is None:
        rng = np.random.default_rng()
    if basis is None:
        lam, phi = _kl_basis(N, n_kl, length_scale)
    else:
        lam, phi = basis
    n_kl = int(lam.shape[0])
    xi = rng.standard_normal((n_samples, n_kl))
    # g[s] = sum_m xi[s,m] * sqrt(lam[m]) * phi[m]
    g = np.einsum("sm,mijk->sijk", xi * np.sqrt(lam)[None, :], phi)
    E = E0 * np.exp(sigma * g)
    E = np.clip(E, E_min, E_max).astype(np.float64)
    meta = {
        "n_kl": n_kl,
        "length_scale": length_scale,
        "E0": E0,
        "sigma": sigma,
        "E_min": E_min,
        "E_max": E_max,
        "kl_eigenvalues": lam.tolist(),
    }
    return E, meta


# --------------------------------------------------------------------------- #
# Hex FEM: reference stiffness Ke0 for E=1
# --------------------------------------------------------------------------- #
_HEX_REF = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)


def _hex_Ke0(h: float, nu: float) -> np.ndarray:
    """24x24 element stiffness for unit E, isotropic, axis-aligned hex of edge h."""
    # Lame parameters for E=1
    mu = 1.0 / (2.0 * (1.0 + nu))
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # 2-pt Gauss on [-1,1]
    g = 1.0 / np.sqrt(3.0)
    gps = np.array([-g, g])
    Ke = np.zeros((24, 24), dtype=np.float64)
    # Jacobian of ref->phys for cube edge h: x = 0.5*h*(xi+1) + x0 => J = (h/2) I
    J = 0.5 * h
    detJ = J ** 3
    invJ = 1.0 / J
    for xi in gps:
        for eta in gps:
            for zeta in gps:
                dN_dxi = np.empty((8, 3), dtype=np.float64)
                for a, (sx, sy, sz) in enumerate(_HEX_REF):
                    dN_dxi[a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
                    dN_dxi[a, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
                    dN_dxi[a, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
                dN_dx = dN_dxi * invJ  # (8,3)
                B = np.zeros((6, 24), dtype=np.float64)
                for a in range(8):
                    c = 3 * a
                    dNx, dNy, dNz = dN_dx[a]
                    B[0, c + 0] = dNx
                    B[1, c + 1] = dNy
                    B[2, c + 2] = dNz
                    B[3, c + 0] = dNy
                    B[3, c + 1] = dNx
                    B[4, c + 1] = dNz
                    B[4, c + 2] = dNy
                    B[5, c + 0] = dNz
                    B[5, c + 2] = dNx
                # Voigt C for isotropic (engineering shear)
                C = np.zeros((6, 6), dtype=np.float64)
                C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
                C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
                C[3, 3] = C[4, 4] = C[5, 5] = mu
                Ke += (B.T @ C @ B) * detJ
    return Ke


def _element_connectivity(N: int) -> np.ndarray:
    """(n_elem, 8) node indices for (N-1)^3 hexes, VTK/hex ordering."""
    ne = N - 1
    ii, jj, kk = np.meshgrid(
        np.arange(ne), np.arange(ne), np.arange(ne), indexing="ij"
    )
    i = ii.ravel()
    j = jj.ravel()
    k = kk.ravel()
    n000 = _node_index(i, j, k, N)
    n100 = _node_index(i + 1, j, k, N)
    n110 = _node_index(i + 1, j + 1, k, N)
    n010 = _node_index(i, j + 1, k, N)
    n001 = _node_index(i, j, k + 1, N)
    n101 = _node_index(i + 1, j, k + 1, N)
    n111 = _node_index(i + 1, j + 1, k + 1, N)
    n011 = _node_index(i, j + 1, k + 1, N)
    return np.stack([n000, n100, n110, n010, n001, n101, n111, n011], axis=1)


def _node_index(i, j, k, N: int):
    return i + N * (j + N * k)


def _dirichlet_dofs(N: int) -> np.ndarray:
    """All 3 DOFs on bottom face k=0."""
    nodes = np.arange(N * N, dtype=np.int64)  # k=0 slab
    dofs = np.empty(3 * nodes.size, dtype=np.int64)
    dofs[0::3] = 3 * nodes
    dofs[1::3] = 3 * nodes + 1
    dofs[2::3] = 3 * nodes + 2
    return dofs


def _top_traction_force(N: int, h: float, t0: float) -> np.ndarray:
    """Consistent nodal force for uniform traction (0,0,-t0) on top face."""
    ndof = 3 * N * N * N
    f = np.zeros(ndof, dtype=np.float64)
    contrib = -t0 * (h * h) / 4.0
    ne = N - 1
    k = N - 1
    ii, jj = np.meshgrid(np.arange(ne), np.arange(ne), indexing="ij")
    i = ii.ravel()
    j = jj.ravel()
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        n = _node_index(i + di, j + dj, k, N)
        f[3 * n + 2] += contrib
    return f


def assemble_K(E_nodes: np.ndarray, Ke0: np.ndarray, conn: np.ndarray) -> sp.csr_matrix:
    """Assemble global stiffness; E_elem = mean of 8 corner nodal E."""
    n_elem = conn.shape[0]
    E_elem = E_nodes.ravel()[conn].mean(axis=1)  # (n_elem,)
    ii_local, jj_local = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    ii_local = ii_local.ravel()
    jj_local = jj_local.ravel()
    Ke0_flat = Ke0.ravel()

    node_of_ldof = ii_local // 3
    comp_of_ldof_i = ii_local % 3
    node_of_jdof = jj_local // 3
    comp_of_ldof_j = jj_local % 3

    n_nodes = int(E_nodes.size)
    ndof = 3 * n_nodes

    # Large meshes: one-shot COO (peak ~ nnz_raw * 16B; N=128 ≈ 30GB) then sum
    # duplicates in tocsc. Mid-size: chunked CSR accumulate (lower peak).
    if n_elem > 250_000:
        # Chunked COO buffers then one tocsc (avoids one giant broadcast temp)
        chunk = 20000
        rows_l, cols_l, data_l = [], [], []
        for s in range(0, n_elem, chunk):
            e_idx = slice(s, min(s + chunk, n_elem))
            nodes = conn[e_idx]
            gi = 3 * nodes[:, node_of_ldof] + comp_of_ldof_i[None, :]
            gj = 3 * nodes[:, node_of_jdof] + comp_of_ldof_j[None, :]
            data = (E_elem[e_idx, None] * Ke0_flat[None, :]).ravel()
            rows_l.append(gi.ravel())
            cols_l.append(gj.ravel())
            data_l.append(data)
        rows = np.concatenate(rows_l).astype(np.int32, copy=False)
        cols = np.concatenate(cols_l).astype(np.int32, copy=False)
        data = np.concatenate(data_l)
        del rows_l, cols_l, data_l
        K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof), dtype=np.float64)
        del rows, cols, data
        return K.tocsr()

    chunk = 20000
    K = sp.csr_matrix((ndof, ndof), dtype=np.float64)
    for s in range(0, n_elem, chunk):
        e_idx = np.arange(s, min(s + chunk, n_elem))
        nodes = conn[e_idx]
        gi = 3 * nodes[:, node_of_ldof] + comp_of_ldof_i[None, :]
        gj = 3 * nodes[:, node_of_jdof] + comp_of_ldof_j[None, :]
        data = E_elem[e_idx, None] * Ke0_flat[None, :]
        Kc = sp.coo_matrix(
            (data.ravel(), (gi.ravel(), gj.ravel())), shape=(ndof, ndof)
        ).tocsr()
        K = K + Kc
    return K


def _element_matvec(
    x: np.ndarray,
    E_elem: np.ndarray,
    Ke0: np.ndarray,
    conn: np.ndarray,
) -> np.ndarray:
    """y = K(E) x without forming K (chunked element matvecs)."""
    y = np.zeros_like(x, dtype=np.float64)
    n_elem = conn.shape[0]
    chunk = 8000
    # local dof index helpers
    loc = np.arange(24)
    node_l = loc // 3
    comp_l = loc % 3
    for s in range(0, n_elem, chunk):
        e_idx = np.arange(s, min(s + chunk, n_elem))
        nodes = conn[e_idx]  # (c, 8)
        gdofs = 3 * nodes[:, node_l] + comp_l[None, :]  # (c, 24)
        ue = x[gdofs]  # (c, 24)
        fe = (ue @ Ke0.T) * E_elem[e_idx, None]  # (c, 24)
        np.add.at(y, gdofs.ravel(), fe.ravel())
    return y


def make_matrix_free_operator(
    E_nodes: np.ndarray,
    Ke0: np.ndarray,
    conn: np.ndarray,
    dirichlet_dofs: np.ndarray,
) -> Tuple[spla.LinearOperator, np.ndarray]:
    """LinearOperator for Dirichlet-reduced system + Jacobi diagonal estimate."""
    E_elem = E_nodes.ravel()[conn].mean(axis=1)
    ndof = 3 * int(E_nodes.size)
    dmask = np.zeros(ndof, dtype=bool)
    dmask[dirichlet_dofs] = True

    # Jacobi diag: sum_e E_e * diag(Ke0) scattered
    diag = np.zeros(ndof, dtype=np.float64)
    loc = np.arange(24)
    node_l = loc // 3
    comp_l = loc % 3
    ke_diag = np.diag(Ke0)
    chunk = 8000
    n_elem = conn.shape[0]
    for s in range(0, n_elem, chunk):
        e_idx = np.arange(s, min(s + chunk, n_elem))
        nodes = conn[e_idx]
        gdofs = 3 * nodes[:, node_l] + comp_l[None, :]
        np.add.at(diag, gdofs.ravel(), (E_elem[e_idx, None] * ke_diag[None, :]).ravel())
    diag[dmask] = 1.0
    diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)

    def matvec(x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=np.float64)
        x_c = x.copy()
        x_c[dmask] = 0.0  # homogeneous Dirichlet
        y = _element_matvec(x_c, E_elem, Ke0, conn)
        y[dmask] = x[dmask]
        return y

    A = spla.LinearOperator((ndof, ndof), matvec=matvec, dtype=np.float64)
    return A, diag


def apply_dirichlet(
    K: sp.csr_matrix, f: np.ndarray, dirichlet_dofs: np.ndarray
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Zero rows/cols of Dirichlet DOFs, put 1 on diagonal, RHS=0 there."""
    ndof = K.shape[0]
    free = np.ones(ndof, dtype=np.float64)
    free[dirichlet_dofs] = 0.0
    D = sp.diags(free, format="csr")
    # Zero rows and columns of constrained DOFs
    K = D @ K @ D
    # Unit diagonal on Dirichlet DOFs
    eye_d = sp.csr_matrix(
        (
            np.ones(dirichlet_dofs.size, dtype=np.float64),
            (dirichlet_dofs, dirichlet_dofs),
        ),
        shape=(ndof, ndof),
    )
    K = K + eye_d
    f = f.copy()
    f[dirichlet_dofs] = 0.0
    return K, f


def solve_elasticity(
    E_nodes: np.ndarray,
    *,
    nu: float,
    t0: float,
    Ke0: np.ndarray,
    conn: np.ndarray,
    dirichlet_dofs: np.ndarray,
    f_top: np.ndarray,
    solver: str = "auto",
    cg_tol: float = 1e-6,
    cg_maxiter: int = 8000,
) -> Tuple[np.ndarray, Dict]:
    """Solve one sample. Returns u shaped (3, N, N, N) and info dict."""
    N = E_nodes.shape[0]
    ndof = 3 * N * N * N
    f = f_top.copy()
    f[dirichlet_dofs] = 0.0

    if solver == "auto":
        # Direct tiny; PETSc+GAMG mid; scipy CG at N=128 (petsc4py createAIJ SEGV)
        if N <= 16:
            solver = "direct"
        elif N <= 96:
            solver = "petsc"
        else:
            solver = "cg"

    info: Dict = {"solver": solver, "ndof": ndof}

    if solver in ("direct", "cg"):
        t_asm0 = time.perf_counter()
        K = assemble_K(E_nodes, Ke0, conn)
        K, f = apply_dirichlet(K, f_top, dirichlet_dofs)
        info["assemble_s"] = time.perf_counter() - t_asm0
        t_sol0 = time.perf_counter()
        if solver == "direct":
            u = spla.spsolve(K, f)
            info["iters"] = 0
        else:
            # ILU fill explodes on 3D elasticity — Jacobi only for scipy CG
            diag = K.diagonal()
            diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
            M = spla.LinearOperator(K.shape, matvec=lambda x: x / diag)
            info["precond"] = "jacobi"
            niter = [0]

            def _cb(_xk):
                niter[0] += 1

            u, flag = spla.cg(K, f, M=M, rtol=cg_tol, maxiter=cg_maxiter, callback=_cb)
            info["iters"] = int(niter[0])
            info["cg_flag"] = int(flag)
            if flag != 0:
                raise RuntimeError(f"CG failed flag={flag} iters={niter[0]}")
        info["solve_s"] = time.perf_counter() - t_sol0
    elif solver == "petsc":
        u, pinfo = _solve_petsc_assembled(
            E_nodes, Ke0, conn, dirichlet_dofs, f_top,
            cg_tol=cg_tol, maxiter=cg_maxiter,
        )
        info.update(pinfo)
    elif solver == "mf_cg":
        t_asm0 = time.perf_counter()
        A, diag = make_matrix_free_operator(E_nodes, Ke0, conn, dirichlet_dofs)
        info["assemble_s"] = time.perf_counter() - t_asm0  # Jacobi setup
        info["precond"] = "jacobi_mf"
        M = spla.LinearOperator(A.shape, matvec=lambda x: x / diag)
        niter = [0]

        def _cb(_xk):
            niter[0] += 1

        t_sol0 = time.perf_counter()
        u, flag = spla.cg(A, f, M=M, rtol=cg_tol, maxiter=cg_maxiter, callback=_cb)
        info["solve_s"] = time.perf_counter() - t_sol0
        info["iters"] = int(niter[0])
        info["cg_flag"] = int(flag)
        if flag != 0:
            raise RuntimeError(f"mf_cg failed flag={flag} iters={niter[0]}")
    else:
        raise ValueError(f"Unknown solver={solver}")

    # DOF order: n = i + N*j + N*N*k, flat[3*n+c] => reshape as (k,j,i,c)
    u = (
        np.asarray(u, dtype=np.float64)
        .reshape(N, N, N, 3)
        .transpose(3, 2, 1, 0)
    )  # (comp, i, j, k)
    return u, info


def _solve_petsc_assembled(
    E_nodes: np.ndarray,
    Ke0: np.ndarray,
    conn: np.ndarray,
    dirichlet_dofs: np.ndarray,
    f_top: np.ndarray,
    *,
    cg_tol: float,
    maxiter: int,
) -> Tuple[np.ndarray, Dict]:
    """PETSc CG + GAMG/hypre. SciPy assemble; Dirichlet via PETSc zeroRowsColumns."""
    from petsc4py import PETSc

    t0 = time.perf_counter()
    K = assemble_K(E_nodes, Ke0, conn).tocsr()
    # petsc4py AIJ CSR path requires int32 indices (nnz at N=128 fits int32)
    K.indptr = np.asarray(K.indptr, dtype=np.int32)
    K.indices = np.asarray(K.indices, dtype=np.int32)
    A = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr, K.indices, K.data))
    A.assemble()
    del K
    f = np.asarray(f_top, dtype=np.float64).copy()
    dd = np.asarray(dirichlet_dofs, dtype=np.int32)
    A.zeroRowsColumns(dd, diag=1.0)
    f[dd] = 0.0
    assemble_s = time.perf_counter() - t0

    b = PETSc.Vec().createWithArray(f)
    x = A.createVecRight()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    precond = "gamg"
    try:
        pc.setType("gamg")
    except Exception:
        try:
            pc.setType("hypre")
            precond = "hypre"
        except Exception:
            pc.setType("jacobi")
            precond = "jacobi"
    ksp.setTolerances(rtol=cg_tol, max_it=maxiter)
    ksp.setFromOptions()
    t1 = time.perf_counter()
    ksp.solve(b, x)
    solve_s = time.perf_counter() - t1
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(f"PETSc KSP failed reason={reason}")
    u = np.array(x.getArray(), dtype=np.float64, copy=True)
    info = {
        "assemble_s": assemble_s,
        "solve_s": solve_s,
        "iters": int(ksp.getIterationNumber()),
        "precond": precond,
        "petsc_reason": int(reason),
    }
    A.destroy()
    return u, info



def _solve_petsc(K: sp.csr_matrix, f: np.ndarray, cg_tol: float, maxiter: int) -> np.ndarray:
    from petsc4py import PETSc

    A = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr, K.indices, K.data))
    A.assemble()
    b = PETSc.Vec().createWithArray(f.copy())
    x = PETSc.Vec().createSeq(f.size)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    try:
        pc.setType("gamg")
    except Exception:
        try:
            pc.setType("hypre")
        except Exception:
            pc.setType("ilu")
    ksp.setTolerances(rtol=cg_tol, max_it=maxiter)
    ksp.setFromOptions()
    ksp.solve(b, x)
    if ksp.getConvergedReason() < 0:
        raise RuntimeError(f"PETSc KSP failed reason={ksp.getConvergedReason()}")
    return np.array(x.getArray(), dtype=np.float64, copy=True)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def make_dataset(
    n_samples: int,
    N: int,
    *,
    seed: int = 0,
    nu: float = 0.3,
    t0: float = 1.0,
    n_kl: int = 8,
    length_scale: float = 0.35,
    E0: float = 1.0,
    sigma: float = 0.5,
    E_min: float = 0.2,
    E_max: float = 5.0,
    solver: str = "auto",
    out: Optional[str] = None,
    save_every: int = 0,
    resume: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    h = 1.0 / max(N - 1, 1)
    Ke0 = _hex_Ke0(h, nu)
    conn = _element_connectivity(N)
    ddofs = _dirichlet_dofs(N)
    f_top = _top_traction_force(N, h, t0)
    basis = _kl_basis(N, n_kl, length_scale)
    rng = np.random.default_rng(seed)

    meta = {
        "pde": "linear_elastostatics_3d",
        "N": N,
        "h": h,
        "nu": nu,
        "t0": t0,
        "seed": seed,
        "bc": "u=0 on z=0; traction (0,0,-t0) on z=1; free sides",
        "discretization": "trilinear_hex_fem",
        "solver": solver,
        "n_samples_target": n_samples,
    }

    n_done = 0
    X_done = Y_done = None
    if resume and out and os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        X_done, Y_done = z["X"], z["Y"]
        old = z["meta"].item()
        if int(old.get("N", -1)) != N:
            raise ValueError(f"resume N mismatch {old.get('N')} vs {N}")
        if int(old.get("seed", -1)) != seed:
            raise ValueError(f"resume seed mismatch {old.get('seed')} vs {seed}")
        n_done = int(X_done.shape[0])
        if n_done >= n_samples:
            print(f"elastostatics: already {n_done}/{n_samples} in {out}", flush=True)
            return X_done, Y_done, {**old, "n_samples": n_done}

    X = np.empty((n_samples, N, N, N), dtype=np.float32)
    Y = np.empty((n_samples, 3, N, N, N), dtype=np.float32)
    if n_done > 0:
        X[:n_done] = X_done
        Y[:n_done] = Y_done
        # burn RNG / KL draws already consumed
        sample_E_field(
            n_done, N, n_kl=n_kl, length_scale=length_scale, E0=E0,
            sigma=sigma, E_min=E_min, E_max=E_max, rng=rng, basis=basis,
        )

    t0_wall = time.time()
    save_every = int(save_every) if save_every and save_every > 0 else n_samples
    for i in range(n_done, n_samples):
        E, e_meta = sample_E_field(
            1, N, n_kl=n_kl, length_scale=length_scale, E0=E0,
            sigma=sigma, E_min=E_min, E_max=E_max, rng=rng, basis=basis,
        )
        if i == n_done:
            meta.update(e_meta)
        u, sinfo = solve_elasticity(
            E[0], nu=nu, t0=t0, Ke0=Ke0, conn=conn,
            dirichlet_dofs=ddofs, f_top=f_top, solver=solver,
        )
        X[i] = E[0].astype(np.float32)
        Y[i] = u.astype(np.float32)
        if (i + 1) % 1 == 0:
            print(
                f"sample {i+1}/{n_samples}  "
                f"|u|_max={np.abs(u).max():.3e}  "
                f"asm={sinfo['assemble_s']:.2f}s solve={sinfo['solve_s']:.2f}s "
                f"solver={sinfo['solver']} iters={sinfo.get('iters')}  "
                f"elapsed={time.time()-t0_wall:.1f}s",
                flush=True,
            )
        if out and ((i + 1) % save_every == 0 or (i + 1) == n_samples):
            meta["n_samples"] = i + 1
            _atomic_savez(
                out,
                X=X[: i + 1],
                Y=Y[: i + 1],
                meta=np.array(meta, dtype=object),
            )
            print(f"checkpoint {i+1}/{n_samples} -> {out}", flush=True)

    meta["n_samples"] = n_samples
    return X, Y, meta


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-N", type=int, default=32, help="nodes per axis (N^3 grid)")
    p.add_argument("-n", "--n-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--t0", type=float, default=1.0, help="top traction magnitude")
    p.add_argument("--n-kl", type=int, default=8, help="KL modes for log-E")
    p.add_argument("--length-scale", type=float, default=0.35)
    p.add_argument("--E0", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.5, help="log-E std scale")
    p.add_argument("--E-min", type=float, default=0.2)
    p.add_argument("--E-max", type=float, default=5.0)
    p.add_argument(
        "--solver",
        choices=("auto", "direct", "cg", "mf_cg", "petsc"),
        default="auto",
    )
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out = args.out or f"elastostatics3d_N{args.N}_n{args.n_samples}.npz"
    t0 = time.time()
    X, Y, meta = make_dataset(
        args.n_samples,
        args.N,
        seed=args.seed,
        nu=args.nu,
        t0=args.t0,
        n_kl=args.n_kl,
        length_scale=args.length_scale,
        E0=args.E0,
        sigma=args.sigma,
        E_min=args.E_min,
        E_max=args.E_max,
        solver=args.solver,
        out=out,
        save_every=args.save_every,
        resume=args.resume,
    )
    if not (args.resume and os.path.exists(out) and X.shape[0] >= args.n_samples):
        # final save already done inside make_dataset when out set; ensure written
        meta["n_samples"] = int(X.shape[0])
        _atomic_savez(out, X=X, Y=Y, meta=np.array(meta, dtype=object))
    dt = time.time() - t0
    print(
        f"elastostatics: X{X.shape} Y{Y.shape}  "
        f"E[{X.min():.3g},{X.max():.3g}]  "
        f"|u|_max={np.abs(Y).max():.3g}  {dt:.1f}s -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
