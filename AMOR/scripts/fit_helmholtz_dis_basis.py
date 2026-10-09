#!/usr/bin/env python3
"""
Faithful (Roseberry-style) derivative-informed subspace (DIS) basis for the
Helmholtz parametric map  m -> u = (u_real, u_imag), using the TRUE FEM Jacobian
exposed by the hippylib PDE problem (no surrogate).

Default input metric: Zahm option-2 prior_cov (GEVP Hφ = λ C^{-1} φ) on the
extraction grid. Output probes are J z with z = C^{1/2} ξ (same low-rank prior),
then Euclidean SVD on the state DoFs and extraction to the grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import argparse
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", f"dolfin_cache_{os.environ.get('USER','dv')}"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

_AMOR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_AMOR_ROOT, "data")
sys.path.insert(0, DATA_DIR)

import dolfin as dl  # noqa: E402
import hippylib as hp  # noqa: E402
from mpi4py import MPI  # noqa: E402
from helmholtz_pde import (  # noqa: E402
    setup_helmholtz_problem,
    extract_to_grid,
    extract_vector_to_grid,
    grid_to_parameter_vector,
)
from AMOR.models.prior_cov import LowRankPriorCov  # noqa: E402

SETTINGS = {
    "sigma_0.2_rho_0.4": dict(seed=0, nx=100, frequency=600, sigma=0.2, rho=0.4,
                   theta0=1.0, theta1=1.0, alpha=0.25 * math.pi, comm_mesh=MPI.COMM_SELF),
    "sigma_0.4_rho_0.2": dict(seed=0, nx=100, frequency=600, sigma=0.4, rho=0.2,
                     theta0=1.0, theta1=1.0, alpha=0.25 * math.pi, comm_mesh=MPI.COMM_SELF),
    "sigma_0.1_rho_0.6": dict(seed=0, nx=100, frequency=600, sigma=0.1, rho=0.6,
                     theta0=1.0, theta1=1.0, alpha=0.25 * math.pi, comm_mesh=MPI.COMM_SELF),
}
BOX = [0.0, 0.0, 3.0, 3.0]


class HelmholtzJacobian:
    """True FEM Jacobian actions J v and J^T w at a fixed linearization point m."""

    def __init__(self, pde):
        self.pde = pde

    def relinearize(self, m):
        u = self.pde.generate_state()
        self.pde.solveFwd(u, [u, m, None])
        p = self.pde.generate_state()
        self.pde.setLinearizationPoint([u, m, p], gn_approx=True)

    def Jv(self, v):  # param -> state:  du = -A^{-1} C v
        rhs = self.pde.generate_state()
        self.pde.C.mult(v, rhs)
        rhs *= -1.0
        du = self.pde.generate_state()
        self.pde.solveIncremental(du, rhs, False)
        return du

    def Jtw(self, w):  # state -> param:  -C^T A^{-T} w
        p = self.pde.generate_state()
        self.pde.solveIncremental(p, w, True)
        out = self.pde.generate_parameter()
        self.pde.C.transpmult(p, out)
        out *= -1.0
        return out


def _set_param(pde, arr):
    v = pde.generate_parameter(); v.set_local(np.ascontiguousarray(arr)); v.apply(""); return v


def _set_state(pde, arr):
    v = pde.generate_state(); v.set_local(np.ascontiguousarray(arr)); v.apply(""); return v


def collect_and_svd(
    jac,
    pde,
    prior,
    k,
    n_samples,
    n_probes,
    rng,
    *,
    m_grids=None,
    Vh=None,
    box=None,
    nx: int = 100,
    prior_cov: LowRankPriorCov | None = None,
    x_std_flat: np.ndarray | None = None,
):
    """Monte-Carlo probe estimator of E_m[J^T J] and E_m[J J^T].

    If prior_cov: extract J^T probes to grid, apply C^{1/2} (after optional z-score
    scale), SVD in grid space → rows_w; caller maps φ = C^{1/2} ψ.
    Else: SVD in DoF space (legacy), then extract.
    """
    from AMOR.models.dis_mc_svd import ProbeMemmapStack

    d_m = pde.generate_parameter().get_local().shape[0]
    d_u = pde.generate_state().get_local().shape[0]
    d_grid = int(nx * nx)
    if m_grids is not None:
        n_samples = int(m_grids.shape[0])
    n_tot = n_samples * n_probes
    scale = 1.0 / math.sqrt(max(1, n_tot))
    tmp = os.environ.get("DIS_MEMMAP_DIR", f"/tmp/helm_dis_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    d_in_stack = d_grid if prior_cov is not None else d_m
    stack_in = ProbeMemmapStack(os.path.join(tmp, "A.npy"), n_tot, d_in_stack)
    stack_out = ProbeMemmapStack(os.path.join(tmp, "B.npy"), n_tot, d_u)
    x_std = None if x_std_flat is None else np.asarray(x_std_flat, dtype=np.float64).reshape(-1)
    t = time.time()
    try:
        for s in range(n_samples):
            if m_grids is not None:
                m = grid_to_parameter_vector(Vh[hp.PARAMETER], m_grids[s], box)
            else:
                noise = dl.Vector(); prior.init_vector(noise, "noise")
                noise.set_local(np.random.randn(noise.local_size())); noise.apply("")
                m = pde.generate_parameter(); prior.init_vector(m, 0); prior.sample(noise, m)
            jac.relinearize(m)
            for _ in range(n_probes):
                g = _set_state(pde, rng.standard_normal(d_u))
                a_dof = jac.Jtw(g).get_local()
                if prior_cov is not None:
                    f = hp.vector2Function(_set_param(pde, a_dof), Vh[hp.PARAMETER])
                    a_grid = extract_to_grid(f, box, nx, nx).ravel().astype(np.float64)
                    if x_std is not None:
                        a_grid = a_grid * x_std
                    a_w = prior_cov.apply_sqrt(a_grid)
                    stack_in.append(a_w * scale)
                else:
                    stack_in.append(a_dof * scale)
                # Output: J z, z = C^{1/2} ξ in the same z-scored grid prior as the input.
                if prior_cov is not None:
                    z = prior_cov.apply_sqrt(rng.standard_normal(prior_cov.dim))
                    if x_std is not None and np.asarray(x_std).size == z.size:
                        z = z * np.asarray(x_std, dtype=np.float64).reshape(-1)
                    v = grid_to_parameter_vector(
                        Vh[hp.PARAMETER], z.reshape(nx, nx), box
                    )
                    stack_out.append(jac.Jv(v).get_local() * scale)
                else:
                    h = _set_param(pde, rng.standard_normal(d_m))
                    stack_out.append(jac.Jv(h).get_local() * scale)
            if (s + 1) % 8 == 0 or (s + 1) == n_samples:
                print(f"  sample {s+1}/{n_samples} ({time.time()-t:.1f}s)", flush=True)
        rows_in, ev_in = stack_in.finalize_svd(k, random_state=int(rng.integers(0, 2**31 - 1)))
        rows_out, ev_out = stack_out.finalize_svd(k, random_state=int(rng.integers(0, 2**31 - 1)))
    finally:
        stack_in.close(delete=True)
        stack_out.close(delete=True)
    return rows_in.astype(np.float32), ev_in, rows_out.astype(np.float32), ev_out


def _orthonormal_rows(rows):
    q, _ = np.linalg.qr(np.asarray(rows, dtype=np.float64).T, mode="reduced")
    return q.T.astype(np.float32)


VARIANT_DATA = {
    "sigma_0.2_rho_0.4": expand("${AMOR_SCRATCH_DATA}/helmholtz_sigma_0.2_rho_0.4.npz"),  # rename path on a different cluster
    "sigma_0.4_rho_0.2": expand("${AMOR_SCRATCH_DATA}/helmholtz_sigma_0.4_rho_0.2.npz"),  # rename path on a different cluster
    "sigma_0.1_rho_0.6": expand("${AMOR_SCRATCH_DATA}/helmholtz_sigma_0.1_rho_0.6.npz"),  # rename path on a different cluster
}


def main():
    ap = argparse.ArgumentParser(description="Fit faithful Helmholtz DIS basis (true FEM Jacobian)")
    ap.add_argument("--variant", choices=list(SETTINGS), default="sigma_0.2_rho_0.4")
    ap.add_argument("--k_in", type=int, default=144)
    ap.add_argument("--k_out", type=int, default=144)
    ap.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="MC linearization points; <=0 = ALL dataset m (default). "
        "With --prior_sample, draws this many fresh prior samples instead.",
    )
    ap.add_argument("--n_probes", type=int, default=6, help="random probes per sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_path", default=None, help="override Helmholtz NPZ path")
    ap.add_argument(
        "--prior_sample",
        action="store_true",
        help="legacy: sample m from GP prior instead of dataset states",
    )
    ap.add_argument(
        "--input_metric",
        choices=("prior_cov", "identity"),
        default="prior_cov",
    )
    ap.add_argument("--cov_rank", type=int, default=128)
    ap.add_argument("--cov_jitter_frac", type=float, default=1e-4)
    ap.add_argument("--encoder", choices=("sigma", "cinv"), default="sigma")
    ap.add_argument("--sigma_states", type=int, default=48, help="states for ΣΦ average")
    ap.add_argument(
        "--prior",
        choices=("exact", "lowrank"),
        default="exact",
        help="exact hippylib BiLaplacian C vs low-rank+jitter sample cov",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate operator/signs")
    args = ap.parse_args()

    if args.smoke:
        args.k_in, args.k_out, args.n_samples, args.n_probes = 4, 4, 2, 4
        args.prior_sample = True
        if args.input_metric == "prior_cov":
            # smoke with prior samples has no dataset m — fall back to identity
            args.input_metric = "identity"
            print("smoke: forcing --input_metric identity (no dataset m for C)", flush=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    settings = SETTINGS[args.variant]
    print(f"Setting up Helmholtz '{args.variant}' problem...", flush=True)
    Vh, pde, prior, _ = setup_helmholtz_problem(settings)
    jac = HelmholtzJacobian(pde)
    nx = settings["nx"]

    m_grids = None
    prior_cov = None
    x_std_flat = None
    if not args.prior_sample:
        data_path = args.data_path or VARIANT_DATA[args.variant]
        d = np.load(data_path, mmap_mode="r")
        n_all = int(d["m"].shape[0])
        n_use = n_all if args.n_samples is None or args.n_samples <= 0 else min(args.n_samples, n_all)
        m_grids = np.asarray(d["m"][:n_use], dtype=np.float64)
        n_samples = n_use
        print(f"Using dataset m from {data_path}: {n_use}/{n_all} states (no prior subsample)", flush=True)
        if args.input_metric == "prior_cov":
            # MC probe whitening always uses fast low-rank sample cov on the grid.
            # Exact hippylib prior is applied later for Φ/E (k modes), not 60k probes.
            ntr = int(0.7 * n_all)
            m_tr = np.asarray(d["m"][:ntr], dtype=np.float64)
            Xm = m_tr.mean(axis=(0, 1, 2), keepdims=True)
            Xs = m_tr.std(axis=(0, 1, 2), keepdims=True) + 1e-8
            m_z = ((m_tr - Xm) / Xs).reshape(ntr, -1)
            x_std_flat = np.full(nx * nx, float(Xs.reshape(-1)[0]), dtype=np.float64)
            print(
                f"Fitting LowRankPriorCov on z-scored m rank={args.cov_rank} "
                f"shape={m_z.shape} (probe whitening)",
                flush=True,
            )
            prior_cov = LowRankPriorCov.from_samples(
                m_z,
                rank=args.cov_rank,
                jitter_frac=args.cov_jitter_frac,
                random_state=args.seed,
            )
    else:
        n_samples = 48 if args.n_samples is None or args.n_samples <= 0 else args.n_samples
        print(f"Using {n_samples} fresh prior samples (--prior_sample)", flush=True)
        if args.input_metric == "prior_cov":
            raise SystemExit("--input_metric prior_cov needs dataset m (drop --prior_sample)")

    ntot = n_samples * args.n_probes
    kmax = max(args.k_in, args.k_out)
    if ntot < kmax:
        raise SystemExit(f"n_samples*n_probes={ntot} < max(k_in,k_out)={kmax}; increase probing")

    print(
        f"Collecting DIS ({n_samples} samples x {args.n_probes} probes = {ntot}, "
        f"metric={args.input_metric})...",
        flush=True,
    )
    os.environ.setdefault(
        "DIS_MEMMAP_DIR",
        expand(f"${{AMOR_SCRATCH_DATA}}/helmholtz_{args.variant}_basis/_dis_scratch"),  # rename path on a different cluster
    )
    rows_in_raw, ev_in, rows_out_dof, ev_out = collect_and_svd(
        jac, pde, prior, kmax, n_samples, args.n_probes, rng,
        m_grids=m_grids, Vh=Vh, box=BOX, nx=nx,
        prior_cov=prior_cov, x_std_flat=x_std_flat,
    )

    from AMOR.models.dis_dual import (  # noqa: E402
        assert_dual,
        build_dual_encoder_from_sigma,
        encoder_from_cinv,
        phi_from_psi,
    )

    if prior_cov is not None:
        psi = np.asarray(rows_in_raw[: args.k_in], dtype=np.float64)
        ev_in = ev_in[: args.k_in]
        # Φ/E must use the same C as probe whitening. Mixing exact hippylib C with
        # lowrank-whitened ψ breaks E Φᵀ ≈ I (observed ||EΦᵀ-I||~1e4). Exact prior
        # wrapper remains available for DoF-space callers; grid DIS uses lowrank here.
        cov_for_phi = prior_cov
        if args.prior == "exact":
            print(
                "note: --prior exact requested but probes used LowRankPriorCov; "
                "Φ/E use the same lowrank C for dual consistency "
                "(GridMappedExactPrior kept for DoF-space use)",
                flush=True,
            )
        phi = phi_from_psi(psi, cov_for_phi)
        lam = np.asarray(ev_in, dtype=np.float64).reshape(-1)[: phi.shape[0]]

        def _jtj_avg_grid(phi_flat: np.ndarray) -> np.ndarray:
            """Average J^* J φ on the extraction grid (z-score / physical as in probes)."""
            phi_g = np.asarray(phi_flat, dtype=np.float64).reshape(nx, nx)
            # Map grid φ → DoF, apply J^* J, extract back.
            n_use = n_samples if args.sigma_states <= 0 else min(args.sigma_states, n_samples)
            acc = np.zeros(nx * nx, dtype=np.float64)
            for s in range(n_use):
                if m_grids is not None:
                    m = grid_to_parameter_vector(Vh[hp.PARAMETER], m_grids[s], BOX)
                else:
                    noise = dl.Vector()
                    prior.init_vector(noise, "noise")
                    noise.set_local(np.random.randn(noise.local_size()))
                    noise.apply("")
                    m = pde.generate_parameter()
                    prior.init_vector(m, 0)
                    prior.sample(noise, m)
                jac.relinearize(m)
                # grid → dof for Jv direction
                v = grid_to_parameter_vector(Vh[hp.PARAMETER], phi_g, BOX)
                du = jac.Jv(v)
                a_dof = jac.Jtw(du).get_local()
                f = hp.vector2Function(_set_param(pde, a_dof), Vh[hp.PARAMETER])
                a_grid = extract_to_grid(f, BOX, nx, nx).ravel().astype(np.float64)
                if x_std_flat is not None:
                    a_grid = a_grid * np.asarray(x_std_flat, dtype=np.float64).reshape(-1)
                acc += a_grid
            return acc / float(n_use)

        if args.encoder == "sigma":
            print("Building Helmholtz E = Λ^{-1}(Σ Φ)ᵀ …", flush=True)
            rows_enc_in = build_dual_encoder_from_sigma(
                phi, lam, _jtj_avg_grid, name="helm DIS_in sigma", atol=0.5
            )
            err = float(np.linalg.norm(rows_enc_in @ phi.T - np.eye(phi.shape[0])))
            if err > 0.5:
                print(f"ΣΦ residual {err:.3e} → fallback E=Φᵀ C^{-1}", flush=True)
                rows_enc_in = encoder_from_cinv(phi, cov_for_phi)
                assert_dual(rows_enc_in, phi, name="helm DIS_in cinv_fallback")
                dis_version = "dual_encoder_cinv_fallback"
            else:
                dis_version = "dual_sigma"
        else:
            rows_enc_in = encoder_from_cinv(phi, cov_for_phi)
            assert_dual(rows_enc_in, phi, name="helm DIS_in cinv")
            dis_version = "dual_encoder_cinv"
        rows_in = phi.astype(np.float32)
        mass_whiten = "none"
    else:
        rows_in_dof, ev_in = rows_in_raw[: args.k_in], ev_in[: args.k_in]
        print("Extracting input eigenvectors to grid...", flush=True)
        rows_in_list = []
        for r in rows_in_dof:
            f = hp.vector2Function(_set_param(pde, r), Vh[hp.PARAMETER])
            g = extract_to_grid(f, BOX, nx, nx)
            rows_in_list.append(g.ravel())
        rows_in = _orthonormal_rows(np.stack(rows_in_list, 0))
        rows_enc_in = rows_in
        dis_version = "euclidean_dual"
        mass_whiten = "none"

    rows_out_dof, ev_out = rows_out_dof[: args.k_out], ev_out[: args.k_out]
    print("Extracting output eigenvectors to grid...", flush=True)
    rows_out = []
    for r in rows_out_dof:
        f = hp.vector2Function(_set_state(pde, r), Vh[hp.STATE])
        gr, gi = extract_vector_to_grid(f, BOX, nx, nx)
        rows_out.append(np.concatenate([gr.ravel(), gi.ravel()]))
    # Output: M_out = I on the extraction grid ⇒ Euclidean M-orthonormal = QR.
    rows_out = _orthonormal_rows(np.stack(rows_out, 0))
    rows_enc_out = np.asarray(rows_out, dtype=np.float32)  # M=I ⇒ E=Φ

    out = args.out or expand(
        f"${{AMOR_SCRATCH_DATA}}/helmholtz_{args.variant}_basis/dis_rows_{args.variant}_dual_sigma.npz"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = dict(
        rows_in=rows_in,
        rows_enc_in=np.asarray(rows_enc_in, dtype=np.float32),
        rows_out=rows_out,
        rows_enc_out=rows_enc_out,
        eigvals_in=ev_in,
        eigvals_out=ev_out,
        nx=nx,
        c_in=1,
        c_out=2,
        variant=args.variant,
        n_samples=n_samples,
        n_probes=args.n_probes,
        from_dataset=bool(m_grids is not None),
        input_metric=np.asarray(args.input_metric),
        ortho_metric=np.asarray("C_inv" if prior_cov is not None else "I"),
        mass_whiten=np.asarray(mass_whiten),
        dis_version=np.asarray(dis_version),
        prior_kind=np.asarray(args.prior if prior_cov is not None else "none"),
        output_probe=np.asarray("C_sqrt_xi" if prior_cov is not None else "standard_normal"),
    )
    if prior_cov is not None and hasattr(prior_cov, "save_dict"):
        payload.update(prior_cov.save_dict())
    if x_std_flat is not None:
        payload["x_std_flat"] = np.asarray(x_std_flat, dtype=np.float32)
    np.savez_compressed(out, **payload)
    print(f"rows_in(Φ) {rows_in.shape} rows_enc_in(E) {np.asarray(rows_enc_in).shape} rows_out {rows_out.shape}", flush=True)
    print(f"eig_in[:5]={ev_in[:5]}  eig_out[:5]={ev_out[:5]}  version={dis_version}", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
