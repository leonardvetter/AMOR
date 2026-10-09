#!/usr/bin/env python3
"""Fit Roseberry DIS rows using the FiPy Jacobian from generate_nonlinear_diffusion.py.

Default: Zahm option-2 prior_cov. Dual encoder E = Λ^{-1}(Σ Φ)ᵀ from averaged J^*J.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AMOR.paths import expand  # noqa: E402
from AMOR.models.nonlinear_diffusion_jacobian import (  # noqa: E402
    NonlinearDiffusionJacobian,
    fit_dis_rows_from_data,
)
from AMOR.models.dis_basis_utils import channel_std_flat, scale_dis_rows_to_normalized  # noqa: E402
from AMOR.models.dis_mc_svd import resolve_n_states  # noqa: E402
from AMOR.models.prior_cov import LowRankPriorCov  # noqa: E402
from AMOR.models.dis_dual import (  # noqa: E402
    assert_dual,
    build_dual_encoder_from_sigma,
    encoder_from_cinv,
    m_orthonormalize_rows,
    phi_from_psi,
)


def _make_jtj_avg_nl(
    f_dis: np.ndarray,
    m_dis: np.ndarray,
    n: int,
    L: float,
    x_std_flat: np.ndarray,
    max_states: int,
    seed: int,
):
    """Return apply_jtj(φ) = avg_i J_i^* J_i φ in z-scored (f,m) coordinates."""
    rng = np.random.default_rng(seed + 17)
    ntr = f_dis.shape[0]
    s = min(int(max_states), ntr) if max_states > 0 else ntr
    idx = np.arange(ntr) if s == ntr else rng.choice(ntr, size=s, replace=False)
    jac = NonlinearDiffusionJacobian(n, L=L)
    x_std = np.asarray(x_std_flat, dtype=np.float64).reshape(-1)
    d = n * n

    def apply_jtj(phi_flat: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi_flat, dtype=np.float64).reshape(-1)
        # z-score φ → physical input direction for J
        phi_phys = phi / np.maximum(x_std, 1e-30)
        vf, vm = phi_phys[:d], phi_phys[d:]
        acc = np.zeros_like(phi)
        for i in idx:
            jac.set_state(f_dis[i], m_dis[i].reshape(n, n))
            ju = jac.Jv(vf, vm)
            gf, gm = jac.Jtw(ju)
            a_phys = np.concatenate([gf, gm])
            acc += a_phys * x_std  # physical adjoint → z-score
        return acc / float(len(idx))

    return apply_jtj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=expand("${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz"))
    ap.add_argument("--num_samples", type=int, default=-1)
    ap.add_argument("--k_in", type=int, default=144)
    ap.add_argument("--k_out", type=int, default=144)
    ap.add_argument("--n_states", type=int, default=-1)
    ap.add_argument("--n_probes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memmap_dir", default=None)
    ap.add_argument("--input_metric", choices=("prior_cov", "identity"), default="prior_cov")
    ap.add_argument("--cov_rank", type=int, default=128)
    ap.add_argument("--cov_jitter_frac", type=float, default=1e-4)
    ap.add_argument("--sigma_states", type=int, default=64, help="states for ΣΦ average (<=0 = all train)")
    ap.add_argument("--encoder", choices=("sigma", "cinv"), default="sigma")
    ap.add_argument("--out", default=expand("${AMOR_DATA_ROOT}/nonlinear_diffusion_basis/dis_rows_dual_sigma.npz"))
    args = ap.parse_args()

    d = np.load(args.data_path, mmap_mode="r")
    n_all = int(d["f"].shape[0])
    n_load = n_all if args.num_samples is None or args.num_samples <= 0 else min(args.num_samples, n_all)
    f = np.asarray(d["f"][:n_load], dtype=np.float64).reshape(n_load, -1)
    m = np.asarray(d["m"][:n_load], dtype=np.float64).reshape(n_load, -1)
    u = np.asarray(d["u"][:n_load], dtype=np.float64)
    n = int(d["N"]) if "N" in d.files else int(round(np.sqrt(f.shape[1])))
    L = float(d["L"]) if "L" in d.files else 1.0

    ntr = int(0.7 * n_load)
    X = np.stack([f.reshape(n_load, n, n), m.reshape(n_load, n, n)], axis=1)
    Y = u[:, np.newaxis, :, :]
    x_std = X[:ntr].std((0, 2, 3), keepdims=True) + 1e-8
    y_std = Y[:ntr].std((0, 2, 3), keepdims=True) + 1e-8
    x_std_flat = channel_std_flat(x_std, 2, n * n)
    y_std_flat = channel_std_flat(y_std, 1, n * n)

    f_dis, m_dis = f[:ntr], m[:ntr]
    n_states = resolve_n_states(ntr, args.n_states)

    prior_cov = None
    if args.input_metric == "prior_cov":
        Xm = X[:ntr].mean((0, 2, 3), keepdims=True)
        X_z = ((X[:ntr] - Xm) / x_std).reshape(ntr, -1)
        print(f"Fitting LowRankPriorCov on z-scored X_z={X_z.shape} rank={args.cov_rank}", flush=True)
        prior_cov = LowRankPriorCov.from_samples(
            X_z, rank=args.cov_rank, jitter_frac=args.cov_jitter_frac, random_state=args.seed
        )

    print(
        f"Fitting DIS: N={n}, L={L}, k_in={args.k_in}, k_out={args.k_out}, "
        f"loaded={n_load}/{n_all}, train_states={n_states}/{ntr}, probes={args.n_probes}, "
        f"metric={args.input_metric}, encoder={args.encoder}",
        flush=True,
    )
    if n_states * args.n_probes < max(args.k_in, args.k_out):
        raise SystemExit(f"n_states*n_probes={n_states * args.n_probes} < max(k_in,k_out)")

    memmap_dir = args.memmap_dir or os.environ.get(
        "DIS_MEMMAP_DIR", expand("${AMOR_DATA_ROOT}/nonlinear_diffusion_basis/_dis_scratch")
    )
    rows_in, ev_in, rows_out, ev_out = fit_dis_rows_from_data(
        f_dis, m_dis, args.k_in, args.k_out, n_states, args.n_probes, args.seed,
        L=L, memmap_dir=memmap_dir,
        prior_cov=prior_cov,
        x_std_flat=x_std_flat if prior_cov is not None else None,
    )

    if prior_cov is None:
        rows_in, rows_out = scale_dis_rows_to_normalized(
            rows_in, rows_out, x_std_flat=x_std_flat, y_std_flat=y_std_flat
        )
        rows_enc_in = rows_in
        rows_enc_out = rows_out
        ortho_metric = "I"
        dis_version = "euclidean_dual"
    else:
        psi = np.asarray(rows_in, dtype=np.float64)
        phi = phi_from_psi(psi, prior_cov)
        lam = np.asarray(ev_in, dtype=np.float64).reshape(-1)[: phi.shape[0]]
        if args.encoder == "sigma":
            apply_jtj = _make_jtj_avg_nl(
                f_dis, m_dis, n, L, x_std_flat, args.sigma_states, args.seed
            )
            print("Building E = Λ^{-1}(Σ Φ)ᵀ from averaged J^*J …", flush=True)
            rows_enc_in = build_dual_encoder_from_sigma(
                phi, lam, apply_jtj, name="DIS_in sigma"
            )
            # Fallback if dual residual is large: C^{-1} encoder
            err = assert_dual(rows_enc_in, phi, atol=1e9, name="check")
            if err > 0.5:
                print(f"ΣΦ dual residual {err:.3e} large → fallback E=Φᵀ C^{-1}", flush=True)
                rows_enc_in = encoder_from_cinv(phi, prior_cov)
                assert_dual(rows_enc_in, phi, name="DIS_in cinv_fallback")
                dis_version = "dual_encoder_cinv_fallback"
            else:
                dis_version = "dual_sigma"
        else:
            rows_enc_in = encoder_from_cinv(phi, prior_cov)
            assert_dual(rows_enc_in, phi, name="DIS_in cinv")
            dis_version = "dual_encoder_cinv"
        rows_in = phi.astype(np.float32)
        rows_out = rows_out / np.maximum(y_std_flat.reshape(1, -1), 1e-30)
        rows_out = m_orthonormalize_rows(rows_out)  # M_out = I
        rows_enc_out = np.asarray(rows_out, dtype=np.float32)  # M-adjoint = Φ when M=I
        ortho_metric = "C_inv"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = dict(
        rows_in=np.asarray(rows_in, dtype=np.float32),
        rows_enc_in=np.asarray(rows_enc_in, dtype=np.float32),
        rows_out=np.asarray(rows_out, dtype=np.float32),
        rows_enc_out=np.asarray(rows_enc_out, dtype=np.float32),
        eigvals_in=ev_in,
        eigvals_out=ev_out,
        nx=n,
        c_in=2,
        c_out=1,
        n_states=n_states,
        n_probes=args.n_probes,
        rows_normalized=True,
        x_std_flat=x_std_flat.astype(np.float32),
        y_std_flat=y_std_flat.astype(np.float32),
        input_metric=np.asarray(args.input_metric),
        ortho_metric=np.asarray(ortho_metric),
        mass_whiten=np.asarray("none"),
        dis_version=np.asarray(dis_version),
        output_probe=np.asarray("C_sqrt_xi" if prior_cov is not None else "standard_normal"),
    )
    if prior_cov is not None:
        payload.update(prior_cov.save_dict())
    np.savez_compressed(args.out, **payload)
    print(
        f"Saved {args.out}  Φ={np.asarray(rows_in).shape} E={np.asarray(rows_enc_in).shape} "
        f"out={np.asarray(rows_out).shape} version={dis_version}",
        flush=True,
    )


if __name__ == "__main__":
    main()
