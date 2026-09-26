#!/usr/bin/env python3
"""Fit Roseberry DIS rows using the FiPy Jacobian from generate_nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.py.

Default: Zahm option-2 prior_cov (GEVP Hφ=λ C^{-1}φ). Use --input_metric identity for legacy.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path as _PathForAmor

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.paths import expand  # noqa: E402

from AMOR.models.nonlinear_diffusion_jacobian import fit_dis_rows_from_data  # noqa: E402
from AMOR.models.dis_basis_utils import channel_std_flat, scale_dis_rows_to_normalized  # noqa: E402
from AMOR.models.dis_mc_svd import resolve_n_states  # noqa: E402
from AMOR.models.prior_cov import LowRankPriorCov  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=expand("${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz"))  # rename path on a different cluster
    ap.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Load first N samples from NPZ; <=0 = entire file (default)",
    )
    ap.add_argument("--k_in", type=int, default=144)
    ap.add_argument("--k_out", type=int, default=144)
    ap.add_argument(
        "--n_states",
        type=int,
        default=-1,
        help="DIS linearization states among loaded samples; <=0 = ALL (default)",
    )
    ap.add_argument("--n_probes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memmap_dir", default=None)
    ap.add_argument(
        "--input_metric",
        choices=("prior_cov", "identity"),
        default="prior_cov",
    )
    ap.add_argument("--cov_rank", type=int, default=128)
    ap.add_argument("--cov_jitter_frac", type=float, default=1e-4)
    ap.add_argument("--out", default=expand("${AMOR_DATA_ROOT}/nonlinear_diffusion_basis/dis_rows.npz"))
    args = ap.parse_args()

    d = np.load(args.data_path, mmap_mode="r")
    n_all = int(d["f"].shape[0])
    n_load = n_all if args.num_samples is None or args.num_samples <= 0 else min(args.num_samples, n_all)
    f = np.asarray(d["f"][:n_load], dtype=np.float64).reshape(n_load, -1)
    m = np.asarray(d["m"][:n_load], dtype=np.float64).reshape(n_load, -1)
    u = np.asarray(d["u"][:n_load], dtype=np.float64)
    n = int(d["N"]) if "N" in d.files else int(round(np.sqrt(f.shape[1])))
    L = float(d["L"]) if "L" in d.files else 1.0

    n_states = resolve_n_states(n_load, args.n_states)
    ntr = int(0.7 * n_load)
    X = np.stack([f.reshape(n_load, n, n), m.reshape(n_load, n, n)], axis=1)
    Y = u[:, np.newaxis, :, :]
    x_std = X[:ntr].std((0, 2, 3), keepdims=True) + 1e-8
    y_std = Y[:ntr].std((0, 2, 3), keepdims=True) + 1e-8
    x_std_flat = channel_std_flat(x_std, 2, n * n)
    y_std_flat = channel_std_flat(y_std, 1, n * n)

    # DIS probes on train only (match PCA / avoid leakage into val/test).
    f_dis, m_dis = f[:ntr], m[:ntr]
    n_states = resolve_n_states(ntr, args.n_states)

    prior_cov = None
    if args.input_metric == "prior_cov":
        # C on z-scored train (f,m) — match AMOR trainer: (X - mean) / std
        Xm = X[:ntr].mean((0, 2, 3), keepdims=True)
        X_z = ((X[:ntr] - Xm) / x_std).reshape(ntr, -1)
        print(
            f"Fitting LowRankPriorCov on z-scored X_z={X_z.shape} rank={args.cov_rank}",
            flush=True,
        )
        prior_cov = LowRankPriorCov.from_samples(
            X_z,
            rank=args.cov_rank,
            jitter_frac=args.cov_jitter_frac,
            random_state=args.seed,
        )

    print(
        f"Fitting DIS: N={n}, L={L}, k_in={args.k_in}, k_out={args.k_out}, "
        f"loaded={n_load}/{n_all}, train_states={n_states}/{ntr}, probes={args.n_probes}, "
        f"metric={args.input_metric}",
        flush=True,
    )
    if n_states * args.n_probes < max(args.k_in, args.k_out):
        raise SystemExit(
            f"n_states*n_probes={n_states * args.n_probes} < max(k_in,k_out); increase probing"
        )
    memmap_dir = args.memmap_dir or os.environ.get(
        "DIS_MEMMAP_DIR", expand("${AMOR_DATA_ROOT}/nonlinear_diffusion_basis/_dis_scratch")
    )
    rows_in, ev_in, rows_out, ev_out = fit_dis_rows_from_data(
        f_dis,
        m_dis,
        args.k_in,
        args.k_out,
        n_states,
        args.n_probes,
        args.seed,
        L=L,
        memmap_dir=memmap_dir,
        prior_cov=prior_cov,
        x_std_flat=x_std_flat if prior_cov is not None else None,
    )
    if prior_cov is None:
        rows_in, rows_out = scale_dis_rows_to_normalized(
            rows_in, rows_out, x_std_flat=x_std_flat, y_std_flat=y_std_flat
        )
        ortho_metric = "I"
        mass_whiten = "none"
    else:
        # rows_in = ψ (Euclidean in C^{-1/2}-whitened space); scale Euclidean rows_out
        rows_out = rows_out / np.maximum(y_std_flat.reshape(1, -1), 1e-30)
        ortho_metric = "C_inv"
        mass_whiten = "C_inv_sqrt"  # AMOR whitens x with C^{-1/2} before lift

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = dict(
        rows_in=rows_in.astype(np.float32),
        rows_out=rows_out.astype(np.float32),
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
        mass_whiten=np.asarray(mass_whiten),
    )
    if prior_cov is not None:
        payload.update(prior_cov.save_dict())
    np.savez_compressed(args.out, **payload)
    print(f"Saved {args.out}  rows_in={rows_in.shape} rows_out={rows_out.shape}", flush=True)


if __name__ == "__main__":
    main()
