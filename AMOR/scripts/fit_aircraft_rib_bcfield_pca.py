#!/usr/bin/env python3
"""
Fit PCA + train-split normalization for bcfield rib (same path as training).

Methodology matches train_amor_aircraft_rib_bcfield.py:
  build_field_tensors -> split(seed,70/20/10) -> normalize(train) ->
  PCAReducerMultiChannel(svd_solver=randomized, random_state=seed).fit(xtr4, ytr4)

Saves one NPZ bundle (PCA bases, norm stats, FiLM norm scalars, mesh shape).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.models.PCAReducer import PCAReducerMultiChannel  # noqa: E402
from AMOR.datasets.aircraft_rib import (  # noqa: E402
    build_field_tensors,
    load_bcfield_dataset,
    normalize_per_channel,
    split_indices,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--use_solve_ok_only", action="store_true")
    p.add_argument(
        "--n_components_in",
        type=int,
        default=None,
        help="Fit only this many input PCA modes (pass model k_in). Default: full basis.",
    )
    p.add_argument(
        "--n_components_out",
        type=int,
        default=None,
        help="Fit only this many output PCA modes (pass model k_out). Default: full basis.",
    )
    args = p.parse_args()

    d, n_use = load_bcfield_dataset(args.data_path, use_solve_ok_only=args.use_solve_ok_only)
    x, y = build_field_tensors(d, n_use)
    nu = np.asarray(d["nu"], dtype=np.float32).reshape(-1)[:n_use]
    case = np.asarray(d["load_case_id"], dtype=np.float32).reshape(-1)[:n_use]
    print(
        f"samples={n_use} nodes={x.shape[2]} in_ch={x.shape[1]} out_ch={y.shape[1]}",
        flush=True,
    )

    tr_idx, va_idx, te_idx = split_indices(n_use, args.seed, args.train_frac, args.val_frac)
    xn, yn, stats = normalize_per_channel(x, y, tr_idx)

    nu_mu, nu_std = float(nu[tr_idx].mean()), float(nu[tr_idx].std() + 1e-8)
    case_mu, case_std = float(case[tr_idx].mean()), float(case[tr_idx].std() + 1e-8)
    film_norm = {
        "nu_mu": nu_mu,
        "nu_std": nu_std,
        "case_mu": case_mu,
        "case_std": case_std,
    }

    xtr4 = xn[tr_idx][:, :, :, None].astype(np.float32)
    ytr4 = yn[tr_idx][:, :, :, None].astype(np.float32)
    print(
        f"PCA fit request n_components_in={args.n_components_in} "
        f"n_components_out={args.n_components_out}",
        flush=True,
    )
    rb = PCAReducerMultiChannel(svd_solver="randomized", random_state=args.seed).fit(
        xtr4,
        ytr4,
        n_components_in=args.n_components_in,
        n_components_out=args.n_components_out,
    )
    print(
        f"PCA fit n_train={len(tr_idx)} shape={rb.shape_} "
        f"max_k_in={rb.pca_in.n_components_} max_k_out={rb.pca_out.n_components_}",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    meta = {
        "data_path": os.path.abspath(args.data_path),
        "svd_solver": "randomized",
        "random_state": args.seed,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_samples": int(n_use),
        "n_components_in": args.n_components_in,
        "n_components_out": args.n_components_out,
        "max_k_in": int(rb.pca_in.n_components_),
        "max_k_out": int(rb.pca_out.n_components_),
    }
    rb.save_training_bundle(args.out_npz, normalization=stats, film_norm=film_norm, meta=meta)

    rb2, stats2, film2, _ = PCAReducerMultiChannel.load_training_bundle(args.out_npz)
    x_chk = xtr4[0].reshape(1, -1).astype(np.float64)
    z_in = rb2.pca_in.transform(x_chk)
    z_out = rb2.pca_out.transform(ytr4[0].reshape(1, -1).astype(np.float64))
    print(
        f"verify load: z_in nan={np.isnan(z_in).any()} z_out nan={np.isnan(z_out).any()} "
        f"stats_match={np.allclose(stats['x_mu'], stats2['x_mu'])}",
        flush=True,
    )

    sidecar = args.out_npz.replace(".npz", ".json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {args.out_npz}", flush=True)
    print(f"Saved {sidecar}", flush=True)


if __name__ == "__main__":
    main()
