#!/usr/bin/env python3
"""AMOR basis ablation on nonlinear diffusion (f,m)->u, frozen best config."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path as _PathForAmor

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.paths import expand  # noqa: E402

from AMOR.models.amor import AMOR  # noqa: E402

from AMOR.models.PCAReducer import (  # noqa: E402
    PCAReducerMultiChannel,
    _pca_like_from_rows,
    fit_alternative_basis,
    fit_random_init_basis,
)
from AMOR.models.prior_cov import LowRankPriorCov  # noqa: E402

LEARNED = ("learned_qr", "learned_cayley")
FIXED = ("pca", "chebyshev", "legendre", "jacobi", "laplacian")
ALL = FIXED + LEARNED + ("dis",)

K_IN = K_OUT = 144
WIDTH, LAYERS = 24, 4
BATCH = 32


def _npz_str(d, key: str, default: str = "") -> str:
    if key not in d.files:
        return default
    v = d[key]
    if isinstance(v, np.ndarray):
        v = v.item() if v.ndim == 0 else v
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def load_data(path, num_samples):
    """Load (f, m) -> u on NxN grid. Operator: -div(exp(m) grad u) = f, u|dOmega=0."""
    d = np.load(path, mmap_mode="r")
    n = int(d["N"]) if "N" in d.files else int(d["nx"]) if "nx" in d.files else int(np.sqrt(d["f"].shape[-1]))
    f = np.asarray(d["f"][:num_samples], dtype=np.float32)
    m = np.asarray(d["m"][:num_samples], dtype=np.float32)
    u = np.asarray(d["u"][:num_samples], dtype=np.float32)
    if f.shape[-2:] != (n, n) or m.shape[-2:] != (n, n) or u.shape[-2:] != (n, n):
        raise ValueError(f"expected ({n},{n}) fields, got f{f.shape} m{m.shape} u{u.shape}")
    X = np.stack([f, m], axis=1)  # (S, 2, N, N)
    Y = u[:, np.newaxis, :, :]    # (S, 1, N, N)
    return X, Y, n


def rb_from_dis(path, Xtr, Ytr, k_in, k_out):
    d = np.load(path)
    n = int(d["nx"]) if "nx" in d.files else int(d["N"]) if "N" in d.files else int(round(np.sqrt(Xtr.shape[2] * Xtr.shape[3])))
    ns = Xtr.shape[0]
    rb = PCAReducerMultiChannel()
    rb.n_channels, rb.out_channels = 2, 1
    rb.shape_, rb.output_shape_ = (n, n), (1, n, n)
    ortho = _npz_str(d, "ortho_metric")
    input_metric = _npz_str(d, "input_metric")
    use_cinv = ortho == "C_inv" or input_metric == "prior_cov" or "cov_V" in d.files
    cov = LowRankPriorCov.from_save_dict(d) if use_cinv else None
    # Never keep more modes than prior rank (Cinv) or stored DIS rows.
    k_in_eff = min(int(k_in), int(d["rows_in"].shape[0]))
    if cov is not None:
        k_in_eff = min(k_in_eff, int(cov.rank))
        if k_in_eff < int(k_in):
            print(f"clamp k_in {k_in} → {k_in_eff} (cov_rank={cov.rank})", flush=True)
    k_out_eff = min(int(k_out), int(d["rows_out"].shape[0]))
    rows_in = np.asarray(d["rows_in"][:k_in_eff], dtype=np.float64)
    rows_out = np.asarray(d["rows_out"][:k_out_eff], dtype=np.float64)
    if use_cinv:
        # ψ Euclidean-ortho in C^{-1/2} space ≡ encode (C^{-1}Φ)^T with Φ=C^{1/2}ψ.
        mean = np.zeros(rows_in.shape[1], dtype=np.float64)
        rb.pca_in = _pca_like_from_rows(mean, rows_in, ns)
        rb.prior_cov_ = cov
        print(f"DIS C^{{-1}}-in({rows_in.shape[0]}) cov_rank={cov.rank} (whiten+ψ)", flush=True)
    else:
        rb.pca_in = _pca_like_from_rows(Xtr.reshape(ns, -1).mean(0), rows_in, ns)
    rb.pca_out = _pca_like_from_rows(Ytr.reshape(ns, -1).mean(0), rows_out, ns)
    rb.basis_type = "dis"
    rb.output_on_increment = False
    return rb


def fit_basis(bt, X, Y, k_in, k_out, rs, lap_p, ja, jb, dis_path=None):
    if bt == "dis":
        return rb_from_dis(dis_path, X, Y, k_in, k_out)
    if bt == "pca":
        rb = PCAReducerMultiChannel(svd_solver="randomized", random_state=rs).fit(
            X, Y, n_components_in=k_in, n_components_out=k_out,
        )
        rb.output_on_increment = False
        return rb
    if bt in LEARNED:
        return fit_random_init_basis(X, Y, random_state=rs, n_components_in=k_in, n_components_out=k_out)
    rb = fit_alternative_basis(X, Y, bt, random_state=rs, laplacian_periodic=lap_p,
                               jacobi_alpha=ja, jacobi_beta=jb,
                               n_components_in=k_in, n_components_out=k_out)
    rb.output_on_increment = False
    return rb


@torch.no_grad()
def eval_phys(model, loader, dev, ym, ys):
    ym, ys = torch.from_numpy(ym).float().to(dev), torch.from_numpy(ys).float().to(dev)
    model.eval()
    l1s, l2s = [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        pd, gt = model(xb) * ys + ym, yb * ys + ym
        diff = (pd - gt).reshape(pd.size(0), -1)
        gtf = gt.reshape(gt.size(0), -1)
        l1s.append((diff.abs().sum(1) / (gtf.abs().sum(1) + 1e-10)).cpu())
        l2s.append((diff.norm(dim=1) / (gtf.norm(dim=1) + 1e-10)).cpu())
    return torch.cat(l1s).mean().item() * 100, torch.cat(l2s).mean().item() * 100


@torch.no_grad()
def infer_ms(model, loader, dev):
    model.eval()
    xb = next(iter(loader))[0].to(dev)
    bs = xb.size(0)
    for _ in range(2):
        model(xb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        model(xb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (5 * bs) * 1000


def train(model, tr, va, dev, args):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    l1 = lambda p, t: torch.mean(torch.abs(p - t))
    best, be, best_st, pat, ep_ran, t0 = float("inf"), 0, copy.deepcopy(model.state_dict()), 0, 0, time.time()
    for ep in range(args.epochs):
        ep_ran = ep + 1
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = l1(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite train loss at epoch {ep + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
        sch.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += l1(model(xb), yb).item() * xb.size(0)
        vl /= max(1, len(va.dataset))
        if vl < best:
            best, be, best_st, pat = vl, ep, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep + 1} val_l1={vl:.6f} best={best:.6f}@{be + 1}", flush=True)
        if pat >= (args.epochs + 1 if args.basis_type in LEARNED else args.early_stopping_patience):
            break
    model.load_state_dict(best_st)
    return best, be + 1, ep_ran, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=expand("${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz"))  # rename path on a different cluster
    p.add_argument("--checkpoint_dir", default=expand("${AMOR_DATA_ROOT}/nonlinear_diffusion_basis"))
    p.add_argument("--basis_type", choices=ALL, default="pca")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--dis_rows_path", default=None)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument(
        "--tag",
        default=None,
        help="Filename tag (default: basis_type, or dis_k{k} when k≠144). "
             "Writes amor_nonlinear_{tag}_metrics.json / _best.pt",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_in", type=int, default=K_IN)
    p.add_argument("--k_out", type=int, default=K_OUT)
    p.add_argument("--laplacian_periodic", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag_early = args.tag or args.basis_type
    mp_early = os.path.join(args.checkpoint_dir, f"amor_nonlinear_{tag_early}_metrics.json")
    if args.skip_if_exists and os.path.isfile(mp_early):
        print(f"Skip {mp_early}")
        return

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, Y, grid_n = load_data(args.data_path, args.num_samples)
    print(f"Loaded nonlinear diffusion: samples={len(X)} grid={grid_n}  map (f,m)->u", flush=True)
    n = len(X)
    ntr, nva = int(0.7 * n), int(0.2 * n)
    Xtr, Xva, Xte = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    Ytr, Yva, Yte = Y[:ntr], Y[ntr:ntr + nva], Y[ntr + nva:]
    Xm = Xtr.mean((0, 2, 3), keepdims=True)
    Xs = Xtr.std((0, 2, 3), keepdims=True) + 1e-8
    Ym = Ytr.mean((0, 2, 3), keepdims=True)
    Ys = Ytr.std((0, 2, 3), keepdims=True) + 1e-8
    Xtr = (Xtr - Xm) / Xs
    Xva = (Xva - Xm) / Xs
    Xte = (Xte - Xm) / Xs
    Ytr = (Ytr - Ym) / Ys
    Yva = (Yva - Ym) / Ys
    Yte = (Yte - Ym) / Ys

    mk = lambda Xa, Ya, sh: DataLoader(
        TensorDataset(torch.from_numpy(Xa), torch.from_numpy(Ya)),
        batch_size=BATCH, shuffle=sh,
    )
    tr, va, te = mk(Xtr, Ytr, True), mk(Xva, Yva, False), mk(Xte, Yte, False)

    dis_path = args.dis_rows_path or os.path.join(args.checkpoint_dir, "dis_rows.npz")
    rb = fit_basis(args.basis_type, Xtr, Ytr, args.k_in, args.k_out, args.seed, False, 1.0, 1.0, dis_path)
    kin, kout = min(args.k_in, rb.pca_in.n_components_), min(args.k_out, rb.pca_out.n_components_)
    if hasattr(rb, "truncate"):
        rb.truncate(kin, kout)
    learned = args.basis_type in LEARNED
    tag = args.tag
    if tag is None:
        tag = args.basis_type
        if args.basis_type == "dis" and (args.k_in != K_IN or args.k_out != K_OUT):
            tag = f"dis_k{kin}"
    model = AMOR(width=WIDTH, rb=rb, k_in=kin, k_out=kout, layers=LAYERS, dropout=0.0, init_scale=0.1,
                  use_skip_connections=True, use_batch_norm=False, use_coordinates=False,
                  trainable_basis=learned, trainable_basis_param=args.basis_type.replace("learned_", "") if learned else "qr")
    model.to(dev)
    _, be, ep_ran, ttrain = train(model, tr, va, dev, args)
    tl1, tl2 = eval_phys(model, te, dev, Ym, Ys)
    infer = infer_ms(model, te, dev)
    summary = dict(dataset="nonlinear_diffusion", basis_type=args.basis_type, k_in=kin, k_out=kout,
                   width=WIDTH, layers=LAYERS, batch_size=BATCH, epochs_ran=ep_ran, best_epoch=be,
                   test_rel_l1=tl1, test_rel_l2=tl2,
                   train_sec_per_epoch=ttrain / max(1, ep_ran), inference_ms_per_sample=infer,
                   training_time_sec=ttrain, dis_rows_path=dis_path if args.basis_type == "dis" else None,
                   tag=tag)
    mp = os.path.join(args.checkpoint_dir, f"amor_nonlinear_{tag}_metrics.json")
    with open(mp, "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), **summary},
               os.path.join(args.checkpoint_dir, f"amor_nonlinear_{tag}_best.pt"))
    print(summary)


if __name__ == "__main__":
    main()
