#!/usr/bin/env python3
"""
Train AMORWithFieldInputs on bcfield rib dataset.

Input fields (4 channels, nodal):
  - E_nodal
  - load_x, load_y, load_z nodal (load_field_boundary scattered to mesh)
Output field (3 channels, nodal):
  - u_x, u_y, u_z from dataset key u
FiLM conditioning (required):
  - nu (scalar per sample)
  - load_case_id (0=uniform allowed boundary, 1=localized patch, 2=easy GP Neumann tip)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Tuple

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.models.AMOR_field_inputs import AMORWithFieldInputs  # noqa: E402
from AMOR.models.PCAReducer import PCAReducerMultiChannel  # noqa: E402
from AMOR.datasets.aircraft_rib import (  # noqa: E402
    build_field_tensors,
    load_bcfield_dataset,
    normalize_per_channel,
    split_indices,
)

from AMOR.training.slurm import eval_deadline_walltime  # noqa: E402


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    lr,
    weight_decay,
    max_grad_norm,
    early_stopping_patience,
    device,
    eval_deadline_ts=None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    crit = nn.L1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_rel_l1, best_epoch, wait = float("inf"), 0, 0
    t0 = time.time()
    has_val = len(val_loader.dataset) > 0
    stopped_by_slurm_budget = False

    for ep in range(epochs):
        if eval_deadline_ts is not None and time.time() >= eval_deadline_ts:
            print(f"[time budget] stop before epoch {ep + 1}/{epochs}", flush=True)
            stopped_by_slurm_budget = True
            break
        model.train()
        train_loss = 0.0
        for xb, yb, fb in train_loader:
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            opt.zero_grad()
            pred = model(xb, fb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        if has_val:
            model.eval()
            rel_l1_sum, n = 0.0, 0
            with torch.no_grad():
                for xb, yb, fb in val_loader:
                    xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
                    pred = model(xb, fb)
                    b = pred.size(0)
                    pf = pred.view(b, -1)
                    yf = yb.view(b, -1)
                    rel = torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)
                    rel_l1_sum += float(rel.sum().item())
                    n += b
            val_rel_l1 = rel_l1_sum / max(1, n)
        else:
            val_rel_l1 = train_loss

        if val_rel_l1 < best_val_rel_l1:
            best_val_rel_l1 = val_rel_l1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            wait = 0
        else:
            wait += 1
        sch.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[{ep+1:03d}/{epochs}] train_l1={train_loss:.4e} val_rel_l1={val_rel_l1:.4e} "
                f"lr={sch.get_last_lr()[0]:.2e} best={best_val_rel_l1:.4e}@{best_epoch+1}",
                flush=True,
            )
        if wait >= early_stopping_patience:
            print(f"Early stopping at epoch {ep+1}", flush=True)
            break
        if eval_deadline_ts is not None and time.time() >= eval_deadline_ts:
            stopped_by_slurm_budget = True
            break

    model.load_state_dict(best_state)
    return best_val_rel_l1, best_epoch + 1, time.time() - t0, stopped_by_slurm_budget


def compute_rel_metrics_normalized(model, loader, device):
    model.eval()
    rel_l1_sum, rel_l2_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb, fb in loader:
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            pred = model(xb, fb)
            b = pred.size(0)
            pf = pred.view(b, -1)
            yf = yb.view(b, -1)
            rel_l1_sum += float((torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)).sum().item())
            rel_l2_sum += float((torch.norm(pf - yf, p=2, dim=1) / torch.norm(yf, p=2, dim=1).clamp(min=1e-8)).sum().item())
            n += b
    if n == 0:
        return float("nan"), float("nan")
    return rel_l1_sum / n, rel_l2_sum / n


def compute_rel_metrics(model, loader, device, y_mu, y_std):
    y_mu_t = torch.from_numpy(np.asarray(y_mu, dtype=np.float32)).to(device)
    y_std_t = torch.from_numpy(np.asarray(y_std, dtype=np.float32)).to(device)
    model.eval()
    rel_l1_sum, rel_l2_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb, fb in loader:
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            pred = model(xb, fb)
            pred_phys = pred * y_std_t + y_mu_t
            y_phys = yb * y_std_t + y_mu_t
            b = pred_phys.size(0)
            pf = pred_phys.view(b, -1)
            yf = y_phys.view(b, -1)
            rel_l1_sum += float((torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)).sum().item())
            rel_l2_sum += float((torch.norm(pf - yf, p=2, dim=1) / torch.norm(yf, p=2, dim=1).clamp(min=1e-8)).sum().item())
            n += b
    if n == 0:
        return float("nan"), float("nan")
    return rel_l1_sum / n, rel_l2_sum / n


def _loader_for_indices(xn, yn, film, idx, batch_size):
    if len(idx) == 0:
        z = xn.shape[2]
        empty_x = torch.empty((0, xn.shape[1], z))
        empty_y = torch.empty((0, yn.shape[1], z))
        empty_f = torch.empty((0, film.shape[1]))
        return DataLoader(TensorDataset(empty_x, empty_y, empty_f), batch_size=batch_size, shuffle=False)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(xn[idx]).float(),
            torch.from_numpy(yn[idx]).float(),
            torch.from_numpy(film[idx]).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--k_in", type=int, default=80)
    p.add_argument("--k_out", type=int, default=40)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.7)
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--time_buffer_sec", type=float, default=900.0)
    p.add_argument("--use_solve_ok_only", action="store_true")
    p.add_argument(
        "--pca_npz",
        type=str,
        required=True,
        help="Precomputed PCA+norm bundle from fit_aircraft_rib_bcfield_pca.py",
    )
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    print(f"Device: {device}", flush=True)
    eval_deadline_ts = eval_deadline_walltime(buffer_sec=args.time_buffer_sec)

    d, n_use = load_bcfield_dataset(args.data_path, use_solve_ok_only=args.use_solve_ok_only)
    x, y = build_field_tensors(d, n_use)
    nu = np.asarray(d["nu"], dtype=np.float32).reshape(-1)[:n_use]
    case = np.asarray(d["load_case_id"], dtype=np.float32).reshape(-1)[:n_use]
    print(f"samples={n_use} nodes={x.shape[2]} in_ch={x.shape[1]} out_ch={y.shape[1]}", flush=True)

    rb, stats, film_norm, pca_meta = PCAReducerMultiChannel.load_training_bundle(args.pca_npz)
    if pca_meta["seed"] != args.seed or pca_meta["train_frac"] != args.train_frac or pca_meta["val_frac"] != args.val_frac:
        raise ValueError(
            f"PCA bundle split meta {pca_meta['seed']},{pca_meta['train_frac']},{pca_meta['val_frac']} "
            f"!= train args {args.seed},{args.train_frac},{args.val_frac}"
        )
    if rb.shape_[0] != x.shape[2] or rb.n_channels != x.shape[1] or rb.out_channels != y.shape[1]:
        raise ValueError(
            f"PCA bundle shape/channels {rb.shape_},{rb.n_channels},{rb.out_channels} "
            f"!= data {x.shape[2]},{x.shape[1]},{y.shape[1]}"
        )
    print(f"Loaded PCA {args.pca_npz} n_train={pca_meta['n_train']} shape={rb.shape_}", flush=True)

    tr_idx, va_idx, te_idx = split_indices(n_use, args.seed, args.train_frac, args.val_frac)
    xn, yn, stats_re = normalize_per_channel(x, y, tr_idx)
    for key in ("x_mu", "x_std", "y_mu", "y_std"):
        if not np.allclose(stats[key], stats_re[key], rtol=1e-5, atol=1e-6):
            raise ValueError(f"PCA bundle {key} mismatch vs recomputed train normalization")

    nu_mu, nu_std = film_norm["nu_mu"], film_norm["nu_std"]
    case_mu, case_std = film_norm["case_mu"], film_norm["case_std"]
    nun = ((nu - nu_mu) / nu_std).astype(np.float32)
    casen = ((case - case_mu) / case_std).astype(np.float32)
    film = np.stack([nun, casen], axis=1).astype(np.float32)

    def pack(idx):
        if len(idx) == 0:
            z = x.shape[2]
            return (
                torch.empty((0, x.shape[1], z)),
                torch.empty((0, y.shape[1], z)),
                torch.empty((0, 2)),
            )
        return (
            torch.from_numpy(xn[idx]).float(),
            torch.from_numpy(yn[idx]).float(),
            torch.from_numpy(film[idx]).float(),
        )

    xtr, ytr, ftr = pack(tr_idx)
    xva, yva, fva = pack(va_idx)
    xte, yte, fte = pack(te_idx)
    train_loader = DataLoader(TensorDataset(xtr, ytr, ftr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(xva, yva, fva), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(xte, yte, fte), batch_size=args.batch_size, shuffle=False)

    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    rb.truncate(k_in, k_out)
    print(f"PCA clamp k_in={k_in} k_out={k_out}", flush=True)

    model = AMORWithFieldInputs(
        width=args.width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args.layers,
        dropout=args.dropout,
        init_scale=args.init_scale,
        use_skip_connections=True,
        use_batch_norm=False,
        use_coordinates=False,
        use_film_conditioning=True,
        film_hidden=args.film_hidden,
        film_in_dim=2,
    )
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    model.to(device)
    with torch.no_grad():
        xb0 = xtr[: min(4, len(xtr))].to(device)
        yb0 = ytr[: min(4, len(ytr))].to(device)
        fb0 = ftr[: min(4, len(ftr))].to(device)
        p0 = model(xb0, fb0)
        if not torch.isfinite(p0).all():
            raise RuntimeError(f"Non-finite pred on first batch: max={p0.abs().max().item()}")

    best_val, best_ep, train_time, stopped = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        eval_deadline_ts=eval_deadline_ts,
    )
    tr_l1, tr_l2 = compute_rel_metrics_normalized(model, train_loader, device)
    va_l1, va_l2 = compute_rel_metrics_normalized(model, val_loader, device)
    te_l1, te_l2 = compute_rel_metrics(model, test_loader, device, stats["y_mu"], stats["y_std"])

    case_int = np.asarray(case, dtype=np.int64)
    te_by_case = {}
    for cid in sorted(set(int(c) for c in np.unique(case_int[te_idx]))):
        te_c = te_idx[case_int[te_idx] == cid]
        l1_c, l2_c = compute_rel_metrics(
            model,
            _loader_for_indices(xn, yn, film, te_c, args.batch_size),
            device,
            stats["y_mu"],
            stats["y_std"],
        )
        te_by_case[cid] = (te_c, l1_c, l2_c)
        print(
            f"test case_id={cid} n={len(te_c)} rel_l1={l1_c:.4e} rel_l2={l2_c:.4e}",
            flush=True,
        )
    te0 = te_by_case.get(0, (np.array([], dtype=np.int64), float("nan"), float("nan")))
    te1 = te_by_case.get(1, (np.array([], dtype=np.int64), float("nan"), float("nan")))
    te2 = te_by_case.get(2, (np.array([], dtype=np.int64), float("nan"), float("nan")))
    te0_idx, te0_l1, te0_l2 = te0
    te1_idx, te1_l1, te1_l2 = te1
    te2_idx, te2_l1, te2_l2 = te2

    ckpt = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_bcfield_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "k_in": k_in,
            "k_out": k_out,
            "width": args.width,
            "layers": args.layers,
            "film_in_dim": 2,
            "normalization": stats,
            "film_norm": film_norm,
            "pca_npz": os.path.abspath(args.pca_npz),
            "data_path": args.data_path,
        },
        ckpt,
    )
    summary = {
        "data_path": args.data_path,
        "num_samples": int(n_use),
        "split": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "in_channels": int(x.shape[1]),
        "out_channels": int(y.shape[1]),
        "use_film_conditioning": True,
        "film_cond": ["nu", "load_case_id"],
        "use_coordinates": False,
        "width": int(args.width),
        "layers": int(args.layers),
        "best_epoch": int(best_ep),
        "best_val_rel_l1": float(best_val),
        "train_rel_l1_normalized": float(tr_l1),
        "train_rel_l2_normalized": float(tr_l2),
        "val_rel_l1_normalized": float(va_l1),
        "val_rel_l2_normalized": float(va_l2),
        "test_rel_l1": float(te_l1),
        "test_rel_l2": float(te_l2),
        "test_rel_l1_case_id_0": float(te0_l1),
        "test_rel_l2_case_id_0": float(te0_l2),
        "test_n_case_id_0": int(len(te0_idx)),
        "test_rel_l1_case_id_1": float(te1_l1),
        "test_rel_l2_case_id_1": float(te1_l2),
        "test_n_case_id_1": int(len(te1_idx)),
        "test_rel_l1_case_id_2": float(te2_l1),
        "test_rel_l2_case_id_2": float(te2_l2),
        "test_n_case_id_2": int(len(te2_idx)),
        "k_in": int(k_in),
        "k_out": int(k_out),
        "pca_npz": os.path.abspath(args.pca_npz),
        "checkpoint": ckpt,
        "stopped_by_slurm_time_budget": bool(stopped),
    }
    summary_path = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_bcfield_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {ckpt}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
