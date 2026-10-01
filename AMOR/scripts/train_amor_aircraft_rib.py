#!/usr/bin/env python3
"""Train AMOR on aircraft-rib linear elasticity.

Datasets
  cantilever_tip  mass-PCA, scalar-load FiLM, optional mesh bases
  bcfield         Euclidean PCA from a precomputed --pca_npz bundle

Tip defaults match the published easy-10k mass-PCA protocol (coords + FiLM on).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from AMOR.models.amor import AMOR  # noqa: E402
from AMOR.models.PCAReducer import (  # noqa: E402
    PCAReducerMultiChannel,
    PCAReducerMultiChannelMass,
    fit_random_init_basis_mass,
)
from AMOR.models.mass.mesh_mass import lumped_nodal_mass  # noqa: E402
from AMOR.models.mesh_bases import fit_mesh_alternative_basis_mass  # noqa: E402
from AMOR.datasets.aircraft_rib import (  # noqa: E402
    build_field_tensors,
    load_bcfield_dataset,
    normalize_per_channel,
    split_indices,
)
from AMOR.paths import expand  # noqa: E402
from AMOR.training.slurm import eval_deadline_walltime  # noqa: E402

LEARNED = ("learned_qr", "learned_cayley")
FIXED = ("pca", "chebyshev", "legendre", "jacobi", "laplacian")
ALL_BASES = FIXED + LEARNED

TIP_DATA = expand("${AMOR_DATA_ROOT}/aircraft_rib/datasets/aircraft_rib_cantilever_tip.npz")
TIP_CKPT = expand("${AMOR_DATA_ROOT}/aircraft_rib/amor_rib_h15_gp10000_easy_best_mass")
TIP_BASIS_CKPT = expand("${AMOR_DATA_ROOT}/aircraft_rib/amor_rib_h15_basis")


def _build_tip_input_channels(d: Dict[str, np.ndarray]) -> np.ndarray:
    if "u" not in d:
        raise ValueError("Dataset missing required key: u")
    u = np.asarray(d["u"], dtype=np.float32)
    if u.ndim != 3 or u.shape[2] != 3:
        raise ValueError(f"Expected u shape (S,N,3), got {u.shape}")
    s, n, _ = u.shape
    if "E_nodal" not in d:
        raise ValueError("Dataset missing `E_nodal` (required).")
    e_nodal = np.asarray(d["E_nodal"], dtype=np.float32)
    if e_nodal.shape != (s, n):
        raise ValueError(f"E_nodal shape mismatch: expected {(s, n)}, got {e_nodal.shape}")
    return e_nodal[:, None, :].astype(np.float32)


def _build_tip_load_scalar(d: Dict[str, np.ndarray], s: int) -> np.ndarray:
    if "load" not in d:
        raise ValueError("Dataset missing required key: load")
    load = np.asarray(d["load"], dtype=np.float32).reshape(-1)
    if load.shape[0] != s:
        raise ValueError(f"load length mismatch: {load.shape[0]} vs samples {s}")
    return load


def _build_tip_targets(d: Dict[str, np.ndarray]) -> np.ndarray:
    u = np.asarray(d["u"], dtype=np.float32)
    return np.transpose(u, (0, 2, 1)).astype(np.float32)


def _split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("Need at least 1 sample.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    if n_train <= 0:
        n_train = 1
        n_val = max(0, min(n - n_train, n_val))
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    return tr, va, te


def _normalize_per_channel(x: np.ndarray, y: np.ndarray, tr_idx: np.ndarray):
    x_mu = x[tr_idx].mean(axis=(0, 2), keepdims=True)
    x_std = x[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    y_mu = y[tr_idx].mean(axis=(0, 2), keepdims=True)
    y_std = y[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    xn = (x - x_mu) / x_std
    yn = (y - y_mu) / y_std
    stats = {
        "x_mu": x_mu.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mu": y_mu.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
    return xn.astype(np.float32), yn.astype(np.float32), stats


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    early_stopping_patience: int,
    device: torch.device,
    eval_deadline_ts: float | None = None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    crit = nn.L1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_rel_l1, best_epoch, wait = float("inf"), 0, 0
    t0 = time.time()
    stopped_by_slurm_budget = False
    has_val = len(val_loader.dataset) > 0

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
                    pf, yf = pred.view(b, -1), yb.view(b, -1)
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
            print(f"[time budget] stop after epoch {ep + 1}/{epochs}", flush=True)
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
            pf, yf = pred.view(b, -1), yb.view(b, -1)
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
            pf, yf = pred_phys.view(b, -1), y_phys.view(b, -1)
            rel_l1_sum += float((torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)).sum().item())
            rel_l2_sum += float((torch.norm(pf - yf, p=2, dim=1) / torch.norm(yf, p=2, dim=1).clamp(min=1e-8)).sum().item())
            n += b
    if n == 0:
        return float("nan"), float("nan")
    return rel_l1_sum / n, rel_l2_sum / n


def _loader_for_indices(xn, yn, film, idx, batch_size, shuffle=False):
    if len(idx) == 0:
        z = xn.shape[2]
        return DataLoader(
            TensorDataset(
                torch.empty((0, xn.shape[1], z)),
                torch.empty((0, yn.shape[1], z)),
                torch.empty((0, film.shape[1])),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    return DataLoader(
        TensorDataset(
            torch.from_numpy(xn[idx]).float(),
            torch.from_numpy(yn[idx]).float(),
            torch.from_numpy(film[idx]).float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def fit_tip_basis(bt, xtr4, ytr4, nodes, elems, mass_nodes, k_in, k_out, seed):
    k_out_pca = max(k_in, k_out)
    if bt in LEARNED:
        return fit_random_init_basis_mass(
            xtr4, ytr4, mass_nodes, random_state=seed,
            n_components_in=k_in, n_components_out=k_out,
        )
    if bt == "pca":
        return PCAReducerMultiChannelMass(svd_solver="randomized", random_state=seed).fit(
            xtr4, ytr4, mass_nodes=mass_nodes, k_in=k_in, k_out=k_out_pca,
        )
    return fit_mesh_alternative_basis_mass(
        xtr4, ytr4, bt, nodes=nodes, elements=elems, mass_nodes=mass_nodes,
        random_state=seed, n_components_in=k_in, n_components_out=k_out,
    )


@torch.no_grad()
def infer_ms(model, loader, dev):
    model.eval()
    xb, _, fb = next(iter(loader))
    xb, fb = xb.to(dev), fb.to(dev)
    bs = xb.size(0)
    for _ in range(2):
        model(xb, fb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        model(xb, fb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (5 * bs) * 1000


def _run_tip(args, device, eval_deadline_ts):
    d = np.load(args.data_path, allow_pickle=True)
    if args.use_solve_ok_only and "solve_ok" in d.files:
        ok = np.asarray(d["solve_ok"], dtype=bool).reshape(-1)
        if np.count_nonzero(ok) == 0:
            raise RuntimeError("use_solve_ok_only set but zero solve_ok samples.")
        print(f"Filtering solve_ok samples: {np.count_nonzero(ok)}/{len(ok)}", flush=True)
        d = {k: np.asarray(d[k])[ok] if np.asarray(d[k]).shape[0] == len(ok) else np.asarray(d[k]) for k in d.files}
    else:
        d = {k: np.asarray(d[k]) for k in d.files}

    x = _build_tip_input_channels(d)
    y = _build_tip_targets(d)
    l = _build_tip_load_scalar(d, x.shape[0])
    s = x.shape[0]
    nodes = np.asarray(d["nodes"], dtype=np.float64)
    elems = np.asarray(d["elements"], dtype=np.int64)
    mass_nodes = lumped_nodal_mass(nodes, elems)
    print(
        f"samples={s} nodes={x.shape[2]} in_ch={x.shape[1]} out_ch={y.shape[1]} "
        f"basis={args.basis_type}",
        flush=True,
    )

    tr_idx, va_idx, te_idx = _split_indices(s, args.seed, args.train_frac, args.val_frac)
    xn, yn, stats = _normalize_per_channel(x, y, tr_idx)
    l_mu = float(np.mean(l[tr_idx]))
    l_std = float(np.std(l[tr_idx]) + 1e-8)
    film = ((l - l_mu) / l_std).astype(np.float32)[:, None]

    train_loader = _loader_for_indices(xn, yn, film, tr_idx, args.batch_size, shuffle=True)
    val_loader = _loader_for_indices(xn, yn, film, va_idx, args.batch_size)
    test_loader = _loader_for_indices(xn, yn, film, te_idx, args.batch_size)

    xtr4 = xn[tr_idx][:, :, :, None]
    ytr4 = yn[tr_idx][:, :, :, None]
    rb = fit_tip_basis(
        args.basis_type, xtr4, ytr4, nodes, elems, mass_nodes,
        args.k_in, args.k_out, args.seed,
    )
    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    print(f"rank clamp k_in={k_in} k_out={k_out}", flush=True)

    learned = args.basis_type in LEARNED
    use_coords = not args.no_coordinates
    use_film = not args.no_film_conditioning
    model = AMOR(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=args.dropout, init_scale=args.init_scale, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=use_coords,
        use_film_conditioning=use_film, film_hidden=args.film_hidden,
        trainable_basis=learned,
        trainable_basis_param=args.basis_type.replace("learned_", "") if learned else "qr",
    )
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    es_patience = (args.epochs + 1) if learned else args.early_stopping_patience
    best_val, best_ep, ttrain, stopped = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm, early_stopping_patience=es_patience,
        device=device, eval_deadline_ts=eval_deadline_ts,
    )
    te_l1n, te_l2n = compute_rel_metrics_normalized(model, test_loader, device)
    te_l1p, te_l2p = compute_rel_metrics(model, test_loader, device, stats["y_mu"], stats["y_std"])
    tr_l1, tr_l2 = compute_rel_metrics_normalized(model, train_loader, device)
    va_l1, va_l2 = compute_rel_metrics_normalized(model, val_loader, device)

    tag = args.tag or args.basis_type
    basis_run = args.basis_type != "pca" or args.tag is not None
    if basis_run:
        ckpt_path = os.path.join(args.checkpoint_dir, f"amor_rib_{tag}_best.pt")
        summary_path = os.path.join(args.checkpoint_dir, f"amor_rib_{tag}_metrics.json")
        infer = infer_ms(model, test_loader, device)
        summary = {
            "dataset": "aircraft_rib_cantilever_tip",
            "data_path": args.data_path,
            "basis_type": args.basis_type,
            "tag": tag,
            "trainable_basis": learned,
            "trainable_basis_param": args.basis_type.replace("learned_", "") if learned else None,
            "use_mass_weighted_pca": True,
            "use_coordinates": use_coords,
            "use_film_conditioning": use_film,
            "use_solve_ok_only": bool(args.use_solve_ok_only),
            "k_in": k_in,
            "k_out": k_out,
            "width": args.width,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "best_epoch": best_ep,
            "best_val_rel_l1": float(best_val),
            "test_rel_l1": float(te_l1p),
            "test_rel_l2": float(te_l2p),
            "test_rel_l1_normalized": float(te_l1n),
            "test_rel_l2_normalized": float(te_l2n),
            "train_sec_per_epoch": ttrain / max(1, best_ep),
            "inference_ms_per_sample": infer,
            "training_time_sec": ttrain,
            "split": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
            "stopped_by_slurm_time_budget": bool(stopped),
        }
        torch.save({"model_state_dict": model.state_dict(), **summary}, ckpt_path)
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_linear_best.pt")
        summary_path = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_linear_summary.json")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_rel_l1": best_val,
                "best_epoch": best_ep,
                "train_rel_l1_normalized": tr_l1,
                "train_rel_l2_normalized": tr_l2,
                "val_rel_l1_normalized": va_l1,
                "val_rel_l2_normalized": va_l2,
                "test_rel_l1": te_l1p,
                "test_rel_l2": te_l2p,
                "test_rel_l1_normalized": te_l1n,
                "test_rel_l2_normalized": te_l2n,
                "k_in": k_in,
                "k_out": k_out,
                "width": args.width,
                "layers": args.layers,
                "train_time_s": ttrain,
                "use_mass_weighted_pca": True,
                "mass_nodes": mass_nodes.astype(np.float32),
                "normalization": {k: v for k, v in stats.items()},
                "load_norm": {"mu": l_mu, "std": l_std},
                "data_path": args.data_path,
                "stopped_by_slurm_time_budget": stopped,
                "time_buffer_sec": float(args.time_buffer_sec),
                "use_coordinates": use_coords,
                "use_film_conditioning": use_film,
                "film_hidden": int(args.film_hidden),
                "init_scale": float(args.init_scale),
                "pca_mass": {
                    "pca_in_components": rb.pca_in.components_[:k_in].astype(np.float32),
                    "pca_in_mean": rb.pca_in.mean_.astype(np.float32),
                    "pca_in_evr": rb.pca_in.explained_variance_ratio_[:k_in].astype(np.float32),
                    "pca_out_components": rb.pca_out.components_[:k_out].astype(np.float32),
                    "pca_out_mean": rb.pca_out.mean_.astype(np.float32),
                    "pca_out_evr": rb.pca_out.explained_variance_ratio_[:k_out].astype(np.float32),
                    "mass_diag_in": rb.mass_diag_in_.astype(np.float32),
                    "mass_diag_out": rb.mass_diag_out_.astype(np.float32),
                    "shape_": tuple(rb.shape_),
                    "n_channels": int(rb.n_channels),
                    "out_channels": int(rb.out_channels),
                },
                "seed": int(args.seed),
            },
            ckpt_path,
            pickle_protocol=4,
        )
        summary = {
            "data_path": args.data_path,
            "num_samples": int(s),
            "nodes": int(x.shape[2]),
            "split": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
            "best_val_rel_l1": float(best_val),
            "best_epoch": int(best_ep),
            "train_rel_l1_normalized": float(tr_l1),
            "val_rel_l1_normalized": float(va_l1),
            "test_rel_l1": float(te_l1p),
            "test_rel_l1_normalized": float(te_l1n),
            "test_rel_l2_normalized": float(te_l2n),
            "train_rel_l2_normalized": float(tr_l2),
            "val_rel_l2_normalized": float(va_l2),
            "test_rel_l2": float(te_l2p),
            "checkpoint": ckpt_path,
            "train_time_s": float(ttrain),
            "k_in": int(k_in),
            "k_out": int(k_out),
            "width": int(args.width),
            "layers": int(args.layers),
            "use_coordinates": use_coords,
            "use_film_conditioning": use_film,
            "film_hidden": int(args.film_hidden),
            "use_mass_weighted_pca": True,
            "stopped_by_slurm_time_budget": bool(stopped),
            "time_buffer_sec": float(args.time_buffer_sec),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary, flush=True)
    print(f"Saved {ckpt_path}", flush=True)


def _run_bcfield(args, device, eval_deadline_ts):
    if not args.pca_npz:
        raise SystemExit("--dataset bcfield requires --pca_npz")

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

    tr_idx, va_idx, te_idx = split_indices(n_use, args.seed, args.train_frac, args.val_frac)
    xn, yn, stats_re = normalize_per_channel(x, y, tr_idx)
    for key in ("x_mu", "x_std", "y_mu", "y_std"):
        if not np.allclose(stats[key], stats_re[key], rtol=1e-5, atol=1e-6):
            raise ValueError(f"PCA bundle {key} mismatch vs recomputed train normalization")

    nun = ((nu - film_norm["nu_mu"]) / film_norm["nu_std"]).astype(np.float32)
    casen = ((case - film_norm["case_mu"]) / film_norm["case_std"]).astype(np.float32)
    film = np.stack([nun, casen], axis=1).astype(np.float32)

    train_loader = _loader_for_indices(xn, yn, film, tr_idx, args.batch_size, shuffle=True)
    val_loader = _loader_for_indices(xn, yn, film, va_idx, args.batch_size)
    test_loader = _loader_for_indices(xn, yn, film, te_idx, args.batch_size)

    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    rb.truncate(k_in, k_out)
    print(f"PCA clamp k_in={k_in} k_out={k_out}", flush=True)

    model = AMOR(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=args.dropout, init_scale=args.init_scale, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False,
        use_film_conditioning=True, film_hidden=args.film_hidden, film_in_dim=2,
    )
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    model.to(device)
    with torch.no_grad():
        xb0 = torch.from_numpy(xn[tr_idx[: min(4, len(tr_idx))]]).float().to(device)
        fb0 = torch.from_numpy(film[tr_idx[: min(4, len(tr_idx))]]).float().to(device)
        p0 = model(xb0, fb0)
        if not torch.isfinite(p0).all():
            raise RuntimeError(f"Non-finite pred on first batch: max={p0.abs().max().item()}")

    best_val, best_ep, train_time, stopped = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm, early_stopping_patience=args.early_stopping_patience,
        device=device, eval_deadline_ts=eval_deadline_ts,
    )
    tr_l1, tr_l2 = compute_rel_metrics_normalized(model, train_loader, device)
    va_l1, va_l2 = compute_rel_metrics_normalized(model, val_loader, device)
    te_l1, te_l2 = compute_rel_metrics(model, test_loader, device, stats["y_mu"], stats["y_std"])

    case_int = np.asarray(case, dtype=np.int64)
    te_by_case = {}
    for cid in sorted(set(int(c) for c in np.unique(case_int[te_idx]))):
        te_c = te_idx[case_int[te_idx] == cid]
        l1_c, l2_c = compute_rel_metrics(
            model, _loader_for_indices(xn, yn, film, te_c, args.batch_size),
            device, stats["y_mu"], stats["y_std"],
        )
        te_by_case[cid] = (te_c, l1_c, l2_c)
        print(f"test case_id={cid} n={len(te_c)} rel_l1={l1_c:.4e} rel_l2={l2_c:.4e}", flush=True)

    def _case(cid):
        return te_by_case.get(cid, (np.array([], dtype=np.int64), float("nan"), float("nan")))

    te0_idx, te0_l1, te0_l2 = _case(0)
    te1_idx, te1_l1, te1_l2 = _case(1)
    te2_idx, te2_l1, te2_l2 = _case(2)

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


def main() -> None:
    p = argparse.ArgumentParser(description="Train AMOR on aircraft-rib datasets")
    p.add_argument("--dataset", choices=("cantilever_tip", "bcfield"), default="cantilever_tip")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--basis_type", choices=ALL_BASES, default="pca", help="tip only")
    p.add_argument("--tag", default=None, help="tip basis-sweep filename tag (default: basis_type)")
    p.add_argument("--skip_if_exists", action="store_true", help="tip basis-sweep: skip if metrics json exists")
    p.add_argument("--pca_npz", type=str, default=None, help="bcfield: bundle from fit_aircraft_rib_bcfield_pca.py")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--k_in", type=int, default=None)
    p.add_argument("--k_out", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.7)
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--time_buffer_sec", type=float, default=None)
    p.add_argument("--use_solve_ok_only", action="store_true", default=None)
    p.add_argument("--no_coordinates", action="store_true", help="tip: disable coordinate channels")
    p.add_argument("--no_film_conditioning", action="store_true", help="tip: disable FiLM")
    # Accepted for older tip CLIs; tip protocol enables coords/FiLM by default.
    p.add_argument("--use_coordinates", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--use_film_conditioning", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.dataset == "cantilever_tip":
        if args.data_path is None:
            args.data_path = TIP_DATA
        if args.checkpoint_dir is None:
            args.checkpoint_dir = TIP_BASIS_CKPT if (args.basis_type != "pca" or args.tag) else TIP_CKPT
        if args.batch_size is None:
            args.batch_size = 64
        if args.width is None:
            args.width = 16
        if args.layers is None:
            args.layers = 5
        if args.k_in is None:
            args.k_in = 100
        if args.k_out is None:
            args.k_out = 30
        if args.time_buffer_sec is None:
            args.time_buffer_sec = float(
                os.environ.get("AMOR_SLURM_BUFFER_SEC")
                or os.environ.get("RIB_TIME_BUFFER_SEC")
                or "1200"
            )
        if args.use_solve_ok_only is None:
            args.use_solve_ok_only = True
    else:
        if args.data_path is None or args.checkpoint_dir is None:
            raise SystemExit("--dataset bcfield requires --data_path and --checkpoint_dir")
        if args.batch_size is None:
            args.batch_size = 8
        if args.width is None:
            args.width = 24
        if args.layers is None:
            args.layers = 4
        if args.k_in is None:
            args.k_in = 80
        if args.k_out is None:
            args.k_out = 40
        if args.time_buffer_sec is None:
            args.time_buffer_sec = 900.0
        if args.use_solve_ok_only is None:
            args.use_solve_ok_only = False
        if args.basis_type != "pca":
            raise SystemExit("--basis_type is only supported for --dataset cantilever_tip")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.dataset == "cantilever_tip" and args.skip_if_exists:
        tag = args.tag or args.basis_type
        mp = os.path.join(args.checkpoint_dir, f"amor_rib_{tag}_metrics.json")
        if os.path.isfile(mp):
            print(f"Skip {mp}", flush=True)
            return

    if args.device == "cuda" and not torch.cuda.is_available():
        if args.dataset == "bcfield":
            args.device = "cpu"
        else:
            raise RuntimeError("Requested --device cuda but CUDA is unavailable")
    device = torch.device(args.device)
    print(f"Device: {device} dataset={args.dataset}", flush=True)
    eval_deadline_ts = eval_deadline_walltime(buffer_sec=args.time_buffer_sec)

    if args.dataset == "cantilever_tip":
        _run_tip(args, device, eval_deadline_ts)
    else:
        _run_bcfield(args, device, eval_deadline_ts)


if __name__ == "__main__":
    main()
