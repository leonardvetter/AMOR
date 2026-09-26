#!/usr/bin/env python3
"""
Train AMOR (M-weighted PCA + encode) on aircraft-rib linear elasticity.

Uses models/mass/: lumped nodal M, M-PCA fit/encode per mesh_transfer.tex.
Fields E,u: per-channel z-score (train split) like legacy rib — metrics comparable to old run.
FiLM load: z-scored scalar. Ambiguity: M-PCA on z-scored fields, not raw physical (hybrid).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.models.mass.AMOR_load_film_mass import AMORWithLoadFiLM  # noqa: E402
from AMOR.models.mass.PCAReducer_mass import PCAReducerMultiChannelMass  # noqa: E402
from AMOR.models.mass.mesh_mass import lumped_nodal_mass  # noqa: E402

from AMOR.training.slurm import eval_deadline_walltime  # noqa: E402


def _build_input_channels(d: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Return X with shape (S, 2, N): [E(x), load].
    """
    if "u" not in d:
        raise ValueError("Dataset missing required key: u")

    u = np.asarray(d["u"], dtype=np.float32)  # (S, N, 3)
    if u.ndim != 3 or u.shape[2] != 3:
        raise ValueError(f"Expected u shape (S,N,3), got {u.shape}")
    s, n, _ = u.shape

    if "E_nodal" in d:
        e_nodal = np.asarray(d["E_nodal"], dtype=np.float32)
        if e_nodal.shape != (s, n):
            raise ValueError(f"E_nodal shape mismatch: expected {(s, n)}, got {e_nodal.shape}")
        e_ch = e_nodal
        print("Using nodal E(x) from dataset key `E_nodal`.")
    else:
        raise ValueError(
            "Dataset missing `E_nodal` (required). "
            "Regenerate dataset with updated generator."
        )

    x = e_ch[:, None, :].astype(np.float32)  # (S,1,N)
    return x


def _build_load_scalar(d: Dict[str, np.ndarray], s: int) -> np.ndarray:
    if "load" not in d:
        raise ValueError("Dataset missing required key: load")
    load = np.asarray(d["load"], dtype=np.float32).reshape(-1)
    if load.shape[0] != s:
        raise ValueError(f"load length mismatch: {load.shape[0]} vs samples {s}")
    return load


def _build_targets(d: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Return Y with shape (S, 3, N): displacement channels.
    """
    u = np.asarray(d["u"], dtype=np.float32)  # (S,N,3)
    y = np.transpose(u, (0, 2, 1)).astype(np.float32)  # (S,3,N)
    return y


def _split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("Need at least 1 sample.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    # Keep at least one train sample.
    if n_train <= 0:
        n_train = 1
        n_val = max(0, min(n - n_train, n_val))
        n_test = n - n_train - n_val

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


def _subset_tensors(xn: np.ndarray, yn: np.ndarray, ln: np.ndarray, idx: np.ndarray):
    return torch.from_numpy(xn[idx]).float(), torch.from_numpy(yn[idx]).float(), torch.from_numpy(ln[idx]).float()


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
    best_val_rel_l1 = float("inf")
    best_epoch = 0
    wait = 0
    t0 = time.time()
    stopped_by_slurm_budget = False

    has_val = len(val_loader.dataset) > 0

    for ep in range(epochs):
        if eval_deadline_ts is not None and time.time() >= eval_deadline_ts:
            print(f"[time budget] stop before epoch {ep + 1}/{epochs} (leave time for test+save)", flush=True)
            stopped_by_slurm_budget = True
            break
        model.train()
        train_loss = 0.0
        for xb, yb, lb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            opt.zero_grad()
            pred = model(xb, lb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        if has_val:
            model.eval()
            rel_l1_sum = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb, lb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    lb = lb.to(device)
                    pred = model(xb, lb)
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
                f"[{ep+1:03d}/{epochs}] train_l1={train_loss:.4e} "
                f"val_rel_l1={val_rel_l1:.4e} lr={sch.get_last_lr()[0]:.2e} "
                f"(best={best_val_rel_l1:.4e}@{best_epoch+1})",
                flush=True,
            )

        if wait >= early_stopping_patience:
            print(f"Early stopping at epoch {ep+1}", flush=True)
            break

        if eval_deadline_ts is not None and time.time() >= eval_deadline_ts:
            print(f"[time budget] stop after epoch {ep + 1}/{epochs} (leave time for test+save)", flush=True)
            stopped_by_slurm_budget = True
            break

    model.load_state_dict(best_state)
    return best_val_rel_l1, best_epoch + 1, time.time() - t0, stopped_by_slurm_budget


def compute_rel_metrics_normalized(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    rel_l1_sum = 0.0
    rel_l2_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            pred = model(xb, lb)
            b = pred.size(0)
            pf = pred.view(b, -1)
            yf = yb.view(b, -1)
            rel_l1 = torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)
            rel_l2 = torch.norm(pf - yf, p=2, dim=1) / torch.norm(yf, p=2, dim=1).clamp(min=1e-8)
            rel_l1_sum += float(rel_l1.sum().item())
            rel_l2_sum += float(rel_l2.sum().item())
            n += b
    if n == 0:
        return float("nan"), float("nan")
    return rel_l1_sum / n, rel_l2_sum / n


def compute_rel_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mu: np.ndarray,
    y_std: np.ndarray,
):
    """Relative L1/L2 after undo of z-score: y = y_norm * y_std + y_mu (per-channel)."""
    y_mu_t = torch.from_numpy(np.asarray(y_mu, dtype=np.float32)).to(device)
    y_std_t = torch.from_numpy(np.asarray(y_std, dtype=np.float32)).to(device)
    model.eval()
    rel_l1_sum = 0.0
    rel_l2_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lb = lb.to(device)
            pred = model(xb, lb)
            pred_phys = pred * y_std_t + y_mu_t
            y_phys = yb * y_std_t + y_mu_t
            b = pred_phys.size(0)
            pf = pred_phys.view(b, -1)
            yf = y_phys.view(b, -1)
            rel_l1 = torch.abs(pf - yf).sum(dim=1) / torch.abs(yf).sum(dim=1).clamp(min=1e-8)
            rel_l2 = torch.norm(pf - yf, p=2, dim=1) / torch.norm(yf, p=2, dim=1).clamp(min=1e-8)
            rel_l1_sum += float(rel_l1.sum().item())
            rel_l2_sum += float(rel_l2.sum().item())
            n += b
    if n == 0:
        return float("nan"), float("nan")
    return rel_l1_sum / n, rel_l2_sum / n


def _loaded_nodes_from_dataset(d: Dict[str, np.ndarray], nodes: np.ndarray) -> np.ndarray:
    if "neumann_tris" in d:
        tris = np.asarray(d["neumann_tris"])
        if tris.ndim == 2 and tris.shape[1] == 3 and tris.size > 0:
            return np.unique(tris.reshape(-1).astype(np.int64))
    ext = nodes.max(axis=0) - nodes.min(axis=0)
    axis = int(np.argmax(ext))
    rib_len = float(ext[axis])
    x_max = float(nodes[:, axis].max())
    tol = 0.02 * rib_len
    return np.where(nodes[:, axis] > (x_max - tol))[0]


def _compute_tip_pct_from_u(u_nodes_3: np.ndarray, nodes: np.ndarray, loaded_nodes: np.ndarray | None = None) -> float:
    ext = nodes.max(axis=0) - nodes.min(axis=0)
    axis = int(np.argmax(ext))
    rib_len = float(ext[axis])
    umag = np.linalg.norm(u_nodes_3, axis=1)
    if loaded_nodes is not None and len(loaded_nodes) > 0:
        tip = float(np.max(umag[np.asarray(loaded_nodes, dtype=np.int64)]))
    else:
        tip = float(np.max(umag))
    return 100.0 * tip / max(rib_len, 1e-30)



def main():
    p = argparse.ArgumentParser(description="Train AMOR on aircraft-rib linear elasticity dataset")
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset .npz")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Output directory for checkpoint/results")
    p.add_argument("--batch_size", type=int, default=16)
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
    p.add_argument("--k_out", type=int, default=80)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.7)
    p.add_argument("--use_coordinates", action="store_true")
    p.add_argument("--use_film_conditioning", action="store_true", help="Enable FiLM modulation from scalar load")
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--time_buffer_sec",
        type=float,
        default=900.0,
        help="Under Slurm, stop training this many seconds before job end so test+checkpoint can run (override with env AMOR_SLURM_BUFFER_SEC).",
    )
    p.add_argument("--use_solve_ok_only", action="store_true", help="Filter to solve_ok==True if available")
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda but torch.cuda.is_available() is False. "
            "Use a GPU Slurm partition or pass --device cpu."
        )
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    eval_deadline_ts = eval_deadline_walltime(buffer_sec=args.time_buffer_sec)
    if eval_deadline_ts is not None:
        print(f"Slurm time budget: will stop training before {eval_deadline_ts:.0f} unix time (buffer_sec={args.time_buffer_sec})", flush=True)

    d = np.load(args.data_path, allow_pickle=True)
    if args.use_solve_ok_only and "solve_ok" in d.files:
        ok = np.asarray(d["solve_ok"], dtype=bool).reshape(-1)
        if np.count_nonzero(ok) == 0:
            raise RuntimeError("use_solve_ok_only set but zero solve_ok samples.")
        print(f"Filtering solve_ok samples: {np.count_nonzero(ok)}/{len(ok)}", flush=True)
        dd = {k: np.asarray(d[k])[ok] if np.asarray(d[k]).shape[0] == len(ok) else np.asarray(d[k]) for k in d.files}
        d = dd
    else:
        d = {k: np.asarray(d[k]) for k in d.files}

    x = _build_input_channels(d)  # (S,1,N)
    y = _build_targets(d)         # (S,3,N)
    l = _build_load_scalar(d, x.shape[0])  # (S,)
    s = x.shape[0]
    print(f"Dataset samples={s}, nodes={x.shape[2]}, in_ch={x.shape[1]}, out_ch={y.shape[1]}", flush=True)

    tr_idx, va_idx, te_idx = _split_indices(s, args.seed, args.train_frac, args.val_frac)
    print(f"Split sizes: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}", flush=True)

    xn, yn, stats = _normalize_per_channel(x, y, tr_idx)
    l_mu = float(np.mean(l[tr_idx]))
    l_std = float(np.std(l[tr_idx]) + 1e-8)
    ln = ((l - l_mu) / l_std).astype(np.float32)

    xtr, ytr, ltr = _subset_tensors(xn, yn, ln, tr_idx)
    xva, yva, lva = _subset_tensors(xn, yn, ln, va_idx) if len(va_idx) else (
        torch.empty((0, 1, x.shape[2])), torch.empty((0, 3, x.shape[2])), torch.empty((0,))
    )
    xte, yte, lte = _subset_tensors(xn, yn, ln, te_idx) if len(te_idx) else (
        torch.empty((0, 1, x.shape[2])), torch.empty((0, 3, x.shape[2])), torch.empty((0,))
    )

    nodes = np.asarray(d["nodes"], dtype=np.float64)
    elems = np.asarray(d["elements"], dtype=np.int64)
    mass_nodes = lumped_nodal_mass(nodes, elems)
    print(f"Lumped mass nodes: min={mass_nodes.min():.4e} sum={mass_nodes.sum():.4e}", flush=True)

    train_loader = DataLoader(TensorDataset(xtr, ytr, ltr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(xva, yva, lva), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(xte, yte, lte), batch_size=args.batch_size, shuffle=False)

    # PCA reducer expects 4D tensors (S,C,N,M). Use M=1 for node sequence.
    xtr4 = xtr.numpy()[:, :, :, None]
    ytr4 = ytr.numpy()[:, :, :, None]
    k_in = int(args.k_in)
    k_out = int(args.k_out)
    # Output PCA must store >= k_in modes: later blocks encode on output basis with k_in coeffs.
    k_out_pca = max(k_in, k_out)
    rb = PCAReducerMultiChannelMass(
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(xtr4, ytr4, mass_nodes=mass_nodes, k_in=k_in, k_out=k_out_pca)
    k_in = min(k_in, rb.pca_in.n_components_)
    k_out = min(k_out, rb.pca_out.n_components_)
    print(f"PCA rank clamp: k_in={k_in}, k_out={k_out} (output PCA fitted with {rb.pca_out.n_components_} modes)", flush=True)

    model = AMORWithLoadFiLM(
        width=args.width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args.layers,
        dropout=args.dropout,
        init_scale=args.init_scale,
        use_skip_connections=True,
        use_batch_norm=False,
        use_coordinates=args.use_coordinates,
        add_constant_mode=False,
        use_factorized_weights=False,
        use_film_conditioning=args.use_film_conditioning,
        film_hidden=args.film_hidden,
    )
    print(f"AMOR params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    best_val_rel_l1, best_epoch, train_time, stopped_by_slurm_budget = train_model(
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
    te_l1_norm, te_l2_norm = compute_rel_metrics_normalized(model, test_loader, device)
    te_l1_phys, te_l2_phys = compute_rel_metrics(
        model, test_loader, device, stats["y_mu"], stats["y_std"]
    )
    print(
        f"Test rel_normalized:   l1={te_l1_norm:.6e} l2={te_l2_norm:.6e}",
        flush=True,
    )
    print(
        f"Test rel:     l1={te_l1_phys:.6e} l2={te_l2_phys:.6e}",
        flush=True,
    )

    ckpt_path = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_linear_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_rel_l1": best_val_rel_l1,
            "best_epoch": best_epoch,
            "train_rel_l1_normalized": tr_l1,
            "train_rel_l2_normalized": tr_l2,
            "val_rel_l1_normalized": va_l1,
            "val_rel_l2_normalized": va_l2,
            "test_rel_l1": te_l1_phys,
            "test_rel_l2": te_l2_phys,
            "test_rel_l1_normalized": te_l1_norm,
            "test_rel_l2_normalized": te_l2_norm,
            "k_in": k_in,
            "k_out": k_out,
            "width": args.width,
            "layers": args.layers,
            "train_time_s": train_time,
            "use_mass_weighted_pca": True,
            "mass_nodes": mass_nodes.astype(np.float32),
            "normalization": {k: v for k, v in stats.items()},
            "load_norm": {"mu": l_mu, "std": l_std},
            "data_path": args.data_path,
            "stopped_by_slurm_time_budget": stopped_by_slurm_budget,
            "time_buffer_sec": float(args.time_buffer_sec),
            "use_coordinates": bool(args.use_coordinates),
            "use_film_conditioning": bool(args.use_film_conditioning),
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
        "best_val_rel_l1": float(best_val_rel_l1),
        "best_epoch": int(best_epoch),
        "train_rel_l1_normalized": float(tr_l1),
        "val_rel_l1_normalized": float(va_l1),
        "test_rel_l1": float(te_l1_phys),
        "test_rel_l1_normalized": float(te_l1_norm),
        "test_rel_l2_normalized": float(te_l2_norm),
        "train_rel_l2_normalized": float(tr_l2),
        "val_rel_l2_normalized": float(va_l2),
        "test_rel_l2": float(te_l2_phys),
        "checkpoint": ckpt_path,
        "train_time_s": float(train_time),
        "k_in": int(k_in),
        "k_out": int(k_out),
        "width": int(args.width),
        "layers": int(args.layers),
        "use_coordinates": bool(args.use_coordinates),
        "use_film_conditioning": bool(args.use_film_conditioning),
        "film_hidden": int(args.film_hidden),
        "use_mass_weighted_pca": True,
        "stopped_by_slurm_time_budget": bool(stopped_by_slurm_budget),
        "time_buffer_sec": float(args.time_buffer_sec),
    }
    summary_path = os.path.join(args.checkpoint_dir, "amor_aircraft_rib_linear_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)



if __name__ == "__main__":
    main()

