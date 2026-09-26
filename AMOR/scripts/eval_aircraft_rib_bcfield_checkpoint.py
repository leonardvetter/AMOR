#!/usr/bin/env python3
"""
Re-evaluate a saved bcfield AMOR checkpoint on the training test split.

Uses identical split / normalization / PCA bundle as training.
Reports:
  - normalized: rel_l1, rel_l2, mean_l1 (same space as training loss)
  - physical (denormalized u): rel_l1, rel_l2, mean_l1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
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


def _rel_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    b = pred.size(0)
    pf = pred.view(b, -1)
    tf = target.view(b, -1)
    rel_l1 = (torch.abs(pf - tf).sum(dim=1) / torch.abs(tf).sum(dim=1).clamp(min=eps)).sum().item()
    rel_l2 = (
        torch.norm(pf - tf, p=2, dim=1) / torch.norm(tf, p=2, dim=1).clamp(min=eps)
    ).sum().item()
    return rel_l1, rel_l2, b


def _mean_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item() * pred.numel() / pred.size(0))


def _denorm_y(y_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    mu = torch.from_numpy(stats["y_mu"]).to(y_norm.device, y_norm.dtype)
    std = torch.from_numpy(stats["y_std"]).to(y_norm.device, y_norm.dtype)
    return y_norm * std + mu


@torch.no_grad()
def evaluate_loader(model, loader, stats: dict, device: torch.device) -> dict:
    model.eval()
    norm = {"rel_l1": 0.0, "rel_l2": 0.0, "mean_l1": 0.0, "n": 0}
    phys = {"rel_l1": 0.0, "rel_l2": 0.0, "mean_l1": 0.0, "n": 0}
    for xb, yb, fb in loader:
        xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
        pred = model(xb, fb)
        l1, l2, b = _rel_metrics(pred, yb)
        norm["rel_l1"] += l1
        norm["rel_l2"] += l2
        norm["mean_l1"] += _mean_l1(pred, yb) * b
        norm["n"] += b

        pred_p = _denorm_y(pred, stats)
        yb_p = _denorm_y(yb, stats)
        l1, l2, b = _rel_metrics(pred_p, yb_p)
        phys["rel_l1"] += l1
        phys["rel_l2"] += l2
        phys["mean_l1"] += _mean_l1(pred_p, yb_p) * b
        phys["n"] += b

    for d in (norm, phys):
        n = max(1, d["n"])
        d["rel_l1"] /= n
        d["rel_l2"] /= n
        d["mean_l1"] /= n
    return {"rel_l1": phys["rel_l1"], "rel_l2": phys["rel_l2"], "mean_l1": phys["mean_l1"], "n": phys["n"], "normalized": norm}


def _loader(xn, yn, film, idx, batch_size):
    if len(idx) == 0:
        z = xn.shape[2]
        ds = TensorDataset(
            torch.empty((0, xn.shape[1], z)),
            torch.empty((0, yn.shape[1], z)),
            torch.empty((0, 2)),
        )
    else:
        ds = TensorDataset(
            torch.from_numpy(xn[idx]).float(),
            torch.from_numpy(yn[idx]).float(),
            torch.from_numpy(film[idx]).float(),
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--pca_npz", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_solve_ok_only", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    d, n_use = load_bcfield_dataset(args.data_path, use_solve_ok_only=args.use_solve_ok_only)
    x, y = build_field_tensors(d, n_use)
    case = np.asarray(d["load_case_id"], dtype=np.int64).reshape(-1)[:n_use]
    nu = np.asarray(d["nu"], dtype=np.float32).reshape(-1)[:n_use]
    del d

    rb, stats, film_norm, pca_meta = PCAReducerMultiChannel.load_training_bundle(args.pca_npz)
    if pca_meta["seed"] != args.seed or pca_meta["train_frac"] != args.train_frac or pca_meta["val_frac"] != args.val_frac:
        raise ValueError("PCA bundle split meta != eval args")

    tr_idx, va_idx, te_idx = split_indices(n_use, args.seed, args.train_frac, args.val_frac)
    xn, yn, stats_re = normalize_per_channel(x, y, tr_idx)
    del x, y
    for key in ("x_mu", "x_std", "y_mu", "y_std"):
        if not np.allclose(stats[key], stats_re[key], rtol=1e-5, atol=1e-6):
            raise ValueError(f"PCA bundle {key} mismatch vs recomputed train normalization")

    nun = ((nu - film_norm["nu_mu"]) / film_norm["nu_std"]).astype(np.float32)
    casen = ((case - film_norm["case_mu"]) / film_norm["case_std"]).astype(np.float32)
    film = np.stack([nun, casen], axis=1).astype(np.float32)

    k_in = int(ckpt.get("k_in", 15))
    k_out = int(ckpt.get("k_out", 2))
    width = int(ckpt.get("width", 24))
    layers = int(ckpt.get("layers", 4))
    model = AMORWithFieldInputs(
        width=width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=layers,
        use_batch_norm=bool(ckpt.get("use_batch_norm", False)),
        use_film_conditioning=True,
        film_in_dim=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint epoch keys k_in={k_in} k_out={k_out} w={width} L={layers}", flush=True)

    test_loader = _loader(xn, yn, film, te_idx, args.batch_size)

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "data_path": os.path.abspath(args.data_path),
        "pca_npz": os.path.abspath(args.pca_npz),
        "split": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "test_all": evaluate_loader(model, test_loader, stats, device),
    }
    for cid in sorted(set(int(c) for c in np.unique(case[te_idx]))):
        te_c = te_idx[case[te_idx] == cid]
        metrics = evaluate_loader(model, _loader(xn, yn, film, te_c, args.batch_size), stats, device)
        metrics["n"] = int(len(te_c))
        out[f"test_case_id_{cid}"] = metrics
        print(
            f"test case_id={cid} n={len(te_c)} "
            f"rel_l1={metrics['rel_l1']:.6f} "
            f"rel_l2={metrics['rel_l2']:.6f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Patch training summary sidecar with all case metrics if present.
    summary_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "amor_aircraft_rib_bcfield_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for cid in sorted(set(int(c) for c in np.unique(case[te_idx]))):
            m = out[f"test_case_id_{cid}"]
            summary[f"test_rel_l1_case_id_{cid}"] = float(m["rel_l1"])
            summary[f"test_rel_l2_case_id_{cid}"] = float(m["rel_l2"])
            summary[f"test_n_case_id_{cid}"] = int(m["n"])
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated {summary_path}", flush=True)

    t = out["test_all"]
    print(
        f"test rel_l1_normalized={t['normalized']['rel_l1']:.6f} rel_l2={t['normalized']['rel_l2']:.6f} "
        f"mean_l1={t['normalized']['mean_l1']:.6f}",
        flush=True,
    )
    print(
        f"test rel_l1={t['rel_l1']:.6f} rel_l2={t['rel_l2']:.6f} "
        f"mean_l1={t['mean_l1']:.6f}",
        flush=True,
    )
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
