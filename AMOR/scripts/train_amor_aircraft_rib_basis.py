#!/usr/bin/env python3
"""
AMOR basis ablation on the aircraft-rib cantilever-tip set.

Same mass-PCA, FiLM, and coordinate settings as
train_amor_aircraft_rib_easy_10000_best_mass.py when those flags are passed;
only the reduced basis changes across the sweep.
"""

from __future__ import annotations

import argparse
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from AMOR.models.mesh_bases import fit_mesh_alternative_basis_mass  # noqa: E402
from AMOR.models.mass.mesh_mass import lumped_nodal_mass  # noqa: E402
from AMOR.models.mass.PCAReducer_mass import (  # noqa: E402
    PCAReducerMultiChannelMass,
    fit_random_init_basis_mass,
)
from AMOR.models.mass.AMOR_load_film_mass import AMORWithLoadFiLM  # noqa: E402

from AMOR.paths import expand  # noqa: E402


# Reuse exact data split / train / metrics from the reference mass trainer.
_REF = os.path.join(SCRIPT_DIR, "train_amor_aircraft_rib_linear_mass.py")
_spec = importlib.util.spec_from_file_location("rib_linear_mass", _REF)
_lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lm)

LEARNED = ("learned_qr", "learned_cayley")
FIXED = ("pca", "chebyshev", "legendre", "jacobi", "laplacian")
ALL = FIXED + LEARNED

DATA_PATH = expand("${AMOR_DATA_ROOT}/aircraft_rib/datasets/aircraft_rib_cantilever_tip.npz")
OUT_DIR = expand("${AMOR_DATA_ROOT}/aircraft_rib/amor_rib_h15_basis")


def fit_basis(bt, xtr4, ytr4, nodes, elems, mass_nodes, k_in, k_out, seed):
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
    xb, _, lb = next(iter(loader))
    xb, lb = xb.to(dev), lb.to(dev)
    bs = xb.size(0)
    for _ in range(2):
        model(xb, lb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        model(xb, lb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (5 * bs) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--checkpoint_dir", default=OUT_DIR)
    p.add_argument("--basis_type", choices=ALL, default="pca")
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument(
        "--tag",
        default=None,
        help="Filename tag (default: basis_type). Writes amor_rib_{tag}_metrics.json / _best.pt",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--k_in", type=int, default=100)
    p.add_argument("--k_out", type=int, default=30)
    p.add_argument("--init_scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_solve_ok_only", action="store_true")
    p.add_argument("--use_coordinates", action="store_true")
    p.add_argument("--use_film_conditioning", action="store_true")
    p.add_argument("--film_hidden", type=int, default=64)
    p.add_argument("--time_buffer_sec", type=float, default=1200.0)
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = args.tag or args.basis_type
    mp = os.path.join(args.checkpoint_dir, f"amor_rib_{tag}_metrics.json")
    if args.skip_if_exists and os.path.isfile(mp):
        print(f"Skip {mp}", flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    dev = torch.device(args.device)
    eval_deadline = _lm.eval_deadline_walltime(buffer_sec=args.time_buffer_sec)

    d = np.load(args.data_path, allow_pickle=True)
    if args.use_solve_ok_only and "solve_ok" in d.files:
        ok = np.asarray(d["solve_ok"], dtype=bool).reshape(-1)
        d = {k: np.asarray(d[k])[ok] if np.asarray(d[k]).shape[0] == len(ok) else np.asarray(d[k]) for k in d.files}
    else:
        d = {k: np.asarray(d[k]) for k in d.files}

    x = _lm._build_input_channels(d)
    y = _lm._build_targets(d)
    l = _lm._build_load_scalar(d, x.shape[0])
    nodes = np.asarray(d["nodes"], dtype=np.float64)
    elems = np.asarray(d["elements"], dtype=np.int64)
    mass_nodes = lumped_nodal_mass(nodes, elems)
    print(f"samples={x.shape[0]} nodes={x.shape[2]} basis={args.basis_type}", flush=True)

    tr_idx, va_idx, te_idx = _lm._split_indices(x.shape[0], args.seed, args.train_frac, args.val_frac)
    xn, yn, stats = _lm._normalize_per_channel(x, y, tr_idx)
    l_mu = float(np.mean(l[tr_idx]))
    l_std = float(np.std(l[tr_idx]) + 1e-8)
    ln = ((l - l_mu) / l_std).astype(np.float32)

    from torch.utils.data import DataLoader, TensorDataset

    def _ld(idx, sh):
        xb, yb, lb = _lm._subset_tensors(xn, yn, ln, idx)
        return DataLoader(TensorDataset(xb, yb, lb), batch_size=args.batch_size, shuffle=sh)

    tr_ld = _ld(tr_idx, True)
    va_ld = _ld(va_idx, False)
    te_ld = _ld(te_idx, False)

    xtr4 = xn[tr_idx][:, :, :, None]
    ytr4 = yn[tr_idx][:, :, :, None]
    rb = fit_basis(args.basis_type, xtr4, ytr4, nodes, elems, mass_nodes, args.k_in, args.k_out, args.seed)
    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    print(f"rank clamp k_in={k_in} k_out={k_out}", flush=True)

    learned = args.basis_type in LEARNED
    model = AMORWithLoadFiLM(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, init_scale=args.init_scale, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=args.use_coordinates,
        use_film_conditioning=args.use_film_conditioning, film_hidden=args.film_hidden,
        trainable_basis=learned,
        trainable_basis_param=args.basis_type.replace("learned_", "") if learned else "qr",
    )
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    es_patience = (args.epochs + 1) if learned else args.early_stopping_patience
    _, best_ep, ttrain, _ = _lm.train_model(
        model, tr_ld, va_ld, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        early_stopping_patience=es_patience, device=dev,
        eval_deadline_ts=eval_deadline,
    )
    te_l1n, te_l2n = _lm.compute_rel_metrics_normalized(model, te_ld, dev)
    te_l1p, te_l2p = _lm.compute_rel_metrics(model, te_ld, dev, stats["y_mu"], stats["y_std"])
    infer = infer_ms(model, te_ld, dev)

    summary = {
        "dataset": "aircraft_rib_cantilever_tip",
        "data_path": args.data_path,
        "basis_type": args.basis_type,
        "tag": tag,
        "trainable_basis": learned,
        "trainable_basis_param": args.basis_type.replace("learned_", "") if learned else None,
        "use_mass_weighted_pca": True,
        "use_coordinates": bool(args.use_coordinates),
        "use_film_conditioning": bool(args.use_film_conditioning),
        "use_solve_ok_only": bool(args.use_solve_ok_only),
        "k_in": k_in,
        "k_out": k_out,
        "width": args.width,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "best_epoch": best_ep,
        "test_rel_l1": float(te_l1p),
        "test_rel_l2": float(te_l2p),
        "test_rel_l1_normalized": float(te_l1n),
        "test_rel_l2_normalized": float(te_l2n),
        "train_sec_per_epoch": ttrain / max(1, best_ep),
        "inference_ms_per_sample": infer,
        "training_time_sec": ttrain,
        "split": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
    }
    with open(mp, "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), **summary},
               os.path.join(args.checkpoint_dir, f"amor_rib_{tag}_best.pt"))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
