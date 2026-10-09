#!/usr/bin/env python3
"""
AMOR on Kolmogorov Re40 with alternative spatial bases.

Same data split, normalization, Euler residual, optimizer, and hyperparameters as
train_amor_kolmogorov_Re40.py. By default:
  - input basis: snapshots u(t)
  - output basis: increments Δu = u(t+1) - u(t)
Use --no-output_on_increment to fit the output basis on next states u(t+1)
(matching the main-table AMOR PCA protocol).
"""

import argparse
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
_PKG_ROOT = os.path.dirname(SCRIPT_DIR)

from AMOR.paths import expand  # noqa: E402

from AMOR.models.amor import AMOR  # noqa: E402
from AMOR.datasets.kolmogorov import load_kolmogorov_dataset  # noqa: E402
from AMOR.training.loop_timestep import (  # noqa: E402
    compute_rel_metrics,
    compute_rel_metrics_normalized,
    train_model,
)

from AMOR.models.PCAReducer import (  # noqa: E402
    BASIS_TYPES,
    PCAReducerMultiChannel,
    fit_alternative_basis,
    fit_dmd_hybrid_basis,
    fit_random_init_basis,
    kolmogorov_dmd_aligned_pairs,
    kolmogorov_train_trajectory_indices,
)

LEARNED_BASIS_TYPES = ("learned_qr", "learned_cayley", "learned_householder")
ALL_BASIS_TYPES = tuple(BASIS_TYPES) + LEARNED_BASIS_TYPES


def _is_learned(basis_type: str) -> bool:
    return basis_type in LEARNED_BASIS_TYPES


def _learnable_param(basis_type: str) -> str:
    return basis_type.replace("learned_", "")


@torch.no_grad()
def _infer_ms(model, loader, device) -> float:
    model.eval()
    xb = next(iter(loader))[0].to(device)
    bs = xb.size(0)
    for _ in range(2):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (5 * bs) * 1000.0


def _normalize_basis_type(basis_type: str) -> str:
    if basis_type == "pca_increment":
        return "pca"
    return basis_type


def _fit_basis(
    basis_type: str,
    X_train,
    Y_train,
    data_path: str,
    random_state: int,
    laplacian_periodic: bool,
    jacobi_alpha: float,
    jacobi_beta: float,
    k_in: int,
    k_out: int,
    dmd_svd_rank: int | None = None,
    xy_mean: float | None = None,
    xy_std: float | None = None,
    x_val_t=None,
    y_val_t=None,
    output_on_increment: bool = True,
):
    # Output-side snapshots used to center / fit the decoder basis.
    # Increment: Δu = u(t+1)-u(t). Next-state: u(t+1) (matches benchmark AMOR PCA).
    y_out = (Y_train - X_train) if output_on_increment else Y_train

    if basis_type == "pca":
        rb = PCAReducerMultiChannel(
            bc_weight=1.0,
            boundary_threshold=0.1,
            use_boundary_weighting=False,
        ).fit(X_train, y_out)
        rb.output_on_increment = output_on_increment
        return rb

    if _is_learned(basis_type):
        rb = fit_random_init_basis(X_train, y_out, random_state=random_state)
        rb.output_on_increment = output_on_increment
        return rb

    if basis_type == "dmd_hybrid":
        raw = np.load(data_path, mmap_mode="r")
        n_traj = raw.shape[0]
        train_idx = kolmogorov_train_trajectory_indices(n_traj, seed=42)
        xy_mean = float(np.concatenate([X_train.flatten(), Y_train.flatten()]).mean())
        xy_std = float(np.concatenate([X_train.flatten(), Y_train.flatten()]).std())
        _, _, d_out, d_next = kolmogorov_dmd_aligned_pairs(
            data_path, train_idx, xy_mean, xy_std
        )
        rb = fit_dmd_hybrid_basis(
            X_train,
            Y_train,
            dmd_output_next=d_next,
            random_state=random_state,
            dmd_svd_rank=dmd_svd_rank,
        )
        # DMD hybrid always builds an increment-oriented output side.
        rb.output_on_increment = True
        return rb

    rb = fit_alternative_basis(
        X_train,
        y_out,
        basis_type,
        random_state=random_state,
        laplacian_periodic=laplacian_periodic,
        jacobi_alpha=jacobi_alpha,
        jacobi_beta=jacobi_beta,
    )
    rb.output_on_increment = output_on_increment
    return rb


def main():
    parser = argparse.ArgumentParser(description="AMOR Kolmogorov Re40 basis ablation")
    parser.add_argument("--data_path", type=str, default=expand("${AMOR_DATA_ROOT}/2D_NS_Re40.npy"))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=3.0)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k_in", type=int, default=100)
    parser.add_argument("--k_out", type=int, default=100)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--use_factorized_weights",
        choices=("auto", "true", "false"),
        default="auto",
        help="Factorized T (default auto).",
    )
    parser.add_argument("--tag", default=None, help="Filename tag override")
    parser.add_argument("--use_coordinates", action="store_true")
    parser.add_argument(
        "--basis_type",
        type=str,
        default="pca",
        choices=ALL_BASIS_TYPES + ("pca_increment",),
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--laplacian_periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--jacobi_alpha", type=float, default=1.0)
    parser.add_argument("--jacobi_beta", type=float, default=1.0)
    parser.add_argument(
        "--dmd_svd_rank",
        type=int,
        default=None,
        help="DMD internal SVD rank (dmd_hybrid default: k_out)",
    )
    parser.add_argument(
        "--output_on_increment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, fit output basis on Δu; with --no-output_on_increment fit on u(t+1).",
    )
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    args.basis_type = _normalize_basis_type(args.basis_type)


    if args.checkpoint_dir is None:
        args.checkpoint_dir = expand("${AMOR_DATA_ROOT}/checkpoints/kolmogorov_re40_basis")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    label = args.tag or args.basis_type
    metrics_path = os.path.join(args.checkpoint_dir, f"amor_Re40_{label}_metrics.json")
    if args.skip_if_exists and os.path.isfile(metrics_path):
        print(f"Skip: {metrics_path} exists")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    (
        X_train_t,
        Y_train_t,
        X_val_t,
        Y_val_t,
        X_test_t,
        Y_test_t,
        X_train,
        Y_train,
        xy_mean,
        xy_std,
    ) = load_kolmogorov_dataset(args.data_path, return_norm_stats=True)

    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, Y_test_t), batch_size=args.batch_size, shuffle=False
    )

    out_side = "Δu = u(t+1) - u(t)" if args.output_on_increment else "u(t+1) next state"
    print("\n" + "=" * 70)
    print(f"FITTING BASIS: {args.basis_type}")
    print("  pca_in / input side: snapshots u(t)")
    print(f"  pca_out / output side: {out_side}")
    print("=" * 70)
    dmd_rank = args.dmd_svd_rank
    if args.basis_type == "dmd_hybrid" and dmd_rank is None:
        dmd_rank = args.k_out
    rb = _fit_basis(
        args.basis_type,
        X_train,
        Y_train,
        args.data_path,
        args.random_state,
        args.laplacian_periodic,
        args.jacobi_alpha,
        args.jacobi_beta,
        args.k_in,
        args.k_out,
        dmd_svd_rank=dmd_rank,
        xy_mean=xy_mean,
        xy_std=xy_std,
        x_val_t=X_val_t,
        y_val_t=Y_val_t,
        output_on_increment=args.output_on_increment,
    )

    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    if hasattr(rb, "truncate"):
        rb.truncate(k_in, k_out)
    if k_in != args.k_in or k_out != args.k_out:
        print(f"Note: k clamped: requested {args.k_in}/{args.k_out}, using {k_in}/{k_out}")
    print(f"k_in={k_in}, k_out={k_out}")

    learned = _is_learned(args.basis_type)
    fact = {"auto": "auto", "true": True, "false": False}[args.use_factorized_weights]
    model = AMOR(
        width=args.width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args.layers,
        dropout=0.0,
        init_scale=0.7,
        use_skip_connections=True,
        use_batch_norm=False,
        use_coordinates=False,
        add_constant_mode=False,
        use_factorized_weights=fact,
        trainable_basis=learned,
        trainable_basis_param=_learnable_param(args.basis_type) if learned else "qr",
        share_bases=not learned,
    )

    print("\n" + "=" * 70)
    print("TRAINING AMOR")
    print("=" * 70)
    params = sum(p.numel() for p in model.parameters())
    n_blocks = model.n_nonlocal_blocks()
    print(f"Parameters: {params:,}  blocks={n_blocks}")

    t0 = time.time()
    learned = _is_learned(args.basis_type)
    es_patience = (args.epochs + 1) if learned else args.early_stopping_patience
    _, _, best_val_rel_l1, training_time, best_epoch, epochs_ran = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
        early_stopping_patience=es_patience,
        dt=args.dt,
    )

    train_rel_l1, train_rel_l2 = compute_rel_metrics_normalized(model, train_loader, device, dt=args.dt)
    val_rel_l1, val_rel_l2 = compute_rel_metrics_normalized(model, val_loader, device, dt=args.dt)
    test_rel_l1, test_rel_l2 = compute_rel_metrics(model, test_loader, device, xy_mean, xy_std, dt=args.dt)
    infer_ms = _infer_ms(model, test_loader, device)

    print(f"Train rel_l1_normalized: {train_rel_l1:.6f}, Train rel_l2: {train_rel_l2:.6f}")
    print(f"Val rel_l1_normalized: {val_rel_l1:.6f}, Val rel_l2: {val_rel_l2:.6f}")
    print(f"Test rel_l1: {test_rel_l1:.6f}, Test rel_l2: {test_rel_l2:.6f}")

    ckpt_path = os.path.join(args.checkpoint_dir, f"amor_Re40_{label}_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "basis_type": args.basis_type,
            "output_on_increment": bool(args.output_on_increment),
            "trainable_basis": learned,
            "trainable_basis_param": _learnable_param(args.basis_type) if learned else None,
            "train_rel_l1_normalized": train_rel_l1,
            "train_rel_l2_normalized": train_rel_l2,
            "val_rel_l1_normalized": val_rel_l1,
            "val_rel_l2_normalized": val_rel_l2,
            "test_rel_l1": test_rel_l1,
            "test_rel_l2": test_rel_l2,
            "best_val_rel_l1": best_val_rel_l1,
            "best_epoch": best_epoch,
            "k_in": k_in,
            "k_out": k_out,
            "width": args.width,
            "layers": args.layers,
            "wall_time_sec": training_time,
        },
        ckpt_path,
    )

    summary = {
        "basis_type": args.basis_type,
        "output_on_increment": bool(args.output_on_increment),
        "trainable_basis": learned,
        "trainable_basis_param": _learnable_param(args.basis_type) if learned else None,
        "k_in": k_in,
        "k_out": k_out,
        "width": args.width,
        "layers": args.layers,
        "n_blocks": n_blocks,
        "num_params": params,
        "batch_size": args.batch_size,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_rel_l1": best_val_rel_l1,
        "train_rel_l1_normalized": train_rel_l1,
        "train_rel_l2_normalized": train_rel_l2,
        "val_rel_l1_normalized": val_rel_l1,
        "val_rel_l2_normalized": val_rel_l2,
        "test_rel_l1": test_rel_l1,
        "test_rel_l2": test_rel_l2,
        "training_time_sec": training_time,
        "inference_ms_per_sample": infer_ms,
        "total_wall_sec": time.time() - t0,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best val rel_l1: {best_val_rel_l1:.4e} (epoch {best_epoch})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
