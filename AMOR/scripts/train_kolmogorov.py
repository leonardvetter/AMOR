#!/usr/bin/env python3
"""
Train FNO, AMOR and PCANet on Kolmogorov flow.

One entry point for the three published Reynolds numbers; ``--re`` selects the
dataset and the per-Re defaults used in the paper. Every default can be
overridden on the command line.

    python train_kolmogorov.py --re 40
    python train_kolmogorov.py --re 5000 --models AMOR
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from AMOR.data.kolmogorov import DEFAULT_DATA_PATHS, load_kolmogorov_dataset  # noqa: E402
from AMOR.data.splits import (  # noqa: E402
    SPLIT_RULE,
    apply_weight_seed,
    shuffle_generator,
)
from AMOR.models.FNO_model import FNO2d  # noqa: E402
from AMOR.models.PCANet import PCANet  # noqa: E402
from AMOR.models.PCAReducer import PCAReducerMultiChannel  # noqa: E402
from AMOR.models.amor import AMOR  # noqa: E402
from AMOR.training.loop_timestep import (  # noqa: E402
    compute_rel_metrics,
    compute_rel_metrics_normalized,
    train_model,
)

# Per-Re settings of the published runs. Re=500 fits the PCA basis with the
# randomized SVD solver; the other two use the exact solver.
RE_DEFAULTS = {
    40: {
        "width": 24,
        "layers": 4,
        "k_in": 144,
        "k_out": 144,
        "pca_randomized": False,
    },
    500: {
        "width": 32,
        "layers": 5,
        "k_in": 144,
        "k_out": 144,
        "pca_randomized": True,
    },
    5000: {
        "width": 40,
        "layers": 5,
        "k_in": 144,
        "k_out": 144,
        "pca_randomized": False,
    },
}

MODEL_NAMES = ("FNO", "AMOR", "PCANet")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO/AMOR/PCANet on Kolmogorov flow")
    parser.add_argument("--re", type=int, required=True, choices=sorted(RE_DEFAULTS),
                        help="Reynolds number; selects dataset and per-Re defaults")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES,
                        help="Models to train")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to Kolmogorov flow dataset (default: per-Re array)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=3.0, help="Gradient clipping norm")
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time step between snapshots for the Euler update")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints (default: ${AMOR_DATA_ROOT}/checkpoints)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_in", type=int, default=None,
                        help="PCA input modes for AMOR and PCANet (clamped to fitted PCA rank)")
    parser.add_argument("--k_out", type=int, default=None,
                        help="PCA output modes for AMOR and PCANet (clamped to fitted PCA rank)")
    parser.add_argument("--width", type=int, default=None, help="Hidden width for AMOR and FNO")
    parser.add_argument("--layers", type=int, default=None,
                        help="AMOR layers and FNO spectral blocks")
    parser.add_argument("--pca_fit_samples", type=int, default=0,
                        help="Number of training samples used to fit the PCA basis (0 = all)")
    parser.add_argument("--use_coordinates", action="store_true",
                        help=argparse.SUPPRESS)  # kept off; not used in reported runs
    parser.add_argument("--fno_add_coords", action="store_true",
                        help="FNO: pass add_coords=True (coordinate channels in lift)")
    args = parser.parse_args()

    defaults = RE_DEFAULTS[args.re]
    for key in ("width", "layers", "k_in", "k_out"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.data_path is None:
        args.data_path = DEFAULT_DATA_PATHS[args.re]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.expandvars("${AMOR_DATA_ROOT}/checkpoints")
    args.pca_randomized = defaults["pca_randomized"]
    return args


def fit_pca(x_train, y_train, args):
    if 0 < args.pca_fit_samples < len(x_train):
        import numpy as np

        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x_train), size=args.pca_fit_samples, replace=False)
        x_fit, y_fit = x_train[idx], y_train[idx]
        print(f"Using {len(x_fit)} / {len(x_train)} training samples for PCA fit")
    else:
        x_fit, y_fit = x_train, y_train
        print(f"Using all {len(x_fit)} training samples for PCA fit")

    solver_kwargs = {"svd_solver": "randomized", "random_state": args.seed} if args.pca_randomized else {"random_state": args.seed}
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
        **solver_kwargs,
    ).fit(x_fit, y_fit - x_fit, n_components_in=args.k_in, n_components_out=args.k_out)
    rb.output_on_increment = True

    k_in = min(args.k_in, rb.pca_in.n_components_)
    k_out = min(args.k_out, rb.pca_out.n_components_)
    rb.truncate(k_in, k_out)
    if k_in != args.k_in or k_out != args.k_out:
        print(f"Note: k_in/k_out clamped to PCA rank: requested k_in={args.k_in}, k_out={args.k_out}")
    print(f"PCA components: k_in={k_in}, k_out={k_out}")
    print(f"Boundary condition awareness: bc_weight={rb.bc_weight}, threshold={rb.boundary_threshold}")
    return rb, k_in, k_out


def build_models(args, rb, k_in, k_out):
    built = {}
    for name in args.models:
        apply_weight_seed(args.seed)
        if name == "FNO":
            model = FNO2d(
                in_channels=1,
                out_channels=1,
                width=args.width,
                modes=(12, 12),
                n_layers=args.layers,
                add_coords=args.fno_add_coords,
            )
        elif name == "AMOR":
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
                use_factorized_weights=False,
            )
        else:
            model = PCANet(
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                hidden=1024,
                depth=5,
                dropout=0.0,
                init_scale=0.1,
            )
        built[name] = (model, f"{name.lower()}_kolmogorov_Re{args.re}_best.pt")
    return built


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    (x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t,
     x_train, y_train, xy_mean, xy_std    ) = load_kolmogorov_dataset(
        args.data_path, return_norm_stats=True, seed=args.seed)
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator(args.seed),
    )
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=args.batch_size, shuffle=False)
    print(f"\nData loaders created: batch_size={args.batch_size}")

    print("\n" + "=" * 70)
    print("COMPUTING PCA")
    print("=" * 70)
    rb, k_in, k_out = fit_pca(x_train, y_train, args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for model_name, (model, ckpt_name) in build_models(args, rb, k_in, k_out).items():
        print("\n" + "=" * 70)
        print(f"TRAINING {model_name}")
        print("=" * 70)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name} Parameters: {n_params:,}")
        from AMOR.training.torch_utils import run_external_protocol
        if run_external_protocol(
            model, train_loader, device,
            lambda m, x, y: torch.nn.functional.l1_loss(x + args.dt * m(x), y),
            {"entry": os.environ.get("AMOR_PROTOCOL_ENTRY", f"kolm_Re{args.re}_{model_name}"),
             "dataset": f"kolmogorov_Re{args.re}", "model": model_name,
             "params": n_params, "precision": "fp32",
             "block_checkpoint": os.environ.get("AMOR_BLOCK_CHECKPOINT", "1")},
        ):
            continue

        _, _, best_val_rel_l1, training_time, best_epoch, _ = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            dt=args.dt,
        )

        train_rel_l1, train_rel_l2 = compute_rel_metrics_normalized(model, train_loader, device, dt=args.dt)
        val_rel_l1, val_rel_l2 = compute_rel_metrics_normalized(model, val_loader, device, dt=args.dt)
        test_rel_l1, test_rel_l2 = compute_rel_metrics(model, test_loader, device, xy_mean, xy_std, dt=args.dt)

        print(f"Train rel_l1_normalized: {train_rel_l1:.6f}, Train rel_l2: {train_rel_l2:.6f}")
        print(f"Val rel_l1_normalized: {val_rel_l1:.6f}, Val rel_l2: {val_rel_l2:.6f}")
        print(f"Test rel_l1: {test_rel_l1:.6f}, Test rel_l2: {test_rel_l2:.6f}")

        checkpoint_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_rel_l1_normalized': train_rel_l1,
            'train_rel_l2_normalized': train_rel_l2,
            'val_rel_l1_normalized': val_rel_l1,
            'val_rel_l2_normalized': val_rel_l2,
            'test_rel_l1': test_rel_l1,
            'test_rel_l2': test_rel_l2,
            'best_val_rel_l1': best_val_rel_l1,
            'best_val_loss': best_val_rel_l1,
            'best_epoch': best_epoch,
            'model_name': model_name,
            'reynolds': args.re,
            'k_in': k_in,
            'k_out': k_out,
            'width': args.width,
            'layers': args.layers,
            'use_coordinates': args.use_coordinates,
            'fno_add_coords': args.fno_add_coords,
        }, checkpoint_path)
        metrics_path = os.path.join(
            args.checkpoint_dir, f"{model_name.lower()}_kolmogorov_Re{args.re}_s{args.seed}_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump({
                "model": model_name,
                "seed": int(args.seed),
                "test_rel_l1": float(test_rel_l1),
                "test_rel_l2": float(test_rel_l2),
                "num_params": int(sum(p.numel() for p in model.parameters())),
                "training_time_s": float(training_time),
                "best_epoch": int(best_epoch),
                "train_sec_per_epoch": float(training_time) / max(1, int(best_epoch)),
                "k_in": int(k_in),
                "k_out": int(k_out),
                "width": int(args.width),
                "layers": int(args.layers),
                "checkpoint": checkpoint_path,
                "data_path": args.data_path,
                "split_rule": SPLIT_RULE,
            }, f, indent=2)
        print(f"Saved {metrics_path}")

        print(f"Best validation rel_l1: {best_val_rel_l1:.4e} (epoch {best_epoch})")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()
