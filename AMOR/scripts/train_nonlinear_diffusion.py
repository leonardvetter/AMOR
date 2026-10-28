#!/usr/bin/env python3
"""
Train AMOR, FNO, and PCANet on nonlinear diffusion datasets
(nonlinear_fvar_{fvar}_flen_{flen}_mvar_{mvar}_mlen_{mlen}.npz).

Trains each model on each prior, then cross-eval on all three. Results saved
incrementally per (dataset, model); interrupt-safe resume.

Hyperparams: AdamW lr=2e-3 wd=1e-4; CosineAnnealingLR T_max=500; L1 train loss;
AMOR k_in=k_out=144; PCANet hidden=1024 depth=5; FNO modes=(12,12) width=24 layers=4.
Split: sequential 70%/20%/10% of first num_samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import argparse
import copy
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))

from AMOR.models.amor import AMOR
from AMOR.models.FNO_model import FNO2d
from AMOR.models.PCANet import PCANet
from AMOR.models.PCAReducer import PCAReducerMultiChannel
from AMOR.data.splits import SPLIT_RULE, apply_weight_seed, seeded_split, shuffle_generator
from AMOR.training.loop2d import l1_loss, train_model

HYPERPARAMETERS = {
    'layers': 4,
    'width': 24,
    'k': 144,
    'fno_modes': (12, 12),
    'lr': 2e-3,
    'weight_decay': 1e-4,
    'max_grad_norm': 3.0,
    'epochs': 300,
    'early_stopping_patience': 50,
    'early_stopping_min_delta': 0.0,
    'batch_size': 32,
    'seed': 42,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'test_frac': 0.1,
    'num_samples': 10000,
}

DATASETS = {
    'nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.npz': 'fvar_5_flen_0.2_mvar_1_mlen_0.3',
    'nonlinear_fvar_2_flen_0.4_mvar_0.5_mlen_0.5.npz': 'fvar_2_flen_0.4_mvar_0.5_mlen_0.5',
    'nonlinear_fvar_10_flen_0.1_mvar_2_mlen_0.15.npz': 'fvar_10_flen_0.1_mvar_2_mlen_0.15',
}




def l1_loss(pred, target):
    """Mean per-sample relative L1."""
    batch = pred.shape[0]
    pred = pred.reshape(batch, -1)
    target = target.reshape(batch, -1)
    return ((pred - target).abs().sum(1) / target.abs().sum(1).clamp_min(1e-8)).mean()


def load_nonlinear_dataset(dataset_path, num_samples=10000):
    data = np.load(dataset_path, allow_pickle=True)
    f = data['f'][:num_samples]
    m = data['m'][:num_samples]
    u = data['u'][:num_samples]
    X = np.stack([f, m], axis=1)
    Y = u[:, np.newaxis, :, :]
    return X.astype(np.float32), Y.astype(np.float32)




def evaluate_model(model, loader, device, y_mean_np, y_std_np):
    y_mean_t = torch.from_numpy(y_mean_np).float().to(device)
    y_std_t = torch.from_numpy(y_std_np).float().to(device)

    model.eval()
    rel_l1_errors = []
    rel_l2_errors = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            pred_denorm = pred * y_std_t + y_mean_t
            yb_denorm = yb * y_std_t + y_mean_t

            for i in range(pred_denorm.size(0)):
                pred_f = pred_denorm[i, 0].cpu().numpy()
                true_f = yb_denorm[i, 0].cpu().numpy()

                l1_err = np.abs(pred_f - true_f).sum()
                l1_norm = np.abs(true_f).sum()
                rel_l1_errors.append(l1_err / (l1_norm + 1e-10))

                l2_err = np.sqrt(np.sum((pred_f - true_f) ** 2))
                l2_norm = np.sqrt(np.sum(true_f ** 2))
                rel_l2_errors.append(l2_err / (l2_norm + 1e-10))

    return np.mean(rel_l1_errors) * 100, np.mean(rel_l2_errors) * 100


def rebuild_csvs(results_dir):
    """Reconstruct aggregate CSVs from all saved per-run JSON files."""
    results_main = []
    results_cross = []

    for fname in os.listdir(results_dir):
        if not fname.endswith('_results.json'):
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            results_main.append(data['main'])
            results_cross.extend(data['cross'])
        except Exception as e:
            print(f"  Warning: could not read {fname}: {e}")

    if results_main:
        pd.DataFrame(results_main).to_csv(
            os.path.join(results_dir, 'nonlinear_main_results.csv'), index=False)
    if results_cross:
        pd.DataFrame(results_cross).to_csv(
            os.path.join(results_dir, 'nonlinear_cross_dataset_results.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Train nonlinear diffusion model comparison runs.')
    parser.add_argument(
        '--dataset',
        choices=sorted(set(DATASETS.values())),
        default=None,
        help='Train only this dataset variant (default: all).',
    )
    parser.add_argument(
        '--model',
        choices=['AMOR', 'FNO', 'PCANet'],
        default=None,
        help='Train only this model (default: all).',
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--data_path', default=None, help='Train this npz only (same sequential split).')
    parser.add_argument('--skip_cross', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure conda env bin is in PATH so torch.compile can find g++ for C++ codegen
    import os as _os
    _env_bin = _os.path.join(sys.prefix, 'bin')
    if _env_bin not in _os.environ.get('PATH', ''):
        _os.environ['PATH'] = _env_bin + ':' + _os.environ.get('PATH', '')

    if args.seed is not None:
        HYPERPARAMETERS['seed'] = int(args.seed)
    data_dir = expand('${AMOR_SCRATCH_DATA}')  # rename path on a different cluster
    results_dir = args.results_dir or expand('${AMOR_SCRATCH_DATA}/nonlinear_model_comparison')
    os.makedirs(results_dir, exist_ok=True)
    if args.data_path:
        data_dir = os.path.dirname(os.path.abspath(args.data_path))
        base = os.path.basename(args.data_path)
        datasets = {base: os.path.splitext(base)[0]}
    else:
        datasets = DATASETS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")

    for train_dataset_name, train_dataset_type in datasets.items():
        if args.dataset is not None and train_dataset_type != args.dataset:
            continue
        print(f"\n{'='*70}")
        print(f"Training on: {train_dataset_name} ({train_dataset_type})")
        print(f"{'='*70}")

        train_dataset_path = os.path.join(data_dir, train_dataset_name)
        if not os.path.exists(train_dataset_path):
            print(f"Warning: {train_dataset_path} not found. Skipping.")
            continue

        X_full, Y_full = load_nonlinear_dataset(
            train_dataset_path, HYPERPARAMETERS['num_samples']
        )
        n = len(X_full)
        train_idx, val_idx, test_idx = seeded_split(
            n, HYPERPARAMETERS['seed'], HYPERPARAMETERS['train_frac'], HYPERPARAMETERS['val_frac'],
        )
        n_train, n_val = len(train_idx), len(val_idx)
        X_train, X_val, X_test = X_full[train_idx], X_full[val_idx], X_full[test_idx]
        Y_train, Y_val, Y_test = Y_full[train_idx], Y_full[val_idx], Y_full[test_idx]

        X_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        X_std = X_train.std(axis=(0, 2, 3), keepdims=True)
        Y_mean = Y_train.mean(axis=(0, 2, 3), keepdims=True)
        Y_std = Y_train.std(axis=(0, 2, 3), keepdims=True)

        X_train = (X_train - X_mean) / (X_std + 1e-8)
        X_val = (X_val - X_mean) / (X_std + 1e-8)
        X_test = (X_test - X_mean) / (X_std + 1e-8)
        Y_train = (Y_train - Y_mean) / (Y_std + 1e-8)
        Y_val = (Y_val - Y_mean) / (Y_std + 1e-8)
        Y_test = (Y_test - Y_mean) / (Y_std + 1e-8)

        # Free full arrays — split + normalised copies are now independent
        del X_full, Y_full
        gc.collect()

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(Y_train).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, num_workers=0,
            generator=shuffle_generator(HYPERPARAMETERS['seed']),
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(Y_val).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(Y_test).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2,
        )

        # Fit PCA basis (shared by AMOR and PCANet)
        print("Computing PCA...")
        rb = PCAReducerMultiChannel(
            bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False,
            random_state=HYPERPARAMETERS['seed'],
        ).fit(X_train, Y_train)
        k = HYPERPARAMETERS['k']
        k_in = min(k, rb.pca_in.n_components_)
        k_out = min(k, rb.pca_out.n_components_)
        print(f"PCA modes: k_in={k_in}, k_out={k_out}")

        # Model factories (called fresh for each training run)
        def make_amor():
            return AMOR(
                width=HYPERPARAMETERS['width'],
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=HYPERPARAMETERS['layers'],
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=False,
                use_coordinates=False,
                add_constant_mode=False,
                use_factorized_weights="auto",
            )

        def make_fno():
            return FNO2d(
                in_channels=2,
                out_channels=1,
                width=HYPERPARAMETERS['width'],
                modes=HYPERPARAMETERS['fno_modes'],
                n_layers=HYPERPARAMETERS['layers'],
            )

        def make_pcanet():
            return PCANet(
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                hidden=1024,
                depth=5,
                dropout=0.0,
                init_scale=0.1,
            )

        model_factories = {
            'AMOR': make_amor,
            'FNO': make_fno,
            'PCANet': make_pcanet,
        }

        if args.model is not None:
            model_factories = {args.model: model_factories[args.model]}

        for model_name, model_factory in model_factories.items():
            seed_tag = f"_s{HYPERPARAMETERS['seed']}" if args.results_dir else ""
            result_path = os.path.join(
                results_dir, f"{train_dataset_type}_{model_name}{seed_tag}_results.json"
            )

            # Resume: skip if already done
            if os.path.exists(result_path) and not os.environ.get("AMOR_PROTOCOL_OUT"):
                print(f"\n--- {model_name}: result exists, skipping ({result_path}) ---")
                continue

            print(f"\n--- Training {model_name} ---")
            apply_weight_seed(HYPERPARAMETERS['seed'])

            model = model_factory()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")
            from AMOR.training.torch_utils import run_external_protocol
            if run_external_protocol(
                model, train_loader, device,
                lambda m, x, y: l1_loss(m(x), y),
                {"entry": os.environ.get("AMOR_PROTOCOL_ENTRY", f"nl_{train_dataset_type}_{model_name}"),
                 "dataset": train_dataset_type, "model": model_name,
                 "params": n_params, "precision": "fp32",
                 "block_checkpoint": os.environ.get("AMOR_BLOCK_CHECKPOINT", "1")},
            ):
                continue

            try:
                model = torch.compile(model)
            except Exception:
                pass  # fall back to eager mode

            t0 = time.time()
            best_val_loss, best_epoch = train_model(
                model, train_loader, val_loader, device, HYPERPARAMETERS
            )
            elapsed = time.time() - t0
            print(f"  Training time: {elapsed:.1f}s, best epoch: {best_epoch+1}")

            # Save model checkpoint (best weights)
            checkpoint_path = os.path.join(
                results_dir, f"{train_dataset_type}_{model_name}_model.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

            # Evaluate on train / val / test sets
            train_l1, train_l2 = evaluate_model(
                model, train_loader, device, Y_mean, Y_std
            )
            val_l1, val_l2 = evaluate_model(
                model, val_loader, device, Y_mean, Y_std
            )
            test_l1, test_l2 = evaluate_model(
                model, test_loader, device, Y_mean, Y_std
            )
            print(f"  Train: Rel L1={train_l1:.4f}%  Rel L2={train_l2:.4f}%")
            print(f"  Val:   Rel L1={val_l1:.4f}%  Rel L2={val_l2:.4f}%")
            print(f"  Test:  Rel L1={test_l1:.4f}%  Rel L2={test_l2:.4f}%")

            main_result = {
                'train_dataset': train_dataset_type,
                'model': model_name,
                'train_rel_l1': train_l1,
                'train_rel_l2': train_l2,
                'val_rel_l1': val_l1,
                'val_rel_l2': val_l2,
                'test_rel_l1': test_l1,
                'test_rel_l2': test_l2,
                'best_val_loss': float(best_val_loss),
                'best_epoch': best_epoch + 1,
                'training_time_s': elapsed,
                'num_params': n_params,
                'seed': int(HYPERPARAMETERS['seed']),
                'checkpoint': checkpoint_path,
                'layers': int(HYPERPARAMETERS['layers']),
                'width': int(HYPERPARAMETERS['width']),
                'k': int(HYPERPARAMETERS['k']),
                'epochs': int(HYPERPARAMETERS['epochs']),
                'data_path': train_dataset_path,
                'split_rule': SPLIT_RULE,
                'n_train': int(n_train),
                'n_val': int(n_val),
                'n_test': int(n - n_train - n_val),
            }

            # Cross-dataset evaluation (model already has best weights)
            cross_results = []
            cross_iter = [] if args.skip_cross else list(DATASETS.items())
            for test_ds_name, test_ds_type in cross_iter:
                if test_ds_name == train_dataset_name:
                    continue
                test_ds_path = os.path.join(data_dir, test_ds_name)
                if not os.path.exists(test_ds_path):
                    continue

                X_cross, Y_cross = load_nonlinear_dataset(
                    test_ds_path, HYPERPARAMETERS['num_samples']
                )
                n_test_cross = int(HYPERPARAMETERS['test_frac'] * len(X_cross))
                X_cross = X_cross[:n_test_cross]
                Y_cross = Y_cross[:n_test_cross]

                X_cross = (X_cross - X_mean) / (X_std + 1e-8)
                Y_cross = (Y_cross - Y_mean) / (Y_std + 1e-8)

                cross_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_cross).float(),
                        torch.from_numpy(Y_cross).float(),
                    ),
                    batch_size=HYPERPARAMETERS['batch_size'], shuffle=False, num_workers=2,
                )
                cross_l1, cross_l2 = evaluate_model(
                    model, cross_loader, device, Y_mean, Y_std
                )
                del X_cross, Y_cross, cross_loader
                gc.collect()

                cross_results.append({
                    'train_dataset': train_dataset_type,
                    'test_dataset': test_ds_type,
                    'model': model_name,
                    'test_rel_l1': cross_l1,
                    'test_rel_l2': cross_l2,
                })
                print(f"  Cross ({test_ds_type}): "
                      f"Rel L1={cross_l1:.4f}%  Rel L2={cross_l2:.4f}%")

            # Save all results for this (train_dataset, model) run immediately
            with open(result_path, 'w') as f:
                json.dump({'main': main_result, 'cross': cross_results}, f, indent=2)
            print(f"  Saved: {result_path}")

            # Free memory before next model
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Rebuild aggregate CSVs from all saved per-run JSONs
    print(f"\n{'='*70}")
    print("Rebuilding aggregate CSVs...")
    rebuild_csvs(results_dir)
    print("Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
