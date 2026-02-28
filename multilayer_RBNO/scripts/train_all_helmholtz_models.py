#!/usr/bin/env python3
"""
Train MRBNO Lift Simple, FNO, and PCANet on all 3 Helmholtz datasets.

Trains each model on each dataset, then evaluates on all 3 datasets (cross-dataset
evaluation). Results are saved incrementally per (dataset, model) run so the script
can be safely interrupted and resumed at any time.

Training strategy matches the leonard branch Helmholtz notebooks:
- Optimizer: AdamW (lr=2e-3, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8)
- Scheduler: CosineAnnealingLR (T_max=300, eta_min=0.0)
- Loss: absolute L1 (training), relative L1/L2 (evaluation)
- MRBNO: k_in=k_out=144 (fixed), use_batch_norm=False, use_coordinates=False
- PCANet: hidden=1024, depth=5
- FNO: modes=(12,12), width=24, layers=4
- Best model weights are restored after early stopping
"""

import copy
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
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from MRBNO import MRBNO
from FNO_model import FNO2d
from PCANet import PCANet
from PCAReducer import PCAReducerMultiChannel

HYPERPARAMETERS = {
    'layers': 4,
    'width': 24,
    'k': 144,           # Fixed number of PCA modes (k_in = k_out = k), matching leonard notebooks
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
    'helmholtz_dataset.npz': 'standard',
    'helmholtz_dataset_smooth.npz': 'smooth',
    'helmholtz_dataset_rough.npz': 'rough',
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def l1_loss(pred, target):
    """Absolute L1 loss used for training."""
    return torch.mean(torch.abs(pred - target))


def load_helmholtz_dataset(dataset_path, num_samples=10000):
    data = np.load(dataset_path, allow_pickle=True)
    m = data['m'][:num_samples]
    u_real = data['u_real'][:num_samples]
    u_imag = data['u_imag'][:num_samples]
    X = m[:, np.newaxis, :, :]                         # (N, 1, H, W)
    Y = np.stack([u_real, u_imag], axis=1)             # (N, 2, H, W)
    return X, Y


def train_model(model, train_loader, val_loader, device, hyperparams):
    """
    Train model with AdamW + CosineAnnealingLR + early stopping.
    Saves best model weights during training and restores them at the end.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=0.0)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(hyperparams['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += l1_loss(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss - hyperparams['early_stopping_min_delta']:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hyperparams['early_stopping_patience']:
            print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{hyperparams['epochs']}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}")

    # Restore best model weights
    model.load_state_dict(best_model_state)
    return best_val_loss, best_epoch


def evaluate_model(model, loader, device, y_mean_np, y_std_np):
    """
    Evaluate model on a data loader.
    Returns (rel_l1_pct, rel_l2_pct) computed on the de-normalised magnitude field.
    """
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
                pred_mag = torch.sqrt(
                    pred_denorm[i, 0] ** 2 + pred_denorm[i, 1] ** 2 + 1e-10
                ).cpu().numpy()
                true_mag = torch.sqrt(
                    yb_denorm[i, 0] ** 2 + yb_denorm[i, 1] ** 2 + 1e-10
                ).cpu().numpy()

                l1_err = np.abs(pred_mag - true_mag).sum()
                l1_norm = np.abs(true_mag).sum()
                rel_l1_errors.append(l1_err / (l1_norm + 1e-10))

                l2_err = np.sqrt(np.sum((pred_mag - true_mag) ** 2))
                l2_norm = np.sqrt(np.sum(true_mag ** 2))
                rel_l2_errors.append(l2_err / (l2_norm + 1e-10))

    return np.mean(rel_l1_errors) * 100, np.mean(rel_l2_errors) * 100


def rebuild_csvs(results_dir):
    """Reconstruct the main and cross-dataset CSVs from all saved per-run JSON files."""
    results_main = []
    results_cross = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('_results.json'):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        if 'main' in data:
            results_main.append(data['main'])
        if 'cross' in data:
            results_cross.extend(data['cross'])

    main_path = os.path.join(results_dir, 'helmholtz_main_results.csv')
    cross_path = os.path.join(results_dir, 'helmholtz_cross_dataset_results.csv')

    if results_main:
        pd.DataFrame(results_main).to_csv(main_path, index=False)
        print(f"  Main results CSV: {main_path}")

    if results_cross:
        pd.DataFrame(results_cross).to_csv(cross_path, index=False)
        print(f"  Cross-dataset results CSV: {cross_path}")


def main():
    data_dir = '/mnt/c/Users/dvod/MRBNO/data'
    results_dir = '/mnt/c/Users/dvod/MRBNO/data/helmholtz_model_comparison'
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")

    set_seed(HYPERPARAMETERS['seed'])

    for train_dataset_name, train_dataset_type in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Training on: {train_dataset_name} ({train_dataset_type})")
        print(f"{'='*70}")

        train_dataset_path = os.path.join(data_dir, train_dataset_name)
        if not os.path.exists(train_dataset_path):
            print(f"Warning: {train_dataset_path} not found. Skipping.")
            continue

        X_full, Y_full = load_helmholtz_dataset(
            train_dataset_path, HYPERPARAMETERS['num_samples']
        )
        n = len(X_full)
        n_train = int(HYPERPARAMETERS['train_frac'] * n)
        n_val = int(HYPERPARAMETERS['val_frac'] * n)

        X_train = X_full[:n_train]
        X_val = X_full[n_train:n_train + n_val]
        X_test = X_full[n_train + n_val:]
        Y_train = Y_full[:n_train]
        Y_val = Y_full[n_train:n_train + n_val]
        Y_test = Y_full[n_train + n_val:]

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

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(Y_train).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(Y_val).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(Y_test).float(),
            ),
            batch_size=HYPERPARAMETERS['batch_size'], shuffle=False,
        )

        # Fit PCA basis (shared by MRBNO and PCANet)
        print("Computing PCA...")
        rb = PCAReducerMultiChannel(
            bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False
        ).fit(X_train, Y_train)
        k = HYPERPARAMETERS['k']
        k_in = min(k, rb.pca_in.n_components_)
        k_out = min(k, rb.pca_out.n_components_)
        print(f"PCA modes: k_in={k_in}, k_out={k_out}")

        # Model factories (called fresh for each training run)
        def make_mrbno():
            return MRBNO(
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
                use_factorized_weights=False,
            )

        def make_fno():
            return FNO2d(
                modes1=HYPERPARAMETERS['fno_modes'][0],
                modes2=HYPERPARAMETERS['fno_modes'][1],
                width=HYPERPARAMETERS['width'],
                in_channels=1,
                out_channels=2,
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
            'MRBNO': make_mrbno,
            'FNO': make_fno,
            'PCANet': make_pcanet,
        }

        for model_name, model_factory in model_factories.items():
            result_path = os.path.join(
                results_dir, f"{train_dataset_type}_{model_name}_results.json"
            )

            # Resume: skip if already done
            if os.path.exists(result_path):
                print(f"\n--- {model_name}: result exists, skipping ({result_path}) ---")
                continue

            print(f"\n--- Training {model_name} ---")
            set_seed(HYPERPARAMETERS['seed'])

            model = model_factory()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

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
            }

            # Cross-dataset evaluation (model already has best weights)
            cross_results = []
            for test_ds_name, test_ds_type in DATASETS.items():
                if test_ds_name == train_dataset_name:
                    continue
                test_ds_path = os.path.join(data_dir, test_ds_name)
                if not os.path.exists(test_ds_path):
                    continue

                X_cross, Y_cross = load_helmholtz_dataset(
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
                    batch_size=HYPERPARAMETERS['batch_size'], shuffle=False,
                )
                cross_l1, cross_l2 = evaluate_model(
                    model, cross_loader, device, Y_mean, Y_std
                )
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

            # Free GPU memory before next model
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Rebuild aggregate CSVs from all saved per-run JSONs
    print(f"\n{'='*70}")
    print("Rebuilding aggregate CSVs...")
    rebuild_csvs(results_dir)
    print("Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
