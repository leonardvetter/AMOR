#!/usr/bin/env python3
"""
PCA Sweep Analysis for MRBNO Lift Simple

Sweeps over PCA decomposition parameters (epsilon and number of modes) for all datasets.
Creates two plots per dataset:
1. Train/val/test loss vs epsilon (identical for input/output basis)
2. Test loss as colour in 2D plot with num input modes vs num output modes as axes

Results are saved after every individual training run so the script can be safely
interrupted and resumed. On restart, already-computed configurations are skipped.

Training strategy matches the leonard branch:
- Optimizer: AdamW (lr=2e-3, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8)
- Scheduler: CosineAnnealingLR (T_max=300, eta_min=0.0)
- MRBNO: use_batch_norm=False, use_coordinates=False, init_scale=0.1
- Loss: absolute L1 (training)
"""

import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel

# Standard hyperparameters (from leonard notebooks)
HYPERPARAMETERS = {
    'layers': 4,
    'width': 24,
    'lr': 2e-3,
    'weight_decay': 1e-4,
    'max_grad_norm': 3.0,
    'epochs': 300,
    'early_stopping_patience': 50,
    'early_stopping_min_delta': 0.0,
    'batch_size': 16,
    'seed': 42,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'test_frac': 0.1,
    'num_samples': 10000,
}

# Training schedule: CosineAnnealingLR scheduler
TRAINING_SCHEDULE = {
    'scheduler': 'CosineAnnealingLR',
    'T_max': 300,
    'eta_min': 0.0,
}

# Epsilon sweep values (fraction of variance to KEEP; identical for input/output)
EPSILON_VALUES = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]

# Mode sweep values (input and output modes swept independently)
MODE_VALUES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 144]

# Datasets to process
DATASETS = {
    'nonlinear_diffusion_dataset.npz': 'nonlinear',
    'nonlinear_diffusion_dataset_smooth.npz': 'nonlinear',
    'nonlinear_diffusion_dataset_rough.npz': 'nonlinear',
    'helmholtz_dataset.npz': 'helmholtz',
    'helmholtz_dataset_smooth.npz': 'helmholtz',
    'helmholtz_dataset_rough.npz': 'helmholtz',
    '2D_NS_Re40.npy': 'kolmogorov',
    '2D_NS_Re500.npy': 'kolmogorov',
    '2D_NS_Re5000.npy': 'kolmogorov',
}


def l1_loss(pred, target):
    """Absolute L1 loss for training."""
    return torch.mean(torch.abs(pred - target))


def load_dataset(dataset_path, dataset_type, num_samples=10000):
    """Load dataset based on type. Returns (X, Y, is_complex)."""
    if dataset_type == 'kolmogorov':
        data = np.load(dataset_path)
        u_trajectories = data['u']
        num_trajectories, num_timesteps, H, W = u_trajectories.shape
        X_list, Y_list = [], []
        for traj_idx in range(num_trajectories):
            for t in range(num_timesteps - 1):
                if len(X_list) >= num_samples:
                    break
                X_list.append(u_trajectories[traj_idx, t])
                Y_list.append(u_trajectories[traj_idx, t + 1])
            if len(X_list) >= num_samples:
                break
        X = np.array(X_list[:num_samples])[:, np.newaxis, :, :]
        Y = np.array(Y_list[:num_samples])[:, np.newaxis, :, :]
        return X, Y, False

    elif dataset_type == 'nonlinear':
        data = np.load(dataset_path)
        f = data['f'][:num_samples]
        m = data['m'][:num_samples]
        u = data['u'][:num_samples]
        X = np.stack([f, m], axis=1)
        Y = u[:, np.newaxis, :, :]
        return X, Y, False

    elif dataset_type == 'helmholtz':
        data = np.load(dataset_path, allow_pickle=True)
        m = data['m'][:num_samples]
        u_real = data['u_real'][:num_samples]
        u_imag = data['u_imag'][:num_samples]
        X = m[:, np.newaxis, :, :]
        Y = np.stack([u_real, u_imag], axis=1)
        return X, Y, True

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def train_model(model, train_loader, val_loader, device, hyperparameters):
    """
    Train model with AdamW + CosineAnnealingLR + early stopping.
    Saves best model weights and restores them after early stopping.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters['lr'],
        weight_decay=hyperparameters['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_SCHEDULE['T_max'],
        eta_min=TRAINING_SCHEDULE['eta_min'],
    )

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), hyperparameters['max_grad_norm']
            )
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += l1_loss(pred, yb).item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss - hyperparameters['early_stopping_min_delta']:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters['early_stopping_patience']:
                break

    # Restore best weights
    model.load_state_dict(best_model_state)
    return train_loss


def evaluate_model(model, loader, device):
    """Evaluate model and return mean absolute L1 loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += l1_loss(pred, yb).item() * xb.size(0)
    return total_loss / len(loader.dataset)


def save_results(results, results_file):
    """Atomically save results dict to JSON."""
    tmp_path = results_file + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, results_file)


def create_plots(results, results_dir, dataset_name):
    """Create epsilon sweep plot and mode sweep heatmap."""
    dataset_name_clean = dataset_name.replace('.', '_')

    # Plot 1: Epsilon sweep
    eps_data = results.get('epsilon_sweep', {})
    if eps_data:
        epsilons = sorted([float(e) for e in eps_data.keys()])
        train_losses = [eps_data[str(e)]['train_loss'] for e in epsilons]
        val_losses = [eps_data[str(e)]['val_loss'] for e in epsilons]
        test_losses = [eps_data[str(e)]['test_loss'] for e in epsilons]

        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, train_losses, 'o-', label='Train', linewidth=2)
        plt.plot(epsilons, val_losses, 's-', label='Val', linewidth=2)
        plt.plot(epsilons, test_losses, '^-', label='Test', linewidth=2)
        plt.xlabel('Epsilon (explained variance kept)', fontsize=12)
        plt.ylabel('Absolute L1 Loss', fontsize=12)
        plt.title(f'PCA Epsilon Sweep: {dataset_name}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = os.path.join(results_dir, f"{dataset_name_clean}_epsilon_sweep.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Epsilon sweep plot saved: {plot_file}")

    # Plot 2: Mode sweep heatmap
    mode_data = results.get('mode_sweep', {})
    if mode_data:
        # keys are "k_in_raw_k_out_raw"
        k_in_vals = sorted(set(int(k.split('_')[0]) for k in mode_data))
        k_out_vals = sorted(set(int(k.split('_')[1]) for k in mode_data))

        test_matrix = np.full((len(k_in_vals), len(k_out_vals)), np.nan)
        for i, ki in enumerate(k_in_vals):
            for j, ko in enumerate(k_out_vals):
                key = f"{ki}_{ko}"
                if key in mode_data:
                    test_matrix[i, j] = mode_data[key]['test_loss']

        plt.figure(figsize=(10, 8))
        im = plt.imshow(test_matrix, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, label='Test Loss (Absolute L1)')
        plt.xlabel('Number of Output Modes (k_out)', fontsize=12)
        plt.ylabel('Number of Input Modes (k_in)', fontsize=12)
        plt.title(f'PCA Mode Sweep: {dataset_name}', fontsize=14)
        plt.xticks(range(len(k_out_vals)), k_out_vals)
        plt.yticks(range(len(k_in_vals)), k_in_vals)
        plt.tight_layout()
        plot_file = os.path.join(results_dir, f"{dataset_name_clean}_mode_sweep.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Mode sweep plot saved: {plot_file}")


def process_dataset(dataset_name, dataset_type, results_dir, data_dir):
    """
    Process a single dataset with epsilon and mode sweeps.
    Results are saved incrementally and the function resumes from where it left off.
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    results_file = os.path.join(
        results_dir, f"{dataset_name.replace('.', '_')}_pca_sweep.json"
    )

    # Load existing results for resume
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        n_eps_done = len(results.get('epsilon_sweep', {}))
        n_mode_done = len(results.get('mode_sweep', {}))
        print(f"Resuming: {n_eps_done} epsilon runs and {n_mode_done} mode runs already done.")
    else:
        results = {
            'dataset': dataset_name,
            'dataset_type': dataset_type,
            'epsilon_sweep': {},
            'mode_sweep': {},
        }

    # Load and split dataset
    X, Y, is_complex = load_dataset(dataset_path, dataset_type, HYPERPARAMETERS['num_samples'])

    n = len(X)
    n_train = int(HYPERPARAMETERS['train_frac'] * n)
    n_val = int(HYPERPARAMETERS['val_frac'] * n)

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    Y_train = Y[:n_train]
    Y_val = Y[n_train:n_train + n_val]
    Y_test = Y[n_train + n_val:]

    # Normalize
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

    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute full PCA once (shared across all configurations)
    print("Computing full PCA...")
    rb_full = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train, Y_train)

    # Data loaders (created once, reused)
    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=HYPERPARAMETERS['batch_size'], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=HYPERPARAMETERS['batch_size'], shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, Y_test_t),
        batch_size=HYPERPARAMETERS['batch_size'], shuffle=False,
    )

    # ------------------------------------------------------------------ #
    # Epsilon sweep                                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Epsilon Sweep ---")
    for epsilon in EPSILON_VALUES:
        eps_key = str(epsilon)
        if eps_key in results['epsilon_sweep']:
            print(f"  Epsilon {epsilon}: already done, skipping")
            continue

        print(f"  Epsilon: {epsilon}")

        # k_in / k_out from explained-variance cutoff
        k_in, _ = rb_full.k_from_cutoff(j_in=epsilon)
        _, k_out = rb_full.k_from_cutoff(j_out=epsilon)
        k_in = min(k_in, rb_full.pca_in.n_components_)
        k_out = min(k_out, rb_full.pca_out.n_components_)

        model = MRBNO(
            width=HYPERPARAMETERS['width'],
            rb=rb_full,
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

        torch.manual_seed(HYPERPARAMETERS['seed'])
        train_loss = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)
        val_loss = evaluate_model(model, val_loader, device)
        test_loss = evaluate_model(model, test_loader, device)

        results['epsilon_sweep'][eps_key] = {
            'k_in': int(k_in),
            'k_out': int(k_out),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'test_loss': float(test_loss),
        }
        print(f"    k_in={k_in}, k_out={k_out}, "
              f"train={train_loss:.6f}, val={val_loss:.6f}, test={test_loss:.6f}")

        # Save after each epsilon run
        save_results(results, results_file)

        # Free GPU memory
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Mode sweep (k_in × k_out grid)                                      #
    # ------------------------------------------------------------------ #
    print("\n--- Mode Sweep ---")
    total_mode_runs = len(MODE_VALUES) ** 2
    done_count = 0
    for k_in_raw in MODE_VALUES:
        for k_out_raw in MODE_VALUES:
            mode_key = f"{k_in_raw}_{k_out_raw}"
            if mode_key in results['mode_sweep']:
                done_count += 1
                continue

            k_in = min(k_in_raw, rb_full.pca_in.n_components_)
            k_out = min(k_out_raw, rb_full.pca_out.n_components_)

            done_count += 1
            print(f"  [{done_count}/{total_mode_runs}] "
                  f"k_in={k_in_raw} -> {k_in}, k_out={k_out_raw} -> {k_out}")

            model = MRBNO(
                width=HYPERPARAMETERS['width'],
                rb=rb_full,
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

            torch.manual_seed(HYPERPARAMETERS['seed'])
            train_loss = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)
            test_loss = evaluate_model(model, test_loader, device)

            results['mode_sweep'][mode_key] = {
                'k_in_requested': k_in_raw,
                'k_out_requested': k_out_raw,
                'k_in_used': int(k_in),
                'k_out_used': int(k_out),
                'train_loss': float(train_loss),
                'test_loss': float(test_loss),
            }
            print(f"    train={train_loss:.6f}, test={test_loss:.6f}")

            # Save after each mode combination
            save_results(results, results_file)

            # Free GPU memory
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final plots
    print(f"\nCreating plots for {dataset_name}...")
    create_plots(results, results_dir, dataset_name)
    print(f"Completed: {dataset_name}")


def main():
    results_dir = '/mnt/c/Users/dvod/MRBNO/data/pca_sweep_analysis'
    data_dir = '/mnt/c/Users/dvod/MRBNO/data'
    os.makedirs(results_dir, exist_ok=True)

    print("PCA Sweep Analysis for MRBNO Lift Simple")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print(f"Training schedule: {TRAINING_SCHEDULE}")
    print(f"Epsilon values ({len(EPSILON_VALUES)}): {EPSILON_VALUES}")
    print(f"Mode values ({len(MODE_VALUES)}): {MODE_VALUES}")
    print(f"Total mode combinations per dataset: {len(MODE_VALUES)**2}")
    print("=" * 60)

    for dataset_name, dataset_type in DATASETS.items():
        try:
            process_dataset(dataset_name, dataset_type, results_dir, data_dir)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("All datasets processed.")
    print(f"Results in: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
