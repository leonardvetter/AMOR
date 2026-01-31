#!/usr/bin/env python3
"""
PCA Sweep Analysis for MRBNO Lift Simple

Sweeps over PCA decomposition parameters (epsilon and number of modes) for all datasets.
Creates two plots per dataset:
1. Train/val/test loss vs epsilon (identical for input/output basis)
2. Test loss as color in 2D plot with num input modes vs num output modes as axes
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
import json
from pathlib import Path

# Add models directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel

# Standard hyperparameters (from notebooks)
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

# Epsilon sweep values (identical for input/output)
EPSILON_VALUES = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]

# Mode sweep values (input and output modes separately)
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

def relative_l1_loss(pred, target):
    """Compute relative L1 loss."""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

def load_dataset(dataset_path, dataset_type, num_samples=1000):
    """Load dataset based on type."""
    if dataset_type == 'kolmogorov':
        data = np.load(dataset_path)
        u_trajectories = data['u']
        num_trajectories, num_timesteps, H, W = u_trajectories.shape
        
        X_list = []
        Y_list = []
        for traj_idx in range(min(num_trajectories, num_samples // num_timesteps + 1)):
            for t in range(num_timesteps - 1):
                if len(X_list) >= num_samples:
                    break
                X_list.append(u_trajectories[traj_idx, t])
                Y_list.append(u_trajectories[traj_idx, t + 1])
            if len(X_list) >= num_samples:
                break
        
        X = np.array(X_list[:num_samples])[:, np.newaxis, :, :]
        Y = np.array(Y_list[:num_samples])[:, np.newaxis, :, :]
        is_complex = False
    elif dataset_type == 'nonlinear':
        data = np.load(dataset_path)
        f = data['f'][:num_samples]
        m = data['m'][:num_samples]
        u = data['u'][:num_samples]
        X = np.stack([f, m], axis=1)
        Y = u[:, np.newaxis, :, :]
        is_complex = False
    elif dataset_type == 'helmholtz':
        data = np.load(dataset_path, allow_pickle=True)
        m = data['m'][:num_samples]
        u_real = data['u_real'][:num_samples]
        u_imag = data['u_imag'][:num_samples]
        X = m[:, np.newaxis, :, :]
        Y = np.stack([u_real, u_imag], axis=1)
        is_complex = True
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, Y, is_complex

def train_model(model, train_loader, val_loader, device, hyperparameters, training_schedule):
    """Train model with standard training schedule."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters['lr'],
        weight_decay=hyperparameters['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_schedule['T_max'],
        eta_min=training_schedule['eta_min']
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hyperparameters['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        
        if val_loss < best_val_loss - hyperparameters['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters['early_stopping_patience']:
                break
    
    return train_loss

def evaluate_model(model, loader, device):
    """Evaluate model and return loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            total_loss += loss.item()
    return total_loss / len(loader)

def process_dataset(dataset_name, dataset_type, results_dir, scratch_dir):
    """Process a single dataset with epsilon and mode sweeps."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_path = os.path.join(scratch_dir, 'data', dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    X, Y, is_complex = load_dataset(dataset_path, dataset_type, HYPERPARAMETERS['num_samples'])
    
    # Split data
    n = len(X)
    n_train = int(HYPERPARAMETERS['train_frac'] * n)
    n_val = int(HYPERPARAMETERS['val_frac'] * n)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train+n_val], Y[n_train+n_val:]
    
    # Normalize
    X_mean, X_std = X_train.mean(axis=(0, 2, 3), keepdims=True), X_train.std(axis=(0, 2, 3), keepdims=True)
    Y_mean, Y_std = Y_train.mean(axis=(0, 2, 3), keepdims=True), Y_train.std(axis=(0, 2, 3), keepdims=True)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_val = (X_val - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    Y_train = (Y_train - Y_mean) / (Y_std + 1e-8)
    Y_val = (Y_val - Y_mean) / (Y_std + 1e-8)
    Y_test = (Y_test - Y_mean) / (Y_std + 1e-8)
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Compute full PCA (for mode sweep)
    print("Computing full PCA...")
    rb_full = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train, Y_train)
    
    in_channels = X.shape[1]
    out_channels = Y.shape[1]
    
    # Determine which PCA to use for spectral conv
    if in_channels == 1 and out_channels > 1:
        pca_components_base = rb_full.pca_in.components_
        pca_mean_base = rb_full.pca_in.mean_
    else:
        pca_components_base = rb_full.pca_out.components_
        pca_mean_base = rb_full.pca_out.mean_
    
    results = {
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'epsilon_sweep': {},
        'mode_sweep': {},
    }
    
    # Epsilon sweep
    print("\n--- Epsilon Sweep ---")
    for epsilon in EPSILON_VALUES:
        print(f"Epsilon: {epsilon}")
        
        # Compute k_in and k_out from epsilon
        k_in = rb_full.k_from_cutoff(j_in=epsilon)[0]
        k_out = rb_full.k_from_cutoff(j_out=epsilon)[1]
        
        # Use same k for both if using same PCA basis
        if in_channels == 1 and out_channels > 1:
            k = k_in
        else:
            k = k_out
        
        k = min(k, pca_components_base.shape[0])
        
        # Create model
        model = MRBNO(width=width, rb=rb, k_in=k_in, k_out=k_out, layers=HYPERPARAMETERS['layers'], dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=True, use_coordinates=True, add_constant_mode=False, use_factorized_weights=False)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                                 batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t),
                               batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t),
                                batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        
        # Train
        train_loss = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS, TRAINING_SCHEDULE)
        
        # Evaluate
        val_loss = evaluate_model(model, val_loader, device)
        test_loss = evaluate_model(model, test_loader, device)
        
        results['epsilon_sweep'][str(epsilon)] = {
            'k_in': int(k_in),
            'k_out': int(k_out),
            'k_used': int(k),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'test_loss': float(test_loss),
        }
        
        print(f"  k_in={k_in}, k_out={k_out}, k_used={k}, train={train_loss:.6f}, val={val_loss:.6f}, test={test_loss:.6f}")
    
    # Mode sweep
    print("\n--- Mode Sweep ---")
    mode_sweep_results = {}
    for k_in in MODE_VALUES:
        for k_out in MODE_VALUES:
            k_in = min(k_in, rb_full.pca_in.n_components_)
            k_out = min(k_out, rb_full.pca_out.n_components_)
            
            # Determine which PCA to use
            if in_channels == 1 and out_channels > 1:
                k_used = min(k_in, pca_components_base.shape[0])
                pca_components = pca_components_base[:k_used]
                pca_mean = pca_mean_base
            else:
                k_used = min(k_out, pca_components_base.shape[0])
                pca_components = pca_components_base[:k_used]
                pca_mean = pca_mean_base
            
            # Create model
            model = MRBNO(width=width, rb=rb, k_in=k_in, k_out=k_out, layers=HYPERPARAMETERS['layers'], dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=True, use_coordinates=True, add_constant_mode=False, use_factorized_weights=False)
            
            # Create data loaders
            train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                                     batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t),
                                   batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
            test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t),
                                    batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
            
            # Train
            train_loss = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS, TRAINING_SCHEDULE)
            
            # Evaluate
            test_loss = evaluate_model(model, test_loader, device)
            
            mode_sweep_results[(k_in, k_out)] = {
                'k_used': int(k_used),
                'train_loss': float(train_loss),
                'test_loss': float(test_loss),
            }
            
            print(f"  k_in={k_in}, k_out={k_out}, k_used={k_used}, train={train_loss:.6f}, test={test_loss:.6f}")
    
    results['mode_sweep'] = {f"{k_in}_{k_out}": v for (k_in, k_out), v in mode_sweep_results.items()}
    
    # Save results
    result_file = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_pca_sweep.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    # Create plots
    dataset_name_clean = dataset_name.replace('.', '_')
    
    # Plot 1: Epsilon sweep
    epsilons = sorted([float(e) for e in results['epsilon_sweep'].keys()])
    train_losses = [results['epsilon_sweep'][str(e)]['train_loss'] for e in epsilons]
    val_losses = [results['epsilon_sweep'][str(e)]['val_loss'] for e in epsilons]
    test_losses = [results['epsilon_sweep'][str(e)]['test_loss'] for e in epsilons]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, train_losses, 'o-', label='Train', linewidth=2)
    plt.plot(epsilons, val_losses, 's-', label='Val', linewidth=2)
    plt.plot(epsilons, test_losses, '^-', label='Test', linewidth=2)
    plt.xlabel('Epsilon (explained variance ratio)', fontsize=12)
    plt.ylabel('Relative L1 Loss', fontsize=12)
    plt.title(f'PCA Epsilon Sweep: {dataset_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file1 = os.path.join(results_dir, f"{dataset_name_clean}_epsilon_sweep.png")
    plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Epsilon sweep plot saved to: {plot_file1}")
    
    # Plot 2: Mode sweep (2D color plot)
    k_in_vals = sorted(set([int(k.split('_')[0]) for k in results['mode_sweep'].keys()]))
    k_out_vals = sorted(set([int(k.split('_')[1]) for k in results['mode_sweep'].keys()]))
    
    test_loss_matrix = np.zeros((len(k_in_vals), len(k_out_vals)))
    for i, k_in in enumerate(k_in_vals):
        for j, k_out in enumerate(k_out_vals):
            key = f"{k_in}_{k_out}"
            if key in results['mode_sweep']:
                test_loss_matrix[i, j] = results['mode_sweep'][key]['test_loss']
            else:
                test_loss_matrix[i, j] = np.nan
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(test_loss_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='Test Loss (Relative L1)')
    plt.xlabel('Number of Output Modes', fontsize=12)
    plt.ylabel('Number of Input Modes', fontsize=12)
    plt.title(f'PCA Mode Sweep: {dataset_name}', fontsize=14)
    plt.xticks(range(len(k_out_vals)), k_out_vals)
    plt.yticks(range(len(k_in_vals)), k_in_vals)
    plt.tight_layout()
    plot_file2 = os.path.join(results_dir, f"{dataset_name_clean}_mode_sweep.png")
    plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mode sweep plot saved to: {plot_file2}")

def main():
    scratch_dir = os.environ.get('SCRATCH', '/tmp')
    results_dir = os.path.join(scratch_dir, 'pca_sweep_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    print("PCA Sweep Analysis for MRBNO Lift Simple")
    print("="*60)
    print(f"Results directory: {results_dir}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print(f"Training schedule: {TRAINING_SCHEDULE}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Mode values: {MODE_VALUES}")
    print("="*60)
    
    for dataset_name, dataset_type in DATASETS.items():
        try:
            process_dataset(dataset_name, dataset_type, results_dir, scratch_dir)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All datasets processed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

