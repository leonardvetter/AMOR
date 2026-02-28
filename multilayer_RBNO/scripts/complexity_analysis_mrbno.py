#!/usr/bin/env python3
"""
Complexity Analysis for MRBNO Lift Simple Model

This script creates plots similar to Figure 1 in the paper (arxiv.org/pdf/2304.13221),
showing the relationship between modes × width and model performance.

For each dataset, two versions are run:
Version 1: Sweep over width and k (modes) with fixed width × k
  - k_in = k_out = modes (same number of input and output modes)
  - 4 curves per dataset (fixed products: 288, 576, 1152, 2304)
  - 6-7 configurations per curve

Version 2: Sweep over width and epsilon for fixed width × (1/epsilon)
  - k_in and k_out computed from epsilon using rb.k_from_cutoff()
  - 4 curves per dataset (fixed products: 24000, 48000, 96000, 192000)
  - 6 configurations per curve

Results saved after each dataset. Can restart from where it left off.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
import json
import pickle
from pathlib import Path

# Add models directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel

# Fixed hyperparameters (from notebooks)
HYPERPARAMETERS = {
    'layers': 4,
    'lr': 2e-3,
    'weight_decay': 1e-4,
    'max_grad_norm': 3.0,
    'epochs': 300,
    'early_stopping_patience': 50,  # 30 for Helmholtz, but using 50 for consistency
    'early_stopping_min_delta': 0.0,
    'batch_size': 32,
    'seed': 42,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'test_frac': 0.1,
    'num_samples': 10000,
}

# Fixed modes × width products (4 curves per dataset) - Version 1
FIXED_PRODUCTS = [288, 576, 1152, 2304]

# Configurations for each fixed product (modes, width) pairs - Version 1
# Format: {product: [(modes, width), ...]}
# Note: For version 1, k_in = k_out = modes
CONFIGURATIONS = {
    288: [(8, 36), (12, 24), (16, 18), (18, 16), (24, 12), (36, 8)],  # 6 configs
    576: [(12, 48), (16, 36), (18, 32), (24, 24), (32, 18), (36, 16), (48, 12)],  # 7 configs
    1152: [(16, 72), (18, 64), (24, 48), (32, 36), (36, 32), (48, 24), (64, 18)],  # 7 configs
    2304: [(24, 96), (32, 72), (36, 64), (48, 48), (64, 36), (72, 32), (96, 24)],  # 7 configs
}

# Fixed width × (1/epsilon) products - Version 2
# Format: {product: [(width, epsilon), ...]}
# For each fixed product, we sweep over width and epsilon such that width × (1/epsilon) is constant
FIXED_WIDTH_OVER_EPSILON = [24000, 48000, 96000, 192000]

# Configurations for each fixed width/epsilon product - Version 2
# Format: {product: [(width, epsilon), ...]}
# epsilon values are chosen to give reasonable k_in/k_out values (typically 50-200)
CONFIGURATIONS_V2 = {
    24000: [(12, 0.0005), (16, 0.00067), (24, 0.001), (32, 0.00133), (36, 0.0015), (48, 0.002)],  # 6 configs
    48000: [(16, 0.00033), (24, 0.0005), (32, 0.00067), (36, 0.00075), (48, 0.001), (64, 0.00133)],  # 6 configs
    96000: [(24, 0.00025), (32, 0.00033), (36, 0.000375), (48, 0.0005), (64, 0.00067), (96, 0.001)],  # 6 configs
    192000: [(32, 0.000167), (48, 0.00025), (64, 0.00033), (72, 0.000375), (96, 0.0005), (128, 0.00067)],  # 6 configs
}

# Training schedule: CosineAnnealingLR scheduler (same as notebooks)
TRAINING_SCHEDULE = {
    'scheduler': 'CosineAnnealingLR',
    'T_max': HYPERPARAMETERS['epochs'],
    'eta_min': 0.0,
}

# Datasets to process
DATASETS = [
    # Nonlinear diffusion
    ('nonlinear_diffusion_dataset.npz', 'nonlinear'),
    ('nonlinear_diffusion_dataset_smooth.npz', 'nonlinear'),
    ('nonlinear_diffusion_dataset_rough.npz', 'nonlinear'),
    # Helmholtz
    ('helmholtz_dataset.npz', 'helmholtz'),
    ('helmholtz_dataset_smooth.npz', 'helmholtz'),
    ('helmholtz_dataset_rough.npz', 'helmholtz'),
    # Kolmogorov
    ('2D_NS_Re40.npy', 'kolmogorov'),
    ('2D_NS_Re500.npy', 'kolmogorov'),
    ('2D_NS_Re5000.npy', 'kolmogorov'),
]

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def l1_loss(pred, target):
    """Absolute L1 loss for training."""
    return torch.mean(torch.abs(pred - target))

def relative_l1_loss(pred, target):
    """Compute relative L1 loss: ||pred - target||_1 / ||target||_1 (for eval/logging)."""
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    # Handle complex output (2 channels: real, imag)
    if pred.shape[1] == 2 and target.shape[1] == 2:
        pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-10)
        target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-10)
        error = torch.abs(pred_mag - target_mag)
        return torch.mean(error / (target_mag + 1e-10))
    else:
        error = torch.abs(pred - target)
        return torch.mean(error / (torch.abs(target) + 1e-10))

def train_model(model, train_loader, val_loader, device, hyperparams):
    """Train model with cosine annealing LR schedule and early stopping."""
    import time
    start_time = time.time()
    
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=hyperparams['epochs'], 
        eta_min=TRAINING_SCHEDULE['eta_min']
    )
    criterion = l1_loss
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    has_val = len(val_loader.dataset) > 0
    
    for epoch in range(hyperparams['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        if has_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping logic
        improved = False
        if has_val:
            if val_loss < best_val_loss - hyperparams['early_stopping_min_delta']:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
        else:
            if train_loss < best_val_loss - hyperparams['early_stopping_min_delta']:
                best_val_loss = train_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

        scheduler.step()

        # Early stopping
        if patience_counter >= hyperparams['early_stopping_patience']:
            break

    # Restore best model weights
    model.load_state_dict(best_model_state)

    training_time = time.time() - start_time

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'training_time': training_time,
    }

def compute_rel_errors(model, loader, device, y_mean, y_std, is_complex=False):
    """Compute relative L1 and L2 errors."""
    model.eval()
    rel_l1_errors = []
    rel_l2_errors = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            pred_denorm = pred * y_std + y_mean
            yb_denorm = yb * y_std + y_mean
            
            for i in range(pred_denorm.size(0)):
                if is_complex:
                    pred_mag = torch.sqrt(pred_denorm[i, 0]**2 + pred_denorm[i, 1]**2 + 1e-10).cpu().numpy()
                    true_mag = torch.sqrt(yb_denorm[i, 0]**2 + yb_denorm[i, 1]**2 + 1e-10).cpu().numpy()
                else:
                    pred_mag = pred_denorm[i, 0].cpu().numpy() if pred_denorm.ndim == 4 else pred_denorm[i].cpu().numpy()
                    true_mag = yb_denorm[i, 0].cpu().numpy() if yb_denorm.ndim == 4 else yb_denorm[i].cpu().numpy()
                
                l1_err = np.abs(pred_mag - true_mag).sum()
                l1_norm = np.abs(true_mag).sum()
                rel_l1 = l1_err / (l1_norm + 1e-10)
                rel_l1_errors.append(rel_l1)
                
                l2_err = np.sqrt(np.sum((pred_mag - true_mag)**2))
                l2_norm = np.sqrt(np.sum(true_mag**2))
                rel_l2 = l2_err / (l2_norm + 1e-10)
                rel_l2_errors.append(rel_l2)
    
    return np.mean(rel_l1_errors) * 100, np.mean(rel_l2_errors) * 100

def load_dataset(dataset_path, dataset_type, num_samples=10000):
    """Load dataset and return X, Y, and metadata."""
    if dataset_type == 'kolmogorov':
        # For Kolmogorov: load .npy file and create input-output pairs
        data = np.load(dataset_path, mmap_mode='r')
        # Shape: (num_trajectories, num_timesteps, H, W)
        if data.ndim == 4:
            num_traj, num_t, H, W = data.shape
            # Create input-output pairs: u(t) -> u(t+1)
            X_list = []
            Y_list = []
            for traj_idx in range(num_traj):
                for t in range(num_t - 1):
                    X_list.append(data[traj_idx, t])
                    Y_list.append(data[traj_idx, t + 1])
                    if len(X_list) >= num_samples:
                        break
                if len(X_list) >= num_samples:
                    break
            X = np.array(X_list[:num_samples], dtype=np.float32)
            Y = np.array(Y_list[:num_samples], dtype=np.float32)
            # Add channel dimension
            X = X[:, np.newaxis, :, :]  # (N, 1, H, W)
            Y = Y[:, np.newaxis, :, :]  # (N, 1, H, W)
            is_complex = False
        else:
            raise ValueError(f"Unexpected Kolmogorov data shape: {data.shape}")
    elif dataset_type == 'nonlinear':
        # For nonlinear: load .npz file with 'f', 'm', 'u' keys
        data = np.load(dataset_path)
        f = data['f'][:num_samples]
        m = data['m'][:num_samples]
        u = data['u'][:num_samples]
        # Stack f and m as input channels
        X = np.stack([f, m], axis=1)  # (N, 2, H, W)
        Y = u[:, np.newaxis, :, :]  # (N, 1, H, W)
        is_complex = False
    elif dataset_type == 'helmholtz':
        # For Helmholtz: load .npz file with 'm', 'u_real', 'u_imag' keys
        data = np.load(dataset_path, allow_pickle=True)
        m = data['m'][:num_samples]
        u_real = data['u_real'][:num_samples]
        u_imag = data['u_imag'][:num_samples]
        # Input is m (single channel), output is u_real and u_imag (2 channels)
        X = m[:, np.newaxis, :, :]  # (N, 1, H, W)
        Y = np.stack([u_real, u_imag], axis=1)  # (N, 2, H, W)
        is_complex = True
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, Y, is_complex

def process_dataset(dataset_name, dataset_type, results_dir, data_dir):
    """Process a single dataset."""
    print(f"\n{'='*70}")
    print(f"Processing dataset: {dataset_name} ({dataset_type})")
    print(f"{'='*70}")

    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset {dataset_path} not found. Skipping.")
        return None
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    X, Y, is_complex = load_dataset(dataset_path, dataset_type, HYPERPARAMETERS['num_samples'])
    print(f"Loaded {len(X)} samples. X shape: {X.shape}, Y shape: {Y.shape}, is_complex: {is_complex}")
    
    # Split data
    n_train = int(HYPERPARAMETERS['train_frac'] * len(X))
    n_val = int(HYPERPARAMETERS['val_frac'] * len(X))
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Normalize
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)
    X_test = (X_test - x_mean) / (x_std + 1e-8)
    Y_test = (Y_test - y_mean) / (y_std + 1e-8)
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()
    
    # Create data loaders
    # Use smaller batch size for Re5000 dataset to avoid OOM
    batch_size = HYPERPARAMETERS['batch_size']
    if 'Re5000' in dataset_name:
        batch_size = 8
        print(f"Using reduced batch size {batch_size} for Re5000 dataset")
    
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), 
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), 
                             batch_size=batch_size, shuffle=False)
    
    # Compute PCA (shared by all configurations)
    print("Computing PCA...")
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train, Y_train)
    
    # Determine input/output channels
    in_channels = X.shape[1]
    out_channels = Y.shape[1]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Results storage
    results = {
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'hyperparameters': HYPERPARAMETERS,
        'training_schedule': TRAINING_SCHEDULE,
        'configurations': {},
    }
    
    # Process each fixed product
    for product in FIXED_PRODUCTS:
        print(f"\n--- Fixed Product: {product} (modes × width) ---")
        product_results = []
        
        for modes, width in CONFIGURATIONS[product]:
            config_key = f"modes_{modes}_width_{width}"
            print(f"\nTraining with modes={modes}, width={width} (product={modes*width})...")
            
            # Check if already computed
            checkpoint_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_{config_key}_checkpoint.pt")
            result_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_{config_key}_result.json")
            
            if os.path.exists(result_path):
                print(f"  Loading existing result from {result_path}...")
                with open(result_path, 'r') as f:
                    config_result = json.load(f)
                product_results.append(config_result)
                continue
            
            # Create model
            # For version 1 sweep: k_in = k_out = modes (same number of input and output modes)
            # This ensures width × k is fixed as we sweep
            k_in_used = min(modes, rb.pca_in.n_components_)
            k_out_used = min(modes, rb.pca_out.n_components_)
            
            model = MRBNO(
                width=width,
                rb=rb,
                k_in=k_in_used,
                k_out=k_out_used,
                layers=HYPERPARAMETERS['layers'],
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=False,
                use_coordinates=False,
                add_constant_mode=False,
                use_factorized_weights=False
            )

            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters: {num_params:,}")

            # Train
            set_seed(HYPERPARAMETERS['seed'])
            train_result = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)

            # Evaluate
            train_l1, train_l2 = compute_rel_errors(model, train_loader, device, y_mean, y_std, is_complex)
            val_l1, val_l2 = compute_rel_errors(model, val_loader, device, y_mean, y_std, is_complex)
            test_l1, test_l2 = compute_rel_errors(model, test_loader, device, y_mean, y_std, is_complex)

            config_result = {
                'modes': modes,
                'width': width,
                'product': modes * width,
                'num_params': num_params,
                'train_rel_l1': train_l1,
                'train_rel_l2': train_l2,
                'val_rel_l1': val_l1,
                'val_rel_l2': val_l2,
                'test_rel_l1': test_l1,
                'test_rel_l2': test_l2,
                'best_val_loss': train_result['best_val_loss'],
                'best_epoch': train_result['best_epoch'],
                'training_time': train_result['training_time'],
            }

            # Save individual config result immediately
            with open(result_path, 'w') as f:
                json.dump(config_result, f, indent=2)

            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_path)

            product_results.append(config_result)
            print(f"  Train Rel L1: {train_l1:.4f}%, Val Rel L1: {val_l1:.4f}%, Test Rel L1: {test_l1:.4f}%")

            # Free GPU memory before next config
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results['configurations'][f'product_{product}'] = product_results

        # Save aggregated results after each product (not just at end)
        results_file = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Final save (also updates plots)
    results_file = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plot
    create_plot(results, results_dir, dataset_name)
    
    print(f"\nCompleted dataset: {dataset_name}")
    return results


def process_dataset_v2(dataset_name, dataset_type, results_dir, data_dir):
    """
    Process dataset with Version 2 sweep: width and epsilon for fixed width × (1/epsilon).
    k_in and k_out are computed from epsilon using rb.k_from_cutoff().
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset {dataset_path} not found. Skipping.")
        return None
    
    # Load dataset (same as version 1)
    print(f"Loading dataset from {dataset_path}...")
    X, Y, is_complex = load_dataset(dataset_path, dataset_type, HYPERPARAMETERS['num_samples'])
    print(f"Loaded {len(X)} samples. X shape: {X.shape}, Y shape: {Y.shape}, is_complex: {is_complex}")
    
    # Split data
    n_train = int(HYPERPARAMETERS['train_frac'] * len(X))
    n_val = int(HYPERPARAMETERS['val_frac'] * len(X))
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Normalize
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)
    X_test = (X_test - x_mean) / (x_std + 1e-8)
    Y_test = (Y_test - y_mean) / (y_std + 1e-8)
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()
    
    # Create data loaders
    batch_size = HYPERPARAMETERS['batch_size']
    if 'Re5000' in dataset_name:
        batch_size = 8
        print(f"Using reduced batch size {batch_size} for Re5000 dataset")
    
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), 
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), 
                             batch_size=batch_size, shuffle=False)
    
    # Compute PCA (shared by all configurations)
    print("Computing PCA...")
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train, Y_train)
    
    # Determine input/output channels
    in_channels = X.shape[1]
    out_channels = Y.shape[1]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Results storage
    results = {
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'version': 2,
        'hyperparameters': HYPERPARAMETERS,
        'training_schedule': TRAINING_SCHEDULE,
        'configurations': {},
    }
    
    # Process each fixed width/epsilon product
    for product in FIXED_WIDTH_OVER_EPSILON:
        print(f"\n--- Fixed Product: {product} (width × 1/epsilon) ---")
        product_results = []
        
        for width, epsilon in CONFIGURATIONS_V2[product]:
            config_key = f"width_{width}_epsilon_{epsilon:.6f}"
            print(f"\nTraining with width={width}, epsilon={epsilon} (product={width/epsilon:.0f})...")
            
            # Check if already computed
            checkpoint_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_v2_{config_key}_checkpoint.pt")
            result_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_v2_{config_key}_result.json")
            
            if os.path.exists(result_path):
                print(f"  Loading existing result from {result_path}...")
                with open(result_path, 'r') as f:
                    config_result = json.load(f)
                product_results.append(config_result)
                continue
            
            # Compute k_in and k_out from epsilon
            j_in = 1 - epsilon
            j_out = 1 - epsilon
            k_in, k_out = rb.k_from_cutoff(j_in=j_in, j_out=j_out)
            k_in = min(k_in, rb.pca_in.n_components_)
            k_out = min(k_out, rb.pca_out.n_components_)
            
            print(f"  Computed k_in={k_in}, k_out={k_out} from epsilon={epsilon}")
            
            # Create model
            model = MRBNO(
                width=width,
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
                use_factorized_weights=False
            )

            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters: {num_params:,}")

            # Train
            set_seed(HYPERPARAMETERS['seed'])
            train_result = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)

            # Evaluate
            train_l1, train_l2 = compute_rel_errors(model, train_loader, device, y_mean, y_std, is_complex)
            val_l1, val_l2 = compute_rel_errors(model, val_loader, device, y_mean, y_std, is_complex)
            test_l1, test_l2 = compute_rel_errors(model, test_loader, device, y_mean, y_std, is_complex)

            config_result = {
                'width': width,
                'epsilon': epsilon,
                'product': width / epsilon,
                'k_in': k_in,
                'k_out': k_out,
                'num_params': num_params,
                'train_rel_l1': train_l1,
                'train_rel_l2': train_l2,
                'val_rel_l1': val_l1,
                'val_rel_l2': val_l2,
                'test_rel_l1': test_l1,
                'test_rel_l2': test_l2,
                'best_val_loss': train_result['best_val_loss'],
                'best_epoch': train_result['best_epoch'],
                'training_time': train_result['training_time'],
            }

            # Save individual config result immediately
            with open(result_path, 'w') as f:
                json.dump(config_result, f, indent=2)

            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_path)

            product_results.append(config_result)
            print(f"  Train Rel L1: {train_l1:.4f}%, Val Rel L1: {val_l1:.4f}%, Test Rel L1: {test_l1:.4f}%")

            # Free GPU memory before next config
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results['configurations'][f'product_{product}'] = product_results

        # Save aggregated results after each product (not just at end)
        results_file = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_v2_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    results_file = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_v2_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plot
    create_plot_v2(results, results_dir, dataset_name)
    
    print(f"\nCompleted dataset (Version 2): {dataset_name}")
    return results


def create_plot_v2(results, results_dir, dataset_name):
    """Create plot for Version 2: width and epsilon sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, product in enumerate(FIXED_WIDTH_OVER_EPSILON):
        product_key = f'product_{product}'
        if product_key not in results['configurations']:
            continue
        
        configs = results['configurations'][product_key]
        width_list = [c['width'] for c in configs]
        test_l1_list = [c['test_rel_l1'] for c in configs]
        test_l2_list = [c['test_rel_l2'] for c in configs]
        
        # Sort by width
        sorted_data = sorted(zip(width_list, test_l1_list, test_l2_list))
        width_list, test_l1_list, test_l2_list = zip(*sorted_data)
        
        axes[0].plot(width_list, test_l1_list, 'o-', label=f'Product={product}', 
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
        axes[1].plot(width_list, test_l2_list, 'o-', label=f'Product={product}', 
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Width', fontsize=12)
    axes[0].set_ylabel('Test Relative L1 Error (%)', fontsize=12)
    axes[0].set_title(f'Complexity Analysis V2: {dataset_name}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    axes[1].set_xlabel('Width', fontsize=12)
    axes[1].set_ylabel('Test Relative L2 Error (%)', fontsize=12)
    axes[1].set_title(f'Complexity Analysis V2: {dataset_name}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_v2_complexity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")

def create_plot(results, results_dir, dataset_name):
    """Create plot similar to Figure 1 in the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, product in enumerate(FIXED_PRODUCTS):
        product_key = f'product_{product}'
        if product_key not in results['configurations']:
            continue
        
        configs = results['configurations'][product_key]
        modes_list = [c['modes'] for c in configs]
        test_l1_list = [c['test_rel_l1'] for c in configs]
        test_l2_list = [c['test_rel_l2'] for c in configs]
        
        # Sort by modes
        sorted_data = sorted(zip(modes_list, test_l1_list, test_l2_list))
        modes_list, test_l1_list, test_l2_list = zip(*sorted_data)
        
        axes[0].plot(modes_list, test_l1_list, 'o-', label=f'Product={product}', 
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
        axes[1].plot(modes_list, test_l2_list, 'o-', label=f'Product={product}', 
                    color=colors[idx % len(colors)], linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Number of Modes', fontsize=12)
    axes[0].set_ylabel('Test Relative L1 Error (%)', fontsize=12)
    axes[0].set_title(f'Complexity Analysis: {dataset_name}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    axes[1].set_xlabel('Number of Modes', fontsize=12)
    axes[1].set_ylabel('Test Relative L2 Error (%)', fontsize=12)
    axes[1].set_title(f'Complexity Analysis: {dataset_name}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{dataset_name.replace('.', '_')}_complexity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")

def main():
    """Main function."""
    scratch_dir = '/mnt/c/Users/dvod/MRBNO/data'
    results_dir = '/mnt/c/Users/dvod/MRBNO/data/complexity_analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("MRBNO Lift Simple Complexity Analysis")
    print("="*70)
    print(f"\nHyperparameters:")
    for key, value in HYPERPARAMETERS.items():
        print(f"  {key}: {value}")
    print(f"\nTraining Schedule:")
    for key, value in TRAINING_SCHEDULE.items():
        print(f"  {key}: {value}")
    print(f"\nVersion 1 - Fixed Products: {FIXED_PRODUCTS}")
    print(f"\nConfigurations per product (V1):")
    for product in FIXED_PRODUCTS:
        print(f"  Product {product}: {len(CONFIGURATIONS[product])} configurations")
        for modes, width in CONFIGURATIONS[product]:
            print(f"    modes={modes}, width={width} (product={modes*width})")
    
    print(f"\nVersion 2 - Fixed Products: {FIXED_WIDTH_OVER_EPSILON}")
    print(f"\nConfigurations per product (V2):")
    for product in FIXED_WIDTH_OVER_EPSILON:
        print(f"  Product {product}: {len(CONFIGURATIONS_V2[product])} configurations")
        for width, epsilon in CONFIGURATIONS_V2[product]:
            print(f"    width={width}, epsilon={epsilon:.6f} (product={width/epsilon:.0f})")
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing {len(DATASETS)} datasets...")
    
    all_results = {}
    all_results_v2 = {}
    
    # Process Version 1 (width × k sweep)
    print("\n" + "="*70)
    print("VERSION 1: Sweep over width and k (modes) with fixed width × k")
    print("="*70)
    for dataset_name, dataset_type in DATASETS:
        try:
            result = process_dataset(dataset_name, dataset_type, results_dir, scratch_dir)  # scratch_dir is now the data dir
            if result:
                all_results[dataset_name] = result
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Process Version 2 (width × 1/epsilon sweep)
    print("\n" + "="*70)
    print("VERSION 2: Sweep over width and epsilon with fixed width × (1/epsilon)")
    print("="*70)
    for dataset_name, dataset_type in DATASETS:
        try:
            result = process_dataset_v2(dataset_name, dataset_type, results_dir, scratch_dir)  # scratch_dir is now the data dir
            if result:
                all_results_v2[dataset_name] = result
        except Exception as e:
            print(f"Error processing {dataset_name} (V2): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'hyperparameters': HYPERPARAMETERS,
            'training_schedule': TRAINING_SCHEDULE,
            'configurations_v1': CONFIGURATIONS,
            'configurations_v2': CONFIGURATIONS_V2,
            'datasets_processed_v1': list(all_results.keys()),
            'datasets_processed_v2': list(all_results_v2.keys()),
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("All datasets processed!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

