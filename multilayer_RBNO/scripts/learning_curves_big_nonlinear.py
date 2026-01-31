#!/usr/bin/env python3
"""
Generate learning curves for big nonlinear dataset comparing FNO vs MRBNO vs PCANet.

Uses fixed 10% test split. For remaining 90%, creates training splits of 10%, 20%, ..., 100%.
Each split uses 80% train, 20% validation. Trains all three models and evaluates on test/train/val.
Generates plots showing rel L1 and rel L2 error on test split vs number of training samples.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from FNO_model import FNO2d
from MRBNO import MRBNO
from PCANet import PCANet
from PCAReducer import PCAReducerMultiChannel

HYPERPARAMETERS = {
    'layers': 4,
    'width': 24,
    'modes': 144,
    'fno_modes': (12, 12),
    'lr': 2e-3,
    'weight_decay': 1e-4,
    'max_grad_norm': 3.0,
    'epochs': 300,
    'early_stopping_patience': 50,
    'early_stopping_min_delta': 0.0,
    'batch_size': 32,
    'seed': 42,
    'test_frac': 0.1,
    'train_frac': 0.8,
    'val_frac': 0.2,
    'epsilon': 1e-5,
}

TRAINING_SPLITS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def relative_l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

def compute_rel_errors(model, loader, device, y_mean, y_std):
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
                pred_mag = pred_denorm[i, 0].cpu().numpy()
                true_mag = yb_denorm[i, 0].cpu().numpy()
                
                l1_err = np.abs(pred_mag - true_mag).sum()
                l1_norm = np.abs(true_mag).sum()
                rel_l1 = l1_err / (l1_norm + 1e-10)
                rel_l1_errors.append(rel_l1)
                
                l2_err = np.sqrt(np.sum((pred_mag - true_mag)**2))
                l2_norm = np.sqrt(np.sum(true_mag**2))
                rel_l2 = l2_err / (l2_norm + 1e-10)
                rel_l2_errors.append(rel_l2)
    
    return np.mean(rel_l1_errors) * 100, np.mean(rel_l2_errors) * 100

def train_model(model, train_loader, val_loader, device, hyperparams):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=0.0)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(hyperparams['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss - hyperparams['early_stopping_min_delta']:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= hyperparams['early_stopping_patience']:
            break
    
    return best_val_loss, best_epoch

def main():
    scratch_dir = os.environ.get('SCRATCH', '/tmp')
    data_dir = os.path.join(scratch_dir, 'data')
    results_dir = os.path.join(scratch_dir, 'learning_curves_big_nonlinear')
    os.makedirs(results_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, 'nonlinear_diffusion_dataset_big.npz')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    print("="*70)
    print("LEARNING CURVES: FNO vs MRBNO vs PCANet on Big Nonlinear Dataset")
    print("="*70)
    print(f"\nDataset: {dataset_path}")
    print(f"Results directory: {results_dir}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    set_seed(HYPERPARAMETERS['seed'])
    
    data = np.load(dataset_path, allow_pickle=True)
    f_all = data['f']
    m_all = data['m']
    u_all = data['u']
    
    X = np.stack([f_all, m_all], axis=1)
    Y = u_all[:, np.newaxis, :, :]
    
    print(f"Loaded dataset: {len(X)} samples")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}\n")
    
    n_total = len(X)
    n_test = int(HYPERPARAMETERS['test_frac'] * n_total)
    
    indices = np.arange(n_total)
    np.random.seed(HYPERPARAMETERS['seed'])
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_val_indices = indices[n_test:]
    
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    
    n_train_val = len(train_val_indices)
    print(f"Fixed test split: {n_test} samples ({HYPERPARAMETERS['test_frac']*100:.0f}%)")
    print(f"Remaining for train/val: {n_train_val} samples ({100-HYPERPARAMETERS['test_frac']*100:.0f}%)\n")
    
    results = []
    
    for split_frac in TRAINING_SPLITS:
        print(f"{'='*70}")
        print(f"Training Split: {split_frac*100:.0f}% of remaining data ({int(split_frac * n_train_val)} samples)")
        print(f"{'='*70}")
        
        n_split = int(split_frac * n_train_val)
        split_indices = train_val_indices[:n_split]
        
        X_split = X[split_indices]
        Y_split = Y[split_indices]
        
        n_train = int(HYPERPARAMETERS['train_frac'] * n_split)
        n_val = n_split - n_train
        
        X_train = X_split[:n_train]
        Y_train = Y_split[:n_train]
        X_val = X_split[n_train:]
        Y_val = Y_split[n_train:]
        
        print(f"  Train: {n_train} samples, Val: {n_val} samples")
        
        X_mean, X_std = X_train.mean(axis=(0, 2, 3), keepdims=True), X_train.std(axis=(0, 2, 3), keepdims=True)
        Y_mean, Y_std = Y_train.mean(axis=(0, 2, 3), keepdims=True), Y_train.std(axis=(0, 2, 3), keepdims=True)
        
        X_train = (X_train - X_mean) / (X_std + 1e-8)
        X_val = (X_val - X_mean) / (X_std + 1e-8)
        X_test_norm = (X_test - X_mean) / (X_std + 1e-8)
        Y_train = (Y_train - Y_mean) / (Y_std + 1e-8)
        Y_val = (Y_val - Y_mean) / (Y_std + 1e-8)
        Y_test_norm = (Y_test - Y_mean) / (Y_std + 1e-8)
        
        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).float()
        X_val_t = torch.from_numpy(X_val).float()
        Y_val_t = torch.from_numpy(Y_val).float()
        X_test_t = torch.from_numpy(X_test_norm).float()
        Y_test_t = torch.from_numpy(Y_test_norm).float()
        
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        
        rb = PCAReducerMultiChannel(bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False).fit(X_train, Y_train)
        k_in, k_out = rb.k_from_cutoff(j_in=1-HYPERPARAMETERS['epsilon'], j_out=1-HYPERPARAMETERS['epsilon'])
        k_in = min(k_in, rb.pca_in.n_components_)
        k_out = min(k_out, rb.pca_out.n_components_)
        
        models_to_train = {
            'FNO': FNO2d(
                modes1=HYPERPARAMETERS['fno_modes'][0], modes2=HYPERPARAMETERS['fno_modes'][1],
                width=HYPERPARAMETERS['width'], in_channels=2, out_channels=1
            ),
            'MRBNO': MRBNO(
                width=HYPERPARAMETERS['width'],
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=HYPERPARAMETERS['layers'],
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=True,
                use_coordinates=True,
                add_constant_mode=False,
                use_factorized_weights=False,
            ),
            'PCANet': PCANet(
                rb=rb, k_in=k_in, k_out=k_out, hidden=1024, depth=5, dropout=0.0, init_scale=0.1
            ),
        }
        
        for model_name, model in models_to_train.items():
            print(f"\n  Training {model_name}...")
            set_seed(HYPERPARAMETERS['seed'])
            
            best_val_loss, best_epoch = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)
            
            train_l1, train_l2 = compute_rel_errors(model, train_loader, device, Y_mean, Y_std)
            val_l1, val_l2 = compute_rel_errors(model, val_loader, device, Y_mean, Y_std)
            test_l1, test_l2 = compute_rel_errors(model, test_loader, device, Y_mean, Y_std)
            
            results.append({
                'split_frac': split_frac,
                'n_train': n_train,
                'n_val': n_val,
                'model': model_name,
                'train_rel_l1': train_l1,
                'train_rel_l2': train_l2,
                'val_rel_l1': val_l1,
                'val_rel_l2': val_l2,
                'test_rel_l1': test_l1,
                'test_rel_l2': test_l2,
            })
            
            print(f"    Train: Rel L1={train_l1:.4f}%, Rel L2={train_l2:.4f}%")
            print(f"    Val: Rel L1={val_l1:.4f}%, Rel L2={val_l2:.4f}%")
            print(f"    Test: Rel L1={test_l1:.4f}%, Rel L2={test_l2:.4f}%")
    
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    models = ['FNO', 'MRBNO_Lift_Simple', 'PCANet']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for model_name in models:
        model_results = [r for r in results if r['model'] == model_name]
        n_train_vals = [r['n_train'] for r in model_results]
        test_l1_vals = [r['test_rel_l1'] for r in model_results]
        test_l2_vals = [r['test_rel_l2'] for r in model_results]
        
        sorted_data = sorted(zip(n_train_vals, test_l1_vals, test_l2_vals))
        n_train_vals, test_l1_vals, test_l2_vals = zip(*sorted_data)
        
        axes[0].plot(n_train_vals, test_l1_vals, 'o-', label=model_name, linewidth=2, markersize=6)
        axes[1].plot(n_train_vals, test_l2_vals, 'o-', label=model_name, linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Number of Training Samples', fontsize=12)
    axes[0].set_ylabel('Test Relative L1 Error (%)', fontsize=12)
    axes[0].set_title('Learning Curves: Relative L1 Error', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of Training Samples', fontsize=12)
    axes[1].set_ylabel('Test Relative L2 Error (%)', fontsize=12)
    axes[1].set_title('Learning Curves: Relative L2 Error', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'learning_curves_big_nonlinear.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")
    
    import json
    results_file = os.path.join(results_dir, 'learning_curves_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

