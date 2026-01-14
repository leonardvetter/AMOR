#!/usr/bin/env python3
"""
Train MRBNO Lift Simple, FNO, and PCANet on all 3 nonlinear diffusion datasets.

Trains each model on each dataset, then evaluates on all 3 datasets (cross-dataset evaluation).
Saves results to tables with train/val/test errors in relative L1 and L2.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(repo_root, 'multilayer_RBNO', 'models')
sys.path.insert(0, models_dir)

from MRBNO_lift_simple import MRBNOLiftSimple
from FNO_model import FNO2d
from PCANet import PCANet
from MRBNO_simple import PCAReducerMultiChannel

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
    'train_frac': 0.7,
    'val_frac': 0.2,
    'test_frac': 0.1,
    'num_samples': 10000,
    'epsilon': 1e-5,
}

DATASETS = {
    'nonlinear_diffusion_dataset.npz': 'standard',
    'nonlinear_diffusion_dataset_smooth.npz': 'smooth',
    'nonlinear_diffusion_dataset_rough.npz': 'rough',
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def relative_l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

def relative_l2_loss(pred, target):
    return torch.sqrt(torch.mean(((pred - target) / (torch.abs(target) + 1e-8))**2))

def load_nonlinear_dataset(dataset_path, num_samples=10000):
    data = np.load(dataset_path, allow_pickle=True)
    f = data['f'][:num_samples]
    m = data['m'][:num_samples]
    u = data['u'][:num_samples]
    X = np.stack([f, m], axis=1)
    Y = u[:, np.newaxis, :, :]
    return X, Y

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
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{hyperparams['epochs']}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    return best_val_loss, best_epoch

def evaluate_model(model, loader, device, y_mean, y_std):
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

def main():
    scratch_dir = os.environ.get('SCRATCH', '/tmp')
    data_dir = os.path.join(scratch_dir, 'data')
    results_dir = os.path.join(scratch_dir, 'nonlinear_model_comparison')
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(HYPERPARAMETERS['seed'])
    
    results_main = []
    results_cross = []
    
    for train_dataset_name, train_dataset_type in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Training on: {train_dataset_name} ({train_dataset_type})")
        print(f"{'='*70}")
        
        train_dataset_path = os.path.join(data_dir, train_dataset_name)
        if not os.path.exists(train_dataset_path):
            print(f"Warning: {train_dataset_path} not found. Skipping.")
            continue
        
        X_train_full, Y_train_full = load_nonlinear_dataset(train_dataset_path, HYPERPARAMETERS['num_samples'])
        
        n = len(X_train_full)
        n_train = int(HYPERPARAMETERS['train_frac'] * n)
        n_val = int(HYPERPARAMETERS['val_frac'] * n)
        
        X_train, X_val, X_test = X_train_full[:n_train], X_train_full[n_train:n_train+n_val], X_train_full[n_train+n_val:]
        Y_train, Y_val, Y_test = Y_train_full[:n_train], Y_train_full[n_train:n_train+n_val], Y_train_full[n_train+n_val:]
        
        X_mean, X_std = X_train.mean(axis=(0, 2, 3), keepdims=True), X_train.std(axis=(0, 2, 3), keepdims=True)
        Y_mean, Y_std = Y_train.mean(axis=(0, 2, 3), keepdims=True), Y_train.std(axis=(0, 2, 3), keepdims=True)
        
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
        
        train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
        
        rb = PCAReducerMultiChannel(bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False).fit(X_train, Y_train)
        k_in, k_out = rb.k_from_cutoff(j_in=1-HYPERPARAMETERS['epsilon'], j_out=1-HYPERPARAMETERS['epsilon'])
        k_in = min(k_in, rb.pca_in.n_components_)
        k_out = min(k_out, rb.pca_out.n_components_)
        
        pca_components = rb.pca_out.components_[:k_out]
        pca_mean = rb.pca_out.mean_
        
        models_to_train = {
            'MRBNO_Lift_Simple': MRBNOLiftSimple(
                in_channels=2, out_channels=1, width=HYPERPARAMETERS['width'],
                n_modes=k_out, n_layers=HYPERPARAMETERS['layers'], add_coords=True,
                norm="group", dropout_p=0.0, residual=True,
                pca_components=pca_components, pca_mean=pca_mean
            ),
            'FNO': FNO2d(
                modes1=HYPERPARAMETERS['fno_modes'][0], modes2=HYPERPARAMETERS['fno_modes'][1],
                width=HYPERPARAMETERS['width'], in_channels=2, out_channels=1
            ),
            'PCANet': PCANet(
                rb=rb, k_in=k_in, k_out=k_out, hidden=256, depth=3, dropout=0.0, init_scale=0.1
            ),
        }
        
        for model_name, model in models_to_train.items():
            print(f"\n--- Training {model_name} ---")
            set_seed(HYPERPARAMETERS['seed'])
            
            best_val_loss, best_epoch = train_model(model, train_loader, val_loader, device, HYPERPARAMETERS)
            
            train_l1, train_l2 = evaluate_model(model, train_loader, device, Y_mean, Y_std)
            val_l1, val_l2 = evaluate_model(model, val_loader, device, Y_mean, Y_std)
            test_l1, test_l2 = evaluate_model(model, test_loader, device, Y_mean, Y_std)
            
            results_main.append({
                'train_dataset': train_dataset_type,
                'model': model_name,
                'train_rel_l1': train_l1,
                'train_rel_l2': train_l2,
                'val_rel_l1': val_l1,
                'val_rel_l2': val_l2,
                'test_rel_l1': test_l1,
                'test_rel_l2': test_l2,
            })
            
            print(f"  Train: Rel L1={train_l1:.4f}%, Rel L2={train_l2:.4f}%")
            print(f"  Val: Rel L1={val_l1:.4f}%, Rel L2={val_l2:.4f}%")
            print(f"  Test: Rel L1={test_l1:.4f}%, Rel L2={test_l2:.4f}%")
            
            for test_dataset_name, test_dataset_type in DATASETS.items():
                if test_dataset_name == train_dataset_name:
                    continue
                
                test_dataset_path = os.path.join(data_dir, test_dataset_name)
                if not os.path.exists(test_dataset_path):
                    continue
                
                X_test_cross, Y_test_cross = load_nonlinear_dataset(test_dataset_path, HYPERPARAMETERS['num_samples'])
                n_test = int(HYPERPARAMETERS['test_frac'] * len(X_test_cross))
                X_test_cross = X_test_cross[:n_test]
                Y_test_cross = Y_test_cross[:n_test]
                
                X_test_cross = (X_test_cross - X_mean) / (X_std + 1e-8)
                Y_test_cross = (Y_test_cross - Y_mean) / (Y_std + 1e-8)
                
                X_test_cross_t = torch.from_numpy(X_test_cross).float()
                Y_test_cross_t = torch.from_numpy(Y_test_cross).float()
                test_cross_loader = DataLoader(TensorDataset(X_test_cross_t, Y_test_cross_t), batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
                
                test_cross_l1, test_cross_l2 = evaluate_model(model, test_cross_loader, device, Y_mean, Y_std)
                
                results_cross.append({
                    'train_dataset': train_dataset_type,
                    'test_dataset': test_dataset_type,
                    'model': model_name,
                    'test_rel_l1': test_cross_l1,
                    'test_rel_l2': test_cross_l2,
                })
    
    df_main = pd.DataFrame(results_main)
    df_cross = pd.DataFrame(results_cross)
    
    main_table_path = os.path.join(results_dir, 'nonlinear_main_results.csv')
    cross_table_path = os.path.join(results_dir, 'nonlinear_cross_dataset_results.csv')
    
    df_main.to_csv(main_table_path, index=False)
    df_cross.to_csv(cross_table_path, index=False)
    
    print(f"\n{'='*70}")
    print("Results saved:")
    print(f"  Main results: {main_table_path}")
    print(f"  Cross-dataset results: {cross_table_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

