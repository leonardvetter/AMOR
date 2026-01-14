#!/usr/bin/env python3
"""
Baseline training: Train MRBNOLiftSimple on full uncompressed dataset.
Uses same hyperparameters and training schedule as two-stage training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

from train_two_stage_nonlinear import (
    load_nonlinear_dataset, PCAReducerMultiChannel,
    relative_l1_loss, relative_l2_error
)

try:
    from MRBNO_lift_simple import MRBNOLiftSimple
except ImportError:
    print("Error: Could not import MRBNOLiftSimple")
    sys.exit(1)


def train_baseline(model, train_loader, val_loader, test_loader, epochs, lr, device):
    """Train baseline model on full dataset."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Baseline Epoch {epoch+1:03d}/{epochs}] train_rel_l1={train_loss:.4e} train_rel_l2={train_rel_l2:.4e} val_rel_l1={val_loss:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    model.eval()
    test_rel_l1 = 0.0
    test_rel_l2 = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            test_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
    test_rel_l1 /= len(test_loader.dataset)
    test_rel_l2 /= len(test_loader.dataset)
    
    return best_val_loss, test_rel_l1, test_rel_l2


def main():
    parser = argparse.ArgumentParser(description='Baseline training on full dataset')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    H, W = X_train.shape[2], X_train.shape[3]
    in_channels = X_train.shape[1]
    out_channels = Y_train.shape[1]
    
    print("\n" + "="*70)
    print("COMPUTING PCA FOR BLOCKS")
    print("="*70)
    rb = PCAReducerMultiChannel().fit(X_train, Y_train)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    pca_components = rb.output_components[0] if rb.output_components.ndim == 3 else rb.output_components
    pca_mean = rb.output_mean[0] if rb.output_mean.ndim == 2 else rb.output_mean
    
    print("\n" + "="*70)
    print("BASELINE: Training on full dataset")
    print("="*70)
    
    baseline_model = MRBNOLiftSimple(
        in_channels=in_channels, out_channels=out_channels, width=args.width,
        n_modes=k_out, n_layers=args.n_layers, add_coords=False,
        norm="group", dropout_p=0.0, residual=True,
        pca_components=pca_components, pca_mean=pca_mean
    )
    
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    _, test_rel_l1, test_rel_l2 = train_baseline(baseline_model, train_loader, val_loader, test_loader, args.epochs, args.lr, device)
    
    print("\n" + "="*70)
    print("BASELINE TEST RESULTS")
    print("="*70)
    print(f"Test error - rel L1: {test_rel_l1:.6e}, rel L2: {test_rel_l2:.6e}")
    
    checkpoint_path = os.path.join(checkpoint_dir, 'baseline_nonlinear.pt')
    torch.save(baseline_model.state_dict(), checkpoint_path)
    print(f"\nBaseline checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    main()

