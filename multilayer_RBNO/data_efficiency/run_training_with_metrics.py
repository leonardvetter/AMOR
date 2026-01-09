#!/usr/bin/env python3
"""
Enhanced training script with memory tracking and baseline comparison.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import psutil
import gc

from train_two_stage_kolmogorov import (
    load_kolmogorov_dataset, PCAReducerMultiChannel, 
    compute_pca_coefficients, train_stage1, train_stage2
)
from mrbno_two_stage import MRBNOStage1, MRBNOStage2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
try:
    from MRBNO_lift_simple import MRBNOLiftSimple
except ImportError:
    print("Warning: Could not import MRBNOLiftSimple for baseline comparison")
    MRBNOLiftSimple = None


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(test_loader.dataset)


def train_baseline(X_train, Y_train, X_val, Y_val, X_test, Y_test, rb, k_in, k_out, H, W, 
                   width, n_layers, batch_size, epochs, lr, device):
    """Train baseline model on full dataset."""
    if MRBNOLiftSimple is None:
        return None, None
    
    print("\n" + "="*70)
    print("BASELINE: Training on full dataset")
    print("="*70)
    
    baseline_model = MRBNOLiftSimple(
        in_channels=1, out_channels=1, width=width, n_modes=k_out, n_layers=n_layers,
        add_coords=False, norm="group", dropout_p=0.0, residual=True,
        pca_components=rb.output_components, pca_mean=rb.output_mean
    )
    
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    ), batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    ), batch_size=batch_size, shuffle=False)
    
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float()
    ), batch_size=batch_size, shuffle=False)
    
    baseline_model.to(device)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        baseline_model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = baseline_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        baseline_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = baseline_model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[Baseline Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e}")
    
    test_loss = evaluate_model(baseline_model, test_loader, device)
    
    checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    baseline_checkpoint_path = os.path.join(checkpoint_dir, f'baseline_{epochs}epochs.pt')
    torch.save(baseline_model.state_dict(), baseline_checkpoint_path)
    print(f"Baseline checkpoint saved to {baseline_checkpoint_path}")
    
    return baseline_model, test_loss


def main():
    parser = argparse.ArgumentParser(description='Two-stage training with metrics')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/2D_NS_Re500.npy')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stage1_epochs', type=int, default=300)
    parser.add_argument('--stage2_epochs', type=int, default=100)
    parser.add_argument('--baseline_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline training')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("MEMORY TRACKING: Loading dataset")
    print("="*70)
    mem_before = get_memory_usage()
    print(f"Memory before loading: {mem_before:.2f} GB")
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_kolmogorov_dataset(args.data_path)
    H, W = X_train.shape[2], X_train.shape[3]
    
    mem_after_load = get_memory_usage()
    dataset_memory_gb = mem_after_load - mem_before
    
    dataset_size_bytes = X_train.nbytes + Y_train.nbytes + X_val.nbytes + Y_val.nbytes + X_test.nbytes + Y_test.nbytes
    dataset_memory_gb_exact = dataset_size_bytes / (1024 ** 3)
    
    print(f"Memory after loading: {mem_after_load:.2f} GB")
    print(f"Dataset memory (calculated): {dataset_memory_gb_exact:.2f} GB")
    print(f"Dataset shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    rb = PCAReducerMultiChannel().fit(X_train, Y_train)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    X_train_pca, Y_train_pca = compute_pca_coefficients(X_train, Y_train, rb, k_in, k_out)
    X_val_pca, Y_val_pca = compute_pca_coefficients(X_val, Y_val, rb, k_in, k_out)
    X_test_pca, Y_test_pca = compute_pca_coefficients(X_test, Y_test, rb, k_in, k_out)
    
    pca_memory_bytes = X_train_pca.nbytes + Y_train_pca.nbytes + X_val_pca.nbytes + Y_val_pca.nbytes + X_test_pca.nbytes + Y_test_pca.nbytes
    pca_memory_gb = pca_memory_bytes / (1024 ** 3)
    
    print(f"\nPCA compressed dataset memory: {pca_memory_gb:.4f} GB")
    print(f"Memory reduction: {dataset_memory_gb_exact / pca_memory_gb:.2f}x")
    print(f"Compression ratio: {pca_memory_gb / dataset_memory_gb_exact * 100:.2f}% of original")
    
    print("\n" + "="*70)
    print("STAGE 1: Training in PCA space")
    print("="*70)
    
    train_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_train_pca).float(),
        torch.from_numpy(Y_train_pca).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_val_pca).float(),
        torch.from_numpy(Y_val_pca).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    stage1_model = MRBNOStage1(
        in_channels=1, out_channels=1, width=args.width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=args.n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=rb.input_components, pca_mean_in=rb.input_mean,
        pca_components_out=rb.output_components, pca_mean_out=rb.output_mean
    )
    
    train_stage1(stage1_model, train_loader_pca, val_loader_pca, args.stage1_epochs, args.lr, device)
    
    checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    stage1_checkpoint_path = os.path.join(checkpoint_dir, f'stage1_{args.stage1_epochs}epochs.pt')
    torch.save(stage1_model.state_dict(), stage1_checkpoint_path)
    print(f"\nStage 1 checkpoint saved to {stage1_checkpoint_path}")
    
    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning in real space")
    print("="*70)
    
    stage2_model = MRBNOStage2(stage1_model, in_channels=1, out_channels=1, width=args.width, add_coords=False)
    
    n_stage2_samples = min(1000, len(X_train))
    X_train_stage2 = X_train[:n_stage2_samples]
    Y_train_stage2 = Y_train[:n_stage2_samples]
    X_val_stage2 = X_val[:min(200, len(X_val))]
    Y_val_stage2 = Y_val[:min(200, len(X_val))]
    
    train_loader_stage2 = DataLoader(TensorDataset(
        torch.from_numpy(X_train_stage2).float(),
        torch.from_numpy(Y_train_stage2).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_stage2 = DataLoader(TensorDataset(
        torch.from_numpy(X_val_stage2).float(),
        torch.from_numpy(Y_val_stage2).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    train_stage2(stage2_model, train_loader_stage2, val_loader_stage2, args.stage2_epochs, args.lr, device)
    
    checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    stage2_checkpoint_path = os.path.join(checkpoint_dir, f'stage2_{args.stage2_epochs}epochs.pt')
    torch.save(stage2_model.state_dict(), stage2_checkpoint_path)
    print(f"\nStage 2 checkpoint saved to {stage2_checkpoint_path}")
    
    print("\n" + "="*70)
    print("EVALUATION: Stage 2 model on test set")
    print("="*70)
    
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    stage2_test_loss = evaluate_model(stage2_model, test_loader, device)
    print(f"Stage 2 test loss (physical space): {stage2_test_loss:.6e}")
    
    baseline_test_loss = None
    if not args.skip_baseline and MRBNOLiftSimple is not None:
        gc.collect()
        baseline_model, baseline_test_loss = train_baseline(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, rb, k_in, k_out, H, W,
            args.width, args.n_layers, args.batch_size, args.baseline_epochs, args.lr, device
        )
        if baseline_test_loss is not None:
            print(f"Baseline test loss (full dataset): {baseline_test_loss:.6e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Dataset: 2D_NS_Re500.npy (Kolmogorov Flow)")
    print(f"Original dataset memory: {dataset_memory_gb_exact:.4f} GB")
    print(f"PCA compressed dataset memory: {pca_memory_gb:.4f} GB")
    print(f"Memory reduction: {dataset_memory_gb_exact / pca_memory_gb:.2f}x")
    print(f"Stage 2 test error (physical space): {stage2_test_loss:.6e}")
    if baseline_test_loss is not None:
        print(f"Baseline test error (full dataset): {baseline_test_loss:.6e}")
        print(f"Error ratio (stage2/baseline): {stage2_test_loss / baseline_test_loss:.4f}")


if __name__ == '__main__':
    main()

