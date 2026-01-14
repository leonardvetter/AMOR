#!/usr/bin/env python3
"""
Run Stage 2 experiments with different configurations:
1. Stage 2 with frozen blocks (original)
2. Stage 2 with unfrozen blocks (small LR for blocks)
3. Stage 2 with 1% random samples from all trajectories
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

from train_two_stage_nonlinear import (
    load_nonlinear_dataset, PCAReducerMultiChannel, 
    compute_pca_coefficients, train_stage2, train_stage2_unfrozen,
    relative_l1_loss, relative_l2_error
)
from mrbno_two_stage import MRBNOStage1, MRBNOStage2


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_rel_l1 = 0.0
    total_rel_l2 = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            total_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
    return total_rel_l1 / len(test_loader.dataset), total_rel_l2 / len(test_loader.dataset)


def load_stage1_model(stage1_path, rb, k_in, k_out, H, W, in_channels, out_channels, width, n_layers, device):
    """Load Stage 1 model from checkpoint."""
    pca_components_in = rb.input_components
    pca_mean_in = rb.input_mean
    pca_components_out = rb.output_components[0] if rb.output_components.ndim == 3 else rb.output_components
    pca_mean_out = rb.output_mean[0] if rb.output_mean.ndim == 2 else rb.output_mean
    
    stage1_model = MRBNOStage1(
        in_channels=in_channels, out_channels=out_channels, width=width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=pca_components_in, pca_mean_in=pca_mean_in,
        pca_components_out=pca_components_out, pca_mean_out=pca_mean_out
    )
    stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
    stage1_model.to(device)
    return stage1_model


def main():
    parser = argparse.ArgumentParser(description='Stage 2 experiments')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--stage1_checkpoint', type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stage2_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_blocks', type=float, default=2e-5, help='Learning rate for blocks in unfrozen version')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_frozen', action='store_true', help='Skip frozen blocks experiment')
    parser.add_argument('--skip_unfrozen', action='store_true', help='Skip unfrozen blocks experiment')
    parser.add_argument('--skip_random', action='store_true', help='Skip random 1% sampling experiment')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    
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
    print(f"Detected channels: in={in_channels}, out={out_channels}")
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    rb = PCAReducerMultiChannel().fit(X_train, Y_train)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    print("\n" + "="*70)
    print("LOADING STAGE 1 MODEL")
    print("="*70)
    stage1_model = load_stage1_model(
        args.stage1_checkpoint, rb, k_in, k_out, H, W,
        in_channels, out_channels, args.width, args.n_layers, device
    )
    print(f"Stage 1 model loaded from {args.stage1_checkpoint}")
    
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    results = {}
    
    if not args.skip_frozen:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Stage 2 with frozen blocks (original)")
        print("="*70)
        
        stage2_model_frozen = MRBNOStage2(stage1_model, in_channels=in_channels, out_channels=out_channels, 
                                          width=args.width, add_coords=False, freeze_blocks=True)
        
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
        
        train_stage2(stage2_model_frozen, train_loader_stage2, val_loader_stage2, 
                     args.stage2_epochs, args.lr, device)
        
        checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'stage2_frozen.pt')
        torch.save(stage2_model_frozen.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        test_rel_l1_frozen, test_rel_l2_frozen = evaluate_model(stage2_model_frozen, test_loader, device)
        results['frozen'] = (test_rel_l1_frozen, test_rel_l2_frozen)
        print(f"\nFrozen blocks test error - rel L1: {test_rel_l1_frozen:.6e}, rel L2: {test_rel_l2_frozen:.6e}")
    
    if not args.skip_unfrozen:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Stage 2 with unfrozen blocks (small LR for blocks)")
        print("="*70)
        
        stage2_model_unfrozen = MRBNOStage2(stage1_model, in_channels=in_channels, out_channels=out_channels, 
                                           width=args.width, add_coords=False, freeze_blocks=False)
        
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
        
        train_stage2_unfrozen(stage2_model_unfrozen, train_loader_stage2, val_loader_stage2, 
                            args.stage2_epochs, args.lr, args.lr_blocks, device)
        
        checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'stage2_unfrozen.pt')
        torch.save(stage2_model_unfrozen.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        test_rel_l1_unfrozen, test_rel_l2_unfrozen = evaluate_model(stage2_model_unfrozen, test_loader, device)
        results['unfrozen'] = (test_rel_l1_unfrozen, test_rel_l2_unfrozen)
        print(f"\nUnfrozen blocks test error - rel L1: {test_rel_l1_unfrozen:.6e}, rel L2: {test_rel_l2_unfrozen:.6e}")
    
    if not args.skip_random:
        print("\n" + "="*70)
        print("EXPERIMENT 3: Stage 2 with 1% random samples from all trajectories")
        print("="*70)
        
        np.random.seed(42)
        n_samples = int(len(X_train) * 0.01)
        random_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        random_indices = np.sort(random_indices)
        
        X_train_random = X_train[random_indices]
        Y_train_random = Y_train[random_indices]
        
        print(f"Random sample: {len(X_train_random)} samples (1% of {len(X_train)})")
        print(f"Random sample shapes: X={X_train_random.shape}, Y={Y_train_random.shape}")
        
        stage2_model_random = MRBNOStage2(stage1_model, in_channels=in_channels, out_channels=out_channels, 
                                         width=args.width, add_coords=False, freeze_blocks=True)
        
        n_val_samples = min(200, len(X_val))
        X_val_stage2 = X_val[:n_val_samples]
        Y_val_stage2 = Y_val[:n_val_samples]
        
        train_loader_stage2 = DataLoader(TensorDataset(
            torch.from_numpy(X_train_random).float(),
            torch.from_numpy(Y_train_random).float()
        ), batch_size=args.batch_size, shuffle=True)
        
        val_loader_stage2 = DataLoader(TensorDataset(
            torch.from_numpy(X_val_stage2).float(),
            torch.from_numpy(Y_val_stage2).float()
        ), batch_size=args.batch_size, shuffle=False)
        
        train_stage2(stage2_model_random, train_loader_stage2, val_loader_stage2, 
                     args.stage2_epochs, args.lr, device)
        
        checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'stage2_random_1pct.pt')
        torch.save(stage2_model_random.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        test_rel_l1_random, test_rel_l2_random = evaluate_model(stage2_model_random, test_loader, device)
        results['random_1pct'] = (test_rel_l1_random, test_rel_l2_random)
        print(f"\nRandom 1% sample test error - rel L1: {test_rel_l1_random:.6e}, rel L2: {test_rel_l2_random:.6e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for exp_name, (test_rel_l1, test_rel_l2) in results.items():
        print(f"{exp_name}: rel L1={test_rel_l1:.6e}, rel L2={test_rel_l2:.6e}")
    print("="*70)


if __name__ == '__main__':
    main()

