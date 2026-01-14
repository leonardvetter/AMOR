#!/usr/bin/env python3
"""
Main script for two-stage training on Nonlinear Diffusion dataset.
Runs Stage 1 with 600 epochs and saves checkpoint.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
from train_two_stage_nonlinear import (
    load_nonlinear_dataset, PCAReducerMultiChannel, 
    compute_pca_coefficients, train_stage1, relative_l2_error
)
from mrbno_two_stage import MRBNOStage1
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    parser = argparse.ArgumentParser(description='Two-stage training on Nonlinear Diffusion')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stage1_epochs', type=int, default=600)
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
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    H, W = X_train.shape[2], X_train.shape[3]
    in_channels = X_train.shape[1]
    out_channels = Y_train.shape[1]
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    rb = PCAReducerMultiChannel().fit(X_train, Y_train)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    X_train_pca, Y_train_pca = compute_pca_coefficients(X_train, Y_train, rb, k_in, k_out)
    X_val_pca, Y_val_pca = compute_pca_coefficients(X_val, Y_val, rb, k_in, k_out)
    X_test_pca, Y_test_pca = compute_pca_coefficients(X_test, Y_test, rb, k_in, k_out)
    
    train_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_train_pca).float(),
        torch.from_numpy(Y_train_pca).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_val_pca).float(),
        torch.from_numpy(Y_val_pca).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    test_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_test_pca).float(),
        torch.from_numpy(Y_test_pca).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    print("\n" + "="*70)
    print("STAGE 1: Training in PCA space")
    print("="*70)
    
    pca_components_in = rb.input_components
    pca_mean_in = rb.input_mean
    pca_components_out = rb.output_components[0] if rb.output_components.ndim == 3 else rb.output_components
    pca_mean_out = rb.output_mean[0] if rb.output_mean.ndim == 2 else rb.output_mean
    
    stage1_model = MRBNOStage1(
        in_channels=in_channels, out_channels=out_channels, width=args.width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=args.n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=pca_components_in, pca_mean_in=pca_mean_in,
        pca_components_out=pca_components_out, pca_mean_out=pca_mean_out
    )
    
    train_stage1(stage1_model, train_loader_pca, val_loader_pca, args.stage1_epochs, args.lr, device)
    
    stage1_path = os.path.join(checkpoint_dir, f'stage1_nonlinear_{args.stage1_epochs}epochs.pt')
    torch.save(stage1_model.state_dict(), stage1_path)
    print(f"\nStage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    stage1_model.eval()
    test_rel_l1 = 0.0
    test_rel_l2 = 0.0
    from train_two_stage_nonlinear import relative_l1_loss
    with torch.no_grad():
        for xb, yb in test_loader_pca:
            xb, yb = xb.to(device), yb.to(device)
            pred = stage1_model(xb)
            test_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            test_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
    test_rel_l1 /= len(test_loader_pca.dataset)
    test_rel_l2 /= len(test_loader_pca.dataset)
    print(f"Test error - rel L1: {test_rel_l1:.6e}, rel L2: {test_rel_l2:.6e}")


if __name__ == '__main__':
    main()

