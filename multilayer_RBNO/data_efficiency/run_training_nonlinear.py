#!/usr/bin/env python3
"""
Main script for two-stage training on Nonlinear Diffusion dataset.
Runs Stage 1 with 600 epochs and saves checkpoint.
Uses new MRBNO.py model architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
from train_two_stage_nonlinear import (
    load_nonlinear_dataset, PCAReducerMultiChannel, 
    MRBNOStage1, train_stage1, relative_l2_error, relative_l1_loss
)
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    parser = argparse.ArgumentParser(description='Two-stage training on Nonlinear Diffusion')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stage1_epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
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
    
    print("\n" + "="*70)
    print("STAGE 1: Training in real space, loss in PCA space")
    print("="*70)
    
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
    
    stage1_model = MRBNOStage1(
        rb=rb, width=args.width, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, use_batch_norm=True, use_skip_connections=True,
        use_coordinates=False, add_constant_mode=False, use_factorized_weights=False
    )
    
    train_stage1(stage1_model, train_loader, val_loader, rb, k_out, args.stage1_epochs, args.lr, device)
    
    stage1_path = os.path.join(checkpoint_dir, f'stage1_nonlinear_{args.stage1_epochs}epochs.pt')
    torch.save({
        'model_state_dict': stage1_model.state_dict(),
        'rb': rb,
        'k_in': k_in,
        'k_out': k_out,
        'width': args.width,
        'layers': args.layers
    }, stage1_path)
    print(f"\nStage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET (in real space)")
    print("="*70)
    stage1_model.eval()
    
    pca_out_mean = torch.from_numpy(rb.pca_out.mean_).float().to(device)
    pca_out_components = torch.from_numpy(rb.pca_out.components_[:k_out]).float().to(device)
    
    test_rel_l1 = 0.0
    test_rel_l2 = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred_spatial = stage1_model(xb)
            test_rel_l1 += relative_l1_loss(pred_spatial, yb).item() * xb.size(0)
            test_rel_l2 += relative_l2_error(pred_spatial, yb) * xb.size(0)
    test_rel_l1 /= len(test_loader.dataset)
    test_rel_l2 /= len(test_loader.dataset)
    print(f"Test error - rel L1: {test_rel_l1:.6e}, rel L2: {test_rel_l2:.6e}")


if __name__ == '__main__':
    main()
