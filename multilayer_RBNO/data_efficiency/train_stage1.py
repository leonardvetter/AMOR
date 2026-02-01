#!/usr/bin/env python3
"""
Stage 1 training: Train in PCA coefficient space on 90% of train/val data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from train_two_stage import (
    load_nonlinear_dataset, PCAReducerMultiChannel, MRBNOStage1,
    train_stage1, spatial_to_pca_coefficients, spatial_to_pca_coefficients_output
)


def main():
    parser = argparse.ArgumentParser(description='Stage 1 training')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--k_in', type=int, default=144)
    parser.add_argument('--k_out', type=int, default=144)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    scratch_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'data_efficiency')
    checkpoint_dir = os.path.join(scratch_dir, 'checkpoints')
    log_dir = os.path.join(scratch_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    
    print("\n" + "="*70)
    print("STAGE 1 DATA SPLIT: 90% of train/val for Stage 1")
    print("="*70)
    
    n_train_val = len(X_train) + len(X_val)
    n_stage1 = int(0.9 * n_train_val)
    n_stage1_train = int(0.7 * n_stage1)
    
    X_stage1_all = np.concatenate([X_train, X_val], axis=0)
    Y_stage1_all = np.concatenate([Y_train, Y_val], axis=0)
    
    X_stage1_train = X_stage1_all[:n_stage1_train]
    Y_stage1_train = Y_stage1_all[:n_stage1_train]
    X_stage1_val = X_stage1_all[n_stage1_train:n_stage1]
    Y_stage1_val = Y_stage1_all[n_stage1_train:n_stage1]
    
    print(f"Stage 1 training samples: {len(X_stage1_train)}")
    print(f"Stage 1 validation samples: {len(X_stage1_val)}")
    print(f"Stage 2 will use: {n_train_val - n_stage1} samples")
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    
    rb = PCAReducerMultiChannel().fit(X_stage1_train, Y_stage1_train)
    if args.k_in is None or args.k_out is None:
        k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    else:
        k_in, k_out = args.k_in, args.k_out
    print(f"Using k_in={k_in}, k_out={k_out}")
    
    print("\n" + "="*70)
    print("CONVERTING TO PCA COEFFICIENTS")
    print("="*70)
    
    X_stage1_train_pca = spatial_to_pca_coefficients(X_stage1_train, rb, k_in)
    Y_stage1_train_pca = spatial_to_pca_coefficients_output(Y_stage1_train, rb, k_out)
    X_stage1_val_pca = spatial_to_pca_coefficients(X_stage1_val, rb, k_in)
    Y_stage1_val_pca = spatial_to_pca_coefficients_output(Y_stage1_val, rb, k_out)
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage1_train_pca).float(),
                     torch.from_numpy(Y_stage1_train_pca).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage1_val_pca).float(),
                     torch.from_numpy(Y_stage1_val_pca).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    print("\n" + "="*70)
    print("STAGE 1: Training in PCA coefficient space")
    print("="*70)
    
    model = MRBNOStage1(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_stage1(model, train_loader, val_loader, args.epochs, args.lr, device, checkpoint_dir, start_epoch=0)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'stage1_final.pt')
    import pickle
    rb_data = {
        'pca_in': rb.pca_in,
        'pca_out': rb.pca_out,
        'n_channels': rb.n_channels,
        'out_channels': rb.out_channels,
        'shape_': rb.shape_
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'rb_data': rb_data,
        'k_in': k_in,
        'k_out': k_out,
        'width': args.width,
        'layers': args.layers,
        'epoch': args.epochs
    }, checkpoint_path)
    print(f"\nStage 1 checkpoint saved to {checkpoint_path}")
    
    resume_path = os.path.join(checkpoint_dir, 'stage1_resume.pt')
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == '__main__':
    main()
