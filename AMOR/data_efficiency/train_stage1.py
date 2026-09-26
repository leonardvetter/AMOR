#!/usr/bin/env python3
"""
Stage 1 training: Train in PCA coefficient space on 90% of train/val data.
PCA is computed on ALL train/val data (including Stage 2 data) to ensure
identical PCA basis with baseline for fair weight comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from train_two_stage import (
    load_nonlinear_dataset, PCAReducerMultiChannel, AMORStage1,
    train_stage1, spatial_to_pca_coefficients, spatial_to_pca_coefficients_output,
    _atomic_torch_save,
)


def main():
    parser = argparse.ArgumentParser(description='Stage 1 training')
    parser.add_argument('--data_path', type=str, default=expand('${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz'))  # rename path on a different cluster
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--k_in', type=int, default=None)
    parser.add_argument('--k_out', type=int, default=None)
    parser.add_argument('--j_in', type=float, default=0.99,
                       help='PCA variance cutoff for input (default: 0.99)')
    parser.add_argument('--j_out', type=float, default=0.99,
                       help='PCA variance cutoff for output (default: 0.99)')
    parser.add_argument('--stage1_ratio', type=float, default=0.9,
                       help='Reserved for Stage 2 slicing; Stage 1 now always uses all train/val data.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help=expand('Directory to save checkpoints (default: ${AMOR_DATA_ROOT}/data_efficiency/checkpoints)'))
    parser.add_argument('--checkpoint_name', type=str, default='stage1_final.pt',
                       help='Name for final checkpoint file (default: stage1_final.pt)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--train_loss',
        type=str,
        choices=('l1', 'mse'),
        default='l1',
        help="Optimization objective for Stage 1 (default: l1). Best checkpoint always uses val_rel_l1.",
    )
    args = parser.parse_args()
    
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = expand('${AMOR_DATA_ROOT}/data_efficiency/checkpoints')
    log_dir = expand('${AMOR_DATA_ROOT}/data_efficiency/logs')
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
    print("STAGE 1 DATA SPLIT: 100% of train/val for Stage 1")
    print("="*70)
    
    n_train_val = len(X_train) + len(X_val)
    n_stage1 = n_train_val
    n_stage1_train = int(0.7 * n_stage1)
    
    X_stage1_all = np.concatenate([X_train, X_val], axis=0)
    Y_stage1_all = np.concatenate([Y_train, Y_val], axis=0)
    
    X_stage1_train = X_stage1_all[:n_stage1_train]
    Y_stage1_train = Y_stage1_all[:n_stage1_train]
    X_stage1_val = X_stage1_all[n_stage1_train:n_stage1]
    Y_stage1_val = Y_stage1_all[n_stage1_train:n_stage1]
    
    print(f"Stage 1 training samples: {len(X_stage1_train)}")
    print(f"Stage 1 validation samples: {len(X_stage1_val)}")
    n_stage2_reserved = n_train_val - int(args.stage1_ratio * n_train_val)
    print(f"Stage 2 reserved slice (from stage1_ratio): {n_stage2_reserved} samples")
    
    print("\n" + "="*70)
    print("COMPUTING PCA ON ALL TRAIN/VAL DATA")
    print("="*70)
    print("Note: PCA and Stage 1 training both use all non-test data")
    
    X_all_train_val = np.concatenate([X_train, X_val], axis=0)
    Y_all_train_val = np.concatenate([Y_train, Y_val], axis=0)
    rb = PCAReducerMultiChannel().fit(X_all_train_val, Y_all_train_val)
    if args.k_in is None or args.k_out is None:
        k_in, k_out = rb.k_from_cutoff(j_in=args.j_in, j_out=args.j_out)
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
    
    model = AMORStage1(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_stage1(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
        checkpoint_dir,
        start_epoch=0,
        train_loss=args.train_loss,
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_name)
    import pickle
    rb_data = {
        'pca_in': rb.pca_in,
        'pca_out': rb.pca_out,
        'n_channels': rb.n_channels,
        'out_channels': rb.out_channels,
        'shape_': rb.shape_
    }
    _atomic_torch_save({
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
