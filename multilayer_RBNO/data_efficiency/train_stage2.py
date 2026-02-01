#!/usr/bin/env python3
"""
Stage 2 training: Fine-tune complete MRBNO.py model on 10% of train/val data.
Three versions:
  - Version 1: Only new weights trained (frozen blocks)
  - Version 2: All weights trained, higher LR for new weights
  - Version 3: All weights trained equally
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
    train_stage2, evaluate_model
)


def main():
    parser = argparse.ArgumentParser(description='Stage 2 training')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_blocks_factor', type=float, default=0.1,
                       help='LR multiplier for blocks (Version 2). Use 1.0 for Version 3.')
    parser.add_argument('--version', type=int, choices=[1, 2, 3], default=1,
                       help='Version: 1=frozen blocks, 2=different LR, 3=equal LR')
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
    
    checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    
    print("\n" + "="*70)
    print("STAGE 2 DATA SPLIT: 10% of train/val for Stage 2")
    print("="*70)
    
    n_train_val = len(X_train) + len(X_val)
    n_stage1 = int(0.9 * n_train_val)
    n_stage2 = n_train_val - n_stage1
    n_stage2_train = int(0.7 * n_stage2)
    
    X_stage2_all = np.concatenate([X_train, X_val], axis=0)[n_stage1:]
    Y_stage2_all = np.concatenate([Y_train, Y_val], axis=0)[n_stage1:]
    
    X_stage2_train = X_stage2_all[:n_stage2_train]
    Y_stage2_train = Y_stage2_all[:n_stage2_train]
    X_stage2_val = X_stage2_all[n_stage2_train:]
    Y_stage2_val = Y_stage2_all[n_stage2_train:]
    
    print(f"Stage 2 training samples: {len(X_stage2_train)}")
    print(f"Stage 2 validation samples: {len(X_stage2_val)}")
    
    rb_data = checkpoint['rb_data']
    rb = PCAReducerMultiChannel()
    rb.pca_in = rb_data['pca_in']
    rb.pca_out = rb_data['pca_out']
    rb.n_channels = rb_data['n_channels']
    rb.out_channels = rb_data['out_channels']
    rb.shape_ = rb_data['shape_']
    k_in = checkpoint['k_in']
    k_out = checkpoint['k_out']
    width = checkpoint['width']
    layers = checkpoint['layers']
    
    print("\n" + "="*70)
    print(f"STAGE 2 VERSION {args.version}: Fine-tuning complete MRBNO")
    print("="*70)
    
    from MRBNO import MRBNO
    
    stage1_model = MRBNOStage1(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False
    )
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    
    freeze_blocks = (args.version == 1)
    lr_blocks_factor = args.lr_blocks_factor if args.version == 2 else 1.0
    
    stage2_base = MRBNO(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False
    )
    
    stage1_state = stage1_model.state_dict()
    stage2_state = stage2_base.state_dict()
    
    for name, param in stage1_state.items():
        if name in stage2_state:
            if name.startswith('block') or name.startswith('gate'):
                stage2_state[name] = param
    
    stage2_base.load_state_dict(stage2_state, strict=False)
    
    if freeze_blocks:
        for param in stage2_base.block0.parameters():
            param.requires_grad = False
        for param in stage2_base.block1.parameters():
            param.requires_grad = False
        for block in stage2_base.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in stage2_base.gate0.parameters():
            param.requires_grad = False
        for param in stage2_base.gate1.parameters():
            param.requires_grad = False
        for gate in stage2_base.gates:
            for param in gate.parameters():
                param.requires_grad = False
    
    model = stage2_base
    
    version_name = f"v{args.version}"
    if args.version == 1:
        print("Version 1: Only new weights trained (blocks frozen)")
    elif args.version == 2:
        print(f"Version 2: All weights trained, blocks LR = {args.lr * lr_blocks_factor:.2e} (factor {lr_blocks_factor})")
    else:
        print("Version 3: All weights trained equally")
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage2_train).float(),
                     torch.from_numpy(Y_stage2_train).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage2_val).float(),
                     torch.from_numpy(Y_stage2_val).float()),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                     torch.from_numpy(Y_test).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    train_stage2(model, train_loader, val_loader, args.epochs, args.lr, lr_blocks_factor,
                 device, checkpoint_dir, version_name)
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    test_rel_l1, test_rel_l2 = evaluate_model(model, test_loader, device, is_stage1=False)
    print(f"Test relative L1 error: {test_rel_l1:.4e}")
    print(f"Test relative L2 error: {test_rel_l2:.4e}")


if __name__ == '__main__':
    main()
