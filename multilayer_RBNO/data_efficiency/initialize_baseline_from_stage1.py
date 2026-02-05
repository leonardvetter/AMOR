#!/usr/bin/env python3
"""
Initialize baseline MRBNO model with PCA space weights from Stage 1 training.

This allows the baseline to start with the same PCA space weights as Stage 1,
enabling fair comparison of weight evolution during training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import torch
import numpy as np

from train_two_stage import PCAReducerMultiChannel, MRBNOStage1
from MRBNO import MRBNO


def initialize_baseline_from_stage1(baseline_model, stage1_model):
    """
    Copy PCA space weights (blocks and gates) from Stage 1 model to Baseline model.
    
    Args:
        baseline_model: MRBNO model to initialize
        stage1_model: MRBNOStage1 model with trained weights
    """
    stage1_state = stage1_model.state_dict()
    baseline_state = baseline_model.state_dict()
    
    copied_weights = []
    for name, param in stage1_state.items():
        if name in baseline_state:
            if name.startswith('block') or name.startswith('gate'):
                baseline_state[name] = param.clone()
                copied_weights.append(name)
    
    baseline_model.load_state_dict(baseline_state, strict=False)
    
    print(f"Copied {len(copied_weights)} PCA space weights from Stage 1 to Baseline:")
    for name in copied_weights:
        print(f"  - {name}")
    
    return baseline_model


def main():
    parser = argparse.ArgumentParser(description='Initialize baseline from Stage 1')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--baseline_checkpoint_out', type=str, required=True,
                       help='Output path for initialized baseline checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to load models on (default: cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("INITIALIZING BASELINE FROM STAGE 1")
    print("="*70)
    
    print(f"\nLoading Stage 1 checkpoint: {args.stage1_checkpoint}")
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    
    rb_data = stage1_ckpt['rb_data']
    rb = PCAReducerMultiChannel()
    rb.pca_in = rb_data['pca_in']
    rb.pca_out = rb_data['pca_out']
    rb.n_channels = rb_data['n_channels']
    rb.out_channels = rb_data['out_channels']
    rb.shape_ = rb_data['shape_']
    
    k_in = stage1_ckpt['k_in']
    k_out = stage1_ckpt['k_out']
    width = stage1_ckpt['width']
    layers = stage1_ckpt['layers']
    
    print(f"Model config: width={width}, layers={layers}, k_in={k_in}, k_out={k_out}")
    
    print("\nCreating Stage 1 model...")
    stage1_model = MRBNOStage1(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False
    )
    stage1_model.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1_model.eval()
    
    print("Creating Baseline model...")
    baseline_model = MRBNO(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False
    )
    
    print("\nCopying PCA space weights from Stage 1 to Baseline...")
    initialize_baseline_from_stage1(baseline_model, stage1_model)
    
    print(f"\nSaving initialized baseline checkpoint: {args.baseline_checkpoint_out}")
    torch.save({
        'model_state_dict': baseline_model.state_dict(),
        'rb': rb,
        'k_in': k_in,
        'k_out': k_out,
        'width': width,
        'layers': layers,
        'initialized_from_stage1': True,
        'stage1_checkpoint': args.stage1_checkpoint
    }, args.baseline_checkpoint_out)
    
    print("\nDone! Baseline model initialized with Stage 1 PCA space weights.")
    print("Note: lift, proj, and global_skip weights remain randomly initialized.")


if __name__ == '__main__':
    main()
