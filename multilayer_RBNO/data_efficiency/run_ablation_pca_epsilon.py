#!/usr/bin/env python3
"""
Ablation study: Vary PCA epsilon (cutoff threshold) from 1e-1 to 1e-5.
Uses Stage 2 v3 (equal LR) with 90/10 data split.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description='PCA epsilon ablation study')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--stage1_checkpoint_base', type=str, 
                       default='$SCRATCH/data_efficiency/checkpoints/stage1_final.pt',
                       help='Base path for stage1 checkpoints (will append _eps{epsilon})')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    args.stage1_checkpoint_base = os.path.expandvars(args.stage1_checkpoint_base)
    
    scratch_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'data_efficiency')
    log_dir = os.path.join(scratch_dir, 'logs', 'ablation_pca_epsilon')
    os.makedirs(log_dir, exist_ok=True)
    
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    print("="*70)
    print("PCA EPSILON ABLATION STUDY")
    print("="*70)
    print(f"Epsilons to test: {epsilons}")
    print(f"Log directory: {log_dir}")
    print("\nNote: This script sets up the ablation study.")
    print("Stage 1 training needs to be run with different epsilons first.")
    print("Then Stage 2 v3 training will be run for each epsilon.")
    print("="*70)
    
    for eps in epsilons:
        j_in = 1.0 - eps
        j_out = 1.0 - eps
        stage1_checkpoint = f"{args.stage1_checkpoint_base}_eps{eps:.0e}"
        log_file = os.path.join(log_dir, f'stage2_v3_eps{eps:.0e}.log')
        
        print(f"\nEpsilon: {eps:.0e} (j_in={j_in:.6f}, j_out={j_out:.6f})")
        print(f"  Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"  Log file: {log_file}")
        print(f"  Command:")
        print(f"    python3 train_stage2.py \\")
        print(f"      --data_path {args.data_path} \\")
        print(f"      --stage1_checkpoint {stage1_checkpoint} \\")
        print(f"      --batch_size {args.batch_size} \\")
        print(f"      --epochs {args.epochs} \\")
        print(f"      --lr {args.lr} \\")
        print(f"      --version 3 \\")
        print(f"      > {log_file} 2>&1")


if __name__ == '__main__':
    main()
