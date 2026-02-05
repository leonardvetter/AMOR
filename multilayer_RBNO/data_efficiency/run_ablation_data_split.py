#!/usr/bin/env python3
"""
Ablation study: Vary data split between Stage 1 and Stage 2.
Splits: 50/50, 70/30, 90/10, 99/1
Uses Stage 2 v3 (equal LR) with default PCA epsilon (0.99).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description='Data split ablation study')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--k_in', type=int, default=144)
    parser.add_argument('--k_out', type=int, default=144)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    args.data_path = os.path.expandvars(args.data_path)
    
    scratch_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'data_efficiency')
    log_dir = os.path.join(scratch_dir, 'logs', 'ablation_data_split')
    checkpoint_dir = os.path.join(scratch_dir, 'checkpoints', 'ablation_data_split')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    splits = [
        (0.5, 0.5, '50_50'),
        (0.7, 0.3, '70_30'),
        (0.9, 0.1, '90_10'),
        (0.99, 0.01, '99_1')
    ]
    
    print("="*70)
    print("DATA SPLIT ABLATION STUDY")
    print("="*70)
    print(f"Splits to test: {[f'{s[0]*100:.0f}/{s[1]*100:.0f}' for s in splits]}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("\nNote: This script sets up the ablation study.")
    print("Stage 1 and Stage 2 training will be run for each split.")
    print("="*70)
    
    for stage1_ratio, stage2_ratio, split_name in splits:
        stage1_log = os.path.join(log_dir, f'stage1_{split_name}.log')
        stage1_checkpoint = os.path.join(checkpoint_dir, f'stage1_{split_name}_final.pt')
        stage2_log = os.path.join(log_dir, f'stage2_v3_{split_name}.log')
        
        print(f"\nSplit: {split_name} (Stage 1: {stage1_ratio*100:.0f}%, Stage 2: {stage2_ratio*100:.0f}%)")
        print(f"  Stage 1 log: {stage1_log}")
        print(f"  Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"  Stage 2 log: {stage2_log}")
        print(f"  Commands:")
        print(f"    # Stage 1:")
        print(f"    python3 train_stage1.py \\")
        print(f"      --data_path {args.data_path} \\")
        print(f"      --batch_size {args.batch_size} \\")
        print(f"      --epochs {args.epochs} \\")
        print(f"      --lr {args.lr} \\")
        print(f"      --width {args.width} \\")
        print(f"      --layers {args.layers} \\")
        print(f"      --k_in {args.k_in} \\")
        print(f"      --k_out {args.k_out} \\")
        print(f"      --stage1_ratio {stage1_ratio} \\")
        print(f"      > {stage1_log} 2>&1")
        print(f"    # Stage 2:")
        print(f"    python3 train_stage2.py \\")
        print(f"      --data_path {args.data_path} \\")
        print(f"      --stage1_checkpoint {stage1_checkpoint} \\")
        print(f"      --batch_size {args.batch_size} \\")
        print(f"      --epochs {args.epochs} \\")
        print(f"      --lr {args.lr} \\")
        print(f"      --version 3 \\")
        print(f"      --stage1_ratio {stage1_ratio} \\")
        print(f"      > {stage2_log} 2>&1")


if __name__ == '__main__':
    main()
