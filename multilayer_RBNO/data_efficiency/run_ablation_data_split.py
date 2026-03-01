#!/usr/bin/env python3
"""
Ablation study: Vary data split between Stage 1 and Stage 2.
Splits: 50/50, 70/30, 90/10, 99/1
Uses Stage 2 v1 (frozen blocks, same PCA weights as Stage 1).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description='Data split ablation study')
    parser.add_argument('--data_path', type=str, default='/mnt/c/Users/dvod/MRBNO/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    base_dir = '/mnt/c/Users/dvod/MRBNO/data/data_efficiency'
    log_dir = os.path.join(base_dir, 'logs', 'ablation_data_split')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'ablation_data_split')
    master_log = os.path.join(base_dir, 'logs', 'ablation_data_split_master.log')
    
    for d in [log_dir, checkpoint_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    splits = [
        (0.5, '50_50'),
        (0.7, '70_30'),
        (0.9, '90_10'),
        (0.99, '99_1')
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_cmd = os.environ.get('ABLATION_PYTHON', 'python3')
    
    with open(master_log, 'w') as mlog:
        mlog.write("DATA SPLIT ABLATION (Stage 2 v1 - frozen blocks)\n")
        mlog.write("="*70 + "\n")
        mlog.write(f"Splits: {[s[1] for s in splits]}\n")
        mlog.write(f"Log dir: {log_dir}\n")
        mlog.write(f"Checkpoint dir: {checkpoint_dir}\n")
        mlog.flush()
    
    print("="*70)
    print("DATA SPLIT ABLATION STUDY (Stage 2 v1)")
    print("="*70)
    print(f"Splits: {[f'{s[0]*100:.0f}/{100-s[0]*100:.0f}' for s in splits]}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("Old ablation results deleted. Running sequentially...")
    
    for stage1_ratio, split_name in splits:
        stage1_ckpt = os.path.join(checkpoint_dir, f'stage1_{split_name}_final.pt')
        stage1_log = os.path.join(log_dir, f'stage1_{split_name}.log')
        stage2_log = os.path.join(log_dir, f'stage2_v1_{split_name}.log')
        stage2_ckpt_dir = os.path.join(checkpoint_dir, f'stage2_{split_name}')
        os.makedirs(stage2_ckpt_dir, exist_ok=True)
        
        print(f"\nSplit {split_name}: Stage 1...")
        ret1 = subprocess.run([
            python_cmd, '-u', os.path.join(script_dir, 'train_stage1.py'),
            '--data_path', args.data_path,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--stage1_ratio', str(stage1_ratio),
            '--checkpoint_dir', checkpoint_dir,
            '--checkpoint_name', f'stage1_{split_name}_final.pt',
        ], cwd=script_dir, stdout=open(stage1_log, 'w'), stderr=subprocess.STDOUT)
        
        if ret1.returncode != 0:
            with open(master_log, 'a') as mlog:
                mlog.write(f"FAILED: Stage 1 {split_name}\n")
            print(f"  Stage 1 FAILED (see {stage1_log})")
            continue
        
        print(f"  Stage 1 done. Stage 2 v1...")
        ret2 = subprocess.run([
            python_cmd, '-u', os.path.join(script_dir, 'train_stage2.py'),
            '--data_path', args.data_path,
            '--stage1_checkpoint', stage1_ckpt,
            '--checkpoint_dir', stage2_ckpt_dir,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--version', '1',
            '--stage1_ratio', str(stage1_ratio),
        ], cwd=script_dir, stdout=open(stage2_log, 'w'), stderr=subprocess.STDOUT)
        
        if ret2.returncode != 0:
            with open(master_log, 'a') as mlog:
                mlog.write(f"FAILED: Stage 2 v1 {split_name}\n")
            print(f"  Stage 2 v1 FAILED (see {stage2_log})")
        else:
            with open(master_log, 'a') as mlog:
                mlog.write(f"OK: {split_name}\n")
            print(f"  Stage 2 v1 done.")
    
    print("\n" + "="*70)
    print("DATA SPLIT ABLATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
