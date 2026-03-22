#!/usr/bin/env python3
"""
Ablation study: Vary PCA epsilon (cutoff threshold) from 1e-1 to 1e-5.
Uses Stage 2 v1 (frozen blocks, same PCA weights as Stage 1).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description='PCA epsilon ablation study')
    parser.add_argument('--data_path', type=str, default='/users/PAS3244/dvoderhol/data/nonlinear_diffusion_dataset_big.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    base_dir = '/users/PAS3244/dvoderhol/data/data_efficiency'
    log_dir = os.path.join(base_dir, 'logs', 'ablation_pca_epsilon')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'ablation_pca_epsilon')
    master_log = os.path.join(base_dir, 'logs', 'ablation_pca_epsilon_master.log')
    
    for d in [log_dir, checkpoint_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_cmd = os.environ.get('ABLATION_PYTHON', 'python3')
    
    with open(master_log, 'w') as mlog:
        mlog.write("PCA EPSILON ABLATION (Stage 2 v1 - frozen blocks)\n")
        mlog.write("="*70 + "\n")
        mlog.write(f"Epsilons: {epsilons}\n")
        mlog.write(f"Log dir: {log_dir}\n")
        mlog.write(f"Checkpoint dir: {checkpoint_dir}\n")
        mlog.flush()
    
    print("="*70)
    print("PCA EPSILON ABLATION STUDY (Stage 2 v1)")
    print("="*70)
    print(f"Epsilons: {epsilons}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("Old ablation results deleted. Running sequentially...")
    
    for eps in epsilons:
        j_in = 1.0 - eps
        j_out = 1.0 - eps
        eps_str = f"{eps:.0e}".replace('-0', '-')
        eps_file = eps_str.replace('.', '_').replace('-', 'm')
        stage1_ckpt = os.path.join(checkpoint_dir, f'stage1_eps{eps_file}_final.pt')
        stage1_log = os.path.join(log_dir, f'stage1_eps{eps_str}.log')
        stage2_log = os.path.join(log_dir, f'stage2_v1_eps{eps_str}.log')
        stage2_ckpt_dir = os.path.join(checkpoint_dir, f'stage2_eps{eps_file}')
        os.makedirs(stage2_ckpt_dir, exist_ok=True)
        
        print(f"\nEpsilon {eps:.0e}: Stage 1...")
        ret1 = subprocess.run([
            python_cmd, '-u', os.path.join(script_dir, 'train_stage1.py'),
            '--data_path', args.data_path,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--j_in', str(j_in), '--j_out', str(j_out),
            '--checkpoint_dir', checkpoint_dir,
            '--checkpoint_name', f'stage1_eps{eps_file}_final.pt',
        ], cwd=script_dir, stdout=open(stage1_log, 'w'), stderr=subprocess.STDOUT)
        
        if ret1.returncode != 0:
            with open(master_log, 'a') as mlog:
                mlog.write(f"FAILED: Stage 1 eps={eps:.0e}\n")
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
        ], cwd=script_dir, stdout=open(stage2_log, 'w'), stderr=subprocess.STDOUT)
        
        if ret2.returncode != 0:
            with open(master_log, 'a') as mlog:
                mlog.write(f"FAILED: Stage 2 v1 eps={eps:.0e}\n")
            print(f"  Stage 2 v1 FAILED (see {stage2_log})")
        else:
            with open(master_log, 'a') as mlog:
                mlog.write(f"OK: eps={eps:.0e}\n")
            print(f"  Stage 2 v1 done.")
    
    print("\n" + "="*70)
    print("PCA EPSILON ABLATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
