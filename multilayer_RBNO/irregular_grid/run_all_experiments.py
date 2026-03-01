#!/usr/bin/env python3
"""
Run visualization and training experiments sequentially.

This script runs:
1. Visualization of datasets
2. Training on elasticity dataset
3. Training on airfoil dataset

All output is logged to files for monitoring.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description, log_file):
    """Run a command and log output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*60}\n\n")
        f.flush()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            return_code = process.returncode
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Finished at: {datetime.now()}\n")
            f.write(f"Return code: {return_code}\n")
            
            if return_code != 0:
                print(f"\nERROR: {description} failed with return code {return_code}")
                return False
            else:
                print(f"\nSUCCESS: {description} completed")
                return True
                
        except Exception as e:
            error_msg = f"Exception running {description}: {str(e)}\n"
            print(error_msg)
            f.write(error_msg)
            return False


def main():
    base_dir = "/global/homes/d/dvod/multilayer_RBNO/multilayer_RBNO/irregular_grid"
    data_base_dir = '/mnt/c/Users/dvod/MRBNO/data'
    log_dir = os.path.join(data_base_dir, 'logs', 'geofno_experiments')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("MRBNO Geo-FNO Experiments")
    print("="*60)
    print(f"Log directory: {log_dir}")
    print(f"Timestamp: {timestamp}")
    print("="*60)
    
    results = {}
    
    # 1. Visualization
    vis_log = os.path.join(log_dir, f"visualization_{timestamp}.log")
    cmd = [sys.executable, os.path.join(base_dir, "visualize_geofno_datasets.py")]
    results['visualization'] = run_command(cmd, "Dataset Visualization", vis_log)
    
    if not results['visualization']:
        print("\nWARNING: Visualization failed, but continuing with training...")
    
    # 2. Elasticity training
    elasticity_log = os.path.join(log_dir, f"training_elasticity_{timestamp}.log")
    elasticity_data_dir = os.path.join(data_base_dir, 'Geo-FNO', 'data', 'diffeo')
    cmd = [
        sys.executable,
        os.path.join(base_dir, "train_geofno.py"),
        "--dataset_path", elasticity_data_dir,
        "--dataset_type", "elasticity",
        "--n_modes", "144",
        "--width", "24",
        "--n_layers", "4",
        "--batch_size", "32",
        "--epochs", "300",
        "--lr", "2e-3"
    ]
    results['elasticity'] = run_command(cmd, "Elasticity Dataset Training", elasticity_log)
    
    # 3. Airfoil training
    airfoil_log = os.path.join(log_dir, f"training_airfoil_{timestamp}.log")
    airfoil_data_dir = os.path.join(scratch_dir, 'data', 'Geo-FNO', 'data', 'naca')
    cmd = [
        sys.executable,
        os.path.join(base_dir, "train_geofno.py"),
        "--dataset_path", airfoil_data_dir,
        "--dataset_type", "airfoil",
        "--n_modes", "144",
        "--width", "24",
        "--n_layers", "4",
        "--batch_size", "32",
        "--epochs", "300",
        "--lr", "2e-3"
    ]
    results['airfoil'] = run_command(cmd, "Airfoil Dataset Training", airfoil_log)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name:20s}: {status}")
    print("="*60)
    print(f"\nAll logs saved to: {log_dir}")


if __name__ == "__main__":
    main()

