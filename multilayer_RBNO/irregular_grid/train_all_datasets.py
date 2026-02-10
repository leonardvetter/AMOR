"""
Train MRBNO on all Geo-FNO datasets.

Runs training for all datasets: elasticity, airfoil, pipe, plasticity, car-cfd, channel-shocks.
"""

import os
import sys
import subprocess
import time

SCRATCH_DIR = os.environ.get('SCRATCH', '/pscratch/sd/dvod')
DATA_DIR = os.path.join(SCRATCH_DIR, 'data/Geo-FNO/data')

# Dataset configurations
DATASETS = [
    {
        'name': 'elasticity',
        'path': os.path.join(DATA_DIR, 'diffeo'),
        'type': 'elasticity',
    },
    {
        'name': 'airfoil',
        'path': os.path.join(DATA_DIR, 'naca'),
        'type': 'airfoil',
    },
    {
        'name': 'pipe',
        'path': os.path.join(DATA_DIR, 'pipe'),
        'type': 'pipe',
    },
    {
        'name': 'plasticity',
        'path': os.path.join(DATA_DIR, 'plasticity'),
        'type': 'plasticity',
    },
    {
        'name': 'car-cfd',
        'path': os.path.join(DATA_DIR, 'car-cfd'),
        'type': 'car-cfd',
    },
    {
        'name': 'channel-shocks',
        'path': os.path.join(DATA_DIR, 'channel-shocks/channel'),
        'type': 'channel-shocks',
    },
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train_geofno.py')

def run_training(dataset_config, background=True):
    """Run training for a dataset."""
    name = dataset_config['name']
    path = dataset_config['path']
    dataset_type = dataset_config['type']
    
    print("=" * 80)
    print(f"Starting training for {name}")
    print("=" * 80)
    print(f"Dataset path: {path}")
    print(f"Dataset type: {dataset_type}")
    
    if not os.path.exists(path):
        print(f"ERROR: Dataset path does not exist: {path}")
        return None
    
    log_file = os.path.join(SCRIPT_DIR, f'training_{name}.log')
    
    cmd = [
        'python3', TRAIN_SCRIPT,
        '--dataset_path', path,
        '--dataset_type', dataset_type,
        '--n_modes', '144',
        '--width', '24',
        '--n_layers', '4',
        '--batch_size', '32',
        '--epochs', '300',
        '--lr', '2e-3',
        '--weight_decay', '1e-4',
        '--max_grad_norm', '3.0',
        '--patience', '50',
        '--min_delta', '0.0',
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    if background:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=SCRIPT_DIR
            )
        print(f"Started training in background (PID: {process.pid})")
        print(f"Monitor with: tail -f {log_file}")
        return process
    else:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        return result


def main():
    """Run training for all datasets."""
    print("=" * 80)
    print("MRBNO Training on All Geo-FNO Datasets")
    print("=" * 80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Script directory: {SCRIPT_DIR}")
    print()
    
    processes = []
    
    for dataset_config in DATASETS:
        process = run_training(dataset_config, background=True)
        if process:
            processes.append((dataset_config['name'], process))
        time.sleep(2)  # Small delay between starting processes
    
    print("\n" + "=" * 80)
    print("All training processes started")
    print("=" * 80)
    print(f"Started {len(processes)} training jobs:")
    for name, process in processes:
        print(f"  {name}: PID {process.pid}")
    print("\nMonitor training progress with:")
    for name, _ in processes:
        print(f"  tail -f {os.path.join(SCRIPT_DIR, f'training_{name}.log')}")


if __name__ == "__main__":
    main()
