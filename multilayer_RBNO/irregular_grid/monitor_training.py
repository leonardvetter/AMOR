#!/usr/bin/env python3
"""Monitor training progress by checking log files and checkpoints."""

import os
import time
import glob
from datetime import datetime

def monitor_training():
    data_dir = '/users/PAS3244/dvoderhol/data'
    log_dir = os.path.join(data_dir, 'logs', 'geofno_experiments')
    checkpoint_dir = os.path.join(data_dir, 'checkpoints', 'geofno')
    
    print("=" * 60)
    print("Training Progress Monitor")
    print("=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 60)
    
    # Check elasticity training
    elasticity_log = os.path.join(log_dir, 'training_elasticity.log')
    if os.path.exists(elasticity_log):
        print("\nElasticity Training Log:")
        print("-" * 60)
        with open(elasticity_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print("Last 20 lines:")
                for line in lines[-20:]:
                    print(line.rstrip())
            else:
                print("Log file is empty")
    else:
        print("\nElasticity log file not found yet")
    
    # Check airfoil training
    airfoil_log = os.path.join(log_dir, 'training_airfoil.log')
    if os.path.exists(airfoil_log):
        print("\n\nAirfoil Training Log:")
        print("-" * 60)
        with open(airfoil_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print("Last 20 lines:")
                for line in lines[-20:]:
                    print(line.rstrip())
            else:
                print("Log file is empty")
    else:
        print("\nAirfoil log file not found yet")
    
    # Check checkpoints
    print("\n\nCheckpoints:")
    print("-" * 60)
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            print(f"Found {len(checkpoints)} checkpoint(s):")
            for cp in checkpoints[:5]:
                mtime = datetime.fromtimestamp(os.path.getmtime(cp))
                size = os.path.getsize(cp) / (1024 * 1024)
                print(f"  {os.path.basename(cp)} - {mtime.strftime('%Y-%m-%d %H:%M:%S')} - {size:.2f} MB")
        else:
            print("No checkpoints found yet")
    else:
        print("Checkpoint directory does not exist yet")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    monitor_training()

