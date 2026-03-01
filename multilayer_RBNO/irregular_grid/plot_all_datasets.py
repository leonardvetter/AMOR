"""
Generate plots for all trained MRBNO models on Geo-FNO datasets.

Creates plots showing ground truth, prediction, and grid for all datasets.
"""

import os
import sys
import subprocess
import shutil

DATA_BASE_DIR = '/mnt/c/Users/dvod/MRBNO/data'
CHECKPOINT_DIR = os.path.join(DATA_BASE_DIR, 'checkpoints/geofno')
OUTPUT_DIR = os.path.join(DATA_BASE_DIR, 'geofno_predictions')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
PLOT_SCRIPT = os.path.join(SCRIPT_DIR, 'plot_predictions.py')

DATASETS = [
    'elasticity',
    'airfoil',
    'pipe',
    'plasticity',
    'car-cfd',
    'channel-shocks',
]

def generate_plots():
    """Generate plots for all datasets. Save to both $SCRATCH and local figures/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 80)
    print("Generating Plots for All Datasets")
    print("=" * 80)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Output: {OUTPUT_DIR} and {FIGURES_DIR}")
    print()
    for dataset_type in DATASETS:
        checkpoint_name = f'checkpoint_best_{dataset_type}.pt'
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Skipping {dataset_type}: checkpoint not found at {checkpoint_path}")
            continue
        print(f"Generating plot for {dataset_type}...")
        cmd = [
            'python3', PLOT_SCRIPT,
            '--dataset_type', dataset_type,
            '--checkpoint_path', checkpoint_path,
            '--sample_idx', '0',
            '--output_dir', OUTPUT_DIR,
            '--device', 'cpu',
        ]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ✗ Error: {result.stderr[:200]}")
            continue
        print(f"  ✓ Plot generated for {dataset_type}")
        fname = f"{dataset_type.replace('-', '_')}_prediction_sample_0.png"
        if dataset_type == 'channel-shocks':
            fname = "channel_shocks_prediction_sample_0.png"
        src = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(FIGURES_DIR, fname))
    print("\n" + "=" * 80)
    print("Plot generation complete")
    print("=" * 80)
    print(f"Plots saved to: {OUTPUT_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    generate_plots()
