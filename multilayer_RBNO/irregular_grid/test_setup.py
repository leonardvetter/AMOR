#!/usr/bin/env python3
"""Quick test to verify all imports and basic functionality."""

import sys
import os

print("Testing imports and setup...")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ torch: {e}")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib: {e}")
    sys.exit(1)

try:
    from geofno_loader import load_geofno_dataset
    print("✓ geofno_loader")
except ImportError as e:
    print(f"✗ geofno_loader: {e}")
    sys.exit(1)

try:
    from data_utils import compute_pca_for_model, IrregularGridDataset
    print("✓ data_utils")
except ImportError as e:
    print(f"✗ data_utils: {e}")
    sys.exit(1)

try:
    from MRBNO_lift_irregular import MRBNOLiftIrregular
    print("✓ MRBNO_lift_irregular")
except ImportError as e:
    print(f"✗ MRBNO_lift_irregular: {e}")
    sys.exit(1)

try:
    from losses import LpLoss
    print("✓ losses")
except ImportError as e:
    print(f"✗ losses: {e}")
    sys.exit(1)

# Check data directories
scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
elasticity_dir = os.path.join(scratch_dir, 'data', 'Geo-FNO', 'data', 'diffeo')
airfoil_dir = os.path.join(scratch_dir, 'data', 'Geo-FNO', 'data', 'naca')

print("\nChecking data directories...")
if os.path.exists(elasticity_dir):
    files = os.listdir(elasticity_dir)
    print(f"✓ Elasticity directory exists: {elasticity_dir}")
    print(f"  Files: {files}")
else:
    print(f"✗ Elasticity directory not found: {elasticity_dir}")

if os.path.exists(airfoil_dir):
    files = os.listdir(airfoil_dir)
    print(f"✓ Airfoil directory exists: {airfoil_dir}")
    print(f"  Files: {files}")
else:
    print(f"✗ Airfoil directory not found: {airfoil_dir}")

print("\n✓ All basic checks passed!")
print("\nYou can now run:")
print("  python visualize_geofno_datasets.py")
print("  python train_geofno.py --dataset_path <path> --dataset_type <type> [options]")
print("  python run_all_experiments.py")

