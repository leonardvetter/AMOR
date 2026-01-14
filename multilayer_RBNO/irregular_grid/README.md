# Irregular Grid MRBNO

This directory contains implementations and experiments for MRBNO on irregular grids and arbitrary geometries.

## Overview

The `MRBNO_lift_irregular.py` model adapts MRBNO for point clouds and irregular meshes. Unlike the regular grid version (`MRBNO_lift_simple`), this model:

- Uses 1D convolutions (`Conv1d`) instead of 2D (`Conv2d`)
- Operates on point clouds with arbitrary node positions
- Assumes constant mesh topology (same node positions across all samples)
- Supports material-adaptive PCA basis transformation
- Supports material property conditioning via multi-resolution positional encoding

## Key Files

- **`MRBNO_lift_irregular.py`**: Main model architecture for irregular grids
- **`train_geofno.py`**: Training script for Geo-FNO datasets (elasticity, airfoil)
- **`geofno_loader.py`**: Data loader for Geo-FNO datasets
- **`pca_utils.py`**: Utilities for computing PCA on point cloud data
- **`data_utils.py`**: Data preprocessing utilities
- **`run_all_experiments.py`**: Script to run all experiments sequentially
- **`compute_metrics.py`**: Metrics computation for evaluation
- **`plot_predictions.py`**: Visualization utilities
- **`monitor_training.py`**: Training monitoring utilities

## Datasets

Supports Geo-FNO datasets:
- Elasticity (Darcy flow on irregular domains)
- Airfoil (flow around airfoils)

## Model Architecture

The `MRBNOLiftIrregular` model:
- Uses a single PCA basis (computed on the point cloud data)
- Supports coordinate concatenation (`add_coords`)
- Supports grid-based skip connections (`use_grid_skip`) similar to Geo-FNO
- Supports direct input-to-output skip connections (`use_input_skip`)
- Can use material-adaptive PCA basis transformation

## Usage

```bash
# Train on elasticity dataset
python train_geofno.py --dataset elasticity --epochs 300

# Train on airfoil dataset
python train_geofno.py --dataset airfoil --epochs 300

# Run all experiments
python run_all_experiments.py
```

## Differences from MRBNO_lift_simple

1. **Spatial dimensions**: 1D (point clouds) vs 2D (regular grids)
2. **Convolution type**: `Conv1d` vs `Conv2d`
3. **Mesh handling**: Supports arbitrary node positions vs fixed grid
4. **Material conditioning**: Optional material property conditioning
5. **PCA basis**: Single basis (same as lift_simple), but computed on point cloud data

