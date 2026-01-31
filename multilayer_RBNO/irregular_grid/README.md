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
- **Elasticity**: Darcy flow on irregular domains with varying material properties
- **Airfoil**: Flow around airfoils with varying geometries

### Dataset Sources

The datasets are available from:
- **Google Drive**: https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8
- **GitHub Repository**: https://github.com/neuraloperator/Geo-FNO

### Dataset Details

**Elasticity Dataset**:
- Input: Material property field (permeability) on irregular mesh
- Output: Solution field (pressure/velocity) on the same mesh
- Mesh: Irregular triangular mesh with varying node positions
- Split: First 1000 samples for training, last 200 for testing (Geo-FNO standard)

**Airfoil Dataset**:
- Input: Flow field channels 0-3 (Q field) + X and Y coordinates
- Output: Flow field channel 4 (Q field)
- Mesh: Regular grid extracted as point cloud
- Split: First 1000 samples for training, next 200 for testing (Geo-FNO standard)

### Downloading Datasets

Use the provided scripts to download and prepare the datasets:
```bash
# Download datasets from Google Drive or clone Geo-FNO repository
python download_geofno_datasets.py

# Or use the shell script
./download_geofno_from_drive.sh
```

The datasets should be placed in `$SCRATCH/data/Geo-FNO/data/` with the following structure:
- `diffeo/`: Elasticity dataset files
- `naca/`: Airfoil dataset files

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

