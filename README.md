# Multi-layer Reduced Basis Neural Operator (MRBNO)

Repository for training and evaluating Multi-layer Reduced Basis Neural Operators (MRBNO) and related models on PDE datasets.

## Overview

This repository implements several neural operator architectures:
- **FNO** (Fourier Neural Operator): Uses Fourier transforms for spectral operations
- **MRBNO Lift Simple**: Uses PCA basis instead of Fourier basis, with same block structure as FNO
- **MRBNO Lift v3/v5**: More complex MRBNO variants with separate input/output PCA bases
- **PCANet**: Simple neural network operating in PCA coefficient space

## Models

### MRBNO Lift Simple

- **Input channels**: Handles arbitrary number via lifting layer `fc0` (Conv2d with `in_channels + 2*add_coords` input channels)
- **Architecture**: Same structure as FNO2d (lifting → blocks → projection)
- **PCA basis**: Single PCA basis shared for both encoding and decoding across all blocks, processes hidden representation with spatial dimensions (H, W)
- **Key difference from FNO**: Uses PCA encoding/decoding instead of FFT/IFFT in spectral convolution blocks
- **Note**: Uses the same PCA basis for input and output transformations (unlike MRBNO v3/v5 which use separate input/output bases)

### FNO

- **Input channels**: Handles arbitrary number via lifting layer (Conv2d)
- **Architecture**: Lifting → FNO blocks → projection
- **Spectral operations**: FFT → multiply in frequency domain → IFFT
- **Modes**: Tuple (modes_x, modes_y) for low-frequency modes

### MRBNO v3/v5

- **Input channels**: Handles arbitrary number via encoder/decoder structure
- **Architecture**: Separate input and output PCA bases, more flexible basis selection
- **Key difference**: Can use input basis for encoder and output basis for decoder independently

## Datasets

### Nonlinear Diffusion

PDE: `-∇·(e^m(x) ∇u) = f`

- **Standard**: `variance=5.0, lengthscales=(0.2, 0.2)` for f; `variance=1.0, lengthscales=(0.3, 0.3)` for m
- **Smooth**: `variance=2.0, lengthscales=(0.4, 0.4)` for f; `variance=0.5, lengthscales=(0.5, 0.5)` for m
- **Rough**: `variance=10.0, lengthscales=(0.1, 0.1)` for f; `variance=2.0, lengthscales=(0.15, 0.15)` for m

### Helmholtz

PDE: `-Δu - k²(1 + m(x))u = f`

- **Standard**: `sigma=0.2, rho=0.4`
- **Smooth**: `sigma=0.2, rho=0.4` (same as standard)
- **Rough**: `sigma=0.8, rho=0.1`

### Kolmogorov Flow (Navier-Stokes)

Time-dependent 2D Navier-Stokes equation at different Reynolds numbers:
- Re40, Re500, Re5000

## Standard Hyperparameters

All training uses consistent hyperparameters:
- **Layers**: 4
- **Width**: 24
- **MRBNO/PCANet modes**: 144 (fixed for input and output)
- **FNO modes**: (12, 12)
- **Learning rate**: 2e-3
- **Weight decay**: 1e-4
- **Max grad norm**: 3.0
- **Epochs**: 300
- **Early stopping patience**: 50
- **Batch size**: 32 (8 for Re5000)
- **Loss**: Relative L1
- **LR schedule**: CosineAnnealingLR (T_max=300, eta_min=0.0)
- **Data split**: 0.7/0.2/0.1 (train/val/test)
- **Samples**: 10000 per dataset

## Directory Structure

```
multilayer_RBNO/
├── models/              # Model architectures
├── data/                # Dataset generation scripts
├── scripts/             # Training and analysis scripts
│   ├── train_*.ipynb    # Training notebooks
│   ├── complexity_analysis_mrbno.py  # Complexity analysis
│   └── pca_sweep_analysis.py         # PCA parameter sweeps
├── irregular_grid/      # MRBNO for irregular grids and point clouds (see irregular_grid/README.md)
└── data_efficiency/     # Two-stage training experiments for data efficiency (see data_efficiency/README.md)
```

## Usage

### Generate Datasets

```bash
# Generate 10k samples for all datasets
./generate_10k_samples.sh

# Or generate individual datasets
cd multilayer_RBNO/data
python3 create_nonlinear_dataset.py --num_samples 10000
python3 generate_nonlinear_smooth.py --num_samples 10000
python3 generate_nonlinear_rough.py --num_samples 10000
python3 generate_helmholtz_smooth.py --num_samples 10000
python3 generate_helmholtz_rough.py --num_samples 10000
```

### Run Training

Training notebooks are in `multilayer_RBNO/scripts/`:
- `train_nonlinear_*.ipynb`: Nonlinear diffusion datasets
- `train_helmholtz_*.ipynb`: Helmholtz datasets
- `train_kolmogorov_*.ipynb`: Kolmogorov flow datasets

### Run Analysis Scripts

```bash
# Complexity analysis (modes × width sweeps)
module load python && module load pytorch
python3 multilayer_RBNO/scripts/complexity_analysis_mrbno.py

# PCA sweep analysis (epsilon and mode sweeps)
python3 multilayer_RBNO/scripts/pca_sweep_analysis.py
```

## Output Locations

All outputs are saved to `$SCRATCH`:
- **Datasets**: `$SCRATCH/data/`
- **Checkpoints**: `$SCRATCH/checkpoints/`
- **Logs**: `$SCRATCH/logs/`
- **Analysis results**: `$SCRATCH/complexity_analysis/`, `$SCRATCH/pca_sweep_analysis/`

## Requirements

- Python 3.11/3.12
- PyTorch
- FiPy (for nonlinear diffusion)
- FEniCS/dolfin (for Helmholtz)
- hippylib, hippyflow (for Helmholtz)
- JAX (for GP sampling)

## Environment Setup

```bash
module load python
module load pytorch
# For Helmholtz datasets:
conda activate test_fenics  # or appropriate environment with dolfin
```

## Key Features

1. **Consistent hyperparameters** across all experiments
2. **Standardized data splits** (0.7/0.2/0.1)
3. **Relative L1 loss** for training
4. **Cosine annealing LR schedule**
5. **Early stopping** with patience=50
6. **Gradient clipping** (max_norm=3.0)

## Model Comparison

| Model | Spectral Basis | Input Channels | Output Channels |
|-------|---------------|----------------|-----------------|
| FNO | Fourier | Arbitrary (via lifting) | Arbitrary (via projection) |
| MRBNO Lift Simple | PCA (single basis) | Arbitrary (via lifting) | Arbitrary (via projection) |
| MRBNO v3/v5 | PCA (separate input/output) | Arbitrary (via encoder) | Arbitrary (via decoder) |

## Notes

- MRBNO Lift Simple uses input PCA for Helmholtz (1 input channel, 2 output channels) and output PCA for nonlinear (2 input channels, 1 output channel) to match hidden representation spatial dimensions
- All checkpoints and logs are saved to `$SCRATCH` to avoid home directory quota issues
- Datasets are generated with 10000 samples by default

