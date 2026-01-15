# Multi-layer Reduced Basis Neural Operator (MRBNO)

Repository for training and evaluating Multi-layer Reduced Basis Neural Operators (MRBNO) and related models on PDE datasets.

## Overview

This repository implements several neural operator architectures:
- **FNO** (Fourier Neural Operator): Uses Fourier transforms for spectral operations
- **MRBNO**: Multi-layer Reduced Basis Neural Operator using separate input/output PCA bases with basis transition
- **PCANet**: Simple neural network operating in PCA coefficient space

## Models

### MRBNO

- **Architecture**: Multi-stage basis transition with separate input/output PCA bases
  - Block 0: Input basis → Input basis
  - Block 1: Input basis → Output basis (transition)
  - Blocks 2+: Output basis → Output basis
- **Data representation**: Flattens spatial dimensions after lifting, works in (B, width, C*H*W) space
- **Weight structure**: 4D tensor mixing width channels and PCA modes, with optional factorized representation for parameter efficiency
- **Features**: Scalar gates for adaptive skip connections, global skip connection, batch normalization in latent space
- **Parameter efficiency**: Auto mode uses factorized weights when k_in × k_out > 10,000

### FNO

- **Input channels**: Handles arbitrary number via lifting layer (Conv2d)
- **Architecture**: Lifting → FNO blocks → projection
- **Spectral operations**: FFT → multiply in frequency domain → IFFT
- **Modes**: Tuple (modes_x, modes_y) for low-frequency modes

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

| Model | Spectral Basis | Architecture |
|-------|---------------|--------------|
| FNO | Fourier | Lifting → FNO blocks → projection |
| MRBNO | PCA (separate input/output bases) | Multi-stage basis transition with scalar gates |
| PCANet | PCA (coefficient space) | Neural network operating directly in PCA space |

## Notes

- MRBNO uses separate input and output PCA bases, with k_in and k_out computed from epsilon cutoff
- All checkpoints and logs are saved to `$SCRATCH` to avoid home directory quota issues
- Datasets are generated with 10000 samples by default
- Complexity analysis script supports two sweep versions: fixed width×k and fixed width×(1/epsilon)

