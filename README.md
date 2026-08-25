# AMOR -- Adaptive Modal Operator Refinement

Repository for training and evaluating AMOR and related models on PDE datasets.

## Overview

- **FNO**: Fourier Neural Operator (spectral operations via FFT).
- **AMOR**: Adaptive Modal Operator Refinement.
- **PCANet**: Neural network in PCA coefficient space.

## Models

### AMOR

- Blocks: Block 0 (input → input), Block 1 (input → output), Blocks 2+ (output → output). Data as (B, width, C*H*W). Weight tensor 4D (width, modes); optional factorized form when k_in×k_out > 10000. Scalar gates, global skip, batch norm in latent space.

### FNO

- Lifting → FNO blocks → projection. Modes (modes_x, modes_y). Lifting layer supports arbitrary input channels.

## Datasets

- **Nonlinear diffusion**: -∇·(e^m ∇u) = f (standard/smooth/rough variants).
- **Helmholtz**: -Δu - k²(1+m)u = f.
- **Kolmogorov flow**: 2D Navier–Stokes (Re40, Re500, Re5000).

## Standard hyperparameters

- Layers: 4. Width: 24. MRBNO/PCANet modes: 144. FNO modes: (12, 12).
- LR: 2e-3. Weight decay: 1e-4. Max grad norm: 3.0. Epochs: 300. Early stopping patience: 50. Batch size: 32 (8 for Re5000).
- LR schedule: CosineAnnealingLR (eta_min=0.0). Split: 0.7/0.2/0.1 (train/val/test). Samples: 10000 per dataset.

## Directory structure

```
multilayer_RBNO/
├── models/           # FNO, MRBNO, PCANet, PCAReducer
├── data/             # Dataset generation scripts
├── scripts/          # Training notebooks and analysis scripts
├── data_efficiency/  # Two-stage training (see data_efficiency/README.md)
├── irregular_grid/   # MRBNO on Geo-FNO-style irregular grids (see irregular_grid/README.md)
└── mesh_generalization/  # Resolution transfer experiments (see mesh_generalization/README.md)
```

The folders `data_efficiency`, `irregular_grid`, and `mesh_generalization` are described in their own READMEs.

## Usage

### Generate datasets

```bash
cd multilayer_RBNO/data
python3 create_nonlinear_dataset.py --num_samples 10000
python3 generate_nonlinear_smooth.py --num_samples 10000
python3 generate_nonlinear_rough.py --num_samples 10000
python3 generate_helmholtz_*.py --num_samples 10000
```

### Training

Notebooks in `multilayer_RBNO/scripts/`: `train_nonlinear_*.ipynb`, `train_helmholtz_*.ipynb`, `train_kolmogorov_*.ipynb`.

### Analysis

```bash
module load python && module load pytorch
python3 multilayer_RBNO/scripts/complexity_analysis_mrbno.py
python3 multilayer_RBNO/scripts/pca_sweep_analysis.py
```

## Output locations

- Datasets: `$SCRATCH/data/`
- Checkpoints: `$SCRATCH/checkpoints/`
- Logs: `$SCRATCH/logs/`

## Requirements

Python 3.11/3.12, PyTorch, FiPy (nonlinear diffusion), FEniCS/dolfin and hippylib/hippyflow (Helmholtz), JAX (GP sampling).

## Environment

```bash
module load python
module load pytorch
# Helmholtz: conda environment with dolfin as needed
```
