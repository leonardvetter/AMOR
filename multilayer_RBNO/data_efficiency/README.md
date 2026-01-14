# Data Efficiency Experiments

This directory contains two-stage training experiments for improved data efficiency.

## Overview

The `mrbno_two_stage.py` model implements a two-stage training approach:

1. **Stage 1**: Train entirely in PCA coefficient space (loss computed in PCA space)
   - Removes real-space input/output layers (`fc0` and `fc2`)
   - Trains blocks to learn mappings in PCA space
   - Input/output are PCA coefficients directly

2. **Stage 2**: Fine-tune on real-space data
   - Adds back real-space layers (`fc0` and `fc2`)
   - Freezes pre-trained blocks from Stage 1
   - Fine-tunes only the real-space layers

## Key Difference from MRBNO_lift_simple

The two-stage model uses **separate input and output PCA bases**:
- `pca_components_in`, `pca_mean_in`: For encoding input data
- `pca_components_out`, `pca_mean_out`: For decoding output data

This allows the model to use different bases optimized for input and output distributions.

## Key Files

- **`mrbno_two_stage.py`**: Two-stage MRBNO model implementation
- **`train_two_stage_nonlinear.py`**: Training script for nonlinear diffusion dataset
- **`train_two_stage_kolmogorov.py`**: Training script for Kolmogorov flow datasets
- **`train_two_stage_well.py`**: Training script for well dataset
- **`train_baseline_nonlinear.py`**: Baseline training (single-stage) for comparison
- **`run_training_nonlinear.py`**: Main script to run two-stage training
- **`run_stage2_experiments.py`**: Script to run Stage 2 fine-tuning experiments
- **`run_training_with_metrics.py`**: Training with detailed metrics logging

## Usage

```bash
# Run two-stage training on nonlinear diffusion
python run_training_nonlinear.py --stage1_epochs 600 --stage2_epochs 200

# Run baseline for comparison
python train_baseline_nonlinear.py --epochs 300

# Run Stage 2 experiments
python run_stage2_experiments.py
```

## Model Architecture

### Stage 1 (`MRBNOStage1`)
- Input: PCA coefficients `(B, in_channels, n_modes_in)`
- Output: PCA coefficients `(B, out_channels, n_modes_out)`
- Decodes input PCA to spatial for blocks, encodes output back to PCA
- Uses output PCA basis for blocks

### Stage 2 (`MRBNOStage2`)
- Input: Real-space data `(B, in_channels, H, W)`
- Output: Real-space data `(B, out_channels, H, W)`
- Uses frozen blocks from Stage 1
- Only `fc0` and `fc2` are trainable

## Differences from MRBNO_lift_simple

1. **PCA bases**: Separate input/output bases vs single shared basis
2. **Training**: Two-stage approach vs single-stage
3. **Stage 1**: Operates entirely in PCA space vs real space
4. **Flexibility**: Can use different numbers of modes for input/output

