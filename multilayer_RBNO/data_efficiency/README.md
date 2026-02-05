# Two-Stage Training on Nonlinear Diffusion Dataset

This directory contains code for two-stage training of MRBNO models on the nonlinear diffusion dataset (`nonlinear_diffusion_dataset_big.npz`).

## Dataset

- **Dataset**: `nonlinear_diffusion_dataset_big.npz` (20,000 samples)
- **Input**: 2 channels (f, m), spatial dimensions 100×100
- **Output**: 1 channel (u), spatial dimensions 100×100
- **PDE**: -∇·(exp(m)∇u) = f

## Training Approaches

### 1. Two-Stage Training

**Stage 1**: Train in PCA coefficient space on 90% of train/val data
- Model: `MRBNOStage1` (removes parts operating on original spatial data)
- Loss: Relative L1 loss in PCA space
- Output: PCA coefficients
- **PCA computation**: Uses ALL train/val data (including Stage 2 data) to ensure identical PCA basis with baseline

**Stage 2**: Fine-tune complete MRBNO.py model on 10% of train/val data
- Model: `MRBNOStage2` (complete MRBNO.py architecture)
- **PCA reuse**: Uses PCA from Stage 1 checkpoint (computed on all train/val data)
- Seven versions:
  - **Version 1**: Only new weights trained (blocks frozen)
  - **Version 2**: All weights trained, higher LR for new weights (blocks LR = 0.1 × base LR)
  - **Version 3**: All weights trained equally
  - **Version 4**: Progressive unfreezing
  - **Version 5**: LR warmup for new layers
  - **Version 6**: Knowledge distillation
  - **Version 7**: Better initialization for new layers
- Loss: Relative L1 loss in real space

### 2. Baseline Training

Train MRBNO.py directly on full 20k samples (one-stage training)
- Model: Complete MRBNO.py
- Loss: Relative L1 loss
- All samples used for training
- **PCA computation**: Uses ALL train/val data (same as Stage 1) to ensure identical PCA basis for fair weight comparison

## Hyperparameters

From `train_FNO_MRBNO_PCANet.ipynb`:
- Batch size: 32
- Learning rate: 2e-3
- Epochs: 500 (with early stopping patience=50)
- Width: 24
- Layers: 4
- Dropout: 0.0
- Weight decay: 1e-4
- Max gradient norm: 3.0
- Optimizer: AdamW (betas=(0.9, 0.999), eps=1e-8)
- Scheduler: CosineAnnealingLR (eta_min=0.0)
- PCA modes: k_in=144, k_out=144 (99% variance cutoff)

## Usage

### Stage 1 Training

```bash
python3 train_stage1.py --data_path $SCRATCH/data/nonlinear_diffusion_dataset_big.npz \
                        --batch_size 32 --epochs 300 --lr 2e-3 \
                        --width 24 --layers 4 --k_in 144 --k_out 144
```

Checkpoint saved to: `$SCRATCH/checkpoints/stage1_final.pt`

### Stage 2 Training

```bash
# Version 1: Frozen blocks
python3 train_stage2.py --stage1_checkpoint $SCRATCH/checkpoints/stage1_final.pt \
                        --version 1 --batch_size 32 --epochs 300 --lr 2e-3

# Version 2: Different LR for blocks
python3 train_stage2.py --stage1_checkpoint $SCRATCH/checkpoints/stage1_final.pt \
                        --version 2 --lr_blocks_factor 0.1 --batch_size 32 --epochs 300 --lr 2e-3

# Version 3: Equal LR
python3 train_stage2.py --stage1_checkpoint $SCRATCH/checkpoints/stage1_final.pt \
                        --version 3 --batch_size 32 --epochs 300 --lr 2e-3
```

### Baseline Training

```bash
python3 train_baseline.py --data_path $SCRATCH/data/nonlinear_diffusion_dataset_big.npz \
                           --batch_size 32 --epochs 300 --lr 2e-3 \
                           --width 24 --layers 4 --k_in 144 --k_out 144
```

## Metrics

All models report:
- Relative L1 loss (training metric)
- Relative L2 error (evaluation metric)
- Metrics computed on train, validation, and test splits

## Files

- `train_two_stage.py`: Core training functions and model definitions
- `train_stage1.py`: Stage 1 training script
- `train_stage2.py`: Stage 2 training script (3 versions)
- `train_baseline.py`: Baseline training script

## Notes

- Checkpoints are saved to `$SCRATCH/checkpoints/`
- Data split: 70% train, 20% val, 10% test (from full dataset)
- Stage 1 uses 90% of train/val (63% of total)
- Stage 2 uses 10% of train/val (7% of total)
- Test set (10% of total) used for final evaluation
