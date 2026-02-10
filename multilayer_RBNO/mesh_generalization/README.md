# Mesh Generalization

Evaluates generalization of FNO, MRBNO, and PCANet when trained on one grid resolution and evaluated on others. Uses the nonlinear diffusion dataset (`nonlinear_diffusion_dataset.npz`).

## Contents

- **`test_mesh_generalization.py`**: Trains (or loads) FNO, MRBNO, and PCANet on a base resolution, then evaluates on multiple resolutions using transfer methods.
- **`results/`**: Checkpoints (`fno_checkpoint.pt`, `mrbno_checkpoint.pt`) and generated outputs (e.g. `mesh_generalization_results.json`, plots).
- **`run_test.sh`**: Wrapper to run the test with `module load python` and `module load pytorch`.

## Transfer methods

1. **Nodal interpolation**: Resize input to training resolution, forward pass, resize output to target resolution.
2. **L2 projection**: Interpolate PCA components (MRBNO/PCANet) to target resolution and build a new model with interpolated basis.
3. **PCA refitting**: Interpolate training data to target resolution, refit PCA, create new model and copy compatible weights.
4. **PCA on finest mesh**: Define PCA at a finest resolution; for any target, interpolate input to finest, run model, interpolate output to target.
5. **PCA with mass matrix**: M-orthonormal PCA per grid; transfer by interpolating modes and re-orthonormalizing with target mass matrix.

## Usage

```bash
# From this directory
./run_test.sh
# or
python3 test_mesh_generalization.py
```

Expects `$SCRATCH/data/nonlinear_diffusion_dataset.npz`. Checkpoints may be loaded from `results/` or from `$SCRATCH/checkpoints/` (e.g. FNO/PCANet notebook checkpoints). MRBNO uses only the checkpoint trained in this script (from `multilayer_RBNO.models.MRBNO`).

## Training

Training uses L1 loss, AdamW, CosineAnnealingLR, batch size 16, and the same hyperparameters (width, modes, layers) as the main training notebooks. Data split: 80% train, 10% val, 10% test.

## Files in this directory

- **.sh**: `run_test.sh` (run wrapper).
- **.md / .txt**: None except this README.
- **results/**: Contains generated checkpoints (`.pt`); can be recreated by running the script.
