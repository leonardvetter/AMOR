# Mesh generalization

Train FNO / AMOR / PCANet on one nonlinear-diffusion resolution and evaluate on others.

## Requirements

Under `$AMOR_SCRATCH_DATA` (see `AMOR/paths.py`):

- `nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.npz` (train at \(N=100\))
- `nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_res{50,75,150,200}.npz`

Generate with:

```bash
python3 AMOR/data/generate_nonlinear_diffusion.py --preset medium --num_samples 10000
python3 AMOR/data/generate_nonlinear_diffusion.py --preset medium --N 50 --num_samples 1000 --suffix res50
# likewise res75, res150, res200
```

## Run

```bash
python3 AMOR/mesh_generalization/run_mesh_generalization.py
```

Writes metrics JSON (and a short markdown summary) under the script’s configured results directory. Transfer strategies are implemented in the runner (interpolation / L2 / PCA variants); see the script header and argparse.
