# Adaptive Modal Operator Refinement (AMOR)

Code for the AMOR paper: train AMOR / FNO / PCANet, regenerate the paper datasets, and apply the same AMOR class to a new grid or mesh problem.

**Dataset binaries are not in this repository.** AMOR-generated datasets are on Zenodo ([upload 18815516](https://zenodo.org/uploads/18815516)). Third-party sets (Kolmogorov, FlowBench) are linked under Citation.

There is **one** AMOR implementation: `AMOR.models.amor.AMOR`. Mass weighting, FiLM, coordinates, 2D/3D layout, and trainable QR/Cayley bases are constructor options.

## Repository layout

```
AMOR/
  paths.py              # AMOR_DATA_ROOT / AMOR_SCRATCH_DATA (default: ~/data)
  models/
    amor.py             # unified AMOR
    PCAReducer.py       # 2D / mesh PCA (+ mass-weighted PCA)
    PCAReducerVolume.py # 3D volume PCA
    FNO_model.py        # 2D FNO baseline
    FNO3d_neuralop.py   # 3D FNO baseline
    PCANet.py
    bases/              # PCA helpers, polynomials, Laplacian, learnable QR/Cayley
    mass/               # lumped nodal mass helper
  training/             # shared loops, metrics, checkpoint helpers
  data/                 # generators and loaders (no binaries)
  scripts/              # train / eval / fit / timing entry points
  data_efficiency/      # two-stage memory-efficiency experiment
  mesh_generalization/  # resolution-transfer experiment
  Geo-FNO_benchmark/    # airfoil and elasticity; own data.py, runner, and slurm
```

Scripts put the repository root on `sys.path`. From any working directory:

```bash
python AMOR/scripts/train_kolmogorov.py --re 40 --models AMOR
```

Paths default through environment variables (see Local paths). Override with `--data_path`, `--data-path`, or `--data-dir` on each script (`--help`).

## Environment

```bash
conda env create -f environment.yml
conda activate amor
```

Helmholtz generation also needs hippylib / hippyflow (`HIPPYLIB_PATH`, `HIPPYFLOW_PATH`).

## Local paths

| Variable | Default | Role |
|---|---|---|
| `AMOR_DATA_ROOT` | `~/data` | Zenodo downloads and paper arrays |
| `AMOR_SCRATCH_DATA` | same as `AMOR_DATA_ROOT` | large intermediates / results |
| `AMOR_MESHES_DIR` | `$AMOR_DATA_ROOT/meshes_3d` | rib mesh NPZs |
| `AMOR_RIB_DATASETS` | `$AMOR_DATA_ROOT/aircraft_rib/datasets` | rib NPZs |

After downloading Zenodo into `$AMOR_DATA_ROOT`, the trainers expect names such as:

- `helmholtz_sigma_*_rho_*.npz`
- `nonlinear_fvar_*.npz`
- `2D_NS_Re{40,500,5000}.npy` (Kolmogorov; separate Zenodo)
- `3d_elastostatics/elastostatics3d_*.npz`
- FlowBench LDC under the path used by `train_flowbench_ldc3d_single.py --data-dir`
- `aircraft_rib/datasets/aircraft_rib_*.npz`

## Reproduce paper experiments

On structured 2D benchmarks, AMOR and PCANet use `k_in = k_out = 144` and FNO uses `12×12` Fourier modes unless a trainer overrides them. Width and depth follow each script (SI model table). AMOR `layers` is the **total** number of nonlocal blocks including the input→output transition (`block0` + transition + `layers-2` output residual blocks). Reported `test_rel_l1` / `test_rel_l2` are relative errors in physical units after undoing the train-set z-score.

| Experiment | Command |
|---|---|
| Helmholtz (all priors × models) | `python AMOR/scripts/train_helmholtz.py` |
| Helmholtz (one cell) | `python AMOR/scripts/train_helmholtz.py --dataset sigma_0.2_rho_0.4 --model AMOR` |
| Nonlinear diffusion | `python AMOR/scripts/train_nonlinear_diffusion.py [--dataset …] [--model …]` |
| Kolmogorov | `python AMOR/scripts/train_kolmogorov.py --re {40,500,5000} [--models AMOR]` |
| Basis ablations (2D) | `python AMOR/scripts/train_amor_{helmholtz,kolmogorov_Re40,nonlinear_diffusion}_basis.py --basis_type …` |
| DIS bases | `python AMOR/scripts/fit_{helmholtz,nonlinear_diffusion}_dis_basis.py` |
| Elastostatics accuracy | `python AMOR/scripts/train_elastostatics_single.py --model AMOR` |
| Elastostatics timing | `python AMOR/scripts/bench_elastostatics_timing_scaling.py` |
| FlowBench LDC 3D | `python AMOR/scripts/train_flowbench_ldc3d_single.py --model AMOR` |
| Aircraft rib tip (mass-PCA) | `python AMOR/scripts/train_amor_aircraft_rib.py` |
| Rib tip basis sweep | `python AMOR/scripts/train_amor_aircraft_rib.py --basis_type learned_qr` |
| Rib bcfield / mixed | `python AMOR/scripts/fit_aircraft_rib_bcfield_pca.py --data_path …` then `train_amor_aircraft_rib.py --dataset bcfield --data_path … --checkpoint_dir … --pca_npz …` |
| Mesh generalization | `python AMOR/mesh_generalization/run_mesh_generalization.py` |
| Data efficiency | `python AMOR/data_efficiency/run_memory_efficiency.py` |
| Geo-FNO airfoil / elasticity | `python AMOR/Geo-FNO_benchmark/run_geofno.py --dataset {airfoil,elasticity} --seed 0` |

`train_kolmogorov.py` stores the published per-Reynolds width, depth, and PCA SVD solver in `RE_DEFAULTS`. All trainers keep `use_coordinates=False` and `use_batch_norm=False`. Rib tip defaults enable FiLM; pass `--no_film_conditioning` only if you intentionally disable it.

### Aircraft-rib mass weighting

`--dataset cantilever_tip` assembles the consistent P1 mass from `nodes` / `elements`, fits `PCAReducerMultiChannelMass` in that inner product, and encodes with `z = Φᵀ M (x−μ)` (the M-adjoint rows, not a diagonal multiply). Mesh transfer keeps `B = (P Φ)ᵀ` and sets `E = G^{-1}(B M)` with the same consistent mass on the target mesh (`evaluate_aircraft_rib_mass_generalization.py --dual_mode paper --mass_kind consistent`). `--dataset bcfield` uses Euclidean `PCAReducerMultiChannel` from a precomputed `--pca_npz` bundle (no mass matrix).

## Recreate datasets

Prefer the Zenodo deposit above. Generators and loaders both live under `AMOR/data/` (there is no separate `datasets` package). `aircraft_rib/datasets/` in the path table is only the on-disk directory for generated rib NPZs. Generators rebuild the same files:

```bash
# Nonlinear diffusion (paper priors)
python AMOR/data/generate_nonlinear_diffusion.py --preset medium
python AMOR/data/generate_nonlinear_diffusion.py --preset smooth
python AMOR/data/generate_nonlinear_diffusion.py --preset rough

# Helmholtz (needs hippylib / hippyflow)
python AMOR/data/generate_helmholtz.py --preset ultrasmooth   # sigma=0.1, rho=0.6
python AMOR/data/generate_helmholtz.py --preset smooth        # sigma=0.2, rho=0.4
python AMOR/data/generate_helmholtz.py --preset standard      # sigma=0.4, rho=0.2

# 3D elastostatics
python AMOR/data/3D_elastostatics.py --help

# Aircraft rib (STEP is not redistributed; cite the paper and place a local copy)
# See AMOR/data/aircraft_rib/README.md
```

## Apply AMOR to a new problem

One class: `from AMOR.models.amor import AMOR`.

1. Store fields as `(S, C, *spatial)` arrays (grid) or `(S, C, N, 1)` nodal fields.
2. Fit a reducer on the **train** split only:
   - grids 2D: `PCAReducerMultiChannel`
   - grids 3D: `PCAReducerVolume`
   - FEM mesh with lumped mass: `PCAReducerMultiChannelMass` (pass `mass_nodes`)
3. Build the model. Dimension, channel counts, and mode counts follow `rb`:

```python
import numpy as np
import torch
from AMOR.models.amor import AMOR
from AMOR.models.PCAReducer import PCAReducerMultiChannel

# Synthetic field-to-field map: (S, 1, 32, 32) -> (S, 1, 32, 32)
rng = np.random.default_rng(0)
x = rng.standard_normal((200, 1, 32, 32)).astype(np.float32)
y = np.sin(x)

rb = PCAReducerMultiChannel(random_state=0).fit(
    x[:140], y[:140], n_components_in=64, n_components_out=64
)
model = AMOR(
    width=16,
    rb=rb,
    k_in=64,
    k_out=64,
    layers=4,
    use_coordinates=False,
    use_film_conditioning=False,  # if True, forward(x, film_cond)
    trainable_basis=False,        # or True with trainable_basis_param="qr"|"cayley"
)
pred = model(torch.from_numpy(x[:8]))
assert pred.shape == (8, 1, 32, 32)
```

For a mass-weighted mesh problem, fit `PCAReducerMultiChannelMass` and call `AMOR(..., mass=True)` (or rely on the reducer: mass is inferred when `rb` exposes `encoder_mass_diag`). For FiLM, pass `use_film_conditioning=True` and `model(x, film_cond)`.

Training schedules in the paper scripts are AdamW \(2\cdot10^{-3}\), weight decay \(10^{-4}\), grad clip \(3\), cosine annealing, relative \(L^1\). Reusable loops: `AMOR/training/loop2d.py` (field-to-field 2D), `loop3d.py` (3D), `loop_timestep.py` (Euler step).

## Citation

If you use this code, please cite the AMOR paper and this repository. If you use a dataset, please also cite that dataset.

Geo-FNO airfoil and elasticity are trained only by `AMOR/Geo-FNO_benchmark/run_geofno.py`: published 1000/200 split, no validation split, relative L2, batch size 20, 500 epochs, final epoch, width `floor(k_out/2)`, 6 blocks (airfoil) and 4 blocks (elasticity). Data directory is `AMOR_DATA_DIR` (default `<repo>/AMOR/data`, with `Airfoil_dataset/` and `Elasticity_dataset/` inside it). The Slurm scripts read `DATA_DIR` and `OUT_DIR` from `slurm/_env.sh`.

```bibtex
@software{voderholzer2026amor,
  author       = {Voderholzer, David and Vetter, Leonard and O'Leary-Roseberry, Thomas},
  title        = {Adaptive Modal Operator Refinement ({AMOR})},
  year         = {2026},
  url          = {https://github.com/leonardvetter/AMOR},
  note         = {Software accompanying the AMOR paper; cite the paper and Zenodo data records}
}
```

**AMOR-generated datasets** (nonlinear diffusion, Helmholtz, elastostatics, aircraft rib, …) — [Zenodo upload 18815516](https://zenodo.org/uploads/18815516).

**Kolmogorov** — Li et al., ICLR 2021; Zenodo DOI [10.5281/zenodo.7495555](https://doi.org/10.5281/zenodo.7495555).

**Helmholtz generation** — hippyflow / hippylib.

**FlowBench** — Tali et al., arXiv:2409.18032.

**Geo-FNO airfoil and elasticity** — Li et al., *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries*, JMLR 2023.

See `LICENSE` (placeholder until the final license text is added).
