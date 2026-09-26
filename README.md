# Adaptive Modal Operator Refinement (AMOR)

Code to train AMOR / FNO / PCANet and reproduce the numerical experiments in the AMOR paper.
**Datasets are not shipped in this repository**; obtain them from Zenodo (see the paper and Citation below). Generators under `AMOR/data/` document how those datasets were produced.

## Layout

```
AMOR/
  paths.py           # AMOR_DATA_ROOT / AMOR_SCRATCH_DATA (default: ~/data)
  models/            # amor.py (unified AMOR), FNO, PCANet, reducers, bases/, mass/
  datasets/          # loaders and splits (kolmogorov, helmholtz family, rib, 3D, …)
  training/          # loops (2D / 3D / timestep), metrics, checkpoint I/O, SLURM
  data/              # generators only (no dataset binaries)
  scripts/           # runnable entry points only (train / eval / fit / timing)
  data_efficiency/
  mesh_generalization/
```

Import from the repository root as a package, e.g.
`from AMOR.models.amor import AMOR`. Scripts put that root on `sys.path`
themselves, so they run from any working directory with no extra setup.

## Environment

```bash
conda env create -f environment.yml
conda activate amor
```

Also required for some experiments: hippylib/hippyflow (`HIPPYLIB_PATH`, `HIPPYFLOW_PATH`) for Helmholtz; place Zenodo datasets under `$AMOR_DATA_ROOT` as expected by each script.

## Data

Download the project Zenodo release (paper / record) into `$AMOR_DATA_ROOT` (override with `AMOR_SCRATCH_DATA` if needed). Expected layout is defined by the trainers (e.g. `helmholtz_sigma_*_rho_*.npz`, `nonlinear_fvar_*.npz`, `2D_NS_Re{40,500,5000}.npy`, elastostatics NPZ, FlowBench LDC, aircraft-rib NPZ).

The aircraft-rib CAD STEP used to build the mesh is **not** redistributed here; it is cited in the paper. Generators under `AMOR/data/aircraft_rib/` assume a local STEP path if you rebuild meshes.

## Reproduce experiments

On structured 2D benchmarks, AMOR and PCANet use `k_in = k_out = 144`; FNO uses `12×12` Fourier modes per axis. Width/depth follow each trainer (SI model table).

All entry points live in `AMOR/scripts/`. Trainers that cover several dataset
variants select them with flags rather than separate files; with no flags they
run every variant and model of that family.

| Experiment | Entry point |
|---|---|
| Helmholtz | `train_helmholtz.py [--dataset sigma_*_rho_*] [--model {AMOR,FNO,PCANet}]` |
| Nonlinear diffusion | `train_nonlinear_diffusion.py [--dataset fvar_*] [--model {AMOR,FNO,PCANet}]` |
| Kolmogorov | `train_kolmogorov.py --re {40,500,5000} [--models ...]` |
| Basis ablations (wrappers) | `train_amor_{helmholtz,kolmogorov_Re40,nonlinear_diffusion,aircraft_rib}_basis.py --basis_type ...`; DIS: `fit_*_dis_basis.py` |
| Elastostatics accuracy | `train_elastostatics_single.py --model {FNO,AMOR}` |
| Elastostatics timing (`fig:scaling`, SI) | `bench_elastostatics_timing_scaling.py` (recon-matched \(k\)); `bench_elastostatics_independent_timing.py` (\(k_{\mathrm{in}}=k_{\mathrm{out}}\)) |
| FlowBench LDC 3D | `train_flowbench_ldc3d_single.py --model {FNO,AMOR}` |
| Aircraft rib tip (mass-PCA) | `train_amor_aircraft_rib_linear_mass.py` (or launcher `train_amor_aircraft_rib_easy_10000_best_mass.py`) |
| Rib tip basis ablation | `train_amor_aircraft_rib_basis.py --basis_type ...` |
| Rib bcfield / mixed | `fit_aircraft_rib_bcfield_pca.py` + `train_amor_aircraft_rib_bcfield.py` |
| Mesh generalization | `mesh_generalization/run_mesh_generalization.py` |
| Data efficiency | `data_efficiency/train_*.py` |

`train_kolmogorov.py` keeps the per-Reynolds settings of the published runs in
`RE_DEFAULTS` (width, depth, and the SVD solver used for the PCA fit), so
`--re 500` reproduces the Re=500 configuration rather than a shared default.
Every value can be overridden on the command line. The `*_basis.py` scripts are
thin wrappers over those trainers; they stay separate so each ablation keeps its
own checkpoint layout.

Aircraft-rib tip training is still four entry points (mass tip, basis sweep,
bcfield, tip launcher). They are not yet folded into one `--dataset`/`--pca`
trainer; see the rib provenance note before changing defaults.

Metrics `test_rel_l1` / `test_rel_l2` are relative errors in physical units after undoing train-set z-score.

## Apply AMOR to a new problem

1. Represent fields as tensors (grid or nodal).
2. Fit a reducer on the **train** split only: `PCAReducerMultiChannel` for 2D
   grids, `PCAReducerVolume` for 3D grids, `PCAReducerMultiChannelMass` for FEM
   meshes with a lumped nodal mass matrix. Bases are fitted outside the model
   and passed in, so any basis exposing the same interface works.
3. Instantiate `AMOR` (`AMOR/models/amor.py`). One class covers all published
   variants; the spatial dimension, channel counts and mode counts follow the
   reducer you pass as `rb`. Mass weighting (`mass=True`), FiLM conditioning
   (`use_film_conditioning=True`), coordinate embedding, and trainable bases
   (`trainable_basis` with `qr` or `cayley`) are constructor options.
   `AMOR_conditioned.py` keeps `AMORWithFieldInputs` / `AMORWithLoadFiLM` as
   aliases for code written against the earlier class names.
4. Train with the closest paper script’s schedule (typically AdamW \(2\cdot10^{-3}\), weight decay \(10^{-4}\), grad clip \(3\), cosine annealing, relative \(L^1\)). The loops in `AMOR/training/` are reusable: `loop2d.py` for field-to-field 2D problems, `loop3d.py` for 3D, `loop_timestep.py` for Euler-step time stepping.

## Citation

Cite the AMOR paper and this software. Cite Zenodo / third-party data records when using those datasets.

```bibtex
@software{voderholzer2026amor,
  author       = {Voderholzer, David and Vetter, Leonard and O'Leary-Roseberry, Thomas},
  title        = {Adaptive Modal Operator Refinement ({AMOR})},
  year         = {2026},
  url          = {https://github.com/leonardvetter/AMOR},
  note         = {Software accompanying the AMOR paper; cite the paper and Zenodo data records}
}
```

**Kolmogorov** — Li et al., ICLR 2021; Zenodo DOI [10.5281/zenodo.7495555](https://doi.org/10.5281/zenodo.7495555).

**Helmholtz generation** — hippyflow / hippylib.

**FlowBench** — Tali et al., arXiv:2409.18032.

**AMOR-generated datasets (nonlinear, elastostatics, rib, …)** — Zenodo release accompanying the paper.

See `LICENSE` (placeholder until the final license text is added).
