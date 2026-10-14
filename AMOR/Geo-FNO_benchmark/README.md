# AMOR vs Geo-FNO (`AMOR/Geo-FNO_benchmark/`; Table `tab:irregular`, airfoil / elasticity figures)

Everything specific to this experiment lives here. It reuses `AMOR.models.amor.AMOR`,
`PCAReducerMultiChannelMass` and `AMOR.training.loop2d.train_model`. Baselines (Geo-FNO, FNO-interp, GraphNO, DeepONet, DSE) are
**quoted** from the papers, only AMOR is run.

| file | purpose |
|---|---|
| `run_geofno.py` | train + evaluate one (dataset, seed); writes `results/`, `predictions/`, `checkpoints/` |
| `data.py` | loaders, published split, lumped node mass (quad dual cells / Delaunay triangles) |
| `aggregate.py` | mean ± std over seeds -> `geofno_summary.csv`, `tab_irregular.tex`, `setup_snippet.tex` |
| `make_figures.py` | `airfoil_sample_visualization.png`, `elasticity_sample_visualization.png` |
| `slurm/` | `smoke.sh` (dev partition), `array.sh` (10 tasks = 2 datasets x seeds 0-4) |

## Setup (what is run)
Published Geo-FNO split 1000 train / 200 test (airfoil test = samples 1000-1199, elasticity test = last 200;
checked against the Geo-FNO scripts). Input = node coordinates (2 channels), output = Mach
number / stress. Z-score (inputs per node and channel as Geo-FNO's normalizer, std floored at 1% of the channel's median node std; outputs channel-wise) and mass-weighted PCA fit on the train split only; lumped node mass of the
mean training mesh; (k_in, k_out) from 99.9% explained variance; AMOR with factorized weights, width
floor(k_out/2), 6 (airfoil) / 4 (elasticity) blocks. AdamW lr 2e-3, wd 1e-4, cosine, `HP["epochs"]` epochs (paper: 300), clip 3.0, no validation / early stopping (final model evaluated),
batch size 20 (Geo-FNO's; the paper does not state it), relative L2 loss on the physical fields.
Metrics: per-sample relative L1 / L2 of the physical fields, mean over the 200 test samples, in percent.
Seeds 0-4 vary initialisation and batch order (the split is fixed).

## Local test
```bash
python AMOR/Geo-FNO_benchmark/run_geofno.py --smoke                      # ~30 s, 3 epochs, 200 train samples
python AMOR/Geo-FNO_benchmark/run_geofno.py --epochs 5 --out-dir experiments_out/geofno_test
python AMOR/Geo-FNO_benchmark/aggregate.py --out-dir experiments_out/geofno_test
python AMOR/Geo-FNO_benchmark/make_figures.py --out-dir experiments_out/geofno_test
```

## Cluster
Data: `<repo>/AMOR/data/Airfoil_dataset/` and `.../Elasticity_dataset/` (override with `DATA_DIR`). From the repo root:
```bash
sbatch AMOR/Geo-FNO_benchmark/slurm/smoke.sh      # first: quick check of environment + paths
sbatch AMOR/Geo-FNO_benchmark/slurm/array.sh      # then 10 tasks (airfoil 0-4, elasticity 5-9)
```
Results: `experiments_out/geofno/` (override with `OUT_DIR`). Copy it back, then run `aggregate.py` and `make_figures.py`
with `--out-dir`.
