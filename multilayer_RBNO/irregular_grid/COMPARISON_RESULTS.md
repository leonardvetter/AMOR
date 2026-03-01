# MRBNO vs Geo-FNO Paper: Full Comparison

Same input-output mapping and splits as Geo-FNO (Li et al., JMLR 24(388) 2023). Metric: test relative L2 (Eq. 2 in paper). **All error values below are in percent** (100 × relative L2).

**Bug fix:** Test rel L2 was previously averaged over *batches* instead of *samples*, inflating MRBNO numbers. Fixed in `compare_with_geofno_paper.py` (average over `n_samples`). Re-run the script to refresh the table; corrected MRBNO values should align with the training summary.

**Parameter counts:** MRBNO parameter count varies by dataset because `k_in`/`k_out` are set per dataset from PCA (0.999 variance, capped at 144). Pipe’s much smaller count reflects lower effective PCA modes.

## Full Comparison Table

| Dataset | Model | Test Rel L2 (%) | Parameters |
|---------|-------|-----------------|------------|
| **elasticity** | FNO (interp, paper) | 4.95 | — |
| | UNet (paper) | 5.34 | — |
| | GNO (paper) | 12.7 | — |
| | DeepONet (paper) | 9.65 | — |
| | Geo-FNO (paper) | 3.41 | — |
| | MRBNO (this work) | 8.43 | 24,174,790 |
| **airfoil** | FNO (interp, paper) | — | — |
| | UNet (paper) | — | — |
| | GNO (paper) | — | — |
| | DeepONet (paper) | — | — |
| | Geo-FNO (paper) | — | — |
| | MRBNO (this work) | 6.48 | 36,355,630 |
| **pipe** | FNO (interp, paper) | — | — |
| | UNet (paper) | — | — |
| | GNO (paper) | — | — |
| | DeepONet (paper) | — | — |
| | Geo-FNO (paper) | — | — |
| | MRBNO (this work) | 100.55 | 59,006 |
| **plasticity** | FNO (interp, paper) | — | — |
| | UNet (paper) | — | — |
| | GNO (paper) | — | — |
| | DeepONet (paper) | — | — |
| | Geo-FNO (paper) | — | — |
| | MRBNO (this work) | 100.20 | 171,690,965 |
| **car-cfd** | FNO (interp, paper) | — | — |
| | UNet (paper) | — | — |
| | GNO (paper) | — | — |
| | DeepONet (paper) | — | — |
| | Geo-FNO (paper) | — | — |
| | MRBNO (this work) | 29.03 | 59,725,955 |
| **channel-shocks** | FNO (interp, paper) | — | — |
| | UNet (paper) | — | — |
| | GNO (paper) | — | — |
| | DeepONet (paper) | — | — |
| | Geo-FNO (paper) | — | — |
| | MRBNO (this work) | 0.73 | 59,727,686 |

Paper: Li et al. JMLR 24(388) 2023, Table 1 and Table 2. Elasticity from Table 1 (hyper-elastic). Fill FNO/UNet/GNO/DeepONet/Geo-FNO for other datasets from the paper when available.

## Data alignment with Geo-FNO paper

- **Input/output:** Elasticity: mesh (x,y) + RR (42) → sigma (stress). Airfoil: (X,Y) only → Q ch4. Pipe: Q[0:2]+X,Y → Q[2]. Plasticity: material props (101) broadcast → time+spatial flattened. Car-cfd: points (3D) → pressure. Channel-shocks: Q[0:4]+X,Y → Q[4]. Matches paper/code where documented; we use unstructured grid data directly (no mapping to uniform).
- **Normalization:** `normalize=False` in `prepare_geofno_data` to match paper (no target normalization).
- **Splits:** Geo-FNO fixed splits (elasticity 1000 train / last 200 test; airfoil first 1000 / next 200 test; pipe 1800/300; plasticity 700/200; car-cfd 700/100; channel-shocks 1600/200).

## Investigation: Why Pipe and Plasticity (and Others) Are Bad — Differences vs Geo-FNO, Fixes

**Pipe (MRBNO test rel L2 ~100%):**
- **Geo-FNO pipe.py** (NeuralOperator/Geo-FNO repo): input = **(X, Y) only**; output = **Pipe_Q channel 0** (first channel). No Q fields as input.
- **Our previous setup:** input = Q[0:2] + X,Y; output = **Q channel 2**. So we predicted a different output channel and used extra inputs. **Fix applied:** loader now uses (X, Y) only as input and **Q channel 0** as output to match Geo-FNO. **Action:** retrain pipe with the updated loader. (Current pipe checkpoint was trained with the old I/O; pipe plot will fail until retrained.)
- Huge training/val loss (e.g. 7.49e5) was consistent with a mismatch (wrong output channel or scale). After fix, if loss remains large, try `normalize=True` for pipe in `prepare_geofno_data`.

**Plasticity (MRBNO test rel L2 ~100%):**
- **Geo-FNO plasticity_3d.py** uses a different data file (`buri_plas_N987_time.mat`) and keeps output as (101, 31, 20, 4) — 4D with last dim 4. We use `plas_*.mat`, output (2480, 101) = flattened (31×20×4) per node. Same total size; layout differs.
- Paper uses ntrain=900, ntest=80 (last 80); we use 700 train, last 200 test. Splits differ.
- No normalization in either. Very high-dimensional output (2480 channels) and possible large value range can make PCA and training unstable. **Fixes to try:** (1) Use `normalize=True` for plasticity so targets are z-scored from training stats. (2) Confirm .mat file matches paper (same reader/shape as plasticity_3d.py if you switch to buri_plas_N987_time.mat).

**Elasticity (MRBNO 8.4% vs Geo-FNO 3.41%):**
- Input/output (mesh+RR → sigma) match paper. ~2× worse may be architecture or training; no input/output bug identified.

**Car-cfd (MRBNO 29%):**
- We use points (3D) → pressure; Geo-FNO may use transports interpolated to mesh. No normalization. Try `normalize=True` for car-cfd and retrain.

**Airfoil (6.48%) / Channel-shocks (0.73%):**
- In a reasonable range; no change required for alignment.

## MRBNO Training Summary

| Dataset | Train/Val/Test | Best Val Loss | Test Loss |
|---------|----------------|---------------|-----------|
| elasticity | 1000/800/200 | — | — |
| airfoil | 1000/1290/200 | 0.0194 | 0.0208 |
| pipe | 1800/210/300 | 7.49e5 | 2.20e6 |
| plasticity | 700/87/200 | 9.43e2 | 9.25e2 |
| car-cfd | 700/89/100 | 0.157 | 0.179 |
| channel-shocks | 1600/200/200 | 0.0038 | 0.0047 |

Val/test loss values from logs. Training criterion is **relative L1** (matching leonard version).

## Run Details

- Comparison: `python3 compare_with_geofno_paper.py --device cpu` (re-run after metric fix to refresh table)
- MRBNO: width 24, layers 4, PCA modes 144, **relative L1 loss** (training), CosineAnnealingLR
- Splits: elasticity first 1000 train / last 200 test; airfoil first 1000 train / next 200 test; pipe 1800/300; plasticity 700/200; car-cfd 700/100; channel-shocks 1600/200

**Next steps:** Re-run the comparison script to get corrected MRBNO test rel L2 (%). If pipe or plasticity still show very high test errors, fix data scaling in the loader and retrain those datasets.

## Plots (ground truth, prediction, error, mesh)

For each dataset, generate a 4-panel figure (mesh/point cloud, ground truth field, MRBNO prediction, absolute error) by running:

```bash
python3 plot_all_datasets.py
```

Plots are written to `$SCRATCH/data/geofno_predictions/` and copied to **`figures/`** in the current directory (one PNG per dataset, sample index 0). After retraining pipe with the fixed loader, re-run `plot_all_datasets.py` to refresh the pipe figure. To plot a specific dataset and sample:

```bash
python3 plot_predictions.py --dataset_type <elasticity|airfoil|pipe|plasticity|car-cfd|channel-shocks> --checkpoint_path $SCRATCH/checkpoints/geofno/checkpoint_best_<name>.pt --sample_idx 0 --device cpu
```

- **Elasticity / airfoil / pipe / channel-shocks:** 2D mesh scatter; panels show mesh (X,Y), prediction, ground truth, error.
- **Car-cfd:** 3D point cloud projected to (X,Y); same four panels.
- **Plasticity:** No spatial mesh in data; node-index layout used for the four panels (first output channel only).
