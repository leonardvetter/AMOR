# Irregular Grid MRBNO

MRBNO training and evaluation on Geo-FNO-style irregular-grid datasets. Uses local `MRBNO.py` and `PCAReducer.py` with 3D (point-cloud) input handling.

## Key files

- **`train_geofno.py`**: Training script. Requires `--dataset_path` and `--dataset_type` (airfoil, elasticity, pipe, plasticity, car-cfd, channel-shocks).
- **`geofno_loader.py`**: Data loader; input/output and splits follow Geo-FNO (see docstring).
- **`MRBNO.py`**, **`PCAReducer.py`**: Model and PCA reducer for this directory (3D support).
- **`losses.py`**: LpLoss (absolute/relative Lp, rel_l1). Training uses relative L2 loss.
- **`compute_metrics.py`**: Computes rel_l1, rel_l2, and Geo-FNO metric on train/val/test.
- **`download_datasets.py`**, **`download_geofno_datasets.py`**: Dataset download. **`download_geofno_from_drive.sh`**: Shell wrapper for Google Drive download.
- **`train_all_datasets.py`**: Runs training for all dataset types. **`run_all_experiments.py`**: Runs experiments sequentially.
- **`plot_predictions.py`**, **`plot_all_datasets.py`**, **`visualize_geofno_datasets.py`**: Plotting. **`monitor_training.py`**, **`monitor_training.sh`**: Training monitoring.
- **`compare_with_geofno_paper.py`**: Compares with Geo-FNO paper results. **`test_geofno_loader.py`**: Loader tests. **`data_utils.py`**, **`pca_utils.py`**: Data and PCA utilities.

## Datasets

Geo-FNO datasets (elasticity, airfoil, pipe, plasticity, car-cfd, channel-shocks). Sources: [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8), [Geo-FNO GitHub](https://github.com/neuraloperator/Geo-FNO). Place under `$SCRATCH/data/Geo-FNO/data/` (e.g. `diffeo/`, `naca/`).

## Usage

```bash
# Train (required: --dataset_path, --dataset_type)
python train_geofno.py --dataset_path $SCRATCH/data/Geo-FNO/data/diffeo --dataset_type elasticity
python train_geofno.py --dataset_path $SCRATCH/data/Geo-FNO/data/naca --dataset_type airfoil

# Download
python download_geofno_datasets.py
# or
./download_geofno_from_drive.sh
```

## .sh, .md, .txt in this directory

- **.sh**: `download_geofno_from_drive.sh`, `monitor_training.sh`
- **.md**: `README.md`, `COMPARISON_RESULTS.md`
- **.txt**: `comparison_results.txt`, `comparison_run.txt` (generated/output)
