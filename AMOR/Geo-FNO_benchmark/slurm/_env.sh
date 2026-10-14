#!/bin/bash
# Shared setup for the Geo-FNO jobs (sourced, not submitted). Submit from the repo root (or set REPO_DIR).
#   REPO_DIR  repo checkout            (default: $SLURM_SUBMIT_DIR)
#   DATA_DIR  contains Airfoil_dataset/ and Elasticity_dataset/   (default: $REPO_DIR/AMOR/data)
#   OUT_DIR   results, checkpoints, predictions                   (default: $REPO_DIR/experiments_out/geofno)
set -u
module purge
module load jupyter/ai        # PyTorch, numpy, scipy, scikit-learn, pandas, matplotlib

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/AMOR/data}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/experiments_out/geofno}"

[ -f "$REPO_DIR/AMOR/Geo-FNO_benchmark/run_geofno.py" ] || { echo "ERROR: AMOR/Geo-FNO_benchmark/run_geofno.py not found under REPO_DIR=$REPO_DIR (submit from the repo root or set REPO_DIR)"; exit 2; }
for f in Airfoil_dataset/NACA_Cylinder_X.npy Airfoil_dataset/NACA_Cylinder_Y.npy Airfoil_dataset/NACA_Cylinder_Q.npy \
         Elasticity_dataset/Random_UnitCell_XY_10.npy Elasticity_dataset/Random_UnitCell_sigma_10.npy; do
  [ -f "$DATA_DIR/$f" ] || { echo "ERROR: missing $DATA_DIR/$f"; exit 2; }
done
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
cd "$REPO_DIR"

echo "job $SLURM_JOB_ID  task ${SLURM_ARRAY_TASK_ID:-none}  node $SLURM_NODELIST  start $(date)"
echo "REPO_DIR=$REPO_DIR  DATA_DIR=$DATA_DIR  OUT_DIR=$OUT_DIR"
echo "git: $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) modified/untracked"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,sklearn,scipy,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"
