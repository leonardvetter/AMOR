#!/bin/bash
#SBATCH --job-name=geofno
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
# AMOR on Geo-FNO airfoil (array tasks 0-4) and elasticity (5-9); task t -> seed t%5. Each task trains
# 500 epochs and writes $OUT_DIR/results/<dataset>_seed<S>.json (finished tasks are skipped on resubmission).
# Submit from the repo root:   sbatch AMOR/Geo-FNO_benchmark/slurm/array.sh
# Subset of tasks:             sbatch --array=0,5 AMOR/Geo-FNO_benchmark/slurm/array.sh
ENVF="${REPO_DIR:-$SLURM_SUBMIT_DIR}/AMOR/Geo-FNO_benchmark/slurm/_env.sh"  # Slurm runs a spooled copy of this script
[ -f "$ENVF" ] || { echo "ERROR: $ENVF not found. Submit from the repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR) or set REPO_DIR, and pull the latest files."; exit 2; }
source "$ENVF" || exit 2
DATASETS=(airfoil elasticity)
DS="${DATASETS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 5))
python -u AMOR/Geo-FNO_benchmark/run_geofno.py --dataset "$DS" --seed "$SEED" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
code=$?
echo "$DS seed $SEED exit $code  end $(date)"
exit $code
