#!/bin/bash
#SBATCH --job-name=geofno_smoke
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
# Tiny end-to-end check (3 epochs, 200 training samples, both datasets). Submit from the repo root:
#   sbatch AMOR/Geo-FNO_benchmark/slurm/smoke.sh
ENVF="${REPO_DIR:-$SLURM_SUBMIT_DIR}/AMOR/Geo-FNO_benchmark/slurm/_env.sh"  # Slurm runs a spooled copy of this script
[ -f "$ENVF" ] || { echo "ERROR: $ENVF not found. Submit from the repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR) or set REPO_DIR, and pull the latest files."; exit 2; }
source "$ENVF" || exit 2
python -u AMOR/Geo-FNO_benchmark/run_geofno.py --smoke --force --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
echo "smoke exit $?  end $(date)"
