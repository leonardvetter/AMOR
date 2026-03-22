#!/bin/bash
#SBATCH --job-name=helmholtz_all
#SBATCH --account=PAS3244
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=d.vod@posteo.de

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda3
source activate multilayer-RBNO_env

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT=/users/PAS3244/dvoderhol/multilayer_RBNO
SCRIPT=$REPO_ROOT/multilayer_RBNO/scripts/train_all_helmholtz_models.py

mkdir -p "$REPO_ROOT/slurm_logs"
mkdir -p /users/PAS3244/dvoderhol/data/helmholtz_model_comparison

# ── Run ───────────────────────────────────────────────────────────────────────
# Trains MRBNO, FNO, and PCANet on helmholtz_dataset{,_smooth,_rough}.npz
# Results saved to /users/PAS3244/dvoderhol/data/helmholtz_model_comparison/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/multilayer_RBNO/models:$PYTHONPATH"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "$SCRIPT"

echo "Job finished: $(date)"
