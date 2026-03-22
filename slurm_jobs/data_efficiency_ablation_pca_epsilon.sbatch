#!/bin/bash
#SBATCH --job-name=eff_pca_eps
#SBATCH --account=PAS3244
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=d.vod@posteo.de

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda3
source activate multilayer-RBNO_env

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT=/users/PAS3244/dvoderhol/multilayer_RBNO
SCRIPT=$REPO_ROOT/multilayer_RBNO/data_efficiency/run_ablation_pca_epsilon.py
DATA=/users/PAS3244/dvoderhol/data/nonlinear_diffusion_dataset_big.npz

mkdir -p "$REPO_ROOT/slurm_logs"
mkdir -p /users/PAS3244/dvoderhol/data/data_efficiency/logs
mkdir -p /users/PAS3244/dvoderhol/data/data_efficiency/checkpoints

# ── Run ───────────────────────────────────────────────────────────────────────
# Sweeps PCA variance cutoff (epsilon): 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
# Each epsilon runs Stage 1 + Stage 2 sequentially.
# Results saved under /users/PAS3244/dvoderhol/data/data_efficiency/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/multilayer_RBNO/models:$PYTHONPATH"
export ABLATION_PYTHON=python

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "$SCRIPT" \
    --data_path "$DATA" \
    --epochs 500 \
    --batch_size 32 \
    --device cuda

echo "Job finished: $(date)"
