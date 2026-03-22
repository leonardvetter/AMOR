#!/bin/bash
#SBATCH --job-name=kflow_Re40
#SBATCH --account=PAS3244
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=d.vod@posteo.de

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda3
source activate multilayer-RBNO_env

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT=/users/PAS3244/dvoderhol/multilayer_RBNO
SCRIPT=$REPO_ROOT/multilayer_RBNO/scripts/train_mrbno_kolmogorov_Re40.py
DATA=/users/PAS3244/dvoderhol/data/2D_NS_Re40.npy
CKPT_DIR=/users/PAS3244/dvoderhol/data/checkpoints

mkdir -p "$CKPT_DIR"
mkdir -p "$REPO_ROOT/slurm_logs"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/multilayer_RBNO/models:$REPO_ROOT/multilayer_RBNO/models/old:$PYTHONPATH"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "$SCRIPT" \
    --data_path "$DATA" \
    --checkpoint_dir "$CKPT_DIR" \
    --epochs 300 \
    --batch_size 16 \
    --device cuda

echo "Job finished: $(date)"
