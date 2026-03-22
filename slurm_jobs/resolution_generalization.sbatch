#!/bin/bash
#SBATCH --job-name=res_generalization
#SBATCH --account=PAS3244
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=d.vod@posteo.de

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda3
source activate multilayer-RBNO_env

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT=/users/PAS3244/dvoderhol/multilayer_RBNO
SCRIPT=$REPO_ROOT/multilayer_RBNO/mesh_generalization/test_mesh_generalization.py
DATA_DIR=/users/PAS3244/dvoderhol/data

mkdir -p "$REPO_ROOT/slurm_logs"
mkdir -p "$DATA_DIR/mesh_generalization_results"

# ── Run ───────────────────────────────────────────────────────────────────────
# Trains FNO, MRBNO, and PCANet on nonlinear_diffusion_dataset.npz (base resolution),
# then evaluates mesh-transfer performance on nonlinear_diffusion_dataset_res{N}.npz
# for multiple target resolutions. Checkpoints are cached in $DATA_DIR/mesh_generalization_results/.
#
# NOTE: Resolution-variant datasets must be pre-generated and placed at:
#   $DATA_DIR/nonlinear_diffusion_dataset_res<N>.npz
# e.g. res16, res32, res64, res128 — generate with the scripts in multilayer_RBNO/data/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/multilayer_RBNO/models:$PYTHONPATH"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "$SCRIPT"

echo "Job finished: $(date)"
