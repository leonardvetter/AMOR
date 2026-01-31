#!/bin/bash
# Generate 10k samples for all Helmholtz and nonlinear datasets

module load python
module load pytorch

cd multilayer_RBNO/data

# Nonlinear datasets (NUM_SAMPLES is set in the script)
echo "Generating nonlinear_diffusion_dataset.npz (10k samples)..."
python3 create_nonlinear_dataset.py

echo "Generating nonlinear_diffusion_dataset_smooth.npz (10k samples)..."
python3 generate_nonlinear_smooth.py

echo "Generating nonlinear_diffusion_dataset_rough.npz (10k samples)..."
python3 generate_nonlinear_rough.py

# Helmholtz datasets (use --num_samples argument)
echo "Generating helmholtz_dataset_smooth.npz (10k samples)..."
conda run -n test_fenics python3 generate_helmholtz_smooth.py --num_samples 10000 || python3 generate_helmholtz_smooth.py --num_samples 10000

echo "Generating helmholtz_dataset_rough.npz (10k samples)..."
conda run -n test_fenics python3 generate_helmholtz_rough.py --num_samples 10000 || python3 generate_helmholtz_rough.py --num_samples 10000

echo "All datasets generated!"
