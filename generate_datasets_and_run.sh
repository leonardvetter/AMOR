#!/bin/bash
set -e

SCRATCH_DIR=${SCRATCH:-/tmp}
DATA_DIR="$SCRATCH_DIR/data"
cd "$(dirname "$0")/multilayer_RBNO/data"

echo "Generating smooth nonlinear dataset..."
python3 generate_nonlinear_smooth.py || echo "Warning: Could not generate smooth nonlinear dataset (may need fipy)"

echo "Generating rough nonlinear dataset..."
python3 generate_nonlinear_rough.py || echo "Warning: Could not generate rough nonlinear dataset (may need fipy)"

echo "Generating smooth helmholtz dataset..."
python3 generate_helmholtz_smooth.py --num_samples 1000 || echo "Warning: Could not generate smooth helmholtz dataset (may need dolfin)"

echo "Generating rough helmholtz dataset..."
python3 generate_helmholtz_rough.py --num_samples 1000 || echo "Warning: Could not generate rough helmholtz dataset (may need dolfin)"

echo "Dataset generation complete!"
