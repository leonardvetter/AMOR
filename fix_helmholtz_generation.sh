#!/bin/bash
# Fix Helmholtz dataset generation environment

module load python
module load pytorch

# Set up environment variables (same as before)
export XDG_CACHE_HOME=$SCRATCH/dolfin_cache
export DOLFIN_CACHE_DIR=$SCRATCH/dolfin_cache
mkdir -p $SCRATCH/dolfin_cache

# Set PYTHONPATH for hippylib and hippyflow
if [ -n "$HIPPYLIB_PATH" ]; then
    export PYTHONPATH=$HIPPYLIB_PATH:$PYTHONPATH
fi
if [ -n "$HIPPYFLOW_PATH" ]; then
    export PYTHONPATH=$HIPPYFLOW_PATH:$PYTHONPATH
fi

# Add $SCRATCH/python_packages if it exists
if [ -d "$SCRATCH/python_packages" ]; then
    export PYTHONPATH=$SCRATCH/python_packages:$PYTHONPATH
fi

cd multilayer_RBNO/data

# Generate Helmholtz datasets
echo "Generating helmholtz_dataset_smooth.npz (10k samples)..."
python3 generate_helmholtz_smooth.py --num_samples 10000

echo "Generating helmholtz_dataset_rough.npz (10k samples)..."
python3 generate_helmholtz_rough.py --num_samples 10000

echo "Helmholtz datasets generated!"
