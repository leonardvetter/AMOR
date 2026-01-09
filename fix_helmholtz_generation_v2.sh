#!/bin/bash
# Fix Helmholtz dataset generation with conda environment

# Set up environment variables
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
if [ -d "$SCRATCH/python_packages" ]; then
    export PYTHONPATH=$SCRATCH/python_packages:$PYTHONPATH
fi

cd multilayer_RBNO/data

# Try with conda environment first, fallback to regular python
echo "Generating helmholtz_dataset_smooth.npz (10k samples)..."
conda run -n test_fenics python3 generate_helmholtz_smooth.py --num_samples 10000 2>&1 || python3 generate_helmholtz_smooth.py --num_samples 10000

echo "Generating helmholtz_dataset_rough.npz (10k samples)..."
conda run -n test_fenics python3 generate_helmholtz_rough.py --num_samples 10000 2>&1 || python3 generate_helmholtz_rough.py --num_samples 10000

echo "Helmholtz datasets generated!"
