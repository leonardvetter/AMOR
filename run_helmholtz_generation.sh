#!/bin/bash
# Wrapper script to run Helmholtz generation with proper environment

SCRIPT_NAME=$1
NUM_SAMPLES=$2

if [ -z "$SCRIPT_NAME" ] || [ -z "$NUM_SAMPLES" ]; then
    echo "Usage: $0 <script_name> <num_samples>"
    exit 1
fi

# Set up environment
export XDG_CACHE_HOME=$SCRATCH/dolfin_cache
export DOLFIN_CACHE_DIR=$SCRATCH/dolfin_cache
export PKG_CONFIG_PATH=$HOME/.conda/envs/test_fenics/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/.conda/envs/test_fenics/lib:$LD_LIBRARY_PATH
mkdir -p $SCRATCH/dolfin_cache

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
~/.conda/envs/test_fenics/bin/python3 "$SCRIPT_NAME" --num_samples "$NUM_SAMPLES"
