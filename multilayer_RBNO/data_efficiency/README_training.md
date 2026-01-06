# Two-Stage Training Execution Instructions

This document describes how to execute the two-stage training pipeline and obtain the requested metrics.

## Metrics to Report

1. Dataset used and memory usage
2. PCA compressed dataset memory usage  
3. Final error after Stage 2 in physical space
4. Comparison with baseline model trained on full dataset

## Execution

The script `run_training_with_metrics.py` has been created to track all metrics. To execute:

```bash
cd /global/homes/d/dvod/multilayer_RBNO/multilayer_RBNO/data_efficiency
python3 run_training_with_metrics.py --data_path $SCRATCH/data/2D_NS_Re500.npy
```

For faster testing with fewer epochs:
```bash
python3 run_training_with_metrics.py --stage1_epochs 50 --stage2_epochs 20 --baseline_epochs 50
```

For full training:
```bash
python3 run_training_with_metrics.py --stage1_epochs 300 --stage2_epochs 100 --baseline_epochs 300
```

## Expected Output

The script will report:
- Original dataset memory (GB)
- PCA compressed dataset memory (GB)  
- Memory reduction factor
- Stage 2 test error (physical space)
- Baseline test error (full dataset)
- Error ratio comparison

## Notes

- Ensure PyTorch and required packages are installed in your Python environment
- The dataset path should point to the 16GB 2D_NS_Re500.npy file
- Training will take several hours for full epochs
- Baseline training can be skipped with --skip_baseline flag

