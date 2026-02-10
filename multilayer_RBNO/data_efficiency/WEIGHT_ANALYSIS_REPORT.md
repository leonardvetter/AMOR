# Weight Similarity Analysis Report

## Summary

This report analyzes weight similarities between Stage 1, Stage 2 (v1-v3), and Baseline models to understand why Stage 2 performs significantly worse than the baseline.

## Test Performance Comparison

- **Baseline**: Rel L1 = 1.10e-02, Rel L2 = 1.22e-02
- **Stage 2 v1**: Rel L1 = ~9.0e-02, Rel L2 = ~9.8e-02 (8-9x worse)
- **Stage 2 v2**: Rel L1 = ~9.0e-02, Rel L2 = ~9.8e-02 (8-9x worse)
- **Stage 2 v3**: Rel L1 = ~9.0e-02, Rel L2 = ~9.8e-02 (8-9x worse)

## Weight Similarity Results

### 1. PCA Space Weights: Stage 1 vs Baseline

**Average Cosine Similarity: 0.85**

| Weight | Cosine Sim | Rel L2 Dist | L2 Dist |
|--------|------------|-------------|---------|
| block0.weight | 0.893 | 0.670 | 839.4 |
| block1.weight | 0.802 | 0.861 | 1107.1 |
| blocks.0.weight | 0.817 | 0.791 | 850.0 |
| blocks.1.weight | 0.819 | 0.733 | 1038.5 |
| blocks.2.weight | 0.912 | 0.505 | 956.8 |
| **AVERAGE** | **0.849** | **0.712** | **958.4** |

**Analysis**: Stage 1 PCA space weights are reasonably similar to baseline (cosine similarity ~0.85), but not identical. This suggests that training in PCA space alone doesn't perfectly capture the optimal PCA-space representation that emerges from full-space training.

### 2. PCA Space Weights: Stage 2 (v1-v3) vs Baseline

**Average Cosine Similarity: ~0.06 (very low)**

All Stage 2 versions show very low cosine similarity (~0.06) with baseline PCA space weights, indicating significant divergence. The relative L2 distances are ~1.9x, meaning Stage 2 PCA weights are almost twice as far from baseline as the baseline weights are from zero.

**Key Observation**: During Stage 2 fine-tuning, the PCA space weights (which were frozen in v1, or trained with lower LR in v2, or trained equally in v3) diverge significantly from the baseline. This suggests that:

1. The PCA space representation learned in Stage 1 is not stable when fine-tuning in real space
2. There may be a conflict between optimizing in PCA space vs real space
3. The limited Stage 2 data (10% of train/val) may not be sufficient to maintain the PCA space representation quality

### 3. Real Space Weights: Stage 2 (v1-v3) vs Baseline

**Mixed Results with Low Similarities**

- **skip_conv weights**: Very low cosine similarity (often negative), high relative L2 distances (~1.5-2.0x)
- **global_skip_proj**: Varies by version (v1: -0.13, v2: 0.98, v3: -0.88)
- **lift/proj weights**: Low to moderate similarity, high relative distances

**Analysis**: The real space weights (lift, proj, skip connections) are also quite different from baseline, suggesting that Stage 2 training doesn't converge to the same solution as baseline training.

## Why Stage 2 Performs Worse: Hypotheses

### 1. **PCA Space Instability**
The PCA space weights learned in Stage 1 (cosine sim ~0.85 with baseline) are not optimal for the full model. When fine-tuning in Stage 2, these weights either:
- Remain frozen (v1) but are suboptimal
- Get updated (v2, v3) but diverge from the optimal solution

### 2. **Limited Stage 2 Data**
Stage 2 uses only 10% of the training data. This may be insufficient to:
- Learn good real-space mappings (lift/proj)
- Maintain PCA space quality if weights are updated
- Learn effective skip connections

### 3. **Training Objective Mismatch**
- Stage 1 optimizes in PCA space (reduced dimensionality)
- Stage 2 optimizes in real space (full dimensionality)
- The optimal solution in PCA space may not translate well to real space

### 4. **Initialization Issues**
The lift/proj layers are initialized randomly or with minimal PCA-based initialization. These layers are critical for mapping between PCA and real space, and poor initialization may lead to suboptimal solutions.

### 5. **Optimization Landscape**
The two-stage approach may get trapped in a local minimum that's different from the global minimum found by baseline training.

## Suggestions for Improvement

### 1. **Better Lift/Proj Initialization**
- Initialize lift/proj to better approximate PCA encode/decode operations
- Use learned PCA components to initialize these layers more intelligently

### 2. **Joint Training Approach**
- Instead of strict two-stage training, use a curriculum learning approach:
  - Start with high weight on PCA space loss, gradually shift to real space loss
  - Or use a multi-task loss combining both PCA and real space objectives

### 3. **More Stage 2 Data**
- Increase the Stage 2 data split (e.g., 20-30% instead of 10%)
- Or use data augmentation for Stage 2

### 4. **Regularization**
- Add regularization to keep PCA space weights close to Stage 1 weights during Stage 2
- Use knowledge distillation from a baseline model

### 5. **Different Architecture for Lift/Proj**
- Use more expressive architectures for lift/proj (e.g., small MLPs instead of simple convolutions)
- Allow these layers to learn more complex mappings

### 6. **Warmup and Learning Rate Scheduling**
- Use longer warmup periods for real-space layers
- Use different learning rates for different layer groups
- Consider cosine annealing with restarts

### 7. **Progressive Unfreezing Refinement**
- Version 4 (progressive unfreezing) may help, but needs careful tuning of unfreezing schedule
- Consider unfreezing from output to input (reverse order)

### 8. **Ensemble or Multi-Scale Training**
- Train multiple Stage 2 models with different random seeds and ensemble
- Use multi-scale features in real space

## Next Steps

1. Monitor v5-7 reruns to see if they improve performance
2. Complete ablation studies to understand:
   - Impact of PCA epsilon (dimensionality) on performance
   - Impact of data split ratio on performance
3. Experiment with improved lift/proj initialization
4. Try joint training approaches
5. Increase Stage 2 data fraction
