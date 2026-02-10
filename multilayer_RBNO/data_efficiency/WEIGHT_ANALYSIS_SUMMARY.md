# Weight Similarity Analysis Summary

## Critical Findings

### 1. Loss Computation Verification
✓ **CONFIRMED**: Stage 2 loss is computed in **real space** (not PCA space)
- Stage 1: `loss = relative_l1_loss(pred_pca, yb_pca)` - PCA space
- Stage 2: `loss = relative_l1_loss(pred, yb)` - Real space

### 2. Stage 1 vs Stage 2 v1 PCA Weights - CRITICAL ISSUE
✗ **PROBLEM**: Stage 1 and Stage 2 v1 PCA weights are **NOT identical** (cosine sim ~0.08)
- Expected: Should be identical since v1 freezes blocks
- Actual: All block weights differ significantly
- **This suggests blocks may not be properly frozen, or weights are being modified during Stage 2 initialization**

### 3. Weight Similarity Breakdown by Layer Type

#### Stage 2 v1 (Frozen Blocks):
- **Lift/Proj Layers**: Cosine sim = -0.055 (very poor, often negative)
  - `lift.weight`: -0.197
  - `proj1.weight`: -0.014
  - `proj2.weight`: 0.125
  - `global_skip_proj.weight`: -0.133
- **Block Skip Conv Layers**: Cosine sim = 0.109 (poor)
  - Range: -0.029 to 0.367
- **Global Skip**: Cosine sim = -0.133 (very poor)

#### Stage 2 v2 (Lower LR for blocks):
- **Lift/Proj Layers**: Cosine sim = 0.186 (slightly better)
  - `global_skip_proj.weight`: 0.975 (excellent!)
  - `lift.weight`: 0.141
  - `proj1.weight`: 0.020
  - `proj2.weight`: -0.394
- **Block Skip Conv Layers**: Cosine sim = 0.111 (similar to v1)
- **Global Skip**: Cosine sim = 0.975 (excellent)

#### Stage 2 v3 (Equal LR):
- **Lift/Proj Layers**: Cosine sim = -0.142 (worse than v1)
  - `global_skip_proj.weight`: -0.879 (very poor)
  - `lift.weight`: 0.279 (best among lift/proj)
  - `proj1.weight`: -0.003
  - `proj2.weight`: 0.036
- **Block Skip Conv Layers**: Cosine sim = 0.109 (similar to v1/v2)
- **Global Skip**: Cosine sim = -0.879 (very poor)

## Key Insights

1. **Lift/Proj layers are the main problem**: Very low cosine similarity (often negative), indicating these layers learn very different mappings than baseline
2. **Block skip conv layers are also problematic**: Low similarity (~0.11) across all versions
3. **Global skip is inconsistent**: v2 has excellent similarity (0.975), but v1 and v3 have poor similarity
4. **Version 2 performs best** in terms of weight similarity for lift/proj and global skip, but still worse than baseline in test performance

## Recommendations for Training Lift/Proj Layers

### Option 1: Train Only Lift/Proj First (Recommended)
Create a new Stage 2 version that:
1. Freezes all blocks and skip connections
2. Trains ONLY lift and proj layers for first N epochs (e.g., 100-200 epochs)
3. Then unfreezes skip connections and continues training
4. Optionally unfreezes blocks later

**Rationale**: Lift/proj are critical for mapping between real space and PCA space. They need to learn good mappings before other layers can be optimized effectively.

### Option 2: Higher Weight for Lift/Proj Loss
Modify loss function to weight lift/proj contributions more heavily:
```python
# Compute loss with higher weight for lift/proj gradient contributions
loss = relative_l1_loss(pred, yb)
# Add explicit regularization or higher learning rate for lift/proj
```

### Option 3: Better Initialization + Higher LR for Lift/Proj
- Initialize lift/proj from PCA components (improve `initialize_lift_proj_from_pca`)
- Use higher learning rate for lift/proj (e.g., 10x) compared to other layers
- Use longer warmup specifically for lift/proj

### Option 4: Two-Phase Training
1. **Phase 1**: Train lift/proj only with frozen blocks (100-200 epochs)
2. **Phase 2**: Unfreeze skip connections, continue training (remaining epochs)
3. **Phase 3** (optional): Unfreeze blocks with lower LR

### Option 5: Auxiliary Loss on PCA Space
Add auxiliary loss that encourages lift/proj to produce good PCA representations:
```python
# Main loss in real space
loss_real = relative_l1_loss(pred, yb)
# Auxiliary loss: ensure lift produces good PCA representation
pred_pca = encode_to_pca(pred)  # or intermediate representation
target_pca = encode_to_pca(yb)
loss_pca = relative_l1_loss(pred_pca, target_pca)
loss = loss_real + alpha * loss_pca
```

## Implementation Priority

1. **First**: Fix the frozen blocks issue in Stage 2 v1 (verify blocks are actually frozen)
2. **Second**: Implement Option 1 (train lift/proj only first) as a new version
3. **Third**: Try Option 3 (better initialization + higher LR for lift/proj)
4. **Fourth**: Experiment with Option 5 (auxiliary PCA loss)
