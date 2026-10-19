# Memory-efficiency experiment

`run_memory_efficiency.py` is the two-stage run.

Stage 1 loads PCA coefficients only and trains each block's factorized coefficient map. The loss is relative L1 in that coefficient space. Stage 2 adds the rest of AMOR, freezes those coefficient maps, and trains the remaining weights on full fields with relative L1.

One-stage runs use the same data prefixes with every weight trainable.
