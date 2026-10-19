"""Train/validation/test splits.

Percentages stay fixed. The run seed permutes which samples fall in each part,
and the same seed is used for initialization, batch order, and basis RNGs.
Model width, depth, and mode counts do not depend on the seed.
"""
from __future__ import annotations

import numpy as np
import torch

TRAIN_FRAC = 0.7
VAL_FRAC = 0.2
SPLIT_RULE = (
    "permutation with the run seed, fractions 0.7/0.2/remainder; "
    "the same seed sets initialization, batch order, and basis RNGs; "
    "architecture does not depend on the seed"
)


def seeded_split(
    n: int,
    seed: int,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
):
    """Return train, validation, and test index arrays."""
    n = int(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    if n_train < 1 or n_train + n_val >= n:
        raise ValueError(f"split of n={n} leaves an empty part (train={n_train}, val={n_val})")
    perm = np.random.default_rng(int(seed)).permutation(n)
    train = perm[:n_train]
    val = perm[n_train:n_train + n_val]
    test = perm[n_train + n_val:]
    return train, val, test


def shuffle_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def apply_weight_seed(seed: int) -> None:
    """Seed torch and NumPy from the run seed. Does not change architecture."""
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
