"""Random orthonormal rows (pad / learned-random init)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .common import orthonormal_rows


def random_orthonormal_rows(d: int, k: int, random_state: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    raw = rng.standard_normal((k, d))
    return orthonormal_rows(raw, k)
