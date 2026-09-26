"""Helpers: map Roseberry DIS rows from physical Jacobian space to training-normalized coords."""

from __future__ import annotations

import numpy as np


def channel_std_flat(x_std: np.ndarray, c_in: int, spatial_size: int) -> np.ndarray:
    """Per-channel std -> flat (c_in * spatial_size,) matching (S,C,*,*) -> (S,-1) row-major."""
    s = np.asarray(x_std, dtype=np.float64).reshape(-1)
    if s.size == c_in:
        return np.repeat(s, spatial_size)
    if s.size == c_in * spatial_size:
        return s
    raise ValueError(f"expected std length {c_in} or {c_in * spatial_size}, got {s.size}")


def scale_dis_rows_to_normalized(
    rows_in: np.ndarray,
    rows_out: np.ndarray,
    *,
    x_std_flat: np.ndarray,
    y_std_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical DIS rows into z-score coords for shared encode/decode φ.

    AMOR uses the same φ for encode and decode on x_n = x_phys / std:
      decode:  x_n_hat = φ @ z  ⇒  φ ≈ ψ_phys / std
      (encode-only IP would want ψ_phys * std; that warps decode when φ shared.)

    So row_norm = row_phys / std (not * std).
    """
    x_std_flat = np.asarray(x_std_flat, dtype=np.float64).reshape(1, -1)
    y_std_flat = np.asarray(y_std_flat, dtype=np.float64).reshape(1, -1)
    if rows_in.shape[1] != x_std_flat.shape[1]:
        raise ValueError(f"rows_in width {rows_in.shape[1]} != x_std {x_std_flat.shape[1]}")
    if rows_out.shape[1] != y_std_flat.shape[1]:
        raise ValueError(f"rows_out width {rows_out.shape[1]} != y_std {y_std_flat.shape[1]}")
    return rows_in / np.maximum(x_std_flat, 1e-30), rows_out / np.maximum(y_std_flat, 1e-30)
