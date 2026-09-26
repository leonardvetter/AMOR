"""Learnable orthonormal bases for AMOR blocks.

Row-orthonormal (k, D) bases with encode/decode. With ``mass_diag`` the
orthonormality and the encode inner product are M-weighted (Phi M Phi^T = I),
which is what FEM meshes with a lumped nodal mass matrix require; without it
both are Euclidean.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnableOrthonormalBasis(nn.Module):
    """Base: row-orthonormal (k, D) basis with shared encode/decode."""

    def __init__(
        self,
        dim: int,
        k: int,
        init_components: Optional[np.ndarray] = None,
        init_mean: Optional[np.ndarray] = None,
        mass_diag: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        if init_mean is not None:
            self.register_buffer("mean", torch.from_numpy(init_mean.astype(np.float32)))
        else:
            self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        if mass_diag is not None:
            self.register_buffer(
                "mass_diag",
                torch.from_numpy(np.asarray(mass_diag, dtype=np.float32).reshape(-1)),
            )
        else:
            self.mass_diag = None

    def components(self) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor, d: int) -> torch.Tensor:
        c = self.components()
        if self.mass_diag is None:
            return (x - self.mean[:d]) @ c[:, :d].T
        w = self.mass_diag[:d]
        return ((x - self.mean[:d]) * w.unsqueeze(0)) @ c[:, :d].transpose(0, 1)

    def decode(self, z: torch.Tensor, d_out: int) -> torch.Tensor:
        c = self.components()
        return z @ c[:, :d_out] + self.mean[:d_out]

    def _qr_rows(self, raw: torch.Tensor) -> torch.Tensor:
        """Orthonormal rows of ``raw``; M-orthonormal when mass weights are set.

        Mass case: QR of (R * sqrt(m)), then Phi = Q^T / sqrt(m) so Phi M Phi^T = I.
        """
        if self.mass_diag is None:
            q, _ = torch.linalg.qr(raw.T, mode="reduced")
            return q.T
        sqrt_w = torch.sqrt(self.mass_diag[: raw.shape[1]].clamp(min=1e-30))
        q, _ = torch.linalg.qr((raw * sqrt_w).T, mode="reduced")
        return q.T / sqrt_w

    def _cayley_rotation(self, ref: torch.Tensor) -> torch.Tensor:
        """W = (I+S)^-1 (I-S) with S skew from the learnable ``s`` parameters."""
        skew = self.s - self.s.T
        eye = torch.eye(self.k, device=ref.device, dtype=ref.dtype)
        return torch.linalg.solve(eye + skew, eye - skew)


class QROrthonormalBasis(LearnableOrthonormalBasis):
    """Learn a (k, D) raw matrix; QR each forward for orthonormal rows."""

    def __init__(
        self,
        dim: int,
        k: int,
        init_components: Optional[np.ndarray] = None,
        init_mean: Optional[np.ndarray] = None,
        mass_diag: Optional[np.ndarray] = None,
    ):
        super().__init__(dim, k, init_components, init_mean, mass_diag)
        if init_components is not None:
            raw = torch.from_numpy(init_components.astype(np.float32))
        else:
            raw = torch.randn(k, dim) * 0.01
        self.raw = nn.Parameter(raw)

    def components(self) -> torch.Tensor:
        return self._qr_rows(self.raw)


class CayleyOrthonormalBasis(LearnableOrthonormalBasis):
    """QR span plus a k x k Cayley rotation W=(I+S)^-1(I-S), S skew and learnable."""

    def __init__(
        self,
        dim: int,
        k: int,
        init_components: Optional[np.ndarray] = None,
        init_mean: Optional[np.ndarray] = None,
        mass_diag: Optional[np.ndarray] = None,
    ):
        super().__init__(dim, k, init_components, init_mean, mass_diag)
        if init_components is not None:
            raw = torch.from_numpy(init_components.astype(np.float32))
        else:
            raw = torch.randn(k, dim) * 0.01
        self.raw = nn.Parameter(raw)
        self.s = nn.Parameter(torch.zeros(k, k))

    def components(self) -> torch.Tensor:
        r = self._qr_rows(self.raw)
        return self._cayley_rotation(r) @ r


class SpanFrozenCayleyBasis(LearnableOrthonormalBasis):
    """Learned k x k Cayley rotation of a frozen orthonormal span (``q0`` buffer).

    Feasible at 3D resolutions where a trainable (k, D) matrix and its Adam state
    do not fit in memory. Encode/decode apply the rotation in coefficient space so
    autograd never materializes ``w @ q0`` per block.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        init_components: Optional[np.ndarray] = None,
        init_mean: Optional[np.ndarray] = None,
        mass_diag: Optional[np.ndarray] = None,
    ):
        if mass_diag is not None:
            raise ValueError("SpanFrozenCayleyBasis does not support mass weighting")
        super().__init__(dim, k, init_components, init_mean, mass_diag)
        if init_components is None:
            raise ValueError("SpanFrozenCayleyBasis requires init_components")
        q, _ = np.linalg.qr(init_components.T.astype(np.float64), mode="reduced")
        self.register_buffer("q0", torch.from_numpy(q.T.astype(np.float32)))
        self.s = nn.Parameter(torch.zeros(k, k))

    def components(self) -> torch.Tensor:
        return self._cayley_rotation(self.q0) @ self.q0

    def encode(self, x: torch.Tensor, d: int) -> torch.Tensor:
        z0 = (x - self.mean[:d]) @ self.q0[:, :d].T
        return z0 @ self._cayley_rotation(self.q0).T

    def decode(self, z: torch.Tensor, d_out: int) -> torch.Tensor:
        z0 = z @ self._cayley_rotation(self.q0)
        return z0 @ self.q0[:, :d_out] + self.mean[:d_out]


def make_learnable_basis(
    param: str,
    dim: int,
    k: int,
    init_components: Optional[np.ndarray] = None,
    init_mean: Optional[np.ndarray] = None,
    mass_diag: Optional[np.ndarray] = None,
) -> LearnableOrthonormalBasis:
    if param == "qr":
        return QROrthonormalBasis(dim, k, init_components, init_mean, mass_diag)
    if param == "cayley":
        return CayleyOrthonormalBasis(dim, k, init_components, init_mean, mass_diag)
    if param in ("cayley_frozen", "qr_frozen"):
        return SpanFrozenCayleyBasis(dim, k, init_components, init_mean, mass_diag)
    raise ValueError(
        f"Unknown learnable basis param {param!r}; expected qr, cayley, cayley_frozen/qr_frozen"
    )
