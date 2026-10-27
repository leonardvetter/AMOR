"""Learnable orthonormal bases for AMOR blocks.

Row-orthonormal (k, D) bases with encode/decode. With a mass matrix the
orthonormality and the encode inner product are M-weighted (Φ M Φᵀ = I).
``mass_matrix`` may be any SPD sparse/dense operator; a 1-d vector is treated
as a lumped diagonal only as a fallback.
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
        mass_matrix=None,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        if init_mean is not None:
            self.register_buffer("mean", torch.from_numpy(init_mean.astype(np.float32)))
        else:
            self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        # Prefer general sparse/dense M; keep lumped diag buffer only as fallback.
        object.__setattr__(self, "_mass_sp", None)
        if mass_matrix is not None:
            import scipy.sparse as sp

            M = mass_matrix
            if not sp.issparse(M):
                M = sp.csr_matrix(np.asarray(M, dtype=np.float64))
            else:
                M = M.tocsr()
            object.__setattr__(self, "_mass_sp", M)
            # Row-sum lumping stored for QR whitening init only when needed.
            lump = np.asarray(M.sum(axis=1)).reshape(-1).astype(np.float32)
            self.register_buffer("mass_diag", torch.from_numpy(lump))
        elif mass_diag is not None:
            self.register_buffer(
                "mass_diag",
                torch.from_numpy(np.asarray(mass_diag, dtype=np.float32).reshape(-1)),
            )
        else:
            self.mass_diag = None
        self._eval_components = None

    def train(self, mode: bool = True):
        if mode:
            self._eval_components = None
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._eval_components = None
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, *args, **kwargs):
        # A parent module's load_state_dict reaches this, not load_state_dict above.
        self._eval_components = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _components_cached(self, build):
        """Reuse the orthonormal rows at inference. Training recomputes them."""
        if not self.training and self._eval_components is not None:
            return self._eval_components
        computed = build()
        if self.training:
            return computed
        self._eval_components = computed.detach()
        return self._eval_components

    def components(self) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor, d: int) -> torch.Tensor:
        c = self.components()
        x_c = x - self.mean[:d]
        M = getattr(self, "_mass_sp", None)
        if M is not None:
            xc = x_c.detach().cpu().numpy()
            Md = M[:d, :d] if (M.shape[0] != d or M.shape[1] != d) else M
            Mx = Md @ xc.T  # (d, B)
            z = (c[:, :d].detach().cpu().numpy() @ Mx).T
            return torch.from_numpy(np.asarray(z, dtype=np.float32)).to(
                device=x.device, dtype=x.dtype
            )
        if self.mass_diag is None:
            return x_c @ c[:, :d].T
        w = self.mass_diag[:d]
        return (x_c * w.unsqueeze(0)) @ c[:, :d].transpose(0, 1)

    def decode(self, z: torch.Tensor, d_out: int) -> torch.Tensor:
        c = self.components()
        return z @ c[:, :d_out] + self.mean[:d_out]

    def _qr_rows(self, raw: torch.Tensor) -> torch.Tensor:
        """Orthonormal rows; M-orthonormal via Gram when a mass matrix is set."""
        M = getattr(self, "_mass_sp", None)
        if M is not None:
            from AMOR.models.dis_dual import m_orthonormalize_rows_general

            rows = raw.detach().cpu().numpy().astype(np.float64)
            Md = M[: rows.shape[1], : rows.shape[1]]
            out = m_orthonormalize_rows_general(rows, Md)
            return torch.from_numpy(out.astype(np.float32)).to(device=raw.device, dtype=raw.dtype)
        if self.mass_diag is None:
            q, _ = torch.linalg.qr(raw.T, mode="reduced")
            return q.T
        # Lumped-diag fallback (sqrt-whitened QR).
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
        mass_matrix=None,
    ):
        super().__init__(dim, k, init_components, init_mean, mass_diag, mass_matrix)
        if init_components is not None:
            raw = torch.from_numpy(init_components.astype(np.float32))
        else:
            raw = torch.randn(k, dim) * 0.01
        self.raw = nn.Parameter(raw)

    def components(self) -> torch.Tensor:
        return self._components_cached(lambda: self._qr_rows(self.raw))


class CayleyOrthonormalBasis(LearnableOrthonormalBasis):
    """QR span plus a k x k Cayley rotation W=(I+S)^-1(I-S), S skew and learnable."""

    def __init__(
        self,
        dim: int,
        k: int,
        init_components: Optional[np.ndarray] = None,
        init_mean: Optional[np.ndarray] = None,
        mass_diag: Optional[np.ndarray] = None,
        mass_matrix=None,
    ):
        super().__init__(dim, k, init_components, init_mean, mass_diag, mass_matrix)
        if init_components is not None:
            raw = torch.from_numpy(init_components.astype(np.float32))
        else:
            raw = torch.randn(k, dim) * 0.01
        self.raw = nn.Parameter(raw)
        self.s = nn.Parameter(torch.zeros(k, k))

    def components(self) -> torch.Tensor:
        def build():
            r = self._qr_rows(self.raw)
            return self._cayley_rotation(r) @ r

        return self._components_cached(build)


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
        mass_matrix=None,
    ):
        if mass_diag is not None or mass_matrix is not None:
            raise ValueError("SpanFrozenCayleyBasis does not support mass weighting")
        super().__init__(dim, k, init_components, init_mean, mass_diag, mass_matrix)
        if init_components is None:
            raise ValueError("SpanFrozenCayleyBasis requires init_components")
        q, _ = np.linalg.qr(init_components.T.astype(np.float64), mode="reduced")
        self.register_buffer("q0", torch.from_numpy(q.T.astype(np.float32)))
        self.s = nn.Parameter(torch.zeros(k, k))

    def _rotation(self) -> torch.Tensor:
        return self._components_cached(lambda: self._cayley_rotation(self.q0))

    def components(self) -> torch.Tensor:
        return self._rotation() @ self.q0

    def encode(self, x: torch.Tensor, d: int) -> torch.Tensor:
        z0 = (x - self.mean[:d]) @ self.q0[:, :d].T
        return z0 @ self._rotation().T

    def decode(self, z: torch.Tensor, d_out: int) -> torch.Tensor:
        z0 = z @ self._rotation()
        return z0 @ self.q0[:, :d_out] + self.mean[:d_out]


def make_learnable_basis(
    param: str,
    dim: int,
    k: int,
    init_components: Optional[np.ndarray] = None,
    init_mean: Optional[np.ndarray] = None,
    mass_diag: Optional[np.ndarray] = None,
    mass_matrix=None,
) -> LearnableOrthonormalBasis:
    kw = dict(
        dim=dim,
        k=k,
        init_components=init_components,
        init_mean=init_mean,
        mass_diag=mass_diag,
        mass_matrix=mass_matrix,
    )
    if param == "qr":
        return QROrthonormalBasis(**kw)
    if param == "cayley":
        return CayleyOrthonormalBasis(**kw)
    if param in ("cayley_frozen", "qr_frozen"):
        return SpanFrozenCayleyBasis(**kw)
    raise ValueError(
        f"Unknown learnable basis param {param!r}; expected qr, cayley, cayley_frozen/qr_frozen"
    )
