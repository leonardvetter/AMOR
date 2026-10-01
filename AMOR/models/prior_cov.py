"""Low-rank empirical prior covariance for Zahm/Roseberry DIS option 2.

GEVP: H φ = λ C^{-1} φ with ‖φ‖_{C^{-1}} = 1.
Monte-Carlo SVD: ã = C^{1/2} J^* g, SVD → ψ, φ = C^{1/2} ψ.

C ≈ V diag(λ) V^T + ε I  (sample cov of training inputs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass
class LowRankPriorCov:
    """C = V diag(lam) V^T + jitter * I."""

    V: np.ndarray  # (d, r)
    lam: np.ndarray  # (r,)
    jitter: float
    mean: np.ndarray  # (d,)

    @property
    def rank(self) -> int:
        return int(self.lam.size)

    @property
    def dim(self) -> int:
        return int(self.V.shape[0])

    @classmethod
    def from_samples(
        cls,
        X: np.ndarray,
        *,
        rank: int = 256,
        jitter_frac: float = 1e-4,
        random_state: int = 42,
        min_lam_over_jitter: float = 10.0,
    ) -> "LowRankPriorCov":
        """Fit C ≈ V diag(λ) Vᵀ + εI from samples.

        Truncates SVD modes with λ < min_lam_over_jitter * ε so C^{1/2}
        whitening is not polluted by noise-floor directions (critical when
        requested rank ≫ effective prior rank).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be (n,d), got {X.shape}")
        n, d = X.shape
        if n < 2:
            raise ValueError(f"need >=2 samples for cov, got {n}")
        mean = X.mean(axis=0)
        Xc = X - mean
        k = int(min(rank, n - 1, d))
        if k < 1:
            raise ValueError("cov rank < 1")
        # Xc ≈ U S Vt  ⇒ sample cov ≈ Vt.T @ diag(S^2/(n-1)) @ Vt
        _, s, vt = randomized_svd(
            Xc,
            n_components=k,
            n_oversamples=min(20, max(2, n - k)),
            random_state=random_state,
        )
        lam = (s.astype(np.float64) ** 2) / float(max(1, n - 1))
        V = vt.T.astype(np.float64, copy=False)
        # ε relative to leading eigenvalue (stable when spectrum decays).
        jitter = float(jitter_frac) * float(max(lam[0], 1e-30))
        keep = lam >= float(min_lam_over_jitter) * jitter
        if not np.any(keep):
            keep[0] = True
        n_drop = int((~keep).sum())
        if n_drop:
            print(
                f"LowRankPriorCov: dropping {n_drop}/{k} modes with "
                f"λ < {min_lam_over_jitter:g}·ε (ε_pre={jitter:.3e})",
                flush=True,
            )
        V = V[:, keep]
        lam = lam[keep]
        jitter = float(jitter_frac) * float(max(lam[0], 1e-30))
        return cls(
            V=V,
            lam=lam,
            jitter=jitter,
            mean=mean.astype(np.float64, copy=False),
        )

    def apply_sqrt(self, x: np.ndarray) -> np.ndarray:
        """Exact C^{1/2} x for C = V Λ V^T + ε I (V column-orthonormal).

        C^{1/2} = √ε I + V (√(λ+ε) - √ε) V^T
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"dim {x.size} != {self.dim}")
        eps = self.jitter
        sqrt_eps = math.sqrt(eps)
        coef = self.V.T @ x
        scale = np.sqrt(self.lam + eps) - sqrt_eps
        return sqrt_eps * x + self.V @ (scale * coef)

    def save_dict(self) -> dict:
        return {
            "cov_V": self.V.astype(np.float32),
            "cov_lam": self.lam.astype(np.float64),
            "cov_jitter": np.float64(self.jitter),
            "cov_mean": self.mean.astype(np.float32),
            "ortho_metric": np.asarray("C_inv"),
            "input_metric": np.asarray("prior_cov"),
        }

    @classmethod
    def from_save_dict(cls, d) -> "LowRankPriorCov":
        return cls(
            V=np.asarray(d["cov_V"], dtype=np.float64),
            lam=np.asarray(d["cov_lam"], dtype=np.float64),
            jitter=float(np.asarray(d["cov_jitter"]).reshape(-1)[0]),
            mean=np.asarray(d["cov_mean"], dtype=np.float64),
        )
