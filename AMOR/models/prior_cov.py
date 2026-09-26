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

        Truncates SVD modes with λ < min_lam_over_jitter * ε so whitening /
        C^{-1} are not polluted by noise-floor directions (critical when
        requested rank ≫ effective prior rank, e.g. rib cov_rank=256).
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

    def apply_inv_sqrt(self, x: np.ndarray) -> np.ndarray:
        """Exact C^{-1/2} x for C = V Λ V^T + ε I.

        C^{-1/2} = ε^{-1/2} I + V ((λ+ε)^{-1/2} - ε^{-1/2}) V^T
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"dim {x.size} != {self.dim}")
        eps = self.jitter
        inv_sqrt_eps = 1.0 / math.sqrt(eps)
        coef = self.V.T @ x
        scale = 1.0 / np.sqrt(self.lam + eps) - inv_sqrt_eps
        return inv_sqrt_eps * x + self.V @ (scale * coef)

    def apply_inv(self, x: np.ndarray) -> np.ndarray:
        """Exact (V Λ V^T + ε I)^{-1} via Woodbury."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"dim {x.size} != {self.dim}")
        eps = self.jitter
        coef = self.V.T @ x
        scale = self.lam / (eps * (eps + self.lam))
        return x / eps - self.V @ (scale * coef)

    def rows_to_physical(self, rows_w: np.ndarray) -> np.ndarray:
        """ψ rows (Euclidean from SVD of C^{1/2}-whitened probes) → φ = C^{1/2} ψ."""
        rows_w = np.asarray(rows_w, dtype=np.float64)
        return np.stack([self.apply_sqrt(r) for r in rows_w], axis=0)

    def encode(self, X: np.ndarray, Phi: np.ndarray) -> np.ndarray:
        """z = Φ^T C^{-1} (x - mean); X (S,d), Phi (k,d) → (S,k)."""
        Phi = np.asarray(Phi, dtype=np.float64)
        Xc = np.asarray(X, dtype=np.float64) - self.mean
        # (C^{-1} Xc^T)^T Φ^T ... z_s = Phi @ C^{-1} x_s  if Phi rows are φ_i
        # z = (C^{-1} Xc) @ Phi.T  with C^{-1} applied row-wise
        Y = np.stack([self.apply_inv(x) for x in Xc], axis=0)
        return Y @ Phi.T

    def gram_Cinv(self, Phi: np.ndarray) -> np.ndarray:
        """Φ C^{-1} Φ^T should be ~I for C^{-1}-orthonormal rows."""
        Phi = np.asarray(Phi, dtype=np.float64)
        Y = np.stack([self.apply_inv(r) for r in Phi], axis=0)
        return Phi @ Y.T

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
