# -*- coding: utf-8 -*-
"""Shared GP + FiPy helpers for nonlinear diffusion dataset generation.

PDE: -∇·(e^m ∇u) = f on [0,L]^2 with homogeneous Dirichlet BCs.
Inputs (f, m) from independent RBF GPs; output u from FiPy DiffusionTerm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import os
import sys

from fipy import Grid2D, CellVariable, DiffusionTerm
import numpy as onp

# FiPy/JAX + NumPy 1.26: asarray(copy=...) unsupported — strip unknown kwargs.
_original_asarray = onp.asarray


def _patched_asarray(a, dtype=None, copy=None, order=None, subok=False, ndmin=0):
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if order is not None:
        kwargs["order"] = order
    if subok:
        kwargs["subok"] = subok
    if ndmin > 0:
        kwargs["ndmin"] = ndmin
    return _original_asarray(a, **kwargs)


onp.asarray = _patched_asarray
sys.modules["numpy"].asarray = _patched_asarray

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")


class Gaussian:
    """Gaussian with mean mu and covariance Sigma."""

    def __init__(self, mu, Sigma, jitter=1e-6):
        self.mu = mu
        self.Sigma = Sigma
        self.jitter = jitter
        self._L = None

    @property
    def L(self):
        if self._L is None:
            n = self.Sigma.shape[0]
            S = 0.5 * (self.Sigma + self.Sigma.T)
            self._L = jnp.linalg.cholesky(S + self.jitter * jnp.eye(n, dtype=S.dtype))
        return self._L

    def sample(self, key, num_samples=1):
        n = self.mu.shape[0]
        z = jax.random.normal(key, shape=(num_samples, n), dtype=self.mu.dtype)
        return self.mu + z @ self.L.T


def rbf_kernel(X, Y, variance=1.0, lengthscales=(0.2, 0.2), dtype=jnp.float64):
    lx, ly = lengthscales
    Xs = X / jnp.array([lx, ly], dtype=dtype)
    Ys = Y / jnp.array([lx, ly], dtype=dtype)
    diff = Xs[:, None, :] - Ys[None, :, :]
    sqdist = jnp.sum(diff * diff, axis=-1)
    return jnp.asarray(variance, dtype=dtype) * jnp.exp(-0.5 * sqdist)


def sample_gp_on_grid(
    key,
    nx=80,
    ny=80,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    variance=1.0,
    lengthscales=(0.15, 0.25),
    jitter=1e-5,
    num_samples=1,
    dtype=jnp.float64,
):
    """Sample a 2D RBF GP on a regular grid. Returns (S, ny, nx), xs, ys."""
    xs = jnp.linspace(xlim[0], xlim[1], nx, dtype=dtype)
    ys = jnp.linspace(ylim[0], ylim[1], ny, dtype=dtype)
    Xg, Yg = jnp.meshgrid(xs, ys, indexing="xy")
    XY = jnp.stack([Xg.ravel(), Yg.ravel()], axis=-1)
    K = rbf_kernel(XY, XY, variance=variance, lengthscales=lengthscales, dtype=dtype)
    mu = jnp.zeros(K.shape[0], dtype=dtype)
    gp = Gaussian(mu=mu, Sigma=K, jitter=jitter)
    key, sub = jax.random.split(key)
    samples_flat = gp.sample(sub, num_samples=num_samples)
    samples_img = samples_flat.reshape(num_samples, ny, nx)
    return samples_img, xs, ys


def dataset_basename(fvar: float, flen: float, mvar: float, mlen: float, suffix: str = "") -> str:
    """Filename stem: nonlinear_fvar_*_flen_*_mvar_*_mlen_*[+_suffix]."""
    base = f"nonlinear_fvar_{fvar}_flen_{flen}_mvar_{mvar}_mlen_{mlen}"
    if suffix:
        base = f"{base}_{suffix}"
    return base


def generate_nonlinear_diffusion(
    *,
    fvar: float,
    flen: float,
    mvar: float,
    mlen: float,
    N: int = 100,
    L: float = 1.0,
    num_samples: int = 10000,
    suffix: str = "",
    seed: int = 356,
    output_dir: str = expand("${AMOR_SCRATCH_DATA}"),  # rename path on a different cluster
    save_path: str | None = None,
):
    """Generate and save (f, m, u) dataset. Resume-safe if partial file exists."""
    os.makedirs(output_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(output_dir, dataset_basename(fvar, flen, mvar, mlen, suffix) + ".npz")

    dx = L / N
    mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)
    key0 = jax.random.PRNGKey(seed)

    print(f"Config: N={N}, L={L}, num_samples={num_samples}")
    print(f"GP f: var={fvar}, len={flen}; m: var={mvar}, len={mlen}")
    print(f"Output: {save_path}")

    start_idx = 0
    if os.path.exists(save_path):
        try:
            existing = onp.load(save_path)
            existing_samples = len(existing["f"])
            existing.close()
            print(f"Found {existing_samples} existing samples at {save_path}")
            if existing_samples >= num_samples:
                print("Dataset already complete. Exiting.")
                return save_path
            start_idx = existing_samples
            print(f"Resuming from sample {start_idx}")
        except Exception as e:
            print(f"Could not load existing file ({e}); starting from scratch.")
            start_idx = 0

    if start_idx == 0:
        f_dataset = onp.zeros((num_samples, N, N))
        m_dataset = onp.zeros((num_samples, N, N))
        u_dataset = onp.zeros((num_samples, N, N))
        key, subkey = jax.random.split(key0)
        f_samples, _, _ = sample_gp_on_grid(
            subkey, nx=N, ny=N, variance=fvar, lengthscales=(flen, flen),
            jitter=1e-5, num_samples=num_samples,
        )
        key, subkey = jax.random.split(key)
        m_samples, _, _ = sample_gp_on_grid(
            subkey, nx=N, ny=N, variance=mvar, lengthscales=(mlen, mlen),
            jitter=1e-5, num_samples=num_samples,
        )
    else:
        existing = onp.load(save_path)
        f_dataset = onp.zeros((num_samples, N, N))
        m_dataset = onp.zeros((num_samples, N, N))
        u_dataset = onp.zeros((num_samples, N, N))
        f_dataset[:start_idx] = existing["f"][:start_idx]
        m_dataset[:start_idx] = existing["m"][:start_idx]
        u_dataset[:start_idx] = existing["u"][:start_idx]
        existing.close()
        key = key0
        for _ in range(start_idx):
            key, _ = jax.random.split(key)
        key, subkey = jax.random.split(key)
        f_samples, _, _ = sample_gp_on_grid(
            subkey, nx=N, ny=N, variance=fvar, lengthscales=(flen, flen),
            jitter=1e-5, num_samples=num_samples - start_idx,
        )
        key, subkey = jax.random.split(key)
        m_samples, _, _ = sample_gp_on_grid(
            subkey, nx=N, ny=N, variance=mvar, lengthscales=(mlen, mlen),
            jitter=1e-5, num_samples=num_samples - start_idx,
        )

    print("Solving PDEs...")
    for i in range(start_idx, num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_samples}", flush=True)

        f_array = onp.array(f_samples[i - start_idx])
        m_array = onp.array(m_samples[i - start_idx])
        f_dataset[i] = f_array
        m_dataset[i] = m_array

        f_cv = CellVariable(mesh=mesh, value=f_array.ravel(order="C"))
        D_cv = CellVariable(mesh=mesh, value=onp.exp(m_array.ravel(order="C")))
        u = CellVariable(name="u", mesh=mesh, value=0.0)
        u.constrain(0.0, where=mesh.exteriorFaces)
        eq = DiffusionTerm(coeff=D_cv, var=u) == -f_cv
        eq.solve(var=u)
        u_dataset[i] = u.value.reshape((N, N), order="C")

        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            onp.savez_compressed(
                save_path,
                f=f_dataset[: i + 1],
                m=m_dataset[: i + 1],
                u=u_dataset[: i + 1],
                N=N,
                L=L,
                num_samples=i + 1,
                fvar=fvar,
                flen=flen,
                mvar=mvar,
                mlen=mlen,
            )
            mb = os.path.getsize(save_path) / (1024 ** 2)
            print(f"  Saved {i + 1} samples ({mb:.2f} MB)", flush=True)

    onp.savez_compressed(
        save_path,
        f=f_dataset,
        m=m_dataset,
        u=u_dataset,
        N=N,
        L=L,
        num_samples=num_samples,
        fvar=fvar,
        flen=flen,
        mvar=mvar,
        mlen=mlen,
    )
    print(f"Done: {save_path} ({os.path.getsize(save_path) / (1024 ** 2):.2f} MB)")
    return save_path
