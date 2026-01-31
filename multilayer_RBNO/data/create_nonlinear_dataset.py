# -*- coding: utf-8 -*-
"""
Dataset generation for the nonlinear diffusion equation:
    -∇·(e^m(x) ∇u) = f

where:
    - f is the source term (sampled from GP)
    - m(x) is a spatially varying coefficient field (sampled from GP)
    - u is the solution
    
The dataset consists of tuples (f, m, u) where f and m are inputs and u is the output.
"""

from fipy import Grid2D, CellVariable, DiffusionTerm, numerix as np
import numpy as onp
import matplotlib.pyplot as plt
from typing import Tuple
import os
import sys

# Patch numpy.asarray to handle JAX compatibility issue with NumPy 1.26.4
_original_asarray = onp.asarray
def _patched_asarray(a, dtype=None, copy=None, order=None, subok=False, ndmin=0):
    kwargs = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if order is not None:
        kwargs['order'] = order
    if subok:
        kwargs['subok'] = subok
    if ndmin > 0:
        kwargs['ndmin'] = ndmin
    return _original_asarray(a, **kwargs)
onp.asarray = _patched_asarray
sys.modules['numpy'].asarray = _patched_asarray

import jax
import jax.numpy as jnp

# ============================================================================
# Configuration
# ============================================================================

NUM_SAMPLES = 10000      # Number of dataset samples
N = 100                 # Grid resolution (NxN)
L = 1.0                 # Domain size [0,L]^2
scratch = os.environ.get('SCRATCH', '/tmp')
output_dir = os.path.join(scratch, 'data')
os.makedirs(output_dir, exist_ok=True)
SAVE_PATH = os.path.join(output_dir, "nonlinear_diffusion_dataset.npz")  # Output file

# Random seed
os.environ.setdefault('JAX_TRACEBACK_FILTERING', 'off')
KEY = jax.random.PRNGKey(356)

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Gaussian Process Sampling
# ============================================================================

class Gaussian:
    """Gaussian distribution with mean mu and covariance Sigma."""
    
    def __init__(self, mu, Sigma, jitter=1e-6):
        self.mu = mu
        self.Sigma = Sigma
        self.jitter = jitter
        self._L = None
    
    @property
    def L(self):
        """Cholesky decomposition of covariance matrix."""
        if self._L is None:
            n = self.Sigma.shape[0]
            # Enforce symmetry (tiny fp errors can hurt Cholesky)
            S = 0.5 * (self.Sigma + self.Sigma.T)
            self._L = jnp.linalg.cholesky(S + self.jitter * jnp.eye(n, dtype=S.dtype))
        return self._L

    def sample(self, key, num_samples=1):
        """Sample from the Gaussian distribution."""
        n = self.mu.shape[0]
        z = jax.random.normal(key, shape=(num_samples, n), dtype=self.mu.dtype)
        return self.mu + z @ self.L.T


def rbf_kernel(X, Y, variance=1.0, lengthscales=(0.2, 0.2), dtype=jnp.float64):
    """RBF (squared exponential) kernel."""
    ℓx, ℓy = lengthscales
    Xs = X / jnp.array([ℓx, ℓy], dtype=dtype)
    Ys = Y / jnp.array([ℓx, ℓy], dtype=dtype)
    diff = Xs[:, None, :] - Ys[None, :, :]
    sqdist = jnp.sum(diff * diff, axis=-1)
    return jnp.asarray(variance, dtype=dtype) * jnp.exp(-0.5 * sqdist)


def sample_gp_on_grid(key, nx=80, ny=80, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                       variance=1.0, lengthscales=(0.15, 0.25),
                       jitter=1e-5, num_samples=1, dtype=jnp.float64):
    """Sample from a 2D Gaussian Process on a regular grid."""
    xs = jnp.linspace(xlim[0], xlim[1], nx, dtype=dtype)
    ys = jnp.linspace(ylim[0], ylim[1], ny, dtype=dtype)
    Xg, Yg = jnp.meshgrid(xs, ys, indexing="xy")       # (ny, nx)
    XY = jnp.stack([Xg.ravel(), Yg.ravel()], axis=-1)  # (N,2), N=nx*ny

    K = rbf_kernel(XY, XY, variance=variance, lengthscales=lengthscales, dtype=dtype)
    mu = jnp.zeros(K.shape[0], dtype=dtype)
    gp = Gaussian(mu=mu, Sigma=K, jitter=jitter)

    key, sub = jax.random.split(key)
    samples_flat = gp.sample(sub, num_samples=num_samples)         # (S, N)
    samples_img = samples_flat.reshape(num_samples, ny, nx)        # (S, ny, nx)
    return samples_img, xs, ys


# ============================================================================
# FiPy Mesh Setup
# ============================================================================

dx = L / N
mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)
xc, yc = mesh.cellCenters

print(f"Mesh setup: {N}x{N} grid, dx={dx:.4f}")
print(f"Number of cells: {mesh.numberOfCells}")

# Check for existing dataset
start_idx = 0
if os.path.exists(SAVE_PATH):
    print(f"Existing dataset found at {SAVE_PATH}")
    try:
        existing_data = onp.load(SAVE_PATH)
        existing_samples = len(existing_data['f'])
        print(f"Found {existing_samples} existing samples")
        if existing_samples >= NUM_SAMPLES:
            print(f"Dataset already complete ({existing_samples} >= {NUM_SAMPLES}). Exiting.")
            import sys
            sys.exit(0)
        start_idx = existing_samples
        print(f"Resuming from sample {start_idx}")
    except Exception as e:
        print(f"Warning: Could not load existing dataset ({e}). Starting from scratch.")
        start_idx = 0

if start_idx == 0:
    print(f"Generating {NUM_SAMPLES} samples...")
else:
    remaining = NUM_SAMPLES - start_idx
    print(f"Generating {remaining} additional samples (will have {NUM_SAMPLES} total)...")

# ============================================================================
# Generate Dataset
# ============================================================================

# Storage arrays
if start_idx == 0:
    f_dataset = onp.zeros((NUM_SAMPLES, N, N))  # Source term
    m_dataset = onp.zeros((NUM_SAMPLES, N, N))  # Coefficient field
    u_dataset = onp.zeros((NUM_SAMPLES, N, N))  # Solution
else:
    existing_data = onp.load(SAVE_PATH)
    existing_f = existing_data['f']
    existing_m = existing_data['m']
    existing_u = existing_data['u']
    existing_data.close()
    # Create new arrays with space for all samples
    f_dataset = onp.zeros((NUM_SAMPLES, N, N))
    m_dataset = onp.zeros((NUM_SAMPLES, N, N))
    u_dataset = onp.zeros((NUM_SAMPLES, N, N))
    # Copy existing data
    f_dataset[:start_idx] = existing_f[:start_idx]
    m_dataset[:start_idx] = existing_m[:start_idx]
    u_dataset[:start_idx] = existing_u[:start_idx]

# Sample f (source term) from GP
if start_idx == 0:
    print("\nSampling source term f from GP...")
    key, subkey = jax.random.split(KEY)
    f_samples, xs, ys = sample_gp_on_grid(
        subkey, nx=N, ny=N,
        variance=5.0, lengthscales=(0.2, 0.2),
        jitter=1e-5, num_samples=NUM_SAMPLES
    )
    
    # Sample m (coefficient field) from GP
    print("Sampling coefficient field m from GP...")
    key, subkey = jax.random.split(key)
    m_samples, _, _ = sample_gp_on_grid(
        subkey, nx=N, ny=N,
        variance=1.0, lengthscales=(0.3, 0.3),  # Smoother than f
        jitter=1e-5, num_samples=NUM_SAMPLES
    )
else:
    # For resume, we need to regenerate samples starting from the correct key
    print("\nSampling source term f from GP (resuming)...")
    key = jax.random.PRNGKey(356)
    # Advance key to start_idx
    for _ in range(start_idx):
        key, _ = jax.random.split(key)
    key, subkey = jax.random.split(key)
    f_samples, xs, ys = sample_gp_on_grid(
        subkey, nx=N, ny=N,
        variance=5.0, lengthscales=(0.2, 0.2),
        jitter=1e-5, num_samples=NUM_SAMPLES - start_idx
    )
    
    print("Sampling coefficient field m from GP (resuming)...")
    key, subkey = jax.random.split(key)
    m_samples, _, _ = sample_gp_on_grid(
        subkey, nx=N, ny=N,
        variance=1.0, lengthscales=(0.3, 0.3),
        jitter=1e-5, num_samples=NUM_SAMPLES - start_idx
    )

# Solve for each sample
print("\nSolving PDEs...")
for i in range(start_idx, NUM_SAMPLES):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{NUM_SAMPLES}")
    
    # Get f and m as numpy arrays
    f_array = onp.array(f_samples[i - start_idx])
    m_array = onp.array(m_samples[i - start_idx])
    
    # Store the inputs
    f_dataset[i] = f_array
    m_dataset[i] = m_array
    
    # Flatten for FiPy (cell order)
    f_flat = f_array.ravel(order="C")
    m_flat = m_array.ravel(order="C")
    
    # Compute diffusion coefficient D = exp(m)
    D_flat = onp.exp(m_flat)
    
    # Create FiPy cell variables
    f_cv = CellVariable(mesh=mesh, value=f_flat)
    D_cv = CellVariable(mesh=mesh, value=D_flat)
    
    # Setup solution variable with Dirichlet BCs
    u = CellVariable(name="u", mesh=mesh, value=0.0)
    u.constrain(0.0, where=mesh.exteriorFaces)
    
    # Solve: -∇·(D∇u) = f  =>  DiffusionTerm(coeff=D, var=u) == -f
    eq = DiffusionTerm(coeff=D_cv, var=u) == -f_cv
    eq.solve(var=u)
    
    # Store solution
    u_grid = u.value.reshape((N, N), order="C")
    u_dataset[i] = u_grid
    
    # Save periodically
    if (i + 1) % 100 == 0 or (i + 1) == NUM_SAMPLES:
        onp.savez_compressed(
            SAVE_PATH,
            f=f_dataset[:i+1],
            m=m_dataset[:i+1],
            u=u_dataset[:i+1],
            N=N,
            L=L,
            num_samples=i+1
        )
        file_size = os.path.getsize(SAVE_PATH) / (1024**2)
        print(f"  Saved {i+1} samples. File size: {file_size:.2f} MB")

print(f"\nDataset generation complete!")
print(f"  f shape: {f_dataset.shape}")
print(f"  m shape: {m_dataset.shape}")
print(f"  u shape: {u_dataset.shape}")

# ============================================================================
# Dataset Statistics
# ============================================================================

print("\n" + "="*70)
print("Dataset Statistics:")
print("="*70)
print(f"\nSource term f:")
print(f"  min:  {onp.min(f_dataset):.4f}")
print(f"  max:  {onp.max(f_dataset):.4f}")
print(f"  mean: {onp.mean(f_dataset):.4f}")
print(f"  std:  {onp.std(f_dataset):.4f}")

print(f"\nCoefficient field m:")
print(f"  min:  {onp.min(m_dataset):.4f}")
print(f"  max:  {onp.max(m_dataset):.4f}")
print(f"  mean: {onp.mean(m_dataset):.4f}")
print(f"  std:  {onp.std(m_dataset):.4f}")

print(f"\nDiffusion coefficient D = exp(m):")
D_dataset = onp.exp(m_dataset)
print(f"  min:  {onp.min(D_dataset):.4f}")
print(f"  max:  {onp.max(D_dataset):.4f}")
print(f"  mean: {onp.mean(D_dataset):.4f}")
print(f"  std:  {onp.std(D_dataset):.4f}")

print(f"\nSolution u:")
print(f"  min:  {onp.min(u_dataset):.4f}")
print(f"  max:  {onp.max(u_dataset):.4f}")
print(f"  mean: {onp.mean(u_dataset):.4f}")
print(f"  std:  {onp.std(u_dataset):.4f}")

# ============================================================================
# Save Dataset
# ============================================================================

print("\n" + "="*70)
print(f"Saving dataset to: {SAVE_PATH}")
print("="*70)

onp.savez_compressed(
    SAVE_PATH,
    f=f_dataset,
    m=m_dataset,
    u=u_dataset,
    N=N,
    L=L,
    num_samples=NUM_SAMPLES
)

print(f"\nFinal dataset saved successfully!")
print(f"File size: {os.path.getsize(SAVE_PATH) / (1024**2):.2f} MB")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("Generating visualization...")
print("="*70)

# Visualize a random sample
idx = onp.random.randint(0, NUM_SAMPLES)
print(f"\nVisualizing sample #{idx}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot f
im0 = axes[0, 0].imshow(f_dataset[idx], origin="lower", 
                        extent=[0, L, 0, L], aspect="equal", cmap="RdBu_r")
axes[0, 0].set_title(f"Source term f(x,y)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Plot m
im1 = axes[0, 1].imshow(m_dataset[idx], origin="lower", 
                        extent=[0, L, 0, L], aspect="equal", cmap="viridis")
axes[0, 1].set_title(f"Coefficient field m(x,y)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Plot D = exp(m)
D_sample = onp.exp(m_dataset[idx])
im2 = axes[1, 0].imshow(D_sample, origin="lower", 
                        extent=[0, L, 0, L], aspect="equal", cmap="plasma")
axes[1, 0].set_title(f"Diffusion coefficient D(x,y) = exp(m)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Plot u
im3 = axes[1, 1].imshow(u_dataset[idx], origin="lower", 
                        extent=[0, L, 0, L], aspect="equal", cmap="coolwarm")
axes[1, 1].set_title(f"Solution u(x,y)")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.suptitle(f"Nonlinear Diffusion Equation: -∇·(exp(m)∇u) = f\nSample #{idx}", 
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig("nonlinear_diffusion_sample.png", dpi=150, bbox_inches="tight")
print(f"\nVisualization saved to: nonlinear_diffusion_sample.png")
plt.show()

print("\n" + "="*70)
print("Done!")
print("="*70)

