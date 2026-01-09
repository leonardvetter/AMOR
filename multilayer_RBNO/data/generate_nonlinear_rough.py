from fipy import Grid2D, CellVariable, DiffusionTerm, numerix as np
import numpy as onp
import os
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, os.path.dirname(__file__))
from create_nonlinear_dataset import Gaussian, rbf_kernel, sample_gp_on_grid

NUM_SAMPLES = 10000
N = 100
L = 1.0
KEY = jax.random.PRNGKey(356)
jax.config.update("jax_enable_x64", True)

dx = L / N
mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)

f_dataset = onp.zeros((NUM_SAMPLES, N, N))
m_dataset = onp.zeros((NUM_SAMPLES, N, N))
u_dataset = onp.zeros((NUM_SAMPLES, N, N))

print("Sampling source term f from GP (rough, high variance)...")
key, subkey = jax.random.split(KEY)
f_samples, xs, ys = sample_gp_on_grid(
    subkey, nx=N, ny=N,
    variance=10.0, lengthscales=(0.1, 0.1),
    jitter=1e-5, num_samples=NUM_SAMPLES
)

print("Sampling coefficient field m from GP (rough, high variance)...")
key, subkey = jax.random.split(key)
m_samples, _, _ = sample_gp_on_grid(
    subkey, nx=N, ny=N,
    variance=2.0, lengthscales=(0.15, 0.15),
    jitter=1e-5, num_samples=NUM_SAMPLES
)

print("\nSolving PDEs...")
for i in range(NUM_SAMPLES):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{NUM_SAMPLES}")
    
    f_array = onp.array(f_samples[i])
    m_array = onp.array(m_samples[i])
    
    f_dataset[i] = f_array
    m_dataset[i] = m_array
    
    f_flat = f_array.ravel(order="C")
    m_flat = m_array.ravel(order="C")
    D_flat = onp.exp(m_flat)
    
    f_cv = CellVariable(mesh=mesh, value=f_flat)
    D_cv = CellVariable(mesh=mesh, value=D_flat)
    
    u = CellVariable(name="u", mesh=mesh, value=0.0)
    u.constrain(0.0, where=mesh.exteriorFaces)
    
    eq = DiffusionTerm(coeff=D_cv, var=u) == -f_cv
    eq.solve(var=u)
    
    u_grid = u.value.reshape((N, N), order="C")
    u_dataset[i] = u_grid

scratch = os.environ.get('SCRATCH', '/tmp')
output_dir = os.path.join(scratch, 'data')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'nonlinear_diffusion_dataset_rough.npz')

onp.savez_compressed(
    output_file,
    f=f_dataset,
    m=m_dataset,
    u=u_dataset,
    N=N,
    L=L,
    num_samples=NUM_SAMPLES
)

print(f"Dataset saved to {output_file}, {NUM_SAMPLES} samples")



