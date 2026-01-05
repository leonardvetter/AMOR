import os
import sys
import numpy as np
import math
sys.path.insert(0, os.path.dirname(__file__))
from helmholtz_pde import setup_helmholtz_problem, extract_to_grid, extract_vector_to_grid
import dolfin as dl
from mpi4py import MPI
import hippylib as hp

def helmholtz_settings_rough():
    settings = {}
    settings["seed"] = 0
    settings["nx"] = 100
    settings["frequency"] = 600
    settings["sigma"] = 0.8
    settings["rho"] = 0.1
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25*math.pi
    settings["comm_mesh"] = MPI.COMM_SELF
    return settings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate rough Helmholtz dataset")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    
    scratch = os.environ.get('SCRATCH', '/tmp')
    output_dir = os.path.join(scratch, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'helmholtz_dataset_rough.npz')
    
    settings = helmholtz_settings_rough()
    Vh, pde, prior, observable = setup_helmholtz_problem(settings)
    mesh = Vh[hp.PARAMETER].mesh()
    box = [0., 0., 3., 3.]
    nx = settings["nx"]
    
    m_dataset = []
    u_real_dataset = []
    u_imag_dataset = []
    
    np.random.seed(settings["seed"])
    
    for i in range(args.num_samples):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{args.num_samples}")
        
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
        noise.set_local(np.random.randn(noise.local_size()))
        noise.apply("")
        m = pde.generate_parameter()
        prior.init_vector(m, 0)
        prior.sample(noise, m)
        x = [None, m, None]
        state = pde.generate_state()
        pde.solveFwd(state, x)
        
        m_func = hp.vector2Function(m, Vh[hp.PARAMETER])
        u_func = hp.vector2Function(state, Vh[hp.STATE])
        
        m_grid = extract_to_grid(m_func, box, nx, nx)
        u_real_grid, u_imag_grid = extract_vector_to_grid(u_func, box, nx, nx)
        
        m_dataset.append(m_grid)
        u_real_dataset.append(u_real_grid)
        u_imag_dataset.append(u_imag_grid)
        
        if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
            np.savez_compressed(
                output_file,
                m=np.array(m_dataset),
                u_real=np.array(u_real_dataset),
                u_imag=np.array(u_imag_dataset),
                nx=nx,
                box=box,
                num_samples=len(m_dataset),
                settings=settings
            )
    
    print(f"Dataset saved to {output_file}, {len(m_dataset)} samples")


