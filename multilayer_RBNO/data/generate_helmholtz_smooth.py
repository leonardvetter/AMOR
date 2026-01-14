import os
import sys
import numpy as np
import math

# Set environment variables before importing dolfin
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(os.environ.get('SCRATCH', '/tmp'), 'dolfin_cache'))
os.environ.setdefault('DOLFIN_CACHE_DIR', os.path.join(os.environ.get('SCRATCH', '/tmp'), 'dolfin_cache'))
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)

# Set PKG_CONFIG_PATH for DOLFIN
conda_env_path = os.path.expanduser('~/.conda/envs/test_fenics')
if os.path.exists(conda_env_path):
    pkgconfig_path = os.path.join(conda_env_path, 'lib', 'pkgconfig')
    if os.path.exists(pkgconfig_path):
        current_pkg_config = os.environ.get('PKG_CONFIG_PATH', '')
        os.environ['PKG_CONFIG_PATH'] = pkgconfig_path + (':' + current_pkg_config if current_pkg_config else '')
    lib_path = os.path.join(conda_env_path, 'lib')
    if os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = lib_path + (':' + current_ld_path if current_ld_path else '')

# Set PYTHONPATH for hippylib and hippyflow
if os.environ.get('HIPPYLIB_PATH'):
    sys.path.insert(0, os.environ.get('HIPPYLIB_PATH'))
if os.environ.get('HIPPYFLOW_PATH'):
    sys.path.insert(0, os.environ.get('HIPPYFLOW_PATH'))
if os.path.exists(os.path.join(os.environ.get('SCRATCH', '/tmp'), 'python_packages')):
    sys.path.insert(0, os.path.join(os.environ.get('SCRATCH', '/tmp'), 'python_packages'))

# Add sysroot lib64 to LIBRARY_PATH for linker to find libmvec_nonshared.a
sysroot_lib64 = os.path.join(conda_env_path, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64')
if os.path.exists(sysroot_lib64):
    current_lib_path = os.environ.get('LIBRARY_PATH', '')
    os.environ['LIBRARY_PATH'] = sysroot_lib64 + (':' + current_lib_path if current_lib_path else '')

sys.path.insert(0, os.path.dirname(__file__))

# Patch dijitso.build.make_compile_command to add sysroot lib64 to linker command BEFORE importing dolfin
# The linker looks for /usr/lib64/libmvec_nonshared.a which doesn't exist on the system.
# The library exists in the conda sysroot. We add -L flag and also try to ensure
# the linker checks sysroot before system paths by adding it early in the command.
import dijitso.build as dijitso_build
_original_make_compile_command = dijitso_build.make_compile_command
def patched_make_compile_command(src_filename, lib_filename, dependencies, build_params, cache_params):
    cmd = _original_make_compile_command(src_filename, lib_filename, dependencies, build_params, cache_params)
    # Add sysroot lib64 to command if it exists
    if os.path.exists(sysroot_lib64):
        sysroot_lib64_abs = os.path.abspath(sysroot_lib64)
        sysroot_abs = os.path.abspath(os.path.join(conda_env_path, 'x86_64-conda-linux-gnu', 'sysroot'))
        
        # Strategy 1: Add -L flag very early (right after compiler) to prioritize sysroot
        lib_path_flag = f'-L{sysroot_lib64_abs}'
        if lib_path_flag not in cmd:
            # Insert right after compiler name (index 1) to check sysroot first
            cmd.insert(1, lib_path_flag)
        
        # Strategy 2: Use -Wl,--sysroot to make linker use conda sysroot as system root
        # This makes /usr/lib64 resolve to sysroot/usr/lib64
        sysroot_flag = f'-Wl,--sysroot={sysroot_abs}'
        if sysroot_flag not in ' '.join(cmd):
            # Insert after compiler but before other flags
            cmd.insert(1, sysroot_flag)
        
        # Strategy 3: Replace any hardcoded /usr/lib64/libmvec_nonshared.a references
        lib_file = os.path.join(sysroot_lib64_abs, 'libmvec_nonshared.a')
        if os.path.exists(lib_file):
            # Replace any absolute path references to /usr/lib64/libmvec_nonshared.a
            for i, arg in enumerate(cmd):
                if '/usr/lib64/libmvec_nonshared.a' in arg:
                    cmd[i] = arg.replace('/usr/lib64/libmvec_nonshared.a', lib_file)
        
        # Strategy 4: Add -Wl,--library-path as additional search path
        library_path_flag = f'-Wl,--library-path={sysroot_lib64_abs}'
        if library_path_flag not in ' '.join(cmd):
            for i, arg in enumerate(cmd):
                if arg.startswith('-o') or (i > 0 and cmd[i-1] == '-o'):
                    cmd.insert(i, library_path_flag)
                    break
    return cmd
dijitso_build.make_compile_command = patched_make_compile_command

import dolfin as dl
from mpi4py import MPI
import hippylib as hp
from helmholtz_pde import setup_helmholtz_problem, extract_to_grid, extract_vector_to_grid

def helmholtz_settings_smooth():
    settings = {}
    settings["seed"] = 0
    settings["nx"] = 100
    settings["frequency"] = 600
    settings["sigma"] = 0.2
    settings["rho"] = 0.4
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25*math.pi
    settings["comm_mesh"] = MPI.COMM_SELF
    return settings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate smooth Helmholtz dataset")
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    
    scratch = os.environ.get('SCRATCH', '/tmp')
    output_dir = os.path.join(scratch, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'helmholtz_dataset_smooth.npz')
    
    start_idx = 0
    if os.path.exists(output_file):
        print(f"Existing dataset found at {output_file}")
        try:
            existing_data = np.load(output_file)
            existing_samples = len(existing_data['m'])
            print(f"Found {existing_samples} existing samples")
            if existing_samples >= args.num_samples:
                print(f"Dataset already complete ({existing_samples} >= {args.num_samples}). Exiting.")
                import sys
                sys.exit(0)
            start_idx = existing_samples
            print(f"Resuming from sample {start_idx}")
        except Exception as e:
            print(f"Warning: Could not load existing dataset ({e}). Starting from scratch.")
            start_idx = 0
    
    settings = helmholtz_settings_smooth()
    Vh, pde, prior, observable = setup_helmholtz_problem(settings)
    mesh = Vh[hp.PARAMETER].mesh()
    box = [0., 0., 3., 3.]
    nx = settings["nx"]
    
    if start_idx == 0:
        m_dataset = []
        u_real_dataset = []
        u_imag_dataset = []
    else:
        existing_data = np.load(output_file)
        m_dataset = list(existing_data['m'])
        u_real_dataset = list(existing_data['u_real'])
        u_imag_dataset = list(existing_data['u_imag'])
        existing_data.close()
    
    np.random.seed(settings["seed"] + start_idx)
    
    if start_idx == 0:
        print(f"Generating {args.num_samples} Helmholtz samples...")
    else:
        remaining = args.num_samples - start_idx
        print(f"Generating {remaining} additional samples (will have {args.num_samples} total)...")
    
    for i in range(start_idx, args.num_samples):
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
            file_size = os.path.getsize(output_file) / (1024**2)
            print(f"Saved {len(m_dataset)} samples. File size: {file_size:.2f} MB")
    
    print(f"Dataset saved to {output_file}, {len(m_dataset)} samples")


