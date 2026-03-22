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
    import fcntl
    import time
    parser = argparse.ArgumentParser(description="Generate rough Helmholtz dataset")
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    
    output_dir = '/users/PAS3244/dvoderhol/data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'helmholtz_dataset_rough.npz')
    lock_file = os.path.join(output_dir, 'helmholtz_dataset_rough.lock')
    
    # Process locking to ensure only one process per dataset
    lock_fd = None
    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except IOError:
        print(f"Another process is already running for {output_file}. Exiting.", flush=True)
        if lock_fd:
            lock_fd.close()
        import sys
        sys.exit(1)
    
    # Check for recovery file (.tmp.npz) first - np.savez_compressed adds .npz extension
    temp_file = output_file + '.tmp.npz'
    if os.path.exists(temp_file):
        print(f"Found temporary file at {temp_file}, attempting recovery...", flush=True)
        try:
            test_data = np.load(temp_file, allow_pickle=False)
            test_samples = len(test_data['m'])
            test_data.close()
            print(f"Temporary file is valid with {test_samples} samples. Recovering...", flush=True)
            if os.path.exists(output_file):
                backup_file = output_file + '.corrupted_backup'
                print(f"Backing up corrupted main file to {backup_file}", flush=True)
                try:
                    os.rename(output_file, backup_file)
                except:
                    pass
            os.rename(temp_file, output_file)
            print(f"Recovery successful. Using recovered file with {test_samples} samples.", flush=True)
        except Exception as e:
            print(f"Temporary file is also corrupted ({e}). Will try main file.", flush=True)
    
    start_idx = 0
    if os.path.exists(output_file):
        print(f"Existing dataset found at {output_file}", flush=True)
        existing_samples = None
        max_retries = 50
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                existing_data = np.load(output_file, allow_pickle=False)
                existing_samples = len(existing_data['m'])
                existing_data.close()
                print(f"Found {existing_samples} existing samples", flush=True)
                if existing_samples >= args.num_samples:
                    print(f"Dataset already complete ({existing_samples} >= {args.num_samples}). Exiting.", flush=True)
                    import sys
                    sys.exit(0)
                start_idx = existing_samples
                print(f"Resuming from sample {start_idx}", flush=True)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries}: Could not read file ({e}). Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: Could not load existing dataset after {max_retries} attempts ({e}).", flush=True)
                    print(f"File exists but cannot be read - this may indicate corruption.", flush=True)
                    print(f"EXITING to prevent data loss. Please manually recover the file or remove it if you want to start from scratch.", flush=True)
                    import sys
                    sys.exit(1)
    
    settings = helmholtz_settings_rough()
    Vh, pde, prior, observable = setup_helmholtz_problem(settings)
    mesh = Vh[hp.PARAMETER].mesh()
    box = [0., 0., 3., 3.]
    nx = settings["nx"]
    
    if start_idx == 0:
        m_dataset = []
        u_real_dataset = []
        u_imag_dataset = []
    else:
        max_retries = 50
        retry_delay = 5
        loaded = False
        for attempt in range(max_retries):
            try:
                existing_data = np.load(output_file, allow_pickle=False)
                m_dataset = list(existing_data['m'])
                u_real_dataset = list(existing_data['u_real'])
                u_imag_dataset = list(existing_data['u_imag'])
                existing_data.close()
                loaded = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries}: Could not load data ({e}). Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: Could not load existing data after {max_retries} attempts ({e}).", flush=True)
                    print(f"EXITING to prevent data loss. Please manually recover the file or remove it if you want to start from scratch.", flush=True)
                    import sys
                    sys.exit(1)
    
    np.random.seed(settings["seed"] + start_idx)
    
    if start_idx == 0:
        print(f"Generating {args.num_samples} Helmholtz samples...", flush=True)
    else:
        remaining = args.num_samples - start_idx
        print(f"Generating {remaining} additional samples (will have {args.num_samples} total)...", flush=True)
    
    for i in range(start_idx, args.num_samples):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{args.num_samples}", flush=True)
        
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
        
        if (i + 1) % 25 == 0 or (i + 1) == args.num_samples:
            # Atomic save: write to temp file then rename
            # np.savez_compressed adds .npz extension, so we need to account for that
            temp_file_base = output_file + '.tmp'
            temp_file = temp_file_base + '.npz'
            try:
                np.savez_compressed(
                    temp_file_base,
                    m=np.array(m_dataset),
                    u_real=np.array(u_real_dataset),
                    u_imag=np.array(u_imag_dataset),
                    nx=nx,
                    box=box,
                    num_samples=len(m_dataset),
                    settings=settings
                )
                # Verify temp file exists (np.savez_compressed adds .npz extension)
                if not os.path.exists(temp_file):
                    raise IOError(f"Temp file {temp_file} was not created")
                # Verify temp file can be read
                test_load = np.load(temp_file, allow_pickle=False)
                test_load.close()
                # Atomic rename: remove old file if it exists, then rename temp
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_file, output_file)
                if not os.path.exists(output_file):
                    raise IOError(f"Rename failed: {output_file} does not exist after rename")
                file_size = os.path.getsize(output_file) / (1024**2)
                print(f"Saved {len(m_dataset)} samples. File size: {file_size:.2f} MB", flush=True)
            except Exception as e:
                print(f"ERROR saving at sample {i+1}: {e}", flush=True)
                # Clean up temp files if they exist
                for cleanup_file in [temp_file, temp_file_base]:
                    if os.path.exists(cleanup_file):
                        try:
                            os.remove(cleanup_file)
                        except:
                            pass
                raise
    
    print(f"Dataset saved to {output_file}, {len(m_dataset)} samples", flush=True)
    if lock_fd:
        lock_fd.close()
        try:
            os.remove(lock_file)
        except:
            pass


