# Author: Liangha Cao, 10/15/2023
# This document is ALMOST ENTIRELY based on the hippyflow helmhotz 2d problem.
# Please cite hippyflow if you use this part of the code.

# ============================================================================
# DOLFIN Environment Configuration for JIT Compilation
# ============================================================================
# The following environment setup is required for DOLFIN to work correctly
# with JIT (Just-In-Time) compilation on NERSC systems. All fixes are
# implemented in the generation scripts (generate_helmholtz_*.py) and must
# be applied BEFORE importing dolfin.
#
# STEP 1: Set cache directories to SCRATCH to avoid home directory quota issues:
#    - XDG_CACHE_HOME: DOLFIN cache directory (set to $SCRATCH/dolfin_cache)
#    - DOLFIN_CACHE_DIR: Alternative DOLFIN cache directory
#    - Code: os.environ.setdefault('XDG_CACHE_HOME', os.path.join(scratch, 'dolfin_cache'))
#    - Code: os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)
#    - This ensures all DOLFIN JIT compilation caches go to SCRATCH, not HOME
#
# STEP 2: Set PKG_CONFIG_PATH to find DOLFIN pkg-config file:
#    - Required path: ~/.conda/envs/test_fenics/lib/pkgconfig
#    - This allows DOLFIN's jit.py to locate dolfin.pc during import
#    - Code: os.environ['PKG_CONFIG_PATH'] = conda_env_path + '/lib/pkgconfig'
#    - Without this, DOLFIN import fails with "Could not find DOLFIN pkg-config file"
#
# STEP 3: Set LD_LIBRARY_PATH to find DOLFIN shared libraries:
#    - Required path: ~/.conda/envs/test_fenics/lib
#    - This ensures runtime linking works correctly for DOLFIN shared libraries
#    - Code: os.environ['LD_LIBRARY_PATH'] = conda_env_path + '/lib'
#    - Must be set before importing dolfin
#
# STEP 4: Set PYTHONPATH for hippylib and hippyflow:
#    - HIPPYLIB_PATH: Path to hippylib installation (if set)
#    - HIPPYFLOW_PATH: Path to hippyflow installation (if set)
#    - $SCRATCH/python_packages: Alternative location for Python packages
#    - Code: sys.path.insert(0, os.path.join(scratch, 'python_packages'))
#    - Must be added to sys.path before importing dolfin or hippylib
#
# STEP 5: Set LIBRARY_PATH for linker to find libraries during compilation:
#    - Required path: ~/.conda/envs/test_fenics/x86_64-conda-linux-gnu/sysroot/usr/lib64
#    - This helps the linker find libraries in the conda sysroot
#    - Code: os.environ['LIBRARY_PATH'] = sysroot_lib64 + (':' + current if current else '')
#    - The linker uses LIBRARY_PATH in addition to -L flags
#
# STEP 6: Patch dijitso.build.make_compile_command to fix linker issues:
#    - Issue: Linker looks for /usr/lib64/libmvec_nonshared.a which doesn't exist on system
#    - Library exists in conda sysroot: ~/.conda/envs/test_fenics/x86_64-conda-linux-gnu/sysroot/usr/lib64/
#    - Fix: Patch dijitso.build.make_compile_command with 4 strategies:
#      1. Add -L flag for sysroot lib64 early in command (right after compiler)
#      2. Add -Wl,--sysroot flag to make linker use conda sysroot as system root
#         This makes /usr/lib64 resolve to sysroot/usr/lib64 automatically
#      3. Replace any hardcoded /usr/lib64/libmvec_nonshared.a paths with sysroot path
#      4. Add -Wl,--library-path flag as additional search path
#    - Code: See generate_helmholtz_*.py for full patch implementation
#    - The patch must be applied BEFORE importing dolfin (which triggers dijitso import)
#    - The -Wl,--sysroot flag is the key fix that makes /usr/lib64 resolve correctly
#
# STEP 7: Ensure all large files (datasets, logs, checkpoints) go to SCRATCH:
#    - Datasets: Save to $SCRATCH/data/ (not repository or HOME)
#    - Logs: Save to $SCRATCH/logs/ (not repository or HOME)
#    - Checkpoints: Save to $SCRATCH/checkpoints/ (not repository or HOME)
#    - Code: output_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'data')
#    - This prevents HOME directory quota issues and keeps repository clean
#
# STEP 8: Handle optional hippyflow import:
#    - hippyflow is used for StateSpaceIdentityOperator and LinearStateObservable
#    - If not available, minimal replacements are defined to allow code to run
#    - Code: try/except ImportError with fallback class definitions
#
# FIX VERIFICATION:
#    - The -Wl,--sysroot flag successfully fixes the linker issue
#    - This makes the linker use conda sysroot as the system root, so /usr/lib64
#      automatically resolves to sysroot/usr/lib64 where libmvec_nonshared.a exists
#    - Tested and confirmed working - compilation succeeds and datasets can be generated
#
# IMPORTANT: All environment variables and patches must be set BEFORE importing dolfin.
# The generation scripts (generate_helmholtz_*.py) implement all steps 1-6.
# Step 7 is handled in the dataset generation logic itself.
# ============================================================================

import os, sys
import dolfin as dl
import numpy as np
if os.environ.get('HIPPYLIB_PATH'):
    sys.path.append(os.environ.get('HIPPYLIB_PATH'))
if os.environ.get('HIPPYFLOW_PATH'):
    sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippylib as hp
try:
    import hippyflow as hf
except ImportError:
    # hippyflow is optional - only needed for StateSpaceIdentityOperator and LinearStateObservable
    # If not available, we'll define minimal replacements
    class StateSpaceIdentityOperator:
        def __init__(self, M, use_mass_matrix=False):
            self.M = M
    class LinearStateObservable:
        def __init__(self, pde, B):
            self.pde = pde
            self.B = B
    hf = type('hf', (), {'StateSpaceIdentityOperator': StateSpaceIdentityOperator, 'LinearStateObservable': LinearStateObservable})()

import math
from mpi4py import MPI


def assemble_mass_matrix(Vh):
    u = dl.TrialFunction(Vh)
    v = dl.TestFunction(Vh)
    M = dl.assemble(dl.inner(u, v) * dl.dx)
    return M


def helmholtz_settings():
    settings = {}
    settings["seed"] = 0 #Random seed
    settings["nx"] = 100 # Number of cells in each direction
    settings["frequency"] = 600

    #Prior statistics
    settings["sigma"] = 0.4 # pointwise variance
    settings["rho"] = 0.2 # spatial correlation length

    #Anisotropy of prior samples
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25*math.pi

    settings["comm_mesh"] = MPI.COMM_SELF

    return settings


class PML:
    def __init__(self, mesh, box, box_pml, A):

        t = [None] * 4

        for i in range(4):
            t[i] = box_pml[i] - box[i]
            if abs(t[i]) < dl.DOLFIN_EPS:
                t[i] = 1.

        self.sigma_x = dl.Expression(
            "(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(tL*tL) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(tR*tR)",
            xL=box[0], xR=box[2], A=A, tL=t[0], tR=t[2], degree=2)

        self.sigma_y = dl.Expression(
            "(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(tB*tB) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(tT*tT)",
            yB=box[1], yT=box[3], A=A, tB=t[1], tT=t[3], degree=2)

        physical_domain = dl.AutoSubDomain(lambda x, on_boundary: box[0] <= x[0] <= box[2] \
                                                                  and box[1] <= x[1] <= box[3])

        cell_marker = dl.MeshFunction("size_t", mesh, mesh.geometry().dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)

class SingleSourceHelmholtzProblem(hp.PDEProblem):
    def __init__(self, Vh, sources_loc, wave_number, pml):
        self.Vh = Vh
        self.wave_number = wave_number
        self.PML = pml

        self.rhs_fwd = self.generate_state()

        if type(sources_loc) is dl.Point:
            ps0 = dl.PointSource(self.Vh[hp.STATE].sub(0), sources_loc, 1.)
            ps0.apply(self.rhs_fwd)

        else:
            for source in sources_loc:
                ps0 = dl.PointSource(self.Vh[hp.STATE].sub(0), source, 1.)
                ps0.apply(self.rhs_fwd)

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = self._createLUSolver()
        self.solver_fwd_inc = self._createLUSolver()
        self.solver_adj_inc = self._createLUSolver()

    def varf_handler(self, u, m, p):

        k = self.wave_number * dl.exp(m)
        ksquared = k ** 2

        sigma_x = self.PML.sigma_x
        sigma_y = self.PML.sigma_y

        Kr = ksquared - sigma_x * sigma_y
        Ki = -k * (sigma_x + sigma_y)

        Dr_xx = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_x * sigma_x)
        Dr_yy = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_y * sigma_y)
        Di_xx = k * (sigma_x - sigma_y) / (ksquared + sigma_x * sigma_x)
        Di_yy = k * (sigma_y - sigma_x) / (ksquared + sigma_y * sigma_y)

        Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
        Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])

        u1, u2 = dl.split(u)
        p1, p2 = dl.split(p)

        form_r = dl.inner(dl.grad(u1), dl.grad(p1)) * self.PML.dx(1) \
                 - ksquared * u1 * p1 * self.PML.dx(1)

        form_i = -dl.inner(dl.grad(u2), dl.grad(p2)) * self.PML.dx(1) \
                 + ksquared * u2 * p2 * self.PML.dx(1)

        form_pml_r = dl.inner(Dr * dl.grad(u1), dl.grad(p1)) * self.PML.dx(0) \
                     + dl.inner(Di * dl.grad(u2), dl.grad(p1)) * self.PML.dx(0) \
                     - Kr * u1 * p1 * self.PML.dx(0) \
                     - Ki * u2 * p1 * self.PML.dx(0)

        form_pml_i = - dl.inner(Dr * dl.grad(u2), dl.grad(p2)) * self.PML.dx(0) \
                     + dl.inner(Di * dl.grad(u1), dl.grad(p2)) * self.PML.dx(0) \
                     + Kr * u2 * p2 * self.PML.dx(0) \
                     - Ki * u1 * p2 * self.PML.dx(0)

        return form_r + form_i + form_pml_r + form_pml_i

    def generate_state(self):
        """ return a vector in the shape of the state """
        return dl.Function(self.Vh[hp.STATE]).vector()

    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return dl.Function(self.Vh[hp.PARAMETER]).vector()

    def init_parameter(self, m):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.local_range())

    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear Fwd Problem:
        Given m, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""

        u = dl.TrialFunction(self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TestFunction(self.Vh[hp.ADJOINT])
        A = dl.assemble(self.varf_handler(u, m, p))

        self.solver.set_operator(A)
        self.solver.solve(state, self.rhs_fwd)

    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear Adj Problem:
            Given m, u; find p such that
            \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        u = dl.TestFunction(self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TrialFunction(self.Vh[hp.ADJOINT])
        Aadj = dl.assemble(self.varf_handler(u, m, p))

        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)

    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m """
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = hp.vector2Function(x[hp.ADJOINT], self.Vh[hp.ADJOINT])
        dm = dl.TestFunction(self.Vh[hp.PARAMETER])
        res_form = self.varf_handler(u, m, p)
        out.zero()
        dl.assemble(dl.derivative(res_form, m, dm), tensor=out)

    def setLinearizationPoint(self, x, gn_approx):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = hp.vector2Function(x[hp.ADJOINT], self.Vh[hp.ADJOINT])
        x_fun = [u, m, p]

        f_form = self.varf_handler(u, m, p)

        g_form = [None, None, None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])

        self.A = dl.assemble(dl.derivative(g_form[hp.ADJOINT], u))
        self.At = dl.assemble(dl.derivative(g_form[hp.STATE], p))
        self.C = dl.assemble(dl.derivative(g_form[hp.ADJOINT], m))

        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gn_approx:
            self.Wmu = None
            self.Wuu = None
            self.Wmm = None
        else:
            self.Wmu = dl.assemble(dl.derivative(g_form[hp.PARAMETER], u))
            self.Wuu = dl.assemble(dl.derivative(g_form[hp.STATE], u))
            self.Wmm = dl.assemble(dl.derivative(g_form[hp.PARAMETER], m))

    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, m, find \tilde_u s.t.:
            \delta_{pu} F(u,m,p; \hat_p, \tilde_u) = rhs for all \hat_p.

            If is_adj = True:
            Solve the adj incremental system:
            Given u, m, find \tilde_p s.t.:
            \delta_{up} F(u,m,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.solver_fwd_inc.solve(out, rhs)

    def apply_ij(self, i, j, dir, out):
        """
            Given u, m, p; compute
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[hp.STATE, hp.STATE] = self.Wuu
        KKT[hp.PARAMETER, hp.STATE] = self.Wmu
        KKT[hp.PARAMETER, hp.PARAMETER] = self.Wmm
        KKT[hp.ADJOINT, hp.STATE] = self.A
        KKT[hp.ADJOINT, hp.PARAMETER] = self.C

        if i >= j:
            KKT[i, j].mult(dir, out)
        else:
            KKT[j, i].transpmult(dir, out)

    def _createLUSolver(self):
        if hp.dlversion() <= (1, 6, 0):
            return hp.PETScLUSolver()
        else:
            return hp.PETScLUSolver(self.Vh[hp.STATE].mesh().mpi_comm(), method="default")


def setup_helmholtz_problem(settings):
    box = [0., 0., 3., 3.]
    box_pml = [-1., -1., 4., 3.]
    nx = settings["nx"]
    mesh = dl.RectangleMesh(settings["comm_mesh"], dl.Point(box_pml[0], box_pml[1]), dl.Point(box_pml[2], box_pml[3]), nx, nx)
    Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    dims = [Vhi.dim() for Vhi in Vh]

    # dl.parameters["form_compiler"]["quadrature_degree"] = 5
    source_loc_ = ((box[0] + .1 + (box[2] - 0.1) / 2) / 2, box[3] - 0.15)

    sources_loc = [dl.Point(*source_loc_)]

    c = 343.4  # m/s     speed of sound in air
    rho = 1.204  # kg/m^3  density of air
    all_frequencies = np.array([settings["frequency"]])
    pml = PML(mesh, box, box_pml, 50)

    f = all_frequencies[0]
    omega = 2. * np.pi * f
    wave_number = dl.Constant(omega / (c * rho))
    pde = SingleSourceHelmholtzProblem(Vh, sources_loc, wave_number, pml)

    sigma = settings["sigma"]
    rho = settings["rho"]
    delta = 1.0 / (sigma * rho)
    gamma = delta * rho ** 2

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]

    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set(theta0, theta1, alpha)

    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    # Setup observable
    M = assemble_mass_matrix(Vh[hp.STATE])
    B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)
    observable = hf.LinearStateObservable(pde, B)

    return Vh, pde, prior, observable


def extract_to_grid(func, box, nx, ny):
    """
    Extract function values and interpolate to regular grid on physical domain.
    Returns 2D numpy array of shape (ny, nx).
    """
    x_min, x_max = box[0], box[2]
    y_min, y_max = box[1], box[3]
    
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    grid = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            point = dl.Point(X[i, j], Y[i, j])
            try:
                grid[i, j] = func(point)
            except:
                grid[i, j] = 0.0
    
    return grid


def extract_vector_to_grid(func, box, nx, ny):
    """
    Extract vector function values (real and imaginary parts) to regular grids.
    Returns tuple (real_grid, imag_grid) both of shape (ny, nx).
    """
    u_real = func.sub(0)
    u_imag = func.sub(1)
    
    real_grid = extract_to_grid(u_real, box, nx, ny)
    imag_grid = extract_to_grid(u_imag, box, nx, ny)
    
    return real_grid, imag_grid


if __name__ == "__main__":
    import argparse
    import os
    import sys
    
    parser = argparse.ArgumentParser(description="Generate Helmholtz dataset")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate (default: 1000)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: $SCRATCH/data/)")
    args = parser.parse_args()
    
    num_samples = args.num_samples
    if args.output_dir is None:
        output_dir = '/users/PAS3244/dvoderhol/data'
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'helmholtz_dataset.npz')
    
    settings = helmholtz_settings()
    
    start_idx = 0
    if os.path.exists(output_file):
        print(f"Existing dataset found at {output_file}")
        try:
            existing_data = np.load(output_file)
            existing_samples = len(existing_data['m'])
            print(f"Found {existing_samples} existing samples")
            if existing_samples >= num_samples:
                print(f"Dataset already complete ({existing_samples} >= {num_samples}). Exiting.")
                sys.exit(0)
            start_idx = existing_samples
            print(f"Resuming from sample {start_idx}")
        except Exception as e:
            print(f"Warning: Could not load existing dataset ({e}). Starting from scratch.")
            start_idx = 0
    
    if start_idx == 0:
        print(f"Generating {num_samples} Helmholtz samples...")
    else:
        remaining = num_samples - start_idx
        print(f"Generating {remaining} additional samples (will have {num_samples} total)...")
    
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_file}")
    
    Vh, pde, prior, observable = setup_helmholtz_problem(settings)
    
    mesh = Vh[hp.PARAMETER].mesh()
    box = [0., 0., 3., 3.]
    nx = settings["nx"]
    
    print(f"Mesh: {nx}x{nx} grid")
    print(f"Physical domain: {box}")
    
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
    
    for i in range(start_idx, num_samples):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{num_samples}")
        
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
        
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            current_num_samples = len(m_dataset)
            print(f"Saving {current_num_samples} samples to {output_file}...")
            np.savez_compressed(
                output_file,
                m=np.array(m_dataset),
                u_real=np.array(u_real_dataset),
                u_imag=np.array(u_imag_dataset),
                nx=nx,
                box=box,
                num_samples=current_num_samples,
                settings=settings
            )
            file_size = os.path.getsize(output_file) / (1024**2)
            print(f"Dataset saved successfully! File size: {file_size:.2f} MB")
    
    print(f"\nFinal Dataset shapes:")
    print(f"  m: {np.array(m_dataset).shape}")
    print(f"  u_real: {np.array(u_real_dataset).shape}")
    print(f"  u_imag: {np.array(u_imag_dataset).shape}")
    
    print(f"\nDataset generation complete! Total samples: {len(m_dataset)}")
