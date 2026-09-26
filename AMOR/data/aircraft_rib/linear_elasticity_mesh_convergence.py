"""Linear elasticity FEM on rib tet meshes; mesh-to-mesh solution change study.

Also provides solve_elasticity / GP-E helpers used by dataset generators.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Clip GP nodal E to [DEFAULT_GP_E_MIN, DEFAULT_GP_E_MAX]; same numeric scale as `E` / `mean_E`
# in solve_elasticity (rib scripts use ~70e3 as nominal Al-like stiffness; not SI-Pa unless stated elsewhere).
DEFAULT_GP_E_MIN = 5_000.0
DEFAULT_GP_E_MAX = 250_000.0

try:
    import psutil
    AVAILABLE_RAM_GB = psutil.virtual_memory().available / 1e9
except ImportError:
    AVAILABLE_RAM_GB = 16.0  # conservative fallback

_PETSC_INITIALIZED = False
_DOLFIN_IMPORTED = None


def _petsc_init_once():
    """Initialize petsc4py once (required before any PETSc object)."""
    global _PETSC_INITIALIZED
    if _PETSC_INITIALIZED:
        return
    import petsc4py

    petsc4py.init(sys.argv)
    _PETSC_INITIALIZED = True


def _petsc_direct_solver_names():
    raw = os.environ.get("CONVERGENCE_PETSC_SOLVERS", "").strip()
    if not raw:
        return ("superlu_dist", "mumps", "petsc")
    names = tuple(s.strip() for s in raw.split(",") if s.strip())
    return names if names else ("superlu_dist", "mumps", "petsc")


def _import_dolfin():
    """
    Import dolfin after patching PKG_CONFIG_PATH/LD_LIBRARY_PATH for conda FEniCS.
    Raises RuntimeError with actionable guidance when unavailable.
    """
    global _DOLFIN_IMPORTED
    if _DOLFIN_IMPORTED is not None:
        return _DOLFIN_IMPORTED
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        pkg_cfg = os.path.join(conda_prefix, "lib", "pkgconfig")
        if os.path.isdir(pkg_cfg):
            cur = os.environ.get("PKG_CONFIG_PATH", "")
            if pkg_cfg not in cur.split(":"):
                os.environ["PKG_CONFIG_PATH"] = f"{pkg_cfg}:{cur}" if cur else pkg_cfg
        lib_dir = os.path.join(conda_prefix, "lib")
        if os.path.isdir(lib_dir):
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in cur.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur}" if cur else lib_dir
    try:
        import dolfin as df
    except Exception as ex:
        raise RuntimeError(
            "FEniCS (dolfin) is not importable in the current runtime. "
            "Activate a FEniCS-capable env/module and retry. "
            f"Original error: {type(ex).__name__}: {ex}"
        ) from ex
    _DOLFIN_IMPORTED = df
    return df


def _sample_gp_field(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mean_E: float = 70e3,
    std_m: float = 0.4,
    corr_len: float = 80.0,
    grid_resolution: int = 64,
    seed: int = 42,
    e_min: float = DEFAULT_GP_E_MIN,
    e_max: float = DEFAULT_GP_E_MAX,
) -> dict:
    """Sample lognormal material field E(x)=exp(m(x))+1 on regular 3D grid; clip E to [e_min, e_max]."""
    emin = float(e_min)
    emax = float(e_max)
    if emin <= 0.0:
        raise ValueError(f"e_min must be positive, got {emin}")
    if emax <= emin:
        raise ValueError(f"require e_max > e_min, got e_min={emin}, e_max={emax}")
    rng = np.random.default_rng(seed)
    extent = np.maximum(np.asarray(bbox_max, dtype=float) - np.asarray(bbox_min, dtype=float), 1e-12)
    max_extent = float(np.max(extent))
    nx = max(4, int(grid_resolution * extent[0] / max_extent))
    ny = max(4, int(grid_resolution * extent[1] / max_extent))
    nz = max(4, int(grid_resolution * extent[2] / max_extent))

    white = rng.standard_normal((nx, ny, nz))
    dx = extent[0] / nx
    dy = extent[1] / ny
    dz = extent[2] / nz
    sigma = [max(corr_len / max(dx, 1e-12), 1e-6),
             max(corr_len / max(dy, 1e-12), 1e-6),
             max(corr_len / max(dz, 1e-12), 1e-6)]
    m0 = gaussian_filter(white, sigma=sigma, mode="wrap")
    m0 = (m0 - float(m0.mean())) / (float(m0.std()) + 1e-12)
    m_mean = float(np.log(max(mean_E - 1.0, 1e-12)))
    m = m_mean + std_m * m0
    E = np.clip(np.exp(m) + 1.0, emin, emax)
    return {
        "m_field": m,
        "E_field": E,
        "x": np.linspace(float(bbox_min[0]), float(bbox_max[0]), nx),
        "y": np.linspace(float(bbox_min[1]), float(bbox_max[1]), ny),
        "z": np.linspace(float(bbox_min[2]), float(bbox_max[2]), nz),
        "seed": int(seed),
        "std_m": float(std_m),
        "corr_len": float(corr_len),
        "grid_resolution": int(grid_resolution),
        "mean_E_target": float(mean_E),
        "m_mean": m_mean,
        "e_min": emin,
        "e_max": emax,
    }


def _interp_gp_E_to_nodes(gp_data: dict, nodes: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of regular-grid GP E-field to mesh nodes."""
    x = gp_data["x"]; y = gp_data["y"]; z = gp_data["z"]; field = gp_data["E_field"]
    ix = (nodes[:, 0] - x[0]) / (x[-1] - x[0] + 1e-30) * (len(x) - 1)
    iy = (nodes[:, 1] - y[0]) / (y[-1] - y[0] + 1e-30) * (len(y) - 1)
    iz = (nodes[:, 2] - z[0]) / (z[-1] - z[0] + 1e-30) * (len(z) - 1)
    ix = np.clip(ix, 0, len(x) - 1.001); iy = np.clip(iy, 0, len(y) - 1.001); iz = np.clip(iz, 0, len(z) - 1.001)
    ix0 = ix.astype(int); iy0 = iy.astype(int); iz0 = iz.astype(int)
    fx = ix - ix0; fy = iy - iy0; fz = iz - iz0
    ix1 = np.minimum(ix0 + 1, len(x) - 1); iy1 = np.minimum(iy0 + 1, len(y) - 1); iz1 = np.minimum(iz0 + 1, len(z) - 1)
    return (
        field[ix0, iy0, iz0] * (1 - fx) * (1 - fy) * (1 - fz)
        + field[ix1, iy0, iz0] * fx * (1 - fy) * (1 - fz)
        + field[ix0, iy1, iz0] * (1 - fx) * fy * (1 - fz)
        + field[ix0, iy0, iz1] * (1 - fx) * (1 - fy) * fz
        + field[ix1, iy1, iz0] * fx * fy * (1 - fz)
        + field[ix1, iy0, iz1] * fx * (1 - fy) * fz
        + field[ix0, iy1, iz1] * (1 - fx) * fy * fz
        + field[ix1, iy1, iz1] * fx * fy * fz
    )


def _solve_free_system_petsc(K_free, f_free):
    """
    Direct LU via PETSc KSP PREONLY + PC LU; tries each name from _petsc_direct_solver_names().
    Each attempt uses a new KSP/Mat/Vec (PETSc forbids changing factor solver after setup).
    """
    _petsc_init_once()
    from petsc4py import PETSc

    K_csr = csr_matrix(K_free, dtype=np.float64)
    K_csr.sort_indices()
    K_csr.sum_duplicates()
    n = int(K_csr.shape[0])
    f64 = np.asarray(f_free, dtype=np.float64).ravel()
    if f64.shape[0] != n:
        raise ValueError("K_free and f_free size mismatch")

    ia = K_csr.indptr.astype(PETSc.IntType, copy=False)
    ja = K_csr.indices.astype(PETSc.IntType, copy=False)
    data = K_csr.data.astype(PETSc.ScalarType, copy=False)

    fn = float(np.linalg.norm(f64))
    t0 = time.time()
    last_err = None
    for solver_name in _petsc_direct_solver_names():
        mat = b = x = ksp = None
        try:
            mat = PETSc.Mat().createAIJ(size=K_csr.shape, csr=(ia, ja, data))
            mat.assemblyBegin()
            mat.assemblyEnd()
            b = PETSc.Vec().createSeq(n)
            b.setArray(f64.astype(PETSc.ScalarType, copy=False))
            b.assemblyBegin()
            b.assemblyEnd()
            x = PETSc.Vec().createSeq(n)
            x.set(0.0)
            ksp = PETSc.KSP().create()
            ksp.setOperators(mat)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType(solver_name)
            ksp.setFromOptions()
            ksp.solve(b, x)
            u_free = np.array(x.getArray(readonly=True), copy=True)
            ru = K_csr @ u_free - f64
            relres = float(np.linalg.norm(ru) / fn) if fn > 0 else 0.0
            t_solve = time.time() - t0
            tag = f"direct (PETSc LU / {solver_name})"
            print(f"{t_solve:.1f}s (relres={relres:.2e}, {tag})", flush=True)
            return u_free, True, tag, 0, relres, t_solve
        except Exception as ex:
            last_err = ex
            print(
                f"\n    PETSc LU backend {solver_name!r} failed: {type(ex).__name__}: {ex}",
                flush=True,
            )
        finally:
            if ksp is not None:
                ksp.destroy()
            if x is not None:
                x.destroy()
            if b is not None:
                b.destroy()
            if mat is not None:
                mat.destroy()

    t_solve = time.time() - t0
    print(f"{t_solve:.1f}s (all PETSc LU backends failed)", flush=True)
    u_free = np.zeros(n)
    msg = f"PETSc LU failed ({type(last_err).__name__}: {last_err})" if last_err else "PETSc LU failed"
    return u_free, False, msg, -1, float("nan"), t_solve

# Fractions of max_ram_gb used when comparing to rough memory models (raised from 0.6/0.5).
_MEM_ASSEMBLE_FRAC = 0.75
_MEM_DIRECT_FRAC = 0.78
_MEM_ITER_FRAC = 0.78
# Default cap for direct factorization DOF count (OOM risk). Override with env
# CONVERGENCE_DIRECT_MAX_DOFS (integer); 0 or negative disables the cap.
def _direct_safe_dof_limit() -> int:
    raw = os.environ.get("CONVERGENCE_DIRECT_MAX_DOFS", "").strip()
    if not raw:
        return 2_000_000
    try:
        n = int(float(raw))
    except ValueError:
        return 2_000_000
    if n <= 0:
        return 2**62
    return n


def _primary_axis(nodes: np.ndarray) -> int:
    extents = nodes.max(axis=0) - nodes.min(axis=0)
    return int(np.argmax(extents))


def _bc_geom_slabs(nodes: np.ndarray, primary_axis: int, total_load: float):
    """2% extent slab: full clamp on min, axial nodal forces on max (total axial load = total_load)."""
    extents = nodes.max(axis=0) - nodes.min(axis=0)
    x_min = nodes[:, primary_axis].min()
    x_max = nodes[:, primary_axis].max()
    tol = extents[primary_axis] * 0.02
    fixed_nodes = np.where(nodes[:, primary_axis] < x_min + tol)[0]
    loaded_nodes = np.where(nodes[:, primary_axis] > x_max - tol)[0]
    n_dofs = 3 * len(nodes)
    f = np.zeros(n_dofs)
    fpn = total_load / max(len(loaded_nodes), 1)
    f[3 * loaded_nodes + primary_axis] = fpn
    fixed_dofs = np.concatenate([3 * fixed_nodes + j for j in range(3)])
    return fixed_dofs, f, fixed_nodes, loaded_nodes


def _dolfin_petsc_to_csr(df, A) -> csr_matrix:
    """Convert dolfin PETScMatrix to SciPy CSR."""
    mat = df.as_backend_type(A).mat()
    indptr, indices, data = mat.getValuesCSR()
    n0, n1 = mat.size
    return csr_matrix((data, indices, indptr), shape=(n0, n1))


def _solve_free_system(K_free, f_free, use_iterative: bool, petsc_direct: bool = False):
    """Direct spsolve / optional PETSc LU, or multi-attempt CG+ILU; returns rel. residual ||r||/||b||."""
    t0 = time.time()
    fn = float(np.linalg.norm(f_free))
    if not use_iterative:
        if petsc_direct:
            try:
                petsc_out = _solve_free_system_petsc(K_free, f_free)
                if petsc_out[1]:
                    return petsc_out
                print(
                    "    PETSc LU backends did not succeed; falling back to scipy spsolve",
                    flush=True,
                )
            except ImportError as ex:
                t_solve = time.time() - t0
                print(
                    f"{t_solve:.1f}s (PETSc unavailable: {ex}; falling back to scipy spsolve)",
                    flush=True,
                )
            except Exception as ex:
                t_solve = time.time() - t0
                print(
                    f"{t_solve:.1f}s (PETSc path error: {type(ex).__name__}: {ex}; "
                    f"falling back to scipy spsolve)",
                    flush=True,
                )
        try:
            u_free = spsolve(K_free, f_free)
        except Exception as ex:
            t_solve = time.time() - t0
            print(f"{t_solve:.1f}s (direct spsolve failed: {ex})", flush=True)
            u_free = np.zeros(len(f_free))
            return (
                u_free,
                False,
                f"direct failed ({type(ex).__name__})",
                -1,
                float("nan"),
                t_solve,
            )
        t_solve = time.time() - t0
        ru = K_free @ u_free - f_free
        relres = float(np.linalg.norm(ru) / fn) if fn > 0 else 0.0
        tag = "direct (spsolve)"
        if petsc_direct:
            tag += " [fallback after PETSc]"
        print(f"{t_solve:.1f}s (relres={relres:.2e})")
        return u_free, True, tag, 0, relres, t_solve

    K_csc = csc_matrix(K_free)
    configs = [
        (1e-5, 28, 6000, 1e-7),
        (1e-6, 45, 15000, 1e-6),
        (5e-7, 60, 25000, 1e-5),
    ]
    last_info = -999
    for drop_tol, fill_factor, maxiter, rtol in configs:
        try:
            t_pre = time.time()
            ilu = spilu(K_csc, drop_tol=drop_tol, fill_factor=fill_factor)
            M = LinearOperator(K_free.shape, matvec=ilu.solve)
            t_b = time.time() - t_pre
            u_free, info = cg(K_free, f_free, M=M, rtol=rtol, atol=0.0, maxiter=maxiter)
            last_info = int(info)
            ru = K_free @ u_free - f_free
            relres = float(np.linalg.norm(ru) / fn) if fn > 0 else 0.0
            ok = (info == 0) or relres < 1e-5
            if ok:
                t_solve = time.time() - t0
                print(f"{t_solve:.1f}s (ILU dt={drop_tol} build={t_b:.1f}s info={info} relres={relres:.2e})")
                return u_free, True, f"iter CG ILU(dt={drop_tol})", last_info, relres, t_solve
            print(f"\n    CG info={info} relres={relres:.2e} — retry...", flush=True)
        except Exception as ex:
            print(f"\n    ILU/CG attempt failed: {ex}", flush=True)
            continue
    u_free = np.zeros(len(f_free))
    t_solve = time.time() - t0
    print(f"{t_solve:.1f}s (all iterative attempts failed)")
    return u_free, False, "iterative (CG failed)", int(last_info), float("nan"), t_solve


def _bc_surface_faces(nodes: np.ndarray, elements: np.ndarray, primary_axis: int,
                      total_load: float, cos_thresh: float = 0.55):
    """
    Boundary triangles from connectivity; clamp all DOFs on faces whose outward normal
    aligns with -e_axis; axial nodal forces from area-weighted lumping on +e_axis faces.
    Falls back to geom slabs if classification is empty.
    """
    nt = elements.shape[0]
    opp = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)
    fi_all = elements[:, opp]
    keys_sorted = np.sort(fi_all, axis=2)
    flat_k = keys_sorted.reshape(-1, 3)
    _, inv, cnt = np.unique(flat_k, return_inverse=True, return_counts=True, axis=0)
    bd_flat = (cnt == 1)[inv]
    bd_idx = np.flatnonzero(bd_flat)
    if bd_idx.size == 0:
        return None
    ti = bd_idx // 4
    lf = bd_idx % 4
    v_face = fi_all[ti, lf]
    el_t = elements[ti]
    p0 = nodes[v_face[:, 0]]
    p1 = nodes[v_face[:, 1]]
    p2 = nodes[v_face[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    alen = np.linalg.norm(cross, axis=1, keepdims=True)
    nvec = np.divide(cross, alen, out=np.zeros_like(cross), where=alen > 1e-20)
    c_face = (p0 + p1 + p2) / 3.0
    c_tet = nodes[el_t].mean(axis=1)
    outvec = c_face - c_tet
    flip = np.sum(nvec * outvec, axis=1) < 0
    nvec[flip] *= -1.0
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    e_axis = np.zeros(3, dtype=np.float64)
    e_axis[primary_axis] = 1.0
    nd = np.dot(nvec, e_axis)
    left_m = nd < -cos_thresh
    right_m = nd > cos_thresh
    if not np.any(left_m) or not np.any(right_m):
        return None

    fixed_nodes = np.unique(v_face[left_m].ravel())
    A_r = float(area[right_m].sum())
    if A_r <= 0:
        return None
    n_nodes = len(nodes)
    n_dofs = 3 * n_nodes
    f = np.zeros(n_dofs)
    nodes_r = v_face[right_m]
    ar = area[right_m][:, np.newaxis]
    w = total_load * ar / (3.0 * A_r)
    f_ax = np.zeros(n_nodes, dtype=np.float64)
    for j in range(3):
        np.add.at(f_ax, nodes_r[:, j], w[:, 0])
    f[np.arange(n_nodes) * 3 + primary_axis] = f_ax
    fixed_dofs = np.concatenate([3 * fixed_nodes + j for j in range(3)])
    loaded_nodes = np.unique(nodes_r.ravel())
    return fixed_dofs, f, fixed_nodes, loaded_nodes


# ============================================================================
# VECTORIZED 3D LINEAR ELASTICITY SOLVER
# ============================================================================

def _facet_keys_from_tri_arrays(tris: np.ndarray) -> set:
    """Map (n,3) int node indices of surface triangles to frozenset keys for facet matching."""
    if tris is None or len(tris) == 0:
        return set()
    a = np.asarray(tris, dtype=np.int64)
    return {frozenset(int(a[i, j]) for j in range(3)) for i in range(a.shape[0])}


def solve_elasticity(nodes: np.ndarray, elements: np.ndarray,
                     E: float = 70e3, nu: float = 0.3,
                     max_ram_gb: float = None,
                     bc_marking: str = "geom",
                     total_load: float = 100.0,
                     load_direction: np.ndarray = None,
                     petsc_direct: bool = None,
                     solver_backend: str = "petsc_auto",
                     E_field: np.ndarray = None,
                     dirichlet_face_set: set = None,
                     neumann_face_set: set = None):
    """
    Solve 3D linear elasticity on a tet mesh using FEniCS (dolfin) assembly
    and PETSc linear solvers.
    """
    df = _import_dolfin()
    n_nodes = len(nodes)
    n_tets = len(elements)
    n_dofs = 3 * n_nodes
    primary_axis = _primary_axis(nodes)
    bc_marking_l = (bc_marking or "geom").strip().lower()

    if max_ram_gb is None:
        max_ram_gb = AVAILABLE_RAM_GB * 0.7

    if petsc_direct is None:
        petsc_direct = os.environ.get("CONVERGENCE_USE_PETSC", "").strip() not in (
            "",
            "0",
            "false",
            "False",
            "no",
            "No",
        )

    # Rough memory screens retained to avoid pathological OOM jobs.
    mem_assembly_gb = n_tets * 144 * 24 / 1e9 + n_tets * 4 * 3 * 8 / 1e9
    mem_solve_direct_gb = 25 * n_dofs ** (4 / 3) * 8 / 1e9
    mem_ilu_gb = 80 * n_dofs * 3 * 8 / 1e9

    can_assemble = mem_assembly_gb < max_ram_gb * _MEM_ASSEMBLE_FRAC
    dof_cap = _direct_safe_dof_limit()
    can_direct = (
        mem_solve_direct_gb < max_ram_gb * _MEM_DIRECT_FRAC
        and n_dofs <= dof_cap
    )
    can_iterative = mem_ilu_gb < max_ram_gb * _MEM_ITER_FRAC
    use_iterative = not can_direct

    print(f"    n_tets={n_tets:,}, n_dofs={n_dofs:,}")
    print(f"    Est. mem — assembly: {mem_assembly_gb:.1f} GB, "
          f"direct: {mem_solve_direct_gb:.1f} GB, "
          f"iterative: {mem_ilu_gb:.1f} GB, "
          f"avail: {max_ram_gb:.1f} GB")

    if not can_assemble:
        print(f"    ✗ SKIPPING — assembly alone would exceed RAM")
        return {
            "u_field": np.zeros((n_nodes, 3)),
            "u_magnitude": np.zeros(n_nodes),
            "max_displacement": 0.0,
            "rms_displacement": 0.0,
            "mean_displacement_axial": 0.0,
            "strain_energy": 0.0,
            "n_nodes": n_nodes, "n_tets": n_tets, "n_dofs": n_dofs,
            "n_free_dofs": 0, "n_fixed_nodes": 0, "n_loaded_nodes": 0,
            "nnz": 0, "t_assemble": 0.0, "t_solve": 0.0,
            "solve_ok": False, "solver": "skipped (OOM)",
            "solver_info": -1,
            "relative_residual": float("nan"),
            "bc_marking": bc_marking_l,
            "primary_axis": primary_axis,
            "total_load": float(total_load),
            "volume_domain": 0.0,
            "h_characteristic": 0.0,
        }

    if not can_direct and not can_iterative:
        print(f"    ✗ SKIPPING — neither solver fits in RAM")
        return {
            "u_field": np.zeros((n_nodes, 3)),
            "u_magnitude": np.zeros(n_nodes),
            "max_displacement": 0.0,
            "rms_displacement": 0.0,
            "mean_displacement_axial": 0.0,
            "strain_energy": 0.0,
            "n_nodes": n_nodes, "n_tets": n_tets, "n_dofs": n_dofs,
            "n_free_dofs": 0, "n_fixed_nodes": 0, "n_loaded_nodes": 0,
            "nnz": 0, "t_assemble": 0.0, "t_solve": 0.0,
            "solve_ok": False, "solver": "skipped (OOM)",
            "solver_info": -1,
            "relative_residual": float("nan"),
            "bc_marking": bc_marking_l,
            "primary_axis": primary_axis,
            "total_load": float(total_load),
            "volume_domain": 0.0,
            "h_characteristic": 0.0,
        }

    if solver_backend == "petsc_auto":
        if use_iterative:
            print("    → Using PETSc iterative solve (cg + ilu)")
        else:
            print(f"    → Using PETSc direct solve (LU; try-order={_petsc_direct_solver_names()})")
    else:
        print(f"    → Using requested backend: {solver_backend}")

    # ---- FEniCS mesh + forms ----
    print(f"    Assembling K...", end=" ", flush=True)
    t0 = time.time()
    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(n_nodes)
    for i in range(n_nodes):
        editor.add_vertex(i, nodes[i])
    editor.init_cells(n_tets)
    for i in range(n_tets):
        editor.add_cell(i, elements[i])
    editor.close()
    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    V0 = df.FunctionSpace(mesh, "Lagrange", 1)
    if E_field is None:
        E_fn = df.Constant(float(E))
        E_mean = float(E)
        E_std = 0.0
    else:
        E_arr = np.asarray(E_field, dtype=np.float64).reshape(-1)
        if E_arr.shape[0] != n_nodes:
            raise ValueError(f"E_field size mismatch: got {E_arr.shape[0]}, expected {n_nodes}")
        E_arr = np.maximum(E_arr, 1e-9)
        # Vertex-ordered E_nodal → CG1 DOF order (NOT set_local(E) raw — that
        # scrambled physical E when vertex_to_dof ≠ identity).
        probe = df.Function(V0)
        probe.vector().set_local(np.arange(V0.dim(), dtype=np.float64))
        probe.vector().apply("insert")
        vertex_to_dof0 = probe.compute_vertex_values(mesh).astype(np.int64).reshape(-1)
        if vertex_to_dof0.size != n_nodes:
            raise RuntimeError(
                f"E_field DOF map size {vertex_to_dof0.size} != n_nodes {n_nodes}"
            )
        Ev = df.Function(V0)
        e_dof = np.zeros(V0.dim(), dtype=np.float64)
        e_dof[vertex_to_dof0] = E_arr
        Ev.vector().set_local(e_dof)
        Ev.vector().apply("insert")
        E_fn = Ev
        E_mean = float(np.mean(E_arr))
        E_std = float(np.std(E_arr))
    lam = E_fn * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_fn / (2 * (1 + nu))
    def _eps(w):
        return df.sym(df.grad(w))
    def _sigma(w):
        return 2.0 * mu * _eps(w) + lam * df.tr(_eps(w)) * df.Identity(3)
    t_assemble = time.time() - t0
    volume_domain = float(df.assemble(df.Constant(1.0) * df.dx(domain=mesh)))
    h_char = float((volume_domain * 6.0 / max(n_tets, 1)) ** (1.0 / 3.0))

    # ---- Boundary conditions (facet marking) ----
    extents = nodes.max(axis=0) - nodes.min(axis=0)
    x_min = float(nodes[:, primary_axis].min())
    x_max = float(nodes[:, primary_axis].max())
    tol = float(extents[primary_axis] * 0.02)
    D = mesh.topology().dim()
    facets = df.MeshFunction("size_t", mesh, D - 1, 0)

    if bc_marking_l == "marked":
        if dirichlet_face_set is None or neumann_face_set is None:
            raise ValueError(
                "bc_marking='marked' requires dirichlet_face_set and neumann_face_set "
                "(load from *_marked.npz)."
            )
        if len(dirichlet_face_set) == 0:
            raise ValueError("marked BC: dirichlet_face_set is empty.")
        mesh.init(D - 1, D)
        n_facet = mesh.topology().size(D - 1)
        n_mark_dir = 0
        n_mark_neu = 0
        n_mark_skip = 0
        for fid in range(n_facet):
            f = df.Facet(mesh, fid)
            if not f.exterior():
                continue
            vs = f.entities(0)
            key = frozenset(int(vs[j]) for j in range(len(vs)))
            if key in dirichlet_face_set:
                facets[fid] = 1
                n_mark_dir += 1
            elif key in neumann_face_set:
                facets[fid] = 2
                n_mark_neu += 1
            else:
                n_mark_skip += 1
        print(
            f"    marked BC: Dirichlet facets={n_mark_dir:,}, Neumann facets={n_mark_neu:,}, "
            f"other boundary facets={n_mark_skip:,}",
            flush=True,
        )
        if n_mark_dir == 0:
            raise RuntimeError("marked BC: no Dirichlet facets matched mesh boundary.")
        if n_mark_neu == 0:
            print("    WARNING: marked BC has zero Neumann facets; traction ds(2) area is zero.", flush=True)
        bc_effective = "marked_npz"
        fixed_nodes = np.unique(np.fromiter((i for k in dirichlet_face_set for i in k), dtype=np.int64))
        loaded_nodes = (
            np.unique(np.fromiter((i for k in neumann_face_set for i in k), dtype=np.int64))
            if neumann_face_set else np.array([], dtype=np.int64)
        )
    else:
        fixed_nodes = np.where(nodes[:, primary_axis] < x_min + tol)[0]
        loaded_nodes = np.where(nodes[:, primary_axis] > x_max - tol)[0]
        bc_effective = "geom"
        if bc_marking_l == "surface":
            # Surface-face classification in the former SciPy path is not replicated in UFL forms.
            # Keep explicit fallback behavior.
            bc_effective = "geom_fallback"
            print("    surface BC requested; using geom slab BC in FEniCS path", flush=True)
        elif bc_marking_l != "geom":
            raise ValueError("bc_marking must be 'geom', 'surface', or 'marked'")

        class _Left(df.SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[primary_axis] <= x_min + tol + df.DOLFIN_EPS)

        class _Right(df.SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[primary_axis] >= x_max - tol - df.DOLFIN_EPS)

        _Left().mark(facets, 1)
        _Right().mark(facets, 2)
    ds = df.Measure("ds", domain=mesh, subdomain_data=facets)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0, 0.0)), facets, 1)
    right_area = float(df.assemble(df.Constant(1.0) * ds(2)))
    traction = np.zeros(3, dtype=float)
    if load_direction is None:
        load_dir_vec = np.zeros(3, dtype=float)
        load_dir_vec[primary_axis] = 1.0
    else:
        load_dir_vec = np.asarray(load_direction, dtype=float).reshape(3)
        nrm = float(np.linalg.norm(load_dir_vec))
        if nrm <= 0.0:
            raise ValueError("load_direction must have non-zero norm")
        load_dir_vec = load_dir_vec / nrm
    if right_area > 0.0:
        traction[:] = load_dir_vec * (total_load / right_area)
    T = df.Constant(tuple(float(z) for z in traction))
    a = df.inner(_sigma(u), _eps(v)) * df.dx(domain=mesh)
    L = df.dot(T, v) * ds(2)

    A = df.PETScMatrix()
    b = df.PETScVector()
    df.assemble(a, tensor=A)
    df.assemble(L, tensor=b)
    bc.apply(A, b)
    nnz = int(df.as_backend_type(A).mat().getInfo().get("nz_used", 0))
    print(f"{t_assemble:.1f}s (nnz={nnz:,})")

    # Forced direct backends can still be unsafe on very large systems even if iterative would fit.
    # Skip early instead of risking native-factorization crashes (keeps full convergence run alive).
    if solver_backend == "scipy_direct" and not can_direct:
        print("SKIPPED: scipy_direct marked unsafe by memory/DOF guard; continuing to next mesh.", flush=True)
        rib_length = float(extents[primary_axis])
        return {
            "u_field": np.zeros((n_nodes, 3)),
            "u_magnitude": np.zeros(n_nodes),
            "force_axis": np.zeros(n_nodes),
            "max_displacement": 0.0,
            "rms_displacement": 0.0,
            "mean_displacement_axial": 0.0,
            "strain_energy": 0.0,
            "n_nodes": n_nodes,
            "n_tets": n_tets,
            "n_dofs": n_dofs,
            "n_free_dofs": 0,
            "n_fixed_nodes": len(fixed_nodes),
            "n_loaded_nodes": len(loaded_nodes),
            "nnz": nnz,
            "t_assemble": t_assemble,
            "t_solve": 0.0,
            "solve_ok": False,
            "solver": "skipped (scipy_direct unsafe)",
            "solver_info": -2,
            "relative_residual": float("nan"),
            "bc_marking": bc_effective,
            "primary_axis": int(primary_axis),
            "total_load": float(total_load),
            "load_direction": load_dir_vec.copy(),
            "volume_domain": volume_domain,
            "h_characteristic": h_char,
            "rib_length": rib_length,
            "tip_deflection": 0.0,
            "tip_deflection_pct": 0.0,
            "solver_backend": solver_backend,
            "E_mean": E_mean,
            "E_std": E_std,
        }
    if solver_backend == "petsc_lu" and not can_direct:
        print("SKIPPED: petsc_lu marked unsafe by memory/DOF guard; continuing to next mesh.", flush=True)
        rib_length = float(extents[primary_axis])
        return {
            "u_field": np.zeros((n_nodes, 3)),
            "u_magnitude": np.zeros(n_nodes),
            "force_axis": np.zeros(n_nodes),
            "max_displacement": 0.0,
            "rms_displacement": 0.0,
            "mean_displacement_axial": 0.0,
            "strain_energy": 0.0,
            "n_nodes": n_nodes,
            "n_tets": n_tets,
            "n_dofs": n_dofs,
            "n_free_dofs": 0,
            "n_fixed_nodes": len(fixed_nodes),
            "n_loaded_nodes": len(loaded_nodes),
            "nnz": nnz,
            "t_assemble": t_assemble,
            "t_solve": 0.0,
            "solve_ok": False,
            "solver": "skipped (petsc_lu unsafe)",
            "solver_info": -2,
            "relative_residual": float("nan"),
            "bc_marking": bc_effective,
            "primary_axis": int(primary_axis),
            "total_load": float(total_load),
            "load_direction": load_dir_vec.copy(),
            "volume_domain": volume_domain,
            "h_characteristic": h_char,
            "rib_length": rib_length,
            "tip_deflection": 0.0,
            "tip_deflection_pct": 0.0,
            "solver_backend": solver_backend,
            "E_mean": E_mean,
            "E_std": E_std,
        }

    # ---- Solve ----
    free_dofs = V.dim() - len(bc.get_boundary_values())
    print(f"    Solving ({free_dofs:,} free DOFs)...", end=" ", flush=True)
    t0s = time.time()
    try:
        uh = df.Function(V)
        solver_tag = "failed"
        solver_info = -1
        solve_ok = False
        relres = float("nan")
        if solver_backend in ("petsc_auto", "petsc_lu", "petsc_cgilu"):
            run_iter = use_iterative if solver_backend == "petsc_auto" else (solver_backend == "petsc_cgilu")
            if run_iter:
                solver = df.PETScKrylovSolver("cg", "ilu")
                solver.solve(A, uh.vector(), b)
                solver_tag = "PETSc KSP(cg,ilu)"
                solve_ok = True
                solver_info = 0
            else:
                last_ex = None
                for name in _petsc_direct_solver_names():
                    try:
                        solver = df.PETScLUSolver(name)
                        solver.solve(A, uh.vector(), b)
                        solver_tag = f"PETSc LU({name})"
                        solve_ok = True
                        solver_info = 0
                        break
                    except Exception as ex:
                        last_ex = ex
                        print(f"\n    PETSc LU backend {name!r} failed: {type(ex).__name__}: {ex}", flush=True)
                if not solve_ok and last_ex is not None:
                    solver_tag = f"PETSc LU failed ({type(last_ex).__name__})"
        elif solver_backend in ("scipy_direct", "scipy_cgilu"):
            K_full = _dolfin_petsc_to_csr(df, A)
            b_full = np.asarray(b.get_local(), dtype=np.float64)
            bc_map = bc.get_boundary_values()
            fixed = np.array(sorted(int(k) for k in bc_map.keys()), dtype=np.int64)
            free = np.setdiff1d(np.arange(V.dim(), dtype=np.int64), fixed, assume_unique=False)
            if free.size == 0:
                raise RuntimeError("No free DOFs remain after BC application.")
            K_free = K_full[free][:, free]
            b_free = b_full[free]
            if solver_backend == "scipy_direct":
                u_free = spsolve(K_free, b_free)
                solver_tag = "SciPy direct(spsolve)"
                solve_ok = True
                solver_info = 0
            else:
                u_free, ok, tag, info, _, _ = _solve_free_system(
                    K_free, b_free, use_iterative=True, petsc_direct=False
                )
                solver_tag = f"SciPy {tag}"
                solve_ok = bool(ok)
                solver_info = int(info)
            u_full = np.zeros(V.dim(), dtype=np.float64)
            u_full[free] = np.asarray(u_free, dtype=np.float64)
            uh.vector().set_local(u_full)
            uh.vector().apply("insert")
        else:
            raise ValueError(f"Unknown solver_backend={solver_backend}")
        t_solve = time.time() - t0s
        if solve_ok:
            Au = df.PETScVector()
            A.mult(uh.vector(), Au)
            r = df.PETScVector()
            r.init(Au.size())
            r.zero()
            r.axpy(1.0, Au)
            r.axpy(-1.0, b)
            bnorm = b.norm("l2")
            relres = float(r.norm("l2") / bnorm) if bnorm > 0 else 0.0
            print(f"{t_solve:.1f}s (relres={relres:.2e}, {solver_tag})", flush=True)
        else:
            print(f"{t_solve:.1f}s ({solver_tag})", flush=True)
    except Exception as e:
        print(f"FAILED: {e}")
        t_solve = time.time() - t0s
        solve_ok = False
        solver_tag = "failed"
        solver_info = -1
        relres = float("nan")
        uh = df.Function(V)

    uv = uh.compute_vertex_values(mesh).reshape(3, n_nodes).T
    u_field = np.asarray(uv, dtype=float)
    force_axis = np.zeros(n_nodes, dtype=float)
    if len(loaded_nodes) > 0:
        force_axis[loaded_nodes] = total_load / len(loaded_nodes)

    u_magnitude = np.linalg.norm(u_field, axis=1)
    strain_energy = float(0.5 * df.assemble(df.inner(_sigma(uh), _eps(uh)) * df.dx(domain=mesh)))
    max_disp = u_magnitude.max()
    rms_disp = float(np.sqrt(np.mean(u_magnitude ** 2)))
    mean_axial = float(np.mean(u_field[:, primary_axis]))
    rib_length = float(extents[primary_axis])
    if len(loaded_nodes) > 0:
        tip_deflection = float(np.max(u_magnitude[loaded_nodes]))
    else:
        tip_deflection = float(np.max(u_magnitude))
    tip_deflection_pct = 100.0 * tip_deflection / max(rib_length, 1e-30)

    return {
        "u_field": u_field,
        "u_magnitude": u_magnitude,
        "force_axis": force_axis,
        "max_displacement": float(max_disp),
        "rms_displacement": rms_disp,
        "mean_displacement_axial": mean_axial,
        "strain_energy": float(strain_energy),
        "n_nodes": n_nodes,
        "n_tets": n_tets,
        "n_dofs": n_dofs,
        "n_free_dofs": int(free_dofs),
        "n_fixed_nodes": len(fixed_nodes),
        "n_loaded_nodes": len(loaded_nodes),
        "nnz": nnz,
        "t_assemble": t_assemble,
        "t_solve": t_solve,
        "solve_ok": solve_ok,
        "solver": solver_tag,
        "solver_info": int(solver_info),
        "relative_residual": float(relres),
        "bc_marking": bc_effective,
        "primary_axis": int(primary_axis),
        "total_load": float(total_load),
        "load_direction": load_dir_vec.copy(),
        "volume_domain": volume_domain,
        "h_characteristic": h_char,
        "rib_length": rib_length,
        "tip_deflection": tip_deflection,
        "tip_deflection_pct": tip_deflection_pct,
        "solver_backend": solver_backend,
        "E_mean": E_mean,
        "E_std": E_std,
    }


# ============================================================================
# INTERPOLATION
# ============================================================================

def interpolate_to_reference(nodes_src, u_src, nodes_ref):
    tree = cKDTree(nodes_src)
    dist, idx = tree.query(nodes_ref, k=4)
    weights = 1.0 / (dist + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    u_interp = np.zeros((len(nodes_ref), 3))
    for j in range(3):
        u_interp[:, j] = (weights * u_src[:, j][idx]).sum(axis=1)
    return u_interp


def _failed_elasticity_result(n_nodes: int, solver_msg: str) -> dict:
    """Placeholder result when solve_elasticity raises; keeps run_convergence running."""
    return {
        "u_field": np.zeros((n_nodes, 3)),
        "u_magnitude": np.zeros(n_nodes),
        "force_axis": np.zeros(n_nodes),
        "max_displacement": 0.0,
        "rms_displacement": 0.0,
        "mean_displacement_axial": 0.0,
        "strain_energy": 0.0,
        "n_nodes": n_nodes,
        "n_tets": 0,
        "n_dofs": 3 * n_nodes,
        "n_free_dofs": 0,
        "n_fixed_nodes": 0,
        "n_loaded_nodes": 0,
        "nnz": 0,
        "t_assemble": 0.0,
        "t_solve": 0.0,
        "solve_ok": False,
        "solver": solver_msg[:120],
        "solver_info": -1,
        "relative_residual": float("nan"),
        "bc_marking": "n/a",
        "primary_axis": 0,
        "total_load": 0.0,
        "volume_domain": 0.0,
        "h_characteristic": 0.0,
        "rib_length": 0.0,
        "tip_deflection": 0.0,
        "tip_deflection_pct": float("nan"),
        "solver_backend": "n/a",
    }


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def _load_mesh_entry(fp: Path, bc_marking: str) -> dict:
    """Load nodes/elements from .npz; for bc_marking=='marked' require dirichlet_tris/neumann_tris."""
    fp = Path(fp).expanduser().resolve()
    if fp.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported mesh file type: {fp} (expected .npz)")
    data = np.load(fp, allow_pickle=True)
    nodes = np.asarray(data["nodes"])
    elements = np.asarray(data["elements"])
    n_nodes = int(data["n_nodes"]) if "n_nodes" in data.files else int(nodes.shape[0])
    name = fp.stem.replace("_arrays", "").replace("_marked", "")
    dface, nface = None, None
    if (bc_marking or "geom").strip().lower() == "marked":
        if "dirichlet_tris" not in data.files or "neumann_tris" not in data.files:
            raise ValueError(
                f"marked BC requires *_marked.npz with dirichlet_tris and neumann_tris; got {fp.name}"
            )
        dface = _facet_keys_from_tri_arrays(np.asarray(data["dirichlet_tris"]))
        nface = _facet_keys_from_tri_arrays(np.asarray(data["neumann_tris"]))
    return {
        "path": str(fp),
        "name": name,
        "nodes": nodes,
        "elements": elements,
        "n_nodes": n_nodes,
        "dirichlet_face_set": dface,
        "neumann_face_set": nface,
    }


def _pick_calibration_mesh(mesh_data: list, prefer_substring: str) -> dict:
    key = (prefer_substring or "").strip().lower()
    if key:
        for m in mesh_data:
            if key in m["name"].lower():
                return m
    return mesh_data[-1]


def _calibrate_total_load(
    mesh_entry: dict,
    max_ram_gb: Optional[float],
    bc_marking: str,
    petsc_direct: Optional[bool],
    solver_backend: str,
    initial_total_load: float,
    target_tip_defl_pct_min: float,
    target_tip_defl_pct_max: float,
) -> tuple[float, dict]:
    target_mid = 0.5 * (target_tip_defl_pct_min + target_tip_defl_pct_max)
    res0 = solve_elasticity(
        mesh_entry["nodes"],
        mesh_entry["elements"],
        max_ram_gb=max_ram_gb,
        bc_marking=bc_marking,
        total_load=initial_total_load,
        petsc_direct=petsc_direct,
        solver_backend=solver_backend,
        E_field=mesh_entry.get("E_field"),
        dirichlet_face_set=mesh_entry.get("dirichlet_face_set"),
        neumann_face_set=mesh_entry.get("neumann_face_set"),
    )
    d0 = float(res0.get("tip_deflection_pct", 0.0))
    if (not res0.get("solve_ok", False)) or d0 <= 0.0:
        print("  Load calibration skipped (baseline solve failed or zero tip deflection).")
        return initial_total_load, res0
    scale = target_mid / d0
    scale = max(0.05, min(20.0, scale))
    calibrated = float(initial_total_load * scale)
    print(
        f"  Load calibration: baseline tip={d0:.4f}% @ load={initial_total_load:.4f}; "
        f"scaled load={calibrated:.4f} for target midpoint={target_mid:.4f}%",
        flush=True,
    )
    return calibrated, res0


def run_convergence(
    npz_files: list,
    output_dir: str = ".",
    max_ram_gb: float = None,
    bc_marking: str = "geom",
    skip_mesh_substrings: tuple = (),
    petsc_direct: bool = None,
    solver_backend: str = "petsc_auto",
    total_load: float = 100.0,
    auto_calibrate_load: bool = False,
    calibrate_mesh_substring: str = "rib_h3.0",
    target_tip_defl_pct_min: float = 0.1,
    target_tip_defl_pct_max: float = 0.1,
    slice_y_value: Optional[float] = None,
    use_gp_material: bool = False,
    gp_mean_E: float = 70e3,
    gp_std_m: float = 0.4,
    gp_corr_len: float = 80.0,
    gp_grid_resolution: int = 64,
    gp_seed: int = 42,
    gp_e_min: float = DEFAULT_GP_E_MIN,
    gp_e_max: float = DEFAULT_GP_E_MAX,
):
    Path(output_dir).mkdir(exist_ok=True)

    mesh_data = []
    bcm = (bc_marking or "geom").strip().lower()
    for f in npz_files:
        fp = Path(f).expanduser().resolve()
        if bcm == "marked" and fp.suffix.lower() == ".npz" and "_marked" not in fp.name:
            if fp.name.endswith("_arrays.npz"):
                alt = fp.parent / fp.name.replace("_arrays.npz", "_marked.npz")
                if alt.is_file():
                    fp = alt
        try:
            mesh_data.append(_load_mesh_entry(fp, bcm))
        except Exception as e:
            print(f"\n  ── {fp.name} (load) ──\n    ✗ {type(e).__name__}: {e}", flush=True)
            continue
    mesh_data.sort(key=lambda m: m["n_nodes"])
    if skip_mesh_substrings:
        subs = tuple(s.strip().lower() for s in skip_mesh_substrings if s and s.strip())
        if subs:
            kept = []
            for m in mesh_data:
                nl = m["name"].lower()
                if any(sub in nl for sub in subs):
                    print(f"  Skipping mesh (--skip-mesh-substrings): {m['name']}", flush=True)
                    continue
                kept.append(m)
            mesh_data = kept
    if not mesh_data:
        print("No meshes loaded (missing keys, corrupt files, or empty list).", file=sys.stderr)
        sys.exit(1)

    if use_gp_material:
        bbox_min = np.min(np.vstack([m["nodes"] for m in mesh_data]), axis=0)
        bbox_max = np.max(np.vstack([m["nodes"] for m in mesh_data]), axis=0)
        gp = _sample_gp_field(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mean_E=gp_mean_E,
            std_m=gp_std_m,
            corr_len=gp_corr_len,
            grid_resolution=gp_grid_resolution,
            seed=gp_seed,
            e_min=gp_e_min,
            e_max=gp_e_max,
        )
        for m in mesh_data:
            m["E_field"] = _interp_gp_E_to_nodes(gp, m["nodes"])

    print(f"\n{'='*70}")
    print("PHYSICS-BASED MESH CONVERGENCE STUDY")
    print(f"{'='*70}")
    print(f"  Test: linear elasticity, uniform end load, Al (E=70GPa, nu=0.3)")
    print(f"  BC marking: {bc_marking}")
    print(f"  Solver backend: {solver_backend}")
    if use_gp_material:
        print(
            f"  Material: GP E(x)=clip(exp(m)+1,[{gp_e_min:.4g},{gp_e_max:.4g}]), "
            f"target mean_E~{gp_mean_E:.4g}, std(m)={gp_std_m:.4g}, "
            f"corr_len={gp_corr_len:.4g}, seed={gp_seed}"
        )
    print(f"  {len(mesh_data)} meshes: {mesh_data[0]['n_nodes']:,} → "
          f"{mesh_data[-1]['n_nodes']:,} nodes")
    print(f"  Available RAM: ~{AVAILABLE_RAM_GB:.0f} GB "
          f"(limit: {max_ram_gb or AVAILABLE_RAM_GB * 0.7:.0f} GB)\n")

    if auto_calibrate_load:
        cm = _pick_calibration_mesh(mesh_data, calibrate_mesh_substring)
        print(f"  Calibrating load on mesh: {cm['name']}")
        total_load, _ = _calibrate_total_load(
            cm,
            max_ram_gb=max_ram_gb,
            bc_marking=bc_marking,
            petsc_direct=petsc_direct,
            solver_backend=solver_backend,
            initial_total_load=total_load,
            target_tip_defl_pct_min=target_tip_defl_pct_min,
            target_tip_defl_pct_max=target_tip_defl_pct_max,
        )
    print(f"  Total load used in all solves: {total_load:.6g}\n")

    results = []
    for m in mesh_data:
        print(f"\n  ── {m['name']} ({m['n_nodes']:,} nodes) ──")
        try:
            res = solve_elasticity(
                m["nodes"], m["elements"],
                max_ram_gb=max_ram_gb,
                bc_marking=bc_marking,
                total_load=total_load,
                petsc_direct=petsc_direct,
                solver_backend=solver_backend,
                E_field=m.get("E_field"),
                dirichlet_face_set=m.get("dirichlet_face_set"),
                neumann_face_set=m.get("neumann_face_set"),
            )
        except Exception as e:
            print(f"    ✗ {type(e).__name__}: {e}", flush=True)
            res = _failed_elasticity_result(
                m["n_nodes"], f"failed ({type(e).__name__})")
        res["name"] = m["name"]
        results.append(res)
        m["u_field"] = res["u_field"]

        if not res["solve_ok"]:
            sv = res["solver"]
            if not (sv.startswith("skipped") or sv.startswith("failed (")):
                print(f"    ⚠ Solve did not fully converge — results may be approximate")

    for r in results:
        r["l2_error"] = float("nan")
        r["l2_abs_error"] = float("nan")
        r["linf_rel_error"] = float("nan")
        r["energy_error"] = float("nan")
        r["energy_relative"] = float("nan")

    # Filter to meshes that actually solved
    solved = [(m, r) for m, r in zip(mesh_data, results) if r["solve_ok"]]
    if len(solved) < 2:
        print(f"\n  ⚠ Only {len(solved)} mesh(es) solved successfully.")
        print(f"    Need at least 2 for L2/Linf/energy comparison vs finest solved mesh.")
        print(f"    Per-mesh columns in CONVERGENCE RESULTS still list each solve.")
        print(f"    Try --max-ram with a higher value, or use fewer/smaller meshes.")
    else:
        solved_meshes, solved_results = zip(*solved)

        # Compare to finest solved mesh
        finest_m, finest_r = solved_meshes[-1], solved_results[-1]
        print(f"\n  Reference: {finest_m['name']} ({finest_m['n_nodes']:,} nodes)")
        print(f"  Interpolating solutions to reference grid...")

        for m, r in zip(solved_meshes[:-1], solved_results[:-1]):
            u_interp = interpolate_to_reference(m["nodes"], m["u_field"],
                                                finest_m["nodes"])
            u_ref = finest_m["u_field"]
            norm_ref = np.linalg.norm(u_ref)

            diff = u_interp - u_ref
            l2_abs = float(np.linalg.norm(diff))
            r["l2_error"] = float(l2_abs / norm_ref) if norm_ref > 0 else 0.0
            r["l2_abs_error"] = l2_abs
            uinf = float(np.max(np.abs(u_ref)))
            r["linf_rel_error"] = float(np.max(np.abs(diff)) / uinf) if uinf > 0 else 0.0
            r["energy_error"] = float(
                abs(r["strain_energy"] - finest_r["strain_energy"]) /
                abs(finest_r["strain_energy"])
            ) if finest_r["strain_energy"] != 0 else 0.0
            Uf, Uc = finest_r["strain_energy"], r["strain_energy"]
            r["energy_relative"] = float(
                2.0 * abs(np.sqrt(max(Uc, 0.0)) - np.sqrt(max(Uf, 0.0)))
                / (np.sqrt(abs(Uf)) + 1e-30)
            )

        finest_r["l2_error"] = 0.0
        finest_r["l2_abs_error"] = 0.0
        finest_r["linf_rel_error"] = 0.0
        finest_r["energy_error"] = 0.0
        finest_r["energy_relative"] = 0.0

    # Print table (always: includes failed/skipped meshes; L2 columns n/a if no comparison)
    print(f"\n{'='*120}")
    print("CONVERGENCE RESULTS")
    print(f"{'─'*120}")
    print(f"{'Mesh':<14}{'DOFs':>9}{'max|u|':>11}{'rms|u|':>11}{'energy':>12}"
          f"{'tip%L':>8}{'L2 rel':>9}{'Linf rel':>9}{'E rel':>9}{'rel res':>9}"
          f"{'t_tot':>8}{'solver':<22}")
    print(f"{'─'*120}")
    for r in results:
        if not r["solve_ok"]:
            l2s = lfs = ees = "   skip"
        elif not (np.isfinite(r["l2_error"]) and np.isfinite(r["linf_rel_error"])
                  and np.isfinite(r["energy_error"])):
            l2s = lfs = ees = "    n/a"
        else:
            l2s = f"{r['l2_error']:>8.2%}"
            lfs = f"{r['linf_rel_error']:>8.2%}"
            ees = f"{r['energy_error']:>8.2%}"
        if r["solve_ok"] and np.isfinite(r["relative_residual"]):
            rs = f"{r['relative_residual']:.1e}"
        else:
            rs = "n/a"
        tt = r["t_assemble"] + r["t_solve"]
        print(f"{r['name']:<14}{r['n_dofs']:>9,}{r['max_displacement']:>11.4e}"
              f"{r['rms_displacement']:>11.4e}{r['strain_energy']:>12.5e}{r.get('tip_deflection_pct', float('nan')):>7.3f}%"
              f"{l2s}{lfs}{ees}{rs:>9}{tt:>7.1f}s{r['solver']:<22}")
    print(f"{'='*120}")

    # Recommendation (only from solved meshes)
    converged_idx = None
    solved_results_list = [r for r in results if r["solve_ok"]]
    for i, r in enumerate(solved_results_list[:-1]):
        if r["l2_error"] < 0.02:
            converged_idx = i
            break

    print(f"\n  HOW TO READ:")
    print(f"    max_u should stabilize as meshes get finer")
    print(f"    L2 err < 2% means the mesh is converged")
    print(f"    t_slv is the solve time (×N for N training samples)\n")

    if converged_idx is not None:
        rec = solved_results_list[converged_idx]
        print(f"  ► FINE MESH (training data): {rec['name']} "
              f"({rec['n_dofs']:,} DOFs, L2={rec['l2_error']:.2%}, "
              f"{rec['t_solve']:.1f}s/sample)")
        if converged_idx > 0:
            coarse = solved_results_list[converged_idx - 1]
            print(f"  ► COARSE MESH (prototyping):  {coarse['name']} "
                  f"({coarse['n_dofs']:,} DOFs, L2={coarse['l2_error']:.2%}, "
                  f"{coarse['t_solve']:.1f}s/sample)")
        print(f"\n  Training budget (5k samples):")
        print(f"    Fine:   {rec['t_solve'] * 5000 / 3600:.1f} hours")
        if converged_idx > 0:
            print(f"    Coarse: {coarse['t_solve'] * 5000 / 3600:.1f} hours")
    else:
        print(f"  ⚠ No mesh reached <2% L2 error vs finest solved mesh.")
        if len(solved_results_list) >= 2:
            print(f"    Finest solved: {solved_results_list[-1]['name']} "
                  f"({solved_results_list[-1]['n_dofs']:,} DOFs)")
            print(f"    Next coarser L2 error: "
                  f"{solved_results_list[-2]['l2_error']:.2%}")
            print(f"    You may need finer meshes, or the solution is still")
            print(f"    changing significantly between your two finest meshes.")
        elif len(solved_results_list) == 1:
            print(f"    One mesh solved ({solved_results_list[0]['name']}); "
                  f"need a second successful solve for L2 comparison.")
        else:
            print(f"    No successful solves; fix failures above before interpreting convergence.")

    # Save
    out = [{k: v for k, v in r.items() if k not in ("u_field", "u_magnitude", "force_axis")}
           for r in results]
    with open(f"{output_dir}/convergence_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    return results


# ============================================================================
# PLOT
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mesh convergence study")
    p.add_argument(
        "files",
        nargs="*",
        default=[],
        help="Mesh *_arrays.npz files (optional if --meshes-dir or default meshes_3d/)",
    )
    p.add_argument(
        "--meshes-dir",
        type=str,
        default=None,
        help="Directory containing *_arrays.npz; all matches are run, sorted by n_nodes",
    )
    p.add_argument("--output-dir", type=str, default="figures")
    p.add_argument("--max-ram", type=float, default=None,
                   help="Max RAM in GB for solver (default: 70%% of available)")
    p.add_argument(
        "--bc-marking",
        type=str,
        choices=("geom", "surface", "marked"),
        default="geom",
        help="geom: 2%% bbox slab; surface: falls back to geom in FEniCS path; "
        "marked: Dirichlet/Neumann facets from *_marked.npz (dirichlet_tris, neumann_tris).",
    )
    p.add_argument(
        "--skip-mesh-substrings",
        type=str,
        default="",
        help="Comma-separated substrings; meshes whose stem matches any (case-insensitive) are skipped",
    )
    p.add_argument(
        "--petsc-direct",
        action="store_true",
        help="Use PETSc direct LU for the free-DOF system (when the direct path is chosen); "
        "overrides env if set. Requires petsc4py in the active environment.",
    )
    p.add_argument(
        "--solver-backend",
        type=str,
        default="petsc_auto",
        choices=("petsc_auto", "petsc_lu", "petsc_cgilu", "scipy_direct", "scipy_cgilu"),
        help="Solver backend for the assembled linear system.",
    )
    p.add_argument("--total-load", type=float, default=100.0, help="Total Neumann load magnitude.")
    p.add_argument("--auto-calibrate-load", action="store_true",
                   help="Scale total load to hit requested tip-deflection percent band.")
    p.add_argument("--calibrate-mesh-substring", type=str, default="rib_h3.0",
                   help="Prefer this mesh name substring for load calibration.")
    p.add_argument("--target-tip-defl-pct-min", type=float, default=0.1,
                   help="Lower target of tip deflection in percent of rib length.")
    p.add_argument("--target-tip-defl-pct-max", type=float, default=0.1,
                   help="Upper target of tip deflection in percent of rib length.")
    p.add_argument("--slice-y-value", type=float, default=None,
                   help="Fixed y value for XZ slice plot. Default: median y on finest solved mesh.")
    p.add_argument("--use-gp-material", action="store_true",
                   help="Spatial E(x): sample m, then E=clip(exp(m)+1,[gp-e-min,gp-e-max]) (see _sample_gp_field).")
    p.add_argument("--gp-mean-E", type=float, default=70e3,
                   help="Target mean_E for GP (sets shift of m before exp; same numeric scale as solve_elasticity E).")
    p.add_argument("--gp-std-m", type=float, default=0.4,
                   help="Std-dev of zero-mean GP field m(x) before exponentiation.")
    p.add_argument("--gp-corr-len", type=float, default=80.0,
                   help="GP correlation length in geometry units.")
    p.add_argument("--gp-grid-resolution", type=int, default=64,
                   help="Regular-grid resolution along longest bbox axis for GP sampling.")
    p.add_argument("--gp-seed", type=int, default=42,
                   help="Random seed for GP material field.")
    p.add_argument("--gp-e-min", type=float, default=DEFAULT_GP_E_MIN,
                   help="Lower clip on sampled E(x) (same units as Young modulus E in this script).")
    p.add_argument("--gp-e-max", type=float, default=DEFAULT_GP_E_MAX,
                   help="Upper clip on sampled E(x).")

    args = p.parse_args()
    if args.meshes_dir:
        md = Path(args.meshes_dir).expanduser().resolve()
        if args.bc_marking == "marked":
            paths = sorted(md.glob("*_marked.npz"))
        else:
            paths = sorted(md.glob("*_arrays.npz"))
    elif args.files:
        paths = [Path(f).expanduser().resolve() for f in args.files]
    else:
        md = Path(__file__).resolve().parent / "meshes_3d"
        paths = sorted(md.glob("*_arrays.npz")) if md.is_dir() else []

    if not paths:
        hint = "*_marked.npz" if args.bc_marking == "marked" else "*_arrays.npz"
        print(f"No mesh files: pass paths, or --meshes-dir (expected {hint}).", file=sys.stderr)
        sys.exit(1)
    skip_subs = tuple(s for s in args.skip_mesh_substrings.split(",") if s.strip())
    petsc_direct = None
    if args.petsc_direct:
        petsc_direct = True
    run_convergence(
        [str(x) for x in paths],
        args.output_dir,
        max_ram_gb=args.max_ram,
        bc_marking=args.bc_marking,
        skip_mesh_substrings=skip_subs,
        petsc_direct=petsc_direct,
        solver_backend=args.solver_backend,
        total_load=args.total_load,
        auto_calibrate_load=args.auto_calibrate_load,
        calibrate_mesh_substring=args.calibrate_mesh_substring,
        target_tip_defl_pct_min=args.target_tip_defl_pct_min,
        target_tip_defl_pct_max=args.target_tip_defl_pct_max,
        slice_y_value=args.slice_y_value,
        use_gp_material=args.use_gp_material,
        gp_mean_E=args.gp_mean_E,
        gp_std_m=args.gp_std_m,
        gp_corr_len=args.gp_corr_len,
        gp_grid_resolution=args.gp_grid_resolution,
        gp_seed=args.gp_seed,
        gp_e_min=args.gp_e_min,
        gp_e_max=args.gp_e_max,
    )
