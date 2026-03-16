"""
fenics_mesh_transfer_demo_no_dolfin.py

Identical demo to fenics_mesh_transfer_demo.ipynb but using only
numpy / scipy — no dolfin / FEniCS required.

Demonstrates the same 4 mesh-to-mesh transfer methods:
  A. Algebraic transfer / prolongation matrix
  B. Target-space L² projection (sparse solve)
  C. Same L² projection assembled manually as a dense matrix solve
  D. Oversampled least-squares / pseudoinverse fit
  + weighted variant of D

The mesh is a uniform triangular tessellation of [0,1]^2 using P1
(continuous piecewise-linear) finite elements, exactly matching the
UnitSquareMesh + CG1 setup from the FEniCS notebook.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")          # headless backend — remove if running interactively
import matplotlib.pyplot as plt

# ============================================================
# P1 FEM utilities on uniform triangular meshes
# ============================================================

def make_mesh(n):
    """
    UnitSquareMesh(n, n) equivalent: uniform triangulation of [0,1]^2.

    Node (i, j) -> index i*(n+1)+j,  coordinates (i/n, j/n).

    Each unit square cell (i,j) is split by its '\\' diagonal
    (from bottom-left to top-right) into two triangles:
      Tri0: (i,j), (i+1,j), (i+1,j+1)    <- lower-right half
      Tri1: (i,j), (i+1,j+1), (i,j+1)    <- upper-left half

    Returns
    -------
    nodes  : (n+1)^2 x 2   float64   node coordinates
    tris   : 2*n^2   x 3   int64     triangle connectivity
    n      : int            subdivision count (kept for O(1) lookup)
    """
    def nid(i, j):
        return i * (n + 1) + j

    nodes = np.array([[i / n, j / n]
                      for i in range(n + 1)
                      for j in range(n + 1)], dtype=float)

    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([nid(i, j),   nid(i+1, j),   nid(i+1, j+1)])  # Tri0
            tris.append([nid(i, j),   nid(i+1, j+1), nid(i,   j+1)])  # Tri1

    return nodes, np.array(tris, dtype=int), n


def _tri_area(v):
    """Signed area of triangle with rows v[0],v[1],v[2]."""
    return 0.5 * ((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1]))


def _tri_grads(v):
    """
    Gradients of the three P1 basis functions on triangle v (shape 3x2).

    Standard formula: ∇λ_k = ( (y_{k+1}-y_{k+2})/D,  (x_{k+2}-x_{k+1})/D )
    where D = 2 * signed_area.
    """
    D = 2.0 * _tri_area(v)
    grads = np.zeros((3, 2))
    for k in range(3):
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        grads[k, 0] = (v[k1, 1] - v[k2, 1]) / D
        grads[k, 1] = (v[k2, 0] - v[k1, 0]) / D
    return grads


def _find_tri(n_mesh, x, y):
    """
    O(1) triangle index for point (x, y) on a uniform n_mesh x n_mesh mesh.

    Uses the '\\' diagonal: in cell (i,j) the split is fx >= fy -> Tri0.
    """
    xi = x * n_mesh
    yi = y * n_mesh
    i  = int(min(xi, n_mesh - 1))
    j  = int(min(yi, n_mesh - 1))
    fx = xi - i
    fy = yi - j
    base = 2 * (i * n_mesh + j)
    return base if (fx >= fy) else base + 1


def _bary(v, x, y):
    """
    Barycentric coordinates of (x, y) w.r.t. triangle v (shape 3x2).
    Returns (λ0, λ1, λ2).
    """
    T  = np.array([[v[1, 0] - v[0, 0], v[2, 0] - v[0, 0]],
                   [v[1, 1] - v[0, 1], v[2, 1] - v[0, 1]]])
    lm = np.linalg.solve(T, np.array([x - v[0, 0], y - v[0, 1]]))
    return np.array([1.0 - lm[0] - lm[1], lm[0], lm[1]])


def _clamp(val, lo=1e-10, hi=None):
    if hi is None:
        hi = 1.0 - lo
    return min(max(val, lo), hi)


def eval_p1(nodes, tris, n_mesh, coeffs, x, y):
    """Evaluate P1 FEM function at a single point (x, y) — O(1) lookup."""
    x = _clamp(x); y = _clamp(y)
    tid = _find_tri(n_mesh, x, y)
    tri = tris[tid]
    lm  = _bary(nodes[tri], x, y)
    return float(lm @ coeffs[tri])


def eval_p1_batch(nodes, tris, n_mesh, coeffs, pts):
    """Evaluate P1 FEM function at many points (shape Nx2)."""
    out = np.empty(len(pts))
    for q, (x, y) in enumerate(pts):
        out[q] = eval_p1(nodes, tris, n_mesh, coeffs, x, y)
    return out


# ============================================================
# FEM assembly
# ============================================================

def assemble_mass(nodes, tris):
    """
    Consistent P1 mass matrix:  M_ij = ∫_Ω φ_i φ_j dx

    For a triangle with area A the local matrix is
        M_local = (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
    """
    N = len(nodes)
    rows, cols, vals = [], [], []
    for tri in tris:
        v = nodes[tri]
        A = abs(_tri_area(v))
        for p in range(3):
            for q in range(3):
                rows.append(tri[p])
                cols.append(tri[q])
                vals.append(A / 12.0 * (2.0 if p == q else 1.0))
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def assemble_rhs(nodes_t, tris_t,
                 nodes_s, tris_s, n_s, coeffs_s):
    """
    RHS for L² projection:  b_i = ∫_Ω u_s(x) φ_i^(t)(x) dx

    Uses 3-point edge-midpoint quadrature (exact for degree-2 integrands):
      quadrature points: m01=(v0+v1)/2, m12=(v1+v2)/2, m20=(v2+v0)/2
      weight: A/3 per point

    Basis-function values at edge midpoints:
      φ_0(m01)=½, φ_0(m12)=0, φ_0(m20)=½  →  b_0^e = (A/6)*(u_s(m01)+u_s(m20))
      φ_1(m01)=½, φ_1(m12)=½, φ_1(m20)=0  →  b_1^e = (A/6)*(u_s(m01)+u_s(m12))
      φ_2(m01)=0, φ_2(m12)=½, φ_2(m20)=½  →  b_2^e = (A/6)*(u_s(m12)+u_s(m20))
    """
    b = np.zeros(len(nodes_t))

    def us(pt):
        return eval_p1(nodes_s, tris_s, n_s, coeffs_s, pt[0], pt[1])

    for tri in tris_t:
        v  = nodes_t[tri]
        A  = abs(_tri_area(v))
        m01 = 0.5 * (v[0] + v[1])
        m12 = 0.5 * (v[1] + v[2])
        m20 = 0.5 * (v[2] + v[0])
        u01 = us(m01); u12 = us(m12); u20 = us(m20)
        b[tri[0]] += (A / 6.0) * (u01 + u20)
        b[tri[1]] += (A / 6.0) * (u01 + u12)
        b[tri[2]] += (A / 6.0) * (u12 + u20)

    return b


def create_transfer_matrix(nodes_s, tris_s, n_s, nodes_t):
    """
    Prolongation matrix P (shape: n_t × n_s).

    P[i, j] = barycentric weight of source node j for target node i,
    found by locating target node i in the source mesh.

    For P1 elements this is exact: u_t = P @ u_s recovers u_s exactly
    when u_s is piecewise-linear and the target nodes lie inside the
    source domain.
    """
    nt, ns = len(nodes_t), len(nodes_s)
    rows, cols, vals = [], [], []
    for i, (x, y) in enumerate(nodes_t):
        x = _clamp(x); y = _clamp(y)
        tid  = _find_tri(n_s, x, y)
        tri  = tris_s[tid]
        lm   = _bary(nodes_s[tri], x, y)
        for k in range(3):
            rows.append(i)
            cols.append(tri[k])
            vals.append(float(lm[k]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nt, ns))


# ============================================================
# Error norms
# ============================================================

def _l2_norm_sq(nodes, tris, c):
    """∫ u² dx via 1-point centroid quadrature (exact for P1²)."""
    total = 0.0
    for tri in tris:
        v = nodes[tri]
        A = abs(_tri_area(v))
        # centroid value = mean of vertex values for P1
        u_c = c[tri].mean()
        total += A * u_c ** 2
    return total


def _h1_sq(nodes, tris, c):
    """∫ |∇u|² dx (exact for P1 — gradient is piecewise constant)."""
    total = 0.0
    for tri in tris:
        v = nodes[tri]
        A = abs(_tri_area(v))
        g = _tri_grads(v)          # (3, 2)
        grad_u = g.T @ c[tri]     # (2,)
        total += A * np.dot(grad_u, grad_u)
    return total


def rel_l2_error(nodes, tris, c_num, c_ref):
    diff = c_num - c_ref
    return np.sqrt(_l2_norm_sq(nodes, tris, diff) /
                   _l2_norm_sq(nodes, tris, c_ref))


def rel_h1_error(nodes, tris, c_num, c_ref):
    diff = c_num - c_ref
    denom = _h1_sq(nodes, tris, c_ref)
    if denom == 0:
        return float("nan")
    return np.sqrt(_h1_sq(nodes, tris, diff) / denom)


def report(name, nodes_t, tris_t, c_num, c_ref):
    print(f"{name:35s}  rel L2 = {rel_l2_error(nodes_t, tris_t, c_num, c_ref):.6e}"
          f"   rel H1 = {rel_h1_error(nodes_t, tris_t, c_num, c_ref):.6e}")


# ============================================================
# Exact test function  (same as notebook)
# ============================================================

def u_exact(x, y):
    """sin(2π x)cos(3π y) + 0.2 cos(5π xy) + x(1-y)"""
    return (  np.sin(2 * np.pi * x) * np.cos(3 * np.pi * y)
            + 0.2 * np.cos(5 * np.pi * x * y)
            + x * (1.0 - y))


# ============================================================
# Section 1 — Meshes and interpolation of exact function
# ============================================================

print("=" * 65)
print("Section 1: Meshes")
print("=" * 65)

nodes_s, tris_s, n_s = make_mesh(8)    # coarse  (9×9  = 81 nodes)
nodes_t, tris_t, n_t = make_mesh(24)   # fine   (25×25 = 625 nodes)

print(f"dim(V_s) = {len(nodes_s)}")
print(f"dim(V_t) = {len(nodes_t)}")

# P1 interpolation of u_exact on each mesh (= interpolate(u_exact_expr, V))
c_s       = u_exact(nodes_s[:, 0], nodes_s[:, 1])   # source coefficients
c_t_exact = u_exact(nodes_t[:, 0], nodes_t[:, 1])   # reference ("exact") target

# ============================================================
# Section 3 — Method A: algebraic transfer (prolongation)
# ============================================================

print("\n" + "=" * 65)
print("Section 3: Method A — algebraic transfer")
print("=" * 65)

P_st = create_transfer_matrix(nodes_s, tris_s, n_s, nodes_t)
c_t_alg = P_st @ c_s

report("algebraic transfer", nodes_t, tris_t, c_t_alg, c_t_exact)
print(f"P_st shape = {P_st.shape[0]} x {P_st.shape[1]}")

# Round-trip test
P_ts         = create_transfer_matrix(nodes_t, tris_t, n_t, nodes_s)
c_s_roundtrip = P_ts @ c_t_alg
rt_err = np.linalg.norm(c_s_roundtrip - c_s) / np.linalg.norm(c_s)
print(f"\nround-trip coarse-space rel L2 error = {rt_err:.6e}")

# ============================================================
# Section 4 — Method B: L² projection (sparse solve)
# ============================================================

print("\n" + "=" * 65)
print("Section 4: Method B — target L² projection (sparse solve)")
print("=" * 65)

M_t = assemble_mass(nodes_t, tris_t)
b_t = assemble_rhs(nodes_t, tris_t, nodes_s, tris_s, n_s, c_s)

c_t_l2 = spla.spsolve(M_t, b_t)

report("target L2 projection", nodes_t, tris_t, c_t_l2, c_t_exact)

# ============================================================
# Section 5 — Method C: same projection, dense scipy solve
# ============================================================

print("\n" + "=" * 65)
print("Section 5: Method C — manual mass-matrix solve (dense)")
print("=" * 65)

c_t_manual = la.solve(M_t.toarray(), b_t, assume_a="sym")

report("manual mass-matrix solve", nodes_t, tris_t, c_t_manual, c_t_exact)
print(f"difference between B and C: {np.linalg.norm(c_t_l2 - c_t_manual):.6e}")

# ============================================================
# Section 6 — Method D: oversampled pseudoinverse
# ============================================================

print("\n" + "=" * 65)
print("Section 6: Method D — oversampled pseudoinverse fit")
print("=" * 65)

nx_samp = 41
xx = np.linspace(0.0, 1.0, nx_samp)
yy = np.linspace(0.0, 1.0, nx_samp)
pts = np.array([[x, y] for y in yy for x in xx])

m_pts   = len(pts)
n_dofs  = len(nodes_t)
print(f"number of sample points = {m_pts}")
print(f"number of target dofs   = {n_dofs}")

# Build sampling matrix S_t[q, i] = φ_i^(t)(x_q)  (sparse → dense)
rows_S, cols_S, vals_S = [], [], []
for q, (x, y) in enumerate(pts):
    x = _clamp(x); y = _clamp(y)
    tid = _find_tri(n_t, x, y)
    tri = tris_t[tid]
    lm  = _bary(nodes_t[tri], x, y)
    for k in range(3):
        rows_S.append(q); cols_S.append(tri[k]); vals_S.append(float(lm[k]))

S_t = sp.csr_matrix((vals_S, (rows_S, cols_S)),
                    shape=(m_pts, n_dofs)).toarray()

# Evaluate u_s at sample points
y_samp = eval_p1_batch(nodes_s, tris_s, n_s, c_s, pts)

c_t_pinv = la.pinv(S_t) @ y_samp

report("oversampled pinv fit", nodes_t, tris_t, c_t_pinv, c_t_exact)
print(f"least-squares residual ||S_t c - y||_2 = "
      f"{np.linalg.norm(S_t @ c_t_pinv - y_samp):.6e}")
print(f"distance to L2-proj coefficients       = "
      f"{np.linalg.norm(c_t_pinv - c_t_manual):.6e}")

# ============================================================
# Section 7 — Weighted least-squares variant
# ============================================================

print("\n" + "=" * 65)
print("Section 7: Weighted least-squares")
print("=" * 65)

dx_s = 1.0 / (nx_samp - 1)
dy_s = 1.0 / (nx_samp - 1)
w    = np.full(m_pts, dx_s * dy_s)
Wsqrt = np.sqrt(w)

S_w = S_t * Wsqrt[:, None]
y_w = y_samp * Wsqrt

c_t_weighted = la.lstsq(S_w, y_w)[0]

report("weighted least-squares fit", nodes_t, tris_t, c_t_weighted, c_t_exact)
print(f"distance to L2-proj coefficients = "
      f"{np.linalg.norm(c_t_weighted - c_t_manual):.6e}")

# ============================================================
# Section 8 — Summary table
# ============================================================

print("\n" + "=" * 65)
print("Section 8: Summary table")
print("=" * 65)

methods = [
    ("algebraic transfer",       c_t_alg),
    ("target L2 projection",     c_t_l2),
    ("manual mass-matrix solve", c_t_manual),
    ("oversampled pinv fit",     c_t_pinv),
    ("weighted least-squares",   c_t_weighted),
]

print(f"\n{'method':35s} {'rel L2':>14s} {'rel H1':>14s}")
print("-" * 65)
for name, c in methods:
    print(f"{name:35s} "
          f"{rel_l2_error(nodes_t, tris_t, c, c_t_exact):14.6e} "
          f"{rel_h1_error(nodes_t, tris_t, c, c_t_exact):14.6e}")

# ============================================================
# Section 9 — Plots
# ============================================================

print("\nGenerating plots...")

import os
out_dir = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(2, 3, figsize=(14, 10))

def tri_plot(ax, nodes, tris, coeffs, title):
    tc = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris, coeffs,
                      shading="gouraud", cmap="viridis")
    plt.colorbar(tc, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

tri_plot(axes[0, 0], nodes_t, tris_t, c_t_exact,   "exact target reference")
tri_plot(axes[0, 1], nodes_t, tris_t, c_t_alg,     "algebraic transfer")
tri_plot(axes[0, 2], nodes_t, tris_t, c_t_l2,      "target L² projection")
tri_plot(axes[1, 0], nodes_t, tris_t, c_t_manual,  "manual mass solve")
tri_plot(axes[1, 1], nodes_t, tris_t, c_t_pinv,    "oversampled pinv fit")
tri_plot(axes[1, 2], nodes_t, tris_t, c_t_weighted,"weighted LS fit")

plt.suptitle("Mesh-to-mesh transfer comparison (no dolfin)", fontsize=13)
plt.tight_layout()
out_path = os.path.join(out_dir, "transfer_comparison_no_dolfin.png")
plt.savefig(out_path, dpi=120)
print(f"Figure saved to {out_path}")
