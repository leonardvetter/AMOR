import matplotlib
matplotlib.use('Agg')
from dolfin import *
import numpy as np
import scipy.linalg as la

parameters['allow_extrapolation'] = True

mesh_s = UnitSquareMesh(8, 8)
mesh_t = UnitSquareMesh(24, 24)
V_s = FunctionSpace(mesh_s, 'CG', 1)
V_t = FunctionSpace(mesh_t, 'CG', 1)

u_exact_expr = Expression(
    'sin(2*pi*x[0])*cos(3*pi*x[1]) + 0.2*cos(5*pi*x[0]*x[1]) + x[0]*(1.0 - x[1])',
    degree=6, pi=np.pi)

u_s       = interpolate(u_exact_expr, V_s)
u_t_exact = interpolate(u_exact_expr, V_t)
print('dim(V_s) =', V_s.dim())
print('dim(V_t) =', V_t.dim())

def rel_l2(u_num, u_ref):
    return errornorm(u_ref, u_num, norm_type='L2') / norm(u_ref, 'L2')
def rel_h1(u_num, u_ref):
    return errornorm(u_ref, u_num, norm_type='H1') / norm(u_ref, 'H1')
def report(name, u_num):
    print(f'{name:35s}  rel L2 = {rel_l2(u_num, u_t_exact):.6e}   rel H1 = {rel_h1(u_num, u_t_exact):.6e}')

# Method A
# Note: on WSL2, PETSc's MatCreateVecs fails with MPI error 98 when using
# u_t_alg.vector()[:] = P_st * u_s.vector() (PETSc mat-vec multiply).
# Workaround: extract P_st as a dense numpy matrix and use numpy multiply.
P_st = PETScDMCollection.create_transfer_matrix(V_s, V_t)
P_st_np = P_st.array()  # extract as dense numpy array
u_t_alg = Function(V_t)
u_t_alg.vector().set_local(P_st_np @ u_s.vector().get_local())
u_t_alg.vector().apply('insert')
report('algebraic transfer', u_t_alg)
print('P_st shape =', P_st.size(0), 'x', P_st.size(1))

P_ts    = PETScDMCollection.create_transfer_matrix(V_t, V_s)
P_ts_np = P_ts.array()
u_s_rt  = Function(V_s)
u_s_rt.vector().set_local(P_ts_np @ u_t_alg.vector().get_local())
u_s_rt.vector().apply('insert')
print('round-trip rel L2 =', rel_l2(u_s_rt, interpolate(u_exact_expr, V_s)))

# Method B
class SrcExpr(UserExpression):
    def __init__(self, u_src, **kwargs):
        super().__init__(**kwargs); self.u_src = u_src
    def eval(self, vals, x): vals[0] = self.u_src(x)
    def value_shape(self): return ()

u_s_expr = SrcExpr(u_s, degree=6)
u_t_l2   = project(u_s_expr, V_t)
report('target L2 projection', u_t_l2)

# Method C
phi = TrialFunction(V_t); v = TestFunction(V_t)
M_t = assemble(phi * v * dx).array()
b_t = assemble(u_s_expr * v * dx).get_local()
c_t_manual = la.solve(M_t, b_t, assume_a='sym')
u_t_manual = Function(V_t)
u_t_manual.vector().set_local(c_t_manual); u_t_manual.vector().apply('insert')
report('manual mass-matrix solve', u_t_manual)
print('diff B vs C:', np.linalg.norm(u_t_l2.vector().get_local() - c_t_manual))

# Method D
nx_samp = 41
pts = np.array([[x,y] for y in np.linspace(0,1,nx_samp) for x in np.linspace(0,1,nx_samp)])
n_t = V_t.dim()
S_t = np.zeros((len(pts), n_t))
for j in range(n_t):
    e_j = Function(V_t); vec = np.zeros(n_t); vec[j] = 1.0
    e_j.vector().set_local(vec); e_j.vector().apply('insert')
    S_t[:, j] = np.array([e_j(Point(x,y)) for x,y in pts])
y_vals = np.array([u_s(Point(x,y)) for x,y in pts])
c_t_pinv = la.pinv(S_t) @ y_vals
u_t_pinv = Function(V_t)
u_t_pinv.vector().set_local(c_t_pinv); u_t_pinv.vector().apply('insert')
report('oversampled pinv fit', u_t_pinv)

h = 1.0/(nx_samp-1); Wsqrt = np.full(len(pts), h)
c_t_w = la.lstsq(S_t * Wsqrt[:,None], y_vals * Wsqrt)[0]
u_t_w = Function(V_t)
u_t_w.vector().set_local(c_t_w); u_t_w.vector().apply('insert')
report('weighted least-squares', u_t_w)

methods = [('algebraic transfer',u_t_alg),('target L2 projection',u_t_l2),
           ('manual mass-matrix solve',u_t_manual),('oversampled pinv fit',u_t_pinv),
           ('weighted least-squares',u_t_w)]
print(f"\n{'method':35s} {'rel L2':>14s} {'rel H1':>14s}")
print('-'*65)
for name, uf in methods:
    print(f'{name:35s} {rel_l2(uf,u_t_exact):14.6e} {rel_h1(uf,u_t_exact):14.6e}')
print('\nAll sections OK.')
