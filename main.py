import ufl
import numpy as np
import dolfinx as dfx
from dolfinx import fem
from mpi4py.MPI import COMM_WORLD as comm

"""parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8"""

# ----------------------------------------------------------------------------------------------------------------------
# SHORTHANDS
# ----------------------------------------------------------------------------------------------------------------------

Xc = [.5, .5]
R = .2

# Mesh
mesh = dfx.mesh.create_unit_square(comm, 100, 100, dfx.mesh.CellType.triangle)

# Time stepping parameters
dt = .01
t_end = 10.
theta=fem.Constant(mesh,.5) # theta schema
g=fem.Constant(mesh,(0.,-1.)) # Gravity !

# Fluid properties
rho1,rho2=1e3,1e2
mu1, mu2 =10,1

# Numerical properties
eps=1e-6 # Precision
N=2000 # Iteration nb
rp=.95 # Relaxation parameter

# Define function spaces
FE_vector =ufl.VectorElement("CG",mesh.ufl_cell(),2)
FE_scalar =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
# Taylor Hodd elements ; stable element pair + level-set
FS = fem.FunctionSpace(mesh,ufl.MixedElement(FE_vector,FE_scalar,FE_scalar))
D = fem.FunctionSpace(mesh,FE_scalar)

# Define unknown and test function(s)
w, w0 = fem.Function(FS), fem.Function(FS)

# UFL objects
v_,p_,l_=ufl.TestFunctions(FS)
v, p, l =ufl.split(w)
v0,p0,l0=ufl.split(w0)

# Integrand
dx = ufl.dx(degree=4)

def Sign(q): return ufl.conditional(ufl.lt(abs(q),eps),q/eps,ufl.sign(q))
def rho(l): return .5*(rho1 * (1+Sign(l)) + rho2 * (1-Sign(l)))
def nu(l):  return .5*(mu1  * (1+Sign(l)) + mu2  * (1-Sign(l)))

n = ufl.FacetNormal(mesh)
I = ufl.Identity(2)    # Identity tensor
h = ufl.avg(ufl.Circumradius(mesh))

# ----------------------------------------------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ----------------------------------------------------------------------------------------------------------------------

FS0,FS2=FS.sub(0),FS.sub(2)
FS0c,_=FS0.collapse()
FS2c,_=FS2.collapse()
# Level Set BCs
dofs_wall = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: (np.isclose(x[0],0)+np.isclose(x[0],1)+np.isclose(x[1],0)+np.isclose(x[1],1))>=1)
wall_f = fem.Function(FS2c)
wall_f.interpolate(lambda x: np.ones_like(x[0]))
bcs=[fem.dirichletbc(wall_f,dofs_wall,FS2)]
# Wall velocity BCs
dofs = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: (np.isclose(x[0],0)*(x[1]<.25)*(.75<x[1])+np.isclose(x[0],1)+(np.isclose(x[1],0)+np.isclose(x[1],1))*(x[0]<.75))>=1)
cst = fem.Function(FS0c)
cst.interpolate(lambda x: np.zeros_like(x[:2]))
bcs.append(fem.dirichletbc(cst,dofs,FS0))
# TBR
X_i=np.linspace(.25,.75,50)
X_o=np.linspace(.75,1,25)
U_i=(X_i-.25)*(.75-X_i)
U_o=(X_o-.75)*(1-X_o)
D_i=np.trapz(U_i,X_i)
D_o=np.trapz(U_o,X_o)
def inflow(x):
    u=np.zeros_like(x[:2])
    u[0]=(x[1]-.25)*(.75-x[1])/D_i
    return u
def outflow_top(x):
    u=np.zeros_like(x[:2])
    u[1]=(x[0]-.75)*(1-x[1])/D_o/2
    return u
def outflow_bottom(x):
    u=np.zeros_like(x[:2])
    u[1]=-(x[0]-.75)*(1-x[1])/D_o/2
    return u
dofs_inflow = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: np.isclose(x[0],0)*(x[1]>=.25)*(.75>=x[1])>=1)
dofs_outflow_top = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: np.isclose(x[1],1)*(x[0]>=.75)>=1)
dofs_outflow_bottom = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: np.isclose(x[1],0)*(x[0]>=.75)>=1)
inflow_f,outflow_top_f,outflow_bottom_f = fem.Function(FS0c), fem.Function(FS0c), fem.Function(FS0c)
inflow_f.interpolate(inflow); outflow_top_f.interpolate(outflow_top); outflow_bottom_f.interpolate(outflow_bottom)
bcs.extend([fem.dirichletbc(inflow_f,dofs_inflow,FS0),
            fem.dirichletbc(outflow_top_f,dofs_outflow_top,FS0),
            fem.dirichletbc(outflow_bottom_f,dofs_outflow_bottom,FS0)])

# ----------------------------------------------------------------------------------------------------------------------
# INITIAL CONDITIONS
# ----------------------------------------------------------------------------------------------------------------------

v,p,l = w.split()
def IC(x):
    u=np.zeros_like(x[:2])
    msk1 = (x[1]>.25)*(x[1]<.75)*(x[0]<.75)
    msk2 = (x[0]>.75)*(x[1]> .5)
    msk3 = (x[0]>.75)*(x[1]<=.5)
    u[0,msk1]= (x[1,msk1]-.25)*(.75-x[1,msk1])/D_i
    u[1,msk2]= (x[0,msk2]-.75)*(1  -x[1,msk2])/D_o/2
    u[1,msk3]=-(x[0,msk3]-.75)*(1  -x[1,msk3])/D_o/2
    return u
v.interpolate(IC)
p.interpolate(lambda x: np.zeros_like(x)[0])
l.interpolate(lambda x: np.ones_like(x)[0])
#l.interpolate(lambda x: (x[0]-Xc[0])**2 + (x[1]-Xc[1])**2 - R**2)

# ----------------------------------------------------------------------------------------------------------------------
# VARIATIONAL FORM
# ----------------------------------------------------------------------------------------------------------------------

# Main form (theta scheme)
F=.5*(theta*rho(l)+(1.-theta)*rho(l0))*ufl.inner(v-v0,v_) + ufl.inner(l-l0,l_) # March in time impetus and LS
F/=dt

# Navier Stokes
def EQ(v,p,l,v_,_,l_):
    F  = ufl.inner(ufl.grad(l),l_*v) # Convection of l
    F += ufl.inner(-p*I + nu(l)*(ufl.grad(v)+ufl.grad(v).T),ufl.grad(v_)) # Strain
    F += rho(l)*ufl.inner(ufl.grad(v)*v, v_) # Advection
    #F -= rho(l)*ufl.inner(g,v_) # Gravity
    return F

F+=theta*EQ(v,p,l,v_,p_,l_) + (1.-theta)*EQ(v0,p0,l0,v_,p_,l_) + ufl.div(v)*p_ # Enforce physics (including compressibility)
F*=dx # Volume integral
#F+=.1*h**2*ufl.inner(ufl.jump(ufl.grad(l),n), ufl.jump(ufl.grad(l_),n))*ufl.dS # Surface tension

# Evolve LS
gd_l=ufl.grad(l)
n=ufl.sqrt(ufl.inner(gd_l,gd_l))
def S(l,eps): return l/ufl.sqrt(l**2+eps**2)
G=(ufl.inner(v,v_)+ufl.inner(p,p_)+ufl.inner((l-l0)/dt-S(l0,eps)*(1-n),l_))*dx

# ----------------------------------------------------------------------------------------------------------------------
# SOLVERS
# ----------------------------------------------------------------------------------------------------------------------

problem=fem.petsc.NonlinearProblem(F,w,bcs,jit_options={"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]})
reinit =fem.petsc.NonlinearProblem(G,w,bcs,jit_options={"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]})
solver_pb=dfx.nls.petsc.NewtonSolver(comm,problem)
solver_reinit=dfx.nls.petsc.NewtonSolver(comm,reinit)
for solver in (solver_pb,solver_reinit):
    solver.atol, solver.rtol = eps, eps
    solver.max_iter = N
    solver.relaxation_parameter = rp

# ----------------------------------------------------------------------------------------------------------------------
# TIME-MARCH
# ----------------------------------------------------------------------------------------------------------------------

# Create files for storing solution
for name,fun in zip(["velocity", "pressure", "levelset"],[v,p,l]):
    with dfx.io.XDMFFile(comm, "dat/"+name+"_t=0.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(fun)

# Time-stepping
t,c = dt,0
while t < t_end:
    V=fem.assemble_scalar(fem.form(ufl.conditional(ufl.lt(l,0.),1.,0.)*dx))
    print("volume=",V)
    print("t =", t)
    print("Solving transport...")
    solver_pb.solve(w)
    print("Reinitialising level-set...")
    #solver_reinit.solve(w)
    v,p,l=w.split()
    if c%10==0:
        for name,fun in zip(["velocity", "pressure", "levelset"],[v,p,l]):
            with dfx.io.XDMFFile(comm, "dat/"+name+f"_t={t:.2f}".replace('.',',')+".xdmf", "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(fun,t)
    w0.interpolate(w)
    # Move to next time step
    t += dt
    c += 1