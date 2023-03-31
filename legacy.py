import ufl
import numpy as np
import dolfinx as dfx
from dolfinx import fem
from mpi4py.MPI import COMM_WORLD as comm

"""parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8"""

Xc = [.5, .5]
R = .2

# Mesh
mesh = dfx.mesh.create_unit_square(comm, 100, 100, dfx.mesh.CellType.triangle)

# Time stepping parameters
dt = .01
t_end = 10.
theta=fem.Constant(mesh,.5)   # theta schema
k=fem.Constant(mesh,1/dt)
g=fem.Constant(mesh,(0.,-1.))

# Define function spaces
FE_vector =ufl.VectorElement("CG",mesh.ufl_cell(),2)
FE_scalar =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
# Taylor Hodd elements ; stable element pair + level-set
FS = fem.FunctionSpace(mesh,ufl.MixedElement(FE_vector,FE_scalar,FE_scalar))
D = fem.FunctionSpace(mesh,FE_scalar)

# Define unknown and test function(s)
w, w0 = fem.Function(FS), fem.Function(FS)

v_,p_,l_=ufl.TestFunctions(FS)
v, p, l =ufl.split(w)
v0,p0,l0=ufl.split(w0)

bcs = []
FS0=FS.sub(0)
FS0c,_=FS0.collapse()
# Degrees of freedom
dofs = fem.locate_dofs_geometrical((FS0, FS0c), lambda x: (np.isclose(x[0],0)+np.isclose(x[0],1)+np.isclose(x[1],0))>=1)
cst = fem.Function(FS0c)
cst.interpolate(lambda x: np.zeros_like(x[:2]))
bcs.append(fem.dirichletbc(cst,dofs,FS0))

rho1,rho2=1e3,1e2
mu1, mu2 =10,1
eps=1e-6

def Sign(q):  return ufl.conditional(ufl.lt(abs(q),eps),q/eps,ufl.sign(q))

def Delta(q): return ufl.conditional(ufl.lt(abs(q),eps),.5/eps*(1.+ufl.cos(np.pi*q/eps)),fem.Constant(mesh,0.))

def rho(l): return .5*(rho1 * (1+Sign(l)) + rho2 * (1-Sign(l)))

def nu(l):  return .5*(mu1  * (1+Sign(l)) + mu2  * (1-Sign(l)))

def EQ(v,p,l,v_,_,l_):
    F_ls = ufl.inner(ufl.div(l*v),l_) 
    T = -p*I + nu(l)*(ufl.grad(v)+ufl.grad(v).T)
    F_ns = ufl.inner(T,ufl.grad(v_)) + rho(l)*ufl.inner(ufl.grad(v)*v, v_) - rho(l)*ufl.inner(g,v_)
    return (F_ls+F_ns)*ufl.dx

n = ufl.FacetNormal(mesh)
I = ufl.Identity(2)    # Identity tensor
h = ufl.avg(ufl.Circumradius(mesh))
r = .1*h**2*ufl.inner(ufl.jump(ufl.grad(l),n), ufl.jump(ufl.grad(l_),n))*ufl.dS

F=k*0.5*(theta*rho(l)+(1.0-theta)*rho(l0))*ufl.inner(v-v0,v_)*ufl.dx + k*ufl.inner(l-l0,l_)*ufl.dx + theta*EQ(v,p,l,v_,p_,l_) + (1.0-theta)*EQ(v0,p,l0,v_,p_,l_) + ufl.div(v)*p_*ufl.dx + r

J = ufl.derivative(F, w)
#ffc_options = {"quadrature_degree": 4, "optimize": True, "eliminate_zeros": False}
problem=fem.petsc.NonlinearProblem(F,w,bcs,J)#,ffc_options)
solver=dfx.nls.petsc.NewtonSolver(comm,problem)
solver.atol, solver.rtol = 1e-10, 1e-10
solver.max_iter = 20

v,p,l = w.split()
v.interpolate(lambda x: np.zeros_like(x)[:2])
p.interpolate(lambda x: np.zeros_like(x)[0])
l.interpolate(lambda x: (x[0]-Xc[0])**2 + (x[1]-Xc[1])**2 - R**2)

def reinit(l,mesh):
    #implement here:
    #   given mesh and function l on the mesh
    # reinitialize function l such that |grad(l)|=1
    # and the zero levelset does not change (too much)
    return l

# Create files for storing solution
for name,fun in zip(["velocity", "pressure", "levelset"],[v,p,l]):
    with dfx.io.XDMFFile(comm, name+"_t=0.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(fun)

# Time-stepping
t = dt
c=0
while t < t_end:
   V=fem.assemble_scalar(fem.form(ufl.conditional(ufl.lt(l,0.),1.,0.)*ufl.dx))
   print("volume=",V)
   print("t =", t)
   print("Solving transport...")
   solver.solve(w)

   v,p,l=w.split()
   
   if c%10==0:
    for name,fun in zip(["velocity", "pressure", "levelset"],[v,p,l]):
            with dfx.io.XDMFFile(comm, name+f"_t={t:.2f}".replace('.',',')+".xdmf", "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(fun,t)

   # Move to next time step
   l1=reinit(l,mesh)

   w0.interpolate(w)
   t += dt
   c += 1