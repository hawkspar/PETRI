import ufl
import numpy as np
import dolfinx as dfx
from dolfinx import fem
from mpi4py.MPI import COMM_WORLD as comm

# get file name
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8

Xc = [.5, .5]
r = .2

# Mesh
mesh = dfx.mesh.create_rectangle(comm, 100, 100, dfx.mesh.CellType.quadrilateral)

# Time stepping parameters
dt = .01
t_end = 10.
theta=fem.Constant(mesh,.5)   # theta schema
k=fem.Constant(mesh,1.0/dt)
g=fem.Constant(mesh,(0.,-1.))

X=ufl.SpatialCoordinate(mesh)

# Distance function
dist = fem.Expression(ufl.sqrt((X[0]-Xc[0])**2 + (X[1]-Xc[1])**2)-r)

class InitialCondition(Expression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = ufl.sqrt((X[0]-Xc[0])**2 + (X[1]-Xc[1])**2)-r
    def value_shape(self):
        return (4,)

ic=InitialCondition()

# Define function spaces
FE_vector =ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
# Taylor Hodd elements ; stable element pair + eddy viscosity
FS = fem.FunctionSpace(mesh,FE_vector*FE_scalar*FE_scalar)
# Define unknown and test function(s)
w = fem.Function(FS)
w0 = fem.Function(FS)



(v_, p_, l_) = ufl.TestFunctions(FS)

(v,p,l)=ufl.split(w)
(v0,p0,l0)=ufl.split(w0)

bcs = []
FS0=FS.sub(0)
FS0c,_=FS0.collapse()
# Degrees of freedom
dofs = dfx.fem.locate_dofs_geometrical((FS0, FS0c), lambda x: np.minimum(np.isclose(x[0],0)+np.isclose(x[0],1)+np.isclose(x[1],0),1))
bcs.append(fem.dirichletbc(fem.Constant(mesh,0),dofs,FS.sub(0)))

rho1=1e3
rho2=1e2
mu1=1e1
mu2=1e0
eps=1e-4

def Sign(q): return ufl.conditional(ufl.lt(abs(q),eps),q/eps,ufl.sign(q))

def Delta(q): return ufl.conditional(ufl.lt(abs(q),eps),1./eps*.5*(1.+ufl.cos(3.14159*q/eps)),fem.Constant(mesh,0.))

def rho(l): return(rho1 * .5* (1.+ Sign(l)) + rho2 * .5*(1. - Sign(l)))

def nu(l): return(mu1 * .5* (1.+ Sign(l)) + mu2 * .5*(1. - Sign(l)))

def EQ(v,p,l,v_,p_,l_):
    F_ls = ufl.inner(ufl.div(l*v),l_) 
    T= -p*I + nu(l)*(ufl.grad(v)+ufl.grad(v).T)
    F_ns = ufl.inner(T,ufl.grad(v_)) + rho(l)*ufl.inner(ufl.grad(v)*v, v_) - rho(l)*ufl.inner(g,v_)
    F=F_ls+F_ns
    return F*ufl.dx(quadrature_degree=8)

n = ufl.FacetNormal(mesh)
I = ufl.Identity(V.cell().geometric_dimension())    # Identity tensor
h = ufl.CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
alpha=ufl.Constant(mesh,.1)
r = alpha('+')*h_avg*h_avg*ufl.inner(ufl.jump(ufl.grad(l),n), ufl.jump(grad(l_),n))*dS

F=k*0.5*(theta*rho(l)+(1.0-theta)*rho(l0))*inner(v-v0,v_)*dx + k*inner(l-l0,l_)*dx + theta*EQ(v,p,l,v_,p_,l_) + (1.0-theta)*EQ(v0,p,l0,v_,p_,l_) + div(v)*p_*dx + r

J = derivative(F, w)
ffc_options = {"quadrature_degree": 4, "optimize": True, "eliminate_zeros": False}
problem=NonlinearVariationalProblem(F,w,bcs,J,ffc_options)
solver=NonlinearVariationalSolver(problem)

prm = solver.parameters
#info(prm, True)
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'umfpack'
prm['newton_solver']['lu_solver']['report'] = False
prm['newton_solver']['lu_solver']['same_nonzero_pattern']=True
prm['newton_solver']['absolute_tolerance'] = 1E-10
prm['newton_solver']['relative_tolerance'] = 1E-10
prm['newton_solver']['maximum_iterations'] = 20
prm['newton_solver']['report'] = True
#prm['newton_solver']['error_on_nonconvergence'] = False


w.assign(interpolate(ic,W))
w0.assign(interpolate(ic,W))

(v,p,l) = w.split()
(v0,p0,l0) = w0.split()


def reinit(l,mesh):
    #implement here:
    #   given mesh and function l on the mesh
    # reinitialize function l such that |grad(l)|=1
    # and the zero levelset does not change (too much)

    

    return l


#assign(l, interpolate (dist,L))
#assign(l0, interpolate (dist,L))

#plot(l0,interactive=True)
#plot(rho(l),interactive=True)
# Create files for storing solution
vfile = File("%s.results/velocity.pvd" % (fileName))
pfile = File("%s.results/pressure.pvd" % (fileName))
lfile = File("%s.results/levelset.pvd" % (fileName))

v.rename("v", "velocity") ; vfile << v
p.rename("p", "pressure") ; pfile << p
l.rename("l", "levelset") ; lfile << l

# Time-stepping
t = dt
while t < t_end:

   print "t =", t

   begin("Solving transport...")
   solver.solve()
   end()

   (v,p,l)=w.split(True)
   v.rename("v", "velocity") ; vfile << v
   p.rename("p", "pressure") ; pfile << p
   l.rename("l", "levelset") ; lfile << l

   V=assemble(conditional(lt(l,0.0),1.0,0.0)*dx)
   print "volume= %e"%V

   plot(v,interactive=True)
   # Move to next time step
   l1=reinit(l,mesh)
   #assign(w.sub(2),interpolate(l1,L))

   w0.assign(w)
   t += dt  # t:=t+1