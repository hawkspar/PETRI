# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl, re
import numpy as np
import dolfinx as dfx
from dolfinx import fem
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Numerical Parameters
params = {"rp":.95,    #relaxation_parameter
		  "atol":1e-12, #absolute_tolerance
		  "rtol":1e-9, #DOLFIN_EPS does not work well
		  "max_iter":100}
Re=100
dt=1e-4

mesh = dfx.mesh.create_unit_square(comm, 100, 100, dfx.mesh.CellType.quadrilateral)

r = ufl.SpatialCoordinate(mesh)[1]
# Finite elements & function spaces
FE_vector =ufl.VectorElement("CG",mesh.ufl_cell(),2,3)
FE_scalar =ufl.FiniteElement("CG",mesh.ufl_cell(),1)
FE_scalar2=ufl.FiniteElement("DG",mesh.ufl_cell(),0)
FS0 = fem.FunctionSpace(mesh,FE_vector)
FS1 = fem.FunctionSpace(mesh,FE_scalar)
FS2 = fem.FunctionSpace(mesh,FE_scalar2)
# Taylor Hodd elements ; stable element pair + eddy viscosity
FS = fem.FunctionSpace(mesh,FE_vector*FE_scalar*FE_scalar2)
# Trial & test functions
Q  = fem.Function(FS)
Qn = ufl.TrialFunction(FS)
T  = ufl.TestFunction( FS)
# Trivial BC
mesh.topology.create_connectivity(1, 2)
boundary_facets = mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = fem.locate_dofs_topological(FS, 1, boundary_facets)
bc = fem.dirichletbc(fem.Constant(mesh,0), boundary_dofs)

# Cylindrical operators
def grd(v):
	if len(v.ufl_shape)==0: return ufl.as_vector([v.dx(0), v.dx(1), 0])
	return ufl.as_tensor([[v[0].dx(0), v[0].dx(1),  0],
						  [v[1].dx(0), v[1].dx(1), -v[2]/r],
						  [v[2].dx(0), v[2].dx(1),  v[1]/r]])

def div(v):
	if len(v.ufl_shape)==1: return v[0].dx(0) + (r*v[1]).dx(1)/r
	return ufl.as_vector([v[0,0].dx(0)+(r*v[1,0]).dx(1),
						  v[0,1].dx(0)+(r*v[1,1]).dx(1)-v[2,2]/r,
						  v[0,2].dx(0)+(r*v[1,2]).dx(1)+v[2,1]/r])

def dirCreator(path:str):
	if not os.path.isdir(path):
		if p0: os.mkdir(path)
	comm.barrier() # Wait for all other processors

def checkComm(f:str):
	match = re.search(r'n=(\d*)',f)
	if int(match.group(1))!=comm.size: return False
	match = re.search(r'p=([0-9]*)',f)
	if int(match.group(1))!=comm.rank: return False
	return True
	
# Krylov subspace
def configureKSP(KSP:pet.KSP,params:dict,icntl:bool=False) -> None:
	KSP.setTolerances(rtol=params['rtol'], atol=params['atol'], max_it=params['max_iter'])
	# Krylov subspace
	KSP.setType('preonly')
	# Preconditioner
	PC = KSP.getPC(); PC.setType('lu')
	PC.setFactorSolverType('mumps')
	KSP.setFromOptions()
	if icntl: PC.getFactorMatrix().setMumpsIcntl(14,500)

# Naive save with dir creation
def saveStuff(dir:str,name:str,fun:fem.Function) -> None:
	dirCreator(dir)
	proc_name=dir+name.replace('.',',')+f"_n={comm.size:d}_p={comm.rank:d}"
	fun.x.scatter_forward()
	np.save(proc_name,fun.x.array)
	if p0: print("Saved "+proc_name+".npy",flush=True)

# Memoisation routine - find closest in param
def findStuff(path:str,params:dict,format=lambda f:True,distributed=True):
	closest_file_name=path
	file_names = [f for f in os.listdir(path) if format(f)]
	d=np.infty
	for file_name in file_names:
		if not distributed or checkComm(file_name): # Lazy evaluation !
			fd=0 # Compute distance according to all params
			for param in params:
				match = re.search(param+r'=(\d*(,|e|-|j|\+)?\d*)',file_name)
				param_file = float(match.group(1).replace(',','.')) # Take advantage of file format
				fd += abs(params[param]-param_file)
			if fd<d: d,closest_file_name=fd,path+file_name
	return closest_file_name

def loadStuff(path:str,params:dict,fun:fem.Function) -> None:
	closest_file_name=findStuff(path,params,lambda f: f[-3:]=="npy")
	fun.x.array[:]=np.load(closest_file_name,allow_pickle=True)
	fun.x.scatter_forward()
	# Loading eddy viscosity too
	if p0: print("Loaded "+closest_file_name,flush=True)

# Shorthands
U,  P,  F  = ufl.split(Q)
Un, Pn, Fn = ufl.split(Qn)
v,  s,  f  = ufl.split(T)

R=1/Re*div(grd(U))-grd(U)*U

F =ufl.inner(div(grd(Pn))-div(U+dt*R)/dt,s)*ufl.conditional(ufl.le(F,1),0,1)
F+=ufl.inner(Un-U-dt*R+dt*grd(Pn),v)

def marchP() -> ufl.Form:
	# Functions
	U,  P,  F  = ufl.split(Q)
	Un, Pn, Fn = ufl.split(Qn)
	nu = 1/Re
	_, s, _ = ufl.split(T)
	F  = ufl.inner(div(grd(Pn)),  s)
	F  = ufl.inner(div(grd(Pn)),  s)
	# Momentum (different test functions and IBP)
	F += ufl.inner(Un-U,	 v)/dt # Time march
	F += ufl.inner(grd(U)*U, v) # Convection
	F -= ufl.inner(	  Pn,div(v)) # Pressure
	F += ufl.inner(grd(U)+grd(U).T,
						 grd(v))*nu # Diffusion (grad u.T significant with nut)
	F += ufl.inner(Fn-F,	 f)/dt # Time march
	F += ufl.inner(div(F*U), f) # Convection
	return F*r*ufl.dx

# Heart of this entire code
def navierStokes() -> ufl.Form:
	# Functions
	U,  P,  F  = ufl.split(Q)
	Un, Pn, Fn = ufl.split(Qn)
	nu = 1/Re
	v, s, f = ufl.split(T)
	# Mass (variational formulation)
	F  = ufl.inner(div(Un),  s)
	# Momentum (different test functions and IBP)
	F += ufl.inner(Un-U,	 v)/dt # Time march
	F += ufl.inner(grd(U)*U, v) # Convection
	F -= ufl.inner(	  Pn,div(v)) # Pressure
	F += ufl.inner(grd(U)+grd(U).T,
						 grd(v))*nu # Diffusion (grad u.T significant with nut)
	F += ufl.inner(Fn-F,	 f)/dt # Time march
	F += ufl.inner(grd(U)*F, f) # Convection
	return F*r*ufl.dx

def printStuff(name:str,fun:fem.Function) -> None:
	with dfx.io.XDMFFile(comm, name.replace('.',',')+".xdmf", "w") as xdmf:
		xdmf.write_mesh(mesh)
		xdmf.write_function(fun)
	if p0: print("Printed "+name.replace('.',',')+".xdmf",flush=True)