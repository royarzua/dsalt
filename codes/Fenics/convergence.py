# Copyright (C) 2023-2025 by the Dsalt team
#
# This file is part of Dsalt project.


"""
This Python script corresponds to the first numerical example in the paper

A finite element model for concentration polarization and osmotic effects in a membrane channel
from Nicolás Carro, David Mora and Jesus Vellojin 
DOI: https://doi.org/10.1002/fld.5252

@author: Jesus Vellojin
@license: MIT
@date: 2024-01-24
"""

from dolfin import *
import numpy as np

parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"
parameters["refinement_algorithm"] = 'plaza_with_parent_facets'
set_log_level(LogLevel.ERROR)


def test(nk):
    # Malla solo rectangulo
    nps = pow(2, nk)+1
    mesh = UnitSquareMesh(nps, nps,)
    
    bdry = MeshFunction('size_t', mesh, 1, 0)
    # bdryf = MeshFunction('size_t', mesh_fine, 1, 0)

    boundaries = {inlet:    CompiledSubDomain('near(x[0], 0)'),
                  wall:     CompiledSubDomain('near(x[1], 1)'),
                  membrane: CompiledSubDomain('near(x[1], 0)'),
                  outlet:   CompiledSubDomain('near(x[0], 1)')
                  }

    [subdomain.mark(bdry, tag) for tag, subdomain in boundaries.items()]

    ds = Measure('ds', domain=mesh, subdomain_data=bdry)

    # Define function spaces
    # Choose mini for Mini-Element of TH for Taylor-Hood
    set_element = 'mini'
    if set_element == 'mini':
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        B = FiniteElement("B", mesh.ufl_cell(), mesh.topology().dim() + 1)
        Vh = VectorElement(NodalEnrichedElement(P1, B))
        element = MixedElement([Vh, P1, P1])
    elif set_element == 'TH':
        Vh = VectorElement("CG", mesh.ufl_cell(), 2)
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        P2 = FiniteElement("CG", mesh.ufl_cell(), 2)
        element = MixedElement([Vh, P1, P2])
    else:
        raise KeyError('Wrong choice of elements. Please use mini or TH')

    # Create function spaces
    W = FunctionSpace(mesh, element)
    ndof = W.dim()
    # Test and trial functions
    u,  p, sigma = TrialFunctions(W)
    v,  q, tau = TestFunctions(W)

    # ********* Create exact solutions ******* #

    u_ex = Expression(('cos(pi*x[0])*sin(pi*x[1])',
                      '-cos(pi*x[1])*sin(pi*x[0])'), domain=mesh, degree=4)

    u_ext_m = Expression('cos(pi*x[0])*sin(pi*x[1])', domain=mesh, degree=4)

    p_ex = Expression('sin(pow(x[0],2) + pow(x[1],2))', domain=mesh, degree=4)

    theta_ex = Expression('exp(-x[0]*x[1])', domain=mesh, degree=4)

    sig_ex = grad(u_ex) - p_ex*Identity(2)
    tflux_ex = u_ex*theta_ex - grad(theta_ex)
    RhsNS = -div(grad(u_ex)) + grad(u_ex)*u_ex + grad(p_ex)
    rhsDF = -div(grad(theta_ex)) + dot(u_ex, grad(theta_ex))

    T0 = interpolate(Constant(1.), W.sub(2).collapse())

    UP = Function(W)

    uold, _, _ = split(UP)
    sigmaold = UP.sub(2) # This goes to RHS so we use sub()
    sigmaold.assign(T0) # This helps the algorithm

    # Define bcs
    membrane_ut = DirichletBC(W.sub(0).sub(0), u_ext_m, bdry, membrane)
    wall_u = DirichletBC(W.sub(0), u_ex, bdry, wall)
    inlet_u = DirichletBC(W.sub(0), u_ex, bdry, inlet)
    inlet_C = DirichletBC(W.sub(2), theta_ex, bdry, inlet)

    bcu = [wall_u, inlet_u, membrane_ut, inlet_C]
    n = FacetNormal(mesh)

    # Define membrane conditions for LHS and RHS
    conditionLHS = dot(u, n)
    conditionRHS = (dot(u_ex, n) + theta_ex - sigmaold)

    he = FacetArea(mesh)
    # Mixed formulation for NS
    K = inner(grad(u), grad(v))*dx  # a(u,v)
    K += - inner(p, div(v))*dx  # b(v,p)
    K += - inner(q, div(u))*dx  # b(u,q)
    K += inner(grad(u)*uold, v)*dx  # tilde_a(u,(u,v))
    K += - inner(grad(u)*n - p*n, n*dot(v, n))*ds(membrane)

    # Add Nitsche terms for LHS
    K += - inner(grad(v)*n - q*n, n * conditionLHS)*ds(membrane)
    K += alpha / he*inner(conditionLHS, dot(v, n))*ds(membrane)

    RHS = inner(dot(sig_ex, n), v)*ds(outlet)
    RHS += inner(RhsNS, v)*dx

    # Add Nitsche terms for RHS
    RHS += - inner(grad(v)*n - q*n, n * conditionRHS)*ds(membrane)
    RHS += alpha / he*inner(conditionRHS, dot(v, n))*ds(membrane)

    # Define  variational problem for concentration-velocity
    K += inner(grad(sigma), grad(tau))*dx  # d(C,tau)
    K += inner(dot(grad(sigma), uold), tau)*dx  # tilde_c(tau,(u,C))
    K += -inner(sigma*(dot(uold, n)), tau)*ds((wall, membrane))

    # Compute fluxes for RHS
    fluxbetaex = grad(theta_ex)
    flux = -inner(dot(tflux_ex, n), tau)*ds((membrane, wall))
    flux += inner(dot(fluxbetaex, n), tau)*ds((outlet))

    RHS += inner(rhsDF, tau)*dx
    RHS += flux

    tol = 1E-5
    maxiter = 200
    eps = 1.0
    k = 0

    # Create updating new functions
    UPnew = Function(W)
    niter = 0
    for k in range(maxiter):
        niter += 1
        # Solve linearized Navier Stokes
        solve(K == RHS, UPnew, bcu, solver_parameters={'linear_solver': 'mumps',
                                                       'preconditioner': 'lu'})

        # Compute distance before update
        localUPnew = UPnew.vector().get_local()
        localUP = UP.vector().get_local()
        diffUP = localUPnew - localUP

        # compute relative l2 error
        diffe = diffUP
        coeff = localUPnew

        eps = np.linalg.norm(diffe, ord=2) / \
            np.linalg.norm(coeff, ord=2)

        UP.assign(UPnew)   # update for solve Ts equation

        print('Picard iteration %2d | eps %1.3e ' % (k, eps))

        if eps < tol and k > 0:
            if k == maxiter - 1:
                print('Max number iterations reached...')
            if eps < tol:
                print('Algorithm converged with eps...', eps)
            break

    exactas = (u_ex, p_ex, theta_ex)
    spaces = W
    return ndof, mesh.hmax(), niter, UPnew, exactas, spaces, n, he


if __name__ == '__main__':

    # Nitsche parameter
    alpha = 10

    # mesh marks
    inlet = 2
    wall = 3
    membrane = 4
    outlet = 5

    nkmax = 6

    hh = []
    hht = []
    nn = []
    eth = []
    rth = []
    rene = []
    eu = []
    ru = []
    e_ener = []
    r_ener = []
    ep = []
    rp = []
    it = []

    rene.append(0.)
    rth.append(0.)
    r_ener.append(0.0)
    ru.append(0.0)
    rp.append(0.)
    # Use reffs is different meshing is needed
    # reffs = [15, 25, 35, 45, 55, 65]
    for nk in range(nkmax):
        print("....... Refinement level : nk = ", nk)
        ndof, h, niter, UP, exactas, spaces, normal, he = test(nk)

        W = spaces
        u, p, theta = UP.split(True)
        u_ex, p_ex, theta_ex = exactas
        hh.append(h)
        nn.append(ndof)
        it.append(niter)
        eu.append(errornorm(u_ex, u, 'H1'))
        ep.append(errornorm(p_ex, p, 'L2'))
        eth.append(errornorm(theta_ex, theta, 'H1'))

        ener1 = grad(u_ex-u)
        ener2 = -inner(grad(u_ex - u)*normal - (p_ex-p)*normal,
                       normal*dot(u_ex-u, normal))
        ener3 = dot(u_ex-u, normal)
        err_ener = sqrt(assemble(ener1**2*dx + ener2*ds(membrane) +
                        alpha / he*ener3**2*ds(membrane)))
        e_ener.append(err_ener)

        if (nk > 0):
            ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
            rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
            rth.append(ln(eth[nk]/eth[nk-1])/ln(hh[nk]/hh[nk-1]))
            rene.append(ln(e_ener[nk]/e_ener[nk-1])/ln(hh[nk]/hh[nk-1]))

    print('===========================================================================================')
    print('  DoFs     h       e(u)     r(u)   e(eu)     r(eu)   e(p)     r(p)    e(thet)  r(thet)  it')
    print('===========================================================================================')
    for nk in range(nkmax):
        print('{:6d}   {:.3f}   {:1.2e}   {:.2f}  {:1.2e}   {:.2f}  {:1.2e}   {:.2f}   {:1.2e}   {:.2f}   {:2d} \\'.format(
            nn[nk], hh[nk], eu[nk], ru[nk], e_ener[nk], rene[nk], ep[nk], rp[nk], eth[nk], rth[nk], it[nk]))
    print('===========================================================================================')
