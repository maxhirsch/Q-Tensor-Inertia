import numpy as np
import cProfile
from tqdm import tqdm

from initialization import get_n0, get_Q0, get_dn0, get_D2Q0
from algebraic_system_solver import solve_Q_flow_linear_system
from PQ import P
from basis_functions import *

import sys

def initialize_Q_flow(interior_point_coords, t_final, Nt, a, b, c, A0, sigma, L, power1, power2, PERTURB1, PERTURB2):
    """
    returns: Q1, Q2, p1, p2, r
    """

    num_interior_points = interior_point_coords.shape[0]

    dt = t_final / Nt

    Q1 = np.zeros((Nt, num_interior_points))
    Q2 = np.zeros((Nt, num_interior_points))

    p1 = np.zeros((Nt, num_interior_points))
    p2 = np.zeros((Nt, num_interior_points))

    r  = np.zeros((Nt, num_interior_points))

    # do not actually need to track s for the scheme
    s1 = np.zeros((Nt, num_interior_points))
    s2 = np.zeros((Nt, num_interior_points))

    index = 0
    for i, point in enumerate(interior_point_coords):

        x, y = point[0], point[1]

        if not np.isclose(sigma, 0.0, atol=1e-10, rtol=1e-10):
            perturbation = np.eye(2)
            perturbation[1, 1] = -1
            n0_ij = get_n0(x, y)
            Q0_ij = get_Q0(n0_ij) + PERTURB1 * (sigma**power1 * perturbation / 2)
        else:
            n0_ij = get_n0(x, y)
            Q0_ij = get_Q0(n0_ij)

        Q0sq_ij = Q0_ij @ Q0_ij
        r[0, index] = np.sqrt(2 * ((a/2)*np.trace(Q0sq_ij) \
            - (b/3)*np.trace(Q0sq_ij @ Q0_ij) + (c/4)*np.trace(Q0sq_ij)**2 + A0))
        Q1[0, index] = Q0_ij[0, 0] # store the top left entry of Q0_ij
        Q2[0, index] = Q0_ij[0, 1] # store the top right entry of Q0_ij

        # make the initial P^{n+1/2} value P(Q0)
        P0_ij, SQ_ij = P(Q0_ij, a, b, c, A0)
        p1[0, index] = P0_ij[0, 0] # store the top left entry of P0_ij
        p2[0, index] = P0_ij[0, 1] # store the top right entry of P0_ij

        s1[0, index] = SQ_ij[0, 0]
        s2[0, index] = SQ_ij[0, 1]


        # now do the same for time = 1
        D2Q0_ij = get_D2Q0(x, y, sigma)
        if not np.isclose(sigma, 0.0, atol=1e-10, rtol=1e-10):
            perturbation = np.eye(2)
            perturbation[1, 1] = -1
            Q1_ij = Q0_ij + dt * (L * D2Q0_ij - r[0, index] * P0_ij + PERTURB2 * (sigma**power2 * perturbation / 2))
        else:
            Q1_ij = Q0_ij + dt * (L * D2Q0_ij - r[0, index] * P0_ij)
        
        Q1sq_ij = Q1_ij @ Q1_ij
        r[1, index] = np.sqrt(2 * ((a/2)*np.trace(Q1sq_ij) \
            - (b/3)*np.trace(Q0sq_ij @ Q0_ij) + (c/4)*np.trace(Q1sq_ij)**2 + A0))
        Q1[1, index] = Q1_ij[0, 0]
        Q2[1, index] = Q1_ij[0, 1]
        
        P1_ij, SQ_ij = P(Q1_ij, a, b, c, A0)
        p1[1, index] = P1_ij[0, 0]
        p2[1, index] = P1_ij[0, 1]
        
        s1[1, index] = SQ_ij[0, 0]
        s2[1, index] = SQ_ij[0, 1]
        
        index += 1
    
    return Q1, Q2, p1, p2, r, s1, s2, num_interior_points

def update_P(time, Q1, Q2, p1, p2, s1, s2, a, b, c, A0):
    Q = np.concatenate((
            np.concatenate((Q1[time, :, None, None], Q2[time, :, None, None]), axis=2),
            np.concatenate((Q2[time, :, None, None], -Q1[time, :, None, None]), axis=2),
    ), axis=1)
    PQ, SQ = P(Q, a, b, c, A0)
    p1[time] = PQ[:, 0, 0]
    p2[time] = PQ[:, 0, 1]

    s1[time] = SQ[:, 0, 0]
    s2[time] = SQ[:, 0, 1]

def solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements=5, power1=1.0, power2=1.0, PERTURB1=0.0, PERTURB2=0.0):
    """
    solve_Q_flow: 
    
    returns: (xVals, yVals, tVals, Q1, Q2) where
        xVals (np.array) with xVals.shape == (Nx,)
        yVals (np.array) with yVals.shape == (Nx,)
        tVals (np.array) with tVals.shape == (Nt,)
        Q1 (np.array) with Q1.shape == (Nt, Nx**2)
        Q2 (np.array) with Q2.shape == (Nt, Nx**2)
    """

    # define the mesh in space and time
    #points = get_points(Nx+2)
    mesh, interior_point_coords = get_triangulation(refinements)
    tVals = np.linspace(0, t_final, Nt)
    dt = tVals[1] - tVals[0] # delta t

    # get vectors for the components of matrices (flattened over space) 
    # over time with time 0 initialized
    # Note that we only keep track of nodes in the interior of \Omega
    Q1, Q2, p1, p2, r, s1, s2, num_interior_points = initialize_Q_flow(interior_point_coords, t_final, Nt, a, b, c, A0, sigma, L, power1, power2, PERTURB1, PERTURB2)
    
    stiffness_matrix, gamma = calculate_basis_integrals(mesh)

    print("Computing scheme...")
    
    for time in tqdm(range(2, Nt)):
        # update Q1, Q2
        Q1_, Q2_ = solve_Q_flow_linear_system(time, Q1, Q2, p1, p2, r, gamma, stiffness_matrix, M, sigma, L, dt)
        
        Q1[time] = Q1_
        Q2[time] = Q2_

        # update r
        r[time] = r[time-1] + 2 * p1[time-1] * (Q1[time] - Q1[time-1]) + 2 * p2[time-1] * (Q2[time] - Q2[time-1])

        # update P(Q)
        update_P(time, Q1, Q2, p1, p2, s1, s2, a, b, c, A0)
    

    return (mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1, Q2, r)
