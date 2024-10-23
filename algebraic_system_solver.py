import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import kron
from scipy.sparse import linalg
from scipy.sparse import bmat
from scipy.sparse import eye
import cProfile

def is_Q_tensor(Q):
    """
    is_Q_tensor
    -----------
    Requires: Q is a numeric np.array with shape (2,2)
    Ensures: Returns False if Q is not trace-free, symmetric, or
        has eigenvalues outside of the expected range of 
        Q tensor eigenvalues and True otherwise
    """
    if abs(Q[0, 0] + Q[1, 1]) > 10e-6:
        print("\nNot trace-free")
        return False
    if abs(Q[0, 1] - Q[1, 0]) > 10e-6:
        print("\nNot symmetric")
        return False
    evals, evecs = np.linalg.eig(Q)
    if evals[0] < -1/3 or evals[0] > 2/3:
        print("\n|Eigenvalue| too large:", evals[0], "\nQ =", Q)
        return False
    if evals[1] < -1/3 or evals[1] > 2/3:
        print("\n|Eigenvalue| too large:", evals[1], "\nQ =", Q)
        return False
    return True

def all_are_Q_tensors(Q1, Q2):
    for i in range(Q1.shape[0]):
        Q = np.array([
            [Q1[i], Q2[i]],
            [Q2[i], -Q1[i]]
        ])
        if not is_Q_tensor(Q):
            return False
    return True

def solve_Q_flow_linear_system(time, Q1, Q2, p1, p2, r, gamma, stiffness_matrix, M, sigma, L, dt):
    """
    returns: Q1[time], Q2[time]
    """
    Q1_ = Q1[time-1]
    Q2_ = Q2[time-1]
    Q1_2 = Q1[time-2]
    Q2_2 = Q2[time-2]
    p1_ = p1[time-1]
    p2_ = p2[time-1]
    r_ = r[time-1]

    num_interior_points = gamma.shape[0]

    gamma_over_dt = 1/dt * spdiags([gamma], [0], num_interior_points, num_interior_points)
    gamma_sigma_over_dt2 = sigma / dt**2 * spdiags([gamma], [0], num_interior_points, num_interior_points)

    M_diag_p1sq_gamma = spdiags([M*p1_*p1_*gamma], [0], num_interior_points, num_interior_points)
    M_diag_p2sq_gamma = spdiags([M*p2_*p2_*gamma], [0], num_interior_points, num_interior_points)
    M_p1p2 = M*p1_*p2_
    M_diag_p1p2_gamma = spdiags([M_p1p2*gamma], [0], num_interior_points, num_interior_points)

    # create matrix for system Aq = g
    T1 = gamma_over_dt + gamma_sigma_over_dt2 + M*L/2 * stiffness_matrix + M_diag_p1sq_gamma
    T2 = gamma_over_dt + gamma_sigma_over_dt2 + M*L/2 * stiffness_matrix + M_diag_p2sq_gamma

    A = bmat([
        [T1, M_diag_p1p2_gamma],
        [M_diag_p1p2_gamma, T2]
    ], format = 'csr')

    # create vector g for system Aq = g
    F1 = gamma_over_dt + 2*gamma_sigma_over_dt2 - M*L/2 * stiffness_matrix + M_diag_p1sq_gamma
    F2 = gamma_over_dt + 2*gamma_sigma_over_dt2 - M*L/2 * stiffness_matrix + M_diag_p2sq_gamma

    g1 = F1 @ Q1_.reshape(-1,1) + (gamma * M_p1p2 * Q2_).reshape(-1,1) - M * (gamma * p1_ * r_).reshape(-1, 1) - sigma/dt**2 * (gamma * Q1_2).reshape(-1,1)
    g2 = F2 @ Q2_.reshape(-1,1) + (gamma * M_p1p2 * Q1_).reshape(-1,1) - M * (gamma * p2_ * r_).reshape(-1, 1) - sigma/dt**2 * (gamma * Q2_2).reshape(-1,1)

    g = np.concatenate((g1, g2), axis=0)
    Q_ = linalg.spsolve(A, g).reshape(-1)

    return Q_[:num_interior_points], Q_[num_interior_points:]
