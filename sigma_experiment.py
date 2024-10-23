import time
import cProfile

# imports from local files
from energy import plot_energy
from main_solver import solve_Q_flow

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path

Path("./Paper Figures/").mkdir(parents=True, exist_ok=True)
Path("./Paper Data/").mkdir(parents=True, exist_ok=True)

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# Omega = [0, Lx] x [0, Lx]
Lx = 2 # we currently assume it's 2, so changing this does nothing.

a = -0.2
b = 1
c = 1
A0 = 500

M = 1

L = 0.001
Nt = 10000
t_final = 0.1

refinements = 4
print("Mesh size:", f"dt = {t_final/Nt}", f"h = {2/2**refinements}")

# first solve the Q-tensor flow for sigma = 0
sigma = 0.0
mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1_0, Q2_0, r_0 = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements, 0.0, 0.0, 0.0, 0.0)


# power1, power2, PERTURB1, PERTURB2
for si, setting in enumerate([(0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.0), 
                              (0.0, 0.5, 0.0, 1.0), (0.5, 0.5, 1.0, 1.0), (1.0, 0.5, 1.0, 1.0)]):
    
    power1, power2, PERTURB1, PERTURB2 = setting

    errors = []
    
    # solve Q-tensor flow for various sigma values and compute error
    sigmas = np.array([0.00001, 0.000025, 0.00005, 
        0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 
        0.05, 0.075, 0.1])
    
    for sigma in sigmas:  
        # compute solution
        mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1, Q2, r = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements, power1, power2, PERTURB1, PERTURB2)
        
        # compute H1 error of Q at time T
        dif_Q1 = Q1[-1] - Q1_0[-1]
        dif_Q2 = Q2[-1] - Q2_0[-1]
        
        Q1_error = np.sqrt((dif_Q1.T @ stiffness_matrix @ dif_Q1)) + np.sqrt(np.sum(dif_Q1**2 * gamma))
        Q2_error = np.sqrt((dif_Q2.T @ stiffness_matrix @ dif_Q2)) + np.sqrt(np.sum(dif_Q2**2 * gamma))
        
        total_error = Q1_error + Q2_error
        errors.append(total_error)
        
        
    plt.loglog(sigmas, errors, label="$H^1$ Error", c='C0')
    plt.scatter(sigmas, errors, c='C0')
    plt.loglog(sigmas, np.sqrt(sigmas), label="$O(\sqrt{\sigma})$ Reference", c='C1')
    plt.loglog(sigmas, sigmas, label="$O(\sigma)$ Reference", c='C2')
    plt.xlabel("$\sigma$")
    plt.ylabel("$H^1$ Error")
    plt.legend()
    plt.savefig(f"{si}.pdf", bbox_inches="tight")
    plt.show()
    
    errors = np.array(errors)
    rates = (np.log(errors[1:]) - np.log(errors[:-1])) / (np.log(sigmas[1:]) - np.log(sigmas[:-1]))
    print(setting, rates)

    
