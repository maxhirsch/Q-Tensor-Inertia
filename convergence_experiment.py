import time
import cProfile

# imports from local files
from energy import plot_energy
from main_solver import solve_Q_flow

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, griddata
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

space_experiment_only = True

# Omega = [0, Lx] x [0, Lx]
Lx = 2 # we currently assume it's 2, so changing this does nothing.

a = -0.2
b = 1
c = 1
A0 = 500

M = 1
sigma = 0.025

L = 0.001
Nt = 1600
t_final = 0.1

if not space_experiment_only:
    refinements = 5
    
    errors_Q1 = []
    errors_Q2 = []
    errors_r = []
    
    # first solve the Q-tensor flow for sigma = 0
    mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1_0, Q2_0, r_0 = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements)

    # solve Q-tensor flow for various dt values and compute error
    Nts = np.array([25, 50, 100, 200, 400])
    for Nt in Nts:  
        # compute solution
        mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1, Q2, r = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements)
    
        # compute H1 error of Q at time T
        dif_Q1 = Q1[-1] - Q1_0[-1]
        dif_Q2 = Q2[-1] - Q2_0[-1]
        dif_r  = r[-1] - r_0[-1]
    
        Q1_error = np.sqrt((dif_Q1.T @ stiffness_matrix @ dif_Q1)) + np.sqrt(np.sum(dif_Q1**2 * gamma))
        Q2_error = np.sqrt((dif_Q2.T @ stiffness_matrix @ dif_Q2)) + np.sqrt(np.sum(dif_Q2**2 * gamma))
        r_error  = np.sqrt(np.sum(dif_r**2 * gamma))
    
        errors_Q1.append(Q1_error)
        errors_Q2.append(Q2_error)
        errors_r.append(r_error)
    
    dts = t_final / Nts
    print(dts)
    
    plt.loglog(dts, errors_Q1, label="$H^1$ Error, $Q_{11}$", c='C0')
    plt.loglog(dts, errors_Q2, label="$H^1$ Error, $Q_{12}$", c='C1')
    plt.scatter(dts, errors_Q1, c='C0')
    plt.scatter(dts, errors_Q2, c='C1')
    plt.loglog(dts, dts, label="$O(\Delta t)$ Reference", c='C2')
    plt.xlabel("$\Delta t$")
    plt.ylabel("Error")
    plt.legend()
    plt.xticks(dts, dts)
    plt.savefig("time-experiment-Q.pdf", bbox_inches="tight")
    plt.show()

    plt.loglog(dts, errors_r, label="$L^2$ Error, $r$", c='C0')
    plt.scatter(dts, errors_r, c='C0')
    plt.loglog(dts, dts/1000, label="$O(\Delta t)$ Reference", c='C2')
    plt.xlabel("$\Delta t$")
    plt.ylabel("Error")
    plt.legend()
    plt.xticks(dts, dts)
    plt.savefig("time-experiment-r.pdf", bbox_inches="tight")
    plt.show()

    errors_Q1 = np.array(errors_Q1)
    rates_Q1 = (np.log(errors_Q1[1:]) - np.log(errors_Q1[:-1])) / (np.log(dts[1:]) - np.log(dts[:-1]))
    print("Q1 errors", errors_Q1)
    print("Q1 rates", rates_Q1)

    errors_Q2 = np.array(errors_Q2)
    rates_Q2 = (np.log(errors_Q2[1:]) - np.log(errors_Q2[:-1])) / (np.log(dts[1:]) - np.log(dts[:-1]))
    print("Q2 errors", errors_Q2)
    print("Q2 rates", rates_Q2)

    errors_r = np.array(errors_r)
    rates_r = (np.log(errors_r[1:]) - np.log(errors_r[:-1])) / (np.log(dts[1:]) - np.log(dts[:-1]))
    print("r errors", errors_r)
    print("r rates", rates_r)



Nt = 400
refinements = 9
errors_Q1 = []
errors_Q2 = []
errors_r  = []
hs = []

print("Reference solution h =", 2 / 2**refinements)

# first solve the Q-tensor flow for sigma = 0
mesh, interior_point_coords_0, num_interior_points, tVals, gamma_0, stiffness_matrix_0, Q1_0, Q2_0, r_0 = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements)

refinements_list = [2, 3, 4, 5, 6]
for refinements in refinements_list:
    
    # solve on coarse mesh

    mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1, Q2, r = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L, refinements)

    # scale up to fine mesh
    Q1_ = griddata(interior_point_coords, Q1[-1], interior_point_coords_0, fill_value=0.0, method='linear')
    Q2_ = griddata(interior_point_coords, Q2[-1], interior_point_coords_0, fill_value=0.0, method='linear')
    r_  = griddata(interior_point_coords, r[-1], interior_point_coords_0, fill_value=0.0, method='linear')

    # compute error between coarse and fine mesh
    dif_Q1 = Q1_ - Q1_0[-1]
    dif_Q2 = Q2_ - Q2_0[-1]
    dif_r  = r_ - r_0[-1]

    Q1_error = np.sqrt((dif_Q1.T @ stiffness_matrix_0 @ dif_Q1)) + np.sqrt(np.sum(dif_Q1**2 * gamma_0))
    Q2_error = np.sqrt((dif_Q2.T @ stiffness_matrix_0 @ dif_Q2)) + np.sqrt(np.sum(dif_Q2**2 * gamma_0))
    r_error = np.sqrt(np.sum(dif_r**2 * gamma_0))
    
    errors_Q1.append(Q1_error)
    errors_Q2.append(Q2_error)
    errors_r.append(r_error)

    hs.append(2 / 2**refinements)

hs = np.array(hs)
print(hs)

plt.loglog(hs, errors_Q1, label="$H^1$ Error, $Q_{11}$", c="C0")
plt.loglog(hs, errors_Q2, label="$H^1$ Error, $Q_{12}$", c="C1")
plt.scatter(hs, errors_Q1, c="C0")
plt.scatter(hs, errors_Q2, c="C1")
plt.loglog(hs, hs, label="$O(h)$ Reference", c="C2")
plt.xlabel("$h$")
plt.ylabel("Error")
plt.legend()
plt.xticks([], [], minor=True)
plt.xticks([], [], minor=False)
plt.xticks(hs, hs)
plt.savefig("space-experiment-Q.pdf", bbox_inches="tight")
#plt.show()
plt.clf()


plt.loglog(hs, errors_r, label="$L^2$ Error, $r$", c="C0")
plt.scatter(hs, errors_r, c="C0")
plt.loglog(hs, 100*hs, label="$O(h)$ Reference", c="C2")
plt.xlabel("$h$")
plt.ylabel("Error")
plt.legend()
plt.xticks([], [], minor=True)
plt.xticks([], [], minor=False)
plt.xticks(hs, hs)
plt.savefig("space-experiment-r.pdf", bbox_inches="tight")
#plt.show()
plt.clf()

errors_Q1 = np.array(errors_Q1)
rates_Q1  = (np.log(errors_Q1[1:]) - np.log(errors_Q1[:-1])) / (np.log(hs[1:]) - np.log(hs[:-1]))
print("Q1 errors", errors_Q1)
print("Q1 rates", rates_Q1)

errors_Q2 = np.array(errors_Q2)
rates_Q2  = (np.log(errors_Q2[1:]) - np.log(errors_Q2[:-1])) / (np.log(hs[1:]) - np.log(hs[:-1]))
print("Q2 errors", errors_Q2)
print("Q2 rates", rates_Q2)

errors_r = np.array(errors_r)
rates_r = (np.log(errors_r[1:]) - np.log(errors_r[:-1])) / (np.log(hs[1:]) - np.log(hs[:-1]))
print("r errors", errors_r)
print("r rates", rates_r)


