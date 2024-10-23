import time
import cProfile

# imports from local files
from energy import plot_energy
from main_solver import solve_Q_flow

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def visualize(points, Q1, Q2, r, num_interior_points, Nt, sigma, M, L, t_final, A0):
    director1x = np.zeros((Nt, points.shape[0]))
    director1y = np.zeros((Nt, points.shape[0]))
    director2x = np.zeros((Nt, points.shape[0]))
    director2y = np.zeros((Nt, points.shape[0]))
    
    eVals = np.zeros((Nt, num_interior_points))

    Q = np.zeros((Nt, num_interior_points, 2, 2))

    max_eigenvals_in_time = []
    all_eigenvalues = np.zeros((Nt, len(points)))

    for t in tqdm(range(Nt)):
        eigs_list = np.zeros(2*len(points))
        max_eigenvals_in_time.append(0)
        idx = 0
        for i, x in enumerate(points):
            
            Q[t, idx, 0, 0] = Q1[t, idx]
            Q[t, idx, 0, 1] = Q2[t, idx]
            Q[t, idx, 1, 0] = Q2[t, idx]
            Q[t, idx, 1, 1] = -Q1[t, idx]
            eigenvalues, eigenvectors = np.linalg.eigh(Q[t, idx])
            eigs_list[i] = np.max(eigenvalues)
            eigs_list[len(points)+i] = np.min(eigenvalues)
            all_eigenvalues[t, i] = np.max(eigenvalues)
            if eigenvalues[1] > max_eigenvals_in_time[-1]: # check the positive eigenvalue (not the other since trace free)
                max_eigenvals_in_time[-1] = eigenvalues[1]
            #print(eigenvalues)
            eVals[t, idx] = eigenvalues[0]
            v1 = eigenvectors[:, 0]
            v2 = eigenvectors[:, 1]
            director1x[t, i] = v1[0]
            director1y[t, i] = v1[1]
            director2x[t, i] = v2[0]
            director2y[t, i] = v2[1]

            idx += 1
        
        #plt.clf()
        #plt.hist(eigs_list)
        #plt.pause(0.001)
        plt.clf()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        plt.quiver(points[:, 0], points[:, 1], director2x[t], director2y[t], headaxislength=0, headwidth=0, headlength=0, color='b')
        #plt.title(f"t = {t_final*t/Nt}")
        plt.pause(0.001)
        plt.savefig(f"./plots/time-{t}.png")
    
    plt.show()
        
    return np.arange(Nt), max_eigenvals_in_time, all_eigenvalues

if __name__=='__main__':

    # Omega = [0, Lx] x [0, Lx]
    Lx = 2 # we currently assume it's 2, so changing this does nothing.

    a = -0.2
    b = 1
    c = 1
    A0 = 500

    M = 1
    sigma = 0.0#1.0

    L = 0.1
    Nt = 200
    t_final = 5#10
    
    for sigma in [1.0]:#, 1.0, 2.0]:#, 0.25, 0.5, 1.0, 2.0]:
        mesh, interior_point_coords, num_interior_points, tVals, gamma, stiffness_matrix, Q1, Q2, r = solve_Q_flow(Lx, t_final, Nt, a, b, c, A0, M, sigma, L)
        print(Q1.shape, Q2.shape)



        times, eigvals, all_eigvals = visualize(interior_point_coords, Q1, Q2, r, num_interior_points, Nt, sigma, M, L, t_final, A0)
        energy = plot_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix)

    plt.imshow(Q1[0].reshape(31, 31))
    plt.show()
    plt.imshow(Q1[-1].reshape(31, 31))
    plt.show()
    

        #plt.plot(times, eigvals, label=f"$\sigma = ${sigma}")
        #dt = tVals[1] - tVals[0]
        #eigvals_bound = (eigvals[0]*np.sqrt(3*num_interior_points*np.max(gamma)) + np.sqrt(energy[0]*dt*(np.arange(len(eigvals))))) / np.sqrt(3/2*np.min(gamma))
        #plt.plot(times, eigvals_bound, label="bound")
        #plt.plot(times, np.mean(all_eigvals, axis=1), label="avg")

    #plt.xlabel("Time")
    #plt.ylabel("Eigenvalue")
    #plt.legend()
    #plt.show()

    # for i in range(0, all_eigvals.shape[1], 20):
    #     plt.plot(np.arange(all_eigvals.shape[0]), all_eigvals[:, i])
    # plt.show()
