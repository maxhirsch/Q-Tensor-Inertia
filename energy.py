import numpy as np
import matplotlib.pyplot as plt


def get_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix):
    assert len(Q1.shape) == 2 and len(Q2.shape) == 2
    assert Q1.shape[0] == Nt and Q2.shape[0] == Nt
    #assert Q1.shape[1] == Nx**2 and Q2.shape[1] == Nx**2

    energy = []
    expected_energy = []
    
    L1 = L

    tVals = np.linspace(0, t_final, Nt)
    dt = tVals[1] - tVals[0]

    
    # iterate over n + 1
    for np1 in range(Nt):
        
        #### first calculate norm of D_t^+ Q^n ####
        if np1 == 0:
            # we assume that the time -1 is same as time 0
            norm_D_t_plus_Qsq = 0
        else:
            D_t_plus_Q1 = (Q1[np1] - Q1[np1-1]) / dt
            D_t_plus_Q2 = (Q2[np1] - Q2[np1-1]) / dt
            
            norm_D_t_plus_Qsq = np.sum(2 * (D_t_plus_Q1**2 + D_t_plus_Q2**2) * gamma)

        #### calculate norm of grad Q^{n+1} ####
        norm_grad_Qsq = np.sum(2 * (
            Q1[np1].reshape(1, -1) @ stiffness_matrix @ Q1[np1].reshape(-1, 1) + \
            Q2[np1].reshape(1, -1) @ stiffness_matrix @ Q2[np1].reshape(-1, 1)
        ))
        #print(norm_grad_Qsq)

        #### calculate norm of r^{n+1} ####
        norm_rsq = np.sum(r[np1]**2 * gamma)
        print(norm_rsq)
        
        assert norm_D_t_plus_Qsq >= 0 and norm_grad_Qsq >= 0 and norm_rsq >= 0

        energy.append(
            sigma/2 * norm_D_t_plus_Qsq + M*L1/2 * norm_grad_Qsq + M/2 * norm_rsq
        )

        if np1 <= 1:
            expected_energy.append(energy[-1])
        else:
            D_t_nm1_Q1 = (Q1[np1-1] - Q1[np1-2]) / dt
            D_t_nm1_Q2 = (Q2[np1-1] - Q2[np1-2]) / dt

            norm_D_t_diff_sq = np.sum(2 * ((D_t_plus_Q1 - D_t_nm1_Q1)**2 + (D_t_plus_Q2 - D_t_nm1_Q2)**2) * gamma)

            expected_energy.append(
                expected_energy[-1] - dt * norm_D_t_plus_Qsq - sigma/2 * norm_D_t_diff_sq
            )

    return energy, expected_energy

def plot_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix):
    energy, expected = get_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix)
    plt.plot(np.arange(len(energy)), energy, label="Energy")
    plt.plot(np.arange(len(energy)), expected, label="Expected Energy")
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(len(energy)), np.abs(np.array(energy) - np.array(expected)))
    plt.title("|Expected Energy - Energy|")
    plt.show()

    return energy

