import numpy as np
import cProfile

################### Functions for the scheme ########################
def P(Q, a, b, c, A0):
    """ 
    Q (np.array): Nx2x2 Q-tensor
    returns: P(Q) as defined in the paper
    """
    if len(Q.shape) == 2:
        Q2 = np.matmul(Q, Q)
        trQ2 = np.trace(Q2)
        SQ = a*Q - b*(Q2 - 1/2 * trQ2*np.eye(2)) + c*trQ2*Q
        rQ = np.sqrt(2*(a/2 * trQ2 - b/3*np.trace(np.matmul(Q2, Q)) + c/4*trQ2**2 + A0))

        return SQ / rQ, SQ
    else:
        Q2 = np.matmul(Q, Q)
        trQ2 = np.trace(Q2, axis1=1, axis2=2)
        SQ = a*Q - b*(Q2 - 1/2 * trQ2.reshape(-1,1,1)*np.eye(2)) + c*trQ2.reshape(-1,1,1)*Q
        rQ = np.sqrt(2*(a/2 * trQ2 - b/3*np.trace(np.matmul(Q2, Q), axis1=1, axis2=2) + c/4*trQ2**2 + A0)).reshape(-1,1,1)
        return SQ / rQ, SQ