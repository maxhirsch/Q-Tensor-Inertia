import numpy as np
import cProfile
from tqdm import tqdm
import matplotlib.pyplot as plt

#################### Intialization for the Q tensor Q0 ####################
def get_n0(x, y):
    n0 = x * (2-x) * y * (2-y)
    n1 = np.sin(np.pi*x) * np.sin(0.5*np.pi*y)
    return np.array([[n0], [n1]])

def get_Q0(n0):
    Q0 = n0 @ n0.T - np.sum(n0*n0)/2.0 * np.eye(2)
    return Q0

def get_dn0(x, y):
    n0 = 0.1 + np.abs(x + 0.1)
    n1 = 0.1 + np.abs(y + 0.1)

    res = np.array([[n0], [n1]])
    res = res / np.linalg.norm(res)

    return res

def get_D2Q0(x, y, sigma=0.0):
    dQ0 = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    dQ0[0, 0] = 2 * (2-x)**2 * x**2 * (2-y)**2 - 8 * (2-x)**2 * x**2 * (2-y) * y + 2 * (2-x)**2 * x**2 * y**2 + 2 * (2-x)**2 * (2-y)**2 * y**2 - 8 * (2-x) * x * (2-y)**2 * y**2 + 2 * x**2 * (2-y)**2 * y**2 + 0.5 * (-2 * (2-x)**2 * x**2 * (2-y)**2 + 8 * (2-x)**2 * x**2 * (2-y) * y - 2 * (2-x)**2 * x**2 * y**2 - 0.5 * np.pi**2 * np.cos(np.pi*y/2) * np.sin(np.pi*x)**2 + 0.5 * np.pi**2 * np.sin(np.pi*x)**2 * np.sin(np.pi*y/2)**2) + 0.5 * (-2 * (2-x)**2 * (2-y)**2 * y**2 + 8 * (2-x) * x * (2-y)**2 * y**2 - 2 * x**2 * (2-y)**2 * y**2 - 2 * np.pi**2 * np.cos(np.pi*x)**2 * np.sin(np.pi*y/2)**2 + 2 * np.pi**2 * np.sin(np.pi*x)**2 * np.sin(np.pi*y/2)**2)

    dQ0[0, 1] = np.pi * (2-x) * x * (2-y) * np.cos(np.pi*y/2) * np.sin(np.pi*x) - np.pi * (2-x) * x * y * np.cos(np.pi*y/2) * np.sin(np.pi*x) + 2 * np.pi * (2-x) * (2-y) * y * np.cos(np.pi*x) * np.sin(np.pi*y/2) - 2 * np.pi * x * (2-y) * y * np.cos(np.pi*x) * np.sin(np.pi*y/2) - 2 * (2-x) * x * np.sin(np.pi*x) * np.sin(np.pi*y/2) - 2 * (2-y) * y * np.sin(np.pi*x) * np.sin(np.pi*y/2) - 5/4 * np.pi**2 * (2-x) * x * (2-y) * y * np.sin(np.pi*x) * np.sin(np.pi*y/2)

    dQ0[1, 0] = np.pi * (2-x) * x * (2-y) * np.cos(np.pi*y/2) * np.sin(np.pi*x) - np.pi * (2-x) * x * y * np.cos(np.pi*y/2) * np.sin(np.pi*x) + 2 * np.pi * (2-x) * (2-y) * y * np.cos(np.pi*x) * np.sin(np.pi*y/2) - 2 * np.pi * x * (2-y) * y * np.cos(np.pi*x) * np.sin(np.pi*y/2) - 2 * (2-x) * x * np.sin(np.pi*x) * np.sin(np.pi*y/2) - 2 * (2-y) * y * np.sin(np.pi*x) * np.sin(np.pi*y/2) - 5/4 * np.pi**2 * (2-x) * x * (2-y) * y * np.sin(np.pi*x) * np.sin(np.pi*y/2)

    dQ0[1, 1] = 0.5 * np.pi**2 * np.cos(np.pi*y/2)**2 * np.sin(np.pi*x)**2 + 2 * np.pi**2 * np.cos(np.pi*x)**2 * np.sin(np.pi*y/2)**2 - 5/2 * np.pi**2 * np.sin(np.pi*x)**2 * np.sin(np.pi*y/2)**2 + 0.5 * (-2 * (2-x)**2 * x**2 * (2-y)**2 + 8 * (2-x)**2 * x**2 * (2-y) * y - 2 * (2-x)**2 * x**2 * y**2 - 0.5 * np.pi**2 * np.cos(np.pi*y/2)**2 * np.sin(np.pi*x)**2 + 0.5 * np.pi**2 * np.sin(np.pi*x)**2 * np.sin(np.pi*y/2)**2) + 0.5 * (-2 * (2-x)**2 * (2-y)**2 * y**2 + 8 * (2-x) * x * (2-y)**2 * y**2 - 2 * x**2 * (2-y)**2 * y**2 - 2 * np.pi**2 * np.cos(np.pi*x)**2 * np.sin(np.pi*y/2)**2 + 2 * np.pi**2 * np.sin(np.pi*x)**2 * np.sin(np.pi*y/2)**2)

    return dQ0

