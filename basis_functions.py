import numpy as np
from scipy.sparse import csr_matrix
import skfem as fem
from skfem.helpers import dot, grad
import matplotlib.pyplot as plt

import os
import sys

#######################################################################
"""
Bilinear form is \int grad(u) * grad(v) dx
"""
@fem.BilinearForm
def B(u, v, _):
    return dot(grad(u), grad(v))

"""
Linear form is \int v dx
"""
@fem.LinearForm
def L(v, w):
    return v

#######################################################################

def get_triangulation(refinements=5):
    """
    refinements (int): more refinements = finer mesh
    """
    mesh = fem.MeshTri(doflocs = np.array([[0., 2., 0., 2.], [0., 0., 2., 2.]])).refined(refinements)
    interior_point_coords = np.array(mesh.to_dict()['p'])[mesh.interior_nodes()]
    return mesh, interior_point_coords

def calculate_basis_integrals(mesh):
    """
    Returns a vector with gammas and stiffness matrix for
    interior points of the mesh

    mesh: calculated by get_triangulation
    """

    # define piecewise linear basis functions from the mesh
    Vh = fem.Basis(mesh, fem.ElementTriP1())


    # calculate the stiffness matrix and gamma for all points
    # (including the boundary)
    stiffness_matrix = B.assemble(Vh)#.toarray()
    gamma = L.assemble(Vh)


    # filter the stiffness matrix and gamma to just the values
    # for interior points
    stiffness_matrix = csr_matrix(stiffness_matrix[mesh.interior_nodes()][:, mesh.interior_nodes()])
    gamma = gamma[mesh.interior_nodes()]

    return stiffness_matrix, gamma


