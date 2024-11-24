# NOTE: CP is used for the Equation of State:
#from CoolProp.CoolProp import PropsSI
# Imports and physical parameters
import numpy as np
import scipy
#from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#from skimage import data, io
import sys
#import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from ddgclib import *
from ddgclib._complex import Complex


# Colour scheme for surfaces
db = np.array([129, 160, 189]) / 255  # Dark blue
lb = np.array([176, 206, 234]) / 255  # Light blue


def mean_flow(HC, bV, m_0, tau=0.5, h_0=0.0, gravity_field=True):
    """
    Compute a single iteration of mean curvature flow
    :param HC: Simplicial complex
    :param bV: A set of boundary vertices
    :param m_0: The initial mass of the water droplet
    :return:


    # Test changes for errors:
    mean_flow(HC, bV, 1.0, tau=0.01)

    """
    # Define the normals (NOTE: ONLY WORKS FOR CONVEX SURFACES(?))
    # Note that the cache HC.V is an OrderedDict:
    N = []

    V = set(HC.V) - bV
    V_a = []
    f = []

    for v in V:
        f.append(v.x_a)

    f = np.array(f)
    # COM = numpy.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3])
    com = np.average(f, axis=0)
    # print(f'com = {com}')
    for v in V:
        N.append(normalized(v.x_a - com)[0])

    NdA = []
    HNdA = []
    HN = []  # sum(HNdA_ij, axes=0) / C_ijk
    C = []
    for i, v in enumerate(V):
        F, nn = vectorise_vnn(v)
        try:
            c_outd = curvatures(F, nn, n_i=N[i])
        except IndexError:
            print(f'WARNING, IndexError in loop')
            c_outd = {}
            c_outd['HNdA_i'] = np.zeros(3)
            c_outd['C_ijk'] = np.array([1.0])
            c_outd['NdA_i'] = np.zeros(3)

        NdA.append(c_outd['NdA_i'])
        HN.append(c_outd['HNdA_i'] / np.sum(c_outd['C_ijk']))
        C.append(np.sum(c_outd['C_ijk']))
        # f.append(v.x_a)

    H = np.nan_to_num(-np.array(HN))
    NdA = np.nan_to_num(np.array(NdA))
    dP = -gamma * H  # Hydrostatic pressure

    # Add volume perservation
    # First note that the volume gradient points in the same direction as the
    # vector area NV (i.e. they are the same up to a constant factor).
    # Therefore we simply need to compute the different between the current
    # volume and the equilibrium volume and add a scalar multiplier
    Rho = []
    for dp in dP:
        # p = 101.325 + dp
        p = 101.325 + np.sum(dp)  # TODO: Validate
        # print(f'p = {p}')
        rho = eos(P=p)  # kg m-3
        Rho.append(rho)
        # TODO: Compute local signed
    Rho = np.array(Rho)
    # print(f'Rho = {Rho}')
    V_eq = m_0 / Rho  # m-3
    V_eq = np.mean(m_0 / Rho)  # m-3
    V_eq = 0.02962e-6  # TEMPORARY
    print(f'V_eq = {V_eq}')

    N = np.array(N)
    # print(f'N_i = {N}')
    # print(f'NdA = {NdA}')
    if 0:
        print(f'np.sum(N) = {np.sum(N)}')
        print(f'np.sum(NdA) = {np.sum(NdA)}')
        print(f'np.sum(NdA*N) = {np.sum(NdA * N)}')
        print(f'NdA.dot(N) = {NdA.T.dot(N)}')
        print(f'sum NdA.dot(N) = {np.sum(NdA.T.dot(N))}')

    V_current = np.sum(C) / 6.0
    print(f'V_current = {V_current}')
    # dV =  -(V_current - V_eq)*H
    V_current = V_eq * 0.9  # TEMPORARY
    dV = 2 * 6 * (V_eq - V_current) * N

    # Add gravity force rho * g * h (Validated)
    if gravity_field:
        h = f[:, 2] - h_0
        g_v = np.zeros_like(dP)
        g_v[:, 2] = -g
        dg = rho * (h * g_v.T).T
    else:
        dg = np.zeros_like(dP)

    # dV = np.zeros_like(dP)

    df = dP + dV + dg
    # print(f'df = {df}')
    f_k = f + tau * df
    # print(f'df = {df}')
    for i, v in enumerate(V):
        # print(f'f_k[i] = {f_k[i]}')
        HC.V.move(HC.V[tuple(v.x_a)], tuple(f_k[i]))

    return HC



