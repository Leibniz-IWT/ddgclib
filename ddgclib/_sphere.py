import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt

def sphere_N(R, Phi, Theta, refinement=2, cdist=1e-10, equilibrium=True):
    def sphere(R, theta, phi):
        return (R * np.cos(theta) * np.sin(phi),
                R * np.sin(theta) * np.sin(phi),
                R * np.cos(phi)
                )

    domain = [#(-2.0, 2.0),  # u
              (Theta[0], Theta[-1]),  # u
              (Phi[0], Phi[-1])  # v
              ]


    HC_plane = Complex(2, domain)
    HC_plane.triangulate()
    for i in range(refinement):
        HC_plane.refine_all()

    # H
    HC = Complex(3, domain)
    bV = set()
    cdist = 1e-8

    u_list = []
    v_list = []
    for v in HC_plane.V:
        #x, y, z = sphere(R, theta, phi)
        #  theta = v.x_a[0]
        #  phi = v.x_a[1]
        x, y, z = sphere(R, v.x_a[0], v.x_a[1])
        v2 = HC.V[tuple([x, y, z])]

        u_list.append(v.x_a[0])
        v_list.append(v.x_a[1])

        #TODO: Does not work at all:
        boundary_bool = (
                        v.x[1] == domain[1][0] or v.x[1] == domain[1][1]
                        #v.x[2] == domain[0][0] or v.x[2] == domain[0][1]
                       # or v.x[1] == domain[1][0] or v.x[1] == domain[1][1]
                         )
        if boundary_bool:
            bV.add(v2)

    # Connect neighbours
    for v in HC_plane.V:
        #print(f'ist(HC.V) = {list(HC.V)}')
        for vn in v.nn:
            try:
                v1 = list(HC.V)[v.index]
            except IndexError:
                x, y, z = sphere(R, v.x_a[0], v.x_a[1])
                v_read = HC.V[tuple([x, y, z])]
                v1 = list(HC.V)[v_read.index]
            try:
                v2 = list(HC.V)[vn.index]
            except IndexError:
                x, y, z = sphere(R, vn.x_a[0], vn.x_a[1])
                v2_read = HC.V[tuple([x, y, z])]
                v2 = list(HC.V)[v2_read.index]
            v1.connect(v2)

    # Remerge:
    HC.V.merge_all(cdist=cdist)
    bVc = copy.copy(bV)
    for v in bVc:
        #print(f'bv = {v.x}')
        if not (v in HC.V):
            bV.remove(v)


    return HC, bV#, K_f, H_f, neck_verts, neck_sols
