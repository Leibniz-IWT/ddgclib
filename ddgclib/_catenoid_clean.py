import os, sys

import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#from ddgclib._catenoid_clean import *
from ddgclib._curvaturesplay import *
import copy

from timeit import default_timer as timer
#%%
def sech(x):
    return 1 / np.cosh(x)

def catenoid(u, v, a):
    x = a * np.cos(u) * np.cosh(v / a)
    y = a * np.sin(u) * np.cosh(v / a)
    z = v
    return x, y, z
#%%
def catenoid_clean_N(r, length, refinement=2):

    v_l = -length/2
    v_u = length/2
    a = r

    domain = [
              (0.0, 2 * np.pi),  # u
              (v_l, v_u)  # v
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
        #print(f'-')
        #print(f'v.x = {v.x}')
        x, y, z = catenoid(*v.x_a,r)
        #print(f'tuple(x, y, z) = {tuple([x, y, z])}')
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
        for vn in v.nn:
            v1 = list(HC.V)[v.index]
            v2 = list(HC.V)[vn.index]
            v1.connect(v2)

    # Remerge:
    HC.V.merge_all(cdist=cdist)
    bVc = copy.copy(bV)
    for v in bVc:
        #print(f'bv = {v.x}')
        if not (v in HC.V):
            bV.remove(v)

    H_f = []
    K_f = []

    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if not (vert in bV):
            for u_i, v_i in zip(u_list, v_list):
                x, y, z = catenoid(u_i, v_i,r)
                va = np.array([x, y, z])
                #print(f'va == vert.x_a = {va == vert.x_a}')
                ba = va == vert.x_a
                if ba.all():
                    #print(f'ba.all() = {ba.all()}')
                    u = u_i
                    v = v_i
                    H_f_i = 0.0
                    K_f_i = -(sech(v/a))**4 / (a**2)
                    H_f.append(H_f_i)
                    K_f.append(K_f_i)
                    #TODO: MERGE NEAR VERTICES

                    if z == 0.0:
                        neck_verts.append(vert.index)
                        neck_sols.append((H_f_i, K_f_i))


    return HC, bV, K_f, H_f, neck_verts, neck_sols