import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt

def catenoid_N(r, theta_p, gamma, abc, refinement=2, cdist=1e-10, equilibrium=True):
   # Theta = np.linspace(0.0, 2 * np.pi, N)  # range of theta
   # v = Theta  # 0 to 2 pi
  #  u = 0.0    # [-2, 2.0
    #R = r / np.cos(theta_p)  # = R at theta = 0

    v_l, v_u = -1.5, 1.5
    #a, b, c = 1, 0.0, 1
    a, b, c = abc  # Test for unit sphere

    def sech(x):
        return 1 / np.cosh(x)

    def catenoid(u, v):
        x = a * np.cos(u) * np.cosh(v / a)
        y = a * np.sin(u) * np.cosh(v / a)
        z = v
        return x, y, z

    # Equation: x**2/a**2 + y**2/b**2 + z**2/c**2 = 1
    # TODO: Assume that a > b > c?
    # Exact values:
    R = a
    domain = [#(-2.0, 2.0),  # u
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
        x, y, z = catenoid(*v.x_a)
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

    for v in bV:
        pass
        #print(f'bv = {v.x}')

    if 0:
        u_listc, v_listc = copy.copy(u_list), copy.copy(v_list)
        print(f'u_list = {u_list}')
        print(f'v_list = {v_list}')
        for u_i, v_i in zip(u_listc, v_listc):
            x, y, z = catenoid(u_i, v_i)
            print(f' -')
            print(f' tuple([x, y, z]) in HC.V.cache = {tuple([x, y, z]) in HC.V.cache}')

            # print(f'ind = {ind}')
            # if tuple(x, y, z)
            if not (tuple([x, y, z]) in HC.V.cache):
                # pass
                u_ind = u_list.index(u_i)
                v_ind = v_list.index(v_i)
                print(f'u_i = {u_i}')
                print(f'u_ind = {u_ind}')
                print(f'v_i = {v_i}')
                print(f'v_ind = {v_ind}')
                u_list.pop(u_ind)
                v_list.pop(v_ind)

        u = np.array(u_list)
        v = np.array(v_list)
        print(f'u = {u}')
        print(f'v = {v}')
        nom = c * (a**2*(u**2 - 1) + c**2 * (u**2 + 1))
        denom = 2 * a * (u**2 * (a**2 + c**2) + c**2)**(3/2.0)
        H_f = nom / denom
        K_f = - c**2 / ((u**2 * (a**2 + c**2) + c**2)**2)

        for ind, vert in enumerate(HC.V):
            print(f'-')
            print(f'ind = {ind}')
            print(f'v = {vert.x}')
            #x, y, z = hyperboloid(u[ind], v[ind])
            print(f'x, y, z =  {x, y, z}')

            nom = c * (a**2*(u**2 - 1) + c**2 * (u**2 + 1))
            denom = 2 * a * (u**2 * (a**2 + c**2) + c**2)**(3/2.0)


        for u_i, v_i in zip(u_list, v_list):
            x, y, z = hyperboloid(u_i, v_i)
            print(f' tuple([x, y, z]) in HC.V.cache = {tuple([x, y, z]) in HC.V.cache}')

            vert = HC.V[tuple([x, y, z])]
            print(f'vert.index = {vert.index}')
            #print(f'ind = {ind}')
            #if tuple(x, y, z)
            if not (tuple([x, y, z]) in HC.V.cache):
                print('WARRNIGN! not in cache')

    H_f = []
    bV_H_f = []
    K_f = []
    bV_K_f = []

    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if not (vert in bV):
            for u_i, v_i in zip(u_list, v_list):
                x, y, z = catenoid(u_i, v_i)
                va = np.array([x, y, z])
                #print(f'va == vert.x_a = {va == vert.x_a}')
                ba = va == vert.x_a
                if ba.all():
                    #print(f'ba.all() = {ba.all()}')
                    u = u_i
                    v = v_i
                    #nom = c * (a ** 2 * (u ** 2 - 1) + c ** 2 * (u ** 2 + 1))
                    #denom = 2 * a * (u ** 2 * (a ** 2 + c ** 2) + c ** 2) ** (3 / 2.0)
                    H_f_i = 0.0
                    #K_f_i = - c ** 2 / ((u ** 2 * (a ** 2 + c ** 2) + c ** 2) ** 2)
                    #K_f_i = -(np.sech(v/a))**4 / (a**2)
                    #K_f_i = -(math.sech(v/a))**4 / (a**2)
                    K_f_i = -(sech(v/a))**4 / (a**2)
                    H_f.append(H_f_i)
                    K_f.append(K_f_i)
                    #TODO: MERGE NEAR VERTICES

                    if z == 0.0:
                        neck_verts.append(vert.index)
                        neck_sols.append((H_f_i, K_f_i))


    return HC, bV, K_f, H_f, neck_verts, neck_sols