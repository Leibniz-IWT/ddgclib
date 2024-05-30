import numpy as np
import copy

from ._curvaturesplay import *

def sech(x):
    return 1 / np.cosh(x)

def truncated_cone(u, v, r_l, r_u, length,v_l=0):
    '''
    u =
    v =
    r_l = radian of the initial truncated cone on the lower site
    r_u = radian of the initial truncated cone on the upper site
    length = length of the liquid bridge
    v_l = Starting position of the truncated cone i the z-direction
    '''
    v_u = length
    r = r_l + (r_u - r_l) * v / (v_u - v_l)

    # Parametrisierung Zylinder
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = v

    return x, y, z


def truncated_cone_initial_N(r_l, r_u, length, refinement,v_l = 0,shift_activated = False, x_shift = 0, y_shift=0, z_shift = 0,rotation=False,rotation_angle=45, shearing = False, shearing_parameter = 1):
    '''
    r_l = radian of the initial truncated cone on the lower site
    r_u = radian of the initial truncated cone on the upper site
    length = length of the liquid bridge
    refinement = refinement
    v_l = Starting position of the truncated cone i the z-direction

    '''

    v_u = length + v_l
    a = 1

    domain = [#(-2.0, 2.0),  # u
              #(0.0, 2 * np.pi),  # u
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
        x, y, z = truncated_cone(*v.x_a,r_l,r_u,length)#,shift_activated = shift_activated, x_shift = x_shift, y_shift=y_shift, z_shift = z_shift,rotation=rotation,rotation_angle=rotation_angle, shearing=shearing, shearing_parameter=shearing_parameter)
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
    bV_H_f = []
    K_f = []
    bV_K_f = []

    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if not (vert in bV):
            for u_i, v_i in zip(u_list, v_list):
                #catenoid(u, v,r_l,r_u,length)
                #x, y, z = truncated_cone(*v.x_a,r_l,r_u,length,rotation=rotation,rotation_angle=rotation_angle, shearing=shearing, shearing_parameter=shearing_parameter)
                x, y, z = truncated_cone(u_i, v_i,r_l,r_u,length)#,shift_activated = shift_activated, x_shift = x_shift, y_shift=y_shift, z_shift = z_shift,rotation=rotation,rotation_angle=rotation_angle, shearing=shearing, shearing_parameter=shearing_parameter)
                #x, y, z = truncated_cone(*v.x_a, r_l, r_u, length)
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

    if rotation:
        hcv = copy.copy(HC.V)
        for v in hcv:
            x1, y1, z1 = v.x[0], v.x[1], v.x[2]

            x = x1
            y = (np.cos(rotation_angle* np.pi/180.0) *y1 - np.sin(rotation_angle* np.pi/180.0)*z1)
            z = (np.sin(rotation_angle* np.pi/180.0) *y1 + np.cos(rotation_angle* np.pi/180.0) *z1)

            xa = (x, y, z)
            HC.V.move(v, xa)
        #


    if shift_activated:
        hcv = copy.copy(HC.V)
        for v in hcv:
            x, y, z = v.x[0], v.x[1], v.x[2]

            x += x_shift
            y += y_shift
            z += z_shift

            xa = (x, y, z)
            HC.V.move(v, xa)

    if shearing:
        hcv = copy.copy(HC.V)
        for v in hcv:
            x1, y1, z1 = v.x[0], v.x[1], v.x[2]

            x = x1
            y = z1 * shearing_parameter + y1
            z = z1

            xa = (x, y, z)
            HC.V.move(v, xa)

    return HC, bV, K_f, H_f, neck_verts, neck_sols


