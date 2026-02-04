import numpy as np
from scipy.spatial import Delaunay
import math

from ddgclib.hyperct._complex import Complex
from ddgclib._misc import _set_boundary
from ddgclib.barycentric._duals import compute_vd



def unit_cylinder(r, refinements=1, height=5e-3, up='z', distr_law='sinusoidal'):
    # Construct the initial cube
    lb = -0.5
    ub = 0.5
    domain = [(lb, ub), ] * 3
    # symmetry = [0, 1, 1]
    HC = Complex(3, domain=domain, symmetry=None)
    HC.triangulate()
    for i in range(refinements):
        HC.refine_all()

    # NEW
    # Compute boundaries
    bV = set()
    for v in HC.V:
        if ((v.x_a[0] == lb or v.x_a[1] == lb or v.x_a[2] == lb) or
                (v.x_a[0] == ub or v.x_a[1] == ub or v.x_a[2] == ub)):
            bV.add(v)

    # boundaries which exclude the interior vertices on the top/bottom of
    # the cyllinder
    bV_sides = set()
    for v in HC.V:
        # if ((v.x_a[0] == lb or v.x_a[0] == ub) and
        #    (v.x_a[1] == lb or v.x_a[1] == ub)):
        if ((v.x_a[0] == lb or v.x_a[0] == ub) or
                (v.x_a[1] == lb or v.x_a[1] == ub)):
            bV_sides.add(v)
            # Special side boundary property
            v.side_boundary = True
        else:
            v.side_boundary = False
    # print(f'bV_sides = {bV_sides}')

    for bv in bV:
        _set_boundary(bv, True)
    for v in HC.V:
        if not (v in bV):
            _set_boundary(v, False)

    # for bv in bV:
    #    print(f'bv = {bv.x}')
    # Move the vertices to the tube radius
    for v in bV_sides:
        r_eff = r  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)
        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        #print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    for v in HC.V:
        # if v.boundary:
        if v.side_boundary:
            continue
        d = np.linalg.norm(v.x_a[:2])  # This is already a normalized distance for 0.5 bounds
        # Power law scaling:
        if 0:
            n = 0.5  # 0.6  # Aribtrarily chosen power law scaling, should be n<=r
            r_eff = d ** n * r  # Trancated radius projection
        # log law scaling:
        if 0:
            r_eff = r * (np.log(d + 1) / np.log(2))
        # Sinusoidal scaling:
        if distr_law=='sinusoidal':
            r_eff = (r * ((1 - np.cos(np.pi * d ** 0.5)) / 2))

        # r_eff = r/d  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        # Height scaling
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)

        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        #print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    return HC


# TODO: delete unused duplicate code below (originally used for caprise)
def cube_to_tube(r, refinements=1, height=5e-3):
    # Construct the initial cube
    lb = -0.5
    ub = 0.5
    domain = [(lb, ub), ] * 3
    # symmetry = [0, 1, 1]
    HC = Complex(3, domain=domain, symmetry=None)
    HC.triangulate()
    for i in range(refinements):
        HC.refine_all()

    # NEW
    # Compute boundaries
    bV = set()
    for v in HC.V:
        if ((v.x_a[0] == lb or v.x_a[1] == lb or v.x_a[2] == lb) or
                (v.x_a[0] == ub or v.x_a[1] == ub or v.x_a[2] == ub)):
            bV.add(v)

    # boundaries which exclude the interior vertices on the top/bottom of
    # the cyllinder
    bV_sides = set()
    for v in HC.V:
        # if ((v.x_a[0] == lb or v.x_a[0] == ub) and
        #    (v.x_a[1] == lb or v.x_a[1] == ub)):
        if ((v.x_a[0] == lb or v.x_a[0] == ub) or
                (v.x_a[1] == lb or v.x_a[1] == ub)):
            bV_sides.add(v)
            # Special side boundary property
            v.side_boundary = True
        else:
            v.side_boundary = False
    # print(f'bV_sides = {bV_sides}')

    for bv in bV:
        _set_boundary(bv, True)
    for v in HC.V:
        if not (v in bV):
            _set_boundary(v, False)

    # for bv in bV:
    #    print(f'bv = {bv.x}')
    # Move the vertices to the tube radius
    for v in bV_sides:
        r_eff = r  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)
        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        #print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    for v in HC.V:
        # if v.boundary:
        if v.side_boundary:
            continue
        d = np.linalg.norm(v.x_a[:2])  # This is already a normalized distance for 0.5 bounds
        # Power law scaling:
        if 0:
            n = 0.5  # 0.6  # Aribtrarily chosen power law scaling, should be n<=r
            r_eff = d ** n * r  # Trancated radius projection
        # log law scaling:
        if 0:
            r_eff = r * (np.log(d + 1) / np.log(2))
        # Sinusoidal scaling:
        if 1:
            r_eff = (r * ((1 - np.cos(np.pi * d ** 0.5)) / 2))

        # r_eff = r/d  # Trancated radius projection
        nv = np.zeros(3)
        theta = math.atan2(v.x_a[1], v.x_a[0])
        nv[0] = r_eff * np.cos(theta)
        nv[1] = r_eff * np.sin(theta)
        # Height scaling
        if 1:
            h_eff = height  # * ((1 - np.cos(np.pi * v.x[2]**0.5)) / 2)

        nv[2] = h_eff * (v.x[2] - 0.5)  # * 1e-3
        #print(f'nv[2] = {nv[2]}')
        HC.V.move(v, tuple(nv))

    return HC