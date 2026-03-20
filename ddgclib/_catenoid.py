import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt

from ddgclib.geometry._parametric_surfaces import parametric_surface, _second_axis_boundary


def catenoid_N(r, theta_p, gamma, abc, refinement=2, cdist=1e-10, equilibrium=True):
    """Catenoid surface mesh.  Delegates to ``parametric_surface()``."""
    v_l, v_u = -1.5, 1.5
    a, b, c = abc

    def sech(x):
        return 1 / np.cosh(x)

    def catenoid_fn(u, v):
        return (
            a * np.cos(u) * np.cosh(v / a),
            a * np.sin(u) * np.cosh(v / a),
            v,
        )

    # Build mesh via parametric_surface
    HC, bV = parametric_surface(
        catenoid_fn, [(0.0, 2 * np.pi), (v_l, v_u)],
        refinement, cdist=1e-8,
        boundary_fn=_second_axis_boundary,
    )

    # Compute analytical curvatures for interior vertices
    H_f = []
    K_f = []
    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if vert not in bV:
            z = vert.x_a[2]
            H_f_i = 0.0  # Catenoid is a minimal surface
            K_f_i = -(sech(z / a)) ** 4 / (a ** 2)
            H_f.append(H_f_i)
            K_f.append(K_f_i)
            if z == 0.0:
                neck_verts.append(vert.index)
                neck_sols.append((H_f_i, K_f_i))

    return HC, bV, K_f, H_f, neck_verts, neck_sols



def acatenoid_N(r, theta_p, gamma, abc, refinement=2, cdist=1e-10, equilibrium=True):
    """Asymmetric catenoid surface mesh.  Delegates to ``parametric_surface()``."""
    v_l, v_u = -1.5, 1.5
    a, b, c = abc

    def sech(x):
        return 1 / np.cosh(x)

    def acatenoid_fn(u, v):
        return (
            a * np.cos(u) * np.cosh(v / a) + 0.5 * v * np.cos(u),
            a * np.sin(u) * np.cosh(v / a) + 0.5 * v * np.sin(u),
            v,
        )

    HC, bV = parametric_surface(
        acatenoid_fn, [(0.0, 2 * np.pi), (v_l, v_u)],
        refinement, cdist=1e-8,
        boundary_fn=_second_axis_boundary,
    )

    # Analytical curvatures (same as catenoid — H=0 is not exact for asymmetric)
    H_f = []
    K_f = []
    neck_verts = []
    neck_sols = []
    for vert in HC.V:
        if vert not in bV:
            z = vert.x_a[2]
            H_f_i = 0.0
            K_f_i = -(sech(z / a)) ** 4 / (a ** 2)
            H_f.append(H_f_i)
            K_f.append(K_f_i)
            if z == 0.0:
                neck_verts.append(vert.index)
                neck_sols.append((H_f_i, K_f_i))

    return HC, bV, K_f, H_f, neck_verts, neck_sols


