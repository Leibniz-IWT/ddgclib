"""Hagen-Poiseuille case: setup functions using new IC/BC classes.

Sets up the fully-developed Poiseuille flow problem using the clean
IC/BC abstractions from ddgclib, replacing the manual loops in the
old _analytical_equil.py.

2D: Planar Poiseuille (channel) flow
3D: Hagen-Poiseuille (pipe) flow
"""

import numpy as np
from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    DirichletVelocityBC,
    NoSlipWallBC,
    identify_boundary_vertices,
    identify_cube_boundaries,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    LinearPressureGradient,
    PoiseuillePlanar,
    UniformMass,
    ZeroVelocity,
)


def setup_poiseuille_2d(
    G: float = 1.0,
    mu: float = 1.0,
    n_refine: int = 2,
    L: float = 1.0,
    h: float = 1.0,
    rho: float = 1.0,
) -> tuple:
    """Set up 2D planar Poiseuille flow on [0, L] x [0, h].

    Flow in x-direction, walls at y=0 and y=h.
    Analytical: u_x(y) = (G/(2*mu)) * y * (h - y), u_y = 0.

    Parameters
    ----------
    G : float
        Pressure gradient magnitude (dP/dx).
    mu : float
        Dynamic viscosity.
    n_refine : int
        Mesh refinement level.
    L : float
        Channel length in x.
    h : float
        Channel height in y.
    rho : float
        Fluid density.

    Returns
    -------
    HC, bV, ic, bc_set, params
    """
    HC = Complex(2, domain=[(0.0, L), (0.0, h)])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = identify_cube_boundaries(HC, lb=0.0, ub=max(L, h), dim=2)
    # More precise: find all boundary verts at domain edges
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14):
            bV.add(v)

    # Wall vertices (y=0 and y=h)
    bV_wall = identify_boundary_vertices(
        HC, lambda v: abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14
    )

    # ICs: analytical Poiseuille + linear pressure
    poiseuille_ic = PoiseuillePlanar(
        G=G, mu=mu, y_lb=0.0, y_ub=h,
        flow_axis=0, normal_axis=1, dim=2,
    )
    ic = CompositeIC(
        poiseuille_ic,
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=L * h, rho=rho),
    )

    # BCs: no-slip on walls, Dirichlet velocity (analytical) on inlet/outlet
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=2), bV_wall)

    params = {
        'dim': 2,
        'G': G,
        'mu': mu,
        'L': L,
        'h': h,
        'rho': rho,
        'flow_axis': 0,
        'normal_axis': 1,
        'U_max': G * h**2 / (8 * mu),
        'poiseuille_ic': poiseuille_ic,
    }

    return HC, bV, ic, bc_set, params
