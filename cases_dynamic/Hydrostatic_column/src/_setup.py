"""Hydrostatic column case: setup functions for 1D, 2D, and 3D.

The hydrostatic equilibrium is the simplest test for the pressure gradient
operator. At equilibrium, the body force (gravity) exactly balances the
pressure gradient, giving zero acceleration everywhere.

Analytical solution: P(x) = P_ref + rho * g * (h_ref - x[gravity_axis])
                     u(x) = 0
"""

import numpy as np
from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    NoSlipWallBC,
    identify_cube_boundaries,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    HydrostaticPressure,
    UniformMass,
    ZeroVelocity,
)


def setup_hydrostatic(
    dim: int,
    n_refine: int = 2,
    rho: float = 1000.0,
    g: float = 9.81,
    h: float = 1.0,
    P_ref: float = 0.0,
    gravity_axis: int = None,
) -> tuple:
    """Set up a hydrostatic column problem.

    Parameters
    ----------
    dim : int
        Spatial dimension (1, 2, or 3).
    n_refine : int
        Number of mesh refinement passes.
    rho : float
        Fluid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    h : float
        Column height [m].
    P_ref : float
        Reference pressure at top (x[axis]=h).
    gravity_axis : int or None
        Axis along which gravity acts (default: dim-1, i.e. last axis).

    Returns
    -------
    HC : Complex
        Simplicial complex.
    bV : set
        Boundary vertex set.
    ic : CompositeIC
        Initial conditions.
    bc_set : BoundaryConditionSet
        Boundary conditions.
    params : dict
        Physical parameters for reference.
    """
    if gravity_axis is None:
        gravity_axis = dim - 1

    # Build domain [0, h]^dim
    domain = [(0.0, h)] * dim
    HC = Complex(dim, domain=domain)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    # Identify boundary
    bV = identify_cube_boundaries(HC, 0.0, h, dim=dim)

    # Initial conditions: hydrostatic pressure + zero velocity + uniform mass
    # h_ref = h (top of column), P_ref = 0 at top
    # P(x) = P_ref + rho * g * (h - x[gravity_axis])
    ic = CompositeIC(
        ZeroVelocity(dim=dim),
        HydrostaticPressure(
            rho=rho, g=g, axis=gravity_axis,
            h_ref=h, P_ref=P_ref,
        ),
        UniformMass(total_volume=h**dim, rho=rho),
    )

    # Boundary conditions: no-slip walls on all boundaries
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    params = {
        'dim': dim,
        'rho': rho,
        'g': g,
        'h': h,
        'P_ref': P_ref,
        'gravity_axis': gravity_axis,
    }

    return HC, bV, ic, bc_set, params
