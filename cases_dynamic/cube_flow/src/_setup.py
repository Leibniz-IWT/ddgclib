"""Cube flow case: setup functions for 1D, 2D, and 3D.

Uniform flow through a cube domain with a periodic inlet (Lagrangian)
and an open outlet. A small linear pressure drop is imposed along the
flow axis (x). The mesh advects with the flow; ``PeriodicInletBC``
injects new vertices at the inlet and ``OutletDeleteBC`` removes
vertices that exit at the outlet.

Initial conditions:
    u(x) = [u_inlet, 0, ...]   (uniform velocity in x)
    P(x) = P_ref - G * x[0]    (linear pressure drop in x)
"""

import numpy as np
from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    OutletDeleteBC,
    PeriodicInletBC,
    identify_boundary_vertices,
    identify_cube_boundaries,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    LinearPressureGradient,
    UniformMass,
    UniformVelocity,
)


def setup_cube_flow(
    dim: int,
    n_refine: int = 1,
    L: float = 1.0,
    u_inlet: float = 0.05,
    G: float = 0.001,
    mu: float = 0.1,
    rho: float = 1.0,
    static_walls: bool = False,
) -> tuple:
    """Set up uniform flow through a cube domain.

    Parameters
    ----------
    dim : int
        Spatial dimension (1, 2, or 3).
    n_refine : int
        Number of mesh refinement passes (default 1).
    L : float
        Domain side length (cube is [0, L]^dim).
    u_inlet : float
        Inlet velocity in x-direction [m/s].
    G : float
        Linear pressure gradient magnitude [Pa/m].
    mu : float
        Dynamic viscosity [Pa*s].
    rho : float
        Fluid density [kg/m^3].
    static_walls : bool
        If True, apply no-slip (zero velocity) on wall boundaries
        (all boundary faces except inlet/outlet). Wall vertices will
        not advect. If False (default), wall vertices flow dynamically
        with the fluid.

    Returns
    -------
    HC : Complex
        Simplicial complex (main mesh).
    bV : set
        All boundary vertices.
    ic : CompositeIC
        Initial conditions.
    bc_set : BoundaryConditionSet
        Boundary conditions (outlet delete + periodic inlet).
    unit_mesh : Complex
        Unit mesh used for periodic inlet injection (kept for reference).
    params : dict
        Physical parameters and identified boundary subsets.
    """
    # Main mesh
    domain = [(0.0, L)] * dim
    HC = Complex(dim, domain=domain)
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = identify_cube_boundaries(HC, 0.0, L, dim=dim)

    # Build velocity vector: u_inlet in x-direction
    u_vec = np.zeros(dim)
    u_vec[0] = u_inlet

    # Initial conditions
    ic = CompositeIC(
        UniformVelocity(u_vec),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=L**dim, rho=rho),
    )

    # Unit mesh for periodic inlet (separate Complex with same ICs)
    unit_mesh = Complex(dim, domain=domain)
    unit_mesh.triangulate()
    for _ in range(n_refine):
        unit_mesh.refine_all()
    # Apply ICs to unit mesh so ghost vertices carry correct field values
    unit_bV = identify_cube_boundaries(unit_mesh, 0.0, L, dim=dim)
    ic.apply(unit_mesh, unit_bV)

    # Boundary conditions (Lagrangian)
    bc_set = BoundaryConditionSet()
    # Outlet first: delete vertices that exit at x >= L
    bc_set.add(OutletDeleteBC(outlet_pos=L, axis=0))
    # Inlet: periodic injection from ghost at x = 0
    bc_set.add(PeriodicInletBC(
        unit_mesh, velocity=u_inlet, axis=0, inlet_pos=0.0,
        fields=['u', 'p', 'm'], period=L,
    ))

    # Identify inlet/outlet subsets for reference
    tol = 1e-14
    bV_inlet = identify_boundary_vertices(
        HC, lambda v: abs(v.x_a[0]) < tol
    )
    bV_outlet = identify_boundary_vertices(
        HC, lambda v: abs(v.x_a[0] - L) < tol
    )

    # Wall vertices: boundary vertices that are neither inlet nor outlet
    bV_walls = bV - bV_inlet - bV_outlet

    # Optionally apply no-slip wall BC to keep wall vertices static
    if static_walls and dim >= 2 and bV_walls:
        from ddgclib._boundary_conditions import NoSlipWallBC
        bc_set.add(NoSlipWallBC(dim=dim), bV_walls)

    params = {
        'dim': dim,
        'L': L,
        'u_inlet': u_inlet,
        'G': G,
        'mu': mu,
        'rho': rho,
        'flow_axis': 0,
        'static_walls': static_walls,
        'bV_inlet': bV_inlet,
        'bV_outlet': bV_outlet,
        'bV_walls': bV_walls,
    }

    return HC, bV, ic, bc_set, unit_mesh, params
