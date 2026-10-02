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


def setup_poiseuille_2d_lagrangian(
    L: float = 15.0,
    D: float = 1.0,
    U_avg: float = 0.1,
    rho: float = 1.0,
    mu: float = 1e-3,
    G: float | None = None,
    n_refine: int = 1,
    buffer_width: float = 2.0,
    cdist: float = 1e-10,
    wall_tol: float = 1e-10,
) -> tuple:
    """Developing Lagrangian channel flow on [0, L] x [0, D]: the mesh,
    BCs and ICs of ``Hagen_Poiseuile_2D.py``.

    Mesh: unit square ``[0, 1] x [0, D]`` refined *n_refine* times and
    extruded to length *L*.  BCs, in this order:
    ``OutletBufferedDeleteBC`` (outlet at ``L``, ghost buffer of
    *buffer_width*), ``PeriodicInletBC`` (ghost copy of the unit mesh
    advancing at *U_avg*), ``PositionalNoSlipWallBC`` on the wall lines
    ``y = 0`` and ``y = D``.  The wall BC runs LAST so that a wall-row
    vertex injected by the inlet is zeroed and added to *bV* in the same
    pass; under ``frozen_set='membership'`` nothing else would freeze it
    (the retopology no longer captures hull vertices).  ICs: plug flow
    ``U_avg``, linear pressure ``-G x``, uniform mass.

    Returns
    -------
    HC, bV, bc_set, wall_criterion, params
        *bV* is the whole hull; pass ``boundary_filter=wall_criterion`` to
        the integrator so that only wall vertices are frozen.
    """
    from hyperct.ddg import compute_vd

    from ddgclib._boundary_conditions import (
        OutletBufferedDeleteBC,
        PeriodicInletBC,
        PositionalNoSlipWallBC,
    )
    from ddgclib.geometry._complex_operations import extrude
    from ddgclib.initial_conditions import UniformVelocity

    dim = 2
    if G is None:
        G = 8 * mu * U_avg / (D / 2) ** 2

    def unit():
        HC_unit = Complex(dim, domain=[(0.0, 1), (0.0, D)])
        HC_unit.triangulate()
        for _ in range(n_refine):
            HC_unit.refine_all()
        return HC_unit

    HC = extrude(unit(), L, axis=0, cdist=1e-10)
    bV = HC.boundary(HC.V)
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method="barycentric")

    def wall_criterion(v):
        return abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - D) < wall_tol

    # Ghost source of the periodic inlet: one unit cell with the inlet state.
    unit_mesh = unit()
    CompositeIC(
        UniformVelocity(u_vec=np.array([U_avg, 0.0])),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=1 * D, rho=rho),
    ).apply(unit_mesh, set())

    bc_set = BoundaryConditionSet()
    bc_set.add(
        OutletBufferedDeleteBC(outlet_pos=L, buffer_width=buffer_width,
                               axis=0, bV=bV),
        None,
    )
    bc_set.add(
        PeriodicInletBC(unit_mesh=unit_mesh, velocity=U_avg, axis=0,
                        inlet_pos=0.0, cdist=cdist, fields=['u', 'p', 'm'],
                        period=1.0),  # = x-span of the unit mesh
        None,
    )
    bc_set.add(
        PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=dim, bV=bV),
        None,
    )

    CompositeIC(
        UniformVelocity(u_vec=np.array([U_avg, 0.0])),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=L * D, rho=rho),
    ).apply(HC, bV)
    bc_set.apply_all(HC, bV, dt=0.0)   # walls start at rest

    params = {
        'dim': dim, 'L': L, 'D': D, 'U_avg': U_avg, 'rho': rho, 'mu': mu,
        'G': G, 'n_refine': n_refine, 'buffer_width': buffer_width,
        'cdist': cdist, 'wall_tol': wall_tol,
        'poiseuille_ic': PoiseuillePlanar(
            G=G, mu=mu, y_lb=0.0, y_ub=D, flow_axis=0, normal_axis=1,
            dim=dim),
    }
    return HC, bV, bc_set, wall_criterion, params


def wall_snapshot(HC, wall_criterion) -> dict:
    """``{id(v): (v, position)}`` of the vertices on the walls now."""
    return {id(v): (v, v.x_a.copy()) for v in HC.V if wall_criterion(v)}


def wall_report(HC, bV, snapshot: dict) -> dict:
    """What became of the wall vertices of *snapshot*: how many are still
    in the complex, how many are still frozen, how far the farthest one
    moved."""
    alive = {id(v) for v in HC.V}
    frozen = {id(v) for v in bV}
    moved = [float(np.linalg.norm(v.x_a - x0))
             for v, x0 in snapshot.values()]
    return {
        'n_wall_start': len(snapshot),
        'n_in_complex': sum(k in alive for k in snapshot),
        'n_frozen': sum(k in frozen for k in snapshot),
        'n_moved': sum(d > 0.0 for d in moved),
        'max_displacement': max(moved, default=0.0),
    }
