"""Hydrostatic column case: setup functions for 1D, 2D, and 3D.

The hydrostatic equilibrium is the simplest test for the pressure gradient
operator. At equilibrium, the body force (gravity) exactly balances the
pressure gradient, giving zero acceleration everywhere.

Analytical solution: P(x) = P_ref + rho * g * (h_ref - x[gravity_axis])
                     u(x) = 0
"""

from functools import partial

import numpy as np
from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    NoSlipWallBC,
    identify_cube_boundaries,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    DualVolumeMass,
    HydrostaticPressure,
    UniformMass,
    ZeroVelocity,
)
from ddgclib.operators.stress import cache_dual_volumes, stress_acceleration


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


# ---------------------------------------------------------------------------
# Gravity body-force wrapper
# ---------------------------------------------------------------------------

def make_gravity_dudt(
    dim: int,
    mu: float,
    HC,
    g: float = 9.81,
    gravity_axis: int | None = None,
    pressure_model=None,
):
    """Create a dudt_fn that adds gravity as a body force.

    Returns a callable ``dudt_fn(v) -> ndarray`` compatible with the
    dynamic integrators::

        a_total = stress_acceleration(v, dim, mu, HC, pressure_model) + g_vec

    Parameters
    ----------
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex
        Simplicial complex with duals computed.
    g : float
        Gravitational acceleration magnitude [m/s^2].
    gravity_axis : int or None
        Axis along which gravity acts (default: dim-1).
    pressure_model : None, callable, or EquationOfState
        Pressure source for :func:`stress_acceleration`.  See
        :func:`ddgclib.operators.stress.stress_force` for details.

    Returns
    -------
    callable
        ``dudt_fn(v) -> ndarray`` of shape ``(dim,)``.
    """
    if gravity_axis is None:
        gravity_axis = dim - 1

    g_vec = np.zeros(dim)
    g_vec[gravity_axis] = -g

    dudt_stress = partial(
        stress_acceleration, dim=dim, mu=mu, HC=HC,
        pressure_model=pressure_model,
    )

    def dudt_with_gravity(v):
        return dudt_stress(v) + g_vec

    return dudt_with_gravity


# ---------------------------------------------------------------------------
# Extended setup using domain builders and EOS-compatible mass
# ---------------------------------------------------------------------------

def setup_hydrostatic_column(
    dim: int,
    n_refine: int = 2,
    H: float = 10.0,
    rho: float = 1000.0,
    g: float = 9.81,
    P_ref: float = 0.0,
    mu: float = 1e-3,
    gravity_axis: int | None = None,
    pressure_model=None,
    free_surface: bool = False,
    freeze_walls: str = 'all',
) -> tuple:
    """Set up a hydrostatic column with domain builders and gravity dudt.

    Uses :func:`~ddgclib.geometry.domains.rectangle` for 2D,
    :func:`~ddgclib.geometry.domains.box` for 3D, and a raw
    :class:`~hyperct.Complex` for 1D.  Mass is set via
    :class:`~ddgclib.initial_conditions.DualVolumeMass` for exact
    density consistency with the EOS.

    Parameters
    ----------
    dim : int
        Spatial dimension (1, 2, or 3).
    n_refine : int
        Number of mesh refinement passes.
    H : float
        Column height [m].
    rho : float
        Fluid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    P_ref : float
        Reference pressure at the free surface (x[axis]=H).
        Use 101325.0 for absolute pressure (atmospheric).
    mu : float
        Dynamic viscosity [Pa.s].
    gravity_axis : int or None
        Axis along which gravity acts (default: dim-1).
    free_surface : bool
        If True, the top face (x[gravity_axis] = H) is a free surface:
        vertices there are NOT frozen and advect freely.  Their pressure
        is determined by the EOS from density.  Default False (all walls
        frozen).
    freeze_walls : str
        Controls which walls are frozen (no-slip):
        - ``'all'`` (default): bottom + sides (+ top unless free_surface)
        - ``'bottom_only'``: only the bottom face is frozen; sides are
          free (eliminates wall meniscus effects, quasi-periodic).

    Returns
    -------
    HC : Complex
        Simplicial complex with duals computed.
    bV : set
        Boundary vertex set (frozen vertices only).
    bc_set : BoundaryConditionSet
        Boundary conditions (no-slip walls on frozen boundaries).
    params : dict
        Physical parameters for reference.
    dudt_fn : callable
        Acceleration function including gravity body force.
    """
    from ddgclib.geometry.domains._boundary_groups import identify_face_groups

    if gravity_axis is None:
        gravity_axis = dim - 1

    # --- Build domain ---
    bottom_only = (freeze_walls == 'bottom_only')
    tol = 1e-14

    if dim == 1:
        HC = Complex(1, domain=[(0.0, H)])
        HC.triangulate()
        for _ in range(n_refine):
            HC.refine_all()
        if free_surface or bottom_only:
            bV = {v for v in HC.V if v.x_a[gravity_axis] <= 0.0 + tol}
        else:
            bV = identify_cube_boundaries(HC, 0.0, H, dim=1)
        volume = H
    elif dim == 2:
        from ddgclib.geometry.domains import rectangle
        result = rectangle(L=1.0, h=H, refinement=n_refine, flow_axis=0)
        HC = result.HC
        groups = identify_face_groups(HC, {
            'bottom': (gravity_axis, 0.0),
            'top': (gravity_axis, H),
            'left': (0, 0.0),
            'right': (0, 1.0),
        })
        if bottom_only:
            bV = groups['bottom']
        elif free_surface:
            bV = groups['bottom'] | groups['left'] | groups['right']
        else:
            bV = result.bV
        volume = result.metadata['volume']
    elif dim == 3:
        from ddgclib.geometry.domains import box
        result = box(Lx=1.0, Ly=1.0, Lz=H, refinement=n_refine, flow_axis=2)
        HC = result.HC
        groups = identify_face_groups(HC, {
            'bottom': (gravity_axis, 0.0),
            'top': (gravity_axis, H),
            'x0': (0, 0.0), 'x1': (0, 1.0),
            'y0': (1, 0.0), 'y1': (1, 1.0),
        })
        if bottom_only:
            bV = groups['bottom']
        elif free_surface:
            bV = (groups['bottom'] | groups['x0'] | groups['x1']
                  | groups['y0'] | groups['y1'])
        else:
            bV = result.bV
        volume = result.metadata['volume']
    else:
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    # --- Tag TOPOLOGICAL boundaries and compute duals ---
    # v.boundary must be True for all topological boundary vertices
    # (required by compute_vd), even if they are not frozen (not in bV).
    # This matters when free_surface=True: top-face vertices are not
    # in bV but are still on the mesh boundary.
    if free_surface or bottom_only:
        # bV may be a subset of topological boundary; tag ALL domain-face
        # vertices for compute_vd (required for correct dual computation)
        all_face_verts = set()
        for v in HC.V:
            coords = v.x_a[:dim]
            for i in range(dim):
                lb_i = 0.0
                ub_i = H if i == gravity_axis else 1.0
                if coords[i] <= lb_i + tol or coords[i] >= ub_i - tol:
                    all_face_verts.add(v)
                    break
        for v in HC.V:
            v.boundary = v in all_face_verts
    else:
        for v in HC.V:
            v.boundary = v in bV
    compute_vd(HC, cdist=1e-10)
    cache_dual_volumes(HC, dim)

    # --- Initial conditions ---
    # 1) Zero velocity
    ZeroVelocity(dim=dim).apply(HC, bV)
    # 2) Hydrostatic pressure: P = P_ref + rho*g*(H - x[axis])
    HydrostaticPressure(
        rho=rho, g=g, axis=gravity_axis, h_ref=H, P_ref=P_ref,
    ).apply(HC, bV)
    # 3) Mass from dual volumes (EOS-consistent)
    DualVolumeMass(rho=rho).apply(HC, bV)

    # --- Boundary conditions ---
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    # --- Gravity-aware dudt ---
    dudt_fn = make_gravity_dudt(
        dim=dim, mu=mu, HC=HC, g=g, gravity_axis=gravity_axis,
        pressure_model=pressure_model,
    )

    params = {
        'dim': dim,
        'rho': rho,
        'g': g,
        'H': H,
        'P_ref': P_ref,
        'mu': mu,
        'gravity_axis': gravity_axis,
        'volume': volume,
        'free_surface': free_surface,
    }

    return HC, bV, bc_set, params, dudt_fn
