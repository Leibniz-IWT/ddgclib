"""Setup functions for the dam break test case.

Two families of setups:

* :func:`setup_dam_break_multiphase` — tank filled with liquid + air,
  using the multiphase FVM pipeline with surface tension at the
  (initially rectangular) liquid–air interface.

* :func:`setup_dam_break_single_phase` — only the liquid column as
  the mesh.  Walls on the bottom and the "upstream" side; the top and
  downstream side are free-surface boundaries that advect freely under
  gravity.  The absolute pressure is tracked by the EOS so that the
  free surface relaxes toward the atmospheric reference pressure.

Both build the force (stress + gravity) and the retopology function
from :class:`ddgclib.methods.SolverMethods` (``methods.dudt_fn(...,
body_force=g)`` / ``methods.retopologize_fn``), never by hand (laneW,
2026-10-05).
"""
from __future__ import annotations

import numpy as np

from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.geometry.domains import rectangle, box
from ddgclib.geometry.domains._boundary_groups import identify_face_groups
from ddgclib.methods import SolverMethods
from ddgclib.operators.stress import cache_dual_volumes


# =====================================================================
# Multiphase dam break (liquid + air)
# =====================================================================

def setup_dam_break_multiphase(
    dim: int,
    a: float,
    L: float,
    H: float,
    W: float,
    col_w: float,
    col_h: float,
    col_d: float,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    gamma: float,
    K_l: float,
    K_g: float,
    g: float,
    gravity_axis: int,
    P_atm: float,
    n_refine: int,
    alpha_art: float = 0.0,
    redistribute_mass: bool = True,
    methods=None,
):
    """Build a rectangular tank (2D) or box (3D) filled with two phases.

    Phase 0 = gas (air), phase 1 = liquid (water).  Liquid occupies a
    rectangular column in the corner of the tank.  No-slip walls are
    placed on all exterior faces.  Surface tension acts on the
    liquid–air interface via the multiphase stress pipeline.

    Parameters
    ----------
    redistribute_mass : bool
        If True (default), per-phase mass is redistributed after each
        Delaunay reconnection so that the pre-retopo per-phase pressure
        field is preserved while total per-phase mass is conserved.
        See ``setup_oscillating_droplet`` for the full rationale.
        Ignored when *methods* is given.
    methods : ddgclib.methods.SolverMethods or None
        The solver configuration (``PRESETS['dam_break_2D']`` /
        ``['dam_break_3D']``, normally).  ``split_method`` and
        ``redistribute_mass`` are taken from it, ``dudt_fn`` is
        ``methods.dudt_fn(HC, mps=mps, pressure_model=meos,
        body_force=g_vec)`` (so ``curvature_path`` and
        ``area_orientation`` of the preset are applied; gravity is the
        library's ``body_force`` wrapper, ``dudt_fn.stress_fn`` and
        ``dudt_fn.body_force`` expose the parts) and ``retopo_fn`` is
        ``methods.retopologize_fn(mps=mps)`` (laneW, 2026-10-05).
        ``None`` builds ``SolverMethods(dim, phases='multi',
        redistribute_mass=...)`` from the explicit kwarg (per-step
        Delaunay, no remap, hull-frozen walls: the partial the setup
        used to build by hand, ``ddgclib/tests/test_methods.py``).

    Returns
    -------
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params
    """
    if methods is None:
        methods = SolverMethods(dim=dim, phases='multi',
                                redistribute_mass=redistribute_mass)
    elif methods.dim != dim:
        raise ValueError(f"methods.dim={methods.dim} != dim={dim}")
    split_method = methods.split_method

    # -- Build the tank mesh (single-phase geometry) --
    if dim == 2:
        result = rectangle(
            L=L, h=H, refinement=n_refine, flow_axis=0,
            origin=(0.0, 0.0),
        )
    elif dim == 3:
        # Flow axis = 0 (x), gravity axis = 1 (y), depth axis = 2 (z)
        result = box(
            Lx=L, Ly=H, Lz=W, refinement=n_refine, flow_axis=0,
            origin=(0.0, 0.0, 0.0),
        )
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    HC = result.HC
    bV_walls = result.bV

    # -- Estimate mean edge length for artificial viscosity --
    edges = [
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    ]
    dx_mean = float(np.mean(edges)) if edges else 0.0
    c_s_l = float(np.sqrt(K_l / rho_l))
    mu_art_l = alpha_art * rho_l * c_s_l * dx_mean
    mu_art_g = alpha_art * rho_g * c_s_l * dx_mean   # same c_s scale
    mu_l_eff = mu_l + mu_art_l
    mu_g_eff = mu_g + mu_art_g

    # -- Tag phases by spatial position (lower-left column = liquid) --
    def in_column(x):
        ok = (x[0] <= col_w + 1e-12) and (x[1] <= col_h + 1e-12)
        if dim == 3:
            ok = ok and (x[2] <= col_d + 1e-12)
        return ok

    for v in HC.V:
        v.phase = 1 if in_column(v.x_a[:dim]) else 0
        v.boundary = v in bV_walls

    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)

    # -- Build multiphase system --
    #   Use linear EOS (n=1) so HydrostaticEOSMass has a closed form
    #   and the pressure field is well behaved at low bulk modulus.
    eos_gas = TaitMurnaghan(
        rho0=rho_g, P0=P_atm, K=K_g, n=1.0, rho_clip=(0.2, 5.0),
    )
    eos_liq = TaitMurnaghan(
        rho0=rho_l, P0=P_atm, K=K_l, n=1.0, rho_clip=(0.8, 1.2),
    )
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_gas, mu=mu_g_eff, rho0=rho_g, name="air"),
            PhaseProperties(eos=eos_liq, mu=mu_l_eff, rho0=rho_l, name="water"),
        ],
        gamma={(0, 1): gamma},
    )

    # -- Initial conditions --
    #
    # NOTE(laneF-hydrostatic-ic): the phases are initialised in the
    # HYDROSTATIC state via a per-phase mass preload (the oscillating-
    # droplet Young–Laplace preload pattern), NOT with a flat pressure
    # field.  Under ``redistribute_mass=True`` the per-vertex pressure
    # STRUCTURE is preserved across every step (redistribution rebuilds
    # masses from the pre-step pressure snapshot; only a uniform
    # per-phase offset can evolve — laneD §1.1).  With the previous
    # flat-at-P_atm IC the hydrostatic gradient could therefore NEVER
    # develop: measured after 0.098 s under gravity the liquid column
    # read a uniform +0.121 Pa gauge instead of the ~981 Pa hydrostatic
    # head, the dam face saw ~0 horizontal driving force, and the
    # collapse stalled at |u| ~ 0.02 m/s (laneF log).  The hydrostatic
    # IC puts the collapse-driving pressure structure (liquid head vs
    # atmospheric air across the dam face) into the state that the
    # redistribution preserves.
    #
    # We run the Delaunay retopologisation once here and assign mass
    # AFTER retopology so that density matches the target pressure on
    # the mesh that the integrator will actually see at step 0.  This is
    # a one-shot GEOMETRY rebuild (connectivity, boundary, duals; no
    # multiphase refresh, no remap), deliberately not the preset's
    # per-step retopology, which the integrator runs at step 0 anyway.
    ZeroVelocity(dim=dim).apply(HC, bV_walls)
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    _retopologize(HC, bV_walls, dim)
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)
    # NOTE(laneF-vote-labels, 2026-10-05): the refresh above labels the
    # simplices with the spatial criterion; every refresh the integrator
    # runs labels them by the vertex vote (assign_simplex_phases_from_
    # vertices).  In 3D the two differ (laneS: 9 of 189 vertices change
    # label), so a preload on the criterion sub-volumes is not
    # hydrostatic on the vote sub-volumes: measured p_liq 101203 to
    # 120945 Pa (the EOS clip, +9.6 kPa) at t = 0 instead of P_atm + the
    # head, which blew the 3D column apart (|a| 4e3 m/s^2 at step 0).
    # One vote refresh BEFORE the preload puts the masses on the labels
    # that run.  In 2D the vote reproduces the criterion labels, so the
    # setup state is bit-identical (digest 952d4544676ca366 of the
    # shipped run).
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # Hydrostatic targets (linear EOS n=1 -> exact closed-form density).
    # Gas: atmospheric column over the full tank height.  Liquid:
    # continuous with the gas pressure at the column top y = col_h.
    def _p_gas(y: float) -> float:
        return P_atm + rho_g * g * (H - y)

    def _p_liq(y: float) -> float:
        return P_atm + rho_g * g * (H - col_h) + rho_l * g * (col_h - y)

    _p_target = (_p_gas, _p_liq)
    for v in HC.V:
        y = float(v.x_a[gravity_axis])
        for k in (0, 1):
            vol_k = v.dual_vol_phase[k]
            if vol_k > 1e-30:
                rho_k = float(mps.phases[k].eos.density(_p_target[k](y)))
                v.m_phase[k] = rho_k * vol_k
            else:
                # NOTE(laneF-vote-labels): no mass where the vote
                # sub-volume is zero.  The criterion-label refresh above
                # left liquid mass on 3D vertices the vote gives no
                # liquid, and the volume ledger released it into the
                # pool at step 0 (+8.7 % liquid mass, +8.5 kPa).
                v.m_phase[k] = 0.0
        v.m = float(np.sum(v.m_phase))

    # Final refresh: recompute per-phase pressures from the preloaded
    # masses (reset_mass=False preserves them).
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # -- Boundary conditions: all outer walls no-slip --
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV_walls)

    # -- Acceleration (pressure + viscous + surface tension) + gravity --
    meos = MultiphaseEOS([eos_gas, eos_liq])
    g_vec = np.zeros(dim)
    g_vec[gravity_axis] = -g
    dudt_fn = methods.dudt_fn(HC, mps=mps, pressure_model=meos,
                              body_force=g_vec)
    retopo_fn = methods.retopologize_fn(mps=mps)

    params = {
        'dim': dim,
        'a': a, 'L': L, 'H': H, 'W': W,
        'col_w': col_w, 'col_h': col_h, 'col_d': col_d,
        'rho_l': rho_l, 'rho_g': rho_g,
        'mu_l': mu_l, 'mu_g': mu_g,
        'gamma': gamma, 'K_l': K_l, 'K_g': K_g,
        'g': g, 'gravity_axis': gravity_axis, 'P_atm': P_atm,
        'n_refine': n_refine,
    }

    return HC, bV_walls, mps, bc_set, dudt_fn, retopo_fn, params


# =====================================================================
# Single-phase dam break (liquid only, implicit atmosphere)
# =====================================================================

def setup_dam_break_single_phase(
    dim: int,
    a: float,
    col_w: float,
    col_h: float,
    col_d: float,
    rho_l: float,
    mu_l: float,
    K_l: float,
    g: float,
    gravity_axis: int,
    P_atm: float,
    n_refine: int,
    alpha_art: float = 0.0,
    methods=None,
):
    """Build only the liquid column mesh with a free surface.

    The mesh is the rectangular water column.  ``bV`` (frozen) holds
    **only** the tank walls (bottom + left in 2D, bottom + x0 + z walls
    in 3D).  The top face and the ``x = col_w`` face are free surfaces
    — their vertices are still topological boundary vertices (so the
    dual mesh is well defined) but are NOT frozen, so they advect under
    gravity.

    The liquid pressure is initialised to the absolute hydrostatic
    profile ``P(y) = P_atm + rho_l * g * (col_h - y)``.  At the free
    surface this collapses to ``P = P_atm`` which the EOS tracks
    through the weakly-compressible Tait–Murnaghan relation.

    *methods* (``PRESETS['dam_break_2D_no_air']`` / ``['dam_break_3D_no_air']``,
    normally; ``None`` = ``SolverMethods(dim)``, the single-phase
    default: symplectic Euler, per-step Delaunay, hull-frozen) builds
    ``dudt_fn = methods.dudt_fn(HC, mu=mu_eff, pressure_model=eos,
    body_force=g_vec)``; the runners pass ``boundary_filter`` (the wall
    criterion ``v.is_wall``) to ``methods.integrate``.  The configuration
    is laneK's measured-unstable one (bare Delaunay + EOS, no remap):
    construction of the force warns.

    Returns
    -------
    HC, bV, bc_set, dudt_fn, params
    """
    if methods is None:
        methods = SolverMethods(dim=dim)
    elif methods.dim != dim:
        raise ValueError(f"methods.dim={methods.dim} != dim={dim}")
    if dim == 2:
        result = rectangle(
            L=col_w, h=col_h, refinement=n_refine, flow_axis=0,
            origin=(0.0, 0.0),
        )
        HC = result.HC
        groups = identify_face_groups(HC, {
            'bottom': (1, 0.0),
            'top':    (1, col_h),
            'left':   (0, 0.0),
            'right':  (0, col_w),
        })
        # Frozen walls: bottom + left
        bV_walls = groups['bottom'] | groups['left']
        # Free surface vertices: top + right (not frozen but still boundary)
        free_face = groups['top'] | groups['right']
        volume = col_w * col_h
    elif dim == 3:
        result = box(
            Lx=col_w, Ly=col_h, Lz=col_d, refinement=n_refine,
            flow_axis=0, origin=(0.0, 0.0, 0.0),
        )
        HC = result.HC
        groups = identify_face_groups(HC, {
            'x0': (0, 0.0),      'x1': (0, col_w),
            'y0': (1, 0.0),      'y1': (1, col_h),
            'z0': (2, 0.0),      'z1': (2, col_d),
        })
        # Frozen walls: bottom (y0) + left (x0) + both z faces
        bV_walls = groups['y0'] | groups['x0'] | groups['z0'] | groups['z1']
        # Free surface: top (y1) + right (x1)
        free_face = groups['y1'] | groups['x1']
        volume = col_w * col_h * col_d
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    # Tag ALL topological boundary vertices so compute_vd is consistent,
    # but only ``bV_walls`` are frozen (returned as bV to the integrator).
    # ``v.is_wall`` lets the integrator's ``boundary_filter`` distinguish
    # frozen walls from free-surface boundary vertices.
    all_face_verts = bV_walls | free_face
    for v in HC.V:
        v.boundary = v in all_face_verts
        v.is_wall = v in bV_walls

    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)

    # -- Artificial viscosity: mu_art = alpha * rho * c_s * dx --
    edges = [
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    ]
    dx_mean = float(np.mean(edges)) if edges else 0.0
    c_s_l = float(np.sqrt(K_l / rho_l))
    mu_art = alpha_art * rho_l * c_s_l * dx_mean
    mu_eff = mu_l + mu_art

    # -- EOS (linear Tait–Murnaghan so we have a closed-form hydrostatic) --
    eos_liq = TaitMurnaghan(
        rho0=rho_l, P0=P_atm, K=K_l, n=1.0, rho_clip=(0.5, 2.0),
    )

    # -- Initial conditions --
    #
    # Uniform density ``rho_l`` gives a uniform initial pressure
    # ``EOS(rho_l) = P_atm``.  Gravity is the only force at t=0;
    # the hydrostatic profile develops dynamically.  This avoids the
    # spurious impulse that a hydrostatic IC would create at the
    # truncated-dual corner vertices of the free surface.
    ZeroVelocity(dim=dim).apply(HC, bV_walls)
    for v in HC.V:
        dv = float(getattr(v, 'dual_vol', 0.0))
        if dv < 1e-30:
            v.m = rho_l * 1e-30
        else:
            v.m = rho_l * dv
        v.p = P_atm

    # -- Boundary conditions: only the frozen walls are no-slip --
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV_walls)

    # -- Gravity-augmented single-phase stress acceleration --
    g_vec = np.zeros(dim)
    g_vec[gravity_axis] = -g
    dudt_fn = methods.dudt_fn(HC, mu=mu_eff, pressure_model=eos_liq,
                              body_force=g_vec)

    params = {
        'dim': dim, 'a': a,
        'col_w': col_w, 'col_h': col_h, 'col_d': col_d,
        'rho_l': rho_l, 'mu_l': mu_l, 'K_l': K_l,
        'g': g, 'gravity_axis': gravity_axis, 'P_atm': P_atm,
        'n_refine': n_refine, 'volume': volume,
        'free_face': free_face,
    }

    return HC, bV_walls, bc_set, dudt_fn, params


# =====================================================================
# Shared helpers
# =====================================================================

def cfl_timestep(HC, dim: int, c_s: float, cfl: float = 0.25) -> float:
    """CFL timestep from the minimum edge length of the mesh."""
    dx_min = min(
        (
            np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
        ),
        default=1.0,
    )
    return cfl * dx_min / max(c_s, 1e-12)
