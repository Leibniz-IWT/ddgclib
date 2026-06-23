"""Setup for the shearing-plate droplet test case.

Builds:
- A rectangular (2D) / cuboidal (3D) domain of water containing an
  oil droplet at the centre.
- Wall plates at ``y = +/- L_y`` that slide in opposite directions
  (no-slip moving wall BC).
- Periodic boundary conditions along ``x`` (and ``z`` in 3D).

The mesh is constructed by reusing the library's multiphase droplet
builder (``droplet_in_box_2d`` / ``droplet_in_box_3d``), then
post-processing its boundary set to:

1. Split the outer box ``walls`` group into ``top_wall``,
   ``bottom_wall`` and ``periodic_faces``.
2. Remove periodic-face vertices from ``bV`` and from ``v.boundary``
   so that they are treated as interior after the first periodic
   retriangulation.

The retopologize function composes the periodic retriangulation
(``retopologize_periodic``) with the multiphase state refresh
(``mps.refresh``) — effectively the periodic analogue of
``_retopologize_multiphase``.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import (
    BoundaryConditionSet, ShearingPlateBC,
)
from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d
from ddgclib.geometry.periodic import retopologize_periodic
from ddgclib.operators.multiphase_stress import multiphase_dudt_i


_WALL_TOL = 1e-10


def _classify_box_faces(HC, dim, L_x, L_y, L_z=None):
    """Identify and classify outer-box face vertices by position.

    Scans all vertices in ``HC.V`` (not a pre-built set, because the
    anisotropic rescale above mutates vertex hashes via
    ``HC.V.move`` — any set populated before the rescale is broken).

    Returns
    -------
    dict with keys:
    - ``top_wall`` / ``bottom_wall``: vertices exactly on the plates.
    - ``periodic_faces``: vertices on the periodic side faces *and not*
      on a plate (corners are counted as plates, so the plate BC wins).
    - ``all_walls``: ``top_wall | bottom_wall`` (frozen wall set).
    """
    top, bottom, periodic = set(), set(), set()
    for v in HC.V:
        x = v.x_a
        on_top = abs(x[1] - L_y) < _WALL_TOL
        on_bottom = abs(x[1] - (-L_y)) < _WALL_TOL
        on_x_face = abs(abs(x[0]) - L_x) < _WALL_TOL
        on_z_face = (dim == 3
                     and L_z is not None
                     and abs(abs(x[2]) - L_z) < _WALL_TOL)

        if on_top:
            top.add(v)
        elif on_bottom:
            bottom.add(v)
        elif on_x_face or on_z_face:
            periodic.add(v)

    return {'top_wall': top, 'bottom_wall': bottom,
            'periodic_faces': periodic,
            'all_walls': top | bottom}


def setup_shearing_plate_droplet(
    dim: int = 2,
    R0: float = 0.005,
    L_x: float = 0.015,
    L_y: float = 0.01,
    L_z: float = 0.01,
    U_wall: float = 0.05,
    rho_d: float = 900.0,
    rho_o: float = 1000.0,
    mu_d: float = 0.05,
    mu_o: float = 0.05,
    gamma: float = 0.03,
    K_d: float | None = None,
    K_o: float | None = None,
    refinement_outer: int = 3,
    refinement_droplet: int = 3,
    P0: float = 0.0,
    distr_law: str = "sinusoidal",
    redistribute_mass: bool = True,
):
    """Build the shearing-plate droplet problem.

    Parameters
    ----------
    redistribute_mass : bool
        If True (default), per-phase mass is redistributed after each
        periodic Delaunay reconnection so that the pre-retopo per-phase
        pressure field is preserved while total per-phase mass is
        conserved.  See ``setup_oscillating_droplet`` for the full
        rationale.

    Returns
    -------
    HC : Complex
    bV : set
        Frozen wall vertices (top + bottom plates only).
    mps : MultiphaseSystem
    bc_set : BoundaryConditionSet
        With ShearingPlateBC on top (+U_wall) and bottom (-U_wall).
    dudt_fn : callable
        Bound multiphase acceleration function for integrators.
    retopo_fn : callable
        Periodic multiphase retopologize fn (drop-in for
        ``retopologize_fn`` in the dynamic integrators).
    groups : dict
        ``{'top_wall', 'bottom_wall', 'periodic_faces', 'corners',
        'interface'}`` vertex sets for diagnostics / BCs.
    params : dict
        Echo of all input parameters plus ``periodic_axes``,
        ``domain_bounds``, ``shear_rate``.
    """
    # -- Weakly-compressible sound speed default --
    if K_d is None or K_o is None:
        u_scale = max(U_wall, np.sqrt(gamma / (rho_o * R0)))
        c_s = max(10.0 * u_scale, 1.0)
        if K_d is None:
            K_d = rho_d * c_s ** 2
        if K_o is None:
            K_o = rho_o * c_s ** 2

    # -- Multiphase EOS / properties --
    eos_outer = TaitMurnaghan(rho0=rho_o, P0=P0, K=K_o, n=7.15,
                               rho_clip=(0.8, 1.2))
    eos_drop = TaitMurnaghan(rho0=rho_d, P0=P0, K=K_d, n=7.15,
                              rho_clip=(0.8, 1.2))
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_outer, mu=mu_o, rho0=rho_o, name="outer"),
            PhaseProperties(eos=eos_drop, mu=mu_d, rho0=rho_d, name="droplet"),
        ],
        gamma={(0, 1): gamma},
    )

    # -- Build multiphase mesh (droplet-in-box) --
    # The builder assumes a symmetric cube of half-extent ``L``.  We
    # invoke it with ``L = max(L_x, L_y, L_z)`` and then rescale the
    # outer vertices anisotropically so the final box matches the
    # requested rectangular extents.  Droplet-interior vertices are
    # left untouched so the droplet stays circular.
    L_build = max(L_x, L_y, L_z if dim == 3 else 0.0)
    if dim == 2:
        result = droplet_in_box_2d(
            R=R0, L=L_build,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )
        scale = np.array([L_x / L_build, L_y / L_build])
    elif dim == 3:
        result = droplet_in_box_3d(
            R=R0, L=L_build,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )
        scale = np.array([L_x / L_build, L_y / L_build, L_z / L_build])
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    HC = result.HC
    bV = result.bV
    # Rescale OUTER (non-droplet) vertices anisotropically.
    # Droplet vertices keep their radial spacing; the outer shell
    # ring/shell built just outside R is treated as outer (phase 0).
    if not np.allclose(scale, 1.0):
        moved = []
        for v in list(HC.V):
            r = float(np.linalg.norm(v.x_a[:dim]))
            if r > R0 + 1e-12:
                new_pos = v.x_a.copy()
                new_pos[:dim] = v.x_a[:dim] * scale
                moved.append((v, tuple(new_pos)))
        for v, new_pos in moved:
            HC.V.move(v, new_pos)

    # -- Classify walls into plates + periodic faces --
    # Must be done *after* the rescale because HC.V.move mutates the
    # coordinate-based vertex hash, which corrupts any set populated
    # before the move.  Scanning HC.V fresh avoids the issue.
    L_z_used = L_z if dim == 3 else None
    groups = _classify_box_faces(HC, dim, L_x, L_y, L_z_used)

    # Rebuild bV from the freshly-classified plate vertices.  Periodic
    # face vertices become interior (they will be handled by the
    # periodic retopologization).
    bV.clear()
    bV.update(groups['all_walls'])
    for v in HC.V:
        v.boundary = v in bV

    # -- Phase / interface setup --
    # Transfer simplex-phase criterion from the builder so that
    # retriangulation re-labels simplices correctly.
    builder_mps = result.metadata['mps']
    mps.simplex_phase = builder_mps.simplex_phase
    mps._simplex_criterion_fn = builder_mps._simplex_criterion_fn

    # -- Initial conditions --
    # 1. Zero velocity (wall velocities are imposed by bc_set below).
    ZeroVelocity(dim=dim).apply(HC, bV)

    # 2. Multiphase refresh on the non-periodic mesh so that
    #    v.dual_vol_phase / v.m_phase exist before the first periodic
    #    retopo consumes them.
    split_method = 'neighbour_count'
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    # -- Periodic + multiphase retopologize (build the closure now so
    # we can apply it once before the Young-Laplace preload).
    periodic_axes = [0] if dim == 2 else [0, 2]
    domain_bounds = [(-L_x, L_x), (-L_y, L_y)]
    if dim == 3:
        domain_bounds.append((-L_z, L_z))

    retopo_fn = _make_periodic_multiphase_retopo(
        mps=mps, periodic_axes=periodic_axes,
        domain_bounds=domain_bounds, split_method=split_method,
        redistribute_mass=redistribute_mass,
    )

    # 3. Apply the periodic retopologize ONCE so the dual volumes,
    #    per-phase splits and bV reflect the final periodic topology.
    #    Running the Young-Laplace preload on the pre-periodic mesh
    #    leaves a tiny residual pressure imbalance because the
    #    first-step retopo changes dual_vol_phase (ub-face vertices
    #    get merged into lb-face, shifting phase volumes).
    retopo_fn(HC, bV, dim)

    # 4. Re-identify plate vertices *after* the retopo (positions may
    #    have drifted by floating-point epsilon under the periodic
    #    wrap, and the ub-face merge has rehashed vertices).
    groups = _classify_box_faces(HC, dim, L_x, L_y, L_z_used)
    bV.clear()
    bV.update(groups['all_walls'])
    for v in HC.V:
        v.boundary = v in bV

    # 5. Young-Laplace equilibrium preload on the periodic-final duals.
    #    Setting rho_d = eos_drop.density(p_outer + gamma*kappa) makes
    #    the droplet-phase pressure equal (p_outer + gamma*kappa), so
    #    the interface pressure jump exactly balances the surface
    #    tension force F_st = gamma*kappa*n*dA at t=0.
    curvature = (dim - 1) / R0   # kappa = 1/R (2D), 2/R (3D)
    gamma_val = mps.get_gamma_pair(0, 1)
    delta_p = gamma_val * curvature
    p_outer = float(eos_outer.pressure(rho_o))
    rho_d_eq = float(eos_drop.density(p_outer + delta_p))
    for v in HC.V:
        vol_d = v.dual_vol_phase[1]
        if vol_d > 1e-30:
            v.m_phase[1] = rho_d_eq * vol_d
            v.m = float(np.sum(v.m_phase))

    # 6. Recompute pressures from adjusted masses (no mass reset).
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # -- Initial wall velocities (so the first step sees the shear) --
    u_top = np.zeros(dim); u_top[0] = +U_wall
    u_bot = np.zeros(dim); u_bot[0] = -U_wall
    for v in groups['top_wall']:
        v.u = u_top.copy()
    for v in groups['bottom_wall']:
        v.u = u_bot.copy()

    # -- Boundary conditions: shearing plates that physically slide.
    #    Wall vertices stay at y = ±L_y (clamped each step by the BC)
    #    but translate in x at ±U_wall; the x-wrap sends them back to
    #    the other side of the periodic domain when they cross ±L_x.
    bc_set = BoundaryConditionSet()
    x_wrap = [(0, (-L_x, L_x))]
    bc_set.add(
        ShearingPlateBC(u_top, plate_axis=1, plate_coord=+L_y,
                        wrap_axes=x_wrap, dim=dim),
        groups['top_wall'],
    )
    bc_set.add(
        ShearingPlateBC(u_bot, plate_axis=1, plate_coord=-L_y,
                        wrap_axes=x_wrap, dim=dim),
        groups['bottom_wall'],
    )

    # -- Acceleration function --
    meos = MultiphaseEOS([eos_outer, eos_drop])
    dudt_fn = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )

    params = {
        'dim': dim, 'R0': R0,
        'L_x': L_x, 'L_y': L_y, 'L_z': L_z,
        'U_wall': U_wall, 'shear_rate': U_wall / L_y,
        'rho_d': rho_d, 'rho_o': rho_o, 'mu_d': mu_d, 'mu_o': mu_o,
        'gamma': gamma, 'K_d': K_d, 'K_o': K_o,
        'refinement_outer': refinement_outer,
        'refinement_droplet': refinement_droplet,
        'P0': P0, 'distr_law': distr_law,
        'periodic_axes': periodic_axes,
        'domain_bounds': domain_bounds,
        'remesh_mode': 'delaunay',
        'remesh_kwargs': None,
    }

    groups['interface'] = result.boundary_groups.get('interface', set())

    return HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params


def _make_periodic_multiphase_retopo(
    mps, periodic_axes, domain_bounds, split_method='neighbour_count',
    redistribute_mass=False,
):
    """Build a retopologize_fn that combines periodic Delaunay with
    multiphase state refresh.

    Mirrors ``_retopologize_multiphase`` but swaps the non-periodic
    Delaunay step for :func:`retopologize_periodic`.  Accepts
    ``remesh_mode`` / ``remesh_kwargs`` for signature compatibility
    with the integrator — they are currently ignored (periodic adaptive
    remesh is not implemented).

    When ``redistribute_mass`` is True the pre-retopo per-phase
    ``dual_vol_phase`` is snapshotted and used as the gating mask in
    ``redistribute_mass_multiphase`` so that per-phase pressure is
    preserved across reconnection (mirrors the geometry-aware path in
    ``_retopologize_multiphase``).
    """
    def _retopo(HC, bV, dim, remesh_mode='delaunay', remesh_kwargs=None):
        _p_snap = None
        if redistribute_mass and mps is not None:
            from ddgclib.operators.mass_redistribution import (
                snapshot_geometry_multiphase,
            )
            _p_snap = snapshot_geometry_multiphase(HC, mps.n_phases)

        retopologize_periodic(
            HC, bV, dim,
            periodic_axes=periodic_axes,
            domain_bounds=domain_bounds,
        )
        if mps is not None:
            mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

            if redistribute_mass and _p_snap is not None:
                from ddgclib.operators.mass_redistribution import (
                    redistribute_mass_multiphase,
                )
                redistribute_mass_multiphase(
                    HC, dim, mps, bV=bV, pressure_snapshot=_p_snap,
                )
                mps.compute_phase_pressures(HC)

    return _retopo
