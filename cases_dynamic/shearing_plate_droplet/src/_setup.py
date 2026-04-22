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
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties, mass_conserving_merge
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import (
    BoundaryConditionSet, MovingWallBC,
)
from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d
from ddgclib.geometry.periodic import retopologize_periodic
from ddgclib.operators.multiphase_stress import multiphase_dudt_i


_WALL_TOL = 1e-10


def _classify_walls(bV_all, dim, L_x, L_y, L_z=None):
    """Split the outer-box boundary set into plate and periodic groups.

    Parameters
    ----------
    bV_all : set
        All vertices on the outer box faces (from droplet_in_box_*).
    dim : int
    L_x, L_y, L_z : float
        Half-extents of the box (domain is [-L_*, L_*]).

    Returns
    -------
    dict with keys ``top_wall``, ``bottom_wall``, ``periodic_faces``,
    ``corners`` (vertices that lie on both a periodic face AND a plate;
    these are returned in *both* the matching plate group and the
    ``corners`` set so that plate BCs win).
    """
    top, bottom, periodic, corners = set(), set(), set(), set()
    for v in bV_all:
        x = v.x_a
        on_top = abs(x[1] - L_y) < _WALL_TOL
        on_bottom = abs(x[1] - (-L_y)) < _WALL_TOL
        on_x_face = abs(abs(x[0]) - L_x) < _WALL_TOL
        on_z_face = (dim == 3
                     and L_z is not None
                     and abs(abs(x[2]) - L_z) < _WALL_TOL)

        if on_top:
            top.add(v)
            if on_x_face or on_z_face:
                corners.add(v)
        elif on_bottom:
            bottom.add(v)
            if on_x_face or on_z_face:
                corners.add(v)
        elif on_x_face or on_z_face:
            # Pure periodic-face vertex (not on a plate).
            periodic.add(v)

    return {'top_wall': top, 'bottom_wall': bottom,
            'periodic_faces': periodic, 'corners': corners}


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
):
    """Build the shearing-plate droplet problem.

    Returns
    -------
    HC : Complex
    bV : set
        Frozen wall vertices (top + bottom plates only).
    mps : MultiphaseSystem
    bc_set : BoundaryConditionSet
        With MovingWallBC on top (+U_wall) and bottom (-U_wall).
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
    L_z_used = L_z if dim == 3 else None
    groups = _classify_walls(bV, dim, L_x, L_y, L_z_used)
    periodic_face_verts = groups['periodic_faces']

    # -- Remove periodic-face vertices from bV and untag their boundary
    #    flag.  They become interior after the first periodic retopo.
    bV -= periodic_face_verts
    for v in periodic_face_verts:
        v.boundary = False

    # -- Phase / interface setup --
    # Transfer simplex-phase criterion from the builder so that
    # retriangulation re-labels simplices correctly.
    builder_mps = result.metadata['mps']
    mps.simplex_phase = builder_mps.simplex_phase
    mps._simplex_criterion_fn = builder_mps._simplex_criterion_fn

    # -- Initial conditions --
    # 1. Zero velocity (wall velocities will be imposed by bc_set).
    ZeroVelocity(dim=dim).apply(HC, bV)

    # 2. Multiphase refresh: per-phase fields, split dual volumes, mass, p.
    split_method = 'neighbour_count'
    mps.refresh(HC, dim, reset_mass=True, split_method=split_method)

    # 3. Young-Laplace equilibrium: preload droplet density so that
    #    P_d(rho_d_eq) = P_o(rho_o) + gamma * kappa.
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

    # 4. Recompute pressures from adjusted masses.
    mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    # -- Initial wall velocities (so the first step sees the shear) --
    u_top = np.zeros(dim); u_top[0] = +U_wall
    u_bot = np.zeros(dim); u_bot[0] = -U_wall
    for v in groups['top_wall']:
        v.u = u_top.copy()
    for v in groups['bottom_wall']:
        v.u = u_bot.copy()

    # -- Boundary conditions: moving plates --
    bc_set = BoundaryConditionSet()
    bc_set.add(MovingWallBC(u_top, dim=dim), groups['top_wall'])
    bc_set.add(MovingWallBC(u_bot, dim=dim), groups['bottom_wall'])

    # -- Acceleration function --
    meos = MultiphaseEOS([eos_outer, eos_drop])
    dudt_fn = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )

    # -- Periodic + multiphase retopologize --
    periodic_axes = [0] if dim == 2 else [0, 2]
    domain_bounds = [(-L_x, L_x), (-L_y, L_y)]
    if dim == 3:
        domain_bounds.append((-L_z, L_z))

    retopo_fn = _make_periodic_multiphase_retopo(
        mps=mps, periodic_axes=periodic_axes,
        domain_bounds=domain_bounds, split_method=split_method,
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
):
    """Build a retopologize_fn that combines periodic Delaunay with
    multiphase state refresh.

    Mirrors ``_retopologize_multiphase`` but swaps the non-periodic
    Delaunay step for :func:`retopologize_periodic`.  Accepts
    ``remesh_mode`` / ``remesh_kwargs`` for signature compatibility
    with the integrator — they are currently ignored (periodic adaptive
    remesh is not implemented).
    """
    def _retopo(HC, bV, dim, remesh_mode='delaunay', remesh_kwargs=None):
        retopologize_periodic(
            HC, bV, dim,
            periodic_axes=periodic_axes,
            domain_bounds=domain_bounds,
        )
        if mps is not None:
            mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

    return _retopo
