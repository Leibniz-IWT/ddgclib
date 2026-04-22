"""Setup routine for the electrolysis-bubble dynamic case.

Proof-of-concept geometry
-------------------------
A hydrogen gas bubble immersed in a rectangular / cubic column of
electrolyte, with gravity acting downward.  The domain's bottom wall
is the "electrode" where the bubble is initially positioned.  As gas
mass is injected (placeholder for electrochemical generation), the
bubble grows and buoyancy drives it upward, demonstrating the
fundamental detachment mechanism (buoyancy overcoming gravity on the
gas column).

The mesh is the same closed-loop, fully-immersed multiphase droplet
as in ``cases_dynamic/oscillating_droplet`` -- the shape is a full
circle/sphere (NOT truncated) sitting near the bottom of the box --
because the closed-loop interface is the only configuration that the
multiphase surface-tension operator handles stably in 2D and 3D.  A
truncated-cap mesh with a 3-phase contact line breaks the closed-
curve assumption of ``_select_curve_neighbours`` and causes the
interface to collapse / expand unboundedly within a few hundred
steps.  We leave proper contact-angle physics for a future iteration
(requires a dedicated contact-line BC and a Young-Laplace shape
initialiser).

Hydrostatic pressure is pre-imposed in the liquid (avoiding the
startup pressure wave), gas pressure is stratified to roughly match
(so no fast-scale mismatch at the interface), and the retopology is
the standard ``_retopologize_multiphase`` (Delaunay + multiphase
re-labeling), because dual-only retopology on this problem is
markedly less stable than the full Delaunay pass (contrary to the
static_droplet_2D case, which has no gravity).
"""
from __future__ import annotations

from functools import partial

import numpy as np

from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import (
    MultiphaseSystem, PhaseProperties, mass_conserving_merge,
)
from ddgclib.initial_conditions import ZeroVelocity
from ddgclib._boundary_conditions import (
    BoundaryConditionSet, NoSlipWallBC, BoundaryCondition,
)
from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d
from ddgclib.geometry.domains._disks import disk
from ddgclib.geometry.domains._rectangles import rectangle
from ddgclib.geometry.domains._multiphase_droplet import (
    _build_combined_mesh,
    _estimate_edge_length,
)
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)
from hyperct.ddg import compute_vd


# =====================================================================
# Wall-clamp BC
# =====================================================================

class WallClampBC(BoundaryCondition):
    """Prevent interior vertices from crossing a flat wall plane.

    Interior (non-``bV``) vertices advected past a wall produce a
    bogus dual polygon on the other side, breaking mass / pressure
    conservation.  This BC clips them back to ``level + min_gap`` and
    zeroes the wall-normal velocity component.
    """

    def __init__(self, axis: int, level: float, direction: int = +1,
                 min_gap: float = 0.0, exclude: set | None = None):
        """
        Parameters
        ----------
        direction : +1 or -1
            ``+1`` = wall is a lower bound (clip vertices with
            ``x_a[axis] < level + min_gap``).  ``-1`` = wall is an
            upper bound (clip vertices with ``x_a[axis] > level -
            min_gap``).
        """
        super().__init__(axis=axis)
        self.level = float(level)
        self.direction = int(direction)
        self.min_gap = float(min_gap)
        self.exclude = exclude if exclude is not None else set()

    def apply(self, mesh, dt, target_vertices=None):
        axis = self.axis
        dirn = self.direction
        count = 0
        for v in list(mesh.V):
            if v in self.exclude:
                continue
            y = v.x_a[axis]
            if dirn > 0 and y < self.level + self.min_gap:
                pos = v.x_a.copy()
                pos[axis] = self.level + self.min_gap
                mesh.V.move(v, tuple(pos))
                u = v.u.copy()
                if u[axis] < 0.0:
                    u[axis] = 0.0
                v.u = u
                count += 1
            elif dirn < 0 and y > self.level - self.min_gap:
                pos = v.x_a.copy()
                pos[axis] = self.level - self.min_gap
                mesh.V.move(v, tuple(pos))
                u = v.u.copy()
                if u[axis] > 0.0:
                    u[axis] = 0.0
                v.u = u
                count += 1
        return count


# =====================================================================
# Mesh builder: full circular/spherical bubble in a fixed box
# =====================================================================

def _build_offcenter_bubble_box_2d(
    R_bubble: float,
    L_domain: float,
    bubble_center: np.ndarray,
    refinement_outer: int,
    refinement_droplet: int,
    distr_law: str = "sinusoidal",
):
    """Build a 2D off-centre bubble-in-box mesh.

    The outer box stays at ``[-L_domain, L_domain]^2`` regardless of
    where the bubble is placed.  Returns ``(HC, bV_walls)``.
    """
    import math

    center_arr = np.asarray(bubble_center, dtype=float)
    cx, cy = float(center_arr[0]), float(center_arr[1])
    dim = 2

    # Outer box spanning [-L, L]^2.
    outer_res = rectangle(
        L=2 * L_domain, h=2 * L_domain,
        refinement=refinement_outer, flow_axis=0,
    )
    HC_outer = outer_res.HC
    for v in list(HC_outer.V):
        pos = v.x_a.copy()
        pos[0] -= L_domain
        pos[1] -= L_domain
        HC_outer.V.move(v, tuple(pos))

    # Bubble disk centred at the desired offset.
    drop_res = disk(
        R=R_bubble, center=(cx, cy),
        refinement=refinement_droplet, distr_law=distr_law,
    )
    HC_drop = drop_res.HC

    # Collect combined positions.
    positions = []
    phases = []

    for v in HC_outer.V:
        pos = np.round(v.x_a[:dim], decimals=10)
        d = float(np.linalg.norm(pos - center_arr))
        if d > R_bubble:
            positions.append(pos)
            phases.append(0)

    for v in HC_drop.V:
        positions.append(np.round(v.x_a[:dim], decimals=10))
        phases.append(1)

    # Ring of phase-0 vertices just outside the bubble.
    boundary_verts = [
        v for v in HC_drop.V
        if abs(np.linalg.norm(v.x_a[:dim] - center_arr) - R_bubble)
        < R_bubble * 0.01
    ]
    h_drop = _estimate_edge_length(HC_drop, dim)
    ring_R = R_bubble + h_drop
    n_ring = max(len(boundary_verts), 16)
    for i in range(n_ring):
        theta = 2.0 * math.pi * i / n_ring
        pos_r = np.round(np.array([
            cx + ring_R * math.cos(theta),
            cy + ring_R * math.sin(theta),
        ]), decimals=10)
        if (abs(pos_r[0]) < L_domain - 1e-10
                and abs(pos_r[1]) < L_domain - 1e-10):
            positions.append(pos_r)
            phases.append(0)

    HC = _build_combined_mesh(positions, phases, dim)

    # Identify walls.
    dV = HC.boundary()
    tol = 1e-10
    bV_walls = set()
    for v in dV:
        if any(abs(abs(v.x_a[k]) - L_domain) < tol for k in range(dim)):
            bV_walls.add(v)

    for v in HC.V:
        v.boundary = v in dV

    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)

    return HC, bV_walls


# =====================================================================
# Setup
# =====================================================================

def setup_electrolysis_bubble(
    dim: int = 2,
    R0: float = 1.0e-3,
    L_domain: float = 4.0e-3,
    rho_liq: float = 1000.0,
    rho_gas: float = 10.0,
    mu_liq: float = 0.1,
    mu_gas: float = 0.01,
    gamma: float = 0.072,
    K_liq: float = 1.0e5,
    K_gas: float = 1.0e5,
    g: float = 9.81,
    P0: float = 0.0,
    refinement_outer: int = 2,
    refinement_droplet: int = 3,
    distr_law: str = "sinusoidal",
    apply_hydrostatic_ic: bool = True,
    use_wall_clamp: bool = True,
    nucleation_frac: float = 0.5,
):
    """Build a bubble-on-electrode dynamic multiphase problem.

    Parameters
    ----------
    dim : int
        2 or 3.
    R0 : float
        Initial bubble radius [m].
    L_domain : float
        Half-side of the outer box [m].  The domain is
        ``[-L, L]^dim``.  Both bubble and outer box are centred at
        the origin; the bubble sits near the centre and gravity
        drives it upward.  ``nucleation_frac`` controls how far the
        bubble starts below the domain centre.
    nucleation_frac : float
        Fraction of ``(L_domain - 2*R0)`` to offset the bubble
        downward from the domain centre.  ``0.0`` = bubble at centre,
        ``1.0`` = bubble as low as possible without touching the
        bottom wall.  Default 0.5 places the bubble in the lower half
        of the domain, simulating a freshly-detached nucleation site.
    rho_liq, rho_gas, mu_liq, mu_gas, K_liq, K_gas, gamma, g, P0
        Physical parameters.  See ``src/_params.py`` for defaults.
        ``mu_liq`` / ``mu_gas`` are raised above the real
        water/hydrogen values (see ``_params.py`` comment) for
        numerical stability at this mesh resolution.
    apply_hydrostatic_ic : bool
        Pre-impose hydrostatic pressure in the liquid at t=0 so
        there is no startup pressure wave.  Strongly recommended.
    use_wall_clamp : bool
        Add clamps on the TOP (destination wall) and BOTTOM
        (electrode) that prevent interior vertices from passing
        through walls.

    Returns
    -------
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params
    """
    axis = dim - 1
    floor = -L_domain
    ceiling = +L_domain

    # Bubble centre: offset below the domain centre so the bubble
    # starts near the electrode and has room to rise.
    max_offset = L_domain - 2.0 * R0
    if max_offset <= 0:
        raise ValueError(
            f"L_domain ({L_domain:.3e}) too small for R0 ({R0:.3e}); "
            f"need L_domain > 2*R0."
        )
    y_center = -nucleation_frac * max_offset

    # -- Multiphase system --------------------------------------------------
    eos_liq = TaitMurnaghan(
        rho0=rho_liq, P0=P0, K=K_liq, n=1.0, rho_clip=(0.5, 2.0),
    )
    eos_gas = TaitMurnaghan(
        rho0=rho_gas, P0=P0, K=K_gas, n=1.0, rho_clip=(0.5, 2.0),
    )
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_liq, mu=mu_liq, rho0=rho_liq,
                            name="liquid"),
            PhaseProperties(eos=eos_gas, mu=mu_gas, rho0=rho_gas,
                            name="gas"),
        ],
        gamma={(0, 1): gamma},
    )

    # -- Build the mesh --
    # In 2D we use a custom off-centre builder so the bubble sits
    # near the electrode while the outer box stays fixed at
    # ``[-L, L]^2``.  In 3D we use the canonical ``droplet_in_box_3d``
    # which co-shifts the outer box with the bubble: equivalent to
    # placing the bubble at the centre of a box that happens to be
    # ``2 L_domain`` on a side, so the "bottom" wall (electrode) is
    # always ``L_domain`` below the bubble.  A 3D off-centre builder
    # is on the roadmap but has proven numerically fragile in the
    # Delaunay / DEC dual-mesh pipeline.
    if dim == 2:
        bubble_center = np.zeros(dim)
        bubble_center[axis] = y_center
        wall_bottom = -L_domain
        wall_top = +L_domain

        HC, bV = _build_offcenter_bubble_box_2d(
            R_bubble=R0, L_domain=L_domain,
            bubble_center=bubble_center,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )

        R2 = R0 * R0

        def _phase_criterion(centroid):
            d = np.asarray(centroid[:dim]) - bubble_center
            return 1 if float(d @ d) < R2 else 0

        mps.simplex_phase = {}
        mps.assign_simplex_phases(HC, dim, criterion_fn=_phase_criterion)
        mps._simplex_criterion_fn = _phase_criterion
    else:
        # 3D: canonical builder, bubble at centre, box co-shifted.
        builder_res = droplet_in_box_3d(
            R=R0, L=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            distr_law=distr_law,
        )
        HC = builder_res.HC
        bV = builder_res.bV
        bubble_center = np.zeros(dim)  # origin
        wall_bottom = -L_domain
        wall_top = +L_domain
        # y_center used downstream points to the origin (bubble is
        # centred in the domain in 3D).
        y_center = 0.0

        builder_mps = builder_res.metadata['mps']
        mps.simplex_phase = builder_mps.simplex_phase
        mps._simplex_criterion_fn = builder_mps._simplex_criterion_fn

    # -- Initial conditions --
    ZeroVelocity(dim=dim).apply(HC, bV)
    mps.refresh(HC, dim, reset_mass=True, split_method='neighbour_count')

    # Some 3D wall-corner vertices can end up with NaN dual volumes
    # (the barycentric-dual construction is degenerate on domain
    # corners in 3D).  Replace NaN / infinite volumes and masses with
    # safe defaults before any IC pass touches them.
    for v in HC.V:
        if not np.all(np.isfinite(v.dual_vol_phase)):
            v.dual_vol_phase = np.nan_to_num(v.dual_vol_phase, nan=0.0,
                                             posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(v.m_phase)):
            v.m_phase = np.nan_to_num(v.m_phase, nan=0.0,
                                      posinf=0.0, neginf=0.0)
        if not np.isfinite(getattr(v, 'dual_vol', 0.0)):
            v.dual_vol = 0.0
        if not np.isfinite(v.m):
            v.m = float(np.sum(v.m_phase))

    # Hydrostatic liquid.  P(y) = P0 + rho_liq*g*(wall_top - y).
    if apply_hydrostatic_ic:
        for v in HC.V:
            vol_l = float(v.dual_vol_phase[0])
            if vol_l <= 1e-30 or not np.isfinite(vol_l):
                continue
            y = float(v.x_a[axis])
            P_target = P0 + rho_liq * g * (wall_top - y)
            rho_target = float(eos_liq.density(P_target))
            v.m_phase[0] = rho_target * vol_l
            v.m = float(np.sum(v.m_phase))

    # Gas: Young-Laplace pre-load at the bubble centre, stratified
    # downward by its own gas hydrostatic (small correction).
    curvature = (dim - 1) / R0
    P_liq_top_of_bubble = (
        P0 + rho_liq * g * (wall_top - (y_center + R0))
        if apply_hydrostatic_ic else P0
    )
    P_gas_top = P_liq_top_of_bubble + gamma * curvature
    for v in HC.V:
        vol_g = float(v.dual_vol_phase[1])
        if vol_g <= 1e-30 or not np.isfinite(vol_g):
            continue
        y = float(v.x_a[axis])
        P_target_gas = P_gas_top + rho_gas * g * ((y_center + R0) - y)
        rho_target = float(eos_gas.density(P_target_gas))
        v.m_phase[1] = rho_target * vol_g
        v.m = float(np.sum(v.m_phase))

    # Final NaN sweep AFTER the two IC passes (in case
    # ``compute_phase_pressures`` inside the second refresh produced
    # fresh NaNs for degenerate corner vertices).
    for v in HC.V:
        if not np.isfinite(v.m):
            v.m = 0.0
            v.m_phase = np.zeros_like(v.m_phase)
        if not np.all(np.isfinite(v.u)):
            v.u = np.zeros_like(v.u)

    rho_gas_eq = float(
        eos_gas.density(
            (P0 + rho_liq * g * (wall_top - y_center) if apply_hydrostatic_ic else P0)
            + gamma * curvature
        )
    )

    mass_conserving_merge(HC, cdist=1e-12)
    mps.refresh(HC, dim, reset_mass=False, split_method='neighbour_count')

    # -- Boundary conditions --
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)

    if use_wall_clamp:
        bc_set.add(
            WallClampBC(axis=axis, level=wall_bottom, direction=+1,
                        min_gap=0.02 * R0, exclude=bV),
            None,
        )
        bc_set.add(
            WallClampBC(axis=axis, level=wall_top, direction=-1,
                        min_gap=0.02 * R0, exclude=bV),
            None,
        )

    # -- Acceleration: stress + gravity --
    meos = MultiphaseEOS([eos_liq, eos_gas])
    _base_dudt = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )
    gravity_vec = np.zeros(dim)
    gravity_vec[axis] = -g

    def dudt_fn(v, **_kw):
        # Skip vertices with degenerate mass (3D domain corners can
        # have NaN dual volume in the barycentric construction).
        if not np.isfinite(v.m) or v.m < 1e-30:
            return np.zeros(dim)
        a = _base_dudt(v)
        if not np.all(np.isfinite(a)):
            return np.zeros(dim)
        return a + gravity_vec

    retopo_fn = partial(
        _retopologize_multiphase, mps=mps, split_method='neighbour_count',
    )

    params = {
        'dim': dim,
        'R0': R0,
        'L_domain': L_domain,
        'nucleation_frac': nucleation_frac,
        'bubble_center': tuple(bubble_center),
        'wall_bottom': wall_bottom,
        'wall_top': wall_top,
        'electrode_level': wall_bottom,
        'gravity_axis': axis,
        'gravity_vec': gravity_vec,
        'rho_liq': rho_liq, 'rho_gas': rho_gas,
        'mu_liq': mu_liq, 'mu_gas': mu_gas,
        'gamma': gamma,
        'K_liq': K_liq, 'K_gas': K_gas,
        'g': g, 'P0': P0,
        'eos_liq': eos_liq,
        'eos_gas': eos_gas,
        'meos': meos,
        'rho_gas_eq': rho_gas_eq,
        'remesh_mode': 'delaunay',
        'remesh_kwargs': None,
    }
    return HC, bV, mps, bc_set, dudt_fn, retopo_fn, params
