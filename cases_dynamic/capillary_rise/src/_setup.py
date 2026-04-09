"""Capillary rise case: setup function for 2D and 3D.

Builds a tall liquid column inside a tube (3D cylinder or 2D slit channel)
that extends well below the reservoir water line. The domain represents the
fluid already inside the tube:

    - Walls: no-slip (tube surface)
    - Bottom: periodic inlet (reservoir supplies fluid from below)
    - Top: free surface (meniscus, free to advect)

Capillary driving: a body-force model approximating the contact-line force
at the meniscus.  This is a placeholder for the full dynamic contact angle
model to be implemented later.

    >>> # PLACEHOLDER: replace with resolved contact-line force model
    >>> a_cap = P_cap / (rho * h)   # Washburn-equivalent body force

Pressure is set via compressible hydrostatic EOS (TaitMurnaghan) following
the hydrostatic column recipe — NOT point-wise P(x), but EOS-consistent
mass so that eos.pressure(rho_local) returns the correct hydrostatic profile.

References
----------
Lunowa et al. (2022). Dynamic effects during capillary rise.
    Langmuir, 38(5), 1748-1758.
"""
from __future__ import annotations

from functools import partial

import numpy as np
from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan
from ddgclib.initial_conditions import HydrostaticEOSMass, ZeroVelocity
from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    PositionalNoSlipWallBC,
    PeriodicInletBC,
)
from ddgclib.operators.stress import cache_dual_volumes, stress_acceleration
from ddgclib.geometry.domains._boundary_groups import identify_face_groups

from ._params import capillary_pressure, jurin_height


# ---------------------------------------------------------------------------
# Gravity + capillary dudt wrapper
# ---------------------------------------------------------------------------

def make_capillary_dudt(
    dim: int,
    mu: float,
    HC,
    g_val: float,
    gravity_axis: int,
    P_cap: float,
    rho: float,
    h_ref_fn=None,
    pressure_model=None,
):
    """Create dudt_fn: stress + gravity + capillary body force.

    The capillary driving is modelled as a uniform body force on the
    column, matching the Washburn equation:

        a_cap = P_cap / (rho * h)   [m/s^2, upward]

    where *h* is the current column height tracked via ``h_ref_fn()``.

    .. note::
       PLACEHOLDER — replace with resolved contact-line / dynamic contact
       angle force model.  The body-force approximation is exact only in
       the fully-developed Poiseuille limit.
    """
    g_vec = np.zeros(dim)
    g_vec[gravity_axis] = -g_val

    dudt_stress = partial(
        stress_acceleration, dim=dim, mu=mu, HC=HC,
        pressure_model=pressure_model,
    )

    def dudt_fn(v):
        a = dudt_stress(v) + g_vec
        # PLACEHOLDER: Washburn-equivalent capillary body force (upward)
        if h_ref_fn is not None:
            h = max(h_ref_fn(), 1e-10)
        else:
            h = 1.0
        a[gravity_axis] += P_cap / (rho * h)
        return a

    return dudt_fn


# ---------------------------------------------------------------------------
# Inlet unit-mesh builders
# ---------------------------------------------------------------------------

def _build_inlet_unit_2d(r: float, n_refine: int, rho: float,
                         gravity_axis: int, g_val: float,
                         eos: TaitMurnaghan, h_ref: float) -> Complex:
    """Build the 2D periodic inlet unit mesh (one layer at the bottom).

    The unit mesh is a rectangle matching the tube cross-section width,
    with height = width (aspect ratio ~1) to avoid degenerate triangulation.
    """
    from ddgclib.geometry.domains import rectangle
    width = 2 * r
    # Use the same width as height for the unit mesh (period = width)
    result = rectangle(L=width, h=width, refinement=n_refine, flow_axis=0)
    HC_unit = result.HC
    unit_bV = result.bV

    # Compute duals on unit mesh (rectangle() doesn't do this)
    for v in HC_unit.V:
        v.boundary = v in unit_bV
    compute_vd(HC_unit, cdist=1e-10)
    cache_dual_volumes(HC_unit, dim=2)

    # ICs on unit mesh: zero velocity, EOS-consistent mass at reservoir depth
    ZeroVelocity(dim=2).apply(HC_unit, unit_bV)
    HydrostaticEOSMass(
        eos=eos, rho0=rho, g=g_val, gravity_axis=gravity_axis,
        h_ref=h_ref, P_ref=eos.P0,
    ).apply(HC_unit, unit_bV)

    return HC_unit, width  # return period = width


def _build_inlet_unit_3d(r: float, n_refine: int, rho: float,
                         gravity_axis: int, g_val: float,
                         eos: TaitMurnaghan, h_ref: float) -> Complex:
    """Build the 3D periodic inlet unit mesh (one cylinder segment).

    The unit mesh is a short cylinder of length = 2*R (approximately
    isotropic cells) to avoid extreme aspect ratios.
    """
    from ddgclib.geometry.domains import cylinder_volume

    seg_len = 2 * r  # keep cells roughly isotropic
    result = cylinder_volume(R=r, L=seg_len, refinement=n_refine,
                             flow_axis=gravity_axis)
    HC_unit = result.HC
    unit_bV = result.bV

    # Compute duals on unit mesh
    for v in HC_unit.V:
        v.boundary = v in unit_bV
    compute_vd(HC_unit, cdist=1e-10)
    cache_dual_volumes(HC_unit, dim=3)

    ZeroVelocity(dim=3).apply(HC_unit, unit_bV)
    HydrostaticEOSMass(
        eos=eos, rho0=rho, g=g_val, gravity_axis=gravity_axis,
        h_ref=h_ref, P_ref=eos.P0,
    ).apply(HC_unit, unit_bV)

    return HC_unit, seg_len  # return period = seg_len


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup_capillary_rise(
    dim: int = 2,
    r: float = 0.5e-3,
    H: float | None = None,
    h_init: float | None = None,
    gamma: float = 0.0728,
    theta_deg: float = 9.99,
    mu: float = 0.0011,
    rho: float = 997.0,
    g: float = 9.81,
    n_refine: int = 3,
    artificial_viscosity_alpha: float = 0.1,
) -> tuple:
    """Set up capillary rise problem.

    Builds a tall liquid column (height ``h_init``) inside a tube.
    The bottom is a periodic inlet (reservoir), the top is a free
    surface (meniscus).

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    r : float
        Tube radius (3D) or half-width (2D) [m].
    H : float or None
        Reserved for future use (total tube height including empty part).
    h_init : float or None
        Initial liquid column height [m].  Default: ``0.3 * h_jurin``.
    gamma, theta_deg, mu, rho, g
        Fluid and surface properties (SI).
    n_refine : int
        Number of mesh refinement passes.
    artificial_viscosity_alpha : float
        Scaling factor for artificial viscosity.

    Returns
    -------
    HC, bV, bc_set, dudt_fn, free_surface_verts, params
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    gravity_axis = 1 if dim == 2 else 2

    # ---- Derived quantities ------------------------------------------------
    h_jurin = jurin_height(r, gamma, theta_deg, rho, g, dim=dim)
    P_cap = capillary_pressure(r, gamma, theta_deg, dim=dim)

    if h_init is None:
        h_init = 0.3 * h_jurin

    # ---- EOS (compressible hydrostatic) ------------------------------------
    # Sound speed: ~10x gravity wave speed for weak compressibility
    K_eos = rho * (10.0 * np.sqrt(g * h_jurin)) ** 2
    c0 = np.sqrt(K_eos / rho)
    # Use gauge pressure (P0=0) for the EOS.  This keeps the baseline
    # pressure near zero so that discretization errors in the pressure
    # gradient don't produce catastrophic forces.  The hydrostatic profile
    # is P(z) = rho*g*(h_init - z), which is O(rho*g*h) ~ O(100) Pa.
    # TODO: switch to absolute P0=101325 when EOS stability is improved.
    P_atm = 0.0  # gauge reference [Pa]
    eos = TaitMurnaghan(rho0=rho, P0=P_atm, K=K_eos, n=1.0,
                        rho_clip=(0.5, 2.0))

    # ---- Mesh construction -------------------------------------------------
    if dim == 2:
        # 2D slit channel: width=2r, height=h_init
        from ddgclib.geometry.domains import rectangle
        result = rectangle(L=2 * r, h=h_init, refinement=n_refine,
                           flow_axis=0)
        HC = result.HC

        groups = identify_face_groups(HC, {
            'bottom': (gravity_axis, 0.0),
            'top': (gravity_axis, h_init),
            'left_wall': (0, 0.0),
            'right_wall': (0, 2 * r),
        })
        wall_verts = groups['left_wall'] | groups['right_wall']
        bottom_verts = groups['bottom']
        free_surface_verts = groups['top']

    else:  # dim == 3
        # 3D cylinder: radius=r, height=h_init along z
        from ddgclib.geometry.domains import cylinder_volume
        result = cylinder_volume(R=r, L=h_init, refinement=n_refine,
                                 flow_axis=gravity_axis)
        HC = result.HC

        wall_verts = result.boundary_groups['walls']
        bottom_verts = result.boundary_groups['inlet']   # z = 0
        free_surface_verts = result.boundary_groups['outlet']  # z = h_init

    # Frozen: walls only.  Bottom becomes inlet, top is free surface.
    # Corner vertices shared between wall and top/bottom groups:
    #   - wall-top corners → free surface (not frozen)
    #   - wall-bottom corners → will be managed by inlet BC
    bV = wall_verts - free_surface_verts - bottom_verts

    # ---- Tag boundaries and compute duals ----------------------------------
    all_boundary = wall_verts | bottom_verts | free_surface_verts
    for v in HC.V:
        v.boundary = v in all_boundary

    compute_vd(HC, cdist=1e-10)
    cache_dual_volumes(HC, dim)

    # ---- Initial conditions ------------------------------------------------
    # 1. Zero velocity
    ZeroVelocity(dim=dim).apply(HC, bV)

    # 2. EOS-consistent mass and pressure (compressible hydrostatic)
    #    P_ref = P_atm at the top (free surface at h_init)
    #    Pressure increases downward: P(z) = P_atm + ∫ rho(P)*g dz
    HydrostaticEOSMass(
        eos=eos, rho0=rho, g=g, gravity_axis=gravity_axis,
        h_ref=h_init, P_ref=P_atm,
    ).apply(HC, bV)

    # ---- Artificial viscosity ----------------------------------------------
    edges = [
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
    ]
    dx_mean = np.mean(edges)
    mu_art = artificial_viscosity_alpha * rho * c0 * dx_mean

    # ---- Column height tracker for body force ------------------------------
    _current_h = [h_init]

    def get_h():
        return _current_h[0]

    def set_h(h_new):
        _current_h[0] = h_new

    # ---- Build dudt_fn -----------------------------------------------------
    dudt_fn = make_capillary_dudt(
        dim=dim, mu=mu_art, HC=HC, g_val=g, gravity_axis=gravity_axis,
        P_cap=P_cap, rho=rho, h_ref_fn=get_h,
        pressure_model=eos,
    )

    # ---- Boundary conditions -----------------------------------------------
    bc_set = BoundaryConditionSet()

    # Estimate rise velocity from Washburn for reference
    if dim == 3:
        v_rise = r**2 * P_cap / (8 * mu * h_init)
    else:
        v_rise = r**2 * P_cap / (3 * mu * h_init)
    v_rise = min(v_rise, 0.1 * c0)

    # No-slip walls (simple, no inlet for now)
    from ddgclib._boundary_conditions import NoSlipWallBC
    bc_set.add(NoSlipWallBC(dim=dim), wall_verts - free_surface_verts)
    # Bottom: also no-slip (reservoir, fixed)
    bc_set.add(NoSlipWallBC(dim=dim), bottom_verts - free_surface_verts)

    # TODO: Add PeriodicInletBC at bottom for continuous fluid supply.
    # For now, the fixed bottom acts as a closed reservoir — the column
    # stretches upward under capillary body force.

    # ---- Return ------------------------------------------------------------
    params = {
        'dim': dim,
        'r': r,
        'H': H,
        'h_init': h_init,
        'gamma': gamma,
        'theta_deg': theta_deg,
        'mu': mu,
        'rho': rho,
        'g': g,
        'h_jurin': h_jurin,
        'P_cap': P_cap,
        'gravity_axis': gravity_axis,
        'P_atm': P_atm,
        'K_eos': K_eos,
        'c0': c0,
        'eos': eos,
        'mu_art': mu_art,
        'dx_mean': dx_mean,
        'n_refine': n_refine,
        'v_rise': v_rise,
        'set_h': set_h,
    }

    return HC, bV, bc_set, dudt_fn, free_surface_verts, params
