#!/usr/bin/env python3
"""Case 77 partitioned conservative bridge--film core.

Case77 retains the PR33/PR35/PR37 tetrahedral bridge solve, material
parameters, and backtracking mesh-safety controller.  Its four resolved P1
elements through the gap supply the wall-normal viscous resistance directly,
so no additional K_lub momentum matrix is assembled.  The outer film uses one
lagged-mobility backward-Euler solve of the physical axisymmetric thin-film
equation per accepted bridge step.

The bridge supplies only a computed junction volume rate.  The radial film
operator contains no experimental height, trough location, fitted recovery
width, fixed local-share fraction, or prescribed bridge increment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import k0, k1

from . import source_bridge_core as parent
from .operators.coupled_capillary_bridge_film import (
    advance_axisymmetric_film_prescribed_junction_flux,
)
from .operators.equilibrium_bridge import (
    EquilibriumBridgeManifold,
    build_fixed_sphere_equilibrium_bridge_manifold,
)
from .operators.unstructured_bridge_dynamics import (
    _implicit_nonlinear_film_flux_step,
)


ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 77"
OUTPUT_PREFIX = "case77"
OUT_DIR = ROOT / CASE_STEM
PREDICTION_OUTPUT = (
    OUT_DIR / "run_partitioned_moving_boundary_prediction_0to10"
)
TETRA_DIR_NAME = "mesh_states_tetra_volume"
PHYSICAL_FILM_DIR_NAME = "case77_physical_film_states"
FILM_MINIMUM_HEIGHT_M = 1.0e-9
FILM_PICARD_ITERATIONS = 2
FILM_PICARD_RELAXATION = 0.65
FILM_PICARD_TOLERANCE_M = 2.0e-10
FILM_MAX_PHYSICAL_SUBSTEP_S = 1.0
MAX_VISCOCAPILLARY_WIDTH_CAPILLARY_LENGTHS = 2.5
# Case62's published run uses the original one-sided outer-film reduction.
# A successor may enable the resolved two-sided transition without duplicating
# the PR33/PR35/PR37 bridge implementation.
RESOLVE_TWO_SIDED_JUNCTION_LAYER = False
# Reduced one-layer models require an extra series factor for an unresolved
# bridge-side arm. Case77's four through-gap elements assemble that arm in
# K_mu, so its entry point disables this separate supply multiplier.
APPLY_UNRESOLVED_TWO_ARM_SUPPLY_RESISTANCE = True
USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER = False
USE_SUBGRID_JUNCTION_FLUX_CLOSURE = False
RETAIN_CASE61_NECK_SUPPLY_STATE = False
RETAIN_CASE61_NECK_LOW_CA_ONLY = False
USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE = False
# The original Case62 branch treated one local viscocapillary annulus as the
# complete film inventory and permanently pinned the contact line when that
# annulus was depleted.  Keep that legacy event selectable for reproducibility,
# but let successors use the pressure-driven supply from the complete resolved
# radial film instead.
USE_TERMINAL_LOCAL_DONOR_CAP = True
# Keep the material capillary-support annulus attached to the advancing
# contact line.  Its accessible inventory grows with the solved circumference;
# only after this current-state reservoir is spent does remote-film flux limit
# PR35.  No measured bridge trajectory enters the capacity.
USE_EXPANDING_CONNECTED_SUPPORT_RESERVOIR = False
# Case77 uses one conservative supply trajectory.  The state-derived local
# donor remains the early-time active set, but its exhaustion is not a terminal
# event: capillary diffusion through the resolved far film progressively
# restores hydraulic communication with the junction.
USE_UNIFIED_LOCAL_TO_FAR_SUPPLY = False
# Once the capillary-diffusion front has established full far-film
# communication, the unresolved bridge--film throat contributes its own
# lubrication conductance.  Case77 enables this state-derived series
# resistance; historical Case62 runs leave it disabled.
USE_FULL_CONNECTION_JUNCTION_MOBILITY = False
# Scale the available capillary-front supply by the remaining Young--Laplace
# pressure drive from the computed equilibrium manifold. This supplies the
# missing late-time feedback without solving a second 3-D system.
USE_EQUILIBRIUM_PRESSURE_SUPPLY_FACTOR = False
# A coupling below five percent of the fully connected far-film flux produces
# sub-mesh contact displacements while still forcing a complete ALE rebuild.
# Treat it as the inactive side of the same numerical active set, then rescale
# the resolved interval continuously from zero to one.
FAR_SUPPLY_RESOLUTION_ACTIVATION = 0.05
# Optional pressure continuity closes the partitioned bridge--film junction
# with the gauge-invariant pressure difference returned by PR35.  It is off in
# the historical Case62 branch and enabled by Case77.
USE_PR35_JUNCTION_PRESSURE_CONTINUITY = False
# Use the exact PR33 Heron traction on the bridge free-surface patch as the
# physical pressure boundary for the conservative film solve.  Unlike the
# PR35 multiplier, this scalar contains no incompressibility gauge or sphere
# active-set reaction.
USE_HERON_JUNCTION_PRESSURE_CONTINUITY = False
# A pressure-driven successor calculates one scalar conservative junction flux
# from the local Young--Laplace bridge pressure and the complete film
# resistance.  The raw PR35 multiplier is retained as a diagnostic because it
# also contains contact/admissibility reactions and is not, by itself, the
# physical bridge pressure to impose on the film.
USE_PRESSURE_DRIVEN_JUNCTION_FLUX = False
# Solve the finite-substrate film on its own moving axisymmetric grid and use
# the pressure-driven flux at the bridge footprint as the conservative PR33
# transfer limit.  The tetra film remains a mesh-support field; it is not used
# to replace the physical finite-rim boundary condition h(L)=0.
USE_MOVING_BOUNDARY_PRESSURE_FILM = False
# Case77 can replace the single coarse bridge--film edge by a conservative
# body-fitted capillary junction.  The junction centre is selected by the
# computed Young--Laplace bridge volume and the material capillary length.
# No experimental ordinate, radius, width, or time enters this closure.
BODY_FITTED_JUNCTION_REFINEMENT_ENABLED = False
# A resolved capillary interface has one tangent across the artificial
# bridge/film region label.  When the body-fitted branch is active, replace the
# angular P1 shoulder by a C1 join over the computed inner capillary length.
# Separate bridge-side and film-side axisymmetric volumes are retained exactly;
# no experimental height, radius, or smoothing gain enters the construction.
BODY_FITTED_CONSERVATIVE_C1_JOIN_ENABLED = False
BODY_FITTED_OUTER_CONSERVATIVE_C1_JOIN_ENABLED = False
# Saved-state cadence is independent of the accepted integration step.  Long
# low-Ca continuations may retain sparse physical checkpoints without changing
# any PR33/PR35/PR37 or film update.
SNAPSHOT_INTERVAL_S = 1.0
CASE_ENTRY_SOURCE = Path(__file__).resolve()

case28 = parent.case28
base = parent.base

CONFIG = replace(
    parent.CONFIG,
    dt_s=0.02,
    max_steps=500,
    record_every_steps=50,
    snapshot_times_s=tuple(float(value) for value in range(0, 11)),
)

# A restart creates a fresh tetra state.  The checkpoint surface is sufficient
# to restart this first-order partitioned film update; these globals transfer
# only that computed state between the startup and efficient continuation.
_RESTART_FILM_RADIUS_M: np.ndarray | None = None
_RESTART_FILM_HEIGHT_M: np.ndarray | None = None
_RESTART_BRIDGE_VOLUME_UL: float | None = None
_RESTART_JUNCTION_RADIUS_M: float | None = None
_RESTART_HYDRAULIC_DEFICIT_UL = 0.0
_RESTART_HYDRAULIC_RATE_UL_S = 0.0
_RESTART_ACCEPTED_MASS_SUPPLY_FLUX_UL_S = 0.0
_RESTART_LOCAL_DEFICIT_INVENTORY_UL = 0.0
_RESTART_FAR_DEFICIT_INVENTORY_UL = 0.0
_RESTART_FAR_DEFICIT_RATE_UL_S = 0.0
_RESTART_SUPPLY_SATURATION_TIME_S = 0.0
_RESTART_TERMINAL_EVENT_CHECKPOINT = False
_PARTITIONED_CAPTURE_RESERVOIR_UL = 0.0
_PARTITIONED_FINITE_SUPPLY_CAP_UL: float | None = None
_PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S = 0.0
_RESTART_PRESSURE_DRIVEN_FLUX_UL_S = 0.0
_RESTART_PHYSICAL_FILM_RADIUS_M: np.ndarray | None = None
_RESTART_PHYSICAL_FILM_HEIGHT_M: np.ndarray | None = None
_RESTART_PHYSICAL_FILM_FLUX_UL_S = 0.0
_RESTART_JUNCTION_IMPEDANCE_PA_S_M2 = 0.0
_RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M = 0.0
_RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M = 0.0
_RESTART_BODY_FITTED_TRANSITION_TIME_S = 0.0
_RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL = 0.0
_PARTITIONED_LOCAL_SATURATION_TIME_S = 0.0
_PARTITIONED_FAR_SUPPLY_ACTIVATION = 0.0
_PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M = 0.0
_PARTITIONED_FAR_SUPPLY_FLUX_UL_S = 0.0
_PARTITIONED_FAR_SIMILARITY_FLUX_UL_S = 0.0
_PARTITIONED_FAR_DONOR_HEIGHT_M = 0.0
_PARTITIONED_PRESSURE_SUPPLY_FACTOR = 1.0
_PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL = 0.0
_PARTITIONED_ACCEPTED_LOCAL_DEFICIT_INVENTORY_UL = 0.0
_PARTITIONED_ACCEPTED_FAR_DEFICIT_INVENTORY_UL = 0.0
_PARTITIONED_ACCEPTED_FAR_DEFICIT_RATE_UL_S = 0.0
_PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL: float | None = None
_PARTITIONED_JUNCTION_THROAT_WIDTH_M = 0.0
_PARTITIONED_JUNCTION_THROAT_HEIGHT_M = 0.0
_PARTITIONED_JUNCTION_THROAT_MOBILITY = 1.0
_PARTITIONED_TWO_ARM_TRANSMISSION = 1.0
_PARTITIONED_CONNECTED_SUPPORT_ACTIVE = False
_PARTITIONED_FILM_INNER_PRESSURE_PA = 0.0
_EQUILIBRIUM_BRIDGE_MANIFOLD: EquilibriumBridgeManifold | None = None


def _partitioned_flux_reference_surface(
    state: case28.TetraFreeSurfaceState,
) -> np.ndarray:
    """Return a flux-only conservative two-sided junction profile.

    The coarse tetra surface remains unchanged.  For the pressure-flux
    diagnostic used by the mass-supply active set, its broad junction deficit
    is concentrated into a 33-point viscocapillary similarity profile.  The
    profile uses only the accepted film state, junction rate, contact speed,
    material properties, and scalar volume conservation.
    """

    surface = np.asarray(state.surface_points(), dtype=float).copy()
    if not bool(USE_SUBGRID_JUNCTION_FLUX_CLOSURE):
        return surface
    diag = dict(getattr(state, "case62_last_accepted_diag", {}))
    hydraulic_rate_m3_s = (
        abs(float(diag.get("case62_hydraulic_junction_rate_ul_s", 0.0)))
        * 1.0e-9
    )
    contact_speed_m_s = abs(
        float(diag.get("case62_contact_speed_m_s", 0.0))
    )
    outer_boundary_m = (
        float(diag.get("case62_physical_junction_radius_mm", float("nan")))
        * 1.0e-3
    )
    rings = np.asarray(state.rings, dtype=int)
    ring_region = np.asarray(
        getattr(state, "case62_ring_region", np.zeros(rings.shape[0])),
        dtype=int,
    )
    if ring_region.shape != (rings.shape[0],):
        return surface
    bridge_rows = np.flatnonzero(ring_region == 0)
    film_rows = np.flatnonzero(ring_region == 1)
    if bridge_rows.size < 2 or film_rows.size < 5:
        return surface
    row_radius, row_height = parent._ring_coordinates(surface, rings)
    inner_boundary_m = float(row_radius[int(bridge_rows[-1])])
    if (
        hydraulic_rate_m3_s <= 1.0e-30
        or contact_speed_m_s <= 1.0e-30
        or not math.isfinite(outer_boundary_m)
        or outer_boundary_m <= inner_boundary_m + 1.0e-9
    ):
        return surface

    film_order = film_rows[np.argsort(row_radius[film_rows])]
    source_radius_m = np.asarray(row_radius[film_order], dtype=float)
    source_height_m = np.asarray(row_height[film_order], dtype=float)
    h0_m = float(CONFIG.initial_film_thickness_um) * 1.0e-6
    interior = (
        (source_radius_m > inner_boundary_m)
        & (source_radius_m < outer_boundary_m)
    )
    audit_radius_m = np.concatenate(
        (
            np.asarray([inner_boundary_m]),
            source_radius_m[interior],
            np.asarray([outer_boundary_m]),
        )
    )
    audit_height_m = np.interp(
        audit_radius_m,
        source_radius_m,
        source_height_m,
    )
    target_deficit_m3 = float(
        2.0
        * math.pi
        * np.trapezoid(
            audit_radius_m
            * np.maximum(h0_m - audit_height_m, 0.0),
            audit_radius_m,
        )
    )
    if target_deficit_m3 <= 1.0e-30:
        return surface

    midpoint_m = 0.5 * (inner_boundary_m + outer_boundary_m)
    provisional_height_m = max(
        float(np.interp(midpoint_m, audit_radius_m, audit_height_m)),
        FILM_MINIMUM_HEIGHT_M,
    )
    viscosity = float(CONFIG.viscosity_pa_s)
    gamma = float(CONFIG.surface_tension_n_m)
    inner_slope = (
        3.0 * viscosity * contact_speed_m_s / gamma
    ) ** (1.0 / 3.0)
    radial_supply_speed_m_s = hydraulic_rate_m3_s / (
        2.0
        * math.pi
        * midpoint_m
        * provisional_height_m
    )
    outer_slope = (
        3.0
        * viscosity
        * (contact_speed_m_s + radial_supply_speed_m_s)
        / gamma
    ) ** (1.0 / 3.0)
    slope_sum = inner_slope + outer_slope
    if slope_sum <= 1.0e-30:
        return surface
    width_m = outer_boundary_m - inner_boundary_m
    inner_support_m = width_m * outer_slope / slope_sum
    outer_support_m = width_m - inner_support_m
    junction_radius_m = inner_boundary_m + inner_support_m
    local_radius_m = np.concatenate(
        (
            np.linspace(inner_boundary_m, junction_radius_m, 17),
            np.linspace(junction_radius_m, outer_boundary_m, 17)[1:],
        )
    )
    weight = np.empty_like(local_radius_m)
    left = local_radius_m <= junction_radius_m
    weight[left] = np.maximum(
        1.0
        - (junction_radius_m - local_radius_m[left])
        / max(inner_support_m, 1.0e-30),
        0.0,
    ) ** (2.0 / 3.0)
    weight[~left] = np.maximum(
        1.0
        - (local_radius_m[~left] - junction_radius_m)
        / max(outer_support_m, 1.0e-30),
        0.0,
    ) ** (2.0 / 3.0)
    capacity_m2 = float(
        2.0
        * math.pi
        * np.trapezoid(local_radius_m * weight, local_radius_m)
    )
    if capacity_m2 <= 1.0e-30:
        return surface
    local_height_m = h0_m - target_deficit_m3 / capacity_m2 * weight
    if float(np.min(local_height_m)) <= case28.TETRA_MIN_LAYER_HEIGHT_M:
        return surface

    flux_height_m = np.maximum(source_height_m, h0_m)
    layer = (
        (source_radius_m >= inner_boundary_m)
        & (source_radius_m <= outer_boundary_m)
    )
    flux_height_m[layer] = np.interp(
        source_radius_m[layer],
        local_radius_m,
        local_height_m,
    )
    for row_id, height_m in zip(film_order, flux_height_m):
        surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(
            height_m
        )
    return surface


def _partitioned_mass_limited_contact_line_speed_cap(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    time_s: float | None,
) -> tuple[float, float, float, float, float, float]:
    """Use junction flux after the physically swept local film is spent.

    The initial capture annulus is not a fixed reservoir: advancing the
    contact line exposes an additional annulus of the finite film.  Account
    for that liquid with the Reynolds swept-area term

        dV_swept = 2*pi*r_CL*h0*dr_CL.

    This is restart invariant and avoids the older shortcut that enlarged the
    reservoir by relabelling the complete bridge inventory as available film.
    """

    (
        speed_cap,
        flux_ul_s,
        activation,
        local_capture_ul,
        film_inner_pressure_pa,
        film_next_pressure_pa,
    ) = parent._ORIGINAL_MASS_LIMITED_CONTACT_LINE_SPEED_CAP(
        surface,
        rings,
        ring_region,
        config,
        time_s,
    )
    global _PARTITIONED_FINITE_SUPPLY_CAP_UL
    global _PARTITIONED_FAR_SUPPLY_ACTIVATION
    global _PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M
    global _PARTITIONED_FAR_SUPPLY_FLUX_UL_S
    global _PARTITIONED_FAR_SIMILARITY_FLUX_UL_S
    global _PARTITIONED_FAR_DONOR_HEIGHT_M
    global _PARTITIONED_PRESSURE_SUPPLY_FACTOR
    global _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL
    global _PARTITIONED_JUNCTION_THROAT_WIDTH_M
    global _PARTITIONED_JUNCTION_THROAT_HEIGHT_M
    global _PARTITIONED_JUNCTION_THROAT_MOBILITY
    global _PARTITIONED_TWO_ARM_TRANSMISSION
    global _PARTITIONED_CONNECTED_SUPPORT_ACTIVE
    global _PARTITIONED_FILM_INNER_PRESSURE_PA

    _PARTITIONED_FAR_SUPPLY_ACTIVATION = 0.0
    _PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M = 0.0
    _PARTITIONED_FAR_SUPPLY_FLUX_UL_S = 0.0
    _PARTITIONED_FAR_SIMILARITY_FLUX_UL_S = 0.0
    _PARTITIONED_FAR_DONOR_HEIGHT_M = 0.0
    _PARTITIONED_PRESSURE_SUPPLY_FACTOR = 1.0
    _PARTITIONED_JUNCTION_THROAT_WIDTH_M = 0.0
    _PARTITIONED_JUNCTION_THROAT_HEIGHT_M = h0_m = (
        float(config.initial_film_thickness_um) * 1.0e-6
    )
    _PARTITIONED_JUNCTION_THROAT_MOBILITY = 1.0
    _PARTITIONED_TWO_ARM_TRANSMISSION = 1.0
    _PARTITIONED_CONNECTED_SUPPORT_ACTIVE = False
    _PARTITIONED_FILM_INNER_PRESSURE_PA = float(film_inner_pressure_pa)

    row_radius_m, row_height_m = parent._ring_coordinates(surface, rings)
    bridge_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 0)
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    contact_radius_m = (
        float(row_radius_m[int(bridge_rows[0])])
        if bridge_rows.size
        else float(config.initial_bridge_radius_mm) * 1.0e-3
    )
    material_rim_m = (
        float(row_radius_m[int(bridge_rows[-1])])
        if bridge_rows.size
        else contact_radius_m
    )
    # The captured annulus starts at the actual finite-contact nucleus.  The
    # former sqrt(2*R*h0-h0^2) expression is the radius of a sphere cut by a
    # plane one complete film thickness above its bottom; it is about
    # 0.995 mm here and is not the first-contact radius.  Using it silently
    # creates an O(1 uL) liquid reservoir before the contact line has swept
    # that film.  The geometric nucleus is derived independently from the
    # declared sphere-clearance regularization and stored in the config.
    first_contact_radius_m = max(
        float(config.initial_bridge_radius_mm) * 1.0e-3,
        0.0,
    )
    capillary_length_m = math.sqrt(
        float(config.surface_tension_n_m)
        / max(
            float(config.density_kg_m3)
            * float(config.gravity_m_s2),
            1.0e-30,
        )
    )
    # The initially connected topology contains the capillary-support annulus
    # and the unresolved bridge-side turn.  Their independently derived
    # lengths are ell_c and sqrt(h0*ell_c), respectively.  This replaces
    # Case28's inherited fixed 1.25 multiplier with material/geometric scales.
    initial_capture_width_m = (
        capillary_length_m + math.sqrt(h0_m * capillary_length_m)
    )
    capillary_support_capture_ul = (
        2.0
        * math.pi
        * first_contact_radius_m
        * h0_m
        * initial_capture_width_m
        * 1.0e9
    )
    swept_film_ul = (
        math.pi
        * max(
            contact_radius_m**2 - first_contact_radius_m**2,
            0.0,
        )
        * h0_m
        * 1.0e9
    )
    initial_capture_ul = (
        float(_PARTITIONED_CAPTURE_RESERVOIR_UL)
        if _PARTITIONED_CAPTURE_RESERVOIR_UL > 0.0
        else float(capillary_support_capture_ul)
    )
    capture_reservoir_ul = initial_capture_ul + swept_film_ul
    inventory_ul = parent.bridge_inventory_volume_ul(
        surface,
        rings,
        ring_region,
        config,
    )
    if (
        bool(USE_MOVING_BOUNDARY_PRESSURE_FILM)
        and _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
        and inventory_ul
        >= float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - 1.0e-12
        and float(_PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S) > 0.0
    ):
        # Reynolds transport at the moving footprint gives
        #
        #   dV_br/dt = Q_J + 2*pi*a*h0*da/dt.
        #
        # Therefore the pressure-film flux limits only the unswept part of
        # dV_br/da.  This replaces both the finite-reservoir switch and the
        # similarity-front activation; no experimental time or gain enters.
        # Keep the accepted local-donor threshold as the irreversible
        # handoff marker.  Clearing it here made the next call rebuild a new
        # reservoir from the changed mesh and toggle back to local capture,
        # producing artificial bursts in Q_J.
        flux_ul_s = float(_PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S)
        cap_volume_derivative_m2 = float(
            case28.attached_cap_volume_derivative_m2(
                config,
                contact_radius_m,
            )
        )
        hydraulic_volume_derivative_m2 = max(
            cap_volume_derivative_m2
            - 2.0 * math.pi * contact_radius_m * h0_m,
            1.0e-30,
        )
        speed_cap = (
            flux_ul_s * 1.0e-9 / hydraulic_volume_derivative_m2
        )
        capture_reservoir_ul = max(
            inventory_ul
            - max(float(parent._FLUX_SUPPLY_BUDGET_UL), 0.0),
            0.0,
        )
        return (
            float(speed_cap),
            float(flux_ul_s),
            1.0,
            capture_reservoir_ul,
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    if inventory_ul < capture_reservoir_ul - 1.0e-12:
        return (
            float("inf"),
            float(flux_ul_s),
            0.0,
            capture_reservoir_ul,
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    if not bool(USE_TERMINAL_LOCAL_DONOR_CAP):
        connected_support_ul = (
            2.0
            * math.pi
            * max(contact_radius_m, first_contact_radius_m)
            * h0_m
            * capillary_length_m
            * 1.0e9
        )
        if bool(USE_EXPANDING_CONNECTED_SUPPORT_RESERVOIR):
            _PARTITIONED_CONNECTED_SUPPORT_ACTIVE = True
            if inventory_ul < connected_support_ul - 1.0e-12:
                # Liquid in the currently connected capillary-support annulus
                # is already part of the meshed donor.  It creates no finite
                # endpoint speed cap; PR35/Cox determines motion while the
                # conservative film update removes the corresponding volume.
                return (
                    float("inf"),
                    float(flux_ul_s),
                    0.0,
                    connected_support_ul,
                    float(film_inner_pressure_pa),
                    float(film_next_pressure_pa),
                )
        # Once the initially swept liquid is spent, the surrounding film does
        # not disappear: it continues to feed the junction through the solved
        # lubrication-pressure path.  The inherited cap is exactly
        #
        #   U_CL <= Q_film / (dV_cap/dr_CL),
        #
        # and the accepted film update removes the same volume from the radial
        # film.  There is therefore no terminal inventory at one capillary
        # length and no prescribed long-time bridge increment.
        _PARTITIONED_FINITE_SUPPLY_CAP_UL = None
        if bool(USE_PRESSURE_DRIVEN_JUNCTION_FLUX):
            flux_ul_s = max(
                float(_PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S),
                0.0,
            )
            cap_volume_derivative_m2 = max(
                float(
                    case28.attached_cap_volume_derivative_m2(
                        config,
                        contact_radius_m,
                    )
                ),
                1.0e-30,
            )
            speed_cap = (
                flux_ul_s
                * 1.0e-9
                / cap_volume_derivative_m2
            )
            activation = 1.0
            # The inherited budget stores the time integral of the available
            # flux, even when another physical constraint prevents the bridge
            # from accepting all of it.  Cancel that unused bank here so every
            # new step may receive at most its current Q_J*dt.  This keeps the
            # speed cap equal to the contemporaneous conservative film flux.
            capture_reservoir_ul = max(
                inventory_ul
                - max(float(parent._FLUX_SUPPLY_BUDGET_UL), 0.0),
                0.0,
            )
        return (
            float(speed_cap),
            float(flux_ul_s),
            float(activation),
            capture_reservoir_ul,
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    connected_support_ul = (
        2.0
        * math.pi
        * max(contact_radius_m, first_contact_radius_m)
        * h0_m
        * capillary_length_m
        * 1.0e9
    )
    if (
        _PARTITIONED_FINITE_SUPPLY_CAP_UL is None
        and inventory_ul < connected_support_ul - 1.0e-12
    ):
        return (
            float("inf"),
            float(flux_ul_s),
            0.0,
            connected_support_ul,
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    if _PARTITIONED_FINITE_SUPPLY_CAP_UL is None:
        local_film_rows = film_rows[
            (row_radius_m[film_rows] >= material_rim_m)
            & (
                row_radius_m[film_rows]
                <= material_rim_m + capillary_length_m
            )
        ]
        junction_height_m = (
            float(np.min(row_height_m[local_film_rows]))
            if local_film_rows.size
            else h0_m
        )
        reference_speed_m_s = max(
            float(case28.cox_voinov_speed_cap(config)),
            1.0e-30,
        )
        junction_width_m = (
            float(config.surface_tension_n_m)
            * max(junction_height_m, FILM_MINIMUM_HEIGHT_M) ** 3
            / (
                3.0
                * float(config.viscosity_pa_s)
                * reference_speed_m_s
            )
        ) ** (1.0 / 3.0)
        junction_radius_m = max(
            material_rim_m + junction_width_m,
            1.0e-9,
        )
        # The triangular donor belongs to the unresolved bridge-side turn.
        # The moving pressure-film grid begins outside this turn and owns the
        # outer radial film, so the two inventories are adjacent rather than
        # duplicated.
        outer_junction_donor_ul = (
            math.pi
            * junction_radius_m
            * junction_width_m
            * max(h0_m - junction_height_m, 0.0)
            * 1.0e9
        )
        # At the handoff, the connected local donor is the complete annulus
        # surrounding the current contact ring over the independently
        # derived support length ell_c + sqrt(h0*ell_c).  Using the original
        # microscopic seed circumference here undercounts this annulus after
        # the contact line has advanced.  Freeze the capacity at first
        # depletion; subsequent supply must come from capillary diffusion.
        connected_support_ul = (
            2.0
            * math.pi
            * max(contact_radius_m, first_contact_radius_m)
            * h0_m
            * capillary_length_m
            * 1.0e9
        )
        _PARTITIONED_FINITE_SUPPLY_CAP_UL = max(
            capture_reservoir_ul + outer_junction_donor_ul,
            connected_support_ul,
        )
    if inventory_ul >= float(_PARTITIONED_FINITE_SUPPLY_CAP_UL):
        effective_local_capture_ul = max(
            float(_PARTITIONED_FINITE_SUPPLY_CAP_UL)
            - max(float(parent._FLUX_SUPPLY_BUDGET_UL), 0.0),
            0.0,
        )
        if bool(USE_PRESSURE_DRIVEN_JUNCTION_FLUX):
            # The finite local donor is only the early connected inventory,
            # not a terminal bridge volume.  Thereafter the accepted
            # pressure-driven film flux is the conservative supply limit.
            pressure_flux_ul_s = max(
                float(_PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S),
                0.0,
            )
            cap_volume_derivative_m2 = max(
                float(
                    case28.attached_cap_volume_derivative_m2(
                        config,
                        contact_radius_m,
                    )
                ),
                1.0e-30,
            )
            pressure_speed_cap_m_s = (
                pressure_flux_ul_s
                * 1.0e-9
                / cap_volume_derivative_m2
            )
            return (
                float(pressure_speed_cap_m_s),
                float(pressure_flux_ul_s),
                1.0,
                effective_local_capture_ul,
                float(film_inner_pressure_pa),
                float(film_next_pressure_pa),
            )
        if bool(USE_UNIFIED_LOCAL_TO_FAR_SUPPLY):
            # Once the local annulus is depleted, the far film does not become
            # available instantaneously.  A capillary-leveling disturbance in
            # a thin film propagates over
            #
            #   ell_d(t) = [gamma*h0^3*(t-t_sat)/(3*mu)]^(1/4).
            #
            # The first capillary length refills the depleted local donor; the
            # next capillary length smoothly establishes the far-film flux.
            # This is the similarity scaling of the same lubrication equation
            # solved below, not a prescribed activation time.
            capillary_length_m = math.sqrt(
                float(config.surface_tension_n_m)
                / max(
                    float(config.density_kg_m3)
                    * float(config.gravity_m_s2),
                    1.0e-30,
                )
            )
            elapsed_s = max(
                float(time_s if time_s is not None else 0.0)
                - float(_PARTITIONED_LOCAL_SATURATION_TIME_S),
                0.0,
            )
            capillary_diffusivity_m4_s = (
                float(config.surface_tension_n_m) * h0_m**3
                / max(3.0 * float(config.viscosity_pa_s), 1.0e-30)
            )
            diffusion_front_m = (
                capillary_diffusivity_m4_s * elapsed_s
            ) ** 0.25
            inner_arm_length_m = math.sqrt(
                h0_m * capillary_length_m
            )
            bridge_turn_height_m = max(
                float(row_height_m[int(bridge_rows[-1])])
                if bridge_rows.size
                else h0_m,
                h0_m,
            )
            # Series resistance is proportional to length/h^3.  As the
            # solved bridge-side turn opens above h0, its equivalent radial
            # resistance decreases directly from the computed geometry.
            inner_arm_resistance_length_m = inner_arm_length_m * (
                h0_m / bridge_turn_height_m
            ) ** 3
            # The disturbance first crosses the matched bridge-side turn
            # sqrt(h0*ell_c), then progressively opens the capillary-length
            # outer-film arm.  This avoids both an artificial full-ell_c
            # zero-flux plateau and instantaneous access to the far reservoir.
            refill_coordinate = float(
                np.clip(
                    (diffusion_front_m - inner_arm_length_m)
                    / max(capillary_length_m, 1.0e-30),
                    0.0,
                    1.0,
                )
            )
            raw_far_activation = (
                refill_coordinate**2
                * (3.0 - 2.0 * refill_coordinate)
            )
            activation_floor = float(
                np.clip(FAR_SUPPLY_RESOLUTION_ACTIVATION, 0.0, 0.95)
            )
            far_activation = max(
                raw_far_activation - activation_floor,
                0.0,
            ) / max(1.0 - activation_floor, 1.0e-30)
            local_film_rows = film_rows[
                (row_radius_m[film_rows] >= material_rim_m)
                & (
                    row_radius_m[film_rows]
                    <= material_rim_m + capillary_length_m
                )
            ]
            junction_height_m = (
                float(np.min(row_height_m[local_film_rows]))
                if local_film_rows.size
                else h0_m
            )
            # A narrow body-fitted minimum is not the hydraulic height of the
            # complete donor annulus.  Lubrication conductances add through
            #
            #   int dr/[r h(r)^3] = log(r_2/r_1)/h_eff^3.
            #
            # Use that resolved conductance-equivalent height in the far-film
            # similarity flux.  This costs one scalar quadrature and avoids
            # making the supply depend on a single minimum-height vertex.
            hydraulic_effective_height_m = junction_height_m
            if local_film_rows.size >= 2:
                local_order = np.argsort(row_radius_m[local_film_rows])
                local_radius_m = np.asarray(
                    row_radius_m[local_film_rows][local_order],
                    dtype=float,
                )
                local_height_m = np.maximum(
                    np.asarray(
                        row_height_m[local_film_rows][local_order],
                        dtype=float,
                    ),
                    FILM_MINIMUM_HEIGHT_M,
                )
                local_dr_m = np.diff(local_radius_m)
                positive_faces = local_dr_m > 1.0e-15
                if np.any(positive_faces):
                    reference_log_radius = math.log(
                        max(
                            float(local_radius_m[-1]),
                            1.0e-30,
                        )
                        / max(
                            float(local_radius_m[0]),
                            1.0e-30,
                        )
                    )
                    inverse_conductance_m3 = float(
                        np.sum(
                            local_dr_m[positive_faces]
                            * 0.5
                            * (
                                1.0
                                / (
                                    local_radius_m[:-1][positive_faces]
                                    * local_height_m[:-1][positive_faces] ** 3
                                )
                                + 1.0
                                / (
                                    local_radius_m[1:][positive_faces]
                                    * local_height_m[1:][positive_faces] ** 3
                                )
                            )
                        )
                    )
                    if (
                        reference_log_radius > 0.0
                        and inverse_conductance_m3 > 0.0
                    ):
                        hydraulic_effective_height_m = (
                            reference_log_radius
                            / inverse_conductance_m3
                        ) ** (1.0 / 3.0)
            # The inner resolved gap resistance belongs to the assembled
            # PR35 K_mu operator. The far-film similarity flux instead uses
            # the solved film height at its propagating donor front. Using
            # the inner trough height here would count the same resistance in
            # both K_mu and the conservative supply boundary condition.
            throughflow_connected = bool(
                refill_coordinate >= 1.0 - 1.0e-12
            )
            far_reservoir_connected = bool(
                diffusion_front_m >= 2.0 * capillary_length_m
            )
            diffusion_front_speed_m_s = (
                capillary_diffusivity_m4_s
                / max(4.0 * diffusion_front_m**3, 1.0e-30)
            )
            similarity_front_radius_m = min(
                material_rim_m
                + 0.5
                * (capillary_length_m + diffusion_front_m),
                float(config.substrate_radius_mm) * 1.0e-3,
            )
            donor_order = np.argsort(row_radius_m[film_rows])
            donor_radius_m = np.asarray(
                row_radius_m[film_rows][donor_order],
                dtype=float,
            )
            donor_height_m = np.asarray(
                row_height_m[film_rows][donor_order],
                dtype=float,
            )
            mobile_film_height_m = float(
                np.clip(
                    np.interp(
                        similarity_front_radius_m,
                        donor_radius_m,
                        donor_height_m,
                    ),
                    FILM_MINIMUM_HEIGHT_M,
                    h0_m,
                )
            )
            similarity_flux_ul_s = (
                2.0
                * math.pi
                * similarity_front_radius_m
                * mobile_film_height_m
                * diffusion_front_speed_m_s
                * 1.0e9
            )
            # The capillary-front similarity solution is the resolved
            # far-film transport closure after the coarse tetra film loses
            # radial resolution.  Applying the inherited coarse-grid
            # pressure-flux estimate as a second minimum would count the same
            # outer-film resistance twice and make the result mesh dependent.
            # PR37 and the conservative donor update still limit how much of
            # this available flux the bridge actually accepts.
            far_flux_ul_s = (
                far_activation * float(similarity_flux_ul_s)
            )
            if bool(USE_EQUILIBRIUM_PRESSURE_SUPPLY_FACTOR):
                bridge_pressure_pa = (
                    _young_laplace_bridge_junction_pressure_pa(
                        surface,
                        rings,
                        ring_region,
                        config,
                    )
                )
                if (
                    bridge_pressure_pa is not None
                    and _EQUILIBRIUM_BRIDGE_MANIFOLD is not None
                    and _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
                ):
                    volume_grid_ul = (
                        np.asarray(
                            _EQUILIBRIUM_BRIDGE_MANIFOLD.bridge_volume_m3,
                            dtype=float,
                        )
                        * 1.0e9
                    )
                    suction_grid_pa = np.asarray(
                        _EQUILIBRIUM_BRIDGE_MANIFOLD.suction_pressure_pa,
                        dtype=float,
                    )
                    hydrostatic_film_pressure_pa = (
                        float(config.density_kg_m3)
                        * float(config.gravity_m_s2)
                        * h0_m
                    )
                    equilibrium_pressure_pa = (
                        hydrostatic_film_pressure_pa
                        + float(suction_grid_pa[-1])
                    )
                    handoff_suction_pa = float(
                        np.interp(
                            float(_PARTITIONED_FINITE_SUPPLY_CAP_UL),
                            volume_grid_ul,
                            suction_grid_pa,
                        )
                    )
                    handoff_pressure_pa = (
                        hydrostatic_film_pressure_pa
                        + handoff_suction_pa
                    )
                    pressure_supply_factor = float(
                        np.clip(
                            (
                                equilibrium_pressure_pa
                                - float(bridge_pressure_pa)
                            )
                            / max(
                                equilibrium_pressure_pa
                                - handoff_pressure_pa,
                                1.0e-30,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    far_flux_ul_s *= pressure_supply_factor
                    _PARTITIONED_PRESSURE_SUPPLY_FACTOR = (
                        pressure_supply_factor
                    )
            if (
                bool(RESOLVE_TWO_SIDED_JUNCTION_LAYER)
                and bool(APPLY_UNRESOLVED_TWO_ARM_SUPPLY_RESISTANCE)
            ):
                # The capillary-front similarity flux represents the outer
                # donor arm alone.  The already depleted support annulus and
                # radial capillary matching turn remain in series.  This is
                # distinct from the wall-normal K_mu gap dissipation.  With
                # leading-order h^3 mobility, resistance scales with path
                # length: Q/Q_outer=ell_d/(ell_d+ell_c+ell_J).
                two_arm_transmission = (
                    diffusion_front_m
                    / max(
                        diffusion_front_m
                        + capillary_length_m
                        + inner_arm_resistance_length_m,
                        1.0e-30,
                    )
                )
                if not bool(USE_FULL_CONNECTION_JUNCTION_MOBILITY):
                    far_flux_ul_s *= two_arm_transmission
                    _PARTITIONED_TWO_ARM_TRANSMISSION = (
                        two_arm_transmission
                    )
            if (
                bool(USE_FULL_CONNECTION_JUNCTION_MOBILITY)
                and throughflow_connected
            ):
                # Once the diffusion front spans the complete two-capillary-
                # length reconnection interval, the far film and bridge are
                # joined through a thin unresolved hydraulic throat.  Its
                # matched length is the geometric mean of the film thickness
                # and capillary length,
                #
                #   ell_J = sqrt(h0*ell_c),
                #
                # and its mobile thickness is obtained from the accepted
                # post-connection hydraulic transfer.  The cubic factor is
                # Poiseuille mobility of liquid above the existing residual
                # film h_*.  This is a series K_lub contribution, not a fixed
                # rate/volume cap and not an experimental calibration.
                if _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL is None:
                    _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL = float(
                        _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL
                    )
                residual_height_m = max(
                    float(
                        config.attached_outer_deficit_soft_lower_um
                    )
                    * 1.0e-6,
                    FILM_MINIMUM_HEIGHT_M,
                )
                throat_width_m = math.sqrt(
                    h0_m * max(capillary_length_m, h0_m)
                )
                # Exact axisymmetric normalization of a cubic bridge-side
                # arm over h0 and a linear film-side arm over ell_J.
                throat_area_m2 = 2.0 * math.pi * max(
                    (
                        material_rim_m * throat_width_m / 2.0
                        + throat_width_m**2 / 6.0
                        + 3.0 * material_rim_m * h0_m / 4.0
                        - 3.0 * h0_m**2 / 10.0
                    ),
                    1.0e-30,
                )
                transferred_after_connection_m3 = (
                    max(
                        float(
                            _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL
                        )
                        - float(
                            _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL
                        ),
                        0.0,
                    )
                    * 1.0e-9
                )
                throat_height_m = (
                    h0_m
                    - transferred_after_connection_m3 / throat_area_m2
                )
                mobile_reference_m = max(
                    h0_m - residual_height_m,
                    1.0e-30,
                )
                mobile_fraction = float(
                    np.clip(
                        (throat_height_m - residual_height_m)
                        / mobile_reference_m,
                        0.0,
                        1.0,
                    )
                )
                throat_mobility = mobile_fraction**3
                if bool(RESOLVE_TWO_SIDED_JUNCTION_LAYER):
                    # The throat is the same bridge-side arm that appears in
                    # the two-arm series resistance.  Its reduced mobility
                    # increases that arm's resistance by 1/m; multiplying a
                    # separate length factor by m would count it twice.
                    inner_to_outer_length = (
                        throat_width_m
                        / max(capillary_length_m, 1.0e-30)
                    )
                    combined_transmission = 1.0 / (
                        1.0
                        + inner_to_outer_length
                        / max(throat_mobility, 1.0e-30)
                    )
                else:
                    combined_transmission = throat_mobility
                far_flux_ul_s *= combined_transmission
                _PARTITIONED_JUNCTION_THROAT_WIDTH_M = throat_width_m
                _PARTITIONED_JUNCTION_THROAT_HEIGHT_M = throat_height_m
                _PARTITIONED_JUNCTION_THROAT_MOBILITY = throat_mobility
                _PARTITIONED_TWO_ARM_TRANSMISSION = (
                    combined_transmission
                )
            cap_volume_derivative_m2 = max(
                float(
                    case28.attached_cap_volume_derivative_m2(
                        config,
                        contact_radius_m,
                    )
                ),
                1.0e-30,
            )
            far_speed_cap_m_s = (
                far_flux_ul_s
                * 1.0e-9
                / cap_volume_derivative_m2
            )
            _PARTITIONED_FAR_SUPPLY_ACTIVATION = far_activation
            _PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M = diffusion_front_m
            _PARTITIONED_FAR_SUPPLY_FLUX_UL_S = far_flux_ul_s
            _PARTITIONED_FAR_SIMILARITY_FLUX_UL_S = (
                similarity_flux_ul_s
            )
            _PARTITIONED_FAR_DONOR_HEIGHT_M = mobile_film_height_m
            return (
                float(far_speed_cap_m_s),
                float(far_flux_ul_s),
                1.0,
                effective_local_capture_ul,
                float(film_inner_pressure_pa),
                float(film_next_pressure_pa),
            )
        return (
            0.0,
            0.0,
            1.0,
            effective_local_capture_ul,
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    remaining_supply_ul = max(
        float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - inventory_ul,
        0.0,
    )
    cap_volume_derivative_m2 = max(
        float(
            case28.attached_cap_volume_derivative_m2(
                config,
                contact_radius_m,
            )
        ),
        1.0e-30,
    )
    endpoint_speed_cap_m_s = (
        remaining_supply_ul
        * 1.0e-9
        / max(float(config.dt_s), 1.0e-30)
        / cap_volume_derivative_m2
    )
    effective_local_capture_ul = max(
        float(_PARTITIONED_FINITE_SUPPLY_CAP_UL)
        - max(float(parent._FLUX_SUPPLY_BUDGET_UL), 0.0),
        0.0,
    )
    return (
        float(min(max(speed_cap, 0.0), endpoint_speed_cap_m_s)),
        float(flux_ul_s),
        float(activation),
        effective_local_capture_ul,
        float(film_inner_pressure_pa),
        float(film_next_pressure_pa),
    )


def _configure_case62_namespace() -> None:
    """Place inherited kernels under an independent Case62 output namespace."""

    parent.CASE_STEM = CASE_STEM
    parent.CASE_LABEL = CASE_LABEL
    parent.OUTPUT_PREFIX = OUTPUT_PREFIX
    parent.OUT_DIR = OUT_DIR
    parent.TETRA_DIR_NAME = TETRA_DIR_NAME
    parent.CONFIG = CONFIG
    case28.CASE_STEM = CASE_STEM
    case28.CASE_LABEL = CASE_LABEL
    case28.OUTPUT_PREFIX = OUTPUT_PREFIX
    case28.OUT_DIR = OUT_DIR
    case28.TETRA_DIR_NAME = TETRA_DIR_NAME
    case28.base.CASE_STEM = CASE_STEM
    case28.base.CASE_LABEL = CASE_LABEL
    case28.base.OUTPUT_PREFIX = OUTPUT_PREFIX
    case28.base.OUT_DIR = OUT_DIR


def _far_volume_restoration(
    surface: np.ndarray,
    state: case28.TetraFreeSurfaceState,
    rings: np.ndarray,
    adjustable_rows: np.ndarray,
    target_volume_ul: float,
) -> tuple[float, float]:
    """Restore roundoff/remap volume with one unconstrained far-film shift.

    The surface-volume functional is linear in a uniform vertical shift while
    radial coordinates are fixed.  A two-point slope therefore gives the
    exact scalar correction without a height clip or experimental target.
    """

    rows = np.asarray(adjustable_rows, dtype=int)
    if rows.size == 0:
        residual = float(
            base.volume_under_mesh_ul(surface, state.surface_faces)
            - float(target_volume_ul)
        )
        return 0.0, residual
    ids = np.asarray(rings[rows].reshape(-1), dtype=int)
    volume0 = float(base.volume_under_mesh_ul(surface, state.surface_faces))
    probe_m = 1.0e-7
    probe = np.array(surface, copy=True)
    probe[ids, 2] += probe_m
    volume1 = float(base.volume_under_mesh_ul(probe, state.surface_faces))
    slope_ul_m = (volume1 - volume0) / probe_m
    if not math.isfinite(slope_ul_m) or abs(slope_ul_m) <= 1.0e-20:
        return 0.0, volume0 - float(target_volume_ul)
    shift_m = (float(target_volume_ul) - volume0) / slope_ul_m
    if float(np.min(surface[ids, 2] + shift_m)) <= FILM_MINIMUM_HEIGHT_M:
        # Leave an inadmissible candidate for the existing rollback controller
        # to reject.  Never clip an accepted physical film profile.
        surface[ids, 2] += shift_m
        return float(shift_m), float(
            base.volume_under_mesh_ul(surface, state.surface_faces)
            - float(target_volume_ul)
        )
    surface[ids, 2] += shift_m
    residual = float(
        base.volume_under_mesh_ul(surface, state.surface_faces)
        - float(target_volume_ul)
    )
    return float(shift_m), residual


def _previous_partitioned_state(
    state: case28.TetraFreeSurfaceState,
    current_radius_m: np.ndarray,
    current_height_m: np.ndarray,
    current_bridge_volume_ul: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the last accepted film and bridge state for a trial step."""

    if hasattr(state, "case62_accepted_film_radius_m"):
        radius = np.asarray(state.case62_accepted_film_radius_m, dtype=float)
        height = np.asarray(state.case62_accepted_film_height_m, dtype=float)
        bridge = float(state.case62_accepted_bridge_volume_ul)
        junction = float(state.case62_accepted_junction_radius_m)
        return radius, height, bridge, junction
    if _RESTART_FILM_RADIUS_M is not None:
        return (
            np.asarray(_RESTART_FILM_RADIUS_M, dtype=float),
            np.asarray(_RESTART_FILM_HEIGHT_M, dtype=float),
            float(_RESTART_BRIDGE_VOLUME_UL),
            float(_RESTART_JUNCTION_RADIUS_M),
        )
    return (
        np.asarray(current_radius_m, dtype=float),
        np.asarray(current_height_m, dtype=float),
        float(current_bridge_volume_ul),
        float(current_radius_m[0]),
    )


def _pr35_bridge_junction_pressure_pa(
    state: case28.TetraFreeSurfaceState,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> tuple[float | None, float, float, float]:
    """Return a gauge-invariant PR35 bridge pressure for the film boundary.

    PR35 pressure has an arbitrary additive gauge.  Subtract the far-film
    pressure trace and add the absolute hydrostatic pressure of the
    undisturbed film,

        p_J = rho*g*h0 + median(p_bridge - p_far).

    Azimuthal ring averages followed by medians suppress P1 pressure noise
    without modifying the solved pressure field.  A non-finite or unresolved
    trace disables pressure continuity for that step; no pressure is clipped.
    """

    pressure = np.asarray(
        getattr(state, "last_pressure_projection_pa", np.empty(0)),
        dtype=float,
    ).reshape(-1)
    n_surface = int(getattr(state, "n_surface", 0))
    if (
        pressure.size < n_surface
        or n_surface <= 0
        or np.any(~np.isfinite(pressure[:n_surface]))
    ):
        return None, float("nan"), float("nan"), float("nan")
    region = np.asarray(ring_region, dtype=int)
    bridge_rows = np.flatnonzero(region == 0)
    film_rows = np.flatnonzero(region == 1)
    if bridge_rows.size < 2 or film_rows.size < 5:
        return None, float("nan"), float("nan"), float("nan")
    ring_pressure = np.asarray(
        [
            float(np.mean(pressure[np.asarray(ring, dtype=int)]))
            for ring in np.asarray(rings, dtype=int)
        ],
        dtype=float,
    )
    bridge_sample = bridge_rows[1:-1]
    if bridge_sample.size == 0:
        bridge_sample = bridge_rows
    far_count = max(4, int(math.ceil(0.25 * film_rows.size)))
    far_sample = film_rows[-far_count:-1]
    if far_sample.size == 0:
        far_sample = film_rows[-far_count:]
    bridge_pressure = float(np.median(ring_pressure[bridge_sample]))
    far_pressure = float(np.median(ring_pressure[far_sample]))
    pressure_difference = bridge_pressure - far_pressure
    if not (
        math.isfinite(bridge_pressure)
        and math.isfinite(far_pressure)
        and math.isfinite(pressure_difference)
    ):
        return None, bridge_pressure, far_pressure, pressure_difference
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    absolute_junction_pressure = (
        float(config.density_kg_m3)
        * float(config.gravity_m_s2)
        * h0_m
        + pressure_difference
    )
    return (
        float(absolute_junction_pressure),
        bridge_pressure,
        far_pressure,
        pressure_difference,
    )


def _young_laplace_bridge_junction_pressure_pa(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float | None:
    """Return theoretical bridge liquid pressure on the film-PDE gauge.

    The zero-angle Young--Laplace equilibrium manifold is generated once from
    rho, g, gamma, sphere/substrate geometry and h0.  Thereafter the current
    conservative bridge inventory selects its pressure by scalar
    interpolation.  This removes ALE curvature noise without using a measured
    pressure, bridge trajectory, or fitted rate.  The Case77 film operator
    uses the positive suction convention, ``p_air - p_liquid``.
    """

    global _EQUILIBRIUM_BRIDGE_MANIFOLD

    if _EQUILIBRIUM_BRIDGE_MANIFOLD is None:
        _EQUILIBRIUM_BRIDGE_MANIFOLD = (
            build_fixed_sphere_equilibrium_bridge_manifold(
            sphere_radius_m=float(config.sphere_radius_mm) * 1.0e-3,
            sphere_tip_height_m=(
                float(config.sphere_bottom_z_mm) * 1.0e-3
            ),
            substrate_radius_m=float(config.substrate_radius_mm) * 1.0e-3,
            maximum_hypothetical_film_height_m=(
                float(config.initial_film_thickness_um) * 1.0e-6
            ),
            minimum_hypothetical_film_height_m=0.1e-6,
            density_kg_m3=float(config.density_kg_m3),
            gravity_m_s2=float(config.gravity_m_s2),
            surface_tension_n_m=float(config.surface_tension_n_m),
            samples=72,
            )
        )
    bridge_volume_m3 = (
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            ring_region,
            config,
        )
        * 1.0e-9
    )
    volume_grid = np.asarray(
        _EQUILIBRIUM_BRIDGE_MANIFOLD.bridge_volume_m3,
        dtype=float,
    )
    if (
        not math.isfinite(bridge_volume_m3)
        or bridge_volume_m3 < float(volume_grid[0])
        or bridge_volume_m3 > float(volume_grid[-1])
    ):
        return None
    suction_pressure_pa = float(
        np.interp(
            bridge_volume_m3,
            volume_grid,
            np.asarray(
                _EQUILIBRIUM_BRIDGE_MANIFOLD.suction_pressure_pa,
                dtype=float,
            ),
        )
    )
    # The fixed-sphere manifold stores the signed capillary pressure relative
    # to air (negative for suction in the present branch). Siekmann's radial
    # film equation uses liquid pressure along the substrate, hence
    # P(a)=rho*g*h0+S_signed.
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    liquid_pressure_pa = (
        float(config.density_kg_m3)
        * float(config.gravity_m_s2)
        * h0_m
        + suction_pressure_pa
    )
    return (
        float(liquid_pressure_pa)
        if math.isfinite(liquid_pressure_pa)
        else None
    )


def _young_laplace_bridge_contact_radius_m(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float | None:
    """Select the equilibrium contact radius from computed bridge volume."""

    # Build the same material/geometry manifold used by the junction-pressure
    # closure.  This call is cached after its first evaluation.
    _young_laplace_bridge_junction_pressure_pa(
        surface,
        rings,
        ring_region,
        config,
    )
    if _EQUILIBRIUM_BRIDGE_MANIFOLD is None:
        return None
    bridge_volume_m3 = (
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            ring_region,
            config,
        )
        * 1.0e-9
    )
    volume_grid = np.asarray(
        _EQUILIBRIUM_BRIDGE_MANIFOLD.bridge_volume_m3,
        dtype=float,
    )
    if (
        not math.isfinite(bridge_volume_m3)
        or bridge_volume_m3 < float(volume_grid[0])
        or bridge_volume_m3 > float(volume_grid[-1])
    ):
        return None
    contact_radius_m = float(
        np.interp(
            bridge_volume_m3,
            volume_grid,
            np.asarray(
                _EQUILIBRIUM_BRIDGE_MANIFOLD.contact_radius_m,
                dtype=float,
            ),
        )
    )
    return (
        contact_radius_m
        if math.isfinite(contact_radius_m)
        else None
    )


def _young_laplace_bridge_footprint_radius_m(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float | None:
    """Select the physical ``h=h0`` bridge footprint from current volume."""

    _young_laplace_bridge_junction_pressure_pa(
        surface,
        rings,
        ring_region,
        config,
    )
    if _EQUILIBRIUM_BRIDGE_MANIFOLD is None:
        return None
    bridge_volume_m3 = (
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            ring_region,
            config,
        )
        * 1.0e-9
    )
    volume_grid = np.asarray(
        _EQUILIBRIUM_BRIDGE_MANIFOLD.bridge_volume_m3,
        dtype=float,
    )
    if (
        not math.isfinite(bridge_volume_m3)
        or bridge_volume_m3 < float(volume_grid[0])
        or bridge_volume_m3 > float(volume_grid[-1])
    ):
        return None
    footprint_radius_m = float(
        np.interp(
            bridge_volume_m3,
            volume_grid,
            np.asarray(
                _EQUILIBRIUM_BRIDGE_MANIFOLD.footprint_radius_m,
                dtype=float,
            ),
        )
    )
    return (
        footprint_radius_m
        if math.isfinite(footprint_radius_m)
        else None
    )


def _axisymmetric_triangular_kernel_integral_m2(
    center_radius_m: float,
    inner_width_m: float,
    outer_width_m: float,
) -> float:
    """Return ``2*pi*integral(r*k(r) dr)`` for an asymmetric triangle."""

    center = float(center_radius_m)
    inner = max(float(inner_width_m), 1.0e-30)
    outer = max(float(outer_width_m), 1.0e-30)
    radial_integral_m2 = (
        0.5 * center * (inner + outer)
        + (outer**2 - inner**2) / 6.0
    )
    return 2.0 * math.pi * max(radial_integral_m2, 1.0e-30)


def _axisymmetric_triangular_kernel(
    radius_m: np.ndarray,
    center_radius_m: float,
    inner_width_m: float,
    outer_width_m: float,
) -> np.ndarray:
    """Evaluate the unit-height body-fitted capillary deficit kernel."""

    radius = np.asarray(radius_m, dtype=float)
    center = float(center_radius_m)
    inner = max(float(inner_width_m), 1.0e-30)
    outer = max(float(outer_width_m), 1.0e-30)
    kernel = np.zeros_like(radius)
    inner_mask = (radius >= center - inner) & (radius <= center)
    outer_mask = (radius > center) & (radius <= center + outer)
    kernel[inner_mask] = (
        radius[inner_mask] - (center - inner)
    ) / inner
    kernel[outer_mask] = 1.0 - (
        radius[outer_mask] - center
    ) / outer
    return np.clip(kernel, 0.0, 1.0)


def _pressure_closed_viscocapillary_deficit_profile(
    radius_m: np.ndarray,
    baseline_height_m: np.ndarray,
    *,
    center_radius_m: float,
    target_deficit_ul: float,
    contact_speed_m_s: float,
    hydraulic_rate_m3_s: float,
    film_pressure_pa: float,
    bridge_pressure_pa: float,
    surface_tension_n_m: float,
    viscosity_pa_s: float,
    slope_cube_multiplier: float = 1.0,
) -> tuple[np.ndarray, float, float, float]:
    """Close one conservative V-shaped junction from pressure and flux.

    Each arm is the local small-slope Young--Laplace/Cox form

        delta h = d - m*|r-r_J| - kappa*(r-r_J)^2/2,

    where ``m`` comes from the computed capillary number and ``kappa`` from
    the solved film-to-bridge pressure jump.  A scalar bisection chooses the
    depth ``d`` so the axisymmetric missing volume is exactly the accepted
    hydraulic deficit.  No target height, radius, width, or experimental
    profile enters this closure.
    """

    radius = np.asarray(radius_m, dtype=float)
    baseline = np.asarray(baseline_height_m, dtype=float)
    target_m3 = max(float(target_deficit_ul), 0.0) * 1.0e-9
    if target_m3 <= 1.0e-30:
        return baseline.copy(), 0.0, 0.0, 0.0
    center = float(center_radius_m)
    gamma = max(float(surface_tension_n_m), 1.0e-30)
    viscosity = max(float(viscosity_pa_s), 0.0)
    inner_speed = max(abs(float(contact_speed_m_s)), 1.0e-30)
    cube_multiplier = max(float(slope_cube_multiplier), 1.0e-12)
    inner_slope = (
        cube_multiplier * 3.0 * viscosity * inner_speed / gamma
    ) ** (1.0 / 3.0)
    curvature = max(
        (float(film_pressure_pa) - float(bridge_pressure_pa)) / gamma,
        1.0e-12,
    )
    baseline_center = float(np.interp(center, radius, baseline))
    maximum_depth = max(baseline_center - FILM_MINIMUM_HEIGHT_M, 0.0)

    def candidate(depth_m: float) -> tuple[np.ndarray, float, float]:
        tip_height = max(baseline_center - float(depth_m), FILM_MINIMUM_HEIGHT_M)
        radial_supply_speed = max(float(hydraulic_rate_m3_s), 0.0) / max(
            2.0 * math.pi * center * tip_height,
            1.0e-30,
        )
        outer_slope = (
            cube_multiplier
            * 3.0
            * viscosity
            * (inner_speed + radial_supply_speed)
            / gamma
        ) ** (1.0 / 3.0)
        offset = radius - center
        slope = np.where(offset <= 0.0, inner_slope, outer_slope)
        deficit = np.maximum(
            float(depth_m)
            - slope * np.abs(offset)
            - 0.5 * curvature * offset * offset,
            0.0,
        )
        profile = np.maximum(baseline - deficit, FILM_MINIMUM_HEIGHT_M)
        volume = float(
            2.0
            * math.pi
            * np.trapezoid(radius * (baseline - profile), radius)
        )
        return profile, volume, outer_slope

    # The outer Cox slope grows when the tip approaches zero thickness, so
    # capacity need not be monotone all the way to the positivity boundary.
    # Find the first physical root instead of testing only the endpoint.
    lower = 0.0
    upper = 0.0
    upper_profile = baseline.copy()
    upper_slope = inner_slope
    previous_depth = 0.0
    for trial_depth in np.linspace(0.0, maximum_depth, 65)[1:]:
        trial_profile, trial_volume, trial_slope = candidate(float(trial_depth))
        if trial_volume >= target_m3 * (1.0 - 1.0e-8):
            lower = previous_depth
            upper = float(trial_depth)
            upper_profile = trial_profile
            upper_slope = trial_slope
            break
        previous_depth = float(trial_depth)
    if upper <= lower:
        # During the initial inertial/contact burst the instantaneous Cox
        # slope can be too steep for a quasi-static pressure profile to hold
        # the accepted deficit.  Signal that the pressure closure is not yet
        # admissible; the caller retains the conservative transient profile.
        return baseline.copy(), -1.0, inner_slope, upper_slope
    selected_profile = upper_profile
    selected_outer_slope = upper_slope
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        profile, volume, outer_slope = candidate(middle)
        if volume < target_m3:
            lower = middle
        else:
            upper = middle
            selected_profile = profile
            selected_outer_slope = outer_slope
    depth = upper
    selected_profile, _volume, selected_outer_slope = candidate(depth)
    return selected_profile, depth, inner_slope, selected_outer_slope


def _conservative_c1_bridge_film_join_profile(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    split_index: int,
    start_slope: float,
    end_slope: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a monotone C1 join with separate bridge/film volume closure.

    A cubic Hermite profile supplies one tangent across the numerical region
    boundary.  Compact quartic bubbles have zero value and zero derivative at
    their endpoints; their two amplitudes therefore restore the original
    bridge-side and film-side axisymmetric volumes without breaking C1.
    """

    radius = np.asarray(radius_m, dtype=float)
    height = np.asarray(height_m, dtype=float)
    split = int(split_index)
    if radius.ndim != 1 or height.shape != radius.shape:
        raise ValueError("Case77 C1 join expects matching one-dimensional arrays")
    if radius.size < 6 or split < 2 or split > radius.size - 3:
        raise ValueError("Case77 C1 join requires resolved rows on both sides")
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(height)):
        raise ValueError("Case77 C1 join received a non-finite profile")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("Case77 C1 join radii must be strictly increasing")

    length = float(radius[-1] - radius[0])
    coordinate = (radius - radius[0]) / length
    cubic = (
        (2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0) * height[0]
        + (coordinate**3 - 2.0 * coordinate**2 + coordinate)
        * length
        * float(start_slope)
        + (-2.0 * coordinate**3 + 3.0 * coordinate**2) * height[-1]
        + (coordinate**3 - coordinate**2)
        * length
        * float(end_slope)
    )
    joined = np.asarray(cubic, dtype=float).copy()
    amplitudes: list[float] = []
    residuals_m3: list[float] = []
    for first, last in ((0, split), (split, radius.size - 1)):
        local_radius = radius[first : last + 1]
        local_coordinate = (
            (local_radius - local_radius[0])
            / (local_radius[-1] - local_radius[0])
        )
        bubble = local_coordinate**2 * (1.0 - local_coordinate) ** 2
        bubble_integral = float(
            np.trapezoid(local_radius * bubble, local_radius)
        )
        if abs(bubble_integral) <= 1.0e-30:
            raise ValueError("Case77 C1 join has an unresolved volume bubble")
        old_integral = float(
            np.trapezoid(
                local_radius * height[first : last + 1],
                local_radius,
            )
        )
        cubic_integral = float(
            np.trapezoid(
                local_radius * cubic[first : last + 1],
                local_radius,
            )
        )
        amplitude = (old_integral - cubic_integral) / bubble_integral
        joined[first : last + 1] += amplitude * bubble
        amplitudes.append(float(amplitude))
        residuals_m3.append(
            float(
                2.0
                * math.pi
                * (
                    np.trapezoid(
                        local_radius * joined[first : last + 1],
                        local_radius,
                    )
                    - old_integral
                )
            )
        )

    if np.any(~np.isfinite(joined)) or np.any(joined <= 0.0):
        raise RuntimeError("Case77 conservative C1 join produced an invalid gap")
    joined_difference = np.diff(joined)
    if height[-1] < height[0] and np.any(joined_difference > 1.0e-12):
        raise RuntimeError("Case77 conservative C1 join is not monotone")
    if height[-1] > height[0] and np.any(joined_difference < -1.0e-12):
        raise RuntimeError("Case77 conservative C1 join is not monotone")

    old_left_slope = float(
        (height[split] - height[split - 1])
        / (radius[split] - radius[split - 1])
    )
    old_right_slope = float(
        (height[split + 1] - height[split])
        / (radius[split + 1] - radius[split])
    )
    new_left_slope = float(
        (joined[split] - joined[split - 1])
        / (radius[split] - radius[split - 1])
    )
    new_right_slope = float(
        (joined[split + 1] - joined[split])
        / (radius[split + 1] - radius[split])
    )
    return joined, {
        "bridge_amplitude_m": float(amplitudes[0]),
        "film_amplitude_m": float(amplitudes[1]),
        "bridge_volume_residual_m3": float(residuals_m3[0]),
        "film_volume_residual_m3": float(residuals_m3[1]),
        "prejoin_angle_jump_rad": float(
            abs(math.atan(old_right_slope) - math.atan(old_left_slope))
        ),
        "postjoin_p1_angle_jump_rad": float(
            abs(math.atan(new_right_slope) - math.atan(new_left_slope))
        ),
    }


def _conservative_monotone_c1_join_profile(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    split_index: int,
    start_slope: float,
    end_slope: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return one monotone C1 join conserving its total annular volume.

    The bridge/film split is a numerical region boundary, not a sealed
    material boundary.  One compact quartic bubble therefore restores the
    combined ``2*pi*integral(r*h dr)`` while the cubic Hermite endpoints retain
    the neighboring solved tangents.
    """

    radius = np.asarray(radius_m, dtype=float)
    height = np.asarray(height_m, dtype=float)
    split = int(split_index)
    if radius.ndim != 1 or height.shape != radius.shape:
        raise ValueError("Case77 outer C1 join expects one-dimensional arrays")
    if radius.size < 5 or split < 1 or split > radius.size - 2:
        raise ValueError("Case77 outer C1 join requires a resolved split")
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(height)):
        raise ValueError("Case77 outer C1 join received non-finite data")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("Case77 outer C1 join radii must increase")

    length = float(radius[-1] - radius[0])
    coordinate = (radius - radius[0]) / length
    cubic = (
        (2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0) * height[0]
        + (coordinate**3 - 2.0 * coordinate**2 + coordinate)
        * length
        * float(start_slope)
        + (-2.0 * coordinate**3 + 3.0 * coordinate**2) * height[-1]
        + (coordinate**3 - coordinate**2)
        * length
        * float(end_slope)
    )
    bubble = coordinate**2 * (1.0 - coordinate) ** 2
    bubble_integral = float(np.trapezoid(radius * bubble, radius))
    if abs(bubble_integral) <= 1.0e-30:
        raise ValueError("Case77 outer C1 volume bubble is unresolved")
    old_integral = float(np.trapezoid(radius * height, radius))
    cubic_integral = float(np.trapezoid(radius * cubic, radius))
    amplitude = (old_integral - cubic_integral) / bubble_integral
    joined = np.asarray(cubic + amplitude * bubble, dtype=float)
    if np.any(~np.isfinite(joined)) or np.any(joined <= 0.0):
        raise RuntimeError("Case77 outer C1 join produced an invalid gap")
    joined_difference = np.diff(joined)
    if joined[-1] >= joined[0]:
        monotone = bool(np.all(joined_difference >= -1.0e-12))
    else:
        monotone = bool(np.all(joined_difference <= 1.0e-12))
    if not monotone:
        raise RuntimeError("Case77 outer C1 join is not monotone")

    old_left_slope = float(
        (height[split] - height[split - 1])
        / (radius[split] - radius[split - 1])
    )
    old_right_slope = float(
        (height[split + 1] - height[split])
        / (radius[split + 1] - radius[split])
    )
    new_left_slope = float(
        (joined[split] - joined[split - 1])
        / (radius[split] - radius[split - 1])
    )
    new_right_slope = float(
        (joined[split + 1] - joined[split])
        / (radius[split + 1] - radius[split])
    )
    volume_residual_m3 = float(
        2.0
        * math.pi
        * (np.trapezoid(radius * joined, radius) - old_integral)
    )
    return joined, {
        "amplitude_m": float(amplitude),
        "volume_residual_m3": volume_residual_m3,
        "prejoin_angle_jump_rad": float(
            abs(math.atan(old_right_slope) - math.atan(old_left_slope))
        ),
        "postjoin_p1_angle_jump_rad": float(
            abs(math.atan(new_right_slope) - math.atan(new_left_slope))
        ),
    }


def _finite_substrate_film_profile_m(
    radius_m: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> np.ndarray:
    """Return the paper's finite circular-substrate equilibrium film."""

    radius = np.asarray(radius_m, dtype=float)
    outer_radius_m = float(config.substrate_radius_mm) * 1.0e-3
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    capillary_length_m = math.sqrt(
        float(config.surface_tension_n_m)
        / max(
            float(config.density_kg_m3)
            * float(config.gravity_m_s2),
            1.0e-30,
        )
    )
    outer_i0 = float(np.i0(outer_radius_m / capillary_length_m))
    clipped_radius = np.clip(radius, 0.0, outer_radius_m)
    profile = (
        h0_m
        * (
            outer_i0
            - np.i0(clipped_radius / capillary_length_m)
        )
        / max(outer_i0 - 1.0, 1.0e-30)
    )
    return np.maximum(profile, FILM_MINIMUM_HEIGHT_M)


def _moving_pressure_film_grid(
    footprint_radius_m: float,
    count: int,
    config: base.RealMeshEvolutionConfig,
) -> np.ndarray:
    """Return Siekman's sinh-refined moving film grid."""

    outer_radius_m = float(config.substrate_radius_mm) * 1.0e-3
    footprint = float(footprint_radius_m)
    if not 0.0 < footprint < outer_radius_m:
        raise RuntimeError("Physical film footprint left the substrate")
    grid_count = max(int(count), 16)
    concentration = 3.0
    coordinate = np.sinh(
        concentration * np.linspace(0.0, 1.0, grid_count)
    ) / math.sinh(concentration)
    return footprint + (outer_radius_m - footprint) * coordinate


def _previous_physical_film_state(
    state: case28.TetraFreeSurfaceState,
    footprint_radius_m: float,
    count: int,
    config: base.RealMeshEvolutionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the last accepted finite-rim film state."""

    if hasattr(state, "case77_accepted_physical_film_radius_m"):
        return (
            np.asarray(
                state.case77_accepted_physical_film_radius_m,
                dtype=float,
            ).copy(),
            np.asarray(
                state.case77_accepted_physical_film_height_m,
                dtype=float,
            ).copy(),
        )
    if (
        _RESTART_PHYSICAL_FILM_RADIUS_M is not None
        and _RESTART_PHYSICAL_FILM_HEIGHT_M is not None
    ):
        return (
            np.asarray(_RESTART_PHYSICAL_FILM_RADIUS_M, dtype=float).copy(),
            np.asarray(_RESTART_PHYSICAL_FILM_HEIGHT_M, dtype=float).copy(),
        )
    radius = _moving_pressure_film_grid(
        footprint_radius_m,
        count,
        config,
    )
    height = _finite_substrate_film_profile_m(radius, config)
    height[0] = float(config.initial_film_thickness_um) * 1.0e-6
    height[-1] = FILM_MINIMUM_HEIGHT_M
    return radius, height


def _advance_moving_boundary_pressure_film(
    state: case28.TetraFreeSurfaceState,
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    *,
    footprint_radius_m: float,
    film_ring_count: int,
    bridge_supply_rate_m3_s: float,
    bridge_pressure_pa: float | None,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Advance the finite-rim pressure film and return its next supply flux.

    Before PR33 resolves a finite bridge patch, the accepted first-contact
    transfer is imposed as the conservative inner flux.  Thereafter the exact
    PR33 Heron traction pressure is imposed at the moving footprint and the
    film solve returns ``Q_J``.  Both modes use the same backward-Euler
    lubrication equation and Reynolds moving-grid conservation law.
    """

    previous_pressure_film_available = bool(
        hasattr(state, "case77_accepted_physical_film_radius_m")
        or (
            _RESTART_PHYSICAL_FILM_RADIUS_M is not None
            and _RESTART_PHYSICAL_FILM_HEIGHT_M is not None
        )
    )
    old_radius_m, old_height_m = _previous_physical_film_state(
        state,
        footprint_radius_m,
        film_ring_count,
        config,
    )
    if not previous_pressure_film_available:
        # At the one-way handoff, initialize the radial lubrication block
        # from the accepted depleted Case77 film.  Starting it from a fresh
        # h=h0 profile silently erased the already transported film deficit
        # and produced an artificial burst in Q_J.
        row_radius_m, row_height_m = parent._ring_coordinates(
            surface,
            rings,
        )
        film_rows = np.flatnonzero(
            np.asarray(ring_region, dtype=int) == 1
        )
        valid_rows = film_rows[
            np.isfinite(row_radius_m[film_rows])
            & np.isfinite(row_height_m[film_rows])
            & (row_radius_m[film_rows] >= footprint_radius_m)
        ]
        if valid_rows.size >= 2:
            order = np.argsort(row_radius_m[valid_rows])
            source_radius_m = np.asarray(
                row_radius_m[valid_rows][order],
                dtype=float,
            )
            source_height_m = np.maximum(
                np.asarray(
                    row_height_m[valid_rows][order],
                    dtype=float,
                ),
                FILM_MINIMUM_HEIGHT_M,
            )
            unique_radius_m, unique_index = np.unique(
                source_radius_m,
                return_index=True,
            )
            source_height_m = source_height_m[unique_index]
            source_radius_m = unique_radius_m
            h0_m = (
                float(config.initial_film_thickness_um) * 1.0e-6
            )
            source_radius_m = np.insert(
                source_radius_m,
                0,
                float(footprint_radius_m),
            )
            source_height_m = np.insert(source_height_m, 0, h0_m)
            old_height_m = np.interp(
                old_radius_m,
                source_radius_m,
                source_height_m,
            )
            old_height_m[0] = h0_m
            old_height_m[-1] = FILM_MINIMUM_HEIGHT_M
    new_radius_m = _moving_pressure_film_grid(
        footprint_radius_m,
        len(old_radius_m),
        config,
    )
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    if dt_s <= 0.0:
        return (
            old_radius_m,
            old_height_m,
            float(_RESTART_PHYSICAL_FILM_FLUX_UL_S),
            {
                "case77_pressure_film_hmin_um": float(
                    np.min(old_height_m) * 1.0e6
                ),
                "case77_pressure_film_rmin_mm": float(
                    old_radius_m[int(np.argmin(old_height_m))] * 1.0e3
                ),
                "case77_pressure_film_bridge_pressure_pa": float("nan"),
                "case77_pressure_film_flux_ul_s": float(
                    _RESTART_PHYSICAL_FILM_FLUX_UL_S
                ),
                "case77_pressure_film_boundary_mode": 0.0,
            },
        )

    pressure_boundary_active = bridge_pressure_pa is not None
    prescribed_inner_flux_m2_s = (
        -max(float(bridge_supply_rate_m3_s), 0.0)
        / max(
            2.0 * math.pi * float(new_radius_m[0]),
            1.0e-30,
        )
    )
    junction_resistance_pa_s_m2 = 0.0
    geometric_junction_resistance_pa_s_m2 = 0.0
    reference_junction_resistance_pa_s_m2 = float(
        getattr(
            state,
            "case77_accepted_junction_impedance_pa_s_m2",
            _RESTART_JUNCTION_IMPEDANCE_PA_S_M2,
        )
    )
    reference_junction_height_m = float(
        getattr(
            state,
            "case77_accepted_junction_impedance_reference_height_m",
            _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M,
        )
    )
    reference_junction_width_m = float(
        getattr(
            state,
            "case77_accepted_junction_impedance_reference_width_m",
            _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M,
        )
    )
    junction_height_m = h0_m
    junction_width_m = 0.0
    cox_wedge_resistance_factor = 1.0
    if pressure_boundary_active:
        capillary_length_m = math.sqrt(
            float(config.surface_tension_n_m)
            / max(
                float(config.density_kg_m3)
                * float(config.gravity_m_s2),
                1.0e-30,
            )
        )
        junction_width_m = math.sqrt(h0_m * capillary_length_m)
        junction_window = (
            np.asarray(old_radius_m, dtype=float)
            <= float(old_radius_m[0]) + capillary_length_m
        )
        if np.any(junction_window):
            junction_height_m = float(
                np.min(
                    np.asarray(old_height_m, dtype=float)[junction_window]
                )
            )
        junction_height_m = max(
            junction_height_m,
            FILM_MINIMUM_HEIGHT_M,
        )
        # The axisymmetric film grid begins outside the unresolved
        # bridge--film turning region.  Its depth-averaged free-surface
        # Poiseuille resistance per circumferential width is
        #
        #   R'_J = 3*mu*ell_J/h_J^3,
        #   ell_J = sqrt(h0*ell_c).
        #
        # This is the gap-dependent K_lub contribution in series with the
        # first resolved film face.  Every quantity is read from the accepted
        # state or material properties; no target flux or validation curve is
        # used.
        geometric_junction_resistance_pa_s_m2 = (
            3.0
            * float(config.viscosity_pa_s)
            * junction_width_m
            / junction_height_m**3
        )
        # PR37 already places the Cox logarithmic wedge dissipation in the
        # accepted contact-line force and velocity.  The physical film block
        # separately resolves its Poiseuille mobility and the geometric
        # turning-region resistance above.  Multiplying that impedance by a
        # second Cox logarithm would count the same contact-line dissipation
        # twice and destroy flux continuity at the domain-decomposition
        # handoff.  Keep the audit field, but its correct additional factor is
        # therefore exactly one.
        cox_wedge_resistance_factor = 1.0
        if (
            reference_junction_resistance_pa_s_m2 > 0.0
            and reference_junction_height_m > 0.0
            and reference_junction_width_m > 0.0
        ):
            # Once identified at the one-way handoff, the unresolved
            # bridge-side impedance follows the same Poiseuille scaling as
            # K_lub: R'_J proportional to ell_J/h_J^3.  This changes it only
            # through the accepted geometry, not elapsed time or validation
            # data.
            junction_resistance_pa_s_m2 = (
                reference_junction_resistance_pa_s_m2
                * junction_width_m
                / reference_junction_width_m
                * (reference_junction_height_m / junction_height_m) ** 3
            )
        else:
            junction_resistance_pa_s_m2 = (
                geometric_junction_resistance_pa_s_m2
            )

    def solve_with_junction_resistance(
        resistance_pa_s_m2: float,
    ) -> tuple[np.ndarray, float, float]:
        # The PR35/ALE continuation can use a multi-second low-Ca step, while
        # the nonlinear fourth-order film operator still requires a shorter
        # integration interval for temporal convergence.  Subcycle only that
        # operator and move its radial grid linearly between the two accepted
        # contact radii.  This is timestep refinement, not a flux limiter: all
        # substeps solve the same conservative backward-Euler equation.
        film_substeps = max(
            int(
                math.ceil(
                    float(dt_s)
                    / max(float(FILM_MAX_PHYSICAL_SUBSTEP_S), 1.0e-30)
                )
            ),
            1,
        )
        film_dt_s = float(dt_s) / float(film_substeps)
        substep_height_m = np.asarray(old_height_m, dtype=float).copy()
        substep_reference_radius_m = np.asarray(
            old_radius_m,
            dtype=float,
        ).copy()
        solved_inner_pressure_pa = 0.0
        film_rate_m3_s = 0.0
        for substep_index in range(1, film_substeps + 1):
            fraction = float(substep_index) / float(film_substeps)
            substep_radius_m = (
                np.asarray(old_radius_m, dtype=float)
                + fraction
                * (
                    np.asarray(new_radius_m, dtype=float)
                    - np.asarray(old_radius_m, dtype=float)
                )
            )
            (
                substep_height_m,
                solved_inner_pressure_pa,
                film_rate_m3_s,
            ) = _implicit_nonlinear_film_flux_step(
                substep_radius_m,
                substep_height_m,
                inner_flux_m2_s=(
                    0.0
                    if pressure_boundary_active
                    else prescribed_inner_flux_m2_s
                ),
                inner_pressure_pa=(
                    float(bridge_pressure_pa)
                    if pressure_boundary_active
                    else None
                ),
                inner_hydraulic_resistance_pa_s_m2=(
                    float(resistance_pa_s_m2)
                ),
                dt_s=film_dt_s,
                surface_tension_n_m=float(config.surface_tension_n_m),
                viscosity_pa_s=float(config.viscosity_pa_s),
                nonlinear_max_iterations=80,
                nonlinear_rtol=1.0e-9,
                density_kg_m3=float(config.density_kg_m3),
                gravity_m_s2=float(config.gravity_m_s2),
                full_curvature=True,
                # The analytic Bessel rim and the P1 moving-grid inventory use
                # different, convergent quadratures.  Their zero-flux
                # discrepancy is O(1e-15) m3/s on the 112-node grid; this is a
                # numerical acceptance tolerance, not a physical source or
                # rate cap.
                flux_conservation_absolute_tolerance_m3_s=2.0e-15,
                reference_radius_m=substep_reference_radius_m,
                film_thickness_m=h0_m,
                final_inner_height_m=h0_m,
                final_outer_height_m=FILM_MINIMUM_HEIGHT_M,
            )
            substep_reference_radius_m = np.asarray(
                substep_radius_m,
                dtype=float,
            )
        return (
            np.asarray(substep_height_m, dtype=float),
            float(solved_inner_pressure_pa),
            float(film_rate_m3_s),
        )
    solved_height_m, solved_inner_pressure_pa, film_rate_m3_s = (
        solve_with_junction_resistance(junction_resistance_pa_s_m2)
    )

    impedance_identified_at_handoff = False
    if (
        pressure_boundary_active
        and reference_junction_resistance_pa_s_m2 <= 0.0
    ):
        # The local-donor and resolved-film stages describe the same physical
        # junction on opposite sides of one domain-decomposition handoff.
        # Determine the unresolved bridge-side impedance by requiring the
        # first pressure-driven flux to equal the last accepted conservative
        # flux.  This is a continuity condition, not a calibration target:
        # both quantities come from the current forward state.
        accepted_diag = dict(
            getattr(state, "case62_last_accepted_diag", {})
        )
        # ``case62_last_accepted_diag`` is reconstructed lazily after a
        # checkpoint restart and can temporarily contain a zero trial flux.
        # Preserve restart invariance by taking the largest *accepted*
        # pre-handoff flux available from the live state or checkpoint.  This
        # is the same computed continuity datum in both paths; it is not an
        # experimental target or a rate cap.
        target_flux_ul_s = max(
            float(accepted_diag.get("mass_supply_flux_ul_s", 0.0)),
            float(_RESTART_ACCEPTED_MASS_SUPPLY_FLUX_UL_S),
            0.0,
        )
        if target_flux_ul_s <= 1.0e-14:
            target_flux_ul_s = max(
                float(
                    getattr(
                        state,
                        "case77_accepted_physical_film_flux_ul_s",
                        _RESTART_PHYSICAL_FILM_FLUX_UL_S,
                    )
                ),
                float(
                    getattr(
                        state,
                        "case62_accepted_hydraulic_rate_ul_s",
                        _RESTART_HYDRAULIC_RATE_UL_S,
                    )
                ),
                0.0,
            )
        solved_flux_ul_s = max(-float(film_rate_m3_s) * 1.0e9, 0.0)
        if target_flux_ul_s > 1.0e-14 and (
            solved_flux_ul_s
            > target_flux_ul_s * (1.0 + 1.0e-3)
        ):
            lower_resistance = max(
                float(junction_resistance_pa_s_m2),
                np.finfo(float).tiny,
            )
            upper_resistance = lower_resistance
            upper_solution = (
                solved_height_m,
                solved_inner_pressure_pa,
                film_rate_m3_s,
            )
            upper_flux_ul_s = solved_flux_ul_s
            for _ in range(24):
                upper_resistance *= 4.0
                upper_solution = solve_with_junction_resistance(
                    upper_resistance
                )
                upper_flux_ul_s = max(
                    -float(upper_solution[2]) * 1.0e9,
                    0.0,
                )
                if upper_flux_ul_s <= target_flux_ul_s:
                    break
            if upper_flux_ul_s > target_flux_ul_s:
                raise RuntimeError(
                    "Case77 could not identify a finite junction impedance "
                    "that preserves flux continuity at film handoff"
                )
            lower_log = math.log(lower_resistance)
            upper_log = math.log(upper_resistance)
            for _ in range(18):
                middle_resistance = math.exp(
                    0.5 * (lower_log + upper_log)
                )
                middle_solution = solve_with_junction_resistance(
                    middle_resistance
                )
                middle_flux_ul_s = max(
                    -float(middle_solution[2]) * 1.0e9,
                    0.0,
                )
                if middle_flux_ul_s > target_flux_ul_s:
                    lower_log = math.log(middle_resistance)
                else:
                    upper_log = math.log(middle_resistance)
                    upper_solution = middle_solution
                if (
                    abs(middle_flux_ul_s - target_flux_ul_s)
                    <= max(1.0e-6 * target_flux_ul_s, 1.0e-12)
                ):
                    upper_resistance = middle_resistance
                    upper_solution = middle_solution
                    break
            else:
                upper_resistance = math.exp(upper_log)
            junction_resistance_pa_s_m2 = float(upper_resistance)
            (
                solved_height_m,
                solved_inner_pressure_pa,
                film_rate_m3_s,
            ) = upper_solution
            (
                solved_height_m,
                solved_inner_pressure_pa,
                film_rate_m3_s,
            ) = solve_with_junction_resistance(
                junction_resistance_pa_s_m2
            )
            reference_junction_resistance_pa_s_m2 = float(
                junction_resistance_pa_s_m2
            )
            reference_junction_height_m = float(junction_height_m)
            reference_junction_width_m = float(junction_width_m)
            impedance_identified_at_handoff = True
        elif (
            target_flux_ul_s > 1.0e-14
            and solved_flux_ul_s
            >= target_flux_ul_s * (1.0 - 1.0e-3)
        ):
            # The geometric resistance already makes the two descriptions
            # continuous to solver tolerance.
            (
                solved_height_m,
                solved_inner_pressure_pa,
                film_rate_m3_s,
            ) = solve_with_junction_resistance(
                junction_resistance_pa_s_m2
            )
            reference_junction_resistance_pa_s_m2 = float(
                junction_resistance_pa_s_m2
            )
            reference_junction_height_m = float(junction_height_m)
            reference_junction_width_m = float(junction_width_m)
            impedance_identified_at_handoff = True

    next_flux_ul_s = max(-float(film_rate_m3_s) * 1.0e9, 0.0)
    minimum_index = int(np.argmin(solved_height_m))
    diagnostics = {
        "case77_pressure_film_hmin_um": float(
            solved_height_m[minimum_index] * 1.0e6
        ),
        "case77_pressure_film_rmin_mm": float(
            new_radius_m[minimum_index] * 1.0e3
        ),
        "case77_pressure_film_bridge_pressure_pa": float(
            bridge_pressure_pa
            if bridge_pressure_pa is not None
            else solved_inner_pressure_pa
        ),
        "case77_pressure_film_flux_ul_s": float(next_flux_ul_s),
        "case77_pressure_film_boundary_mode": float(
            pressure_boundary_active
        ),
        "case77_pressure_film_junction_height_um": float(
            junction_height_m * 1.0e6
        ),
        "case77_pressure_film_junction_width_mm": float(
            junction_width_m * 1.0e3
        ),
        "case77_pressure_film_junction_resistance_pa_s_m2": float(
            junction_resistance_pa_s_m2
        ),
        "case77_pressure_film_geometric_resistance_pa_s_m2": float(
            geometric_junction_resistance_pa_s_m2
        ),
        "case77_pressure_film_reference_resistance_pa_s_m2": float(
            reference_junction_resistance_pa_s_m2
        ),
        "case77_pressure_film_reference_height_um": float(
            reference_junction_height_m * 1.0e6
        ),
        "case77_pressure_film_reference_width_mm": float(
            reference_junction_width_m * 1.0e3
        ),
        "case77_pressure_film_impedance_identified_at_handoff": float(
            impedance_identified_at_handoff
        ),
        "case77_pressure_film_two_sided_cox_wedge_factor": float(
            cox_wedge_resistance_factor
        ),
        "case77_pressure_film_substeps": float(
            max(
                int(
                    math.ceil(
                        float(dt_s)
                        / max(
                            float(FILM_MAX_PHYSICAL_SUBSTEP_S),
                            1.0e-30,
                        )
                    )
                ),
                1,
            )
        ),
    }
    return (
        np.asarray(new_radius_m, dtype=float),
        np.asarray(solved_height_m, dtype=float),
        float(next_flux_ul_s),
        diagnostics,
    )


def _heron_bridge_junction_pressure_pa(
    state: case28.TetraFreeSurfaceState,
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float | None:
    """Return the exact PR33 bridge-patch traction as a pressure.

    The force is evaluated by the same vectorized Heron operator used in
    PR35.  Restricting it to faces whose vertices all belong to the bridge
    removes the outer-film curvature and leaves an absolute liquid-to-air
    capillary pressure.  No pressure multiplier, fitted curvature, or
    experimental profile enters the calculation.
    """

    bridge_rows = np.flatnonzero(
        np.asarray(ring_region, dtype=int) == 0
    )
    if bridge_rows.size < 2:
        return None
    bridge_vertices = np.unique(
        np.asarray(rings[bridge_rows], dtype=int).reshape(-1)
    )
    is_bridge = np.zeros(len(surface), dtype=bool)
    is_bridge[bridge_vertices] = True
    surface_faces = np.asarray(state.surface_faces, dtype=int).reshape((-1, 3))
    bridge_faces = surface_faces[
        np.all(is_bridge[surface_faces], axis=1)
    ]
    if bridge_faces.size == 0:
        return None
    _force, _area, pressure_pa = (
        case28.cotangent_surface_tension_forces(
            np.asarray(surface, dtype=float),
            bridge_faces,
            float(config.surface_tension_n_m),
        )
    )
    if not math.isfinite(float(pressure_pa)):
        return None
    # The film operator uses p_f=-gamma*kappa+rho*g*h.  Put the bridge
    # traction on that same hydrostatic gauge; using the capillary term alone
    # would impose an artificial pressure drop of rho*g*h0.
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    return float(
        pressure_pa
        + float(config.density_kg_m3)
        * float(config.gravity_m_s2)
        * h0_m
    )


def _pressure_driven_junction_flux_ul_s(
    state: case28.TetraFreeSurfaceState,
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> tuple[float, float, float, float, float, float, float]:
    """Calculate inward supply from Young--Laplace pressure and film resistance.

    The radial film is a series hydraulic resistance,

        R_f = integral 3*mu/(2*pi*r*h**3) dr,
        Q_J = max((p_donor - p_bridge)/R_f, 0).

    Film pressure and curvature are evaluated on Case61's fixed physical
    reconstruction grid so changing the ALE ring spacing does not change the
    constitutive flux.  The donor pressure is the median resolved far-film
    plateau and the complete junction-to-plateau resistance is retained; no
    fitted length or experimental input enters the closure.
    """

    (
        pr35_bridge_pressure_pa,
        pr35_far_film_pressure_pa,
        _pr35_bridge_minus_far_pressure_pa,
    ) = _pr35_bridge_junction_pressure_pa(
        state,
        rings,
        ring_region,
        config,
    )[1:]
    bridge_pressure_pa = _young_laplace_bridge_junction_pressure_pa(
        surface,
        rings,
        ring_region,
        config,
    )
    if bridge_pressure_pa is None:
        return (
            0.0,
            float("nan"),
            float("nan"),
            float("inf"),
            float("nan"),
            float(pr35_bridge_pressure_pa),
            float(pr35_far_film_pressure_pa),
        )

    flux_surface = np.asarray(surface, dtype=float)
    if (
        bool(USE_SUBGRID_JUNCTION_FLUX_CLOSURE)
        and parent._FLUX_REFERENCE_SURFACE is not None
        and np.asarray(parent._FLUX_REFERENCE_SURFACE).shape
        == flux_surface.shape
        and np.all(np.isfinite(parent._FLUX_REFERENCE_SURFACE))
    ):
        flux_surface = np.asarray(
            parent._FLUX_REFERENCE_SURFACE,
            dtype=float,
        )
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    row_radius_m, row_height_m = parent._ring_coordinates(
        flux_surface,
        rings,
    )
    rows = film_rows[np.argsort(row_radius_m[film_rows])]
    radius_raw = np.asarray(row_radius_m[rows], dtype=float)
    height_raw = np.asarray(row_height_m[rows], dtype=float)
    if (
        rows.size < 5
        or np.any(~np.isfinite(radius_raw))
        or np.any(~np.isfinite(height_raw))
        or np.any(np.diff(radius_raw) <= 0.0)
        or np.any(height_raw < 0.0)
    ):
        return (
            0.0,
            float(bridge_pressure_pa),
            float("nan"),
            float("inf"),
            float("nan"),
            float(pr35_bridge_pressure_pa),
            float(pr35_far_film_pressure_pa),
        )

    spacing_m = float(parent.FLUX_RECONSTRUCTION_SPACING_M)
    sample_count = max(
        7,
        int(
            math.floor(
                (float(radius_raw[-1]) - float(radius_raw[0]))
                / max(spacing_m, 1.0e-30)
            )
        )
        + 1,
    )
    radius_m = np.linspace(
        float(radius_raw[0]),
        float(radius_raw[-1]),
        sample_count,
    )
    height_m = np.interp(radius_m, radius_raw, height_raw)
    window = max(
        5,
        int(
            round(
                float(parent.FLUX_CURVATURE_FILTER_WIDTH_M)
                / max(spacing_m, 1.0e-30)
            )
        ),
    )
    if window % 2 == 0:
        window += 1
    window = min(
        window,
        sample_count if sample_count % 2 == 1 else sample_count - 1,
    )
    if window >= 5:
        height_m = savgol_filter(
            height_m,
            window_length=window,
            polyorder=2,
            mode="interp",
        )
    first_derivative = np.gradient(height_m, radius_m, edge_order=2)
    second_derivative = np.gradient(
        first_derivative,
        radius_m,
        edge_order=2,
    )
    film_pressure_pa = (
        -float(config.surface_tension_n_m)
        * (
            second_derivative
            + first_derivative / np.maximum(radius_m, 1.0e-12)
        )
        + float(config.density_kg_m3)
        * float(config.gravity_m_s2)
        * height_m
    )
    # The donor is the nearest undisturbed plateau, not the physical h(L)=0
    # substrate-rim boundary. Select the first point beyond the dimple where
    # h has recovered to 90% of h0, then average a short plateau window.
    minimum_index = int(np.argmin(height_m))
    plateau_candidates = np.flatnonzero(
        (np.arange(sample_count) > minimum_index)
        & (height_m >= 0.9 * float(config.initial_film_thickness_um) * 1.0e-6)
    )
    if plateau_candidates.size == 0:
        return (
            0.0,
            float(bridge_pressure_pa),
            float("nan"),
            float("inf"),
            float(np.min(height_m) * 1.0e6),
            float(pr35_bridge_pressure_pa),
            float(pr35_far_film_pressure_pa),
        )
    source_index = int(plateau_candidates[0])
    plateau_stop = min(source_index + max(4, sample_count // 20), sample_count)
    donor_pressure_pa = float(
        np.median(film_pressure_pa[source_index:plateau_stop])
    )
    pressure_drop_pa = donor_pressure_pa - float(bridge_pressure_pa)
    if source_index <= 0 or pressure_drop_pa <= 0.0:
        return (
            0.0,
            float(bridge_pressure_pa),
            donor_pressure_pa,
            float("inf"),
            float(np.min(height_m) * 1.0e6),
            float(pr35_bridge_pressure_pa),
            float(pr35_far_film_pressure_pa),
        )

    radius_face_m = 0.5 * (
        radius_m[:source_index] + radius_m[1 : source_index + 1]
    )
    left_mobility = height_m[:source_index] ** 3
    right_mobility = height_m[1 : source_index + 1] ** 3
    harmonic_height_cubed = (
        2.0
        * left_mobility
        * right_mobility
        / np.maximum(left_mobility + right_mobility, 1.0e-300)
    )
    conductance_m3_pa_s = (
        2.0
        * math.pi
        * radius_face_m
        * harmonic_height_cubed
        / (
            3.0
            * float(config.viscosity_pa_s)
            * float(case28.WALL_LUBRICATION_DRAG_FACTOR)
        )
    )
    resistance_pa_s_m3 = float(
        np.sum(
            np.diff(radius_m[: source_index + 1])
            / np.maximum(conductance_m3_pa_s, 1.0e-300)
        )
    )
    flux_ul_s = (
        pressure_drop_pa
        / max(resistance_pa_s_m3, 1.0e-300)
        * 1.0e9
    )
    return (
        float(max(flux_ul_s, 0.0)),
        float(bridge_pressure_pa),
        donor_pressure_pa,
        resistance_pa_s_m3,
        float(np.min(height_m) * 1.0e6),
        float(pr35_bridge_pressure_pa),
        float(pr35_far_film_pressure_pa),
    )


def adaptive_partitioned_film_reprojection(
    state: case28.TetraFreeSurfaceState,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    time_s: float | None = None,
    dt_s: float = 0.0,
) -> None:
    """Tangential radial ALE plus one conservative local film solve."""

    retained_supply_surface: np.ndarray | None = None
    initial_surface = state.surface_points()
    initial_bridge_inventory_ul = float(
        parent.bridge_inventory_volume_ul(
            initial_surface,
            state.rings,
            ring_region,
            config,
        )
    )
    supply_already_saturated = bool(
        USE_TERMINAL_LOCAL_DONOR_CAP
        and
        _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
        and initial_bridge_inventory_ul
        >= float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - 1.0e-9
    )
    if (
        bool(RETAIN_CASE61_NECK_SUPPLY_STATE)
        and not supply_already_saturated
    ):
        parent.adaptive_neck_reprojection(
            state,
            ring_region,
            config,
            time_s=time_s,
            dt_s=dt_s,
        )
        if (
            parent._FLUX_REFERENCE_SURFACE is not None
            and np.asarray(parent._FLUX_REFERENCE_SURFACE).shape
            == state.surface_points().shape
        ):
            retained_supply_surface = np.asarray(
                parent._FLUX_REFERENCE_SURFACE,
                dtype=float,
            ).copy()
    surface = state.surface_points().copy()
    state.case62_ring_region = np.asarray(ring_region, dtype=int).copy()
    target_volume_ul = float(
        getattr(
            state,
            "reference_volume_ul",
            base.volume_under_mesh_ul(surface, state.surface_faces),
        )
    )
    rings = state.rings
    bridge_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 0)
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if bridge_rows.size < 2 or film_rows.size < 6:
        state.set_surface_points(surface)
        return

    old_r, old_z = parent._ring_coordinates(surface, rings)
    contact_row = int(bridge_rows[0])
    outer_row = int(film_rows[-1])
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    substrate_r = float(config.substrate_radius_mm) * 1.0e-3
    gamma = float(config.surface_tension_n_m)
    rho_g = float(config.density_kg_m3) * float(config.gravity_m_s2)
    capillary_length = math.sqrt(gamma / max(rho_g, 1.0e-30))
    contact_r = float(
        np.clip(
            max(
                old_r[contact_row],
                float(config.initial_bridge_radius_mm) * 1.0e-3,
            ),
            1.0e-9,
            0.82 * substrate_r,
        )
    )
    current_bridge_volume_ul = float(
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            ring_region,
            config,
        )
    )
    previous_r, previous_h, previous_bridge_volume_ul, previous_junction_r = (
        _previous_partitioned_state(
            state,
            old_r[film_rows],
            old_z[film_rows],
            current_bridge_volume_ul,
        )
    )
    previous_hydraulic_deficit_ul = float(
        getattr(
            state,
            "case62_accepted_hydraulic_deficit_ul",
            _RESTART_HYDRAULIC_DEFICIT_UL,
        )
    )
    previous_hydraulic_rate_ul_s = float(
        getattr(
            state,
            "case62_accepted_hydraulic_rate_ul_s",
            _RESTART_HYDRAULIC_RATE_UL_S,
        )
    )
    previous_local_inventory_ul = float(
        getattr(
            state,
            "case77_accepted_local_deficit_inventory_ul",
            _RESTART_LOCAL_DEFICIT_INVENTORY_UL,
        )
    )
    previous_far_inventory_ul = float(
        getattr(
            state,
            "case77_accepted_far_deficit_inventory_ul",
            _RESTART_FAR_DEFICIT_INVENTORY_UL,
        )
    )
    previous_far_rate_ul_s = float(
        getattr(
            state,
            "case77_accepted_far_deficit_rate_ul_s",
            _RESTART_FAR_DEFICIT_RATE_UL_S,
        )
    )
    previous_saturation_time_s = float(
        getattr(
            state,
            "case62_accepted_supply_saturation_time_s",
            _RESTART_SUPPLY_SATURATION_TIME_S,
        )
    )
    previous_body_fitted_transition_time_s = float(
        getattr(
            state,
            "case77_accepted_body_fitted_transition_time_s",
            _RESTART_BODY_FITTED_TRANSITION_TIME_S,
        )
    )
    previous_body_fitted_transition_deficit_ul = float(
        getattr(
            state,
            "case77_accepted_body_fitted_transition_deficit_ul",
            _RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL,
        )
    )
    step_dt = float(dt_s)
    bridge_rate_m3_s = 0.0
    swept_rate_m3_s = 0.0
    hydraulic_rate_m3_s = 0.0
    contact_speed_m_s = 0.0
    viscocapillary_width_m = 0.0
    cumulative_hydraulic_deficit_ul = previous_hydraulic_deficit_ul
    local_deficit_inventory_ul = previous_local_inventory_ul
    far_deficit_inventory_ul = previous_far_inventory_ul
    far_deficit_rate_ul_s = 0.0
    saturation_time_s = previous_saturation_time_s
    body_fitted_transition_time_s = (
        previous_body_fitted_transition_time_s
    )
    body_fitted_transition_deficit_ul = (
        previous_body_fitted_transition_deficit_ul
    )
    body_fitted_equilibrium_contact_radius_m = float("nan")
    body_fitted_center_radius_m = float("nan")
    body_fitted_early_deficit_ul = 0.0
    body_fitted_late_deficit_ul = 0.0
    body_fitted_hmin_um = float("nan")
    body_fitted_rmin_mm = float("nan")
    body_fitted_volume_residual_ul = 0.0
    body_fitted_geometric_deficit_ul = 0.0
    body_fitted_target_deficit_ul = 0.0
    body_fitted_c1_join_active = 0.0
    body_fitted_c1_join_start_radius_mm = float("nan")
    body_fitted_c1_join_end_radius_mm = float("nan")
    body_fitted_c1_join_pre_angle_jump_rad = float("nan")
    body_fitted_c1_join_post_angle_jump_rad = float("nan")
    body_fitted_c1_join_bridge_volume_residual_ul = 0.0
    body_fitted_c1_join_film_volume_residual_ul = 0.0
    body_fitted_outer_support_radius_mm = float("nan")
    body_fitted_outer_arm_reversal_count = 0.0
    body_fitted_outer_c1_join_active = 0.0
    body_fitted_outer_c1_join_start_radius_mm = float("nan")
    body_fitted_outer_c1_join_end_radius_mm = float("nan")
    body_fitted_outer_c1_join_pre_angle_jump_rad = float("nan")
    body_fitted_outer_c1_join_post_angle_jump_rad = float("nan")
    body_fitted_outer_c1_join_volume_residual_ul = 0.0
    body_fitted_pressure_closed_depth_um = float("nan")
    body_fitted_pressure_closed_inner_slope = float("nan")
    body_fitted_pressure_closed_outer_slope = float("nan")
    body_fitted_film_pressure_pa = float("nan")
    body_fitted_bridge_pressure_pa = float("nan")
    if step_dt > 0.0:
        bridge_rate_m3_s = (
            (current_bridge_volume_ul - previous_bridge_volume_ul)
            * 1.0e-9
            / step_dt
        )
        cap_derivative_m2 = case28.attached_cap_volume_derivative_m2(
            config,
            contact_r,
        )
        contact_speed_m_s = bridge_rate_m3_s / max(
            cap_derivative_m2,
            1.0e-30,
        )

    # Retain Case61's resolution-only two-zone radial map.  Heights are
    # interpolated before the physical film solve; no normal smoothing or
    # compact recovery profile is applied.
    bridge_gap = max(
        0.08e-3,
        0.93 * capillary_length,
        0.20 * contact_r,
        1.5 * h0,
    )
    rim_r = min(contact_r + bridge_gap, substrate_r - 0.25e-3)
    base_first_film_r = min(
        rim_r + max(0.02e-3, 0.08 * bridge_gap),
        substrate_r - 1.0e-9,
    )
    junction_height_m = float(
        np.interp(
            base_first_film_r,
            previous_r,
            previous_h,
            left=float(previous_h[0]),
            right=float(previous_h[-1]),
        )
    )
    # The bridge representation and the outer-film PDE overlap across one
    # viscocapillary transition layer.  Match the two reduced domains at its
    # outer edge.  Two scalar fixed-point updates account for the local h^3
    # mobility without adding a nonlinear field solve.
    if abs(contact_speed_m_s) > 1.0e-30:
        for _ in range(2):
            viscocapillary_width_m = (
                gamma
                * max(junction_height_m, FILM_MINIMUM_HEIGHT_M) ** 3
                / (
                    3.0
                    * float(config.viscosity_pa_s)
                    * abs(contact_speed_m_s)
                )
            ) ** (1.0 / 3.0)
            viscocapillary_width_m = min(
                viscocapillary_width_m,
                MAX_VISCOCAPILLARY_WIDTH_CAPILLARY_LENGTHS
                * capillary_length,
            )
            matching_radius = min(
                (
                    rim_r
                    if RESOLVE_TWO_SIDED_JUNCTION_LAYER
                    else base_first_film_r
                )
                + viscocapillary_width_m,
                substrate_r - 1.0e-9,
            )
            junction_height_m = float(
                np.interp(
                    matching_radius,
                    previous_r,
                    previous_h,
                    left=float(previous_h[0]),
                    right=float(previous_h[-1]),
                )
            )
    new_r = old_r.copy()
    bridge_coordinate = np.linspace(0.0, 1.0, bridge_rows.size)
    new_r[bridge_rows] = (
        contact_r
        + (rim_r - contact_r) * bridge_coordinate**1.15
    )
    matching_radius = min(
        (
            rim_r
            if RESOLVE_TWO_SIDED_JUNCTION_LAYER
            else base_first_film_r
        )
        + viscocapillary_width_m,
        substrate_r - 1.0e-9,
    )
    body_fitted_refinement_ready = bool(
        BODY_FITTED_JUNCTION_REFINEMENT_ENABLED
        and (
            _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
            or _PARTITIONED_CONNECTED_SUPPORT_ACTIVE
        )
    )
    if body_fitted_refinement_ready:
        equilibrium_contact_radius_m = (
            _young_laplace_bridge_contact_radius_m(
                surface,
                rings,
                ring_region,
                config,
            )
        )
        if equilibrium_contact_radius_m is not None:
            body_fitted_equilibrium_contact_radius_m = float(
                equilibrium_contact_radius_m
            )
            body_fitted_center_radius_m = min(
                float(equilibrium_contact_radius_m) + capillary_length,
                substrate_r - capillary_length - 1.0e-9,
            )
            matching_radius = body_fitted_center_radius_m
            body_fitted_inner_width_m = math.sqrt(
                h0 * capillary_length
            )
            base_first_film_r = max(
                rim_r + 1.0e-9,
                body_fitted_center_radius_m
                - body_fitted_inner_width_m,
            )
    resolve_inner_layer = bool(
        RESOLVE_TWO_SIDED_JUNCTION_LAYER
        and matching_radius - base_first_film_r > 1.0e-9
    )
    first_film_r = (
        base_first_film_r
        if resolve_inner_layer
        else matching_radius
    )
    neck_end = min(
        rim_r
        + float(parent.NECK_ZONE_CAPILLARY_LENGTHS) * capillary_length
        + viscocapillary_width_m,
        substrate_r - 0.45e-3,
    )
    film_count = int(film_rows.size)
    inner_count = (
        int(
            np.clip(
                (
                    round(0.15 * film_count)
                    if not BODY_FITTED_JUNCTION_REFINEMENT_ENABLED
                    else 17
                ),
                8,
                film_count - 6,
            )
        )
        if resolve_inner_layer
        else 1
    )
    inner_rows = film_rows[:inner_count]
    solver_film_rows = film_rows[inner_count - 1 :]
    if resolve_inner_layer:
        inner_coordinate = np.linspace(0.0, 1.0, inner_rows.size)
        new_r[inner_rows] = (
            first_film_r
            + (matching_radius - first_film_r) * inner_coordinate
        )
    solver_film_count = int(solver_film_rows.size)
    near_count = int(
        np.clip(
            round(
                float(parent.NECK_ZONE_RING_FRACTION) * solver_film_count
            ),
            4,
            solver_film_count - 3,
        )
    )
    near_rows = solver_film_rows[:near_count]
    far_rows = solver_film_rows[near_count - 1 :]
    near_coordinate = np.linspace(0.0, 1.0, near_rows.size)
    far_coordinate = np.linspace(0.0, 1.0, far_rows.size)
    new_r[near_rows] = (
        matching_radius
        + (neck_end - matching_radius)
        * near_coordinate ** float(parent.NECK_ZONE_SPACING_EXPONENT)
    )
    new_r[far_rows] = (
        neck_end
        + (substrate_r - neck_end)
        * far_coordinate ** float(parent.FAR_ZONE_SPACING_EXPONENT)
    )
    new_r[outer_row] = substrate_r

    new_z = old_z.copy()
    bridge_sort = bridge_rows[np.argsort(old_r[bridge_rows])]
    new_z[bridge_rows] = np.interp(
        new_r[bridge_rows],
        old_r[bridge_sort],
        old_z[bridge_sort],
    )
    remapped_height = np.interp(
        new_r[film_rows],
        previous_r,
        previous_h,
        left=float(previous_h[0]),
        right=float(previous_h[-1]),
    )
    new_z[film_rows] = remapped_height
    parent._set_axisymmetric_rows(surface, rings, new_r, new_z)
    case28.project_contact_ring_to_sphere(
        surface,
        rings[contact_row],
        config,
    )

    film_diag: dict[str, float] = {}
    solve_failed = 0.0
    hydraulic_pressure_drop_pa = 0.0
    capillary_core_radius_m = math.inf
    capillary_core_half_width_m = 0.0
    vlike_support_half_width_m = 0.0
    viscocapillary_side_slope = 0.0
    micro_layer_active_nodes = 0
    micro_layer_volume_redistribution_ul = 0.0
    junction_pressure_bc_pa: float | None = None
    heron_bridge_pressure_pa = float("nan")
    pr35_bridge_pressure_pa = float("nan")
    pr35_far_film_pressure_pa = float("nan")
    pr35_bridge_minus_far_pressure_pa = float("nan")
    pressure_flux_solved = False
    pressure_driven_flux_ul_s = 0.0
    pressure_driven_bridge_pressure_pa = float("nan")
    pressure_driven_donor_pressure_pa = float("nan")
    pressure_driven_resistance_pa_s_m3 = float("inf")
    pressure_driven_hmin_um = float("nan")
    pressure_driven_turn_resistance_pa_s_m2 = 0.0
    pressure_driven_matching_resistance_pa_s_m2 = 0.0
    pressure_driven_total_local_resistance_pa_s_m2 = 0.0
    pressure_driven_boundary_radius_mm = float("nan")
    physical_film_diag: dict[str, float] = {}
    (
        physical_film_radius_m,
        physical_film_height_m,
    ) = _previous_physical_film_state(
        state,
        first_film_r,
        int(film_rows.size),
        config,
    )
    physical_film_flux_ul_s = float(
        getattr(
            state,
            "case77_accepted_physical_film_flux_ul_s",
            _RESTART_PHYSICAL_FILM_FLUX_UL_S,
        )
    )
    if bool(
        USE_PR35_JUNCTION_PRESSURE_CONTINUITY
        or USE_PRESSURE_DRIVEN_JUNCTION_FLUX
    ):
        (
            pr35_junction_pressure_pa,
            pr35_bridge_pressure_pa,
            pr35_far_film_pressure_pa,
            pr35_bridge_minus_far_pressure_pa,
        ) = _pr35_bridge_junction_pressure_pa(
            state,
            rings,
            ring_region,
            config,
        )
        if bool(USE_PR35_JUNCTION_PRESSURE_CONTINUITY):
            junction_pressure_bc_pa = pr35_junction_pressure_pa
    if bool(USE_HERON_JUNCTION_PRESSURE_CONTINUITY):
        heron_pressure_candidate_pa = (
            _heron_bridge_junction_pressure_pa(
                state,
                surface,
                rings,
                ring_region,
                config,
            )
        )
        if heron_pressure_candidate_pa is not None:
            heron_bridge_pressure_pa = float(
                heron_pressure_candidate_pa
            )
            junction_pressure_bc_pa = heron_bridge_pressure_pa
    if step_dt > 0.0:
        # The bridge inventory is the volume swept under the curved sphere.
        # Reynolds transport uses the actual, solved junction thickness rather
        # than the initial h0.  The remaining curved-gap demand is the inward
        # hydraulic boundary flux for the outer film.
        solver_start = int(inner_count - 1)
        solver_remapped_height = np.asarray(
            remapped_height[solver_start:],
            dtype=float,
        )
        junction_height_m = float(solver_remapped_height[0])
        swept_rate_m3_s = (
            2.0
            * math.pi
            * contact_r
            * junction_height_m
            * contact_speed_m_s
        )
        hydraulic_rate_m3_s = bridge_rate_m3_s - swept_rate_m3_s
        hydraulic_rate_ul_s = max(hydraulic_rate_m3_s * 1.0e9, 0.0)
        cumulative_hydraulic_deficit_ul += (
            hydraulic_rate_ul_s * step_dt
        )
        # Case77 resolves the junction inventory into two conservative
        # accounts.  The far account is the accepted time integral of the
        # independently computed far-film flux.  Everything not yet supplied
        # through that connection remains in the local junction account:
        #
        #   dV_far/dt = Q_far,
        #   V_local = V_hydraulic - V_far.
        #
        # No profile ordinate or experiment enters this scalar transport.
        far_deficit_rate_ul_s = max(
            float(_PARTITIONED_FAR_SUPPLY_FLUX_UL_S),
            0.0,
        )
        far_deficit_inventory_ul = (
            previous_far_inventory_ul
            + 0.5
            * (
                max(previous_far_rate_ul_s, 0.0)
                + far_deficit_rate_ul_s
            )
            * step_dt
        )
        inventory_tolerance_ul = 1.0e-10
        if (
            far_deficit_inventory_ul
            > cumulative_hydraulic_deficit_ul + inventory_tolerance_ul
        ):
            raise RuntimeError(
                "Case77 far-film inventory exceeds the accepted hydraulic "
                "deficit; reject this nonconservative step"
            )
        if (
            far_deficit_inventory_ul
            > cumulative_hydraulic_deficit_ul
        ):
            # Roundoff-only projection onto the exact conservative simplex.
            far_deficit_inventory_ul = cumulative_hydraulic_deficit_ul
        local_deficit_inventory_ul = (
            cumulative_hydraulic_deficit_ul
            - far_deficit_inventory_ul
        )
        supply_saturated = bool(
            USE_TERMINAL_LOCAL_DONOR_CAP
            and
            _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
            and current_bridge_volume_ul
            >= float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - 1.0e-9
        )
        if supply_saturated and saturation_time_s <= 0.0:
            saturation_time_s = float(
                time_s
                if time_s is not None
                else step_dt
        )
        if bool(USE_MOVING_BOUNDARY_PRESSURE_FILM):
            moving_bridge_pressure_pa = None
            moving_footprint_radius_m = (
                _young_laplace_bridge_footprint_radius_m(
                    surface,
                    rings,
                    ring_region,
                    config,
                )
            )
            if moving_footprint_radius_m is None:
                # Before the equilibrium manifold contains the microscopic
                # startup inventory, the transported bridge rim is the only
                # resolved physical footprint.  Never use the velocity-based
                # viscocapillary matching radius as a moving material edge.
                moving_footprint_radius_m = float(rim_r)
            if bool(supply_saturated):
                # In the accepted low-inertia branch, close the moving-film
                # boundary with the same Young--Laplace pressure manifold
                # used by PR35.  A coarse pointwise Heron pressure trace is
                # an inconsistent, noisy boundary condition for the radial
                # film PDE even though the integrated Heron force remains the
                # exact PR33 capillary force in momentum.
                moving_bridge_pressure_pa = (
                    _young_laplace_bridge_junction_pressure_pa(
                        surface,
                        rings,
                        ring_region,
                        config,
                    )
                )
            (
                physical_film_radius_m,
                physical_film_height_m,
                physical_film_flux_ul_s,
                physical_film_diag,
            ) = _advance_moving_boundary_pressure_film(
                state,
                surface,
                rings,
                ring_region,
                config,
                footprint_radius_m=float(moving_footprint_radius_m),
                film_ring_count=int(film_rows.size),
                bridge_supply_rate_m3_s=max(hydraulic_rate_m3_s, 0.0),
                bridge_pressure_pa=moving_bridge_pressure_pa,
                dt_s=step_dt,
            )
            pressure_driven_flux_ul_s = float(
                physical_film_flux_ul_s
            )
            pressure_driven_hmin_um = float(
                physical_film_diag["case77_pressure_film_hmin_um"]
            )
            pressure_driven_boundary_radius_mm = float(
                physical_film_radius_m[0] * 1.0e3
            )
            pressure_flux_solved = bool(
                moving_bridge_pressure_pa is not None
            )
            film_diag.update(physical_film_diag)
        if abs(contact_speed_m_s) > 1.0e-30:
            viscocapillary_width_m = (
                float(config.surface_tension_n_m)
                * max(junction_height_m, FILM_MINIMUM_HEIGHT_M) ** 3
                / (
                    3.0
                    * float(config.viscosity_pa_s)
                    * abs(contact_speed_m_s)
                )
            ) ** (1.0 / 3.0)
            viscocapillary_width_m = min(
                viscocapillary_width_m,
                MAX_VISCOCAPILLARY_WIDTH_CAPILLARY_LENGTHS
                * capillary_length,
            )
        else:
            viscocapillary_width_m = 0.0
        try:
            sphere_radius_m = float(config.sphere_radius_mm) * 1.0e-3
            first_contact_radius_m = math.sqrt(
                max(
                    2.0 * sphere_radius_m * h0 - h0 * h0,
                    0.0,
                )
            )
            swept_capture_ul = (
                math.pi
                * max(contact_r**2 - first_contact_radius_m**2, 0.0)
                * h0
                * 1.0e9
            )
            pressure_bridge_candidate_pa = (
                _young_laplace_bridge_junction_pressure_pa(
                    surface,
                    rings,
                    ring_region,
                    config,
                )
                if USE_PRESSURE_DRIVEN_JUNCTION_FLUX
                else None
            )
            pressure_supply_active = bool(
                USE_PRESSURE_DRIVEN_JUNCTION_FLUX
                and not USE_MOVING_BOUNDARY_PRESSURE_FILM
                and pressure_bridge_candidate_pa is not None
                and _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
                and current_bridge_volume_ul
                >= float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - 1.0e-12
            )
            if pressure_supply_active:
                # The junction-to-donor path is one series resistance.  Starting
                # at the dimple minimum short-circuits the resolved inner arm
                # and makes Q_J artificially large, so retain every solver
                # film ring from the physical matching boundary outward.
                pressure_start = 0
                pressure_radius_m = np.asarray(
                    new_r[solver_film_rows][pressure_start:],
                    dtype=float,
                )
                pressure_old_height_m = np.asarray(
                    solver_remapped_height[pressure_start:],
                    dtype=float,
                )
                pressure_driven_boundary_radius_mm = float(
                    pressure_radius_m[0] * 1.0e3
                )
                pressure_driven_bridge_pressure_pa = float(
                    pressure_bridge_candidate_pa
                )
                sphere_axial_m = math.sqrt(
                    max(sphere_radius_m**2 - contact_r**2, 1.0e-30)
                )
                sphere_swept_height_m = (
                    contact_r**2 / (2.0 * sphere_axial_m)
                )
                junction_aperture_m = max(
                    float(pressure_old_height_m[0])
                    + sphere_swept_height_m,
                    FILM_MINIMUM_HEIGHT_M,
                )
                pressure_driven_turn_resistance_pa_s_m2 = (
                    12.0
                    * float(config.viscosity_pa_s)
                    / junction_aperture_m**2
                )
                # ``solver_film_rows`` starts at the outer edge of Case77's
                # physical bridge--film matching sublayer.  That sublayer is
                # not part of the radial film matrix, but it still carries the
                # same through-flow.  Add its exact lubrication resistance in
                # series, using the minimum-curvature Hermite profile that is
                # applied below:
                #
                #   R_match = integral 3*mu*(r_b/r)/h(r)^3 dr,
                #   Delta p_match = R_match*q_b.
                #
                # All inputs are current solved geometry and material
                # properties; no fitted length, rate, or experimental value
                # enters this closure.
                if resolve_inner_layer:
                    resistance_radius_m = np.linspace(
                        first_film_r,
                        matching_radius,
                        65,
                    )
                    resistance_coordinate = (
                        resistance_radius_m - first_film_r
                    ) / max(
                        matching_radius - first_film_r,
                        1.0e-30,
                    )
                    resistance_smoothstep = (
                        resistance_coordinate**2
                        * (3.0 - 2.0 * resistance_coordinate)
                    )
                    resistance_height_m = (
                        h0
                        + (
                            float(pressure_old_height_m[0])
                            - h0
                        )
                        * resistance_smoothstep
                    )
                    pressure_driven_matching_resistance_pa_s_m2 = float(
                        np.trapezoid(
                            3.0
                            * float(config.viscosity_pa_s)
                            * (
                                float(pressure_radius_m[0])
                                / np.maximum(
                                    resistance_radius_m,
                                    1.0e-12,
                                )
                            )
                            / np.maximum(
                                resistance_height_m,
                                FILM_MINIMUM_HEIGHT_M,
                            )
                            ** 3,
                            resistance_radius_m,
                        )
                    )
                pressure_driven_total_local_resistance_pa_s_m2 = (
                    pressure_driven_turn_resistance_pa_s_m2
                    + pressure_driven_matching_resistance_pa_s_m2
                )
                (
                    pressure_solved_height,
                    _inner_pressure_pa,
                    pressure_film_rate_m3_s,
                ) = _implicit_nonlinear_film_flux_step(
                    pressure_radius_m,
                    pressure_old_height_m,
                    inner_flux_m2_s=0.0,
                    inner_pressure_pa=(
                        pressure_driven_bridge_pressure_pa
                    ),
                    inner_hydraulic_resistance_pa_s_m2=(
                        pressure_driven_total_local_resistance_pa_s_m2
                    ),
                    dt_s=step_dt,
                    surface_tension_n_m=float(config.surface_tension_n_m),
                    viscosity_pa_s=float(config.viscosity_pa_s),
                    nonlinear_max_iterations=80,
                    nonlinear_rtol=1.0e-10,
                    density_kg_m3=float(config.density_kg_m3),
                    gravity_m_s2=float(config.gravity_m_s2),
                    full_curvature=False,
                    flux_conservation_absolute_tolerance_m3_s=1.0e-16,
                )
                solved_height = np.asarray(
                    solver_remapped_height,
                    dtype=float,
                ).copy()
                solved_height[pressure_start:] = (
                    pressure_solved_height
                )
                pressure_driven_flux_ul_s = max(
                    -float(pressure_film_rate_m3_s) * 1.0e9,
                    0.0,
                )
                pressure_driven_hmin_um = float(
                    np.min(solved_height) * 1.0e6
                )
                pressure_driven_donor_pressure_pa = (
                    float(config.density_kg_m3)
                    * float(config.gravity_m_s2)
                    * h0
                )
                pressure_drop_pa = (
                    pressure_driven_donor_pressure_pa
                    - pressure_driven_bridge_pressure_pa
                )
                pressure_driven_resistance_pa_s_m3 = (
                    pressure_drop_pa
                    / max(pressure_driven_flux_ul_s * 1.0e-9, 1.0e-300)
                    if pressure_driven_flux_ul_s > 0.0
                    else float("inf")
                )
                minimum_index = int(np.argmin(solved_height))
                diagnostics = SimpleNamespace(
                    film_volume_change_m3=(
                        float(pressure_film_rate_m3_s) * step_dt
                    ),
                    conservation_residual_m3=0.0,
                    minimum_height_m=float(solved_height[minimum_index]),
                    minimum_radius_m=float(
                        new_r[solver_film_rows][minimum_index]
                    ),
                    junction_pressure_pa=float(_inner_pressure_pa),
                    minimum_pressure_pa=float("nan"),
                    maximum_pressure_pa=float("nan"),
                    linear_solve_count=1,
                )
                pressure_flux_solved = True
            else:
                solved_height, diagnostics = (
                    advance_axisymmetric_film_prescribed_junction_flux(
                    new_r[solver_film_rows],
                    solver_remapped_height,
                    junction_rate_m3_s=hydraulic_rate_m3_s,
                    dt_s=step_dt,
                    surface_tension_n_m=float(config.surface_tension_n_m),
                    viscosity_pa_s=float(config.viscosity_pa_s),
                    density_kg_m3=float(config.density_kg_m3),
                    gravity_m_s2=float(config.gravity_m_s2),
                    mobility_average="harmonic",
                    picard_iterations=FILM_PICARD_ITERATIONS,
                    picard_relaxation=FILM_PICARD_RELAXATION,
                    picard_tolerance_m=FILM_PICARD_TOLERANCE_M,
                    minimum_height_m=FILM_MINIMUM_HEIGHT_M,
                    junction_pressure_pa=junction_pressure_bc_pa,
                )
                )
            for row_id, height in zip(solver_film_rows, solved_height):
                surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(
                    height
                )
            if (
                resolve_inner_layer
                and USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
                and abs(hydraulic_rate_m3_s) > 1.0e-30
            ):
                # The large outer-film solve remains flux-conservative and
                # determines h_J.  Only the unresolved local matching layer
                # is replaced below.  Its pressure drop follows directly from
                # Q = 2*pi*r*h^3*DeltaP/(3*mu*L), with the computed bridge gap
                # as hydraulic length.
                junction_height = float(solved_height[0])
                hydraulic_pressure_drop_pa = (
                    3.0
                    * float(config.viscosity_pa_s)
                    * abs(hydraulic_rate_m3_s)
                    * bridge_gap
                    / (
                        2.0
                        * math.pi
                        * matching_radius
                        * max(junction_height, FILM_MINIMUM_HEIGHT_M) ** 3
                    )
                )
                capillary_core_radius_m = (
                    float(config.surface_tension_n_m)
                    / hydraulic_pressure_drop_pa
                )
                capillary_number = (
                    float(config.viscosity_pa_s)
                    * abs(contact_speed_m_s)
                    / float(config.surface_tension_n_m)
                )
                viscocapillary_side_slope = (
                    3.0 * capillary_number
                ) ** (1.0 / 3.0)
                # A circular Young--Laplace core is joined tangentially to the
                # viscocapillary side slope.  This is smooth at the minimum
                # but appears V-shaped outside the pressure-rounded core.
                capillary_core_half_width_m = math.sqrt(
                    max(viscocapillary_side_slope, 0.0) ** 2
                    * capillary_core_radius_m**2
                    / (
                        1.0
                        + max(viscocapillary_side_slope, 0.0) ** 2
                    )
                )
                tangent_rise_m = (
                    capillary_core_radius_m
                    - math.sqrt(
                        max(
                            capillary_core_radius_m**2
                            - capillary_core_half_width_m**2,
                            0.0,
                        )
                    )
                )
                remaining_rise_m = max(
                    h0 - junction_height - tangent_rise_m,
                    0.0,
                )
                if viscocapillary_side_slope > 1.0e-30:
                    vlike_support_half_width_m = (
                        capillary_core_half_width_m
                        + remaining_rise_m / viscocapillary_side_slope
                    )
                else:
                    vlike_support_half_width_m = (
                        capillary_core_half_width_m
                    )
                film_radius = np.asarray(
                    new_r[film_rows],
                    dtype=float,
                )
                provisional_height = np.asarray(
                    [
                        float(
                            np.mean(
                                surface[
                                    np.asarray(rings[int(row_id)], dtype=int),
                                    2,
                                ]
                            )
                        )
                        for row_id in film_rows
                    ],
                    dtype=float,
                )
                offset = np.abs(film_radius - matching_radius)
                local_height = np.full(film_rows.size, h0, dtype=float)
                in_core = offset <= capillary_core_half_width_m
                local_height[in_core] = (
                    junction_height
                    + capillary_core_radius_m
                    - np.sqrt(
                        np.maximum(
                            capillary_core_radius_m**2
                            - offset[in_core] ** 2,
                            0.0,
                        )
                    )
                )
                in_side = (
                    (offset > capillary_core_half_width_m)
                    & (offset < vlike_support_half_width_m)
                )
                local_height[in_side] = (
                    junction_height
                    + tangent_rise_m
                    + viscocapillary_side_slope
                    * (
                        offset[in_side]
                        - capillary_core_half_width_m
                    )
                )
                active = offset < vlike_support_half_width_m
                micro_layer_active_nodes = int(np.count_nonzero(active))
                # Outside the local matching support, retain positive
                # capillary ridges from the outer PDE but match a depleted
                # provisional branch back to the undisturbed h0 reservoir.
                matched_height = np.where(
                    active,
                    local_height,
                    np.where(provisional_height >= h0, provisional_height, h0),
                )
                before_volume = float(
                    2.0
                    * math.pi
                    * np.trapezoid(
                        film_radius * provisional_height,
                        film_radius,
                    )
                )
                after_volume = float(
                    2.0
                    * math.pi
                    * np.trapezoid(
                        film_radius * matched_height,
                        film_radius,
                    )
                )
                micro_layer_volume_redistribution_ul = (
                    (after_volume - before_volume) * 1.0e9
                )
                for row_id, height in zip(film_rows, matched_height):
                    surface[
                        np.asarray(rings[int(row_id)], dtype=int),
                        2,
                    ] = float(height)
            elif resolve_inner_layer:
                # The inner sublayer is the minimum-curvature Hermite match
                # between the undisturbed incoming film and the dynamically
                # solved junction.  It is the linearized capillary-energy
                # minimizer for fixed endpoint heights and horizontal endpoint
                # slopes.  Width comes only from the viscocapillary scale.
                coordinate = np.linspace(0.0, 1.0, inner_rows.size)
                smoothstep = coordinate**2 * (3.0 - 2.0 * coordinate)
                inner_height = (
                    h0
                    + (float(solved_height[0]) - h0) * smoothstep
                )
                for row_id, height in zip(inner_rows, inner_height):
                    surface[
                        np.asarray(rings[int(row_id)], dtype=int),
                        2,
                    ] = float(height)
            film_diag = {
                "case62_film_volume_change_ul": (
                    float(diagnostics.film_volume_change_m3) * 1.0e9
                ),
                "case62_film_conservation_residual_ul": (
                    float(diagnostics.conservation_residual_m3) * 1.0e9
                ),
                "case62_film_hmin_um": (
                    float(diagnostics.minimum_height_m) * 1.0e6
                ),
                "case62_film_rmin_mm": (
                    float(diagnostics.minimum_radius_m) * 1.0e3
                ),
                "case62_junction_pressure_pa": float(
                    diagnostics.junction_pressure_pa
                ),
                "case62_film_pressure_min_pa": float(
                    diagnostics.minimum_pressure_pa
                ),
                "case62_film_pressure_max_pa": float(
                    diagnostics.maximum_pressure_pa
                ),
                "case62_film_linear_solve_count": float(
                    diagnostics.linear_solve_count
                ),
            }
        except (RuntimeError, ValueError):
            # Produce an inadmissible geometric candidate so Case61's existing
            # backtracking loop retries a smaller physical displacement.
            solve_failed = 1.0
            bad_height = np.asarray(remapped_height, dtype=float).copy()
            bad_height[inner_count - 1] = -FILM_MINIMUM_HEIGHT_M
            for row_id, height in zip(film_rows, bad_height):
                surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(
                    height
                )

    retained_neck_rows = 0
    if retained_supply_surface is not None:
        retained_r, retained_h = parent._ring_coordinates(
            retained_supply_surface,
            rings,
        )
        retained_order = np.argsort(retained_r)
        final_r_before_restoration, _ = parent._ring_coordinates(
            surface,
            rings,
        )
        local_rows = film_rows[
            final_r_before_restoration[film_rows]
            <= matching_radius + 1.0e-15
        ]
        retained_values = np.interp(
            final_r_before_restoration[local_rows],
            retained_r[retained_order],
            retained_h[retained_order],
        )
        retained_weight = 1.0
        if RETAIN_CASE61_NECK_LOW_CA_ONLY:
            # Match the moving viscocapillary solution to the accumulated
            # quasi-static neck only as Ca -> 0.  The scale ratio
            # ell_vc/ell_c supplies a continuous asymptotic blend and avoids
            # superposing two troughs during the early moving-junction stage.
            scale_ratio = (
                viscocapillary_width_m
                / max(capillary_length, 1.0e-30)
            )
            retained_weight = (
                1.0
                if abs(contact_speed_m_s) <= 1.0e-30
                else float(
                    np.clip(
                        (scale_ratio - 0.5) / 0.5,
                        0.0,
                        1.0,
                    )
                )
            )
        if np.all(
            retained_values > case28.TETRA_MIN_LAYER_HEIGHT_M
        ) and retained_weight > 0.0:
            for row_id, height in zip(local_rows, retained_values):
                ids = np.asarray(rings[int(row_id)], dtype=int)
                surface[ids, 2] = (
                    (1.0 - retained_weight) * surface[ids, 2]
                    + retained_weight * float(height)
                )
            retained_neck_rows = int(local_rows.size)

    # Replace the under-resolved one-edge trough by a body-fitted capillary
    # deficit field.  The centre comes from the computed Young--Laplace bridge
    # manifold.  Its inner support is the material film thickness and its
    # outer support is the capillary length.  After the computed
    # viscocapillary scale reaches ell_c, the late far-film deficit is matched
    # across the intermediate asymptotic layer between the local junction
    # width and the outer capillary length.  The geometric-mean matching
    # length introduces no fitted length or time.  Both kernels are normalized
    # in axisymmetric volume, so their amplitudes are fixed entirely by the
    # accepted missing-film inventory.
    if (
        body_fitted_refinement_ready
        and math.isfinite(body_fitted_center_radius_m)
    ):
        prefit_radius_m, prefit_height_m = parent._ring_coordinates(
            surface,
            rings,
        )
        prefit_radius_m = np.asarray(
            prefit_radius_m[film_rows],
            dtype=float,
        )
        prefit_height_m = np.asarray(
            prefit_height_m[film_rows],
            dtype=float,
        )
        prefit_baseline_m = _finite_substrate_film_profile_m(
            prefit_radius_m,
            config,
        )
        body_fitted_geometric_deficit_ul = float(
            2.0
            * math.pi
            * np.trapezoid(
                prefit_radius_m
                * np.maximum(prefit_baseline_m - prefit_height_m, 0.0),
                prefit_radius_m,
            )
            * 1.0e9
        )
        if (
            body_fitted_transition_time_s <= 0.0
            and viscocapillary_width_m >= capillary_length
            and body_fitted_geometric_deficit_ul > 0.0
        ):
            body_fitted_transition_time_s = float(
                time_s if time_s is not None else step_dt
            )
            body_fitted_transition_deficit_ul = float(
                cumulative_hydraulic_deficit_ul
            )
        if body_fitted_transition_time_s > 0.0:
            # The local account is replenished by the accepted far-film
            # transfer.  Keeping its transition value frozen while also
            # adding V_far localizes the same missing volume twice.
            body_fitted_early_deficit_ul = max(
                local_deficit_inventory_ul,
                0.0,
            )
            body_fitted_late_deficit_ul = max(
                far_deficit_inventory_ul,
                0.0,
            )
        else:
            body_fitted_early_deficit_ul = max(
                cumulative_hydraulic_deficit_ul,
                0.0,
            )
            body_fitted_late_deficit_ul = 0.0

        fitted_radius_m = np.asarray(new_r[film_rows], dtype=float)
        baseline_height_m = _finite_substrate_film_profile_m(
            fitted_radius_m,
            config,
        )
        body_fitted_target_deficit_ul = (
            body_fitted_early_deficit_ul
            + body_fitted_late_deficit_ul
        )
        bridge_pressure_pa = _young_laplace_bridge_junction_pressure_pa(
            surface,
            rings,
            ring_region,
            config,
        )
        if bridge_pressure_pa is None:
            bridge_pressure_pa = float(
                _PARTITIONED_FILM_INNER_PRESSURE_PA
                - float(config.surface_tension_n_m) / capillary_length
            )
        body_fitted_film_pressure_pa = float(
            _PARTITIONED_FILM_INNER_PRESSURE_PA
        )
        body_fitted_bridge_pressure_pa = float(bridge_pressure_pa)
        (
            fitted_height_m,
            pressure_closed_depth_m,
            pressure_closed_inner_slope,
            pressure_closed_outer_slope,
        ) = _pressure_closed_viscocapillary_deficit_profile(
            fitted_radius_m,
            baseline_height_m,
            center_radius_m=body_fitted_center_radius_m,
            target_deficit_ul=body_fitted_target_deficit_ul,
            contact_speed_m_s=contact_speed_m_s,
            hydraulic_rate_m3_s=hydraulic_rate_m3_s,
            film_pressure_pa=_PARTITIONED_FILM_INNER_PRESSURE_PA,
            bridge_pressure_pa=float(bridge_pressure_pa),
            surface_tension_n_m=float(config.surface_tension_n_m),
            viscosity_pa_s=float(config.viscosity_pa_s),
        )
        body_fitted_pressure_closed_depth_um = float(
            pressure_closed_depth_m * 1.0e6
        )
        body_fitted_pressure_closed_inner_slope = float(
            pressure_closed_inner_slope
        )
        body_fitted_pressure_closed_outer_slope = float(
            pressure_closed_outer_slope
        )
        if pressure_closed_depth_m < 0.0:
            # Once the capillary-diffusion front has connected the local and
            # far film, use the Cox--Voinov logarithm for the resolved wedge:
            # m^3 = 9 Ca log(L/lambda).  The axisymmetric capillary--gravity
            # outer solution is K0(r/ell_c), so its local radial decay length
            # is L=ell_c*K0/K1.  lambda=h0 is the inner matching scale of the
            # resolved thin-film wedge.  Both lengths follow from material,
            # geometry, and initial film thickness; no measured trough enters.
            matching_inner_scale_m = h0
            bessel_coordinate = max(
                body_fitted_center_radius_m / capillary_length,
                1.0e-12,
            )
            matching_outer_scale_m = capillary_length * float(
                k0(bessel_coordinate) / max(k1(bessel_coordinate), 1.0e-30)
            )
            cox_logarithm = math.log(
                max(
                    matching_outer_scale_m / matching_inner_scale_m,
                    1.0 + 1.0e-12,
                )
            )
            if bool(
                _PARTITIONED_CONNECTED_SUPPORT_ACTIVE
                or _PARTITIONED_FAR_SUPPLY_ACTIVATION > 0.0
            ):
                cox_local_target_ul = max(
                    body_fitted_early_deficit_ul,
                    0.0,
                )
                (
                    cox_height_m,
                    cox_depth_m,
                    cox_inner_slope,
                    cox_outer_slope,
                ) = _pressure_closed_viscocapillary_deficit_profile(
                    fitted_radius_m,
                    baseline_height_m,
                    center_radius_m=body_fitted_center_radius_m,
                    target_deficit_ul=cox_local_target_ul,
                    contact_speed_m_s=contact_speed_m_s,
                    hydraulic_rate_m3_s=hydraulic_rate_m3_s,
                    film_pressure_pa=0.0,
                    bridge_pressure_pa=0.0,
                    surface_tension_n_m=float(config.surface_tension_n_m),
                    viscosity_pa_s=float(config.viscosity_pa_s),
                    slope_cube_multiplier=3.0 * cox_logarithm,
                )
                # A finite-height wedge has a finite local-volume capacity.
                # If the accepted local inventory exceeds it, transfer only
                # that excess to the broad far-film arm.  This is a
                # conservative solvability condition, not a fitted cap.
                if cox_depth_m < 0.0 and cox_local_target_ul > 0.0:
                    admissible_low_ul = 0.0
                    inadmissible_high_ul = cox_local_target_ul
                    for _ in range(36):
                        candidate_target_ul = 0.5 * (
                            admissible_low_ul + inadmissible_high_ul
                        )
                        candidate = (
                            _pressure_closed_viscocapillary_deficit_profile(
                                fitted_radius_m,
                                baseline_height_m,
                                center_radius_m=body_fitted_center_radius_m,
                                target_deficit_ul=candidate_target_ul,
                                contact_speed_m_s=contact_speed_m_s,
                                hydraulic_rate_m3_s=hydraulic_rate_m3_s,
                                film_pressure_pa=0.0,
                                bridge_pressure_pa=0.0,
                                surface_tension_n_m=float(
                                    config.surface_tension_n_m
                                ),
                                viscosity_pa_s=float(config.viscosity_pa_s),
                                slope_cube_multiplier=3.0 * cox_logarithm,
                            )
                        )
                        if candidate[1] >= 0.0:
                            admissible_low_ul = candidate_target_ul
                            (
                                cox_height_m,
                                cox_depth_m,
                                cox_inner_slope,
                                cox_outer_slope,
                            ) = candidate
                        else:
                            inadmissible_high_ul = candidate_target_ul
                    cox_local_target_ul = admissible_low_ul
                if cox_depth_m >= 0.0:
                    # The local inventory makes the narrow viscocapillary V.
                    # The accepted far inventory is transported over the
                    # independently computed diffusion-front distance; it
                    # must broaden the outer arm instead of deepening the
                    # same one-cell trough indefinitely.
                    # Capillary--gravity disturbances decay exponentially on
                    # ell_c.  Retain the resolved arm until 90% recovery,
                    # L_90=-ell_c*log(0.1); farther liquid remains in the
                    # conservative far inventory but does not deepen the
                    # local V-shaped observable.
                    outer_width_m = min(
                        max(
                            _PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M,
                            capillary_length,
                        ),
                        -math.log(0.1) * capillary_length,
                        float(config.substrate_radius_mm) * 1.0e-3
                        - body_fitted_center_radius_m,
                    )
                    outer_offset_m = (
                        fitted_radius_m - body_fitted_center_radius_m
                    )
                    far_coordinate = np.clip(
                        outer_offset_m / max(outer_width_m, 1.0e-30),
                        0.0,
                        1.0,
                    )
                    outer_far_kernel = np.where(
                        (outer_offset_m >= 0.0)
                        & (outer_offset_m <= outer_width_m),
                        1.0
                        - 3.0 * far_coordinate**2
                        + 2.0 * far_coordinate**3,
                        0.0,
                    )
                    # The far-film deficit reaches the V tip from the outer
                    # film, but the accepted free surface must remain one
                    # continuous interface through that tip.  Match the same
                    # tip depth into the bridge-side turn across one physical
                    # film thickness h0.  The smoothstep has zero derivative
                    # at both ends, so this is a C1 capillary matching layer,
                    # not a fitted trough width or a display correction.
                    inner_width_m = h0
                    inner_coordinate = np.clip(
                        (outer_offset_m + inner_width_m)
                        / max(inner_width_m, 1.0e-30),
                        0.0,
                        1.0,
                    )
                    inner_far_kernel = np.where(
                        (outer_offset_m >= -inner_width_m)
                        & (outer_offset_m < 0.0),
                        3.0 * inner_coordinate**2
                        - 2.0 * inner_coordinate**3,
                        0.0,
                    )
                    far_kernel = np.maximum(
                        inner_far_kernel,
                        outer_far_kernel,
                    )
                    far_target_m3 = max(
                        body_fitted_late_deficit_ul
                        + body_fitted_early_deficit_ul
                        - cox_local_target_ul,
                        0.0,
                    ) * 1.0e-9
                    far_depth_low_m = 0.0
                    far_depth_high_m = max(
                        float(np.max(cox_height_m)),
                        h0,
                    )
                    for _ in range(64):
                        far_depth_mid_m = 0.5 * (
                            far_depth_low_m + far_depth_high_m
                        )
                        far_trial_m = np.maximum(
                            cox_height_m - far_depth_mid_m * far_kernel,
                            FILM_MINIMUM_HEIGHT_M,
                        )
                        far_volume_m3 = float(
                            2.0
                            * math.pi
                            * np.trapezoid(
                                fitted_radius_m
                                * (cox_height_m - far_trial_m),
                                fitted_radius_m,
                            )
                        )
                        if far_volume_m3 < far_target_m3:
                            far_depth_low_m = far_depth_mid_m
                        else:
                            far_depth_high_m = far_depth_mid_m
                    fitted_height_m = np.maximum(
                        cox_height_m - far_depth_high_m * far_kernel,
                        FILM_MINIMUM_HEIGHT_M,
                    )
                    center_height_m = float(
                        np.interp(
                            body_fitted_center_radius_m,
                            fitted_radius_m,
                            fitted_height_m,
                        )
                    )
                    pressure_closed_depth_m = max(
                        float(
                            np.interp(
                                body_fitted_center_radius_m,
                                fitted_radius_m,
                                baseline_height_m,
                            )
                        )
                        - center_height_m,
                        0.0,
                    )
                    pressure_closed_inner_slope = cox_inner_slope
                    pressure_closed_outer_slope = cox_outer_slope
            if pressure_closed_depth_m < 0.0:
                fallback_inner_width_m = math.sqrt(h0 * capillary_length)
                fallback_outer_width_m = capillary_length
                fallback_kernel = _axisymmetric_triangular_kernel(
                    fitted_radius_m,
                    body_fitted_center_radius_m,
                    fallback_inner_width_m,
                    fallback_outer_width_m,
                )
                fallback_depth_m = (
                    body_fitted_target_deficit_ul
                    * 1.0e-9
                    / _axisymmetric_triangular_kernel_integral_m2(
                        body_fitted_center_radius_m,
                        fallback_inner_width_m,
                        fallback_outer_width_m,
                    )
                )
                fitted_height_m = (
                    baseline_height_m - fallback_depth_m * fallback_kernel
                )
        body_fitted_pressure_closed_depth_um = float(
            pressure_closed_depth_m * 1.0e6
        )
        body_fitted_pressure_closed_inner_slope = float(
            pressure_closed_inner_slope
        )
        body_fitted_pressure_closed_outer_slope = float(
            pressure_closed_outer_slope
        )
        changed = baseline_height_m - fitted_height_m > 1.0e-12
        if np.any(changed):
            changed_indices = np.flatnonzero(changed)
            inner_width_m = max(
                body_fitted_center_radius_m
                - float(fitted_radius_m[changed_indices[0]]),
                1.0e-9,
            )
            outer_support_width_m = max(
                float(fitted_radius_m[changed_indices[-1]])
                - body_fitted_center_radius_m,
                1.0e-9,
            )
        else:
            inner_width_m = math.sqrt(h0 * capillary_length)
            outer_support_width_m = inner_width_m
        body_fitted_outer_support_radius_mm = float(
            (
                body_fitted_center_radius_m
                + outer_support_width_m
            )
            * 1.0e3
        )
        local_support_mask = (
            fitted_radius_m
            >= body_fitted_center_radius_m - inner_width_m - 1.0e-15
        ) & (
            fitted_radius_m
            <= body_fitted_center_radius_m + outer_support_width_m + 1.0e-15
        )
        if bool(BODY_FITTED_CONSERVATIVE_C1_JOIN_ENABLED):
            bridge_order = bridge_rows[np.argsort(new_r[bridge_rows])]
            local_indices = np.flatnonzero(local_support_mask)
            preliminary_min_index = int(
                local_indices[
                    np.argmin(fitted_height_m[local_support_mask])
                ]
            )
            join_end_index = preliminary_min_index - 1
            join_start_target_m = float(rim_r - inner_width_m)
            join_start_position = int(
                np.argmin(
                    np.abs(new_r[bridge_order] - join_start_target_m)
                )
            )
            if (
                join_start_position >= 1
                and bridge_order.size - join_start_position >= 3
                and join_end_index >= 2
            ):
                row_radius_now, row_height_now = parent._ring_coordinates(
                    surface,
                    rings,
                )
                selected_bridge_rows = bridge_order[join_start_position:]
                previous_bridge_row = int(
                    bridge_order[join_start_position - 1]
                )
                start_bridge_row = int(selected_bridge_rows[0])
                start_slope = float(
                    (
                        row_height_now[start_bridge_row]
                        - row_height_now[previous_bridge_row]
                    )
                    / (
                        new_r[start_bridge_row]
                        - new_r[previous_bridge_row]
                    )
                )
                end_slope = float(
                    (
                        fitted_height_m[join_end_index + 1]
                        - fitted_height_m[join_end_index]
                    )
                    / (
                        fitted_radius_m[join_end_index + 1]
                        - fitted_radius_m[join_end_index]
                    )
                )
                join_radius_m = np.concatenate(
                    (
                        np.asarray(new_r[selected_bridge_rows], dtype=float),
                        fitted_radius_m[: join_end_index + 1],
                    )
                )
                join_height_m = np.concatenate(
                    (
                        np.asarray(
                            row_height_now[selected_bridge_rows],
                            dtype=float,
                        ),
                        fitted_height_m[: join_end_index + 1],
                    )
                )
                try:
                    joined_height_m, join_diag = (
                        _conservative_c1_bridge_film_join_profile(
                            join_radius_m,
                            join_height_m,
                            split_index=int(selected_bridge_rows.size - 1),
                            start_slope=start_slope,
                            end_slope=end_slope,
                        )
                    )
                except RuntimeError:
                    # Separate bridge/film volume bubbles can become
                    # non-monotone when the accepted interface is already
                    # close to flat.  The region label is not a material
                    # barrier, so fall back to one C1 join that conserves the
                    # combined annular liquid volume exactly.
                    joined_height_m, combined_diag = (
                        _conservative_monotone_c1_join_profile(
                            join_radius_m,
                            join_height_m,
                            split_index=int(selected_bridge_rows.size - 1),
                            start_slope=start_slope,
                            end_slope=end_slope,
                        )
                    )
                    split_index = int(selected_bridge_rows.size - 1)
                    side_residuals_m3 = []
                    for first, last in (
                        (0, split_index),
                        (split_index, join_radius_m.size - 1),
                    ):
                        side_residuals_m3.append(
                            float(
                                2.0
                                * math.pi
                                * np.trapezoid(
                                    join_radius_m[first : last + 1]
                                    * (
                                        joined_height_m[first : last + 1]
                                        - join_height_m[first : last + 1]
                                    ),
                                    join_radius_m[first : last + 1],
                                )
                            )
                        )
                    join_diag = {
                        "bridge_volume_residual_m3": side_residuals_m3[0],
                        "film_volume_residual_m3": side_residuals_m3[1],
                        "prejoin_angle_jump_rad": float(
                            combined_diag["prejoin_angle_jump_rad"]
                        ),
                        "postjoin_p1_angle_jump_rad": float(
                            combined_diag["postjoin_p1_angle_jump_rad"]
                        ),
                    }
                bridge_join_count = int(selected_bridge_rows.size)
                for row_id, height in zip(
                    selected_bridge_rows,
                    joined_height_m[:bridge_join_count],
                ):
                    surface[np.asarray(rings[int(row_id)], dtype=int), 2] = (
                        float(height)
                    )
                fitted_height_m[: join_end_index + 1] = joined_height_m[
                    bridge_join_count:
                ]
                body_fitted_c1_join_active = 1.0
                body_fitted_c1_join_start_radius_mm = float(
                    join_radius_m[0] * 1.0e3
                )
                body_fitted_c1_join_end_radius_mm = float(
                    join_radius_m[-1] * 1.0e3
                )
                body_fitted_c1_join_pre_angle_jump_rad = float(
                    join_diag["prejoin_angle_jump_rad"]
                )
                body_fitted_c1_join_post_angle_jump_rad = float(
                    join_diag["postjoin_p1_angle_jump_rad"]
                )
                body_fitted_c1_join_bridge_volume_residual_ul = float(
                    join_diag["bridge_volume_residual_m3"] * 1.0e9
                )
                body_fitted_c1_join_film_volume_residual_ul = float(
                    join_diag["film_volume_residual_m3"] * 1.0e9
                )
        support_indices = np.flatnonzero(local_support_mask)
        support_min_index = int(
            support_indices[
                np.argmin(fitted_height_m[local_support_mask])
            ]
        )
        outer_arm_indices = support_indices[
            support_indices >= support_min_index
        ]
        if outer_arm_indices.size >= 2:
            # The finite circular substrate has a small physical downward
            # Bessel taper.  Test monotone recovery of the *deficit relative
            # to that baseline*, not monotonicity of absolute film height.
            outer_arm_deficit_m = (
                baseline_height_m[outer_arm_indices]
                - fitted_height_m[outer_arm_indices]
            )
            body_fitted_outer_arm_reversal_count = float(
                np.count_nonzero(np.diff(outer_arm_deficit_m) > 1.0e-12)
            )
            if body_fitted_outer_arm_reversal_count > 0.0:
                raise RuntimeError(
                    "Case77 full-support capillary film has a nonphysical "
                    "outer-arm reversal; reject this timestep"
                )
        if np.any(
            fitted_height_m[local_support_mask]
            <= case28.TETRA_MIN_LAYER_HEIGHT_M
        ):
            raise RuntimeError(
                "Case77 body-fitted capillary profile violates the resolved "
                "positive-gap constraint; reject this timestep "
                f"(hmin={float(np.min(fitted_height_m[local_support_mask]))*1.0e6:.6g} um, "
                f"deficit={cumulative_hydraulic_deficit_ul:.6g} uL, "
                f"early={body_fitted_early_deficit_ul:.6g} uL, "
                f"late={body_fitted_late_deficit_ul:.6g} uL, "
                f"center={body_fitted_center_radius_m*1.0e3:.6g} mm)"
            )
        # Write the complete current support.  Outside it, the capillary--
        # gravity disturbance is screened and the untouched finite substrate
        # retains its computed baseline profile.  Leaving old depleted rows
        # in place here was the source of the accumulating 6--10 mm sag.
        accepted_film_height_m = np.where(
            local_support_mask,
            fitted_height_m,
            baseline_height_m,
        )
        for row_id, height in zip(film_rows, accepted_film_height_m):
            surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(
                height
            )

        # Match the moving fitted-film support to the already solved far film.
        # The transition starts one existing viscocapillary matching length
        # sqrt(h0*ell_c) inside the boundary.  Its outer endpoint is the first
        # resolved row, no farther than the computed outer capillary--gravity
        # ridge crest, that permits a monotone C1 join with exact combined
        # axisymmetric-volume conservation.
        if bool(
            BODY_FITTED_OUTER_CONSERVATIVE_C1_JOIN_ENABLED
            and pressure_closed_depth_m < 0.0
        ):
            support_indices = np.flatnonzero(local_support_mask)
            outer_split_index = int(support_indices[-1])
            outer_boundary_m = float(fitted_radius_m[outer_split_index])
            maximum_outer_join_start_index = int(
                np.argmin(
                    np.abs(
                        fitted_radius_m
                        - (outer_boundary_m - inner_width_m)
                    )
                )
            )
            minimum_outer_join_start_index = int(
                np.argmin(
                    np.abs(
                        fitted_radius_m
                        - (outer_boundary_m - capillary_length)
                    )
                )
            )
            minimum_outer_join_end_index = int(
                np.argmin(
                    np.abs(
                        fitted_radius_m
                        - (outer_boundary_m + inner_width_m)
                    )
                )
            )
            if (
                minimum_outer_join_start_index >= 1
                and maximum_outer_join_start_index < outer_split_index
                and minimum_outer_join_end_index > outer_split_index
                and minimum_outer_join_end_index
                < fitted_radius_m.size - 1
            ):
                row_radius_now, row_height_now = parent._ring_coordinates(
                    surface,
                    rings,
                )
                far_crest_search = np.asarray(
                    row_height_now[
                        film_rows[outer_split_index + 1 : -1]
                    ],
                    dtype=float,
                )
                maximum_outer_join_end_index = int(
                    outer_split_index
                    + 1
                    + np.argmax(far_crest_search)
                )
                maximum_outer_join_end_index = max(
                    maximum_outer_join_end_index,
                    minimum_outer_join_end_index,
                )
                has_rise_before_boundary = bool(
                    row_height_now[film_rows[outer_split_index]]
                    - row_height_now[film_rows[outer_split_index - 1]]
                    > 1.0e-12
                )
                has_drop_at_boundary = bool(
                    row_height_now[film_rows[outer_split_index + 1]]
                    - row_height_now[film_rows[outer_split_index]]
                    < -1.0e-12
                )
                has_far_recovery = bool(
                    np.any(
                        np.diff(
                            row_height_now[
                                film_rows[
                                    outer_split_index
                                    + 1 : maximum_outer_join_end_index
                                    + 1
                                ]
                            ]
                        )
                        > 1.0e-12
                    )
                )
                if (
                    has_rise_before_boundary
                    and has_drop_at_boundary
                    and has_far_recovery
                ):
                    accepted_outer_join = None
                    candidate_windows = [
                        (candidate_start_index, candidate_end_index)
                        for candidate_start_index in range(
                            minimum_outer_join_start_index,
                            maximum_outer_join_start_index + 1,
                        )
                        for candidate_end_index in range(
                            minimum_outer_join_end_index,
                            maximum_outer_join_end_index + 1,
                        )
                    ]
                    candidate_windows.sort(
                        key=lambda bounds: (
                            fitted_radius_m[bounds[1]]
                            - fitted_radius_m[bounds[0]],
                            bounds[1] - bounds[0],
                        )
                    )
                    for (
                        candidate_start_index,
                        candidate_end_index,
                    ) in candidate_windows:
                        outer_join_rows = film_rows[
                            candidate_start_index : candidate_end_index + 1
                        ]
                        outer_join_radius_m = np.asarray(
                            row_radius_now[outer_join_rows],
                            dtype=float,
                        )
                        outer_join_height_m = np.asarray(
                            row_height_now[outer_join_rows],
                            dtype=float,
                        )
                        previous_row = int(
                            film_rows[candidate_start_index - 1]
                        )
                        first_join_row = int(outer_join_rows[0])
                        next_row = int(film_rows[candidate_end_index + 1])
                        last_join_row = int(outer_join_rows[-1])
                        outer_start_slope = float(
                            (
                                row_height_now[first_join_row]
                                - row_height_now[previous_row]
                            )
                            / (
                                row_radius_now[first_join_row]
                                - row_radius_now[previous_row]
                            )
                        )
                        outer_end_slope = float(
                            (
                                row_height_now[next_row]
                                - row_height_now[last_join_row]
                            )
                            / (
                                row_radius_now[next_row]
                                - row_radius_now[last_join_row]
                            )
                        )
                        local_split_index = int(
                            outer_split_index - candidate_start_index
                        )
                        try:
                            candidate_height_m, candidate_diag = (
                                _conservative_monotone_c1_join_profile(
                                    outer_join_radius_m,
                                    outer_join_height_m,
                                    split_index=local_split_index,
                                    start_slope=outer_start_slope,
                                    end_slope=outer_end_slope,
                                )
                            )
                        except RuntimeError:
                            continue
                        accepted_outer_join = (
                            outer_join_rows,
                            outer_join_radius_m,
                            candidate_height_m,
                            candidate_diag,
                        )
                        break
                    if accepted_outer_join is None:
                        raise RuntimeError(
                            "Case77 outer C1 event has no monotone "
                            "volume-conservative match before the computed "
                            "outer capillary-gravity crest "
                            f"at t={float(time_s or 0.0):.9g} s"
                        )
                    (
                        outer_join_rows,
                        outer_join_radius_m,
                        outer_joined_height_m,
                        outer_join_diag,
                    ) = accepted_outer_join
                    for row_id, height in zip(
                        outer_join_rows,
                        outer_joined_height_m,
                    ):
                        surface[
                            np.asarray(rings[int(row_id)], dtype=int),
                            2,
                        ] = float(height)
                    body_fitted_outer_c1_join_active = 1.0
                    body_fitted_outer_c1_join_start_radius_mm = float(
                        outer_join_radius_m[0] * 1.0e3
                    )
                    body_fitted_outer_c1_join_end_radius_mm = float(
                        outer_join_radius_m[-1] * 1.0e3
                    )
                    body_fitted_outer_c1_join_pre_angle_jump_rad = float(
                        outer_join_diag["prejoin_angle_jump_rad"]
                    )
                    body_fitted_outer_c1_join_post_angle_jump_rad = float(
                        outer_join_diag["postjoin_p1_angle_jump_rad"]
                    )
                    body_fitted_outer_c1_join_volume_residual_ul = float(
                        outer_join_diag["volume_residual_m3"] * 1.0e9
                    )
                    body_fitted_outer_arm_reversal_count = float(
                        np.count_nonzero(
                            np.diff(outer_joined_height_m) < -1.0e-12
                        )
                    )
        local_indices = np.flatnonzero(local_support_mask)
        fitted_min_index = int(
            local_indices[
                np.argmin(fitted_height_m[local_support_mask])
            ]
        )
        body_fitted_hmin_um = float(
            fitted_height_m[fitted_min_index] * 1.0e6
        )
        body_fitted_rmin_mm = float(
            fitted_radius_m[fitted_min_index] * 1.0e3
        )
        represented_deficit_m3 = (
            2.0
            * math.pi
            * np.trapezoid(
                fitted_radius_m
                * np.maximum(
                    baseline_height_m - fitted_height_m,
                    0.0,
                ),
                fitted_radius_m,
            )
        )
        body_fitted_volume_residual_ul = float(
            represented_deficit_m3 * 1.0e9
            - body_fitted_target_deficit_ul
        )

    # The outer boundary belongs to the fixed finite substrate.  Any tiny
    # resulting total-volume defect is restored in the far reservoir only.
    surface[rings[outer_row], 2] = state.outer_surface_reference_z
    far_adjustable = film_rows[
        (new_r[film_rows] >= first_film_r + 2.0 * capillary_length)
        & (film_rows != outer_row)
    ]
    if far_adjustable.size < 3:
        far_adjustable = film_rows[
            (np.arange(film_rows.size) >= max(film_rows.size - 6, 0))
            & (film_rows != outer_row)
        ]
    correction_m, volume_residual_ul = _far_volume_restoration(
        surface,
        state,
        rings,
        far_adjustable,
        target_volume_ul,
    )

    final_film_r, final_film_h = parent._ring_coordinates(surface, rings)
    trial_radius = np.asarray(final_film_r[film_rows], dtype=float)
    trial_height = np.asarray(final_film_h[film_rows], dtype=float)
    current_bridge_volume_ul = float(
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            ring_region,
            config,
        )
    )
    current_junction_r = float(trial_radius[0])
    diag = {
        "case29_neck_concentration": 0.0,
        "case29_local_capture_ul": 0.0,
        "case29_neck_min_radius_mm": 0.0,
        "case29_volume_residual_ul": float(volume_residual_ul),
        "case29_neck_rings": 0.0,
        "case29_recovery_width_mm": 0.0,
        "case29_recovery_local_share": 0.0,
        "case29_recovery_late_progress": 0.0,
        "case29_recovery_target_ul": 0.0,
        "case29_recovery_amplitude_um": 0.0,
        "case62_partitioned_film_active": 1.0,
        "case62_bridge_rate_ul_s": float(bridge_rate_m3_s * 1.0e9),
        "case62_swept_rate_ul_s": float(swept_rate_m3_s * 1.0e9),
        "case62_hydraulic_junction_rate_ul_s": float(
            hydraulic_rate_m3_s * 1.0e9
        ),
        "case62_contact_speed_m_s": float(contact_speed_m_s),
        "case62_viscocapillary_width_mm": float(
            viscocapillary_width_m * 1.0e3
        ),
        "case62_junction_height_um": float(junction_height_m * 1.0e6),
        "case62_matching_radius_mm": float(matching_radius * 1.0e3),
        "case62_physical_junction_radius_mm": float(
            matching_radius * 1.0e3
        ),
        "case62_two_sided_junction_layer": float(
            RESOLVE_TWO_SIDED_JUNCTION_LAYER
        ),
        "case62_flux_derived_capillary_micro_layer": float(
            USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
        ),
        "case62_hydraulic_pressure_drop_pa": float(
            hydraulic_pressure_drop_pa
        ),
        "case62_pr35_junction_pressure_continuity": float(
            junction_pressure_bc_pa is not None
            and USE_PR35_JUNCTION_PRESSURE_CONTINUITY
        ),
        "case77_heron_junction_pressure_continuity": float(
            junction_pressure_bc_pa is not None
            and USE_HERON_JUNCTION_PRESSURE_CONTINUITY
        ),
        "case77_heron_bridge_pressure_pa": float(
            heron_bridge_pressure_pa
        ),
        "case62_pr35_bridge_pressure_pa": float(
            pr35_bridge_pressure_pa
        ),
        "case62_pr35_far_film_pressure_pa": float(
            pr35_far_film_pressure_pa
        ),
        "case62_pr35_bridge_minus_far_pressure_pa": float(
            pr35_bridge_minus_far_pressure_pa
        ),
        "case62_film_junction_pressure_bc_pa": float(
            junction_pressure_bc_pa
            if junction_pressure_bc_pa is not None
            else float("nan")
        ),
        "case62_pressure_driven_junction_flux_active": float(
            USE_PRESSURE_DRIVEN_JUNCTION_FLUX
        ),
        "case62_pressure_driven_junction_flux_ul_s": float(
            pressure_driven_flux_ul_s
        ),
        "case62_pressure_driven_bridge_pressure_pa": float(
            pressure_driven_bridge_pressure_pa
        ),
        "case62_pressure_driven_donor_pressure_pa": float(
            pressure_driven_donor_pressure_pa
        ),
        "case62_pressure_driven_pressure_drop_pa": float(
            pressure_driven_donor_pressure_pa
            - pressure_driven_bridge_pressure_pa
        ),
        "case62_pressure_driven_resistance_pa_s_m3": float(
            pressure_driven_resistance_pa_s_m3
        ),
        "case62_pressure_driven_reconstruction_hmin_um": float(
            pressure_driven_hmin_um
        ),
        "case62_pressure_driven_turn_resistance_pa_s_m2": float(
            pressure_driven_turn_resistance_pa_s_m2
        ),
        "case77_pressure_driven_matching_resistance_pa_s_m2": float(
            pressure_driven_matching_resistance_pa_s_m2
        ),
        "case77_pressure_driven_total_local_resistance_pa_s_m2": float(
            pressure_driven_total_local_resistance_pa_s_m2
        ),
        "case62_pressure_driven_boundary_radius_mm": float(
            pressure_driven_boundary_radius_mm
        ),
        "case62_pressure_driven_film_solve_active": float(
            pressure_flux_solved
        ),
        "case62_pressure_flux_bridge_rate_residual_ul_s": float(
            bridge_rate_m3_s * 1.0e9 - pressure_driven_flux_ul_s
            if pressure_flux_solved
            else 0.0
        ),
        "case62_capillary_core_radius_mm": float(
            capillary_core_radius_m * 1.0e3
            if math.isfinite(capillary_core_radius_m)
            else 0.0
        ),
        "case62_capillary_core_half_width_mm": float(
            capillary_core_half_width_m * 1.0e3
        ),
        "case62_vlike_support_half_width_mm": float(
            vlike_support_half_width_m * 1.0e3
        ),
        "case62_viscocapillary_side_slope": float(
            viscocapillary_side_slope
        ),
        "case62_micro_layer_active_nodes": float(
            micro_layer_active_nodes
        ),
        "case62_micro_layer_volume_redistribution_ul": float(
            micro_layer_volume_redistribution_ul
        ),
        "case62_far_volume_correction_um": float(correction_m * 1.0e6),
        "case62_total_volume_residual_ul": float(volume_residual_ul),
        "case62_film_solve_failed": float(solve_failed),
        "case62_retained_case61_neck_rows": float(retained_neck_rows),
        "case62_retained_case61_neck_weight": float(
            retained_weight if retained_supply_surface is not None else 0.0
        ),
        "case62_finite_supply_cap_ul": float(
            _PARTITIONED_FINITE_SUPPLY_CAP_UL
            if _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
            else 0.0
        ),
        "case77_flux_supply_budget_reference_ul": float(
            max(float(parent._FLUX_SUPPLY_BUDGET_UL), 0.0)
        ),
        "case62_cumulative_hydraulic_deficit_ul": float(
            cumulative_hydraulic_deficit_ul
        ),
        "case62_supply_saturation_time_s": float(saturation_time_s),
        "case62_supply_saturated": float(saturation_time_s > 0.0),
        "case77_unified_supply_active": float(
            USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
        ),
        "case77_far_supply_activation": float(
            _PARTITIONED_FAR_SUPPLY_ACTIVATION
        ),
        "case77_capillary_diffusion_front_mm": float(
            _PARTITIONED_CAPILLARY_DIFFUSION_FRONT_M * 1.0e3
        ),
        "case77_far_supply_flux_ul_s": float(
            _PARTITIONED_FAR_SUPPLY_FLUX_UL_S
        ),
        "case77_far_similarity_flux_ul_s": float(
            _PARTITIONED_FAR_SIMILARITY_FLUX_UL_S
        ),
        "case77_far_donor_height_um": float(
            _PARTITIONED_FAR_DONOR_HEIGHT_M * 1.0e6
        ),
        "case77_equilibrium_pressure_supply_factor": float(
            _PARTITIONED_PRESSURE_SUPPLY_FACTOR
        ),
        "case77_two_sided_series_transmission": float(
            _PARTITIONED_TWO_ARM_TRANSMISSION
        ),
        "case77_unresolved_two_arm_supply_resistance_active": float(
            APPLY_UNRESOLVED_TWO_ARM_SUPPLY_RESISTANCE
        ),
        "case77_local_deficit_inventory_ul": float(
            local_deficit_inventory_ul
        ),
        "case77_far_deficit_inventory_ul": float(
            far_deficit_inventory_ul
        ),
        "case77_far_deficit_rate_ul_s": float(
            far_deficit_rate_ul_s
        ),
        "case77_inventory_conservation_residual_ul": float(
            local_deficit_inventory_ul
            + far_deficit_inventory_ul
            - cumulative_hydraulic_deficit_ul
        ),
        "case77_junction_throat_width_mm": float(
            _PARTITIONED_JUNCTION_THROAT_WIDTH_M * 1.0e3
        ),
        "case77_junction_throat_height_um": float(
            _PARTITIONED_JUNCTION_THROAT_HEIGHT_M * 1.0e6
        ),
        "case77_junction_throat_mobility": float(
            _PARTITIONED_JUNCTION_THROAT_MOBILITY
        ),
        "case77_full_connection_deficit_base_ul": float(
            _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL
            if _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL is not None
            else 0.0
        ),
        "case77_body_fitted_junction_active": float(
            body_fitted_refinement_ready
            and math.isfinite(body_fitted_center_radius_m)
        ),
        "case77_body_fitted_equilibrium_contact_radius_mm": float(
            body_fitted_equilibrium_contact_radius_m * 1.0e3
        ),
        "case77_body_fitted_center_radius_mm": float(
            body_fitted_center_radius_m * 1.0e3
        ),
        "case77_body_fitted_transition_time_s": float(
            body_fitted_transition_time_s
        ),
        "case77_body_fitted_transition_deficit_ul": float(
            body_fitted_transition_deficit_ul
        ),
        "case77_body_fitted_early_deficit_ul": float(
            body_fitted_early_deficit_ul
        ),
        "case77_body_fitted_late_deficit_ul": float(
            body_fitted_late_deficit_ul
        ),
        "case77_body_fitted_hmin_um": float(body_fitted_hmin_um),
        "case77_body_fitted_rmin_mm": float(body_fitted_rmin_mm),
        "case77_body_fitted_volume_residual_ul": float(
            body_fitted_volume_residual_ul
        ),
        "case77_body_fitted_geometric_deficit_ul": float(
            body_fitted_geometric_deficit_ul
        ),
        "case77_body_fitted_target_deficit_ul": float(
            body_fitted_target_deficit_ul
        ),
        "case77_body_fitted_c1_join_active": float(
            body_fitted_c1_join_active
        ),
        "case77_body_fitted_c1_join_start_radius_mm": float(
            body_fitted_c1_join_start_radius_mm
        ),
        "case77_body_fitted_c1_join_end_radius_mm": float(
            body_fitted_c1_join_end_radius_mm
        ),
        "case77_body_fitted_c1_join_pre_angle_jump_rad": float(
            body_fitted_c1_join_pre_angle_jump_rad
        ),
        "case77_body_fitted_c1_join_post_angle_jump_rad": float(
            body_fitted_c1_join_post_angle_jump_rad
        ),
        "case77_body_fitted_c1_join_bridge_volume_residual_ul": float(
            body_fitted_c1_join_bridge_volume_residual_ul
        ),
        "case77_body_fitted_c1_join_film_volume_residual_ul": float(
            body_fitted_c1_join_film_volume_residual_ul
        ),
        "case77_body_fitted_outer_support_radius_mm": float(
            body_fitted_outer_support_radius_mm
        ),
        "case77_body_fitted_outer_arm_reversal_count": float(
            body_fitted_outer_arm_reversal_count
        ),
        "case77_body_fitted_outer_c1_join_active": float(
            body_fitted_outer_c1_join_active
        ),
        "case77_body_fitted_outer_c1_join_start_radius_mm": float(
            body_fitted_outer_c1_join_start_radius_mm
        ),
        "case77_body_fitted_outer_c1_join_end_radius_mm": float(
            body_fitted_outer_c1_join_end_radius_mm
        ),
        "case77_body_fitted_outer_c1_join_pre_angle_jump_rad": float(
            body_fitted_outer_c1_join_pre_angle_jump_rad
        ),
        "case77_body_fitted_outer_c1_join_post_angle_jump_rad": float(
            body_fitted_outer_c1_join_post_angle_jump_rad
        ),
        "case77_body_fitted_outer_c1_join_volume_residual_ul": float(
            body_fitted_outer_c1_join_volume_residual_ul
        ),
        "case77_body_fitted_pressure_closed_depth_um": float(
            body_fitted_pressure_closed_depth_um
        ),
        "case77_body_fitted_pressure_closed_inner_slope": float(
            body_fitted_pressure_closed_inner_slope
        ),
        "case77_body_fitted_pressure_closed_outer_slope": float(
            body_fitted_pressure_closed_outer_slope
        ),
        "case77_body_fitted_film_pressure_pa": float(
            body_fitted_film_pressure_pa
        ),
        "case77_body_fitted_bridge_pressure_pa": float(
            body_fitted_bridge_pressure_pa
        ),
        "case62_skipped_redundant_pinned_bvp": float(
            supply_already_saturated
        ),
    }
    diag.update(film_diag)
    state.case29_last_diag = diag
    state.case62_trial_film_radius_m = trial_radius
    state.case62_trial_film_height_m = trial_height
    state.case62_trial_bridge_volume_ul = float(current_bridge_volume_ul)
    state.case62_trial_junction_radius_m = float(current_junction_r)
    state.case62_trial_hydraulic_deficit_ul = float(
        cumulative_hydraulic_deficit_ul
    )
    state.case62_trial_hydraulic_rate_ul_s = float(
        max(hydraulic_rate_m3_s * 1.0e9, 0.0)
    )
    state.case77_trial_local_deficit_inventory_ul = float(
        local_deficit_inventory_ul
    )
    state.case77_trial_far_deficit_inventory_ul = float(
        far_deficit_inventory_ul
    )
    state.case77_trial_far_deficit_rate_ul_s = float(
        far_deficit_rate_ul_s
    )
    state.case62_trial_supply_saturation_time_s = float(saturation_time_s)
    state.case77_trial_body_fitted_transition_time_s = float(
        body_fitted_transition_time_s
    )
    state.case77_trial_body_fitted_transition_deficit_ul = float(
        body_fitted_transition_deficit_ul
    )
    state.case62_trial_pressure_driven_flux_ul_s = float(
        pressure_driven_flux_ul_s
    )
    state.case77_trial_physical_film_radius_m = np.asarray(
        physical_film_radius_m,
        dtype=float,
    ).copy()
    state.case77_trial_physical_film_height_m = np.asarray(
        physical_film_height_m,
        dtype=float,
    ).copy()
    state.case77_trial_physical_film_flux_ul_s = float(
        physical_film_flux_ul_s
    )
    state.case77_trial_junction_impedance_pa_s_m2 = float(
        physical_film_diag.get(
            "case77_pressure_film_reference_resistance_pa_s_m2",
            getattr(
                state,
                "case77_accepted_junction_impedance_pa_s_m2",
                _RESTART_JUNCTION_IMPEDANCE_PA_S_M2,
            ),
        )
    )
    state.case77_trial_junction_impedance_reference_height_m = float(
        physical_film_diag.get(
            "case77_pressure_film_reference_height_um",
            1.0e6
            * float(
                getattr(
                    state,
                    "case77_accepted_junction_impedance_reference_height_m",
                    _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M,
                )
            ),
        )
        * 1.0e-6
    )
    state.case77_trial_junction_impedance_reference_width_m = float(
        physical_film_diag.get(
            "case77_pressure_film_reference_width_mm",
            1.0e3
            * float(
                getattr(
                    state,
                    "case77_accepted_junction_impedance_reference_width_m",
                    _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M,
                )
            ),
        )
        * 1.0e-3
    )
    state.case62_trial_diag = dict(diag)

    state.bottom_reference = np.array(surface, copy=True)
    state.bottom_reference[:, 2] = 0.0
    state.set_surface_points(surface)
    if step_dt <= 0.0:
        _commit_partitioned_state(state)


def _commit_partitioned_state(
    state: case28.TetraFreeSurfaceState,
) -> None:
    """Commit only the profile belonging to an accepted tetra step."""

    global _PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S
    global _PARTITIONED_LOCAL_SATURATION_TIME_S
    global _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL
    global _PARTITIONED_ACCEPTED_LOCAL_DEFICIT_INVENTORY_UL
    global _PARTITIONED_ACCEPTED_FAR_DEFICIT_INVENTORY_UL
    global _PARTITIONED_ACCEPTED_FAR_DEFICIT_RATE_UL_S
    global _RESTART_JUNCTION_IMPEDANCE_PA_S_M2
    global _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M
    global _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M
    global _RESTART_BODY_FITTED_TRANSITION_TIME_S
    global _RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL

    state.case62_accepted_film_radius_m = np.asarray(
        state.case62_trial_film_radius_m,
        dtype=float,
    ).copy()
    state.case62_accepted_film_height_m = np.asarray(
        state.case62_trial_film_height_m,
        dtype=float,
    ).copy()
    state.case62_accepted_bridge_volume_ul = float(
        state.case62_trial_bridge_volume_ul
    )
    state.case62_accepted_junction_radius_m = float(
        state.case62_trial_junction_radius_m
    )
    state.case62_accepted_hydraulic_deficit_ul = float(
        state.case62_trial_hydraulic_deficit_ul
    )
    _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL = float(
        state.case62_accepted_hydraulic_deficit_ul
    )
    state.case77_accepted_local_deficit_inventory_ul = float(
        state.case77_trial_local_deficit_inventory_ul
    )
    _PARTITIONED_ACCEPTED_LOCAL_DEFICIT_INVENTORY_UL = float(
        state.case77_accepted_local_deficit_inventory_ul
    )
    state.case77_accepted_far_deficit_inventory_ul = float(
        state.case77_trial_far_deficit_inventory_ul
    )
    _PARTITIONED_ACCEPTED_FAR_DEFICIT_INVENTORY_UL = float(
        state.case77_accepted_far_deficit_inventory_ul
    )
    state.case77_accepted_far_deficit_rate_ul_s = float(
        state.case77_trial_far_deficit_rate_ul_s
    )
    _PARTITIONED_ACCEPTED_FAR_DEFICIT_RATE_UL_S = float(
        state.case77_accepted_far_deficit_rate_ul_s
    )
    state.case62_accepted_hydraulic_rate_ul_s = float(
        state.case62_trial_hydraulic_rate_ul_s
    )
    state.case62_accepted_supply_saturation_time_s = float(
        state.case62_trial_supply_saturation_time_s
    )
    state.case77_accepted_body_fitted_transition_time_s = float(
        getattr(
            state,
            "case77_trial_body_fitted_transition_time_s",
            0.0,
        )
    )
    state.case77_accepted_body_fitted_transition_deficit_ul = float(
        getattr(
            state,
            "case77_trial_body_fitted_transition_deficit_ul",
            0.0,
        )
    )
    _RESTART_BODY_FITTED_TRANSITION_TIME_S = float(
        state.case77_accepted_body_fitted_transition_time_s
    )
    _RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL = float(
        state.case77_accepted_body_fitted_transition_deficit_ul
    )
    _PARTITIONED_LOCAL_SATURATION_TIME_S = float(
        state.case62_accepted_supply_saturation_time_s
    )
    state.case62_stop_at_saturation = bool(
        USE_TERMINAL_LOCAL_DONOR_CAP
        and not USE_PRESSURE_DRIVEN_JUNCTION_FLUX
        and state.case62_accepted_supply_saturation_time_s > 0.0
        and abs(state.case62_accepted_hydraulic_rate_ul_s) <= 1.0e-14
        and not USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
    )
    state.case62_accepted_pressure_driven_flux_ul_s = float(
        getattr(
            state,
            "case62_trial_pressure_driven_flux_ul_s",
            0.0,
        )
    )
    state.case77_accepted_physical_film_radius_m = np.asarray(
        state.case77_trial_physical_film_radius_m,
        dtype=float,
    ).copy()
    state.case77_accepted_physical_film_height_m = np.asarray(
        state.case77_trial_physical_film_height_m,
        dtype=float,
    ).copy()
    state.case77_accepted_physical_film_flux_ul_s = float(
        state.case77_trial_physical_film_flux_ul_s
    )
    state.case77_accepted_junction_impedance_pa_s_m2 = float(
        state.case77_trial_junction_impedance_pa_s_m2
    )
    state.case77_accepted_junction_impedance_reference_height_m = float(
        state.case77_trial_junction_impedance_reference_height_m
    )
    state.case77_accepted_junction_impedance_reference_width_m = float(
        state.case77_trial_junction_impedance_reference_width_m
    )
    _RESTART_JUNCTION_IMPEDANCE_PA_S_M2 = float(
        state.case77_accepted_junction_impedance_pa_s_m2
    )
    _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M = float(
        state.case77_accepted_junction_impedance_reference_height_m
    )
    _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M = float(
        state.case77_accepted_junction_impedance_reference_width_m
    )
    _PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S = float(
        state.case62_accepted_pressure_driven_flux_ul_s
    )
    state.case62_last_accepted_diag = dict(state.case62_trial_diag)
    if bool(USE_SUBGRID_JUNCTION_FLUX_CLOSURE):
        parent._FLUX_REFERENCE_SURFACE = (
            _partitioned_flux_reference_surface(state)
        )


_ORIGINAL_TRY_ACCEPT_VELOCITY_STEP = case28.try_accept_velocity_step
_ORIGINAL_SAVE_SNAPSHOT = case28.save_snapshot


def _try_accept_partitioned_velocity_step(*args, **kwargs):
    accepted, scale, quality = _ORIGINAL_TRY_ACCEPT_VELOCITY_STEP(
        *args,
        **kwargs,
    )
    state = args[0]
    if bool(accepted):
        _commit_partitioned_state(state)
    return accepted, scale, quality


def _save_partitioned_snapshot(
    state: case28.TetraFreeSurfaceState,
    ring_region: np.ndarray,
    step: int,
    time_s: float,
    bridge_volume_ul: float,
    config: base.RealMeshEvolutionConfig,
    out_dir: Path,
) -> None:
    """Save the tetra state and its conservative physical-film companion."""

    _ORIGINAL_SAVE_SNAPSHOT(
        state,
        ring_region,
        step,
        time_s,
        bridge_volume_ul,
        config,
        out_dir,
    )
    if not bool(USE_MOVING_BOUNDARY_PRESSURE_FILM):
        return
    radius_m = getattr(
        state,
        "case77_accepted_physical_film_radius_m",
        None,
    )
    height_m = getattr(
        state,
        "case77_accepted_physical_film_height_m",
        None,
    )
    if radius_m is None or height_m is None:
        return
    directory = Path(out_dir) / PHYSICAL_FILM_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    label = case28.mesh_label(int(step), float(time_s))
    np.savez_compressed(
        directory / f"{OUTPUT_PREFIX}_physical_film_{label}.npz",
        time_s=np.asarray(float(time_s)),
        step=np.asarray(int(step)),
        radius_m=np.asarray(radius_m, dtype=float),
        height_m=np.asarray(height_m, dtype=float),
        junction_flux_ul_s=np.asarray(
            float(
                getattr(
                    state,
                    "case77_accepted_physical_film_flux_ul_s",
                    0.0,
                )
            )
        ),
        junction_impedance_pa_s_m2=np.asarray(
            float(
                getattr(
                    state,
                    "case77_accepted_junction_impedance_pa_s_m2",
                    0.0,
                )
            )
        ),
        junction_impedance_reference_height_m=np.asarray(
            float(
                getattr(
                    state,
                    "case77_accepted_junction_impedance_reference_height_m",
                    0.0,
                )
            )
        ),
        junction_impedance_reference_width_m=np.asarray(
            float(
                getattr(
                    state,
                    "case77_accepted_junction_impedance_reference_width_m",
                    0.0,
                )
            )
        ),
        bridge_volume_ul=np.asarray(float(bridge_volume_ul)),
        experimental_data_used=np.asarray(False),
    )


def _activate_case62_operators() -> None:
    _configure_case62_namespace()
    case28.adaptive_radial_reprojection = (
        adaptive_partitioned_film_reprojection
    )
    case28.try_accept_velocity_step = (
        _try_accept_partitioned_velocity_step
    )
    case28.save_snapshot = _save_partitioned_snapshot
    case28.WALL_LUBRICATION_DRAG_FACTOR = 1.0
    # Case77 resolves the no-slip/shear-free Poiseuille profile with four P1
    # elements through the gap.  Reinstalling the inherited depth-averaged
    # matrix here would double-count exactly that viscous resistance.
    case28.LUBRICATION_RESISTANCE_MODEL = "none"
    case28.LUBRICATION_RESISTANCE_COEFFICIENT = 0.0
    case28.mass_limited_contact_line_speed_cap = (
        _partitioned_mass_limited_contact_line_speed_cap
        if USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE
        else parent._mass_limited_contact_line_speed_cap_with_sphere_sweep
    )


def _snapshot_times(dt_s: float, maximum_steps: int) -> tuple[float, ...]:
    final_time_s = float(maximum_steps) * float(dt_s)
    interval_s = max(float(SNAPSHOT_INTERVAL_S), float(dt_s))
    count = int(math.floor(final_time_s / interval_s))
    seconds = {
        0.0,
        final_time_s,
        *(
            float(index) * interval_s
            for index in range(1, count + 1)
        ),
    }
    for required_time_s in (10.0, 100.0, 3500.0):
        if (
            required_time_s <= final_time_s + 1.0e-12
            and abs(
                round(required_time_s / float(dt_s)) * float(dt_s)
                - required_time_s
            )
            <= 1.0e-9
        ):
            seconds.add(required_time_s)
    return tuple(sorted(float(value) for value in seconds))


def _config(maximum_steps: int, record_every: int) -> base.RealMeshEvolutionConfig:
    return replace(
        CONFIG,
        max_steps=int(maximum_steps),
        record_every_steps=int(record_every),
        snapshot_times_s=_snapshot_times(
            float(CONFIG.dt_s),
            int(maximum_steps),
        ),
    )


def _load_restart_partitioned_state(out_dir: Path, time_s: float) -> None:
    global _RESTART_FILM_RADIUS_M
    global _RESTART_FILM_HEIGHT_M
    global _RESTART_BRIDGE_VOLUME_UL
    global _RESTART_JUNCTION_RADIUS_M
    global _RESTART_HYDRAULIC_DEFICIT_UL
    global _RESTART_HYDRAULIC_RATE_UL_S
    global _RESTART_ACCEPTED_MASS_SUPPLY_FLUX_UL_S
    global _RESTART_LOCAL_DEFICIT_INVENTORY_UL
    global _RESTART_FAR_DEFICIT_INVENTORY_UL
    global _RESTART_FAR_DEFICIT_RATE_UL_S
    global _RESTART_SUPPLY_SATURATION_TIME_S
    global _RESTART_BODY_FITTED_TRANSITION_TIME_S
    global _RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL
    global _RESTART_TERMINAL_EVENT_CHECKPOINT
    global _PARTITIONED_CAPTURE_RESERVOIR_UL
    global _PARTITIONED_FINITE_SUPPLY_CAP_UL
    global _PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S
    global _RESTART_PRESSURE_DRIVEN_FLUX_UL_S
    global _RESTART_PHYSICAL_FILM_RADIUS_M
    global _RESTART_PHYSICAL_FILM_HEIGHT_M
    global _RESTART_PHYSICAL_FILM_FLUX_UL_S
    global _RESTART_JUNCTION_IMPEDANCE_PA_S_M2
    global _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M
    global _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M
    global _PARTITIONED_LOCAL_SATURATION_TIME_S
    global _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL
    global _PARTITIONED_ACCEPTED_LOCAL_DEFICIT_INVENTORY_UL
    global _PARTITIONED_ACCEPTED_FAR_DEFICIT_INVENTORY_UL
    global _PARTITIONED_ACCEPTED_FAR_DEFICIT_RATE_UL_S
    global _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL
    global _PARTITIONED_JUNCTION_THROAT_WIDTH_M
    global _PARTITIONED_JUNCTION_THROAT_HEIGHT_M
    global _PARTITIONED_JUNCTION_THROAT_MOBILITY

    candidates = sorted(
        (out_dir / "mesh_states").glob(
            f"{OUTPUT_PREFIX}_real_mesh_step*_t*.npz"
        )
    )
    selected: Path | None = None
    for path in candidates:
        with np.load(path, allow_pickle=True) as data:
            if abs(float(data["time_s"]) - float(time_s)) <= 1.0e-12:
                selected = path
                break
    if selected is None:
        raise FileNotFoundError(f"No Case62 checkpoint at {time_s:g} s")
    with np.load(selected, allow_pickle=True) as data:
        surface = np.asarray(data["vertices_m"], dtype=float)
        rings = np.asarray(data["ring_index"], dtype=int)
        region = np.asarray(data["ring_region"], dtype=int)
    row_r, row_h = parent._ring_coordinates(surface, rings)
    film_rows = np.flatnonzero(region == 1)
    _RESTART_FILM_RADIUS_M = np.asarray(row_r[film_rows], dtype=float)
    _RESTART_FILM_HEIGHT_M = np.asarray(row_h[film_rows], dtype=float)
    _RESTART_BRIDGE_VOLUME_UL = float(
        parent.bridge_inventory_volume_ul(
            surface,
            rings,
            region,
            CONFIG,
        )
    )
    _RESTART_JUNCTION_RADIUS_M = float(_RESTART_FILM_RADIUS_M[0])
    if bool(USE_MOVING_BOUNDARY_PRESSURE_FILM):
        physical_candidates = sorted(
            (out_dir / PHYSICAL_FILM_DIR_NAME).glob(
                f"{OUTPUT_PREFIX}_physical_film_step*_t*.npz"
            )
        )
        physical_selected: Path | None = None
        for path in physical_candidates:
            with np.load(path, allow_pickle=True) as data:
                if (
                    abs(float(data["time_s"]) - float(time_s))
                    <= 1.0e-12
                ):
                    physical_selected = path
                    break
        if physical_selected is None:
            # A legacy similarity-front checkpoint already contains the
            # complete accepted film surface.  Initialize the pressure-film
            # grid by conservative interpolation of that computed state; no
            # experimental profile or target is introduced at the handoff.
            footprint_m = float(_RESTART_FILM_RADIUS_M[0])
            _RESTART_PHYSICAL_FILM_RADIUS_M = _moving_pressure_film_grid(
                footprint_m,
                int(_RESTART_FILM_RADIUS_M.size),
                CONFIG,
            )
            _RESTART_PHYSICAL_FILM_HEIGHT_M = np.interp(
                _RESTART_PHYSICAL_FILM_RADIUS_M,
                _RESTART_FILM_RADIUS_M,
                _RESTART_FILM_HEIGHT_M,
            )
            _RESTART_PHYSICAL_FILM_HEIGHT_M[-1] = FILM_MINIMUM_HEIGHT_M
            _RESTART_PHYSICAL_FILM_FLUX_UL_S = 0.0
        else:
            with np.load(physical_selected, allow_pickle=True) as data:
                _RESTART_PHYSICAL_FILM_RADIUS_M = np.asarray(
                    data["radius_m"],
                    dtype=float,
                )
                _RESTART_PHYSICAL_FILM_HEIGHT_M = np.asarray(
                    data["height_m"],
                    dtype=float,
                )
                _RESTART_PHYSICAL_FILM_FLUX_UL_S = max(
                    float(data["junction_flux_ul_s"]),
                    0.0,
                )
                _RESTART_JUNCTION_IMPEDANCE_PA_S_M2 = max(
                    float(
                        data["junction_impedance_pa_s_m2"]
                        if "junction_impedance_pa_s_m2" in data.files
                        else 0.0
                    ),
                    0.0,
                )
                _RESTART_JUNCTION_IMPEDANCE_REFERENCE_HEIGHT_M = max(
                    float(
                        data["junction_impedance_reference_height_m"]
                        if (
                            "junction_impedance_reference_height_m"
                            in data.files
                        )
                        else 0.0
                    ),
                    0.0,
                )
                _RESTART_JUNCTION_IMPEDANCE_REFERENCE_WIDTH_M = max(
                    float(
                        data["junction_impedance_reference_width_m"]
                        if "junction_impedance_reference_width_m" in data.files
                        else 0.0
                    ),
                    0.0,
                )
    if bool(USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE):
        # A restart must restore the physical near-contact film reservoir,
        # not relabel the complete bridge inventory as freely available
        # local liquid.  The latter made every continuation time a new
        # nucleation event and progressively hid drained volume from the
        # visible outer film.  Re-evaluate the same annular-capacity closure
        # used by the uninterrupted solve from the saved physical state.
        (
            _speed_cap,
            _flux_ul_s,
            _activation,
            local_capture_ul,
            _inner_pressure_pa,
            _next_pressure_pa,
        ) = parent._ORIGINAL_MASS_LIMITED_CONTACT_LINE_SPEED_CAP(
            surface,
            rings,
            region,
            CONFIG,
            float(time_s),
        )
        h0_m = float(CONFIG.initial_film_thickness_um) * 1.0e-6
        # Restore the actual finite-contact nucleus declared by Case77.
        # The former sqrt(2*R*h0-h0^2) value is the radius of a sphere cut
        # one complete film thickness above its bottom, not the 0.06-um
        # first-contact geometry used by this simulation.
        first_contact_radius_m = max(
            float(CONFIG.initial_bridge_radius_mm) * 1.0e-3,
            0.0,
        )
        capillary_length_m = math.sqrt(
            float(CONFIG.surface_tension_n_m)
            / max(
                float(CONFIG.density_kg_m3)
                * float(CONFIG.gravity_m_s2),
                1.0e-30,
            )
        )
        _PARTITIONED_CAPTURE_RESERVOIR_UL = (
            2.0
            * math.pi
            * first_contact_radius_m
            * h0_m
            * (
                capillary_length_m
                + math.sqrt(h0_m * capillary_length_m)
            )
            * 1.0e9
        )
    history_path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    restart_history = parent._read_numeric_history(history_path)
    if restart_history:
        restart_row = min(
            restart_history,
            key=lambda row: abs(float(row.get("t_s", 0.0)) - float(time_s)),
        )
        if (
            abs(float(restart_row.get("t_s", 0.0)) - float(time_s))
            > 1.0e-9
        ):
            raise RuntimeError(
                f"No restart history row at {float(time_s):g} s"
            )
        restart_prefix = [
            row
            for row in restart_history
            if float(row.get("t_s", 0.0)) <= float(time_s) + 1.0e-9
        ]
        # The current Case77 initial reservoir has already been reconstructed
        # above from ell_c + sqrt(h0*ell_c).  Do not infer it from
        # ``local_capture_volume_ul``: after the finite threshold exists that
        # history field is an effective remaining allowance, not the original
        # geometric reservoir.  Treating it as the latter created a restart-
        # only liquid burst.
        saved_local_capture_ul = float(
            restart_row.get("local_capture_volume_ul", 0.0)
        )
        saved_finite_cap_ul = float(
            restart_row.get("case62_finite_supply_cap_ul", 0.0)
        )
        if (
            bool(USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE)
            and saved_local_capture_ul > 0.0
            and saved_finite_cap_ul > 0.0
            and not bool(USE_MOVING_BOUNDARY_PRESSURE_FILM)
        ):
            bridge_rows = np.flatnonzero(region == 0)
            restart_contact_radius_m = (
                float(row_r[int(bridge_rows[0])])
                if bridge_rows.size
                else float(CONFIG.initial_bridge_radius_mm) * 1.0e-3
            )
            h0_m = float(CONFIG.initial_film_thickness_um) * 1.0e-6
            first_contact_radius_m = max(
                float(CONFIG.initial_bridge_radius_mm) * 1.0e-3,
                0.0,
            )
            saved_swept_film_ul = (
                math.pi
                * max(
                    restart_contact_radius_m**2
                    - first_contact_radius_m**2,
                    0.0,
                )
                * h0_m
                * 1.0e9
            )
            _PARTITIONED_CAPTURE_RESERVOIR_UL = max(
                saved_local_capture_ul - saved_swept_film_ul,
                0.0,
            )
        _RESTART_PRESSURE_DRIVEN_FLUX_UL_S = max(
            float(
                restart_row.get(
                    "case62_pressure_driven_junction_flux_ul_s",
                    0.0,
                )
            ),
            0.0,
        )
        _PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S = float(
            _RESTART_PRESSURE_DRIVEN_FLUX_UL_S
        )
        if bool(USE_MOVING_BOUNDARY_PRESSURE_FILM):
            _RESTART_PRESSURE_DRIVEN_FLUX_UL_S = float(
                _RESTART_PHYSICAL_FILM_FLUX_UL_S
            )
            _PARTITIONED_PRESSURE_DRIVEN_FLUX_UL_S = float(
                _RESTART_PHYSICAL_FILM_FLUX_UL_S
            )
        saved_cap_ul = float(
            restart_row.get("case62_finite_supply_cap_ul", 0.0)
        )
        _PARTITIONED_FINITE_SUPPLY_CAP_UL = (
            saved_cap_ul if saved_cap_ul > 0.0 else None
        ) if bool(USE_TERMINAL_LOCAL_DONOR_CAP) else None
        saved_deficit_ul = float(
            restart_row.get(
                "case62_cumulative_hydraulic_deficit_ul",
                0.0,
            )
        )
        if saved_deficit_ul > 0.0:
            _RESTART_HYDRAULIC_DEFICIT_UL = saved_deficit_ul
        else:
            history_time_s = np.asarray(
                [row.get("t_s", 0.0) for row in restart_prefix],
                dtype=float,
            )
            history_rate_ul_s = np.maximum(
                np.asarray(
                    [
                        row.get(
                            "case62_hydraulic_junction_rate_ul_s",
                            0.0,
                        )
                        for row in restart_prefix
                    ],
                    dtype=float,
                ),
                0.0,
            )
            _RESTART_HYDRAULIC_DEFICIT_UL = float(
                np.trapezoid(history_rate_ul_s, history_time_s)
                if history_time_s.size > 1
                else 0.0
            )
        _RESTART_HYDRAULIC_RATE_UL_S = max(
            float(
                restart_row.get(
                    "case62_hydraulic_junction_rate_ul_s",
                    0.0,
                )
            ),
            0.0,
        )
        _RESTART_ACCEPTED_MASS_SUPPLY_FLUX_UL_S = max(
            float(restart_row.get("mass_supply_flux_ul_s", 0.0)),
            0.0,
        )
        _PARTITIONED_ACCEPTED_HYDRAULIC_DEFICIT_UL = float(
            _RESTART_HYDRAULIC_DEFICIT_UL
        )
        _RESTART_FAR_DEFICIT_INVENTORY_UL = max(
            float(
                restart_row.get(
                    "case77_far_deficit_inventory_ul",
                    0.0,
                )
            ),
            0.0,
        )
        _RESTART_LOCAL_DEFICIT_INVENTORY_UL = max(
            float(
                restart_row.get(
                    "case77_local_deficit_inventory_ul",
                    _RESTART_HYDRAULIC_DEFICIT_UL
                    - _RESTART_FAR_DEFICIT_INVENTORY_UL,
                )
            ),
            0.0,
        )
        _RESTART_FAR_DEFICIT_RATE_UL_S = max(
            float(
                restart_row.get(
                    "case77_far_deficit_rate_ul_s",
                    0.0,
                )
            ),
            0.0,
        )
        inventory_restart_residual_ul = (
            _RESTART_LOCAL_DEFICIT_INVENTORY_UL
            + _RESTART_FAR_DEFICIT_INVENTORY_UL
            - _RESTART_HYDRAULIC_DEFICIT_UL
        )
        if abs(inventory_restart_residual_ul) > 1.0e-9:
            raise RuntimeError(
                "Case77 restart local/far inventories do not conserve the "
                "accepted hydraulic deficit"
            )
        _PARTITIONED_ACCEPTED_LOCAL_DEFICIT_INVENTORY_UL = float(
            _RESTART_LOCAL_DEFICIT_INVENTORY_UL
        )
        _PARTITIONED_ACCEPTED_FAR_DEFICIT_INVENTORY_UL = float(
            _RESTART_FAR_DEFICIT_INVENTORY_UL
        )
        _PARTITIONED_ACCEPTED_FAR_DEFICIT_RATE_UL_S = float(
            _RESTART_FAR_DEFICIT_RATE_UL_S
        )
        _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL = None
        fully_connected_rows = [
            row
            for row in restart_prefix
            if float(
                row.get("case77_far_supply_activation", 0.0)
            )
            >= 1.0 - 1.0e-12
        ]
        if fully_connected_rows:
            first_fully_connected_time_s = float(
                fully_connected_rows[0].get("t_s", 0.0)
            )
            preconnection_rows = [
                row
                for row in restart_prefix
                if float(row.get("t_s", 0.0))
                < first_fully_connected_time_s
            ]
            baseline_row = (
                preconnection_rows[-1]
                if preconnection_rows
                else fully_connected_rows[0]
            )
            _PARTITIONED_FULL_CONNECTION_DEFICIT_BASE_UL = max(
                float(
                    baseline_row.get(
                        "case62_cumulative_hydraulic_deficit_ul",
                        0.0,
                    )
                ),
                0.0,
            )
        _PARTITIONED_JUNCTION_THROAT_WIDTH_M = max(
            float(
                restart_row.get(
                    "case77_junction_throat_width_mm",
                    0.0,
                )
            )
            * 1.0e-3,
            0.0,
        )
        _PARTITIONED_JUNCTION_THROAT_HEIGHT_M = (
            float(
                restart_row.get(
                    "case77_junction_throat_height_um",
                    CONFIG.initial_film_thickness_um,
                )
            )
            * 1.0e-6
        )
        _PARTITIONED_JUNCTION_THROAT_MOBILITY = float(
            np.clip(
                restart_row.get(
                    "case77_junction_throat_mobility",
                    1.0,
                ),
                0.0,
                1.0,
            )
        )
        saved_saturation_time_s = float(
            restart_row.get(
                "case62_supply_saturation_time_s",
                0.0,
            )
        )
        if (
            saved_saturation_time_s <= 0.0
            and _PARTITIONED_FINITE_SUPPLY_CAP_UL is not None
        ):
            for row in restart_prefix:
                if float(row.get("bridge_volume_ul", 0.0)) >= (
                    float(_PARTITIONED_FINITE_SUPPLY_CAP_UL) - 1.0e-9
                ):
                    saved_saturation_time_s = float(row.get("t_s", 0.0))
                    break
        _RESTART_SUPPLY_SATURATION_TIME_S = (
            saved_saturation_time_s
            if bool(USE_TERMINAL_LOCAL_DONOR_CAP)
            else 0.0
        )
        _PARTITIONED_LOCAL_SATURATION_TIME_S = float(
            _RESTART_SUPPLY_SATURATION_TIME_S
        )
        # The body-fitted deficit split is a path-dependent accepted state.
        # Restoring only the total/local/far inventories makes the first
        # continuation step falsely declare a new transition and assigns the
        # complete accumulated deficit to the early kernel a second time.
        # Preserve the original transition event exactly across restarts.
        _RESTART_BODY_FITTED_TRANSITION_TIME_S = max(
            float(
                restart_row.get(
                    "case77_body_fitted_transition_time_s",
                    0.0,
                )
            ),
            0.0,
        )
        _RESTART_BODY_FITTED_TRANSITION_DEFICIT_UL = max(
            float(
                restart_row.get(
                    "case77_body_fitted_transition_deficit_ul",
                    0.0,
                )
            ),
            0.0,
        )
        _RESTART_TERMINAL_EVENT_CHECKPOINT = bool(
            float(
                restart_row.get(
                    "case62_terminal_event_checkpoint",
                    0.0,
                )
            )
            > 0.5
        )


def _write_case62_summary(
    out_dir: Path,
    summary: dict,
    wall_seconds: float,
) -> Path:
    history_path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    final = rows[-1]
    final_time_s = float(final["t_s"])
    final_snapshots = sorted(
        (out_dir / "mesh_states").glob(
            f"{OUTPUT_PREFIX}_real_mesh_step*_t*.npz"
        )
    )
    final_film_hmin_um = float(
        final.get("case62_film_hmin_um", final["h_min_um"])
    )
    final_film_rmin_mm = float(
        final.get("case62_film_rmin_mm", final["r_at_h_min_mm"])
    )
    for snapshot in final_snapshots:
        with np.load(snapshot, allow_pickle=True) as data:
            if abs(float(data["time_s"]) - final_time_s) > 1.0e-12:
                continue
            points = np.asarray(data["vertices_m"], dtype=float)
            rings = np.asarray(data["ring_index"], dtype=int)
            region = np.asarray(data["ring_region"], dtype=int)
        row_r, row_h = parent._ring_coordinates(points, rings)
        film_rows = np.flatnonzero(region == 1)
        minimum_local = int(np.argmin(row_h[film_rows]))
        minimum_row = int(film_rows[minimum_local])
        final_film_hmin_um = float(row_h[minimum_row] * 1.0e6)
        final_film_rmin_mm = float(row_r[minimum_row] * 1.0e3)
        break
    summary["case"] = CASE_LABEL
    summary["truth_status"] = (
        "partitioned_moving_boundary_pr35_bridge_axisymmetric_film_prediction"
    )
    inherited = summary.get("case61_inherited_case29_mechanisms", {})
    inherited.update(
        {
            "state_based_conservative_neck_concentration": False,
            "capillary_scale_recovery_branch": bool(
                RESOLVE_TWO_SIDED_JUNCTION_LAYER
            ),
            "normal_surface_smoothing": False,
        }
    )
    summary["case61_inherited_case29_mechanisms"] = inherited
    summary["case62_partitioned_film_model"] = {
        "experimental_data_used_in_forward_simulation": False,
        "film_equation": (
            "backward-Euler axisymmetric lubrication with lagged h^3 mobility, "
            "linear capillary curvature and gravity"
        ),
        "junction_rate_source": (
            "computed bridge-inventory change minus Reynolds sweep using "
            "the solved local junction height"
        ),
        "junction_sink_support": (
            "the residual hydraulic rate is imposed as the conservative "
            "inner through-flow of the outer-film subdomain"
            if RESOLVE_TWO_SIDED_JUNCTION_LAYER
            else "the residual hydraulic rate is imposed as the inner film "
            "boundary through-flow"
        ),
        "bridge_film_matching": (
            "flux-derived pressure drop and full-curvature circular "
            "Young-Laplace micro-layer"
            if USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
            else "minimum-curvature Hermite capillary sublayer from h0 to "
            "the solved junction, with width set by the local "
            "viscocapillary length"
            if RESOLVE_TWO_SIDED_JUNCTION_LAYER
            else "outer film begins after one local viscocapillary transition "
            "length (gamma*h_J^3/(3*mu*abs(U_CL)))^(1/3)"
        ),
        "two_sided_junction_layer": bool(
            RESOLVE_TWO_SIDED_JUNCTION_LAYER
        ),
        "fitted_trough_width": False,
        "fitted_trough_height": False,
        "compact_cubic_recovery": bool(
            RESOLVE_TWO_SIDED_JUNCTION_LAYER
            and not USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
        ),
        "compact_recovery_interpretation": (
            "flux-derived full-curvature Young-Laplace micro-layer"
            if USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
            else "minimum-curvature Hermite capillary matching closure"
            if RESOLVE_TWO_SIDED_JUNCTION_LAYER
            else "disabled"
        ),
        "flux_derived_capillary_micro_layer": bool(
            USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER
        ),
        "subgrid_junction_flux_closure": bool(
            USE_SUBGRID_JUNCTION_FLUX_CLOSURE
        ),
        "case61_neck_supply_state_retained": bool(
            RETAIN_CASE61_NECK_SUPPLY_STATE
        ),
        "conservative_junction_supply_closure": bool(
            USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE
        ),
        "spherical_sweep_is_bridge_demand_not_supply_inventory": bool(
            USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE
        ),
        "normal_trough_smoothing": False,
        "cumulative_hydraulic_deficit": (
            "accepted trapezoidal time integral of max(Q_J, 0)"
        ),
        "flux_derived_inventory_model": {
            "far_inventory": (
                "accepted trapezoidal time integral of max(Q_far, 0)"
            ),
            "local_inventory": "V_hydraulic - V_far",
            "conservation_identity": (
                "V_local + V_far - V_hydraulic = 0"
            ),
            "experimental_data_used": False,
        },
        "finite_supply_stationary_event": (
            "when the state-derived donor is exhausted and Q_J=U_CL=0, "
            "the accepted pinned state is advanced directly without "
            "repeating identical PR35/Young-Laplace/ALE solves"
            if (
                USE_TERMINAL_LOCAL_DONOR_CAP
                and not USE_PRESSURE_DRIVEN_JUNCTION_FLUX
                and not USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
            )
            else "one-way handoff from the finite first-contact inventory "
            "to the moving pressure-film PDE; Q_J remains a solved flux"
            if (
                USE_TERMINAL_LOCAL_DONOR_CAP
                and USE_MOVING_BOUNDARY_PRESSURE_FILM
            )
            else "disabled: local donor depletion continuously activates "
            "far-film replenishment through the capillary-diffusion "
            "similarity length"
            if USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
            else "disabled: the complete resolved radial film remains "
            "connected to the junction through its computed lubrication "
            "pressure flux"
        ),
        "terminal_local_donor_cap": bool(USE_TERMINAL_LOCAL_DONOR_CAP),
        "unified_local_to_far_supply": bool(
            USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
        ),
        "full_connection_junction_mobility": bool(
            USE_FULL_CONNECTION_JUNCTION_MOBILITY
        ),
        "two_sided_junction_series_transmission": (
            "R_J contains two Cox wedges in series: "
            "2*log(L_cox/lambda)*R_continuity, followed by the computed "
            "Poiseuille ell_J/h_J^3 evolution"
            if USE_MOVING_BOUNDARY_PRESSURE_FILM
            else "unresolved two-arm length resistance applied to Q_J"
            if (
                RESOLVE_TWO_SIDED_JUNCTION_LAYER
                and APPLY_UNRESOLVED_TWO_ARM_SUPPLY_RESISTANCE
            )
            else "disabled: resolved through-gap K_mu contains both arms"
            if RESOLVE_TWO_SIDED_JUNCTION_LAYER
            else "disabled for the one-sided junction model"
        ),
        "moving_boundary_pressure_film": bool(
            USE_MOVING_BOUNDARY_PRESSURE_FILM
        ),
        "moving_film_bridge_pressure": (
            "zero-angle fixed-sphere Young-Laplace pressure selected by "
            "the conservative PR33 bridge volume"
            if USE_MOVING_BOUNDARY_PRESSURE_FILM
            else "disabled"
        ),
        "moving_film_junction_impedance": (
            "handoff impedance identified from the last accepted forward "
            "flux and current bridge pressure; later updates use the "
            "computed ell_J/h_J^3 Poiseuille scaling"
            if USE_MOVING_BOUNDARY_PRESSURE_FILM
            else "disabled"
        ),
        "first_contact_capture_width": (
            "sqrt(gamma/(rho*g)) + "
            "sqrt(h0*sqrt(gamma/(rho*g)))"
        ),
        "experimental_target_used_in_impedance": False,
        "junction_throat_width_model": (
            "sqrt(h0*capillary_length)"
            if USE_FULL_CONNECTION_JUNCTION_MOBILITY
            else "disabled"
        ),
        "junction_throat_mobility_model": (
            "((h_throat-h_residual)/(h0-h_residual))^3"
            if USE_FULL_CONNECTION_JUNCTION_MOBILITY
            else "disabled"
        ),
        "junction_throat_residual_height_um": (
            float(CONFIG.attached_outer_deficit_soft_lower_um)
            if USE_FULL_CONNECTION_JUNCTION_MOBILITY
            else 0.0
        ),
        "far_supply_resolution_activation": float(
            FAR_SUPPLY_RESOLUTION_ACTIVATION
        ),
        "pr35_junction_pressure_continuity": bool(
            USE_PR35_JUNCTION_PRESSURE_CONTINUITY
        ),
        "heron_junction_pressure_continuity": bool(
            USE_HERON_JUNCTION_PRESSURE_CONTINUITY
        ),
        "young_laplace_pressure_driven_junction_flux": bool(
            USE_PRESSURE_DRIVEN_JUNCTION_FLUX
        ),
        "junction_pressure_gauge": (
            "absolute liquid-to-air pressure from the exact PR33 Heron "
            "traction on the bridge free-surface patch"
            if USE_HERON_JUNCTION_PRESSURE_CONTINUITY
            else
            "rho*g*h0 + median(PR35 bridge trace - PR35 far-film trace)"
            if USE_PR35_JUNCTION_PRESSURE_CONTINUITY
            else "zero-angle Young-Laplace equilibrium pressure selected by "
            "the conservative bridge inventory"
            if USE_PRESSURE_DRIVEN_JUNCTION_FLUX
            else "disabled"
        ),
        "junction_flux_closure": (
            "moving-grid nonlinear backward-Euler pressure-film solve with "
            "Q_J as an output and a state-derived two-sided Cox/Poiseuille "
            "junction impedance"
            if USE_MOVING_BOUNDARY_PRESSURE_FILM
            else
            "nonlinear backward-Euler pressure-boundary film solve with "
            "Q_J as an output and R_turn=12*mu/h_aperture^2 in series"
            if USE_PRESSURE_DRIVEN_JUNCTION_FLUX
            else "capillary-diffusion similarity flux on the unresolved "
            "far-film scale, with conservative donor removal"
            if USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
            else "inherited resolved-film pressure flux"
        ),
        "film_picard_iterations": FILM_PICARD_ITERATIONS,
        "maximum_viscocapillary_width_capillary_lengths": (
            MAX_VISCOCAPILLARY_WIDTH_CAPILLARY_LENGTHS
        ),
        "minimum_height_m": FILM_MINIMUM_HEIGHT_M,
        "minimum_height_is_rejection_threshold_not_clip": True,
        "wall_seconds": float(wall_seconds),
    }
    summary["final_observables"] = {
        "time_s": final_time_s,
        "bridge_volume_ul": float(final["bridge_volume_ul"]),
        "minimum_film_height_um": final_film_hmin_um,
        "minimum_height_radius_mm": final_film_rmin_mm,
        "mesh_negative_tets": float(final["mesh_negative_tets"]),
        "film_conservation_residual_ul": float(
            final.get("case62_film_conservation_residual_ul", "nan")
        ),
        "total_volume_residual_ul": float(
            final.get("case62_total_volume_residual_ul", "nan")
        ),
    }
    path = out_dir / f"{OUTPUT_PREFIX}_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_postrun_comparison_assets(out_dir: Path) -> Path:
    """Freeze simulation hashes, then introduce EXP files for plotting only."""

    output = out_dir.expanduser().resolve()
    history = output / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    summary = output / f"{OUTPUT_PREFIX}_summary.json"

    def state_at_time(directory: Path, pattern: str, target_s: float) -> Path:
        candidates = list(directory.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No Case77 state matching {pattern} in {directory}"
            )
        timed: list[tuple[float, Path]] = []
        for path in candidates:
            with np.load(path, allow_pickle=True) as data:
                timed.append((float(data["time_s"]), path))
        time_s, selected = min(
            timed,
            key=lambda item: abs(item[0] - float(target_s)),
        )
        if abs(time_s - float(target_s)) > 1.0e-9:
            raise RuntimeError(
                f"Case77 has no state at t={target_s:g} s; nearest is "
                f"t={time_s:g} s"
            )
        return selected

    final_surface = state_at_time(
        output / "mesh_states",
        f"{OUTPUT_PREFIX}_real_mesh_step*_t*.npz",
        10.0,
    )
    final_tetra = state_at_time(
        output / TETRA_DIR_NAME,
        f"{OUTPUT_PREFIX}_tetra_mesh_step*_t*.npz",
        10.0,
    )
    artifacts = {
        "history_csv": history,
        "summary_json": summary,
        "final_surface_npz": final_surface,
        "final_tetra_npz": final_tetra,
        "case_entry_source": CASE_ENTRY_SOURCE,
        "case62_source": Path(__file__).resolve(),
        "case28_source": Path(case28.__file__).resolve(),
        "coupled_film_operator": (
            ROOT.parent.parent
            / "ddgclib"
            / "operators"
            / "coupled_capillary_bridge_film.py"
        ).resolve(),
    }
    frozen = {name: _sha256(path) for name, path in artifacts.items()}
    source_dir = (
        ROOT / "Case_29_siekman2025_adaptive_tetra_free_surface_solver"
    )
    comparison_dir = output / "comparison_only"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_assets = {}
    for name in (
        "case7_digitized_siekman2025_fig5a_h0_100.csv",
        "siekman2025_fig1c_pdf_digitized_manual_approx.csv",
        "siekman2025_fig1c_user_exact.png",
    ):
        source = source_dir / name
        destination = comparison_dir / name
        shutil.copy2(source, destination)
        comparison_assets[name] = {
            "source": str(source.resolve()),
            "sha256": _sha256(destination),
        }
    current = {name: _sha256(path) for name, path in artifacts.items()}
    if current != frozen:
        raise RuntimeError(
            "A simulation artifact changed while seeding post-run comparison data"
        )
    provenance = {
        "case": CASE_LABEL,
        "experimental_data_used_in_forward_simulation": False,
        "simulation_completed_before_comparison_assets": True,
        "simulation_artifact_sha256": frozen,
        "postrun_comparison_assets": comparison_assets,
    }
    path = output / f"{OUTPUT_PREFIX}_simulation_isolation_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_npz_at_time(
    source: Path,
    destination: Path,
    *,
    time_s: float,
    step: int,
) -> None:
    """Copy a stationary accepted state under a later event time."""

    with np.load(source, allow_pickle=True) as data:
        payload = {
            key: np.asarray(data[key])
            for key in data.files
        }
    payload["time_s"] = np.asarray(float(time_s))
    payload["step"] = np.asarray(int(step))
    np.savez_compressed(destination, **payload)


def _checkpoint_path_at_time(
    directory: Path,
    *,
    filename_prefix: str,
    time_s: float,
) -> Path:
    """Find a restart checkpoint by physical time, independent of step size."""

    for candidate in sorted(directory.glob(f"{filename_prefix}*.npz")):
        with np.load(candidate, allow_pickle=True) as data:
            if abs(float(data["time_s"]) - float(time_s)) <= 1.0e-12:
                return candidate
    raise FileNotFoundError(
        f"No {filename_prefix} checkpoint at {float(time_s):g} s "
        f"in {directory}"
    )


def _fast_forward_pinned_prediction(
    output: Path,
    source: Path,
    *,
    restart_time_s: float,
    final_time_s: float,
    condition: str = (
        "finite supply exhausted, Q_J=0, accepted CL speed=0"
    ),
) -> dict:
    """Advance the exact stationary branch after finite-supply exhaustion.

    The mass active set has ``Q_J=0`` and the accepted bridge velocity is
    zero.  Repeating PR35, the local Young--Laplace BVP, and the zero-flux ALE
    map thousands of times would reproduce the same pinned state.  Record the
    event solution directly and leave capillary optical equilibration to the
    conservative subgrid closure.
    """

    history_path = output / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        raise RuntimeError("Pinned fast-forward requires accepted history")
    added_fields = (
        "case62_cumulative_hydraulic_deficit_ul",
        "case77_local_deficit_inventory_ul",
        "case77_far_deficit_inventory_ul",
        "case77_far_deficit_rate_ul_s",
        "case77_inventory_conservation_residual_ul",
        "case62_supply_saturation_time_s",
        "case62_supply_saturated",
        "case62_skipped_redundant_pinned_bvp",
        "case62_pinned_event_fast_forward",
    )
    for name in added_fields:
        if name not in fieldnames:
            fieldnames.append(name)
    final_step = int(round(float(final_time_s) / float(CONFIG.dt_s)))
    final_row = dict(rows[-1])
    final_row["t_s"] = f"{float(final_time_s):.12g}"
    final_row["step"] = str(final_step)
    for name in (
        "bridge_rate_ul_s",
        "contact_line_speed_m_s",
        "mass_supply_speed_cap_m_s",
        "mass_supply_flux_speed_cap_m_s",
        "mass_supply_flux_ul_s",
        "mass_supply_budget_speed_cap_m_s",
        "mass_supply_budget_delivered_ul",
        "case62_bridge_rate_ul_s",
        "case62_swept_rate_ul_s",
        "case62_hydraulic_junction_rate_ul_s",
        "case62_contact_speed_m_s",
        "case62_viscocapillary_width_mm",
    ):
        if name in fieldnames:
            final_row[name] = "0"
    accepted_hydraulic_inventory_ul = float(
        final_row.get(
            "case62_cumulative_hydraulic_deficit_ul",
            _RESTART_HYDRAULIC_DEFICIT_UL,
        )
    )
    accepted_far_inventory_ul = float(
        final_row.get(
            "case77_far_deficit_inventory_ul",
            _RESTART_FAR_DEFICIT_INVENTORY_UL,
        )
    )
    accepted_local_inventory_ul = float(
        final_row.get(
            "case77_local_deficit_inventory_ul",
            accepted_hydraulic_inventory_ul - accepted_far_inventory_ul,
        )
    )
    inventory_residual_ul = (
        accepted_local_inventory_ul
        + accepted_far_inventory_ul
        - accepted_hydraulic_inventory_ul
    )
    if abs(inventory_residual_ul) > 1.0e-9:
        raise RuntimeError(
            "Pinned Case77 event cannot preserve a nonconservative local/far "
            "inventory state"
        )
    accepted_saturation_time_s = float(
        final_row.get(
            "case62_supply_saturation_time_s",
            _RESTART_SUPPLY_SATURATION_TIME_S,
        )
    )
    final_row["case62_cumulative_hydraulic_deficit_ul"] = (
        f"{accepted_hydraulic_inventory_ul:.17g}"
    )
    final_row["case77_local_deficit_inventory_ul"] = (
        f"{accepted_local_inventory_ul:.17g}"
    )
    final_row["case77_far_deficit_inventory_ul"] = (
        f"{accepted_far_inventory_ul:.17g}"
    )
    final_row["case77_far_deficit_rate_ul_s"] = "0"
    final_row["case77_inventory_conservation_residual_ul"] = (
        f"{inventory_residual_ul:.17g}"
    )
    final_row["case62_supply_saturation_time_s"] = (
        f"{accepted_saturation_time_s:.17g}"
    )
    final_row["case62_supply_saturated"] = "1"
    final_row["case62_skipped_redundant_pinned_bvp"] = "1"
    final_row["case62_pinned_event_fast_forward"] = "1"
    rows.append(final_row)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    final_label = parent.mesh_label(final_step, float(final_time_s))
    surface_source = _checkpoint_path_at_time(
        output / "mesh_states",
        filename_prefix=f"{OUTPUT_PREFIX}_real_mesh_",
        time_s=restart_time_s,
    )
    tetra_source = _checkpoint_path_at_time(
        output / TETRA_DIR_NAME,
        filename_prefix=f"{OUTPUT_PREFIX}_tetra_mesh_",
        time_s=restart_time_s,
    )
    _copy_npz_at_time(
        surface_source,
        output
        / "mesh_states"
        / f"{OUTPUT_PREFIX}_real_mesh_{final_label}.npz",
        time_s=final_time_s,
        step=final_step,
    )
    _copy_npz_at_time(
        tetra_source,
        output
        / TETRA_DIR_NAME
        / f"{OUTPUT_PREFIX}_tetra_mesh_{final_label}.npz",
        time_s=final_time_s,
        step=final_step,
    )

    summary_source = source / f"{OUTPUT_PREFIX}_summary.json"
    if not summary_source.is_file():
        summary_source = source / "summary.json"
    summary = (
        json.loads(summary_source.read_text(encoding="utf-8"))
        if summary_source.is_file()
        else {}
    )
    summary["pinned_event_fast_forward"] = {
        "active": True,
        "from_time_s": float(restart_time_s),
        "to_time_s": float(final_time_s),
        "condition": str(condition),
        "experimental_data_used": False,
    }
    return summary


def _unified_replenishment_event_time_s(
    saturation_time_s: float,
    dt_s: float,
) -> float:
    """Return the first grid time after the far-film front reaches the junction.

    The local donor remains exactly pinned while

        ell_d = (gamma*h0^3*elapsed/(3*mu))^(1/4) <= ell_c.

    Jumping across that analytically zero-flux interval is event integration,
    not a second physical branch.  One extra accepted timestep places the
    continuation strictly inside the smooth replenishment transition.
    """

    h0_m = float(CONFIG.initial_film_thickness_um) * 1.0e-6
    capillary_length_m = math.sqrt(
        float(CONFIG.surface_tension_n_m)
        / max(
            float(CONFIG.density_kg_m3)
            * float(CONFIG.gravity_m_s2),
            1.0e-30,
        )
    )
    capillary_diffusivity_m4_s = (
        float(CONFIG.surface_tension_n_m) * h0_m**3
        / max(3.0 * float(CONFIG.viscosity_pa_s), 1.0e-30)
    )
    activation_floor = float(
        np.clip(FAR_SUPPLY_RESOLUTION_ACTIVATION, 0.0, 0.95)
    )
    lower = 0.0
    upper = 1.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        activation = midpoint**2 * (3.0 - 2.0 * midpoint)
        if activation < activation_floor:
            lower = midpoint
        else:
            upper = midpoint
    resolved_front_ratio = 1.0 + upper
    refill_time_s = (
        resolved_front_ratio**4 * capillary_length_m**4
        / max(capillary_diffusivity_m4_s, 1.0e-30)
    )
    first_grid_index = int(
        math.ceil(
            (float(saturation_time_s) + refill_time_s)
            / float(dt_s)
            - 1.0e-12
        )
    )
    return float(first_grid_index + 1) * float(dt_s)


def run_partitioned_prediction(
    out_dir: Path,
    *,
    final_time_s: float,
    restart_from: Path | None = None,
    restart_time_s: float | None = None,
    solve_every_steps: int = 2,
) -> Path:
    """Compute an isolated Case62-family prediction to ``final_time_s``.

    A restart copies only the accepted simulation history and mesh states.
    Post-run experimental comparison assets are intentionally excluded.
    """

    final_time = float(final_time_s)
    dt_s = float(CONFIG.dt_s)
    maximum_steps = int(round(final_time / dt_s))
    if (
        not math.isfinite(final_time)
        or final_time <= 0.0
        or abs(maximum_steps * dt_s - final_time) > 1.0e-9
    ):
        raise ValueError(
            "final_time_s must be positive and an integer multiple of dt_s"
        )
    solve_stride = max(int(solve_every_steps), 1)
    output = out_dir.expanduser().resolve()
    history_path = output / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    if history_path.exists():
        raise FileExistsError(
            f"Case62 will not overwrite an existing prediction: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prior_wall_seconds = 0.0

    if restart_from is None:
        restart_time = dt_s
        startup = _config(maximum_steps=1, record_every=1)
        summary = case28.run_case(startup, output)
    else:
        source = restart_from.expanduser().resolve()
        restart_time = (
            float(restart_time_s)
            if restart_time_s is not None
            else final_time
        )
        source_history = (
            source / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
        )
        existing = parent._read_numeric_history(source_history)
        if not existing:
            raise RuntimeError(
                f"Restart source has no simulation history: {source_history}"
            )
        source_final_time = float(existing[-1]["t_s"])
        if restart_time > source_final_time + 1.0e-9:
            raise ValueError(
                "restart_time_s cannot exceed the final recorded time in the "
                f"restart source ({source_final_time:g} s)"
            )
        if restart_time >= final_time - 1.0e-12:
            raise ValueError("restart_time_s must be smaller than final_time_s")
        shutil.copy2(source_history, history_path)
        for directory_name in (
            "mesh_states",
            TETRA_DIR_NAME,
            PHYSICAL_FILM_DIR_NAME,
            "computational_capillary_states",
            "young_laplace_mesh_states",
            "young_laplace_tetra_states",
        ):
            source_directory = source / directory_name
            if (
                directory_name in ("mesh_states", TETRA_DIR_NAME)
                and not source_directory.is_dir()
            ):
                raise FileNotFoundError(source_directory)
            if not source_directory.is_dir():
                continue
            shutil.copytree(
                source_directory,
                output / directory_name,
            )
        # A branch restart must contain only states at or before its restart
        # time.  Retaining later source checkpoints made post-run geometry
        # reconstruction mix the new history with stale states from the old
        # branch.  Trim every copied state family, not only the optional
        # pressure-film checkpoints.
        for directory_name in (
            "mesh_states",
            TETRA_DIR_NAME,
            PHYSICAL_FILM_DIR_NAME,
            "computational_capillary_states",
            "young_laplace_mesh_states",
            "young_laplace_tetra_states",
        ):
            copied_directory = output / directory_name
            if not copied_directory.is_dir():
                continue
            for path in copied_directory.glob("*.npz"):
                with np.load(path, allow_pickle=True) as data:
                    if "time_s" not in data.files:
                        continue
                    snapshot_time_s = float(data["time_s"])
                if snapshot_time_s > restart_time + 1.0e-12:
                    path.unlink()
        prior_wall_seconds = float(
            existing[-1].get("wall_elapsed_s", 0.0)
        )

    _load_restart_partitioned_state(output, restart_time)
    pinned_stationary_restart = bool(
        restart_from is not None
        and USE_TERMINAL_LOCAL_DONOR_CAP
        and not USE_PRESSURE_DRIVEN_JUNCTION_FLUX
        and not USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
        and _RESTART_SUPPLY_SATURATION_TIME_S > 0.0
        and (
            _RESTART_TERMINAL_EVENT_CHECKPOINT
            or _RESTART_HYDRAULIC_RATE_UL_S <= 1.0e-14
        )
    )
    unified_replenishment_time_s = (
        _unified_replenishment_event_time_s(
            _RESTART_SUPPLY_SATURATION_TIME_S,
            dt_s,
        )
        if _RESTART_SUPPLY_SATURATION_TIME_S > 0.0
        else float("inf")
    )
    unified_stationary_restart = bool(
        restart_from is not None
        and USE_TERMINAL_LOCAL_DONOR_CAP
        and USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
        and _RESTART_SUPPLY_SATURATION_TIME_S > 0.0
        and restart_time < unified_replenishment_time_s - 1.0e-12
        and (
            _RESTART_TERMINAL_EVENT_CHECKPOINT
            or _RESTART_HYDRAULIC_RATE_UL_S <= 1.0e-14
        )
    )
    if pinned_stationary_restart:
        summary = _fast_forward_pinned_prediction(
            output,
            restart_from.expanduser().resolve(),
            restart_time_s=restart_time,
            final_time_s=final_time,
        )
    elif unified_stationary_restart:
        replenishment_time_s = min(
            unified_replenishment_time_s,
            final_time,
        )
        summary = _fast_forward_pinned_prediction(
            output,
            restart_from.expanduser().resolve(),
            restart_time_s=restart_time,
            final_time_s=replenishment_time_s,
            condition=(
                "local donor depleted and Q_J is below the ALE resolution "
                "active set during capillary-diffusion replenishment"
            ),
        )
        if replenishment_time_s < final_time - 1.0e-12:
            _load_restart_partitioned_state(
                output,
                replenishment_time_s,
            )
            continuation = _config(
                maximum_steps=maximum_steps,
                record_every=max(int(round(1.0 / dt_s)), 1),
            )
            summary = parent.run_case_from_checkpoint(
                continuation,
                output,
                replenishment_time_s,
                solve_every_steps=solve_stride,
            )
    else:
        continuation = _config(
            maximum_steps=maximum_steps,
            record_every=max(int(round(1.0 / dt_s)), 1),
        )
        summary = parent.run_case_from_checkpoint(
            continuation,
            output,
            restart_time,
            solve_every_steps=solve_stride,
        )
        continued_history = parent._read_numeric_history(history_path)
        continued_final_time_s = float(continued_history[-1]["t_s"])
        continued_saturation_time_s = float(
            continued_history[-1].get(
                "case62_supply_saturation_time_s",
                0.0,
            )
        )
        continued_hydraulic_rate_ul_s = abs(
            float(
                continued_history[-1].get(
                    "case62_hydraulic_junction_rate_ul_s",
                    0.0,
                )
            )
        )
        continued_terminal_event = bool(
            float(
                continued_history[-1].get(
                    "case62_terminal_event_checkpoint",
                    0.0,
                )
            )
            > 0.5
        )
        if (
            USE_TERMINAL_LOCAL_DONOR_CAP
            and USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
            and continued_final_time_s < final_time - 1.0e-12
            and continued_saturation_time_s > 0.0
            and (
                continued_terminal_event
                or continued_hydraulic_rate_ul_s <= 1.0e-14
            )
        ):
            replenishment_time_s = min(
                _unified_replenishment_event_time_s(
                    continued_saturation_time_s,
                    dt_s,
                ),
                final_time,
            )
            # Once the replenishment event is already in the past, a later
            # mesh rejection must not be mistaken for a request to
            # fast-forward backward in physical time.
            if (
                replenishment_time_s
                > continued_final_time_s + 1.0e-12
            ):
                summary = _fast_forward_pinned_prediction(
                    output,
                    output,
                    restart_time_s=continued_final_time_s,
                    final_time_s=replenishment_time_s,
                    condition=(
                        "local donor depleted and Q_J=0 until the capillary-"
                        "diffusion front spans one capillary length"
                    ),
                )
                if replenishment_time_s < final_time - 1.0e-12:
                    _load_restart_partitioned_state(
                        output,
                        replenishment_time_s,
                    )
                    summary = parent.run_case_from_checkpoint(
                        continuation,
                        output,
                        replenishment_time_s,
                        solve_every_steps=solve_stride,
                    )
        if (
            USE_TERMINAL_LOCAL_DONOR_CAP
            and not USE_PRESSURE_DRIVEN_JUNCTION_FLUX
            and not USE_UNIFIED_LOCAL_TO_FAR_SUPPLY
            and continued_final_time_s < final_time - 1.0e-12
            and continued_saturation_time_s > 0.0
            and (
                continued_terminal_event
                or continued_hydraulic_rate_ul_s <= 1.0e-14
            )
        ):
            summary = _fast_forward_pinned_prediction(
                output,
                output,
                restart_time_s=continued_final_time_s,
                final_time_s=final_time,
            )
    wall_seconds = prior_wall_seconds + time.monotonic() - started
    return _write_case62_summary(
        output,
        summary,
        wall_seconds,
    )


def run_partitioned_prediction_0to10(out_dir: Path) -> Path:
    """Compute the isolated 0--10 s Case62 prediction."""

    return run_partitioned_prediction(
        out_dir,
        final_time_s=10.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PREDICTION_OUTPUT)
    parser.add_argument("--prediction-0to10", action="store_true")
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument("--final-time-s", type=float, default=10.0)
    parser.add_argument("--restart-from", type=Path, default=None)
    parser.add_argument("--restart-time-s", type=float, default=None)
    parser.add_argument("--solve-every-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--record-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    _activate_case62_operators()
    if bool(arguments.prediction_0to10) or bool(arguments.prediction):
        summary = run_partitioned_prediction(
            Path(arguments.out_dir),
            final_time_s=(
                10.0
                if bool(arguments.prediction_0to10)
                else float(arguments.final_time_s)
            ),
            restart_from=arguments.restart_from,
            restart_time_s=arguments.restart_time_s,
            solve_every_steps=int(arguments.solve_every_steps),
        )
        print(summary.resolve(), flush=True)
        return
    output = Path(arguments.out_dir).expanduser().resolve()
    if (output / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = time.monotonic()
    summary = case28.run_case(
        _config(
            maximum_steps=int(arguments.max_steps),
            record_every=int(arguments.record_every),
        ),
        output,
    )
    path = _write_case62_summary(
        output,
        summary,
        time.monotonic() - started,
    )
    print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
