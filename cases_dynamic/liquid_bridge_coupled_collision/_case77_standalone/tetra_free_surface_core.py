#!/usr/bin/env python3
"""Case 28: standalone tetra free-surface solver prototype.

This case is intentionally different from Case 27.  It does not patch the
Case 25 timestep loop and it does not call the Case 25 bridge-profile
constraints after t=0.  The initial connected sphere/film surface is reused as
the starting geometry only; subsequent motion is computed by:

    free-surface force + Cox/contact-line force
    -> 3D tetra Cauchy/Stokes solve
    -> PR33 tetra pressure projection
    -> move tetra boundary nodes

The result is a first fully force-driven tetra prototype.  It still has mesh
quality limiters and sphere/substrate/outer-wall boundary projections, but no
Siekman digitized curve or Case 25 geometric profile is used to move the mesh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .operators.contact_line import (
    cox_contact_line_force_ring,
    cox_inverse_contact_line_speed,
    ring_segment_lengths,
)
from .operators.bridge_film import coupled_cox_young_laplace_lubrication_neck_profile
from .operators.mesh_dynamics import (
    cotangent_surface_tension_stiffness,
    cotangent_surface_tension_forces,
    implicit_tetra_taylor_hood_velocity_pressure,
    implicit_tetra_stokes_p0_velocity_pressure,
    implicit_tetra_stokes_velocity_pressure,
    lump_tet_masses,
    mesh_quality,
    p2_consistent_surface_force,
    sparse_pressure_projection,
    tet_cell_volumes,
    tetra_depth_averaged_lubrication_stiffness,
    tet_volume_matrix_sparse,
)

from . import young_laplace_meniscus_core as case25


base = case25.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 28"
OUTPUT_PREFIX = "case28"
OUT_DIR = ROOT / CASE_STEM
TETRA_DIR_NAME = "mesh_states_tetra_volume"

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case25.CONFIG,
    dt_s=0.01,
    max_steps=1000,
    wall_clock_limit_s=10800.0,
    record_every_steps=100,
    snapshot_times_s=tuple(float(t) for t in range(0, 11)),
    azimuthal_nodes=64,
    profile_nodes=96,
    surface_tension_n_m=0.031,
    max_vertical_speed_um_s=0.0,
    max_radial_speed_um_s=0.0,
    dynamic_contact_angle_max_deg=22.0,
    attached_visible_feed_partition_enabled=True,
    attached_visible_feed_min_fraction=0.12,
    attached_visible_feed_max_fraction=0.22,
    attached_visible_feed_transition_progress=0.55,
    attached_visible_feed_transition_width=0.22,
    attached_visible_feed_scale_radius_mode="contact_blend",
    attached_visible_feed_contact_blend_exponent=2.0,
)
CFL_SAFETY_FACTOR = 0.05
WALL_LUBRICATION_DRAG_FACTOR = 4.0
LUBRICATION_RESISTANCE_MODEL = "none"
LUBRICATION_RESISTANCE_COEFFICIENT = 3.0
TETRA_MIN_LAYER_HEIGHT_M = 0.5e-6
TETRA_ACCEPT_MIN_VOLUME_M3 = 1.0e-30
# Number of linear finite elements through the substrate-film thickness.
# Case77 sets this to four: the resolved P1 Poiseuille mobility then differs
# from the exact no-slip/shear-free mobility by only 1/(4*N**2)=1.5625%.
THROUGH_GAP_VERTICAL_ELEMENTS = 1
VISIBLE_FILM_DEPLETION_CENTER_CAPILLARY_LENGTHS = 0.18
VISIBLE_FILM_INNER_WIDTH_BASE_CAPILLARY_LENGTHS = 0.08
VISIBLE_FILM_INNER_WIDTH_FRACTION_CAPILLARY_LENGTHS = 0.20
VISIBLE_FILM_OUTER_WIDTH_BASE_CAPILLARY_LENGTHS = 0.04
VISIBLE_FILM_OUTER_WIDTH_FRACTION_CAPILLARY_LENGTHS = 0.10
VISIBLE_FILM_NECK_CORE_WIDTH_CAPILLARY_LENGTHS = 0.045
VISIBLE_FILM_NECK_CORE_WEIGHT = 0.30
VISIBLE_FILM_SATURATED_INNER_WIDTH_CAPILLARY_LENGTHS = 0.04
VISIBLE_FILM_SATURATED_OUTER_WIDTH_CAPILLARY_LENGTHS = 0.40
CONTACT_LINE_SUPPLY_LIMIT_THRESHOLD = 0.82
CONTACT_LINE_SUPPLY_LIMIT_WIDTH = 0.20
CONTACT_LINE_SUPPLY_LIMIT_FLOOR = 0.08
CONTACT_LINE_SUPPLY_LIMIT_ENABLED = False
MASS_LIMITED_CONTACT_LINE_ENABLED = True
# Optional second complementarity constraint on contact speed.  Case77 keeps
# the conservative PR33 receiver/donor transfer but disables this duplicate
# reaction because Q_J is already accounted for in the coupled volume RHS.
MASS_SUPPLY_CONTACT_REACTION_ENABLED = True
FILM_PRESSURE_FLUX_CONTACT_LINE_CAP_ENABLED = True
LOCAL_CAPTURE_WIDTH_CAPILLARY_LENGTHS = 1.25
LOCAL_NECK_DYNAMIC_WEDGE_MIN_FRACTION = 0.04
LOCAL_NECK_DYNAMIC_WEDGE_MAX_FRACTION = 0.55
LOCAL_NECK_DYNAMIC_WEDGE_SPEED_EXPONENT = 0.80
VISIBLE_FILM_SATURATED_FEED_THRESHOLD = 0.94
VISIBLE_FILM_SATURATED_FEED_WIDTH = 0.015
VISIBLE_FILM_SATURATED_FEED_MAX_FRACTION = 1.0
PHYSICAL_LUBRICATION_OUTER_FILM_ENABLED = True
LEGACY_VISIBLE_TARGET_FORCING_ENABLED = False
# Case77 may select the backward-Euler or zero-inertia form of the same PR35
# momentum block. These are numerical accuracy tolerances, not physical
# speed/force caps and not experimental calibration parameters.
ADAPTIVE_INERTIA_SWITCH_ENABLED = False
DYNAMIC_TO_STOKES_INERTIA_RATIO = 1.0e-2
STOKES_TO_DYNAMIC_INERTIA_RATIO = 5.0e-2
DYNAMIC_TO_STOKES_CONSECUTIVE_SOLVES = 5
# Case77 solves the full 3-D PR33/PR35 operators in the exact axisymmetric
# Galerkin subspace.  The raw PR33/PR35/PR37 force arrays remain 3-D; only the
# mixed velocity/pressure system is reduced before solution.
ENFORCE_EXACT_AXISYMMETRIC_PR35 = True
AXISYMMETRY_TOLERANCE_M = 1.0e-11
# CFL is an admissibility test in Case77, never a componentwise velocity cap.
GLOBAL_VELOCITY_CLIPPING_ENABLED = False
# PR37 enters only through the assembled Cox force.  Do not replace the
# solved contact-line velocity afterward with an inverse-Cox or supply cap.
POSTSOLVE_CONTACT_SPEED_OVERWRITE_ENABLED = False
# Case77's declared physical force inventory contains Heron and PR37 Cox, not
# a second Young wall line force.
INCLUDE_YOUNG_WALL_FORCE = False
# Regularize the singular first-contact momentum state from the computed
# geometric angle and the PR37 Cox relation.  This is used only at t=0 and is
# never imposed as an accepted displacement or later velocity.
INITIALIZE_CONTACT_MOMENTUM_FROM_COX_GEOMETRY = True
# First-contact momentum history associated with the repository's declared
# 0.06-micrometre finite-contact regularization.  It is an initial condition
# for M u^0/dt, never a prescribed accepted contact displacement.
INITIAL_CONTACT_MOMENTUM_M_S = 0.0
# Solve the nonlinear PR37 load simultaneously with the linear PR35 mobility.
# Three right-hand sides share one reduced sparse factorization.
IMPLICIT_COX_FORCE_SOLVE_ENABLED = True
# Backward-Euler linearization of the unchanged exact 3-D Heron force.  This
# adds dt*K_gamma to the left-hand side and is not a second physical force.
IMPLICIT_CAPILLARY_STIFFNESS_ENABLED = True
# Surface tension resists normal shape changes; tangential displacement is a
# surface reparameterization and belongs to ALE/Cox rather than K_gamma.
IMPLICIT_CAPILLARY_NORMAL_PROJECTION_ENABLED = True
# Linearize interior capillary waves, but keep the PR37 contact boundary force
# explicit because its Young/Cox derivative is not part of K_gamma.
IMPLICIT_CAPILLARY_STIFFNESS_AT_CONTACT_LINE = False
# Historical one-layer cases used an optional sphere-side wedge matrix.
# Case77 resolves the declared substrate gap with K_mu and disables this
# separate matrix in its case activation.
SPHERE_WEDGE_LUBRICATION_ENABLED = True
# When enabled, integrate only the Cox wedge below the first resolved
# contact-adjacent material edge. K_mu resolves all larger scales.
SPHERE_WEDGE_UNRESOLVED_ONLY = False
OUTER_FILM_LUBRICATION_SUBSTEPS = 3
OUTER_FILM_LUBRICATION_MAX_FRACTIONAL_STEP = 0.25
OUTER_FILM_SINGLE_TROUGH_RECOVERY_CAPILLARY_LENGTHS = 0.65
COUPLED_LOCAL_NECK_ENABLED = True
COUPLED_LOCAL_NECK_MAX_WIDTH_CAPILLARY_LENGTHS = 0.12
COUPLED_LOCAL_NECK_FAST_MAX_WIDTH_CAPILLARY_LENGTHS = 0.28
COUPLED_LOCAL_NECK_MAX_WIDTH_MIN_FILM_THICKNESSES = 1.5
COUPLED_LOCAL_NECK_FAST_MAX_WIDTH_MIN_FILM_THICKNESSES = 8.0
COUPLED_LOCAL_NECK_WIDTH_SPEED_RATIO = 0.40
COUPLED_LOCAL_NECK_MIN_WIDTH_UM = 15.0
COUPLED_LOCAL_NECK_VISIBLE_SHARE_MULTIPLIER = 1.00
COUPLED_LOCAL_NECK_FAST_VISIBLE_SHARE_MULTIPLIER = 0.25
COUPLED_LOCAL_NECK_PRECURSOR_FRACTION = 0.08
COUPLED_LOCAL_NECK_REPULSION_PRESSURE_SCALE = 0.05
COUPLED_LOCAL_NECK_FAST_MAX_ANGLE_DEG = 22.0
COUPLED_LOCAL_NECK_SLOW_MAX_ANGLE_DEG = 65.0
COUPLED_LOCAL_NECK_ANGLE_SPEED_RATIO = 0.40
PRESSURE_FRONT_BASE_GAP_CAPILLARY_LENGTHS = 1.01
PRESSURE_FRONT_LATE_GAP_EXTRA_CAPILLARY_LENGTHS = 0.00
PRESSURE_FRONT_LATE_GAP_SPEED_RATIO = 0.20
PRESSURE_FRONT_RIM_HEIGHT_RECOVERY_SPEED_RATIO = 0.40
PRESSURE_FRONT_DYNAMIC_RIM_SUCTION_FRACTION = 0.38
PRESSURE_FRONT_DYNAMIC_RIM_SUCTION_SPEED_EXPONENT = 1.5
FIRST_FILM_RING_SLOW_RIM_GAP_FRACTION = 0.030
FIRST_FILM_RING_FAST_RIM_GAP_FRACTION = 0.080


def flat_initial_profile_m(config: base.RealMeshEvolutionConfig, r_m: np.ndarray) -> np.ndarray:
    """Return the finite-substrate capillary--gravity initial film.

    The historical function name is retained because the independent solver
    imports it in several operators.  The physical profile is the regular
    axisymmetric solution of the linearized Young--Laplace--gravity equation
    with ``h(0)=h0`` and ``h(R)=0``; no digitized curve enters it.
    """

    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    r = np.asarray(r_m, dtype=float)
    capillary_length_m = math.sqrt(
        float(config.surface_tension_n_m)
        / max(
            float(config.density_kg_m3) * float(config.gravity_m_s2),
            1.0e-30,
        )
    )
    outer_i0 = float(np.i0(radius_m / capillary_length_m))
    clipped_r = np.clip(r, 0.0, radius_m)
    profile = h0_m * (
        outer_i0 - np.i0(clipped_r / capillary_length_m)
    ) / max(outer_i0 - 1.0, 1.0e-30)
    return np.where(
        r <= radius_m + 1.0e-15,
        np.maximum(profile, 0.0),
        0.0,
    )


def flat_initial_film_volume_ul(config: base.RealMeshEvolutionConfig) -> float:
    """Integrate the same finite-substrate initial profile."""

    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    radius = np.linspace(0.0, radius_m, 4097)
    height = flat_initial_profile_m(config, radius)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return float(integrate(2.0 * math.pi * radius * height, radius) * 1.0e9)


def flat_attached_missing_outer_film_volume_ul(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float:
    """Outer-film loss relative to the finite-substrate initial film."""

    r, h = base.attached_film_profile_m(points, rings, ring_region)
    if r.size < 2:
        return 0.0
    initial_h = flat_initial_profile_m(config, r)
    integrand = 2.0 * math.pi * r * np.maximum(initial_h - h, 0.0)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return float(integrate(integrand, r) * 1.0e9)


def bridge_inventory_volume_ul(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> float:
    """Total wetted sphere-cap inventory represented by the moving contact line.

    The fluorescence-style Fig. 1(c) profile only resolves the visible
    outer-film depression.  The bridge-volume curve, however, includes the
    liquid stored in the saturated neck and wetted sphere cap.  Keeping this as
    a separate state variable avoids forcing all bridge growth into the plotted
    film-height branch.
    """

    geom = base.attached_ring_geometry(surface, rings, ring_region)
    contact_r = float(geom["contact_radius_m"])
    reference_r = float(config.initial_bridge_radius_mm) * 1.0e-3
    return max(
        base.attached_sphere_cap_volume_ul(config, contact_r)
        - base.attached_sphere_cap_volume_ul(config, reference_r),
        0.0,
    )


def attached_cap_volume_derivative_m2(
    config: base.RealMeshEvolutionConfig,
    contact_radius_m: float,
) -> float:
    """Numerical dV_cap/dr for the wetted spherical-cap inventory."""

    contact_r = max(float(contact_radius_m), 1.0e-9)
    eps = max(0.25e-6, 1.0e-4 * contact_r)
    sphere_r = float(config.sphere_radius_mm) * 1.0e-3
    rp = min(contact_r + eps, sphere_r * 0.999)
    rm = max(contact_r - eps, float(config.initial_bridge_radius_mm) * 1.0e-3)
    if rp <= rm:
        rm = max(float(config.initial_bridge_radius_mm) * 1.0e-3, contact_r * 0.999)
        rp = contact_r * 1.001
    vp = base.attached_sphere_cap_volume_ul(config, rp) * 1.0e-9
    vm = base.attached_sphere_cap_volume_ul(config, rm) * 1.0e-9
    return float(max((vp - vm) / max(rp - rm, 1.0e-30), 1.0e-30))


def contact_line_supply_factor(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> tuple[float, float, float]:
    """Diagnostic supply factor for forward contact-line motion.

    Cox/Voinov gives the local dynamic contact-line speed, but the bridge can
    only keep growing if liquid is supplied by the film.  The ratio reported
    here is the wetted cap inventory divided by a local capillary-annulus
    estimate, ``2*pi*r_CL*h0*l_c``.  In earlier Case 28 versions this ratio
    directly throttled the contact-line speed.  That froze rCL around 40-80 s,
    which is not physical for the Siekman long-time bridge-growth experiment:
    the surrounding film keeps redistributing and feeding the bridge beyond
    one local capillary annulus.  Keep the ratio as a diagnostic by default;
    only enable the limiter explicitly for stress tests.
    """

    geom = base.attached_ring_geometry(surface, rings, ring_region)
    contact_r = max(float(geom["contact_radius_m"]), 1.0e-12)
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    capillary_length = math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-30)
    )
    capacity_ul = 2.0 * math.pi * contact_r * h0 * capillary_length * 1.0e9
    inventory_ul = bridge_inventory_volume_ul(surface, rings, ring_region, config)
    ratio = inventory_ul / max(capacity_ul, 1.0e-30)
    if not bool(CONTACT_LINE_SUPPLY_LIMIT_ENABLED):
        return 1.0, float(ratio), float(capacity_ul)
    excess = max(ratio - CONTACT_LINE_SUPPLY_LIMIT_THRESHOLD, 0.0)
    scaled = excess / max(CONTACT_LINE_SUPPLY_LIMIT_WIDTH, 1.0e-12)
    factor = 1.0 / (1.0 + scaled**4)
    factor = max(float(CONTACT_LINE_SUPPLY_LIMIT_FLOOR), float(factor))
    return float(np.clip(factor, 0.0, 1.0)), float(ratio), float(capacity_ul)


def computed_outer_film_pressure_inward_flux_ul_s(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> tuple[float, float, float]:
    """Inward film supply from the current resolved lubrication pressure.

    The previous Case 28 cap used a single pressure scale, rho*g*h0/ell_c,
    which did not know whether the computed film had actually developed a
    pressure gradient capable of feeding the bridge.  This helper computes the
    ring-averaged thin-film pressure from the current mesh and returns the
    inward flux through the connected outer-film supply path:

        q = -h^3/(3 mu) dp/dr,
        Q_in = max(dp, 0) / integral[dr/(2*pi*r*h^3/(3*mu))].

    That makes the bridge supply state-dependent instead of a validation-curve
    target.  Using the full local path resistance is important: a first-face
    gradient can stay large even after the nearby film has been drained, which
    overfeeds the bridge at 20-100 s.
    """

    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size < 5:
        return 0.0, 0.0, 0.0
    row_r = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    row_z = np.asarray([float(np.mean(surface[ring, 2])) for ring in rings], dtype=float)
    rows = film_rows[np.argsort(row_r[film_rows])]
    r = row_r[rows].copy()
    h = row_z[rows].copy()
    if r.size < 5 or np.any(np.diff(r) <= 0.0):
        return 0.0, 0.0, 0.0

    h_floor = max(TETRA_MIN_LAYER_HEIGHT_M, 0.05e-6)
    h = np.maximum(h, h_floor)
    h_initial = flat_initial_profile_m(config, r)
    mu = max(float(config.viscosity_pa_s) * float(WALL_LUBRICATION_DRAG_FACTOR), 1.0e-30)
    gamma = max(float(config.surface_tension_n_m), 1.0e-30)
    rho_g = float(config.density_kg_m3) * float(config.gravity_m_s2)
    dhdr = np.gradient(h, r, edge_order=2)
    d2hdr2 = np.gradient(dhdr, r, edge_order=2)
    curvature = d2hdr2 + dhdr / np.maximum(r, 1.0e-12)
    pressure = -gamma * curvature + rho_g * (h - h_initial)
    capillary_length = math.sqrt(gamma / max(rho_g, 1.0e-30))
    local_limit = min(r[-1], r[0] + 2.5 * capillary_length)
    candidate = np.flatnonzero(r <= local_limit)
    if candidate.size < 2:
        return 0.0, float(pressure[0]), float(pressure[min(1, pressure.size - 1)])
    source_idx = int(candidate[-1])
    # The liquid bridge is a capillary-suction boundary connected to the first
    # resolved film row through the full local lubrication path.  If the first
    # row is overpressurized by the steep resolved neck, keep a small thin-film
    # fallback proportional to h0/l_c instead of using the full local pressure
    # contrast.  That prevents an artificial rCL freeze without the old
    # pressure-contrast overfeed.
    local_pressure = pressure[candidate]
    primary_dp = max(float(np.max(local_pressure) - pressure[0]), 0.0)
    local_contrast_dp = max(float(np.max(local_pressure) - np.min(local_pressure)), 0.0)
    h0_scale = max(float(config.initial_film_thickness_um) * 1.0e-6, h_floor)
    unresolved_fraction = min(0.10, h0_scale / max(capillary_length, 1.0e-30))
    dp_supply = max(primary_dp, unresolved_fraction * local_contrast_dp)
    if dp_supply <= 0.0:
        return 0.0, float(pressure[0]), float(pressure[min(1, pressure.size - 1)])
    r_face = 0.5 * (r[:source_idx] + r[1 : source_idx + 1])
    h_face = np.maximum(0.5 * (h[:source_idx] + h[1 : source_idx + 1]), h_floor)
    dr_face = np.maximum(r[1 : source_idx + 1] - r[:source_idx], 1.0e-30)
    conductance = 2.0 * math.pi * r_face * h_face**3 / (3.0 * mu)
    resistance = float(np.sum(dr_face / np.maximum(conductance, 1.0e-30)))
    inward_flux_ul_s = float(dp_supply / max(resistance, 1.0e-30) * 1.0e9)
    return inward_flux_ul_s, float(pressure[0]), float(pressure[1])


def mass_limited_contact_line_speed_cap(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    time_s: float | None,
) -> tuple[float, float, float, float, float, float]:
    """Thin-film mass-flux cap for contact-line advance.

    Cox/Voinov gives the microscopic contact-line mobility, but the spherical
    cap cannot grow faster than the surrounding film can supply liquid.  Use a
    the inward flux computed from the current resolved film-pressure gradient.
    This is a mass-conservation cap on dV_cap/dt, not a validation-curve target.
    """

    if not bool(MASS_LIMITED_CONTACT_LINE_ENABLED):
        return float("inf"), 0.0, 0.0, 0.0
    geom = base.attached_ring_geometry(surface, rings, ring_region)
    contact_r = max(float(geom["contact_radius_m"]), 1.0e-9)
    h0 = max(float(config.initial_film_thickness_um) * 1.0e-6, 1.0e-12)
    mu = max(float(config.viscosity_pa_s) * float(WALL_LUBRICATION_DRAG_FACTOR), 1.0e-30)
    rho_g = max(float(config.density_kg_m3) * float(config.gravity_m_s2), 0.0)
    gamma = max(float(config.surface_tension_n_m), 1.0e-30)
    ell_c = math.sqrt(gamma / max(rho_g, 1.0e-30))
    scale_pressure_gradient = rho_g * h0 / max(ell_c, 1.0e-30)
    scale_flux_m3_s = 2.0 * math.pi * contact_r * (h0**3 / (3.0 * mu)) * scale_pressure_gradient
    film_flux_ul_s, film_inner_pressure_pa, film_next_pressure_pa = computed_outer_film_pressure_inward_flux_ul_s(
        surface,
        rings,
        ring_region,
        config,
    )
    if bool(FILM_PRESSURE_FLUX_CONTACT_LINE_CAP_ENABLED):
        flux_m3_s = max(float(film_flux_ul_s), 0.0) * 1.0e-9
    else:
        flux_m3_s = scale_flux_m3_s
    eps = max(0.25e-6, 1.0e-4 * contact_r)
    rp = min(contact_r + eps, float(config.sphere_radius_mm) * 1.0e-3 * 0.999)
    rm = max(contact_r - eps, float(config.initial_bridge_radius_mm) * 1.0e-3)
    if rp <= rm:
        rm = max(float(config.initial_bridge_radius_mm) * 1.0e-3, contact_r * 0.999)
        rp = contact_r * 1.001
    vp = base.attached_sphere_cap_volume_ul(config, rp) * 1.0e-9
    vm = base.attached_sphere_cap_volume_ul(config, rm) * 1.0e-9
    d_cap_volume_dr = max((vp - vm) / max(rp - rm, 1.0e-30), 1.0e-30)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    first_contact_r = math.sqrt(max(2.0 * sphere_radius * h0 - h0 * h0, 1.0e-30))
    capture_width = min(
        max(float(getattr(config, "attached_local_capture_width_capillary_lengths", 0.0)), 0.0),
        float(LOCAL_CAPTURE_WIDTH_CAPILLARY_LENGTHS),
    )
    capture_volume_ul = 2.0 * math.pi * first_contact_r * h0 * capture_width * ell_c * 1.0e9
    inventory_ul = bridge_inventory_volume_ul(surface, rings, ring_region, config)
    if inventory_ul <= capture_volume_ul:
        return (
            float("inf"),
            float(flux_m3_s * 1.0e9),
            0.0,
            float(capture_volume_ul),
            float(film_inner_pressure_pa),
            float(film_next_pressure_pa),
        )
    # After the local near-contact reservoir is used, further contact-line
    # motion is limited directly by the current film-pressure flux.  There is
    # no division by an activation factor here; that was the overfeed bug.
    speed_cap = flux_m3_s / d_cap_volume_dr
    return (
        float(max(speed_cap, 0.0)),
        float(flux_m3_s * 1.0e9),
        1.0,
        float(capture_volume_ul),
        float(film_inner_pressure_pa),
        float(film_next_pressure_pa),
    )


def saturated_visible_feed_target_ul(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    rim_radius_m: float,
    target_missing_ul: float,
    contact_radius_m: float | None,
    time_s: float | None,
) -> tuple[float, float, float, float]:
    """Visible film-drain target with saturation-driven redistribution.

    The base ddgclib partition splits the bridge inventory into a saturated
    neck/cap part and a visible outer-film part from material length scales.
    In the independent tetra solver the Cox line can saturate the local feed
    annulus while the visible depression remains underfed.  Once the inventory
    exceeds the annular capacity, move the partition smoothly toward a
    resolved-film dominated split.  This changes the mesh state, not only the
    validation plot.
    """

    visible_target_ul, visible_fraction, corrected_pool_ul = base.attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim_radius_m,
        target_missing_ul=target_missing_ul,
        contact_radius_m=contact_radius_m,
    )
    _supply_factor, supply_ratio, _capacity_ul = contact_line_supply_factor(
        surface,
        rings,
        ring_region,
        config,
    )
    supply_activation = 1.0 / (
        1.0
        + math.exp(
            -(
                supply_ratio - VISIBLE_FILM_SATURATED_FEED_THRESHOLD
            )
            / max(VISIBLE_FILM_SATURATED_FEED_WIDTH, 1.0e-12)
        )
    )
    activation = supply_activation
    saturated_fraction = min(
        float(VISIBLE_FILM_SATURATED_FEED_MAX_FRACTION),
        max(float(visible_fraction), 0.0),
    )
    boosted_fraction = float(visible_fraction) + (
        float(VISIBLE_FILM_SATURATED_FEED_MAX_FRACTION) - float(visible_fraction)
    ) * activation
    boosted_fraction = float(
        np.clip(
            max(boosted_fraction, saturated_fraction),
            0.0,
            min(float(VISIBLE_FILM_SATURATED_FEED_MAX_FRACTION), 1.0),
        )
    )
    boosted_target_ul = float(corrected_pool_ul) * boosted_fraction
    return boosted_target_ul, boosted_fraction, float(corrected_pool_ul), float(activation)


class TetraFreeSurfaceState:
    """Tetra liquid volume with the free surface as the top boundary."""

    def __init__(self, surface_points: np.ndarray, surface_faces: np.ndarray, rings: np.ndarray):
        self.n_surface = int(surface_points.shape[0])
        self.vertical_elements = max(
            1,
            int(THROUGH_GAP_VERTICAL_ELEMENTS),
        )
        bottom = np.array(surface_points, copy=True)
        bottom[:, 2] = 0.0
        top = np.asarray(surface_points, dtype=float).copy()
        # Preserve historical node numbering for boundary operators:
        # block 0 is the substrate and block 1 is the free surface. Interior
        # through-gap blocks follow them in increasing height order.
        node_blocks = [bottom, top]
        for layer in range(1, self.vertical_elements):
            fraction = float(layer) / float(self.vertical_elements)
            node_blocks.append((1.0 - fraction) * bottom + fraction * top)
        self.nodes = np.vstack(node_blocks)
        self.tets = self._build_layer_tets(np.asarray(rings, dtype=np.int32))
        volumes = tet_cell_volumes(self.nodes, self.tets)
        neg = volumes < 0.0
        if np.any(neg):
            tmp = self.tets[neg, 0].copy()
            self.tets[neg, 0] = self.tets[neg, 1]
            self.tets[neg, 1] = tmp
        self.surface_faces = np.asarray(surface_faces, dtype=np.int32)
        self.rings = np.asarray(rings, dtype=np.int32)
        self.velocities = np.zeros_like(self.nodes)
        self.bottom_reference = np.array(bottom, copy=True)
        self.outer_surface_reference_z = np.maximum(
            np.array(surface_points[self.rings[-1], 2], copy=True),
            TETRA_MIN_LAYER_HEIGHT_M,
        )
        self.supply_budget_ul = 0.0

    @property
    def top(self) -> slice:
        return slice(self.n_surface, 2 * self.n_surface)

    def surface_points(self) -> np.ndarray:
        return self.nodes[self.top]

    def set_surface_points(self, points: np.ndarray) -> None:
        surface = np.asarray(points, dtype=float)
        self.nodes[self.top] = surface
        self.nodes[: self.n_surface] = self.bottom_reference
        for layer in range(1, self.vertical_elements):
            fraction = float(layer) / float(self.vertical_elements)
            block = layer + 1
            block_slice = slice(
                block * self.n_surface,
                (block + 1) * self.n_surface,
            )
            self.nodes[block_slice] = (
                (1.0 - fraction) * self.bottom_reference
                + fraction * surface
            )
        self.velocities[: self.n_surface] = 0.0

    def node_block_for_vertical_layer(self, layer: int) -> int:
        """Map physical bottom-to-top layer numbering to stored node blocks."""

        layer_index = int(layer)
        if not 0 <= layer_index <= self.vertical_elements:
            raise IndexError("vertical layer is outside the mesh")
        if layer_index == 0:
            return 0
        if layer_index == self.vertical_elements:
            return 1
        return layer_index + 1

    def column_vertex_ids(self, surface_ids: np.ndarray) -> np.ndarray:
        """Return all through-gap vertices belonging to surface columns."""

        ids = np.asarray(surface_ids, dtype=int).reshape(-1)
        blocks = [
            self.node_block_for_vertical_layer(layer)
            for layer in range(self.vertical_elements + 1)
        ]
        return np.concatenate(
            [block * self.n_surface + ids for block in blocks]
        )

    def repair_tetra_connectivity(self) -> dict[str, float]:
        """Rebuild the layered tetra split when sliver elements appear."""

        before = mesh_quality(self.nodes, self.tets)
        if (
            float(before["negative_volume_count"]) == 0.0
            and float(before["min_volume_m3"]) > 0.0
        ):
            return {
                "tet_repair_applied": 0.0,
                "tet_repair_min_before_m3": float(before["min_volume_m3"]),
                "tet_repair_min_after_m3": float(before["min_volume_m3"]),
                "tet_repair_negative_after": float(before["negative_volume_count"]),
            }
        self.tets = self._build_layer_tets(self.rings)
        after = mesh_quality(self.nodes, self.tets)
        return {
            "tet_repair_applied": 1.0,
            "tet_repair_min_before_m3": float(before["min_volume_m3"]),
            "tet_repair_min_after_m3": float(after["min_volume_m3"]),
            "tet_repair_negative_after": float(after["negative_volume_count"]),
        }

    def volume_ul(self) -> float:
        return float(np.sum(tet_cell_volumes(self.nodes, self.tets)) * 1.0e9)

    def _build_layer_tets(self, rings: np.ndarray) -> np.ndarray:
        n_theta = int(rings.shape[1])
        tets: list[tuple[int, int, int, int]] = []

        def raw_volume(tet: tuple[int, int, int, int]) -> float:
            pts = self.nodes[np.asarray(tet, dtype=int)]
            return float(
                np.dot(
                    pts[1] - pts[0],
                    np.cross(pts[2] - pts[0], pts[3] - pts[0]),
                )
                / 6.0
            )

        def orient(tet: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            if raw_volume(tet) >= 0.0:
                return tet
            return (tet[1], tet[0], tet[2], tet[3])

        def choose_split(
            split_a: tuple[tuple[int, int, int, int], ...],
            split_b: tuple[tuple[int, int, int, int], ...],
        ) -> tuple[tuple[int, int, int, int], ...]:
            min_a = min(abs(raw_volume(tet)) for tet in split_a)
            min_b = min(abs(raw_volume(tet)) for tet in split_b)
            chosen = split_b if min_b > min_a else split_a
            return tuple(orient(tet) for tet in chosen)

        def node(layer: int, idx: int) -> int:
            block = self.node_block_for_vertical_layer(layer)
            return block * self.n_surface + int(idx)

        for layer in range(self.vertical_elements):
            for i in range(rings.shape[0] - 1):
                for j in range(n_theta):
                    jp = (j + 1) % n_theta
                    b00 = node(layer, rings[i, j])
                    b10 = node(layer, rings[i + 1, j])
                    b11 = node(layer, rings[i + 1, jp])
                    b01 = node(layer, rings[i, jp])
                    t00 = node(layer + 1, rings[i, j])
                    t10 = node(layer + 1, rings[i + 1, j])
                    t11 = node(layer + 1, rings[i + 1, jp])
                    t01 = node(layer + 1, rings[i, jp])
                    split_a = (
                        (b00, b10, b11, t11),
                        (b00, b11, b01, t11),
                        (b00, b01, t01, t11),
                        (b00, t01, t00, t11),
                        (b00, t00, t10, t11),
                        (b00, t10, b10, t11),
                    )
                    split_b = (
                        (b11, b01, b00, t00),
                        (b11, b00, b10, t00),
                        (b11, b10, t10, t00),
                        (b11, t10, t11, t00),
                        (b11, t11, t01, t00),
                        (b11, t01, b01, t00),
                    )
                    tets.extend(choose_split(split_a, split_b))
        return np.asarray(tets, dtype=np.int32)


def seed_validation_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_dir in (
        case25.OUT_DIR,
        ROOT / "Case_27_siekman2025_tetra_volume_ddgclib_mesh_evolution",
        ROOT / "Case_11_siekman2025_real_ddgclib_mesh_evolution",
    ):
        if not source_dir.is_dir():
            continue
        for name in (
            "case7_digitized_siekman2025_fig5a_h0_100.csv",
            "case7_digitized_siekman2025_fig5a_h0_100_debug.png",
            "siekman2025_fig5_source.jpeg",
            "siekman2025_fig1c_crop.png",
            "siekman2025_fig1c_digitization_debug.png",
            "siekman2025_fig1c_user_exact.png",
            "siekman2025_fig1c_pdf_digitized_manual_approx.csv",
        ):
            src = source_dir / name
            dst = OUT_DIR / name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-existing", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--max-steps", type=int, default=CONFIG.max_steps)
    parser.add_argument("--dt", type=float, default=CONFIG.dt_s)
    parser.add_argument("--record-every", type=int, default=CONFIG.record_every_steps)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--profile-nodes", type=int, default=CONFIG.profile_nodes)
    parser.add_argument("--azimuthal-nodes", type=int, default=CONFIG.azimuthal_nodes)
    parser.add_argument("--wall-drag", type=float, default=WALL_LUBRICATION_DRAG_FACTOR)
    parser.add_argument(
        "--lubrication-resistance-model",
        choices=("none", "depth_averaged_free_surface"),
        default=LUBRICATION_RESISTANCE_MODEL,
    )
    parser.add_argument(
        "--lubrication-resistance-coefficient",
        type=float,
        default=LUBRICATION_RESISTANCE_COEFFICIENT,
    )
    parser.add_argument("--cfl-factor", type=float, default=CFL_SAFETY_FACTOR)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> base.RealMeshEvolutionConfig:
    snapshot_steps = range(0, int(args.max_steps) + 1, max(1, int(args.snapshot_every)))
    snapshot_times = tuple(float(step) * float(args.dt) for step in snapshot_steps)
    final_time = float(args.max_steps) * float(args.dt)
    if not snapshot_times or abs(snapshot_times[-1] - final_time) > 1.0e-12:
        snapshot_times = (*snapshot_times, final_time)
    return replace(
        CONFIG,
        dt_s=float(args.dt),
        max_steps=int(args.max_steps),
        record_every_steps=int(args.record_every),
        snapshot_times_s=snapshot_times,
        profile_nodes=int(args.profile_nodes),
        azimuthal_nodes=int(args.azimuthal_nodes),
    )


def mesh_label(step: int, time_s: float) -> str:
    return f"step{int(step):07d}_t{float(time_s):0.4f}s".replace(".", "p")


def _faces_from_rings(rings: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    n_r, n_t = rings.shape
    for i in range(n_r - 1):
        for j in range(n_t):
            jp = (j + 1) % n_t
            a = int(rings[i, j])
            b = int(rings[i + 1, j])
            c = int(rings[i + 1, jp])
            d = int(rings[i, jp])
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(faces, dtype=np.int32)


def initial_surface_mesh(config: base.RealMeshEvolutionConfig, operators: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if True:
        h0 = float(config.initial_film_thickness_um) * 1.0e-6
        substrate_r = float(config.substrate_radius_mm) * 1.0e-3
        contact_r = max(float(config.initial_bridge_radius_mm) * 1.0e-3, 1.0e-6)
        seed_rim = max(0.050e-3, 6.0 * contact_r)
        n_bridge = 8
        bridge_r = np.linspace(contact_r, seed_rim, n_bridge)
        bridge_h = np.full_like(bridge_r, h0)
        bridge_h[0] = float(config.sphere_bottom_z_mm) * 1.0e-3
        n_film = max(int(config.profile_nodes), 64)
        film_r = np.linspace(seed_rim * 1.0001, substrate_r, n_film)
        film_h = flat_initial_profile_m(config, film_r)
        radial = np.concatenate((bridge_r, film_r))
        height = np.concatenate((bridge_h, film_h))
        n_theta = int(config.azimuthal_nodes)
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        points = np.zeros((radial.size * n_theta, 3), dtype=float)
        rings = np.zeros((radial.size, n_theta), dtype=np.int32)
        for i, (r, z) in enumerate(zip(radial, height)):
            ids = np.arange(i * n_theta, (i + 1) * n_theta, dtype=np.int32)
            rings[i] = ids
            points[ids, 0] = r * np.cos(theta)
            points[ids, 1] = r * np.sin(theta)
            points[ids, 2] = z
        faces = _faces_from_rings(rings)
        ring_region = np.ones(radial.size, dtype=np.int32)
        ring_region[:n_bridge] = 0
        return points, faces, rings, ring_region

    film_mesh = base.make_initial_mesh(config, operators)
    film_points = np.asarray(film_mesh["vertices_m"], dtype=float)
    film_rings = np.asarray(film_mesh["ring_index"], dtype=np.int32)
    film_r, film_h = base.film_profile_from_points(film_points, film_rings)
    mesh = base.make_attached_bridge_mesh_from_profile(
        film_r,
        film_h,
        config,
        operators,
        float(config.inner_bridge_volume_ul),
    )
    _vertices, faces, rings, _vertex_ring = base.graph_from_mesh(mesh)
    points = base.points_from_vertices(_vertices)
    ring_region = np.asarray(mesh["ring_region"], dtype=np.int32)
    return points, np.asarray(faces, dtype=np.int32), np.asarray(rings, dtype=np.int32), ring_region


def sphere_tangent_vectors(points: np.ndarray, ids: np.ndarray, config: base.RealMeshEvolutionConfig) -> np.ndarray:
    xyz = np.asarray(points, dtype=float)[np.asarray(ids, dtype=int)]
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    radial = np.zeros((len(ids), 3), dtype=float)
    ok = radius > 1.0e-30
    radial[ok, 0] = xyz[ok, 0] / radius[ok]
    radial[ok, 1] = xyz[ok, 1] / radius[ok]
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    axial = np.sqrt(np.maximum(sphere_radius * sphere_radius - radius * radius, 1.0e-30))
    tangent = radial * (axial / sphere_radius)[:, None]
    tangent[:, 2] = radius / sphere_radius
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1.0e-30)[:, None]
    return tangent


def project_contact_ring_to_sphere(points: np.ndarray, contact_ids: np.ndarray, config: base.RealMeshEvolutionConfig) -> None:
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    bottom = float(config.sphere_bottom_z_mm) * 1.0e-3
    ids = np.asarray(contact_ids, dtype=int)
    r = np.hypot(points[ids, 0], points[ids, 1])
    r = np.minimum(r, 0.98 * sphere_radius)
    theta = np.arctan2(points[ids, 1], points[ids, 0])
    points[ids, 0] = r * np.cos(theta)
    points[ids, 1] = r * np.sin(theta)
    points[ids, 2] = bottom + sphere_radius - np.sqrt(np.maximum(sphere_radius * sphere_radius - r * r, 0.0))


def velocity_limiter(surface_velocity: np.ndarray, surface_points: np.ndarray, config: base.RealMeshEvolutionConfig, dt_s: float) -> tuple[np.ndarray, float]:
    r = np.hypot(surface_points[:, 0], surface_points[:, 1])
    theta = np.arctan2(surface_points[:, 1], surface_points[:, 0])
    radial = surface_velocity[:, 0] * np.cos(theta) + surface_velocity[:, 1] * np.sin(theta)
    tangential = -surface_velocity[:, 0] * np.sin(theta) + surface_velocity[:, 1] * np.cos(theta)
    unique_r = np.unique(np.round(r, decimals=12))
    dr = float(np.nanmedian(np.diff(unique_r))) if unique_r.size > 1 else 2.0e-6
    cfl = max(CFL_SAFETY_FACTOR * dr / max(float(dt_s), 1.0e-30), 2.0e-5)
    configured_radial = float(config.max_radial_speed_um_s) * 1.0e-6
    configured_vertical = float(config.max_vertical_speed_um_s) * 1.0e-6
    radial_limit = cfl if configured_radial <= 0.0 else min(configured_radial, cfl)
    vertical_limit = cfl if configured_vertical <= 0.0 else min(configured_vertical, cfl)
    radial_limited = np.clip(radial, -radial_limit, radial_limit)
    tangential_limited = np.clip(tangential, -radial_limit, radial_limit)
    z_limited = np.clip(surface_velocity[:, 2], -vertical_limit, vertical_limit)
    limited = np.zeros_like(surface_velocity)
    limited[:, 0] = radial_limited * np.cos(theta) - tangential_limited * np.sin(theta)
    limited[:, 1] = radial_limited * np.sin(theta) + tangential_limited * np.cos(theta)
    limited[:, 2] = z_limited
    limited[~np.isfinite(limited)] = 0.0
    raw_max = float(np.max(np.linalg.norm(surface_velocity, axis=1))) if surface_velocity.size else 0.0
    return limited, raw_max


def cox_voinov_speed_cap(config: base.RealMeshEvolutionConfig) -> float:
    theta_eq = float(config.contact_angle_deg) * math.pi / 180.0
    theta_max = float(config.dynamic_contact_angle_max_deg) * math.pi / 180.0
    log_factor = math.log(
        max(
            float(config.contact_line_cox_macro_length_m)
            / max(float(config.contact_line_cox_slip_length_m), 1.0e-30),
            math.e,
        )
    )
    return (
        float(config.surface_tension_n_m)
        / max(float(config.viscosity_pa_s), 1.0e-30)
        * max(theta_max**3 - theta_eq**3, 0.0)
        / max(9.0 * log_factor, 1.0e-30)
    )


def physical_lubrication_stiffness(
    state: TetraFreeSurfaceState,
    solve_tets: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    *,
    velocity_order: int = 1,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    """Return Case59's physical ``K_lub(h)`` and auditable diagnostics.

    The ordinary PR35 Cauchy block retains the measured viscosity ``mu``.
    This additional block uses the no-slip/shear-free thin-film mobility
    coefficient ``beta=3*mu/h**2``. It replaces the old extra ``3*mu`` proxy
    with its physical in-plane thin-film counterpart; it is not algebraically
    equivalent for normal motion. A changing local gap now changes the
    resistance through the derived ``h**-2`` law.
    """

    shape = (3 * len(state.nodes), 3 * len(state.nodes))
    model = str(LUBRICATION_RESISTANCE_MODEL).strip().lower()
    inactive = {
        "lubrication_resistance_active": 0.0,
        "lubrication_resistance_coefficient": 0.0,
        "lubrication_resistance_gap_min_m": 0.0,
        "lubrication_resistance_gap_max_m": 0.0,
        "lubrication_resistance_beta_min_pa_s_m2": 0.0,
        "lubrication_resistance_beta_max_pa_s_m2": 0.0,
        "lubrication_resistance_operator_l2_n_s_m": 0.0,
        "lubrication_resistance_film_tet_count": 0.0,
    }
    if model == "none":
        return sparse.csr_matrix(shape, dtype=float), inactive
    if model != "depth_averaged_free_surface":
        raise ValueError(
            "LUBRICATION_RESISTANCE_MODEL must be 'none' or "
            "'depth_averaged_free_surface'"
        )
    if not math.isclose(
        float(WALL_LUBRICATION_DRAG_FACTOR),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Gap-dependent K_lub requires wall-drag=1 so the old uniform "
            "viscosity multiplier is not counted at the same time"
        )

    tet_arr = np.asarray(solve_tets, dtype=int).reshape((-1, 4))
    surface_height = np.asarray(state.surface_points()[:, 2], dtype=float)
    column_nodes = np.mod(tet_arr, state.n_surface)
    # K_lub(h) is the depth-averaged resistance of the thin film above the
    # planar substrate.  It must not be assembled on bridge/contact cells:
    # their small sphere clearance is governed by the separate Cox-wedge
    # operator and is not a collapsed substrate film.  Keep only tetrahedra
    # whose complete vertical footprint belongs to the film topology.
    surface_is_film = np.zeros(state.n_surface, dtype=bool)
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size:
        surface_is_film[
            np.asarray(state.rings[film_rows], dtype=int).reshape(-1)
        ] = True
    film_tet = np.all(surface_is_film[column_nodes], axis=1)
    tet_arr = tet_arr[film_tet]
    column_nodes = column_nodes[film_tet]
    if tet_arr.size == 0:
        return sparse.csr_matrix(shape, dtype=float), inactive
    local_gap = np.mean(surface_height[column_nodes], axis=1)
    if (
        np.any(~np.isfinite(local_gap))
        or np.any(local_gap < TETRA_MIN_LAYER_HEIGHT_M)
    ):
        raise RuntimeError(
            "The physical lubrication operator found a non-admissible local "
            "gap; the step is rejected rather than clipping h or beta"
        )

    coefficient = float(LUBRICATION_RESISTANCE_COEFFICIENT)
    viscosity = float(config.viscosity_pa_s)
    matrix = tetra_depth_averaged_lubrication_stiffness(
        state.nodes,
        tet_arr,
        local_gap,
        viscosity,
        mobility_coefficient=coefficient,
        velocity_order=int(velocity_order),
    )
    beta = coefficient * viscosity / local_gap**2
    return matrix, {
        "lubrication_resistance_active": 1.0,
        "lubrication_resistance_coefficient": coefficient,
        "lubrication_resistance_gap_min_m": float(np.min(local_gap)),
        "lubrication_resistance_gap_max_m": float(np.max(local_gap)),
        "lubrication_resistance_beta_min_pa_s_m2": float(np.min(beta)),
        "lubrication_resistance_beta_max_pa_s_m2": float(np.max(beta)),
        "lubrication_resistance_operator_l2_n_s_m": float(
            sparse.linalg.norm(matrix)
        ),
        "lubrication_resistance_film_tet_count": float(len(tet_arr)),
    }


def solve_tetra_velocity(
    state: TetraFreeSurfaceState,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    dt_s: float,
    time_s: float,
) -> tuple[np.ndarray, dict[str, float]]:
    repair_diag = state.repair_tetra_connectivity()
    surface = state.surface_points()
    geom = base.attached_ring_geometry(surface, state.rings, ring_region)
    contact_ids = np.asarray(geom["contact_ring"], dtype=int)
    outer_ids = np.asarray(state.rings[-1], dtype=int)

    surface_force, surface_area, surface_pressure = cotangent_surface_tension_forces(
        surface,
        state.surface_faces,
        float(config.surface_tension_n_m),
    )
    top = slice(state.n_surface, 2 * state.n_surface)
    full_forces = np.zeros_like(state.nodes)
    full_forces[top] += surface_force

    tangents = sphere_tangent_vectors(surface, contact_ids, config)
    line_lengths = ring_segment_lengths(surface[contact_ids])
    wall_force = np.zeros_like(tangents)
    if bool(INCLUDE_YOUNG_WALL_FORCE):
        wall_force = (
            float(config.surface_tension_n_m)
            * math.cos(float(config.contact_angle_deg) * math.pi / 180.0)
            * line_lengths[:, None]
            * tangents
        )
    initial_contact_momentum_speed = np.zeros(
        len(contact_ids),
        dtype=float,
    )
    if (
        bool(INITIALIZE_CONTACT_MOMENTUM_FROM_COX_GEOMETRY)
        and float(time_s) <= 1.0e-14
        and contact_ids.size
        and float(
            np.max(
                np.linalg.norm(
                    state.velocities[top][contact_ids],
                    axis=1,
                )
            )
        )
        <= 1.0e-14
    ):
        initial_tangent_traction = (
            np.einsum(
                "ij,ij->i",
                surface_force[contact_ids],
                tangents,
            )
            / np.maximum(line_lengths, 1.0e-30)
        )
        initial_cosine = np.clip(
            -initial_tangent_traction
            / max(float(config.surface_tension_n_m), 1.0e-30),
            -1.0,
            1.0,
        )
        initial_theta = np.arccos(initial_cosine)
        if float(INITIAL_CONTACT_MOMENTUM_M_S) > 0.0:
            initial_contact_momentum_speed = np.full(
                len(contact_ids),
                float(INITIAL_CONTACT_MOMENTUM_M_S),
                dtype=float,
            )
        else:
            initial_contact_momentum_speed = np.asarray(
                [
                    cox_inverse_contact_line_speed(
                        theta_geo_rad=float(theta),
                        theta_eq_rad=(
                            float(config.contact_angle_deg)
                            * math.pi
                            / 180.0
                        ),
                        viscosity_pa_s=float(config.viscosity_pa_s),
                        surface_tension_n_m=float(
                            config.surface_tension_n_m
                        ),
                        macro_length_m=float(
                            config.contact_line_cox_macro_length_m
                        ),
                        slip_length_m=float(
                            config.contact_line_cox_slip_length_m
                        ),
                    )
                    for theta in initial_theta
                ],
                dtype=float,
            )
            initial_contact_momentum_speed = np.clip(
                initial_contact_momentum_speed,
                0.0,
                cox_voinov_speed_cap(config),
            )
        state.velocities[
            state.n_surface + contact_ids
        ] = initial_contact_momentum_speed[:, None] * tangents
    cox_force, theta_dyn, slide_speed = cox_contact_line_force_ring(
        points_m=surface[contact_ids],
        velocities_m_s=state.velocities[top][contact_ids],
        line_direction=tangents,
        slide_direction=tangents,
        surface_tension_n_m=float(config.surface_tension_n_m),
        theta_eq_rad=float(config.contact_angle_deg) * math.pi / 180.0,
        viscosity_pa_s=float(config.viscosity_pa_s),
        macro_length_m=float(config.contact_line_cox_macro_length_m),
        slip_length_m=float(config.contact_line_cox_slip_length_m),
        min_angle_rad=float(config.dynamic_contact_angle_min_deg) * math.pi / 180.0,
        max_angle_rad=float(config.dynamic_contact_angle_max_deg) * math.pi / 180.0,
    )
    # Evaluate the conservative film-supply active set on the same old state
    # used to assemble PR35.  If it becomes active, its speed bound is imposed
    # below by a contact-ring force reaction and the momentum block is solved
    # again.  No returned velocity component is overwritten.
    solver_mass_supply_speed_cap = float("inf")
    solver_mass_supply_flux_ul_s = 0.0
    solver_mass_supply_activation = 0.0
    solver_local_capture_volume_ul = 0.0
    solver_film_pressure_inner_pa = 0.0
    solver_film_pressure_next_pa = 0.0
    if contact_ids.size:
        (
            solver_mass_supply_speed_cap,
            solver_mass_supply_flux_ul_s,
            solver_mass_supply_activation,
            solver_local_capture_volume_ul,
            solver_film_pressure_inner_pa,
            solver_film_pressure_next_pa,
        ) = mass_limited_contact_line_speed_cap(
            surface,
            state.rings,
            ring_region,
            config,
            time_s=time_s,
        )
    full_forces[state.n_surface + contact_ids] += wall_force
    implicit_cox_this_step = bool(IMPLICIT_COX_FORCE_SOLVE_ENABLED)
    if not implicit_cox_this_step:
        full_forces[state.n_surface + contact_ids] += cox_force

    tet_volumes, volume_matrix = tet_volume_matrix_sparse(state.nodes, state.tets)
    solve_tets = state.tets[np.asarray(tet_volumes, dtype=float) > 1.0e-30]
    if solve_tets.size == 0:
        solve_tets = state.tets
    masses = lump_tet_masses(state.nodes.shape[0], state.tets, tet_volumes, float(config.density_kg_m3))
    full_forces[:, 2] += -masses * float(config.gravity_m_s2)
    constrained = np.zeros(state.nodes.shape[0], dtype=bool)
    constrained[: state.n_surface] = True
    constrained[state.column_vertex_ids(outer_ids)] = True
    mode_before = str(
        getattr(state, "case77_momentum_mode", "dynamic")
    )
    if mode_before not in {"dynamic", "stokes"}:
        mode_before = "dynamic"
    if not bool(ADAPTIVE_INERTIA_SWITCH_ENABLED):
        mode_before = "stokes"
    inertia_weight = 1.0 if mode_before == "dynamic" else 0.0

    effective_viscosity = float(config.viscosity_pa_s) * WALL_LUBRICATION_DRAG_FACTOR
    additional_stiffness, lubrication_diag = physical_lubrication_stiffness(
        state,
        solve_tets,
        ring_region,
        config,
        velocity_order=(
            1 if bool(ENFORCE_EXACT_AXISYMMETRIC_PR35) else 2
        ),
    )
    sphere_wedge_stiffness_l2 = 0.0
    sphere_wedge_zeta_mean = 0.0
    sphere_wedge_cutoff_mean_m = 0.0
    sphere_wedge_logarithm_mean = 0.0
    if bool(SPHERE_WEDGE_LUBRICATION_ENABLED) and contact_ids.size:
        declared_macro_length = max(
            float(config.contact_line_cox_macro_length_m),
            1.0e-30,
        )
        slip_length = max(
            float(config.contact_line_cox_slip_length_m),
            1.0e-30,
        )
        cutoff_length = np.full(
            len(contact_ids),
            declared_macro_length,
            dtype=float,
        )
        if bool(SPHERE_WEDGE_UNRESOLVED_ONLY):
            bridge_rows = np.flatnonzero(
                np.asarray(ring_region, dtype=int) == 0
            )
            if bridge_rows.size >= 2:
                contact_row = int(bridge_rows[0])
                neighbor_row = int(bridge_rows[1])
                contact_ring = np.asarray(
                    state.rings[contact_row],
                    dtype=int,
                )
                neighbor_ring = np.asarray(
                    state.rings[neighbor_row],
                    dtype=int,
                )
                local_edge = np.linalg.norm(
                    surface[neighbor_ring] - surface[contact_ring],
                    axis=1,
                )
                cutoff_length = np.minimum(
                    cutoff_length,
                    np.maximum(local_edge, math.e * slip_length),
                )
        logarithm = np.maximum(
            np.log(cutoff_length / slip_length),
            1.0,
        )
        minimum_resolved_angle = max(
            slip_length / float(np.max(cutoff_length)),
            1.0e-12,
        )
        wedge_angle = np.maximum(
            np.abs(np.asarray(theta_dyn, dtype=float)),
            minimum_resolved_angle,
        )
        wedge_zeta = (
            3.0
            * float(config.viscosity_pa_s)
            * logarithm
            / wedge_angle
        )
        wedge_rows: list[int] = []
        wedge_cols: list[int] = []
        wedge_data: list[float] = []
        for local_id, surface_id in enumerate(contact_ids):
            global_vertex = int(state.n_surface + int(surface_id))
            block = (
                float(wedge_zeta[local_id])
                * float(line_lengths[local_id])
                * np.outer(tangents[local_id], tangents[local_id])
            )
            base_dof = 3 * global_vertex
            for row_component in range(3):
                for col_component in range(3):
                    value = float(block[row_component, col_component])
                    if abs(value) <= 1.0e-30:
                        continue
                    wedge_rows.append(base_dof + row_component)
                    wedge_cols.append(base_dof + col_component)
                    wedge_data.append(value)
        wedge_stiffness = sparse.coo_matrix(
            (wedge_data, (wedge_rows, wedge_cols)),
            shape=additional_stiffness.shape,
        ).tocsr()
        additional_stiffness = additional_stiffness + wedge_stiffness
        sphere_wedge_stiffness_l2 = float(
            sparse.linalg.norm(wedge_stiffness)
        )
        sphere_wedge_zeta_mean = float(np.mean(wedge_zeta))
        sphere_wedge_cutoff_mean_m = float(np.mean(cutoff_length))
        sphere_wedge_logarithm_mean = float(np.mean(logarithm))
    p1_momentum_shape = (
        3 * len(state.nodes),
        3 * len(state.nodes),
    )
    capillary_stiffness = sparse.csr_matrix(
        p1_momentum_shape,
        dtype=float,
    )
    capillary_stiffness_l2 = 0.0
    if bool(IMPLICIT_CAPILLARY_STIFFNESS_ENABLED):
        surface_capillary_stiffness = cotangent_surface_tension_stiffness(
            surface,
            state.surface_faces,
            float(config.surface_tension_n_m),
        ).tocsr()
        if bool(IMPLICIT_CAPILLARY_NORMAL_PROJECTION_ENABLED):
            surface_normals = np.zeros_like(surface)
            face = np.asarray(state.surface_faces, dtype=int)
            face_normal = np.cross(
                surface[face[:, 1]] - surface[face[:, 0]],
                surface[face[:, 2]] - surface[face[:, 0]],
            )
            for local_vertex in range(3):
                np.add.at(
                    surface_normals,
                    face[:, local_vertex],
                    face_normal,
                )
            normal_length = np.linalg.norm(surface_normals, axis=1)
            valid_normal = normal_length > 1.0e-30
            surface_normals[valid_normal] /= normal_length[
                valid_normal, None
            ]
            projector_rows = []
            projector_cols = []
            projector_data = []
            for vertex_id, normal in enumerate(surface_normals):
                block = np.outer(normal, normal)
                base_dof = 3 * int(vertex_id)
                for row_component in range(3):
                    for col_component in range(3):
                        value = float(block[row_component, col_component])
                        if abs(value) <= 1.0e-30:
                            continue
                        projector_rows.append(base_dof + row_component)
                        projector_cols.append(base_dof + col_component)
                        projector_data.append(value)
            normal_projector = sparse.coo_matrix(
                (
                    projector_data,
                    (projector_rows, projector_cols),
                ),
                shape=surface_capillary_stiffness.shape,
            ).tocsr()
            surface_capillary_stiffness = (
                normal_projector.T
                @ surface_capillary_stiffness
                @ normal_projector
            ).tocsr()
        surface_capillary_stiffness = surface_capillary_stiffness.tocoo()
        if not bool(IMPLICIT_CAPILLARY_STIFFNESS_AT_CONTACT_LINE):
            contact_surface_vertex = np.zeros(
                state.n_surface,
                dtype=bool,
            )
            contact_surface_vertex[contact_ids] = True
            keep_entry = ~(
                contact_surface_vertex[
                    surface_capillary_stiffness.row // 3
                ]
                | contact_surface_vertex[
                    surface_capillary_stiffness.col // 3
                ]
            )
            surface_capillary_stiffness = sparse.coo_matrix(
                (
                    surface_capillary_stiffness.data[keep_entry],
                    (
                        surface_capillary_stiffness.row[keep_entry],
                        surface_capillary_stiffness.col[keep_entry],
                    ),
                ),
                shape=surface_capillary_stiffness.shape,
            )
        top_dof_offset = 3 * int(state.n_surface)
        capillary_stiffness = sparse.coo_matrix(
            (
                float(dt_s) * surface_capillary_stiffness.data,
                (
                    surface_capillary_stiffness.row + top_dof_offset,
                    surface_capillary_stiffness.col + top_dof_offset,
                ),
            ),
            shape=p1_momentum_shape,
        ).tocsr()
        if additional_stiffness.shape == p1_momentum_shape:
            additional_stiffness = (
                additional_stiffness + capillary_stiffness
            )
        else:
            capillary_coo = capillary_stiffness.tocoo()
            capillary_stiffness_p2 = sparse.coo_matrix(
                (
                    capillary_coo.data,
                    (capillary_coo.row, capillary_coo.col),
                ),
                shape=additional_stiffness.shape,
            ).tocsr()
            additional_stiffness = (
                additional_stiffness + capillary_stiffness_p2
            )
        capillary_stiffness_l2 = float(
            sparse.linalg.norm(capillary_stiffness)
        )
    tangent_vertex_directions = np.zeros_like(state.nodes)
    tangent_vertex_directions[
        state.n_surface + contact_ids
    ] = tangents
    full_surface_force = np.zeros_like(state.nodes)
    full_surface_force[top] = surface_force
    full_surface_area = np.zeros(state.nodes.shape[0], dtype=float)
    full_surface_area[state.n_surface : 2 * state.n_surface] = surface_area
    pressure_space_code = 0.0
    p2_edge_count = 0.0
    use_p0_solver = True
    implicit_cox_speed_m_s = 0.0
    implicit_cox_mobility_m2_n_s = 0.0
    implicit_cox_force_residual_n = 0.0
    implicit_cox_factorization_hits = 0.0
    implicit_cox_angle_active_set = 0.0
    implicit_cox_reaction_force_l2_n = 0.0
    implicit_mass_supply_active_set = 0.0
    try:
        if bool(ENFORCE_EXACT_AXISYMMETRIC_PR35):
            # P1/MINI is the compact meridional PR35 space.  It keeps the
            # exact 3-D nodal Heron array and projects that array with T.T.
            pressure_space_code = 1.1
            linear_system_cache: dict = {}

            def solve_reduced_pr35(
                force_array: np.ndarray,
            ) -> tuple[np.ndarray, object, object]:
                return implicit_tetra_stokes_velocity_pressure(
                    points=state.nodes,
                    tets=solve_tets,
                    velocities_m_s=state.velocities,
                    external_forces_n=force_array,
                    masses_kg=masses,
                    viscosity_pa_s=effective_viscosity,
                    dt_s=dt_s,
                    inertia_weight=inertia_weight,
                    constrained_vertices=constrained,
                    tangent_vertex_directions=tangent_vertex_directions,
                    enforce_exact_axisymmetry=True,
                    axisymmetry_tolerance_m=float(
                        AXISYMMETRY_TOLERANCE_M
                    ),
                    additional_stiffness_n_s_m=additional_stiffness,
                    pressure_stabilization_method="mini",
                    pressure_relative_regularization=1.0e-12,
                    rtol=2.0e-5,
                    maxiter=500,
                    linear_system_cache=linear_system_cache,
                )

            base_velocity, base_pressure, base_viscous = (
                solve_reduced_pr35(full_forces)
            )
            if (
                implicit_cox_this_step
                and contact_ids.size
            ):
                test_traction_n_m = max(
                    float(config.surface_tension_n_m),
                    1.0e-12,
                )
                test_forces = np.asarray(full_forces, dtype=float).copy()
                test_forces[state.n_surface + contact_ids] += (
                    test_traction_n_m
                    * line_lengths[:, None]
                    * tangents
                )
                test_velocity, _test_pressure, _test_viscous = (
                    solve_reduced_pr35(test_forces)
                )
                base_speed = float(
                    np.mean(
                        np.einsum(
                            "ij,ij->i",
                            base_velocity[top][contact_ids],
                            tangents,
                        )
                    )
                )
                test_speed = float(
                    np.mean(
                        np.einsum(
                            "ij,ij->i",
                            test_velocity[top][contact_ids],
                            tangents,
                        )
                    )
                )
                implicit_cox_mobility_m2_n_s = (
                    (test_speed - base_speed) / test_traction_n_m
                )

                def cox_traction(speed_m_s: float) -> float:
                    theta_value = cox_contact_line_force_ring(
                        points_m=surface[contact_ids],
                        velocities_m_s=(
                            float(speed_m_s) * tangents
                        ),
                        line_direction=tangents,
                        slide_direction=tangents,
                        surface_tension_n_m=float(
                            config.surface_tension_n_m
                        ),
                        theta_eq_rad=(
                            float(config.contact_angle_deg)
                            * math.pi
                            / 180.0
                        ),
                        viscosity_pa_s=float(config.viscosity_pa_s),
                        macro_length_m=float(
                            config.contact_line_cox_macro_length_m
                        ),
                        slip_length_m=float(
                            config.contact_line_cox_slip_length_m
                        ),
                        min_angle_rad=(
                            float(config.dynamic_contact_angle_min_deg)
                            * math.pi
                            / 180.0
                        ),
                        max_angle_rad=(
                            float(config.dynamic_contact_angle_max_deg)
                            * math.pi
                            / 180.0
                        ),
                    )[0]
                    return float(
                        np.mean(
                            np.einsum(
                                "ij,ij->i",
                                theta_value,
                                tangents,
                            )
                            / np.maximum(line_lengths, 1.0e-30)
                        )
                    )

                if implicit_cox_mobility_m2_n_s > 0.0:
                    lower_speed = min(base_speed, 0.0)
                    # Cox--Voinov is admissible only while the dynamic angle
                    # remains below its declared maximum.  If the unconstrained
                    # PR35 balance asks for a larger speed, activate a scalar
                    # contact reaction in the momentum solve.  This is a
                    # force-space complementarity condition, not a post-solve
                    # velocity overwrite or a global CFL clip.
                    cox_upper_speed = cox_voinov_speed_cap(config)
                    supply_upper_speed = (
                        max(float(solver_mass_supply_speed_cap), 0.0)
                        if (
                            bool(MASS_SUPPLY_CONTACT_REACTION_ENABLED)
                            and math.isfinite(
                                float(solver_mass_supply_speed_cap)
                            )
                        )
                        else float("inf")
                    )
                    upper_speed = min(
                        cox_upper_speed,
                        supply_upper_speed,
                    )

                    def residual(speed_m_s: float) -> float:
                        return (
                            float(speed_m_s)
                            - base_speed
                            - implicit_cox_mobility_m2_n_s
                            * cox_traction(float(speed_m_s))
                        )

                    cox_reaction_traction_n_m = 0.0
                    if residual(lower_speed) >= 0.0:
                        implicit_cox_speed_m_s = lower_speed
                    elif residual(upper_speed) <= 0.0:
                        implicit_cox_speed_m_s = upper_speed
                        if cox_upper_speed <= supply_upper_speed + 1.0e-15:
                            implicit_cox_angle_active_set = 1.0
                        if supply_upper_speed <= cox_upper_speed + 1.0e-15:
                            implicit_mass_supply_active_set = 1.0
                        cox_reaction_traction_n_m = (
                            (implicit_cox_speed_m_s - base_speed)
                            / implicit_cox_mobility_m2_n_s
                            - cox_traction(implicit_cox_speed_m_s)
                        )
                    else:
                        for _ in range(80):
                            middle_speed = 0.5 * (
                                lower_speed + upper_speed
                            )
                            if residual(middle_speed) <= 0.0:
                                lower_speed = middle_speed
                            else:
                                upper_speed = middle_speed
                        implicit_cox_speed_m_s = 0.5 * (
                            lower_speed + upper_speed
                        )
                else:
                    implicit_cox_speed_m_s = base_speed

                cox_force, theta_dyn, slide_speed = (
                    cox_contact_line_force_ring(
                        points_m=surface[contact_ids],
                        velocities_m_s=(
                            implicit_cox_speed_m_s * tangents
                        ),
                        line_direction=tangents,
                        slide_direction=tangents,
                        surface_tension_n_m=float(
                            config.surface_tension_n_m
                        ),
                        theta_eq_rad=(
                            float(config.contact_angle_deg)
                            * math.pi
                            / 180.0
                        ),
                        viscosity_pa_s=float(config.viscosity_pa_s),
                        macro_length_m=float(
                            config.contact_line_cox_macro_length_m
                        ),
                        slip_length_m=float(
                            config.contact_line_cox_slip_length_m
                        ),
                        min_angle_rad=(
                            float(config.dynamic_contact_angle_min_deg)
                            * math.pi
                            / 180.0
                        ),
                        max_angle_rad=(
                            float(config.dynamic_contact_angle_max_deg)
                            * math.pi
                            / 180.0
                        ),
                    )
                )
                final_forces = np.asarray(full_forces, dtype=float).copy()
                final_forces[state.n_surface + contact_ids] += cox_force
                if (
                    implicit_cox_angle_active_set > 0.5
                    or implicit_mass_supply_active_set > 0.5
                ):
                    cox_reaction_force = (
                        float(cox_reaction_traction_n_m)
                        * line_lengths[:, None]
                        * tangents
                    )
                    final_forces[
                        state.n_surface + contact_ids
                    ] += cox_reaction_force
                    implicit_cox_reaction_force_l2_n = float(
                        np.linalg.norm(cox_reaction_force)
                    )
                velocity, pressure_diag, viscous_diag = (
                    solve_reduced_pr35(final_forces)
                )
                full_forces = final_forces
                final_contact_speed = float(
                    np.mean(
                        np.einsum(
                            "ij,ij->i",
                            velocity[top][contact_ids],
                            tangents,
                        )
                    )
                )
                implicit_cox_force_residual_n = float(
                    abs(final_contact_speed - implicit_cox_speed_m_s)
                )
                implicit_cox_factorization_hits = float(
                    linear_system_cache.get("hits", 0)
                )
            else:
                velocity = base_velocity
                pressure_diag = base_pressure
                viscous_diag = base_viscous
        else:
            p2_surface_force, p2_edges = p2_consistent_surface_force(
                state.nodes,
                solve_tets,
                state.surface_faces + state.n_surface,
                full_surface_force,
                full_surface_area,
            )
            p2_force = p2_surface_force
            p2_force[: state.nodes.shape[0]] += (
                full_forces - full_surface_force
            )
            pressure_space_code = 2.1
            p2_edge_count = float(len(p2_edges))
            velocity, pressure_diag, viscous_diag = (
                implicit_tetra_taylor_hood_velocity_pressure(
                    points=state.nodes,
                    tets=solve_tets,
                    velocities_m_s=state.velocities,
                    external_forces_n=full_forces,
                    viscosity_pa_s=effective_viscosity,
                    density_kg_m3=float(config.density_kg_m3),
                    dt_s=dt_s,
                    inertia_weight=inertia_weight,
                    p2_external_forces_n=p2_force,
                    constrained_vertices=constrained,
                    tangent_vertex_directions=tangent_vertex_directions,
                    additional_stiffness_n_s_m=additional_stiffness,
                    pressure_relative_regularization=1.0e-12,
                    rtol=2.0e-5,
                    maxiter=500,
                )
            )
        use_p0_solver = False
    except (KeyError, ValueError, RuntimeError) as error:
        print(
            "Case77 reduced PR35 fallback: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        use_p0_solver = True

    if use_p0_solver:
        pressure_space_code = 1.1
        p2_edge_count = 0.0
        additional_stiffness, lubrication_diag = (
            physical_lubrication_stiffness(
                state,
                solve_tets,
                ring_region,
                config,
                velocity_order=1,
            )
        )
        additional_stiffness = additional_stiffness + capillary_stiffness
        velocity, pressure_diag, viscous_diag = implicit_tetra_stokes_velocity_pressure(
            points=state.nodes,
            tets=solve_tets,
            velocities_m_s=state.velocities,
            external_forces_n=full_forces,
            masses_kg=masses,
            viscosity_pa_s=effective_viscosity,
            dt_s=dt_s,
            inertia_weight=inertia_weight,
            constrained_vertices=constrained,
            tangent_vertex_directions=tangent_vertex_directions,
            enforce_exact_axisymmetry=bool(
                ENFORCE_EXACT_AXISYMMETRIC_PR35
            ),
            axisymmetry_tolerance_m=float(AXISYMMETRY_TOLERANCE_M),
            additional_stiffness_n_s_m=additional_stiffness,
            pressure_stabilization_method="mini",
            pressure_relative_regularization=1.0e-10,
            rtol=2.0e-5,
            maxiter=500,
        )
    # Use the nodal mass contribution of the just-computed PR35 velocity as
    # the regime trigger. In Stokes mode this is the omitted, or "virtual",
    # inertia. Comparing it with the exact assembled external-force array
    # makes the trigger dimensionless and independent of experimental data.
    free_vertices = ~constrained
    velocity_increment = (
        np.asarray(velocity[free_vertices], dtype=float)
        - np.asarray(state.velocities[free_vertices], dtype=float)
    )
    inertial_force_n = (
        np.asarray(masses[free_vertices], dtype=float)[:, None]
        * velocity_increment
        / max(float(dt_s), 1.0e-30)
    )
    inertial_force_l2_n = float(np.linalg.norm(inertial_force_n))
    comparison_force_l2_n = float(
        np.linalg.norm(np.asarray(full_forces[free_vertices], dtype=float))
    )
    inertia_ratio = inertial_force_l2_n / max(
        comparison_force_l2_n,
        1.0e-30,
    )
    quiet_count = int(
        getattr(state, "case77_dynamic_quiet_count", 0)
    )
    switch_event = 0.0
    if bool(ADAPTIVE_INERTIA_SWITCH_ENABLED):
        if (
            mode_before == "stokes"
            and inertia_ratio
            > float(STOKES_TO_DYNAMIC_INERTIA_RATIO)
        ):
            # Reject the Stokes trial and re-solve this identical geometry and
            # force state with backward-Euler inertia. No quasi-static
            # displacement is allowed to leak into the dynamic branch.
            state.case77_momentum_mode = "dynamic"
            state.case77_dynamic_quiet_count = 0
            dynamic_velocity, dynamic_diagnostics = solve_tetra_velocity(
                state,
                ring_region,
                config,
                dt_s,
                time_s,
            )
            dynamic_diagnostics["case77_switch_event"] = -1.0
            dynamic_diagnostics[
                "case77_rejected_stokes_trial_inertia_ratio"
            ] = float(inertia_ratio)
            dynamic_diagnostics["case77_stokes_trial_rejected"] = 1.0
            return dynamic_velocity, dynamic_diagnostics
        if mode_before == "dynamic":
            quiet_count = (
                quiet_count + 1
                if inertia_ratio
                < float(DYNAMIC_TO_STOKES_INERTIA_RATIO)
                else 0
            )
            if quiet_count >= max(
                int(DYNAMIC_TO_STOKES_CONSECUTIVE_SOLVES),
                1,
            ):
                state.case77_momentum_mode = "stokes"
                state.case77_dynamic_quiet_count = 0
                switch_event = 1.0
            else:
                state.case77_momentum_mode = "dynamic"
                state.case77_dynamic_quiet_count = quiet_count
        else:
            state.case77_momentum_mode = "stokes"
            state.case77_dynamic_quiet_count = 0
    else:
        state.case77_momentum_mode = "stokes"
        state.case77_dynamic_quiet_count = 0
    volume_diag = pressure_diag
    surface_velocity_raw = velocity[top].copy()
    raw_vmax = (
        float(np.max(np.linalg.norm(surface_velocity_raw, axis=1)))
        if surface_velocity_raw.size
        else 0.0
    )
    if not np.all(np.isfinite(surface_velocity_raw)):
        raise RuntimeError(
            "Case77 PR35 returned a non-finite velocity; reject the step"
        )
    if bool(GLOBAL_VELOCITY_CLIPPING_ENABLED):
        surface_velocity, raw_vmax = velocity_limiter(
            surface_velocity_raw,
            surface,
            config,
            dt_s,
        )
    else:
        surface_velocity = surface_velocity_raw.copy()
    cox_speed_cap = cox_voinov_speed_cap(config)
    supply_factor = 1.0
    supply_ratio = 0.0
    supply_capacity_ul = 0.0
    mass_supply_speed_cap = float("inf")
    mass_supply_flux_ul_s = 0.0
    mass_supply_activation = 0.0
    local_capture_volume_ul = 0.0
    mass_supply_flux_speed_cap = float("inf")
    mass_supply_budget_before_ul = float(getattr(state, "supply_budget_ul", 0.0))
    mass_supply_budget_allowed_ul = 0.0
    mass_supply_budget_remaining_ul = 0.0
    mass_supply_budget_speed_cap = float("inf")
    film_pressure_inner_pa = 0.0
    film_pressure_next_pa = 0.0
    pr35_contact_tangent_speed = np.zeros(0, dtype=float)
    if contact_ids.size:
        supply_factor, supply_ratio, supply_capacity_ul = contact_line_supply_factor(
            surface,
            state.rings,
            ring_region,
            config,
        )
        (
            mass_supply_speed_cap,
            mass_supply_flux_ul_s,
            mass_supply_activation,
            local_capture_volume_ul,
            film_pressure_inner_pa,
            film_pressure_next_pa,
        ) = (
            float(solver_mass_supply_speed_cap),
            float(solver_mass_supply_flux_ul_s),
            float(solver_mass_supply_activation),
            float(solver_local_capture_volume_ul),
            float(solver_film_pressure_inner_pa),
            float(solver_film_pressure_next_pa),
        )
        mass_supply_flux_speed_cap = float(mass_supply_speed_cap)
        inventory_ul = bridge_inventory_volume_ul(surface, state.rings, ring_region, config)
        mass_supply_budget_allowed_ul = (
            float(local_capture_volume_ul)
            + float(mass_supply_budget_before_ul)
            + max(float(mass_supply_flux_ul_s), 0.0) * float(dt_s)
        )
        if float(mass_supply_activation) > 0.0:
            budget_remaining_ul = max(float(mass_supply_budget_allowed_ul) - float(inventory_ul), 0.0)
            d_cap_volume_dr = attached_cap_volume_derivative_m2(config, float(geom["contact_radius_m"]))
            mass_supply_budget_speed_cap = (
                budget_remaining_ul * 1.0e-9 / max(float(dt_s), 1.0e-30) / d_cap_volume_dr
            )
            if max(float(mass_supply_flux_ul_s), 0.0) <= 1.0e-14:
                mass_supply_speed_cap = float(mass_supply_budget_speed_cap)
            else:
                mass_supply_speed_cap = min(float(mass_supply_speed_cap), float(mass_supply_budget_speed_cap))
            mass_supply_budget_remaining_ul = float(budget_remaining_ul)
        contact_velocity = surface_velocity[contact_ids]
        signed_tangent_speed = np.einsum("ij,ij->i", contact_velocity, tangents)
        pr35_contact_tangent_speed = np.asarray(
            signed_tangent_speed,
            dtype=float,
        ).copy()
        line_length = np.maximum(line_lengths, 1.0e-30)
        tangent_traction = np.einsum("ij,ij->i", surface_force[contact_ids], tangents) / line_length
        cos_geometric = np.clip(
            -tangent_traction / max(float(config.surface_tension_n_m), 1.0e-30),
            -1.0,
            1.0,
        )
        theta_geometric = np.arccos(cos_geometric)
        inverse_cox_speed = np.asarray(
            [
                cox_inverse_contact_line_speed(
                    theta_geo_rad=float(theta),
                    theta_eq_rad=float(config.contact_angle_deg) * math.pi / 180.0,
                    viscosity_pa_s=float(config.viscosity_pa_s),
                    surface_tension_n_m=float(config.surface_tension_n_m),
                    macro_length_m=float(config.contact_line_cox_macro_length_m),
                    slip_length_m=float(config.contact_line_cox_slip_length_m),
                )
                for theta in theta_geometric
            ],
            dtype=float,
        )
        inverse_cox_speed = (
            np.clip(inverse_cox_speed, 0.0, cox_speed_cap) * supply_factor
        )
        if bool(POSTSOLVE_CONTACT_SPEED_OVERWRITE_ENABLED):
            signed_tangent_speed = np.maximum(
                signed_tangent_speed,
                inverse_cox_speed,
            )
            advancing = signed_tangent_speed > 0.0
            signed_tangent_speed[advancing] = np.minimum(
                signed_tangent_speed[advancing],
                float(mass_supply_speed_cap),
            )
            signed_tangent_speed = np.clip(
                signed_tangent_speed,
                -0.25 * cox_speed_cap,
                cox_speed_cap,
            )
            surface_velocity[contact_ids] = (
                signed_tangent_speed[:, None] * tangents
            )
    velocity[top] = surface_velocity
    velocity[: state.n_surface] = 0.0
    velocity[state.column_vertex_ids(outer_ids)] = 0.0
    node_radius = np.hypot(state.nodes[:, 0], state.nodes[:, 1])
    active_radius = node_radius > float(AXISYMMETRY_TOLERANCE_M)
    force_theta = np.zeros(len(state.nodes), dtype=float)
    velocity_theta = np.zeros(len(state.nodes), dtype=float)
    force_theta[active_radius] = (
        -full_forces[active_radius, 0] * state.nodes[active_radius, 1]
        + full_forces[active_radius, 1] * state.nodes[active_radius, 0]
    ) / node_radius[active_radius]
    velocity_theta[active_radius] = (
        -velocity[active_radius, 0] * state.nodes[active_radius, 1]
        + velocity[active_radius, 1] * state.nodes[active_radius, 0]
    ) / node_radius[active_radius]
    # Preserve the solved PR35 pressure field for downstream multiphysics
    # closures.  The momentum solve previously returned only pressure norms,
    # which forced bridge--film junction models to reconstruct pressure from
    # noisy surface geometry.  Consumers must use pressure differences because
    # the incompressible pressure gauge is arbitrary.
    state.last_pressure_projection_pa = np.asarray(
        pressure_diag.pressure_pa,
        dtype=float,
    ).copy()
    state.last_pressure_space_code = float(pressure_space_code)
    diagnostics = {
        "surface_pressure_equivalent_pa": float(surface_pressure),
        "surface_force_l2_n": float(np.linalg.norm(surface_force)),
        "cox_force_l2_n": float(np.linalg.norm(cox_force)),
        "wall_force_l2_n": float(np.linalg.norm(wall_force)),
        "case77_young_wall_force_active": (
            1.0 if bool(INCLUDE_YOUNG_WALL_FORCE) else 0.0
        ),
        "case77_initial_contact_momentum_speed_mean_m_s": (
            float(np.mean(initial_contact_momentum_speed))
            if initial_contact_momentum_speed.size
            else 0.0
        ),
        "raw_velocity_max_m_s": raw_vmax,
        "velocity_max_m_s": float(np.max(np.linalg.norm(surface_velocity, axis=1))),
        "case77_global_velocity_clipping_active": (
            1.0 if bool(GLOBAL_VELOCITY_CLIPPING_ENABLED) else 0.0
        ),
        "case77_postsolve_contact_speed_overwrite_active": (
            1.0
            if bool(POSTSOLVE_CONTACT_SPEED_OVERWRITE_ENABLED)
            else 0.0
        ),
        "case77_implicit_cox_force_solve_active": (
            1.0 if bool(IMPLICIT_COX_FORCE_SOLVE_ENABLED) else 0.0
        ),
        "case77_implicit_cox_speed_m_s": float(
            implicit_cox_speed_m_s
        ),
        "case77_implicit_cox_mobility_m2_n_s": float(
            implicit_cox_mobility_m2_n_s
        ),
        "case77_implicit_cox_speed_residual_m_s": float(
            implicit_cox_force_residual_n
        ),
        "case77_implicit_cox_factorization_hits": float(
            implicit_cox_factorization_hits
        ),
        "case77_cox_angle_active_set": float(
            implicit_cox_angle_active_set
        ),
        "case77_cox_reaction_force_l2_n": float(
            implicit_cox_reaction_force_l2_n
        ),
        "case77_mass_supply_active_set": float(
            implicit_mass_supply_active_set
        ),
        "case77_exact_axisymmetric_pr35_active": (
            1.0 if bool(ENFORCE_EXACT_AXISYMMETRIC_PR35) else 0.0
        ),
        "case77_implicit_capillary_stiffness_active": (
            1.0 if bool(IMPLICIT_CAPILLARY_STIFFNESS_ENABLED) else 0.0
        ),
        "case77_implicit_capillary_stiffness_l2_n_s_m": (
            capillary_stiffness_l2
        ),
        "case77_sphere_wedge_lubrication_active": (
            1.0 if bool(SPHERE_WEDGE_LUBRICATION_ENABLED) else 0.0
        ),
        "case77_sphere_wedge_zeta_mean_pa_s": float(
            sphere_wedge_zeta_mean
        ),
        "case77_sphere_wedge_cutoff_mean_m": float(
            sphere_wedge_cutoff_mean_m
        ),
        "case77_sphere_wedge_logarithm_mean": float(
            sphere_wedge_logarithm_mean
        ),
        "case77_sphere_wedge_unresolved_only": (
            1.0 if bool(SPHERE_WEDGE_UNRESOLVED_ONLY) else 0.0
        ),
        "case77_sphere_wedge_stiffness_l2_n_s_m": float(
            sphere_wedge_stiffness_l2
        ),
        "case77_full_3d_heron_force_l2_n": float(
            np.linalg.norm(surface_force)
        ),
        "case77_force_theta_preprojection_l2_n": float(
            np.linalg.norm(force_theta)
        ),
        "case77_velocity_theta_postsolve_linf_m_s": float(
            np.max(np.abs(velocity_theta))
            if velocity_theta.size
            else 0.0
        ),
        "pressure_linf_pa": float(pressure_diag.pressure_linf_pa),
        "pressure_residual_after_m3_s": float(pressure_diag.residual_after_m3_s),
        "volume_projection_residual_after_m3_s": float(volume_diag.residual_after_m3_s),
        "viscous_residual_l2_n": float(viscous_diag.residual_l2_n),
        "effective_tetra_viscosity_pa_s": float(effective_viscosity),
        "wall_lubrication_drag_factor": float(WALL_LUBRICATION_DRAG_FACTOR),
        "case77_momentum_mode_code": (
            1.0 if mode_before == "dynamic" else 0.0
        ),
        "case77_next_momentum_mode_code": (
            1.0
            if str(getattr(state, "case77_momentum_mode", "dynamic"))
            == "dynamic"
            else 0.0
        ),
        "case77_momentum_inertia_weight": float(inertia_weight),
        "case77_inertial_force_l2_n": inertial_force_l2_n,
        "case77_inertia_comparison_force_l2_n": comparison_force_l2_n,
        "case77_inertia_ratio": float(inertia_ratio),
        "case77_dynamic_to_stokes_ratio": float(
            DYNAMIC_TO_STOKES_INERTIA_RATIO
        ),
        "case77_stokes_to_dynamic_ratio": float(
            STOKES_TO_DYNAMIC_INERTIA_RATIO
        ),
        "case77_dynamic_quiet_count": float(quiet_count),
        "case77_switch_event": float(switch_event),
        "case77_stokes_trial_rejected": 0.0,
        "case77_rejected_stokes_trial_inertia_ratio": 0.0,
        "tetra_pressure_space": float(pressure_space_code),
        "p2_edge_count": float(p2_edge_count),
        "contact_theta_dynamic_rad": float(np.mean(theta_dyn)) if np.size(theta_dyn) else 0.0,
        "contact_slide_speed_m_s": float(np.mean(slide_speed)) if np.size(slide_speed) else 0.0,
        "case77_pr35_contact_tangent_speed_mean_m_s": (
            float(np.mean(pr35_contact_tangent_speed))
            if pr35_contact_tangent_speed.size
            else 0.0
        ),
        "case77_pr35_contact_tangent_speed_linf_m_s": (
            float(np.max(np.abs(pr35_contact_tangent_speed)))
            if pr35_contact_tangent_speed.size
            else 0.0
        ),
        "cox_voinov_speed_cap_m_s": float(cox_speed_cap),
        "inverse_cox_speed_m_s": float(np.mean(inverse_cox_speed)) if contact_ids.size else 0.0,
        "inverse_cox_theta_geo_rad": float(np.mean(theta_geometric)) if contact_ids.size else 0.0,
        "contact_line_supply_factor": float(supply_factor),
        "contact_line_supply_ratio": float(supply_ratio),
        "contact_line_supply_capacity_ul": float(supply_capacity_ul),
        "mass_supply_speed_cap_m_s": float(mass_supply_speed_cap) if math.isfinite(mass_supply_speed_cap) else 0.0,
        "mass_supply_flux_speed_cap_m_s": (
            float(mass_supply_flux_speed_cap) if math.isfinite(mass_supply_flux_speed_cap) else 0.0
        ),
        "mass_supply_flux_ul_s": float(mass_supply_flux_ul_s),
        "mass_supply_activation": float(mass_supply_activation),
        "local_capture_volume_ul": float(local_capture_volume_ul),
        "mass_supply_budget_before_ul": float(mass_supply_budget_before_ul),
        "mass_supply_budget_allowed_ul": float(mass_supply_budget_allowed_ul),
        "mass_supply_budget_remaining_ul": float(mass_supply_budget_remaining_ul),
        "mass_supply_budget_speed_cap_m_s": (
            float(mass_supply_budget_speed_cap) if math.isfinite(mass_supply_budget_speed_cap) else 0.0
        ),
        "film_pressure_flux_inner_pa": float(film_pressure_inner_pa),
        "film_pressure_flux_next_pa": float(film_pressure_next_pa),
        "film_pressure_flux_cap_enabled": 1.0 if bool(FILM_PRESSURE_FLUX_CONTACT_LINE_CAP_ENABLED) else 0.0,
        "contact_radius_m": float(geom["contact_radius_m"]),
        "rim_radius_m": float(geom["rim_radius_m"]),
    }
    diagnostics.update(lubrication_diag)
    diagnostics.update(repair_diag)
    return velocity, diagnostics


def apply_boundary_projection(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    outer_reference_z: np.ndarray | None = None,
) -> None:
    geom = base.attached_ring_geometry(surface, rings, ring_region)
    contact_ids = np.asarray(geom["contact_ring"], dtype=int)
    outer_ids = np.asarray(rings[-1], dtype=int)
    project_contact_ring_to_sphere(surface, contact_ids, config)
    h_min = max(0.1e-6, float(config.attached_capillary_bridge_min_height_um) * 1.0e-6)
    non_contact = np.ones(surface.shape[0], dtype=bool)
    non_contact[contact_ids] = False
    surface[non_contact, 2] = np.maximum(surface[non_contact, 2], h_min)
    if outer_reference_z is None:
        surface[outer_ids, 2] = surface[outer_ids, 2]
    else:
        surface[outer_ids, 2] = np.asarray(outer_reference_z, dtype=float)


def apply_axisymmetric_outer_film_lubrication(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    dt_s: float,
) -> dict[str, float]:
    """Evolve the resolved outer-film height by the thin-film equation.

    This is a physical replacement for the old visible-profile target forcing.
    It uses the ring-averaged free-surface height and advances

        dh/dt = (1/r) d/dr [ r h^3/(3 mu) d/dr (gamma kappa - rho g h) ]

    with no imposed experimental target.  The contact/bridge geometry remains
    provided by the tetra free-surface solve; this operator only lets the
    outer film redistribute under its own capillary/gravity pressure gradient.
    """

    if dt_s <= 0.0 or not bool(PHYSICAL_LUBRICATION_OUTER_FILM_ENABLED):
        return {"outer_film_lubrication_max_dh_um": 0.0}
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size < 5:
        return {"outer_film_lubrication_max_dh_um": 0.0}

    row_r = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    row_z = np.asarray([float(np.mean(surface[ring, 2])) for ring in rings], dtype=float)
    rows = film_rows
    r = row_r[rows].copy()
    h = row_z[rows].copy()
    order = np.argsort(r)
    rows = rows[order]
    r = r[order]
    h = h[order]
    if np.any(np.diff(r) <= 0.0):
        return {"outer_film_lubrication_max_dh_um": 0.0}

    h0 = np.asarray(flat_initial_profile_m(config, r), dtype=float)
    h_floor = max(TETRA_MIN_LAYER_HEIGHT_M, 0.05e-6)
    h_ceiling = np.maximum(1.05 * h0, h_floor)
    mu = max(float(config.viscosity_pa_s), 1.0e-30)
    gamma = max(float(config.surface_tension_n_m), 1.0e-30)
    rho_g = float(config.density_kg_m3) * float(config.gravity_m_s2)
    substeps = max(1, int(OUTER_FILM_LUBRICATION_SUBSTEPS))
    sub_dt = float(dt_s) / float(substeps)
    max_dh = 0.0
    dr_cell = np.empty_like(r)
    dr_cell[1:-1] = 0.5 * (r[2:] - r[:-2])
    dr_cell[0] = r[1] - r[0]
    dr_cell[-1] = r[-1] - r[-2]

    for _ in range(substeps):
        h = np.clip(h, h_floor, h_ceiling)
        dhdr = np.gradient(h, r, edge_order=2)
        d2hdr2 = np.gradient(dhdr, r, edge_order=2)
        curvature = d2hdr2 + dhdr / np.maximum(r, 1.0e-12)
        pressure = -gamma * curvature + rho_g * (h - h0)
        dpdr_face = (pressure[1:] - pressure[:-1]) / np.maximum(r[1:] - r[:-1], 1.0e-30)
        h_face = np.maximum(0.5 * (h[1:] + h[:-1]), h_floor)
        q_face = -(h_face**3 / (3.0 * mu)) * dpdr_face
        r_face = 0.5 * (r[1:] + r[:-1])
        q_edge = np.zeros(h.size + 1, dtype=float)
        q_edge[1:-1] = q_face
        r_edge = np.empty(h.size + 1, dtype=float)
        r_edge[1:-1] = r_face
        r_edge[0] = max(r[0] - 0.5 * dr_cell[0], 1.0e-12)
        r_edge[-1] = r[-1] + 0.5 * dr_cell[-1]
        divergence = (
            r_edge[1:] * q_edge[1:] - r_edge[:-1] * q_edge[:-1]
        ) / np.maximum(r * dr_cell, 1.0e-30)
        dh = -sub_dt * divergence
        limit = max(float(OUTER_FILM_LUBRICATION_MAX_FRACTIONAL_STEP), 1.0e-6) * np.maximum(h0, h_floor)
        scale = min(1.0, float(np.min(limit / np.maximum(np.abs(dh), 1.0e-30))))
        dh *= scale
        h_new = np.clip(h + dh, h_floor, h_ceiling)
        h_new[-1] = h0[-1]
        max_dh = max(max_dh, float(np.max(np.abs(h_new - h))))
        h = h_new

    for row_id, z in zip(rows, h):
        ids = np.asarray(rings[int(row_id)], dtype=int)
        surface[ids, 2] = float(z)
    return {"outer_film_lubrication_max_dh_um": float(max_dh * 1.0e6)}


def enforce_outer_film_single_trough_recovery(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
) -> None:
    """Remove ALE-created detached troughs in the local outer-film recovery.

    The bridge/film junction should produce one capillary depression followed
    by recovery toward the surrounding film.  A second lower minimum outside
    the rim is a remapping artifact because there is no separate pressure
    extremum or contact line there.  Enforce monotone recovery only over the
    local capillary-length neighborhood; the far finite-film edge remains
    governed by the mesh state.
    """

    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size < 3:
        return
    geom = base.attached_ring_geometry(surface, rings, ring_region)
    rim_r = float(geom["rim_radius_m"])
    gamma = max(float(config.surface_tension_n_m), 1.0e-30)
    capillary_length = math.sqrt(
        gamma / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-30)
    )
    substrate_r = float(config.substrate_radius_mm) * 1.0e-3
    cutoff = min(
        substrate_r - 0.5e-3,
        rim_r + float(OUTER_FILM_SINGLE_TROUGH_RECOVERY_CAPILLARY_LENGTHS) * capillary_length,
    )
    row_r = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    local_rows = film_rows[row_r[film_rows] <= cutoff]
    if local_rows.size < 2:
        return
    h0_local = np.asarray(flat_initial_profile_m(config, row_r[local_rows]), dtype=float)
    row_z = np.asarray([float(np.mean(surface[ring, 2])) for ring in rings], dtype=float)
    local_z = row_z[local_rows].copy()
    min_pos = int(np.argmin(local_z))
    recovered = local_z.copy()
    recovered[min_pos:] = np.maximum.accumulate(local_z[min_pos:])
    recovered = np.minimum(recovered, 1.02 * h0_local)
    for row_id, z in zip(local_rows, recovered):
        surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(max(z, TETRA_MIN_LAYER_HEIGHT_M))


def apply_coupled_local_neck_profile(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    time_s: float | None,
    target_visible_missing_ul: float,
) -> dict[str, float]:
    """Redistribute the resolved film deficit through a local YL/Cox neck solve."""

    if (not bool(COUPLED_LOCAL_NECK_ENABLED)) or target_visible_missing_ul <= 1.0e-8:
        return {"coupled_neck_applied": 0.0}
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size < 5:
        return {"coupled_neck_applied": 0.0}
    geom = base.attached_ring_geometry(surface, rings, ring_region)
    rim_r = float(geom["rim_radius_m"])
    contact_r = float(geom["contact_radius_m"])
    row_r = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    rows = film_rows[row_r[film_rows] >= rim_r]
    if rows.size < 5:
        return {"coupled_neck_applied": 0.0}
    rows = rows[np.argsort(row_r[rows])]
    r = row_r[rows]
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    initial_h = flat_initial_profile_m(config, r)
    capillary_length = math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-30)
    )
    elapsed = max(float(time_s) if time_s is not None else float(config.dt_s), float(config.dt_s), 1.0e-9)
    reference_r = float(config.initial_bridge_radius_mm) * 1.0e-3
    slide_speed = max(contact_r - reference_r, 0.0) / elapsed
    rim_speed_ratio = float(
        np.clip(slide_speed / max(cox_voinov_speed_cap(config), 1.0e-30), 0.0, 1.0)
    )
    width_speed_ratio = np.clip(
        slide_speed
        / max(cox_voinov_speed_cap(config) * float(COUPLED_LOCAL_NECK_WIDTH_SPEED_RATIO), 1.0e-30),
        0.0,
        1.0,
    )
    width_speed_weight = float(width_speed_ratio) ** 0.5
    angle_slow_weight = 1.0 - float(
        np.clip(
            slide_speed
            / max(cox_voinov_speed_cap(config) * float(COUPLED_LOCAL_NECK_ANGLE_SPEED_RATIO), 1.0e-30),
            0.0,
            1.0,
        )
    ) ** 0.5
    local_max_angle_deg = (
        float(COUPLED_LOCAL_NECK_FAST_MAX_ANGLE_DEG)
        + (
            float(COUPLED_LOCAL_NECK_SLOW_MAX_ANGLE_DEG)
            - float(COUPLED_LOCAL_NECK_FAST_MAX_ANGLE_DEG)
        )
        * angle_slow_weight
    )
    width_capillary_lengths = (
        float(COUPLED_LOCAL_NECK_MAX_WIDTH_CAPILLARY_LENGTHS)
        + (
            float(COUPLED_LOCAL_NECK_FAST_MAX_WIDTH_CAPILLARY_LENGTHS)
            - float(COUPLED_LOCAL_NECK_MAX_WIDTH_CAPILLARY_LENGTHS)
        )
        * width_speed_weight
    )
    width_film_thicknesses = (
        float(COUPLED_LOCAL_NECK_MAX_WIDTH_MIN_FILM_THICKNESSES)
        + (
            float(COUPLED_LOCAL_NECK_FAST_MAX_WIDTH_MIN_FILM_THICKNESSES)
            - float(COUPLED_LOCAL_NECK_MAX_WIDTH_MIN_FILM_THICKNESSES)
        )
        * width_speed_weight
    )
    max_width = max(
        width_capillary_lengths * capillary_length,
        width_film_thicknesses * h0,
    )
    neck_precursor_m = max(
        TETRA_MIN_LAYER_HEIGHT_M,
        float(COUPLED_LOCAL_NECK_PRECURSOR_FRACTION) * h0,
    )
    repulsion_scale_pa = (
        float(COUPLED_LOCAL_NECK_REPULSION_PRESSURE_SCALE)
        * float(config.surface_tension_n_m)
        / max(capillary_length, 1.0e-30)
    )
    result = coupled_cox_young_laplace_lubrication_neck_profile(
        r_m=r,
        initial_h_m=initial_h,
        rim_radius_m=rim_r,
        target_visible_missing_m3=float(target_visible_missing_ul) * 1.0e-9,
        initial_film_thickness_m=h0,
        slide_speed_m_s=slide_speed,
        viscosity_pa_s=float(config.viscosity_pa_s),
        surface_tension_n_m=float(config.surface_tension_n_m),
        density_kg_m3=float(config.density_kg_m3),
        gravity_m_s2=float(config.gravity_m_s2),
        theta_eq_rad=float(config.contact_angle_deg) * math.pi / 180.0,
        macro_length_m=float(config.contact_line_cox_macro_length_m),
        slip_length_m=float(config.contact_line_cox_slip_length_m),
        min_angle_rad=float(config.dynamic_contact_angle_min_deg) * math.pi / 180.0,
        max_angle_rad=float(local_max_angle_deg) * math.pi / 180.0,
        min_width_m=float(COUPLED_LOCAL_NECK_MIN_WIDTH_UM) * 1.0e-6,
        max_width_m=max_width,
        min_z_m=neck_precursor_m,
        precursor_height_m=neck_precursor_m,
        repulsion_pressure_scale_pa=repulsion_scale_pa,
        bvp_nodes=96,
    )
    z = np.asarray(result.get("z_m", initial_h), dtype=float)
    if z.shape != r.shape or not np.all(np.isfinite(z)):
        return {"coupled_neck_applied": 0.0}
    z = np.clip(z, neck_precursor_m, 1.02 * h0)
    for row_id, row_z in zip(rows, z):
        surface[np.asarray(rings[int(row_id)], dtype=int), 2] = float(row_z)
    return {
        "coupled_neck_applied": 1.0,
        "coupled_neck_width_mm": float(result.get("neck_width_m", 0.0)) * 1.0e3,
        "coupled_neck_visible_missing_ul": float(result.get("visible_missing_m3", 0.0)) * 1.0e9,
        "coupled_neck_success": 1.0 if bool(result.get("success", False)) else 0.0,
    }


def adaptive_radial_reprojection(
    state: TetraFreeSurfaceState,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    time_s: float | None = None,
    dt_s: float = 0.0,
) -> None:
    """ALE remap for the axisymmetric tetra layer after a free-surface move.

    The tetra solve moves boundary nodes in physical space, but a fixed ring
    connectivity can invert when the sphere contact line advances faster than
    the bridge/film rim.  This remap preserves the computed meridian heights,
    rebuilds a monotone radial distribution, and moves the substrate-side mesh
    vertices consistently so the next tetra solve uses a valid volume mesh.
    """

    surface = state.surface_points().copy()
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
    if bridge_rows.size < 2 or film_rows.size < 2:
        state.set_surface_points(surface)
        return

    theta = np.arctan2(surface[:, 1], surface[:, 0])
    old_r_by_row = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    old_z_by_row = np.asarray([float(np.mean(surface[ring, 2])) for ring in rings], dtype=float)
    contact_row = int(bridge_rows[0])
    rim_row = int(bridge_rows[-1])
    first_film_row = int(film_rows[0])
    outer_row = int(film_rows[-1])
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    substrate_r = float(config.substrate_radius_mm) * 1.0e-3
    capillary_length = math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-30)
    )
    contact_r = max(old_r_by_row[contact_row], float(config.initial_bridge_radius_mm) * 1.0e-3)
    contact_r = min(contact_r, 0.82 * substrate_r)

    # The bridge/film rim has to stay outside the sphere contact line.  Use a
    # capillary-length gap so the dimple/recovery front is controlled by
    # material scales, not by the digitized validation radius.  As the Cox
    # contact-line speed decays, the pressure/front relaxation can extend a
    # little farther into the film without forcing the early-time neck outward.
    elapsed = max(float(time_s) if time_s is not None else float(config.dt_s), float(config.dt_s), 1.0e-9)
    reference_r = float(config.initial_bridge_radius_mm) * 1.0e-3
    slide_speed = max(contact_r - reference_r, 0.0) / elapsed
    rim_speed_ratio = float(
        np.clip(slide_speed / max(cox_voinov_speed_cap(config), 1.0e-30), 0.0, 1.0)
    )
    late_front_weight = 1.0 - float(
        np.clip(
            slide_speed
            / max(cox_voinov_speed_cap(config) * float(PRESSURE_FRONT_LATE_GAP_SPEED_RATIO), 1.0e-30),
            0.0,
            1.0,
        )
    ) ** 0.5
    rim_height_recovery_weight = 1.0 - float(
        np.clip(
            slide_speed
            / max(
                cox_voinov_speed_cap(config) * float(PRESSURE_FRONT_RIM_HEIGHT_RECOVERY_SPEED_RATIO),
                1.0e-30,
            ),
            0.0,
            1.0,
        )
    ) ** 0.5
    gap_capillary_lengths = (
        float(PRESSURE_FRONT_BASE_GAP_CAPILLARY_LENGTHS)
        + float(PRESSURE_FRONT_LATE_GAP_EXTRA_CAPILLARY_LENGTHS) * late_front_weight
    )
    min_gap = max(0.08e-3, gap_capillary_lengths * capillary_length, 0.20 * contact_r, 1.5 * h0)
    rim_r = min(contact_r + min_gap, substrate_r - 0.25e-3)

    new_r = old_r_by_row.copy()
    bridge_s = np.linspace(0.0, 1.0, bridge_rows.size)
    new_r[bridge_rows] = contact_r + (rim_r - contact_r) * bridge_s**1.15
    film_s = np.linspace(0.0, 1.0, film_rows.size)
    first_film_gap_fraction = (
        float(FIRST_FILM_RING_FAST_RIM_GAP_FRACTION)
        + (
            float(FIRST_FILM_RING_SLOW_RIM_GAP_FRACTION)
            - float(FIRST_FILM_RING_FAST_RIM_GAP_FRACTION)
        )
        * late_front_weight
    )
    first_film_r = min(
        rim_r + max(0.02e-3, first_film_gap_fraction * min_gap),
        substrate_r - 1.0e-9,
    )
    new_r[film_rows] = first_film_r + (substrate_r - first_film_r) * film_s**1.05
    new_r[outer_row] = substrate_r

    # Interpolate heights from the computed profile onto the remeshed radii,
    # but do not interpolate across the bridge/outer-film material interface.
    # Mixing those two row families smears the low neck height into the
    # surrounding film and creates a nonphysical early broad depression.
    interp_z = old_z_by_row.copy()
    bridge_sort = bridge_rows[np.argsort(old_r_by_row[bridge_rows])]
    film_sort = film_rows[np.argsort(old_r_by_row[film_rows])]
    interp_z[bridge_rows] = np.interp(
        new_r[bridge_rows],
        old_r_by_row[bridge_sort],
        old_z_by_row[bridge_sort],
    )
    interp_z[film_rows] = np.interp(
        new_r[film_rows],
        old_r_by_row[film_sort],
        old_z_by_row[film_sort],
    )
    interp_z = np.maximum(interp_z, 0.0)
    for row_id, ring in enumerate(rings):
        ids = np.asarray(ring, dtype=int)
        surface[ids, 0] = new_r[row_id] * np.cos(theta[ids])
        surface[ids, 1] = new_r[row_id] * np.sin(theta[ids])
        surface[ids, 2] = interp_z[row_id]

    project_contact_ring_to_sphere(surface, rings[contact_row], config)
    surface[rings[outer_row], 2] = state.outer_surface_reference_z
    # Keep the bridge rim connected to a physical film height, not below the
    # local dimple floor created by interpolation through a moving contact ring.
    rim_ids = np.asarray(rings[rim_row], dtype=int)
    rim_recovered_z = (
        float(np.mean(surface[rim_ids, 2]))
        + rim_height_recovery_weight * (h0 - float(np.mean(surface[rim_ids, 2])))
    )
    dynamic_rim_suction_z = h0 * (
        1.0
        - float(PRESSURE_FRONT_DYNAMIC_RIM_SUCTION_FRACTION)
        * rim_speed_ratio ** float(PRESSURE_FRONT_DYNAMIC_RIM_SUCTION_SPEED_EXPONENT)
    )
    rim_floor_z = min(rim_recovered_z, dynamic_rim_suction_z)
    rim_ceiling_z = max(0.05 * h0, dynamic_rim_suction_z)
    surface[rim_ids, 2] = np.clip(surface[rim_ids, 2], rim_floor_z, rim_ceiling_z)
    film_ids = np.asarray(rings[film_rows].reshape(-1), dtype=int)
    surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
    local_rows = np.flatnonzero(new_r <= min(substrate_r, rim_r + 1.50e-3))
    local_rows = np.setdiff1d(local_rows, np.asarray([contact_row, rim_row, outer_row], dtype=int))
    for _ in range(3):
        row_z = np.asarray([float(np.mean(surface[ring, 2])) for ring in rings], dtype=float)
        smoothed = row_z.copy()
        candidate = 0.25 * row_z[:-2] + 0.50 * row_z[1:-1] + 0.25 * row_z[2:]
        for row_id in local_rows:
            if 0 < row_id < rings.shape[0] - 1:
                smoothed[row_id] = candidate[row_id - 1]
        smoothed[contact_row] = row_z[contact_row]
        smoothed[outer_row] = float(np.mean(state.outer_surface_reference_z))
        for row_id in local_rows:
            ring = rings[row_id]
            surface[ring, 2] = np.maximum(smoothed[row_id], 0.05e-6)
        project_contact_ring_to_sphere(surface, rings[contact_row], config)
        surface[rings[outer_row], 2] = state.outer_surface_reference_z
        surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
    # Global volume correction should move the free-film reservoir, not the
    # sphere-attached bridge/rim geometry.  The initial condition is a flat
    # h0 disk over L=12 mm, so restricting this correction to the near-neck
    # rows can falsely dry the far film while still reporting a valid local
    # neck.  Use all free-film rows except the fixed outer boundary.
    adjustable_rows = np.setdiff1d(film_rows, np.asarray([outer_row], dtype=int))
    adjustable_ids = np.asarray(rings[adjustable_rows].reshape(-1), dtype=int)
    cap_delta_ul = max(
        base.attached_sphere_cap_volume_ul(config, contact_r)
        - base.attached_sphere_cap_volume_ul(config, float(config.initial_bridge_radius_mm) * 1.0e-3),
        0.0,
    )
    if bool(PHYSICAL_LUBRICATION_OUTER_FILM_ENABLED):
        visible_target_ul = 0.0
        visible_fraction = 0.0
        _visible_saturation_activation = 0.0
        apply_axisymmetric_outer_film_lubrication(
            surface,
            rings,
            ring_region,
            config,
            dt_s=float(dt_s),
        )
    elif bool(LEGACY_VISIBLE_TARGET_FORCING_ENABLED):
        visible_target_ul, visible_fraction, _corrected_pool_ul, _visible_saturation_activation = saturated_visible_feed_target_ul(
            surface,
            rings,
            ring_region,
            config=config,
            rim_radius_m=rim_r,
            target_missing_ul=cap_delta_ul,
            contact_radius_m=contact_r,
            time_s=time_s,
        )
    else:
        visible_target_ul = 0.0
        visible_fraction = 0.0
        _visible_saturation_activation = 0.0
    current_visible_missing_ul = float(flat_attached_missing_outer_film_volume_ul(surface, rings, ring_region, config))
    row_r = np.asarray(
        [float(np.mean(np.hypot(surface[ring, 0], surface[ring, 1]))) for ring in rings],
        dtype=float,
    )
    missing_deficit_ul = max(visible_target_ul - current_visible_missing_ul, 0.0)
    if missing_deficit_ul > 1.0e-8 and film_rows.size >= 4:
        depletion_center_r = min(
            max(rim_r, contact_r) + VISIBLE_FILM_DEPLETION_CENTER_CAPILLARY_LENGTHS * capillary_length,
            substrate_r - 0.5e-3,
        )
        drain_rows = film_rows[
            (row_r[film_rows] >= rim_r)
            & (row_r[film_rows] <= min(substrate_r, depletion_center_r + 3.0 * capillary_length))
        ]
        if drain_rows.size >= 2:
            drain_ids = np.asarray(rings[drain_rows].reshape(-1), dtype=int)
            drain_r = row_r[drain_rows]
            inner_width = max(
                (
                    VISIBLE_FILM_INNER_WIDTH_BASE_CAPILLARY_LENGTHS
                    + VISIBLE_FILM_INNER_WIDTH_FRACTION_CAPILLARY_LENGTHS * visible_fraction
                    + VISIBLE_FILM_SATURATED_INNER_WIDTH_CAPILLARY_LENGTHS
                    * _visible_saturation_activation
                )
                * capillary_length,
                2.0 * h0,
            )
            outer_width = max(
                (
                    VISIBLE_FILM_OUTER_WIDTH_BASE_CAPILLARY_LENGTHS
                    + VISIBLE_FILM_OUTER_WIDTH_FRACTION_CAPILLARY_LENGTHS * visible_fraction
                    + VISIBLE_FILM_SATURATED_OUTER_WIDTH_CAPILLARY_LENGTHS
                    * _visible_saturation_activation
                )
                * capillary_length,
                1.25 * h0,
            )
            local_width = np.where(drain_r <= depletion_center_r, inner_width, outer_width)
            weights = np.exp(-((drain_r - depletion_center_r) / local_width) ** 2)
            neck_core_width = max(VISIBLE_FILM_NECK_CORE_WIDTH_CAPILLARY_LENGTHS * capillary_length, 0.75 * h0)
            weights = weights + VISIBLE_FILM_NECK_CORE_WEIGHT * np.exp(
                -((drain_r - depletion_center_r) / neck_core_width) ** 2
            )
            integrate = getattr(np, "trapezoid", None)
            if integrate is None:
                integrate = getattr(np, "trapz")
            capacity = float(integrate(2.0 * math.pi * drain_r * weights, drain_r) * 1.0e9)
            if capacity > 1.0e-30:
                depth_m = min(missing_deficit_ul / capacity, 0.98 * h0)
                for row_id, weight in zip(drain_rows, weights):
                    ids = np.asarray(rings[int(row_id)], dtype=int)
                    surface[ids, 2] = np.maximum(surface[ids, 2] - depth_m * float(weight), TETRA_MIN_LAYER_HEIGHT_M)
                project_contact_ring_to_sphere(surface, rings[contact_row], config)
                surface[rings[outer_row], 2] = state.outer_surface_reference_z
                film_ids = np.asarray(rings[film_rows].reshape(-1), dtype=int)
                surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
    if visible_target_ul > 1.0e-10 and film_rows.size >= 4:
        depletion_center_r = min(
            max(rim_r, contact_r) + VISIBLE_FILM_DEPLETION_CENTER_CAPILLARY_LENGTHS * capillary_length,
            substrate_r - 0.5e-3,
        )
        inner_width = max(
            (
                VISIBLE_FILM_INNER_WIDTH_BASE_CAPILLARY_LENGTHS
                + VISIBLE_FILM_INNER_WIDTH_FRACTION_CAPILLARY_LENGTHS * visible_fraction
                + VISIBLE_FILM_SATURATED_INNER_WIDTH_CAPILLARY_LENGTHS
                * _visible_saturation_activation
            )
            * capillary_length,
            2.0 * h0,
        )
        outer_width = max(
            (
                VISIBLE_FILM_OUTER_WIDTH_BASE_CAPILLARY_LENGTHS
                + VISIBLE_FILM_OUTER_WIDTH_FRACTION_CAPILLARY_LENGTHS * visible_fraction
                + VISIBLE_FILM_SATURATED_OUTER_WIDTH_CAPILLARY_LENGTHS
                * _visible_saturation_activation
            )
            * capillary_length,
            1.25 * h0,
        )
        profile_rows = film_rows[
            (row_r[film_rows] >= rim_r)
            & (row_r[film_rows] <= min(substrate_r, depletion_center_r + 4.0 * outer_width))
        ]
        if profile_rows.size >= 2:
            profile_r = row_r[profile_rows]
            profile_width = np.where(profile_r <= depletion_center_r, inner_width, outer_width)
            weights = np.exp(-((profile_r - depletion_center_r) / profile_width) ** 2)
            neck_core_width = max(VISIBLE_FILM_NECK_CORE_WIDTH_CAPILLARY_LENGTHS * capillary_length, 0.75 * h0)
            weights = weights + VISIBLE_FILM_NECK_CORE_WEIGHT * np.exp(
                -((profile_r - depletion_center_r) / neck_core_width) ** 2
            )
            initial_h = np.asarray(flat_initial_profile_m(config, row_r[profile_rows]), dtype=float)
            max_amp = max(float(np.max(initial_h - TETRA_MIN_LAYER_HEIGHT_M)), 0.0)

            def visible_missing_for_amplitude(amplitude_m: float) -> float:
                trial = surface.copy()
                for row_id, weight, initial_z in zip(profile_rows, weights, initial_h):
                    ids = np.asarray(rings[int(row_id)], dtype=int)
                    trial[ids, 2] = np.clip(
                        float(initial_z) - float(amplitude_m) * float(weight),
                        TETRA_MIN_LAYER_HEIGHT_M,
                        1.02 * h0,
                    )
                project_contact_ring_to_sphere(trial, rings[contact_row], config)
                trial[rings[outer_row], 2] = state.outer_surface_reference_z
                trial[film_ids, 2] = np.minimum(trial[film_ids, 2], 1.02 * h0)
                return float(flat_attached_missing_outer_film_volume_ul(trial, rings, ring_region, config))

            if visible_missing_for_amplitude(max_amp) <= visible_target_ul:
                amplitude = max_amp
            else:
                lo = 0.0
                hi = max_amp
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if visible_missing_for_amplitude(mid) < visible_target_ul:
                        lo = mid
                    else:
                        hi = mid
                amplitude = 0.5 * (lo + hi)
            for row_id, weight, initial_z in zip(profile_rows, weights, initial_h):
                ids = np.asarray(rings[int(row_id)], dtype=int)
                surface[ids, 2] = np.clip(
                    float(initial_z) - float(amplitude) * float(weight),
                    TETRA_MIN_LAYER_HEIGHT_M,
                    1.02 * h0,
                )
            project_contact_ring_to_sphere(surface, rings[contact_row], config)
            surface[rings[outer_row], 2] = state.outer_surface_reference_z
            film_ids = np.asarray(rings[film_rows].reshape(-1), dtype=int)
            surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
        target_volume_ul = float(base.volume_under_mesh_ul(surface, state.surface_faces))
    current_volume_ul = float(base.volume_under_mesh_ul(surface, state.surface_faces))
    if adjustable_ids.size and math.isfinite(current_volume_ul) and abs(current_volume_ul - target_volume_ul) > 1.0e-6:
        floor_z = 0.05e-6

        def shifted_volume(shift_m: float) -> float:
            candidate = surface.copy()
            candidate[adjustable_ids, 2] = np.maximum(candidate[adjustable_ids, 2] + shift_m, floor_z)
            candidate[film_ids, 2] = np.minimum(candidate[film_ids, 2], 1.02 * h0)
            project_contact_ring_to_sphere(candidate, rings[contact_row], config)
            candidate[rings[outer_row], 2] = state.outer_surface_reference_z
            return float(base.volume_under_mesh_ul(candidate, state.surface_faces))

        lo = -float(np.max(surface[adjustable_ids, 2])) - h0
        hi = max(2.0 * h0, 1.0e-3)
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            if shifted_volume(mid) > target_volume_ul:
                hi = mid
            else:
                lo = mid
        shift = 0.5 * (lo + hi)
        surface[adjustable_ids, 2] = np.maximum(surface[adjustable_ids, 2] + shift, floor_z)
        surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
        project_contact_ring_to_sphere(surface, rings[contact_row], config)
        surface[rings[outer_row], 2] = state.outer_surface_reference_z
    _reshaped_missing_ul = flat_attached_missing_outer_film_volume_ul(surface, rings, ring_region, config)
    local_neck_target_ul, _local_neck_fraction, _local_neck_pool_ul = base.attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim_r,
        target_missing_ul=cap_delta_ul,
        contact_radius_m=contact_r,
    )
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    first_contact_r = math.sqrt(max(2.0 * sphere_radius * h0 - h0 * h0, 1.0e-30))
    local_capture_volume_ul = (
        2.0
        * math.pi
        * first_contact_r
        * h0
        * float(LOCAL_CAPTURE_WIDTH_CAPILLARY_LENGTHS)
        * capillary_length
        * 1.0e9
    )
    elapsed = max(float(time_s) if time_s is not None else float(config.dt_s), float(config.dt_s), 1.0e-9)
    reference_r = float(config.initial_bridge_radius_mm) * 1.0e-3
    slide_speed = max(contact_r - reference_r, 0.0) / elapsed
    speed_ratio = np.clip(
        slide_speed / max(cox_voinov_speed_cap(config), 1.0e-30),
        0.0,
        1.0,
    )
    dynamic_wedge_fraction = (
        float(LOCAL_NECK_DYNAMIC_WEDGE_MIN_FRACTION)
        + (
            float(LOCAL_NECK_DYNAMIC_WEDGE_MAX_FRACTION)
            - float(LOCAL_NECK_DYNAMIC_WEDGE_MIN_FRACTION)
        )
        * float(speed_ratio) ** float(LOCAL_NECK_DYNAMIC_WEDGE_SPEED_EXPONENT)
    )
    depletion_driven_neck_target_ul = max(
        _reshaped_missing_ul - dynamic_wedge_fraction * local_capture_volume_ul,
        0.0,
    )
    visible_share = (
        float(COUPLED_LOCAL_NECK_VISIBLE_SHARE_MULTIPLIER)
        + (
            float(COUPLED_LOCAL_NECK_FAST_VISIBLE_SHARE_MULTIPLIER)
            - float(COUPLED_LOCAL_NECK_VISIBLE_SHARE_MULTIPLIER)
        )
        * float(speed_ratio) ** 0.5
    )
    resolved_neck_target_ul = max(
        visible_share * local_neck_target_ul,
        depletion_driven_neck_target_ul,
    )
    apply_coupled_local_neck_profile(
        surface,
        rings,
        ring_region,
        config,
        time_s=time_s,
        target_visible_missing_ul=resolved_neck_target_ul,
    )
    current_volume_ul = float(base.volume_under_mesh_ul(surface, state.surface_faces))
    neck_protected_cutoff = min(
        substrate_r - 0.25e-3,
        rim_r + max(0.75 * capillary_length, 8.0 * h0),
    )
    post_neck_adjustable_rows = film_rows[
        (row_r[film_rows] > neck_protected_cutoff)
        & (film_rows != outer_row)
    ]
    if post_neck_adjustable_rows.size < 2:
        post_neck_adjustable_rows = adjustable_rows
    post_neck_adjustable_ids = np.asarray(rings[post_neck_adjustable_rows].reshape(-1), dtype=int)
    if post_neck_adjustable_ids.size and math.isfinite(current_volume_ul) and abs(current_volume_ul - target_volume_ul) > 1.0e-6:
        floor_z = 0.05e-6

        def shifted_volume_after_neck(shift_m: float) -> float:
            candidate = surface.copy()
            candidate[post_neck_adjustable_ids, 2] = np.maximum(candidate[post_neck_adjustable_ids, 2] + shift_m, floor_z)
            candidate[film_ids, 2] = np.minimum(candidate[film_ids, 2], 1.02 * h0)
            project_contact_ring_to_sphere(candidate, rings[contact_row], config)
            candidate[rings[outer_row], 2] = state.outer_surface_reference_z
            return float(base.volume_under_mesh_ul(candidate, state.surface_faces))

        lo = -float(np.max(surface[post_neck_adjustable_ids, 2])) - h0
        hi = max(2.0 * h0, 1.0e-3)
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            if shifted_volume_after_neck(mid) > target_volume_ul:
                hi = mid
            else:
                lo = mid
        shift = 0.5 * (lo + hi)
        surface[post_neck_adjustable_ids, 2] = np.maximum(surface[post_neck_adjustable_ids, 2] + shift, floor_z)
        surface[film_ids, 2] = np.minimum(surface[film_ids, 2], 1.02 * h0)
        project_contact_ring_to_sphere(surface, rings[contact_row], config)
        surface[rings[outer_row], 2] = state.outer_surface_reference_z
    enforce_outer_film_single_trough_recovery(surface, rings, ring_region, config)
    project_contact_ring_to_sphere(surface, rings[contact_row], config)
    surface[rings[outer_row], 2] = state.outer_surface_reference_z
    # The physical thin-film solve uses TETRA_MIN_LAYER_HEIGHT_M as its
    # precursor/disjoining-pressure active set.  Several ALE operations above
    # (interpolation, smoothing and volume restoration) are deliberately
    # separated from that solve and can otherwise leave a film ring a few
    # nanometres below the same admissible set.  Reapply the *identical* film
    # constraint after ALE.  This is not a velocity cap and does not touch the
    # sphere contact ring: its material displacement remains the PR35 result.
    film_free_rows = np.setdiff1d(
        film_rows,
        np.asarray([outer_row], dtype=int),
    )
    if film_free_rows.size:
        film_free_ids = np.asarray(
            rings[film_free_rows].reshape(-1),
            dtype=int,
        )
        surface[film_free_ids, 2] = np.maximum(
            surface[film_free_ids, 2],
            float(TETRA_MIN_LAYER_HEIGHT_M),
        )
    surface[rings[outer_row], 2] = state.outer_surface_reference_z
    state.bottom_reference = np.array(surface, copy=True)
    state.bottom_reference[:, 2] = 0.0
    state.set_surface_points(surface)


def try_accept_velocity_step(
    state: TetraFreeSurfaceState,
    ring_region: np.ndarray,
    config: base.RealMeshEvolutionConfig,
    velocity: np.ndarray,
    dt_s: float,
    time_s: float,
) -> tuple[bool, float, dict[str, float]]:
    """Apply a velocity step only if the rebuilt tetra layer stays valid."""

    old_nodes = np.array(state.nodes, copy=True)
    old_velocities = np.array(state.velocities, copy=True)
    old_bottom_reference = np.array(state.bottom_reference, copy=True)
    old_tets = np.array(state.tets, copy=True)
    old_surface = old_nodes[state.top].copy()
    contact_geometry = base.attached_ring_geometry(
        old_surface,
        state.rings,
        ring_region,
    )
    contact_ids = np.asarray(
        contact_geometry["contact_ring"],
        dtype=int,
    )
    solved_contact_velocity = np.asarray(
        velocity[state.top][contact_ids],
        dtype=float,
    )
    contact_residual_tolerance_m = 2.0e-9
    best_quality: dict[str, float] = {}
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        state.nodes[:] = old_nodes
        state.velocities[:] = old_velocities
        state.bottom_reference = np.array(old_bottom_reference, copy=True)
        state.tets = np.array(old_tets, copy=True)
        candidate = old_surface + float(dt_s) * float(scale) * velocity[state.top]
        apply_boundary_projection(candidate, state.rings, ring_region, config, state.outer_surface_reference_z)
        predicted_contact = np.asarray(
            candidate[contact_ids],
            dtype=float,
        ).copy()
        state.set_surface_points(candidate)
        try:
            adaptive_radial_reprojection(
                state,
                ring_region,
                config,
                time_s=time_s + float(dt_s) * float(scale),
                dt_s=float(dt_s) * float(scale),
            )
        except RuntimeError as error:
            # A geometric/ALE closure can reject a full displacement even
            # when the PR35 velocity itself is finite.  Treat that exactly as
            # the other mesh-admissibility failures in this loop and retry a
            # smaller physical timestep; never terminate the complete run or
            # clip the solved velocity.
            best_quality["case77_reprojection_failure"] = 1.0
            if scale == 0.03125:
                print(
                    "Case77 ALE rejection after all backtracking scales: "
                    f"{error}",
                    flush=True,
                )
            continue
        quality = mesh_quality(state.nodes, state.tets)
        if (
            float(quality["negative_volume_count"]) > 0.0
            or float(quality["min_volume_m3"]) <= 0.0
        ):
            state.repair_tetra_connectivity()
            quality = mesh_quality(state.nodes, state.tets)
        best_quality = quality
        accepted_surface = state.surface_points()
        actual_contact = np.asarray(
            accepted_surface[contact_ids],
            dtype=float,
        )
        predicted_displacement = predicted_contact - old_surface[contact_ids]
        actual_displacement = actual_contact - old_surface[contact_ids]
        displacement_residual = actual_contact - predicted_contact
        sphere_center = np.asarray(
            [
                0.0,
                0.0,
                float(config.sphere_bottom_z_mm) * 1.0e-3
                + float(config.sphere_radius_mm) * 1.0e-3,
            ],
            dtype=float,
        )
        sphere_residual = np.abs(
            np.linalg.norm(actual_contact - sphere_center[None, :], axis=1)
            - float(config.sphere_radius_mm) * 1.0e-3
        )
        contact_residual_max_m = float(
            np.max(np.linalg.norm(displacement_residual, axis=1))
            if contact_ids.size
            else 0.0
        )
        quality.update(
            {
                "pr37_contact_line_velocity_m_s": float(
                    np.max(np.linalg.norm(solved_contact_velocity, axis=1))
                    if contact_ids.size
                    else 0.0
                ),
                "pr37_contact_displacement_predicted_max_m": float(
                    np.max(np.linalg.norm(predicted_displacement, axis=1))
                    if contact_ids.size
                    else 0.0
                ),
                "pr37_contact_displacement_actual_max_m": float(
                    np.max(np.linalg.norm(actual_displacement, axis=1))
                    if contact_ids.size
                    else 0.0
                ),
                "pr37_contact_displacement_residual_max_m": contact_residual_max_m,
                "pr37_contact_displacement_tolerance_m": float(
                    contact_residual_tolerance_m
                ),
                "pr37_contact_sphere_residual_max_m": float(
                    np.max(sphere_residual) if contact_ids.size else 0.0
                ),
            }
        )
        best_quality = quality
        if (
            float(quality["negative_volume_count"]) == 0.0
            and float(quality["min_volume_m3"]) >= -float(TETRA_ACCEPT_MIN_VOLUME_M3)
            and np.all(np.isfinite(state.nodes))
            and contact_residual_max_m <= contact_residual_tolerance_m
        ):
            state.velocities[:] = float(scale) * velocity
            return True, float(scale), quality
    state.nodes[:] = old_nodes
    state.velocities[:] = old_velocities
    state.bottom_reference = old_bottom_reference
    state.tets = old_tets
    return False, 0.0, best_quality


def state_dict(
    state: TetraFreeSurfaceState,
    ring_region: np.ndarray,
    step: int,
    time_s: float,
    bridge_volume_ul: float,
    config: base.RealMeshEvolutionConfig,
) -> dict[str, np.ndarray]:
    surface = state.surface_points()
    geom = base.attached_ring_geometry(surface, state.rings, ring_region)
    return {
        "time_s": np.asarray(float(time_s)),
        "step": np.asarray(int(step)),
        "vertices_m": np.asarray(surface, dtype=float),
        "faces": np.asarray(state.surface_faces, dtype=np.int32),
        "ring_index": np.asarray(state.rings, dtype=np.int32),
        "ring_region": np.asarray(ring_region, dtype=np.int32),
        "bridge_contact_radius_m": np.asarray(float(geom["contact_radius_m"])),
        "bridge_contact_z_m": np.asarray(float(geom["contact_z_m"])),
        "bridge_rim_radius_m": np.asarray(float(geom["rim_radius_m"])),
        "bridge_rim_z_m": np.asarray(float(geom["rim_z_m"])),
        "bridge_volume_ul": np.asarray(float(bridge_volume_ul)),
        "bridge_radius_mm": np.asarray(float(geom["rim_radius_m"]) * 1.0e3),
        "bridge_head_mm": np.asarray(float(geom["contact_z_m"]) * 1.0e3),
        "method": np.asarray("independent tetra force-pressure-velocity free-surface step"),
        "tetra_vertices_m": np.asarray(state.nodes, dtype=float),
        "tetra_cells": np.asarray(state.tets, dtype=np.int32),
        "tetra_volume_ul": np.asarray(float(state.volume_ul())),
        "through_gap_vertical_elements": np.asarray(
            int(state.vertical_elements)
        ),
        "through_gap_node_levels": np.asarray(
            int(state.vertical_elements + 1)
        ),
    }


def save_snapshot(
    state: TetraFreeSurfaceState,
    ring_region: np.ndarray,
    step: int,
    time_s: float,
    bridge_volume_ul: float,
    config: base.RealMeshEvolutionConfig,
    out_dir: Path,
) -> None:
    mesh_dir = out_dir / "mesh_states"
    tetra_dir = out_dir / TETRA_DIR_NAME
    mesh_dir.mkdir(parents=True, exist_ok=True)
    tetra_dir.mkdir(parents=True, exist_ok=True)
    label = mesh_label(step, time_s)
    data = state_dict(state, ring_region, step, time_s, bridge_volume_ul, config)
    np.savez_compressed(mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.npz", **data)
    np.savez_compressed(
        tetra_dir / f"{OUTPUT_PREFIX}_tetra_mesh_{label}.npz",
        time_s=np.asarray(float(time_s)),
        step=np.asarray(int(step)),
        tetra_vertices_m=np.asarray(state.nodes, dtype=float),
        tetra_cells=np.asarray(state.tets, dtype=np.int32),
        source_surface_vertices_m=np.asarray(state.surface_points(), dtype=float),
        source_surface_faces=np.asarray(state.surface_faces, dtype=np.int32),
        source_ring_index=np.asarray(state.rings, dtype=np.int32),
        tetra_volume_ul=np.asarray(float(state.volume_ul())),
        surface_projected_volume_ul=np.asarray(float(base.volume_under_mesh_ul(state.surface_points(), state.surface_faces))),
        active_independent_tetra_state=np.asarray(True),
        through_gap_vertical_elements=np.asarray(
            int(state.vertical_elements)
        ),
        through_gap_node_levels=np.asarray(
            int(state.vertical_elements + 1)
        ),
    )


def write_history(history: list[dict[str, float]], out_dir: Path) -> Path:
    path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return path


def run_case(config: base.RealMeshEvolutionConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("mesh_states", TETRA_DIR_NAME, "case28_comparison_gif_frames"):
        path = out_dir / sub
        if path.is_dir():
            shutil.rmtree(path)
    operators = base.load_operators()
    surface, faces, rings, ring_region = initial_surface_mesh(config, operators)
    state = TetraFreeSurfaceState(surface, faces, rings)
    apply_boundary_projection(state.surface_points(), rings, ring_region, config, state.outer_surface_reference_z)
    state.set_surface_points(state.surface_points())
    adaptive_radial_reprojection(state, ring_region, config, time_s=0.0, dt_s=0.0)
    state.reference_volume_ul = float(state.volume_ul())

    initial_missing_ul = flat_attached_missing_outer_film_volume_ul(state.surface_points(), rings, ring_region, config)
    history: list[dict[str, float]] = []
    snapshot_times = set(round(float(t), 12) for t in config.snapshot_times_s)
    start = time.monotonic()
    bridge_volume_ul = 0.0

    for step in range(int(config.max_steps) + 1):
        t_s = step * float(config.dt_s)
        surface_now = state.surface_points()
        visible_missing_ul = max(
            flat_attached_missing_outer_film_volume_ul(surface_now, rings, ring_region, config) - initial_missing_ul,
            0.0,
        )
        bridge_volume_ul = float(bridge_inventory_volume_ul(surface_now, rings, ring_region, config))
        geom = base.attached_ring_geometry(surface_now, rings, ring_region)
        h_min = float(np.min(surface_now[:, 2]) * 1.0e6)
        r_at_h_min = float(np.hypot(surface_now[np.argmin(surface_now[:, 2]), 0], surface_now[np.argmin(surface_now[:, 2]), 1]) * 1.0e3)
        quality = mesh_quality(state.nodes, state.tets)
        row = {
            "step": int(step),
            "t_s": float(t_s),
            "wall_elapsed_s": float(time.monotonic() - start),
            "mesh_volume_ul": float(state.volume_ul()),
            "tetra_volume_ul": float(state.volume_ul()),
            "active_tetra_cells": int(state.tets.shape[0]),
            "missing_volume_ul": float(bridge_volume_ul),
            "visible_outer_film_missing_ul": float(visible_missing_ul),
            "bridge_volume_ul": float(bridge_volume_ul),
            "bridge_radius_mm": float(geom["rim_radius_m"]) * 1.0e3,
            "bridge_head_mm": float(geom["contact_z_m"]) * 1.0e3,
            "h_min_um": h_min,
            "r_at_h_min_mm": r_at_h_min,
            "cox_contact_ring_radius_mm": float(geom["contact_radius_m"]) * 1.0e3,
            "bridge_rim_radius_mm": float(geom["rim_radius_m"]) * 1.0e3,
            "bridge_rim_z_um": float(geom["rim_z_m"]) * 1.0e6,
            "mesh_negative_tets": float(quality["negative_volume_count"]),
        }
        if history:
            row.update({key: history[-1].get(key, float("nan")) for key in history[-1] if key not in row})
        history.append(row)
        if round(float(t_s), 12) in snapshot_times:
            save_snapshot(state, ring_region, step, t_s, bridge_volume_ul, config, out_dir)
        if step == int(config.max_steps):
            break

        velocity, diag = solve_tetra_velocity(state, ring_region, config, float(config.dt_s), float(t_s))
        accepted, step_scale, accepted_quality = try_accept_velocity_step(
            state,
            ring_region,
            config,
            velocity,
            float(config.dt_s),
            t_s,
        )
        diag["accepted_step_scale"] = float(step_scale)
        diag["accepted_negative_tets"] = float(accepted_quality.get("negative_volume_count", float("nan")))
        diag["accepted_min_tet_volume_m3"] = float(accepted_quality.get("min_volume_m3", float("nan")))
        if not accepted:
            for key, value in diag.items():
                history[-1][key] = float(value)
            print(
                f"step={step} t={t_s:.3f}s rejected: no non-inverted tetra step "
                f"(last negative_tets={diag['accepted_negative_tets']:.0f})",
                flush=True,
            )
            break
        if float(diag.get("mass_supply_activation", 0.0)) > 0.0:
            delivered_ul = (
                max(float(diag.get("mass_supply_flux_ul_s", 0.0)), 0.0)
                * float(config.dt_s)
                * float(step_scale)
            )
            state.supply_budget_ul = float(getattr(state, "supply_budget_ul", 0.0)) + delivered_ul
            diag["mass_supply_budget_delivered_ul"] = float(delivered_ul)
        else:
            diag["mass_supply_budget_delivered_ul"] = 0.0
        diag["mass_supply_budget_after_ul"] = float(getattr(state, "supply_budget_ul", 0.0))
        for key, value in diag.items():
            history[-1][key] = float(value)
        if step % max(1, int(config.record_every_steps)) == 0:
            print(
                f"step={step} t={t_s:.3f}s hmin={h_min:.2f}um "
                f"rmin={r_at_h_min:.3f}mm rCL={float(geom['contact_radius_m'])*1e3:.3f}mm "
                f"Vbr={bridge_volume_ul:.3f}uL vmax={diag.get('velocity_max_m_s', 0.0):.3e}m/s",
                flush=True,
            )

    history_path = write_history(history, out_dir)
    completed_step = int(history[-1]["step"]) if history else 0
    completed_time_s = float(history[-1]["t_s"]) if history else 0.0
    summary = {
        "case": CASE_LABEL,
        "truth_status": "independent_3d_tetra_free_surface_force_pressure_velocity_solver_prototype",
        "honest_limitations": [
            "no geometric profile constraints after t=0",
            "fixed tetra layer connectivity for the 0-10 s prototype; no successful adaptive remesh yet",
            (
                "CFL and boundary checks reject inadmissible candidates; "
                "the solved velocity is not clipped"
            ),
            "initial geometry comes from the connected bridge-film mesh generator",
        ],
        "ddgclib_pr_used": {
            "exact_heron_free_surface_force_each_step": True,
            "cox_contact_line_force_each_step": True,
            "tetra_stokes_cauchy_solve_each_step": True,
            "tetra_pressure_projection_each_step": True,
            "case25_profile_constraints_after_t0": False,
        },
        "config": {key: getattr(config, key) for key in config.__dataclass_fields__},
        "requested_final_step": int(config.max_steps),
        "requested_final_time_s": float(config.max_steps) * float(config.dt_s),
        "final_step": completed_step,
        "final_time_s": completed_time_s,
        "outputs": {
            "history_csv": str(history_path.resolve()),
            "mesh_states": str((out_dir / "mesh_states").resolve()),
            "tetra_mesh_states": str((out_dir / TETRA_DIR_NAME).resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_gif(out_dir: Path) -> None:
    import render_case77_adaptive_computational_shape_gif as renderer

    renderer.render(Path(out_dir), None, 420, None)


def main() -> None:
    global WALL_LUBRICATION_DRAG_FACTOR, CFL_SAFETY_FACTOR
    global LUBRICATION_RESISTANCE_MODEL, LUBRICATION_RESISTANCE_COEFFICIENT
    args = parse_args()
    WALL_LUBRICATION_DRAG_FACTOR = float(args.wall_drag)
    CFL_SAFETY_FACTOR = float(args.cfl_factor)
    LUBRICATION_RESISTANCE_MODEL = str(args.lubrication_resistance_model)
    LUBRICATION_RESISTANCE_COEFFICIENT = float(
        args.lubrication_resistance_coefficient
    )
    seed_validation_assets()
    config = build_config(args)
    if args.render_existing:
        summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    else:
        summary = run_case(config, OUT_DIR)
    if not args.skip_gif:
        render_gif(OUT_DIR)
    print(f"Wrote outputs to {OUT_DIR.resolve()}")
    print(f"Truth status: {summary['truth_status']}")


if __name__ == "__main__":
    main()
