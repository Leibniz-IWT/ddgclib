"""Axisymmetric bridge-film coupling operators.

This module is intentionally case-agnostic.  It provides a reusable
one-sphere/flat-film moving-boundary thin-film model that can be driven by
geometry, material parameters, and an optional quasi-static bridge table.  The
case scripts are responsible for selecting validation data and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize
from scipy.special import i0, k0


@dataclass(frozen=True)
class AxisymmetricBridgeFilmConfig:
    """Parameters for a reusable one-sphere/flat-film bridge solver.

    All primary quantities are SI.  The ``*_mm`` and ``*_um`` names are kept
    only for case-input convenience and explicit unit visibility.
    """

    sphere_radius_mm: float = 5.0
    substrate_radius_mm: float = 12.0
    initial_film_thickness_um: float = 100.0
    surface_tension_n_m: float = 0.021
    viscosity_pa_s: float = 0.10
    density_kg_m3: float = 1065.0
    gravity_m_s2: float = 9.80665
    t_end_s: float = 3500.0
    dt_s: float = 0.1
    snapshot_times_s: tuple[float, ...] = (10.0, 100.0, 3500.0)
    diagnostic_times_s: tuple[float, ...] = (
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        2000.0,
        3500.0,
    )
    grid_nodes: int = 180
    moving_grid_stretch: float = 3.0
    sphere_tip_at_initial_film: bool = False

    bridge_table_volume_ul: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.20,
        0.50,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        55.0,
    )
    bridge_table_radius_mm: tuple[float, ...] = (
        1.00,
        1.12,
        1.23,
        1.38,
        1.65,
        1.92,
        2.28,
        2.82,
        3.35,
        4.12,
        5.02,
        5.48,
    )
    bridge_table_head_mm: tuple[float, ...] = (
        9.80,
        8.00,
        6.35,
        4.90,
        3.55,
        2.75,
        2.08,
        1.45,
        1.02,
        0.68,
        0.35,
        0.22,
    )
    bridge_table_head_scale: float = 0.85
    initial_bridge_volume_ul: float = 1.75
    initial_bridge_radius_mm: float = 1.00
    bridge_volume_shape_factor: float = 1.15
    bridge_radius_speed_limit_mm_s: float = 5.0

    bridge_pressure_activation_time_s: float = 25.0
    bridge_pressure_activation_exponent: float = 1.0
    bridge_pressure_relax_time_s: float = 120.0
    bridge_pressure_relax_exponent: float = 0.40
    bridge_rim_pressure_multiplier: float = 0.0
    bridge_rim_pressure_width_mm: float = 0.06
    bridge_rim_pressure_offset_mm: float = 0.06

    bridge_inflow_bottleneck_enabled: bool = True
    bridge_inflow_bottleneck_volume_ul: float = 0.020
    bridge_inflow_bottleneck_exponent: float = 2.0
    bridge_inflow_bottleneck_floor: float = 0.0
    local_capture_enabled: bool = False
    local_capture_width_capillary_lengths: float = 0.0
    local_capture_time_s: float = 10.0
    local_capture_exponent: float = 1.0
    local_capture_release_time_s: float = 4000.0
    local_capture_release_exponent: float = 1.0
    local_capture_release_saturation_enabled: bool = False
    local_capture_release_delay_s: float = 0.0
    local_capture_release_extra_width_capillary_lengths: float = 0.0
    moving_boundary_advection_multiplier: float = 0.70
    bridge_boundary_flux_correction: bool = False
    algebraic_volume_projection: bool = False
    no_flux_outer_edge: bool = True
    film_mobility_average: str = "harmonic"
    inner_pressure_boundary_height_enabled: bool = False
    inner_pressure_boundary_samples: int = 96

    finite_inner_reconstruction_enabled: bool = True
    finite_inner_width_mm: float = 0.036
    quasistatic_dimple_closure_enabled: bool = True
    dimple_min_height_inf_um: float = 14.2
    dimple_min_height_tau_s: float = 15.3
    dimple_min_height_exponent: float = 1.108
    dimple_min_radius_initial_mm: float = 3.20858
    dimple_min_radius_inf_mm: float = 4.03627
    dimple_min_radius_tau_s: float = 652.006
    dimple_min_radius_exponent: float = 0.67855
    dimple_recovery_width_base_mm: float = 0.35
    dimple_recovery_width_growth_mm: float = 2.85
    dimple_recovery_width_tau_s: float = 1700.0
    dimple_recovery_width_exponent: float = 0.72
    dimple_recovery_power_early: float = 2.0
    dimple_recovery_power_middle: float = 2.0
    dimple_recovery_power_late: float = 0.90
    dimple_recovery_middle_time_s: float = 50.0
    dimple_recovery_late_time_s: float = 1000.0

    min_film_thickness_m: float = 0.25e-6
    t_start_s: float = 0.1
    ode_rtol: float = 4.0e-4
    ode_atol: float = 1.0e-8
    ode_max_step_s: float = 0.8


def _capillary_length_m(config: AxisymmetricBridgeFilmConfig) -> float:
    return math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-300)
    )


def attached_bridge_state_closure(
    *,
    initial_film_thickness_m: float,
    capillary_length_m: float,
    intrinsic_contact_radius_m: float,
    bridge_radius_m: float,
    bridge_volume_ul: float,
    initial_bridge_volume_ul: float = 0.0,
    rim_base_contact_radius_fraction: float = 1.05,
    rim_transient_contact_radius_fraction: float = 0.94,
    rim_decay_progress: float = 0.35,
    rim_decay_exponent: float = 1.0,
    feed_start_annulus_fraction: float = 0.26,
    feed_capacity_annulus_fraction: float = 0.90,
    feed_growth_progress: float = 0.55,
    feed_growth_exponent: float = 0.80,
    feed_max_annulus_fraction: float = 0.93,
    pressure_start_annulus_fraction: float = 0.25,
    pressure_growth_progress: float = 0.40,
    pressure_growth_exponent: float = 1.0,
    pressure_boost_factor: float = 1.35,
    pressure_front_offset_factor: float = 160.0,
    pressure_front_offset_h0_lc_exponent: float = 3.0,
    pressure_front_width_factor: float = 18.0,
    pressure_front_width_h0_lc_exponent: float = 2.0,
) -> dict[str, float]:
    """State-based attached bridge-film closure for one-sphere thin films.

    The closure is intentionally expressed in dimensionless ratios of local
    geometric scales instead of validation-specific times.  The natural
    feed-volume scale is the annulus ``2*pi*a*h0*ell_c`` around the current
    bridge radius ``a``; rim offsets are scaled by the intrinsic first-contact
    radius ``sqrt(2 R h0 - h0^2)`` supplied by the caller.
    """

    h0 = max(float(initial_film_thickness_m), 1.0e-30)
    ell_c = max(float(capillary_length_m), 1.0e-30)
    contact_r = max(float(intrinsic_contact_radius_m), 1.0e-12)
    bridge_r = max(float(bridge_radius_m), contact_r, 1.0e-12)
    annulus_ul = max(2.0 * math.pi * bridge_r * h0 * ell_c * 1.0e9, 1.0e-30)
    grown_ul = max(float(bridge_volume_ul) - float(initial_bridge_volume_ul), 0.0)
    progress = grown_ul / annulus_ul

    rim_decay = math.exp(
        -(
            progress / max(float(rim_decay_progress), 1.0e-12)
        )
        ** max(float(rim_decay_exponent), 1.0e-12)
    )
    rim_offset_m = contact_r * (
        float(rim_base_contact_radius_fraction)
        + float(rim_transient_contact_radius_fraction) * rim_decay
    )

    feed_activation = 1.0 - math.exp(
        -(
            max(progress - float(feed_start_annulus_fraction), 0.0)
            / max(float(feed_growth_progress), 1.0e-12)
        )
        ** max(float(feed_growth_exponent), 1.0e-12)
    )
    allowed_missing_ul = annulus_ul * (
        float(feed_start_annulus_fraction)
        + float(feed_capacity_annulus_fraction) * feed_activation
    )
    allowed_missing_ul = min(
        allowed_missing_ul,
        annulus_ul * max(float(feed_max_annulus_fraction), 0.0),
    )

    pressure_activation = 1.0 - math.exp(
        -(
            max(progress - float(pressure_start_annulus_fraction), 0.0)
            / max(float(pressure_growth_progress), 1.0e-12)
        )
        ** max(float(pressure_growth_exponent), 1.0e-12)
    )
    pressure_gain = 1.0 + float(pressure_boost_factor) * pressure_activation
    thickness_ratio = h0 / ell_c
    pressure_center_offset_m = (
        ell_c
        * float(pressure_front_offset_factor)
        * thickness_ratio ** max(float(pressure_front_offset_h0_lc_exponent), 0.0)
        * pressure_activation
    )
    pressure_width_m = (
        ell_c
        * float(pressure_front_width_factor)
        * thickness_ratio ** max(float(pressure_front_width_h0_lc_exponent), 0.0)
        * pressure_activation
    )

    return {
        "annular_feed_scale_ul": float(annulus_ul),
        "growth_progress": float(progress),
        "rim_offset_m": float(rim_offset_m),
        "allowed_missing_ul": float(allowed_missing_ul),
        "pressure_gain": float(pressure_gain),
        "pressure_center_offset_m": float(pressure_center_offset_m),
        "pressure_width_m": float(pressure_width_m),
    }


def initial_bessel_film_profile_m(
    config: AxisymmetricBridgeFilmConfig,
    r_m: np.ndarray | float,
) -> np.ndarray:
    """Static initial film on a circular substrate before bridge growth."""

    radius = float(config.substrate_radius_mm) * 1.0e-3
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    capillary_length = _capillary_length_m(config)
    i0_edge = float(i0(radius / capillary_length))
    profile = h0 * (i0_edge - i0(np.asarray(r_m, dtype=float) / capillary_length)) / max(i0_edge - 1.0, 1.0e-300)
    return np.clip(profile, 0.0, h0)


def intrinsic_sphere_film_radius_m(config: AxisymmetricBridgeFilmConfig) -> float:
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    radius = float(config.sphere_radius_mm) * 1.0e-3
    return math.sqrt(max(2.0 * radius * h0 - h0 * h0, 0.0))


def minimum_bridge_radius_m(config: AxisymmetricBridgeFilmConfig) -> float:
    """Geometric lower footprint bound for the selected contact setup."""

    if bool(config.sphere_tip_at_initial_film):
        return 0.0
    return intrinsic_sphere_film_radius_m(config)


def bridge_table_arrays(
    config: AxisymmetricBridgeFilmConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume_ul = np.asarray(config.bridge_table_volume_ul, dtype=float)
    radius_mm = np.asarray(config.bridge_table_radius_mm, dtype=float)
    head_mm = np.asarray(config.bridge_table_head_mm, dtype=float)
    if not (volume_ul.size == radius_mm.size == head_mm.size) or volume_ul.size < 4:
        return (np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float))
    order = np.argsort(volume_ul)
    volume_ul = volume_ul[order]
    radius_mm = radius_mm[order]
    head_mm = head_mm[order]
    keep = np.concatenate(([True], np.diff(volume_ul) > 0.0))
    return volume_ul[keep], radius_mm[keep], head_mm[keep]


def bridge_radius_from_volume_m(
    config: AxisymmetricBridgeFilmConfig,
    volume_m3: float,
) -> float:
    volume_ul, radius_mm, _ = bridge_table_arrays(config)
    if volume_ul.size >= 4:
        value_ul = float(np.clip(float(volume_m3) * 1.0e9, volume_ul[0], volume_ul[-1]))
        radius = float(PchipInterpolator(volume_ul, radius_mm, extrapolate=True)(value_ul)) * 1.0e-3
        contact = minimum_bridge_radius_m(config)
        return min(max(radius, contact * 1.001), float(config.substrate_radius_mm) * 1.0e-3 * 0.985)

    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    shape_factor = max(float(config.bridge_volume_shape_factor), 1.0e-12)
    contact = minimum_bridge_radius_m(config)
    from_volume = math.sqrt(max(float(volume_m3), 0.0) / max(shape_factor * math.pi * h0, 1.0e-30))
    return min(max(contact, math.sqrt(contact * contact + from_volume * from_volume)), float(config.substrate_radius_mm) * 1.0e-3 * 0.985)


def bridge_volume_from_radius_m3(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
) -> float:
    volume_ul, radius_mm, _ = bridge_table_arrays(config)
    if volume_ul.size >= 4:
        radius_value_mm = float(np.clip(float(bridge_radius_m) * 1.0e3, radius_mm[0], radius_mm[-1]))
        return max(float(PchipInterpolator(radius_mm, volume_ul, extrapolate=True)(radius_value_mm)), 0.0) * 1.0e-9

    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    contact = minimum_bridge_radius_m(config)
    a = max(float(bridge_radius_m), contact * 1.001)
    shape_factor = max(float(config.bridge_volume_shape_factor), 1.0e-9)
    return shape_factor * math.pi * h0 * max(a * a - contact * contact, 0.0)


def sphere_lower_surface_z_m(
    *,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    r_m: np.ndarray | float,
) -> np.ndarray | float:
    """Lower surface of a sphere whose bottom point is at ``sphere_bottom_z_m``."""

    radius = float(sphere_radius_m)
    r = np.asarray(r_m, dtype=float)
    center_z = float(sphere_bottom_z_m) + radius
    z = center_z - np.sqrt(np.maximum(radius * radius - r * r, 0.0))
    if np.ndim(r_m) == 0:
        return float(z)
    return z


def sphere_lower_cap_volume_m3(
    *,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    contact_radius_m: float,
) -> float:
    """Volume below the lower spherical surface from the axis to contact radius."""

    radius = float(sphere_radius_m)
    contact = float(np.clip(contact_radius_m, 0.0, radius * 0.999999))
    center_z = float(sphere_bottom_z_m) + radius
    root = math.sqrt(max(radius * radius - contact * contact, 0.0))
    return float(
        math.pi * center_z * contact * contact
        - (2.0 * math.pi / 3.0) * (radius**3 - root**3)
    )


def axisymmetric_profile_volume_m3(r_m: np.ndarray, z_m: np.ndarray) -> float:
    """Volume under an axisymmetric free-surface profile over the substrate."""

    r = np.asarray(r_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if r.size < 2:
        return 0.0
    return float(2.0 * math.pi * np.trapezoid(r * z, r))


def axisymmetric_profile_area_m2(r_m: np.ndarray, z_m: np.ndarray) -> float:
    """Area of an axisymmetric free-surface profile."""

    r = np.asarray(r_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if r.size < 2:
        return 0.0
    dr = np.diff(r)
    dz = np.diff(z)
    r_mid = 0.5 * (r[:-1] + r[1:])
    return float(np.sum(2.0 * math.pi * r_mid * np.sqrt(dr * dr + dz * dz)))


def linearized_young_laplace_bridge_profile(
    *,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    contact_radius_m: float,
    rim_radius_m: float,
    rim_z_m: float,
    bridge_volume_m3: float,
    surface_tension_n_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    nodes: int = 96,
    min_z_m: float = 0.25e-6,
    enforce_positive_profile: bool = True,
) -> dict[str, np.ndarray | float | bool | str]:
    """Linearized Young-Laplace bridge profile with volume-selected pressure.

    This profile solves the small-slope axisymmetric Young-Laplace equation

    ``h'' + h'/r - h/l_c^2 = -C/l_c^2``

    between the sphere contact ring and the film/bridge rim.  The pressure
    offset ``C`` is selected from the bridge volume.  If the requested volume
    would require the liquid-air surface to pass below ``min_z_m``, the solver
    raises the geometric bridge volume to the smallest value that keeps the
    profile non-negative.  The added amount is returned explicitly as
    ``positive_volume_offset_m3`` so callers can audit the hidden/saturated
    bridge volume instead of creating a hard floor in the profile.
    """

    sphere_radius = float(sphere_radius_m)
    contact_radius = float(np.clip(contact_radius_m, 1.0e-12, sphere_radius * 0.999999))
    rim_radius = max(float(rim_radius_m), contact_radius + 1.0e-9)
    node_count = max(4, int(nodes))
    r = np.linspace(contact_radius, rim_radius, node_count)
    rim_z = max(float(rim_z_m), float(min_z_m))
    contact_z = float(
        sphere_lower_surface_z_m(
            sphere_radius_m=sphere_radius,
            sphere_bottom_z_m=float(sphere_bottom_z_m),
            r_m=contact_radius,
        )
    )
    sphere_cap = sphere_lower_cap_volume_m3(
        sphere_radius_m=sphere_radius,
        sphere_bottom_z_m=float(sphere_bottom_z_m),
        contact_radius_m=contact_radius,
    )
    target_annulus = max(float(bridge_volume_m3) - sphere_cap, 0.0)
    gamma = max(float(surface_tension_n_m), 1.0e-30)
    rho_g = max(float(density_kg_m3) * float(gravity_m_s2), 1.0e-300)
    capillary_length = math.sqrt(gamma / rho_g)

    x0 = contact_radius / capillary_length
    x1 = rim_radius / capillary_length
    matrix = np.asarray([[i0(x0), k0(x0)], [i0(x1), k0(x1)]], dtype=float)

    def profile_for_offset(offset_m: float) -> np.ndarray:
        rhs = np.asarray([contact_z - offset_m, rim_z - offset_m], dtype=float)
        try:
            coeff_i, coeff_k = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            s = np.linspace(0.0, 1.0, node_count)
            return contact_z + (rim_z - contact_z) * s
        return offset_m + coeff_i * i0(r / capillary_length) + coeff_k * k0(r / capillary_length)

    def annulus_volume_for_offset(offset_m: float) -> float:
        return axisymmetric_profile_volume_m3(r, profile_for_offset(offset_m))

    def bracket_root(fn, center: float = 0.0) -> tuple[float, float] | None:
        span = max(abs(float(center)), capillary_length, 1.0e-4)
        lo = float(center) - span
        hi = float(center) + span
        f_lo = fn(lo)
        f_hi = fn(hi)
        for _ in range(80):
            if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                return lo, hi
            span *= 2.0
            lo = float(center) - span
            hi = float(center) + span
            f_lo = fn(lo)
            f_hi = fn(hi)
        return None

    clamped = False
    message = "linearized Young-Laplace bridge profile"
    volume_bracket = bracket_root(lambda c: annulus_volume_for_offset(c) - target_annulus)
    if volume_bracket is None:
        pressure_offset = 0.0
        clamped = True
        message = "linearized Young-Laplace bridge profile; volume bracket failed"
    else:
        pressure_offset = float(brentq(lambda c: annulus_volume_for_offset(c) - target_annulus, *volume_bracket))

    z = profile_for_offset(pressure_offset)
    requested_total = sphere_cap + axisymmetric_profile_volume_m3(r, z)
    positive_offset = 0.0
    if bool(enforce_positive_profile) and float(np.min(z)) < float(min_z_m):
        floor_bracket = bracket_root(lambda c: float(np.min(profile_for_offset(c))) - float(min_z_m), pressure_offset)
        if floor_bracket is not None:
            pressure_offset = float(brentq(lambda c: float(np.min(profile_for_offset(c))) - float(min_z_m), *floor_bracket))
            z = profile_for_offset(pressure_offset)
            clamped = True
            message = "linearized Young-Laplace bridge profile; positive-volume floor applied"
        else:
            z = np.maximum(z, float(min_z_m))
            clamped = True
            message = "linearized Young-Laplace bridge profile; pointwise positivity fallback"
        actual_total = sphere_cap + axisymmetric_profile_volume_m3(r, z)
        positive_offset = max(actual_total - float(bridge_volume_m3), 0.0)
    else:
        actual_total = requested_total

    z = np.maximum(z, float(min_z_m))
    z[0] = contact_z
    z[-1] = rim_z
    annulus_volume = axisymmetric_profile_volume_m3(r, z)
    total_volume = sphere_cap + annulus_volume
    return {
        "r_m": r,
        "z_m": z,
        "sphere_cap_volume_m3": float(sphere_cap),
        "annulus_volume_m3": float(annulus_volume),
        "total_volume_m3": float(total_volume),
        "target_total_volume_m3": float(bridge_volume_m3),
        "target_annulus_volume_m3": float(target_annulus),
        "positive_volume_offset_m3": float(max(total_volume - float(bridge_volume_m3), positive_offset, 0.0)),
        "pressure_offset_m": float(pressure_offset),
        "area_m2": float(axisymmetric_profile_area_m2(r, z)),
        "success": True,
        "volume_clamped": bool(clamped),
        "message": message,
    }


def _softplus_lower_bound(y_m: np.ndarray | float, min_z_m: float, length_m: float) -> np.ndarray:
    """Smooth near-wall positive-height map with no flat clipping interval."""

    length = max(float(length_m), 1.0e-12)
    y = np.asarray(y_m, dtype=float)
    return float(min_z_m) + length * np.logaddexp(0.0, (y - float(min_z_m)) / length)


def _inverse_softplus_lower_bound(z_m: float, min_z_m: float, length_m: float) -> float:
    """Inverse of _softplus_lower_bound for scalar boundary values."""

    length = max(float(length_m), 1.0e-12)
    x = (float(z_m) - float(min_z_m)) / length
    if x > 50.0:
        return float(z_m)
    return float(min_z_m) + length * math.log(max(math.expm1(max(x, 1.0e-12)), 1.0e-300))


def soft_repulsive_young_laplace_bridge_profile(
    *,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    contact_radius_m: float,
    rim_radius_m: float,
    rim_z_m: float,
    bridge_volume_m3: float,
    surface_tension_n_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    nodes: int = 96,
    min_z_m: float = 0.25e-6,
    repulsion_length_m: float = 8.0e-6,
) -> dict[str, np.ndarray | float | bool | str]:
    """Linearized Young-Laplace profile with a smooth near-wall repulsion.

    The pressure offset is still selected by the requested bridge volume, but
    the lower-height constraint is applied through a soft positive map instead
    of pointwise clipping.  This avoids creating an artificial flat-bottom
    bridge shelf while preserving a strictly positive liquid height.
    """

    sphere_radius = float(sphere_radius_m)
    contact_radius = float(np.clip(contact_radius_m, 1.0e-12, sphere_radius * 0.999999))
    rim_radius = max(float(rim_radius_m), contact_radius + 1.0e-9)
    node_count = max(4, int(nodes))
    r = np.linspace(contact_radius, rim_radius, node_count)
    rim_z = max(float(rim_z_m), float(min_z_m))
    contact_z = float(
        sphere_lower_surface_z_m(
            sphere_radius_m=sphere_radius,
            sphere_bottom_z_m=float(sphere_bottom_z_m),
            r_m=contact_radius,
        )
    )
    sphere_cap = sphere_lower_cap_volume_m3(
        sphere_radius_m=sphere_radius,
        sphere_bottom_z_m=float(sphere_bottom_z_m),
        contact_radius_m=contact_radius,
    )
    target_annulus = max(float(bridge_volume_m3) - sphere_cap, 0.0)
    gamma = max(float(surface_tension_n_m), 1.0e-30)
    rho_g = max(float(density_kg_m3) * float(gravity_m_s2), 1.0e-300)
    capillary_length = math.sqrt(gamma / rho_g)
    repulsion_length = max(float(repulsion_length_m), 1.0e-12)

    x0 = contact_radius / capillary_length
    x1 = rim_radius / capillary_length
    matrix = np.asarray([[i0(x0), k0(x0)], [i0(x1), k0(x1)]], dtype=float)
    contact_raw = _inverse_softplus_lower_bound(contact_z, float(min_z_m), repulsion_length)
    rim_raw = _inverse_softplus_lower_bound(rim_z, float(min_z_m), repulsion_length)

    def raw_profile_for_offset(offset_m: float) -> np.ndarray:
        rhs = np.asarray([contact_raw - offset_m, rim_raw - offset_m], dtype=float)
        try:
            coeff_i, coeff_k = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            s = np.linspace(0.0, 1.0, node_count)
            return contact_raw + (rim_raw - contact_raw) * s
        return offset_m + coeff_i * i0(r / capillary_length) + coeff_k * k0(r / capillary_length)

    def profile_for_offset(offset_m: float) -> np.ndarray:
        z = _softplus_lower_bound(raw_profile_for_offset(offset_m), float(min_z_m), repulsion_length)
        z = np.asarray(z, dtype=float)
        z[0] = contact_z
        z[-1] = rim_z
        return z

    def annulus_volume_for_offset(offset_m: float) -> float:
        return axisymmetric_profile_volume_m3(r, profile_for_offset(offset_m))

    def bracket_root(fn, center: float = 0.0) -> tuple[float, float] | None:
        span = max(abs(float(center)), capillary_length, repulsion_length, 1.0e-4)
        lo = float(center) - span
        hi = float(center) + span
        f_lo = fn(lo)
        f_hi = fn(hi)
        for _ in range(80):
            if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                return lo, hi
            span *= 2.0
            lo = float(center) - span
            hi = float(center) + span
            f_lo = fn(lo)
            f_hi = fn(hi)
        return None

    bracket = bracket_root(lambda c: annulus_volume_for_offset(c) - target_annulus)
    success = bracket is not None
    if success:
        pressure_offset = float(brentq(lambda c: annulus_volume_for_offset(c) - target_annulus, *bracket))
    else:
        # Pick the least-volume-error pressure if the smooth positivity map
        # makes the requested volume geometrically unreachable.
        def objective(c_arr: np.ndarray) -> float:
            c = float(np.asarray(c_arr).ravel()[0])
            err = annulus_volume_for_offset(c) - target_annulus
            return float(err * err)

        opt = minimize(objective, np.asarray([0.0]), method="Nelder-Mead", options={"maxiter": 120})
        pressure_offset = float(np.asarray(opt.x).ravel()[0]) if opt.success else 0.0

    z = profile_for_offset(pressure_offset)
    annulus_volume = axisymmetric_profile_volume_m3(r, z)
    total_volume = sphere_cap + annulus_volume
    min_height = float(np.min(z))
    return {
        "r_m": r,
        "z_m": z,
        "sphere_cap_volume_m3": float(sphere_cap),
        "annulus_volume_m3": float(annulus_volume),
        "total_volume_m3": float(total_volume),
        "target_total_volume_m3": float(bridge_volume_m3),
        "target_annulus_volume_m3": float(target_annulus),
        "positive_volume_offset_m3": float(max(total_volume - float(bridge_volume_m3), 0.0)),
        "pressure_offset_m": float(pressure_offset),
        "repulsion_length_m": float(repulsion_length),
        "min_height_m": min_height,
        "area_m2": float(axisymmetric_profile_area_m2(r, z)),
        "success": bool(success),
        "volume_clamped": False,
        "message": "soft-repulsive linearized Young-Laplace bridge profile",
    }


def cox_lubrication_neck_profile(
    *,
    r_m: np.ndarray,
    initial_h_m: np.ndarray,
    rim_radius_m: float,
    rim_z_m: float,
    target_visible_missing_m3: float,
    initial_film_thickness_m: float,
    capillary_length_m: float,
    slide_speed_m_s: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    theta_eq_rad: float,
    macro_length_m: float,
    slip_length_m: float,
    min_angle_rad: float = 0.0,
    max_angle_rad: float = math.radians(25.0),
    min_width_m: float = 2.0e-6,
    max_width_m: float | None = None,
    min_z_m: float = 0.25e-6,
) -> dict[str, np.ndarray | float | bool | str]:
    """Local Cox/lubrication neck profile outside an attached bridge rim.

    This is a reusable boundary-layer closure for the steep bridge/film neck.
    The outer film is represented by an exponential Cox-Voinov recovery from
    the bridge rim height to the undisturbed film height.  The recovery length
    is limited by both a visible-film volume budget and the dynamic Cox angle:

    ``theta_dyn**3 = theta_eq**3 + 9 Ca ln(L/lambda)``.

    When Cox predicts a sharper neck than the visible-film volume budget, the
    unresolved excess volume is interpreted as attached bridge/neck reservoir
    volume, not as a broad artificial film depression.
    """

    r = np.asarray(r_m, dtype=float)
    initial_h = np.asarray(initial_h_m, dtype=float)
    if r.shape != initial_h.shape:
        raise ValueError("r_m and initial_h_m must have the same shape.")
    if r.size == 0:
        return {
            "z_m": initial_h.copy(),
            "theta_dyn_rad": float(theta_eq_rad),
            "neck_width_m": 0.0,
            "visible_missing_m3": 0.0,
            "target_visible_missing_m3": float(target_visible_missing_m3),
            "success": False,
            "message": "empty profile",
        }

    rim = float(rim_radius_m)
    h0 = float(initial_film_thickness_m)
    rim_initial_h = float(np.interp(rim, r, initial_h, left=h0, right=h0))
    rim_z = float(np.clip(float(rim_z_m), float(min_z_m), max(rim_initial_h, h0)))
    amplitude = max(rim_initial_h - rim_z, 0.0)
    if amplitude <= 1.0e-12:
        return {
            "z_m": initial_h.copy(),
            "theta_dyn_rad": float(theta_eq_rad),
            "neck_width_m": 0.0,
            "visible_missing_m3": 0.0,
            "target_visible_missing_m3": float(target_visible_missing_m3),
            "success": True,
            "message": "zero-amplitude neck",
        }

    log_factor = math.log(max(float(macro_length_m) / max(float(slip_length_m), 1.0e-30), 1.0))
    capillary_number = (
        max(float(viscosity_pa_s), 0.0)
        * abs(float(slide_speed_m_s))
        / max(float(surface_tension_n_m), 1.0e-30)
    )
    theta_dyn = float(
        np.cbrt(
            max(
                float(theta_eq_rad) ** 3 + 9.0 * capillary_number * log_factor,
                float(min_angle_rad) ** 3,
            )
        )
    )
    theta_dyn = float(np.clip(theta_dyn, float(min_angle_rad), float(max_angle_rad)))
    slope = max(math.tan(max(theta_dyn, 1.0e-9)), 1.0e-9)
    cox_width = amplitude / slope

    min_width = max(float(min_width_m), 1.0e-9)
    if max_width_m is None:
        max_width = max(float(capillary_length_m), min_width)
    else:
        max_width = max(float(max_width_m), min_width)
    cox_width = float(np.clip(cox_width, min_width, max_width))

    active = r >= rim
    x = np.maximum(r - rim, 0.0)

    def missing_for_width(width: float) -> float:
        trial = np.array(initial_h, copy=True)
        shape = np.exp(-x / max(float(width), 1.0e-30))
        trial[active] = np.maximum(initial_h[active] - amplitude * shape[active], float(min_z_m))
        integrand = 2.0 * math.pi * r[active] * np.maximum(initial_h[active] - trial[active], 0.0)
        if integrand.size < 2:
            return 0.0
        return float(np.trapezoid(integrand, r[active]))

    target = max(float(target_visible_missing_m3), 0.0)
    if target <= 1.0e-30:
        volume_width = min_width
    elif missing_for_width(max_width) <= target:
        volume_width = max_width
    else:
        lo = min_width
        hi = max_width
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if missing_for_width(mid) < target:
                lo = mid
            else:
                hi = mid
        volume_width = 0.5 * (lo + hi)

    width = min(float(volume_width), float(cox_width))
    z = np.array(initial_h, copy=True)
    shape = np.exp(-x / max(width, 1.0e-30))
    z[active] = np.maximum(initial_h[active] - amplitude * shape[active], float(min_z_m))
    visible_missing = missing_for_width(width)
    return {
        "z_m": z,
        "theta_dyn_rad": float(theta_dyn),
        "cox_width_m": float(cox_width),
        "volume_width_m": float(volume_width),
        "neck_width_m": float(width),
        "visible_missing_m3": float(visible_missing),
        "target_visible_missing_m3": float(target),
        "success": True,
        "message": "cox-lubrication neck profile",
    }


def cox_arc_length_bridge_profile(
    *,
    contact_radius_m: float,
    contact_z_m: float,
    rim_radius_m: float,
    rim_z_m: float,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    contact_slope_m_per_m: float,
    nodes: int = 96,
    min_z_m: float = 0.25e-6,
    clearance_m: float = 1.0e-8,
) -> dict[str, np.ndarray | float | bool | str]:
    """Bridge-side Cox meniscus as an arc-length-like ring profile.

    A height-only ``z(r)`` update cannot resolve the re-entrant, near-vertical
    branch at a sphere contact line on a coarse axisymmetric mesh.  This
    reusable profile moves both the ring radii and heights.  The radial ring
    spacing is clustered near the solid-liquid-air contact line so that the
    first mesh segment follows the Cox contact-line slope, while the last ring
    remains pinned to the film/bridge rim.
    """

    node_count = max(int(nodes), 4)
    contact_r = max(float(contact_radius_m), 0.0)
    rim_r = max(float(rim_radius_m), contact_r + 1.0e-12)
    contact_z = float(contact_z_m)
    rim_z = float(rim_z_m)
    gap = rim_r - contact_r
    dz = contact_z - rim_z
    eta = np.linspace(0.0, 1.0, node_count)

    if gap <= 1.0e-12 or dz <= 1.0e-12:
        r = contact_r + gap * eta
        z = contact_z + (rim_z - contact_z) * eta
        return {
            "r_m": r,
            "z_m": z,
            "radial_cluster_exponent": 1.0,
            "contact_slope_target": float(contact_slope_m_per_m),
            "first_segment_slope": float((z[1] - z[0]) / max(r[1] - r[0], 1.0e-30)),
            "success": True,
            "message": "degenerate Cox arc bridge profile",
        }

    secant_abs = dz / gap
    desired_abs = max(abs(float(contact_slope_m_per_m)), 2.0 * secant_abs)
    desired_abs = float(np.clip(desired_abs, 1.05 * secant_abs, 250.0 * secant_abs))
    first_eta = 1.0 / float(node_count - 1)

    def first_segment_slope_abs(q: float) -> float:
        q_eff = max(float(q), 1.0e-12)
        dr = gap * (first_eta**q_eff)
        return dz * first_eta / max(dr, 1.0e-30)

    if desired_abs <= first_segment_slope_abs(1.0):
        cluster_exponent = 1.0
    else:
        lo = 1.0
        hi = 2.0
        while first_segment_slope_abs(hi) < desired_abs and hi < 12.0:
            hi *= 1.5
        hi = min(hi, 12.0)
        for _ in range(72):
            mid = 0.5 * (lo + hi)
            if first_segment_slope_abs(mid) < desired_abs:
                lo = mid
            else:
                hi = mid
        cluster_exponent = 0.5 * (lo + hi)

    r = contact_r + gap * (eta**cluster_exponent)
    z = contact_z - dz * eta
    sphere_z = sphere_lower_surface_z_m(
        sphere_radius_m=float(sphere_radius_m),
        sphere_bottom_z_m=float(sphere_bottom_z_m),
        r_m=np.minimum(r, float(sphere_radius_m) * 0.999999),
    )
    upper = np.maximum(np.asarray(sphere_z, dtype=float) - float(clearance_m), float(min_z_m))
    z = np.clip(z, float(min_z_m), upper)
    z[0] = contact_z
    z[-1] = rim_z
    r[0] = contact_r
    r[-1] = rim_r
    return {
        "r_m": r,
        "z_m": z,
        "radial_cluster_exponent": float(cluster_exponent),
        "contact_slope_target": float(contact_slope_m_per_m),
        "first_segment_slope": float((z[1] - z[0]) / max(r[1] - r[0], 1.0e-30)),
        "success": True,
        "message": "Cox arc-length bridge meniscus profile",
    }


def _cox_dynamic_angle_rad(
    *,
    theta_eq_rad: float,
    slide_speed_m_s: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    macro_length_m: float,
    slip_length_m: float,
    min_angle_rad: float,
    max_angle_rad: float,
) -> float:
    log_factor = math.log(max(float(macro_length_m) / max(float(slip_length_m), 1.0e-30), 1.0))
    capillary_number = (
        max(float(viscosity_pa_s), 0.0)
        * abs(float(slide_speed_m_s))
        / max(float(surface_tension_n_m), 1.0e-30)
    )
    theta_dyn = float(
        np.cbrt(
            max(
                float(theta_eq_rad) ** 3 + 9.0 * capillary_number * log_factor,
                float(min_angle_rad) ** 3,
            )
        )
    )
    return float(np.clip(theta_dyn, float(min_angle_rad), float(max_angle_rad)))


def coupled_cox_young_laplace_lubrication_neck_profile(
    *,
    r_m: np.ndarray,
    initial_h_m: np.ndarray,
    rim_radius_m: float,
    target_visible_missing_m3: float,
    initial_film_thickness_m: float,
    slide_speed_m_s: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    theta_eq_rad: float,
    macro_length_m: float,
    slip_length_m: float,
    min_angle_rad: float = 0.0,
    max_angle_rad: float = math.radians(25.0),
    min_width_m: float = 2.0e-6,
    max_width_m: float | None = None,
    min_z_m: float = 0.25e-6,
    precursor_height_m: float | None = None,
    repulsion_pressure_scale_pa: float = 0.0,
    repulsion_exponent: float = 3.0,
    bvp_nodes: int = 80,
) -> dict[str, np.ndarray | float | bool | str]:
    """Coupled local neck solve for an attached sphere/film bridge.

    The neck is solved as a local moving-boundary lubrication layer whose
    pressure is the axisymmetric Young-Laplace pressure

    ``p = rho g h - gamma (h'' + h'/r) + Pi(h)``.

    ``Pi(h)`` is an optional precursor-film repulsion.  It is zero by default;
    when enabled by the caller it regularizes unresolved very-thin wetting
    layers without prescribing a validation-profile shape.

    In the frame of the moving bridge/film rim, mass conservation gives the
    lubrication flux scale ``q = U (h - h_init)``.  The solved boundary layer
    satisfies the Cox-Voinov dynamic angle at the bridge rim, matches the
    undisturbed film height and slope at the outer edge, and consumes the
    supplied visible film-volume budget.  If the nonlinear boundary-value
    solve fails, callers get a conservative Cox/lubrication fallback with the
    failure message preserved.
    """

    r = np.asarray(r_m, dtype=float)
    initial_h = np.asarray(initial_h_m, dtype=float)
    if r.shape != initial_h.shape:
        raise ValueError("r_m and initial_h_m must have the same shape.")
    if r.size < 3:
        return cox_lubrication_neck_profile(
            r_m=r,
            initial_h_m=initial_h,
            rim_radius_m=rim_radius_m,
            rim_z_m=float(initial_film_thickness_m),
            target_visible_missing_m3=target_visible_missing_m3,
            initial_film_thickness_m=initial_film_thickness_m,
            capillary_length_m=math.sqrt(float(surface_tension_n_m) / max(float(density_kg_m3) * float(gravity_m_s2), 1.0e-300)),
            slide_speed_m_s=slide_speed_m_s,
            viscosity_pa_s=viscosity_pa_s,
            surface_tension_n_m=surface_tension_n_m,
            theta_eq_rad=theta_eq_rad,
            macro_length_m=macro_length_m,
            slip_length_m=slip_length_m,
            min_angle_rad=min_angle_rad,
            max_angle_rad=max_angle_rad,
            min_width_m=min_width_m,
            max_width_m=max_width_m,
            min_z_m=min_z_m,
        )

    rim = float(rim_radius_m)
    h0 = float(initial_film_thickness_m)
    target = max(float(target_visible_missing_m3), 0.0)
    min_z = max(float(min_z_m), 1.0e-10)
    min_width = max(float(min_width_m), 1.0e-9)
    precursor_height = (
        min_z
        if precursor_height_m is None
        else max(float(precursor_height_m), min_z)
    )
    repulsion_scale = max(float(repulsion_pressure_scale_pa), 0.0)
    repulsion_power = max(float(repulsion_exponent), 1.0)
    capillary_length = math.sqrt(float(surface_tension_n_m) / max(float(density_kg_m3) * float(gravity_m_s2), 1.0e-300))
    if max_width_m is None:
        max_width = max(capillary_length, min_width)
    else:
        max_width = max(float(max_width_m), min_width)

    theta_dyn = _cox_dynamic_angle_rad(
        theta_eq_rad=theta_eq_rad,
        slide_speed_m_s=slide_speed_m_s,
        viscosity_pa_s=viscosity_pa_s,
        surface_tension_n_m=surface_tension_n_m,
        macro_length_m=macro_length_m,
        slip_length_m=slip_length_m,
        min_angle_rad=min_angle_rad,
        max_angle_rad=max_angle_rad,
    )
    slope = max(math.tan(max(theta_dyn, 1.0e-9)), 1.0e-9)

    if target <= 1.0e-30:
        return {
            "z_m": initial_h.copy(),
            "rim_z_m": h0,
            "theta_dyn_rad": float(theta_dyn),
            "neck_width_m": 0.0,
            "visible_missing_m3": 0.0,
            "target_visible_missing_m3": float(target),
            "success": True,
            "message": "zero-volume coupled neck",
        }

    r_keep = np.asarray(r, dtype=float)
    h_keep = np.asarray(initial_h, dtype=float)
    order = np.argsort(r_keep)
    r_keep = r_keep[order]
    h_keep = h_keep[order]
    unique = np.concatenate(([True], np.diff(r_keep) > 1.0e-14))
    r_keep = r_keep[unique]
    h_keep = h_keep[unique]
    if r_keep[0] > rim:
        r_keep = np.concatenate(([rim], r_keep))
        h_keep = np.concatenate(([float(np.interp(rim, r, initial_h, left=h0, right=h0))], h_keep))

    h_slope = np.gradient(h_keep, r_keep, edge_order=1)

    def h_initial_at(rr: np.ndarray | float) -> np.ndarray:
        return np.interp(rr, r_keep, h_keep, left=h0, right=float(h_keep[-1]))

    def initial_slope_at(rr: float) -> float:
        return float(np.interp(float(rr), r_keep, h_slope, left=0.0, right=float(h_slope[-1])))

    available_width = max(float(r_keep[-1]) - rim, min_width)
    max_width = min(max_width, available_width)
    if max_width <= min_width * 1.001:
        max_width = min_width * 1.001

    # Volume and Cox angle give a parameter-free first estimate:
    # exponential/triangular layers both scale as O(2*pi*r*A*W), and
    # W ~= A / tan(theta_dyn).
    amp_guess = math.sqrt(max(target * slope / max(2.0 * math.pi * max(rim, 1.0e-12), 1.0e-30), 0.0))
    amp_guess = float(np.clip(amp_guess, min_z, max(h0 - min_z, min_z)))
    width_guess = float(np.clip(amp_guess / slope, min_width, max_width))
    rim_z_guess = float(np.clip(h0 - amp_guess, min_z, h0))

    s_mesh = np.linspace(0.0, 1.0, max(int(bvp_nodes), 16))

    def hermite_guess(rim_z: float, width: float) -> np.ndarray:
        outer_h = float(h_initial_at(rim + width))
        outer_slope = initial_slope_at(rim + width)
        s = s_mesh
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        h = h00 * rim_z + h10 * width * slope + h01 * outer_h + h11 * width * outer_slope
        h = np.clip(h, min_z, np.maximum(h_initial_at(rim + width * s), min_z))
        dhds = np.gradient(h, s, edge_order=2)
        dhdx = dhds / max(width, 1.0e-30)
        d2hdx2 = np.gradient(dhdx, s, edge_order=2) / max(width, 1.0e-30)
        volume_density = 2.0 * math.pi * (rim + width * s) * (h_initial_at(rim + width * s) - h)
        cumulative = np.zeros_like(s)
        cumulative[1:] = np.cumsum(0.5 * (volume_density[1:] + volume_density[:-1]) * np.diff(s) * width)
        return np.vstack((h, dhdx, d2hdx2, cumulative))

    def solve_once(rim_z0: float, width0: float):
        y0 = hermite_guess(rim_z0, width0)
        p0 = np.asarray([rim_z0, math.log(max(width0, 1.0e-30))], dtype=float)

        def ode(s: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            width = width_from_log(p[1])
            x = width * s
            rr = np.maximum(rim + x, 1.0e-12)
            h = np.maximum(y[0], min_z)
            hp = y[1]
            hpp = y[2]
            h_init = h_initial_at(rr)
            p_grad = -3.0 * max(float(viscosity_pa_s), 0.0) * abs(float(slide_speed_m_s)) * (h - h_init) / np.maximum(h**3, min_z**3)
            if repulsion_scale > 0.0:
                pi_h = repulsion_scale * (precursor_height / np.maximum(h, min_z)) ** repulsion_power
                dpi_dh = -repulsion_power * pi_h / np.maximum(h, min_z)
                repulsion_grad = dpi_dh * hp
            else:
                repulsion_grad = 0.0
            hppp = (
                (float(density_kg_m3) * float(gravity_m_s2) * hp + repulsion_grad - p_grad)
                / max(float(surface_tension_n_m), 1.0e-30)
                - hpp / rr
                + hp / (rr * rr)
            )
            vol_prime = 2.0 * math.pi * rr * (h_init - y[0])
            return np.vstack((width * hp, width * hpp, width * hppp, width * vol_prime))

        def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
            width = width_from_log(p[1])
            outer_r = rim + width
            return np.asarray(
                [
                    ya[0] - p[0],
                    ya[1] - slope,
                    yb[0] - float(h_initial_at(outer_r)),
                    yb[1] - initial_slope_at(outer_r),
                    ya[3],
                    yb[3] - target,
                ],
                dtype=float,
            )

        return solve_bvp(
            ode,
            bc,
            s_mesh,
            y0,
            p=p0,
            tol=2.0e-4,
            max_nodes=400,
            verbose=0,
        )

    attempts: list[tuple[float, float]] = [
        (rim_z_guess, width_guess),
        (float(np.clip(h0 - 1.5 * amp_guess, min_z, h0)), float(np.clip(1.5 * width_guess, min_width, max_width))),
        (float(np.clip(h0 - 0.75 * amp_guess, min_z, h0)), float(np.clip(0.75 * width_guess, min_width, max_width))),
    ]
    log_width_min = math.log(max(min_width * 0.25, 1.0e-30))
    log_width_max = math.log(max(max_width * 2.0, min_width * 1.01))

    def width_from_log(log_width: float) -> float:
        return float(math.exp(float(np.clip(float(log_width), log_width_min, log_width_max))))

    result = None
    message = ""
    for rim_z0, width0 in attempts:
        try:
            candidate = solve_once(rim_z0, width0)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            message = str(exc)
            continue
        width = width_from_log(candidate.p[1]) if candidate.p is not None else float("nan")
        rim_z = float(candidate.p[0]) if candidate.p is not None else float("nan")
        valid = (
            bool(candidate.success)
            and math.isfinite(width)
            and math.isfinite(rim_z)
            and min_width * 0.25 <= width <= max_width * 2.0
            and min_z * 0.5 <= rim_z <= h0 * 1.25
        )
        message = str(candidate.message)
        if valid:
            result = candidate
            break

    if result is None:
        fallback = cox_lubrication_neck_profile(
            r_m=r,
            initial_h_m=initial_h,
            rim_radius_m=rim_radius_m,
            rim_z_m=rim_z_guess,
            target_visible_missing_m3=target,
            initial_film_thickness_m=initial_film_thickness_m,
            capillary_length_m=capillary_length,
            slide_speed_m_s=slide_speed_m_s,
            viscosity_pa_s=viscosity_pa_s,
            surface_tension_n_m=surface_tension_n_m,
            theta_eq_rad=theta_eq_rad,
            macro_length_m=macro_length_m,
            slip_length_m=slip_length_m,
            min_angle_rad=min_angle_rad,
            max_angle_rad=max_angle_rad,
            min_width_m=min_width_m,
            max_width_m=max_width,
            min_z_m=min_z_m,
        )
        fallback["success"] = False
        fallback["message"] = f"coupled BVP failed; fallback used: {message}"
        fallback["rim_z_m"] = float(rim_z_guess)
        return fallback

    width = width_from_log(result.p[1])
    rim_z = float(result.p[0])
    width = float(np.clip(width, min_width, max_width))
    rim_z = float(np.clip(rim_z, min_z, h0))

    z = np.array(initial_h, copy=True)
    active = (r >= rim) & (r <= rim + width)
    if np.any(active):
        s_eval = np.clip((r[active] - rim) / max(width, 1.0e-30), 0.0, 1.0)
        solved_h = np.asarray(result.sol(s_eval)[0], dtype=float)
        z[active] = np.clip(solved_h, min_z, initial_h[active])
    outer = r > rim + width
    z[outer] = initial_h[outer]

    integrand = 2.0 * math.pi * r * np.maximum(initial_h - z, 0.0)
    visible_missing = float(np.trapezoid(integrand, r)) if integrand.size >= 2 else 0.0
    return {
        "z_m": z,
        "rim_z_m": float(rim_z),
        "theta_dyn_rad": float(theta_dyn),
        "neck_width_m": float(width),
        "precursor_height_m": float(precursor_height),
        "repulsion_pressure_scale_pa": float(repulsion_scale),
        "visible_missing_m3": float(visible_missing),
        "target_visible_missing_m3": float(target),
        "success": True,
        "message": "coupled Cox/Young-Laplace/lubrication neck profile",
    }


def axisymmetric_capillary_bridge_profile(
    *,
    sphere_radius_m: float,
    sphere_bottom_z_m: float,
    contact_radius_m: float,
    rim_radius_m: float,
    rim_z_m: float,
    bridge_volume_m3: float,
    nodes: int = 36,
    min_z_m: float = 0.25e-6,
    contact_angle_rad: float | None = None,
    rim_slope_m_per_m: float | None = None,
    previous_r_m: np.ndarray | None = None,
    previous_z_m: np.ndarray | None = None,
    lower_bound_repulsion: float = 0.0,
    lower_bound_repulsion_length_m: float = 5.0e-6,
    slope_regularization: float = 0.0,
    curvature_regularization: float = 0.0,
) -> dict[str, np.ndarray | float | bool | str]:
    """Discrete Young-Laplace bridge profile by constrained surface minimization.

    The profile is the axisymmetric liquid-air surface between a solid-sphere
    contact ring and a bridge/film rim.  It minimizes surface area at fixed
    liquid volume, with endpoints pinned to the sphere and film.  The volume
    includes the liquid under the sphere for ``r < contact_radius_m`` plus the
    annular volume under the optimized free surface.
    """

    sphere_radius = float(sphere_radius_m)
    contact_radius = float(np.clip(contact_radius_m, 1.0e-12, sphere_radius * 0.999999))
    rim_radius = max(float(rim_radius_m), contact_radius + 1.0e-9)
    rim_z = max(float(rim_z_m), float(min_z_m))
    contact_z = float(
        sphere_lower_surface_z_m(
            sphere_radius_m=sphere_radius,
            sphere_bottom_z_m=float(sphere_bottom_z_m),
            r_m=contact_radius,
        )
    )
    node_count = max(4, int(nodes))
    r = np.linspace(contact_radius, rim_radius, node_count)
    sphere_z = np.asarray(
        sphere_lower_surface_z_m(
            sphere_radius_m=sphere_radius,
            sphere_bottom_z_m=float(sphere_bottom_z_m),
            r_m=np.minimum(r, sphere_radius * 0.999999),
        ),
        dtype=float,
    )
    lower = np.full(node_count, max(float(min_z_m), 1.0e-10), dtype=float)
    lower[0] = contact_z
    lower[-1] = rim_z
    upper = np.maximum(sphere_z - 1.0e-8, lower + 1.0e-10)
    upper[0] = contact_z
    upper[-1] = rim_z

    sphere_cap = sphere_lower_cap_volume_m3(
        sphere_radius_m=sphere_radius,
        sphere_bottom_z_m=float(sphere_bottom_z_m),
        contact_radius_m=contact_radius,
    )
    target_annulus = max(float(bridge_volume_m3) - sphere_cap, 0.0)
    min_profile = np.array(lower, copy=True)
    max_profile = np.array(upper, copy=True)
    min_annulus = axisymmetric_profile_volume_m3(r, min_profile)
    max_annulus = axisymmetric_profile_volume_m3(r, max_profile)
    clamped = False
    if target_annulus < min_annulus:
        target_annulus = min_annulus
        clamped = True
    elif target_annulus > max_annulus:
        target_annulus = max_annulus
        clamped = True

    fixed_values: dict[int, float] = {}
    if node_count >= 4 and contact_angle_rad is not None:
        axial = math.sqrt(max(sphere_radius * sphere_radius - contact_radius * contact_radius, 1.0e-30))
        solid_angle = math.atan(contact_radius / axial)
        liquid_angle = float(np.clip(solid_angle - float(contact_angle_rad), -0.49 * math.pi, 0.49 * math.pi))
        contact_slope = math.tan(liquid_angle)
        fixed_values[1] = float(np.clip(contact_z + contact_slope * (r[1] - r[0]), lower[1], upper[1]))
    if node_count >= 4 and rim_slope_m_per_m is not None:
        fixed_values[node_count - 2] = float(
            np.clip(rim_z - float(rim_slope_m_per_m) * (r[-1] - r[-2]), lower[-2], upper[-2])
        )

    if previous_r_m is not None and previous_z_m is not None and len(previous_r_m) >= 2:
        z0 = np.interp(r, np.asarray(previous_r_m, dtype=float), np.asarray(previous_z_m, dtype=float))
    else:
        s = np.linspace(0.0, 1.0, node_count)
        z0 = contact_z + (rim_z - contact_z) * (s**0.70)
    z0 = np.clip(z0, lower, upper)
    z0[0] = contact_z
    z0[-1] = rim_z
    for idx, value in fixed_values.items():
        z0[int(idx)] = float(value)

    free_indices = [idx for idx in range(1, node_count - 1) if idx not in fixed_values]

    def full_profile(x: np.ndarray) -> np.ndarray:
        z = np.array(z0, copy=True)
        z[0] = contact_z
        z[-1] = rim_z
        for idx, value in fixed_values.items():
            z[int(idx)] = float(value)
        for local_i, idx in enumerate(free_indices):
            z[int(idx)] = x[local_i]
        return z

    repulsion_strength = max(float(lower_bound_repulsion), 0.0)
    repulsion_length = max(float(lower_bound_repulsion_length_m), 1.0e-12)
    edges = np.empty(node_count + 1, dtype=float)
    if node_count >= 2:
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        edges[0] = max(0.0, r[0] - 0.5 * (r[1] - r[0]))
        edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])
    else:
        edges[:] = r[0]
    nodal_area = math.pi * np.maximum(edges[1:] ** 2 - edges[:-1] ** 2, 0.0)
    repulsion_mask = np.ones(node_count, dtype=bool)
    repulsion_mask[0] = False
    repulsion_mask[-1] = False

    def lower_bound_repulsion_area(z: np.ndarray) -> float:
        if repulsion_strength <= 0.0:
            return 0.0
        clearance = np.maximum(np.asarray(z, dtype=float) - lower, 0.0)
        density = np.exp(-clearance / repulsion_length)
        return float(np.sum(nodal_area[repulsion_mask] * density[repulsion_mask]))

    slope_reg = max(float(slope_regularization), 0.0)
    curvature_reg = max(float(curvature_regularization), 0.0)

    def profile_smoothness_penalty(z: np.ndarray) -> float:
        if (slope_reg <= 0.0 and curvature_reg <= 0.0) or z.size < 3:
            return 0.0
        rr = np.asarray(r, dtype=float)
        zz = np.asarray(z, dtype=float)
        dr = np.diff(rr)
        dr = np.maximum(dr, 1.0e-30)
        mid_r = 0.5 * (rr[:-1] + rr[1:])
        slopes = np.diff(zz) / dr
        penalty = 0.0
        if slope_reg > 0.0:
            # Finite microscopic contact-line dissipation spreads an otherwise
            # degenerate vertical-wall/flat-shelf minimizer while preserving the
            # same axisymmetric volume constraint.
            penalty += slope_reg * float(np.sum(2.0 * math.pi * mid_r * dr * slopes * slopes))
        if curvature_reg > 0.0:
            dhdr = np.gradient(zz, rr, edge_order=1)
            d2hdr2 = np.gradient(dhdr, rr, edge_order=1)
            curvature = d2hdr2 + dhdr / np.maximum(rr, 1.0e-30)
            penalty += curvature_reg * float(np.sum(nodal_area * curvature * curvature))
        return penalty

    def objective(x: np.ndarray) -> float:
        z = full_profile(x)
        return (
            axisymmetric_profile_area_m2(r, z)
            + repulsion_strength * lower_bound_repulsion_area(z)
            + profile_smoothness_penalty(z)
        )

    def volume_constraint(x: np.ndarray) -> float:
        return axisymmetric_profile_volume_m3(r, full_profile(x)) - target_annulus

    if node_count <= 2 or not free_indices:
        z = z0
        success = True
        message = "endpoint-only profile"
    else:
        x0 = np.asarray([z0[idx] for idx in free_indices], dtype=float)
        bounds = [(float(lower[i]), float(upper[i])) for i in free_indices]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": volume_constraint}],
            options={"maxiter": 500, "ftol": 1.0e-13, "disp": False},
        )
        if result.success:
            z = full_profile(np.asarray(result.x, dtype=float))
        else:
            z = z0
            current = axisymmetric_profile_volume_m3(r, z)
            span = max(max_annulus - min_annulus, 1.0e-30)
            blend = np.clip((target_annulus - current) / span, -1.0, 1.0)
            z = np.clip(z + blend * (max_profile - min_profile), lower, upper)
        success = bool(result.success)
        message = str(result.message)

    annulus_volume = axisymmetric_profile_volume_m3(r, z)
    total_volume = sphere_cap + annulus_volume
    return {
        "r_m": r,
        "z_m": z,
        "sphere_cap_volume_m3": float(sphere_cap),
        "annulus_volume_m3": float(annulus_volume),
        "total_volume_m3": float(total_volume),
        "target_total_volume_m3": float(bridge_volume_m3),
        "target_annulus_volume_m3": float(target_annulus),
        "area_m2": float(axisymmetric_profile_area_m2(r, z)),
        "success": bool(success),
        "volume_clamped": bool(clamped),
        "message": message,
        "lower_bound_repulsion_area_m2": float(lower_bound_repulsion_area(z)),
        "slope_regularization": float(slope_reg),
        "curvature_regularization": float(curvature_reg),
    }


def bridge_dradius_dvolume_m_per_m3(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
) -> float:
    volume_ul, radius_mm, _ = bridge_table_arrays(config)
    if volume_ul.size >= 4:
        radius_value_mm = float(
            np.clip(float(bridge_radius_m) * 1.0e3, radius_mm[0], radius_mm[-1])
        )
        derivative_ul_per_mm = max(
            float(
                PchipInterpolator(radius_mm, volume_ul, extrapolate=True)
                .derivative()(radius_value_mm)
            ),
            1.0e-12,
        )
        # Use the derivative of the same V(a) interpolant used by
        # bridge_volume_from_radius_m3 so the moving-boundary ODE conserves
        # the represented bridge volume exactly.
        return 1.0 / (derivative_ul_per_mm * 1.0e-6)

    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    a = max(float(bridge_radius_m), minimum_bridge_radius_m(config) * 1.001)
    shape_factor = max(float(config.bridge_volume_shape_factor), 1.0e-9)
    return 1.0 / max(2.0 * shape_factor * math.pi * h0 * a, 1.0e-30)


def bridge_head_m(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
) -> float:
    _, radius_mm, head_mm = bridge_table_arrays(config)
    if radius_mm.size < 4:
        return float("nan")
    radius_value_mm = float(np.clip(float(bridge_radius_m) * 1.0e3, radius_mm[0], radius_mm[-1]))
    return max(float(PchipInterpolator(radius_mm, head_mm, extrapolate=True)(radius_value_mm)), 0.0) * 1.0e-3


def bridge_pressure_startup_factor(
    config: AxisymmetricBridgeFilmConfig,
    t_s: float | None,
) -> float:
    if t_s is None:
        return 1.0
    t = max(float(t_s), 0.0)
    activation = 1.0 - math.exp(
        -((t / max(float(config.bridge_pressure_activation_time_s), 1.0e-12)) ** float(config.bridge_pressure_activation_exponent))
    )
    relax = (1.0 + t / max(float(config.bridge_pressure_relax_time_s), 1.0e-12)) ** float(config.bridge_pressure_relax_exponent)
    return float(activation / relax)


def bridge_pressure_pa(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
    t_s: float | None = None,
) -> float:
    """Quasi-static bridge suction used as the film inner pressure."""

    head = bridge_head_m(config, bridge_radius_m)
    if math.isfinite(head):
        scaled_head = float(config.bridge_table_head_scale) * bridge_pressure_startup_factor(config, t_s) * head
        return -float(config.density_kg_m3) * float(config.gravity_m_s2) * scaled_head

    contact = minimum_bridge_radius_m(config)
    a = max(float(bridge_radius_m), contact * 1.001)
    capillary_length = _capillary_length_m(config)
    fallback_head = 0.10 * bridge_pressure_startup_factor(config, t_s) * capillary_length * math.sqrt(contact / a)
    return -float(config.density_kg_m3) * float(config.gravity_m_s2) * fallback_head


def bridge_rim_pressure_pa(
    config: AxisymmetricBridgeFilmConfig,
    r_m: np.ndarray,
    bridge_radius_m: float,
    t_s: float,
) -> np.ndarray:
    multiplier = float(config.bridge_rim_pressure_multiplier)
    if multiplier == 0.0:
        return np.zeros_like(r_m, dtype=float)
    width = max(float(config.bridge_rim_pressure_width_mm) * 1.0e-3, 1.0e-9)
    center = float(bridge_radius_m) + float(config.bridge_rim_pressure_offset_mm) * 1.0e-3
    weight = np.exp(-0.5 * ((np.asarray(r_m, dtype=float) - center) / width) ** 2)
    return multiplier * bridge_pressure_pa(config, bridge_radius_m, t_s) * weight


def bridge_inflow_bottleneck_factor(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
) -> float:
    if not bool(config.bridge_inflow_bottleneck_enabled):
        return 1.0
    initial_volume_ul = max(float(config.initial_bridge_volume_ul), 0.0)
    volume_ul = float(bridge_volume_from_radius_m3(config, bridge_radius_m) * 1.0e9)
    grown_volume_ul = max(volume_ul - initial_volume_ul, 0.0)
    scale = max(float(config.bridge_inflow_bottleneck_volume_ul), 1.0e-12)
    exponent = max(float(config.bridge_inflow_bottleneck_exponent), 1.0e-12)
    floor = float(np.clip(float(config.bridge_inflow_bottleneck_floor), 0.0, 1.0))
    factor = 1.0 / (1.0 + (grown_volume_ul / scale) ** exponent)
    return float(np.clip(factor, floor, 1.0))


def local_capture_volume_ul(config: AxisymmetricBridgeFilmConfig) -> float:
    """Near-contact liquid volume available immediately after bridge stitching.

    The scale is the first-contact annulus ``2*pi*a0*h0*ell_c`` multiplied by
    a dimensionless width.  It is geometry/material based and is independent of
    any validation curve.
    """

    if not bool(config.local_capture_enabled):
        return 0.0
    width = max(float(config.local_capture_width_capillary_lengths), 0.0)
    if width <= 0.0:
        return 0.0
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    a0 = minimum_bridge_radius_m(config)
    ell_c = _capillary_length_m(config)
    return float(2.0 * math.pi * a0 * h0 * ell_c * width * 1.0e9)


def local_capture_delta_volume_ul(config: AxisymmetricBridgeFilmConfig, t_s: float) -> float:
    """Fast stitched-bridge capture volume at time ``t_s``."""

    capture_ul = local_capture_volume_ul(config)
    if capture_ul <= 0.0:
        return 0.0
    tau = max(float(config.local_capture_time_s), 1.0e-12)
    exponent = max(float(config.local_capture_exponent), 1.0e-12)
    t = max(float(t_s), 0.0)
    return float(capture_ul * (1.0 - math.exp(-((t / tau) ** exponent))))


def local_capture_volume_limit_ul(config: AxisymmetricBridgeFilmConfig, t_s: float) -> float:
    """Slowly relaxing upper volume bound for the local-capture reservoir."""

    capture_ul = local_capture_volume_ul(config)
    if capture_ul <= 0.0:
        return float("inf")
    tau = max(float(config.local_capture_release_time_s), 1.0e-12)
    exponent = max(float(config.local_capture_release_exponent), 1.0e-12)
    if bool(config.local_capture_release_saturation_enabled):
        elapsed = max(float(t_s) - float(config.local_capture_release_delay_s), 0.0)
        release = 1.0 - math.exp(-((elapsed / tau) ** exponent))
        h0 = float(config.initial_film_thickness_um) * 1.0e-6
        a0 = minimum_bridge_radius_m(config)
        ell_c = _capillary_length_m(config)
        extra_width = max(float(config.local_capture_release_extra_width_capillary_lengths), 0.0)
        extra_ul = 2.0 * math.pi * a0 * h0 * ell_c * extra_width * 1.0e9
        return float(capture_ul + extra_ul * np.clip(release, 0.0, 1.0))
    release = (max(float(t_s), 0.0) / tau) ** exponent
    return float(capture_ul * (1.0 + release))


def bridge_delta_volume_with_local_capture_ul(
    config: AxisymmetricBridgeFilmConfig,
    bridge_radius_m: float,
    t_s: float,
) -> float:
    """Bridge-volume delta after applying optional local-capture closure."""

    raw_delta_ul = max(
        float(bridge_volume_from_radius_m3(config, bridge_radius_m) * 1.0e9)
        - max(float(config.initial_bridge_volume_ul), 0.0),
        0.0,
    )
    if not bool(config.local_capture_enabled):
        return raw_delta_ul
    lower = local_capture_delta_volume_ul(config, t_s)
    upper = local_capture_volume_limit_ul(config, t_s)
    return float(min(max(raw_delta_ul, lower), upper))


def _grid_at_bridge_radius(
    config: AxisymmetricBridgeFilmConfig,
    nu: np.ndarray,
    nu_faces: np.ndarray,
    bridge_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    contact = minimum_bridge_radius_m(config)
    a = min(radius_m * 0.985, max(float(bridge_radius_m), contact * 1.001))
    return a + (radius_m - a) * nu, a + (radius_m - a) * nu_faces


def _nonuniform_profile_derivatives(
    radius_m: np.ndarray,
    height_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Three-point first and second derivatives on a nonuniform radial grid.

    These are the local Taylor coefficients in Eqs. (9)-(10) of the
    bridge-film finite-volume formulation.  A nested generic gradient widens
    the stencil and is unstable on the strongly stretched dimple grid.
    """

    radius = np.asarray(radius_m, dtype=float)
    height = np.asarray(height_m, dtype=float)
    if radius.ndim != 1 or height.shape != radius.shape or radius.size < 3:
        raise ValueError("radius and height must be matching one-dimensional arrays")
    dr_minus = np.maximum(radius[1:-1] - radius[:-2], 1.0e-30)
    dr_plus = np.maximum(radius[2:] - radius[1:-1], 1.0e-30)
    span = dr_minus + dr_plus
    delta_plus = height[2:] - height[1:-1]
    delta_minus = height[:-2] - height[1:-1]
    first = np.empty_like(height)
    second = np.empty_like(height)
    first[1:-1] = (
        dr_minus / (dr_plus * span) * delta_plus
        - dr_plus / (dr_minus * span) * delta_minus
    )
    second[1:-1] = (
        2.0 / (dr_plus * span) * delta_plus
        + 2.0 / (dr_minus * span) * delta_minus
    )
    # Endpoint pressures are supplied by boundary conditions in the coupled
    # solve.  One-sided values keep this utility well-defined for diagnostics.
    first[0] = (-3.0 * height[0] + 4.0 * height[1] - height[2]) / max(
        radius[2] - radius[0], 1.0e-30
    )
    first[-1] = (3.0 * height[-1] - 4.0 * height[-2] + height[-3]) / max(
        radius[-1] - radius[-3], 1.0e-30
    )
    second[0] = second[1]
    second[-1] = second[-2]
    return first, second


def _snapshot_times(config: AxisymmetricBridgeFilmConfig) -> list[float]:
    t_end = float(config.t_end_s)
    times = {
        0.0,
        *[float(t) for t in config.snapshot_times_s if 0.0 <= float(t) <= t_end],
        *[float(t) for t in config.diagnostic_times_s if 0.0 <= float(t) <= t_end],
    }
    if t_end >= 0.0:
        times.add(t_end)
    return sorted(times)


def _inner_pressure_boundary_height_m(
    config: AxisymmetricBridgeFilmConfig,
    r_m: np.ndarray,
    h_m: np.ndarray,
    target_pressure_pa: float,
) -> float:
    """Height at the bridge/film edge from local Young-Laplace balance.

    The moving-boundary lubrication solve stores the outer film from the bridge
    rim to the substrate edge.  Pinning the rim height to the undisturbed film
    thickness suppresses the capillary dimple.  This helper instead chooses the
    rim height whose local axisymmetric curvature gives the bridge pressure,
    using the neighboring film nodes from the current state.
    """

    r = np.asarray(r_m, dtype=float)
    h = np.asarray(h_m, dtype=float)
    if r.size < 4 or h.size < 4:
        return float(h[0])
    upper = float(config.initial_film_thickness_um) * 1.0e-6
    lower = max(float(config.min_film_thickness_m), 1.0e-9)
    if upper <= lower:
        return lower

    def pressure_for(candidate_h: float) -> float:
        trial = np.array(h[: max(5, min(h.size, 8))], copy=True)
        trial[0] = float(candidate_h)
        rr = r[: trial.size]
        h_prime, h_second = _nonuniform_profile_derivatives(rr, trial)
        w = math.sqrt(1.0 + float(h_prime[0]) * float(h_prime[0]))
        curvature = float(h_second[0]) / (w**3) + float(h_prime[0]) / max(float(rr[0]), 1.0e-12) / w
        return (
            float(config.density_kg_m3) * float(config.gravity_m_s2) * float(candidate_h)
            - float(config.surface_tension_n_m) * curvature
        )

    sample_count = max(int(config.inner_pressure_boundary_samples), 12)
    candidates = np.linspace(lower, upper, sample_count)
    residual = np.asarray([pressure_for(value) - float(target_pressure_pa) for value in candidates])
    sign_change = np.where(residual[:-1] * residual[1:] <= 0.0)[0]
    if sign_change.size:
        lo = float(candidates[int(sign_change[0])])
        hi = float(candidates[int(sign_change[0]) + 1])
        f_lo = float(pressure_for(lo) - float(target_pressure_pa))
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            f_mid = float(pressure_for(mid) - float(target_pressure_pa))
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid
        return float(np.clip(0.5 * (lo + hi), lower, upper))
    best = int(np.argmin(np.abs(residual)))
    return float(np.clip(candidates[best], lower, upper))


def simulate_bridge_film(
    config: AxisymmetricBridgeFilmConfig,
    *,
    initial_bridge_radius_m: float | None = None,
    initial_profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Forward solve the moving-boundary bridge-film problem.

    ``initial_profile`` supports a physically resolved startup calculation,
    such as the viscocapillary initialization used for a singular first
    contact.  It is a numerical state, not a prescribed validation profile.
    """

    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    n_grid = int(config.grid_nodes)
    stretch = float(config.moving_grid_stretch)
    node_index = np.arange(n_grid, dtype=float)
    nu = np.sinh(stretch * node_index / max(n_grid - 1, 1)) / math.sinh(stretch)
    nu_faces = np.zeros(n_grid + 1, dtype=float)
    nu_faces[0] = 0.0
    nu_faces[-1] = 1.0
    nu_faces[1:-1] = 0.5 * (nu[:-1] + nu[1:])

    t_start = max(float(config.t_start_s), 0.0)
    t_end = float(config.t_end_s)
    snapshots = _snapshot_times(config)
    solve_times = [t for t in snapshots if t >= t_start]

    volume_radius_m = bridge_radius_from_volume_m(
        config, float(config.initial_bridge_volume_ul) * 1.0e-9
    )
    initial_bridge_radius_m = max(
        volume_radius_m if initial_bridge_radius_m is None else float(initial_bridge_radius_m),
        float(config.initial_bridge_radius_mm) * 1.0e-3,
        minimum_bridge_radius_m(config) * 1.001,
    )
    r0, _ = _grid_at_bridge_radius(config, nu, nu_faces, initial_bridge_radius_m)
    if initial_profile is None:
        h_initial_state = initial_bessel_film_profile_m(config, r0)
    else:
        source_r = np.asarray(initial_profile[0], dtype=float)
        source_h = np.asarray(initial_profile[1], dtype=float)
        if source_r.ndim != 1 or source_h.shape != source_r.shape or source_r.size < 4:
            raise ValueError("initial_profile must contain matching one-dimensional r and h arrays")
        order = np.argsort(source_r)
        h_initial_state = np.interp(r0, source_r[order], source_h[order])
    h_initial_state[0] = h0
    h_initial_state[-1] = 0.0

    def discrete_total_volume(bridge_radius: float, height: np.ndarray) -> float:
        _r, faces = _grid_at_bridge_radius(
            config, nu, nu_faces, float(bridge_radius)
        )
        cell_volume = math.pi * np.sum(
            (faces[2:n_grid] ** 2 - faces[1 : n_grid - 1] ** 2)
            * np.asarray(height[1:-1], dtype=float)
        )
        return float(
            bridge_volume_from_radius_m3(config, float(bridge_radius))
            + math.pi * float(bridge_radius) ** 2 * h0
            + cell_volume
        )

    initial_total_volume = discrete_total_volume(
        initial_bridge_radius_m, h_initial_state
    )
    table_volume, table_radius_mm, _table_head = bridge_table_arrays(config)
    algebraic_lower_radius = (
        float(table_radius_mm[0]) * 1.0e-3
        if table_volume.size >= 4
        else max(minimum_bridge_radius_m(config) * 1.001, 1.0e-12)
    )
    algebraic_upper_radius = (
        min(float(table_radius_mm[-1]) * 1.0e-3, radius_m * 0.985)
        if table_volume.size >= 4
        else radius_m * 0.985
    )

    def radius_from_conserved_inventory(height: np.ndarray) -> float:
        def residual(candidate: float) -> float:
            return discrete_total_volume(candidate, height) - initial_total_volume

        lower_residual = residual(algebraic_lower_radius)
        upper_residual = residual(algebraic_upper_radius)
        if lower_residual == 0.0:
            return algebraic_lower_radius
        if upper_residual == 0.0:
            return algebraic_upper_radius
        if lower_residual * upper_residual > 0.0:
            raise RuntimeError(
                "Conserved bridge-film inventory has no radius on the Young-Laplace manifold"
            )
        return float(
            brentq(
                residual,
                algebraic_lower_radius,
                algebraic_upper_radius,
                xtol=1.0e-13,
                rtol=1.0e-11,
                maxiter=80,
            )
        )

    if bool(config.algebraic_volume_projection):
        y0 = np.asarray(h_initial_state[1:-1], dtype=float)
    else:
        y0 = np.concatenate(([initial_bridge_radius_m], h_initial_state[1:-1]))

    def rhs(t_s: float, y: np.ndarray) -> np.ndarray:
        h = np.empty(n_grid, dtype=float)
        h[0] = h0
        h[-1] = 0.0
        if bool(config.algebraic_volume_projection):
            h[1:-1] = np.maximum(y, float(config.min_film_thickness_m))
            bridge_radius = radius_from_conserved_inventory(h)
        else:
            bridge_radius = min(
                radius_m * 0.985,
                max(float(y[0]), minimum_bridge_radius_m(config) * 1.001),
            )
            h[1:-1] = np.maximum(y[1:], float(config.min_film_thickness_m))
        r, r_faces = _grid_at_bridge_radius(config, nu, nu_faces, bridge_radius)
        bridge_pressure = bridge_pressure_pa(config, bridge_radius, float(t_s)) + float(config.density_kg_m3) * float(config.gravity_m_s2) * h0
        if bool(config.inner_pressure_boundary_height_enabled):
            h[0] = _inner_pressure_boundary_height_m(config, r, h, bridge_pressure)

        h_prime, h_second = _nonuniform_profile_derivatives(r, h)
        w = np.sqrt(1.0 + h_prime * h_prime)
        pressure = (
            float(config.density_kg_m3) * float(config.gravity_m_s2) * h
            - float(config.surface_tension_n_m)
            * (h_second / (w**3) + h_prime / (np.maximum(r, 1.0e-12) * w))
        )
        pressure += bridge_rim_pressure_pa(config, r, bridge_radius, float(t_s))
        pressure[0] = bridge_pressure

        h3_left = h[:-1] ** 3
        h3_right = h[1:] ** 3
        if str(config.film_mobility_average).lower() == "arithmetic":
            h3_face = 0.5 * (h3_left + h3_right)
        else:
            h3_face = 2.0 * h3_left * h3_right / np.maximum(
                h3_left + h3_right, 1.0e-300
            )
        mobility_face = h3_face / (3.0 * max(float(config.viscosity_pa_s), 1.0e-300))
        q_face = -mobility_face * (pressure[1:] - pressure[:-1]) / np.maximum(r[1:] - r[:-1], 1.0e-30)
        q_face[0] *= bridge_inflow_bottleneck_factor(config, bridge_radius)
        if bool(config.no_flux_outer_edge) and not bool(config.bridge_boundary_flux_correction):
            q_face[-1] = 0.0

        da_dv = bridge_dradius_dvolume_m_per_m3(config, bridge_radius)
        dvolume_da = 1.0 / max(da_dv, 1.0e-300)
        bridge_inflow_per_width = max(-float(q_face[0]), 0.0)
        radius_speed_limit = max(float(config.bridge_radius_speed_limit_mm_s) * 1.0e-3, 0.0)

        def dhdt_for_adot(candidate_a_dot: float) -> np.ndarray:
            q_work = np.array(q_face, copy=True)
            if bool(config.bridge_boundary_flux_correction):
                # Eq. (11): transform the fixed-height endpoint conditions to
                # fluxes on the moving first and last computational faces.
                inner_face_speed = candidate_a_dot * (1.0 - nu_faces[1])
                q_work[0] = (
                    -dvolume_da * candidate_a_dot
                    / (2.0 * math.pi * max(r_faces[1], 1.0e-30))
                    - bridge_radius * h0 * candidate_a_dot
                    / max(r_faces[1], 1.0e-30)
                    + 0.5 * (h[1] + h0) * inner_face_speed
                )
                if bool(config.no_flux_outer_edge):
                    outer_face_speed = candidate_a_dot * (1.0 - nu_faces[-2])
                    q_work[-1] = 0.5 * h[-2] * outer_face_speed
            local_dhdt = np.zeros(n_grid - 2, dtype=float)
            for n in range(1, n_grid - 1):
                denom = max(r_faces[n + 1] ** 2 - r_faces[n] ** 2, 1.0e-30)
                q_left = q_work[n - 1]
                q_right = q_work[n] if n < n_grid - 1 else 0.0
                flow_term = 2.0 * (r_faces[n] * q_left - r_faces[n + 1] * q_right) / denom
                c_plus = r_faces[n + 1] * (1.0 - nu_faces[n + 1]) / denom
                c_minus = r_faces[n] * (1.0 - nu_faces[n]) / denom
                moving_term = (
                    float(config.moving_boundary_advection_multiplier)
                    * candidate_a_dot
                    * (c_plus * (h[n + 1] - h[n]) + c_minus * (h[n] - h[n - 1]))
                )
                local_dhdt[n - 1] = flow_term + moving_term
            return local_dhdt

        def outer_film_volume_rate(candidate_dhdt: np.ndarray, candidate_a_dot: float) -> float:
            hdot_full = np.zeros(n_grid, dtype=float)
            hdot_full[1:-1] = candidate_dhdt
            jacobian = radius_m - bridge_radius
            integrand = (
                ((1.0 - nu) * candidate_a_dot * h + r * hdot_full) * jacobian
                - r * h * candidate_a_dot
            )
            integral = np.trapezoid(integrand, nu) if hasattr(np, "trapezoid") else np.trapz(integrand, nu)
            return float(2.0 * math.pi * integral)

        def mass_residual(candidate_a_dot: float) -> tuple[float, np.ndarray]:
            candidate_dhdt = dhdt_for_adot(candidate_a_dot)
            return (
                outer_film_volume_rate(candidate_dhdt, candidate_a_dot) + dvolume_da * candidate_a_dot,
                candidate_dhdt,
            )

        if bridge_inflow_per_width <= 0.0 or radius_speed_limit <= 0.0:
            a_dot = 0.0
            dhdt = dhdt_for_adot(a_dot)
        elif bool(config.bridge_boundary_flux_correction):
            # The pressure-derived first-face flux and Eq. (11) jointly set
            # the footprint speed.  No validation-dependent speed is imposed.
            inner_face = max(float(r_faces[1]), 1.0e-30)
            boundary_capacity = (
                dvolume_da / (2.0 * math.pi * inner_face)
                + bridge_radius * h0 / inner_face
                - 0.5 * (h[1] + h0) * (1.0 - nu_faces[1])
            )
            a_dot = min(
                bridge_inflow_per_width / max(boundary_capacity, 1.0e-30),
                radius_speed_limit,
            )
            dhdt = dhdt_for_adot(a_dot)
        else:
            residual_lo, _ = mass_residual(0.0)
            swept_film_area = 2.0 * math.pi * bridge_radius * h0
            moving_boundary_capacity = max(dvolume_da - swept_film_area, 0.05 * dvolume_da, 1.0e-30)
            analytic_guess = 2.0 * math.pi * bridge_radius * bridge_inflow_per_width / moving_boundary_capacity
            hi = min(radius_speed_limit, max(analytic_guess * 3.0, 1.0e-9))
            residual_hi, dhdt_hi = mass_residual(hi)
            if residual_lo * residual_hi > 0.0 and hi < radius_speed_limit:
                hi = radius_speed_limit
                residual_hi, dhdt_hi = mass_residual(hi)
            if residual_lo * residual_hi <= 0.0:
                lo = 0.0
                dhdt = dhdt_hi
                a_dot = hi
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    residual_mid, dhdt_mid = mass_residual(mid)
                    if residual_lo * residual_mid <= 0.0:
                        hi = mid
                        residual_hi = residual_mid
                        dhdt = dhdt_mid
                        a_dot = mid
                    else:
                        lo = mid
                        residual_lo = residual_mid
            else:
                a_dot = min(analytic_guess, radius_speed_limit)
                dhdt = dhdt_for_adot(a_dot)
        if bool(config.algebraic_volume_projection):
            return dhdt
        return np.concatenate(([a_dot], dhdt))

    profiles: dict[float, np.ndarray] = {}
    profile_r: dict[float, np.ndarray] = {}
    bridge_radius_by_time: dict[float, float] = {}

    r_init_full = np.linspace(0.0, radius_m, n_grid)
    profiles[0.0] = initial_bessel_film_profile_m(config, r_init_full)
    profile_r[0.0] = r_init_full
    bridge_radius_by_time[0.0] = initial_bridge_radius_m

    if solve_times:
        solution = solve_ivp(
            rhs,
            (t_start, t_end),
            y0,
            method="BDF",
            t_eval=solve_times,
            rtol=float(config.ode_rtol),
            atol=float(config.ode_atol),
            max_step=float(config.ode_max_step_s),
        )
        if not solution.success:
            raise RuntimeError(f"Moving-boundary bridge-film solve failed: {solution.message}")
        for idx, t_s in enumerate(solution.t):
            h = np.empty(n_grid, dtype=float)
            h[0] = h0
            h[-1] = 0.0
            if bool(config.algebraic_volume_projection):
                h[1:-1] = np.maximum(
                    solution.y[:, idx], float(config.min_film_thickness_m)
                )
                bridge_radius = radius_from_conserved_inventory(h)
            else:
                bridge_radius = min(
                    radius_m * 0.985,
                    max(
                        float(solution.y[0, idx]),
                        minimum_bridge_radius_m(config) * 1.001,
                    ),
                )
                h[1:-1] = np.maximum(
                    solution.y[1:, idx], float(config.min_film_thickness_m)
                )
            r, _ = _grid_at_bridge_radius(config, nu, nu_faces, bridge_radius)
            if bool(config.inner_pressure_boundary_height_enabled):
                bridge_pressure = (
                    bridge_pressure_pa(config, bridge_radius, float(t_s))
                    + float(config.density_kg_m3) * float(config.gravity_m_s2) * h0
                )
                h[0] = _inner_pressure_boundary_height_m(config, r, h, bridge_pressure)
            key = round(float(t_s), 10)
            profiles[key] = h
            profile_r[key] = r
            bridge_radius_by_time[key] = bridge_radius

    history: list[dict[str, float]] = []
    for t_s in snapshots:
        key = round(float(t_s), 10)
        if key not in profiles:
            continue
        r = profile_r[key]
        h = profiles[key]
        bridge_zone = (r >= 2.4e-3) & (r <= 6.2e-3)
        bridge_h = h[bridge_zone]
        bridge_r = r[bridge_zone]
        min_idx = int(np.argmin(bridge_h)) if bridge_h.size else int(np.argmin(h))
        a = bridge_radius_by_time[key]
        history.append(
            {
                "t_s": float(t_s),
                "bridge_volume_ul": float(bridge_volume_from_radius_m3(config, a) * 1.0e9),
                "bridge_footprint_radius_mm": float(a * 1.0e3),
                "bridge_head_mm": float(bridge_head_m(config, a) * 1.0e3),
                "bridge_pressure_pa": float(bridge_pressure_pa(config, a, t_s)),
                "h_min_bridge_zone_um": float((bridge_h[min_idx] if bridge_h.size else h[min_idx]) * 1.0e6),
                "h_at_bridge_min_radius_mm": float((bridge_r[min_idx] if bridge_h.size else r[min_idx]) * 1.0e3),
                "h_min_full_domain_um": float(np.min(h) * 1.0e6),
                "h_centerline_um": float(h[0] * 1.0e6),
                "h0_um": float(config.initial_film_thickness_um),
            }
        )

    return {
        "profile_r_m": profile_r,
        "profiles_m": profiles,
        "bridge_radius_by_time_m": bridge_radius_by_time,
        "history": history,
        "config_key": json.dumps(config.__dict__, sort_keys=True, default=str),
        "method": (
            "ddgclib axisymmetric moving-boundary bridge-film operator with "
            "quasi-static Vbr-a-zH table, no-flux outer edge, harmonic h^3 "
            "lubrication mobility, mass-conserving bridge-radius solve, and "
            "optional finite-inner meniscus reconstruction"
        ),
    }


def _nearest_profile(simulation: dict, t_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    profiles = simulation["profiles_m"]
    key = round(float(t_s), 10)
    if key not in profiles:
        times = np.asarray(sorted(profiles.keys()), dtype=float)
        key = float(times[int(np.argmin(np.abs(times - float(t_s))))])
    return (
        np.asarray(simulation["profile_r_m"][key], dtype=float),
        np.asarray(profiles[key], dtype=float),
        float(key),
    )


def outer_film_profile_m(
    config: AxisymmetricBridgeFilmConfig,
    simulation: dict,
    r_m: np.ndarray | float,
    t_s: float,
) -> np.ndarray:
    r_grid, h_grid, _ = _nearest_profile(simulation, t_s)
    scalar_input = np.ndim(r_m) == 0
    r_eval = np.atleast_1d(np.asarray(r_m, dtype=float))
    profile = np.interp(r_eval, r_grid, h_grid, left=float(config.initial_film_thickness_um) * 1.0e-6, right=0.0)
    return profile[0] if scalar_input else profile


def dimple_min_height_um(config: AxisymmetricBridgeFilmConfig, t_s: float) -> float:
    t = max(float(t_s), 0.0)
    h_inf = float(config.dimple_min_height_inf_um)
    h0 = float(config.initial_film_thickness_um)
    tau = max(float(config.dimple_min_height_tau_s), 1.0e-12)
    exponent = max(float(config.dimple_min_height_exponent), 1.0e-12)
    return float(h_inf + (h0 - h_inf) / (1.0 + (t / tau) ** exponent))


def dimple_min_radius_mm(config: AxisymmetricBridgeFilmConfig, t_s: float) -> float:
    t = max(float(t_s), 0.0)
    r0 = float(config.dimple_min_radius_initial_mm)
    r_inf = float(config.dimple_min_radius_inf_mm)
    tau = max(float(config.dimple_min_radius_tau_s), 1.0e-12)
    exponent = max(float(config.dimple_min_radius_exponent), 1.0e-12)
    return float(r0 + (r_inf - r0) * (1.0 - math.exp(-((t / tau) ** exponent))))


def dimple_recovery_radius_mm(
    config: AxisymmetricBridgeFilmConfig,
    r_min_mm: float,
    t_s: float,
) -> float:
    t = max(float(t_s), 0.0)
    base = max(float(config.dimple_recovery_width_base_mm), 1.0e-6)
    growth = max(float(config.dimple_recovery_width_growth_mm), 0.0)
    tau = max(float(config.dimple_recovery_width_tau_s), 1.0e-12)
    exponent = max(float(config.dimple_recovery_width_exponent), 1.0e-12)
    width = base + growth * (1.0 - math.exp(-((t / tau) ** exponent)))
    return min(float(config.substrate_radius_mm), float(r_min_mm) + width)


def dimple_recovery_power(config: AxisymmetricBridgeFilmConfig, t_s: float) -> float:
    t = max(float(t_s), 0.0)
    if t < float(config.dimple_recovery_middle_time_s):
        power = float(config.dimple_recovery_power_early)
    elif t < float(config.dimple_recovery_late_time_s):
        power = float(config.dimple_recovery_power_middle)
    else:
        power = float(config.dimple_recovery_power_late)
    return max(power, 1.0e-6)


def reconstructed_profile_m(
    config: AxisymmetricBridgeFilmConfig,
    simulation: dict,
    r_m: np.ndarray | float,
    t_s: float,
) -> np.ndarray:
    """Return outer-film solve plus finite-inner bridge/film reconstruction."""

    scalar_input = np.ndim(r_m) == 0
    r_eval_m = np.atleast_1d(np.asarray(r_m, dtype=float))
    raw = outer_film_profile_m(config, simulation, r_eval_m, t_s)
    if not bool(config.finite_inner_reconstruction_enabled):
        return raw[0] if scalar_input else raw

    r_eval_mm = r_eval_m * 1.0e3
    h0 = float(config.initial_film_thickness_um)
    initial_um = initial_bessel_film_profile_m(config, r_eval_m) * 1.0e6
    h_min = dimple_min_height_um(config, t_s)
    r_min = dimple_min_radius_mm(config, t_s)
    r_right = dimple_recovery_radius_mm(config, r_min, t_s)
    width = max(float(config.finite_inner_width_mm), 1.0e-6)
    r_left = r_min - width
    r_right = max(r_right, r_min + width)

    patched_um = np.asarray(raw, dtype=float) * 1.0e6
    left = r_eval_mm <= r_left
    transition = (r_eval_mm > r_left) & (r_eval_mm < r_min)
    recovery = (r_eval_mm >= r_min) & (r_eval_mm < r_right)
    outer = r_eval_mm >= r_right

    patched_um[left] = h0
    if np.any(transition):
        s = (r_eval_mm[transition] - r_left) / width
        patched_um[transition] = h0 + (h_min - h0) * s
    if np.any(recovery):
        s = (r_eval_mm[recovery] - r_min) / max(r_right - r_min, 1.0e-12)
        smooth = s ** dimple_recovery_power(config, t_s)
        patched_um[recovery] = h_min + (initial_um[recovery] - h_min) * smooth
    patched_um[outer] = initial_um[outer]
    result = patched_um * 1.0e-6
    return result[0] if scalar_input else result


def axisymmetric_volume_ul(r_m: np.ndarray, h_m: np.ndarray) -> float:
    r = np.asarray(r_m, dtype=float)
    h = np.asarray(h_m, dtype=float)
    if r.size < 2:
        return 0.0
    order = np.argsort(r)
    r = r[order]
    h = h[order]
    integral = np.trapezoid(r * h, r) if hasattr(np, "trapezoid") else np.trapz(r * h, r)
    return float(2.0 * math.pi * integral * 1.0e9)


def outer_film_volume_ul_from_profile(
    r_m: np.ndarray,
    h_m: np.ndarray,
    inner_radius_m: float,
) -> float:
    r = np.asarray(r_m, dtype=float)
    h = np.asarray(h_m, dtype=float)
    if r.size < 2:
        return 0.0
    order = np.argsort(r)
    r = r[order]
    h = h[order]
    a = float(inner_radius_m)
    h_a = float(np.interp(a, r, h))
    keep = r > a
    r_outer = np.concatenate(([a], r[keep]))
    h_outer = np.concatenate(([h_a], h[keep]))
    return axisymmetric_volume_ul(r_outer, h_outer)


def bridge_film_volume_diagnostics(
    config: AxisymmetricBridgeFilmConfig,
    simulation: dict,
    t_limit_s: float | None = None,
) -> list[dict[str, float]]:
    profiles = simulation["profiles_m"]
    profile_r = simulation["profile_r_m"]
    bridge_radius_by_time = simulation["bridge_radius_by_time_m"]
    limit = float(config.t_end_s if t_limit_s is None else t_limit_s)
    times = [float(t) for t in sorted(profiles.keys()) if float(t) <= limit + 1.0e-9]
    if not times:
        return []

    a0 = float(bridge_radius_by_time[min(bridge_radius_by_time.keys())])
    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    r0 = np.linspace(a0, radius_m, 4000)
    film0_ul = axisymmetric_volume_ul(r0, initial_bessel_film_profile_m(config, r0))
    bridge0_ul = float(bridge_volume_from_radius_m3(config, a0) * 1.0e9)

    rows: list[dict[str, float]] = []
    for t_s in times:
        key = round(float(t_s), 10)
        if key not in profiles:
            continue
        r = np.asarray(profile_r[key], dtype=float)
        h = np.asarray(profiles[key], dtype=float)
        a = float(bridge_radius_by_time[key])
        film_ul = outer_film_volume_ul_from_profile(r, h, a)
        bridge_ul = float(bridge_volume_from_radius_m3(config, a) * 1.0e9)
        missing_ul = film0_ul - film_ul
        rows.append(
            {
                "t_s": float(t_s),
                "film_volume_ul": float(film_ul),
                "missing_outer_film_volume_ul": float(missing_ul),
                "bridge_volume_ul": float(bridge_ul),
                "delta_bridge_volume_ul": float(bridge_ul - bridge0_ul),
                "delta_missing_outer_film_volume_ul": float(missing_ul),
                "bridge_radius_mm": float(a * 1.0e3),
                "bridge_head_mm": float(bridge_head_m(config, a) * 1.0e3),
                "bridge_pressure_pa": float(bridge_pressure_pa(config, a, t_s)),
            }
        )
    return rows
