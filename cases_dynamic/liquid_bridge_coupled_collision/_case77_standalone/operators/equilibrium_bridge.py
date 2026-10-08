"""Young-Laplace equilibrium manifolds for sphere-film capillary bridges.

The operator solves the axisymmetric arc-length equations using only geometry,
material properties, and total liquid volume.  It is independent of any
validation curve and provides the quasi-static bridge pressure and footprint
used by a coupled film or three-dimensional mesh calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import i0, i1


@dataclass(frozen=True)
class EquilibriumBridgeManifold:
    hypothetical_film_height_m: np.ndarray
    bridge_volume_m3: np.ndarray
    footprint_radius_m: np.ndarray
    contact_radius_m: np.ndarray
    contact_angle_rad: np.ndarray
    suction_pressure_pa: np.ndarray
    hydraulic_head_m: np.ndarray
    substrate_contact_radius_m: np.ndarray
    # Optional fixed-sphere meridian segments, sampled on a common normalized
    # arc coordinate from the sphere contact line to the h=h0 footprint.
    # They let a 3-D tetra solver recover the same zero-angle equilibrium
    # shape used to construct the scalar V-P-a lookup table.
    profile_radius_m: np.ndarray | None = None
    profile_height_m: np.ndarray | None = None
    profile_tangent_angle_rad: np.ndarray | None = None


def _sphere_volume_integral(contact_radius_m: float, sphere_radius_m: float) -> float:
    radius = np.linspace(0.0, float(contact_radius_m), 800)
    sphere = float(sphere_radius_m)
    height = sphere - np.sqrt(np.maximum(sphere * sphere - radius * radius, 0.0))
    return float(np.trapezoid(math.pi * radius * radius, height))


def _initial_film_volume(
    height_m: float,
    substrate_radius_m: float,
    capillary_length_m: float,
) -> float:
    ratio = float(substrate_radius_m) / float(capillary_length_m)
    conversion = (
        ratio * float(i0(ratio)) - 2.0 * float(i1(ratio))
    ) / max(ratio * (float(i0(ratio)) - 1.0), 1.0e-30)
    return float(conversion * math.pi * float(substrate_radius_m) ** 2 * float(height_m))


def build_equilibrium_bridge_manifold(
    *,
    sphere_radius_m: float,
    substrate_radius_m: float,
    maximum_film_height_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    minimum_film_height_m: float = 0.1e-6,
    samples: int = 72,
) -> EquilibriumBridgeManifold:
    """Compute the partial-wetting Young-Laplace bridge family.

    Pressure, arc length, and the sphere contact angle are solved together.
    The outer liquid contact on the flat substrate is horizontal and its
    radius is an output.  This is the physical branch used while that radius
    remains inside the substrate edge, as it does for the 100 um Siekman case.
    """

    sphere = float(sphere_radius_m)
    substrate = float(substrate_radius_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)
    gamma = float(surface_tension_n_m)
    capillary_length = math.sqrt(gamma / max(rho_g, 1.0e-30))
    maximum_height = float(maximum_film_height_m)
    minimum_height = max(float(minimum_film_height_m), 1.0e-10)
    if not 0.0 < minimum_height <= maximum_height:
        raise ValueError("film-height bounds must satisfy 0 < minimum <= maximum")

    collocation = np.linspace(0.0, 1.0, 700)
    theta_seed = 0.80
    contact_seed = sphere * math.sin(theta_seed)
    contact_z_seed = maximum_height + sphere * (1.0 - math.cos(theta_seed))
    endpoint_seed = min(0.45 * substrate, contact_seed + 5.0 * capillary_length)
    parameter = np.linspace(0.0, 1.0, len(collocation))
    radius_seed = contact_seed + (endpoint_seed - contact_seed) * (
        3.0 * parameter**2 - 2.0 * parameter**3
    )
    height_seed = contact_z_seed * (1.0 - parameter) ** 2
    tangent_seed = np.unwrap(
        np.arctan2(
            np.gradient(height_seed, parameter),
            np.gradient(radius_seed, parameter),
        )
    )
    radius_nd = radius_seed / capillary_length
    height_nd = height_seed / capillary_length
    integral_seed = np.zeros_like(parameter)
    integral_seed[1:] = np.cumsum(
        0.5
        * math.pi
        * (radius_nd[1:] ** 2 + radius_nd[:-1] ** 2)
        * np.diff(height_nd)
    )
    guess = np.vstack((radius_nd, height_nd, tangent_seed, integral_seed))
    solver_parameters = np.asarray(
        (0.6, math.log(max(endpoint_seed / capillary_length, 1.0e-12)), theta_seed),
        dtype=float,
    )

    heights_descending = np.geomspace(
        maximum_height, minimum_height, max(int(samples), 20)
    )
    records: list[tuple[float, float, float, float, float, float, float, float]] = []
    previous_height = maximum_height

    for index, film_height in enumerate(heights_descending):
        target_total_nd = _initial_film_volume(
            float(film_height), substrate, capillary_length
        ) / capillary_length**3

        def ode(_x: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
            pressure_nd = float(params[0])
            arc_nd = math.exp(float(np.clip(params[1], -30.0, 10.0)))
            radius = np.maximum(state[0], 1.0e-10)
            tangent = state[2]
            return np.vstack(
                (
                    arc_nd * np.cos(tangent),
                    arc_nd * np.sin(tangent),
                    arc_nd * (pressure_nd + state[1] - np.sin(tangent) / radius),
                    arc_nd * math.pi * radius * radius * np.sin(tangent),
                )
            )

        def boundary(
            left: np.ndarray, right: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            theta = float(params[2])
            contact = sphere * math.sin(theta)
            contact_z = float(film_height) + sphere * (1.0 - math.cos(theta))
            sphere_integral_nd = _sphere_volume_integral(
                contact, sphere
            ) / capillary_length**3
            return np.asarray(
                (
                    left[0] - contact / capillary_length,
                    left[1] - contact_z / capillary_length,
                    left[2] - (theta - math.pi),
                    left[3],
                    right[1],
                    right[2],
                    right[3] - (-target_total_nd - sphere_integral_nd),
                ),
                dtype=float,
            )

        if index:
            guess = np.asarray(result.sol(collocation), dtype=float)
            guess[1] += (
                (float(film_height) - float(previous_height))
                / capillary_length
                * (1.0 - collocation)
            )
            guess[3] *= float(film_height) / float(previous_height)
        result = solve_bvp(
            ode,
            boundary,
            collocation,
            guess,
            p=solver_parameters,
            tol=3.0e-5,
            max_nodes=100000,
        )
        if not bool(result.success):
            raise RuntimeError(
                f"Equilibrium bridge continuation failed at h={film_height:g} m: "
                f"{result.message}"
            )
        solver_parameters = np.asarray(result.p, dtype=float)
        previous_height = float(film_height)
        dense = np.asarray(result.sol(np.linspace(0.0, 1.0, 6000)), dtype=float)
        radius = capillary_length * dense[0]
        height = capillary_length * dense[1]
        theta = float(result.p[2])
        contact = sphere * math.sin(theta)
        crossing = np.flatnonzero(
            (height[:-1] >= float(film_height))
            & (height[1:] < float(film_height))
        )
        if crossing.size == 0:
            raise RuntimeError("Equilibrium bridge did not cross its footprint height")
        stop = int(crossing[0]) + 1
        denominator = float(height[stop] - height[stop - 1])
        if abs(denominator) <= 1.0e-30:
            denominator = -1.0e-30
        fraction = (float(film_height) - height[stop - 1]) / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        footprint = radius[stop - 1] + fraction * (radius[stop] - radius[stop - 1])
        segment_r = np.concatenate((radius[:stop], (footprint,)))
        segment_z = np.concatenate((height[:stop], (float(film_height),)))
        free_integral = float(np.trapezoid(math.pi * segment_r**2, segment_z))
        bridge_volume = -(
            free_integral + _sphere_volume_integral(contact, sphere)
        )
        pressure_scale = gamma / capillary_length
        pressure_magnitude = float(result.p[0]) * pressure_scale
        suction = -(pressure_magnitude - rho_g * float(film_height))
        records.append(
            (
                float(film_height),
                float(bridge_volume),
                float(footprint),
                float(contact),
                float(theta),
                float(suction),
                float(-suction / rho_g),
                float(radius[-1]),
            )
        )

    data = np.asarray(records[::-1], dtype=float)
    if np.any(np.diff(data[:, 1]) <= 0.0):
        raise RuntimeError("Equilibrium bridge volume is not monotone")
    if np.any(data[:, 7] > substrate * (1.0 + 1.0e-6)):
        raise RuntimeError(
            "Equilibrium family reached the substrate edge; pinned-edge continuation is required"
        )
    return EquilibriumBridgeManifold(
        hypothetical_film_height_m=data[:, 0],
        bridge_volume_m3=data[:, 1],
        footprint_radius_m=data[:, 2],
        contact_radius_m=data[:, 3],
        contact_angle_rad=data[:, 4],
        suction_pressure_pa=data[:, 5],
        hydraulic_head_m=data[:, 6],
        substrate_contact_radius_m=data[:, 7],
    )


def build_fixed_sphere_equilibrium_bridge_manifold(
    *,
    sphere_radius_m: float,
    sphere_tip_height_m: float,
    substrate_radius_m: float,
    maximum_hypothetical_film_height_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    minimum_hypothetical_film_height_m: float = 0.1e-6,
    samples: int = 72,
) -> EquilibriumBridgeManifold:
    """Young--Laplace family with the experimental sphere position fixed.

    Siekman et al.'s transient closure varies a *hypothetical total oil
    volume* while the real sphere apex remains at ``sphere_tip_height_m``.
    The original continuation above instead translates the sphere with that
    hypothetical height.  This variant keeps the sphere fixed, extracts the
    bridge volume above the real initial-film plane, and records the radius at
    which the equilibrium surface crosses that plane.
    """

    sphere = float(sphere_radius_m)
    tip = float(sphere_tip_height_m)
    substrate = float(substrate_radius_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)
    gamma = float(surface_tension_n_m)
    capillary_length = math.sqrt(gamma / max(rho_g, 1.0e-30))
    maximum_height = float(maximum_hypothetical_film_height_m)
    minimum_height = max(float(minimum_hypothetical_film_height_m), 1.0e-10)
    if not 0.0 < minimum_height <= maximum_height:
        raise ValueError("hypothetical film-height bounds must satisfy 0 < minimum <= maximum")

    collocation = np.linspace(0.0, 1.0, 700)
    theta_seed = 0.80
    contact_seed = sphere * math.sin(theta_seed)
    contact_z_seed = tip + sphere * (1.0 - math.cos(theta_seed))
    endpoint_seed = min(0.45 * substrate, contact_seed + 5.0 * capillary_length)
    parameter = np.linspace(0.0, 1.0, len(collocation))
    radius_seed = contact_seed + (endpoint_seed - contact_seed) * (
        3.0 * parameter**2 - 2.0 * parameter**3
    )
    height_seed = contact_z_seed * (1.0 - parameter) ** 2
    tangent_seed = np.unwrap(
        np.arctan2(
            np.gradient(height_seed, parameter),
            np.gradient(radius_seed, parameter),
        )
    )
    radius_nd = radius_seed / capillary_length
    height_nd = height_seed / capillary_length
    integral_seed = np.zeros_like(parameter)
    integral_seed[1:] = np.cumsum(
        0.5
        * math.pi
        * (radius_nd[1:] ** 2 + radius_nd[:-1] ** 2)
        * np.diff(height_nd)
    )
    guess = np.vstack((radius_nd, height_nd, tangent_seed, integral_seed))
    solver_parameters = np.asarray(
        (0.6, math.log(max(endpoint_seed / capillary_length, 1.0e-12)), theta_seed),
        dtype=float,
    )

    heights_descending = np.geomspace(
        maximum_height, minimum_height, max(int(samples), 20)
    )
    records: list[tuple[float, float, float, float, float, float, float, float]] = []
    profile_records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    previous_height = maximum_height
    result = None
    for index, hypothetical_height in enumerate(heights_descending):
        target_total_nd = _initial_film_volume(
            float(hypothetical_height), substrate, capillary_length
        ) / capillary_length**3

        def ode(_x: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
            pressure_nd = float(params[0])
            arc_nd = math.exp(float(np.clip(params[1], -30.0, 10.0)))
            radius = np.maximum(state[0], 1.0e-10)
            tangent = state[2]
            return np.vstack(
                (
                    arc_nd * np.cos(tangent),
                    arc_nd * np.sin(tangent),
                    arc_nd * (pressure_nd + state[1] - np.sin(tangent) / radius),
                    arc_nd * math.pi * radius * radius * np.sin(tangent),
                )
            )

        def boundary(
            left: np.ndarray, right: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            theta = float(params[2])
            contact = sphere * math.sin(theta)
            contact_z = tip + sphere * (1.0 - math.cos(theta))
            sphere_integral_nd = _sphere_volume_integral(
                contact, sphere
            ) / capillary_length**3
            return np.asarray(
                (
                    left[0] - contact / capillary_length,
                    left[1] - contact_z / capillary_length,
                    left[2] - (theta - math.pi),
                    left[3],
                    right[1],
                    right[2],
                    right[3] - (-target_total_nd - sphere_integral_nd),
                ),
                dtype=float,
            )

        if result is not None:
            guess = np.asarray(result.sol(collocation), dtype=float)
            # Continue only in total volume.  Translating this profile would
            # reintroduce the moving-sphere error this function removes.
            guess[3] *= float(hypothetical_height) / float(previous_height)
        result = solve_bvp(
            ode,
            boundary,
            collocation,
            guess,
            p=solver_parameters,
            tol=3.0e-5,
            max_nodes=100000,
        )
        if not bool(result.success):
            raise RuntimeError(
                "Fixed-sphere equilibrium continuation failed at "
                f"h_l={hypothetical_height:g} m: {result.message}"
            )
        solver_parameters = np.asarray(result.p, dtype=float)
        previous_height = float(hypothetical_height)
        dense = np.asarray(result.sol(np.linspace(0.0, 1.0, 6000)), dtype=float)
        radius = capillary_length * dense[0]
        height = capillary_length * dense[1]
        theta = float(result.p[2])
        contact = sphere * math.sin(theta)
        crossing = np.flatnonzero(
            (height[:-1] >= tip) & (height[1:] < tip)
        )
        if crossing.size == 0:
            continue
        stop = int(crossing[0]) + 1
        denominator = float(height[stop] - height[stop - 1])
        if abs(denominator) <= 1.0e-30:
            denominator = -1.0e-30
        fraction = float(
            np.clip((tip - height[stop - 1]) / denominator, 0.0, 1.0)
        )
        footprint = radius[stop - 1] + fraction * (
            radius[stop] - radius[stop - 1]
        )
        segment_r = np.concatenate((radius[:stop], (footprint,)))
        segment_z = np.concatenate((height[:stop], (tip,)))
        crossing_tangent = dense[2, stop - 1] + fraction * (
            dense[2, stop] - dense[2, stop - 1]
        )
        segment_tangent = np.concatenate(
            (dense[2, :stop], (crossing_tangent,))
        )
        segment_length = np.concatenate(
            (
                (0.0,),
                np.cumsum(
                    np.hypot(np.diff(segment_r), np.diff(segment_z))
                ),
            )
        )
        common_arc = np.linspace(0.0, float(segment_length[-1]), 96)
        profile_records.append(
            (
                np.interp(common_arc, segment_length, segment_r),
                np.interp(common_arc, segment_length, segment_z),
                np.interp(common_arc, segment_length, segment_tangent),
            )
        )
        bridge_volume = -(
            float(np.trapezoid(math.pi * segment_r**2, segment_z))
            + _sphere_volume_integral(contact, sphere)
        )
        pressure_at_substrate = float(result.p[0]) * gamma / capillary_length
        pressure_at_film_plane = pressure_at_substrate - rho_g * tip
        suction_pressure = -pressure_at_film_plane
        records.append(
            (
                float(hypothetical_height),
                float(bridge_volume),
                float(footprint),
                float(contact),
                float(theta),
                float(suction_pressure),
                float(-suction_pressure / rho_g),
                float(radius[-1]),
            )
        )

    data = np.asarray(records, dtype=float)
    order = np.argsort(data[:, 1])
    data = data[order]
    profile_radius = np.asarray([profile_records[index][0] for index in order])
    profile_height = np.asarray([profile_records[index][1] for index in order])
    profile_tangent = np.asarray([profile_records[index][2] for index in order])
    if len(data) < 20 or np.any(np.diff(data[:, 1]) <= 0.0):
        raise RuntimeError("Fixed-sphere bridge volume is not a usable monotone family")
    return EquilibriumBridgeManifold(
        hypothetical_film_height_m=data[:, 0],
        bridge_volume_m3=data[:, 1],
        footprint_radius_m=data[:, 2],
        contact_radius_m=data[:, 3],
        contact_angle_rad=data[:, 4],
        suction_pressure_pa=data[:, 5],
        hydraulic_head_m=data[:, 6],
        substrate_contact_radius_m=data[:, 7],
        profile_radius_m=profile_radius,
        profile_height_m=profile_height,
        profile_tangent_angle_rad=profile_tangent,
    )
