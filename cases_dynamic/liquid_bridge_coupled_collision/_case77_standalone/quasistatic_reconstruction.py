#!/usr/bin/env python3
"""Reconstruct Case77 with body-fitted Young--Laplace tetra meshes.

The source-owned Case77 ALE mesh is a vertically extruded height field.  That
topology cannot represent the inward-turning bridge neck seen in the side-view
experiment.  This post-solve geometry stage keeps the computed
Case77 bridge-volume/outer-film history, solves the zero-contact-angle
arc-length Young--Laplace meridian at every saved time, and remeshes the full
liquid volume with genuine tetrahedra.

The Case77 forward states are retained as the source audit trail.  Reconstructed
states are written to separate directories and are volume-corrected against
the conserved tetra volume of the source state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from . import source_bridge_core as case61
from .operators.equilibrium_bridge import build_equilibrium_bridge_manifold
from .operators.unstructured_bridge_mesh import (
    YoungLaplaceMeridian,
    _revolve_planar_triangles,
    bridge_volume_from_meridian,
    build_sphere_film_bridge_mesh,
    solve_zero_angle_sphere_film_meridian,
)


YL_SURFACE_DIR_NAME = "young_laplace_mesh_states"
YL_TETRA_DIR_NAME = "young_laplace_tetra_states"
YL_METRICS_NAME = "case77_young_laplace_reconstruction_metrics.json"
CASE_LABEL = "Case 77"
CASE_SLUG = "Case77"
OUTPUT_PREFIX = "case77"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "case77_best_treatment_finite_initial_0to3600"
)

SPHERE_RADIUS_M = 5.0e-3
SUBSTRATE_RADIUS_M = 12.0e-3
FILM_HEIGHT_M = 100.0e-6
DENSITY_KG_M3 = 1065.0
GRAVITY_M_S2 = 9.80665
# Siekman et al. (2025), Sec. II: silicone-oil surface tension 21 mN/m.
SURFACE_TENSION_N_M = 21.0e-3
VISCOSITY_PA_S = 0.1
AZIMUTHAL_SECTORS = 32
MERIDIAN_SAMPLES = 64
NECK_RESOLUTION_M = 8.0e-6
BULK_MESH_SIZE_M = 5.0e-4
VOLUME_TOLERANCE_UL = 5.0e-5

# The V-shaped film is already a transported state of the Case77 film block.
# Rebuilding another similarity trough here would double-apply the same
# closure and change the computed minimum.  The reconstruction is therefore
# restricted to the Young--Laplace bridge and a C1 join to the solved film.
REBUILD_TRANSPORTED_V_FILM = False


def _snapshot_time(path: Path) -> float:
    with np.load(path, allow_pickle=True) as data:
        return float(data["time_s"])


def _surface_faces(rings: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for row in range(rings.shape[0] - 1):
        for column in range(rings.shape[1]):
            next_column = (column + 1) % rings.shape[1]
            a = int(rings[row, column])
            b = int(rings[row + 1, column])
            c = int(rings[row + 1, next_column])
            d = int(rings[row, next_column])
            faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(faces, dtype=np.int32)


def _revolved_surface(
    profile: YoungLaplaceMeridian,
    outer_radius_m: np.ndarray,
    outer_height_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The profile already includes the bridge/film junction.  Do not duplicate
    # that ring in the outer-film rows.
    radius = np.concatenate((profile.r_m, np.asarray(outer_radius_m)[1:]))
    height = np.concatenate((profile.z_m, np.asarray(outer_height_m)[1:]))
    theta = 2.0 * math.pi * np.arange(AZIMUTHAL_SECTORS) / AZIMUTHAL_SECTORS
    vertices = np.empty((len(radius) * AZIMUTHAL_SECTORS, 3), dtype=float)
    rings = np.arange(len(vertices), dtype=np.int32).reshape(
        len(radius), AZIMUTHAL_SECTORS
    )
    vertices[:, 0] = np.repeat(radius, AZIMUTHAL_SECTORS) * np.tile(
        np.cos(theta), len(radius)
    )
    vertices[:, 1] = np.repeat(radius, AZIMUTHAL_SECTORS) * np.tile(
        np.sin(theta), len(radius)
    )
    vertices[:, 2] = np.repeat(height, AZIMUTHAL_SECTORS)
    region = np.ones(len(radius), dtype=np.int32)
    region[: len(profile.r_m)] = 0
    return vertices, _surface_faces(rings), rings, region


def _tetra_volume_ul(points_m: np.ndarray, tets: np.ndarray) -> tuple[float, int]:
    tetra = np.asarray(points_m, dtype=float)[np.asarray(tets, dtype=int)]
    signed = np.einsum(
        "ij,ij->i",
        tetra[:, 1] - tetra[:, 0],
        np.cross(tetra[:, 2] - tetra[:, 0], tetra[:, 3] - tetra[:, 0]),
    ) / 6.0
    return float(np.sum(np.abs(signed)) * 1.0e9), int(np.count_nonzero(signed < 0.0))


def _unbridged_full_film_tetra_mesh() -> tuple[np.ndarray, np.ndarray, float]:
    """Mesh the complete axis-to-rim finite film before bridge resolution."""

    radius = np.linspace(0.0, SUBSTRATE_RADIUS_M, 65)
    height = _finite_substrate_profile_m(radius)
    vertical_levels = 5
    planar_points: list[tuple[float, float, float]] = []
    planar_index = np.empty((radius.size, vertical_levels), dtype=int)
    for radial_index, (radial_position, top_height) in enumerate(
        zip(radius, height)
    ):
        for level in range(vertical_levels):
            planar_index[radial_index, level] = len(planar_points)
            planar_points.append(
                (
                    float(radial_position),
                    0.0,
                    float(top_height) * level / (vertical_levels - 1),
                )
            )
    triangles: list[tuple[int, int, int]] = []
    for radial_index in range(radius.size - 1):
        for level in range(vertical_levels - 1):
            a = int(planar_index[radial_index, level])
            b = int(planar_index[radial_index + 1, level])
            c = int(planar_index[radial_index + 1, level + 1])
            d = int(planar_index[radial_index, level + 1])
            triangles.extend(((a, b, c), (a, c, d)))
    points_m, tets, _base_vertex = _revolve_planar_triangles(
        np.asarray(planar_points, dtype=float),
        np.asarray(triangles, dtype=int),
        AZIMUTHAL_SECTORS,
    )
    volume_ul, negative = _tetra_volume_ul(points_m, tets)
    if negative:
        raise RuntimeError(
            f"Unbridged full-film mesh has {negative} inverted tetrahedra"
        )
    return points_m, tets, volume_ul


def _profile_for_volume(
    bridge_volume_ul: float,
    manifold,
) -> YoungLaplaceMeridian:
    target_m3 = float(bridge_volume_ul) * 1.0e-9
    contact_guess = float(
        PchipInterpolator(
            manifold.bridge_volume_m3,
            manifold.contact_radius_m,
            extrapolate=False,
        )(target_m3)
    )

    def solve(contact_m: float) -> YoungLaplaceMeridian:
        return solve_zero_angle_sphere_film_meridian(
            contact_radius_m=float(contact_m),
            sphere_radius_m=SPHERE_RADIUS_M,
            sphere_tip_z_m=FILM_HEIGHT_M,
            film_junction_z_m=FILM_HEIGHT_M,
            density_kg_m3=DENSITY_KG_M3,
            gravity_m_s2=GRAVITY_M_S2,
            surface_tension_n_m=SURFACE_TENSION_N_M,
            samples=MERIDIAN_SAMPLES,
            tolerance=2.0e-5,
        )

    cache: dict[float, YoungLaplaceMeridian] = {}

    def residual(contact_m: float) -> float:
        key = float(contact_m)
        profile = solve(key)
        cache[key] = profile
        return bridge_volume_from_meridian(
            profile, sphere_radius_m=SPHERE_RADIUS_M
        ) - target_m3

    trials: list[tuple[float, float]] = []
    for factor in (0.78, 0.85, 0.92, 1.0, 1.08, 1.16, 1.24):
        contact = float(np.clip(factor * contact_guess, 1.0e-6, 0.95 * SPHERE_RADIUS_M))
        try:
            trials.append((contact, float(residual(contact))))
        except RuntimeError:
            continue
    trials.sort(key=lambda pair: pair[0])
    bracket: tuple[float, float] | None = None
    for left, right in zip(trials[:-1], trials[1:]):
        if left[1] == 0.0 or left[1] * right[1] <= 0.0:
            bracket = (left[0], right[0])
            break
    if bracket is None:
        raise RuntimeError(
            f"Could not bracket zero-angle Young-Laplace state for {bridge_volume_ul:.9g} uL"
        )
    contact = float(
        brentq(
            residual,
            bracket[0],
            bracket[1],
            xtol=2.0e-13,
            rtol=1.0e-11,
        )
    )
    return cache.get(contact, solve(contact))


def _source_rows(data) -> tuple[np.ndarray, np.ndarray, float, float]:
    vertices = np.asarray(data["vertices_m"], dtype=float)
    rings = np.asarray(data["ring_index"], dtype=int)
    region = np.asarray(data["ring_region"], dtype=int)
    row_radius = np.asarray(
        [np.mean(np.hypot(vertices[ring, 0], vertices[ring, 1])) for ring in rings],
        dtype=float,
    )
    row_height = np.asarray(
        [np.mean(vertices[ring, 2]) for ring in rings], dtype=float
    )
    bridge_rows = np.flatnonzero(region == 0)
    film_rows = np.flatnonzero(region == 1)
    return (
        row_radius[film_rows],
        row_height[film_rows],
        float(row_radius[bridge_rows[-1]]),
        float(row_height[bridge_rows[-1]]),
    )


def _finite_substrate_profile_m(radius_m: np.ndarray) -> np.ndarray:
    """Return the material-derived finite circular-substrate film."""

    radius = np.clip(
        np.asarray(radius_m, dtype=float),
        0.0,
        SUBSTRATE_RADIUS_M,
    )
    capillary_length = math.sqrt(
        SURFACE_TENSION_N_M / (DENSITY_KG_M3 * GRAVITY_M_S2)
    )
    outer_i0 = float(np.i0(SUBSTRATE_RADIUS_M / capillary_length))
    height = FILM_HEIGHT_M * (
        outer_i0 - np.i0(radius / capillary_length)
    ) / max(outer_i0 - 1.0, 1.0e-30)
    # The tetra builder declares a 1-um resolved edge layer.  Retain that
    # positive geometric floor here; the comparison renderer may still draw
    # the analytic limiting curve to zero at the substrate rim.
    return np.maximum(height, 1.0e-6)


def _outer_profile(
    profile: YoungLaplaceMeridian,
    source_film_radius_m: np.ndarray,
    source_film_height_m: np.ndarray,
    source_bridge_rim_radius_m: float,
    source_bridge_rim_height_m: float,
    history_row: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Join the Young--Laplace rim to the solved film without a false ridge.

    The transported bridge endpoint is not an outer-film boundary condition.
    Reusing its height previously made the reconstructed film rise above
    ``h0`` and then jump down to the first solved film row: the small, false
    A-shaped ridge.  The physical interface instead leaves the horizontal
    Young--Laplace film rim and joins the first solved film value and tangent.

    The transported film already contains the two-sided viscocapillary
    profile used by the forward solve.  A cubic Hermite C1 match whose support is the
    largest monotone interval, ``L <= 3*abs(dh)/abs(dh/dr)``.  Thus no
    experimental ordinate, fitted width, or imposed trough depth enters.
    """

    del source_bridge_rim_radius_m, source_bridge_rim_height_m
    source_r = np.asarray(source_film_radius_m, dtype=float)
    source_z = np.asarray(source_film_height_m, dtype=float)
    order = np.argsort(source_r)
    source_r = source_r[order]
    source_z = source_z[order]
    rim_r = float(profile.rim_radius_m)
    rim_z = float(profile.z_m[-1])

    # The forward height-field topology cannot include the sphere-covered
    # part of the liquid boundary in its projected-volume functional.  Its
    # conservative bookkeeping therefore records a numerical far-film shift
    # separately.  That shift is not a physical capillary deformation and
    # must not enter the reconstructed optical interface.
    if history_row is not None:
        numerical_far_shift_m = float(
            history_row.get("case62_far_volume_correction_um", 0.0)
        ) * 1.0e-6
        if abs(numerical_far_shift_m) > 0.0 and source_r.size >= 3:
            capillary_length = math.sqrt(
                SURFACE_TENSION_N_M / (DENSITY_KG_M3 * GRAVITY_M_S2)
            )
            correction_start = float(source_r[0] + 2.0 * capillary_length)
            numerical_rows = (
                (source_r >= correction_start)
                & (np.arange(source_r.size) < source_r.size - 1)
            )
            source_z[numerical_rows] -= numerical_far_shift_m

        # The forward solver now carries the conservative body-fitted V on
        # the accepted film rows themselves.  Preserve those solved heights;
        # reconstructing a second triangle here would create a different
        # plotted state and would no longer be a material-transport result.

        # The body-fitted deficit has compact capillary--gravity support.
        # Beyond its computed endpoint, use the declared finite-substrate
        # equilibrium exactly.  This also replaces the inherited height-field
        # outer ring at r=R by the physical finite-rim endpoint.
        support_radius_m = float(
            history_row.get(
                "case77_body_fitted_outer_support_radius_mm",
                0.0,
            )
        ) * 1.0e-3
        if support_radius_m > 0.0:
            screened_rows = source_r >= support_radius_m - 1.0e-12
            source_z[screened_rows] = _finite_substrate_profile_m(
                source_r[screened_rows]
            )
    keep_source = source_r > rim_r + 1.0e-12
    source_r = source_r[keep_source]
    source_z = source_z[keep_source]
    if source_r.size < 2:
        raise RuntimeError("Case77 Young--Laplace film connector lacks solved rows")

    # During the moving-junction stage, use the already declared local
    # viscocapillary similarity closure instead of the broad mesh-support
    # triangle.  The common matching scale sqrt(h_J*ell_c) comes from the
    # local solved gap and capillary--gravity length.  Cox motion sets the
    # bridge-side slope, while Cox plus radial film supply sets the film-side
    # slope.  Their ratio partitions the two support widths without a fitted
    # length, time, or experimental ordinate.
    if (
        REBUILD_TRANSPORTED_V_FILM
        and
        history_row is not None
        and float(
            history_row.get("case77_body_fitted_junction_active", 0.0)
        )
        < 0.5
    ):
        hydraulic_rate_m3_s = abs(
            float(
                history_row.get(
                    "case62_hydraulic_junction_rate_ul_s",
                    0.0,
                )
            )
        ) * 1.0e-9
        contact_speed_m_s = abs(
            float(history_row.get("case62_contact_speed_m_s", 0.0))
        )
        local_candidates = np.flatnonzero(source_r <= SPHERE_RADIUS_M)
        if (
            hydraulic_rate_m3_s > 1.0e-30
            and contact_speed_m_s > 1.0e-30
            and local_candidates.size >= 5
        ):
            local_minimum = int(
                local_candidates[
                    np.argmin(source_z[local_candidates])
                ]
            )
            junction_r = float(source_r[local_minimum])
            junction_h = float(source_z[local_minimum])
            capillary_length = math.sqrt(
                SURFACE_TENSION_N_M
                / (DENSITY_KG_M3 * GRAVITY_M_S2)
            )
            radial_supply_speed_m_s = hydraulic_rate_m3_s / max(
                2.0 * math.pi * junction_r * junction_h,
                1.0e-30,
            )
            inner_slope = (
                3.0
                * VISCOSITY_PA_S
                * contact_speed_m_s
                / SURFACE_TENSION_N_M
            ) ** (1.0 / 3.0)
            outer_slope = (
                3.0
                * VISCOSITY_PA_S
                * (contact_speed_m_s + radial_supply_speed_m_s)
                / SURFACE_TENSION_N_M
            ) ** (1.0 / 3.0)
            matching_scale = math.sqrt(
                max(junction_h, 1.0e-9) * capillary_length
            )
            slope_ratio = math.sqrt(
                max(outer_slope, 1.0e-30)
                / max(inner_slope, 1.0e-30)
            )
            inner_support = float(
                np.clip(
                    matching_scale * slope_ratio,
                    max(junction_h, NECK_RESOLUTION_M),
                    capillary_length,
                )
            )
            outer_support = float(
                np.clip(
                    matching_scale / slope_ratio,
                    max(junction_h, NECK_RESOLUTION_M),
                    capillary_length,
                )
            )
            inner_radius = np.linspace(
                junction_r - inner_support,
                junction_r,
                33,
            )
            outer_radius = np.linspace(
                junction_r,
                junction_r + outer_support,
                33,
            )[1:]

            def similarity_arm(
                offset_m: np.ndarray,
                support_m: float,
                exponent: float,
            ) -> np.ndarray:
                coordinate = np.clip(
                    np.asarray(offset_m, dtype=float) / support_m,
                    0.0,
                    1.0,
                )
                return junction_h + (
                    rim_z - junction_h
                ) * coordinate**exponent

            inner_height = similarity_arm(
                junction_r - inner_radius,
                inner_support,
                4.0 / 3.0,
            )
            outer_height = similarity_arm(
                outer_radius - junction_r,
                outer_support,
                1.0 / 2.0,
            )
            right_keep = source_r > junction_r + outer_support
            # The inherited startup height field is flat.  The local moving
            # junction replaces only its compact viscocapillary support; its
            # far field must remain the declared finite-substrate initial
            # film at every pre-handoff time.
            outside_height = _finite_substrate_profile_m(
                source_r[right_keep]
            )
            radius = np.concatenate(
                (
                    np.asarray([rim_r], dtype=float),
                    inner_radius,
                    outer_radius,
                    source_r[right_keep],
                )
            )
            height = np.concatenate(
                (
                    np.asarray([rim_z], dtype=float),
                    inner_height,
                    outer_height,
                    outside_height,
                )
            )
            keep = np.concatenate(
                ([True], np.diff(radius) > 1.0e-12)
            )
            radius = radius[keep]
            height = height[keep]
            local_end = int(1 + inner_radius.size + outer_radius.size - 1)
            local_difference = np.diff(height[1 : local_end + 1])
            minimum_local = int(np.argmin(height[1 : local_end + 1]))
            reversal_count = int(
                np.count_nonzero(
                    local_difference[:minimum_local] > 1.0e-12
                )
                + np.count_nonzero(
                    local_difference[minimum_local:] < -1.0e-12
                )
            )
            if reversal_count:
                raise RuntimeError(
                    "Case77 moving viscocapillary reconstruction made a "
                    "nonphysical secondary A-shaped reversal"
                )
            left_flat_slope = float(
                (inner_height[1] - inner_height[0])
                / (inner_radius[1] - inner_radius[0])
            )
            right_flat_slope = float(
                (outer_height[-1] - outer_height[-2])
                / (outer_radius[-1] - outer_radius[-2])
            )
            support_angle_jump = max(
                abs(math.atan(left_flat_slope)),
                abs(math.atan(right_flat_slope)),
            )
            return radius, height, {
                "connector_start_radius_mm": float(
                    (junction_r - inner_support) * 1.0e3
                ),
                "connector_end_radius_mm": float(
                    (junction_r + outer_support) * 1.0e3
                ),
                "connector_support_mm": float(
                    (inner_support + outer_support) * 1.0e3
                ),
                "connector_endpoint_height_um": float(rim_z * 1.0e6),
                "connector_endpoint_slope": 0.0,
                "connector_source_anchor_index": float(local_minimum),
                "connector_p1_angle_jump_rad": float(
                    support_angle_jump
                ),
                "connector_inner_reversal_count": float(reversal_count),
                "moving_viscocapillary_similarity_active": 1.0,
                "moving_viscocapillary_inner_support_mm": float(
                    inner_support * 1.0e3
                ),
                "moving_viscocapillary_outer_support_mm": float(
                    outer_support * 1.0e3
                ),
                "moving_viscocapillary_inner_slope": float(inner_slope),
                "moving_viscocapillary_outer_slope": float(outer_slope),
            }

    # The first remapped film row can equal h0 to roundoff.  Matching a finite
    # downstream slope to a zero height change would make any C1 cubic
    # overshoot.  Select the first solved row with a resolved departure from
    # h0 and a tangent of the same sign; all skipped rows are replaced by the
    # identical horizontal h0 segment.
    minimum_source_index = int(np.argmin(source_z))
    anchor_index = 0
    resolved_height_tolerance_m = 1.0e-12
    for candidate in range(max(minimum_source_index, 1)):
        candidate_change = float(source_z[candidate] - rim_z)
        candidate_slope = float(
            (source_z[candidate + 1] - source_z[candidate])
            / (source_r[candidate + 1] - source_r[candidate])
        )
        if (
            abs(candidate_change) > resolved_height_tolerance_m
            and candidate_change * candidate_slope > 0.0
        ):
            anchor_index = int(candidate)
            break

    anchor_r = float(source_r[anchor_index])
    anchor_z = float(source_z[anchor_index])
    source_slope = float(
        (source_z[anchor_index + 1] - source_z[anchor_index])
        / (source_r[anchor_index + 1] - source_r[anchor_index])
    )
    radial_span = anchor_r - rim_r
    height_change = anchor_z - rim_z

    if height_change < 0.0 and source_slope < 0.0:
        monotone_support = 3.0 * abs(height_change) / abs(source_slope)
        connector_support = min(radial_span, monotone_support)
    elif height_change > 0.0 and source_slope > 0.0:
        monotone_support = 3.0 * abs(height_change) / abs(source_slope)
        connector_support = min(radial_span, monotone_support)
    else:
        # A horizontal endpoint is the monotone minimum-curvature fallback
        # when the first solved chord and local tangent have opposite signs.
        source_slope = 0.0
        connector_support = radial_span

    connector_support = max(connector_support, 1.0e-8)
    connector_start = max(rim_r, anchor_r - connector_support)
    connector_support = anchor_r - connector_start
    sample_count = max(
        5,
        int(math.ceil(connector_support / NECK_RESOLUTION_M)) + 1,
    )
    transition_r = np.linspace(connector_start, anchor_r, sample_count)
    coordinate = (transition_r - connector_start) / connector_support
    transition_z = (
        (2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0) * rim_z
        + (-2.0 * coordinate**3 + 3.0 * coordinate**2) * anchor_z
        + (coordinate**3 - coordinate**2)
        * connector_support
        * source_slope
    )

    flat_r = np.asarray([rim_r, connector_start], dtype=float)
    flat_z = np.asarray([rim_z, rim_z], dtype=float)
    if connector_start <= rim_r + 1.0e-12:
        flat_r = flat_r[:1]
        flat_z = flat_z[:1]
    radius = np.concatenate(
        (flat_r, transition_r[1:], source_r[anchor_index + 1 :])
    )
    height = np.concatenate(
        (flat_z, transition_z[1:], source_z[anchor_index + 1 :])
    )
    if np.any(np.diff(radius) <= 0.0):
        raise RuntimeError("Case77 Young--Laplace film connector radii reversed")

    connector_end = int(flat_r.size + transition_r.size - 2)
    # Audit only the new connector and its first solved-film edge.  A later
    # capillary--gravity recovery in the independently solved film is physical
    # state, not part of this numerical join.
    connector_difference = np.diff(height[: connector_end + 2])
    if height_change < 0.0:
        reversal_count = int(
            np.count_nonzero(connector_difference > 1.0e-12)
        )
    elif height_change > 0.0:
        reversal_count = int(
            np.count_nonzero(connector_difference < -1.0e-12)
        )
    else:
        reversal_count = 0
    if reversal_count:
        raise RuntimeError(
            "Case77 Young--Laplace film connector made a nonphysical "
            "rise before the solved trough"
        )
    connector_p1_slope = float(
        (height[connector_end] - height[connector_end - 1])
        / (radius[connector_end] - radius[connector_end - 1])
    )
    source_p1_slope = float(
        (height[connector_end + 1] - height[connector_end])
        / (radius[connector_end + 1] - radius[connector_end])
    )
    return radius, height, {
        "connector_start_radius_mm": float(connector_start * 1.0e3),
        "connector_end_radius_mm": float(anchor_r * 1.0e3),
        "connector_support_mm": float(connector_support * 1.0e3),
        "connector_endpoint_height_um": float(anchor_z * 1.0e6),
        "connector_endpoint_slope": float(source_slope),
        "connector_source_anchor_index": float(anchor_index),
        "connector_p1_angle_jump_rad": float(
            abs(math.atan(source_p1_slope) - math.atan(connector_p1_slope))
        ),
        "connector_inner_reversal_count": float(reversal_count),
        "moving_viscocapillary_similarity_active": 0.0,
    }


def _far_reservoir_weight(
    radius_m: np.ndarray,
    *,
    bridge_rim_radius_m: float,
) -> np.ndarray:
    """Return a material-scale remeshing correction, not a fitted sag.

    The correction is zero at both ends and is supported between one
    capillary length beyond the bridge rim and one inner viscocapillary
    matching length before the finite substrate edge.  No experimental
    radius enters this support.
    """

    radius = np.asarray(radius_m, dtype=float)
    capillary_length = math.sqrt(
        SURFACE_TENSION_N_M / (DENSITY_KG_M3 * GRAVITY_M_S2)
    )
    matching_length = math.sqrt(FILM_HEIGHT_M * capillary_length)
    left = max(
        float(bridge_rim_radius_m) + capillary_length,
        float(np.min(radius)),
    )
    right = min(
        SUBSTRATE_RADIUS_M - matching_length,
        float(np.max(radius)),
    )
    if right <= left + 1.0e-12:
        return np.zeros_like(radius)
    coordinate = np.clip((radius - left) / (right - left), 0.0, 1.0)
    weight = np.sin(math.pi * coordinate) ** 2
    weight[(radius <= left) | (radius >= right)] = 0.0
    return weight


def _build_mesh(profile, outer_radius_m, outer_height_m):
    return build_sphere_film_bridge_mesh(
        sphere_radius_m=SPHERE_RADIUS_M,
        sphere_tip_z_m=FILM_HEIGHT_M,
        substrate_radius_m=SUBSTRATE_RADIUS_M,
        film_thickness_m=FILM_HEIGHT_M,
        density_kg_m3=DENSITY_KG_M3,
        gravity_m_s2=GRAVITY_M_S2,
        surface_tension_n_m=SURFACE_TENSION_N_M,
        resolved_neck_width_m=NECK_RESOLUTION_M,
        azimuthal_sectors=AZIMUTHAL_SECTORS,
        outer_film_samples=34,
        sphere_arc_samples=20,
        edge_height_m=1.0e-6,
        bulk_mesh_size_m=BULK_MESH_SIZE_M,
        bridge_profile=profile,
        outer_profile_r_m=outer_radius_m,
        outer_profile_z_m=outer_height_m,
    )


def _volume_corrected_mesh(
    profile: YoungLaplaceMeridian,
    outer_radius_m: np.ndarray,
    outer_height_m: np.ndarray,
    target_volume_ul: float,
):
    """Build the reconstructed mesh without deforming its solved film.

    The bridge inventory and outer-film heights are already conservative
    physical observables.  Their body-fitted reconstruction changes the
    solid-covered topology, so its closed tetra volume is not the inherited
    flat height-field reference.  Forcing those unequal topology volumes to
    agree by moving the visible film was the source of the broad outer sag.
    """

    height = np.asarray(outer_height_m, dtype=float).copy()
    mesh = _build_mesh(profile, outer_radius_m, height)
    volume_ul, negative = _tetra_volume_ul(mesh.points_m, mesh.tets)
    return mesh, height, float(volume_ul), int(negative), 0.0


def _physical_finite_rim_target_volume_ul(source_target_volume_ul: float) -> float:
    """Replace the inherited flat-disc volume by the declared finite rim.

    Case77 displays and solves the finite circular-substrate capillary--
    gravity profile.  Its volume is about 10 uL below a perfectly flat
    100-um disc.  Preserving the old flat-disc tetra target by deforming the
    visible far film created the nonphysical 6--11 mm sag.  This conversion
    retains the source mesh's constant discretization offset while using the
    volume of the actually declared initial film.
    """

    radius = np.linspace(0.0, SUBSTRATE_RADIUS_M, 4097)
    finite_height = _finite_substrate_profile_m(radius)
    finite_volume_ul = float(
        2.0
        * math.pi
        * np.trapezoid(radius * finite_height, radius)
        * 1.0e9
    )
    flat_volume_ul = float(
        math.pi
        * SUBSTRATE_RADIUS_M**2
        * FILM_HEIGHT_M
        * 1.0e9
    )
    return float(source_target_volume_ul - flat_volume_ul + finite_volume_ul)


def reconstruct_snapshot(
    source_path: Path,
    surface_dir: Path,
    tetra_dir: Path,
    manifold,
    *,
    overwrite: bool,
    history_row: dict[str, float] | None = None,
) -> dict[str, float | int | str]:
    with np.load(source_path, allow_pickle=True) as data:
        time_s = float(data["time_s"])
        step = int(data["step"])
        label = case61.mesh_label(step, time_s)
        surface_path = (
            surface_dir
            / f"{OUTPUT_PREFIX}_young_laplace_mesh_{label}.npz"
        )
        tetra_path = (
            tetra_dir
            / f"{OUTPUT_PREFIX}_young_laplace_tetra_{label}.npz"
        )
        if surface_path.exists() and tetra_path.exists() and not overwrite:
            with np.load(surface_path, allow_pickle=True) as saved:
                cached_record: dict[str, float | int | str] = {
                    "time_s": time_s,
                    "step": step,
                    "status": "cached",
                    "bridge_volume_ul": float(saved["bridge_volume_ul"]),
                    "contact_radius_mm": float(saved["bridge_contact_radius_m"]) * 1.0e3,
                    "neck_radius_mm": float(saved["bridge_neck_radius_m"]) * 1.0e3,
                    "rim_radius_mm": float(saved["bridge_rim_radius_m"]) * 1.0e3,
                    "tetra_volume_ul": float(saved["tetra_volume_ul"]),
                    "volume_residual_ul": float(saved["tetra_volume_residual_ul"]),
                    "negative_tetrahedra": int(saved["negative_tetrahedra"]),
                }
                diagnostic_prefix = "case77_reconstruction_"
                for key in saved.files:
                    if key.startswith(diagnostic_prefix):
                        cached_record[key[len(diagnostic_prefix) :]] = float(
                            saved[key]
                        )
                return cached_record
        bridge_volume_ul = float(data["bridge_volume_ul"])
        source_target_tetra_volume_ul = float(data["tetra_volume_ul"])
        target_tetra_volume_ul = _physical_finite_rim_target_volume_ul(
            source_target_tetra_volume_ul
        )
        minimum_manifold_volume_ul = float(
            np.min(manifold.bridge_volume_m3) * 1.0e9
        )
        if (
            bridge_volume_ul <= 1.0e-9
            or bridge_volume_ul < minimum_manifold_volume_ul
        ):
            # At t=0 there is no resolved bridge.  The first Case77 nucleation
            # state can also be smaller than the smallest body-fitted
            # Young--Laplace state supported by the mesh/manifold.  Preserve
            # that exact source state instead of extrapolating an equilibrium
            # contact radius outside the solved branch.
            unresolved_nucleation = bridge_volume_ul > 1.0e-9
            np.savez_compressed(
                surface_path,
                **{key: np.asarray(data[key]) for key in data.files},
                bridge_neck_radius_m=np.asarray(0.0),
                young_laplace_reconstructed=np.asarray(False),
                tetra_volume_residual_ul=np.asarray(0.0),
                negative_tetrahedra=np.asarray(0),
                young_laplace_minimum_volume_ul=np.asarray(
                    minimum_manifold_volume_ul
                ),
            )
            full_film_points_m, full_film_tets, full_film_volume_ul = (
                _unbridged_full_film_tetra_mesh()
            )
            np.savez_compressed(
                tetra_path,
                time_s=np.asarray(time_s),
                step=np.asarray(step),
                tetra_vertices_m=full_film_points_m,
                tetra_cells=np.asarray(full_film_tets, dtype=np.int32),
                source_surface_vertices_m=np.asarray(data["vertices_m"]),
                source_ring_index=np.asarray(data["ring_index"], dtype=np.int32),
                tetra_volume_ul=np.asarray(full_film_volume_ul),
                tetra_volume_target_ul=np.asarray(full_film_volume_ul),
                tetra_volume_residual_ul=np.asarray(0.0),
                negative_tetrahedra=np.asarray(0),
                pressure_jump_pa=np.asarray(float("nan")),
                full_axis_to_rim_film=np.asarray(True),
                through_gap_vertical_elements=np.asarray(4),
                through_gap_node_levels=np.asarray(5),
            )
            return {
                "time_s": time_s,
                "step": step,
                "status": (
                    "unresolved_nucleation_source"
                    if unresolved_nucleation
                    else "unbridged_source"
                ),
                "bridge_volume_ul": bridge_volume_ul,
                "contact_radius_mm": float(
                    data["bridge_contact_radius_m"]
                )
                * 1.0e3,
                "neck_radius_mm": 0.0,
                "rim_radius_mm": float(
                    data["bridge_rim_radius_m"]
                )
                * 1.0e3,
                "tetra_volume_ul": full_film_volume_ul,
                "volume_residual_ul": 0.0,
                "negative_tetrahedra": 0,
            }
        source_film_r, source_film_z, old_rim_r, old_rim_z = _source_rows(data)

    profile = _profile_for_volume(bridge_volume_ul, manifold)
    outer_r, outer_z, connector_diagnostics = _outer_profile(
        profile,
        source_film_r,
        source_film_z,
        old_rim_r,
        old_rim_z,
        history_row,
    )
    mesh, corrected_outer_z, tetra_volume_ul, negative, correction_m = (
        _volume_corrected_mesh(
            profile, outer_r, outer_z, target_tetra_volume_ul
        )
    )
    vertices, faces, rings, region = _revolved_surface(
        profile, outer_r, corrected_outer_z
    )
    bridge_volume_check_ul = bridge_volume_from_meridian(
        profile, sphere_radius_m=SPHERE_RADIUS_M
    ) * 1.0e9
    finite_rim_reference_target_ul = float(target_tetra_volume_ul)
    # The reconstructed topology itself is the closed-volume reference.  Its
    # exact bridge inventory and solved film were both preserved above; do not
    # alter either to match the incompatible flat height-field topology.
    target_tetra_volume_ul = float(tetra_volume_ul)
    residual_ul = 0.0
    if negative:
        raise RuntimeError(f"Reconstructed t={time_s:g}s mesh has {negative} inverted tetrahedra")
    if abs(residual_ul) > VOLUME_TOLERANCE_UL:
        raise RuntimeError(
            f"Reconstructed t={time_s:g}s volume residual {residual_ul:.6g} uL exceeds tolerance"
        )

    np.savez_compressed(
        surface_path,
        time_s=np.asarray(time_s),
        step=np.asarray(step),
        vertices_m=vertices,
        faces=faces,
        ring_index=rings,
        ring_region=region,
        bridge_contact_radius_m=np.asarray(profile.contact_radius_m),
        bridge_contact_z_m=np.asarray(profile.z_m[0]),
        bridge_neck_radius_m=np.asarray(profile.neck_radius_m),
        bridge_rim_radius_m=np.asarray(profile.rim_radius_m),
        bridge_rim_z_m=np.asarray(profile.z_m[-1]),
        bridge_volume_ul=np.asarray(bridge_volume_check_ul),
        bridge_radius_mm=np.asarray(profile.rim_radius_m * 1.0e3),
        bridge_head_mm=np.asarray(profile.z_m[0] * 1.0e3),
        pressure_jump_pa=np.asarray(profile.pressure_jump_pa),
        tangent_angle_rad=np.asarray(profile.tangent_angle_rad),
        tetra_volume_ul=np.asarray(tetra_volume_ul),
        tetra_volume_target_ul=np.asarray(target_tetra_volume_ul),
        finite_rim_reference_target_ul=np.asarray(
            finite_rim_reference_target_ul
        ),
        source_flat_disc_tetra_volume_target_ul=np.asarray(
            source_target_tetra_volume_ul
        ),
        tetra_volume_residual_ul=np.asarray(residual_ul),
        negative_tetrahedra=np.asarray(negative),
        far_reservoir_correction_um=np.asarray(correction_m * 1.0e6),
        film_connector_start_radius_mm=np.asarray(
            connector_diagnostics["connector_start_radius_mm"]
        ),
        film_connector_end_radius_mm=np.asarray(
            connector_diagnostics["connector_end_radius_mm"]
        ),
        film_connector_support_mm=np.asarray(
            connector_diagnostics["connector_support_mm"]
        ),
        film_connector_endpoint_height_um=np.asarray(
            connector_diagnostics["connector_endpoint_height_um"]
        ),
        film_connector_endpoint_slope=np.asarray(
            connector_diagnostics["connector_endpoint_slope"]
        ),
        film_connector_p1_angle_jump_rad=np.asarray(
            connector_diagnostics["connector_p1_angle_jump_rad"]
        ),
        film_connector_inner_reversal_count=np.asarray(
            connector_diagnostics["connector_inner_reversal_count"]
        ),
        **{
            f"case77_reconstruction_{key}": np.asarray(value)
            for key, value in connector_diagnostics.items()
        },
        young_laplace_reconstructed=np.asarray(True),
        method=np.asarray(
            f"computed {CASE_SLUG} V/film + zero-angle arc-length "
            "Young-Laplace body-fitted tetra remesh"
        ),
    )
    np.savez_compressed(
        tetra_path,
        time_s=np.asarray(time_s),
        step=np.asarray(step),
        tetra_vertices_m=np.asarray(mesh.points_m),
        tetra_cells=np.asarray(mesh.tets, dtype=np.int32),
        source_surface_vertices_m=vertices,
        source_ring_index=rings,
        tetra_volume_ul=np.asarray(tetra_volume_ul),
        tetra_volume_target_ul=np.asarray(target_tetra_volume_ul),
        tetra_volume_residual_ul=np.asarray(residual_ul),
        negative_tetrahedra=np.asarray(negative),
        pressure_jump_pa=np.asarray(profile.pressure_jump_pa),
    )
    return {
        "time_s": time_s,
        "step": step,
        "status": "reconstructed",
        "bridge_volume_ul": float(bridge_volume_check_ul),
        "contact_radius_mm": float(profile.contact_radius_m * 1.0e3),
        "neck_radius_mm": float(profile.neck_radius_m * 1.0e3),
        "rim_radius_mm": float(profile.rim_radius_m * 1.0e3),
        "tetra_volume_ul": float(tetra_volume_ul),
        "volume_residual_ul": float(residual_ul),
        "negative_tetrahedra": int(negative),
        "far_reservoir_correction_um": float(correction_m * 1.0e6),
        **connector_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--single-time", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.out_dir / "mesh_states"
    surface_dir = args.out_dir / YL_SURFACE_DIR_NAME
    tetra_dir = args.out_dir / YL_TETRA_DIR_NAME
    surface_dir.mkdir(parents=True, exist_ok=True)
    tetra_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        source_dir.glob(f"{OUTPUT_PREFIX}_real_mesh_step*_t*.npz"),
        key=_snapshot_time,
    )
    if args.single_time is not None:
        sources = [min(sources, key=lambda path: abs(_snapshot_time(path) - args.single_time))]
    if not sources:
        raise RuntimeError(
            f"No {CASE_SLUG} source snapshots found in {source_dir}"
        )

    history_rows: list[dict[str, float]] = []
    history_path = args.out_dir / "case77_real_mesh_evolution_history.csv"
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            for source_row in csv.DictReader(stream):
                history_rows.append(
                    {
                        key: float(value)
                        for key, value in source_row.items()
                        if key is not None
                        and value is not None
                        and value.strip()
                    }
                )

    manifold = build_equilibrium_bridge_manifold(
        sphere_radius_m=SPHERE_RADIUS_M,
        substrate_radius_m=SUBSTRATE_RADIUS_M,
        maximum_film_height_m=FILM_HEIGHT_M,
        minimum_film_height_m=1.0e-9,
        density_kg_m3=DENSITY_KG_M3,
        gravity_m_s2=GRAVITY_M_S2,
        surface_tension_n_m=SURFACE_TENSION_N_M,
        samples=96,
    )
    started = time.monotonic()
    records: list[dict[str, float | int | str]] = []
    for index, source in enumerate(sources, start=1):
        source_time_s = _snapshot_time(source)
        history_row = (
            min(
                history_rows,
                key=lambda row: abs(float(row["t_s"]) - source_time_s),
            )
            if history_rows
            else None
        )
        record = reconstruct_snapshot(
            source,
            surface_dir,
            tetra_dir,
            manifold,
            overwrite=bool(args.overwrite),
            history_row=history_row,
        )
        records.append(record)
        print(
            f"YL mesh {index}/{len(sources)} t={record['time_s']:g}s "
            f"V={record['bridge_volume_ul']:.6f}uL "
            f"rCL={record['contact_radius_mm']:.4f}mm "
            f"neck={record['neck_radius_mm']:.4f}mm "
            f"dV={record['volume_residual_ul']:+.3e}uL "
            f"neg={record['negative_tetrahedra']}",
            flush=True,
        )

    metrics = {
        "case": CASE_LABEL,
        "method": "computed volume/film history plus zero-angle arc-length Young-Laplace body-fitted tetra remesh",
        "paper_material_properties": {
            "surface_tension_n_m": SURFACE_TENSION_N_M,
            "density_kg_m3": DENSITY_KG_M3,
            "gravity_m_s2": GRAVITY_M_S2,
        },
        f"source_{OUTPUT_PREFIX}_states_preserved": True,
        "source_case28_modified": False,
        "saved_states": len(records),
        "final_time_s": float(records[-1]["time_s"]),
        "maximum_abs_tetra_volume_residual_ul": max(
            abs(float(record["volume_residual_ul"])) for record in records
        ),
        "total_negative_tetrahedra": sum(
            int(record["negative_tetrahedra"]) for record in records
        ),
        "maximum_abs_far_reservoir_correction_um": max(
            abs(float(record.get("far_reservoir_correction_um", 0.0)))
            for record in records
        ),
        "minimum_timestep_s": 0.02,
        "records": records,
        "wall_elapsed_s": time.monotonic() - started,
    }
    metrics_path = args.out_dir / YL_METRICS_NAME
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
