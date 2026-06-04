"""ddgclib-operator-backed flux-aware HC/FHeron droplet benchmark.

This is a separate finite-volume/ALE-style diagnostic from
``sphere_fheron_eos_projection_benchmark.py``.  It keeps the same HC Heron
surface force, but updates nodal mass, nodal dual volume, nodal density, and
boundary area vectors every step.  It also computes internal tet-face mass and
momentum fluxes.

The flux model here is deliberately compact for a benchmark GIF.  It is not a
production CFD discretization.

This copy keeps the old benchmark file untouched, but routes the force,
volume-gradient pressure, ALE flux, and active-retopology remap through new
non-breaking ddgclib operator functions added under ``ddgclib.operators``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib.operators import stress as ddg_stress

import sphere_fheron_eos_projection_benchmark as base

plt = base.plt
FuncAnimation = base.FuncAnimation
PillowWriter = base.PillowWriter
Line3DCollection = base.Line3DCollection
Poly3DCollection = base.Poly3DCollection
ScalarFormatter = base.ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent


def install_ddgclib_operator_aliases() -> None:
    """Route benchmark kernels through ddgclib operator add-ons.

    The old benchmark file stays untouched.  This runner uses ddgclib
    operators for:
    - HC/FHeron interface force, Heron (curvature force)
    - B=dV/dx tet-volume pressure/continuity operator, Chorin/Temam style
    - ALE material-face flux, Hirt-Amsden-Cook/Toro
    - active Delaunay retopology remap, tet-volume lumping workaround
    """
    global tet_face_pairs, delaunay_tets_for_points, nodal_from_tets, compute_ale_fluxes

    base.fheron_forces_for_points = ddg_stress.heron_forces_for_points
    base.tet_cell_volumes = ddg_stress.tet_cell_volumes
    base.tet_volume_matrix = ddg_stress.tet_volume_matrix
    base.lump_tet_masses = ddg_stress.lump_tet_masses
    base._local_volume_stiffness = ddg_stress.local_volume_stiffness
    base._solve_regularized = ddg_stress.solve_regularized

    tet_face_pairs = ddg_stress.tet_face_pairs
    delaunay_tets_for_points = ddg_stress.delaunay_retriangulate_points
    nodal_from_tets = ddg_stress.tet_volume_lumped_nodal_values
    compute_ale_fluxes = ddg_stress.tet_face_ale_fluxes


def tet_face_pairs(tets: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """Return internal faces as ``(face_vertices, left_tet, right_tet)``."""
    face_map: dict[tuple[int, int, int], list[int]] = {}
    for tet_idx, tet in enumerate(np.asarray(tets, dtype=int)):
        a, b, c, d = [int(index) for index in tet]
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_map.setdefault(tuple(sorted(face)), []).append(tet_idx)
    pairs: list[tuple[np.ndarray, int, int]] = []
    for face, owners in face_map.items():
        if len(owners) == 2:
            pairs.append((np.asarray(face, dtype=int), int(owners[0]), int(owners[1])))
    return pairs


def canonical_tet_set(tets: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(sorted(int(index) for index in tet)) for tet in np.asarray(tets, dtype=int)}


def delaunay_tets_for_points(points: np.ndarray, fallback_tets: np.ndarray) -> np.ndarray:
    """Fresh Delaunay tets for diagnostics without changing material masses."""
    try:
        triangulation = base.Delaunay(np.asarray(points, dtype=float), qhull_options="Qbb Qc")
        raw_tets = np.asarray(triangulation.simplices, dtype=int)
        valid = np.all(raw_tets < points.shape[0], axis=1)
        raw_tets = raw_tets[valid]
        if raw_tets.size == 0:
            return np.asarray(fallback_tets, dtype=int).copy()
        volumes = base.tet_cell_volumes(points, raw_tets)
        positive = volumes[volumes > 0.0]
        min_volume = 0.1 * float(np.median(positive)) if positive.size else 0.0
        filtered = raw_tets[volumes > min_volume]
        if filtered.size == 0:
            return np.asarray(fallback_tets, dtype=int).copy()
        return np.asarray(filtered, dtype=int)
    except Exception:
        return np.asarray(fallback_tets, dtype=int).copy()


def nodal_from_tets(n_vertices: int, tets: np.ndarray, tet_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros(n_vertices, dtype=float)
    for tet, value in zip(np.asarray(tets, dtype=int), np.asarray(tet_values, dtype=float)):
        share = 0.25 * float(value)
        for vertex_idx in tet:
            nodal[int(vertex_idx)] += share
    return nodal


def current_area_vectors(
    points: np.ndarray,
    surface_indices: np.ndarray,
    surface_faces: np.ndarray,
) -> np.ndarray:
    area_vectors = np.zeros_like(points)
    area_vectors[np.asarray(surface_indices, dtype=int)] = base.vertex_area_vectors(
        points[np.asarray(surface_indices, dtype=int)],
        surface_faces,
    )
    return area_vectors


def compute_ale_fluxes(
    points: np.ndarray,
    velocities: np.ndarray,
    tets: np.ndarray,
    face_pairs: list[tuple[np.ndarray, int, int]],
    tet_masses: np.ndarray,
    reference_tet_masses: np.ndarray,
    tet_volumes: np.ndarray,
    dt: float,
    *,
    reference_density: float,
    sound_speed: float,
    mass_floor_fraction: float = 0.35,
    mass_update_coupling: float = 0.02,
    momentum_force_coupling: float = 0.0,
    density_change_limit: float | None = None,
    flux_cfl_limit: float = 0.25,
    flux_scheme: str = "upwind",
    face_flux_mode: str = "cell-average",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compute internal ALE face mass/momentum flux and update cell masses."""
    if flux_scheme not in {"upwind", "rusanov"}:
        raise ValueError("flux_scheme must be 'upwind' or 'rusanov'")
    if face_flux_mode not in {"cell-average", "lagrangian"}:
        raise ValueError("face_flux_mode must be 'cell-average' or 'lagrangian'")

    n_tets = np.asarray(tets).shape[0]
    cell_centers = np.mean(points[np.asarray(tets, dtype=int)], axis=1)
    cell_velocities = np.mean(velocities[np.asarray(tets, dtype=int)], axis=1)
    cell_density = np.asarray(tet_masses, dtype=float) / np.maximum(tet_volumes, 1.0e-300)

    dm_dt = np.zeros(n_tets, dtype=float)
    cell_force = np.zeros((n_tets, 3), dtype=float)
    mass_flux_l1 = 0.0
    max_abs_mass_flux = 0.0
    momentum_flux_l1 = 0.0
    volume_flux_l1 = 0.0

    for face_vertices, left, right in face_pairs:
        pa, pb, pc = points[face_vertices]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        center_delta = cell_centers[right] - cell_centers[left]
        if float(np.dot(area_vec, center_delta)) < 0.0:
            area_vec = -area_vec

        area = float(np.linalg.norm(area_vec))
        if area <= 0.0:
            continue
        normal = area_vec / area
        mesh_face_velocity = np.mean(velocities[face_vertices], axis=0)
        left_velocity = cell_velocities[left]
        right_velocity = cell_velocities[right]
        left_density = float(cell_density[left])
        right_density = float(cell_density[right])

        if face_flux_mode == "lagrangian":
            # In a Lagrangian vertex mesh the material face velocity is the
            # mesh-face velocity.  This preserves the ALE geometric conservation
            # law and prevents artificial inter-tet flux from cell averaging.
            relative_volume_flux = 0.0
            mass_flux = 0.0
            momentum_flux = np.zeros(3, dtype=float)

        elif flux_scheme == "rusanov":
            left_relative_normal = float(np.dot(left_velocity - mesh_face_velocity, normal))
            right_relative_normal = float(np.dot(right_velocity - mesh_face_velocity, normal))
            max_wave_speed = max(
                abs(left_relative_normal) + float(sound_speed),
                abs(right_relative_normal) + float(sound_speed),
                1.0e-30,
            )
            mass_flux = area * (
                0.5 * (left_density * left_relative_normal + right_density * right_relative_normal)
                - 0.5 * max_wave_speed * (right_density - left_density)
            )
            momentum_flux = area * (
                0.5 * (left_density * left_velocity * left_relative_normal + right_density * right_velocity * right_relative_normal)
                - 0.5 * max_wave_speed * (right_density * right_velocity - left_density * left_velocity)
            )
            relative_volume_flux = mass_flux / max(0.5 * (left_density + right_density), 1.0e-300)
        else:
            fluid_face_velocity = 0.5 * (left_velocity + right_velocity)
            relative_volume_flux = float(np.dot(fluid_face_velocity - mesh_face_velocity, area_vec))
            if relative_volume_flux >= 0.0:
                density = left_density
                upwind_velocity = left_velocity
            else:
                density = right_density
                upwind_velocity = right_velocity
            mass_flux = density * relative_volume_flux
            momentum_flux = mass_flux * upwind_velocity

        dm_dt[left] -= mass_flux
        dm_dt[right] += mass_flux
        cell_force[left] -= momentum_flux
        cell_force[right] += momentum_flux

        abs_mass_flux = abs(float(mass_flux))
        mass_flux_l1 += abs_mass_flux
        max_abs_mass_flux = max(max_abs_mass_flux, abs_mass_flux)
        momentum_flux_l1 += float(np.linalg.norm(momentum_flux))
        volume_flux_l1 += abs(float(relative_volume_flux))

    effective_dm_dt = float(mass_update_coupling) * dm_dt
    effective_cell_force = float(momentum_force_coupling) * cell_force

    scale = 1.0
    mass_delta = float(dt) * effective_dm_dt
    mass_abs_delta = np.abs(mass_delta)
    if flux_cfl_limit > 0.0 and np.any(mass_abs_delta > 0.0):
        cfl_scale = np.min(
            float(flux_cfl_limit) * np.asarray(tet_masses, dtype=float)[mass_abs_delta > 0.0]
            / mass_abs_delta[mass_abs_delta > 0.0]
        )
        scale = min(scale, max(0.0, float(cfl_scale)))

    floors = float(mass_floor_fraction) * np.asarray(reference_tet_masses, dtype=float)
    proposed = np.asarray(tet_masses, dtype=float) + mass_delta
    floor_bad = proposed < floors
    if np.any(floor_bad):
        denom = -mass_delta[floor_bad]
        valid = denom > 0.0
        if np.any(valid):
            floor_scale = np.min((tet_masses[floor_bad][valid] - floors[floor_bad][valid]) / denom[valid])
            scale = min(scale, max(0.0, float(floor_scale)))

    if density_change_limit is not None and density_change_limit > 0.0:
        lower_masses = (1.0 - float(density_change_limit)) * float(reference_density) * np.maximum(
            tet_volumes,
            1.0e-300,
        )
        upper_masses = (1.0 + float(density_change_limit)) * float(reference_density) * np.maximum(
            tet_volumes,
            1.0e-300,
        )
        increasing_too_much = (mass_delta > 0.0) & (proposed > upper_masses)
        if np.any(increasing_too_much):
            density_scale = np.min(
                (upper_masses[increasing_too_much] - tet_masses[increasing_too_much])
                / mass_delta[increasing_too_much]
            )
            scale = min(scale, max(0.0, float(density_scale)))
        decreasing_too_much = (mass_delta < 0.0) & (proposed < lower_masses)
        if np.any(decreasing_too_much):
            density_scale = np.min(
                (lower_masses[decreasing_too_much] - tet_masses[decreasing_too_much])
                / mass_delta[decreasing_too_much]
            )
            scale = min(scale, max(0.0, float(density_scale)))

    effective_dm_dt *= scale
    effective_cell_force *= scale

    updated_masses = np.maximum(np.asarray(tet_masses, dtype=float) + float(dt) * effective_dm_dt, floors)
    nodal_flux_force = np.zeros_like(points)
    for tet, force in zip(np.asarray(tets, dtype=int), effective_cell_force):
        for vertex_idx in tet:
            nodal_flux_force[int(vertex_idx)] += 0.25 * force

    stats = {
        "mass_flux_l1_kg_s": float(mass_flux_l1),
        "mass_flux_max_kg_s": float(max_abs_mass_flux),
        "momentum_flux_l1_n": float(momentum_flux_l1),
        "volume_flux_l1_m3_s": float(volume_flux_l1),
        "flux_limiter_scale": float(scale),
        "mass_update_coupling": float(mass_update_coupling),
        "momentum_force_coupling": float(momentum_force_coupling),
        "density_change_limit": float(density_change_limit) if density_change_limit is not None else 0.0,
        "flux_cfl_limit": float(flux_cfl_limit),
        "flux_scheme_rusanov": 1.0 if flux_scheme == "rusanov" else 0.0,
        "face_flux_lagrangian": 1.0 if face_flux_mode == "lagrangian" else 0.0,
    }
    return updated_masses, nodal_flux_force, stats


def linearized_mass_flux_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    face_pairs: list[tuple[np.ndarray, int, int]],
    tet_masses: np.ndarray,
    tet_volumes: np.ndarray,
    *,
    reference_density: float,
    mass_update_coupling: float,
) -> np.ndarray:
    """Linearized ALE mass-flux operator ``dm_dt = G u`` for the pressure solve."""
    n_tets = np.asarray(tets).shape[0]
    n_dof = np.asarray(points).shape[0] * 3
    matrix = np.zeros((n_tets, n_dof), dtype=float)
    if not face_pairs or mass_update_coupling == 0.0:
        return matrix

    cell_centers = np.mean(points[np.asarray(tets, dtype=int)], axis=1)
    cell_density = np.asarray(tet_masses, dtype=float) / np.maximum(tet_volumes, 1.0e-300)
    for face_vertices, left, right in face_pairs:
        pa, pb, pc = points[face_vertices]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        center_delta = cell_centers[right] - cell_centers[left]
        if float(np.dot(area_vec, center_delta)) < 0.0:
            area_vec = -area_vec
        if float(np.linalg.norm(area_vec)) <= 0.0:
            continue

        rho_face = max(0.5 * (float(cell_density[left]) + float(cell_density[right])), 1.0e-12)
        weights: dict[int, float] = {}
        for vertex_idx in np.asarray(tets[left], dtype=int):
            weights[int(vertex_idx)] = weights.get(int(vertex_idx), 0.0) + 0.125
        for vertex_idx in np.asarray(tets[right], dtype=int):
            weights[int(vertex_idx)] = weights.get(int(vertex_idx), 0.0) + 0.125
        for vertex_idx in np.asarray(face_vertices, dtype=int):
            weights[int(vertex_idx)] = weights.get(int(vertex_idx), 0.0) - 1.0 / 3.0

        for vertex_idx, weight in weights.items():
            for component in range(3):
                coefficient = float(mass_update_coupling) * rho_face * float(weight) * float(area_vec[component])
                dof = 3 * int(vertex_idx) + component
                matrix[int(left), dof] -= coefficient
                matrix[int(right), dof] += coefficient
    return matrix


def barycentric_edge_dual_areas(points: np.ndarray, tets: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Barycentric dual-face area vectors ``A_ij`` for mesh vertex pairs.

    The returned vector is oriented from the lower vertex index toward the
    higher vertex index.  For vertex ``i`` the pairwise pressure force uses
    ``A_ij`` outward from ``i``; the sign is flipped automatically when
    ``i`` is the higher-index endpoint.
    """
    edge_areas: dict[tuple[int, int], np.ndarray] = {}
    pts = np.asarray(points, dtype=float)
    for tet in np.asarray(tets, dtype=int):
        centroid = np.mean(pts[tet], axis=0)
        for a_local in range(4):
            for b_local in range(a_local + 1, 4):
                i = int(tet[a_local])
                j = int(tet[b_local])
                other = [int(tet[k]) for k in range(4) if k not in (a_local, b_local)]
                k, l = other
                midpoint = 0.5 * (pts[i] + pts[j])
                face_centroid_1 = (pts[i] + pts[j] + pts[k]) / 3.0
                face_centroid_2 = (pts[i] + pts[j] + pts[l]) / 3.0
                area_vec = (
                    0.5 * np.cross(face_centroid_1 - midpoint, centroid - midpoint)
                    + 0.5 * np.cross(centroid - midpoint, face_centroid_2 - midpoint)
                )
                if float(np.dot(area_vec, pts[j] - pts[i])) < 0.0:
                    area_vec = -area_vec
                if i < j:
                    key = (i, j)
                    oriented = area_vec
                else:
                    key = (j, i)
                    oriented = -area_vec
                edge_areas[key] = edge_areas.get(key, np.zeros(3, dtype=float)) + oriented
    return edge_areas


def tet_pressure_to_vertex_weights(n_vertices: int, tets: np.ndarray, tet_volumes: np.ndarray) -> np.ndarray:
    """Volume-weighted map from tet pressure values to vertex pressure values."""
    n_tets = int(np.asarray(tets).shape[0])
    weights = np.zeros((int(n_vertices), n_tets), dtype=float)
    incident = np.zeros(int(n_vertices), dtype=float)
    for tet_idx, tet in enumerate(np.asarray(tets, dtype=int)):
        volume = max(float(tet_volumes[int(tet_idx)]), 1.0e-300)
        for vertex_idx in tet:
            weights[int(vertex_idx), int(tet_idx)] += volume
            incident[int(vertex_idx)] += volume
    weights /= np.maximum(incident[:, None], 1.0e-300)
    return weights


def pairwise_pressure_force_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
) -> np.ndarray:
    """Linear force matrix for ``F_i^p = sum_j -0.5(p_i+p_j) A_ij``."""
    n_vertices = int(np.asarray(points).shape[0])
    n_tets = int(np.asarray(tets).shape[0])
    pressure_weights = tet_pressure_to_vertex_weights(n_vertices, tets, tet_volumes)
    matrix = np.zeros((3 * n_vertices, n_tets), dtype=float)
    for (i, j), area_vec in barycentric_edge_dual_areas(points, tets).items():
        coefficient = -0.5 * (pressure_weights[int(i)] + pressure_weights[int(j)])
        for component in range(3):
            force_coeff = coefficient * float(area_vec[component])
            matrix[3 * int(i) + component] += force_coeff
            matrix[3 * int(j) + component] -= force_coeff
    return matrix


def pairwise_vertex_pressure_matrix(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Force matrix for vertex pressures in ``-0.5(p_i+p_j) A_ij``."""
    n_vertices = int(np.asarray(points).shape[0])
    matrix = np.zeros((3 * n_vertices, n_vertices), dtype=float)
    for (i, j), area_vec in barycentric_edge_dual_areas(points, tets).items():
        for component in range(3):
            coefficient = -0.5 * float(area_vec[component])
            matrix[3 * int(i) + component, int(i)] += coefficient
            matrix[3 * int(i) + component, int(j)] += coefficient
            matrix[3 * int(j) + component, int(i)] -= coefficient
            matrix[3 * int(j) + component, int(j)] -= coefficient
    return matrix


def tet_vertex_pressure_average_matrix(n_vertices: int, tets: np.ndarray) -> np.ndarray:
    """Map vertex pressures to tet pressures by arithmetic vertex average."""
    n_tets = int(np.asarray(tets).shape[0])
    matrix = np.zeros((n_tets, int(n_vertices)), dtype=float)
    for tet_idx, tet in enumerate(np.asarray(tets, dtype=int)):
        for vertex_idx in tet:
            matrix[int(tet_idx), int(vertex_idx)] += 0.25
    return matrix


def solve_regularized_lstsq(matrix: np.ndarray, rhs: np.ndarray, *, relative: float = 1.0e-12) -> np.ndarray:
    """Least-squares solve with small Tikhonov regularization for rectangular systems."""
    lhs = np.asarray(matrix, dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)
    scale = max(float(np.mean(np.abs(lhs))) if lhs.size else 0.0, 1.0e-30)
    damping = math.sqrt(float(relative) * scale)
    augmented = np.vstack([lhs, damping * np.eye(lhs.shape[1])])
    augmented_rhs = np.concatenate([rhs_array, np.zeros(lhs.shape[1], dtype=float)])
    return np.linalg.lstsq(augmented, augmented_rhs, rcond=1.0e-12)[0]


def pairwise_forcefit_pressure_matrix(
    points: np.ndarray,
    tets: np.ndarray,
    local_matrix: np.ndarray,
) -> np.ndarray:
    """Map tet pressures to vertex-pair forces via fitted vertex pressures."""
    vertex_matrix = pairwise_vertex_pressure_matrix(points, tets)
    pressure_transfer = np.linalg.lstsq(vertex_matrix, np.asarray(local_matrix.T, dtype=float), rcond=1.0e-12)[0]
    return vertex_matrix @ pressure_transfer


def pressure_force_matrix_for_mode(
    mode: str,
    points: np.ndarray,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
    local_matrix: np.ndarray,
) -> np.ndarray:
    if mode == "volume-gradient":
        return np.asarray(local_matrix.T, dtype=float)
    if mode in {"pairwise-dual", "pairwise-dual-consistent"}:
        return pairwise_pressure_force_matrix(points, tets, tet_volumes)
    if mode == "pairwise-vertex-eos":
        return pairwise_vertex_pressure_matrix(points, tets)
    if mode == "pairwise-dual-scalar":
        pairwise_matrix = pairwise_pressure_force_matrix(points, tets, tet_volumes)
        uniform_force = pairwise_matrix @ np.ones(int(np.asarray(tets).shape[0]), dtype=float)
        scalar_average = np.full(int(np.asarray(tets).shape[0]), 1.0 / float(np.asarray(tets).shape[0]))
        return np.outer(uniform_force, scalar_average)
    if mode == "pairwise-dual-forcefit":
        return pairwise_forcefit_pressure_matrix(points, tets, local_matrix)
    raise ValueError(
        "pressure_force_mode must be volume-gradient, pairwise-dual, "
        "pairwise-dual-scalar, pairwise-dual-forcefit, pairwise-dual-consistent, "
        "or pairwise-vertex-eos"
    )


def pressure_solve_matrix_for_mode(
    mode: str,
    pressure_force_matrix: np.ndarray,
    local_matrix: np.ndarray,
) -> np.ndarray:
    if mode in {"pairwise-dual-consistent", "pairwise-vertex-eos"}:
        return pressure_force_matrix
    return np.asarray(local_matrix.T, dtype=float)


def project_local_volume_velocity_with_pressure_matrix(
    velocities: np.ndarray,
    volume_matrix: np.ndarray,
    pressure_force_matrix: np.ndarray,
    masses: np.ndarray,
    target_rates: np.ndarray,
) -> np.ndarray:
    velocity_vec = velocities.reshape(-1)
    inv_mass_dof = np.repeat(1.0 / np.maximum(masses, 1.0e-300), 3)
    stiffness = (volume_matrix * inv_mass_dof[None, :]) @ pressure_force_matrix
    rhs = volume_matrix @ velocity_vec - np.asarray(target_rates, dtype=float)
    multipliers = base._solve_regularized(stiffness, rhs)
    projected_vec = velocity_vec - inv_mass_dof * (pressure_force_matrix @ multipliers)
    return projected_vec.reshape(velocities.shape)


def row_array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def write_rows(rows: list[dict[str, float]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def run_flux_dynamic(
    *,
    closure: str,
    subdivisions: int,
    radius: float,
    gamma: float,
    rho: float,
    bulk_modulus: float,
    amplitude: float,
    n_steps: int,
    t_final: float,
    substeps_per_sample: int,
    damping_ratio: float,
    mass_model: str,
    inertia_scale: float,
    mass_flux_coupling: float,
    momentum_flux_coupling: float,
    density_change_limit: float | None,
    flux_cfl_limit: float,
    flux_scheme: str,
    flux_timing: str,
    face_flux_mode: str,
    mass_floor_fraction: float,
    coupled_flux_continuity: bool,
    flux_continuity_sign: float,
    incompressible_target_mode: str,
    pressure_force_mode: str,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
    rayleigh_mode: int,
    ambient_pressure: float,
    retriangulation_mode: str,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if closure not in {"compressible", "incompressible"}:
        raise ValueError("closure must be compressible or incompressible")
    if incompressible_target_mode not in {"reference", "mass"}:
        raise ValueError("incompressible_target_mode must be reference or mass")
    if flux_timing not in {"pre-pressure", "post-pressure"}:
        raise ValueError("flux_timing must be pre-pressure or post-pressure")
    if mass_model not in {"star-volume", "rayleigh-added"}:
        raise ValueError("mass_model must be star-volume or rayleigh-added")
    if pressure_force_mode not in {
        "volume-gradient",
        "pairwise-dual",
        "pairwise-dual-scalar",
        "pairwise-dual-forcefit",
        "pairwise-dual-consistent",
        "pairwise-vertex-eos",
    }:
        raise ValueError(
            "pressure_force_mode must be volume-gradient, pairwise-dual, "
            "pairwise-dual-scalar, pairwise-dual-forcefit, pairwise-dual-consistent, "
            "or pairwise-vertex-eos"
        )
    if retriangulation_mode not in {"none", "diagnostic", "active"}:
        raise ValueError("retriangulation_mode must be none, diagnostic, or active")

    reference_points, tets, surface_indices, surface_faces, _surface_faces_global, _unit_points = base.build_ball_tet_mesh(
        subdivisions,
        radius,
    )
    points = base.deform_l2_volume_points(
        reference_points,
        tets,
        radius=radius,
        amplitude=amplitude,
        target_volume=float(np.sum(base.tet_cell_volumes(reference_points, tets))),
    )
    reference_tet_volumes, _matrix0 = base.tet_volume_matrix(points, tets)
    reference_total_volume = float(np.sum(reference_tet_volumes))
    tet_masses = float(rho) * reference_tet_volumes
    reference_tet_masses = tet_masses.copy()
    reference_total_mass = float(np.sum(tet_masses))
    reference_nodal_volumes = nodal_from_tets(points.shape[0], tets, reference_tet_volumes)
    velocities = np.zeros_like(points)
    face_pairs = tet_face_pairs(tets)
    reference_display_tet_set = canonical_tet_set(tets)
    inertial_mass_addition = np.zeros(points.shape[0], dtype=float)
    if mass_model == "rayleigh-added":
        reference_scalar_areas = base.vertex_scalar_areas(reference_points[surface_indices], surface_faces)
        surface_added_mass = float(rho) * float(radius) / float(rayleigh_mode)
        inertial_mass_addition[surface_indices] = surface_added_mass * reference_scalar_areas
        mass_model_label = "bulk plus Rayleigh surface mass"
    else:
        mass_model_label = "bulk-tet lumped mass"

    _forces0, _areas0, reference_pressure, _volume0 = base.fheron_forces_for_points(
        reference_points[surface_indices],
        surface_faces,
        gamma,
    )

    dt = float(t_final) / float((int(n_steps) - 1) * int(substeps_per_sample))
    damping_rate = 2.0 * float(damping_ratio) * float(rayleigh_omega_rad_s)
    rows: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    display_tets_snapshots: list[np.ndarray] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    last_flux = {
        "mass_flux_l1_kg_s": 0.0,
        "mass_flux_max_kg_s": 0.0,
        "momentum_flux_l1_n": 0.0,
        "volume_flux_l1_m3_s": 0.0,
        "flux_limiter_scale": 1.0,
        "mass_update_coupling": float(mass_flux_coupling),
        "momentum_force_coupling": float(momentum_flux_coupling),
        "density_change_limit": float(density_change_limit) if density_change_limit is not None else 0.0,
        "flux_cfl_limit": float(flux_cfl_limit),
        "flux_scheme_rusanov": 1.0 if flux_scheme == "rusanov" else 0.0,
        "face_flux_lagrangian": 1.0 if face_flux_mode == "lagrangian" else 0.0,
        "flux_post_pressure": 1.0 if flux_timing == "post-pressure" else 0.0,
        "mass_floor_fraction": float(mass_floor_fraction),
        "coupled_flux_continuity": 1.0 if coupled_flux_continuity else 0.0,
        "flux_continuity_sign": float(flux_continuity_sign),
        "pressure_force_pairwise_dual": 1.0 if pressure_force_mode != "volume-gradient" else 0.0,
        "pressure_force_scalar_pairwise": 1.0 if pressure_force_mode == "pairwise-dual-scalar" else 0.0,
        "pressure_force_forcefit_pairwise": 1.0 if pressure_force_mode == "pairwise-dual-forcefit" else 0.0,
        "pressure_solve_pairwise_dual": 1.0 if pressure_force_mode == "pairwise-dual-consistent" else 0.0,
        "pressure_solve_pairwise_vertex": 1.0 if pressure_force_mode == "pairwise-vertex-eos" else 0.0,
    }
    sound_speed = math.sqrt(float(bulk_modulus) / float(rho))

    def active_retriangulate() -> None:
        nonlocal tets, reference_tet_volumes, tet_masses, reference_tet_masses
        nonlocal reference_nodal_volumes, face_pairs

        new_tets, new_reference_volumes, new_masses, _retopo_stats = ddg_stress.active_retopology_tet_remap(
            points,
            tets,
            target_total_mass=reference_total_mass,
            target_total_volume=reference_total_volume,
        )
        if np.asarray(new_tets).size == 0:
            return

        tets = np.asarray(new_tets, dtype=int)
        reference_tet_volumes = np.asarray(new_reference_volumes, dtype=float)
        tet_masses = np.asarray(new_masses, dtype=float)
        reference_tet_masses = tet_masses.copy()
        reference_nodal_volumes = nodal_from_tets(points.shape[0], tets, reference_tet_volumes)
        face_pairs = tet_face_pairs(tets)

    def collect(sample_index: int, time_value: float) -> None:
        tet_volumes, local_matrix = base.tet_volume_matrix(points, tets)
        if retriangulation_mode == "diagnostic":
            display_tets = delaunay_tets_for_points(points, tets)
        else:
            display_tets = tets
        display_tet_set = canonical_tet_set(display_tets)
        display_union = reference_display_tet_set | display_tet_set
        display_changed = reference_display_tet_set ^ display_tet_set
        display_change_fraction = float(len(display_changed)) / max(float(len(display_union)), 1.0)
        display_face_pairs = tet_face_pairs(display_tets)
        nodal_mass = nodal_from_tets(points.shape[0], tets, tet_masses)
        inertial_nodal_mass = np.maximum(float(inertia_scale) * (nodal_mass + inertial_mass_addition), 1.0e-18)
        nodal_volume = nodal_from_tets(points.shape[0], tets, tet_volumes)
        nodal_density = nodal_mass / np.maximum(nodal_volume, 1.0e-300)
        area_vectors = current_area_vectors(points, surface_indices, surface_faces)
        surface_points = points[surface_indices]
        _surface_forces, _surface_area_vectors, fheron_equiv, _surface_volume = base.fheron_forces_for_points(
            surface_points,
            surface_faces,
            gamma,
        )
        fitted_amplitude = base.fit_l2_shape_amplitude(surface_points, surface_faces)
        theory_amplitude = float(amplitude) * math.cos(float(rayleigh_omega_rad_s) * float(time_value))
        radial_ratio = np.linalg.norm(surface_points, axis=1) / float(radius)
        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start
        substeps_done = int(sample_index) * int(substeps_per_sample)
        cpu_ms_per_substep = 1.0e3 * cpu_elapsed / float(substeps_done) if substeps_done else 0.0
        wall_ms_per_substep = 1.0e3 * wall_elapsed / float(substeps_done) if substeps_done else 0.0
        if closure == "compressible" or incompressible_target_mode == "mass":
            target_volumes = tet_masses / float(rho)
        else:
            target_volumes = reference_tet_volumes
        local_volume_rel = tet_volumes / np.maximum(target_volumes, 1.0e-300)
        density_rel = nodal_density / float(rho)
        nodal_volume_rel = nodal_volume / np.maximum(reference_nodal_volumes, 1.0e-300)
        area_norm = np.linalg.norm(area_vectors[surface_indices], axis=1)

        rows.append(
            {
                "sample": float(sample_index),
                "t": float(time_value),
                "closure": closure,
                "shape_amplitude_theory": float(theory_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "fheron_pressure_pa": float(fheron_equiv),
                "reference_pressure_pa": float(reference_pressure),
                "closure_pressure_pa": float(getattr(collect, "last_pressure", reference_pressure)),
                "ambient_pressure_pa": float(ambient_pressure),
                "liquid_pressure_absolute_pa": float(ambient_pressure + getattr(collect, "last_pressure", reference_pressure)),
                "mass_model": mass_model_label,
                "inertia_scale": float(inertia_scale),
                "total_mass_kg": float(np.sum(tet_masses)),
                "total_inertial_mass_kg": float(np.sum(inertial_nodal_mass)),
                "mass_rel_error": float(np.sum(tet_masses) / reference_total_mass - 1.0),
                "vol_m3": float(np.sum(tet_volumes)),
                "vol0_mesh_m3": float(np.sum(reference_tet_volumes)),
                "vol_rel": float(np.sum(tet_volumes) / np.sum(reference_tet_volumes)),
                "local_volume_rel_rms": float(math.sqrt(float(np.mean((local_volume_rel - 1.0) ** 2)))),
                "local_volume_rel_max_abs": float(np.max(np.abs(local_volume_rel - 1.0))),
                "nodal_mass_min_kg": float(np.min(nodal_mass)),
                "nodal_mass_max_kg": float(np.max(nodal_mass)),
                "inertial_mass_min_kg": float(np.min(inertial_nodal_mass)),
                "inertial_mass_max_kg": float(np.max(inertial_nodal_mass)),
                "nodal_volume_rel_min": float(np.min(nodal_volume_rel[nodal_volume > 0.0])),
                "nodal_volume_rel_max": float(np.max(nodal_volume_rel[nodal_volume > 0.0])),
                "nodal_density_rel_min": float(np.min(density_rel[nodal_volume > 0.0])),
                "nodal_density_rel_max": float(np.max(density_rel[nodal_volume > 0.0])),
                "area_vector_mean_m2": float(np.mean(area_norm)),
                "area_vector_max_m2": float(np.max(area_norm)),
                "mass_flux_l1_kg_s": float(last_flux["mass_flux_l1_kg_s"]),
                "mass_flux_max_kg_s": float(last_flux["mass_flux_max_kg_s"]),
                "momentum_flux_l1_n": float(last_flux["momentum_flux_l1_n"]),
                "volume_flux_l1_m3_s": float(last_flux["volume_flux_l1_m3_s"]),
                "flux_limiter_scale": float(last_flux["flux_limiter_scale"]),
                "mass_update_coupling": float(last_flux["mass_update_coupling"]),
                "momentum_force_coupling": float(last_flux["momentum_force_coupling"]),
                "density_change_limit": float(last_flux["density_change_limit"]),
                "flux_cfl_limit": float(last_flux["flux_cfl_limit"]),
                "flux_scheme_rusanov": float(last_flux["flux_scheme_rusanov"]),
                "face_flux_lagrangian": float(last_flux["face_flux_lagrangian"]),
                "flux_post_pressure": float(last_flux["flux_post_pressure"]),
                "mass_floor_fraction": float(last_flux["mass_floor_fraction"]),
                "coupled_flux_continuity": float(last_flux["coupled_flux_continuity"]),
                "flux_continuity_sign": float(last_flux["flux_continuity_sign"]),
                "pressure_force_pairwise_dual": float(last_flux["pressure_force_pairwise_dual"]),
                "pressure_force_scalar_pairwise": float(last_flux["pressure_force_scalar_pairwise"]),
                "pressure_force_forcefit_pairwise": float(last_flux["pressure_force_forcefit_pairwise"]),
                "pressure_solve_pairwise_dual": float(last_flux["pressure_solve_pairwise_dual"]),
                "pressure_solve_pairwise_vertex": float(last_flux["pressure_solve_pairwise_vertex"]),
                "retriangulation_diagnostic": 1.0 if retriangulation_mode == "diagnostic" else 0.0,
                "retriangulation_active": 1.0 if retriangulation_mode == "active" else 0.0,
                "solver_tet_count": float(tets.shape[0]),
                "display_tet_count": float(display_tets.shape[0]),
                "display_internal_face_count": float(len(display_face_pairs)),
                "display_changed_tets": float(len(display_changed)),
                "display_changed_fraction": float(display_change_fraction),
                "incompressible_target_from_mass": 1.0 if incompressible_target_mode == "mass" else 0.0,
                "cpu_time_s": float(cpu_elapsed),
                "wall_time_s": float(wall_elapsed),
                "cpu_ms_per_substep": float(cpu_ms_per_substep),
                "wall_ms_per_substep": float(wall_ms_per_substep),
                "max_speed_m_s": float(np.max(np.linalg.norm(velocities, axis=1))),
                "radial_ratio_min": float(np.min(radial_ratio)),
                "radial_ratio_max": float(np.max(radial_ratio)),
            }
        )
        snapshots.append(points.copy())
        display_tets_snapshots.append(np.asarray(display_tets, dtype=int).copy())

    collect.last_pressure = reference_pressure  # type: ignore[attr-defined]
    collect(0, 0.0)

    total_substeps = (int(n_steps) - 1) * int(substeps_per_sample)
    sample_index = 1
    for substep in range(1, total_substeps + 1):
        if retriangulation_mode == "active":
            active_retriangulate()
        tet_volumes, local_matrix = base.tet_volume_matrix(points, tets)
        if flux_timing == "pre-pressure":
            tet_masses, flux_forces, last_flux = compute_ale_fluxes(
                points,
                velocities,
                tets,
                face_pairs,
                tet_masses,
                reference_tet_masses,
                tet_volumes,
                dt,
                reference_density=rho,
                sound_speed=sound_speed,
                mass_floor_fraction=mass_floor_fraction,
                mass_update_coupling=mass_flux_coupling,
                momentum_force_coupling=momentum_flux_coupling,
                density_change_limit=density_change_limit,
                flux_cfl_limit=flux_cfl_limit,
                flux_scheme=flux_scheme,
                face_flux_mode=face_flux_mode,
            )
            last_flux["flux_post_pressure"] = 0.0
            last_flux["mass_floor_fraction"] = float(mass_floor_fraction)
            last_flux["coupled_flux_continuity"] = 1.0 if coupled_flux_continuity else 0.0
            last_flux["flux_continuity_sign"] = float(flux_continuity_sign)
            last_flux["pressure_force_pairwise_dual"] = 1.0 if pressure_force_mode != "volume-gradient" else 0.0
            last_flux["pressure_force_scalar_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-scalar" else 0.0
            last_flux["pressure_force_forcefit_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-forcefit" else 0.0
            last_flux["pressure_solve_pairwise_dual"] = 1.0 if pressure_force_mode == "pairwise-dual-consistent" else 0.0
            last_flux["pressure_solve_pairwise_vertex"] = 1.0 if pressure_force_mode == "pairwise-vertex-eos" else 0.0
        else:
            flux_forces = np.zeros_like(points)
            last_flux = dict(last_flux)
            last_flux["mass_flux_l1_kg_s"] = 0.0
            last_flux["mass_flux_max_kg_s"] = 0.0
            last_flux["momentum_flux_l1_n"] = 0.0
            last_flux["volume_flux_l1_m3_s"] = 0.0
            last_flux["flux_limiter_scale"] = 1.0
            last_flux["flux_post_pressure"] = 1.0
            last_flux["mass_floor_fraction"] = float(mass_floor_fraction)
            last_flux["coupled_flux_continuity"] = 1.0 if coupled_flux_continuity else 0.0
            last_flux["flux_continuity_sign"] = float(flux_continuity_sign)
            last_flux["pressure_force_pairwise_dual"] = 1.0 if pressure_force_mode != "volume-gradient" else 0.0
            last_flux["pressure_force_scalar_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-scalar" else 0.0
            last_flux["pressure_force_forcefit_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-forcefit" else 0.0
            last_flux["pressure_solve_pairwise_dual"] = 1.0 if pressure_force_mode == "pairwise-dual-consistent" else 0.0
            last_flux["pressure_solve_pairwise_vertex"] = 1.0 if pressure_force_mode == "pairwise-vertex-eos" else 0.0
        nodal_mass = nodal_from_tets(points.shape[0], tets, tet_masses)
        inertial_nodal_mass = np.maximum(float(inertia_scale) * (nodal_mass + inertial_mass_addition), 1.0e-18)
        surface_points = points[surface_indices]
        surface_forces, _surface_area_vectors, _fheron_equiv, _surface_volume = base.fheron_forces_for_points(
            surface_points,
            surface_faces,
            gamma,
        )
        forces = np.zeros_like(points)
        forces[surface_indices] += surface_forces
        damping_forces = -damping_rate * inertial_nodal_mass[:, None] * velocities
        if closure == "compressible" or incompressible_target_mode == "mass":
            target_tet_volumes = tet_masses / float(rho)
        else:
            target_tet_volumes = reference_tet_volumes
        pressure_matrix = pressure_force_matrix_for_mode(
            pressure_force_mode,
            points,
            tets,
            tet_volumes,
            local_matrix,
        )
        pressure_solve_matrix = pressure_solve_matrix_for_mode(
            pressure_force_mode,
            pressure_matrix,
            local_matrix,
        )

        if closure == "compressible":
            if pressure_force_mode == "pairwise-vertex-eos":
                base_pressures = np.full(points.shape[0], float(reference_pressure), dtype=float)
            else:
                base_pressures = np.full(tets.shape[0], float(reference_pressure), dtype=float)
            nonpressure_forces = (
                forces
                + (pressure_matrix @ base_pressures).reshape(points.shape)
                + damping_forces
                + flux_forces
            )
            u_star = velocities + dt * nonpressure_forces / inertial_nodal_mass[:, None]
            _stiffness, inv_mass_dof = base._local_volume_stiffness(local_matrix, inertial_nodal_mass)
            compressibility = float(bulk_modulus) / np.maximum(reference_tet_volumes, 1.0e-300)
            if coupled_flux_continuity and mass_flux_coupling and face_flux_mode == "cell-average":
                mass_flux_matrix = linearized_mass_flux_matrix(
                    points,
                    tets,
                    face_pairs,
                    tet_masses,
                    tet_volumes,
                    reference_density=rho,
                    mass_update_coupling=mass_flux_coupling,
                )
                continuity_matrix = local_matrix + float(flux_continuity_sign) * mass_flux_matrix / float(rho)
            else:
                continuity_matrix = local_matrix
            if pressure_force_mode == "pairwise-vertex-eos":
                vertex_average = tet_vertex_pressure_average_matrix(points.shape[0], tets)
                nonpressure_forces = (
                    forces
                    + (pressure_matrix @ base_pressures).reshape(points.shape)
                    + damping_forces
                    + flux_forces
                )
                u_star = velocities + dt * nonpressure_forces / inertial_nodal_mass[:, None]
                coupled_stiffness = (continuity_matrix * inv_mass_dof[None, :]) @ pressure_matrix
                linear_volume_error = tet_volumes - target_tet_volumes + dt * (continuity_matrix @ u_star.reshape(-1))
                system = vertex_average + (compressibility[:, None] * dt * dt) * coupled_stiffness
                rhs = -compressibility * linear_volume_error
                pressure_delta_vertices = solve_regularized_lstsq(system, rhs, relative=1.0e-6)
                velocities = (
                    u_star.reshape(-1)
                    + dt * inv_mass_dof * (pressure_matrix @ pressure_delta_vertices)
                ).reshape(velocities.shape)
                collect.last_pressure = float(np.mean(vertex_average @ (base_pressures + pressure_delta_vertices)))  # type: ignore[attr-defined]
            elif pressure_force_mode == "pairwise-dual-scalar":
                pressure_direction = pressure_matrix @ np.ones(tets.shape[0], dtype=float)
                linear_volume_error = tet_volumes - target_tet_volumes + dt * (continuity_matrix @ u_star.reshape(-1))
                global_volume_error = float(np.sum(linear_volume_error))
                global_compressibility = float(bulk_modulus) / max(float(np.sum(reference_tet_volumes)), 1.0e-300)
                scalar_stiffness = float(np.sum(continuity_matrix @ (inv_mass_dof * pressure_direction)))
                denom = 1.0 + float(dt) * float(dt) * global_compressibility * scalar_stiffness
                pressure_delta_scalar = -global_compressibility * global_volume_error / max(denom, 1.0e-300)
                velocities = (
                    u_star.reshape(-1)
                    + dt * inv_mass_dof * pressure_direction * float(pressure_delta_scalar)
                ).reshape(velocities.shape)
                collect.last_pressure = float(reference_pressure + pressure_delta_scalar)  # type: ignore[attr-defined]
            else:
                coupled_stiffness = (continuity_matrix * inv_mass_dof[None, :]) @ pressure_solve_matrix
                linear_volume_error = tet_volumes - target_tet_volumes + dt * (continuity_matrix @ u_star.reshape(-1))
                system = np.eye(tets.shape[0]) + (compressibility[:, None] * dt * dt) * coupled_stiffness
                rhs = -compressibility * linear_volume_error
                pressure_delta = base._solve_regularized(system, rhs)
                velocities = (u_star.reshape(-1) + dt * inv_mass_dof * (pressure_matrix @ pressure_delta)).reshape(
                    velocities.shape
                )
                collect.last_pressure = float(np.mean(base_pressures + pressure_delta))  # type: ignore[attr-defined]
        else:
            inv_mass_dof = np.repeat(1.0 / np.maximum(inertial_nodal_mass, 1.0e-300), 3)
            target_rates = -(tet_volumes - target_tet_volumes) / float(dt)
            velocity_vec = velocities.reshape(-1)
            nonpressure_vec = (forces + damping_forces + flux_forces).reshape(-1)
            pressure_stiffness = (local_matrix * inv_mass_dof[None, :]) @ pressure_solve_matrix
            pressure_rhs = (
                (target_rates - local_matrix @ velocity_vec) / float(dt)
                - local_matrix @ (inv_mass_dof * nonpressure_vec)
            )
            pressure_values = base._solve_regularized(pressure_stiffness, pressure_rhs)
            total_forces = (
                forces
                + (pressure_matrix @ pressure_values).reshape(points.shape)
                + damping_forces
                + flux_forces
            )
            velocities = velocities + dt * total_forces / inertial_nodal_mass[:, None]
            velocities = project_local_volume_velocity_with_pressure_matrix(
                velocities,
                local_matrix,
                pressure_solve_matrix,
                inertial_nodal_mass,
                target_rates,
            )
            collect.last_pressure = float(np.mean(pressure_values))  # type: ignore[attr-defined]

        if flux_timing == "post-pressure":
            tet_masses, post_flux_forces, last_flux = compute_ale_fluxes(
                points,
                velocities,
                tets,
                face_pairs,
                tet_masses,
                reference_tet_masses,
                tet_volumes,
                dt,
                reference_density=rho,
                sound_speed=sound_speed,
                mass_floor_fraction=mass_floor_fraction,
                mass_update_coupling=mass_flux_coupling,
                momentum_force_coupling=momentum_flux_coupling,
                density_change_limit=density_change_limit,
                flux_cfl_limit=flux_cfl_limit,
                flux_scheme=flux_scheme,
                face_flux_mode=face_flux_mode,
            )
            last_flux["flux_post_pressure"] = 1.0
            last_flux["mass_floor_fraction"] = float(mass_floor_fraction)
            last_flux["coupled_flux_continuity"] = 1.0 if coupled_flux_continuity else 0.0
            last_flux["flux_continuity_sign"] = float(flux_continuity_sign)
            last_flux["pressure_force_pairwise_dual"] = 1.0 if pressure_force_mode != "volume-gradient" else 0.0
            last_flux["pressure_force_scalar_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-scalar" else 0.0
            last_flux["pressure_force_forcefit_pairwise"] = 1.0 if pressure_force_mode == "pairwise-dual-forcefit" else 0.0
            last_flux["pressure_solve_pairwise_dual"] = 1.0 if pressure_force_mode == "pairwise-dual-consistent" else 0.0
            last_flux["pressure_solve_pairwise_vertex"] = 1.0 if pressure_force_mode == "pairwise-vertex-eos" else 0.0
            if momentum_flux_coupling:
                post_nodal_mass = nodal_from_tets(points.shape[0], tets, tet_masses)
                post_inertial_mass = np.maximum(
                    float(inertia_scale) * (post_nodal_mass + inertial_mass_addition),
                    1.0e-18,
                )
                velocities = velocities + dt * post_flux_forces / post_inertial_mass[:, None]

        points = points + dt * velocities
        points, velocities = base._remove_rigid_translation(points, velocities, inertial_nodal_mass)

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(velocities)):
            raise FloatingPointError(f"{closure} flux run became non-finite at substep {substep}")

        if substep % int(substeps_per_sample) == 0:
            collect(sample_index, sample_index * float(t_final) / float(int(n_steps) - 1))
            sample_index += 1

    return rows, np.asarray(snapshots, dtype=float), surface_faces, tets, surface_indices, display_tets_snapshots


def write_snapshots(
    snapshots: np.ndarray,
    faces: np.ndarray,
    tets: np.ndarray,
    surface_indices: np.ndarray,
    out_dir: Path,
    stem: str,
    display_tets_snapshots: list[np.ndarray] | None = None,
) -> Path:
    path = out_dir / f"{stem}.npz"
    payload: dict[str, np.ndarray] = {
        "points": snapshots,
        "faces": faces,
        "tets": tets,
        "surface_indices": surface_indices,
    }
    if display_tets_snapshots is not None:
        payload["display_tets"] = np.asarray(display_tets_snapshots, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def write_flux_animation(
    rows: list[dict[str, float]],
    snapshots: np.ndarray,
    faces: np.ndarray,
    tets: np.ndarray,
    surface_indices: np.ndarray,
    out_dir: Path,
    *,
    radius: float,
    closure: str,
    max_frames: int,
    fps: int,
    display_tets_snapshots: list[np.ndarray] | None = None,
) -> Path:
    t = row_array(rows, "t")
    amplitude_theory = row_array(rows, "shape_amplitude_theory")
    amplitude_fit = row_array(rows, "shape_amplitude_fit")
    global_volume_pct = (row_array(rows, "vol_rel") - 1.0) * 100.0
    local_volume_pct = row_array(rows, "local_volume_rel_max_abs") * 100.0
    density_spread_ppm = (row_array(rows, "nodal_density_rel_max") - row_array(rows, "nodal_density_rel_min")) * 1.0e6
    mass_flux = row_array(rows, "mass_flux_l1_kg_s")
    momentum_flux = row_array(rows, "momentum_flux_l1_n")
    fheron = row_array(rows, "fheron_pressure_pa")
    pressure = row_array(rows, "closure_pressure_pa")
    ambient_pressure = row_array(rows, "ambient_pressure_pa")
    display_tet_count = row_array(rows, "display_tet_count")
    display_face_count = row_array(rows, "display_internal_face_count")
    display_changed_tets = row_array(rows, "display_changed_tets")
    display_changed_fraction = row_array(rows, "display_changed_fraction")
    cpu_time = row_array(rows, "cpu_time_s")
    cpu_ms = row_array(rows, "cpu_ms_per_substep")
    limiter_scale = row_array(rows, "flux_limiter_scale")
    density_limit = float(rows[0].get("density_change_limit", 0.0))
    flux_scheme = "Rusanov" if float(rows[0].get("flux_scheme_rusanov", 0.0)) > 0.5 else "upwind"
    face_flux_mode = "Lagrangian face" if float(rows[0].get("face_flux_lagrangian", 0.0)) > 0.5 else "cell-average face"
    if float(rows[0].get("pressure_force_forcefit_pairwise", 0.0)) > 0.5:
        pressure_force_label = "pairwise Aij fitted p"
    elif float(rows[0].get("pressure_force_scalar_pairwise", 0.0)) > 0.5:
        pressure_force_label = "pairwise Aij scalar p"
    elif float(rows[0].get("pressure_solve_pairwise_vertex", 0.0)) > 0.5:
        pressure_force_label = "pairwise Aij vertex p"
    elif float(rows[0].get("pressure_force_pairwise_dual", 0.0)) > 0.5:
        pressure_force_label = "pairwise Aij"
    else:
        pressure_force_label = "volume-gradient Bt"
    cpu_total = float(cpu_time[-1])
    cpu_avg = float(cpu_ms[-1])
    active_retopology = float(rows[0].get("retriangulation_active", 0.0)) > 0.5
    show_retriangulation = active_retopology or float(rows[0].get("retriangulation_diagnostic", 0.0)) > 0.5
    retopo_prefix = "Active-retopology " if active_retopology else ""

    if closure == "compressible":
        facecolor = "#9ecae1"
        edgecolor = "#1f77b4"
        gif_name = "flux_compressible_sphere.gif"
        title = f"{retopo_prefix}one-phase free-surface compressible EOS, {face_flux_mode} ALE flux"
    else:
        facecolor = "#fdd0a2"
        edgecolor = "#ff7f0e"
        gif_name = "flux_incompressible_sphere.gif"
        title = f"{retopo_prefix}one-phase free-surface incompressible projection with ALE flux"

    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), max(2, int(max_frames))), dtype=int))
    radial_max = max(float(row["radial_ratio_max"]) for row in rows)
    limit = 1.15 * radial_max * float(radius)
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)
    time_tick_labels = [f"{value:.3f}" for value in time_ticks]

    fig = plt.figure(figsize=(10.4, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.35, 1.0), wspace=0.27, hspace=0.50)
    ax3d = fig.add_subplot(grid[:2, 0], projection="3d")
    ax_info = fig.add_subplot(grid[2, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_density = fig.add_subplot(grid[1, 1])
    ax_flux = fig.add_subplot(grid[2, 1])

    reference_surface = snapshots[0, surface_indices]

    def draw(frame_pos: int) -> None:
        row_idx = int(frame_indices[frame_pos])
        points = snapshots[row_idx]
        surface = points[surface_indices]
        display_tets = (
            np.asarray(display_tets_snapshots[row_idx], dtype=int)
            if display_tets_snapshots is not None
            else np.asarray(tets, dtype=int)
        )
        edge_indices = base.tet_edge_indices(display_tets, max_edges=280)
        ax3d.cla()
        ax_info.cla()
        ax_amp.cla()
        ax_density.cla()
        ax_flux.cla()

        ax3d.add_collection3d(
            Poly3DCollection(
                reference_surface[faces],
                facecolor=(0.0, 0.0, 0.0, 0.025),
                edgecolor=(0.25, 0.25, 0.25, 0.22),
                linewidth=0.22,
            )
        )
        ax3d.add_collection3d(
            Poly3DCollection(
                surface[faces],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.28,
                alpha=0.46,
            )
        )
        ax3d.add_collection3d(
            Line3DCollection(points[edge_indices], colors=(0.1, 0.1, 0.1, 0.24), linewidths=0.32)
        )
        interior = np.setdiff1d(np.arange(points.shape[0], dtype=int), surface_indices)
        center = interior[np.linalg.norm(points[interior], axis=1) <= 1.0e-12]
        shells = np.setdiff1d(interior, center)
        if shells.size:
            shell_points = points[shells]
            ax3d.scatter(
                shell_points[:, 0],
                shell_points[:, 1],
                shell_points[:, 2],
                s=12,
                c="#2ca02c",
                alpha=0.82,
                edgecolors="none",
                depthshade=False,
            )
        if center.size:
            center_points = points[center]
            ax3d.scatter(
                center_points[:, 0],
                center_points[:, 1],
                center_points[:, 2],
                s=22,
                c="#111111",
                edgecolors="white",
                linewidths=0.35,
                depthshade=False,
            )
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_title(title, fontsize=10)
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))

        ax_info.axis("off")
        left = (
            f"t = {t[row_idx]:.5f} s\n"
            f"theory a2 = {amplitude_theory[row_idx]:+.4f}\n"
            f"fit a2 = {amplitude_fit[row_idx]:+.4f}\n"
            f"global dV/V0 = {global_volume_pct[row_idx]:+.3e} %\n"
            f"max tet |dV|/V = {local_volume_pct[row_idx]:.3e} %\n"
            f"rho spread = {density_spread_ppm[row_idx]:.3f} ppm\n"
            f"mass flux L1 = {mass_flux[row_idx]:.3e} kg/s\n"
            f"mom flux L1 = {momentum_flux[row_idx]:.3e} N\n"
            f"flux limiter = {limiter_scale[row_idx]:.3e}\n"
            f"retopology = {'active' if active_retopology else 'off/diagnostic'}"
        )
        retriangulation_line = (
            f"display Delaunay = {display_tet_count[row_idx]:.0f} tets\n"
            f"Delaunay flips = {display_changed_tets[row_idx]:.0f} ({100.0 * display_changed_fraction[row_idx]:.1f}%)"
            if show_retriangulation
            else f"display mesh = solver mesh\nsolver tets = {tets.shape[0]}"
        )
        right = (
            f"solver mesh = {points.shape[0]} verts, {tets.shape[0]} tets\n"
            f"{retriangulation_line}\n"
            f"internal faces = {display_face_count[row_idx]:.0f}\n"
            f"interior = {interior.size} verts\n"
            f"flux = {flux_scheme}, {face_flux_mode}\n"
            f"Fp = {pressure_force_label}\n"
            f"rho limit = {100.0 * density_limit:.2f} %\n"
            f"p_gas = {ambient_pressure[row_idx]:.0f} Pa\n"
            f"p_gauge = {pressure[row_idx]:.3f} Pa\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa\n"
            f"CPU = {cpu_time[row_idx]:.2f}/{cpu_total:.1f} s\n"
            f"CPU/step = {cpu_ms[row_idx]:.2f} ms"
        )
        ax_info.text(0.02, 0.98, left, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.6)
        ax_info.text(0.52, 0.98, right, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.6)

        pad = max(0.1 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.6, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=edgecolor, lw=1.8, label="Ftot + flux")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=5.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_xticklabels(time_tick_labels)
        ax_amp.set_ylim(float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - pad), float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + pad))
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        ax_density.plot(t, global_volume_pct, color=edgecolor, lw=1.8, label="global dV/V0")
        ax_density.plot(t, local_volume_pct, color="0.3", lw=1.2, ls=":", label="max local tet")
        ax_density.plot(t[row_idx], global_volume_pct[row_idx], "o", color="#d62728", ms=5.0)
        ax_density.set_xlim(float(t[0]), float(t[-1]))
        ax_density.set_xticks(time_ticks)
        ax_density.set_xticklabels(time_tick_labels)
        ymin = min(float(np.min(global_volume_pct)), float(np.min(local_volume_pct)))
        ymax = max(float(np.max(global_volume_pct)), float(np.max(local_volume_pct)))
        ypad = max(0.12 * (ymax - ymin), 1.0e-8)
        ax_density.set_ylim(ymin - ypad, ymax + ypad)
        ax_density.set_ylabel("percent")
        ax_density.set_title("Global and local volume error")
        ax_density.grid(True, alpha=0.25)
        ax_density.legend(frameon=False, fontsize=8, loc="upper right")

        mass_flux_plot = np.maximum(mass_flux, 1.0e-30)
        momentum_flux_plot = np.maximum(momentum_flux, 1.0e-30)
        ax_flux.plot(t, mass_flux_plot, color=edgecolor, lw=1.8, label="mass flux L1")
        ax_flux_twin = ax_flux.twinx()
        ax_flux_twin.plot(t, momentum_flux_plot, color="0.25", lw=1.2, ls=":", label="momentum flux L1")
        ax_flux.plot(t[row_idx], max(mass_flux[row_idx], 1.0e-30), "o", color="#d62728", ms=5.0)
        ax_flux.set_yscale("log")
        ax_flux_twin.set_yscale("log")
        mass_flux_max = max(float(np.max(mass_flux_plot)), 1.0e-30)
        momentum_flux_max = max(float(np.max(momentum_flux_plot)), 1.0e-30)
        ax_flux.set_ylim(max(mass_flux_max * 1.0e-6, 1.0e-30), mass_flux_max * 5.0)
        ax_flux_twin.set_ylim(max(momentum_flux_max * 1.0e-6, 1.0e-30), momentum_flux_max * 5.0)
        ax_flux.set_xlim(float(t[0]), float(t[-1]))
        ax_flux.set_xticks(time_ticks)
        ax_flux.set_xticklabels(time_tick_labels)
        ax_flux.set_xlabel("t [s]")
        ax_flux.set_ylabel("kg/s")
        ax_flux_twin.set_ylabel("N")
        ax_flux.set_title("Internal face fluxes")
        ax_flux.grid(True, alpha=0.25)
        lines, labels = ax_flux.get_legend_handles_labels()
        lines2, labels2 = ax_flux_twin.get_legend_handles_labels()
        ax_flux.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")

        fig.suptitle(
            f"{closure.capitalize()} flux-aware run, CPU {cpu_total:.1f} s ({cpu_avg:.1f} ms/step)",
            y=0.985,
        )

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / gif_name
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux-aware FHeron compressible/incompressible droplet GIFs.")
    parser.add_argument("--radius", type=float, default=1.0e-3)
    parser.add_argument("--gamma", type=float, default=0.072)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--bulk-modulus", type=float, default=2.2e9)
    parser.add_argument(
        "--ambient-pressure",
        type=float,
        default=101325.0,
        help="Passive gas/reference pressure [Pa]. Dynamics use liquid gauge pressure relative to this value.",
    )
    parser.add_argument("--rayleigh-mode", type=int, default=2)
    parser.add_argument("--shape-mode-ar", type=float, default=1.05)
    parser.add_argument(
        "--t-final",
        type=float,
        default=None,
        help="Physical simulation end time [s]. Default is one Rayleigh-Lamb period.",
    )
    parser.add_argument("--subdivision", type=int, default=0, help="0 is fast and still has two interior shells.")
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--substeps-per-sample", type=int, default=20)
    parser.add_argument("--damping-ratio", type=float, default=0.015)
    parser.add_argument(
        "--mass-model",
        choices=("star-volume", "rayleigh-added"),
        default="star-volume",
        help="Vertex inertia model. rayleigh-added reproduces Rayleigh-Lamb inertia while tet mass remains physical.",
    )
    parser.add_argument(
        "--inertia-scale",
        type=float,
        default=1.0,
        help="Scale only the vertex inertial mass used by Ftot integration; tet mass/density/flux stay unchanged.",
    )
    parser.add_argument(
        "--mass-flux-coupling",
        type=float,
        default=1.0e-6,
        help="Fraction of computed ALE mass flux used to update cell/nodal masses.",
    )
    parser.add_argument(
        "--momentum-flux-coupling",
        type=float,
        default=0.0,
        help="Fraction of computed momentum flux applied as nodal force. Default keeps it diagnostic.",
    )
    parser.add_argument(
        "--density-change-limit",
        type=float,
        default=None,
        help="Optional conservative limiter for |rho/rho0 - 1|. Example: 0.01 allows 1 percent.",
    )
    parser.add_argument(
        "--flux-cfl-limit",
        type=float,
        default=0.25,
        help="Limit the mass transferred by internal flux in one substep as a fraction of cell mass.",
    )
    parser.add_argument(
        "--flux-scheme",
        choices=("upwind", "rusanov"),
        default="upwind",
        help="Internal ALE face flux scheme. Rusanov adds acoustic numerical diffusion.",
    )
    parser.add_argument(
        "--face-flux-mode",
        choices=("cell-average", "lagrangian"),
        default="cell-average",
        help="Use old cell-averaged face flux or Lagrangian material-face flux.",
    )
    parser.add_argument(
        "--flux-timing",
        choices=("pre-pressure", "post-pressure"),
        default="pre-pressure",
        help="Apply active flux before the pressure solve or as an operator-split update after pressure.",
    )
    parser.add_argument(
        "--mass-floor-fraction",
        type=float,
        default=0.35,
        help="Minimum tet mass as a fixed fraction of the initial tet mass.",
    )
    parser.add_argument(
        "--coupled-flux-continuity",
        action="store_true",
        help="Include the linearized active mass flux in the compressible EOS pressure-correction equation.",
    )
    parser.add_argument(
        "--flux-continuity-sign",
        type=float,
        default=-1.0,
        help="Sign for the mass-flux term in the coupled continuity operator; -1 gives B - G/rho0.",
    )
    parser.add_argument(
        "--incompressible-target-mode",
        choices=("reference", "mass"),
        default="reference",
        help="For incompressible closure, project to fixed reference tet volumes or Vtar=m_t/rho after mass flux.",
    )
    parser.add_argument(
        "--pressure-force-mode",
        choices=(
            "volume-gradient",
            "pairwise-dual",
            "pairwise-dual-scalar",
            "pairwise-dual-forcefit",
            "pairwise-dual-consistent",
            "pairwise-vertex-eos",
        ),
        default="volume-gradient",
        help=(
            "Use tet volume-gradient pressure force, pairwise -0.5(p_i+p_j) A_ij force "
            "with mapped vertex pressure, pairwise scalar-pressure force, fitted vertex pressure, "
            "fully pairwise-consistent solve, or direct vertex-pressure EOS solve."
        ),
    )
    parser.add_argument(
        "--retriangulation-mode",
        choices=("none", "diagnostic", "active"),
        default="none",
        help=(
            "none: fixed solver tets; diagnostic: Delaunay display mesh only; "
            "active: rebuild solver tets each substep and remap mass/target volumes."
        ),
    )
    parser.add_argument(
        "--closure",
        choices=("both", "compressible", "incompressible"),
        default="both",
        help="Run both closures or just one closure.",
    )
    parser.add_argument("--animation-max-frames", type=int, default=32)
    parser.add_argument("--animation-fps", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "out" / "sphere_fheron_flux_fv")
    return parser.parse_args()


def main() -> int:
    install_ddgclib_operator_aliases()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    omega, frequency, period = base.rayleigh_lamb_frequency(args.rayleigh_mode, args.gamma, args.rho, args.radius)
    t_final = float(args.t_final) if args.t_final is not None else period
    amplitude = base.shape_amplitude_from_aspect_ratio(args.shape_mode_ar)

    outputs: list[Path] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        closures = ("compressible", "incompressible") if args.closure == "both" else (args.closure,)
        for closure in closures:
            rows, snapshots, faces, tets, surface_indices, display_tets_snapshots = run_flux_dynamic(
                closure=closure,
                subdivisions=args.subdivision,
                radius=args.radius,
                gamma=args.gamma,
                rho=args.rho,
                bulk_modulus=args.bulk_modulus,
                amplitude=amplitude,
                n_steps=args.steps,
                t_final=t_final,
                substeps_per_sample=args.substeps_per_sample,
                damping_ratio=args.damping_ratio,
                mass_model=args.mass_model,
                inertia_scale=args.inertia_scale,
                mass_flux_coupling=args.mass_flux_coupling,
                momentum_flux_coupling=args.momentum_flux_coupling,
                density_change_limit=args.density_change_limit,
                flux_cfl_limit=args.flux_cfl_limit,
                flux_scheme=args.flux_scheme,
                flux_timing=args.flux_timing,
                face_flux_mode=args.face_flux_mode,
                mass_floor_fraction=args.mass_floor_fraction,
                coupled_flux_continuity=args.coupled_flux_continuity,
                flux_continuity_sign=args.flux_continuity_sign,
                incompressible_target_mode=args.incompressible_target_mode,
                pressure_force_mode=args.pressure_force_mode,
                rayleigh_frequency_hz=frequency,
                rayleigh_omega_rad_s=omega,
                rayleigh_mode=args.rayleigh_mode,
                ambient_pressure=args.ambient_pressure,
                retriangulation_mode=args.retriangulation_mode,
            )
            csv_path, json_path = write_rows(rows, args.out_dir, f"flux_{closure}_timeseries")
            npz_path = write_snapshots(
                snapshots,
                faces,
                tets,
                surface_indices,
                args.out_dir,
                f"flux_{closure}_snapshots",
                display_tets_snapshots,
            )
            gif_path = write_flux_animation(
                rows,
                snapshots,
                faces,
                tets,
                surface_indices,
                args.out_dir,
                radius=args.radius,
                closure=closure,
                max_frames=args.animation_max_frames,
                fps=args.animation_fps,
                display_tets_snapshots=display_tets_snapshots,
            )
            outputs.extend([csv_path, json_path, npz_path, gif_path])
            print(
                f"{closure}: CPU={rows[-1]['cpu_time_s']:.3f}s, "
                f"ms/step={rows[-1]['cpu_ms_per_substep']:.3f}, "
                f"mass_flux_L1={rows[-1]['mass_flux_l1_kg_s']:.3e}, "
                f"mom_flux_L1={rows[-1]['momentum_flux_l1_n']:.3e}"
            )

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
