"""Two-phase HC/FHeron droplet benchmark with an explicit outer gas mesh.

This preview case is closer to the GitHub ``oscillating_droplet`` setup than
the one-phase free-surface GIFs: it has liquid tets inside the interface, gas
tets outside, and shared interface vertices.  It remains a compact benchmark,
not a full replacement for the upstream multiphase package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np

import sphere_fheron_eos_projection_benchmark as base

plt = base.plt
FuncAnimation = base.FuncAnimation
PillowWriter = base.PillowWriter
Line3DCollection = base.Line3DCollection
Poly3DCollection = base.Poly3DCollection


SCRIPT_DIR = Path(__file__).resolve().parent
LIQUID = 1
GAS = 0


def build_liquid_gas_mesh(
    subdivisions: int,
    radius: float,
    outer_radius_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unit_points, surface_faces = base.icosphere(subdivisions, 1.0)
    n_surface = int(unit_points.shape[0])

    center = np.zeros((1, 3), dtype=float)
    liquid_shells = [0.45, 0.75, 1.0]
    liquid_points = np.vstack([center, *(unit_points * (shell * radius) for shell in liquid_shells)])
    interface_offset = 1 + 2 * n_surface
    gas_outer_offset = 1 + 3 * n_surface
    outer_points = unit_points * (float(outer_radius_scale) * float(radius))
    points = np.vstack([liquid_points, outer_points])

    triangulation = base.Delaunay(liquid_points, qhull_options="Qbb Qc")
    raw_liquid_tets = np.asarray(triangulation.simplices, dtype=int)
    valid = np.all(raw_liquid_tets < liquid_points.shape[0], axis=1)
    raw_liquid_tets = raw_liquid_tets[valid]
    liquid_volumes = base.tet_cell_volumes(liquid_points, raw_liquid_tets)
    positive = liquid_volumes[liquid_volumes > 0.0]
    min_volume = 0.1 * float(np.median(positive)) if positive.size else 0.0
    liquid_tets = raw_liquid_tets[liquid_volumes > min_volume]

    gas_tets: list[list[int]] = []
    for a, b, c in np.asarray(surface_faces, dtype=int):
        ia, ib, ic = interface_offset + int(a), interface_offset + int(b), interface_offset + int(c)
        oa, ob, oc = gas_outer_offset + int(a), gas_outer_offset + int(b), gas_outer_offset + int(c)
        gas_tets.extend(
            [
                [ia, ib, ic, oc],
                [ia, ib, ob, oc],
                [ia, oa, ob, oc],
            ]
        )

    gas_tets_array = np.asarray(gas_tets, dtype=int)
    tets = np.vstack([liquid_tets, gas_tets_array])
    phases = np.concatenate(
        [
            np.full(liquid_tets.shape[0], LIQUID, dtype=int),
            np.full(gas_tets_array.shape[0], GAS, dtype=int),
        ]
    )
    interface_indices = np.arange(interface_offset, interface_offset + n_surface, dtype=int)
    outer_indices = np.arange(gas_outer_offset, gas_outer_offset + n_surface, dtype=int)
    liquid_vertex_count = liquid_points.shape[0]
    return points, tets, phases, interface_indices, outer_indices, surface_faces, unit_points, np.arange(liquid_vertex_count)


def deform_initial_interface(
    points: np.ndarray,
    tets: np.ndarray,
    phases: np.ndarray,
    liquid_vertex_indices: np.ndarray,
    radius: float,
    amplitude: float,
) -> np.ndarray:
    deformed = points.copy()
    liquid_tets = tets[phases == LIQUID]
    liquid_points = points[liquid_vertex_indices]
    target_volume = float(np.sum(base.tet_cell_volumes(liquid_points, liquid_tets)))
    deformed_liquid = base.deform_l2_volume_points(
        liquid_points,
        liquid_tets,
        radius=radius,
        amplitude=amplitude,
        target_volume=target_volume,
    )
    deformed[liquid_vertex_indices] = deformed_liquid
    return deformed


def nodal_from_tets(n_vertices: int, tets: np.ndarray, tet_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros(n_vertices, dtype=float)
    for tet, value in zip(np.asarray(tets, dtype=int), np.asarray(tet_values, dtype=float)):
        share = 0.25 * float(value)
        for vertex_idx in tet:
            nodal[int(vertex_idx)] += share
    return nodal


def row_array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def pressure_project_velocity(
    velocities: np.ndarray,
    volume_matrix: np.ndarray,
    inv_mass_dof: np.ndarray,
    target_rates: np.ndarray,
) -> np.ndarray:
    velocity_vec = velocities.reshape(-1)
    stiffness = (volume_matrix * inv_mass_dof[None, :]) @ volume_matrix.T
    rhs = volume_matrix @ velocity_vec - np.asarray(target_rates, dtype=float)
    multipliers = base._solve_regularized(stiffness, rhs)
    projected = velocity_vec - inv_mass_dof * (volume_matrix.T @ multipliers)
    return projected.reshape(velocities.shape)


def write_rows(rows: list[dict[str, float]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def run_twophase(
    *,
    closure: str,
    subdivisions: int,
    radius: float,
    outer_radius_scale: float,
    gamma: float,
    rho_liquid: float,
    rho_gas: float,
    bulk_liquid: float,
    bulk_gas: float,
    amplitude: float,
    n_steps: int,
    t_final: float,
    substeps_per_sample: int,
    inertia_scale: float,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
    rayleigh_mode: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if closure not in {"compressible", "incompressible"}:
        raise ValueError("closure must be compressible or incompressible")

    reference_points, tets, phases, interface_indices, outer_indices, surface_faces, _unit, liquid_vertex_indices = (
        build_liquid_gas_mesh(subdivisions, radius, outer_radius_scale)
    )
    points = deform_initial_interface(reference_points, tets, phases, liquid_vertex_indices, radius, amplitude)
    fixed_points = reference_points[outer_indices].copy()
    fixed_mask = np.zeros(points.shape[0], dtype=bool)
    fixed_mask[outer_indices] = True

    reference_tet_volumes, _matrix0 = base.tet_volume_matrix(points, tets)
    rho_phase = np.where(phases == LIQUID, float(rho_liquid), float(rho_gas))
    bulk_phase = np.where(phases == LIQUID, float(bulk_liquid), float(bulk_gas))
    tet_masses = rho_phase * reference_tet_volumes
    reference_total_mass = float(np.sum(tet_masses))
    reference_liquid_volume = float(np.sum(reference_tet_volumes[phases == LIQUID]))
    reference_gas_volume = float(np.sum(reference_tet_volumes[phases == GAS]))
    target_volumes = reference_tet_volumes.copy()

    velocities = np.zeros_like(points)
    nodal_mass0 = nodal_from_tets(points.shape[0], tets, tet_masses)
    inertial_mass_addition = np.zeros(points.shape[0], dtype=float)
    reference_scalar_areas = base.vertex_scalar_areas(reference_points[interface_indices], surface_faces)
    surface_added_mass = (float(rho_liquid) + float(rho_gas)) * float(radius) / float(rayleigh_mode)
    inertial_mass_addition[interface_indices] = surface_added_mass * reference_scalar_areas

    _forces0, _areas0, reference_pressure, _volume0 = base.fheron_forces_for_points(
        reference_points[interface_indices],
        surface_faces,
        gamma,
    )

    dt = float(t_final) / float((int(n_steps) - 1) * int(substeps_per_sample))
    rows: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    def nodal_masses() -> np.ndarray:
        return np.maximum(float(inertia_scale) * (nodal_mass0 + inertial_mass_addition), 1.0e-18)

    def inverse_mass_dof(masses: np.ndarray) -> np.ndarray:
        inv = np.repeat(1.0 / np.maximum(masses, 1.0e-300), 3)
        fixed_dofs = np.repeat(fixed_mask, 3)
        inv[fixed_dofs] = 0.0
        return inv

    def collect(sample_index: int, time_value: float) -> None:
        tet_volumes, _local_matrix = base.tet_volume_matrix(points, tets)
        interface_points = points[interface_indices]
        _surface_forces, _surface_area_vectors, fheron_equiv, _surface_volume = base.fheron_forces_for_points(
            interface_points,
            surface_faces,
            gamma,
        )
        fitted_amplitude = base.fit_l2_shape_amplitude(interface_points, surface_faces)
        theory_amplitude = float(amplitude) * math.cos(float(rayleigh_omega_rad_s) * float(time_value))
        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start
        substeps_done = int(sample_index) * int(substeps_per_sample)
        liquid_volumes = tet_volumes[phases == LIQUID]
        gas_volumes = tet_volumes[phases == GAS]
        liquid_rel = liquid_volumes / np.maximum(target_volumes[phases == LIQUID], 1.0e-300)
        gas_rel = gas_volumes / np.maximum(target_volumes[phases == GAS], 1.0e-300)

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
                "liquid_pressure_gauge_pa": float(getattr(collect, "last_liquid_pressure", reference_pressure)),
                "gas_pressure_gauge_pa": float(getattr(collect, "last_gas_pressure", 0.0)),
                "reference_pressure_pa": float(reference_pressure),
                "total_mass_kg": float(np.sum(tet_masses)),
                "mass_rel_error": float(np.sum(tet_masses) / reference_total_mass - 1.0),
                "liquid_vol_rel": float(np.sum(liquid_volumes) / reference_liquid_volume),
                "gas_vol_rel": float(np.sum(gas_volumes) / reference_gas_volume),
                "liquid_local_volume_rel_max_abs": float(np.max(np.abs(liquid_rel - 1.0))),
                "gas_local_volume_rel_max_abs": float(np.max(np.abs(gas_rel - 1.0))),
                "n_vertices": float(points.shape[0]),
                "n_tets": float(tets.shape[0]),
                "n_liquid_tets": float(np.sum(phases == LIQUID)),
                "n_gas_tets": float(np.sum(phases == GAS)),
                "n_interface_vertices": float(interface_indices.shape[0]),
                "n_gas_vertices": float(points.shape[0] - liquid_vertex_indices.shape[0]),
                "cpu_time_s": float(cpu_elapsed),
                "wall_time_s": float(wall_elapsed),
                "cpu_ms_per_substep": float(1.0e3 * cpu_elapsed / float(substeps_done)) if substeps_done else 0.0,
                "max_speed_m_s": float(np.max(np.linalg.norm(velocities, axis=1))),
                "mass_flux_l1_kg_s": 0.0,
                "momentum_flux_l1_n": 0.0,
            }
        )
        snapshots.append(points.copy())

    collect.last_liquid_pressure = reference_pressure  # type: ignore[attr-defined]
    collect.last_gas_pressure = 0.0  # type: ignore[attr-defined]
    collect(0, 0.0)

    total_substeps = (int(n_steps) - 1) * int(substeps_per_sample)
    sample_index = 1
    fixed_reference = reference_points.copy()
    for substep in range(1, total_substeps + 1):
        tet_volumes, local_matrix = base.tet_volume_matrix(points, tets)
        masses = nodal_masses()
        inv_mass = inverse_mass_dof(masses)

        interface_points = points[interface_indices]
        surface_forces, _surface_areas, _fheron_equiv, _surface_volume = base.fheron_forces_for_points(
            interface_points,
            surface_faces,
            gamma,
        )
        forces = np.zeros_like(points)
        forces[interface_indices] += surface_forces
        pressure_matrix = local_matrix.T

        if closure == "compressible":
            base_pressures = np.where(phases == LIQUID, float(reference_pressure), 0.0)
            nonpressure = forces + (pressure_matrix @ base_pressures).reshape(points.shape)
            u_star = velocities + dt * (inv_mass * nonpressure.reshape(-1)).reshape(points.shape)
            u_star[fixed_mask] = 0.0
            compressibility = bulk_phase / np.maximum(reference_tet_volumes, 1.0e-300)
            stiffness = (local_matrix * inv_mass[None, :]) @ pressure_matrix
            linear_error = tet_volumes - target_volumes + dt * (local_matrix @ u_star.reshape(-1))
            system = np.eye(tets.shape[0]) + (compressibility[:, None] * dt * dt) * stiffness
            rhs = -compressibility * linear_error
            pressure_delta = base._solve_regularized(system, rhs)
            velocities = (u_star.reshape(-1) + dt * inv_mass * (pressure_matrix @ pressure_delta)).reshape(points.shape)
            collect.last_liquid_pressure = float(np.mean(base_pressures[phases == LIQUID] + pressure_delta[phases == LIQUID]))  # type: ignore[attr-defined]
            collect.last_gas_pressure = float(np.mean(pressure_delta[phases == GAS]))  # type: ignore[attr-defined]
        else:
            target_rates = -(tet_volumes - target_volumes) / float(dt)
            velocity_vec = velocities.reshape(-1)
            nonpressure_vec = forces.reshape(-1)
            stiffness = (local_matrix * inv_mass[None, :]) @ pressure_matrix
            rhs = (target_rates - local_matrix @ velocity_vec) / float(dt) - local_matrix @ (inv_mass * nonpressure_vec)
            pressure_values = base._solve_regularized(stiffness, rhs)
            total_forces = forces + (pressure_matrix @ pressure_values).reshape(points.shape)
            velocities = velocities + dt * (inv_mass * total_forces.reshape(-1)).reshape(points.shape)
            velocities = pressure_project_velocity(velocities, local_matrix, inv_mass, target_rates)
            collect.last_liquid_pressure = float(np.mean(pressure_values[phases == LIQUID]))  # type: ignore[attr-defined]
            collect.last_gas_pressure = float(np.mean(pressure_values[phases == GAS]))  # type: ignore[attr-defined]

        velocities[fixed_mask] = 0.0
        points = points + dt * velocities
        points[outer_indices] = fixed_reference[outer_indices]
        velocities[fixed_mask] = 0.0

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(velocities)):
            raise FloatingPointError(f"{closure} two-phase run became non-finite at substep {substep}")

        if substep % int(substeps_per_sample) == 0:
            collect(sample_index, sample_index * float(t_final) / float(int(n_steps) - 1))
            sample_index += 1

    return rows, np.asarray(snapshots, dtype=float), surface_faces, tets, phases, interface_indices, outer_indices


def write_snapshots(
    snapshots: np.ndarray,
    faces: np.ndarray,
    tets: np.ndarray,
    phases: np.ndarray,
    interface_indices: np.ndarray,
    outer_indices: np.ndarray,
    out_dir: Path,
    stem: str,
) -> Path:
    path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        points=snapshots,
        faces=faces,
        tets=tets,
        phases=phases,
        interface_indices=interface_indices,
        outer_indices=outer_indices,
    )
    return path


def write_animation(
    rows: list[dict[str, float]],
    snapshots: np.ndarray,
    faces: np.ndarray,
    tets: np.ndarray,
    phases: np.ndarray,
    interface_indices: np.ndarray,
    outer_indices: np.ndarray,
    out_dir: Path,
    *,
    closure: str,
    max_frames: int,
    fps: int,
) -> Path:
    t = row_array(rows, "t")
    amplitude_theory = row_array(rows, "shape_amplitude_theory")
    amplitude_fit = row_array(rows, "shape_amplitude_fit")
    liquid_vol_pct = (row_array(rows, "liquid_vol_rel") - 1.0) * 100.0
    gas_vol_pct = (row_array(rows, "gas_vol_rel") - 1.0) * 100.0
    liquid_local_pct = row_array(rows, "liquid_local_volume_rel_max_abs") * 100.0
    gas_local_pct = row_array(rows, "gas_local_volume_rel_max_abs") * 100.0
    liquid_pressure = row_array(rows, "liquid_pressure_gauge_pa")
    gas_pressure = row_array(rows, "gas_pressure_gauge_pa")
    cpu_time = row_array(rows, "cpu_time_s")

    if closure == "compressible":
        liquid_color = "#1f77b4"
        gif_name = "flux_compressible_twophase_sphere.gif"
        title = "#11 two-phase compressible: liquid droplet + gas mesh"
    else:
        liquid_color = "#ff7f0e"
        gif_name = "flux_incompressible_twophase_sphere.gif"
        title = "#12 two-phase incompressible: liquid droplet + gas mesh"

    gas_color = "#8fb8de"
    liquid_tets = tets[phases == LIQUID]
    gas_tets = tets[phases == GAS]
    liquid_edges = base.tet_edge_indices(liquid_tets, max_edges=250)
    gas_edges = base.tet_edge_indices(gas_tets, max_edges=420)
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), int(max_frames)), dtype=int))
    outer_reference = snapshots[0, outer_indices]
    max_radius = float(np.max(np.linalg.norm(snapshots.reshape((-1, 3)), axis=1)))
    limit = 1.08 * max_radius
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)

    fig = plt.figure(figsize=(5.2, 7.3))
    grid = fig.add_gridspec(3, 1, height_ratios=(3.25, 1.0, 1.0), hspace=0.24)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax_amp = fig.add_subplot(grid[1])
    ax_vol = fig.add_subplot(grid[2])

    def draw(frame_pos: int) -> None:
        row_idx = int(frame_indices[frame_pos])
        points = snapshots[row_idx]
        surface = points[interface_indices]
        outer = points[outer_indices]

        ax3d.cla()
        ax_amp.cla()
        ax_vol.cla()

        ax3d.add_collection3d(
            Poly3DCollection(
                outer[faces],
                facecolor=(0.52, 0.70, 0.88, 0.085),
                edgecolor=(0.22, 0.34, 0.50, 0.20),
                linewidth=0.30,
            )
        )
        ax3d.add_collection3d(
            Line3DCollection(points[gas_edges], colors=(0.20, 0.35, 0.55, 0.19), linewidths=0.32)
        )
        ax3d.add_collection3d(
            Poly3DCollection(
                surface[faces],
                facecolor=liquid_color,
                edgecolor=liquid_color,
                linewidth=0.38,
                alpha=0.52,
            )
        )
        ax3d.add_collection3d(
            Line3DCollection(points[liquid_edges], colors=(0.05, 0.05, 0.05, 0.30), linewidths=0.34)
        )
        gas_vertices = np.setdiff1d(np.unique(gas_tets.reshape(-1)), interface_indices)
        ax3d.scatter(
            points[gas_vertices, 0],
            points[gas_vertices, 1],
            points[gas_vertices, 2],
            s=5,
            c=gas_color,
            alpha=0.48,
            edgecolors="none",
            depthshade=False,
        )
        interior = np.setdiff1d(np.unique(liquid_tets.reshape(-1)), interface_indices)
        if interior.size:
            ax3d.scatter(
                points[interior, 0],
                points[interior, 1],
                points[interior, 2],
                s=6,
                c="#2ca02c",
                alpha=0.65,
                edgecolors="none",
                depthshade=False,
            )

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=8.8, pad=4)
        ax3d.text2D(
            0.02,
            0.97,
            (
                f"t = {t[row_idx]:.4f} s\n"
                f"phase 1 liquid tets = {int(rows[row_idx]['n_liquid_tets'])}\n"
                f"phase 0 gas tets = {int(rows[row_idx]['n_gas_tets'])}\n"
                f"interface vertices = {int(rows[row_idx]['n_interface_vertices'])}\n"
                f"p_liq gauge = {liquid_pressure[row_idx]:.1f} Pa\n"
                f"p_gas gauge = {gas_pressure[row_idx]:.2e} Pa\n"
                f"CPU = {cpu_time[row_idx]:.1f}/{cpu_time[-1]:.1f} s"
            ),
            transform=ax3d.transAxes,
            va="top",
            ha="left",
            fontsize=7.3,
            family="monospace",
        )

        pad = max(0.1 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.4, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=liquid_color, lw=1.7, label="simulation")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=4.2)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(
            float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - pad),
            float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + pad),
        )
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_ylabel("a2", fontsize=8)
        ax_amp.set_title("Shape amplitude", fontsize=8)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=7, loc="upper right")
        ax_amp.tick_params(labelsize=7)

        ax_vol.plot(t, liquid_vol_pct, color=liquid_color, lw=1.5, label="liquid global dV/V")
        ax_vol.plot(t, gas_vol_pct, color="#4c78a8", lw=1.2, ls="--", label="gas global dV/V")
        ax_vol.plot(t, liquid_local_pct, color="#111111", lw=1.0, ls=":", label="liquid local max")
        ax_vol.plot(t, gas_local_pct, color="#777777", lw=1.0, ls="-.", label="gas local max")
        ax_vol.plot(t[row_idx], liquid_vol_pct[row_idx], "o", color="#d62728", ms=4.0)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        ax_vol.set_xticks(time_ticks)
        ax_vol.set_xlabel("t [s]", fontsize=8)
        ax_vol.set_ylabel("%", fontsize=8)
        ax_vol.set_title("Phase volume error", fontsize=8)
        ax_vol.grid(True, alpha=0.25)
        ax_vol.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax_vol.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.06)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / gif_name
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase liquid/gas HC/FHeron droplet benchmark.")
    parser.add_argument("--radius", type=float, default=1.0e-3)
    parser.add_argument("--outer-radius-scale", type=float, default=1.60)
    parser.add_argument("--gamma", type=float, default=0.072)
    parser.add_argument("--rho-liquid", type=float, default=1000.0)
    parser.add_argument("--rho-gas", type=float, default=1.2)
    parser.add_argument("--bulk-liquid", type=float, default=2.2e9)
    parser.add_argument("--bulk-gas", type=float, default=1.42e5)
    parser.add_argument("--rayleigh-mode", type=int, default=2)
    parser.add_argument("--shape-mode-ar", type=float, default=1.05)
    parser.add_argument("--t-final", type=float, default=0.016)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--steps", type=int, default=81)
    parser.add_argument("--substeps-per-sample", type=int, default=10)
    parser.add_argument("--inertia-scale", type=float, default=1.08)
    parser.add_argument("--closure", choices=("both", "compressible", "incompressible"), default="both")
    parser.add_argument("--animation-max-frames", type=int, default=32)
    parser.add_argument("--animation-fps", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "out" / "sphere_fheron_twophase_gasmesh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    omega, frequency, _period = base.rayleigh_lamb_frequency(args.rayleigh_mode, args.gamma, args.rho_liquid, args.radius)
    amplitude = base.shape_amplitude_from_aspect_ratio(args.shape_mode_ar)
    closures = ("compressible", "incompressible") if args.closure == "both" else (args.closure,)
    outputs: list[Path] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for closure in closures:
            rows, snapshots, faces, tets, phases, interface_indices, outer_indices = run_twophase(
                closure=closure,
                subdivisions=args.subdivision,
                radius=args.radius,
                outer_radius_scale=args.outer_radius_scale,
                gamma=args.gamma,
                rho_liquid=args.rho_liquid,
                rho_gas=args.rho_gas,
                bulk_liquid=args.bulk_liquid,
                bulk_gas=args.bulk_gas,
                amplitude=amplitude,
                n_steps=args.steps,
                t_final=args.t_final,
                substeps_per_sample=args.substeps_per_sample,
                inertia_scale=args.inertia_scale,
                rayleigh_frequency_hz=frequency,
                rayleigh_omega_rad_s=omega,
                rayleigh_mode=args.rayleigh_mode,
            )
            csv_path, json_path = write_rows(rows, args.out_dir, f"twophase_{closure}_timeseries")
            npz_path = write_snapshots(
                snapshots,
                faces,
                tets,
                phases,
                interface_indices,
                outer_indices,
                args.out_dir,
                f"twophase_{closure}_snapshots",
            )
            gif_path = write_animation(
                rows,
                snapshots,
                faces,
                tets,
                phases,
                interface_indices,
                outer_indices,
                args.out_dir,
                closure=closure,
                max_frames=args.animation_max_frames,
                fps=args.animation_fps,
            )
            outputs.extend([csv_path, json_path, npz_path, gif_path])
            print(
                f"{closure}: CPU={rows[-1]['cpu_time_s']:.3f}s, "
                f"liquid_local={rows[-1]['liquid_local_volume_rel_max_abs']:.3e}, "
                f"gas_local={rows[-1]['gas_local_volume_rel_max_abs']:.3e}, "
                f"p_liq={rows[-1]['liquid_pressure_gauge_pa']:.3f}, "
                f"p_gas={rows[-1]['gas_pressure_gauge_pa']:.3e}"
            )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
