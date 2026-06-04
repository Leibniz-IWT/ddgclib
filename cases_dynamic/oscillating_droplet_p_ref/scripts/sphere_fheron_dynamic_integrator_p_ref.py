"""Built-in dynamic-integrator validation for the pressure-reference droplet.

This runner validates the Stefan/Mädler pressure-reference implementation:

* the state/cache object is ``ddgclib.operators.stress.VolumeGradientPressureState``;
* the acceleration is the dynamic-integrator ``dudt_fn``
  ``volume_gradient_pressure_acceleration(v, state=...)``;
* ``symplectic_euler`` advances the vertices with ``du/dt=Ftot/m`` and
  ``dx/dt=u``;
* active-retopology cases rebuild the tet-volume pressure operator through the
  integrator ``retopologize_fn`` callback before each step.

References in the operator comments: Chorin (1968), Hirt et al. (1974),
Toro (2009), and the ddgclib/HyperCT HC Heron curvature construction.
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

from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators import multiphase_stress as ddg_multiphase
from ddgclib.operators import stress as ddg_stress

import sphere_fheron_eos_projection_benchmark as base
import sphere_fheron_twophase_gasmesh_benchmark as twophase


SCRIPT_DIR = Path(__file__).resolve().parent
LIQUID = 1
GAS = 0


def build_vertex_complex(
    points: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    vertex_phases: np.ndarray | None = None,
):
    """Build a minimal HC vertex complex for the built-in ODE integrator."""
    from hyperct import Complex

    HC = Complex(3, domain=None)
    vertices = []
    for index, point in enumerate(np.asarray(points, dtype=float)):
        vertex = HC.V[tuple(float(x) for x in point)]
        vertex.u = np.asarray(velocities[index], dtype=float).copy()
        vertex.m = float(masses[index])
        vertex.p = 0.0
        vertex.phase = int(vertex_phases[index]) if vertex_phases is not None else 1
        vertex.boundary = False
        setattr(vertex, ddg_stress.VolumeGradientPressureState.vertex_index_attr, int(index))
        vertices.append(vertex)
    return HC, vertices


def write_rows(rows: list[dict[str, float]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def run_case(
    *,
    case_id: int,
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
    inertia_scale: float,
    mass_flux_coupling: float,
    momentum_flux_coupling: float,
    density_change_limit: float | None,
    face_flux_mode: str,
    active_retopology: bool,
    rayleigh_mode: int,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_points, tets, surface_indices, surface_faces, _surface_faces_global, _unit_points = base.build_ball_tet_mesh(
        subdivisions,
        radius,
    )
    points = base.deform_l2_volume_points(
        reference_points,
        tets,
        radius=radius,
        amplitude=amplitude,
        target_volume=float(np.sum(ddg_stress.tet_cell_volumes(reference_points, tets))),
    )
    reference_tet_volumes, _matrix0 = ddg_stress.tet_volume_matrix(points, tets)
    tet_masses = float(rho) * reference_tet_volumes
    nodal_masses = ddg_stress.tet_volume_lumped_nodal_values(points.shape[0], tets, tet_masses)
    velocities = np.zeros_like(points)

    _forces0, _areas0, reference_pressure, _volume0 = ddg_stress.heron_forces_for_points(
        reference_points[surface_indices],
        surface_faces,
        gamma,
    )
    dt = float(t_final) / float((int(n_steps) - 1) * int(substeps_per_sample))
    damping_rate = 2.0 * float(damping_ratio) * float(rayleigh_omega_rad_s)
    sound_speed = math.sqrt(float(bulk_modulus) / float(rho))

    HC, _vertices = build_vertex_complex(points, velocities, np.maximum(nodal_masses, 1.0e-18))
    bV: set = set()
    state = ddg_stress.VolumeGradientPressureState(
        tets=tets,
        surface_indices=surface_indices,
        surface_faces=surface_faces,
        gamma=gamma,
        density=rho,
        bulk_modulus=bulk_modulus,
        closure=closure,
        reference_tet_volumes=reference_tet_volumes,
        tet_masses=tet_masses,
        base_tet_pressures=reference_pressure,
        dt=dt,
        inertia_scale=inertia_scale,
        inertial_mass_addition=np.zeros(points.shape[0], dtype=float),
        damping_rate=damping_rate,
        active_retopology=active_retopology,
        mass_flux_coupling=mass_flux_coupling,
        momentum_flux_coupling=momentum_flux_coupling,
        density_change_limit=density_change_limit,
        face_flux_mode=face_flux_mode,
        sound_speed=sound_speed,
        incompressible_target_mode="reference",
    )
    state.HC = HC

    rows: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    display_tets: list[np.ndarray] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    def collect(sample: int, time_value: float) -> None:
        current_points, current_velocities = state.read_hc_state(HC)
        stats = state.diagnostics(HC)
        surface_points = current_points[surface_indices]
        fitted_amplitude = base.fit_l2_shape_amplitude(surface_points, surface_faces)
        theory_amplitude = float(amplitude) * math.cos(float(rayleigh_omega_rad_s) * float(time_value))
        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start
        substeps_done = int(sample) * int(substeps_per_sample)
        rows.append(
            {
                "case_id": float(case_id),
                "sample": float(sample),
                "t": float(time_value),
                "closure": closure,
                "integrator": "symplectic_euler",
                "shape_amplitude_theory": float(theory_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "shape_amplitude_abs_error": float(abs(fitted_amplitude - theory_amplitude)),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "reference_pressure_pa": float(reference_pressure),
                "closure_pressure_pa": float(stats["closure_pressure_pa"]),
                "fheron_pressure_pa": float(stats["fheron_pressure_pa"]),
                "vol_rel": float(stats["vol_rel"]),
                "local_volume_rel_max_abs": float(stats["local_volume_rel_max_abs"]),
                "nodal_density_rel_min": float(stats["nodal_density_rel_min"]),
                "nodal_density_rel_max": float(stats["nodal_density_rel_max"]),
                "total_mass_kg": float(stats["total_mass_kg"]),
                "mass_flux_l1_kg_s": float(stats["mass_flux_l1_kg_s"]),
                "momentum_flux_l1_n": float(stats["momentum_flux_l1_n"]),
                "flux_limiter_scale": float(stats["flux_limiter_scale"]),
                "mass_flux_coupling": float(mass_flux_coupling),
                "momentum_flux_coupling": float(momentum_flux_coupling),
                "face_flux_lagrangian": 1.0 if face_flux_mode == "lagrangian" else 0.0,
                "density_change_limit": float(density_change_limit) if density_change_limit is not None else 0.0,
                "inertia_scale": float(inertia_scale),
                "active_retopology": 1.0 if active_retopology else 0.0,
                "solver_tet_count": float(state.tets.shape[0]),
                "display_tet_count": float(stats["display_tet_count"]),
                "display_changed_tets": float(stats["display_changed_tets"]),
                "display_changed_fraction": float(stats["display_changed_fraction"]),
                "cpu_time_s": float(cpu_elapsed),
                "wall_time_s": float(wall_elapsed),
                "cpu_ms_per_substep": float(1.0e3 * cpu_elapsed / float(substeps_done)) if substeps_done else 0.0,
                "wall_ms_per_substep": float(1.0e3 * wall_elapsed / float(substeps_done)) if substeps_done else 0.0,
                "max_speed_m_s": float(np.max(np.linalg.norm(current_velocities, axis=1))),
            }
        )
        snapshots.append(current_points.copy())
        display_tets.append(np.asarray(state.tets, dtype=int).copy())

    state.prepare(HC)
    collect(0, 0.0)

    total_substeps = (int(n_steps) - 1) * int(substeps_per_sample)

    def callback(step, t, HC_obj, bV_obj, _diagnostics) -> None:
        state.remove_rigid_translation(HC_obj, bV_obj)
        if (int(step) + 1) % int(substeps_per_sample) == 0:
            sample = (int(step) + 1) // int(substeps_per_sample)
            collect(sample, float(t))

    symplectic_euler(
        HC,
        bV,
        ddg_stress.volume_gradient_pressure_acceleration,
        dt,
        total_substeps,
        dim=3,
        callback=callback,
        retopologize_fn=state.retopologize_callback,
        state=state,
        mu=0.0,
    )
    return rows, np.asarray(snapshots, dtype=float), surface_faces, surface_indices, np.asarray(display_tets, dtype=object)


def run_twophase_case(
    *,
    case_id: int,
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
    rayleigh_mode: int,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_points, tets, phases, interface_indices, outer_indices, surface_faces, _unit, liquid_vertex_indices = (
        twophase.build_liquid_gas_mesh(subdivisions, radius, outer_radius_scale)
    )
    points = twophase.deform_initial_interface(
        reference_points,
        tets,
        phases,
        liquid_vertex_indices,
        radius,
        amplitude,
    )
    liquid_tets = np.asarray(tets[phases == LIQUID], dtype=int)
    gas_tets = np.asarray(tets[phases == GAS], dtype=int)
    gas_tets_current = gas_tets.copy()
    reference_liquid_volumes, _matrix0 = ddg_stress.tet_volume_matrix(points, liquid_tets)
    reference_gas_volumes = ddg_stress.tet_cell_volumes(points, gas_tets)
    tet_masses = float(rho_liquid) * reference_liquid_volumes
    gas_tet_masses = float(rho_gas) * reference_gas_volumes
    nodal_masses = ddg_stress.tet_volume_lumped_nodal_values(points.shape[0], liquid_tets, tet_masses)
    velocities = np.zeros_like(points)
    vertex_phases = np.full(points.shape[0], GAS, dtype=int)
    vertex_phases[np.asarray(liquid_vertex_indices, dtype=int)] = LIQUID

    inertial_mass_addition = np.zeros(points.shape[0], dtype=float)

    _forces0, _areas0, reference_pressure, _volume0 = ddg_stress.heron_forces_for_points(
        reference_points[interface_indices],
        surface_faces,
        gamma,
    )
    dt = float(t_final) / float((int(n_steps) - 1) * int(substeps_per_sample))

    HC, _vertices = build_vertex_complex(points, velocities, np.maximum(nodal_masses, 1.0e-18), vertex_phases)
    bV: set = set()
    state = ddg_multiphase.MultiphaseVolumeGradientPressureState(
        tets=liquid_tets,
        surface_indices=interface_indices,
        surface_faces=surface_faces,
        gamma=gamma,
        density=rho_liquid,
        bulk_modulus=bulk_liquid,
        closure=closure,
        reference_tet_volumes=reference_liquid_volumes,
        tet_masses=tet_masses,
        base_tet_pressures=reference_pressure,
        dt=dt,
        inertia_scale=inertia_scale,
        inertial_mass_addition=inertial_mass_addition,
        damping_rate=0.0,
        fixed_indices=outer_indices,
        phase_ids=np.full(liquid_tets.shape[0], LIQUID, dtype=int),
        phase_densities={LIQUID: float(rho_liquid), GAS: float(rho_gas)},
        phase_bulk_moduli={LIQUID: float(bulk_liquid), GAS: float(bulk_gas)},
        phase_base_pressures={LIQUID: float(reference_pressure), GAS: 0.0},
        active_retopology=False,
        mass_flux_coupling=1.0,
        momentum_flux_coupling=1.0,
        density_change_limit=None,
        face_flux_mode="lagrangian",
        sound_speed=math.sqrt(float(bulk_liquid) / float(rho_liquid)),
        incompressible_target_mode="reference",
    )
    state.HC = HC

    rows: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    display_tets: list[np.ndarray] = []
    phases_snapshots: list[np.ndarray] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    reference_liquid_volume = float(np.sum(reference_liquid_volumes))
    reference_gas_volume = float(np.sum(reference_gas_volumes))
    liquid_vertex_indices = np.asarray(liquid_vertex_indices, dtype=int)
    reference_combined_tet_set = {tuple(sorted(int(v) for v in tet)) for tet in np.asarray(tets, dtype=int)}
    state.reference_display_tet_set = set(reference_combined_tet_set)

    def twophase_retopology_callback(HC_obj, _bV, _dim, **_kwargs) -> None:
        """Rebuild the two-phase Delaunay tet list and refresh liquid pressure.

        The full liquid+gas vertex cloud is retriangulated.  New tetrahedra are
        phase-labelled by the persistent Lagrangian vertex phase: a tet is
        liquid only when all four vertices belong to the liquid vertex set;
        every tet touching a gas/outer vertex is gas.  The pressure-reference
        operator is then rebuilt only on the liquid tets, with tet masses and
        target volumes remapped from the new tet volumes.  This is the
        retopology workaround for the HC-dual-volume density/pressure jump
        observed in the original oscillating-droplet case.
        """
        nonlocal gas_tets_current
        current_points, _current_velocities = state.read_hc_state(HC_obj)
        combined_old_tets = np.vstack([np.asarray(state.tets, dtype=int), np.asarray(gas_tets_current, dtype=int)])
        new_all_tets = ddg_stress.delaunay_retriangulate_points(current_points, combined_old_tets)
        new_all_tets = np.asarray(new_all_tets, dtype=int)
        new_tet_phases = np.where(
            np.all(vertex_phases[new_all_tets] == LIQUID, axis=1),
            LIQUID,
            GAS,
        )
        new_liquid_tets = np.asarray(new_all_tets[new_tet_phases == LIQUID], dtype=int)
        new_gas_tets = np.asarray(new_all_tets[new_tet_phases == GAS], dtype=int)
        if new_liquid_tets.size == 0 or new_gas_tets.size == 0:
            new_liquid_tets = np.asarray(state.tets, dtype=int)
            new_gas_tets = np.asarray(gas_tets_current, dtype=int)
        new_liquid_volumes = ddg_stress.tet_cell_volumes(current_points, new_liquid_tets)
        current_liquid_volume = float(np.sum(new_liquid_volumes))
        if current_liquid_volume <= 1.0e-300:
            new_liquid_tets = np.asarray(state.tets, dtype=int)
            new_liquid_volumes = ddg_stress.tet_cell_volumes(current_points, new_liquid_tets)
            current_liquid_volume = max(float(np.sum(new_liquid_volumes)), 1.0e-300)
        # The pressure target is remapped from the new liquid tet volumes.
        # Without this step, a Delaunay connectivity flip changes local volume
        # bookkeeping and the EOS/projection interprets that as a physical
        # density jump.
        mass_density = float(state.reference_total_mass) / current_liquid_volume
        volume_scale = float(state.reference_total_volume) / current_liquid_volume
        state.tets = np.asarray(new_liquid_tets, dtype=int)
        state.reference_tet_volumes = volume_scale * np.asarray(new_liquid_volumes, dtype=float)
        state.tet_masses = mass_density * np.asarray(new_liquid_volumes, dtype=float)
        state.reference_tet_masses = state.tet_masses.copy()
        state.phase_ids = np.full(state.tets.shape[0], LIQUID, dtype=int)
        state.face_pairs = ddg_stress.tet_face_pairs(state.tets)
        gas_tets_current = np.asarray(new_gas_tets, dtype=int)
        current_combined = np.vstack([state.tets, gas_tets_current])
        current_set = {tuple(sorted(int(v) for v in tet)) for tet in current_combined}
        changed = reference_combined_tet_set ^ current_set
        state.last_retopology_stats = {
            "retriangulation_active": 1.0,
            "display_tet_count": float(current_combined.shape[0]),
            "display_internal_face_count": float(len(ddg_stress.tet_face_pairs(current_combined))),
            "display_changed_tets": float(len(changed)),
            "display_changed_fraction": float(len(changed)) / max(float(len(reference_combined_tet_set | current_set)), 1.0),
        }
        state.mark_dirty()

    def combined_display() -> tuple[np.ndarray, np.ndarray]:
        combined_tets = np.vstack([np.asarray(state.tets, dtype=int), np.asarray(gas_tets_current, dtype=int)])
        combined_phases = np.concatenate(
            [
                np.full(state.tets.shape[0], LIQUID, dtype=int),
                np.full(gas_tets_current.shape[0], GAS, dtype=int),
            ]
        )
        return combined_tets, combined_phases

    def collect(sample: int, time_value: float) -> None:
        current_points, current_velocities = state.read_hc_state(HC)
        stats = state.diagnostics(HC)
        combined_tets, current_phases = combined_display()
        liquid_volumes, _matrix = ddg_stress.tet_volume_matrix(current_points, state.tets)
        gas_volumes = ddg_stress.tet_cell_volumes(current_points, gas_tets_current)
        interface_points = current_points[interface_indices]
        fitted_amplitude = base.fit_l2_shape_amplitude(interface_points, surface_faces)
        theory_amplitude = float(amplitude) * math.cos(float(rayleigh_omega_rad_s) * float(time_value))
        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start
        substeps_done = int(sample) * int(substeps_per_sample)
        liquid_mask = current_phases == LIQUID
        gas_mask = current_phases == GAS
        liquid_vol = float(np.sum(liquid_volumes))
        gas_vol = float(np.sum(gas_volumes))
        rows.append(
            {
                "case_id": float(case_id),
                "sample": float(sample),
                "t": float(time_value),
                "closure": closure,
                "integrator": "symplectic_euler",
                "mesh_phases": 2.0,
                "n_liquid_tets": float(np.sum(liquid_mask)),
                "n_gas_tets": float(np.sum(gas_mask)),
                "n_interface_vertices": float(interface_indices.shape[0]),
                "n_outer_gas_vertices": float(outer_indices.shape[0]),
                "shape_amplitude_theory": float(theory_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "shape_amplitude_abs_error": float(abs(fitted_amplitude - theory_amplitude)),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "reference_pressure_pa": float(reference_pressure),
                "closure_pressure_pa": float(stats["closure_pressure_pa"]),
                "fheron_pressure_pa": float(stats["fheron_pressure_pa"]),
                "liquid_vol_rel": float(liquid_vol / max(reference_liquid_volume, 1.0e-300)),
                "gas_vol_rel": float(gas_vol / max(reference_gas_volume, 1.0e-300)),
                "local_volume_rel_max_abs": float(stats["local_volume_rel_max_abs"]),
                "nodal_density_rel_min": float(stats["nodal_density_rel_min"]),
                "nodal_density_rel_max": float(stats["nodal_density_rel_max"]),
                "total_mass_kg": float(stats["total_mass_kg"] + np.sum(gas_tet_masses)),
                "mass_flux_l1_kg_s": float(stats["mass_flux_l1_kg_s"]),
                "momentum_flux_l1_n": float(stats["momentum_flux_l1_n"]),
                "flux_limiter_scale": float(stats["flux_limiter_scale"]),
                "face_flux_lagrangian": 1.0,
                "inertia_scale": float(inertia_scale),
                "active_retopology": 1.0,
                "solver_tet_count": float(state.tets.shape[0]),
                "display_tet_count": float(combined_tets.shape[0]),
                "display_changed_tets": float(state.last_retopology_stats["display_changed_tets"]),
                "display_changed_fraction": float(state.last_retopology_stats["display_changed_fraction"]),
                "cpu_time_s": float(cpu_elapsed),
                "wall_time_s": float(wall_elapsed),
                "cpu_ms_per_substep": float(1.0e3 * cpu_elapsed / float(substeps_done)) if substeps_done else 0.0,
                "wall_ms_per_substep": float(1.0e3 * wall_elapsed / float(substeps_done)) if substeps_done else 0.0,
                "max_speed_m_s": float(np.max(np.linalg.norm(current_velocities, axis=1))),
            }
        )
        snapshots.append(current_points.copy())
        display_tets.append(np.asarray(combined_tets, dtype=int).copy())
        phases_snapshots.append(np.asarray(current_phases, dtype=int).copy())

    state.prepare(HC)
    collect(0, 0.0)
    total_substeps = (int(n_steps) - 1) * int(substeps_per_sample)

    def callback(step, t, HC_obj, bV_obj, _diagnostics) -> None:
        state.mark_dirty()
        if (int(step) + 1) % int(substeps_per_sample) == 0:
            sample = (int(step) + 1) // int(substeps_per_sample)
            collect(sample, float(t))

    symplectic_euler(
        HC,
        bV,
        ddg_multiphase.multiphase_volume_gradient_pressure_acceleration,
        dt,
        total_substeps,
        dim=3,
        callback=callback,
        retopologize_fn=twophase_retopology_callback,
        state=state,
        mu=0.0,
    )
    return (
        rows,
        np.asarray(snapshots, dtype=float),
        surface_faces,
        interface_indices,
        outer_indices,
        np.asarray(display_tets, dtype=object),
        np.asarray(phases_snapshots, dtype=object),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate volume-gradient pressure through ddgclib dynamic integrators.")
    parser.add_argument("--case-id", type=int, choices=(5, 6, 11, 12), required=True)
    parser.add_argument("--radius", type=float, default=1.0e-3)
    parser.add_argument("--outer-radius-scale", type=float, default=1.60)
    parser.add_argument("--gamma", type=float, default=0.072)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--rho-gas", type=float, default=1.2)
    parser.add_argument("--bulk-modulus", type=float, default=2.2e9)
    parser.add_argument("--bulk-gas", type=float, default=1.42e5)
    parser.add_argument("--rayleigh-mode", type=int, default=2)
    parser.add_argument("--shape-mode-ar", type=float, default=1.05)
    parser.add_argument("--t-final", type=float, default=0.016)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--steps", type=int, default=81)
    parser.add_argument("--substeps-per-sample", type=int, default=None)
    parser.add_argument("--damping-ratio", type=float, default=0.0)
    parser.add_argument("--inertia-scale", type=float, default=1.08)
    parser.add_argument("--mass-flux-coupling", type=float, default=1.0)
    parser.add_argument("--momentum-flux-coupling", type=float, default=1.0)
    parser.add_argument("--density-change-limit", type=float, default=0.0)
    parser.add_argument("--face-flux-mode", choices=("lagrangian", "cell-average"), default="lagrangian")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "out" / "dynamic_integrator_p_ref")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    closure = "compressible" if int(args.case_id) in (5, 11) else "incompressible"
    active_retopology = int(args.case_id) in (11, 12)
    default_substeps = 5 if active_retopology else 20
    substeps_per_sample = int(args.substeps_per_sample or default_substeps)
    omega, frequency, _period = base.rayleigh_lamb_frequency(args.rayleigh_mode, args.gamma, args.rho, args.radius)
    amplitude = base.shape_amplitude_from_aspect_ratio(args.shape_mode_ar)
    density_limit = None if args.density_change_limit is None or float(args.density_change_limit) <= 0.0 else float(args.density_change_limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if int(args.case_id) in (11, 12):
            rows, snapshots, faces, surface_indices, outer_indices, display_tets, phases_snapshots = run_twophase_case(
                case_id=args.case_id,
                closure=closure,
                subdivisions=args.subdivision,
                radius=args.radius,
                outer_radius_scale=args.outer_radius_scale,
                gamma=args.gamma,
                rho_liquid=args.rho,
                rho_gas=args.rho_gas,
                bulk_liquid=args.bulk_modulus,
                bulk_gas=args.bulk_gas,
                amplitude=amplitude,
                n_steps=args.steps,
                t_final=args.t_final,
                substeps_per_sample=substeps_per_sample,
                inertia_scale=args.inertia_scale,
                rayleigh_mode=args.rayleigh_mode,
                rayleigh_frequency_hz=frequency,
                rayleigh_omega_rad_s=omega,
            )
            stem = f"case{int(args.case_id)}_dynamic_integrator_twophase_{closure}"
        else:
            rows, snapshots, faces, surface_indices, display_tets = run_case(
                case_id=args.case_id,
                closure=closure,
                subdivisions=args.subdivision,
                radius=args.radius,
                gamma=args.gamma,
                rho=args.rho,
                bulk_modulus=args.bulk_modulus,
                amplitude=amplitude,
                n_steps=args.steps,
                t_final=args.t_final,
                substeps_per_sample=substeps_per_sample,
                damping_ratio=args.damping_ratio,
                inertia_scale=args.inertia_scale,
                mass_flux_coupling=args.mass_flux_coupling,
                momentum_flux_coupling=args.momentum_flux_coupling,
                density_change_limit=density_limit,
                face_flux_mode=args.face_flux_mode,
                active_retopology=active_retopology,
                rayleigh_mode=args.rayleigh_mode,
                rayleigh_frequency_hz=frequency,
                rayleigh_omega_rad_s=omega,
            )
            outer_indices = np.asarray([], dtype=int)
            phases_snapshots = np.asarray([], dtype=object)
            stem = f"case{int(args.case_id)}_dynamic_integrator_{closure}"
    csv_path, json_path = write_rows(rows, args.out_dir, stem)
    npz_path = args.out_dir / f"{stem}_snapshots.npz"
    np.savez_compressed(
        npz_path,
        points=snapshots,
        faces=faces,
        surface_indices=surface_indices,
        outer_indices=outer_indices,
        display_tets=display_tets,
        phases=phases_snapshots,
    )
    final = rows[-1]
    print(
        f"#{int(args.case_id)} {closure}: "
        f"a2={final['shape_amplitude_fit']:.8f}, "
        f"theory={final['shape_amplitude_theory']:.8f}, "
        f"|err|={final['shape_amplitude_abs_error']:.3e}, "
        f"local={final['local_volume_rel_max_abs']:.3e}, "
        f"changed_tets={final['display_changed_tets']:.0f}, "
        f"CPU={final['cpu_time_s']:.2f}s"
    )
    print(f"rows: {json_path}")
    print(f"snapshots: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
