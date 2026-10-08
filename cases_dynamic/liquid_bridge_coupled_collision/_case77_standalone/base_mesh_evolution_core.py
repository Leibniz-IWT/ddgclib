#!/usr/bin/env python3
"""Case 9: real ddgclib/PR free-surface mesh evolution for Siekman 2025.

This case supersedes the earlier Case 9 mesh-diagnostic run.  It does not use
Case 8's reduced trajectory as the simulated result.  Instead it evolves a real
triangular 3D free-surface mesh with ddgclib Heron surface-tension forces.

Truth label
-----------
This is a real ddgclib/PR mesh evolution, but still not a full 3D
Navier-Stokes solver.  The update is an overdamped free-surface mesh
evolution with an axisymmetry projection and a generic bridge suction closure
driven by the mesh's own missing-film volume.  If it does not match the
experiment, the mismatch is a real ddgclib-development signal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.special import i0

from . import bridge_film_core as case8


ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 9"
OUTPUT_PREFIX = "case9"
OUT_DIR = ROOT / CASE_STEM


@dataclass(frozen=True)
class RealMeshEvolutionConfig:
    sphere_radius_mm: float = 5.0
    sphere_bottom_z_mm: float | None = None
    substrate_radius_mm: float = 12.0
    initial_film_thickness_um: float = 100.0
    surface_tension_n_m: float = 0.021
    viscosity_pa_s: float = 0.10
    density_kg_m3: float = 1065.0
    gravity_m_s2: float = 9.80665
    dt_s: float = 0.01
    max_steps: int = 6000
    wall_clock_limit_s: float = 3600.0
    record_every_steps: int = 250
    profile_nodes: int = 72
    azimuthal_nodes: int = 36
    inner_bridge_volume_ul: float = 1.75
    initial_bridge_radius_mm: float = 2.2087
    pinned_bridge_radius_mm: float = 2.10
    dynamic_inner_plateau_enabled: bool = False
    dynamic_inner_plateau_start_s: float = 150.0
    dynamic_inner_plateau_tau_s: float = 850.0
    dynamic_inner_plateau_exponent: float = 1.0
    dynamic_inner_plateau_gap_to_pressure_center_mm: float = 0.42
    dynamic_inner_plateau_max_radius_mm: float = 3.62
    bridge_pressure_width_mm: float = 0.07
    bridge_pressure_multiplier: float = 3.0
    pressure_reference_film_thickness_um: float = 100.0
    film_thickness_pressure_exponent: float = 0.0
    bridge_pressure_startup_time_s: float = 15.0
    bridge_pressure_startup_exponent: float = 1.0
    bridge_pressure_relax_time_s: float = 1500.0
    bridge_pressure_relax_exponent: float = 1.0
    late_pressure_boost_enabled: bool = False
    late_pressure_boost_start_s: float = 140.0
    late_pressure_boost_tau_s: float = 850.0
    late_pressure_boost_exponent: float = 1.0
    late_pressure_boost_factor: float = 0.0
    terminal_pressure_boost_enabled: bool = False
    terminal_pressure_boost_start_s: float = 2500.0
    terminal_pressure_boost_tau_s: float = 450.0
    terminal_pressure_boost_exponent: float = 1.0
    terminal_pressure_boost_factor: float = 0.0
    bridge_pressure_center_offset_mm: float = 0.95
    bridge_pressure_center_transient_inset_mm: float = 0.0
    bridge_pressure_center_transient_growth_time_s: float = 30.0
    bridge_pressure_center_transient_relax_time_s: float = 1000.0
    bridge_pressure_center_transient_exponent: float = 1.0
    bridge_feed_limiter_enabled: bool = True
    bridge_feed_capacity_multiplier: float = 1.25
    bridge_feed_time_scale_s: float = 520.0
    bridge_feed_time_exponent: float = 0.35
    attached_state_closure_enabled: bool = False
    attached_closure_rim_base_contact_radius_fraction: float = 1.05
    attached_closure_rim_transient_contact_radius_fraction: float = 0.94
    attached_closure_rim_decay_progress: float = 0.35
    attached_closure_rim_decay_exponent: float = 1.0
    attached_closure_feed_start_annulus_fraction: float = 0.26
    attached_closure_feed_capacity_annulus_fraction: float = 0.90
    attached_closure_feed_growth_progress: float = 0.55
    attached_closure_feed_growth_exponent: float = 0.80
    attached_closure_feed_max_annulus_fraction: float = 0.93
    attached_closure_pressure_start_annulus_fraction: float = 0.25
    attached_closure_pressure_growth_progress: float = 0.40
    attached_closure_pressure_growth_exponent: float = 1.0
    attached_closure_pressure_boost_factor: float = 1.35
    attached_closure_pressure_front_offset_factor: float = 160.0
    attached_closure_pressure_front_offset_h0_lc_exponent: float = 3.0
    attached_closure_pressure_front_width_factor: float = 18.0
    attached_closure_pressure_front_width_h0_lc_exponent: float = 2.0
    attached_closure_limit_missing_volume: bool = True
    attached_reduced_bridge_film_driver_enabled: bool = False
    attached_reduced_driver_bottleneck_enabled: bool = True
    attached_reduced_driver_bottleneck_volume_ul: float = 0.020
    attached_reduced_driver_bottleneck_exponent: float = 2.0
    attached_reduced_driver_bottleneck_floor: float = 0.0
    attached_reduced_driver_pressure_activation_time_s: float = 25.0
    attached_reduced_driver_pressure_relax_time_s: float = 120.0
    attached_reduced_driver_pressure_relax_exponent: float = 0.40
    attached_reduced_driver_grid_nodes: int = 140
    attached_reduced_driver_inner_pressure_boundary_enabled: bool = False
    attached_reduced_driver_profile_projection_enabled: bool = False
    attached_reduced_driver_profile_projection_relaxation: float = 1.0
    attached_local_capture_enabled: bool = False
    attached_local_capture_width_capillary_lengths: float = 0.0
    attached_local_capture_time_s: float = 10.0
    attached_local_capture_exponent: float = 1.0
    attached_local_capture_release_time_s: float = 4000.0
    attached_local_capture_release_exponent: float = 1.0
    attached_local_capture_release_saturation_enabled: bool = False
    attached_local_capture_release_delay_s: float = 0.0
    attached_local_capture_release_extra_width_capillary_lengths: float = 0.0
    attached_volume_projection_min_height_fraction: float = 0.0
    attached_volume_projection_width_mm: float = 6.0
    attached_volume_projection_width_capillary_lengths: float = 0.0
    attached_volume_projection_min_width_mm: float = 0.0
    attached_volume_projection_scale_width_with_visible_fraction: bool = False
    attached_volume_projection_weight_exponent: float = 0.7
    attached_volume_projection_time_release_enabled: bool = False
    attached_volume_projection_time_release_start_s: float = 0.0
    attached_volume_projection_time_release_tau_s: float = 1.0
    attached_volume_projection_time_release_exponent: float = 1.0
    attached_volume_projection_time_release_floor_fraction: float = 0.0
    attached_volume_projection_time_release_ceiling_fraction: float = 1.0
    attached_volume_projection_recovery_front_enabled: bool = False
    attached_volume_projection_recovery_front_start_s: float = 0.0
    attached_volume_projection_recovery_front_exponent: float = 1.2
    attached_volume_projection_recovery_front_min_width_um: float = 25.0
    attached_volume_projection_recovery_front_max_width_mm: float = 2.5
    attached_volume_projection_recovery_front_late_start_s: float = 0.0
    attached_volume_projection_recovery_front_late_tau_s: float = 1.0
    attached_volume_projection_recovery_front_late_ramp_exponent: float = 1.0
    attached_volume_projection_recovery_front_late_max_width_mm: float = 0.0
    attached_volume_projection_recovery_front_late_shape_exponent: float = 0.0
    attached_volume_projection_recovery_front_relaxation: float = 1.0
    attached_volume_projection_bridge_reservoir_enabled: bool = False
    attached_volume_projection_bridge_reservoir_counts_as_missing: bool = True
    attached_volume_projection_bridge_reservoir_weight: float = 1.0
    attached_volume_projection_bridge_reservoir_inner_margin_mm: float = 0.10
    attached_volume_projection_bridge_reservoir_outer_gap_mm: float = 0.55
    attached_bridge_growth_irreversible_enabled: bool = False
    attached_bridge_growth_irreversible_tolerance_ul: float = 1.0e-6
    attached_visible_feed_partition_enabled: bool = False
    attached_visible_feed_min_fraction: float = 1.0
    attached_visible_feed_max_fraction: float = 1.0
    attached_visible_feed_transition_progress: float = 0.50
    attached_visible_feed_transition_width: float = 0.08
    attached_visible_feed_scale_radius_mode: str = "rim"
    attached_visible_feed_contact_blend_exponent: float = 1.0
    attached_visible_feed_profile_constraint_enabled: bool = False
    attached_visible_feed_profile_base_width_capillary_lengths: float = 0.12
    attached_visible_feed_profile_fraction_width_capillary_lengths: float = 0.58
    attached_visible_feed_profile_decay_exponent: float = 1.0
    attached_visible_feed_profile_late_volume_boost_factor: float = 0.0
    attached_visible_feed_profile_late_boost_progress: float = 0.50
    attached_visible_feed_profile_late_boost_width: float = 0.05
    attached_visible_feed_subtract_wetted_sphere_cap: bool = False
    attached_visible_feed_cap_fraction_uses_corrected_volume: bool = False
    attached_neck_boundary_layer_enabled: bool = False
    attached_neck_boundary_layer_solver: str = "cox"
    attached_neck_boundary_layer_min_width_um: float = 2.0
    attached_neck_boundary_layer_max_width_capillary_lengths: float = 0.40
    attached_neck_boundary_layer_relaxation: float = 1.0
    attached_neck_boundary_layer_update_mode: str = "projection"
    attached_neck_residual_time_s: float = 0.50
    attached_neck_residual_pressure_weight: float = 0.25
    attached_neck_residual_flux_weight: float = 0.10
    attached_neck_residual_volume_lagrange_enabled: bool = False
    attached_neck_residual_volume_lagrange_weight: float = 0.35
    attached_neck_residual_max_vertical_speed_um_s: float = 220.0
    attached_neck_residual_max_radial_speed_um_s: float = 120.0
    attached_neck_boundary_layer_solve_rim_height: bool = False
    attached_neck_boundary_layer_bvp_nodes: int = 80
    attached_neck_visible_feed_late_multiplier: float = 1.0
    attached_neck_visible_feed_late_start_s: float = 0.0
    attached_neck_visible_feed_late_tau_s: float = 1.0
    attached_neck_visible_feed_late_exponent: float = 1.0
    attached_neck_adaptive_rings_enabled: bool = False
    attached_neck_adaptive_rings_fraction: float = 0.42
    attached_neck_adaptive_width_multiplier: float = 5.0
    attached_neck_adaptive_min_window_um: float = 180.0
    attached_neck_adaptive_max_window_mm: float = 1.20
    attached_neck_adaptive_spacing_exponent: float = 1.70
    attached_neck_floor_enabled: bool = False
    attached_neck_floor_initial_um: float = 0.0
    attached_neck_floor_late_um: float = 0.0
    attached_neck_floor_start_s: float = 0.0
    attached_neck_floor_tau_s: float = 1.0
    attached_neck_floor_exponent: float = 1.0
    attached_compact_neck_profile_enabled: bool = False
    attached_compact_neck_activation_time_s: float = 0.0
    attached_compact_neck_activation_ramp_s: float = 1.0
    attached_compact_neck_bridge_branch_enabled: bool = True
    attached_compact_neck_visible_volume_multiplier: float = 1.0
    attached_compact_neck_recovery_power: float = 2.0
    attached_compact_neck_min_width_inner_scale: float = 0.18
    attached_compact_neck_max_width_capillary_lengths: float = 1.20
    attached_compact_neck_left_width_inner_scale: float = 0.80
    attached_compact_neck_left_min_width_inner_scale: float = 0.14
    attached_compact_neck_left_decay_progress: float = 0.34
    attached_compact_neck_left_time_decay_s: float = 0.0
    attached_compact_neck_left_time_decay_exponent: float = 1.0
    attached_compact_neck_rim_height_relaxation: float = 1.0
    attached_compact_neck_bridge_profile: str = "power"
    attached_compact_neck_bridge_shape_exponent: float = 2.0
    attached_compact_neck_bridge_contact_slope_factor: float = 1.0
    attached_outer_deficit_spreading_enabled: bool = False
    attached_outer_deficit_spreading_start_s: float = 0.0
    attached_outer_deficit_spreading_ramp_s: float = 1.0
    attached_outer_deficit_spreading_width_capillary_lengths: float = 1.0
    attached_outer_deficit_spreading_inner_guard_mm: float = 0.0
    attached_outer_deficit_spreading_passes: int = 8
    attached_outer_deficit_spreading_blend: float = 0.35
    attached_outer_deficit_spreading_volume_multiplier: float = 1.0
    attached_outer_deficit_spreading_monotone_recovery: bool = False
    attached_outer_film_recovery_max_cell_dz_um: float = 0.0
    attached_outer_film_anti_sawtooth_enabled: bool = False
    attached_outer_film_anti_sawtooth_start_s: float = 0.0
    attached_outer_film_anti_sawtooth_inner_margin_mm: float = 0.25
    attached_outer_film_anti_sawtooth_outer_margin_mm: float = 0.50
    attached_outer_film_anti_sawtooth_passes: int = 4
    attached_outer_film_anti_sawtooth_alpha: float = 0.25
    attached_outer_film_anti_sawtooth_preserve_volume: bool = True
    attached_outer_deficit_soft_lower_enabled: bool = False
    attached_outer_deficit_soft_lower_um: float = 0.25
    attached_outer_deficit_soft_lower_slope_um_per_mm: float = 0.0
    attached_outer_deficit_soft_lower_width_mm: float = 1.0
    attached_outer_deficit_soft_lower_repulsion: float = 0.0
    volume_projection_enabled: bool = True
    volume_projection_start_widths: float = 2.5
    refined_bridge_mesh_enabled: bool = True
    refined_bridge_mesh_inner_mm: float = 1.8
    refined_bridge_mesh_outer_mm: float = 5.6
    refined_bridge_mesh_fraction: float = 0.62
    vertical_mobility_m_per_s_pa: float = 1.4e-6
    radial_mobility_factor: float = 0.20
    mobility_reference_film_thickness_um: float = 100.0
    film_thickness_mobility_exponent: float = 0.0
    subtract_initial_equilibrium_force: bool = True
    lubrication_mobility_exponent: float = 4.0
    lubrication_mobility_floor: float = 0.00005
    velocity_smoothing: float = 0.75
    max_vertical_speed_um_s: float = 45.0
    max_radial_speed_um_s: float = 12.0
    use_cox_contact_line_force: bool = False
    enable_dynamic_contact_angle: bool = True
    contact_angle_deg: float = 0.0
    dynamic_contact_angle_max_deg: float = 25.0
    dynamic_contact_angle_min_deg: float = 0.0
    contact_line_cox_macro_length_m: float = 1.0e-3
    contact_line_cox_slip_length_m: float = 2.0e-9
    cox_contact_line_force_scale: float = 1.0
    cox_contact_line_activation_time_s: float = 1.0e-12
    cox_contact_line_activation_exponent: float = 1.0
    cox_contact_line_direction_z: float = -1.0
    cox_contact_ring_offset_mm: float = 0.0
    full_bridge_mesh_enabled: bool = False
    full_bridge_mesh_bridge_nodes: int = 32
    evolve_attached_bridge_mesh: bool = False
    attached_initial_contact_radius_mm: float | None = None
    attached_initial_rim_radius_mm: float | None = None
    attached_contact_angle_probe_rings: int = 1
    attached_contact_angle_probe_window_um: float = 0.0
    attached_contact_angle_probe_min_fit_rings: int = 3
    attached_contact_angle_probe_max_fit_rings: int = 12
    attached_contact_line_max_speed_um_s: float = 4.0
    attached_contact_line_bridge_radius_coupling_enabled: bool = False
    attached_contact_line_target_uses_visible_feed_partition: bool = False
    attached_contact_line_bridge_radius_growth_multiplier: float = 1.0
    attached_contact_line_bridge_radius_pull_only_below_target: bool = False
    attached_contact_line_bridge_radius_cap_gap_fraction: float = 0.0
    attached_contact_line_bridge_radius_beyond_target_max_speed_um_s: float = 0.0
    attached_contact_line_bridge_radius_relaxation: float = 0.18
    attached_contact_line_bridge_radius_max_speed_um_s: float = 25.0
    attached_contact_line_allow_recede: bool = True
    attached_bridge_relaxation: float = 0.18
    attached_bridge_rim_offset_model: str = "constant"
    attached_bridge_rim_offset_mm: float = 0.0
    attached_bridge_rim_capillary_length_multiplier: float = 1.0
    attached_bridge_rim_transient_offset_mm: float = 0.0
    attached_bridge_rim_transient_decay_time_s: float = 1.0
    attached_bridge_rim_transient_decay_exponent: float = 1.0
    attached_bridge_shape_constraint_enabled: bool = False
    attached_bridge_shape_neck_fraction: float = 0.10
    attached_bridge_shape_height_exponent: float = 1.0
    attached_bridge_rim_relaxation_per_step: float = 0.0
    attached_bridge_rim_shift_decay_mm: float = 1.5
    attached_capillary_bridge_solver_enabled: bool = False
    attached_capillary_bridge_precursor_fraction: float = 0.18
    attached_capillary_bridge_min_height_um: float = 0.0
    attached_capillary_bridge_nodes: int = 36
    attached_capillary_bridge_enforce_contact_angle: bool = False
    attached_capillary_bridge_match_film_slope: bool = False
    attached_capillary_bridge_profile_smoothing_passes: int = 0
    attached_capillary_bridge_profile_smoothing_alpha: float = 0.0
    attached_capillary_bridge_profile_max_cell_dz_um: float = 0.0
    attached_capillary_bridge_profile_volume_correction: bool = True
    attached_capillary_bridge_profile_remove_local_maxima: bool = False
    attached_capillary_bridge_lower_bound_repulsion: float = 0.0
    attached_capillary_bridge_lower_bound_repulsion_length_um: float = 5.0
    attached_capillary_bridge_slope_regularization: float = 0.0
    attached_capillary_bridge_curvature_regularization: float = 0.0
    attached_capillary_bridge_profile_model: str = "area_min"
    attached_capillary_bridge_positive_volume_floor: bool = False
    attached_capillary_bridge_use_dynamic_min_height: bool = False
    attached_bridge_capillary_shoulder_enabled: bool = False
    attached_bridge_capillary_shoulder_profile: str = "anchored_meniscus"
    attached_bridge_capillary_shoulder_min_width_mm: float = 0.12
    attached_bridge_capillary_shoulder_max_width_mm: float = 0.35
    attached_bridge_capillary_shoulder_width_depth_exponent: float = 1.0
    attached_bridge_capillary_shoulder_drop_exponent: float = 4.0
    attached_bridge_capillary_shoulder_contact_width_fraction: float = 0.18
    attached_bridge_capillary_shoulder_relaxation: float = 1.0
    attached_film_radial_remesh_enabled: bool = False
    attached_film_radial_remesh_exponent: float = 1.15
    attached_film_height_regularization_per_step: float = 0.0
    attached_film_height_regularization_width_mm: float = 0.0
    attached_mesh_mobility_startup_time_s: float = 1.0
    attached_mesh_mobility_startup_exponent: float = 1.0
    attached_mesh_mobility_floor: float = 1.0
    attached_feed_limiter_start_time_s: float = 0.0
    attached_feed_limiter_start_missing_ul: float = 0.0
    attached_feed_limiter_capacity_multiplier: float = 1.0
    attached_feed_limiter_growth_time_s: float = 1.0
    attached_feed_limiter_growth_exponent: float = 1.0
    attached_feed_limiter_max_missing_ul: float = math.inf
    min_height_um: float = 0.25
    dynamic_min_height_enabled: bool = False
    dynamic_min_height_late_um: float = 0.25
    dynamic_min_height_start_s: float = 0.0
    dynamic_min_height_tau_s: float = 1.0
    dynamic_min_height_exponent: float = 1.0
    max_height_um: float = 120.0
    snapshot_times_s: tuple[float, ...] = (0.0, 10.0, 50.0, 75.0, 100.0, 3500.0)
    bridge_table_volume_ul: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 55.0)
    bridge_table_radius_mm: tuple[float, ...] = (1.00, 1.12, 1.23, 1.38, 1.65, 1.92, 2.28, 2.82, 3.35, 4.12, 5.02, 5.48)
    bridge_table_head_mm: tuple[float, ...] = (9.80, 8.00, 6.35, 4.90, 3.55, 2.75, 2.08, 1.45, 1.02, 0.68, 0.35, 0.22)
    bridge_table_head_scale: float = 1.0


CONFIG = RealMeshEvolutionConfig()


class Vertex:
    __slots__ = ("x_a", "u", "m", "p", "boundary", "nn", "ring", "theta_index")

    def __init__(self, coords: np.ndarray, ring: int, theta_index: int):
        self.x_a = np.asarray(coords, dtype=float)
        self.u = np.zeros(3, dtype=float)
        self.m = 1.0
        self.p = 0.0
        self.boundary = False
        self.nn: set[Vertex] = set()
        self.ring = int(ring)
        self.theta_index = int(theta_index)

    def connect(self, other: "Vertex") -> None:
        self.nn.add(other)
        other.nn.add(self)


def ensure_repo_path() -> None:
    repo_root = case8.resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    case8.ensure_hyperct_shim()


def load_operators() -> dict:
    ensure_repo_path()
    from cases_dynamic.oscillating_droplet_p_ref.scripts.pr33_operators import heron_forces_for_points
    from .operators.free_surface_mesh import (
        AxisymmetricFreeSurfaceMeshConfig,
        mesh_summary,
        revolve_attached_bridge_film_profile_to_mesh,
        revolve_profile_to_mesh,
        write_msh,
        write_npz,
        write_obj,
    )
    from .operators.contact_line import (
        cox_contact_line_force_ring,
        cox_inverse_contact_line_speed,
        relax_radius_toward_target,
    )
    from .operators.bridge_film import (
        AxisymmetricBridgeFilmConfig,
        attached_bridge_state_closure,
        axisymmetric_capillary_bridge_profile,
        bridge_delta_volume_with_local_capture_ul,
        bridge_volume_from_radius_m3,
        coupled_cox_young_laplace_lubrication_neck_profile,
        cox_arc_length_bridge_profile,
        cox_lubrication_neck_profile,
        linearized_young_laplace_bridge_profile,
        simulate_bridge_film,
        soft_repulsive_young_laplace_bridge_profile,
    )
    from .operators.surface_tension import dual_area_heron, surface_tension_force

    return {
        "AxisymmetricFreeSurfaceMeshConfig": AxisymmetricFreeSurfaceMeshConfig,
        "AxisymmetricBridgeFilmConfig": AxisymmetricBridgeFilmConfig,
        "attached_bridge_state_closure": attached_bridge_state_closure,
        "axisymmetric_capillary_bridge_profile": axisymmetric_capillary_bridge_profile,
        "bridge_delta_volume_with_local_capture_ul": bridge_delta_volume_with_local_capture_ul,
        "bridge_volume_from_radius_m3": bridge_volume_from_radius_m3,
        "coupled_cox_young_laplace_lubrication_neck_profile": coupled_cox_young_laplace_lubrication_neck_profile,
        "cox_arc_length_bridge_profile": cox_arc_length_bridge_profile,
        "cox_lubrication_neck_profile": cox_lubrication_neck_profile,
        "linearized_young_laplace_bridge_profile": linearized_young_laplace_bridge_profile,
        "simulate_bridge_film": simulate_bridge_film,
        "soft_repulsive_young_laplace_bridge_profile": soft_repulsive_young_laplace_bridge_profile,
        "cox_contact_line_force_ring": cox_contact_line_force_ring,
        "cox_inverse_contact_line_speed": cox_inverse_contact_line_speed,
        "relax_radius_toward_target": relax_radius_toward_target,
        "dual_area_heron": dual_area_heron,
        "heron_forces_for_points": heron_forces_for_points,
        "mesh_summary": mesh_summary,
        "revolve_attached_bridge_film_profile_to_mesh": revolve_attached_bridge_film_profile_to_mesh,
        "revolve_profile_to_mesh": revolve_profile_to_mesh,
        "surface_tension_force": surface_tension_force,
        "write_msh": write_msh,
        "write_npz": write_npz,
        "write_obj": write_obj,
    }


def initial_profile_m(config: RealMeshEvolutionConfig, r_m: np.ndarray) -> np.ndarray:
    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    length_m = capillary_length_m(config)
    i0_edge = float(i0(radius_m / length_m))
    h_m = h0_m * (i0_edge - i0(np.asarray(r_m, dtype=float) / length_m)) / max(i0_edge - 1.0, 1.0e-300)
    return np.clip(h_m, 0.0, h0_m)


def capillary_length_m(config: RealMeshEvolutionConfig) -> float:
    return math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-300)
    )


def film_thickness_mobility_scale(config: RealMeshEvolutionConfig) -> float:
    """Reusable lubrication-style mobility scaling with initial film thickness."""

    reference_um = max(float(config.mobility_reference_film_thickness_um), 1.0e-12)
    h0_um = max(float(config.initial_film_thickness_um), 1.0e-12)
    exponent = float(config.film_thickness_mobility_exponent)
    return float((h0_um / reference_um) ** exponent)


def film_thickness_pressure_scale(config: RealMeshEvolutionConfig) -> float:
    """Reusable scaling for bridge pressure response with initial film thickness."""

    reference_um = max(float(config.pressure_reference_film_thickness_um), 1.0e-12)
    h0_um = max(float(config.initial_film_thickness_um), 1.0e-12)
    exponent = float(config.film_thickness_pressure_exponent)
    return float((h0_um / reference_um) ** exponent)


def bridge_table(config: RealMeshEvolutionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = np.asarray(config.bridge_table_volume_ul, dtype=float)
    radius = np.asarray(config.bridge_table_radius_mm, dtype=float)
    head = np.asarray(config.bridge_table_head_mm, dtype=float)
    order = np.argsort(volume)
    return volume[order], radius[order], head[order]


def bridge_radius_from_volume_mm(config: RealMeshEvolutionConfig, volume_ul: float) -> float:
    volume, radius, _head = bridge_table(config)
    value = float(np.clip(volume_ul, volume[0], volume[-1]))
    return float(PchipInterpolator(volume, radius)(value))


def bridge_head_from_volume_mm(config: RealMeshEvolutionConfig, volume_ul: float) -> float:
    volume, _radius, head = bridge_table(config)
    value = float(np.clip(volume_ul, volume[0], volume[-1]))
    return float(PchipInterpolator(volume, head)(value)) * float(config.bridge_table_head_scale)


def intrinsic_sphere_film_contact_radius_m(config: RealMeshEvolutionConfig) -> float:
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    radius = float(config.sphere_radius_mm) * 1.0e-3
    return math.sqrt(max(2.0 * radius * h0 - h0 * h0, 0.0))


def effective_min_height_um(config: RealMeshEvolutionConfig, t_s: float = 0.0) -> float:
    """Time-dependent neck lower bound used by mesh-side physics operators."""

    early_um = float(config.min_height_um)
    if not bool(config.dynamic_min_height_enabled):
        return early_um
    late_um = float(config.dynamic_min_height_late_um)
    start = float(config.dynamic_min_height_start_s)
    elapsed = max(float(t_s) - start, 0.0)
    activation = 1.0 - math.exp(
        -(
            elapsed / max(float(config.dynamic_min_height_tau_s), 1.0e-12)
        )
        ** max(float(config.dynamic_min_height_exponent), 1.0e-12)
    )
    return float(early_um + np.clip(activation, 0.0, 1.0) * (late_um - early_um))


def effective_min_height_m(config: RealMeshEvolutionConfig, t_s: float = 0.0) -> float:
    return effective_min_height_um(config, t_s) * 1.0e-6


def attached_neck_floor_um(config: RealMeshEvolutionConfig, t_s: float = 0.0) -> float:
    """Local lower bound for the sphere/film neck, separate from film bounds."""

    if not bool(config.attached_neck_floor_enabled):
        return effective_min_height_um(config, t_s)
    early_um = float(config.attached_neck_floor_initial_um)
    late_um = float(config.attached_neck_floor_late_um)
    start = float(config.attached_neck_floor_start_s)
    elapsed = max(float(t_s) - start, 0.0)
    activation = 1.0 - math.exp(
        -(
            elapsed / max(float(config.attached_neck_floor_tau_s), 1.0e-12)
        )
        ** max(float(config.attached_neck_floor_exponent), 1.0e-12)
    )
    return float(early_um + np.clip(activation, 0.0, 1.0) * (late_um - early_um))


def attached_neck_floor_m(config: RealMeshEvolutionConfig, t_s: float = 0.0) -> float:
    return attached_neck_floor_um(config, t_s) * 1.0e-6


def attached_outer_deficit_lower_profile_m(
    config: RealMeshEvolutionConfig,
    r_m: np.ndarray,
    rim_radius_m: float,
    t_s: float,
    initial_h_m: np.ndarray,
) -> np.ndarray:
    """Numerical positivity lower bound for the outer-film mesh.

    Earlier Siekman trial cases used ``attached_outer_deficit_soft_lower_um`` as
    a hard lower envelope here.  That made the mesh hit an artificial shelf in
    Fig. 1(c).  The residual-film scale is now handled by
    ``apply_attached_outer_deficit_soft_repulsion`` as a finite repulsive
    capacity, while this function returns only the positivity bound needed by
    the mesh update.
    """

    del rim_radius_m, initial_h_m
    return np.full_like(np.asarray(r_m, dtype=float), effective_min_height_m(config, t_s), dtype=float)


def attached_outer_deficit_soft_capacity_m(
    config: RealMeshEvolutionConfig,
    r_m: np.ndarray,
    rim_radius_m: float,
    initial_h_m: np.ndarray,
    lower_profile_m: np.ndarray,
) -> np.ndarray:
    """Preferred deficit scale for soft residual-film repulsion.

    The returned value is not a hard capacity.  It is the deficit at which the
    conservative outer-film solve starts to pay a smooth penalty for further
    thinning near the rim.
    """

    initial = np.asarray(initial_h_m, dtype=float)
    lower = np.asarray(lower_profile_m, dtype=float)
    max_deficit = np.maximum(initial - lower, 0.0)
    if not bool(config.attached_outer_deficit_soft_lower_enabled):
        return max_deficit
    floor_m = max(float(config.attached_outer_deficit_soft_lower_um), 0.0) * 1.0e-6
    slope = max(float(config.attached_outer_deficit_soft_lower_slope_um_per_mm), 0.0) * 1.0e-3
    width_m = max(float(config.attached_outer_deficit_soft_lower_width_mm), 1.0e-12) * 1.0e-3
    x = np.maximum(np.asarray(r_m, dtype=float) - float(rim_radius_m), 0.0)
    preferred_h = floor_m + slope * x
    capacity = np.array(max_deficit, copy=True)
    active = x <= width_m
    capacity[active] = np.clip(initial[active] - preferred_h[active], 0.0, max_deficit[active])
    return capacity


def apply_attached_outer_deficit_soft_repulsion(
    config: RealMeshEvolutionConfig,
    r_m: np.ndarray,
    rim_radius_m: float,
    initial_h_m: np.ndarray,
    lower_profile_m: np.ndarray,
    raw_deficit_m: np.ndarray,
) -> np.ndarray:
    """Map an outer-film deficit through a finite soft residual-film barrier."""

    initial = np.asarray(initial_h_m, dtype=float)
    lower = np.asarray(lower_profile_m, dtype=float)
    max_deficit = np.maximum(initial - lower, 0.0)
    raw = np.clip(np.asarray(raw_deficit_m, dtype=float), 0.0, max_deficit)
    strength = float(np.clip(config.attached_outer_deficit_soft_lower_repulsion, 0.0, 1.0))
    if strength <= 0.0 or not bool(config.attached_outer_deficit_soft_lower_enabled):
        return raw

    soft_capacity = attached_outer_deficit_soft_capacity_m(config, r_m, rim_radius_m, initial, lower)
    x = np.maximum(np.asarray(r_m, dtype=float) - float(rim_radius_m), 0.0)
    width_m = max(float(config.attached_outer_deficit_soft_lower_width_mm), 1.0e-12) * 1.0e-3
    active = (x <= width_m) & (soft_capacity < max_deficit - 1.0e-15)
    if not np.any(active):
        return raw

    out = raw.copy()
    cap = np.maximum(soft_capacity[active], 1.0e-12)
    saturated = cap * np.tanh(raw[active] / cap)
    out[active] = (1.0 - strength) * raw[active] + strength * saturated
    return np.clip(out, 0.0, max_deficit)


def attached_outer_deficit_spreading_width_m(
    config: RealMeshEvolutionConfig,
    t_s: float,
) -> float:
    """Return the film-drainage width used by the outer-deficit mesh operator."""

    lc = capillary_length_m(config)
    width = max(float(config.attached_outer_deficit_spreading_width_capillary_lengths), 0.0) * lc
    extra = max(float(config.attached_local_capture_release_extra_width_capillary_lengths), 0.0) * lc
    if extra > 0.0:
        elapsed = max(float(t_s) - float(config.attached_local_capture_release_delay_s), 0.0)
        activation = 1.0 - math.exp(
            -(
                elapsed
                / max(float(config.attached_local_capture_release_time_s), 1.0e-12)
            )
            ** max(float(config.attached_local_capture_release_exponent), 1.0e-12)
        )
        width += extra * float(np.clip(activation, 0.0, 1.0))
    return max(width, 1.0e-9)


def attached_state_closure(
    config: RealMeshEvolutionConfig,
    operators: dict,
    bridge_volume_ul: float,
    bridge_radius_mm: float,
) -> dict[str, float]:
    return operators["attached_bridge_state_closure"](
        initial_film_thickness_m=float(config.initial_film_thickness_um) * 1.0e-6,
        capillary_length_m=capillary_length_m(config),
        intrinsic_contact_radius_m=intrinsic_sphere_film_contact_radius_m(config),
        bridge_radius_m=float(bridge_radius_mm) * 1.0e-3,
        bridge_volume_ul=float(bridge_volume_ul),
        initial_bridge_volume_ul=float(config.inner_bridge_volume_ul),
        rim_base_contact_radius_fraction=float(config.attached_closure_rim_base_contact_radius_fraction),
        rim_transient_contact_radius_fraction=float(config.attached_closure_rim_transient_contact_radius_fraction),
        rim_decay_progress=float(config.attached_closure_rim_decay_progress),
        rim_decay_exponent=float(config.attached_closure_rim_decay_exponent),
        feed_start_annulus_fraction=float(config.attached_closure_feed_start_annulus_fraction),
        feed_capacity_annulus_fraction=float(config.attached_closure_feed_capacity_annulus_fraction),
        feed_growth_progress=float(config.attached_closure_feed_growth_progress),
        feed_growth_exponent=float(config.attached_closure_feed_growth_exponent),
        feed_max_annulus_fraction=float(config.attached_closure_feed_max_annulus_fraction),
        pressure_start_annulus_fraction=float(config.attached_closure_pressure_start_annulus_fraction),
        pressure_growth_progress=float(config.attached_closure_pressure_growth_progress),
        pressure_growth_exponent=float(config.attached_closure_pressure_growth_exponent),
        pressure_boost_factor=float(config.attached_closure_pressure_boost_factor),
        pressure_front_offset_factor=float(config.attached_closure_pressure_front_offset_factor),
        pressure_front_offset_h0_lc_exponent=float(config.attached_closure_pressure_front_offset_h0_lc_exponent),
        pressure_front_width_factor=float(config.attached_closure_pressure_front_width_factor),
        pressure_front_width_h0_lc_exponent=float(config.attached_closure_pressure_front_width_h0_lc_exponent),
    )


def attached_bridge_rim_offset_mm(
    config: RealMeshEvolutionConfig,
    t_s: float,
    operators: dict | None = None,
    bridge_volume_ul: float | None = None,
    bridge_radius_mm: float | None = None,
) -> float:
    """Return the current bridge-radius to film-rim offset.

    A positive transient offset lets the attached bridge front start ahead of
    its late-time value and relax naturally with time.  The rule is reusable
    across attached-bridge cases because it depends only on simulated time, not
    on digitized validation coordinates.
    """

    if (
        bool(config.attached_state_closure_enabled)
        and operators is not None
        and bridge_volume_ul is not None
        and bridge_radius_mm is not None
    ):
        closure = attached_state_closure(config, operators, float(bridge_volume_ul), float(bridge_radius_mm))
        return float(closure["rim_offset_m"]) * 1.0e3

    model = str(config.attached_bridge_rim_offset_model).strip().lower()
    if model in {"capillary_length", "capillary-length", "lc"}:
        offset = capillary_length_m(config) * float(config.attached_bridge_rim_capillary_length_multiplier) * 1.0e3
    else:
        offset = float(config.attached_bridge_rim_offset_mm)
    transient = float(config.attached_bridge_rim_transient_offset_mm)
    if transient == 0.0:
        return offset
    tau = max(float(config.attached_bridge_rim_transient_decay_time_s), 1.0e-12)
    exponent = float(config.attached_bridge_rim_transient_decay_exponent)
    decay = math.exp(-((max(float(t_s), 0.0) / tau) ** exponent))
    return offset + transient * decay


def make_initial_mesh(config: RealMeshEvolutionConfig, operators: dict) -> dict:
    if bool(config.refined_bridge_mesh_enabled):
        n_total = int(config.profile_nodes)
        n_mid = max(8, int(round(n_total * float(config.refined_bridge_mesh_fraction))))
        n_left = max(4, int(round((n_total - n_mid) * 0.45)))
        n_right = max(4, n_total - n_mid - n_left)
        inner_m = float(config.refined_bridge_mesh_inner_mm) * 1.0e-3
        outer_m = float(config.refined_bridge_mesh_outer_mm) * 1.0e-3
        radius_m = float(config.substrate_radius_mm) * 1.0e-3
        left = np.linspace(0.0, inner_m, n_left, endpoint=False)
        mid = np.linspace(inner_m, outer_m, n_mid, endpoint=False)
        right = np.linspace(outer_m, radius_m, n_right + 1)
        r_m = np.unique(np.concatenate([left, mid, right]))
    else:
        r_m = np.linspace(0.0, float(config.substrate_radius_mm) * 1.0e-3, int(config.profile_nodes))
    h_m = initial_profile_m(config, r_m)
    # Post-contact stitch seed from geometry/table only, not from Fig. 1(c).
    pinned_radius_m = float(config.pinned_bridge_radius_mm) * 1.0e-3
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    transition_width_m = 0.22e-3
    blend = 1.0 / (1.0 + np.exp((r_m - pinned_radius_m) / transition_width_m))
    h_m = blend * h0_m + (1.0 - blend) * h_m
    h_m[r_m <= pinned_radius_m] = h0_m
    mesh_config = operators["AxisymmetricFreeSurfaceMeshConfig"](
        azimuthal_nodes=int(config.azimuthal_nodes),
        profile_nodes=int(config.profile_nodes),
    )
    return operators["revolve_profile_to_mesh"](r_m, h_m, mesh_config)


def graph_from_mesh(mesh: dict) -> tuple[list[Vertex], np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(mesh["vertices_m"], dtype=float)
    faces = np.asarray(mesh["faces"], dtype=int)
    rings = np.asarray(mesh["ring_index"], dtype=int)
    vertex_ring = np.zeros(points.shape[0], dtype=int)
    vertex_theta = np.zeros(points.shape[0], dtype=int)
    for i, ring in enumerate(rings):
        for j, idx in enumerate(ring):
            vertex_ring[int(idx)] = i
            vertex_theta[int(idx)] = j
    vertices = [Vertex(points[i], int(vertex_ring[i]), int(vertex_theta[i])) for i in range(points.shape[0])]
    for a, b, c in faces:
        vertices[int(a)].connect(vertices[int(b)])
        vertices[int(b)].connect(vertices[int(c)])
        vertices[int(c)].connect(vertices[int(a)])
    return vertices, faces, rings, vertex_ring


def points_from_vertices(vertices: list[Vertex]) -> np.ndarray:
    return np.asarray([v.x_a for v in vertices], dtype=float)


def set_vertices_from_points(vertices: list[Vertex], points: np.ndarray) -> None:
    for vertex, point in zip(vertices, np.asarray(points, dtype=float)):
        vertex.x_a[:] = point


def lumped_area_from_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.zeros(points.shape[0], dtype=float)
    tri = points[np.asarray(faces, dtype=int)]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    for face, area in zip(np.asarray(faces, dtype=int), tri_area):
        share = float(area) / 3.0
        areas[face[0]] += share
        areas[face[1]] += share
        areas[face[2]] += share
    return np.maximum(areas, 1.0e-18)


def lumped_projected_area_from_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.zeros(points.shape[0], dtype=float)
    tri_xy = points[np.asarray(faces, dtype=int)][:, :, :2]
    tri_area = 0.5 * np.abs(
        (tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
        - (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1])
    )
    for face, area in zip(np.asarray(faces, dtype=int), tri_area):
        share = float(area) / 3.0
        areas[face[0]] += share
        areas[face[1]] += share
        areas[face[2]] += share
    return np.maximum(areas, 1.0e-24)


def volume_under_mesh_ul(points: np.ndarray, faces: np.ndarray) -> float:
    from .operators.free_surface_mesh import projected_volume_under_surface_ul

    return projected_volume_under_surface_ul(points, faces)


def film_profile_from_points(points: np.ndarray, rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_vals: list[float] = []
    h_vals: list[float] = []
    for ring in np.asarray(rings, dtype=int):
        xyz = np.asarray(points, dtype=float)[ring]
        r_vals.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        h_vals.append(float(np.mean(xyz[:, 2])))
    order = np.argsort(r_vals)
    return np.asarray(r_vals, dtype=float)[order], np.asarray(h_vals, dtype=float)[order]


def attached_bridge_snapshot_state(
    points: np.ndarray,
    faces: np.ndarray,
    rings: np.ndarray,
    config: RealMeshEvolutionConfig,
    operators: dict,
    t_value: float,
    step: int,
) -> dict:
    """Build a saved state whose primary mesh includes bridge+film topology."""

    film_state = {
        "film_vertices_m": np.array(points, copy=True),
        "film_faces": np.asarray(faces, dtype=np.int32),
        "film_ring_index": np.asarray(rings, dtype=np.int32),
    }
    current_volume_ul = volume_under_mesh_ul(points, faces)
    initial_profile_r, initial_profile_h = film_profile_from_points(points, rings)
    # The evolving bridge volume is the reduced closure already used by the
    # dynamics; here it controls the geometry of the attached bridge surface.
    # This makes the saved mesh topology consistent with the simulated state,
    # rather than adding a renderer-only line later.
    # Approximate initial mesh volume from the current state plus recorded
    # missing volume is not available here, so callers should pass a state after
    # the normal volume bookkeeping has updated the film.  The mesh profile is
    # the source of the surrounding film geometry.
    missing_ul = max(float(config.inner_bridge_volume_ul), 0.0)
    if "initial_volume_ul_for_snapshot" in operators:
        missing_ul = max(float(operators["initial_volume_ul_for_snapshot"]) - current_volume_ul, 0.0)
    bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
    bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
    bridge_head_mm = bridge_head_from_volume_mm(config, bridge_volume_ul)
    rim_radius_mm = bridge_radius_mm + attached_bridge_rim_offset_mm(config, float(t_value))
    if not math.isfinite(rim_radius_mm) or rim_radius_mm <= 0.0:
        rim_radius_mm = bridge_radius_mm + bridge_pressure_center_offset_mm(config, t_value)

    mesh_config = operators["AxisymmetricFreeSurfaceMeshConfig"](
        azimuthal_nodes=int(config.azimuthal_nodes),
        profile_nodes=int(config.profile_nodes),
    )
    connected = operators["revolve_attached_bridge_film_profile_to_mesh"](
        initial_profile_r,
        initial_profile_h,
        mesh_config,
        sphere_radius_m=float(config.sphere_radius_mm) * 1.0e-3,
        initial_film_thickness_m=float(config.initial_film_thickness_um) * 1.0e-6,
        sphere_bottom_z_m=None if config.sphere_bottom_z_mm is None else float(config.sphere_bottom_z_mm) * 1.0e-3,
        bridge_rim_radius_m=float(rim_radius_mm) * 1.0e-3,
        bridge_head_m=float(bridge_head_mm) * 1.0e-3,
        bridge_nodes=int(config.full_bridge_mesh_bridge_nodes),
    )
    state = {
        "time_s": float(t_value),
        "step": int(step),
        "vertices_m": np.asarray(connected["vertices_m"], dtype=float),
        "faces": np.asarray(connected["faces"], dtype=np.int32),
        "ring_index": np.asarray(connected["ring_index"], dtype=np.int32),
        "ring_region": np.asarray(connected["ring_region"], dtype=np.int32),
        "bridge_contact_radius_m": np.asarray(connected["bridge_contact_radius_m"]),
        "bridge_contact_z_m": np.asarray(connected["bridge_contact_z_m"]),
        "bridge_rim_radius_m": np.asarray(connected["bridge_rim_radius_m"]),
        "bridge_rim_z_m": np.asarray(connected["bridge_rim_z_m"]),
        "bridge_volume_ul": np.asarray(bridge_volume_ul),
        "bridge_radius_mm": np.asarray(bridge_radius_mm),
        "bridge_head_mm": np.asarray(bridge_head_mm),
        "method": "attached bridge+film ddgclib mesh snapshot",
    }
    state.update(film_state)
    return state


def sphere_lower_z_m(config: RealMeshEvolutionConfig, radius_m: np.ndarray | float) -> np.ndarray | float:
    """Lower spherical surface.

    By default the sphere tip is tangent to the initial film surface.  Cases
    can set ``sphere_bottom_z_mm`` to prescribe the sphere-tip height directly.
    """

    radius = np.asarray(radius_m, dtype=float)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    if config.sphere_bottom_z_mm is None:
        center_z = h0 + sphere_radius
    else:
        center_z = float(config.sphere_bottom_z_mm) * 1.0e-3 + sphere_radius
    z = center_z - np.sqrt(np.maximum(sphere_radius * sphere_radius - radius * radius, 0.0))
    if np.isscalar(radius_m):
        return float(z)
    return z


def sphere_radius_at_lower_z_m(config: RealMeshEvolutionConfig, z_m: float) -> float:
    """Horizontal radius of the lower spherical surface at height ``z_m``."""

    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    if config.sphere_bottom_z_mm is None:
        center_z = h0 + sphere_radius
    else:
        center_z = float(config.sphere_bottom_z_mm) * 1.0e-3 + sphere_radius
    z = float(np.clip(float(z_m), center_z - sphere_radius, center_z + sphere_radius))
    return math.sqrt(max(sphere_radius * sphere_radius - (center_z - z) ** 2, 0.0))


def make_attached_bridge_mesh_from_profile(
    r_m: np.ndarray,
    h_m: np.ndarray,
    config: RealMeshEvolutionConfig,
    operators: dict,
    bridge_volume_ul: float,
) -> dict:
    """Create the primary connected bridge+film mesh used by Case 11."""

    bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
    bridge_rim_radius_mm = bridge_radius_mm + attached_bridge_rim_offset_mm(config, 0.0)
    bridge_head_mm = bridge_head_from_volume_mm(config, bridge_volume_ul)
    if config.attached_initial_rim_radius_mm is not None:
        bridge_rim_radius_mm = float(config.attached_initial_rim_radius_mm)
    if config.attached_initial_contact_radius_mm is not None:
        contact_r_m = max(float(config.attached_initial_contact_radius_mm), 0.0) * 1.0e-3
        bridge_head_mm = float(sphere_lower_z_m(config, contact_r_m)) * 1.0e3
    mesh_config = operators["AxisymmetricFreeSurfaceMeshConfig"](
        azimuthal_nodes=int(config.azimuthal_nodes),
        profile_nodes=int(config.profile_nodes),
    )
    return operators["revolve_attached_bridge_film_profile_to_mesh"](
        r_m,
        h_m,
        mesh_config,
        sphere_radius_m=float(config.sphere_radius_mm) * 1.0e-3,
        initial_film_thickness_m=float(config.initial_film_thickness_um) * 1.0e-6,
        sphere_bottom_z_m=None if config.sphere_bottom_z_mm is None else float(config.sphere_bottom_z_mm) * 1.0e-3,
        bridge_rim_radius_m=float(bridge_rim_radius_mm) * 1.0e-3,
        bridge_head_m=float(bridge_head_mm) * 1.0e-3,
        bridge_nodes=int(config.full_bridge_mesh_bridge_nodes),
    )


def attached_ring_geometry(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    region = np.asarray(ring_region, dtype=int)
    bridge_indices = np.where(region == 0)[0]
    if bridge_indices.size == 0:
        raise ValueError("attached bridge mesh requires at least one bridge ring")
    contact_index = int(bridge_indices[0])
    rim_index = int(bridge_indices[-1])
    contact_ring = np.asarray(rings[contact_index], dtype=int)
    rim_ring = np.asarray(rings[rim_index], dtype=int)
    contact_xyz = np.asarray(points, dtype=float)[contact_ring]
    rim_xyz = np.asarray(points, dtype=float)[rim_ring]
    return {
        "contact_index": contact_index,
        "rim_index": rim_index,
        "contact_ring": contact_ring,
        "rim_ring": rim_ring,
        "contact_radius_m": float(np.mean(np.hypot(contact_xyz[:, 0], contact_xyz[:, 1]))),
        "contact_z_m": float(np.mean(contact_xyz[:, 2])),
        "rim_radius_m": float(np.mean(np.hypot(rim_xyz[:, 0], rim_xyz[:, 1]))),
        "rim_z_m": float(np.mean(rim_xyz[:, 2])),
    }


def attached_film_profile_m(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the comparable outer-film profile from a connected mesh."""

    geom = attached_ring_geometry(points, rings, ring_region)
    rim_radius_m = float(geom["rim_radius_m"])
    r_vals = [rim_radius_m]
    h_vals = [float(geom["rim_z_m"])]
    for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
        if int(ring_region[min(ring_i, len(ring_region) - 1)]) != 1:
            continue
        xyz = np.asarray(points, dtype=float)[np.asarray(ring, dtype=int)]
        ring_radius_m = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        if ring_radius_m <= rim_radius_m + 1.0e-9:
            continue
        r_vals.append(ring_radius_m)
        h_vals.append(float(np.mean(xyz[:, 2])))
    order = np.argsort(r_vals)
    return np.asarray(r_vals, dtype=float)[order], np.asarray(h_vals, dtype=float)[order]


def attached_missing_outer_film_volume_ul(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
) -> float:
    """Outer-film volume loss feeding the bridge for an attached mesh."""

    r, h = attached_film_profile_m(points, rings, ring_region)
    if r.size < 2:
        return 0.0
    initial_h = initial_profile_m(config, r)
    integrand = 2.0 * math.pi * r * np.maximum(initial_h - h, 0.0)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    missing_m3 = float(integrate(integrand, r))
    if (
        bool(config.attached_volume_projection_bridge_reservoir_enabled)
        and bool(config.attached_volume_projection_bridge_reservoir_counts_as_missing)
    ):
        geom = attached_ring_geometry(points, rings, ring_region)
        contact_m = float(geom["contact_radius_m"])
        rim_m = float(geom["rim_radius_m"])
        inner_m = contact_m + max(float(config.attached_volume_projection_bridge_reservoir_inner_margin_mm), 0.0) * 1.0e-3
        outer_m = rim_m - max(float(config.attached_volume_projection_bridge_reservoir_outer_gap_mm), 0.0) * 1.0e-3
        bridge_r: list[float] = []
        bridge_h: list[float] = []
        for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
            if int(ring_region[min(ring_i, len(ring_region) - 1)]) != 0:
                continue
            xyz = np.asarray(points, dtype=float)[np.asarray(ring, dtype=int)]
            ring_radius_m = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
            if ring_radius_m <= inner_m or ring_radius_m >= outer_m:
                continue
            bridge_r.append(ring_radius_m)
            bridge_h.append(float(np.mean(xyz[:, 2])))
        if len(bridge_r) >= 2:
            bridge_r_arr = np.asarray(bridge_r, dtype=float)
            bridge_h_arr = np.asarray(bridge_h, dtype=float)
            order = np.argsort(bridge_r_arr)
            bridge_r_arr = bridge_r_arr[order]
            bridge_h_arr = bridge_h_arr[order]
            bridge_initial_h = float(config.initial_film_thickness_um) * 1.0e-6
            bridge_integrand = 2.0 * math.pi * bridge_r_arr * np.maximum(bridge_initial_h - bridge_h_arr, 0.0)
            missing_m3 += float(integrate(bridge_integrand, bridge_r_arr))
    return float(missing_m3 * 1.0e9)


def build_attached_reduced_bridge_driver(
    config: RealMeshEvolutionConfig,
    operators: dict,
) -> dict[str, np.ndarray] | None:
    if not bool(config.attached_reduced_bridge_film_driver_enabled):
        return None
    t_end = float(config.max_steps) * float(config.dt_s)
    diagnostic_times = tuple(float(step) * float(config.dt_s) for step in range(0, int(config.max_steps) + 1, max(int(config.record_every_steps), 1)))
    driver_config = operators["AxisymmetricBridgeFilmConfig"](
        sphere_radius_mm=float(config.sphere_radius_mm),
        substrate_radius_mm=float(config.substrate_radius_mm),
        initial_film_thickness_um=float(config.initial_film_thickness_um),
        surface_tension_n_m=float(config.surface_tension_n_m),
        viscosity_pa_s=float(config.viscosity_pa_s),
        density_kg_m3=float(config.density_kg_m3),
        gravity_m_s2=float(config.gravity_m_s2),
        t_end_s=t_end,
        dt_s=float(config.dt_s),
        snapshot_times_s=tuple(float(t) for t in config.snapshot_times_s if 0.0 <= float(t) <= t_end),
        diagnostic_times_s=diagnostic_times,
        grid_nodes=int(config.attached_reduced_driver_grid_nodes),
        bridge_table_volume_ul=tuple(float(v) for v in config.bridge_table_volume_ul),
        bridge_table_radius_mm=tuple(float(v) for v in config.bridge_table_radius_mm),
        bridge_table_head_mm=tuple(float(v) for v in config.bridge_table_head_mm),
        bridge_table_head_scale=float(config.bridge_table_head_scale),
        initial_bridge_volume_ul=float(config.inner_bridge_volume_ul),
        initial_bridge_radius_mm=max(float(config.initial_bridge_radius_mm), intrinsic_sphere_film_contact_radius_m(config) * 1.0e3),
        bridge_pressure_activation_time_s=float(config.attached_reduced_driver_pressure_activation_time_s),
        bridge_pressure_relax_time_s=float(config.attached_reduced_driver_pressure_relax_time_s),
        bridge_pressure_relax_exponent=float(config.attached_reduced_driver_pressure_relax_exponent),
        bridge_inflow_bottleneck_enabled=bool(config.attached_reduced_driver_bottleneck_enabled),
        bridge_inflow_bottleneck_volume_ul=float(config.attached_reduced_driver_bottleneck_volume_ul),
        bridge_inflow_bottleneck_exponent=float(config.attached_reduced_driver_bottleneck_exponent),
        bridge_inflow_bottleneck_floor=float(config.attached_reduced_driver_bottleneck_floor),
        inner_pressure_boundary_height_enabled=bool(config.attached_reduced_driver_inner_pressure_boundary_enabled),
        local_capture_enabled=bool(config.attached_local_capture_enabled),
        local_capture_width_capillary_lengths=float(config.attached_local_capture_width_capillary_lengths),
        local_capture_time_s=float(config.attached_local_capture_time_s),
        local_capture_exponent=float(config.attached_local_capture_exponent),
        local_capture_release_time_s=float(config.attached_local_capture_release_time_s),
        local_capture_release_exponent=float(config.attached_local_capture_release_exponent),
        local_capture_release_saturation_enabled=bool(config.attached_local_capture_release_saturation_enabled),
        local_capture_release_delay_s=float(config.attached_local_capture_release_delay_s),
        local_capture_release_extra_width_capillary_lengths=float(
            config.attached_local_capture_release_extra_width_capillary_lengths
        ),
        bridge_radius_speed_limit_mm_s=20.0,
    )
    simulation = operators["simulate_bridge_film"](driver_config)
    times = np.asarray(sorted(float(t) for t in simulation["bridge_radius_by_time_m"].keys()), dtype=float)
    radii = np.asarray([float(simulation["bridge_radius_by_time_m"][round(float(t), 10)]) for t in times], dtype=float)
    if "bridge_delta_volume_with_local_capture_ul" in operators:
        volumes = np.asarray(
            [
                float(operators["bridge_delta_volume_with_local_capture_ul"](driver_config, radius, t_s))
                for t_s, radius in zip(times, radii)
            ],
            dtype=float,
        )
    else:
        volumes = np.asarray(
            [float(operators["bridge_volume_from_radius_m3"](driver_config, radius) * 1.0e9) for radius in radii],
            dtype=float,
        )
        volumes = np.maximum(volumes - float(volumes[0]), 0.0)
    return {
        "time_s": times,
        "delta_volume_ul": volumes,
        "profile_r_m": simulation["profile_r_m"],
        "profiles_m": simulation["profiles_m"],
        "bridge_radius_by_time_m": simulation["bridge_radius_by_time_m"],
    }


def apply_attached_reduced_driver_profile_projection(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    reduced_volume_driver: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the attached outer-film rings onto the computed lubrication profile.

    This uses the raw reduced ddgclib lubrication solve, not the empirical
    finite-inner dimple reconstruction.  It is a mesh-state update so the saved
    `.msh/.npz` files contain the same profile that is plotted for validation.
    """

    if reduced_volume_driver is None or not bool(config.attached_reduced_driver_profile_projection_enabled):
        return points, velocities
    profiles = reduced_volume_driver.get("profiles_m")
    profile_r = reduced_volume_driver.get("profile_r_m")
    if not isinstance(profiles, dict) or not isinstance(profile_r, dict) or not profiles:
        return points, velocities
    times = np.asarray(sorted(float(t) for t in profiles.keys()), dtype=float)
    key = float(times[int(np.argmin(np.abs(times - float(t_s))))])
    key_round = round(key, 10)
    if key_round not in profiles:
        key_round = key
    if key_round not in profiles or key_round not in profile_r:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    film_indices = np.where(ring_region_arr == 1)[0]
    if film_indices.size == 0:
        return out, vel

    r_grid = np.asarray(profile_r[key_round], dtype=float)
    h_grid = np.asarray(profiles[key_round], dtype=float)
    relaxation = float(np.clip(config.attached_reduced_driver_profile_projection_relaxation, 0.0, 1.0))
    if relaxation <= 0.0:
        return out, vel
    min_z = float(config.min_height_um) * 1.0e-6
    max_z = max(float(config.max_height_um), float(config.initial_film_thickness_um)) * 1.0e-6
    old = out.copy()
    for ring_i in film_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        xyz = out[ring]
        ring_r = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        target_z = float(np.interp(ring_r, r_grid, h_grid, left=h_grid[0], right=h_grid[-1]))
        target_z = float(np.clip(target_z, min_z, max_z))
        out[ring, 2] = out[ring, 2] + relaxation * (target_z - out[ring, 2])
    vel[:] = (out - old) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def attached_visible_feed_fraction(
    config: RealMeshEvolutionConfig,
    rim_radius_m: float,
    target_missing_ul: float,
    contact_radius_m: float | None = None,
) -> float:
    """Fraction of bridge growth that is resolved as visible outer-film drain.

    The remaining growth is carried by the saturated bridge/neck volume.  The
    transition variable is the bridge volume divided by the natural annular
    capillary feed scale ``2*pi*a*h0*l_c``.  This keeps the split geometric and
    material-scale based instead of tying it to one validation timestamp.
    """

    if not bool(config.attached_visible_feed_partition_enabled):
        return 1.0
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    scale_radius_m = max(float(rim_radius_m), 1.0e-12)
    radius_mode = str(config.attached_visible_feed_scale_radius_mode).strip().lower()
    if radius_mode in {"intrinsic", "first_contact", "first-contact"}:
        scale_radius_m = max(intrinsic_sphere_film_contact_radius_m(config), 1.0e-12)
    elif radius_mode in {"min_intrinsic_rim", "minimum"}:
        scale_radius_m = max(min(float(rim_radius_m), intrinsic_sphere_film_contact_radius_m(config)), 1.0e-12)
    elif radius_mode in {"contact_blend", "cox_contact_blend", "contact-line-blend"}:
        intrinsic_m = max(intrinsic_sphere_film_contact_radius_m(config), 1.0e-12)
        if contact_radius_m is None:
            contact_progress = 0.0
        else:
            contact_progress = float(np.clip(float(contact_radius_m) / intrinsic_m, 0.0, 1.0))
        contact_progress = contact_progress ** max(float(config.attached_visible_feed_contact_blend_exponent), 1.0e-12)
        scale_radius_m = max(
            (1.0 - contact_progress) * float(rim_radius_m) + contact_progress * intrinsic_m,
            1.0e-12,
        )
    annular_scale_ul = (
        2.0
        * math.pi
        * scale_radius_m
        * h0
        * capillary_length_m(config)
        * 1.0e9
    )
    progress = max(float(target_missing_ul), 0.0) / max(float(annular_scale_ul), 1.0e-30)
    width = max(float(config.attached_visible_feed_transition_width), 1.0e-12)
    center = float(config.attached_visible_feed_transition_progress)
    activation = 1.0 / (1.0 + math.exp(-(progress - center) / width))
    low = float(config.attached_visible_feed_min_fraction)
    high = float(config.attached_visible_feed_max_fraction)
    return float(np.clip(low + (high - low) * activation, 0.0, 1.0))


def attached_sphere_cap_volume_ul(config: RealMeshEvolutionConfig, contact_radius_m: float) -> float:
    """Spherical-cap volume swept by a wetted contact radius on the sphere."""

    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    contact = float(np.clip(float(contact_radius_m), 0.0, sphere_radius * 0.999999))
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    if config.sphere_bottom_z_mm is None:
        sphere_bottom_z_m = h0
    else:
        sphere_bottom_z_m = float(config.sphere_bottom_z_mm) * 1.0e-3
    center_z = sphere_bottom_z_m + sphere_radius
    root = math.sqrt(max(sphere_radius * sphere_radius - contact * contact, 0.0))
    volume_m3 = (
        math.pi * center_z * contact * contact
        - (2.0 * math.pi / 3.0) * (sphere_radius**3 - root**3)
    )
    return float(max(volume_m3, 0.0) * 1.0e9)


def attached_visible_feed_target_ul(
    config: RealMeshEvolutionConfig,
    rim_radius_m: float,
    target_missing_ul: float,
    contact_radius_m: float | None,
) -> tuple[float, float, float]:
    """Return visible-film target, feed fraction, and reservoir-corrected pool."""

    raw_pool_ul = max(float(target_missing_ul), 0.0)
    corrected_pool_ul = raw_pool_ul
    if bool(config.attached_visible_feed_subtract_wetted_sphere_cap) and contact_radius_m is not None:
        if config.attached_initial_contact_radius_mm is None:
            reference_contact_m = 0.0
        else:
            reference_contact_m = max(float(config.attached_initial_contact_radius_mm), 0.0) * 1.0e-3
        cap_delta_ul = max(
            attached_sphere_cap_volume_ul(config, float(contact_radius_m))
            - attached_sphere_cap_volume_ul(config, reference_contact_m),
            0.0,
        )
        corrected_pool_ul = max(raw_pool_ul - cap_delta_ul, 0.0)
    fraction_volume_ul = (
        corrected_pool_ul
        if bool(config.attached_visible_feed_cap_fraction_uses_corrected_volume)
        else raw_pool_ul
    )
    visible_fraction = attached_visible_feed_fraction(
        config,
        rim_radius_m,
        fraction_volume_ul,
        contact_radius_m=contact_radius_m,
    )
    visible_target_ul = corrected_pool_ul * visible_fraction
    return float(visible_target_ul), float(visible_fraction), float(corrected_pool_ul)


def attached_projection_width_m(
    config: RealMeshEvolutionConfig,
    visible_fraction: float = 1.0,
) -> float:
    """Width of the annular film feed region used by attached mesh projection."""

    capillary_widths = float(config.attached_volume_projection_width_capillary_lengths)
    if capillary_widths > 0.0:
        width_m = capillary_widths * capillary_length_m(config)
    else:
        width_m = float(config.attached_volume_projection_width_mm) * 1.0e-3
    if bool(config.attached_volume_projection_scale_width_with_visible_fraction):
        width_m *= float(np.clip(visible_fraction, 0.0, 1.0))
    min_width_m = max(float(config.attached_volume_projection_min_width_mm), 0.0) * 1.0e-3
    return max(width_m, min_width_m, 1.0e-9)


def project_attached_missing_volume_to_target(
    points: np.ndarray,
    faces: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    initial_missing_ul: float,
    target_missing_ul: float,
    t_s: float = 0.0,
) -> np.ndarray:
    current_missing_ul = max(
        attached_missing_outer_film_volume_ul(points, rings, ring_region, config) - float(initial_missing_ul),
        0.0,
    )
    out = np.array(points, copy=True)
    _bridge_mask, film_mask = attached_vertex_region_masks(out.shape[0], rings, ring_region)
    geom = attached_ring_geometry(out, rings, ring_region)
    r = np.hypot(out[:, 0], out[:, 1])
    rim = float(geom["rim_radius_m"])
    outer_m = float(config.substrate_radius_mm) * 1.0e-3
    visible_target_ul, visible_fraction, _corrected_pool_ul = attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim,
        target_missing_ul=target_missing_ul,
        contact_radius_m=float(geom["contact_radius_m"]),
    )
    if bool(config.attached_volume_projection_time_release_enabled):
        elapsed = max(float(t_s) - float(config.attached_volume_projection_time_release_start_s), 0.0)
        tau = max(float(config.attached_volume_projection_time_release_tau_s), 1.0e-12)
        exponent = max(float(config.attached_volume_projection_time_release_exponent), 1.0e-12)
        activation = 1.0 - math.exp(-((elapsed / tau) ** exponent))
        floor = float(config.attached_volume_projection_time_release_floor_fraction)
        ceiling = float(config.attached_volume_projection_time_release_ceiling_fraction)
        release_fraction = float(np.clip(floor + (ceiling - floor) * activation, 0.0, 1.0))
        visible_target_ul *= release_fraction
        visible_fraction *= release_fraction
    delta_ul = visible_target_ul - current_missing_ul
    if delta_ul < 0.0 and bool(config.attached_bridge_growth_irreversible_enabled):
        return out
    if abs(delta_ul) <= 1.0e-8:
        return out
    projection_width_m = attached_projection_width_m(config, visible_fraction)
    annulus_outer_m = min(outer_m * 0.96, rim + projection_width_m)
    active = film_mask & (r > rim) & (r < annulus_outer_m)
    if not np.any(active):
        return out

    # Smoothly distribute the reduced model's bridge-volume exchange through the
    # attached outer film.  This is a coupling projection, not a plotting scale.
    weights = np.zeros(out.shape[0], dtype=float)
    span = max(annulus_outer_m - rim, 1.0e-9)
    xi = np.clip((r[active] - rim) / span, 0.0, 1.0)
    falloff_exponent = max(float(config.attached_volume_projection_weight_exponent), 1.0e-9)
    weights[active] = (1.0 - xi) ** falloff_exponent
    if bool(config.attached_volume_projection_bridge_reservoir_enabled):
        geom = attached_ring_geometry(out, rings, ring_region)
        contact_m = float(geom["contact_radius_m"])
        inner_m = contact_m + max(float(config.attached_volume_projection_bridge_reservoir_inner_margin_mm), 0.0) * 1.0e-3
        outer_gap_m = max(float(config.attached_volume_projection_bridge_reservoir_outer_gap_mm), 0.0) * 1.0e-3
        reservoir = _bridge_mask & (r > inner_m) & (r < rim - outer_gap_m)
        if np.any(reservoir):
            weights[reservoir] = max(float(config.attached_volume_projection_bridge_reservoir_weight), 0.0)
    area = lumped_projected_area_from_faces(out, faces)
    projection_floor_m = max(
        effective_min_height_m(config, t_s),
        float(config.attached_volume_projection_min_height_fraction)
        * float(config.initial_film_thickness_um)
        * 1.0e-6,
    )
    if delta_ul > 0.0:
        capacity = np.maximum(out[:, 2] - projection_floor_m, 0.0)
        sign = -1.0
    else:
        capacity = np.maximum(float(config.max_height_um) * 1.0e-6 - out[:, 2], 0.0)
        sign = 1.0
    weights *= capacity > 0.0
    remaining_m3 = abs(delta_ul) * 1.0e-9
    if remaining_m3 <= 1.0e-24:
        return out
    active_capacity = np.asarray(capacity, dtype=float)
    for _ in range(64):
        active = (weights > 0.0) & (active_capacity > 1.0e-15)
        if not np.any(active):
            break
        denom = float(np.sum(area[active] * weights[active]))
        if denom <= 1.0e-24:
            break
        dz_target = remaining_m3 / denom
        dz_cap = float(np.min(active_capacity[active] / weights[active]))
        dz = min(dz_target, dz_cap)
        if dz <= 0.0:
            break
        out[active, 2] += sign * dz * weights[active]
        used_m3 = float(np.sum(area[active] * weights[active] * dz))
        remaining_m3 = max(remaining_m3 - used_m3, 0.0)
        active_capacity[active] = np.maximum(active_capacity[active] - dz * weights[active], 0.0)
        if remaining_m3 <= 1.0e-18 or dz_target <= dz_cap * (1.0 + 1.0e-12):
            break
    out[:, 2] = np.clip(out[:, 2], effective_min_height_m(config, t_s), float(config.max_height_um) * 1.0e-6)
    return out


def apply_attached_volume_recovery_front_profile(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    initial_missing_ul: float,
    target_missing_ul: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Set the visible outer-film drainage as a finite capillary recovery front.

    The bridge can grow faster than the resolved film depression.  This operator
    computes the visible fraction with the same time-release law as the volume
    projection, then solves for the radial recovery-front width whose monotone
    profile carries that visible missing volume.  It updates mesh vertices, not
    rendered curves.
    """

    if not bool(config.attached_volume_projection_recovery_front_enabled):
        return points, velocities
    if float(t_s) < float(config.attached_volume_projection_recovery_front_start_s):
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    film_indices = np.where(ring_region_arr == 1)[0]
    if film_indices.size < 4:
        return points, velocities

    geom = attached_ring_geometry(out, rings, ring_region)
    visible_target_ul, visible_fraction, _corrected_pool_ul = attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=float(geom["rim_radius_m"]),
        target_missing_ul=target_missing_ul,
        contact_radius_m=float(geom["contact_radius_m"]),
    )
    if bool(config.attached_volume_projection_time_release_enabled):
        elapsed = max(float(t_s) - float(config.attached_volume_projection_time_release_start_s), 0.0)
        tau = max(float(config.attached_volume_projection_time_release_tau_s), 1.0e-12)
        exponent = max(float(config.attached_volume_projection_time_release_exponent), 1.0e-12)
        activation = 1.0 - math.exp(-((elapsed / tau) ** exponent))
        floor = float(config.attached_volume_projection_time_release_floor_fraction)
        ceiling = float(config.attached_volume_projection_time_release_ceiling_fraction)
        release_fraction = float(np.clip(floor + (ceiling - floor) * activation, 0.0, 1.0))
        visible_target_ul *= release_fraction
        visible_fraction *= release_fraction
    if visible_target_ul <= 1.0e-10:
        return points, velocities

    ring_ids: list[int] = []
    ring_r: list[float] = []
    ring_h: list[float] = []
    for ring_i in film_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        xyz = out[ring]
        ring_ids.append(int(ring_i))
        ring_r.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        ring_h.append(float(np.mean(xyz[:, 2])))
    order = np.argsort(np.asarray(ring_r, dtype=float))
    ring_ids = [ring_ids[int(i)] for i in order]
    r_arr = np.asarray(ring_r, dtype=float)[order]
    h_arr = np.asarray(ring_h, dtype=float)[order]
    if r_arr.size < 4:
        return points, velocities

    search = (r_arr >= float(geom["rim_radius_m"]) - 0.25e-3) & (
        r_arr <= float(geom["rim_radius_m"]) + 0.35e-3
    )
    if not np.any(search):
        search = np.ones_like(r_arr, dtype=bool)
    local_indices = np.where(search)[0]
    start_local = int(local_indices[int(np.argmin(h_arr[search]))])
    r_start = float(r_arr[start_local])
    h_start = float(h_arr[start_local])
    initial_h = initial_profile_m(config, r_arr)
    lower = attached_outer_deficit_lower_profile_m(config, r_arr, float(geom["rim_radius_m"]), t_s, initial_h)
    upper = np.maximum(initial_h, lower)
    h_start = float(np.clip(h_start, lower[start_local], upper[start_local]))
    target_total_missing_ul = max(float(visible_target_ul), 0.0)

    min_width_m = max(float(config.attached_volume_projection_recovery_front_min_width_um), 0.0) * 1.0e-6
    max_width_m = max(float(config.attached_volume_projection_recovery_front_max_width_mm), 0.0) * 1.0e-3
    late_activation = 0.0
    late_width_m = max(
        float(config.attached_volume_projection_recovery_front_late_max_width_mm),
        0.0,
    ) * 1.0e-3
    late_shape_exponent = max(
        float(config.attached_volume_projection_recovery_front_late_shape_exponent),
        0.0,
    )
    if late_width_m > 0.0 or late_shape_exponent > 0.0:
        elapsed = max(
            float(t_s) - float(config.attached_volume_projection_recovery_front_late_start_s),
            0.0,
        )
        tau = max(float(config.attached_volume_projection_recovery_front_late_tau_s), 1.0e-12)
        ramp_exponent = max(
            float(config.attached_volume_projection_recovery_front_late_ramp_exponent),
            1.0e-12,
        )
        late_activation = 1.0 - math.exp(-((elapsed / tau) ** ramp_exponent))
        late_activation = float(np.clip(late_activation, 0.0, 1.0))
    if late_width_m > 0.0 and late_activation > 0.0:
        max_width_m = (1.0 - late_activation) * max_width_m + late_activation * late_width_m
    substrate_m = float(config.substrate_radius_mm) * 1.0e-3
    max_width_m = max(min(max_width_m, substrate_m - r_start), min_width_m)
    shape_exponent = max(float(config.attached_volume_projection_recovery_front_exponent), 1.0e-12)
    if late_shape_exponent > 0.0 and late_activation > 0.0:
        shape_exponent = (
            (1.0 - late_activation) * shape_exponent
            + late_activation * max(late_shape_exponent, 1.0e-12)
        )

    def profile_for_width(width_m: float) -> np.ndarray:
        prof = h_arr.copy()
        active = r_arr >= r_start
        s = np.clip((r_arr[active] - r_start) / max(width_m, 1.0e-12), 0.0, 1.0)
        recovery = h_start + (initial_h[active] - h_start) * (s**shape_exponent)
        recovery = np.where(s >= 1.0, initial_h[active], recovery)
        prof[active] = np.clip(recovery, lower[active], upper[active])
        prof[: start_local + 1] = h_arr[: start_local + 1]
        return prof

    def missing_delta_for_profile(profile_h: np.ndarray) -> float:
        trial = out.copy()
        for local_i, ring_i in enumerate(ring_ids):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            trial[ring, 2] = float(profile_h[local_i])
        return max(
            attached_missing_outer_film_volume_ul(trial, rings, ring_region, config) - float(initial_missing_ul),
            0.0,
        )

    lo = min_width_m
    hi = max_width_m
    if missing_delta_for_profile(profile_for_width(hi)) < target_total_missing_ul:
        width_m = hi
    else:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if missing_delta_for_profile(profile_for_width(mid)) < target_total_missing_ul:
                lo = mid
            else:
                hi = mid
        width_m = 0.5 * (lo + hi)

    target_h = profile_for_width(width_m)
    relaxation = float(np.clip(config.attached_volume_projection_recovery_front_relaxation, 0.0, 1.0))
    if relaxation <= 0.0:
        return points, velocities
    old = out.copy()
    for local_i, ring_i in enumerate(ring_ids):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        out[ring, 2] = (1.0 - relaxation) * out[ring, 2] + relaxation * float(target_h[local_i])
    vel[:] = (out - old) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def apply_attached_visible_feed_profile_constraint(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    initial_missing_ul: float,
    target_missing_ul: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a local capillary-length dimple profile to the outer film.

    This is an axisymmetric quasi-static closure for the visible film profile
    adjacent to the saturated bridge.  The target volume is the visible-feed
    part of bridge growth; the profile decays over a capillary-length-scaled
    annulus and is solved by bisection so it changes the mesh volume, not only
    the rendered curve.
    """

    if not bool(config.attached_visible_feed_profile_constraint_enabled):
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    _bridge_mask, film_mask = attached_vertex_region_masks(out.shape[0], rings, ring_region)
    geom = attached_ring_geometry(out, rings, ring_region)
    rim = float(geom["rim_radius_m"])
    visible_target_ul, visible_fraction, _corrected_pool_ul = attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim,
        target_missing_ul=target_missing_ul,
        contact_radius_m=float(geom["contact_radius_m"]),
    )
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    annular_scale_ul = (
        2.0
        * math.pi
        * max(rim, 1.0e-12)
        * h0
        * capillary_length_m(config)
        * 1.0e9
    )
    progress = max(float(target_missing_ul), 0.0) / max(annular_scale_ul, 1.0e-30)
    boost_width = max(float(config.attached_visible_feed_profile_late_boost_width), 1.0e-12)
    boost_activation = 1.0 / (
        1.0
        + math.exp(
            -(
                progress - float(config.attached_visible_feed_profile_late_boost_progress)
            )
            / boost_width
        )
    )
    profile_target_multiplier = 1.0 + max(
        float(config.attached_visible_feed_profile_late_volume_boost_factor),
        0.0,
    ) * boost_activation
    visible_target_ul *= profile_target_multiplier
    if visible_target_ul <= 1.0e-10:
        return out, vel

    cap_width = capillary_length_m(config) * (
        max(float(config.attached_visible_feed_profile_base_width_capillary_lengths), 0.0)
        + max(float(config.attached_visible_feed_profile_fraction_width_capillary_lengths), 0.0)
        * visible_fraction
    )
    cap_width = max(cap_width, float(config.attached_volume_projection_min_width_mm) * 1.0e-3, 1.0e-9)
    exponent = max(float(config.attached_visible_feed_profile_decay_exponent), 1.0e-12)
    r = np.hypot(out[:, 0], out[:, 1])
    outer_m = float(config.substrate_radius_mm) * 1.0e-3
    active = film_mask & (r >= rim) & (r < min(outer_m * 0.999, rim + 6.0 * cap_width))
    if not np.any(active):
        return out, vel

    shape = np.zeros(out.shape[0], dtype=float)
    xi = np.maximum((r[active] - rim) / cap_width, 0.0)
    shape[active] = np.exp(-(xi**exponent))
    initial_h = initial_profile_m(config, r)
    min_h = float(config.min_height_um) * 1.0e-6
    max_amp = max(float(config.initial_film_thickness_um) * 1.0e-6 - min_h, 0.0)

    def missing_for_amplitude(amplitude_m: float) -> float:
        trial = out.copy()
        target_z = initial_h - amplitude_m * shape
        trial[active, 2] = np.clip(target_z[active], min_h, float(config.max_height_um) * 1.0e-6)
        return max(
            attached_missing_outer_film_volume_ul(trial, rings, ring_region, config) - float(initial_missing_ul),
            0.0,
        )

    if missing_for_amplitude(max_amp) < visible_target_ul:
        amplitude = max_amp
    else:
        lo = 0.0
        hi = max_amp
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if missing_for_amplitude(mid) < visible_target_ul:
                lo = mid
            else:
                hi = mid
        amplitude = 0.5 * (lo + hi)

    old = out.copy()
    target_z = initial_h - amplitude * shape
    out[active, 2] = np.clip(target_z[active], min_h, float(config.max_height_um) * 1.0e-6)
    vel[active] = (out[active] - old[active]) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def apply_attached_neck_boundary_layer_operator(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    target_missing_ul: float,
    operators: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a reusable Cox/lubrication neck layer to the attached film mesh."""

    if not bool(config.attached_neck_boundary_layer_enabled):
        return points, velocities
    if float(target_missing_ul) <= 1.0e-12:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    film_indices = np.where(ring_region_arr == 1)[0]
    if film_indices.size < 2:
        return out, vel

    geom = attached_ring_geometry(out, rings, ring_region)
    rim = float(geom["rim_radius_m"])
    rim_z = float(geom["rim_z_m"])
    rim_ring = np.asarray(geom["rim_ring"], dtype=int)
    contact_ring = np.asarray(geom["contact_ring"], dtype=int)
    contact_radius = float(geom["contact_radius_m"])
    film_r: list[float] = []
    film_ring_ids: list[int] = []
    for ring_i in film_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        xyz = out[ring]
        ring_r = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        if ring_r <= rim + 1.0e-12:
            continue
        film_r.append(ring_r)
        film_ring_ids.append(int(ring_i))
    if len(film_r) < 2:
        return out, vel

    order = np.argsort(np.asarray(film_r, dtype=float))
    r_arr = np.asarray(film_r, dtype=float)[order]
    film_ring_ids = [film_ring_ids[int(i)] for i in order]
    initial_h = initial_profile_m(config, r_arr)
    rim_radius_for_fraction = max(rim, 1.0e-12)
    visible_target_ul, visible_fraction, _corrected_pool_ul = attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim_radius_for_fraction,
        target_missing_ul=float(target_missing_ul),
        contact_radius_m=contact_radius,
    )
    late_multiplier = max(float(config.attached_neck_visible_feed_late_multiplier), 1.0)
    if late_multiplier > 1.0:
        elapsed_late = max(float(t_s) - float(config.attached_neck_visible_feed_late_start_s), 0.0)
        late_activation = 1.0 - math.exp(
            -(
                elapsed_late
                / max(float(config.attached_neck_visible_feed_late_tau_s), 1.0e-12)
            )
            ** max(float(config.attached_neck_visible_feed_late_exponent), 1.0e-12)
        )
        visible_target_ul *= 1.0 + (late_multiplier - 1.0) * float(np.clip(late_activation, 0.0, 1.0))
    visible_target_m3 = visible_target_ul * 1.0e-9

    contact_xyz = out[contact_ring]
    contact_radius_vertices = np.hypot(contact_xyz[:, 0], contact_xyz[:, 1])
    contact_radial = np.zeros_like(contact_xyz)
    valid = contact_radius_vertices > 1.0e-30
    contact_radial[valid, 0] = contact_xyz[valid, 0] / contact_radius_vertices[valid]
    contact_radial[valid, 1] = contact_xyz[valid, 1] / contact_radius_vertices[valid]
    cox_speed = float(np.mean(np.sum(vel[contact_ring] * contact_radial, axis=1))) if contact_ring.size else 0.0
    if float(t_s) > 0.0:
        cox_speed = math.copysign(
            max(abs(cox_speed), abs(contact_radius) / max(float(t_s), 1.0e-12)),
            cox_speed if cox_speed != 0.0 else 1.0,
        )
    min_width_m = max(float(config.attached_neck_boundary_layer_min_width_um), 0.0) * 1.0e-6
    max_width_m = (
        max(float(config.attached_neck_boundary_layer_max_width_capillary_lengths), 1.0e-12)
        * capillary_length_m(config)
    )
    solver_name = str(config.attached_neck_boundary_layer_solver).strip().lower()

    if bool(config.attached_neck_adaptive_rings_enabled):
        log_factor = math.log(
            max(
                float(config.contact_line_cox_macro_length_m)
                / max(float(config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
        capillary_number = (
            max(float(config.viscosity_pa_s), 0.0)
            * abs(cox_speed)
            / max(float(config.surface_tension_n_m), 1.0e-30)
        )
        theta_for_grid = float(
            np.cbrt(
                max(
                    math.radians(float(config.contact_angle_deg)) ** 3
                    + 9.0 * capillary_number * log_factor,
                    math.radians(float(config.dynamic_contact_angle_min_deg)) ** 3,
                )
            )
        )
        theta_for_grid = float(
            np.clip(
                theta_for_grid,
                math.radians(float(config.dynamic_contact_angle_min_deg)),
                math.radians(float(config.dynamic_contact_angle_max_deg)),
            )
        )
        slope_for_grid = max(math.tan(max(theta_for_grid, 1.0e-9)), 1.0e-9)
        width_est = min_width_m
        if visible_target_m3 > 1.0e-30:
            amp_est = math.sqrt(
                max(
                    visible_target_m3 * slope_for_grid / max(2.0 * math.pi * max(rim, 1.0e-12), 1.0e-30),
                    0.0,
                )
            )
            width_est = float(np.clip(amp_est / slope_for_grid, max(min_width_m, 1.0e-9), max_width_m))
        substrate_r = float(config.substrate_radius_mm) * 1.0e-3
        adaptive_window_m = float(config.attached_neck_adaptive_width_multiplier) * width_est
        adaptive_window_m = float(
            np.clip(
                adaptive_window_m,
                max(float(config.attached_neck_adaptive_min_window_um) * 1.0e-6, min_width_m),
                max(float(config.attached_neck_adaptive_max_window_mm) * 1.0e-3, min_width_m),
            )
        )
        adaptive_window_m = min(adaptive_window_m, max(substrate_r - rim, min_width_m))
        n_total = int(len(film_ring_ids))
        if n_total >= 6 and adaptive_window_m > 1.0e-12:
            n_neck = int(round(n_total * float(config.attached_neck_adaptive_rings_fraction)))
            n_neck = int(np.clip(n_neck, 4, max(n_total - 2, 4)))
            n_outer = n_total - n_neck
            exponent = max(float(config.attached_neck_adaptive_spacing_exponent), 1.0e-6)
            eta_neck = (np.arange(1, n_neck + 1, dtype=float) / float(n_neck)) ** exponent
            neck_r = rim + adaptive_window_m * eta_neck
            if n_outer > 0 and neck_r[-1] < substrate_r:
                eta_outer = np.arange(1, n_outer + 1, dtype=float) / float(n_outer)
                outer_r = neck_r[-1] + (substrate_r - neck_r[-1]) * eta_outer
                r_arr = np.concatenate((neck_r, outer_r))
            else:
                r_arr = neck_r
            r_arr[-1] = substrate_r
            initial_h = initial_profile_m(config, r_arr)

    if bool(config.attached_neck_boundary_layer_solve_rim_height) and solver_name not in {
        "coupled",
        "yl",
        "young-laplace",
        "young_laplace",
        "cox-young-laplace-lubrication",
    }:
        log_factor = math.log(
            max(
                float(config.contact_line_cox_macro_length_m)
                / max(float(config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
        capillary_number = (
            max(float(config.viscosity_pa_s), 0.0)
            * abs(cox_speed)
            / max(float(config.surface_tension_n_m), 1.0e-30)
        )
        theta_dyn = float(
            np.cbrt(
                max(
                    math.radians(float(config.contact_angle_deg)) ** 3
                    + 9.0 * capillary_number * log_factor,
                    math.radians(float(config.dynamic_contact_angle_min_deg)) ** 3,
                )
            )
        )
        theta_dyn = float(
            np.clip(
                theta_dyn,
                math.radians(float(config.dynamic_contact_angle_min_deg)),
                math.radians(float(config.dynamic_contact_angle_max_deg)),
            )
        )
        slope = max(math.tan(max(theta_dyn, 1.0e-9)), 1.0e-9)
        x = np.maximum(r_arr - rim, 0.0)
        min_h = attached_neck_floor_m(config, t_s)
        max_amplitude = max(float(config.initial_film_thickness_um) * 1.0e-6 - min_h, 0.0)

        def visible_missing_for_amplitude(amplitude_m: float) -> float:
            width_m = float(np.clip(amplitude_m / slope, max(min_width_m, 1.0e-9), max_width_m))
            trial = np.maximum(initial_h - amplitude_m * np.exp(-x / max(width_m, 1.0e-30)), min_h)
            integrand = 2.0 * math.pi * r_arr * np.maximum(initial_h - trial, 0.0)
            if integrand.size < 2:
                return 0.0
            return float(np.trapezoid(integrand, r_arr))

        if visible_target_m3 <= 1.0e-30:
            amplitude = 0.0
        elif visible_missing_for_amplitude(max_amplitude) <= visible_target_m3:
            amplitude = max_amplitude
        else:
            lo = 0.0
            hi = max_amplitude
            for _ in range(56):
                mid = 0.5 * (lo + hi)
                if visible_missing_for_amplitude(mid) < visible_target_m3:
                    lo = mid
                else:
                    hi = mid
            amplitude = 0.5 * (lo + hi)
        solved_rim_z = float(config.initial_film_thickness_um) * 1.0e-6 - amplitude
        solved_rim_z = float(np.clip(solved_rim_z, min_h, float(config.initial_film_thickness_um) * 1.0e-6))
        old_rim = out[rim_ring].copy()
        out[rim_ring, 2] = solved_rim_z
        vel[rim_ring] = (out[rim_ring] - old_rim) / max(float(dt_s), 1.0e-30)
        rim_z = solved_rim_z

    if solver_name in {
        "coupled",
        "yl",
        "young-laplace",
        "young_laplace",
        "cox-young-laplace-lubrication",
    }:
        profile = operators["coupled_cox_young_laplace_lubrication_neck_profile"](
            r_m=r_arr,
            initial_h_m=initial_h,
            rim_radius_m=rim,
            target_visible_missing_m3=visible_target_m3,
            initial_film_thickness_m=float(config.initial_film_thickness_um) * 1.0e-6,
            slide_speed_m_s=cox_speed,
            viscosity_pa_s=float(config.viscosity_pa_s),
            surface_tension_n_m=float(config.surface_tension_n_m),
            density_kg_m3=float(config.density_kg_m3),
            gravity_m_s2=float(config.gravity_m_s2),
            theta_eq_rad=math.radians(float(config.contact_angle_deg)),
            macro_length_m=float(config.contact_line_cox_macro_length_m),
            slip_length_m=float(config.contact_line_cox_slip_length_m),
            min_angle_rad=math.radians(float(config.dynamic_contact_angle_min_deg)),
            max_angle_rad=math.radians(float(config.dynamic_contact_angle_max_deg)),
            min_width_m=min_width_m,
            max_width_m=max_width_m,
            min_z_m=attached_neck_floor_m(config, t_s),
            bvp_nodes=int(config.attached_neck_boundary_layer_bvp_nodes),
        )
    else:
        profile = operators["cox_lubrication_neck_profile"](
            r_m=r_arr,
            initial_h_m=initial_h,
            rim_radius_m=rim,
            rim_z_m=rim_z,
            target_visible_missing_m3=visible_target_m3,
            initial_film_thickness_m=float(config.initial_film_thickness_um) * 1.0e-6,
            capillary_length_m=capillary_length_m(config),
            slide_speed_m_s=cox_speed,
            viscosity_pa_s=float(config.viscosity_pa_s),
            surface_tension_n_m=float(config.surface_tension_n_m),
            theta_eq_rad=math.radians(float(config.contact_angle_deg)),
            macro_length_m=float(config.contact_line_cox_macro_length_m),
            slip_length_m=float(config.contact_line_cox_slip_length_m),
            min_angle_rad=math.radians(float(config.dynamic_contact_angle_min_deg)),
            max_angle_rad=math.radians(float(config.dynamic_contact_angle_max_deg)),
            min_width_m=min_width_m,
            max_width_m=max_width_m,
            min_z_m=attached_neck_floor_m(config, t_s),
        )
    target_h = np.asarray(profile["z_m"], dtype=float)
    relaxation = float(np.clip(config.attached_neck_boundary_layer_relaxation, 0.0, 1.0))
    if relaxation <= 0.0:
        return out, vel

    update_mode = str(config.attached_neck_boundary_layer_update_mode).strip().lower()
    if update_mode in {"residual", "residual-mobility", "residual_mobility", "mobility"}:
        old = out.copy()
        current_r = np.zeros(len(film_ring_ids), dtype=float)
        current_h = np.zeros(len(film_ring_ids), dtype=float)
        for local_i, ring_i in enumerate(film_ring_ids):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            xyz = out[ring]
            current_r[local_i] = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
            current_h[local_i] = float(np.mean(xyz[:, 2]))

        h0 = float(config.initial_film_thickness_um) * 1.0e-6
        min_h = attached_neck_floor_m(config, t_s)
        max_h = max(float(config.max_height_um) * 1.0e-6, h0)
        residual_time = max(float(config.attached_neck_residual_time_s), float(dt_s), 1.0e-12)
        max_vertical_speed = max(float(config.attached_neck_residual_max_vertical_speed_um_s), 0.0) * 1.0e-6
        max_radial_speed = max(float(config.attached_neck_residual_max_radial_speed_um_s), 0.0) * 1.0e-6

        def young_laplace_pressure(radius_m: np.ndarray, height_m: np.ndarray) -> np.ndarray:
            if radius_m.size < 3:
                return np.zeros_like(height_m)
            rr = np.maximum(np.asarray(radius_m, dtype=float), 1.0e-12)
            hh = np.asarray(height_m, dtype=float)
            dhdr = np.gradient(hh, rr, edge_order=1)
            d2hdr2 = np.gradient(dhdr, rr, edge_order=1)
            curvature = d2hdr2 + dhdr / rr
            return float(config.density_kg_m3) * float(config.gravity_m_s2) * hh - float(config.surface_tension_n_m) * curvature

        h_factor_local = np.clip(
            current_h / max(h0, 1.0e-30),
            float(config.lubrication_mobility_floor)
            ** (1.0 / max(float(config.lubrication_mobility_exponent), 1.0e-12)),
            1.0,
        ) ** float(config.lubrication_mobility_exponent)
        pressure_velocity = np.zeros_like(current_h)
        drainage_velocity = np.zeros_like(current_h)
        if len(film_ring_ids) >= 3:
            p_current = young_laplace_pressure(np.asarray(r_arr, dtype=float), current_h)
            p_target = young_laplace_pressure(np.asarray(r_arr, dtype=float), target_h)
            p_residual = p_target - p_current
            pressure_velocity = (
                film_thickness_mobility_scale(config)
                * float(config.vertical_mobility_m_per_s_pa)
                * h_factor_local
                * p_residual
            )

            dpdr = np.gradient(p_current, np.asarray(r_arr, dtype=float), edge_order=1)
            conductance = np.maximum(current_h, min_h) ** 3 / max(3.0 * float(config.viscosity_pa_s), 1.0e-30)
            flux = -conductance * dpdr
            r_flux = np.asarray(r_arr, dtype=float) * flux
            drainage_velocity = -np.gradient(r_flux, np.asarray(r_arr, dtype=float), edge_order=1) / np.maximum(
                np.asarray(r_arr, dtype=float),
                1.0e-12,
            )
            drainage_velocity = np.nan_to_num(drainage_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        vertical_velocity = (
            (target_h - current_h) / residual_time
            + float(config.attached_neck_residual_pressure_weight) * pressure_velocity
            + float(config.attached_neck_residual_flux_weight) * drainage_velocity
        )

        predicted_h = np.clip(current_h + float(dt_s) * relaxation * vertical_velocity, min_h, max_h)
        if bool(config.attached_neck_residual_volume_lagrange_enabled) and len(film_ring_ids) >= 3:
            target_missing = float(visible_target_m3)
            predicted_missing = float(
                np.trapezoid(
                    2.0 * math.pi * np.asarray(r_arr, dtype=float) * np.maximum(initial_h - predicted_h, 0.0),
                    np.asarray(r_arr, dtype=float),
                )
            )
            volume_residual = target_missing - predicted_missing
            shape_weight = np.maximum(initial_h - target_h, 0.0)
            if float(np.max(shape_weight, initial=0.0)) <= 1.0e-14:
                shape_weight = np.maximum(initial_h - predicted_h, 0.0)
            denom = float(
                np.trapezoid(
                    2.0 * math.pi * np.asarray(r_arr, dtype=float) * shape_weight,
                    np.asarray(r_arr, dtype=float),
                )
            )
            if abs(denom) > 1.0e-30 and math.isfinite(volume_residual):
                corrected_h = predicted_h - (
                    float(config.attached_neck_residual_volume_lagrange_weight)
                    * volume_residual
                    * shape_weight
                    / denom
                )
                corrected_h = np.clip(corrected_h, min_h, max_h)
                vertical_velocity += (corrected_h - predicted_h) / max(float(dt_s), 1.0e-30)

        if max_vertical_speed > 0.0:
            vertical_velocity = np.clip(vertical_velocity, -max_vertical_speed, max_vertical_speed)
        radial_velocity = (np.asarray(r_arr, dtype=float) - current_r) / residual_time
        if max_radial_speed > 0.0:
            radial_velocity = np.clip(radial_velocity, -max_radial_speed, max_radial_speed)

        if "rim_z_m" in profile:
            target_rim_z = float(profile["rim_z_m"])
            if math.isfinite(target_rim_z):
                rim_old_z = float(np.mean(out[rim_ring, 2]))
                rim_speed_z = (target_rim_z - rim_old_z) / residual_time
                if max_vertical_speed > 0.0:
                    rim_speed_z = float(np.clip(rim_speed_z, -max_vertical_speed, max_vertical_speed))
                out[rim_ring, 2] = np.clip(
                    out[rim_ring, 2] + float(dt_s) * relaxation * rim_speed_z,
                    min_h,
                    max_h,
                )

        for local_i, ring_i in enumerate(film_ring_ids):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            rr = np.maximum(np.hypot(out[ring, 0], out[ring, 1]), 1.0e-30)
            radial_dir_x = out[ring, 0] / rr
            radial_dir_y = out[ring, 1] / rr
            out[ring, 0] = out[ring, 0] + float(dt_s) * relaxation * radial_velocity[local_i] * radial_dir_x
            out[ring, 1] = out[ring, 1] + float(dt_s) * relaxation * radial_velocity[local_i] * radial_dir_y
            out[ring, 2] = np.clip(
                out[ring, 2] + float(dt_s) * relaxation * vertical_velocity[local_i],
                min_h,
                max_h,
            )
        changed = np.zeros(out.shape[0], dtype=bool)
        for ring_i in film_ring_ids:
            changed[np.asarray(rings[int(ring_i)], dtype=int)] = True
        changed[rim_ring] = True
        vel[changed] = (out[changed] - old[changed]) / max(float(dt_s), 1.0e-30)
        out, vel = axisymmetrize(out, vel, rings)
        return out, vel

    old = out.copy()
    if "rim_z_m" in profile:
        target_rim_z = float(profile["rim_z_m"])
        if math.isfinite(target_rim_z):
            out[rim_ring, 2] = out[rim_ring, 2] + relaxation * (target_rim_z - out[rim_ring, 2])
    for local_i, ring_i in enumerate(film_ring_ids):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        theta_ring = np.arctan2(out[ring, 1], out[ring, 0])
        target_radius = float(r_arr[local_i])
        out[ring, 0] = out[ring, 0] + relaxation * (target_radius * np.cos(theta_ring) - out[ring, 0])
        out[ring, 1] = out[ring, 1] + relaxation * (target_radius * np.sin(theta_ring) - out[ring, 1])
        out[ring, 2] = out[ring, 2] + relaxation * (float(target_h[local_i]) - out[ring, 2])
    vel[:] = (out - old) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def apply_attached_compact_neck_profile_constraint(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    initial_missing_ul: float,
    target_missing_ul: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain the attached bridge/film neck with a compact capillary dimple.

    The previous visible-feed closure used an exponential annulus, which
    spreads the dimple over too much of the film.  This operator keeps the same
    saved ddgclib mesh state but redistributes the bridge-fed missing volume
    into a compact near-rim profile.  The width is solved from volume balance
    using the current rim radius and neck height; no experimental coordinates
    are used.
    """

    if not bool(config.attached_compact_neck_profile_enabled):
        return points, velocities
    start_time = float(config.attached_compact_neck_activation_time_s)
    if float(t_s) < start_time:
        return points, velocities
    activation = 1.0 - math.exp(
        -(
            (float(t_s) - start_time)
            / max(float(config.attached_compact_neck_activation_ramp_s), 1.0e-12)
        )
    )
    activation = float(np.clip(activation, 0.0, 1.0))

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    bridge_mask, film_mask = attached_vertex_region_masks(out.shape[0], rings, ring_region)
    geom = attached_ring_geometry(out, rings, ring_region)
    rim = float(geom["rim_radius_m"])
    rim_ring = np.asarray(geom["rim_ring"], dtype=int)
    contact_ring = np.asarray(geom["contact_ring"], dtype=int)
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    min_h = attached_neck_floor_m(config, t_s)
    max_h = max(float(config.max_height_um) * 1.0e-6, h0)
    neck_z_current = float(np.mean(out[rim_ring, 2]))
    neck_z = float(
        (1.0 - float(config.attached_compact_neck_rim_height_relaxation)) * h0
        + float(config.attached_compact_neck_rim_height_relaxation) * neck_z_current
    )
    neck_z = float(np.clip(neck_z, min_h, max_h))
    amplitude = max(h0 - neck_z, 0.0)
    if amplitude <= 1.0e-12:
        return out, vel

    visible_target_ul, visible_fraction, _corrected_pool_ul = attached_visible_feed_target_ul(
        config=config,
        rim_radius_m=rim,
        target_missing_ul=target_missing_ul,
        contact_radius_m=float(geom["contact_radius_m"]),
    )
    visible_target_ul *= max(float(config.attached_compact_neck_visible_volume_multiplier), 0.0) * activation
    if visible_target_ul <= 1.0e-10:
        return out, vel

    capillary_length = capillary_length_m(config)
    inner_scale = math.sqrt(max(h0 * capillary_length, 1.0e-30))
    annular_scale_ul = (
        2.0
        * math.pi
        * max(rim, 1.0e-12)
        * h0
        * capillary_length
        * 1.0e9
    )
    progress = max(float(target_missing_ul), 0.0) / max(float(annular_scale_ul), 1.0e-30)
    power = max(float(config.attached_compact_neck_recovery_power), 1.0e-12)
    min_width = max(
        float(config.attached_compact_neck_min_width_inner_scale) * inner_scale,
        1.0e-8,
    )
    max_width = max(
        float(config.attached_compact_neck_max_width_capillary_lengths) * capillary_length,
        min_width,
    )
    r = np.hypot(out[:, 0], out[:, 1])
    outer_m = float(config.substrate_radius_mm) * 1.0e-3
    initial_h = initial_profile_m(config, r)

    def apply_width(width_m: float, source: np.ndarray) -> np.ndarray:
        trial = np.array(source, copy=True)
        active = film_mask & (r >= rim) & (r <= min(outer_m * 0.999, rim + float(width_m)))
        if np.any(active):
            xi = np.clip((r[active] - rim) / max(float(width_m), 1.0e-12), 0.0, 1.0)
            shape = (1.0 - xi) ** power
            target_z = initial_h[active] - amplitude * shape
            trial[active, 2] = np.clip(target_z, min_h, max_h)
        trial[rim_ring, 2] = neck_z
        return trial

    def missing_for_width(width_m: float) -> float:
        trial = apply_width(width_m, out)
        return max(
            attached_missing_outer_film_volume_ul(trial, rings, ring_region, config) - float(initial_missing_ul),
            0.0,
        )

    missing_min = missing_for_width(min_width)
    missing_max = missing_for_width(max_width)
    if missing_min >= visible_target_ul:
        recovery_width = min_width
    elif missing_max <= visible_target_ul:
        recovery_width = max_width
    else:
        lo = min_width
        hi = max_width
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if missing_for_width(mid) < visible_target_ul:
                lo = mid
            else:
                hi = mid
        recovery_width = 0.5 * (lo + hi)

    old = out.copy()
    out = apply_width(recovery_width, out)

    if bool(config.attached_compact_neck_bridge_branch_enabled):
        decay = max(float(config.attached_compact_neck_left_decay_progress), 1.0e-12)
        if float(config.attached_compact_neck_left_time_decay_s) > 0.0:
            decay_factor = math.exp(
                -(
                    max(float(t_s), 0.0)
                    / max(float(config.attached_compact_neck_left_time_decay_s), 1.0e-12)
                )
                ** max(float(config.attached_compact_neck_left_time_decay_exponent), 1.0e-12)
            )
        else:
            decay_factor = math.exp(-((progress / decay) ** 2.0))
        left_scale = float(config.attached_compact_neck_left_min_width_inner_scale) + (
            float(config.attached_compact_neck_left_width_inner_scale)
            - float(config.attached_compact_neck_left_min_width_inner_scale)
        ) * decay_factor
        left_width = max(left_scale * inner_scale, 1.0e-8)
        contact_radius = float(geom["contact_radius_m"])
        contact_z = float(np.mean(out[contact_ring, 2]))
        branch = bridge_mask & (r <= rim) & (r > contact_radius + 1.0e-12)
        if np.any(branch):
            span = max(rim - contact_radius, left_width, 1.0e-12)
            s = np.clip((r[branch] - contact_radius) / span, 0.0, 1.0)
            profile = str(config.attached_compact_neck_bridge_profile).strip().lower()
            if profile == "hermite":
                slope_factor = float(config.attached_compact_neck_bridge_contact_slope_factor)
                average_slope = (neck_z - contact_z) / span
                contact_slope = slope_factor * average_slope
                neck_slope = 0.0
                h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
                h10 = s**3 - 2.0 * s**2 + s
                h01 = -2.0 * s**3 + 3.0 * s**2
                h11 = s**3 - s**2
                target_z = (
                    h00 * contact_z
                    + h10 * span * contact_slope
                    + h01 * neck_z
                    + h11 * span * neck_slope
                )
            else:
                exponent = max(float(config.attached_compact_neck_bridge_shape_exponent), 1.0e-12)
                target_z = neck_z + (contact_z - neck_z) * (1.0 - s) ** exponent
            sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
            sphere_z = sphere_lower_z_m(config, np.minimum(r[branch], sphere_radius * 0.999999))
            out[branch, 2] = np.clip(target_z, min_h, np.asarray(sphere_z, dtype=float) - 1.0e-8)

    vel[:] = (out - old) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    # Restore the physical contact boundary after the profile projection.
    contact_r = np.hypot(out[contact_ring, 0], out[contact_ring, 1])
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    out[contact_ring, 2] = sphere_lower_z_m(config, np.minimum(contact_r, sphere_radius * 0.999999))
    vel[contact_ring, 2] = 0.0
    return out, vel


def apply_attached_outer_deficit_spreading_operator(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    target_missing_ul: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively spread the outer-film height deficit over capillary length.

    The Cox/Young-Laplace neck solve can leave too much missing film volume in a
    narrow ring next to the bridge.  This operator applies a ring-space
    capillary diffusion step to the *deficit* ``h0 - h`` and restores the
    requested volume, so the mesh stores a broader film depression without
    moving curves in the renderer.
    """

    if not bool(config.attached_outer_deficit_spreading_enabled):
        return points, velocities
    start_s = float(config.attached_outer_deficit_spreading_start_s)
    if float(t_s) < start_s:
        return points, velocities
    activation = 1.0 - math.exp(
        -(
            (float(t_s) - start_s)
            / max(float(config.attached_outer_deficit_spreading_ramp_s), 1.0e-12)
        )
    )
    activation = float(np.clip(activation, 0.0, 1.0))
    if activation <= 1.0e-12:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    film_indices = np.where(ring_region_arr == 1)[0]
    if film_indices.size < 4:
        return out, vel

    geom = attached_ring_geometry(out, rings, ring_region)
    rim = float(geom["rim_radius_m"])
    inner_guard_m = max(float(config.attached_outer_deficit_spreading_inner_guard_mm), 0.0) * 1.0e-3
    inner_start_m = rim + inner_guard_m
    width_m = attached_outer_deficit_spreading_width_m(config, t_s)
    outer_limit_m = min(float(config.substrate_radius_mm) * 1.0e-3, rim + width_m)

    ring_ids: list[int] = []
    ring_r: list[float] = []
    ring_h: list[float] = []
    for ring_i in film_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        xyz = out[ring]
        r_mean = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        if r_mean < inner_start_m - 1.0e-12 or r_mean > outer_limit_m + 1.0e-12:
            continue
        ring_ids.append(int(ring_i))
        ring_r.append(r_mean)
        ring_h.append(float(np.mean(xyz[:, 2])))
    if len(ring_ids) < 4:
        return out, vel

    order = np.argsort(np.asarray(ring_r, dtype=float))
    ring_ids = [ring_ids[int(i)] for i in order]
    r_arr = np.asarray(ring_r, dtype=float)[order]
    h_arr = np.asarray(ring_h, dtype=float)[order]
    initial_h = initial_profile_m(config, r_arr)
    lower_profile = attached_outer_deficit_lower_profile_m(config, r_arr, rim, t_s, initial_h)
    max_deficit = np.maximum(initial_h - lower_profile, 0.0)
    deficit = apply_attached_outer_deficit_soft_repulsion(
        config,
        r_arr,
        rim,
        initial_h,
        lower_profile,
        initial_h - h_arr,
    )
    if np.max(deficit) <= 1.0e-12:
        return out, vel

    edges = np.empty(r_arr.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (r_arr[:-1] + r_arr[1:])
    edges[0] = max(inner_start_m, r_arr[0] - 0.5 * (r_arr[1] - r_arr[0]))
    edges[-1] = min(outer_limit_m, r_arr[-1] + 0.5 * (r_arr[-1] - r_arr[-2]))
    edges = np.maximum.accumulate(edges)
    area = math.pi * np.maximum(edges[1:] ** 2 - edges[:-1] ** 2, 0.0)
    current_volume = float(np.sum(area * deficit))
    if current_volume <= 1.0e-18:
        return out, vel

    smoothed = deficit.copy()
    blend = float(np.clip(config.attached_outer_deficit_spreading_blend, 0.0, 1.0))
    passes = max(int(config.attached_outer_deficit_spreading_passes), 0)
    for _ in range(passes):
        old = smoothed.copy()
        smoothed[1:-1] = (1.0 - blend) * old[1:-1] + 0.5 * blend * (old[:-2] + old[2:])
        smoothed[0] = (1.0 - blend) * old[0] + blend * old[1]
        smoothed[-1] = (1.0 - blend) * old[-1]
        smoothed = apply_attached_outer_deficit_soft_repulsion(
            config,
            r_arr,
            rim,
            initial_h,
            lower_profile,
            smoothed,
        )

    multiplier = 1.0 + activation * (
        max(float(config.attached_outer_deficit_spreading_volume_multiplier), 0.0) - 1.0
    )
    target_volume = min(
        current_volume * multiplier,
        max(float(target_missing_ul), 0.0) * 1.0e-9,
        float(np.sum(area * max_deficit)),
    )
    if target_volume <= 1.0e-18:
        return out, vel

    base_volume = float(np.sum(area * smoothed))
    if base_volume <= 1.0e-18:
        return out, vel
    lo = 0.0
    hi = max(target_volume / base_volume, 1.0)
    def softened_scaled_deficit(scale: float, source: np.ndarray) -> np.ndarray:
        return apply_attached_outer_deficit_soft_repulsion(
            config,
            r_arr,
            rim,
            initial_h,
            lower_profile,
            source * float(scale),
        )

    for _ in range(64):
        trial_volume = float(np.sum(area * softened_scaled_deficit(hi, smoothed)))
        if trial_volume >= target_volume or hi > 1.0e6:
            break
        hi *= 2.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        trial_volume = float(np.sum(area * softened_scaled_deficit(mid, smoothed)))
        if trial_volume < target_volume:
            lo = mid
        else:
            hi = mid
    target_deficit = softened_scaled_deficit(hi, smoothed)

    old_points = out.copy()
    target_h = initial_h - target_deficit
    if bool(config.attached_outer_deficit_spreading_monotone_recovery):
        # The post-neck outer film should recover as a single trough toward
        # the undisturbed film.  Ring-space diffusion can otherwise leave a
        # small local bump/drop at the guard boundary, which is not a physical
        # extra neck and becomes very visible in Fig. 1(c)-style profiles.
        monotone_h = np.minimum.accumulate(target_h[::-1])[::-1]
        monotone_h = np.clip(monotone_h, lower_profile, initial_h)
        monotone_deficit = apply_attached_outer_deficit_soft_repulsion(
            config,
            r_arr,
            rim,
            initial_h,
            lower_profile,
            initial_h - monotone_h,
        )
        monotone_volume = float(np.sum(area * monotone_deficit))
        if monotone_volume > 1.0e-18:
            lo = 0.0
            hi = max(target_volume / monotone_volume, 1.0)
            for _ in range(64):
                trial_deficit = softened_scaled_deficit(hi, monotone_deficit)
                trial_volume = float(np.sum(area * trial_deficit))
                if trial_volume >= target_volume or hi > 1.0e6:
                    break
                hi *= 2.0
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                trial_deficit = softened_scaled_deficit(mid, monotone_deficit)
                trial_volume = float(np.sum(area * trial_deficit))
                if trial_volume < target_volume:
                    lo = mid
                else:
                    hi = mid
            target_deficit = softened_scaled_deficit(hi, monotone_deficit)
            target_h = initial_h - target_deficit
    for local_i, ring_i in enumerate(ring_ids):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        out[ring, 2] = float(target_h[local_i])
    vel[:] = (out - old_points) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def repair_attached_outer_film_monotone_recovery(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove post-rim extra troughs from the stored outer-film free surface.

    This is a mesh-side consistency repair, not a plotting filter: after the
    sphere-attached neck reaches its minimum, the outer film should recover
    once toward the undisturbed film.  If later operators introduce a small
    one-ring bump/drop at the rim transition, lower the bump and compensate the
    volume by raising farther-out rings while keeping the rim fixed and all
    film rings within their physical bounds.
    """

    if not bool(config.attached_outer_deficit_spreading_monotone_recovery):
        return points, velocities
    start_s = float(config.attached_outer_deficit_spreading_start_s)
    if float(t_s) < start_s:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    geom = attached_ring_geometry(out, rings, ring_region)
    rim_index = int(geom["rim_index"])
    rim_radius_m = float(geom["rim_radius_m"])
    rim_anchor_z_m = float(geom["rim_z_m"])
    if bool(config.attached_outer_deficit_soft_lower_enabled):
        # The attached bridge throat can become a microscopic liquid layer near
        # the substrate, but the visible outer oil-film recovery in Fig. 1(c)
        # should not be anchored to that hidden bridge-interior height.  Use
        # the same soft residual-film scale as the outer-film repulsion as a
        # virtual recovery anchor; this leaves the bridge mesh unchanged and
        # only prevents the film repair pass from dragging the observable film
        # below the residual-film scale at the rim.
        rim_anchor_z_m = max(
            rim_anchor_z_m,
            max(float(config.attached_outer_deficit_soft_lower_um), 0.0) * 1.0e-6,
        )
    repair_width_m = attached_outer_deficit_spreading_width_m(config, t_s)
    if bool(config.attached_neck_adaptive_rings_enabled):
        repair_width_m = max(
            repair_width_m,
            max(float(config.attached_neck_adaptive_max_window_mm), 0.0) * 1.0e-3,
        )
    repair_outer_limit_m = min(
        float(config.substrate_radius_mm) * 1.0e-3,
        rim_radius_m + repair_width_m,
    )

    ring_ids: list[int] = [rim_index]
    ring_r: list[float] = [rim_radius_m]
    ring_h: list[float] = [rim_anchor_z_m]
    for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
        if int(ring_region_arr[min(ring_i, len(ring_region_arr) - 1)]) != 1:
            continue
        ring_arr = np.asarray(ring, dtype=int)
        xyz = out[ring_arr]
        r_mean = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        if r_mean <= rim_radius_m + 1.0e-9:
            continue
        if r_mean > repair_outer_limit_m + 1.0e-12:
            continue
        ring_ids.append(int(ring_i))
        ring_r.append(r_mean)
        ring_h.append(float(np.mean(xyz[:, 2])))
    if len(ring_ids) < 4:
        return points, velocities

    order = np.argsort(np.asarray(ring_r, dtype=float))
    ring_ids = [ring_ids[int(i)] for i in order]
    r_arr = np.asarray(ring_r, dtype=float)[order]
    h_arr = np.asarray(ring_h, dtype=float)[order]
    raw_h_arr = h_arr.copy()
    initial_h = initial_profile_m(config, r_arr)
    lower = attached_outer_deficit_lower_profile_m(config, r_arr, rim_radius_m, t_s, initial_h)
    upper = np.maximum(initial_h, lower)
    h_arr = np.clip(h_arr, lower, upper)
    clip_changed = not np.allclose(h_arr, raw_h_arr, rtol=0.0, atol=1.0e-12)

    edges = np.empty(r_arr.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (r_arr[:-1] + r_arr[1:])
    edges[0] = max(0.0, r_arr[0] - 0.5 * (r_arr[1] - r_arr[0]))
    edges[-1] = min(
        float(config.substrate_radius_mm) * 1.0e-3,
        r_arr[-1] + 0.5 * (r_arr[-1] - r_arr[-2]),
    )
    edges = np.maximum.accumulate(edges)
    area = math.pi * np.maximum(edges[1:] ** 2 - edges[:-1] ** 2, 0.0)
    target_missing = float(np.sum(area * np.maximum(initial_h - h_arr, 0.0)))

    repaired = h_arr.copy()
    trough_i = int(np.argmin(h_arr))
    left_branch_replaced = False
    if trough_i > 1:
        start_h = float(h_arr[0])
        trough_h = float(h_arr[trough_i])
        if start_h > trough_h and np.any(h_arr[1:trough_i] > start_h + 0.05e-6):
            s = np.linspace(0.0, 1.0, trough_i + 1, dtype=float)
            smooth = 3.0 * s**2 - 2.0 * s**3
            repaired[: trough_i + 1] = start_h + (trough_h - start_h) * smooth
            left_branch_replaced = True
        else:
            left = _weighted_isotonic_non_decreasing(h_arr[: trough_i + 1][::-1], area[: trough_i + 1][::-1])[::-1]
            left = _smooth_isotonic_plateaus(left[::-1], atol=0.05e-6)[::-1]
            repaired[: trough_i + 1] = left
    if trough_i < repaired.size - 2:
        tail = _weighted_isotonic_non_decreasing(h_arr[trough_i:], area[trough_i:])
        tail = _smooth_isotonic_plateaus(tail, atol=0.05e-6)
        max_dz = max(float(config.attached_outer_film_recovery_max_cell_dz_um), 0.0) * 1.0e-6
        if max_dz > 0.0 and tail.size >= 2:
            for _ in range(3):
                for local_i in range(tail.size - 2, -1, -1):
                    tail[local_i] = max(float(tail[local_i]), float(tail[local_i + 1]) - max_dz)
                tail = np.maximum.accumulate(tail)
        repaired[trough_i:] = tail
    repaired = _smooth_leading_recovery_plateau(repaired, lower, upper, atol=0.05e-6)
    repaired = np.clip(repaired, lower, upper)
    repaired[0] = h_arr[0]
    if not clip_changed and np.allclose(repaired, h_arr, rtol=0.0, atol=1.0e-12):
        return points, velocities

    repaired_missing = float(np.sum(area * np.maximum(initial_h - repaired, 0.0)))
    excess_missing = repaired_missing - target_missing
    if excess_missing > 1.0e-18 and not left_branch_replaced:
        mode = np.zeros(repaired.size, dtype=float)
        if trough_i + 1 < repaired.size:
            mode[trough_i + 1 :] = np.linspace(0.0, 1.0, repaired.size - trough_i - 1)
        if not np.any(mode > 1.0e-12):
            mode[1:] = np.linspace(0.0, 1.0, repaired.size - 1)

        def trial_profile(scale: float) -> np.ndarray:
            trial = np.minimum(repaired + scale * mode, upper)
            trial[0] = repaired[0]
            return np.maximum.accumulate(trial)

        active = mode > 1.0e-12
        hi = float(np.max((upper[active] - repaired[active]) / mode[active])) if np.any(active) else 0.0
        if hi > 0.0:
            hi_profile = trial_profile(hi)
            hi_missing = float(np.sum(area * np.maximum(initial_h - hi_profile, 0.0)))
            if hi_missing > target_missing:
                repaired = hi_profile
            else:
                lo = 0.0
                for _ in range(56):
                    mid = 0.5 * (lo + hi)
                    trial = trial_profile(mid)
                    trial_missing = float(np.sum(area * np.maximum(initial_h - trial, 0.0)))
                    if trial_missing > target_missing:
                        lo = mid
                    else:
                        hi = mid
                repaired = trial_profile(hi)

    old_points = out.copy()
    for local_i, ring_i in enumerate(ring_ids):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        out[ring, 2] = float(repaired[local_i])
    vel[:] = (out - old_points) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def regularize_attached_outer_film_sawtooth(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Damp nonphysical adjacent-ring sawtooth on the outer film.

    This is a mesh update, not a plotting filter.  It deliberately excludes the
    sphere/neck/rim neighborhood and the outer boundary, then applies a small
    conservative radial Laplacian to the remaining film rings.  The local
    annular liquid volume is restored after smoothing so the operator removes
    numerical ring chatter without creating or deleting film.
    """

    if not bool(config.attached_outer_film_anti_sawtooth_enabled):
        return points, velocities
    if float(t_s) < float(config.attached_outer_film_anti_sawtooth_start_s):
        return points, velocities

    passes = max(int(config.attached_outer_film_anti_sawtooth_passes), 0)
    alpha = float(np.clip(config.attached_outer_film_anti_sawtooth_alpha, 0.0, 0.5))
    if passes <= 0 or alpha <= 0.0:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    geom = attached_ring_geometry(out, rings, ring_region)
    rim_radius_m = float(geom["rim_radius_m"])
    substrate_radius_m = float(config.substrate_radius_mm) * 1.0e-3
    inner_limit_m = rim_radius_m + max(float(config.attached_outer_film_anti_sawtooth_inner_margin_mm), 0.0) * 1.0e-3
    outer_limit_m = substrate_radius_m - max(float(config.attached_outer_film_anti_sawtooth_outer_margin_mm), 0.0) * 1.0e-3
    if outer_limit_m <= inner_limit_m:
        return points, velocities

    ring_ids: list[int] = []
    ring_r: list[float] = []
    ring_h: list[float] = []
    for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
        if int(ring_region_arr[min(ring_i, len(ring_region_arr) - 1)]) != 1:
            continue
        ring_arr = np.asarray(ring, dtype=int)
        xyz = out[ring_arr]
        r_mean = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
        if r_mean <= inner_limit_m or r_mean >= outer_limit_m:
            continue
        ring_ids.append(int(ring_i))
        ring_r.append(r_mean)
        ring_h.append(float(np.mean(xyz[:, 2])))
    if len(ring_ids) < 5:
        return points, velocities

    order = np.argsort(np.asarray(ring_r, dtype=float))
    ring_ids = [ring_ids[int(i)] for i in order]
    r_arr = np.asarray(ring_r, dtype=float)[order]
    h_arr = np.asarray(ring_h, dtype=float)[order]

    initial_h = initial_profile_m(config, r_arr)
    lower = attached_outer_deficit_lower_profile_m(config, r_arr, rim_radius_m, t_s, initial_h)
    max_height_m = max(float(config.max_height_um), float(config.initial_film_thickness_um)) * 1.0e-6
    upper = np.full_like(lower, max_height_m, dtype=float)
    upper = np.maximum(upper, lower)
    bounded = np.clip(h_arr, lower, upper)
    target_volume = _axisymmetric_ring_profile_volume_m3(r_arr, bounded)

    smoothed = bounded.copy()
    for _ in range(passes):
        trial = smoothed.copy()
        # Nonuniform-grid radial Laplacian.  End rings are held fixed because
        # they couple to the excluded neck and outer-boundary regions.
        for local_i in range(1, smoothed.size - 1):
            dr_l = max(float(r_arr[local_i] - r_arr[local_i - 1]), 1.0e-15)
            dr_r = max(float(r_arr[local_i + 1] - r_arr[local_i]), 1.0e-15)
            left_grad = (smoothed[local_i] - smoothed[local_i - 1]) / dr_l
            right_grad = (smoothed[local_i + 1] - smoothed[local_i]) / dr_r
            correction = 0.5 * (dr_l + dr_r) * (right_grad - left_grad)
            trial[local_i] = smoothed[local_i] + alpha * correction
        smoothed = np.clip(trial, lower, upper)
        smoothed[0] = bounded[0]
        smoothed[-1] = bounded[-1]

    if bool(config.attached_outer_film_anti_sawtooth_preserve_volume):
        smoothed = _restore_axisymmetric_profile_volume(r_arr, smoothed, lower, upper, target_volume)
    smoothed = np.clip(smoothed, lower, upper)
    smoothed[0] = bounded[0]
    smoothed[-1] = bounded[-1]

    if np.allclose(smoothed, h_arr, rtol=0.0, atol=1.0e-12):
        return points, velocities

    old_points = out.copy()
    for local_i, ring_i in enumerate(ring_ids):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        out[ring, 2] = float(smoothed[local_i])
    vel[:] = (out - old_points) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def attached_vertex_region_masks(
    point_count: int,
    rings: np.ndarray,
    ring_region: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bridge_mask = np.zeros(int(point_count), dtype=bool)
    film_mask = np.zeros(int(point_count), dtype=bool)
    for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
        target = bridge_mask if int(ring_region[min(ring_i, len(ring_region) - 1)]) == 0 else film_mask
        target[np.asarray(ring, dtype=int)] = True
    bridge_indices = np.where(np.asarray(ring_region, dtype=int) == 0)[0]
    if bridge_indices.size:
        rim_ring = np.asarray(rings[int(bridge_indices[-1])], dtype=int)
        film_mask[rim_ring] = True
    return bridge_mask, film_mask


def project_attached_missing_volume_limit(
    points: np.ndarray,
    faces: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    t_s: float,
    initial_missing_ul: float = 0.0,
    operators: dict | None = None,
) -> np.ndarray:
    if not bool(config.bridge_feed_limiter_enabled):
        return points
    if bool(config.attached_state_closure_enabled) and not bool(config.attached_closure_limit_missing_volume):
        return points
    start_time = float(config.attached_feed_limiter_start_time_s)
    if not bool(config.attached_state_closure_enabled) and float(t_s) < start_time:
        return points
    missing_ul = max(
        attached_missing_outer_film_volume_ul(points, rings, ring_region, config) - float(initial_missing_ul),
        0.0,
    )
    geom = attached_ring_geometry(points, rings, ring_region)
    pressure_center_m = max(float(geom["rim_radius_m"]), 1.0e-12)
    width_m = max(float(config.bridge_pressure_width_mm) * 1.0e-3, 1.0e-8)
    if bool(config.attached_state_closure_enabled) and operators is not None:
        bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
        bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
        closure = attached_state_closure(config, operators, bridge_volume_ul, bridge_radius_mm)
        pressure_center_m = max(
            pressure_center_m + float(closure["pressure_center_offset_m"]),
            1.0e-12,
        )
        width_m = max(
            width_m,
            float(closure["pressure_width_m"]),
            1.0e-8,
        )
        allowed_missing_ul = min(
            float(closure["allowed_missing_ul"]),
            bridge_feed_capacity_ul(config, pressure_center_m, t_s),
        )
    elif start_time > 0.0:
        elapsed = max(float(t_s) - start_time, 0.0)
        growth = 1.0 - math.exp(
            -(
                elapsed
                / max(float(config.attached_feed_limiter_growth_time_s), 1.0e-12)
            )
            ** float(config.attached_feed_limiter_growth_exponent)
        )
        allowed_missing_ul = float(config.attached_feed_limiter_start_missing_ul) + float(
            config.attached_feed_limiter_capacity_multiplier
        ) * growth * bridge_feed_capacity_ul(config, pressure_center_m, elapsed)
    else:
        allowed_missing_ul = bridge_feed_capacity_ul(config, pressure_center_m, t_s)
    allowed_missing_ul = min(float(allowed_missing_ul), float(config.attached_feed_limiter_max_missing_ul))
    excess_ul = missing_ul - allowed_missing_ul
    if excess_ul <= 0.0:
        return points

    out = np.array(points, copy=True)
    _bridge_mask, film_mask = attached_vertex_region_masks(out.shape[0], rings, ring_region)
    r = np.hypot(out[:, 0], out[:, 1])
    start_m = pressure_center_m + float(config.volume_projection_start_widths) * width_m
    outer_m = float(config.substrate_radius_mm) * 1.0e-3
    weights = np.zeros(out.shape[0], dtype=float)
    active = film_mask & (r > start_m) & (r < outer_m * 0.999)
    weights[active] = 1.0 / (1.0 + np.exp(-(r[active] - start_m) / max(width_m, 1.0e-9)))
    capacity = np.maximum(float(config.max_height_um) * 1.0e-6 - out[:, 2], 0.0)
    weights *= capacity > 0.0
    projected_area = lumped_projected_area_from_faces(out, faces)
    denom = float(np.sum(projected_area * weights))
    if denom <= 1.0e-24:
        return out
    dz = (excess_ul * 1.0e-9) / denom
    # Apply the projection only to active outer-film vertices.  Clipping the
    # whole mesh here would move the sphere contact ring off the spherical wall.
    candidate_z = out[:, 2] + dz * weights
    out[active, 2] = np.minimum(candidate_z[active], float(config.max_height_um) * 1.0e-6)
    return out


def attached_profile_rows_for_validation(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_m, h_m = attached_film_profile_m(points, rings, ring_region)
    return r_m * 1.0e3, h_m * 1.0e6


def attached_full_profile_rows_for_validation(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full attached wall/bridge/film meridian for validation plots.

    The connected mesh stores the free liquid surface from the sphere contact
    ring out to the film edge.  The inner wetted branch from r=0 to the contact
    line is the spherical solid/liquid boundary, so it is reconstructed from the
    same sphere geometry used to generate the mesh instead of clipping the SIM
    curve at the bridge rim.
    """

    points = np.asarray(points, dtype=float)
    rings = np.asarray(rings, dtype=int)
    ring_region = np.asarray(ring_region, dtype=int)
    geom = attached_ring_geometry(points, rings, ring_region)
    contact_radius_m = max(float(geom["contact_radius_m"]), 0.0)
    wall_nodes = max(int(config.full_bridge_mesh_bridge_nodes), 8)
    wall_r = np.linspace(0.0, contact_radius_m, wall_nodes)
    wall_h = np.asarray(sphere_lower_z_m(config, wall_r), dtype=float)
    r_vals = [float(r) for r in wall_r]
    h_vals = [float(h) for h in wall_h]
    for ring in rings:
        xyz = points[np.asarray(ring, dtype=int)]
        r_vals.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        h_vals.append(float(np.mean(xyz[:, 2])))
    order = np.argsort(r_vals)
    r_sorted = np.asarray(r_vals, dtype=float)[order]
    h_sorted = np.asarray(h_vals, dtype=float)[order]
    keep_r: list[float] = []
    keep_h: list[float] = []
    for r_val, h_val in zip(r_sorted, h_sorted):
        if keep_r and abs(float(r_val) - keep_r[-1]) < 1.0e-10:
            keep_h[-1] = 0.5 * (keep_h[-1] + float(h_val))
            continue
        keep_r.append(float(r_val))
        keep_h.append(float(h_val))
    return np.asarray(keep_r, dtype=float) * 1.0e3, np.asarray(keep_h, dtype=float) * 1.0e6


def attached_contact_angle_rad(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
) -> float:
    """Estimate the meridian angle between the bridge surface and sphere."""

    geom = attached_ring_geometry(points, rings, ring_region)
    contact_index = int(geom["contact_index"])
    if contact_index + 1 >= len(rings):
        return math.radians(float(config.contact_angle_deg))
    ring_region_arr = np.asarray(ring_region, dtype=int)
    bridge_indices = np.where(ring_region_arr == 0)[0]
    contact_ring = np.asarray(rings[contact_index], dtype=int)
    contact_xyz = np.asarray(points, dtype=float)[contact_ring]
    r0 = float(np.mean(np.hypot(contact_xyz[:, 0], contact_xyz[:, 1])))
    z0 = float(np.mean(contact_xyz[:, 2]))
    window_m = max(float(config.attached_contact_angle_probe_window_um), 0.0) * 1.0e-6
    liquid_tangent: np.ndarray
    if window_m > 0.0 and bridge_indices.size >= 3:
        candidates: list[tuple[float, float]] = []
        for ring_i in bridge_indices:
            if int(ring_i) <= contact_index:
                continue
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            xyz = np.asarray(points, dtype=float)[ring]
            rr = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
            zz = float(np.mean(xyz[:, 2]))
            if rr <= r0 + 1.0e-12:
                continue
            if rr - r0 <= window_m or len(candidates) < int(config.attached_contact_angle_probe_min_fit_rings):
                candidates.append((rr, zz))
            if rr - r0 >= window_m and len(candidates) >= int(config.attached_contact_angle_probe_min_fit_rings):
                break
            if len(candidates) >= int(config.attached_contact_angle_probe_max_fit_rings):
                break
        if len(candidates) >= 2:
            rr = np.asarray([r0] + [item[0] for item in candidates], dtype=float)
            zz = np.asarray([z0] + [item[1] for item in candidates], dtype=float)
            scale = max(float(rr[-1] - rr[0]), 1.0e-12)
            x = (rr - rr[0]) / scale
            # Fit a local tangent in a fixed metric window; this makes the Cox
            # boundary condition less dependent on the current ring spacing.
            degree = min(2, len(x) - 1)
            coeff = np.polyfit(x, zz, degree)
            slope = float(np.polyder(np.poly1d(coeff))(0.0)) / scale
            liquid_tangent = np.array([1.0, slope], dtype=float)
        else:
            probe_rings = max(int(config.attached_contact_angle_probe_rings), 1)
            probe_index = min(contact_index + probe_rings, len(rings) - 1)
            next_ring = np.asarray(rings[probe_index], dtype=int)
            next_xyz = np.asarray(points, dtype=float)[next_ring]
            r1 = float(np.mean(np.hypot(next_xyz[:, 0], next_xyz[:, 1])))
            z1 = float(np.mean(next_xyz[:, 2]))
            liquid_tangent = np.array([r1 - r0, z1 - z0], dtype=float)
    else:
        probe_rings = max(int(config.attached_contact_angle_probe_rings), 1)
        probe_index = min(contact_index + probe_rings, len(rings) - 1)
        next_ring = np.asarray(rings[probe_index], dtype=int)
        next_xyz = np.asarray(points, dtype=float)[next_ring]
        r1 = float(np.mean(np.hypot(next_xyz[:, 0], next_xyz[:, 1])))
        z1 = float(np.mean(next_xyz[:, 2]))
        liquid_tangent = np.array([r1 - r0, z1 - z0], dtype=float)
    liquid_norm = float(np.linalg.norm(liquid_tangent))
    if liquid_norm <= 1.0e-30:
        return math.radians(float(config.contact_angle_deg))
    liquid_tangent /= liquid_norm
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    axial = math.sqrt(max(sphere_radius * sphere_radius - r0 * r0, 1.0e-30))
    solid_tangent = np.array([1.0, r0 / axial], dtype=float)
    solid_tangent /= max(float(np.linalg.norm(solid_tangent)), 1.0e-30)
    theta = math.acos(float(np.clip(abs(float(np.dot(liquid_tangent, solid_tangent))), -1.0, 1.0)))
    return float(theta)


def cox_contact_line_slide_speed_m_s(
    points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    operators: dict,
) -> float:
    theta_geo = attached_contact_angle_rad(points, rings, ring_region, config)
    theta_eq = math.radians(float(config.contact_angle_deg))
    speed = operators["cox_inverse_contact_line_speed"](
        theta_geo_rad=theta_geo,
        theta_eq_rad=theta_eq,
        viscosity_pa_s=float(config.viscosity_pa_s),
        surface_tension_n_m=float(config.surface_tension_n_m),
        macro_length_m=float(config.contact_line_cox_macro_length_m),
        slip_length_m=float(config.contact_line_cox_slip_length_m),
    )
    limit = float(config.attached_contact_line_max_speed_um_s) * 1.0e-6
    return float(np.clip(speed, -limit, limit))


def enforce_attached_bridge_constraints(
    points: np.ndarray,
    velocities: np.ndarray,
    initial_points: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    target_rim_radius_m: float,
    target_contact_radius_m: float | None,
    operators: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    out, vel = axisymmetrize(out, vel, rings)

    geom_before = attached_ring_geometry(out, rings, ring_region)
    contact_ring = np.asarray(geom_before["contact_ring"], dtype=int)
    rim_ring = np.asarray(geom_before["rim_ring"], dtype=int)
    old_contact_radius = float(geom_before["contact_radius_m"])
    slide_speed = cox_contact_line_slide_speed_m_s(out, rings, ring_region, config, operators)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    activation = 1.0 - math.exp(
        -(
            max(float(t_s), 0.0)
            / max(float(config.cox_contact_line_activation_time_s), 1.0e-12)
        )
        ** float(config.cox_contact_line_activation_exponent)
    )
    new_contact_radius = old_contact_radius + slide_speed * activation * float(dt_s)
    target_contact_radius = None
    target_contact_cap = None
    if target_contact_radius_m is not None and bool(config.attached_contact_line_bridge_radius_coupling_enabled):
        target_contact_radius = float(np.clip(target_contact_radius_m, 1.0e-8, sphere_radius * 0.999))
        if (not bool(config.attached_contact_line_bridge_radius_pull_only_below_target)) or (
            float(new_contact_radius) < target_contact_radius
        ):
            new_contact_radius = operators["relax_radius_toward_target"](
                current_radius_m=new_contact_radius,
                target_radius_m=target_contact_radius,
                dt_s=float(dt_s),
                relaxation_fraction=float(config.attached_contact_line_bridge_radius_relaxation),
                max_speed_m_s=float(config.attached_contact_line_bridge_radius_max_speed_um_s) * 1.0e-6,
            )
        # Cox-Voinov gives the contact-line speed, but the attached bridge mesh
        # must remain compatible with the bridge-volume/rim geometry.  Without
        # this bound, the contact ring can climb far up the sphere while the
        # bridge-film rim stays near the substrate, producing an unphysical
        # nearly cylindrical shell.
        target_contact_cap = target_contact_radius + max(
            float(config.attached_contact_line_bridge_radius_cap_gap_fraction),
            0.0,
        ) * max(float(target_rim_radius_m) - target_contact_radius, 0.0)
        target_contact_cap = float(np.clip(target_contact_cap, target_contact_radius, sphere_radius * 0.999))
        beyond_speed_limit = max(
            float(config.attached_contact_line_bridge_radius_beyond_target_max_speed_um_s),
            0.0,
        ) * 1.0e-6
        if beyond_speed_limit > 0.0 and float(new_contact_radius) > target_contact_radius:
            continuation_cap = max(float(old_contact_radius), target_contact_radius) + beyond_speed_limit * float(dt_s)
            new_contact_radius = min(float(new_contact_radius), continuation_cap)
        new_contact_radius = min(new_contact_radius, target_contact_cap)
    rim_guard = max(2.5e-6, 0.005 * max(float(target_rim_radius_m) - old_contact_radius, 0.0))
    max_contact_from_rim = max(1.0e-8, float(target_rim_radius_m) - rim_guard)
    new_contact_radius = min(float(new_contact_radius), max_contact_from_rim)
    if not bool(config.attached_contact_line_allow_recede):
        new_contact_radius = max(float(new_contact_radius), float(old_contact_radius))
    if target_contact_radius is not None:
        # The no-recede guard is only a contact-line hysteresis model; it must
        # not override the bridge-volume/head compatibility target.  Re-apply
        # the target bound after the guard so long-time runs cannot retain an
        # over-advanced sphere contact line from a transient Cox step.
        cap = float(target_contact_cap) if target_contact_cap is not None else float(target_contact_radius)
        new_contact_radius = min(float(new_contact_radius), cap)
    new_contact_radius = min(float(new_contact_radius), max_contact_from_rim)
    new_contact_radius = float(np.clip(new_contact_radius, 1.0e-8, sphere_radius * 0.999))
    theta = np.arctan2(out[contact_ring, 1], out[contact_ring, 0])
    out[contact_ring, 0] = new_contact_radius * np.cos(theta)
    out[contact_ring, 1] = new_contact_radius * np.sin(theta)
    out[contact_ring, 2] = float(sphere_lower_z_m(config, new_contact_radius))
    radial_velocity = (new_contact_radius - old_contact_radius) / max(float(dt_s), 1.0e-30)
    vel[contact_ring, 0] = radial_velocity * np.cos(theta)
    vel[contact_ring, 1] = radial_velocity * np.sin(theta)
    vel[contact_ring, 2] = radial_velocity * (
        new_contact_radius / math.sqrt(max(sphere_radius * sphere_radius - new_contact_radius * new_contact_radius, 1.0e-30))
    )

    old_rim_radius = float(geom_before["rim_radius_m"])
    target_rim_radius = float(np.clip(target_rim_radius_m, 1.0e-8, float(config.substrate_radius_mm) * 1.0e-3))
    rim_activation = 1.0 - math.exp(
        -(
            max(float(t_s), 0.0)
            / max(float(config.attached_mesh_mobility_startup_time_s), 1.0e-12)
        )
        ** float(config.attached_mesh_mobility_startup_exponent)
    )
    rim_activation = float(config.attached_mesh_mobility_floor) + (
        1.0 - float(config.attached_mesh_mobility_floor)
    ) * rim_activation
    rim_relax = float(np.clip(config.attached_bridge_rim_relaxation_per_step, 0.0, 1.0)) * rim_activation
    new_rim_radius = old_rim_radius + rim_relax * (target_rim_radius - old_rim_radius)
    rim_delta = new_rim_radius - old_rim_radius
    if abs(rim_delta) > 1.0e-14:
        theta_rim = np.arctan2(out[rim_ring, 1], out[rim_ring, 0])
        out[rim_ring, 0] = new_rim_radius * np.cos(theta_rim)
        out[rim_ring, 1] = new_rim_radius * np.sin(theta_rim)
        vel[rim_ring, 0] += (rim_delta / max(float(dt_s), 1.0e-30)) * np.cos(theta_rim)
        vel[rim_ring, 1] += (rim_delta / max(float(dt_s), 1.0e-30)) * np.sin(theta_rim)
        decay_m = max(float(config.attached_bridge_rim_shift_decay_mm) * 1.0e-3, 1.0e-12)
        ring_means = []
        for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
            xyz = out[np.asarray(ring, dtype=int)]
            ring_means.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
            if int(ring_region[min(ring_i, len(ring_region) - 1)]) != 1:
                continue
            ring = np.asarray(ring, dtype=int)
            rr = ring_means[ring_i]
            if rr <= old_rim_radius:
                continue
            shift = rim_delta * math.exp(-max(rr - old_rim_radius, 0.0) / decay_m)
            theta_f = np.arctan2(out[ring, 1], out[ring, 0])
            new_rr = np.clip(rr + shift, new_rim_radius + 1.0e-7, float(config.substrate_radius_mm) * 1.0e-3)
            out[ring, 0] = new_rr * np.cos(theta_f)
            out[ring, 1] = new_rr * np.sin(theta_f)
            vel[ring, 0] += (new_rr - rr) / max(float(dt_s), 1.0e-30) * np.cos(theta_f)
            vel[ring, 1] += (new_rr - rr) / max(float(dt_s), 1.0e-30) * np.sin(theta_f)

    bridge_ring_indices = np.where(np.asarray(ring_region, dtype=int) == 0)[0]
    if bool(config.attached_bridge_shape_constraint_enabled) and bridge_ring_indices.size >= 3:
        contact_z = float(sphere_lower_z_m(config, new_contact_radius))
        rim_z = float(np.mean(out[rim_ring, 2]))
        neck_fraction = max(float(config.attached_bridge_shape_neck_fraction), 0.0)
        height_exponent = max(float(config.attached_bridge_shape_height_exponent), 1.0e-12)
        bridge_gap = new_rim_radius - new_contact_radius
        ring_count = max(int(bridge_ring_indices.size) - 1, 1)
        min_spacing = max(abs(bridge_gap) * 0.015 / float(ring_count), 2.5e-7)
        radius_exponent = 1.0 + 5.0 * neck_fraction
        for local_i, ring_i in enumerate(bridge_ring_indices):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            if int(ring_i) == int(bridge_ring_indices[0]):
                continue
            if int(ring_i) == int(bridge_ring_indices[-1]):
                continue
            s = float(local_i) / max(float(bridge_ring_indices.size - 1), 1.0)
            ease = s * s * (3.0 - 2.0 * s)
            height_ease = s**height_exponent
            if bridge_gap >= 0.0:
                target_radius = new_contact_radius + bridge_gap * (s**radius_exponent)
                lower = new_contact_radius + min_spacing * float(local_i)
                upper = new_rim_radius - min_spacing * float(ring_count - local_i)
                target_radius = float(np.clip(target_radius, lower, max(lower, upper)))
            else:
                target_radius = new_contact_radius + bridge_gap * (s**radius_exponent)
                upper = new_contact_radius - min_spacing * float(local_i)
                lower = new_rim_radius + min_spacing * float(ring_count - local_i)
                target_radius = float(np.clip(target_radius, min(lower, upper), upper))
            target_z = (1.0 - height_ease) * contact_z + height_ease * rim_z
            theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
            target_xy = np.column_stack((target_radius * np.cos(theta_bridge), target_radius * np.sin(theta_bridge)))
            old_xyz = out[ring].copy()
            out[ring, 0] = target_xy[:, 0]
            out[ring, 1] = target_xy[:, 1]
            out[ring, 2] = target_z
            vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)

    outer_ring = np.asarray(rings[-1], dtype=int)

    r = np.hypot(out[:, 0], out[:, 1])
    sphere_z = sphere_lower_z_m(config, np.minimum(r, sphere_radius * 0.999999))
    bridge_mask = np.zeros(out.shape[0], dtype=bool)
    for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
        if int(ring_region[min(ring_i, len(ring_region) - 1)]) == 0:
            bridge_mask[np.asarray(ring, dtype=int)] = True
    free_bridge = bridge_mask.copy()
    free_bridge[contact_ring] = False
    min_bridge_z = attached_neck_floor_m(config, t_s)
    out[free_bridge, 2] = np.clip(out[free_bridge, 2], min_bridge_z, sphere_z[free_bridge] - 1.0e-8)

    film_mask = ~bridge_mask
    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    film_min_z = effective_min_height_m(config, t_s)
    out[film_mask, 2] = np.clip(
        out[film_mask, 2],
        film_min_z,
        max(float(config.max_height_um) * 1.0e-6, h0),
    )
    out[rim_ring, 2] = np.clip(
        np.mean(out[rim_ring, 2]),
        film_min_z,
        max(float(config.max_height_um) * 1.0e-6, h0),
    )
    # The substrate-edge boundary is prescribed by the experimental geometry.
    # Re-apply it after film clipping so the outer radius is not lifted to the
    # numerical minimum film height.
    out[outer_ring] = initial_points[outer_ring]
    vel[outer_ring] = 0.0
    out, vel = axisymmetrize(out, vel, rings)
    # Re-pin the bridge contact ring to the spherical wall after every mesh
    # projection/regularization pass.  This is the physical contact line; if it
    # is left at the old film height the saved mesh shows a false air gap and an
    # L-shaped bridge corner in the meridian view.
    contact_r_after = np.hypot(out[contact_ring, 0], out[contact_ring, 1])
    contact_z_after = sphere_lower_z_m(config, np.minimum(contact_r_after, sphere_radius * 0.999999))
    out[contact_ring, 2] = contact_z_after
    vel[contact_ring, 2] = 0.0
    geom_after = attached_ring_geometry(out, rings, ring_region)
    return out, vel, {
        "cox_contact_ring_radius_mm": float(geom_after["contact_radius_m"]) * 1.0e3,
        "cox_contact_ring_z_mm": float(geom_after["contact_z_m"]) * 1.0e3,
        "cox_slide_speed_mean_um_s": slide_speed * activation * 1.0e6,
        "cox_theta_mean_deg": attached_contact_angle_rad(out, rings, ring_region, config) * 180.0 / math.pi,
        "cox_activation": float(activation),
        "bridge_rim_radius_mm": float(geom_after["rim_radius_m"]) * 1.0e3,
        "bridge_rim_z_um": float(geom_after["rim_z_m"]) * 1.0e6,
    }


def repin_attached_contact_ring_to_sphere(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the spherical-wall contact-line boundary condition last.

    Several film-volume operators work on ring heights after the bridge
    constraint.  This final pass prevents those operators from pulling the
    sphere/liquid contact ring back to the flat film height before snapshots are
    saved.
    """

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    geom_before = attached_ring_geometry(out, rings, ring_region)
    contact_ring = np.asarray(geom_before["contact_ring"], dtype=int)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    contact_r = np.hypot(out[contact_ring, 0], out[contact_ring, 1])
    out[contact_ring, 2] = sphere_lower_z_m(config, np.minimum(contact_r, sphere_radius * 0.999999))
    vel[contact_ring, 2] = 0.0
    out, vel = axisymmetrize(out, vel, rings)
    geom_after = attached_ring_geometry(out, rings, ring_region)
    return out, vel, {
        "cox_contact_ring_radius_mm": float(geom_after["contact_radius_m"]) * 1.0e3,
        "cox_contact_ring_z_mm": float(geom_after["contact_z_m"]) * 1.0e3,
        "cox_slide_speed_mean_um_s": 0.0,
        "cox_theta_mean_deg": attached_contact_angle_rad(out, rings, ring_region, config) * 180.0 / math.pi,
        "cox_activation": 0.0,
        "bridge_rim_radius_mm": float(geom_after["rim_radius_m"]) * 1.0e3,
        "bridge_rim_z_um": float(geom_after["rim_z_m"]) * 1.0e6,
    }


def _axisymmetric_ring_profile_volume_m3(r_m: np.ndarray, z_m: np.ndarray) -> float:
    """Volume under an axisymmetric ring profile."""

    r = np.asarray(r_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if r.size < 2:
        return 0.0
    integrand = 2.0 * math.pi * r * z
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))


def _restore_axisymmetric_profile_volume(
    r_m: np.ndarray,
    z_m: np.ndarray,
    lower_m: np.ndarray,
    upper_m: np.ndarray,
    target_volume_m3: float,
) -> np.ndarray:
    """Nudge internal rings so a smoothed profile keeps the same annular volume."""

    z = np.asarray(z_m, dtype=float).copy()
    if z.size < 4:
        return z

    current = _axisymmetric_ring_profile_volume_m3(r_m, z)
    residual = float(target_volume_m3) - current
    if abs(residual) <= 1.0e-18:
        return z

    s = np.linspace(0.0, 1.0, z.size)
    mode = np.sin(math.pi * s)
    mode[0] = 0.0
    mode[-1] = 0.0
    active = mode > 1.0e-12
    if not np.any(active):
        return z

    if residual > 0.0:
        capacity = np.asarray(upper_m, dtype=float) - z
        direction = 1.0
    else:
        capacity = z - np.asarray(lower_m, dtype=float)
        direction = -1.0
    max_scale = float(np.min(capacity[active] / mode[active]))
    if not np.isfinite(max_scale) or max_scale <= 0.0:
        return z

    lo = 0.0
    hi = max_scale
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        trial = np.clip(z + direction * mid * mode, lower_m, upper_m)
        trial[0] = z[0]
        trial[-1] = z[-1]
        trial_residual = float(target_volume_m3) - _axisymmetric_ring_profile_volume_m3(r_m, trial)
        if residual > 0.0:
            if trial_residual > 0.0:
                lo = mid
            else:
                hi = mid
        else:
            if trial_residual < 0.0:
                lo = mid
            else:
                hi = mid
    corrected = np.clip(z + direction * hi * mode, lower_m, upper_m)
    corrected[0] = z[0]
    corrected[-1] = z[-1]
    return corrected


def _weighted_isotonic_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA projection onto non-decreasing ring-height profiles."""

    y = np.asarray(values, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if y.size <= 1:
        return y.copy()
    blocks: list[dict[str, float | int]] = []
    for i, (value, weight) in enumerate(zip(y, w)):
        block_weight = float(weight) if float(weight) > 0.0 else 1.0
        blocks.append({"start": i, "end": i + 1, "weight": block_weight, "mean": float(value)})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left["weight"]) + float(right["weight"])
            mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / max(total_weight, 1.0e-30)
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total_weight,
                    "mean": float(mean),
                }
            )
    out = np.empty_like(y)
    for block in blocks:
        out[int(block["start"]) : int(block["end"])] = float(block["mean"])
    return out


def _smooth_isotonic_plateaus(values: np.ndarray, atol: float = 1.0e-12) -> np.ndarray:
    """Replace interior PAVA plateaus by monotone capillary-smooth ramps."""

    out = np.asarray(values, dtype=float).copy()
    n = out.size
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n - 1 and abs(float(out[j + 1] - out[i])) <= atol:
            j += 1
        run_len = j - i + 1
        if run_len >= 3 and j + 1 < n:
            s = np.linspace(0.0, 1.0, run_len + 2, dtype=float)[1:-1]
            smooth = 3.0 * s**2 - 2.0 * s**3
            if i == 0 and out[i] < out[j + 1]:
                out[i : j + 1] = out[i] + (out[j + 1] - out[i]) * smooth
                out[i] = values[i]
            elif i > 0 and out[i - 1] < out[i] < out[j + 1]:
                out[i : j + 1] = out[i - 1] + (out[j + 1] - out[i - 1]) * smooth
        i = j + 1
    return np.maximum.accumulate(out)


def _smooth_leading_recovery_plateau(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    atol: float = 1.0e-12,
) -> np.ndarray:
    """Replace a rim-attached shelf by a smooth one-sided capillary recovery."""

    out = np.asarray(values, dtype=float).copy()
    if out.size < 4:
        return np.clip(out, lower, upper)
    j = 0
    while j + 1 < out.size and abs(float(out[j + 1] - out[0])) <= atol:
        j += 1
    if j >= 3 and j + 1 < out.size and out[j + 1] > out[0] + atol:
        s = np.linspace(0.0, 1.0, j + 2, dtype=float)[: j + 1]
        smooth = 3.0 * s**2 - 2.0 * s**3
        out[: j + 1] = out[0] + (out[j + 1] - out[0]) * smooth
        out[0] = values[0]
    out = np.maximum.accumulate(out)
    return np.clip(out, lower, upper)


def regularize_attached_capillary_bridge_ring_profile(
    r_m: np.ndarray,
    z_m: np.ndarray,
    lower_m: np.ndarray,
    upper_m: np.ndarray,
    config: RealMeshEvolutionConfig,
) -> np.ndarray:
    """Remove one-ring free-surface jumps while preserving endpoints and volume."""

    z = np.asarray(z_m, dtype=float).copy()
    if z.size < 4:
        return z

    passes = max(int(config.attached_capillary_bridge_profile_smoothing_passes), 0)
    alpha = float(np.clip(config.attached_capillary_bridge_profile_smoothing_alpha, 0.0, 1.0))
    max_dz = max(float(config.attached_capillary_bridge_profile_max_cell_dz_um), 0.0) * 1.0e-6
    if passes <= 0 and max_dz <= 0.0:
        return z

    lower = np.asarray(lower_m, dtype=float)
    upper = np.asarray(upper_m, dtype=float)
    endpoint0 = float(z[0])
    endpoint1 = float(z[-1])
    target_volume = _axisymmetric_ring_profile_volume_m3(r_m, z)

    def restore_and_clip(values: np.ndarray) -> np.ndarray:
        out = np.clip(values, lower, upper)
        out[0] = endpoint0
        out[-1] = endpoint1
        return out

    def limit_cell_jumps(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        if max_dz <= 0.0:
            return restore_and_clip(out)
        for _ in range(3):
            out[0] = endpoint0
            for i in range(1, out.size - 1):
                out[i] = float(np.clip(out[i], out[i - 1] - max_dz, out[i - 1] + max_dz))
                out[i] = float(np.clip(out[i], lower[i], upper[i]))
            out[-1] = endpoint1
            for i in range(out.size - 2, 0, -1):
                out[i] = float(np.clip(out[i], out[i + 1] - max_dz, out[i + 1] + max_dz))
                out[i] = float(np.clip(out[i], lower[i], upper[i]))
            out = restore_and_clip(out)
        return out

    def remove_local_maxima(values: np.ndarray) -> np.ndarray:
        out = restore_and_clip(values)
        if out.size < 4:
            return out
        throat_i = int(np.argmin(out))
        for i in range(1, throat_i + 1):
            out[i] = min(out[i], out[i - 1])
            out[i] = float(np.clip(out[i], lower[i], upper[i]))
        for i in range(out.size - 2, throat_i - 1, -1):
            out[i] = min(out[i], out[i + 1])
            out[i] = float(np.clip(out[i], lower[i], upper[i]))
        return restore_and_clip(out)

    z = restore_and_clip(z)
    for _ in range(passes):
        old = z.copy()
        z[1:-1] = (1.0 - alpha) * old[1:-1] + 0.5 * alpha * (old[:-2] + old[2:])
        z = restore_and_clip(z)

    z = limit_cell_jumps(z)

    if bool(config.attached_capillary_bridge_profile_volume_correction):
        z = _restore_axisymmetric_profile_volume(r_m, z, lower, upper, target_volume)
        z = restore_and_clip(z)
    if bool(config.attached_capillary_bridge_profile_remove_local_maxima):
        z = remove_local_maxima(z)
        z = limit_cell_jumps(z)
    return z


def apply_final_attached_bridge_shape_constraint(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    bridge_volume_ul: float,
    operators: dict,
    t_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Save the attached bridge as a smooth free surface between wall and film."""

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    bridge_ring_indices = np.where(ring_region_arr == 0)[0]
    if bridge_ring_indices.size < 2:
        return repin_attached_contact_ring_to_sphere(out, vel, rings, ring_region, config)

    geom = attached_ring_geometry(out, rings, ring_region)
    contact_ring = np.asarray(geom["contact_ring"], dtype=int)
    rim_ring = np.asarray(geom["rim_ring"], dtype=int)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3
    contact_radius = float(geom["contact_radius_m"])
    rim_radius = float(geom["rim_radius_m"])
    contact_z = float(sphere_lower_z_m(config, min(contact_radius, sphere_radius * 0.999999)))
    rim_z = float(np.mean(out[rim_ring, 2]))

    if bool(config.attached_capillary_bridge_solver_enabled):
        h0 = float(config.initial_film_thickness_um) * 1.0e-6
        if float(bridge_volume_ul) <= max(float(config.inner_bridge_volume_ul), 0.0) + 1.0e-12:
            bridge_r = np.linspace(contact_radius, rim_radius, bridge_ring_indices.size)
            bridge_z = np.linspace(contact_z, rim_z, bridge_ring_indices.size)
            for local_i, ring_i in enumerate(bridge_ring_indices):
                ring = np.asarray(rings[int(ring_i)], dtype=int)
                theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
                old_xyz = out[ring].copy()
                out[ring, 0] = bridge_r[local_i] * np.cos(theta_bridge)
                out[ring, 1] = bridge_r[local_i] * np.sin(theta_bridge)
                out[ring, 2] = bridge_z[local_i]
                vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)
            out, vel = axisymmetrize(out, vel, rings)
            return repin_attached_contact_ring_to_sphere(out, vel, rings, ring_region, config)
        configured_min = float(config.attached_capillary_bridge_min_height_um) * 1.0e-6
        precursor_min = h0 * max(float(config.attached_capillary_bridge_precursor_fraction), 0.0)
        dynamic_min = attached_neck_floor_m(config, t_s) if bool(config.attached_capillary_bridge_use_dynamic_min_height) else 0.0
        bridge_min_z = max(configured_min, precursor_min, dynamic_min, 0.25e-6)
        rim_slope_m_per_m = None
        if bool(config.attached_capillary_bridge_match_film_slope):
            film_after = np.where(ring_region_arr == 1)[0]
            film_after = film_after[film_after > int(bridge_ring_indices[-1])]
            if film_after.size:
                next_ring = np.asarray(rings[int(film_after[0])], dtype=int)
                next_xyz = out[next_ring]
                next_r = float(np.mean(np.hypot(next_xyz[:, 0], next_xyz[:, 1])))
                next_z = float(np.mean(next_xyz[:, 2]))
                if next_r > rim_radius + 1.0e-12:
                    rim_slope_m_per_m = (next_z - rim_z) / (next_r - rim_radius)
        old_bridge_r: list[float] = []
        old_bridge_z: list[float] = []
        for ring_i in bridge_ring_indices:
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            xyz = out[ring]
            old_bridge_r.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
            old_bridge_z.append(float(np.mean(xyz[:, 2])))
        bridge_profile_model = str(config.attached_capillary_bridge_profile_model).strip().lower()
        sphere_bottom_z_m = (
            float(config.initial_film_thickness_um) * 1.0e-6
            if config.sphere_bottom_z_mm is None
            else float(config.sphere_bottom_z_mm) * 1.0e-3
        )
        if bridge_profile_model in {
            "soft_young_laplace",
            "soft_linear_young_laplace",
            "soft_linearized_young_laplace",
            "soft_repulsive_young_laplace",
            "soft_repulsive_linearized_young_laplace",
        }:
            profile = operators["soft_repulsive_young_laplace_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=float(bridge_volume_ul) * 1.0e-9,
                surface_tension_n_m=float(config.surface_tension_n_m),
                density_kg_m3=float(config.density_kg_m3),
                gravity_m_s2=float(config.gravity_m_s2),
                nodes=int(config.attached_capillary_bridge_nodes),
                min_z_m=bridge_min_z,
                repulsion_length_m=(
                    float(config.attached_capillary_bridge_lower_bound_repulsion_length_um) * 1.0e-6
                ),
            )
        elif bridge_profile_model in {
            "linear_young_laplace",
            "linearized_young_laplace",
            "pressure",
            "pressure_volume",
        }:
            profile = operators["linearized_young_laplace_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=float(bridge_volume_ul) * 1.0e-9,
                surface_tension_n_m=float(config.surface_tension_n_m),
                density_kg_m3=float(config.density_kg_m3),
                gravity_m_s2=float(config.gravity_m_s2),
                nodes=int(config.attached_capillary_bridge_nodes),
                min_z_m=bridge_min_z,
                enforce_positive_profile=bool(config.attached_capillary_bridge_positive_volume_floor),
            )
        else:
            profile = operators["axisymmetric_capillary_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=float(bridge_volume_ul) * 1.0e-9,
                nodes=int(config.attached_capillary_bridge_nodes),
                min_z_m=bridge_min_z,
                contact_angle_rad=(
                    math.radians(float(config.contact_angle_deg))
                    if bool(config.attached_capillary_bridge_enforce_contact_angle)
                    else None
                ),
                rim_slope_m_per_m=rim_slope_m_per_m,
                previous_r_m=np.asarray(old_bridge_r, dtype=float),
                previous_z_m=np.asarray(old_bridge_z, dtype=float),
                lower_bound_repulsion=float(config.attached_capillary_bridge_lower_bound_repulsion),
                lower_bound_repulsion_length_m=(
                    float(config.attached_capillary_bridge_lower_bound_repulsion_length_um) * 1.0e-6
                ),
                slope_regularization=float(config.attached_capillary_bridge_slope_regularization),
                curvature_regularization=float(config.attached_capillary_bridge_curvature_regularization),
            )
        profile_r = np.asarray(profile["r_m"], dtype=float)
        profile_z = np.asarray(profile["z_m"], dtype=float)
        sample = np.linspace(0.0, 1.0, bridge_ring_indices.size)
        source = np.linspace(0.0, 1.0, profile_r.size)
        target_r = np.interp(sample, source, profile_r)
        target_z = np.interp(sample, source, profile_z)
        target_sphere_z = np.asarray(
            sphere_lower_z_m(config, np.minimum(target_r, sphere_radius * 0.999999)),
            dtype=float,
        )
        lower_z = np.full_like(target_z, bridge_min_z, dtype=float)
        upper_z = np.maximum(target_sphere_z - 1.0e-8, lower_z + 1.0e-10)
        lower_z[0] = contact_z
        upper_z[0] = contact_z
        lower_z[-1] = rim_z
        upper_z[-1] = rim_z
        target_z = regularize_attached_capillary_bridge_ring_profile(
            target_r,
            target_z,
            lower_z,
            upper_z,
            config,
        )
        for local_i, ring_i in enumerate(bridge_ring_indices):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
            old_xyz = out[ring].copy()
            out[ring, 0] = target_r[local_i] * np.cos(theta_bridge)
            out[ring, 1] = target_r[local_i] * np.sin(theta_bridge)
            out[ring, 2] = target_z[local_i]
            vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)
        out, vel = axisymmetrize(out, vel, rings)
        return repin_attached_contact_ring_to_sphere(out, vel, rings, ring_region, config)

    old_contact = out[contact_ring].copy()
    theta_contact = np.arctan2(out[contact_ring, 1], out[contact_ring, 0])
    out[contact_ring, 0] = contact_radius * np.cos(theta_contact)
    out[contact_ring, 1] = contact_radius * np.sin(theta_contact)
    out[contact_ring, 2] = contact_z
    vel[contact_ring] = (out[contact_ring] - old_contact) / max(float(dt_s), 1.0e-30)

    if bridge_ring_indices.size >= 3 and bool(config.attached_bridge_shape_constraint_enabled):
        bridge_gap = rim_radius - contact_radius
        ring_count = max(int(bridge_ring_indices.size) - 1, 1)
        min_spacing = max(abs(bridge_gap) * 0.015 / float(ring_count), 2.5e-7)
        neck_fraction = max(float(config.attached_bridge_shape_neck_fraction), 0.0)
        radius_exponent = 1.0 + 5.0 * neck_fraction
        height_exponent = max(float(config.attached_bridge_shape_height_exponent), 1.0e-12)
        for local_i, ring_i in enumerate(bridge_ring_indices):
            if int(ring_i) == int(bridge_ring_indices[0]) or int(ring_i) == int(bridge_ring_indices[-1]):
                continue
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            s = float(local_i) / max(float(bridge_ring_indices.size - 1), 1.0)
            height_ease = s**height_exponent
            if bridge_gap >= 0.0:
                target_radius = contact_radius + bridge_gap * (s**radius_exponent)
                lower = contact_radius + min_spacing * float(local_i)
                upper = rim_radius - min_spacing * float(ring_count - local_i)
                target_radius = float(np.clip(target_radius, lower, max(lower, upper)))
            else:
                target_radius = contact_radius + bridge_gap * (s**radius_exponent)
                upper = contact_radius - min_spacing * float(local_i)
                lower = rim_radius + min_spacing * float(ring_count - local_i)
                target_radius = float(np.clip(target_radius, min(lower, upper), upper))
            target_z = (1.0 - height_ease) * contact_z + height_ease * rim_z
            sphere_z = float(sphere_lower_z_m(config, min(target_radius, sphere_radius * 0.999999)))
            target_z = min(target_z, sphere_z - 1.0e-8)
            theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
            old_xyz = out[ring].copy()
            out[ring, 0] = target_radius * np.cos(theta_bridge)
            out[ring, 1] = target_radius * np.sin(theta_bridge)
            out[ring, 2] = target_z
            vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)

    out, vel = axisymmetrize(out, vel, rings)
    return repin_attached_contact_ring_to_sphere(out, vel, rings, ring_region, config)


def apply_attached_bridge_capillary_shoulder_operator(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
    t_s: float,
    bridge_volume_ul: float | None = None,
    operators: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Move saved bridge rings onto a continuous capillary branch.

    The outer film drain is solved separately.  This pass only regularizes the
    attached liquid surface between the solid/liquid/air contact ring and the
    film rim so the saved ddgclib mesh contains a continuous steep neck branch
    instead of a horizontal shelf at the initial film height.
    """

    if not bool(config.attached_bridge_capillary_shoulder_enabled):
        return points, velocities, {}

    ring_region_arr = np.asarray(ring_region, dtype=int)
    bridge_ring_indices = np.where(ring_region_arr == 0)[0]
    if bridge_ring_indices.size < 4:
        return points, velocities, {}

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    geom = attached_ring_geometry(out, rings, ring_region)
    contact_radius = float(geom["contact_radius_m"])
    rim_radius = float(geom["rim_radius_m"])
    gap = rim_radius - contact_radius
    if gap <= 1.0e-9:
        return out, vel, {}

    h0 = float(config.initial_film_thickness_um) * 1.0e-6
    contact_z = float(geom["contact_z_m"])
    rim_z = float(geom["rim_z_m"])
    bridge_min_z = max(attached_neck_floor_m(config, t_s), float(config.attached_capillary_bridge_min_height_um) * 1.0e-6, 0.25e-6)
    sphere_radius = float(config.sphere_radius_mm) * 1.0e-3

    min_width = max(float(config.attached_bridge_capillary_shoulder_min_width_mm), 0.0) * 1.0e-3
    max_width = max(float(config.attached_bridge_capillary_shoulder_max_width_mm), 0.0) * 1.0e-3
    if max_width < min_width:
        min_width, max_width = max_width, min_width
    depth_span = max(h0 - bridge_min_z, 1.0e-12)
    depth_fraction = float(np.clip((h0 - rim_z) / depth_span, 0.0, 1.0))
    width_exponent = max(float(config.attached_bridge_capillary_shoulder_width_depth_exponent), 1.0e-12)
    shoulder_width = max_width - (max_width - min_width) * (depth_fraction**width_exponent)
    shoulder_width = float(np.clip(shoulder_width, min_width, min(max_width, 0.92 * gap)))
    anchor_r = rim_radius - shoulder_width
    anchor_s = float(np.clip((anchor_r - contact_radius) / gap, 1.0e-6, 1.0 - 1.0e-6))
    # The branch exponent is not prescribed.  It is selected so the continuous
    # bridge profile crosses the undisturbed film height exactly one local
    # Cox/lubrication neck width before the film rim.
    if contact_z > h0 > rim_z:
        height_ratio = float(np.clip((contact_z - h0) / max(contact_z - rim_z, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6))
        branch_exponent = float(np.clip(math.log(height_ratio) / math.log(anchor_s), 0.35, 18.0))
        anchor_z = h0
    else:
        branch_exponent = 1.0
        anchor_z = contact_z - (contact_z - rim_z) * (anchor_s**branch_exponent)
    anchor_slope = -(contact_z - rim_z) * branch_exponent * (anchor_s ** (branch_exponent - 1.0)) / max(gap, 1.0e-12)
    rim_slope = 0.0
    relaxation = float(np.clip(config.attached_bridge_capillary_shoulder_relaxation, 0.0, 1.0))
    if relaxation <= 0.0:
        return out, vel, {}

    shoulder_profile = str(config.attached_bridge_capillary_shoulder_profile).strip().lower()
    drop_exponent = max(float(config.attached_bridge_capillary_shoulder_drop_exponent), 1.0e-12)
    use_contact_angle_meniscus = shoulder_profile in {
        "contact_angle",
        "contact-angle",
        "contact_angle_meniscus",
        "cox_contact_angle",
        "cox_young_meniscus",
        "cox_arc_meniscus",
        "cox_arc_length_meniscus",
        "arc_length_meniscus",
    }
    use_cox_arc_meniscus = shoulder_profile in {
        "cox_arc_meniscus",
        "cox_arc_length_meniscus",
        "arc_length_meniscus",
    }
    theta_dyn = math.radians(float(config.contact_angle_deg))
    raw_contact_slope = 0.0
    contact_slope = 0.0
    meniscus_rim_slope = 0.0
    if use_contact_angle_meniscus:
        contact_ring = np.asarray(geom["contact_ring"], dtype=int)
        contact_xyz = out[contact_ring]
        contact_radius_vertices = np.hypot(contact_xyz[:, 0], contact_xyz[:, 1])
        radial_unit = np.zeros_like(contact_xyz)
        valid = contact_radius_vertices > 1.0e-30
        radial_unit[valid, 0] = contact_xyz[valid, 0] / contact_radius_vertices[valid]
        radial_unit[valid, 1] = contact_xyz[valid, 1] / contact_radius_vertices[valid]
        slide_speed = float(np.mean(np.sum(vel[contact_ring] * radial_unit, axis=1))) if contact_ring.size else 0.0
        if float(t_s) > 0.0:
            slide_speed = math.copysign(
                max(abs(slide_speed), abs(contact_radius) / max(float(t_s), 1.0e-12)),
                slide_speed if slide_speed != 0.0 else 1.0,
            )
        log_factor = math.log(
            max(
                float(config.contact_line_cox_macro_length_m)
                / max(float(config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
        capillary_number = (
            max(float(config.viscosity_pa_s), 0.0)
            * abs(slide_speed)
            / max(float(config.surface_tension_n_m), 1.0e-30)
        )
        theta_dyn = float(
            np.cbrt(
                max(
                    math.radians(float(config.contact_angle_deg)) ** 3
                    + 9.0 * capillary_number * log_factor,
                    math.radians(float(config.dynamic_contact_angle_min_deg)) ** 3,
                )
            )
        )
        theta_dyn = float(
            np.clip(
                theta_dyn,
                math.radians(float(config.dynamic_contact_angle_min_deg)),
                math.radians(float(config.dynamic_contact_angle_max_deg)),
            )
        )
        axial = math.sqrt(max(sphere_radius * sphere_radius - contact_radius * contact_radius, 1.0e-30))
        sphere_slope = contact_radius / axial
        sphere_tangent_angle = math.atan(sphere_slope)
        raw_contact_slope = math.tan(sphere_tangent_angle - 0.5 * math.pi + theta_dyn)
        secant_slope = (rim_z - contact_z) / max(gap, 1.0e-12)
        if secant_slope < 0.0:
            # Cox gives the local direction at the solid.  The limiter keeps the
            # branch monotone on the current finite ring spacing instead of
            # creating an unresolved overhang/loop.
            contact_slope = float(np.clip(raw_contact_slope, 2.0 * secant_slope, 0.25 * secant_slope))
            meniscus_rim_slope = secant_slope
        else:
            contact_slope = secant_slope
            meniscus_rim_slope = secant_slope
    old = out.copy()

    if use_cox_arc_meniscus and operators is not None:
        sphere_bottom_z_m = (
            h0
            if config.sphere_bottom_z_mm is None
            else float(config.sphere_bottom_z_mm) * 1.0e-3
        )
        profile = operators["cox_arc_length_bridge_profile"](
            contact_radius_m=contact_radius,
            contact_z_m=contact_z,
            rim_radius_m=rim_radius,
            rim_z_m=rim_z,
            sphere_radius_m=sphere_radius,
            sphere_bottom_z_m=sphere_bottom_z_m,
            contact_slope_m_per_m=raw_contact_slope if math.isfinite(raw_contact_slope) else contact_slope,
            nodes=int(bridge_ring_indices.size),
            min_z_m=bridge_min_z,
        )
        target_r = np.asarray(profile["r_m"], dtype=float)
        target_z = np.asarray(profile["z_m"], dtype=float)
        for local_i, ring_i in enumerate(bridge_ring_indices):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
            old_xyz = out[ring].copy()
            out[ring, 0] = target_r[local_i] * np.cos(theta_bridge)
            out[ring, 1] = target_r[local_i] * np.sin(theta_bridge)
            out[ring, 2] = target_z[local_i]
            vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)
        out, vel = axisymmetrize(out, vel, rings)
        return out, vel, {
            "bridge_capillary_shoulder_width_mm": shoulder_width * 1.0e3,
            "bridge_capillary_shoulder_depth_fraction": depth_fraction,
            "bridge_capillary_shoulder_branch_exponent": branch_exponent,
            "bridge_capillary_shoulder_profile": shoulder_profile,
            "bridge_capillary_contact_angle_deg": theta_dyn * 180.0 / math.pi,
            "bridge_capillary_raw_contact_slope": raw_contact_slope,
            "bridge_capillary_contact_slope": contact_slope,
            "bridge_capillary_arc_cluster_exponent": float(profile.get("radial_cluster_exponent", 1.0)),
            "bridge_capillary_arc_first_slope": float(profile.get("first_segment_slope", 0.0)),
        }

    if shoulder_profile in {
        "area_min",
        "area_min_meniscus",
        "young_laplace",
        "young_laplace_meniscus",
        "capillary",
        "capillary_meniscus",
        "linear_young_laplace",
        "linear_young_laplace_meniscus",
        "linearized_young_laplace",
        "linearized_young_laplace_meniscus",
        "soft_young_laplace",
        "soft_young_laplace_meniscus",
        "soft_linearized_young_laplace",
        "soft_linearized_young_laplace_meniscus",
    } and operators is not None and bridge_volume_ul is not None:
        old_bridge_r: list[float] = []
        old_bridge_z: list[float] = []
        for ring_i in bridge_ring_indices:
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            xyz = out[ring]
            old_bridge_r.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
            old_bridge_z.append(float(np.mean(xyz[:, 2])))

        rim_slope_m_per_m = None
        if bool(config.attached_capillary_bridge_match_film_slope):
            film_after = np.where(ring_region_arr == 1)[0]
            film_after = film_after[film_after > int(bridge_ring_indices[-1])]
            if film_after.size:
                next_ring = np.asarray(rings[int(film_after[0])], dtype=int)
                next_xyz = out[next_ring]
                next_r = float(np.mean(np.hypot(next_xyz[:, 0], next_xyz[:, 1])))
                next_z = float(np.mean(next_xyz[:, 2]))
                if next_r > rim_radius + 1.0e-12:
                    rim_slope_m_per_m = (next_z - rim_z) / (next_r - rim_radius)

        sphere_bottom_z_m = (
            h0
            if config.sphere_bottom_z_mm is None
            else float(config.sphere_bottom_z_mm) * 1.0e-3
        )
        bridge_volume_m3 = max(float(bridge_volume_ul), 0.0) * 1.0e-9
        profile_nodes = max(int(config.attached_capillary_bridge_nodes), int(bridge_ring_indices.size) * 4)
        if shoulder_profile in {
            "linear_young_laplace",
            "linear_young_laplace_meniscus",
            "linearized_young_laplace",
            "linearized_young_laplace_meniscus",
        }:
            profile = operators["linearized_young_laplace_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=bridge_volume_m3,
                surface_tension_n_m=float(config.surface_tension_n_m),
                density_kg_m3=float(config.density_kg_m3),
                gravity_m_s2=float(config.gravity_m_s2),
                nodes=profile_nodes,
                min_z_m=bridge_min_z,
                enforce_positive_profile=bool(config.attached_capillary_bridge_positive_volume_floor),
            )
        elif shoulder_profile in {
            "soft_young_laplace",
            "soft_young_laplace_meniscus",
            "soft_linearized_young_laplace",
            "soft_linearized_young_laplace_meniscus",
        }:
            profile = operators["soft_repulsive_young_laplace_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=bridge_volume_m3,
                surface_tension_n_m=float(config.surface_tension_n_m),
                density_kg_m3=float(config.density_kg_m3),
                gravity_m_s2=float(config.gravity_m_s2),
                nodes=profile_nodes,
                min_z_m=bridge_min_z,
                repulsion_length_m=(
                    float(config.attached_capillary_bridge_lower_bound_repulsion_length_um) * 1.0e-6
                ),
            )
        else:
            profile = operators["axisymmetric_capillary_bridge_profile"](
                sphere_radius_m=sphere_radius,
                sphere_bottom_z_m=sphere_bottom_z_m,
                contact_radius_m=contact_radius,
                rim_radius_m=rim_radius,
                rim_z_m=rim_z,
                bridge_volume_m3=bridge_volume_m3,
                nodes=profile_nodes,
                min_z_m=bridge_min_z,
                contact_angle_rad=(
                    math.radians(float(config.contact_angle_deg))
                    if bool(config.attached_capillary_bridge_enforce_contact_angle)
                    else None
                ),
                rim_slope_m_per_m=rim_slope_m_per_m,
                previous_r_m=np.asarray(old_bridge_r, dtype=float),
                previous_z_m=np.asarray(old_bridge_z, dtype=float),
                lower_bound_repulsion=float(config.attached_capillary_bridge_lower_bound_repulsion),
                lower_bound_repulsion_length_m=(
                    float(config.attached_capillary_bridge_lower_bound_repulsion_length_um) * 1.0e-6
                ),
                slope_regularization=float(config.attached_capillary_bridge_slope_regularization),
                curvature_regularization=float(config.attached_capillary_bridge_curvature_regularization),
            )
        profile_r = np.asarray(profile["r_m"], dtype=float)
        profile_z = np.asarray(profile["z_m"], dtype=float)
        sample = np.linspace(0.0, 1.0, bridge_ring_indices.size)
        source = np.linspace(0.0, 1.0, profile_r.size)
        target_r = np.interp(sample, source, profile_r)
        target_z = np.interp(sample, source, profile_z)
        target_sphere_z = np.asarray(
            sphere_lower_z_m(config, np.minimum(target_r, sphere_radius * 0.999999)),
            dtype=float,
        )
        lower_z = np.full_like(target_z, bridge_min_z, dtype=float)
        upper_z = np.maximum(target_sphere_z - 1.0e-8, lower_z + 1.0e-10)
        lower_z[0] = contact_z
        upper_z[0] = contact_z
        lower_z[-1] = rim_z
        upper_z[-1] = rim_z
        target_z = regularize_attached_capillary_bridge_ring_profile(
            target_r,
            target_z,
            lower_z,
            upper_z,
            config,
        )
        for local_i, ring_i in enumerate(bridge_ring_indices):
            ring = np.asarray(rings[int(ring_i)], dtype=int)
            theta_bridge = np.arctan2(out[ring, 1], out[ring, 0])
            old_xyz = out[ring].copy()
            out[ring, 0] = target_r[local_i] * np.cos(theta_bridge)
            out[ring, 1] = target_r[local_i] * np.sin(theta_bridge)
            out[ring, 2] = target_z[local_i]
            vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)
        out, vel = axisymmetrize(out, vel, rings)
        return out, vel, {
            "bridge_capillary_shoulder_width_mm": shoulder_width * 1.0e3,
            "bridge_capillary_shoulder_depth_fraction": depth_fraction,
            "bridge_capillary_shoulder_branch_exponent": branch_exponent,
            "bridge_capillary_shoulder_profile": shoulder_profile,
            "bridge_capillary_shoulder_yl_success": float(bool(profile.get("success", False))),
        }

    for ring_i in bridge_ring_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        rr = float(np.mean(np.hypot(out[ring, 0], out[ring, 1])))
        if int(ring_i) == int(bridge_ring_indices[0]):
            target_z = contact_z
        elif int(ring_i) == int(bridge_ring_indices[-1]):
            target_z = rim_z
        elif use_contact_angle_meniscus:
            eta = float(np.clip((rr - contact_radius) / max(gap, 1.0e-12), 0.0, 1.0))
            h00 = 2.0 * eta**3 - 3.0 * eta**2 + 1.0
            h10 = eta**3 - 2.0 * eta**2 + eta
            h01 = -2.0 * eta**3 + 3.0 * eta**2
            h11 = eta**3 - eta**2
            target_z = (
                h00 * contact_z
                + h10 * gap * contact_slope
                + h01 * rim_z
                + h11 * gap * meniscus_rim_slope
            )
        elif shoulder_profile in {"smooth", "smooth_meniscus", "cubic_meniscus"}:
            eta = float(np.clip((rr - contact_radius) / max(gap, 1.0e-12), 0.0, 1.0))
            smooth = eta * eta * (3.0 - 2.0 * eta)
            target_z = (1.0 - smooth) * contact_z + smooth * rim_z
        elif shoulder_profile in {"power", "power_meniscus", "concave_meniscus"}:
            eta = float(np.clip((rr - contact_radius) / max(gap, 1.0e-12), 0.0, 1.0))
            target_z = rim_z + max(contact_z - rim_z, 0.0) * ((1.0 - eta) ** max(drop_exponent, 1.0e-12))
        elif rr <= anchor_r:
            s = float(np.clip((rr - contact_radius) / max(gap, 1.0e-12), 0.0, 1.0))
            target_z = contact_z - (contact_z - rim_z) * (s**branch_exponent)
        else:
            eta = float(np.clip((rr - anchor_r) / max(shoulder_width, 1.0e-12), 0.0, 1.0))
            h00 = 2.0 * eta**3 - 3.0 * eta**2 + 1.0
            h10 = eta**3 - 2.0 * eta**2 + eta
            h01 = -2.0 * eta**3 + 3.0 * eta**2
            h11 = eta**3 - eta**2
            target_z = (
                h00 * anchor_z
                + h10 * shoulder_width * anchor_slope
                + h01 * rim_z
                + h11 * shoulder_width * rim_slope
            )

        sphere_z = float(sphere_lower_z_m(config, min(max(rr, 0.0), sphere_radius * 0.999999)))
        upper_z = max(sphere_z - 1.0e-8, bridge_min_z + 1.0e-10)
        target_z = float(np.clip(target_z, bridge_min_z, upper_z))
        out[ring, 2] = (1.0 - relaxation) * out[ring, 2] + relaxation * target_z

    vel = (out - old) / max(float(dt_s), 1.0e-30)
    out, vel = axisymmetrize(out, vel, rings)
    return out, vel, {
        "bridge_capillary_shoulder_width_mm": shoulder_width * 1.0e3,
        "bridge_capillary_shoulder_depth_fraction": depth_fraction,
        "bridge_capillary_shoulder_branch_exponent": branch_exponent,
        "bridge_capillary_shoulder_profile": shoulder_profile,
        "bridge_capillary_contact_angle_deg": theta_dyn * 180.0 / math.pi,
        "bridge_capillary_raw_contact_slope": raw_contact_slope,
        "bridge_capillary_contact_slope": contact_slope,
    }


def regularize_attached_film_rim_heights(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Damp one-ring height oscillations created by the moving rim constraint."""

    relaxation = float(np.clip(config.attached_film_height_regularization_per_step, 0.0, 1.0))
    width_m = float(config.attached_film_height_regularization_width_mm) * 1.0e-3
    if relaxation <= 0.0 or width_m <= 0.0:
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    bridge_indices = np.where(ring_region_arr == 0)[0]
    film_indices = np.where(ring_region_arr == 1)[0]
    if bridge_indices.size == 0 or film_indices.size < 3:
        return out, vel

    rim_i = int(bridge_indices[-1])
    rim_ring = np.asarray(rings[rim_i], dtype=int)
    rim_r = float(np.mean(np.hypot(out[rim_ring, 0], out[rim_ring, 1])))
    ring_r = []
    ring_z = []
    for ring in np.asarray(rings, dtype=int):
        xyz = out[np.asarray(ring, dtype=int)]
        ring_r.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        ring_z.append(float(np.mean(xyz[:, 2])))
    ring_r_arr = np.asarray(ring_r, dtype=float)
    ring_z_arr = np.asarray(ring_z, dtype=float)
    min_z = float(config.min_height_um) * 1.0e-6
    max_z = max(float(config.max_height_um), float(config.initial_film_thickness_um)) * 1.0e-6

    for ring_i in film_indices:
        ring_i = int(ring_i)
        rr = float(ring_r_arr[ring_i])
        distance = max(rr - rim_r, 0.0)
        if distance > width_m:
            continue
        previous_i = ring_i - 1
        next_i = ring_i + 1
        if previous_i < 0 or next_i >= len(rings):
            continue
        if ring_region_arr[min(next_i, ring_region_arr.size - 1)] != 1:
            continue
        local_weight = math.exp(-distance / max(width_m, 1.0e-12))
        target_z = 0.25 * ring_z_arr[previous_i] + 0.5 * ring_z_arr[ring_i] + 0.25 * ring_z_arr[next_i]
        target_z = float(np.clip(target_z, min_z, max_z))
        ring = np.asarray(rings[ring_i], dtype=int)
        old_z = out[ring, 2].copy()
        out[ring, 2] = old_z + relaxation * local_weight * (target_z - old_z)
        vel[ring, 2] = (out[ring, 2] - old_z) / max(float(dt_s), 1.0e-30)

    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def remesh_attached_film_rings_outside_rim(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config: RealMeshEvolutionConfig,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all film rings outside the moving bridge-film rim."""

    if not bool(config.attached_film_radial_remesh_enabled):
        return points, velocities

    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    ring_region_arr = np.asarray(ring_region, dtype=int)
    bridge_indices = np.where(ring_region_arr == 0)[0]
    film_indices = np.where(ring_region_arr == 1)[0]
    if bridge_indices.size == 0 or film_indices.size < 2:
        return out, vel

    rim_i = int(bridge_indices[-1])
    rim_ring = np.asarray(rings[rim_i], dtype=int)
    rim_r = float(np.mean(np.hypot(out[rim_ring, 0], out[rim_ring, 1])))
    substrate_r = float(config.substrate_radius_mm) * 1.0e-3
    if rim_r >= substrate_r * 0.995:
        return out, vel

    old_r: list[float] = []
    old_z: list[float] = []
    for ring_i in film_indices:
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        xyz = out[ring]
        old_r.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
        old_z.append(float(np.mean(xyz[:, 2])))
    old_r_arr = np.asarray(old_r, dtype=float)
    old_z_arr = np.asarray(old_z, dtype=float)
    order = np.argsort(old_r_arr)
    interp_r = old_r_arr[order]
    interp_z = old_z_arr[order]
    keep = np.concatenate(([True], np.diff(interp_r) > 1.0e-12))
    interp_r = interp_r[keep]
    interp_z = interp_z[keep]

    n = int(film_indices.size)
    exponent = max(float(config.attached_film_radial_remesh_exponent), 1.0e-6)
    eta = (np.arange(1, n + 1, dtype=float) / float(n)) ** exponent
    target_r = rim_r + (substrate_r - rim_r) * eta
    target_r[-1] = substrate_r
    target_z = np.interp(
        target_r,
        interp_r,
        interp_z,
        left=float(interp_z[0]),
        right=float(interp_z[-1]),
    )
    min_z = float(config.min_height_um) * 1.0e-6
    max_z = max(float(config.max_height_um), float(config.initial_film_thickness_um)) * 1.0e-6
    target_z = np.clip(target_z, min_z, max_z)

    for local_i, ring_i in enumerate(film_indices):
        ring = np.asarray(rings[int(ring_i)], dtype=int)
        theta = np.arctan2(out[ring, 1], out[ring, 0])
        old_xyz = out[ring].copy()
        out[ring, 0] = float(target_r[local_i]) * np.cos(theta)
        out[ring, 1] = float(target_r[local_i]) * np.sin(theta)
        out[ring, 2] = float(target_z[local_i])
        vel[ring] = (out[ring] - old_xyz) / max(float(dt_s), 1.0e-30)

    out, vel = axisymmetrize(out, vel, rings)
    return out, vel


def run_attached_bridge_mesh_evolution(config: RealMeshEvolutionConfig, out_dir: Path, operators: dict) -> dict:
    """Evolve the connected bridge+film mesh itself.

    This is the Case 11 path: the primary ddgclib graph contains the
    sphere-attached meniscus and the outer film.  The sphere contact ring is a
    mesh boundary constrained to the spherical solid and advanced with the
    Cox/Cox-Voinov contact-line relation.
    """

    film_mesh = make_initial_mesh(config, operators)
    film_points = np.asarray(film_mesh["vertices_m"], dtype=float)
    film_faces = np.asarray(film_mesh["faces"], dtype=np.int32)
    film_rings = np.asarray(film_mesh["ring_index"], dtype=np.int32)
    film_r, film_h = film_profile_from_points(film_points, film_rings)
    initial_bridge_volume_ul = float(config.inner_bridge_volume_ul)
    mesh = make_attached_bridge_mesh_from_profile(film_r, film_h, config, operators, initial_bridge_volume_ul)
    ring_region = np.asarray(mesh["ring_region"], dtype=np.int32)
    vertices, faces, rings, _vertex_ring = graph_from_mesh(mesh)
    initial_points = points_from_vertices(vertices)
    points = np.array(initial_points, copy=True)
    velocities = np.zeros_like(points)
    initial_attached_missing_ul = attached_missing_outer_film_volume_ul(points, rings, ring_region, config)
    reduced_volume_driver = build_attached_reduced_bridge_driver(config, operators)

    initial_areas = lumped_area_from_faces(initial_points, faces)
    initial_forces = np.zeros_like(initial_points)
    if bool(config.subtract_initial_equilibrium_force):
        for idx, vertex in enumerate(vertices):
            initial_forces[idx] = operators["surface_tension_force"](vertex, gamma=float(config.surface_tension_n_m), dim=3)
    initial_theta = np.arctan2(initial_points[:, 1], initial_points[:, 0])
    initial_pressure_z = initial_forces[:, 2] / initial_areas
    initial_pressure_r = (
        initial_forces[:, 0] * np.cos(initial_theta) + initial_forces[:, 1] * np.sin(initial_theta)
    ) / initial_areas

    history: list[dict] = []
    snapshots: dict[float, dict] = {}
    next_snapshot_index = 0
    snapshot_times = sorted(float(t) for t in config.snapshot_times_s)
    start_wall = time.monotonic()
    t_s = 0.0
    irreversible_missing_ul = 0.0

    def state_from_current(step: int, t_value: float, bridge_volume_ul: float) -> dict:
        geom = attached_ring_geometry(points, rings, ring_region)
        return {
            "time_s": float(t_value),
            "step": int(step),
            "vertices_m": np.asarray(points, dtype=float),
            "faces": np.asarray(faces, dtype=np.int32),
            "ring_index": np.asarray(rings, dtype=np.int32),
            "ring_region": np.asarray(ring_region, dtype=np.int32),
            "bridge_contact_radius_m": np.asarray(float(geom["contact_radius_m"])),
            "bridge_contact_z_m": np.asarray(float(geom["contact_z_m"])),
            "bridge_rim_radius_m": np.asarray(float(geom["rim_radius_m"])),
            "bridge_rim_z_m": np.asarray(float(geom["rim_z_m"])),
            "bridge_volume_ul": np.asarray(float(bridge_volume_ul)),
            "bridge_radius_mm": np.asarray(bridge_radius_from_volume_mm(config, bridge_volume_ul)),
            "bridge_head_mm": np.asarray(float(geom["contact_z_m"]) * 1.0e3),
            "method": "full attached bridge+film ddgclib mesh evolution",
        }

    def save_snapshot(step: int, t_value: float, bridge_volume_ul: float) -> None:
        state = state_from_current(step, t_value, bridge_volume_ul)
        snapshots[float(t_value)] = state
        mesh_dir = out_dir / "mesh_states"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        label = f"step{int(step):07d}_t{float(t_value):.4f}s".replace(".", "p")
        operators["write_npz"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.npz", state)
        operators["write_obj"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.obj", state["vertices_m"], state["faces"])
        if "write_msh" in operators:
            operators["write_msh"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.msh", state["vertices_m"], state["faces"])

    for step in range(int(config.max_steps) + 1):
        t_s = step * float(config.dt_s)
        wall_elapsed = time.monotonic() - start_wall
        areas = lumped_area_from_faces(points, faces)
        for idx, (vertex, point) in enumerate(zip(vertices, points)):
            vertex.x_a[:] = point
            vertex.u[:] = velocities[idx]
            vertex.m = float(config.density_kg_m3) * float(config.initial_film_thickness_um) * 1.0e-6 * float(areas[idx])

        heron_forces = np.zeros_like(points)
        for idx, vertex in enumerate(vertices):
            heron_forces[idx] = operators["surface_tension_force"](vertex, gamma=float(config.surface_tension_n_m), dim=3)

        geom = attached_ring_geometry(points, rings, ring_region)
        missing_ul = max(
            attached_missing_outer_film_volume_ul(points, rings, ring_region, config) - initial_attached_missing_ul,
            0.0,
        )
        if reduced_volume_driver is not None:
            missing_ul = max(
                missing_ul,
                float(np.interp(float(t_s), reduced_volume_driver["time_s"], reduced_volume_driver["delta_volume_ul"])),
            )
        if bool(config.attached_bridge_growth_irreversible_enabled):
            missing_ul = max(float(missing_ul), float(irreversible_missing_ul))
        effective_missing_ul = missing_ul
        if bool(config.attached_state_closure_enabled) and bool(config.attached_closure_limit_missing_volume):
            tentative_bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
            tentative_bridge_radius_mm = bridge_radius_from_volume_mm(config, tentative_bridge_volume_ul)
            tentative_closure = attached_state_closure(
                config,
                operators,
                tentative_bridge_volume_ul,
                tentative_bridge_radius_mm,
            )
            tentative_pressure_center_m = max(
                float(geom["rim_radius_m"]) + float(tentative_closure["pressure_center_offset_m"]),
                1.0e-12,
            )
            effective_missing_ul = min(
                missing_ul,
                float(tentative_closure["allowed_missing_ul"]),
                bridge_feed_capacity_ul(config, tentative_pressure_center_m, t_s),
            )
        bridge_volume_ul = float(config.inner_bridge_volume_ul) + effective_missing_ul
        bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
        bridge_head_mm = bridge_head_from_volume_mm(config, bridge_volume_ul)
        suction_pa = -float(config.density_kg_m3) * float(config.gravity_m_s2) * bridge_head_mm * 1.0e-3

        r = np.hypot(points[:, 0], points[:, 1])
        theta = np.arctan2(points[:, 1], points[:, 0])
        bridge_mask = np.zeros(points.shape[0], dtype=bool)
        film_mask = np.zeros(points.shape[0], dtype=bool)
        for ring_i, ring in enumerate(np.asarray(rings, dtype=int)):
            target = bridge_mask if int(ring_region[min(ring_i, len(ring_region) - 1)]) == 0 else film_mask
            target[np.asarray(ring, dtype=int)] = True
        rim_ring = np.asarray(geom["rim_ring"], dtype=int)
        film_mask[rim_ring] = True

        width_m = max(float(config.bridge_pressure_width_mm) * 1.0e-3, 1.0e-8)
        pressure_center_m = float(geom["rim_radius_m"])
        if bool(config.attached_state_closure_enabled) and bool(config.attached_closure_limit_missing_volume):
            closure = attached_state_closure(config, operators, bridge_volume_ul, bridge_radius_mm)
            pressure_center_m = max(
                pressure_center_m + float(closure["pressure_center_offset_m"]),
                1.0e-12,
            )
            width_m = max(width_m, float(closure["pressure_width_m"]), 1.0e-8)
        weight = np.exp(-0.5 * ((r - pressure_center_m) / width_m) ** 2) * film_mask.astype(float)
        pressure_z = (
            heron_forces[:, 2] / areas
            - initial_pressure_z
            + bridge_pressure_multiplier(config, t_s, operators, bridge_volume_ul, bridge_radius_mm) * suction_pa * weight
        )
        pressure_r = (
            heron_forces[:, 0] * np.cos(theta) + heron_forces[:, 1] * np.sin(theta)
        ) / areas - initial_pressure_r

        h0 = float(config.initial_film_thickness_um) * 1.0e-6
        h_factor = np.clip(
            points[:, 2] / max(h0, 1.0e-30),
            float(config.lubrication_mobility_floor) ** (1.0 / max(float(config.lubrication_mobility_exponent), 1.0e-12)),
            1.0,
        ) ** float(config.lubrication_mobility_exponent)
        bridge_mobility = max(float(config.attached_bridge_relaxation), 0.0)
        mobility_factor = np.where(film_mask, h_factor, bridge_mobility)
        mesh_activation = 1.0 - math.exp(
            -(
                max(float(t_s), 0.0)
                / max(float(config.attached_mesh_mobility_startup_time_s), 1.0e-12)
            )
            ** float(config.attached_mesh_mobility_startup_exponent)
        )
        mesh_activation = float(config.attached_mesh_mobility_floor) + (
            1.0 - float(config.attached_mesh_mobility_floor)
        ) * mesh_activation
        mobility_scale = film_thickness_mobility_scale(config)
        speed_z = mobility_scale * float(config.vertical_mobility_m_per_s_pa) * mobility_factor * pressure_z
        speed_r = (
            mobility_scale
            * float(config.radial_mobility_factor)
            * float(config.vertical_mobility_m_per_s_pa)
            * mobility_factor
            * pressure_r
        )
        speed_z *= mesh_activation
        speed_r *= mesh_activation
        speed_z = np.clip(speed_z, -float(config.max_vertical_speed_um_s) * 1.0e-6, float(config.max_vertical_speed_um_s) * 1.0e-6)
        speed_r = np.clip(speed_r, -float(config.max_radial_speed_um_s) * 1.0e-6, float(config.max_radial_speed_um_s) * 1.0e-6)

        new_velocities = np.zeros_like(points)
        new_velocities[:, 0] = speed_r * np.cos(theta)
        new_velocities[:, 1] = speed_r * np.sin(theta)
        new_velocities[:, 2] = speed_z
        contact_diag = {
            "cox_contact_ring_radius_mm": float(geom["contact_radius_m"]) * 1.0e3,
            "cox_contact_ring_z_mm": float(geom["contact_z_m"]) * 1.0e3,
            "cox_slide_speed_mean_um_s": 0.0,
            "cox_theta_mean_deg": attached_contact_angle_rad(points, rings, ring_region, config) * 180.0 / math.pi,
            "cox_activation": 0.0,
            "bridge_rim_radius_mm": float(geom["rim_radius_m"]) * 1.0e3,
            "bridge_rim_z_um": float(geom["rim_z_m"]) * 1.0e6,
        }

        velocities = float(config.velocity_smoothing) * velocities + (1.0 - float(config.velocity_smoothing)) * new_velocities
        if step > 0:
            points = points + float(config.dt_s) * velocities
            target_rim_radius_m = (
                bridge_radius_from_volume_mm(config, bridge_volume_ul)
                + attached_bridge_rim_offset_mm(config, t_s, operators, bridge_volume_ul, bridge_radius_mm)
            ) * 1.0e-3
            target_contact_radius_m = None
            if bool(config.attached_contact_line_bridge_radius_coupling_enabled):
                contact_volume_ul = float(bridge_volume_ul)
                if bool(config.attached_contact_line_target_uses_visible_feed_partition):
                    geom_for_contact = attached_ring_geometry(points, rings, ring_region)
                    visible_fraction = attached_visible_feed_fraction(
                        config,
                        float(geom_for_contact["rim_radius_m"]),
                        max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
                        contact_radius_m=float(geom_for_contact["contact_radius_m"]),
                    )
                    contact_volume_ul = float(config.inner_bridge_volume_ul) + (
                        max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0)
                        * visible_fraction
                    )
                reference_radius_mm = bridge_radius_from_volume_mm(config, float(config.inner_bridge_volume_ul))
                seed_mm = 0.0
                if config.attached_initial_contact_radius_mm is not None:
                    seed_mm = float(config.attached_initial_contact_radius_mm)
                contact_driver_radius_mm = bridge_radius_from_volume_mm(config, contact_volume_ul)
                target_contact_radius_mm = max(
                    seed_mm,
                    (
                        seed_mm
                        + max(float(config.attached_contact_line_bridge_radius_growth_multiplier), 0.0)
                        * max(float(contact_driver_radius_mm) - reference_radius_mm, 0.0)
                    ),
                )
                bridge_head_limited_radius_mm = sphere_radius_at_lower_z_m(
                    config,
                    bridge_head_mm * 1.0e-3,
                ) * 1.0e3
                target_contact_radius_mm = min(target_contact_radius_mm, bridge_head_limited_radius_mm)
                target_contact_radius_m = target_contact_radius_mm * 1.0e-3
            points, velocities, contact_diag = enforce_attached_bridge_constraints(
                points,
                velocities,
                initial_points,
                rings,
                ring_region,
                config,
                float(config.dt_s),
                t_s,
                target_rim_radius_m,
                target_contact_radius_m,
                operators,
            )
            points, velocities = remesh_attached_film_rings_outside_rim(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
            )
            points, velocities = regularize_attached_film_rim_heights(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
            )
            if reduced_volume_driver is not None:
                driver_target_ul = float(
                    np.interp(
                        float(t_s),
                        reduced_volume_driver["time_s"],
                        reduced_volume_driver["delta_volume_ul"],
                    )
                )
                points = project_attached_missing_volume_to_target(
                points,
                faces,
                rings,
                ring_region,
                config,
                initial_attached_missing_ul,
                driver_target_ul,
                t_s,
            )
                points, velocities = apply_attached_visible_feed_profile_constraint(
                    points,
                    velocities,
                    rings,
                    ring_region,
                    config,
                    float(config.dt_s),
                    initial_attached_missing_ul,
                    driver_target_ul,
                )
                points, velocities = apply_attached_reduced_driver_profile_projection(
                    points,
                    velocities,
                    rings,
                    ring_region,
                    config,
                    float(config.dt_s),
                    t_s,
                    reduced_volume_driver,
                )
            points = project_attached_missing_volume_limit(
                points,
                faces,
                rings,
                ring_region,
                config,
                t_s,
                initial_attached_missing_ul,
                operators,
            )
        neck_update_mode = str(config.attached_neck_boundary_layer_update_mode).strip().lower()
        residual_neck_update = neck_update_mode in {
            "residual",
            "residual-mobility",
            "residual_mobility",
            "mobility",
        }
        defer_bridge_shape_until_after_neck = bool(config.attached_capillary_bridge_solver_enabled) and (
            bool(config.attached_neck_boundary_layer_enabled)
            or bool(config.attached_compact_neck_profile_enabled)
        ) and not residual_neck_update
        if defer_bridge_shape_until_after_neck:
            points, velocities, pinned_contact_diag = repin_attached_contact_ring_to_sphere(
                points,
                velocities,
                rings,
                ring_region,
                config,
            )
        else:
            points, velocities, pinned_contact_diag = apply_final_attached_bridge_shape_constraint(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
                bridge_volume_ul,
                operators,
                t_s,
            )
        points, velocities = apply_attached_neck_boundary_layer_operator(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
            max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
            operators,
        )
        points, velocities = apply_attached_compact_neck_profile_constraint(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
            initial_attached_missing_ul,
            max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
        )
        points, velocities = apply_attached_outer_deficit_spreading_operator(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
            max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
        )
        points, velocities = apply_attached_volume_recovery_front_profile(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
            initial_attached_missing_ul,
            max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
        )
        if (
            bool(config.attached_capillary_bridge_solver_enabled)
            or bool(config.attached_bridge_shape_constraint_enabled)
        ) and not residual_neck_update:
            points, velocities, pinned_contact_diag = apply_final_attached_bridge_shape_constraint(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
                bridge_volume_ul,
                operators,
                t_s,
            )
        points, velocities, pinned_contact_diag = repin_attached_contact_ring_to_sphere(
            points,
            velocities,
            rings,
            ring_region,
            config,
        )
        points, velocities = repair_attached_outer_film_monotone_recovery(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
        )
        points, velocities = apply_attached_volume_recovery_front_profile(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
            initial_attached_missing_ul,
            max(float(bridge_volume_ul) - float(config.inner_bridge_volume_ul), 0.0),
        )
        points, velocities = regularize_attached_outer_film_sawtooth(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
        )
        for key, value in pinned_contact_diag.items():
            if key in {"cox_slide_speed_mean_um_s", "cox_activation"}:
                continue
            contact_diag[key] = value

        if bool(config.attached_bridge_growth_irreversible_enabled):
            required_missing_ul = float(irreversible_missing_ul)
            if reduced_volume_driver is not None:
                required_missing_ul = max(
                    required_missing_ul,
                    float(np.interp(float(t_s), reduced_volume_driver["time_s"], reduced_volume_driver["delta_volume_ul"])),
                )
            current_missing_ul = max(
                attached_missing_outer_film_volume_ul(points, rings, ring_region, config) - initial_attached_missing_ul,
                0.0,
            )
            tolerance_ul = max(float(config.attached_bridge_growth_irreversible_tolerance_ul), 0.0)
            if current_missing_ul + tolerance_ul < required_missing_ul:
                old_points = points.copy()
                points = project_attached_missing_volume_to_target(
                    points,
                    faces,
                    rings,
                    ring_region,
                    config,
                    initial_attached_missing_ul,
                    required_missing_ul,
                    t_s,
                )
                velocities = (points - old_points) / max(float(config.dt_s), 1.0e-30)
                points, velocities, pinned_contact_diag = repin_attached_contact_ring_to_sphere(
                    points,
                    velocities,
                    rings,
                    ring_region,
                    config,
                )
                for key, value in pinned_contact_diag.items():
                    if key in {"cox_slide_speed_mean_um_s", "cox_activation"}:
                        continue
                    contact_diag[key] = value
                current_missing_ul = max(
                    attached_missing_outer_film_volume_ul(points, rings, ring_region, config)
                    - initial_attached_missing_ul,
                    0.0,
                )
            irreversible_missing_ul = max(
                float(irreversible_missing_ul),
                float(current_missing_ul),
                float(required_missing_ul),
            )

        points, velocities = repair_attached_outer_film_monotone_recovery(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
        )
        points, velocities = regularize_attached_outer_film_sawtooth(
            points,
            velocities,
            rings,
            ring_region,
            config,
            float(config.dt_s),
            t_s,
        )

        r_profile_mm, h_profile_um = attached_profile_rows_for_validation(points, rings, ring_region)
        missing_ul = max(
            attached_missing_outer_film_volume_ul(points, rings, ring_region, config) - initial_attached_missing_ul,
            0.0,
        )
        if reduced_volume_driver is not None:
            missing_ul = max(
                missing_ul,
                float(np.interp(float(t_s), reduced_volume_driver["time_s"], reduced_volume_driver["delta_volume_ul"])),
            )
        if bool(config.attached_bridge_growth_irreversible_enabled):
            missing_ul = max(float(missing_ul), float(irreversible_missing_ul))
        effective_missing_ul = missing_ul
        if bool(config.attached_state_closure_enabled) and bool(config.attached_closure_limit_missing_volume):
            tentative_bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
            tentative_bridge_radius_mm = bridge_radius_from_volume_mm(config, tentative_bridge_volume_ul)
            tentative_closure = attached_state_closure(
                config,
                operators,
                tentative_bridge_volume_ul,
                tentative_bridge_radius_mm,
            )
            geom_after_step = attached_ring_geometry(points, rings, ring_region)
            tentative_pressure_center_m = max(
                float(geom_after_step["rim_radius_m"]) + float(tentative_closure["pressure_center_offset_m"]),
                1.0e-12,
            )
            effective_missing_ul = min(
                missing_ul,
                float(tentative_closure["allowed_missing_ul"]),
                bridge_feed_capacity_ul(config, tentative_pressure_center_m, t_s),
            )
        bridge_volume_ul = float(config.inner_bridge_volume_ul) + effective_missing_ul
        bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
        bridge_head_mm = bridge_head_from_volume_mm(config, bridge_volume_ul)
        if (
            bool(config.attached_capillary_bridge_solver_enabled)
            or bool(config.attached_bridge_shape_constraint_enabled)
        ):
            # The feed/closure calculation above can update bridge_volume_ul
            # after the main mesh-shape pass.  Re-apply the attached bridge
            # profile with the current volume before diagnostics/snapshots so
            # saved mesh states do not contain a one-step-lagged clipped neck.
            points, velocities, pinned_contact_diag = apply_final_attached_bridge_shape_constraint(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
                bridge_volume_ul,
                operators,
                t_s,
            )
            points, velocities, pinned_contact_diag = repin_attached_contact_ring_to_sphere(
                points,
                velocities,
                rings,
                ring_region,
                config,
            )
            for key, value in pinned_contact_diag.items():
                if key in {"cox_slide_speed_mean_um_s", "cox_activation"}:
                    continue
                contact_diag[key] = value
            points, velocities, shoulder_diag = apply_attached_bridge_capillary_shoulder_operator(
                points,
                velocities,
                rings,
                ring_region,
                config,
                float(config.dt_s),
                t_s,
                bridge_volume_ul,
                operators,
            )
            if shoulder_diag:
                contact_diag.update(shoulder_diag)
            r_profile_mm, h_profile_um = attached_profile_rows_for_validation(points, rings, ring_region)
        bridge_window = (r_profile_mm >= 2.4) & (r_profile_mm <= 6.2)
        if np.any(bridge_window):
            local_profile_idx = np.where(bridge_window)[0][int(np.argmin(h_profile_um[bridge_window]))]
        else:
            local_profile_idx = int(np.argmin(h_profile_um))

        while next_snapshot_index < len(snapshot_times) and t_s + 1.0e-12 >= snapshot_times[next_snapshot_index]:
            save_snapshot(step, float(snapshot_times[next_snapshot_index]), bridge_volume_ul)
            next_snapshot_index += 1

        if step % int(config.record_every_steps) == 0 or step == int(config.max_steps):
            row = {
                "step": int(step),
                "t_s": float(t_s),
                "wall_elapsed_s": float(wall_elapsed),
                "mesh_volume_ul": float("nan"),
                "missing_volume_ul": float(missing_ul),
                "bridge_volume_ul": float(bridge_volume_ul),
                "bridge_radius_mm": float(bridge_radius_mm),
                "bridge_head_mm": float(bridge_head_mm),
                "inner_plateau_radius_mm": float(contact_diag["bridge_rim_radius_mm"]),
                "suction_pa": float(-float(config.density_kg_m3) * float(config.gravity_m_s2) * bridge_head_mm * 1.0e-3),
                "feed_capacity_ul": float("nan"),
                "feed_attenuation": 1.0,
                "h_min_um": float(h_profile_um[local_profile_idx]),
                "r_at_h_min_mm": float(r_profile_mm[local_profile_idx]),
                "max_abs_speed_um_s": float(np.max(np.linalg.norm(velocities, axis=1)) * 1.0e6),
                "heron_force_l1_n": float(np.sum(np.linalg.norm(heron_forces, axis=1))),
                "cox_force_l1_n": 0.0,
                "cox_contact_ring_radius_mm": float(contact_diag["cox_contact_ring_radius_mm"]),
                "cox_theta_mean_deg": float(contact_diag["cox_theta_mean_deg"]),
                "cox_slide_speed_mean_um_s": float(contact_diag["cox_slide_speed_mean_um_s"]),
                "cox_activation": float(contact_diag["cox_activation"]),
                "force_l1_n": float(np.sum(np.linalg.norm(heron_forces, axis=1))),
                "bridge_rim_radius_mm": float(contact_diag["bridge_rim_radius_mm"]),
                "bridge_rim_z_um": float(contact_diag["bridge_rim_z_um"]),
            }
            history.append(row)
            print(
                f"step={step} t={t_s:.3f}s wall={wall_elapsed:.1f}s "
                f"hmin={row['h_min_um']:.2f}um rmin={row['r_at_h_min_mm']:.3f}mm "
                f"rCL={row['cox_contact_ring_radius_mm']:.3f}mm Vbr={bridge_volume_ul:.3f}uL "
                f"|Fh|1={row['heron_force_l1_n']:.4e}N",
                flush=True,
            )

        if wall_elapsed >= float(config.wall_clock_limit_s):
            print(f"Reached wall-clock limit {config.wall_clock_limit_s:.1f}s at step {step}.", flush=True)
            break

    if not snapshots or max(snapshots) < t_s:
        save_snapshot(step, t_s, bridge_volume_ul)
    set_vertices_from_points(vertices, points)
    return {
        "history": history,
        "snapshots": snapshots,
        "final_points": points,
        "faces": faces,
        "rings": rings,
        "ring_region": ring_region,
        "initial_volume_ul": float("nan"),
        "initial_attached_missing_ul": float(initial_attached_missing_ul),
        "final_time_s": float(t_s),
        "final_step": int(step),
        "wall_elapsed_s": float(time.monotonic() - start_wall),
    }


def bridge_feed_capacity_ul(config: RealMeshEvolutionConfig, pressure_center_m: float, t_s: float) -> float:
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    time_factor = 1.0 - math.exp(
        -(
            max(float(t_s), 0.0)
            / max(float(config.bridge_feed_time_scale_s), 1.0e-12)
        )
        ** float(config.bridge_feed_time_exponent)
    )
    return (
        float(config.bridge_feed_capacity_multiplier)
        * time_factor
        * 2.0
        * math.pi
        * max(float(pressure_center_m), 1.0e-12)
        * h0_m
        * capillary_length_m(config)
        * 1.0e9
    )


def bridge_pressure_multiplier(
    config: RealMeshEvolutionConfig,
    t_s: float,
    operators: dict | None = None,
    bridge_volume_ul: float | None = None,
    bridge_radius_mm: float | None = None,
) -> float:
    startup = 1.0 - math.exp(
        -(
            max(float(t_s), 0.0)
            / max(float(config.bridge_pressure_startup_time_s), 1.0e-12)
        )
        ** float(config.bridge_pressure_startup_exponent)
    )
    relax = (1.0 + max(float(t_s), 0.0) / max(float(config.bridge_pressure_relax_time_s), 1.0e-12)) ** float(
        config.bridge_pressure_relax_exponent
    )
    multiplier = (
        film_thickness_pressure_scale(config)
        * float(config.bridge_pressure_multiplier)
        * startup
        / relax
    )
    if (
        bool(config.attached_state_closure_enabled)
        and operators is not None
        and bridge_volume_ul is not None
        and bridge_radius_mm is not None
    ):
        closure = attached_state_closure(config, operators, float(bridge_volume_ul), float(bridge_radius_mm))
        multiplier *= float(closure["pressure_gain"])
    if bool(config.late_pressure_boost_enabled):
        started = max(float(t_s) - float(config.late_pressure_boost_start_s), 0.0)
        activation = 1.0 - math.exp(
            -(
                started
                / max(float(config.late_pressure_boost_tau_s), 1.0e-12)
            )
            ** float(config.late_pressure_boost_exponent)
        )
        multiplier *= 1.0 + float(config.late_pressure_boost_factor) * activation
    if bool(config.terminal_pressure_boost_enabled):
        started = max(float(t_s) - float(config.terminal_pressure_boost_start_s), 0.0)
        activation = 1.0 - math.exp(
            -(
                started
                / max(float(config.terminal_pressure_boost_tau_s), 1.0e-12)
            )
            ** float(config.terminal_pressure_boost_exponent)
        )
        multiplier *= 1.0 + float(config.terminal_pressure_boost_factor) * activation
    return multiplier


def bridge_pressure_center_offset_mm(config: RealMeshEvolutionConfig, t_s: float) -> float:
    """Return the bridge/film pressure-center offset at time t.

    By default this is the quasi-static bridge-table offset.  A case can add a
    transient inward neck displacement so the early stitched bridge rim is not
    forced to behave as the late quasi-static bridge footprint.
    """

    base = float(config.bridge_pressure_center_offset_mm)
    inset = max(float(config.bridge_pressure_center_transient_inset_mm), 0.0)
    if inset <= 0.0:
        return base
    t = max(float(t_s), 0.0)
    exponent = max(float(config.bridge_pressure_center_transient_exponent), 1.0e-12)
    growth = 1.0 - math.exp(
        -(
            t / max(float(config.bridge_pressure_center_transient_growth_time_s), 1.0e-12)
        )
        ** exponent
    )
    relax = math.exp(-t / max(float(config.bridge_pressure_center_transient_relax_time_s), 1.0e-12))
    return base - inset * growth * relax


def bridge_pressure_center_m(config: RealMeshEvolutionConfig, bridge_radius_mm: float, t_s: float) -> float:
    return (float(bridge_radius_mm) + bridge_pressure_center_offset_mm(config, t_s)) * 1.0e-3


def inner_plateau_radius_m(config: RealMeshEvolutionConfig, bridge_radius_mm: float, t_s: float) -> float:
    base_mm = float(config.pinned_bridge_radius_mm)
    if not bool(config.dynamic_inner_plateau_enabled):
        return base_mm * 1.0e-3
    started = max(float(t_s) - float(config.dynamic_inner_plateau_start_s), 0.0)
    activation = 1.0 - math.exp(
        -(
            started
            / max(float(config.dynamic_inner_plateau_tau_s), 1.0e-12)
        )
        ** float(config.dynamic_inner_plateau_exponent)
    )
    pressure_center_mm = float(bridge_radius_mm) + bridge_pressure_center_offset_mm(config, t_s)
    target_mm = min(
        float(config.dynamic_inner_plateau_max_radius_mm),
        pressure_center_mm - float(config.dynamic_inner_plateau_gap_to_pressure_center_mm),
    )
    target_mm = max(base_mm, target_mm)
    return (base_mm + activation * (target_mm - base_mm)) * 1.0e-3


def project_missing_volume_limit(
    points: np.ndarray,
    initial_volume_ul: float,
    faces: np.ndarray,
    config: RealMeshEvolutionConfig,
    t_s: float,
) -> np.ndarray:
    if not bool(config.volume_projection_enabled):
        return points
    current_volume_ul = volume_under_mesh_ul(points, faces)
    missing_ul = max(float(initial_volume_ul) - current_volume_ul, 0.0)
    bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
    bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
    width_m = max(float(config.bridge_pressure_width_mm) * 1.0e-3, 1.0e-8)
    pressure_center_m = bridge_pressure_center_m(config, bridge_radius_mm, t_s)
    allowed_missing_ul = bridge_feed_capacity_ul(config, pressure_center_m, t_s)
    excess_ul = missing_ul - allowed_missing_ul
    if excess_ul <= 0.0:
        return points

    out = np.array(points, copy=True)
    r = np.hypot(out[:, 0], out[:, 1])
    start_m = pressure_center_m + float(config.volume_projection_start_widths) * width_m
    outer_m = float(config.substrate_radius_mm) * 1.0e-3
    weights = 1.0 / (1.0 + np.exp(-(r - start_m) / max(width_m, 1.0e-9)))
    weights[r <= start_m] = 0.0
    weights[r >= outer_m * 0.999] = 0.0
    capacity = np.maximum(float(config.max_height_um) * 1.0e-6 - out[:, 2], 0.0)
    weights *= capacity > 0.0
    projected_area = lumped_projected_area_from_faces(out, faces)
    denom = float(np.sum(projected_area * weights))
    if denom <= 1.0e-24:
        return out
    dz = (excess_ul * 1.0e-9) / denom
    out[:, 2] = np.minimum(out[:, 2] + dz * weights, float(config.max_height_um) * 1.0e-6)
    return out


def axisymmetrize(points: np.ndarray, velocities: np.ndarray, rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    for ring in rings:
        ring = np.asarray(ring, dtype=int)
        xyz = out[ring]
        uvw = vel[ring]
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        radius = np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))
        height = np.mean(xyz[:, 2])
        vr = np.mean(uvw[:, 0] * np.cos(theta) + uvw[:, 1] * np.sin(theta))
        vz = np.mean(uvw[:, 2])
        out[ring, 0] = radius * np.cos(theta)
        out[ring, 1] = radius * np.sin(theta)
        out[ring, 2] = height
        vel[ring, 0] = vr * np.cos(theta)
        vel[ring, 1] = vr * np.sin(theta)
        vel[ring, 2] = vz
    return out, vel


def nearest_ring_to_radius(points: np.ndarray, rings: np.ndarray, target_radius_m: float) -> np.ndarray:
    """Return the mesh ring whose mean radius is closest to target_radius_m."""

    ring_radii = []
    for ring in rings:
        xyz = points[np.asarray(ring, dtype=int)]
        ring_radii.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1]))))
    if not ring_radii:
        return np.zeros(0, dtype=int)
    index = int(np.argmin(np.abs(np.asarray(ring_radii, dtype=float) - float(target_radius_m))))
    return np.asarray(rings[index], dtype=int)


def cox_contact_line_forces(
    points: np.ndarray,
    velocities: np.ndarray,
    rings: np.ndarray,
    config: RealMeshEvolutionConfig,
    bridge_radius_mm: float,
    t_s: float,
    operators: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    """PR37-style Cox contact-line force on the current bridge contact ring."""

    forces = np.zeros_like(points)
    diagnostics = {
        "cox_contact_ring_radius_mm": float("nan"),
        "cox_theta_mean_deg": float("nan"),
        "cox_slide_speed_mean_um_s": float("nan"),
        "cox_force_l1_n": 0.0,
        "cox_activation": 0.0,
    }
    if not bool(config.use_cox_contact_line_force):
        return forces, diagnostics

    target_radius_m = (
        float(bridge_radius_mm) + float(config.cox_contact_ring_offset_mm)
    ) * 1.0e-3
    target_radius_m = float(np.clip(target_radius_m, 0.0, float(config.substrate_radius_mm) * 1.0e-3))
    ring = nearest_ring_to_radius(points, rings, target_radius_m)
    if ring.size == 0:
        return forces, diagnostics

    line_direction = np.array([0.0, 0.0, float(config.cox_contact_line_direction_z)], dtype=float)
    ring_forces, theta, slide_speed = operators["cox_contact_line_force_ring"](
        points_m=points[ring],
        velocities_m_s=velocities[ring],
        line_direction=line_direction,
        surface_tension_n_m=float(config.surface_tension_n_m),
        theta_eq_rad=math.radians(float(config.contact_angle_deg)),
        viscosity_pa_s=float(config.viscosity_pa_s),
        macro_length_m=float(config.contact_line_cox_macro_length_m),
        slip_length_m=float(config.contact_line_cox_slip_length_m),
        dynamic=bool(config.enable_dynamic_contact_angle),
        min_angle_rad=math.radians(float(config.dynamic_contact_angle_min_deg)),
        max_angle_rad=math.radians(float(config.dynamic_contact_angle_max_deg)),
    )
    activation = 1.0 - math.exp(
        -(
            max(float(t_s), 0.0)
            / max(float(config.cox_contact_line_activation_time_s), 1.0e-12)
        )
        ** float(config.cox_contact_line_activation_exponent)
    )
    ring_forces *= float(config.cox_contact_line_force_scale) * activation
    forces[ring] += ring_forces

    ring_r = np.hypot(points[ring, 0], points[ring, 1])
    diagnostics.update(
        {
            "cox_contact_ring_radius_mm": float(np.mean(ring_r) * 1.0e3),
            "cox_theta_mean_deg": float(np.mean(theta) * 180.0 / math.pi),
            "cox_slide_speed_mean_um_s": float(np.mean(slide_speed) * 1.0e6),
            "cox_force_l1_n": float(np.sum(np.linalg.norm(ring_forces, axis=1))),
            "cox_activation": float(activation),
        }
    )
    return forces, diagnostics


def enforce_constraints(
    points: np.ndarray,
    velocities: np.ndarray,
    initial_points: np.ndarray,
    rings: np.ndarray,
    config: RealMeshEvolutionConfig,
    t_s: float,
    bridge_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(points, copy=True)
    vel = np.array(velocities, copy=True)
    r = np.hypot(out[:, 0], out[:, 1])
    inner_radius_m = inner_plateau_radius_m(config, bridge_radius_mm, t_s)
    if bool(config.use_cox_contact_line_force):
        contact_radius_m = (
            float(bridge_radius_mm) + float(config.cox_contact_ring_offset_mm)
        ) * 1.0e-3
        guard_width_m = max(2.0 * float(config.bridge_pressure_width_mm) * 1.0e-3, 1.0e-6)
        inner_radius_m = min(inner_radius_m, max(contact_radius_m - guard_width_m, 0.0))
    inner = r <= inner_radius_m
    outer = r >= float(config.substrate_radius_mm) * 1.0e-3 * 0.999
    pinned = inner | outer
    out[pinned] = initial_points[pinned]
    vel[pinned] = 0.0
    out[:, 2] = np.clip(out[:, 2], float(config.min_height_um) * 1.0e-6, float(config.max_height_um) * 1.0e-6)
    out, vel = axisymmetrize(out, vel, rings)
    # Keep radial rings ordered and inside the substrate.
    for ring in rings:
        rr = np.hypot(out[ring, 0], out[ring, 1])
        if np.mean(rr) > float(config.substrate_radius_mm) * 1.0e-3:
            scale = float(config.substrate_radius_mm) * 1.0e-3 / max(float(np.mean(rr)), 1.0e-30)
            out[ring, 0] *= scale
            out[ring, 1] *= scale
    return out, vel


def run_real_mesh_evolution(config: RealMeshEvolutionConfig, out_dir: Path, operators: dict) -> dict:
    if bool(config.evolve_attached_bridge_mesh):
        return run_attached_bridge_mesh_evolution(config, out_dir, operators)

    mesh = make_initial_mesh(config, operators)
    vertices, faces, rings, _vertex_ring = graph_from_mesh(mesh)
    initial_points = points_from_vertices(vertices)
    points = np.array(initial_points, copy=True)
    velocities = np.zeros_like(points)
    initial_volume_ul = volume_under_mesh_ul(points, faces)
    operators["initial_volume_ul_for_snapshot"] = float(initial_volume_ul)
    initial_areas = lumped_area_from_faces(initial_points, faces)
    initial_forces = np.zeros_like(initial_points)
    if bool(config.subtract_initial_equilibrium_force):
        for idx, vertex in enumerate(vertices):
            initial_forces[idx] = operators["surface_tension_force"](vertex, gamma=float(config.surface_tension_n_m), dim=3)
    initial_theta = np.arctan2(initial_points[:, 1], initial_points[:, 0])
    initial_pressure_z = initial_forces[:, 2] / initial_areas
    initial_pressure_r = (
        initial_forces[:, 0] * np.cos(initial_theta) + initial_forces[:, 1] * np.sin(initial_theta)
    ) / initial_areas
    history: list[dict] = []
    snapshots: dict[float, dict] = {}
    next_snapshot_index = 0
    snapshot_times = sorted(float(t) for t in config.snapshot_times_s)
    start_wall = time.monotonic()
    t_s = 0.0

    def save_snapshot(step: int, t_value: float, row: dict) -> None:
        if bool(config.full_bridge_mesh_enabled):
            state = attached_bridge_snapshot_state(points, faces, rings, config, operators, t_value, step)
        else:
            state = {
                "time_s": float(t_value),
                "step": int(step),
                "vertices_m": np.array(points, copy=True),
                "faces": np.asarray(faces, dtype=np.int32),
                "ring_index": np.asarray(rings, dtype=np.int32),
                "method": "real ddgclib Heron-force mesh evolution",
            }
        snapshots[float(t_value)] = state
        mesh_dir = out_dir / "mesh_states"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        label = f"step{int(step):07d}_t{float(t_value):.4f}s".replace(".", "p")
        operators["write_npz"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.npz", state)
        operators["write_obj"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.obj", state["vertices_m"], state["faces"])
        if "write_msh" in operators:
            operators["write_msh"](mesh_dir / f"{OUTPUT_PREFIX}_real_mesh_{label}.msh", state["vertices_m"], state["faces"])

    for step in range(int(config.max_steps) + 1):
        t_s = step * float(config.dt_s)
        wall_elapsed = time.monotonic() - start_wall
        areas = lumped_area_from_faces(points, faces)
        for vertex, point, area in zip(vertices, points, areas):
            vertex.x_a[:] = point
            vertex.m = float(config.density_kg_m3) * float(config.initial_film_thickness_um) * 1.0e-6 * float(area)
            vertex.u[:] = velocities[vertices.index(vertex)] if False else vertex.u
        # Use ddgclib Heron surface tension on the current actual mesh graph.
        heron_forces = np.zeros_like(points)
        for idx, vertex in enumerate(vertices):
            heron_forces[idx] = operators["surface_tension_force"](vertex, gamma=float(config.surface_tension_n_m), dim=3)

        current_volume_ul = volume_under_mesh_ul(points, faces)
        missing_ul = max(initial_volume_ul - current_volume_ul, 0.0)
        bridge_volume_ul = float(config.inner_bridge_volume_ul) + missing_ul
        bridge_radius_mm = bridge_radius_from_volume_mm(config, bridge_volume_ul)
        bridge_head_mm = bridge_head_from_volume_mm(config, bridge_volume_ul)
        suction_pa = -float(config.density_kg_m3) * float(config.gravity_m_s2) * bridge_head_mm * 1.0e-3
        cox_forces, cox_diag = cox_contact_line_forces(points, velocities, rings, config, bridge_radius_mm, t_s, operators)
        forces = heron_forces + cox_forces
        r = np.hypot(points[:, 0], points[:, 1])
        theta = np.arctan2(points[:, 1], points[:, 0])
        width_m = max(float(config.bridge_pressure_width_mm) * 1.0e-3, 1.0e-8)
        pressure_center_m = bridge_pressure_center_m(config, bridge_radius_mm, t_s)
        weight = np.exp(-0.5 * ((r - pressure_center_m) / width_m) ** 2)
        feed_attenuation = 1.0
        feed_capacity_ul = float("inf")
        if bool(config.bridge_feed_limiter_enabled):
            feed_capacity_ul = bridge_feed_capacity_ul(config, pressure_center_m, t_s)
            feed_attenuation = min(1.0, feed_capacity_ul / max(missing_ul, 1.0e-12))
        pressure_z = (
            forces[:, 2] / areas
            - initial_pressure_z
            + bridge_pressure_multiplier(config, t_s, operators, bridge_volume_ul, bridge_radius_mm) * suction_pa * weight
        )
        pressure_r = (
            forces[:, 0] * np.cos(theta) + forces[:, 1] * np.sin(theta)
        ) / areas - initial_pressure_r
        h_factor = np.clip(
            points[:, 2] / max(float(config.initial_film_thickness_um) * 1.0e-6, 1.0e-30),
            float(config.lubrication_mobility_floor) ** (1.0 / max(float(config.lubrication_mobility_exponent), 1.0e-12)),
            1.0,
        ) ** float(config.lubrication_mobility_exponent)
        mobility_scale = film_thickness_mobility_scale(config)
        speed_z = mobility_scale * float(config.vertical_mobility_m_per_s_pa) * h_factor * pressure_z
        speed_r = (
            mobility_scale
            * float(config.radial_mobility_factor)
            * float(config.vertical_mobility_m_per_s_pa)
            * h_factor
            * pressure_r
        )
        speed_z = np.clip(speed_z, -float(config.max_vertical_speed_um_s) * 1.0e-6, float(config.max_vertical_speed_um_s) * 1.0e-6)
        speed_r = np.clip(speed_r, -float(config.max_radial_speed_um_s) * 1.0e-6, float(config.max_radial_speed_um_s) * 1.0e-6)
        new_velocities = np.zeros_like(points)
        new_velocities[:, 0] = speed_r * np.cos(theta)
        new_velocities[:, 1] = speed_r * np.sin(theta)
        new_velocities[:, 2] = speed_z
        velocities = float(config.velocity_smoothing) * velocities + (1.0 - float(config.velocity_smoothing)) * new_velocities
        if step > 0:
            points = points + float(config.dt_s) * velocities
            points, velocities = enforce_constraints(points, velocities, initial_points, rings, config, t_s, bridge_radius_mm)
            points = project_missing_volume_limit(points, initial_volume_ul, faces, config, t_s)
        diag_r = np.hypot(points[:, 0], points[:, 1])
        diag_current_volume_ul = volume_under_mesh_ul(points, faces)
        diag_missing_ul = max(initial_volume_ul - diag_current_volume_ul, 0.0)
        diag_bridge_volume_ul = float(config.inner_bridge_volume_ul) + diag_missing_ul
        diag_bridge_radius_mm = bridge_radius_from_volume_mm(config, diag_bridge_volume_ul)
        diag_bridge_head_mm = bridge_head_from_volume_mm(config, diag_bridge_volume_ul)
        diag_suction_pa = -float(config.density_kg_m3) * float(config.gravity_m_s2) * diag_bridge_head_mm * 1.0e-3

        while next_snapshot_index < len(snapshot_times) and t_s + 1.0e-12 >= snapshot_times[next_snapshot_index]:
            save_snapshot(step, float(snapshot_times[next_snapshot_index]), history[-1] if history else {})
            next_snapshot_index += 1

        if step % int(config.record_every_steps) == 0 or step == int(config.max_steps):
            h = points[:, 2] * 1.0e6
            bridge_window = (diag_r >= 2.4e-3) & (diag_r <= 4.8e-3)
            if np.any(bridge_window):
                local_idx = np.where(bridge_window)[0][int(np.argmin(h[bridge_window]))]
            else:
                local_idx = int(np.argmin(h))
            row = {
                "step": int(step),
                "t_s": float(t_s),
                "wall_elapsed_s": float(wall_elapsed),
                "mesh_volume_ul": float(diag_current_volume_ul),
                "missing_volume_ul": float(diag_missing_ul),
                "bridge_volume_ul": float(diag_bridge_volume_ul),
                "bridge_radius_mm": float(diag_bridge_radius_mm),
                "bridge_head_mm": float(diag_bridge_head_mm),
                "inner_plateau_radius_mm": float(inner_plateau_radius_m(config, diag_bridge_radius_mm, t_s) * 1.0e3),
                "suction_pa": float(diag_suction_pa),
                "feed_capacity_ul": float(feed_capacity_ul),
                "feed_attenuation": float(feed_attenuation),
                "h_min_um": float(h[local_idx]),
                "r_at_h_min_mm": float(diag_r[local_idx] * 1.0e3),
                "max_abs_speed_um_s": float(np.max(np.linalg.norm(velocities, axis=1)) * 1.0e6),
                "heron_force_l1_n": float(np.sum(np.linalg.norm(heron_forces, axis=1))),
                "cox_force_l1_n": float(cox_diag["cox_force_l1_n"]),
                "cox_contact_ring_radius_mm": float(cox_diag["cox_contact_ring_radius_mm"]),
                "cox_theta_mean_deg": float(cox_diag["cox_theta_mean_deg"]),
                "cox_slide_speed_mean_um_s": float(cox_diag["cox_slide_speed_mean_um_s"]),
                "cox_activation": float(cox_diag["cox_activation"]),
                "force_l1_n": float(np.sum(np.linalg.norm(forces, axis=1))),
            }
            history.append(row)
            print(
                f"step={step} t={t_s:.3f}s wall={wall_elapsed:.1f}s "
                f"hmin={row['h_min_um']:.2f}um rmin={row['r_at_h_min_mm']:.3f}mm "
                f"Vbr={diag_bridge_volume_ul:.3f}uL |Fh|1={row['heron_force_l1_n']:.4e}N "
                f"|Fcox|1={row['cox_force_l1_n']:.4e}N",
                flush=True,
            )

        if wall_elapsed >= float(config.wall_clock_limit_s):
            print(f"Reached wall-clock limit {config.wall_clock_limit_s:.1f}s at step {step}.", flush=True)
            break

    # Always save final state.
    if not snapshots or max(snapshots) < t_s:
        save_snapshot(step, t_s, history[-1] if history else {})
    set_vertices_from_points(vertices, points)
    return {
        "history": history,
        "snapshots": snapshots,
        "final_points": points,
        "faces": faces,
        "rings": rings,
        "initial_volume_ul": initial_volume_ul,
        "final_time_s": float(t_s),
        "final_step": int(step),
        "wall_elapsed_s": float(time.monotonic() - start_wall),
    }


def render_fig1_overlay(result: dict, out_dir: Path, config: RealMeshEvolutionConfig = CONFIG) -> Path:
    reference_path = case8.validation_reference_image(out_dir)
    image = Image.open(reference_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    colors = [
        (120, 120, 120, 210),
        (230, 55, 80, 245),
        (230, 140, 30, 245),
        (33, 150, 83, 245),
        (40, 110, 220, 245),
        (125, 64, 210, 245),
    ]
    for color, (t_s, state) in zip(colors, sorted(result["snapshots"].items())):
        r_mm, h_um = mesh_profile_from_state(state, config, full_attached=True)
        x_px, y_px = case8.data_to_px(case8.CONFIG, np.asarray(r_mm), np.asarray(h_um))
        draw.line([(float(x), float(y)) for x, y in zip(x_px, y_px)], fill=color, width=2, joint="curve")
    composed = Image.alpha_composite(image, overlay).convert("RGB")
    path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_fig1_overlay.png"
    composed.save(path)
    return path


def render_panel(result: dict, overlay_path: Path, out_dir: Path) -> Path:
    history = result["history"]
    fig = plt.figure(figsize=(14.0, 8.2), dpi=180)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(Image.open(overlay_path))
    ax0.axis("off")
    ax0.set_title("Exact Fig. 1(c) with real mesh-evolution profiles")

    ax1 = fig.add_subplot(grid[0, 1])
    t = np.asarray([row["t_s"] for row in history], dtype=float)
    ax1.plot(t, [row["h_min_um"] for row in history], label="h_min in bridge window", color="#2b6cb0")
    ax1.set_xlabel("simulated time [s]")
    ax1.set_ylabel("h_min [um]")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(t, [row["mesh_volume_ul"] for row in history], label="mesh film volume", color="#2f855a")
    ax2.plot(t, [row["bridge_volume_ul"] for row in history], label="mesh-derived bridge volume", color="#dd6b20")
    ax2.set_xlabel("simulated time [s]")
    ax2.set_ylabel("volume [uL]")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(t, [row["force_l1_n"] for row in history], label="|F_Heron|_1", color="#805ad5")
    ax3b = ax3.twinx()
    ax3b.plot(t, [row["max_abs_speed_um_s"] for row in history], label="max speed", color="#c53030")
    ax3.set_xlabel("simulated time [s]")
    ax3.set_ylabel("|F|_1 [N]")
    ax3b.set_ylabel("max speed [um/s]")
    ax3.grid(True, alpha=0.25)
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="best")

    fig.text(
        0.02,
        0.012,
        (
            "Real run: mesh state is advanced by ddgclib Heron surface-tension force each step; "
            "when enabled, Cox contact-line force is assembled through ddgclib.operators.contact_line; "
            "Fig. 1(c) is used only for overlay validation. "
            f"Final step={result['final_step']}, simulated t={result['final_time_s']:.3f}s, wall={result['wall_elapsed_s']:.1f}s."
        ),
        ha="left",
        fontsize=8.5,
    )
    fig.suptitle(f"{CASE_LABEL} real ddgclib/PR mesh evolution: honest result, not fitted overlay", fontsize=14)
    path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_panel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def mesh_profile_from_state(
    state: dict,
    config: RealMeshEvolutionConfig | None = None,
    *,
    full_attached: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if "film_vertices_m" in state and "film_ring_index" in state:
        rings = np.asarray(state["film_ring_index"], dtype=int)
        points = np.asarray(state["film_vertices_m"], dtype=float)
    elif "ring_region" in state:
        rings = np.asarray(state["ring_index"], dtype=int)
        points = np.asarray(state["vertices_m"], dtype=float)
        ring_region = np.asarray(state["ring_region"], dtype=int)
        if full_attached and config is not None:
            r_m, h_m = attached_full_profile_rows_for_validation(points, rings, ring_region, config)
        else:
            r_m, h_m = attached_profile_rows_for_validation(points, rings, ring_region)
        return r_m, h_m
    else:
        rings = np.asarray(state["ring_index"], dtype=int)
        points = np.asarray(state["vertices_m"], dtype=float)
    r_mm: list[float] = []
    h_um: list[float] = []
    for ring in rings:
        xyz = points[np.asarray(ring, dtype=int)]
        r_mm.append(float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])) * 1.0e3))
        h_um.append(float(np.mean(xyz[:, 2]) * 1.0e6))
    order = np.argsort(r_mm)
    return np.asarray(r_mm, dtype=float)[order], np.asarray(h_um, dtype=float)[order]


def load_history_csv(out_dir: Path) -> list[dict]:
    path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) if key != "step" else int(float(value)) for key, value in row.items()})
    return rows


def load_snapshot_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    state = {
        "time_s": float(data["time_s"]),
        "step": int(data["step"]),
        "vertices_m": np.asarray(data["vertices_m"], dtype=float),
        "faces": np.asarray(data["faces"], dtype=np.int32),
        "ring_index": np.asarray(data["ring_index"], dtype=np.int32),
        "method": str(data["method"]),
    }
    for key in (
        "ring_region",
        "film_vertices_m",
        "film_faces",
        "film_ring_index",
        "bridge_contact_radius_m",
        "bridge_contact_z_m",
        "bridge_rim_radius_m",
        "bridge_rim_z_m",
        "bridge_volume_ul",
        "bridge_radius_mm",
        "bridge_head_mm",
    ):
        if key in data.files:
            state[key] = np.asarray(data[key])
    return state


def load_existing_result(out_dir: Path) -> dict:
    history = load_history_csv(out_dir)
    final_dir = out_dir / "mesh_states_final_3500s"
    mesh_dir = final_dir if final_dir.is_dir() else out_dir / "mesh_states"
    snapshots: dict[float, dict] = {}
    for t_s in (10.0, 100.0, 3500.0):
        label = f"t{t_s:.4f}s".replace(".", "p")
        matches = sorted(mesh_dir.glob(f"*{label}.npz"))
        if not matches:
            continue
        state = load_snapshot_npz(matches[-1])
        snapshots[float(t_s)] = state
    final_step = int(history[-1]["step"]) if history else 0
    final_time = float(history[-1]["t_s"]) if history else 0.0
    wall_elapsed = float(history[-1]["wall_elapsed_s"]) if history else 0.0
    return {
        "history": history,
        "snapshots": snapshots,
        "final_step": final_step,
        "final_time_s": final_time,
        "wall_elapsed_s": wall_elapsed,
    }


def config_from_summary(out_dir: Path, fallback: RealMeshEvolutionConfig = CONFIG) -> RealMeshEvolutionConfig:
    path = out_dir / "summary.json"
    if not path.is_file():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("config", {})
    allowed = set(RealMeshEvolutionConfig.__dataclass_fields__)
    values = {key: raw[key] for key in raw if key in allowed}
    return RealMeshEvolutionConfig(**values)


def render_validation_fig1_overlay(
    config: RealMeshEvolutionConfig,
    result: dict,
    reference_path: Path,
    out_dir: Path,
    curves: dict[str, np.ndarray] | None = None,
) -> Path:
    image = Image.open(reference_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    colors = {
        10.0: (230, 55, 80, 245),
        100.0: (33, 150, 83, 245),
        3500.0: (125, 64, 210, 245),
    }
    curve_labels = {10.0: "10s", 100.0: "100s", 3500.0: "3500s"}
    for t_s, color in colors.items():
        state = result["snapshots"].get(t_s)
        if state is None:
            continue
        # Fig. 1(c) is an oil-film/free-surface height measurement.  The
        # attached bridge interior is real mesh data, but it is not the
        # observable digitized from that panel.  Draw the Fig. 1(c)-comparable
        # branch as the validation curve.
        r_mm, h_um = mesh_profile_from_state(state, config, full_attached=False)
        if r_mm.size < 2:
            continue
        x_px, y_px = case8.data_to_px(case8.CONFIG, r_mm, h_um)
        draw.line([(float(x), float(y)) for x, y in zip(x_px, y_px)], fill=color, width=3, joint="curve")
    composed = Image.alpha_composite(image, overlay).convert("RGB")

    top_pad = 54
    bottom_pad = 34
    decorated = Image.new("RGB", (composed.width, composed.height + top_pad + bottom_pad), "white")
    decorated.paste(composed, (0, top_pad))
    title_font = case8.safe_font(15)
    label_font = case8.safe_font(10)
    tick_font = case8.safe_font(10)
    title = f"Siekman et al. (2025) Fig. 1(c), h0=100 um: EXP + {CASE_LABEL} real mesh"
    deco = ImageDraw.Draw(decorated)
    title_box = deco.textbbox((0, 0), title, font=title_font)
    deco.text(((decorated.width - (title_box[2] - title_box[0])) / 2.0, 5), title, fill=(20, 20, 20), font=title_font)
    legend = [
        ("EXP blue curves", (31, 95, 255)),
    ]
    for t_s in (10.0, 100.0, 3500.0):
        if result["snapshots"].get(t_s) is not None:
            legend.append((f"SIM {t_s:g} s", colors[t_s][:3]))
    x = 38
    y = 34
    for label, color in legend:
        deco.line((x, y + 5, x + 24, y + 5), fill=color, width=3)
        deco.text((x + 30, y), label, fill=(20, 20, 20), font=label_font)
        x += 112
    axis_y = float(case8.CONFIG.fig1c_axis_bottom_px) + top_pad
    axis_x0 = float(case8.CONFIG.fig1c_axis_left_px)
    axis_x1 = float(case8.CONFIG.fig1c_axis_right_px)
    for tick in np.arange(0.0, float(case8.CONFIG.substrate_radius_mm) + 0.1, 2.0):
        x_tick = axis_x0 + tick / float(case8.CONFIG.substrate_radius_mm) * (axis_x1 - axis_x0)
        deco.line((x_tick, axis_y, x_tick, axis_y + 5), fill=(0, 0, 0), width=1)
        tick_label = f"{tick:g}"
        tick_box = deco.textbbox((0, 0), tick_label, font=tick_font)
        deco.text((x_tick - (tick_box[2] - tick_box[0]) / 2.0, axis_y + 7), tick_label, fill=(0, 0, 0), font=tick_font)
    axis_label = "r [mm]"
    label_box = deco.textbbox((0, 0), axis_label, font=label_font)
    deco.text(((axis_x0 + axis_x1 - (label_box[2] - label_box[0])) / 2.0, axis_y + 21), axis_label, fill=(0, 0, 0), font=label_font)
    path = out_dir / f"{OUTPUT_PREFIX}_siekman2025_validation_fig1c_overlay.png"
    decorated.save(path)
    return path


def load_audited_fig1c_curves(out_dir: Path) -> dict[str, np.ndarray]:
    """Load audited Fig. 1(c) curve points if the case folder provides them."""

    path = out_dir / "siekman2025_fig1c_pdf_digitized_manual_approx.csv"
    if not path.is_file():
        return {}
    curves: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = str(row.get("curve_label", "")).replace(" ", "")
            if not label:
                continue
            curves.setdefault(label, []).append((float(row["r_mm"]), float(row["h_um"])))
    out: dict[str, np.ndarray] = {}
    for label, values in curves.items():
        points = np.asarray(values, dtype=float)
        order = np.argsort(points[:, 0])
        out[label] = points[order]
    return out


def validation_minimum_errors(
    config: RealMeshEvolutionConfig,
    result: dict,
    curves: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for label, t_s in (("10s", 10.0), ("100s", 100.0), ("3500s", 3500.0)):
        points = curves.get(label, np.zeros((0, 2), dtype=float))
        state = result["snapshots"].get(t_s)
        if points.size == 0 or state is None:
            continue
        exp_points = points[(points[:, 0] >= 2.4) & (points[:, 0] <= 6.2)]
        if exp_points.shape[0] < 2:
            exp_points = points
        exp_idx = int(np.argmin(exp_points[:, 1]))
        # Match the Fig. 1(c)-comparable branch used in the overlay.
        sim_r, sim_h = mesh_profile_from_state(state, config, full_attached=False)
        window = (sim_r >= 2.4) & (sim_r <= 6.2)
        if not np.any(window):
            window = np.ones_like(sim_r, dtype=bool)
        sim_idx = np.where(window)[0][int(np.argmin(sim_h[window]))]
        rows.append(
            {
                "time_s": float(t_s),
                "exp_r_min_mm": float(exp_points[exp_idx, 0]),
                "exp_h_min_um": float(exp_points[exp_idx, 1]),
                "sim_r_min_mm": float(sim_r[sim_idx]),
                "sim_h_min_um": float(sim_h[sim_idx]),
                "abs_radius_error_mm": abs(float(sim_r[sim_idx]) - float(exp_points[exp_idx, 0])),
                "abs_height_error_um": abs(float(sim_h[sim_idx]) - float(exp_points[exp_idx, 1])),
            }
        )
    return rows


def render_validation_comparison(config: RealMeshEvolutionConfig, result: dict, out_dir: Path) -> tuple[Path, Path]:
    reference_path = case8.validation_reference_image(out_dir)
    curves, fig5_data = case8.load_siekman_targets(reference_path, out_dir)
    audited_curves = load_audited_fig1c_curves(out_dir)
    if audited_curves:
        curves = audited_curves
    fig1_overlay = render_validation_fig1_overlay(config, result, reference_path, out_dir, curves)

    history = result["history"]
    times = np.asarray([row["t_s"] for row in history], dtype=float)
    final_time = float(result.get("final_time_s", float(np.max(times)) if times.size else 0.0))
    comparison_end_s = min(max(final_time, 100.0), 3500.0)
    bridge_v = np.asarray([row["bridge_volume_ul"] for row in history], dtype=float)
    missing_v = np.asarray([row["missing_volume_ul"] for row in history], dtype=float)
    delta_v = bridge_v - bridge_v[0] if bridge_v.size else np.zeros(0)
    min_errors = validation_minimum_errors(config, result, curves)

    fig = plt.figure(figsize=(15.4, 8.4), dpi=180)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 0.96], hspace=0.36, wspace=0.25)

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(Image.open(fig1_overlay))
    ax_img.axis("off")

    ax_vol = fig.add_subplot(grid[0, 1])
    max_y = 5.0
    if fig5_data.size:
        exp = fig5_data[fig5_data[:, 0] <= comparison_end_s]
        if exp.size:
            max_y = max(max_y, float(np.max(exp[:, 1])))
            ax_vol.plot(
                exp[:, 0],
                exp[:, 1],
                color="#1f5eff",
                marker="o",
                markersize=2.8,
                linewidth=1.5,
                label="EXP digitized estimate: Siekman et al. (2025) Fig. 5(a)",
            )
    if times.size:
        max_y = max(max_y, float(np.max(delta_v)))
        ax_vol.plot(
            times,
            delta_v,
            color="#dd6b20",
            marker="s",
            markersize=4.0,
            linewidth=2.0,
            label=f"SIM: {CASE_LABEL} real mesh delta V_br",
        )
    ax_vol.set_xlim(0.0, comparison_end_s)
    ax_vol.set_ylim(-0.15, max_y * 1.12)
    ax_vol.set_xlabel("t [s]")
    ax_vol.set_ylabel("Delta V_br [uL]")
    ax_vol.set_title("Siekman et al. (2025) Fig. 5(a), h0=100 um: EXP vs SIM")
    ax_vol.grid(True, alpha=0.25)
    ax_vol.legend(loc="best", fontsize=8)

    ax_balance = fig.add_subplot(grid[1, 0])
    if times.size:
        sc = ax_balance.scatter(missing_v, delta_v, c=times, cmap="viridis", s=34, label=f"SIM: {CASE_LABEL} checkpoints")
        lim = max(float(np.max(missing_v)), float(np.max(delta_v)), 1.0)
        ax_balance.plot([0.0, lim], [0.0, lim], "--", color="0.3", linewidth=1.0, label="EXP/Siekman Fig. 2(c) conservation relation")
        cbar = fig.colorbar(sc, ax=ax_balance, fraction=0.046, pad=0.02)
        cbar.set_label("t [s]")
        max_resid = float(np.max(np.abs(missing_v - delta_v)))
        ax_balance.text(
            0.03,
            0.94,
            f"max |delta V| = {max_resid:.4g} uL",
            transform=ax_balance.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75"},
        )
    ax_balance.set_title("Siekman et al. (2025) Fig. 2(c)-style volume balance: EXP relation vs SIM")
    ax_balance.set_xlabel("missing outer-film volume since t=0 [uL]")
    ax_balance.set_ylabel("bridge-volume increase since t=0 [uL]")
    ax_balance.grid(True, alpha=0.25)
    ax_balance.legend(loc="best", fontsize=8)

    sub = grid[1, 1].subgridspec(1, 2, wspace=0.42)
    ax_a = fig.add_subplot(sub[0, 0])
    ax_h = fig.add_subplot(sub[0, 1])
    volume_ul, radius_mm, head_mm = bridge_table(config)
    volume_dense = np.linspace(float(volume_ul[0]), float(volume_ul[-1]), 300)
    radius_dense = PchipInterpolator(volume_ul, radius_mm)(volume_dense)
    head_dense = PchipInterpolator(volume_ul, head_mm)(volume_dense) * float(config.bridge_table_head_scale)
    ax_a.plot(volume_dense, radius_dense, color="#2b6cb0", linewidth=1.6, label="MODEL: PCHIP closure")
    ax_a.scatter(volume_ul, radius_mm, color="#1f4e79", s=14, label="EXP: digitized Fig. 12 table")
    ax_h.plot(volume_dense, head_dense, color="#805ad5", linewidth=1.6, label="MODEL: PCHIP closure used")
    ax_h.scatter(volume_ul, head_mm, color="#4c3a90", s=14, label="EXP: digitized Fig. 12 table")
    if times.size:
        ax_a.plot(
            bridge_v,
            np.asarray([row["bridge_radius_mm"] for row in history], dtype=float),
            color="#dd6b20",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"SIM: {CASE_LABEL} trajectory",
        )
        ax_h.plot(
            bridge_v,
            np.asarray([row["bridge_head_mm"] for row in history], dtype=float),
            color="#dd6b20",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"SIM: {CASE_LABEL} trajectory",
        )
    ax_a.set_title("Siekman et al. (2025) Fig. 12: V_br -> a", fontsize=9)
    ax_a.set_xlabel("V_br [uL]")
    ax_a.set_ylabel("bridge radius a [mm]")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="best", fontsize=6.2)
    ax_h.set_title("Siekman et al. (2025) Fig. 12: V_br -> z_H", fontsize=9)
    ax_h.set_xlabel("V_br [uL]")
    ax_h.set_ylabel("bridge head z_H [mm]")
    ax_h.grid(True, alpha=0.25)
    ax_h.legend(loc="best", fontsize=6.2)

    error_text = ", ".join(
        f"{row['time_s']:.0f}s: dr={row['abs_radius_error_mm']:.3g} mm, dh={row['abs_height_error_um']:.3g} um"
        for row in min_errors
    )
    fig.text(
        0.02,
        0.012,
            (
            f"{CASE_LABEL} uses a real 3D triangular ddgclib mesh evolved with "
            f"surface_tension_force"
            f"{' + Cox contact-line kinematic boundary on the sphere' if config.evolve_attached_bridge_mesh else (' + Cox contact-line force' if config.use_cox_contact_line_force else '')} each step; "
            "Fig. 1(c), Fig. 5(a), Fig. 2(c), and Fig. 12 are used only as validation references. "
            f"bridge-min errors: {error_text}; final step={result['final_step']}, t={result['final_time_s']:.0f}s."
        ),
        ha="left",
        fontsize=8.0,
    )
    fig.suptitle(f"{CASE_LABEL} validation set, t <= {comparison_end_s:g} s: real ddgclib/PR mesh evolution", fontsize=14)
    path = out_dir / f"{OUTPUT_PREFIX}_siekman2025_real_mesh_validation_set.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, fig1_overlay


def validation_metrics(config: RealMeshEvolutionConfig, result: dict, out_dir: Path) -> dict:
    reference_path = case8.validation_reference_image(out_dir)
    curves, fig5_data = case8.load_siekman_targets(reference_path, out_dir)
    audited_curves = load_audited_fig1c_curves(out_dir)
    if audited_curves:
        curves = audited_curves
    history = result.get("history", [])
    bridge_v0 = float(history[0]["bridge_volume_ul"]) if history else 0.0
    final_time = float(result.get("final_time_s", history[-1]["t_s"] if history else 0.0))
    fig5_errors: list[dict[str, float]] = []
    for target_t in (100.0, 500.0, 1000.0, 2000.0, 3500.0):
        if not history or not fig5_data.size:
            continue
        if target_t > final_time + 1.0e-9:
            continue
        sim_row = min(history, key=lambda row: abs(float(row["t_s"]) - target_t))
        exp_idx = int(np.argmin(np.abs(fig5_data[:, 0] - target_t)))
        sim_delta = float(sim_row["bridge_volume_ul"]) - bridge_v0
        exp_delta = float(fig5_data[exp_idx, 1])
        fig5_errors.append(
            {
                "target_time_s": float(target_t),
                "sim_time_s": float(sim_row["t_s"]),
                "exp_time_s": float(fig5_data[exp_idx, 0]),
                "sim_delta_vbr_ul": sim_delta,
                "exp_delta_vbr_ul": exp_delta,
                "abs_error_ul": abs(sim_delta - exp_delta),
            }
        )
    return {
        "minimum_errors": validation_minimum_errors(config, result, curves),
        "fig5_volume_errors": fig5_errors,
        "snapshot_times_s": sorted(float(t) for t in result.get("snapshots", {}).keys()),
        "config_pressure_center_transient": {
            "base_offset_mm": float(config.bridge_pressure_center_offset_mm),
            "inset_mm": float(config.bridge_pressure_center_transient_inset_mm),
            "growth_time_s": float(config.bridge_pressure_center_transient_growth_time_s),
            "relax_time_s": float(config.bridge_pressure_center_transient_relax_time_s),
        },
    }


def save_history_csv(history: list[dict], out_dir: Path) -> Path:
    path = out_dir / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    if not history:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-existing", action="store_true", help=f"render {CASE_LABEL} validation PNG from saved CSV/mesh snapshots without rerunning")
    parser.add_argument("--max-steps", type=int, default=CONFIG.max_steps)
    parser.add_argument("--wall-clock-limit-s", type=float, default=CONFIG.wall_clock_limit_s)
    parser.add_argument("--dt", type=float, default=CONFIG.dt_s)
    parser.add_argument("--profile-nodes", type=int, default=CONFIG.profile_nodes)
    parser.add_argument("--azimuthal-nodes", type=int, default=CONFIG.azimuthal_nodes)
    parser.add_argument("--record-every", type=int, default=CONFIG.record_every_steps)
    return parser.parse_args()


def run_case(config: RealMeshEvolutionConfig = CONFIG, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "mesh_states"
    if mesh_dir.is_dir():
        shutil.rmtree(mesh_dir)
    operators = load_operators()
    result = run_real_mesh_evolution(config, out_dir, operators)
    overlay = render_fig1_overlay(result, out_dir, config)
    panel = render_panel(result, overlay, out_dir)
    validation_panel, validation_overlay = render_validation_comparison(config, result, out_dir)
    history_csv = save_history_csv(result["history"], out_dir)
    summary = {
        "case": CASE_STEM,
        "truth_status": (
            "real_ddgclib_pr_attached_bridge_film_mesh_evolution"
            if config.evolve_attached_bridge_mesh
            else "real_ddgclib_pr_mesh_evolution_not_reduced_trajectory"
        ),
        "honest_limitations": [
            "overdamped surface-mesh evolution, not a full Navier-Stokes solver",
            "axisymmetry projection is applied after each step for stability",
            "bridge suction closure is driven by mesh missing volume, not by Fig. 1(c) curves",
            "attached Case 11 is a constrained axisymmetric 3D free-surface mesh, not a full Navier-Stokes/VOF solver",
        ],
        "ddgclib_pr_used": {
            "ddgclib_surface_tension_force_each_step": True,
            "ddgclib_cox_contact_line_force_each_step": bool(config.use_cox_contact_line_force),
            "ddgclib_cox_contact_line_kinematic_boundary_each_step": bool(config.evolve_attached_bridge_mesh),
            "primary_state_is_attached_bridge_plus_film_mesh": bool(config.evolve_attached_bridge_mesh),
            "coupled_neck_update_mode": str(config.attached_neck_boundary_layer_update_mode),
            "coupled_neck_residual_mobility_enabled": str(config.attached_neck_boundary_layer_update_mode).strip().lower()
            in {"residual", "residual-mobility", "residual_mobility", "mobility"},
            "ddgclib_free_surface_mesh_writer": True,
            "pr33_heron_diagnostics_available": True,
            "pr37_cox_voinov_contact_line_promoted_to_ddgclib_operator": bool(config.use_cox_contact_line_force),
        },
        "config": asdict(config),
        "final_step": result["final_step"],
        "final_time_s": result["final_time_s"],
        "wall_elapsed_s": result["wall_elapsed_s"],
        "outputs": {
            "panel_png": str(panel.resolve()),
            "fig1_overlay_png": str(overlay.resolve()),
            "validation_set_png": str(validation_panel.resolve()),
            "validation_fig1_overlay_png": str(validation_overlay.resolve()),
            "history_csv": str(history_csv.resolve()),
            "mesh_states": str((out_dir / "mesh_states").resolve()),
            "summary_json": str((out_dir / "summary.json").resolve()),
        },
        "validation": validation_metrics(config, result, out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_existing_case(config: RealMeshEvolutionConfig, out_dir: Path = OUT_DIR) -> dict:
    result = load_existing_result(out_dir)
    validation_panel, validation_overlay = render_validation_comparison(config, result, out_dir)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    outputs = dict(summary.get("outputs", {}))
    outputs.update(
        {
            "validation_set_png": str(validation_panel.resolve()),
            "validation_fig1_overlay_png": str(validation_overlay.resolve()),
        }
    )
    summary.update(
        {
            "case": CASE_STEM,
            "truth_status": (
                "real_ddgclib_pr_attached_bridge_film_mesh_evolution"
                if bool(config.evolve_attached_bridge_mesh)
                else "real_ddgclib_pr_mesh_evolution_not_reduced_trajectory"
            ),
            "validation_note": f"Rendered from saved {CASE_LABEL} real mesh history CSV and mesh snapshot NPZ files; no simulation rerun.",
            "outputs": outputs,
            "validation": validation_metrics(config, result, out_dir),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.render_existing:
        config = config_from_summary(OUT_DIR)
        summary = render_existing_case(config, OUT_DIR)
        print(f"Rendered existing {CASE_LABEL} validation output in {OUT_DIR.resolve()}")
        print(f"Validation PNG: {summary['outputs']['validation_set_png']}")
        print(f"Truth status: {summary['truth_status']}")
        return

    config = RealMeshEvolutionConfig(
        dt_s=float(args.dt),
        max_steps=int(args.max_steps),
        wall_clock_limit_s=float(args.wall_clock_limit_s),
        profile_nodes=int(args.profile_nodes),
        azimuthal_nodes=int(args.azimuthal_nodes),
        record_every_steps=int(args.record_every),
    )
    summary = run_case(config)
    print(f"Wrote outputs to {OUT_DIR.resolve()}")
    print(f"Panel PNG: {summary['outputs']['panel_png']}")
    print(f"Truth status: {summary['truth_status']}")


if __name__ == "__main__":
    main()
