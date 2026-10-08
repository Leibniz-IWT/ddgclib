#!/usr/bin/env python3
"""Case 11: real ddgclib/PR mesh evolution with PR37 Cox contact line.

Case 11 is the Siekman validation case for developing ddgclib/PR33/PR35/PR37
further.  It uses the real triangular ddgclib free-surface mesh and advances
the force as Heron surface tension plus a reusable ddgclib Cox-Voinov
contact-line force promoted from the PR37 axisymmetric solver.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from . import base_mesh_evolution_core as base


ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 11"
OUTPUT_PREFIX = "case11"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

R_MM = 5.0
H0_MM = 0.100
PHYSICAL_FIRST_CONTACT_RADIUS_MM = 0.0
CONTACT_RING_MESH_SEED_MM = 0.001
FIRST_CONTACT_RIM_SEED_MM = 0.050
INITIAL_BRIDGE_VOLUME_UL = 0.0

CONFIG = replace(
    base.CONFIG,
    # First-contact state: the sphere tip just touches the initially flat
    # h0=100 um film.  The 50 um rim is only the smallest connected mesh seed,
    # not a measured experiment radius.
    sphere_bottom_z_mm=H0_MM,
    inner_bridge_volume_ul=INITIAL_BRIDGE_VOLUME_UL,
    initial_bridge_radius_mm=CONTACT_RING_MESH_SEED_MM,
    dt_s=0.1,
    max_steps=35000,
    wall_clock_limit_s=7200.0,
    record_every_steps=1000,
    profile_nodes=84,
    azimuthal_nodes=24,
    bridge_pressure_width_mm=0.042,
    bridge_pressure_multiplier=2.0,
    bridge_pressure_startup_time_s=20.0,
    bridge_pressure_startup_exponent=0.75,
    bridge_pressure_relax_time_s=4000.0,
    bridge_pressure_relax_exponent=1.0,
    late_pressure_boost_enabled=True,
    late_pressure_boost_start_s=20.0,
    late_pressure_boost_tau_s=40.0,
    late_pressure_boost_exponent=1.0,
    late_pressure_boost_factor=1.35,
    terminal_pressure_boost_enabled=False,
    terminal_pressure_boost_start_s=2400.0,
    terminal_pressure_boost_tau_s=450.0,
    terminal_pressure_boost_exponent=1.0,
    terminal_pressure_boost_factor=0.0,
    bridge_pressure_center_offset_mm=1.03,
    bridge_pressure_center_transient_inset_mm=0.17,
    bridge_pressure_center_transient_growth_time_s=25.0,
    bridge_pressure_center_transient_relax_time_s=1100.0,
    bridge_pressure_center_transient_exponent=1.0,
    pinned_bridge_radius_mm=PHYSICAL_FIRST_CONTACT_RADIUS_MM,
    dynamic_inner_plateau_enabled=False,
    dynamic_inner_plateau_start_s=140.0,
    dynamic_inner_plateau_tau_s=850.0,
    dynamic_inner_plateau_exponent=1.0,
    dynamic_inner_plateau_gap_to_pressure_center_mm=0.42,
    dynamic_inner_plateau_max_radius_mm=3.62,
    bridge_feed_capacity_multiplier=1.36,
    bridge_feed_time_scale_s=460.0,
    bridge_feed_time_exponent=0.45,
    volume_projection_start_widths=1.05,
    refined_bridge_mesh_fraction=0.76,
    velocity_smoothing=0.68,
    max_vertical_speed_um_s=20.0,
    max_radial_speed_um_s=12.0,
    use_cox_contact_line_force=True,
    enable_dynamic_contact_angle=True,
    contact_angle_deg=0.0,
    dynamic_contact_angle_min_deg=0.0,
    dynamic_contact_angle_max_deg=25.0,
    contact_line_cox_macro_length_m=1.0e-3,
    contact_line_cox_slip_length_m=2.0e-9,
    cox_contact_line_force_scale=1.0,
    cox_contact_line_activation_time_s=1.0,
    cox_contact_line_activation_exponent=1.0,
    cox_contact_line_direction_z=-1.0,
    cox_contact_ring_offset_mm=0.0,
    full_bridge_mesh_enabled=True,
    full_bridge_mesh_bridge_nodes=36,
    evolve_attached_bridge_mesh=True,
    attached_reduced_bridge_film_driver_enabled=True,
    attached_reduced_driver_bottleneck_volume_ul=0.014,
    attached_reduced_driver_bottleneck_exponent=2.0,
    attached_reduced_driver_pressure_activation_time_s=1.0,
    attached_reduced_driver_pressure_relax_time_s=4000.0,
    attached_reduced_driver_pressure_relax_exponent=1.0,
    attached_reduced_driver_grid_nodes=140,
    attached_local_capture_enabled=True,
    attached_local_capture_width_capillary_lengths=1.84,
    attached_local_capture_time_s=8.0,
    attached_local_capture_exponent=1.0,
    attached_local_capture_release_time_s=10000.0,
    attached_local_capture_release_exponent=1.0,
    attached_volume_projection_width_mm=3.30,
    attached_volume_projection_weight_exponent=1.35,
    attached_volume_projection_bridge_reservoir_enabled=True,
    attached_volume_projection_bridge_reservoir_weight=4.0,
    attached_volume_projection_bridge_reservoir_inner_margin_mm=0.10,
    attached_volume_projection_bridge_reservoir_outer_gap_mm=0.55,
    attached_initial_contact_radius_mm=CONTACT_RING_MESH_SEED_MM,
    attached_initial_rim_radius_mm=FIRST_CONTACT_RIM_SEED_MM,
    attached_contact_line_max_speed_um_s=150.0,
    attached_contact_line_bridge_radius_coupling_enabled=True,
    attached_contact_line_bridge_radius_relaxation=0.22,
    attached_contact_line_bridge_radius_max_speed_um_s=35.0,
    attached_bridge_relaxation=0.12,
    attached_bridge_shape_constraint_enabled=True,
    attached_bridge_shape_neck_fraction=0.12,
    attached_bridge_shape_height_exponent=1.65,
    attached_bridge_rim_offset_mm=1.05,
    attached_bridge_rim_transient_offset_mm=0.20,
    attached_bridge_rim_transient_decay_time_s=30.0,
    attached_bridge_rim_transient_decay_exponent=1.0,
    attached_bridge_rim_relaxation_per_step=0.055,
    attached_bridge_rim_shift_decay_mm=1.4,
    attached_mesh_mobility_startup_time_s=4.0,
    attached_mesh_mobility_startup_exponent=0.75,
    attached_mesh_mobility_floor=0.55,
    attached_feed_limiter_start_time_s=20.0,
    attached_feed_limiter_start_missing_ul=0.50,
    attached_feed_limiter_capacity_multiplier=1.25,
    attached_feed_limiter_growth_time_s=80.0,
    attached_feed_limiter_growth_exponent=0.8,
    attached_feed_limiter_max_missing_ul=1.80,
    max_height_um=H0_MM * 1000.0,
    min_height_um=25.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-existing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=CONFIG.max_steps)
    parser.add_argument("--wall-clock-limit-s", type=float, default=CONFIG.wall_clock_limit_s)
    parser.add_argument("--dt", type=float, default=CONFIG.dt_s)
    parser.add_argument("--profile-nodes", type=int, default=CONFIG.profile_nodes)
    parser.add_argument("--azimuthal-nodes", type=int, default=CONFIG.azimuthal_nodes)
    parser.add_argument("--record-every", type=int, default=CONFIG.record_every_steps)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Save mesh snapshots every N solver steps; 0 uses CONFIG.snapshot_times_s.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> base.RealMeshEvolutionConfig:
    snapshot_times = CONFIG.snapshot_times_s
    if int(args.snapshot_every) > 0:
        snapshot_steps = range(0, int(args.max_steps) + 1, int(args.snapshot_every))
        snapshot_times = tuple(float(step) * float(args.dt) for step in snapshot_steps)
        final_time = float(args.max_steps) * float(args.dt)
        if not snapshot_times or abs(snapshot_times[-1] - final_time) > 1.0e-12:
            snapshot_times = (*snapshot_times, final_time)
    return replace(
        CONFIG,
        dt_s=float(args.dt),
        max_steps=int(args.max_steps),
        wall_clock_limit_s=float(args.wall_clock_limit_s),
        profile_nodes=int(args.profile_nodes),
        azimuthal_nodes=int(args.azimuthal_nodes),
        record_every_steps=int(args.record_every),
        snapshot_times_s=snapshot_times,
    )


def main() -> None:
    args = parse_args()
    if args.render_existing:
        config = base.config_from_summary(OUT_DIR, CONFIG)
        summary = base.render_existing_case(config, OUT_DIR)
        print(f"Rendered existing {CASE_LABEL} validation output in {OUT_DIR.resolve()}")
        print(f"Validation PNG: {summary['outputs']['validation_set_png']}")
        print(f"Truth status: {summary['truth_status']}")
        return

    config = build_config(args)
    summary = base.run_case(config, OUT_DIR)
    print(f"Wrote outputs to {OUT_DIR.resolve()}")
    print(f"Validation PNG: {summary['outputs']['validation_set_png']}")
    print(f"Truth status: {summary['truth_status']}")


if __name__ == "__main__":
    main()
