#!/usr/bin/env python3
"""Case 23: residual-mobility neck solve for Siekman 2025.

This case keeps the Case 22 real ddgclib/PR attached bridge+film mesh path, but
changes the neck update from a direct profile projection to a residual-mobility
step.  The coupled Cox/Young-Laplace/lubrication neck solve is still used, but
its result enters as a capillary-viscous residual that updates the ring
velocities before the mesh positions are advanced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import soft_neck_core as case22


base = case22.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 23"
OUTPUT_PREFIX = "case23"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case22.CONFIG,
    max_steps=1000,
    wall_clock_limit_s=10800.0,
    record_every_steps=100,
    snapshot_times_s=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
    attached_neck_boundary_layer_update_mode="residual_mobility",
    attached_neck_boundary_layer_relaxation=0.78,
    attached_neck_residual_time_s=1.6,
    attached_neck_residual_pressure_weight=0.020,
    attached_neck_residual_flux_weight=0.005,
    attached_neck_residual_volume_lagrange_enabled=False,
    attached_neck_residual_volume_lagrange_weight=0.0,
    attached_neck_residual_max_vertical_speed_um_s=24.0,
    attached_neck_residual_max_radial_speed_um_s=38.0,
    attached_neck_boundary_layer_bvp_nodes=220,
    attached_neck_adaptive_rings_fraction=0.66,
    attached_neck_adaptive_width_multiplier=7.0,
    attached_neck_adaptive_min_window_um=45.0,
    attached_neck_adaptive_max_window_mm=2.20,
    attached_neck_adaptive_spacing_exponent=1.95,
    attached_visible_feed_partition_enabled=False,
    attached_volume_projection_min_width_mm=1.00,
    attached_volume_projection_width_mm=1.00,
    attached_volume_projection_width_capillary_lengths=0.0,
    attached_volume_projection_weight_exponent=1.50,
    attached_volume_projection_recovery_front_enabled=True,
    attached_volume_projection_recovery_front_start_s=0.0,
    attached_volume_projection_recovery_front_exponent=0.90,
    attached_volume_projection_recovery_front_min_width_um=25.0,
    attached_volume_projection_recovery_front_max_width_mm=1.80,
    attached_volume_projection_recovery_front_relaxation=1.0,
    attached_volume_projection_time_release_enabled=True,
    attached_volume_projection_time_release_start_s=0.0,
    attached_volume_projection_time_release_tau_s=55.0,
    attached_volume_projection_time_release_exponent=2.0,
    attached_volume_projection_time_release_floor_fraction=0.03,
    attached_volume_projection_time_release_ceiling_fraction=0.56,
    attached_neck_visible_feed_late_multiplier=1.0,
    attached_neck_visible_feed_late_start_s=0.0,
    attached_outer_deficit_spreading_enabled=False,
    attached_outer_deficit_spreading_start_s=1.0e9,
    attached_outer_deficit_spreading_volume_multiplier=1.0,
    attached_outer_film_anti_sawtooth_enabled=True,
    attached_outer_film_anti_sawtooth_start_s=0.0,
    attached_outer_film_anti_sawtooth_inner_margin_mm=0.25,
    attached_outer_film_anti_sawtooth_outer_margin_mm=3.00,
    attached_outer_film_anti_sawtooth_passes=6,
    attached_outer_film_anti_sawtooth_alpha=0.32,
    attached_outer_film_anti_sawtooth_preserve_volume=True,
    min_height_um=28.0,
    dynamic_min_height_enabled=True,
    dynamic_min_height_late_um=14.0,
    dynamic_min_height_start_s=100.0,
    dynamic_min_height_tau_s=1000.0,
    dynamic_min_height_exponent=1.0,
    attached_neck_floor_enabled=True,
    attached_neck_floor_initial_um=63.0,
    attached_neck_floor_late_um=14.0,
    attached_neck_floor_start_s=0.0,
    attached_neck_floor_tau_s=75.0,
    attached_neck_floor_exponent=2.0,
    attached_capillary_bridge_use_dynamic_min_height=True,
    attached_bridge_capillary_shoulder_enabled=True,
    attached_bridge_capillary_shoulder_profile="cox_arc_meniscus",
    attached_bridge_capillary_shoulder_min_width_mm=0.13,
    attached_bridge_capillary_shoulder_max_width_mm=0.35,
    attached_bridge_capillary_shoulder_width_depth_exponent=1.0,
    attached_bridge_capillary_shoulder_drop_exponent=4.0,
    attached_bridge_capillary_shoulder_contact_width_fraction=0.18,
    attached_bridge_capillary_shoulder_relaxation=1.0,
    attached_contact_line_bridge_radius_coupling_enabled=False,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_22_siekman2025_soft_neck_ddgclib_mesh_evolution",
        ROOT / "Case_21_siekman2025_dynamic_neck_floor_ddgclib_mesh_evolution",
        ROOT / "Case_20_siekman2025_longtime_release_ddgclib_mesh_evolution",
        ROOT / "Case_11_siekman2025_real_ddgclib_mesh_evolution",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
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
    seed_validation_assets()
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
