#!/usr/bin/env python3
"""Case 22: soft lower-bound neck evolution for Siekman 2025.

Case 21 exposed a real model artifact: the saved bridge/film mesh hit a hard
20 um lower-height floor, which produced a flat Fig. 1(c) minimum.  This case
keeps the same ddgclib/PR attached bridge+film mesh evolution, Cox contact-line
operator, and coupled Young-Laplace/lubrication neck solve, but removes the
large hard floor.  A small numerical bound plus lower-bound repulsion remains
only to keep the mesh positive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import dynamic_neck_floor_core as case21


base = case21.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 22"
OUTPUT_PREFIX = "case22"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case21.CONFIG,
    # Remove the physically-visible hard floor that caused the flat-bottom
    # Fig. 1(c) artifact.  The 0.25 um value is a numerical positivity guard,
    # not a prescribed experimental neck depth.
    min_height_um=0.25,
    dynamic_min_height_enabled=False,
    dynamic_min_height_late_um=0.25,
    attached_neck_floor_enabled=False,
    attached_neck_floor_initial_um=0.25,
    attached_neck_floor_late_um=0.25,
    attached_capillary_bridge_use_dynamic_min_height=False,
    attached_capillary_bridge_precursor_fraction=0.0,
    attached_capillary_bridge_min_height_um=0.25,
    # Use soft near-wall repulsion rather than clipping a wide radial interval
    # to one prescribed height.
    attached_capillary_bridge_lower_bound_repulsion=0.45,
    attached_capillary_bridge_lower_bound_repulsion_length_um=8.0,
    attached_capillary_bridge_profile_smoothing_passes=0,
    attached_capillary_bridge_profile_smoothing_alpha=0.0,
    attached_capillary_bridge_profile_max_cell_dz_um=0.0,
    attached_capillary_bridge_profile_volume_correction=False,
    attached_capillary_bridge_profile_remove_local_maxima=False,
    attached_capillary_bridge_enforce_contact_angle=True,
    attached_capillary_bridge_match_film_slope=True,
    attached_capillary_bridge_profile_model="soft_repulsive_linearized_young_laplace",
    attached_capillary_bridge_positive_volume_floor=False,
    attached_neck_boundary_layer_max_width_capillary_lengths=0.75,
    attached_neck_adaptive_width_multiplier=4.0,
    attached_neck_adaptive_max_window_mm=1.45,
    # The visible outer-film drain should start shallow and then grow on a
    # capillary drainage time scale; otherwise the 10 s branch collapses into an
    # artificially deep compact rim trough.
    attached_visible_feed_min_fraction=0.01,
    attached_visible_feed_max_fraction=0.04,
    attached_neck_visible_feed_late_multiplier=6.0,
    attached_neck_visible_feed_late_start_s=15.0,
    attached_neck_visible_feed_late_tau_s=25.0,
    attached_neck_visible_feed_late_exponent=1.0,
    attached_outer_deficit_spreading_start_s=0.0,
    attached_outer_deficit_spreading_ramp_s=8.0,
    attached_outer_deficit_spreading_inner_guard_mm=0.0,
    attached_outer_deficit_soft_lower_enabled=True,
    attached_outer_deficit_soft_lower_um=20.0,
    attached_outer_deficit_soft_lower_slope_um_per_mm=0.0,
    attached_outer_deficit_soft_lower_width_mm=1.25,
    attached_outer_deficit_soft_lower_repulsion=0.98,
    attached_outer_film_recovery_max_cell_dz_um=18.0,
    # Keep Case 21's validated time/volume driver and comparison checkpoints.
    record_every_steps=100,
    snapshot_times_s=(0.0, 10.0, 100.0, 500.0, 1000.0, 2000.0, 3500.0),
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
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
