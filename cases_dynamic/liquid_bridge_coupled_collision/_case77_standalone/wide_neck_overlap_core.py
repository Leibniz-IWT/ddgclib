#!/usr/bin/env python3
"""Case 18: wider capillary neck evolution for Siekman 2025.

Case 17 fixed the mesh jump but its Fig. 1(c) 100 s recovery branch was still
too narrow.  This case keeps the same real ddgclib/PR attached bridge+film mesh,
Cox contact-line condition, and Young-Laplace/lubrication neck solve, but lets
the reusable neck operator resolve a wider capillary-length drainage region.

No experimental curve coordinates are used to prescribe the bridge shape.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import resolved_neck_core as case16


base = case16.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 18"
OUTPUT_PREFIX = "case18"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case16.CONFIG,
    max_steps=1000,
    wall_clock_limit_s=7200.0,
    record_every_steps=100,
    snapshot_times_s=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
    attached_contact_angle_probe_window_um=140.0,
    attached_contact_angle_probe_min_fit_rings=4,
    attached_contact_angle_probe_max_fit_rings=10,
    attached_contact_line_max_speed_um_s=150.0,
    attached_contact_line_bridge_radius_coupling_enabled=True,
    attached_contact_line_target_uses_visible_feed_partition=False,
    attached_contact_line_bridge_radius_relaxation=0.34,
    attached_contact_line_bridge_radius_max_speed_um_s=120.0,
    attached_contact_line_allow_recede=False,
    attached_visible_feed_subtract_wetted_sphere_cap=True,
    attached_visible_feed_cap_fraction_uses_corrected_volume=True,
    attached_visible_feed_max_fraction=0.12,
    attached_bridge_rim_capillary_length_multiplier=0.82,
    attached_visible_feed_profile_constraint_enabled=False,
    attached_capillary_bridge_solver_enabled=True,
    attached_capillary_bridge_nodes=32,
    attached_capillary_bridge_enforce_contact_angle=True,
    attached_capillary_bridge_match_film_slope=True,
    attached_capillary_bridge_profile_smoothing_passes=4,
    attached_capillary_bridge_profile_smoothing_alpha=0.35,
    attached_capillary_bridge_profile_max_cell_dz_um=35.0,
    attached_capillary_bridge_profile_volume_correction=True,
    attached_capillary_bridge_profile_remove_local_maxima=True,
    attached_neck_boundary_layer_max_width_capillary_lengths=2.20,
    attached_neck_boundary_layer_bvp_nodes=180,
    attached_neck_adaptive_width_multiplier=7.0,
    attached_neck_adaptive_max_window_mm=3.00,
    attached_neck_adaptive_spacing_exponent=1.70,
    attached_neck_visible_feed_late_multiplier=240.0,
    attached_neck_visible_feed_late_start_s=85.0,
    attached_neck_visible_feed_late_tau_s=22.0,
    attached_neck_visible_feed_late_exponent=1.0,
    min_height_um=20.0,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_17_siekman2025_cox_window_neck_ddgclib_mesh_evolution",
        ROOT / "Case_16_siekman2025_resolved_neck_ddgclib_mesh_evolution",
        ROOT / "Case_15_siekman2025_predictive_young_laplace_ddgclib_mesh_evolution",
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
    if args.render_existing:
        seed_validation_assets()
        config = base.config_from_summary(OUT_DIR, CONFIG)
        summary = base.render_existing_case(config, OUT_DIR)
        print(f"Rendered existing {CASE_LABEL} validation output in {OUT_DIR.resolve()}")
        print(f"Validation PNG: {summary['outputs']['validation_set_png']}")
        print(f"Truth status: {summary['truth_status']}")
        return

    config = build_config(args)
    seed_validation_assets()
    summary = base.run_case(config, OUT_DIR)
    print(f"Wrote outputs to {OUT_DIR.resolve()}")
    print(f"Validation PNG: {summary['outputs']['validation_set_png']}")
    print(f"Truth status: {summary['truth_status']}")


if __name__ == "__main__":
    main()
