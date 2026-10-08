#!/usr/bin/env python3
"""Case 24: Cox/contact-angle attached-meniscus bridge for Siekman 2025.

Case 23 matched the first-100-s Fig. 1(c) observable reasonably, but its saved
meridian still showed an unphysical shelf-like attached bridge.  This case
keeps the same ddgclib/PR residual-neck evolution and uses the shared
capillary-shoulder operator in its Cox/contact-angle meniscus mode, so the
saved 3D mesh itself must show a continuous bridge neck.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import residual_neck_core as case23


base = case23.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 24"
OUTPUT_PREFIX = "case24"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case23.CONFIG,
    max_steps=35000,
    wall_clock_limit_s=20000.0,
    record_every_steps=100,
    snapshot_times_s=(
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        500.0,
        1000.0,
        2000.0,
        3500.0,
    ),
    attached_bridge_capillary_shoulder_enabled=True,
    attached_bridge_capillary_shoulder_profile="contact_angle_meniscus",
    attached_bridge_capillary_shoulder_min_width_mm=0.16,
    attached_bridge_capillary_shoulder_max_width_mm=0.55,
    attached_bridge_capillary_shoulder_width_depth_exponent=1.0,
    attached_bridge_capillary_shoulder_drop_exponent=3.0,
    attached_bridge_capillary_shoulder_contact_width_fraction=0.22,
    attached_bridge_capillary_shoulder_relaxation=1.0,
    attached_contact_line_bridge_radius_coupling_enabled=True,
    attached_contact_line_bridge_radius_growth_multiplier=1.30,
    attached_contact_line_bridge_radius_pull_only_below_target=True,
    attached_contact_line_bridge_radius_cap_gap_fraction=0.30,
    attached_contact_line_bridge_radius_beyond_target_max_speed_um_s=4.0,
    attached_contact_line_bridge_radius_relaxation=1.0,
    attached_contact_line_bridge_radius_max_speed_um_s=500.0,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_23_siekman2025_residual_neck_ddgclib_mesh_evolution",
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
