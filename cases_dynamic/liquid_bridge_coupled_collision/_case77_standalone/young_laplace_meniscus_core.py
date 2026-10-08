#!/usr/bin/env python3
"""Case 25: Young-Laplace attached-meniscus bridge for Siekman 2025.

Case 24 removed the vertical-wall/shelf artifact with a smooth connector, but
its late-time outer-film recovery reached the undisturbed 100 um film too
quickly, creating the shoulder visible around r = 5-6 mm in Fig. 1(c).  This
case keeps the same ddgclib/PR attached bridge mesh and Young-Laplace neck, but
lets the outer-film deficit spread over a capillary-length recovery branch
instead of forcing it into the short Case 23/24 recovery front.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import smooth_meniscus_core as case24


base = case24.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 25"
OUTPUT_PREFIX = "case25"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case24.CONFIG,
    max_steps=35000,
    wall_clock_limit_s=30000.0,
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
    attached_volume_projection_recovery_front_exponent=0.90,
    attached_volume_projection_recovery_front_max_width_mm=1.80,
    attached_volume_projection_recovery_front_late_start_s=120.0,
    attached_volume_projection_recovery_front_late_tau_s=650.0,
    attached_volume_projection_recovery_front_late_ramp_exponent=0.90,
    attached_volume_projection_recovery_front_late_max_width_mm=5.50,
    attached_volume_projection_recovery_front_late_shape_exponent=0.70,
    attached_bridge_capillary_shoulder_enabled=True,
    attached_bridge_capillary_shoulder_profile="contact_angle_meniscus",
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_24_siekman2025_smooth_meniscus_ddgclib_mesh_evolution",
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
