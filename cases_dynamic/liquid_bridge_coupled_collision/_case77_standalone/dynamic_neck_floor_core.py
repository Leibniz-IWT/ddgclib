#!/usr/bin/env python3
"""Case 21: dynamic neck floor for the Siekman 2025 long-time run.

Case 20 reached the correct bridge-volume scale and conserved volume, but the
Fig. 1(c) profile retained a too-low early neck and a flat artificial floor at
long time.  This case keeps the same real ddgclib/PR attached bridge+film mesh
and activates the dynamic minimum-height model inside the attached capillary
bridge solve, so the neck floor relaxes during the simulation instead of being
low from the first timestep.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import longtime_release_core as case20


base = case20.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 21"
OUTPUT_PREFIX = "case21"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case20.CONFIG,
    min_height_um=28.0,
    dynamic_min_height_enabled=True,
    dynamic_min_height_late_um=20.0,
    dynamic_min_height_start_s=100.0,
    dynamic_min_height_tau_s=1000.0,
    dynamic_min_height_exponent=1.0,
    attached_neck_floor_enabled=True,
    attached_neck_floor_initial_um=63.0,
    attached_neck_floor_late_um=20.0,
    attached_neck_floor_start_s=0.0,
    attached_neck_floor_tau_s=75.0,
    attached_neck_floor_exponent=2.0,
    attached_capillary_bridge_use_dynamic_min_height=True,
    attached_capillary_bridge_lower_bound_repulsion=0.10,
    attached_capillary_bridge_lower_bound_repulsion_length_um=10.0,
    attached_capillary_bridge_profile_smoothing_passes=0,
    attached_capillary_bridge_profile_smoothing_alpha=0.0,
    attached_capillary_bridge_profile_max_cell_dz_um=0.0,
    attached_capillary_bridge_profile_remove_local_maxima=False,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_20_siekman2025_longtime_release_ddgclib_mesh_evolution",
        ROOT / "Case_19_siekman2025_capillary_spread_ddgclib_mesh_evolution_backup_100s_20260704_174216",
        ROOT / "Case_19_siekman2025_capillary_spread_ddgclib_mesh_evolution",
        ROOT / "Case_18_siekman2025_wide_neck_overlap_ddgclib_mesh_evolution",
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
