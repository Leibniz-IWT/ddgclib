#!/usr/bin/env python3
"""Case 19: capillary spreading of the visible film deficit.

Case 18 improved the 100 s minimum but the Fig. 1(c) recovery branch around
``r = 3.5-4.0 mm`` remained too high.  This case keeps the same real
ddgclib/PR attached bridge+film mesh and adds a conservative outer-film
deficit spreading step.  The step diffuses the saved film-height deficit over a
capillary-length window and restores volume on the mesh; it does not move the
plotted curve in the renderer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import wide_neck_overlap_core as case18


base = case18.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 19"
OUTPUT_PREFIX = "case19"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case18.CONFIG,
    attached_outer_deficit_spreading_enabled=True,
    attached_outer_deficit_spreading_start_s=85.0,
    attached_outer_deficit_spreading_ramp_s=22.0,
    attached_outer_deficit_spreading_width_capillary_lengths=1.15,
    attached_outer_deficit_spreading_inner_guard_mm=0.18,
    attached_outer_deficit_spreading_passes=18,
    attached_outer_deficit_spreading_blend=0.42,
    attached_outer_deficit_spreading_volume_multiplier=2.20,
    attached_outer_deficit_spreading_monotone_recovery=True,
    snapshot_times_s=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 3500.0),
    min_height_um=28.0,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dirs = (
        ROOT / "Case_18_siekman2025_wide_neck_overlap_ddgclib_mesh_evolution",
        ROOT / "Case_17_siekman2025_cox_window_neck_ddgclib_mesh_evolution",
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
