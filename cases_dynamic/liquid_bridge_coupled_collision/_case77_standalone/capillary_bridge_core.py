#!/usr/bin/env python3
"""Case 13: ddgclib capillary-bridge meniscus closure for Siekman 2025.

This case keeps the Case 11 h0=100 um geometry/material setup and real
ddgclib/PR mesh evolution, but replaces the post-step geometric bridge
connector with a reusable ddgclib axisymmetric capillary-bridge operator.  The
operator minimizes the bridge free-surface area at fixed bridge volume with the
sphere contact and bridge/film rim pinned.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil

from . import real_mesh_evolution_core as case11


base = case11.base
ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 13"
OUTPUT_PREFIX = "case13"
OUT_DIR = ROOT / CASE_STEM

base.CASE_STEM = CASE_STEM
base.CASE_LABEL = CASE_LABEL
base.OUTPUT_PREFIX = OUTPUT_PREFIX
base.OUT_DIR = OUT_DIR

CONFIG = replace(
    case11.CONFIG,
    attached_capillary_bridge_solver_enabled=True,
    attached_capillary_bridge_precursor_fraction=0.18,
    attached_capillary_bridge_min_height_um=0.0,
    attached_capillary_bridge_nodes=56,
    full_bridge_mesh_bridge_nodes=56,
    attached_bridge_shape_constraint_enabled=True,
    attached_volume_projection_bridge_reservoir_enabled=False,
    attached_visible_feed_partition_enabled=True,
    attached_visible_feed_min_fraction=0.12,
    attached_visible_feed_max_fraction=0.72,
    attached_visible_feed_transition_progress=0.50,
    attached_visible_feed_transition_width=0.04,
    attached_volume_projection_width_capillary_lengths=1.0,
    attached_volume_projection_min_width_mm=0.25,
    attached_volume_projection_scale_width_with_visible_fraction=True,
    attached_contact_line_target_uses_visible_feed_partition=True,
    attached_visible_feed_profile_constraint_enabled=True,
    attached_visible_feed_profile_base_width_capillary_lengths=0.12,
    attached_visible_feed_profile_fraction_width_capillary_lengths=0.58,
    attached_visible_feed_profile_decay_exponent=1.0,
    attached_visible_feed_profile_late_volume_boost_factor=0.18,
    attached_visible_feed_profile_late_boost_progress=0.50,
    attached_visible_feed_profile_late_boost_width=0.05,
)


def seed_validation_assets() -> None:
    """Reuse already-audited local Siekman digitization assets if available."""

    source_dir = ROOT / "Case_11_siekman2025_real_ddgclib_mesh_evolution"
    if not source_dir.is_dir():
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
