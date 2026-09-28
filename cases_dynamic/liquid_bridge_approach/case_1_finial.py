#!/usr/bin/env python3
"""Case #1: Fig. 6 approach with raw gorge force, wall viscous, and Cox CL."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import run_fig6_approach_case12 as base


ROOT = Path(__file__).resolve().parent
CASE_NAME = Path(__file__).stem
DEFAULT_OUT_DIR = ROOT / CASE_NAME


def build_cli():
    parser = base.build_cli()
    parser.set_defaults(
        out_dir=DEFAULT_OUT_DIR,
        steps=7500,
        record_every=1000,
        history_every_step=False,
        mesh_snapshot_every=100,
        cox_contact_line_force=True,
    )
    return parser


def build_config(core, args):
    config = base.build_config(core, args)
    return replace(
        config,
        name=CASE_NAME,
        title=(
            "Pitois 2000 Fig. 6 approach: Case #1 raw gorge force with "
            "wall-viscous Cauchy traction and Cox contact-line force"
        ),
        reported_gorge_add_lubrication_pressure_force=False,
        reported_gorge_lubrication_pressure_wall_traction=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    if int(args.steps) < 0:
        raise ValueError("--steps must be >= 0")
    if float(args.dt) <= 0.0:
        raise ValueError("--dt must be > 0")
    if int(args.record_every) < 1:
        raise ValueError("--record-every must be >= 1")
    if int(args.mesh_snapshot_every) < 0:
        raise ValueError("--mesh-snapshot-every must be >= 0")
    if float(args.gorge_neck_fit_radius_max_raw_ratio) <= 0.0:
        raise ValueError("--gorge-neck-fit-radius-max-raw-ratio must be > 0")
    if not Path(args.initial_mesh).is_file():
        raise FileNotFoundError(args.initial_mesh)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    core = base.load_case12_core()
    base.disable_core_nonmesh_png_refresh(core)
    config = build_config(core, args)
    history = core.run_motion_case(
        config,
        out_dir=out_dir,
        save_fig=int(args.mesh_snapshot_every) > 0,
        save_results=True,
        verbose=bool(args.verbose),
    )
    sparse = base.sparse_history(history, int(args.record_every))
    chart = base.write_force_vs_time_png(
        history=history,
        config=config,
        out_dir=out_dir,
        record_every=int(args.record_every),
    )
    summary = base.write_summary(
        history=history,
        sparse=sparse,
        config=config,
        out_dir=out_dir,
        chart_path=chart,
    )
    print(f"Wrote {chart}")
    print(f"Wrote {summary}")
    print("Sparse steps:", ", ".join(str(int(row["step"])) for row in sparse))
    print(
        "Sparse F[mN]:",
        ", ".join(f"{base.force_for_fig6(row, config):.6g}" for row in sparse),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
