#!/usr/bin/env python3
"""Build a Pitois Fig. 5 t=0 mesh calibrated to the first force point.

The contact radius is not taken from Pitois Eq. 6 here.  It is fitted by
running the same ddgclib/FHeron gorge-force reporting path used by the raw
simulation at step 0 and matching the first digitized Fig. 5 force point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


SOLVER_NAME = "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib.py"
PITOIS_FIG5_FIRST_D_OVER_R = 0.014403481854782
PITOIS_FIG5_FIRST_FORCE_MN = 1.515311539463195
DEFAULT_SCAN_MIN_MM = 1.0
DEFAULT_SCAN_MAX_MM = 1.7
DEFAULT_SCAN_STEP_MM = 0.005


def _load_solver():
    path = Path(__file__).resolve().with_name(SOLVER_NAME)
    spec = importlib.util.spec_from_file_location("case2b_ddgclib_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_for_radius(module, radius_mm: float):
    return module.VolumetricPitoisConfig(
        name="Case_2b_initialmesh_t0_force_calibrated",
        title="Case 2b: t=0 force-calibrated initial mesh",
        use_pitois_eq6_contact_radius=False,
        target_cap_radius=float(radius_mm) * 1.0e-3,
        initial_contact_radius_mode="pitois_eq6",
        pressure_closure="compressible",
        enable_incompressible_projection=False,
        n_steps=0,
        record_every=1,
        mesh_snapshot_every=0,
    )


def _eval_radius(module, radius_mm: float) -> tuple[float, dict, object]:
    config = _config_for_radius(module, radius_mm)
    state = module._prepare_gmsh_state(config)
    row = module._step_record(state, step=0, t=0.0)
    force_mn = abs(float(row["top_gorge_pressure_total_axial"])) * 1.0e3
    return force_mn, row, state


def _find_force_matching_branches(
    module,
    *,
    r_min_mm: float,
    r_max_mm: float,
    scan_step_mm: float,
    target_force_mn: float,
) -> list[dict[str, float]]:
    if r_min_mm <= 0.0 or r_max_mm <= r_min_mm or scan_step_mm <= 0.0:
        raise ValueError("Invalid radius scan range.")

    samples: list[tuple[float, float]] = []
    n_steps = int(math.ceil((r_max_mm - r_min_mm) / scan_step_mm))
    for k in range(n_steps + 1):
        radius = min(r_max_mm, r_min_mm + k * scan_step_mm)
        force_mn, _row, _state = _eval_radius(module, radius)
        samples.append((radius, force_mn))
        if radius >= r_max_mm:
            break

    branches: list[dict[str, float]] = []
    for (r0, f0), (r1, f1) in zip(samples, samples[1:]):
        y0 = f0 - target_force_mn
        y1 = f1 - target_force_mn
        if y0 == 0.0:
            branches.append({"rcl_mm": r0, "force_mn": f0, "error_mn": 0.0})
            continue
        if y0 * y1 > 0.0:
            continue

        lo, hi = r0, r1
        flo = f0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            fmid, _row, _state = _eval_radius(module, mid)
            if (flo - target_force_mn) * (fmid - target_force_mn) <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid

        root = 0.5 * (lo + hi)
        force_mn, _row, _state = _eval_radius(module, root)
        branches.append(
            {
                "rcl_mm": float(root),
                "force_mn": float(force_mn),
                "error_mn": float(force_mn - target_force_mn),
            }
        )

    return branches


def _write_outputs(
    module,
    state,
    row: dict,
    branches: list[dict[str, float]],
    out_dir: Path,
    *,
    branch_name: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    radius_mm = float(row["initial_geometry_contact_radius"]) * 1.0e3
    title = (
        "Case 2b t=0 force-calibrated initial mesh\n"
        f"r_cl={radius_mm:.6f} mm, |F_gorge|={abs(float(row['top_gorge_pressure_total_axial'])) * 1.0e3:.6f} mN"
    )
    png_path = module._render_raw_volume_mesh_snapshot(
        state,
        title,
        out_dir / "mesh_initial.png",
        elev=0.0,
        azim=90.0,
    )

    mesh_path = png_path.with_suffix(".msh")
    iter0_png = out_dir / "mesh_iter0000.png"
    iter0_msh = out_dir / "mesh_iter0000.msh"
    if png_path != iter0_png:
        iter0_png.write_bytes(png_path.read_bytes())
    if mesh_path != iter0_msh:
        iter0_msh.write_bytes(mesh_path.read_bytes())

    meridian_png = _write_meridian_png(module, state, out_dir)

    force_mn = abs(float(row["top_gorge_pressure_total_axial"])) * 1.0e3
    summary = {
        "method": f"{branch_name}-root t=0 calibration against first Pitois Fig. 5 force point",
        "selected_branch": branch_name,
        "solver_module": SOLVER_NAME,
        "force_model": "ddgclib/FHeron gorge pressure plus neck surface tension",
        "pitois_first_point": {
            "d_over_r": PITOIS_FIG5_FIRST_D_OVER_R,
            "force_mn": PITOIS_FIG5_FIRST_FORCE_MN,
        },
        "chosen": {
            "rcl_mm": radius_mm,
            "d_over_r": float(row["d_over_r"]),
            "force_mn": force_mn,
            "force_error_mn": force_mn - PITOIS_FIG5_FIRST_FORCE_MN,
            "force_error_percent": 100.0 * (force_mn - PITOIS_FIG5_FIRST_FORCE_MN) / PITOIS_FIG5_FIRST_FORCE_MN,
            "neck_radius_mm": float(row["pressure_neck_fit_radius"]) * 1.0e3,
            "target_volume_ul": float(row["initial_geometry_target_volume_m3"]) * 1.0e9,
            "actual_volume_ul": float(row["initial_geometry_actual_volume_m3"]) * 1.0e9,
            "volume_rel_error": float(row["initial_geometry_volume_rel_error"]),
        },
        "branches": branches,
        "config": asdict(state.config),
        "mesh_initial_png": str(png_path.resolve()),
        "mesh_initial_meridian_png": str(meridian_png.resolve()),
        "mesh_initial_msh": str(mesh_path.resolve()),
        "mesh_iter0000_png": str(iter0_png.resolve()),
        "mesh_iter0000_msh": str(iter0_msh.resolve()),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def _write_meridian_png(module, state, out_dir: Path) -> Path:
    rings = list(getattr(state, "outer_rings", []))
    z_mm: list[float] = []
    r_mm: list[float] = []
    for ring in rings:
        if not ring:
            continue
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        points_m = np.asarray(module._length_from_compute(state.config, points), dtype=float)
        z_mm.append(float(np.mean(points_m[:, 2])) * 1.0e3)
        r_mm.append(float(np.max(np.linalg.norm(points_m[:, :2], axis=1))) * 1.0e3)

    z = np.asarray(z_mm, dtype=float)
    r = np.asarray(r_mm, dtype=float)
    order = np.argsort(z)
    z = z[order]
    r = r[order]

    fig, ax = module.plt.subplots(figsize=(8.0, 3.4))
    if z.size:
        ax.fill_betweenx(z, -r, r, color="#9bd3e6", alpha=0.62)
        ax.plot(r, z, color="#08384f", linewidth=2.4)
        ax.plot(-r, z, color="#08384f", linewidth=2.4)
        ax.scatter([r[0], -r[0], r[-1], -r[-1]], [z[0], z[0], z[-1], z[-1]], s=28, color="#111111")
    ax.axvline(0.0, color="#9ca3af", linewidth=0.9, linestyle="--")
    ax.axhline(0.0, color="#c7c7c7", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("r-plane coordinate [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Case 2b t=0 force-calibrated initial mesh: meridian cross-section")
    ax.grid(alpha=0.28)
    path = out_dir / "mesh_initial_meridian.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    module.plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out/Case_2b_initialmesh_t0_force_calibrated")
    parser.add_argument("--r-min-mm", type=float, default=DEFAULT_SCAN_MIN_MM)
    parser.add_argument("--r-max-mm", type=float, default=DEFAULT_SCAN_MAX_MM)
    parser.add_argument("--scan-step-mm", type=float, default=DEFAULT_SCAN_STEP_MM)
    parser.add_argument(
        "--branch",
        choices=("largest", "middle", "smallest"),
        default="largest",
        help="Which exact force-matching root to export when several roots exist.",
    )
    args = parser.parse_args(argv)

    module = _load_solver()
    branches = _find_force_matching_branches(
        module,
        r_min_mm=float(args.r_min_mm),
        r_max_mm=float(args.r_max_mm),
        scan_step_mm=float(args.scan_step_mm),
        target_force_mn=PITOIS_FIG5_FIRST_FORCE_MN,
    )
    if not branches:
        raise RuntimeError("No force-matching contact-radius branch found in the scan range.")

    branches = sorted(branches, key=lambda item: float(item["rcl_mm"]))
    if args.branch == "largest":
        chosen = branches[-1]
    elif args.branch == "middle":
        chosen = branches[len(branches) // 2]
    else:
        chosen = branches[0]
    force_mn, row, state = _eval_radius(module, float(chosen["rcl_mm"]))
    del force_mn
    summary = _write_outputs(
        module,
        state,
        row,
        branches,
        Path(args.out_dir),
        branch_name=str(args.branch),
    )

    chosen_summary = summary["chosen"]
    print("Created force-calibrated t=0 initial mesh")
    print(f"r_cl = {chosen_summary['rcl_mm']:.9f} mm")
    print(f"|F_gorge| = {chosen_summary['force_mn']:.9f} mN")
    print(f"target |F_exp| = {PITOIS_FIG5_FIRST_FORCE_MN:.9f} mN")
    print(f"volume = {chosen_summary['actual_volume_ul']:.9f} uL")
    print(f"mesh = {summary['mesh_initial_msh']}")
    print(f"png = {summary['mesh_initial_png']}")
    print(f"summary = {summary['summary_json']}")


if __name__ == "__main__":
    main()
