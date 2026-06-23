#!/usr/bin/env python3
"""Build a Pitois Fig. 5 t=0 mesh with r_cl=1.54 mm and calibrated neck.

This keeps the proven r_cl=1.54 mm contact-ring branch but uses the solver's
volume-preserving bulged initial-geometry transform to tune the gorge/neck
radius until the raw ddgclib/FHeron gorge force matches the first Fig. 5 point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


SOLVER_NAME = "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib.py"
TARGET_CONTACT_RADIUS_MM = 1.54
PITOIS_FIG5_FIRST_FORCE_MN = 1.515311539463195
DEFAULT_NECK_MIN_MM = 0.9000
DEFAULT_NECK_MAX_MM = 0.9025
DEFAULT_SLOPE_SCALE = -0.85


def _load_solver():
    path = Path(__file__).resolve().with_name(SOLVER_NAME)
    spec = importlib.util.spec_from_file_location("case2b_ddgclib_solver_r154", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_for_neck(module, neck_mm: float, slope_scale: float):
    return module.VolumetricPitoisConfig(
        name="Case_2b_initialmesh_r154_t0_neck_calibrated",
        title="Case 2b: r_cl=1.54 mm, t=0 neck-calibrated initial mesh",
        use_pitois_eq6_contact_radius=False,
        target_cap_radius=TARGET_CONTACT_RADIUS_MM * 1.0e-3,
        initial_contact_radius_mode="pitois_eq6_bulged",
        initial_bulged_neck_radius=float(neck_mm) * 1.0e-3,
        initial_bulged_contact_slope_scale=float(slope_scale),
        pressure_closure="compressible",
        enable_incompressible_projection=False,
        n_steps=0,
        record_every=1,
        mesh_snapshot_every=0,
    )


def _eval_neck(module, neck_mm: float, slope_scale: float) -> tuple[float, dict, object]:
    config = _config_for_neck(module, neck_mm, slope_scale)
    state = module._prepare_gmsh_state(config)
    row = module._step_record(state, step=0, t=0.0)
    force_mn = abs(float(row["top_gorge_pressure_total_axial"])) * 1.0e3
    return force_mn, row, state


def _find_neck_root(module, *, neck_min_mm: float, neck_max_mm: float, slope_scale: float) -> tuple[float, float]:
    f_lo, _row_lo, _state_lo = _eval_neck(module, neck_min_mm, slope_scale)
    f_hi, _row_hi, _state_hi = _eval_neck(module, neck_max_mm, slope_scale)
    y_lo = f_lo - PITOIS_FIG5_FIRST_FORCE_MN
    y_hi = f_hi - PITOIS_FIG5_FIRST_FORCE_MN
    if y_lo == 0.0:
        return neck_min_mm, f_lo
    if y_hi == 0.0:
        return neck_max_mm, f_hi
    if y_lo * y_hi > 0.0:
        raise RuntimeError(
            "Neck scan does not bracket the target force: "
            f"{neck_min_mm:.6f} mm -> {f_lo:.9f} mN, "
            f"{neck_max_mm:.6f} mm -> {f_hi:.9f} mN"
        )

    lo = float(neck_min_mm)
    hi = float(neck_max_mm)
    flo = float(f_lo)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid, _row_mid, _state_mid = _eval_neck(module, mid, slope_scale)
        if (flo - PITOIS_FIG5_FIRST_FORCE_MN) * (f_mid - PITOIS_FIG5_FIRST_FORCE_MN) <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = f_mid
    root = 0.5 * (lo + hi)
    force, _row, _state = _eval_neck(module, root, slope_scale)
    return root, force


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
    ax.set_title("Case 2b r_cl=1.54 mm t=0 neck-calibrated mesh")
    ax.grid(alpha=0.28)
    path = out_dir / "mesh_initial_meridian.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    module.plt.close(fig)
    return path


def _write_outputs(module, state, row: dict, out_dir: Path, *, neck_input_mm: float, slope_scale: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    force_mn = abs(float(row["top_gorge_pressure_total_axial"])) * 1.0e3
    title = (
        "Case 2b r_cl=1.54 mm t=0 neck-calibrated initial mesh\n"
        f"neck input={neck_input_mm:.6f} mm, |F_gorge|={force_mn:.6f} mN"
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

    summary = {
        "method": "fixed r_cl=1.54 mm with volume-preserving neck calibration at t=0",
        "solver_module": SOLVER_NAME,
        "force_model": "ddgclib/FHeron gorge pressure plus neck surface tension",
        "target_contact_radius_mm": TARGET_CONTACT_RADIUS_MM,
        "neck_input_mm": float(neck_input_mm),
        "slope_scale": float(slope_scale),
        "target_force_mn": PITOIS_FIG5_FIRST_FORCE_MN,
        "chosen": {
            "force_mn": force_mn,
            "force_error_mn": force_mn - PITOIS_FIG5_FIRST_FORCE_MN,
            "force_error_percent": 100.0 * (force_mn - PITOIS_FIG5_FIRST_FORCE_MN) / PITOIS_FIG5_FIRST_FORCE_MN,
            "rcl_mm": float(row["initial_geometry_contact_radius"]) * 1.0e3,
            "neck_radius_mm": float(row["pressure_neck_fit_radius"]) * 1.0e3,
            "d_over_r": float(row["d_over_r"]),
            "actual_volume_ul": float(row["initial_geometry_actual_volume_m3"]) * 1.0e9,
            "volume_rel_error": float(row["initial_geometry_volume_rel_error"]),
        },
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out/Case_2b_initialmesh_r154_t0_neck_calibrated")
    parser.add_argument("--neck-min-mm", type=float, default=DEFAULT_NECK_MIN_MM)
    parser.add_argument("--neck-max-mm", type=float, default=DEFAULT_NECK_MAX_MM)
    parser.add_argument("--slope-scale", type=float, default=DEFAULT_SLOPE_SCALE)
    args = parser.parse_args(argv)

    module = _load_solver()
    neck_mm, _force = _find_neck_root(
        module,
        neck_min_mm=float(args.neck_min_mm),
        neck_max_mm=float(args.neck_max_mm),
        slope_scale=float(args.slope_scale),
    )
    _force_mn, row, state = _eval_neck(module, neck_mm, float(args.slope_scale))
    summary = _write_outputs(module, state, row, Path(args.out_dir), neck_input_mm=neck_mm, slope_scale=float(args.slope_scale))
    chosen = summary["chosen"]
    print("Created r_cl=1.54 mm t=0 neck-calibrated initial mesh")
    print(f"neck input = {neck_mm:.9f} mm")
    print(f"r_cl = {chosen['rcl_mm']:.9f} mm")
    print(f"neck fit = {chosen['neck_radius_mm']:.9f} mm")
    print(f"|F_gorge| = {chosen['force_mn']:.9f} mN")
    print(f"target |F_exp| = {PITOIS_FIG5_FIRST_FORCE_MN:.9f} mN")
    print(f"volume = {chosen['actual_volume_ul']:.9f} uL")
    print(f"mesh = {summary['mesh_initial_msh']}")
    print(f"png = {summary['mesh_initial_png']}")
    print(f"summary = {summary['summary_json']}")


if __name__ == "__main__":
    main()
