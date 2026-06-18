#!/usr/bin/env python3
"""Run the Pitois et al. (2000) Fig. 6 approach cases.

The run starts from an approach t0 mesh and reverses the Case 12
moving-sphere velocity so the lower sphere approaches the fixed upper sphere.
The solver can keep a sparse checkpoint history for quick diagnostics or a
per-step history for smooth validation plots.

Literature/equation map used by these cases:
    Pitois, Moucheront, and Chateau, J. Colloid Interface Sci. 231 (2000),
    DOI: 10.1006/jcis.2000.7096, Fig. 6: two ruby spheres approach at
    v = 10 um/s with liquid 2 and V = 1.1 uL.  The open circles in the
    generated charts are digitized from Fig. 6.

    Capillary gorge report:
        F_gorge = Delta p_g pi r_g^2 + 2 pi gamma r_g.

    Newtonian Cauchy stress traction:
        sigma = -p I + mu (grad u + grad u^T),
        F_wall = integral_A sigma n dA.
    This is the standard form in Batchelor, An Introduction to Fluid Dynamics
    (1967), DOI: 10.1017/CBO9780511800955.

    Lubrication wall-pressure traction, motivated by Reynolds lubrication
    theory, Reynolds, Phil. Trans. R. Soc. Lond. 177 (1886), pp. 188-190,
    Eqs. (17), (21), (24), and (27), DOI: 10.1098/rstl.1886.0005:
        F_lub = integral_Awet sigma_lub n dA, with sigma_lub = -p_lub I,
        p_lub(r) = C_lub * 3 mu |dh/dt| (a_lub^2 - r^2) / h^3.

    The approach solver is adapted from the local PR #35 / Case 12 separation
    numerical recipe, but the approach folder carries its own local core copy
    so this contribution is self-contained inside cases_dynamic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CORE_SCRIPT = ROOT / "_ddgclib_case_core.py"
DEFAULT_INITIAL_MESH = ROOT / "fig6_t0_finial" / "mesh" / "fig6_approach_t0.msh"
DEFAULT_OUT_DIR = ROOT / Path(__file__).stem

PARTICLE_RADIUS_M = 4.0e-3
BRIDGE_VOLUME_M3 = 1.10e-9
SURFACE_TENSION_NPM = 21.0e-3
LIQUID_VISCOSITY_PAS = 100.0
LIQUID_DENSITY_KGPM3 = 965.0
LIQUID_BULK_MODULUS_PA = 9.7e8
APPROACH_START_D_OVER_R = 0.20
DEFAULT_APPROACH_RCL_MM = 1.071
# Pitois-style ruby-sphere validations are commonly reported/applied with a
# small hydrophilic static contact angle of about 10 deg.
APPROACH_STATIC_CONTACT_ANGLE_DEG = 10.0
# Fig. 6 reports v = 10 um/s. In this runner the top sphere is fixed and the
# lower sphere alone moves, so the CLI speed is also the gap-closing speed.
FIG6_APPROACH_SPEED_MPS = 10.0e-6
APPROACH_GORGE_PRESSURE_MODEL = "axisym_initial"


def ensure_hyperct_import() -> None:
    try:
        __import__("hyperct")
        return
    except ModuleNotFoundError:
        pass

    hyperct = types.ModuleType("hyperct")

    class Complex:  # pragma: no cover - fallback import shim only.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The optional hyperct package is not installed in this Python environment.")

    def compute_vd(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("The optional hyperct.ddg.compute_vd function is not available.")

    ddg = types.ModuleType("hyperct.ddg")
    ddg.compute_vd = compute_vd
    hyperct.Complex = Complex
    hyperct.ddg = ddg
    sys.modules.setdefault("hyperct", hyperct)
    sys.modules.setdefault("hyperct.ddg", ddg)


def load_case12_core():
    ensure_hyperct_import()
    if not CORE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing local Case 12 core copy: {CORE_SCRIPT}. "
            "This approach package expects the core copy to be committed next "
            "to run_fig6_approach_case12.py."
        )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("fig6_approach_case12_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Case 12 core from {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def disable_core_nonmesh_png_refresh(core: Any) -> None:
    """Keep Case 12 mesh snapshots without its generic per-step PNG refresh."""

    if hasattr(core, "USER_HISTORY_PNG_EVERY_STEPS"):
        core.USER_HISTORY_PNG_EVERY_STEPS = 0

    if hasattr(core, "_refresh_non_mesh_png_outputs"):
        def _skip_nonmesh_png_outputs(*_args: Any, **_kwargs: Any) -> list[Path]:
            return []

        core._refresh_non_mesh_png_outputs = _skip_nonmesh_png_outputs


def fig6_experiment_points() -> tuple[np.ndarray, np.ndarray]:
    """Approximate open-circle Fig. 6 approach points read from the local TIFF."""

    d_over_r = np.array(
        [
            0.200,
            0.190,
            0.180,
            0.170,
            0.160,
            0.150,
            0.140,
            0.130,
            0.120,
            0.110,
            0.100,
            0.090,
            0.080,
            0.070,
            0.060,
            0.050,
            0.045,
            0.040,
            0.035,
            0.030,
            0.027,
            0.024,
            0.021,
            0.018,
        ],
        dtype=float,
    )
    force_mn = np.array(
        [
            0.040,
            0.046,
            0.052,
            0.059,
            0.066,
            0.073,
            0.081,
            0.090,
            0.098,
            0.107,
            0.116,
            0.126,
            0.137,
            0.146,
            0.150,
            0.148,
            0.130,
            0.095,
            0.045,
            -0.020,
            -0.080,
            -0.160,
            -0.280,
            -0.380,
        ],
        dtype=float,
    )
    return d_over_r, force_mn


def build_config(core: Any, args: argparse.Namespace):
    cap_speed = -abs(float(args.approach_speed_um_s)) * 1.0e-6
    use_limit_projection = not bool(args.finite_compressible)
    return core.VolumetricPitoisConfig(
        name="Fig6_approach_case12_cl_growth",
        title="Pitois 2000 Fig. 6 approach: Case 12 recipe with moving contact line",
        initial_msh_path=str(Path(args.initial_mesh).resolve()),
        particle_radius=PARTICLE_RADIUS_M,
        initial_d_over_r=APPROACH_START_D_OVER_R,
        initial_bridge_volume_m3=BRIDGE_VOLUME_M3,
        gamma=SURFACE_TENSION_NPM,
        mu_f=LIQUID_VISCOSITY_PAS,
        rho_f=LIQUID_DENSITY_KGPM3,
        dt=float(args.dt),
        n_steps=int(args.steps),
        cap_speed=cap_speed,
        use_pitois_eq6_contact_radius=False,
        target_cap_radius=float(args.approach_rcl_mm) * 1.0e-3,
        initial_contact_radius_mode="loaded",
        pressure_closure="compressible",
        compressible_bulk_modulus_pa=float(args.bulk_modulus_pa),
        compressible_use_incompressible_limit_projection=use_limit_projection,
        gorge_neck_fit_radius_max_raw_ratio=float(args.gorge_neck_fit_radius_max_raw_ratio),
        gorge_pressure_model=str(args.gorge_pressure_model),
        solver_viscous_force_model="tet_cauchy",
        enable_gmsh_wall_stress_sphere_force=False,
        record_gmsh_wall_stress_history=True,
        gmsh_wall_stress_include_pressure=True,
        gmsh_wall_stress_include_viscous=True,
        gmsh_wall_stress_project_axisym=True,
        reported_gorge_add_wall_viscous_force=True,
        reported_gorge_add_cox_cl_force=True,
        reported_gorge_add_pr33_wall_pressure_force=False,
        contact_angle_deg=float(args.contact_angle_deg),
        include_gravity=True,
        gravity_mps2=9.81,
        enable_solver_pr33_pressure_force=True,
        solver_pr33_pressure_reference_mode="zero",
        solver_pr33_pressure_include_hydrostatic=False,
        solver_pr33_free_surface_dirichlet_pressure=False,
        enable_solver_hydrostatic_force=True,
        solver_hydrostatic_zref_mode="top_cl",
        allow_contact_line_growth=True,
        enable_gmsh_contact_angle_kinematics=True,
        enable_dynamic_contact_angle=bool(args.cox_contact_line_force),
        solver_pr33_pressure_contact_line_mobility=True,
        contact_line_max_slide_um=float(args.contact_line_max_slide_um),
        use_cox_contact_line_force=bool(args.cox_contact_line_force),
        enforce_no_swirl=False,
        record_every=int(args.record_every),
        sparse_history_only=not bool(args.history_every_step),
        mesh_snapshot_every=int(args.mesh_snapshot_every),
        render_raw_mesh_snapshot_png=False,
    )


def raw_force_for_fig6(row: dict[str, Any]) -> float:
    # In the loaded t0 mesh, attraction on the fixed/top sphere is negative z.
    # Fig. 6 plots attractive force positive, so flip that sign.
    return -float(row["fixed_force_axial"]) * 1.0e3


def force_for_fig6(row: dict[str, Any], _config: Any) -> float:
    return raw_force_for_fig6(row)


def force_report_label(config: Any) -> str:
    if bool(getattr(config, "solver_add_lubrication_pressure_force", False)):
        return "Raw sim force + lub. wall pressure"
    return "Raw sim force report"


def sparse_history(history: list[dict[str, Any]], record_every: int) -> list[dict[str, Any]]:
    keep: list[dict[str, Any]] = []
    for row in history:
        step = int(row.get("step", 0))
        if step == 0 or step % max(1, int(record_every)) == 0:
            keep.append(row)
    if history and keep[-1] is not history[-1]:
        keep.append(history[-1])
    return keep


def write_force_vs_time_png(
    *,
    history: list[dict[str, Any]],
    config: Any,
    out_dir: Path,
    record_every: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse = sparse_history(history, record_every)
    t_s = np.array([float(row["t"]) for row in history], dtype=float)
    force_mn = np.array([raw_force_for_fig6(row) for row in history], dtype=float)

    exp_d, exp_f = fig6_experiment_points()
    gap_rate = abs(float(config.cap_speed) * float(config.velocity_scale_mps))
    exp_t = (APPROACH_START_D_OVER_R - exp_d) * PARTICLE_RADIUS_M / max(gap_rate, 1.0e-30)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(
        exp_t,
        exp_f,
        s=28,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        label="Pitois Fig. 6 exp",
        zorder=3,
    )
    ax.plot(
        t_s,
        force_mn,
        color="#d62828",
        linewidth=2.0,
        label=force_report_label(config),
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.75)
    x_min = min(float(np.min(t_s)), float(np.min(exp_t)))
    x_max = max(float(np.max(t_s)), float(np.max(exp_t)))
    pad = 0.03 * max(x_max - x_min, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("F [mN] (attractive positive)")
    ax.set_title("Pitois Fig. 6 approach force vs time")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_times = np.asarray(ax.get_xticks(), dtype=float)
    tick_d = APPROACH_START_D_OVER_R - tick_times * gap_rate / PARTICLE_RADIUS_M
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([f"{value:.3f}" for value in tick_d])
    ax2.set_xlabel("D/R mapped from approach speed")
    fig.tight_layout()
    path = out_dir / "approach_force_vs_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(
    *,
    history: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    config: Any,
    out_dir: Path,
    chart_path: Path,
) -> Path:
    length_scale = float(getattr(config, "length_scale_m", 1.0))
    radius_scale = length_scale if bool(getattr(config, "dimensionless_active", False)) else 1.0
    volume_scale = float(getattr(config, "volume_scale_m3", 1.0)) if bool(getattr(config, "dimensionless_active", False)) else 1.0
    gamma_scale = (
        float(getattr(config, "surface_tension_scale_npm", 1.0))
        if bool(getattr(config, "dimensionless_active", False))
        else 1.0
    )
    viscosity_scale = (
        float(getattr(config, "viscosity_scale_pas", 1.0))
        if bool(getattr(config, "dimensionless_active", False))
        else 1.0
    )
    velocity_scale = (
        float(getattr(config, "velocity_scale_mps", 1.0))
        if bool(getattr(config, "dimensionless_active", False))
        else 1.0
    )
    acceleration_scale = (
        float(getattr(config, "acceleration_scale_mps2", 1.0))
        if bool(getattr(config, "dimensionless_active", False))
        else 1.0
    )
    approach_contact_radius_mm = float(config.target_cap_radius) * radius_scale * 1.0e3
    payload = {
        "case": "Fig. 6 approach using Case 12 settings and the fitted approach mesh",
        "initial_mesh": str(Path(config.initial_msh_path).resolve()),
        "physical_settings": {
            "particle_radius_mm": float(config.particle_radius) * radius_scale * 1.0e3,
            "bridge_volume_ul": float(config.initial_bridge_volume_m3) * volume_scale * 1.0e9,
            "surface_tension_mn_m": float(config.gamma) * gamma_scale * 1.0e3,
            "liquid_viscosity_pa_s": float(config.mu_f) * viscosity_scale,
            "liquid_density_kg_m3": float(config.rho_f),
            "liquid_bulk_modulus_pa": float(config.compressible_bulk_modulus_pa),
            "static_contact_angle_deg": float(config.contact_angle_deg),
            "initial_d_over_r": float(config.initial_d_over_r),
            "gap_closing_speed_um_s": abs(float(config.cap_speed) * velocity_scale) * 1.0e6,
            "gravity_included": bool(config.include_gravity),
            "gravity_m_s2": float(config.gravity_mps2) * acceleration_scale,
            "hydrostatic_force": bool(config.enable_solver_hydrostatic_force),
            "hydrostatic_reference": str(config.solver_hydrostatic_zref_mode),
            "pr33_hydrostatic_double_count": bool(config.solver_pr33_pressure_include_hydrostatic),
            "gorge_pressure_model": str(config.gorge_pressure_model),
        },
        "cox_contact_line_force": bool(config.use_cox_contact_line_force),
        "compressible_limit_projection": bool(config.compressible_use_incompressible_limit_projection),
        "allow_contact_line_growth": bool(config.allow_contact_line_growth),
        "gmsh_contact_angle_kinematics": bool(config.enable_gmsh_contact_angle_kinematics),
        "approach_contact_radius_mm": approach_contact_radius_mm,
        "history_row_count": len(history),
        "history_is_every_step": not bool(config.sparse_history_only),
        "mesh_snapshot_every": int(config.mesh_snapshot_every),
        "recorded_steps": [int(row["step"]) for row in sparse],
        "force_for_fig6_mn": [force_for_fig6(row, config) for row in sparse],
        "force_report": (
            "raw simulation force report: PR35/ddgclib solver includes gap-dependent lubrication pressure; "
            "the plotted force is the same raw reported sphere force with lubrication wall traction"
            if bool(getattr(config, "solver_add_lubrication_pressure_force", False))
            else (
            "raw simulation gorge-force report: capillary neck pressure plus raw PR33 neck pressure, "
            "then FHeron/gorge surface tension; no Eq. 10/11 post-processing"
            if str(config.gorge_pressure_model) == "axisym_initial_pr33_neck"
            else (
                "raw simulation gorge-force report: neck pressure term plus raw viscous Cauchy wall stress "
                "plus Cox contact-line reaction plus FHeron/gorge surface tension; no Eq. 10/11 post-processing"
                if bool(getattr(config, "reported_gorge_add_wall_viscous_force", False))
                else "raw simulation gorge-force report: neck pressure term plus FHeron/gorge surface tension; no Eq. 10/11 post-processing"
            )
            )
        ),
        "d_over_r": [float(row["d_over_r"]) for row in sparse],
        "t_s": [float(row["t"]) for row in sparse],
        "chart_png": str(chart_path.resolve()),
        "config": asdict(config),
    }
    path = out_dir / "approach_case12_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sparse_path = out_dir / "approach_history_record_every_1000.json"
    sparse_path.write_text(
        json.dumps({"config": asdict(config), "history": sparse}, indent=2),
        encoding="utf-8",
    )
    full_path = out_dir / "approach_history_full.json"
    full_path.write_text(
        json.dumps({"config": asdict(config), "history": history}, indent=2),
        encoding="utf-8",
    )
    return path


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-mesh", type=Path, default=DEFAULT_INITIAL_MESH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--record-every", type=int, default=1000)
    parser.add_argument(
        "--history-every-step",
        action="store_true",
        help="Keep every solver step in memory/results so the validation line is smooth.",
    )
    parser.add_argument(
        "--mesh-snapshot-every",
        type=int,
        default=0,
        help="Write raw .msh snapshots every N steps; 0 disables motion mesh snapshots.",
    )
    parser.add_argument("--approach-speed-um-s", type=float, default=FIG6_APPROACH_SPEED_MPS * 1.0e6)
    parser.add_argument("--approach-rcl-mm", type=float, default=DEFAULT_APPROACH_RCL_MM)
    parser.add_argument("--contact-angle-deg", type=float, default=APPROACH_STATIC_CONTACT_ANGLE_DEG)
    parser.add_argument("--contact-line-max-slide-um", type=float, default=5.0)
    parser.add_argument("--bulk-modulus-pa", type=float, default=LIQUID_BULK_MODULUS_PA)
    parser.add_argument(
        "--gorge-neck-fit-radius-max-raw-ratio",
        type=float,
        default=20.0,
        help="Maximum fitted/raw waist-radius ratio allowed in the raw gorge-force report.",
    )
    parser.add_argument(
        "--gorge-pressure-model",
        choices=(
            "axisym_fit",
            "axisym_initial",
            "axisym_initial_pr33_neck",
            "fheron_neck",
            "axisym_fheron_floor",
            "axisym_initial_fheron_neck",
        ),
        default=APPROACH_GORGE_PRESSURE_MODEL,
    )
    parser.add_argument("--cox-contact-line-force", action="store_true")
    parser.add_argument(
        "--finite-compressible",
        action="store_true",
        help="Disable the Case 12 K-infinity projection for faster finite-K compressible dynamics.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


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
    core = load_case12_core()
    disable_core_nonmesh_png_refresh(core)
    config = build_config(core, args)
    history = core.run_motion_case(
        config,
        out_dir=out_dir,
        save_fig=int(args.mesh_snapshot_every) > 0,
        save_results=True,
        verbose=bool(args.verbose),
    )
    sparse = sparse_history(history, int(args.record_every))
    chart = write_force_vs_time_png(
        history=history,
        config=config,
        out_dir=out_dir,
        record_every=int(args.record_every),
    )
    summary = write_summary(
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
        ", ".join(f"{force_for_fig6(row, config):.6g}" for row in sparse),
    )


if __name__ == "__main__":
    main()
