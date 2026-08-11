#!/usr/bin/env python3
"""Create a Case12-style diagnostic GIF for the Pitois Fig. 6 approach run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

try:
    import meshio
except ModuleNotFoundError:
    meshio = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DIAGNOSTIC_GIF_SCRIPT = SCRIPT_DIR / "render_case_diagnostic_gif.py"
MESH_BATCH_SCRIPT = SCRIPT_DIR / "mesh_batch.py"
APPROACH_RUN_SCRIPT = ROOT / "run_fig6_approach_case12.py"
DEFAULT_RUN_DIR = ROOT / "case_1_finial"
DEFAULT_OUTPUT_NAME = "1window_diagnostic_variables_vs_t.gif"
DEFAULT_FRAME_DIR = "1window_diagnostic_variables_vs_t_frames"
PPT_CROP_THRESHOLD = 8
PPT_CROP_PAD = 18


def _load_module(path: Path, module_name: str):
    parent = str(path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


diag = _load_module(DIAGNOSTIC_GIF_SCRIPT, "fig6_approach_case12_diag_base")
approach_runner = _load_module(APPROACH_RUN_SCRIPT, "fig6_approach_case12_runner")


def _read_mesh(path: Path):
    if meshio is None:
        raise ModuleNotFoundError(
            "meshio is required to render mesh PNGs from .msh files; "
            "use --skip-mesh-png-render when existing mesh PNGs are available"
        )
    return meshio.read(path)


def _ensure_step0_mesh(run_dir: Path) -> None:
    initial = run_dir / "mesh_initial.msh"
    step0 = run_dir / "mesh_iter0000.msh"
    if not initial.is_file():
        return
    if not step0.is_file() or initial.stat().st_mtime_ns > step0.stat().st_mtime_ns:
        shutil.copy2(initial, step0)


def render_mesh_pngs(
    run_dir: Path,
    *,
    output_suffix: str,
    approach_speed_mps: float,
    dt_s: float,
    view_elev: float = 7.0,
    view_azim: float = 45.0,
    view_label: str = "oblique",
) -> None:
    _ensure_step0_mesh(run_dir)
    mesh_batch = _load_module(MESH_BATCH_SCRIPT, "fig6_approach_case12_mesh_batch")
    reference_mesh = run_dir / "mesh_iter0000.msh"
    if not reference_mesh.is_file():
        reference_mesh = run_dir / "mesh_initial.msh"
    reference = _read_mesh(reference_mesh)
    reference_points = np.asarray(reference.points, dtype=float)
    reference_faces = mesh_batch._boundary_faces_from_tets(mesh_batch._tetra_blocks(reference))
    side_mask = mesh_batch._liquid_air_vertex_mask(reference_points)
    side_faces = reference_faces[np.all(side_mask[reference_faces], axis=1)]
    if side_faces.size == 0:
        raise RuntimeError(f"Could not extract t0 free-surface faces from {reference_mesh}")
    wire_edges = _structured_wire_edges(side_faces, reference_points, mesh_batch)
    sphere_centers_ref = mesh_batch._sphere_geometry_from_bridge(
        reference_points,
        side_faces,
        sphere_radius_m=float(approach_runner.PARTICLE_RADIUS_M),
    )[:2]

    msh_files = sorted(run_dir.glob("mesh_iter*.msh"), key=_mesh_sort_key)
    fixed_axis_limits_mm = _fixed_axis_limits_mm(
        msh_files=msh_files,
        side_faces=side_faces,
        mesh_batch=mesh_batch,
        reference_sphere_centers=sphere_centers_ref,
        sphere_radius_m=float(approach_runner.PARTICLE_RADIUS_M),
        approach_speed_mps=float(abs(approach_speed_mps)),
        dt_s=float(dt_s),
    )
    print(f"Rendering clean approach mesh PNGs ({view_label}): {len(msh_files)} frame(s)")
    for index, msh_path in enumerate(msh_files, start=1):
        png_path = msh_path.with_name(f"{msh_path.stem}{output_suffix}.png")
        _render_clean_mesh_png(
            mesh_path=msh_path,
            png_path=png_path,
            side_faces=side_faces,
            wire_edges=wire_edges,
            mesh_batch=mesh_batch,
            reference_sphere_centers=sphere_centers_ref,
            sphere_radius_m=float(approach_runner.PARTICLE_RADIUS_M),
            approach_speed_mps=float(abs(approach_speed_mps)),
            dt_s=float(dt_s),
            fixed_axis_limits_mm=fixed_axis_limits_mm,
            view_elev=float(view_elev),
            view_azim=float(view_azim),
            view_label=str(view_label),
        )
        if _mesh_iteration(msh_path) % 1000 == 0 or index == len(msh_files):
            print(f"[{index}/{len(msh_files)}] {msh_path.name} -> {png_path.name}")


def _mesh_iteration(path: Path) -> int:
    stem = path.stem
    marker = "mesh_iter"
    if stem.startswith(marker):
        suffix = stem[len(marker) :]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _mesh_sort_key(path: Path) -> tuple[int, str]:
    return (_mesh_iteration(path), path.name)


def _structured_wire_edges(
    faces: np.ndarray,
    reference_points: np.ndarray,
    mesh_batch,
) -> np.ndarray:
    edges = np.asarray(mesh_batch._triangle_edges(faces), dtype=int)
    if edges.size == 0:
        return edges
    pts = np.asarray(reference_points, dtype=float)
    z = pts[:, 2]
    radii = np.linalg.norm(pts[:, :2], axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    z_tol = max(5.0e-12, 1.0e-7 * max(float(np.ptp(z)), 1.0e-12))
    angle_tol = 2.5e-2

    keep: list[tuple[int, int]] = []
    for a_raw, b_raw in edges:
        a = int(a_raw)
        b = int(b_raw)
        same_ring = abs(float(z[a] - z[b])) <= z_tol
        if same_ring:
            keep.append((a, b))
            continue
        if min(float(radii[a]), float(radii[b])) <= 1.0e-12:
            continue
        angle_delta = abs(float(np.arctan2(np.sin(theta[a] - theta[b]), np.cos(theta[a] - theta[b]))))
        if angle_delta <= angle_tol:
            keep.append((a, b))
    if len(keep) < max(8, edges.shape[0] // 8):
        return edges
    return np.asarray(keep, dtype=int)


def _fixed_axis_limits_mm(
    *,
    msh_files: list[Path],
    side_faces: np.ndarray,
    mesh_batch,
    reference_sphere_centers: tuple[np.ndarray, np.ndarray],
    sphere_radius_m: float,
    approach_speed_mps: float,
    dt_s: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Use one plot box for every frame so the fixed top sphere stays fixed."""

    if not msh_files:
        span = 2.0
        return ((-span, span), (-span, span), (-span, span))

    surface_vertices = np.unique(np.asarray(side_faces, dtype=int).reshape(-1))
    bottom_ref, top_ref = reference_sphere_centers
    sampled_points: list[np.ndarray] = []

    for msh_path in msh_files:
        mesh = _read_mesh(msh_path)
        points = np.asarray(mesh.points, dtype=float)
        if int(np.max(side_faces)) >= points.shape[0]:
            continue
        step = _mesh_iteration(msh_path)
        bottom_center = np.asarray(bottom_ref, dtype=float) + np.array(
            [0.0, 0.0, float(step) * float(dt_s) * float(approach_speed_mps)],
            dtype=float,
        )
        top_center = np.asarray(top_ref, dtype=float)
        bottom_contact_z, top_contact_z, _bottom_contact_radius, _top_contact_radius = (
            mesh_batch._bridge_contact_endpoints(points, side_faces)
        )
        sampled_points.append(1.0e3 * points[surface_vertices])
        for center_m, contact_z_m in (
            (bottom_center, bottom_contact_z),
            (top_center, top_contact_z),
        ):
            phi_min, phi_max = mesh_batch._sphere_cap_phi_limits(
                center_m,
                float(contact_z_m),
                float(sphere_radius_m),
                0.10,
            )
            if phi_max <= phi_min:
                continue
            sx, sy, sz = mesh_batch._sphere_surface_xyz(
                center_m,
                float(sphere_radius_m),
                phi_min=phi_min,
                phi_max=phi_max,
            )
            sampled_points.append(np.column_stack((sx.reshape(-1), sy.reshape(-1), sz.reshape(-1))))

    if not sampled_points:
        plot_points = 1.0e3 * np.asarray(_read_mesh(msh_files[0]).points, dtype=float)
        sampled_points.append(plot_points[surface_vertices])

    all_pts = np.vstack(sampled_points)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    half = max(1.55, 0.54 * span)
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (float(center[2] - half), float(center[2] + half)),
    )


def _render_clean_mesh_png(
    *,
    mesh_path: Path,
    png_path: Path,
    side_faces: np.ndarray,
    wire_edges: np.ndarray,
    mesh_batch,
    reference_sphere_centers: tuple[np.ndarray, np.ndarray],
    sphere_radius_m: float,
    approach_speed_mps: float,
    dt_s: float,
    fixed_axis_limits_mm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
    view_elev: float,
    view_azim: float,
    view_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    mesh = _read_mesh(mesh_path)
    points = np.asarray(mesh.points, dtype=float)
    if int(np.max(side_faces)) >= points.shape[0]:
        raise RuntimeError(f"{mesh_path} node count does not match the t0 side-surface topology")

    step = _mesh_iteration(mesh_path)
    bottom_ref, top_ref = reference_sphere_centers
    bottom_center = np.asarray(bottom_ref, dtype=float) + np.array(
        [0.0, 0.0, float(step) * float(dt_s) * float(approach_speed_mps)],
        dtype=float,
    )
    top_center = np.asarray(top_ref, dtype=float)
    bottom_contact_z, top_contact_z, _bottom_contact_radius, _top_contact_radius = (
        mesh_batch._bridge_contact_endpoints(points, side_faces)
    )

    plot_points = 1.0e3 * points
    triangles_xyz = plot_points[np.asarray(side_faces, dtype=int)]
    edge_segments = plot_points[np.asarray(wire_edges, dtype=int)] if wire_edges.size else np.empty((0, 2, 3))
    surface_vertices = np.unique(np.asarray(side_faces, dtype=int).reshape(-1))

    fig = plt.figure(figsize=(8.4, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    axis_points = [plot_points[surface_vertices]]
    for center_m, contact_z_m in (
        (bottom_center, bottom_contact_z),
        (top_center, top_contact_z),
    ):
        phi_min, phi_max = mesh_batch._sphere_cap_phi_limits(
            center_m,
            float(contact_z_m),
            float(sphere_radius_m),
            0.10,
        )
        if phi_max <= phi_min:
            continue
        sx, sy, sz = mesh_batch._sphere_surface_xyz(
            center_m,
            float(sphere_radius_m),
            phi_min=phi_min,
            phi_max=phi_max,
        )
        ax.plot_surface(
            sx,
            sy,
            sz,
            facecolors=mesh_batch._sphere_facecolors(
                sx,
                sy,
                sz,
                center_m,
                float(sphere_radius_m),
                alpha=0.78,
            ),
            linewidth=0.0,
            antialiased=True,
            shade=False,
            zorder=1,
            axlim_clip=True,
        )
        axis_points.append(np.column_stack((sx.reshape(-1), sy.reshape(-1), sz.reshape(-1))))

    ax.add_collection3d(
        Poly3DCollection(
            triangles_xyz,
            facecolor="#acd2e9",
            edgecolor="none",
            alpha=0.96,
            zorder=10,
            axlim_clip=True,
        )
    )
    if edge_segments.size:
        ax.add_collection3d(
            Line3DCollection(
                edge_segments,
                colors="#5C748B",
                linewidths=0.38,
                alpha=0.58,
                zorder=11,
                axlim_clip=True,
            )
        )

    if fixed_axis_limits_mm is None:
        all_pts = np.vstack(axis_points)
        center = np.mean(all_pts, axis=0)
        span = max(float(np.max(np.ptp(all_pts, axis=0))), 1.0)
        half = max(1.55, 0.54 * span)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
    else:
        ax.set_xlim(*fixed_axis_limits_mm[0])
        ax.set_ylim(*fixed_axis_limits_mm[1])
        ax.set_zlim(*fixed_axis_limits_mm[2])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(
        f"{mesh_path.stem} ({view_label}): liquid-air side surface + sphere caps\n"
        f"surface vertices={len(surface_vertices)}, surface faces={len(side_faces)}",
        pad=12,
    )
    ax.set_proj_type("ortho")
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_history(run_dir: Path, max_step: int) -> list[dict]:
    candidates = [
        run_dir / "approach_history_full.json",
        run_dir / "separation_history.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history")
        if isinstance(history, list) and history:
            return [row for row in history if int(row.get("step", 0)) <= int(max_step)]
    raise RuntimeError(f"No approach history JSON found in {run_dir}")


def _series(history: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray([float(row.get(key, default)) for row in history], dtype=float)


def _prepare_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    target_volume = max(float(history[0].get("initial_geometry_target_volume_m3", 1.0)), 1.0e-30)
    dual_volume = _series(history, "dual_cell_sum_m3", target_volume)
    snapshot_volume = _series(history, "snapshot_msh_volume_m3", target_volume)
    return {
        "step": _series(history, "step"),
        "t": _series(history, "t"),
        "d_over_r": _series(history, "d_over_r"),
        "gap_um": _series(history, "gap") * 1.0e6,
        "force_mn": -_series(history, "fixed_force_axial") * 1.0e3,
        "r_top_mm": _series(history, "top_contact_radius") * 1.0e3,
        "r_bottom_mm": _series(history, "bottom_contact_radius") * 1.0e3,
        "r_neck_mm": _series(history, "pressure_neck_fit_radius") * 1.0e3,
        "dual_vol_rel_pct": (dual_volume / target_volume - 1.0) * 100.0,
        "snapshot_vol_rel_pct": (snapshot_volume / target_volume - 1.0) * 100.0,
        "position_restore_after_pct": _series(history, "position_volume_constraint_rel_after") * 100.0,
        "pressure_scalar_pa": _series(history, "pressure_scalar"),
        "max_dynamic_pressure_pa": _series(history, "solver_pr33_pressure_delta_linf_Pa"),
        "max_speed_um_s": _series(history, "max_free_speed") * 1.0e6,
    }


def _force_report_label(history: list[dict]) -> str:
    has_lubrication_wall_pressure = any(
        "lubrication_pressure" in str(row.get("force_model", ""))
        or abs(float(row.get("lubrication_pressure_force_n", 0.0))) > 0.0
        for row in history
    )
    if has_lubrication_wall_pressure:
        return "raw sim force + lub. wall pressure"
    return "raw sim gorge-force report"


def _experiment_time_series(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    exp_d, exp_force_mn = approach_runner.fig6_experiment_points()
    t = arrays["t"]
    d = arrays["d_over_r"]
    d0 = float(d[0])
    if len(t) >= 2 and abs(float(t[-1] - t[0])) > 1.0e-30:
        d_rate = abs(float(d[-1] - d[0]) / float(t[-1] - t[0]))
    else:
        d_rate = abs(float(approach_runner.FIG6_APPROACH_SPEED_MPS) / float(approach_runner.PARTICLE_RADIUS_M))
    exp_t = (d0 - np.asarray(exp_d, dtype=float)) / max(d_rate, 1.0e-30)
    return exp_t, np.asarray(exp_force_mn, dtype=float)


def _history_index_for_step(history: list[dict], step: int) -> int:
    steps = np.asarray([int(row.get("step", 0)) for row in history], dtype=int)
    return int(np.argmin(np.abs(steps - int(step))))


def _trim_image_top(arr: np.ndarray, fraction: float) -> np.ndarray:
    trim_rows = int(round(float(fraction) * float(arr.shape[0])))
    if trim_rows <= 0 or trim_rows >= arr.shape[0] - 1:
        return arr
    return arr[trim_rows:, :, :]


def _mesh_png_steps_for_suffix(run_dir: Path, output_suffix: str) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    suffix = str(output_suffix)
    for path in sorted(run_dir.glob(f"mesh_iter*{suffix}.png")):
        stem = path.name.removeprefix("mesh_iter")
        step_text = stem.split("_", 1)[0]
        if step_text.isdigit():
            frames.setdefault(int(step_text), path)
    return frames


def _mesh_frame_for_step(
    run_dir: Path,
    step: int,
    *,
    output_suffix: str = "_meshbatch100_fixedtop",
) -> tuple[Path, int]:
    stem = f"mesh_iter{int(step):04d}"
    candidates = [run_dir / f"{stem}{output_suffix}.png"]
    if str(output_suffix) == "_meshbatch100_fixedtop":
        candidates.extend(
            [
                run_dir / f"{stem}_meshbatch100.png",
                run_dir / f"{stem}.png",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate, int(step)
    saved_frames = _mesh_png_steps_for_suffix(run_dir, output_suffix)
    if not saved_frames and str(output_suffix) == "_meshbatch100_fixedtop":
        saved_frames = diag._mesh_png_steps(run_dir)
    if not saved_frames:
        raise FileNotFoundError(f"No mesh PNG found in {run_dir}")
    saved_steps = sorted(saved_frames)
    prior_steps = [saved_step for saved_step in saved_steps if saved_step <= int(step)]
    saved_step = prior_steps[-1] if prior_steps else saved_steps[0]
    return saved_frames[saved_step], int(saved_step)


def _frame_steps(
    run_dir: Path,
    max_step: int,
    *,
    output_suffix: str = "_meshbatch100_fixedtop",
) -> list[int]:
    steps = sorted(
        step for step in _mesh_png_steps_for_suffix(run_dir, output_suffix) if step <= int(max_step)
    )
    if not steps and str(output_suffix) == "_meshbatch100_fixedtop":
        steps = sorted(step for step in diag._mesh_png_steps(run_dir) if step <= int(max_step))
    if not steps:
        raise RuntimeError(f"No mesh PNG frames found in {run_dir}")
    if int(max_step) not in steps:
        steps.append(int(max_step))
    return sorted(set(steps))


def _render_frame(
    *,
    run_dir: Path,
    case_label: str,
    arrays: dict[str, np.ndarray],
    history: list[dict],
    step: int,
    out_path: Path,
    font_scale: float,
    mesh_zoom: float,
    oblique_mesh_zoom_factor: float,
    dual_mesh_xz: bool,
    mesh_output_suffix: str,
    xz_mesh_output_suffix: str,
    mesh_trim_top_frac: float,
    xz_mesh_trim_top_frac: float,
    mesh_fill_panel: bool,
    mesh_square_panel: bool,
    mesh_top_panel_ratio: float,
    mesh_bottom_panel_ratio: float,
    xz_mesh_zoom_factor: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.style.use(str(diag.NEATPLOT_STYLE))
    plt.rcParams["figure.constrained_layout.use"] = False

    idx = _history_index_for_step(history, step)
    row = history[idx]
    t = arrays["t"]
    t_now = float(arrays["t"][idx])
    t_end = float(max(np.max(t), np.max(_experiment_time_series(arrays)[0])))
    exp_t_s, exp_force_mn = _experiment_time_series(arrays)
    exp_mask = (exp_t_s >= 0.0) & (exp_t_s <= t_end)

    title_fs = 10.0 * font_scale
    label_fs = 8.5 * font_scale
    tick_fs = 7.8 * font_scale
    legend_fs = 7.0 * font_scale
    note_fs = 8.1 * font_scale

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=(1.03, 1.18),
        height_ratios=(1.0, 0.82, 0.82),
        left=0.035,
        right=0.960,
        bottom=0.070,
        top=0.940,
        wspace=0.22,
        hspace=0.48,
    )

    def _draw_mesh_panel(
        ax,
        *,
        output_suffix: str,
        title_prefix: str,
        trim_top_frac: float,
        image_zoom: float,
        square_box: bool,
    ) -> None:
        mesh_path, mesh_step = _mesh_frame_for_step(
            run_dir,
            step,
            output_suffix=output_suffix,
        )
        mesh_img = diag._read_cropped_image(mesh_path, zoom=image_zoom)
        mesh_img = _trim_image_top(mesh_img, trim_top_frac)
        if bool(square_box):
            ax.set_box_aspect(1.0)
        ax.imshow(
            mesh_img,
            aspect="equal" if bool(square_box) else ("auto" if bool(mesh_fill_panel) else "equal"),
        )
        ax.set_axis_off()
        if mesh_step == int(step):
            mesh_title = f"{title_prefix}, step {step:04d}"
        else:
            mesh_title = f"{title_prefix}, history step {step:04d}; saved mesh {mesh_step:04d}"
        ax.text(
            0.5,
            0.955,
            mesh_title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=title_fs,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )

    if bool(dual_mesh_xz):
        left_bbox = gs[:2, 0].get_position(fig)
        save_dpi = 120.0
        fig_w_px = fig.get_size_inches()[0] * save_dpi
        fig_h_px = fig.get_size_inches()[1] * save_dpi
        top_panel_px = 370.0 * float(mesh_top_panel_ratio)
        bottom_panel_px = 320.0 * float(mesh_bottom_panel_ratio)
        top_panel_w = top_panel_px / fig_w_px
        top_panel_h = top_panel_px / fig_h_px
        bottom_panel_w = bottom_panel_px / fig_w_px
        bottom_panel_h = bottom_panel_px / fig_h_px
        panel_gap = 4.0 / fig_h_px
        text_gap = 4.0 / fig_h_px
        top_margin = 1.0 / fig_h_px
        text_bottom = 4.0 / fig_h_px
        top_panel_x = left_bbox.x0 + 0.5 * (left_bbox.width - top_panel_w)
        bottom_panel_x = left_bbox.x0 + 0.5 * (left_bbox.width - bottom_panel_w)
        top_y = 1.0 - top_margin - top_panel_h
        bottom_y = top_y - panel_gap - bottom_panel_h
        text_top = max(text_bottom + 0.01, bottom_y - text_gap)
        text_axes_rect = [left_bbox.x0, text_bottom, left_bbox.width, text_top - text_bottom]
        ax_mesh = fig.add_axes([top_panel_x, top_y, top_panel_w, top_panel_h])
        _draw_mesh_panel(
            ax_mesh,
            output_suffix=mesh_output_suffix,
            title_prefix="Mesh snapshot, oblique view",
            trim_top_frac=mesh_trim_top_frac,
            image_zoom=mesh_zoom * float(oblique_mesh_zoom_factor),
            square_box=bool(mesh_square_panel),
        )
        ax_mesh_xz = fig.add_axes([bottom_panel_x, bottom_y, bottom_panel_w, bottom_panel_h])
        _draw_mesh_panel(
            ax_mesh_xz,
            output_suffix=xz_mesh_output_suffix,
            title_prefix="Mesh snapshot, XZ side view",
            trim_top_frac=xz_mesh_trim_top_frac,
            image_zoom=mesh_zoom * float(xz_mesh_zoom_factor),
            square_box=bool(mesh_square_panel),
        )
    else:
        ax_mesh = fig.add_subplot(gs[:2, 0])
        _draw_mesh_panel(
            ax_mesh,
            output_suffix=mesh_output_suffix,
            title_prefix="Mesh snapshot",
            trim_top_frac=mesh_trim_top_frac,
            image_zoom=mesh_zoom * float(oblique_mesh_zoom_factor),
            square_box=bool(mesh_square_panel),
        )

    if bool(dual_mesh_xz):
        ax_text = fig.add_axes(text_axes_rect)
    else:
        ax_text = fig.add_subplot(gs[2, 0])
    ax_text.set_axis_off()
    target_volume = max(float(row.get("initial_geometry_target_volume_m3", 1.0)), 1.0e-30)
    text = (
        f"{case_label}\n"
        f"t = {t_now:7.3f} s    step = {int(row.get('step', 0)):4d}\n"
        f"D/R = {float(row.get('d_over_r', 0.0)):.6f}    gap = {float(row.get('gap', 0.0))*1e6:.2f} um\n"
        f"F = {-float(row.get('fixed_force_axial', 0.0))*1e3:.4f} mN    "
        f"pressure = {float(row.get('pressure_scalar', 0.0)):.3f} Pa\n"
        f"r_neck = {float(row.get('pressure_neck_fit_radius', 0.0))*1e3:.4f} mm    "
        f"r_top/r_bot = {float(row.get('top_contact_radius', 0.0))*1e3:.4f}/"
        f"{float(row.get('bottom_contact_radius', 0.0))*1e3:.4f} mm\n"
        f"volume rel = {(float(row.get('dual_cell_sum_m3', 0.0)) / target_volume - 1.0)*100.0:+.3e} %\n"
        f"max free speed = {float(row.get('max_free_speed', 0.0))*1e6:.3f} um/s\n"
        f"force model: {row.get('force_model', 'unknown')}"
    )
    ax_text.text(0.0, 0.98, text, va="top", ha="left", family="monospace", fontsize=note_fs)

    ax_chart = fig.add_subplot(gs[0, 1])
    ax_chart.scatter(
        exp_t_s[exp_mask],
        exp_force_mn[exp_mask],
        s=18,
        facecolors="white",
        edgecolors="#111111",
        linewidths=0.9,
        label="Pitois Fig. 6 exp",
        zorder=4,
    )
    ax_chart.plot(t, arrays["force_mn"], color="#d62828", linewidth=1.55, label=_force_report_label(history))
    ax_chart.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.75)
    diag._draw_marker(ax_chart, t_now, float(arrays["force_mn"][idx]), "#d62828")
    ax_chart.set_xlim(0.0, t_end)
    ax_chart.set_ylim(*diag._span_ylim([exp_force_mn[exp_mask], arrays["force_mn"], np.asarray([0.0])]))
    ax_chart.set_ylabel("F [mN]", fontsize=label_fs)
    ax_chart.set_title("chart_vs_t: Pitois Fig. 6 force comparison", fontsize=title_fs, pad=2)
    ax_chart.tick_params(labelsize=tick_fs)
    ax_chart.grid(True, alpha=0.28)
    ax_chart.legend(fontsize=legend_fs, loc="best")

    ax_geom = fig.add_subplot(gs[1, 1])
    ax_geom.plot(t, arrays["r_neck_mm"], color="#003049", linewidth=1.25, label="r_neck")
    ax_geom.plot(t, arrays["r_top_mm"], color="#f77f00", linewidth=1.0, label="r_top CL")
    ax_geom.plot(t, arrays["r_bottom_mm"], color="#6a4c93", linewidth=1.0, label="r_bottom CL")
    diag._draw_marker(ax_geom, t_now, float(arrays["r_neck_mm"][idx]), "#003049")
    ax_geom.set_xlim(0.0, t_end)
    ax_geom.set_ylim(*diag._span_ylim([arrays["r_neck_mm"], arrays["r_top_mm"], arrays["r_bottom_mm"]]))
    ax_geom.set_ylabel("mm", fontsize=label_fs)
    ax_geom.set_title("Geometry radii", fontsize=title_fs, pad=2)
    ax_geom.tick_params(labelsize=tick_fs)
    ax_geom.grid(True, alpha=0.28)
    ax_geom.legend(fontsize=legend_fs, loc="upper right")

    ax_vol = fig.add_subplot(gs[2, 1])
    ax_vol.plot(t, arrays["dual_vol_rel_pct"], color="#1d3557", linewidth=1.25, label="dual volume")
    ax_vol.plot(t, arrays["snapshot_vol_rel_pct"], color="#e76f51", linewidth=1.0, label="snapshot msh")
    ax_vol.plot(
        t,
        arrays["position_restore_after_pct"],
        color="#7f7f7f",
        linewidth=0.9,
        linestyle=":",
        label="restore residual",
    )
    diag._draw_marker(ax_vol, t_now, float(arrays["dual_vol_rel_pct"][idx]), "#1d3557")
    ax_vol.set_xlim(0.0, t_end)
    ax_vol.set_ylim(
        *diag._span_ylim(
            [
                arrays["dual_vol_rel_pct"],
                arrays["snapshot_vol_rel_pct"],
                arrays["position_restore_after_pct"],
            ],
            pad=0.15,
        )
    )
    ax_vol.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_vol.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_vol.set_ylabel("%", fontsize=label_fs)
    ax_vol.set_xlabel("t [s]", fontsize=label_fs)
    ax_vol.set_title("Volume residuals with max dynamic pressure", fontsize=title_fs, pad=2)
    ax_vol.tick_params(labelsize=tick_fs)
    ax_vol.grid(True, alpha=0.28)
    ax_pressure = ax_vol.twinx()
    ax_pressure.plot(t, arrays["max_dynamic_pressure_pa"], color="#bc4749", linewidth=1.15, label="max dynamic pressure")
    ax_pressure.scatter([t_now], [float(arrays["max_dynamic_pressure_pa"][idx])], s=18, color="#bc4749", zorder=6)
    ax_pressure.set_ylim(*diag._span_ylim([arrays["max_dynamic_pressure_pa"]]))
    ax_pressure.set_ylabel("Pa", fontsize=label_fs)
    ax_pressure.tick_params(labelsize=tick_fs)
    lines1, labels1 = ax_vol.get_legend_handles_labels()
    lines2, labels2 = ax_pressure.get_legend_handles_labels()
    ax_vol.legend(lines1 + lines2, labels1 + labels2, fontsize=legend_fs, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_gif(
    run_dir: Path,
    *,
    case_label: str,
    max_step: int,
    duration_ms: int,
    output_name: str,
    frame_dir_name: str,
    font_scale: float,
    mesh_zoom: float,
    oblique_mesh_zoom_factor: float,
    dual_mesh_xz: bool,
    mesh_output_suffix: str,
    xz_mesh_output_suffix: str,
    mesh_trim_top_frac: float,
    xz_mesh_trim_top_frac: float,
    mesh_fill_panel: bool,
    mesh_square_panel: bool,
    mesh_top_panel_ratio: float,
    mesh_bottom_panel_ratio: float,
    xz_mesh_zoom_factor: float,
) -> Path:
    history = _load_history(run_dir, max_step=max_step)
    arrays = _prepare_arrays(history)
    frame_steps = _frame_steps(run_dir, max_step=max_step, output_suffix=mesh_output_suffix)
    frames_dir = run_dir / frame_dir_name
    frame_paths: list[Path] = []
    print(f"Rendering {case_label}: {len(frame_steps)} frames")
    for index, step in enumerate(frame_steps, start=1):
        frame_path = frames_dir / f"diagnostic_step{step:04d}.png"
        _render_frame(
            run_dir=run_dir,
            case_label=case_label,
            arrays=arrays,
            history=history,
            step=step,
            out_path=frame_path,
            font_scale=font_scale,
            mesh_zoom=mesh_zoom,
            oblique_mesh_zoom_factor=float(oblique_mesh_zoom_factor),
            dual_mesh_xz=bool(dual_mesh_xz),
            mesh_output_suffix=str(mesh_output_suffix),
            xz_mesh_output_suffix=str(xz_mesh_output_suffix),
            mesh_trim_top_frac=float(mesh_trim_top_frac),
            xz_mesh_trim_top_frac=float(xz_mesh_trim_top_frac),
            mesh_fill_panel=bool(mesh_fill_panel),
            mesh_square_panel=bool(mesh_square_panel),
            mesh_top_panel_ratio=float(mesh_top_panel_ratio),
            mesh_bottom_panel_ratio=float(mesh_bottom_panel_ratio),
            xz_mesh_zoom_factor=float(xz_mesh_zoom_factor),
        )
        frame_paths.append(frame_path)
        if int(step) % 1000 == 0 or index == len(frame_steps):
            print(f"[{index}/{len(frame_steps)}] step {int(step):04d}: {frame_path.name}")

    gif_path = run_dir / output_name
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=False,
        )
    finally:
        for frame in frames:
            frame.close()
    print(gif_path)
    return gif_path


def _frame_bbox(frame: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = frame.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda value: 255 if value > PPT_CROP_THRESHOLD else 0)
    return mask.getbbox()


def _union_bbox(
    boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left = max(min(box[0] for box in boxes) - PPT_CROP_PAD, 0)
    top = max(min(box[1] for box in boxes) - PPT_CROP_PAD, 0)
    right = min(max(box[2] for box in boxes) + PPT_CROP_PAD, size[0])
    bottom = min(max(box[3] for box in boxes) + PPT_CROP_PAD, size[1])
    return left, top, right, bottom


def optimize_gif_for_ppt(src: Path, dst: Path) -> Path:
    image = Image.open(src)
    frames: list[Image.Image] = []
    boxes: list[tuple[int, int, int, int]] = []
    durations: list[int] = []
    try:
        for index in range(getattr(image, "n_frames", 1)):
            image.seek(index)
            frame = image.convert("RGB")
            frames.append(frame.copy())
            durations.append(int(image.info.get("duration", 140)))
            box = _frame_bbox(frame)
            if box is not None:
                boxes.append(box)
        if not boxes:
            raise RuntimeError(f"No visible content found in {src}")
        crop = _union_bbox(boxes, frames[0].size)
        cropped = [frame.crop(crop).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            cropped[0].save(
                dst,
                save_all=True,
                append_images=cropped[1:],
                duration=durations,
                loop=0,
                optimize=False,
            )
        finally:
            for frame in cropped:
                frame.close()
    finally:
        for frame in frames:
            frame.close()
        image.close()
    print(dst)
    return dst


def _ppt_output_names(output_name: str) -> tuple[str, str]:
    if output_name.endswith("_ppt_optimized.gif"):
        return output_name.replace("_ppt_optimized.gif", "_ppt_raw.gif"), output_name
    if output_name.endswith(".gif"):
        stem = output_name[: -len(".gif")]
        return f"{stem}_ppt_raw.gif", f"{stem}_ppt_optimized.gif"
    return f"{output_name}_ppt_raw.gif", f"{output_name}_ppt_optimized.gif"


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-step", type=int, default=7500)
    parser.add_argument("--duration-ms", type=int, default=140)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--frame-dir-name", default=DEFAULT_FRAME_DIR)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--mesh-zoom", type=float, default=1.0)
    parser.add_argument(
        "--oblique-mesh-zoom-factor",
        type=float,
        default=1.20,
        help="Additional image zoom multiplier for the upper oblique mesh panel.",
    )
    parser.add_argument("--case-label", default="Fig. 6 approach Case12-style raw report")
    parser.add_argument(
        "--ppt-optimized",
        action="store_true",
        help="Write a cropped presentation GIF plus the uncropped *_ppt_raw.gif source.",
    )
    parser.add_argument("--mesh-output-suffix", default="_meshbatch100_fixedtop")
    parser.add_argument("--view-elev", type=float, default=7.0)
    parser.add_argument("--view-azim", type=float, default=45.0)
    parser.add_argument(
        "--mesh-trim-top-frac",
        type=float,
        default=0.0,
        help="Trim this fraction from the top of the normal mesh panel after cropping/zooming.",
    )
    parser.add_argument(
        "--mesh-fill-panel",
        action="store_true",
        help="Fill the mesh subplot area with the cropped mesh image to remove side gutters.",
    )
    parser.add_argument(
        "--mesh-square-panel",
        action="store_true",
        help="Force each mesh animation panel to use a 1:1 square display box.",
    )
    parser.add_argument(
        "--dual-mesh-xz",
        action="store_true",
        help="Stack the normal mesh animation with a second XZ side-view animation below it.",
    )
    parser.add_argument("--dual-mesh-top-ratio", type=float, default=1.0)
    parser.add_argument("--dual-mesh-bottom-ratio", type=float, default=1.0)
    parser.add_argument("--xz-mesh-output-suffix", default="_meshbatch100_fixedtop_xz")
    parser.add_argument(
        "--xz-mesh-zoom-factor",
        type=float,
        default=1.8,
        help="Additional image zoom multiplier for the lower XZ mesh panel.",
    )
    parser.add_argument("--xz-view-elev", type=float, default=0.0)
    parser.add_argument("--xz-view-azim", type=float, default=90.0)
    parser.add_argument(
        "--xz-mesh-trim-top-frac",
        type=float,
        default=0.58,
        help="Trim this fraction from the top of the XZ mesh panel after cropping/zooming.",
    )
    parser.add_argument("--approach-speed-um-s", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--skip-mesh-png-render",
        action="store_true",
        help="Use existing mesh PNG frames instead of regenerating them from .msh snapshots.",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    if not bool(args.skip_mesh_png_render):
        render_mesh_pngs(
            run_dir,
            output_suffix=str(args.mesh_output_suffix),
            approach_speed_mps=float(args.approach_speed_um_s) * 1.0e-6,
            dt_s=float(args.dt),
            view_elev=float(args.view_elev),
            view_azim=float(args.view_azim),
            view_label="oblique",
        )
        if bool(args.dual_mesh_xz):
            render_mesh_pngs(
                run_dir,
                output_suffix=str(args.xz_mesh_output_suffix),
                approach_speed_mps=float(args.approach_speed_um_s) * 1.0e-6,
                dt_s=float(args.dt),
                view_elev=float(args.xz_view_elev),
                view_azim=float(args.xz_view_azim),
                view_label="XZ side view",
            )
    output_name = str(args.output_name)
    font_scale = float(args.font_scale)
    mesh_zoom = float(args.mesh_zoom)
    optimized_output_name = None
    if bool(args.ppt_optimized):
        output_name, optimized_output_name = _ppt_output_names(output_name)
        font_scale *= 1.2
        mesh_zoom *= 1.3
    render_gif(
        run_dir,
        case_label=str(args.case_label),
        max_step=int(args.max_step),
        duration_ms=int(args.duration_ms),
        output_name=output_name,
        frame_dir_name=str(args.frame_dir_name),
        font_scale=font_scale,
        mesh_zoom=mesh_zoom,
        oblique_mesh_zoom_factor=float(args.oblique_mesh_zoom_factor),
        dual_mesh_xz=bool(args.dual_mesh_xz),
        mesh_output_suffix=str(args.mesh_output_suffix),
        xz_mesh_output_suffix=str(args.xz_mesh_output_suffix),
        mesh_trim_top_frac=float(args.mesh_trim_top_frac),
        xz_mesh_trim_top_frac=float(args.xz_mesh_trim_top_frac),
        mesh_fill_panel=bool(args.mesh_fill_panel),
        mesh_square_panel=bool(args.mesh_square_panel),
        mesh_top_panel_ratio=float(args.dual_mesh_top_ratio),
        mesh_bottom_panel_ratio=float(args.dual_mesh_bottom_ratio),
        xz_mesh_zoom_factor=float(args.xz_mesh_zoom_factor),
    )
    if optimized_output_name is not None:
        optimize_gif_for_ppt(run_dir / output_name, run_dir / optimized_output_name)


if __name__ == "__main__":
    main()
