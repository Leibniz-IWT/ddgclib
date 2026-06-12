from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from plot_case_chart_vs_t import CASE_DIRS


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
AXISYM_CHART_PATH = PACKAGE_ROOT / "reference_data" / "pitois_fig5_chart.py"
NEATPLOT_STYLE = PACKAGE_ROOT / "reference_data" / "neatplot-main" / "standard.mplstyle"
DEFAULT_CASE_DIR = ROOT / "case1_r1.54mm_compress_corrected_gorge_mesh100_5000"
DEFAULT_GIF_NAME = "case1_diagnostic_variables_vs_t.gif"
PITOIS_RADIUS_MM = 4.0
PITOIS_FIG5_GAP_RATE_MM_S = 5.0e-3

CASE_LABELS = {
    "case1": "Case #1 r1.54 mm compressible: raw ddgclib/FHeron run",
    "case2": "Case #2 r1.54 mm incompressible: raw ddgclib/FHeron run",
    "case3": "Case #3 force-calibrated mesh compressible",
    "case4": "Case #4 force-calibrated mesh incompressible",
    "case5": "Case #5 r1.54 mm compressible tet/Cauchy",
    "case6": "Case #6 r1.54 mm incompressible tet/Cauchy",
    "case7": "Case #7 r1.54 mm compressible hydrostatic top CL",
    "case8": "Case #8 r1.54 mm incompressible hydrostatic top CL",
    "case9": "Case #9 r1.54 mm compressible PR33 B^T p",
    "case10": "Case #10 r1.54 mm incompressible PR33 B^T p",
    "case11": "Case #11 r1.54 mm compressible PR33 pref-Heron",
    "case12": "Case #12 force-calibrated mesh PR33 B^T p",
}

MESH_FRAME_RE = re.compile(r"^mesh_iter(?P<step>\d+)(?:_meshbatch100_fixedtop|_meshbatch100)?\.png$")


def _load_axisym_chart_module():
    spec = importlib.util.spec_from_file_location("axisym_chart", AXISYM_CHART_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {AXISYM_CHART_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_time_series() -> tuple[np.ndarray, np.ndarray]:
    chart = _load_axisym_chart_module()
    exp_d_over_r = np.asarray(chart.EXP_D_OVER_R, dtype=float)
    exp_force_mn = np.asarray(chart.EXP_FORCE_MN, dtype=float)
    exp_t_s = (
        PITOIS_RADIUS_MM
        * (exp_d_over_r - float(exp_d_over_r[0]))
        / PITOIS_FIG5_GAP_RATE_MM_S
    )
    return exp_t_s, exp_force_mn


def _load_history(case_dir: Path, max_step: int) -> list[dict]:
    payload = json.loads((case_dir / "separation_history.json").read_text())
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{case_dir / 'separation_history.json'} has no non-empty history list")
    filtered = [row for row in history if int(row.get("step", 0)) <= int(max_step)]
    if not filtered:
        raise RuntimeError(f"No history rows through step {max_step}")
    return filtered


def _series(history: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray([float(row.get(key, default)) for row in history], dtype=float)


def _axial_abs_mn(history: list[dict], key: str) -> np.ndarray:
    return np.abs(_series(history, key)) * 1.0e3


def _prepare_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    target_volume = max(float(history[0].get("initial_geometry_target_volume_m3", 1.0)), 1.0e-30)
    dual_volume = _series(history, "dual_cell_sum_m3", target_volume)
    snapshot_volume = _series(history, "snapshot_msh_volume_m3", target_volume)
    return {
        "step": _series(history, "step"),
        "t": _series(history, "t"),
        "d_over_r": _series(history, "d_over_r"),
        "gap_um": _series(history, "gap") * 1.0e6,
        "force_mn": _axial_abs_mn(history, "fixed_force_axial"),
        "pressure_force_mn": _axial_abs_mn(history, "top_cap_traction_axial"),
        "neck_surface_mn": _axial_abs_mn(history, "top_gorge_surface_tension_axial"),
        "r_top_mm": _series(history, "top_contact_radius") * 1.0e3,
        "r_bottom_mm": _series(history, "bottom_contact_radius") * 1.0e3,
        "r_neck_mm": _series(history, "pressure_neck_fit_radius") * 1.0e3,
        "dual_vol_rel_pct": (dual_volume / target_volume - 1.0) * 100.0,
        "snapshot_vol_rel_pct": (snapshot_volume / target_volume - 1.0) * 100.0,
        "position_restore_after_pct": _series(history, "position_volume_constraint_rel_after") * 100.0,
        "pressure_scalar_pa": _series(history, "pressure_scalar"),
        "max_speed_um_s": _series(history, "max_free_speed") * 1.0e6,
    }


def _mesh_png_steps(case_dir: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for pattern in (
        "mesh_iter*_meshbatch100_fixedtop.png",
        "mesh_iter*_meshbatch100.png",
        "mesh_iter*.png",
    ):
        for path in sorted(case_dir.glob(pattern)):
            match = MESH_FRAME_RE.match(path.name)
            if not match:
                continue
            step = int(match.group("step"))
            frames.setdefault(step, path)
    return frames


def _mesh_frame_for_step(case_dir: Path, step: int) -> tuple[Path, int]:
    stem = f"mesh_iter{int(step):04d}"
    candidates = (
        case_dir / f"{stem}_meshbatch100_fixedtop.png",
        case_dir / f"{stem}_meshbatch100.png",
        case_dir / f"{stem}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate, int(step)
    saved_frames = _mesh_png_steps(case_dir)
    if not saved_frames:
        raise FileNotFoundError(f"No mesh PNG found in {case_dir}")
    saved_steps = np.asarray(sorted(saved_frames), dtype=int)
    nearest_step = int(saved_steps[int(np.argmin(np.abs(saved_steps - int(step))))])
    return saved_frames[nearest_step], nearest_step


def _center_zoom_image(arr: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return arr
    height, width = arr.shape[:2]
    crop_width = max(1, int(round(width / zoom)))
    crop_height = max(1, int(round(height / zoom)))
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    return arr[y0 : y0 + crop_height, x0 : x0 + crop_width]


def _read_cropped_image(path: Path, *, zoom: float = 1.0) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    content = np.any(arr < 248, axis=2)
    if not np.any(content):
        return _center_zoom_image(arr, zoom)
    rows, cols = np.nonzero(content)
    pad = 10
    y0 = max(int(rows.min()) - pad, 0)
    y1 = min(int(rows.max()) + pad + 1, arr.shape[0])
    x0 = max(int(cols.min()) - pad, 0)
    x1 = min(int(cols.max()) + pad + 1, arr.shape[1])
    return _center_zoom_image(arr[y0:y1, x0:x1], zoom)


def _positive_ylim(values: list[np.ndarray], pad: float = 0.08) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in values])
    finite = finite[finite >= 0.0]
    if finite.size == 0:
        return 0.0, 1.0
    high = max(float(np.max(finite)), 1.0e-12)
    return 0.0, high * (1.0 + pad)


def _span_ylim(values: list[np.ndarray], pad: float = 0.08) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in values])
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high == low:
        delta = max(abs(high), 1.0) * 0.05
        return low - delta, high + delta
    delta = (high - low) * pad
    return low - delta, high + delta


def _draw_marker(ax, t_now: float, y_now: float, color: str) -> None:
    ax.axvline(t_now, color="#555555", linewidth=0.8, alpha=0.55)
    ax.scatter([t_now], [y_now], s=20, color=color, zorder=6)


def _history_index_for_step(history: list[dict], step: int) -> int:
    steps = [int(row.get("step", 0)) for row in history]
    try:
        return steps.index(int(step))
    except ValueError:
        return int(np.argmin(np.abs(np.asarray(steps, dtype=int) - int(step))))


def _render_frame(
    *,
    case_dir: Path,
    case_label: str,
    arrays: dict[str, np.ndarray],
    history: list[dict],
    step: int,
    out_path: Path,
    font_scale: float = 1.0,
    mesh_zoom: float = 1.0,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.style.use(str(NEATPLOT_STYLE))
    plt.rcParams["figure.constrained_layout.use"] = False
    idx = _history_index_for_step(history, step)
    row = history[idx]
    t = arrays["t"]
    t_now = float(arrays["t"][idx])
    t_end = float(t[-1])
    exp_t_s, exp_force_mn = _experiment_time_series()
    exp_mask = exp_t_s <= t_end

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

    ax_mesh = fig.add_subplot(gs[:2, 0])
    mesh_path, mesh_step = _mesh_frame_for_step(case_dir, step)
    mesh_img = _read_cropped_image(mesh_path, zoom=mesh_zoom)
    ax_mesh.imshow(mesh_img)
    ax_mesh.set_axis_off()
    if mesh_step == int(step):
        mesh_title = f"Mesh snapshot, step {step:04d}"
    else:
        mesh_title = f"Mesh snapshot, history step {step:04d}; saved mesh {mesh_step:04d}"
    ax_mesh.set_title(mesh_title, fontsize=title_fs, pad=2)

    ax_text = fig.add_subplot(gs[2, 0])
    ax_text.set_axis_off()
    text = (
        f"{case_label}\n"
        f"t = {t_now:7.3f} s    step = {int(row.get('step', 0)):4d}\n"
        f"D/R = {float(row.get('d_over_r', 0.0)):.6f}    gap = {float(row.get('gap', 0.0))*1e6:.2f} um\n"
        f"|F| = {abs(float(row.get('fixed_force_axial', 0.0)))*1e3:.4f} mN    "
        f"pressure = {float(row.get('pressure_scalar', 0.0)):.3f} Pa\n"
        f"r_neck = {float(row.get('pressure_neck_fit_radius', 0.0))*1e3:.4f} mm    "
        f"r_top/r_bot = {float(row.get('top_contact_radius', 0.0))*1e3:.4f}/"
        f"{float(row.get('bottom_contact_radius', 0.0))*1e3:.4f} mm\n"
        f"volume rel = {(float(row.get('dual_cell_sum_m3', 0.0)) / max(float(row.get('initial_geometry_target_volume_m3', 1.0)), 1e-30) - 1.0)*100.0:+.3e} %\n"
        f"max free speed = {float(row.get('max_free_speed', 0.0))*1e6:.3f} um/s\n"
        f"force model: {row.get('force_model', 'unknown')}"
    )
    ax_text.text(0.0, 0.98, text, va="top", ha="left", family="monospace", fontsize=note_fs)

    ax_chart = fig.add_subplot(gs[0, 1])
    ax_chart.scatter(
        exp_t_s[exp_mask],
        exp_force_mn[exp_mask],
        s=16,
        color="#111111",
        label="Pitois exp",
        zorder=4,
    )
    ax_chart.plot(t, arrays["force_mn"], color="#d62828", linewidth=1.45, label="raw ddgclib")
    _draw_marker(ax_chart, t_now, float(arrays["force_mn"][idx]), "#d62828")
    ax_chart.set_xlim(0.0, t_end)
    ax_chart.set_ylim(*_positive_ylim([exp_force_mn[exp_mask], arrays["force_mn"]]))
    ax_chart.set_ylabel("|F| [mN]", fontsize=label_fs)
    ax_chart.set_title("chart_vs_t: Pitois Fig. 5 force comparison", fontsize=title_fs, pad=2)
    ax_chart.tick_params(labelsize=tick_fs)
    ax_chart.grid(True, alpha=0.28)
    ax_chart.legend(fontsize=legend_fs, loc="upper right")

    ax_geom = fig.add_subplot(gs[1, 1])
    ax_geom.plot(t, arrays["r_neck_mm"], color="#003049", linewidth=1.25, label="r_neck")
    ax_geom.plot(t, arrays["r_top_mm"], color="#f77f00", linewidth=1.0, label="r_top CL")
    ax_geom.plot(t, arrays["r_bottom_mm"], color="#6a4c93", linewidth=1.0, label="r_bottom CL")
    _draw_marker(ax_geom, t_now, float(arrays["r_neck_mm"][idx]), "#003049")
    ax_geom.set_xlim(0.0, t_end)
    ax_geom.set_ylim(*_span_ylim([arrays["r_neck_mm"], arrays["r_top_mm"], arrays["r_bottom_mm"]]))
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
    _draw_marker(ax_vol, t_now, float(arrays["dual_vol_rel_pct"][idx]), "#1d3557")
    ax_vol.set_xlim(0.0, t_end)
    ax_vol.set_ylim(
        *_span_ylim(
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
    ax_vol.set_title("Volume residuals with pressure scalar", fontsize=title_fs, pad=2)
    ax_vol.tick_params(labelsize=tick_fs)
    ax_vol.grid(True, alpha=0.28)
    ax_pressure = ax_vol.twinx()
    ax_pressure.plot(
        t,
        arrays["pressure_scalar_pa"],
        color="#bc4749",
        linewidth=1.15,
        label="pressure",
    )
    ax_pressure.scatter([t_now], [float(arrays["pressure_scalar_pa"][idx])], s=18, color="#bc4749", zorder=6)
    ax_pressure.set_ylim(*_span_ylim([arrays["pressure_scalar_pa"]]))
    ax_pressure.set_ylabel("Pa", fontsize=label_fs)
    ax_pressure.tick_params(labelsize=tick_fs)
    lines1, labels1 = ax_vol.get_legend_handles_labels()
    lines2, labels2 = ax_pressure.get_legend_handles_labels()
    ax_vol.legend(lines1 + lines2, labels1 + labels2, fontsize=legend_fs, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _frame_steps(case_dir: Path, max_step: int) -> list[int]:
    steps = sorted(step for step in _mesh_png_steps(case_dir) if step <= int(max_step))
    if len(steps) >= 3:
        if 0 not in steps:
            steps.insert(0, 0)
        if int(max_step) not in steps:
            steps.append(int(max_step))
        return sorted(set(steps))
    if not steps:
        print(f"[warn] no mesh PNGs found in {case_dir}; cannot render diagnostic GIF")
    return list(range(0, int(max_step) + 1, 100))


def render_case_gif(
    case_dir: Path,
    *,
    case_label: str,
    max_step: int,
    duration_ms: int,
    output_name: str,
    font_scale: float = 1.0,
    mesh_zoom: float = 1.0,
    frame_dir_name: str = "diagnostic_variables_vs_t_frames",
) -> Path:
    history = _load_history(case_dir, max_step=max_step)
    arrays = _prepare_arrays(history)
    frame_steps = _frame_steps(case_dir, max_step=max_step)
    frames_dir = case_dir / frame_dir_name
    frame_paths: list[Path] = []
    print(f"Rendering {case_label}: {len(frame_steps)} frames")
    for index, step in enumerate(frame_steps, start=1):
        frame_path = frames_dir / f"diagnostic_step{step:04d}.png"
        _render_frame(
            case_dir=case_dir,
            case_label=case_label,
            arrays=arrays,
            history=history,
            step=step,
            out_path=frame_path,
            font_scale=font_scale,
            mesh_zoom=mesh_zoom,
        )
        frame_paths.append(frame_path)
        if int(step) % 1000 == 0 or index == len(frame_steps):
            print(f"[{index}/{len(frame_steps)}] step {int(step):04d}: {frame_path.name}")

    gif_path = case_dir / output_name
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


def _selected_cases(tokens: list[str] | None) -> list[tuple[str, Path]]:
    selected = list(CASE_DIRS.items())
    if tokens:
        wanted = set(tokens)
        selected = [
            (case_key, case_dir_name)
            for case_key, case_dir_name in selected
            if case_key in wanted or case_dir_name in wanted
        ]
        if not selected:
            raise RuntimeError(f"No cases matched: {', '.join(tokens)}")
    return [(case_key, ROOT / case_dir_name) for case_key, case_dir_name in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a mesh + important-variable diagnostic GIF from a saved case output."
    )
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render diagnostic GIFs for all numbered case output folders.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Render selected case keys or folder names, e.g. case1 case12.",
    )
    parser.add_argument("--max-step", type=int, default=5000)
    parser.add_argument("--duration-ms", type=int, default=140)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--ppt-optimized",
        action="store_true",
        help="Render a presentation source GIF with 20 percent larger text; the cropper writes caseN_diagnostic_variables_vs_t_ppt_optimized.gif.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale diagnostic plot and annotation fonts.",
    )
    parser.add_argument(
        "--mesh-zoom",
        type=float,
        default=1.0,
        help="Center-zoom the mesh snapshot before placing it in the diagnostic frame.",
    )
    args = parser.parse_args()

    if args.all or args.only:
        for case_key, case_dir in _selected_cases(args.only):
            output_name = args.output_name or f"{case_key}_diagnostic_variables_vs_t.gif"
            frame_dir_name = "diagnostic_variables_vs_t_frames"
            font_scale = float(args.font_scale)
            mesh_zoom = float(args.mesh_zoom)
            if args.ppt_optimized:
                output_name = f"{case_key}_diagnostic_variables_vs_t_ppt_raw.gif"
                frame_dir_name = "diagnostic_variables_vs_t_ppt_frames"
                font_scale = 1.2 * font_scale
                mesh_zoom = 1.3 * mesh_zoom
            render_case_gif(
                case_dir,
                case_label=CASE_LABELS.get(case_key, f"{case_key}: {case_dir.name}"),
                max_step=args.max_step,
                duration_ms=args.duration_ms,
                output_name=output_name,
                font_scale=font_scale,
                mesh_zoom=mesh_zoom,
                frame_dir_name=frame_dir_name,
            )
        return

    case_dir = args.case_dir or DEFAULT_CASE_DIR
    output_name = args.output_name or DEFAULT_GIF_NAME
    frame_dir_name = "diagnostic_variables_vs_t_frames"
    font_scale = float(args.font_scale)
    mesh_zoom = float(args.mesh_zoom)
    if args.ppt_optimized:
        case_key = next((key for key, dirname in CASE_DIRS.items() if ROOT / dirname == case_dir), case_dir.name)
        output_name = f"{case_key}_diagnostic_variables_vs_t_ppt_raw.gif"
        frame_dir_name = "diagnostic_variables_vs_t_ppt_frames"
        font_scale = 1.2 * font_scale
        mesh_zoom = 1.3 * mesh_zoom
    render_case_gif(
        case_dir,
        case_label="Custom case diagnostic" if args.case_dir else CASE_LABELS["case1"],
        max_step=args.max_step,
        duration_ms=args.duration_ms,
        output_name=output_name,
        font_scale=font_scale,
        mesh_zoom=mesh_zoom,
        frame_dir_name=frame_dir_name,
    )


if __name__ == "__main__":
    main()
