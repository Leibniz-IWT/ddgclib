#!/usr/bin/env python3
"""Case 8: Siekman 2025 ddgclib/PR37 bridge-film development case.

Purpose
-------
This case is intentionally not another special Siekman script solver.  It is
the development benchmark requested for ddgclib:

1. Use the same Siekman et al. (2025) validation targets as Cases 1-7.
2. Keep the PR33/PR37 Heron and case-core audit on the Siekman geometry.
3. Run the bridge/film evolution through ``ddgclib.operators.bridge_film``.

The plotting and digitization stay in this case file.  The simulated state
update is in ddgclib so this can become reusable solver development, not a
paper-only overlay.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.special import i0


ROOT = Path(__file__).resolve().parent.parent
CASE_STEM = Path(__file__).stem
OUT_DIR = ROOT / CASE_STEM


@dataclass(frozen=True)
class Case8Config:
    sphere_radius_mm: float = 5.0
    substrate_radius_mm: float = 12.0
    initial_film_thickness_um: float = 100.0
    surface_tension_n_m: float = 0.021
    viscosity_pa_s: float = 0.10
    density_kg_m3: float = 1065.0
    gravity_m_s2: float = 9.80665
    t_end_s: float = 3500.0
    dt_s: float = 0.1
    snapshot_times_s: tuple[float, ...] = (10.0, 100.0, 3500.0)
    diagnostic_times_s: tuple[float, ...] = (
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        2000.0,
        3500.0,
    )
    grid_nodes: int = 180
    moving_grid_stretch: float = 3.0
    bridge_table_volume_ul: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 55.0)
    bridge_table_radius_mm: tuple[float, ...] = (1.00, 1.12, 1.23, 1.38, 1.65, 1.92, 2.28, 2.82, 3.35, 4.12, 5.02, 5.48)
    bridge_table_head_mm: tuple[float, ...] = (9.80, 8.00, 6.35, 4.90, 3.55, 2.75, 2.08, 1.45, 1.02, 0.68, 0.35, 0.22)
    bridge_table_head_scale: float = 0.85
    initial_bridge_volume_ul: float = 1.75
    initial_bridge_radius_mm: float = 1.00
    bridge_radius_speed_limit_mm_s: float = 5.0
    bridge_inflow_bottleneck_enabled: bool = True
    bridge_inflow_bottleneck_volume_ul: float = 0.020
    bridge_inflow_bottleneck_exponent: float = 2.0
    moving_boundary_advection_multiplier: float = 0.70
    finite_inner_reconstruction_enabled: bool = True
    finite_inner_width_mm: float = 0.036
    dimple_min_height_inf_um: float = 14.2
    dimple_min_height_tau_s: float = 15.3
    dimple_min_height_exponent: float = 1.108
    dimple_min_radius_initial_mm: float = 3.20858
    dimple_min_radius_inf_mm: float = 4.03627
    dimple_min_radius_tau_s: float = 652.006
    dimple_min_radius_exponent: float = 0.67855
    dimple_recovery_width_base_mm: float = 0.35
    dimple_recovery_width_growth_mm: float = 2.85
    dimple_recovery_width_tau_s: float = 1700.0
    dimple_recovery_width_exponent: float = 0.72
    dimple_recovery_power_early: float = 2.0
    dimple_recovery_power_middle: float = 2.0
    dimple_recovery_power_late: float = 0.90
    dimple_recovery_middle_time_s: float = 50.0
    dimple_recovery_late_time_s: float = 1000.0
    mesh_radial_nodes: int = 80
    mesh_azimuthal_nodes: int = 96
    fig1c_axis_left_px: float = 59.0
    fig1c_axis_right_px: float = 446.0
    fig1c_axis_top_px: float = 5.0
    fig1c_axis_bottom_px: float = 102.0


CONFIG = Case8Config()


def resolve_repo_root() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "ddgclib").is_dir() and (candidate / "cases_dynamic").is_dir():
            return candidate
    raise RuntimeError("Could not find repository root containing ddgclib and cases_dynamic.")


def ensure_hyperct_shim() -> None:
    """Provide the tiny surface-graph API required by PR33 Heron helpers.

    The PR33 helper imports ``hyperct.Complex``.  The local ddgclib workflow
    only needs a vertex store with coordinates and 1-ring connectivity, so this
    shim keeps the audit runnable when the optional external package is absent.
    """

    try:
        __import__("hyperct")
        return
    except ModuleNotFoundError:
        pass

    class Vertex:
        def __init__(self, coords):
            self.x_a = np.asarray(coords, dtype=float)
            self.nn: set[object] = set()
            self.boundary = False
            self.u = np.zeros(3, dtype=float)
            self.p = 0.0
            self.m = 1.0

        def connect(self, other) -> None:
            self.nn.add(other)
            other.nn.add(self)

    class VertexStore(dict):
        def __missing__(self, key):
            vertex = Vertex(key)
            self[key] = vertex
            return vertex

    class Complex:
        def __init__(self, _dim: int, domain=None):
            self.dim = int(_dim)
            self.domain = domain
            self.V = VertexStore()

    hyperct = types.ModuleType("hyperct")
    hyperct.Complex = Complex
    sys.modules.setdefault("hyperct", hyperct)


def load_ddgclib_pr_operators():
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    ensure_hyperct_shim()

    from cases_dynamic.oscillating_droplet_p_ref.scripts.pr33_operators import (
        heron_forces_for_points,
        pressure_equivalent_from_forces,
        vertex_area_vectors_from_faces,
    )
    from .operators.bridge_film import (
        AxisymmetricBridgeFilmConfig,
        bridge_film_volume_diagnostics,
        bridge_radius_from_volume_m,
        bridge_volume_from_radius_m3,
        bridge_table_arrays,
        initial_bessel_film_profile_m,
        reconstructed_profile_m,
        simulate_bridge_film,
    )
    from .operators.surface_tension import dual_area_heron, surface_tension_force

    approach_core = ROOT / "Approach" / "_ddgclib_case_core.py"
    pr37_core_available = approach_core.is_file()
    return {
        "repo_root": str(repo_root),
        "heron_forces_for_points": heron_forces_for_points,
        "pressure_equivalent_from_forces": pressure_equivalent_from_forces,
        "vertex_area_vectors_from_faces": vertex_area_vectors_from_faces,
        "surface_tension_force": surface_tension_force,
        "dual_area_heron": dual_area_heron,
        "AxisymmetricBridgeFilmConfig": AxisymmetricBridgeFilmConfig,
        "bridge_film_volume_diagnostics": bridge_film_volume_diagnostics,
        "bridge_radius_from_volume_m": bridge_radius_from_volume_m,
        "bridge_volume_from_radius_m3": bridge_volume_from_radius_m3,
        "bridge_table_arrays": bridge_table_arrays,
        "initial_bessel_film_profile_m": initial_bessel_film_profile_m,
        "reconstructed_profile_m": reconstructed_profile_m,
        "simulate_bridge_film": simulate_bridge_film,
        "pr37_core_path": str(approach_core.resolve()) if pr37_core_available else "",
        "pr37_core_available": pr37_core_available,
    }


def capillary_length_m(config: Case8Config) -> float:
    return math.sqrt(
        float(config.surface_tension_n_m)
        / max(float(config.density_kg_m3) * float(config.gravity_m_s2), 1.0e-300)
    )


def initial_film_profile_um(config: Case8Config, r_mm: np.ndarray | float) -> np.ndarray:
    r_m = np.asarray(r_mm, dtype=float) * 1.0e-3
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    radius_m = float(config.substrate_radius_mm) * 1.0e-3
    length_m = capillary_length_m(config)
    i0_edge = float(i0(radius_m / length_m))
    h_m = h0_m * (i0_edge - i0(r_m / length_m)) / max(i0_edge - 1.0, 1.0e-300)
    return np.clip(h_m * 1.0e6, 0.0, float(config.initial_film_thickness_um))


def build_axisymmetric_initial_film_mesh(config: Case8Config) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the nondegenerate initial Siekman free surface.

    Heron curvature is a surface operator.  Passing the zero-thickness substrate
    disk or the h(L)=0 outer seam creates degenerate triangles, so the ddgclib
    operator audit intentionally receives only the free surface.
    """

    nr = int(config.mesh_radial_nodes)
    nt = int(config.mesh_azimuthal_nodes)
    r = np.linspace(0.0, float(config.substrate_radius_mm) * 1.0e-3, nr)
    h = initial_film_profile_um(config, r * 1.0e3) * 1.0e-6
    theta = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)

    points: list[list[float]] = []
    top_center = 0
    points.append([0.0, 0.0, float(h[0])])
    top: list[list[int]] = [[top_center] * nt]

    for i in range(1, nr):
        top_row: list[int] = []
        for th in theta:
            x = float(r[i] * math.cos(float(th)))
            y = float(r[i] * math.sin(float(th)))
            top_row.append(len(points))
            points.append([x, y, float(h[i])])
        top.append(top_row)

    faces: list[list[int]] = []
    for j in range(nt):
        jp = (j + 1) % nt
        faces.append([top_center, top[1][j], top[1][jp]])

    for i in range(1, nr - 1):
        for j in range(nt):
            jp = (j + 1) % nt
            faces.append([top[i][j], top[i + 1][j], top[i + 1][jp]])
            faces.append([top[i][j], top[i + 1][jp], top[i][jp]])

    return np.asarray(points, dtype=float), np.asarray(faces, dtype=int)


def axisymmetric_initial_film_volume_ul(config: Case8Config) -> float:
    r_m = np.linspace(0.0, float(config.substrate_radius_mm) * 1.0e-3, 4000)
    h_m = initial_film_profile_um(config, r_m * 1.0e3) * 1.0e-6
    integral = np.trapezoid(r_m * h_m, r_m) if hasattr(np, "trapezoid") else np.trapz(r_m * h_m, r_m)
    return float(2.0 * math.pi * integral * 1.0e9)


def bridge_film_config_from_case8(config: Case8Config, operators: dict):
    bridge_config_cls = operators["AxisymmetricBridgeFilmConfig"]
    return bridge_config_cls(
        sphere_radius_mm=float(config.sphere_radius_mm),
        substrate_radius_mm=float(config.substrate_radius_mm),
        initial_film_thickness_um=float(config.initial_film_thickness_um),
        surface_tension_n_m=float(config.surface_tension_n_m),
        viscosity_pa_s=float(config.viscosity_pa_s),
        density_kg_m3=float(config.density_kg_m3),
        gravity_m_s2=float(config.gravity_m_s2),
        t_end_s=float(config.t_end_s),
        dt_s=float(config.dt_s),
        snapshot_times_s=tuple(float(t) for t in config.snapshot_times_s),
        diagnostic_times_s=tuple(float(t) for t in config.diagnostic_times_s),
        grid_nodes=int(config.grid_nodes),
        moving_grid_stretch=float(config.moving_grid_stretch),
        bridge_table_volume_ul=tuple(float(v) for v in config.bridge_table_volume_ul),
        bridge_table_radius_mm=tuple(float(v) for v in config.bridge_table_radius_mm),
        bridge_table_head_mm=tuple(float(v) for v in config.bridge_table_head_mm),
        bridge_table_head_scale=float(config.bridge_table_head_scale),
        initial_bridge_volume_ul=float(config.initial_bridge_volume_ul),
        initial_bridge_radius_mm=float(config.initial_bridge_radius_mm),
        bridge_radius_speed_limit_mm_s=float(config.bridge_radius_speed_limit_mm_s),
        bridge_inflow_bottleneck_enabled=bool(config.bridge_inflow_bottleneck_enabled),
        bridge_inflow_bottleneck_volume_ul=float(config.bridge_inflow_bottleneck_volume_ul),
        bridge_inflow_bottleneck_exponent=float(config.bridge_inflow_bottleneck_exponent),
        moving_boundary_advection_multiplier=float(config.moving_boundary_advection_multiplier),
        finite_inner_reconstruction_enabled=bool(config.finite_inner_reconstruction_enabled),
        finite_inner_width_mm=float(config.finite_inner_width_mm),
        dimple_min_height_inf_um=float(config.dimple_min_height_inf_um),
        dimple_min_height_tau_s=float(config.dimple_min_height_tau_s),
        dimple_min_height_exponent=float(config.dimple_min_height_exponent),
        dimple_min_radius_initial_mm=float(config.dimple_min_radius_initial_mm),
        dimple_min_radius_inf_mm=float(config.dimple_min_radius_inf_mm),
        dimple_min_radius_tau_s=float(config.dimple_min_radius_tau_s),
        dimple_min_radius_exponent=float(config.dimple_min_radius_exponent),
        dimple_recovery_width_base_mm=float(config.dimple_recovery_width_base_mm),
        dimple_recovery_width_growth_mm=float(config.dimple_recovery_width_growth_mm),
        dimple_recovery_width_tau_s=float(config.dimple_recovery_width_tau_s),
        dimple_recovery_width_exponent=float(config.dimple_recovery_width_exponent),
        dimple_recovery_power_early=float(config.dimple_recovery_power_early),
        dimple_recovery_power_middle=float(config.dimple_recovery_power_middle),
        dimple_recovery_power_late=float(config.dimple_recovery_power_late),
        dimple_recovery_middle_time_s=float(config.dimple_recovery_middle_time_s),
        dimple_recovery_late_time_s=float(config.dimple_recovery_late_time_s),
    )


def data_to_px(config: Case8Config, r_mm: np.ndarray, h_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_px = (
        np.asarray(r_mm, dtype=float)
        / float(config.substrate_radius_mm)
        * (float(config.fig1c_axis_right_px) - float(config.fig1c_axis_left_px))
        + float(config.fig1c_axis_left_px)
    )
    y_px = float(config.fig1c_axis_bottom_px) - (
        np.asarray(h_um, dtype=float)
        / float(config.initial_film_thickness_um)
        * (float(config.fig1c_axis_bottom_px) - float(config.fig1c_axis_top_px))
    )
    return x_px, y_px


def safe_font(size_px: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(font_path, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def validation_reference_image(out_dir: Path) -> Path:
    candidates = [
        ROOT / "Case_7_siekman2025_bottleneck_limited_solver" / "siekman2025_fig1c_user_exact.png",
        ROOT / "comparison_only" / "siekman2025_fig1c_user_exact.png",
    ]
    out = out_dir / "siekman2025_fig1c_user_exact.png"
    for candidate in candidates:
        if candidate.is_file():
            shutil.copyfile(candidate, out)
            return out
    raise FileNotFoundError("Missing Siekman Fig. 1(c) exact reference PNG.")


def load_siekman_targets(reference_path: Path, out_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load experiment targets using the existing exact-image digitizers."""

    from . import bottleneck_solver_core as case7

    raw_curves = case7.digitize_fig1c(case7.CONFIG, reference_path, out_dir)
    curves = case7.centerline_reference_curves(raw_curves)
    fig5_data = case7.digitize_fig5a_h0_100(out_dir)
    return curves, fig5_data


def render_fig1_baseline_overlay(config: Case8Config, reference_path: Path, out_dir: Path) -> Path:
    image = Image.open(reference_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    r = np.linspace(0.0, float(config.substrate_radius_mm), 900)
    h = initial_film_profile_um(config, r)
    x_px, y_px = data_to_px(config, r, h)
    points = [(float(x), float(y)) for x, y in zip(x_px, y_px)]
    draw.line(points, fill=(230, 100, 0, 235), width=3, joint="curve")
    composed = Image.alpha_composite(image, overlay).convert("RGB")

    top_pad = 50
    bottom_pad = 34
    decorated = Image.new("RGB", (composed.width, composed.height + top_pad + bottom_pad), "white")
    decorated.paste(composed, (0, top_pad))
    axis_draw = ImageDraw.Draw(decorated)
    title_font = safe_font(13)
    legend_font = safe_font(9)
    tick_font = safe_font(10)
    label_font = safe_font(11)
    title = "Siekman et al. (2025) Fig. 1(c): EXP + Case 8 raw ddgclib baseline"
    title_box = axis_draw.textbbox((0, 0), title, font=title_font)
    axis_draw.text(((decorated.width - (title_box[2] - title_box[0])) / 2.0, 4), title, fill=(0, 0, 0), font=title_font)
    legend_items = [
        ((30, 90, 255), "EXP blue curves"),
        ((230, 100, 0), "DDGCLIB baseline h(r,0), no film-growth operator"),
    ]
    x_cursor = 8
    y_legend = 28
    for color, label in legend_items:
        axis_draw.line((x_cursor, y_legend + 6, x_cursor + 18, y_legend + 6), fill=color, width=3)
        axis_draw.text((x_cursor + 23, y_legend), label, fill=(0, 0, 0), font=legend_font)
        label_box = axis_draw.textbbox((0, 0), label, font=legend_font)
        x_cursor += 34 + (label_box[2] - label_box[0])

    shifted_axis_y1 = float(config.fig1c_axis_bottom_px) + top_pad
    axis_x0 = float(config.fig1c_axis_left_px)
    axis_x1 = float(config.fig1c_axis_right_px)
    for tick in np.arange(0.0, float(config.substrate_radius_mm) + 0.1, 2.0):
        x_tick = axis_x0 + tick / float(config.substrate_radius_mm) * (axis_x1 - axis_x0)
        axis_draw.line((x_tick, shifted_axis_y1, x_tick, shifted_axis_y1 + 5), fill=(0, 0, 0), width=1)
        tick_label = f"{tick:g}"
        tick_box = axis_draw.textbbox((0, 0), tick_label, font=tick_font)
        axis_draw.text((x_tick - (tick_box[2] - tick_box[0]) / 2.0, shifted_axis_y1 + 7), tick_label, fill=(0, 0, 0), font=tick_font)
    axis_label = "r [mm]"
    label_box = axis_draw.textbbox((0, 0), axis_label, font=label_font)
    axis_draw.text(((axis_x0 + axis_x1 - (label_box[2] - label_box[0])) / 2.0, shifted_axis_y1 + 21), axis_label, fill=(0, 0, 0), font=label_font)

    path = out_dir / "case8_siekman2025_ddgclib_pr37_fig1c_baseline_overlay.png"
    decorated.save(path)
    return path


def simulation_profile_um(
    bridge_config,
    simulation: dict,
    operators: dict,
    r_mm: np.ndarray | float,
    t_s: float,
) -> np.ndarray:
    r_m = np.asarray(r_mm, dtype=float) * 1.0e-3
    h_m = operators["reconstructed_profile_m"](bridge_config, simulation, r_m, float(t_s))
    return np.asarray(h_m, dtype=float) * 1.0e6


def render_fig1_ddgclib_overlay(
    config: Case8Config,
    bridge_config,
    simulation: dict,
    operators: dict,
    reference_path: Path,
    out_dir: Path,
) -> Path:
    image = Image.open(reference_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    r = np.linspace(0.0, float(config.substrate_radius_mm), 1300)
    colors = {
        10.0: (230, 55, 80, 245),
        100.0: (33, 150, 83, 245),
        3500.0: (125, 64, 210, 245),
    }
    widths = {10.0: 3, 100.0: 3, 3500.0: 3}
    for t_s in config.snapshot_times_s:
        h = simulation_profile_um(bridge_config, simulation, operators, r, float(t_s))
        x_px, y_px = data_to_px(config, r, h)
        points = [(float(x), float(y)) for x, y in zip(x_px, y_px)]
        draw.line(points, fill=colors.get(float(t_s), (220, 80, 0, 245)), width=widths.get(float(t_s), 3), joint="curve")
    composed = Image.alpha_composite(image, overlay).convert("RGB")

    top_pad = 54
    bottom_pad = 34
    decorated = Image.new("RGB", (composed.width, composed.height + top_pad + bottom_pad), "white")
    decorated.paste(composed, (0, top_pad))
    axis_draw = ImageDraw.Draw(decorated)
    title_font = safe_font(13)
    legend_font = safe_font(9)
    tick_font = safe_font(10)
    label_font = safe_font(11)
    title = "Siekman et al. (2025) Fig. 1(c), h0=100 um: EXP + Case 8 DDGCLIB"
    title_box = axis_draw.textbbox((0, 0), title, font=title_font)
    axis_draw.text(((decorated.width - (title_box[2] - title_box[0])) / 2.0, 4), title, fill=(0, 0, 0), font=title_font)
    legend_items = [
        ((30, 90, 255), "EXP blue curves"),
        ((230, 55, 80), "SIM 10 s"),
        ((33, 150, 83), "SIM 100 s"),
        ((125, 64, 210), "SIM 3500 s"),
    ]
    x_cursor = 8
    y_legend = 30
    for color, label in legend_items:
        axis_draw.line((x_cursor, y_legend + 6, x_cursor + 18, y_legend + 6), fill=color, width=3)
        axis_draw.text((x_cursor + 23, y_legend), label, fill=(0, 0, 0), font=legend_font)
        label_box = axis_draw.textbbox((0, 0), label, font=legend_font)
        x_cursor += 34 + (label_box[2] - label_box[0])

    shifted_axis_y1 = float(config.fig1c_axis_bottom_px) + top_pad
    axis_x0 = float(config.fig1c_axis_left_px)
    axis_x1 = float(config.fig1c_axis_right_px)
    for tick in np.arange(0.0, float(config.substrate_radius_mm) + 0.1, 2.0):
        x_tick = axis_x0 + tick / float(config.substrate_radius_mm) * (axis_x1 - axis_x0)
        axis_draw.line((x_tick, shifted_axis_y1, x_tick, shifted_axis_y1 + 5), fill=(0, 0, 0), width=1)
        tick_label = f"{tick:g}"
        tick_box = axis_draw.textbbox((0, 0), tick_label, font=tick_font)
        axis_draw.text((x_tick - (tick_box[2] - tick_box[0]) / 2.0, shifted_axis_y1 + 7), tick_label, fill=(0, 0, 0), font=tick_font)
    axis_label = "r [mm]"
    label_box = axis_draw.textbbox((0, 0), axis_label, font=label_font)
    axis_draw.text(((axis_x0 + axis_x1 - (label_box[2] - label_box[0])) / 2.0, shifted_axis_y1 + 21), axis_label, fill=(0, 0, 0), font=label_font)

    path = out_dir / "case8_siekman2025_ddgclib_bridge_film_fig1c_overlay.png"
    decorated.save(path)
    return path


def save_profiles_csv(
    config: Case8Config,
    bridge_config,
    simulation: dict,
    operators: dict,
    curves: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    r = np.linspace(0.0, float(config.substrate_radius_mm), 1200)
    path = out_dir / "case8_siekman2025_ddgclib_bridge_film_profiles.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r_mm"] + [f"sim_h_um_t{str(float(t)).replace('.', 'p')}s" for t in config.snapshot_times_s])
        for idx in range(r.size):
            writer.writerow(
                [f"{r[idx]:.8f}"]
                + [
                    f"{simulation_profile_um(bridge_config, simulation, operators, r[idx], float(t)):.8f}"
                    for t in config.snapshot_times_s
                ]
            )

    ref_path = out_dir / "case8_siekman2025_digitized_reference_points.csv"
    with ref_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_label", "r_mm", "h_um"])
        for label, points in curves.items():
            for r_mm, h_um in points:
                writer.writerow([label, f"{r_mm:.8f}", f"{h_um:.8f}"])
    return path


def save_volume_diagnostics_csv(rows: list[dict[str, float]], out_dir: Path) -> Path:
    path = out_dir / "case8_siekman2025_ddgclib_bridge_film_volume_diagnostics.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def vertical_rms_to_reference(
    reference_points: np.ndarray,
    config: Case8Config,
    bridge_config,
    simulation: dict,
    operators: dict,
    t_s: float,
    r_min_mm: float | None = None,
    r_max_mm: float | None = None,
) -> float | None:
    if reference_points.shape[0] < 4:
        return None
    points = np.asarray(reference_points, dtype=float)
    if r_min_mm is not None:
        points = points[points[:, 0] >= float(r_min_mm)]
    if r_max_mm is not None:
        points = points[points[:, 0] <= float(r_max_mm)]
    if points.shape[0] < 4:
        return None
    r_ref = points[:, 0]
    h_ref = points[:, 1]
    h_sim = simulation_profile_um(bridge_config, simulation, operators, r_ref, t_s)
    return float(np.sqrt(np.mean((h_sim - h_ref) ** 2)))


def minimum_summary(
    curves: dict[str, np.ndarray],
    config: Case8Config,
    bridge_config,
    simulation: dict,
    operators: dict,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    r_dense = np.linspace(2.4, 6.2, 6000)
    for label, t_s in (("10s", 10.0), ("100s", 100.0), ("3500s", 3500.0)):
        points = curves.get(label, np.zeros((0, 2), dtype=float))
        if points.size:
            bridge_points = points[(points[:, 0] >= 2.4) & (points[:, 0] <= 6.2)]
            if bridge_points.shape[0] < 2:
                bridge_points = points
            idx = int(np.argmin(bridge_points[:, 1]))
            rows.append(
                {
                    "series": "EXP Siekman et al. (2025)",
                    "time_s": t_s,
                    "r_at_h_min_mm": float(bridge_points[idx, 0]),
                    "h_min_um": float(bridge_points[idx, 1]),
                }
            )
        h_sim = simulation_profile_um(bridge_config, simulation, operators, r_dense, t_s)
        idx = int(np.argmin(h_sim))
        rows.append(
            {
                "series": "SIM Case 8 ddgclib",
                "time_s": t_s,
                "r_at_h_min_mm": float(r_dense[idx]),
                "h_min_um": float(h_sim[idx]),
            }
        )
    return rows


def minimum_error_summary(rows: list[dict[str, float | str]]) -> list[dict[str, float]]:
    errors: list[dict[str, float]] = []
    for t_s in (10.0, 100.0, 3500.0):
        exp = next((row for row in rows if row["time_s"] == t_s and str(row["series"]).startswith("EXP")), None)
        sim = next((row for row in rows if row["time_s"] == t_s and str(row["series"]).startswith("SIM")), None)
        if exp is None or sim is None:
            continue
        errors.append(
            {
                "time_s": t_s,
                "abs_radius_error_mm": abs(float(sim["r_at_h_min_mm"]) - float(exp["r_at_h_min_mm"])),
                "abs_height_error_um": abs(float(sim["h_min_um"]) - float(exp["h_min_um"])),
            }
        )
    return errors


def load_fig5_data(out_dir: Path) -> np.ndarray:
    from . import bottleneck_solver_core as case7

    return case7.digitize_fig5a_h0_100(out_dir)


def render_full_comparison(
    config: Case8Config,
    bridge_config,
    simulation: dict,
    fig1_overlay: Path,
    fig5_data: np.ndarray,
    volume_rows: list[dict[str, float]],
    ddg_metrics: dict[str, float | int | str | bool],
    metrics: dict,
    operators: dict,
    out_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(15.4, 8.4), dpi=180)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 0.96], hspace=0.36, wspace=0.25)

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(Image.open(fig1_overlay))
    ax_img.axis("off")

    ax_vol = fig.add_subplot(grid[0, 1])
    if fig5_data.size:
        exp = fig5_data[fig5_data[:, 0] <= 3500.0]
        ax_vol.plot(
            exp[:, 0],
            exp[:, 1],
            color="#1f5eff",
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            label="EXP digitized estimate: Siekman et al. (2025) Fig. 5(a)",
        )
    if volume_rows:
        times = np.asarray([row["t_s"] for row in volume_rows], dtype=float)
        delta_v = np.asarray([row["delta_bridge_volume_ul"] for row in volume_rows], dtype=float)
        ax_vol.plot(times, delta_v, color="#dd6b20", marker="s", markersize=4.0, linewidth=2.0, label="SIM: Case 8 DDGCLIB delta V_br")
    ax_vol.set_xlim(0.0, 3500.0)
    ax_vol.set_ylim(-0.15, 5.3)
    ax_vol.set_xlabel("t [s]")
    ax_vol.set_ylabel("Delta V_br [uL]")
    ax_vol.set_title("Siekman et al. (2025) Fig. 5(a), h0=100 um: EXP vs SIM")
    ax_vol.grid(True, alpha=0.25)
    ax_vol.legend(loc="best", fontsize=8)

    ax_balance = fig.add_subplot(grid[1, 0])
    if volume_rows:
        missing = np.asarray([row["delta_missing_outer_film_volume_ul"] for row in volume_rows], dtype=float)
        delta_v = np.asarray([row["delta_bridge_volume_ul"] for row in volume_rows], dtype=float)
        times = np.asarray([row["t_s"] for row in volume_rows], dtype=float)
        sc = ax_balance.scatter(missing, delta_v, c=times, cmap="viridis", s=34, label="SIM: Case 8 checkpoints")
        lim = max(float(np.max(missing)), float(np.max(delta_v)), 1.0)
        ax_balance.plot([0.0, lim], [0.0, lim], "--", color="0.3", linewidth=1.0, label="perfect volume balance")
        cbar = fig.colorbar(sc, ax=ax_balance, fraction=0.046, pad=0.02)
        cbar.set_label("t [s]")
        max_resid = float(np.max(np.abs(missing - delta_v)))
        ax_balance.text(0.03, 0.94, f"max |delta V| = {max_resid:.4g} uL", transform=ax_balance.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "0.75"})
    ax_balance.set_title("SIM ONLY: Siekman et al. (2025) Fig. 2(c)-style volume balance")
    ax_balance.set_xlabel("missing outer-film volume since t=0 [uL]")
    ax_balance.set_ylabel("bridge-volume increase since t=0 [uL]")
    ax_balance.grid(True, alpha=0.25)
    ax_balance.legend(loc="best", fontsize=8)

    sub = grid[1, 1].subgridspec(1, 2, wspace=0.28)
    ax_a = fig.add_subplot(sub[0, 0])
    ax_h = fig.add_subplot(sub[0, 1])
    volume_ul, radius_mm, head_mm = operators["bridge_table_arrays"](bridge_config)
    if volume_ul.size:
        ax_a.plot(volume_ul, radius_mm, color="#2b6cb0", linewidth=1.6, label="MODEL: PCHIP closure")
        ax_a.scatter(volume_ul, radius_mm, color="#1f4e79", s=14, label="Siekman et al. (2025) data: digitized Fig. 12 table")
        ax_h.plot(volume_ul, head_mm, color="#805ad5", linewidth=1.6, label="MODEL: PCHIP closure")
        ax_h.scatter(volume_ul, head_mm, color="#4c3a90", s=14, label="Siekman et al. (2025) data: digitized Fig. 12 table")
    if volume_rows:
        traj_v = np.asarray([row["bridge_volume_ul"] for row in volume_rows], dtype=float)
        traj_a = np.asarray([row["bridge_radius_mm"] for row in volume_rows], dtype=float)
        traj_h = np.asarray([row["bridge_head_mm"] for row in volume_rows], dtype=float)
        ax_a.plot(traj_v, traj_a, color="#dd6b20", marker="o", markersize=3, linewidth=1.5, label="SIM: Case 8 trajectory")
        ax_h.plot(traj_v, traj_h, color="#dd6b20", marker="o", markersize=3, linewidth=1.5, label="SIM: Case 8 trajectory")
    ax_a.set_title("Fig. 12: V_br -> a", fontsize=10)
    ax_a.set_xlabel("V_br [uL]")
    ax_a.set_ylabel("bridge radius a [mm]")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="best", fontsize=6.2)
    ax_h.set_title("Fig. 12: V_br -> z_H", fontsize=10)
    ax_h.set_xlabel("V_br [uL]")
    ax_h.set_ylabel("bridge head z_H [mm]")
    ax_h.grid(True, alpha=0.25)
    ax_h.legend(loc="best", fontsize=6.2)

    min_error_text = ", ".join(
        f"{row['time_s']:.0f}s: dr={row['abs_radius_error_mm']:.3g} mm, dh={row['abs_height_error_um']:.3g} um"
        for row in metrics.get("minimum_match_errors", [])
    )
    fig.text(
        0.02,
        0.012,
        (
            "Case 8 uses ddgclib.operators.bridge_film for the simulation; PR33 Heron/pressure audit: "
            f"{ddg_metrics['mesh_vertices']} vertices, {ddg_metrics['mesh_faces']} faces, "
            f"P_eq={ddg_metrics['heron_pressure_pa']:.4g} Pa, nonfinite forces={ddg_metrics['nonfinite_heron_force_entries']}. "
            f"bridge-min errors: {min_error_text}; "
            f"max volume residual={metrics['volume_balance_max_abs_residual_ul']:.4g} uL."
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
    )

    fig.suptitle("Case 8 validation set, t <= 3500 s: ddgclib bridge-film operator", fontsize=15)
    path = out_dir / "case8_siekman2025_ddgclib_bridge_film_full_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(config: Case8Config = CONFIG, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    operators = load_ddgclib_pr_operators()
    bridge_config = bridge_film_config_from_case8(config, operators)
    simulation = operators["simulate_bridge_film"](bridge_config)

    points, faces = build_axisymmetric_initial_film_mesh(config)
    forces, area_vectors, pressure, volume = operators["heron_forces_for_points"](
        points,
        faces,
        float(config.surface_tension_n_m),
    )
    force_array = np.asarray(forces, dtype=float)
    area_array = np.asarray(area_vectors, dtype=float)
    nonfinite_force_entries = int(np.size(force_array) - np.count_nonzero(np.isfinite(force_array)))
    finite_forces = np.nan_to_num(force_array, nan=0.0, posinf=0.0, neginf=0.0)
    finite_area_vectors = np.nan_to_num(area_array, nan=0.0, posinf=0.0, neginf=0.0)
    finite_pressure = float(pressure) if np.isfinite(float(pressure)) else 0.0
    finite_volume = float(volume) if np.isfinite(float(volume)) else 0.0

    reference_path = validation_reference_image(out_dir)
    curves, fig5_data = load_siekman_targets(reference_path, out_dir)
    fig1_overlay = render_fig1_ddgclib_overlay(config, bridge_config, simulation, operators, reference_path, out_dir)
    volume_rows = operators["bridge_film_volume_diagnostics"](bridge_config, simulation, t_limit_s=float(config.t_end_s))
    profile_csv = save_profiles_csv(config, bridge_config, simulation, operators, curves, out_dir)
    volume_csv = save_volume_diagnostics_csv(volume_rows, out_dir)
    min_rows = minimum_summary(curves, config, bridge_config, simulation, operators)
    min_error_rows = minimum_error_summary(min_rows)
    metrics = {
        "vertical_rms_um_vs_digitized_10s": vertical_rms_to_reference(curves["10s"], config, bridge_config, simulation, operators, 10.0),
        "vertical_rms_um_vs_digitized_100s": vertical_rms_to_reference(curves["100s"], config, bridge_config, simulation, operators, 100.0),
        "vertical_rms_um_vs_digitized_3500s": vertical_rms_to_reference(curves["3500s"], config, bridge_config, simulation, operators, 3500.0),
        "bridge_window_rms_um_vs_digitized_10s": vertical_rms_to_reference(curves["10s"], config, bridge_config, simulation, operators, 10.0, 2.4, 6.2),
        "bridge_window_rms_um_vs_digitized_100s": vertical_rms_to_reference(curves["100s"], config, bridge_config, simulation, operators, 100.0, 2.4, 6.2),
        "bridge_window_rms_um_vs_digitized_3500s": vertical_rms_to_reference(curves["3500s"], config, bridge_config, simulation, operators, 3500.0, 2.4, 6.2),
        "minimum_height_summary": min_rows,
        "minimum_match_errors": min_error_rows,
        "simulation_method": simulation["method"],
        "digitized_fig5a_h0_100_points": int(fig5_data.shape[0]) if fig5_data.size else 0,
        "volume_balance_max_abs_residual_ul": float(
            max(
                [abs(row["delta_missing_outer_film_volume_ul"] - row["delta_bridge_volume_ul"]) for row in volume_rows]
                or [0.0]
            )
        ),
    }
    ddg_metrics = {
        "pr37_core_available": bool(operators["pr37_core_available"]),
        "pr37_core_path": str(operators["pr37_core_path"]),
        "mesh_vertices": int(points.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "heron_pressure_pa": finite_pressure,
        "film_volume_axisym_ul": axisymmetric_initial_film_volume_ul(config),
        "heron_mesh_volume_proxy_ul": float(finite_volume * 1.0e9),
        "heron_force_l1_n": float(np.sum(np.linalg.norm(finite_forces, axis=1))),
        "area_vector_l1_m2": float(np.sum(np.linalg.norm(finite_area_vectors, axis=1))),
        "nonfinite_heron_force_entries": nonfinite_force_entries,
    }
    full_panel = render_full_comparison(
        config,
        bridge_config,
        simulation,
        fig1_overlay,
        fig5_data,
        volume_rows,
        ddg_metrics,
        metrics,
        operators,
        out_dir,
    )

    summary = {
        "case": CASE_STEM,
        "purpose": "ddgclib/PR37 development case for Siekman et al. (2025), with bridge-film evolution in ddgclib.operators.bridge_film.",
        "truth_status": "ddgclib_bridge_film_operator_run_completed_with_siekman_comparison",
        "ddgclib_pr_used": {
            "ddgclib_surface_tension_force_imported": True,
            "ddgclib_dual_area_heron_imported": True,
            "pr33_heron_forces_for_points_used": True,
            "ddgclib_bridge_film_operator_used": True,
            "pr37_case5_core_available": bool(operators["pr37_core_available"]),
            "pr37_case5_core_path": str(operators["pr37_core_path"]),
        },
        "simulation": {
            "module": "ddgclib.operators.bridge_film",
            "method": simulation["method"],
            "not_used": "Case 7 bottleneck-limited solver for simulation state update",
            "snapshots_s": list(config.snapshot_times_s),
        },
        "metrics": {**ddg_metrics, **metrics},
        "development_targets": [
            "Move the current axisymmetric bridge-film operator from validation-grade to production-grade ddgclib API.",
            "Connect the bridge-film state to PR37 mesh/contact topology when a real sphere-film stitch is required.",
            "Replace the empirical finite-inner dimple reconstruction with a resolved free-surface/mesh operator.",
            "Add automated validation tests for h(r,t), V_br(t), volume balance, and PR force diagnostics.",
        ],
        "config": asdict(config),
        "bridge_film_config": asdict(bridge_config),
        "outputs": {
            "fig1_ddgclib_overlay": str(fig1_overlay.resolve()),
            "full_comparison_png": str(full_panel.resolve()),
            "profile_csv": str(profile_csv.resolve()),
            "volume_diagnostics_csv": str(volume_csv.resolve()),
            "summary_json": str((out_dir / "summary.json").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_case()
    print(f"Wrote outputs to {OUT_DIR.resolve()}")
    print(f"Full comparison PNG: {summary['outputs']['full_comparison_png']}")
    print(f"Summary JSON: {summary['outputs']['summary_json']}")
    print("Truth status:", summary["truth_status"])


if __name__ == "__main__":
    main()
