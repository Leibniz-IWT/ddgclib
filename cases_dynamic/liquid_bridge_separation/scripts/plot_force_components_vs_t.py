#!/usr/bin/env python3
"""Plot saved sphere force components versus time from restart history."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ddgclib_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_RESTART_DIR = Path("v5CoxConti_restart1000_no_tetvolcsv_optimized_v2")


def _latest_restart_json(restart_dir: Path) -> Path:
    restart_jsons = sorted((restart_dir / "restart").glob("restart_step*.json"))
    if restart_jsons:
        return restart_jsons[-1]
    history_json = restart_dir / "separation_history.json"
    if history_json.exists():
        return history_json
    raise FileNotFoundError(f"No restart_step*.json or separation_history.json found under {restart_dir}")


def _load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    history = payload.get("history", payload if isinstance(payload, list) else [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"No non-empty history list found in {path}")
    return [row for row in history if isinstance(row, dict)]


def _array(history: list[dict], key: str, scale: float = 1.0) -> np.ndarray:
    return np.array([float(row.get(key, 0.0)) * scale for row in history], dtype=float)


def _time_array(history: list[dict]) -> np.ndarray:
    t_s = _array(history, "t")
    if not np.any(np.isfinite(t_s)) or float(np.max(t_s) - np.min(t_s)) <= 0.0:
        steps = _array(history, "step")
        dt = _array(history, "dt")
        dt_fallback = float(np.median(dt[dt > 0.0])) if np.any(dt > 0.0) else 0.01
        t_s = steps * dt_fallback
    return t_s


def plot_force_components(history: list[dict], *, out_path: Path, source_label: str) -> None:
    t_s = _time_array(history)
    t_label = "time [s]"
    top_total = _array(history, "top_force_axial", 1.0e3)
    top_cap = _array(history, "top_cap_traction_axial", 1.0e3)
    top_cl = _array(history, "top_contact_line_axial", 1.0e3)
    bottom_total = _array(history, "bottom_force_axial", 1.0e3)
    bottom_cap = _array(history, "bottom_cap_traction_axial", 1.0e3)
    bottom_cl = _array(history, "bottom_contact_line_axial", 1.0e3)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True, constrained_layout=True)

    axes[0].plot(t_s, top_total, label="top total", color="#1f77b4", linewidth=1.6)
    axes[0].plot(t_s, top_cap, label="top cap traction", color="#ff7f0e", linewidth=1.2)
    axes[0].plot(t_s, top_cl, label="top contact-line", color="#2ca02c", linewidth=1.2)
    axes[0].set_ylabel("top axial force [mN]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_s, bottom_total, label="bottom total", color="#1f77b4", linewidth=1.6)
    axes[1].plot(t_s, bottom_cap, label="bottom cap traction", color="#ff7f0e", linewidth=1.2)
    axes[1].plot(t_s, bottom_cl, label="bottom contact-line", color="#2ca02c", linewidth=1.2)
    axes[1].set_ylabel("bottom axial force [mN]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_s, top_cl, label="top contact-line", color="#2ca02c", linewidth=1.4)
    axes[2].plot(t_s, bottom_cl, label="bottom contact-line", color="#9467bd", linewidth=1.4)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_xlabel(t_label)
    axes[2].set_ylabel("CL axial force [mN]")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"Saved axial force components vs time\n{source_label}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _has_solver_force_components(history: list[dict]) -> bool:
    return any("solver_force_all_free_stress_axial" in row for row in history)


def plot_solver_force_components(history: list[dict], *, out_path: Path, source_label: str) -> bool:
    solver_history = [row for row in history if "solver_force_all_free_stress_axial" in row]
    if not solver_history:
        return False

    t_s = _time_array(solver_history)
    colors = {
        "stress": "#1f77b4",
        "pressure_contact_line": "#bcbd22",
        "surface_tension_contact_line": "#9467bd",
        "cox_contact_line": "#2ca02c",
        "contact_line": "#17becf",
        "damping": "#d62728",
        "total": "#111111",
    }
    labels = {
        "stress": r"surface-pressure-viscous force $F_{\sigma,p,\mu}$",
        "pressure_contact_line": r"volume-constraint contact-line pressure force $F_{p,\mathrm{vol},cl}$",
        "surface_tension_contact_line": r"liquid-air surface-tension contact-line force $F_{s,cl}$",
        "cox_contact_line": r"Cox dynamic contact-line force $F_{\mathrm{cox},cl}$",
        "contact_line": r"combined contact-line force $F_{p,\mathrm{vol},cl}+F_{s,cl}+F_{\mathrm{cox},cl}$",
        "damping": "linear damping force -c u",
        "total": r"total solver force $F_{\mathrm{tot}}$",
    }
    groups = [
        ("all_free", "all free vertices"),
        ("bottom_clring", "bottom contact-line ring"),
        ("top_clring", "top contact-line ring"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    for ax, (group_key, title) in zip(axes, groups):
        for component in (
            "stress",
            "pressure_contact_line",
            "surface_tension_contact_line",
            "cox_contact_line",
            "contact_line",
            "damping",
            "total",
        ):
            key = f"solver_force_{group_key}_{component}_axial"
            values_mn = _array(solver_history, key, 1.0e3)
            linewidth = 1.7 if component == "total" else 1.2
            linestyle = "--" if component == "total" else "-"
            ax.plot(
                t_s,
                values_mn,
                label=labels[component],
                color=colors[component],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_ylabel("axial force [mN]")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    step0 = int(float(solver_history[0].get("step", 0)))
    step1 = int(float(solver_history[-1].get("step", 0)))
    fig.suptitle(f"Solver Ftot decomposition vs time, steps {step0}-{step1}\n{source_label}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def plot_solver_force_components_direction(
    history: list[dict],
    *,
    out_path: Path,
    source_label: str,
    direction: str,
    groups: list[tuple[str, str]],
    ylabel: str,
    title: str,
) -> bool:
    required_key = f"solver_force_{groups[0][0]}_stress_{direction}"
    solver_history = [row for row in history if required_key in row]
    if not solver_history:
        return False

    t_s = _time_array(solver_history)
    colors = {
        "stress": "#1f77b4",
        "pressure_contact_line": "#bcbd22",
        "surface_tension_contact_line": "#9467bd",
        "cox_contact_line": "#2ca02c",
        "contact_line": "#17becf",
        "damping": "#d62728",
        "total": "#111111",
    }
    labels = {
        "stress": r"surface-pressure-viscous force $F_{\sigma,p,\mu}$",
        "pressure_contact_line": r"volume-constraint contact-line pressure force $F_{p,\mathrm{vol},cl}$",
        "surface_tension_contact_line": r"liquid-air surface-tension contact-line force $F_{s,cl}$",
        "cox_contact_line": r"Cox dynamic contact-line force $F_{\mathrm{cox},cl}$",
        "contact_line": r"combined contact-line force $F_{p,\mathrm{vol},cl}+F_{s,cl}+F_{\mathrm{cox},cl}$",
        "damping": "linear damping force -c u",
        "total": r"total solver force $F_{\mathrm{tot}}$",
    }
    fig, axes = plt.subplots(len(groups), 1, figsize=(10.0, 3.0 * len(groups)), sharex=True, constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (group_key, group_title) in zip(axes, groups):
        for component in (
            "stress",
            "pressure_contact_line",
            "surface_tension_contact_line",
            "cox_contact_line",
            "contact_line",
            "damping",
            "total",
        ):
            key = f"solver_force_{group_key}_{component}_{direction}"
            values_mn = _array(solver_history, key, 1.0e3)
            linewidth = 1.7 if component == "total" else 1.2
            linestyle = "--" if component == "total" else "-"
            ax.plot(
                t_s,
                values_mn,
                label=labels[component],
                color=colors[component],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_ylabel(ylabel)
        ax.set_title(group_title, fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    step0 = int(float(solver_history[0].get("step", 0)))
    step1 = int(float(solver_history[-1].get("step", 0)))
    fig.suptitle(f"{title}, steps {step0}-{step1}\n{source_label}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def plot_surface_tension_contact_line_force(history: list[dict], *, out_path: Path, source_label: str) -> bool:
    key_prefix = "surface_tension_contact_line_force_from_liquid_surface_on_contact_line"
    required_key = f"{key_prefix}_bottom_clring_slide"
    fscl_history = [row for row in history if required_key in row]
    if not fscl_history:
        return False

    t_s = _time_array(fscl_history)
    groups = [
        ("bottom_clring", "bottom contact-line ring"),
        ("top_clring", "top contact-line ring"),
    ]
    projection = "radial"

    fig, ax_left = plt.subplots(1, 1, figsize=(12.8, 6.4), constrained_layout=True)
    ax_right = ax_left.twinx()
    colors = {
        "surface_tension_contact_line": "#9467bd",
        "cox_contact_line": "#2ca02c",
        "pressure_contact_line": "#1f77b4",
        "total": "#111111",
    }
    labels = {
        "surface_tension_contact_line": r"liquid-air surface-tension force $F_{s,cl}^{(r)}$",
        "cox_contact_line": r"Cox dynamic contact-line force $F_{\mathrm{cox},cl}^{(r)}$",
        "pressure_contact_line": r"volume-constraint pressure force $F_{p,\mathrm{vol},cl}^{(r)}$",
        "total": r"clean radial contact-line total $F_{\mathrm{tot},cl}^{(r)}$",
    }

    ring_styles = {
        "bottom_clring": ("bottom contact-line ring", "-", "o"),
        "top_clring": ("top contact-line ring", "--", "s"),
    }
    plot_order = (
        "surface_tension_contact_line",
        "cox_contact_line",
        "pressure_contact_line",
        "total",
    )
    dominant_names = ("surface_tension_contact_line", "total")
    correction_names = ("cox_contact_line", "pressure_contact_line")
    dominant_values_mn = []
    correction_values_mn = []

    for group_key, _group_label in groups:
        ring_label, linestyle, marker = ring_styles[group_key]
        fs_key = f"{key_prefix}_{group_key}_{projection}"
        fcox_key = f"solver_force_{group_key}_cox_contact_line_{projection}"
        fp_key = f"solver_force_{group_key}_pressure_contact_line_{projection}"
        total_key = f"solver_force_{group_key}_total_{projection}"
        series = {
            "surface_tension_contact_line": _array(fscl_history, fs_key, 1.0e3),
            "cox_contact_line": _array(fscl_history, fcox_key, 1.0e3),
            "pressure_contact_line": _array(fscl_history, fp_key, 1.0e3),
            "total": _array(fscl_history, total_key, 1.0e3),
        }
        for name in plot_order:
            values_mn = series[name]
            target_ax = ax_left if name in dominant_names else ax_right
            if name in dominant_names:
                dominant_values_mn.append(values_mn)
            else:
                correction_values_mn.append(values_mn)
            linewidth = 2.4 if name == "total" else 1.8
            alpha = 0.95 if name in ("pressure_contact_line", "cox_contact_line") else 0.86
            target_ax.plot(
                t_s,
                values_mn,
                color=colors[name],
                linewidth=linewidth,
                linestyle=linestyle,
                marker=marker,
                markersize=5.0,
                alpha=alpha,
            )
    dominant_values = np.concatenate(dominant_values_mn)
    dominant_values = dominant_values[np.isfinite(dominant_values)]
    if dominant_values.size:
        ymin = float(np.min(dominant_values))
        ymax = float(np.max(dominant_values))
        pad = max(0.12 * (ymax - ymin), 1.0e-4)
        ax_left.set_ylim(ymin - pad, ymax + pad)
    correction_values = np.concatenate(correction_values_mn)
    correction_values = correction_values[np.isfinite(correction_values)]
    ax_right.set_yscale("symlog", linthresh=1.0e-8, linscale=1.35, base=10)
    if correction_values.size:
        ymin = min(float(np.min(correction_values)) * 2.0, -1.0e-8)
        ymax = max(float(np.max(correction_values)) * 2.0, 1.0e-8)
        ax_right.set_ylim(ymin, ymax)
    ax_right.axhline(0.0, color="#1f77b4", linewidth=0.9, alpha=0.45)
    ax_left.set_xlabel("time [s]")
    ax_left.set_ylabel(r"dominant inward radial force [mN]: $F_{s,cl}^{(r)}$, $F_{\mathrm{tot},cl}^{(r)}$")
    ax_right.set_ylabel(r"small correction radial force [mN], symlog: $F_{\mathrm{cox},cl}^{(r)}$, $F_{p,\mathrm{vol},cl}^{(r)}$")
    ax_left.grid(True, which="major", alpha=0.28)
    ax_right.grid(True, which="minor", axis="y", alpha=0.12)
    legend_handles = [
        Line2D([0], [0], color=colors["surface_tension_contact_line"], lw=2.0, label=labels["surface_tension_contact_line"]),
        Line2D([0], [0], color=colors["total"], lw=2.4, label=labels["total"]),
        Line2D([0], [0], color=colors["cox_contact_line"], lw=2.0, label=labels["cox_contact_line"]),
        Line2D([0], [0], color=colors["pressure_contact_line"], lw=2.0, label=labels["pressure_contact_line"]),
        Line2D([0], [0], color="0.2", lw=2.0, linestyle="-", marker="o", label="bottom contact-line ring"),
        Line2D([0], [0], color="0.2", lw=2.0, linestyle="--", marker="s", label="top contact-line ring"),
    ]
    ax_left.set_title(
        "left axis: dominant forces; right axis: small correction forces. "
        "solid/circle: bottom ring; dashed/square: top ring",
        fontsize=10,
    )
    ax_left.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(0.98, 0.52), ncol=1, fontsize=8)
    step0 = int(float(fscl_history[0].get("step", 0)))
    step1 = int(float(fscl_history[-1].get("step", 0)))
    fig.suptitle(
        r"Clean radial contact-line force balance: "
        r"$F_{s,cl}^{(r)}$, $F_{\mathrm{cox},cl}^{(r)}$, "
        r"$F_{p,\mathrm{vol},cl}^{(r)}$, and $F_{\mathrm{tot},cl}^{(r)}$, "
        f"steps {step0}-{step1}\n"
        f"{source_label}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-dir", type=Path, default=DEFAULT_RESTART_DIR)
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    source = args.history_json if args.history_json is not None else _latest_restart_json(args.restart_dir)
    history = _load_history(source)
    out_path = args.out or (args.restart_dir / "force_components_vs_t.png")
    plot_force_components(history, out_path=out_path, source_label=str(source))
    print(out_path)
    solver_out = out_path.with_name("solver_force_components_vs_t.png")
    if plot_solver_force_components(history, out_path=solver_out, source_label=str(source)):
        print(solver_out)
    else:
        print("No solver_force_* diagnostics found in this history; rerun the solver after this patch to create them.")
    radial_out = out_path.with_name("solver_force_components_radial_vs_t.png")
    if plot_solver_force_components_direction(
        history,
        out_path=radial_out,
        source_label=str(source),
        direction="radial",
        groups=[
            ("all_free", "all free vertices"),
            ("bottom_clring", "bottom contact-line ring"),
            ("top_clring", "top contact-line ring"),
        ],
        ylabel="outward radial force [mN]",
        title="Solver Ftot radial decomposition vs time",
    ):
        print(radial_out)
    else:
        print("No solver_force_*_radial diagnostics found in this history; rerun the solver after this patch.")
    slide_out = out_path.with_name("solver_force_components_cl_slide_vs_t.png")
    if plot_solver_force_components_direction(
        history,
        out_path=slide_out,
        source_label=str(source),
        direction="slide",
        groups=[
            ("bottom_clring", "bottom contact-line ring"),
            ("top_clring", "top contact-line ring"),
        ],
        ylabel="sphere-tangent slide force [mN]",
        title="Solver Ftot contact-line slide decomposition vs time",
    ):
        print(slide_out)
    else:
        print("No solver_force_*_slide diagnostics found in this history; rerun the solver after this patch.")
    fscl_out = out_path.with_name("contact_line_radial_clean_balance_Ftot_Fpvol_Fcox_Fs_vs_t.png")
    if plot_surface_tension_contact_line_force(history, out_path=fscl_out, source_label=str(source)):
        print(fscl_out)
    else:
        print("No surface_tension_contact_line_force_from_liquid_surface_on_contact_line_* diagnostics found; rerun the solver after this patch.")


if __name__ == "__main__":
    main()
