"""Render #11/#12 dynamic-integrator two-phase validation GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sphere_fheron_eos_projection_benchmark as base


plt = base.plt
FuncAnimation = base.FuncAnimation
PillowWriter = base.PillowWriter
Line3DCollection = base.Line3DCollection
Poly3DCollection = base.Poly3DCollection

LIQUID = 1
GAS = 0


def row_array(rows: list[dict[str, float]], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray([float(row.get(key, default)) for row in rows], dtype=float)


def _frame_array(values, index: int) -> np.ndarray:
    item = values[index]
    if isinstance(item, np.ndarray):
        return np.asarray(item)
    return np.asarray(item, dtype=int)


def render_case(
    *,
    case_id: int,
    source_dir: Path,
    out_dir: Path,
    max_frames: int,
    fps: int,
) -> Path:
    closure = "compressible" if int(case_id) == 11 else "incompressible"
    stem = f"case{int(case_id)}_dynamic_integrator_twophase_{closure}"
    rows = json.loads((source_dir / f"{stem}.json").read_text())
    data = np.load(source_dir / f"{stem}_snapshots.npz", allow_pickle=True)
    snapshots = np.asarray(data["points"], dtype=float)
    faces = np.asarray(data["faces"], dtype=int)
    interface_indices = np.asarray(data["surface_indices"], dtype=int)
    outer_indices = np.asarray(data["outer_indices"], dtype=int)
    display_tets = data["display_tets"]
    display_phases = data["phases"]

    t = row_array(rows, "t")
    amplitude_theory = row_array(rows, "shape_amplitude_theory")
    amplitude_fit = row_array(rows, "shape_amplitude_fit")
    liquid_vol_pct = (row_array(rows, "liquid_vol_rel") - 1.0) * 100.0
    gas_vol_pct = (row_array(rows, "gas_vol_rel") - 1.0) * 100.0
    local_pct = row_array(rows, "local_volume_rel_max_abs") * 100.0
    mass_flux = row_array(rows, "mass_flux_l1_kg_s")
    momentum_flux = row_array(rows, "momentum_flux_l1_n")
    pressure = row_array(rows, "closure_pressure_pa")
    fheron = row_array(rows, "fheron_pressure_pa")
    cpu_time = row_array(rows, "cpu_time_s")
    cpu_ms = row_array(rows, "cpu_ms_per_substep")
    changed_tets = row_array(rows, "display_changed_tets")
    changed_fraction = row_array(rows, "display_changed_fraction")

    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), int(max_frames)), dtype=int))
    max_radius = float(np.max(np.linalg.norm(snapshots.reshape((-1, 3)), axis=1)))
    limit = 1.08 * max_radius
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)

    if closure == "compressible":
        liquid_color = "#1f77b4"
        surface_face = "#9ecae1"
        title = "#11 ddgclib two-phase mesh + multiphase EOS solve"
    else:
        liquid_color = "#ff7f0e"
        surface_face = "#fdd0a2"
        title = "#12 ddgclib two-phase mesh + multiphase projection"

    fig = plt.figure(figsize=(10.4, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.35, 1.0), wspace=0.27, hspace=0.50)
    ax3d = fig.add_subplot(grid[:2, 0], projection="3d")
    ax_info = fig.add_subplot(grid[2, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_vol = fig.add_subplot(grid[1, 1])
    ax_flux = fig.add_subplot(grid[2, 1])
    ax_flux_twin = ax_flux.twinx()

    def draw(frame_pos: int) -> None:
        row_idx = int(frame_indices[frame_pos])
        points = snapshots[row_idx]
        tets = np.asarray(_frame_array(display_tets, row_idx), dtype=int)
        phases = np.asarray(_frame_array(display_phases, row_idx), dtype=int)
        liquid_tets = tets[phases == LIQUID]
        gas_tets = tets[phases == GAS]
        liquid_edges = base.tet_edge_indices(liquid_tets, max_edges=260) if liquid_tets.size else np.empty((0, 2), dtype=int)
        gas_edges = base.tet_edge_indices(gas_tets, max_edges=420) if gas_tets.size else np.empty((0, 2), dtype=int)
        interface_surface = points[interface_indices]
        outer_surface = points[outer_indices] if outer_indices.size else np.empty((0, 3), dtype=float)

        ax3d.cla()
        ax_info.cla()
        ax_amp.cla()
        ax_vol.cla()
        ax_flux.cla()
        ax_flux_twin.cla()

        if outer_surface.size:
            ax3d.add_collection3d(
                Poly3DCollection(
                    outer_surface[faces],
                    facecolor=(0.52, 0.70, 0.88, 0.075),
                    edgecolor=(0.22, 0.34, 0.50, 0.18),
                    linewidth=0.26,
                )
            )
        if gas_edges.size:
            ax3d.add_collection3d(Line3DCollection(points[gas_edges], colors=(0.20, 0.35, 0.55, 0.18), linewidths=0.30))
        ax3d.add_collection3d(
            Poly3DCollection(
                interface_surface[faces],
                facecolor=surface_face,
                edgecolor=liquid_color,
                linewidth=0.38,
                alpha=0.54,
            )
        )
        if liquid_edges.size:
            ax3d.add_collection3d(Line3DCollection(points[liquid_edges], colors=(0.05, 0.05, 0.05, 0.30), linewidths=0.34))

        gas_vertices = np.setdiff1d(np.unique(gas_tets.reshape(-1)) if gas_tets.size else np.array([], dtype=int), interface_indices)
        liquid_vertices = np.setdiff1d(np.unique(liquid_tets.reshape(-1)) if liquid_tets.size else np.array([], dtype=int), interface_indices)
        if gas_vertices.size:
            ax3d.scatter(points[gas_vertices, 0], points[gas_vertices, 1], points[gas_vertices, 2], s=5, c="#8fb8de", alpha=0.45, edgecolors="none", depthshade=False)
        if liquid_vertices.size:
            ax3d.scatter(points[liquid_vertices, 0], points[liquid_vertices, 1], points[liquid_vertices, 2], s=7, c="#2ca02c", alpha=0.65, edgecolors="none", depthshade=False)
        ax3d.scatter(interface_surface[:, 0], interface_surface[:, 1], interface_surface[:, 2], s=11, c=liquid_color, alpha=0.86, edgecolors="white", linewidths=0.20, depthshade=False)

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=10, pad=0)

        ax_info.axis("off")
        density_spread_ppm = (row_array(rows, "nodal_density_rel_max")[row_idx] - row_array(rows, "nodal_density_rel_min")[row_idx]) * 1.0e6
        left = (
            f"t = {t[row_idx]:.5f} s\n"
            f"theory a2 = {amplitude_theory[row_idx]:+.4f}\n"
            f"fit a2 = {amplitude_fit[row_idx]:+.4f}\n"
            f"liquid dV/V0 = {liquid_vol_pct[row_idx]:+.3e} %\n"
            f"gas dV/V0 = {gas_vol_pct[row_idx]:+.3e} %\n"
            f"max liquid tet = {local_pct[row_idx]:.3e} %\n"
            f"rho spread = {density_spread_ppm:.3f} ppm\n"
            f"mass flux L1 = {mass_flux[row_idx]:.3e} kg/s\n"
            f"mom flux L1 = {momentum_flux[row_idx]:.3e} N"
        )
        right = (
            f"mesh = {points.shape[0]} verts, {tets.shape[0]} tets\n"
            f"liquid/gas tets = {liquid_tets.shape[0]}/{gas_tets.shape[0]}\n"
            f"interface vertices = {interface_indices.shape[0]}\n"
            f"Delaunay flips = {changed_tets[row_idx]:.0f} ({100.0 * changed_fraction[row_idx]:.1f}%)\n"
            f"integrator = symplectic Euler\n"
            f"flux = Lagrangian material face\n"
            f"Fp = multiphase tet-volume Bt\n"
            f"p = {pressure[row_idx]:.2f} Pa\n"
            f"FHeron = {fheron[row_idx]:.2f} Pa\n"
            f"CPU = {cpu_time[row_idx]:.2f}/{cpu_time[-1]:.1f} s\n"
            f"CPU/step = {cpu_ms[row_idx]:.2f} ms"
        )
        ax_info.text(0.02, 0.98, left, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.4)
        ax_info.text(0.52, 0.98, right, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.4)

        pad = max(0.1 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.4, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=liquid_color, lw=1.7, label="simulation")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=4.2)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - pad), float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + pad))
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        ax_vol.plot(t, liquid_vol_pct, color=liquid_color, lw=1.5, label="liquid global dV/V")
        ax_vol.plot(t, gas_vol_pct, color="#4c78a8", lw=1.2, ls="--", label="gas global dV/V")
        ax_vol.plot(t, local_pct, color="#111111", lw=1.0, ls=":", label="liquid local max")
        ax_vol.plot(t[row_idx], liquid_vol_pct[row_idx], "o", color="#d62728", ms=4.0)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        ax_vol.set_xticks(time_ticks)
        ax_vol.set_ylabel("percent")
        ax_vol.set_title("Global and local volume error")
        ax_vol.grid(True, alpha=0.25)
        ax_vol.legend(frameon=False, fontsize=7, loc="upper right")

        mass_flux_plot = np.maximum(mass_flux, 1.0e-30)
        momentum_flux_plot = np.maximum(momentum_flux, 1.0e-30)
        ax_flux.plot(t, mass_flux_plot, color=liquid_color, lw=1.8, label="mass flux L1")
        ax_flux_twin.plot(t, momentum_flux_plot, color="0.25", lw=1.2, ls=":", label="momentum flux L1")
        ax_flux.plot(t[row_idx], max(mass_flux[row_idx], 1.0e-30), "o", color="#d62728", ms=4.0)
        ax_flux.set_yscale("log")
        ax_flux_twin.set_yscale("log")
        mass_flux_max = max(float(np.max(mass_flux_plot)), 1.0e-30)
        momentum_flux_max = max(float(np.max(momentum_flux_plot)), 1.0e-30)
        ax_flux.set_ylim(max(mass_flux_max * 1.0e-6, 1.0e-30), mass_flux_max * 5.0)
        ax_flux_twin.set_ylim(max(momentum_flux_max * 1.0e-6, 1.0e-30), momentum_flux_max * 5.0)
        ax_flux.set_xlim(float(t[0]), float(t[-1]))
        ax_flux.set_xticks(time_ticks)
        ax_flux.set_xlabel("t [s]")
        ax_flux.set_ylabel("kg/s")
        ax_flux_twin.set_ylabel("N")
        ax_flux.set_title("Internal face fluxes")
        ax_flux.grid(True, alpha=0.25)
        lines, labels = ax_flux.get_legend_handles_labels()
        lines2, labels2 = ax_flux_twin.get_legend_handles_labels()
        ax_flux.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")

        fig.suptitle(
            f"{closure.capitalize()} ddgclib two-phase run, CPU {cpu_time[-1]:.1f} s ({cpu_ms[-1]:.1f} ms/step)",
            y=0.985,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.07)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / f"case{int(case_id)}_ddgclib_twophase_multiphase_layout.gif"
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    draw(len(frame_indices) - 1)
    fig.savefig(gif_path.with_suffix(".preview.png"), dpi=150)
    plt.close(fig)
    return gif_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render #11/#12 dynamic-integrator two-phase GIFs.")
    parser.add_argument("--case-id", type=int, choices=(11, 12), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.source_dir
    gif_path = render_case(
        case_id=args.case_id,
        source_dir=args.source_dir,
        out_dir=out_dir,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    print(f"gif: {gif_path}")
    print(f"preview: {gif_path.with_suffix('.preview.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
