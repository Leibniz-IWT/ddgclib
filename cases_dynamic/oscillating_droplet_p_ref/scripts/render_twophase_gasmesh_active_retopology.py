"""Render active-retopology FHeron runs with an explicit passive gas mesh.

The matching #11/#12 dynamics are computed by ``sphere_fheron_flux_fv_benchmark``
using the stable liquid free-surface FHeron closure.  This renderer adds a
GitHub-style two-phase triangulation view: phase 1 is the solved liquid mesh,
phase 0 is a passive outer gas mesh connected to the moving interface and a
fixed outer shell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sphere_fheron_eos_projection_benchmark as base


plt = base.plt
FuncAnimation = base.FuncAnimation
PillowWriter = base.PillowWriter
Line3DCollection = base.Line3DCollection
Poly3DCollection = base.Poly3DCollection


ROOT = Path(__file__).resolve().parent


def row_array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def gas_tets_for_surface(surface_indices: np.ndarray, faces: np.ndarray, outer_offset: int) -> np.ndarray:
    gas_tets: list[list[int]] = []
    for a, b, c in np.asarray(faces, dtype=int):
        ia = int(surface_indices[int(a)])
        ib = int(surface_indices[int(b)])
        ic = int(surface_indices[int(c)])
        oa = int(outer_offset + int(a))
        ob = int(outer_offset + int(b))
        oc = int(outer_offset + int(c))
        gas_tets.extend(
            [
                [ia, ib, ic, oc],
                [ia, ib, ob, oc],
                [ia, oa, ob, oc],
            ]
        )
    return np.asarray(gas_tets, dtype=int)


def shifted_surface_edges(faces: np.ndarray, offset: int, *, max_edges: int = 220) -> np.ndarray:
    edges = sorted(
        {
            tuple(sorted((int(offset + int(i)), int(offset + int(j)))))
            for tri in np.asarray(faces, dtype=int)
            for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        }
    )
    if len(edges) > int(max_edges):
        keep = np.linspace(0, len(edges) - 1, int(max_edges), dtype=int)
        edges = [edges[int(index)] for index in keep]
    return np.asarray(edges, dtype=int).reshape((-1, 2))


def fixed_outer_shell(surface0: np.ndarray, radius: float, outer_scale: float) -> np.ndarray:
    directions = np.asarray(surface0, dtype=float).copy()
    norms = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(norms[:, None], 1.0e-300)
    return directions * (float(outer_scale) * float(radius))


def render_case(
    *,
    source_dir: Path,
    out_dir: Path,
    closure: str,
    radius: float,
    outer_scale: float,
    max_frames: int,
    fps: int,
) -> Path:
    rows = json.loads((source_dir / f"flux_{closure}_timeseries.json").read_text())
    data = np.load(source_dir / f"flux_{closure}_snapshots.npz", allow_pickle=True)
    snapshots = np.asarray(data["points"], dtype=float)
    faces = np.asarray(data["faces"], dtype=int)
    surface_indices = np.asarray(data["surface_indices"], dtype=int)
    display_tets = data["display_tets"] if "display_tets" in data.files else None

    t = row_array(rows, "t")
    amplitude_theory = row_array(rows, "shape_amplitude_theory")
    amplitude_fit = row_array(rows, "shape_amplitude_fit")
    cpu_time = row_array(rows, "cpu_time_s")
    pressure = row_array(rows, "closure_pressure_pa")
    fheron = row_array(rows, "fheron_pressure_pa")
    global_volume_pct = (row_array(rows, "vol_rel") - 1.0) * 100.0

    outer = fixed_outer_shell(snapshots[0, surface_indices], radius, outer_scale)
    outer_offset = snapshots.shape[1]
    gas_tets = gas_tets_for_surface(surface_indices, faces, outer_offset)
    gas_edges = base.tet_edge_indices(gas_tets, max_edges=520)
    outer_edges = shifted_surface_edges(faces, outer_offset, max_edges=220)

    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), max(2, int(max_frames))), dtype=int))
    limit = float(outer_scale) * float(radius) * 1.08
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)
    time_tick_labels = [f"{value:.3f}" for value in time_ticks]

    if closure == "compressible":
        liquid_color = "#1f77b4"
        gif_name = "flux_compressible_twophase_gasmesh_sphere.gif"
        title = "#11 compressible: phase-1 liquid + passive phase-0 gas mesh"
    else:
        liquid_color = "#ff7f0e"
        gif_name = "flux_incompressible_twophase_gasmesh_sphere.gif"
        title = "#12 incompressible: phase-1 liquid + passive phase-0 gas mesh"

    fig = plt.figure(figsize=(5.35, 7.35))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.5, 1.05), hspace=0.16)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax_amp = fig.add_subplot(grid[1])

    def draw(frame_pos: int) -> None:
        row_idx = int(frame_indices[frame_pos])
        liquid_points = snapshots[row_idx]
        combined = np.vstack([liquid_points, outer])
        surface = liquid_points[surface_indices]
        liquid_tets = (
            np.asarray(display_tets[row_idx], dtype=int)
            if display_tets is not None
            else np.asarray(data["tets"], dtype=int)
        )
        liquid_edges = base.tet_edge_indices(liquid_tets, max_edges=360)
        interior = np.setdiff1d(np.arange(liquid_points.shape[0], dtype=int), surface_indices)

        ax3d.cla()
        ax_amp.cla()

        ax3d.add_collection3d(
            Poly3DCollection(
                outer[faces],
                facecolor=(0.32, 0.58, 0.84, 0.070),
                edgecolor=(0.20, 0.38, 0.60, 0.18),
                linewidth=0.28,
            )
        )
        if gas_edges.size:
            ax3d.add_collection3d(
                Line3DCollection(combined[gas_edges], colors=(0.20, 0.38, 0.62, 0.26), linewidths=0.30)
            )
        if outer_edges.size:
            ax3d.add_collection3d(
                Line3DCollection(combined[outer_edges], colors=(0.14, 0.30, 0.50, 0.18), linewidths=0.22)
            )
        ax3d.add_collection3d(
            Poly3DCollection(
                surface[faces],
                facecolor=liquid_color,
                edgecolor=liquid_color,
                linewidth=0.34,
                alpha=0.58,
            )
        )
        if liquid_edges.size:
            ax3d.add_collection3d(
                Line3DCollection(liquid_points[liquid_edges], colors=(0.04, 0.04, 0.04, 0.26), linewidths=0.30)
            )
        if interior.size:
            shell_points = liquid_points[interior]
            ax3d.scatter(
                shell_points[:, 0],
                shell_points[:, 1],
                shell_points[:, 2],
                s=7,
                c="#2ca02c",
                alpha=0.75,
                edgecolors="none",
                depthshade=False,
            )

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=8.8, pad=4)
        ax3d.text2D(
            0.02,
            0.97,
            (
                f"t = {t[row_idx]:.5f} s\n"
                f"phase 1 liquid verts = {liquid_points.shape[0]}\n"
                f"phase 0 gas verts = {outer.shape[0]}\n"
                f"liquid tets = {liquid_tets.shape[0]}\n"
                f"gas tets = {gas_tets.shape[0]}\n"
                f"p_gauge = {pressure[row_idx]:.2f} Pa\n"
                f"FHeron = {fheron[row_idx]:.2f} Pa\n"
                f"dV/V0 = {global_volume_pct[row_idx]:+.2e} %\n"
                f"CPU = {cpu_time[row_idx]:.1f}/{cpu_time[-1]:.1f} s"
            ),
            transform=ax3d.transAxes,
            va="top",
            ha="left",
            fontsize=6.8,
            family="monospace",
        )

        pad = max(0.12 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.35, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=liquid_color, lw=1.7, label="simulation")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=4.2)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_xticklabels(time_tick_labels)
        ax_amp.set_ylim(
            float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - pad),
            float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + pad),
        )
        ax_amp.set_xlabel("t [s]", fontsize=8)
        ax_amp.set_ylabel("a2", fontsize=8)
        ax_amp.set_title("Shape amplitude", fontsize=8)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.tick_params(labelsize=7)
        ax_amp.legend(frameon=False, fontsize=7, loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.965, bottom=0.07)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / gif_name
    animation.save(path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render active-retopology runs with passive gas mesh overlay.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "out" / "sphere_fheron_case11_12_active_retopology_full")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "sphere_fheron_case11_12_twophase_gasmesh_active_retopology")
    parser.add_argument("--radius", type=float, default=1.0e-3)
    parser.add_argument("--outer-scale", type=float, default=1.62)
    parser.add_argument("--closure", choices=("both", "compressible", "incompressible"), default="both")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    closures = ("compressible", "incompressible") if args.closure == "both" else (args.closure,)
    for closure in closures:
        print(
            render_case(
                source_dir=args.source_dir,
                out_dir=args.out_dir,
                closure=closure,
                radius=args.radius,
                outer_scale=args.outer_scale,
                max_frames=args.max_frames,
                fps=args.fps,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
