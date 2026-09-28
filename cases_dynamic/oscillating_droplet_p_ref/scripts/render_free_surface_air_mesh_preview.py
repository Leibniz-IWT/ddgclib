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


def surface_edge_indices(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tri in np.asarray(faces, dtype=int):
        a, b, c = [int(index) for index in tri]
        edges.update(tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
    return np.asarray(sorted(edges), dtype=int)


def normalised(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1)
    return points / np.maximum(norms[:, None], 1.0e-300)


def render_case(
    *,
    case_dir: Path,
    closure: str,
    output_name: str,
    liquid_color: str,
    outer_radius_scale: float = 1.65,
    max_frames: int = 32,
    fps: int = 12,
) -> Path:
    rows = json.loads((case_dir / f"flux_{closure}_timeseries.json").read_text())
    data = np.load(case_dir / f"flux_{closure}_snapshots.npz", allow_pickle=True)
    snapshots = np.asarray(data["points"], dtype=float)
    faces = np.asarray(data["faces"], dtype=int)
    solver_tets = np.asarray(data["tets"], dtype=int)
    surface_indices = np.asarray(data["surface_indices"], dtype=int)
    if "display_tets" in data.files:
        display_tets = [np.asarray(item, dtype=int) for item in data["display_tets"]]
    else:
        display_tets = [solver_tets for _ in range(snapshots.shape[0])]

    t = row_array(rows, "t")
    amplitude_theory = row_array(rows, "shape_amplitude_theory")
    amplitude_fit = row_array(rows, "shape_amplitude_fit")
    cpu_time = row_array(rows, "cpu_time_s")
    display_changed = row_array(rows, "display_changed_tets")
    p_gas = float(rows[0].get("ambient_pressure_pa", 101325.0))
    p_gauge = row_array(rows, "closure_pressure_pa")

    reference_surface = snapshots[0, surface_indices]
    radius = float(np.mean(np.linalg.norm(reference_surface, axis=1)))
    outer_radius = float(outer_radius_scale) * radius
    outer_surface = normalised(reference_surface) * outer_radius
    air_edges = surface_edge_indices(faces)
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), max_frames), dtype=int))

    if closure == "compressible":
        title = "#11 compressible: one-phase droplet + passive ambient gas boundary"
    else:
        title = "#12 incompressible: one-phase droplet + passive ambient gas boundary"

    fig = plt.figure(figsize=(4.8, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.15, 1.0), hspace=0.08)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax_amp = fig.add_subplot(grid[1])
    limit = outer_radius * 1.08
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)

    def draw(frame_pos: int) -> None:
        row_idx = int(frame_indices[frame_pos])
        points = snapshots[row_idx]
        surface = points[surface_indices]
        liquid_edges = base.tet_edge_indices(display_tets[row_idx], max_edges=260)

        ax3d.cla()
        ax_amp.cla()

        ax3d.add_collection3d(
            Poly3DCollection(
                outer_surface[faces],
                facecolor=(0.45, 0.66, 0.88, 0.055),
                edgecolor=(0.22, 0.35, 0.52, 0.18),
                linewidth=0.35,
            )
        )
        ax3d.add_collection3d(
            Line3DCollection(
                outer_surface[air_edges],
                colors=(0.20, 0.33, 0.48, 0.26),
                linewidths=0.45,
            )
        )
        radial_segments = np.stack([surface, outer_surface], axis=1)
        keep = np.linspace(0, radial_segments.shape[0] - 1, min(42, radial_segments.shape[0]), dtype=int)
        ax3d.add_collection3d(
            Line3DCollection(
                radial_segments[keep],
                colors=(0.20, 0.33, 0.48, 0.20),
                linewidths=0.42,
            )
        )

        ax3d.add_collection3d(
            Poly3DCollection(
                surface[faces],
                facecolor=liquid_color,
                edgecolor=liquid_color,
                linewidth=0.38,
                alpha=0.50,
            )
        )
        ax3d.add_collection3d(
            Line3DCollection(points[liquid_edges], colors=(0.05, 0.05, 0.05, 0.28), linewidths=0.36)
        )

        interior = np.setdiff1d(np.arange(points.shape[0], dtype=int), surface_indices)
        if interior.size:
            interior_points = points[interior]
            ax3d.scatter(
                interior_points[:, 0],
                interior_points[:, 1],
                interior_points[:, 2],
                s=6,
                c="#2ca02c",
                alpha=0.62,
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
        ax3d.view_init(elev=19.0, azim=30.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=8.6, pad=4)
        ax3d.text2D(
            0.02,
            0.96,
            (
                f"t = {t[row_idx]:.4f} s\n"
                f"p_gas = {p_gas:.0f} Pa passive\n"
                f"p_gauge = {p_gauge[row_idx]:.1f} Pa\n"
                f"air mesh = visual boundary\n"
                f"Delaunay changes = {display_changed[row_idx]:.0f} tets"
            ),
            transform=ax3d.transAxes,
            va="top",
            ha="left",
            fontsize=7.4,
            family="monospace",
            color="#17202A",
        )

        pad = max(0.1 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.4, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=liquid_color, lw=1.7, label="simulation")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=4.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(
            float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - pad),
            float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + pad),
        )
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_xlabel("t [s]", fontsize=8)
        ax_amp.set_ylabel("a2", fontsize=8)
        ax_amp.tick_params(labelsize=7)
        ax_amp.set_title(f"Shape amplitude, CPU {cpu_time[-1]:.1f} s", fontsize=8)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=7, loc="upper right")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.07)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    output = case_dir / output_name
    animation.save(output, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one-phase droplet previews with a passive ambient-gas boundary mesh.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [
        (
            ROOT / "out/sphere_fheron_case11_compressible_free_surface_retriang_diag",
            "compressible",
            "flux_compressible_sphere_air_boundary_preview.gif",
            "#1f77b4",
        ),
        (
            ROOT / "out/sphere_fheron_case12_incompressible_free_surface_retriang_diag",
            "incompressible",
            "flux_incompressible_sphere_air_boundary_preview.gif",
            "#ff7f0e",
        ),
    ]
    for case_dir, closure, output_name, color in cases:
        print(render_case(case_dir=case_dir, closure=closure, output_name=output_name, liquid_color=color, max_frames=args.max_frames, fps=args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
