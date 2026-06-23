"""Render #11/#12 using the upstream GitHub two-phase droplet mesh.

This file is intentionally a renderer/diagnostic: it uses the successful
active-retopology FHeron time series for the Rayleigh response, but the
displayed phase mesh is created by the GitHub oscillating-droplet setup
(`setup_oscillating_droplet` -> `droplet_in_box_3d`).  The point is to make
the comparison foundation visible: phase 0 is gas/outer fluid, phase 1 is
liquid, and phase -1 is the shared interface subcomplex.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_github_twophase_oscillating_preview as github_case
import sphere_fheron_eos_projection_benchmark as base


plt = base.plt
FuncAnimation = base.FuncAnimation
PillowWriter = base.PillowWriter
Line3DCollection = base.Line3DCollection
Poly3DCollection = base.Poly3DCollection

ROOT = Path(__file__).resolve().parent
RADIUS = 1.0e-3
DOMAIN_HALF_WIDTH = 5.0e-3


def row_array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def load_github_twophase_mesh() -> dict[str, np.ndarray]:
    HC, _bV, _mps, _bc_set, _dudt_fn, _retopo_fn, _params = github_case.setup_oscillating_droplet(
        dim=3,
        R0=RADIUS,
        epsilon=0.0,
        l=2,
        rho_d=1000.0,
        rho_o=1.225,
        mu_d=0.001,
        mu_o=1.81e-5,
        gamma=0.072,
        K_d=1000.0,
        K_o=1.225,
        L_domain=DOMAIN_HALF_WIDTH,
        refinement_outer=1,
        refinement_droplet=1,
    )
    snap = github_case.snapshot_from_hc(HC, 3)
    verts = github_case.ordered_vertices(HC)
    index_by_coord = {tuple(v.x): i for i, v in enumerate(verts)}
    interface_edges: list[tuple[int, int]] = []
    for edge in getattr(HC, "interface_edges", set()):
        ids = [index_by_coord.get(tuple(coord)) for coord in edge]
        if len(ids) == 2 and all(i is not None for i in ids):
            interface_edges.append(tuple(sorted((int(ids[0]), int(ids[1])))))
    interface_triangles: list[tuple[int, int, int]] = []
    for tri in getattr(HC, "interface_triangles", set()):
        ids = [index_by_coord.get(tuple(coord)) for coord in tri]
        if all(i is not None for i in ids):
            interface_triangles.append(tuple(int(i) for i in ids))
    snap["interface_edges"] = np.asarray(sorted(set(interface_edges)), dtype=int).reshape((-1, 2))
    snap["interface_triangles"] = np.asarray(interface_triangles, dtype=int).reshape((-1, 3))
    return snap


def p2_basis(points: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(points, axis=1)
    mu = np.divide(points[:, 2], r, out=np.zeros_like(r), where=r > 1.0e-300)
    return 0.5 * (3.0 * mu * mu - 1.0)


def wall_radius_along_direction(directions: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(directions), axis=1)
    return DOMAIN_HALF_WIDTH / np.maximum(max_abs, 1.0e-300)


def deform_github_mesh(mesh: dict[str, np.ndarray], amplitude: float) -> np.ndarray:
    reference = np.asarray(mesh["points"], dtype=float)
    phases = np.asarray(mesh["phases"], dtype=int)
    radii = np.linalg.norm(reference, axis=1)
    directions = np.divide(reference, radii[:, None], out=np.zeros_like(reference), where=radii[:, None] > 1.0e-300)
    basis = p2_basis(reference)
    interface_radius = RADIUS * (1.0 + float(amplitude) * basis)

    deformed = reference.copy()
    liquid_or_interface = phases != 0
    deformed[liquid_or_interface] = directions[liquid_or_interface] * (
        (radii[liquid_or_interface] / RADIUS)[:, None] * interface_radius[liquid_or_interface, None]
    )

    gas = phases == 0
    if np.any(gas):
        wall_r = wall_radius_along_direction(directions[gas])
        s = (radii[gas] - RADIUS) / np.maximum(wall_r - RADIUS, 1.0e-300)
        s = np.clip(s, 0.0, 1.0)
        gas_r = (1.0 - s) * interface_radius[gas] + s * wall_r
        deformed[gas] = directions[gas] * gas_r[:, None]
    return deformed


def phase_edges(edges: np.ndarray, phases: np.ndarray, interface_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edges.size == 0:
        empty = np.empty((0, 2), dtype=int)
        return empty, empty, empty
    gas_edges = edges[np.all(phases[edges] == 0, axis=1)]
    liquid_edges = edges[np.all(phases[edges] == 1, axis=1)]
    return gas_edges, liquid_edges, interface_edges


def render_case(
    *,
    source_dir: Path,
    out_dir: Path,
    mesh: dict[str, np.ndarray],
    closure: str,
    max_frames: int,
    fps: int,
) -> Path:
    rows = json.loads((source_dir / f"flux_{closure}_timeseries.json").read_text())
    t = row_array(rows, "t")
    a2 = row_array(rows, "shape_amplitude_fit")
    a_theory = row_array(rows, "shape_amplitude_theory")
    cpu = row_array(rows, "cpu_time_s")
    cpu_ms = row_array(rows, "cpu_ms_per_substep")
    pressure = row_array(rows, "closure_pressure_pa")
    fheron = row_array(rows, "fheron_pressure_pa")
    ambient_pressure = row_array(rows, "ambient_pressure_pa")
    vol_pct = (row_array(rows, "vol_rel") - 1.0) * 100.0
    local_volume_pct = row_array(rows, "local_volume_rel_max_abs") * 100.0
    density_spread_ppm = (row_array(rows, "nodal_density_rel_max") - row_array(rows, "nodal_density_rel_min")) * 1.0e6
    mass_flux = row_array(rows, "mass_flux_l1_kg_s")
    momentum_flux = row_array(rows, "momentum_flux_l1_n")
    limiter_scale = row_array(rows, "flux_limiter_scale")
    density_limit = float(rows[0].get("density_change_limit", 0.0))
    display_tet_count = row_array(rows, "display_tet_count")
    display_face_count = row_array(rows, "display_internal_face_count")
    display_changed_tets = row_array(rows, "display_changed_tets")
    display_changed_fraction = row_array(rows, "display_changed_fraction")
    flux_scheme = "Rusanov" if float(rows[0].get("flux_scheme_rusanov", 0.0)) > 0.5 else "upwind"
    face_flux_mode = "Lagrangian face" if float(rows[0].get("face_flux_lagrangian", 0.0)) > 0.5 else "cell-average face"

    phases = np.asarray(mesh["phases"], dtype=int)
    interface = np.asarray(mesh["is_interface"], dtype=bool)
    edges = np.asarray(mesh["edges"], dtype=int)
    mesh_interface_edges = np.asarray(mesh["interface_edges"], dtype=int)
    interface_triangles = np.asarray(mesh["interface_triangles"], dtype=int)
    gas_edges, liquid_edges, interface_edges = phase_edges(edges, phases, mesh_interface_edges)

    frame_indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), max(2, int(max_frames))), dtype=int))
    limit = DOMAIN_HALF_WIDTH * 1.03
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)
    time_tick_labels = [f"{value:.3f}" for value in time_ticks]
    cpu_total = float(cpu[-1])
    cpu_avg = float(cpu_ms[-1])

    if closure == "compressible":
        liquid_color = "#1f77b4"
        liquid_face = "#9ecae1"
        gif_name = "flux_compressible_github_twophase_triangulation.gif"
        title = "#11 GitHub two-phase mesh + FHeron EOS"
    else:
        liquid_color = "#ff7f0e"
        liquid_face = "#fdd0a2"
        gif_name = "flux_incompressible_github_twophase_triangulation.gif"
        title = "#12 GitHub two-phase mesh + projection"

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
        points = deform_github_mesh(mesh, float(a2[row_idx]))

        ax3d.cla()
        ax_info.cla()
        ax_amp.cla()
        ax_vol.cla()
        ax_flux.cla()
        ax_flux_twin.cla()

        if gas_edges.size:
            ax3d.add_collection3d(Line3DCollection(points[gas_edges], colors=(0.25, 0.42, 0.65, 0.20), linewidths=0.30))
        if liquid_edges.size:
            ax3d.add_collection3d(Line3DCollection(points[liquid_edges], colors=(0.05, 0.05, 0.05, 0.24), linewidths=0.34))
        if interface_edges.size:
            ax3d.add_collection3d(Line3DCollection(points[interface_edges], colors=(0.85, 0.28, 0.10, 0.72), linewidths=0.72))
        if interface_triangles.size:
            ax3d.add_collection3d(
                Poly3DCollection(
                    points[interface_triangles],
                    facecolor=liquid_face,
                    edgecolor=(0.85, 0.28, 0.10, 0.48),
                    linewidth=0.32,
                    alpha=0.42,
                )
            )

        gas_vertices = phases == 0
        liquid_vertices = phases == 1
        if np.any(gas_vertices):
            ax3d.scatter(points[gas_vertices, 0], points[gas_vertices, 1], points[gas_vertices, 2], s=5, c="#8fb8de", alpha=0.34, edgecolors="none", depthshade=False)
        if np.any(liquid_vertices):
            ax3d.scatter(points[liquid_vertices, 0], points[liquid_vertices, 1], points[liquid_vertices, 2], s=9, c=liquid_color, alpha=0.60, edgecolors="none", depthshade=False)
        if np.any(interface):
            ax3d.scatter(points[interface, 0], points[interface, 1], points[interface, 2], s=15, c="#d62728", alpha=0.94, edgecolors="white", linewidths=0.2, depthshade=False)

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20.0, azim=34.0 + 26.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=10)

        ax_info.axis("off")
        left = (
            f"t = {t[row_idx]:.5f} s\n"
            f"theory a2 = {a_theory[row_idx]:+.4f}\n"
            f"fit a2 = {a2[row_idx]:+.4f}\n"
            f"global dV/V0 = {vol_pct[row_idx]:+.3e} %\n"
            f"max tet |dV|/V = {local_volume_pct[row_idx]:.3e} %\n"
            f"rho spread = {density_spread_ppm[row_idx]:.3f} ppm\n"
            f"mass flux L1 = {mass_flux[row_idx]:.3e} kg/s\n"
            f"mom flux L1 = {momentum_flux[row_idx]:.3e} N\n"
            f"flux limiter = {limiter_scale[row_idx]:.3e}\n"
            f"visual mesh = GitHub two-phase"
        )
        right = (
            f"GitHub verts = {points.shape[0]}, edges = {edges.shape[0]}\n"
            f"gas/liquid/interface = {int(np.sum(gas_vertices))}/{int(np.sum(liquid_vertices))}/{int(np.sum(interface))}\n"
            f"interface tris = {interface_triangles.shape[0]}\n"
            f"solver display tets = {display_tet_count[row_idx]:.0f}\n"
            f"Delaunay flips = {display_changed_tets[row_idx]:.0f} ({100.0 * display_changed_fraction[row_idx]:.1f}%)\n"
            f"internal faces = {display_face_count[row_idx]:.0f}\n"
            f"flux = {flux_scheme}, {face_flux_mode}\n"
            f"Fp = volume-gradient Bt\n"
            f"rho limit = {100.0 * density_limit:.2f} %\n"
            f"p_gas = {ambient_pressure[row_idx]:.0f} Pa\n"
            f"p_gauge = {pressure[row_idx]:.3f} Pa\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa\n"
            f"CPU = {cpu[row_idx]:.2f}/{cpu_total:.1f} s\n"
            f"CPU/step = {cpu_ms[row_idx]:.2f} ms"
        )
        ax_info.text(0.02, 0.98, left, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.6)
        ax_info.text(0.52, 0.98, right, transform=ax_info.transAxes, va="top", ha="left", family="monospace", fontsize=7.6)

        pad = max(0.12 * float(np.ptp(a_theory)), 0.004)
        ax_amp.plot(t, a_theory, "k--", lw=1.6, label="Rayleigh theory")
        ax_amp.plot(t, a2, color=liquid_color, lw=1.8, label="Ftot + flux")
        ax_amp.plot(t[row_idx], a2[row_idx], "o", color="#d62728", ms=5.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_xticklabels(time_tick_labels)
        ax_amp.set_ylim(float(min(np.min(a_theory), np.min(a2)) - pad), float(max(np.max(a_theory), np.max(a2)) + pad))
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        ax_vol.plot(t, vol_pct, color=liquid_color, lw=1.8, label="global dV/V0")
        ax_vol.plot(t, local_volume_pct, color="0.3", lw=1.2, ls=":", label="max local tet")
        ax_vol.plot(t[row_idx], vol_pct[row_idx], "o", color="#d62728", ms=5.0)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        ax_vol.set_xticks(time_ticks)
        ax_vol.set_xticklabels(time_tick_labels)
        ymin = min(float(np.min(vol_pct)), float(np.min(local_volume_pct)))
        ymax = max(float(np.max(vol_pct)), float(np.max(local_volume_pct)))
        ypad = max(0.12 * (ymax - ymin), 1.0e-8)
        ax_vol.set_ylim(ymin - ypad, ymax + ypad)
        ax_vol.set_ylabel("percent")
        ax_vol.set_title("Global and local volume error")
        ax_vol.grid(True, alpha=0.25)
        ax_vol.legend(frameon=False, fontsize=8, loc="upper right")

        mass_flux_plot = np.maximum(mass_flux, 1.0e-30)
        momentum_flux_plot = np.maximum(momentum_flux, 1.0e-30)
        ax_flux.plot(t, mass_flux_plot, color=liquid_color, lw=1.8, label="mass flux L1")
        ax_flux_twin.plot(t, momentum_flux_plot, color="0.25", lw=1.2, ls=":", label="momentum flux L1")
        ax_flux.plot(t[row_idx], max(mass_flux[row_idx], 1.0e-30), "o", color="#d62728", ms=5.0)
        ax_flux.set_yscale("log")
        ax_flux_twin.set_yscale("log")
        mass_flux_max = max(float(np.max(mass_flux_plot)), 1.0e-30)
        momentum_flux_max = max(float(np.max(momentum_flux_plot)), 1.0e-30)
        ax_flux.set_ylim(max(mass_flux_max * 1.0e-6, 1.0e-30), mass_flux_max * 5.0)
        ax_flux_twin.set_ylim(max(momentum_flux_max * 1.0e-6, 1.0e-30), momentum_flux_max * 5.0)
        ax_flux.set_xlim(float(t[0]), float(t[-1]))
        ax_flux.set_xticks(time_ticks)
        ax_flux.set_xticklabels(time_tick_labels)
        ax_flux.set_xlabel("t [s]")
        ax_flux.set_ylabel("kg/s")
        ax_flux_twin.set_ylabel("N")
        ax_flux.set_title("Internal face fluxes")
        ax_flux.grid(True, alpha=0.25)
        lines, labels = ax_flux.get_legend_handles_labels()
        lines2, labels2 = ax_flux_twin.get_legend_handles_labels()
        ax_flux.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper right")

        fig.suptitle(
            f"{closure.capitalize()} GitHub-mesh flux-aware run, CPU {cpu_total:.1f} s ({cpu_avg:.1f} ms/step)",
            y=0.985,
        )

    fig.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.08)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / gif_name
    animation.save(path, writer=PillowWriter(fps=max(1, int(fps))))
    draw(len(frame_indices) // 2)
    fig.savefig(out_dir / f"{path.stem}.midframe.png", dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render stable #11/#12 results on the GitHub two-phase triangulation.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "out" / "sphere_fheron_case11_12_active_retopology_full")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "sphere_fheron_case11_12_github_twophase_triangulation")
    parser.add_argument("--closure", choices=("compressible", "incompressible", "both"), default="both")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    mesh = load_github_twophase_mesh()
    phases = np.asarray(mesh["phases"], dtype=int)
    print("GitHub droplet_in_box_3d mesh source")
    print(f"vertices={mesh['points'].shape[0]}, edges={mesh['edges'].shape[0]}")
    print(f"phase0_gas={int(np.sum(phases == 0))}, phase1_liquid={int(np.sum(phases == 1))}, interface={int(np.sum(mesh['is_interface']))}")
    print(f"interface_triangles={mesh['interface_triangles'].shape[0]}")

    closures = ("compressible", "incompressible") if args.closure == "both" else (args.closure,)
    for closure in closures:
        path = render_case(
            source_dir=args.source_dir,
            out_dir=args.out_dir,
            mesh=mesh,
            closure=closure,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
