from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GITHUB_REPO = Path("/private/tmp/ddgclib_latest_for_oscillating_droplet")
if str(GITHUB_REPO) not in sys.path:
    sys.path.insert(0, str(GITHUB_REPO))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402


def _angular_sort_3d_compat(vertices):
    verts = list(vertices)
    if len(verts) < 3:
        return verts
    pts = np.asarray([v.x_a[:3] for v in verts], dtype=float)
    center = np.mean(pts, axis=0)
    shifted = pts - center[None, :]
    try:
        _u, _s, vh = np.linalg.svd(shifted, full_matrices=False)
        normal = vh[-1]
    except np.linalg.LinAlgError:
        normal = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(normal) <= 1.0e-30:
        normal = np.array([0.0, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    ref = shifted[np.argmax(np.linalg.norm(shifted, axis=1))]
    ref = ref - np.dot(ref, normal) * normal
    if np.linalg.norm(ref) <= 1.0e-30:
        ref = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ref) <= 1.0e-30:
            ref = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    ref = ref / np.linalg.norm(ref)
    tangent = np.cross(normal, ref)
    angles = np.arctan2(shifted @ tangent, shifted @ ref)
    order = np.argsort(angles)
    return [verts[int(i)] for i in order]


dual_cell_module = types.ModuleType("hyperct.ddg._dual_cell")
dual_cell_module._angular_sort_3d = _angular_sort_3d_compat
sys.modules.setdefault("hyperct.ddg._dual_cell", dual_cell_module)

try:
    import hyperct.ddg as hyperct_ddg  # noqa: E402

    _batch_e_star_original = hyperct_ddg.batch_e_star

    def _orient_and_sum_edge_areas(edge_areas, HC):
        vertices_by_id = {id(v): v for v in HC.V}
        summed = {}
        for vid, nb_map in edge_areas.items():
            v = vertices_by_id.get(vid)
            if v is None:
                continue
            summed[vid] = {}
            for nbid, raw in nb_map.items():
                nb = vertices_by_id.get(nbid)
                arr = np.asarray(raw, dtype=float)
                if arr.ndim == 1:
                    summed[vid][nbid] = arr[:3]
                    continue
                if arr.size == 0:
                    summed[vid][nbid] = np.zeros(3)
                    continue
                vec_to_i = None
                if nb is not None:
                    vc_12_pos = 0.5 * (nb.x_a - v.x_a) + v.x_a
                    try:
                        vc_12 = HC.Vd[tuple(vc_12_pos)]
                        vec_to_i = v.x_a[:3] - vc_12.x_a[:3]
                    except Exception:
                        vec_to_i = v.x_a[:3] - 0.5 * (v.x_a[:3] + nb.x_a[:3])
                A_ij = np.zeros(3)
                for A_ijk in arr.reshape((-1, 3)):
                    A = np.asarray(A_ijk, dtype=float).copy()
                    if vec_to_i is not None and np.dot(A, vec_to_i) > 0:
                        A = -A
                    A_ij += A
                summed[vid][nbid] = A_ij
        return summed

    def _batch_e_star_compat(vertices, HC, dim=3, backend=None, orient=None, compute_volumes=False):
        result = _batch_e_star_original(
            vertices,
            HC,
            dim=dim,
            backend=backend,
            compute_volumes=compute_volumes,
        )
        if not orient:
            return result
        if compute_volumes:
            edge_areas, failed, vols = result
            return _orient_and_sum_edge_areas(edge_areas, HC), failed, vols
        edge_areas, failed = result
        return _orient_and_sum_edge_areas(edge_areas, HC), failed

    hyperct_ddg.batch_e_star = _batch_e_star_compat
except Exception:
    pass

from cases_dynamic.oscillating_droplet.src._setup import setup_oscillating_droplet  # noqa: E402
from cases_dynamic.oscillating_droplet.src._analytical import rayleigh_frequency  # noqa: E402
from ddgclib.dynamic_integrators import symplectic_euler  # noqa: E402


def vertex_phase(v) -> int:
    return int(getattr(v, "phase", 0))


def ordered_vertices(HC) -> list:
    return list(HC.V)


def snapshot_from_hc(HC, dim: int) -> dict[str, np.ndarray]:
    verts = ordered_vertices(HC)
    index = {v: i for i, v in enumerate(verts)}
    points = np.asarray([v.x_a[:dim] for v in verts], dtype=float)
    if dim == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    phases = np.asarray([vertex_phase(v) for v in verts], dtype=int)
    is_interface = np.asarray([bool(getattr(v, "is_interface", False)) for v in verts], dtype=bool)
    edges: set[tuple[int, int]] = set()
    for v in verts:
        i = index[v]
        for nb in getattr(v, "nn", []):
            j = index.get(nb)
            if j is not None:
                edges.add(tuple(sorted((i, j))))
    return {
        "points": points,
        "phases": phases,
        "is_interface": is_interface,
        "edges": np.asarray(sorted(edges), dtype=int).reshape((-1, 2)),
    }


def fit_cos2_amplitude(points: np.ndarray, interface_mask: np.ndarray, radius: float) -> float:
    interface = points[interface_mask]
    if interface.size == 0:
        return 0.0
    r = np.linalg.norm(interface, axis=1)
    theta = np.arctan2(
        np.sqrt(interface[:, 0] * interface[:, 0] + interface[:, 1] * interface[:, 1]),
        interface[:, 2],
    )
    basis = np.cos(2.0 * theta)
    return float(np.dot((r / float(radius)) - 1.0, basis) / max(np.dot(basis, basis), 1.0e-300))


def run_case(
    *,
    out_dir: Path,
    closure_label: str,
    R0: float,
    epsilon: float,
    gamma: float,
    rho_d: float,
    rho_o: float,
    mu_d: float,
    mu_o: float,
    K_d: float,
    K_o: float,
    t_final: float,
    dt: float,
    refinement_outer: int,
    refinement_droplet: int,
    enable_retopology: bool,
) -> tuple[list[dict[str, float]], list[dict[str, np.ndarray]], float]:
    HC, bV, _mps, bc_set, dudt_fn, retopo_fn, _params = setup_oscillating_droplet(
        dim=3,
        R0=R0,
        epsilon=epsilon,
        l=2,
        rho_d=rho_d,
        rho_o=rho_o,
        mu_d=mu_d,
        mu_o=mu_o,
        gamma=gamma,
        K_d=K_d,
        K_o=K_o,
        L_domain=5.0 * R0,
        refinement_outer=refinement_outer,
        refinement_droplet=refinement_droplet,
    )
    omega = rayleigh_frequency(2, gamma, rho_d, R0, dim=3, rho_outer=rho_o)
    n_steps = int(math.ceil(float(t_final) / float(dt)))
    record_every = max(1, n_steps // 80)
    rows: list[dict[str, float]] = []
    snapshots: list[dict[str, np.ndarray]] = []
    cpu_start = time.process_time()

    def record(step: int, t: float) -> None:
        snap = snapshot_from_hc(HC, 3)
        a2 = fit_cos2_amplitude(snap["points"], snap["is_interface"], R0)
        rows.append(
            {
                "step": float(step),
                "t": float(t),
                "closure": closure_label,
                "shape_amplitude_fit": float(a2),
                "shape_amplitude_theory": float(epsilon * math.cos(omega * t)),
                "rayleigh_omega_rad_s": float(omega),
                "cpu_time_s": float(time.process_time() - cpu_start),
                "n_vertices": float(snap["points"].shape[0]),
                "n_edges": float(snap["edges"].shape[0]),
                "n_interface": float(np.sum(snap["is_interface"])),
                "n_liquid": float(np.sum(snap["phases"] == 1)),
                "n_gas": float(np.sum(snap["phases"] == 0)),
                "retopology_active": float(bool(enable_retopology)),
            }
        )
        snapshots.append(snap)

    record(0, 0.0)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if step % record_every == 0 or step == n_steps - 1:
            record(step + 1, t)

    symplectic_euler(
        HC,
        bV,
        dudt_fn,
        dt=float(dt),
        n_steps=n_steps,
        dim=3,
        bc_set=bc_set,
        callback=callback,
        retopologize_fn=retopo_fn if enable_retopology else False,
    )
    return rows, snapshots, omega


def write_github_preview_gif(
    rows: list[dict[str, float]],
    snapshots: list[dict[str, np.ndarray]],
    out_dir: Path,
    *,
    closure_label: str,
    fps: int,
) -> Path:
    t = np.asarray([row["t"] for row in rows], dtype=float)
    a_fit = np.asarray([row["shape_amplitude_fit"] for row in rows], dtype=float)
    a_theory = np.asarray([row["shape_amplitude_theory"] for row in rows], dtype=float)
    cpu = np.asarray([row["cpu_time_s"] for row in rows], dtype=float)
    retopo_active = bool(rows and rows[0].get("retopology_active", 0.0))
    if closure_label == "compressible":
        liquid_color = "#1f77b4"
        gif_name = "github_twophase_compressible_preview.gif"
        title = "#11 GitHub-style two-phase water-air mesh"
    else:
        liquid_color = "#ff7f0e"
        gif_name = "github_twophase_incompressible_preview.gif"
        title = "#12 GitHub-style two-phase mesh, projection comparison"

    all_points = np.vstack([snap["points"] for snap in snapshots])
    limit = 1.05 * float(np.max(np.linalg.norm(all_points, axis=1)))
    frame_indices = np.arange(len(snapshots), dtype=int)
    time_ticks = np.linspace(float(t[0]), float(t[-1]), 5)

    fig = plt.figure(figsize=(5.2, 7.3))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.3, 1.0), hspace=0.16)
    ax3d = fig.add_subplot(grid[0], projection="3d")
    ax_amp = fig.add_subplot(grid[1])

    def draw(frame_pos: int) -> None:
        idx = int(frame_indices[frame_pos])
        snap = snapshots[idx]
        points = snap["points"]
        phases = snap["phases"]
        interface = snap["is_interface"]
        edges = snap["edges"]

        ax3d.cla()
        ax_amp.cla()
        gas_vertices = phases == 0
        liquid_vertices = phases == 1
        interface_vertices = interface
        if edges.size:
            edge_phase = np.maximum(phases[edges[:, 0]], phases[edges[:, 1]])
            gas_edges = edges[edge_phase == 0]
            liquid_edges = edges[edge_phase == 1]
            interface_edges = edges[np.any(interface[edges], axis=1)]
            if gas_edges.size:
                ax3d.add_collection3d(Line3DCollection(points[gas_edges], colors=(0.25, 0.42, 0.65, 0.18), linewidths=0.28))
            if liquid_edges.size:
                ax3d.add_collection3d(Line3DCollection(points[liquid_edges], colors=(0.05, 0.05, 0.05, 0.22), linewidths=0.32))
            if interface_edges.size:
                ax3d.add_collection3d(Line3DCollection(points[interface_edges], colors=(0.85, 0.28, 0.10, 0.55), linewidths=0.55))

        if gas_vertices.any():
            ax3d.scatter(points[gas_vertices, 0], points[gas_vertices, 1], points[gas_vertices, 2], s=5, c="#8fb8de", alpha=0.34, edgecolors="none", depthshade=False)
        if liquid_vertices.any():
            ax3d.scatter(points[liquid_vertices, 0], points[liquid_vertices, 1], points[liquid_vertices, 2], s=7, c=liquid_color, alpha=0.58, edgecolors="none", depthshade=False)
        if interface_vertices.any():
            ax3d.scatter(points[interface_vertices, 0], points[interface_vertices, 1], points[interface_vertices, 2], s=13, c="#d62728", alpha=0.92, edgecolors="white", linewidths=0.2, depthshade=False)

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20.0, azim=34.0 + 26.0 * float(t[idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.set_title(title, fontsize=8.8, pad=4)
        ax3d.text2D(
            0.02,
            0.97,
            (
                f"t = {t[idx]:.5f} s\n"
                f"phase 1 liquid verts = {int(rows[idx]['n_liquid'])}\n"
                f"phase 0 gas verts = {int(rows[idx]['n_gas'])}\n"
                f"interface verts = {int(rows[idx]['n_interface'])}\n"
                f"retopology = {'active' if retopo_active else 'off'}\n"
                f"CPU = {cpu[idx]:.1f}/{cpu[-1]:.1f} s"
            ),
            transform=ax3d.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            family="monospace",
        )

        pad = max(0.1 * float(np.ptp(a_theory)), 0.004)
        ax_amp.plot(t, a_theory, "k--", lw=1.3, label="Rayleigh theory")
        ax_amp.plot(t, a_fit, color=liquid_color, lw=1.6, label="GitHub two-phase")
        ax_amp.plot(t[idx], a_fit[idx], "o", color="#d62728", ms=4.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(float(min(np.min(a_theory), np.min(a_fit)) - pad), float(max(np.max(a_theory), np.max(a_fit)) + pad))
        ax_amp.set_xticks(time_ticks)
        ax_amp.set_xlabel("t [s]", fontsize=8)
        ax_amp.set_ylabel("a2", fontsize=8)
        ax_amp.set_title("Shape amplitude", fontsize=8)
        ax_amp.tick_params(labelsize=7)
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=7, loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.07)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    path = out_dir / gif_name
    animation.save(path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest GitHub two-phase oscillating droplet preview.")
    parser.add_argument("--closure-label", choices=("compressible", "incompressible"), default="compressible")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "github_twophase_oscillating_preview")
    parser.add_argument("--t-final", type=float, default=0.016)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--refinement-outer", type=int, default=1)
    parser.add_argument("--refinement-droplet", type=int, default=1)
    parser.add_argument("--enable-retopology", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    R0 = 1.0e-3
    gamma = 0.072
    rho_d = 1000.0
    rho_o = 1.225
    epsilon = 0.05
    c_s = 1.0
    rows, snapshots, omega = run_case(
        out_dir=args.out_dir,
        closure_label=args.closure_label,
        R0=R0,
        epsilon=epsilon,
        gamma=gamma,
        rho_d=rho_d,
        rho_o=rho_o,
        mu_d=0.001,
        mu_o=1.81e-5,
        K_d=rho_d * c_s * c_s,
        K_o=rho_o * c_s * c_s,
        t_final=args.t_final,
        dt=args.dt,
        refinement_outer=args.refinement_outer,
        refinement_droplet=args.refinement_droplet,
        enable_retopology=args.enable_retopology,
    )
    (args.out_dir / f"{args.closure_label}_rows.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(
        args.out_dir / f"{args.closure_label}_snapshots.npz",
        points=np.asarray([snap["points"] for snap in snapshots], dtype=object),
        phases=np.asarray([snap["phases"] for snap in snapshots], dtype=object),
        interface=np.asarray([snap["is_interface"] for snap in snapshots], dtype=object),
        edges=np.asarray([snap["edges"] for snap in snapshots], dtype=object),
    )
    gif = write_github_preview_gif(rows, snapshots, args.out_dir, closure_label=args.closure_label, fps=args.fps)
    print(f"omega={omega:.6g}")
    print(gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
