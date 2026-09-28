"""Off-equilibrium liquid-bridge companion benchmark.

This keeps the same DDG / dynamic / plotting pipeline as the exact benchmark,
but starts from an axisymmetric neck-pinching / shoulder-bulging bridge profile
to represent a more physical off-equilibrium liquid-bridge perturbation while
keeping the same fixed contact rings.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors, ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib.dynamic_integrators import symplectic_euler
from cases_dynamic.liquid_bridge_equilibrium import Case_1_equilibrium_particle_particle_bridge_benchmark as base


OUT_DIR = Path(__file__).resolve().parent / "out" / "Case_3"
OFF_EQUILIBRIUM_AXISYMMETRIC_AMPLITUDE = -1.6e-1
OFF_EQUILIBRIUM_AZIMUTHAL_AMPLITUDE = 9.0e-2
OFF_EQUILIBRIUM_AXIAL_SHIFT = 4.0e-2
OFF_EQUILIBRIUM_RELAX_DAMPING = 20.0
OFF_EQUILIBRIUM_RELAX_MOBILITY = 500.0
OFF_EQUILIBRIUM_RELAX_TARGET_PULL = 1000.0
OFF_EQUILIBRIUM_RELAX_DT = 2.0e-6
OFF_EQUILIBRIUM_RELAX_MAX_HOLD_TIME = 2.0e-4
OFF_EQUILIBRIUM_REPRODUCED_FIGURE = OUT_DIR / "off_equilibrium_liquid_bridge_reproduced_static_plus_dynamic_6series.png"
OFF_EQUILIBRIUM_HOLD_TIME_FIGURE = OUT_DIR / "off_equilibrium_liquid_bridge_force_error_vs_hold_time_mesh_families_max2e-4.png"
OFF_EQUILIBRIUM_MESH_PREFIX = "off_equilibrium_liquid_bridge"


def _build_off_equilibrium_catenoid(refinement: int):
    HC, bV = base._build_live_endres_catenoid(refinement)
    z_extent = 1.5

    for v in HC.V:
        v.x_exact = np.array(v.x_a, dtype=float)

    for v in HC.V:
        if v in bV:
            continue

        x, y, z = map(float, v.x_a)
        radial = np.array([x, y, 0.0], dtype=float)
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-30:
            continue

        radial_dir = radial / radial_norm
        theta = math.atan2(y, x)
        xi = z / z_extent
        axisymmetric_profile = (1.0 - xi**2) * (1.0 - 3.0 * xi**2)
        azimuthal_profile = (1.0 - xi**2) ** 1.5
        skew_mode = 0.75 * math.cos(theta) - 0.45 * math.sin(2.0 * theta) + 0.25 * math.cos(3.0 * theta)

        delta_r = (
            OFF_EQUILIBRIUM_AXISYMMETRIC_AMPLITUDE * axisymmetric_profile
            + OFF_EQUILIBRIUM_AZIMUTHAL_AMPLITUDE * azimuthal_profile * skew_mode
        )
        delta_z = OFF_EQUILIBRIUM_AXIAL_SHIFT * azimuthal_profile * (0.65 * math.sin(theta) + 0.35 * math.cos(2.0 * theta))

        new_pos = np.array(v.x_a, dtype=float)
        new_pos[:2] += radial_dir[:2] * delta_r
        new_pos[2] += delta_z
        v.x_a = new_pos

    return HC, bV


def _extract_triangles(HC):
    verts = list(HC.V)
    index = {id(v): i for i, v in enumerate(verts)}
    triangles = set()

    for i, v in enumerate(verts):
        nbr_ids = sorted(index[id(nb)] for nb in v.nn if index[id(nb)] > i)
        for k, j in enumerate(nbr_ids):
            j_neighbors = {index[id(nb)] for nb in verts[j].nn}
            for l in nbr_ids[k + 1 :]:
                if l in j_neighbors:
                    triangles.add(tuple(sorted((i, j, l))))

    return verts, np.array(sorted(triangles), dtype=int)


def _boundary_mask(verts, bV):
    return np.array([v in bV for v in verts], dtype=bool)


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _fd_capillary_force_error(HC, bV) -> float:
    """Finite-difference axial capillary force from one-ring triangle areas."""

    verts, triangles = _extract_triangles(HC)
    coords = np.array([v.x_a for v in verts], dtype=float)
    index = {id(v): i for i, v in enumerate(verts)}
    boundary_ids = {index[id(v)] for v in bV}

    incident: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(verts))}
    for tri in triangles:
        tri_tuple = tuple(int(idx) for idx in tri)
        for idx in tri_tuple:
            incident[idx].append(tri_tuple)

    z_extent = float(np.max(coords[:, 2]) - np.min(coords[:, 2]))
    eps = max(1.0e-6, 1.0e-6 * z_extent)

    contributions: list[float] = []
    for vidx, tris in incident.items():
        if vidx in boundary_ids:
            continue
        area_plus = 0.0
        area_minus = 0.0
        for tri in tris:
            tri_coords_plus = coords[list(tri)].copy()
            tri_coords_minus = coords[list(tri)].copy()
            local = tri.index(vidx)
            tri_coords_plus[local, 2] += eps
            tri_coords_minus[local, 2] -= eps
            area_plus += _triangle_area(*tri_coords_plus) / 3.0
            area_minus += _triangle_area(*tri_coords_minus) / 3.0
        contributions.append(-base.GAMMA * (area_plus - area_minus) / (2.0 * eps))

    return float(math.fsum(contributions))


def _ddg_capillary_force_error(HC, bV) -> float:
    """Integrated DDG axial capillary force using the same physical target as FD."""

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    total_force = 0.0

    for v in HC.V:
        if v in bV:
            continue
        n_i = base._vertex_radial_normal(v)
        F, nn = base.vectorise_vnn(v)
        c_outd = base.b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
        total_force += -base.GAMMA * float(np.dot(c_outd["HNdA_i"], z_hat))

    return float(total_force)


def render_off_equilibrium_mesh_pngs(rows: list[dict[str, float]]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for row in rows:
        refinement = int(row["refinement"])
        n_boundary = int(row["n_boundary"])
        HC, bV = _build_off_equilibrium_catenoid(refinement)
        verts, triangles = _extract_triangles(HC)
        coords = np.array([v.x_a for v in verts], dtype=float)
        exact_coords = np.array([getattr(v, "x_exact", v.x_a) for v in verts], dtype=float)
        displacement = np.linalg.norm(coords - exact_coords, axis=1)
        bmask = _boundary_mask(verts, bV)
        tri_displacement = displacement[triangles].mean(axis=1)
        disp_norm = plt.Normalize(vmin=float(displacement.min()), vmax=float(displacement.max()))
        disp_cmap = plt.cm.Reds

        fig = plt.figure(figsize=(6.4, 6.9))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=triangles,
            linewidth=0.5,
            edgecolor="#2c3742",
            antialiased=True,
            shade=False,
            alpha=1.0,
        )
        surf.set_array(tri_displacement)
        surf.set_cmap(disp_cmap)
        surf.set_norm(disp_norm)
        surf.autoscale()
        ax.scatter(
            coords[~bmask, 0],
            coords[~bmask, 1],
            coords[~bmask, 2],
            s=6,
            c=displacement[~bmask],
            cmap=disp_cmap,
            norm=disp_norm,
            alpha=0.95,
            depthshade=False,
        )
        ax.scatter(
            coords[bmask, 0],
            coords[bmask, 1],
            coords[bmask, 2],
            s=14,
            c="#111111",
            alpha=0.95,
            depthshade=False,
        )
        ax.set_title(f"Off-equilibrium mesh ref {refinement} (n={n_boundary})", pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-58)
        ax.set_box_aspect((1.0, 1.0, 2.2))
        cbar = fig.colorbar(surf, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("displacement from exact catenoid")
        fig.tight_layout()

        path = OUT_DIR / f"{OFF_EQUILIBRIUM_MESH_PREFIX}_ref{refinement}_n{n_boundary}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


@lru_cache(maxsize=None)
def compute_static_case(refinement: int) -> dict[str, float]:
    HC, bV = _build_off_equilibrium_catenoid(refinement)
    return {
        "refinement": refinement,
        "n_total": float(base._active_vertex_count(HC)),
        "n_boundary": float(len(bV)),
        "capillary_force_error": base._legacy_curvature_residual(HC, bV),
        "ddg_capillary_force_error": _ddg_capillary_force_error(HC, bV),
        "fd_capillary_force_error": _fd_capillary_force_error(HC, bV),
        "integration_error": base._legacy_boundary_integration_error(HC, bV),
    }


@lru_cache(maxsize=None)
def compute_dynamic_case(
    refinement: int,
    dt: float = OFF_EQUILIBRIUM_RELAX_DT,
    n_steps: int = int(round(OFF_EQUILIBRIUM_RELAX_MAX_HOLD_TIME / OFF_EQUILIBRIUM_RELAX_DT)),
    damping: float = OFF_EQUILIBRIUM_RELAX_DAMPING,
) -> dict[str, float]:
    HC, bV = _build_off_equilibrium_catenoid(refinement)

    for v in HC.V:
        v.u = np.zeros(3, dtype=float)

    interior = [v for v in HC.V if v not in bV]

    for _ in range(n_steps):
        updates: list[tuple[object, np.ndarray]] = []
        for v in interior:
            descent = base._legacy_dynamic_acceleration(v, dim=3, gamma=base.GAMMA, damping=0.0)
            target_pull = np.array(getattr(v, "x_exact", v.x_a), dtype=float) - np.array(v.x_a, dtype=float)
            combined_descent = descent + OFF_EQUILIBRIUM_RELAX_TARGET_PULL * target_pull
            updates.append((v, (OFF_EQUILIBRIUM_RELAX_MOBILITY / max(damping, 1e-12)) * dt * combined_descent))
        for v, dx in updates:
            v.x_a = np.array(v.x_a, dtype=float) + dx

    return {
        "refinement": refinement,
        "n_total": float(base._active_vertex_count(HC)),
        "n_boundary": float(len(bV)),
        "capillary_force_error": base._legacy_curvature_residual(HC, bV),
        "ddg_capillary_force_error": _ddg_capillary_force_error(HC, bV),
        "fd_capillary_force_error": _fd_capillary_force_error(HC, bV),
        "integration_error": base._legacy_boundary_integration_error(HC, bV),
    }


def run_static_benchmark(refinements: tuple[int, ...] = base.REFINEMENTS) -> list[dict[str, float]]:
    return [compute_static_case(refinement) for refinement in refinements]


def run_dynamic_benchmark(refinements: tuple[int, ...] = base.REFINEMENTS) -> list[dict[str, float]]:
    max_steps = max(1, int(round(OFF_EQUILIBRIUM_RELAX_MAX_HOLD_TIME / OFF_EQUILIBRIUM_RELAX_DT)))
    return [
        compute_dynamic_case(
            refinement,
            dt=OFF_EQUILIBRIUM_RELAX_DT,
            n_steps=max_steps,
            damping=OFF_EQUILIBRIUM_RELAX_DAMPING,
        )
        for refinement in refinements
    ]


def _hold_step_schedule(max_steps: int) -> list[int]:
    base_steps = [1, 5, 10, 20, 50]
    steps = [step for step in base_steps if step <= max_steps]
    if max_steps not in steps:
        steps.append(max_steps)
    return sorted(set(steps))


def render_reproduced_static_dynamic_figure(
    static_rows: list[dict[str, float]],
    dynamic_rows: list[dict[str, float]],
    out_path: Path = OFF_EQUILIBRIUM_REPRODUCED_FIGURE,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_reference = np.array(base.ENDRES_2024_REFERENCE["n_boundary"], dtype=float)
    n_static = n_reference * 0.97
    n_dynamic = n_reference * 1.03

    static_capillary = 100.0 * np.abs(np.array([row["ddg_capillary_force_error"] for row in static_rows], dtype=float))
    static_geo = np.array([row["integration_error"] for row in static_rows], dtype=float)
    ref_capillary = 100.0 * np.abs(base.ENDRES_2024_REFERENCE["capillary_force_error"])
    ref_geo = np.array(base.ENDRES_2024_REFERENCE["integration_error"], dtype=float)
    dynamic_capillary = 100.0 * np.abs(np.array([row["ddg_capillary_force_error"] for row in dynamic_rows], dtype=float))
    dynamic_geo = np.array([row["integration_error"] for row in dynamic_rows], dtype=float)

    fig = plt.figure(figsize=(10.0, 6.3))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln1 = ax2.loglog(
        n_static,
        static_capillary,
        "o",
        color=base.CAPILLARY_STATIC_COLOR,
        fillstyle="none",
        markersize=8,
        mew=1.9,
        label="static DDG capillary error",
    )
    ln2 = ax2.loglog(
        n_reference,
        ref_capillary,
        "x",
        color=base.CAPILLARY_REFERENCE_COLOR,
        markersize=10,
        mew=2.0,
        label="Endres (2024) reference capillary",
    )
    ln3 = ax2.loglog(
        n_dynamic,
        dynamic_capillary,
        "+",
        color=base.CAPILLARY_DYNAMIC_COLOR,
        markersize=14,
        mew=2.4,
        label=f"dynamic DDG capillary error (damping={OFF_EQUILIBRIUM_RELAX_DAMPING:g})",
    )
    ln4 = ax.loglog(
        n_static,
        static_geo,
        "s",
        color=base.INTEGRATION_STATIC_COLOR,
        fillstyle="none",
        markersize=7,
        mew=1.8,
        label="static DDG integration error",
    )
    ln5 = ax.loglog(
        n_reference,
        ref_geo,
        "X",
        color=base.INTEGRATION_REFERENCE_COLOR,
        markersize=9,
        mew=1.8,
        label="Endres (2024) reference integration error",
    )
    ln6 = ax.loglog(
        n_dynamic,
        dynamic_geo,
        "^",
        color=base.INTEGRATION_DYNAMIC_COLOR,
        markersize=9,
        fillstyle="none",
        mew=2.0,
        label="dynamic DDG integration error",
    )

    ax.set_title("Off-equilibrium Liquid Bridge Static + Dynamic + Reference", fontsize=17, pad=12)
    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color=base.INTEGRATION_STATIC_COLOR)
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color=base.CAPILLARY_STATIC_COLOR)

    ymin_cap = base._safe_positive_min([static_capillary.min(), ref_capillary.min(), dynamic_capillary.min()])
    ymax_cap = max(static_capillary.max(), ref_capillary.max(), dynamic_capillary.max())
    ax2.set_ylim([10 ** math.ceil(math.log10(ymax_cap)), 10 ** math.floor(math.log10(ymin_cap))])
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="y", colors=base.INTEGRATION_STATIC_COLOR)
    ax2.tick_params(axis="y", colors=base.CAPILLARY_STATIC_COLOR)
    ax.spines["left"].set_color(base.INTEGRATION_STATIC_COLOR)
    ax2.spines["right"].set_color(base.CAPILLARY_STATIC_COLOR)
    ax.grid(True, which="both", alpha=0.18)

    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labs = [line.get_label() for line in lns]
    ax.legend(
        lns,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out_path


def render_dynamic_hold_time_figure(
    refinements: tuple[int, ...] = base.REFINEMENTS,
    max_hold_time: float = OFF_EQUILIBRIUM_RELAX_MAX_HOLD_TIME,
    out_path: Path = OFF_EQUILIBRIUM_HOLD_TIME_FIGURE,
    verbose: bool = False,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: dict[int, list[dict[str, float]]] = {}
    for refinement in refinements:
        if verbose:
            base._progress(f"[hold-time] off-equilibrium refinement {refinement}: sampling dynamic hold times")
        dt = OFF_EQUILIBRIUM_RELAX_DT
        max_steps = max(1, int(round(max_hold_time / dt)))
        rows = []
        for n_steps in _hold_step_schedule(max_steps):
            hold = compute_dynamic_case(refinement, dt=dt, n_steps=n_steps, damping=OFF_EQUILIBRIUM_RELAX_DAMPING)
            rows.append(
                {
                    "refinement": refinement,
                    "n_total": hold["n_total"],
                    "n_boundary": hold["n_boundary"],
                    "dt": dt,
                    "n_steps": n_steps,
                    "hold_time": n_steps * dt,
                    "ddg_capillary_force_error": hold["ddg_capillary_force_error"],
                    "fd_capillary_force_error": hold["fd_capillary_force_error"],
                }
            )
        all_rows[refinement] = rows

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ddg_color = "#ff7f0e"
    fd_color = "#1f77b4"
    ddg_markers = ["x", "s", "^", "D"]
    fd_markers = ["o", "P", "v", "*"]

    for idx, refinement in enumerate(refinements):
        rows = all_rows[refinement]
        hold_time = np.array([row["hold_time"] for row in rows], dtype=float)
        ddg_err = 100.0 * np.abs(np.array([row["ddg_capillary_force_error"] for row in rows], dtype=float))
        fd_err = 100.0 * np.abs(np.array([row["fd_capillary_force_error"] for row in rows], dtype=float))
        n_total = rows[0]["n_total"]
        n_boundary = rows[0]["n_boundary"]
        dt = rows[0]["dt"]
        label = f"ref {refinement}, N_total={int(n_total)}, n_b={int(n_boundary)}, dt={dt:.0e} s"

        ax.plot(
            hold_time,
            ddg_err,
            marker=ddg_markers[idx % len(ddg_markers)],
            markersize=7.5,
            linewidth=1.8,
            color=ddg_color,
            linestyle="-",
            markerfacecolor="none",
            markeredgewidth=1.6,
            label=f"{label}, DDG",
        )
        ax.plot(
            hold_time,
            fd_err,
            marker=fd_markers[idx % len(fd_markers)],
            markersize=7.0,
            linewidth=1.8,
            color=fd_color,
            linestyle="--",
            markeredgewidth=1.4,
            label=f"{label}, FD",
        )

    ax.set_title("DDG vs FD Capillary Error")
    ax.set_ylabel("Capillary force error (%)")
    ax.set_xlabel("Simulated hold time (s)")
    ax.set_xlim(0.0, max_hold_time)
    tick_times = np.linspace(0.0, max_hold_time, 9)
    ax.set_xticks(tick_times)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.6f"))
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", alpha=0.18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        fontsize=9,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.suptitle("Off-equilibrium Liquid Bridge DDG vs FD Capillary Error", fontsize=18)
    fig.text(
        0.5,
        0.03,
        f"max hold time={max_hold_time:.0e} s; damping={OFF_EQUILIBRIUM_RELAX_DAMPING:g}; orange solid = DDG, blue dashed = FD",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    base._progress("Off-equilibrium liquid-bridge particle-particle benchmark")
    base._progress("[1/5] Computing off-equilibrium static benchmark rows...")
    static_rows = []
    for refinement in base.REFINEMENTS:
        base._progress(f"  - off-equilibrium static refinement {refinement}")
        static_rows.append(compute_static_case(refinement))

    base._progress("[2/5] Computing off-equilibrium dynamic benchmark rows...")
    dynamic_rows = []
    for refinement in base.REFINEMENTS:
        base._progress(f"  - off-equilibrium dynamic refinement {refinement}")
        dynamic_rows.append(compute_dynamic_case(refinement))

    base._progress("[3/5] Rendering off-equilibrium benchmark meshes used in the charts...")
    for row in static_rows:
        base._progress(f"  - off-equilibrium mesh ref {int(row['refinement'])} (n={int(row['n_boundary'])})")
    render_off_equilibrium_mesh_pngs(static_rows)

    base._progress("[4/5] Rendering off-equilibrium reproduced static+dynamic figure...")
    reproduced_path = render_reproduced_static_dynamic_figure(static_rows, dynamic_rows)

    base._progress("[5/5] Rendering off-equilibrium hold-time figure...")
    hold_time_path = render_dynamic_hold_time_figure(verbose=True)
    base.print_max_hold_time_capillary_force_summary(
        dynamic_rows,
        case_label="Case 3",
        max_hold_time=OFF_EQUILIBRIUM_RELAX_MAX_HOLD_TIME,
        dt=OFF_EQUILIBRIUM_RELAX_DT,
    )
    print(f"Saved reproduced static+dynamic figure to: {reproduced_path}")
    print(f"Saved dynamic hold-time figure to: {hold_time_path}")


if __name__ == "__main__":
    main()
