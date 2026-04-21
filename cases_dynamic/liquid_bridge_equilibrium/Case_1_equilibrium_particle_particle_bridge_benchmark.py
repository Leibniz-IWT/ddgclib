"""Equilibrium particle-particle bridge benchmark with Endres (2024) checks.

This script follows the historical static catenoid path used in the older
`bridge.py` / `_case2.py` benchmarks:

* build the exact catenoid surface with ``catenoid_N``
* evaluate discrete curvature with ``b_curvatures_hn_ij_c_ij``
* accumulate the integrated mean-curvature residual in the +z direction
* use the boundary edge spacing as the O(h^2) integration error proxy

The output is a Figure 4.10-style plot that overlays the current repository
results with the historical Endres (2024) reference values embedded in the old
case study scripts.
"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker, colors as mcolors

from hyperct import Complex

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib._curvatures import b_curvatures_hn_ij_c_ij, normalized, vectorise_vnn
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.operators.stress import stress_acceleration, stress_force
from ddgclib.operators.surface_tension import (
    dual_area_heron,
    surface_tension_force,
)

GAMMA = 0.0728
RADIUS = 1.0
THETA_P = 20.0 * math.pi / 180.0
ABC = (1.0, 0.0, 1.0)
REFINEMENTS = (2, 3, 4, 5)
CDIST = 1e-5
DYNAMIC_DT = 2.0e-6
DYNAMIC_STEPS = 100
DYNAMIC_DAMPING = 20.0
OUT_DIR = Path(__file__).resolve().parent / "out" / "Case_1"
DEFAULT_FIGURE = OUT_DIR / "Endres_2024_static_benchmark_vs_reference.png"
CURRENT_ONLY_FIGURE = OUT_DIR / "Endres_2024_static_benchmark_current_only.png"
DYNAMIC_VISIBLE_FIGURE = OUT_DIR / "Endres_2024_static_benchmark_vs_reference_dynamic_visible.png"
REPRODUCED_STATIC_DYNAMIC_FIGURE = OUT_DIR / "Endres_2024_reproduced_static_plus_dynamic_6series.png"
HOLD_TIME_FIGURE = OUT_DIR / "force_error_vs_hold_time_mesh_families_max2e-4.png"
MAX_HOLD_TIME = 2.0e-4
OBSOLETE_BENCHMARK_FIGURES = (
    DEFAULT_FIGURE,
    CURRENT_ONLY_FIGURE,
    DYNAMIC_VISIBLE_FIGURE,
    OUT_DIR / "endres_2024_reproduced_static_plus_dynamic.png",
    OUT_DIR / "stefan_static_benchmark_vs_reference_dynamic_visible.png",
    OUT_DIR / "stefan_static_benchmark_vs_reference.png",
    OUT_DIR / "stefan_static_benchmark_current_only.png",
    OUT_DIR / "endres_2024_benchmark_6series_two_colors.png",
    OUT_DIR / "endres_2024_benchmark_6series_blue_axis.png",
    OUT_DIR / "stefan_static_benchmark_6series.png",
    OUT_DIR / "catenoid_mesh_exact.png",
    OUT_DIR / "catenoid_mesh_perturbed.png",
    OUT_DIR / "catenoid_mesh_relaxed.png",
    OUT_DIR / "catenoid_mesh_comparison.png",
    OUT_DIR / "catenoid_convergence_figure_4_10_style.png",
    OUT_DIR / "catenoid_convergence_mesh_families.png",
    OUT_DIR / "force_estimator_figure_4_10_style_ddg_central_dynamic_static_refined8.png",
    OUT_DIR / "force_error_vs_hold_time_mesh_families_max1e-5_ref_ntotal_dt_short.png",
    OUT_DIR / "force_error_vs_hold_time_mesh_families_max1e-5_ref_ntotal_dt.png",
    OUT_DIR / "force_error_vs_hold_time_mesh_families_max1e-5_ntotal_dt.png",
    OUT_DIR / "force_error_vs_hold_time_mesh_families_max1e-5.png",
    OUT_DIR / "force_error_vs_hold_time_ref5_max1e-5.png",
    OUT_DIR / "force_estimator_figure_4_10_style_ddg_central_only_refined8.png",
    OUT_DIR / "force_estimator_figure_4_10_style_ddg_central_only.png",
    OUT_DIR / "force_estimator_comparison.png",
    OUT_DIR / "force_estimator_figure_4_10_style_ddg_fd_only.png",
    OUT_DIR / "force_estimator_figure_4_10_style_fd_variants.png",
    OUT_DIR / "force_estimator_figure_4_10_style.png",
    OUT_DIR / "force_estimator_figure_4_10_style_all_methods_single_axis.png",
    OUT_DIR / "force_estimator_figure_4_10_style_single_axis_no_trend.png",
    OUT_DIR / "time_integrator_force_comparison.png",
)

INTEGRATION_STATIC_COLOR = "tab:blue"
INTEGRATION_DYNAMIC_COLOR = "tab:blue"
INTEGRATION_REFERENCE_COLOR = "tab:blue"
CAPILLARY_STATIC_COLOR = "tab:orange"
CAPILLARY_DYNAMIC_COLOR = "tab:orange"
CAPILLARY_REFERENCE_COLOR = "tab:orange"


ENDRES_2024_REFERENCE = {
    "n_boundary": np.array([8, 16, 32, 64], dtype=float),
    "capillary_force_error": np.array(
        [
            4.5902409477605044e-17,
            2.4001525593397854e-16,
            -3.5415463964239e-16,
            -1.652188585596348e-16,
        ],
        dtype=float,
    ),
    "integration_error": np.array(
        [
            5.533830997888883,
            1.6208215733413345,
            0.4212378025628348,
            0.1063310109203463,
        ],
        dtype=float,
    ),
}


def _active_vertex_count(HC) -> int:
    """Count active vertices, excluding merged-away cache entries."""

    return sum(1 for _ in HC.V)


def _build_live_endres_catenoid(refinement: int):
    """Build the catenoid with deterministic parameter-order insertion/connectivity.

    The historical benchmark is extremely sensitive to floating-point cancellation.
    Rebuilding the same geometry in a stable parameter-space order removes hidden
    dependence on cache traversal order and gives a reproducible live DDG residual.
    """

    a, _, _ = ABC
    v_l, v_u = -1.5, 1.5
    domain = [
        (0.0, 2 * np.pi),
        (v_l, v_u),
    ]

    def catenoid(u: float, v: float) -> tuple[float, float, float]:
        x = a * np.cos(u) * np.cosh(v / a)
        y = a * np.sin(u) * np.cosh(v / a)
        z = v
        return x, y, z

    HC_plane = Complex(2, domain)
    HC_plane.triangulate()
    for _ in range(refinement):
        HC_plane.refine_all()

    # Sort in parameter space so the 3D vertex cache and neighbor lists no longer
    # depend on the internal cache traversal of the 2D complex.
    plane_vertices = sorted(
        list(HC_plane.V),
        key=lambda v: (-float(v.x_a[0]), float(v.x_a[1])),
    )

    HC = Complex(3, domain)
    bV = set()
    plane_to_spatial = {}

    for v in plane_vertices:
        plane_to_spatial[v] = HC.V[catenoid(*v.x_a)]
        if v.x[1] == domain[1][0] or v.x[1] == domain[1][1]:
            bV.add(plane_to_spatial[v])

    for v in plane_vertices:
        v_spatial = plane_to_spatial[v]
        for vn in v.nn:
            v_spatial.connect(plane_to_spatial[vn])

    HC.V.merge_all(cdist=1e-8)
    for v in list(bV):
        if v not in HC.V:
            bV.remove(v)

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


def render_benchmark_mesh_pngs(
    rows: list[dict[str, float]],
    out_dir: Path = OUT_DIR,
) -> list[Path]:
    """Render only the exact benchmark meshes used in the charts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for row in rows:
        refinement = int(row["refinement"])
        n_boundary = int(row["n_boundary"])
        HC, bV = _build_live_endres_catenoid(refinement)
        verts, triangles = _extract_triangles(HC)
        coords = np.array([v.x_a for v in verts], dtype=float)
        exact_coords = coords.copy()
        displacement = np.linalg.norm(coords - exact_coords, axis=1)
        bmask = _boundary_mask(verts, bV)
        tri_displacement = displacement[triangles].mean(axis=1)
        disp_max = float(displacement.max())
        disp_norm = plt.Normalize(vmin=0.0, vmax=max(disp_max, 1.0e-18))
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
            s=8,
            c=displacement[~bmask],
            cmap=disp_cmap,
            norm=disp_norm,
            alpha=0.9,
            depthshade=False,
        )
        ax.scatter(
            coords[bmask, 0],
            coords[bmask, 1],
            coords[bmask, 2],
            s=18,
            c="#111111",
            alpha=0.95,
            depthshade=False,
        )
        ax.set_title(f"Benchmark Mesh ref {refinement} (n={n_boundary})", pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-58)
        ax.set_box_aspect((1.0, 1.0, 2.2))
        cbar = fig.colorbar(surf, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("displacement from exact catenoid")
        if disp_max == 0.0:
            cbar.set_ticks([0.0])
            cbar.set_ticklabels(["0"])
        fig.tight_layout()

        path = out_dir / f"benchmark_mesh_ref{refinement}_n{n_boundary}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


@lru_cache(maxsize=None)
def compute_static_case(refinement: int, cdist: float = CDIST) -> dict[str, float]:
    """Compute one exact static catenoid benchmark row."""

    HC, bV = _build_live_endres_catenoid(refinement)
    _prepare_surface_benchmark_state(HC, bV)

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    capillary_force_error = 0.0

    for v in HC.V:
        if v in bV:
            continue
        radial = np.array([v.x_a[0], v.x_a[1], 0.0], dtype=float)
        n_i = normalized(radial)[0]
        F, nn = vectorise_vnn(v)
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
        capillary_force_error += np.dot(
            c_outd["HNdA_i"] / np.sum(c_outd["C_ij"]),
            z_hat,
        )

    integration_error = None
    for v in bV:
        boundary_neighbors = [vn for vn in v.nn if vn in bV]
        if not boundary_neighbors:
            continue
        integration_error = 0.5 * np.linalg.norm(v.x_a - boundary_neighbors[0].x_a) ** 2
        break

    if integration_error is None:
        raise RuntimeError("Failed to find a boundary edge for the integration error.")

    return {
        "refinement": refinement,
        "n_total": float(_active_vertex_count(HC)),
        "n_boundary": float(len(bV)),
        "capillary_force_error": float(capillary_force_error),
        "ddg_capillary_force_error": _ddg_capillary_force_error(HC, bV),
        "stress_capillary_force_error": _stress_capillary_force_error(HC, bV),
        "surface_tension_capillary_force_error": _surface_tension_capillary_force_error(HC, bV),
        "multiphase_stress_capillary_force_error": _multiphase_stress_capillary_force_error(HC, bV),
        "fd_capillary_force_error": _fd_capillary_force_error(HC, bV),
        "integration_error": float(integration_error),
    }


def _vertex_radial_normal(v) -> np.ndarray:
    radial = np.array([v.x_a[0], v.x_a[1], 0.0], dtype=float)
    if np.linalg.norm(radial) < 1e-30:
        radial = np.array([1.0, 0.0, 0.0], dtype=float)
    return normalized(radial)[0]


def _prepare_surface_benchmark_state(HC, bV) -> None:
    """Populate the vertex state expected by the built-in surface operators."""

    for v in HC.V:
        v.boundary = v in bV
        v.u = np.zeros(3, dtype=float)
        v.p = 0.0
        v.m = max(float(dual_area_heron(v)), 1.0e-12)
        v.phase = 0
        v.is_interface = v not in bV
        v.interface_phases = frozenset({0, 1}) if v not in bV else frozenset()


def _surface_multiphase_model():
    """Minimal sharp-interface model for the multiphase surface operator."""

    return SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: GAMMA,
        get_mu=lambda phase: 0.0,
    )


def _legacy_curvature_residual(HC, bV) -> float:
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    capillary_force_error = 0.0

    for v in HC.V:
        if v in bV:
            continue
        n_i = _vertex_radial_normal(v)
        F, nn = vectorise_vnn(v)
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
        capillary_force_error += np.dot(
            c_outd["HNdA_i"] / np.sum(c_outd["C_ij"]),
            z_hat,
        )

    return float(capillary_force_error)


def _surface_tension_capillary_force_error(HC, bV) -> float:
    """Integrated axial capillary force from the built-in surface operator."""

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    contributions = [
        float(np.dot(surface_tension_force(v, gamma=GAMMA, dim=3), z_hat))
        for v in HC.V
        if v not in bV
    ]
    return float(math.fsum(contributions))


def _surface_stress_force(v, HC, *, dim: int = 3, mu: float = 0.0, gamma: float = GAMMA) -> np.ndarray:
    """Benchmark-local stress path plus interface surface tension."""

    F = stress_force(v, dim=dim, mu=mu, HC=HC)
    if gamma != 0.0 and getattr(v, "is_interface", False):
        F = F + surface_tension_force(v, gamma=gamma, dim=dim)
    return F


def _surface_stress_acceleration(
    v,
    *,
    dim: int = 3,
    mu: float = 0.0,
    HC=None,
    gamma: float = GAMMA,
    damping: float = 0.0,
) -> np.ndarray:
    F = _surface_stress_force(v, HC=HC, dim=dim, mu=mu, gamma=gamma)
    if damping > 0.0:
        F -= damping * v.u[:dim]
    if v.m < 1.0e-30:
        return np.zeros(dim)
    return F / v.m


def _stress_capillary_force_error(HC, bV) -> float:
    """Integrated axial capillary force from the built-in stress operator."""

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    contributions = [
        float(np.dot(_surface_stress_force(v, HC=HC, dim=3, mu=0.0, gamma=GAMMA), z_hat))
        for v in HC.V
        if v not in bV
    ]
    return float(math.fsum(contributions))


def _multiphase_stress_capillary_force_error(HC, bV) -> float:
    """Integrated axial capillary force from the built-in multiphase operator."""

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    mps = _surface_multiphase_model()
    contributions = [
        float(np.dot(multiphase_stress_force(v, dim=3, mps=mps, HC=HC), z_hat))
        for v in HC.V
        if v not in bV
    ]
    return float(math.fsum(contributions))


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
        contributions.append(-GAMMA * (area_plus - area_minus) / (2.0 * eps))

    return float(math.fsum(contributions))


def _ddg_capillary_force_error(HC, bV) -> float:
    """Integrated DDG axial capillary force using the same physical target as FD."""

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    total_force = 0.0

    for v in HC.V:
        if v in bV:
            continue
        n_i = _vertex_radial_normal(v)
        F, nn = vectorise_vnn(v)
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
        total_force += -GAMMA * float(np.dot(c_outd["HNdA_i"], z_hat))

    return float(total_force)


def _legacy_boundary_integration_error(HC, bV) -> float:
    for v in bV:
        boundary_neighbors = [vn for vn in v.nn if vn in bV]
        if not boundary_neighbors:
            continue
        return float(0.5 * np.linalg.norm(v.x_a - boundary_neighbors[0].x_a) ** 2)
    raise RuntimeError("Failed to find a boundary edge for the integration error.")


def _legacy_dynamic_acceleration(v, dim: int = 3, gamma: float = GAMMA, damping: float = DYNAMIC_DAMPING):
    n_i = _vertex_radial_normal(v)
    F, nn = vectorise_vnn(v)
    c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
    dual_scale = max(float(np.sum(c_outd["C_ij"])), 1e-12)
    force = -gamma * c_outd["HNdA_i"][:dim]
    force -= damping * v.u[:dim]
    return force / dual_scale


@lru_cache(maxsize=None)
def compute_dynamic_case(
    refinement: int,
    cdist: float = CDIST,
    dt: float = DYNAMIC_DT,
    n_steps: int = DYNAMIC_STEPS,
    damping: float = DYNAMIC_DAMPING,
) -> dict[str, float]:
    """Run the benchmark through a damped built-in stress-operator relaxation."""

    HC, bV = _build_live_endres_catenoid(refinement)
    _prepare_surface_benchmark_state(HC, bV)
    dudt_fn = partial(_surface_stress_acceleration, mu=0.0, HC=HC, gamma=GAMMA, damping=damping)

    symplectic_euler(
        HC,
        bV,
        dudt_fn,
        dt=dt,
        n_steps=n_steps,
        dim=3,
        retopologize_fn=False,
    )

    return {
        "refinement": refinement,
        "n_total": float(_active_vertex_count(HC)),
        "n_boundary": float(len(bV)),
        "capillary_force_error": _legacy_curvature_residual(HC, bV),
        "ddg_capillary_force_error": _ddg_capillary_force_error(HC, bV),
        "stress_capillary_force_error": _stress_capillary_force_error(HC, bV),
        "surface_tension_capillary_force_error": _surface_tension_capillary_force_error(HC, bV),
        "multiphase_stress_capillary_force_error": _multiphase_stress_capillary_force_error(HC, bV),
        "fd_capillary_force_error": _fd_capillary_force_error(HC, bV),
        "integration_error": _legacy_boundary_integration_error(HC, bV),
    }


def run_static_benchmark(refinements: tuple[int, ...] = REFINEMENTS) -> list[dict[str, float]]:
    """Compute the full Stefan-static benchmark table."""

    return [compute_static_case(refinement) for refinement in refinements]


def run_dynamic_benchmark(refinements: tuple[int, ...] = REFINEMENTS) -> list[dict[str, float]]:
    """Compute the dynamic-hold variant of the Stefan-static benchmark."""

    return [compute_dynamic_case(refinement) for refinement in refinements]


def _format_scientific(value: float) -> str:
    return f"{value:.6e}"


def _safe_positive_min(values: np.ndarray | list[float], fallback: float = 1e-18) -> float:
    arr = np.asarray(values, dtype=float)
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return fallback
    return max(float(positive.min()), fallback)


def _progress(message: str) -> None:
    print(message, flush=True)


def print_summary(rows: list[dict[str, float]]) -> None:
    """Print a concise comparison table against the historical reference."""

    ref_cap = ENDRES_2024_REFERENCE["capillary_force_error"]
    ref_geo = ENDRES_2024_REFERENCE["integration_error"]

    header = (
        "ref  n_boundary  n_total  current_capillary    Endres(2024)_capillary"
        "     current_geo    Endres(2024)_integration"
    )
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(rows):
        print(
            f"{int(row['refinement']):>3}  "
            f"{int(row['n_boundary']):>10}  "
            f"{int(row['n_total']):>7}  "
            f"{_format_scientific(row['capillary_force_error']):>18}  "
            f"{_format_scientific(ref_cap[idx]):>18}  "
            f"{row['integration_error']:>14.12f}  "
            f"{ref_geo[idx]:>12.12f}"
        )


def print_dynamic_summary(rows: list[dict[str, float]]) -> None:
    header = "ref  n_boundary  n_total  dynamic_capillary      dynamic_geo"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['refinement']):>3}  "
            f"{int(row['n_boundary']):>10}  "
            f"{int(row['n_total']):>7}  "
            f"{_format_scientific(row['capillary_force_error']):>17}  "
            f"{row['integration_error']:>14.12f}"
        )


def print_max_hold_time_capillary_force_summary(
    rows: list[dict[str, float]],
    *,
    case_label: str,
    max_hold_time: float,
    dt: float,
) -> None:
    """Print DDG/FD capillary force errors (%) at the script's max hold time."""

    print(
        f"{case_label} max hold-time capillary force errors (%) "
        f"at t={max_hold_time:.6f} s (dt={dt:.0e} s)"
    )
    header = "ref  n_boundary  n_total  DDG_capillary_%    FD_capillary_%"
    print(header)
    print("-" * len(header))
    for row in rows:
        ddg_pct = 100.0 * abs(float(row["ddg_capillary_force_error"]))
        fd_pct = 100.0 * abs(float(row["fd_capillary_force_error"]))
        print(
            f"{int(row['refinement']):>3}  "
            f"{int(row['n_boundary']):>10}  "
            f"{int(row['n_total']):>7}  "
            f"{ddg_pct:>16.12e}  "
            f"{fd_pct:>16.12e}"
        )


def print_builtin_operator_capillary_force_summary(
    rows: list[dict[str, float]],
    *,
    case_label: str,
    label: str,
    time_value: float | None = None,
    dt: float | None = None,
) -> None:
    """Print the built-in operator capillary force errors (%) for one state."""

    if time_value is None:
        print(f"{case_label} {label} built-in operator capillary force errors (%)")
    else:
        print(
            f"{case_label} {label} built-in operator capillary force errors (%) "
            f"at t={time_value:.6f} s (dt={dt:.0e} s)"
        )
    header = "ref  n_boundary  n_total  stress_%  multiphase_stress_%  curvature_%"
    print(header)
    print("-" * len(header))
    for row in rows:
        stress_pct = 100.0 * abs(float(row["stress_capillary_force_error"]))
        multiphase_pct = 100.0 * abs(float(row["multiphase_stress_capillary_force_error"]))
        curvature_pct = 100.0 * abs(float(row["ddg_capillary_force_error"]))
        print(
            f"{int(row['refinement']):>3}  "
            f"{int(row['n_boundary']):>10}  "
            f"{int(row['n_total']):>7}  "
            f"{stress_pct:>8.12e}  "
            f"{multiphase_pct:>20.12e}  "
            f"{curvature_pct:>12.12e}"
        )


def _hold_step_schedule(max_steps: int) -> list[int]:
    base_steps = [1, 5, 10, 20, 50]
    steps = [step for step in base_steps if step <= max_steps]
    if max_steps not in steps:
        steps.append(max_steps)
    return sorted(set(steps))


def render_dynamic_hold_time_figure(
    refinements: tuple[int, ...] = REFINEMENTS,
    max_hold_time: float = MAX_HOLD_TIME,
    out_path: Path = HOLD_TIME_FIGURE,
    verbose: bool = False,
) -> Path:
    """Render the dynamic hold-time mesh-family figure from the live Endres path."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: dict[int, list[dict[str, float]]] = {}
    for refinement in refinements:
        if verbose:
            _progress(f"[hold-time] refinement {refinement}: sampling dynamic hold times")
        dt = DYNAMIC_DT
        max_steps = max(1, int(round(max_hold_time / dt)))
        rows = []
        for n_steps in _hold_step_schedule(max_steps):
            hold = compute_dynamic_case(refinement, dt=dt, n_steps=n_steps)
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

    fig.suptitle("Equilibrium DDG vs FD Capillary Error", fontsize=18)
    fig.text(
        0.5,
        0.03,
        f"max hold time={max_hold_time:.0e} s; orange solid = DDG, blue dashed = FD",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_figure(
    rows: list[dict[str, float]],
    dynamic_rows: list[dict[str, float]],
    out_path: Path = DEFAULT_FIGURE,
) -> Path:
    """Render a Figure 4.10-style current-vs-reference comparison."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_boundary = np.array([row["n_boundary"] for row in rows], dtype=float)
    current_capillary = np.abs(
        np.array([row["capillary_force_error"] for row in rows], dtype=float)
    )
    current_geo = np.array([row["integration_error"] for row in rows], dtype=float)
    dynamic_capillary = np.abs(
        np.array([row["capillary_force_error"] for row in dynamic_rows], dtype=float)
    )
    dynamic_geo = np.array([row["integration_error"] for row in dynamic_rows], dtype=float)

    ref_n = ENDRES_2024_REFERENCE["n_boundary"]
    ref_capillary = np.abs(ENDRES_2024_REFERENCE["capillary_force_error"])
    ref_geo = ENDRES_2024_REFERENCE["integration_error"]

    fig = plt.figure(figsize=(10.5, 6.6))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln_current_cap = ax2.loglog(
        n_boundary,
        current_capillary,
        "x",
        color="tab:blue",
        markersize=10,
        mew=2.0,
        label=r"Current DDG capillary force error ($|F_{cap}-\hat{F}_{cap}|/F_{cap}$)",
    )
    ln_ref_cap = ax2.loglog(
        ref_n,
        ref_capillary,
        "o",
        color="tab:blue",
        fillstyle="none",
        markersize=8,
        mew=1.8,
        label="Endres (2024) reference capillary force error",
    )
    ln_dynamic_cap = ax2.loglog(
        n_boundary,
        dynamic_capillary,
        "+",
        color="tab:purple",
        markersize=13,
        mew=2.2,
        label=f"Dynamic DDG capillary force error (dt={DYNAMIC_DT:.0e}, steps={DYNAMIC_STEPS})",
    )

    ln_current_geo = ax.loglog(
        n_boundary,
        current_geo,
        "X",
        color=INTEGRATION_STATIC_COLOR,
        markersize=9,
        label=r"Current integration error (p-normed trapezoidal rule $O(h^2)$)",
    )
    ln_ref_geo = ax.loglog(
        ref_n,
        ref_geo,
        "s",
        color=INTEGRATION_REFERENCE_COLOR,
        fillstyle="none",
        markersize=7,
        mew=1.8,
        label="Endres (2024) reference integration error",
    )
    ln_dynamic_geo = ax.loglog(
        n_boundary,
        dynamic_geo,
        "^",
        color="tab:green",
        markersize=8,
        fillstyle="none",
        mew=1.8,
        label="Dynamic integration error (same Stefan proxy after hold)",
    )

    ax.set_title("Endres (2024) Catenoid Benchmark", fontsize=18, pad=14)
    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color=INTEGRATION_STATIC_COLOR)
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color=CAPILLARY_STATIC_COLOR)

    ymin_cap = _safe_positive_min(
        [current_capillary.min(), ref_capillary.min(), dynamic_capillary.min()]
    )
    ymax_cap = max(current_capillary.max(), ref_capillary.max(), dynamic_capillary.max())
    ax2.set_ylim([10 ** math.ceil(math.log10(ymax_cap)), 10 ** math.floor(math.log10(ymin_cap))])

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="y", colors=INTEGRATION_STATIC_COLOR)
    ax2.tick_params(axis="y", colors=CAPILLARY_STATIC_COLOR)
    ax.spines["left"].set_color(INTEGRATION_STATIC_COLOR)
    ax2.spines["right"].set_color(CAPILLARY_STATIC_COLOR)
    ax.grid(True, which="both", alpha=0.18)

    lns = ln_current_cap + ln_ref_cap + ln_dynamic_cap + ln_current_geo + ln_ref_geo + ln_dynamic_geo
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc="upper right", framealpha=1.0, facecolor="white")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out_path


def render_dynamic_visible_figure(
    rows: list[dict[str, float]],
    dynamic_rows: list[dict[str, float]],
    out_path: Path = DYNAMIC_VISIBLE_FIGURE,
) -> Path:
    """Render all six series with small x-offsets so dynamic points stay visible."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_boundary = np.array([row["n_boundary"] for row in rows], dtype=float)
    n_static = n_boundary * 0.97
    n_reference = n_boundary
    n_dynamic = n_boundary * 1.03

    current_capillary = np.abs(
        np.array([row["capillary_force_error"] for row in rows], dtype=float)
    )
    dynamic_capillary = np.abs(
        np.array([row["capillary_force_error"] for row in dynamic_rows], dtype=float)
    )
    current_geo = np.array([row["integration_error"] for row in rows], dtype=float)
    dynamic_geo = np.array([row["integration_error"] for row in dynamic_rows], dtype=float)
    ref_capillary = np.abs(ENDRES_2024_REFERENCE["capillary_force_error"])
    ref_geo = np.array(ENDRES_2024_REFERENCE["integration_error"], dtype=float)

    fig = plt.figure(figsize=(10.2, 6.4))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln1 = ax2.loglog(
        n_static,
        current_capillary,
        "x",
        color=CAPILLARY_STATIC_COLOR,
        markersize=11,
        mew=2.1,
        label="static DDG capillary",
    )
    ln2 = ax2.loglog(
        n_dynamic,
        dynamic_capillary,
        "+",
        color=CAPILLARY_DYNAMIC_COLOR,
        markersize=14,
        mew=2.4,
        label=f"Dynamic DDG capillary error (dt={DYNAMIC_DT:.0e}, steps={DYNAMIC_STEPS})",
    )
    ln_ref_cap = ax2.loglog(
        n_reference,
        ref_capillary,
        "o",
        color=CAPILLARY_REFERENCE_COLOR,
        fillstyle="none",
        markersize=8,
        mew=1.8,
        label="Endres (2024) reference capillary force error",
    )
    ln3 = ax.loglog(
        n_static,
        current_geo,
        "X",
        color=INTEGRATION_STATIC_COLOR,
        markersize=10,
        label=r"Static integration error $O(h^2)$",
    )
    ln4 = ax.loglog(
        n_dynamic,
        dynamic_geo,
        "^",
        color=INTEGRATION_DYNAMIC_COLOR,
        markersize=9,
        fillstyle="none",
        mew=2.0,
        label=r"Dynamic FD / integration error $O(h^2)$",
    )
    ln_ref_geo = ax.loglog(
        n_reference,
        ref_geo,
        "s",
        color=INTEGRATION_REFERENCE_COLOR,
        fillstyle="none",
        markersize=7,
        mew=1.8,
        label="Endres (2024) reference integration error",
    )

    ax.set_title("Endres (2024) Benchmark With Static, Dynamic, and Reference Points", fontsize=17, pad=12)
    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color=INTEGRATION_STATIC_COLOR)
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color=CAPILLARY_STATIC_COLOR)

    ymin_cap = _safe_positive_min(
        [current_capillary.min(), dynamic_capillary.min(), ref_capillary.min()]
    )
    ymax_cap = max(current_capillary.max(), dynamic_capillary.max(), ref_capillary.max())
    ax2.set_ylim([10 ** math.ceil(math.log10(ymax_cap)), 10 ** math.floor(math.log10(ymin_cap))])
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="y", colors=INTEGRATION_STATIC_COLOR)
    ax2.tick_params(axis="y", colors=CAPILLARY_STATIC_COLOR)
    ax.spines["left"].set_color(INTEGRATION_STATIC_COLOR)
    ax2.spines["right"].set_color(CAPILLARY_STATIC_COLOR)
    ax.grid(True, which="both", alpha=0.18)

    lns = ln1 + ln2 + ln_ref_cap + ln3 + ln4 + ln_ref_geo
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc="upper right", framealpha=1.0, facecolor="white")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out_path


def render_reproduced_static_dynamic_figure(
    static_rows: list[dict[str, float]],
    dynamic_rows: list[dict[str, float]],
    out_path: Path = REPRODUCED_STATIC_DYNAMIC_FIGURE,
) -> Path:
    """Render a six-series live-vs-reference figure."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_reference = np.array(ENDRES_2024_REFERENCE["n_boundary"], dtype=float)
    n_static = n_reference * 0.97
    n_dynamic = n_reference * 1.03

    static_capillary = 100.0 * np.abs(
        np.array([row["ddg_capillary_force_error"] for row in static_rows], dtype=float)
    )
    static_geo = np.array([row["integration_error"] for row in static_rows], dtype=float)
    ref_capillary = 100.0 * np.abs(ENDRES_2024_REFERENCE["capillary_force_error"])
    ref_geo = np.array(ENDRES_2024_REFERENCE["integration_error"], dtype=float)
    dynamic_capillary = 100.0 * np.abs(
        np.array([row["ddg_capillary_force_error"] for row in dynamic_rows], dtype=float)
    )
    dynamic_geo = np.array([row["integration_error"] for row in dynamic_rows], dtype=float)

    fig = plt.figure(figsize=(10.0, 6.3))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln1 = ax2.loglog(
        n_static,
        static_capillary,
        "o",
        color=CAPILLARY_STATIC_COLOR,
        fillstyle="none",
        markersize=8,
        mew=1.9,
        label="static DDG capillary error",
    )
    ln2 = ax2.loglog(
        n_reference,
        ref_capillary,
        "x",
        color=CAPILLARY_REFERENCE_COLOR,
        markersize=10,
        mew=2.0,
        label="Endres (2024) reference capillary",
    )
    ln3 = ax2.loglog(
        n_dynamic,
        dynamic_capillary,
        "+",
        color=CAPILLARY_DYNAMIC_COLOR,
        markersize=14,
        mew=2.4,
        label=f"dynamic DDG capillary error (damping={DYNAMIC_DAMPING:g})",
    )
    ln4 = ax.loglog(
        n_static,
        static_geo,
        "s",
        color=INTEGRATION_STATIC_COLOR,
        fillstyle="none",
        markersize=7,
        mew=1.8,
        label="static DDG integration error",
    )
    ln5 = ax.loglog(
        n_reference,
        ref_geo,
        "X",
        color=INTEGRATION_REFERENCE_COLOR,
        markersize=9,
        mew=1.8,
        label="Endres (2024) reference integration error",
    )
    ln6 = ax.loglog(
        n_dynamic,
        dynamic_geo,
        "^",
        color=INTEGRATION_DYNAMIC_COLOR,
        markersize=9,
        fillstyle="none",
        mew=2.0,
        label="dynamic DDG integration error",
    )

    ax.set_title("Equilibrium Static + Dynamic + Reference", fontsize=17, pad=12)
    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color=INTEGRATION_STATIC_COLOR)
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color=CAPILLARY_STATIC_COLOR)

    ymin_cap = _safe_positive_min(
        [static_capillary.min(), ref_capillary.min(), dynamic_capillary.min()]
    )
    ymax_cap = max(static_capillary.max(), ref_capillary.max(), dynamic_capillary.max())
    ax2.set_ylim([10 ** math.ceil(math.log10(ymax_cap)), 10 ** math.floor(math.log10(ymin_cap))])
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="y", colors=INTEGRATION_STATIC_COLOR)
    ax2.tick_params(axis="y", colors=CAPILLARY_STATIC_COLOR)
    ax.spines["left"].set_color(INTEGRATION_STATIC_COLOR)
    ax2.spines["right"].set_color(CAPILLARY_STATIC_COLOR)
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


def render_current_only_figure(
    rows: list[dict[str, float]],
    out_path: Path = CURRENT_ONLY_FIGURE,
) -> Path:
    """Render the clean current-repo Figure 4.10-style static benchmark."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_boundary = np.array([row["n_boundary"] for row in rows], dtype=float)
    current_capillary = np.abs(
        np.array([row["capillary_force_error"] for row in rows], dtype=float)
    )
    current_geo = np.array([row["integration_error"] for row in rows], dtype=float)

    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln1 = ax2.loglog(
        n_boundary,
        current_capillary,
        "x",
        color=CAPILLARY_STATIC_COLOR,
        markersize=10,
        mew=2.0,
        label=r"Capillary force error: $(F_{cap}-\hat{F}_{cap})/F_{cap}$",
    )
    ln2 = ax.loglog(
        n_boundary,
        current_geo,
        "X",
        color=INTEGRATION_STATIC_COLOR,
        markersize=9,
        label=r"Integration error (p-normed trapezoidal rule $O(h^2)$)",
    )

    ax.set_title("Endres (2024) Static Benchmark (Current Repo)", fontsize=17, pad=12)
    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color=INTEGRATION_STATIC_COLOR)
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color=CAPILLARY_STATIC_COLOR)

    ymin_cap = _safe_positive_min(current_capillary)
    ymax_cap = current_capillary.max()
    ax2.set_ylim([10 ** math.ceil(math.log10(ymax_cap)), 10 ** math.floor(math.log10(ymin_cap))])
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="y", colors=INTEGRATION_STATIC_COLOR)
    ax2.tick_params(axis="y", colors=CAPILLARY_STATIC_COLOR)
    ax.spines["left"].set_color(INTEGRATION_STATIC_COLOR)
    ax2.spines["right"].set_color(CAPILLARY_STATIC_COLOR)
    ax.grid(True, which="both", alpha=0.18)

    lns = ln1 + ln2
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc="upper right", framealpha=1.0, facecolor="white")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out_path


def _cleanup_obsolete_benchmark_figures() -> None:
    for path in OBSOLETE_BENCHMARK_FIGURES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    _progress("Equilibrium particle-particle bridge benchmark")
    _progress("[1/5] Computing live static benchmark rows...")
    rows = []
    for refinement in REFINEMENTS:
        _progress(f"  - static refinement {refinement}")
        rows.append(compute_static_case(refinement))

    _progress("[2/5] Computing damped dynamic relaxation rows...")
    dynamic_rows = []
    for refinement in REFINEMENTS:
        _progress(f"  - dynamic refinement {refinement}")
        dynamic_rows.append(compute_dynamic_case(refinement))

    _progress("[3/5] Cleaning old benchmark-only outputs...")
    _cleanup_obsolete_benchmark_figures()

    _progress("[4/5] Rendering benchmark meshes used in the charts...")
    for row in rows:
        _progress(
            f"  - mesh ref {int(row['refinement'])} (n={int(row['n_boundary'])})"
        )
    render_benchmark_mesh_pngs(rows)

    _progress("[5/5] Rendering benchmark figures...")
    reproduced_path = render_reproduced_static_dynamic_figure(rows, dynamic_rows)
    hold_time_path = render_dynamic_hold_time_figure(verbose=True)
    print_builtin_operator_capillary_force_summary(
        rows,
        case_label="Case 1",
        label="static",
    )
    print_builtin_operator_capillary_force_summary(
        dynamic_rows,
        case_label="Case 1",
        label="max hold-time",
        time_value=MAX_HOLD_TIME,
        dt=DYNAMIC_DT,
    )
    print_max_hold_time_capillary_force_summary(
        dynamic_rows,
        case_label="Case 1",
        max_hold_time=MAX_HOLD_TIME,
        dt=DYNAMIC_DT,
    )
    print(f"Saved reproduced static+dynamic figure to: {reproduced_path}")
    print(f"Saved dynamic hold-time figure to: {hold_time_path}")


if __name__ == "__main__":
    main()
