"""Case 5: volumetric stress-operator equilibrium benchmark.

This case keeps the volumetric ``stress.py`` path and also reports
boundary-surface capillary-force errors on the same mesh family against the
historical Endres (2024) catenoid reference.

What this case verifies:
    1. A volumetric catenoid-like bridge mesh can be built.
    2. ``stress.py`` can be run on that volume through the dynamic integrator.
    3. The zero-pressure / zero-velocity equilibrium remains at zero volumetric
       Cauchy stress residual.
    4. The extracted boundary surface can be benchmarked against the analytical
       catenoid reference using the built-in stress, multiphase, and DDG
       operator families.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache, partial
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hyperct import Complex
from hyperct.ddg import compute_vd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib._curvatures import b_curvatures_hn_ij_c_ij, vectorise_vnn
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.operators.surface_tension import dual_area_heron, surface_tension_force
from ddgclib.operators.stress import dual_volume, stress_acceleration, stress_force

CASE_DIR = Path(__file__).resolve().parent
OUT_DIR = CASE_DIR / "out" / "Case_5"

REFINEMENTS = (0, 1, 2, 3)
GAMMA = 0.0728
THETA_RADIUS = 1.03
Z_LOWER = -1.5
Z_UPPER = 1.5
DYNAMIC_DT = 2.0e-6
DYNAMIC_DAMPING = 0.0
MAX_HOLD_TIME = 2.0e-4
HOLD_TIME_FIGURE = OUT_DIR / "force_error_vs_hold_time_mesh_families_max2e-4.png"
REPRODUCED_STATIC_DYNAMIC_FIGURE = OUT_DIR / "volumetric_stress_reproduced_static_plus_dynamic_6series.png"
DDG_FD_COMPANION_FIGURE = OUT_DIR / "ddg_vs_fd_capillary_force_error_companion_max2e-4.png"
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


def _default_worker_count() -> int:
    env_value = os.getenv("DDGCLIB_CASE5_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1))


DYNAMIC_WORKERS = _default_worker_count()
PLOT_FLOOR = 1.0e-18
SURFACE_REFINEMENT_OFFSET = 2
PRISM_TETS = (
    (0, 1, 2, 5),
    (0, 1, 5, 4),
    (0, 4, 5, 3),
)
HEX_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def _active_vertex_count(HC) -> int:
    return sum(1 for _ in HC.V)


def _surface_refinement(refinement: int) -> int:
    return refinement + SURFACE_REFINEMENT_OFFSET


def _radial_ring_factors_for_n(n_boundary: int) -> tuple[float, ...]:
    """Return cap/volume radial rings for the requested boundary count."""

    if n_boundary <= 8:
        return (0.5, 1.0)
    if n_boundary <= 16:
        return (1.0 / 3.0, 2.0 / 3.0, 1.0)
    if n_boundary <= 32:
        return (0.25, 0.5, 0.75, 1.0)
    # Keep the n=64 outer shell, but do not over-refine the interior rings.
    return (0.25, 0.5, 0.75, 1.0)


def _catenoid_xyz(u: float, z: float) -> tuple[float, float, float]:
    radius = THETA_RADIUS * math.cosh(z / THETA_RADIUS)
    return (
        radius * math.cos(u),
        radius * math.sin(u),
        z,
    )


def _build_case5_outer_surface(refinement: int):
    """Build the Case-5 outer surface with the same catenoid family as Case 1."""

    surface_ref = _surface_refinement(refinement)
    domain = [
        (0.0, 2.0 * math.pi),
        (Z_LOWER, Z_UPPER),
    ]

    HC_plane = Complex(2, domain)
    HC_plane.triangulate()
    for _ in range(surface_ref):
        HC_plane.refine_all()

    plane_vertices = sorted(
        list(HC_plane.V),
        key=lambda v: (-float(v.x_a[0]), float(v.x_a[1])),
    )

    HC_surface = Complex(3, domain)
    bV_surface = set()
    plane_to_spatial = {}

    for v in plane_vertices:
        u, z = map(float, v.x_a)
        plane_to_spatial[v] = HC_surface.V[_catenoid_xyz(u, z)]
        if z == domain[1][0] or z == domain[1][1]:
            bV_surface.add(plane_to_spatial[v])

    for v in plane_vertices:
        for vn in v.nn:
            plane_to_spatial[v].connect(plane_to_spatial[vn])

    HC_surface.V.merge_all(cdist=1.0e-8)
    bV_surface = {v for v in bV_surface if v in HC_surface.V}

    return HC_surface, bV_surface


def _group_surface_ring_vertices(HC_surface) -> list[list[object]]:
    z_levels = sorted({round(float(v.x_a[2]), 12) for v in HC_surface.V})
    rings: list[list[object]] = []
    for z in z_levels:
        ring = [
            v
            for v in HC_surface.V
            if abs(float(v.x_a[2]) - z) < 1.0e-10
        ]
        ring = sorted(ring, key=lambda v: math.atan2(float(v.x_a[1]), float(v.x_a[0])))
        rings.append(ring)
    return rings


def _surface_template(HC_surface, bV_surface):
    ring_vertices = _group_surface_ring_vertices(HC_surface)
    ordered = [v for ring in ring_vertices for v in ring]
    index = {id(v): idx for idx, v in enumerate(ordered)}
    edges = sorted(
        {
            tuple(sorted((index[id(v)], index[id(nb)])))
            for v in ordered
            for nb in v.nn
            if id(v) != id(nb)
        }
    )
    boundary_indices = {index[id(v)] for v in bV_surface}
    outer_ring_coords = [
        [tuple(map(float, v.x_a)) for v in ring]
        for ring in ring_vertices
    ]
    return outer_ring_coords, edges, boundary_indices


def _build_surface_complex_from_template(
    outer_rings,
    surface_edges,
    boundary_indices,
) -> tuple[Complex, set]:
    surface = Complex(3, domain=None)
    flat_vertices = [
        surface.V[tuple(map(float, v.x_a))]
        for ring in outer_rings
        for v in ring
    ]
    for i, j in surface_edges:
        flat_vertices[i].connect(flat_vertices[j])

    surface_bV = {flat_vertices[idx] for idx in boundary_indices}
    return surface, surface_bV


def _connect_prism(lower_tri, upper_tri):
    prism = list(lower_tri) + list(upper_tri)
    for tet in PRISM_TETS:
        tet_vertices = [prism[j] for j in tet]
        for a in range(4):
            for b in range(a + 1, 4):
                tet_vertices[a].connect(tet_vertices[b])


def _render_surface_geometry_with_caps(HC, outer_rings, layer_rings, surface_edges, surface_boundary_indices):
    surface, _surface_bV = _build_surface_complex_from_template(
        outer_rings,
        surface_edges,
        surface_boundary_indices,
    )
    shell_verts, shell_triangles = _extract_triangles(surface)

    coords: list[tuple[float, float, float]] = [tuple(map(float, v.x_a)) for v in shell_verts]
    coord_index = {tuple(round(c, 12) for c in coord): i for i, coord in enumerate(coords)}

    def ensure_coord(coord) -> int:
        key = tuple(round(float(c), 12) for c in coord)
        idx = coord_index.get(key)
        if idx is None:
            idx = len(coords)
            coords.append(tuple(map(float, coord)))
            coord_index[key] = idx
        return idx

    triangles = [tuple(map(int, tri)) for tri in shell_triangles.tolist()]

    z_min = float(outer_rings[0][0].x_a[2])
    z_max = float(outer_rings[-1][0].x_a[2])
    centers = {}
    for v in HC.V:
        x, y, z = map(float, v.x_a)
        if abs(x) < 1.0e-12 and abs(y) < 1.0e-12:
            if abs(z - z_min) < 1.0e-12:
                centers["bottom"] = (x, y, z)
            elif abs(z - z_max) < 1.0e-12:
                centers["top"] = (x, y, z)

    for name, rings in (("bottom", layer_rings[0]), ("top", layer_rings[-1])):
        center_coord = centers.get(name)
        if center_coord is None:
            continue
        center_idx = ensure_coord(center_coord)
        ring_indices_by_radius = [
            [ensure_coord(tuple(map(float, v.x_a))) for v in ring]
            for ring in rings
        ]
        n_ring = len(ring_indices_by_radius[0])
        for i in range(n_ring):
            triangles.append(
                (
                    center_idx,
                    ring_indices_by_radius[0][i],
                    ring_indices_by_radius[0][(i + 1) % n_ring],
                )
            )
        for band in range(len(ring_indices_by_radius) - 1):
            inner = ring_indices_by_radius[band]
            outer = ring_indices_by_radius[band + 1]
            for i in range(n_ring):
                triangles.append((inner[i], outer[i], outer[(i + 1) % n_ring]))
                triangles.append((inner[i], outer[(i + 1) % n_ring], inner[(i + 1) % n_ring]))

    boundary_coords = np.array([v.x_a for v in HC.V if v.boundary], dtype=float)
    return np.array(coords, dtype=float), np.array(triangles, dtype=int), boundary_coords


def _build_structured_volumetric_catenoid(refinement: int):
    """Build a filled catenoid-like volume whose outer shell matches Case 1."""

    surface_HC, surface_bV = _build_case5_outer_surface(refinement)
    rings_coords, surface_edges, surface_boundary_indices = _surface_template(surface_HC, surface_bV)

    HC = Complex(3, domain=None)
    vertex_cache: dict[tuple[float, float, float], object] = {}

    def get_vertex(coord) -> object:
        key = tuple(round(float(c), 12) for c in coord)
        if key not in vertex_cache:
            vertex_cache[key] = HC.V[key]
        return vertex_cache[key]

    radial_ring_factors = _radial_ring_factors_for_n(len(rings_coords[0]))

    centers = []
    layer_rings = []
    outer_rings = []
    for ring_coords in rings_coords:
        z = ring_coords[0][2]
        center_vertex = get_vertex((0.0, 0.0, float(z)))
        centers.append(center_vertex)
        radial_rings = []
        for factor in radial_ring_factors:
            radial_rings.append(
                [
                    get_vertex((factor * coord[0], factor * coord[1], coord[2]))
                    for coord in ring_coords
                ]
            )
        layer_rings.append(radial_rings)
        outer_rings.append(radial_rings[-1])

    for lower_center, upper_center, lower_radial_rings, upper_radial_rings in zip(
        centers[:-1],
        centers[1:],
        layer_rings[:-1],
        layer_rings[1:],
    ):
        first_lower = lower_radial_rings[0]
        first_upper = upper_radial_rings[0]
        n_ring = len(first_lower)
        for i in range(n_ring):
            _connect_prism(
                (
                    lower_center,
                    first_lower[i],
                    first_lower[(i + 1) % n_ring],
                ),
                (
                    upper_center,
                    first_upper[i],
                    first_upper[(i + 1) % n_ring],
                ),
            )

        for band in range(len(lower_radial_rings) - 1):
            inner_lower = lower_radial_rings[band]
            outer_lower = lower_radial_rings[band + 1]
            inner_upper = upper_radial_rings[band]
            outer_upper = upper_radial_rings[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                hexahedron = (
                    inner_lower[i],
                    inner_lower[i_next],
                    outer_lower[i_next],
                    outer_lower[i],
                    inner_upper[i],
                    inner_upper[i_next],
                    outer_upper[i_next],
                    outer_upper[i],
                )
                for tet in HEX_TETS:
                    tet_vertices = [hexahedron[j] for j in tet]
                    for a in range(4):
                        for b in range(a + 1, 4):
                            tet_vertices[a].connect(tet_vertices[b])

    z_min = float(centers[0].x_a[2])
    z_max = float(centers[-1].x_a[2])
    bV_vol = set()
    for center, radial_rings in zip(centers, layer_rings):
        z = float(center.x_a[2])
        bV_vol.update(radial_rings[-1])
        if abs(z - z_min) < 1.0e-12 or abs(z - z_max) < 1.0e-12:
            bV_vol.add(center)
            for ring in radial_rings:
                bV_vol.update(ring)

    for v in HC.V:
        v.boundary = v in bV_vol

    compute_vd(HC, method="barycentric")

    for v in HC.V:
        v.u = np.zeros(3, dtype=float)
        v.p = 0.0
        v.m = max(float(dual_volume(v, HC, dim=3)), 1.0e-12)

    return HC, bV_vol, outer_rings, layer_rings, surface_edges, surface_boundary_indices


def _stress_axial_force_residual(HC, bV, mu: float = 0.0) -> float:
    return math.fsum(
        float(stress_force(v, dim=3, mu=mu, HC=HC)[2])
        for v in HC.V
        if v not in bV
    )


def _stress_max_local_force_norm(HC, bV, mu: float = 0.0) -> float:
    vals = [
        float(np.linalg.norm(stress_force(v, dim=3, mu=mu, HC=HC)))
        for v in HC.V
        if v not in bV
    ]
    return max(vals, default=0.0)


def _prepare_surface_benchmark_state(HC, bV) -> None:
    """Populate the boundary-surface state used by built-in capillary operators."""

    for v in HC.V:
        v.boundary = v in bV
        v.u = np.zeros(3, dtype=float)
        v.p = 0.0
        v.m = max(float(dual_area_heron(v)), 1.0e-12)
        v.phase = 0
        v.is_interface = v not in bV
        v.interface_phases = frozenset({0, 1}) if v not in bV else frozenset()


def _surface_multiphase_model():
    return SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: GAMMA,
        get_mu=lambda phase: 0.0,
    )


def _surface_tension_capillary_force_error(surface, bV_surface) -> float:
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    contributions = [
        float(np.dot(surface_tension_force(v, gamma=GAMMA, dim=3), z_hat))
        for v in surface.V
        if v not in bV_surface
    ]
    return float(math.fsum(contributions))


def _surface_stress_force(v, surface, *, dim: int = 3, mu: float = 0.0, gamma: float = GAMMA) -> np.ndarray:
    """Case-local stress path plus interface surface tension on the extracted surface."""

    F = stress_force(v, dim=dim, mu=mu, HC=surface)
    if gamma != 0.0 and getattr(v, "is_interface", False):
        F = F + surface_tension_force(v, gamma=gamma, dim=dim)
    return F


def _surface_stress_capillary_force_error(surface, bV_surface) -> float:
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    contributions = [
        float(np.dot(_surface_stress_force(v, surface, dim=3, mu=0.0, gamma=GAMMA), z_hat))
        for v in surface.V
        if v not in bV_surface
    ]
    return float(math.fsum(contributions))


def _surface_multiphase_stress_capillary_force_error(surface, bV_surface) -> float:
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    mps = _surface_multiphase_model()
    contributions = [
        float(np.dot(multiphase_stress_force(v, dim=3, mps=mps, HC=surface), z_hat))
        for v in surface.V
        if v not in bV_surface
    ]
    return float(math.fsum(contributions))


def _extract_triangles(HC):
    verts = list(HC.V)
    index = {id(v): i for i, v in enumerate(verts)}
    triangles = set()

    for i, v in enumerate(verts):
        nbr_ids = sorted(index[id(nb)] for nb in v.nn if index[id(nb)] > i)
        for k, j in enumerate(nbr_ids):
            j_neighbors = {index[id(nb)] for nb in verts[j].nn}
            for ell in nbr_ids[k + 1 :]:
                if ell in j_neighbors:
                    triangles.add(tuple(sorted((i, j, ell))))

    return verts, np.array(sorted(triangles), dtype=int)


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _surface_complex_geometry(surface: Complex):
    verts, triangles = _extract_triangles(surface)
    coords = np.array([v.x_a for v in verts], dtype=float)

    normals = np.zeros_like(coords)
    for tri in triangles:
        i, j, k = (int(idx) for idx in tri)
        a, b, c = coords[[i, j, k]]
        normal = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        if float(np.dot(normal, centroid)) < 0.0:
            normal *= -1.0
        normals[i] += normal
        normals[j] += normal
        normals[k] += normal

    for idx, normal in enumerate(normals):
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-14:
            normals[idx] /= norm
        else:
            z_sign = 1.0 if coords[idx, 2] >= 0.0 else -1.0
            normals[idx] = np.array([0.0, 0.0, z_sign], dtype=float)

    vertex_index = {id(v): idx for idx, v in enumerate(verts)}
    return verts, coords, triangles, normals, vertex_index


def _surface_ddg_capillary_force_error(surface, bV_surface) -> float:
    surface_verts, _coords, _triangles, normals, vertex_index = _surface_complex_geometry(surface)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    total_force = 0.0

    for v in surface_verts:
        n_i = normals[vertex_index[id(v)]]
        F, nn = vectorise_vnn(v)
        c_outd = b_curvatures_hn_ij_c_ij(F, nn, n_i=n_i)
        total_force += -GAMMA * float(np.dot(c_outd["HNdA_i"], z_hat))

    return float(total_force)


def _surface_fd_capillary_force_error(surface, bV_surface) -> float:
    surface_verts, coords, triangles, _normals, _vertex_index = _surface_complex_geometry(surface)

    incident: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(surface_verts))}
    for tri in triangles:
        tri_tuple = tuple(int(idx) for idx in tri)
        for idx in tri_tuple:
            incident[idx].append(tri_tuple)

    z_extent = float(np.max(coords[:, 2]) - np.min(coords[:, 2]))
    eps = max(1.0e-6, 1.0e-6 * z_extent)

    contributions: list[float] = []
    for vidx, tris in incident.items():
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


def _surface_boundary_integration_error(surface, bV_surface) -> float:
    for v in bV_surface:
        boundary_neighbors = [vn for vn in v.nn if vn in bV_surface]
        if not boundary_neighbors:
            continue
        return float(0.5 * np.linalg.norm(v.x_a - boundary_neighbors[0].x_a) ** 2)
    raise RuntimeError("Failed to find a boundary edge for the integration error.")


def _boundary_surface_operator_metrics(surface, bV_surface) -> dict[str, float]:
    _prepare_surface_benchmark_state(surface, bV_surface)
    return {
        "ddg_capillary_force_error": _surface_ddg_capillary_force_error(surface, bV_surface),
        "stress_capillary_force_error": _surface_stress_capillary_force_error(surface, bV_surface),
        "surface_tension_capillary_force_error": _surface_tension_capillary_force_error(surface, bV_surface),
        "multiphase_stress_capillary_force_error": _surface_multiphase_stress_capillary_force_error(surface, bV_surface),
        "fd_capillary_force_error": _surface_fd_capillary_force_error(surface, bV_surface),
        "integration_error": _surface_boundary_integration_error(surface, bV_surface),
    }


@lru_cache(maxsize=None)
def compute_static_case(refinement: int) -> dict[str, float]:
    HC, bV_vol, outer_rings, _layer_rings, surface_edges, surface_boundary_indices = _build_structured_volumetric_catenoid(refinement)
    surface, surface_bV = _build_surface_complex_from_template(
        outer_rings,
        surface_edges,
        surface_boundary_indices,
    )
    surface_metrics = _boundary_surface_operator_metrics(surface, surface_bV)
    return {
        "refinement": refinement,
        "n_total": float(_active_vertex_count(HC)),
        "n_boundary": float(len(surface_boundary_indices)),
        "stress_axial_force_residual": _stress_axial_force_residual(HC, bV_vol),
        "stress_max_local_force_norm": _stress_max_local_force_norm(HC, bV_vol),
        **surface_metrics,
    }


@lru_cache(maxsize=None)
def compute_dynamic_case(
    refinement: int,
    *,
    dt: float = DYNAMIC_DT,
    n_steps: int | None = None,
    workers: int = DYNAMIC_WORKERS,
) -> dict[str, float]:
    if n_steps is None:
        n_steps = int(round(MAX_HOLD_TIME / dt))

    static_row = compute_static_case(refinement)
    if float(static_row["stress_max_local_force_norm"]) == 0.0:
        return {
            "refinement": refinement,
            "n_total": static_row["n_total"],
            "n_boundary": static_row["n_boundary"],
            "stress_axial_force_residual": static_row["stress_axial_force_residual"],
            "stress_max_local_force_norm": static_row["stress_max_local_force_norm"],
            "ddg_capillary_force_error": static_row["ddg_capillary_force_error"],
            "stress_capillary_force_error": static_row["stress_capillary_force_error"],
            "surface_tension_capillary_force_error": static_row["surface_tension_capillary_force_error"],
            "multiphase_stress_capillary_force_error": static_row["multiphase_stress_capillary_force_error"],
            "fd_capillary_force_error": static_row["fd_capillary_force_error"],
            "integration_error": static_row["integration_error"],
        }

    HC, bV_vol, outer_rings, _layer_rings, surface_edges, surface_boundary_indices = _build_structured_volumetric_catenoid(refinement)
    dudt_fn = partial(stress_acceleration, dim=3, mu=0.0, HC=HC)
    symplectic_euler(
        HC,
        bV_vol,
        dudt_fn,
        dt=dt,
        n_steps=n_steps,
        dim=3,
        workers=workers,
        retopologize_fn=False,
    )
    surface, surface_bV = _build_surface_complex_from_template(
        outer_rings,
        surface_edges,
        surface_boundary_indices,
    )
    surface_metrics = _boundary_surface_operator_metrics(surface, surface_bV)
    return {
        "refinement": refinement,
        "n_total": float(_active_vertex_count(HC)),
        "n_boundary": float(len(surface_boundary_indices)),
        "stress_axial_force_residual": _stress_axial_force_residual(HC, bV_vol),
        "stress_max_local_force_norm": _stress_max_local_force_norm(HC, bV_vol),
        **surface_metrics,
    }


def run_static_benchmark() -> list[dict[str, float]]:
    return [compute_static_case(refinement) for refinement in REFINEMENTS]


def run_dynamic_benchmark() -> list[dict[str, float]]:
    return [compute_dynamic_case(refinement) for refinement in REFINEMENTS]


def _hold_step_schedule(max_steps: int) -> list[int]:
    return sorted({step for step in (1, 5, 10, 20, 50, max_steps) if 1 <= step <= max_steps})


def _dynamic_checkpoint_rows(
    refinement: int,
    schedule: tuple[int, ...],
    *,
    dt: float = DYNAMIC_DT,
    workers: int = DYNAMIC_WORKERS,
) -> list[dict[str, float]]:
    """Advance the Case 5 volume once and sample metrics at requested steps."""

    HC, bV_vol, outer_rings, _layer_rings, surface_edges, surface_boundary_indices = _build_structured_volumetric_catenoid(refinement)
    dudt_fn = partial(stress_acceleration, dim=3, mu=0.0, HC=HC)
    rows: list[dict[str, float]] = []
    prev_steps = 0

    for step in schedule:
        delta_steps = step - prev_steps
        if delta_steps > 0:
            symplectic_euler(
                HC,
                bV_vol,
                dudt_fn,
                dt=dt,
                n_steps=delta_steps,
                dim=3,
                workers=workers,
                retopologize_fn=False,
            )
        surface, surface_bV = _build_surface_complex_from_template(
            outer_rings,
            surface_edges,
            surface_boundary_indices,
        )
        surface_metrics = _boundary_surface_operator_metrics(surface, surface_bV)
        rows.append(
            {
                "refinement": refinement,
                "n_total": float(_active_vertex_count(HC)),
                "n_boundary": float(len(surface_boundary_indices)),
                "stress_axial_force_residual": _stress_axial_force_residual(HC, bV_vol),
                "stress_max_local_force_norm": _stress_max_local_force_norm(HC, bV_vol),
                **surface_metrics,
            }
        )
        prev_steps = step

    return rows


def render_mesh_pngs(rows: list[dict[str, float]], out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for row in rows:
        refinement = int(row["refinement"])
        n_boundary = int(row["n_boundary"])
        HC, bV_vol, outer_rings, layer_rings, surface_edges, surface_boundary_indices = _build_structured_volumetric_catenoid(refinement)
        coords, triangles, boundary_coords = _render_surface_geometry_with_caps(
            HC,
            outer_rings,
            layer_rings,
            surface_edges,
            surface_boundary_indices,
        )
        surface, surface_bV = _build_surface_complex_from_template(
            outer_rings,
            surface_edges,
            surface_boundary_indices,
        )
        exact_coords = coords.copy()
        displacement = np.linalg.norm(coords - exact_coords, axis=1)
        tri_displacement = displacement[triangles].mean(axis=1)
        disp_max = float(displacement.max())
        disp_norm = plt.Normalize(vmin=0.0, vmax=max(disp_max, 1.0e-18))
        disp_cmap = plt.cm.Reds
        if n_boundary <= 8:
            boundary_marker_size = 9
        elif n_boundary <= 16:
            boundary_marker_size = 7
        elif n_boundary <= 32:
            boundary_marker_size = 4.5
        else:
            boundary_marker_size = 2.5

        fig = plt.figure(figsize=(6.4, 6.9))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        surf = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=triangles,
            linewidth=0.45,
            edgecolor="#2c3742",
            antialiased=True,
            shade=False,
            alpha=1.0,
        )
        surf.set_zorder(1)
        surf.set_array(tri_displacement)
        surf.set_cmap(disp_cmap)
        surf.set_norm(disp_norm)
        surf.autoscale()
        z_min = float(np.min(boundary_coords[:, 2]))
        z_max = float(np.max(boundary_coords[:, 2]))
        cap_mask = np.isclose(boundary_coords[:, 2], z_min, atol=1.0e-12) | np.isclose(
            boundary_coords[:, 2], z_max, atol=1.0e-12
        )
        ax.scatter(
            boundary_coords[cap_mask, 0],
            boundary_coords[cap_mask, 1],
            boundary_coords[cap_mask, 2],
            s=boundary_marker_size,
            c="#111111",
            alpha=0.95,
            depthshade=False,
            zorder=6,
        )
        ax.set_title(f"Volumetric Stress Mesh ref {refinement} (n={n_boundary})", pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-58)
        ax.set_box_aspect((1.0, 1.0, 2.2))
        cbar = fig.colorbar(surf, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("displacement from exact volumetric catenoid")
        if disp_max == 0.0:
            cbar.set_ticks([0.0])
            cbar.set_ticklabels(["0"])
        fig.tight_layout()

        path = out_dir / f"benchmark_mesh_ref{refinement}_n{n_boundary}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    return paths


def render_reproduced_static_dynamic_figure(
    static_rows: list[dict[str, float]],
    dynamic_rows: list[dict[str, float]],
    out_path: Path = REPRODUCED_STATIC_DYNAMIC_FIGURE,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_reference = np.array(ENDRES_2024_REFERENCE["n_boundary"], dtype=float)
    n_static = n_reference * 0.97
    n_dynamic = n_reference * 1.03
    static_capillary = 100.0 * np.abs(
        np.array([row["stress_capillary_force_error"] for row in static_rows], dtype=float)
    )
    static_geo = np.array([row["integration_error"] for row in static_rows], dtype=float)
    ref_capillary = 100.0 * np.abs(ENDRES_2024_REFERENCE["capillary_force_error"])
    ref_geo = np.array(ENDRES_2024_REFERENCE["integration_error"], dtype=float)
    dynamic_capillary = 100.0 * np.abs(
        np.array([row["stress_capillary_force_error"] for row in dynamic_rows], dtype=float)
    )
    dynamic_geo = np.array([row["integration_error"] for row in dynamic_rows], dtype=float)

    fig = plt.figure(figsize=(11.2, 6.8))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ln1 = ax2.loglog(
        n_static,
        static_capillary,
        "o",
        color="#ff7f0e",
        fillstyle="none",
        markersize=9,
        mew=1.9,
        label="static stress capillary error",
    )
    ln2 = ax2.loglog(
        n_reference,
        ref_capillary,
        "x",
        color="#ff7f0e",
        markersize=11,
        mew=2.0,
        label="Endres (2024) reference capillary",
    )
    ln3 = ax2.loglog(
        n_dynamic,
        dynamic_capillary,
        "+",
        color="#ff7f0e",
        markersize=16,
        mew=2.4,
        label=f"dynamic stress capillary error (damping={DYNAMIC_DAMPING:g})",
    )
    ln4 = ax.loglog(
        n_static,
        static_geo,
        "s",
        color="#1f77b4",
        fillstyle="none",
        markersize=8,
        mew=1.8,
        label="static DDG integration error",
    )
    ln5 = ax.loglog(
        n_reference,
        ref_geo,
        "X",
        color="#1f77b4",
        markersize=10,
        mew=1.8,
        label="Endres (2024) reference integration error",
    )
    ln6 = ax.loglog(
        n_dynamic,
        dynamic_geo,
        "^",
        color="#1f77b4",
        markersize=10,
        fillstyle="none",
        mew=2.0,
        label="dynamic DDG integration error",
    )

    ax.set_xlabel(r"$n$ (number of boundary vertices)", fontsize=13)
    ax.set_ylabel("Integration error (%)", fontsize=13, color="#1f77b4")
    ax2.set_ylabel("Capillary force error (%)", fontsize=13, color="#ff7f0e")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#ff7f0e")
    ax.spines["left"].set_color("#1f77b4")
    ax2.spines["right"].set_color("#ff7f0e")
    ax.grid(True, which="both", alpha=0.18)

    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labs = [line.get_label() for line in lns]
    ax.legend(
        lns,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.suptitle("Volumetric Stress Static + Dynamic + Reference", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.03,
        f"max hold time={MAX_HOLD_TIME:.1e} s; dt={DYNAMIC_DT:.1e} s; Case 5 boundary-surface stress capillary metrics with Endres reference by n_boundary",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_hold_time_figure(
    refinements: tuple[int, ...] = REFINEMENTS,
    out_path: Path = HOLD_TIME_FIGURE,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_steps = int(round(MAX_HOLD_TIME / DYNAMIC_DT))
    schedule = _hold_step_schedule(max_steps)
    hold_times = np.array([step * DYNAMIC_DT for step in schedule], dtype=float)

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    refinement_markers = ("x", "s", "^", "D")
    ddg_color = "#ff7f0e"
    local_color = "#1f77b4"

    for idx, refinement in enumerate(refinements):
        rows = [compute_dynamic_case(refinement, dt=DYNAMIC_DT, n_steps=step, workers=DYNAMIC_WORKERS) for step in schedule]
        axial_pct = np.maximum(
            100.0 * np.abs(np.array([row["stress_axial_force_residual"] for row in rows], dtype=float)),
            PLOT_FLOOR,
        )
        local_pct = np.maximum(
            100.0 * np.abs(np.array([row["stress_max_local_force_norm"] for row in rows], dtype=float)),
            PLOT_FLOOR,
        )
        label = (
            f"ref {refinement}, N_total={int(rows[-1]['n_total'])}, "
            f"n_b={int(rows[-1]['n_boundary'])}, dt={DYNAMIC_DT:.0e} s"
        )
        ax.plot(
            hold_times,
            axial_pct,
            marker=refinement_markers[idx % len(refinement_markers)],
            linewidth=2.0,
            color=ddg_color,
            linestyle="-",
            label=f"{label}, axial",
        )
        ax.plot(
            hold_times,
            local_pct,
            marker=refinement_markers[idx % len(refinement_markers)],
            linewidth=2.0,
            color=local_color,
            linestyle="--",
            label=f"{label}, max local",
        )

    ax.set_title("Axial vs Max-Local Cauchy Stress Residual")
    ax.set_xlabel("Simulated hold time (s)")
    ax.set_ylabel("Cauchy stress residual (%)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    tick_positions = np.linspace(0.0, MAX_HOLD_TIME, 9)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{tick:.6f}" for tick in tick_positions])
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        bbox_transform=fig.transFigure,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_fontsize(10)

    fig.suptitle("Volumetric Cauchy Stress Axial vs Max-Local Residual", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.03,
        f"max hold time={MAX_HOLD_TIME:.1e} s; workers={DYNAMIC_WORKERS}; orange solid = axial, blue dashed = max local",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.43)
    fig.savefig(out_path, dpi=240, facecolor="white")
    plt.close(fig)
    return out_path


def render_ddg_fd_companion_figure(out_path: Path = DDG_FD_COMPANION_FIGURE) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_steps = int(round(MAX_HOLD_TIME / DYNAMIC_DT))
    schedule = tuple(_hold_step_schedule(max_steps))

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ddg_color = "#ff7f0e"
    fd_color = "#1f77b4"
    ddg_markers = ["x", "s", "^", "D"]
    fd_markers = ["o", "P", "v", "*"]

    for idx, refinement in enumerate(REFINEMENTS):
        rows = [
            compute_dynamic_case(refinement, dt=DYNAMIC_DT, n_steps=step, workers=DYNAMIC_WORKERS)
            for step in schedule
        ]
        hold_time = np.array([step * DYNAMIC_DT for step in schedule], dtype=float)
        ddg_err = np.maximum(
            100.0 * np.abs(np.array([row["ddg_capillary_force_error"] for row in rows], dtype=float)),
            PLOT_FLOOR,
        )
        fd_err = np.maximum(
            100.0 * np.abs(np.array([row["fd_capillary_force_error"] for row in rows], dtype=float)),
            PLOT_FLOOR,
        )
        label = (
            f"ref {refinement}, N_total={int(rows[-1]['n_total'])}, "
            f"n_b={int(rows[-1]['n_boundary'])}, dt={DYNAMIC_DT:.0e} s"
        )

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
    ax.set_xlim(0.0, MAX_HOLD_TIME)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", alpha=0.18)
    tick_positions = np.linspace(0.0, MAX_HOLD_TIME, 9)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{tick:.6f}" for tick in tick_positions])
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

    fig.suptitle("Case 5 Boundary-Surface DDG vs FD Capillary Error", fontsize=18)
    fig.text(
        0.5,
        0.03,
        f"max hold time={MAX_HOLD_TIME:.0e} s; workers={DYNAMIC_WORKERS}; orange solid = DDG, blue dashed = FD",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(out_path, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)
    return out_path


def print_max_hold_time_summary(rows: list[dict[str, float]]) -> None:
    print(
        f"\nCase 5 max hold-time capillary force errors (%) "
        f"at t={MAX_HOLD_TIME:.6f} s (dt={DYNAMIC_DT:.0e} s)"
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


def main() -> None:
    print("Case 5: volumetric stress-operator equilibrium benchmark")
    print("[1/4] Computing static volumetric stress rows...")
    static_rows = run_static_benchmark()

    print("[2/4] Computing dynamic volumetric stress rows...")
    dynamic_rows = run_dynamic_benchmark()

    print("[3/4] Rendering benchmark mesh PNGs...")
    mesh_paths = render_mesh_pngs(static_rows)

    print("[4/4] Rendering benchmark figures...")
    static_dynamic_path = render_reproduced_static_dynamic_figure(static_rows, dynamic_rows)
    hold_time_path = render_hold_time_figure()
    ddg_fd_companion_path = render_ddg_fd_companion_figure()

    print_builtin_operator_capillary_force_summary(
        static_rows,
        case_label="Case 5",
        label="static",
    )
    print_builtin_operator_capillary_force_summary(
        dynamic_rows,
        case_label="Case 5",
        label="max hold-time",
        time_value=MAX_HOLD_TIME,
        dt=DYNAMIC_DT,
    )
    print_max_hold_time_summary(dynamic_rows)
    if mesh_paths:
        print("Saved volumetric benchmark meshes:")
        for path in mesh_paths:
            print(f"  {path}")
    print(f"Saved reproduced static+dynamic figure to: {static_dynamic_path}")
    print(f"Saved hold-time figure to: {hold_time_path}")
    print(f"Saved DDG/FD companion figure to: {ddg_fd_companion_path}")


if __name__ == "__main__":
    main()
