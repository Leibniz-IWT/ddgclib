"""Minimal HC benchmark for Heron surface force vs pressure closures.

The case is deliberately static and small:

* build closed spherical surface meshes as ``hyperct.Complex`` objects,
* compute the ddgclib Heron curvature force ``F_Heron = -gamma HN dA``,
* compare it with the analytical Young-Laplace pressure
  ``Delta p = 2 gamma / R``,
* compare two pressure closures against the same force balance:
  a weak-compressible linear EOS scalar and an incompressible projection
  scalar.

This is not a full acoustic or moving-interface run.  It isolates the part
discussed in the meeting: whether the pressure closure balances the HC/Heron
surface force, and how much time is spent in the Heron kernel versus the
pressure closure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    candidate
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents)
    if (candidate / "ddgclib").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import Delaunay

from hyperct import Complex
from ddgclib._curvatures_heron import hndA_i


BASE_ICOSAHEDRON_VERTICES = np.asarray(
    [
        (-1.0, (1.0 + math.sqrt(5.0)) / 2.0, 0.0),
        (1.0, (1.0 + math.sqrt(5.0)) / 2.0, 0.0),
        (-1.0, -(1.0 + math.sqrt(5.0)) / 2.0, 0.0),
        (1.0, -(1.0 + math.sqrt(5.0)) / 2.0, 0.0),
        (0.0, -1.0, (1.0 + math.sqrt(5.0)) / 2.0),
        (0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0),
        (0.0, -1.0, -(1.0 + math.sqrt(5.0)) / 2.0),
        (0.0, 1.0, -(1.0 + math.sqrt(5.0)) / 2.0),
        ((1.0 + math.sqrt(5.0)) / 2.0, 0.0, -1.0),
        ((1.0 + math.sqrt(5.0)) / 2.0, 0.0, 1.0),
        (-(1.0 + math.sqrt(5.0)) / 2.0, 0.0, -1.0),
        (-(1.0 + math.sqrt(5.0)) / 2.0, 0.0, 1.0),
    ],
    dtype=float,
)

BASE_ICOSAHEDRON_FACES = np.asarray(
    [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ],
    dtype=int,
)


def _normalize_to_radius(points: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot normalize a zero-length icosphere point")
    return radius * points / norms[:, None]


def icosphere(subdivisions: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangular faces for a radius-R icosphere."""
    vertices = _normalize_to_radius(BASE_ICOSAHEDRON_VERTICES.copy(), radius)
    faces = BASE_ICOSAHEDRON_FACES.copy()

    for _ in range(int(subdivisions)):
        midpoint_cache: dict[tuple[int, int], int] = {}
        vertex_list = [vertices[i].copy() for i in range(vertices.shape[0])]
        new_faces = []

        def midpoint_index(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            midpoint = 0.5 * (vertices[i] + vertices[j])
            midpoint = radius * midpoint / np.linalg.norm(midpoint)
            vertex_list.append(midpoint)
            idx = len(vertex_list) - 1
            midpoint_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint_index(int(a), int(b))
            bc = midpoint_index(int(b), int(c))
            ca = midpoint_index(int(c), int(a))
            new_faces.extend(
                [
                    (int(a), ab, ca),
                    (int(b), bc, ab),
                    (int(c), ca, bc),
                    (ab, bc, ca),
                ]
            )

        vertices = np.asarray(vertex_list, dtype=float)
        faces = np.asarray(new_faces, dtype=int)

    return vertices, faces


def build_surface_hc(points: np.ndarray, faces: np.ndarray) -> tuple[Complex, list]:
    """Build an HC surface graph from triangle connectivity."""
    HC = Complex(3, domain=None)
    vertices = [HC.V[tuple(float(x) for x in point)] for point in points]

    for a, b, c in faces:
        va, vb, vc = vertices[int(a)], vertices[int(b)], vertices[int(c)]
        va.connect(vb)
        vb.connect(vc)
        vc.connect(va)

    for vertex in vertices:
        vertex.u = np.zeros(3, dtype=float)
        vertex.p = 0.0
        vertex.m = 1.0
        vertex.phase = 0
        vertex.boundary = False
        vertex.is_interface = True

    return HC, vertices


def heron_force_area(vertices: list, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ddgclib Heron force and Heron dual area in one kernel pass."""
    forces = np.zeros((len(vertices), 3), dtype=float)
    areas = np.zeros(len(vertices), dtype=float)
    for i, vertex in enumerate(vertices):
        hnda_i, area_i = hndA_i(vertex)
        forces[i] = -float(gamma) * hnda_i[:3]
        areas[i] = float(area_i)
    return forces, areas


def linear_eos_pressure(volume: float, reference_volume: float, bulk_modulus: float) -> float:
    """Weak-compressible pressure, positive for compressed volume."""
    return float(bulk_modulus) * (float(reference_volume) / float(volume) - 1.0)


def projection_pressure(forces: np.ndarray, pressure_area_vectors: np.ndarray) -> float:
    """One-scalar incompressibility projection for the static sphere balance."""
    denom = float(np.sum(pressure_area_vectors * pressure_area_vectors))
    if denom <= 0.0:
        return 0.0
    return -float(np.sum(pressure_area_vectors * forces)) / denom


def residual_metrics(forces: np.ndarray, pressure_area_vectors: np.ndarray, pressure: float) -> dict[str, float]:
    residual = forces + float(pressure) * pressure_area_vectors
    force_norm = np.linalg.norm(forces, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    rms_force = math.sqrt(float(np.mean(force_norm * force_norm)))
    rms_residual = math.sqrt(float(np.mean(residual_norm * residual_norm)))
    rel_rms = rms_residual / max(rms_force, 1.0e-300)
    max_rel = float(np.max(residual_norm) / max(float(np.max(force_norm)), 1.0e-300))
    return {
        "residual_rms_n": rms_residual,
        "residual_rel_rms": rel_rms,
        "residual_rel_max": max_rel,
    }


def time_repeated(count: int, callback):
    """Return ``(last_result, seconds_per_call)`` for a repeated callback."""
    start = time.perf_counter()
    result = None
    for _ in range(max(1, int(count))):
        result = callback()
    elapsed = time.perf_counter() - start
    return result, elapsed / max(1, int(count))


def run_one(
    subdivisions: int,
    radius: float,
    gamma: float,
    bulk_modulus: float,
    fheron_repeats: int,
    closure_repeats: int,
) -> dict[str, float]:
    build_start = time.perf_counter()
    points, faces = icosphere(subdivisions, radius)
    _HC, vertices = build_surface_hc(points, faces)
    build_s = time.perf_counter() - build_start

    (forces, areas), fheron_s = time_repeated(
        fheron_repeats,
        lambda: heron_force_area(vertices, gamma),
    )

    normals = points / np.linalg.norm(points, axis=1)[:, None]
    pressure_area_vectors = areas[:, None] * normals
    theory_pressure = 2.0 * float(gamma) / float(radius)
    theory_forces = -theory_pressure * pressure_area_vectors
    heron_error = forces - theory_forces
    heron_rms = math.sqrt(float(np.mean(np.sum(heron_error * heron_error, axis=1))))
    theory_rms = math.sqrt(float(np.mean(np.sum(theory_forces * theory_forces, axis=1))))

    reference_volume = 4.0 * math.pi * float(radius) ** 3 / 3.0
    compressed_volume = reference_volume / (1.0 + theory_pressure / float(bulk_modulus))
    eos_pressure, eos_s = time_repeated(
        closure_repeats,
        lambda: linear_eos_pressure(compressed_volume, reference_volume, bulk_modulus),
    )
    proj_pressure, projection_s = time_repeated(
        closure_repeats,
        lambda: projection_pressure(forces, pressure_area_vectors),
    )

    theory_residual = residual_metrics(forces, pressure_area_vectors, theory_pressure)
    eos_residual = residual_metrics(forces, pressure_area_vectors, eos_pressure)
    projection_residual = residual_metrics(forces, pressure_area_vectors, proj_pressure)

    edges = {
        tuple(sorted((int(i), int(j))))
        for tri in faces
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    }
    mean_edge = float(np.mean([np.linalg.norm(points[i] - points[j]) for i, j in edges]))
    net_force = np.sum(forces, axis=0)

    return {
        "subdivisions": int(subdivisions),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "mean_edge_m": mean_edge,
        "build_ms": 1.0e3 * build_s,
        "fheron_ms": 1.0e3 * fheron_s,
        "eos_us": 1.0e6 * eos_s,
        "projection_us": 1.0e6 * projection_s,
        "theory_pressure_pa": theory_pressure,
        "eos_pressure_pa": float(eos_pressure),
        "projection_pressure_pa": float(proj_pressure),
        "eos_volume_strain": (reference_volume - compressed_volume) / reference_volume,
        "heron_vs_theory_rel_rms": heron_rms / max(theory_rms, 1.0e-300),
        "theory_residual_rel_rms": theory_residual["residual_rel_rms"],
        "eos_residual_rel_rms": eos_residual["residual_rel_rms"],
        "projection_residual_rel_rms": projection_residual["residual_rel_rms"],
        "projection_residual_rel_max": projection_residual["residual_rel_max"],
        "net_fheron_force_n": float(np.linalg.norm(net_force)),
    }


def sphere_volume(radius: float) -> float:
    return 4.0 * math.pi * float(radius) ** 3 / 3.0


def radius_from_volume(volume: float) -> float:
    return (3.0 * float(volume) / (4.0 * math.pi)) ** (1.0 / 3.0)


def mesh_volume(points: np.ndarray, faces: np.ndarray) -> float:
    tetra_volumes = [
        abs(float(np.dot(points[int(a)], np.cross(points[int(b)], points[int(c)])))) / 6.0
        for a, b, c in faces
    ]
    return float(np.sum(tetra_volumes))


def shape_amplitude_from_aspect_ratio(aspect_ratio: float) -> float:
    """Return l=2 mode amplitude giving polar/equatorial AR."""
    ar = float(aspect_ratio)
    if ar <= 1.0:
        raise ValueError("aspect_ratio must be greater than 1")
    return (ar - 1.0) / (1.0 + 0.5 * ar)


def rayleigh_lamb_frequency(l_mode: int, gamma: float, rho: float, radius: float) -> tuple[float, float, float]:
    """Return angular frequency, frequency, and period for Rayleigh-Lamb mode l."""
    l_val = int(l_mode)
    if l_val < 2:
        raise ValueError("Rayleigh-Lamb shape modes require l >= 2")
    omega = math.sqrt(l_val * (l_val - 1) * (l_val + 2) * float(gamma) / (float(rho) * float(radius) ** 3))
    frequency = omega / (2.0 * math.pi)
    period = 1.0 / frequency
    return omega, frequency, period


def volumetric_star_tet_segments(points: np.ndarray, faces: np.ndarray, *, max_segments: int = 420) -> np.ndarray:
    """Wireframe segments for the center-to-surface tetrahedral star mesh."""
    center = np.mean(points, axis=0)
    edge_keys = {
        tuple(sorted((int(i), int(j))))
        for tri in faces
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    }
    surface_edges = [(points[i], points[j]) for i, j in sorted(edge_keys)]
    radial_step = max(1, int(math.ceil(points.shape[0] / max(1, max_segments // 3))))
    radial_edges = [(center, points[i]) for i in range(0, points.shape[0], radial_step)]
    segments = np.asarray(surface_edges + radial_edges, dtype=float)
    if segments.shape[0] > max_segments:
        keep = np.linspace(0, segments.shape[0] - 1, max_segments, dtype=int)
        segments = segments[keep]
    return segments


def tet_edge_indices(tets: np.ndarray, *, max_edges: int = 360) -> np.ndarray:
    """Representative unique edge index pairs from a tetrahedral volume mesh."""
    edge_keys: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=int):
        a, b, c, d = [int(index) for index in tet]
        edge_keys.update(
            tuple(sorted(edge))
            for edge in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))
        )
    edges = sorted(edge_keys)
    if len(edges) > max_edges:
        keep = np.linspace(0, len(edges) - 1, max_edges, dtype=int)
        edges = [edges[int(index)] for index in keep]
    return np.asarray(edges, dtype=int).reshape((-1, 2))


def tet_edge_segments(points: np.ndarray, tets: np.ndarray, *, max_segments: int = 360) -> np.ndarray:
    """Representative unique edge segments from a tetrahedral volume mesh."""
    edges = tet_edge_indices(tets, max_edges=max_segments)
    if edges.size == 0:
        return np.empty((0, 2, 3), dtype=float)
    return np.asarray(points[edges], dtype=float)


def fheron_pressure_for_radius(subdivisions: int, radius: float, gamma: float) -> tuple[float, float]:
    points, faces = icosphere(subdivisions, radius)
    _HC, vertices = build_surface_hc(points, faces)
    forces, areas = heron_force_area(vertices, gamma)
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    pressure_area_vectors = areas[:, None] * normals
    return projection_pressure(forces, pressure_area_vectors), mesh_volume(points, faces)


def vertex_area_vectors(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    area_vectors = np.zeros_like(points, dtype=float)
    for a, b, c in faces:
        a_i, b_i, c_i = int(a), int(b), int(c)
        pa, pb, pc = points[a_i], points[b_i], points[c_i]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        centroid = (pa + pb + pc) / 3.0
        if float(np.dot(area_vec, centroid)) < 0.0:
            area_vec = -area_vec
        share = area_vec / 3.0
        area_vectors[a_i] += share
        area_vectors[b_i] += share
        area_vectors[c_i] += share
    return area_vectors


def vertex_scalar_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.zeros(points.shape[0], dtype=float)
    for a, b, c in faces:
        a_i, b_i, c_i = int(a), int(b), int(c)
        tri_area = 0.5 * float(np.linalg.norm(np.cross(points[b_i] - points[a_i], points[c_i] - points[a_i])))
        share = tri_area / 3.0
        areas[a_i] += share
        areas[b_i] += share
        areas[c_i] += share
    return areas


def vertex_star_tet_masses(points: np.ndarray, faces: np.ndarray, rho: float) -> np.ndarray:
    """Lump closed star-tet volume mass to surface vertices."""
    masses = np.zeros(points.shape[0], dtype=float)
    center = np.mean(points, axis=0)
    shifted = points - center[None, :]
    for a, b, c in faces:
        a_i, b_i, c_i = int(a), int(b), int(c)
        tet_volume = abs(float(np.dot(shifted[a_i], np.cross(shifted[b_i], shifted[c_i])))) / 6.0
        share = float(rho) * tet_volume / 3.0
        masses[a_i] += share
        masses[b_i] += share
        masses[c_i] += share
    return masses


def fheron_pressure_for_points(points: np.ndarray, faces: np.ndarray, gamma: float) -> tuple[float, float]:
    _HC, vertices = build_surface_hc(points, faces)
    forces, _areas = heron_force_area(vertices, gamma)
    return projection_pressure(forces, vertex_area_vectors(points, faces)), mesh_volume(points, faces)


def fheron_forces_for_points(points: np.ndarray, faces: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    _HC, vertices = build_surface_hc(points, faces)
    forces, _areas = heron_force_area(vertices, gamma)
    area_vectors = vertex_area_vectors(points, faces)
    pressure_equivalent = projection_pressure(forces, area_vectors)
    return forces, area_vectors, pressure_equivalent, mesh_volume(points, faces)


def local_tet_volume_gradients(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return star-tet volumes and dV_tet/dx_i for each surface face."""
    volumes = np.zeros(faces.shape[0], dtype=float)
    gradients = np.zeros((faces.shape[0], 3, 3), dtype=float)
    for tet_idx, (a, b, c) in enumerate(faces):
        a_i, b_i, c_i = int(a), int(b), int(c)
        pa, pb, pc = points[a_i], points[b_i], points[c_i]
        triple = float(np.dot(pa, np.cross(pb, pc)))
        sign = 1.0 if triple >= 0.0 else -1.0
        volumes[tet_idx] = abs(triple) / 6.0
        gradients[tet_idx, 0] = sign * np.cross(pb, pc) / 6.0
        gradients[tet_idx, 1] = sign * np.cross(pc, pa) / 6.0
        gradients[tet_idx, 2] = sign * np.cross(pa, pb) / 6.0
    return volumes, gradients


def local_volume_matrix(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volumes, gradients = local_tet_volume_gradients(points, faces)
    matrix = np.zeros((faces.shape[0], points.shape[0] * 3), dtype=float)
    for tet_idx, face in enumerate(faces):
        for local_idx, vertex_idx in enumerate(face):
            start = 3 * int(vertex_idx)
            matrix[tet_idx, start:start + 3] += gradients[tet_idx, local_idx]
    return volumes, matrix


def local_pressure_forces(points: np.ndarray, faces: np.ndarray, pressures: np.ndarray) -> np.ndarray:
    _volumes, matrix = local_volume_matrix(points, faces)
    return np.asarray(matrix.T @ np.asarray(pressures, dtype=float), dtype=float).reshape(points.shape)


def _solve_regularized(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    scale = max(float(np.mean(np.abs(np.diag(system)))) if system.size else 0.0, 1.0e-30)
    regularized = system + (1.0e-12 * scale) * np.eye(system.shape[0])
    try:
        return np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized, rhs, rcond=1.0e-12)[0]


def _mass_inverse_dof(masses: np.ndarray) -> np.ndarray:
    return np.repeat(1.0 / np.maximum(masses, 1.0e-300), 3)


def _local_volume_stiffness(matrix: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_mass_dof = _mass_inverse_dof(masses)
    stiffness = (matrix * inv_mass_dof[None, :]) @ matrix.T
    return stiffness, inv_mass_dof


def build_ball_tet_mesh(
    subdivisions: int,
    radius: float,
    *,
    shell_radii: tuple[float, ...] = (0.45, 0.75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Connected tetrahedral ball mesh with active interior and surface vertices."""
    unit_points, surface_faces = icosphere(subdivisions, 1.0)
    points = [np.zeros((1, 3), dtype=float)]
    for shell_radius in (*shell_radii, 1.0):
        points.append(unit_points * (float(shell_radius) * float(radius)))
    volume_points = np.vstack(points)
    n_surface = unit_points.shape[0]
    surface_offset = 1 + len(shell_radii) * n_surface
    surface_indices = np.arange(surface_offset, surface_offset + n_surface, dtype=int)
    surface_faces_global = surface_faces + surface_offset

    triangulation = Delaunay(volume_points, qhull_options="Qbb Qc")
    raw_tets = np.asarray(triangulation.simplices, dtype=int)
    valid = np.all(raw_tets < volume_points.shape[0], axis=1)
    raw_tets = raw_tets[valid]
    volumes = tet_cell_volumes(volume_points, raw_tets)
    positive = volumes[volumes > 0.0]
    min_volume = 0.1 * float(np.median(positive)) if positive.size else 0.0
    tets = raw_tets[volumes > min_volume]
    return volume_points, tets, surface_indices, surface_faces, surface_faces_global, unit_points


def tet_cell_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    triple = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    return np.abs(triple) / 6.0


def tet_volume_matrix(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volumes = np.zeros(tets.shape[0], dtype=float)
    matrix = np.zeros((tets.shape[0], points.shape[0] * 3), dtype=float)
    for tet_idx, tet in enumerate(tets):
        i0, i1, i2, i3 = [int(i) for i in tet]
        p0, p1, p2, p3 = points[i0], points[i1], points[i2], points[i3]
        triple = float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))
        sign = 1.0 if triple >= 0.0 else -1.0
        volumes[tet_idx] = abs(triple) / 6.0
        g1 = sign * np.cross(p2 - p0, p3 - p0) / 6.0
        g2 = sign * np.cross(p3 - p0, p1 - p0) / 6.0
        g3 = sign * np.cross(p1 - p0, p2 - p0) / 6.0
        g0 = -(g1 + g2 + g3)
        for vertex_idx, gradient in ((i0, g0), (i1, g1), (i2, g2), (i3, g3)):
            start = 3 * vertex_idx
            matrix[tet_idx, start:start + 3] += gradient
    return volumes, matrix


def tet_pressure_forces(points: np.ndarray, tets: np.ndarray, pressures: np.ndarray) -> np.ndarray:
    _volumes, matrix = tet_volume_matrix(points, tets)
    return np.asarray(matrix.T @ np.asarray(pressures, dtype=float), dtype=float).reshape(points.shape)


def lump_tet_masses(n_vertices: int, tets: np.ndarray, volumes: np.ndarray, rho: float) -> np.ndarray:
    masses = np.zeros(n_vertices, dtype=float)
    for tet, volume in zip(tets, volumes):
        share = float(rho) * float(volume) / 4.0
        for vertex_idx in tet:
            masses[int(vertex_idx)] += share
    return np.maximum(masses, 1.0e-18)


def deform_l2_volume_points(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    radius: float,
    amplitude: float,
    target_volume: float,
) -> np.ndarray:
    deformed = points.copy()
    radii = np.linalg.norm(points, axis=1)
    moving = radii > 1.0e-30
    dirs = np.zeros_like(points)
    dirs[moving] = points[moving] / radii[moving, None]
    mu = dirs[:, 2]
    p2 = 0.5 * (3.0 * mu * mu - 1.0)
    scale = 1.0 + float(amplitude) * p2
    if np.any(scale[moving] <= 0.0):
        raise ValueError("shape-mode amplitude creates non-positive radius")
    deformed[moving] = dirs[moving] * (radii[moving] * scale[moving])[:, None]
    current_volume = float(np.sum(tet_cell_volumes(deformed, tets)))
    if current_volume > 0.0:
        deformed *= (float(target_volume) / current_volume) ** (1.0 / 3.0)
    return deformed


def projected_l2_shape_points(
    unit_points: np.ndarray,
    faces: np.ndarray,
    *,
    radius: float,
    amplitude: float,
    target_volume: float,
) -> tuple[np.ndarray, float]:
    mu = unit_points[:, 2]
    p2 = 0.5 * (3.0 * mu * mu - 1.0)
    raw_radius = float(radius) * (1.0 + float(amplitude) * p2)
    if np.any(raw_radius <= 0.0):
        raise ValueError("shape-mode amplitude creates non-positive radius")
    points = unit_points * raw_radius[:, None]
    current_volume = mesh_volume(points, faces)
    scale = (float(target_volume) / max(current_volume, 1.0e-300)) ** (1.0 / 3.0)
    points *= scale
    return points, mesh_volume(points, faces)


def compressible_l2_shape_points(
    unit_points: np.ndarray,
    faces: np.ndarray,
    *,
    radius: float,
    amplitude: float,
    target_shape_volume: float,
    isotropic_radius_scale: float,
) -> tuple[np.ndarray, float]:
    """Shape-mode points with EOS isotropic volume scale.

    The l=2 shape itself is first projected to preserve the reference shape
    volume.  The compressible EOS response is then applied as a uniform scale.
    This keeps the requested aspect ratio independent from the EOS breathing.
    """
    points, _shape_volume = projected_l2_shape_points(
        unit_points,
        faces,
        radius=radius,
        amplitude=amplitude,
        target_volume=target_shape_volume,
    )
    points *= float(isotropic_radius_scale)
    return points, mesh_volume(points, faces)


def eos_radius_scale_from_pressure(pressure: float, bulk_modulus: float) -> float:
    """Linear EOS radius scale for fixed mass under a scalar pressure."""
    volume_denominator = 1.0 + float(pressure) / float(bulk_modulus)
    if volume_denominator <= 0.0:
        raise ValueError("EOS pressure would create non-positive volume")
    return float(volume_denominator ** (-1.0 / 3.0))


def fit_l2_shape_amplitude(points: np.ndarray, faces: np.ndarray) -> float:
    """Fit the l=2 P2(cos theta) amplitude from a generated surface mesh."""
    radii = np.linalg.norm(points, axis=1)
    valid = radii > 1.0e-30
    if not np.any(valid):
        return 0.0
    dirs = points[valid] / radii[valid, None]
    mu = dirs[:, 2]
    p2 = 0.5 * (3.0 * mu * mu - 1.0)
    equivalent_radius = radius_from_volume(mesh_volume(points, faces))
    y = radii[valid] / max(equivalent_radius, 1.0e-300) - 1.0
    design = np.column_stack([np.ones_like(p2), p2])
    _offset, amplitude = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(amplitude)


def run_timeseries(
    *,
    subdivisions: int,
    radius: float,
    gamma: float,
    bulk_modulus: float,
    pressure_drive_fraction: float,
    n_steps: int,
    t_final: float,
) -> list[dict[str, float]]:
    """Two-path sphere run: weak-compressible EOS vs incompressible projection.

    The physical default uses only the capillary pressure p = 2 gamma / R as
    the EOS pressure.  ``pressure_drive_fraction`` is retained as an optional
    diagnostic perturbation, but its default is zero.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if not 0.0 <= pressure_drive_fraction < 0.95:
        raise ValueError("pressure_drive_fraction must be in [0, 0.95)")

    volume0 = sphere_volume(radius)
    points0, faces0 = icosphere(subdivisions, radius)
    mesh_volume0 = mesh_volume(points0, faces0)
    pressure0 = 2.0 * float(gamma) / float(radius)
    times = np.linspace(0.0, float(t_final), int(n_steps))
    rows = []

    for t in times:
        phase = math.sin(2.0 * math.pi * float(t) / float(t_final))
        extra_pressure_drive = float(pressure_drive_fraction) * float(bulk_modulus) * phase
        radius_com = float(radius)
        capillary_pressure = pressure0
        eos_pressure = pressure0 + extra_pressure_drive
        for _iteration in range(16):
            capillary_pressure = 2.0 * float(gamma) / radius_com
            eos_pressure = capillary_pressure + extra_pressure_drive
            next_radius = float(radius) * eos_radius_scale_from_pressure(eos_pressure, bulk_modulus)
            if abs(next_radius - radius_com) <= 1.0e-14 * float(radius):
                radius_com = next_radius
                break
            radius_com = next_radius
        capillary_pressure = 2.0 * float(gamma) / radius_com
        eos_pressure = capillary_pressure + extra_pressure_drive
        volume_com = volume0 * (radius_com / float(radius)) ** 3
        radius_incom = float(radius)

        fheron_com, mesh_volume_com = fheron_pressure_for_radius(subdivisions, radius_com, gamma)
        fheron_incom, mesh_volume_incom = fheron_pressure_for_radius(subdivisions, radius_incom, gamma)
        fheron_analytical_com = 2.0 * float(gamma) / radius_com
        fheron_analytical_incom = 2.0 * float(gamma) / float(radius)
        volume_analytical_com = mesh_volume0 * (radius_com / float(radius)) ** 3
        volume_analytical_incom = mesh_volume0

        rows.append(
            {
                "t": float(t),
                "capillary_pressure_pa": float(capillary_pressure),
                "extra_pressure_drive_pa": float(extra_pressure_drive),
                "eos_pressure_pa": float(eos_pressure),
                "pressure_drive_pa": float(extra_pressure_drive),
                "radius_com_m": float(radius_com),
                "radius_incom_m": float(radius_incom),
                "fheron_com_pa": float(fheron_com),
                "fheron_incom_pa": float(fheron_incom),
                "fheron_analytical_pa": float(fheron_analytical_com),
                "fheron_analytical_com_pa": float(fheron_analytical_com),
                "fheron_analytical_incom_pa": float(fheron_analytical_incom),
                "vol_com_m3": float(mesh_volume_com),
                "vol_incom_m3": float(mesh_volume_incom),
                "vol_analytical_m3": float(volume_analytical_com),
                "vol_analytical_com_m3": float(volume_analytical_com),
                "vol_analytical_incom_m3": float(volume_analytical_incom),
                "vol_com_exact_m3": float(volume_com),
                "vol0_m3": float(volume0),
                "vol0_mesh_m3": float(mesh_volume0),
            }
        )

    return rows


def run_compressible_shape_timeseries(
    *,
    subdivisions: int,
    radius: float,
    gamma: float,
    bulk_modulus: float,
    pressure_drive_fraction: float,
    amplitude: float,
    n_steps: int,
    t_final: float,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
) -> list[dict[str, float]]:
    """Compressible shape-mode run with EOS volume from the FHeron pressure."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if not 0.0 <= pressure_drive_fraction < 0.95:
        raise ValueError("pressure_drive_fraction must be in [0, 0.95)")

    unit_points, faces = icosphere(subdivisions, 1.0)
    reference_points = unit_points * float(radius)
    mesh_volume0 = mesh_volume(reference_points, faces)
    times = np.linspace(0.0, float(t_final), int(n_steps))
    rows = []

    for t in times:
        phase = math.sin(2.0 * math.pi * float(t) / float(t_final))
        mode_amplitude = float(amplitude) * math.cos(2.0 * math.pi * float(t) / float(t_final))
        extra_pressure_drive = float(pressure_drive_fraction) * float(bulk_modulus) * phase
        isotropic_scale = 1.0
        fheron_equiv = 0.0
        eos_pressure = extra_pressure_drive

        for _iteration in range(16):
            trial_points, _trial_volume = compressible_l2_shape_points(
                unit_points,
                faces,
                radius=radius,
                amplitude=mode_amplitude,
                target_shape_volume=mesh_volume0,
                isotropic_radius_scale=isotropic_scale,
            )
            fheron_equiv, _volume_check = fheron_pressure_for_points(trial_points, faces, gamma)
            eos_pressure = fheron_equiv + extra_pressure_drive
            next_scale = eos_radius_scale_from_pressure(eos_pressure, bulk_modulus)
            if abs(next_scale - isotropic_scale) <= 1.0e-14:
                isotropic_scale = next_scale
                break
            isotropic_scale = next_scale

        points, volume = compressible_l2_shape_points(
            unit_points,
            faces,
            radius=radius,
            amplitude=mode_amplitude,
            target_shape_volume=mesh_volume0,
            isotropic_radius_scale=isotropic_scale,
        )
        fheron_equiv, _volume_check = fheron_pressure_for_points(points, faces, gamma)
        eos_pressure = fheron_equiv + extra_pressure_drive
        eos_scale_target = eos_radius_scale_from_pressure(eos_pressure, bulk_modulus)
        fitted_amplitude = fit_l2_shape_amplitude(points, faces)
        radial_ratio = np.linalg.norm(points, axis=1) / float(radius)
        rows.append(
            {
                "t": float(t),
                "capillary_pressure_pa": float(fheron_equiv),
                "extra_pressure_drive_pa": float(extra_pressure_drive),
                "eos_pressure_pa": float(eos_pressure),
                "pressure_drive_pa": float(extra_pressure_drive),
                "shape_amplitude": float(mode_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "eos_radius_scale": float(isotropic_scale),
                "eos_radius_scale_target": float(eos_scale_target),
                "eos_radius_scale_residual": float(isotropic_scale - eos_scale_target),
                "fheron_compressible_shape_pa": float(fheron_equiv),
                "vol_compressible_shape_m3": float(volume),
                "vol0_mesh_m3": float(mesh_volume0),
                "radial_ratio_min": float(np.min(radial_ratio)),
                "radial_ratio_max": float(np.max(radial_ratio)),
            }
        )

    return rows


def run_local_projection_timeseries(
    *,
    subdivisions: int,
    radius: float,
    gamma: float,
    amplitude: float,
    n_steps: int,
    t_final: float,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
) -> list[dict[str, float]]:
    """Incompressible shape-mode run with finite-amplitude volume projection."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if abs(float(amplitude)) >= 0.8:
        raise ValueError("shape-mode amplitude is too large for this simple mesh test")

    unit_points, faces = icosphere(subdivisions, 1.0)
    reference_points = unit_points * float(radius)
    target_volume = mesh_volume(reference_points, faces)
    times = np.linspace(0.0, float(t_final), int(n_steps))
    rows = []

    for t in times:
        mode_amplitude = float(amplitude) * math.cos(2.0 * math.pi * float(t) / float(t_final))
        points, volume = projected_l2_shape_points(
            unit_points,
            faces,
            radius=radius,
            amplitude=mode_amplitude,
            target_volume=target_volume,
        )
        fheron_equiv, _volume_check = fheron_pressure_for_points(points, faces, gamma)
        fitted_amplitude = fit_l2_shape_amplitude(points, faces)
        radial_ratio = np.linalg.norm(points, axis=1) / float(radius)
        rows.append(
            {
                "t": float(t),
                "shape_amplitude": float(mode_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "fheron_local_projection_pa": float(fheron_equiv),
                "vol_local_projection_m3": float(volume),
                "vol0_mesh_m3": float(target_volume),
                "radial_ratio_min": float(np.min(radial_ratio)),
                "radial_ratio_max": float(np.max(radial_ratio)),
            }
        )

    return rows


def _remove_rigid_translation(points: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass_sum = max(float(np.sum(masses)), 1.0e-300)
    center = np.sum(points * masses[:, None], axis=0) / mass_sum
    mean_velocity = np.sum(velocities * masses[:, None], axis=0) / mass_sum
    return points - center[None, :], velocities - mean_velocity[None, :]


def _project_volume_velocity(velocities: np.ndarray, area_vectors: np.ndarray, masses: np.ndarray) -> np.ndarray:
    inv_masses = 1.0 / np.maximum(masses, 1.0e-300)
    denom = float(np.sum(np.sum(area_vectors * area_vectors, axis=1) * inv_masses))
    if denom <= 0.0:
        return velocities
    volume_rate = float(np.sum(area_vectors * velocities))
    correction = (volume_rate / denom) * area_vectors * inv_masses[:, None]
    return velocities - correction


def _project_local_volume_velocity(
    velocities: np.ndarray,
    matrix: np.ndarray,
    masses: np.ndarray,
    target_rates: np.ndarray,
) -> np.ndarray:
    velocity_vec = velocities.reshape(-1)
    stiffness, inv_mass_dof = _local_volume_stiffness(matrix, masses)
    rhs = matrix @ velocity_vec - np.asarray(target_rates, dtype=float)
    multipliers = _solve_regularized(stiffness, rhs)
    projected_vec = velocity_vec - inv_mass_dof * (matrix.T @ multipliers)
    return projected_vec.reshape(velocities.shape)


def _incompressible_pressure(
    nonpressure_forces: np.ndarray,
    area_vectors: np.ndarray,
    masses: np.ndarray,
) -> float:
    inv_masses = 1.0 / np.maximum(masses, 1.0e-300)
    denom = float(np.sum(np.sum(area_vectors * area_vectors, axis=1) * inv_masses))
    if denom <= 0.0:
        return 0.0
    numer = float(np.sum(np.sum(area_vectors * nonpressure_forces, axis=1) * inv_masses))
    return -numer / denom


def _local_incompressible_pressures(
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    matrix: np.ndarray,
    masses: np.ndarray,
    volumes: np.ndarray,
    reference_volumes: np.ndarray,
    dt: float,
) -> np.ndarray:
    stiffness, inv_mass_dof = _local_volume_stiffness(matrix, masses)
    target_rates = -(volumes - reference_volumes) / float(dt)
    velocity_vec = velocities.reshape(-1)
    force_vec = nonpressure_forces.reshape(-1)
    rhs = (target_rates - matrix @ velocity_vec) / float(dt) - matrix @ (inv_mass_dof * force_vec)
    return _solve_regularized(stiffness, rhs)


def run_force_dynamic_timeseries(
    *,
    closure: str,
    subdivisions: int,
    radius: float,
    gamma: float,
    rho: float,
    bulk_modulus: float,
    amplitude: float,
    n_steps: int,
    t_final: float,
    substeps_per_sample: int,
    damping_ratio: float,
    mass_model: str,
    rayleigh_frequency_hz: float,
    rayleigh_omega_rad_s: float,
    rayleigh_mode: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Force-driven surface run: Ftot updates velocity and vertex positions."""
    if closure not in {"compressible", "incompressible"}:
        raise ValueError("closure must be 'compressible' or 'incompressible'")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if substeps_per_sample < 1:
        raise ValueError("substeps_per_sample must be at least 1")

    reference_points, volume_tets, surface_indices, surface_faces, _surface_faces_global, _unit_points = build_ball_tet_mesh(
        subdivisions,
        radius,
    )
    reference_surface_points = reference_points[surface_indices]
    target_volume = float(np.sum(tet_cell_volumes(reference_points, volume_tets)))
    points = deform_l2_volume_points(
        reference_points,
        volume_tets,
        radius=radius,
        amplitude=amplitude,
        target_volume=target_volume,
    )
    reference_tet_volumes, _reference_volume_matrix = tet_volume_matrix(points, volume_tets)
    velocities = np.zeros_like(points)

    _reference_forces, _reference_area_vectors, reference_pressure, _ = fheron_forces_for_points(
        reference_surface_points,
        surface_faces,
        gamma,
    )
    if mass_model == "star-volume":
        masses = lump_tet_masses(points.shape[0], volume_tets, reference_tet_volumes, rho)
        mass_model_label = "bulk-tet lumped mass"
    elif mass_model == "rayleigh-added":
        masses = lump_tet_masses(points.shape[0], volume_tets, reference_tet_volumes, rho)
        reference_scalar_areas = vertex_scalar_areas(reference_surface_points, surface_faces)
        surface_added_mass = float(rho) * float(radius) / float(rayleigh_mode)
        masses[surface_indices] += surface_added_mass * reference_scalar_areas
        mass_model_label = "bulk plus Rayleigh surface mass"
    else:
        raise ValueError("mass_model must be 'star-volume' or 'rayleigh-added'")
    masses = np.maximum(masses, 1.0e-18)
    points, velocities = _remove_rigid_translation(points, velocities, masses)

    dt = float(t_final) / float((int(n_steps) - 1) * int(substeps_per_sample))
    damping_rate = 2.0 * float(damping_ratio) * float(rayleigh_omega_rad_s)
    rows: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    def collect(sample_index: int, time_value: float) -> None:
        substeps_done = int(sample_index) * int(substeps_per_sample)
        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = time.perf_counter() - wall_start
        cpu_ms_per_substep = 1.0e3 * cpu_elapsed / float(substeps_done) if substeps_done else 0.0
        wall_ms_per_substep = 1.0e3 * wall_elapsed / float(substeps_done) if substeps_done else 0.0
        surface_points = points[surface_indices]
        surface_forces, _surface_area_vectors, fheron_equiv, _surface_volume = fheron_forces_for_points(
            surface_points,
            surface_faces,
            gamma,
        )
        forces = np.zeros_like(points)
        forces[surface_indices] += surface_forces
        tet_volumes, local_matrix = tet_volume_matrix(points, volume_tets)
        volume = float(np.sum(tet_volumes))
        local_volume_rel = tet_volumes / np.maximum(reference_tet_volumes, 1.0e-300)
        if closure == "compressible":
            pressure_values = float(reference_pressure) + float(bulk_modulus) * (
                np.asarray(reference_tet_volumes, dtype=float) / np.maximum(tet_volumes, 1.0e-300) - 1.0
            )
            velocity_projected = 0.0
        else:
            damping_forces = -damping_rate * masses[:, None] * velocities
            pressure_values = _local_incompressible_pressures(
                velocities,
                forces + damping_forces,
                local_matrix,
                masses,
                tet_volumes,
                reference_tet_volumes,
                dt,
            )
            velocity_projected = float(np.linalg.norm(local_matrix @ velocities.reshape(-1)))
        pressure = float(np.mean(pressure_values))
        fitted_amplitude = fit_l2_shape_amplitude(surface_points, surface_faces)
        radial_ratio = np.linalg.norm(surface_points, axis=1) / float(radius)
        theory_amplitude = float(amplitude) * math.cos(float(rayleigh_omega_rad_s) * float(time_value))
        rows.append(
            {
                "sample": float(sample_index),
                "t": float(time_value),
                "closure_pressure_pa": float(pressure),
                "closure_pressure_min_pa": float(np.min(pressure_values)),
                "closure_pressure_max_pa": float(np.max(pressure_values)),
                "reference_pressure_pa": float(reference_pressure),
                "mass_model": mass_model_label,
                "total_mass_kg": float(np.sum(masses)),
                "shape_amplitude_theory": float(theory_amplitude),
                "shape_amplitude_fit": float(fitted_amplitude),
                "rayleigh_frequency_hz": float(rayleigh_frequency_hz),
                "rayleigh_omega_rad_s": float(rayleigh_omega_rad_s),
                "fheron_pressure_pa": float(fheron_equiv),
                "vol_m3": float(volume),
                "vol0_mesh_m3": float(target_volume),
                "vol_rel": float(volume / target_volume),
                "local_volume_rel_rms": float(math.sqrt(float(np.mean((local_volume_rel - 1.0) ** 2)))),
                "local_volume_rel_max_abs": float(np.max(np.abs(local_volume_rel - 1.0))),
                "vol_rate_m3_s": float(np.sum(local_matrix @ velocities.reshape(-1))),
                "constraint_volume_rate_m3_s": float(velocity_projected),
                "kinetic_energy_j": float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))),
                "max_speed_m_s": float(np.max(np.linalg.norm(velocities, axis=1))),
                "cpu_time_s": float(cpu_elapsed),
                "wall_time_s": float(wall_elapsed),
                "cpu_ms_per_substep": float(cpu_ms_per_substep),
                "wall_ms_per_substep": float(wall_ms_per_substep),
                "radial_ratio_min": float(np.min(radial_ratio)),
                "radial_ratio_max": float(np.max(radial_ratio)),
            }
        )
        snapshots.append(points.copy())

    collect(0, 0.0)
    sample_index = 1
    total_substeps = (int(n_steps) - 1) * int(substeps_per_sample)
    for substep in range(1, total_substeps + 1):
        surface_points = points[surface_indices]
        surface_forces, _surface_area_vectors, _fheron_equiv, _surface_volume = fheron_forces_for_points(
            surface_points,
            surface_faces,
            gamma,
        )
        forces = np.zeros_like(points)
        forces[surface_indices] += surface_forces
        tet_volumes, local_matrix = tet_volume_matrix(points, volume_tets)
        damping_forces = -damping_rate * masses[:, None] * velocities

        if closure == "compressible":
            base_pressure = float(reference_pressure)
            base_pressures = np.full(volume_tets.shape[0], base_pressure, dtype=float)
            nonpressure_forces = forces + tet_pressure_forces(points, volume_tets, base_pressures) + damping_forces
            u_star = velocities + dt * nonpressure_forces / masses[:, None]
            stiffness, inv_mass_dof = _local_volume_stiffness(local_matrix, masses)
            compressibility = float(bulk_modulus) / np.maximum(reference_tet_volumes, 1.0e-300)
            linear_volume_error = tet_volumes - reference_tet_volumes + dt * (local_matrix @ u_star.reshape(-1))
            system = np.eye(volume_tets.shape[0]) + (compressibility[:, None] * dt * dt) * stiffness
            rhs = -compressibility * linear_volume_error
            pressure_delta = _solve_regularized(system, rhs)
            velocities = (u_star.reshape(-1) + dt * inv_mass_dof * (local_matrix.T @ pressure_delta)).reshape(velocities.shape)
            pressure = float(np.mean(base_pressures + pressure_delta))
        else:
            pressure_values = _local_incompressible_pressures(
                velocities,
                forces + damping_forces,
                local_matrix,
                masses,
                tet_volumes,
                reference_tet_volumes,
                dt,
            )
            pressure = float(np.mean(pressure_values))
            total_forces = forces + tet_pressure_forces(points, volume_tets, pressure_values) + damping_forces
            accelerations = total_forces / masses[:, None]
            velocities = velocities + dt * accelerations
            target_rates = -(tet_volumes - reference_tet_volumes) / float(dt)
            velocities = _project_local_volume_velocity(velocities, local_matrix, masses, target_rates)

        points = points + dt * velocities
        points, velocities = _remove_rigid_translation(points, velocities, masses)

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(velocities)):
            raise FloatingPointError(f"{closure} dynamic run became non-finite at substep {substep}")
        if np.min(np.linalg.norm(points[surface_indices], axis=1)) <= 0.15 * float(radius):
            raise FloatingPointError(f"{closure} dynamic run collapsed at substep {substep}")

        if substep % int(substeps_per_sample) == 0:
            collect(sample_index, sample_index * float(t_final) / float(int(n_steps) - 1))
            sample_index += 1

    return rows, np.asarray(snapshots, dtype=float), surface_faces, volume_tets, surface_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HC Heron surface force with EOS and incompressible pressure closures on a sphere."
    )
    parser.add_argument("--max-subdivision", type=int, default=3)
    parser.add_argument("--radius", type=float, default=1.0e-3, help="Sphere radius [m].")
    parser.add_argument("--gamma", type=float, default=0.072, help="Surface tension [N/m].")
    parser.add_argument("--bulk-modulus", type=float, default=2.2e9, help="EOS bulk modulus [Pa].")
    parser.add_argument("--rho", type=float, default=1000.0, help="Liquid density [kg/m^3] for Rayleigh-Lamb frequency.")
    parser.add_argument("--rayleigh-mode", type=int, default=2, help="Rayleigh-Lamb shape mode used for GIF timing.")
    parser.add_argument(
        "--use-rayleigh-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Rayleigh-Lamb period for shape-mode GIFs.",
    )
    parser.add_argument("--fheron-repeats", type=int, default=3)
    parser.add_argument("--closure-repeats", type=int, default=2000)
    parser.add_argument("--timeseries-subdivision", type=int, default=2)
    parser.add_argument("--timeseries-steps", type=int, default=121)
    parser.add_argument(
        "--dynamic-subdivision",
        type=int,
        default=1,
        help="Icosphere subdivision used for force-driven GIFs.",
    )
    parser.add_argument(
        "--dynamic-substeps-per-sample",
        type=int,
        default=80,
        help="Explicit Ftot integration substeps between saved dynamic samples.",
    )
    parser.add_argument(
        "--dynamic-damping-ratio",
        type=float,
        default=0.003,
        help="Small modal damping ratio used only to control discrete mesh noise in Ftot GIFs.",
    )
    parser.add_argument(
        "--dynamic-mass-model",
        choices=("star-volume", "rayleigh-added"),
        default="star-volume",
        help="Mass used for Ftot integration. star-volume is untuned; rayleigh-added reproduces Rayleigh inertia.",
    )
    parser.add_argument(
        "--timeseries-final-time",
        type=float,
        default=None,
        help="Override final time [s]. Defaults to one Rayleigh-Lamb period when enabled.",
    )
    parser.add_argument("--animation-max-frames", type=int, default=80)
    parser.add_argument("--animation-fps", type=int, default=18)
    parser.add_argument(
        "--shape-mode-amplitude",
        type=float,
        default=None,
        help="Finite-amplitude l=2 shape mode. Overrides --shape-mode-ar when set.",
    )
    parser.add_argument(
        "--shape-mode-ar",
        type=float,
        default=1.05,
        help="Maximum aspect ratio for the local incompressible projection GIF.",
    )
    parser.add_argument(
        "--pressure-drive-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional nonphysical sinusoidal pressure perturbation as a fraction of the EOS bulk modulus. "
            "Default 0 keeps the compressible case driven only by FHeron/capillary pressure."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "out" / "sphere_fheron_eos_projection",
    )
    return parser.parse_args()


def write_outputs(rows: list[dict[str, float]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sphere_fheron_eos_projection_benchmark.csv"
    json_path = out_dir / "sphere_fheron_eos_projection_benchmark.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_timeseries_outputs(rows: list[dict[str, float]], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "sphere_fheron_eos_projection_timeseries.csv"
    json_path = out_dir / "sphere_fheron_eos_projection_timeseries.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_local_projection_outputs(rows: list[dict[str, float]], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "sphere_fheron_incompressible_local_projection_timeseries.csv"
    json_path = out_dir / "sphere_fheron_incompressible_local_projection_timeseries.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_compressible_shape_outputs(rows: list[dict[str, float]], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "sphere_fheron_compressible_shape_timeseries.csv"
    json_path = out_dir / "sphere_fheron_compressible_shape_timeseries.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_named_rows_outputs(rows: list[dict[str, float]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_dynamic_snapshots(
    snapshots: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    tets: np.ndarray | None = None,
    surface_indices: np.ndarray | None = None,
) -> Path:
    path = out_dir / f"{stem}.npz"
    payload = {"points": snapshots, "faces": faces}
    if tets is not None:
        payload["tets"] = tets
    if surface_indices is not None:
        payload["surface_indices"] = surface_indices
    np.savez_compressed(path, **payload)
    return path


def _row_array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _finish_plot(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_timing_plot(rows: list[dict[str, float]], out_dir: Path) -> Path:
    vertices = _row_array(rows, "vertices")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.loglog(vertices, _row_array(rows, "fheron_ms"), "o-", lw=2.0, label="FHeron kernel")
    ax.loglog(vertices, 1.0e-3 * _row_array(rows, "eos_us"), "s-", lw=2.0, label="EOS scalar")
    ax.loglog(vertices, 1.0e-3 * _row_array(rows, "projection_us"), "^-", lw=2.0, label="Incompressible projection scalar")
    ax.set_xlabel("HC surface vertices")
    ax.set_ylabel("time per evaluation [ms]")
    ax.set_title("Compute time: FHeron vs pressure closures")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False)
    return _finish_plot(fig, out_dir / "compute_time_fheron_eos_projection.png")


def write_pressure_plot(rows: list[dict[str, float]], out_dir: Path) -> Path:
    vertices = _row_array(rows, "vertices")
    theory = _row_array(rows, "theory_pressure_pa")
    eos = _row_array(rows, "eos_pressure_pa")
    projection = _row_array(rows, "projection_pressure_pa")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)
    ax0.semilogx(vertices, theory, "k--", lw=2.0, label="Young-Laplace theory")
    ax0.semilogx(vertices, eos, "o-", lw=1.9, label="EOS")
    ax0.semilogx(vertices, projection, "^-", lw=1.9, label="Incompressible projection")
    ax0.set_ylabel("pressure [Pa]")
    ax0.set_title("Pressure closures against theory")
    pressure_formatter = ScalarFormatter(useOffset=False)
    pressure_formatter.set_scientific(False)
    ax0.yaxis.set_major_formatter(pressure_formatter)
    p_ref = float(np.mean(theory))
    p_pad = max(0.005 * abs(p_ref), 20.0 * float(np.max(np.abs(eos - theory))), 1.0e-9)
    ax0.set_ylim(p_ref - p_pad, p_ref + p_pad)
    ax0.grid(True, which="both", alpha=0.28)
    ax0.legend(frameon=False)

    denom = np.maximum(np.abs(theory), 1.0e-300)
    ax1.loglog(vertices, np.maximum(np.abs(eos - theory) / denom, 1.0e-18), "o-", lw=1.9, label="EOS pressure error")
    ax1.loglog(
        vertices,
        np.maximum(np.abs(projection - theory) / denom, 1.0e-18),
        "^-",
        lw=1.9,
        label="Projection pressure error",
    )
    ax1.set_xlabel("HC surface vertices")
    ax1.set_ylabel("relative pressure error")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(frameon=False)
    return _finish_plot(fig, out_dir / "pressure_vs_young_laplace_theory.png")


def write_residual_plot(rows: list[dict[str, float]], out_dir: Path) -> Path:
    vertices = _row_array(rows, "vertices")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.loglog(vertices, np.maximum(_row_array(rows, "heron_vs_theory_rel_rms"), 1.0e-18), "o-", lw=2.0, label="FHeron vs theory force")
    ax.loglog(vertices, np.maximum(_row_array(rows, "eos_residual_rel_rms"), 1.0e-18), "s-", lw=2.0, label="EOS residual")
    ax.loglog(
        vertices,
        np.maximum(_row_array(rows, "projection_residual_rel_rms"), 1.0e-18),
        "^-",
        lw=2.0,
        label="Projection residual",
    )
    ax.set_xlabel("HC surface vertices")
    ax.set_ylabel("relative RMS")
    ax.set_title("Force-balance residual: FHeron + pressure area")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False)
    return _finish_plot(fig, out_dir / "fheron_theory_residuals.png")


def write_fheron_sphere_plot(
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    gamma: float,
) -> Path:
    points, faces = icosphere(subdivisions, radius)
    _HC, vertices = build_surface_hc(points, faces)
    forces, areas = heron_force_area(vertices, gamma)
    normals = points / np.linalg.norm(points, axis=1)[:, None]

    sample_step = max(1, int(math.ceil(len(vertices) / 56)))
    sample = np.arange(0, len(vertices), sample_step)
    f_norm = np.linalg.norm(forces[sample], axis=1)
    f_dirs = np.zeros((sample.size, 3), dtype=float)
    valid = f_norm > 0.0
    f_dirs[valid] = forces[sample][valid] / f_norm[valid, None]

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    triangles = points[faces]
    surface = Poly3DCollection(
        triangles,
        facecolor="#d9ecff",
        edgecolor="#7aa6c8",
        linewidth=0.25,
        alpha=0.45,
    )
    ax.add_collection3d(surface)
    arrow_len = 0.22 * float(radius)
    ax.quiver(
        points[sample, 0],
        points[sample, 1],
        points[sample, 2],
        f_dirs[:, 0],
        f_dirs[:, 1],
        f_dirs[:, 2],
        length=arrow_len,
        normalize=True,
        color="#d62728",
        linewidth=1.2,
        label="FHeron",
    )
    ax.quiver(
        points[sample, 0],
        points[sample, 1],
        points[sample, 2],
        normals[sample, 0],
        normals[sample, 1],
        normals[sample, 2],
        length=0.72 * arrow_len,
        normalize=True,
        color="#1f77b4",
        linewidth=0.9,
        alpha=0.85,
        label="pressure balance",
    )

    limit = 1.35 * float(radius)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("HC sphere: inward FHeron balanced by pressure")
    ax.view_init(elev=22, azim=35)
    ax.legend(loc="upper left", frameon=False)

    mean_area = float(np.mean(areas))
    ax.text2D(
        0.58,
        0.83,
        f"Delta p = 2 gamma / R = {2.0 * gamma / radius:.3g} Pa\n"
        f"mean Heron dual area = {mean_area:.3e} m^2",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4.0},
    )
    return _finish_plot(fig, out_dir / "fheron_sphere_force_balance.png")


def write_plots(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    radius: float,
    gamma: float,
) -> list[Path]:
    plot_paths = [
        write_timing_plot(rows, out_dir),
        write_pressure_plot(rows, out_dir),
        write_residual_plot(rows, out_dir),
    ]
    finest_subdivision = int(max(row["subdivisions"] for row in rows))
    plot_paths.append(
        write_fheron_sphere_plot(
            out_dir,
            subdivisions=finest_subdivision,
            radius=radius,
            gamma=gamma,
        )
    )
    return plot_paths


def write_timeseries_plots(rows: list[dict[str, float]], out_dir: Path) -> list[Path]:
    t = _row_array(rows, "t")
    vol0 = float(rows[0]["vol0_mesh_m3"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(t, _row_array(rows, "fheron_com_pa"), lw=2.0, label="FHeron_com")
    ax.plot(t, _row_array(rows, "fheron_analytical_com_pa"), "k--", lw=2.0, label="FHeron_com analytical")
    ax.plot(t, _row_array(rows, "fheron_incom_pa"), lw=2.0, label="FHeron_incom")
    ax.plot(
        t,
        _row_array(rows, "fheron_analytical_incom_pa"),
        ":",
        color="0.20",
        lw=2.4,
        label="FHeron_incom analytical",
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("FHeron pressure equivalent [Pa]")
    ax.set_title("FHeron vs t: each path against its own analytical reference")
    fheron_formatter = ScalarFormatter(useOffset=False)
    fheron_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(fheron_formatter)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fheron_path = _finish_plot(fig, out_dir / "timeseries_fheron_com_incom_analytical_vs_t.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(t, _row_array(rows, "vol_com_m3") / vol0, lw=2.0, label="Vol_com")
    ax.plot(t, _row_array(rows, "vol_analytical_com_m3") / vol0, "k--", lw=2.0, label="Vol_com analytical")
    ax.plot(t, _row_array(rows, "vol_incom_m3") / vol0, lw=2.0, label="Vol_incom")
    ax.plot(
        t,
        _row_array(rows, "vol_analytical_incom_m3") / vol0,
        ":",
        color="0.20",
        lw=2.4,
        label="Vol_incom analytical",
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("V / V0")
    ax.set_title("Volume vs t: each path against its own analytical reference")
    volume_formatter = ScalarFormatter(useOffset=False)
    volume_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(volume_formatter)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    volume_path = _finish_plot(fig, out_dir / "timeseries_volume_com_incom_analytical_vs_t.png")

    return [fheron_path, volume_path]


def write_sphere_animation(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    radius_key: str,
    volume_key: str,
    fheron_key: str,
    sphere_title: str,
    volume_title: str,
    suptitle_prefix: str,
    gif_name: str,
    facecolor: str,
    edgecolor: str,
    max_frames: int,
    fps: int,
) -> Path:
    t = _row_array(rows, "t")
    radius_values = _row_array(rows, radius_key)
    volume = _row_array(rows, volume_key)
    volume0 = float(rows[0]["vol0_mesh_m3"])
    fheron = _row_array(rows, fheron_key)
    unit_points, faces = icosphere(subdivisions, 1.0)
    reference_points = unit_points * float(radius)

    frame_count = min(len(rows), max(2, int(max_frames)))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count, dtype=int))

    fig = plt.figure(figsize=(9.2, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 1.0), wspace=0.22, hspace=0.34)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_vol = fig.add_subplot(grid[0, 1])
    ax_force = fig.add_subplot(grid[1, 1])

    r_min = float(np.min(radius_values) / float(radius))
    r_max = float(np.max(radius_values) / float(radius))
    limit = 1.10 * max(float(radius), float(np.max(radius_values)))

    def draw_frame(frame_pos: int):
        row_idx = int(frame_indices[frame_pos])
        current_radius = float(radius_values[row_idx])
        current_points = unit_points * current_radius

        ax3d.cla()
        ax_vol.cla()
        ax_force.cla()

        reference = Poly3DCollection(
            reference_points[faces],
            facecolor=(0.0, 0.0, 0.0, 0.025),
            edgecolor=(0.25, 0.25, 0.25, 0.28),
            linewidth=0.25,
        )
        current = Poly3DCollection(
            current_points[faces],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.28,
            alpha=0.70,
        )
        ax3d.add_collection3d(reference)
        ax3d.add_collection3d(current)
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_title(sphere_title)
        ax3d.view_init(elev=20.0, azim=32.0 + 20.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.text2D(
            0.02,
            0.02,
            f"t = {t[row_idx]:.5f} s\n"
            f"R/R0 = {current_radius / float(radius):.5f}\n"
            f"V/V0 = {volume[row_idx] / volume0:.5f}\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa",
            transform=ax3d.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4.0},
        )

        ax_vol.plot(t, volume / volume0, color=edgecolor, lw=1.8)
        ax_vol.plot(t[row_idx], volume[row_idx] / volume0, "o", color="#d62728", ms=5.5)
        ax_vol.axhline(1.0, color="0.25", ls=":", lw=1.4)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        vol_rel = volume / volume0
        if float(np.ptp(vol_rel)) < 1.0e-12:
            ax_vol.set_ylim(0.98, 1.02)
        else:
            ax_vol.set_ylim(0.98 * min(float(np.min(vol_rel)), 1.0), 1.02 * max(float(np.max(vol_rel)), 1.0))
        ax_vol.set_ylabel("V / V0")
        ax_vol.set_title(volume_title)
        ax_vol.grid(True, alpha=0.25)

        ax_force.plot(t, fheron, color=edgecolor, lw=1.8)
        ax_force.plot(t[row_idx], fheron[row_idx], "o", color="#d62728", ms=5.5)
        ax_force.set_xlim(float(t[0]), float(t[-1]))
        force_formatter = ScalarFormatter(useOffset=False)
        force_formatter.set_scientific(False)
        ax_force.yaxis.set_major_formatter(force_formatter)
        if float(np.ptp(fheron)) < 1.0e-10:
            pad = max(0.005 * abs(float(np.mean(fheron))), 0.5)
        else:
            pad = max(0.03 * float(np.ptp(fheron)), 1.0e-9)
        ax_force.set_ylim(float(np.min(fheron) - pad), float(np.max(fheron) + pad))
        ax_force.set_xlabel("t [s]")
        ax_force.set_ylabel("FHeron [Pa]")
        ax_force.set_title("Surface-force pressure equivalent")
        ax_force.grid(True, alpha=0.25)

        fig.suptitle(
            f"{suptitle_prefix}, R/R0 range {r_min:.5f}-{r_max:.5f}",
            y=0.98,
        )

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / gif_name
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def write_compressible_animation(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    max_frames: int,
    fps: int,
) -> Path:
    return write_sphere_animation(
        rows,
        out_dir,
        subdivisions=subdivisions,
        radius=radius,
        radius_key="radius_com_m",
        volume_key="vol_com_m3",
        fheron_key="fheron_com_pa",
        sphere_title="Compressible EOS sphere",
        volume_title="EOS volume response",
        suptitle_prefix="Compressible sphere oscillation",
        gif_name="compressible_sphere_oscillation.gif",
        facecolor="#9ecae1",
        edgecolor="#1f77b4",
        max_frames=max_frames,
        fps=fps,
    )


def write_incompressible_animation(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    max_frames: int,
    fps: int,
) -> Path:
    return write_sphere_animation(
        rows,
        out_dir,
        subdivisions=subdivisions,
        radius=radius,
        radius_key="radius_incom_m",
        volume_key="vol_incom_m3",
        fheron_key="fheron_incom_pa",
        sphere_title="Incompressible projected sphere",
        volume_title="Projected constant volume",
        suptitle_prefix="Incompressible projection",
        gif_name="incompressible_sphere_projection.gif",
        facecolor="#fdd0a2",
        edgecolor="#ff7f0e",
        max_frames=max_frames,
        fps=fps,
    )


def write_compressible_shape_animation(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    max_frames: int,
    fps: int,
) -> Path:
    t = _row_array(rows, "t")
    amplitude = _row_array(rows, "shape_amplitude")
    amplitude_fit = _row_array(rows, "shape_amplitude_fit")
    radius_scale = _row_array(rows, "eos_radius_scale")
    volume = _row_array(rows, "vol_compressible_shape_m3")
    volume0 = float(rows[0]["vol0_mesh_m3"])
    fheron = _row_array(rows, "fheron_compressible_shape_pa")
    capillary_pressure = _row_array(rows, "capillary_pressure_pa")
    extra_pressure_drive = _row_array(rows, "extra_pressure_drive_pa")
    eos_pressure = _row_array(rows, "eos_pressure_pa")
    rayleigh_frequency = float(rows[0].get("rayleigh_frequency_hz", 1.0 / max(float(t[-1] - t[0]), 1.0e-30)))
    unit_points, faces = icosphere(subdivisions, 1.0)
    reference_points = unit_points * float(radius)

    frame_count = min(len(rows), max(2, int(max_frames)))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count, dtype=int))
    radial_min = min(float(row["radial_ratio_min"]) for row in rows)
    radial_max = max(float(row["radial_ratio_max"]) for row in rows)
    limit = 1.12 * radial_max * float(radius)

    fig = plt.figure(figsize=(10.2, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.32, 1.0), wspace=0.25, hspace=0.48)
    ax3d = fig.add_subplot(grid[:2, 0], projection="3d")
    ax_info = fig.add_subplot(grid[2, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_vol = fig.add_subplot(grid[1, 1])
    ax_force = fig.add_subplot(grid[2, 1])

    def draw_frame(frame_pos: int):
        row_idx = int(frame_indices[frame_pos])
        points, _volume_check = compressible_l2_shape_points(
            unit_points,
            faces,
            radius=radius,
            amplitude=float(amplitude[row_idx]),
            target_shape_volume=volume0,
            isotropic_radius_scale=float(radius_scale[row_idx]),
        )

        ax3d.cla()
        ax_info.cla()
        ax_amp.cla()
        ax_vol.cla()
        ax_force.cla()

        reference = Poly3DCollection(
            reference_points[faces],
            facecolor=(0.0, 0.0, 0.0, 0.025),
            edgecolor=(0.25, 0.25, 0.25, 0.28),
            linewidth=0.25,
        )
        current = Poly3DCollection(
            points[faces],
            facecolor="#9ecae1",
            edgecolor="#1f77b4",
            linewidth=0.28,
            alpha=0.76,
        )
        ax3d.add_collection3d(reference)
        ax3d.add_collection3d(current)
        tet_wire = Line3DCollection(
            volumetric_star_tet_segments(points, faces),
            colors=(0.12, 0.47, 0.71, 0.28),
            linewidths=0.35,
        )
        ax3d.add_collection3d(tet_wire)
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_title("Compressible volumetric star-tet mesh")
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.text2D(
            0.02,
            0.02,
            f"t = {t[row_idx]:.5f} s\n"
            f"mode a2 = {amplitude[row_idx]:+.4f}\n"
            f"fit a2 = {amplitude_fit[row_idx]:+.4f}\n"
            f"AR = {float(rows[row_idx]['radial_ratio_max']) / float(rows[row_idx]['radial_ratio_min']):.4f}\n"
            f"dV/V0 = {(volume[row_idx] / volume0 - 1.0) * 1.0e6:+.4f} ppm\n"
            f"p_cap = {capillary_pressure[row_idx]:.3f} Pa\n"
            f"p_extra = {extra_pressure_drive[row_idx]:+.3e} Pa\n"
            f"p_EOS = {eos_pressure[row_idx]:.3f} Pa\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa",
            transform=ax3d.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4.0},
        )

        amp_pad = max(0.08 * float(np.ptp(amplitude)), 0.005)
        ax_amp.plot(t, amplitude, "k--", lw=1.7, label="theory")
        ax_amp.plot(t, amplitude_fit, color="#1f77b4", lw=1.8, label="fit from mesh")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=5.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(
            float(min(np.min(amplitude), np.min(amplitude_fit)) - amp_pad),
            float(max(np.max(amplitude), np.max(amplitude_fit)) + amp_pad),
        )
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        vol_rel_ppm = (volume / volume0 - 1.0) * 1.0e6
        ax_vol.plot(t, vol_rel_ppm, color="#1f77b4", lw=1.8)
        ax_vol.plot(t[row_idx], vol_rel_ppm[row_idx], "o", color="#d62728", ms=5.5)
        ax_vol.axhline(0.0, color="0.25", ls=":", lw=1.4)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        vol_pad = max(0.12 * float(np.ptp(vol_rel_ppm)), 0.01)
        ax_vol.set_ylim(float(np.min(vol_rel_ppm) - vol_pad), float(np.max(vol_rel_ppm) + vol_pad))
        ax_vol.set_ylabel("Delta V / V0 [ppm]")
        ax_vol.set_title("Physical EOS volume response")
        ax_vol.grid(True, alpha=0.25)

        ax_force.plot(t, fheron, color="#1f77b4", lw=1.8)
        ax_force.plot(t[row_idx], fheron[row_idx], "o", color="#d62728", ms=5.5)
        ax_force.set_xlim(float(t[0]), float(t[-1]))
        force_formatter = ScalarFormatter(useOffset=False)
        force_formatter.set_scientific(False)
        ax_force.yaxis.set_major_formatter(force_formatter)
        pad = max(0.04 * float(np.ptp(fheron)), 1.0e-9)
        ax_force.set_ylim(float(np.min(fheron) - pad), float(np.max(fheron) + pad))
        ax_force.set_xlabel("t [s]")
        ax_force.set_ylabel("FHeron [Pa]")
        ax_force.set_title("Shape plus EOS curvature response")
        ax_force.grid(True, alpha=0.25)

        fig.suptitle(
            f"Compressible shape mode, Rayleigh f={rayleigh_frequency:.3f} Hz, radial range {radial_min:.4f}-{radial_max:.4f}",
            y=0.98,
        )

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / "compressible_sphere_oscillation.gif"
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def write_force_dynamic_animation(
    rows: list[dict[str, float]],
    snapshots: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    *,
    radius: float,
    subdivisions: int,
    closure: str,
    max_frames: int,
    fps: int,
    volume_tets: np.ndarray | None = None,
    surface_indices: np.ndarray | None = None,
) -> Path:
    t = _row_array(rows, "t")
    amplitude_theory = _row_array(rows, "shape_amplitude_theory")
    amplitude_fit = _row_array(rows, "shape_amplitude_fit")
    volume = _row_array(rows, "vol_m3")
    volume0 = float(rows[0]["vol0_mesh_m3"])
    local_volume_max_ppm = _row_array(rows, "local_volume_rel_max_abs") * 1.0e6
    fheron = _row_array(rows, "fheron_pressure_pa")
    closure_pressure = _row_array(rows, "closure_pressure_pa")
    speed = _row_array(rows, "max_speed_m_s")
    cpu_time = np.asarray([float(row.get("cpu_time_s", 0.0)) for row in rows], dtype=float)
    cpu_ms_per_substep = np.asarray([float(row.get("cpu_ms_per_substep", 0.0)) for row in rows], dtype=float)
    cpu_total = float(cpu_time[-1]) if cpu_time.size else 0.0
    cpu_avg_ms = float(cpu_ms_per_substep[-1]) if cpu_ms_per_substep.size else 0.0
    mass_model = str(rows[0].get("mass_model", "dynamic mass"))
    rayleigh_frequency = float(rows[0].get("rayleigh_frequency_hz", 1.0 / max(float(t[-1] - t[0]), 1.0e-30)))
    (
        reference_volume_points,
        reference_tets,
        reference_surface_indices,
        _reference_faces,
        _reference_faces_global,
        _reference_unit_points,
    ) = build_ball_tet_mesh(subdivisions, float(radius))
    if volume_tets is None:
        volume_tets = reference_tets
    if surface_indices is None:
        surface_indices = reference_surface_indices
    surface_indices = np.asarray(surface_indices, dtype=int)
    volume_tets = np.asarray(volume_tets, dtype=int)
    reference_surface_points = reference_volume_points[surface_indices]
    reference_radii = np.linalg.norm(reference_volume_points, axis=1)
    interior_indices = np.setdiff1d(np.arange(reference_volume_points.shape[0], dtype=int), surface_indices)
    center_indices = interior_indices[reference_radii[interior_indices] <= 1.0e-12]
    shell_indices = interior_indices[reference_radii[interior_indices] > 1.0e-12]
    volume_edge_indices = tet_edge_indices(volume_tets, max_edges=360)

    frame_count = min(len(rows), max(2, int(max_frames)))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count, dtype=int))
    radial_min = min(float(row["radial_ratio_min"]) for row in rows)
    radial_max = max(float(row["radial_ratio_max"]) for row in rows)
    limit = 1.14 * radial_max * float(radius)

    if closure == "compressible":
        facecolor = "#9ecae1"
        edgecolor = "#1f77b4"
        title = "Compressible EOS: Ftot-driven vertices"
        gif_name = "compressible_sphere_oscillation.gif"
        vol_title = "EOS volume response"
    else:
        facecolor = "#fdd0a2"
        edgecolor = "#ff7f0e"
        title = "Incompressible projection: Ftot-driven vertices"
        gif_name = "incompressible_sphere_projection.gif"
        vol_title = "Projected volume constraint"

    fig = plt.figure(figsize=(10.2, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.32, 1.0), wspace=0.25, hspace=0.48)
    ax3d = fig.add_subplot(grid[:2, 0], projection="3d")
    ax_info = fig.add_subplot(grid[2, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_vol = fig.add_subplot(grid[1, 1])
    ax_force = fig.add_subplot(grid[2, 1])

    def draw_frame(frame_pos: int):
        row_idx = int(frame_indices[frame_pos])
        volume_points = snapshots[row_idx]
        surface_points = volume_points[surface_indices]

        ax3d.cla()
        ax_info.cla()
        ax_amp.cla()
        ax_vol.cla()
        ax_force.cla()

        reference = Poly3DCollection(
            reference_surface_points[faces],
            facecolor=(0.0, 0.0, 0.0, 0.025),
            edgecolor=(0.25, 0.25, 0.25, 0.28),
            linewidth=0.25,
        )
        current = Poly3DCollection(
            surface_points[faces],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.28,
            alpha=0.48,
        )
        ax3d.add_collection3d(reference)
        ax3d.add_collection3d(current)
        edge_segments = (
            np.asarray(volume_points[volume_edge_indices], dtype=float)
            if volume_edge_indices.size
            else np.empty((0, 2, 3), dtype=float)
        )
        tet_wire = Line3DCollection(
            edge_segments,
            colors=(0.12, 0.12, 0.12, 0.26),
            linewidths=0.35,
        )
        ax3d.add_collection3d(tet_wire)
        if shell_indices.size:
            shell_points = volume_points[shell_indices]
            ax3d.scatter(
                shell_points[:, 0],
                shell_points[:, 1],
                shell_points[:, 2],
                s=12.0,
                c="#2ca02c",
                edgecolors="none",
                alpha=0.82,
                depthshade=False,
            )
        if center_indices.size:
            center_points = volume_points[center_indices]
            ax3d.scatter(
                center_points[:, 0],
                center_points[:, 1],
                center_points[:, 2],
                s=22.0,
                c="#111111",
                edgecolors="white",
                linewidths=0.35,
                alpha=0.9,
                depthshade=False,
            )
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_title(title)
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))

        ax_info.axis("off")
        left_info = (
            f"t = {t[row_idx]:.5f} s\n"
            f"theory a2 = {amplitude_theory[row_idx]:+.4f}\n"
            f"fit a2 = {amplitude_fit[row_idx]:+.4f}\n"
            f"AR = {float(rows[row_idx]['radial_ratio_max']) / float(rows[row_idx]['radial_ratio_min']):.4f}\n"
            f"dV/V0 = {(volume[row_idx] / volume0 - 1.0) * 1.0e6:+.3f} ppm\n"
            f"max tet dV = {local_volume_max_ppm[row_idx]:.3f} ppm\n"
            f"CPU now = {cpu_time[row_idx]:.2f} s\n"
            f"CPU/step = {cpu_ms_per_substep[row_idx]:.2f} ms"
        )
        right_info = (
            f"mesh = {volume_points.shape[0]} verts, {volume_tets.shape[0]} tets\n"
            f"interior = {interior_indices.size} verts\n"
            f"mass = {mass_model}\n"
            f"p = {closure_pressure[row_idx]:.3f} Pa\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa\n"
            f"max |u| = {speed[row_idx]:.3e} m/s\n"
            f"CPU total = {cpu_total:.1f} s\n"
            f"avg CPU/step = {cpu_avg_ms:.1f} ms"
        )
        ax_info.text(
            0.02,
            0.98,
            left_info,
            transform=ax_info.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.2,
        )
        ax_info.text(
            0.50,
            0.98,
            right_info,
            transform=ax_info.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.2,
        )

        amp_pad = max(0.1 * float(np.ptp(amplitude_theory)), 0.004)
        ax_amp.plot(t, amplitude_theory, "k--", lw=1.7, label="Rayleigh theory")
        ax_amp.plot(t, amplitude_fit, color=edgecolor, lw=1.8, label="Ftot simulation")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=5.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(
            float(min(np.min(amplitude_theory), np.min(amplitude_fit)) - amp_pad),
            float(max(np.max(amplitude_theory), np.max(amplitude_fit)) + amp_pad),
        )
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        vol_pad = max(0.12 * float(np.ptp(local_volume_max_ppm)), 0.01)
        ax_vol.plot(t, local_volume_max_ppm, color=edgecolor, lw=1.8)
        ax_vol.plot(t[row_idx], local_volume_max_ppm[row_idx], "o", color="#d62728", ms=5.5)
        ax_vol.axhline(0.0, color="0.25", ls=":", lw=1.4)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        ax_vol.set_ylim(float(np.min(local_volume_max_ppm) - vol_pad), float(np.max(local_volume_max_ppm) + vol_pad))
        ax_vol.set_ylabel("max tet |Delta V| [ppm]")
        ax_vol.set_title(vol_title)
        ax_vol.grid(True, alpha=0.25)

        ax_force.plot(t, fheron, color=edgecolor, lw=1.8, label="FHeron")
        ax_force.plot(t, closure_pressure, color="0.25", lw=1.2, ls=":", label="closure p")
        ax_force.plot(t[row_idx], fheron[row_idx], "o", color="#d62728", ms=5.5)
        ax_force.set_xlim(float(t[0]), float(t[-1]))
        force_formatter = ScalarFormatter(useOffset=False)
        force_formatter.set_scientific(False)
        ax_force.yaxis.set_major_formatter(force_formatter)
        force_min = min(float(np.min(fheron)), float(np.min(closure_pressure)))
        force_max = max(float(np.max(fheron)), float(np.max(closure_pressure)))
        force_pad = max(0.04 * (force_max - force_min), 1.0e-9)
        ax_force.set_ylim(force_min - force_pad, force_max + force_pad)
        ax_force.set_xlabel("t [s]")
        ax_force.set_ylabel("pressure [Pa]")
        ax_force.set_title("Forces used in Ftot")
        ax_force.grid(True, alpha=0.25)
        ax_force.legend(frameon=False, fontsize=8, loc="upper right")

        fig.suptitle(
            f"{closure.capitalize()} Ftot dynamic, {mass_model}, Rayleigh f={rayleigh_frequency:.3f} Hz, "
            f"CPU {cpu_total:.1f} s ({cpu_avg_ms:.1f} ms/step)",
            y=0.965,
        )

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / gif_name
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def write_incompressible_local_projection_animation(
    rows: list[dict[str, float]],
    out_dir: Path,
    *,
    subdivisions: int,
    radius: float,
    max_frames: int,
    fps: int,
) -> Path:
    t = _row_array(rows, "t")
    amplitude = _row_array(rows, "shape_amplitude")
    amplitude_fit = _row_array(rows, "shape_amplitude_fit")
    volume = _row_array(rows, "vol_local_projection_m3")
    volume0 = float(rows[0]["vol0_mesh_m3"])
    fheron = _row_array(rows, "fheron_local_projection_pa")
    rayleigh_frequency = float(rows[0].get("rayleigh_frequency_hz", 1.0 / max(float(t[-1] - t[0]), 1.0e-30)))
    unit_points, faces = icosphere(subdivisions, 1.0)
    reference_points = unit_points * float(radius)

    frame_count = min(len(rows), max(2, int(max_frames)))
    frame_indices = np.unique(np.linspace(0, len(rows) - 1, frame_count, dtype=int))
    radial_min = min(float(row["radial_ratio_min"]) for row in rows)
    radial_max = max(float(row["radial_ratio_max"]) for row in rows)
    limit = 1.12 * radial_max * float(radius)

    fig = plt.figure(figsize=(9.4, 6.8))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.35, 1.0), wspace=0.24, hspace=0.46)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_vol = fig.add_subplot(grid[1, 1])
    ax_force = fig.add_subplot(grid[2, 1])

    def draw_frame(frame_pos: int):
        row_idx = int(frame_indices[frame_pos])
        points, _volume_check = projected_l2_shape_points(
            unit_points,
            faces,
            radius=radius,
            amplitude=float(amplitude[row_idx]),
            target_volume=volume0,
        )

        ax3d.cla()
        ax_amp.cla()
        ax_vol.cla()
        ax_force.cla()

        reference = Poly3DCollection(
            reference_points[faces],
            facecolor=(0.0, 0.0, 0.0, 0.025),
            edgecolor=(0.25, 0.25, 0.25, 0.28),
            linewidth=0.25,
        )
        current = Poly3DCollection(
            points[faces],
            facecolor="#fdd0a2",
            edgecolor="#ff7f0e",
            linewidth=0.28,
            alpha=0.76,
        )
        ax3d.add_collection3d(reference)
        ax3d.add_collection3d(current)
        tet_wire = Line3DCollection(
            volumetric_star_tet_segments(points, faces),
            colors=(1.0, 0.45, 0.0, 0.28),
            linewidths=0.35,
        )
        ax3d.add_collection3d(tet_wire)
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_title("Incompressible volumetric star-tet mesh")
        ax3d.view_init(elev=20.0, azim=32.0 + 24.0 * float(t[row_idx]) / max(float(t[-1]), 1.0e-30))
        ax3d.text2D(
            0.02,
            0.02,
            f"t = {t[row_idx]:.5f} s\n"
            f"mode a2 = {amplitude[row_idx]:+.4f}\n"
            f"fit a2 = {amplitude_fit[row_idx]:+.4f}\n"
            f"AR = {float(rows[row_idx]['radial_ratio_max']) / float(rows[row_idx]['radial_ratio_min']):.4f}\n"
            f"V/V0 = {volume[row_idx] / volume0:.6f}\n"
            f"FHeron = {fheron[row_idx]:.3f} Pa",
            transform=ax3d.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4.0},
        )

        amp_pad = max(0.08 * float(np.ptp(amplitude)), 0.005)
        ax_amp.plot(t, amplitude, "k--", lw=1.7, label="theory")
        ax_amp.plot(t, amplitude_fit, color="#ff7f0e", lw=1.8, label="fit from mesh")
        ax_amp.plot(t[row_idx], amplitude_fit[row_idx], "o", color="#d62728", ms=5.0)
        ax_amp.set_xlim(float(t[0]), float(t[-1]))
        ax_amp.set_ylim(
            float(min(np.min(amplitude), np.min(amplitude_fit)) - amp_pad),
            float(max(np.max(amplitude), np.max(amplitude_fit)) + amp_pad),
        )
        ax_amp.set_ylabel("a2")
        ax_amp.set_title("Shape amplitude")
        ax_amp.grid(True, alpha=0.25)
        ax_amp.legend(frameon=False, fontsize=8, loc="upper right")

        ax_vol.plot(t, volume / volume0, color="#ff7f0e", lw=1.8)
        ax_vol.plot(t[row_idx], volume[row_idx] / volume0, "o", color="#d62728", ms=5.5)
        ax_vol.axhline(1.0, color="0.25", ls=":", lw=1.4)
        ax_vol.set_xlim(float(t[0]), float(t[-1]))
        ax_vol.set_ylim(0.98, 1.02)
        ax_vol.set_ylabel("V / V0")
        ax_vol.set_title("Projected constant volume")
        ax_vol.grid(True, alpha=0.25)

        ax_force.plot(t, fheron, color="#ff7f0e", lw=1.8)
        ax_force.plot(t[row_idx], fheron[row_idx], "o", color="#d62728", ms=5.5)
        ax_force.set_xlim(float(t[0]), float(t[-1]))
        force_formatter = ScalarFormatter(useOffset=False)
        force_formatter.set_scientific(False)
        ax_force.yaxis.set_major_formatter(force_formatter)
        pad = max(0.04 * float(np.ptp(fheron)), 1.0e-9)
        ax_force.set_ylim(float(np.min(fheron) - pad), float(np.max(fheron) + pad))
        ax_force.set_xlabel("t [s]")
        ax_force.set_ylabel("FHeron [Pa]")
        ax_force.set_title("Shape-curvature response")
        ax_force.grid(True, alpha=0.25)

        fig.suptitle(
            f"Incompressible local projection, Rayleigh f={rayleigh_frequency:.3f} Hz, radial range {radial_min:.4f}-{radial_max:.4f}",
            y=0.98,
        )

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)))
    gif_path = out_dir / "incompressible_sphere_projection.gif"
    animation.save(gif_path, writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)
    return gif_path


def main() -> int:
    args = parse_args()
    rayleigh_omega, rayleigh_frequency, rayleigh_period = rayleigh_lamb_frequency(
        args.rayleigh_mode,
        args.gamma,
        args.rho,
        args.radius,
    )
    t_final = (
        float(args.timeseries_final_time)
        if args.timeseries_final_time is not None
        else (float(rayleigh_period) if args.use_rayleigh_frequency else 1.0)
    )
    rows = [
        run_one(
            subdivisions=subdivision,
            radius=args.radius,
            gamma=args.gamma,
            bulk_modulus=args.bulk_modulus,
            fheron_repeats=args.fheron_repeats,
            closure_repeats=args.closure_repeats,
        )
        for subdivision in range(args.max_subdivision + 1)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = write_outputs(rows, args.out_dir)
    plot_paths = write_plots(rows, args.out_dir, radius=args.radius, gamma=args.gamma)
    timeseries_rows = run_timeseries(
        subdivisions=args.timeseries_subdivision,
        radius=args.radius,
        gamma=args.gamma,
        bulk_modulus=args.bulk_modulus,
        pressure_drive_fraction=args.pressure_drive_fraction,
        n_steps=args.timeseries_steps,
        t_final=t_final,
    )
    timeseries_csv_path, timeseries_json_path = write_timeseries_outputs(timeseries_rows, args.out_dir)
    timeseries_plot_paths = write_timeseries_plots(timeseries_rows, args.out_dir)
    shape_mode_amplitude = (
        float(args.shape_mode_amplitude)
        if args.shape_mode_amplitude is not None
        else shape_amplitude_from_aspect_ratio(args.shape_mode_ar)
    )
    compressible_shape_rows = run_compressible_shape_timeseries(
        subdivisions=args.timeseries_subdivision,
        radius=args.radius,
        gamma=args.gamma,
        bulk_modulus=args.bulk_modulus,
        pressure_drive_fraction=args.pressure_drive_fraction,
        amplitude=shape_mode_amplitude,
        n_steps=args.timeseries_steps,
        t_final=t_final,
        rayleigh_frequency_hz=rayleigh_frequency,
        rayleigh_omega_rad_s=rayleigh_omega,
    )
    compressible_shape_csv_path, compressible_shape_json_path = write_compressible_shape_outputs(
        compressible_shape_rows,
        args.out_dir,
    )
    local_projection_rows = run_local_projection_timeseries(
        subdivisions=args.timeseries_subdivision,
        radius=args.radius,
        gamma=args.gamma,
        amplitude=shape_mode_amplitude,
        n_steps=args.timeseries_steps,
        t_final=t_final,
        rayleigh_frequency_hz=rayleigh_frequency,
        rayleigh_omega_rad_s=rayleigh_omega,
    )
    local_csv_path, local_json_path = write_local_projection_outputs(local_projection_rows, args.out_dir)
    (
        compressible_dynamic_rows,
        compressible_dynamic_snapshots,
        dynamic_faces,
        dynamic_tets,
        dynamic_surface_indices,
    ) = run_force_dynamic_timeseries(
        closure="compressible",
        subdivisions=args.dynamic_subdivision,
        radius=args.radius,
        gamma=args.gamma,
        rho=args.rho,
        bulk_modulus=args.bulk_modulus,
        amplitude=shape_mode_amplitude,
        n_steps=args.timeseries_steps,
        t_final=t_final,
        substeps_per_sample=args.dynamic_substeps_per_sample,
        damping_ratio=args.dynamic_damping_ratio,
        mass_model=args.dynamic_mass_model,
        rayleigh_frequency_hz=rayleigh_frequency,
        rayleigh_omega_rad_s=rayleigh_omega,
        rayleigh_mode=args.rayleigh_mode,
    )
    (
        incompressible_dynamic_rows,
        incompressible_dynamic_snapshots,
        _dynamic_faces,
        _dynamic_tets,
        _dynamic_surface_indices,
    ) = run_force_dynamic_timeseries(
        closure="incompressible",
        subdivisions=args.dynamic_subdivision,
        radius=args.radius,
        gamma=args.gamma,
        rho=args.rho,
        bulk_modulus=args.bulk_modulus,
        amplitude=shape_mode_amplitude,
        n_steps=args.timeseries_steps,
        t_final=t_final,
        substeps_per_sample=args.dynamic_substeps_per_sample,
        damping_ratio=args.dynamic_damping_ratio,
        mass_model=args.dynamic_mass_model,
        rayleigh_frequency_hz=rayleigh_frequency,
        rayleigh_omega_rad_s=rayleigh_omega,
        rayleigh_mode=args.rayleigh_mode,
    )
    compressible_dynamic_csv_path, compressible_dynamic_json_path = write_named_rows_outputs(
        compressible_dynamic_rows,
        args.out_dir,
        "sphere_fheron_compressible_dynamic_timeseries",
    )
    incompressible_dynamic_csv_path, incompressible_dynamic_json_path = write_named_rows_outputs(
        incompressible_dynamic_rows,
        args.out_dir,
        "sphere_fheron_incompressible_dynamic_timeseries",
    )
    compressible_dynamic_npz_path = write_dynamic_snapshots(
        compressible_dynamic_snapshots,
        dynamic_faces,
        args.out_dir,
        "sphere_fheron_compressible_dynamic_snapshots",
        tets=dynamic_tets,
        surface_indices=dynamic_surface_indices,
    )
    incompressible_dynamic_npz_path = write_dynamic_snapshots(
        incompressible_dynamic_snapshots,
        dynamic_faces,
        args.out_dir,
        "sphere_fheron_incompressible_dynamic_snapshots",
        tets=dynamic_tets,
        surface_indices=dynamic_surface_indices,
    )
    compressible_animation_path = write_force_dynamic_animation(
        compressible_dynamic_rows,
        compressible_dynamic_snapshots,
        dynamic_faces,
        args.out_dir,
        radius=args.radius,
        subdivisions=args.dynamic_subdivision,
        closure="compressible",
        max_frames=args.animation_max_frames,
        fps=args.animation_fps,
        volume_tets=dynamic_tets,
        surface_indices=dynamic_surface_indices,
    )
    incompressible_animation_path = write_force_dynamic_animation(
        incompressible_dynamic_rows,
        incompressible_dynamic_snapshots,
        dynamic_faces,
        args.out_dir,
        radius=args.radius,
        subdivisions=args.dynamic_subdivision,
        closure="incompressible",
        max_frames=args.animation_max_frames,
        fps=args.animation_fps,
        volume_tets=dynamic_tets,
        surface_indices=dynamic_surface_indices,
    )

    print(
        "sub  verts  FHeron[ms]  EOS[us]  Proj[us]  pTheory[Pa]  pEOS[Pa]  pProj[Pa]  "
        "HeronErr  ProjResidual"
    )
    for row in rows:
        print(
            f"{row['subdivisions']:>3d} "
            f"{row['vertices']:>6d} "
            f"{row['fheron_ms']:>11.4f} "
            f"{row['eos_us']:>8.4f} "
            f"{row['projection_us']:>9.4f} "
            f"{row['theory_pressure_pa']:>11.4f} "
            f"{row['eos_pressure_pa']:>9.4f} "
            f"{row['projection_pressure_pa']:>9.4f} "
            f"{row['heron_vs_theory_rel_rms']:>8.2e} "
            f"{row['projection_residual_rel_rms']:>12.2e}"
        )

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Rayleigh-Lamb l={args.rayleigh_mode}: f={rayleigh_frequency:.6f} Hz, T={rayleigh_period:.6e} s")
    print(f"CSV:  {timeseries_csv_path}")
    print(f"JSON: {timeseries_json_path}")
    print(f"CSV:  {compressible_shape_csv_path}")
    print(f"JSON: {compressible_shape_json_path}")
    print(f"CSV:  {local_csv_path}")
    print(f"JSON: {local_json_path}")
    print(f"CSV:  {compressible_dynamic_csv_path}")
    print(f"JSON: {compressible_dynamic_json_path}")
    print(f"NPZ:  {compressible_dynamic_npz_path}")
    print(f"CSV:  {incompressible_dynamic_csv_path}")
    print(f"JSON: {incompressible_dynamic_json_path}")
    print(f"NPZ:  {incompressible_dynamic_npz_path}")
    for plot_path in plot_paths:
        print(f"PNG:  {plot_path}")
    for plot_path in timeseries_plot_paths:
        print(f"PNG:  {plot_path}")
    print(f"GIF:  {compressible_animation_path}")
    print(f"GIF:  {incompressible_animation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
