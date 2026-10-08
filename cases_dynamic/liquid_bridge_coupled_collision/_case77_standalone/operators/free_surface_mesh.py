"""Utilities for axisymmetric 3D free-surface mesh states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class AxisymmetricFreeSurfaceMeshConfig:
    azimuthal_nodes: int = 72
    profile_nodes: int = 180
    include_axis_vertex: bool = True


def resample_profile(
    r_m: np.ndarray,
    h_m: np.ndarray,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a radial height profile onto a monotone radial grid."""

    r = np.asarray(r_m, dtype=float)
    h = np.asarray(h_m, dtype=float)
    order = np.argsort(r)
    r = r[order]
    h = h[order]
    keep = np.concatenate(([True], np.diff(r) > 1.0e-14))
    r = r[keep]
    h = h[keep]
    if r[0] > 0.0:
        r = np.concatenate(([0.0], r))
        h = np.concatenate(([h[0]], h))
    r_new = np.linspace(float(r[0]), float(r[-1]), int(nodes))
    h_new = np.interp(r_new, r, h)
    return r_new, h_new


def revolve_profile_to_mesh(
    r_m: np.ndarray,
    h_m: np.ndarray,
    config: AxisymmetricFreeSurfaceMeshConfig,
) -> dict[str, np.ndarray | list[int]]:
    """Revolve ``h(r)`` into an indexed triangular free-surface mesh.

    The axis is represented by one vertex, avoiding the degenerate duplicate
    center ring that makes Heron/cotangent surface operators unstable.
    """

    r, h = resample_profile(r_m, h_m, int(config.profile_nodes))
    n_theta = int(config.azimuthal_nodes)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices: list[list[float]] = []
    rings: list[list[int]] = []
    start_index = 0
    if bool(config.include_axis_vertex) and abs(float(r[0])) < 1.0e-14:
        vertices.append([0.0, 0.0, float(h[0])])
        rings.append([0] * n_theta)
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, r.size):
        ring: list[int] = []
        for j in range(n_theta):
            ring.append(len(vertices))
            vertices.append([float(r[i] * cos_t[j]), float(r[i] * sin_t[j]), float(h[i])])
        rings.append(ring)

    faces: list[tuple[int, int, int]] = []
    if rings and len(set(rings[0])) == 1 and len(rings) > 1:
        center = rings[0][0]
        for j in range(n_theta):
            jp = (j + 1) % n_theta
            faces.append((center, rings[1][j], rings[1][jp]))
        first_quad_ring = 1
    else:
        first_quad_ring = 0

    for i in range(first_quad_ring, len(rings) - 1):
        if len(set(rings[i])) == 1:
            continue
        for j in range(n_theta):
            jp = (j + 1) % n_theta
            faces.append((rings[i][j], rings[i + 1][j], rings[i + 1][jp]))
            faces.append((rings[i][j], rings[i + 1][jp], rings[i][jp]))

    return {
        "r_m": r,
        "h_m": h,
        "vertices_m": np.asarray(vertices, dtype=float),
        "faces": np.asarray(faces, dtype=np.int32),
        "ring_index": np.asarray(rings, dtype=np.int32),
    }


def revolve_attached_bridge_film_profile_to_mesh(
    r_m: np.ndarray,
    h_m: np.ndarray,
    config: AxisymmetricFreeSurfaceMeshConfig,
    *,
    sphere_radius_m: float,
    initial_film_thickness_m: float,
    sphere_bottom_z_m: float | None = None,
    bridge_rim_radius_m: float,
    bridge_head_m: float,
    bridge_nodes: int = 32,
) -> dict[str, np.ndarray]:
    """Revolve a connected bridge-plus-film free surface into a mesh.

    The returned surface is not a single-valued height field.  It contains a
    bridge meniscus patch that starts on the sphere surface and connects to
    the surrounding film at ``bridge_rim_radius_m``.  Rings outside that rim
    use the supplied film profile.  The top bridge ring is a boundary/contact
    ring on the sphere; the outer ring is the bridge-film contact/rim.
    """

    r_raw = np.asarray(r_m, dtype=float)
    h_raw = np.asarray(h_m, dtype=float)
    order = np.argsort(r_raw)
    r_raw = r_raw[order]
    h_raw = h_raw[order]
    keep = np.concatenate(([True], np.diff(r_raw) > 1.0e-14))
    r_raw = r_raw[keep]
    h_raw = h_raw[keep]

    sphere_radius = float(sphere_radius_m)
    h0 = float(initial_film_thickness_m)
    rim_r = float(np.clip(bridge_rim_radius_m, max(float(r_raw[0]), 1.0e-12), float(r_raw[-1])))
    rim_z = float(np.interp(rim_r, r_raw, h_raw))
    center_z = (h0 if sphere_bottom_z_m is None else float(sphere_bottom_z_m)) + sphere_radius
    contact_z = float(np.clip(bridge_head_m, h0, center_z + sphere_radius))
    contact_r_sq = sphere_radius * sphere_radius - (center_z - contact_z) ** 2
    if contact_r_sq < -1.0e-24:
        contact_r = rim_r
        contact_z = center_z - math.sqrt(max(sphere_radius * sphere_radius - contact_r * contact_r, 0.0))
    else:
        contact_r = math.sqrt(max(contact_r_sq, 0.0))

    bridge_n = max(3, int(bridge_nodes))
    s = np.linspace(0.0, 1.0, bridge_n)
    ease = s * s * (3.0 - 2.0 * s)
    # A small necking term gives a smooth inward meniscus without inventing an
    # additional free parameter that changes the contact/rim positions.
    neck = 0.10 * abs(rim_r - contact_r) * np.sin(math.pi * s)
    bridge_r = (1.0 - ease) * contact_r + ease * rim_r - neck
    bridge_z = (1.0 - ease) * contact_z + ease * rim_z
    bridge_r[0] = contact_r
    bridge_z[0] = contact_z
    bridge_r[-1] = rim_r
    bridge_z[-1] = rim_z

    film_mask = r_raw > rim_r + 1.0e-12
    film_r = np.concatenate(([rim_r], r_raw[film_mask]))
    film_z = np.concatenate(([rim_z], h_raw[film_mask]))
    if film_r.size < 2:
        film_r = np.concatenate((film_r, [float(r_raw[-1])]))
        film_z = np.concatenate((film_z, [float(h_raw[-1])]))

    # Avoid duplicating the bridge-film rim ring.
    surface_r = np.concatenate((bridge_r, film_r[1:]))
    surface_z = np.concatenate((bridge_z, film_z[1:]))
    ring_region = np.concatenate(
        (
            np.zeros(bridge_r.size, dtype=np.int32),
            np.ones(max(film_r.size - 1, 0), dtype=np.int32),
        )
    )

    n_theta = int(config.azimuthal_nodes)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices: list[list[float]] = []
    rings: list[list[int]] = []
    for radius, height in zip(surface_r, surface_z):
        ring: list[int] = []
        if abs(float(radius)) < 1.0e-14:
            vertices.append([0.0, 0.0, float(height)])
            ring = [len(vertices) - 1] * n_theta
        else:
            for j in range(n_theta):
                ring.append(len(vertices))
                vertices.append([float(radius * cos_t[j]), float(radius * sin_t[j]), float(height)])
        rings.append(ring)

    faces: list[tuple[int, int, int]] = []
    for i in range(len(rings) - 1):
        if len(set(rings[i])) == 1 and len(set(rings[i + 1])) > 1:
            center = rings[i][0]
            for j in range(n_theta):
                jp = (j + 1) % n_theta
                faces.append((center, rings[i + 1][j], rings[i + 1][jp]))
            continue
        if len(set(rings[i])) == 1 or len(set(rings[i + 1])) == 1:
            continue
        for j in range(n_theta):
            jp = (j + 1) % n_theta
            faces.append((rings[i][j], rings[i + 1][j], rings[i + 1][jp]))
            faces.append((rings[i][j], rings[i + 1][jp], rings[i][jp]))

    return {
        "r_m": np.asarray(surface_r, dtype=float),
        "h_m": np.asarray(surface_z, dtype=float),
        "vertices_m": np.asarray(vertices, dtype=float),
        "faces": np.asarray(faces, dtype=np.int32),
        "ring_index": np.asarray(rings, dtype=np.int32),
        "ring_region": ring_region,
        "bridge_contact_radius_m": np.asarray(contact_r, dtype=float),
        "bridge_contact_z_m": np.asarray(contact_z, dtype=float),
        "bridge_rim_radius_m": np.asarray(rim_r, dtype=float),
        "bridge_rim_z_m": np.asarray(rim_z, dtype=float),
    }


def triangle_areas(vertices_m: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_m, dtype=float)
    tri = vertices[np.asarray(faces, dtype=int)]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def projected_volume_under_surface_ul(vertices_m: np.ndarray, faces: np.ndarray) -> float:
    """Volume under a triangulated height field over the substrate plane."""

    vertices = np.asarray(vertices_m, dtype=float)
    tri = vertices[np.asarray(faces, dtype=int)]
    xy0 = tri[:, 0, :2]
    xy1 = tri[:, 1, :2]
    xy2 = tri[:, 2, :2]
    projected_area = 0.5 * np.abs(
        (xy1[:, 0] - xy0[:, 0]) * (xy2[:, 1] - xy0[:, 1])
        - (xy2[:, 0] - xy0[:, 0]) * (xy1[:, 1] - xy0[:, 1])
    )
    mean_z = np.mean(tri[:, :, 2], axis=1)
    return float(np.sum(projected_area * mean_z) * 1.0e9)


def mesh_summary(vertices_m: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    areas = triangle_areas(vertices_m, faces)
    edge_lengths: list[float] = []
    vertices = np.asarray(vertices_m, dtype=float)
    for face in np.asarray(faces, dtype=int):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_lengths.append(float(np.linalg.norm(vertices[face[a]] - vertices[face[b]])))
    edge_arr = np.asarray(edge_lengths, dtype=float)
    return {
        "vertices": int(np.asarray(vertices_m).shape[0]),
        "faces": int(np.asarray(faces).shape[0]),
        "surface_area_mm2": float(np.sum(areas) * 1.0e6),
        "projected_volume_ul": projected_volume_under_surface_ul(vertices_m, faces),
        "min_edge_um": float(np.min(edge_arr) * 1.0e6) if edge_arr.size else 0.0,
        "max_edge_um": float(np.max(edge_arr) * 1.0e6) if edge_arr.size else 0.0,
    }


def meridian_profile_from_mesh(mesh_state: dict, theta_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rings = np.asarray(mesh_state["ring_index"], dtype=int)
    vertices = np.asarray(mesh_state["vertices_m"], dtype=float)
    radii: list[float] = []
    heights: list[float] = []
    for ring in rings:
        vertex = vertices[int(ring[int(theta_index) % ring.size])]
        radii.append(float(math.hypot(vertex[0], vertex[1])))
        heights.append(float(vertex[2]))
    return np.asarray(radii, dtype=float), np.asarray(heights, dtype=float)


def write_obj(path: Path | str, vertices_m: np.ndarray, faces: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# ddgclib free-surface mesh\n")
        for x, y, z in np.asarray(vertices_m, dtype=float):
            f.write(f"v {x:.12e} {y:.12e} {z:.12e}\n")
        for a, b, c in np.asarray(faces, dtype=int):
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    return out


def write_msh(path: Path | str, vertices_m: np.ndarray, faces: np.ndarray) -> Path:
    """Write a minimal Gmsh 2.2 ASCII triangular surface mesh."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices_m, dtype=float)
    triangles = np.asarray(faces, dtype=int)
    with out.open("w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{vertices.shape[0]}\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i} {x:.12e} {y:.12e} {z:.12e}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{triangles.shape[0]}\n")
        for i, (a, b, c) in enumerate(triangles, start=1):
            f.write(f"{i} 2 2 1 1 {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
        f.write("$EndElements\n")
    return out


def write_npz(path: Path | str, mesh_state: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **mesh_state)
    return out
