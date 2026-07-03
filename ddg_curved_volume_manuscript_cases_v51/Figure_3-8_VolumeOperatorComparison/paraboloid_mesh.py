#!/usr/bin/env python3
"""Create the paraboloid-cap meshes used by Figure 3-8c."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mesh_geometry_common import load_triangle_mesh, mesh_size, subdivide_projected, write_mesh

HERE = Path(__file__).resolve().parent
SOURCE_MESH = HERE / "msh" / "snapped_paraboloid_ascii.msh"
TOP_Z = 2.0
TOP_RADIUS = math.sqrt(TOP_Z)
EVRARD_BACKGROUND_LEVELS = ((12, 3, 4), (18, 5, 8), (28, 8, 16), (84, 24, 48))


def is_top_point(point, tol=1e-8):
    return abs(point[2] - TOP_Z) < tol


def is_rim_point(point, tol=1e-7):
    return is_top_point(point, tol=tol) and abs(math.hypot(point[0], point[1]) - TOP_RADIUS) < tol


def is_curved_point(point, tol=1e-8):
    return abs(point[2] - point[0] * point[0] - point[1] * point[1]) < tol


def is_top_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 2], TOP_Z, atol=tol))


def is_paraboloid_wall_face(points, face, tol=1e-8):
    return bool(np.all(np.abs(points[face, 2] - points[face, 0] ** 2 - points[face, 1] ** 2) < tol))


def orient_faces(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if is_top_face(points, face):
            desired = np.array([0.0, 0.0, 1.0])
        elif is_paraboloid_wall_face(points, face):
            x, y, _z = pts.mean(axis=0)
            desired = np.array([2.0 * x, 2.0 * y, -1.0])
        else:
            raise ValueError(f"Could not classify paraboloid boundary face {i}: {pts}")
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def project_wall_point(point):
    x, y, z = point
    z = min(max(float(z), 0.0), TOP_Z)
    target = math.sqrt(z)
    r = math.hypot(x, y)
    if r < 1e-14:
        return np.array([target, 0.0, z], dtype=float)
    return np.array([target * x / r, target * y / r, z], dtype=float)


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    if is_top_point(a) and is_top_point(b):
        mid[2] = TOP_Z
        if is_rim_point(a) and is_rim_point(b):
            r = math.hypot(mid[0], mid[1])
            if r > 1e-14:
                mid[0] *= TOP_RADIUS / r
                mid[1] *= TOP_RADIUS / r
        return mid
    if is_curved_point(a) and is_curved_point(b):
        return project_wall_point(mid)
    return mid


def generate_mesh(level: int = 0):
    points, faces = load_triangle_mesh(SOURCE_MESH, orient_faces)
    for _ in range(level):
        points, faces = subdivide_projected(points, faces, project_midpoint, orient_faces)
    return points, faces


def add_triangular_prism_indices(tets, ids):
    for tet in ((0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)):
        tets.append(tuple(ids[i] for i in tet))


def generate_evrard_background_tets(level: int = 0):
    """Create independent PPIC/VOF-style background cells for the paraboloid cap."""
    nseg, nr, nz = EVRARD_BACKGROUND_LEVELS[level]
    z_values = np.linspace(-1.0e-9, TOP_Z, nz + 1)
    outer_radius = TOP_RADIUS / math.cos(math.pi / nseg) * (1.0 + 1.0e-12)
    r_values = np.linspace(0.0, outer_radius, nr + 1)
    points = []
    tets = []
    center_ids = []
    ring_ids = []
    for z in z_values:
        center_ids.append(len(points))
        points.append((0.0, 0.0, float(z)))
        layer = []
        for i in range(nseg):
            theta = 2.0 * math.pi * i / nseg
            radial_ids = []
            for radius in r_values[1:]:
                radial_ids.append(len(points))
                points.append((float(radius) * math.cos(theta), float(radius) * math.sin(theta), float(z)))
            layer.append(radial_ids)
        ring_ids.append(layer)

    def point_id(k, j, i):
        if j == 0:
            return center_ids[k]
        return ring_ids[k][i % nseg][j - 1]

    for k in range(nz):
        for i in range(nseg):
            ip = (i + 1) % nseg
            for j in range(nr):
                if j == 0:
                    ids = (point_id(k, 0, i), point_id(k, 1, i), point_id(k, 1, ip), point_id(k + 1, 0, i), point_id(k + 1, 1, i), point_id(k + 1, 1, ip))
                    add_triangular_prism_indices(tets, ids)
                else:
                    add_triangular_prism_indices(tets, (point_id(k, j, i), point_id(k, j + 1, i), point_id(k, j + 1, ip), point_id(k + 1, j, i), point_id(k + 1, j + 1, i), point_id(k + 1, j + 1, ip)))
                    add_triangular_prism_indices(tets, (point_id(k, j, i), point_id(k, j + 1, ip), point_id(k, j, ip), point_id(k + 1, j, i), point_id(k + 1, j + 1, ip), point_id(k + 1, j, ip)))
    return np.asarray(points, dtype=float), np.asarray(tets, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    points, faces = generate_mesh(args.level)
    print(f"paraboloid level={args.level} points={len(points)} faces={len(faces)} h={mesh_size(points, faces):.6e}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
