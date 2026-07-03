#!/usr/bin/env python3
"""Create the hyperboloid meshes used by Figure 3-8d."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mesh_geometry_common import load_triangle_mesh, mesh_size, orient_positive_volume, subdivide_projected, write_mesh

HERE = Path(__file__).resolve().parent
SOURCE_MESH = HERE / "msh" / "coarse_hyperboloid.msh"
Z_MIN = -1.0
Z_MAX = 1.0
RIM_RADIUS = math.sqrt(2.0)


def analytical_outward_normal(point):
    x, y, z = point
    return np.array([x, y, -z], dtype=float)


def orient_faces(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        z = pts[:, 2]
        if np.allclose(z, Z_MAX, atol=1e-8):
            desired = np.array([0.0, 0.0, 1.0])
        elif np.allclose(z, Z_MIN, atol=1e-8):
            desired = np.array([0.0, 0.0, -1.0])
        else:
            desired = analytical_outward_normal(pts.mean(axis=0))
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def is_cap_point(point):
    return abs(point[2] - Z_MAX) < 1e-8 or abs(point[2] - Z_MIN) < 1e-8


def is_rim_point(point):
    return is_cap_point(point) and abs(math.hypot(point[0], point[1]) - math.sqrt(1.0 + point[2] * point[2])) < 1e-7


def project_side_point(point):
    x, y, z = point
    r = math.hypot(x, y)
    target = math.sqrt(1.0 + z * z)
    if r < 1e-14:
        return np.array([target, 0.0, z], dtype=float)
    return np.array([target * x / r, target * y / r, z], dtype=float)


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8
    if same_top or same_bottom:
        mid[2] = Z_MAX if same_top else Z_MIN
        if is_rim_point(a) and is_rim_point(b):
            r = math.hypot(mid[0], mid[1])
            if r > 1e-14:
                mid[0] *= RIM_RADIUS / r
                mid[1] *= RIM_RADIUS / r
        return mid
    return project_side_point(mid)


def generate_hyperboloid_mesh(nz, nth):
    points = []
    side_ids = []
    for z in np.linspace(Z_MIN, Z_MAX, nz):
        r = math.sqrt(1.0 + z * z)
        row = []
        for j in range(nth):
            th = 2.0 * math.pi * j / nth
            row.append(len(points))
            points.append([r * math.cos(th), r * math.sin(th), z])
        side_ids.append(row)
    top_center = len(points)
    points.append([0.0, 0.0, Z_MAX])
    bottom_center = len(points)
    points.append([0.0, 0.0, Z_MIN])
    faces = []
    for i in range(nz - 1):
        for j in range(nth):
            p00 = side_ids[i][j]
            p01 = side_ids[i][(j + 1) % nth]
            p10 = side_ids[i + 1][j]
            p11 = side_ids[i + 1][(j + 1) % nth]
            faces.append([p00, p10, p11])
            faces.append([p00, p11, p01])
    for j in range(nth):
        faces.append([top_center, side_ids[-1][j], side_ids[-1][(j + 1) % nth]])
        faces.append([bottom_center, side_ids[0][(j + 1) % nth], side_ids[0][j]])
    points = np.asarray(points, dtype=float)
    faces = orient_faces(points, np.asarray(faces, dtype=int))
    return orient_positive_volume(points, faces)


def generate_mesh(level: int = 0):
    points, faces = load_triangle_mesh(SOURCE_MESH, orient_faces)
    for _ in range(level):
        points, faces = subdivide_projected(points, faces, project_midpoint, orient_faces)
    return points, faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    points, faces = generate_mesh(args.level)
    print(f"hyperboloid level={args.level} points={len(points)} faces={len(faces)} h={mesh_size(points, faces):.6e}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
