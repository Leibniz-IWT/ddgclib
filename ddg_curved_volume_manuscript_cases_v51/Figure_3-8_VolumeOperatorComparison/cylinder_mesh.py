#!/usr/bin/env python3
"""Create the circular-cylinder meshes used by Figure 3-8b."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mesh_geometry_common import load_triangle_mesh, mesh_size, subdivide_projected, write_mesh

HERE = Path(__file__).resolve().parent
SOURCE_MESH = HERE / "msh" / "cylinder.msh"
RADIUS = 1.0
Z_MIN = -0.5
Z_MAX = 0.5


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
            p = pts.mean(axis=0)
            desired = np.array([p[0], p[1], 0.0])
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def is_cap_point(point):
    return abs(point[2] - Z_MAX) < 1e-8 or abs(point[2] - Z_MIN) < 1e-8


def is_rim_point(point):
    if not is_cap_point(point):
        return False
    return abs(math.hypot(point[0], point[1]) - RADIUS) < 1e-7


def project_side_point(point):
    x, y, z = point
    r = math.hypot(x, y)
    if r < 1e-14:
        return np.array([RADIUS, 0.0, z], dtype=float)
    return np.array([RADIUS * x / r, RADIUS * y / r, z], dtype=float)


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8
    if same_top or same_bottom:
        mid[2] = Z_MAX if same_top else Z_MIN
        if is_rim_point(a) and is_rim_point(b):
            r = math.hypot(mid[0], mid[1])
            if r > 1e-14:
                mid[0] *= RADIUS / r
                mid[1] *= RADIUS / r
        return mid
    return project_side_point(mid)


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
    print(f"cylinder level={args.level} points={len(points)} faces={len(faces)} h={mesh_size(points, faces):.6e}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
