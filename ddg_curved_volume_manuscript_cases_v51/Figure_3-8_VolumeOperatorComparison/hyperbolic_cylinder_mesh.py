#!/usr/bin/env python3
"""Create the hyperbolic-cylinder meshes used by Figure 3-8f."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import meshio
import numpy as np
from mesh_geometry_common import compact_mesh, mesh_size, orient_positive_volume, subdivide_projected, write_mesh

SOURCE_MESH = HERE / "msh" / "snapped_x2_minus_y2_ascii.msh"
X_MAX = 3.0
Y_MAX = 2.0
Z_MIN = 0.0
Z_MAX = 2.0


def is_curved_point(point, tol=1e-8):
    x, y, _z = point
    return abs(x * x - y * y - 1.0) < tol and x > 0.0


def is_curved_face(points, face, tol=1e-8):
    return bool(all(is_curved_point(p, tol=tol) for p in points[face]))


def is_xright_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 0], X_MAX, atol=tol))


def is_ytop_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 1], Y_MAX, atol=tol))


def is_ybottom_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 1], -Y_MAX, atol=tol))


def is_zcap_face(points, face, tol=1e-8):
    z = points[face, 2]
    return bool(np.allclose(z, Z_MAX, atol=tol) or np.allclose(z, Z_MIN, atol=tol))


def analytical_outward_normal(point):
    x, y, _z = point
    return np.array([-2.0 * x, 2.0 * y, 0.0], dtype=float)


def orient_faces(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if is_xright_face(points, face):
            desired = np.array([1.0, 0.0, 0.0])
        elif is_ytop_face(points, face):
            desired = np.array([0.0, 1.0, 0.0])
        elif is_ybottom_face(points, face):
            desired = np.array([0.0, -1.0, 0.0])
        elif is_zcap_face(points, face):
            desired = np.array([0.0, 0.0, 1.0 if pts[:, 2].mean() > 0.5 * (Z_MIN + Z_MAX) else -1.0])
        elif is_curved_face(points, face):
            desired = analytical_outward_normal(pts.mean(axis=0))
        else:
            raise ValueError(f"Could not classify boundary face {i}: {pts}")
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def load_outer_mesh():
    mesh = meshio.read(SOURCE_MESH)
    points = np.asarray(mesh.points[:, :3], dtype=float)
    z1_id = int(mesh.field_data["Z1Plane"][0]) if "Z1Plane" in mesh.field_data else None
    tris = []
    for block_i, cell_block in enumerate(mesh.cells):
        if cell_block.type != "triangle":
            continue
        data = np.asarray(cell_block.data, dtype=int)
        if z1_id is not None and "gmsh:physical" in mesh.cell_data:
            phys = mesh.cell_data["gmsh:physical"][block_i]
            data = data[phys != z1_id]
        if len(data):
            tris.append(data)
    if not tris:
        raise ValueError(f"No exterior triangle cells found in {SOURCE_MESH}")
    faces = np.vstack(tris).astype(int)
    points, faces = compact_mesh(points, faces)
    faces = orient_faces(points, faces)
    return orient_positive_volume(points, faces)


def project_curved_point(point):
    out = np.array(point, dtype=float)
    out[0] = math.sqrt(1.0 + out[1] * out[1])
    return out


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    both_curved = is_curved_point(a) and is_curved_point(b)
    same_xright = abs(a[0] - X_MAX) < 1e-8 and abs(b[0] - X_MAX) < 1e-8
    same_ytop = abs(a[1] - Y_MAX) < 1e-8 and abs(b[1] - Y_MAX) < 1e-8
    same_ybottom = abs(a[1] + Y_MAX) < 1e-8 and abs(b[1] + Y_MAX) < 1e-8
    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8
    if same_ytop:
        mid[1] = Y_MAX
    elif same_ybottom:
        mid[1] = -Y_MAX
    if both_curved:
        mid = project_curved_point(mid)
    elif same_xright:
        mid[0] = X_MAX
    if same_top:
        mid[2] = Z_MAX
    elif same_bottom:
        mid[2] = Z_MIN
    return mid


def generate_mesh(level: int = 0):
    points, faces = load_outer_mesh()
    for _ in range(level):
        points, faces = subdivide_projected(points, faces, project_midpoint, orient_faces)
    return points, faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    points, faces = generate_mesh(args.level)
    print(f"hyperbolic_cylinder level={args.level} points={len(points)} faces={len(faces)} h={mesh_size(points, faces):.6e}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
