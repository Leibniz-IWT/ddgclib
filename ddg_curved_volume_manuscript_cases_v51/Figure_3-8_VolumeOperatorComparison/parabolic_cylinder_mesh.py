#!/usr/bin/env python3
"""Create the parabolic-cylinder meshes used by Figure 3-8e.

Level 0 is a coarser generated mesh. Level 1 is the Table 3-1 mesh.
Higher levels are midpoint refinements of the Table 3-1 mesh.
"""

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

SOURCE_MESH = HERE / "msh" / "snapped_y_eq_x2_ascii.msh"
Y_MAX = 2.0
Z_MIN = -1.0
Z_MAX = 1.0


def is_curved_point(point, tol=1e-8):
    return abs(point[1] - point[0] * point[0]) < tol


def is_yplus_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 1], Y_MAX, atol=tol))


def is_cap_face(points, face, tol=1e-8):
    z = points[face, 2]
    return bool(np.allclose(z, Z_MAX, atol=tol) or np.allclose(z, Z_MIN, atol=tol))


def is_curved_face(points, face, tol=1e-8):
    pts = points[face]
    return bool(np.all(np.abs(pts[:, 1] - pts[:, 0] * pts[:, 0]) < tol))


def analytical_outward_normal(point):
    x = float(point[0])
    return np.array([2.0 * x, -1.0, 0.0], dtype=float)


def orient_faces(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if is_cap_face(points, face):
            desired = np.array([0.0, 0.0, 1.0 if pts[:, 2].mean() > 0.0 else -1.0])
        elif is_yplus_face(points, face):
            desired = np.array([0.0, 1.0, 0.0])
        elif is_curved_face(points, face):
            desired = analytical_outward_normal(pts.mean(axis=0))
        else:
            raise ValueError(f"Could not classify boundary face {i}: {pts}")
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def remove_degenerate_faces(points, faces, tol=1e-12):
    kept = []
    for face in faces:
        if len(set(int(v) for v in face)) < 3:
            continue
        a, b, c = points[face]
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        if area2 > tol:
            kept.append(face)
    return np.asarray(kept, dtype=int)


def load_outer_mesh():
    mesh = meshio.read(SOURCE_MESH)
    points = np.asarray(mesh.points[:, :3], dtype=float)
    y1_id = int(mesh.field_data["Y1Plane"][0]) if "Y1Plane" in mesh.field_data else None
    tris = []
    for block_i, cell_block in enumerate(mesh.cells):
        if cell_block.type != "triangle":
            continue
        data = np.asarray(cell_block.data, dtype=int)
        if y1_id is not None and "gmsh:physical" in mesh.cell_data:
            phys = mesh.cell_data["gmsh:physical"][block_i]
            data = data[phys != y1_id]
        if len(data):
            tris.append(data)
    if not tris:
        raise ValueError(f"No exterior triangle cells found in {SOURCE_MESH}")
    faces = np.vstack(tris).astype(int)
    points, faces = compact_mesh(points, faces)
    faces = orient_faces(points, faces)
    return orient_positive_volume(points, faces)


def coarse_parametric_mesh(nx: int = 5, nz: int = 4, ne: int = 2):
    """Build one coarser closed parabolic-cylinder surface mesh."""

    points_list = []
    point_ids = {}
    faces = []

    def add_point(point):
        key = tuple(round(float(x), 12) for x in point)
        if key not in point_ids:
            point_ids[key] = len(points_list)
            points_list.append([float(x) for x in point])
        return point_ids[key]

    xs = np.linspace(-math.sqrt(2.0), math.sqrt(2.0), nx + 1)
    zs = np.linspace(Z_MIN, Z_MAX, nz + 1)
    etas = np.linspace(0.0, 1.0, ne + 1)

    curved_grid = [[add_point((x, x * x, z)) for z in zs] for x in xs]
    top_grid = [[add_point((x, Y_MAX, z)) for z in zs] for x in xs]

    for i in range(nx):
        for k in range(nz):
            a = curved_grid[i][k]
            b = curved_grid[i + 1][k]
            c = curved_grid[i + 1][k + 1]
            d = curved_grid[i][k + 1]
            faces.extend([[a, b, c], [a, c, d]])

            a = top_grid[i][k]
            b = top_grid[i][k + 1]
            c = top_grid[i + 1][k + 1]
            d = top_grid[i + 1][k]
            faces.extend([[a, b, c], [a, c, d]])

    for z in (Z_MIN, Z_MAX):
        cap_grid = [[None for _ in range(ne + 1)] for _ in range(nx + 1)]
        for i, x in enumerate(xs):
            for j, eta in enumerate(etas):
                y = x * x + eta * (Y_MAX - x * x)
                cap_grid[i][j] = add_point((x, y, z))
        for i in range(nx):
            for j in range(ne):
                a = cap_grid[i][j]
                b = cap_grid[i + 1][j]
                c = cap_grid[i + 1][j + 1]
                d = cap_grid[i][j + 1]
                faces.extend([[a, b, c], [a, c, d]])

    points = np.asarray(points_list, dtype=float)
    faces = np.asarray(faces, dtype=int)
    points, faces = compact_mesh(points, faces)
    faces = remove_degenerate_faces(points, faces)
    faces = orient_faces(points, faces)
    return orient_positive_volume(points, faces)


def project_curved_point(point):
    out = np.array(point, dtype=float)
    out[1] = out[0] * out[0]
    return out


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    both_curved = is_curved_point(a) and is_curved_point(b)
    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8
    same_yplus = abs(a[1] - Y_MAX) < 1e-8 and abs(b[1] - Y_MAX) < 1e-8
    if both_curved:
        mid = project_curved_point(mid)
    elif same_yplus:
        mid[1] = Y_MAX
    if same_top:
        mid[2] = Z_MAX
    elif same_bottom:
        mid[2] = Z_MIN
    return mid


def centroid_subdivide_projected(points, faces):
    points_list = [p.copy() for p in points]
    new_faces = []
    for face in faces:
        a, b, c = [int(v) for v in face]
        centroid = (points[a] + points[b] + points[c]) / 3.0
        if is_curved_face(points, face):
            centroid = project_curved_point(centroid)
        elif is_yplus_face(points, face):
            centroid[1] = Y_MAX
        elif is_cap_face(points, face):
            centroid[2] = Z_MAX if points[face, 2].mean() > 0.0 else Z_MIN
        center_id = len(points_list)
        points_list.append(centroid)
        new_faces.extend([[a, b, center_id], [b, c, center_id], [c, a, center_id]])
    points_new = np.asarray(points_list, dtype=float)
    faces_new = orient_faces(points_new, np.asarray(new_faces, dtype=int))
    return orient_positive_volume(points_new, faces_new)


def generate_mesh(level: int = 0):
    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0:
        return coarse_parametric_mesh()
    base_points, base_faces = load_outer_mesh()
    if level == 1:
        return base_points, base_faces
    points, faces = base_points, base_faces
    for _ in range(level - 1):
        points, faces = subdivide_projected(points, faces, project_midpoint, orient_faces)
    return points, faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    points, faces = generate_mesh(args.level)
    print(f"parabolic_cylinder level={args.level} points={len(points)} faces={len(faces)} h={mesh_size(points, faces):.6e}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
