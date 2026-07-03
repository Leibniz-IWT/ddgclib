#!/usr/bin/env python3
"""Shared mesh helpers for the top-level Figure 3-8 geometry scripts."""

from __future__ import annotations

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


def signed_surface_volume(points, faces) -> float:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    return float(
        sum(np.dot(points[a], np.cross(points[b], points[c])) / 6.0 for a, b, c in faces)
    )


def orient_positive_volume(points, faces):
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if signed_surface_volume(points, faces) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return points, faces


def compact_mesh(points, faces):
    used = np.unique(faces.ravel())
    remap = {int(old): new for new, old in enumerate(used)}
    new_points = points[used]
    new_faces = np.vectorize(lambda idx: remap[int(idx)])(faces).astype(int)
    return new_points, new_faces


def load_triangle_mesh(path, orient_faces):
    mesh = meshio.read(path)
    points = np.asarray(mesh.points[:, :3], dtype=float)
    tris = [np.asarray(block.data, dtype=int) for block in mesh.cells if block.type == "triangle"]
    if not tris:
        raise ValueError(f"No triangle cells found in {path}")
    faces = np.vstack(tris).astype(int)
    faces = orient_faces(points, faces)
    return orient_positive_volume(points, faces)


def load_outer_triangle_mesh(path, orient_faces):
    mesh = meshio.read(path)
    points = np.asarray(mesh.points[:, :3], dtype=float)
    tris = [np.asarray(block.data, dtype=int) for block in mesh.cells if block.type == "triangle"]
    if not tris:
        raise ValueError(f"No triangle cells found in {path}")
    faces = np.vstack(tris).astype(int)
    faces = orient_faces(points, faces)
    points, faces = compact_mesh(points, faces)
    return orient_positive_volume(points, faces)


def mesh_size(points, faces) -> float:
    edges = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    lengths = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])
    return float(lengths.mean())


def subdivide_projected(points, faces, project_midpoint, orient_faces):
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    points_list = [p.copy() for p in points]
    midpoint_cache = {}

    def midpoint_id(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key in midpoint_cache:
            return midpoint_cache[key]
        p = project_midpoint(points[key[0]], points[key[1]])
        idx = len(points_list)
        points_list.append(p)
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint_id(a, b)
        bc = midpoint_id(b, c)
        ca = midpoint_id(c, a)
        new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])

    points_new = np.asarray(points_list, dtype=float)
    faces_new = orient_faces(points_new, np.asarray(new_faces, dtype=int))
    return orient_positive_volume(points_new, faces_new)


def write_mesh(path, points, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(path, meshio.Mesh(points=np.asarray(points), cells=[("triangle", np.asarray(faces, dtype=int))]))
    return path
