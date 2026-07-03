#!/usr/bin/env python3
"""Create the sphere meshes used by Figure 3-8a."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

from mesh_geometry_common import write_mesh  # noqa: E402

HELPER = COMMON / "sphere_paraboloid_order_test.py"


def _helper():
    spec = importlib.util.spec_from_file_location("sphere_mesh_helpers", HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_mesh(level: int = 0):
    mesh = _helper()
    points, faces = mesh.icosahedron()
    for _ in range(level):
        points, faces = mesh.subdivide(points, faces)
    return points, faces


def mesh_size(points, faces):
    return _helper().mesh_size(points, faces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    points, faces = generate_mesh(args.level)
    print(f"sphere level={args.level} points={len(points)} faces={len(faces)}")
    if args.out:
        print(write_mesh(args.out, points, faces))


if __name__ == "__main__":
    main()
