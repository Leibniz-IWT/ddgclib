"""Visualize all parametric surface geometries using polyscope.

Builds triangle faces from mesh connectivity and registers each surface
as a proper polyscope surface mesh, with boundary vertices highlighted.

Usage::

    python tutorials/visualize_parametric_surfaces.py
    python tutorials/visualize_parametric_surfaces.py --refinement 3

Requires ``polyscope`` (pip install polyscope).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ddgclib.geometry import (
    parametric_surface,
    sphere,
    catenoid,
    cylinder,
    hyperboloid,
    torus,
    plane,
    translate_surface,
)

try:
    import polyscope as ps
except ImportError:
    print("polyscope is required.  Install with:  pip install polyscope")
    sys.exit(1)


def _extract_triangles(HC):
    """Extract triangle faces from mesh connectivity.

    Finds all vertex triples (v_i, v_j, v_k) where all three are
    mutually connected (form a triangle in the simplicial complex).
    """
    v_list = list(HC.V)
    v_to_idx = {id(v): i for i, v in enumerate(v_list)}

    triangles = set()
    for v in v_list:
        v_nn = set(v.nn)
        for n1 in v.nn:
            # Find common neighbours of v and n1 (triangle completion)
            for n2 in n1.nn:
                if n2 in v_nn and id(n2) != id(v):
                    tri = tuple(sorted([v_to_idx[id(v)],
                                        v_to_idx[id(n1)],
                                        v_to_idx[id(n2)]]))
                    triangles.add(tri)

    return np.array(sorted(triangles)) if triangles else np.empty((0, 3), dtype=int)


def _extract_edges(HC):
    """Extract unique edges from mesh connectivity."""
    v_list = list(HC.V)
    v_to_idx = {id(v): i for i, v in enumerate(v_list)}

    edges = set()
    for v in v_list:
        for nb in v.nn:
            edge = tuple(sorted([v_to_idx[id(v)], v_to_idx[id(nb)]]))
            edges.add(edge)

    return np.array(sorted(edges)) if edges else np.empty((0, 2), dtype=int)


def _register_surface(HC, bV, name: str):
    """Register a surface mesh in polyscope with edges and boundary coloring."""
    v_list = list(HC.V)
    positions = np.array([v.x_a[:3] for v in v_list], dtype=np.float64)
    is_boundary = np.array([1.0 if v in bV else 0.0 for v in v_list])

    triangles = _extract_triangles(HC)

    if len(triangles) > 0:
        mesh = ps.register_surface_mesh(name, positions, triangles)
        mesh.add_scalar_quantity("boundary", is_boundary, defined_on="vertices",
                                cmap="coolwarm", enabled=False)
    else:
        # Fallback to point cloud + edges
        mesh = ps.register_point_cloud(name, positions)
        mesh.add_scalar_quantity("boundary", is_boundary, cmap="coolwarm")

    # Always add edge network for wireframe view
    edges = _extract_edges(HC)
    if len(edges) > 0:
        net = ps.register_curve_network(f"{name}_edges", positions, edges)
        net.set_color((0.3, 0.3, 0.3))
        net.set_radius(0.003, relative=False)
        net.set_transparency(0.3)
        net.set_enabled(False)  # hidden by default, toggle in UI

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Visualize parametric surfaces in polyscope"
    )
    parser.add_argument("--refinement", type=int, default=3,
                        help="Mesh refinement level (default: 3)")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save screenshot and exit")
    args = parser.parse_args()
    ref = args.refinement

    # Generate all surfaces
    surfaces = {
        "sphere": sphere(R=1.0, refinement=ref),
        "catenoid": catenoid(a=1.0, v_range=(-1.5, 1.5), refinement=ref),
        "cylinder": cylinder(R=1.0, h_range=(-1.5, 1.5), refinement=ref),
        "hyperboloid": hyperboloid(a=1.0, c=1.0, v_range=(-1.5, 1.5),
                                   refinement=ref),
        "torus": torus(R=2.0, r=0.5, refinement=ref),
        "plane": plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5),
                       refinement=ref),
    }

    # Custom paraboloid
    def paraboloid(u, v):
        return (u, v, 0.3 * (u**2 + v**2))
    surfaces["paraboloid"] = parametric_surface(
        paraboloid, [(-1.5, 1.5), (-1.5, 1.5)], refinement=ref,
    )

    # Lay out surfaces in a grid with spacing
    spacing = 5.0
    positions = [
        (-spacing, 0, spacing),    # sphere
        (0, 0, spacing),           # catenoid
        (spacing, 0, spacing),     # cylinder
        (-spacing, 0, 0),          # hyperboloid
        (0, 0, 0),                 # torus
        (spacing, 0, 0),           # plane
        (0, 0, -spacing),          # paraboloid
    ]

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("y_up")
    ps.set_automatically_compute_scene_extents(True)

    for (name, (HC, bV)), offset in zip(surfaces.items(), positions):
        # Translate to grid position
        translate_surface(HC, list(offset))

        # Register with proper triangle extraction
        _register_surface(HC, bV, name)

        n = HC.V.size()
        nb = len(bV)
        print(f"  {name:<14} {n:>4} vertices, {nb:>3} boundary  @ "
              f"({offset[0]:+.0f}, {offset[1]:+.0f}, {offset[2]:+.0f})")

    print(f"\n  Total: {len(surfaces)} surfaces rendered (refinement={ref})")

    if args.screenshot:
        fig_dir = Path(__file__).parent / "fig" / "parametric"
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Set camera to see all surfaces
        ps.look_at((0, 12, 15), (0, 0, 0))
        ps.screenshot(str(fig_dir / "polyscope_surfaces.png"))
        print(f"  Screenshot saved to {fig_dir / 'polyscope_surfaces.png'}")
    else:
        ps.show()


if __name__ == "__main__":
    main()
