"""Visualize domain builder geometries using polyscope.

Registers each domain using the library's built-in polyscope visualization
from ``ddgclib.visualization.polyscope_3d``, with boundary groups highlighted
by color.

Usage::

    python tutorials/visualize_domains.py
    python tutorials/visualize_domains.py --refinement 3
    python tutorials/visualize_domains.py --screenshot
    python tutorials/visualize_domains.py --only cylinder ball

Requires ``polyscope`` (pip install polyscope).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ddgclib.geometry.domains import (
    DomainResult,
    rectangle,
    l_shape,
    disk,
    annulus,
    box,
    cylinder_volume,
    pipe,
    ball,
)
from ddgclib.visualization.polyscope_3d import (
    _check_polyscope,
    register_surface_mesh,
    register_point_cloud,
)


# ── Domain registration ─────────────────────────────────────────────────

def _register_domain(result: DomainResult, name: str):
    """Register a domain mesh in polyscope with boundary group coloring.

    Uses :func:`register_surface_mesh` (with Delaunay fallback) from the
    library's polyscope module, then adds boundary group scalar quantities.

    Parameters
    ----------
    result : DomainResult
        Domain builder result.
    name : str
        Polyscope structure name.

    Returns
    -------
    ps_struct
        Polyscope SurfaceMesh or PointCloud object.
    """
    HC = result.HC
    dim = result.dim
    v_list = list(HC.V)

    # Register mesh using the library's built-in function.
    # register_surface_mesh handles triangle extraction (vertex_face_mesh,
    # Delaunay fallback, point-cloud fallback) and 2D→3D padding.
    ps_struct = register_surface_mesh(HC, name=name, dim=dim)

    # Build boundary scalar quantities on the registered structure.
    # The vertex ordering must match what register_surface_mesh used.
    # For surface meshes built via Delaunay or vertex_face_mesh, the
    # vertex order follows HC.V iteration order.
    is_boundary = np.array(
        [1.0 if v in result.bV else 0.0 for v in v_list],
        dtype=np.float64,
    )

    # Boundary group ID: each vertex gets a numeric label for its group.
    group_names = list(result.boundary_groups.keys())
    group_id = np.full(len(v_list), -1.0)
    v_id_to_idx = {id(v): i for i, v in enumerate(v_list)}
    for gidx, gname in enumerate(group_names):
        for v in result.boundary_groups[gname]:
            vidx = v_id_to_idx.get(id(v))
            if vidx is not None:
                group_id[vidx] = float(gidx)

    # Add quantities — works for both SurfaceMesh and PointCloud.
    try:
        # SurfaceMesh path (defined_on="vertices")
        ps_struct.add_scalar_quantity(
            "boundary", is_boundary, defined_on="vertices",
            cmap="coolwarm", enabled=False,
        )
        ps_struct.add_scalar_quantity(
            "boundary_group", group_id, defined_on="vertices",
            cmap="spectral", enabled=True,
        )
    except Exception:
        # PointCloud fallback (no defined_on parameter)
        ps_struct.add_scalar_quantity(
            "boundary", is_boundary, cmap="coolwarm", enabled=False,
        )
        ps_struct.add_scalar_quantity(
            "boundary_group", group_id, cmap="spectral", enabled=True,
        )

    return ps_struct


# ── Domain builders ──────────────────────────────────────────────────────

def _build_all_domains(ref: int) -> dict[str, DomainResult]:
    """Build all available domains at the given refinement level."""
    return {
        "rectangle": rectangle(L=2.0, h=1.0, refinement=ref, flow_axis=0),
        "l_shape": l_shape(L=2.0, h=1.0, notch_L=1.0, notch_h=0.5,
                           refinement=ref),
        "disk": disk(R=1.0, refinement=ref),
        "annulus": annulus(R_outer=1.0, R_inner=0.3, refinement=ref),
        "box": box(Lx=2.0, Ly=1.0, Lz=1.0, refinement=max(1, ref - 1),
                   flow_axis=0),
        "cylinder": cylinder_volume(R=0.5, L=1.5, refinement=max(1, ref - 1),
                                    flow_axis=2),
        "ball": ball(R=1.0, refinement=max(1, ref - 1)),
    }


# Grid offsets for laying out domains in 3D space
_GRID_2D = {
    "rectangle": (-3.0, 0, 3.0),
    "l_shape":   ( 0.0, 0, 3.0),
    "disk":      ( 3.0, 0, 3.0),
    "annulus":   ( 6.0, 0, 3.0),
}

_GRID_3D = {
    "box":       (-3.0, 0, -2.0),
    "cylinder":  ( 1.0, 0, -2.0),
    "ball":      ( 4.0, 0, -2.0),
}


def _translate_result(result: DomainResult, offset: tuple):
    """Shift all vertices by offset (in-place)."""
    from ddgclib.geometry import translate_surface
    translate_surface(result.HC, list(offset))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ps = _check_polyscope()

    parser = argparse.ArgumentParser(
        description="Visualize domain builder geometries in polyscope"
    )
    parser.add_argument("--refinement", type=int, default=3,
                        help="Mesh refinement level (default: 3)")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save screenshot and exit")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only show these domains (e.g. --only disk cylinder)")
    args = parser.parse_args()

    ref = args.refinement
    all_domains = _build_all_domains(ref)

    # Filter if --only is given
    if args.only:
        all_domains = {k: v for k, v in all_domains.items() if k in args.only}
        if not all_domains:
            print(f"No matching domains. Available: {list(_build_all_domains(1).keys())}")
            sys.exit(1)

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("y_up")
    ps.set_automatically_compute_scene_extents(True)

    print(f"Domain Builder Viewer (refinement={ref})")
    print("-" * 50)

    for name, result in all_domains.items():
        # Apply grid offset
        offset = _GRID_2D.get(name) or _GRID_3D.get(name, (0, 0, 0))
        _translate_result(result, offset)

        # Register in polyscope
        _register_domain(result, name)

        # Print summary
        n = result.HC.V.size()
        nb = len(result.bV)
        groups = list(result.boundary_groups.keys())
        print(f"  {name:<12} {n:>5} verts, {nb:>4} boundary  "
              f"groups: {groups}")

    print(f"\n  Total: {len(all_domains)} domains rendered")
    print(f"  Boundary group coloring: spectral colormap (-1=interior)")

    if args.screenshot:
        fig_dir = Path(__file__).parent / "fig" / "domains"
        fig_dir.mkdir(parents=True, exist_ok=True)
        ps.look_at((0, 12, 15), (0, 0, 0))
        path = fig_dir / "polyscope_domains.png"
        ps.screenshot(str(path))
        print(f"  Screenshot saved to {path}")
    else:
        ps.show()


if __name__ == "__main__":
    main()
