"""Re-export shim for the 3D dual-volume split.

The 3D tangent-plane split lives next to the 2D split in
:mod:`ddgclib.geometry._dual_split_2d` (one file covers both
dimensions so the interface-neighbour / curve-adjacency helpers are
shared).  This shim exposes the 3D-specific symbols under a
dimension-matched module name for callers that prefer that import
path.
"""
from ddgclib.geometry._dual_split_2d import (  # noqa: F401
    _clip_tet_by_plane,
    _dual_volume_3d,
    _interface_plane_at_3d,
    _polyhedron_faces_to_tets,
    _tet_signed_volume,
    split_dual_polyhedron_3d,
)
