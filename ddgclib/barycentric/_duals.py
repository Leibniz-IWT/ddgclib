"""
Deprecated: use hyperct.ddg instead.

This module re-exports all symbols from ddgclib.barycentric.__init__
for backward compatibility with code that imports from
``ddgclib.barycentric._duals``.
"""
import warnings

warnings.warn(
    "ddgclib.barycentric._duals is deprecated. Use hyperct.ddg instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the package-level shim
from ddgclib.barycentric import (  # noqa: F401
    compute_vd,
    e_star,
    v_star,
    d_area,
    normalized,
    _set_boundary,
    _merge_local_duals_vector,
    _reflect_vertex_over_edge,
    _find_intersection,
    _find_plane_equation,
    area_of_polygon,
    volume_of_geometric_object,
    dP,
    du,
    dudt,
    triang_dual,
    _signed_volume_parallelepiped,
    _volume_parallelepiped,
)

# Legacy functions that had different names — provide them as well
from hyperct.ddg import d_area  # noqa: F811


def plot_dual_mesh_2D(HC, tri=None, points=None):
    """Deprecated: use hyperct.ddg.plot_dual.plot_dual_mesh_2D instead."""
    from hyperct.ddg.plot_dual import plot_dual_mesh_2D as _plot
    return _plot(HC)


def plot_dual(vd=None, HC=None, **kwargs):
    """Deprecated: use hyperct.ddg.plot_dual functions instead."""
    warnings.warn(
        "plot_dual from _duals.py is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    pass
