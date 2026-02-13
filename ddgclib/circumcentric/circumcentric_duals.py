"""
Deprecated: use hyperct.ddg with method="circumcentric" instead.

This module is a compatibility shim. All circumcentric dual mesh
computation has been moved to hyperct.ddg.compute_vd(HC, method="circumcentric").
"""
import warnings
from functools import partial

warnings.warn(
    "ddgclib.circumcentric is deprecated. "
    "Use hyperct.ddg.compute_vd(HC, method='circumcentric') instead.",
    DeprecationWarning,
    stacklevel=2,
)

from hyperct.ddg import compute_vd as _compute_vd  # noqa: F401
from hyperct.ddg import e_star, v_star, d_area, circumcenter  # noqa: F401
from hyperct.ddg._geometry import (  # noqa: F401
    normalized,
    _set_boundary,
    _merge_local_duals_vector,
    _reflect_vertex_over_edge,
    _find_intersection,
    _find_plane_equation,
    area_of_polygon,
    volume_of_geometric_object,
)


def compute_vd(HC, cdist=1e-10):
    """Deprecated wrapper: calls hyperct.ddg.compute_vd with method='circumcentric'."""
    return _compute_vd(HC, method="circumcentric", cdist=cdist)
