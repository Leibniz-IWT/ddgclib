"""
Deprecated: use hyperct.ddg instead.

This module is a compatibility shim that re-exports functions from
hyperct.ddg. All dual mesh computation, discrete operators, and
geometry helpers have been moved to hyperct.ddg.
"""
import warnings

warnings.warn(
    "ddgclib.barycentric is deprecated. Use hyperct.ddg instead.",
    DeprecationWarning,
    stacklevel=2,
)

from hyperct.ddg import compute_vd, e_star, v_star, d_area  # noqa: F401
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
from ddgclib.operators.gradient import (  # noqa: F401
    pressure_gradient as dP,
    velocity_laplacian as du,
    acceleration as dudt,
)
from ddgclib._compat import (  # noqa: F401
    triang_dual,
    _signed_volume_parallelepiped,
    _volume_parallelepiped,
)

__all__ = [s for s in dir() if not s.startswith('_')]
