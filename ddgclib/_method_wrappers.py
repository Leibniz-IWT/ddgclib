"""
Backward-compatible shim for ddgclib._method_wrappers.

All classes have been moved to ddgclib.operators/ for better organization.
This module re-exports them so that existing code continues to work:

    from ddgclib._method_wrappers import Curvature_i, Volume  # still works
"""

import warnings

from ddgclib.operators.curvature import Curvature_i, Curvature_ijk
from ddgclib.operators.area import Area_i, Area_ijk, Area
from ddgclib.operators.volume import Volume, Volume_i

# Pre-instantiated singletons for backward compatibility
curvature_i = Curvature_i()
area_i = Area_i()
area_ijk = Area_ijk()
area = Area()
volume = Volume()
volume_i = Volume_i()

__all__ = [
    'Curvature_i', 'Curvature_ijk',
    'Area_i', 'Area_ijk', 'Area',
    'Volume', 'Volume_i',
    'curvature_i', 'area_i', 'area_ijk', 'area',
    'volume', 'volume_i',
]
