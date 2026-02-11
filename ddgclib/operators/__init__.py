"""
Discrete operators for geometric and physical computations.

This package provides pluggable computational methods for curvature, area,
volume, and gradient operations on simplicial complexes.

Submodules
----------
curvature : Curvature estimators (Curvature_i, Curvature_ijk)
area      : Area estimators (Area_i, Area_ijk, Area, DualArea_i)
volume    : Volume estimators (Volume, Volume_i)
gradient  : Discrete gradient/Laplacian operators (pressure_gradient, velocity_laplacian, acceleration)
"""

from ddgclib.operators._registry import MethodRegistry
from ddgclib.operators.curvature import Curvature_i, Curvature_ijk
from ddgclib.operators.area import Area_i, Area_ijk, Area, DualArea_i
from ddgclib.operators.volume import Volume, Volume_i
from ddgclib.operators.gradient import (
    pressure_gradient,
    velocity_laplacian,
    acceleration,
)

__all__ = [
    'MethodRegistry',
    'Curvature_i', 'Curvature_ijk',
    'Area_i', 'Area_ijk', 'Area', 'DualArea_i',
    'Volume', 'Volume_i',
    'pressure_gradient', 'velocity_laplacian', 'acceleration',
]
