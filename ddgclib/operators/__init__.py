"""
Discrete operators for geometric and physical computations.

This package provides pluggable computational methods for curvature, area,
volume, gradient, and stress operations on simplicial complexes.

Submodules
----------
curvature : Curvature estimators (Curvature_i, Curvature_ijk)
area      : Area estimators (Area_i, Area_ijk, Area, DualArea_i)
volume    : Volume estimators (Volume, Volume_i)
stress    : Cauchy stress tensor operators (dual_area_vector, cauchy_stress, stress_force, etc.)
gradient  : Thin wrappers around stress operators (pressure_gradient, velocity_laplacian, acceleration)
"""

from ddgclib.operators._registry import MethodRegistry
from ddgclib.operators.curvature import Curvature_i, Curvature_ijk
from ddgclib.operators.area import Area_i, Area_ijk, Area, DualArea_i
from ddgclib.operators.volume import Volume, Volume_i
from ddgclib.operators.stress import (
    dual_area_vector,
    dual_volume,
    cache_dual_volumes,
    velocity_difference_tensor,
    velocity_difference_tensor_pointwise,
    strain_rate,
    cauchy_stress,
    integrated_cauchy_stress,
    stress_force,
    stress_acceleration,
    dudt_i,
)
from ddgclib.operators.gradient import (
    pressure_gradient,
    velocity_laplacian,
    acceleration,
)
from ddgclib.operators.surface_tension import (
    surface_tension_force,
    surface_tension_acceleration,
    dual_area_heron,
)
from ddgclib.operators.curvature_2d import (
    integrated_curvature_normal_2d,
    surface_tension_force_2d,
    reconstruct_arc_length_and_bulge_area,
)
from ddgclib.operators.multiphase_stress import (
    multiphase_stress_force,
    multiphase_stress_acceleration,
    multiphase_dudt_i,
)
from ddgclib.operators.mass_redistribution import (
    snapshot_pressure,
    snapshot_pressure_multiphase,
    redistribute_mass_single_phase,
    redistribute_mass_multiphase,
)

__all__ = [
    'MethodRegistry',
    'Curvature_i', 'Curvature_ijk',
    'Area_i', 'Area_ijk', 'Area', 'DualArea_i',
    'Volume', 'Volume_i',
    'dual_area_vector', 'dual_volume', 'cache_dual_volumes',
    'velocity_difference_tensor', 'velocity_difference_tensor_pointwise',
    'strain_rate', 'cauchy_stress', 'integrated_cauchy_stress',
    'stress_force', 'stress_acceleration', 'dudt_i',
    'pressure_gradient', 'velocity_laplacian', 'acceleration',
    'surface_tension_force', 'surface_tension_acceleration', 'dual_area_heron',
    'integrated_curvature_normal_2d', 'surface_tension_force_2d',
    'reconstruct_arc_length_and_bulge_area',
    'multiphase_stress_force', 'multiphase_stress_acceleration', 'multiphase_dudt_i',
    'snapshot_pressure', 'snapshot_pressure_multiphase',
    'redistribute_mass_single_phase', 'redistribute_mass_multiphase',
]
