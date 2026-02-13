"""Visualization utilities for ddgclib simulations.

Submodules
----------
matplotlib_1d : 1D scalar/velocity profile plots
matplotlib_2d : 2D scalar field, vector field, and mesh plots
matplotlib_3d : 3D scatter plots and slice profile extraction
polyscope_3d  : Polyscope point cloud registration (optional)
animation     : Time-series animation from StateHistory
unified       : Dimension-dispatching wrappers (plot_primal, plot_dual)
"""

from ddgclib.visualization.matplotlib_1d import (
    plot_scalar_field_1d,
    plot_velocity_profile_1d,
)
from ddgclib.visualization.matplotlib_2d import (
    plot_scalar_field_2d,
    plot_vector_field_2d,
    plot_mesh_2d,
)
from ddgclib.visualization.matplotlib_3d import (
    plot_scalar_field_3d,
    extract_slice_profile,
)
from ddgclib.visualization.unified import (
    plot_primal,
    plot_dual,
    plot_fluid,
    plot_fluid_ps,
    dynamic_plot_fluid,
    dynamic_plot_fluid_polyscope,
)

__all__ = [
    'plot_scalar_field_1d',
    'plot_velocity_profile_1d',
    'plot_scalar_field_2d',
    'plot_vector_field_2d',
    'plot_mesh_2d',
    'plot_scalar_field_3d',
    'extract_slice_profile',
    'plot_primal',
    'plot_dual',
    'plot_fluid',
    'plot_fluid_ps',
    'dynamic_plot_fluid',
    'dynamic_plot_fluid_polyscope',
]
