"""Domain builder module for constructing common CFD simulation domains.

Provides one-liner functions that return :class:`DomainResult` objects with
the simplicial complex, boundary vertices, and named boundary groups ready
for use with initial conditions, boundary conditions, and integrators.

2D domains
----------
.. autosummary::
   rectangle
   l_shape
   disk
   annulus

3D domains
----------
.. autosummary::
   box
   cylinder_volume
   pipe
   ball

Projection utilities
--------------------
.. autosummary::
   cube_to_disk
   cube_to_sphere
   DISTRIBUTION_LAWS

Example
-------
>>> from ddgclib.geometry.domains import rectangle
>>> result = rectangle(L=10.0, h=1.0, refinement=3)
>>> result.summary()
'DomainResult: ... vertices, ... boundary, dim=2, groups=[...]'
>>> result.boundary_groups.keys()
dict_keys(['inlet', 'outlet', 'bottom_wall', 'top_wall', 'walls'])
"""

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._projection import (
    cube_to_disk,
    cube_to_sphere,
    DISTRIBUTION_LAWS,
)
from ddgclib.geometry.domains._boundary_groups import (
    identify_face_groups,
    identify_radial_boundary,
    identify_all_boundary,
)
from ddgclib.geometry.domains._rectangles import rectangle, l_shape
from ddgclib.geometry.domains._disks import disk, annulus
from ddgclib.geometry.domains._boxes import box
from ddgclib.geometry.domains._cylinders import cylinder_volume, pipe
from ddgclib.geometry.domains._spheres import ball
from ddgclib.geometry.domains._periodic import periodic_rectangle, periodic_box

__all__ = [
    # Result type
    'DomainResult',
    # 2D domains
    'rectangle', 'l_shape', 'disk', 'annulus',
    # 3D domains
    'box', 'cylinder_volume', 'pipe', 'ball',
    # Projection utilities
    'cube_to_disk', 'cube_to_sphere', 'DISTRIBUTION_LAWS',
    # Boundary helpers
    'identify_face_groups', 'identify_radial_boundary', 'identify_all_boundary',
    # Periodic domains
    'periodic_rectangle', 'periodic_box',
]
