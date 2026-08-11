Geometry and Domain Builders
============================

One-liner functions for constructing common CFD simulation domains. Each
returns a ``DomainResult`` with the mesh, boundary vertices, and named
boundary groups.

Domain builders
---------------

.. automodule:: ddgclib.geometry.domains
   :members:
   :undoc-members:
   :show-inheritance:

2D domains
^^^^^^^^^^

- ``rectangle(L, h, refinement, flow_axis)`` -- rectangular channel
- ``l_shape(...)`` -- L-shaped domain
- ``disk(R, refinement)`` -- circular disk
- ``annulus(R_inner, R_outer, refinement)`` -- annular ring

3D domains
^^^^^^^^^^

- ``box(Lx, Ly, Lz, refinement)`` -- rectangular box
- ``cylinder_volume(R, L, refinement, flow_axis)`` -- cylindrical pipe
- ``pipe(R, L, refinement)`` -- alias for cylinder_volume
- ``ball(R, refinement)`` -- solid sphere

Multiphase domains
^^^^^^^^^^^^^^^^^^

- ``droplet_in_box_2d(R, L, refinement)`` -- circular droplet in 2D box
- ``droplet_in_box_3d(R, L, refinement)`` -- spherical droplet in 3D box

.. automodule:: ddgclib.geometry.domains._multiphase_droplet
   :members:
   :undoc-members:
   :show-inheritance:

DomainResult
^^^^^^^^^^^^

.. automodule:: ddgclib.geometry.domains._result
   :members:
   :undoc-members:
   :show-inheritance:

Projection utilities
^^^^^^^^^^^^^^^^^^^^

.. automodule:: ddgclib.geometry.domains._projection
   :members:
   :undoc-members:
   :show-inheritance:

Boundary groups
^^^^^^^^^^^^^^^

.. automodule:: ddgclib.geometry.domains._boundary_groups
   :members:
   :undoc-members:
   :show-inheritance:
