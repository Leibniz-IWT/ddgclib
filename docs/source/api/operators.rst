Operators
=========

The ``ddgclib.operators`` package provides the core computational operators
for the Lagrangian FVM pipeline.

.. contents:: Submodules
   :local:
   :depth: 1

Stress tensor (single-phase)
-----------------------------

Integrated Cauchy momentum equation on barycentric/circumcentric dual meshes.

.. automodule:: ddgclib.operators.stress
   :members:
   :undoc-members:
   :show-inheritance:

Multiphase stress
-----------------

Multiphase Cauchy momentum with per-phase pressure/viscosity and surface
tension. See :doc:`/guides/multiphase` for usage.

.. automodule:: ddgclib.operators.multiphase_stress
   :members:
   :undoc-members:
   :show-inheritance:

Surface tension
---------------

Surface tension from discrete mean curvature (Heron-formula cotangent
weights). Drop-in ``dudt_fn`` for surface meshes without volume duals.

.. automodule:: ddgclib.operators.surface_tension
   :members:
   :undoc-members:
   :show-inheritance:

2D curvature operators
----------------------

Integrated curvature for 2D curves via the fundamental theorem of calculus.

.. automodule:: ddgclib.operators.curvature_2d
   :members:
   :undoc-members:
   :show-inheritance:

Mass redistribution
-------------------

Pressure-preserving mass redistribution after retriangulation, for both
single-phase and multiphase simulations.

.. automodule:: ddgclib.operators.mass_redistribution
   :members:
   :undoc-members:
   :show-inheritance:

Gradient wrappers
-----------------

Thin wrappers around stress operators for pressure gradient, velocity
Laplacian, and acceleration.

.. automodule:: ddgclib.operators.gradient
   :members:
   :undoc-members:
   :show-inheritance:

Pluggable method wrappers
-------------------------

Curvature estimators
^^^^^^^^^^^^^^^^^^^^

.. automodule:: ddgclib.operators.curvature
   :members:
   :undoc-members:
   :show-inheritance:

Area estimators
^^^^^^^^^^^^^^^

.. automodule:: ddgclib.operators.area
   :members:
   :undoc-members:
   :show-inheritance:

Volume estimators
^^^^^^^^^^^^^^^^^

.. automodule:: ddgclib.operators.volume
   :members:
   :undoc-members:
   :show-inheritance:

Method registry
^^^^^^^^^^^^^^^

.. automodule:: ddgclib.operators._registry
   :members:
   :undoc-members:
   :show-inheritance:
