ddgclib -- Discrete Differential Geometry Curvature Library
===========================================================

**ddgclib** is a Python library for discrete differential geometry curvature
computations in Lagrangian fluid simulations. It provides finite volume
operators on simplicial complexes for mean curvature flow, multiphase
dynamics, and problems with complex or changing topologies.

.. note::

   ddgclib is in **Alpha** (v0.4.3) and under active development. APIs may
   change between minor releases.

Key capabilities:

- **Lagrangian FVM on simplicial meshes** -- vertices carry velocity, pressure,
  and mass and are advected with the flow.
- **Multiphase flows** -- sharp interface tracking with per-phase EOS, surface
  tension, and dual-volume splitting.
- **Discrete Differential Geometry operators** -- integrated Cauchy stress
  tensor, mean curvature, and gradient operators on barycentric/circumcentric
  dual meshes.
- **Discrete Element Method (DEM)** -- spherical particle dynamics with
  Hertz contact, sintered bonds, capillary bridges, and two-way fluid coupling.
- **GPU acceleration** -- optional PyTorch backend for dual mesh computations.
- **Domain builders** -- one-liner mesh construction for common CFD geometries
  (rectangles, cylinders, spheres, multiphase droplets).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/simulation_workflow
   guides/multiphase
   guides/dem
   guides/mean_curvature_flow
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/operators
   api/multiphase
   api/eos
   api/boundary_conditions
   api/initial_conditions
   api/integrators
   api/geometry
   api/visualization
   api/dem

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
