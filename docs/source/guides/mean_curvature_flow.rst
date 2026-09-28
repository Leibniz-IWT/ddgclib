Mean Curvature Flow
===================

The legacy mean curvature flow pipeline predates the dynamic Lagrangian
integrators and operates on surface meshes (not volume meshes).

Overview
--------

Mean curvature flow evolves a surface :math:`\Gamma` by moving each vertex
in the direction of its mean curvature normal:

.. math::

   \frac{\partial \mathbf{x}}{\partial t} = H \mathbf{n}

where :math:`H` is the mean curvature and :math:`\mathbf{n}` is the
outward unit normal.

Pipeline
--------

.. code-block:: python

   from ddgclib._curvatures import construct_HC, HC_curvatures, curvatures

   # Build surface mesh
   HC, bV = construct_HC(vertices, simplices)

   # Compute curvatures
   HC_curvatures(HC)

   # Access per-vertex curvature
   for v in HC.V:
       print(v.H)   # mean curvature
       print(v.K)    # Gaussian curvature

Integrators
-----------

Mean curvature flow integrators in ``ddgclib/mean_flow_integrators/``:

- **Euler** -- explicit forward Euler
- **Adams-Bashforth** -- multi-step method
- **Newton-Raphson** -- implicit solver with line search

Method wrappers
---------------

Pluggable curvature, area, and volume methods are registered via the
``MethodRegistry`` in ``ddgclib/operators/_registry.py`` and the wrapper
classes in ``ddgclib/_method_wrappers.py``:

- ``Curvature_i``, ``Curvature_ijk`` -- curvature method selection
- ``Area_i``, ``Area_ijk``, ``Area`` -- area computation methods
- ``Volume``, ``Volume_i`` -- volume computation methods

Volume and area
---------------

Volume conservation utilities in ``ddgclib/geometry/_volume.py``.
Curved surface volume computations in ``ddgclib/geometry/curved_volume/``.

Tutorials
---------

Jupyter notebook tutorials in ``tutorials/``:

- **Case study 1**: Capillary rise
- **Case study 2**: Particle-particle bridge
- **Case study 3**: Sessile droplet comparison

Additional case studies in ``cases_mean_flow/``.
