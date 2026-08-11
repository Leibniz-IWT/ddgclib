Simulation Workflow
===================

This guide describes the complete Lagrangian FVM pipeline for single-phase
dynamic simulations, from mesh creation to post-processing.

Pipeline overview
-----------------

.. code-block:: text

   1. Mesh construction     → HC, bV
   2. Boundary tagging      → v.boundary = True/False
   3. Dual mesh computation → compute_vd(HC, method=...)
   4. Initial conditions    → v.u, v.p, v.m
   5. Boundary conditions   → BoundaryConditionSet
   6. Stress operator       → dudt_fn via functools.partial
   7. Time integration      → symplectic_euler / rk45 / euler_adaptive
   8. Visualization         → plot_fluid / dynamic_plot_fluid

Step 1: Mesh construction
--------------------------

Use domain builders for standard geometries:

.. code-block:: python

   from ddgclib.geometry.domains import rectangle, cylinder_volume, ball

   # 2D channel
   result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)
   HC, bV = result.HC, result.bV

   # Access named boundaries
   walls = result.boundary_groups["walls"]
   inlet = result.boundary_groups["inlet"]
   outlet = result.boundary_groups["outlet"]

Available 2D builders: ``rectangle``, ``l_shape``, ``disk``, ``annulus``.

Available 3D builders: ``box``, ``cylinder_volume``, ``pipe``, ``ball``.

For custom geometries, use ``hyperct.Complex`` directly:

.. code-block:: python

   from hyperct import Complex

   HC = Complex(2, domain=[[0, 10], [0, 1]])

Step 2: Boundary tagging
--------------------------

Every vertex must have ``v.boundary`` set before computing dual meshes.
Domain builders do this automatically. For manual meshes:

.. code-block:: python

   for v in HC.V:
       v.boundary = v in bV

Step 3: Dual mesh computation
------------------------------

.. code-block:: python

   from hyperct.ddg import compute_vd

   compute_vd(HC, method="barycentric")   # recommended
   # or
   compute_vd(HC, method="circumcentric")  # requires Delaunay mesh

The ``"barycentric"`` method works on any triangulation. The
``"circumcentric"`` method requires a Delaunay mesh and can produce
degenerate duals on non-Delaunay meshes.

For GPU acceleration:

.. code-block:: python

   compute_vd(HC, method="barycentric", backend="torch")

Step 4: Initial conditions
---------------------------

Compose initial conditions using the IC framework:

.. code-block:: python

   from ddgclib.initial_conditions import (
       CompositeIC, ZeroVelocity, HydrostaticPressure,
       LinearPressureGradient, UniformMass,
   )

   ic = CompositeIC(
       ZeroVelocity(dim=2),
       HydrostaticPressure(rho=1000.0, g=9.81, axis=1, h_ref=1.0),
       UniformMass(total_volume=result.metadata["volume"], rho=1000.0),
   )
   ic.apply(HC, bV)

Available ICs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``ZeroVelocity``
     - Set ``v.u = 0`` on all vertices
   * - ``UniformPressure``
     - Set ``v.p = P0`` everywhere
   * - ``HydrostaticPressure``
     - Volume-averaged hydrostatic pressure ``P = P_ref + rho*g*(h_ref - x)``
   * - ``LinearPressureGradient``
     - Volume-averaged linear pressure along an axis
   * - ``UniformMass``
     - Set ``v.m = rho * dual_vol`` from total domain volume
   * - ``PhaseAssignment``
     - (Multiphase) assign ``v.phase`` from criterion function
   * - ``MultiphaseMass``
     - (Multiphase) per-phase mass from density and dual volume
   * - ``MultiphasePressure``
     - (Multiphase) per-phase pressure with optional Young-Laplace jump

.. important::

   Pressure fields are **volume-averaged** over dual cells, not point
   values. The IC classes handle this automatically when dual meshes are
   available. Never assign ``v.p = P(x_vertex)`` directly.

Step 5: Boundary conditions
----------------------------

.. code-block:: python

   from ddgclib._boundary_conditions import (
       BoundaryConditionSet, NoSlipWallBC, DirichletVelocityBC,
       OutletDeleteBC, PeriodicInletBC, PositionalNoSlipWallBC,
   )

   bc_set = BoundaryConditionSet()
   bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups["walls"])

Available BCs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``NoSlipWallBC``
     - Zero velocity on wall vertices
   * - ``PositionalNoSlipWallBC``
     - No-slip with position enforcement (Lagrangian walls)
   * - ``DirichletVelocityBC``
     - Prescribed velocity on boundary vertices
   * - ``OutletDeleteBC``
     - Remove vertices exiting the domain at the outlet
   * - ``PeriodicInletBC``
     - Inject vertices from ghost mesh at the inlet

For Lagrangian flows, use ``boundary_filter`` in integrators to control
which boundary vertices are frozen (excluded from integration). Typically
only wall vertices should be frozen, not inlet/outlet.

Step 6: Stress operator
------------------------

The Cauchy stress tensor operator computes the acceleration on each dual
cell. Bind parameters using ``functools.partial``:

.. code-block:: python

   from functools import partial
   from ddgclib.operators.stress import dudt_i

   dudt_fn = partial(dudt_i, dim=2, mu=0.001, HC=HC)

.. warning::

   Always bind ``dim``, ``mu``, ``HC`` via ``partial()``. Do **not** pass
   them as ``**dudt_kwargs`` -- this causes a "multiple values for HC"
   error.

The stress pipeline internally computes:

1. Dual area vectors :math:`\mathbf{A}_{ij}` between vertex pairs
2. Velocity difference tensor
3. Strain rate tensor
4. Cauchy stress :math:`\sigma = -p\mathbf{I} + 2\mu\dot{\epsilon}`
5. Integrated stress force :math:`\mathbf{F}_i = \sum_j \sigma \cdot \mathbf{A}_{ij}`
6. Acceleration :math:`\mathbf{a}_i = \mathbf{F}_i / m_i`

Step 7: Time integration
--------------------------

.. code-block:: python

   from ddgclib.dynamic_integrators import symplectic_euler

   symplectic_euler(
       HC, bV, dudt_fn,
       dt=1e-4,
       n_steps=10000,
       dim=2,
       bc_set=bc_set,
   )

Available integrators:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Integrator
     - Description
   * - ``euler``
     - Forward Euler (1st order, Lagrangian)
   * - ``symplectic_euler``
     - Symplectic Euler (1st order, better energy, Lagrangian)
   * - ``rk45``
     - Runge-Kutta 4/5 (adaptive, Lagrangian)
   * - ``euler_adaptive``
     - Adaptive Euler with error control
   * - ``euler_velocity_only``
     - Eulerian (fixed mesh) -- validation only

All Lagrangian integrators update both velocity and position. The mesh
moves with the fluid.

Remeshing
^^^^^^^^^

Lagrangian simulations require periodic retriangulation as the mesh
deforms. Pass ``remesh_mode`` to integrators:

.. code-block:: python

   symplectic_euler(
       ...,
       remesh_mode="delaunay",   # global Delaunay (default)
       # or
       remesh_mode="adaptive",   # local edge split/collapse/flip (2D)
       remesh_kwargs={"L_min": 0.01, "L_max": 0.05},
   )

Step 8: Visualization
----------------------

Static snapshots:

.. code-block:: python

   from ddgclib.visualization import plot_fluid
   plot_fluid(HC, bV)

Animations using ``StateHistory``:

.. code-block:: python

   from ddgclib.visualization import StateHistory, dynamic_plot_fluid

   history = StateHistory(fields=["u", "p"], record_every=50)

   # Pass history to integrator, or record manually:
   # history.record(HC)

   dynamic_plot_fluid(history, HC, save_path="flow.mp4")

For multiphase, pass ``phase_field="phase"`` and
``interface_field="is_interface"`` to overlay interface markers.

For interactive 3D replay with polyscope:

.. code-block:: bash

   python -m ddgclib.scripts.view_polyscope --snapshots results/snapshots/

Validation
----------

Use integrated comparisons from ``ddgclib.analytical``:

.. code-block:: python

   from ddgclib.analytical import (
       integrated_pressure_error,
       integrated_l2_norm,
       compare_stress_force,
       volume_averaged_scalar,
   )

Never use point-wise comparisons like ``abs(v.p - P(x_vertex))`` as these
conflate discretization error with the point-vs-average mismatch.

Benchmarks
^^^^^^^^^^

Run the integrated validation benchmarks:

.. code-block:: bash

   # Full suite
   python benchmarks/run_integrated_benchmarks.py

   # Linear precision check
   python benchmarks/run_integrated_benchmarks.py --linear-only

   # Convergence study
   python benchmarks/run_integrated_benchmarks.py --convergence --dim 2
