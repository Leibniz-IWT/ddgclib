Architecture
============

This page describes the high-level design of ddgclib and the relationship
between its major components.

Module dependency graph
-----------------------

.. code-block:: text

   hyperct (external)
   ├── Complex          ← simplicial complex data structure
   ├── ddg              ← dual mesh computation (compute_vd, e_star, v_star)
   │   └── _geometry    ← geometry helpers (normalized, area_of_polygon, ...)
   ├── remesh           ← adaptive remeshing (2D)
   ├── _backend         ← NumPy / PyTorch / CUDA / multiprocessing
   └── _plotting        ← base mesh plotting

   ddgclib
   ├── operators/               ← core physics
   │   ├── stress.py            ← integrated Cauchy momentum (single-phase)
   │   ├── multiphase_stress.py ← multiphase Cauchy momentum + surface tension
   │   ├── surface_tension.py   ← surface tension from discrete mean curvature
   │   ├── curvature_2d.py      ← 2D integrated curvature operators
   │   ├── mass_redistribution.py ← pressure-preserving mass redistribution
   │   ├── curvature.py / area.py / volume.py / gradient.py ← pluggable method wrappers
   │   └── _registry.py         ← MethodRegistry for pluggable operators
   ├── multiphase.py            ← MultiphaseSystem, PhaseProperties
   ├── eos/                     ← equations of state (Tait-Murnaghan, IdealGas, MultiphaseEOS)
   ├── initial_conditions.py    ← IC framework (Composite, Hydrostatic, Multiphase, ...)
   ├── _boundary_conditions.py  ← BC framework (NoSlip, Dirichlet, Outlet, Periodic, ...)
   ├── dynamic_integrators/     ← time-stepping (Euler, Symplectic Euler, RK45, adaptive)
   ├── geometry/
   │   ├── domains/             ← domain builders (rectangle, cylinder, ball, droplet_in_box, ...)
   │   ├── _volume.py           ← volume conservation
   │   └── curved_volume/       ← curved surface volume pipeline
   ├── visualization/           ← matplotlib + polyscope plotting, animation, multiphase overlays
   ├── dem/                     ← discrete element method submodule
   ├── analytical.py            ← integrated validation utilities
   ├── _curvatures.py           ← legacy curvature pipeline
   └── _curvatures_heron.py     ← Heron-formula curvature (used by surface tension)

Vertex data model
-----------------

Every vertex ``v`` in ``HC.V`` carries attributes set by initial conditions
and updated by integrators:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Attribute
     - Type
     - Description
   * - ``v.x``
     - ``tuple``
     - Coordinate tuple (used as hash key)
   * - ``v.x_a``
     - ``ndarray``
     - Position array (mutable, advected by integrator)
   * - ``v.u``
     - ``ndarray``
     - Velocity vector
   * - ``v.p``
     - ``float``
     - Volume-averaged pressure over dual cell
   * - ``v.m``
     - ``float``
     - Mass associated with dual cell
   * - ``v.boundary``
     - ``bool``
     - Whether vertex lies on domain boundary
   * - ``v.nn``
     - ``set``
     - 1-ring neighbourhood (nearest neighbours)
   * - ``v.vd``
     - ``list``
     - Dual mesh vertices (set by ``compute_vd``)

For multiphase simulations, additional per-phase arrays are set by
``MultiphaseSystem.init_phase_fields()``:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Attribute
     - Type
     - Description
   * - ``v.phase``
     - ``int``
     - Phase ID (0, 1, ..., n_phases-1)
   * - ``v.is_interface``
     - ``bool``
     - True for sharp-interface vertices
   * - ``v.interface_phases``
     - ``frozenset``
     - Set of phase IDs present at interface vertex
   * - ``v.m_phase``
     - ``list[float]``
     - Per-phase mass
   * - ``v.p_phase``
     - ``list[float]``
     - Per-phase pressure
   * - ``v.rho_phase``
     - ``list[float]``
     - Per-phase density
   * - ``v.dual_vol_phase``
     - ``list[float]``
     - Per-phase dual cell sub-volume

Lagrangian formalism
--------------------

ddgclib follows a **Lagrangian formalism** -- the mesh moves with the fluid.
Vertices are advected by the velocity field and carry conserved quantities.

This differs from Eulerian methods where the mesh is fixed and the fluid
flows through it. The ``euler_velocity_only`` integrator exists for
validation/equilibrium checks only and should not be used for production
simulations.

FVM conventions
---------------

All scalar fields on vertices represent **volume-averaged** values over the
dual cell:

.. math::

   v.p = \frac{1}{\mathrm{Vol}_i} \int_{V_i} P(\mathbf{x})\, dV

This ensures ``v.p * Vol_i`` equals the integral to machine precision for
polynomial pressure fields. Use the ``volume_averaged_scalar`` utility or
IC classes (``HydrostaticPressure``, ``LinearPressureGradient``) which handle
this automatically.

For validation, always use **integrated comparisons** from
``ddgclib.analytical`` -- never point-wise ``abs(v.p - P(x_vertex))``.

Backend architecture
--------------------

Dual mesh computations can be dispatched to different backends:

.. code-block:: python

   from hyperct._backend import get_backend

   backend = get_backend("numpy")           # default
   backend = get_backend("torch")           # PyTorch CPU
   backend = get_backend("gpu")             # PyTorch CUDA (auto-detect)
   backend = get_backend("multiprocessing") # parallel CPU

Pass ``backend="torch"`` to ``compute_vd()`` for GPU-accelerated dual mesh
computation.

Test architecture
-----------------

The test suite (~415 tests) covers:

- **Operators**: stress tensor, surface tension, curvature, gradient, mass redistribution
- **Multiphase**: phase assignment, interface detection, dual splitting, EOS, surface tension
- **Boundary conditions**: no-slip, Dirichlet, outlet, periodic inlet
- **Initial conditions**: pressure, velocity, mass, multiphase ICs
- **Domain builders**: all 2D/3D geometries, boundary groups, projections
- **DEM**: particles, contacts, force models, bonds, bridges, coupling, I/O
- **Integration**: convergence, energy conservation, manuscript tutorials
- **Integrated validation**: linear precision, method comparison, convergence

Run with ``pytest ddgclib/tests/ -v -m "not slow"`` for the fast suite.
