Multiphase Flows
================

ddgclib provides a complete framework for **sharp-interface multiphase
Lagrangian FVM** simulations on simplicial complexes. This guide covers the
data model, operators, and end-to-end workflow for two-phase flows with
surface tension.

Overview
--------

The multiphase pipeline extends the single-phase Cauchy stress tensor
framework with:

- **Phase labelling** -- each vertex carries an integer phase ID.
- **Sharp interface detection** -- interface vertices are identified as
  droplet-phase vertices neighbouring the outer phase.
- **Per-phase fields** -- mass, pressure, density, and dual volume are
  stored per-phase on every vertex.
- **Dual volume splitting** -- interface dual cells are split among phases
  using neighbour-count weighting.
- **Multiphase stress operator** -- pressure and viscous fluxes use
  phase-specific values; surface tension acts only on interface vertices.
- **Per-phase EOS** -- each phase has its own equation of state
  dispatched by ``MultiphaseEOS``.

Data model
----------

PhaseProperties
^^^^^^^^^^^^^^^

Each phase is described by a :class:`~ddgclib.multiphase.PhaseProperties`
dataclass:

.. code-block:: python

   from ddgclib.multiphase import PhaseProperties
   from ddgclib.eos import TaitMurnaghan

   liquid = PhaseProperties(
       eos=TaitMurnaghan(rho0=1000.0, P0=0.0, K=50.0, n=7.15),
       mu=0.01,        # dynamic viscosity [Pa.s]
       rho0=1000.0,    # reference density [kg/m^3]
       name="liquid",
   )
   gas = PhaseProperties(
       eos=TaitMurnaghan(rho0=1.0, P0=0.0, K=1.0, n=1.4),
       mu=0.001,
       rho0=1.0,
       name="gas",
   )

MultiphaseSystem
^^^^^^^^^^^^^^^^

The :class:`~ddgclib.multiphase.MultiphaseSystem` manages all multiphase
state on the mesh:

.. code-block:: python

   from ddgclib.multiphase import MultiphaseSystem

   mps = MultiphaseSystem(
       HC,
       phases=[liquid, gas],
       gamma={(0, 1): 0.072},  # surface tension coefficient [N/m]
   )

Key methods:

- ``assign_phases(criterion_fn)`` -- assign ``v.phase`` based on a spatial
  criterion (e.g., distance to droplet centre).
- ``identify_interface()`` -- mark ``v.is_interface`` on sharp-interface
  vertices.
- ``init_phase_fields()`` -- allocate per-phase arrays on all vertices.
- ``split_dual_volumes()`` -- split dual cells among phases at the interface.
- ``refresh()`` -- one-call refresh of interface identification and dual
  volume splitting.
- ``get_mu(v)`` -- return phase-specific viscosity for vertex ``v``.
- ``get_gamma(v)`` -- return surface tension coefficient for an interface
  vertex.

Per-vertex attributes
^^^^^^^^^^^^^^^^^^^^^

After ``init_phase_fields()``, every vertex carries:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``v.phase``
     - Integer phase ID (0 = outer, 1 = droplet by convention)
   * - ``v.is_interface``
     - ``True`` for sharp-interface vertices
   * - ``v.interface_phases``
     - ``frozenset`` of phase IDs at this vertex
   * - ``v.m_phase[k]``
     - Mass of phase *k* in this dual cell
   * - ``v.p_phase[k]``
     - Pressure of phase *k*
   * - ``v.rho_phase[k]``
     - Density of phase *k*
   * - ``v.dual_vol_phase[k]``
     - Dual cell sub-volume for phase *k*

For bulk vertices only ``v.*_phase[v.phase]`` is non-zero. For interface
vertices, multiple entries are non-zero.

Equation of State
-----------------

Each phase has its own EOS. The ``MultiphaseEOS`` dispatcher routes pressure
updates to per-phase EOS instances:

.. code-block:: python

   from ddgclib.eos import TaitMurnaghan, IdealGas, MultiphaseEOS

   meos = MultiphaseEOS(phases=[liquid, gas])

   # Update per-phase pressures on all vertices
   meos(HC, dim=2)

Available EOS models:

- :class:`~ddgclib.eos.TaitMurnaghan` -- weakly compressible fluid
  (water, oil). Parameters: ``rho0``, ``P0``, ``K`` (bulk modulus), ``n``.
- :class:`~ddgclib.eos.IdealGas` -- ideal gas. Parameters: ``rho0``,
  ``P0``, ``gamma`` (ratio of specific heats).

Multiphase stress operator
--------------------------

The :func:`~ddgclib.operators.multiphase_stress.multiphase_stress_force`
computes the total force on a dual cell including:

1. **Pressure flux** -- uses the vertex's own-phase pressure
   ``v.p_phase[v.phase]``.
2. **Viscous flux** -- uses phase-specific viscosity (no harmonic mean at
   the interface).
3. **Surface tension** -- acts only on interface vertices, computed from
   discrete mean curvature.

.. code-block:: python

   from functools import partial
   from ddgclib.operators.multiphase_stress import multiphase_dudt_i

   dudt_fn = partial(
       multiphase_dudt_i,
       dim=2,
       HC=HC,
       mps=mps,
       pressure_model=meos,
   )

Surface tension computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Surface tension is computed differently in 2D and 3D:

- **3D**: Cotangent-weight Heron curvature via
  :func:`~ddgclib.operators.surface_tension.surface_tension_force`.
  The mean curvature normal is integrated over the dual cell using the
  Heron-formula cotangent weights.

- **2D**: Integrated dual curvature from tangent vector differences via
  :func:`~ddgclib.operators.curvature_2d.surface_tension_force_2d`.
  Based on the fundamental theorem of calculus:
  :math:`\mathbf{F}_{st} = \gamma (\mathbf{t}_{next} - \mathbf{t}_{prev})`.

Both methods are automatically dispatched by
``multiphase_stress_force`` based on the ``dim`` parameter.

Initial conditions
------------------

Multiphase-specific ICs handle phase assignment, per-phase mass, and
pressure with Young-Laplace jumps:

.. code-block:: python

   from ddgclib.initial_conditions import (
       CompositeIC, ZeroVelocity,
       PhaseAssignment, MultiphaseMass, MultiphasePressure,
   )

   ic = CompositeIC(
       ZeroVelocity(dim=2),
       PhaseAssignment(
           criterion_fn=lambda v: 1 if dist(v) < R else 0,
           mps=mps,
       ),
       MultiphaseMass(mps=mps),
       MultiphasePressure(
           mps=mps,
           P_base=[0.0, 0.0],              # base pressure per phase
           young_laplace=True,              # add gamma/R jump
           R0=0.3,                          # droplet radius
           droplet_phase=1,
       ),
   )
   ic.apply(HC, bV)

Domain builders
---------------

Pre-built multiphase domains with phase labels already assigned:

.. code-block:: python

   from ddgclib.geometry.domains import droplet_in_box_2d, droplet_in_box_3d

   # 2D: circular droplet in rectangular box
   result = droplet_in_box_2d(R=0.3, L=1.0, refinement=3)

   # 3D: spherical droplet in cubic box
   result = droplet_in_box_3d(R=0.3, L=1.0, refinement=2)

   HC, bV = result.HC, result.bV
   # Vertices already have v.phase set (0 = outer gas, 1 = inner droplet)
   # result.boundary_groups includes 'walls' and 'interface'

Mass redistribution
-------------------

When the mesh is retriangulated (topology change), masses must be
redistributed to preserve the pressure field:

.. code-block:: python

   from ddgclib.operators.mass_redistribution import (
       snapshot_pressure_multiphase,
       redistribute_mass_multiphase,
   )

   # Before retriangulation: capture per-phase pressures
   pressure_snapshot = snapshot_pressure_multiphase(HC)

   # ... retriangulation happens ...

   # After: restore masses from pressure field
   redistribute_mass_multiphase(HC, pressure_snapshot, mps, dim=2)

The dynamic integrators handle this automatically when
``_retopologize_multiphase`` is used.

Multiphase retopologization
---------------------------

The integrator-level function ``_retopologize_multiphase`` handles
retriangulation while preserving multiphase state:

- Preserves phase labels across topology changes
- Re-identifies the interface
- Optionally redistributes mass per-phase
- Supports both Delaunay and adaptive remeshing modes

Visualization
-------------

Record and animate multiphase simulations with phase/interface overlays:

.. code-block:: python

   from ddgclib.visualization import StateHistory, dynamic_plot_fluid
   from ddgclib.visualization.multiphase import (
       record_multiphase_frame,
       dynamic_plot_multiphase,
   )

   history = StateHistory(
       fields=["u", "p", "phase", "is_interface"],
       record_every=10,
   )

   # During simulation loop:
   record_multiphase_frame(history, HC)

   # After simulation:
   dynamic_plot_multiphase(
       history, HC,
       save_path="animation.mp4",
   )

Complete example: oscillating droplet
--------------------------------------

This example sets up a 2D oscillating droplet with two phases:

.. code-block:: python

   from functools import partial
   import numpy as np

   from ddgclib.geometry.domains import droplet_in_box_2d
   from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
   from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
   from ddgclib.initial_conditions import (
       CompositeIC, ZeroVelocity, MultiphaseMass, MultiphasePressure,
   )
   from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
   from ddgclib.operators.multiphase_stress import multiphase_dudt_i
   from ddgclib.dynamic_integrators import symplectic_euler

   # 1. Domain
   result = droplet_in_box_2d(R=0.3, L=1.0, refinement=3)
   HC, bV = result.HC, result.bV

   # 2. Phase properties
   liquid = PhaseProperties(
       eos=TaitMurnaghan(rho0=1000.0, P0=0.0, K=50.0, n=7.15),
       mu=0.01, rho0=1000.0, name="liquid",
   )
   gas = PhaseProperties(
       eos=TaitMurnaghan(rho0=1.0, P0=0.0, K=1.0, n=1.4),
       mu=0.001, rho0=1.0, name="gas",
   )
   mps = MultiphaseSystem(HC, phases=[liquid, gas], gamma={(0, 1): 0.072})
   meos = MultiphaseEOS(phases=[liquid, gas])

   # 3. Initial conditions
   ic = CompositeIC(
       ZeroVelocity(dim=2),
       MultiphaseMass(mps=mps),
       MultiphasePressure(mps=mps, P_base=[0.0, 0.0],
                          young_laplace=True, R0=0.3, droplet_phase=1),
   )
   ic.apply(HC, bV)

   # 4. Boundary conditions
   bc_set = BoundaryConditionSet()
   bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups["walls"])

   # 5. Time integration
   dudt_fn = partial(multiphase_dudt_i, dim=2, HC=HC, mps=mps,
                     pressure_model=meos)
   symplectic_euler(HC, bV, dudt_fn, dt=1e-5, n_steps=5000,
                    dim=2, bc_set=bc_set)

See ``cases_dynamic/oscillating_droplet/`` and ``cases_dynamic/dam_break/``
for production-quality case studies with animations and analytical validation.

Case studies
------------

Oscillating droplet
^^^^^^^^^^^^^^^^^^^

Located in ``cases_dynamic/oscillating_droplet/``. Available in 2D and 3D.

Validates against the Rayleigh-Lamb analytical solution:

- **Rayleigh frequency** (mode l=2): :math:`\omega^2 = 8\gamma / (\rho R_0^3)`
- **Lamb damping rate**: :math:`\beta = 5\mu / (\rho R_0^2)`
- **Young-Laplace pressure jump**: :math:`\Delta P = \gamma/R` (2D),
  :math:`2\gamma/R` (3D)

Dam break
^^^^^^^^^

Located in ``cases_dynamic/dam_break/``. Available in 2D and 3D, with
single-phase (no air) and two-phase variants.

Sets up a rectangular tank with a liquid column (lower-left) and air (rest).
Validates front position and pressure evolution.
