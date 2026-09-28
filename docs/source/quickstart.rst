Quick Start
===========

This page walks through the essential concepts with minimal examples.

Creating a mesh
---------------

ddgclib uses ``hyperct.Complex`` as its simplicial complex backend.
Domain builders provide one-liner mesh construction:

.. code-block:: python

   from ddgclib.geometry.domains import rectangle

   result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)
   HC = result.HC        # simplicial Complex
   bV = result.bV        # set of boundary vertices

   # Named boundary groups for BCs
   print(result.boundary_groups)
   # {'walls', 'inlet', 'outlet', 'bottom_wall', 'top_wall'}

Computing dual meshes
---------------------

Discrete operators require a dual mesh. Compute it with
``hyperct.ddg.compute_vd``:

.. code-block:: python

   from hyperct.ddg import compute_vd

   # Tag boundaries (domain builders do this automatically)
   for v in HC.V:
       v.boundary = v in bV

   compute_vd(HC, method="barycentric")   # or "circumcentric"

Single-phase Poiseuille flow
----------------------------

A minimal 2D channel flow using the Cauchy stress tensor pipeline:

.. code-block:: python

   from functools import partial

   from ddgclib.geometry.domains import rectangle
   from ddgclib.initial_conditions import (
       CompositeIC, ZeroVelocity, LinearPressureGradient, UniformMass,
   )
   from ddgclib._boundary_conditions import (
       BoundaryConditionSet, NoSlipWallBC,
   )
   from ddgclib.operators.stress import dudt_i
   from ddgclib.dynamic_integrators import symplectic_euler
   from hyperct.ddg import compute_vd

   # 1. Domain
   result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)
   HC, bV = result.HC, result.bV
   compute_vd(HC, method="barycentric")

   # 2. Initial conditions
   ic = CompositeIC(
       ZeroVelocity(dim=2),
       LinearPressureGradient(dPdx=-1.0, axis=0, dim=2),
       UniformMass(total_volume=result.metadata["volume"], rho=1000.0),
   )
   ic.apply(HC, bV)

   # 3. Boundary conditions
   bc_set = BoundaryConditionSet()
   bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups["walls"])

   # 4. Time integration
   dudt_fn = partial(dudt_i, dim=2, mu=0.001, HC=HC)
   symplectic_euler(HC, bV, dudt_fn, dt=1e-4, n_steps=1000,
                    dim=2, bc_set=bc_set)

Multiphase droplet (preview)
----------------------------

A two-phase oscillating droplet -- see the :doc:`guides/multiphase` guide
for the full walkthrough:

.. code-block:: python

   from ddgclib.geometry.domains import droplet_in_box_2d
   from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
   from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
   from ddgclib.operators.multiphase_stress import multiphase_dudt_i

   # Build domain with phase labels already assigned
   result = droplet_in_box_2d(R=0.3, L=1.0, refinement=3)
   HC, bV = result.HC, result.bV

   # Define phases and surface tension
   liquid = PhaseProperties(eos=TaitMurnaghan(...), mu=0.01, rho0=1000.0)
   gas    = PhaseProperties(eos=TaitMurnaghan(...), mu=0.001, rho0=1.0)
   mps = MultiphaseSystem(HC, phases=[liquid, gas],
                          gamma={(0, 1): 0.072})

See :doc:`guides/multiphase` for the complete pipeline.

Running tests
-------------

.. code-block:: bash

   # Fast tests only (~18s)
   pytest ddgclib/tests/ -v -m "not slow"

   # All tests including slow 3D case studies
   pytest ddgclib/tests/ -v

   # Single module
   pytest ddgclib/tests/test_stress.py -v -m "not slow"

Next steps
----------

- :doc:`guides/simulation_workflow` -- full single-phase Lagrangian pipeline
- :doc:`guides/multiphase` -- multiphase flows with surface tension
- :doc:`guides/dem` -- discrete element method for particle dynamics
- :doc:`api/operators` -- operator API reference
