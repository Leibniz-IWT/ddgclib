Discrete Element Method (DEM)
=============================

ddgclib includes a self-contained DEM submodule for spherical particle
dynamics. The DEM particles are separate from the fluid mesh (``HC``) --
they have their own data structure (``ParticleSystem``), contact detection,
force models, and time integration.

Quick start
-----------

.. code-block:: python

   from ddgclib.dem import (
       Particle, ParticleSystem, ContactDetector,
       HertzContact, dem_step,
   )

   # Create particles
   ps = ParticleSystem(dim=3)
   ps.add(Particle.sphere(x=[0.0, 0.0, 0.5], radius=0.1, rho_s=2500.0, dim=3))
   ps.add(Particle.sphere(x=[0.0, 0.0, 0.3], radius=0.1, rho_s=2500.0, dim=3))

   # Contact detection and force model
   detector = ContactDetector(ps)
   model = HertzContact(E=1e7, nu=0.3, mu_friction=0.3, e_rest=0.8)

   # Time integration
   dem_step(ps, detector, model, dt=1e-5, dim=3, n_sub=10)

Particles
---------

Create particles with automatic mass and moment of inertia:

.. code-block:: python

   from ddgclib.dem import Particle

   # Convenience constructor
   p = Particle.sphere(x=[1.0, 2.0, 3.0], radius=0.05, rho_s=2500.0, dim=3)

   # Access properties
   p.position     # ndarray
   p.velocity     # ndarray
   p.radius       # float
   p.mass         # auto-computed from rho_s and volume

ParticleSystem
--------------

Container for managing collections of particles:

.. code-block:: python

   from ddgclib.dem import ParticleSystem

   ps = ParticleSystem(dim=3)
   ps.add(p1)
   ps.add(p2)

   ps.positions()    # (N, dim) array
   ps.velocities()   # (N, dim) array
   ps.radii()        # (N,) array

Import from NumPy arrays:

.. code-block:: python

   from ddgclib.dem import import_particle_cloud

   positions = np.random.rand(100, 3)
   radii = np.full(100, 0.01)
   ps = import_particle_cloud(positions, radii, rho_s=2500.0, dim=3)

Contact detection
-----------------

Spatial-hash broad phase with sphere-sphere narrow phase:

.. code-block:: python

   from ddgclib.dem import ContactDetector

   detector = ContactDetector(ps)
   contacts = detector.detect()  # list of (i, j, overlap, normal) tuples

Force models
------------

Pluggable contact force models:

.. code-block:: python

   from ddgclib.dem import HertzContact, LinearSpringDashpot

   # Hertz-Mindlin contact
   hertz = HertzContact(E=1e7, nu=0.3, mu_friction=0.3, e_rest=0.8)

   # Linear spring-dashpot (simpler)
   lsd = LinearSpringDashpot(k_n=1e5, k_t=1e4, gamma_n=100.0, mu=0.3)

Time integration
----------------

.. code-block:: python

   from ddgclib.dem import dem_step

   # Single macro step with n_sub sub-steps
   dem_step(ps, detector, model, dt=1e-4, dim=3, n_sub=10)

Sintered bonds
--------------

Model sintered inter-particle bonds with Frenkel neck growth:

.. code-block:: python

   from ddgclib.dem import SinterBond, BondManager

   bm = BondManager()
   # Bonds form between particles in contact
   # SinterBond tracks neck radius and bond forces

Capillary bridges
-----------------

Liquid bridge forces between particles (Lian et al. 1993):

.. code-block:: python

   from ddgclib.dem import LiquidBridge, LiquidBridgeManager

   lbm = LiquidBridgeManager(
       gamma=0.072,       # surface tension [N/m]
       theta=0.0,         # contact angle [rad]
       V_bridge=1e-12,    # bridge volume [m^3]
   )

Fluid-particle coupling
------------------------

Two-way drag coupling between the fluid mesh and DEM particles:

.. code-block:: python

   from ddgclib.dem import FluidParticleCoupler

   coupler = FluidParticleCoupler(HC, ps, dim=3, mu=0.001)
   # Computes drag forces on particles from fluid velocity field
   # and reaction forces on fluid vertices

I/O
---

Save and load particle state:

.. code-block:: python

   from ddgclib.dem import save_particles, load_particles

   save_particles(ps, "particles.json")
   ps_loaded = load_particles("particles.json")

Visualization
-------------

.. code-block:: python

   from ddgclib.dem import plot_particles

   plot_particles(ps)

Running DEM tests
-----------------

.. code-block:: bash

   # All DEM tests (fast)
   pytest ddgclib/tests/test_dem_*.py -v -m "not slow"

   # Including slow validation tests
   pytest ddgclib/tests/test_dem_*.py -v

Case studies
------------

- ``cases_dynamic/liquid_bridge_dem/`` -- DEM-only liquid bridge dynamics
- ``cases_dynamic/liquid_bridge_cfd_dem/`` -- two-way coupled CFD-DEM
