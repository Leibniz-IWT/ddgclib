"""Discrete Element Method (DEM) submodule for ddgclib.

Provides spherical particle simulation with contact mechanics,
fluid-particle coupling, sintered bond aggregation, and capillary
liquid bridge models.

Usage
-----
::

    from ddgclib.dem import Particle, ParticleSystem, HertzContact
    from ddgclib.dem import dem_step, ContactDetector
    from ddgclib.dem import SinterBond, BondManager
    from ddgclib.dem import LiquidBridge, LiquidBridgeManager
    from ddgclib.dem import FluidParticleCoupler
    from ddgclib.dem import save_particles, load_particles, import_particle_cloud
    from ddgclib.dem import plot_particles, plot_bridges, plot_bonds
"""

# Phase 1: Core DEM
from ddgclib.dem._particle import Particle, ParticleSystem
from ddgclib.dem._contact import ContactDetector, Contact
from ddgclib.dem._force_models import (
    ContactForceModel,
    ContactForceResult,
    HertzContact,
    LinearSpringDashpot,
    contact_force_registry,
)
from ddgclib.dem._integrators import (
    dem_velocity_verlet,
    dem_symplectic_euler,
    dem_step,
)

# Phase 2: Bridges and Bonds
from ddgclib.dem._bonds import SinterBond, BondManager
from ddgclib.dem._liquid_bridge import LiquidBridge, LiquidBridgeManager

# Phase 3: Coupling, I/O, Visualization
from ddgclib.dem._coupling import FluidParticleCoupler
from ddgclib.dem._io import save_particles, load_particles, import_particle_cloud
from ddgclib.dem._visualization import (
    plot_particles,
    plot_bridges,
    plot_bonds,
)

__all__ = [
    # Core
    "Particle",
    "ParticleSystem",
    "ContactDetector",
    "Contact",
    "ContactForceModel",
    "ContactForceResult",
    "HertzContact",
    "LinearSpringDashpot",
    "contact_force_registry",
    "dem_velocity_verlet",
    "dem_symplectic_euler",
    "dem_step",
    # Bonds
    "SinterBond",
    "BondManager",
    # Liquid bridges
    "LiquidBridge",
    "LiquidBridgeManager",
    # Coupling
    "FluidParticleCoupler",
    # I/O
    "save_particles",
    "load_particles",
    "import_particle_cloud",
    # Visualization
    "plot_particles",
    "plot_bridges",
    "plot_bonds",
]
