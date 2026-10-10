"""Placeholder electrolysis reaction: linear gas mass injection.

Adds mass to the gas phase at a constant rate ``dm/dt`` through the
library source operator :func:`ddgclib.operators.mass_source.add_phase_mass`
(the per-vertex loop that lived here until laneG, 2026-10-06, moved there
so that the conservative remap's level anchor follows the injected mass).
Mass is distributed across phase-1 (gas) dual-volume weightings, which for
a fully-enclosed bubble is equivalent to a uniform gas source.

No charge transport, species diffusion, or Nernst/Butler-Volmer
coupling here -- this is a hook for the eventual reaction-diffusion
pipeline.
"""
from __future__ import annotations

from ddgclib.operators.mass_source import add_phase_mass


def inject_gas_mass(HC, mps, dm_dt: float, dt: float,
                    gas_phase: int = 1) -> float:
    """Add ``dm_dt * dt`` kg of gas mass to the gas-phase sub-volumes.

    Returns the total mass actually added (0.0 if no gas phase
    present).
    """
    return add_phase_mass(HC, mps, gas_phase, dm_dt * dt)
