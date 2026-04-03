"""Boundary conditions for the oscillating droplet case.

The standard setup uses no-slip walls on the outer box boundary.
This module provides convenience wrappers and alternative BC
configurations for experimentation.
"""
from __future__ import annotations

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    NoSlipWallBC,
    NeumannBC,
)


def make_noslip_bc(dim: int, wall_vertices: set) -> BoundaryConditionSet:
    """Standard no-slip wall BC on outer box boundary."""
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), wall_vertices)
    return bc_set


def make_open_bc(wall_vertices: set) -> BoundaryConditionSet:
    """Open (zero-gradient) BC on outer box — less reflective."""
    bc_set = BoundaryConditionSet()
    bc_set.add(NeumannBC(field_name='u', flux_value=0.0), wall_vertices)
    bc_set.add(NeumannBC(field_name='p', flux_value=0.0), wall_vertices)
    return bc_set
