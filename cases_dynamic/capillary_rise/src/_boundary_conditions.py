"""Boundary condition helpers for capillary rise."""
from __future__ import annotations

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    NoSlipWallBC,
    DirichletPressureBC,
)


def make_capillary_bc(
    dim: int,
    wall_verts: set,
    bottom_verts: set,
    free_surface_verts: set,
    P_cap: float,
) -> BoundaryConditionSet:
    """Create boundary condition set for capillary rise.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    wall_verts : set
        Tube wall vertices (no-slip).
    bottom_verts : set
        Bottom face vertices (no-slip, reservoir).
    free_surface_verts : set
        Top face vertices (free surface with capillary pressure).
    P_cap : float
        Capillary pressure magnitude [Pa]. Applied as v.p = -P_cap
        (suction) on free surface vertices.

    Returns
    -------
    BoundaryConditionSet
    """
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), wall_verts)
    bc_set.add(NoSlipWallBC(dim=dim), bottom_verts)
    bc_set.add(DirichletPressureBC(value=-P_cap), free_surface_verts)
    return bc_set
