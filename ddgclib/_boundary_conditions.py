"""
Boundary conditions for dynamic simulations on simplicial complexes.

Provides an abstract BoundaryCondition base class, concrete implementations
(no-slip walls, Dirichlet, Neumann, outlet, periodic inlet), boundary
identification helpers, and a BoundaryConditionSet container.

Usage
-----
    from ddgclib._boundary_conditions import (
        identify_boundary_vertices, BoundaryConditionSet,
        NoSlipWallBC, DirichletVelocityBC, OutletDeleteBC,
    )

    bV_wall = identify_boundary_vertices(HC, lambda v: getattr(v, 'side_boundary', False))
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=3), bV_wall)
    bc_set.apply_all(HC, bV, dt=1e-4)
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Boundary identification helpers
# ---------------------------------------------------------------------------

def identify_boundary_vertices(HC, criterion_fn: Callable) -> set:
    """Return set of vertices matching criterion_fn(v) -> bool.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    criterion_fn : callable
        Function taking a vertex object and returning True/False.

    Examples
    --------
    >>> bV = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-10)
    >>> bV_wall = identify_boundary_vertices(HC,
    ...     lambda v: abs(np.linalg.norm(v.x_a[:2]) - R) < tol)
    """
    return {v for v in HC.V if criterion_fn(v)}


def identify_cube_boundaries(HC, lb: float, ub: float, dim: Optional[int] = None) -> set:
    """Identify all vertices on the faces of an axis-aligned cube [lb, ub]^d.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    lb : float
        Lower bound of cube domain.
    ub : float
        Upper bound of cube domain.
    dim : int or None
        Number of coordinate axes to check. If None, uses len(v.x_a).

    Returns
    -------
    set
        Set of boundary vertex objects.
    """
    tol = 1e-14
    bV = set()
    for v in HC.V:
        d = dim if dim is not None else len(v.x_a)
        for i in range(d):
            if v.x_a[i] <= lb + tol or v.x_a[i] >= ub - tol:
                bV.add(v)
                break
    return bV


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BoundaryCondition(ABC):
    """Base class for boundary conditions on a moving mesh.

    Subclasses must implement ``apply()``. The optional ``target_vertices``
    parameter allows the BoundaryConditionSet to specify which vertices
    this BC applies to.
    """

    def __init__(self, axis: int = 2):
        self.axis = axis

    @abstractmethod
    def apply(self, mesh, dt, target_vertices=None):
        """Apply the BC to the mesh for this time step.

        Parameters
        ----------
        mesh : Complex
            The simplicial complex.
        dt : float
            Time step size.
        target_vertices : set or None
            If provided, only apply to these vertices.
            If None, behavior is subclass-specific.
        """


# ---------------------------------------------------------------------------
# BoundaryConditionSet container
# ---------------------------------------------------------------------------

class BoundaryConditionSet:
    """Container managing multiple BCs for a simulation.

    BCs are applied in insertion order. Each BC can be associated
    with a specific subset of boundary vertices.

    Usage
    -----
        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=3), bV_wall)
        bc_set.add(OutletDeleteBC(outlet_pos=10.0), None)  # applies to all
        bc_set.apply_all(mesh, bV, dt)
    """

    def __init__(self):
        self._bcs: list[tuple[BoundaryCondition, Optional[set]]] = []

    def add(self, bc: BoundaryCondition, vertices: Optional[set] = None) -> 'BoundaryConditionSet':
        """Register a BC. If vertices is None, BC applies to all of bV.

        Returns self for method chaining.
        """
        self._bcs.append((bc, vertices))
        return self

    def apply_all(self, mesh, bV: set, dt: float) -> dict:
        """Apply all BCs in order.

        Parameters
        ----------
        mesh : Complex
            The simplicial complex.
        bV : set
            Full boundary vertex set (used as default when bc has no specific vertices).
        dt : float
            Time step size.

        Returns
        -------
        dict
            Diagnostics keyed by BC name.
        """
        diagnostics = {}
        for i, (bc, verts) in enumerate(self._bcs):
            target = verts if verts is not None else bV
            result = bc.apply(mesh, dt, target_vertices=target)
            diagnostics[f"bc_{i}_{type(bc).__name__}"] = result
        return diagnostics


# ---------------------------------------------------------------------------
# Dirichlet Boundary Conditions (Fixed Value)
# ---------------------------------------------------------------------------

class NoSlipWallBC(BoundaryCondition):
    """Zero velocity on wall vertices (no-slip condition).

    Parameters
    ----------
    dim : int
        Spatial dimension (velocity vector length).
    """

    def __init__(self, dim: int = 3):
        super().__init__()
        self.dim = dim

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        count = 0
        for v in verts:
            v.u = np.zeros(self.dim)
            count += 1
        return count


class DirichletVelocityBC(BoundaryCondition):
    """Fixed velocity on boundary vertices.

    Parameters
    ----------
    value : ndarray or callable
        Constant velocity vector, or callable fn(v) -> ndarray.
    dim : int
        Spatial dimension.
    """

    def __init__(self, value: Union[np.ndarray, Callable], dim: int = 3):
        super().__init__()
        self.value = value
        self.dim = dim

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        count = 0
        for v in verts:
            if callable(self.value):
                v.u = np.asarray(self.value(v), dtype=float)
            else:
                v.u = np.asarray(self.value, dtype=float).copy()
            count += 1
        return count


class DirichletPressureBC(BoundaryCondition):
    """Fixed pressure on boundary vertices.

    Parameters
    ----------
    value : float or callable
        Constant pressure, or callable fn(v) -> float.
    """

    def __init__(self, value: Union[float, Callable]):
        super().__init__()
        self.value = value

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        count = 0
        for v in verts:
            if callable(self.value):
                v.P = float(self.value(v))
            else:
                v.P = float(self.value)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Neumann Boundary Condition (Fixed Flux/Derivative)
# ---------------------------------------------------------------------------

class NeumannBC(BoundaryCondition):
    """Zero-gradient (or fixed gradient) at boundary.

    For zero Neumann (flux_value=0), copies the field value from the
    nearest interior neighbor. For nonzero flux, adjusts by
    ``flux_value * distance_to_neighbor``.

    Parameters
    ----------
    field_name : str
        Vertex attribute to enforce ('u', 'P', etc.).
    flux_value : float
        Gradient value at boundary (0 for zero-gradient).
    """

    def __init__(self, field_name: str = 'u', flux_value: float = 0.0):
        super().__init__()
        self.field_name = field_name
        self.flux_value = flux_value

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else set()
        count = 0
        for v in verts:
            # Find nearest interior neighbor (not in target set)
            interior_nbs = [nb for nb in v.nn if nb not in verts]
            if not interior_nbs:
                continue
            nb = min(interior_nbs,
                     key=lambda nb: np.linalg.norm(v.x_a - nb.x_a))
            dist = np.linalg.norm(v.x_a - nb.x_a)
            interior_val = getattr(nb, self.field_name)
            if isinstance(interior_val, np.ndarray):
                normal_dir = (v.x_a - nb.x_a) / max(dist, 1e-30)
                setattr(v, self.field_name,
                        interior_val.copy() + self.flux_value * dist * normal_dir[:len(interior_val)])
            else:
                setattr(v, self.field_name,
                        interior_val + self.flux_value * dist)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Open (Transparent/Absorbing) Boundary Condition
# ---------------------------------------------------------------------------

class OutletDeleteBC(BoundaryCondition):
    """Simple open outlet: delete vertices that leave the domain.

    Parameters
    ----------
    outlet_pos : float
        Vertices with x[axis] >= outlet_pos are deleted.
    axis : int
        Coordinate axis for position check.
    """

    def __init__(self, outlet_pos, axis=2):
        super().__init__(axis)
        self.outlet_pos = outlet_pos

    def apply(self, mesh, dt, target_vertices=None):
        to_delete = [v for v in mesh.V if v.x_a[self.axis] >= self.outlet_pos]
        for v in to_delete:
            mesh.V.remove(v)
        return len(to_delete)


# ---------------------------------------------------------------------------
# Periodic Boundary Condition
# ---------------------------------------------------------------------------

class PeriodicInletBC(BoundaryCondition):
    """Periodic inlet: continuously injects copies of a unit mesh from upstream.

    Ghost mesh lives in a periodic box just before the inlet.

    Parameters
    ----------
    unit_mesh : Complex
        Source geometry (cylinder, rectangle, ...).
    velocity : float
        Advection velocity along axis.
    axis : int
        Flow axis.
    inlet_pos : float
        Inlet position along axis.
    cdist : float
        Vertex merging tolerance.
    """

    def __init__(self, unit_mesh, velocity, axis=2, inlet_pos=0.0, cdist=1e-10):
        super().__init__(axis)
        self.unit_mesh = unit_mesh
        self.velocity = velocity
        self.inlet_pos = inlet_pos
        self.cdist = cdist
        self.ghost = self._clone_unit(unit_mesh)
        self._reset_ghost()

    def _clone_unit(self, unit_mesh):
        """Safe clone without deepcopy issues."""
        return unit_mesh  # placeholder -- replace with proper clone if needed

    def _reset_ghost(self):
        """Shift ghost so its leading face is exactly 1 unit upstream."""
        shift = self.inlet_pos - 1.0
        for v in list(self.ghost.V):
            pos = v.x_a.copy()
            pos[self.axis] += shift
            self.ghost.V.move(v, tuple(pos))

    def apply(self, mesh, dt, target_vertices=None):
        dx = self.velocity * dt

        # Move ghost forward
        for v in list(self.ghost.V):
            pos = v.x_a.copy()
            pos[self.axis] += dx
            self.ghost.V.move(v, tuple(pos))

        # Inject any vertices that crossed the inlet
        entered = []
        for gv in list(self.ghost.V):
            if gv.x_a[self.axis] >= self.inlet_pos:
                new_v = mesh.V[tuple(gv.x_a)]
                entered.append((gv, new_v))

        # Copy connections for entered vertices
        for gv, new_v in entered:
            for gnb in gv.nn:
                if gnb.x_a[self.axis] >= self.inlet_pos:
                    new_nb = mesh.V[tuple(gnb.x_a)]
                    new_v.connect(new_nb)

        # Periodic reset
        if all(gv.x_a[self.axis] >= self.inlet_pos for gv in self.ghost.V):
            self._reset_ghost()

        mesh.V.merge_all(cdist=self.cdist)


# ---------------------------------------------------------------------------
# Generic time stepper / advancer
# ---------------------------------------------------------------------------

class MeshAdvancer:
    """Coordinate inlet/outlet BCs with bulk mesh advection.

    Parameters
    ----------
    mesh : Complex
        The simplicial complex.
    inlet_bc : BoundaryCondition
        Inlet boundary condition.
    outlet_bc : BoundaryCondition
        Outlet boundary condition.
    velocity : float
        Bulk advection velocity.
    """

    def __init__(self, mesh, inlet_bc, outlet_bc, velocity):
        self.mesh = mesh
        self.inlet = inlet_bc
        self.outlet = outlet_bc
        self.velocity = velocity
        self.time = 0.0
        self.axis = inlet_bc.axis

    def step(self, dt):
        dx = self.velocity * dt
        self.time += dt

        # 1. Advect main mesh
        for v in list(self.mesh.V):
            pos = v.x_a.copy()
            pos[self.axis] += dx
            self.mesh.V.move(v, tuple(pos))

        # 2. Apply boundary conditions
        outflow_count = self.outlet.apply(self.mesh, dt)
        self.inlet.apply(self.mesh, dt)

        return outflow_count
