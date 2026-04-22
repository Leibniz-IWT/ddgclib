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


# Boundary identification helpers

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


# Abstract base class

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


# BoundaryConditionSet container

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


# Dirichlet Boundary Conditions (Fixed Value)

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


class MovingWallBC(BoundaryCondition):
    """No-slip wall moving with a prescribed tangential velocity.

    Imposes ``v.u = wall_velocity`` on all target vertices each step.
    Unlike :class:`NoSlipWallBC` (zero velocity), this sets a specified
    constant velocity vector, suitable for shear cells / Couette flow
    where the plate slides along its own surface at a fixed speed.

    The wall vertices themselves do NOT translate through space — they
    are frozen (members of ``bV``) and only impose a velocity boundary
    condition for the fluid viscous stress.  This is the standard
    Navier–Stokes setup: the plate *velocity* is a boundary condition,
    not a mesh motion.

    Parameters
    ----------
    wall_velocity : ndarray or callable
        Constant velocity vector (length ``dim``), or ``fn(v) -> ndarray``.
        Use a callable when top/bottom plates need different velocities
        applied through a single BC.
    dim : int
        Spatial dimension (velocity vector length).
    """

    def __init__(self, wall_velocity, dim: int = 3):
        super().__init__()
        self.wall_velocity = wall_velocity
        self.dim = dim

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        count = 0
        if callable(self.wall_velocity):
            for v in verts:
                v.u = np.asarray(self.wall_velocity(v), dtype=float).copy()
                count += 1
        else:
            u_const = np.asarray(self.wall_velocity, dtype=float)
            for v in verts:
                v.u = u_const.copy()
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
                v.p = float(self.value(v))
            else:
                v.p = float(self.value)
            count += 1
        return count


# Neumann Boundary Condition (Fixed Flux/Derivative)

class NeumannBC(BoundaryCondition):
    """Zero-gradient (or fixed gradient) at boundary.

    For zero Neumann (flux_value=0), copies the field value from the
    nearest interior neighbor. For nonzero flux, adjusts by
    ``flux_value * distance_to_neighbor``.

    Parameters
    ----------
    field_name : str
        Vertex attribute to enforce ('u', 'p', etc.).
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


# Open (Transparent/Absorbing) Boundary Condition

class OutletDeleteBC(BoundaryCondition):
    """Open outlet: delete vertices that leave the domain and prevent backflow.

    Truncated dual cells at the outlet boundary produce imbalanced stress
    forces that push vertices backward.  The *backflow_clamp* parameter
    defines a zone upstream of the deletion threshold where the flow-
    direction velocity component is clamped to be non-negative.

    Parameters
    ----------
    outlet_pos : float
        Vertices with x[axis] >= outlet_pos are deleted.
    axis : int
        Coordinate axis for position check.
    bV : set or None
        Mutable boundary vertex set. If provided, deleted vertices are
        also removed from this set to avoid stale references.
    backflow_clamp : float or None
        Width of the no-backflow zone upstream of *outlet_pos*.
        Vertices at ``x[axis] >= outlet_pos - backflow_clamp`` have
        their velocity along *axis* clamped to ``max(u[axis], 0)``.
        Set to ``None`` to disable (default for backward compatibility).
    """

    def __init__(self, outlet_pos, axis=2, bV=None, backflow_clamp=None):
        super().__init__(axis)
        self.outlet_pos = outlet_pos
        self.bV = bV
        self.backflow_clamp = backflow_clamp

    def apply(self, mesh, dt, target_vertices=None):
        to_delete = [v for v in mesh.V if v.x_a[self.axis] >= self.outlet_pos]
        for v in to_delete:
            if self.bV is not None:
                self.bV.discard(v)
            mesh.V.remove(v)

        # Prevent backflow in the outlet buffer zone
        if self.backflow_clamp is not None:
            clamp_start = self.outlet_pos - self.backflow_clamp
            for v in mesh.V:
                if (v.x_a[self.axis] >= clamp_start
                        and v.u[self.axis] < 0.0):
                    v.u[self.axis] = 0.0

        return len(to_delete)


class OutletBufferedDeleteBC(BoundaryCondition):
    """Open outlet with a buffer ghost zone for complete dual cells.

    Maintains a buffer zone ``[outlet_pos, outlet_pos + buffer_width]``
    beyond the physical outlet.  When domain vertices cross *outlet_pos*
    they enter the buffer where:

    * Their velocity is frozen to the value at entry.
    * Their position advances at the frozen velocity each step (the
      integrator's stress-contaminated update is corrected).
    * They are deleted when they reach ``outlet_pos + buffer_width``.

    Because buffer vertices remain in the mesh, domain vertices near the
    outlet always have neighbours from Delaunay retriangulation.  Their
    dual cells are therefore complete and stress computation is balanced
    — eliminating the backflow caused by truncated duals.

    Parameters
    ----------
    outlet_pos : float
        Physical outlet position.  Vertices crossing this enter the
        buffer (not the physical domain).
    buffer_width : float
        Width of the ghost buffer zone beyond *outlet_pos*.
    axis : int
        Coordinate axis for position check (flow direction).
    bV : set or None
        Mutable boundary vertex set.  Deleted vertices are removed.
    """

    def __init__(self, outlet_pos, buffer_width, axis=0, bV=None):
        super().__init__(axis)
        self.outlet_pos = outlet_pos
        self.buffer_width = buffer_width
        self.bV = bV
        # id(vertex) → (vertex_ref, frozen_velocity, correct_position)
        # Keyed by id() because mesh.V.move() changes vertex hash.
        self._buffer: dict = {}

    @property
    def buffer_vertices(self):
        """Set of vertex objects currently in the buffer."""
        return {rec[0] for rec in self._buffer.values()}

    def apply(self, mesh, dt, target_vertices=None):
        ax = self.axis
        buffer_end = self.outlet_pos + self.buffer_width

        # 1. Delete vertices past buffer end
        to_delete = [v for v in list(mesh.V)
                     if v.x_a[ax] >= buffer_end]
        for v in to_delete:
            self._buffer.pop(id(v), None)
            if self.bV is not None:
                self.bV.discard(v)
            mesh.V.remove(v)

        # 2. Detect new buffer entries (just crossed outlet_pos)
        buf_ids = set(self._buffer.keys())
        for v in list(mesh.V):
            if id(v) not in buf_ids and v.x_a[ax] > self.outlet_pos:
                self._buffer[id(v)] = (v, v.u.copy(), v.x_a.copy())

        # 3. Correct position and reset velocity for all buffer vertices.
        new_buffer = {}
        for vid, (v, frozen_u, correct_pos) in self._buffer.items():
            try:
                _ = v.x_a
            except (KeyError, AttributeError):
                continue  # vertex removed by another BC

            # Advance correct position at frozen velocity
            new_correct = correct_pos.copy()
            new_correct[:len(frozen_u)] += frozen_u * dt

            # Move vertex to the correct position
            mesh.V.move(v, tuple(new_correct))

            # Reset velocity to frozen value
            v.u[:] = frozen_u

            new_buffer[id(v)] = (v, frozen_u, new_correct)

        self._buffer = new_buffer
        return len(to_delete)


# Periodic Boundary Condition

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
    fields : list of str or None
        Vertex attribute names to copy from ghost to newly injected vertices.
        Defaults to ``['u', 'p', 'm']``.
    period : float
        Domain length along the flow axis. The ghost resets this far upstream
        after all its vertices have crossed the inlet. Defaults to 1.0.
    """

    def __init__(self, unit_mesh, velocity, axis=2, inlet_pos=0.0, cdist=None,
                 fields=None, period=1.0):
        super().__init__(axis)
        self.unit_mesh = unit_mesh
        self.velocity = velocity
        self.inlet_pos = inlet_pos
        self.fields = fields if fields is not None else ['u', 'p', 'm']
        self.period = period
        self.cdist = cdist if cdist is not None else self._auto_merge_cdist(unit_mesh)
        self.ghost = self._clone_unit(unit_mesh)
        self._reset_ghost()

    @staticmethod
    def _auto_merge_cdist(unit_mesh):
        """Compute merge distance from minimum edge length of the unit mesh.

        Returns half the minimum edge length so that injected ghost vertices
        near existing wall/boundary vertices are merged, while distinct mesh
        vertices are preserved.
        """
        min_edge = float('inf')
        for v in unit_mesh.V:
            for nb in v.nn:
                d = np.linalg.norm(v.x_a - nb.x_a)
                if d < min_edge:
                    min_edge = d
        if min_edge == float('inf') or min_edge < 1e-30:
            return 1e-10
        return 0.5 * min_edge

    def _clone_unit(self, unit_mesh):
        """Clone the unit mesh including vertex fields and connectivity."""
        from hyperct import Complex

        # Determine dimension and domain from unit_mesh vertices
        all_coords = [v.x_a for v in unit_mesh.V]
        if not all_coords:
            return unit_mesh
        coords_arr = np.array(all_coords)
        dim = coords_arr.shape[1]
        lb = coords_arr.min(axis=0)
        ub = coords_arr.max(axis=0)
        domain = [(float(lb[i]), float(ub[i])) for i in range(dim)]

        clone = Complex(dim, domain=domain)

        # Add vertices and copy fields
        for v in unit_mesh.V:
            cv = clone.V[tuple(v.x_a)]
            for f in self.fields:
                val = getattr(v, f, None)
                if val is not None:
                    if isinstance(val, np.ndarray):
                        setattr(cv, f, val.copy())
                    else:
                        setattr(cv, f, val)

        # Copy connectivity
        seen = set()
        for v in unit_mesh.V:
            for nb in v.nn:
                edge = frozenset((tuple(v.x_a), tuple(nb.x_a)))
                if edge not in seen:
                    clone.V[tuple(v.x_a)].connect(clone.V[tuple(nb.x_a)])
                    seen.add(edge)

        return clone

    def _reset_ghost(self):
        """Shift ghost so its leading face is exactly one period upstream."""
        shift = self.inlet_pos - self.period
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

        # Inject vertices that just crossed the inlet (strict >
        # so that ghost vertices exactly at the inlet boundary
        # are not injected until they actually enter the domain)
        entered = []
        for gv in list(self.ghost.V):
            if gv.x_a[self.axis] > self.inlet_pos:
                new_v = mesh.V[tuple(gv.x_a)]
                entered.append((gv, new_v))
                # Copy field values from ghost vertex to new mesh vertex
                for f in self.fields:
                    val = getattr(gv, f, None)
                    if val is not None:
                        if isinstance(val, np.ndarray):
                            setattr(new_v, f, val.copy())
                        else:
                            setattr(new_v, f, val)

        # Copy connections for entered vertices.
        # Build a lookup of ghost->main vertex for all entered vertices so we
        # only connect to vertices that were actually injected (avoid auto-
        # creating bare vertices via mesh.V[key] when original vertices have
        # moved away in the Lagrangian frame).
        ghost_to_main = {id(gv): new_v for gv, new_v in entered}
        for gv, new_v in entered:
            for gnb in gv.nn:
                if id(gnb) in ghost_to_main:
                    new_v.connect(ghost_to_main[id(gnb)])

        # Remove injected vertices from ghost (each injected exactly once)
        for gv, _ in entered:
            self.ghost.V.remove(gv)

        # Periodic reset: re-clone when ghost is depleted
        if sum(1 for _ in self.ghost.V) == 0:
            self.ghost = self._clone_unit(self.unit_mesh)
            self._reset_ghost()

        mesh.V.merge_all(cdist=self.cdist)
        return len(entered)


class PositionalNoSlipWallBC(BoundaryCondition):
    """No-slip wall that identifies wall vertices by position each step.

    Unlike :class:`NoSlipWallBC` which acts on a fixed set of vertices,
    this BC dynamically scans all mesh vertices each step and applies
    no-slip (zero velocity) to any vertex matching the position criterion.
    New vertices injected by :class:`PeriodicInletBC` at wall positions
    are automatically detected.

    Parameters
    ----------
    criterion_fn : callable
        ``fn(v) -> bool``.  Returns True for wall vertices.
    dim : int
        Spatial dimension (velocity vector length).
    bV : set or None
        If provided, wall vertices are added to this mutable set and
        tagged ``v.boundary = True``.
    """

    def __init__(self, criterion_fn: Callable, dim: int = 2, bV: set = None):
        super().__init__()
        self.criterion_fn = criterion_fn
        self.dim = dim
        self.bV = bV

    def apply(self, mesh, dt, target_vertices=None):
        count = 0
        for v in mesh.V:
            if self.criterion_fn(v):
                v.u = np.zeros(self.dim)
                v.boundary = True
                if self.bV is not None:
                    self.bV.add(v)
                count += 1
        return count


# Generic time stepper / advancer

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


# ---------------------------------------------------------------------------
# Pressure-reservoir / absorbing boundary conditions
# ---------------------------------------------------------------------------

class PressureReservoirBC(BoundaryCondition):
    """Relax gas-phase vertices toward a target density with a mass sink.

    Replaces :class:`AtmosphericPressureBC`-style hard mass resets with
    a **smooth relaxation** toward the target density.  This avoids
    pressure wave reflections and shocks that a hard reset introduces
    when the interior fluid is still compressible.

    The per-step update is::

        delta_m = -tau_inv * dt * (m - m_target)

    where ``m_target = rho_target * dual_vol``.  The relaxation rate
    ``tau_inv`` should be comparable to the local acoustic timescale
    ``c_s / L_domain``.  With ``tau_inv = c_s / L``, pressure waves
    are damped over about one domain transit time.

    This is equivalent to a **first-order mass sink** (or source) that
    leaks mass out of the domain when the gas is compressed above
    ``rho_target`` and into the domain when it is expanded below.  The
    total mass of the domain is *not* conserved — that is physically
    correct for an open boundary venting to the atmosphere.

    Parameters
    ----------
    rho_target : float
        Target gas density [kg/m^3].
    tau_inv : float
        Relaxation rate [1/s].  Typical value: ``c_s / L_domain``.
    gas_phase : int
        Phase index of the gas (default 0).
    update_m_phase : bool
        If True (multiphase), also update ``v.m_phase[gas_phase]``.

    Notes
    -----
    The BC must be applied *after* the integrator step (same slot as
    ``AtmosphericPressureBC``) so that the EOS pressure is used to
    determine whether the gas is locally compressed.  The mass change
    takes effect on the next step's force evaluation.
    """

    def __init__(
        self,
        rho_target: float,
        tau_inv: float,
        gas_phase: int = 0,
        update_m_phase: bool = True,
    ):
        super().__init__()
        self.rho_target = float(rho_target)
        self.tau_inv = float(tau_inv)
        self.gas_phase = int(gas_phase)
        self.update_m_phase = bool(update_m_phase)

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        alpha = min(1.0, self.tau_inv * dt)  # implicit relaxation
        count = 0
        for v in verts:
            if getattr(v, 'phase', 0) != self.gas_phase:
                continue
            vol = getattr(v, 'dual_vol', 0.0)
            if vol < 1e-30:
                continue
            m_target = self.rho_target * vol
            # First-order relaxation: m <- m + alpha * (m_target - m)
            dm = alpha * (m_target - v.m)
            v.m = v.m + dm
            if self.update_m_phase:
                mp = getattr(v, 'm_phase', None)
                if mp is not None and self.gas_phase < len(mp):
                    mp[self.gas_phase] = v.m
            count += 1
        return count


class AbsorbingPressureBC(BoundaryCondition):
    """Characteristic absorbing BC for outgoing pressure waves.

    Approximates a non-reflecting (Sommerfeld) boundary for acoustic
    waves by damping the deviation of the local density from the
    target on a timescale set by the acoustic transit time across the
    buffer layer.

    Unlike :class:`PressureReservoirBC` (which uses a fixed target
    density), this BC treats the target as the **neighbour-averaged
    density** — effectively a local sponge zone.  The result is that
    pressure disturbances propagating outward are smoothly absorbed
    rather than reflected off the closed wall.

    The per-vertex update is::

        rho_nb_avg = mean(nb.m / nb.dual_vol) for neighbours
        m_new = v.m + alpha * (rho_nb_avg * vol - v.m)

    with ``alpha = min(1, tau_inv * dt)``.

    Parameters
    ----------
    tau_inv : float
        Relaxation rate [1/s].  Typical value: ``c_s / buffer_width``.
    phase : int or None
        If not None, only apply to vertices of this phase.
    update_m_phase : bool
        If True (multiphase), also update ``v.m_phase[phase]``.
    """

    def __init__(
        self,
        tau_inv: float,
        phase: int | None = None,
        update_m_phase: bool = True,
    ):
        super().__init__()
        self.tau_inv = float(tau_inv)
        self.phase = phase
        self.update_m_phase = bool(update_m_phase)

    def apply(self, mesh, dt, target_vertices=None):
        verts = target_vertices if target_vertices is not None else mesh.V
        alpha = min(1.0, self.tau_inv * dt)
        count = 0
        for v in verts:
            if self.phase is not None and getattr(v, 'phase', 0) != self.phase:
                continue
            vol = getattr(v, 'dual_vol', 0.0)
            if vol < 1e-30:
                continue

            # Compute target density from same-phase interior neighbours
            rho_sum, n = 0.0, 0
            for nb in v.nn:
                if self.phase is not None \
                        and getattr(nb, 'phase', 0) != self.phase:
                    continue
                nb_vol = getattr(nb, 'dual_vol', 0.0)
                if nb_vol < 1e-30:
                    continue
                rho_sum += nb.m / nb_vol
                n += 1
            if n == 0:
                continue
            rho_target = rho_sum / n

            m_target = rho_target * vol
            dm = alpha * (m_target - v.m)
            v.m = v.m + dm
            if self.update_m_phase:
                mp = getattr(v, 'm_phase', None)
                phase = self.phase if self.phase is not None else 0
                if mp is not None and phase < len(mp):
                    mp[phase] = v.m
            count += 1
        return count


class ExpandingDomainBC(BoundaryCondition):
    """Allow the outer-phase dual cells near walls to absorb compression
    by adjusting the target density based on the mean interior density.

    This is an alternative to a fixed atmospheric pressure: the "target"
    outer density drifts slowly toward the domain-average outer-phase
    density, so the wall zone acts as a **floating reservoir** rather
    than a stiff reference.  In effect, if the droplet compresses the
    domain, the target density rises to accommodate the new equilibrium
    instead of generating a reflected shock.

    The target density is updated each step with its own relaxation:

        rho_target <- rho_target + alpha_t * (rho_interior_mean - rho_target)

    Then a :class:`PressureReservoirBC`-style relaxation drives wall
    vertices toward the (now slowly drifting) target.

    Parameters
    ----------
    initial_rho_target : float
        Starting reference density [kg/m^3].
    tau_inv : float
        Wall relaxation rate [1/s] (fast — acoustic timescale).
    tau_inv_target : float
        Target drift rate [1/s] (slow — bulk response timescale).
        Typical: ``tau_inv_target << tau_inv``.
    gas_phase : int
        Phase index of the gas (default 0).
    update_m_phase : bool
        If True (multiphase), also update ``v.m_phase[gas_phase]``.
    """

    def __init__(
        self,
        initial_rho_target: float,
        tau_inv: float,
        tau_inv_target: float,
        gas_phase: int = 0,
        update_m_phase: bool = True,
    ):
        super().__init__()
        self.rho_target = float(initial_rho_target)
        self.tau_inv = float(tau_inv)
        self.tau_inv_target = float(tau_inv_target)
        self.gas_phase = int(gas_phase)
        self.update_m_phase = bool(update_m_phase)
        # Diagnostic state
        self.last_rho_interior = self.rho_target

    def apply(self, mesh, dt, target_vertices=None):
        # 1. Estimate the current interior mean density for this phase
        rho_sum, n = 0.0, 0
        for v in mesh.V:
            if getattr(v, 'phase', 0) != self.gas_phase:
                continue
            vol = getattr(v, 'dual_vol', 0.0)
            if vol < 1e-30:
                continue
            if getattr(v, 'boundary', False):
                continue  # skip wall-adjacent vertices from the estimate
            rho_sum += v.m / vol
            n += 1
        if n > 0:
            rho_interior = rho_sum / n
            self.last_rho_interior = rho_interior
            alpha_t = min(1.0, self.tau_inv_target * dt)
            self.rho_target = (
                self.rho_target + alpha_t * (rho_interior - self.rho_target)
            )

        # 2. Relax target vertices toward (drifting) rho_target
        verts = target_vertices if target_vertices is not None else mesh.V
        alpha = min(1.0, self.tau_inv * dt)
        count = 0
        for v in verts:
            if getattr(v, 'phase', 0) != self.gas_phase:
                continue
            vol = getattr(v, 'dual_vol', 0.0)
            if vol < 1e-30:
                continue
            m_target = self.rho_target * vol
            dm = alpha * (m_target - v.m)
            v.m = v.m + dm
            if self.update_m_phase:
                mp = getattr(v, 'm_phase', None)
                if mp is not None and self.gas_phase < len(mp):
                    mp[self.gas_phase] = v.m
            count += 1
        return count
