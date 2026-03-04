"""DEM particle data structures.

Particle objects are deliberately kept SEPARATE from mesh vertices (HC.V).
They do not participate in Delaunay retriangulation, dual mesh computation,
or the fluid stress tensor pipeline. This avoids corrupting the fluid mesh
topology while enabling synchronized fluid-DEM time stepping.

Naming conventions match the HC vertex model where applicable:
``x_a`` for position, ``u`` for velocity, ``m`` for mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(eq=False)
class Particle:
    """A single DEM spherical particle.

    Parameters
    ----------
    x_a : np.ndarray
        Position vector, shape ``(dim,)``.
    u : np.ndarray
        Translational velocity, shape ``(dim,)``.
    omega : np.ndarray
        Angular velocity. Shape ``(3,)`` for 3D, ``(1,)`` for 2D.
    radius : float
        Particle radius [m].
    m : float
        Particle mass [kg].
    I : float
        Moment of inertia [kg m^2].
    rho_s : float
        Solid material density [kg/m^3].
    force : np.ndarray
        Accumulated force vector for current timestep, shape ``(dim,)``.
    torque : np.ndarray
        Accumulated torque vector for current timestep.
    id : int
        Unique particle identifier (assigned by ParticleSystem).
    cluster_id : int or None
        Aggregate/cluster membership. ``None`` = free particle.
    boundary : bool
        Whether this particle is frozen (domain boundary).
    wetted : bool
        Whether the particle surface is wetted (for liquid bridge formation).
    wetting_angle : float
        Contact angle [rad] for liquid bridge formation.
    liquid_volume : float
        Volume of liquid on particle surface [m^3].
    """

    x_a: np.ndarray
    u: np.ndarray
    omega: np.ndarray
    radius: float
    m: float
    I: float
    rho_s: float = 2500.0
    force: np.ndarray = field(default=None, repr=False)
    torque: np.ndarray = field(default=None, repr=False)
    id: int = 0
    cluster_id: Optional[int] = None
    boundary: bool = False
    wetted: bool = False
    wetting_angle: float = 0.0
    liquid_volume: float = 0.0

    def __post_init__(self) -> None:
        dim = len(self.x_a)
        if self.force is None:
            self.force = np.zeros(dim)
        if self.torque is None:
            self.torque = np.zeros(3 if dim == 3 else 1)

    @classmethod
    def sphere(
        cls,
        x: np.ndarray,
        radius: float,
        rho_s: float = 2500.0,
        dim: int = 3,
        u: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Particle:
        """Create a spherical particle from position, radius, and density.

        Computes mass and moment of inertia automatically.

        Parameters
        ----------
        x : array-like
            Center position, shape ``(dim,)``.
        radius : float
            Sphere radius.
        rho_s : float
            Material density [kg/m^3]. Default 2500 (silica).
        dim : int
            Spatial dimension (2 or 3).
        u : array-like or None
            Initial velocity. Defaults to zero.
        **kwargs
            Additional attributes (``wetted``, ``wetting_angle``, etc.).
        """
        x_a = np.asarray(x, dtype=float)
        if u is None:
            u_arr = np.zeros(dim)
        else:
            u_arr = np.asarray(u, dtype=float)
        omega = np.zeros(3 if dim == 3 else 1)

        if dim == 3:
            vol = (4.0 / 3.0) * np.pi * radius**3
            mass = rho_s * vol
            I_val = 0.4 * mass * radius**2  # 2/5 * m * R^2
        else:
            vol = np.pi * radius**2  # disk of unit thickness
            mass = rho_s * vol
            I_val = 0.5 * mass * radius**2  # 1/2 * m * R^2

        return cls(
            x_a=x_a,
            u=u_arr,
            omega=omega,
            radius=radius,
            m=mass,
            I=I_val,
            rho_s=rho_s,
            **kwargs,
        )

    def reset_forces(self) -> None:
        """Zero out accumulated force and torque for a new timestep."""
        self.force[:] = 0.0
        self.torque[:] = 0.0

    @property
    def dim(self) -> int:
        """Spatial dimension inferred from position vector."""
        return len(self.x_a)

    @property
    def kinetic_energy(self) -> float:
        """Translational + rotational kinetic energy."""
        return (
            0.5 * self.m * float(np.dot(self.u, self.u))
            + 0.5 * self.I * float(np.dot(self.omega, self.omega))
        )


class ParticleSystem:
    """Collection of DEM particles with batch operations.

    Manages a list of :class:`Particle` objects with unique-id tracking,
    gravity application, and cluster bookkeeping.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    gravity : np.ndarray or None
        Gravitational acceleration vector. Default: ``[0, 0, -9.81]`` (3D)
        or ``[0, -9.81]`` (2D).
    backend : str or None
        Backend name for batch operations (``"numpy"``, ``"torch"``, ``"gpu"``).
    """

    def __init__(
        self,
        dim: int = 3,
        gravity: Optional[np.ndarray] = None,
        backend: Optional[str] = None,
    ):
        self.dim = dim
        self.particles: list[Particle] = []
        self._id_map: dict[int, Particle] = {}
        self._next_id: int = 0
        self._next_cluster_id: int = 0

        if gravity is None:
            g = np.zeros(dim)
            if dim >= 3:
                g[2] = -9.81
            elif dim == 2:
                g[1] = -9.81
            self.gravity = g
        else:
            self.gravity = np.asarray(gravity, dtype=float)

        self._backend = None
        if backend is not None:
            from hyperct._backend import get_backend

            self._backend = get_backend(backend)

    def add(self, p: Particle) -> Particle:
        """Add a particle, assigning a unique id."""
        p.id = self._next_id
        self._next_id += 1
        self.particles.append(p)
        self._id_map[p.id] = p
        return p

    def remove(self, p: Particle) -> None:
        """Remove a particle from the system."""
        self.particles.remove(p)
        self._id_map.pop(p.id, None)

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, pid: int) -> Particle:
        """Look up a particle by its id."""
        return self._id_map[pid]

    # ---- batch accessors ----

    def positions(self) -> np.ndarray:
        """Return ``(N, dim)`` array of all particle positions."""
        if not self.particles:
            return np.empty((0, self.dim))
        return np.array([p.x_a for p in self.particles])

    def velocities(self) -> np.ndarray:
        """Return ``(N, dim)`` array of all particle velocities."""
        if not self.particles:
            return np.empty((0, self.dim))
        return np.array([p.u for p in self.particles])

    def radii(self) -> np.ndarray:
        """Return ``(N,)`` array of all particle radii."""
        return np.array([p.radius for p in self.particles])

    # ---- force helpers ----

    def reset_all_forces(self) -> None:
        """Reset force/torque accumulators on all particles."""
        for p in self.particles:
            p.reset_forces()

    def apply_gravity(self) -> None:
        """Add gravitational body force ``F = m * g`` to non-boundary particles."""
        for p in self.particles:
            if not p.boundary:
                p.force[: self.dim] += p.m * self.gravity

    # ---- cluster management ----

    def new_cluster_id(self) -> int:
        """Generate a new unique cluster/aggregate identifier."""
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    def cluster_particles(self, cluster_id: int) -> list[Particle]:
        """Return all particles belonging to a given cluster."""
        return [p for p in self.particles if p.cluster_id == cluster_id]

    # ---- diagnostics ----

    def total_kinetic_energy(self) -> float:
        """Sum of kinetic energy across all particles."""
        return sum(p.kinetic_energy for p in self.particles)

    def total_momentum(self) -> np.ndarray:
        """Total translational momentum vector."""
        if not self.particles:
            return np.zeros(self.dim)
        return sum(p.m * p.u[: self.dim] for p in self.particles)

    def center_of_mass(self) -> np.ndarray:
        """Mass-weighted center of mass position."""
        if not self.particles:
            return np.zeros(self.dim)
        total_m = sum(p.m for p in self.particles)
        return sum(p.m * p.x_a[: self.dim] for p in self.particles) / total_m
