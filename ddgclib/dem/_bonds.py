"""Sintered particle bonds and aggregate tracking.

Implements rigid (or semi-rigid) bonds between sintered particles.
Bonds transmit forces along the bond axis (spring-dashpot).
Neck radius growth follows a Frenkel/Kuczynski sintering model.

Bond data structure connects particle pairs, analogous to how
``vertex.nn`` connects mesh vertices in the hyperct simplicial complex.
The :class:`BondManager` container mirrors the
:class:`~ddgclib._boundary_conditions.BoundaryConditionSet` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ddgclib.dem._particle import Particle, ParticleSystem


@dataclass(eq=False)
class SinterBond:
    """A sintered bridge bond between two particles.

    Parameters
    ----------
    p_i : Particle
        First bonded particle.
    p_j : Particle
        Second bonded particle.
    neck_radius : float
        Current sintering neck radius [m].
    rest_length : float
        Equilibrium center-to-center distance at bond formation.
    k_bond : float
        Bond stiffness [N/m].
    c_bond : float
        Bond damping [N s/m].
    max_strain : float
        Fracture criterion: bond breaks if ``|stretch| / rest_length > max_strain``.
    active : bool
        Whether this bond is still intact.
    formation_time : float
        Simulation time when bond was formed.
    """

    p_i: Particle
    p_j: Particle
    neck_radius: float
    rest_length: float
    k_bond: float = 1e6
    c_bond: float = 1e2
    max_strain: float = 0.1
    active: bool = True
    formation_time: float = 0.0

    @classmethod
    def form(
        cls,
        p_i: Particle,
        p_j: Particle,
        t: float = 0.0,
        initial_neck_ratio: float = 0.1,
        k_bond: float = 1e6,
        c_bond: float = 1e2,
        max_strain: float = 0.1,
    ) -> SinterBond:
        """Form a new bond between two particles at current positions.

        Parameters
        ----------
        p_i, p_j : Particle
            The particles to bond.
        t : float
            Current simulation time.
        initial_neck_ratio : float
            Initial neck radius as fraction of ``min(R_i, R_j)``.
        k_bond : float
            Bond stiffness [N/m].
        c_bond : float
            Bond damping [N s/m].
        max_strain : float
            Fracture strain threshold.
        """
        rest_length = float(np.linalg.norm(p_j.x_a - p_i.x_a))
        neck_radius = initial_neck_ratio * min(p_i.radius, p_j.radius)
        return cls(
            p_i=p_i,
            p_j=p_j,
            neck_radius=neck_radius,
            rest_length=rest_length,
            k_bond=k_bond,
            c_bond=c_bond,
            max_strain=max_strain,
            active=True,
            formation_time=t,
        )

    def force_and_torque(
        self, dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute bond force and torque.

        Normal force along bond axis (spring + dashpot).
        Checks fracture criterion and deactivates bond if exceeded.

        Returns
        -------
        F_i : np.ndarray
            Force on particle i.
        F_j : np.ndarray
            Force on particle j.
        tau_i : np.ndarray
            Torque on particle i.
        tau_j : np.ndarray
            Torque on particle j.
        """
        dx = self.p_j.x_a[:dim] - self.p_i.x_a[:dim]
        dist = float(np.linalg.norm(dx))
        n_ij = dx / max(dist, 1e-30)

        # Stretch relative to rest length
        stretch = dist - self.rest_length

        # Normal force (spring + dashpot along bond axis)
        v_rel_n = float(
            np.dot(self.p_j.u[:dim] - self.p_i.u[:dim], n_ij)
        )
        F_bond_mag = self.k_bond * stretch + self.c_bond * v_rel_n
        F_on_i = F_bond_mag * n_ij
        F_on_j = -F_on_i

        # Fracture check
        if abs(stretch) / max(self.rest_length, 1e-30) > self.max_strain:
            self.active = False

        # Torques (zero for pure axial bond; bending moments would extend this)
        tau_i = np.zeros(3 if dim == 3 else 1)
        tau_j = np.zeros(3 if dim == 3 else 1)

        return F_on_i, F_on_j, tau_i, tau_j

    def grow_neck(
        self, dt: float, T: float = 300.0, D: float = 1e-15
    ) -> None:
        """Advance neck radius by Frenkel/Kuczynski sintering model.

        Neck growth rate::

            dx/dt ~ D / (x * R_eff)

        where ``x`` = neck_radius, ``R_eff`` = min particle radius,
        ``D`` = diffusion coefficient (temperature-dependent).

        Parameters
        ----------
        dt : float
            Time step.
        T : float
            Temperature [K] (reserved for Arrhenius-type D(T) in future).
        D : float
            Effective diffusion coefficient [m^2/s].
        """
        R_eff = min(self.p_i.radius, self.p_j.radius)
        if self.neck_radius < 1e-30 or R_eff < 1e-30:
            return
        dx_dt = D / (self.neck_radius * R_eff)
        self.neck_radius += dx_dt * dt
        # Cap at full coalescence
        self.neck_radius = min(self.neck_radius, R_eff)


class BondManager:
    """Container for all sintered bonds in the system.

    Manages bond formation, breakage, force application, and sintering
    growth. Mirrors the
    :class:`~ddgclib._boundary_conditions.BoundaryConditionSet` pattern.
    """

    def __init__(self):
        self.bonds: list[SinterBond] = []

    def add(self, bond: SinterBond, ps: Optional[ParticleSystem] = None) -> SinterBond:
        """Register a new bond and assign shared cluster_id.

        Parameters
        ----------
        bond : SinterBond
            The bond to register.
        ps : ParticleSystem or None
            If given, uses ``ps.new_cluster_id()`` for unique ids.
            Otherwise uses ``id(bond)`` as a placeholder.
        """
        self.bonds.append(bond)
        # Assign same cluster_id to bonded particles
        if bond.p_i.cluster_id is None and bond.p_j.cluster_id is None:
            cid = ps.new_cluster_id() if ps is not None else id(bond)
            bond.p_i.cluster_id = cid
            bond.p_j.cluster_id = cid
        elif bond.p_i.cluster_id is not None:
            bond.p_j.cluster_id = bond.p_i.cluster_id
        else:
            bond.p_i.cluster_id = bond.p_j.cluster_id
        return bond

    def apply_forces(self, ps: ParticleSystem) -> None:
        """Apply bond forces to all active bonds."""
        dim = ps.dim
        for bond in self.bonds:
            if not bond.active:
                continue
            F_i, F_j, tau_i, tau_j = bond.force_and_torque(dim)
            bond.p_i.force[:dim] += F_i[:dim]
            bond.p_j.force[:dim] += F_j[:dim]
            bond.p_i.torque += tau_i
            bond.p_j.torque += tau_j

    def grow_all_necks(
        self, dt: float, T: float = 300.0, D: float = 1e-15
    ) -> None:
        """Advance sintering neck growth on all active bonds."""
        for bond in self.bonds:
            if bond.active:
                bond.grow_neck(dt, T=T, D=D)

    def remove_broken(self) -> int:
        """Remove broken bonds. Returns count removed."""
        n_before = len(self.bonds)
        self.bonds = [b for b in self.bonds if b.active]
        return n_before - len(self.bonds)

    def bonds_for_particle(self, p: Particle) -> list[SinterBond]:
        """Return all active bonds involving particle ``p``."""
        return [
            b for b in self.bonds if b.active and (b.p_i is p or b.p_j is p)
        ]

    @property
    def active_count(self) -> int:
        """Number of currently active bonds."""
        return sum(1 for b in self.bonds if b.active)
