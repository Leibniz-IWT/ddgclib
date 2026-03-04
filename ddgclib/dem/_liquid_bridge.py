"""Capillary liquid bridge force model between wetted particles.

Implements the classical toroidal approximation for liquid bridge forces
between two spheres. This connects conceptually to the existing catenoid
bridge geometry in ``ddgclib._catenoid`` (which models surface geometry),
but here we compute the capillary force directly for DEM.

Bridge formation criteria: particles within a critical distance and both
wetted. Bridge rupture: separation exceeds critical distance (Lian et al.
1993 rupture criterion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ddgclib.dem._particle import Particle, ParticleSystem


@dataclass(eq=False)
class LiquidBridge:
    """A capillary liquid bridge between two wetted particles.

    Parameters
    ----------
    p_i, p_j : Particle
        Bridged particles.
    gamma : float
        Surface tension of the liquid [N/m].
    volume : float
        Bridge liquid volume [m^3].
    half_filling_angle : float
        Half-filling angle beta [rad].
    active : bool
        Whether bridge is intact.
    """

    p_i: Particle
    p_j: Particle
    gamma: float
    volume: float
    half_filling_angle: float = 0.1
    active: bool = True

    @property
    def separation(self) -> float:
        """Surface-to-surface separation distance."""
        dx = self.p_j.x_a - self.p_i.x_a
        dist = float(np.linalg.norm(dx))
        return dist - self.p_i.radius - self.p_j.radius

    @property
    def rupture_distance(self) -> float:
        """Critical rupture distance (Lian et al. 1993).

        ::

            s_rupt = (1 + 0.5 * theta) * V^(1/3)

        where ``theta`` is the average contact angle and ``V`` is bridge
        volume.
        """
        theta_avg = 0.5 * (self.p_i.wetting_angle + self.p_j.wetting_angle)
        return (1.0 + 0.5 * theta_avg) * self.volume ** (1.0 / 3.0)

    def check_rupture(self) -> bool:
        """Check if bridge should rupture. Sets ``active=False`` if so.

        Returns
        -------
        bool
            ``True`` if bridge has ruptured.
        """
        if self.separation > self.rupture_distance:
            self.active = False
        return not self.active

    def capillary_force(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute capillary bridge force (attractive).

        Uses the leading-order toroidal approximation (Willett et al. 2000)::

            F_bridge ~ -2 * pi * gamma * R_star * cos(theta)

        with ``R_star = 2*R_i*R_j / (R_i + R_j)`` and linear decay to zero
        at the rupture distance.

        Returns
        -------
        F_i : np.ndarray
            Force on particle i (toward j).
        F_j : np.ndarray
            Force on particle j (toward i).
        """
        if not self.active:
            return np.zeros(dim), np.zeros(dim)

        R_i, R_j = self.p_i.radius, self.p_j.radius
        R_star = 2.0 * R_i * R_j / (R_i + R_j)
        theta_avg = 0.5 * (self.p_i.wetting_angle + self.p_j.wetting_angle)

        dx = self.p_j.x_a[:dim] - self.p_i.x_a[:dim]
        dist = float(np.linalg.norm(dx))
        n_ij = dx / max(dist, 1e-30)

        s = max(self.separation, 0.0)
        s_rupt = self.rupture_distance

        if s >= s_rupt:
            self.active = False
            return np.zeros(dim), np.zeros(dim)

        # Capillary force magnitude (leading-order + linear decay)
        F_0 = 2.0 * np.pi * self.gamma * R_star * np.cos(theta_avg)
        decay = max(0.0, 1.0 - s / max(s_rupt, 1e-30))
        F_mag = F_0 * decay

        # Attractive: pulls i toward j
        F_i = F_mag * n_ij
        F_j = -F_i

        return F_i, F_j


class LiquidBridgeManager:
    """Container for liquid bridges. Handles formation, rupture, and forces.

    Parameters
    ----------
    gamma : float
        Surface tension [N/m]. Default: 0.072 (water at 25 C).
    formation_distance : float or None
        Maximum surface-to-surface distance for bridge formation.
        If ``None``, defaults to ``0.5 * min(R_i, R_j)`` per pair.
    bridge_volume_fraction : float
        Fraction of particle ``liquid_volume`` used per bridge.
    """

    def __init__(
        self,
        gamma: float = 0.072,
        formation_distance: Optional[float] = None,
        bridge_volume_fraction: float = 0.1,
    ):
        self.gamma = gamma
        self.formation_distance = formation_distance
        self.bridge_volume_fraction = bridge_volume_fraction
        self.bridges: list[LiquidBridge] = []
        self._pair_set: set[frozenset] = set()

    def check_formation(self, ps: ParticleSystem) -> int:
        """Check all wetted particle pairs for new bridge formation.

        Returns the number of new bridges formed.
        """
        n_formed = 0
        particles = ps.particles
        dim = ps.dim
        for i, p_i in enumerate(particles):
            if not p_i.wetted:
                continue
            for j in range(i + 1, len(particles)):
                p_j = particles[j]
                if not p_j.wetted:
                    continue

                pair_key = frozenset((p_i.id, p_j.id))
                if pair_key in self._pair_set:
                    continue

                # Check distance criterion
                sep = (
                    float(np.linalg.norm(p_j.x_a[:dim] - p_i.x_a[:dim]))
                    - p_i.radius
                    - p_j.radius
                )
                max_dist = self.formation_distance or (
                    0.5 * min(p_i.radius, p_j.radius)
                )

                if sep <= max_dist:
                    vol = self.bridge_volume_fraction * min(
                        p_i.liquid_volume, p_j.liquid_volume
                    )
                    bridge = LiquidBridge(
                        p_i=p_i,
                        p_j=p_j,
                        gamma=self.gamma,
                        volume=max(vol, 1e-30),
                    )
                    self.bridges.append(bridge)
                    self._pair_set.add(pair_key)
                    n_formed += 1

        return n_formed

    def check_ruptures(self) -> int:
        """Check all bridges for rupture. Returns count ruptured."""
        n_ruptured = 0
        for bridge in self.bridges:
            if bridge.active and bridge.check_rupture():
                n_ruptured += 1
                pair_key = frozenset((bridge.p_i.id, bridge.p_j.id))
                self._pair_set.discard(pair_key)
        self.bridges = [b for b in self.bridges if b.active]
        return n_ruptured

    def apply_forces(self, ps: ParticleSystem) -> None:
        """Apply capillary bridge forces to particles."""
        dim = ps.dim
        for bridge in self.bridges:
            if not bridge.active:
                continue
            F_i, F_j = bridge.capillary_force(dim)
            bridge.p_i.force[:dim] += F_i
            bridge.p_j.force[:dim] += F_j

    @property
    def active_count(self) -> int:
        """Number of currently active bridges."""
        return sum(1 for b in self.bridges if b.active)
