"""Contact detection for spherical DEM particles.

Two-phase approach:

1. **Broad phase**: cell-linked list (spatial hash) with cell size ``2 * R_max``.
   Follows the same grid pattern used in
   ``hyperct._vertex.VertexCacheBase.merge_all()``.
2. **Narrow phase**: exact sphere-sphere overlap test with contact geometry
   computation (normal, tangential velocity decomposition, contact point).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iterproduct
from typing import Optional

import numpy as np

from ddgclib.dem._particle import Particle, ParticleSystem


@dataclass
class Contact:
    """A detected contact between two particles.

    Attributes
    ----------
    p_i : Particle
        First particle.
    p_j : Particle
        Second particle.
    delta_n : float
        Normal overlap (positive when overlapping):
        ``R_i + R_j - |x_j - x_i|``.
    n_ij : np.ndarray
        Unit normal vector from i to j.
    x_contact : np.ndarray
        Contact point position.
    v_rel : np.ndarray
        Relative velocity at contact point (including rotation).
    v_n : float
        Normal component of relative velocity (positive = approaching).
    v_t : np.ndarray
        Tangential component of relative velocity.
    """

    p_i: Particle
    p_j: Particle
    delta_n: float
    n_ij: np.ndarray
    x_contact: np.ndarray
    v_rel: np.ndarray
    v_n: float
    v_t: np.ndarray


class ContactDetector:
    """Broad-phase + narrow-phase contact detection.

    Uses cell-linked list spatial hashing. Cell size is ``2 * R_max``,
    ensuring any pair of overlapping spheres lies in the same or adjacent cells.

    Parameters
    ----------
    ps : ParticleSystem
        The particle system to detect contacts in.
    cell_size : float or None
        Override cell size. If ``None``, uses ``2 * max(radii)``.
    """

    def __init__(self, ps: ParticleSystem, cell_size: Optional[float] = None):
        self.ps = ps
        self._cell_size = cell_size

    def detect(self) -> list[Contact]:
        """Run broad-phase and narrow-phase detection.

        Returns list of :class:`Contact` objects for all overlapping pairs.
        """
        particles = self.ps.particles
        if len(particles) < 2:
            return []

        dim = self.ps.dim
        radii = self.ps.radii()
        r_max = float(radii.max())
        cell_size = self._cell_size or 2.0 * r_max

        # Build spatial hash grid
        grid: dict[tuple, list[int]] = {}
        for idx, p in enumerate(particles):
            cell = tuple(int(np.floor(c / cell_size)) for c in p.x_a[:dim])
            grid.setdefault(cell, []).append(idx)

        offsets = list(iterproduct([-1, 0, 1], repeat=dim))

        contacts: list[Contact] = []
        seen: set[tuple[int, int]] = set()

        for cell_key, cell_indices in grid.items():
            for i in cell_indices:
                for offset in offsets:
                    neighbor_cell = tuple(
                        c + o for c, o in zip(cell_key, offset)
                    )
                    for j in grid.get(neighbor_cell, []):
                        if j <= i:
                            continue
                        pair = (i, j)
                        if pair in seen:
                            continue
                        seen.add(pair)

                        contact = _sphere_sphere_test(
                            particles[i], particles[j], dim
                        )
                        if contact is not None:
                            contacts.append(contact)

        return contacts


def _sphere_sphere_test(
    p_i: Particle, p_j: Particle, dim: int
) -> Optional[Contact]:
    """Exact sphere-sphere overlap test.

    Returns a :class:`Contact` if the particles overlap, ``None`` otherwise.
    """
    dx = p_j.x_a[:dim] - p_i.x_a[:dim]
    dist = float(np.linalg.norm(dx))
    delta_n = p_i.radius + p_j.radius - dist

    if delta_n <= 0.0:
        return None

    # Unit normal from i to j
    n_ij = dx / max(dist, 1e-30)

    # Contact point: weighted by radii on the overlap midline
    x_contact = p_i.x_a[:dim] + (p_i.radius - 0.5 * delta_n) * n_ij

    # Relative velocity at contact (including tangential from rotation)
    v_rel = p_j.u[:dim] - p_i.u[:dim]
    if dim == 3:
        r_i = x_contact - p_i.x_a[:dim]
        r_j = x_contact - p_j.x_a[:dim]
        v_rel = v_rel + np.cross(p_j.omega, r_j) - np.cross(p_i.omega, r_i)

    v_n = float(np.dot(v_rel, n_ij))
    v_t = v_rel - v_n * n_ij

    return Contact(
        p_i=p_i,
        p_j=p_j,
        delta_n=delta_n,
        n_ij=n_ij,
        x_contact=x_contact,
        v_rel=v_rel,
        v_n=v_n,
        v_t=v_t,
    )
