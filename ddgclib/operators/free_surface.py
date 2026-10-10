"""Capillary force on the free surface of a single-phase mesh (laneI).

The free surface of a single-phase Lagrangian mesh is a set of boundary
facets of its top simplices (edges in 2D, triangles in 3D).  Surface
tension and wall adhesion enter as the gradient of the capillary energy

    E = gamma A_free - gamma cos(theta) A_wet

with ``A_free`` the area of the facets that bound the liquid against the
gas and ``A_wet`` the area of the wall facets that touch the contact
line.  The force on a vertex is ``F_i = -dE/dx_i``:

- on a free-surface vertex ``-gamma sum_T d|T|/dx_i``.  In 2D that is
  ``gamma (t_next - t_prev)`` (the integrated curvature normal of
  :mod:`ddgclib.operators.curvature_2d`), in 3D the cotangent form of the
  mean curvature normal, both integrated over the dual cell already;
- on a contact-line vertex the same, plus the Young term
  ``+gamma cos(theta) sum_T d|T|/dx_i`` over its wall facets: ``gamma
  cos(theta)`` per unit contact-line length, directed along the wall
  towards the dry side.  Its wall-normal part is the wall reaction and
  is discarded by the slide BC that holds the vertex on the wall
  (:class:`ddgclib._boundary_conditions.AxialSlideBC`).

At equilibrium the pressure force of the open dual fan of a surface
vertex (``p A_open``, the free-surface force of laneK / laneP) balances
this tension, which is the discrete Young-Laplace condition
``p = -gamma kappa``; the Young term fixes the angle between the surface
and the wall at ``theta``.  Both are energy gradients, so the static
state is a minimum of ``E + potential energy`` on the discrete mesh.

The facets are read from ``HC._simplices`` (the boundary facets of the
cached top simplices, :func:`ddgclib.methods._retopo.oriented_boundary`)
and classified by vertex membership, so the operator follows a
reconnecting connectivity: the facet lists are rebuilt whenever the
simplex cache object changes.  Vertex sets are held by ``id`` because a
vertex re-hashes when it moves.

The operator is selected by the method axis ``contact_line``
(``SolverMethods(contact_line='energy_gradient')``) and bound by
``SolverMethods.dudt_fn(free_surface=FreeSurface(...))``.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

__all__ = ['FreeSurface', 'facet_area', 'facet_area_gradient']


def facet_area(pts: np.ndarray) -> float:
    """Measure of a facet: length of an edge (``(2, 2)``) or area of a
    triangle (``(3, 3)``)."""
    if pts.shape[0] == 2:
        return float(np.linalg.norm(pts[1] - pts[0]))
    return 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0],
                                               pts[2] - pts[0])))


def facet_area_gradient(pts: np.ndarray, i: int) -> np.ndarray:
    """``d(facet_area) / d(pts[i])``.

    Edge ``(a, b)``: ``d|e|/da = (a - b) / |e|``.  Triangle ``(a, b, c)``:
    ``d|T|/da = n x (c - b) / 2`` with ``n`` the unit normal
    ``(b - a) x (c - a) / |...|`` (the cotangent formula).  A degenerate
    facet has no gradient (zeros).
    """
    if pts.shape[0] == 2:
        e = pts[i] - pts[1 - i]
        n = float(np.linalg.norm(e))
        return e / n if n > 0.0 else np.zeros(2)
    a = pts[i]
    b = pts[(i + 1) % 3]
    c = pts[(i + 2) % 3]
    n = np.cross(b - a, c - a)
    nn = float(np.linalg.norm(n))
    if nn <= 0.0:
        return np.zeros(3)
    return 0.5 * np.cross(n / nn, c - b)


class FreeSurface:
    """Energy-gradient capillary force on the free surface of a
    single-phase mesh (module docstring).

    Parameters
    ----------
    HC : Complex
        Mesh with its simplex cache ``HC._simplices`` (every domain
        builder and :func:`ddgclib.geometry.ensure_simplex_cache` give
        it; every library retopology keeps it).
    dim : int
        2 or 3.
    gamma : float
        Surface tension [N/m].
    theta_deg : float
        Static contact angle of the liquid at the wall [deg].
    walls : iterable of vertices
        Wall vertices (the frozen set): a boundary facet whose vertices
        are all walls or contact vertices, with at least one contact
        vertex, is a wetted wall facet.
    free : iterable of vertices
        Free-surface vertices: a boundary facet whose vertices are all
        free is a free facet.  Includes the contact vertices.
    contact : iterable of vertices
        The contact-line vertices (free vertices on the wall).

    A boundary facet that is neither (a wall facet without a contact
    vertex, a reservoir floor) carries no force.
    """

    def __init__(self, HC, dim: int, gamma: float, theta_deg: float,
                 walls: Iterable, free: Iterable, contact: Iterable):
        if dim not in (2, 3):
            raise ValueError(f"FreeSurface is 2D / 3D, got dim={dim}")
        self.HC = HC
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.theta_deg = float(theta_deg)
        self.cos_theta = math.cos(math.radians(theta_deg))
        self._walls = {id(v) for v in walls}
        self._free = {id(v) for v in free}
        self._contact = {id(v) for v in contact}
        if not self._contact <= self._free:
            raise ValueError("contact vertices must be free-surface vertices")
        self._key = None
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the facet lists from the current simplex cache."""
        from ddgclib.methods._retopo import oriented_boundary
        simplices = getattr(self.HC, '_simplices', None)
        if not simplices:
            raise ValueError("FreeSurface needs HC._simplices (a builder "
                             "mesh, ensure_simplex_cache or a retopology)")
        self.free_facets: list[tuple] = []
        self.wet_facets: list[tuple] = []
        for f in oriented_boundary(simplices, self.dim):
            ids = [id(v) for v in f]
            if all(i in self._free for i in ids):
                self.free_facets.append(tuple(f))
            elif (any(i in self._contact for i in ids)
                  and all(i in self._walls or i in self._contact
                          for i in ids)):
                self.wet_facets.append(tuple(f))
        self._inc_free: dict[int, list[tuple[tuple, int]]] = {}
        self._inc_wet: dict[int, list[tuple[tuple, int]]] = {}
        for facets, inc in ((self.free_facets, self._inc_free),
                            (self.wet_facets, self._inc_wet)):
            for f in facets:
                for i, v in enumerate(f):
                    inc.setdefault(id(v), []).append((f, i))
        self._key = simplices

    def _check(self) -> None:
        if getattr(self.HC, '_simplices', None) is not self._key:
            self.refresh()

    @staticmethod
    def _pts(f: tuple, dim: int) -> np.ndarray:
        return np.array([v.x_a[:dim] for v in f], dtype=float)

    # ------------------------------------------------------------------
    def is_free(self, v) -> bool:
        return id(v) in self._free

    def is_contact(self, v) -> bool:
        return id(v) in self._contact

    def area_free(self) -> float:
        self._check()
        return sum(facet_area(self._pts(f, self.dim)) for f in self.free_facets)

    def area_wet(self) -> float:
        self._check()
        return sum(facet_area(self._pts(f, self.dim)) for f in self.wet_facets)

    def energy(self) -> float:
        """``gamma A_free - gamma cos(theta) A_wet`` [J] (per unit depth
        in 2D)."""
        return self.gamma * (self.area_free() - self.cos_theta * self.area_wet())

    def force(self, v) -> np.ndarray:
        """``-dE/dx_v`` [N]; zero on a vertex without free or wetted
        facets."""
        self._check()
        F = np.zeros(self.dim)
        for f, i in self._inc_free.get(id(v), ()):
            F -= self.gamma * facet_area_gradient(self._pts(f, self.dim), i)
        for f, i in self._inc_wet.get(id(v), ()):
            F += (self.gamma * self.cos_theta
                  * facet_area_gradient(self._pts(f, self.dim), i))
        return F
