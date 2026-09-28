"""Fluid-particle two-way coupling for DEM.

Provides drag force computation, fluid velocity interpolation at particle
centres, and reaction force feedback onto mesh vertices. Follows operator
splitting: fluid step → coupling → DEM step → feedback.

The coupler uses inverse-distance-weighted (IDW) interpolation from nearby
mesh vertices, consistent with the finite volume stencils in
``ddgclib.operators.stress``.

Usage
-----
::

    coupler = FluidParticleCoupler(HC, ps, dim=3, mu=8.9e-4, rho_f=1000.0)

    for step in range(n_steps):
        symplectic_euler(HC, bV, dudt_fn, dt=dt, n_steps=1, dim=dim)
        coupler.fluid_to_particle(dt)
        dem_step(ps, detector, model, dt=dt, dim=dim,
                 external_forces_fn=coupler.get_external_forces_fn())
        coupler.particle_to_fluid(dt)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ddgclib.dem._particle import Particle, ParticleSystem


def _build_kdtree(positions: np.ndarray):
    """Build a scipy KDTree from vertex positions.

    Returns ``None`` if scipy is not available (graceful degradation).
    """
    try:
        from scipy.spatial import KDTree
        return KDTree(positions)
    except ImportError:
        return None


def interpolate_fluid_at_particle(
    p: Particle,
    vertices: list,
    positions: np.ndarray,
    dim: int,
    n_nearest: int = 6,
    kdtree=None,
) -> tuple[np.ndarray, float]:
    """Interpolate fluid velocity and pressure at particle centre.

    Uses inverse-distance-weighted (IDW) interpolation from the
    ``n_nearest`` mesh vertices.

    Parameters
    ----------
    p : Particle
        The DEM particle whose centre is the query point.
    vertices : list
        All mesh vertex objects (must have ``.x_a``, ``.u``, ``.p``).
    positions : np.ndarray
        Shape ``(N, dim)`` array of vertex positions (precomputed).
    dim : int
        Spatial dimension.
    n_nearest : int
        Number of neighbours for IDW interpolation.
    kdtree : scipy.spatial.KDTree or None
        Pre-built KDTree (if ``None``, brute-force nearest-neighbour).

    Returns
    -------
    u_fluid : np.ndarray
        Interpolated fluid velocity at particle centre, shape ``(dim,)``.
    p_fluid : float
        Interpolated fluid pressure at particle centre.
    """
    query = p.x_a[:dim]

    if kdtree is not None:
        dists, indices = kdtree.query(query, k=min(n_nearest, len(vertices)))
        if np.ndim(indices) == 0:
            indices = [int(indices)]
            dists = [float(dists)]
    else:
        # Brute-force fallback
        diff = positions - query[np.newaxis, :]
        all_dists = np.linalg.norm(diff, axis=1)
        k = min(n_nearest, len(vertices))
        if k >= len(vertices):
            indices = np.arange(len(vertices))
        else:
            indices = np.argpartition(all_dists, k)[:k]
        dists = all_dists[indices]

    # Inverse-distance weights
    dists = np.asarray(dists, dtype=float)
    weights = 1.0 / np.maximum(dists, 1e-30)
    weights /= weights.sum()

    u_fluid = np.zeros(dim)
    p_fluid = 0.0
    for w, idx in zip(weights, indices):
        v = vertices[idx]
        u_fluid += w * np.asarray(v.u[:dim], dtype=float)
        p_fluid += w * float(np.asarray(v.p).ravel()[0])

    return u_fluid, p_fluid


def apply_particle_feedback(
    force: np.ndarray,
    p: Particle,
    vertices: list,
    positions: np.ndarray,
    dim: int,
    n_nearest: int = 6,
    kdtree=None,
) -> None:
    """Distribute a reaction force from a particle back onto mesh vertices.

    Uses the same IDW weighting as :func:`interpolate_fluid_at_particle`
    so that momentum is conserved (action = −reaction).

    Parameters
    ----------
    force : np.ndarray
        The reaction force to distribute (typically ``-F_drag``).
    p : Particle
        The particle sourcing the reaction.
    vertices : list
        Mesh vertex objects (must have ``.force`` or we add to ``.u``
        via direct force attribute).
    positions : np.ndarray
        Shape ``(N, dim)`` vertex positions.
    dim : int
    n_nearest : int
    kdtree : scipy.spatial.KDTree or None
    """
    query = p.x_a[:dim]

    if kdtree is not None:
        dists, indices = kdtree.query(query, k=min(n_nearest, len(vertices)))
        if np.ndim(indices) == 0:
            indices = [int(indices)]
            dists = [float(dists)]
    else:
        diff = positions - query[np.newaxis, :]
        all_dists = np.linalg.norm(diff, axis=1)
        k = min(n_nearest, len(vertices))
        if k >= len(vertices):
            indices = np.arange(len(vertices))
        else:
            indices = np.argpartition(all_dists, k)[:k]
        dists = all_dists[indices]

    dists = np.asarray(dists, dtype=float)
    weights = 1.0 / np.maximum(dists, 1e-30)
    weights /= weights.sum()

    for w, idx in zip(weights, indices):
        v = vertices[idx]
        # Add feedback force as acceleration impulse on fluid vertex
        f_share = w * force
        if hasattr(v, "dem_feedback_force"):
            v.dem_feedback_force[:dim] += f_share[:dim]
        else:
            v.dem_feedback_force = np.zeros(dim)
            v.dem_feedback_force[:dim] = f_share[:dim]


def _stokes_drag(
    u_rel: np.ndarray, R: float, mu: float
) -> np.ndarray:
    """Stokes drag: ``F = 6 * pi * mu * R * u_rel``."""
    return 6.0 * np.pi * mu * R * u_rel


def _schiller_naumann_drag(
    u_rel: np.ndarray, R: float, mu: float, rho_f: float
) -> np.ndarray:
    """Schiller-Naumann drag correlation.

    ::

        F = 6 * pi * mu * R * u_rel * (1 + 0.15 * Re^0.687)

    Valid for ``Re < 800``.
    """
    u_mag = float(np.linalg.norm(u_rel))
    Re = rho_f * u_mag * (2.0 * R) / max(mu, 1e-30)
    correction = 1.0 + 0.15 * Re**0.687
    return 6.0 * np.pi * mu * R * u_rel * correction


class FluidParticleCoupler:
    """Two-way coupling between the fluid mesh and DEM particles.

    Parameters
    ----------
    HC : Complex
        The hyperct simplicial complex (fluid mesh).
    ps : ParticleSystem
        The DEM particle system.
    dim : int
        Spatial dimension.
    drag_model : str
        ``"stokes"`` or ``"schiller_naumann"``.
    mu : float
        Dynamic viscosity [Pa.s].
    rho_f : float
        Fluid density [kg/m^3].
    n_nearest : int
        Number of nearest vertices for IDW interpolation.
    """

    def __init__(
        self,
        HC,
        ps: ParticleSystem,
        dim: int = 3,
        drag_model: str = "stokes",
        mu: float = 8.9e-4,
        rho_f: float = 1000.0,
        n_nearest: int = 6,
    ):
        self.HC = HC
        self.ps = ps
        self.dim = dim
        self.mu = mu
        self.rho_f = rho_f
        self.n_nearest = n_nearest

        if drag_model == "stokes":
            self._drag_fn = _stokes_drag
        elif drag_model == "schiller_naumann":
            self._drag_fn = _schiller_naumann_drag
        else:
            raise ValueError(
                f"Unknown drag model {drag_model!r}. "
                f"Available: 'stokes', 'schiller_naumann'"
            )
        self._drag_model = drag_model

        # Per-particle drag forces (stored for feedback phase)
        self._drag_forces: dict[int, np.ndarray] = {}

        # Cached mesh data (call cache_mesh_data() for fixed meshes)
        self._cached_vertices: Optional[list] = None
        self._cached_positions: Optional[np.ndarray] = None
        self._cached_kdtree = None

    def cache_mesh_data(self) -> None:
        """Pre-compute and cache vertex list, positions, and KDTree.

        Call once after mesh setup for fixed (non-moving) meshes.  Avoids
        rebuilding these data structures every coupling step.
        """
        self._cached_vertices, self._cached_positions = (
            self._get_vertices_and_positions()
        )
        self._cached_kdtree = _build_kdtree(self._cached_positions)

    def invalidate_cache(self) -> None:
        """Clear cached mesh data (call after mesh topology changes)."""
        self._cached_vertices = None
        self._cached_positions = None
        self._cached_kdtree = None

    def _get_vertices_and_positions(self) -> tuple[list, np.ndarray]:
        """Extract vertex list and position array from the fluid mesh."""
        vertices = list(self.HC.V)
        dim = self.dim
        positions = np.array([v.x_a[:dim] for v in vertices])
        return vertices, positions

    def fluid_to_particle(self, dt: float) -> None:
        """Interpolate fluid at each particle, compute drag, store forces.

        Call after fluid step, before DEM step.
        """
        if self._cached_vertices is not None:
            vertices = self._cached_vertices
            positions = self._cached_positions
            kdtree = self._cached_kdtree
        else:
            vertices, positions = self._get_vertices_and_positions()
            kdtree = _build_kdtree(positions)

        if len(vertices) == 0:
            return

        self._drag_forces.clear()

        for p in self.ps.particles:
            if p.boundary:
                continue

            u_fluid, _ = interpolate_fluid_at_particle(
                p, vertices, positions, self.dim,
                n_nearest=self.n_nearest, kdtree=kdtree,
            )

            u_rel = u_fluid - p.u[:self.dim]

            if self._drag_model == "stokes":
                F_drag = self._drag_fn(u_rel, p.radius, self.mu)
            else:
                F_drag = self._drag_fn(u_rel, p.radius, self.mu, self.rho_f)

            # Store for feedback phase and for get_external_forces_fn()
            self._drag_forces[p.id] = F_drag

    def particle_to_fluid(self, dt: float) -> None:
        """Distribute reaction drag forces back onto mesh vertices.

        Call after DEM step. Reaction is ``-F_drag`` (Newton III).
        """
        if not self._drag_forces:
            return

        if self._cached_vertices is not None:
            vertices = self._cached_vertices
            positions = self._cached_positions
            kdtree = self._cached_kdtree
        else:
            vertices, positions = self._get_vertices_and_positions()
            kdtree = _build_kdtree(positions)

        if len(vertices) == 0:
            return

        for p in self.ps.particles:
            if p.id not in self._drag_forces:
                continue
            F_drag = self._drag_forces[p.id]
            reaction = -F_drag  # Newton III
            apply_particle_feedback(
                reaction, p, vertices, positions, self.dim,
                n_nearest=self.n_nearest, kdtree=kdtree,
            )

    def get_external_forces_fn(self):
        """Return a callable ``fn(ps)`` for use as ``external_forces_fn``.

        This allows the coupler to be plugged into ``dem_step()`` directly.
        The returned function applies the stored drag forces.
        """
        def _apply_drag(ps: ParticleSystem) -> None:
            for p in ps.particles:
                if p.id in self._drag_forces:
                    p.force[:self.dim] += self._drag_forces[p.id]

        return _apply_drag
