"""Structured return type for domain builder functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperct import Complex


@dataclass
class DomainResult:
    """Result of a domain builder function.

    Attributes
    ----------
    HC : Complex
        The simplicial complex with the filled domain mesh.
    bV : set
        Set of all boundary vertex objects.
    boundary_groups : dict[str, set]
        Named boundary regions.  Standard keys (when applicable):
        ``'walls'``, ``'inlet'``, ``'outlet'``.
        Custom keys depend on the domain type.
    dim : int
        Spatial dimension (2 or 3).
    metadata : dict[str, Any]
        Domain-specific metadata (radius, length, volume, etc.)
        for downstream use.
    """

    HC: Complex
    bV: set
    boundary_groups: dict[str, set] = field(default_factory=dict)
    dim: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    def tag_boundaries(self) -> None:
        """Set ``v.boundary = True/False`` on every vertex based on *bV*.

        Must be called before ``compute_vd()``.  Called automatically
        by every domain builder.
        """
        for v in self.HC.V:
            v.boundary = v in self.bV

    def retopologize(self, **retopo_kwargs) -> None:
        """Run ``_retopologize`` in place on this domain.

        Drops the structured Kuhn-decomposition produced by
        ``Complex.triangulate() + refine_all()``, runs a scipy
        ``Delaunay`` re-triangulation, populates ``HC._simplices`` and
        rebuilds duals via :func:`hyperct.ddg.compute_vd`.  After this,
        ``boundary_from_simplices`` (and the simplex-aware dual paths)
        are active.

        This is the canonical way to obtain "already-retopologized"
        initial conditions for dynamic case studies — the dynamic
        integrators perform the same call at the start of every time
        step, so starting from this state can lead to more stable
        initial transients.

        Parameters
        ----------
        **retopo_kwargs
            Forwarded to
            :func:`ddgclib.dynamic_integrators._integrators_dynamic._retopologize`
            (e.g. ``merge_cdist``, ``boundary_filter``, ``backend``).

        Notes
        -----
        Delaunay does not in general produce the same tessellation as
        the structured ``triangulate + refine_all`` path, especially in
        3D where the structured path produces a Kuhn decomposition
        useful for tests with axis-aligned interfaces (see memory
        ``project_flat_interface_3d_mesh.md``).  This is *intentional*
        — pass ``retopologize=True`` to a builder only when a Delaunay
        starting state is acceptable.
        """
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize,
        )
        # _retopologize updates v.boundary, recomputes duals, and
        # populates HC._simplices via connect_and_cache_simplices.
        _retopologize(self.HC, self.bV, self.dim, **retopo_kwargs)

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_verts = self.HC.V.size()
        groups_str = ", ".join(
            f"{k}: {len(v)}" for k, v in self.boundary_groups.items()
        )
        return (
            f"DomainResult: {n_verts} vertices, {len(self.bV)} boundary, "
            f"dim={self.dim}, groups=[{groups_str}]"
        )
