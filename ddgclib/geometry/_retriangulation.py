"""Delaunay retriangulation helpers for ddgclib.

Centralizes the pattern of running a Delaunay triangulation on a vertex
cloud, connecting edges, and caching the explicit simplex list on the
Complex for use by the simplex-aware 3D dual construction.

Why a helper
------------
The pattern

    tri = Delaunay(coords)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                verts[simplex[i]].connect(verts[simplex[j]])
    if dim == 3:
        HC._simplices = [
            tuple(verts[s[i]] for i in range(4)) for s in tri.simplices
        ]

appears in several places (the dynamic retopologization loop, multiphase
domain builders, periodic-BC ghost resolution, and benchmark setup).
Forgetting the last block silently drops the simplex-aware 3D dual path
and reintroduces ghost-tet connectivity bugs near domain boundaries.

This helper encapsulates the invariant: after calling it, ``HC`` has
correct primal edges AND a correct ``HC._simplices`` cache (when dim==3).

See also
--------
- ``docs/3d_simplex_aware_dual_fix.md`` — the fix documentation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def connect_and_cache_simplices(
    HC,
    verts: list,
    dim: int,
    simplices=None,
    coords=None,
    qhull_options: str | None = None,
) -> None:
    """Triangulate (if needed), connect primal edges, and cache simplices.

    This replaces the triplet (Delaunay → connect → cache HC._simplices)
    that was duplicated across ``_retopologize``, multiphase domain
    builders, periodic BCs, and benchmarks.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.  ``HC._simplices`` is set when ``dim == 3``.
    verts : list of vertex
        Ordered list of primal vertices matching ``coords``/``simplices``
        indices.
    dim : int
        Spatial dimension (2 or 3).
    simplices : array-like of shape (N, dim+1), optional
        Pre-computed simplex index list.  When provided, ``coords`` is
        ignored and no Delaunay call is made — useful for callers that
        already ran Delaunay (e.g. periodic-ghost resolution).
    coords : array-like, optional
        Vertex coordinate array of shape ``(len(verts), dim)``.  Required
        when ``simplices`` is None.
    qhull_options : str, optional
        Passed through to ``scipy.spatial.Delaunay`` when ``simplices``
        is computed here.  ``None`` uses scipy defaults; falls back to
        ``"Qbb Qt Qz"`` on cospherical/cocircular failure.

    Raises
    ------
    ValueError
        If neither ``simplices`` nor ``coords`` is provided, or if
        ``dim`` is not 2 or 3.

    Notes
    -----
    For ``dim == 2`` only primal edges are connected; there is no
    ``HC._simplices`` cache because the simplex-aware path exists only
    for 3D.
    """
    if dim not in (2, 3):
        raise ValueError(
            f"connect_and_cache_simplices supports dim ∈ {{2, 3}}; got {dim}"
        )

    if simplices is None:
        if coords is None:
            raise ValueError("Must provide either simplices or coords")
        coords = np.asarray(coords)
        try:
            if qhull_options is None:
                tri = Delaunay(coords)
            else:
                tri = Delaunay(coords, qhull_options=qhull_options)
        except Exception:
            # Cospherical / cocircular fallback
            tri = Delaunay(coords, qhull_options="Qbb Qt Qz")
        simplices = tri.simplices

    # Connect edges from the simplex list
    for simplex in simplices:
        n = len(simplex)
        for i in range(n):
            for j in range(i + 1, n):
                verts[simplex[i]].connect(verts[simplex[j]])

    # Cache explicit simplex list for the simplex-aware 3D dual path.
    # Only cache true 4-vertex tets (skip any lower-dim face entries that
    # some callers pass in, e.g. after ghost-dedup in periodic BCs).
    if dim == 3:
        HC._simplices = [
            tuple(verts[s[i]] for i in range(4))
            for s in simplices
            if len(s) == 4
        ]


def invalidate_simplex_cache(HC) -> None:
    """Clear ``HC._simplices`` (call after any topology change not routed
    through :func:`connect_and_cache_simplices`)."""
    if hasattr(HC, '_simplices'):
        HC._simplices = None
