"""Simplex-cache helpers.  The canonical rebuild / invalidate routines live
in :mod:`hyperct.ddg`; this module re-exports them and adds the one
ddgclib entry point for a complex that was built by hand.

Existing imports

    from ddgclib.geometry import connect_and_cache_simplices
    from ddgclib.geometry._retriangulation import invalidate_simplex_cache

continue to work; new code should prefer

    from hyperct.ddg import connect_and_cache_simplices, invalidate_simplex_cache

The 2D simplex-cache (in addition to 3D) is populated automatically by the
domain builders (``DomainResult``) through :func:`ensure_simplex_cache`;
see ``hyperct/ddg/_retriangulation.py`` for the canonical docstring.
"""
from hyperct.ddg import (  # noqa: F401
    connect_and_cache_simplices,
    invalidate_simplex_cache,
)

__all__ = ["connect_and_cache_simplices", "invalidate_simplex_cache",
           "ensure_simplex_cache"]


def ensure_simplex_cache(HC, dim: int) -> int:
    """Give a hand-built complex its top-simplex cache ``HC._simplices``.

    NOTE(laneI-hand-built-cache): the counterpart of the builder hook in
    ``DomainResult.__post_init__`` (laneS) for a ``Complex`` that was
    assembled by hand (``Complex(...).triangulate(); refine_all()``, an
    ``extrude``, a cube-to-tube projection).  Without the cache the first
    ``cache_dual_volumes`` call reads the 3D ``v_star`` fan walk, whose
    volumes do not tile the domain (box total 0.9167 at refinement 1),
    and in 2D the (repaired) ``dual_cell_area_2d`` fallback.  With it the
    setup volumes are the exact barycentric ones,
    ``Vol_i = sum_{T contains i} |T| / (dim + 1)``, the same source every
    later retopology uses.

    The cache is rebuilt from the EXISTING connectivity
    (``hyperct.ddg.rebuild_simplex_cache_2d`` / ``_3d``: the K_3 / K_4
    cliques of the 1-skeleton, nothing re-triangulated); a cache that is
    already there is left alone.  In 1D there is nothing to cache.  Call
    it after the last connectivity edit of the setup; a later hand edit
    of the connectivity needs ``invalidate_simplex_cache`` or a rebuild.

    Returns the number of cached simplices (0 when nothing was done or
    the 3D clique enumeration refused the mesh, see
    ``rebuild_simplex_cache_3d``).
    """
    if getattr(HC, '_simplices', None) is not None:
        return len(HC._simplices)
    if dim == 2:
        from hyperct.ddg import rebuild_simplex_cache_2d
        return rebuild_simplex_cache_2d(HC)
    if dim == 3:
        from hyperct.ddg import rebuild_simplex_cache_3d
        return rebuild_simplex_cache_3d(HC)
    return 0
