"""Backward-compat shim — the canonical implementations now live in
:mod:`hyperct.ddg`.

Existing imports

    from ddgclib.geometry import connect_and_cache_simplices
    from ddgclib.geometry._retriangulation import invalidate_simplex_cache

continue to work; new code should prefer

    from hyperct.ddg import connect_and_cache_simplices, invalidate_simplex_cache

The 2D simplex-cache (in addition to 3D) is populated automatically — see
``hyperct/ddg/_retriangulation.py`` for the canonical docstring.
"""
from hyperct.ddg import (  # noqa: F401
    connect_and_cache_simplices,
    invalidate_simplex_cache,
)

__all__ = ["connect_and_cache_simplices", "invalidate_simplex_cache"]
