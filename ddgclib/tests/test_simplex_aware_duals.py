"""Regression tests for the simplex-aware dual + boundary refactor.

These tests cover the new code added in the
"vertex-simplex Complex storage" refactor:

- 2D simplex caching via :func:`hyperct.ddg.connect_and_cache_simplices`.
- :func:`hyperct.ddg.boundary_from_simplices` (2D + 3D, regular grid +
  jittered).
- The simplex-aware path in :func:`hyperct.ddg.compute_vd`
  (`_compute_vd_2d_simplex_aware`).
- The opt-in ``retopologize=True`` flag on the domain builders.
- Cache invalidation regression.

The 3D jittered linear-precision regression is in
``test_stress.py::test_p_ij_linear_precision_jittered_3d``.  We add a
2D analogue here.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from hyperct import Complex
from hyperct.ddg import (
    boundary_from_simplices,
    compute_vd,
    connect_and_cache_simplices,
    invalidate_simplex_cache,
)
from scipy.spatial import Delaunay


def _build_jittered_2d(refinement: int = 2, seed: int = 42):
    """Build a jittered 2D rectangle and re-Delaunay (matches _retopologize)."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        v.boundary = any(
            abs(v.x_a[d]) < 1e-14 or abs(v.x_a[d] - 1.0) < 1e-14
            for d in range(2)
        )
        if v.boundary:
            bV.add(v)

    rng = np.random.default_rng(seed)
    for v in HC.V:
        if v not in bV and v.nn:
            el = min(np.linalg.norm(v.x_a - vn.x_a) for vn in v.nn)
            off = rng.uniform(-0.05 * el, 0.05 * el, size=2)
            HC.V.move(v, tuple(v.x_a[d] + off[d] for d in range(2)))

    verts = list(HC.V)
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)
    coords = np.array([v.x_a[:2] for v in verts])
    connect_and_cache_simplices(HC, verts, 2, coords=coords)
    return HC, bV


class TestConnectAndCacheSimplices2D:
    """2D simplex caching via connect_and_cache_simplices."""

    def test_2d_simplex_cache_populated(self):
        HC, _ = _build_jittered_2d()
        assert HC._simplices is not None
        assert len(HC._simplices) > 0
        # Each entry is a 3-tuple of vertex objects (top-dim simplex in 2D)
        for s in HC._simplices:
            assert len(s) == 3
            for v in s:
                # Vertex objects must remain in HC.V (referential integrity)
                assert v in HC.V

    def test_2d_simplex_count_matches_delaunay(self):
        HC, _ = _build_jittered_2d()
        verts = list(HC.V)
        coords = np.array([v.x_a[:2] for v in verts])
        tri = Delaunay(coords)
        assert len(HC._simplices) == len(tri.simplices)


class TestComputeVd2DSimplexAware:
    """The 2D dispatch path (_compute_vd_2d_simplex_aware)."""

    def test_compute_vd_2d_uses_simplex_path_when_cached(self):
        HC, _ = _build_jittered_2d()
        compute_vd(HC, method='barycentric')
        # Every interior primal vertex should have a non-empty dual cell
        for v in HC.V:
            if not v.boundary:
                assert len(v.vd) > 0, (
                    f"Interior vertex {v.x} has empty v.vd after "
                    "simplex-aware compute_vd"
                )

    def test_p_ij_linear_precision_jittered_2d(self):
        """Linear precision on JITTERED 2D mesh with simplex-aware compute_vd.

        2D analogue of ``test_p_ij_linear_precision_jittered_3d``.  The
        identity p_ij = 0.5 sum_j (d_ij outer A_ij) must be diagonal
        (off-diagonal entries vanish) for any valid dual; this is the
        first-order linear precision check.  Without the simplex-aware
        path, jittered-mesh ghost K_3 cliques can violate this near
        boundaries.
        """
        from ddgclib.operators.stress import dual_area_vector
        HC, bV = _build_jittered_2d()
        compute_vd(HC, method='barycentric', cdist=1e-10)

        interior = [v for v in HC.V if v not in bV]
        assert len(interior) > 0

        for v in interior:
            tensor = np.zeros((2, 2))
            for nb in v.nn:
                A_ij = dual_area_vector(v, nb, HC, dim=2)
                d_ij = nb.x_a[:2] - v.x_a[:2]
                tensor += 0.5 * np.outer(d_ij, A_ij)
            diag = np.diag(tensor)
            off_diag = tensor - np.diag(diag)
            npt.assert_allclose(
                off_diag, 0.0, atol=1e-12,
                err_msg=(
                    f"Linear precision violated on jittered 2D mesh at {v.x} "
                    "— simplex-aware 2D fix may have regressed"
                ),
            )


class TestBoundaryFromSimplices:
    """boundary_from_simplices on 2D and 3D meshes."""

    def test_2d_regular_grid(self):
        HC, bV = _build_jittered_2d()
        b = boundary_from_simplices(HC, dim=2)
        # Boundary set from cached simplices must equal the analytical
        # rectangular boundary set.
        assert b == bV

    def test_3d_regular_grid(self):
        from ddgclib.geometry.domains import box
        result = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2, retopologize=True)
        b = boundary_from_simplices(result.HC, dim=3)
        # Domain builder identifies bV via geometric position; the
        # simplex-aware boundary must match that exactly.
        assert b == result.bV

    def test_raises_when_simplices_none(self):
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        with pytest.raises(ValueError, match="HC._simplices is None"):
            boundary_from_simplices(HC, dim=2)

    def test_raises_unsupported_dim(self):
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC._simplices = [(None,)]  # dummy
        with pytest.raises(ValueError, match=r"dim ∈"):
            boundary_from_simplices(HC, dim=4)


class TestInvalidation:
    """invalidate_simplex_cache and the legacy fallback."""

    def test_invalidate_resets_to_none(self):
        HC, _ = _build_jittered_2d()
        assert HC._simplices is not None
        invalidate_simplex_cache(HC)
        assert HC._simplices is None

    def test_compute_vd_falls_back_after_invalidation(self):
        """After invalidation, compute_vd uses the legacy nn path without
        error (regression: must not read a stale simplex list)."""
        HC, bV = _build_jittered_2d()
        # Drop the cache; we want to verify the legacy path still runs.
        invalidate_simplex_cache(HC)
        compute_vd(HC, method='barycentric')
        for v in HC.V:
            if not v.boundary:
                assert len(v.vd) > 0


class TestDomainBuildersRetopologize:
    """The opt-in retopologize=True flag on domain builders."""

    def test_rectangle_retopologize_populates_simplices(self):
        from ddgclib.geometry.domains import rectangle
        # Default behaviour: no simplex cache.
        r0 = rectangle(L=2.0, h=1.0, refinement=2)
        assert r0.HC._simplices is None
        # Opt-in: cache populated.
        r1 = rectangle(L=2.0, h=1.0, refinement=2, retopologize=True)
        assert r1.HC._simplices is not None
        assert len(r1.HC._simplices) > 0

    def test_cylinder_retopologize_populates_simplices(self):
        from ddgclib.geometry.domains import cylinder_volume
        r = cylinder_volume(R=0.5, L=1.0, refinement=2, retopologize=True)
        assert r.HC._simplices is not None
        assert len(r.HC._simplices) > 0
        # Vertex correspondence: every cached simplex vertex object
        # must still live in HC.V (no stale references).
        for s in r.HC._simplices:
            for v in s:
                assert v in r.HC.V

    def test_rectangle_retopologize_boundary_matches(self):
        """Boundary set survives a retopologize pass (vertex objects
        preserved, fields preserved by reference)."""
        from ddgclib.geometry.domains import rectangle
        r = rectangle(L=2.0, h=1.0, refinement=2, retopologize=True)
        b = boundary_from_simplices(r.HC, dim=2)
        # bV is updated in-place by _retopologize so it matches the
        # post-retopo boundary.  These must agree.
        assert b == r.bV
