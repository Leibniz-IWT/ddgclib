"""Hand-built complexes get exact simplex volumes through one library call
(laneI, 2026-10-06).

laneS gave every domain builder its top-simplex cache through
``DomainResult``; a complex assembled by hand (``Complex(...)`` +
``triangulate`` + ``refine_all``, an ``extrude``, a cube-to-tube
projection) still read the 3D ``v_star`` fan walk at setup (box total
0.9167 at refinement 1: laneS section 7).  ``ddgclib.geometry
.ensure_simplex_cache`` is the one entry point; every such setup under
``cases_dynamic/`` calls it, and the dual volumes it produces tile the
domain.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.geometry import ensure_simplex_cache, invalidate_simplex_cache
from ddgclib.operators.stress import cache_dual_volumes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.tests.test_builder_simplex_cache import _enclosed_measure  # noqa: E402


def _total(HC, dim):
    hull = HC.boundary()
    for v in HC.V:
        v.boundary = v in hull
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, dim)
    return sum(v.dual_vol for v in HC.V)


class TestEnsureSimplexCache:
    @pytest.mark.parametrize('dim', [2, 3])
    def test_populates_a_bare_complex_once(self, dim):
        HC = Complex(dim, domain=[(0.0, 1.0)] * dim)
        HC.triangulate()
        HC.refine_all()
        assert getattr(HC, '_simplices', None) is None
        n = ensure_simplex_cache(HC, dim)
        assert n == len(HC._simplices) > 0
        cache = HC._simplices
        assert ensure_simplex_cache(HC, dim) == n
        assert HC._simplices is cache            # an existing cache is kept
        assert _total(HC, dim) == pytest.approx(1.0, abs=1e-12)

    def test_without_the_cache_the_3d_fan_walk_does_not_tile(self):
        HC = Complex(3, domain=[(0.0, 1.0)] * 3)
        HC.triangulate()
        HC.refine_all()
        assert _total(HC, 3) == pytest.approx(0.9166666666666667, abs=1e-12)
        invalidate_simplex_cache(HC)
        ensure_simplex_cache(HC, 3)
        assert _total(HC, 3) == pytest.approx(1.0, abs=1e-12)

    def test_1d_is_a_no_op(self):
        HC = Complex(1, domain=[(0.0, 1.0)])
        HC.triangulate()
        assert ensure_simplex_cache(HC, 1) == 0


class TestHandBuiltSetupsTileTheirDomain:
    """Every setup under cases_dynamic/ that builds a Complex by hand."""

    @pytest.mark.parametrize('dim', [2, 3])
    def test_cube_flow(self, dim):
        from cases_dynamic.cube_flow.src._setup import setup_cube_flow
        HC, bV, ic, bc_set, unit_mesh, params = setup_cube_flow(
            dim=dim, n_refine=1, L=2.0)
        for mesh in (HC, unit_mesh):
            assert mesh._simplices is not None
            assert _total(mesh, dim) == pytest.approx(2.0**dim, rel=1e-12)

    @pytest.mark.parametrize('dim, n', [(2, 2), (3, 1)])
    def test_cube_to_droplet(self, dim, n):
        from cases_dynamic.cube2droplet.src._setup import setup_cube_to_droplet
        out = setup_cube_to_droplet(dim=dim, n_refine=n, R=0.01,
                                    L_domain=0.03)
        HC = out[0]
        assert HC._simplices is not None
        # the setup caches dual volumes itself (dual_volume per vertex)
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(
            0.06**dim, rel=1e-12)

    def test_unit_cylinder_and_cube_to_tube(self):
        from cases_dynamic.Hagen_Poiseuile.src._geometry import (
            cube_to_tube, unit_cylinder)
        for HC in (unit_cylinder(0.5, refinements=1, height=2.0),
                   cube_to_tube(0.5, refinements=1, height=2.0)):
            assert HC._simplices is not None
            total = _total(HC, 3)
            assert total == pytest.approx(_enclosed_measure(HC, 3), rel=1e-12)
            assert 0.9 * np.pi * 0.25 * 2.0 < total < np.pi * 0.25 * 2.0

    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_hydrostatic_setup(self, dim):
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(dim=dim, n_refine=1,
                                                       h=1.0)
        if dim > 1:
            assert HC._simplices is not None
        assert _total(HC, dim) == pytest.approx(1.0, rel=1e-12)

    def test_poiseuille_2d_setups(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import (
            setup_poiseuille_2d, setup_poiseuille_2d_lagrangian)
        HC = setup_poiseuille_2d(n_refine=1, L=2.0, h=1.0)[0]
        assert HC._simplices is not None
        assert _total(HC, 2) == pytest.approx(2.0, rel=1e-12)
        HC = setup_poiseuille_2d_lagrangian(L=2.0, D=1.0)[0]
        assert HC._simplices is not None
        assert _total(HC, 2) == pytest.approx(2.0, rel=1e-12)
