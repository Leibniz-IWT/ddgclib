"""Method axis ``merge_method`` (laneV, 2026-10-07): the mass-conserving
pre-retopology merge with the per-phase ledger, against ``merge_all``."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from hyperct import Complex  # noqa: E402
from hyperct.ddg import compute_vd  # noqa: E402

from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize  # noqa: E402
from ddgclib.methods import PRESETS, SolverMethods  # noqa: E402
from ddgclib.multiphase import mass_conserving_merge  # noqa: E402


def _square(refinement=2, phases=True):
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()
    bV = HC.boundary(HC.V)
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method='barycentric')
    for i, v in enumerate(HC.V):
        v.m = 1.0 + 0.1 * i
        v.u = np.array([0.1 * i, -0.2 * i])
        v.p = float(i)
        if phases:
            v.m_phase = np.array([0.3 * v.m, 0.7 * v.m])
            v.phase = 1
    return HC, bV


def _totals(HC):
    m = sum(v.m for v in HC.V)
    mp = np.sum([v.m_phase for v in HC.V if hasattr(v, 'm_phase')], axis=0)
    mom = np.sum([v.m * v.u for v in HC.V], axis=0)
    return m, mp, mom


class TestMassConservingMerge:
    def test_sums_mass_ledger_and_momentum(self):
        HC, bV = _square()
        inner = [v for v in HC.V if v not in bV]
        a, b = inner[0], inner[1]
        HC.V.move(a, tuple(np.asarray(b.x) + 1e-12))
        m0, mp0, mom0 = _totals(HC)
        n = len(HC.V)
        assert mass_conserving_merge(HC, cdist=1e-9) == 1
        assert len(HC.V) == n - 1
        m1, mp1, mom1 = _totals(HC)
        np.testing.assert_allclose(m1, m0, rtol=1e-14)
        np.testing.assert_allclose(mp1, mp0, rtol=1e-14)
        np.testing.assert_allclose(mom1, mom0, rtol=1e-13)

    def test_prefer_keeps_the_wall_vertex(self):
        HC, bV = _square()
        wall = next(iter(sorted(bV, key=lambda v: tuple(v.x))))
        mobile = next(v for v in HC.V if v not in bV)
        HC.V.move(mobile, tuple(np.asarray(wall.x) + 1e-12))
        # without the preference the survivor is whichever comes first in
        # HC.V order; with it the wall vertex survives
        assert mass_conserving_merge(HC, cdist=1e-9, prefer=bV) == 1
        assert wall in HC.V and mobile not in HC.V

    def test_retopologize_axis(self):
        res = {}
        for method in ('merge_all', 'mass_conserving'):
            HC, bV = _square(phases=False)
            inner = [v for v in HC.V if v not in bV]
            HC.V.move(inner[0], tuple(np.asarray(inner[1].x) + 1e-12))
            m0 = sum(v.m for v in HC.V)
            _retopologize(HC, bV, 2, merge_cdist=1e-9, merge_method=method)
            res[method] = (sum(v.m for v in HC.V) / m0, len(HC.V))
        assert res['merge_all'][1] == res['mass_conserving'][1]
        assert res['merge_all'][0] < 1.0 - 1e-3           # mass lost
        np.testing.assert_allclose(res['mass_conserving'][0], 1.0, rtol=1e-14)
        HC, bV = _square(phases=False)
        with pytest.raises(ValueError, match='merge_method'):
            _retopologize(HC, bV, 2, merge_cdist=1e-9, merge_method='bogus')


class TestWiring:
    def test_solver_methods(self):
        assert SolverMethods(dim=2).merge_method == 'merge_all'
        with pytest.raises(ValueError, match='merge_cdist'):
            SolverMethods(dim=2, merge_method='mass_conserving')
        with pytest.raises(ValueError, match='connectivity'):
            SolverMethods(dim=2, connectivity='dual_only', merge_cdist=1e-3,
                          merge_method='mass_conserving')
        with pytest.raises(ValueError):
            SolverMethods(dim=2, merge_cdist=1e-3, merge_method='bogus')
        m = SolverMethods(dim=2, merge_cdist=1e-3, merge_method='mass_conserving')
        assert m.retopologize_fn().keywords == {'merge_method': 'mass_conserving'}
        assert SolverMethods(dim=2, merge_cdist=1e-3).retopologize_fn() is None
        mps = object()
        arm = PRESETS['dam_break_2D'].replace(merge_cdist=1e-3,
                                              merge_method='mass_conserving')
        kw = arm.retopologize_fn(mps=mps).keywords
        assert kw['merge_method'] == 'mass_conserving'
        assert arm.integrator_kwargs(mps=mps)['merge_cdist'] == 1e-3
