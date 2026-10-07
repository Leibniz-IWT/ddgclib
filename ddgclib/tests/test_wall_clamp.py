"""Method axis ``wall_clamp`` (laneV, 2026-10-07): the library
:class:`WallClampBC` and its wiring through ``SolverMethods`` into the
dam break, Hagen-Poiseuille and electrolysis setups.

The clamp is one-sided: it is the identity wherever impenetrability
holds, so a run in which no vertex leaves is bit-identical with and
without it (``test_identity_on_a_run_that_stays_inside``); a vertex
pushed through the floor is put back on it and loses its inward
velocity (``test_pushed_through_the_floor``).
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from hyperct import Complex  # noqa: E402
from hyperct.ddg import compute_vd  # noqa: E402

from ddgclib._boundary_conditions import (  # noqa: E402
    BoundaryConditionSet, WallClampBC,
)
from ddgclib.methods import PRESETS, SolverMethods  # noqa: E402


def _square(refinement=1):
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()
    bV = HC.boundary(HC.V)
    for v in HC.V:
        v.boundary = v in bV
        v.u = np.zeros(2)
    compute_vd(HC, method='barycentric')
    return HC, bV


class TestWallClampBC:
    def test_planes_and_box(self):
        bc = WallClampBC.box([(0.0, 2.0), (-1.0, 1.0)], axes=(1,), min_gap=0.1)
        assert bc.planes == [(1, -1.0, 1), (1, 1.0, -1)]
        assert bc.min_gap == 0.1
        bc = WallClampBC.box([(0.0, 2.0), (-1.0, 1.0)])
        assert bc.planes == [(0, 0.0, 1), (0, 2.0, -1), (1, -1.0, 1),
                             (1, 1.0, -1)]
        with pytest.raises(ValueError):
            WallClampBC([])
        with pytest.raises(ValueError):
            WallClampBC([(1, 0.0, 2)])

    def test_put_back_and_inward_velocity_zeroed(self):
        HC, bV = _square(1)
        inner = [v for v in HC.V if v not in bV]
        v = inner[0]
        HC.V.move(v, (0.3, -0.02))
        v.u = np.array([0.1, -0.2])
        bc = WallClampBC([(1, 0.0, +1)], min_gap=0.0, exclude=bV)
        assert bc.apply(HC, 0.0) == 1
        assert v.x_a[1] == 0.0 and v.x_a[0] == 0.3
        np.testing.assert_array_equal(v.u, [0.1, 0.0])
        # a vertex on the fluid side is never touched
        assert bc.apply(HC, 0.0) == 0
        # the velocity AWAY from the wall is kept
        HC.V.move(v, (0.3, -0.02))
        v.u = np.array([0.1, 0.5])
        assert bc.apply(HC, 0.0) == 1
        np.testing.assert_array_equal(v.u, [0.1, 0.5])
        # the gap: put down at level + gap, an upper wall at level - gap
        bc = WallClampBC([(1, 1.0, -1)], min_gap=0.05, exclude=bV)
        HC.V.move(v, (0.3, 1.1))
        v.u = np.array([0.0, 0.3])
        assert bc.apply(HC, 0.0) == 1
        assert v.x_a[1] == 0.95 and v.u[1] == 0.0

    def test_excluded_walls_and_target_ignored(self):
        HC, bV = _square(1)
        bc = WallClampBC([(1, 0.5, +1)], exclude=bV)
        # every wall vertex at y = 0 is below the plane but excluded; the
        # set's default target (bV) is ignored: the interior is clamped
        n_in = sum(1 for v in HC.V if v not in bV and v.x_a[1] < 0.5)
        assert n_in > 0
        bc_set = BoundaryConditionSet().add(bc, None)
        diag = bc_set.apply_all(HC, bV, 0.0)
        assert diag['bc_0_WallClampBC'] == n_in
        assert all(v.x_a[1] >= 0.5 for v in HC.V if v not in bV)
        assert any(v.x_a[1] < 0.5 for v in bV)      # the walls stayed


class TestSolverMethodsWiring:
    def test_axis_and_builder(self):
        m = SolverMethods(dim=2)
        assert m.wall_clamp is None
        assert m.wall_clamp_bc(box=[(0, 1), (0, 1)]) is None
        m = SolverMethods(dim=2, wall_clamp='project')
        bc = m.wall_clamp_bc(box=[(0, 1), (0, 1)], axes=(1,), min_gap=0.1,
                             exclude=set())
        assert isinstance(bc, WallClampBC) and bc.planes == [(1, 0.0, 1),
                                                             (1, 1.0, -1)]
        bc = m.wall_clamp_bc(planes=[(0, 2.0, -1)])
        assert bc.planes == [(0, 2.0, -1)]
        with pytest.raises(ValueError, match='planes= or box='):
            m.wall_clamp_bc()
        with pytest.raises(ValueError):
            SolverMethods(dim=2, wall_clamp='bogus')
        with pytest.raises(ValueError, match='not available in 1D'):
            SolverMethods(dim=1, wall_clamp='project')
        d = m.to_dict()
        assert d['wall_clamp'] == 'project'
        assert SolverMethods.from_dict(d) == m

    def test_electrolysis_presets_record_the_clamp(self):
        for name in ('electrolysis_bubble_2D', 'electrolysis_bubble_3D',
                     'electrolysis_bubble_fritz_2D'):
            assert PRESETS[name].wall_clamp == 'project'
        from cases_dynamic.electrolysis_bubble.src._setup import (
            setup_electrolysis_bubble,
        )
        for m, n_clamp in ((PRESETS['electrolysis_bubble_2D'], 1),
                           (PRESETS['electrolysis_bubble_2D'].replace(
                               wall_clamp=None), 0)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                HC, bV, mps, bc_set, *_ = setup_electrolysis_bubble(
                    dim=2, refinement_outer=1, refinement_droplet=2,
                    methods=m)
            clamps = [bc for bc, _ in bc_set._bcs
                      if isinstance(bc, WallClampBC)]
            assert len(clamps) == n_clamp
            if clamps:
                assert clamps[0].planes == [(1, -4e-3, 1), (1, 4e-3, -1)]
                assert clamps[0].min_gap == 0.02 * 1e-3
                assert clamps[0].exclude is bV

    def test_hagen_poiseuille_setups(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import (
            setup_poiseuille_2d_lagrangian, setup_poiseuille_developing,
        )
        base = PRESETS['hagen_poiseuille_2D']
        assert base.wall_clamp == 'project'
        assert PRESETS['hagen_poiseuille_3D'].wall_clamp is None
        for fn, kw in ((setup_poiseuille_2d_lagrangian, dict(L=2.0)),
                       (setup_poiseuille_developing,
                        dict(dim=2, L=3.0, mu=0.1, n_refine=1))):
            HC, bV, bc_set, wall, params = fn(
                methods=base.replace(wall_clamp=None), **kw)
            assert not any(isinstance(bc, WallClampBC) for bc, _ in bc_set._bcs)
            assert params['clamp_gap'] is None
            # the preset: 0.1 of the wall vertex spacing (0.5 at refinement 1)
            HC, bV, bc_set, wall, params = fn(methods=base, **kw)
            clamp = [bc for bc, _ in bc_set._bcs if isinstance(bc, WallClampBC)]
            assert len(clamp) == 1
            assert clamp[0].planes == [(1, 0.0, 1), (1, 1.0, -1)]
            assert clamp[0].min_gap == 0.05 and params['clamp_gap'] == 0.05
            assert clamp[0].exclude is bV
            HC, bV, bc_set, wall, params = fn(methods=base, clamp_gap=0.01,
                                              **kw)
            assert params['clamp_gap'] == 0.01
        with pytest.raises(ValueError, match='round'):
            setup_poiseuille_developing(
                dim=3, L=2.0, mu=0.1, n_refine=1,
                methods=PRESETS['hagen_poiseuille_3D'].replace(
                    wall_clamp='project'))


def _dam_break(methods, clamp_gap=None):
    from cases_dynamic.dam_break.src import _params as p
    from cases_dynamic.dam_break.src._setup import (
        cfl_timestep, setup_dam_break_multiphase,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        HC, bV, mps, bc_set, dudt_fn, _r, params = setup_dam_break_multiphase(
            dim=2, a=p.a, L=p.L, H=p.H, W=p.W, col_w=p.col_w, col_h=p.col_h,
            col_d=p.col_d, rho_l=p.rho_l, rho_g=p.rho_g, mu_l=p.mu_l,
            mu_g=p.mu_g, gamma=p.gamma, K_l=p.K_l, K_g=p.K_g, g=p.g,
            gravity_axis=p.gravity_axis, P_atm=p.P_atm, n_refine=2,
            alpha_art=0.3, methods=methods, clamp_gap=clamp_gap)
    dt = cfl_timestep(HC, 2, float(np.sqrt(p.K_l / p.rho_l)), cfl=p.cfl)
    return HC, bV, mps, bc_set, dudt_fn, params, dt


def _digest(HC):
    state = sorted((tuple(v.x_a[:2]), tuple(v.u[:2]), float(v.m)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


# The state of the refinement 2 dam break (alpha 0.3, dt 2.524e-4) 30
# steps after one air vertex was displaced 4 mm under the floor with 0.3
# m/s downward: the preset's clamp puts it back at the gap 0.1 * 0.05 m
# on the first step.  Digest of sorted (x, u, m); 2D runs are
# process-independent (protocol rule 8).
PIN_PUSHED_DIGEST = '39710661593753f5'


class TestDamBreak:
    def test_identity_on_a_run_that_stays_inside(self):
        """40 steps of the refinement 2 dam break: no vertex leaves, so the
        preset (wall_clamp='project' since laneV) and its wall_clamp=None
        arm end bit-identical."""
        out = {}
        for key, m in (('base', PRESETS['dam_break_2D'].replace(wall_clamp=None)),
                       ('clamp', PRESETS['dam_break_2D'])):
            assert PRESETS['dam_break_2D'].wall_clamp == 'project'
            assert PRESETS['dam_break_3D'].wall_clamp == 'project'
            HC, bV, mps, bc_set, dudt_fn, params, dt = _dam_break(m)
            clamps = [bc for bc, _ in bc_set._bcs if isinstance(bc, WallClampBC)]
            assert len(clamps) == (1 if key == 'clamp' else 0)
            if key == 'clamp':
                assert np.isclose(params['clamp_gap'], 0.1 * 0.05, rtol=1e-12)
            else:
                assert params['clamp_gap'] is None
            if clamps:
                assert clamps[0].planes == [(0, 0.0, 1), (0, 0.2, -1),
                                            (1, 0.0, 1), (1, 0.1, -1)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=40, bc_set=bc_set,
                            mps=mps)
            out[key] = _digest(HC)
        assert out['base'] == out['clamp']

    def test_pushed_through_the_floor(self):
        """An air vertex displaced through the floor before one step is
        back on the floor with no downward velocity after it (and stays
        outside without the clamp)."""
        res = {}
        for key, m in (('base', PRESETS['dam_break_2D'].replace(wall_clamp=None)),
                       ('clamp', PRESETS['dam_break_2D'])):
            HC, bV, mps, bc_set, dudt_fn, params, dt = _dam_break(m)
            air = [v for v in HC.V if v not in bV and v.phase == 0
                   and v.x_a[0] > 0.15]
            v = min(air, key=lambda w: w.x_a[1])
            x0 = v.x_a[0]
            HC.V.move(v, (x0, -0.004))
            v.u = np.array([0.0, -0.3])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=1, bc_set=bc_set,
                            mps=mps)
            res[key] = (float(v.x_a[1]), float(v.u[1]))
            if key == 'clamp':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=29,
                                bc_set=bc_set, mps=mps)
                assert all(w.x_a[1] >= 0.0 for w in HC.V)
                digest = _digest(HC)
        assert res['base'][0] < 0.0 and res['base'][1] < 0.0
        assert np.isclose(res['clamp'][0], 0.1 * 0.05, rtol=1e-12)
        assert res['clamp'][1] == 0.0
        assert digest == PIN_PUSHED_DIGEST
