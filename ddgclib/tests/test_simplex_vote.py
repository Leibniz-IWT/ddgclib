"""Method axis ``simplex_vote`` (laneV, 2026-10-07): the mass-fraction
vote of ``MultiphaseSystem.assign_simplex_phases_from_vertices`` against
the bulk majority, on a hand-labelled grid and on the dam break setup."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from hyperct import Complex  # noqa: E402
from hyperct.ddg import compute_vd  # noqa: E402

from ddgclib.eos._tait_murnaghan import TaitMurnaghan  # noqa: E402
from ddgclib.methods import PRESETS, SolverMethods  # noqa: E402
from ddgclib.multiphase import (  # noqa: E402
    INTERFACE_PHASE, MultiphaseSystem, PhaseProperties, _simplex_key,
    iter_top_simplices,
)

RHO = (1.0, 1000.0)      # air, liquid


def _mps():
    eos = [TaitMurnaghan(rho0=r, P0=0.0, K=1e5, n=1.0) for r in RHO]
    return MultiphaseSystem(phases=[
        PhaseProperties(eos=eos[0], mu=1e-5, rho0=RHO[0], name='air'),
        PhaseProperties(eos=eos[1], mu=1e-3, rho0=RHO[1], name='liq')],
        gamma={(0, 1): 0.0})


def _strip(f_liq, lone=False):
    """Unit square refined once (vertex rows y = 0, 0.25, 0.5, 0.75, 1):
    the rows y <= 0.25 liquid bulk, the row y = 0.5 interface vertices
    carrying the liquid volume fraction *f_liq* (volume-equivalent mass
    share), the rows y >= 0.75 air bulk; *lone* makes (0.25, 0.75) a bulk
    liquid vertex surrounded by air and interface vertices (two of its
    triangles are (liquid, air, interface) ties)."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = HC.boundary(HC.V)
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method='barycentric')
    for v in HC.V:
        x, y = v.x_a[0], v.x_a[1]
        vol = 1.0
        if y < 0.4:
            v.phase, m = 1, np.array([0.0, RHO[1] * vol])
        elif y > 0.6:
            v.phase, m = 0, np.array([RHO[0] * vol, 0.0])
            if lone and abs(x - 0.25) < 1e-9 and abs(y - 0.75) < 1e-9:
                v.phase, m = 1, np.array([0.0, RHO[1] * vol])
        else:
            v.phase = INTERFACE_PHASE
            m = np.array([RHO[0] * (1 - f_liq) * vol, RHO[1] * f_liq * vol])
        v.interface_phases = frozenset({0, 1}) if v.phase < 0 else frozenset({v.phase})
        v.m_phase = m
        v.m = float(m.sum())
    return HC


def _census(simplex):
    """(liquid bulk, air bulk, interface) vertex counts."""
    ph = [int(v.phase) for v in simplex]
    return ph.count(1), ph.count(0), ph.count(-1)


class TestVote:
    def test_bulk_majority_is_the_default_and_unchanged(self):
        HC = _strip(0.5)
        mps = _mps()
        mps.assign_simplex_phases_from_vertices(HC, 2)
        for s in iter_top_simplices(HC, 2):
            nL, nA, nI = _census(s)
            k = mps.simplex_phase[_simplex_key(s)]
            assert k == (1 if nL > 0 else 0)
        with pytest.raises(ValueError, match='vote'):
            mps.assign_simplex_phases_from_vertices(HC, 2, vote='bogus')

    @pytest.mark.parametrize('f_liq, top_label', [(1 / 3, 0), (0.9, 1)])
    def test_mass_fraction_reads_the_ledger(self, f_liq, top_label):
        """Triangles (interface, interface, air) go liquid when the
        interface row carries more than 3/4 liquid: the interface has moved
        in the ledger.  Triangles (liquid, interface, interface) stay liquid
        above 1/4."""
        HC = _strip(f_liq)
        mps = _mps()
        mps.assign_simplex_phases_from_vertices(HC, 2, vote='mass_fraction')
        for s in iter_top_simplices(HC, 2):
            nL, nA, nI = _census(s)
            k = mps.simplex_phase[_simplex_key(s)]
            if nL > 0:
                assert k == 1              # (I, L, L) and (I, I, L)
            elif nI == 2:
                assert k == top_label      # (I, I, A)
            else:
                assert k == 0              # (I, A, A)

    @pytest.mark.parametrize('f_liq, expected', [(0.6, 1), (0.4, 0)])
    def test_lone_bulk_vertex_tie(self, f_liq, expected):
        """A triangle (liquid bulk, air bulk, interface) is a tie for the
        bulk majority (-> air, the lower phase ID); the mass fraction gives
        it to the phase the interface vertex carries more of."""
        HC = _strip(f_liq, lone=True)
        mps = _mps()
        ties = []
        for s in iter_top_simplices(HC, 2):
            phases = sorted(int(v.phase) for v in s)
            if phases == [-1, 0, 1]:
                ties.append(s)
        assert ties
        mps.assign_simplex_phases_from_vertices(HC, 2)
        assert all(mps.simplex_phase[_simplex_key(s)] == 0 for s in ties)
        mps.assign_simplex_phases_from_vertices(HC, 2, vote='mass_fraction')
        assert all(mps.simplex_phase[_simplex_key(s)] == expected for s in ties)

    def test_vertex_without_ledger_votes_by_label(self):
        HC = _strip(0.5)
        mps = _mps()
        for v in HC.V:
            del v.m_phase
        mps.assign_simplex_phases_from_vertices(HC, 2, vote='mass_fraction')
        ref = dict(mps.simplex_phase)
        mps.assign_simplex_phases_from_vertices(HC, 2)
        assert mps.simplex_phase == ref


class TestWiring:
    def test_solver_methods(self):
        assert SolverMethods(dim=2, phases='multi').simplex_vote == 'bulk_majority'
        with pytest.raises(ValueError, match="phases='multi'"):
            SolverMethods(dim=2, simplex_vote='mass_fraction')
        with pytest.raises(ValueError):
            SolverMethods(dim=2, phases='multi', simplex_vote='bogus')
        mps = object()
        base = PRESETS['dam_break_2D']
        assert 'simplex_vote' not in base.retopologize_fn(mps=mps).keywords
        arm = base.replace(simplex_vote='mass_fraction')
        assert arm.retopologize_fn(mps=mps).keywords['simplex_vote'] == 'mass_fraction'
        per = PRESETS['shearing_plate_droplet_2D'].replace(
            simplex_vote='mass_fraction')
        fn = per.retopologize_fn(mps=mps, domain_bounds=[(0, 1), (0, 1)])
        assert fn.keywords['simplex_vote'] == 'mass_fraction'

    def test_refresh_forwards_the_vote_on_the_dam_break_setup(self):
        """On the clean ledger of the setup the two votes agree on every
        simplex that has no interface vertex; the dam break then runs 20
        steps on the arm that the lane measured through its horizon
        (simplex split + vote; under the neighbour-count split the vote
        ejects at step 0, laneV log section 2.2)."""
        from cases_dynamic.dam_break.src import _params as p
        from cases_dynamic.dam_break.src._setup import (
            cfl_timestep, setup_dam_break_multiphase,
        )

        def build(m):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return setup_dam_break_multiphase(
                    dim=2, a=p.a, L=p.L, H=p.H, W=p.W, col_w=p.col_w,
                    col_h=p.col_h, col_d=p.col_d, rho_l=p.rho_l,
                    rho_g=p.rho_g, mu_l=p.mu_l, mu_g=p.mu_g,
                    gamma=p.gamma, K_l=p.K_l, K_g=p.K_g, g=p.g,
                    gravity_axis=p.gravity_axis, P_atm=p.P_atm,
                    n_refine=2, alpha_art=0.3, methods=m)

        labels = {}
        for key, m in (('base', PRESETS['dam_break_2D']),
                       ('arm', PRESETS['dam_break_2D'].replace(
                           simplex_vote='mass_fraction'))):
            HC, bV, mps, bc_set, dudt_fn, _r, params = build(m)
            labels[key] = {
                frozenset(tuple(v.x_a[:2]) for v in s): (
                    mps.simplex_phase[_simplex_key(s)],
                    any(v.phase < 0 for v in s))
                for s in iter_top_simplices(HC, 2)}
        assert labels['base'].keys() == labels['arm'].keys()
        for key, (k0, touches) in labels['base'].items():
            k1, _ = labels['arm'][key]
            if not touches:
                assert k0 == k1
        # ... and the arm must differ where the interface is touched (the
        # (liquid, air, interface) ties of the column face go to the
        # liquid the preloaded density tips, laneV log section 2.2);
        # without this the forwarding could be dropped unnoticed (review
        # 2026-10-10).
        assert any(k0 != labels['arm'][key][0]
                   for key, (k0, touches) in labels['base'].items()
                   if touches)
        m = PRESETS['dam_break_2D'].replace(split_method='simplex',
                                            simplex_vote='mass_fraction')
        HC, bV, mps, bc_set, dudt_fn, _r, params = build(m)
        dt = cfl_timestep(HC, 2, float(np.sqrt(p.K_l / p.rho_l)), cfl=p.cfl)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=20, bc_set=bc_set,
                        mps=mps)
        assert all(np.all(np.isfinite(v.u)) for v in HC.V)
        assert max(float(np.linalg.norm(v.u[:2])) for v in HC.V) < 1.0

    def test_vote_needs_a_retopology_that_applies_it(self):
        for conn in ('frozen', 'dual_only_bare'):
            with pytest.raises(ValueError, match='simplex_vote'):
                SolverMethods(dim=2, phases='multi', connectivity=conn,
                              simplex_vote='mass_fraction')


class TestForwarding:
    """The two axes reach the mesh through the multiphase retopology
    itself (``_retopologize_multiphase`` -> ``multiphase_rebuild_with_
    ledger`` -> ``mps.refresh`` / ``_retopologize``), not only through
    the partial's keywords (laneV review 2026-10-10: the wiring tests
    above pass when either function drops the keyword)."""

    @pytest.mark.parametrize('vote, tie_label', [('bulk_majority', 0),
                                                 ('mass_fraction', 1)])
    def test_vote_reaches_the_simplices(self, vote, tie_label):
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        HC = _strip(0.6, lone=True)
        bV = {v for v in HC.V if v.boundary}
        mps = _mps()
        mps.assign_simplex_phases_from_vertices(HC, 2)
        ties = [s for s in iter_top_simplices(HC, 2)
                if sorted(int(v.phase) for v in s) == [-1, 0, 1]]
        assert ties
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _retopologize_multiphase(HC, bV, 2, mps=mps,
                                     skip_triangulation=True,
                                     simplex_vote=vote)
        assert all(mps.simplex_phase[_simplex_key(s)] == tie_label
                   for s in ties)

    @pytest.mark.parametrize('method, conserved', [('merge_all', False),
                                                   ('mass_conserving', True)])
    def test_merge_method_reaches_the_rebuild(self, method, conserved):
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        HC = _strip(0.5)
        bV = {v for v in HC.V if v.boundary}
        mps = _mps()
        mps.assign_simplex_phases_from_vertices(HC, 2)
        air = [v for v in HC.V if v not in bV and v.phase == 0]
        HC.V.move(air[0], tuple(np.asarray(air[1].x) + 1e-12))
        mp0 = np.sum([v.m_phase for v in HC.V], axis=0)
        n0 = sum(1 for _ in HC.V)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _retopologize_multiphase(HC, bV, 2, mps=mps, merge_cdist=1e-9,
                                     merge_method=method)
        assert sum(1 for _ in HC.V) == n0 - 1
        mp1 = np.sum([v.m_phase for v in HC.V], axis=0)
        if conserved:
            np.testing.assert_allclose(mp1, mp0, rtol=1e-14)
        else:
            assert mp1[0] < mp0[0] - 0.5      # the merged air mass is gone
