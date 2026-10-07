"""Frozen vertices by wall membership, not by hull membership (lane L).

The integrators do not move the vertices in ``bV``.  With the default
``frozen_set='hull'`` every retopology rebuilds ``bV`` from the hull of
the new connectivity, so a vertex is frozen because it is on the hull.
One vertex that steps past a straight wall takes the wall vertices next
to it off the hull: they are released, integrated, and the wall
collapses (audit 2026-09-25, F10 C1; ``Hagen_Poiseuile_2D`` at step
1248).  ``frozen_set='membership'`` keeps ``bV`` persistent and leaves
only the tag ``v.boundary`` to the topology.

Evidence: docs_temp/debug_session/laneL-frozen-set-membership.md
"""
from __future__ import annotations

import os
import sys
import warnings
from functools import partial

import numpy as np
import pytest

from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize,
    _retopologize_multiphase,
)
from ddgclib.geometry.domains import box, rectangle
from ddgclib.methods import PRESETS, SolverMethods

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

TOL = 1e-12


def _wall(v) -> bool:
    return abs(v.x_a[1]) < TOL or abs(v.x_a[1] - 1.0) < TOL


def _channel(frozen_set: str):
    """Channel [0, 2] x [0, 1] after one retopology: walls y = 0 and
    y = 1 frozen (boundary_filter), inlet and outlet columns free."""
    result = rectangle(L=2.0, h=1.0, refinement=3, flow_axis=0)
    HC, bV = result.HC, set(result.bV)
    for v in HC.V:
        v.u = np.zeros(2)
        v.m = 1.0
    _retopologize(HC, bV, 2, boundary_filter=_wall, frozen_set=frozen_set)
    return HC, bV


def _step_past_bottom_wall(HC):
    """Put the interior vertex nearest (1, 0.25) just below the bottom
    wall line, as an outlet buffer vertex does in Hagen_Poiseuile_2D."""
    v = min((v for v in HC.V if not _wall(v)),
            key=lambda v: (v.x_a[0] - 1.0) ** 2 + (v.x_a[1] - 0.25) ** 2)
    HC.V.move(v, (float(v.x_a[0]), -1e-3))
    return v


def _bottom(vertices):
    return sorted(float(v.x_a[0]) for v in vertices if abs(v.x_a[1]) < TOL)


# ---------------------------------------------------------------------------
# the mechanism, on _retopologize
# ---------------------------------------------------------------------------
class TestWallReleaseMechanism:
    def test_first_call_is_the_same_under_both_values(self):
        _, hull = _channel('hull')
        _, member = _channel('membership')
        key = lambda bV: sorted(v.x for v in bV)  # noqa: E731
        assert key(hull) == key(member)
        assert all(_wall(v) for v in member) and len(member) == 18

    def test_hull_releases_the_wall_when_one_vertex_steps_past_it(self):
        """The documented failure of the default: only the two corners of
        the bottom wall stay frozen."""
        HC, bV = _channel('hull')
        assert len(_bottom(bV)) == 9
        _step_past_bottom_wall(HC)
        _retopologize(HC, bV, 2, boundary_filter=_wall, frozen_set='hull')
        assert _bottom(bV) == [0.0, 2.0]
        assert len(bV) == 2 + 9   # bottom corners + the untouched top wall

    def test_membership_keeps_the_wall(self):
        HC, bV = _channel('membership')
        before = set(bV)
        _step_past_bottom_wall(HC)
        _retopologize(HC, bV, 2, boundary_filter=_wall,
                      frozen_set='membership')
        assert bV == before
        assert len(_bottom(bV)) == 9

    def test_boundary_tag_follows_the_topology_not_the_frozen_set(self):
        HC, bV = _channel('membership')
        gone = _step_past_bottom_wall(HC)
        _retopologize(HC, bV, 2, boundary_filter=_wall,
                      frozen_set='membership')
        # the vertex outside the wall line is on the hull and free
        assert gone.boundary and gone not in bV
        # the bottom wall vertices between the corners are frozen but no
        # longer on the hull: closed dual cells, no boundary tag
        inner = [v for v in bV if abs(v.x_a[1]) < TOL
                 and 0.0 < v.x_a[0] < 2.0]
        assert len(inner) == 7
        assert not any(v.boundary for v in inner)
        assert all(v.dual_vol > 0.0 for v in inner)
        # inlet / outlet columns: tagged, not frozen
        ends = [v for v in HC.V if (abs(v.x_a[0]) < TOL
                                    or abs(v.x_a[0] - 2.0) < TOL)
                and not _wall(v)]
        assert ends and all(v.boundary and v not in bV for v in ends)

    def test_membership_never_captures_a_hull_vertex(self):
        """No boundary_filter: under 'hull' every hull vertex is frozen,
        under 'membership' only the vertices handed in."""
        result = rectangle(L=1.0, h=1.0, refinement=2)
        HC = result.HC
        walls = {v for v in result.bV if _wall(v)}
        bV = set(walls)
        _retopologize(HC, bV, 2, frozen_set='membership')
        assert bV == walls
        hull = set()
        _retopologize(HC, hull, 2, frozen_set='hull')
        assert walls < hull and len(hull) == len(result.bV)

    def test_members_can_be_added_by_a_bc_and_stay(self):
        HC, bV = _channel('membership')
        extra = next(v for v in HC.V if abs(v.x_a[1] - 0.5) < TOL
                     and abs(v.x_a[0] - 1.0) < TOL)
        bV.add(extra)
        _retopologize(HC, bV, 2, frozen_set='membership')
        assert extra in bV
        # with the wall filter the same vertex is released again
        _retopologize(HC, bV, 2, boundary_filter=_wall,
                      frozen_set='membership')
        assert extra not in bV and len(bV) == 18

    def test_deleted_members_are_pruned(self):
        HC, bV = _channel('membership')
        victim = next(v for v in bV if abs(v.x_a[0] - 1.0) < TOL
                      and abs(v.x_a[1]) < TOL)
        HC.V.remove(victim)
        _retopologize(HC, bV, 2, boundary_filter=_wall,
                      frozen_set='membership')
        assert victim not in bV and len(bV) == 17

    def test_skip_triangulation_reads_the_boundary_from_the_connectivity(self):
        """dual_only stage of the multiphase remap: under 'membership' bV
        is not the boundary, so it must not be used as the tag."""
        HC, bV = _channel('membership')
        _retopologize(HC, bV, 2, boundary_filter=_wall,
                      skip_triangulation=True, frozen_set='membership')
        assert len(bV) == 18
        tagged = {v for v in HC.V if v.boundary}
        assert bV < tagged and len(tagged) == len(bV) + 14  # + inlet, outlet
        # the default carries bV as the tag (rule carried_bV)
        HC2, bV2 = _channel('hull')
        _retopologize(HC2, bV2, 2, boundary_filter=_wall,
                      skip_triangulation=True)
        assert {v for v in HC2.V if v.boundary} == bV2

    def test_3d_wall_is_kept(self):
        out = {}
        for fs in ('hull', 'membership'):
            result = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2)
            HC, bV = result.HC, set(result.bV)
            _retopologize(HC, bV, 3, frozen_set=fs)
            n0 = len(bV)
            v = min((v for v in HC.V if v not in bV),
                    key=lambda v: float(np.sum((v.x_a - [0.5, 0.5, 0.25]) ** 2)))
            HC.V.move(v, (float(v.x_a[0]), float(v.x_a[1]), -1e-3))
            floor0 = sum(1 for w in bV if abs(w.x_a[2]) < TOL)
            _retopologize(HC, bV, 3, frozen_set=fs)
            out[fs] = (n0, len(bV),
                       floor0, sum(1 for w in bV if abs(w.x_a[2]) < TOL))
        n0, n1, floor0, floor1 = out['membership']
        assert n1 == n0 and floor1 == floor0
        n0, n1, floor0, floor1 = out['hull']
        assert floor1 < floor0   # floor vertices released

    def test_invalid_values(self):
        HC, bV = _channel('hull')
        with pytest.raises(ValueError, match='frozen_set'):
            _retopologize(HC, bV, 2, frozen_set='walls')
        with pytest.raises(ValueError, match='periodic'):
            _retopologize(HC, bV, 2, frozen_set='membership',
                          periodic_axes=[0],
                          domain_bounds=[(0.0, 2.0), (0.0, 1.0)])

    def test_adaptive_remesh_is_refused_before_anything_changes(self):
        """hyperct.remesh splits wall edges into vertices that are not
        members and collapses / smooths members that are off the hull, so
        the combination is an error, in both retopology functions."""
        kw = dict(boundary_filter=_wall, remesh_mode='adaptive',
                  remesh_kwargs=dict(L_min=0.05, L_max=0.2, max_iterations=2))
        for fn in (_retopologize, _retopologize_multiphase):
            HC, bV = _channel('membership')
            state = (sorted(v.x for v in HC.V), set(bV))
            with pytest.raises(ValueError, match='adaptive'):
                fn(HC, bV, 2, frozen_set='membership', **kw)
            assert (sorted(v.x for v in HC.V), set(bV)) == state
        # what the refusal prevents is the default's job: under 'hull' the
        # wall vertices created by the splits are on the hull and frozen
        HC, bV = _channel('hull')
        n_vertices = sum(1 for _ in HC.V)
        _retopologize(HC, bV, 2, frozen_set='hull', **kw)
        on_wall = [v for v in HC.V if _wall(v)]
        assert sum(1 for _ in HC.V) > n_vertices and len(on_wall) > 18
        assert set(on_wall) == bV
        # with the connectivity kept the remesh mode is not used
        HC, bV = _channel('membership')
        _retopologize(HC, bV, 2, frozen_set='membership',
                      skip_triangulation=True, **kw)
        assert len(bV) == 18


# ---------------------------------------------------------------------------
# through the integrator
# ---------------------------------------------------------------------------
class TestIntegratorKeepsTheWall:
    @staticmethod
    def _run(frozen_set: str):
        methods = SolverMethods(dim=2, frozen_set=frozen_set)
        HC, bV = _channel(frozen_set)
        walls = {id(v): v.x_a.copy() for v in bV}
        _step_past_bottom_wall(HC)
        methods.integrate(HC, bV, lambda v: np.array([0.0, -1.0]), dt=1e-2,
                          n_steps=5, boundary_filter=_wall)
        moved = sum(not np.array_equal(v.x_a, walls[id(v)])
                    for v in HC.V if id(v) in walls)
        return moved, len(bV)

    def test_hull_moves_wall_vertices(self):
        moved, n_frozen = self._run('hull')
        assert moved == 7 and n_frozen == 11

    def test_membership_does_not(self):
        moved, n_frozen = self._run('membership')
        assert moved == 0 and n_frozen == 18


# ---------------------------------------------------------------------------
# SolverMethods plumbing
# ---------------------------------------------------------------------------
class TestSolverMethodsAxis:
    def test_default_binds_nothing(self):
        assert SolverMethods(dim=2).frozen_set == 'hull'
        assert SolverMethods(dim=2).retopologize_fn() is None
        remap = SolverMethods(dim=2, remap='conservative',
                              redistribute_mass=True).retopologize_fn()
        assert remap.keywords == {'retopo_remap': 'conservative'}
        for name, m in PRESETS.items():
            if (m.phases == 'multi' and m.frozen_set == 'hull'
                    and m.connectivity in ('delaunay', 'dual_only')):
                fn = m.retopologize_fn(mps=object())
                assert 'frozen_set' not in fn.keywords, name
        # every pinned droplet preset stays on the default
        assert all(m.frozen_set == 'hull' for name, m in PRESETS.items()
                   if 'droplet' in name or 'hydrostatic' in name
                   or 'electrolysis' in name)

    def test_membership_single_phase_partial(self):
        fn = SolverMethods(dim=2, frozen_set='membership').retopologize_fn()
        assert fn.func is _retopologize
        assert fn.keywords == {'frozen_set': 'membership'}
        fn = SolverMethods(dim=2, frozen_set='membership',
                           remap='conservative',
                           redistribute_mass=True).retopologize_fn()
        assert fn.keywords == {'retopo_remap': 'conservative',
                               'frozen_set': 'membership'}

    def test_membership_multiphase_partial(self):
        m = PRESETS['dam_break_2D'].replace(frozen_set='membership')
        mps = object()
        fn = m.retopologize_fn(mps=mps)
        assert fn.func is _retopologize_multiphase
        assert fn.keywords['frozen_set'] == 'membership'
        assert fn.keywords['retopo_remap'] == 'conservative'

    @pytest.mark.parametrize('kw', [
        dict(dim=2, frozen_set='membership', connectivity='dual_only'),
        dict(dim=2, frozen_set='membership', connectivity='dual_only_bare'),
        dict(dim=2, frozen_set='membership', connectivity='frozen'),
        dict(dim=2, frozen_set='membership', connectivity='custom'),
        dict(dim=2, frozen_set='membership', connectivity='delaunay_material'),
        dict(dim=2, frozen_set='membership', connectivity='adaptive'),
        dict(dim=2, frozen_set='membership', connectivity='adaptive',
             phases='multi', remap='conservative', redistribute_mass=True),
        dict(dim=2, frozen_set='membership', connectivity='periodic',
             periodic_axes=(0,)),
        dict(dim=1, frozen_set='membership'),
        dict(dim=2, frozen_set='walls'),
    ])
    def test_combinations_that_would_be_ignored_raise(self, kw):
        with pytest.raises(ValueError):
            SolverMethods(**kw)

    def test_preset_and_round_trip(self):
        m = PRESETS['hagen_poiseuille_2D']
        assert m.frozen_set == 'membership'
        assert m.status_of('frozen_set') == 'opt-in'
        assert SolverMethods.from_dict(m.to_dict()) == m
        # a config recorded before the axis existed reads as 'hull'
        d = m.to_dict()
        del d['frozen_set']
        assert SolverMethods.from_dict(d).frozen_set == 'hull'


# ---------------------------------------------------------------------------
# multiphase path
# ---------------------------------------------------------------------------
class TestMultiphaseMembership:
    @staticmethod
    def _run(frozen_set: str):
        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, mps, bc_set, dudt_fn, _, params = \
                setup_oscillating_droplet(
                    dim=2, R0=0.01, epsilon=0.05, l=2, L_domain=0.05,
                    refinement_outer=2, refinement_droplet=2)
        methods = PRESETS['oscillating_droplet_2D'].replace(
            frozen_set=frozen_set)
        retopo = methods.retopologize_fn(mps=mps)
        L = 0.05
        with warnings.catch_warnings():   # EOS clip (coarse mesh, displaced cell)
            warnings.simplefilter('ignore')
            retopo(HC, bV, 2)
            walls = set(bV)
            v = min((v for v in HC.V if v not in bV),
                    key=lambda v: (v.x_a[0]) ** 2 + (v.x_a[1] + 0.6 * L) ** 2)
            HC.V.move(v, (float(v.x_a[0]), -L - 1e-4))
            retopo(HC, bV, 2)
        floor = lambda s: sum(1 for w in s if abs(w.x_a[1] + L) < TOL)  # noqa: E731
        return floor(walls), floor(bV), walls, set(bV)

    def test_hull_releases_the_floor(self):
        n0, n1, _, _ = self._run('hull')
        assert n0 > 2 and n1 < n0

    def test_membership_keeps_it_through_the_remap(self):
        n0, n1, walls, bV = self._run('membership')
        assert n1 == n0 and bV == walls


# ---------------------------------------------------------------------------
# the case: Hagen_Poiseuile_2D (shortened channel, larger time step)
# ---------------------------------------------------------------------------
class TestHagenPoiseuille2D:
    """``Hagen_Poiseuile_2D.py`` as it was before laneH, with L = 2 and
    dt = 0.05 (the shipped L = 15, dt = 0.01 collapsed at step 1248; this
    one at step 250).  The reproducer of the wall collapse is the old
    configuration: the setup ``setup_poiseuille_2d_lagrangian`` (hull
    inlet, pressure advected with the vertices) and the two-point viscous
    flux; the preset has moved on to ``viscous_flux='simplex_gradient'``
    on the buffered setup (laneH), so that axis is pinned back here."""

    N_STEPS = 300

    @classmethod
    def _run(cls, frozen_set: str):
        from cases_dynamic.Hagen_Poiseuile.src._setup import (
            setup_poiseuille_2d_lagrangian, wall_report, wall_snapshot,
        )
        HC, bV, bc_set, wall, params = setup_poiseuille_2d_lagrangian(L=2.0)
        methods = PRESETS['hagen_poiseuille_2D'].replace(
            frozen_set=frozen_set, viscous_flux='two_point', workers=None)
        start = wall_snapshot(HC, wall)
        n_on_wall = []

        def callback(step, t, HC, bV=None, diagnostics=None):
            n_on_wall.append(sum(1 for v in HC.V if wall(v)))

        methods.integrate(HC, bV, methods.dudt_fn(HC, mu=params['mu']),
                          dt=0.05, n_steps=cls.N_STEPS, bc_set=bc_set,
                          boundary_filter=wall, callback=callback)
        return wall_report(HC, bV, start), n_on_wall

    def test_hull_policy_collapses_the_wall(self):
        """The documented failure: two outlet buffer vertices drift past
        the wall lines (their frozen velocity keeps its wall-normal
        component), the walls leave the hull and are integrated."""
        report, n_on_wall = self._run('hull')
        assert report['n_wall_start'] == 10
        assert report['n_frozen'] == 2          # the two inlet corners
        assert report['n_moved'] == 8
        assert report['max_displacement'] > 1e-3
        first_drop = next(i for i in range(1, len(n_on_wall))
                          if n_on_wall[i] < n_on_wall[i - 1])
        assert 200 <= first_drop <= 299
        assert n_on_wall[-1] == 2

    def test_membership_runs_past_it_and_the_walls_do_not_move(self):
        report, n_on_wall = self._run('membership')
        assert report['n_wall_start'] == 10
        assert report['n_in_complex'] == 10 and report['n_frozen'] == 10
        assert report['n_moved'] == 0
        assert report['max_displacement'] == 0.0
        # 10 walls + the two wall-row vertices the first inlet column
        # leaves one advection step from the corners (audit C3b), never less
        assert min(n_on_wall) == 12 and n_on_wall[-1] == 12
