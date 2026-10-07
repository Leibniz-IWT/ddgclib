"""laneB (2026-10-05): the droplet-in-box builders keep every outer vertex.

``droplet_in_box_2d`` / ``_3d`` build the outer box on ``[0, 2L]^dim``
and shift it by ``-L``.  Until laneB the shift was a loop of single
``HC.V.move`` calls, and every move onto a key another vertex still held
dropped that vertex: 2D refinement 3 lost 6 of 145 outer vertices, 3D
refinement 2 lost 3 of 189, the ``(L, ..., L)`` corner always among
them, and the builder's Delaunay papered over the holes with larger
cells (2D refinement 3: total dual volume 9.8828e-3 instead of 1e-2,
largest outer cell 1.30e-4 instead of 1.04e-4).  The shift is one
``HC.V.move_all`` now (``box_shift='move_all'``); the old loop stays
reachable as ``box_shift='evict'`` so the numbers pinned on those
meshes can be reproduced.  ``cases_dynamic/oscillating_droplet/
diagnose_box_shift.py`` measures both arms.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from ddgclib.geometry.domains import (
    BOX_SHIFTS, box, droplet_in_box_2d, droplet_in_box_3d, rectangle,
)
from ddgclib.geometry.domains._multiphase_droplet import _shift_outer_box

L = 0.05
R = 0.01


def _keys(HC):
    return {tuple(float(c) for c in v.x_a) for v in HC.V}


def _corners(dim):
    return [tuple(float(s * L) for s in signs)
            for signs in itertools.product((-1.0, 1.0), repeat=dim)]


def _outer(dim, refinement):
    if dim == 2:
        return rectangle(L=2 * L, h=2 * L, refinement=refinement,
                         flow_axis=0).HC
    return box(Lx=2 * L, Ly=2 * L, Lz=2 * L, refinement=refinement).HC


# ---------------------------------------------------------------------------
# the shift itself
# ---------------------------------------------------------------------------
class TestShiftOuterBox:
    @pytest.mark.parametrize('dim, refinement, built, lost', [
        (2, 1, 13, 1), (2, 2, 41, 2), (2, 3, 145, 6),
        (3, 1, 35, 2), (3, 2, 189, 3),
    ])
    def test_move_all_keeps_every_vertex_and_evict_loses_the_census(
            self, dim, refinement, built, lost):
        corner = tuple([L] * dim)
        counts = {}
        for arm in BOX_SHIFTS:
            HC = _outer(dim, refinement)
            assert len(HC.V) == built
            _shift_outer_box(HC, (-L,) * dim, arm)
            keys = _keys(HC)
            counts[arm] = len(keys)
            dangling = sum(1 for v in HC.V for nb in v.nn
                           if tuple(float(c) for c in nb.x_a) not in keys)
            if arm == 'move_all':
                assert len(keys) == built
                assert corner in keys
                assert dangling == 0
                assert all(c in keys for c in _corners(dim))
            else:
                assert len(keys) == built - lost
                assert corner not in keys
                assert dangling > 0
        assert counts['move_all'] - counts['evict'] == lost

    def test_no_collision_is_bit_identical_to_the_loop(self):
        """A shift that collides with nothing gives the same cache
        (keys AND order) in both arms."""
        states = []
        for arm in BOX_SHIFTS:
            HC = _outer(2, 2)
            _shift_outer_box(HC, (0.37 * L, -0.21 * L), arm)
            states.append([tuple(float(c) for c in v.x_a) for v in HC.V])
        assert states[0] == states[1]

    def test_bad_value_raises(self):
        HC = _outer(2, 1)
        with pytest.raises(ValueError, match='box_shift'):
            _shift_outer_box(HC, (-L, -L), 'bogus')
        assert len(HC.V) == 13


# ---------------------------------------------------------------------------
# the builders
# ---------------------------------------------------------------------------
class TestDropletInBox2D:
    def test_default_walls_hold_the_four_corners(self):
        res = droplet_in_box_2d(R=R, L=L, refinement_outer=3,
                                refinement_droplet=3)
        assert res.metadata['box_shift'] == 'move_all'
        walls = _keys(type('o', (), {'V': res.bV}))
        assert all(c in walls for c in _corners(2))
        assert len(res.bV) == 32
        assert len(res.HC.V) == 317

    def test_evict_reproduces_the_pre_laneb_mesh(self):
        res = droplet_in_box_2d(R=R, L=L, refinement_outer=3,
                                refinement_droplet=3, box_shift='evict')
        assert res.metadata['box_shift'] == 'evict'
        walls = _keys(type('o', (), {'V': res.bV}))
        assert (L, L) not in walls
        assert len(res.bV) == 31
        assert len(res.HC.V) == 311

    def test_total_dual_volume_is_the_box(self):
        vols = {}
        for arm in BOX_SHIFTS:
            res = droplet_in_box_2d(R=R, L=L, refinement_outer=2,
                                    refinement_droplet=2, box_shift=arm)
            vols[arm] = sum(float(v.dual_vol) for v in res.HC.V)
        assert vols['move_all'] == pytest.approx((2 * L) ** 2, rel=1e-12)
        # the lossy mesh is the box minus the cut-off corner region
        assert vols['evict'] == pytest.approx(9.6875e-3, rel=1e-12)

    def test_bad_value_raises(self):
        with pytest.raises(ValueError, match='box_shift'):
            droplet_in_box_2d(R=R, L=L, refinement_outer=1,
                              refinement_droplet=1, box_shift='bogus')


class TestDropletInBox3D:
    def test_default_walls_hold_the_eight_corners(self):
        res = droplet_in_box_3d(R=R, L=L, refinement_outer=1,
                                refinement_droplet=1)
        walls = _keys(type('o', (), {'V': res.bV}))
        assert all(c in walls for c in _corners(3))
        assert len(res.bV) == 26
        assert len(res.HC.V) == 95
        assert sum(float(v.dual_vol) for v in res.HC.V) == pytest.approx(
            (2 * L) ** 3, rel=1e-12)

    def test_evict_reproduces_the_pre_laneb_mesh(self):
        res = droplet_in_box_3d(R=R, L=L, refinement_outer=1,
                                refinement_droplet=1, box_shift='evict')
        walls = _keys(type('o', (), {'V': res.bV}))
        assert (L, L, L) not in walls
        assert len(res.bV) == 24
        assert len(res.HC.V) == 93


# ---------------------------------------------------------------------------
# the setups record the choice
# ---------------------------------------------------------------------------
class TestSetupsRecordBoxShift:
    def test_oscillating_droplet_setup(self):
        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        n = {}
        for arm in BOX_SHIFTS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                HC, bV, mps, bc_set, dudt_fn, _r, params = \
                    setup_oscillating_droplet(
                        dim=2, R0=R, epsilon=0.05, l=2, L_domain=L,
                        refinement_outer=1, refinement_droplet=2,
                        box_shift=arm)
            assert params['box_shift'] == arm
            n[arm] = (len(HC.V), len(bV))
        assert n == {'move_all': (69, 8), 'evict': (68, 8)}

    def test_electrolysis_offcentre_builder(self):
        from cases_dynamic.electrolysis_bubble.src._setup import (
            setup_electrolysis_bubble,
        )
        for arm in BOX_SHIFTS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                HC, bV, mps, bc_set, dudt_fn, _r, params = \
                    setup_electrolysis_bubble(
                        dim=2, refinement_outer=2, refinement_droplet=2,
                        box_shift=arm)
            assert params['box_shift'] == arm
            Ld = params['L_domain']
            walls = {tuple(float(c) for c in v.x_a) for v in bV}
            corners = [tuple(float(s * Ld) for s in signs)
                       for signs in itertools.product((-1.0, 1.0), repeat=2)]
            present = sum(c in walls for c in corners)
            assert present == (4 if arm == 'move_all' else 3)
