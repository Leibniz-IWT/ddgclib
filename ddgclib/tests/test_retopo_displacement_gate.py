"""Tests for the ``displacement_eps`` skip-retopology gate in
:mod:`ddgclib.dynamic_integrators._integrators_dynamic`.

The gate short-circuits ``_do_retopologize`` when every vertex has moved
less than ``displacement_eps`` since the last call.  Motivation: 3D
Delaunay non-uniqueness on a near-cospherical static interface cloud
reconnects ~48 cross-phase edges per step even when the mesh has not
moved (Phase 2 finding 2026-04-29).  Gating by displacement collapses
the A.5.b (retopology + u=0) residual onto the A.5.a (frozen-mesh)
floor in the static-equilibrium limit.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyperct import Complex

from ddgclib.dynamic_integrators._integrators_dynamic import (
    _displacement_gate_should_skip,
    _do_retopologize,
    _snapshot_retopo_positions,
    euler,
    symplectic_euler,
)


def _build_simple_2d_mesh():
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14
                or abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
        v.u = np.zeros(2)
        v.p = 0.0
        v.m = 1.0
    return HC, bV


def _zero_accel(v, dim=2, **kwargs):
    return np.zeros(dim)


class TestDisplacementGatePrimitive:
    """Unit tests for the gate predicate itself."""

    def test_no_snapshot_takes_snapshot_and_returns_true(self):
        HC, _ = _build_simple_2d_mesh()
        # Fresh HC: no _retopo_prev_positions attr
        assert not hasattr(HC, '_retopo_prev_positions')

        should_skip = _displacement_gate_should_skip(HC, displacement_eps=1e-6)
        assert should_skip is True
        # Snapshot has been populated
        assert hasattr(HC, '_retopo_prev_positions')
        assert len(HC._retopo_prev_positions) == sum(1 for _ in HC.V)

    def test_unchanged_positions_returns_true(self):
        HC, _ = _build_simple_2d_mesh()
        _snapshot_retopo_positions(HC)
        # Positions match snapshot exactly
        assert _displacement_gate_should_skip(HC, displacement_eps=1e-6) is True

    def test_large_displacement_returns_false(self):
        HC, _ = _build_simple_2d_mesh()
        _snapshot_retopo_positions(HC)
        # Perturb one interior vertex by more than eps
        for v in HC.V:
            if abs(v.x_a[0] - 0.5) < 1e-3 and abs(v.x_a[1] - 0.5) < 1e-3:
                HC.V.move(v, (v.x_a[0] + 0.01, v.x_a[1]))
                break
        else:
            pytest.skip("No central vertex to perturb")
        assert _displacement_gate_should_skip(HC, displacement_eps=1e-6) is False

    def test_small_displacement_returns_true(self):
        HC, _ = _build_simple_2d_mesh()
        _snapshot_retopo_positions(HC)
        # Perturb one vertex by less than eps
        for v in HC.V:
            if abs(v.x_a[0] - 0.5) < 1e-3 and abs(v.x_a[1] - 0.5) < 1e-3:
                HC.V.move(v, (v.x_a[0] + 1e-12, v.x_a[1]))
                break
        else:
            pytest.skip("No central vertex to perturb")
        assert _displacement_gate_should_skip(HC, displacement_eps=1e-6) is True

    def test_vertex_count_change_returns_false(self):
        HC, _ = _build_simple_2d_mesh()
        _snapshot_retopo_positions(HC)
        # Add a vertex — vertex-id set changes
        HC.V[(0.123, 0.456)]
        assert _displacement_gate_should_skip(HC, displacement_eps=1e-6) is False


class _CountingRetopo:
    """Custom retopologize_fn that counts calls without doing any work.

    The displacement gate sits in ``_do_retopologize`` ahead of the
    dispatch to ``retopologize_fn``, so a call to this object means
    the gate did NOT short-circuit.
    """
    def __init__(self):
        self.count = 0

    def __call__(self, HC, bV, dim):
        self.count += 1


class TestDoRetopologizeGate:
    """End-to-end gate behavior via _do_retopologize using a counting
    custom retopo_fn so the test is robust to whether batch_e_star or
    the fallback path runs."""

    def test_default_eps_none_is_backward_compatible(self):
        """Without displacement_eps, every call runs the retopo_fn."""
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter)
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter)
        assert counter.count == 2
        # Snapshot attr should NOT be set when gate is disabled
        assert not hasattr(HC, '_retopo_prev_positions')

    def test_positive_eps_first_call_short_circuits(self):
        """displacement_eps set + no snapshot → no retopo, snapshot taken."""
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        assert counter.count == 0
        assert hasattr(HC, '_retopo_prev_positions')

    def test_positive_eps_skips_when_static(self):
        """When vertices have not moved, the gate skips subsequent calls."""
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        # First call: snapshot only
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        # Second call: no movement → still skipped
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        assert counter.count == 0

    def test_positive_eps_runs_when_displacement_exceeds(self):
        """When a vertex moved more than eps, retopology runs."""
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        # Move one interior vertex by more than eps
        for v in HC.V:
            if v not in bV:
                HC.V.move(v, (v.x_a[0] + 1e-3, v.x_a[1]))
                break
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        assert counter.count == 1
        # Subsequent call with no further movement → skipped again
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=1e-6)
        assert counter.count == 1

    def test_zero_eps_disables_gate(self):
        """displacement_eps <= 0 disables the gate even when set."""
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=0.0)
        _do_retopologize(HC, bV, dim=2, retopologize_fn=counter,
                         displacement_eps=0.0)
        assert counter.count == 2


class TestIntegratorGate:
    """Gate behaviour when wired through an integrator."""

    def test_euler_with_gate_zero_velocity_skips_every_step(self):
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        t = euler(HC, bV, _zero_accel, dt=1e-4, n_steps=5, dim=2,
                  retopologize_fn=counter, displacement_eps=1e-8)
        assert abs(t - 5e-4) < 1e-15
        # Zero velocity + tight eps → no retopo calls
        assert counter.count == 0

    def test_symplectic_euler_with_gate_zero_velocity_skips_every_step(self):
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        symplectic_euler(HC, bV, _zero_accel, dt=1e-4, n_steps=5, dim=2,
                         retopologize_fn=counter, displacement_eps=1e-8)
        assert counter.count == 0

    def test_euler_without_gate_calls_retopo_each_step(self):
        HC, bV = _build_simple_2d_mesh()
        counter = _CountingRetopo()
        euler(HC, bV, _zero_accel, dt=1e-4, n_steps=3, dim=2,
              retopologize_fn=counter)
        # No gate → retopo called every step
        assert counter.count == 3
