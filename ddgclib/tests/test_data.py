"""Tests for ddgclib.data package (save/load, history)."""

import json
import os

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_1d():
    """1D mesh with fields."""
    HC = Complex(1, domain=[(0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()

    bV = set()
    for v in HC.V:
        if abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14:
            bV.add(v)
        v.u = np.array([v.x_a[0] * 2])
        v.P = v.x_a[0] * 100
        v.m = 1.0

    return HC, bV


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / 'test_state.json')


# ---------------------------------------------------------------------------
# Save/Load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, mesh_1d, state_path):
        from ddgclib.data import save_state
        HC, bV = mesh_1d
        result = save_state(HC, bV, t=0.5, path=state_path)
        assert os.path.exists(result)

    def test_save_format(self, mesh_1d, state_path):
        from ddgclib.data import save_state
        HC, bV = mesh_1d
        save_state(HC, bV, t=1.0, path=state_path)

        with open(state_path) as f:
            data = json.load(f)

        assert data['format'] == 'ddgclib_state_v1'
        assert data['time'] == 1.0
        assert data['dim'] == 1
        assert data['n_vertices'] == sum(1 for _ in HC.V)
        assert 'u' in data['fields']
        assert 'P' in data['fields']

    def test_round_trip_fields(self, mesh_1d, state_path):
        from ddgclib.data import save_state, load_state

        HC, bV = mesh_1d
        # Record original values
        original = {}
        for v in HC.V:
            key = tuple(float(x) for x in v.x_a)
            original[key] = {'u': v.u.copy(), 'P': float(v.P), 'm': float(v.m)}

        save_state(HC, bV, t=0.42, fields=['u', 'P', 'm'], path=state_path)
        HC2, bV2, meta = load_state(state_path)

        assert meta['time'] == 0.42
        assert meta['dim'] == 1

        # Verify field values match
        for v in HC2.V:
            key = tuple(float(x) for x in v.x_a)
            if key in original:
                npt.assert_allclose(v.u, original[key]['u'], atol=1e-10)
                npt.assert_allclose(v.P, original[key]['P'], atol=1e-10)
                npt.assert_allclose(v.m, original[key]['m'], atol=1e-10)

    def test_round_trip_boundary(self, mesh_1d, state_path):
        from ddgclib.data import save_state, load_state

        HC, bV = mesh_1d
        n_bv = len(bV)
        save_state(HC, bV, t=0.0, path=state_path)
        _, bV2, _ = load_state(state_path)

        assert len(bV2) == n_bv

    def test_round_trip_connectivity(self, mesh_1d, state_path):
        from ddgclib.data import save_state, load_state

        HC, bV = mesh_1d
        # Count total edges
        edge_count = 0
        for v in HC.V:
            edge_count += len(v.nn)
        edge_count //= 2  # undirected

        save_state(HC, bV, t=0.0, path=state_path)
        HC2, _, _ = load_state(state_path)

        edge_count2 = 0
        for v in HC2.V:
            edge_count2 += len(v.nn)
        edge_count2 //= 2

        assert edge_count2 == edge_count

    def test_extra_meta(self, mesh_1d, state_path):
        from ddgclib.data import save_state, load_state

        HC, bV = mesh_1d
        save_state(HC, bV, t=0.0, path=state_path,
                   extra_meta={'case': 'poiseuille', 'mu': 1e-3})
        _, _, meta = load_state(state_path)
        assert meta['case'] == 'poiseuille'
        assert meta['mu'] == 1e-3

    def test_subdirectory_creation(self, mesh_1d, tmp_path):
        from ddgclib.data import save_state
        HC, bV = mesh_1d
        deep_path = str(tmp_path / 'a' / 'b' / 'c' / 'state.json')
        save_state(HC, bV, t=0.0, path=deep_path)
        assert os.path.exists(deep_path)


# ---------------------------------------------------------------------------
# StateHistory tests
# ---------------------------------------------------------------------------

class TestStateHistory:
    def test_callback_records(self, mesh_1d):
        from ddgclib.data import StateHistory
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV = mesh_1d

        def zero_accel(v, dim=1, **kw):
            return np.zeros(dim)

        history = StateHistory(fields=['u', 'P'], record_every=1)
        euler_velocity_only(HC, bV, zero_accel, dt=0.01, n_steps=5, dim=1,
                            callback=history.callback)

        assert history.n_snapshots == 5
        assert len(history.times) == 5

    def test_record_every(self, mesh_1d):
        from ddgclib.data import StateHistory
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV = mesh_1d

        def zero_accel(v, dim=1, **kw):
            return np.zeros(dim)

        history = StateHistory(fields=['P'], record_every=3)
        euler_velocity_only(HC, bV, zero_accel, dt=0.01, n_steps=9, dim=1,
                            callback=history.callback)

        # Steps 0, 3, 6 -> 3 snapshots
        assert history.n_snapshots == 3

    def test_query_vertex(self, mesh_1d):
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['P'])

        # Manual snapshots
        for v in HC.V:
            v.P = 10.0
        history.append(0.0, HC)

        for v in HC.V:
            v.P = 20.0
        history.append(0.1, HC)

        # Query a vertex
        some_v = next(iter(HC.V))
        key = tuple(float(x) for x in some_v.x_a)
        times, values = history.query_vertex(key, 'P')

        assert len(times) == 2
        assert times[0] == 0.0
        assert times[1] == 0.1
        assert values[0] == 10.0
        assert values[1] == 20.0

    def test_query_field_at_time(self, mesh_1d):
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['P'])

        for v in HC.V:
            v.P = 42.0
        history.append(1.0, HC)

        result = history.query_field_at_time(1.0, 'P')
        assert len(result) > 0
        for key, val in result.items():
            assert val == 42.0

    def test_query_field_closest_time(self, mesh_1d):
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['P'])

        for v in HC.V:
            v.P = 10.0
        history.append(0.0, HC)

        for v in HC.V:
            v.P = 20.0
        history.append(1.0, HC)

        # Query at t=0.3 should find t=0.0 snapshot (closer)
        result = history.query_field_at_time(0.3, 'P')
        some_val = next(iter(result.values()))
        assert some_val == 10.0

        # Query at t=0.7 should find t=1.0 snapshot (closer)
        result = history.query_field_at_time(0.7, 'P')
        some_val = next(iter(result.values()))
        assert some_val == 20.0

    def test_clear(self, mesh_1d):
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['P'])
        history.append(0.0, HC)
        assert history.n_snapshots == 1
        history.clear()
        assert history.n_snapshots == 0

    def test_query_diagnostics(self, mesh_1d):
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['P'])
        history.append(0.0, HC, diagnostics={'dt': 0.01})
        history.append(0.01, HC, diagnostics={'dt': 0.005})

        diags = history.query_diagnostics()
        assert len(diags) == 2
        assert diags[0] == (0.0, {'dt': 0.01})
        assert diags[1] == (0.01, {'dt': 0.005})

    def test_vector_field_snapshot(self, mesh_1d):
        """Ensure array fields are deep-copied (not referencing original)."""
        from ddgclib.data import StateHistory

        HC, bV = mesh_1d
        history = StateHistory(fields=['u'])

        for v in HC.V:
            v.u = np.array([1.0])
        history.append(0.0, HC)

        # Mutate original
        for v in HC.V:
            v.u = np.array([99.0])

        some_v = next(iter(HC.V))
        key = tuple(float(x) for x in some_v.x_a)
        _, values = history.query_vertex(key, 'u')
        npt.assert_allclose(values[0], [1.0])  # Should be original, not 99


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_exports(self):
        from ddgclib.data import save_state, load_state, StateHistory
        assert callable(save_state)
        assert callable(load_state)
        assert StateHistory is not None
