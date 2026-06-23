"""Tests for ddgclib.data._conservation diagnostics.

Covers:
- Kinetic energy, momentum, total mass, total volume on a simple 2D mesh
- Per-phase KE / mass / volume when v.m_phase / v.dual_vol_phase are set
- h_min / h_max from edge lengths
- u_max / u_min / p_max / p_min extrema
- StateHistory integration (conservation=True merges into diagnostics)
- as_jsonable conversion and drift_fractions helper
"""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd


@pytest.fixture
def mesh_2d_with_fields():
    """2D unit square mesh with u, p, m, dual_vol set on every vertex."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()

    bV = set()
    for v in HC.V:
        x, y = float(v.x_a[0]), float(v.x_a[1])
        on_boundary = (abs(x) < 1e-14 or abs(x - 1.0) < 1e-14
                       or abs(y) < 1e-14 or abs(y - 1.0) < 1e-14)
        if on_boundary:
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False
        v.u = np.array([1.0, 2.0])
        v.p = 10.0
        v.m = 1.0

    compute_vd(HC, method='barycentric')
    from ddgclib.operators.stress import cache_dual_volumes
    cache_dual_volumes(HC, dim=2)

    return HC, bV


class TestComputeConservation:
    def test_kinetic_energy_uniform(self, mesh_2d_with_fields):
        """KE = 0.5 * sum(m * |u|^2). With m=1, u=(1,2): |u|^2 = 5 -> KE = 2.5 per vertex."""
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        n = sum(1 for _ in HC.V)
        diag = compute_conservation(HC, dim=2)

        expected_ke = 0.5 * n * 1.0 * 5.0
        npt.assert_allclose(diag['ke'], expected_ke, rtol=1e-14)

    def test_momentum(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        n = sum(1 for _ in HC.V)
        diag = compute_conservation(HC, dim=2)

        npt.assert_allclose(diag['momentum'], [n * 1.0 * 1.0, n * 1.0 * 2.0],
                            rtol=1e-14)

    def test_total_mass_and_volume(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        n = sum(1 for _ in HC.V)
        diag = compute_conservation(HC, dim=2)

        npt.assert_allclose(diag['mass_total'], float(n), rtol=1e-14)
        # Volume_total must equal the sum of v.dual_vol on every vertex
        # (self-consistency; no assumption about how barycentric duals
        # partition the domain at boundaries).
        manual = sum(float(v.dual_vol) for v in HC.V)
        npt.assert_allclose(diag['volume_total'], manual, rtol=1e-14)
        assert diag['volume_total'] > 0.0

    def test_vertex_count(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        n = sum(1 for _ in HC.V)
        diag = compute_conservation(HC, dim=2)
        assert diag['n_vertices'] == n

    def test_velocity_and_pressure_extrema(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        # Make one vertex fast and one slow to probe extrema.
        vs = list(HC.V)
        vs[0].u = np.array([0.0, 0.0])
        vs[1].u = np.array([10.0, 0.0])
        vs[0].p = -5.0
        vs[1].p = 50.0

        diag = compute_conservation(HC, dim=2)
        npt.assert_allclose(diag['u_min'], 0.0, atol=1e-14)
        npt.assert_allclose(diag['u_max'], 10.0, rtol=1e-14)
        npt.assert_allclose(diag['p_min'], -5.0, rtol=1e-14)
        npt.assert_allclose(diag['p_max'], 50.0, rtol=1e-14)

    def test_edge_length_extrema(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        diag = compute_conservation(HC, dim=2)
        assert diag['h_min'] > 0.0
        assert diag['h_max'] >= diag['h_min']

    def test_dim_inferred_when_omitted(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        diag_explicit = compute_conservation(HC, dim=2)
        diag_inferred = compute_conservation(HC)
        npt.assert_allclose(diag_inferred['ke'], diag_explicit['ke'],
                            rtol=1e-14)
        assert diag_inferred['n_vertices'] == diag_explicit['n_vertices']

    def test_fallback_mass_from_dual_vol(self, mesh_2d_with_fields):
        """When v.m is missing/zero, fall back to v.dual_vol as mass proxy."""
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        for v in HC.V:
            v.m = 0.0  # force fallback

        diag = compute_conservation(HC, dim=2)
        # Fallback means mass_total == volume_total.
        npt.assert_allclose(diag['mass_total'], diag['volume_total'],
                            rtol=1e-14)


class TestPerPhaseDiagnostics:
    def test_per_phase_mass_and_volume(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        # Two-phase: half the vertices are phase 0, half phase 1.
        vs = list(HC.V)
        half = len(vs) // 2
        for i, v in enumerate(vs):
            if i < half:
                v.m_phase = np.array([v.m, 0.0])
                v.dual_vol_phase = np.array([v.dual_vol, 0.0])
            else:
                v.m_phase = np.array([0.0, v.m])
                v.dual_vol_phase = np.array([0.0, v.dual_vol])

        diag = compute_conservation(HC, dim=2)
        assert 'mass_phase' in diag
        assert 'volume_phase' in diag
        npt.assert_allclose(diag['mass_phase'].sum(), diag['mass_total'],
                            rtol=1e-14)
        # Per-phase volume sum must equal total to machine precision.
        npt.assert_allclose(diag['volume_phase'].sum(), diag['volume_total'],
                            atol=1e-12)

    def test_per_phase_ke_splits_correctly(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        vs = list(HC.V)
        half = len(vs) // 2
        for i, v in enumerate(vs):
            if i < half:
                v.m_phase = np.array([1.0, 0.0])
            else:
                v.m_phase = np.array([0.0, 1.0])

        diag = compute_conservation(HC, dim=2)
        npt.assert_allclose(diag['ke_phase'].sum(), diag['ke'], rtol=1e-14)

    def test_no_phase_fields_when_absent(self, mesh_2d_with_fields):
        from ddgclib.data import compute_conservation

        HC, _ = mesh_2d_with_fields
        diag = compute_conservation(HC, dim=2)
        assert 'mass_phase' not in diag
        assert 'volume_phase' not in diag
        assert 'ke_phase' not in diag


class TestStateHistoryIntegration:
    def test_conservation_flag_populates_diagnostics(self, mesh_2d_with_fields):
        from ddgclib.data import StateHistory

        HC, _ = mesh_2d_with_fields
        history = StateHistory(fields=['u', 'p'], conservation=True, dim=2)
        history.append(0.0, HC)

        assert history.n_snapshots == 1
        _, diag = history.query_diagnostics()[0]
        # Conservation keys should be present.
        for key in ('ke', 'momentum', 'mass_total', 'volume_total',
                    'n_vertices', 'h_min', 'h_max'):
            assert key in diag, f'missing {key}'

    def test_conservation_off_by_default(self, mesh_2d_with_fields):
        from ddgclib.data import StateHistory

        HC, _ = mesh_2d_with_fields
        history = StateHistory(fields=['u', 'p'])
        history.append(0.0, HC)
        _, diag = history.query_diagnostics()[0]
        assert 'ke' not in diag
        assert 'mass_total' not in diag

    def test_conservation_via_integrator_callback(self, mesh_2d_with_fields):
        from ddgclib.data import StateHistory
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV = mesh_2d_with_fields

        def zero_accel(v, dim=2, **kw):
            return np.zeros(dim)

        history = StateHistory(fields=['u'], conservation=True, dim=2,
                               record_every=1)
        euler_velocity_only(HC, bV, zero_accel, dt=0.01, n_steps=3, dim=2,
                            callback=history.callback)

        # Every recorded snapshot should carry conservation diagnostics;
        # and with zero acceleration, KE should be constant.
        kes = [d['ke'] for _, d in history.query_diagnostics()]
        assert len(kes) == 3
        for ke in kes:
            npt.assert_allclose(ke, kes[0], rtol=1e-12)


class TestSerializationHelpers:
    def test_as_jsonable_converts_arrays(self):
        from ddgclib.data import as_jsonable

        diag = {
            'ke': 3.14,
            'momentum': np.array([1.0, 2.0]),
            'mass_phase': np.array([0.5, 0.5]),
            'n_vertices': np.int64(7),
        }
        out = as_jsonable(diag)
        assert isinstance(out['momentum'], list)
        assert out['momentum'] == [1.0, 2.0]
        assert isinstance(out['mass_phase'], list)
        assert out['n_vertices'] == 7
        assert isinstance(out['n_vertices'], int)

    def test_as_jsonable_roundtrip_via_json(self):
        import json
        from ddgclib.data import as_jsonable

        diag = {
            'ke': 1.5,
            'momentum': np.array([0.1, 0.2, 0.3]),
        }
        s = json.dumps(as_jsonable(diag))
        loaded = json.loads(s)
        assert loaded['ke'] == 1.5
        assert loaded['momentum'] == [0.1, 0.2, 0.3]

    def test_drift_fractions_basic(self):
        from ddgclib.data import drift_fractions

        initial = {'mass_total': 10.0, 'volume_total': 1.0}
        current = {'mass_total': 10.001, 'volume_total': 1.0}

        drift = drift_fractions(initial, current)
        npt.assert_allclose(drift['mass_total'], 1e-4, rtol=1e-10)
        npt.assert_allclose(drift['volume_total'], 0.0, atol=1e-14)

    def test_drift_fractions_handles_zero_initial(self):
        from ddgclib.data import drift_fractions

        initial = {'ke': 0.0}
        current = {'ke': 1e-10}

        drift = drift_fractions(initial, current, keys=('ke',))
        # Zero-initial falls back to absolute value to avoid div-by-zero.
        npt.assert_allclose(drift['ke'], 1e-10, rtol=1e-14)
