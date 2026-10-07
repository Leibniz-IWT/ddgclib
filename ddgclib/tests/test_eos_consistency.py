"""Regression tests for EOS consistency fixes.

Covers the confirmed defects from docs_temp/audit/eos-formulas.md and
docs_temp/audit/multiphase-eos-interface.md:

1. ``TaitMurnaghan.rho_clip`` was applied only inside ``pressure()``;
   ``density()`` and ``sound_speed()`` ignored it — the round trip broke
   at the band edges and the three methods gave mutually inconsistent
   stiffness answers.  Now the clipped EOS is a coherent *saturating*
   model (all three methods clamp to the same band, idempotent round
   trip) and saturation is never silent (``clip_count`` diagnostics +
   one ``RuntimeWarning`` per instance).
2. ``MultiphaseEOS.__call__`` had no ``v.phase >= 0`` guard — the
   INTERFACE_PHASE sentinel (-1) silently wrapped to the LAST phase.
   Now interface vertices use the shared ``interface_mean_pressure``
   convention (identical to ``compute_phase_pressures``), and the
   no-per-phase-arrays fallback raises for interface vertices.
3. ``IdealGas.density`` returned negative densities for negative
   pressures.  Now floored at 0.
"""
import unittest
import warnings

import numpy as np
import pytest

from ddgclib.eos import TaitMurnaghan, IdealGas, MultiphaseEOS
from ddgclib.eos._multiphase_eos import interface_mean_pressure


# Soft droplet EOS from the shipped oscillating-droplet case
# (cases_dynamic/oscillating_droplet/src/_params.py: c_s floor 1 m/s
# engages -> K_d = 800 Pa): representable pressure window with clip
# (0.8, 1.2) is [P(0.8*rho0), P(1.2*rho0)] = [-89.196, +300.143] Pa.
def _soft_droplet_eos():
    return TaitMurnaghan(rho0=800.0, P0=0.0, K=800.0, n=7.15,
                         rho_clip=(0.8, 1.2))


class TestTaitClipConsistency(unittest.TestCase):
    """rho_clip must act identically in pressure/density/sound_speed."""

    def test_round_trip_bijection_inside_band(self):
        """density(pressure(rho)) == rho to fp precision inside the band."""
        eos = _soft_droplet_eos()
        rho_grid = np.linspace(0.8 * 800.0, 1.2 * 800.0, 201)
        rho_back = np.asarray(eos.density(eos.pressure(rho_grid)))
        np.testing.assert_allclose(rho_back, rho_grid, rtol=1e-12)
        # And the P-side round trip inside the representable window:
        p_lo = float(eos.pressure(0.8 * 800.0))
        p_hi = float(eos.pressure(1.2 * 800.0))
        p_grid = np.linspace(p_lo, p_hi, 201)
        p_back = np.asarray(eos.pressure(eos.density(p_grid)))
        np.testing.assert_allclose(p_back, p_grid, atol=1e-9)

    def test_saturation_is_idempotent(self):
        """Out-of-band states map to the band edge in BOTH directions."""
        eos = _soft_droplet_eos()
        lo, hi = 0.8 * 800.0, 1.2 * 800.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # rho -> P -> rho lands exactly on the band edge
            self.assertEqual(float(eos.density(eos.pressure(400.0))), lo)
            self.assertEqual(float(eos.density(eos.pressure(1600.0))), hi)
            # density() of an unrepresentable pressure returns the band
            # edge (previously: silent cavitation clamp rho ~ 6e-2)
            self.assertEqual(float(eos.density(-1e4)), lo)
            self.assertEqual(float(eos.density(1e5)), hi)
            # P -> rho -> P saturates at the window edge, idempotently
            p_lo = float(eos.pressure(lo))
            p_hi = float(eos.pressure(hi))
            self.assertAlmostEqual(float(eos.pressure(eos.density(-1e4))),
                                   p_lo, places=9)
            self.assertAlmostEqual(float(eos.pressure(eos.density(1e5))),
                                   p_hi, places=9)

    def test_sound_speed_consistent_with_clip(self):
        """sound_speed outside the band reports the band-edge stiffness.

        Previously sound_speed(1300) reported 3285 m/s for the default
        water EOS while the clipped pressure law was exactly flat there.
        """
        eos = TaitMurnaghan(rho0=1000.0, P0=0.0, K=2.15e9, n=7.15,
                            rho_clip=(0.9, 1.1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c_out = float(eos.sound_speed(1300.0))
            c_edge = float(eos.sound_speed(1100.0))
        self.assertEqual(c_out, c_edge)

    def test_clip_engagement_counter_and_warning(self):
        """Saturation is never silent: warn once, count every engagement."""
        eos = _soft_droplet_eos()
        self.assertEqual(eos.clip_count,
                         {'pressure': 0, 'density': 0, 'sound_speed': 0})
        with pytest.warns(RuntimeWarning, match="rho_clip engaged"):
            eos.pressure(1600.0)
        self.assertEqual(eos.clip_count['pressure'], 1)
        # Second engagement: counted, but no second warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            eos.pressure(np.array([100.0, 900.0, 2000.0]))  # 2 of 3 clip
        self.assertEqual(eos.clip_count['pressure'], 3)
        # density() and sound_speed() engagements are counted per method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eos.density(-1e4)
            eos.sound_speed(1600.0)
        self.assertEqual(eos.clip_count['density'], 1)
        self.assertEqual(eos.clip_count['sound_speed'], 1)

    def test_in_band_calls_do_not_engage(self):
        eos = _soft_droplet_eos()
        eos.pressure(800.0)
        eos.density(0.0)
        eos.sound_speed(850.0)
        self.assertEqual(sum(eos.clip_count.values()), 0)
        self.assertFalse(eos._clip_warned)

    def test_no_clip_unchanged(self):
        """rho_clip=None: no clipping, no counting, exact round trip."""
        eos = TaitMurnaghan(rho0=800.0, P0=0.0, K=800.0, n=7.15,
                            rho_clip=None)
        rho_grid = np.linspace(100.0, 2000.0, 101)
        rho_back = np.asarray(eos.density(eos.pressure(rho_grid)))
        np.testing.assert_allclose(rho_back, rho_grid, rtol=1e-10)
        self.assertEqual(sum(eos.clip_count.values()), 0)


class TestIdealGasDensityFloor(unittest.TestCase):
    def test_negative_pressure_floors_at_zero(self):
        eos = IdealGas(rho0=1.225, T=293.15, R_specific=287.058)
        self.assertEqual(float(eos.density(-1e4)), 0.0)
        self.assertEqual(float(eos.density(0.0)), 0.0)

    def test_positive_pressure_unchanged(self):
        eos = IdealGas(rho0=1.225, T=293.15, R_specific=287.058)
        P = float(eos.pressure(1.225))
        self.assertAlmostEqual(float(eos.density(P)), 1.225, places=10)


class _FakeInterfaceVertex:
    """Interface vertex with populated per-phase arrays (path A)."""

    def __init__(self, m_phase, dual_vol_phase,
                 interface_phases=(0, 1)):
        self.phase = -1  # INTERFACE_PHASE sentinel
        self.is_interface = True
        self.interface_phases = set(interface_phases)
        self.m_phase = np.asarray(m_phase, dtype=float)
        self.dual_vol_phase = np.asarray(dual_vol_phase, dtype=float)
        self.m = float(np.sum(self.m_phase))
        self.dual_vol = float(np.sum(self.dual_vol_phase))


class TestMultiphaseEOSInterface(unittest.TestCase):
    """MultiphaseEOS.__call__ must honour the INTERFACE_PHASE sentinel."""

    def setUp(self):
        self.eos0 = TaitMurnaghan(rho0=1000.0, P0=1000.0, K=1e6, n=7.15,
                                  rho_clip=None)
        self.eos1 = TaitMurnaghan(rho0=800.0, P0=1000.0, K=8e5, n=7.15,
                                  rho_clip=None)
        self.meos = MultiphaseEOS([self.eos0, self.eos1])

    def test_interface_vertex_returns_mean_convention(self):
        """No last-phase wraparound: v.p is the mean of present phases.

        This is the compute_phase_pressures convention (shared helper
        interface_mean_pressure) — previously __call__ returned
        p_phase[-1] (the LAST phase) for interface vertices.
        """
        # phase 0 compressed 1%, phase 1 compressed 2%
        v = _FakeInterfaceVertex(
            m_phase=[1010.0 * 0.001, 816.0 * 0.001],
            dual_vol_phase=[0.001, 0.001],
        )
        p = self.meos(v)
        p0 = float(self.eos0.pressure(1010.0))
        p1 = float(self.eos1.pressure(816.0))
        expected = 0.5 * (p0 + p1)
        self.assertAlmostEqual(p, expected, places=9)
        self.assertAlmostEqual(v.p, expected, places=9)
        self.assertNotAlmostEqual(p, p1, places=1)  # no [-1] wrap
        # Shared helper agrees bit-for-bit
        self.assertEqual(p, interface_mean_pressure(v, 2))
        # Representative density is the mixture density
        self.assertAlmostEqual(v.rho, v.m / v.dual_vol, places=12)

    def test_zero_gauge_pressure_included_in_mean(self):
        """A legitimate exactly-0.0 gauge pressure is NOT 'missing'."""
        eos0 = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1e6, n=7.15,
                             rho_clip=None)
        eos1 = TaitMurnaghan(rho0=800.0, P0=0.0, K=8e5, n=7.15,
                             rho_clip=None)
        meos = MultiphaseEOS([eos0, eos1])
        # phase 0 exactly at reference (gauge p == 0.0), phase 1 compressed
        v = _FakeInterfaceVertex(
            m_phase=[1000.0 * 0.001, 816.0 * 0.001],
            dual_vol_phase=[0.001, 0.001],
        )
        p = meos(v)
        p1 = float(eos1.pressure(816.0))
        self.assertAlmostEqual(v.p_phase[0], 0.0, places=12)
        self.assertAlmostEqual(p, 0.5 * p1, places=9)

    def test_one_sided_presence(self):
        """Phase with zero sub-volume is excluded from the mean."""
        v = _FakeInterfaceVertex(
            m_phase=[0.0, 816.0 * 0.001],
            dual_vol_phase=[0.0, 0.001],
        )
        p = self.meos(v)
        p1 = float(self.eos1.pressure(816.0))
        self.assertAlmostEqual(p, p1, places=9)

    def test_fallback_without_per_phase_arrays_raises(self):
        """Path B (mixture density) must refuse the sentinel phase."""

        class Bare:
            phase = -1
            is_interface = True
            m = 0.9
            dual_vol = 0.001

        with self.assertRaises(ValueError):
            self.meos(Bare())

    def test_bulk_vertex_unchanged(self):
        """Bulk dispatch (phase >= 0) is byte-compatible with before."""

        class Bulk:
            def __init__(self, phase, m, dual_vol):
                self.phase = phase
                self.m = m
                self.dual_vol = dual_vol
                self.is_interface = False

        v0 = Bulk(0, 1.0, 0.001)
        self.assertAlmostEqual(self.meos(v0),
                               float(self.eos0.pressure(1000.0)), places=9)
        v1 = Bulk(1, 0.8, 0.001)
        self.assertAlmostEqual(self.meos(v1),
                               float(self.eos1.pressure(800.0)), places=9)

    def test_matches_compute_phase_pressures_on_mesh(self):
        """meos(v) == compute_phase_pressures' v.p on a real interface."""
        from hyperct import Complex
        from hyperct.ddg import compute_vd
        from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
        from ddgclib.operators.stress import cache_dual_volumes

        HC = Complex(2, domain=[(-1.0, 1.0), (-1.0, 1.0)])
        HC.triangulate()
        for _ in range(2):
            HC.refine_all()
        bV = HC.boundary()
        for v in HC.V:
            v.boundary = v in bV
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, 2)

        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=self.eos0, mu=0.1, rho0=1000.0),
                PhaseProperties(eos=self.eos1, mu=0.5, rho0=800.0),
            ],
            gamma={(0, 1): 0.05},
        )
        mps.refresh(HC, 2, reset_mass=True,
                    criterion_fn=lambda c: 0 if c[0] < 0 else 1)

        meos = MultiphaseEOS([self.eos0, self.eos1])
        n_iface = 0
        for v in HC.V:
            p_ref = v.p  # set by compute_phase_pressures inside refresh
            p_call = meos(v)
            self.assertAlmostEqual(p_call, p_ref, places=9)
            if getattr(v, 'is_interface', False):
                n_iface += 1
        self.assertGreater(n_iface, 0)


if __name__ == '__main__':
    unittest.main()
