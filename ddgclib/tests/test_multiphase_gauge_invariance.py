"""Gauge invariance + momentum conservation of the multiphase stress force.

Regression tests for the ``p_phase[k] == 0.0`` "phase absent" sentinel
bug (docs_temp/audit/zero-gauge-pressure.md and
docs_temp/audit/multiphase-momentum.md, both CONFIRMED):

- a legitimate zero *gauge* pressure (``TaitMurnaghan`` at ``P0=0`` with
  ``rho == rho0``) was indistinguishable from "phase not present", so
  ``_phase_pressure`` substituted the neighbour-side fallback and the
  two ends of a dual face booked different face pressures — breaking
  gauge invariance and Newton's third law (net spurious momentum of
  ~70-79 % of max|F| on the perturbed 2D droplet).

Phase presence is now keyed on geometry (``dual_vol_phase``), never on
the stored pressure value.  These tests pin:

1. gauge-offset invariance:  F(p + 1000 Pa) == F(p) to machine precision
2. net momentum: |sum F| over free vertices << max|F| when all
   free-frozen faces are quiescent (pure pairwise-antisymmetry check)
3. genuinely-absent phases still fall back (unchanged behaviour)
4. the interface ``v.p`` average includes a present phase whose gauge
   pressure is exactly 0.0
"""
import unittest

import numpy as np

from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from ddgclib.operators.multiphase_stress import (
    multiphase_stress_force,
    _phase_pressure,
    _phase_present_at,
)


# ---------------------------------------------------------------------------
# Shared fixture: small 2D droplet with a compression front at gauge P0=0
# ---------------------------------------------------------------------------
#
# The perturbed oscillating-droplet fixture (case defaults: P0 = 0) with
# outer-phase mass scaled by +1 % inside r < 1.3 R0.  This reproduces the
# mid-simulation state of a real P0=0 run: a pressure front (~250 Pa)
# against a quiescent far field whose stored p_phase[0] is exactly 0.0.
# The front radius is chosen so every free vertex adjacent to a frozen
# wall vertex stays quiescent -- then all free-frozen faces carry zero
# pressure flux and |sum F| over free vertices measures ONLY the pairwise
# antisymmetry defect of free-free faces.

_CACHE: dict = {}


def _front_state():
    if 'state' in _CACHE:
        return _CACHE['state']

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = (
        setup_oscillating_droplet(
            dim=2, R0=0.01, epsilon=0.05, l=2,
            rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
            gamma=0.05, L_domain=0.05,
            refinement_outer=1, refinement_droplet=2,
        )
    )
    R0 = params['R0']

    # Compression front: +1% outer-phase mass for r < 1.3 R0
    for v in HC.V:
        r = float(np.linalg.norm(v.x_a[:2]))
        if r < 1.3 * R0 and v.dual_vol_phase[0] > 1e-30:
            v.m_phase[0] *= 1.01
            v.m = float(np.sum(v.m_phase))
    mps.compute_phase_pressures(HC)

    def _forces():
        return {
            v.x: multiphase_stress_force(v, dim=2, mps=mps, HC=HC)
            for v in HC.V if v not in bV
        }

    F_base = _forces()

    # Gauge shift: +1000 Pa on every geometrically present phase entry
    # (exactly what rebuilding the same state at P0=1000 produces —
    # TaitMurnaghan is additive in P0 and densities are P0-independent).
    for v in HC.V:
        for k in range(mps.n_phases):
            if v.dual_vol_phase[k] > 1e-30:
                v.p_phase[k] = float(v.p_phase[k]) + 1000.0
    F_shift = _forces()

    # Restore the unshifted pressures for any later consumers.
    for v in HC.V:
        for k in range(mps.n_phases):
            if v.dual_vol_phase[k] > 1e-30:
                v.p_phase[k] = float(v.p_phase[k]) - 1000.0

    _CACHE['state'] = (HC, bV, mps, params, F_base, F_shift)
    return _CACHE['state']


class TestGaugeInvariance(unittest.TestCase):
    """F must be invariant under a constant offset of all phase pressures."""

    def test_state_exercises_the_bug(self):
        """The fixture must contain present phases stored as exactly 0.0
        next to O(100 Pa) perturbed vertices — the configuration the
        old ``== 0.0`` sentinel misread."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        n_zero_present = 0
        p_max = 0.0
        for v in HC.V:
            for k in range(mps.n_phases):
                if v.dual_vol_phase[k] > 1e-30:
                    pk = float(v.p_phase[k])
                    p_max = max(p_max, abs(pk))
                    if pk == 0.0:
                        n_zero_present += 1
        self.assertGreaterEqual(n_zero_present, 1)
        self.assertGreater(p_max, 100.0)
        # and the force field is non-trivial
        f_max = max(float(np.linalg.norm(f)) for f in F_base.values())
        self.assertGreater(f_max, 0.1)

    def test_gauge_offset_invariance(self):
        """max_v |F(p+1000) - F(p)| at machine precision (was ~1.2 N)."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        dF = max(
            float(np.linalg.norm(F_shift[key] - F_base[key]))
            for key in F_base
        )
        self.assertLess(dF, 1e-10)


class TestNetMomentum(unittest.TestCase):
    """Pairwise antisymmetry: internal stresses must not create momentum."""

    def test_net_momentum_free_vertices(self):
        """|sum F| / max|F| over free vertices < 1e-10 (was ~0.70)."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        F_sum = np.sum(list(F_base.values()), axis=0)
        f_max = max(float(np.linalg.norm(f)) for f in F_base.values())
        self.assertLess(float(np.linalg.norm(F_sum)) / f_max, 1e-10)

    def test_net_momentum_gauge_shifted(self):
        """Same conservation in the shifted gauge (no exact zeros)."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        F_sum = np.sum(list(F_shift.values()), axis=0)
        f_max = max(float(np.linalg.norm(f)) for f in F_shift.values())
        self.assertLess(float(np.linalg.norm(F_sum)) / f_max, 1e-10)


class TestAbsentPhaseFallback(unittest.TestCase):
    """Genuinely absent phases (zero sub-volume) keep the old fallback."""

    def test_unit_semantics(self):
        class _V:
            pass

        v = _V()
        v.p_phase = np.array([0.0, 7.5])
        v.dual_vol_phase = np.array([1e-6, 0.0])
        # phase 0 PRESENT with legitimate 0.0 gauge pressure -> stored
        # value, NOT the fallback (this was the bug)
        self.assertEqual(_phase_pressure(v, 0, fallback=123.0), 0.0)
        self.assertTrue(_phase_present_at(v, 0))
        # phase 1 geometrically absent -> fallback (unchanged behaviour)
        self.assertEqual(_phase_pressure(v, 1, fallback=123.0), 123.0)
        self.assertFalse(_phase_present_at(v, 1))
        # gauge-shift consistency of the present-phase read
        v.p_phase[0] += 1000.0
        self.assertEqual(_phase_pressure(v, 0, fallback=1123.0), 1000.0)

        # no p_phase at all -> fallback
        w = _V()
        self.assertEqual(_phase_pressure(w, 0, fallback=42.0), 42.0)
        # index out of range -> fallback
        self.assertEqual(_phase_pressure(v, 5, fallback=42.0), 42.0)
        # p_phase without geometric info -> trust the stored value
        u = _V()
        u.p_phase = np.array([0.0, 3.0])
        self.assertEqual(_phase_pressure(u, 0, fallback=42.0), 0.0)
        self.assertEqual(_phase_pressure(u, 1, fallback=42.0), 3.0)

    def test_bulk_vertex_far_from_interface(self):
        """A bulk outer vertex has no droplet phase: fallback fires."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        R0 = params['R0']
        checked = 0
        for v in HC.V:
            r = float(np.linalg.norm(v.x_a[:2]))
            if (r > 3.0 * R0 and v.phase == 0
                    and not getattr(v, 'is_interface', False)):
                self.assertLessEqual(float(v.dual_vol_phase[1]), 1e-30)
                self.assertEqual(_phase_pressure(v, 1, fallback=999.0), 999.0)
                checked += 1
        self.assertGreater(checked, 0)


class TestInterfacePressureAverage(unittest.TestCase):
    """multiphase.compute_phase_pressures: the interface ``v.p`` average
    must include a present phase whose gauge pressure is exactly 0.0."""

    def test_average_includes_zero_gauge_phase(self):
        from hyperct import Complex
        from ddgclib.eos import TaitMurnaghan
        from ddgclib.multiphase import (
            MultiphaseSystem, PhaseProperties, INTERFACE_PHASE,
        )

        rho_o, rho_d = 1000.0, 800.0
        eos_o = TaitMurnaghan(rho0=rho_o, P0=0.0, K=25000.0, n=7.15)
        eos_d = TaitMurnaghan(rho0=rho_d, P0=0.0, K=20000.0, n=7.15)
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=eos_o, mu=0.1, rho0=rho_o),
                PhaseProperties(eos=eos_d, mu=0.5, rho0=rho_d),
            ],
            gamma={(0, 1): 0.05},
        )

        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()

        vol = 2.0 ** -20  # power of two: rho = m/vol is bitwise exact
        rho_d_eq = float(eos_d.density(5.0))  # droplet at ~Laplace 5 Pa
        for i, v in enumerate(HC.V):
            v.rho_phase = np.zeros(2)
            v.p_phase = np.zeros(2)
            if i == 0:
                # interface vertex: outer phase at EXACTLY rho0
                # -> p_phase[0] == 0.0 bitwise; droplet phase at ~5 Pa
                v.phase = INTERFACE_PHASE
                v.is_interface = True
                v.interface_phases = {0, 1}
                v.dual_vol_phase = np.array([vol, vol])
                v.m_phase = np.array([rho_o * vol, rho_d_eq * vol])
            else:
                v.phase = 0
                v.is_interface = False
                v.interface_phases = set()
                v.dual_vol_phase = np.array([vol, 0.0])
                v.m_phase = np.array([rho_o * vol, 0.0])

        mps.compute_phase_pressures(HC)

        iface = [v for v in HC.V if getattr(v, 'is_interface', False)]
        self.assertEqual(len(iface), 1)
        v = iface[0]
        self.assertEqual(float(v.p_phase[0]), 0.0)  # exact gauge zero
        self.assertGreater(float(v.p_phase[1]), 1.0)
        expected = 0.5 * (float(v.p_phase[0]) + float(v.p_phase[1]))
        # Old filter (p_phase[k] != 0.0) dropped the outer phase and
        # reported the full droplet pressure (100 % of the Laplace
        # jump too high, gauge-dependently).
        self.assertAlmostEqual(float(v.p), expected, places=12)

    def test_fixture_interface_average_consistent(self):
        """On the real fixture, v.p equals the mean over geometrically
        present phases (independent recomputation)."""
        HC, bV, mps, params, F_base, F_shift = _front_state()
        n_checked = 0
        for v in HC.V:
            if not getattr(v, 'is_interface', False):
                continue
            active = [
                float(v.p_phase[k])
                for k in v.interface_phases
                if (0 <= k < mps.n_phases
                    and v.dual_vol_phase[k] > 1e-30
                    and v.m_phase[k] > 1e-30)
            ]
            self.assertTrue(active)
            self.assertAlmostEqual(
                float(v.p), float(np.mean(active)), places=10,
            )
            n_checked += 1
        self.assertGreater(n_checked, 0)


if __name__ == '__main__':
    unittest.main()
