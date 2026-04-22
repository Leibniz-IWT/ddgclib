"""Unit tests for multiphase data model, EOS, and operators.

Tests cover:
- Phase assignment and interface identification (Phase 1)
- IdealGas and MultiphaseEOS (Phase 2)
- Multiphase stress operators (Phase 3)
- Analytical solutions (Phase 5.2)
"""
import math
import unittest

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.multiphase import INTERFACE_PHASE, MultiphaseSystem, PhaseProperties
from ddgclib.eos import TaitMurnaghan, IdealGas, MultiphaseEOS
from ddgclib.initial_conditions import (
    ZeroVelocity, PhaseAssignment, MultiphaseMass, MultiphasePressure,
)
from ddgclib.operators.stress import cache_dual_volumes


def _make_2d_rectangle(n_refine=2):
    """Helper: create a simple 2D rectangular mesh centered at origin."""
    HC = Complex(2, domain=[(-1.0, 1.0), (-1.0, 1.0)])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()
    bV = HC.boundary()
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, 2)
    return HC, bV


class TestPhaseAssignment(unittest.TestCase):
    """Tests for MultiphaseSystem.assign_phases and PhaseAssignment IC."""

    def test_two_phase_left_right(self):
        """Vertices left of x=0 get phase 0, right get phase 1."""
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000, name="left"),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800, name="right"),
            ],
        )
        mps.assign_phases(HC, lambda x: 0 if x[0] < 0 else 1)

        left = {v for v in HC.V if v.phase == 0}
        right = {v for v in HC.V if v.phase == 1}
        self.assertTrue(len(left) > 0)
        self.assertTrue(len(right) > 0)
        for v in left:
            self.assertTrue(v.x_a[0] <= 0)
        for v in right:
            self.assertTrue(v.x_a[0] >= 0)

    def test_phase_assignment_ic(self):
        """PhaseAssignment IC sets v.phase correctly."""
        HC, bV = _make_2d_rectangle()
        ic = PhaseAssignment(lambda x: 1 if np.linalg.norm(x[:2]) < 0.5 else 0)
        ic.apply(HC, bV)

        for v in HC.V:
            r = np.linalg.norm(v.x_a[:2])
            if r < 0.5:
                self.assertEqual(v.phase, 1)
            elif r > 0.5:
                self.assertEqual(v.phase, 0)


class TestInterfaceIdentification(unittest.TestCase):
    """Tests for MultiphaseSystem.identify_interface."""

    def setUp(self):
        self.HC, self.bV = _make_2d_rectangle()
        self.mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
            gamma={(0, 1): 0.05},
        )

    def test_interface_detected(self):
        """Interface vertices form a closed polyline at the phase boundary."""
        self.mps.assign_simplex_phases(
            self.HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        interface = self.mps.identify_interface_from_subcomplex(self.HC, 2)

        self.assertTrue(len(interface) > 0)
        for v in interface:
            self.assertTrue(v.is_interface)
            self.assertEqual(v.phase, INTERFACE_PHASE)
            self.assertGreaterEqual(len(v.interface_phases), 2)

    def test_no_interface_single_phase(self):
        """No interface when all top-simplices share one phase."""
        self.mps.assign_simplex_phases(self.HC, 2, criterion_fn=lambda c: 0)
        interface = self.mps.identify_interface_from_subcomplex(self.HC, 2)
        self.assertEqual(len(interface), 0)

    def test_bulk_not_interface(self):
        """Bulk vertices far from phase boundary have v.is_interface == False."""
        self.mps.assign_simplex_phases(
            self.HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        self.mps.identify_interface_from_subcomplex(self.HC, 2)

        for v in self.HC.V:
            if abs(v.x_a[0]) > 0.5:  # far from x=0
                self.assertFalse(v.is_interface)
                self.assertIn(v.phase, (0, 1))


class TestDualVolumeSplitting(unittest.TestCase):
    """Tests for MultiphaseSystem.split_dual_volumes and per-phase fields."""

    def test_bulk_volume_single_phase(self):
        """Bulk vertices have all dual volume in their own phase."""
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
        )
        mps.assign_simplex_phases(
            HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        mps.refresh(HC, dim=2)

        for v in HC.V:
            if not v.is_interface:
                self.assertAlmostEqual(
                    v.dual_vol_phase[v.phase],
                    getattr(v, 'dual_vol', 0.0), places=12)
                other = 1 - v.phase
                self.assertAlmostEqual(v.dual_vol_phase[other], 0.0, places=12)

    def test_interface_volume_both_phases(self):
        """Interface vertices have non-zero volume in multiple phases."""
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
        )
        mps.assign_simplex_phases(
            HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        mps.refresh(HC, dim=2)

        has_split = False
        for v in HC.V:
            if v.is_interface:
                if v.dual_vol_phase[0] > 1e-10 and v.dual_vol_phase[1] > 1e-10:
                    has_split = True
                    break
        self.assertTrue(has_split, "Expected interface vertices with "
                        "dual volume in both phases")

    def test_per_phase_mass(self):
        """Per-phase mass is rho_k * dual_vol_phase_k."""
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=500), mu=0.5,
                                rho0=500),
            ],
        )
        mps.assign_simplex_phases(
            HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        mps.refresh(HC, dim=2)

        for v in HC.V:
            for k in range(2):
                expected = mps.phases[k].rho0 * v.dual_vol_phase[k]
                self.assertAlmostEqual(v.m_phase[k], expected, places=10)
            self.assertAlmostEqual(v.m, sum(v.m_phase), places=10)


class TestPhaseViscosity(unittest.TestCase):
    """Tests for MultiphaseSystem.get_mu (per-phase, no harmonic mean)."""

    def test_phase_viscosity(self):
        """get_mu returns the phase's viscosity directly."""
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
        )
        self.assertAlmostEqual(mps.get_mu(0), 0.1)
        self.assertAlmostEqual(mps.get_mu(1), 0.5)


class TestIdealGas(unittest.TestCase):
    """Tests for IdealGas EOS."""

    def test_pressure(self):
        eos = IdealGas(rho0=1.225, T=293.15, R_specific=287.058)
        P = eos.pressure(1.225)
        self.assertAlmostEqual(P, 1.225 * 287.058 * 293.15, places=2)

    def test_density(self):
        eos = IdealGas(rho0=1.225, T=293.15, R_specific=287.058)
        P = eos.pressure(1.225)
        rho_back = eos.density(P)
        self.assertAlmostEqual(float(rho_back), 1.225, places=10)

    def test_sound_speed(self):
        eos = IdealGas(rho0=1.225, T=293.15, R_specific=287.058)
        c = float(eos.sound_speed(1.225))
        self.assertAlmostEqual(c, np.sqrt(287.058 * 293.15), places=5)


class TestMultiphaseEOS(unittest.TestCase):
    """Tests for MultiphaseEOS pressure model."""

    def test_bulk_vertex_pressure(self):
        """Bulk vertex uses single-phase EOS."""
        eos0 = TaitMurnaghan(rho0=1000, P0=0, K=1e6, n=7.15)
        eos1 = TaitMurnaghan(rho0=800, P0=0, K=8e5, n=7.15)
        meos = MultiphaseEOS([eos0, eos1])

        class FakeVertex:
            def __init__(self, phase, m, dual_vol):
                self.phase = phase
                self.m = m
                self.dual_vol = dual_vol
                self.is_interface = False

        v = FakeVertex(phase=0, m=1.0, dual_vol=0.001)
        p = meos(v)
        expected = float(eos0.pressure(1.0 / 0.001))
        self.assertAlmostEqual(p, expected, places=5)

    def test_updates_vertex_pressure(self):
        """MultiphaseEOS updates v.p and v.rho in place."""
        eos0 = TaitMurnaghan(rho0=1000, P0=101325, K=1e6, n=7.15)
        meos = MultiphaseEOS([eos0])

        class FakeVertex:
            def __init__(self):
                self.phase = 0
                self.m = 1.1  # rho = 1100 != rho0, so P != P0
                self.dual_vol = 0.001
                self.is_interface = False
                self.p = -999
                self.rho = -999

        v = FakeVertex()
        meos(v)
        self.assertNotAlmostEqual(v.p, -999)
        self.assertAlmostEqual(v.rho, 1100.0, places=5)


class TestMultiphaseMass(unittest.TestCase):
    """Tests for MultiphaseMass IC."""

    def test_per_phase_density(self):
        """Vertices in different phases get different densities."""
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=500), mu=0.5,
                                rho0=500),
            ],
        )
        PhaseAssignment(lambda x: 0 if x[0] < 0 else 1).apply(HC, bV)
        MultiphaseMass(mps).apply(HC, bV)

        for v in HC.V:
            if v.phase == 0 and v.dual_vol > 1e-20:
                rho = v.m / v.dual_vol
                self.assertAlmostEqual(rho, 1000, places=1)
            elif v.phase == 1 and v.dual_vol > 1e-20:
                rho = v.m / v.dual_vol
                self.assertAlmostEqual(rho, 500, places=1)


class TestAnalyticalSolution(unittest.TestCase):
    """Tests for the Lamb/Rayleigh analytical solution."""

    def test_rayleigh_frequency_3d_l2(self):
        """3D mode l=2: omega^2 = 8*gamma/(rho*R0^3)."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            rayleigh_frequency,
        )
        gamma, rho, R0 = 0.05, 800.0, 0.01
        omega = rayleigh_frequency(2, gamma, rho, R0, dim=3)
        expected = np.sqrt(8 * gamma / (rho * R0**3))
        self.assertAlmostEqual(omega, expected, places=8)

    def test_lamb_damping_3d_l2(self):
        """3D mode l=2: beta = 5*mu/(rho*R0^2)."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            lamb_damping_rate,
        )
        mu, rho, R0 = 0.5, 800.0, 0.01
        beta = lamb_damping_rate(2, mu, rho, R0, dim=3)
        expected = 5 * mu / (rho * R0**2)
        self.assertAlmostEqual(beta, expected, places=8)

    def test_pressure_jump_2d(self):
        """2D Young-Laplace: ΔP = γ/R."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            pressure_jump_analytical,
        )
        dp = pressure_jump_analytical(0.05, 0.01, dim=2)
        self.assertAlmostEqual(dp, 5.0, places=10)

    def test_pressure_jump_3d(self):
        """3D Young-Laplace: ΔP = 2γ/R."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            pressure_jump_analytical,
        )
        dp = pressure_jump_analytical(0.05, 0.01, dim=3)
        self.assertAlmostEqual(dp, 10.0, places=10)

    def test_overdamped_no_oscillation(self):
        """Overdamped case: damped frequency is 0."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            damped_frequency,
        )
        omega_d = damped_frequency(omega=100, beta=200)
        self.assertEqual(omega_d, 0.0)

    def test_underdamped_frequency(self):
        """Underdamped: omega_d = sqrt(omega^2 - beta^2)."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            damped_frequency,
        )
        omega_d = damped_frequency(omega=200, beta=100)
        expected = np.sqrt(200**2 - 100**2)
        self.assertAlmostEqual(omega_d, expected, places=8)

    def test_radius_perturbation_t0(self):
        """At t=0, R(theta=0) = R0*(1+epsilon)."""
        from cases_dynamic.oscillating_droplet.src._analytical import (
            radius_perturbation,
        )
        R0, eps = 0.01, 0.05
        R = radius_perturbation(0.0, 0.0, R0, eps, l=2, omega=250, beta=50)
        self.assertAlmostEqual(float(R), R0 * (1 + eps), places=10)


class TestSurfaceTensionGamma(unittest.TestCase):
    """Tests for MultiphaseSystem.get_gamma."""

    def test_same_phase_zero(self):
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
            gamma={(0, 1): 0.05},
        )

        class FakeVertex:
            def __init__(self, phase):
                self.phase = phase

        self.assertEqual(mps.get_gamma(FakeVertex(0), FakeVertex(0)), 0.0)

    def test_cross_phase_gamma(self):
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1,
                                rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5,
                                rho0=800),
            ],
            gamma={(0, 1): 0.05},
        )

        class FakeVertex:
            def __init__(self, phase):
                self.phase = phase

        self.assertAlmostEqual(
            mps.get_gamma(FakeVertex(0), FakeVertex(1)), 0.05)
        # Order shouldn't matter
        self.assertAlmostEqual(
            mps.get_gamma(FakeVertex(1), FakeVertex(0)), 0.05)


if __name__ == '__main__':
    unittest.main()
