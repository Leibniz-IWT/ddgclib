"""Fast tests for the two-fluid oscillating-droplet reference (lane C).

Covers the NEW analytical functions in
``cases_dynamic/oscillating_droplet/src/_analytical.py`` (the pinned
single-fluid functions are exercised by ``test_case_oscillating_droplet``):

- ``lamb_damping_rate_two_fluid`` closed form + limiting cases
  (``rho_outer -> 0`` and ``mu_outer -> 0`` must recover the
  single-fluid Lamb value),
- ``mode_temporal_ivp`` started-from-rest exactness in all regimes,
- ``two_fluid_dispersion_roots_2d`` / ``two_fluid_omega_beta_2d``
  limiting cases (inviscid, vanishing outer phase) and the pinned
  case-parameter root,
- ``add_two_fluid_reference`` metric attachment (pinned fields
  untouched).

This file is deliberately separate from
``test_case_oscillating_droplet.py`` so the pinned floor battery
(``pytest ddgclib/tests/test_case_oscillating_droplet.py -m ""``)
keeps its exact test count.
"""
import math
import unittest

import numpy as np

from cases_dynamic.oscillating_droplet.src._analytical import (
    lamb_damping_rate,
    lamb_damping_rate_two_fluid,
    mode_temporal_ivp,
    radius_perturbation,
    radius_perturbation_two_fluid,
    rayleigh_frequency,
    two_fluid_dispersion_roots_2d,
    two_fluid_omega_beta_2d,
)

# Case parameters (overdamped set in src/_params.py)
L = 2
GAMMA = 0.05
R0 = 0.01
RHO_D, MU_D = 800.0, 0.5
RHO_O, MU_O = 1000.0, 0.1


class TestLambDampingRateTwoFluid(unittest.TestCase):
    """Closed-form (energy-method) two-fluid damping rate."""

    def test_limit_recovers_single_fluid(self):
        """mu_outer -> 0 and rho_outer -> 0 recovers lamb_damping_rate."""
        for l in (2, 3, 4):
            for mu, rho, R in ((0.5, 800.0, 0.01), (1e-3, 1000.0, 1e-3)):
                b2f = lamb_damping_rate_two_fluid(
                    l, mu, rho, R, mu_outer=0.0, rho_outer=0.0)
                b1f = lamb_damping_rate(l, mu, rho, R, dim=2)
                self.assertTrue(math.isclose(b2f, b1f, rel_tol=1e-14),
                                f"l={l}: {b2f} != {b1f}")

    def test_target_value(self):
        """beta_2f = 2l[(l-1)mu_d + (l+1)mu_o]/((rho_d+rho_o)R0^2)."""
        b = lamb_damping_rate_two_fluid(
            L, MU_D, RHO_D, R0, mu_outer=MU_O, rho_outer=RHO_O)
        # 4 * (1*0.5 + 3*0.1) / (1800 * 1e-4) = 3.2 / 0.18
        self.assertTrue(math.isclose(b, 3.2 / 0.18, rel_tol=1e-14))

    def test_monotone_in_mu_outer(self):
        b0 = lamb_damping_rate_two_fluid(
            L, MU_D, RHO_D, R0, mu_outer=0.0, rho_outer=RHO_O)
        b1 = lamb_damping_rate_two_fluid(
            L, MU_D, RHO_D, R0, mu_outer=MU_O, rho_outer=RHO_O)
        self.assertGreater(b1, b0)

    def test_dim3_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            lamb_damping_rate_two_fluid(
                L, MU_D, RHO_D, R0, mu_outer=MU_O, rho_outer=RHO_O, dim=3)


class TestModeTemporalIVP(unittest.TestCase):
    """Exact started-from-rest temporal factor."""

    REGIMES = {
        'underdamped': (10.0, 3.0),
        'overdamped': (10.0, 25.0),
        'critical': (10.0, 10.0),
    }

    def test_initial_conditions(self):
        """x(0) = 1 and x'(0) = 0 in every regime."""
        h = 1e-7
        for name, (omega, beta) in self.REGIMES.items():
            x0 = float(mode_temporal_ivp(0.0, omega, beta))
            self.assertTrue(math.isclose(x0, 1.0, rel_tol=1e-12), name)
            xp = (float(mode_temporal_ivp(h, omega, beta))
                  - float(mode_temporal_ivp(-h, omega, beta))) / (2 * h)
            self.assertLess(abs(xp), 1e-5, f"{name}: x'(0) = {xp}")

    def test_ode_residual(self):
        """x'' + 2 beta x' + omega^2 x = 0 (finite differences)."""
        h = 1e-5
        for name, (omega, beta) in self.REGIMES.items():
            for t in (0.05, 0.2, 0.5):
                xm = float(mode_temporal_ivp(t - h, omega, beta))
                x0 = float(mode_temporal_ivp(t, omega, beta))
                xp = float(mode_temporal_ivp(t + h, omega, beta))
                xpp = (xp - 2 * x0 + xm) / h ** 2
                xd = (xp - xm) / (2 * h)
                res = xpp + 2 * beta * xd + omega ** 2 * x0
                self.assertLess(abs(res), 1e-3, f"{name} t={t}: {res}")

    def test_overdamped_matches_pinned_radius_perturbation(self):
        """Overdamped branch is identical to the pinned biexponential."""
        omega, beta = 12.909944487358056, 25.0
        t = np.linspace(0.0, 0.2, 41)
        pinned = radius_perturbation(t, 0.0, 1.0, 1.0, L, omega, beta) - 1.0
        new = mode_temporal_ivp(t, omega, beta)
        np.testing.assert_allclose(new, pinned, rtol=1e-13, atol=1e-15)

    def test_undamped_is_cosine(self):
        t = np.linspace(0.0, 1.0, 17)
        np.testing.assert_allclose(
            mode_temporal_ivp(t, 7.0, 0.0), np.cos(7.0 * t),
            rtol=1e-12, atol=1e-14)

    def test_radius_perturbation_two_fluid_shape(self):
        """R(theta, 0) = R0 (1 + eps cos(l theta))."""
        theta = np.linspace(0.0, 2 * np.pi, 9)
        r = radius_perturbation_two_fluid(
            0.0, theta, R0, 0.05, L, 8.75, 6.83)
        np.testing.assert_allclose(
            r, R0 * (1 + 0.05 * np.cos(L * theta)), rtol=1e-12)


class TestTwoFluidDispersion(unittest.TestCase):
    """2D two-fluid viscous normal-mode dispersion relation."""

    def test_inviscid_limit(self):
        """mu -> 0 (both phases): s -> +/- i omega with rho+rho_o inertia."""
        fac = 1e-4
        roots = two_fluid_dispersion_roots_2d(
            L, GAMMA, fac * MU_D, RHO_D, R0, fac * MU_O, RHO_O)
        self.assertTrue(roots)
        s = roots[0]
        omega = rayleigh_frequency(L, GAMMA, RHO_D, R0, dim=2,
                                   rho_outer=RHO_O)
        self.assertLess(abs(abs(s.imag) - omega) / omega, 5e-3)
        self.assertLess(abs(s.real), 0.05 * omega)

    def test_single_fluid_weak_viscosity_limit(self):
        """Vanishing outer phase + weak mu: Re s -> -lamb_damping_rate."""
        mu_w = 0.005
        roots = two_fluid_dispersion_roots_2d(
            L, GAMMA, mu_w, RHO_D, R0, mu_outer=1e-5, rho_outer=1.0)
        self.assertTrue(roots)
        s = roots[0]
        beta_lamb = lamb_damping_rate(L, mu_w, RHO_D, R0, dim=2)
        omega_1f = rayleigh_frequency(L, GAMMA, RHO_D, R0, dim=2)
        self.assertLess(abs(-s.real - beta_lamb) / beta_lamb, 0.05)
        self.assertLess(abs(abs(s.imag) - omega_1f) / omega_1f, 0.01)

    def test_case_parameters_pinned_root(self):
        """Least-damped mode at the case parameters (lane-C pin)."""
        roots = two_fluid_dispersion_roots_2d(
            L, GAMMA, MU_D, RHO_D, R0, mu_outer=MU_O, rho_outer=RHO_O)
        s = roots[0]
        self.assertTrue(math.isclose(s.real, -6.8325510, rel_tol=1e-4),
                        f"Re s = {s.real}")
        self.assertTrue(math.isclose(abs(s.imag), 5.4707015, rel_tol=1e-4),
                        f"Im s = {s.imag}")

    def test_omega_beta_mapping(self):
        omega_tf, beta_tf = two_fluid_omega_beta_2d(
            L, GAMMA, MU_D, RHO_D, R0, mu_outer=MU_O, rho_outer=RHO_O)
        self.assertTrue(math.isclose(beta_tf, 6.8325510, rel_tol=1e-4))
        w_d = math.sqrt(omega_tf ** 2 - beta_tf ** 2)
        self.assertTrue(math.isclose(w_d, 5.4707015, rel_tol=1e-4))

    def test_rejects_nonpositive_outer(self):
        with self.assertRaises(ValueError):
            two_fluid_dispersion_roots_2d(
                L, GAMMA, MU_D, RHO_D, R0, mu_outer=0.0, rho_outer=RHO_O)


class TestAddTwoFluidReference(unittest.TestCase):
    """Metric attachment: pinned fields untouched, new keys correct."""

    def test_perfect_two_fluid_trajectory_scores_zero(self):
        from cases_dynamic.oscillating_droplet.src._metrics import (
            add_two_fluid_reference, oscillation_score,
        )
        epsilon = 0.05
        omega_tf, beta_tf = 8.7527, 6.8326
        omega_1f, beta_1f = 12.909944487358056, 25.0
        t = np.linspace(0.0, 0.1, 51)
        diags = [
            {
                't': float(ti),
                'r_apex': float(radius_perturbation_two_fluid(
                    ti, 0.0, R0, epsilon, L, omega_tf, beta_tf)),
                'theta_apex': 0.0,
                'R_max': R0,
                'KE': float(np.exp(-2 * beta_tf * ti)),
                'total_mass': 1.0,
            }
            for ti in t
        ]
        score = oscillation_score(
            diags, R0=R0, epsilon=epsilon, l=L,
            omega=omega_1f, beta=beta_1f)
        pinned = {k: score[k] for k in
                  ('l2_error_normalized', 'linf_error_normalized',
                   'summary', 'tail_growth', 'mass_drift')}
        score = add_two_fluid_reference(
            score, diags, R0=R0, epsilon=epsilon, l=L,
            omega_two_fluid=omega_tf, beta_two_fluid=beta_tf,
            beta_energy=3.2 / 0.18)
        # exact two-fluid trajectory -> zero error vs two-fluid reference
        self.assertLess(score['l2_error_normalized_two_fluid'], 1e-12)
        self.assertLess(score['linf_error_normalized_two_fluid'], 1e-12)
        # pinned single-fluid fields bit-unchanged
        for k, v in pinned.items():
            self.assertEqual(score[k], v, k)
        self.assertEqual(score['inputs_two_fluid']['beta'], beta_tf)
        # and the two references genuinely differ on this trajectory
        self.assertGreater(score['l2_error_normalized'], 1e-3)


if __name__ == '__main__':
    unittest.main()
