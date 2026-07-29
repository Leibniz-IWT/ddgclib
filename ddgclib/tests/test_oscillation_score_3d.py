"""Fast guards for the 3D oscillating-droplet score harness.

Exercises ``oscillation_score_3d`` (Tier 3B, 06 §1.6) on SYNTHETIC
diagnostic frames only — no simulation is run.  Verifies the scoring
math (R_max envelope L2/Linf, apex score, KE tail, mass drift), the
boundary-saturation artefact flag, the optional dual-volume /
interface-count bookkeeping fields, and JSON round-tripping.
"""
import math
import unittest

import numpy as np

from cases_dynamic.oscillating_droplet.src._analytical import (
    max_radius_envelope, radius_perturbation,
)
from cases_dynamic.oscillating_droplet.src._metrics import (
    load_score, oscillation_score_3d, save_score,
)

# Case parameters (overdamped 3D droplet, matching src/_params.py)
R0 = 0.01
EPSILON = 0.05
L = 2
OMEGA = 16.514456476859126   # Miller-Scriven corrected, rho_o=1000
BETA = 31.25                 # Lamb 3D: (l-1)(2l+1) mu_d / (rho_d R0^2)
L_DOMAIN = 5.0 * R0


def _synthetic_diags(n=41, t_end=0.16, R_max_fn=None, KE_fn=None,
                     mass_fn=None, with_apex=False, with_extras=False):
    """Build a list of synthetic per-frame diagnostic dicts."""
    t = np.linspace(0.0, t_end, n)
    if R_max_fn is None:
        R_max_fn = lambda ti: float(
            max_radius_envelope(ti, R0, EPSILON, OMEGA, BETA, l=L))
    if KE_fn is None:
        KE_fn = lambda ti: 1e-7 * math.exp(-2 * BETA * ti)
    if mass_fn is None:
        mass_fn = lambda ti: 1.0
    diags = []
    for ti in t:
        d = {
            't': float(ti),
            'R_max': float(R_max_fn(ti)),
            'KE': float(KE_fn(ti)),
            'total_mass': float(mass_fn(ti)),
        }
        if with_apex:
            d['r_apex'] = float(
                radius_perturbation(ti, 0.0, R0, EPSILON, L, OMEGA, BETA))
            d['theta_apex'] = 0.0
        if with_extras:
            d['total_dual_vol'] = 8.0 * L_DOMAIN ** 3
            d['n_interface'] = 98
        diags.append(d)
    return diags


class TestOscillationScore3D(unittest.TestCase):

    def test_perfect_trajectory_scores_zero(self):
        """R_max exactly on the analytical envelope -> ~0 error."""
        diags = _synthetic_diags()
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
            r_boundary=L_DOMAIN,
        )
        self.assertEqual(score['kind'], 'oscillation_3d')
        self.assertLess(score['l2_error_normalized'], 1e-12)
        self.assertLess(score['linf_error_normalized'], 1e-12)
        self.assertLess(score['tail_growth'], 1.0)
        self.assertEqual(score['mass_drift'], 0.0)
        self.assertFalse(score['boundary_saturation'])
        self.assertEqual(score['n_saturated_frames'], 0)
        self.assertLess(score['summary'], 1e-12)

    def test_known_offset_l2(self):
        """A constant +delta on R_max scores l2 = delta/(eps*R0)."""
        delta = 2e-4
        diags = _synthetic_diags(
            R_max_fn=lambda ti: float(
                max_radius_envelope(ti, R0, EPSILON, OMEGA, BETA, l=L))
            + delta,
        )
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        expected = delta / (EPSILON * R0)
        self.assertAlmostEqual(score['l2_error_normalized'], expected,
                               places=10)
        self.assertAlmostEqual(score['linf_error_normalized'], expected,
                               places=10)

    def test_boundary_saturation_flagged(self):
        """The documented artefact — R_max inflating to the mesh
        boundary (~5*R0) — must set boundary_saturation, not score
        silently."""
        diags = _synthetic_diags(
            R_max_fn=lambda ti: R0 + (L_DOMAIN - R0) * min(ti / 0.05, 1.0),
        )
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
            r_boundary=L_DOMAIN,
        )
        self.assertTrue(score['boundary_saturation'])
        self.assertGreater(score['n_saturated_frames'], 0)
        self.assertAlmostEqual(score['R_max_peak'], L_DOMAIN, places=12)

    def test_no_saturation_check_without_r_boundary(self):
        """Without r_boundary the flag stays False (no threshold)."""
        diags = _synthetic_diags(R_max_fn=lambda ti: L_DOMAIN)
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertFalse(score['boundary_saturation'])
        self.assertEqual(score['n_saturated_frames'], 0)

    def test_healthy_amplitude_not_flagged(self):
        """R_max ~ 1.05*R0 is far below 0.9*L_domain -> no flag."""
        diags = _synthetic_diags()
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
            r_boundary=L_DOMAIN,
        )
        self.assertFalse(score['boundary_saturation'])

    def test_tail_growth_penalized(self):
        """Growing KE in the tail raises tail_growth above 1 and
        feeds the summary via (tail_growth - 1)."""
        diags = _synthetic_diags(KE_fn=lambda ti: 1e-7 * (1.0 + 10.0 * ti))
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertGreater(score['tail_growth'], 1.0)
        self.assertGreaterEqual(score['summary'],
                                score['tail_growth'] - 1.0 - 1e-15)

    def test_mass_drift_detected(self):
        diags = _synthetic_diags(mass_fn=lambda ti: 1.0 + 0.01 * ti)
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertAlmostEqual(score['mass_drift'], 0.01 * 0.16, places=12)
        self.assertGreaterEqual(score['summary'], score['mass_drift'])

    def test_apex_score_present_and_exact(self):
        """Apex fields appear when r_apex/theta_apex are recorded and
        score ~0 for the exact analytical apex trajectory."""
        diags = _synthetic_diags(with_apex=True)
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertIsNotNone(score['apex_l2_error_normalized'])
        self.assertLess(score['apex_l2_error_normalized'], 1e-12)
        self.assertLess(score['apex_linf_error_normalized'], 1e-12)

    def test_apex_score_absent_without_fields(self):
        diags = _synthetic_diags(with_apex=False)
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertIsNone(score['apex_l2_error_normalized'])
        self.assertIsNone(score['apex_linf_error_normalized'])

    def test_bookkeeping_fields(self):
        """dual_vol_* and n_interface_* appear when recorded (the
        dual_only A/B verification channels)."""
        diags = _synthetic_diags(with_extras=True)
        # Inject the known frame-0 -> 1 boundary-zeroing jump pattern.
        diags[0]['total_dual_vol'] = 1.0
        for d in diags[1:]:
            d['total_dual_vol'] = 0.9
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
        )
        self.assertAlmostEqual(score['dual_vol_step0_jump'], 0.1, places=12)
        self.assertEqual(score['dual_vol_drift_post'], 0.0)
        self.assertEqual(score['n_interface_start'], 98)
        self.assertEqual(score['n_interface_end'], 98)
        self.assertEqual(score['n_interface_min'], 98)
        self.assertEqual(score['n_interface_max'], 98)

    def test_json_roundtrip(self):
        """save_score/load_score round-trips the 3D score (bool flag
        included) without loss."""
        import tempfile
        from pathlib import Path

        diags = _synthetic_diags(with_apex=True, with_extras=True)
        score = oscillation_score_3d(
            diags, R0=R0, epsilon=EPSILON, l=L, omega=OMEGA, beta=BETA,
            r_boundary=L_DOMAIN,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.json'
            save_score(path, score)
            loaded = load_score(path)
        self.assertEqual(loaded['kind'], 'oscillation_3d')
        self.assertIs(loaded['boundary_saturation'], False)
        self.assertEqual(loaded['n_frames'], score['n_frames'])
        self.assertAlmostEqual(loaded['summary'], score['summary'],
                               places=15)
        self.assertEqual(loaded['inputs']['r_boundary'], L_DOMAIN)


class TestDualOnlyRetopoPolicy3D(unittest.TestCase):
    """Regression net for the 3D 'dual_only' default retopo policy
    (laneB-3d-score-harness A/B, 2026-07-29: l2 1.52446 -> 0.24811,
    tail 0.47067 -> 0.08410 vs per-step Delaunay).

    Small refine-1/1 fixture (~1 s): frozen 1-skeleton across a short
    symplectic run under skip_triangulation, interface/boundary sets
    preserved, exact simplex volumes engaged, mass at machine
    precision — the lane-5 boundary-bookkeeping caveat checks.
    """

    def test_default_policy_is_dual_only(self):
        from cases_dynamic.oscillating_droplet.src._params import (
            retopo_policy_3d,
        )
        self.assertEqual(retopo_policy_3d, 'dual_only')

    def test_dual_only_run_bookkeeping(self):
        from functools import partial

        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        from ddgclib.dynamic_integrators import symplectic_euler
        from ddgclib.operators.stress import _use_exact_barycentric_volume

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=3, refinement_outer=1, refinement_droplet=1,
            )
        retopo_fn = partial(retopo_fn, skip_triangulation=True)

        m0 = sum(v.m for v in HC.V)
        iface0 = {v.x for v in HC.V if getattr(v, 'is_interface', False)}
        edges0 = {
            frozenset((v.x, nb.x)) for v in HC.V for nb in v.nn
        }
        self.assertGreater(len(iface0), 0)

        symplectic_euler(
            HC, bV, dudt_fn, dt=1e-5, n_steps=5, dim=3,
            bc_set=bc_set, retopologize_fn=retopo_fn,
            remesh_mode=params['remesh_mode'],
            remesh_kwargs=params['remesh_kwargs'],
        )

        # Mass at machine precision.
        m1 = sum(v.m for v in HC.V)
        self.assertLess(abs(m1 - m0) / m0, 1e-12)

        # Interface set preserved (dual_only cannot retag phases via
        # reconnection; positions move but count and identity of the
        # interface stay).
        n_iface1 = sum(
            1 for v in HC.V if getattr(v, 'is_interface', False))
        self.assertEqual(n_iface1, len(iface0))

        # Connectivity frozen: same number of undirected edges.
        edges1 = {
            frozenset((v.x, nb.x)) for v in HC.V for nb in v.nn
        }
        self.assertEqual(len(edges1), len(edges0))

        # Exact simplex volumes engaged (HC._simplices survives the
        # skip path) and interior dual volumes positive, boundary
        # zeroed (batch_e_star zeroing convention).
        self.assertTrue(_use_exact_barycentric_volume(HC))
        for v in HC.V:
            if v in bV:
                self.assertEqual(float(v.dual_vol), 0.0)
            else:
                self.assertGreater(float(v.dual_vol), 0.0)


if __name__ == '__main__':
    unittest.main()
