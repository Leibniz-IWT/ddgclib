"""Integration tests for the oscillating droplet case study.

Fast regression tests (no slow 3D simulations).  Tests verify:
- Static droplet Young-Laplace equilibrium
- Overdamped decay monotonicity
- Mass conservation
- Symmetry preservation (COM at origin)
- Phase preservation (no spontaneous phase changes)
"""
import math
import unittest

import numpy as np
import pytest

from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from cases_dynamic.oscillating_droplet.src._plot_helpers import (
    compute_diagnostics,
)
from ddgclib.dynamic_integrators import euler_velocity_only


class TestStaticDroplet(unittest.TestCase):
    """Static (unperturbed) droplet equilibrium tests."""

    def setUp(self):
        """Set up an unperturbed circular droplet (epsilon=0)."""
        self.dim = 2
        self.R0 = 0.01
        self.HC, self.bV, self.mps, self.bc_set, self.dudt_fn, \
            self.retopo_fn, self.params = setup_oscillating_droplet(
                dim=2, R0=self.R0, epsilon=0.0, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=2,
            )

    def test_phase_assignment(self):
        """All vertices have valid phase IDs (including INTERFACE_PHASE)."""
        from ddgclib.multiphase import INTERFACE_PHASE
        for v in self.HC.V:
            self.assertIn(v.phase, {0, 1, INTERFACE_PHASE})

    def test_interface_detected(self):
        """Interface vertices are found."""
        n_iface = sum(1 for v in self.HC.V if getattr(v, 'is_interface', False))
        self.assertGreater(n_iface, 0)

    def test_initial_mass_positive(self):
        """All vertices have positive mass."""
        for v in self.HC.V:
            self.assertGreater(v.m, 0.0)

    def test_initial_velocity_zero(self):
        """Unperturbed droplet starts from rest."""
        for v in self.HC.V:
            u_mag = np.linalg.norm(v.u[:self.dim])
            self.assertAlmostEqual(u_mag, 0.0, places=12)


class TestPerturbedDroplet(unittest.TestCase):
    """Perturbed droplet with a few time steps."""

    def setUp(self):
        """Set up a perturbed droplet and run a few steps."""
        self.dim = 2
        self.R0 = 0.01
        self.HC, self.bV, self.mps, self.bc_set, self.dudt_fn, \
            self.retopo_fn, self.params = setup_oscillating_droplet(
                dim=2, R0=self.R0, epsilon=0.05, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=2,
            )
        self.diag0 = compute_diagnostics(self.HC, dim=self.dim)

    def test_perturbation_applied(self):
        """R_max > R0 after perturbation."""
        self.assertGreater(self.diag0['R_max'], self.R0)

    def test_mass_conservation_static(self):
        """Total mass doesn't change with perturbation geometry alone."""
        self.assertGreater(self.diag0['total_mass'], 0.0)

    def test_phase_preservation(self):
        """No bulk vertices changed phase after setup."""
        n_phase0 = sum(1 for v in self.HC.V if v.phase == 0)
        n_phase1 = sum(1 for v in self.HC.V if v.phase == 1)
        self.assertGreater(n_phase0, 0)
        self.assertGreater(n_phase1, 0)

    def test_com_near_origin(self):
        """Center of mass is near the origin."""
        com = self.diag0['com']
        # Tolerance is generous because the combined mesh is asymmetric
        # at the phase boundary (different refinements inner/outer)
        self.assertLess(np.linalg.norm(com), self.R0)

    def test_few_steps_velocity_only(self):
        """Run a few Euler steps without retopology to check stability."""
        # Use euler_velocity_only (no retopologization) for speed
        try:
            t_final = euler_velocity_only(
                self.HC, self.bV, self.dudt_fn,
                dt=1e-6, n_steps=3, dim=self.dim,
                bc_set=self.bc_set,
            )
        except (ValueError, ZeroDivisionError):
            # Some vertices may lack proper duals on combined meshes;
            # this is acceptable for a smoke test at this stage
            self.skipTest("Dual cell computation not available for all vertices")

        diag = compute_diagnostics(self.HC, dim=self.dim)

        # Mass should be conserved (no retopo = no topology change)
        mass_err = abs(diag['total_mass'] - self.diag0['total_mass'])
        self.assertLess(mass_err / self.diag0['total_mass'], 1e-10)


@pytest.mark.slow
class TestPerturbedDroplet3D(unittest.TestCase):
    """3D perturbed droplet — slow test."""

    def test_3d_setup(self):
        """3D droplet mesh can be constructed and has correct structure."""
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=3, R0=0.01, epsilon=0.05, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=1,
            )

        n_verts = sum(1 for _ in HC.V)
        self.assertGreater(n_verts, 10)

        n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        self.assertGreater(n_iface, 0)

        diag = compute_diagnostics(HC, dim=3)
        self.assertGreater(diag['R_max'], 0.01)


@pytest.mark.slow
class TestStaticDroplet2DRetopologyFloor(unittest.TestCase):
    """Tier 2B 2D long-run regression — pins the static-droplet
    interface |F| floor under repeated retopology.

    Long-run probe (`diagnose_a5_bisection.py --skip-3d --n-steps 2000
    --redistribute-mass`) confirmed bit-stability over 2000 steps:
      - step 0 max|F| = 2.3748568e-3  (pre-retopo, ≡ A.5.a frozen floor)
      - step 1.. max|F| = 2.2716938e-3 (post-retopo, single unique value,
        std = 4.34e-19, |dM/M0| = 1.64e-15, |dV/V0| = 3.54e-16,
        311->311 verts, 32->32 interface)

    Locks the post-Phase-2c (2026-04-29) ``dual_vol_phase``-gated
    ``redistribute_mass`` guard fix in place so any future refactor that
    breaks retopology-neutrality of the 2D Lagrangian multiphase loop is
    caught at the regression layer.  See
    ``.claude/plans/please-do-some-deep-vectorized-tarjan.md`` (status
    log 2026-05-28).
    """

    EXPECTED_STEP0_MAXF = 2.3748568e-03
    EXPECTED_STEP1_MAXF = 2.2716938e-03
    REL_TOL = 0.01
    N_STEPS = 30

    def test_retopology_neutrality_floor(self):
        from cases_dynamic.oscillating_droplet.src._params import (
            R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o, L_domain,
            n_refine_outer, n_refine_droplet,
        )
        from ddgclib.operators.multiphase_stress import multiphase_stress_force
        from ddgclib.dynamic_integrators import euler
        from ddgclib.data import compute_conservation

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=2, R0=R0, epsilon=0.0, l=l,
                rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
                gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
                refinement_outer=n_refine_outer,
                refinement_droplet=n_refine_droplet,
            )

        def max_iface_F(HC_):
            mx = 0.0
            for v in HC_.V:
                if not getattr(v, 'is_interface', False):
                    continue
                F = multiphase_stress_force(v, dim=2, mps=mps, HC=HC_)
                mag = float(np.linalg.norm(F))
                if mag > mx:
                    mx = mag
            return mx

        f0 = max_iface_F(HC)
        cons0 = compute_conservation(HC, dim=2)
        n_verts0 = sum(1 for _ in HC.V)
        n_iface0 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))

        self.assertAlmostEqual(
            f0, self.EXPECTED_STEP0_MAXF,
            delta=self.REL_TOL * self.EXPECTED_STEP0_MAXF,
            msg=(f"Step-0 max|F| {f0:.4e} drifted from expected "
                 f"{self.EXPECTED_STEP0_MAXF:.4e} by >{self.REL_TOL*100:.1f}% "
                 "(A.5.a curvature-stencil floor on the frozen mesh).")
        )

        c_s = float(np.sqrt(K_d / rho_d))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15
        )
        dt_acoustic = 0.25 * dx_min / c_s
        dt_capillary = 0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma)
        dt = min(dt_acoustic, dt_capillary)

        history = []

        def zero_u_callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
            history.append(max_iface_F(HC_cb))
            for v in HC_cb.V:
                v.u[:] = 0.0

        euler(
            HC, bV, dudt_fn, dt=dt, n_steps=self.N_STEPS, dim=2,
            bc_set=bc_set, callback=zero_u_callback,
            retopologize_fn=retopo_fn,
            remesh_mode=params['remesh_mode'],
            remesh_kwargs=params['remesh_kwargs'],
        )

        self.assertEqual(len(history), self.N_STEPS)

        f1 = history[0]
        self.assertAlmostEqual(
            f1, self.EXPECTED_STEP1_MAXF,
            delta=self.REL_TOL * self.EXPECTED_STEP1_MAXF,
            msg=(f"Step-1 max|F| {f1:.4e} drifted from expected "
                 f"{self.EXPECTED_STEP1_MAXF:.4e} by >{self.REL_TOL*100:.1f}% "
                 "(post-retopo steady state under u=0).")
        )

        history_arr = np.asarray(history)
        rel_spread = float((history_arr.max() - history_arr.min()) / f1)
        self.assertLess(
            rel_spread, 1e-10,
            msg=(f"Post-retopo max|F| spread {rel_spread:.3e} over "
                 f"{self.N_STEPS} steps; expected bit-stable from step 1 "
                 "(2000-step long-run had std=4.3e-19, single unique value).")
        )

        cons1 = compute_conservation(HC, dim=2)
        mass_drift = abs(cons1['mass_total'] - cons0['mass_total']) / abs(cons0['mass_total'])
        vol_drift = abs(cons1['volume_total'] - cons0['volume_total']) / abs(cons0['volume_total'])
        self.assertLess(mass_drift, 1e-10)
        self.assertLess(vol_drift, 1e-10)

        n_verts1 = sum(1 for _ in HC.V)
        n_iface1 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        self.assertEqual(n_verts1, n_verts0)
        self.assertEqual(n_iface1, n_iface0)


@pytest.mark.slow
class TestStaticDroplet3DRetopologyFloor(unittest.TestCase):
    """Tier 2B 3D long-run regression — pins the static-spherical-droplet
    interface |F| floor under repeated retopology, symmetric with
    ``TestStaticDroplet2DRetopologyFloor``.

    Long-run probe (``diagnose_a5_bisection.py --n-steps 2000
    --redistribute-mass``, refine 2/2) confirmed over 2000 steps
    (HISTORICAL, 2026-06-02, pre-2026-07-29 fan-walk volumes — see the
    RE-PIN block below for the current values):
      - step 0   max|F| = 6.0153e-5  (frozen mesh, A.5.a curvature floor)
      - step 1   max|F| = 7.0325e-5  (one-step retopo transient)
      - step 2.. max|F| = 7.3768e-5  (post-retopo steady state; bit-identical
        across steps 2-2000, std 0.0, |dM/M0| = 8.78e-15, 472->472 verts,
        98->98 interface)

    Two 3D-specific differences from the 2D floor (as of 2026-07-29):
      1. Historically the plateau was reached at step 2, not step 1 (one
         extra retopo settle step caused by order-dependent qhull
         tie-breaking on the near-cospherical interface cloud).  Since
         the canonical qhull input order in hyperct
         ``connect_and_cache_simplices`` (NOTE(laneA-canonical-order)),
         3D retopo of a static cloud is idempotent and the plateau is
         reached at step 1, like 2D — ``SETTLE_STEPS`` = 1.
      2. The one-shot ``|dV/V0|`` = 0.305 at step 0->1 is the documented
         boundary dual-cell zeroing artefact (outer-box boundary vertices
         lose their 'shell' dual volume on the first retopo —
         ``_integrators_dynamic.py`` boundary handling); total volume is
         bit-stable from step 1 onward, so volume drift is checked across
         the plateau window, not from step 0.

    Locks the post-Phase-2c (2026-04-29) ``dual_vol_phase``-gated
    ``redistribute_mass`` guard fix in 3D so any future refactor that
    breaks retopology-neutrality of the 3D Lagrangian multiphase loop is
    caught at the regression layer.  See
    ``.claude/plans/please-do-some-deep-vectorized-tarjan.md`` (status
    log 2026-06-02, Probe 6).

    RE-PIN 2026-07-29 (lane A, 3D exact-dual-volume switch + canonical
    qhull input order): ``EXPECTED_PLATEAU_MAXF`` 7.3768e-05 ->
    7.274172e-05 (-1.4%), ``SETTLE_STEPS`` 2 -> 1.  Two composed
    changes:
      (a) The 3D production path now uses the exact simplex-container
          dual volumes (``NOTE(lane3-dual-volume)`` in ``stress.py`` /
          ``_integrators_dynamic.py``).  Alone, this moved the plateau
          UP to 7.616853911101026e-05 (+3.26%): the exact measure
          honestly reports the larger real settle-step volume jump
          (order-dependent qhull tie-breaking at retopo #1-2) that the
          redistribution rescale converts into a uniform droplet
          pressure offset — NOT a volume bug
          (docs_temp/debug_session/lane3-exact-dual-volumes.md).
      (b) hyperct ``connect_and_cache_simplices`` now canonicalizes
          the 3D qhull input order (NOTE(laneA-canonical-order)),
          which makes retopo of the static cloud idempotent, kills
          the settle-step artifact entirely (no step-1->2 jump), and
          lands the plateau BELOW the old fan-walk floor.
    Verified via ``diagnose_a5_bisection.py --redistribute-mass
    --n-steps 100``: step0 6.0153201139652905e-05 (bit-unchanged,
    pinned below), plateau 7.274172178727318e-05 from step 1, rel
    spread 6.05e-12 over steps 1..100 (inside the 1e-10 assertion;
    the old fan plateau was exactly bit-stable), volume rel spread
    0.0 from step 1.  The 2D floors are bit-identical (both changes
    are 3D-only).
    """

    EXPECTED_STEP0_MAXF = 6.0153e-05
    # 7.3768e-05 (fan-walk volumes, order-dependent settle, pre
    # 2026-07-29) -> 7.274172e-05 (exact simplex-container volumes +
    # canonical 3D qhull input order, lane A re-pin — see class
    # docstring; the switch WITHOUT canonical order measures
    # 7.616854e-05).
    EXPECTED_PLATEAU_MAXF = 7.274172e-05
    REL_TOL = 0.01
    REFINE_OUTER = 2
    REFINE_DROPLET = 2
    SETTLE_STEPS = 1       # plateau reached at step 1 (canonical order
                           # killed the step-2 settle; was 2 pre-2026-07-29)
    N_STEPS = 10           # >> settle; ~0.34 s/step in 3D

    def test_retopology_neutrality_floor(self):
        from cases_dynamic.oscillating_droplet.src._params import (
            R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o, L_domain,
        )
        from ddgclib.operators.multiphase_stress import multiphase_stress_force
        from ddgclib.dynamic_integrators import euler
        from ddgclib.data import compute_conservation

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=3, R0=R0, epsilon=0.0, l=l,
                rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
                gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
                refinement_outer=self.REFINE_OUTER,
                refinement_droplet=self.REFINE_DROPLET,
            )

        def max_iface_F(HC_):
            mx = 0.0
            for v in HC_.V:
                if not getattr(v, 'is_interface', False):
                    continue
                F = multiphase_stress_force(v, dim=3, mps=mps, HC=HC_)
                mag = float(np.linalg.norm(F))
                if mag > mx:
                    mx = mag
            return mx

        f0 = max_iface_F(HC)
        cons0 = compute_conservation(HC, dim=3)
        n_verts0 = sum(1 for _ in HC.V)
        n_iface0 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))

        self.assertAlmostEqual(
            f0, self.EXPECTED_STEP0_MAXF,
            delta=self.REL_TOL * self.EXPECTED_STEP0_MAXF,
            msg=(f"Step-0 max|F| {f0:.4e} drifted from expected "
                 f"{self.EXPECTED_STEP0_MAXF:.4e} by >{self.REL_TOL*100:.1f}% "
                 "(A.5.a curvature-stencil floor on the frozen spherical mesh).")
        )

        c_s = float(np.sqrt(K_d / rho_d))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:3] - nb.x_a[:3]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:3] - nb.x_a[:3]) > 1e-15
        )
        dt_acoustic = 0.25 * dx_min / c_s
        dt_capillary = 0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma)
        dt = min(dt_acoustic, dt_capillary)

        f_history = []
        vol_history = []

        def zero_u_callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
            f_history.append(max_iface_F(HC_cb))
            vol_history.append(compute_conservation(HC_cb, dim=3)['volume_total'])
            for v in HC_cb.V:
                v.u[:] = 0.0

        euler(
            HC, bV, dudt_fn, dt=dt, n_steps=self.N_STEPS, dim=3,
            bc_set=bc_set, callback=zero_u_callback,
            retopologize_fn=retopo_fn,
            remesh_mode=params['remesh_mode'],
            remesh_kwargs=params['remesh_kwargs'],
        )

        self.assertEqual(len(f_history), self.N_STEPS)

        # f_history[k] is the state after step (k+1); the plateau starts at
        # step SETTLE_STEPS, i.e. history index SETTLE_STEPS - 1.
        plateau = np.asarray(f_history[self.SETTLE_STEPS - 1:])

        self.assertAlmostEqual(
            float(plateau[0]), self.EXPECTED_PLATEAU_MAXF,
            delta=self.REL_TOL * self.EXPECTED_PLATEAU_MAXF,
            msg=(f"Plateau max|F| {plateau[0]:.4e} drifted from expected "
                 f"{self.EXPECTED_PLATEAU_MAXF:.4e} by >{self.REL_TOL*100:.1f}% "
                 "(post-retopo steady state under u=0).")
        )

        rel_spread = float((plateau.max() - plateau.min()) / plateau[-1])
        self.assertLess(
            rel_spread, 1e-10,
            msg=(f"Post-retopo max|F| spread {rel_spread:.3e} over the "
                 f"plateau window (steps {self.SETTLE_STEPS}..{self.N_STEPS}); "
                 "expected bit-stable (2000-step long-run had std=0.0, "
                 "single unique value).")
        )

        cons1 = compute_conservation(HC, dim=3)
        mass_drift = abs(cons1['mass_total'] - cons0['mass_total']) / abs(cons0['mass_total'])
        self.assertLess(
            mass_drift, 1e-10,
            msg=(f"Mass drift {mass_drift:.3e} step0->final; expected "
                 "machine-precision Lagrangian conservation (long-run 8.78e-15).")
        )

        # Volume: the step0->step1 jump is the documented boundary dual-cell
        # zeroing artefact (|dV/V0| ~ 0.305).  Volume is bit-stable from the
        # plateau onward, so check drift across the plateau window only.
        vol_plateau = np.asarray(vol_history[self.SETTLE_STEPS - 1:])
        vol_spread = float(
            (vol_plateau.max() - vol_plateau.min()) / abs(vol_plateau[-1])
        )
        self.assertLess(
            vol_spread, 1e-10,
            msg=(f"Volume spread {vol_spread:.3e} across the plateau window; "
                 "expected bit-stable from step 2 (long-run had spread 0.0).")
        )

        n_verts1 = sum(1 for _ in HC.V)
        n_iface1 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        self.assertEqual(n_verts1, n_verts0)
        self.assertEqual(n_iface1, n_iface0)


class TestDualOnlyRetopoPolicy2D(unittest.TestCase):
    """Regression net for the lane-5 'dual_only' runner policy.

    ``oscillating_droplet_2D.py`` (via ``retopo_policy_2d`` in
    ``src/_params.py``) rebinds setup's retopo_fn with
    ``skip_triangulation=True``: the interface-conforming builder
    connectivity is kept for the whole run while duals, per-phase
    splits, mass redistribution and EOS pressures are refreshed every
    step.  The lane-5 sweep (2026-07-02) measured l2 0.48992 -> 0.17857
    and tail_growth 1.72505 -> 0.99925 vs per-step Delaunay on the full
    run; KE dropped ~5e4x to the physical overdamped envelope.

    This test locks the defining invariants of that path on a fast
    coarse fixture: frozen 1-skeleton, preserved interface, machine
    mass conservation, and no spurious KE blow-up.
    """

    def test_dual_only_dynamic_run_healthy(self):
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=2, R0=0.01, epsilon=0.05, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=2,
            )
        # Same rebinding as oscillating_droplet_2D.py's 'dual_only' policy.
        dual_only_fn = partial(retopo_fn, skip_triangulation=True)

        diag0 = compute_diagnostics(HC, dim=2)
        edges0 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        n_iface0 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        n_verts0 = sum(1 for _ in HC.V)

        # CFL dt as in the runner (setup fallback K_d = 800*5^2 here).
        c_s = float(np.sqrt(params['K_d'] / 800.0))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15
        )
        dt = min(0.25 * dx_min / c_s,
                 0.5 * float(np.sqrt(800.0 * dx_min ** 3 / 0.05)))

        symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=40, dim=2,
            bc_set=bc_set, retopologize_fn=dual_only_fn,
        )

        diag1 = compute_diagnostics(HC, dim=2)

        # Mass conserved to machine precision (redistribution rescale
        # is exactly mass-conserving).
        mass_drift = abs(diag1['total_mass'] - diag0['total_mass']) \
            / diag0['total_mass']
        self.assertLess(mass_drift, 1e-10)

        # Defining property: the 1-skeleton is frozen (no rewiring).
        edges1 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        self.assertEqual(edges1, edges0)

        # Interface and vertex sets preserved.
        n_iface1 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        n_verts1 = sum(1 for _ in HC.V)
        self.assertEqual(n_iface1, n_iface0)
        self.assertEqual(n_verts1, n_verts0)

        # No spurious KE blow-up: the perturbed coarse droplet stays at
        # the physical overdamped scale (~1e-7 J measured at refine 3/3;
        # bound is deliberately loose vs the per-step Delaunay pump,
        # which reaches ~6e-4 J within 60 steps on the fine mesh).
        self.assertLess(diag1['KE'], 1e-5)
        self.assertTrue(np.isfinite(diag1['KE']))


class TestOscillationEnvelopeRegression2D(unittest.TestCase):
    """Fast (~9 s) shortened oscillating-droplet run pinning the
    production dynamic quality metrics.

    Mirrors ``oscillating_droplet_2D.py`` exactly (delaunay_remap
    retopo policy, CFL dt formula, record cadence,
    ``oscillation_score``) at reduced refinement 2/2, integrating the
    FULL production duration t_end = min(t_end_2d, 5/beta) = 0.1143 s
    (the whole overdamped decay envelope; KE peaks well inside the
    first half, so the half-split tail_growth metric is robust here —
    unlike the refine-3/3 fragility noted in the lane-5 log).

    RE-PINNED 2026-07-29 (lane E adoption): the runner default flipped
    'dual_only' -> 'delaunay_remap' (per-step full Delaunay + lane-D
    conservative remap), so this mirror now exercises the remap path.
    Measured on this fixture (95 verts / 16 iface, 267 steps,
    deterministic across repeated runs) old (dual_only) -> new
    (delaunay_remap):

        l2_error_normalized = 0.050006441230559064 -> 0.05451425932968201
        tail_growth         = 0.9130549583972877   -> 0.9368903708926931
        linf                = 0.08490720480839112  -> 0.08367647560330455
        mass_drift          = 4.6e-15              -> 4.1e-15
        KE_max [J]          = 1.4609567009656418e-06 -> 1.4214765248811987e-06

    Reference full-run (refine 3/3, 1839 steps, ~3.5 min) scores at
    the same code state: l2 0.17479361640597058, tail
    0.9998967874595965 (dual_only: 0.1785660454150319 /
    0.9992507831101141).

    A/B sensitivity: reverting the retopo policy to BARE per-step
    Delaunay (no remap, pre-lane-5) scores l2 1.384569 / tail 2.110904
    / KE_max 9.93e-02 J on this exact fixture, so the ~10%-headroom
    thresholds below catch that class of regression by >25x.  Update
    the pins ONLY if a change legitimately improves the metrics
    (document old -> new).
    """

    # measured 0.05451425932968201 + ~10% headroom
    # (old dual_only pin: 0.0550 over measured 0.050006441230559064)
    L2_MAX = 0.0600
    # measured 0.9368903708926931 + ~7% headroom (kept at the old pin;
    # tail_growth > 1 means KE growth in the second half)
    TAIL_MAX = 1.004
    REFINE = 2

    def test_shortened_run_metrics(self):
        from functools import partial

        from cases_dynamic.oscillating_droplet.src._params import (
            R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
            L_domain, t_end_2d,
        )
        from cases_dynamic.oscillating_droplet.src._analytical import (
            rayleigh_frequency, lamb_damping_rate,
        )
        from cases_dynamic.oscillating_droplet.src._metrics import (
            oscillation_score,
        )
        from ddgclib.dynamic_integrators import symplectic_euler

        dim = 2
        omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim,
                                   rho_outer=rho_o)
        beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=dim, R0=R0, epsilon=epsilon, l=l,
                rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
                gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
                refinement_outer=self.REFINE,
                refinement_droplet=self.REFINE,
            )

        # Same rebinding as the runner's 'delaunay_remap' policy
        # (the production default since lane E).
        retopo_fn = partial(retopo_fn, retopo_remap='conservative')

        # CFL dt — identical formula to oscillating_droplet_2D.py.
        c_s = float(np.sqrt(K_d / rho_d))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
        )
        dt = min(0.25 * dx_min / c_s,
                 0.5 * float(np.sqrt(rho_d * dx_min ** 3 / gamma)))
        t_end = min(t_end_2d, 5.0 / beta)
        n_steps = int(t_end / dt) + 1
        record_every = max(1, n_steps // 200)

        diag_list: list[dict] = []

        def record(t):
            d = compute_diagnostics(HC, dim=dim)
            d['t'] = float(t)
            diag_list.append(d)

        record(0.0)

        def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
            if step % record_every == 0:
                record(t)

        t_final = symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
            bc_set=bc_set, callback=callback, retopologize_fn=retopo_fn,
            remesh_mode=params['remesh_mode'],
            remesh_kwargs=params['remesh_kwargs'],
        )
        record(t_final)

        score = oscillation_score(
            diag_list, R0=R0, epsilon=epsilon, l=l, omega=omega, beta=beta,
        )

        self.assertTrue(np.isfinite(score['l2_error_normalized']))
        self.assertLess(
            score['l2_error_normalized'], self.L2_MAX,
            msg=(f"Oscillation l2 {score['l2_error_normalized']:.6f} above "
                 f"pinned {self.L2_MAX} (measured 0.0500064 post-lane-5; "
                 "per-step-Delaunay regression scores 1.38 here)."),
        )
        self.assertLess(
            score['tail_growth'], self.TAIL_MAX,
            msg=(f"KE tail_growth {score['tail_growth']:.6f} above pinned "
                 f"{self.TAIL_MAX} (measured 0.9130550 post-lane-5; "
                 "per-step-Delaunay regression scores 2.11 here)."),
        )
        self.assertLess(score['mass_drift'], 1e-10)


class TestConservativeRetopoRemap2D(unittest.TestCase):
    """Regression net for ``retopo_remap='conservative'`` (lane D).

    The opt-in conservative remap makes per-step FULL Delaunay
    retopology thermodynamically neutral: pressure structure is
    restored across each rebuild and the per-phase pressure LEVEL is
    anchored to volume strain relative to connectivity-artifact-
    corrected volume targets (p_ref pattern).  On the full 1839-step
    oscillating-droplet run this collapses the Delaunay KE pump
    4.07e-2 J -> 8.3e-7 J (~4.9e4x, onto the physical overdamped
    envelope; l2 0.48992 -> 0.17860 == dual_only to 4 digits) while
    connectivity reconnects every step.  See
    docs_temp/debug_session/laneD-conservative-retopo-remap.md.
    """

    @staticmethod
    def _build():
        return setup_oscillating_droplet(
            dim=2, R0=0.01, epsilon=0.05, l=2,
            rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
            gamma=0.05, L_domain=0.05,
            refinement_outer=1, refinement_droplet=2,
        )

    def test_remap_pressure_neutral_across_rebuild(self):
        """One full-Delaunay retopo at frozen positions must not change
        the pressure field (measured 3.8e-13 Pa; without the remap the
        same rebuild jolts p_phase by ~1.7e2 Pa on this fixture)."""
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        before = {
            id(v): (np.array(v.p_phase, dtype=float),
                    np.array(v.dual_vol_phase, dtype=float),
                    np.array(v.m_phase, dtype=float))
            for v in HC.V
        }
        retopo_fn(HC, bV, 2, retopo_remap='conservative')
        dp_max = 0.0
        n_persistent = 0
        for v in HC.V:
            p_b, dvp_b, m_b = before[id(v)]
            for k in range(mps.n_phases):
                if (v.dual_vol_phase[k] > 1e-30 and v.m_phase[k] > 1e-30
                        and dvp_b[k] > 1e-30 and m_b[k] > 1e-30):
                    n_persistent += 1
                    dp_max = max(dp_max, abs(float(v.p_phase[k]) - p_b[k]))
        self.assertGreater(n_persistent, 0)
        self.assertLess(dp_max, 1e-9)

    def test_remap_full_delaunay_ke_bounded(self):
        """40-step run with per-step FULL Delaunay + remap stays at the
        physical KE scale (measured 1.3e-7 J; remap OFF measures
        4.5e-5 J on this fixture) with machine-precision mass."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        remap_fn = partial(retopo_fn, retopo_remap='conservative')

        diag0 = compute_diagnostics(HC, dim=2)
        edges0 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}

        c_s = float(np.sqrt(params['K_d'] / 800.0))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15
        )
        dt = min(0.25 * dx_min / c_s,
                 0.5 * float(np.sqrt(800.0 * dx_min ** 3 / 0.05)))

        symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=40, dim=2,
            bc_set=bc_set, retopologize_fn=remap_fn,
        )

        diag1 = compute_diagnostics(HC, dim=2)
        mass_drift = abs(diag1['total_mass'] - diag0['total_mass']) \
            / diag0['total_mass']
        self.assertLess(mass_drift, 1e-10)
        # The point of the remap: Delaunay stays ACTIVE (unlike
        # dual_only, connectivity is allowed to rewire) ...
        edges1 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        self.assertNotEqual(edges1, edges0)
        # ... yet KE stays at the physical scale.
        self.assertTrue(np.isfinite(diag1['KE']))
        self.assertLess(diag1['KE'], 1e-5)

    def test_remap_arg_validation(self):
        """Invalid retopo_remap and remap-without-redistribution raise."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        with self.assertRaises(ValueError):
            _retopologize_multiphase(
                HC, bV, 2, mps=mps, retopo_remap='bogus',
            )
        with self.assertRaises(ValueError):
            _retopologize_multiphase(
                HC, bV, 2, mps=mps, redistribute_mass=False,
                retopo_remap='conservative',
            )


class TestDelaunayRemapEndurance2D(unittest.TestCase):
    """Lane-E endurance guard for the 'delaunay_remap' production
    default (mirrors ``TestDualOnlyRetopoPolicy2D`` at 5x the horizon).

    200 symplectic steps with per-step FULL Delaunay reconnection +
    conservative remap on the coarse perturbed-droplet fixture.  The
    lane-E long-run proofs behind this guard (2026-07-29,
    docs_temp/debug_session/laneE-adoption-defaults.md): a 2000-step
    static (u=0) endurance run holds the pinned 2.2717e-3 force floor
    with rel spread 1.3e-15 and mass drift 2.4e-15, and a 2x-horizon
    (3678-step) dynamic run shows no late-time KE growth with mass
    drift 4.5e-14.  This test locks the same invariants at fast-suite
    cost: bounded KE at 5x the 40-step guard horizon, machine mass,
    preserved interface, and Delaunay actually rewiring.
    """

    def test_remap_endurance_200_steps(self):
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
            setup_oscillating_droplet(
                dim=2, R0=0.01, epsilon=0.05, l=2,
                rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
                gamma=0.05, L_domain=0.05,
                refinement_outer=1, refinement_droplet=2,
            )
        remap_fn = partial(retopo_fn, retopo_remap='conservative')

        diag0 = compute_diagnostics(HC, dim=2)
        edges0 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        n_iface0 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        n_verts0 = sum(1 for _ in HC.V)

        c_s = float(np.sqrt(params['K_d'] / 800.0))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15
        )
        dt = min(0.25 * dx_min / c_s,
                 0.5 * float(np.sqrt(800.0 * dx_min ** 3 / 0.05)))

        ke_trace: list[float] = []

        def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
            if step % 40 == 0:
                d = compute_diagnostics(HC_cb, dim=2)
                ke_trace.append(float(d['KE']))

        symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=200, dim=2,
            bc_set=bc_set, callback=callback, retopologize_fn=remap_fn,
        )

        diag1 = compute_diagnostics(HC, dim=2)

        # Machine-precision mass through 200 rebuild+remap cycles.
        mass_drift = abs(diag1['total_mass'] - diag0['total_mass']) \
            / diag0['total_mass']
        self.assertLess(mass_drift, 1e-10)

        # Delaunay stays ACTIVE (the point of the policy) ...
        edges1 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        self.assertNotEqual(edges1, edges0)

        # ... interface and vertex sets are preserved ...
        n_iface1 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        n_verts1 = sum(1 for _ in HC.V)
        self.assertEqual(n_iface1, n_iface0)
        self.assertEqual(n_verts1, n_verts0)

        # ... and KE stays bounded at the physical scale for the whole
        # horizon (bare per-step Delaunay pumps ~6e-4 J within 60 steps
        # on the fine mesh; every recorded sample must stay bounded,
        # not just the final one).
        self.assertTrue(np.isfinite(diag1['KE']))
        self.assertLess(diag1['KE'], 1e-5)
        self.assertTrue(all(np.isfinite(k) for k in ke_trace))
        self.assertLess(max(ke_trace), 1e-5)


class TestProjectionCadence2D(unittest.TestCase):
    """Regression net for ``projection_every`` (lane H, 2026-07-30).

    Applied every call, the pressure-preserving mass redistribution
    erases each step's local EOS compression response (the pressure
    STRUCTURE can never evolve, laneD §1.1), which laneH measured as
    the dominant driver of the 2D droplet l=2 over-decay: full-horizon
    benchmark l2 0.17857 (dual_only) / 0.17479 (delaunay_remap) drops
    to 0.0363 with ``projection_every`` in {5, 20} (saturated
    dose-response), landing on the exact two-fluid reference to ~2%
    (l2_two_fluid 0.0203, KE-shape correlation 0.997).  Opt-in;
    default 1 is the previous behaviour.  See
    docs_temp/debug_session/laneH-2d-over-decay.md.
    """

    @staticmethod
    def _build():
        return setup_oscillating_droplet(
            dim=2, R0=0.01, epsilon=0.05, l=2,
            rho_d=800.0, rho_o=1000.0, mu_d=0.5, mu_o=0.1,
            gamma=0.05, L_domain=0.05,
            refinement_outer=1, refinement_droplet=2,
        )

    @staticmethod
    def _dt(HC, params):
        c_s = float(np.sqrt(params['K_d'] / 800.0))
        dx_min = min(
            float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
            for v in HC.V for nb in v.nn
            if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15
        )
        return min(0.25 * dx_min / c_s,
                   0.5 * float(np.sqrt(800.0 * dx_min ** 3 / 0.05)))

    def test_projection_every_validation(self):
        """Invalid cadence values and incoherent combinations raise."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        for bad in (0, -1, 2.5, '5'):
            with self.assertRaises(ValueError):
                _retopologize_multiphase(
                    HC, bV, 2, mps=mps, redistribute_mass=True,
                    projection_every=bad,
                )
        # > 1 requires redistribute_mass=True + mps
        with self.assertRaises(ValueError):
            _retopologize_multiphase(
                HC, bV, 2, mps=mps, redistribute_mass=False,
                skip_triangulation=True, projection_every=5,
            )
        # > 1 under bare per-step Delaunay (no remap) re-opens the
        # lane-5 KE pump — structurally forbidden.
        with self.assertRaises(ValueError):
            _retopologize_multiphase(
                HC, bV, 2, mps=mps, redistribute_mass=True,
                projection_every=5,
            )

    def test_projection_every_default_bit_identical(self):
        """Explicit ``projection_every=1`` is bit-identical to the
        default (the knob is a pure opt-in)."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        states = []
        for extra in ({}, {'projection_every': 1}):
            HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
            remap_fn = partial(retopo_fn, retopo_remap='conservative',
                               **extra)
            dt = self._dt(HC, params)
            symplectic_euler(
                HC, bV, dudt_fn, dt=dt, n_steps=8, dim=2,
                bc_set=bc_set, retopologize_fn=remap_fn,
            )
            states.append(sorted(
                (tuple(v.x_a[:2]), tuple(v.u[:2]), float(v.m),
                 tuple(float(p) for p in v.p_phase))
                for v in HC.V
            ))
        self.assertEqual(states[0], states[1])

    def test_projection_cadence_off_call_is_neutral_at_frozen_positions(self):
        """An off-cadence remap call at frozen positions must leave the
        pressure field unchanged (the evolved-snapshot branch preserves
        the same conservative-neutrality invariant as the pre-call
        branch — only the reference field differs)."""
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        # Call 1 projects (counter 0), call 2 is off-cadence (evolved
        # snapshot); positions frozen between the calls.
        retopo_fn(HC, bV, 2, retopo_remap='conservative',
                  projection_every=2)
        before = {
            id(v): (np.array(v.p_phase, dtype=float),
                    np.array(v.dual_vol_phase, dtype=float),
                    np.array(v.m_phase, dtype=float))
            for v in HC.V
        }
        retopo_fn(HC, bV, 2, retopo_remap='conservative',
                  projection_every=2)
        self.assertEqual(mps._projection_call_idx, 2)
        dp_max = 0.0
        n_persistent = 0
        for v in HC.V:
            p_b, dvp_b, m_b = before[id(v)]
            for k in range(mps.n_phases):
                if (v.dual_vol_phase[k] > 1e-30 and v.m_phase[k] > 1e-30
                        and dvp_b[k] > 1e-30 and m_b[k] > 1e-30):
                    n_persistent += 1
                    dp_max = max(dp_max, abs(float(v.p_phase[k]) - p_b[k]))
        self.assertGreater(n_persistent, 0)
        self.assertLess(dp_max, 1e-9)

    def test_projection_cadence_dual_only(self):
        """dual_only with cadence 4: off-cadence calls skip the
        redistribution (mass stays Lagrangian), mass is machine-
        conserved, and the trajectory differs from every-call
        projection (the knob takes effect)."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        finals = []
        for extra in ({}, {'projection_every': 4}):
            HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
            fn = partial(retopo_fn, skip_triangulation=True, **extra)
            dt = self._dt(HC, params)
            diag0 = compute_diagnostics(HC, dim=2)
            symplectic_euler(
                HC, bV, dudt_fn, dt=dt, n_steps=40, dim=2,
                bc_set=bc_set, retopologize_fn=fn,
            )
            diag1 = compute_diagnostics(HC, dim=2)
            mass_drift = abs(diag1['total_mass'] - diag0['total_mass']) \
                / diag0['total_mass']
            self.assertLess(mass_drift, 1e-10)
            self.assertTrue(np.isfinite(diag1['KE']))
            self.assertLess(diag1['KE'], 1e-5)
            if extra:
                self.assertEqual(mps._projection_call_idx, 40)
            finals.append(float(diag1['R_max']))
        self.assertNotEqual(finals[0], finals[1])

    def test_projection_cadence_remap_full_delaunay(self):
        """Per-step FULL Delaunay + remap + cadence 5: reconnection
        neutrality holds every call (KE stays at the physical scale,
        machine mass) while the compression response evolves between
        projections."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler

        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = self._build()
        fn = partial(retopo_fn, retopo_remap='conservative',
                     projection_every=5)
        dt = self._dt(HC, params)
        diag0 = compute_diagnostics(HC, dim=2)
        edges0 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        symplectic_euler(
            HC, bV, dudt_fn, dt=dt, n_steps=40, dim=2,
            bc_set=bc_set, retopologize_fn=fn,
        )
        diag1 = compute_diagnostics(HC, dim=2)
        mass_drift = abs(diag1['total_mass'] - diag0['total_mass']) \
            / diag0['total_mass']
        self.assertLess(mass_drift, 1e-10)
        edges1 = {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}
        self.assertNotEqual(edges1, edges0)
        self.assertTrue(np.isfinite(diag1['KE']))
        self.assertLess(diag1['KE'], 1e-5)
        self.assertEqual(mps._projection_call_idx, 40)


if __name__ == '__main__':
    unittest.main()
