"""Tests for the per-phase summed multiphase stress scheme.

Covers :func:`ddgclib.operators.multiphase_stress.multiphase_stress_force`
after the Phase 6 rewrite.  The key invariants verified here:

- **Bulk-only mesh**: the per-phase sum collapses to the single-phase
  formula, so bulk physics is unchanged vs ``stress_force``.
- **Uniform-pressure two-phase mesh**: with no surface tension and no
  velocity, the net force at every vertex is zero to within the
  mesh-closure residual.
- **Harmonic-mean μ**: the cross-phase-face viscosity used for the
  viscous flux matches ``2 μ_i μ_j / (μ_i + μ_j)``.
"""
import unittest

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.operators.multiphase_stress import (
    multiphase_stress_force,
    _face_viscosity_for_phase,
)
from ddgclib.operators.stress import cache_dual_volumes, stress_force


def _make_2d_rectangle(n_refine: int = 2):
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


class TestBulkOnlyCollapse(unittest.TestCase):
    """With every vertex in the same phase, the per-phase sum reduces
    to the single-phase stress_force formula (no cross-phase logic
    fires)."""

    def test_matches_stress_force_uniform_phase(self):
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
        )
        mps.assign_simplex_phases(HC, 2, criterion_fn=lambda c: 0)
        mps.refresh(HC, dim=2)

        for v in HC.V:
            v.u = np.zeros(2)
            v.p = 101325.0
            v.p_phase[0] = 101325.0

        for v in HC.V:
            if v in bV:
                continue
            F_multi = multiphase_stress_force(v, dim=2, mps=mps, HC=HC)
            F_single = stress_force(v, dim=2, mu=mps.get_mu(0), HC=HC)
            np.testing.assert_allclose(F_multi, F_single, atol=1e-12,
                                       err_msg=f"vertex {v.x_a[:2]}")


class TestUniformPressureTwoPhase(unittest.TestCase):
    """With uniform pressure across the whole domain and zero velocity,
    every interior vertex has zero net force."""

    def test_no_force_on_any_vertex(self):
        HC, bV = _make_2d_rectangle()
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
            gamma={(0, 1): 0.0},  # no surface tension → pure pressure test
        )
        mps.assign_simplex_phases(
            HC, 2, criterion_fn=lambda c: 0 if c[0] < 0 else 1,
        )
        mps.refresh(HC, dim=2)

        for v in HC.V:
            v.u = np.zeros(2)
            # Uniform pressure everywhere in both phase slots
            v.p_phase[0] = 5.0
            v.p_phase[1] = 5.0
            v.p = 5.0

        for v in HC.V:
            if v in bV:
                continue
            F = multiphase_stress_force(v, dim=2, mps=mps, HC=HC)
            self.assertLess(float(np.linalg.norm(F)), 1e-8,
                            msg=f"|F| = {np.linalg.norm(F)} "
                                f"at {v.x_a[:2]} "
                                f"(is_interface={v.is_interface})")


class TestExactPhaseViscosity(unittest.TestCase):
    """Each per-phase sub-face uses the exact viscosity mu_k of that phase.

    The interface geometry splits the dual cell into per-phase
    sub-domains.  Each sub-face lies entirely within one phase, so
    its viscosity is unambiguously mu_k — no blending needed.
    """

    def test_face_viscosity_same_phase(self):
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
        )

        class V:
            def __init__(self, phase, iface=False):
                self.phase = phase
                self.is_interface = iface

        v_i = V(phase=0, iface=False)
        v_j = V(phase=0, iface=False)
        mu = _face_viscosity_for_phase(mps, v_i, v_j, k=0)
        self.assertAlmostEqual(mu, 0.1)

    def test_face_viscosity_cross_phase_uses_exact_mu_k(self):
        """Cross-phase sub-faces use the exact phase viscosity, not a blend."""
        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
        )

        class V:
            def __init__(self, phase, iface=False):
                self.phase = phase
                self.is_interface = iface

        v_i = V(phase=1, iface=True)
        v_j = V(phase=0, iface=False)
        # phase-1 sub-face → exact mu_1 = 0.5 (not harmonic mean)
        mu1 = _face_viscosity_for_phase(mps, v_i, v_j, k=1)
        self.assertAlmostEqual(mu1, 0.5)
        # phase-0 sub-face → exact mu_0 = 0.1
        mu0 = _face_viscosity_for_phase(mps, v_i, v_j, k=0)
        self.assertAlmostEqual(mu0, 0.1)


class TestBulkOnlyCollapse3D(unittest.TestCase):
    """Same invariant as the 2D TestBulkOnlyCollapse, but for 3D: the
    per-phase summed force equals the single-phase stress_force when
    all vertices are in one phase."""

    def test_matches_stress_force_uniform_phase_3d(self):
        from ddgclib.geometry.domains import box
        result = box(Lx=2.0, Ly=2.0, Lz=2.0, refinement=2,
                     origin=(-1.0, -1.0, -1.0))
        HC = result.HC
        bV = result.bV
        for v in HC.V:
            v.boundary = v in bV
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, 3)

        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
        )
        mps.assign_simplex_phases(HC, 3, criterion_fn=lambda c: 0)
        mps.refresh(HC, dim=3)

        for v in HC.V:
            v.u = np.zeros(3)
            v.p = 101325.0
            v.p_phase[0] = 101325.0

        for v in HC.V:
            if v in bV:
                continue
            F_multi = multiphase_stress_force(v, dim=3, mps=mps, HC=HC)
            F_single = stress_force(v, dim=3, mu=mps.get_mu(0), HC=HC)
            np.testing.assert_allclose(F_multi, F_single, atol=1e-10,
                                       err_msg=f"vertex {v.x_a[:3]}")


class TestUniformPressureTwoPhase3D(unittest.TestCase):
    """3D analog of TestUniformPressureTwoPhase — uniform per-phase
    pressure, zero velocity, no surface tension → zero net force."""

    def test_no_force_on_any_vertex_3d(self):
        from ddgclib.geometry.domains import box
        result = box(Lx=2.0, Ly=2.0, Lz=2.0, refinement=2,
                     origin=(-1.0, -1.0, -1.0))
        HC = result.HC
        bV = result.bV
        for v in HC.V:
            v.boundary = v in bV
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, 3)

        mps = MultiphaseSystem(
            phases=[
                PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000),
                PhaseProperties(eos=TaitMurnaghan(rho0=800), mu=0.5, rho0=800),
            ],
            gamma={(0, 1): 0.0},  # no surface tension
        )
        mps.assign_simplex_phases(
            HC, 3,
            criterion_fn=lambda c: 1 if np.linalg.norm(c[:3]) < 0.5 else 0,
        )
        mps.refresh(HC, dim=3)

        for v in HC.V:
            v.u = np.zeros(3)
            v.p_phase[0] = 5.0
            v.p_phase[1] = 5.0
            v.p = 5.0

        for v in HC.V:
            if v in bV:
                continue
            F = multiphase_stress_force(v, dim=3, mps=mps, HC=HC)
            self.assertLess(
                float(np.linalg.norm(F)), 1e-6,
                msg=f"|F| = {np.linalg.norm(F)} at {v.x_a[:3]} "
                    f"(is_interface={getattr(v, 'is_interface', False)})",
            )


if __name__ == '__main__':
    unittest.main()
