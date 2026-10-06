"""Tests for pressure-preserving mass redistribution after retriangulation."""
import unittest

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.eos._tait_murnaghan import TaitMurnaghan
from ddgclib.eos._ideal_gas import IdealGas
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.operators.mass_redistribution import (
    snapshot_pressure,
    snapshot_pressure_multiphase,
    redistribute_mass_single_phase,
    redistribute_mass_multiphase,
    _is_redistributable,
)


def _make_2d_mesh(refinement=3):
    """Create a small 2D mesh with duals and mass."""
    HC = Complex(2, domain=[[0, 1], [0, 1]])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()
    bV = HC.boundary()
    for v in HC.V:
        v.boundary = v in bV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim=2)
    return HC, bV


def _assign_eos_pressure(HC, bV, eos, rho0=None):
    """Assign mass from EOS reference density, then compute pressure."""
    if rho0 is None:
        rho0 = eos.rho0
    for v in HC.V:
        vol = getattr(v, 'dual_vol', 0.0)
        if vol > 1e-30:
            v.m = rho0 * vol
            v.rho = rho0
            v.p = float(eos.pressure(rho0))
        else:
            v.m = rho0 * 1e-30
            v.rho = rho0
            v.p = float(eos.pressure(rho0))


class TestSnapshotPressure(unittest.TestCase):
    """Test pressure snapshot capture."""

    def test_snapshot_captures_all_vertices(self):
        HC, bV = _make_2d_mesh(refinement=2)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)
        snap = snapshot_pressure(HC)
        n_verts = sum(1 for _ in HC.V)
        self.assertEqual(len(snap), n_verts)

    def test_snapshot_values_match(self):
        HC, bV = _make_2d_mesh(refinement=2)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)
        snap = snapshot_pressure(HC)
        for v in HC.V:
            self.assertAlmostEqual(snap[id(v)], v.p, places=10)


class TestMassConservation(unittest.TestCase):
    """Total mass must be conserved to machine precision."""

    def test_total_mass_conserved_tait(self):
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)

        M_before = sum(v.m for v in HC.V)
        snap = snapshot_pressure(HC)

        # Simulate a retriangulation that changes dual volumes slightly
        # by perturbing vertex positions
        interior = [v for v in HC.V if v not in bV]
        rng = np.random.RandomState(42)
        for v in interior:
            dx = rng.randn(2) * 0.001
            HC.V.move(v, tuple(v.x_a[:2] + dx))

        # Recompute duals (simulating what _retopologize does)
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim=2)

        # Redistribute
        diag = redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )

        M_after = sum(v.m for v in HC.V if v not in bV
                      and getattr(v, 'dual_vol', 0.0) > 1e-30
                      and id(v) in snap)
        self.assertAlmostEqual(diag['total_mass_before'],
                               diag['total_mass_after'], places=10)

    def test_total_mass_conserved_ideal_gas(self):
        HC, bV = _make_2d_mesh(refinement=3)
        eos = IdealGas(rho0=1.225, T=293.15)
        _assign_eos_pressure(HC, bV, eos)

        snap = snapshot_pressure(HC)

        # Perturb and recompute
        interior = [v for v in HC.V if v not in bV]
        rng = np.random.RandomState(123)
        for v in interior:
            dx = rng.randn(2) * 0.001
            HC.V.move(v, tuple(v.x_a[:2] + dx))
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim=2)

        diag = redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )
        self.assertAlmostEqual(diag['total_mass_before'],
                               diag['total_mass_after'], places=10)


class TestPressurePreservation(unittest.TestCase):
    """Redistribution should preserve pressure field."""

    def test_static_mesh_pressure_unchanged(self):
        """No vertex movement: pressure should be exactly preserved."""
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)

        snap = snapshot_pressure(HC)

        # No movement — just call redistribute
        diag = redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )

        # Pressure should be unchanged
        for v in HC.V:
            if v in bV or getattr(v, 'dual_vol', 0.0) < 1e-30:
                continue
            vol = v.dual_vol
            p_new = float(eos.pressure(v.m / vol))
            self.assertAlmostEqual(p_new, snap[id(v)], places=5,
                                   msg=f"Pressure changed on static mesh")

    def test_perturbed_mesh_pressure_closer(self):
        """After perturbation, redistribution should keep pressure closer."""
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0, K=1e6, n=7.15)
        _assign_eos_pressure(HC, bV, eos)
        snap = snapshot_pressure(HC)

        # Perturb vertices
        interior = [v for v in HC.V if v not in bV]
        rng = np.random.RandomState(99)
        for v in interior:
            dx = rng.randn(2) * 0.005
            HC.V.move(v, tuple(v.x_a[:2] + dx))

        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim=2)

        # Measure pressure error WITHOUT redistribution
        errors_no_redist = []
        for v in interior:
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 1e-30 and id(v) in snap:
                p_no = float(eos.pressure(v.m / vol))
                errors_no_redist.append(abs(p_no - snap[id(v)]))

        # Now redistribute
        diag = redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )

        # Measure pressure error WITH redistribution
        errors_with_redist = []
        for v in interior:
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 1e-30 and id(v) in snap:
                p_yes = float(eos.pressure(v.m / vol))
                errors_with_redist.append(abs(p_yes - snap[id(v)]))

        if errors_no_redist and errors_with_redist:
            max_err_no = max(errors_no_redist)
            max_err_yes = max(errors_with_redist)
            # Redistribution should significantly reduce pressure error
            self.assertLess(max_err_yes, max_err_no * 0.5,
                            f"Redistribution did not reduce pressure error: "
                            f"{max_err_yes:.6e} vs {max_err_no:.6e}")


class TestBoundaryVertices(unittest.TestCase):
    """Boundary/wall vertex mass must be unchanged."""

    def test_wall_mass_unchanged(self):
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)

        wall_masses = {id(v): v.m for v in bV}
        snap = snapshot_pressure(HC)

        redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )

        for v in bV:
            self.assertEqual(v.m, wall_masses[id(v)],
                             msg="Wall vertex mass was modified")


class TestNewlyInjectedVertices(unittest.TestCase):
    """Vertices not in snapshot should be excluded from redistribution."""

    def test_unknown_vertex_excluded(self):
        HC, bV = _make_2d_mesh(refinement=2)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)

        snap = snapshot_pressure(HC)

        # Simulate a newly injected vertex by removing one from the snapshot
        some_interior = None
        for v in HC.V:
            if v not in bV and getattr(v, 'dual_vol', 0.0) > 1e-30:
                some_interior = v
                break
        self.assertIsNotNone(some_interior)

        original_mass = some_interior.m
        del snap[id(some_interior)]  # pretend this vertex is new

        redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )

        # This vertex should NOT have been modified
        self.assertEqual(some_interior.m, original_mass)


class TestBackwardCompatibility(unittest.TestCase):
    """Default redistribute_mass=False should be a no-op."""

    def test_integrator_default_no_redistribution(self):
        """Verify that the integrator signature accepts the new params."""
        from ddgclib.dynamic_integrators._integrators_dynamic import euler
        import inspect
        sig = inspect.signature(euler)
        self.assertIn('pressure_model', sig.parameters)
        self.assertIn('redistribute_mass', sig.parameters)
        # Default should be False
        self.assertEqual(sig.parameters['redistribute_mass'].default, False)
        self.assertIsNone(sig.parameters['pressure_model'].default)

    def test_all_integrators_have_params(self):
        """All 5 integrators should accept the new parameters."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            euler, symplectic_euler, rk45, euler_velocity_only,
            euler_adaptive,
        )
        import inspect
        for fn in [euler, symplectic_euler, rk45, euler_velocity_only,
                   euler_adaptive]:
            sig = inspect.signature(fn)
            self.assertIn('pressure_model', sig.parameters,
                          msg=f"{fn.__name__} missing pressure_model")
            self.assertIn('redistribute_mass', sig.parameters,
                          msg=f"{fn.__name__} missing redistribute_mass")


class TestScaleFactor(unittest.TestCase):
    """Scale factor should be close to 1.0 for small perturbations."""

    def test_scale_near_unity(self):
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)
        snap = snapshot_pressure(HC)

        # Small perturbation
        rng = np.random.RandomState(7)
        for v in HC.V:
            if v not in bV:
                dx = rng.randn(2) * 0.001
                HC.V.move(v, tuple(v.x_a[:2] + dx))
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim=2)

        diag = redistribute_mass_single_phase(
            HC, dim=2, eos=eos, bV=bV, pressure_snapshot=snap,
        )
        # Scale factor should be within 1% of 1.0
        self.assertAlmostEqual(diag['scale_factor'], 1.0, delta=0.01)


class TestIsRedistributable(unittest.TestCase):
    """Test the redistributable vertex filter."""

    def test_boundary_excluded(self):
        HC, bV = _make_2d_mesh(refinement=2)
        snap = {id(v): 0.0 for v in HC.V}
        for v in bV:
            self.assertFalse(_is_redistributable(v, bV, snap))

    def test_zero_volume_excluded(self):
        HC, bV = _make_2d_mesh(refinement=2)
        snap = {id(v): 0.0 for v in HC.V}
        # Artificially set a vertex dual_vol to 0
        for v in HC.V:
            if v not in bV:
                v.dual_vol = 0.0
                self.assertFalse(_is_redistributable(v, bV, snap))
                break

    def test_missing_snapshot_excluded(self):
        HC, bV = _make_2d_mesh(refinement=2)
        snap = {}  # empty snapshot
        for v in HC.V:
            if v not in bV and getattr(v, 'dual_vol', 0.0) > 1e-30:
                self.assertFalse(_is_redistributable(v, bV, snap))
                break


class TestIntegrationWithRetopologize(unittest.TestCase):
    """Test that redistribution integrates with _retopologize."""

    def test_retopologize_with_redistribution(self):
        """Full retopologize call with redistribution enabled."""
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize,
        )
        HC, bV = _make_2d_mesh(refinement=3)
        eos = TaitMurnaghan(rho0=1000.0)
        _assign_eos_pressure(HC, bV, eos)

        M_before = sum(v.m for v in HC.V)

        # Perturb slightly
        for v in HC.V:
            if v not in bV:
                rng = np.random.RandomState(id(v) % 2**31)
                dx = rng.randn(2) * 0.001
                HC.V.move(v, tuple(v.x_a[:2] + dx))

        # Call full retopologize with redistribution
        _retopologize(HC, bV, dim=2,
                      pressure_model=eos, redistribute_mass=True)

        M_after = sum(v.m for v in HC.V)
        # Total mass (including boundary) should be very close
        # (boundary mass unchanged, interior mass conserved)
        self.assertAlmostEqual(M_before, M_after, places=8)


class TestPhaseLedger(unittest.TestCase):
    """The ``ledger=`` rule of ``redistribute_mass_multiphase`` (method
    axis ``phase_ledger``, laneF 2026-10-05).

    A reconnection can give a vertex a sub-volume of a phase it had no
    mass of (the phase APPEARS there) or take the last sub-volume of a
    phase it still carries mass of (the phase DISAPPEARS).  Under the
    historic ``'snapshot'`` rule the appeared phase stays massless, so
    ``compute_phase_pressures`` publishes ``p_phase[k] = 0`` absolute for
    a sub-volume the force reads as present: a pressure hole of ``P0``
    (the dam-break ejection).  Under ``'volume'`` the mass follows the
    sub-volume.
    """

    P0 = 101325.0

    def _two_phase_mesh(self):
        """Left half phase 0 (gas), right half phase 1 (liquid); every
        vertex a bulk vertex of its phase at the reference pressure."""
        from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
        from ddgclib.operators.mass_redistribution import (
            snapshot_geometry_multiphase,
        )
        HC, bV = _make_2d_mesh(refinement=2)
        eos_g = TaitMurnaghan(rho0=1.225, P0=self.P0, K=120.0, n=1.0,
                              rho_clip=(0.2, 5.0))
        eos_l = TaitMurnaghan(rho0=1000.0, P0=self.P0, K=1e5, n=1.0,
                              rho_clip=(0.8, 1.2))
        mps = MultiphaseSystem(phases=[
            PhaseProperties(eos=eos_g, mu=1e-5, rho0=1.225, name='gas'),
            PhaseProperties(eos=eos_l, mu=1e-3, rho0=1000.0, name='liq')],
            gamma={(0, 1): 0.0})
        for v in HC.V:
            k = 0 if v.x_a[0] < 0.5 else 1
            vol = float(v.dual_vol)
            v.phase = k
            v.is_interface = False
            v.dual_vol_phase = np.zeros(2)
            v.dual_vol_phase[k] = vol
            v.m_phase = np.zeros(2)
            v.m_phase[k] = mps.phases[k].rho0 * vol
            v.p_phase = np.zeros(2)
            v.p_phase[k] = self.P0
            v.rho_phase = np.zeros(2)
            v.interface_phases = frozenset()
            v.m = float(v.m_phase.sum())
        snap = snapshot_geometry_multiphase(HC, 2)
        return HC, bV, mps, snap

    def _presence_change(self, HC, bV):
        """After the 'rebuild': one interior gas vertex at the seam gains a
        liquid sub-volume (no liquid mass), one interior liquid vertex
        loses its sub-volume (keeps its mass).  Returns the two."""
        interior = [v for v in HC.V if v not in bV]
        gained = min((v for v in interior if v.x_a[0] < 0.5),
                     key=lambda v: 0.5 - v.x_a[0])
        lost = min((v for v in interior if v.x_a[0] >= 0.5),
                   key=lambda v: v.x_a[0] - 0.5)
        vol = float(gained.dual_vol)
        gained.dual_vol_phase = np.array([0.7 * vol, 0.3 * vol])
        lost.dual_vol_phase = np.zeros(2)
        return gained, lost

    def test_snapshot_rule_leaves_a_hole_and_a_stranded_mass(self):
        from ddgclib.operators.multiphase_stress import _phase_present_at
        HC, bV, mps, snap = self._two_phase_mesh()
        M0 = np.sum([v.m_phase for v in HC.V], axis=0)
        gained, lost = self._presence_change(HC, bV)
        m_lost_before = float(lost.m_phase[1])
        diag = redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='snapshot')
        self.assertEqual(diag['adopted'], {})
        # the appeared phase has a sub-volume but no mass ...
        self.assertEqual(float(gained.m_phase[1]), 0.0)
        mps.compute_phase_pressures(HC)
        # ... so it publishes 0 Pa ABSOLUTE while the force reads it as
        # present: the hole
        self.assertEqual(float(gained.p_phase[1]), 0.0)
        self.assertTrue(_phase_present_at(gained, 1))
        # the lost phase keeps its mass without a sub-volume
        self.assertEqual(float(lost.m_phase[1]), m_lost_before)
        self.assertEqual(float(lost.dual_vol_phase[1]), 0.0)
        M1 = np.sum([v.m_phase for v in HC.V], axis=0)
        np.testing.assert_allclose(M1, M0, rtol=1e-12)

    def test_volume_rule_mass_follows_the_sub_volume(self):
        from ddgclib.operators.mass_redistribution import (
            restore_pressure_multiphase,
        )
        HC, bV, mps, snap = self._two_phase_mesh()
        M0 = np.sum([v.m_phase for v in HC.V], axis=0)
        gained, lost = self._presence_change(HC, bV)
        diag = redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='volume')
        # the appeared phase was targeted at the local pressure of its
        # neighbours (all at P0) and carries the matching mass
        self.assertEqual(list(diag['adopted']), [id(gained)])
        self.assertEqual(list(diag['adopted'][id(gained)]), [1])
        self.assertAlmostEqual(diag['adopted'][id(gained)][1], self.P0,
                               places=6)
        rho_l = float(mps.phases[1].eos.density(self.P0))
        d1 = diag['per_phase_diagnostics'][1]
        # target rho(p_local) * sub-volume, times the conserving rescale
        # of the phase (the released cell is shared out over the pool)
        self.assertAlmostEqual(
            float(gained.m_phase[1])
            / (rho_l * gained.dual_vol_phase[1] * d1['scale_factor']),
            1.0, places=9)
        self.assertGreater(d1['scale_factor'], 1.0)
        # the lost phase released its mass to the pool
        self.assertEqual(float(lost.m_phase[1]), 0.0)
        self.assertGreater(d1['mass_released'], 0.0)
        self.assertEqual(d1['n_adopted'], 1)
        # exact per-phase conservation, holes and strandings gone
        M1 = np.sum([v.m_phase for v in HC.V], axis=0)
        np.testing.assert_allclose(M1, M0, rtol=1e-12)
        for v in HC.V:
            for k in range(2):
                self.assertEqual(v.dual_vol_phase[k] > 1e-30,
                                 v.m_phase[k] > 1e-30)
        # the remap's restore keeps the adopted pressure
        mps.compute_phase_pressures(HC)
        self.assertGreater(float(gained.p_phase[1]), 0.9 * self.P0)
        restore_pressure_multiphase(HC, mps, snap, adopted=diag['adopted'])
        self.assertAlmostEqual(float(gained.p_phase[1]), self.P0, places=6)

    def test_volume_rule_adopts_the_local_pressure(self):
        """The adopted pressure is the sub-volume weighted mean of the
        snapshot pressure over the neighbours that had the phase."""
        HC, bV, mps, snap = self._two_phase_mesh()
        gained, lost = self._presence_change(HC, bV)
        num = den = 0.0
        for w in gained.nn:
            if snap[id(w)]['dual_vol_phase'][1] > 0.0:
                snap[id(w)]['p_phase'][1] = self.P0 + 100.0 * w.x_a[1]
                num += snap[id(w)]['p_phase'][1] * snap[id(w)]['dual_vol_phase'][1]
                den += snap[id(w)]['dual_vol_phase'][1]
        self.assertGreater(den, 0.0)
        diag = redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='volume')
        self.assertAlmostEqual(diag['adopted'][id(gained)][1], num / den,
                               places=9)

    def test_rules_agree_to_the_bit_without_a_presence_change(self):
        HC, bV, mps, snap = self._two_phase_mesh()
        rng = np.random.RandomState(7)
        for v in HC.V:
            if v not in bV:
                HC.V.move(v, tuple(v.x_a[:2] + rng.randn(2) * 1e-3))
        compute_vd(HC, method="barycentric")
        cache_dual_volumes(HC, dim=2)
        for v in HC.V:
            v.dual_vol_phase = np.zeros(2)
            v.dual_vol_phase[v.phase] = float(v.dual_vol)
        before = {id(v): v.m_phase.copy() for v in HC.V}
        redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='snapshot')
        after_snapshot = {id(v): v.m_phase.copy() for v in HC.V}
        for v in HC.V:
            v.m_phase = before[id(v)].copy()
        diag = redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='volume')
        self.assertEqual(diag['adopted'], {})
        for v in HC.V:
            np.testing.assert_array_equal(v.m_phase, after_snapshot[id(v)])

    def test_adopt_rule_keeps_the_lost_mass_as_inertia(self):
        HC, bV, mps, snap = self._two_phase_mesh()
        M0 = np.sum([v.m_phase for v in HC.V], axis=0)
        gained, lost = self._presence_change(HC, bV)
        m_lost_before = float(lost.m_phase[1])
        diag = redistribute_mass_multiphase(
            HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='adopt')
        self.assertEqual(list(diag['adopted']), [id(gained)])
        self.assertGreater(float(gained.m_phase[1]), 0.0)
        self.assertEqual(float(lost.m_phase[1]), m_lost_before)
        self.assertEqual(diag['per_phase_diagnostics'][1]['mass_released'],
                         0.0)
        M1 = np.sum([v.m_phase for v in HC.V], axis=0)
        np.testing.assert_allclose(M1, M0, rtol=1e-12)

    def test_unknown_rule_is_refused(self):
        HC, bV, mps, snap = self._two_phase_mesh()
        with self.assertRaises(ValueError):
            redistribute_mass_multiphase(
                HC, 2, mps, bV=bV, pressure_snapshot=snap, ledger='other')
        with self.assertRaises(ValueError):
            redistribute_mass_multiphase(
                HC, 2, mps, bV=bV,
                pressure_snapshot=snapshot_pressure_multiphase(HC, 2),
                ledger='volume')


if __name__ == '__main__':
    unittest.main()
