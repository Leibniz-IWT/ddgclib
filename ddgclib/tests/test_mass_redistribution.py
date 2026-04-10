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


if __name__ == '__main__':
    unittest.main()
