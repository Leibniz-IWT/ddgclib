"""Tests for ddgclib.initial_conditions module."""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex

from ddgclib.initial_conditions import (
    CompositeIC,
    CustomFieldIC,
    HagenPoiseuille3D,
    HydrostaticPressure,
    LinearPressureGradient,
    PoiseuillePlanar,
    UniformMass,
    UniformPressure,
    UniformVelocity,
    ZeroVelocity,
)


# Fixtures

@pytest.fixture
def complex_1d():
    """1D complex on [0, 1] with a few refinements."""
    HC = Complex(1, domain=[(0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14:
            bV.add(v)
    return HC, bV


@pytest.fixture
def complex_2d():
    """2D complex on [0, 1]^2 with one refinement."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
    return HC, bV


@pytest.fixture
def complex_3d():
    """3D complex on [0, 1]^3 with one refinement."""
    HC = Complex(3, domain=[(0.0, 1.0)] * 3)
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        for i in range(3):
            if abs(v.x_a[i]) < 1e-14 or abs(v.x_a[i] - 1.0) < 1e-14:
                bV.add(v)
                break
    return HC, bV


# Scalar field IC tests

class TestUniformPressure:
    def test_sets_scalar_pressure(self, complex_2d):
        HC, bV = complex_2d
        ic = UniformPressure(P0=101325.0)
        ic.apply(HC, bV)
        for v in HC.V:
            assert v.P == 101325.0

    def test_zero_default(self, complex_1d):
        HC, bV = complex_1d
        ic = UniformPressure()
        ic.apply(HC, bV)
        for v in HC.V:
            assert v.P == 0.0


class TestHydrostaticPressure:
    def test_linear_pressure_profile(self, complex_1d):
        HC, bV = complex_1d
        rho, g, h_ref = 1000.0, 9.81, 1.0
        ic = HydrostaticPressure(rho=rho, g=g, axis=0, h_ref=h_ref, P_ref=0.0)
        ic.apply(HC, bV)
        for v in HC.V:
            expected = rho * g * (h_ref - v.x_a[0])
            npt.assert_allclose(v.P, expected, atol=1e-12)

    def test_pressure_at_reference_height(self, complex_2d):
        HC, bV = complex_2d
        ic = HydrostaticPressure(rho=1.0, g=10.0, axis=1, h_ref=0.5, P_ref=100.0)
        ic.apply(HC, bV)
        # At y=0.5 (h_ref), P should equal P_ref
        for v in HC.V:
            if abs(v.x_a[1] - 0.5) < 1e-14:
                npt.assert_allclose(v.P, 100.0, atol=1e-12)


class TestLinearPressureGradient:
    def test_gradient_along_axis(self, complex_2d):
        HC, bV = complex_2d
        G = 5.0
        ic = LinearPressureGradient(G=G, axis=0, P_ref=100.0)
        ic.apply(HC, bV)
        for v in HC.V:
            expected = 100.0 - G * v.x_a[0]
            npt.assert_allclose(v.P, expected, atol=1e-12)


# Vector field IC tests

class TestZeroVelocity:
    def test_all_zero(self, complex_3d):
        HC, bV = complex_3d
        ic = ZeroVelocity(dim=3)
        ic.apply(HC, bV)
        for v in HC.V:
            npt.assert_array_equal(v.u, np.zeros(3))

    def test_dim_2(self, complex_2d):
        HC, bV = complex_2d
        ic = ZeroVelocity(dim=2)
        ic.apply(HC, bV)
        for v in HC.V:
            assert len(v.u) == 2
            npt.assert_array_equal(v.u, np.zeros(2))


class TestUniformVelocity:
    def test_sets_velocity(self, complex_2d):
        HC, bV = complex_2d
        u_vec = np.array([1.5, -0.3])
        ic = UniformVelocity(u_vec)
        ic.apply(HC, bV)
        for v in HC.V:
            npt.assert_array_equal(v.u, u_vec)

    def test_independent_copies(self, complex_2d):
        """Each vertex should get its own copy, not a shared reference."""
        HC, bV = complex_2d
        ic = UniformVelocity(np.array([1.0, 0.0]))
        ic.apply(HC, bV)
        verts = list(HC.V)
        verts[0].u[0] = 999.0
        assert verts[1].u[0] == 1.0  # Not affected


class TestPoiseuillePlanar:
    def test_parabolic_profile(self, complex_2d):
        HC, bV = complex_2d
        G, mu = 2.0, 0.5
        ic = PoiseuillePlanar(G=G, mu=mu, y_lb=0.0, y_ub=1.0,
                              flow_axis=0, normal_axis=1, dim=2)
        ic.apply(HC, bV)
        for v in HC.V:
            y = v.x_a[1]
            expected_ux = (G / (2 * mu)) * y * (1.0 - y)
            npt.assert_allclose(v.u[0], expected_ux, atol=1e-12)
            assert v.u[1] == 0.0  # No cross-flow

    def test_zero_at_walls(self, complex_2d):
        HC, bV = complex_2d
        ic = PoiseuillePlanar(G=1.0, mu=1.0, y_lb=0.0, y_ub=1.0, dim=2)
        ic.apply(HC, bV)
        for v in HC.V:
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14:
                npt.assert_allclose(v.u[0], 0.0, atol=1e-12)

    def test_analytical_velocity_method(self):
        ic = PoiseuillePlanar(G=2.0, mu=0.5, y_lb=0.0, y_ub=1.0, dim=2)
        # At midpoint y=0.5: u = (2/(2*0.5)) * 0.5 * 0.5 = 0.5
        npt.assert_allclose(ic.analytical_velocity(np.array([0.0, 0.5])), 0.5)


class TestHagenPoiseuille3D:
    def test_parabolic_profile(self, complex_3d):
        HC, bV = complex_3d
        U_max, R = 1.0, 0.5
        ic = HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=2, dim=3)
        ic.apply(HC, bV)
        for v in HC.V:
            r = np.linalg.norm(v.x_a[:2])
            expected_uz = U_max * max(0.0, 1.0 - (r / R) ** 2)
            npt.assert_allclose(v.u[2], expected_uz, atol=1e-12)
            assert v.u[0] == 0.0
            assert v.u[1] == 0.0

    def test_zero_at_wall(self):
        ic = HagenPoiseuille3D(U_max=2.0, R=1.0, flow_axis=2, dim=3)
        # At r=R, velocity should be zero
        npt.assert_allclose(ic.analytical_velocity(np.array([1.0, 0.0, 0.0])), 0.0)
        # At center, velocity should be U_max
        npt.assert_allclose(ic.analytical_velocity(np.array([0.0, 0.0, 5.0])), 2.0)


# Custom / generic IC tests

class TestCustomFieldIC:
    def test_custom_pressure(self, complex_2d):
        HC, bV = complex_2d
        ic = CustomFieldIC(fn=lambda x: x[0] ** 2 + x[1] ** 2, field_name='P')
        ic.apply(HC, bV)
        for v in HC.V:
            expected = v.x_a[0] ** 2 + v.x_a[1] ** 2
            npt.assert_allclose(v.P, expected, atol=1e-12)

    def test_custom_velocity(self, complex_2d):
        HC, bV = complex_2d
        ic = CustomFieldIC(fn=lambda x: np.array([x[1], -x[0]]), field_name='u')
        ic.apply(HC, bV)
        for v in HC.V:
            expected = np.array([v.x_a[1], -v.x_a[0]])
            npt.assert_allclose(v.u, expected, atol=1e-12)


class TestUniformMass:
    def test_mass_distribution(self, complex_2d):
        HC, bV = complex_2d
        total_volume = 1.0
        rho = 1000.0
        ic = UniformMass(total_volume=total_volume, rho=rho)
        ic.apply(HC, bV)
        n_verts = sum(1 for _ in HC.V)
        expected_mass = rho * total_volume / n_verts
        for v in HC.V:
            npt.assert_allclose(v.m, expected_mass, atol=1e-12)

    def test_total_mass_conserved(self, complex_3d):
        HC, bV = complex_3d
        total_volume = 1.0
        rho = 1.0
        ic = UniformMass(total_volume=total_volume, rho=rho)
        ic.apply(HC, bV)
        total_mass = sum(v.m for v in HC.V)
        npt.assert_allclose(total_mass, rho * total_volume, rtol=1e-12)


# CompositeIC tests

class TestCompositeIC:
    def test_applies_all_in_order(self, complex_2d):
        HC, bV = complex_2d
        ic = CompositeIC(
            ZeroVelocity(dim=2),
            UniformPressure(P0=100.0),
            UniformMass(total_volume=1.0, rho=1.0),
        )
        ic.apply(HC, bV)
        for v in HC.V:
            npt.assert_array_equal(v.u, np.zeros(2))
            assert v.P == 100.0
            assert hasattr(v, 'm')

    def test_later_ics_can_overwrite(self, complex_1d):
        """Second IC overwrites the first."""
        HC, bV = complex_1d
        ic = CompositeIC(
            UniformPressure(P0=100.0),
            UniformPressure(P0=200.0),
        )
        ic.apply(HC, bV)
        for v in HC.V:
            assert v.P == 200.0
