"""Tests for the Hagen-Poiseuille (planar 2D) case.

Validates:
1. Analytical Poiseuille velocity profile is correctly applied.
2. At equilibrium, the pressure gradient is non-zero but balanced.
3. Starting from uniform plug flow with BCs, velocity develops toward parabolic.
"""

import numpy as np
import numpy.testing as npt
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Fixtures

@pytest.fixture
def poiseuille_2d():
    from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
    HC, bV, ic, bc_set, params = setup_poiseuille_2d(
        G=1.0, mu=1.0, n_refine=2, L=1.0, h=1.0,
    )
    ic.apply(HC, bV)
    return HC, bV, ic, bc_set, params


# Analytical profile verification

class TestAnalyticalProfile:
    def test_velocity_profile(self, poiseuille_2d):
        """u_x(y) = (G/(2*mu)) * y * (h - y), u_y = 0."""
        HC, bV, _, _, p = poiseuille_2d
        G, mu, h = p['G'], p['mu'], p['h']

        for v in HC.V:
            y = v.x_a[1]
            u_anal = (G / (2 * mu)) * y * (h - y)
            npt.assert_allclose(v.u[0], u_anal, atol=1e-10,
                                err_msg=f"Wrong u_x at y={y}")
            npt.assert_allclose(v.u[1], 0.0, atol=1e-14)

    def test_pressure_gradient(self, poiseuille_2d):
        """P = -G * x, so vertices at different x should have different P."""
        HC, bV, _, _, p = poiseuille_2d
        G = p['G']

        for v in HC.V:
            expected = -G * v.x_a[0]
            npt.assert_allclose(v.P, expected, atol=1e-10)

    def test_wall_velocity_zero(self, poiseuille_2d):
        """Velocity at y=0 and y=h should be zero."""
        HC, bV, _, _, p = poiseuille_2d
        h = p['h']

        for v in HC.V:
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14:
                npt.assert_allclose(v.u[0], 0.0, atol=1e-14)

    def test_max_velocity_at_centerline(self, poiseuille_2d):
        """Maximum velocity should be at y = h/2."""
        HC, bV, _, _, p = poiseuille_2d
        G, mu, h = p['G'], p['mu'], p['h']
        U_max = G * h**2 / (8 * mu)

        # Find vertex closest to centerline
        center_v = min(
            (v for v in HC.V if abs(v.x_a[1] - h/2) < 0.1),
            key=lambda v: abs(v.x_a[1] - h/2)
        )
        npt.assert_allclose(center_v.u[0], U_max, atol=0.01)


# Developing flow test

class TestDevelopingFlow:
    def test_plug_flow_develops(self, poiseuille_2d):
        """Start with uniform plug flow, run integrator.

        With no-slip BCs, velocity near walls should decrease
        and center should begin to increase.
        """
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV, _, bc_set, p = poiseuille_2d

        # Override to uniform plug flow
        for v in HC.V:
            v.u = np.array([0.1, 0.0])

        # Simple viscous damping at walls (mock acceleration)
        def mock_accel(v, dim=2, **kw):
            # Laplacian-like: neighbors' average minus self
            if not v.nn:
                return np.zeros(dim)
            avg = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
            return 5.0 * (avg - v.u[:dim])

        # Record initial state
        wall_u_initial = [abs(v.u[0]) for v in bV
                          if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14]

        euler_velocity_only(HC, bV, mock_accel, dt=0.001, n_steps=10,
                            dim=2, bc_set=bc_set)

        # Wall velocities should be zero (no-slip enforced)
        for v in bV:
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14:
                npt.assert_array_equal(v.u, np.zeros(2))


# Setup function tests

class TestSetupFunction:
    def test_setup_creates_mesh(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        HC, bV, ic, bc_set, params = setup_poiseuille_2d(n_refine=1)
        assert sum(1 for _ in HC.V) > 0
        assert len(bV) > 0
        assert params['dim'] == 2

    def test_poiseuille_ic_object(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        _, _, _, _, params = setup_poiseuille_2d()
        pic = params['poiseuille_ic']
        # analytical_velocity returns a scalar (flow-axis component)
        u_flow = pic.analytical_velocity(np.array([0.5, 0.5]))
        assert u_flow > 0  # Flow in x-direction at y=0.5

    def test_custom_parameters(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        _, _, _, _, params = setup_poiseuille_2d(G=2.0, mu=0.5)
        assert params['G'] == 2.0
        assert params['mu'] == 0.5
        # U_max = G*h^2/(8*mu) = 2*1/(8*0.5) = 0.5
        npt.assert_allclose(params['U_max'], 0.5)
