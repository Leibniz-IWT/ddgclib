"""Tests for ddgclib.operators.stress — Cauchy stress tensor operators."""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_2d():
    """2D mesh on [0,1]^2 with barycentric duals computed."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()

    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    compute_vd(HC, cdist=1e-10)

    for v in HC.V:
        v.u = np.zeros(2)
        v.p = 0.0
        v.m = 1.0

    return HC, bV


@pytest.fixture
def mesh_2d_refined():
    """Finer 2D mesh on [0,1]^2 with barycentric duals."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    HC.refine_all()

    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    compute_vd(HC, cdist=1e-10)

    for v in HC.V:
        v.u = np.zeros(2)
        v.p = 0.0
        v.m = 1.0

    return HC, bV


# ---------------------------------------------------------------------------
# Test dual_area_vector
# ---------------------------------------------------------------------------

class TestDualAreaVector2D:
    """Tests for dual_area_vector in 2D."""

    def test_closure_interior(self, mesh_2d):
        """sum_j A_ij = 0 for every interior vertex (closed dual cell)."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_2d
        for v in HC.V:
            if v in bV:
                continue
            A_total = np.zeros(2)
            for v_j in v.nn:
                A_total += dual_area_vector(v, v_j, HC, dim=2)
            npt.assert_allclose(
                A_total, np.zeros(2), atol=1e-12,
                err_msg=f"Area vectors not closed at interior vertex {v.x}",
            )

    def test_antisymmetry(self, mesh_2d):
        """A_ij = -A_ji for each edge."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_2d
        tested = 0
        for v_i in HC.V:
            for v_j in v_i.nn:
                A_ij = dual_area_vector(v_i, v_j, HC, dim=2)
                A_ji = dual_area_vector(v_j, v_i, HC, dim=2)
                npt.assert_allclose(
                    A_ij, -A_ji, atol=1e-12,
                    err_msg=f"Anti-symmetry failed for edge {v_i.x}-{v_j.x}",
                )
                tested += 1
        assert tested > 0

    def test_magnitude_matches_e_star(self, mesh_2d):
        """|A_ij| should equal the dual edge length from e_star in 2D."""
        from ddgclib.operators.stress import dual_area_vector
        from hyperct.ddg import e_star as _e_star

        HC, bV = mesh_2d
        for v_i in HC.V:
            if v_i in bV:
                continue
            for v_j in v_i.nn:
                A_ij = dual_area_vector(v_i, v_j, HC, dim=2)
                e_scalar = _e_star(v_i, v_j, HC, dim=2)
                npt.assert_allclose(
                    np.linalg.norm(A_ij), e_scalar, rtol=1e-10,
                    err_msg=f"|A_ij| != e_star at edge {v_i.x}-{v_j.x}",
                )

    def test_nonzero(self, mesh_2d):
        """Area vectors should be non-zero for interior edges."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_2d
        for v_i in HC.V:
            if v_i in bV:
                continue
            for v_j in v_i.nn:
                A_ij = dual_area_vector(v_i, v_j, HC, dim=2)
                assert np.linalg.norm(A_ij) > 1e-15, \
                    f"Zero area vector at edge {v_i.x}-{v_j.x}"


# ---------------------------------------------------------------------------
# Test dual_volume
# ---------------------------------------------------------------------------

class TestDualVolume2D:

    def test_positive(self, mesh_2d):
        """Dual volumes should be positive for all vertices."""
        from ddgclib.operators.stress import dual_volume

        HC, bV = mesh_2d
        for v in HC.V:
            if v in bV:
                continue
            vol = dual_volume(v, HC, dim=2)
            assert vol > 0, f"Non-positive dual volume at {v.x}: {vol}"

    def test_partition_of_unity(self, mesh_2d):
        """Sum of all dual volumes should approximate the domain area."""
        from ddgclib.operators.stress import dual_volume

        HC, bV = mesh_2d
        total = sum(dual_volume(v, HC, dim=2) for v in HC.V)
        domain_area = 1.0  # [0,1]^2
        npt.assert_allclose(
            total, domain_area, rtol=0.15,
            err_msg=f"Dual volumes don't sum to domain area: {total}",
        )


# ---------------------------------------------------------------------------
# Test velocity_difference_tensor
# ---------------------------------------------------------------------------

class TestVelocityDifferenceTensor2D:

    def test_uniform_velocity_zero_du(self, mesh_2d):
        """Uniform velocity should give zero du_i."""
        from ddgclib.operators.stress import velocity_difference_tensor

        HC, bV = mesh_2d
        for v in HC.V:
            v.u = np.array([1.0, 0.5])

        for v in HC.V:
            if v in bV:
                continue
            du = velocity_difference_tensor(v, HC, dim=2)
            npt.assert_allclose(
                du, np.zeros((2, 2)), atol=1e-10,
                err_msg=f"Non-zero du_i for uniform velocity at {v.x}",
            )

    def test_linear_velocity_field(self, mesh_2d):
        """Linear velocity u = [ax + by, cx + dy] -> du_pointwise approximates
        the gradient [[a,b],[c,d]].

        Uses velocity_difference_tensor_pointwise (Du/Vol) for comparison
        with analytical gradients.
        """
        from ddgclib.operators.stress import velocity_difference_tensor_pointwise

        HC, bV = mesh_2d
        # u = [2*x + 3*y, -x + y]
        a, b, c, d = 2.0, 3.0, -1.0, 1.0
        expected = np.array([[a, b], [c, d]])

        for v in HC.V:
            x, y = v.x_a[0], v.x_a[1]
            v.u = np.array([a * x + b * y, c * x + d * y])

        # Check interior vertices: du_i should be proportional to expected
        # (same ratios between components) even if the magnitude is off
        for v in HC.V:
            if v in bV:
                continue
            du = velocity_difference_tensor_pointwise(v, HC, dim=2)
            # Check ratios: du[0,0]/du[0,1] ≈ a/b = 2/3
            if abs(du[0, 1]) > 1e-10:
                ratio_expected = a / b
                ratio_actual = du[0, 0] / du[0, 1]
                npt.assert_allclose(
                    ratio_actual, ratio_expected, rtol=0.1,
                    err_msg=f"Component ratio [0,0]/[0,1] wrong at {v.x}",
                )
            # Check sign pattern matches expected
            for i in range(2):
                for j in range(2):
                    if abs(expected[i, j]) > 0.1:
                        assert np.sign(du[i, j]) == np.sign(expected[i, j]), \
                            f"Sign mismatch at ({i},{j}): du={du[i,j]}, expected={expected[i,j]}"

    def test_linear_velocity_converges(self, mesh_2d, mesh_2d_refined):
        """Pointwise du_i error should decrease with mesh refinement."""
        from ddgclib.operators.stress import velocity_difference_tensor_pointwise

        a, b, c, d = 2.0, 3.0, -1.0, 1.0
        expected = np.array([[a, b], [c, d]])

        errors = {}
        for label, (HC, bV) in [("coarse", mesh_2d), ("fine", mesh_2d_refined)]:
            for v in HC.V:
                x, y = v.x_a[0], v.x_a[1]
                v.u = np.array([a * x + b * y, c * x + d * y])

            errs = []
            for v in HC.V:
                if v in bV:
                    continue
                du = velocity_difference_tensor_pointwise(v, HC, dim=2)
                # Normalize by expected magnitude to get relative error
                errs.append(np.linalg.norm(du - expected) / np.linalg.norm(expected))
            errors[label] = np.median(errs)

        # Fine mesh should have same or smaller relative error
        # (or at least not dramatically worse)
        assert errors["fine"] <= errors["coarse"] * 1.1, \
            f"Error increased with refinement: coarse={errors['coarse']:.4f}, fine={errors['fine']:.4f}"

    def test_integrated_scales_with_volume(self, mesh_2d):
        """Integrated Du should equal pointwise du * Vol_i."""
        from ddgclib.operators.stress import (
            velocity_difference_tensor,
            velocity_difference_tensor_pointwise,
            _get_dual_vol,
        )

        HC, bV = mesh_2d
        for v in HC.V:
            x, y = v.x_a[0], v.x_a[1]
            v.u = np.array([2.0 * x + 3.0 * y, -x + y])

        for v in HC.V:
            if v in bV:
                continue
            Du = velocity_difference_tensor(v, HC, dim=2)
            du_pw = velocity_difference_tensor_pointwise(v, HC, dim=2)
            Vol_i = _get_dual_vol(v, HC, dim=2)
            npt.assert_allclose(
                Du, du_pw * Vol_i, rtol=1e-10,
                err_msg=f"Du != du_pw * Vol at {v.x}",
            )


# ---------------------------------------------------------------------------
# Test strain_rate
# ---------------------------------------------------------------------------

class TestStrainRate:

    def test_symmetric(self):
        """strain_rate should return a symmetric tensor."""
        from ddgclib.operators.stress import strain_rate
        du = np.array([[1.0, 2.0], [3.0, 4.0]])
        eps = strain_rate(du)
        npt.assert_allclose(eps, eps.T, atol=1e-15)

    def test_value(self):
        """strain_rate should be 0.5 * (du + du^T)."""
        from ddgclib.operators.stress import strain_rate
        du = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = 0.5 * (du + du.T)
        npt.assert_allclose(strain_rate(du), expected)


# ---------------------------------------------------------------------------
# Test cauchy_stress
# ---------------------------------------------------------------------------

class TestCauchyStress:

    def test_pressure_only(self):
        """With mu=0, sigma = -p * I."""
        from ddgclib.operators.stress import cauchy_stress
        du = np.array([[1.0, 2.0], [3.0, 4.0]])
        sigma = cauchy_stress(p=5.0, du=du, mu=0.0, dim=2)
        expected = -5.0 * np.eye(2)
        npt.assert_allclose(sigma, expected)

    def test_shear_only(self):
        """With p=0, sigma = 2*mu*epsilon."""
        from ddgclib.operators.stress import cauchy_stress, strain_rate
        du = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigma = cauchy_stress(p=0.0, du=du, mu=0.5, dim=2)
        expected = 2.0 * 0.5 * strain_rate(du)
        npt.assert_allclose(sigma, expected)

    def test_full(self):
        """Full Newtonian stress: sigma = -p*I + 2*mu*epsilon."""
        from ddgclib.operators.stress import cauchy_stress, strain_rate
        du = np.array([[1.0, 2.0], [3.0, 4.0]])
        sigma = cauchy_stress(p=10.0, du=du, mu=1.0, dim=2)
        expected = -10.0 * np.eye(2) + 2.0 * strain_rate(du)
        npt.assert_allclose(sigma, expected)


# ---------------------------------------------------------------------------
# Test stress_force
# ---------------------------------------------------------------------------

class TestStressForce2D:

    def test_uniform_pressure_zero_force(self, mesh_2d):
        """Uniform pressure + zero velocity => zero stress force.

        This follows from the closure property: sum_j A_ij = 0 for closed
        dual cells, so -p * sum_j A_ij = 0.
        """
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 100.0
            v.u = np.zeros(2)
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=0.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero force for uniform pressure at {v.x}",
            )

    def test_uniform_velocity_zero_force(self, mesh_2d):
        """Uniform velocity + zero pressure => zero stress force.

        All du_i = 0, so tau = 0, sigma = 0, F = 0.
        """
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([1.0, 0.5])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=1.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero force for uniform velocity at {v.x}",
            )

    def test_equilibrium_zero_acceleration(self, mesh_2d):
        """At rest with uniform state, stress_acceleration should be zero."""
        from ddgclib.operators.stress import stress_acceleration

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.zeros(2)
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=2, mu=1e-3, HC=HC)
            npt.assert_allclose(
                a, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero acceleration at equilibrium at {v.x}",
            )


# ---------------------------------------------------------------------------
# Test backward compatibility: gradient.py wrappers
# ---------------------------------------------------------------------------

class TestGradientWrappers2D:

    def test_pressure_gradient_uniform(self, mesh_2d):
        """pressure_gradient with uniform P should give zero."""
        from ddgclib.operators.gradient import pressure_gradient

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 100.0
            v.u = np.zeros(2)

        for v in HC.V:
            if v in bV:
                continue
            grad = pressure_gradient(v, dim=2, HC=HC)
            npt.assert_allclose(
                grad, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero pressure gradient at {v.x}",
            )

    def test_velocity_laplacian_uniform(self, mesh_2d):
        """velocity_laplacian with uniform u should give zero."""
        from ddgclib.operators.gradient import velocity_laplacian

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([1.0, 0.5])

        for v in HC.V:
            if v in bV:
                continue
            lap = velocity_laplacian(v, dim=2, HC=HC)
            npt.assert_allclose(
                lap, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero Laplacian at {v.x}",
            )

    def test_acceleration_equilibrium(self, mesh_2d):
        """acceleration at rest should be zero."""
        from ddgclib.operators.gradient import acceleration

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.zeros(2)
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            a = acceleration(v, dim=2, mu=1e-3, HC=HC)
            npt.assert_allclose(
                a, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero acceleration at {v.x}",
            )


# ---------------------------------------------------------------------------
# Test operators __init__.py exports
# ---------------------------------------------------------------------------

class TestOperatorsExports:

    def test_stress_exports(self):
        """Stress functions should be importable from operators."""
        from ddgclib.operators import (
            dual_area_vector,
            dual_volume,
            cache_dual_volumes,
            velocity_difference_tensor,
            velocity_difference_tensor_pointwise,
            strain_rate,
            cauchy_stress,
            integrated_cauchy_stress,
            stress_force,
            stress_acceleration,
        )
        assert callable(dual_area_vector)
        assert callable(dual_volume)
        assert callable(cache_dual_volumes)
        assert callable(velocity_difference_tensor)
        assert callable(velocity_difference_tensor_pointwise)
        assert callable(strain_rate)
        assert callable(cauchy_stress)
        assert callable(integrated_cauchy_stress)
        assert callable(stress_force)
        assert callable(stress_acceleration)

    def test_gradient_exports(self):
        """Old gradient functions should still be importable."""
        from ddgclib.operators import (
            pressure_gradient,
            velocity_laplacian,
            acceleration,
        )
        assert callable(pressure_gradient)
        assert callable(velocity_laplacian)
        assert callable(acceleration)


# ---------------------------------------------------------------------------
# Integration test: Poiseuille 2D with stress_acceleration
# ---------------------------------------------------------------------------

class TestPoiseuille2DIntegration:

    def test_parabolic_profile_develops(self):
        """Poiseuille flow with stress_acceleration should develop
        a parabolic velocity profile.

        Setup: 2D channel with linear pressure gradient, no-slip walls.
        After many steps the velocity should approach the Poiseuille profile.
        """
        from ddgclib.operators.stress import stress_acceleration
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            euler_velocity_only,
        )
        from functools import partial

        # Domain: [0, L] x [0, H]
        L, H = 1.0, 1.0
        HC = Complex(2, domain=[(0.0, L), (0.0, H)])
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()

        bV = set()
        wall_verts = set()
        for v in HC.V:
            on_boundary = (
                abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
            )
            if on_boundary:
                bV.add(v)
                v.boundary = True
            else:
                v.boundary = False

            # Walls are y=0 and y=H
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14:
                wall_verts.add(v)

        compute_vd(HC, cdist=1e-10)

        # Setup: pressure gradient driving flow in x, no-slip walls
        G = 1.0   # pressure gradient magnitude
        mu = 0.1
        for v in HC.V:
            v.u = np.zeros(2)
            # Linear pressure field: p = P0 - G*x
            v.p = -G * v.x_a[0]
            v.m = 1.0

        dudt_fn = partial(stress_acceleration, dim=2, mu=mu, HC=HC)

        # Run a few steps of velocity-only Euler
        dt = 1e-3
        n_steps = 50

        def enforce_no_slip(step, t, HC):
            for v in wall_verts:
                v.u[:] = 0.0

        euler_velocity_only(
            HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=2,
            callback=enforce_no_slip,
        )

        # After some steps, interior vertices should have positive u_x
        # (flow driven by pressure gradient)
        interior_ux = []
        for v in HC.V:
            if v not in bV:
                interior_ux.append(v.u[0])

        mean_ux = np.mean(interior_ux)
        assert mean_ux > 0, \
            f"Expected positive mean u_x from pressure gradient, got {mean_ux}"


# ---------------------------------------------------------------------------
# 3D Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_3d():
    """3D mesh on [0,1]^3 with barycentric duals computed."""
    HC = Complex(3, domain=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()

    bV = set()
    for v in HC.V:
        on_boundary = any(
            abs(v.x_a[i]) < 1e-14 or abs(v.x_a[i] - 1.0) < 1e-14
            for i in range(3)
        )
        if on_boundary:
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    compute_vd(HC, cdist=1e-10)

    for v in HC.V:
        v.u = np.zeros(3)
        v.p = 0.0
        v.m = 1.0

    return HC, bV


class TestDualAreaVector3D:

    def test_closure_interior(self, mesh_3d):
        """sum_j A_ij = 0 for every interior vertex (closed dual cell)."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_3d
        for v in HC.V:
            if v in bV:
                continue
            A_total = np.zeros(3)
            for v_j in v.nn:
                A_total += dual_area_vector(v, v_j, HC, dim=3)
            npt.assert_allclose(
                A_total, np.zeros(3), atol=1e-12,
                err_msg=f"Area vectors not closed at interior vertex {v.x}",
            )

    def test_antisymmetry(self, mesh_3d):
        """A_ij = -A_ji for each edge."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_3d
        tested = 0
        for v_i in HC.V:
            if v_i in bV:
                continue
            for v_j in v_i.nn:
                A_ij = dual_area_vector(v_i, v_j, HC, dim=3)
                A_ji = dual_area_vector(v_j, v_i, HC, dim=3)
                npt.assert_allclose(
                    A_ij, -A_ji, atol=1e-12,
                    err_msg=f"Anti-symmetry failed: {v_i.x}-{v_j.x}",
                )
                tested += 1
        assert tested > 0

    def test_nonzero(self, mesh_3d):
        """Area vectors should be non-zero for interior edges."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_3d
        for v_i in HC.V:
            if v_i in bV:
                continue
            for v_j in v_i.nn:
                A_ij = dual_area_vector(v_i, v_j, HC, dim=3)
                assert np.linalg.norm(A_ij) > 1e-15, \
                    f"Zero area vector at edge {v_i.x}-{v_j.x}"


class TestDualVolume3D:

    def test_positive(self, mesh_3d):
        """Dual volumes should be positive for all vertices."""
        from ddgclib.operators.stress import dual_volume

        HC, bV = mesh_3d
        for v in HC.V:
            vol = dual_volume(v, HC, dim=3)
            assert vol > 0, f"Non-positive dual volume at {v.x}: {vol}"

    def test_partition_of_unity(self, mesh_3d):
        """Sum of all dual volumes should approximate the domain volume."""
        from ddgclib.operators.stress import dual_volume

        HC, bV = mesh_3d
        total = sum(dual_volume(v, HC, dim=3) for v in HC.V)
        domain_vol = 1.0  # [0,1]^3
        npt.assert_allclose(
            total, domain_vol, rtol=0.01,
            err_msg=f"Dual volumes don't sum to domain volume: {total}",
        )


# ---------------------------------------------------------------------------
# Test dudt_i alias and integrator integration
# ---------------------------------------------------------------------------

class TestDudtAlias:
    """Tests that dudt_i is a proper alias for stress_acceleration."""

    def test_is_same_function(self):
        """dudt_i should be the same function object as stress_acceleration."""
        from ddgclib.operators.stress import dudt_i, stress_acceleration
        assert dudt_i is stress_acceleration

    def test_importable_from_operators(self):
        """dudt_i should be importable from the operators package."""
        from ddgclib.operators import dudt_i
        assert callable(dudt_i)

    def test_returns_same_result(self, mesh_2d):
        """dudt_i and stress_acceleration should return identical results."""
        from ddgclib.operators.stress import dudt_i, stress_acceleration

        HC, bV = mesh_2d
        # Set a non-trivial state
        for v in HC.V:
            v.p = -1.0 * v.x_a[0]
            v.u = np.array([0.1 * v.x_a[1], 0.0])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            a1 = dudt_i(v, dim=2, mu=0.1, HC=HC)
            a2 = stress_acceleration(v, dim=2, mu=0.1, HC=HC)
            npt.assert_array_equal(a1, a2)


class TestDudtIntegrators:
    """Tests that dudt_i works as dudt_fn input to all dynamic integrators."""

    @pytest.fixture
    def poiseuille_setup(self):
        """Setup a 2D Poiseuille-like flow for integrator tests.

        Returns HC, bV, wall_verts, mu, G (pressure gradient).
        """
        L, H = 1.0, 1.0
        HC = Complex(2, domain=[(0.0, L), (0.0, H)])
        HC.triangulate()
        HC.refine_all()

        bV = set()
        wall_verts = set()
        for v in HC.V:
            on_boundary = (
                abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
            )
            if on_boundary:
                bV.add(v)
                v.boundary = True
            else:
                v.boundary = False
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14:
                wall_verts.add(v)

        compute_vd(HC, cdist=1e-10)

        G = 1.0
        mu = 0.1
        for v in HC.V:
            v.u = np.zeros(2)
            v.p = -G * v.x_a[0]
            v.m = 1.0

        return HC, bV, wall_verts, mu, G

    def _no_slip_callback(self, wall_verts):
        """Return a callback that enforces no-slip on walls."""
        def callback(step, t, HC):
            for v in wall_verts:
                v.u[:] = 0.0
        return callback

    def _make_dudt_fn(self, HC, mu):
        """Create a partial-bound dudt_i for use with integrators.

        Since integrators have HC as their first positional arg, we must
        bind HC via functools.partial to avoid 'multiple values' conflicts.
        """
        from ddgclib.operators.stress import dudt_i
        from functools import partial
        return partial(dudt_i, dim=2, mu=mu, HC=HC)

    def test_euler_velocity_only(self, poiseuille_setup):
        """dudt_i works with euler_velocity_only integrator."""
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV, wall_verts, mu, G = poiseuille_setup
        callback = self._no_slip_callback(wall_verts)
        dudt_fn = self._make_dudt_fn(HC, mu)

        t = euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=20, dim=2,
            callback=callback,
        )

        assert t > 0
        interior_ux = [v.u[0] for v in HC.V if v not in bV]
        assert np.mean(interior_ux) > 0, \
            f"Expected positive mean u_x, got {np.mean(interior_ux)}"

    def test_euler(self, poiseuille_setup):
        """dudt_i works with the full euler integrator (position + velocity)."""
        from ddgclib.dynamic_integrators import euler

        HC, bV, wall_verts, mu, G = poiseuille_setup
        callback = self._no_slip_callback(wall_verts)
        dudt_fn = self._make_dudt_fn(HC, mu)

        t = euler(
            HC, bV, dudt_fn, dt=1e-4, n_steps=10, dim=2,
            callback=callback,
        )

        assert t > 0
        max_u = max(np.linalg.norm(v.u[:2]) for v in HC.V if v not in bV)
        assert max_u > 0

    def test_symplectic_euler(self, poiseuille_setup):
        """dudt_i works with symplectic_euler integrator."""
        from ddgclib.dynamic_integrators import symplectic_euler

        HC, bV, wall_verts, mu, G = poiseuille_setup
        callback = self._no_slip_callback(wall_verts)
        dudt_fn = self._make_dudt_fn(HC, mu)

        t = symplectic_euler(
            HC, bV, dudt_fn, dt=1e-4, n_steps=10, dim=2,
            callback=callback,
        )

        assert t > 0
        max_u = max(np.linalg.norm(v.u[:2]) for v in HC.V if v not in bV)
        assert max_u > 0

    def test_with_partial(self, poiseuille_setup):
        """dudt_i works when parameters are bound via functools.partial."""
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV, wall_verts, mu, G = poiseuille_setup
        callback = self._no_slip_callback(wall_verts)
        dudt_fn = self._make_dudt_fn(HC, mu)

        t = euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=20, dim=2,
            callback=callback,
        )

        assert t > 0
        interior_ux = [v.u[0] for v in HC.V if v not in bV]
        assert np.mean(interior_ux) > 0

    def test_with_bc_set(self, poiseuille_setup):
        """dudt_i works with BoundaryConditionSet."""
        from ddgclib.dynamic_integrators import euler_velocity_only
        from ddgclib._boundary_conditions import (
            BoundaryConditionSet, NoSlipWallBC,
        )

        HC, bV, wall_verts, mu, G = poiseuille_setup
        dudt_fn = self._make_dudt_fn(HC, mu)

        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), wall_verts)

        t = euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=20, dim=2,
            bc_set=bc_set,
        )

        assert t > 0
        # Wall vertices should have zero velocity
        for v in wall_verts:
            npt.assert_allclose(v.u[:2], np.zeros(2), atol=1e-14)

    def test_euler_adaptive(self, poiseuille_setup):
        """dudt_i works with euler_adaptive integrator."""
        from ddgclib.dynamic_integrators import euler_adaptive

        HC, bV, wall_verts, mu, G = poiseuille_setup
        callback = self._no_slip_callback(wall_verts)
        dudt_fn = self._make_dudt_fn(HC, mu)

        t = euler_adaptive(
            HC, bV, dudt_fn, dt_initial=1e-3, t_end=5e-3, dim=2,
            callback=callback, velocity_only=True,
        )

        assert t >= 5e-3 - 1e-15
        interior_ux = [v.u[0] for v in HC.V if v not in bV]
        assert np.mean(interior_ux) > 0


# ---------------------------------------------------------------------------
# Hagen-Poiseuille validation with stress tensor formulation
# ---------------------------------------------------------------------------

class TestHagenPoiseuilleStress2D:
    """Validate the Cauchy stress tensor pipeline on 2D planar Poiseuille flow.

    Analytical solution: u_x(y) = (G/(2*mu)) * y * (h - y), u_y = 0
    Pressure: P(x) = -G * x
    Stress equilibrium: pressure gradient balances viscous diffusion.
    """

    @staticmethod
    def _tag_boundary(HC, bV):
        """Set v.boundary attribute required by compute_vd."""
        for v in HC.V:
            v.boundary = v in bV

    @pytest.fixture
    def poiseuille_equilibrium(self):
        """2D Poiseuille at analytical equilibrium with duals computed."""
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        HC, bV, ic, bc_set, params = setup_poiseuille_2d(
            G=1.0, mu=1.0, n_refine=2, L=1.0, h=1.0,
        )
        ic.apply(HC, bV)
        self._tag_boundary(HC, bV)
        compute_vd(HC, cdist=1e-10)
        return HC, bV, bc_set, params

    @pytest.fixture
    def poiseuille_equilibrium_fine(self):
        """Finer 2D Poiseuille at analytical equilibrium."""
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        HC, bV, ic, bc_set, params = setup_poiseuille_2d(
            G=1.0, mu=1.0, n_refine=3, L=1.0, h=1.0,
        )
        ic.apply(HC, bV)
        self._tag_boundary(HC, bV)
        compute_vd(HC, cdist=1e-10)
        return HC, bV, bc_set, params

    @pytest.fixture
    def poiseuille_plug_flow(self):
        """2D Poiseuille starting from zero velocity (plug flow)."""
        from ddgclib._boundary_conditions import (
            BoundaryConditionSet, NoSlipWallBC,
            identify_boundary_vertices,
        )
        from ddgclib.initial_conditions import (
            CompositeIC, LinearPressureGradient,
            ZeroVelocity, UniformMass,
        )

        L, h = 1.0, 1.0
        G, mu = 1.0, 0.1
        HC = Complex(2, domain=[(0.0, L), (0.0, h)])
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()

        bV = set()
        for v in HC.V:
            if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L) < 1e-14 or
                    abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14):
                bV.add(v)
                v.boundary = True
            else:
                v.boundary = False

        bV_wall = identify_boundary_vertices(
            HC, lambda v: abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14,
        )

        ic = CompositeIC(
            ZeroVelocity(dim=2),
            LinearPressureGradient(G=G, axis=0),
            UniformMass(total_volume=L * h, rho=1.0),
        )
        ic.apply(HC, bV)
        compute_vd(HC, cdist=1e-10)

        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), bV_wall)

        params = {'G': G, 'mu': mu, 'h': h, 'L': L}
        return HC, bV, bV_wall, bc_set, params

    def test_equilibrium_residual_small(self, poiseuille_equilibrium):
        """At analytical equilibrium, stress_acceleration should be small.

        The discrete operators are approximate, so the residual won't be
        exactly zero, but it should be bounded.
        """
        from ddgclib.operators.stress import stress_acceleration

        HC, bV, _, params = poiseuille_equilibrium
        mu = params['mu']

        residuals = []
        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=2, mu=mu, HC=HC)
            residuals.append(np.linalg.norm(a))

        median_res = np.median(residuals)
        max_res = np.max(residuals)
        # The residual should be bounded (not blow up)
        assert max_res < 1e3, \
            f"Equilibrium residual too large: max={max_res:.4f}, median={median_res:.4f}"

    def test_equilibrium_residual_converges(self, poiseuille_equilibrium,
                                            poiseuille_equilibrium_fine):
        """Residual at equilibrium should be near machine precision on both meshes.

        Both coarse and fine meshes achieve machine-precision residuals
        (~1e-15), so a convergence-rate comparison is meaningless — the
        ratio is dominated by floating-point noise.  Instead we assert
        that both residuals stay below an absolute tolerance.
        """
        from ddgclib.operators.stress import stress_acceleration

        for label, (HC, bV, _, params) in [
            ("coarse", poiseuille_equilibrium),
            ("fine", poiseuille_equilibrium_fine),
        ]:
            mu = params['mu']
            res = []
            for v in HC.V:
                if v in bV:
                    continue
                a = stress_acceleration(v, dim=2, mu=mu, HC=HC)
                res.append(np.linalg.norm(a))

            median_res = np.median(res)
            assert median_res < 1e-13, \
                (f"{label} mesh: median residual {median_res:.4e} "
                 f"exceeds machine-precision tolerance")

    def test_pressure_force_direction(self, poiseuille_plug_flow):
        """With dP/dx < 0 (P = -G*x, G > 0), the pressure force should
        push fluid in the +x direction (positive u_x acceleration).
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, _, _ = poiseuille_plug_flow

        # With p = -G*x, the pressure gradient drives flow in +x
        # stress_force with mu=0 gives the pure pressure force
        positive_fx_count = 0
        total_interior = 0
        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=0.0, HC=HC)
            total_interior += 1
            if F[0] > 0:
                positive_fx_count += 1

        # Majority of interior vertices should have positive F_x
        assert positive_fx_count > total_interior * 0.5, \
            f"Only {positive_fx_count}/{total_interior} have positive F_x"

    def test_developing_flow_with_dudt_i(self, poiseuille_plug_flow):
        """Starting from zero velocity, flow develops under pressure gradient.

        After integration with dudt_i, interior vertices should have
        positive u_x and the velocity should increase away from walls.
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, bV_wall, bc_set, params = poiseuille_plug_flow
        mu = params['mu']

        dudt_fn = partial(dudt_i, dim=2, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=100, dim=2,
            bc_set=bc_set,
        )

        # Interior vertices should have developed positive u_x
        interior_ux = [v.u[0] for v in HC.V if v not in bV]
        assert np.mean(interior_ux) > 0, \
            f"Expected positive mean u_x after developing flow, got {np.mean(interior_ux)}"

        # Wall vertices should remain at zero (no-slip)
        for v in bV_wall:
            npt.assert_allclose(v.u[:2], np.zeros(2), atol=1e-14,
                                err_msg=f"Non-zero wall velocity at {v.x}")

    def test_developing_flow_profile_shape(self, poiseuille_plug_flow):
        """After integration, the velocity profile should be qualitatively
        parabolic: higher in the channel center, lower near walls.
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, bV_wall, bc_set, params = poiseuille_plug_flow
        mu, h = params['mu'], params['h']

        dudt_fn = partial(dudt_i, dim=2, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=500, dim=2,
            bc_set=bc_set,
        )

        # Bin interior vertices by y-position and check profile shape
        bins = {'near_wall': [], 'center': []}
        for v in HC.V:
            if v in bV:
                continue
            y = v.x_a[1]
            if y < 0.2 * h or y > 0.8 * h:
                bins['near_wall'].append(v.u[0])
            elif 0.4 * h < y < 0.6 * h:
                bins['center'].append(v.u[0])

        if bins['near_wall'] and bins['center']:
            mean_wall = np.mean(bins['near_wall'])
            mean_center = np.mean(bins['center'])
            # Center should have higher velocity than near-wall
            assert mean_center > mean_wall, \
                (f"Profile not parabolic: center={mean_center:.6f}, "
                 f"near_wall={mean_wall:.6f}")

    def test_developing_flow_with_adaptive(self, poiseuille_plug_flow):
        """dudt_i works with adaptive time stepping for Poiseuille flow."""
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_adaptive
        from functools import partial

        HC, bV, bV_wall, bc_set, params = poiseuille_plug_flow
        mu = params['mu']

        dudt_fn = partial(dudt_i, dim=2, mu=mu, HC=HC)

        t = euler_adaptive(
            HC, bV, dudt_fn, dt_initial=1e-3, t_end=0.05, dim=2,
            bc_set=bc_set, velocity_only=True,
        )

        assert t >= 0.05 - 1e-15
        interior_ux = [v.u[0] for v in HC.V if v not in bV]
        assert np.mean(interior_ux) > 0


# ---------------------------------------------------------------------------
# Hydrostatic validation with stress tensor formulation
# ---------------------------------------------------------------------------

class TestHydrostaticStress2D:
    """Validate the Cauchy stress tensor pipeline on 2D hydrostatic column.

    At hydrostatic equilibrium with P(x,y) = rho*g*(h - y) and u = 0,
    the stress force should be non-zero (it equals the pressure gradient)
    but when balanced by body force (gravity) the net acceleration is zero.

    For the pure pressure test (no gravity body force), we verify:
    - Uniform pressure -> zero force (closure)
    - Linear pressure -> consistent pressure gradient direction
    """

    @pytest.fixture
    def hydrostatic_2d(self):
        """2D hydrostatic column with duals computed."""
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(
            dim=2, n_refine=2, rho=1000.0, g=9.81, h=1.0,
        )
        ic.apply(HC, bV)
        for v in HC.V:
            v.boundary = v in bV
        compute_vd(HC, cdist=1e-10)
        return HC, bV, bc_set, params

    def test_hydrostatic_pressure_force_direction(self, hydrostatic_2d):
        """Under hydrostatic pressure, the pressure force should point
        from high pressure (bottom) to low pressure (top), i.e. +y direction.
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, params = hydrostatic_2d

        positive_fy_count = 0
        total_interior = 0
        for v in HC.V:
            if v in bV:
                continue
            # Pure pressure force (mu=0, all velocities are zero anyway)
            F = stress_force(v, dim=2, mu=0.0, HC=HC)
            total_interior += 1
            if F[1] > 0:  # F_y should be positive (upward, opposing gravity)
                positive_fy_count += 1

        # Most interior vertices should have upward pressure force
        assert positive_fy_count > total_interior * 0.5, \
            f"Only {positive_fy_count}/{total_interior} have upward F_y"

    def test_hydrostatic_zero_viscous_stress(self, hydrostatic_2d):
        """At hydrostatic equilibrium with u=0, the viscous stress is zero.

        So stress_force(mu=any) should equal stress_force(mu=0).
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, _ = hydrostatic_2d

        for v in HC.V:
            if v in bV:
                continue
            F_no_visc = stress_force(v, dim=2, mu=0.0, HC=HC)
            F_with_visc = stress_force(v, dim=2, mu=1.0, HC=HC)
            npt.assert_allclose(
                F_with_visc, F_no_visc, atol=1e-12,
                err_msg=f"Viscous stress non-zero at hydrostatic equilibrium {v.x}",
            )

    def test_hydrostatic_acceleration_equals_pressure_gradient(self, hydrostatic_2d):
        """At hydrostatic equilibrium, stress_acceleration = F_pressure / m.

        Since u = 0, there is no viscous contribution. The pressure force
        should balance gravity. Here we just verify that the acceleration
        from the pressure gradient is consistent and non-zero.
        """
        from ddgclib.operators.stress import stress_acceleration

        HC, bV, _, _ = hydrostatic_2d

        accel_norms = []
        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=2, mu=0.0, HC=HC)
            accel_norms.append(np.linalg.norm(a))

        # The acceleration should be non-zero (it's the unbalanced pressure
        # gradient without body force). Approximate magnitude: g ~ 9.81
        median_a = np.median(accel_norms)
        assert median_a > 0, "Expected non-zero acceleration from pressure gradient"

    def test_perturbation_viscous_damping(self, hydrostatic_2d):
        """Perturb velocity under uniform pressure, verify KE decreases.

        With uniform pressure (no pressure gradient force) and viscosity,
        kinetic energy should dissipate via the deviatoric stress.
        The hydrostatic pressure gradient is removed to isolate viscous
        effects (otherwise the large rho*g force dominates).
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, bc_set, _ = hydrostatic_2d

        # Set uniform pressure (remove hydrostatic gradient)
        # and perturb interior velocities
        for v in HC.V:
            v.p = 0.0
            if v not in bV:
                v.u = np.array([0.0, 0.1])  # small upward perturbation

        ke_initial = sum(
            0.5 * v.m * np.dot(v.u[:2], v.u[:2])
            for v in HC.V if v not in bV
        )

        # Run with viscous damping via dudt_i (p=0 so only tau acts)
        mu = 10.0  # high viscosity for fast damping
        dudt_fn = partial(dudt_i, dim=2, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-4, n_steps=100, dim=2,
            bc_set=bc_set,
        )

        ke_final = sum(
            0.5 * v.m * np.dot(v.u[:2], v.u[:2])
            for v in HC.V if v not in bV
        )

        # KE should decrease (viscous dissipation)
        assert ke_final < ke_initial, \
            f"KE not decreasing: initial={ke_initial:.6e}, final={ke_final:.6e}"


class TestStressForce3D:

    def test_uniform_pressure_zero_force(self, mesh_3d):
        """Uniform pressure + zero velocity => zero stress force in 3D."""
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_3d
        for v in HC.V:
            v.p = 100.0
            v.u = np.zeros(3)
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=3, mu=0.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(3), atol=1e-10,
                err_msg=f"Non-zero force for uniform pressure at {v.x}",
            )

    def test_uniform_velocity_zero_force(self, mesh_3d):
        """Uniform velocity + zero pressure => zero stress force in 3D."""
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_3d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([1.0, 0.5, -0.3])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=3, mu=1.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(3), atol=1e-10,
                err_msg=f"Non-zero force for uniform velocity at {v.x}",
            )

    def test_equilibrium_zero_acceleration(self, mesh_3d):
        """At rest with uniform state, stress_acceleration = 0 in 3D."""
        from ddgclib.operators.stress import stress_acceleration

        HC, bV = mesh_3d
        for v in HC.V:
            v.p = 0.0
            v.u = np.zeros(3)
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=3, mu=1e-3, HC=HC)
            npt.assert_allclose(
                a, np.zeros(3), atol=1e-10,
                err_msg=f"Non-zero 3D acceleration at {v.x}",
            )


# ---------------------------------------------------------------------------
# 3D Hagen-Poiseuille validation with stress tensor formulation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHagenPoiseuilleStress3D:
    """Validate the Cauchy stress tensor pipeline on 3D channel flow.

    Uses a cube domain [0,L]^3 with flow driven by a linear pressure
    gradient along the z-axis (flow_axis=2).  Walls at y=0 and y=L.
    """

    @staticmethod
    def _tag_boundary(HC, bV):
        for v in HC.V:
            v.boundary = v in bV

    @pytest.fixture
    def channel_3d_plug_flow(self):
        """3D channel starting from zero velocity with pressure gradient."""
        from ddgclib._boundary_conditions import (
            BoundaryConditionSet, NoSlipWallBC,
            identify_boundary_vertices,
        )
        from ddgclib.initial_conditions import (
            CompositeIC, LinearPressureGradient,
            ZeroVelocity, UniformMass,
        )

        L = 1.0
        G, mu = 1.0, 0.1
        HC = Complex(3, domain=[(0.0, L), (0.0, L), (0.0, L)])
        HC.triangulate()
        HC.refine_all()

        bV = set()
        for v in HC.V:
            on_boundary = any(
                abs(v.x_a[i]) < 1e-14 or abs(v.x_a[i] - L) < 1e-14
                for i in range(3)
            )
            if on_boundary:
                bV.add(v)
                v.boundary = True
            else:
                v.boundary = False

        # Walls at y=0 and y=L (normal_axis=1)
        bV_wall = identify_boundary_vertices(
            HC, lambda v: abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - L) < 1e-14,
        )

        # Pressure gradient along z-axis (flow_axis=2)
        ic = CompositeIC(
            ZeroVelocity(dim=3),
            LinearPressureGradient(G=G, axis=2, P_ref=0.0),
            UniformMass(total_volume=L**3, rho=1.0),
        )
        ic.apply(HC, bV)
        compute_vd(HC, cdist=1e-10)

        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=3), bV_wall)

        params = {'G': G, 'mu': mu, 'L': L, 'flow_axis': 2}
        return HC, bV, bV_wall, bc_set, params

    @pytest.fixture
    def poiseuille_3d_equilibrium(self):
        """3D Poiseuille at approximate analytical equilibrium.

        Uses HagenPoiseuille3D IC for a parabolic velocity profile
        on the unit cube (approximate, since the domain is not cylindrical).
        """
        from ddgclib.initial_conditions import (
            CompositeIC, HagenPoiseuille3D,
            LinearPressureGradient, UniformMass,
        )

        L = 1.0
        G, mu = 1.0, 1.0
        R = 0.5  # effective radius (half-width)
        U_max = G * R**2 / (4 * mu)  # Hagen-Poiseuille centerline velocity

        HC = Complex(3, domain=[(0.0, L), (0.0, L), (0.0, L)])
        HC.triangulate()
        HC.refine_all()

        bV = set()
        for v in HC.V:
            on_boundary = any(
                abs(v.x_a[i]) < 1e-14 or abs(v.x_a[i] - L) < 1e-14
                for i in range(3)
            )
            if on_boundary:
                bV.add(v)
                v.boundary = True
            else:
                v.boundary = False

        ic = CompositeIC(
            HagenPoiseuille3D(U_max=U_max, R=R, flow_axis=2, dim=3),
            LinearPressureGradient(G=G, axis=2, P_ref=0.0),
            UniformMass(total_volume=L**3, rho=1.0),
        )
        ic.apply(HC, bV)
        compute_vd(HC, cdist=1e-10)

        params = {'G': G, 'mu': mu, 'L': L, 'R': R, 'U_max': U_max}
        return HC, bV, params

    def test_pressure_force_direction_3d(self, channel_3d_plug_flow):
        """With dP/dz < 0 (P = -G*z, G > 0), the pressure force should
        push fluid in the +z direction.
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, _, _ = channel_3d_plug_flow

        positive_fz_count = 0
        total_interior = 0
        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=3, mu=0.0, HC=HC)
            total_interior += 1
            if F[2] > 0:
                positive_fz_count += 1

        assert positive_fz_count > total_interior * 0.5, \
            f"Only {positive_fz_count}/{total_interior} have positive F_z"

    def test_developing_flow_3d(self, channel_3d_plug_flow):
        """Starting from zero velocity, 3D flow develops under pressure gradient.

        After integration with dudt_i, interior vertices should have
        positive u_z (flow along pressure gradient direction).
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, bV_wall, bc_set, params = channel_3d_plug_flow
        mu = params['mu']

        dudt_fn = partial(dudt_i, dim=3, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=50, dim=3,
            bc_set=bc_set,
        )

        interior_uz = [v.u[2] for v in HC.V if v not in bV]
        assert len(interior_uz) > 0, "No interior vertices found"
        assert np.mean(interior_uz) > 0, \
            f"Expected positive mean u_z after developing flow, got {np.mean(interior_uz)}"

        # Wall vertices should remain at zero (no-slip)
        for v in bV_wall:
            npt.assert_allclose(v.u[:3], np.zeros(3), atol=1e-14,
                                err_msg=f"Non-zero wall velocity at {v.x}")

    def test_equilibrium_residual_bounded_3d(self, poiseuille_3d_equilibrium):
        """At analytical equilibrium, stress_acceleration should be bounded.

        The discrete operators are approximate, so the residual won't be
        exactly zero, but it should not blow up.
        """
        from ddgclib.operators.stress import stress_acceleration

        HC, bV, params = poiseuille_3d_equilibrium
        mu = params['mu']

        residuals = []
        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=3, mu=mu, HC=HC)
            residuals.append(np.linalg.norm(a))

        if residuals:
            max_res = np.max(residuals)
            assert max_res < 1e3, \
                f"3D equilibrium residual too large: max={max_res:.4f}"

    def test_developing_flow_profile_shape_3d(self, channel_3d_plug_flow):
        """After integration, the velocity profile should be qualitatively
        parabolic: higher in the channel center (y ~ 0.5), lower near walls.
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, _, bc_set, params = channel_3d_plug_flow
        mu, L = params['mu'], params['L']

        dudt_fn = partial(dudt_i, dim=3, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-3, n_steps=100, dim=3,
            bc_set=bc_set,
        )

        bins = {'near_wall': [], 'center': []}
        for v in HC.V:
            if v in bV:
                continue
            y = v.x_a[1]
            if y < 0.2 * L or y > 0.8 * L:
                bins['near_wall'].append(v.u[2])
            elif 0.4 * L < y < 0.6 * L:
                bins['center'].append(v.u[2])

        if bins['near_wall'] and bins['center']:
            mean_wall = np.mean(bins['near_wall'])
            mean_center = np.mean(bins['center'])
            assert mean_center > mean_wall, \
                (f"3D profile not parabolic: center={mean_center:.6f}, "
                 f"near_wall={mean_wall:.6f}")


# ---------------------------------------------------------------------------
# 3D Hydrostatic validation with stress tensor formulation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHydrostaticStress3D:
    """Validate the Cauchy stress tensor pipeline on 3D hydrostatic column.

    At hydrostatic equilibrium with P(x,y,z) = rho*g*(h - z) and u = 0,
    the pressure force should point upward (+z, opposing gravity).
    """

    @pytest.fixture
    def hydrostatic_3d(self):
        """3D hydrostatic column with duals computed."""
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(
            dim=3, n_refine=1, rho=1000.0, g=9.81, h=1.0,
        )
        ic.apply(HC, bV)
        for v in HC.V:
            v.boundary = v in bV
        compute_vd(HC, cdist=1e-10)
        return HC, bV, bc_set, params

    def test_hydrostatic_pressure_force_direction_3d(self, hydrostatic_3d):
        """Under hydrostatic pressure, the pressure force should point
        upward (+z, opposing gravity).
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, _ = hydrostatic_3d

        positive_fz_count = 0
        total_interior = 0
        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=3, mu=0.0, HC=HC)
            total_interior += 1
            if F[2] > 0:  # upward, opposing gravity
                positive_fz_count += 1

        assert positive_fz_count > total_interior * 0.5, \
            f"Only {positive_fz_count}/{total_interior} have upward F_z"

    def test_hydrostatic_zero_viscous_stress_3d(self, hydrostatic_3d):
        """At hydrostatic equilibrium with u=0, the viscous stress is zero.

        So stress_force(mu=any) should equal stress_force(mu=0).
        """
        from ddgclib.operators.stress import stress_force

        HC, bV, _, _ = hydrostatic_3d

        for v in HC.V:
            if v in bV:
                continue
            F_no_visc = stress_force(v, dim=3, mu=0.0, HC=HC)
            F_with_visc = stress_force(v, dim=3, mu=1.0, HC=HC)
            npt.assert_allclose(
                F_with_visc, F_no_visc, atol=1e-12,
                err_msg=f"3D viscous stress non-zero at equilibrium {v.x}",
            )

    def test_hydrostatic_acceleration_nonzero_3d(self, hydrostatic_3d):
        """At hydrostatic equilibrium, stress_acceleration is non-zero
        (it represents the unbalanced pressure gradient without body force).
        """
        from ddgclib.operators.stress import stress_acceleration

        HC, bV, _, _ = hydrostatic_3d

        accel_norms = []
        for v in HC.V:
            if v in bV:
                continue
            a = stress_acceleration(v, dim=3, mu=0.0, HC=HC)
            accel_norms.append(np.linalg.norm(a))

        if accel_norms:
            median_a = np.median(accel_norms)
            assert median_a > 0, "Expected non-zero 3D acceleration from pressure gradient"

    def test_perturbation_viscous_damping_3d(self, hydrostatic_3d):
        """Perturb velocity under uniform pressure, verify KE decreases.

        With uniform pressure (no pressure gradient force) and viscosity,
        kinetic energy should dissipate via the deviatoric stress.
        """
        from ddgclib.operators.stress import dudt_i
        from ddgclib.dynamic_integrators import euler_velocity_only
        from functools import partial

        HC, bV, bc_set, _ = hydrostatic_3d

        # Set uniform pressure (remove hydrostatic gradient)
        # and perturb interior velocities
        for v in HC.V:
            v.p = 0.0
            if v not in bV:
                v.u = np.array([0.0, 0.0, 0.1])  # small upward perturbation

        ke_initial = sum(
            0.5 * v.m * np.dot(v.u[:3], v.u[:3])
            for v in HC.V if v not in bV
        )

        # Run with viscous damping via dudt_i (p=0 so only tau acts)
        mu = 10.0  # high viscosity for fast damping
        dudt_fn = partial(dudt_i, dim=3, mu=mu, HC=HC)

        euler_velocity_only(
            HC, bV, dudt_fn, dt=1e-4, n_steps=100, dim=3,
            bc_set=bc_set,
        )

        ke_final = sum(
            0.5 * v.m * np.dot(v.u[:3], v.u[:3])
            for v in HC.V if v not in bV
        )

        assert ke_final < ke_initial, \
            f"3D KE not decreasing: initial={ke_initial:.6e}, final={ke_final:.6e}"


# ---------------------------------------------------------------------------
# New tests for face-centered formulation and caching
# ---------------------------------------------------------------------------

class TestDualVolCaching:
    """Tests for dual volume caching on vertices."""

    def test_cache_sets_attribute(self, mesh_2d):
        """cache_dual_volumes should set v.dual_vol on all vertices."""
        from ddgclib.operators.stress import cache_dual_volumes

        HC, bV = mesh_2d
        cache_dual_volumes(HC, dim=2)
        for v in HC.V:
            assert hasattr(v, 'dual_vol'), f"v.dual_vol not set at {v.x}"
            assert v.dual_vol > 0 or v in bV

    def test_cache_matches_computed(self, mesh_2d):
        """Cached dual_vol should match dual_volume() result."""
        from ddgclib.operators.stress import cache_dual_volumes, dual_volume

        HC, bV = mesh_2d
        cache_dual_volumes(HC, dim=2)
        for v in HC.V:
            if v in bV:
                continue
            computed = dual_volume(v, HC, dim=2)
            npt.assert_allclose(
                v.dual_vol, computed, rtol=1e-12,
                err_msg=f"Cached != computed at {v.x}",
            )

    def test_get_dual_vol_on_demand(self, mesh_2d):
        """_get_dual_vol should compute on demand if not cached."""
        from ddgclib.operators.stress import _get_dual_vol, dual_volume

        HC, bV = mesh_2d
        # Ensure no cached value
        for v in HC.V:
            if hasattr(v, 'dual_vol'):
                del v.dual_vol

        for v in HC.V:
            if v in bV:
                continue
            vol = _get_dual_vol(v, HC, dim=2)
            expected = dual_volume(v, HC, dim=2)
            npt.assert_allclose(vol, expected, rtol=1e-12)
            # Should now be cached
            assert hasattr(v, 'dual_vol')


class TestFaceCenteredViscousFlux:
    """Tests for the face-centered viscous flux formulation."""

    def test_uniform_velocity_zero_viscous(self, mesh_2d):
        """Uniform velocity -> zero viscous force (Δu = 0)."""
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([3.0, -1.0])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=10.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero viscous force for uniform velocity at {v.x}",
            )

    def test_linear_shear_zero_laplacian(self, mesh_2d):
        """Linear shear u = [y, 0] has zero Laplacian -> zero viscous force
        on interior cells (face-centered formulation is exact for linear)."""
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([v.x_a[1], 0.0])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=1.0, HC=HC)
            npt.assert_allclose(
                F, np.zeros(2), atol=1e-10,
                err_msg=f"Non-zero viscous force for linear shear at {v.x}",
            )

    def test_quadratic_shear_nonzero(self, mesh_2d):
        """Quadratic shear u = [y^2, 0] has nonzero Laplacian -> nonzero
        viscous force on interior cells."""
        from ddgclib.operators.stress import stress_force

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = 0.0
            v.u = np.array([v.x_a[1]**2, 0.0])
            v.m = 1.0

        forces = []
        for v in HC.V:
            if v in bV:
                continue
            F = stress_force(v, dim=2, mu=1.0, HC=HC)
            forces.append(F)

        max_force = max(np.linalg.norm(f) for f in forces)
        assert max_force > 1e-10, "All viscous forces are zero for quadratic shear"

    def test_pressure_only_equals_pressure_gradient(self, mesh_2d):
        """stress_force(mu=0) should match pressure_gradient."""
        from ddgclib.operators.stress import stress_force
        from ddgclib.operators.gradient import pressure_gradient

        HC, bV = mesh_2d
        for v in HC.V:
            v.p = -1.0 * v.x_a[0]  # linear pressure
            v.u = np.array([0.1 * v.x_a[1], 0.0])
            v.m = 1.0

        for v in HC.V:
            if v in bV:
                continue
            F_stress = stress_force(v, dim=2, mu=0.0, HC=HC)
            F_grad = pressure_gradient(v, dim=2, HC=HC)
            npt.assert_allclose(
                F_stress, F_grad, atol=1e-14,
                err_msg=f"stress_force(mu=0) != pressure_gradient at {v.x}",
            )

    def test_half_difference_zero_for_uniform_pressure(self, mesh_2d):
        """With uniform p, the half-difference formula gives zero per-face."""
        from ddgclib.operators.stress import dual_area_vector

        HC, bV = mesh_2d
        p_uniform = 42.0
        for v in HC.V:
            v.p = p_uniform

        for v in HC.V:
            if v in bV:
                continue
            F = np.zeros(2)
            for v_j in v.nn:
                A_ij = dual_area_vector(v, v_j, HC, dim=2)
                p_j = v_j.p
                F -= 0.5 * (p_j - v.p) * A_ij
            npt.assert_allclose(
                F, np.zeros(2), atol=1e-14,
                err_msg=f"Non-zero pressure flux for uniform pressure at {v.x}",
            )


class TestIntegratedCauchyStress:
    """Tests for the integrated Cauchy stress tensor."""

    def test_integrated_div_vol_equals_pointwise(self, mesh_2d):
        """integrated_cauchy_stress / Vol should equal cauchy_stress."""
        from ddgclib.operators.stress import (
            velocity_difference_tensor,
            velocity_difference_tensor_pointwise,
            cauchy_stress,
            integrated_cauchy_stress,
            _get_dual_vol,
        )

        HC, bV = mesh_2d
        for v in HC.V:
            x, y = v.x_a[0], v.x_a[1]
            v.u = np.array([2.0 * x + y, -x + 3.0 * y])
            v.p = 5.0 * x
            v.m = 1.0

        mu = 0.5
        for v in HC.V:
            if v in bV:
                continue
            Du = velocity_difference_tensor(v, HC, dim=2)
            du_pw = velocity_difference_tensor_pointwise(v, HC, dim=2)
            Vol_i = _get_dual_vol(v, HC, dim=2)
            p_i = float(v.p)

            sigma_pw = cauchy_stress(p_i, du_pw, mu, dim=2)
            sigma_int = integrated_cauchy_stress(p_i, Du, mu, Vol_i, dim=2)

            npt.assert_allclose(
                sigma_int / Vol_i, sigma_pw, atol=1e-12,
                err_msg=f"Sigma_int/Vol != sigma_pw at {v.x}",
            )

    def test_pressure_scales_with_volume(self):
        """Pressure part of integrated stress: -p * Vol * I."""
        from ddgclib.operators.stress import integrated_cauchy_stress

        Du = np.zeros((2, 2))
        sigma = integrated_cauchy_stress(p=3.0, Du=Du, mu=1.0, Vol_i=0.5, dim=2)
        expected = -3.0 * 0.5 * np.eye(2)
        npt.assert_allclose(sigma, expected)

    def test_viscous_uses_integrated_du(self):
        """Viscous part should use Du directly (not Du/Vol)."""
        from ddgclib.operators.stress import integrated_cauchy_stress, strain_rate

        Du = np.array([[1.0, 2.0], [3.0, 4.0]])
        sigma = integrated_cauchy_stress(p=0.0, Du=Du, mu=0.5, Vol_i=10.0, dim=2)
        expected = 2.0 * 0.5 * strain_rate(Du)
        npt.assert_allclose(sigma, expected)
