"""Tests for ddgclib.operators package."""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex


# Registry tests

class TestMethodRegistry:
    def test_register_and_retrieve(self):
        from ddgclib.operators._registry import MethodRegistry
        reg = MethodRegistry("test")
        reg.register("foo", lambda x: x + 1)
        assert reg["foo"](5) == 6

    def test_unknown_key_raises(self):
        from ddgclib.operators._registry import MethodRegistry
        reg = MethodRegistry("test")
        with pytest.raises(KeyError, match="Unknown test method"):
            reg["nonexistent"]

    def test_available(self):
        from ddgclib.operators._registry import MethodRegistry
        reg = MethodRegistry("test")
        reg.register("a", lambda: None)
        reg.register("b", lambda: None)
        assert set(reg.available()) == {"a", "b"}

    def test_contains(self):
        from ddgclib.operators._registry import MethodRegistry
        reg = MethodRegistry("test")
        reg.register("a", lambda: None)
        assert "a" in reg
        assert "b" not in reg


# Backward compatibility: _method_wrappers still works

class TestMethodWrappersShim:
    def test_import_classes(self):
        from ddgclib._method_wrappers import (
            Curvature_i, Curvature_ijk,
            Area_i, Area_ijk, Area,
            Volume, Volume_i,
        )
        # These should be the same classes from operators/
        from ddgclib.operators.curvature import Curvature_i as C_i
        from ddgclib.operators.area import Area_i as A_i
        from ddgclib.operators.volume import Volume as V
        assert Curvature_i is C_i
        assert Area_i is A_i
        assert Volume is V

    def test_pre_instantiated_singletons(self):
        from ddgclib._method_wrappers import curvature_i, area_i, volume
        assert curvature_i is not None
        assert area_i is not None
        assert volume is not None


# Operator imports from operators/__init__.py

class TestOperatorsImports:
    def test_all_exports(self):
        from ddgclib.operators import (
            Curvature_i, Curvature_ijk,
            Area_i, Area_ijk, Area, DualArea_i,
            Volume, Volume_i,
            pressure_gradient, velocity_laplacian, acceleration,
            MethodRegistry,
        )
        # Just verify they imported without error
        assert callable(pressure_gradient)
        assert callable(velocity_laplacian)
        assert callable(acceleration)


# Gradient operator tests (requires mesh with computed duals)

@pytest.fixture
def simple_2d_mesh_with_duals():
    """2D mesh on [0,1]^2 with barycentric duals computed."""
    from hyperct.ddg import compute_vd

    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()

    # Identify boundaries
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    # Compute duals
    compute_vd(HC, cdist=1e-10)

    # Initialize fields
    for v in HC.V:
        v.u = np.zeros(2)
        v.P = 0.0
        v.m = 1.0

    return HC, bV


class TestPressureGradient:
    def test_uniform_pressure_zero_gradient(self, simple_2d_mesh_with_duals):
        """Uniform pressure should produce zero gradient."""
        from ddgclib.operators.gradient import pressure_gradient

        HC, bV = simple_2d_mesh_with_duals
        for v in HC.V:
            v.P = 100.0  # Uniform

        for v in HC.V:
            if v not in bV:
                grad = pressure_gradient(v, dim=2, HC=HC)
                npt.assert_allclose(grad, np.zeros(2), atol=1e-10,
                                    err_msg=f"Non-zero gradient at {v.x}")


class TestVelocityLaplacian:
    def test_uniform_velocity_zero_laplacian(self, simple_2d_mesh_with_duals):
        """Uniform velocity should produce zero Laplacian."""
        from ddgclib.operators.gradient import velocity_laplacian

        HC, bV = simple_2d_mesh_with_duals
        for v in HC.V:
            v.u = np.array([1.0, 0.5])  # Uniform

        for v in HC.V:
            if v not in bV:
                lap = velocity_laplacian(v, dim=2, HC=HC)
                npt.assert_allclose(lap, np.zeros(2), atol=1e-10,
                                    err_msg=f"Non-zero Laplacian at {v.x}")


class TestAcceleration:
    def test_zero_at_equilibrium(self, simple_2d_mesh_with_duals):
        """At rest with uniform pressure, acceleration should be zero."""
        from ddgclib.operators.gradient import acceleration

        HC, bV = simple_2d_mesh_with_duals
        for v in HC.V:
            v.P = 0.0
            v.u = np.zeros(2)
            v.m = 1.0

        for v in HC.V:
            if v not in bV:
                a = acceleration(v, dim=2, mu=1e-3, HC=HC)
                npt.assert_allclose(a, np.zeros(2), atol=1e-10,
                                    err_msg=f"Non-zero acceleration at {v.x}")


# DualArea_i test

class TestDualArea:
    def test_positive_area(self, simple_2d_mesh_with_duals):
        from ddgclib.operators.area import DualArea_i

        HC, bV = simple_2d_mesh_with_duals
        dual_area = DualArea_i()

        for v in HC.V:
            if v not in bV:
                a = dual_area(v)
                assert a > 0, f"Dual area should be positive, got {a} at {v.x}"
