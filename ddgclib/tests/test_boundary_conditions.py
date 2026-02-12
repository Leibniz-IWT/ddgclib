"""Tests for ddgclib._boundary_conditions module."""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex

from ddgclib._boundary_conditions import (
    BoundaryConditionSet,
    DirichletPressureBC,
    DirichletVelocityBC,
    NeumannBC,
    NoSlipWallBC,
    OutletDeleteBC,
    identify_boundary_vertices,
    identify_cube_boundaries,
)


# Fixtures

@pytest.fixture
def mesh_2d():
    """2D mesh on [0, 1]^2 with fields initialized."""
    HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
    HC.triangulate()
    HC.refine_all()
    bV = set()
    for v in HC.V:
        if (abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14 or
                abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14):
            bV.add(v)
        v.u = np.array([1.0, 0.5])
        v.P = 100.0
        v.m = 1.0
    return HC, bV


# Boundary identification tests

class TestIdentifyBoundaryVertices:
    def test_identifies_left_wall(self, mesh_2d):
        HC, _ = mesh_2d
        left = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-14)
        assert len(left) > 0
        for v in left:
            assert abs(v.x_a[0]) < 1e-14

    def test_empty_for_impossible_criterion(self, mesh_2d):
        HC, _ = mesh_2d
        result = identify_boundary_vertices(HC, lambda v: v.x_a[0] > 100)
        assert len(result) == 0


class TestIdentifyCubeBoundaries:
    def test_finds_all_boundary_verts(self, mesh_2d):
        HC, bV_expected = mesh_2d
        bV = identify_cube_boundaries(HC, 0.0, 1.0, dim=2)
        assert bV == bV_expected

    def test_no_interior_verts(self, mesh_2d):
        HC, _ = mesh_2d
        bV = identify_cube_boundaries(HC, 0.0, 1.0, dim=2)
        for v in bV:
            on_boundary = any(
                abs(v.x_a[i]) < 1e-13 or abs(v.x_a[i] - 1.0) < 1e-13
                for i in range(2)
            )
            assert on_boundary


# Concrete BC tests

class TestNoSlipWallBC:
    def test_zeros_velocity(self, mesh_2d):
        HC, bV = mesh_2d
        bc = NoSlipWallBC(dim=2)
        count = bc.apply(HC, dt=0.01, target_vertices=bV)
        assert count == len(bV)
        for v in bV:
            npt.assert_array_equal(v.u, np.zeros(2))

    def test_leaves_interior_alone(self, mesh_2d):
        HC, bV = mesh_2d
        bc = NoSlipWallBC(dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in HC.V:
            if v not in bV:
                assert v.u[0] == 1.0  # Unchanged


class TestDirichletVelocityBC:
    def test_constant_value(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletVelocityBC(np.array([0.0, 0.1]), dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_array_equal(v.u, np.array([0.0, 0.1]))

    def test_callable_value(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletVelocityBC(lambda v: np.array([v.x_a[1], 0.0]), dim=2)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_allclose(v.u[0], v.x_a[1])
            assert v.u[1] == 0.0


class TestDirichletPressureBC:
    def test_constant_pressure(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletPressureBC(value=200.0)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            assert v.P == 200.0

    def test_callable_pressure(self, mesh_2d):
        HC, bV = mesh_2d
        bc = DirichletPressureBC(value=lambda v: v.x_a[0] * 100)
        bc.apply(HC, dt=0.01, target_vertices=bV)
        for v in bV:
            npt.assert_allclose(v.P, v.x_a[0] * 100)


class TestNeumannBC:
    def test_zero_gradient_copies_neighbor(self, mesh_2d):
        """Zero Neumann should copy the nearest interior neighbor's value."""
        HC, bV = mesh_2d
        # Set a known field pattern
        for v in HC.V:
            v.P = v.x_a[0] * 10  # Linear in x

        bc = NeumannBC(field_name='P', flux_value=0.0)
        bc.apply(HC, dt=0.01, target_vertices=bV)

        # Boundary vertices should now have their nearest interior neighbor's P
        for v in bV:
            interior_nbs = [nb for nb in v.nn if nb not in bV]
            if interior_nbs:
                # Should be close to nearest interior neighbor value
                nb = min(interior_nbs, key=lambda nb: np.linalg.norm(v.x_a - nb.x_a))
                npt.assert_allclose(v.P, nb.P, atol=1e-10)


class TestOutletDeleteBC:
    def test_deletes_past_outlet(self):
        HC = Complex(1, domain=[(0.0, 10.0)])
        HC.triangulate()
        HC.refine_all()
        HC.refine_all()
        initial_count = sum(1 for _ in HC.V)
        bc = OutletDeleteBC(outlet_pos=8.0, axis=0)
        deleted = bc.apply(HC, dt=0.01)
        final_count = sum(1 for _ in HC.V)
        assert deleted > 0
        assert final_count < initial_count
        # No vertex should be past outlet
        for v in HC.V:
            assert v.x_a[0] < 8.0


# BoundaryConditionSet tests

class TestBoundaryConditionSet:
    def test_applies_all_in_order(self, mesh_2d):
        HC, bV = mesh_2d
        left = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0]) < 1e-14)
        right = identify_boundary_vertices(HC, lambda v: abs(v.x_a[0] - 1.0) < 1e-14)

        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), left)
        bc_set.add(DirichletVelocityBC(np.array([2.0, 0.0]), dim=2), right)

        diagnostics = bc_set.apply_all(HC, bV, dt=0.01)

        assert len(diagnostics) == 2
        for v in left:
            npt.assert_array_equal(v.u, np.zeros(2))
        for v in right:
            npt.assert_array_equal(v.u, np.array([2.0, 0.0]))

    def test_method_chaining(self, mesh_2d):
        HC, bV = mesh_2d
        bc_set = (BoundaryConditionSet()
                  .add(NoSlipWallBC(dim=2), bV)
                  .add(DirichletPressureBC(0.0), bV))
        assert len(bc_set._bcs) == 2

    def test_default_applies_to_all_bV(self, mesh_2d):
        HC, bV = mesh_2d
        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2))  # No specific vertices

        bc_set.apply_all(HC, bV, dt=0.01)
        for v in bV:
            npt.assert_array_equal(v.u, np.zeros(2))
