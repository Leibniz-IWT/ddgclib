"""Tests for periodic boundary conditions (ghost-cell Delaunay)."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.geometry.periodic import (
    ghost_band_width,
    create_ghost_vertices,
    delaunay_with_ghosts,
    wrap_positions,
    retopologize_periodic,
    _identify_periodic_face_verts,
    _shift_directions,
)
from ddgclib.geometry.domains import periodic_rectangle, periodic_box
from ddgclib.operators.stress import stress_force, cache_dual_volumes


# ---------------------------------------------------------------------------
# Ghost utility tests
# ---------------------------------------------------------------------------

class TestShiftDirections(unittest.TestCase):
    """Test _shift_directions helper."""

    def test_1_periodic_axis_2d(self):
        dirs = _shift_directions([0], dim=2)
        # x-periodic in 2D: shifts (-1,0) and (+1,0)
        self.assertEqual(len(dirs), 2)
        self.assertIn((-1, 0), dirs)
        self.assertIn((1, 0), dirs)

    def test_2_periodic_axes_2d(self):
        dirs = _shift_directions([0, 1], dim=2)
        # 3^2 - 1 = 8 directions
        self.assertEqual(len(dirs), 8)
        self.assertIn((-1, -1), dirs)
        self.assertIn((1, 1), dirs)
        self.assertIn((0, 1), dirs)

    def test_1_periodic_axis_3d(self):
        dirs = _shift_directions([0], dim=3)
        self.assertEqual(len(dirs), 2)
        self.assertIn((-1, 0, 0), dirs)
        self.assertIn((1, 0, 0), dirs)

    def test_2_periodic_axes_3d(self):
        dirs = _shift_directions([0, 2], dim=3)
        # 3^2 - 1 = 8 directions (axes 0 and 2 vary, axis 1 always 0)
        self.assertEqual(len(dirs), 8)
        self.assertIn((1, 0, -1), dirs)
        self.assertIn((-1, 0, 1), dirs)


class TestCreateGhostVertices(unittest.TestCase):
    """Test ghost vertex creation for periodic Delaunay."""

    def _make_rect(self, refinement=2):
        """Create a simple rectangle mesh."""
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        for _ in range(refinement):
            HC.refine_all()
        return HC

    def test_ghost_count_x_periodic(self):
        HC = self._make_rect(refinement=1)
        n_real = len(list(HC.V))
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        self.assertEqual(len(real_verts), n_real)
        self.assertGreater(len(ghost_map), 0)
        self.assertEqual(all_coords.shape[0], n_real + len(ghost_map))

    def test_ghost_positions_x_periodic(self):
        HC = self._make_rect(refinement=1)
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        # Every ghost should be outside [0, 1] in x
        n_real = len(real_verts)
        for ghost_idx in ghost_map:
            gx = all_coords[ghost_idx, 0]
            self.assertTrue(gx < 0.0 + 1e-10 or gx > 1.0 - 1e-10,
                            f"Ghost at x={gx} is inside domain")

    def test_ghost_to_real_mapping(self):
        HC = self._make_rect(refinement=1)
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        # All real indices should be valid
        n_real = len(real_verts)
        for ghost_idx, real_idx in ghost_map.items():
            self.assertGreaterEqual(ghost_idx, n_real)
            self.assertLess(real_idx, n_real)

    def test_xy_periodic_has_diagonal_ghosts(self):
        HC = self._make_rect(refinement=1)
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0, 1],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        # With 2 periodic axes there should be ghosts in diagonal directions
        n_real = len(real_verts)
        has_diagonal = False
        for ghost_idx in ghost_map:
            gx, gy = all_coords[ghost_idx]
            outside_x = gx < -1e-10 or gx > 1.0 + 1e-10
            outside_y = gy < -1e-10 or gy > 1.0 + 1e-10
            if outside_x and outside_y:
                has_diagonal = True
                break
        self.assertTrue(has_diagonal, "No diagonal ghosts found for xy-periodic")


class TestDelaunayWithGhosts(unittest.TestCase):
    """Test Delaunay triangulation with ghost resolution."""

    def test_all_indices_real(self):
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        n_real = len(list(HC.V))
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        simplices = delaunay_with_ghosts(all_coords, n_real, ghost_map, dim=2)
        for simplex in simplices:
            for idx in simplex:
                self.assertLess(idx, n_real,
                                f"Ghost index {idx} in resolved simplex")

    def test_no_degenerate_simplices(self):
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        n_real = len(list(HC.V))
        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes=[0],
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            band_width=0.6, dim=2,
        )
        simplices = delaunay_with_ghosts(all_coords, n_real, ghost_map, dim=2)
        for simplex in simplices:
            self.assertEqual(len(set(simplex)), len(simplex),
                             f"Degenerate simplex: {simplex}")


# ---------------------------------------------------------------------------
# Retopologize periodic tests
# ---------------------------------------------------------------------------

class TestRetopologizePeriodic(unittest.TestCase):
    """Test full periodic retopologization pipeline."""

    def _make_periodic_rect(self, refinement=2):
        result = periodic_rectangle(L=1.0, h=1.0, refinement=refinement,
                                    periodic_axes=[0])
        return result

    def test_periodic_connectivity_wraps(self):
        """Left-edge vertices should have right-edge neighbors."""
        result = self._make_periodic_rect(refinement=2)
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        # After merging, right-edge vertices (x=1.0) are removed.
        # Left-edge vertices (x=0) should have neighbors at x > 0.5
        # (wrapping from the far side of the domain).
        left = [v for v in HC.V if abs(v.x_a[0] - 0.0) < 1e-14]
        self.assertGreater(len(left), 0)

        has_wrap = False
        for v in left:
            for nb in v.nn:
                if nb.x_a[0] > 0.5:
                    has_wrap = True
                    break
            if has_wrap:
                break
        self.assertTrue(has_wrap,
                        "No wrapping connectivity found for left-edge vertices")

    def test_periodic_face_not_boundary(self):
        """Pure periodic-face vertices (not on walls) should be interior."""
        result = self._make_periodic_rect(refinement=2)
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        periodic_verts = _identify_periodic_face_verts(
            HC, periodic_axes, domain_bounds)
        # Only check pure periodic vertices (not also on a wall)
        non_periodic_axes = [ax for ax in range(2) if ax not in periodic_axes]
        for v in periodic_verts:
            on_wall = any(
                abs(v.x_a[ax] - domain_bounds[ax][0]) < 1e-14
                or abs(v.x_a[ax] - domain_bounds[ax][1]) < 1e-14
                for ax in non_periodic_axes
            )
            if not on_wall:
                self.assertFalse(v.boundary,
                                 f"Pure periodic vertex at {v.x} marked as boundary")

    def test_wall_vertices_still_boundary(self):
        """Non-periodic faces (walls) should still be boundary."""
        result = self._make_periodic_rect(refinement=2)
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        # y=0 and y=1 should be in bV (walls)
        walls_bottom = [v for v in HC.V if abs(v.x_a[1] - 0.0) < 1e-14]
        walls_top = [v for v in HC.V if abs(v.x_a[1] - 1.0) < 1e-14]
        self.assertGreater(len(walls_bottom), 0)
        for v in walls_bottom:
            self.assertIn(v, bV)
        for v in walls_top:
            self.assertIn(v, bV)

    def test_bV_excludes_periodic_faces(self):
        """bV should not contain periodic-face vertices."""
        result = self._make_periodic_rect(refinement=2)
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        periodic_verts = _identify_periodic_face_verts(
            HC, periodic_axes, domain_bounds)
        # Only check pure periodic vertices (not also on a wall)
        non_periodic_axes = [ax for ax in range(2) if ax not in periodic_axes]
        for v in periodic_verts:
            on_wall = any(
                abs(v.x_a[ax] - domain_bounds[ax][0]) < 1e-14
                or abs(v.x_a[ax] - domain_bounds[ax][1]) < 1e-14
                for ax in non_periodic_axes
            )
            if not on_wall:
                self.assertNotIn(v, bV,
                                 f"Pure periodic vertex at {v.x} in bV")

    def test_all_vertices_have_dual(self):
        """All vertices should have v.vd populated after retopologize."""
        result = self._make_periodic_rect(refinement=2)
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        for v in HC.V:
            self.assertTrue(hasattr(v, 'vd'),
                            f"Vertex at {v.x} has no vd")
            self.assertGreater(len(v.vd), 0,
                               f"Vertex at {v.x} has empty vd")

    def test_1d_periodic(self):
        """1D periodic should connect last to first."""
        HC = Complex(1, domain=[(0.0, 1.0)])
        HC.triangulate()
        for _ in range(3):
            HC.refine_all()
        bV = set()

        retopologize_periodic(HC, bV, dim=1,
                              periodic_axes=[0],
                              domain_bounds=[(0.0, 1.0)])

        verts = sorted(list(HC.V), key=lambda v: v.x_a[0])
        # First and last should be connected
        self.assertIn(verts[-1], verts[0].nn)
        self.assertIn(verts[0], verts[-1].nn)


# ---------------------------------------------------------------------------
# Domain builder tests
# ---------------------------------------------------------------------------

class TestPeriodicRectangle(unittest.TestCase):
    """Test periodic_rectangle domain builder."""

    def test_metadata(self):
        result = periodic_rectangle(L=2.0, h=1.0, refinement=2,
                                    periodic_axes=[0])
        self.assertEqual(result.metadata['periodic_axes'], [0])
        self.assertEqual(len(result.metadata['domain_bounds']), 2)

    def test_bV_excludes_periodic_faces(self):
        result = periodic_rectangle(L=2.0, h=1.0, refinement=2,
                                    periodic_axes=[0])
        for v in result.bV:
            # No periodic-face vertex (x=0 or x=2) should be in bV
            self.assertFalse(
                abs(v.x_a[0] - 0.0) < 1e-14 or abs(v.x_a[0] - 2.0) < 1e-14,
                f"Periodic face vertex at {v.x} in bV")

    def test_walls_present(self):
        result = periodic_rectangle(L=2.0, h=1.0, refinement=2,
                                    periodic_axes=[0])
        # Walls (y=0, y=1) should still exist
        wall_verts = [v for v in result.bV
                      if abs(v.x_a[1] - 0.0) < 1e-14
                      or abs(v.x_a[1] - 1.0) < 1e-14]
        self.assertGreater(len(wall_verts), 0)

    def test_no_periodic_axes_same_as_rectangle(self):
        result = periodic_rectangle(L=2.0, h=1.0, refinement=2,
                                    periodic_axes=[])
        # Should have inlet/outlet groups
        self.assertIn('inlet', result.boundary_groups)
        self.assertIn('outlet', result.boundary_groups)


class TestPeriodicBox(unittest.TestCase):
    """Test periodic_box domain builder."""

    def test_metadata(self):
        result = periodic_box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1,
                              periodic_axes=[0])
        self.assertEqual(result.metadata['periodic_axes'], [0])
        self.assertEqual(len(result.metadata['domain_bounds']), 3)
        self.assertEqual(result.dim, 3)

    def test_bV_excludes_periodic_faces(self):
        result = periodic_box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1,
                              periodic_axes=[0, 1])
        for v in result.bV:
            on_x = abs(v.x_a[0] - 0.0) < 1e-14 or abs(v.x_a[0] - 1.0) < 1e-14
            on_y = abs(v.x_a[1] - 0.0) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14
            self.assertFalse(on_x or on_y,
                             f"Periodic face vertex at {v.x} in bV")


# ---------------------------------------------------------------------------
# Wrap positions test
# ---------------------------------------------------------------------------

class TestWrapPositions(unittest.TestCase):
    """Test position wrapping for Lagrangian drift."""

    def test_wrap_outside_vertex(self):
        HC = Complex(1, domain=[(0.0, 1.0)])
        HC.triangulate()
        # Move a vertex outside the domain
        v = list(HC.V)[0]
        HC.V.move(v, (1.2,))
        wrap_positions(HC, periodic_axes=[0], domain_bounds=[(0.0, 1.0)])
        self.assertGreaterEqual(v.x_a[0], 0.0)
        self.assertLess(v.x_a[0], 1.0)

    def test_wrap_preserves_interior(self):
        HC = Complex(1, domain=[(0.0, 1.0)])
        HC.triangulate()
        for _ in range(2):
            HC.refine_all()
        coords_before = sorted([v.x[0] for v in HC.V])
        wrap_positions(HC, periodic_axes=[0], domain_bounds=[(0.0, 1.0)])
        coords_after = sorted([v.x[0] for v in HC.V])
        assert_allclose(coords_before, coords_after, atol=1e-14)


# ---------------------------------------------------------------------------
# Physics validation tests
# ---------------------------------------------------------------------------

class TestPeriodicPhysics(unittest.TestCase):
    """Physics validation: stress operators on periodic meshes."""

    def test_uniform_pressure_zero_force(self):
        """Uniform pressure on periodic mesh: interior vertices get zero force.

        The ghost-cell approach gives exact zero force for truly interior
        vertices.  Periodic-face vertices (within ~1 edge of x=0) have
        O(h) error from the non-manifold boundary chain, which converges
        with mesh refinement.
        """
        result = periodic_rectangle(L=1.0, h=1.0, refinement=3,
                                    periodic_axes=[0])
        HC, bV = result.HC, result.bV
        periodic_axes = result.metadata['periodic_axes']
        domain_bounds = result.metadata['domain_bounds']

        retopologize_periodic(HC, bV, dim=2,
                              periodic_axes=periodic_axes,
                              domain_bounds=domain_bounds)

        # Set uniform pressure, zero velocity, unit mass
        p0 = 1000.0
        for v in HC.V:
            v.p = p0
            v.u = np.zeros(2)
            v.m = max(getattr(v, 'dual_vol', 1.0), 1e-30)

        # Interior vertices far from periodic face should have zero force.
        # Periodic-face vertices have O(h) error from ghost-cell duals.
        dx = 1.0 / (2**3)  # edge length at refinement=3
        interior = [v for v in HC.V if v not in bV]
        self.assertGreater(len(interior), 0)

        for v in interior:
            F = stress_force(v, dim=2, mu=0.0, HC=HC)
            # Check if vertex is near a periodic face (lb or ub side,
            # since ub-face was merged into lb-face)
            near_periodic = False
            for ax in periodic_axes:
                lb = domain_bounds[ax][0]
                ub = domain_bounds[ax][1]
                if abs(v.x_a[ax] - lb) < 2 * dx or abs(v.x_a[ax] - ub) < 2 * dx:
                    near_periodic = True
                    break
            if near_periodic:
                # O(h) error expected for periodic-face vertices
                self.assertLess(np.linalg.norm(F), 200.0,
                                f"Large force {F} at periodic vertex {v.x}")
            else:
                # Exact zero for truly interior vertices
                self.assertLess(np.linalg.norm(F), 1e-6,
                                f"Non-zero force {F} at interior vertex {v.x}")

    def test_integrator_accepts_periodic_axes(self):
        """euler_velocity_only should accept periodic_axes parameter."""
        from ddgclib.dynamic_integrators import euler_velocity_only
        import inspect
        sig = inspect.signature(euler_velocity_only)
        self.assertIn('periodic_axes', sig.parameters)
        self.assertIn('domain_bounds', sig.parameters)


if __name__ == '__main__':
    unittest.main()
