"""Tests for ddgclib.geometry.domains.

Covers every domain builder, the projection engine, boundary group
identification, and integration with compute_vd / BoundaryConditionSet.
"""

import math
import unittest

import numpy as np

from ddgclib.geometry.domains import (
    DomainResult,
    rectangle,
    l_shape,
    disk,
    annulus,
    box,
    cylinder_volume,
    pipe,
    ball,
    cube_to_disk,
    cube_to_sphere,
    DISTRIBUTION_LAWS,
    identify_face_groups,
    identify_radial_boundary,
    identify_all_boundary,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _assert_structural(test: unittest.TestCase, result: DomainResult) -> None:
    """Shared structural assertions for every DomainResult."""
    HC = result.HC
    test.assertGreater(HC.V.size(), 0, "Mesh has no vertices")
    for v in HC.V:
        test.assertGreater(len(v.nn), 0, f"Vertex {v.x} has no neighbours")
    test.assertGreater(len(result.bV), 0, "bV is empty")
    # bV must be a subset of HC.V
    for v in result.bV:
        test.assertIn(v, HC.V)
    # Every group vertex must be in bV
    for name, group in result.boundary_groups.items():
        for v in group:
            test.assertIn(v, result.bV,
                          f"Group '{name}' vertex not in bV")
    # tag_boundaries should work
    result.tag_boundaries()
    for v in HC.V:
        if v in result.bV:
            test.assertTrue(v.boundary, f"Expected boundary=True for {v.x}")
        else:
            test.assertFalse(v.boundary, f"Expected boundary=False for {v.x}")
    # summary should not raise
    s = result.summary()
    test.assertIsInstance(s, str)


# ── DomainResult ─────────────────────────────────────────────────────────

class TestDomainResult(unittest.TestCase):
    """Tests for the DomainResult dataclass."""

    def test_summary(self):
        r = rectangle(L=1.0, h=1.0, refinement=1)
        s = r.summary()
        self.assertIn("DomainResult", s)
        self.assertIn("vertices", s)

    def test_tag_boundaries(self):
        r = rectangle(L=1.0, h=1.0, refinement=1)
        r.tag_boundaries()
        for v in r.HC.V:
            self.assertIsInstance(v.boundary, bool)


# ── Rectangle ────────────────────────────────────────────────────────────

class TestRectangle(unittest.TestCase):

    def test_structural(self):
        r = rectangle(L=2.0, h=1.0, refinement=2)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = rectangle(L=2.0, h=1.0, refinement=2)
        for key in ('walls', 'inlet', 'outlet', 'bottom_wall', 'top_wall'):
            self.assertIn(key, r.boundary_groups)
            self.assertGreater(len(r.boundary_groups[key]), 0, f"Empty group: {key}")

    def test_vertex_bounds(self):
        r = rectangle(L=3.0, h=2.0, refinement=1, origin=(1.0, 0.5))
        for v in r.HC.V:
            self.assertGreaterEqual(v.x_a[0], 1.0 - 1e-12)
            self.assertLessEqual(v.x_a[0], 4.0 + 1e-12)
            self.assertGreaterEqual(v.x_a[1], 0.5 - 1e-12)
            self.assertLessEqual(v.x_a[1], 2.5 + 1e-12)

    def test_dim(self):
        r = rectangle()
        self.assertEqual(r.dim, 2)

    def test_metadata_volume(self):
        r = rectangle(L=3.0, h=2.0)
        self.assertAlmostEqual(r.metadata['volume'], 6.0)

    def test_refinement_increases_vertices(self):
        r1 = rectangle(refinement=1)
        r2 = rectangle(refinement=2)
        self.assertGreater(r2.HC.V.size(), r1.HC.V.size())

    def test_flow_axis_1(self):
        """flow_axis=1 should swap inlet/outlet to y-faces."""
        r = rectangle(L=2.0, h=1.0, refinement=1, flow_axis=1)
        self.assertIn('inlet', r.boundary_groups)
        # Inlet should be on the y=0 face (normal_axis=0 for flow_axis=1)
        for v in r.boundary_groups['inlet']:
            self.assertAlmostEqual(v.x_a[1], 0.0, places=12)

    def test_inlet_outlet_positions(self):
        r = rectangle(L=5.0, h=1.0, refinement=1, flow_axis=0)
        for v in r.boundary_groups['inlet']:
            self.assertAlmostEqual(v.x_a[0], 0.0, places=12)
        for v in r.boundary_groups['outlet']:
            self.assertAlmostEqual(v.x_a[0], 5.0, places=12)


# ── L-shape ──────────────────────────────────────────────────────────────

class TestLShape(unittest.TestCase):

    def test_structural(self):
        r = l_shape(refinement=2)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = l_shape(refinement=2)
        for key in ('walls', 'outer', 'notch'):
            self.assertIn(key, r.boundary_groups)

    def test_notch_removed(self):
        """No vertices should exist deep inside the notch region."""
        r = l_shape(L=2.0, h=1.0, notch_L=1.0, notch_h=0.5, refinement=2)
        for v in r.HC.V:
            # Interior of notch: x > 1.0 + epsilon AND y < 0.5 - epsilon
            self.assertFalse(
                v.x_a[0] > 1.0 + 1e-10 and v.x_a[1] < 0.5 - 1e-10,
                f"Vertex {v.x} is inside the notch",
            )


# ── Disk ─────────────────────────────────────────────────────────────────

class TestDisk(unittest.TestCase):

    def test_structural(self):
        r = disk(R=1.0, refinement=2)
        _assert_structural(self, r)

    def test_boundary_at_radius(self):
        R = 1.0
        r = disk(R=R, refinement=2)
        for v in r.boundary_groups['walls']:
            dist = np.linalg.norm(v.x_a[:2])
            self.assertAlmostEqual(dist, R, places=6,
                                   msg=f"Wall vertex {v.x} not at radius {R}")

    def test_interior_inside(self):
        R = 1.5
        r = disk(R=R, refinement=2)
        for v in r.HC.V:
            dist = np.linalg.norm(v.x_a[:2])
            self.assertLessEqual(dist, R + 1e-8,
                                 msg=f"Vertex {v.x} outside radius {R}")

    def test_center_offset(self):
        r = disk(R=0.5, center=(2.0, 3.0), refinement=1)
        cx, cy = 2.0, 3.0
        for v in r.boundary_groups['walls']:
            dist = math.hypot(v.x_a[0] - cx, v.x_a[1] - cy)
            self.assertAlmostEqual(dist, 0.5, places=6)

    def test_metadata_volume(self):
        R = 2.0
        r = disk(R=R, refinement=1)
        self.assertAlmostEqual(r.metadata['volume'], math.pi * R ** 2)

    def test_dim(self):
        r = disk()
        self.assertEqual(r.dim, 2)


# ── Annulus ──────────────────────────────────────────────────────────────

class TestAnnulus(unittest.TestCase):

    def test_structural(self):
        r = annulus(R_outer=1.0, R_inner=0.3, refinement=2)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = annulus(R_outer=1.0, R_inner=0.3, refinement=2)
        self.assertIn('outer_wall', r.boundary_groups)
        self.assertIn('inner_wall', r.boundary_groups)

    def test_no_vertices_inside_inner_radius(self):
        R_inner = 0.3
        r = annulus(R_outer=1.0, R_inner=R_inner, refinement=2)
        for v in r.HC.V:
            dist = np.linalg.norm(v.x_a[:2])
            self.assertGreaterEqual(dist, R_inner - 1e-8,
                                    msg=f"Vertex {v.x} inside inner radius")

    def test_invalid_radii(self):
        with self.assertRaises(ValueError):
            annulus(R_outer=1.0, R_inner=2.0)


# ── Box ──────────────────────────────────────────────────────────────────

class TestBox(unittest.TestCase):

    def test_structural(self):
        r = box(Lx=2.0, Ly=1.0, Lz=1.0, refinement=1)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = box(refinement=1)
        for key in ('walls', 'inlet', 'outlet',
                     'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
            self.assertIn(key, r.boundary_groups)
            self.assertGreater(len(r.boundary_groups[key]), 0, f"Empty group: {key}")

    def test_vertex_bounds(self):
        r = box(Lx=3.0, Ly=2.0, Lz=1.0, refinement=1, origin=(1.0, 0.5, 0.0))
        for v in r.HC.V:
            self.assertGreaterEqual(v.x_a[0], 1.0 - 1e-12)
            self.assertLessEqual(v.x_a[0], 4.0 + 1e-12)
            self.assertGreaterEqual(v.x_a[1], 0.5 - 1e-12)
            self.assertLessEqual(v.x_a[1], 2.5 + 1e-12)
            self.assertGreaterEqual(v.x_a[2], 0.0 - 1e-12)
            self.assertLessEqual(v.x_a[2], 1.0 + 1e-12)

    def test_dim(self):
        r = box(refinement=1)
        self.assertEqual(r.dim, 3)

    def test_metadata_volume(self):
        r = box(Lx=3.0, Ly=2.0, Lz=1.5, refinement=1)
        self.assertAlmostEqual(r.metadata['volume'], 9.0)

    def test_flow_axis(self):
        r = box(refinement=1, flow_axis=2)
        for v in r.boundary_groups['inlet']:
            self.assertAlmostEqual(v.x_a[2], 0.0, places=12)


# ── Cylinder ─────────────────────────────────────────────────────────────

class TestCylinderVolume(unittest.TestCase):

    def test_structural(self):
        r = cylinder_volume(R=0.5, L=1.0, refinement=1)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = cylinder_volume(R=0.5, L=1.0, refinement=1)
        for key in ('walls', 'inlet', 'outlet'):
            self.assertIn(key, r.boundary_groups)
            self.assertGreater(len(r.boundary_groups[key]), 0, f"Empty group: {key}")

    def test_wall_at_radius(self):
        R = 0.5
        r = cylinder_volume(R=R, L=1.0, refinement=1, flow_axis=2)
        for v in r.boundary_groups['walls']:
            dist = math.hypot(v.x_a[0], v.x_a[1])
            self.assertAlmostEqual(dist, R, places=5,
                                   msg=f"Wall vertex not at radius {R}")

    def test_axial_extent(self):
        L = 2.0
        r = cylinder_volume(R=0.5, L=L, refinement=1, flow_axis=2)
        axial = [v.x_a[2] for v in r.HC.V]
        self.assertAlmostEqual(min(axial), 0.0, places=10)
        self.assertAlmostEqual(max(axial), L, places=10)

    def test_interior_within_radius(self):
        R = 0.5
        r = cylinder_volume(R=R, L=1.0, refinement=1, flow_axis=2)
        for v in r.HC.V:
            dist = math.hypot(v.x_a[0], v.x_a[1])
            self.assertLessEqual(dist, R + 1e-8)

    def test_dim(self):
        r = cylinder_volume(refinement=1)
        self.assertEqual(r.dim, 3)

    def test_metadata(self):
        R, L = 0.5, 2.0
        r = cylinder_volume(R=R, L=L, refinement=1)
        self.assertAlmostEqual(r.metadata['volume'], math.pi * R ** 2 * L, places=5)

    def test_flow_axis_0(self):
        r = cylinder_volume(R=0.5, L=1.0, refinement=1, flow_axis=0)
        for v in r.boundary_groups['inlet']:
            self.assertAlmostEqual(v.x_a[0], 0.0, places=10)


# ── Pipe ─────────────────────────────────────────────────────────────────

class TestPipe(unittest.TestCase):

    def test_structural(self):
        r = pipe(R=0.5, L=3.0, refinement=1)
        _assert_structural(self, r)

    def test_boundary_groups(self):
        r = pipe(R=0.5, L=3.0, refinement=1)
        for key in ('walls', 'inlet', 'outlet'):
            self.assertIn(key, r.boundary_groups)

    def test_longer_than_cylinder(self):
        """Pipe should have more vertices than a unit cylinder."""
        r_cyl = cylinder_volume(R=0.5, L=1.0, refinement=1)
        r_pipe = pipe(R=0.5, L=3.0, refinement=1)
        self.assertGreater(r_pipe.HC.V.size(), r_cyl.HC.V.size())


# ── Ball ─────────────────────────────────────────────────────────────────

class TestBall(unittest.TestCase):

    def test_structural(self):
        r = ball(R=1.0, refinement=1)
        _assert_structural(self, r)

    def test_wall_at_radius(self):
        R = 1.0
        r = ball(R=R, refinement=1)
        for v in r.boundary_groups['walls']:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertAlmostEqual(dist, R, places=5,
                                   msg=f"Wall vertex not at radius {R}")

    def test_interior_within_radius(self):
        R = 1.0
        r = ball(R=R, refinement=1)
        for v in r.HC.V:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertLessEqual(dist, R + 1e-8)

    def test_center_offset(self):
        r = ball(R=0.5, center=(1.0, 2.0, 3.0), refinement=1)
        c = np.array([1.0, 2.0, 3.0])
        for v in r.boundary_groups['walls']:
            dist = np.linalg.norm(v.x_a[:3] - c)
            self.assertAlmostEqual(dist, 0.5, places=5)

    def test_dim(self):
        r = ball(refinement=1)
        self.assertEqual(r.dim, 3)

    def test_metadata_volume(self):
        R = 2.0
        r = ball(R=R, refinement=1)
        self.assertAlmostEqual(r.metadata['volume'], (4.0 / 3.0) * math.pi * R ** 3)


# ── Projection engine ───────────────────────────────────────────────────

class TestProjection(unittest.TestCase):

    def test_distribution_laws_registry(self):
        self.assertIn('sinusoidal', DISTRIBUTION_LAWS)
        self.assertIn('linear', DISTRIBUTION_LAWS)
        self.assertIn('power', DISTRIBUTION_LAWS)
        self.assertIn('log', DISTRIBUTION_LAWS)

    def test_cube_to_disk_basic(self):
        from hyperct import Complex
        HC = Complex(2, domain=[(-0.5, 0.5), (-0.5, 0.5)])
        HC.triangulate()
        HC.refine_all()
        bV_wall, bV_int = cube_to_disk(HC, R=1.0)
        self.assertGreater(len(bV_wall), 0)
        # All vertices should be within radius
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:2])
            self.assertLessEqual(dist, 1.0 + 1e-8)

    def test_cube_to_sphere_basic(self):
        from hyperct import Complex
        HC = Complex(3, domain=[(-0.5, 0.5)] * 3)
        HC.triangulate()
        bV_wall, bV_int = cube_to_sphere(HC, R=2.0)
        self.assertGreater(len(bV_wall), 0)
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertLessEqual(dist, 2.0 + 1e-8)

    def test_different_distribution_laws(self):
        """Different laws should produce different vertex distributions."""
        from hyperct import Complex

        positions = {}
        for law_name in ('sinusoidal', 'linear'):
            HC = Complex(2, domain=[(-0.5, 0.5), (-0.5, 0.5)])
            HC.triangulate()
            HC.refine_all()
            cube_to_disk(HC, R=1.0, distr_law=law_name)
            positions[law_name] = sorted(
                [tuple(v.x_a[:2]) for v in HC.V]
            )
        # Distributions should differ
        self.assertNotEqual(positions['sinusoidal'], positions['linear'])


# ── Boundary helpers ────────────────────────────────────────────────────

class TestBoundaryHelpers(unittest.TestCase):

    def test_identify_face_groups(self):
        from hyperct import Complex
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        groups = identify_face_groups(HC, {
            'left': (0, 0.0),
            'right': (0, 1.0),
        })
        self.assertGreater(len(groups['left']), 0)
        self.assertGreater(len(groups['right']), 0)
        for v in groups['left']:
            self.assertAlmostEqual(v.x_a[0], 0.0)

    def test_identify_all_boundary(self):
        from hyperct import Complex
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        HC.triangulate()
        HC.refine_all()
        bV = identify_all_boundary(HC, [0.0, 0.0], [1.0, 1.0])
        self.assertGreater(len(bV), 0)
        # Interior vertices should not be in bV
        interior = [v for v in HC.V if v not in bV]
        self.assertGreater(len(interior), 0)

    def test_identify_radial_boundary(self):
        R = 1.0
        r = disk(R=R, refinement=2)
        bV_radial = identify_radial_boundary(r.HC, R, center_axes=(0, 1), tol=1e-6)
        self.assertGreater(len(bV_radial), 0)
        for v in bV_radial:
            dist = np.linalg.norm(v.x_a[:2])
            self.assertAlmostEqual(dist, R, places=5)


# ── Integration with compute_vd ─────────────────────────────────────────

class TestComputeVdIntegration(unittest.TestCase):
    """Verify that domains work with the dual mesh computation pipeline."""

    def test_rectangle_compute_vd(self):
        from hyperct.ddg import compute_vd
        r = rectangle(L=1.0, h=1.0, refinement=2)
        r.tag_boundaries()
        compute_vd(r.HC, cdist=1e-10)
        # Verify dual vertices exist
        for v in r.HC.V:
            self.assertTrue(hasattr(v, 'vd'))

    def test_box_compute_vd(self):
        from hyperct.ddg import compute_vd
        r = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1)
        r.tag_boundaries()
        compute_vd(r.HC, cdist=1e-10)
        for v in r.HC.V:
            self.assertTrue(hasattr(v, 'vd'))

    def test_cylinder_compute_vd(self):
        """Projected 3D meshes need retopologization before compute_vd.

        After cube-to-cylinder projection, the original cube connectivity
        may contain degenerate tetrahedra.  The dynamic integrators call
        ``_retopologize()`` (Delaunay retriangulation) before every
        ``compute_vd()`` step, which fixes this.  We test that the
        retopologize → compute_vd pipeline works.
        """
        from hyperct.ddg import compute_vd
        from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
        r = cylinder_volume(R=0.5, L=1.0, refinement=1)
        r.tag_boundaries()
        _retopologize(r.HC, r.bV, dim=3)
        compute_vd(r.HC, cdist=1e-10)
        for v in r.HC.V:
            self.assertTrue(hasattr(v, 'vd'))


if __name__ == '__main__':
    unittest.main()
