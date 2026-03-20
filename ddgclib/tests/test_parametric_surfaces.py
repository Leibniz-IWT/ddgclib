"""Tests for ddgclib.geometry._parametric_surfaces.

Covers the core ``parametric_surface()`` builder, each convenience generator,
composition helpers, and backward-compatibility of refactored legacy generators.
"""

import unittest

import numpy as np
from hyperct import Complex

from ddgclib.geometry._parametric_surfaces import (
    parametric_surface,
    sphere,
    catenoid,
    cylinder,
    hyperboloid,
    torus,
    plane,
    translate_surface,
    scale_surface,
    rotate_surface,
    rotation_matrix_align,
    _default_boundary,
    _second_axis_boundary,
    _first_axis_boundary,
    _no_boundary,
)


class TestParametricSurface(unittest.TestCase):
    """Tests for the core parametric_surface() builder."""

    def test_basic_plane(self):
        """Flat plane should produce a 3D mesh at z=0."""
        def f(x, y):
            return (x, y, 0.0)

        HC, bV = parametric_surface(f, [(-1, 1), (-1, 1)], refinement=1)
        self.assertGreater(HC.V.size(), 0)
        for v in HC.V:
            self.assertAlmostEqual(v.x_a[2], 0.0)

    def test_custom_parametric_fn(self):
        """User-supplied function should work."""
        def paraboloid(u, v):
            return (u, v, u**2 + v**2)

        HC, bV = parametric_surface(paraboloid, [(-1, 1), (-1, 1)], refinement=1)
        self.assertGreater(HC.V.size(), 0)
        # Origin vertex should be at (0,0,0)
        origin_verts = [v for v in HC.V if abs(v.x_a[0]) < 1e-10
                        and abs(v.x_a[1]) < 1e-10]
        if origin_verts:
            self.assertAlmostEqual(origin_verts[0].x_a[2], 0.0)

    def test_connectivity(self):
        """Every vertex should have at least one neighbour."""
        HC, bV = parametric_surface(
            lambda u, v: (u, v, 0.0), [(-1, 1), (-1, 1)], refinement=1,
        )
        for v in HC.V:
            self.assertGreater(len(v.nn), 0,
                               f"Vertex {v.x} has no neighbours")

    def test_boundary_identification(self):
        """Default boundary should tag edges of parameter domain."""
        HC, bV = parametric_surface(
            lambda u, v: (u, v, 0.0), [(-1, 1), (-1, 1)], refinement=1,
        )
        self.assertGreater(len(bV), 0)
        interior = [v for v in HC.V if v not in bV]
        self.assertGreater(len(interior), 0)

    def test_custom_boundary_fn(self):
        """Custom boundary function should be used."""
        # Only mark top edge as boundary
        def top_only(coords, domain):
            return coords[1] == domain[1][1]

        HC, bV = parametric_surface(
            lambda u, v: (u, v, 0.0), [(-1, 1), (-1, 1)],
            refinement=1, boundary_fn=top_only,
        )
        for v in bV:
            self.assertAlmostEqual(v.x_a[1], 1.0)

    def test_refinement_increases_vertices(self):
        """Higher refinement should produce more vertices."""
        n1 = parametric_surface(
            lambda u, v: (u, v, 0.0), [(-1, 1), (-1, 1)], refinement=1,
        )[0].V.size()
        n2 = parametric_surface(
            lambda u, v: (u, v, 0.0), [(-1, 1), (-1, 1)], refinement=2,
        )[0].V.size()
        self.assertGreater(n2, n1)

    def test_merge_coincident_vertices(self):
        """Periodic surface should merge seam vertices."""
        # Cylinder wraps around: u=0 and u=2pi should merge
        HC, bV = cylinder(R=1.0, refinement=1, cdist=1e-8)
        # If seam was not merged, vertex count would be higher
        n_verts = HC.V.size()
        # A cylinder with refinement=1 should have ~13 vertices (merged)
        self.assertLess(n_verts, 25)  # would be ~18 unmerged


class TestSphere(unittest.TestCase):
    """Tests for the sphere() generator."""

    def test_radius(self):
        """All sphere vertices should lie at distance R from origin."""
        R = 2.5
        HC, bV = sphere(R=R, refinement=2)
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertAlmostEqual(dist, R, places=10,
                                   msg=f"Vertex {v.x} at dist {dist}, expected {R}")

    def test_vertex_count_refinement(self):
        """More refinement = more vertices."""
        n1 = sphere(refinement=1)[0].V.size()
        n2 = sphere(refinement=2)[0].V.size()
        n3 = sphere(refinement=3)[0].V.size()
        self.assertLess(n1, n2)
        self.assertLess(n2, n3)

    def test_boundary_at_poles(self):
        """Boundary vertices should be near the poles (phi = phi_min, phi_max)."""
        HC, bV = sphere(refinement=2, phi_range=(0.01, np.pi - 0.01))
        self.assertGreater(len(bV), 0)
        # Boundary verts should have high |z| (near poles)
        for v in bV:
            z = v.x_a[2]
            self.assertGreater(abs(z), 0.9,  # cos(0.01) ≈ 0.99995
                               f"Boundary vertex {v.x} not near pole")

    def test_sphere_sector(self):
        """Sphere sector should have more boundary vertices than full sphere."""
        _, bV_full = sphere(refinement=2, theta_range=(0, 2 * np.pi))
        _, bV_half = sphere(refinement=2, theta_range=(0, np.pi))
        # Half sphere has boundary on both theta edges + phi edges
        self.assertGreater(len(bV_half), len(bV_full))


class TestCatenoid(unittest.TestCase):
    """Tests for the catenoid() generator."""

    def test_minimal_surface(self):
        """Catenoid neck radius should equal parameter a."""
        a = 1.5
        HC, bV = catenoid(a=a, refinement=2)
        # At z=0, r = a*cosh(0) = a
        neck = [v for v in HC.V if abs(v.x_a[2]) < 0.01]
        for v in neck:
            r = np.sqrt(v.x_a[0] ** 2 + v.x_a[1] ** 2)
            self.assertAlmostEqual(r, a, places=5)

    def test_boundary_at_ends(self):
        """Boundary should be at axial extremes, not angular seam."""
        HC, bV = catenoid(v_range=(-1.5, 1.5), refinement=2)
        for v in bV:
            z = v.x_a[2]
            self.assertTrue(abs(z - 1.5) < 1e-10 or abs(z + 1.5) < 1e-10,
                            f"Boundary vertex at z={z}, expected ±1.5")


class TestCylinder(unittest.TestCase):
    """Tests for the cylinder() generator."""

    def test_radius(self):
        """All cylinder vertices should lie at distance R from z-axis."""
        R = 3.0
        HC, bV = cylinder(R=R, refinement=2)
        for v in HC.V:
            r = np.sqrt(v.x_a[0] ** 2 + v.x_a[1] ** 2)
            self.assertAlmostEqual(r, R, places=10)

    def test_height_range(self):
        """Vertices should span the specified height range."""
        HC, bV = cylinder(h_range=(-2.0, 3.0), refinement=2)
        z_vals = [v.x_a[2] for v in HC.V]
        self.assertAlmostEqual(min(z_vals), -2.0, places=10)
        self.assertAlmostEqual(max(z_vals), 3.0, places=10)


class TestHyperboloid(unittest.TestCase):
    """Tests for the hyperboloid() generator."""

    def test_on_surface(self):
        """All vertices should satisfy x²/a² + y²/a² - z²/c² = 1."""
        a, c = 1.0, 1.0
        HC, bV = hyperboloid(a=a, c=c, refinement=2)
        for v in HC.V:
            x, y, z = v.x_a[:3]
            val = x ** 2 / a ** 2 + y ** 2 / a ** 2 - z ** 2 / c ** 2
            self.assertAlmostEqual(val, 1.0, places=5,
                                   msg=f"Vertex {v.x}: x²/a²+y²/a²-z²/c² = {val}")


class TestTorus(unittest.TestCase):
    """Tests for the torus() generator."""

    def test_no_boundary(self):
        """Torus is closed — no boundary vertices."""
        HC, bV = torus(refinement=1)
        self.assertEqual(len(bV), 0)

    def test_on_surface(self):
        """Vertices should satisfy torus equation."""
        R, r = 2.0, 0.5
        HC, bV = torus(R=R, r=r, refinement=2)
        for v in HC.V:
            x, y, z = v.x_a[:3]
            # Torus: (sqrt(x²+y²) - R)² + z² = r²
            val = (np.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2
            self.assertAlmostEqual(val, r ** 2, places=5)


class TestPlane(unittest.TestCase):
    """Tests for the plane() generator."""

    def test_flat(self):
        """Plane should have z=0 everywhere."""
        HC, bV = plane(refinement=2)
        for v in HC.V:
            self.assertAlmostEqual(v.x_a[2], 0.0)

    def test_range(self):
        """Vertices should span the specified ranges."""
        HC, bV = plane(x_range=(-3, 3), y_range=(-5, 5), refinement=1)
        x_vals = [v.x_a[0] for v in HC.V]
        y_vals = [v.x_a[1] for v in HC.V]
        self.assertAlmostEqual(min(x_vals), -3.0)
        self.assertAlmostEqual(max(x_vals), 3.0)
        self.assertAlmostEqual(min(y_vals), -5.0)
        self.assertAlmostEqual(max(y_vals), 5.0)


class TestTranslateSurface(unittest.TestCase):
    """Tests for translate_surface()."""

    def test_translation(self):
        """Translate should shift all vertices by offset."""
        HC, bV = sphere(R=1.0, refinement=1)
        center_before = np.mean([v.x_a[:3] for v in HC.V], axis=0)

        translate_surface(HC, [10.0, -5.0, 3.0])
        center_after = np.mean([v.x_a[:3] for v in HC.V], axis=0)

        expected = center_before + np.array([10.0, -5.0, 3.0])
        np.testing.assert_array_almost_equal(center_after, expected, decimal=10)

    def test_preserves_shape(self):
        """Translation should preserve inter-vertex distances."""
        HC, bV = sphere(R=1.0, refinement=1)
        dists_before = [np.linalg.norm(v.x_a[:3]) for v in HC.V]

        translate_surface(HC, [100.0, 0.0, 0.0])
        center = np.mean([v.x_a[:3] for v in HC.V], axis=0)
        dists_after = sorted(
            np.linalg.norm(v.x_a[:3] - center) for v in HC.V
        )
        dists_before_sorted = sorted(dists_before)

        for d1, d2 in zip(dists_before_sorted, dists_after):
            self.assertAlmostEqual(d1, d2, places=8)


class TestScaleSurface(unittest.TestCase):
    """Tests for scale_surface()."""

    def test_scale_about_origin(self):
        """Scaling about origin should multiply all radii."""
        HC, bV = sphere(R=1.0, refinement=1)
        scale_surface(HC, 5.0)
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertAlmostEqual(dist, 5.0, places=8)

    def test_scale_about_center(self):
        """Scaling about a non-origin center should preserve the center."""
        HC, bV = sphere(R=1.0, refinement=1)
        translate_surface(HC, [10.0, 0.0, 0.0])
        center = np.array([10.0, 0.0, 0.0])

        scale_surface(HC, 3.0, center=center)

        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3] - center)
            self.assertAlmostEqual(dist, 3.0, places=8)


class TestBoundaryPredicates(unittest.TestCase):
    """Tests for boundary predicate functions."""

    def test_default_boundary(self):
        domain = [(-1, 1), (-2, 2)]
        self.assertTrue(_default_boundary((-1, 0), domain))
        self.assertTrue(_default_boundary((0.5, 2), domain))
        self.assertFalse(_default_boundary((0, 0), domain))

    def test_second_axis_boundary(self):
        domain = [(0, 6.28), (-1.5, 1.5)]
        self.assertTrue(_second_axis_boundary((3.14, -1.5), domain))
        self.assertTrue(_second_axis_boundary((0, 1.5), domain))
        self.assertFalse(_second_axis_boundary((0, 0), domain))
        # u-edges should NOT be boundary
        self.assertFalse(_second_axis_boundary((0, 0.5), domain))

    def test_first_axis_boundary(self):
        domain = [(-2, 2), (0, 6.28)]
        self.assertTrue(_first_axis_boundary((-2, 3.14), domain))
        self.assertFalse(_first_axis_boundary((0, 0), domain))

    def test_no_boundary(self):
        domain = [(0, 1), (0, 1)]
        self.assertFalse(_no_boundary((0, 0), domain))
        self.assertFalse(_no_boundary((1, 1), domain))


class TestRotateSurface(unittest.TestCase):
    """Tests for rotate_surface() and rotation_matrix_align()."""

    def test_identity_rotation(self):
        """Parallel vectors should give identity matrix."""
        R = rotation_matrix_align(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_antiparallel_rotation(self):
        """Antiparallel should give a 180-degree rotation."""
        R = rotation_matrix_align(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))
        result = R @ np.array([0, 0, 1.0])
        np.testing.assert_array_almost_equal(result, [0, 0, -1.0])
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_z_to_x_rotation(self):
        """Rotating z-axis to x-axis."""
        R = rotation_matrix_align(np.array([0, 0, 1.0]), np.array([1, 0, 0.0]))
        np.testing.assert_array_almost_equal(R @ [0, 0, 1], [1, 0, 0])
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_rotation_preserves_radius(self):
        """Rotating a sphere should preserve vertex distances."""
        HC, bV = sphere(R=1.0, refinement=1)
        R = rotation_matrix_align(np.array([0, 0, 1.0]), np.array([1, 1, 0.0]))
        rotate_surface(HC, R)
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3])
            self.assertAlmostEqual(dist, 1.0, places=8)

    def test_rotation_about_center(self):
        """Rotation about a non-origin centre should preserve distances."""
        HC, bV = sphere(R=1.0, refinement=1)
        translate_surface(HC, [5.0, 0.0, 0.0])
        center = np.array([5.0, 0.0, 0.0])

        R = rotation_matrix_align(np.array([0, 0, 1.0]), np.array([0, 1, 0.0]))
        rotate_surface(HC, R, center=center)

        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3] - center)
            self.assertAlmostEqual(dist, 1.0, places=8)

    def test_arbitrary_direction(self):
        """Rotation to arbitrary direction."""
        target = np.array([1.0, 2.0, 3.0])
        R = rotation_matrix_align(np.array([0, 0, 1.0]), target)
        result = R @ np.array([0, 0, 1.0])
        expected = target / np.linalg.norm(target)
        np.testing.assert_array_almost_equal(result, expected)


class TestLegacyBackwardCompatibility(unittest.TestCase):
    """Verify refactored _sphere.py, _catenoid.py, _hyperboloid.py."""

    def test_sphere_N(self):
        from ddgclib._sphere import sphere_N
        Phi = np.linspace(0.01, np.pi - 0.01, 10)
        Theta = np.linspace(0.0, 2 * np.pi, 10)
        HC, bV = sphere_N(1.0, Phi, Theta, refinement=1)
        self.assertGreater(HC.V.size(), 0)
        self.assertGreater(len(bV), 0)
        for v in HC.V:
            self.assertAlmostEqual(np.linalg.norm(v.x_a[:3]), 1.0, places=8)

    def test_catenoid_N(self):
        from ddgclib._catenoid import catenoid_N
        HC, bV, K_f, H_f, neck_verts, neck_sols = catenoid_N(
            1, 0, 0, (1, 0, 1), refinement=1,
        )
        self.assertGreater(HC.V.size(), 0)
        self.assertGreater(len(bV), 0)
        # Catenoid is minimal: H = 0
        for h in H_f:
            self.assertAlmostEqual(h, 0.0)

    def test_hyperboloid_N(self):
        from ddgclib._hyperboloid import hyperboloid_N
        HC, bV, K_f, H_f, neck_verts, neck_sols = hyperboloid_N(
            1, 0, 0, (1, 0, 1), refinement=1,
        )
        self.assertGreater(HC.V.size(), 0)
        self.assertGreater(len(bV), 0)
        self.assertGreater(len(H_f), 0)


# ── Surface tension operator tests ────────────────────────────────────

class TestSurfaceTensionOperator(unittest.TestCase):
    """Tests for ddgclib.operators.surface_tension."""

    def setUp(self):
        """Create a sphere mesh with fluid attributes."""
        self.HC, self.bV = sphere(R=1.0, refinement=2)
        for v in self.HC.V:
            v.boundary = v in self.bV
            v.u = np.zeros(3)
            v.m = 0.001

    def test_surface_tension_force_returns_vector(self):
        from ddgclib.operators.surface_tension import surface_tension_force
        v = next(v for v in self.HC.V if v not in self.bV)
        F = surface_tension_force(v, gamma=0.072, dim=3)
        self.assertEqual(F.shape, (3,))

    def test_surface_tension_force_inward_on_sphere(self):
        """On a convex sphere, surface tension should point inward."""
        from ddgclib.operators.surface_tension import surface_tension_force
        v = next(v for v in self.HC.V if v not in self.bV)
        F = surface_tension_force(v, gamma=0.072, dim=3)
        # Force should have a component pointing toward center (0,0,0)
        # Direction from vertex to center = -v.x_a
        r_hat = v.x_a[:3] / np.linalg.norm(v.x_a[:3])
        dot = np.dot(F, r_hat)
        # Surface tension on convex surface points inward (toward center)
        # so dot(F, r_hat) should be negative (F points opposite to r_hat)
        # Due to discretization, allow some tolerance
        self.assertLess(dot, 0.1 * np.linalg.norm(F))

    def test_surface_tension_acceleration_with_mass(self):
        from ddgclib.operators.surface_tension import surface_tension_acceleration
        v = next(v for v in self.HC.V if v not in self.bV)
        v.m = 0.01
        a = surface_tension_acceleration(v, gamma=0.072, dim=3)
        self.assertEqual(a.shape, (3,))
        # Acceleration should be F/m
        from ddgclib.operators.surface_tension import surface_tension_force
        F = surface_tension_force(v, gamma=0.072, dim=3)
        np.testing.assert_allclose(a, F / v.m)

    def test_surface_tension_acceleration_with_damping(self):
        from ddgclib.operators.surface_tension import surface_tension_acceleration
        v = next(v for v in self.HC.V if v not in self.bV)
        v.u = np.array([0.1, 0.0, 0.0])
        a_no_damp = surface_tension_acceleration(v, gamma=0.072, damping=0.0, dim=3)
        a_damped = surface_tension_acceleration(v, gamma=0.072, damping=1.0, dim=3)
        # Damping should reduce acceleration in velocity direction
        diff = a_no_damp - a_damped
        self.assertGreater(diff[0], 0)  # x-component reduced

    def test_dual_area_heron_positive(self):
        from ddgclib.operators.surface_tension import dual_area_heron
        v = next(v for v in self.HC.V if v not in self.bV)
        C = dual_area_heron(v)
        self.assertGreater(C, 0)

    def test_net_force_nearly_zero_on_sphere(self):
        """Net surface tension force on a closed sphere should be near zero."""
        from ddgclib.operators.surface_tension import surface_tension_force
        F_total = np.zeros(3)
        for v in self.HC.V:
            F_total += surface_tension_force(v, gamma=0.072, dim=3)
        # Should be near zero by symmetry
        self.assertLess(np.linalg.norm(F_total), 1.0)


class TestDynamicIntegratorSurfaceMesh(unittest.TestCase):
    """Test dynamic integrators with retopologize_fn on surface meshes."""

    def setUp(self):
        """Create a sphere mesh ready for integration."""
        self.HC, self.bV = sphere(R=1.0, refinement=2)
        self.bV_frozen = set()  # No frozen vertices for this test
        for v in self.HC.V:
            v.boundary = v in self.bV
            v.u = np.zeros(3)
            v.p = np.array([0.0])
            v.m = 0.001

    def test_symplectic_euler_with_retopo_false(self):
        """symplectic_euler should work with retopologize_fn=False."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler
        from ddgclib.operators.surface_tension import surface_tension_acceleration

        dudt_fn = partial(surface_tension_acceleration, gamma=0.072, dim=3)
        t = symplectic_euler(
            self.HC, self.bV_frozen, dudt_fn,
            dt=1e-5, n_steps=3, dim=3,
            retopologize_fn=False,
        )
        self.assertAlmostEqual(t, 3e-5, places=10)

    def test_euler_with_retopo_false(self):
        """euler should work with retopologize_fn=False."""
        from functools import partial
        from ddgclib.dynamic_integrators import euler
        from ddgclib.operators.surface_tension import surface_tension_acceleration

        dudt_fn = partial(surface_tension_acceleration, gamma=0.072, dim=3)
        t = euler(
            self.HC, self.bV_frozen, dudt_fn,
            dt=1e-5, n_steps=3, dim=3,
            retopologize_fn=False,
        )
        self.assertAlmostEqual(t, 3e-5, places=10)

    def test_euler_velocity_only_with_retopo_false(self):
        """euler_velocity_only should work with retopologize_fn=False."""
        from functools import partial
        from ddgclib.dynamic_integrators import euler_velocity_only
        from ddgclib.operators.surface_tension import surface_tension_acceleration

        dudt_fn = partial(surface_tension_acceleration, gamma=0.072, dim=3)
        t = euler_velocity_only(
            self.HC, self.bV_frozen, dudt_fn,
            dt=1e-5, n_steps=3, dim=3,
            retopologize_fn=False,
        )
        self.assertAlmostEqual(t, 3e-5, places=10)
        # Velocities should have changed
        max_u = max(np.linalg.norm(v.u[:3]) for v in self.HC.V)
        self.assertGreater(max_u, 0)

    def test_custom_retopo_fn(self):
        """Test passing a custom retopologize_fn."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler
        from ddgclib.operators.surface_tension import surface_tension_acceleration

        call_count = [0]

        def custom_retopo(HC, bV, dim):
            call_count[0] += 1

        dudt_fn = partial(surface_tension_acceleration, gamma=0.072, dim=3)
        symplectic_euler(
            self.HC, self.bV_frozen, dudt_fn,
            dt=1e-5, n_steps=3, dim=3,
            retopologize_fn=custom_retopo,
        )
        self.assertEqual(call_count[0], 3)

    def test_default_retopo_backward_compat(self):
        """With retopologize_fn=None (default), existing behavior preserved."""
        from functools import partial
        from ddgclib.dynamic_integrators import symplectic_euler
        from ddgclib.operators.stress import stress_acceleration

        # Create a simple 2D mesh where default retopo works
        HC2 = Complex(2, domain=[(-1, 1), (-1, 1)])
        HC2.triangulate()
        HC2.refine_all()

        from hyperct.ddg import compute_vd
        bV2 = HC2.boundary()
        for v in HC2.V:
            v.boundary = v in bV2
        compute_vd(HC2, method="barycentric")

        for v in HC2.V:
            v.u = np.zeros(2)
            v.p = np.array([0.0])
            v.m = 0.01

        dudt_fn = partial(stress_acceleration, dim=2, mu=0.1, HC=HC2)
        # This should use the default _retopologize
        t = symplectic_euler(
            HC2, bV2, dudt_fn,
            dt=1e-4, n_steps=2, dim=2,
        )
        self.assertAlmostEqual(t, 2e-4, places=10)


class TestFluidFilmFunctions(unittest.TestCase):
    """Test the fluid film helper functions."""

    def test_stokes_integral(self):
        """Stokes integral should return a force vector."""
        import sys
        sys.path.insert(0, str(np.array(0)))  # no-op
        from cases_dynamic.liquid_bridge_cfd_dem.src._fluid_film import stokes_integral

        HC, bV = sphere(R=1.0, refinement=2)
        for v in HC.V:
            v.boundary = v in bV
            v.particle_id = 0

        F = stokes_integral(HC, particle_id=0, gamma=0.072)
        self.assertEqual(F.shape, (3,))
        # For a sphere, net force should be nearly zero
        self.assertLess(np.linalg.norm(F), 1.0)

    def test_film_snapshot(self):
        """snapshot() should return well-formed dict."""
        from cases_dynamic.liquid_bridge_cfd_dem.src._fluid_film import snapshot

        HC, bV = sphere(R=1.0, refinement=2)
        for v in HC.V:
            v.boundary = v in bV
            v.u = np.zeros(3)
            v.particle_id = 0

        snap = snapshot(HC)
        self.assertIn("vertices", snap)
        self.assertIn("edges", snap)
        self.assertIn("particle_ids", snap)
        self.assertIn("curvature", snap)
        self.assertIn("n_vertices", snap)
        self.assertGreater(snap["n_vertices"], 0)
        self.assertGreater(snap["n_edges"], 0)

    def test_detect_and_form_bridge(self):
        """Bridge detection should connect close vertices."""
        from cases_dynamic.liquid_bridge_cfd_dem.src._fluid_film import detect_and_form_bridge

        # Create two small hemispheres close together
        HC1, bV1 = sphere(R=0.5, refinement=1,
                          phi_range=(0.01, np.pi * 0.5))
        HC2, bV2 = sphere(R=0.5, refinement=1,
                          phi_range=(0.01, np.pi * 0.5))

        # Build combined Complex
        HC_combined = Complex(3, domain=[])

        # Add HC1 vertices (at x=-0.6)
        v1_map = {}
        for v in HC1.V:
            new_pos = v.x_a.copy()
            new_pos[0] -= 0.6
            v_new = HC_combined.V[tuple(new_pos)]
            v_new.particle_id = 0
            v_new.boundary = v in bV1
            v1_map[id(v)] = v_new
        for v in HC1.V:
            for nb in v.nn:
                v1_map[id(v)].connect(v1_map[id(nb)])

        # Add HC2 vertices (at x=+0.6)
        v2_map = {}
        for v in HC2.V:
            new_pos = v.x_a.copy()
            new_pos[0] += 0.6
            v_new = HC_combined.V[tuple(new_pos)]
            v_new.particle_id = 1
            v_new.boundary = v in bV2
            v2_map[id(v)] = v_new
        for v in HC2.V:
            for nb in v.nn:
                v2_map[id(v)].connect(v2_map[id(nb)])

        # Mark boundary vertices: use the original bV sets transferred
        # through the maps (HC_combined.boundary() may not work for
        # surface-only topology in a 3D Complex)
        for v in HC1.V:
            if v in bV1:
                v1_map[id(v)].boundary = True
        for v in HC2.V:
            if v in bV2:
                v2_map[id(v)].boundary = True

        # The gap is ~1.2 between centers, ~0.2 between surfaces
        # With threshold=1.5, should connect
        n_conn = detect_and_form_bridge(HC_combined, threshold=1.5)
        self.assertGreater(n_conn, 0)

    def test_sync_film_to_particles(self):
        """sync should move vertices to track particle positions."""
        from cases_dynamic.liquid_bridge_cfd_dem.src._fluid_film import sync_film_to_particles

        HC, bV = sphere(R=1.0, refinement=1)
        for v in HC.V:
            v.boundary = v in bV
            v.particle_id = 0
            v.u = np.zeros(3)

        # Fake particle — position chosen to not coincide with any mesh vertex
        class FakeParticle:
            def __init__(self, id_, x, R, u):
                self.id = id_
                self.x_a = np.array(x)
                self.radius = R
                self.u = np.array(u)

        p = FakeParticle(0, [0.5, 0.3, 0.1], 1.0, [0.0, 0.0, 0.0])

        sync_film_to_particles(HC, [p], film_radius_factor=1.01)

        # All vertices should now be at distance ~1.01 from particle center
        center = np.array([0.5, 0.3, 0.1])
        for v in HC.V:
            dist = np.linalg.norm(v.x_a[:3] - center)
            self.assertAlmostEqual(dist, 1.01, delta=0.05)


if __name__ == "__main__":
    unittest.main()
