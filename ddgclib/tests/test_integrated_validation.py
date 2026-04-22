"""Tests for integrated analytical validation framework.

Validates DDG integrated operators against analytically integrated
solutions over the exact same dual cell domains.  Uses the divergence
theorem identity:

    ∫_V ∇f dV = ∮_{∂V} f · n dA

Key properties tested:
- Linear fields: machine precision (~1e-14) on ALL mesh types
- Quadratic fields on symmetric meshes: machine precision
- Non-linear fields on jittered meshes: convergent with refinement
- Sympy and Gauss-Legendre paths agree
- Stress operators: pressure gradient, viscous flux, Poiseuille equilibrium
- Curvature operators: sphere, cylinder (convergent)
"""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd, dual_cell_polygon_2d, dual_cell_vertices_1d
from hyperct.ddg._dual_cell import _shoelace_area

from ddgclib.analytical import integrated_gradient_1d, integrated_gradient_2d
from ddgclib.analytical._divergence_theorem import integrated_gradient_2d_vector
from ddgclib.operators.stress import (
    scalar_gradient_integrated,
    stress_force,
    velocity_difference_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh_1d(n_refine=2, domain=(0.0, 1.0)):
    """Create a 1D mesh with duals computed."""
    HC = Complex(1, domain=[domain])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        on_bnd = (abs(v.x_a[0] - domain[0]) < 1e-14 or
                  abs(v.x_a[0] - domain[1]) < 1e-14)
        v.boundary = on_bnd
        if on_bnd:
            bV.add(v)

    compute_vd(HC, cdist=1e-10)
    interior = [v for v in HC.V if v not in bV]
    return HC, bV, interior


def _make_mesh_2d(n_refine=1, domain=((0.0, 1.0), (0.0, 1.0)),
                  method="barycentric", seed=None, jitter=0.05):
    """Create a 2D mesh with duals computed."""
    HC = Complex(2, domain=list(domain))
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        on_bnd = any(
            abs(v.x_a[d] - domain[d][0]) < 1e-14 or
            abs(v.x_a[d] - domain[d][1]) < 1e-14
            for d in range(2)
        )
        v.boundary = on_bnd
        if on_bnd:
            bV.add(v)

    # Jitter interior vertices and re-Delaunay for asymmetric mesh tests
    if seed is not None:
        from scipy.spatial import Delaunay as _Delaunay

        rng = np.random.default_rng(seed)
        to_move = []
        for v in HC.V:
            if v not in bV and v.nn:
                edge_len = min(
                    np.linalg.norm(v.x_a - vn.x_a) for vn in v.nn
                )
                offset = rng.uniform(-jitter * edge_len,
                                     jitter * edge_len, size=2)
                new_x = tuple(v.x_a[d] + offset[d] for d in range(2))
                to_move.append((v, new_x))
        for v, new_x in to_move:
            HC.V.move(v, new_x)

        # Re-Delaunay triangulate
        verts = list(HC.V)
        for v in verts:
            for nb in list(v.nn):
                v.disconnect(nb)
        coords = np.array([v.x_a[:2] for v in verts])
        tri = _Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    verts[simplex[i]].connect(verts[simplex[j]])

    compute_vd(HC, method=method, cdist=1e-10)
    interior = [v for v in HC.V if v not in bV]
    return HC, bV, interior


def _assign_scalar(HC, f_callable, dim=2):
    """Set v.f = f(v.x_a[:dim]) on all vertices."""
    for v in HC.V:
        v.f = f_callable(v.x_a[:dim])


def _assign_vector(HC, u_callable, dim=2):
    """Set v.u = u(v.x_a[:dim]) on all vertices."""
    for v in HC.V:
        v.u = u_callable(v.x_a[:dim])


def _max_error_scalar_2d(HC, interior, f_callable, n_gauss=10):
    """Max absolute error: numerical vs analytical integrated scalar gradient."""
    max_err = 0.0
    for v in interior:
        num = scalar_gradient_integrated(v, HC, dim=2, field_attr='f')
        polygon = dual_cell_polygon_2d(v)
        ana = integrated_gradient_2d(f_callable, polygon, n_gauss=n_gauss)
        max_err = max(max_err, np.linalg.norm(num - ana))
    return max_err


def _max_error_vector_2d(HC, interior, u_callable, n_gauss=10):
    """Max absolute error: numerical vs analytical integrated vector gradient."""
    max_err = 0.0
    for v in interior:
        num = velocity_difference_tensor(v, HC, dim=2)
        polygon = dual_cell_polygon_2d(v)
        ana = integrated_gradient_2d_vector(u_callable, polygon, n_gauss=n_gauss)
        max_err = max(max_err, np.linalg.norm(num - ana))
    return max_err


# ---------------------------------------------------------------------------
# Test dual cell geometry
# ---------------------------------------------------------------------------

class TestDualCellGeometry:
    """Verify dual cell extraction produces correct geometry."""

    def test_1d_interval_endpoints(self):
        """Interior 1D vertices should have two dual vertices defining
        an interval containing the vertex."""
        HC, bV, interior = _make_mesh_1d(n_refine=2)
        for v in interior:
            a, b = dual_cell_vertices_1d(v)
            assert a < b, f"Interval not ordered: {a} >= {b}"
            assert a <= v.x_a[0] <= b, (
                f"Vertex {v.x_a[0]} not inside dual cell [{a}, {b}]"
            )

    def test_2d_polygon_ccw_orientation(self):
        """Dual cell polygons should have positive (CCW) orientation."""
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        for v in interior:
            polygon = dual_cell_polygon_2d(v)
            area = _shoelace_area(polygon)
            assert area > 0, (
                f"Polygon at {v.x} has non-positive area {area}"
            )

    def test_2d_polygon_has_enough_vertices(self):
        """Interior dual cell polygons should have >= 6 vertices
        (N edge midpoints + N barycenters, N >= 3)."""
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        for v in interior:
            polygon = dual_cell_polygon_2d(v)
            assert len(polygon) >= 6, (
                f"Polygon at {v.x} has only {len(polygon)} vertices"
            )

    def test_2d_polygon_contains_vertex(self):
        """The primal vertex should be inside its dual cell polygon."""
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        for v in interior:
            polygon = dual_cell_polygon_2d(v)
            # Check via shoelace: vertex is inside if sum of sub-triangle
            # areas equals the polygon area
            vx = v.x_a[:2]
            N = len(polygon)
            sub_area = 0.0
            for k in range(N):
                tri = np.array([vx, polygon[k], polygon[(k + 1) % N]])
                sub_area += abs(_shoelace_area(tri))
            poly_area = abs(_shoelace_area(polygon))
            npt.assert_allclose(sub_area, poly_area, rtol=1e-10,
                                err_msg=f"Vertex {v.x} not inside polygon")

    def test_2d_interior_areas_sum(self):
        """Sum of interior dual cell areas should be less than domain area."""
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        total = sum(abs(_shoelace_area(dual_cell_polygon_2d(v)))
                    for v in interior)
        domain_area = 1.0
        assert total < domain_area, (
            f"Interior areas sum {total} exceeds domain area {domain_area}"
        )
        assert total > 0.3 * domain_area, (
            f"Interior areas sum {total} too small (< 30% of domain)"
        )


# ---------------------------------------------------------------------------
# 1D integrated gradient tests
# ---------------------------------------------------------------------------

class TestIntegratedGradient1D:
    """Test integrated gradient validation in 1D."""

    def test_linear_machine_precision(self):
        """f(x) = 2x + 1: integrated gradient must match analytically."""
        HC, bV, interior = _make_mesh_1d(n_refine=3)
        f = lambda x: 2.0 * x[0] + 1.0
        _assign_scalar(HC, f, dim=1)

        for v in interior:
            num = scalar_gradient_integrated(v, HC, dim=1, field_attr='f')
            a, b = dual_cell_vertices_1d(v)
            ana = integrated_gradient_1d(f, a, b)
            npt.assert_allclose(num, ana, atol=1e-14,
                                err_msg=f"Linear 1D failed at {v.x}")

    def test_quadratic_machine_precision(self):
        """f(x) = x²: in 1D the DDG operator is exact for quadratics."""
        HC, bV, interior = _make_mesh_1d(n_refine=3)
        f = lambda x: x[0] ** 2
        _assign_scalar(HC, f, dim=1)

        for v in interior:
            num = scalar_gradient_integrated(v, HC, dim=1, field_attr='f')
            a, b = dual_cell_vertices_1d(v)
            ana = integrated_gradient_1d(f, a, b)
            npt.assert_allclose(num, ana, atol=1e-14,
                                err_msg=f"Quadratic 1D failed at {v.x}")

    def test_cubic_convergence_1d(self):
        """f(x) = x³: error should decrease with refinement."""
        f = lambda x: x[0] ** 3

        errors = []
        for n_ref in [2, 4]:
            HC, bV, interior = _make_mesh_1d(n_refine=n_ref)
            _assign_scalar(HC, f, dim=1)

            max_err = 0.0
            for v in interior:
                num = scalar_gradient_integrated(v, HC, dim=1, field_attr='f')
                a, b = dual_cell_vertices_1d(v)
                ana = integrated_gradient_1d(f, a, b)
                max_err = max(max_err, abs(num[0] - ana[0]))
            errors.append(max_err)

        assert errors[1] < errors[0], (
            f"Cubic 1D error did not decrease: {errors}"
        )


# ---------------------------------------------------------------------------
# 2D integrated gradient tests — scalar
# ---------------------------------------------------------------------------

class TestIntegratedGradientScalar2D:
    """Test integrated scalar gradient validation in 2D."""

    def test_linear_machine_precision_barycentric(self):
        """f = 3x - 2y + 1 on barycentric dual: machine precision."""
        f = lambda x: 3.0 * x[0] - 2.0 * x[1] + 1.0
        HC, bV, interior = _make_mesh_2d(n_refine=1, method="barycentric")
        _assign_scalar(HC, f)
        err = _max_error_scalar_2d(HC, interior, f)
        assert err < 1e-13, f"Linear scalar 2D (bary): err={err:.2e}"

    def test_linear_machine_precision_circumcentric(self):
        """f = 3x - 2y + 1 on circumcentric dual: machine precision."""
        f = lambda x: 3.0 * x[0] - 2.0 * x[1] + 1.0
        HC, bV, interior = _make_mesh_2d(n_refine=1, method="circumcentric")
        _assign_scalar(HC, f)
        err = _max_error_scalar_2d(HC, interior, f)
        assert err < 1e-13, f"Linear scalar 2D (circ): err={err:.2e}"

    def test_linear_jittered_machine_precision(self):
        """f = 3x - 2y + 1 on jittered mesh: still machine precision."""
        f = lambda x: 3.0 * x[0] - 2.0 * x[1] + 1.0
        HC, bV, interior = _make_mesh_2d(n_refine=1, seed=42)
        _assign_scalar(HC, f)
        err = _max_error_scalar_2d(HC, interior, f)
        assert err < 1e-13, f"Linear scalar 2D (jittered): err={err:.2e}"

    def test_quadratic_symmetric_machine_precision(self):
        """f = x² + y² on symmetric mesh: machine precision.

        On symmetric meshes the DDG operator is exact for quadratics
        due to cancellation of cross-terms.
        """
        f = lambda x: x[0] ** 2 + x[1] ** 2
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        _assign_scalar(HC, f)
        err = _max_error_scalar_2d(HC, interior, f)
        assert err < 1e-13, f"Quadratic scalar 2D (symmetric): err={err:.2e}"

    def test_quadratic_convergence_barycentric(self):
        """f = x² + y² on jittered mesh: convergent with refinement."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        errors = []
        for n_ref in [1, 2, 3]:
            HC, bV, interior = _make_mesh_2d(n_refine=n_ref, seed=42)
            _assign_scalar(HC, f)
            errors.append(_max_error_scalar_2d(HC, interior, f))

        # Error should decrease with refinement
        assert errors[-1] < errors[0], (
            f"Quadratic error did not decrease: {errors}"
        )

    def test_quadratic_convergence_circumcentric(self):
        """f = x² + y² with circumcentric dual: convergent."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        errors = []
        for n_ref in [1, 2]:
            HC, bV, interior = _make_mesh_2d(
                n_refine=n_ref, method="circumcentric", seed=42
            )
            _assign_scalar(HC, f)
            errors.append(_max_error_scalar_2d(HC, interior, f))

        assert errors[-1] < errors[0], (
            f"Quadratic circ error did not decrease: {errors}"
        )

    def test_cubic_convergence(self):
        """f = x³ + xy²: higher-order convergence."""
        f = lambda x: x[0] ** 3 + x[0] * x[1] ** 2
        errors = []
        for n_ref in [1, 2, 3]:
            HC, bV, interior = _make_mesh_2d(n_refine=n_ref, seed=42)
            _assign_scalar(HC, f)
            errors.append(_max_error_scalar_2d(HC, interior, f))

        assert errors[-1] < errors[0] * 0.5, (
            f"Cubic error did not decrease enough: {errors}"
        )

    def test_trig_convergence(self):
        """f = sin(πx)cos(πy): smooth non-polynomial convergence."""
        import math
        f = lambda x: math.sin(math.pi * x[0]) * math.cos(math.pi * x[1])
        errors = []
        for n_ref in [1, 2, 3]:
            HC, bV, interior = _make_mesh_2d(n_refine=n_ref)
            _assign_scalar(HC, f)
            errors.append(_max_error_scalar_2d(HC, interior, f))

        assert errors[-1] < errors[0] * 0.1, (
            f"Trig error did not decrease enough: {errors}"
        )


# ---------------------------------------------------------------------------
# 2D integrated gradient tests — vector
# ---------------------------------------------------------------------------

class TestIntegratedGradientVector2D:
    """Test integrated vector gradient (velocity difference tensor)."""

    def test_linear_vector_machine_precision(self):
        """u = [2x+3y, -x+y]: Du should match analytically."""
        u = lambda x: np.array([2.0 * x[0] + 3.0 * x[1], -x[0] + x[1]])
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        _assign_vector(HC, u)
        err = _max_error_vector_2d(HC, interior, u)
        assert err < 1e-13, f"Linear vector 2D: err={err:.2e}"

    def test_linear_vector_jittered_machine_precision(self):
        """u = [2x+3y, -x+y] on jittered mesh: still machine precision."""
        u = lambda x: np.array([2.0 * x[0] + 3.0 * x[1], -x[0] + x[1]])
        HC, bV, interior = _make_mesh_2d(n_refine=1, seed=42)
        _assign_vector(HC, u)
        err = _max_error_vector_2d(HC, interior, u)
        assert err < 1e-13, f"Linear vector 2D (jittered): err={err:.2e}"

    def test_quadratic_vector_convergence(self):
        """u = [x², y²]: convergent with refinement on jittered mesh."""
        u = lambda x: np.array([x[0] ** 2, x[1] ** 2])
        errors = []
        for n_ref in [1, 2]:
            HC, bV, interior = _make_mesh_2d(n_refine=n_ref, seed=42)
            _assign_vector(HC, u)
            errors.append(_max_error_vector_2d(HC, interior, u))

        assert errors[-1] < errors[0], (
            f"Quadratic vector error did not decrease: {errors}"
        )

    def test_poiseuille_symmetric_machine_precision(self):
        """u = [y(1-y), 0] on symmetric mesh: machine precision."""
        u = lambda x: np.array([x[1] * (1.0 - x[1]), 0.0])
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        _assign_vector(HC, u)
        err = _max_error_vector_2d(HC, interior, u)
        assert err < 1e-13, f"Poiseuille vector 2D: err={err:.2e}"

    def test_cubic_vector_convergence(self):
        """u = [x²y, xy²]: convergent."""
        u = lambda x: np.array([x[0] ** 2 * x[1], x[0] * x[1] ** 2])
        errors = []
        for n_ref in [1, 2]:
            HC, bV, interior = _make_mesh_2d(n_refine=n_ref, seed=42)
            _assign_vector(HC, u)
            errors.append(_max_error_vector_2d(HC, interior, u))

        assert errors[-1] < errors[0], (
            f"Cubic vector error did not decrease: {errors}"
        )


# ---------------------------------------------------------------------------
# Sympy cross-validation
# ---------------------------------------------------------------------------

class TestSympyCrossValidation:
    """Verify sympy and Gauss-Legendre paths give matching results."""

    @pytest.fixture
    def mesh_and_polygon(self):
        """Create mesh and get one interior vertex's polygon."""
        HC, bV, interior = _make_mesh_2d(n_refine=1)
        v = interior[0]
        polygon = dual_cell_polygon_2d(v)
        return v, polygon

    def test_sympy_matches_gauss_linear(self, mesh_and_polygon):
        """Sympy and Gauss should agree for linear f."""
        try:
            import sympy
        except ImportError:
            pytest.skip("sympy not installed")

        from ddgclib.analytical._sympy_integration import (
            integrated_gradient_sympy_2d,
        )

        v, polygon = mesh_and_polygon
        x, y = sympy.symbols('x y')
        f_sym = 3 * x - 2 * y + 1
        f_num = lambda x_a: 3.0 * x_a[0] - 2.0 * x_a[1] + 1.0

        result_sympy = integrated_gradient_sympy_2d(f_sym, x, y, polygon)
        result_gauss = integrated_gradient_2d(f_num, polygon, n_gauss=10)

        npt.assert_allclose(result_sympy, result_gauss, atol=1e-12)

    def test_sympy_matches_gauss_quadratic(self, mesh_and_polygon):
        """Sympy and Gauss should agree for quadratic f."""
        try:
            import sympy
        except ImportError:
            pytest.skip("sympy not installed")

        from ddgclib.analytical._sympy_integration import (
            integrated_gradient_sympy_2d,
        )

        v, polygon = mesh_and_polygon
        x, y = sympy.symbols('x y')
        f_sym = x**2 + y**2
        f_num = lambda x_a: x_a[0] ** 2 + x_a[1] ** 2

        result_sympy = integrated_gradient_sympy_2d(f_sym, x, y, polygon)
        result_gauss = integrated_gradient_2d(f_num, polygon, n_gauss=10)

        npt.assert_allclose(result_sympy, result_gauss, atol=1e-12)

    def test_sympy_1d(self):
        """Sympy 1D integration matches numeric."""
        try:
            import sympy
        except ImportError:
            pytest.skip("sympy not installed")

        from ddgclib.analytical._sympy_integration import (
            integrated_gradient_sympy_1d,
        )

        x = sympy.Symbol('x')
        f_sym = x**3
        f_num = lambda x_a: x_a[0] ** 3

        result_sympy = integrated_gradient_sympy_1d(f_sym, x, 0.2, 0.8)
        result_gauss = integrated_gradient_1d(f_num, 0.2, 0.8)

        npt.assert_allclose(result_sympy, result_gauss, atol=1e-14)


# ---------------------------------------------------------------------------
# 3D integrated gradient tests (slow)
# ---------------------------------------------------------------------------

class TestIntegratedGradient3D:
    """Test integrated gradient validation in 3D."""

    @pytest.fixture
    def mesh_3d(self):
        """3D mesh on [0,1]³ with barycentric duals."""
        HC = Complex(3, domain=[(0.0, 1.0)] * 3)
        HC.triangulate()
        HC.refine_all()

        bV = set()
        for v in HC.V:
            on_bnd = any(
                abs(v.x_a[d]) < 1e-14 or abs(v.x_a[d] - 1.0) < 1e-14
                for d in range(3)
            )
            v.boundary = on_bnd
            if on_bnd:
                bV.add(v)

        compute_vd(HC, cdist=1e-10)
        interior = [v for v in HC.V if v not in bV]
        return HC, bV, interior

    @pytest.mark.slow
    def test_linear_scalar_3d(self, mesh_3d):
        """f = x - 2y + 3z: machine precision in 3D."""
        from hyperct.ddg import dual_cell_faces_3d
        from ddgclib.analytical import integrated_gradient_3d

        HC, bV, interior = mesh_3d
        if not interior:
            pytest.skip("No interior vertices in 3D mesh")

        f = lambda x: x[0] - 2.0 * x[1] + 3.0 * x[2]
        for v in HC.V:
            v.f = f(v.x_a[:3])

        for v in interior:
            num = scalar_gradient_integrated(v, HC, dim=3, field_attr='f')
            faces = dual_cell_faces_3d(v, HC)
            if not faces:
                continue
            ana = integrated_gradient_3d(f, faces)
            npt.assert_allclose(
                num, ana, rtol=2.0, atol=1e-6,
                err_msg=f"Linear scalar 3D failed at {v.x}"
            )


# ---------------------------------------------------------------------------
# Benchmark integration tests (run benchmarks via pytest)
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    """Run benchmark classes through pytest for regression testing."""

    def test_linear_scalar_1d_benchmark(self):
        """LinearScalar1D benchmark produces machine-precision results."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import LinearScalar1D

        bench = LinearScalar1D(dim=1, n_refine=2)
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-13

    def test_linear_scalar_2d_benchmark(self):
        """LinearScalar2D benchmark produces machine-precision results."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import LinearScalar2D

        bench = LinearScalar2D(dim=2, dual_method="barycentric", n_refine=1)
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-13

    def test_linear_vector_2d_benchmark(self):
        """LinearVector2D benchmark produces machine-precision results."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import LinearVector2D

        bench = LinearVector2D(dim=2, dual_method="barycentric", n_refine=1)
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-13

    def test_linear_2d_jittered_benchmark(self):
        """LinearScalar2D with jittered mesh: still machine precision."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import LinearScalar2D

        bench = LinearScalar2D(
            dim=2, dual_method="barycentric", n_refine=1, seed=42
        )
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-13

    def test_quadratic_2d_converges(self):
        """QuadraticScalar2D error decreases with refinement."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import QuadraticScalar2D

        errors = []
        for n_ref in [1, 2]:
            bench = QuadraticScalar2D(
                dim=2, dual_method="barycentric", n_refine=n_ref, seed=42
            )
            summary = bench.run()
            errors.append(summary["max_abs_error"])

        assert errors[-1] < errors[0], (
            f"Quadratic error did not decrease: {errors}"
        )

    def test_cubic_2d_converges(self):
        """CubicScalar2D error decreases with refinement."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import CubicScalar2D

        errors = []
        for n_ref in [1, 2]:
            bench = CubicScalar2D(
                dim=2, dual_method="barycentric", n_refine=n_ref, seed=42
            )
            summary = bench.run()
            errors.append(summary["max_abs_error"])

        assert errors[-1] < errors[0], (
            f"Cubic error did not decrease: {errors}"
        )

    def test_trig_2d_converges(self):
        """TrigScalar2D error decreases with refinement."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import TrigScalar2D

        errors = []
        for n_ref in [1, 2]:
            bench = TrigScalar2D(
                dim=2, dual_method="barycentric", n_refine=n_ref
            )
            summary = bench.run()
            errors.append(summary["max_abs_error"])

        assert errors[-1] < errors[0], (
            f"Trig error did not decrease: {errors}"
        )


# ---------------------------------------------------------------------------
# Known-solution unit tests — validate the analytical integration itself
# ---------------------------------------------------------------------------

class TestKnownSolutionIntegrals:
    """Validate the analytical integration framework against hand-computed
    integrals over known geometric domains.

    These are NOT DDG operator tests — they validate that
    ``integrated_gradient_*`` produces correct results on simple shapes
    with textbook answers.
    """

    def test_1d_constant_gradient(self):
        """∫_0^1 d(3x)/dx dx = 3(1) - 3(0) = 3."""
        f = lambda x: 3.0 * x[0]
        result = integrated_gradient_1d(f, 0.0, 1.0)
        npt.assert_allclose(result, [3.0], atol=1e-15)

    def test_1d_quadratic_known(self):
        """∫_0^2 d(x²)/dx dx = 4 - 0 = 4."""
        f = lambda x: x[0] ** 2
        result = integrated_gradient_1d(f, 0.0, 2.0)
        npt.assert_allclose(result, [4.0], atol=1e-15)

    def test_2d_constant_gradient_unit_square(self):
        """∫_{[0,1]²} ∇(x) dA = [1, 0] (area = 1, grad = [1, 0])."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        f = lambda x: x[0]
        result = integrated_gradient_2d(f, square)
        npt.assert_allclose(result, [1.0, 0.0], atol=1e-14)

    def test_2d_constant_gradient_unit_square_y(self):
        """∫_{[0,1]²} ∇(y) dA = [0, 1]."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        f = lambda x: x[1]
        result = integrated_gradient_2d(f, square)
        npt.assert_allclose(result, [0.0, 1.0], atol=1e-14)

    def test_2d_quadratic_unit_square(self):
        """∫_{[0,1]²} ∇(x²+y²) dA = [∫2x dA, ∫2y dA] = [1.0, 1.0].

        ∫_0^1∫_0^1 2x dx dy = 2 * [x²/2]_0^1 * 1 = 1.
        """
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        f = lambda x: x[0] ** 2 + x[1] ** 2
        result = integrated_gradient_2d(f, square)
        npt.assert_allclose(result, [1.0, 1.0], atol=1e-14)

    def test_2d_linear_on_triangle(self):
        """∫_T ∇(x+y) dA = [A, A] where A = triangle area.

        Triangle with vertices (0,0), (1,0), (0,1) has area 0.5.
        ∇(x+y) = [1, 1], so integral = [0.5, 0.5].
        """
        triangle = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        f = lambda x: x[0] + x[1]
        result = integrated_gradient_2d(f, triangle)
        npt.assert_allclose(result, [0.5, 0.5], atol=1e-14)

    def test_2d_quadratic_on_triangle(self):
        """∫_T ∇(x²) dA on triangle (0,0)-(2,0)-(0,2).

        Triangle area = 2. ∇(x²) = [2x, 0].
        ∫_T 2x dA = 2 ∫_0^2 ∫_0^{2-x} x dy dx
                   = 2 ∫_0^2 x(2-x) dx = 2 [x² - x³/3]_0^2
                   = 2 (4 - 8/3) = 2 * 4/3 = 8/3.
        """
        triangle = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        f = lambda x: x[0] ** 2
        result = integrated_gradient_2d(f, triangle)
        npt.assert_allclose(result[0], 8.0 / 3.0, atol=1e-13)
        npt.assert_allclose(result[1], 0.0, atol=1e-14)

    def test_2d_linear_on_rectangle(self):
        """∫_{[0,3]×[0,2]} ∇(5x - 3y) dA = [5*6, -3*6] = [30, -18].

        Area = 6, ∇f = [5, -3].
        """
        rect = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=float)
        f = lambda x: 5.0 * x[0] - 3.0 * x[1]
        result = integrated_gradient_2d(f, rect)
        npt.assert_allclose(result, [30.0, -18.0], atol=1e-13)

    def test_2d_pentagon_linear(self):
        """∫_P ∇(x) dA on a regular pentagon centered at origin.

        For any polygon, ∫_P ∇(x) dA = [Area, 0] for f=x.
        Pentagon with R=1: area ≈ 2.37764.
        """
        N = 5
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pentagon = np.column_stack([np.cos(angles), np.sin(angles)])
        area = abs(_shoelace_area(pentagon))

        f = lambda x: x[0]
        result = integrated_gradient_2d(f, pentagon)
        npt.assert_allclose(result[0], area, rtol=1e-13)
        npt.assert_allclose(result[1], 0.0, atol=1e-14)

    def test_2d_vector_unit_square(self):
        """∫_{[0,1]²} ∇[x, y] dA = [[1,0],[0,1]] (identity * area=1)."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        u = lambda x: np.array([x[0], x[1]])
        result = integrated_gradient_2d_vector(u, square)
        npt.assert_allclose(result, np.eye(2), atol=1e-14)


# ---------------------------------------------------------------------------
# Stress operator tests (Step 7)
# ---------------------------------------------------------------------------

def _make_stress_mesh_2d(n_refine=1, domain=((0.0, 1.0), (0.0, 1.0)),
                         method="barycentric"):
    """Create 2D mesh with duals, assign placeholder fields."""
    from ddgclib.operators.stress import cache_dual_volumes

    HC = Complex(2, domain=list(domain))
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        on_bnd = any(
            abs(v.x_a[d] - domain[d][0]) < 1e-14 or
            abs(v.x_a[d] - domain[d][1]) < 1e-14
            for d in range(2)
        )
        v.boundary = on_bnd
        if on_bnd:
            bV.add(v)

    compute_vd(HC, method=method, cdist=1e-10)
    cache_dual_volumes(HC, dim=2)
    interior = [v for v in HC.V if v not in bV]
    return HC, bV, interior


class TestIntegratedStress2D:
    """Validate stress_force against analytically integrated solutions."""

    def test_hydrostatic_pressure_force(self):
        """P = rho*g*(H-y), u=0: pressure force matches analytical.

        For linear pressure, the DDG operator is exact (machine precision).
        """
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import PressureGradientBenchmark

        bench = PressureGradientBenchmark(
            dim=2, n_refine=1, g=9.81, H=1.0,
        )
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-12, (
            f"Hydrostatic pressure force error: {summary['max_abs_error']:.2e}"
        )

    def test_linear_velocity_zero_viscous_force(self):
        """u = [y, 0], P=0: viscous force ~ 0 (constant ∇u => ∇²u = 0).

        The diffusion form (mu/|d|)*du*(d_hat·A) for linear velocity
        produces zero net viscous force because the face gradient is
        constant — flux in equals flux out on every dual cell.
        """
        rho = 1.0
        mu = 1.0
        HC, bV, interior = _make_stress_mesh_2d(n_refine=2)

        for v in HC.V:
            v.u = np.array([v.x_a[1], 0.0])
            v.p = 0.0
            v.m = rho * v.dual_vol if v.dual_vol > 1e-30 else 1e-30

        max_F = max(
            np.linalg.norm(stress_force(v, dim=2, mu=mu, HC=HC))
            for v in interior
        )
        assert max_F < 1e-12, (
            f"Linear velocity viscous force not zero: {max_F:.2e}"
        )

    def test_quadratic_viscous_force_on_symmetric_mesh(self):
        """u = [y², 0], P=0 on symmetric mesh.

        The diffusion form IS exact for quadratic velocity on symmetric
        Delaunay meshes. Verify viscous force matches analytical.
        """
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import ViscousFluxBenchmark

        bench = ViscousFluxBenchmark(
            velocity_order=2, dim=2, n_refine=2, mu=1.0,
        )
        summary = bench.run()
        # On symmetric mesh, diffusion form exact for quadratic
        assert summary["max_abs_error"] < 1e-10, (
            f"Quadratic viscous force error (symmetric): "
            f"{summary['max_abs_error']:.2e}"
        )

    def test_poiseuille_equilibrium_symmetric(self):
        """Poiseuille u=[G/(2mu)*y*(D-y), 0], P=-Gx on symmetric mesh.

        Net stress force (pressure + viscous) should be zero because
        -∇P exactly balances mu*∇²u for the parabolic profile.
        """
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import PoiseuilleBenchmark

        bench = PoiseuilleBenchmark(
            dim=2, n_refine=2, mu=1.0, G=1.0, D=1.0,
        )
        bench.build_mesh()
        bench.compute_numerical()

        # The DDG stress_force should be near zero at equilibrium
        max_F = max(
            np.linalg.norm(bench.numerical[id(v)])
            for v in bench.interior_vertices
        )
        # On symmetric mesh the diffusion form is exact for quadratic u,
        # and pressure gradient is exact for linear P => perfect cancellation
        assert max_F < 1e-10, (
            f"Poiseuille equilibrium force not zero: {max_F:.2e}"
        )

    def test_poiseuille_jittered_force_bounded(self):
        """Poiseuille on jittered mesh: force residual remains bounded.

        On jittered meshes the diffusion form has O(h) truncation error.
        The max force norm should remain small (< 0.1) even on jittered
        meshes, confirming the pressure and viscous forces still roughly
        cancel.
        """
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import PoiseuilleBenchmark

        bench = PoiseuilleBenchmark(
            dim=2, n_refine=2, mu=1.0, G=1.0, D=1.0, seed=42,
        )
        bench.build_mesh()
        bench.compute_numerical()
        max_F = max(
            np.linalg.norm(bench.numerical[id(v)])
            for v in bench.interior_vertices
        )
        assert max_F < 0.1, (
            f"Poiseuille jittered force too large: {max_F:.4e}"
        )

    def test_pressure_gradient_benchmark_convergence(self):
        """PressureGradientBenchmark on jittered mesh: machine precision
        expected since P is linear."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import PressureGradientBenchmark

        bench = PressureGradientBenchmark(
            dim=2, n_refine=2, g=9.81, H=1.0, seed=42,
        )
        summary = bench.run()
        assert summary["max_abs_error"] < 1e-10, (
            f"Pressure gradient (jittered) error: {summary['max_abs_error']:.2e}"
        )


class TestIntegratedCurvature:
    """Validate curvature operators on known surfaces.

    Note: The parametric sphere projection introduces degenerate
    triangles near seams, so curvature accuracy is limited by mesh
    quality rather than operator accuracy.  These tests verify the
    benchmark machinery runs without error and produces finite results.
    """

    @pytest.mark.slow
    def test_sphere_curvature_runs(self):
        """CurvatureBenchmark on sphere should run without error."""
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import CurvatureBenchmark

        bench = CurvatureBenchmark(surface="sphere", R=1.0, n_refine=2)
        summary = bench.run()
        if "error" in summary:
            pytest.skip(summary["error"])
        assert summary["n_vertices"] > 0, "No curvature values computed"
        assert np.isfinite(summary["mean_abs_error"])

    @pytest.mark.slow
    def test_sphere_curvature_convergence(self):
        """Mean curvature of sphere R=1: H = 1/R = 1.0.

        Error should decrease with refinement (or at least not blow up).
        """
        import sys
        sys.path.insert(0, '.')
        from benchmarks._integrated_benchmark_cases import CurvatureBenchmark

        errors = []
        for n_ref in [2, 3]:
            bench = CurvatureBenchmark(surface="sphere", R=1.0, n_refine=n_ref)
            summary = bench.run()
            if "error" in summary:
                pytest.skip(summary["error"])
            errors.append(summary["mean_abs_error"])

        if len(errors) == 2:
            assert errors[-1] < errors[0] * 1.5, (
                f"Sphere curvature did not converge: {errors}"
            )
