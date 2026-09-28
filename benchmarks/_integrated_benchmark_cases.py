"""
Concrete benchmark cases for integrated gradient validation.

Each class defines a test function and its expected behavior.  Linear
fields should produce machine-precision agreement; non-linear fields
should converge with mesh refinement.

Also includes stress operator benchmarks (Step 7) that validate
``stress_force`` against analytically integrated solutions.

Usage::

    from benchmarks._integrated_benchmark_cases import (
        LinearScalar1D, QuadraticScalar2D, run_all_gradient_benchmarks,
        PressureGradientBenchmark, ViscousFluxBenchmark,
        PoiseuilleBenchmark, CurvatureBenchmark,
    )

    # Run a single benchmark
    bench = LinearScalar1D(dim=1, dual_method="barycentric")
    results = bench.run()
    print(results)

    # Run the full suite
    results = run_all_gradient_benchmarks()
"""
from __future__ import annotations

import math

import numpy as np

from ._integrated_benchmark_classes import (
    IntegratedGradientBenchmark,
    run_integrated_benchmarks,
)


# ---------------------------------------------------------------------------
# Linear fields — must give machine precision (~1e-14)
# ---------------------------------------------------------------------------

class LinearScalar1D(IntegratedGradientBenchmark):
    """f(x) = 2x + 1.  Gradient = [2].  Machine precision expected."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_scalar_1d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return 2.0 * x[0] + 1.0


class LinearVector1D(IntegratedGradientBenchmark):
    """u(x) = [3x + 1].  Gradient = [[3]].  Machine precision expected."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_vector_1d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([3.0 * x[0] + 1.0])


class LinearScalar2D(IntegratedGradientBenchmark):
    """f(x,y) = 3x - 2y + 1.  Gradient = [3, -2].  Machine precision."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_scalar_2d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return 3.0 * x[0] - 2.0 * x[1] + 1.0


class LinearVector2D(IntegratedGradientBenchmark):
    """u(x,y) = [2x+3y, -x+y].  Gradient = [[2,3],[-1,1]].  Machine."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_vector_2d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([2.0 * x[0] + 3.0 * x[1], -x[0] + x[1]])


class LinearScalar3D(IntegratedGradientBenchmark):
    """f(x,y,z) = x - 2y + 3z.  Gradient = [1, -2, 3].  Machine."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_scalar_3d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return x[0] - 2.0 * x[1] + 3.0 * x[2]


class LinearVector3D(IntegratedGradientBenchmark):
    """u = [x+2y-z, -x+3z, 2y+z].  Machine precision."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "linear_vector_3d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([
            x[0] + 2.0 * x[1] - x[2],
            -x[0] + 3.0 * x[2],
            2.0 * x[1] + x[2],
        ])


# ---------------------------------------------------------------------------
# Quadratic fields — convergent with mesh refinement
# ---------------------------------------------------------------------------

class QuadraticScalar1D(IntegratedGradientBenchmark):
    """f(x) = x².  Gradient = [2x]."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_scalar_1d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return x[0] ** 2


class QuadraticVector1D(IntegratedGradientBenchmark):
    """u(x) = [x²].  Gradient = [[2x]]."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_vector_1d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([x[0] ** 2])


class QuadraticScalar2D(IntegratedGradientBenchmark):
    """f(x,y) = x² + y².  Gradient = [2x, 2y]."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_scalar_2d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return x[0] ** 2 + x[1] ** 2


class QuadraticVector2D(IntegratedGradientBenchmark):
    """u(x,y) = [x², y²].  Gradient = [[2x, 0], [0, 2y]]."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_vector_2d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([x[0] ** 2, x[1] ** 2])


class QuadraticScalar3D(IntegratedGradientBenchmark):
    """f(x,y,z) = x² + y² + z².  Gradient = [2x, 2y, 2z]."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_scalar_3d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


class QuadraticVector3D(IntegratedGradientBenchmark):
    """u(x,y,z) = [x², y², z²].  Gradient = diag(2x, 2y, 2z)."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "quadratic_vector_3d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([x[0] ** 2, x[1] ** 2, x[2] ** 2])


# ---------------------------------------------------------------------------
# Higher-order non-linear fields — convergent
# ---------------------------------------------------------------------------

class CubicScalar2D(IntegratedGradientBenchmark):
    """f(x,y) = x³ + x·y².  Gradient = [3x²+y², 2xy]."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "cubic_scalar_2d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return x[0] ** 3 + x[0] * x[1] ** 2


class CubicVector2D(IntegratedGradientBenchmark):
    """u(x,y) = [x²y, xy²].  Gradient = [[2xy, x²], [y², 2xy]]."""

    field_type = "vector"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "cubic_vector_2d")
        super().__init__(**kwargs)

    def u_callable(self, x):
        return np.array([x[0] ** 2 * x[1], x[0] * x[1] ** 2])


class TrigScalar2D(IntegratedGradientBenchmark):
    """f(x,y) = sin(πx)cos(πy).  Non-polynomial — convergent."""

    field_type = "scalar"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "trig_scalar_2d")
        super().__init__(**kwargs)

    def f_callable(self, x):
        return math.sin(math.pi * x[0]) * math.cos(math.pi * x[1])


class PoiseuilleVector2D(IntegratedGradientBenchmark):
    """u(x,y) = [G/(2μ) · y·(D-y), 0].  Poiseuille profile (quadratic)."""

    field_type = "vector"

    def __init__(self, G: float = 1.0, mu: float = 0.1, D: float = 1.0,
                 **kwargs):
        kwargs.setdefault("name", "poiseuille_vector_2d")
        super().__init__(**kwargs)
        self.G = G
        self.mu = mu
        self.D = D

    def u_callable(self, x):
        return np.array([
            self.G / (2.0 * self.mu) * x[1] * (self.D - x[1]),
            0.0,
        ])


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

# All benchmark classes, organized by category
LINEAR_BENCHMARKS = [
    LinearScalar1D, LinearVector1D,
    LinearScalar2D, LinearVector2D,
    LinearScalar3D, LinearVector3D,
]

NONLINEAR_BENCHMARKS = [
    QuadraticScalar1D, QuadraticVector1D,
    QuadraticScalar2D, QuadraticVector2D,
    CubicScalar2D, CubicVector2D,
    TrigScalar2D,
    PoiseuilleVector2D,
    QuadraticScalar3D, QuadraticVector3D,
]

ALL_GRADIENT_BENCHMARKS = LINEAR_BENCHMARKS + NONLINEAR_BENCHMARKS


# ---------------------------------------------------------------------------
# Stress operator benchmarks (Step 7)
# ---------------------------------------------------------------------------

class StressGradientBenchmark:
    """Base class for stress operator validation benchmarks.

    Compares ``stress_force`` against analytically integrated forces
    over dual cells.  Subclasses define velocity, pressure, and
    physical parameters.

    The pipeline:
    1. Build mesh, compute duals
    2. Assign velocity, pressure, mass fields
    3. Compute ``stress_force`` (DDG integrated) per interior vertex
    4. Compute analytical force per interior vertex via divergence theorem
    5. Report per-vertex and aggregate errors

    Parameters
    ----------
    name : str
        Benchmark name.
    dim : int
        Spatial dimension.
    domain : list of tuple, optional
        Domain bounds.
    dual_method : str
        ``"barycentric"`` or ``"circumcentric"``.
    n_refine : int
        Refinement passes.
    seed : int or None
        Random seed for jittering (None = symmetric).
    mu : float
        Dynamic viscosity.
    rho : float
        Density.
    G : float
        Pressure gradient magnitude.
    """

    def __init__(
        self,
        name: str = "stress_benchmark",
        dim: int = 2,
        domain: list[tuple[float, float]] | None = None,
        dual_method: str = "barycentric",
        polygon_method: str = "barycentric_dual_p_ij",
        n_refine: int = 1,
        seed: int | None = None,
        jitter_amplitude: float = 0.05,
        n_gauss: int = 10,
        mu: float = 1.0,
        rho: float = 1.0,
        G: float = 1.0,
    ):
        self.name = name
        self.dim = dim
        self.domain = domain or [(0.0, 1.0)] * dim
        self.dual_method = dual_method
        self.polygon_method = polygon_method
        self.n_refine = n_refine
        self.seed = seed
        self.jitter_amplitude = jitter_amplitude
        self.n_gauss = n_gauss
        self.mu = mu
        self.rho = rho
        self.G = G

        self.HC = None
        self.bV: set = set()
        self.interior_vertices: list = []
        self.numerical: dict = {}
        self.analytical: dict = {}
        self.errors: dict = {}
        self.rel_errors: dict = {}

    def u_field(self, x: np.ndarray) -> np.ndarray:
        """Velocity field.  Override in subclasses."""
        return np.zeros(self.dim)

    def p_field(self, x: np.ndarray) -> float:
        """Pressure field.  Override in subclasses."""
        return 0.0

    def analytical_force(self, v) -> np.ndarray:
        """Analytical integrated force on dual cell of v.

        Default: compute analytically via divergence theorem integration
        of the Cauchy stress tensor sigma = -p*I + mu*(grad_u + grad_u^T).

        For incompressible flow:
            F_i = ∫_{V_i} div(sigma) dV = ∮_{∂V_i} sigma · n dA

        This is evaluated as:
            F_i = -∮ p*n dA + mu * ∮ (grad_u + grad_u^T) · n dA
        """
        from hyperct.ddg import dual_cell_polygon_2d
        from ddgclib.analytical._divergence_theorem import (
            integrated_gradient_2d,
        )

        include_midpts = (self.polygon_method == "barycentric_dual_p_ij")

        if self.dim == 2:
            polygon = dual_cell_polygon_2d(
                v, include_edge_midpoints=include_midpts
            )
            # Pressure force: F_p = -∮ p * n dA = -∫ ∇p dV
            F_p = -integrated_gradient_2d(
                self.p_field, polygon, n_gauss=self.n_gauss
            )
            # Viscous force via surface integral of stress tensor:
            # F_v = ∮ mu * (∇u + ∇u^T) · n dA
            F_v = self._viscous_force_analytical_2d(polygon)
            return F_p + F_v

        elif self.dim == 1:
            from hyperct.ddg import dual_cell_vertices_1d
            a, b = dual_cell_vertices_1d(v)
            # F_p = -[p(b) - p(a)] (1D pressure gradient integral)
            F_p = np.array([-(self.p_field(np.array([b])) -
                              self.p_field(np.array([a])))])
            # F_v = mu * [du/dx(b) - du/dx(a)]
            # For the diffusion form: this is the flux difference
            F_v = self._viscous_force_analytical_1d(a, b)
            return F_p + F_v

        raise NotImplementedError(f"dim={self.dim}")

    def _viscous_force_analytical_2d(self, polygon: np.ndarray) -> np.ndarray:
        """Compute ∮ mu * (∇u + ∇u^T) · n dA analytically via quadrature.

        This is the surface integral of the viscous stress tensor.
        For each edge of the polygon, evaluate sigma_v · n using
        the analytical velocity gradient.
        """
        from ddgclib.analytical._divergence_theorem import _gauss_legendre_01

        nodes, weights = _gauss_legendre_01(self.n_gauss)
        F_v = np.zeros(2)
        N = len(polygon)

        for k in range(N):
            P_k = polygon[k]
            P_next = polygon[(k + 1) % N]
            edge = P_next - P_k
            # Outward normal (unnormalized, |n| = |edge|)
            n_out = np.array([edge[1], -edge[0]])

            for t, w in zip(nodes, weights):
                x_t = P_k + t * edge
                # Evaluate ∇u analytically via finite difference
                grad_u = self._grad_u_analytical(x_t)
                # sigma_v = mu * (∇u + ∇u^T)
                sigma_v = self.mu * (grad_u + grad_u.T)
                # sigma_v · n_out (but n_out already has magnitude |edge|)
                F_v += w * sigma_v @ n_out

        return F_v

    def _grad_u_analytical(self, x: np.ndarray) -> np.ndarray:
        """Analytical velocity gradient at point x.

        Override for exact gradients; default uses finite differences.
        """
        eps = 1e-8
        dim = self.dim
        u0 = self.u_field(x)
        grad = np.zeros((dim, dim))
        for j in range(dim):
            x_plus = x.copy()
            x_plus[j] += eps
            grad[:, j] = (self.u_field(x_plus) - u0) / eps
        return grad

    def _viscous_force_analytical_1d(self, a: float, b: float) -> np.ndarray:
        """1D viscous force: mu * [du/dx(b) - du/dx(a)]."""
        eps = 1e-8
        dudx_b = (self.u_field(np.array([b + eps]))[0] -
                  self.u_field(np.array([b - eps]))[0]) / (2 * eps)
        dudx_a = (self.u_field(np.array([a + eps]))[0] -
                  self.u_field(np.array([a - eps]))[0]) / (2 * eps)
        return np.array([self.mu * (dudx_b - dudx_a)])

    def build_mesh(self):
        """Create mesh, tag boundaries, compute duals, assign fields."""
        from hyperct import Complex
        from hyperct.ddg import compute_vd
        from ddgclib.operators.stress import cache_dual_volumes

        HC = Complex(self.dim, domain=self.domain)
        HC.triangulate()
        for _ in range(self.n_refine):
            HC.refine_all()

        if self.seed is not None:
            self._jitter_mesh(HC)

        bV = set()
        for v in HC.V:
            on_boundary = False
            for d in range(self.dim):
                lo, hi = self.domain[d]
                if abs(v.x_a[d] - lo) < 1e-14 or abs(v.x_a[d] - hi) < 1e-14:
                    on_boundary = True
                    break
            v.boundary = on_boundary
            if on_boundary:
                bV.add(v)

        compute_vd(HC, method=self.dual_method, cdist=1e-10)

        # Assign fields
        for v in HC.V:
            x = v.x_a[:self.dim]
            v.u = self.u_field(x)
            v.p = self.p_field(x)

        # Mass from density * dual volume
        cache_dual_volumes(HC, dim=self.dim)
        for v in HC.V:
            v.m = self.rho * v.dual_vol if v.dual_vol > 1e-30 else 1e-30

        self.HC = HC
        self.bV = bV
        self.interior_vertices = [v for v in HC.V if v not in bV]

    def _jitter_mesh(self, HC):
        """Reuse the jitter logic from IntegratedGradientBenchmark."""
        from scipy.spatial import Delaunay
        rng = np.random.default_rng(self.seed)
        to_move = []
        for v in HC.V:
            on_boundary = False
            for d in range(self.dim):
                lo, hi = self.domain[d]
                if abs(v.x_a[d] - lo) < 1e-14 or abs(v.x_a[d] - hi) < 1e-14:
                    on_boundary = True
                    break
            if not on_boundary:
                if v.nn:
                    edge_len = min(
                        np.linalg.norm(v.x_a - vn.x_a) for vn in v.nn
                    )
                else:
                    edge_len = 0.1
                offset = rng.uniform(
                    -self.jitter_amplitude * edge_len,
                    self.jitter_amplitude * edge_len,
                    size=self.dim,
                )
                new_x = tuple(v.x_a[d] + offset[d] for d in range(self.dim))
                to_move.append((v, new_x))
        for v, new_x in to_move:
            HC.V.move(v, new_x)

        verts = list(HC.V)
        for v in verts:
            for nb in list(v.nn):
                v.disconnect(nb)
        if self.dim == 1:
            sorted_verts = sorted(verts, key=lambda v: v.x_a[0])
            for i in range(len(sorted_verts) - 1):
                sorted_verts[i].connect(sorted_verts[i + 1])
        else:
            from ddgclib.geometry import connect_and_cache_simplices
            coords = np.array([v.x_a[:self.dim] for v in verts])
            connect_and_cache_simplices(
                HC, verts, self.dim, coords=coords,
            )

    def compute_numerical(self):
        """Compute stress_force for each interior vertex."""
        from ddgclib.operators.stress import stress_force
        for v in self.interior_vertices:
            self.numerical[id(v)] = stress_force(
                v, dim=self.dim, mu=self.mu, HC=self.HC
            )

    def compute_analytical(self):
        """Compute analytical force for each interior vertex."""
        for v in self.interior_vertices:
            self.analytical[id(v)] = self.analytical_force(v)

    def compute_errors(self):
        """Compute absolute and relative errors."""
        for v in self.interior_vertices:
            vid = id(v)
            num = self.numerical[vid]
            ana = self.analytical[vid]
            abs_err = np.linalg.norm(num - ana)
            ana_norm = np.linalg.norm(ana)
            rel_err = abs_err / ana_norm if ana_norm > 1e-30 else abs_err
            self.errors[vid] = abs_err
            self.rel_errors[vid] = rel_err

    def run(self) -> dict:
        """Full benchmark pipeline."""
        self.build_mesh()
        self.compute_numerical()
        self.compute_analytical()
        self.compute_errors()
        return self.summary()

    def summary(self) -> dict:
        if not self.errors:
            return {}
        abs_errs = np.array(list(self.errors.values()))
        rel_errs = np.array(list(self.rel_errors.values()))
        return {
            "name": self.name,
            "dim": self.dim,
            "dual_method": self.dual_method,
            "polygon_method": self.polygon_method,
            "n_refine": self.n_refine,
            "n_interior": len(self.interior_vertices),
            "seed": self.seed,
            "mu": self.mu,
            "max_abs_error": float(np.max(abs_errs)),
            "mean_abs_error": float(np.mean(abs_errs)),
            "max_rel_error": float(np.max(rel_errs)),
            "mean_rel_error": float(np.mean(rel_errs)),
        }


class PressureGradientBenchmark(StressGradientBenchmark):
    """Hydrostatic pressure P = rho*g*(H-y), zero velocity.

    Analytical force: F_i = -∫_{V_i} ∇P dV = -rho*g*[0, -1]*Vol_i
    (constant gradient, so exact at machine precision).
    """

    def __init__(self, g: float = 9.81, H: float = 1.0, **kwargs):
        kwargs.setdefault("name", "pressure_gradient")
        kwargs.setdefault("mu", 0.0)
        super().__init__(**kwargs)
        self.g = g
        self.H = H

    def u_field(self, x):
        return np.zeros(self.dim)

    def p_field(self, x):
        # Gravity along last axis (y in 2D)
        return self.rho * self.g * (self.H - x[-1])

    def _grad_u_analytical(self, x):
        return np.zeros((self.dim, self.dim))

    def analytical_force(self, v):
        """Pressure force: F = -∫ ∇P dV.

        Since ∇P = [0, ..., -rho*g], the force is [0, ..., rho*g*Vol_i].
        """
        from hyperct.ddg import dual_cell_polygon_2d
        from ddgclib.analytical._divergence_theorem import integrated_gradient_2d

        include_midpts = (self.polygon_method == "barycentric_dual_p_ij")

        if self.dim == 2:
            polygon = dual_cell_polygon_2d(
                v, include_edge_midpoints=include_midpts
            )
            return -integrated_gradient_2d(
                self.p_field, polygon, n_gauss=self.n_gauss
            )
        elif self.dim == 1:
            from hyperct.ddg import dual_cell_vertices_1d
            a, b = dual_cell_vertices_1d(v)
            return np.array([-(self.p_field(np.array([b])) -
                               self.p_field(np.array([a])))])
        raise NotImplementedError


class ViscousFluxBenchmark(StressGradientBenchmark):
    """Validate viscous force with zero pressure.

    Linear velocity u=[y, 0]: viscous force should be zero (constant
    gradient => ∇²u = 0 => no viscous flux difference).

    Quadratic velocity u=[y², 0]: viscous force is non-zero and the
    diffusion form (mu/|d|)*du*(d_hat·A) IS exact for quadratic
    velocity on symmetric (Delaunay) meshes because the face gradient
    is exact when u varies quadratically and mesh is symmetric.
    """

    def __init__(self, velocity_order: int = 1, D: float = 1.0, **kwargs):
        kwargs.setdefault("name", f"viscous_flux_order{velocity_order}")
        kwargs.setdefault("G", 0.0)
        super().__init__(**kwargs)
        self.velocity_order = velocity_order
        self.D = D

    def u_field(self, x):
        if self.velocity_order == 1:
            # Linear: u = [y, 0]
            return np.array([x[-1], 0.0]) if self.dim == 2 else np.array([x[0]])
        elif self.velocity_order == 2:
            # Quadratic: u = [y², 0]
            return np.array([x[-1] ** 2, 0.0]) if self.dim == 2 else np.array([x[0] ** 2])
        raise ValueError(f"velocity_order={self.velocity_order}")

    def p_field(self, x):
        return 0.0

    def _grad_u_analytical(self, x):
        if self.dim == 2:
            if self.velocity_order == 1:
                # ∂u_x/∂y = 1, all others zero
                return np.array([[0.0, 1.0], [0.0, 0.0]])
            elif self.velocity_order == 2:
                # ∂u_x/∂y = 2y, all others zero
                return np.array([[0.0, 2.0 * x[-1]], [0.0, 0.0]])
        raise NotImplementedError


class PoiseuilleBenchmark(StressGradientBenchmark):
    """Full Poiseuille: pressure + viscous forces cancel at equilibrium.

    u_x(y) = G/(2*mu) * y * (D - y),  u_y = 0
    P(x) = -G * x

    The net stress force should be zero (machine precision on symmetric
    meshes, convergent on jittered meshes).

    The diffusion form is exact for the quadratic Poiseuille profile
    on Delaunay meshes: each edge has du = u_j - u_i which, combined
    with the face-centered diffusion (mu/|d|)*du*(d_hat·A), exactly
    reconstructs the viscous flux for a quadratic field.
    """

    def __init__(self, D: float = 1.0, **kwargs):
        kwargs.setdefault("name", "poiseuille_equilibrium")
        super().__init__(**kwargs)
        self.D = D

    def u_field(self, x):
        if self.dim == 2:
            return np.array([
                self.G / (2.0 * self.mu) * x[1] * (self.D - x[1]),
                0.0,
            ])
        elif self.dim == 1:
            return np.array([
                self.G / (2.0 * self.mu) * x[0] * (self.D - x[0]),
            ])
        raise NotImplementedError

    def p_field(self, x):
        if self.dim == 2:
            return -self.G * x[0]
        return 0.0

    def _grad_u_analytical(self, x):
        if self.dim == 2:
            dudy = self.G / (2.0 * self.mu) * (self.D - 2.0 * x[1])
            return np.array([[0.0, dudy], [0.0, 0.0]])
        raise NotImplementedError

    def analytical_force(self, v):
        """At Poiseuille equilibrium, net force = 0 on every parcel."""
        # Compute pressure + viscous analytically and return their sum.
        # For validation, we actually compute both terms and sum them.
        return super().analytical_force(v)


class CurvatureBenchmark:
    """Validate integrated mean curvature on known surfaces.

    For a sphere of radius R, the integrated mean curvature over each
    dual cell should be H * A_dual where H = 1/R (mean curvature) and
    A_dual is the dual cell area.

    For a cylinder of radius R, H = 1/(2R).

    This benchmark builds a mesh on the surface and checks that the
    DDG curvature operator matches the analytical value.
    """

    def __init__(
        self,
        surface: str = "sphere",
        R: float = 1.0,
        n_refine: int = 2,
        name: str | None = None,
    ):
        self.surface = surface
        self.R = R
        self.n_refine = n_refine
        self.name = name or f"curvature_{surface}"
        self.errors: dict = {}
        self.rel_errors: dict = {}

    def analytical_H(self) -> float:
        """Analytical mean curvature."""
        if self.surface == "sphere":
            return 1.0 / self.R
        elif self.surface == "cylinder":
            return 1.0 / (2.0 * self.R)
        raise ValueError(f"Unknown surface: {self.surface}")

    def run(self) -> dict:
        """Build surface mesh and compare DDG curvature to analytical.

        Uses the ddgclib curvature pipeline which operates on surface
        meshes embedded in 3D.  Per-vertex curvature is computed via
        ``b_curvatures_hn_ij_c_ij(F, nn, n_i)`` and compared against
        the analytical mean curvature ``H = 1/R`` (sphere).
        """
        try:
            from ddgclib._curvatures import (
                b_curvatures_hn_ij_c_ij,
                vectorise_vnn,
                construct_HC,
            )
            from ddgclib._curvatures import normalized
        except ImportError:
            return {"name": self.name, "error": "curvature module not available"}

        from hyperct import Complex

        if self.surface == "sphere":
            # Build a 2D mesh and project to sphere surface
            HC = Complex(2, domain=[(0.15, 0.85)] * 2)
            HC.triangulate()
            for _ in range(self.n_refine):
                HC.refine_all()

            verts = list(HC.V)

            # Map [0.15, 0.85]^2 -> sphere via spherical coordinates
            # (avoid poles/seams by restricting parameter range)
            pos_map = {}
            for v in verts:
                x = np.array(v.x_a[:2])
                theta = np.pi * x[0]
                phi = 2.0 * np.pi * x[1]
                pos_3d = np.array([
                    self.R * np.sin(theta) * np.cos(phi),
                    self.R * np.sin(theta) * np.sin(phi),
                    self.R * np.cos(theta),
                ])
                pos_map[v.x] = pos_3d

            # Build F (vertex positions) and nn (neighbor indices)
            F = np.array([pos_map[v.x] for v in verts])
            nn = []
            for v in verts:
                nn_idx = [verts.index(nb) for nb in v.nn]
                nn.append(nn_idx)

            # Identify boundary vertices (on edges of parameter domain)
            bV_idx = set()
            for i, v in enumerate(verts):
                x = v.x_a[:2]
                if (abs(x[0] - 0.15) < 1e-12 or abs(x[0] - 0.85) < 1e-12 or
                        abs(x[1] - 0.15) < 1e-12 or abs(x[1] - 0.85) < 1e-12):
                    bV_idx.add(i)

            # Reconstruct HC with 3D positions for vectorise_vnn
            HC_3d = construct_HC(F, nn)
            verts_3d = list(HC_3d.V)
            bV_3d = {verts_3d[i] for i in bV_idx if i < len(verts_3d)}

            H_analytical = self.analytical_H()
            abs_errs = []
            rel_errs = []

            # Compute curvature per interior vertex
            for i, v in enumerate(verts_3d):
                if i in bV_idx or v in bV_3d:
                    continue
                try:
                    Fv, nnv = vectorise_vnn(v)
                    n_i = normalized(v.x_a)[0]  # outward normal = position/R
                    c_outd = b_curvatures_hn_ij_c_ij(Fv, nnv, n_i=n_i)
                    HN_i = c_outd.get('HN_i')
                    if HN_i is not None:
                        H_v = float(np.linalg.norm(np.array(HN_i)))
                        err = abs(H_v - H_analytical)
                        abs_errs.append(err)
                        rel_errs.append(err / abs(H_analytical))
                except Exception:
                    continue

            if not abs_errs:
                return {"name": self.name, "error": "no curvature values computed"}

            abs_errs = np.array(abs_errs)
            rel_errs = np.array(rel_errs)
            return {
                "name": self.name,
                "surface": self.surface,
                "R": self.R,
                "H_analytical": H_analytical,
                "n_refine": self.n_refine,
                "n_vertices": len(abs_errs),
                "max_abs_error": float(np.max(abs_errs)),
                "mean_abs_error": float(np.mean(abs_errs)),
                "max_rel_error": float(np.max(rel_errs)),
                "mean_rel_error": float(np.mean(rel_errs)),
            }

        return {"name": self.name, "error": f"surface={self.surface} not implemented"}


# Stress benchmark collections
STRESS_BENCHMARKS = [
    PressureGradientBenchmark,
    ViscousFluxBenchmark,
    PoiseuilleBenchmark,
]

ALL_BENCHMARKS = ALL_GRADIENT_BENCHMARKS


def run_all_gradient_benchmarks(
    dims: list[int] | None = None,
    dual_methods: list[str] | None = None,
    seeds: list[int | None] | None = None,
    n_refine: int = 1,
) -> list[dict]:
    """Run the full integrated gradient benchmark suite.

    Parameters
    ----------
    dims : list of int, optional
        Dimensions to test.  Each benchmark filters to its compatible dim.
    dual_methods : list of str, optional
        Dual methods to test (default: ["barycentric"]).
    seeds : list of (int or None), optional
        Seeds to test (default: [None, 42]).
    n_refine : int
        Number of refinement passes.

    Returns
    -------
    list of dict
        Summary results for each benchmark run.
    """
    dims = dims or [1, 2]
    dual_methods = dual_methods or ["barycentric"]
    seeds = seeds or [None, 42]

    results = []
    for BenchClass in ALL_GRADIENT_BENCHMARKS:
        for dim in dims:
            # Skip benchmarks not compatible with this dimension
            # (e.g. 3D benchmarks when testing 1D)
            bench_test = BenchClass(dim=dim)
            try:
                bench_test.f_callable(np.zeros(dim)) if bench_test.field_type == "scalar" else bench_test.u_callable(np.zeros(dim))
            except (IndexError, ValueError):
                continue

            for method in dual_methods:
                for seed in seeds:
                    try:
                        bench = BenchClass(
                            dim=dim,
                            dual_method=method,
                            n_refine=n_refine,
                            seed=seed,
                        )
                        summary = bench.run()
                        results.append(summary)
                    except Exception as e:
                        results.append({
                            "name": BenchClass.__name__,
                            "dim": dim,
                            "dual_method": method,
                            "seed": seed,
                            "error": str(e),
                        })
    return results
