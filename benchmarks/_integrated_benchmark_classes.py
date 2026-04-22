"""
Reusable benchmark framework for validating DDG integrated operators
against analytically integrated solutions over dual cells.

The key comparison is::

    numerical:  Df_i = 0.5 * sum_j (f_j - f_i) * A_ij   (DDG operator)
    analytical: ∫_{V_i} ∇f dV = ∮_{∂V_i} f · n dA        (divergence theorem)

Both quantities are integrated over the exact same dual cell domain,
making the comparison mathematically consistent.

Usage::

    from benchmarks._integrated_benchmark_classes import (
        IntegratedGradientBenchmark,
    )

    class MyBenchmark(IntegratedGradientBenchmark):
        def f_callable(self, x):
            return x[0]**2 + x[1]**2

    bench = MyBenchmark("quadratic_2d", dim=2)
    bench.run()
    print(bench.summary())
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from hyperct import Complex
from hyperct.ddg import compute_vd, dual_cell_polygon_2d, dual_cell_vertices_1d

from ddgclib.analytical import (
    integrated_gradient_1d,
    integrated_gradient_2d,
)
from ddgclib.analytical._divergence_theorem import integrated_gradient_2d_vector
from ddgclib.operators.stress import (
    dual_area_vector,
    scalar_gradient_integrated,
    velocity_difference_tensor,
)

logger = logging.getLogger(__name__)


class IntegratedGradientBenchmark:
    """Base class for integrated gradient validation benchmarks.

    Compares the DDG integrated gradient operator against analytically
    integrated gradients (via the divergence theorem) over the same
    dual cell domains.

    Subclasses must override :meth:`f_callable` and optionally
    :meth:`f_sympy`.  For vector fields, override :meth:`u_callable`
    instead and set ``self.field_type = "vector"``.

    Parameters
    ----------
    name : str
        Benchmark name.
    dim : int
        Spatial dimension (1, 2, or 3).
    domain : list of tuple, optional
        Domain bounds per dimension.  Defaults to unit hypercube.
    dual_method : str
        Dual mesh method passed to ``compute_vd``:
        ``"barycentric"`` or ``"circumcentric"``.
    polygon_method : str
        Dual cell polygon formulation:

        - ``"barycentric_dual_p_ij"`` (default): polygon includes both
          dual vertices (barycenters/circumcenters) AND edge midpoints.
          This is the standard DEC dual cell.
        - ``"barycentric"``: polygon uses only the dual vertices in
          ``v.vd`` (no explicit edge midpoints for interior edges).
    n_refine : int
        Number of ``refine_all()`` calls.
    seed : int or None
        Random seed for vertex jittering.  ``None`` = symmetric mesh.
    jitter_amplitude : float
        Amplitude of vertex perturbation (fraction of edge length).
    n_gauss : int
        Gauss quadrature points for analytical integration.
    """

    field_type: str = "scalar"  # "scalar" or "vector"

    def __init__(
        self,
        name: str = "unnamed",
        dim: int = 2,
        domain: list[tuple[float, float]] | None = None,
        dual_method: str = "barycentric",
        polygon_method: str = "barycentric_dual_p_ij",
        n_refine: int = 1,
        seed: int | None = None,
        jitter_amplitude: float = 0.05,
        n_gauss: int = 10,
    ):
        self.name = name
        self.dim = dim
        self.domain = domain or [(0.0, 1.0)] * dim
        self.dual_method = dual_method
        self.polygon_method = polygon_method
        self.include_edge_midpoints = (polygon_method == "barycentric_dual_p_ij")
        self.n_refine = n_refine
        self.seed = seed
        self.jitter_amplitude = jitter_amplitude
        self.n_gauss = n_gauss

        # Results (populated by run())
        self.HC = None
        self.bV: set = set()
        self.interior_vertices: list = []
        self.numerical: dict = {}   # vertex -> ndarray
        self.analytical: dict = {}  # vertex -> ndarray
        self.errors: dict = {}      # vertex -> float (absolute error)
        self.rel_errors: dict = {}  # vertex -> float (relative error)

    # --- Override these in subclasses ---

    def f_callable(self, x: np.ndarray) -> float:
        """Scalar field to integrate.  Override for scalar benchmarks."""
        raise NotImplementedError

    def u_callable(self, x: np.ndarray) -> np.ndarray:
        """Vector field to integrate.  Override for vector benchmarks."""
        raise NotImplementedError

    # --- Pipeline ---

    def build_mesh(self) -> None:
        """Create the simplicial complex and compute duals."""
        HC = Complex(self.dim, domain=self.domain)
        HC.triangulate()
        for _ in range(self.n_refine):
            HC.refine_all()

        # Jitter interior vertices for asymmetric mesh tests
        if self.seed is not None:
            self._jitter_mesh(HC)

        # Tag boundaries
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

        self.HC = HC
        self.bV = bV
        self.interior_vertices = [v for v in HC.V if v not in bV]

    def _jitter_mesh(self, HC) -> None:
        """Perturb interior vertices and re-Delaunay triangulate.

        Uses ``HC.V.move()`` to correctly update the vertex cache,
        then re-triangulates via ``scipy.spatial.Delaunay`` to
        maintain a valid Delaunay mesh (required for circumcentric duals).
        """
        rng = np.random.default_rng(self.seed)

        # 1. Move interior vertices
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

        # 2. Re-Delaunay triangulate (disconnect all, then reconnect)
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

    def assign_field(self) -> None:
        """Set field values on all vertices from the analytical function."""
        if self.field_type == "scalar":
            for v in self.HC.V:
                v.f = self.f_callable(v.x_a[:self.dim])
        else:
            for v in self.HC.V:
                v.u = self.u_callable(v.x_a[:self.dim])

    def compute_numerical(self) -> None:
        """Compute DDG integrated gradient for each interior vertex."""
        for v in self.interior_vertices:
            if self.field_type == "scalar":
                self.numerical[id(v)] = scalar_gradient_integrated(
                    v, self.HC, dim=self.dim, field_attr='f'
                )
            else:
                self.numerical[id(v)] = velocity_difference_tensor(
                    v, self.HC, dim=self.dim
                )

    def compute_analytical(self) -> None:
        """Compute analytically integrated gradient for each interior vertex."""
        for v in self.interior_vertices:
            if self.dim == 1:
                a, b = dual_cell_vertices_1d(v)
                if self.field_type == "scalar":
                    self.analytical[id(v)] = integrated_gradient_1d(
                        self.f_callable, a, b
                    )
                else:
                    # For 1D vector, use scalar on each component
                    u_val = self.u_callable
                    result = np.zeros((1, 1))
                    result[0, 0] = integrated_gradient_1d(
                        lambda x: u_val(x)[0], a, b
                    )[0]
                    self.analytical[id(v)] = result

            elif self.dim == 2:
                polygon = dual_cell_polygon_2d(
                    v, include_edge_midpoints=self.include_edge_midpoints
                )
                if self.field_type == "scalar":
                    self.analytical[id(v)] = integrated_gradient_2d(
                        self.f_callable, polygon, n_gauss=self.n_gauss
                    )
                else:
                    self.analytical[id(v)] = integrated_gradient_2d_vector(
                        self.u_callable, polygon, n_gauss=self.n_gauss
                    )

            elif self.dim == 3:
                from hyperct.ddg import dual_cell_faces_3d
                from ddgclib.analytical import integrated_gradient_3d
                from ddgclib.analytical._divergence_theorem import (
                    integrated_gradient_3d_vector,
                )
                faces = dual_cell_faces_3d(v, self.HC)
                if self.field_type == "scalar":
                    self.analytical[id(v)] = integrated_gradient_3d(
                        self.f_callable, faces, n_gauss=self.n_gauss
                    )
                else:
                    self.analytical[id(v)] = integrated_gradient_3d_vector(
                        self.u_callable, faces, n_gauss=self.n_gauss
                    )

    def compute_errors(self) -> None:
        """Compute absolute and relative errors per interior vertex."""
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
        self.assign_field()
        self.compute_numerical()
        self.compute_analytical()
        self.compute_errors()
        return self.summary()

    def summary(self) -> dict:
        """Return error statistics."""
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
            "max_abs_error": float(np.max(abs_errs)),
            "mean_abs_error": float(np.mean(abs_errs)),
            "max_rel_error": float(np.max(rel_errs)),
            "mean_rel_error": float(np.mean(rel_errs)),
        }


def run_integrated_benchmarks(
    benchmark_classes: list[type],
    dims: list[int] | None = None,
    dual_methods: list[str] | None = None,
    seeds: list[int | None] | None = None,
    n_refine: int = 1,
) -> list[dict]:
    """Run a suite of integrated gradient benchmarks.

    Parameters
    ----------
    benchmark_classes : list of IntegratedGradientBenchmark subclasses
    dims : list of int
        Dimensions to test (default: [1, 2]).
    dual_methods : list of str
        Dual methods to test (default: ["barycentric", "circumcentric"]).
    seeds : list of (int or None)
        Seeds to test (default: [None, 42]).
    n_refine : int
        Number of refinement passes.

    Returns
    -------
    list of dict
        Summary results for each benchmark run.
    """
    dims = dims or [1, 2]
    dual_methods = dual_methods or ["barycentric", "circumcentric"]
    seeds = seeds or [None, 42]

    results = []
    for BenchClass in benchmark_classes:
        for dim in dims:
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
                        tag = f"seed={seed}" if seed is not None else "symmetric"
                        logger.info(
                            f"{summary['name']} ({dim}D, {method}, {tag}): "
                            f"max_abs={summary['max_abs_error']:.2e}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"{BenchClass.__name__} ({dim}D, {method}, "
                            f"seed={seed}) failed: {e}"
                        )
    return results
