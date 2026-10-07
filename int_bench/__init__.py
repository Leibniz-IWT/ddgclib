"""
int_bench — n-dimensional integrated gradient & Hessian benchmarks.

Public API
----------

Core data structure:

* :class:`SimplexMesh` — minimal n-D simplicial complex with
  per-vertex 1-rings, primal stars, and a regular-grid factory.

Operators (act on vertex-sampled scalar/vector fields):

* :func:`integrated_gradient` — ``∫_V ∇f dV`` on the chosen control
  volume around an interior vertex (primal star or barycentric dual).
* :func:`integrated_gradient_tensor` — same for vector fields.
* :func:`integrated_hessian` — ``∫_V ∇⊗∇f dV`` via two gradient passes.
* :func:`vertex_gradient_field` — cell-averaged ``g_i`` everywhere.

Analytical reference (act on a callable ``f``):

* :func:`integrated_gradient_analytical`
* :func:`integrated_gradient_tensor_analytical`
* :func:`integrated_hessian_analytical`

Finite-difference reference:

* :func:`fd_gradient`, :func:`fd_hessian`
* :func:`integrated_gradient_fd`, :func:`integrated_hessian_fd`

Benchmark suite:

* :class:`TestField`, :func:`standard_fields`
* :func:`run_benchmark`, :func:`run_suite`, :func:`print_results`

Quick example
-------------

>>> from int_bench import SimplexMesh, integrated_gradient
>>> mesh = SimplexMesh.regular_grid(dim=2, n=9)
>>> f = lambda x: x[0]**2 + x[1]**2
>>> f_vals = mesh.sample_scalar(f)
>>> v = mesh.interior_vertices[0]
>>> G = integrated_gradient(mesh, f_vals, v, control="dual")
"""
from .mesh import SimplexMesh

from .geometry import (
    area_vector,
    barycentric_dual_face_vector,
    polytope_area_vector,
    simplex_centroid,
    simplex_volume,
)

from .operators import (
    ControlVolume,
    control_volume,
    dual_cell_boundary,
    integrated_gradient,
    integrated_gradient_tensor,
    integrated_hessian,
    primal_star_boundary,
    vertex_gradient_field,
)

from .analytical import (
    integrated_gradient_analytical,
    integrated_gradient_tensor_analytical,
    integrated_hessian_analytical,
)

from .finite_difference import (
    fd_gradient,
    fd_hessian,
    integrated_gradient_fd,
    integrated_hessian_fd,
)

from .benchmarks import (
    BenchmarkResult,
    TestField,
    print_results,
    run_benchmark,
    run_suite,
    standard_fields,
)

__all__ = [
    # mesh
    "SimplexMesh",
    # geometry
    "area_vector",
    "barycentric_dual_face_vector",
    "polytope_area_vector",
    "simplex_centroid",
    "simplex_volume",
    # operators
    "ControlVolume",
    "control_volume",
    "dual_cell_boundary",
    "integrated_gradient",
    "integrated_gradient_tensor",
    "integrated_hessian",
    "primal_star_boundary",
    "vertex_gradient_field",
    # analytical
    "integrated_gradient_analytical",
    "integrated_gradient_tensor_analytical",
    "integrated_hessian_analytical",
    # finite difference
    "fd_gradient",
    "fd_hessian",
    "integrated_gradient_fd",
    "integrated_hessian_fd",
    # benchmarks
    "BenchmarkResult",
    "TestField",
    "print_results",
    "run_benchmark",
    "run_suite",
    "standard_fields",
]

__version__ = "0.1.0"
