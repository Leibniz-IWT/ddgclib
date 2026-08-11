"""
Validation suite: linear and quadratic test fields in 1D, 2D, 3D.

Each benchmark constructs a regular grid, picks one interior vertex,
and compares three quantities at that vertex:

* ``num``  — discrete integrated operator (DDG, dual or primal CV)
* ``ref``  — analytical integral via Gauss quadrature on the same CV
* ``fd``   — pointwise central difference times the CV measure

Linear fields: ``num`` and ``ref`` agree to machine precision; ``fd``
agrees because the true gradient is constant.

Quadratic fields: ``ref`` is exact (Gauss order 10 is exact for
polynomials up to degree 19); ``num`` agrees to machine precision on
symmetric meshes for the dual CV (a known DDG identity), and is
mildly non-zero on the primal CV (still convergent under refinement).
``fd`` agrees because the true Hessian is constant and the cell is
centred on the vertex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .analytical import (
    integrated_gradient_analytical,
    integrated_hessian_analytical,
)
from .finite_difference import integrated_gradient_fd, integrated_hessian_fd
from .mesh import SimplexMesh
from .operators import (
    ControlVolume,
    control_volume,
    integrated_gradient,
    integrated_hessian,
)


# ---------------------------------------------------------------------------
# Test field definitions
# ---------------------------------------------------------------------------

@dataclass
class TestField:
    """A scalar field with known gradient and Hessian.

    Used as the ground truth for benchmark comparisons.
    """
    name: str
    dim: int
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]
    is_polynomial: bool = True
    polynomial_order: int = 1


def _linear_field(dim: int, a: np.ndarray, b: float = 0.0) -> TestField:
    a = np.asarray(a, dtype=float)
    return TestField(
        name=f"linear_{dim}d",
        dim=dim,
        f=lambda x: float(np.dot(a, x[:dim]) + b),
        grad=lambda x: a.copy(),
        hessian=lambda x: np.zeros((dim, dim)),
        polynomial_order=1,
    )


def _quadratic_field(
    dim: int,
    A: np.ndarray,
    b: np.ndarray,
    c: float = 0.0,
) -> TestField:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    H = A + A.T

    def f(x):
        x = x[:dim]
        return float(x @ A @ x + np.dot(b, x) + c)

    def grad(x):
        x = x[:dim]
        return H @ x + b

    def hess(x):
        return H

    return TestField(
        name=f"quadratic_{dim}d",
        dim=dim,
        f=f,
        grad=grad,
        hessian=hess,
        polynomial_order=2,
    )


def standard_fields(dim: int) -> list[TestField]:
    """Default benchmark fields for a given dimension."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, size=dim)
    A = rng.uniform(-1.0, 1.0, size=(dim, dim))
    b = rng.uniform(-1.0, 1.0, size=dim)
    return [
        _linear_field(dim, a),
        _quadratic_field(dim, A, b),
    ]


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    field_name: str
    dim: int
    control: str
    n: int
    jitter_seed: int | None
    v_idx: int
    cell_volume: float
    grad_num: np.ndarray
    grad_ref: np.ndarray
    grad_fd: np.ndarray
    hess_num: np.ndarray
    hess_ref: np.ndarray
    hess_fd: np.ndarray

    def errors(self) -> dict[str, float]:
        return {
            "grad_num_vs_ref": float(np.linalg.norm(self.grad_num - self.grad_ref)),
            "grad_fd_vs_ref":  float(np.linalg.norm(self.grad_fd  - self.grad_ref)),
            "hess_num_vs_ref": float(np.linalg.norm(self.hess_num - self.hess_ref)),
            "hess_fd_vs_ref":  float(np.linalg.norm(self.hess_fd  - self.hess_ref)),
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _pick_central_vertex(mesh: SimplexMesh) -> int:
    """Interior vertex closest to the domain centroid."""
    interior = mesh.interior_vertices
    if len(interior) == 0:
        raise ValueError("mesh has no interior vertices")
    centroid = np.array([0.5 * (lo + hi) for lo, hi in mesh.domain])
    pts = mesh.vertices[interior]
    return int(interior[np.argmin(np.linalg.norm(pts - centroid, axis=1))])


def run_benchmark(
    field: TestField,
    n: int = 7,
    control: ControlVolume = "dual",
    jitter_seed: int | None = None,
    jitter_amplitude: float = 0.1,
    n_gauss: int = 10,
    fd_h: float = 1e-5,
    fd_h_hess: float = 1e-4,
    use_analytical_grad: bool = True,
) -> BenchmarkResult:
    """Run one benchmark on a regular grid mesh.

    Parameters
    ----------
    field : TestField
    n : int
        Number of vertices per axis.
    control : {"primal", "dual"}
    jitter_seed : int or None
        ``None`` => symmetric mesh; integer => deterministic jittering.
    n_gauss : int
        Quadrature order for the analytical reference.
    fd_h, fd_h_hess : float
        Finite-difference step sizes for gradient / Hessian.
    use_analytical_grad : bool
        Use ``field.grad`` for the Hessian boundary integrand instead
        of finite-differencing ``f``.
    """
    mesh = SimplexMesh.regular_grid(dim=field.dim, n=n)
    if jitter_seed is not None:
        mesh = mesh.jittered(seed=jitter_seed, amplitude=jitter_amplitude)
    v_idx = _pick_central_vertex(mesh)

    f_vals = mesh.sample_scalar(field.f)

    grad_num = integrated_gradient(mesh, f_vals, v_idx, control=control)
    grad_ref = integrated_gradient_analytical(
        mesh, field.f, v_idx, control=control, n_gauss=n_gauss,
    )
    grad_fd = integrated_gradient_fd(
        mesh, field.f, v_idx, control=control, h=fd_h,
    )

    hess_num = integrated_hessian(mesh, f_vals, v_idx, control=control)
    hess_ref = integrated_hessian_analytical(
        mesh,
        field.f,
        v_idx,
        grad_f=field.grad if use_analytical_grad else None,
        control=control,
        n_gauss=n_gauss,
    )
    hess_fd = integrated_hessian_fd(
        mesh, field.f, v_idx, control=control, h=fd_h_hess,
    )

    return BenchmarkResult(
        field_name=field.name,
        dim=field.dim,
        control=str(control),
        n=n,
        jitter_seed=jitter_seed,
        v_idx=v_idx,
        cell_volume=control_volume(mesh, v_idx, control=control),
        grad_num=grad_num,
        grad_ref=grad_ref,
        grad_fd=grad_fd,
        hess_num=hess_num,
        hess_ref=hess_ref,
        hess_fd=hess_fd,
    )


def run_suite(
    dims: tuple[int, ...] = (1, 2, 3),
    controls: tuple[ControlVolume, ...] = ("primal", "dual"),
    n: int = 7,
    jitter_seed: int | None = None,
    n_gauss: int = 10,
) -> list[BenchmarkResult]:
    """Run the linear+quadratic suite over the given dims and controls."""
    results = []
    for d in dims:
        for fld in standard_fields(d):
            for cv in controls:
                results.append(run_benchmark(
                    fld, n=n, control=cv,
                    jitter_seed=jitter_seed, n_gauss=n_gauss,
                ))
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt_err(err: float) -> str:
    if err < 1e-12:
        return f"\033[92m{err:.2e}\033[0m"
    if err < 1e-3:
        return f"\033[93m{err:.2e}\033[0m"
    return f"\033[91m{err:.2e}\033[0m"


def print_results(results: list[BenchmarkResult]) -> None:
    """Compact tabular summary of a list of benchmark results."""
    header = (
        f"{'field':<18} {'dim':>3} {'CV':<7} {'mesh':<10}"
        f" {'|G_num-G_ref|':>14} {'|G_fd-G_ref|':>14}"
        f" {'|H_num-H_ref|':>14} {'|H_fd-H_ref|':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        e = r.errors()
        mesh_tag = f"n={r.n}" + (f"/s{r.jitter_seed}" if r.jitter_seed is not None else "/sym")
        print(
            f"{r.field_name:<18} {r.dim:>3} {r.control:<7} {mesh_tag:<10}"
            f" {_fmt_err(e['grad_num_vs_ref']):>22}"
            f" {_fmt_err(e['grad_fd_vs_ref']):>22}"
            f" {_fmt_err(e['hess_num_vs_ref']):>22}"
            f" {_fmt_err(e['hess_fd_vs_ref']):>22}"
        )


__all__ = [
    "TestField",
    "BenchmarkResult",
    "standard_fields",
    "run_benchmark",
    "run_suite",
    "print_results",
]
