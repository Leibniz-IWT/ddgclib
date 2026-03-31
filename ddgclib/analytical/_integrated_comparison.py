"""
Integrated comparison utilities for dynamic case validation.

Replaces point-wise ``abs(v.p - P_analytical(v.x_a))`` with
integrated comparisons over dual cells using the divergence theorem
framework.

For a scalar field (e.g. pressure)::

    point-wise:   err_i = |f_i - f_analytical(x_i)|
    integrated:   err_i = |f_i * Vol_i - ∫_{V_i} f_analytical dV|

The integrated form is mathematically consistent with the FVM
discretization: it compares volume-averaged quantities over the
same dual cells.

Usage::

    from ddgclib.analytical._integrated_comparison import (
        integrated_pressure_error,
        integrated_l2_norm,
        compare_stress_force,
    )

    # Pressure error at each interior vertex
    errs = integrated_pressure_error(HC, interior, P_analytical, dim=2)
    print(f"Max integrated pressure error: {max(errs):.4e}")

    # Weighted L2 norm
    l2 = integrated_l2_norm(HC, interior, P_analytical, dim=2)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _dual_cell_pressure_integral_1d(
    P_analytical: Callable,
    v,
    n_gauss: int = 10,
) -> float:
    """Integrate P_analytical over 1D dual cell of v.

    Returns ∫_{V_i} P(x) dx = ∫_a^b P(x) dx via Gauss quadrature.
    Falls back to point-wise * dual_vol for boundary/degenerate vertices.
    """
    from ddgclib.analytical._divergence_theorem import _gauss_legendre_01

    try:
        from hyperct.ddg import dual_cell_vertices_1d
        a, b = dual_cell_vertices_1d(v)
    except (ValueError, AttributeError):
        # Boundary or degenerate vertex — use point-wise approximation
        vol = getattr(v, 'dual_vol', 0.0)
        return P_analytical(v.x_a[:1]) * vol

    nodes, weights = _gauss_legendre_01(n_gauss)
    result = 0.0
    length = b - a
    for t, w in zip(nodes, weights):
        x = a + t * length
        result += w * P_analytical(np.array([x])) * length
    return result


def _dual_cell_pressure_integral_2d(
    P_analytical: Callable,
    v,
    polygon_method: str = "barycentric_dual_p_ij",
    n_gauss: int = 10,
) -> float:
    """Integrate P_analytical over 2D dual cell polygon of v.

    Uses the identity: ∫_V P dV = (for a polygon, triangulate from
    centroid and sum triangle integrals).
    """
    from hyperct.ddg import dual_cell_polygon_2d
    from ddgclib.analytical._divergence_theorem import _gauss_legendre_01

    include_midpts = (polygon_method == "barycentric_dual_p_ij")
    polygon = dual_cell_polygon_2d(v, include_edge_midpoints=include_midpts)
    N = len(polygon)
    centroid = polygon.mean(axis=0)

    nodes, weights = _gauss_legendre_01(n_gauss)
    result = 0.0

    for k in range(N):
        # Triangle: centroid, polygon[k], polygon[(k+1)%N]
        A = centroid
        B = polygon[k]
        C = polygon[(k + 1) % N]
        # Area of triangle via cross product
        tri_area = 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
        )
        if tri_area < 1e-30:
            continue
        # Gauss quadrature on triangle using barycentric coords
        # Use midpoint rule (sufficient for low-degree P)
        for ti, wi in zip(nodes, weights):
            for tj, wj in zip(nodes, weights):
                if ti + tj > 1.0:
                    continue
                l1 = ti
                l2 = tj * (1.0 - ti)
                l3 = 1.0 - l1 - l2
                x = l1 * A + l2 * B + l3 * C
                result += wi * wj * (1.0 - ti) * P_analytical(x) * 2.0 * tri_area

    return result


def _dual_cell_pressure_integral_2d_simple(
    P_analytical: Callable,
    v,
    polygon_method: str = "barycentric_dual_p_ij",
    n_gauss: int = 7,
) -> float:
    """Integrate P_analytical over 2D dual cell using triangle quadrature.

    Triangulates the polygon from its centroid and uses symmetric
    quadrature on each sub-triangle.
    """
    from hyperct.ddg import dual_cell_polygon_2d
    from ddgclib.analytical._divergence_theorem import (
        _triangle_quadrature_points,
    )

    include_midpts = (polygon_method == "barycentric_dual_p_ij")
    try:
        polygon = dual_cell_polygon_2d(v, include_edge_midpoints=include_midpts)
    except (ValueError, AttributeError, IndexError):
        # Degenerate dual cell — fall back to point-wise
        vol = getattr(v, 'dual_vol', 0.0)
        return P_analytical(v.x_a[:2]) * vol

    N = len(polygon)
    if N < 3:
        vol = getattr(v, 'dual_vol', 0.0)
        return P_analytical(v.x_a[:2]) * vol

    centroid = polygon.mean(axis=0)

    quad = _triangle_quadrature_points(n_gauss)
    result = 0.0

    for k in range(N):
        A = centroid
        B = polygon[k]
        C = polygon[(k + 1) % N]
        tri_area = 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
        )
        if tri_area < 1e-30:
            continue
        for l1, l2, l3, w in quad:
            x = l1 * A + l2 * B + l3 * C
            result += w * P_analytical(x) * 2.0 * tri_area

    return result


def volume_averaged_scalar(
    f_analytical: Callable,
    v,
    dim: int = 2,
    polygon_method: str = "barycentric_dual_p_ij",
    n_gauss: int = 7,
) -> float:
    """Volume-averaged value of a scalar field over the dual cell.

    Computes ``(1/Vol_i) * ∫_{V_i} f(x) dV``.

    This is the correct FVM cell-averaged value to assign to ``v.p``
    (or any other FVM scalar field) so that ``v.p * Vol_i = ∫ f dV``.

    Parameters
    ----------
    f_analytical : callable
        Scalar field ``f(x_a) -> float``.
    v : vertex object
        Must have ``v.vd`` populated.
    dim : int
        Spatial dimension.
    polygon_method : str
        Dual cell polygon formulation.
    n_gauss : int
        Quadrature points.

    Returns
    -------
    float
        Volume-averaged scalar value.
    """
    from ddgclib.operators.stress import _get_dual_vol

    vol = _get_dual_vol(v, None, dim)
    if vol < 1e-30:
        return f_analytical(v.x_a[:dim])

    if dim == 1:
        integral = _dual_cell_pressure_integral_1d(
            f_analytical, v, n_gauss=n_gauss
        )
    elif dim == 2:
        integral = _dual_cell_pressure_integral_2d_simple(
            f_analytical, v,
            polygon_method=polygon_method, n_gauss=n_gauss,
        )
    elif dim == 3:
        # 3D: use point-wise approximation (exact integration deferred)
        integral = f_analytical(v.x_a[:3]) * vol
    else:
        raise NotImplementedError(f"dim={dim}")

    return integral / vol


def integrated_pressure_error(
    HC,
    interior_vertices: list,
    P_analytical: Callable,
    dim: int = 2,
    polygon_method: str = "barycentric_dual_p_ij",
    n_gauss: int = 7,
) -> list[float]:
    """Integrated pressure error per interior vertex.

    Computes ``|p_i * Vol_i - ∫_{V_i} P_analytical dV|`` for each vertex.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with duals computed.
    interior_vertices : list
        Interior vertex objects.
    P_analytical : callable
        Analytical pressure field ``P(x_a) -> float``.
    dim : int
        Spatial dimension.
    polygon_method : str
        Dual cell polygon formulation.
    n_gauss : int
        Quadrature points.

    Returns
    -------
    list of float
        Absolute integrated error per vertex.
    """
    from ddgclib.operators.stress import _get_dual_vol

    errors = []
    for v in interior_vertices:
        vol = _get_dual_vol(v, HC, dim)
        p_numerical_integrated = float(v.p) * vol

        if dim == 1:
            p_analytical_integrated = _dual_cell_pressure_integral_1d(
                P_analytical, v, n_gauss=n_gauss
            )
        elif dim == 2:
            p_analytical_integrated = _dual_cell_pressure_integral_2d_simple(
                P_analytical, v,
                polygon_method=polygon_method, n_gauss=n_gauss,
            )
        elif dim == 3:
            # For 3D, use the point-wise value * volume as approximation
            # (exact 3D dual cell integration via faces is deferred)
            p_analytical_integrated = P_analytical(v.x_a[:3]) * vol
        else:
            raise NotImplementedError(f"dim={dim}")

        errors.append(abs(p_numerical_integrated - p_analytical_integrated))

    return errors


def integrated_l2_norm(
    HC,
    interior_vertices: list,
    P_analytical: Callable,
    dim: int = 2,
    polygon_method: str = "barycentric_dual_p_ij",
    n_gauss: int = 7,
) -> float:
    """Volume-weighted L2 norm of pressure error.

    Computes::

        sqrt(sum_i (p_i - <P>_i)^2 * Vol_i / sum_i Vol_i)

    where ``<P>_i = ∫_{V_i} P dV / Vol_i`` is the volume-averaged
    analytical pressure over the dual cell.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with duals computed.
    interior_vertices : list
        Interior vertex objects.
    P_analytical : callable
        Analytical pressure ``P(x_a) -> float``.
    dim : int
        Spatial dimension.

    Returns
    -------
    float
        L2 norm of pressure error.
    """
    from ddgclib.operators.stress import _get_dual_vol

    sum_sq_vol = 0.0
    sum_vol = 0.0

    for v in interior_vertices:
        vol = _get_dual_vol(v, HC, dim)
        if vol < 1e-30:
            continue

        if dim == 1:
            p_int = _dual_cell_pressure_integral_1d(
                P_analytical, v, n_gauss=n_gauss
            )
        elif dim == 2:
            p_int = _dual_cell_pressure_integral_2d_simple(
                P_analytical, v,
                polygon_method=polygon_method, n_gauss=n_gauss,
            )
        elif dim == 3:
            # Point-wise approximation for 3D (exact 3D integration deferred)
            p_int = P_analytical(v.x_a[:3]) * vol
        else:
            raise NotImplementedError(f"dim={dim}")

        p_avg_analytical = p_int / vol
        err = float(v.p) - p_avg_analytical
        sum_sq_vol += err ** 2 * vol
        sum_vol += vol

    if sum_vol < 1e-30:
        return 0.0
    return float(np.sqrt(sum_sq_vol / sum_vol))


def compare_stress_force(
    HC,
    interior_vertices: list,
    dim: int = 2,
    mu: float = 1e-3,
) -> dict:
    """Compute stress force diagnostics for interior vertices.

    Returns summary dict with force norms, useful for checking
    equilibrium (all forces should be near zero).
    """
    from ddgclib.operators.stress import stress_force as _sf

    F_norms = []
    for v in interior_vertices:
        F = _sf(v, dim=dim, mu=mu, HC=HC)
        F_norms.append(float(np.linalg.norm(F)))

    F_norms = np.array(F_norms)
    return {
        "max_F": float(np.max(F_norms)) if len(F_norms) > 0 else 0.0,
        "mean_F": float(np.mean(F_norms)) if len(F_norms) > 0 else 0.0,
        "median_F": float(np.median(F_norms)) if len(F_norms) > 0 else 0.0,
        "F_norms": F_norms,
    }
