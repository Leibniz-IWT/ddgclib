"""
Numerical integration of gradients over dual cells via the divergence theorem.

Uses the identity::

    ∫_V ∇f dV = ∮_{∂V} f · n dA

to reduce volume integrals to boundary integrals, which are then
evaluated using Gauss-Legendre quadrature.

For polynomial test functions of degree d, a quadrature rule with
(d+1)//2 + 1 points per edge is exact.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    nodes : np.ndarray, shape (n,)
    weights : np.ndarray, shape (n,)
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [0, 1]
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return nodes, weights


def integrated_gradient_1d(
    f: Callable[[np.ndarray], float],
    a: float,
    b: float,
) -> np.ndarray:
    """Integrated gradient of a scalar field over a 1D interval.

    Uses the fundamental theorem of calculus::

        ∫_a^b df/dx dx = f(b) - f(a)

    Parameters
    ----------
    f : callable
        Scalar field ``f(x) -> float`` where ``x`` is a 1D array.
    a, b : float
        Interval endpoints with ``a < b``.

    Returns
    -------
    np.ndarray, shape (1,)
        Integrated gradient vector.
    """
    return np.array([f(np.array([b])) - f(np.array([a]))])


def integrated_gradient_2d(
    f: Callable[[np.ndarray], float],
    polygon: np.ndarray,
    n_gauss: int = 10,
) -> np.ndarray:
    """Integrated gradient of a scalar field over a 2D polygon.

    Applies the divergence theorem::

        ∫_V ∇f dV = ∮_{∂V} f · n dA

    The boundary integral is a sum of line integrals over polygon edges.
    Each edge is parameterized as ``x(t) = P_k + t * (P_{k+1} - P_k)``
    for ``t ∈ [0, 1]``, and the integral is evaluated with Gauss-Legendre
    quadrature.

    Parameters
    ----------
    f : callable
        Scalar field ``f(x) -> float`` where ``x`` is a 2D array.
    polygon : np.ndarray, shape (N, 2)
        Ordered polygon vertices (counterclockwise).
    n_gauss : int
        Number of Gauss-Legendre quadrature points per edge.

    Returns
    -------
    np.ndarray, shape (2,)
        Integrated gradient vector.
    """
    nodes, weights = _gauss_legendre_01(n_gauss)
    result = np.zeros(2)
    N = len(polygon)

    for k in range(N):
        P_k = polygon[k]
        P_next = polygon[(k + 1) % N]
        edge = P_next - P_k

        # Outward normal (unnormalized, magnitude = edge length)
        # For CCW polygon, rotate edge 90° clockwise
        n_out = np.array([edge[1], -edge[0]])

        # Gauss quadrature along the edge
        for t, w in zip(nodes, weights):
            x_t = P_k + t * edge
            f_val = f(x_t)
            result += w * f_val * n_out

    return result


def integrated_gradient_2d_vector(
    u: Callable[[np.ndarray], np.ndarray],
    polygon: np.ndarray,
    n_gauss: int = 10,
) -> np.ndarray:
    """Integrated gradient tensor of a vector field over a 2D polygon.

    Computes::

        ∫_V ∇u dV = ∮_{∂V} u ⊗ n dA

    where ``u ⊗ n`` is the outer product of the vector field with the
    outward normal.

    Parameters
    ----------
    u : callable
        Vector field ``u(x) -> ndarray of shape (dim,)`` where ``x`` is
        a 2D array.
    polygon : np.ndarray, shape (N, 2)
        Ordered polygon vertices (counterclockwise).
    n_gauss : int
        Number of Gauss-Legendre quadrature points per edge.

    Returns
    -------
    np.ndarray, shape (dim, 2)
        Integrated gradient tensor.
    """
    nodes, weights = _gauss_legendre_01(n_gauss)
    # Probe to get vector dimension
    u_sample = u(polygon[0])
    dim_u = len(u_sample)

    result = np.zeros((dim_u, 2))
    N = len(polygon)

    for k in range(N):
        P_k = polygon[k]
        P_next = polygon[(k + 1) % N]
        edge = P_next - P_k
        n_out = np.array([edge[1], -edge[0]])

        for t, w in zip(nodes, weights):
            x_t = P_k + t * edge
            u_val = u(x_t)
            result += w * np.outer(u_val, n_out)

    return result


def integrated_gradient_3d(
    f: Callable[[np.ndarray], float],
    faces: list[np.ndarray],
    n_gauss: int = 7,
) -> np.ndarray:
    """Integrated gradient of a scalar field over a 3D polyhedron.

    Applies the divergence theorem::

        ∫_V ∇f dV = ∮_{∂V} f · n dA

    Each face is a planar polygon, triangulated from its centroid.
    The surface integral over each triangle uses a symmetric quadrature
    rule on the reference triangle.

    Parameters
    ----------
    f : callable
        Scalar field ``f(x) -> float`` where ``x`` is a 3D array.
    faces : list[np.ndarray]
        Each element is an ``(M, 3)`` array of ordered face polygon
        vertices with outward-facing orientation.
    n_gauss : int
        Number of quadrature points for triangle integration.

    Returns
    -------
    np.ndarray, shape (3,)
        Integrated gradient vector.
    """
    result = np.zeros(3)

    for face_verts in faces:
        M = len(face_verts)
        if M < 3:
            continue

        centroid = face_verts.mean(axis=0)

        for k in range(M):
            v0 = face_verts[k]
            v1 = face_verts[(k + 1) % M]

            # Triangle: centroid, v0, v1
            # Outward normal (unnormalized, magnitude = 2 * triangle area)
            n_tri = np.cross(v0 - centroid, v1 - centroid)

            # Integrate f over this triangle using midpoint rule on
            # sub-triangles (sufficient for low-degree polynomials)
            # Use barycentric quadrature points
            result += _triangle_integral_scalar(
                f, centroid, v0, v1, n_tri, n_gauss
            )

    return result


def integrated_gradient_3d_vector(
    u: Callable[[np.ndarray], np.ndarray],
    faces: list[np.ndarray],
    n_gauss: int = 7,
) -> np.ndarray:
    """Integrated gradient tensor of a vector field over a 3D polyhedron.

    Computes ``∫_V ∇u dV = ∮_{∂V} u ⊗ n dA``.

    Parameters
    ----------
    u : callable
        Vector field ``u(x) -> ndarray of shape (3,)``.
    faces : list[np.ndarray]
        Face polygons with outward-facing orientation.
    n_gauss : int
        Number of quadrature points for triangle integration.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Integrated gradient tensor.
    """
    result = np.zeros((3, 3))

    for face_verts in faces:
        M = len(face_verts)
        if M < 3:
            continue

        centroid = face_verts.mean(axis=0)

        for k in range(M):
            v0 = face_verts[k]
            v1 = face_verts[(k + 1) % M]
            n_tri = np.cross(v0 - centroid, v1 - centroid)

            result += _triangle_integral_vector(
                u, centroid, v0, v1, n_tri, n_gauss
            )

    return result


def _triangle_quadrature_points(n: int) -> list[tuple[float, float, float, float]]:
    """Symmetric quadrature points and weights on the reference triangle.

    Returns barycentric coordinates (l1, l2, l3) and weight w for
    integration over a triangle with vertices A, B, C::

        x = l1*A + l2*B + l3*C

    The integral is approximated as::

        ∫_T f dA ≈ 2*Area * Σ w_i * f(x_i)

    Parameters
    ----------
    n : int
        Approximate number of points (actual count may differ).

    Returns
    -------
    list of (l1, l2, l3, w)
    """
    if n <= 1:
        # 1-point centroid rule (exact for degree 1)
        return [(1/3, 1/3, 1/3, 1.0)]
    elif n <= 3:
        # 3-point midpoint rule (exact for degree 2)
        return [
            (0.5, 0.5, 0.0, 1/3),
            (0.0, 0.5, 0.5, 1/3),
            (0.5, 0.0, 0.5, 1/3),
        ]
    elif n <= 4:
        # 4-point rule (exact for degree 3)
        return [
            (1/3, 1/3, 1/3, -27/48),
            (0.6, 0.2, 0.2, 25/48),
            (0.2, 0.6, 0.2, 25/48),
            (0.2, 0.2, 0.6, 25/48),
        ]
    else:
        # 7-point rule (exact for degree 5)
        a1 = 0.059715871789770
        b1 = 0.470142064105115
        a2 = 0.797426985353087
        b2 = 0.101286507323456
        return [
            (1/3, 1/3, 1/3, 0.225),
            (a1, b1, b1, 0.132394152788506),
            (b1, a1, b1, 0.132394152788506),
            (b1, b1, a1, 0.132394152788506),
            (a2, b2, b2, 0.125939180544827),
            (b2, a2, b2, 0.125939180544827),
            (b2, b2, a2, 0.125939180544827),
        ]


def _triangle_integral_scalar(
    f: Callable,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    n_vec: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Integrate f * n_vec over triangle ABC.

    Parameters
    ----------
    f : callable
        Scalar field.
    A, B, C : np.ndarray
        Triangle vertices (3D).
    n_vec : np.ndarray
        Unnormalized normal vector (magnitude = 2 * triangle area).
    n_gauss : int
        Number of quadrature points.

    Returns
    -------
    np.ndarray, shape (3,)
        ∫_T f · n dA ≈ Σ w_i * f(x_i) * n_vec * 0.5
    """
    quad = _triangle_quadrature_points(n_gauss)
    result = np.zeros(3)
    for l1, l2, l3, w in quad:
        x = l1 * A + l2 * B + l3 * C
        result += w * f(x) * n_vec * 0.5
    return result


def _triangle_integral_vector(
    u: Callable,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    n_vec: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Integrate u ⊗ n over triangle ABC.

    Returns
    -------
    np.ndarray, shape (3, 3)
        ∫_T u ⊗ n dA
    """
    quad = _triangle_quadrature_points(n_gauss)
    result = np.zeros((3, 3))
    for l1, l2, l3, w in quad:
        x = l1 * A + l2 * B + l3 * C
        u_val = u(x)
        result += w * np.outer(u_val, n_vec) * 0.5
    return result
