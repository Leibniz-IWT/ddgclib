"""
Analytical reference: integrate ∇f and ∇⊗∇f exactly over the same
control volume that the discrete operators use, via Gauss quadrature
on the boundary.

The boundary integrand is::

    ∫_V ∇f dV = ∮_∂V f n dA
    ∫_V ∇⊗∇f dV = ∮_∂V (∇f) ⊗ n dA

where ``n dA`` is the outward area vector. We sample ``f`` (and
``grad_f`` for the Hessian) at quadrature nodes along each boundary
piece and accumulate.

For polynomial test fields the quadrature order is chosen high enough
to give machine precision; for the small polygons / triangles used
here a moderate ``n_gauss`` is sufficient.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .geometry import area_vector, barycentric_dual_face_vector
from .mesh import SimplexMesh
from .operators import ControlVolume


# ---------------------------------------------------------------------------
# 1D Gauss-Legendre on [0, 1]
# ---------------------------------------------------------------------------

def _gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


# ---------------------------------------------------------------------------
# Triangle quadrature (symmetric, exact for degree <= 5 with 7-pt rule)
# ---------------------------------------------------------------------------

def _triangle_quadrature(n_gauss: int) -> list[tuple[float, float, float, float]]:
    """Barycentric points (l1, l2, l3) and weights w on the unit triangle.

    Integration uses ``∫_T f dA ≈ Σ w_i f(x_i) * Area(T)`` (no factor of 2).
    """
    if n_gauss <= 1:
        return [(1 / 3, 1 / 3, 1 / 3, 1.0)]
    if n_gauss <= 3:
        return [
            (0.5, 0.5, 0.0, 1 / 3),
            (0.0, 0.5, 0.5, 1 / 3),
            (0.5, 0.0, 0.5, 1 / 3),
        ]
    if n_gauss <= 4:
        return [
            (1 / 3, 1 / 3, 1 / 3, -27 / 48),
            (0.6, 0.2, 0.2, 25 / 48),
            (0.2, 0.6, 0.2, 25 / 48),
            (0.2, 0.2, 0.6, 25 / 48),
        ]
    a1, b1 = 0.059715871789770, 0.470142064105115
    a2, b2 = 0.797426985353087, 0.101286507323456
    return [
        (1 / 3, 1 / 3, 1 / 3, 0.225),
        (a1, b1, b1, 0.132394152788506),
        (b1, a1, b1, 0.132394152788506),
        (b1, b1, a1, 0.132394152788506),
        (a2, b2, b2, 0.125939180544827),
        (b2, a2, b2, 0.125939180544827),
        (b2, b2, a2, 0.125939180544827),
    ]


# ---------------------------------------------------------------------------
# Boundary descriptions
# ---------------------------------------------------------------------------

def _primal_star_boundary_pieces(
    mesh: SimplexMesh,
    v_idx: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Same as operators.primal_star_boundary, but local to this file
    to avoid circular imports."""
    out = []
    for s_idx in mesh.primal_star(v_idx):
        simplex = mesh.simplices[s_idx]
        face_idx = np.array([v for v in simplex if v != v_idx])
        face_pts = mesh.vertices[face_idx]
        A = area_vector(face_pts)
        v_pos = mesh.vertices[v_idx]
        if np.dot(A, face_pts.mean(axis=0) - v_pos) < 0:
            A = -A
        out.append((face_pts, A))
    return out


def _dual_cell_boundary_pieces(
    mesh: SimplexMesh,
    v_idx: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """For each (simplex, neighbor j) pair, return the dual face piece.

    Returns
    -------
    list of (piece_vertices, piece_area_vector, simplex_index, j_global)
        ``piece_vertices`` are the 2^(n-1) corners of the (n-1)-cube
        that forms one piece of the dual face boundary inside one
        n-simplex. The corners are ordered by binary mask of included
        "other" vertices (suitable for Kuhn triangulation).
        ``piece_area_vector`` is the corresponding (n-1)-area vector
        pointing from ``v_idx`` to ``j`` (sums to A_ij over simplices).
    """
    n = mesh.dim
    pieces = []
    for s_idx in mesh.primal_star(v_idx):
        simplex = mesh.simplices[s_idx]
        i_local = int(np.where(simplex == v_idx)[0][0])
        for j_local, j_global in enumerate(simplex):
            if j_local == i_local:
                continue
            simplex_pts = mesh.vertices[simplex]
            A = barycentric_dual_face_vector(simplex_pts, i_local, j_local)
            other = [k for k in range(n + 1) if k not in (i_local, j_local)]
            n_other = len(other)
            corners = np.empty((1 << n_other, n))
            for mask in range(1 << n_other):
                face_idx = [i_local, j_local]
                for b in range(n_other):
                    if mask & (1 << b):
                        face_idx.append(other[b])
                corners[mask] = simplex_pts[face_idx].mean(axis=0)
            pieces.append((corners, A, s_idx, int(j_global)))
    return pieces


# ---------------------------------------------------------------------------
# Boundary surface integral of a callable
# ---------------------------------------------------------------------------

def _integrate_face_1d(
    f: Callable[[np.ndarray], float],
    point: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """1D 'face' is a single point; the integral is just f(point) * A."""
    return f(point) * A


def _integrate_face_2d(
    f: Callable[[np.ndarray], float],
    poly_corners: np.ndarray,
    A_total: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Integrate f * n dA over a 1D boundary piece in 2D.

    For the dual cell each piece is a single segment between
    ``poly_corners[0]`` (= edge midpoint) and ``poly_corners[1]``
    (= simplex barycenter). For the primal star each piece is just
    an edge (poly_corners are the two endpoints).
    """
    P0 = poly_corners[0]
    P1 = poly_corners[-1]
    nodes, weights = _gauss_legendre_01(n_gauss)
    out = np.zeros(2)
    for t, w in zip(nodes, weights):
        x = P0 + t * (P1 - P0)
        out += w * f(x) * A_total
    return out


def _integrate_face_3d_primal(
    f: Callable[[np.ndarray], float],
    triangle: np.ndarray,
    A: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Triangle face on primal star boundary in 3D."""
    quad = _triangle_quadrature(n_gauss)
    out = np.zeros(3)
    for l1, l2, l3, w in quad:
        x = l1 * triangle[0] + l2 * triangle[1] + l3 * triangle[2]
        out += w * f(x) * A
    return out


def _integrate_face_3d_dual(
    f: Callable[[np.ndarray], float],
    quad_corners: np.ndarray,
    A_total: np.ndarray,
    n_gauss: int,
) -> np.ndarray:
    """Quadrilateral piece (per simplex) of the dual cell boundary in 3D.

    ``quad_corners`` has shape (4, 3) ordered by the binary mask used
    in the Kuhn construction:

        00 = m_ij                       (edge midpoint)
        10 = b_{ijk}                    (barycenter of face containing k)
        01 = b_{ijl}                    (barycenter of face containing l)
        11 = b_T                        (simplex barycenter)

    To respect this layout we triangulate as ((00, 10, 11), (00, 11, 01))
    and integrate each triangle, weighting by its share of the total
    area vector ``A_total``.
    """
    tris = [
        np.stack([quad_corners[0], quad_corners[1], quad_corners[3]]),
        np.stack([quad_corners[0], quad_corners[3], quad_corners[2]]),
    ]
    A_tris = [area_vector(t) for t in tris]
    sum_A = A_tris[0] + A_tris[1]
    if np.dot(sum_A, A_total) < 0:
        # Flip triangle orientation to match the canonical A_total.
        A_tris = [-a for a in A_tris]
        sum_A = -sum_A

    quad = _triangle_quadrature(n_gauss)
    out = np.zeros(3)
    for tri, A_tri in zip(tris, A_tris):
        for l1, l2, l3, w in quad:
            x = l1 * tri[0] + l2 * tri[1] + l3 * tri[2]
            out += w * f(x) * A_tri
    return out


# ---------------------------------------------------------------------------
# Public: analytical integrated gradient (scalar field)
# ---------------------------------------------------------------------------

def integrated_gradient_analytical(
    mesh: SimplexMesh,
    f: Callable[[np.ndarray], float],
    v_idx: int,
    control: ControlVolume = "dual",
    n_gauss: int = 10,
) -> np.ndarray:
    """``∫_V ∇f dV = ∮_∂V f n dA`` evaluated by Gauss quadrature.

    Parameters
    ----------
    mesh : SimplexMesh
    f : callable
        Scalar field ``f(x: ndarray) -> float``.
    v_idx : int
    control : {"primal", "dual"}
    n_gauss : int
        Quadrature order on each boundary piece.

    Returns
    -------
    ndarray, shape (dim,)
    """
    dim = mesh.dim

    if control == "primal":
        result = np.zeros(dim)
        for face_pts, A in _primal_star_boundary_pieces(mesh, v_idx):
            if dim == 1:
                result += _integrate_face_1d(f, face_pts[0], A)
            elif dim == 2:
                result += _integrate_face_2d(f, face_pts, A, n_gauss)
            elif dim == 3:
                result += _integrate_face_3d_primal(f, face_pts, A, n_gauss)
            else:
                raise NotImplementedError(f"dim={dim}")
        return result

    if control == "dual":
        result = np.zeros(dim)
        for corners, A_piece, _, _ in _dual_cell_boundary_pieces(mesh, v_idx):
            if dim == 1:
                # corners has 2^0 = 1 point: the edge midpoint
                result += _integrate_face_1d(f, corners[0], A_piece)
            elif dim == 2:
                # corners has 2 points: m_ij and b_T
                result += _integrate_face_2d(f, corners, A_piece, n_gauss)
            elif dim == 3:
                # corners has 4 points (a quadrilateral piece)
                result += _integrate_face_3d_dual(
                    f, corners, A_piece, n_gauss
                )
            else:
                raise NotImplementedError(f"dim={dim}")
        return result

    raise ValueError(f"unknown control volume: {control!r}")


# ---------------------------------------------------------------------------
# Public: analytical integrated gradient (vector field)
# ---------------------------------------------------------------------------

def integrated_gradient_tensor_analytical(
    mesh: SimplexMesh,
    u: Callable[[np.ndarray], np.ndarray],
    v_idx: int,
    control: ControlVolume = "dual",
    n_gauss: int = 10,
) -> np.ndarray:
    """``∫_V ∇u dV = ∮_∂V u ⊗ n dA`` evaluated by Gauss quadrature.

    Returns
    -------
    ndarray, shape (m, dim)
        Where ``m`` is the number of components of ``u``.
    """
    # Probe to get vector dimension.
    sample = np.atleast_1d(u(mesh.vertices[v_idx]))
    m = len(sample)

    def _f_a(a):
        return lambda x: u(x)[a]

    out = np.zeros((m, mesh.dim))
    for a in range(m):
        out[a] = integrated_gradient_analytical(
            mesh, _f_a(a), v_idx, control=control, n_gauss=n_gauss,
        )
    return out


# ---------------------------------------------------------------------------
# Public: analytical integrated Hessian
# ---------------------------------------------------------------------------

def integrated_hessian_analytical(
    mesh: SimplexMesh,
    f: Callable[[np.ndarray], float],
    v_idx: int,
    grad_f: Callable[[np.ndarray], np.ndarray] | None = None,
    control: ControlVolume = "dual",
    n_gauss: int = 10,
    fd_step: float = 1e-6,
) -> np.ndarray:
    """``∫_V ∇⊗∇f dV = ∮_∂V (∇f) ⊗ n dA`` evaluated by Gauss quadrature.

    The boundary integrand needs ``∇f``; either pass a callable
    ``grad_f`` (preferred for analytical exactness) or one is built
    from ``f`` via central finite differences with step ``fd_step``.

    Returns
    -------
    ndarray, shape (dim, dim)
    """
    dim = mesh.dim
    if grad_f is None:
        def grad_f(x):
            g = np.empty(dim)
            for a in range(dim):
                xp, xm = x.copy(), x.copy()
                xp[a] += fd_step
                xm[a] -= fd_step
                g[a] = (f(xp) - f(xm)) / (2 * fd_step)
            return g

    return integrated_gradient_tensor_analytical(
        mesh, grad_f, v_idx, control=control, n_gauss=n_gauss,
    )


__all__ = [
    "integrated_gradient_analytical",
    "integrated_gradient_tensor_analytical",
    "integrated_hessian_analytical",
]
