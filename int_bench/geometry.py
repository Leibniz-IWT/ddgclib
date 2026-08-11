"""
Pure geometric primitives for n-dimensional simplicial meshes.

All routines are dimension-agnostic and operate on raw NumPy arrays.
The two key primitives are:

* ``simplex_volume`` — the unsigned n-volume of an n-simplex in R^n
  (via the determinant of the edge matrix divided by n!).
* ``area_vector`` — the (n-1)-area vector of an oriented (n-1)-simplex
  in R^n, computed via the generalized cross product
  (Hodge dual of the wedge product). Magnitude equals the (n-1)-volume,
  direction is normal to the simplex.

These are the only n-dimensional building blocks we need: every higher
construct (dual cell area vector, primal star boundary, etc.) is a
weighted sum of triangulated (n-1)-simplices.
"""
from __future__ import annotations

import math

import numpy as np


def simplex_volume(verts: np.ndarray) -> float:
    """Unsigned volume of an n-simplex with (n+1) vertices in R^n.

    Parameters
    ----------
    verts : ndarray, shape (n+1, n)
        Vertices of the simplex.

    Returns
    -------
    float
        n-volume (length in 1D, area in 2D, volume in 3D, ...).
    """
    verts = np.asarray(verts, dtype=float)
    n_plus_1, n = verts.shape
    if n_plus_1 != n + 1:
        raise ValueError(
            f"expected (n+1, n) shape, got {verts.shape}"
        )
    edges = verts[1:] - verts[0]                     # (n, n)
    return abs(np.linalg.det(edges)) / math.factorial(n)


def simplex_centroid(verts: np.ndarray) -> np.ndarray:
    """Centroid (barycenter) of a simplex."""
    return np.asarray(verts, dtype=float).mean(axis=0)


def area_vector(simplex_verts: np.ndarray) -> np.ndarray:
    """(n-1)-area vector of an (n-1)-simplex embedded in R^n.

    Implements the generalized cross product / Hodge dual of the
    wedge product of edges: for vertices q_0, ..., q_{n-1} in R^n,

        A_i = (1 / (n-1)!) * (-1)^i * det(M_i)

    where ``M_i`` is the (n-1) x (n-1) matrix obtained by stacking
    edges ``q_k - q_0`` (k = 1..n-1) and removing column ``i``.

    The magnitude equals the (n-1)-volume of the simplex; the
    direction is normal to it. For n=2 this is the perpendicular to
    a line segment; for n=3 this is half a cross product.

    Parameters
    ----------
    simplex_verts : ndarray, shape (n, n)
        n vertices of the (n-1)-simplex in R^n, in oriented order.

    Returns
    -------
    ndarray, shape (n,)
        Area vector.
    """
    pts = np.asarray(simplex_verts, dtype=float)
    n_pts, n = pts.shape
    if n_pts != n:
        raise ValueError(
            f"expected (n, n) shape for an (n-1)-simplex in R^n, "
            f"got {pts.shape}"
        )
    if n == 1:
        # A "0-simplex" in R^1 is a point; area = 1, sign carried by caller.
        return np.array([1.0])

    edges = pts[1:] - pts[0]                          # (n-1, n)
    A = np.empty(n)
    for i in range(n):
        cols = [j for j in range(n) if j != i]
        A[i] = ((-1) ** i) * np.linalg.det(edges[:, cols])
    return A / math.factorial(n - 1)


def polytope_area_vector(
    poly_verts: np.ndarray,
    fan_apex: np.ndarray | None = None,
) -> np.ndarray:
    """(n-1)-area vector of an oriented polytope in R^n.

    The polytope is fan-triangulated from ``fan_apex`` (or its
    centroid) into (n-1)-simplices and the contributions are summed.
    Vertex order is treated as a cyclic boundary in 2D / a list of
    triangle-fan vertices in 3D / etc.

    For n=2: ``poly_verts`` has shape (k, 2). The output is the area
    vector of the segment from ``poly_verts[0]`` to ``poly_verts[-1]``
    only if all points are colinear; otherwise the user should pass a
    flat polygon and we sum oriented triangles from its centroid.

    Parameters
    ----------
    poly_verts : ndarray, shape (k, n)
        Boundary vertices of an (n-1)-polytope in R^n. For n=2, two
        endpoints of a segment. For n=3, a polygon. Must be planar.
    fan_apex : ndarray, optional
        Apex from which to fan-triangulate. Defaults to the centroid.

    Returns
    -------
    ndarray, shape (n,)
        Sum of area vectors over fan triangles.
    """
    pts = np.asarray(poly_verts, dtype=float)
    if pts.ndim != 2:
        raise ValueError("poly_verts must be 2D")
    k, n = pts.shape

    if n == 1:
        if k != 1:
            raise ValueError("R^1: expected exactly one boundary point")
        return np.array([1.0])
    if n == 2:
        # In R^2 the boundary of a 2-cell is a closed polyline; area
        # vector contributions come from each segment. But here we
        # interpret poly_verts as the two endpoints of one (n-1)=1
        # segment and just return its area_vector.
        if k == 2:
            return area_vector(pts)
        raise ValueError(
            f"R^2 expects 2 endpoints; got {k}. Use sum over edges."
        )

    if k < n:
        # Not enough vertices to form a single (n-1)-simplex.
        return np.zeros(n)

    apex = pts.mean(axis=0) if fan_apex is None else np.asarray(fan_apex)
    A_total = np.zeros(n)
    for k_i in range(k):
        # Triangle fan: (apex, pts[k_i], pts[(k_i+1)%k]) — only
        # closes correctly when poly_verts is a true cycle in R^3
        # (n=3) or an oriented (n-1)-cycle.
        tri = np.vstack(
            [apex, pts[k_i], pts[(k_i + 1) % k]]
        )
        # In R^3 the (n-1)-simplex is a 2-simplex (triangle).
        # ``area_vector`` works for any n where simplex_verts has
        # shape (n, n). Here we have (3, 3), so direct call works
        # only when n == 3. For higher n we need true (n-1)-simplex
        # tesselation.
        if n == 3:
            A_total += area_vector(tri)
        else:
            raise NotImplementedError(
                "polytope_area_vector for n > 3 needs cube-Kuhn "
                "triangulation; use ``barycentric_dual_face_vector`` "
                "directly."
            )
    return A_total


# ---------------------------------------------------------------------------
# Barycentric subdivision: dual face area vector contribution per simplex
# ---------------------------------------------------------------------------

def _permutation_sign(perm: tuple[int, ...]) -> int:
    """Sign of a permutation (+1 even, -1 odd)."""
    n = len(perm)
    sign = 1
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        j = i
        cycle_len = 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign = -sign
    return sign


def _kuhn_simplices_unit_cube(d: int) -> list[tuple[tuple[int, ...], int]]:
    """Kuhn triangulation of the unit d-cube into d! signed simplices.

    Each simplex is described by a path through the lattice
    ``{0, 1}^d`` that flips one coordinate at a time, plus a sign
    (= sign of the underlying permutation). When summing the area
    vectors of these (d)-simplices to obtain the (d)-area vector of
    the cube embedded in higher-dimensional space, signs are needed
    to keep all per-simplex normals consistent: every simplex of the
    Kuhn triangulation inherits the cube's natural orientation only
    up to the permutation sign.

    Returns
    -------
    list of (path, sign)
        ``path`` is a sequence of (d+1) corner indices into the
        2^d lattice points of the d-cube; ``sign`` is +1 or -1.
    """
    from itertools import permutations

    out = []
    for perm in permutations(range(d)):
        path = [0]
        bits = 0
        for axis in perm:
            bits |= (1 << axis)
            path.append(bits)
        out.append((tuple(path), _permutation_sign(perm)))
    return out


def barycentric_dual_face_vector(
    simplex_verts: np.ndarray,
    i_local: int,
    j_local: int,
) -> np.ndarray:
    """Dual face area vector contribution from one n-simplex.

    The barycentric dual face between vertices ``i`` and ``j`` of an
    n-simplex T is the (n-1)-cell whose vertices are the barycenters
    of all faces F of T satisfying ``{i, j} ⊆ F``. There are 2^(n-1)
    such faces (one for each subset of the "other" vertices), and the
    cell is combinatorially an (n-1)-cube.

    The cell is split into (n-1)! simplices via the Kuhn triangulation
    of the (n-1)-cube; their area vectors are summed and re-oriented
    to point from ``i`` to ``j``.

    Parameters
    ----------
    simplex_verts : ndarray, shape (n+1, n)
        Vertices of the n-simplex T.
    i_local, j_local : int
        Local indices of the edge endpoints in T.

    Returns
    -------
    ndarray, shape (n,)
        Dual face area vector pointing from vertex i to vertex j.
    """
    pts = np.asarray(simplex_verts, dtype=float)
    n_plus_1, n = pts.shape
    if i_local == j_local or not (0 <= i_local < n_plus_1) or not (
        0 <= j_local < n_plus_1
    ):
        raise ValueError("invalid edge endpoints")

    other = [k for k in range(n_plus_1) if k not in (i_local, j_local)]
    # Enumerate subsets of `other` indexed by binary mask.
    n_other = len(other)             # = n - 1
    bary_pts = np.empty((1 << n_other, n))
    for mask in range(1 << n_other):
        face_idx = [i_local, j_local]
        for b in range(n_other):
            if mask & (1 << b):
                face_idx.append(other[b])
        bary_pts[mask] = pts[face_idx].mean(axis=0)

    # Trivial 1D case: T is an edge, "dual face" is just the
    # endpoint of the dual segment closest to i (the edge midpoint).
    # In R^1 area is a scalar with sign = direction from i to j.
    if n == 1:
        sign = 1.0 if pts[j_local, 0] > pts[i_local, 0] else -1.0
        return np.array([sign])

    # General nD: Kuhn-triangulate the (n-1)-cube of bary_pts.
    A = np.zeros(n)
    cube_dim = n_other                   # = n - 1
    if cube_dim == 1:
        # 1-cube = segment between bary_pts[0] (= m_ij) and bary_pts[1] (= b_T)
        seg = bary_pts                   # (2, n)
        A_seg = area_vector(seg)         # (n,)  — perpendicular in R^2
        A = A_seg
    else:
        for path, sign in _kuhn_simplices_unit_cube(cube_dim):
            simplex = bary_pts[list(path)]   # (cube_dim+1, n) = (n, n)
            A += sign * area_vector(simplex)

    # Re-orient so the result points from i to j.
    edge_vec = pts[j_local] - pts[i_local]
    if np.dot(A, edge_vec) < 0:
        A = -A
    return A


# ---------------------------------------------------------------------------
# Polynomial integration over a simplex (exact)
# ---------------------------------------------------------------------------

def integrate_polynomial_over_simplex(
    coeffs: dict[tuple[int, ...], float],
    verts: np.ndarray,
) -> float:
    """Exact integral of a polynomial in n variables over an n-simplex.

    Uses the standard barycentric formula:

        ∫_T x_1^{a_1} ... x_n^{a_n} dV = vol(T) * Σ_{multi-indices}
            (a! / (|a| + n)! * something) ...

    For our purposes we only need this for low-degree validation; we
    delegate to Gauss quadrature in :mod:`int_bench.analytical`.
    """
    raise NotImplementedError(
        "use Gauss quadrature in int_bench.analytical instead"
    )


__all__ = [
    "simplex_volume",
    "simplex_centroid",
    "area_vector",
    "polytope_area_vector",
    "barycentric_dual_face_vector",
]
