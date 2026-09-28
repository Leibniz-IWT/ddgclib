"""
Exact symbolic integration of gradients over dual cells via sympy.

Uses the divergence theorem identity::

    ∫_V ∇f dV = ∮_{∂V} f · n dA

with sympy's symbolic integrator for exact (machine-precision) results.
This module requires sympy as a dependency.
"""
from __future__ import annotations

import numpy as np
import sympy
from sympy import Rational, Symbol, integrate


def integrated_gradient_sympy_1d(
    f_expr,
    x_sym: Symbol,
    a: float,
    b: float,
) -> np.ndarray:
    """Exact integrated gradient of a scalar field over a 1D interval.

    Computes ``f(b) - f(a)`` symbolically.

    Parameters
    ----------
    f_expr : sympy expression
        Scalar field as a sympy expression in ``x_sym``.
    x_sym : sympy.Symbol
        The spatial variable.
    a, b : float
        Interval endpoints.

    Returns
    -------
    np.ndarray, shape (1,)
        Integrated gradient.
    """
    f_b = float(f_expr.subs(x_sym, sympy.Rational(b).limit_denominator(10**15)))
    f_a = float(f_expr.subs(x_sym, sympy.Rational(a).limit_denominator(10**15)))
    return np.array([f_b - f_a])


def integrated_gradient_sympy_2d(
    f_expr,
    x_sym: Symbol,
    y_sym: Symbol,
    polygon: np.ndarray,
) -> np.ndarray:
    """Exact integrated gradient of a scalar field over a 2D polygon.

    Applies the divergence theorem via sympy line integrals on each edge.

    For each edge ``P_k → P_{k+1}``, parameterizes as::

        x(t) = P_k_x + t * (P_{k+1}_x - P_k_x)
        y(t) = P_k_y + t * (P_{k+1}_y - P_k_y)

    and integrates ``f(x(t), y(t)) * n_outward`` from ``t = 0`` to ``1``.

    Parameters
    ----------
    f_expr : sympy expression
        Scalar field as a sympy expression in ``x_sym, y_sym``.
    x_sym, y_sym : sympy.Symbol
        Spatial variables.
    polygon : np.ndarray, shape (N, 2)
        Ordered polygon vertices (counterclockwise).

    Returns
    -------
    np.ndarray, shape (2,)
        Integrated gradient vector.
    """
    t = Symbol('t')
    result = np.zeros(2)
    N = len(polygon)

    for k in range(N):
        P_k = polygon[k]
        P_next = polygon[(k + 1) % N]
        edge = P_next - P_k

        # Outward normal (CCW polygon: rotate edge 90° CW)
        n_out = np.array([edge[1], -edge[0]])

        # Parameterize the edge
        x_t = sympy.Float(P_k[0]) + t * sympy.Float(edge[0])
        y_t = sympy.Float(P_k[1]) + t * sympy.Float(edge[1])

        # Substitute into f
        f_on_edge = f_expr.subs({x_sym: x_t, y_sym: y_t})

        # Integrate from 0 to 1
        integral_val = float(integrate(f_on_edge, (t, 0, 1)))

        result += integral_val * n_out

    return result


def integrated_gradient_sympy_2d_vector(
    u_exprs: list,
    x_sym: Symbol,
    y_sym: Symbol,
    polygon: np.ndarray,
) -> np.ndarray:
    """Exact integrated gradient tensor of a vector field over a 2D polygon.

    Computes ``∫_V ∇u dV = ∮_{∂V} u ⊗ n dA`` symbolically.

    Parameters
    ----------
    u_exprs : list of sympy expressions
        Vector field components ``[u_x(x,y), u_y(x,y)]``.
    x_sym, y_sym : sympy.Symbol
        Spatial variables.
    polygon : np.ndarray, shape (N, 2)
        Ordered polygon vertices (counterclockwise).

    Returns
    -------
    np.ndarray, shape (len(u_exprs), 2)
        Integrated gradient tensor.
    """
    t = Symbol('t')
    dim_u = len(u_exprs)
    result = np.zeros((dim_u, 2))
    N = len(polygon)

    for k in range(N):
        P_k = polygon[k]
        P_next = polygon[(k + 1) % N]
        edge = P_next - P_k
        n_out = np.array([edge[1], -edge[0]])

        x_t = sympy.Float(P_k[0]) + t * sympy.Float(edge[0])
        y_t = sympy.Float(P_k[1]) + t * sympy.Float(edge[1])

        for i, u_expr in enumerate(u_exprs):
            u_on_edge = u_expr.subs({x_sym: x_t, y_sym: y_t})
            integral_val = float(integrate(u_on_edge, (t, 0, 1)))
            result[i] += integral_val * n_out

    return result


def integrated_gradient_sympy_3d(
    f_expr,
    x_sym: Symbol,
    y_sym: Symbol,
    z_sym: Symbol,
    faces: list[np.ndarray],
) -> np.ndarray:
    """Exact integrated gradient of a scalar field over a 3D polyhedron.

    Applies the divergence theorem via sympy surface integrals over
    face polygons.  Each face is triangulated from its centroid.

    Parameters
    ----------
    f_expr : sympy expression
        Scalar field in ``x_sym, y_sym, z_sym``.
    x_sym, y_sym, z_sym : sympy.Symbol
        Spatial variables.
    faces : list[np.ndarray]
        Face polygons with outward-facing orientation.

    Returns
    -------
    np.ndarray, shape (3,)
        Integrated gradient vector.
    """
    s, t_var = Symbol('s'), Symbol('t')
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
            # Parameterize: x = centroid + s*(v0-centroid) + t*(v1-centroid)
            # with s >= 0, t >= 0, s+t <= 1

            # Normal vector (cross product, unnormalized)
            e1 = v0 - centroid
            e2 = v1 - centroid
            n_tri = np.cross(e1, e2)

            # Parameterize
            x_st = (sympy.Float(centroid[0])
                    + s * sympy.Float(e1[0])
                    + t_var * sympy.Float(e2[0]))
            y_st = (sympy.Float(centroid[1])
                    + s * sympy.Float(e1[1])
                    + t_var * sympy.Float(e2[1]))
            z_st = (sympy.Float(centroid[2])
                    + s * sympy.Float(e1[2])
                    + t_var * sympy.Float(e2[2]))

            f_on_tri = f_expr.subs({
                x_sym: x_st, y_sym: y_st, z_sym: z_st,
            })

            # Integrate over reference triangle: s in [0,1], t in [0, 1-s]
            integral_val = float(
                integrate(integrate(f_on_tri, (t_var, 0, 1 - s)), (s, 0, 1))
            )

            # The surface integral is: integral_val * n_tri
            # (the Jacobian |e1 x e2| cancels with 1/|n| * |n|)
            result += integral_val * n_tri

    return result
