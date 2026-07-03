#!/usr/bin/env python3
"""THINC/QQ-type quadratic Gaussian-quadrature operators for Figure 3-8.

Literature reference
--------------------
Xie and Xiao (2017), THINC/QQ.

What is implemented here
------------------------
This file implements the quadratic-surface / Gaussian-quadrature volume model
used for the Section 3.4 comparison.  It is not a full THINC/QQ transport
solver.  The full method in Xie and Xiao includes the hyperbolic-tangent
indicator, shift-parameter solve, flux update, and advection procedure.  For
Figure 3-8, we isolate the part relevant to the reviewer question: a quadratic
surface is used, and its volume contribution is evaluated by finite Gaussian
quadrature rather than by the present paper's class-specific closed-form
subtraction.

Equations mirrored by this comparison
-------------------------------------
Xie and Xiao use a quadratic interface polynomial, written generically here as

    F(x, y, z) =
        A x^2 + B y^2 + C z^2 + D x y + E x z + F y z
        + G x + H y + I z + J = 0.

This corresponds to the quadratic surface model in their Eq. (9).  In a local
triangle frame, we write a patch as

    x(u, v) = p0 + u e1 + v e2 + h(u, v) n,

where h is either obtained by solving the implicit quadratic at each quadrature
point, or by fitting a local quadratic graph

    h(u, v) = a u^2 + b u v + c v^2 + d u + e v + f.

The volume contribution follows the divergence theorem,

    V_face = (1/3) int_S x . n_S dA,

which, for the graph above, becomes

    V_face = (1/3) int_Omega [
        x . n - (x . e1) dh/du - (x . e2) dh/dv
    ] du dv.

The THINC/QQ-style comparison evaluates this projected integral by Gaussian
quadrature, in the spirit of Xie and Xiao's Eqs. (20)-(23) and Appendix B.

Functions used by Figure 3-8
----------------------------
gaussian_quadric_volume(points, faces, is_curved_face, coeffs)
    Static benchmark panels.  The benchmark quadric coefficients are supplied
    directly by the case script.

local_quadratic_gq_volume(points, faces, is_curved_face)
    Dynamic panels.  A local quadratic graph is fitted from the current surface
    mesh and integrated by Gaussian quadrature.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import numpy as np  # noqa: E402


def face_volume_contribution(points: np.ndarray, face: np.ndarray) -> float:
    """Planar closed-surface contribution for one oriented triangle."""
    a, b, c = face
    return float(np.dot(points[a], np.cross(points[b], points[c]))) / 6.0


def tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal local basis (e1, e2, n) around a face normal."""
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, n))) > 0.8:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2, n


def signed_area2(uv: np.ndarray) -> float:
    """Twice the signed area of a projected triangle in (u, v)."""
    return float(
        (uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        - (uv[1, 1] - uv[0, 1]) * (uv[2, 0] - uv[0, 0])
    )


def orient_projected_triangle(uv: np.ndarray) -> np.ndarray:
    """Use positive projected orientation for quadrature."""
    if signed_area2(uv) >= 0.0:
        return uv
    return uv[[0, 2, 1]]


def triangle_quadrature_points() -> list[tuple[float, tuple[float, float, float]]]:
    """Seven-point Dunavant triangle rule for the implicit-quadric integral."""
    return [
        (0.225000000000000, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        (0.132394152788506, (0.059715871789770, 0.470142064105115, 0.470142064105115)),
        (0.132394152788506, (0.470142064105115, 0.059715871789770, 0.470142064105115)),
        (0.132394152788506, (0.470142064105115, 0.470142064105115, 0.059715871789770)),
        (0.125939180544827, (0.797426985353087, 0.101286507323456, 0.101286507323456)),
        (0.125939180544827, (0.101286507323456, 0.797426985353087, 0.101286507323456)),
        (0.125939180544827, (0.101286507323456, 0.101286507323456, 0.797426985353087)),
    ]


def graph_quadrature_points() -> list[tuple[float, tuple[float, float, float]]]:
    """Three-point rule used for the fitted local quadratic graph surrogate."""
    return [
        (1.0 / 3.0, (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0)),
        (1.0 / 3.0, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)),
        (1.0 / 3.0, (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)),
    ]


def fit_quadratic_graph(
    points: np.ndarray,
    vertex_ids: list[int],
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """Fit h(u,v)=a u^2+buv+c v^2+d u+e v+f by least squares."""
    rel = points[np.asarray(vertex_ids, dtype=int)] - p0
    u = rel @ e1
    v = rel @ e2
    h = rel @ n
    mat = np.column_stack((u * u, u * v, v * v, u, v, np.ones_like(u)))
    coeffs, *_ = np.linalg.lstsq(mat, h, rcond=None)
    return coeffs


def graph_volume_gq(
    uv: np.ndarray,
    coeffs: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
) -> float:
    """Gaussian-quadrature volume of a fitted quadratic graph patch."""
    a, b, c, d, e, f = [float(v) for v in coeffs]
    area = 0.5 * signed_area2(uv)
    if area <= 0.0:
        return 0.0

    p0n = float(np.dot(p0, n))
    p0e1 = float(np.dot(p0, e1))
    p0e2 = float(np.dot(p0, e2))
    total = 0.0
    for weight, bary in graph_quadrature_points():
        u, v = np.asarray(bary) @ uv
        h = a * u * u + b * u * v + c * v * v + d * u + e * v + f
        hu = 2.0 * a * u + b * v + d
        hv = b * u + 2.0 * c * v + e
        integrand = p0n + h - (p0e1 + u) * hu - (p0e2 + v) * hv
        total += weight * integrand
    return area * total / 3.0


def quadric_matrix(coeffs: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert [A,B,C,D,E,F,G,H,I,J] into x^T Q x + l^T x + J."""
    A, B, C, D, E, F, G, H, I, J = [float(v) for v in coeffs]
    qmat = np.array(
        [
            [A, 0.5 * D, 0.5 * E],
            [0.5 * D, B, 0.5 * F],
            [0.5 * E, 0.5 * F, C],
        ],
        dtype=float,
    )
    lin = np.array([G, H, I], dtype=float)
    return qmat, lin, float(J)


def quadric_value(qmat: np.ndarray, lin: np.ndarray, const: float, x: np.ndarray) -> float:
    return float(x @ qmat @ x + lin @ x + const)


def quadric_grad(qmat: np.ndarray, lin: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 2.0 * (qmat @ x) + lin


def implicit_quadric_graph_volume_gq(
    uv: np.ndarray,
    coeffs: tuple[float, ...],
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
    *,
    quadrature_points: list[tuple[float, tuple[float, float, float]]] | None = None,
) -> float | None:
    """Volume contribution of an implicit quadratic patch by Gaussian quadrature."""
    area = 0.5 * signed_area2(uv)
    if area <= 0.0:
        return None

    qmat, lin, const = quadric_matrix(coeffs)
    qnn = float(n @ qmat @ n)
    total = 0.0
    if quadrature_points is None:
        qrule = triangle_quadrature_points()
    elif callable(quadrature_points):
        qrule = quadrature_points()
    else:
        qrule = quadrature_points

    for weight, bary in qrule:
        u, v = np.asarray(bary) @ uv
        base = p0 + u * e1 + v * e2
        qb = 2.0 * float(n @ qmat @ base) + float(lin @ n)
        qc = quadric_value(qmat, lin, const, base)

        if abs(qnn) < 1e-14:
            if abs(qb) < 1e-14:
                return None
            h = -qc / qb
        else:
            disc = qb * qb - 4.0 * qnn * qc
            if disc < -1e-12:
                return None
            disc = max(disc, 0.0)
            root = math.sqrt(disc)
            candidates = [(-qb + root) / (2.0 * qnn), (-qb - root) / (2.0 * qnn)]
            h = min(candidates, key=abs)

        x = base + h * n
        grad = quadric_grad(qmat, lin, x)
        fh = float(grad @ n)
        if abs(fh) < 1e-14:
            return None
        hu = -float(grad @ e1) / fh
        hv = -float(grad @ e2) / fh
        integrand = float(x @ n) - float(x @ e1) * hu - float(x @ e2) * hv
        total += weight * integrand

    return area * total / 3.0


def build_curved_adjacency(points: np.ndarray, faces: np.ndarray, is_curved_face):
    """Return curved face ids, face adjacency by vertex, and curved vertices."""
    curved_face_ids = [i for i, face in enumerate(faces) if is_curved_face(points, face)]
    vertex_faces: list[list[int]] = [[] for _ in range(len(points))]
    for fi in curved_face_ids:
        for v in faces[fi]:
            vertex_faces[int(v)].append(fi)
    all_curved_vertices = sorted({int(v) for fi in curved_face_ids for v in faces[fi]})
    return curved_face_ids, vertex_faces, all_curved_vertices


def gaussian_quadric_volume(
    points,
    faces,
    is_curved_face,
    coeffs,
    *,
    quadrature_points: list[tuple[float, tuple[float, float, float]]] | None = None,
) -> float:
    """Static Figure 3-8 THINC/QQ-type volume from supplied quadric coefficients."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    curved_face_ids, _vertex_faces, _all_curved_vertices = build_curved_adjacency(
        points, faces, is_curved_face
    )
    curved_set = set(curved_face_ids)
    total = 0.0

    for fi, face in enumerate(faces):
        if fi not in curved_set:
            total += face_volume_contribution(points, face)
            continue

        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        normal /= np.linalg.norm(normal)
        p0 = pts.mean(axis=0)
        e1, e2, normal = tangent_frame(normal)

        rel = pts - p0
        uv = orient_projected_triangle(np.column_stack((rel @ e1, rel @ e2)))
        value = implicit_quadric_graph_volume_gq(
            uv,
            coeffs,
            p0,
            e1,
            e2,
            normal,
            quadrature_points=quadrature_points,
        )
        if value is None:
            value = face_volume_contribution(points, face)
        total += value

    return abs(float(total))


def local_quadratic_gq_volume(points, faces, is_curved_face) -> float:
    """Dynamic Figure 3-8 THINC/QQ-type volume from a fitted local quadratic."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    curved_face_ids, vertex_faces, all_curved_vertices = build_curved_adjacency(
        points, faces, is_curved_face
    )
    curved_set = set(curved_face_ids)
    total = 0.0

    for fi, face in enumerate(faces):
        if fi not in curved_set:
            total += face_volume_contribution(points, face)
            continue

        stencil_faces: set[int] = set()
        for v in face:
            stencil_faces.update(vertex_faces[int(v)])
        stencil_vertices = sorted({int(v) for sid in stencil_faces for v in faces[sid]})
        if len(stencil_vertices) < 6:
            stencil_vertices = all_curved_vertices

        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        normal /= np.linalg.norm(normal)
        p0 = pts.mean(axis=0)
        e1, e2, normal = tangent_frame(normal)

        rel = pts - p0
        uv = orient_projected_triangle(np.column_stack((rel @ e1, rel @ e2)))
        coeffs = fit_quadratic_graph(points, stencil_vertices, p0, e1, e2, normal)
        total += graph_volume_gq(uv, coeffs, p0, e1, e2, normal)

    return abs(float(total))


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
