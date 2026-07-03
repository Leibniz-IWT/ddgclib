#!/usr/bin/env python3
"""Evrard-type paraboloid volume operators for Figure 3-8.

Literature reference
--------------------
Evrard et al. (2023), "First moments of a polyhedron clipped by a paraboloid."

What is implemented here
------------------------
The reference paper derives closed-form moments for a polyhedron clipped by a
paraboloid.  In Figure 3-8 we use this literature line in two ways:

1. For the static paraboloid panel, ``paraboloid_forward_volume`` calls the
   bundled IRL paraboloid-clipping wrapper for the model-matched zeroth moment.
2. For other closed surface meshes, ``paraboloid_taylor_volume`` and
   ``surface_ppic_volume`` use a local paraboloid ``Vpatch`` surrogate.  This is
   not a full PPIC/VOF solver; it applies the Evrard-type paraboloid idea to the
   same triangular surface-patch setting used by the Section 3.4 comparison.

Equations mirrored by this comparison
-------------------------------------
The local patch is written as a quadratic graph,

    x(u, v) = p0 + u e1 + v e2 + h(u, v) n,
    h(u, v) = a u^2 + b u v + c v^2 + d u + e v + f.

The volume contribution is obtained from the divergence theorem,

    V_face = (1/3) int_S x . n_S dA.

For the graph above, this reduces to exact polynomial moments over the
projected triangle Omega:

    V_face = (1/3) int_Omega [
        x . n - (x . e1) dh/du - (x . e2) dh/dv
    ] du dv.

This matches the spirit of Evrard et al.'s moment reduction: use analytic
moments for a paraboloid-like local graph, rather than numerical quadrature.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMMON = HERE / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bootstrap import ensure_runtime  # noqa: E402

ensure_runtime()

import numpy as np  # noqa: E402

IRL_WRAPPER = COMMON / "irl_paraboloid_clip_volume"
IRL_BUILD_SCRIPT = COMMON / "build_irl_paraboloid_clip_volume.sh"


def _parse_irl_key_values(text: str) -> dict[str, float]:
    """Parse key-value output from the bundled IRL paraboloid executable."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def ensure_irl_wrapper() -> None:
    """Build the local IRL paraboloid helper when the executable is unusable."""
    probe = None
    if IRL_WRAPPER.exists() and os.access(IRL_WRAPPER, os.X_OK):
        try:
            probe = subprocess.run(
                [str(IRL_WRAPPER)],
                input="",
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except OSError:
            probe = None
        else:
            # With empty input a valid executable returns cleanly with its own
            # diagnostic, while a wrong-platform binary fails before running.
            if probe.returncode == 1 and "Expected npts ntets" in probe.stderr:
                return

    if not IRL_BUILD_SCRIPT.exists():
        raise FileNotFoundError(
            f"IRL wrapper is not runnable and build script is missing: {IRL_BUILD_SCRIPT}"
        )

    print("Building local IRL paraboloid clipping helper...", flush=True)
    subprocess.run([str(IRL_BUILD_SCRIPT)], cwd=str(COMMON), check=True)


def run_irl_paraboloid_clip(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    datum: tuple[float, float, float],
    frame: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    coefficients: tuple[float, float],
    use_above_region: bool = False,
) -> dict[str, float]:
    """Run the bundled IRL paraboloid-clipping executable on tetrahedra.

    This is used only for the model-matched Evrard-type paraboloid forward
    zeroth-moment comparison.  The executable is kept in ``common/``; the Python
    call path and the comparison equations are kept here in the author-year
    file.
    """
    ensure_irl_wrapper()

    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=int)
    lines = [
        f"{len(points)} {len(tets)}",
        f"{datum[0]} {datum[1]} {datum[2]}",
        " ".join(str(v) for axis in frame for v in axis),
        f"{coefficients[0]} {coefficients[1]} {1 if use_above_region else 0}",
    ]
    lines.extend(f"{p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for p in points)
    lines.extend(f"{int(t[0])} {int(t[1])} {int(t[2])} {int(t[3])}" for t in tets)

    result = subprocess.run(
        [str(IRL_WRAPPER)],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return _parse_irl_key_values(result.stdout)


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


def orient_projected_triangle(
    uv: np.ndarray, values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Orient projected triangle data with positive signed area."""
    if signed_area2(uv) >= 0.0:
        return uv, values
    uv = uv[[0, 2, 1]]
    if values is not None:
        values = values[[0, 2, 1]]
    return uv, values


def tri_moments_uv(uv: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Exact projected triangle moments: 1, u, v, u^2, uv, v^2."""
    area = 0.5 * signed_area2(uv)
    u = uv[:, 0]
    v = uv[:, 1]
    int_u = area / 3.0 * np.sum(u)
    int_v = area / 3.0 * np.sum(v)
    int_u2 = area / 6.0 * (
        np.sum(u * u) + u[0] * u[1] + u[1] * u[2] + u[2] * u[0]
    )
    int_v2 = area / 6.0 * (
        np.sum(v * v) + v[0] * v[1] + v[1] * v[2] + v[2] * v[0]
    )
    int_uv = area / 12.0 * np.sum(u * v)
    int_uv += area / 24.0 * sum(
        u[i] * v[j] for i in range(3) for j in range(3) if i != j
    )
    return float(area), float(int_u), float(int_v), float(int_u2), float(int_uv), float(int_v2)


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


def fit_ppic_integral_graph(
    points: np.ndarray,
    faces: np.ndarray,
    stencil_face_ids: list[int],
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
) -> np.ndarray | None:
    """Fit a quadratic graph using integral moments over neighboring triangles."""
    rows = []
    rhs = []
    for sid in stencil_face_ids:
        pts = points[faces[sid]]
        rel = pts - p0
        uv = np.column_stack((rel @ e1, rel @ e2))
        hvals = rel @ n
        uv, hvals = orient_projected_triangle(uv, hvals)
        if abs(signed_area2(uv)) < 1e-14:
            continue
        area, int_u, int_v, int_u2, int_uv, int_v2 = tri_moments_uv(uv)
        rows.append([int_u2, int_uv, int_v2, int_u, int_v, area])
        rhs.append(area * float(np.mean(hvals)))
    if len(rows) < 6:
        return None
    coeffs, *_ = np.linalg.lstsq(
        np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), rcond=None
    )
    return coeffs


def translate_graph_to_match_plane_patch(
    uv: np.ndarray, hvals: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    """Shift graph constant so its mean height matches the PL triangle patch."""
    area, int_u, int_v, int_u2, int_uv, int_v2 = tri_moments_uv(uv)
    if abs(area) < 1e-14:
        return coeffs
    a, b, c, d, e, f = coeffs
    target = area * float(np.mean(hvals))
    current = a * int_u2 + b * int_uv + c * int_v2 + d * int_u + e * int_v + f * area
    out = np.array(coeffs, dtype=float)
    out[5] += (target - current) / area
    return out


def graph_volume_moments(
    uv: np.ndarray,
    coeffs: np.ndarray | tuple[float, ...],
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
) -> float:
    """Analytic divergence-theorem volume of a quadratic graph patch."""
    a, b, c, d, e, f = [float(v) for v in coeffs]
    area, int_u, int_v, int_u2, int_uv, int_v2 = tri_moments_uv(uv)
    p0n = float(np.dot(p0, n))
    p0e1 = float(np.dot(p0, e1))
    p0e2 = float(np.dot(p0, e2))
    const = p0n + f - p0e1 * d - p0e2 * e
    lin_u = -(2.0 * a * p0e1 + b * p0e2)
    lin_v = -(b * p0e1 + 2.0 * c * p0e2)
    return float(
        (
            const * area
            + lin_u * int_u
            + lin_v * int_v
            - a * int_u2
            - b * int_uv
            - c * int_v2
        )
        / 3.0
    )


def build_curved_adjacency(points: np.ndarray, faces: np.ndarray, is_curved_face):
    """Return curved face ids, face adjacency by vertex, and curved vertices."""
    curved_face_ids = [i for i, face in enumerate(faces) if is_curved_face(points, face)]
    vertex_faces: list[list[int]] = [[] for _ in range(len(points))]
    for fi in curved_face_ids:
        for v in faces[fi]:
            vertex_faces[int(v)].append(fi)
    all_curved_vertices = sorted({int(v) for fi in curved_face_ids for v in faces[fi]})
    return curved_face_ids, vertex_faces, all_curved_vertices


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


def project_to_quadric_along_direction(
    coeffs: tuple[float, ...], base: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Project a point to an implicit quadric along a selected direction."""
    qmat, lin, const = quadric_matrix(coeffs)
    qa = float(direction @ qmat @ direction)
    qb = 2.0 * float(direction @ qmat @ base) + float(lin @ direction)
    qc = quadric_value(qmat, lin, const, base)

    if abs(qa) < 1e-14:
        if abs(qb) < 1e-14:
            return base
        return base - (qc / qb) * direction

    disc = qb * qb - 4.0 * qa * qc
    if disc < -1e-12:
        return base
    disc = max(disc, 0.0)
    root = math.sqrt(disc)
    candidates = [(-qb + root) / (2.0 * qa), (-qb - root) / (2.0 * qa)]
    return base + min(candidates, key=abs) * direction


def paraboloid_taylor_volume(points, faces, is_curved_face, coeffs) -> float:
    """Osculating-paraboloid ``Vpatch`` approximation from an implicit quadric."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    curved_face_ids, _vertex_faces, _all_curved_vertices = build_curved_adjacency(
        points, faces, is_curved_face
    )
    curved_set = set(curved_face_ids)
    qmat, lin, _const = quadric_matrix(coeffs)
    hessian = 2.0 * qmat
    total = 0.0

    for fi, face in enumerate(faces):
        if fi not in curved_set:
            total += face_volume_contribution(points, face)
            continue

        pts = points[face]
        face_normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        face_normal /= np.linalg.norm(face_normal)
        p0 = project_to_quadric_along_direction(coeffs, pts.mean(axis=0), face_normal)

        grad = quadric_grad(qmat, lin, p0)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            total += face_volume_contribution(points, face)
            continue
        normal = grad / grad_norm
        if np.dot(normal, face_normal) < 0.0:
            normal = -normal

        e1, e2, normal = tangent_frame(normal)
        denom = float(grad @ normal)
        if abs(denom) < 1e-14:
            total += face_volume_contribution(points, face)
            continue

        rel = pts - p0
        uv, _ = orient_projected_triangle(np.column_stack((rel @ e1, rel @ e2)))
        a = -0.5 * float(e1 @ hessian @ e1) / denom
        b = -float(e1 @ hessian @ e2) / denom
        c = -0.5 * float(e2 @ hessian @ e2) / denom
        total += graph_volume_moments(uv, (a, b, c, 0.0, 0.0, 0.0), p0, e1, e2, normal)

    return abs(float(total))


def surface_ppic_volume(points, faces, is_curved_face) -> float:
    """Dynamic surface-mesh Evrard-type paraboloid ``Vpatch`` surrogate."""
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
        if len(stencil_faces) < 6:
            stencil_faces = set(curved_face_ids)

        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        normal /= np.linalg.norm(normal)
        p0 = pts.mean(axis=0)
        e1, e2, normal = tangent_frame(normal)

        rel = pts - p0
        uv = np.column_stack((rel @ e1, rel @ e2))
        hvals = rel @ normal
        uv, hvals = orient_projected_triangle(uv, hvals)

        coeffs = fit_ppic_integral_graph(points, faces, sorted(stencil_faces), p0, e1, e2, normal)
        if coeffs is None:
            coeffs = fit_quadratic_graph(points, all_curved_vertices, p0, e1, e2, normal)
        coeffs = translate_graph_to_match_plane_patch(uv, hvals, coeffs)
        total += graph_volume_moments(uv, coeffs, p0, e1, e2, normal)

    return abs(float(total))


def paraboloid_forward_volume(
    points,
    tets,
    *,
    datum=(0.0, 0.0, 0.0),
    frame=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    coefficients=(-1.0, -1.0),
    use_above_region=True,
) -> float:
    """Return the IRL paraboloid-clipping zeroth moment for tetrahedra."""
    out = run_irl_paraboloid_clip(
        np.asarray(points, dtype=float),
        np.asarray(tets, dtype=int),
        datum=datum,
        frame=frame,
        coefficients=coefficients,
        use_above_region=use_above_region,
    )
    return float(out["selected_volume"])


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
