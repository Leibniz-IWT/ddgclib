"""Parametric surface mesh generation.

Provides a generic ``parametric_surface()`` builder that creates a 3D
simplicial complex from any parametric function *f(u, v) → (x, y, z)*,
plus convenience wrappers for standard surfaces (sphere, catenoid,
cylinder, hyperboloid, torus, plane).

The pipeline is:

    Complex(2, domain) → triangulate → refine → project via f(u,v)→(x,y,z)
      → copy connectivity to Complex(3) → merge_all → identify boundary

All convenience generators delegate to ``parametric_surface()``.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
from hyperct import Complex


# ── Core builder ──────────────────────────────────────────────────────────


def parametric_surface(
    f: Callable[[float, float], tuple[float, float, float]],
    domain: list[tuple[float, float]],
    refinement: int = 2,
    cdist: float = 1e-8,
    boundary_fn: Callable[[tuple, list[tuple[float, float]]], bool] | None = None,
) -> tuple[Complex, set]:
    """Create a 3D surface mesh from a parametric function.

    Parameters
    ----------
    f : callable(u, v) → (x, y, z)
        Parametric mapping from 2D parameter space to 3D.
    domain : list of 2 tuples
        Parameter bounds ``[(u_lo, u_hi), (v_lo, v_hi)]``.
    refinement : int
        Number of ``refine_all()`` passes on the parameter-space mesh.
    cdist : float
        Merge tolerance for ``merge_all()`` (joins coincident vertices
        at seams, poles, or periodic boundaries).
    boundary_fn : callable(param_coords, domain) → bool, optional
        Predicate that returns True for parameter-space coordinates that
        should be tagged as boundary.  Receives the 2D parameter tuple
        ``(u, v)`` and the domain list.  If *None*, vertices on any edge
        of the parameter domain are treated as boundary.

    Returns
    -------
    HC : Complex
        3D simplicial complex with the projected surface mesh.
    bV : set
        Set of boundary vertex objects.
    """
    if boundary_fn is None:
        boundary_fn = _default_boundary

    # 1. Build 2D parameter-space mesh
    HC_plane = Complex(2, domain)
    HC_plane.triangulate()
    for _ in range(refinement):
        HC_plane.refine_all()

    # 2. Project to 3D — create vertices and track mapping
    HC = Complex(3, domain)
    bV: set = set()
    plane_to_3d: dict = {}  # id(plane_vertex) → 3d_vertex

    for v_plane in HC_plane.V:
        u_val, v_val = v_plane.x_a[0], v_plane.x_a[1]
        x, y, z = f(u_val, v_val)
        v3d = HC.V[tuple([x, y, z])]
        plane_to_3d[id(v_plane)] = v3d

        if boundary_fn(v_plane.x, domain):
            bV.add(v3d)

    # 3. Copy connectivity from 2D mesh
    for v_plane in HC_plane.V:
        v3d_src = plane_to_3d[id(v_plane)]
        for nb_plane in v_plane.nn:
            v3d_dst = plane_to_3d[id(nb_plane)]
            v3d_src.connect(v3d_dst)

    # 4. Merge coincident vertices (poles, periodic seams)
    HC.V.merge_all(cdist=cdist)

    # 5. Clean up bV — remove any vertices that were merged away
    bV = {v for v in bV if v in HC.V}

    return HC, bV


# ── Boundary predicates ──────────────────────────────────────────────────


def _default_boundary(
    param_coords: tuple, domain: list[tuple[float, float]]
) -> bool:
    """True if the vertex lies on any edge of the parameter domain."""
    for i, (lo, hi) in enumerate(domain):
        if param_coords[i] == lo or param_coords[i] == hi:
            return True
    return False


def _second_axis_boundary(
    param_coords: tuple, domain: list[tuple[float, float]]
) -> bool:
    """True if the vertex lies on the v-axis boundary (index 1) only.

    Used for surfaces periodic in u (catenoid, cylinder, hyperboloid)
    where the u-edges wrap around and should NOT be treated as boundary.
    """
    lo, hi = domain[1]
    return param_coords[1] == lo or param_coords[1] == hi


def _first_axis_boundary(
    param_coords: tuple, domain: list[tuple[float, float]]
) -> bool:
    """True if the vertex lies on the u-axis boundary (index 0) only."""
    lo, hi = domain[0]
    return param_coords[0] == lo or param_coords[0] == hi


def _no_boundary(
    param_coords: tuple, domain: list[tuple[float, float]]
) -> bool:
    """Always False — for closed surfaces (torus)."""
    return False


# ── Convenience generators ───────────────────────────────────────────────


def sphere(
    R: float = 1.0,
    refinement: int = 2,
    theta_range: tuple[float, float] = (0.0, 2 * np.pi),
    phi_range: tuple[float, float] = (0.01, np.pi - 0.01),
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """Sphere (or spherical sector) of radius *R*.

    Parameters
    ----------
    R : float
        Radius.
    refinement : int
        Mesh refinement level.
    theta_range : tuple
        Azimuthal angle range ``(theta_lo, theta_hi)``.
    phi_range : tuple
        Polar angle range ``(phi_lo, phi_hi)``.
        Avoid exact 0 and pi to prevent pole singularity; use e.g.
        ``(0.01, pi-0.01)`` for a nearly-full sphere.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    def f(theta: float, phi: float) -> tuple[float, float, float]:
        return (
            R * np.cos(theta) * np.sin(phi),
            R * np.sin(theta) * np.sin(phi),
            R * np.cos(phi),
        )

    return parametric_surface(
        f, [theta_range, phi_range], refinement, cdist,
        boundary_fn=_second_axis_boundary,
    )


def catenoid(
    a: float = 1.0,
    v_range: tuple[float, float] = (-1.5, 1.5),
    refinement: int = 2,
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """Catenoid (minimal surface) with neck radius *a*.

    Parameters
    ----------
    a : float
        Neck radius parameter.
    v_range : tuple
        Axial extent ``(v_lo, v_hi)``.
    refinement : int
        Mesh refinement level.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    def f(u: float, v: float) -> tuple[float, float, float]:
        return (
            a * np.cos(u) * np.cosh(v / a),
            a * np.sin(u) * np.cosh(v / a),
            v,
        )

    return parametric_surface(
        f, [(0.0, 2 * np.pi), v_range], refinement, cdist,
        boundary_fn=_second_axis_boundary,
    )


def cylinder(
    R: float = 1.0,
    h_range: tuple[float, float] = (-1.0, 1.0),
    refinement: int = 2,
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """Open cylinder of radius *R*.

    Parameters
    ----------
    R : float
        Cylinder radius.
    h_range : tuple
        Height range ``(z_lo, z_hi)``.
    refinement : int
        Mesh refinement level.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    def f(theta: float, z: float) -> tuple[float, float, float]:
        return (R * np.cos(theta), R * np.sin(theta), z)

    return parametric_surface(
        f, [(0.0, 2 * np.pi), h_range], refinement, cdist,
        boundary_fn=_second_axis_boundary,
    )


def hyperboloid(
    a: float = 1.0,
    b: float | None = None,
    c: float = 1.0,
    v_range: tuple[float, float] = (-2.0, 2.0),
    refinement: int = 2,
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """One-sheeted hyperboloid of revolution.

    Parametric form:  x = a*sqrt(u²+1)*cos(v),  y = b*sqrt(u²+1)*sin(v),  z = c*u

    Parameters
    ----------
    a, b, c : float
        Shape parameters.  If *b* is None, uses ``b = a`` (surface of revolution).
    v_range : tuple
        Axial parameter range.
    refinement : int
        Mesh refinement level.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    if b is None:
        b = a

    def f(u: float, v: float) -> tuple[float, float, float]:
        return (
            a * np.sqrt(u ** 2 + 1) * np.cos(v),
            b * np.sqrt(u ** 2 + 1) * np.sin(v),
            c * u,
        )

    return parametric_surface(
        f, [v_range, (0.0, 2 * np.pi)], refinement, cdist,
        boundary_fn=_first_axis_boundary,
    )


def torus(
    R: float = 2.0,
    r: float = 0.5,
    refinement: int = 2,
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """Torus with major radius *R* and minor radius *r*.

    Closed surface — no boundary vertices.

    Parameters
    ----------
    R : float
        Major radius (centre of tube to centre of torus).
    r : float
        Minor radius (tube radius).
    refinement : int
        Mesh refinement level.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    def f(u: float, v: float) -> tuple[float, float, float]:
        return (
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v),
        )

    return parametric_surface(
        f, [(0.0, 2 * np.pi), (0.0, 2 * np.pi)], refinement, cdist,
        boundary_fn=_no_boundary,
    )


def plane(
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    refinement: int = 2,
    cdist: float = 1e-8,
) -> tuple[Complex, set]:
    """Flat plane in the xy-plane at z = 0.

    Parameters
    ----------
    x_range, y_range : tuple
        Extent of the plane.
    refinement : int
        Mesh refinement level.
    cdist : float
        Merge tolerance.

    Returns
    -------
    HC, bV
    """
    def f(x: float, y: float) -> tuple[float, float, float]:
        return (x, y, 0.0)

    return parametric_surface(
        f, [x_range, y_range], refinement, cdist,
    )


# ── Composition helpers ──────────────────────────────────────────────────


def translate_surface(HC: Complex, offset: np.ndarray | list) -> None:
    """Translate all vertices of a 3D surface mesh by *offset* (in-place).

    Parameters
    ----------
    HC : Complex
        Surface mesh to translate.
    offset : array-like of length 3
        Translation vector ``[dx, dy, dz]``.
    """
    offset = np.asarray(offset, dtype=float)
    for v in list(HC.V):
        new_pos = v.x_a.copy()
        new_pos[:len(offset)] += offset
        HC.V.move(v, tuple(new_pos))


def rotate_surface(
    HC: Complex, rotation_matrix: np.ndarray, center: np.ndarray | list | None = None
) -> None:
    """Rotate all vertices of a surface mesh (in-place).

    Parameters
    ----------
    HC : Complex
        Surface mesh to rotate.
    rotation_matrix : (3, 3) array
        Rotation matrix to apply.
    center : array-like or None
        Centre of rotation.  If None, rotates about the origin.
    """
    R = np.asarray(rotation_matrix, dtype=float)
    if center is not None:
        center = np.asarray(center, dtype=float)
    for v in list(HC.V):
        pos = v.x_a[:3].copy()
        if center is not None:
            pos = center + R @ (pos - center)
        else:
            pos = R @ pos
        new = v.x_a.copy()
        new[:3] = pos
        HC.V.move(v, tuple(new))


def rotation_matrix_align(axis_from: np.ndarray, axis_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps *axis_from* direction to *axis_to*.

    Uses the Rodrigues rotation formula.

    Parameters
    ----------
    axis_from, axis_to : (3,) array-like
        Direction vectors (need not be unit length).

    Returns
    -------
    R : (3, 3) ndarray
        Rotation matrix.
    """
    a = np.asarray(axis_from, dtype=float)
    b = np.asarray(axis_to, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        # Vectors are (anti-)parallel
        if c > 0:
            return np.eye(3)
        # 180-degree rotation: find a perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def scale_surface(
    HC: Complex, factor: float, center: np.ndarray | list | None = None
) -> None:
    """Scale all vertices of a surface mesh about *center* (in-place).

    Parameters
    ----------
    HC : Complex
        Surface mesh to scale.
    factor : float
        Scale factor.
    center : array-like or None
        Centre of scaling.  If None, scales about the origin.
    """
    if center is not None:
        center = np.asarray(center, dtype=float)
    for v in list(HC.V):
        pos = v.x_a.copy()
        if center is not None:
            pos = center + factor * (pos - center)
        else:
            pos = factor * pos
        HC.V.move(v, tuple(pos))
