"""Helpers for identifying named boundary vertex groups on domain meshes."""
from __future__ import annotations

import numpy as np
from hyperct import Complex


def identify_face_groups(
    HC: Complex,
    bounds: dict[str, tuple[int, float]],
    tol: float = 1e-14,
) -> dict[str, set]:
    """Identify boundary vertex groups by axis-aligned face position.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    bounds : dict
        Maps group name to ``(axis, position)``.
        Example::

            {'inlet': (0, 0.0), 'outlet': (0, 10.0),
             'bottom_wall': (1, 0.0), 'top_wall': (1, 1.0)}

    tol : float
        Position tolerance for face detection.

    Returns
    -------
    dict[str, set]
        Mapping from group name to set of vertex objects.
    """
    groups: dict[str, set] = {name: set() for name in bounds}
    for v in HC.V:
        for name, (axis, pos) in bounds.items():
            if abs(v.x_a[axis] - pos) < tol:
                groups[name].add(v)
    return groups


def identify_radial_boundary(
    HC: Complex,
    R: float,
    center_axes: tuple[int, ...] = (0, 1),
    center: np.ndarray | None = None,
    tol: float = 1e-10,
) -> set:
    """Identify vertices on a cylindrical or spherical boundary at radius *R*.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    R : float
        Target radius.
    center_axes : tuple of int
        Which coordinate axes define the radial plane/space.
    center : array-like or None
        Center point (projected onto *center_axes*).  Defaults to origin.
    tol : float
        Tolerance for radius matching.

    Returns
    -------
    set
        Vertices whose distance from the center (along *center_axes*) is
        within *tol* of *R*.
    """
    if center is not None:
        c = np.asarray(center, dtype=float)
    else:
        c = np.zeros(len(center_axes))

    result: set = set()
    for v in HC.V:
        coords = np.array([v.x_a[ax] for ax in center_axes])
        dist = np.linalg.norm(coords - c)
        if abs(dist - R) < tol:
            result.add(v)
    return result


def identify_all_boundary(
    HC: Complex,
    lb: np.ndarray | list[float],
    ub: np.ndarray | list[float],
    tol: float = 1e-14,
) -> set:
    """Identify all vertices on the axis-aligned bounding box faces.

    Parameters
    ----------
    HC : Complex
        Simplicial complex.
    lb : array-like
        Lower bounds per axis.
    ub : array-like
        Upper bounds per axis.
    tol : float
        Position tolerance.

    Returns
    -------
    set
        All vertices on any bounding-box face.
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    bV: set = set()
    for v in HC.V:
        for ax in range(len(lb_arr)):
            if abs(v.x_a[ax] - lb_arr[ax]) < tol or abs(v.x_a[ax] - ub_arr[ax]) < tol:
                bV.add(v)
                break
    return bV
