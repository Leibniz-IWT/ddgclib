"""Cube-to-shape projection engine for volume domain construction.

Generalizes the cube-to-cylinder pattern from
``cases_dynamic/Hagen_Poiseuile/src/_geometry.py`` into reusable primitives.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from hyperct import Complex


# ---------------------------------------------------------------------------
# Distribution laws
# ---------------------------------------------------------------------------
# Each law maps a normalized distance *d* (from the cube center, in the range
# [0, ~0.5*sqrt(2)] for a unit cube) and a target radius *R* to an effective
# radius *r_eff*.  The "sinusoidal" law provides smooth vertex clustering
# and is the recommended default (empirically validated in prior case studies).

DISTRIBUTION_LAWS: dict[str, Callable[[float, float], float]] = {
    "sinusoidal": lambda d, R: R * ((1 - np.cos(np.pi * d ** 0.5)) / 2),
    "linear": lambda d, R: R * 2.0 * d,
    "power": lambda d, R: (2.0 * d) ** 0.5 * R,
    "log": lambda d, R: R * (np.log(2.0 * d + 1) / np.log(2)),
}


def _get_law(distr_law: str | Callable) -> Callable[[float, float], float]:
    """Resolve a distribution law name or callable."""
    if callable(distr_law):
        return distr_law
    if distr_law not in DISTRIBUTION_LAWS:
        raise ValueError(
            f"Unknown distribution law {distr_law!r}. "
            f"Choose from {list(DISTRIBUTION_LAWS)} or pass a callable."
        )
    return DISTRIBUTION_LAWS[distr_law]


# ---------------------------------------------------------------------------
# cube_to_disk  (2D square → disk  OR  3D cube cross-section → cylinder)
# ---------------------------------------------------------------------------

def cube_to_disk(
    HC: Complex,
    R: float,
    cross_axes: tuple[int, int] = (0, 1),
    distr_law: str | Callable = "sinusoidal",
) -> tuple[set, set]:
    """Project the cross-section of a cube mesh to a disk of radius *R*.

    The mesh is modified **in-place** via ``HC.V.move()``.

    Two-phase projection (same algorithm as ``unit_cylinder``):

    1. **Side boundary vertices** — vertices on cube faces perpendicular to
       *cross_axes* — are projected to the full radius *R* at their polar
       angle ``theta = atan2(y, x)``.
    2. **Interior vertices** — their normalized distance *d* from the center
       is mapped through the distribution law to get ``r_eff``, then
       projected to ``(r_eff * cos(theta), r_eff * sin(theta))``.

    Parameters
    ----------
    HC : Complex
        Cube mesh (from ``Complex(n, domain)`` + ``triangulate`` +
        ``refine_all``).  Modified in-place.
    R : float
        Target disk radius.
    cross_axes : tuple of 2 ints
        Which two coordinate axes form the cross-section to project.
        Default ``(0, 1)`` projects the xy-plane.
    distr_law : str or callable
        Distribution law name (key in :data:`DISTRIBUTION_LAWS`) or a
        callable ``(d, R) -> r_eff``.

    Returns
    -------
    bV_wall : set
        Vertices that were projected to the disk boundary (radius *R*).
    bV_non_wall : set
        All other (interior) vertices.
    """
    law = _get_law(distr_law)
    ax0, ax1 = cross_axes

    # Detect the cube bounds along the cross-section axes.
    coords_0 = [v.x_a[ax0] for v in HC.V]
    coords_1 = [v.x_a[ax1] for v in HC.V]
    lb0, ub0 = min(coords_0), max(coords_0)
    lb1, ub1 = min(coords_1), max(coords_1)

    bV_wall: set = set()
    bV_non_wall: set = set()

    # Tag which vertices sit on a side face of the cube (perpendicular to
    # either cross-section axis).  Use a list snapshot because we'll move
    # vertices.
    side_flags: dict[int, bool] = {}
    for v in HC.V:
        on_side = (
            v.x_a[ax0] == lb0 or v.x_a[ax0] == ub0
            or v.x_a[ax1] == lb1 or v.x_a[ax1] == ub1
        )
        side_flags[id(v)] = on_side

    # Phase 1: project side boundary vertices to full radius R.
    for v in list(HC.V):
        if not side_flags[id(v)]:
            continue

        pos = v.x_a.copy()
        theta = math.atan2(pos[ax1], pos[ax0])
        pos[ax0] = R * math.cos(theta)
        pos[ax1] = R * math.sin(theta)
        HC.V.move(v, tuple(pos))
        bV_wall.add(v)

    # Phase 2: project interior vertices using the distribution law.
    for v in list(HC.V):
        if side_flags.get(id(v), False):
            continue

        pos = v.x_a.copy()
        d = math.hypot(pos[ax0], pos[ax1])

        if d < 1e-15:
            # Vertex at the exact center — leave it.
            bV_non_wall.add(v)
            continue

        r_eff = law(d, R)
        theta = math.atan2(pos[ax1], pos[ax0])
        pos[ax0] = r_eff * math.cos(theta)
        pos[ax1] = r_eff * math.sin(theta)
        HC.V.move(v, tuple(pos))
        bV_non_wall.add(v)

    return bV_wall, bV_non_wall


# ---------------------------------------------------------------------------
# cube_to_sphere  (3D cube → filled sphere)
# ---------------------------------------------------------------------------

def cube_to_sphere(
    HC: Complex,
    R: float,
    distr_law: str | Callable = "sinusoidal",
) -> tuple[set, set]:
    """Project a 3D cube mesh to a filled sphere of radius *R*.

    The mesh is modified **in-place**.

    Two-phase projection:

    1. **Surface boundary vertices** — vertices on any face of the cube —
       are projected to the sphere surface at radius *R*.
    2. **Interior vertices** — their 3D distance from the center is mapped
       through the distribution law, then projected radially.

    Parameters
    ----------
    HC : Complex
        3D cube mesh.  Modified in-place.
    R : float
        Target sphere radius.
    distr_law : str or callable
        Distribution law name or callable ``(d, R) -> r_eff``.

    Returns
    -------
    bV_wall : set
        Vertices on the sphere surface.
    bV_non_wall : set
        Interior vertices.
    """
    law = _get_law(distr_law)

    # Detect cube bounds along all three axes.
    bounds_lo = [min(v.x_a[ax] for v in HC.V) for ax in range(3)]
    bounds_hi = [max(v.x_a[ax] for v in HC.V) for ax in range(3)]

    bV_wall: set = set()
    bV_non_wall: set = set()

    # Tag surface vertices (on any face of the cube).
    surface_flags: dict[int, bool] = {}
    for v in HC.V:
        on_surface = any(
            v.x_a[ax] == bounds_lo[ax] or v.x_a[ax] == bounds_hi[ax]
            for ax in range(3)
        )
        surface_flags[id(v)] = on_surface

    # Phase 1: project surface vertices to sphere surface.
    for v in list(HC.V):
        if not surface_flags[id(v)]:
            continue

        pos = v.x_a.copy()
        d = np.linalg.norm(pos[:3])
        if d < 1e-15:
            bV_wall.add(v)
            continue

        # Normalize direction, scale to R.
        pos[:3] = pos[:3] / d * R
        HC.V.move(v, tuple(pos))
        bV_wall.add(v)

    # Phase 2: project interior vertices using distribution law.
    for v in list(HC.V):
        if surface_flags.get(id(v), False):
            continue

        pos = v.x_a.copy()
        d = np.linalg.norm(pos[:3])
        if d < 1e-15:
            bV_non_wall.add(v)
            continue

        r_eff = law(d, R)
        direction = pos[:3] / d
        pos[:3] = direction * r_eff
        HC.V.move(v, tuple(pos))
        bV_non_wall.add(v)

    return bV_wall, bV_non_wall
