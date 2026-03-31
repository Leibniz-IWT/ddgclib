"""2D disk (filled circle) domain builders."""
from __future__ import annotations

import math

import numpy as np
from hyperct import Complex

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._projection import cube_to_disk
from ddgclib.geometry.domains._boundary_groups import identify_radial_boundary


def disk(
    R: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    refinement: int = 2,
    distr_law: str = "sinusoidal",
) -> DomainResult:
    """Build a filled 2D disk of radius *R*.

    Constructed by creating a 2D square ``[-0.5, 0.5]^2``, refining, then
    projecting to polar coordinates via :func:`cube_to_disk`.

    Parameters
    ----------
    R : float
        Disk radius.
    center : tuple of float
        Center ``(cx, cy)``.
    refinement : int
        Number of ``refine_all()`` passes.
    distr_law : str
        Distribution law for radial vertex placement.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'`` (the circumference).
    """
    HC = Complex(2, domain=[(-0.5, 0.5), (-0.5, 0.5)])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV_wall, _ = cube_to_disk(HC, R, cross_axes=(0, 1), distr_law=distr_law)

    # Shift to requested center
    if center != (0.0, 0.0):
        cx, cy = center
        for v in list(HC.V):
            pos = v.x_a.copy()
            pos[0] += cx
            pos[1] += cy
            HC.V.move(v, tuple(pos))

    bV = bV_wall
    groups = {'walls': bV_wall}

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=2,
        metadata={
            'R': R,
            'center': center,
            'volume': math.pi * R ** 2,
        },
    )


def annulus(
    R_outer: float = 1.0,
    R_inner: float = 0.3,
    center: tuple[float, float] = (0.0, 0.0),
    refinement: int = 2,
    distr_law: str = "sinusoidal",
) -> DomainResult:
    """Build a filled 2D annulus (ring) domain.

    Constructed by creating a disk and removing interior vertices within
    *R_inner*.

    Parameters
    ----------
    R_outer : float
        Outer radius.
    R_inner : float
        Inner radius (must be < R_outer).
    center : tuple of float
        Center ``(cx, cy)``.
    refinement : int
        Number of ``refine_all()`` passes.
    distr_law : str
        Distribution law for radial vertex placement.

    Returns
    -------
    DomainResult
        Boundary groups: ``'outer_wall'``, ``'inner_wall'``.
    """
    if R_inner >= R_outer:
        raise ValueError(f"R_inner ({R_inner}) must be < R_outer ({R_outer})")

    # Build full disk first
    HC = Complex(2, domain=[(-0.5, 0.5), (-0.5, 0.5)])
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV_wall, _ = cube_to_disk(HC, R_outer, cross_axes=(0, 1), distr_law=distr_law)

    # Shift to requested center
    cx, cy = center
    if center != (0.0, 0.0):
        for v in list(HC.V):
            pos = v.x_a.copy()
            pos[0] += cx
            pos[1] += cy
            HC.V.move(v, tuple(pos))

    # Remove vertices strictly inside R_inner
    to_remove = []
    for v in HC.V:
        dist = math.hypot(v.x_a[0] - cx, v.x_a[1] - cy)
        if dist < R_inner - 1e-10:
            to_remove.append(v)
    for v in to_remove:
        HC.V.remove(v)

    # Identify boundary groups
    outer_wall = identify_radial_boundary(HC, R_outer, center_axes=(0, 1),
                                          center=np.array([cx, cy]), tol=1e-10)
    inner_wall: set = set()
    for v in HC.V:
        dist = math.hypot(v.x_a[0] - cx, v.x_a[1] - cy)
        if abs(dist - R_inner) < R_inner * 0.3:
            # After removal, the innermost remaining vertices form the inner wall.
            # Use a generous tolerance since removal doesn't leave vertices
            # exactly at R_inner.
            is_innermost = True
            for nb in v.nn:
                nb_dist = math.hypot(nb.x_a[0] - cx, nb.x_a[1] - cy)
                if nb_dist < dist - 1e-12:
                    is_innermost = False
                    break
            if is_innermost and dist < R_inner + (R_outer - R_inner) * 0.3:
                inner_wall.add(v)

    bV = outer_wall | inner_wall
    groups = {'outer_wall': outer_wall, 'inner_wall': inner_wall}

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=2,
        metadata={
            'R_outer': R_outer,
            'R_inner': R_inner,
            'center': center,
            'volume': math.pi * (R_outer ** 2 - R_inner ** 2),
        },
    )
