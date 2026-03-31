"""3D ball (filled sphere) domain builder."""
from __future__ import annotations

import math

import numpy as np
from hyperct import Complex

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._projection import cube_to_sphere


def ball(
    R: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    refinement: int = 2,
    distr_law: str = "sinusoidal",
) -> DomainResult:
    """Build a filled 3D ball (sphere) of radius *R*.

    Constructed by projecting a 3D cube ``[-0.5, 0.5]^3`` to spherical
    coordinates via :func:`cube_to_sphere`.

    Parameters
    ----------
    R : float
        Sphere radius.
    center : tuple of float
        Center ``(cx, cy, cz)``.
    refinement : int
        Number of ``refine_all()`` passes.
    distr_law : str
        Distribution law for radial vertex placement.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'`` (the sphere surface).
    """
    HC = Complex(3, domain=[(-0.5, 0.5)] * 3)
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV_wall, _ = cube_to_sphere(HC, R, distr_law=distr_law)

    # Shift to requested center.
    if center != (0.0, 0.0, 0.0):
        cx, cy, cz = center
        for v in list(HC.V):
            pos = v.x_a.copy()
            pos[0] += cx
            pos[1] += cy
            pos[2] += cz
            HC.V.move(v, tuple(pos))

    bV = bV_wall
    groups = {'walls': bV_wall}

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=3,
        metadata={
            'R': R,
            'center': center,
            'volume': (4.0 / 3.0) * math.pi * R ** 3,
        },
    )
