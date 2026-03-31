"""2D rectangular domain builders."""
from __future__ import annotations

import numpy as np
from hyperct import Complex

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._boundary_groups import (
    identify_face_groups,
    identify_all_boundary,
)


def rectangle(
    L: float = 2.0,
    h: float = 1.0,
    refinement: int = 2,
    origin: tuple[float, float] = (0.0, 0.0),
    flow_axis: int = 0,
) -> DomainResult:
    """Build a filled 2D rectangular domain.

    Parameters
    ----------
    L : float
        Length along the flow axis.
    h : float
        Height (perpendicular to the flow axis).
    refinement : int
        Number of ``refine_all()`` passes.
    origin : tuple of float
        Lower-left corner ``(x0, y0)``.
    flow_axis : int
        Primary flow direction (0 or 1).  Determines which faces are
        labelled ``'inlet'`` and ``'outlet'``.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'``, ``'inlet'``, ``'outlet'``,
        ``'bottom_wall'``, ``'top_wall'``.
    """
    normal_axis = 1 - flow_axis
    o = origin

    # Bounds per axis: flow_axis spans [o, o+L], normal_axis spans [o, o+h]
    lb = [0.0, 0.0]
    ub = [0.0, 0.0]
    lb[flow_axis] = o[flow_axis]
    ub[flow_axis] = o[flow_axis] + L
    lb[normal_axis] = o[normal_axis]
    ub[normal_axis] = o[normal_axis] + h

    domain = [(lb[0], ub[0]), (lb[1], ub[1])]
    HC = Complex(2, domain=domain)
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV = identify_all_boundary(HC, lb, ub)

    # Named face groups
    groups = identify_face_groups(HC, {
        'inlet': (flow_axis, lb[flow_axis]),
        'outlet': (flow_axis, ub[flow_axis]),
        'bottom_wall': (normal_axis, lb[normal_axis]),
        'top_wall': (normal_axis, ub[normal_axis]),
    })
    groups['walls'] = groups['bottom_wall'] | groups['top_wall']

    # Tag boundaries
    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=2,
        metadata={
            'L': L, 'h': h, 'volume': L * h,
            'flow_axis': flow_axis, 'normal_axis': normal_axis,
            'origin': origin,
            'lb': lb, 'ub': ub,
        },
    )


def l_shape(
    L: float = 2.0,
    h: float = 1.0,
    notch_L: float = 1.0,
    notch_h: float = 0.5,
    refinement: int = 2,
    origin: tuple[float, float] = (0.0, 0.0),
) -> DomainResult:
    """Build an L-shaped 2D domain (rectangle with rectangular notch removed).

    The notch is removed from the upper-right corner::

        +----------+
        |          |
        |   +------+
        |   | notch
        +---+

    Parameters
    ----------
    L : float
        Full rectangle length.
    h : float
        Full rectangle height.
    notch_L : float
        Notch length (from right edge).
    notch_h : float
        Notch height (from bottom edge).
    refinement : int
        Number of ``refine_all()`` passes.
    origin : tuple of float
        Lower-left corner.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'`` (all boundary), ``'outer'``, ``'notch'``.
    """
    o = origin
    lb = [o[0], o[1]]
    ub = [o[0] + L, o[1] + h]

    domain = [(lb[0], ub[0]), (lb[1], ub[1])]
    HC = Complex(2, domain=domain)
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    # Remove vertices inside the notch region
    notch_x_min = ub[0] - notch_L
    notch_y_max = lb[1] + notch_h
    to_remove = [
        v for v in HC.V
        if v.x_a[0] > notch_x_min + 1e-14 and v.x_a[1] < notch_y_max - 1e-14
    ]
    for v in to_remove:
        HC.V.remove(v)

    # After removal, boundary vertices are those with reduced connectivity
    # or on the original bounding box edges or on the notch edges.
    bV: set = set()
    notch_bV: set = set()
    outer_bV: set = set()

    for v in HC.V:
        on_outer = (
            abs(v.x_a[0] - lb[0]) < 1e-14
            or abs(v.x_a[0] - ub[0]) < 1e-14
            or abs(v.x_a[1] - lb[1]) < 1e-14
            or abs(v.x_a[1] - ub[1]) < 1e-14
        )
        on_notch_x = abs(v.x_a[0] - notch_x_min) < 1e-14 and v.x_a[1] < notch_y_max + 1e-14
        on_notch_y = abs(v.x_a[1] - notch_y_max) < 1e-14 and v.x_a[0] > notch_x_min - 1e-14

        if on_outer:
            bV.add(v)
            outer_bV.add(v)
        if on_notch_x or on_notch_y:
            bV.add(v)
            notch_bV.add(v)

    groups = {
        'walls': bV,
        'outer': outer_bV,
        'notch': notch_bV,
    }

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=2,
        metadata={
            'L': L, 'h': h,
            'notch_L': notch_L, 'notch_h': notch_h,
            'origin': origin,
        },
    )
