"""Periodic domain builders (rectangle, box) with periodic axis metadata."""
from __future__ import annotations

from ddgclib.geometry.domains._rectangles import rectangle
from ddgclib.geometry.domains._boxes import box
from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._boundary_groups import identify_face_groups


def _strip_periodic_faces(result: DomainResult, periodic_axes: list[int],
                          domain_bounds: list[tuple[float, float]],
                          tol: float = 1e-14) -> None:
    """Remove periodic-face vertices from bV and boundary_groups (in-place)."""
    axis_names = ['x', 'y', 'z']
    periodic_verts: set = set()
    for v in result.HC.V:
        for ax in periodic_axes:
            lb, ub = domain_bounds[ax]
            if abs(v.x_a[ax] - lb) < tol or abs(v.x_a[ax] - ub) < tol:
                periodic_verts.add(v)
                break

    # Remove from bV
    result.bV -= periodic_verts

    # Remove from boundary groups
    for key in list(result.boundary_groups):
        result.boundary_groups[key] -= periodic_verts
        # Remove empty groups
        if not result.boundary_groups[key]:
            del result.boundary_groups[key]

    # Re-tag boundaries
    for v in result.HC.V:
        v.boundary = v in result.bV


def periodic_rectangle(
    L: float = 2.0,
    h: float = 1.0,
    refinement: int = 2,
    origin: tuple[float, float] = (0.0, 0.0),
    periodic_axes: list[int] | None = None,
    flow_axis: int = 0,
) -> DomainResult:
    """Build a 2D rectangle with periodic boundary conditions.

    Wraps :func:`rectangle` and removes periodic-face vertices from
    the boundary set.

    Parameters
    ----------
    L : float
        Length along axis 0.
    h : float
        Height along axis 1.
    refinement : int
        Number of ``refine_all()`` passes.
    origin : tuple of float
        Lower-left corner.
    periodic_axes : list[int] or None
        Axes to make periodic (e.g. ``[0]`` for x-periodic).
    flow_axis : int
        Primary flow direction for inlet/outlet labelling.

    Returns
    -------
    DomainResult
        With ``metadata['periodic_axes']`` and ``metadata['domain_bounds']``.
    """
    if periodic_axes is None:
        periodic_axes = []

    result = rectangle(L=L, h=h, refinement=refinement, origin=origin,
                       flow_axis=flow_axis)

    domain_bounds = [
        (result.metadata['lb'][0], result.metadata['ub'][0]),
        (result.metadata['lb'][1], result.metadata['ub'][1]),
    ]

    if periodic_axes:
        _strip_periodic_faces(result, periodic_axes, domain_bounds)

    result.metadata['periodic_axes'] = periodic_axes
    result.metadata['domain_bounds'] = domain_bounds

    return result


def periodic_box(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    refinement: int = 2,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic_axes: list[int] | None = None,
    flow_axis: int = 0,
) -> DomainResult:
    """Build a 3D box with periodic boundary conditions.

    Wraps :func:`box` and removes periodic-face vertices from
    the boundary set.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Dimensions along x, y, z.
    refinement : int
        Number of ``refine_all()`` passes.
    origin : tuple of float
        Corner at ``(x0, y0, z0)``.
    periodic_axes : list[int] or None
        Axes to make periodic (e.g. ``[0, 1]``).
    flow_axis : int
        Primary flow direction for inlet/outlet labelling.

    Returns
    -------
    DomainResult
        With ``metadata['periodic_axes']`` and ``metadata['domain_bounds']``.
    """
    if periodic_axes is None:
        periodic_axes = []

    result = box(Lx=Lx, Ly=Ly, Lz=Lz, refinement=refinement, origin=origin,
                 flow_axis=flow_axis)

    domain_bounds = [
        (result.metadata['lb'][0], result.metadata['ub'][0]),
        (result.metadata['lb'][1], result.metadata['ub'][1]),
        (result.metadata['lb'][2], result.metadata['ub'][2]),
    ]

    if periodic_axes:
        _strip_periodic_faces(result, periodic_axes, domain_bounds)

    result.metadata['periodic_axes'] = periodic_axes
    result.metadata['domain_bounds'] = domain_bounds

    return result
