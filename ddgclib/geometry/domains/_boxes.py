"""3D box (rectangular parallelepiped) domain builder."""
from __future__ import annotations

from hyperct import Complex

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._boundary_groups import (
    identify_face_groups,
    identify_all_boundary,
)


def box(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    refinement: int = 2,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    flow_axis: int = 0,
) -> DomainResult:
    """Build a filled 3D box (rectangular parallelepiped).

    Parameters
    ----------
    Lx, Ly, Lz : float
        Dimensions along x, y, z.
    refinement : int
        Number of ``refine_all()`` passes.
    origin : tuple of float
        Corner at ``(x0, y0, z0)``.
    flow_axis : int
        Primary flow direction (0, 1, or 2).  Determines which faces
        are labelled ``'inlet'`` and ``'outlet'``.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'``, ``'inlet'``, ``'outlet'``,
        ``'x_min'``, ``'x_max'``, ``'y_min'``, ``'y_max'``,
        ``'z_min'``, ``'z_max'``.
    """
    dims = [Lx, Ly, Lz]
    o = origin
    lb = [o[0], o[1], o[2]]
    ub = [o[0] + Lx, o[1] + Ly, o[2] + Lz]

    domain = [(lb[i], ub[i]) for i in range(3)]
    HC = Complex(3, domain=domain)
    HC.triangulate()
    for _ in range(refinement):
        HC.refine_all()

    bV = identify_all_boundary(HC, lb, ub)

    # Per-face groups
    axis_names = ['x', 'y', 'z']
    face_bounds: dict[str, tuple[int, float]] = {}
    for ax in range(3):
        face_bounds[f'{axis_names[ax]}_min'] = (ax, lb[ax])
        face_bounds[f'{axis_names[ax]}_max'] = (ax, ub[ax])

    groups = identify_face_groups(HC, face_bounds)

    # Flow-based groups
    groups['inlet'] = groups[f'{axis_names[flow_axis]}_min']
    groups['outlet'] = groups[f'{axis_names[flow_axis]}_max']

    # Walls = all faces except inlet and outlet
    wall_axes = [ax for ax in range(3) if ax != flow_axis]
    walls: set = set()
    for ax in wall_axes:
        walls |= groups[f'{axis_names[ax]}_min']
        walls |= groups[f'{axis_names[ax]}_max']
    groups['walls'] = walls

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=3,
        metadata={
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'volume': Lx * Ly * Lz,
            'flow_axis': flow_axis,
            'origin': origin,
            'lb': lb, 'ub': ub,
        },
    )
