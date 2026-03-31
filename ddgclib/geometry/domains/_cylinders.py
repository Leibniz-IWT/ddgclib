"""3D cylinder (tube) and pipe domain builders."""
from __future__ import annotations

import math

import numpy as np
from hyperct import Complex

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._projection import cube_to_disk
from ddgclib.geometry.domains._boundary_groups import (
    identify_face_groups,
    identify_radial_boundary,
    identify_all_boundary,
)
from ddgclib.geometry._complex_operations import extrude


def cylinder_volume(
    R: float = 0.5,
    L: float = 1.0,
    refinement: int = 2,
    flow_axis: int = 2,
    distr_law: str = "sinusoidal",
) -> DomainResult:
    """Build a filled 3D cylinder (tube) of radius *R* and length *L*.

    Constructed by creating a unit cube, projecting the cross-section
    to a disk via :func:`cube_to_disk`, then using :func:`extrude` to
    tile integer replica segments along the flow axis to reach length *L*.

    This avoids vertex-cache collisions that occur when scaling the axial
    dimension in-place via ``HC.V.move()``.

    Parameters
    ----------
    R : float
        Cylinder radius.
    L : float
        Cylinder length along *flow_axis*.
    refinement : int
        Number of ``refine_all()`` passes.
    flow_axis : int
        Axis along the cylinder length (0, 1, or 2).
    distr_law : str
        Distribution law for radial vertex placement.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'`` (cylindrical surface),
        ``'inlet'``, ``'outlet'``.
    """
    # Cross-section axes: the two axes that are NOT the flow axis.
    cross_axes = tuple(ax for ax in [0, 1, 2] if ax != flow_axis)

    # Build unit cube with [0, 1] along flow axis and [-0.5, 0.5] on
    # cross-section axes.  Using [0, 1] ensures the extruded mesh starts
    # at 0 along the flow axis.
    unit_domain = [None, None, None]
    for ax in cross_axes:
        unit_domain[ax] = (-0.5, 0.5)
    unit_domain[flow_axis] = (0.0, 1.0)

    HC_unit = Complex(3, domain=unit_domain)
    HC_unit.triangulate()
    for _ in range(refinement):
        HC_unit.refine_all()

    cube_to_disk(HC_unit, R, cross_axes=cross_axes, distr_law=distr_law)

    # Extrude to target length using integer replica segments.
    # extrude() uses ceil(L) segments, each scaled to L/ceil(L),
    # and merges at interfaces — no in-place z-scaling needed.
    HC = extrude(HC_unit, L, axis=flow_axis, cdist=1e-10)

    # Identify boundary groups on the final mesh.
    bV_wall = identify_radial_boundary(
        HC, R, center_axes=cross_axes, tol=1e-8,
    )

    axial_coords = [v.x_a[flow_axis] for v in HC.V]
    ax_min = min(axial_coords)
    ax_max = max(axial_coords)

    inlet_bV: set = set()
    outlet_bV: set = set()
    for v in HC.V:
        if abs(v.x_a[flow_axis] - ax_min) < 1e-10:
            inlet_bV.add(v)
        elif abs(v.x_a[flow_axis] - ax_max) < 1e-10:
            outlet_bV.add(v)

    bV = bV_wall | inlet_bV | outlet_bV
    groups = {
        'walls': bV_wall,
        'inlet': inlet_bV,
        'outlet': outlet_bV,
    }

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=3,
        metadata={
            'R': R, 'L': L,
            'volume': math.pi * R ** 2 * L,
            'flow_axis': flow_axis,
            'cross_axes': cross_axes,
        },
    )


def pipe(
    R: float = 0.5,
    L: float = 10.0,
    refinement: int = 2,
    flow_axis: int = 2,
    distr_law: str = "sinusoidal",
) -> DomainResult:
    """Build a long pipe domain by extruding a unit cylinder.

    For pipes where ``L`` is large relative to ``R``, this uses
    :func:`~ddgclib.geometry._complex_operations.extrude` to avoid
    poor aspect ratios from a single cube refinement.

    Parameters
    ----------
    R : float
        Pipe radius.
    L : float
        Pipe length along *flow_axis*.
    refinement : int
        Number of ``refine_all()`` passes for the unit cross-section.
    flow_axis : int
        Axis along the pipe length (0, 1, or 2).
    distr_law : str
        Distribution law for radial vertex placement.

    Returns
    -------
    DomainResult
        Boundary groups: ``'walls'`` (cylindrical surface),
        ``'inlet'``, ``'outlet'``.
    """
    # Build a unit-length cylinder (height=1 along flow_axis).
    cross_axes = tuple(ax for ax in [0, 1, 2] if ax != flow_axis)

    unit_domain = [None, None, None]
    for ax in cross_axes:
        unit_domain[ax] = (-0.5, 0.5)
    unit_domain[flow_axis] = (0.0, 1.0)

    HC_unit = Complex(3, domain=unit_domain)
    HC_unit.triangulate()
    for _ in range(refinement):
        HC_unit.refine_all()

    # Project cross-section to disk.
    cube_to_disk(HC_unit, R, cross_axes=cross_axes, distr_law=distr_law)

    # Extrude to target length along flow_axis.
    HC = extrude(HC_unit, L, axis=flow_axis, cdist=1e-10)

    # Identify boundary groups on the extruded mesh.
    # Cylindrical wall: vertices at radius R from the axis.
    bV_wall = identify_radial_boundary(
        HC, R, center_axes=cross_axes, tol=1e-8,
    )

    # Inlet/outlet: end caps at axial extremes.
    axial_coords = [v.x_a[flow_axis] for v in HC.V]
    ax_min = min(axial_coords)
    ax_max = max(axial_coords)

    inlet_bV: set = set()
    outlet_bV: set = set()
    for v in HC.V:
        if abs(v.x_a[flow_axis] - ax_min) < 1e-10:
            inlet_bV.add(v)
        elif abs(v.x_a[flow_axis] - ax_max) < 1e-10:
            outlet_bV.add(v)

    bV = bV_wall | inlet_bV | outlet_bV
    groups = {
        'walls': bV_wall,
        'inlet': inlet_bV,
        'outlet': outlet_bV,
    }

    for v in HC.V:
        v.boundary = v in bV

    return DomainResult(
        HC=HC,
        bV=bV,
        boundary_groups=groups,
        dim=3,
        metadata={
            'R': R, 'L': L,
            'volume': math.pi * R ** 2 * L,
            'flow_axis': flow_axis,
            'cross_axes': cross_axes,
        },
    )
