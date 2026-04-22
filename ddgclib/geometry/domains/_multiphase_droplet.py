"""Multiphase droplet-in-box domain builders for 2D and 3D.

Constructs a two-phase domain with a circular (2D) or spherical (3D)
droplet embedded in a rectangular/cubic outer domain.

The droplet mesh is built from a ``disk()`` or ``ball()`` projection,
combined with an outer rectangular/box mesh after removing overlapping
outer vertices.  Phase IDs are assigned based on distance from center.
"""
from __future__ import annotations

import math

import numpy as np
from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.geometry.domains._result import DomainResult
from ddgclib.geometry.domains._disks import disk
from ddgclib.geometry.domains._spheres import ball
from ddgclib.geometry.domains._rectangles import rectangle
from ddgclib.geometry.domains._boxes import box


def _estimate_edge_length(HC, dim, stat: str = "median"):
    """Estimate a representative edge length across a mesh.

    Parameters
    ----------
    stat : {'median', 'min'}
        ``'median'`` (default) is appropriate for sizing interior
        refinement; ``'min'`` is needed when a generated ring must not
        land inside an interface arc — a median edge length on a
        non-uniform radial distribution can exceed the smallest arc
        spacing and produce ring vertices coincident with interface
        vertices.
    """
    lengths = []
    for v in HC.V:
        for nb in v.nn:
            lengths.append(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        if len(lengths) > 50:
            break
    if not lengths:
        return 0.0
    if stat == "min":
        return float(np.min(lengths))
    return float(np.median(lengths))


def _interface_min_edge(boundary_verts, dim):
    """Minimum chord length between adjacent interface vertices."""
    if len(boundary_verts) < 2:
        return 0.0
    pts = np.array([v.x_a[:dim] for v in boundary_verts])
    # Sort by polar angle so that consecutive entries are neighbours
    center = pts.mean(axis=0)
    rel = pts - center
    theta = np.arctan2(rel[:, 1], rel[:, 0]) if dim >= 2 else np.zeros(len(pts))
    order = np.argsort(theta)
    pts = pts[order]
    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def _build_combined_mesh(positions, phases, dim):
    """Create a unified Complex from combined positions and triangulate.

    Deduplicates near-coincident vertices (within ``_MERGE_TOL``) by
    snapping coordinates to a grid, then builds the Delaunay
    triangulation.  Snapping ensures that the ``hyperct`` Complex (which
    uses tuple keys) treats near-duplicates as the same vertex.

    Returns the Complex with phase labels set.
    """
    coords = np.array(positions)

    # Snap coordinates to eliminate floating-point noise from
    # cube-to-disk/sphere projections (differences ~1e-18).
    # Rounding to 10 decimals gives bit-identical floats for
    # near-duplicates while preserving geometry.
    coords = np.round(coords, decimals=10)

    # Deduplicate: keep the first occurrence of each unique position
    _, unique_idx = np.unique(
        coords, axis=0, return_index=True,
    )
    unique_idx = np.sort(unique_idx)
    coords = coords[unique_idx]
    phases_clean = [phases[i] for i in unique_idx]

    # Create complex
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    eps = 1e-12
    domain = [(float(mins[i] - eps), float(maxs[i] + eps)) for i in range(dim)]
    HC = Complex(dim, domain=domain)

    for i, pos in enumerate(coords):
        v = HC.V[tuple(pos)]
        v.phase = phases_clean[i]

    # Delaunay triangulation — connect edges and cache simplex list for
    # the simplex-aware 3D dual construction.  The sharp droplet
    # interface produces fixed-interface + denser-bulk geometry that
    # generates boundary slivers (same pattern as domain walls), which
    # the simplex-aware path handles correctly.  See
    # docs/3d_simplex_aware_dual_fix.md.
    from ddgclib.geometry import (
        connect_and_cache_simplices, invalidate_simplex_cache,
    )
    verts = list(HC.V)
    if len(verts) < dim + 1:
        raise ValueError(f"Need at least {dim + 1} vertices, got {len(verts)}")

    tri_coords = np.array([v.x_a[:dim] for v in verts])
    connect_and_cache_simplices(HC, verts, dim, coords=tri_coords)

    # Verify no isolated vertices remain; invalidate the cache if any
    # vertex was removed (will be rebuilt on the next retopologization).
    isolated = [v for v in HC.V if len(v.nn) == 0]
    if isolated:
        for v in isolated:
            HC.V.remove(v)
        invalidate_simplex_cache(HC)

    return HC


def _setup_phases_and_duals(HC, R, center_arr, dim):
    """Tag boundaries, compute duals, label simplices, extract interface.

    Uses the primal-subcomplex model: top-simplices (triangles in 2D,
    tets in 3D) are labelled phase 0 (outer) or phase 1 (droplet) by a
    centroid-inside-circle/sphere test.  The interface subcomplex is
    extracted and closure-validated.

    Returns ``(bV_walls, interface_vertices, mps)`` where ``mps`` is a
    :class:`MultiphaseSystem` with ``simplex_phase`` populated.
    """
    from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
    from ddgclib.eos import TaitMurnaghan

    # Identify domain boundary walls (outer box faces, not droplet interface)
    dV = HC.boundary()
    bV_walls = set()
    for v in dV:
        dist = np.linalg.norm(v.x_a[:dim] - center_arr)
        if dist > R + 1e-10:
            bV_walls.add(v)

    bV = bV_walls
    for v in HC.V:
        v.boundary = v in bV

    # Compute duals
    compute_vd(HC, method="barycentric")

    # Cache dual volumes
    from ddgclib.operators.stress import cache_dual_volumes
    cache_dual_volumes(HC, dim)

    # Label top-simplices by centroid-inside test and extract the
    # interface subcomplex (primal-subcomplex model).
    R2 = R * R

    def _phase_criterion(centroid):
        d2 = float(np.dot(centroid[:dim] - center_arr, centroid[:dim] - center_arr))
        return 1 if d2 < R2 else 0

    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000,
                            name="outer"),
            PhaseProperties(eos=TaitMurnaghan(rho0=1000), mu=0.1, rho0=1000,
                            name="droplet"),
        ],
    )
    mps.assign_simplex_phases(HC, dim, criterion_fn=_phase_criterion)
    interface = mps.identify_interface_from_subcomplex(HC, dim)

    return bV_walls, interface, mps


def droplet_in_box_2d(
    R: float = 0.01,
    L: float = 0.05,
    refinement_outer: int = 2,
    refinement_droplet: int = 2,
    distr_law: str = "sinusoidal",
    center: tuple[float, float] = (0.0, 0.0),
) -> DomainResult:
    """Build 2D domain with a circular droplet in a rectangular box.

    Phase 0 = outer fluid, Phase 1 = droplet (inside circle of radius R).

    Parameters
    ----------
    R : float
        Droplet radius.
    L : float
        Half-side of the outer square domain (domain is [-L, L]^2).
    refinement_outer : int
        Refinement passes for outer domain.
    refinement_droplet : int
        Refinement passes for droplet domain.
    distr_law : str
        Distribution law for radial vertex placement in the droplet.
    center : tuple
        Center of the droplet.

    Returns
    -------
    DomainResult
        With ``boundary_groups['walls']`` (outer box walls) and
        metadata including ``'interface_vertices'``, ``'R'``, ``'L'``.
    """
    cx, cy = center
    dim = 2

    # -- Step 1: Create outer box --
    outer_result = rectangle(L=2 * L, h=2 * L, refinement=refinement_outer,
                             flow_axis=0)
    HC_outer = outer_result.HC

    # Shift to center at droplet
    for v in list(HC_outer.V):
        pos = v.x_a.copy()
        pos[0] += cx - L
        pos[1] += cy - L
        HC_outer.V.move(v, tuple(pos))

    # -- Step 2: Create droplet domain --
    drop_result = disk(R=R, center=center, refinement=refinement_droplet,
                       distr_law=distr_law)
    HC_drop = drop_result.HC

    # -- Step 3: Collect vertex positions --
    # Remove outer vertices strictly inside the droplet (r < R).
    # Add a ring of outer-phase vertices just outside R so that
    # Delaunay produces quality triangles at the phase interface
    # (without this, there is a gap between the droplet boundary
    # and the nearest outer vertex, producing long thin triangles).
    # All coordinates are rounded to 10 decimals so that near-duplicate
    # positions map to the same tuple key in the Complex.
    positions = []
    phases = []

    for v in HC_outer.V:
        pos = np.round(v.x_a[:dim], decimals=10)
        dist = math.hypot(pos[0] - cx, pos[1] - cy)
        if dist > R:
            positions.append(pos)
            phases.append(0)

    for v in HC_drop.V:
        positions.append(np.round(v.x_a[:dim], decimals=10))
        phases.append(1)

    # Add an outer-phase ring just outside the droplet boundary.
    # Match the angular density of the droplet boundary vertices.
    boundary_verts = [v for v in HC_drop.V
                      if abs(np.linalg.norm(v.x_a[:dim]) - R) < R * 0.01]
    n_ring = max(len(boundary_verts), 16)
    h_drop = _estimate_edge_length(HC_drop, dim)
    ring_R = R + h_drop  # one edge length outside
    for i in range(n_ring):
        theta = 2 * math.pi * i / n_ring
        pos = np.round(np.array([
            cx + ring_R * math.cos(theta),
            cy + ring_R * math.sin(theta),
        ]), decimals=10)
        positions.append(pos)
        phases.append(0)

    if len(positions) < 3:
        raise ValueError("Not enough vertices to build mesh")

    # -- Step 4: Build unified mesh --
    HC = _build_combined_mesh(positions, phases, dim)

    # -- Step 5: Boundaries, duals, interface --
    center_arr = np.array([cx, cy])
    bV_walls, interface, mps = _setup_phases_and_duals(HC, R, center_arr, dim)

    groups = {'walls': bV_walls, 'interface': interface}

    return DomainResult(
        HC=HC,
        bV=bV_walls,
        boundary_groups=groups,
        dim=dim,
        metadata={
            'R': R,
            'L': L,
            'center': center,
            'volume_total': (2 * L) ** 2,
            'volume_droplet': math.pi * R ** 2,
            'volume_outer': (2 * L) ** 2 - math.pi * R ** 2,
            'interface_vertices': interface,
            'simplex_phase': mps.simplex_phase,
            'interface_subcomplex': None,  # data lives on HC attributes
            'mps': mps,
        },
    )


def droplet_in_box_3d(
    R: float = 0.01,
    L: float = 0.05,
    refinement_outer: int = 1,
    refinement_droplet: int = 1,
    distr_law: str = "sinusoidal",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DomainResult:
    """Build 3D domain with a spherical droplet in a cubic box.

    Phase 0 = outer fluid, Phase 1 = droplet (inside sphere of radius R).

    Parameters
    ----------
    R : float
        Droplet radius.
    L : float
        Half-side of the outer cubic domain (domain is [-L, L]^3).
    refinement_outer : int
        Refinement passes for outer domain.
    refinement_droplet : int
        Refinement passes for droplet domain.
    distr_law : str
        Distribution law for radial vertex placement in the droplet.
    center : tuple
        Center of the droplet.

    Returns
    -------
    DomainResult
        With ``boundary_groups['walls']`` and ``'interface'``.
    """
    cx, cy, cz = center
    dim = 3

    # -- Step 1: Create outer box --
    outer_result = box(Lx=2 * L, Ly=2 * L, Lz=2 * L,
                       refinement=refinement_outer)
    HC_outer = outer_result.HC

    for v in list(HC_outer.V):
        pos = v.x_a.copy()
        pos[0] += cx - L
        pos[1] += cy - L
        pos[2] += cz - L
        HC_outer.V.move(v, tuple(pos))

    # -- Step 2: Create droplet domain --
    drop_result = ball(R=R, center=center, refinement=refinement_droplet,
                       distr_law=distr_law)
    HC_drop = drop_result.HC

    # -- Step 3: Collect vertex positions (same strategy as 2D) --
    positions = []
    phases = []
    center_3d = np.array([cx, cy, cz])

    for v in HC_outer.V:
        pos = np.round(v.x_a[:dim], decimals=10)
        dist = np.linalg.norm(pos - center_3d)
        if dist > R:
            positions.append(pos)
            phases.append(0)

    for v in HC_drop.V:
        positions.append(np.round(v.x_a[:dim], decimals=10))
        phases.append(1)

    # Add outer-phase shell just outside the droplet boundary
    boundary_verts = [v for v in HC_drop.V
                      if abs(np.linalg.norm(v.x_a[:dim]) - R) < R * 0.01]
    h_drop = _estimate_edge_length(HC_drop, dim)
    shell_R = R + h_drop
    for v in boundary_verts:
        # Project boundary vertex outward by one edge length
        direction = v.x_a[:dim] - center_3d
        d = np.linalg.norm(direction)
        if d < 1e-30:
            continue
        pos = np.round(center_3d + direction * (shell_R / d), decimals=10)
        positions.append(pos)
        phases.append(0)

    if len(positions) < 4:
        raise ValueError("Not enough vertices to build 3D mesh")

    # -- Step 4: Build unified mesh --
    HC = _build_combined_mesh(positions, phases, dim)

    # -- Step 5: Boundaries, duals, interface --
    bV_walls, interface, mps = _setup_phases_and_duals(HC, R, center_3d, dim)

    groups = {'walls': bV_walls, 'interface': interface}

    return DomainResult(
        HC=HC,
        bV=bV_walls,
        boundary_groups=groups,
        dim=dim,
        metadata={
            'R': R,
            'L': L,
            'center': center,
            'volume_total': (2 * L) ** 3,
            'volume_droplet': (4 / 3) * math.pi * R ** 3,
            'volume_outer': (2 * L) ** 3 - (4 / 3) * math.pi * R ** 3,
            'interface_vertices': interface,
            'simplex_phase': mps.simplex_phase,
            'interface_subcomplex': None,  # data lives on HC attributes
            'mps': mps,
        },
    )
