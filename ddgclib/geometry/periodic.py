"""Periodic boundary condition utilities for ghost-cell Delaunay triangulation.

Provides smart ghost-cell replication: only vertices in a thin buffer band
near periodic faces are duplicated.  After Delaunay triangulation on the
extended point set, ghost indices are resolved back to real vertices so that
the resulting connectivity wraps across periodic boundaries.

The ``retopologize_periodic`` function is a drop-in replacement for
``_retopologize`` in the dynamic integrators when ``periodic_axes`` is set.

Backend
-------
Default: ``"ghost"`` — pure-Python + SciPy ghost-cell approach.
Future:  ``"cgal"``  — true periodic Delaunay via ``cgal4py`` (not yet implemented).
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Ghost band width
# ---------------------------------------------------------------------------

def ghost_band_width(HC, dim: int, periodic_axes: list[int]) -> float:
    """Auto-compute buffer width as 2x max edge length along periodic axes.

    Parameters
    ----------
    HC : Complex
    dim : int
    periodic_axes : list[int]

    Returns
    -------
    float
        Recommended band width for ghost vertex creation.
    """
    max_edge = 0.0
    for v in HC.V:
        for nb in v.nn:
            diff = nb.x_a[:dim] - v.x_a[:dim]
            length = float(np.linalg.norm(diff))
            if length > max_edge:
                max_edge = length
    if max_edge == 0.0:
        # Fallback: estimate from domain bounds
        max_edge = 1.0
    return 2.0 * max_edge


# ---------------------------------------------------------------------------
# Ghost vertex creation
# ---------------------------------------------------------------------------

def _shift_directions(periodic_axes: list[int], dim: int) -> list[tuple[int, ...]]:
    """Generate all non-zero shift direction tuples for periodic axes.

    For N periodic axes, returns 3^N - 1 directions (excludes the zero shift).
    Each direction is a dim-length tuple with values in {-1, 0, +1} at periodic
    axes and 0 elsewhere.
    """
    # Build shift components for periodic axes only
    per_ax_options = {ax: [-1, 0, 1] for ax in periodic_axes}

    directions: list[tuple[int, ...]] = []
    for combo in product(*[per_ax_options[ax] for ax in periodic_axes]):
        if all(c == 0 for c in combo):
            continue  # skip zero shift
        d = [0] * dim
        for ax, c in zip(periodic_axes, combo):
            d[ax] = c
        directions.append(tuple(d))
    return directions


def create_ghost_vertices(
    HC,
    periodic_axes: list[int],
    domain_bounds: list[tuple[float, float]],
    band_width: float,
    dim: int,
) -> tuple[np.ndarray, list, dict[int, int]]:
    """Duplicate buffer-band vertices as ghosts at periodic image positions.

    Ghost vertices exist only in the returned numpy array — they are NEVER
    added to ``HC.V``.

    Parameters
    ----------
    HC : Complex
    periodic_axes : list[int]
        Axes along which the domain is periodic.
    domain_bounds : list[tuple[float, float]]
        ``[(lb_0, ub_0), (lb_1, ub_1), ...]`` for each axis.
    band_width : float
        Width of the buffer band near each periodic face.
    dim : int

    Returns
    -------
    all_coords : ndarray, shape (n_real + n_ghost, dim)
    real_verts : list of vertex objects (length n_real)
    ghost_to_real : dict mapping ghost index -> real index in ``real_verts``
    """
    real_verts = list(HC.V)
    n_real = len(real_verts)
    real_coords = np.array([v.x_a[:dim] for v in real_verts])

    periods = {ax: domain_bounds[ax][1] - domain_bounds[ax][0]
               for ax in periodic_axes}

    directions = _shift_directions(periodic_axes, dim)

    ghost_coords: list[np.ndarray] = []
    ghost_to_real: dict[int, int] = {}

    for direction in directions:
        # Build shift vector
        shift = np.zeros(dim)
        for ax in periodic_axes:
            shift[ax] = direction[ax] * periods[ax]

        # Determine which vertices are in the band for this direction
        for i, v in enumerate(real_verts):
            in_band = True
            for ax in periodic_axes:
                if direction[ax] == 0:
                    continue
                lb, ub = domain_bounds[ax]
                if direction[ax] == +1:
                    # Ghost goes to high side -> real vertex must be near low side
                    if v.x_a[ax] > lb + band_width:
                        in_band = False
                        break
                else:  # direction[ax] == -1
                    # Ghost goes to low side -> real vertex must be near high side
                    if v.x_a[ax] < ub - band_width:
                        in_band = False
                        break
            if in_band:
                ghost_pos = real_coords[i] + shift
                # Skip ghosts that land on top of existing real vertices
                dists = np.linalg.norm(real_coords - ghost_pos, axis=1)
                if np.min(dists) < 1e-10:
                    continue
                ghost_idx = n_real + len(ghost_coords)
                ghost_coords.append(ghost_pos)
                ghost_to_real[ghost_idx] = i

    if ghost_coords:
        all_coords = np.vstack([real_coords, np.array(ghost_coords)])
    else:
        all_coords = real_coords

    return all_coords, real_verts, ghost_to_real


# ---------------------------------------------------------------------------
# Delaunay with ghost resolution
# ---------------------------------------------------------------------------

def delaunay_with_ghosts(
    all_coords: np.ndarray,
    n_real: int,
    ghost_to_real: dict[int, int],
    dim: int,
) -> list[tuple[int, ...]]:
    """Run Delaunay on real+ghost coords and resolve ghost indices.

    Parameters
    ----------
    all_coords : ndarray, shape (n_total, dim)
    n_real : int
        Number of real vertices (first n_real rows of all_coords).
    ghost_to_real : dict
        Maps ghost index -> real vertex index.
    dim : int

    Returns
    -------
    list of tuple[int, ...]
        Deduplicated simplex tuples with all indices in ``[0, n_real)``.
    """
    tri = Delaunay(all_coords[:, :dim])

    resolved: set[tuple[int, ...]] = set()
    for simplex in tri.simplices:
        # Replace ghost indices with their real counterparts
        mapped = []
        for idx in simplex:
            mapped.append(ghost_to_real.get(int(idx), int(idx)))

        # Sort for deduplication
        mapped_sorted = tuple(sorted(mapped))

        # Skip degenerate simplices (same real vertex appears twice)
        if len(set(mapped_sorted)) < len(mapped_sorted):
            continue

        resolved.add(mapped_sorted)

    return list(resolved)


# ---------------------------------------------------------------------------
# Position wrapping
# ---------------------------------------------------------------------------

def wrap_positions(
    HC,
    periodic_axes: list[int],
    domain_bounds: list[tuple[float, float]],
) -> None:
    """Wrap vertices that drifted past periodic boundaries back into domain.

    Uses ``HC.V.move()`` to update the vertex cache key.
    """
    for v in list(HC.V):
        x_new = list(v.x)
        moved = False
        for ax in periodic_axes:
            lb, ub = domain_bounds[ax]
            period = ub - lb
            x = x_new[ax]
            # Use tolerance to avoid wrapping vertices exactly on the boundary
            tol = period * 1e-12
            if x < lb - tol or x > ub + tol:
                x_new[ax] = lb + (x - lb) % period
                moved = True
        if moved:
            HC.V.move(v, tuple(x_new))


# ---------------------------------------------------------------------------
# Periodic face identification
# ---------------------------------------------------------------------------

def _identify_periodic_face_verts(
    HC,
    periodic_axes: list[int],
    domain_bounds: list[tuple[float, float]],
    tol: float = 1e-14,
) -> set:
    """Return the set of vertices lying on any periodic boundary face."""
    periodic_verts: set = set()
    for v in HC.V:
        for ax in periodic_axes:
            lb, ub = domain_bounds[ax]
            if abs(v.x_a[ax] - lb) < tol or abs(v.x_a[ax] - ub) < tol:
                periodic_verts.add(v)
                break
    return periodic_verts


def _fixup_periodic_duals(HC, dim, periodic_axes, domain_bounds):
    """Recompute dual vertex positions for triangles crossing periodic boundaries.

    After compute_vd, dual vertices of periodic triangles have wrong
    barycenters because compute_vd uses raw (unwrapped) coordinates.
    This post-processing step identifies affected duals and fixes them
    using minimum-image coordinates.
    """
    periods = {ax: domain_bounds[ax][1] - domain_bounds[ax][0]
               for ax in periodic_axes}

    def min_image(x_ref, x_other):
        """Return x_other wrapped to be closest to x_ref."""
        result = x_other.copy()
        for ax in periodic_axes:
            p = periods[ax]
            delta = result[ax] - x_ref[ax]
            result[ax] -= round(delta / p) * p
        return result

    # For each interior edge (v1, v2) where the raw distance differs
    # from the minimum-image distance, the shared duals need fixing.
    processed_duals = set()
    for v1 in HC.V:
        for v2 in v1.nn:
            if id(v1) >= id(v2):
                continue
            x1 = v1.x_a[:dim]
            x2 = v2.x_a[:dim]
            x2_mi = min_image(x1, x2)

            # Check if this edge crosses a periodic boundary
            if np.allclose(x2, x2_mi, atol=1e-12):
                continue  # not a periodic edge

            # Find shared dual vertices
            shared_duals = v1.vd.intersection(v2.vd)

            # Find the triangles sharing this edge
            common_nbs = v1.nn.intersection(v2.nn)
            for v3 in common_nbs:
                x3 = v3.x_a[:dim]
                x3_mi = min_image(x1, x3)
                # Recompute barycenter with minimum-image coords
                bary = (x1 + x2_mi + x3_mi) / 3.0
                # Find the dual vertex for this triangle
                # It should be the one closest to the raw barycenter
                raw_bary = (x1 + x2 + x3) / 3.0
                best_vd = None
                best_dist = float('inf')
                for vd in shared_duals:
                    d = np.linalg.norm(vd.x_a[:dim] - raw_bary)
                    if d < best_dist:
                        best_dist = d
                        best_vd = vd
                if best_vd is not None and id(best_vd) not in processed_duals:
                    # Move dual vertex to corrected position
                    new_x = list(best_vd.x)
                    for i in range(dim):
                        new_x[i] = bary[i]
                    HC.Vd.move(best_vd, tuple(new_x))
                    processed_duals.add(id(best_vd))


# ---------------------------------------------------------------------------
# retopologize_periodic — main entry point
# ---------------------------------------------------------------------------

def retopologize_periodic(
    HC,
    bV: set,
    dim: int,
    periodic_axes: list[int],
    domain_bounds: list[tuple[float, float]],
    boundary_filter=None,
    merge_cdist: float | None = None,
    band_width: float | None = None,
    backend: str = "ghost",
) -> None:
    """Full periodic retriangulation — drop-in for ``_retopologize``.

    Parameters
    ----------
    HC : Complex
    bV : set
        Boundary vertex set (modified in-place).
    dim : int
    periodic_axes : list[int]
        Axes along which the domain is periodic (e.g. ``[0]``).
    domain_bounds : list[tuple[float, float]]
        Domain extent per axis.
    boundary_filter : callable or None
        Selects which non-periodic boundary vertices are frozen.
    merge_cdist : float or None
        Merge distance for close vertices.
    band_width : float or None
        Ghost buffer width.  ``None`` = auto (2x max edge length).
    backend : str
        ``"ghost"`` (default) or ``"cgal"`` (not yet implemented).
    """
    if backend == "cgal":
        raise NotImplementedError(
            "CGAL periodic Delaunay backend not yet implemented. "
            "Install cgal4py and contribute the backend, or use "
            "backend='ghost' (default)."
        )

    from hyperct.ddg import compute_vd
    from ddgclib.operators.stress import cache_dual_volumes

    verts = list(HC.V)
    if len(verts) < dim + 1:
        return

    # 1. Wrap positions into fundamental domain
    wrap_positions(HC, periodic_axes, domain_bounds)

    # 2. Optional merge of close vertices
    if merge_cdist is not None and merge_cdist > 0:
        HC.V.merge_all(cdist=merge_cdist)
        bV.intersection_update(set(HC.V))
        verts = list(HC.V)
        if len(verts) < dim + 1:
            return

    # 3. Disconnect all existing edges
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)

    # 4-6. Triangulate
    if dim == 1:
        # 1D periodic: merge endpoint duplicates (x=lb and x=ub are
        # the same physical point), then connect as a ring.
        if 0 in periodic_axes:
            lb, ub = domain_bounds[0]
            period = ub - lb
            # Merge ub-endpoint into lb-endpoint
            ub_verts = [v for v in verts if abs(v.x_a[0] - ub) < 1e-14]
            for v_ub in ub_verts:
                HC.V.remove(v_ub)
                bV.discard(v_ub)
            verts = list(HC.V)
        sorted_verts = sorted(verts, key=lambda v: v.x_a[0])
        for i in range(len(sorted_verts) - 1):
            sorted_verts[i].connect(sorted_verts[i + 1])
        if len(sorted_verts) >= 2 and 0 in periodic_axes:
            sorted_verts[-1].connect(sorted_verts[0])
    else:
        # 2D/3D: merge opposite periodic-face vertices.
        # In a periodic domain, vertices on opposite faces of a periodic
        # axis are the same physical point.  Remove ub-face duplicates
        # and keep lb-face vertices only.
        for ax in periodic_axes:
            lb_val, ub_val = domain_bounds[ax]
            # Find matching pairs: ub vertex -> lb vertex with same coords
            # on all other axes
            ub_verts = [v for v in HC.V
                        if abs(v.x_a[ax] - ub_val) < 1e-14]
            lb_verts = [v for v in HC.V
                        if abs(v.x_a[ax] - lb_val) < 1e-14]
            # Build lookup: key = coords on non-periodic axes
            def _key(v, skip_ax=ax):
                return tuple(round(v.x_a[a], 12) for a in range(dim)
                             if a != skip_ax)
            lb_lookup = {_key(v): v for v in lb_verts}
            for v_ub in ub_verts:
                k = _key(v_ub)
                if k in lb_lookup:
                    HC.V.remove(v_ub)
                    bV.discard(v_ub)
            verts = list(HC.V)

        # Ghost-cell Delaunay
        if band_width is None:
            coords = np.array([v.x_a[:dim] for v in verts])
            if len(coords) > 1:
                from scipy.spatial import cKDTree
                tree = cKDTree(coords)
                dists, _ = tree.query(coords, k=2)
                band_width = 2.0 * float(np.max(dists[:, 1]))
            else:
                band_width = 1.0

        all_coords, real_verts, ghost_map = create_ghost_vertices(
            HC, periodic_axes, domain_bounds, band_width, dim,
        )
        simplices = delaunay_with_ghosts(
            all_coords, len(real_verts), ghost_map, dim,
        )

        # Connect real vertices per resolved simplices
        for simplex in simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    real_verts[simplex[i]].connect(real_verts[simplex[j]])

    # 7. Topological boundary detection
    dV = HC.boundary()

    # 8-9. Tag boundaries: periodic faces are interior, UNLESS vertex
    #      is also on a non-periodic boundary (e.g. wall at y=0).
    periodic_face_verts = _identify_periodic_face_verts(
        HC, periodic_axes, domain_bounds,
    )
    # Identify vertices on non-periodic boundary faces
    non_periodic_axes = [ax for ax in range(dim) if ax not in periodic_axes]
    non_periodic_bnd = set()
    for v in HC.V:
        for ax in non_periodic_axes:
            lb, ub = domain_bounds[ax]
            if abs(v.x_a[ax] - lb) < 1e-14 or abs(v.x_a[ax] - ub) < 1e-14:
                non_periodic_bnd.add(v)
                break
    # Pure periodic = on a periodic face but NOT on any non-periodic face
    pure_periodic = periodic_face_verts - non_periodic_bnd
    for v in HC.V:
        if v in pure_periodic:
            v.boundary = False
        else:
            v.boundary = v in dV

    # 10. Compute dual mesh
    compute_vd(HC, method="barycentric")

    # 10b. Store periodic info on HC for use by stress operators.
    # dual_area_vector uses this to apply minimum-image wrapping
    # when computing dual face geometry across periodic boundaries.
    HC._periodic_axes = periodic_axes
    HC._periodic_bounds = domain_bounds

    # 11. Cache dual volumes
    cache_dual_volumes(HC, dim)

    # 12. Populate bV with non-periodic boundary vertices
    non_periodic_boundary = dV - pure_periodic
    if boundary_filter is not None:
        non_periodic_boundary = {v for v in non_periodic_boundary
                                 if boundary_filter(v)}
    bV.clear()
    bV.update(non_periodic_boundary)
