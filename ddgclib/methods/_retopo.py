"""Library versions of the retopology variants that used to live only in
case scripts.

Each function reproduces a case-local closure operation for operation, so
the pinned numbers of the case that introduced it are unchanged (proofs in
``ddgclib/tests/test_methods.py``).  They are selected through
``SolverMethods.connectivity`` and are kept here, outside
``dynamic_integrators/_integrators_dynamic.py``, so the core retopology
code is not touched.

``bare_dual_refresh``
    ``connectivity='dual_only_bare'``.  What ``static_droplet_2D.py``
    called ``_dual_only_retopo``: keep connectivity, retag the boundary
    from ``HC.boundary()``, recompute duals and dual volumes
    (``cache_dual_volumes``, boundary half-cells kept) and re-split the
    per-phase volumes.  It does NOT call ``mps.refresh``, does not
    redistribute mass and does not update EOS pressures, so ``p_phase``
    stays at its setup value (audit 2026-09-25 F11).

``retopologize_multiphase_periodic``
    ``connectivity='periodic'`` with ``phases='multi'``.  What
    ``shearing_plate_droplet/src/_setup.py`` built as
    ``_make_periodic_multiphase_retopo``: geometry snapshot, periodic ghost
    Delaunay, ``mps.refresh``, per-phase mass redistribution against the
    snapshot, EOS pressures.  No conservative remap and no projection
    cadence exist on this path.  ``remesh_mode`` / ``remesh_kwargs`` are
    accepted for the integrator's forwarding and ignored, exactly like the
    case closure.

``retopologize_material_delaunay``
    ``connectivity='delaunay_material'`` (single phase, laneP).  Per-step
    Delaunay rebuild in which the boundary of the previous connectivity
    is MATERIAL: the simplices scipy adds between a free surface and the
    convex hull of the point cloud are removed again (geometric test:
    winding number of the simplex centroid about the old boundary).  2D:
    the fluid domain is kept to round-off.  3D: kept up to the slivers
    between the old free-surface diagonals and the Delaunay ones (the
    rebuild is not constrained to the old surface facets); the relative
    change is returned and warned above ``domain_tol``.  With
    ``retopo_remap='conservative'`` it runs the single-phase conservative
    remap of ``_retopologize`` around the rebuild.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from math import factorial

import numpy as np

__all__ = ['bare_dual_refresh', 'retopologize_multiphase_periodic',
           'retopologize_material_delaunay']


def _set_edge_area_source(HC, dim, edge_area_source, where):
    """The 3D dual face source of a retopology that builds no fan cache
    (method axis ``edge_area_source``, laneQ): ``'p_ij_simplex'`` fills
    ``HC._edge_area_cache`` from ``hyperct.ddg.simplex_dual_face_areas``,
    ``'p_ij'`` and ``'p_ij_ring'`` (the legacy ring walk, = ``None``)
    leave it empty and tag the per-edge construction on
    ``HC._edge_area_source``.  ``'e_star_cache'`` is not built here."""
    if edge_area_source is None:
        edge_area_source = 'p_ij_ring'
    if edge_area_source not in ('p_ij', 'p_ij_simplex', 'p_ij_ring'):
        raise ValueError(
            f"{where} builds no batch_e_star cache: edge_area_source="
            f"{edge_area_source!r} is not applied (use 'p_ij', "
            "'p_ij_simplex' or 'p_ij_ring')")
    if dim != 3:
        if edge_area_source != 'p_ij_ring':
            raise ValueError("edge_area_source is a 3D axis")
        return
    if edge_area_source == 'p_ij_simplex':
        from hyperct.ddg import simplex_dual_face_areas
        HC._edge_area_cache = simplex_dual_face_areas(HC, dim)
    else:
        HC._edge_area_cache = None
    HC._edge_area_source = edge_area_source


def bare_dual_refresh(HC, bV, dim, mps=None, boundary_filter=None,
                      edge_area_source=None, **_kw):
    """Boundary retag + dual rebuild on frozen connectivity; nothing else.

    *boundary_filter* (``fn(v) -> bool``, forwarded by the integrator)
    selects which hull vertices are frozen, as in ``_retopologize``; the
    default ``None`` freezes the whole hull.  *edge_area_source* (3D,
    forwarded by the integrator): ``'p_ij_simplex'`` caches the exact
    dual faces of every edge, ``'p_ij'`` builds them per edge, ``None`` /
    ``'p_ij_ring'`` is the legacy ring walk (see
    :func:`_set_edge_area_source`).  ``_kw`` swallows the
    ``remesh_mode``/``remesh_kwargs`` the integrator forwards to every
    callable retopology function.
    """
    from hyperct.ddg import compute_vd
    from ddgclib.operators.stress import cache_dual_volumes

    dV = HC.boundary()
    for v in HC.V:
        v.boundary = v in dV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)
    _set_edge_area_source(HC, dim, edge_area_source, 'bare_dual_refresh')
    if mps is not None:
        mps.split_dual_volumes(HC, dim)
    if boundary_filter is not None:
        dV = {v for v in dV if boundary_filter(v)}
    bV.clear()
    bV.update(dV)


# A simplex whose measure is below _FLAT_TOL * (longest edge)**dim has no
# volume: three collinear or four coplanar vertices on a planar wall.
_FLAT_TOL = 1e-12


def _facet_keys(simplex) -> list[frozenset]:
    """The facets of one simplex as ``frozenset`` of vertex ids; facet
    ``j`` is the one opposite vertex ``j``."""
    return [frozenset(id(v) for i, v in enumerate(simplex) if i != j)
            for j in range(len(simplex))]


def _points(simplices, dim: int) -> np.ndarray:
    """Vertex coordinates of *simplices*, shape ``(n, dim + 1, dim)``."""
    return np.array([[v.x_a[:dim] for v in s] for s in simplices],
                    dtype=float).reshape(len(simplices), dim + 1, dim)


def _measure(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volume of every simplex and whether it is flat (``_FLAT_TOL``)."""
    dim = pts.shape[2]
    vol = np.abs(np.linalg.det(pts[:, 1:] - pts[:, :1])) / factorial(dim)
    longest = np.linalg.norm(pts[:, :, None] - pts[:, None, :],
                             axis=3).max(axis=(1, 2))
    return vol, vol <= _FLAT_TOL * longest**dim


def _owners(facets: list[list[frozenset]]) -> dict[frozenset, list[int]]:
    owners: dict[frozenset, list[int]] = {}
    for si, keys in enumerate(facets):
        for key in keys:
            owners.setdefault(key, []).append(si)
    return owners


def _drop_exposed_flat(alive: list[bool], flat: np.ndarray,
                       facets: list[list[frozenset]],
                       owners: dict[frozenset, list[int]]) -> None:
    """Mark dead every flat simplex that has a facet no other live simplex
    shares, repeatedly.  Such a simplex lies in the boundary and has no
    volume; dropping it only changes how the boundary is triangulated."""
    stack = [si for si in range(len(alive)) if alive[si] and flat[si]]
    while stack:
        si = stack.pop()
        if not alive[si]:
            continue
        if any(sum(alive[sj] for sj in owners[key]) == 1
               for key in facets[si]):
            alive[si] = False
            stack.extend(sj for key in facets[si] for sj in owners[key]
                         if alive[sj] and flat[sj])


def oriented_boundary(simplices, dim: int) -> list[tuple]:
    """Boundary facets of the region that *simplices* cover, as tuples of
    vertex objects, oriented.

    A facet is on the boundary when exactly one simplex owns it (flat
    simplices lying in the boundary are ignored).  It is oriented by the
    vertex of its owner that is not on it: in 2D the region is on the
    left of the edge, in 3D the normal ``(b - a) x (c - a)`` points away
    from the owner.  With that orientation :func:`winding_number` is 1
    inside the region and 0 outside.

    The orientation is read from the CURRENT positions, so call this
    while the simplices are valid (right after they were built).  The
    tuples stay usable after the vertices have moved, also when a thin
    simplex at the boundary has inverted in the meantime; re-deriving the
    orientation then would give a surface that is not closed.
    """
    pts = _points(simplices, dim)
    _, flat = _measure(pts)
    facets = [_facet_keys(s) for s in simplices]
    owners = _owners(facets)
    alive = [True] * len(simplices)
    _drop_exposed_flat(alive, flat, facets, owners)

    out = []
    for si, s in enumerate(simplices):
        if not alive[si]:
            continue
        for j, key in enumerate(facets[si]):
            if sum(alive[sj] for sj in owners[key]) != 1:
                continue
            f = [v for i, v in enumerate(s) if i != j]
            x = np.delete(pts[si], j, axis=0)
            side = np.linalg.det(np.vstack([x[1:], pts[si, j:j + 1]]) - x[0])
            if (side < 0.0) if dim == 2 else (side > 0.0):
                f[0], f[1] = f[1], f[0]
            out.append(tuple(f))
    return out


def _facet_points(boundary: list[tuple], dim: int) -> np.ndarray:
    """Current coordinates of *boundary*, shape ``(n, dim, dim)``."""
    return np.array([[v.x_a[:dim] for v in f] for f in boundary],
                    dtype=float).reshape(len(boundary), dim, dim)


def enclosed_volume(fac: np.ndarray) -> float:
    """Measure of the region inside the oriented facets *fac*
    (``(n, dim, dim)`` coordinates): divergence theorem."""
    return float(np.linalg.det(fac).sum()) / factorial(fac.shape[2])


def winding_number(x: np.ndarray, fac: np.ndarray) -> np.ndarray:
    """Winding number of the closed oriented facets *fac* (``(n, dim,
    dim)`` coordinates of an :func:`oriented_boundary`) around each point
    of *x* (``(m, dim)``): 1 inside, 0 outside.  2D: sum of the signed
    angles; 3D: sum of the signed solid angles (Van Oosterom and
    Strackee)."""
    dim = fac.shape[2]
    out = np.empty(len(x))
    step = max(1, 200_000 // max(1, len(fac)))
    for i in range(0, len(x), step):
        r = fac[None] - x[i:i + step, None, None]         # (m, n, dim, dim)
        a, b = r[:, :, 0], r[:, :, 1]
        if dim == 2:
            ang = np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
                             (a * b).sum(axis=2))
            out[i:i + step] = ang.sum(axis=1) / (2.0 * np.pi)
        else:
            c = r[:, :, 2]
            la, lb, lc = (np.linalg.norm(q, axis=2) for q in (a, b, c))
            num = (a * np.cross(b, c)).sum(axis=2)
            den = (la * lb * lc + (a * b).sum(axis=2) * lc
                   + (a * c).sum(axis=2) * lb + (b * c).sum(axis=2) * la)
            out[i:i + step] = np.arctan2(num, den).sum(axis=1) / (2.0 * np.pi)
    return out


def peel_outside_boundary(HC, boundary: list[tuple], dim: int) -> int:
    """Remove the simplices of ``HC._simplices`` that lie outside
    *boundary* (the :func:`oriented_boundary` of the previous simplices,
    evaluated at the current positions).

    A Delaunay triangulation covers the convex hull of the point cloud.
    Where the fluid boundary is not convex (a free surface that has
    moved) it therefore adds simplices between that boundary and the
    hull.  Only a simplex whose vertices are ALL boundary vertices can be
    such a fill, so only those are tested, and the test is geometric: the
    simplex goes when its centroid is outside *boundary*
    (:func:`winding_number` below 1/2).  Flat simplices have no inside;
    they go when they are exposed (:func:`_drop_exposed_flat`: the
    coplanar wall tetrahedra qhull returns on a structured mesh).

    The test does not ask whether the facets of *boundary* are facets of
    the new triangulation.  The first version did (a simplex was peeled
    when it exposed a facet that was not an old boundary facet), and that
    removed fluid in 3D: the two diagonals of a planar wall square are
    equally Delaunay, so the wall facets change between rebuilds and
    every tetrahedron with four wall vertices was taken for hull fill
    (a lattice cube lost up to 42 % of its volume in one call).

    A simplex with a vertex that is not on *boundary* is never removed,
    so no interior vertex is orphaned.  If such a simplex reaches across
    *boundary* (a boundary facet that is not in the new triangulation)
    the region changes by the part outside; see
    :func:`retopologize_material_delaunay`.

    Edges that belonged only to removed simplices are disconnected and
    ``HC._simplices`` is replaced.  Returns the number of simplices
    removed.
    """
    simplices = HC._simplices
    pts = _points(simplices, dim)
    _, flat = _measure(pts)
    ids = {id(v) for f in boundary for v in f}
    alive = [True] * len(simplices)
    cand = [si for si, s in enumerate(simplices)
            if not flat[si] and all(id(v) in ids for v in s)]
    if cand:
        inside = winding_number(pts[cand].mean(axis=1),
                                _facet_points(boundary, dim))
        for si, w in zip(cand, inside):
            alive[si] = bool(w > 0.5)
    if flat.any():
        facets = [_facet_keys(s) for s in simplices]
        _drop_exposed_flat(alive, flat, facets, _owners(facets))

    n_removed = alive.count(False)
    if n_removed == 0:
        return 0
    kept = [s for s, a in zip(simplices, alive) if a]
    kept_edges = {frozenset((id(a), id(b)))
                  for s in kept for a, b in combinations(s, 2)}
    for s, a in zip(simplices, alive):
        if a:
            continue
        for p, q in combinations(s, 2):
            if frozenset((id(p), id(q))) not in kept_edges and q in p.nn:
                p.disconnect(q)
    HC._simplices = kept
    if hasattr(HC, '_edge_to_apex'):
        HC._edge_to_apex = None
    return n_removed


def retopologize_material_delaunay(HC, bV, dim, boundary_filter=None,
                                   pressure_model=None,
                                   redistribute_mass=False,
                                   retopo_remap=None, domain_tol=1e-3,
                                   edge_area_source=None,
                                   **_kw) -> float:
    """Delaunay rebuild that keeps the fluid domain (single phase).

    The plain rebuild of ``_retopologize`` triangulates the convex hull.
    On a column with a free surface that fills the gap between the
    surface and the hull with near-degenerate simplices, which flip in
    and out every step and make the run unstable even with the
    conservative remap (laneP: max |u| 42 m/s at 3 acoustic times on the
    2D hydrostatic column, against 0.25 m/s with this function).  Here
    the boundary of the connectivity BEFORE the rebuild is material:
    :func:`peel_outside_boundary` removes the simplices Delaunay adds
    outside it.  Interior edges still reconnect.

    How well the domain is kept.  The new triangulation is not
    constrained to contain the old boundary facets.  On a planar wall
    that is harmless (another triangulation of the same plane).  On a
    free surface that is not planar it is not: where the Delaunay
    diagonal of a surface quadrilateral differs from the old one, or an
    interior vertex reaches through the surface, the domain gains or
    loses the sliver between the two surfaces.  Measured (laneP): 2D
    hydrostatic column 0 to round-off in every call; 3D column at most
    1.2e-6 of the volume in one call (the four corner squares at the
    first rebuild), 1.6e-6 summed over 185 calls; a 0.004 bowl pushed
    into the builder surface in one go 9.8e-5.  The relative change of
    the domain volume across the call is RETURNED, and a ``UserWarning``
    is raised when it exceeds *domain_tol*.  With the remap a change
    ``c`` shifts the pressure everywhere by about ``-K c`` (one mass
    rescale), so it is not harmless on a free surface.

    The boundary is oriented when its simplices are built and cached on
    ``HC._material_boundary`` for the next call (first call: taken from
    ``HC._simplices`` at the current positions).  Re-deriving the
    orientation at the next call fails once a thin simplex at the
    surface has inverted during the step: the facets no longer form a
    closed surface, the winding numbers are not integers, and the 3D
    column domain flickers by up to 8e-5 per call (measured DO-NOT).

    Conventions (those of :func:`bare_dual_refresh`): boundary vertices
    keep their truncated dual cell in 2D and 3D, no edge-area cache is
    kept (2D shared dual vertices, 3D ``p_ij`` ring), ``v.boundary`` is
    the true boundary of the kept simplices, *bV* is that boundary
    narrowed by *boundary_filter* (pass the wall criterion so that
    free-surface vertices stay free).

    ``retopo_remap='conservative'`` wraps the rebuild in the single-phase
    conservative remap (fresh pressure snapshot on the old connectivity,
    re-target every vertex including frozen ones, one exact mass
    rescale); it needs ``redistribute_mass=True`` and an EOS
    *pressure_model*.  Without it the masses are held, which is the
    measured-unstable combination of laneK when an EOS is in the loop.

    *edge_area_source* (3D, forwarded by the integrator): the dual face
    source of the rebuilt connectivity, see :func:`_set_edge_area_source`
    (``None`` = the legacy ring walk, ``'p_ij_simplex'`` = exact cached
    faces, ``'p_ij'`` = exact per edge).

    Needs the simplex cache of the current connectivity
    (``HC._simplices``; every domain builder provides it).  ``_kw``
    swallows ``remesh_mode``/``remesh_kwargs``.
    """
    from hyperct.ddg import (boundary_from_simplices, compute_vd,
                             connect_and_cache_simplices)
    from ddgclib.operators.stress import cache_dual_volumes

    if dim not in (2, 3):
        raise ValueError("retopologize_material_delaunay supports dim 2 "
                         f"and 3, got {dim}")
    if retopo_remap not in (None, 'conservative'):
        raise ValueError("retopo_remap must be None or 'conservative', "
                         f"got {retopo_remap!r}")
    remap = retopo_remap == 'conservative'
    if remap != bool(redistribute_mass):
        raise ValueError(
            "retopologize_material_delaunay: retopo_remap='conservative' "
            "and redistribute_mass=True go together (redistribution "
            "without the remap is the stale-snapshot variant, a measured "
            "DO-NOT of laneK)")
    if remap and not hasattr(pressure_model, 'density'):
        raise ValueError("retopo_remap='conservative' requires an "
                         "EquationOfState pressure_model")
    if getattr(HC, '_simplices', None) is None:
        raise ValueError(
            "retopologize_material_delaunay needs HC._simplices (the "
            "simplices of the current connectivity define the boundary "
            "to keep); domain builders populate it")

    snap = None
    if remap:
        from ddgclib.operators.mass_redistribution import (
            snapshot_pressure_fresh,
        )
        snap = snapshot_pressure_fresh(HC, dim, pressure_model)
    # The boundary to keep, oriented when its simplices were built: by the
    # previous call (cached), else from the cache as it stands now.
    cached = getattr(HC, '_material_boundary', None)
    if cached is not None and cached[0] is HC._simplices:
        boundary = cached[1]
    else:
        boundary = oriented_boundary(HC._simplices, dim)
    volume = enclosed_volume(_facet_points(boundary, dim))

    verts = list(HC.V)
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)
    coords = np.array([v.x_a[:dim] for v in verts])
    connect_and_cache_simplices(HC, verts, dim, coords=coords)
    peel_outside_boundary(HC, boundary, dim)
    boundary = oriented_boundary(HC._simplices, dim)
    HC._material_boundary = (HC._simplices, boundary)
    change = enclosed_volume(_facet_points(boundary, dim)) / volume - 1.0
    if abs(change) > domain_tol:
        warnings.warn(
            f"retopologize_material_delaunay: the domain volume changed by "
            f"{change:+.3e} across the rebuild (domain_tol={domain_tol:g}): "
            "boundary facets of the previous connectivity are missing from "
            "the new triangulation", UserWarning, stacklevel=2)

    dV = boundary_from_simplices(HC, dim)
    for v in HC.V:
        v.boundary = v in dV
    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)
    HC._edge_area_cache = None
    _set_edge_area_source(HC, dim, edge_area_source,
                          'retopologize_material_delaunay')

    if boundary_filter is not None:
        dV = {v for v in dV if boundary_filter(v)}
    bV.clear()
    bV.update(dV)

    if remap:
        from ddgclib.operators.mass_redistribution import (
            redistribute_mass_single_phase,
        )
        redistribute_mass_single_phase(
            HC, dim, pressure_model, bV=bV, pressure_snapshot=snap,
            include_frozen=True,
        )
    return change


def retopologize_multiphase_periodic(HC, bV, dim, mps=None, periodic_axes=None,
                                     domain_bounds=None,
                                     split_method='neighbour_count',
                                     redistribute_mass=False,
                                     remesh_mode='delaunay',
                                     remesh_kwargs=None,
                                     phase_ledger='volume'):
    """Periodic ghost-cell Delaunay + multiphase refresh (+ redistribution).

    Mirrors ``_retopologize_multiphase`` with :func:`retopologize_periodic`
    in place of the plain Delaunay step.  When *redistribute_mass* is True
    the pre-call per-phase ``dual_vol_phase`` is snapshotted and used as the
    gating mask in ``redistribute_mass_multiphase`` so per-phase pressure is
    preserved across reconnection.  *remesh_mode*/*remesh_kwargs* are
    ignored (periodic adaptive remesh is not implemented).  *phase_ledger*
    is the ``ledger=`` rule of that redistribution (method axis
    ``phase_ledger``; default the historic ``'snapshot'``).
    """
    if periodic_axes is None or domain_bounds is None:
        raise ValueError("retopologize_multiphase_periodic needs periodic_axes "
                         "and domain_bounds")
    from ddgclib.geometry.periodic import retopologize_periodic

    _p_snap = None
    if redistribute_mass and mps is not None:
        from ddgclib.operators.mass_redistribution import (
            snapshot_geometry_multiphase,
        )
        _p_snap = snapshot_geometry_multiphase(HC, mps.n_phases)

    retopologize_periodic(
        HC, bV, dim,
        periodic_axes=list(periodic_axes),
        domain_bounds=[tuple(b) for b in domain_bounds],
    )
    if mps is not None:
        mps.refresh(HC, dim, reset_mass=False, split_method=split_method)

        if redistribute_mass and _p_snap is not None:
            from ddgclib.operators.mass_redistribution import (
                redistribute_mass_multiphase,
            )
            redistribute_mass_multiphase(
                HC, dim, mps, bV=bV, pressure_snapshot=_p_snap,
                ledger=phase_ledger,
            )
            mps.compute_phase_pressures(HC)
