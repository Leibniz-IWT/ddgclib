"""Exact dual-volume split at sharp interfaces (2D and 3D).

For an interface vertex ``v`` whose dual cell straddles a sharp phase
interface, partition the dual cell into per-phase sub-cells:

- **2D** (:func:`split_dual_polygon_2d`): clip the barycentric dual
  polygon against the piecewise-linear interface curve.  The curve
  near ``v`` is determined by its two interface-curve neighbours
  ``v_prev`` and ``v_next`` (selected by the same largest-angular-gap
  rule used in :mod:`ddgclib.operators.curvature_2d`).  The polyline
  enters at the midpoint of edge ``(v, v_prev)``, passes through
  ``v``, and exits at the midpoint of edge ``(v, v_next)`` — both
  midpoints are vertices of the barycentric dual polygon (with edge
  midpoints included), so the split is exact on piecewise-linear
  interfaces.

- **3D** (:func:`split_dual_polyhedron_3d`): clip the DEC ``p_ij``
  barycentric dual polyhedron against a **local interface plane** at
  ``v``.  The plane is defined by an approximate surface normal
  derived from the 1-ring interface neighbours (the eigenvector of
  the smallest eigenvalue of their centred covariance matrix — a
  planar least-squares fit through ``v``).  This gives a principled
  volume split for the common case where the interface is locally
  flat to the dual-cell scale.  Bulk vertices fall through to the
  full dual volume in their own phase.

Also provides :func:`edge_phase_area_fractions` for per-edge sub-face
fractions used by :func:`ddgclib.operators.multiphase_stress.multiphase_stress_force`.

Both routines are exposed through
:meth:`MultiphaseSystem.split_dual_volumes` when called with
``method='exact'`` (default remains ``'neighbour_count'`` so existing
cases are unaffected until they opt in).
"""
from __future__ import annotations

import numpy as np

def _interface_neighbours(v):
    """Return the set of 1-ring neighbours of ``v`` flagged as interface."""
    return {nb for nb in v.nn if getattr(nb, 'is_interface', False)}


def _select_curve_neighbours(v, interface_nbs):
    """Lazy wrapper to avoid a circular import with operators.curvature_2d."""
    from ddgclib.operators.curvature_2d import (
        _select_curve_neighbours as _impl,
    )
    return _impl(v, interface_nbs)


def _build_typed_polygon(v) -> list[dict]:
    """Barycentric dual polygon (with edge midpoints) as typed entries.

    Each entry is a dict with keys:

    - ``kind``: ``'bary'`` or ``'midpoint'``
    - ``pos``:  ``(x, y)`` ndarray
    - ``vd`` (for ``bary``):     originating dual vertex object
    - ``nb`` (for ``midpoint``): neighbour primal vertex

    The entries are returned in CCW order around ``v``.
    """
    cx, cy = float(v.x_a[0]), float(v.x_a[1])
    entries: list[dict] = []

    for vd in v.vd:
        entries.append({
            'kind': 'bary',
            'pos': vd.x_a[:2].astype(float).copy(),
            'vd': vd,
        })

    for v_j in v.nn:
        mp = 0.5 * (v.x_a[:2].astype(float) + v_j.x_a[:2].astype(float))
        entries.append({
            'kind': 'midpoint',
            'pos': mp,
            'nb': v_j,
        })

    angles = np.array([
        np.arctan2(e['pos'][1] - cy, e['pos'][0] - cx) for e in entries
    ])
    order = np.argsort(angles)
    return [entries[i] for i in order]


def _shoelace_area(polygon) -> float:
    poly = np.asarray(polygon, dtype=float)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _full_polygon_area(v) -> float:
    from hyperct.ddg._dual_cell import dual_cell_area_2d
    return float(dual_cell_area_2d(v, include_edge_midpoints=True))


def split_dual_polygon_2d(v, n_phases: int = 2, interface=None) -> dict[int, float]:
    """Return per-phase sub-polygon areas for an interior 2D vertex.

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd``, ``v.nn``, ``v.x_a``, ``v.phase`` populated,
        and — if it's an interface vertex — ``v.is_interface = True``
        with its interface 1-ring neighbours also flagged.
    n_phases : int
        Number of phases in the system.
    interface : Complex or None, optional
        When provided, the two curve neighbours of ``v`` are looked up
        from ``HC.interface_edges`` (exact subcomplex model).  When
        ``None``, falls back to the angular-gap heuristic (legacy).

    Returns
    -------
    dict[int, float]
        Sub-polygon area per phase.  Keys are ``0..n_phases-1``.  For
        bulk vertices only ``v.phase`` has non-zero area.
    """
    result = {k: 0.0 for k in range(n_phases)}

    if not getattr(v, 'is_interface', False):
        bulk_phase = int(v.phase) if v.phase >= 0 else 0
        result[bulk_phase] = _full_polygon_area(v)
        return result

    # Look up curve neighbours from the interface subcomplex (new) or
    # from the angular-gap heuristic (legacy fallback).
    if interface is not None:
        from ddgclib.geometry._interface_subcomplex import (
            curve_neighbours as _curve_nbs,
        )
        v_prev, v_next = _curve_nbs(v, interface)
    else:
        iface_nbs = _interface_neighbours(v)
        v_prev, v_next = _select_curve_neighbours(v, iface_nbs)
    if v_prev is None or v_next is None:
        active = [k for k in getattr(v, 'interface_phases', frozenset())
                  if 0 <= k < n_phases]
        if not active:
            active = [0]
        vol = _full_polygon_area(v)
        share = vol / len(active)
        for k in active:
            result[k] = share
        return result

    entries = _build_typed_polygon(v)
    n = len(entries)
    if n < 3:
        result[v.phase] = _full_polygon_area(v)
        return result

    idx_prev = None
    idx_next = None
    for i, e in enumerate(entries):
        if e['kind'] != 'midpoint':
            continue
        if e['nb'] is v_prev and idx_prev is None:
            idx_prev = i
        elif e['nb'] is v_next and idx_next is None:
            idx_next = i

    if idx_prev is None or idx_next is None or idx_prev == idx_next:
        result[v.phase] = _full_polygon_area(v)
        return result

    def _walk_forward(start: int, end: int) -> list[dict]:
        walked: list[dict] = []
        i = (start + 1) % n
        guard = 0
        while i != end and guard < n + 2:
            walked.append(entries[i])
            i = (i + 1) % n
            guard += 1
        return walked

    # Side A: forward from m_prev to m_next (CCW arc of the dual polygon).
    # Side B: forward from m_next to m_prev (the other CCW arc, wrapping
    # through the start of the array).  Walking forward in both cases
    # keeps each sub-polygon simple (non self-intersecting).
    side_A = _walk_forward(idx_prev, idx_next)
    side_B = _walk_forward(idx_next, idx_prev)

    m_prev_pos = entries[idx_prev]['pos']
    m_next_pos = entries[idx_next]['pos']
    v_pos = v.x_a[:2].astype(float).copy()

    poly_A = [m_prev_pos] + [e['pos'] for e in side_A] + [m_next_pos, v_pos]
    poly_B = [m_next_pos] + [e['pos'] for e in side_B] + [m_prev_pos, v_pos]

    area_A = abs(_shoelace_area(poly_A))
    area_B = abs(_shoelace_area(poly_B))

    def _phase_from_side(side_entries: list[dict]) -> int | None:
        for e in side_entries:
            if e['kind'] != 'midpoint':
                continue
            nb = e['nb']
            if not getattr(nb, 'is_interface', False):
                return int(nb.phase)
        return None

    phase_A = _phase_from_side(side_A)
    phase_B = _phase_from_side(side_B)

    if n_phases == 2:
        if phase_A is None and phase_B is not None:
            phase_A = 1 - phase_B
        elif phase_B is None and phase_A is not None:
            phase_B = 1 - phase_A
        elif phase_A is None and phase_B is None:
            iface_ph = sorted(getattr(v, 'interface_phases', frozenset({v.phase})))
            if len(iface_ph) >= 2:
                phase_A, phase_B = iface_ph[0], iface_ph[1]
            else:
                phase_A = v.phase
                phase_B = 1 - v.phase
        if phase_A == phase_B:
            phase_B = 1 - phase_A
    else:
        if phase_A is None:
            phase_A = v.phase
        if phase_B is None:
            phase_B = v.phase

    result[phase_A] = result.get(phase_A, 0.0) + area_A
    result[phase_B] = result.get(phase_B, 0.0) + area_B
    return result


# ---------------------------------------------------------------------------
# 3D dual-volume split (interface-plane clipping)
# ---------------------------------------------------------------------------

def _interface_plane_at_3d(v) -> tuple[np.ndarray, np.ndarray] | None:
    """Local interface plane through an interface vertex in 3D.

    Fits a plane through ``v`` to its interface 1-ring neighbours via
    PCA on the centred neighbour positions.  The plane normal is the
    eigenvector of the smallest eigenvalue of the covariance matrix —
    the direction of least variance is the surface normal for a
    locally flat interface.  Returns ``(point, normal)`` with ``point
    == v.x_a[:3]`` and ``normal`` a unit vector; returns ``None`` when
    there are fewer than three interface neighbours.
    """
    iface_nbs = _interface_neighbours(v)
    if len(iface_nbs) < 3:
        return None
    pts = np.array([nb.x_a[:3] for nb in iface_nbs], dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Plane normal = eigenvector of smallest eigenvalue of cov(centered)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-30:
        return None
    return v.x_a[:3].astype(float), normal / n_norm


def _polyhedron_faces_to_tets(
    faces: list[np.ndarray], apex: np.ndarray,
) -> list[np.ndarray]:
    """Decompose a polyhedron into signed tetrahedra from an interior apex.

    For each face, fan-triangulate from the face centroid and emit
    tetrahedra ``(apex, tri[0], tri[1], tri[2])``.  Signed volumes sum
    to the total polyhedron volume regardless of apex choice, so any
    apex works (we use the primal vertex).

    Parameters
    ----------
    faces : list of (M, 3) arrays
        Ordered polygon vertices for each face (orientation agnostic).
    apex : (3,) array
        Interior apex for the fan.

    Returns
    -------
    list of (4, 3) arrays
        Each element is ``[apex, p0, p1, p2]`` of a fan tetrahedron.
    """
    tets: list[np.ndarray] = []
    for face in faces:
        if len(face) < 3:
            continue
        face = np.asarray(face, dtype=float)
        centroid = face.mean(axis=0)
        for k in range(len(face)):
            p0 = face[k]
            p1 = face[(k + 1) % len(face)]
            tets.append(np.array([apex, centroid, p0, p1]))
    return tets


def _tet_signed_volume(tet: np.ndarray) -> float:
    a, b, c, d = tet
    return float(np.dot(b - a, np.cross(c - a, d - a))) / 6.0


def _clip_tet_by_plane(
    tet: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray,
) -> tuple[float, float]:
    """Return (vol_positive, vol_negative) sub-volumes of a tetrahedron
    split by a plane.

    Signed volumes; ``vol_positive + vol_negative == signed_volume(tet)``
    to machine precision.  The split uses the standard case analysis
    from the Kuipers 1997 tetrahedral-clipping algorithm.
    """
    # Signed distance of each vertex to the plane
    d = np.array([np.dot(p - plane_point, plane_normal) for p in tet])
    total = _tet_signed_volume(tet)
    # Classify every vertex as +/- (on-plane vertices go to +).
    # This guarantees len(pos_idx) + len(neg_idx) == 4 even when
    # a vertex (typically the polyhedron apex at ``v``) lies on the
    # plane, so the case analysis below is always consistent.
    eps = 1e-14
    pos_idx = [i for i in range(4) if d[i] >= -eps]
    neg_idx = [i for i in range(4) if d[i] < -eps]

    if len(neg_idx) == 0:
        return total, 0.0
    if len(pos_idx) == 0:
        return 0.0, total

    def _intersect(i: int, j: int) -> np.ndarray:
        t = d[i] / (d[i] - d[j])
        return tet[i] + t * (tet[j] - tet[i])

    if len(pos_idx) == 1:
        # One vertex above → a small tet on the positive side
        a = pos_idx[0]
        bs = [j for j in range(4) if j != a]
        p_ab = _intersect(a, bs[0])
        p_ac = _intersect(a, bs[1])
        p_ad = _intersect(a, bs[2])
        vol_pos = _tet_signed_volume(np.array([tet[a], p_ab, p_ac, p_ad]))
        return vol_pos, total - vol_pos

    if len(pos_idx) == 3:
        # Three vertices above → symmetric: small tet on the negative side
        a = neg_idx[0]
        bs = [j for j in range(4) if j != a]
        p_ab = _intersect(a, bs[0])
        p_ac = _intersect(a, bs[1])
        p_ad = _intersect(a, bs[2])
        vol_neg = _tet_signed_volume(np.array([tet[a], p_ab, p_ac, p_ad]))
        return total - vol_neg, vol_neg

    # len(pos_idx) == 2 and len(neg_idx) == 2 — two on each side.
    # Compute positive-side volume by splitting into three sub-tets.
    pa, pb = pos_idx
    na, nb = neg_idx
    # Four intersection points on edges pa-na, pa-nb, pb-na, pb-nb
    p_an = _intersect(pa, na)
    p_am = _intersect(pa, nb)
    p_bn = _intersect(pb, na)
    p_bm = _intersect(pb, nb)
    # Positive-side prism with vertices [tet[pa], tet[pb], p_an, p_bn, p_am, p_bm]
    # Split into three tets (Sutherland-Hodgman style):
    t1 = np.array([tet[pa], tet[pb], p_an, p_bn])
    t2 = np.array([tet[pa], p_an, p_bn, p_bm])
    t3 = np.array([tet[pa], p_bm, p_am, tet[pb]])
    vol_pos = (
        _tet_signed_volume(t1)
        + _tet_signed_volume(t2)
        + _tet_signed_volume(t3)
    )
    return vol_pos, total - vol_pos


def _dual_volume_3d(v, HC) -> float:
    """Total dual cell volume at ``v`` (3D)."""
    from ddgclib.operators.stress import dual_volume
    return float(dual_volume(v, HC, dim=3))


def split_dual_polyhedron_3d(v, HC, n_phases: int = 2) -> dict[int, float]:
    """Return per-phase sub-polyhedron volumes for an interior 3D vertex.

    Uses the DEC ``p_ij`` dual polyhedron (from
    :func:`hyperct.ddg._dual_cell.dual_cell_faces_3d`) clipped against
    a local interface plane through ``v``.

    Parameters
    ----------
    v : vertex object
        Must have ``v.vd``, ``v.nn``, ``v.x_a``, ``v.phase`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    n_phases : int
        Number of phases in the system.

    Returns
    -------
    dict[int, float]
        Sub-volume per phase.  Keys are ``0..n_phases-1``.  For bulk
        vertices only ``v.phase`` has non-zero volume.  Falls back to
        the full dual volume in ``v.phase`` when the interface plane
        cannot be reconstructed (fewer than three interface neighbours
        or degenerate covariance).
    """
    result = {k: 0.0 for k in range(n_phases)}

    if not getattr(v, 'is_interface', False):
        bulk_phase = int(v.phase) if v.phase >= 0 else 0
        result[bulk_phase] = _dual_volume_3d(v, HC)
        return result

    plane = _interface_plane_at_3d(v)
    if plane is None:
        # Fallback: can't determine plane; split equally among active
        # phases.
        active = [k for k in getattr(v, 'interface_phases', frozenset())
                  if 0 <= k < n_phases]
        if not active:
            active = [0]
        vol = _dual_volume_3d(v, HC)
        share = vol / len(active)
        for k in active:
            result[k] = share
        return result
    plane_point, plane_normal = plane

    # Build dual polyhedron faces
    from hyperct.ddg._dual_cell import dual_cell_faces_3d
    faces = dual_cell_faces_3d(v, HC, include_face_barycenters=True)
    if len(faces) < 4:
        active = [k for k in getattr(v, 'interface_phases', frozenset())
                  if 0 <= k < n_phases]
        if not active:
            active = [0]
        vol = _dual_volume_3d(v, HC)
        share = vol / len(active)
        for k in active:
            result[k] = share
        return result

    apex = v.x_a[:3].astype(float)
    tets = _polyhedron_faces_to_tets(faces, apex)
    if not tets:
        active = [k for k in getattr(v, 'interface_phases', frozenset())
                  if 0 <= k < n_phases]
        if not active:
            active = [0]
        vol = _dual_volume_3d(v, HC)
        share = vol / len(active)
        for k in active:
            result[k] = share
        return result

    # Determine which two phases are present at this interface vertex.
    # v.phase may be INTERFACE_PHASE (-1); use interface_phases instead.
    active = sorted(
        k for k in getattr(v, 'interface_phases', frozenset()) if 0 <= k < n_phases
    )
    if len(active) < 2:
        active = [0, 1] if n_phases >= 2 else [0]
    own_phase = active[0]
    other_phase = active[1] if len(active) >= 2 else active[0]

    # Pick the plane orientation so that "positive side" contains the
    # own-phase (active[0]) bulk neighbours.
    own_side_votes = 0.0
    other_side_votes = 0.0
    for nb in v.nn:
        if getattr(nb, 'is_interface', False):
            continue
        nb_phase = int(nb.phase) if nb.phase >= 0 else -1
        d_nb = float(np.dot(nb.x_a[:3] - plane_point, plane_normal))
        if nb_phase == own_phase:
            own_side_votes += d_nb
        elif nb_phase == other_phase:
            other_side_votes += d_nb
    if own_side_votes < other_side_votes:
        plane_normal = -plane_normal

    vol_pos = 0.0
    vol_neg = 0.0
    for tet in tets:
        vp, vn = _clip_tet_by_plane(tet, plane_point, plane_normal)
        vol_pos += vp
        vol_neg += vn

    # Signed volumes from the p_ij polyhedron's fan tetrahedralisation.
    # These partition the p_ij dual volume, which may differ from
    # ``v.dual_vol`` (computed from v_star) on non-symmetric meshes.
    # Rescale so the partition sums to the authoritative ``v.dual_vol``
    # used throughout the multiphase pipeline — preserves the ratio of
    # phase-0 to phase-1 sub-volumes from the geometric clip.
    vol_own_raw = abs(vol_pos)
    vol_other_raw = abs(vol_neg)
    vol_sum_raw = vol_own_raw + vol_other_raw
    target_total = _dual_volume_3d(v, HC)
    if vol_sum_raw > 0 and target_total > 0:
        scale = target_total / vol_sum_raw
        vol_own = vol_own_raw * scale
        vol_other = vol_other_raw * scale
    else:
        vol_own = vol_own_raw
        vol_other = vol_other_raw

    result[own_phase] = vol_own
    if other_phase != own_phase:
        result[other_phase] = vol_other
    return result


def _is_curve_adjacent(v_i, v_j, dim: int = 2, interface=None) -> bool:
    """True if the primal edge ``(v_i, v_j)`` is on the interface.

    When ``interface`` (HC with ``interface_edges``) is provided, the
    check is a direct set-membership lookup (exact).  Otherwise falls
    back to the angular-gap heuristic (legacy, 2D only).

    - **2D**: edge is in ``interface.interface_edges``.
    - **3D**: edge is in ``interface.interface_edges`` (edges of
      interface triangles).
    """
    if not getattr(v_i, 'is_interface', False):
        return False
    if not getattr(v_j, 'is_interface', False):
        return False

    if interface is not None:
        ekey = frozenset({v_i.x, v_j.x})
        return ekey in getattr(interface, 'interface_edges', set())

    # Legacy fallback (no subcomplex available).
    iface_nbs = _interface_neighbours(v_i)
    if v_j not in iface_nbs:
        return False
    if dim >= 3:
        return True
    v_prev, v_next = _select_curve_neighbours(v_i, iface_nbs)
    return v_j is v_prev or v_j is v_next


def edge_phase_area_fractions(
    v_i, v_j, dim: int = 2, interface=None,
) -> dict[int, float]:
    """Return fractional split of the dual face between ``v_i`` and ``v_j``.

    ``fractions[k]`` is the fraction of the dual face that lies in
    phase ``k``.  Sums to 1 over phases.

    When ``interface`` (HC with ``interface_edges``) is provided,
    curve/surface adjacency is determined by a direct edge-membership
    lookup (exact).

    Rules
    -----
    - Neither vertex is interface: 100% in the common phase.
    - Exactly one is interface, the other is bulk in phase *k*:
      100% in phase *k*.
    - Both are interface AND the primal edge is on the interface:
      50/50 split between the two interface phases.
    - Both are interface but the primal edge is NOT on the interface
      (interior chord): split equally among phases present at ``v_i``.

    Returns
    -------
    dict[int, float]
        Mapping ``{phase: fraction}`` with fractions summing to 1.0.
    """
    is_i_iface = bool(getattr(v_i, 'is_interface', False))
    is_j_iface = bool(getattr(v_j, 'is_interface', False))

    if not is_i_iface and not is_j_iface:
        phase = int(v_i.phase) if v_i.phase >= 0 else 0
        return {phase: 1.0}

    if is_i_iface and not is_j_iface:
        return {int(v_j.phase): 1.0}

    if is_j_iface and not is_i_iface:
        return {int(v_i.phase): 1.0}

    if _is_curve_adjacent(v_i, v_j, dim=dim, interface=interface):
        pair = sorted(
            k for k in (
                set(getattr(v_i, 'interface_phases', frozenset()))
                | set(getattr(v_j, 'interface_phases', frozenset()))
            ) if k >= 0
        )
        if len(pair) >= 2:
            return {int(pair[0]): 0.5, int(pair[1]): 0.5}

    # Interior chord (both interface, not on interface edge): determine
    # the phase from the shared simplex context.  The common 1-ring
    # neighbours of v_i and v_j form the vertices of the triangles
    # sharing this edge.  Their bulk phases tell us which phase the
    # chord's dual face lies in.
    shared = v_i.nn & v_j.nn
    bulk_shared = [int(nb.phase) for nb in shared if nb.phase >= 0]
    if bulk_shared:
        from collections import Counter
        counts = Counter(bulk_shared)
        chord_phase = counts.most_common(1)[0][0]
        return {chord_phase: 1.0}
    # Fallback: all shared neighbours are also interface — split equally.
    active = [k for k in getattr(v_i, 'interface_phases', frozenset()) if k >= 0]
    if not active:
        active = [0]
    share = 1.0 / len(active)
    return {k: share for k in active}
