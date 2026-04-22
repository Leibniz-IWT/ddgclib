"""Interface subcomplex extraction and validation for multiphase meshes.

The sharp interface between two phases is a **primal subcomplex** of
the bulk mesh, stored as plain set attributes directly on the
``Complex`` object:

- ``HC.interface_vertices`` — set of vertex objects on the interface.
- ``HC.interface_edges`` — set of ``frozenset({v_a.x, v_b.x})``
  coordinate-key pairs identifying primal edges on the interface.
- ``HC.interface_triangles`` — set of ``frozenset({v_a.x, v_b.x, v_c.x})``
  (3D only) identifying primal face triangles on the interface.

These are populated by :func:`extract_interface` and consumed by
downstream operators (surface tension, dual-volume split, stress).

The extraction is deterministic from the per-top-simplex phase
labels in :attr:`MultiphaseSystem.simplex_phase`.

Functions
---------
- :func:`extract_interface` — populate ``HC.interface_*`` attributes.
- :func:`validate_closure` — check the interface is a closed manifold.
- :func:`interface_nn` — interface-restricted neighbours of a vertex.
- :func:`curve_neighbours` — the two polyline neighbours (2D only).
"""
from __future__ import annotations

from ddgclib.multiphase import _simplex_key, iter_top_simplices


# ---------------------------------------------------------------------------
# Extraction: populate HC.interface_* from simplex_phase
# ---------------------------------------------------------------------------


def _all_faces(simplex, face_size: int):
    """Yield all ``face_size``-vertex faces of a top-simplex (as tuples)."""
    n = len(simplex)
    if face_size == 2:
        for i in range(n):
            for j in range(i + 1, n):
                yield (simplex[i], simplex[j])
    elif face_size == 3:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    yield (simplex[i], simplex[j], simplex[k])


def extract_interface(
    HC,
    simplex_phase: dict[frozenset, int],
    dim: int,
) -> None:
    """Populate ``HC.interface_vertices/edges/triangles`` from simplex labels.

    A primal face (edge in 2D, triangle in 3D) is on the interface iff
    it is shared by exactly two top-simplices of different phases.

    Parameters
    ----------
    HC : Complex
        The interface attributes are set in-place on this object.
    simplex_phase : dict
        Per-top-simplex phase labels keyed by frozenset of vertex
        coordinate tuples.
    dim : int
        Spatial dimension (2 or 3).
    """
    if dim not in (2, 3):
        raise ValueError(f"extract_interface supports dim ∈ {{2, 3}}; got {dim}")

    face_size = dim  # edge (2) in 2D, triangle (3) in 3D
    # Map: canonical face key -> list of (phase, face-as-tuple-of-verts)
    incident_phases: dict[frozenset, list] = {}

    for top in iter_top_simplices(HC, dim):
        top_key = _simplex_key(top)
        phase = simplex_phase.get(top_key)
        if phase is None:
            continue
        for face in _all_faces(top, face_size):
            fkey = frozenset(v.x for v in face)
            incident_phases.setdefault(fkey, []).append((phase, face))

    iface_edges: set[frozenset] = set()
    iface_tris: set[frozenset] = set()
    iface_verts: set = set()

    for fkey, entries in incident_phases.items():
        phases = {ph for (ph, _) in entries}
        if len(phases) < 2:
            continue
        _, face_verts = entries[0]

        if dim == 2:
            iface_edges.add(fkey)
        else:
            iface_tris.add(fkey)

        for v in face_verts:
            iface_verts.add(v)

    # In 3D also collect the edge set of the interface triangles.
    if dim == 3:
        for tri in iface_tris:
            vkeys = list(tri)
            for i in range(3):
                for j in range(i + 1, 3):
                    iface_edges.add(frozenset({vkeys[i], vkeys[j]}))

    HC.interface_vertices = iface_verts
    HC.interface_edges = iface_edges
    HC.interface_triangles = iface_tris


# ---------------------------------------------------------------------------
# Neighbourhood helpers
# ---------------------------------------------------------------------------


def interface_nn(v, HC) -> set:
    """Return the interface-restricted neighbours of ``v``.

    Equivalent to ``v.nn & HC.interface_vertices``.
    """
    return v.nn & HC.interface_vertices


def curve_neighbours(v, HC):
    """Return the two interface-polyline neighbours of ``v`` (2D only).

    For a 2D interface vertex, there are exactly two incident interface
    edges (closure invariant).  Returns ``(v_prev, v_next)`` where
    ordering is arbitrary but stable within one extraction.

    Returns ``(None, None)`` if ``v`` is not an interface vertex or
    if the vertex doesn't have exactly 2 polyline neighbours.
    """
    iface_verts = getattr(HC, 'interface_vertices', set())
    iface_edges = getattr(HC, 'interface_edges', set())
    if v not in iface_verts:
        return (None, None)
    nbs = []
    for nb in v.nn:
        if nb in iface_verts:
            if frozenset({v.x, nb.x}) in iface_edges:
                nbs.append(nb)
    if len(nbs) != 2:
        return (None, None)
    return (nbs[0], nbs[1])


# ---------------------------------------------------------------------------
# Closure validation
# ---------------------------------------------------------------------------


def validate_closure(
    HC,
    dim: int,
    boundary_vertices: set | None = None,
) -> None:
    """Assert that the interface on ``HC`` is a closed manifold.

    - 2D: every **interior** interface vertex has exactly 2 incident
      interface edges.
    - 3D: every interior interface edge is shared by exactly 2
      interface triangles.

    Vertices in ``boundary_vertices`` are contact points on the domain
    wall and are excluded from the degree check.

    Reads from ``HC.interface_vertices``, ``HC.interface_edges``,
    ``HC.interface_triangles`` (set by :func:`extract_interface`).

    Raises
    ------
    ValueError
        With a diagnostic identifying the offending vertex/edge.
    """
    bvs = boundary_vertices or set()
    iface_edges = getattr(HC, 'interface_edges', set())

    if dim == 2:
        iface_verts = getattr(HC, 'interface_vertices', set())
        for v in iface_verts:
            if v in bvs:
                continue
            # Count interface edges incident to v.
            degree = sum(
                1 for nb in v.nn
                if nb in iface_verts
                and frozenset({v.x, nb.x}) in iface_edges
            )
            if degree != 2:
                raise ValueError(
                    f"Interface closure violated in 2D: vertex at "
                    f"{tuple(v.x_a)} has degree {degree} "
                    f"(expected 2)."
                )
        return

    if dim == 3:
        iface_tris = getattr(HC, 'interface_triangles', set())
        edge_tri_count: dict[frozenset, int] = {}
        for tri in iface_tris:
            xkeys = list(tri)
            for i in range(3):
                for j in range(i + 1, 3):
                    ekey = frozenset({xkeys[i], xkeys[j]})
                    edge_tri_count[ekey] = edge_tri_count.get(ekey, 0) + 1
        bv_xkeys = {v.x for v in bvs} if bvs else set()
        for ekey, count in edge_tri_count.items():
            if count == 2:
                continue
            if ekey & bv_xkeys:
                continue
            raise ValueError(
                f"Interface closure violated in 3D: edge "
                f"{list(ekey)} shared by {count} interface triangles "
                f"(expected 2)."
            )
        return

    raise ValueError(f"validate_closure supports dim ∈ {{2, 3}}; got {dim}")
