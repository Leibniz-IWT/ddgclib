"""
Cauchy stress tensor operators for discrete fluid dynamics.

Face-centered integrated FVM formulation.  Forces on each Lagrangian
parcel (dual cell) are computed via Stokes' theorem as surface integrals
over dual flux planes:

    F_i = sum_j (F_p_ij + F_v_ij)

Pressure (face-average, conservative):
    F_p_ij = -0.5 * (p_i + p_j) * A_ij

Viscous (face-centered diffusion):
    F_v_ij = mu * (grad u)_f . A_ij
           = (mu / |d_ij|) * du * (d_hat . A_ij)

This is the "diffusion form" (mu * Laplacian u), which equals
div(mu * (grad u + grad u^T)) for incompressible flow (div u = 0).
The symmetric (transpose) term is omitted because the rank-1 face
gradient has spurious discrete compressibility on non-orthogonal edges.

The old pressure_gradient and velocity_laplacian are special cases:
    pressure-only: sigma = -p * I  (mu = 0)
    Laplacian-only: mu * grad(u) . A  (p = 0)

Additional diagnostic operators are provided for analytical comparison:
    velocity_difference_tensor — integrated Du_i (no /Vol)
    velocity_difference_tensor_pointwise — Du_i / Vol_i
    cauchy_stress — pointwise sigma from pointwise du
    integrated_cauchy_stress — volume-integrated sigma from integrated Du

Constitutive relation TODOs
---------------------------
# TODO: Add viscoelastic constitutive relation (Maxwell/Oldroyd-B)
#   sigma = -p*I + tau_elastic + tau_viscous
# TODO: Add non-Newtonian (power-law / Carreau) viscosity
#   mu_eff = K * |strain_rate|^(n-1)
# TODO: Add elastic solid constitutive relation (Hookean)
#   sigma = C : epsilon  (4th-order stiffness tensor)
# TODO: Add surface tension stress (interface parcels)
#   sigma += gamma * (I - n outer n) * kappa
"""

import numpy as np

from ddgclib.operators._registry import MethodRegistry


# ---------------------------------------------------------------------------
# Geometry: dual area vectors and dual volumes
# TODO: move dual_area_vector and dual_volume to hyperct.ddg (pure geometry)
# ---------------------------------------------------------------------------

#: Sign rules of the 2D dual face vector (method axis ``area_orientation``
#: of :mod:`ddgclib.methods`).  ``'primal_edge'``: ``A_ij . (x_j - x_i) > 0``,
#: exact for any pair of non-degenerate triangles (the dual segment crosses
#: the primal edge, laneO).  ``'dual_midpoint'``: the rule before laneO,
#: the vector points away from ``x_i`` as seen from the midpoint of the
#: dual segment; wrong (flipped) when the two triangles at the edge subtend
#: more than 180 degrees at ``x_i`` (status 'broken', kept so that the
#: numbers pinned before the fix can be reproduced).
AREA_ORIENTATIONS = ('primal_edge', 'dual_midpoint')


def _orient_2d(A_ij: np.ndarray, x_i: np.ndarray, x_j: np.ndarray,
               centroid: np.ndarray, orientation: str) -> np.ndarray:
    """Sign of a 2D dual face vector so that it points outward from i."""
    if orientation == 'primal_edge':
        if np.dot(A_ij, x_j - x_i) < 0:
            return -A_ij
        return A_ij
    if orientation == 'dual_midpoint':
        if np.dot(A_ij, x_i - centroid) > 0:
            return -A_ij
        return A_ij
    raise KeyError(f"unknown area orientation {orientation!r}; available: "
                   f"{AREA_ORIENTATIONS}")


def dual_area_vector(v_i, v_j, HC, dim: int = 3,
                     orientation: str = 'primal_edge') -> np.ndarray:
    """Oriented dual area vector for the interface between parcels i and j.

    Computes A_ij, the total outward area vector of the dual face separating
    the dual cells of v_i and v_j.  In the continuum limit this is:

        A_ij = int_{S_ij} n dS

    where n is the outward unit normal from parcel i.

    In 2D the dual face is the line segment between the two shared dual
    vertices; A_ij is the outward-facing normal with magnitude equal to the
    segment length.  Its sign is fixed by *orientation* (see
    :data:`AREA_ORIENTATIONS`): the default ``'primal_edge'`` takes the
    normal on the side of ``x_j`` (``A_ij . d_ij > 0``, which is
    ``(2/3) (|T_left| + |T_right|) > 0`` for the barycentric segment of any
    valid pair of triangles, and ``(2/3) |T|`` for a hull edge), so that
    ``A_ij = -A_ji`` and the cell of an interior vertex closes on every
    mesh.  ``'dual_midpoint'`` is the legacy rule (broken on skewed meshes).
    1D and 3D ignore *orientation*.

    In 3D the dual face is the DEC p_ij polygon: tet barycenters interleaved
    with face barycenters (x_i + x_j + x_k)/3.  This construction guarantees
    linear precision (machine eps) for barycentric duals on any tetrahedral
    mesh.  See :func:`_dual_area_vector_3d_p_ij`.  Falls back to the legacy
    e_star fan-walk (:func:`_dual_area_vector_3d_e_star`) for boundary or
    degenerate edges.

    Parameters
    ----------
    v_i, v_j : vertex objects
        Endpoints of the primal edge.  Must have ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed (``compute_vd``).
    dim : int
        Spatial dimension (1, 2, or 3).
    orientation : {'primal_edge', 'dual_midpoint'}
        2D sign rule (above).

    Returns
    -------
    np.ndarray
        Oriented area vector, shape ``(dim,)``.

    Notes
    -----
    # TODO: move to hyperct.ddg._operators (pure geometry, no physics)
    """
    if dim == 1:
        # In 1D the "area vector" is a signed scalar direction (+/- 1)
        # pointing outward from v_i along the primal edge
        direction = v_j.x_a[0] - v_i.x_a[0]
        return np.array([np.sign(direction)])

    elif dim == 2:
        # When periodic axes are set, always compute dual area from primal
        # geometry using minimum-image coordinates.  compute_vd's dual
        # vertex positions are wrong for any triangle that includes a
        # periodic-face vertex with wrapped neighbors.
        periodic_axes = getattr(HC, '_periodic_axes', None)
        if periodic_axes:
            periodic_bounds = HC._periodic_bounds
            x_i = v_i.x_a[:2]

            def _min_image(x_other):
                result = x_other.copy()
                for ax in periodic_axes:
                    p = periodic_bounds[ax][1] - periodic_bounds[ax][0]
                    delta = result[ax] - x_i[ax]
                    result[ax] -= round(delta / p) * p
                return result

            x_j = _min_image(v_j.x_a[:2])
            common = v_i.nn.intersection(v_j.nn)
            if len(common) < 2:
                # Boundary edge: use single triangle + edge midpoint
                if len(common) < 1:
                    return np.zeros(2)
                v3 = list(common)[0]
                x3 = _min_image(v3.x_a[:2])
                bary = (x_i + x_j + x3) / 3.0
                midpt = 0.5 * (x_i + x_j)
                dual_edge = bary - midpt
                A_ij = np.array([-dual_edge[1], dual_edge[0]])
                return _orient_2d(A_ij, x_i, x_j, 0.5 * (bary + midpt),
                                  orientation)
            # Interior edge: pick two triangles (one on each side of edge).
            # With ghost resolution, shared count can be >2. Use cross
            # product sign to find one neighbor on each side.
            edge_vec = x_j - x_i
            left = None
            right = None
            for v3 in common:
                x3 = _min_image(v3.x_a[:2])
                cross = edge_vec[0] * (x3[1] - x_i[1]) - edge_vec[1] * (x3[0] - x_i[0])
                if cross > 0 and left is None:
                    left = x3
                elif cross <= 0 and right is None:
                    right = x3
                if left is not None and right is not None:
                    break
            if left is None or right is None:
                return np.zeros(2)
            bary_l = (x_i + x_j + left) / 3.0
            bary_r = (x_i + x_j + right) / 3.0
            dual_edge = bary_l - bary_r
            A_ij = np.array([-dual_edge[1], dual_edge[0]])
            return _orient_2d(A_ij, x_i, x_j, 0.5 * (bary_l + bary_r),
                              orientation)

        # Standard (non-periodic) path
        vdnn = v_i.vd.intersection(v_j.vd)
        vd_list = list(vdnn)
        if len(vd_list) < 2:
            # Degenerate: boundary edge with single dual vertex
            return np.zeros(2)
        vd1, vd2 = vd_list[0], vd_list[1]
        dual_edge = vd2.x_a[:2] - vd1.x_a[:2]
        # Normal to dual edge: rotation of the dual edge direction vector
        A_ij = np.array([-dual_edge[1], dual_edge[0]])
        # Orient outward from v_i
        return _orient_2d(A_ij, v_i.x_a[:2], v_j.x_a[:2],
                          0.5 * (vd1.x_a[:2] + vd2.x_a[:2]), orientation)

    elif dim == 3:
        return _dual_area_vector_3d_p_ij(v_i, v_j, HC)

    else:
        raise NotImplementedError(f"dual_area_vector not implemented for dim={dim}")


def _dual_area_vector_3d_p_ij(v_i, v_j, HC) -> np.ndarray:
    """3D dual area vector using the DEC p_ij face construction.

    The dual face polygon interleaves **tet barycenters** with **face
    barycenters** ``(x_i + x_j + x_k) / 3`` of the primal triangular
    faces shared by consecutive tetrahedra around edge ``(i, j)``.

    This gives linear precision (machine epsilon) for barycentric duals
    on any tetrahedral mesh — a property that the simpler polygon of
    tet-barycenters-only does not satisfy because the non-planar face
    produces an incorrect area vector when triangulated without the
    intermediate face-barycenter vertices.

    Falls back to :func:`_dual_area_vector_3d_e_star` when the
    ring-walk or common-neighbor lookup fails (boundary / degenerate
    topologies).
    """
    shared_vd = v_i.vd.intersection(v_j.vd)
    if len(shared_vd) < 3:
        # Boundary or degenerate — fall back
        return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    # --- Build ring order from dual vertex connectivity ---
    shared_list = list(shared_vd)
    ring = [shared_list[0]]
    remaining = set(shared_list[1:])
    while remaining:
        curr = ring[-1]
        nxt = None
        for cand in remaining:
            if cand in curr.nn:
                nxt = cand
                break
        if nxt is None:
            break
        ring.append(nxt)
        remaining.discard(nxt)

    if len(ring) < len(shared_list):
        # Connectivity walk incomplete — fall back to angular sort.
        # This happens for boundary-adjacent edges where dual vertices
        # have truncated connectivity.
        from hyperct.ddg._dual_cell import _angular_sort_3d
        sorted_ring = _angular_sort_3d(shared_list)
        if sorted_ring is not None and len(sorted_ring) >= 3:
            ring = sorted_ring
        elif len(ring) < 3:
            return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    # --- Common neighbors = opposite vertices of shared triangular faces ---
    common_nbs = list(v_i.nn.intersection(v_j.nn))
    if not common_nbs:
        return _dual_area_vector_3d_e_star(v_i, v_j, HC)

    x_i = v_i.x_a[:3]
    x_j = v_j.x_a[:3]

    # --- Build interleaved polygon: tet_bary, face_bary, ... ---
    interleaved = []
    for k in range(len(ring)):
        tet_bary = ring[k].x_a[:3]
        tet_next = ring[(k + 1) % len(ring)].x_a[:3]
        interleaved.append(tet_bary)

        # Find the face barycenter (x_i + x_j + x_k)/3 between these
        # two consecutive tets.  The correct x_k is the common neighbor
        # whose face barycenter lies closest to the midpoint of the two
        # tet barycenters.
        mid = 0.5 * (tet_bary + tet_next)
        best_fb = None
        best_dist = np.inf
        for cn in common_nbs:
            fb = (x_i + x_j + cn.x_a[:3]) / 3.0
            dist = np.linalg.norm(fb - mid)
            if dist < best_dist:
                best_dist = dist
                best_fb = fb
        if best_fb is not None:
            interleaved.append(best_fb)

    polygon = np.array(interleaved)

    # --- Compute area vector by centroid-fan triangulation ---
    centroid = polygon.mean(axis=0)
    A_ij = np.zeros(3)
    n_pts = len(polygon)
    for k in range(n_pts):
        p1 = polygon[k]
        p2 = polygon[(k + 1) % n_pts]
        A_ij += 0.5 * np.cross(p1 - centroid, p2 - centroid)

    # --- Orient outward from v_i ---
    d_ij = x_j - x_i
    if np.dot(A_ij, d_ij) < 0:
        A_ij = -A_ij
    return A_ij


def _dual_area_vector_3d_e_star(v_i, v_j, HC) -> np.ndarray:
    """3D dual area vector via the e_star fan-walk (legacy).

    Uses the ``e_star`` fan-walk triangulation from the primal edge
    midpoint through the shared dual vertices.  This was the original
    implementation.  It does NOT satisfy linear precision for barycentric
    duals on non-symmetric meshes because the non-planar face polygon
    (tet barycenters only, without interleaved face barycenters) produces
    an incorrect area vector.

    Kept as fallback for boundary / degenerate topologies where the
    p_ij ring-walk cannot be performed.
    """
    from hyperct.ddg import e_star as _e_star
    try:
        A_ijk_arr = _e_star(v_i, v_j, HC, dim=3)  # shape (N, 3)
    except (IndexError, KeyError):
        return np.zeros(3)
    if not isinstance(A_ijk_arr, np.ndarray) or A_ijk_arr.size == 0:
        return np.zeros(3)
    vc_12_pos = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
    vc_12 = HC.Vd[tuple(vc_12_pos)]
    vec_to_i = v_i.x_a - vc_12.x_a
    A_ij = np.zeros(3)
    for A_ijk in A_ijk_arr:
        if np.dot(A_ijk, vec_to_i) > 0:
            A_ijk = -A_ijk
        A_ij += A_ijk
    return A_ij


def _use_exact_barycentric_volume(HC) -> bool:
    """True when the exact simplex-based barycentric dual volume applies.

    Requires (a) the explicit top-simplex cache ``HC._simplices`` and
    (b) barycentric duals — the closed form
    ``Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|`` holds only for the
    barycentric dual partition.  ``HC._vd_method`` is recorded by
    ``hyperct.ddg.compute_vd``; when absent, the pipeline default
    (barycentric) is assumed.
    """
    return (
        getattr(HC, '_simplices', None) is not None
        and getattr(HC, '_vd_method', 'barycentric') == 'barycentric'
    )


def dual_volume(v, HC, dim: int = 3) -> float:
    """Volume (area in 2D) of the dual cell around vertex v.

    The dual cell is the Voronoi-like polyhedron (3D) or polygon (2D) whose
    boundary consists of the dual faces separating v from each neighbor.

        Vol_i = dual cell measure of parcel i

    In 2D this is the dual cell area; in 3D the dual cell volume.

    Parameters
    ----------
    v : vertex object
        Must have ``v.nn`` and ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension (1, 2, or 3).

    Returns
    -------
    float
        Dual cell volume (2D: area, 3D: volume).

    Notes
    -----
    For barycentric duals with an explicit simplex cache
    (``HC._simplices``), dim 2 AND dim 3 use the exact closed form
    ``Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|`` from
    ``hyperct.ddg.vertex_dual_volume`` — exact to machine precision,
    tiles the domain including boundary/corner cells, and has no
    degenerate/exception paths.  The legacy geometric reconstruction
    (``dual_cell_area_2d`` / ``v_star`` fan walk) is kept for
    circumcentric duals and as fallback when no simplex cache exists.
    The 3D exact path was enabled 2026-07-29 (lane A), together with a
    canonical 3D qhull input order in hyperct
    ``connect_and_cache_simplices``; the 3D droplet retopology floor
    was re-pinned 7.3768e-5 -> 7.274172e-5 accordingly — see the
    NOTE(lane3-dual-volume) below.
    """
    if dim == 1:
        # 1D dual cell = interval between the two dual vertices
        vd_list = list(v.vd)
        if len(vd_list) < 2:
            # Boundary vertex with single dual vertex: half-edge
            if len(vd_list) == 1 and v.nn:
                v_j = next(iter(v.nn))
                return 0.5 * abs(v_j.x_a[0] - v.x_a[0])
            return 0.0
        positions = [vd.x_a[0] for vd in vd_list]
        return max(positions) - min(positions)

    elif dim == 2:
        if _use_exact_barycentric_volume(HC):
            from hyperct.ddg import vertex_dual_volume
            return vertex_dual_volume(HC, v, dim=2)
        from hyperct.ddg import dual_cell_area_2d
        return dual_cell_area_2d(v, include_edge_midpoints=True)

    elif dim == 3:
        # NOTE(lane3-dual-volume): exact simplex path ENABLED 2026-07-29
        # (lane A), mirroring the dim==2 branch.  All three switch
        # points (this branch, cache_dual_volumes dim in (2, 3), and
        # the _integrators_dynamic.py step-5b batch_e_star preference)
        # flipped TOGETHER — mixed volume sources across setup/retopo
        # create a first-retopo pressure jump much larger than either
        # consistent choice.  Alone, the switch moves the pinned 3D
        # static-droplet retopology floor 7.3768e-5 -> 7.616854e-5
        # (+3.3%): the exact measure honestly reports the larger real
        # settle-step volume jump that the redistribution rescale
        # converts into a uniform pressure offset (order-dependent
        # qhull tie-breaking at retopo #1-2, NOT a volume bug).  The
        # companion canonical 3D qhull input order in hyperct
        # connect_and_cache_simplices (NOTE(laneA-canonical-order))
        # kills that settle artifact; the floor is pinned at
        # 7.274172e-5 (below the old fan floor).  See
        # docs_temp/debug_session/lane3-exact-dual-volumes.md.
        if _use_exact_barycentric_volume(HC):
            from hyperct.ddg import vertex_dual_volume
            return vertex_dual_volume(HC, v, dim=3)
        # Legacy fan walk (circumcentric / no simplex cache): known to
        # undercount 1-4% interior / ~20% boundary on unstructured
        # meshes (audit/dual-volume-3d.md).
        from hyperct.ddg import v_star as _v_star
        total_vol = 0.0
        for v_j in v.nn:
            try:
                result = _v_star(v, v_j, HC, dim=3)
                if isinstance(result, tuple) and len(result) == 2:
                    _, V_ij = result
                    total_vol += np.sum(np.abs(V_ij))
                else:
                    # Scalar return (shouldn't happen in 3D)
                    total_vol += float(result)
            except (KeyError, IndexError, ValueError):
                continue
        return total_vol

    else:
        raise NotImplementedError(f"dual_volume not implemented for dim={dim}")


# ---------------------------------------------------------------------------
# Dual volume caching
# ---------------------------------------------------------------------------

def cache_dual_volumes(HC, dim: int = 3) -> None:
    """Compute and cache dual cell volumes on all vertices.

    Sets ``v.dual_vol = dual_volume(v, HC, dim)`` for every vertex in
    ``HC.V``.  Should be called after ``compute_vd`` (e.g. inside
    ``_retopologize``) so that operators can read ``v.dual_vol``
    instead of recomputing on the fly.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.
    """
    if dim in (2, 3) and _use_exact_barycentric_volume(HC):
        # Exact barycentric dual volumes in one vectorized pass over the
        # simplex cache (Vol_i = (1/(dim+1)) * sum_{T ∋ i} |T|).
        # dim==3 enabled 2026-07-29 together with the dual_volume
        # dim==3 branch and the _integrators_dynamic.py step-5b
        # preference — see the NOTE(lane3-dual-volume) in dual_volume
        # above.
        from hyperct.ddg import simplex_dual_volumes
        vols = simplex_dual_volumes(HC, dim)
        for v in HC.V:
            v.dual_vol = vols.get(v, 0.0)
        return

    for v in HC.V:
        try:
            v.dual_vol = dual_volume(v, HC, dim)
        except (ValueError, IndexError):
            # Degenerate vertex (e.g. domain corner with too few neighbors)
            v.dual_vol = 0.0


def _get_dual_vol(v, HC, dim: int = 3) -> float:
    """Return cached dual volume, computing it on demand if missing."""
    try:
        return v.dual_vol
    except AttributeError:
        v.dual_vol = dual_volume(v, HC, dim)
        return v.dual_vol


# ---------------------------------------------------------------------------
# Physics: velocity difference tensor, strain rate, stress
# ---------------------------------------------------------------------------

def velocity_difference_tensor(v, HC, dim: int = 3) -> np.ndarray:
    """Discrete integrated velocity difference tensor Du_i at vertex v.

    Computes the DDG volume-integrated quantity:

        Du_i = 0.5 * sum_j (u_j - u_i) outer A_ij

    This is analogous to int_{V_i} grad(u) dV (NOT divided by Vol_i).
    It is the natural integrated discrete form — not a pointwise gradient
    approximation.

    To get the pointwise gradient approximation, use
    :func:`velocity_difference_tensor_pointwise` or divide by
    ``v.dual_vol``.

    Parameters
    ----------
    v : vertex object
        Must have ``v.u`` (velocity ndarray), ``v.nn``, ``v.vd``.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Integrated velocity difference tensor, shape ``(dim, dim)``.
        Component ``Du_i[a, b] = 0.5 * sum_j (u_j^a - u_i^a) * A_ij^b``.
    """
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    Du_i = np.zeros((dim, dim))
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_u = v_j.u[:dim] - v.u[:dim]
        Du_i += np.outer(delta_u, A_ij)
    Du_i *= 0.5
    return Du_i


def velocity_difference_tensor_pointwise(v, HC, dim: int = 3) -> np.ndarray:
    """Pointwise velocity gradient approximation at vertex v.

    Returns ``Du_i / Vol_i`` — the volume-averaged velocity gradient.
    Useful for comparison with analytical solutions.

    Parameters
    ----------
    v : vertex object
        Must have ``v.u``, ``v.nn``, ``v.vd``.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Pointwise velocity gradient, shape ``(dim, dim)``.
    """
    Vol_i = _get_dual_vol(v, HC, dim)
    if Vol_i < 1e-30:
        return np.zeros((dim, dim))
    return velocity_difference_tensor(v, HC, dim) / Vol_i


def scalar_gradient_integrated(
    v,
    HC,
    dim: int = 3,
    field_attr: str = 'f',
) -> np.ndarray:
    """Integrated gradient of a scalar field over the dual cell of v.

    Computes the DDG volume-integrated quantity::

        Df_i = 0.5 * sum_j (f_j - f_i) * A_ij

    This is the scalar analog of :func:`velocity_difference_tensor`.
    It approximates ``∫_{V_i} ∇f dV``.

    Parameters
    ----------
    v : vertex object
        Must have the scalar field attribute (default ``v.f``) and
        ``v.nn``, ``v.vd`` populated.
    HC : Complex
        Simplicial complex with duals computed.
    dim : int
        Spatial dimension.
    field_attr : str
        Name of the scalar field attribute on vertices (default ``'f'``).

    Returns
    -------
    np.ndarray
        Integrated gradient vector, shape ``(dim,)``.
    """
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    f_i = getattr(v, field_attr)
    Df_i = np.zeros(dim)
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_f = getattr(v_j, field_attr) - f_i
        Df_i += delta_f * A_ij
    Df_i *= 0.5
    return Df_i


def strain_rate(du: np.ndarray) -> np.ndarray:
    """Symmetric strain rate tensor from velocity difference tensor.

    Computes the symmetric part:

        epsilon = 0.5 * (du + du^T)

    For an incompressible Newtonian fluid, the deviatoric stress is:

        tau = 2 * mu * epsilon

    Parameters
    ----------
    du : np.ndarray
        Velocity difference tensor, shape ``(dim, dim)``.

    Returns
    -------
    np.ndarray
        Symmetric strain rate tensor, shape ``(dim, dim)``.
    """
    return 0.5 * (du + du.T)


def cauchy_stress(
    p: float,
    du: np.ndarray,
    mu: float,
    dim: int = 3,
) -> np.ndarray:
    """Cauchy stress tensor for a Newtonian fluid.

    Constitutive relation:

        sigma = -p * I + tau
        tau = 2 * mu * epsilon
        epsilon = 0.5 * (du + du^T)

    where p is the scalar pressure (positive in compression), mu is the
    dynamic viscosity, and du is the discrete integrated velocity difference
    tensor.

    Parameters
    ----------
    p : float
        Scalar pressure.
    du : np.ndarray
        Velocity difference tensor, shape ``(dim, dim)``.
    mu : float
        Dynamic viscosity [Pa.s].
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Cauchy stress tensor, shape ``(dim, dim)``.
    """
    return -p * np.eye(dim) + 2.0 * mu * strain_rate(du)


def integrated_cauchy_stress(
    p: float,
    Du: np.ndarray,
    mu: float,
    Vol_i: float,
    dim: int = 3,
) -> np.ndarray:
    """Integrated Cauchy stress tensor over the dual cell volume.

    Computes the volume-integrated stress:

        Sigma_int = -p * Vol_i * I + 2 * mu * strain_rate(Du)

    where ``Du`` is the integrated velocity difference tensor (NOT divided
    by volume).  This is the natural discrete quantity — the pointwise
    stress ``cauchy_stress`` is recovered by dividing by ``Vol_i``.

    Parameters
    ----------
    p : float
        Scalar pressure.
    Du : np.ndarray
        Integrated velocity difference tensor, shape ``(dim, dim)``.
        From :func:`velocity_difference_tensor` (without /Vol).
    mu : float
        Dynamic viscosity [Pa.s].
    Vol_i : float
        Dual cell volume.
    dim : int
        Spatial dimension.

    Returns
    -------
    np.ndarray
        Integrated Cauchy stress tensor, shape ``(dim, dim)``.
    """
    return -p * Vol_i * np.eye(dim) + 2.0 * mu * strain_rate(Du)


def _resolve_pressure(v, pressure_model, HC, dim):
    """Resolve the pressure for a single vertex.

    Parameters
    ----------
    v : vertex object
    pressure_model : None, callable, or EquationOfState
        - ``None``: read ``v.p`` as-is (default, incompressible).
        - callable ``fn(v) -> float``: externally defined pressure field.
        - :class:`~ddgclib.eos.EquationOfState`: compute pressure from
          density ``rho = v.m / dual_vol`` via the EOS.  Also updates
          ``v.p`` and ``v.rho`` in-place so downstream code sees fresh
          values.
    HC : Complex
    dim : int

    Returns
    -------
    float
        Pressure at the vertex.
    """
    if pressure_model is None:
        p = v.p
        return float(p) if np.ndim(p) == 0 else float(p[0])

    if callable(pressure_model) and not hasattr(pressure_model, 'pressure'):
        # Plain callable: fn(v) -> float
        return float(pressure_model(v))

    # EquationOfState: P = eos.pressure(m / dual_vol)
    # Uses cached dual_vol (updated by _retopologize at each step).
    vol = _get_dual_vol(v, HC, dim)
    if vol < 1e-30:
        return float(pressure_model.pressure(pressure_model.rho0))
    rho = v.m / vol
    p = float(pressure_model.pressure(rho))
    # Update vertex in-place so other code (callbacks, diagnostics) sees it
    v.rho = rho
    v.p = p
    return p


# ---------------------------------------------------------------------------
# Factored flux primitives (shared by stress_force and multiphase_stress)
# ---------------------------------------------------------------------------

def pressure_flux(p_i: float, p_j: float, A_ij: np.ndarray) -> np.ndarray:
    """Face-average pressure flux: F_p_ij = -0.5 * (p_i + p_j) * A_ij."""
    return -0.5 * (p_i + p_j) * A_ij


def pressure_flux_riemann(p_i: float, p_j: float, rho_i: float, rho_j: float,
                          c_i: float, c_j: float, u_i: np.ndarray, u_j: np.ndarray,
                          A_ij: np.ndarray) -> np.ndarray:
    """Acoustic-Riemann (Lagrangian Godunov) contact pressure flux.

    ::

        p*_ij  = 0.5 (p_i + p_j) - 0.5 rho_f c_f (u_j - u_i) . n_ij
        F_p_ij = -p*_ij A_ij,     rho_f = 0.5 (rho_i + rho_j),  c_f = 0.5 (c_i + c_j)

    The velocity-jump term is the contact pressure of the linearised
    (acoustic) Riemann problem across the dual face: cells separating
    along the face normal see a lower face pressure and are pushed back
    together, cells approaching see a higher one.  It is pairwise
    antisymmetric (momentum conserving), vanishes for rigid translation
    and for any velocity field with no jump normal to the face, and
    damps the grid-scale acoustic (checkerboard) velocity mode that the
    centred flux cannot see.  Its price is a numerical bulk viscosity of
    order ``rho c |d_ij|`` on compressive modes, which at low Mach
    number can exceed the physical viscosity (use the density-diffusion
    stabilisation for density noise instead).
    """
    An = float(np.linalg.norm(A_ij))
    if An == 0.0:
        return np.zeros_like(A_ij)
    w = float((u_j - u_i) @ A_ij) / An          # normal velocity jump
    p_star = 0.5 * (p_i + p_j) - 0.25 * (rho_i + rho_j) * 0.5 * (c_i + c_j) * w
    return -p_star * A_ij


pressure_flux_methods = MethodRegistry("pressure_flux")
pressure_flux_methods.register("centred", pressure_flux)
pressure_flux_methods.register("acoustic-riemann", pressure_flux_riemann)
# stress_force takes the method KEY as a keyword named ``pressure_flux``,
# which shadows the function inside that scope: keep an alias.
_pressure_flux_centred = pressure_flux


def viscous_flux(
    mu: float,
    delta_u: np.ndarray,
    d_ij: np.ndarray,
    A_ij: np.ndarray,
) -> np.ndarray:
    """Face-centered viscous diffusion flux.

    F_v_ij = (mu / |d_ij|) * delta_u * (d_hat . A_ij)
    """
    d_norm = float(np.linalg.norm(d_ij))
    if d_norm < 1e-30:
        return np.zeros_like(A_ij)
    d_hat = d_ij / d_norm
    return (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)


# ---------------------------------------------------------------------------
# Simplex-gradient fluxes (piecewise-linear reconstruction on the primal
# simplices, integrated over the barycentric dual cell)
# ---------------------------------------------------------------------------

# A simplex whose measure is below _SIMPLEX_FLAT_TOL * (shortest edge)**dim
# is left out of the simplex-gradient viscous force: its gradient is not
# defined (flat) or its stiffness, ~ 1 / thickness, is beyond an explicit
# integrator (see viscous_force_simplex_gradient).
_SIMPLEX_FLAT_TOL = 1e-3
_FACTORIAL = {2: 2.0, 3: 6.0}


def _vertex_simplices(HC) -> dict:
    """``{id(v): [top simplices containing v]}`` for ``HC._simplices``.

    Cached on ``HC._vertex_simplices`` together with the simplex list it
    was built from.  ``HC._simplices`` is replaced, never edited in place,
    whenever the connectivity changes, so the identity of the list says
    whether the map is current.  Only the incidence is cached; positions
    are read fresh by the caller.
    """
    simplices = getattr(HC, '_simplices', None)
    if simplices is None:
        raise ValueError(
            "the 'simplex_gradient' fluxes need the top-simplex cache "
            "HC._simplices (a Delaunay retopology or a domain builder "
            "provides it); it is None")
    cached = getattr(HC, '_vertex_simplices', None)
    if cached is not None and cached[0] is simplices:
        return cached[1]
    incidence: dict = {}
    for s in simplices:
        for w in s:
            incidence.setdefault(id(w), []).append(s)
    HC._vertex_simplices = (simplices, incidence)
    return incidence


def _simplex_fan(v, HC, dim: int):
    """Geometry of the top simplices at *v*: ``(verts, idx, X, vol, b)``.

    *verts* are the other vertices of the ``n`` simplices that contain
    *v* (each once), ``idx[t, k]`` is the position in *verts* of the
    ``k``-th other vertex of simplex ``t``, ``X[t, k] = x_k - x_v``,
    ``vol[t]`` the measure of the simplex and ``b[t, k] = |T|
    grad(phi_k)`` with ``phi_k`` the barycentric coordinate of that
    vertex.  ``-b[t, k]`` is the outward area vector of the face opposite
    it divided by ``dim``; it is built from the adjugate of ``X`` (no
    division), so it stays finite on a flat simplex.  ``None`` if no
    simplex contains *v*.
    """
    simplices = _vertex_simplices(HC).get(id(v))
    if not simplices:
        return None
    local: dict = {}
    verts: list = []
    idx = np.empty((len(simplices), dim), dtype=int)
    for t, s in enumerate(simplices):
        k = 0
        for w in s:
            if w is v:
                continue
            j = local.get(id(w))
            if j is None:
                j = local[id(w)] = len(verts)
                verts.append(w)
            idx[t, k] = j
            k += 1
    X = np.array([w.x_a[:dim] for w in verts])[idx] - v.x_a[:dim]
    b = np.empty_like(X)
    if dim == 2:
        det = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1] * X[:, 1, 0]
        b[:, 0, 0] = X[:, 1, 1]
        b[:, 0, 1] = -X[:, 1, 0]
        b[:, 1, 0] = -X[:, 0, 1]
        b[:, 1, 1] = X[:, 0, 0]
    elif dim == 3:
        b[:, 0] = np.cross(X[:, 1], X[:, 2])
        b[:, 1] = np.cross(X[:, 2], X[:, 0])
        b[:, 2] = np.cross(X[:, 0], X[:, 1])
        det = np.einsum('tc,tc->t', X[:, 0], b[:, 0])
    else:
        raise NotImplementedError(
            f"the 'simplex_gradient' fluxes support dim 2 and 3, got {dim}")
    b *= (np.sign(det) / _FACTORIAL[dim])[:, None, None]
    return verts, idx, X, np.abs(det) / _FACTORIAL[dim], b


def simplex_area_vectors(v, HC, dim: int):
    """Exact barycentric dual area vectors of the edges at *v*, from the
    simplex cache: ``(verts, A)`` with ``A[k]`` the vector of the edge
    ``(v, verts[k])``.

    Inside a simplex ``T`` the face between the dual cells of ``i`` and
    ``j`` has the area vector ``|T| (grad(phi_j) - grad(phi_i)) / (dim + 1)``
    (outward from ``i``), so::

        A_ij = 1 / (dim + 1) * sum_{T contains i, j} |T| (grad(phi_j) - grad(phi_i))

    No dual vertex is read and no orientation is chosen: the sign comes
    from the gradients.  Closed (``sum_j A_ij = 0``) at every vertex whose
    simplices surround it, antisymmetric, and the half cell of a hull
    vertex is closed by its hull faces.  An exactly flat simplex
    contributes nothing.

    This is the reference :func:`dual_area_vector` is tested against
    (laneO: equal to the 2D segment and to the 3D ``p_ij`` ring of an
    interior edge to round-off) and the per-vertex form of the registered
    ``edge_area_source='p_ij_simplex'`` (laneJ); no force reads it.
    Needs ``HC._simplices``.
    """
    fan = _simplex_fan(v, HC, dim)
    if fan is None:
        return [], np.zeros((0, dim))
    verts, idx, _, _, b = fan
    # |T| (grad(phi_k) - grad(phi_v)), with grad(phi_v) = -sum_k grad(phi_k)
    piece = b + b.sum(axis=1)[:, None, :]
    A = np.zeros((len(verts), dim))
    np.add.at(A, idx, piece)
    return verts, A / (dim + 1)


def viscous_force_simplex_gradient(v, HC, dim: int, mu: float,
                                   flat_tol: float = _SIMPLEX_FLAT_TOL,
                                   _fan=None) -> np.ndarray:
    """Viscous force on the dual cell of *v* from the simplex gradients.

    The flux through the barycentric dual faces of cell ``i`` with the
    velocity gradient of the piecewise-linear interpolant on each primal
    simplex ``T``::

        F_v_i = mu * sum_{T contains i} G_T . a_iT
        G_T   = sum_{k in T} u_k (x) grad(phi_k)        (constant on T)
        a_iT  = -|T| grad(phi_i)                        (area vector of the
                                                         dual face of i in T)

    ``phi_k`` are the barycentric coordinates of ``T``.  ``a_iT`` is the
    outward vector area of the part of the barycentric dual boundary of
    cell ``i`` that lies inside ``T`` (it equals the outward area vector
    of the face of ``T`` opposite ``i`` divided by ``dim``, whatever the
    interior dual points are).  Written per edge this is
    ``sum_j w_ij (u_j - u_i)`` with ``w_ij = -mu sum_T |T| grad(phi_i) .
    grad(phi_j)``: the cotangent weights in 2D.

    Properties, against the two-point form of :func:`viscous_flux`:

    - LINEAR PRECISION: zero for a linear velocity field at every
      interior vertex of any simplicial mesh.  The two-point form has
      that property only on meshes whose edge stencil is symmetric; on a
      sheared or jittered Delaunay mesh its error is O(|grad u| / h)
      (laneH: residual of a linear field with the wall shear rate of a
      Poiseuille profile 0.6 to 5 of the driving force ``G Vol``, growing
      with refinement).
    - Pairwise antisymmetric (``w_ij = w_ji``): momentum conserving.
    - Negative semi-definite (energy never grows), but ``w_ij`` can be
      negative on an edge whose opposite angles sum to more than 180
      degrees (never on an interior edge of a 2D Delaunay mesh).
    - A hull vertex gets the natural (zero normal gradient) condition.

    Diffusion form (``mu`` Laplacian), like the two-point flux.

    Simplices with ``|T| <= flat_tol * (shortest edge)**dim`` are left
    out.  A flat simplex has no gradient (the coplanar tetrahedra qhull
    returns on a structured mesh, 4 to 8 % of the simplices of a builder
    cylinder), and a nearly flat one couples its vertices with a
    stiffness ``~ 1 / thickness`` that an explicit integrator cannot
    follow.  Leaving one out is a slit of zero width between simplices
    that still share all its vertices: harmless on the hull (another
    triangulation of the boundary), but between interior vertices the
    dual cells no longer close and linear precision is lost at those
    vertices.  Short edges are not filtered: the measure is relative to
    the shortest edge, so a thin simplex between two close vertices is
    kept.

    Needs ``HC._simplices``.
    """
    F = np.zeros(dim)
    fan = _simplex_fan(v, HC, dim) if _fan is None else _fan
    if fan is None:
        return F
    verts, idx, X, vol, b = fan
    # squared edge lengths: the dim edges at v and those among the others
    l2 = np.einsum('tkc,tkc->tk', X, X).min(axis=1)
    for i in range(dim):
        for j in range(i + 1, dim):
            e = X[:, i] - X[:, j]
            l2 = np.minimum(l2, np.einsum('tc,tc->t', e, e))
    keep = vol > flat_tol * l2 ** (0.5 * dim)
    if not keep.any():
        return F
    # w[t, k] = -|T| grad(phi_k) . grad(phi_v), with b_v = -sum_k b_k
    w = (np.einsum('tkc,tc->tk', b, b.sum(axis=1))
         * (keep / np.where(keep, vol, 1.0))[:, None])
    dU = np.array([nb.u[:dim] for nb in verts])[idx] - v.u[:dim]
    return mu * np.einsum('tk,tkc->c', w, dU)


def pressure_force_simplex_gradient(v, HC, dim: int, pressure_model=None,
                                    _fan=None) -> np.ndarray:
    """Pressure force on the dual cell of *v* from the simplex gradients.

    Minus the integral over the barycentric dual cell of the gradient of
    the piecewise-linear pressure (the cell owns ``1 / (dim + 1)`` of
    every simplex at the vertex)::

        F_p_i = - sum_{T contains i} |T| / (dim + 1) * grad(p)_T
              = - 1 / (dim + 1) * sum_T sum_{k in T} (p_k - p_i) |T| grad(phi_k)

    This is the volume form ``-int grad(p) dV``, not the surface form
    ``-int p n dA`` of :func:`pressure_flux`:

    - exact for a linear pressure on ANY simplicial mesh, in 2D and 3D,
      and independent of the dual face areas (the 3D edge-area cache of
      ``batch_e_star`` is not linearly precise, laneJ);
    - zero for a uniform pressure at EVERY vertex, hull vertices
      included.  An open cell therefore feels no ambient pressure: right
      for a prescribed pressure field, wrong for a free surface that an
      EOS pressure should push outwards;
    - total momentum changes by the boundary integral of the
      piecewise-linear pressure only, but the force is not a sum of
      pairwise antisymmetric fluxes.

    ``|T| grad(phi_k)`` is an area vector and stays finite on a flat
    simplex, so nothing is filtered.  Needs ``HC._simplices``.
    """
    F = np.zeros(dim)
    fan = _simplex_fan(v, HC, dim) if _fan is None else _fan
    if fan is None:
        return F
    verts, idx, _, _, b = fan
    dp = (np.array([_resolve_pressure(nb, pressure_model, HC, dim)
                    for nb in verts])[idx]
          - _resolve_pressure(v, pressure_model, HC, dim))
    return -np.einsum('tk,tkc->c', dp, b) / (dim + 1)


pressure_flux_methods.register("simplex_gradient",
                               pressure_force_simplex_gradient)

viscous_flux_methods = MethodRegistry("viscous_flux")
viscous_flux_methods.register("two_point", viscous_flux)
viscous_flux_methods.register("simplex_gradient", viscous_force_simplex_gradient)
# stress_force takes the method KEY as a keyword named ``viscous_flux``
# (as for ``pressure_flux`` above): keep an alias.
_viscous_flux_two_point = viscous_flux


def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None,
                 pressure_model=None, pressure_flux: str = "centred",
                 viscous_flux: str = "two_point",
                 area_orientation: str = "primal_edge") -> np.ndarray:
    """Integrated force on FVM via face-centered fluxes (Stokes' theorem).

    For each dual flux plane between parcels i and j, the force has two
    contributions computed directly from edge data:

    Pressure (face-average, conservative; ``pressure_flux='centred'``):

        F_p_ij = -0.5 * (p_i + p_j) * A_ij

    or, with ``pressure_flux='acoustic-riemann'`` (needs an EOS as
    *pressure_model* for the density and sound speed), the Lagrangian
    Godunov contact pressure of :func:`pressure_flux_riemann`.  The
    registry ``pressure_flux_methods`` lists the available keys; the
    method axis ``pressure_flux`` of :mod:`ddgclib.methods` records them.

    Viscous (face-centered diffusion):

        F_v_ij = mu * (grad u)_f . A_ij
               = (mu / |d_ij|) * du * (d_hat . A_ij)

    where du = u_j - u_i, d_hat = (x_j - x_i) / |x_j - x_i|.

    This is the "diffusion form" of the viscous term (mu * Laplacian u),
    which is equivalent to the full symmetric stress divergence
    div(mu * (grad u + grad u^T)) for incompressible flow (div u = 0).
    The symmetric (transpose) term mu * grad(div u) is omitted because
    the rank-1 face gradient has spurious discrete compressibility.

    With ``viscous_flux='simplex_gradient'`` the viscous part is
    :func:`viscous_force_simplex_gradient` instead (same dual faces, the
    gradient of the piecewise-linear velocity on each primal simplex):
    linearly precise on any mesh, which the two-point form is not.  The
    registry ``viscous_flux_methods`` lists the keys; the method axis
    ``viscous_flux`` of :mod:`ddgclib.methods` records them.
    ``pressure_flux='simplex_gradient'`` is the pressure counterpart
    (:func:`pressure_force_simplex_gradient`, the volume form of the
    pressure force).

    Total: F_i = sum_j (F_p_ij + F_v_ij)

    Parameters
    ----------
    v : vertex object
        Must have ``v.p``, ``v.u``, ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex
        Simplicial complex with duals computed.
    pressure_model : None, callable, or EquationOfState
        Controls how vertex pressure is obtained:

        - ``None`` (default): read ``v.p`` as-is (prescribed or constant).
        - callable ``fn(v) -> float``: externally defined pressure field,
          evaluated each time the force is computed.
        - :class:`~ddgclib.eos.EquationOfState`: weakly compressible
          pressure from density ``rho = m / dual_vol``.  Updates ``v.p``
          and ``v.rho`` in-place.
    pressure_flux : {'centred', 'acoustic-riemann', 'simplex_gradient'}
        Pressure flux formulation (see above).  ``'acoustic-riemann'``
        requires an EOS *pressure_model* (density and sound speed),
        ``'simplex_gradient'`` requires ``HC._simplices``.
    viscous_flux : {'two_point', 'simplex_gradient'}
        Viscous flux formulation (see above).  ``'simplex_gradient'``
        requires ``HC._simplices``.
    area_orientation : {'primal_edge', 'dual_midpoint'}
        Sign rule of the 2D dual face vectors the fluxes read
        (:func:`dual_area_vector`); the method axis ``area_orientation``.

    Returns
    -------
    np.ndarray
        Force vector, shape ``(dim,)``.
    """
    if pressure_flux not in pressure_flux_methods:
        raise KeyError(
            f"unknown pressure_flux {pressure_flux!r}; available: "
            f"{pressure_flux_methods.available()}")
    if viscous_flux not in viscous_flux_methods:
        raise KeyError(
            f"unknown viscous_flux {viscous_flux!r}; available: "
            f"{viscous_flux_methods.available()}")
    two_point = viscous_flux == "two_point"
    simplex_p = pressure_flux == "simplex_gradient"
    riemann = pressure_flux == "acoustic-riemann"
    if riemann and not hasattr(pressure_model, "sound_speed"):
        raise ValueError("pressure_flux='acoustic-riemann' needs an "
                         "EquationOfState pressure_model (density + sound speed)")
    p_i = _resolve_pressure(v, pressure_model, HC, dim)
    u_i = v.u[:dim]
    x_i = v.x_a[:dim]
    if riemann:
        rho_i = v.rho
        c_i = float(pressure_model.sound_speed(rho_i))

    # Use cached oriented edge area vectors when available (set by
    # batch_e_star(..., orient=True) during retopologization).
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    F = np.zeros(dim)
    if simplex_p or not two_point:
        # NOTE(laneH): the simplex-gradient fluxes share the geometry of
        # the simplices at v; with both selected no dual face is read.
        fan = _simplex_fan(v, HC, dim)
        if simplex_p:
            F += pressure_force_simplex_gradient(v, HC, dim, pressure_model,
                                                 _fan=fan)
        if not two_point:
            F += viscous_force_simplex_gradient(v, HC, dim, mu, _fan=fan)
        if simplex_p and not two_point:
            return F

    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim, area_orientation)

        p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i
        if riemann:
            rho_j = v_j.rho
            F += pressure_flux_riemann(p_i, p_j, rho_i, rho_j, c_i,
                                       float(pressure_model.sound_speed(rho_j)),
                                       u_i, v_j.u[:dim], A_ij)
        elif not simplex_p:
            F += _pressure_flux_centred(p_i, p_j, A_ij)
        if two_point:
            F += _viscous_flux_two_point(mu, delta_u, d_ij, A_ij)

    return F


def stress_acceleration(
    v,
    dim: int = 3,
    mu: float = 8.9e-4,
    HC=None,
    pressure_model=None,
    pressure_flux: str = "centred",
    viscous_flux: str = "two_point",
    area_orientation: str = "primal_edge",
) -> np.ndarray:
    """Acceleration from Cauchy stress: a_i = F_stress_i / m_i.

    Newton's second law on Lagrangian parcel i:

        m_i * dv_i/dt = F_stress_i + F_body
        a_i = F_stress_i / m_i

    This is a drop-in replacement for the old ``acceleration()`` function
    and can be used directly as ``dudt_fn`` for the dynamic integrators::

        from functools import partial
        dudt_fn = partial(stress_acceleration, dim=3, mu=1e-3, HC=HC)
        t = euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100)

    For weakly compressible flow with an equation of state::

        from ddgclib.eos import TaitMurnaghan
        eos = TaitMurnaghan(rho0=1000.0, P0=101325.0)
        dudt_fn = partial(stress_acceleration, dim=2, mu=1e-3, HC=HC,
                          pressure_model=eos)

    Parameters
    ----------
    v : vertex object
        Must have ``v.p``, ``v.u``, ``v.m``, ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex
        Simplicial complex with duals computed.
    pressure_model : None, callable, or EquationOfState
        See :func:`stress_force`.
    pressure_flux : {'centred', 'acoustic-riemann'}
        See :func:`stress_force`.
    viscous_flux : {'two_point', 'simplex_gradient'}
        See :func:`stress_force`.
    area_orientation : {'primal_edge', 'dual_midpoint'}
        See :func:`stress_force`.

    Returns
    -------
    np.ndarray
        Acceleration vector, shape ``(dim,)``.
    """
    return stress_force(v, dim=dim, mu=mu, HC=HC,
                        pressure_model=pressure_model,
                        pressure_flux=pressure_flux,
                        viscous_flux=viscous_flux,
                        area_orientation=area_orientation) / v.m


# Simplified alias for use as dudt_fn in dynamic integrators
dudt_i = stress_acceleration
"""Alias for :func:`stress_acceleration`.

Provides simplified notation for the acceleration function used as
``dudt_fn`` in dynamic integrators::

    from ddgclib.operators.stress import dudt_i
    from ddgclib.dynamic_integrators import euler_velocity_only

    # Pass directly with keyword args forwarded by the integrator:
    euler_velocity_only(HC, bV, dudt_i, dt=1e-4, n_steps=100,
                        dim=2, mu=0.1, HC=HC)

    # Or bind parameters with functools.partial:
    from functools import partial
    dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)
    euler_velocity_only(HC, bV, dudt_fn, dt=1e-4, n_steps=100)
"""
