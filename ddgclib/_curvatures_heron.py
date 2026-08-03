"""
This file uses the Heron's formula to compute the curvature which is a much faster
routine than the experimental code.

Note, however, that this might not be latest version. It is taken from the notebook
"Sphere area study p ix [new Dec 2023].ipynb" which might not be the latest version
which was actually validated (which is in lsm?)

"""
import numpy as np


def _apex_via_simplex_cache(HC, vi, vj):
    """Return apex vertex objects for edge (vi, vj), or ``None`` to fall back.

    Uses ``HC._edge_to_apex`` (built lazily from ``HC._simplices``).
    Only valid when the simplex cache holds 2-simplices (triangles,
    3-tuples) — ``hndA_i`` and friends compute *surface* curvature on a
    2-manifold, so a volumetric tet mesh (4-tuples) is not an
    appropriate apex source and we fall back to the legacy
    ``vi.nn.intersection(vj.nn)`` path.

    Returns
    -------
    list of vertex | None
        ``None`` signals the caller to use the legacy nn-intersection
        path.  An empty list means "no triangle in the cache contains
        this edge" (the caller should also fall back, since the
        curvature algorithms expect at least one apex per neighbour).

    Notes
    -----
    The legacy path is unreliable on Delaunay-derived meshes with
    skinny / sliver simplices because the 1-skeleton flag complex can
    contain spurious K_{dim+1} cliques near boundaries — see
    ``grok_1-skeleton-comment.pdf`` and
    :func:`hyperct.ddg.boundary_from_simplices`.
    """
    if HC is None:
        return None
    simplices = getattr(HC, '_simplices', None)
    if not simplices:
        return None
    # Only triangle (3-tuple) simplices are valid for surface-curvature
    # apex enumeration.  Tet caches return apex *pairs* per simplex,
    # which the curvature loops here would silently misuse.
    if len(simplices[0]) != 3:
        return None
    try:
        from hyperct.ddg import get_edge_apex_map
    except ImportError:
        return None
    apex_map = get_edge_apex_map(HC)
    if apex_map is None:
        return None
    return apex_map.get(frozenset((id(vi), id(vj))), [])


def _apex_via_interface_triangles(HC, vi, vj):
    """Return apex vertex objects for edge (vi, vj) on the interface mesh.

    Looks up ``HC.interface_triangles`` (3D set of frozensets of vertex
    coordinate keys; populated by
    :func:`ddgclib.geometry._interface_subcomplex.extract_interface`).

    Returns ``None`` if interface_triangles is not populated, signalling
    caller to fall back.  Returns ``[]`` if no interface triangle
    contains both vi and vj.  Otherwise returns the list of third
    vertices (apex of each interface triangle containing edge ij).
    """
    if HC is None:
        return None
    iface_tris = getattr(HC, 'interface_triangles', None)
    if iface_tris is None:
        return None
    apex_cache = getattr(HC, '_interface_edge_to_apex', None)
    if apex_cache is None:
        # Build coordinate-key -> vertex object lookup once.
        x_to_v = {v.x: v for v in HC.V}
        apex_cache = {}
        for tri_key in iface_tris:
            tri_verts = [x_to_v[xk] for xk in tri_key]
            for a in range(3):
                for b in range(a + 1, 3):
                    va, vb = tri_verts[a], tri_verts[b]
                    apex = tri_verts[3 - a - b]
                    key = frozenset((id(va), id(vb)))
                    apex_cache.setdefault(key, []).append(apex)
        HC._interface_edge_to_apex = apex_cache
    return apex_cache.get(frozenset((id(vi), id(vj))), [])


def HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    """
    Computes the dual edge and dual area using Heron's formula.

    :param e_ij: vector, edge e_ij
    :param l_ij: float, length of edge ij
    :param l_jk: float, length of edge jk
    :param l_ik: float, length of edge ik
    :return: hnda_ijk: vector, curvature vector
             c_ijk: float, dual areas
    """
    lengths = [l_ij, l_jk, l_ik]
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must have use a ≥ b ≥ c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A  # w_ij = abs(w_ij)

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in triangle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def A_i(v, n_i=None, HC=None):
    """
    Compute the discrete normal area of vertex v_i

    :param v: vertex object
    :param HC: optional Complex.  When supplied and ``HC._simplices`` is
        populated, the apex enumeration uses the simplex-aware
        ``HC._edge_to_apex`` cache instead of the legacy
        ``vi.nn.intersection(vj.nn)`` flag-complex path.
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    NdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        # Apex enumeration: prefer simplex cache, fall back to v.nn ∩ v.nn
        apex = _apex_via_simplex_cache(HC, vi, vj)
        if apex is None:
            e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        else:
            # Cache returns list (with possible duplicates if a vertex is
            # apex of >1 simplex containing edge ij).  Triangle meshes
            # produce 1 (boundary) or 2 (interior) distinct apices.
            e_i_int_e_j = list(dict.fromkeys(apex))
        if len(e_i_int_e_j) == 0:
            continue
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        # NOTE: The code below results in the INCORRECT values unless we set
        #      e_ij = - e_ij  # WHY???
        if 1:
            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            # If the wrong direction was chosen, choose the other:
            #  print(f'np.dot(normalized(wedge_ij_ik)[0], n_i) = {np.dot(normalized(wedge_ij_ik)[0], n_i)}')
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                print(f'e_ij_prev = {e_ij}')
                e_ij = vi.x_a - vj.x_a
                # e_ij = vi.x_a - vj.x_a
            #  e_ij = vj.x_a - vi.x_a  # Does not appear to be needed,
            #                          # but more tests need to be done

        if len(e_i_int_e_j) == 1:  # boundary edge
            pass  # ignore for now

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return NdA_i  # , C_i


# TODO: Since sparse arrays are too expensive to recreate and add to,
#      we might want cache edge lengths instead. higher dimensional
#      simplices could be done with a lexigraphic cache.
#      This is simple to parallelise on CPUs, but might be much harder
#      to do on GPUs.

def hndA_i(v, n_i=None, HC=None):
    """
    Compute the mean normal curvature of vertex

    :param v: vertex object
    :param HC: optional Complex.  When supplied and ``HC._simplices`` is
        populated, apex enumeration uses ``HC._edge_to_apex`` instead
        of the legacy flag-complex ``vi.nn.intersection(vj.nn)`` path.
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    HNdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        apex = _apex_via_simplex_cache(HC, vi, vj)
        if apex is None:
            e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        else:
            e_i_int_e_j = list(dict.fromkeys(apex))
        if len(e_i_int_e_j) == 0:
            continue
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        # NOTE: The code below results in the INCORRECT values unless we set
        #      e_ij = - e_ij  # WHY???
        if 0:
            # Discrete vector area:
            # Simplex areas of ijk and normals
            wedge_ij_ik = np.cross(e_ij, e_ik)
            # If the wrong direction was chosen, choose the other:
            #  print(f'np.dot(normalized(wedge_ij_ik)[0], n_i) = {np.dot(normalized(wedge_ij_ik)[0], n_i)}')
            if np.dot(normalized(wedge_ij_ik)[0], n_i) < 0:
                e_ij = vi.x_a - vj.x_a
                # e_ij = vi.x_a - vj.x_a
            #  e_ij = vj.x_a - vi.x_a  # Does not appear to be needed,
            #                          # but more tests need to be done

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


def _pad3(x_a):
    """Pad a coordinate array to 3D (for np.cross compatibility)."""
    if len(x_a) >= 3:
        return x_a[:3]
    out = np.zeros(3)
    out[:len(x_a)] = x_a
    return out


def hndA_i_interface(v, interface_set, n_i=None, HC=None):
    """Mean curvature normal restricted to interface sub-mesh.

    Same algorithm as :func:`hndA_i` but only considers neighbours that are
    in ``interface_set``.  Used by multiphase surface tension to compute
    the curvature of the phase boundary rather than the full mesh.

    Works in both 2D and 3D by padding 2D vectors to 3D so that
    ``np.cross`` (used inside ``HNdC_ijk``) operates correctly.

    Parameters
    ----------
    v : vertex object
        Must have ``v.x_a`` and ``v.nn`` populated.
    interface_set : set
        Set of vertex objects forming the interface sub-mesh.
    n_i : ignored
        Kept for API compatibility.
    HC : Complex, optional
        When supplied and ``HC.interface_triangles`` is populated, apex
        enumeration uses the explicit interface triangle list (the
        primal-subcomplex model) instead of the legacy
        ``vi.nn ∩ vj.nn ∩ interface_set`` flag-complex path.  The
        legacy path can return spurious K_3 cliques on interface
        boundaries adjacent to the domain wall.

    Returns
    -------
    HNdA_i : ndarray, shape (3,)
        Integrated mean curvature normal vector.
    C_i : float
        Dual area of the vertex on the interface.
    """
    HNdA_i = np.zeros(3)
    C_i = 0.0
    vi = v
    for vj in v.nn:
        if vj not in interface_set:
            continue
        apex = _apex_via_interface_triangles(HC, vi, vj)
        if apex is None:
            e_i_int_e_j = vi.nn.intersection(vj.nn).intersection(interface_set)
        else:
            # Filter to interface_set in case caller passes a stricter
            # interface set than HC.interface_triangles spans (e.g. a
            # local sub-region during validation).
            e_i_int_e_j = [vk for vk in dict.fromkeys(apex)
                           if vk in interface_set]
        if len(e_i_int_e_j) == 0:
            continue
        e_ij = _pad3(vj.x_a) - _pad3(vi.x_a)
        e_ij = -e_ij  # Sign convention (matches hndA_i)

        if len(e_i_int_e_j) == 1:
            vk = list(e_i_int_e_j)[0]
            e_ik = _pad3(vk.x_a) - _pad3(vi.x_a)
            e_jk = _pad3(vk.x_a) - _pad3(vj.x_a)
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
            HNdA_i += hnda_ijk
            C_i += c_ijk
        else:
            vk, vl = list(e_i_int_e_j)[:2]
            e_ik = _pad3(vk.x_a) - _pad3(vi.x_a)
            e_jk = _pad3(vk.x_a) - _pad3(vj.x_a)
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)
            l_jk = np.linalg.norm(e_jk)
            hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)

            e_il = _pad3(vl.x_a) - _pad3(vi.x_a)
            e_jl = _pad3(vl.x_a) - _pad3(vj.x_a)
            l_il = np.linalg.norm(e_il)
            l_jl = np.linalg.norm(e_jl)
            hnda_ijl, c_ijl = HNdC_ijk(e_ij, l_ij, l_jl, l_il)

            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijk
            C_i += c_ijl

    return HNdA_i, C_i


def integrated_hndA_i_interface(v, interface_set, HC, gamma=1.0):
    """Integrated surface-tension force on a 3D interface vertex via Stokes.

    Replaces the cotangent-Heron pointwise stencil with a direct
    boundary-integral discretisation::

        F_st_i  =  gamma * integral_{Gamma_i} 2H N dA
                =  gamma * boundary-integral_{partial Gamma_i} nu dl    (Stokes)

    where ``Gamma_i`` is the portion of the interface inside the
    *barycentric* dual cell of ``v_i`` and ``nu`` is the in-surface
    conormal (perpendicular to the boundary, lying in the triangle's
    tangent plane, pointing outward from the dual cell).

    The boundary of the dual cell inside an interface triangle
    ``(v_i, v_j, v_k)`` consists of the two straight segments
    ``midpoint(v_i, v_j) -> centroid(v_i, v_j, v_k) -> midpoint(v_i, v_k)``.
    On a closed, oriented, piecewise-linear interface mesh the conormal
    contributions are summed across every interface triangle containing
    ``v_i``; on a closed manifold the dual cell boundary is closed and
    the formula is the *exact* Stokes-theorem image of the integrated
    mean curvature normal.

    Parities
    --------
    - Planar interface (kappa = 0 everywhere): ``F_st = 0`` by direct
      cancellation of opposite conormal segments around v_i, *not* by
      a kappa = 0 sample.  Holds to machine precision on any
      triangulation, no symmetry required.
    - Spherical interface of radius R (uniform refinement):
      ``F_st = -(2 gamma / R) * A_i * N`` where N is the outward sphere
      normal and ``A_i`` is the barycentric-dual interface area around
      v_i — the analytical Young-Laplace inward pull.

    Sign convention matches the existing
    :func:`_interface_surface_tension` 3D branch: positive components
    of ``F_st`` push the interface vertex outward; on a convex droplet
    ``F_st`` is anti-parallel to the outward normal.

    Parameters
    ----------
    v : vertex object
        Must have ``v.x_a`` and be flagged ``is_interface=True``.
    interface_set : set or iterable of vertex
        Set of interface vertex objects.  Used as a guard so that
        triangles touching v_i but whose other two vertices fall
        outside the *caller-supplied* interface subset are dropped
        (useful for local validation harnesses).  Pass
        ``{v} | interface_neighbours`` for the standard case.
    HC : Complex
        Must have ``HC.interface_triangles`` populated (set by
        :func:`ddgclib.geometry._interface_subcomplex.extract_interface`).
        If absent, returns ``np.zeros(3)`` (caller should fall back).
    gamma : float
        Surface-tension coefficient [N/m].  Returns zero immediately
        when ``gamma == 0``.

    Returns
    -------
    F_st : ndarray, shape (3,)
        The integrated surface-tension force on v_i.
    """
    if gamma == 0.0:
        return np.zeros(3)

    iface_tris = getattr(HC, 'interface_triangles', None) if HC is not None else None
    if iface_tris is None:
        return np.zeros(3)

    # Build (and cache on HC) the coordinate-key -> vertex object lookup.
    x_to_v = getattr(HC, '_interface_x_to_v', None)
    if x_to_v is None:
        x_to_v = {vv.x: vv for vv in HC.V}
        HC._interface_x_to_v = x_to_v

    interface_set = set(interface_set) if not isinstance(interface_set, set) else interface_set
    v_key = v.x
    x_i = _pad3(v.x_a)
    F = np.zeros(3)

    for tri_key in iface_tris:
        if v_key not in tri_key:
            continue
        other_keys = [k for k in tri_key if k != v_key]
        if len(other_keys) != 2:
            continue  # degenerate (shouldn't happen with frozensets of size 3)
        v_j = x_to_v.get(other_keys[0])
        v_k = x_to_v.get(other_keys[1])
        if v_j is None or v_k is None:
            continue
        # Honor caller's interface_set filter: skip triangles whose other
        # vertices are not in the supplied interface set.
        if v_j not in interface_set or v_k not in interface_set:
            continue

        x_j = _pad3(v_j.x_a)
        x_k = _pad3(v_k.x_a)

        # Triangle normal (unit).  ||cross|| = 2 * triangle_area.
        normal2A = np.cross(x_j - x_i, x_k - x_i)
        twoA = np.linalg.norm(normal2A)
        if twoA < 1e-30:
            continue
        n_tri = normal2A / twoA

        # Barycentric centroid and incident-edge midpoints.
        c = (x_i + x_j + x_k) / 3.0
        m_ij = 0.5 * (x_i + x_j)
        m_ik = 0.5 * (x_i + x_k)

        # Two dual-cell boundary segments inside this triangle, traversed
        # as a path from m_ij to c to m_ik (orientation handled per
        # segment via the dot-product check below).
        for p_start, p_end in ((m_ij, c), (c, m_ik)):
            seg = p_end - p_start
            L = float(np.linalg.norm(seg))
            if L < 1e-30:
                continue
            t = seg / L
            nu = np.cross(n_tri, t)
            # Orient nu OUTWARD from v_i (away from dual cell centre).
            seg_mid = 0.5 * (p_start + p_end)
            if np.dot(nu, seg_mid - x_i) < 0.0:
                nu = -nu
            F += gamma * L * nu

    return F


def int_HNdC_ijk(e_ij, l_ij, l_jk, l_ik):
    """
    Computes the dual edge and dual area using Heron's formula.

    :param e_ij: vector, edge e_ij
    :param l_ij: float, length of edge ij
    :param l_jk: float, length of edge jk
    :param l_ik: float, length of edge ik
    :return: hnda_ijk: vector, curvature vector
             c_ijk: float, dual areas
    """
    lengths = [l_ij, l_jk, l_ik]
    # Sort the list, python sorts from the smallest to largest element:
    lengths.sort()
    # We must have use a ≥ b ≥ c in floating-point stable Heron's formula:
    a = lengths[2]
    b = lengths[1]
    c = lengths[0]
    A = (1 / 4.0) * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    # Dual weights (scalar):
    w_ij = (1 / 8.0) * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A  # w_ij = abs(w_ij)

    # Mean normal curvature (1x3 vector):
    hnda_ijk = w_ij * e_ij  # curvature from this edge jk in triangle ijk with w_jk = 1/2 cot(theta_i^jk)

    # Dual areas
    h_ij = 0.5 * l_ij
    b_ij = abs(w_ij) * l_ij  # = ||0.5 cot(theta_i^jk)|| * 0.5*l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def int_hndA_i(v, n_i=None, HC=None):
    """
    Compute the mean normal curvature of vertex

    :param v: vertex object
    :param HC: optional Complex.  When supplied and ``HC._simplices`` is
        populated, apex enumeration uses ``HC._edge_to_apex`` instead
        of the legacy flag-complex ``vi.nn.intersection(vj.nn)`` path.
    :return: HNdA_i: the curvature tensor at input vertex v
             c_i:  the dual area of the vertex
    """
    # NOTE: THIS MUST BE REPLACED WITH THE LEVEL SET PLANE VECTOR:
    if n_i is not None:
        n_i = v.x

    # Initiate
    HNdA_i = np.zeros(3)  # np.zeros([len(v.nn), 3])  # Mean normal curvature
    C_i = 0.0  # np.zeros([len(v.nn), 3])  # Dual area around edge in a surface
    vi = v
    for vj in v.nn:
        apex = _apex_via_simplex_cache(HC, vi, vj)
        if apex is None:
            e_i_int_e_j = vi.nn.intersection(vj.nn)  # Set of size 1 or 2
        else:
            e_i_int_e_j = list(dict.fromkeys(apex))
        if len(e_i_int_e_j) == 0:
            continue
        e_ij = vj.x_a - vi.x_a  # Compute edge ij (1x3 vector)
        e_ij = - e_ij  # WHY???
        vk = list(e_i_int_e_j)[0]  # index in triangle ijk  # NOTE: k = vk.index
        e_ik = vk.x_a - vi.x_a  # Compute edge ik (1x3 vector)

        if len(e_i_int_e_j) == 1:  # boundary edge
            vk = list(e_i_int_e_j)[0]  # Boundary edge index
            # Compute edges in triangle ijk
            e_ik = vk.x_a - vi.x_a
            e_jk = vk.x_a - vj.x_a
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)

            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            C_i += c_ijk

        else:  # len(e_i_int_e_j) == 2 mathematically guaranteed:
            vl = list(e_i_int_e_j)[1]  # index in triangle ijl
            # Compute dual for contact angle alpha
            e_jk = vk.x_a - vj.x_a
            # wedge_ij_ik = np.cross(e_ij, e_ik)
            # Find lengths (norm of the edge vectors):
            l_ij = np.linalg.norm(e_ij)
            l_ik = np.linalg.norm(e_ik)  # NOTE: l_ki = l_ik
            l_jk = np.linalg.norm(e_jk)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijk, c_ijk = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Contact angle beta
            e_il = vl.x_a - vi.x_a
            e_jl = vl.x_a - vj.x_a
            l_il = np.linalg.norm(e_il)  # NOTE: l_ki = l_ik
            l_jl = np.linalg.norm(e_jl)
            e_hnda_ij = vj.hnda_i - vi.hnda_i
            hnda_ijl, c_ijl = int_HNdC_ijk(e_hnda_ij, l_ij, l_jk, l_ik)

            # Save results
            HNdA_i += hnda_ijk
            HNdA_i += hnda_ijl
            C_i += c_ijl
            C_i += c_ijk

    return HNdA_i, C_i


try:
    import torch as _torch
except ImportError:
    _torch = None


def HNdC_ijk_batch(e_ij, l_ij, l_jk, l_ik):
    """
    Vectorized NumPy version of :func:`HNdC_ijk`.

    Parameters
    ----------
    e_ij : ndarray, shape (N, 3)
        Edge vectors.
    l_ij, l_jk, l_ik : ndarray, shape (N,)
        Edge lengths.

    Returns
    -------
    hnda_ijk : ndarray, shape (N, 3)
        Curvature vectors.
    c_ijk : ndarray, shape (N,)
        Dual areas.
    """
    lengths = np.stack((l_ij, l_jk, l_ik), axis=-1)
    lengths_sorted = np.sort(lengths, axis=-1)
    c = lengths_sorted[..., 0]
    b = lengths_sorted[..., 1]
    a = lengths_sorted[..., 2]

    heron_term = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    A = 0.25 * np.sqrt(heron_term)

    w_ij = 0.125 * (l_jk ** 2 + l_ik ** 2 - l_ij ** 2) / A
    hnda_ijk = w_ij[:, np.newaxis] * e_ij

    h_ij = 0.5 * l_ij
    b_ij = np.abs(w_ij) * l_ij
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def HNdC_ijk_batch_torch(e_ij, l_ij, l_jk, l_ik, device=None):
    """
    Vectorized PyTorch version of :func:`HNdC_ijk`.

    Accepts numpy arrays or torch tensors.  Returns torch tensors on the
    given *device* (defaults to the device selected by the hyperct
    ``TorchBackend``, or CPU).

    Parameters
    ----------
    e_ij : array_like, shape (N, 3)
        Edge vectors.
    l_ij, l_jk, l_ik : array_like, shape (N,)
        Edge lengths.
    device : str or torch.device, optional
        PyTorch device.  Defaults to ``'cpu'``.

    Returns
    -------
    hnda_ijk : Tensor, shape (N, 3)
        Curvature vectors.
    c_ijk : Tensor, shape (N,)
        Dual areas.
    """
    if _torch is None:
        raise ImportError("PyTorch is required for HNdC_ijk_batch_torch.")

    if device is None:
        device = _torch.device('cpu')
    else:
        device = _torch.device(device)

    dtype = _torch.float64

    e_ij_t = _torch.as_tensor(e_ij, dtype=dtype, device=device)
    l_ij_t = _torch.as_tensor(l_ij, dtype=dtype, device=device)
    l_jk_t = _torch.as_tensor(l_jk, dtype=dtype, device=device)
    l_ik_t = _torch.as_tensor(l_ik, dtype=dtype, device=device)

    lengths = _torch.stack((l_ij_t, l_jk_t, l_ik_t), dim=-1)
    lengths_sorted, _ = _torch.sort(lengths, dim=-1)

    c = lengths_sorted[..., 0]
    b = lengths_sorted[..., 1]
    a = lengths_sorted[..., 2]

    heron_term = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    A = 0.25 * _torch.sqrt(heron_term)

    w_ij = 0.125 * (l_jk_t ** 2 + l_ik_t ** 2 - l_ij_t ** 2) / A
    hnda_ijk = w_ij.unsqueeze(-1) * e_ij_t

    h_ij = 0.5 * l_ij_t
    b_ij = _torch.abs(w_ij) * l_ij_t
    c_ijk = 0.5 * b_ij * h_ij
    return hnda_ijk, c_ijk


def heron_mean_curvature_vectors(points, faces, backend='numpy', device=None):
    """
    Assemble per-vertex mean-curvature vectors for a triangle mesh using the
    vectorized Heron kernel.

    Parameters
    ----------
    points : array_like, shape (V, 3)
        Vertex coordinates.
    faces : array_like, shape (F, 3)
        Triangle connectivity (integer indices into *points*).
    backend : ``'numpy'``, ``'torch'``, or a hyperct ``BatchBackend`` object
        Computation backend.  When a ``BatchBackend`` instance is passed its
        ``batch_heron_curvature`` method is used directly.
    device : str or torch.device, optional
        PyTorch device (only used when *backend* is ``'torch'``).

    Returns
    -------
    H_vecs : ndarray, shape (V, 3)
        Per-vertex integrated mean-curvature normal vector (HNdA_i).
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    n_verts = points.shape[0]
    if faces.size == 0:
        return np.zeros((n_verts, 3), dtype=np.float64)

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    vi, vj, vk = points[i], points[j], points[k]

    # All six directed edge contributions per triangle
    e_ij = vj - vi;  e_ik = vk - vi
    e_ji = vi - vj;  e_jk = vk - vj
    e_ki = vi - vk;  e_kj = vj - vk

    l_ij = np.linalg.norm(e_ij, axis=1)
    l_ik = np.linalg.norm(e_ik, axis=1)
    l_jk = np.linalg.norm(e_jk, axis=1)

    # Resolve backend: string shorthand or BatchBackend object
    if hasattr(backend, 'batch_heron_curvature'):
        kernel = backend.batch_heron_curvature
        kwargs = {}
        _to_np = np.asarray
    elif backend == 'torch':
        kernel = HNdC_ijk_batch_torch
        kwargs = {'device': device}
        def _to_np(t):
            return t.detach().cpu().numpy()
    else:
        kernel = HNdC_ijk_batch
        kwargs = {}
        def _to_np(t):
            return t

    # Six directed half-edge curvature contributions
    h_ij, _ = kernel(e_ij, l_ij, l_jk, l_ik, **kwargs)
    h_ik, _ = kernel(e_ik, l_ik, l_jk, l_ij, **kwargs)
    h_ji, _ = kernel(e_ji, l_ij, l_ik, l_jk, **kwargs)
    h_jk, _ = kernel(e_jk, l_jk, l_ik, l_ij, **kwargs)
    h_ki, _ = kernel(e_ki, l_ik, l_ij, l_jk, **kwargs)
    h_kj, _ = kernel(e_kj, l_jk, l_ij, l_ik, **kwargs)

    H_vecs = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(H_vecs, i, _to_np(h_ij))
    np.add.at(H_vecs, i, _to_np(h_ik))
    np.add.at(H_vecs, j, _to_np(h_ji))
    np.add.at(H_vecs, j, _to_np(h_jk))
    np.add.at(H_vecs, k, _to_np(h_ki))
    np.add.at(H_vecs, k, _to_np(h_kj))
    return H_vecs


"""
Example usage:

# Start main loop
HNdA_ijk_l, C_ijk_l = [], []
C = 0
HNdA = np.zeros(3)
for v in HC.V:
    n_i = v.x_a - np.array([0.0, 0.0, 0.0])  # First approximation
    n_i = normalized(n_i)[0]  
    n_test = n_i + (np.random.rand(3) - 0.5)
    HNdA_i, C_i = hndA_i(v, n_i=n_test)
    C += C_i
    HNdA += HNdA_i 
    

"""