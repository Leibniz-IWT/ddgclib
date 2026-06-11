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


# ---------------------------------------------------------------------------
# Geometry: dual area vectors and dual volumes
# TODO: move dual_area_vector and dual_volume to hyperct.ddg (pure geometry)
# ---------------------------------------------------------------------------

def dual_area_vector(v_i, v_j, HC, dim: int = 3) -> np.ndarray:
    """Oriented dual area vector for the interface between parcels i and j.

    Computes A_ij, the total outward area vector of the dual face separating
    the dual cells of v_i and v_j.  In the continuum limit this is:

        A_ij = int_{S_ij} n dS

    where n is the outward unit normal from parcel i.

    In 2D the dual face is the line segment between the two shared dual
    vertices; A_ij is the outward-facing normal with magnitude equal to the
    segment length.

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
                vec_to_i = x_i - 0.5 * (bary + midpt)
                if np.dot(A_ij, vec_to_i) > 0:
                    A_ij = -A_ij
                return A_ij
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
            centroid = 0.5 * (bary_l + bary_r)
            vec_to_i = x_i - centroid
            if np.dot(A_ij, vec_to_i) > 0:
                A_ij = -A_ij
            return A_ij

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
        centroid = 0.5 * (vd1.x_a[:2] + vd2.x_a[:2])
        vec_to_i = v_i.x_a[:2] - centroid
        if np.dot(A_ij, vec_to_i) > 0:
            A_ij = -A_ij
        return A_ij

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
    # TODO: move to hyperct.ddg._operators (pure geometry, no physics)
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
        from hyperct.ddg import dual_cell_area_2d
        return dual_cell_area_2d(v, include_edge_midpoints=True)

    elif dim == 3:
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


def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None,
                 pressure_model=None) -> np.ndarray:
    """Integrated force on FVM via face-centered fluxes (Stokes' theorem).

    For each dual flux plane between parcels i and j, the force has two
    contributions computed directly from edge data:

    Pressure (face-average, conservative):

        F_p_ij = -0.5 * (p_i + p_j) * A_ij

    Viscous (face-centered diffusion):

        F_v_ij = mu * (grad u)_f . A_ij
               = (mu / |d_ij|) * du * (d_hat . A_ij)

    where du = u_j - u_i, d_hat = (x_j - x_i) / |x_j - x_i|.

    This is the "diffusion form" of the viscous term (mu * Laplacian u),
    which is equivalent to the full symmetric stress divergence
    div(mu * (grad u + grad u^T)) for incompressible flow (div u = 0).
    The symmetric (transpose) term mu * grad(div u) is omitted because
    the rank-1 face gradient has spurious discrete compressibility.

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

    Returns
    -------
    np.ndarray
        Force vector, shape ``(dim,)``.
    """
    p_i = _resolve_pressure(v, pressure_model, HC, dim)
    u_i = v.u[:dim]
    x_i = v.x_a[:dim]

    # Use cached oriented edge area vectors when available (set by
    # batch_e_star(..., orient=True) during retopologization).
    _cache = getattr(HC, '_edge_area_cache', None)
    _vid = id(v) if _cache is not None else None

    F = np.zeros(dim)
    for v_j in v.nn:
        if _cache is not None and _vid in _cache and id(v_j) in _cache[_vid]:
            A_ij = _cache[_vid][id(v_j)]
        else:
            A_ij = dual_area_vector(v, v_j, HC, dim)

        p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i
        F += pressure_flux(p_i, p_j, A_ij)
        F += viscous_flux(mu, delta_u, d_ij, A_ij)

    return F


def stress_acceleration(
    v,
    dim: int = 3,
    mu: float = 8.9e-4,
    HC=None,
    pressure_model=None,
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

    Returns
    -------
    np.ndarray
        Acceleration vector, shape ``(dim,)``.
    """
    return stress_force(v, dim=dim, mu=mu, HC=HC,
                        pressure_model=pressure_model) / v.m


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
