"""
ARCHIVED — Vertex-centered point-wise stress formulation.

Superseded by the face-centered integrated formulation in stress.py.
Kept for reference only; not used in the production pipeline.

This module implements a **vertex-centered point-wise** approach:

    1. Compute point-wise velocity gradient du_i = (1/Vol_i) * sum_j du outer A_ij
    2. Build point-wise Cauchy stress sigma_i = -p_i*I + 2*mu*eps(du_i)
    3. Face-average: sigma_f = 0.5 * (sigma_i + sigma_j)
    4. Integrate: F_i = sum_j sigma_f @ A_ij

Known issues with this formulation
-----------------------------------
1. The symmetric stress tensor tau = mu*(du + du^T) introduces a spurious
   mu*grad(div u) term because the rank-1 face gradient (du outer d_hat / |d|)
   has nonzero discrete divergence on non-aligned edges, even for exactly
   divergence-free velocity fields.  This causes O(h^0) acceleration residuals
   at equilibrium (e.g. ~0.34 on Poiseuille) instead of machine-precision zero.

2. The vertex-centered approach divides by Vol_i to get a point-wise gradient,
   then multiplies by A_ij during integration — mixing two approximation levels.
   The face-centered approach avoids this by computing fluxes directly from
   velocity differences across primal edges.

The current formulation (stress.py) uses face-centered fluxes:
    Pressure:  F_p_ij = -0.5*(p_i + p_j) * A_ij      (conservative)
    Viscous:   F_v_ij = (mu/|d|) * du * (d_hat . A)   (diffusion form)

See Fundamentals.md for the derivation and Fundamentals_pointwise.md for this
archived formulation's documentation.
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

    In 3D the dual face is a polygon triangulated into small triangles by
    e_star; each triangle contributes a vector area A_ijk = 0.5 * cross(e1, e2).
    The total is A_ij = sum_k A_ijk, oriented outward from v_i using the test:

        vec_to_i = v_i.x_a - vc.x_a   (from dual face toward parcel i)
        if dot(A_ijk, vec_to_i) > 0:   (points inward)
            A_ijk = -A_ijk              (flip to outward)

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
        from hyperct.ddg import e_star as _e_star
        A_ijk_arr = _e_star(v_i, v_j, HC, dim=3)  # shape (N, 3)
        if not isinstance(A_ijk_arr, np.ndarray) or A_ijk_arr.size == 0:
            return np.zeros(3)
        # Orientation reference: dual vertex on the face
        vc_12_pos = 0.5 * (v_j.x_a - v_i.x_a) + v_i.x_a
        vc_12 = HC.Vd[tuple(vc_12_pos)]
        vec_to_i = v_i.x_a - vc_12.x_a
        A_ij = np.zeros(3)
        for A_ijk in A_ijk_arr:
            if np.dot(A_ijk, vec_to_i) > 0:  # points inward -> flip
                A_ijk = -A_ijk
            A_ij += A_ijk
        return A_ij

    else:
        raise NotImplementedError(f"dual_area_vector not implemented for dim={dim}")


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
    if dim <= 2:
        from hyperct.ddg import d_area
        return d_area(v)

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
# Physics: velocity difference tensor, strain rate, stress
# ---------------------------------------------------------------------------

def velocity_difference_tensor(v, HC, dim: int = 3) -> np.ndarray:
    """Discrete integrated velocity difference tensor du_i at vertex v.

    Computes the DDG integrated quantity:

        du_i = (1 / Vol_i) * sum_j (u_j - u_i) outer A_ij

    This is analogous to the volume-averaged velocity gradient
    int_{V_i} grad(u) dV / Vol_i, but computed purely from discrete data
    (velocity differences at neighboring parcels and oriented dual area
    vectors).  It is NOT a pointwise gradient approximation.

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
        Velocity difference tensor, shape ``(dim, dim)``.
        Component ``du_i[a, b] = (1/Vol_i) sum_j (u_j^a - u_i^a) * A_ij^b``.
    """
    Vol_i = dual_volume(v, HC, dim)
    if Vol_i < 1e-30:
        return np.zeros((dim, dim))
    du_i = np.zeros((dim, dim))
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_u = v_j.u[:dim] - v.u[:dim]
        du_i += np.outer(delta_u, A_ij)
    du_i = 0.5 * du_i
    du_i /= Vol_i  # NOTE: This divides by volume to get point-wise gradient.
    # The face-centered formulation in stress.py avoids this entirely.
    return du_i


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
   # Vol_i = dual_volume(v, HC, dim)
   # return -p * np.eye(dim) * Vol_i + 2.0 * mu * strain_rate(du)


def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    """Total stress force on the dual cell of vertex v.

    Implements the integrated Cauchy equation over the dual cell surface:

        F_stress_i = int_{S_i} sigma . n dS
                   = sum_j sigma_f @ A_ij

    where:
        sigma_f = 0.5 * (sigma_i + sigma_j)   — face-averaged stress
        A_ij = oriented dual area vector       — outward from parcel i

    The pressure part alone gives:
        -p_f * A_ij  (since  -p * I @ A = -p * A)

    which reduces to the old pressure_gradient when mu = 0.

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

    Returns
    -------
    np.ndarray
        Force vector, shape ``(dim,)``.
    """
    du_i = velocity_difference_tensor(v, HC, dim)
    p_i = float(v.p) if np.ndim(v.p) == 0 else float(v.p[0])
    sigma_i = cauchy_stress(p_i, du_i, mu, dim)

    F = np.zeros(dim)
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        du_j = velocity_difference_tensor(v_j, HC, dim)
        p_j = float(v_j.p) if np.ndim(v_j.p) == 0 else float(v_j.p[0])
        sigma_j = cauchy_stress(p_j, du_j, mu, dim)
        sigma_f = 0.5 * (sigma_i + sigma_j)
        F += sigma_f @ A_ij
    return F


def stress_acceleration(
    v,
    dim: int = 3,
    mu: float = 8.9e-4,
    HC=None,
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

    Returns
    -------
    np.ndarray
        Acceleration vector, shape ``(dim,)``.
    """
    return stress_force(v, dim=dim, mu=mu, HC=HC) / v.m


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
