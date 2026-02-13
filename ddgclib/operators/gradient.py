"""
Discrete gradient and Laplacian operators for continuum simulations.

Clean reimplementation of the pressure gradient, velocity Laplacian and
acceleration operators with the following improvements over legacy code:

- No debug print() statements
- Passes HC explicitly to e_star (fixes 3D bug)
- Expects scalar v.P (not vector)
- Consistent dim handling

These functions call e_star from hyperct.ddg as the computational
backend.

Usage
-----
    from ddgclib.operators.gradient import acceleration

    # As a standalone call
    a = acceleration(v, dim=3, mu=1e-3, HC=HC)

    # As dudt_fn for integrators (using functools.partial or lambda)
    from functools import partial
    dudt_fn = partial(acceleration, dim=3, mu=1e-3, HC=HC)
    t = euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100)
"""

import numpy as np


def pressure_gradient(v, dim: int = 3, HC=None) -> np.ndarray:
    """Discrete integrated pressure gradient at vertex v.

    Computes the integrated pressure force on the dual cell of v:

        grad_P_i = sum_j  A_ij * (P_j - P_i)

    where A_ij is the dual edge length (2D) or dual face area (3D)
    between vertices v_i and v_j.

    Parameters
    ----------
    v : vertex object
        Must have v.P (scalar pressure), v.nn (neighbors), v.vd (dual vertices).
    dim : int
        Spatial dimension (2 or 3).
    HC : Complex or None
        Required for 3D (e_star needs HC.Vd for edge midpoint lookup).

    Returns
    -------
    np.ndarray
        Integrated pressure gradient vector (length dim).
    """
    from hyperct.ddg import e_star as _e_star

    dP_i = np.zeros(dim)
    P_i = float(v.P) if np.ndim(v.P) == 0 else float(v.P[0])

    for vp2 in v.nn:
        P_j = float(vp2.P) if np.ndim(vp2.P) == 0 else float(vp2.P[0])

        if dim == 2:
            e_dual = _e_star(v, vp2, HC, dim=dim)
            area_flux = e_dual  # scalar edge length in 2D
            dP_i += area_flux * (P_j - P_i)
        elif dim == 3:
            e_dual = _e_star(v, vp2, HC, dim=dim)
            # In 3D, e_star returns a scalar (total dual edge length)
            area_flux = e_dual
            dP_i += area_flux * (P_j - P_i)

    return dP_i


def velocity_laplacian(v, dim: int = 3, HC=None) -> np.ndarray:
    """Discrete Laplacian of the velocity field at vertex v.

    Computes the integrated viscous diffusion term:

        lap_u_i = sum_j  w_ij * (u_j - u_i)

    where w_ij = |e_ij| / |e_ij*| is the ratio of primal edge length
    to dual edge length.

    Parameters
    ----------
    v : vertex object
        Must have v.u (velocity ndarray), v.nn (neighbors), v.vd (dual vertices).
    dim : int
        Spatial dimension.
    HC : Complex or None
        Required for 3D.

    Returns
    -------
    np.ndarray
        Integrated velocity Laplacian vector (length dim).
    """
    from hyperct.ddg import e_star as _e_star

    du_i = np.zeros(dim)

    for vp2 in v.nn:
        l_ij = np.linalg.norm(vp2.x_a[:dim] - v.x_a[:dim])
        e_dual = _e_star(v, vp2, HC, dim=dim)

        if isinstance(e_dual, (int, float)):
            if e_dual == 0 or np.isinf(e_dual):
                continue
            w_ij = l_ij / e_dual
        else:
            # Fallback for unexpected array return
            w_ij = l_ij

        if np.isinf(w_ij) or w_ij == 0:
            continue

        du_i += np.abs(w_ij) * (vp2.u[:dim] - v.u[:dim])

    return du_i


def acceleration(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    """Compute du/dt = (-grad(P) + mu * lap(u)) / m at vertex v.

    This is the right-hand side of the momentum equation for
    incompressible Newtonian flow in the Lagrangian frame.

    Can be used directly as ``dudt_fn`` for the dynamic integrators::

        from functools import partial
        dudt_fn = partial(acceleration, dim=3, mu=1e-3, HC=HC)
        t = euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100)

    Parameters
    ----------
    v : vertex object
        Must have v.P (scalar), v.u (velocity), v.m (mass), v.nn, v.vd.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex or None
        Required for 3D gradient computations.

    Returns
    -------
    np.ndarray
        Acceleration vector (length dim).
    """
    grad_P = pressure_gradient(v, dim=dim, HC=HC)
    lap_u = velocity_laplacian(v, dim=dim, HC=HC)
    a = (-grad_P + mu * lap_u) / v.m
    return a
