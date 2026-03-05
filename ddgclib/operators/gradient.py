"""
Discrete gradient and Laplacian operators for continuum simulations.

These functions are now thin wrappers around the Cauchy stress tensor
operators in ``ddgclib.operators.stress``.  The old scalar-area approach
is the special case of the full tensor pipeline:

- ``pressure_gradient``  ->  ``stress_force`` with ``mu=0``
- ``velocity_laplacian`` ->  viscous part of ``stress_force`` (p=0, mu=1)
- ``acceleration``       ->  ``stress_acceleration``

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

from ddgclib.operators.stress import (
    dual_area_vector,
    stress_force,
    stress_acceleration,
)


def pressure_gradient(v, dim: int = 3, HC=None) -> np.ndarray:
    """Discrete integrated pressure force at vertex v.

    Pressure-only special case of the Cauchy stress tensor:

        sigma = -p * I  (no deviatoric stress)

        F_pressure_i = sum_j  -0.5 * (p_j - p_i) * A_ij

    where A_ij is the oriented dual area vector (outward from i).

    Parameters
    ----------
    v : vertex object
        Must have ``v.p`` (scalar pressure), ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension (1, 2, or 3).
    HC : Complex or None
        Required for gradient computations.

    Returns
    -------
    np.ndarray
        Integrated pressure force vector (length dim).
    """
    return stress_force(v, dim=dim, mu=0.0, HC=HC)


def velocity_laplacian(v, dim: int = 3, HC=None) -> np.ndarray:
    """Discrete integrated viscous diffusion term at vertex v.

    Computed using face-centered tensor contraction (same formula as the
    viscous part of ``stress_force``):

        F_v_ij = (mu/|d_ij|) * [du*(d_hat.A) + d_hat*(du.A)]

    with mu=1 to give the raw diffusion operator.

    Parameters
    ----------
    v : vertex object
        Must have ``v.u`` (velocity ndarray), ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    HC : Complex or None
        Required for gradient computations.

    Returns
    -------
    np.ndarray
        Integrated velocity Laplacian vector (length dim).
    """
    # Use stress_force with p=0, mu=1 to isolate the viscous term.
    # We need to temporarily ensure v and neighbors have p=0 for the
    # pressure part to vanish, but stress_force's pressure flux uses
    # (p_j - p_i), so as long as we call with mu=1, we need the pressure
    # contributions to be zero. Instead, compute the viscous flux directly.
    u_i = v.u[:dim]
    x_i = v.x_a[:dim]

    F = np.zeros(dim)
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        delta_u = v_j.u[:dim] - u_i
        d_ij = v_j.x_a[:dim] - x_i
        d_norm = np.linalg.norm(d_ij)
        if d_norm < 1e-30:
            continue
        d_hat = d_ij / d_norm
        # tau_f . A_ij with mu=1
        F += (1.0 / d_norm) * (
            delta_u * np.dot(d_hat, A_ij)
            + d_hat * np.dot(delta_u, A_ij)
        )

    return F


def acceleration(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    """Compute du/dt = F_stress / m at vertex v.

    Delegates to ``stress_acceleration`` from the Cauchy stress tensor
    module.  This implements:

        m_i * dv_i/dt = F_stress_i
        F_stress_i = sum_j (F_p_ij + F_v_ij)

    Can be used directly as ``dudt_fn`` for the dynamic integrators::

        from functools import partial
        dudt_fn = partial(acceleration, dim=3, mu=1e-3, HC=HC)
        t = euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100)

    Parameters
    ----------
    v : vertex object
        Must have ``v.p``, ``v.u``, ``v.m``, ``v.nn``, ``v.vd``.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity [Pa.s].
    HC : Complex or None
        Simplicial complex with duals computed.

    Returns
    -------
    np.ndarray
        Acceleration vector (length dim).
    """
    return stress_acceleration(v, dim=dim, mu=mu, HC=HC)
