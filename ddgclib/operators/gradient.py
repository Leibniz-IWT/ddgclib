"""
Discrete gradient and Laplacian operators for continuum simulations.

These functions are now thin wrappers around the Cauchy stress tensor
operators in ``ddgclib.operators.stress``.  The old scalar-area approach
is the special case of the full tensor pipeline:

- ``pressure_gradient``  ->  ``stress_force`` with ``mu=0``
- ``velocity_laplacian`` ->  computed from ``velocity_difference_tensor``
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
    velocity_difference_tensor,
    strain_rate,
)


def pressure_gradient(v, dim: int = 3, HC=None) -> np.ndarray:
    """Discrete integrated pressure force at vertex v.

    Pressure-only special case of the Cauchy stress tensor:

        sigma = -p * I  (no deviatoric stress)

        F_pressure_i = sum_j  -p_f * A_ij

    where p_f = 0.5 * (p_i + p_j) and A_ij is the oriented dual area
    vector (outward from i).

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

    Computed from the velocity difference tensor du_i:

        lap_u_i = sum_j  tau_f @ A_ij

    where tau_f is the deviatoric stress (mu * (du + du^T)) only.

    This is the viscous contribution to stress_force (with p = 0).

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
    # Compute tau-only stress force (p=0, mu=1 to get the raw diffusion term)
    du_i = velocity_difference_tensor(v, HC, dim)
    eps_i = strain_rate(du_i)
    tau_i = 2.0 * eps_i  # mu factored out

    F = np.zeros(dim)
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        du_j = velocity_difference_tensor(v_j, HC, dim)
        eps_j = strain_rate(du_j)
        tau_j = 2.0 * eps_j
        tau_f = 0.5 * (tau_i + tau_j)
        F += tau_f @ A_ij
    return F


def acceleration(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    """Compute du/dt = F_stress / m at vertex v.

    Delegates to ``stress_acceleration`` from the Cauchy stress tensor
    module.  This implements:

        m_i * dv_i/dt = F_stress_i
        F_stress_i = sum_j  sigma_f @ A_ij
        sigma = -p * I + 2 * mu * epsilon

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
