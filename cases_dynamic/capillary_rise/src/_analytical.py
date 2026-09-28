"""Analytical and reference solutions for capillary rise dynamics.

Provides the Washburn ODE, early-time Lucas-Washburn scaling, and the
upscaled model of Lunowa et al. (2022) for validation of numerical
simulations.

References
----------
Washburn, E. W. (1921). The dynamics of capillary flow.
    Phys. Rev. 17(3), 273-283.
Lucas, R. (1918). Ueber das Zeitgesetz des kapillaren Aufstiegs von
    Fluessigkeiten. Kolloid-Z. 23(1), 15-22.
Lunowa, S. B., Bringedal, C., & Pop, I. S. (2022). On an averaged model
    for the 2-fluid immiscible flow with surface tension in a thin
    cylindrical tube. IMA J. Appl. Math., 87(3), 387-413.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from ._params import jurin_height  # re-export for convenience

__all__ = [
    "jurin_height",
    "washburn_ode_rhs",
    "washburn_solve",
    "lucas_washburn_early",
    "washburn_velocity",
    "lunowa_upscaled_height",
    "lunowa_dynamic_ca",
]

# Minimum column height to regularise the h -> 0 singularity [m].
_H_MIN: float = 1e-8


def washburn_ode_rhs(
    t: float,
    h: float,
    r: float,
    gamma: float,
    theta_deg: float,
    mu: float,
    rho: float,
    g: float = 9.81,
    dim: int = 3,
) -> float:
    """Right-hand side of the Washburn ODE for meniscus height h(t).

    Parameters
    ----------
    t : float
        Time [s] (unused explicitly, included for ODE solver signature).
    h : float
        Current meniscus height [m].
    r : float
        Tube radius [m].
    gamma : float
        Surface tension [N/m].
    theta_deg : float
        Static contact angle [deg].
    mu : float
        Dynamic viscosity [Pa s].
    rho : float
        Liquid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    dim : int
        3 = cylindrical tube (Poiseuille), 2 = slit channel.

    Returns
    -------
    float
        dh/dt [m/s].

    Notes
    -----
    3D (cylindrical):  dh/dt = r^2 / (8 mu h) * (2 gamma cos(theta)/r - rho g h)
    2D (slit channel): dh/dt = r^2 / (3 mu h) * (gamma cos(theta)/r - rho g h)

    When ``h < _H_MIN`` the Lucas-Washburn early-time limit is used to
    avoid the 1/h singularity.
    """
    theta = math.radians(theta_deg)
    cos_theta = math.cos(theta)

    if dim == 3:
        dp = 2.0 * gamma * cos_theta / r - rho * g * h
        prefactor = r**2 / (8.0 * mu)
    elif dim == 2:
        dp = gamma * cos_theta / r - rho * g * h
        prefactor = r**2 / (3.0 * mu)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    if h < _H_MIN:
        # Lucas-Washburn limit: h ~ sqrt(C t) => dh/dt ~ C/(2h).
        # Use the driving pressure (gravity negligible at h~0).
        # dh/dt = prefactor * dp_cap / h_min (bounded)
        if dim == 3:
            dp_cap = 2.0 * gamma * cos_theta / r
        else:
            dp_cap = gamma * cos_theta / r
        return prefactor * dp_cap / _H_MIN

    return prefactor * dp / h


def washburn_solve(
    t_span: tuple[float, float],
    h0: float,
    r: float,
    gamma: float,
    theta_deg: float,
    mu: float,
    rho: float,
    g: float = 9.81,
    dim: int = 3,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Washburn ODE numerically.

    Parameters
    ----------
    t_span : tuple[float, float]
        Integration interval (t0, tf) in seconds.
    h0 : float
        Initial meniscus height [m].  Should be small but positive
        (e.g. 1e-6).
    r, gamma, theta_deg, mu, rho, g, dim
        Physical parameters (see :func:`washburn_ode_rhs`).
    n_points : int
        Number of output time points.

    Returns
    -------
    t_arr : np.ndarray, shape (n_points,)
        Time values [s].
    h_arr : np.ndarray, shape (n_points,)
        Meniscus height values [m].
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        return [washburn_ode_rhs(t, y[0], r, gamma, theta_deg, mu, rho, g, dim)]

    sol = solve_ivp(
        rhs,
        t_span,
        [h0],
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=(t_span[1] - t_span[0]) / 100.0,
    )

    if not sol.success:
        raise RuntimeError(f"Washburn ODE integration failed: {sol.message}")

    return sol.t, sol.y[0]


def lucas_washburn_early(
    t: float | np.ndarray,
    r: float,
    gamma: float,
    theta_deg: float,
    mu: float,
    dim: int = 3,
) -> float | np.ndarray:
    """Lucas-Washburn early-time approximation: h proportional to sqrt(t).

    Valid when gravity is negligible (h << Jurin height).

    Parameters
    ----------
    t : float or array
        Time [s].
    r : float
        Tube radius [m].
    gamma : float
        Surface tension [N/m].
    theta_deg : float
        Static contact angle [deg].
    mu : float
        Dynamic viscosity [Pa s].
    dim : int
        3 = cylindrical tube, 2 = slit channel.

    Returns
    -------
    float or array
        Meniscus height [m].
        3D: ``h = sqrt(r gamma cos(theta) t / (2 mu))``
        2D: ``h = sqrt(2 r gamma cos(theta) t / (3 mu))``
    """
    theta = math.radians(theta_deg)
    cos_theta = math.cos(theta)
    t = np.asarray(t)

    if dim == 3:
        return np.sqrt(r * gamma * cos_theta * t / (2.0 * mu))
    elif dim == 2:
        return np.sqrt(2.0 * r * gamma * cos_theta * t / (3.0 * mu))
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def washburn_velocity(
    h: float | np.ndarray,
    r: float,
    gamma: float,
    theta_deg: float,
    mu: float,
    rho: float,
    g: float = 9.81,
    dim: int = 3,
) -> float | np.ndarray:
    """Washburn meniscus velocity at a given column height.

    Parameters
    ----------
    h : float or array
        Meniscus height [m].
    r, gamma, theta_deg, mu, rho, g, dim
        Physical parameters (see :func:`washburn_ode_rhs`).

    Returns
    -------
    float or array
        dh/dt [m/s].
    """
    theta = math.radians(theta_deg)
    cos_theta = math.cos(theta)
    h = np.asarray(h, dtype=float)

    if dim == 3:
        dp = 2.0 * gamma * cos_theta / r - rho * g * h
        return r**2 / (8.0 * mu) * dp / np.maximum(h, _H_MIN)
    elif dim == 2:
        dp = gamma * cos_theta / r - rho * g * h
        return r**2 / (3.0 * mu) * dp / np.maximum(h, _H_MIN)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def lunowa_upscaled_height(
    t_nondim: float | np.ndarray,
    eta: float,
    slip: float,
    theta_s_rad: float,
) -> float | np.ndarray:
    """Lunowa et al. (2022) upscaled model for non-dimensional meniscus height.

    Solves the implicit relation for h(t) from the leading-order upscaled
    equation.  The non-dimensional height h in [0, 1] satisfies:

        1 - (1 - h) exp(h / D + (4 lambda + 1) / (8 D) * t) = 0

    where ``D = 1 + (lambda + 0.25) eta``, ``lambda`` is the Navier slip
    length (non-dimensional), and ``eta`` encodes dynamic contact-angle
    effects.

    Parameters
    ----------
    t_nondim : float or array
        Non-dimensional time.
    eta : float
        Dynamic contact-angle sensitivity parameter.
    slip : float
        Non-dimensional Navier slip length (lambda).
    theta_s_rad : float
        Static contact angle in Lunowa's convention (receding fluid) [rad].

    Returns
    -------
    float or np.ndarray
        Non-dimensional meniscus height in [0, 1).
    """
    D = 1.0 + (slip + 0.25) * eta
    coeff = (4.0 * slip + 1.0) / (8.0 * D)

    def _solve_single(t: float) -> float:
        # f(h) = 1 - (1 - h) exp(h/D + coeff * t)
        def f(h: float) -> float:
            return 1.0 - (1.0 - h) * math.exp(h / D + coeff * t)

        # At t=0, h=0: f(0) = 1 - exp(0) = 0 (trivially satisfied).
        # For t > 0, h is in (0, 1).
        if t <= 0.0:
            return 0.0

        try:
            sol = root_scalar(f, bracket=[0.0, 1.0 - 1e-15], method="brentq")
            return sol.root
        except ValueError:
            # If bracketing fails, the solution is very close to 1.
            return 1.0 - 1e-15

    t_nondim = np.asarray(t_nondim, dtype=float)
    scalar_input = t_nondim.ndim == 0
    t_flat = t_nondim.ravel()

    h_arr = np.array([_solve_single(ti) for ti in t_flat])

    if scalar_input:
        return float(h_arr[0])
    return h_arr.reshape(t_nondim.shape)


def lunowa_dynamic_ca(
    velocity_nondim: float | np.ndarray,
    theta_s_rad: float,
    eta: float,
    Ca: float,
) -> float | np.ndarray:
    """Dynamic contact angle from Lunowa's linear Cox-Voinov-type relation.

    theta_d = arccos(cos(theta_s) + eta * Ca * v)

    Parameters
    ----------
    velocity_nondim : float or array
        Non-dimensional meniscus velocity.
    theta_s_rad : float
        Static contact angle (Lunowa convention, receding fluid) [rad].
    eta : float
        Dynamic contact-angle sensitivity parameter.
    Ca : float
        Capillary number (from :func:`_params.nondim_scales`).

    Returns
    -------
    float or array
        Dynamic contact angle [rad].
    """
    v = np.asarray(velocity_nondim)
    arg = math.cos(theta_s_rad) + eta * Ca * v
    # Clamp to [-1, 1] for safety
    arg = np.clip(arg, -1.0, 1.0)
    return np.arccos(arg)
