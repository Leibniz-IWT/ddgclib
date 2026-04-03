"""Analytical solutions for oscillating droplet (Lamb/Rayleigh).

Provides the linearised analytical solution for small-amplitude
oscillations of an inviscid or viscous droplet about its spherical
equilibrium shape.

The solution is parameterised by mode number ``l`` (l=2 is the
ellipsoidal fundamental mode).

References
----------
- Rayleigh (1879): inviscid frequency.
- Lamb (1932): viscous damping rate.
- Prosperetti (1980): complete initial-value solution.
"""
from __future__ import annotations

import numpy as np


def rayleigh_frequency(
    l: int, gamma: float, rho: float, R0: float, dim: int = 3,
) -> float:
    """Rayleigh angular frequency for mode *l*.

    Parameters
    ----------
    l : int
        Mode number (l >= 2).
    gamma : float
        Surface tension [N/m].
    rho : float
        Droplet density [kg/m³].
    R0 : float
        Equilibrium radius [m].
    dim : int
        2 or 3.

    Returns
    -------
    float
        Angular frequency omega [rad/s].
    """
    if dim == 3:
        omega_sq = l * (l - 1) * (l + 2) * gamma / (rho * R0**3)
    elif dim == 2:
        omega_sq = (l**3 - l) * gamma / (rho * R0**3)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return np.sqrt(max(omega_sq, 0.0))


def lamb_damping_rate(
    l: int, mu: float, rho: float, R0: float, dim: int = 3,
) -> float:
    """Lamb viscous damping rate for mode *l*.

    Parameters
    ----------
    l : int
        Mode number.
    mu : float
        Dynamic viscosity [Pa·s].
    rho : float
        Droplet density [kg/m³].
    R0 : float
        Equilibrium radius [m].
    dim : int
        2 or 3.

    Returns
    -------
    float
        Damping rate beta [1/s].
    """
    if dim == 3:
        return (l - 1) * (2 * l + 1) * mu / (rho * R0**2)
    elif dim == 2:
        return (2 * l**2 - 1) * mu / (rho * R0**2)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def damped_frequency(omega: float, beta: float) -> float:
    """Damped oscillation frequency.

    Returns ``sqrt(omega^2 - beta^2)`` if underdamped, else 0.
    """
    discriminant = omega**2 - beta**2
    if discriminant > 0:
        return np.sqrt(discriminant)
    return 0.0


def radius_perturbation(
    t: float | np.ndarray,
    theta: float | np.ndarray,
    R0: float,
    epsilon: float,
    l: int,
    omega: float,
    beta: float,
) -> float | np.ndarray:
    """Analytical interface radius R(theta, t).

    For an initially deformed droplet with:
        R(theta, 0) = R0 * (1 + epsilon * cos(l * theta))
        dR/dt(theta, 0) = 0  (started from rest)

    The linearised solution is:
        R(theta, t) = R0 + epsilon * R0 * exp(-beta*t)
                      * cos(omega_d * t) * cos(l * theta)

    where omega_d = damped_frequency(omega, beta).

    For the overdamped case (beta > omega), the cosine term becomes
    a decaying exponential (no oscillation).
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    omega_d = damped_frequency(omega, beta)
    envelope = np.exp(-beta * t)

    if omega_d > 0:
        # Underdamped: oscillatory decay
        temporal = envelope * np.cos(omega_d * t)
    else:
        # Overdamped: pure exponential decay (dominant root)
        # Two exponential modes: exp(-(beta ± sqrt(beta²-omega²))t)
        # Dominant (slower) mode:
        delta = np.sqrt(beta**2 - omega**2)
        # Solution satisfying R'(0)=0:
        # R(t) = A*exp(-(beta-delta)*t) + B*exp(-(beta+delta)*t)
        # with A+B=1, -(beta-delta)*A - (beta+delta)*B = 0
        if delta > 1e-30:
            A = (beta + delta) / (2 * delta)
            B = 1.0 - A
            temporal = A * np.exp(-(beta - delta) * t) + B * np.exp(-(beta + delta) * t)
        else:
            # Critically damped
            temporal = (1.0 + beta * t) * np.exp(-beta * t)

    return R0 + epsilon * R0 * temporal * np.cos(l * theta)


def max_radius_envelope(
    t: float | np.ndarray,
    R0: float,
    epsilon: float,
    omega: float,
    beta: float,
) -> float | np.ndarray:
    """Maximum interface radius over all angles at time t.

    This is R(theta=0, t) since cos(l*0) = 1.
    """
    return radius_perturbation(t, 0.0, R0, epsilon, 2, omega, beta)


def pressure_jump_analytical(
    gamma: float, R0: float, dim: int = 3,
) -> float:
    """Young-Laplace equilibrium pressure jump.

    Returns
    -------
    float
        ΔP = γ/R (2D) or ΔP = 2γ/R (3D).
    """
    if dim == 3:
        return 2 * gamma / R0
    return gamma / R0


def kinetic_energy_envelope(
    t: float | np.ndarray,
    R0: float,
    epsilon: float,
    rho: float,
    omega: float,
    beta: float,
    dim: int = 3,
) -> float | np.ndarray:
    """Approximate kinetic energy decay envelope.

    For small perturbations the KE decays as exp(-2*beta*t)
    times the initial KE.
    """
    t = np.asarray(t, dtype=float)
    # Initial KE scales as rho * R0^dim * (epsilon * R0 * omega)^2
    if dim == 3:
        KE0 = 0.5 * rho * (4 / 3) * np.pi * R0**3 * (epsilon * R0 * omega)**2
    else:
        KE0 = 0.5 * rho * np.pi * R0**2 * (epsilon * R0 * omega)**2
    return KE0 * np.exp(-2 * beta * t)
