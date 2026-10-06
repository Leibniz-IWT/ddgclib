"""Static meniscus of a wetting liquid in a slit (2D) or a round tube (3D)
under gravity: the Young-Laplace solution (laneI, 2026-10-06).

The liquid is in hydrostatic equilibrium with the pressure ``P(y)``
(``y`` the coordinate along the gravity axis, ``y = 0`` at the level
where ``P = 0``, the free surface of the reservoir the tube stands in),
and the free surface carries the pressure jump of the Young-Laplace
condition ``P = -gamma kappa`` with ``kappa`` the total curvature (sum of
the principal curvatures), positive where the surface is concave seen
from the gas.  The contact angle ``theta`` of the liquid is prescribed
at the wall.  The equations are integrated in the arc-length form

    2D   dx/ds = -cos(phi), dy/ds = sin(phi), dphi/ds = kappa
    3D   dr/ds =  cos(phi), dz/ds = sin(phi), dphi/ds = kappa - sin(phi)/r

from the apex (centre of the slit / axis of the tube) outward with the
shooting on the apex height, until the wall is reached, where the angle
``phi`` between the surface and the horizontal must equal
``pi/2 - theta``.  ``kappa = -P(y) / gamma``.

For the incompressible pressure ``P = -rho g y`` the mean height of the
profile is Jurin's height exactly (``gamma cos(theta) / (rho g r)`` in
2D, ``2 gamma cos(theta) / (rho g r)`` in 3D): that is the force balance
on the column.  For the compressible profile of a Tait EOS with ``n = 1``
(``P = K expm1(-rho0 g y / K)``) the same balance holds with the mass
above ``y = 0``, and the mean height is higher by about ``rho0 g h / (2 K)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

__all__ = ['MeniscusProfile', 'young_laplace_meniscus',
           'hydrostatic_pressure_tait', 'jurin_height']


def jurin_height(r: float, gamma: float, theta_deg: float, rho: float,
                 g: float, dim: int) -> float:
    """Jurin's height for an incompressible liquid: ``gamma cos(theta) /
    (rho g r)`` in a slit of half-width *r* (2D), twice that in a round
    tube of radius *r* (3D)."""
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return (dim - 1) * gamma * math.cos(math.radians(theta_deg)) / (rho * g * r)


def hydrostatic_pressure_tait(rho0: float, g: float, K: float,
                              P_ref: float = 0.0,
                              y_ref: float = 0.0) -> Callable[[float], float]:
    """``P(y)`` of a column of linear Tait fluid (``n = 1``) with
    ``P(y_ref) = P_ref``: ``P = P_ref + (K + P_ref) expm1(rho0 g (y_ref - y)
    / K)`` (exact for ``P = K (rho / rho0 - 1)``; with ``P_ref = 0`` the
    usual ``K expm1(-alpha y)``)."""
    alpha = rho0 * g / K

    def P(y: float) -> float:
        return P_ref + (K + P_ref) * math.expm1(alpha * (y_ref - y))

    return P


@dataclass
class MeniscusProfile:
    """The solved meniscus.  ``x`` is the lateral coordinate (distance
    from the wall in 2D, from the axis in 3D), ``y`` the height; both
    arrays run from the apex to the wall."""

    dim: int
    r: float
    x: np.ndarray
    y: np.ndarray
    apex: float
    contact: float
    mean: float
    residual: float

    def height(self, x) -> np.ndarray:
        """Surface height at lateral position(s) *x* (2D: distance from
        the left wall in ``[0, 2 r]``; 3D: radius in ``[0, r]``)."""
        x = np.asarray(x, dtype=float)
        if self.dim == 2:
            d = np.abs(x - self.r)           # distance from the centre
        else:
            d = np.abs(x)
        return np.interp(d, self.x, self.y)


def young_laplace_meniscus(r: float, gamma: float, theta_deg: float,
                           pressure: Callable[[float], float], dim: int,
                           y_guess: float | None = None,
                           n_points: int = 400) -> MeniscusProfile:
    """Solve the Young-Laplace meniscus (module docstring).

    Parameters
    ----------
    r : float
        Half-width of the slit (2D) or radius of the tube (3D).
    gamma, theta_deg : float
        Surface tension and the contact angle of the liquid.
    pressure : callable
        ``P(y)``, the hydrostatic liquid pressure at height *y*
        (negative above the reservoir level); e.g.
        :func:`hydrostatic_pressure_tait` or ``lambda y: -rho * g * y``.
    dim : int
        2 (slit, 2D curvature) or 3 (round tube, axisymmetric).
    y_guess : float, optional
        Bracket scale for the apex height (default: Jurin's height from
        ``P`` linearised at 0).

    Returns
    -------
    MeniscusProfile
        ``apex`` (height at the centre), ``contact`` (height at the wall),
        ``mean`` (volume-averaged height ``V / A_tube``), the profile
        samples and the shooting residual ``phi_wall - (pi/2 - theta)``.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    phi_wall = 0.5 * math.pi - math.radians(theta_deg)
    eps = 1e-9 * r

    def kappa(y: float) -> float:
        return -pressure(y) / gamma

    def rhs(s, q):
        x, y, phi = q
        if dim == 2:
            return [-math.cos(phi), math.sin(phi), kappa(y)]
        k = kappa(y)
        if x < eps:
            return [math.cos(phi), math.sin(phi), 0.5 * k]
        return [math.cos(phi), math.sin(phi), k - math.sin(phi) / x]

    def at_wall(s, q):
        return (q[0] if dim == 2 else r - q[0])
    at_wall.terminal = True
    at_wall.direction = -1

    s_max = 50.0 * r

    def shoot(y0: float, dense: bool = False):
        q0 = [r if dim == 2 else eps, y0, 0.0]
        sol = solve_ivp(rhs, (0.0, s_max), q0, events=at_wall, rtol=1e-11,
                        atol=1e-13 * r, max_step=r / 50.0, dense_output=dense)
        if not sol.t_events[0].size:
            # never reached the wall: the surface turned over before it
            return math.nan, sol
        return float(sol.y_events[0][0][2]) - phi_wall, sol

    # bracket the apex height: a flat surface (kappa = 0) sits at y = 0
    if y_guess is None:
        dP = (pressure(1e-6 * r) - pressure(0.0)) / (1e-6 * r)
        y_guess = abs((dim - 1) * gamma * math.cos(math.radians(theta_deg))
                      / (dP * r)) if dP != 0.0 else r
    lo, hi = 0.0, y_guess
    f_lo = shoot(lo)[0]
    f_hi = shoot(hi)[0]
    tries = 0
    while (math.isnan(f_hi) or f_hi * f_lo > 0.0) and tries < 60:
        if math.isnan(f_hi):
            hi = 0.5 * (lo + hi)          # overshot: the surface turned over
        else:
            lo, f_lo = hi, f_hi
            hi *= 1.5
        f_hi = shoot(hi)[0]
        tries += 1
    if math.isnan(f_hi) or f_hi * f_lo > 0.0:
        raise RuntimeError("young_laplace_meniscus: could not bracket the "
                           "apex height")
    y_apex = brentq(lambda y: shoot(y)[0], lo, hi, xtol=1e-14 * max(r, 1.0),
                    rtol=1e-13, maxiter=200)
    res, sol = shoot(y_apex, dense=True)
    s_end = float(sol.t_events[0][0])
    s = np.linspace(0.0, s_end, n_points)
    q = sol.sol(s)
    x = np.abs(q[0] - (r if dim == 2 else 0.0)) if dim == 2 else q[0]
    y = q[1]
    # volume-averaged height: int y dA / A_tube
    if dim == 2:
        mean = float(np.trapezoid(y, x)) / r
    else:
        mean = 2.0 * float(np.trapezoid(y * x, x)) / r**2
    return MeniscusProfile(dim=dim, r=r, x=np.asarray(x), y=np.asarray(y),
                           apex=float(y[0]), contact=float(y[-1]),
                           mean=mean, residual=float(res))
