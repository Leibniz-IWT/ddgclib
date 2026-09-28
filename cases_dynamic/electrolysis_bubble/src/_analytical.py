"""Analytical references for bubble detachment.

Implements the capillary length and the Fritz-style detachment volume
estimate for a pinned hemispherical bubble on a flat surface.

References
----------
Fritz, W. (1935). Berechnung des Maximalvolumens von Dampfblasen.
    Physikalische Zeitschrift 36, 379-384.
Chesters, A. K. (1977). An analytical solution for the profile and
    volume of a small drop or bubble symmetric about a vertical axis.
    J. Fluid Mech. 81, 609-624.
"""
from __future__ import annotations

import numpy as np


def capillary_length(gamma: float, rho_diff: float, g: float) -> float:
    """Capillary length lambda = sqrt(gamma / (rho_diff * g))."""
    return float(np.sqrt(gamma / (rho_diff * g)))


def fritz_detachment_radius(r_contact: float, gamma: float,
                             rho_diff: float, g: float) -> float:
    """Detachment radius from the Fritz balance (pinned contact line).

    Equates bubble buoyancy to the vertical component of surface tension
    at a pinned circular contact line of radius ``r_contact``::

        (4/3) * pi * R^3 * (rho_liq - rho_gas) * g = 2 * pi * r_contact * gamma

    Yields::

        R_detach = (3 * r_contact * lambda^2 / 2)^(1/3)

    where ``lambda = sqrt(gamma / (rho_diff * g))``.
    """
    lam = capillary_length(gamma, rho_diff, g)
    return float((1.5 * r_contact * lam * lam) ** (1.0 / 3.0))


def bond_number(R: float, gamma: float, rho_diff: float, g: float) -> float:
    """Bond number Bo = rho_diff * g * R^2 / gamma."""
    return float(rho_diff * g * R * R / gamma)


def young_laplace_jump(gamma: float, R: float, dim: int = 3) -> float:
    """Laplace pressure jump inside a spherical/circular interface."""
    if dim == 3:
        return 2.0 * gamma / R
    return gamma / R
