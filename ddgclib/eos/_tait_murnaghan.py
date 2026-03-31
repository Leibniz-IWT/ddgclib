"""Tait-Murnaghan equation of state for weakly compressible fluids.

The Tait-Murnaghan (or modified Tait) equation relates pressure to
density via a power-law stiffening term:

    P(rho) = P0 + (K / n) * ((rho / rho0)^n - 1)

where *K* is the bulk modulus, *n* the Tait exponent, *rho0* the
reference density, and *P0* the reference pressure.  For water the
standard parameters are K = 2.15e9 Pa, n = 7.15.

Reference
---------
Dymond & Malhotra (1988), "The Tait equation: 100 years on",
Int. J. Thermophysics 9(6), 941–951.
"""
from __future__ import annotations

import numpy as np

from ddgclib.eos._base import EquationOfState


class TaitMurnaghan(EquationOfState):
    """Tait-Murnaghan EOS for weakly compressible fluids.

    Parameters
    ----------
    rho0 : float
        Reference density [kg/m^3].
    P0 : float
        Reference pressure [Pa].
    K : float
        Bulk modulus [Pa].
    n : float
        Tait exponent (7.15 for water).
    rho_clip : tuple[float, float] or None
        Density clipping factors ``(min_ratio, max_ratio)`` relative to
        *rho0*.  Prevents unphysical densities.  ``None`` disables
        clipping.
    """

    def __init__(
        self,
        rho0: float = 1000.0,
        P0: float = 101325.0,
        K: float = 2.15e9,
        n: float = 7.15,
        rho_clip: tuple[float, float] | None = (0.9, 1.1),
    ):
        self.rho0 = rho0
        self.P0 = P0
        self.K = K
        self.n = n
        self.rho_clip = rho_clip

    # -- forward: rho -> P ------------------------------------------------

    def pressure(self, rho: float | np.ndarray) -> float | np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if self.rho_clip is not None:
            rho = np.clip(
                rho,
                self.rho0 * self.rho_clip[0],
                self.rho0 * self.rho_clip[1],
            )
        return self.P0 + (self.K / self.n) * ((rho / self.rho0) ** self.n - 1.0)

    # -- inverse: P -> rho ------------------------------------------------

    def density(self, P: float | np.ndarray) -> float | np.ndarray:
        P = np.asarray(P, dtype=float)
        ratio = (self.n / self.K) * (P - self.P0) + 1.0
        ratio = np.maximum(ratio, 1e-30)
        return self.rho0 * ratio ** (1.0 / self.n)

    # -- thermodynamic derivative ------------------------------------------

    def sound_speed(self, rho: float | np.ndarray) -> float | np.ndarray:
        """c = sqrt(dP/drho) = sqrt((K / rho0) * (rho / rho0)^(n-1))."""
        rho = np.asarray(rho, dtype=float)
        c_sq = (self.K / self.rho0) * (rho / self.rho0) ** (self.n - 1)
        return np.sqrt(c_sq)

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TaitMurnaghan(rho0={self.rho0}, P0={self.P0}, "
            f"K={self.K:.3e}, n={self.n})"
        )
