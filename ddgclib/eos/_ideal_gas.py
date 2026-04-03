"""Ideal gas equation of state.

Simple ideal gas law: P(rho) = rho * R_specific * T

Suitable for low-pressure gas phases in multiphase simulations where
the gas is far from condensation.
"""
from __future__ import annotations

import numpy as np

from ddgclib.eos._base import EquationOfState


class IdealGas(EquationOfState):
    """Ideal gas EOS: P = rho * R_specific * T.

    Parameters
    ----------
    rho0 : float
        Reference density [kg/m^3].
    T : float
        Temperature [K].
    R_specific : float
        Specific gas constant [J/(kg·K)].  Default 287.058 (dry air).
    P0 : float or None
        Reference pressure [Pa].  If None, computed as rho0 * R_specific * T.
    """

    def __init__(
        self,
        rho0: float = 1.225,
        T: float = 293.15,
        R_specific: float = 287.058,
        P0: float | None = None,
    ):
        self.rho0 = rho0
        self.T = T
        self.R_specific = R_specific
        self.P0 = P0 if P0 is not None else rho0 * R_specific * T

    def pressure(self, rho: float | np.ndarray) -> float | np.ndarray:
        rho = np.asarray(rho, dtype=float)
        return rho * self.R_specific * self.T

    def density(self, P: float | np.ndarray) -> float | np.ndarray:
        P = np.asarray(P, dtype=float)
        return P / (self.R_specific * self.T)

    def sound_speed(self, rho: float | np.ndarray) -> float | np.ndarray:
        """Isothermal sound speed: c = sqrt(R_specific * T)."""
        return np.sqrt(self.R_specific * self.T)

    def __repr__(self) -> str:
        return (
            f"IdealGas(rho0={self.rho0}, T={self.T}, "
            f"R_specific={self.R_specific:.3f})"
        )
