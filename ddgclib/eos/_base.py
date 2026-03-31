"""Abstract base class for equations of state."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EquationOfState(ABC):
    """Map between density and pressure for a single-phase fluid.

    Subclasses implement the thermodynamic relation P = f(rho) and its
    inverse rho = f^{-1}(P), plus the isentropic speed of sound
    c = sqrt(dP/drho).
    """

    @abstractmethod
    def pressure(self, rho: float | np.ndarray) -> float | np.ndarray:
        """Compute pressure from density.  P = f(rho)."""

    @abstractmethod
    def density(self, P: float | np.ndarray) -> float | np.ndarray:
        """Compute density from pressure (inverse).  rho = f^{-1}(P)."""

    @abstractmethod
    def sound_speed(self, rho: float | np.ndarray) -> float | np.ndarray:
        """Isentropic speed of sound.  c = sqrt(dP/drho)."""
