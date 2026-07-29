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

import warnings

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

        When set, the clipped EOS is a coherent *saturating* model:
        ``pressure()``, ``density()`` and ``sound_speed()`` all clamp
        to the band ``[min_ratio*rho0, max_ratio*rho0]``, so the
        representable pressure window is
        ``[pressure(min_ratio*rho0), pressure(max_ratio*rho0)]`` and
        the round trip ``density(pressure(rho))`` is idempotent
        (out-of-band states map to the band edge, consistently in both
        directions).  Saturation is never silent: engagements are
        counted in :attr:`clip_count` and a ``RuntimeWarning`` is
        emitted once per instance on first engagement.

    Attributes
    ----------
    clip_count : dict[str, int]
        Number of clip engagements (clipped array elements) per method,
        keys ``'pressure'``, ``'density'``, ``'sound_speed'``.  A
        persistently growing count means the EOS is saturated —
        compressibility physics is effectively switched off for the
        affected states.
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
        # Clip-engagement diagnostics: saturation must never be silent
        # (see docs_temp/audit/eos-formulas.md §2.3).
        self.clip_count: dict[str, int] = {
            'pressure': 0, 'density': 0, 'sound_speed': 0,
        }
        self._clip_warned = False

    # -- clip helper -------------------------------------------------------

    def _clip_rho(self, rho: np.ndarray, method: str) -> np.ndarray:
        """Clamp *rho* to the clip band, counting/warning on engagement."""
        lo = self.rho0 * self.rho_clip[0]
        hi = self.rho0 * self.rho_clip[1]
        clipped = np.clip(rho, lo, hi)
        n_out = int(np.count_nonzero(clipped != rho))
        if n_out:
            self.clip_count[method] += n_out
            if not self._clip_warned:
                self._clip_warned = True
                warnings.warn(
                    f"TaitMurnaghan rho_clip engaged in {method}(): density "
                    f"outside [{lo:g}, {hi:g}] kg/m^3 saturates the EOS at "
                    f"the band edge (zero effective compressibility there). "
                    f"Further engagements are counted in .clip_count; this "
                    f"warning is shown once per instance.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return clipped

    # -- forward: rho -> P ------------------------------------------------

    def pressure(self, rho: float | np.ndarray) -> float | np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if self.rho_clip is not None:
            rho = self._clip_rho(rho, 'pressure')
        return self.P0 + (self.K / self.n) * ((rho / self.rho0) ** self.n - 1.0)

    # -- inverse: P -> rho ------------------------------------------------

    def density(self, P: float | np.ndarray) -> float | np.ndarray:
        P = np.asarray(P, dtype=float)
        ratio = (self.n / self.K) * (P - self.P0) + 1.0
        ratio = np.maximum(ratio, 1e-30)
        rho = self.rho0 * ratio ** (1.0 / self.n)
        if self.rho_clip is not None:
            # Same band as pressure(): keeps density/pressure mutual
            # inverses (bijective inside the band, band-edge saturation
            # outside) instead of the former one-sided clip that broke
            # the round trip at the band edges.
            rho = self._clip_rho(rho, 'density')
        return rho

    # -- thermodynamic derivative ------------------------------------------

    def sound_speed(self, rho: float | np.ndarray) -> float | np.ndarray:
        """c = sqrt(dP/drho) = sqrt((K / rho0) * (rho / rho0)^(n-1)).

        Evaluated on the clipped density, so out-of-band states report
        the band-edge stiffness (the one-sided derivative approached
        from inside the band).  Note the saturating clipped law is flat
        (dP/drho = 0) outside the band; the band-edge value is returned
        as the relevant stiffness scale (e.g. for CFL estimates) and
        the engagement is counted in ``clip_count['sound_speed']``.
        """
        rho = np.asarray(rho, dtype=float)
        if self.rho_clip is not None:
            rho = self._clip_rho(rho, 'sound_speed')
        c_sq = (self.K / self.rho0) * (rho / self.rho0) ** (self.n - 1)
        return np.sqrt(c_sq)

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TaitMurnaghan(rho0={self.rho0}, P0={self.P0}, "
            f"K={self.K:.3e}, n={self.n})"
        )
