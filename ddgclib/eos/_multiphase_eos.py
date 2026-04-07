"""Multiphase equation of state wrapper.

Dispatches to per-phase EOS and stores exact per-phase pressures on
each vertex — no blending or weighted averages.

For **bulk vertices**: computes pressure from the single-phase EOS.
For **interface vertices**: computes pressure for each phase present,
using ``v.m_phase[k] / v.dual_vol_phase[k]`` as the phase density.
Stores results in ``v.p_phase[k]``.

Implements the callable protocol ``__call__(v) -> float`` expected by
``stress_force(..., pressure_model=callable)``.  The returned pressure
is the vertex's own-phase pressure (``v.p_phase[v.phase]``).

Usage
-----
    from ddgclib.eos import TaitMurnaghan, MultiphaseEOS

    eos_outer = TaitMurnaghan(rho0=1000.0, K=1e6)
    eos_drop  = TaitMurnaghan(rho0=800.0, K=8e5)
    meos = MultiphaseEOS([eos_outer, eos_drop])

    # Pass as pressure_model to multiphase_stress_force:
    F = multiphase_stress_force(v, dim=2, mps=mps, HC=HC, pressure_model=meos)
"""
from __future__ import annotations

import numpy as np

from ddgclib.eos._base import EquationOfState


class MultiphaseEOS:
    """Dispatch pressure computation to per-phase EOS.

    Parameters
    ----------
    eos_list : list[EquationOfState]
        EOS instances indexed by phase ID.
    """

    def __init__(self, eos_list: list[EquationOfState]):
        self.eos_list = eos_list
        self.n_phases = len(eos_list)

    def __call__(self, v) -> float:
        """Compute per-phase pressures for vertex *v*.

        Populates ``v.p_phase[k]`` and ``v.rho_phase[k]`` for each
        phase *k*.  Returns ``v.p_phase[v.phase]`` (the own-phase
        pressure) for backward compatibility with the single-pressure
        ``_resolve_pressure`` protocol.
        """
        n = self.n_phases

        # Ensure per-phase arrays exist
        if not hasattr(v, 'p_phase') or v.p_phase is None:
            v.p_phase = np.zeros(n)
        if not hasattr(v, 'rho_phase') or v.rho_phase is None:
            v.rho_phase = np.zeros(n)

        dvp = getattr(v, 'dual_vol_phase', None)
        mp = getattr(v, 'm_phase', None)

        if dvp is not None and mp is not None:
            # Per-phase pressure from per-phase density
            for k in range(n):
                if dvp[k] > 1e-30 and mp[k] > 1e-30:
                    rho_k = mp[k] / dvp[k]
                    v.rho_phase[k] = rho_k
                    v.p_phase[k] = float(self.eos_list[k].pressure(rho_k))
                else:
                    v.rho_phase[k] = 0.0
                    v.p_phase[k] = 0.0
        else:
            # Fallback: single-phase from total mass / total volume
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 1e-30:
                rho = v.m / vol
            else:
                rho = self.eos_list[v.phase].rho0
            p = float(self.eos_list[v.phase].pressure(rho))
            v.rho_phase[v.phase] = rho
            v.p_phase[v.phase] = p

        own_p = v.p_phase[v.phase]
        v.p = own_p
        v.rho = v.rho_phase[v.phase] if v.rho_phase[v.phase] > 0 else v.m / max(getattr(v, 'dual_vol', 1e-30), 1e-30)
        return own_p

    def pressure_for_phase(self, phase_id: int, rho: float) -> float:
        """Direct pressure evaluation for a specific phase."""
        return float(self.eos_list[phase_id].pressure(rho))

    def __repr__(self) -> str:
        return f"MultiphaseEOS(n_phases={self.n_phases})"
