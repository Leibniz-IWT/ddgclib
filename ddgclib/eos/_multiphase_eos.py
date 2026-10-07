"""Multiphase equation of state wrapper.

Dispatches to per-phase EOS and stores exact per-phase pressures on
each vertex — no blending or weighted averages.

For **bulk vertices**: computes pressure from the single-phase EOS.
For **interface vertices**: computes pressure for each phase present,
using ``v.m_phase[k] / v.dual_vol_phase[k]`` as the phase density.
Stores results in ``v.p_phase[k]``.

Implements the callable protocol ``__call__(v) -> float`` expected by
``stress_force(..., pressure_model=callable)``.  The returned pressure
is the vertex's own-phase pressure (``v.p_phase[v.phase]``) for bulk
vertices; for interface vertices (``v.phase == INTERFACE_PHASE``) it is
the mean of the phase pressures present at *v*
(:func:`interface_mean_pressure` — the same convention as
``MultiphaseSystem.compute_phase_pressures``).

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


def interface_mean_pressure(v, n_phases: int) -> float:
    """Representative scalar pressure for an interface vertex.

    Arithmetic mean of ``v.p_phase[k]`` over the phases *geometrically*
    present at ``v`` (``dual_vol_phase[k] > 1e-30`` and
    ``m_phase[k] > 1e-30``).  Presence is keyed on geometry/mass, never
    on the stored pressure value: an exactly-0.0 gauge pressure
    (``P0=0``, ``rho == rho0``) is legitimate and must be included in
    the average (see docs_temp/audit/zero-gauge-pressure.md).

    This is THE convention for interface ``v.p`` — shared by
    ``MultiphaseSystem.compute_phase_pressures`` and
    :meth:`MultiphaseEOS.__call__` so the two can never diverge.
    """
    phases = getattr(v, 'interface_phases', None)
    if phases is None or len(phases) == 0:
        phases = range(n_phases)
    dvp = getattr(v, 'dual_vol_phase', None)
    mp = getattr(v, 'm_phase', None)
    if dvp is None or mp is None:
        return 0.0
    active = [
        float(v.p_phase[k]) for k in phases
        if (0 <= int(k) < n_phases
            and float(dvp[k]) > 1e-30 and float(mp[k]) > 1e-30)
    ]
    return float(np.mean(active)) if active else 0.0


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
        pressure) for bulk vertices, or the mean of the present phase
        pressures (:func:`interface_mean_pressure`) for interface
        vertices, for backward compatibility with the single-pressure
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
            if int(getattr(v, 'phase', 0)) < 0:
                # INTERFACE_PHASE sentinel: the dual cell straddles
                # phases, so m/dual_vol is a *mixture* density and no
                # single-phase EOS applies; indexing eos_list[-1] would
                # silently use the LAST phase (multiphase.py sentinel
                # contract).  Per-phase fields must be populated first.
                raise ValueError(
                    "MultiphaseEOS: cannot compute a phase pressure for an "
                    "interface vertex (v.phase == INTERFACE_PHASE) without "
                    "per-phase arrays (v.m_phase / v.dual_vol_phase). "
                    "Populate them first, e.g. via MultiphaseSystem.refresh()."
                )
            vol = getattr(v, 'dual_vol', 0.0)
            if vol > 1e-30:
                rho = v.m / vol
            else:
                rho = self.eos_list[v.phase].rho0
            p = float(self.eos_list[v.phase].pressure(rho))
            v.rho_phase[v.phase] = rho
            v.p_phase[v.phase] = p

        own_phase = int(getattr(v, 'phase', 0))
        if own_phase >= 0:
            own_p = v.p_phase[own_phase]
            v.rho = (v.rho_phase[own_phase] if v.rho_phase[own_phase] > 0
                     else v.m / max(getattr(v, 'dual_vol', 1e-30), 1e-30))
        else:
            # INTERFACE_PHASE sentinel (-1): never index the per-phase
            # arrays with it (numpy would wrap to the LAST phase).  Use
            # the shared compute_phase_pressures convention instead.
            own_p = interface_mean_pressure(v, n)
            # Representative density: mixture density of the straddling
            # dual cell.
            v.rho = v.m / max(getattr(v, 'dual_vol', 1e-30), 1e-30)
        v.p = own_p
        return own_p

    def pressure_for_phase(self, phase_id: int, rho: float) -> float:
        """Direct pressure evaluation for a specific phase."""
        return float(self.eos_list[phase_id].pressure(rho))

    def __repr__(self) -> str:
        return f"MultiphaseEOS(n_phases={self.n_phases})"
