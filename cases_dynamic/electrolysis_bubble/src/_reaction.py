"""Placeholder electrolysis reaction: linear gas mass injection.

Adds mass to the gas phase at a constant rate ``dm/dt``.  Mass is
distributed across phase-1 (gas) dual-volume weightings, which for
a fully-enclosed bubble is equivalent to a uniform gas source.

No charge transport, species diffusion, or Nernst/Butler-Volmer
coupling here -- this is a hook for the eventual reaction-diffusion
pipeline.
"""
from __future__ import annotations

import numpy as np


def inject_gas_mass(HC, mps, dm_dt: float, dt: float,
                    gas_phase: int = 1) -> float:
    """Add ``dm_dt * dt`` kg of gas mass to the gas-phase sub-volumes.

    Distribution is proportional to each vertex's phase-``gas_phase``
    dual volume so the per-phase density on interior gas vertices
    stays uniform after injection (before the next retopology).

    Returns the total mass actually added (0.0 if no gas phase
    present).
    """
    total_vol = 0.0
    for v in HC.V:
        vol_k = float(v.dual_vol_phase[gas_phase])
        if np.isfinite(vol_k) and vol_k > 1e-30:
            total_vol += vol_k
    if total_vol <= 1e-30:
        return 0.0

    dm_total = dm_dt * dt
    for v in HC.V:
        vol_k = float(v.dual_vol_phase[gas_phase])
        if np.isfinite(vol_k) and vol_k > 1e-30:
            v.m_phase[gas_phase] += dm_total * (vol_k / total_vol)

    # Refresh aggregate mass and phase pressure so the next integrator
    # step sees the new state.
    for v in HC.V:
        if np.all(np.isfinite(v.m_phase)):
            v.m = float(np.sum(v.m_phase))
        else:
            v.m = 0.0
    mps.compute_phase_pressures(HC)

    return float(dm_total)
