"""Per-phase mass sources (an electrolysis reaction, evaporation, ...).

:func:`add_phase_mass` is the library form of the gas injection that
``cases_dynamic/electrolysis_bubble/src/_reaction.py`` carried as case
code until laneG (2026-10-06): the added mass is spread over the sub-volumes
of the phase, the vertex masses and the phase pressures are refreshed,
and, when the conservative retopology remap is running
(``remap='conservative'``: :func:`anchor_phase_pressure_levels` keeps a
reference density per phase on the ``MultiphaseSystem``), the reference
density of that phase is scaled by the same factor as its mass.  Without
that scaling the anchored pressure level of the phase is a function of its
volume only, so a mass source is invisible to the force: measured on the
3D electrolysis bubble (refinement 1/1, 2300 steps), the gas pressure the
EOS reads climbed to 4807 Pa while the anchored level stayed at the
Laplace value and the bubble volume did not change (5.1236e-9 m^3 at every
record), against +5.8 % of volume on the same run without the remap.
"""
from __future__ import annotations

import numpy as np

__all__ = ['add_phase_mass']


def add_phase_mass(HC, mps, phase: int, dm: float) -> float:
    """Add *dm* kg to phase *phase*, in proportion to the sub-volumes.

    Every vertex with a sub-volume ``dual_vol_phase[phase] > 1e-30``
    receives ``dm * dual_vol_phase[phase] / sum``, so the density of the
    phase is raised uniformly where it is present (before the next
    retopology).  ``v.m`` and the phase pressures
    (``mps.compute_phase_pressures``) are refreshed.  If the conservative
    remap keeps a reference density for the level anchor
    (``mps._remap_rho_ref``), it is multiplied by ``(M + dm) / M`` so the
    anchored pressure level follows the mass ledger.

    Returns the mass added (``0.0`` when the phase has no sub-volume
    anywhere; nothing is changed then).
    """
    total_vol = 0.0
    total_mass = 0.0
    for v in HC.V:
        vol_k = float(v.dual_vol_phase[phase])
        if np.isfinite(vol_k) and vol_k > 1e-30:
            total_vol += vol_k
            total_mass += float(v.m_phase[phase])
    if total_vol <= 1e-30:
        return 0.0
    dm = float(dm)
    for v in HC.V:
        vol_k = float(v.dual_vol_phase[phase])
        if np.isfinite(vol_k) and vol_k > 1e-30:
            v.m_phase[phase] += dm * (vol_k / total_vol)
    for v in HC.V:
        if np.all(np.isfinite(v.m_phase)):
            v.m = float(np.sum(v.m_phase))
        else:
            v.m = 0.0
    rho_ref = getattr(mps, '_remap_rho_ref', None)
    if rho_ref is not None and total_mass > 0.0:
        rho_ref[phase] *= (total_mass + dm) / total_mass
    mps.compute_phase_pressures(HC)
    return dm
