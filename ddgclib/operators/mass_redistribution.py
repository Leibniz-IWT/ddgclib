"""Pressure-preserving mass redistribution after retriangulation.

When a Lagrangian mesh is retriangulated (Delaunay reconnection), dual cell
volumes change even though vertex positions barely moved.  Since the EOS
computes ``P = eos.pressure(m / Vol)``, this causes spurious pressure
discontinuities.

This module redistributes vertex masses after retriangulation so that the
pre-retriangulation pressure field is preserved:

    m_new_i = eos.density(p_before_i) * Vol_new_i

Total mass is conserved exactly via global scaling.

Boundary-condition awareness
----------------------------
- **Wall vertices** (in ``bV``): excluded — mass left unchanged.
- **Newly injected vertices** (not in pressure snapshot): excluded — their
  mass was set by the inlet BC and should not be altered.
- **Open outlets**: deletion has not yet happened at redistribution time
  (BCs run after retopo+integration), so all current interior vertices
  participate normally.
- **Periodic domains**: no mass flux — same as closed domain.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pressure snapshots
# ---------------------------------------------------------------------------

def snapshot_pressure(HC) -> dict[int, float]:
    """Capture ``{id(v): v.p}`` for all vertices before retriangulation.

    Vertex ``id(v)`` is stable across retriangulation (only edges change).
    """
    return {id(v): float(getattr(v, 'p', 0.0)) for v in HC.V}


def snapshot_pressure_multiphase(HC, n_phases: int) -> dict[int, np.ndarray]:
    """Capture ``{id(v): v.p_phase.copy()}`` before retriangulation."""
    snap = {}
    for v in HC.V:
        p_phase = getattr(v, 'p_phase', None)
        if p_phase is not None:
            snap[id(v)] = np.array(p_phase, dtype=float).copy()
        else:
            snap[id(v)] = np.zeros(n_phases)
    return snap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_redistributable(v, bV, pressure_snapshot) -> bool:
    """True if *v* should participate in mass redistribution."""
    if bV is not None and v in bV:
        return False  # wall / frozen boundary
    if getattr(v, 'dual_vol', 0.0) < 1e-30:
        return False  # degenerate or boundary vertex
    if id(v) not in pressure_snapshot:
        return False  # newly injected vertex
    return True


def _compute_conserved_mass(HC, bV, pressure_snapshot) -> float:
    """Total mass of interior pre-existing vertices (the conservation target)."""
    M = 0.0
    for v in HC.V:
        if _is_redistributable(v, bV, pressure_snapshot):
            M += v.m
    return M


# ---------------------------------------------------------------------------
# Single-phase redistribution
# ---------------------------------------------------------------------------

def redistribute_mass_single_phase(
    HC,
    dim: int,
    eos,
    bV: set | None = None,
    pressure_snapshot: dict[int, float] | None = None,
) -> dict:
    """Pressure-preserving mass redistribution after retriangulation.

    For every interior vertex that existed before retriangulation, compute
    the mass that would reproduce its pre-retriangulation pressure at the
    new dual volume:

        m_target_i = eos.density(p_before_i) * Vol_new_i

    Then scale all target masses by ``M_total / M_target_sum`` to enforce
    exact total mass conservation.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (post-retriangulation, duals already cached).
    dim : int
        Spatial dimension.
    eos : EquationOfState
        Must have ``.density(P)`` inverse method.
    bV : set or None
        Frozen boundary vertices (excluded from redistribution).
    pressure_snapshot : dict or None
        ``{id(v): p_before}`` captured before retriangulation.
        If ``None``, falls back to current ``v.p`` (less accurate).

    Returns
    -------
    dict
        Diagnostics: ``total_mass_before``, ``total_mass_after``,
        ``max_abs_pressure_change``, ``scale_factor``.
    """
    if pressure_snapshot is None:
        # Fallback: use current v.p (already reflects new dual volumes
        # if EOS was evaluated, but better than nothing)
        pressure_snapshot = snapshot_pressure(HC)

    # Compute target masses for redistributable vertices
    targets = {}
    M_target_sum = 0.0
    for v in HC.V:
        if not _is_redistributable(v, bV, pressure_snapshot):
            continue
        p_before = pressure_snapshot[id(v)]
        vol_new = getattr(v, 'dual_vol', 0.0)
        rho_target = float(eos.density(p_before))
        rho_target = max(rho_target, 1e-30)
        m_target = rho_target * vol_new
        targets[id(v)] = m_target
        M_target_sum += m_target

    # Conservation target: only count mass of vertices that WILL be
    # modified (in targets).  This ensures vertices excluded from
    # redistribution don't inflate or deflate the scaling factor.
    M_total = sum(v.m for v in HC.V if id(v) in targets)

    # Global scaling to conserve total mass exactly
    if M_target_sum < 1e-30 or M_total < 1e-30:
        return {
            'total_mass_before': M_total,
            'total_mass_after': M_total,
            'max_abs_pressure_change': 0.0,
            'scale_factor': 1.0,
        }

    scale = M_total / M_target_sum

    # Assign scaled target masses
    max_dp = 0.0
    for v in HC.V:
        vid = id(v)
        if vid not in targets:
            continue
        p_before = pressure_snapshot[vid]
        v.m = targets[vid] * scale

        # Track pressure change for diagnostics
        vol = getattr(v, 'dual_vol', 0.0)
        if vol > 1e-30:
            p_after = float(eos.pressure(v.m / vol))
            dp = abs(p_after - p_before)
            if dp > max_dp:
                max_dp = dp

    # Machine-precision fixup: distribute any floating-point residual
    M_after = 0.0
    n_redist = 0
    for v in HC.V:
        if id(v) in targets:
            M_after += v.m
            n_redist += 1

    residual = M_total - M_after
    if abs(residual) > 0.0 and n_redist > 0:
        correction = residual / n_redist
        for v in HC.V:
            if id(v) in targets:
                v.m += correction

    return {
        'total_mass_before': M_total,
        'total_mass_after': M_total,
        'max_abs_pressure_change': max_dp,
        'scale_factor': scale,
    }


# ---------------------------------------------------------------------------
# Multiphase redistribution
# ---------------------------------------------------------------------------

def redistribute_mass_multiphase(
    HC,
    dim: int,
    mps,
    bV: set | None = None,
    pressure_snapshot: dict[int, np.ndarray] | None = None,
) -> dict:
    """Per-phase pressure-preserving mass redistribution.

    For each phase *k* independently, computes target per-phase mass:

        m_target_k = eos_k.density(p_phase_k_before) * dual_vol_phase_k_new

    and scales to conserve ``sum(v.m_phase[k])`` per phase.

    Parameters
    ----------
    HC : Complex
    dim : int
    mps : MultiphaseSystem
        Provides ``.phases[k].eos`` and ``.n_phases``.
    bV : set or None
    pressure_snapshot : dict or None
        ``{id(v): p_phase_array_copy}`` from :func:`snapshot_pressure_multiphase`.

    Returns
    -------
    dict
        ``per_phase_diagnostics`` list and ``total_mass_before``/``after``.
    """
    n_phases = mps.n_phases

    if pressure_snapshot is None:
        pressure_snapshot = snapshot_pressure_multiphase(HC, n_phases)

    phase_diag = []

    for k in range(n_phases):
        eos_k = mps.phases[k].eos

        # Compute target per-phase masses (only vertices with valid
        # pressure snapshot AND positive sub-volume for this phase)
        targets_k = {}
        M_k_target = 0.0
        for v in HC.V:
            if not _is_redistributable(v, bV, pressure_snapshot):
                continue
            dvp = getattr(v, 'dual_vol_phase', None)
            if dvp is None or dvp[k] < 1e-30:
                continue
            p_k_before = pressure_snapshot[id(v)][k]
            if p_k_before < 1e-30:
                continue
            rho_target = float(eos_k.density(p_k_before))
            rho_target = max(rho_target, 1e-30)
            m_target = rho_target * dvp[k]
            targets_k[id(v)] = m_target
            M_k_target += m_target

        # Conservation target: only the mass of vertices that WILL be
        # modified (in targets_k).  Vertices with p_phase[k]=0 or
        # dvp[k]=0 keep their mass unchanged and must not inflate the sum.
        M_k_total = 0.0
        for v in HC.V:
            if id(v) in targets_k:
                m_phase = getattr(v, 'm_phase', None)
                if m_phase is not None:
                    M_k_total += m_phase[k]

        # Scale and assign
        if M_k_target < 1e-30 or M_k_total < 1e-30:
            phase_diag.append({
                'phase': k,
                'total_mass_before': M_k_total,
                'total_mass_after': M_k_total,
                'scale_factor': 1.0,
            })
            continue

        scale_k = M_k_total / M_k_target

        for v in HC.V:
            vid = id(v)
            if vid not in targets_k:
                continue
            v.m_phase[k] = targets_k[vid] * scale_k

        # Machine-precision fixup
        M_k_after = 0.0
        n_k = 0
        for v in HC.V:
            if id(v) in targets_k:
                M_k_after += v.m_phase[k]
                n_k += 1
        residual = M_k_total - M_k_after
        if abs(residual) > 0.0 and n_k > 0:
            correction = residual / n_k
            for v in HC.V:
                if id(v) in targets_k:
                    v.m_phase[k] += correction

        phase_diag.append({
            'phase': k,
            'total_mass_before': M_k_total,
            'total_mass_after': M_k_total,
            'scale_factor': scale_k,
        })

    # Recompute total mass from per-phase sums
    for v in HC.V:
        m_phase = getattr(v, 'm_phase', None)
        if m_phase is not None:
            v.m = float(np.sum(m_phase))

    return {
        'per_phase_diagnostics': phase_diag,
    }
