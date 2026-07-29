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


def snapshot_geometry_multiphase(HC, n_phases: int) -> dict[int, dict]:
    """Capture per-vertex pre-retopo per-phase pressure AND sub-volume.

    Returns ``{id(v): {'p_phase': ndarray, 'dual_vol_phase': ndarray}}``.

    The pre-retopo ``dual_vol_phase`` is what lets
    :func:`redistribute_mass_multiphase` decide phase-presence at *v*
    independently of pressure magnitude — without this, a phase at
    reference pressure ``P0=0`` looks identical to an absent phase.
    """
    snap = {}
    for v in HC.V:
        p_phase = getattr(v, 'p_phase', None)
        dvp = getattr(v, 'dual_vol_phase', None)
        snap[id(v)] = {
            'p_phase': (np.array(p_phase, dtype=float).copy()
                        if p_phase is not None else np.zeros(n_phases)),
            'dual_vol_phase': (np.array(dvp, dtype=float).copy()
                               if dvp is not None else np.zeros(n_phases)),
        }
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

def _extract_snapshot_views(snapshot, n_phases: int):
    """Return ``(pressure_view, dual_vol_view)`` from a snapshot dict.

    Accepts both the legacy ``snapshot_pressure_multiphase`` format
    (``{id(v): p_phase_array}``) and the newer
    ``snapshot_geometry_multiphase`` format
    (``{id(v): {'p_phase': ..., 'dual_vol_phase': ...}}``).

    For the legacy format ``dual_vol_view`` is ``None``, which signals
    callers to fall back to the legacy ``p_phase > 1e-30`` guard.
    """
    pressure_view: dict[int, np.ndarray] = {}
    dual_vol_view: dict[int, np.ndarray] | None = {}
    for vid, rec in snapshot.items():
        if isinstance(rec, dict):
            pressure_view[vid] = np.asarray(rec['p_phase'], dtype=float)
            dual_vol_view[vid] = np.asarray(rec['dual_vol_phase'], dtype=float)
        else:
            pressure_view[vid] = np.asarray(rec, dtype=float)
            dual_vol_view = None  # legacy snapshot, no geometry info
    return pressure_view, dual_vol_view


def restore_pressure_multiphase(
    HC,
    mps,
    snapshot: dict,
) -> int:
    """Overwrite ``v.p_phase`` with snapshot pressures after a rebuild.

    Conservative-remap closure (``retopo_remap='conservative'``): a
    connectivity rebuild at frozen vertex positions is a measurement
    change, not a physical compression, so the pressure field that the
    force assembly reads must be exactly invariant across it.

    :func:`redistribute_mass_multiphase` reproduces the snapshot
    pressure field only up to a uniform per-phase offset
    ``~K_k * (scale_k - 1)`` (the global mass-conservation rescale).
    Under per-step Delaunay the measured total phase volume jumps with
    every reconnection, so that offset becomes a per-step pressure jolt
    at the interface.  This helper cancels it by restoring the snapshot
    ``p_phase[k]`` wherever phase *k* persists across the rebuild
    (present in both the snapshot and the new ``dual_vol_phase``),
    then recomputing ``v.p`` under the shared interface-mean
    convention.  Masses (already redistributed, exactly conserved) are
    left untouched; the transient ``p != eos(m/dual_vol)`` mismatch is
    immaterial because the next redistribution regenerates masses from
    the pressure field anyway.

    Parameters
    ----------
    HC : Complex
    mps : MultiphaseSystem
    snapshot : dict
        Geometry-aware snapshot from
        :func:`snapshot_geometry_multiphase` taken at the same vertex
        positions immediately before the connectivity rebuild.

    Returns
    -------
    int
        Number of ``(vertex, phase)`` entries restored.
    """
    from ddgclib.eos._multiphase_eos import interface_mean_pressure

    n_phases = mps.n_phases
    p_snap, dvp_snap = _extract_snapshot_views(snapshot, n_phases)
    n_restored = 0
    for v in HC.V:
        vid = id(v)
        if vid not in p_snap:
            continue  # newly injected vertex: keep EOS pressure
        dvp = getattr(v, 'dual_vol_phase', None)
        if dvp is None:
            continue
        for k in range(n_phases):
            if dvp[k] < 1e-30 or v.m_phase[k] < 1e-30:
                continue  # phase absent on the new duals
            if dvp_snap is not None and dvp_snap[vid][k] < 1e-30:
                continue  # phase newly present at v: keep EOS value
            v.p_phase[k] = p_snap[vid][k]
            n_restored += 1
        # v.p convention identical to compute_phase_pressures
        if getattr(v, 'is_interface', False) or v.phase < 0:
            v.p = interface_mean_pressure(v, n_phases)
        elif 0 <= v.phase < n_phases:
            v.p = float(v.p_phase[v.phase])
    return n_restored


def phase_volume_totals(HC, n_phases: int) -> np.ndarray:
    """Total measured per-phase dual volume ``sum_i dual_vol_phase[k]``."""
    totals = np.zeros(n_phases)
    for v in HC.V:
        dvp = getattr(v, 'dual_vol_phase', None)
        if dvp is not None:
            totals += np.asarray(dvp, dtype=float)
    return totals


def anchor_phase_pressure_levels(
    HC,
    mps,
    vol_mid: np.ndarray,
    vol_new: np.ndarray,
) -> dict:
    """p_ref-style per-phase pressure-level anchor (conservative remap).

    Under redistribution the per-phase pressure LEVEL is the integral of
    per-step global scale factors; per-step Delaunay reconnection makes
    those increments noisy and the noise rectifies into a runaway level
    drift (measured: the oscillating-droplet outer phase ratchets to a
    spurious ~-5 Pa tension in 200 steps, pumping far-field KE).  The
    p_ref benchmark closure fixes the same defect on tet meshes by
    rebuilding volume TARGETS after every retopology and deriving the
    pressure from volume strain relative to the rebuilt targets — a
    state function instead of a noisy integral.

    This is the dual-cell analogue.  Maintained on *mps*:

    - ``_remap_vol_tar[k]``: target phase volume expressed in the
      CURRENT connectivity's measurement gauge.  Initialised to
      *vol_mid* on the first call (the setup-state volume), then
      multiplied by the pure connectivity artifact ratio
      ``vol_new[k]/vol_mid[k]`` at every rebuild (both measured at the
      same frozen vertex positions, so the ratio contains no physics).
    - ``_remap_rho_ref[k]``: reference density consistent with the
      INITIAL pressure level, ``eos_k.density(L_k(0))`` where ``L_k``
      is the volume-weighted mean of ``p_phase[k]``.  Deliberately NOT
      the mass-ledger density ``M_k/V_k`` — setup mass bookkeeping
      (e.g. Young-Laplace mass loading against pre-perturbation
      volumes) may not be volume-consistent, and the anchor must
      reproduce the setup pressure level exactly at t=0.

    The anchored level is ``p_lvl_k = eos_k.pressure(rho_ref_k *
    vol_tar_k / vol_new_k)`` — real global compression moves it with
    the correct EOS stiffness, pure reconnection cancels exactly — and
    the current volume-weighted mean of ``p_phase[k]`` is shifted
    uniformly onto it (structure untouched).  Returns per-phase
    diagnostics.
    """
    n_phases = mps.n_phases
    vol_mid = np.asarray(vol_mid, dtype=float)
    vol_new = np.asarray(vol_new, dtype=float)

    first_call = getattr(mps, '_remap_vol_tar', None) is None
    if first_call:
        mps._remap_vol_tar = vol_mid.copy()
        mps._remap_rho_ref = np.zeros(n_phases)

    diag = {'p_level': [], 'shift': []}
    for k in range(n_phases):
        if vol_mid[k] < 1e-30 or vol_new[k] < 1e-30:
            diag['p_level'].append(0.0)
            diag['shift'].append(0.0)
            continue
        # Rebuild the target in the new measurement gauge (pure
        # connectivity artifact ratio — positions are frozen).
        mps._remap_vol_tar[k] *= vol_new[k] / vol_mid[k]

        # Current volume-weighted mean level of phase k
        num = 0.0
        den = 0.0
        for v in HC.V:
            dvp = getattr(v, 'dual_vol_phase', None)
            if dvp is None or dvp[k] < 1e-30 or v.m_phase[k] < 1e-30:
                continue
            num += float(v.p_phase[k]) * float(dvp[k])
            den += float(dvp[k])
        if den < 1e-30:
            diag['p_level'].append(0.0)
            diag['shift'].append(0.0)
            continue
        L_k = num / den

        if first_call:
            mps._remap_rho_ref[k] = float(mps.phases[k].eos.density(L_k))
        rho_lvl = mps._remap_rho_ref[k] * mps._remap_vol_tar[k] / vol_new[k]
        p_lvl = float(mps.phases[k].eos.pressure(rho_lvl))
        shift = p_lvl - L_k
        for v in HC.V:
            dvp = getattr(v, 'dual_vol_phase', None)
            if dvp is None or dvp[k] < 1e-30 or v.m_phase[k] < 1e-30:
                continue
            v.p_phase[k] = float(v.p_phase[k]) + shift
        diag['p_level'].append(p_lvl)
        diag['shift'].append(shift)

    # Refresh the representative vertex pressure under the shared
    # convention (identical to compute_phase_pressures's tail).
    from ddgclib.eos._multiphase_eos import interface_mean_pressure
    for v in HC.V:
        if getattr(v, 'is_interface', False) or v.phase < 0:
            v.p = interface_mean_pressure(v, n_phases)
        elif 0 <= v.phase < n_phases:
            v.p = float(v.p_phase[v.phase])
    return diag


def redistribute_mass_multiphase(
    HC,
    dim: int,
    mps,
    bV: set | None = None,
    pressure_snapshot: dict | None = None,
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
        Either the legacy
        ``{id(v): p_phase_array_copy}`` from
        :func:`snapshot_pressure_multiphase`, or the geometry-aware
        ``{id(v): {'p_phase': ..., 'dual_vol_phase': ...}}`` from
        :func:`snapshot_geometry_multiphase`.  The geometry-aware
        format gates phase presence at *v* on the pre-retopo
        ``dual_vol_phase[k]`` instead of pressure magnitude, which is
        required for cases at reference pressure ``P0=0`` (otherwise
        the entire reference-pressure phase is skipped).

    Returns
    -------
    dict
        ``per_phase_diagnostics`` list and ``total_mass_before``/``after``.
    """
    n_phases = mps.n_phases

    if pressure_snapshot is None:
        pressure_snapshot = snapshot_geometry_multiphase(HC, n_phases)

    p_snap, dvp_snap = _extract_snapshot_views(pressure_snapshot, n_phases)

    phase_diag = []

    for k in range(n_phases):
        eos_k = mps.phases[k].eos

        targets_k = {}
        M_k_target = 0.0
        for v in HC.V:
            if not _is_redistributable(v, bV, pressure_snapshot):
                continue
            dvp = getattr(v, 'dual_vol_phase', None)
            if dvp is None or dvp[k] < 1e-30:
                continue
            p_k_before = p_snap[id(v)][k]
            # Phase-presence guard.  Prefer the pre-retopo sub-volume
            # when available — pressure can legitimately be zero at the
            # reference state (P0=0), so a pressure-only guard skips
            # whole phases that need redistribution.
            if dvp_snap is not None:
                if dvp_snap[id(v)][k] < 1e-30:
                    continue
            else:
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
