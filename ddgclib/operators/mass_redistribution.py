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

from functools import partial

import numpy as np


# ---------------------------------------------------------------------------
# Pressure snapshots
# ---------------------------------------------------------------------------

def snapshot_pressure(HC) -> dict[int, float]:
    """Capture ``{id(v): v.p}`` for all vertices before retriangulation.

    Vertex ``id(v)`` is stable across retriangulation (only edges change).
    """
    return {id(v): float(getattr(v, 'p', 0.0)) for v in HC.V}


def _connectivity_dual_volumes(HC, dim: int) -> dict | None:
    """Exact barycentric dual volumes of the CURRENT connectivity at the
    CURRENT vertex positions, or ``None`` when they cannot be measured.

    Reads the top-simplex cache ``HC._simplices`` (the connectivity of the
    previous retopology).  When the cache is missing, or references
    vertices that have left ``HC.V`` (outlet deletion, merges), it is
    rebuilt from the 1-skeleton in 2D; in 3D there is no such rebuild and
    ``None`` is returned.
    """
    from hyperct.ddg import simplex_dual_volumes

    simplices = getattr(HC, '_simplices', None)
    if simplices is not None:
        live = {id(v) for v in HC.V}
        if any(id(vv) not in live for s in simplices for vv in s):
            simplices = None
    if simplices is None:
        if dim != 2:
            return None
        from hyperct.ddg import rebuild_simplex_cache_2d
        rebuild_simplex_cache_2d(HC)
    return simplex_dual_volumes(HC, dim)


def snapshot_pressure_fresh(HC, dim: int, eos) -> dict[int, float]:
    """Capture ``{id(v): eos.pressure(v.m / Vol_i)}`` with ``Vol_i``
    re-measured on the current (pre-rebuild) connectivity at the current
    positions: the pressure the fluid has NOW, after the last move.

    This is the snapshot of the single-phase conservative remap
    (``_retopologize(retopo_remap='conservative')``).  The stale ``v.p``
    of :func:`snapshot_pressure` was written by the last force
    evaluation, before the move, so re-targeting masses to it erases the
    step's compression (laneK P3/P12: stable but ``p`` pinned at the IC).

    Vertices without a measurable volume are left out of the snapshot and
    therefore out of the redistribution.  When the connectivity cannot be
    re-measured (3D without a valid simplex cache: the first call on a
    builder mesh, or after vertices were deleted) the cached
    ``v.dual_vol`` is used, which is exact as long as no vertex has moved
    since it was cached.
    """
    vols = _connectivity_dual_volumes(HC, dim)
    snap: dict[int, float] = {}
    for v in HC.V:
        vol = vols.get(v, 0.0) if vols is not None else getattr(v, 'dual_vol', 0.0)
        if vol > 1e-30:
            snap[id(v)] = float(eos.pressure(v.m / vol))
    return snap


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
    include_frozen: bool = False,
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
    include_frozen : bool
        If True, vertices in *bV* are re-targeted too (every vertex with
        a positive dual volume that is in the snapshot).  Required by the
        conservative remap: a reconnection changes the dual volume of a
        frozen wall vertex exactly as it does an interior one, and the
        EOS reads that jump through the wall cell's pressure (laneK P2:
        the interior-only remap blows up).

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
    if include_frozen:
        bV = None

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
    adopted: dict | None = None,
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
    adopted : dict or None
        ``{id(v): {k: p}}`` from :func:`redistribute_mass_multiphase`
        with ``ledger='volume'``: the local phase-k pressure a phase
        that is NEW at *v* was targeted at.  Restored like a snapshot
        value (the phase is otherwise skipped as "newly present").

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
                # phase newly present at v: keep the EOS value, unless
                # the volume ledger targeted it at a local pressure
                if adopted is not None and k in adopted.get(vid, ()):
                    v.p_phase[k] = adopted[vid][k]
                    n_restored += 1
                continue
            v.p_phase[k] = p_snap[vid][k]
            n_restored += 1
        # v.p convention identical to compute_phase_pressures
        if getattr(v, 'is_interface', False) or v.phase < 0:
            v.p = interface_mean_pressure(v, n_phases)
        elif 0 <= v.phase < n_phases:
            v.p = float(v.p_phase[v.phase])
    return n_restored


def evolve_snapshot_local_strain(HC, mps, snapshot: dict) -> dict:
    """Advance a geometry snapshot by the local Lagrangian strain.

    laneH (2026-07-30) helper for ``projection_every > 1`` under the
    conservative remap: on off-cadence calls the field that
    redistribution/restore must reproduce across the connectivity
    rebuild is the pre-call pressure field ADVANCED by this step's
    local EOS compression response — not the raw
    ``eos(m / dual_vol)`` recompute, which loses the restore/anchor
    level corrections that live in ``p_phase`` but not in the mass
    ledger (measured: an off-cadence call at frozen positions jolts
    the field by ~8.8 Pa on the coarse fixture if the raw recompute
    is used; exactly 0 with this construction).

    For each vertex present in *snapshot* and each phase *k* present
    in both the snapshot and the current (refreshed, old-connectivity)
    ``dual_vol_phase``::

        rho_k   = eos_k.density(p_snap_k)
        p_new_k = eos_k.pressure(rho_k * dvp_snap_k / dvp_now_k)

    i.e. mass-conserving compression of the parcel from its snapshot
    sub-volume to its current sub-volume.  At frozen positions
    ``dvp_now == dvp_snap`` and the snapshot is returned unchanged
    (exact neutrality).  The returned snapshot's ``dual_vol_phase`` is
    the CURRENT one (the geometry the pressures now describe), which
    is also what the redistribution/restore presence gates should use.

    Parameters
    ----------
    HC : Complex
    mps : MultiphaseSystem
    snapshot : dict
        Geometry-aware snapshot from
        :func:`snapshot_geometry_multiphase` taken before the call.

    Returns
    -------
    dict
        New snapshot in the same format; vertices absent from
        *snapshot* stay absent (newly injected vertices keep their
        EOS pressure downstream).
    """
    n_phases = mps.n_phases
    out: dict[int, dict] = {}
    for v in HC.V:
        vid = id(v)
        rec = snapshot.get(vid)
        if rec is None:
            continue
        dvp_now = getattr(v, 'dual_vol_phase', None)
        if dvp_now is None:
            dvp_now = np.zeros(n_phases)
        dvp_now = np.asarray(dvp_now, dtype=float)
        p_old = np.asarray(rec['p_phase'], dtype=float)
        dvp_old = np.asarray(rec['dual_vol_phase'], dtype=float)
        p_new = p_old.copy()
        for k in range(n_phases):
            if dvp_old[k] > 1e-30 and dvp_now[k] > 1e-30:
                eos_k = mps.phases[k].eos
                rho_k = float(eos_k.density(p_old[k]))
                p_new[k] = float(
                    eos_k.pressure(rho_k * dvp_old[k] / dvp_now[k])
                )
        out[vid] = {'p_phase': p_new, 'dual_vol_phase': dvp_now.copy()}
    return out


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


#: Rules for a phase that appears or disappears at a vertex across a
#: rebuild (method axis ``phase_ledger``), see
#: :func:`redistribute_mass_multiphase`.
LEDGER_RULES = ('snapshot', 'volume', 'adopt')


def _local_phase_pressure(v, k: int, p_snap, dvp_snap) -> float | None:
    """Snapshot phase-k pressure around *v*: the sub-volume weighted mean
    over the 1-ring neighbours at which phase *k* was present before the
    rebuild (``None`` when no neighbour had it)."""
    num = den = 0.0
    for w in v.nn:
        wid = id(w)
        dvp_w = dvp_snap.get(wid)
        if dvp_w is None or dvp_w[k] < 1e-30:
            continue
        num += float(p_snap[wid][k]) * float(dvp_w[k])
        den += float(dvp_w[k])
    return num / den if den > 0.0 else None


def _phase_level(k: int, p_snap, dvp_snap) -> float | None:
    """Snapshot phase-k pressure level: the sub-volume weighted mean over
    every vertex at which phase *k* was present (``None`` if nowhere)."""
    num = den = 0.0
    for vid, dvp_v in dvp_snap.items():
        if dvp_v[k] < 1e-30:
            continue
        num += float(p_snap[vid][k]) * float(dvp_v[k])
        den += float(dvp_v[k])
    return num / den if den > 0.0 else None


def _targets_by_snapshot(HC, k, eos_k, bV, pressure_snapshot, p_snap,
                         dvp_snap):
    """Targets of the ``'snapshot'`` ledger rule (the historic loop)."""
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
    return targets_k, M_k_target, 0.0, {}


def _targets_by_volume(HC, k, eos_k, bV, pressure_snapshot, p_snap,
                       dvp_snap, release=True):
    """Targets of the ``'volume'`` ledger rule: the phase-k mass follows
    the phase-k sub-volume across the rebuild (``'adopt'`` with
    ``release=False``: a lost phase keeps its mass as inertia).

    Like the snapshot rule for a vertex that had phase *k* before and has
    it now.  In addition, at every vertex of the snapshot with a dual
    cell (frozen ones included):

    * phase *k* NEW at *v* (no snapshot sub-volume, a sub-volume now,
      e.g. an interface vertex that a flip gave its first bulk neighbour
      of that phase): targeted at the local snapshot pressure of phase
      *k* (:func:`_local_phase_pressure`, else :func:`_phase_level`),
      so it joins the conserving rescale instead of staying massless;
    * phase *k* GONE from *v* (mass, no sub-volume now): its mass is
      released into the phase-k pool that the rescale conserves.

    Returns ``(targets, M_target, M_released, adopted)`` where
    ``adopted = {id(v): {k: p_local}}`` lists the pressures the new
    entries were targeted at (for :func:`restore_pressure_multiphase`).
    """
    targets_k = {}
    M_k_target = 0.0
    M_k_released = 0.0
    adopted: dict[int, dict[int, float]] = {}
    level = None
    level_done = False
    for v in HC.V:
        vid = id(v)
        if vid not in pressure_snapshot:
            continue  # newly injected vertex: untouched, as before
        if getattr(v, 'dual_vol', 0.0) < 1e-30:
            continue  # degenerate cell: untouched, as before
        dvp = getattr(v, 'dual_vol_phase', None)
        m_phase = getattr(v, 'm_phase', None)
        if dvp is None or m_phase is None:
            continue
        present_before = not (dvp_snap[vid][k] < 1e-30)
        if dvp[k] < 1e-30:
            if release and m_phase[k] > 1e-30:
                # phase k gone from this cell: mass without a sub-volume
                M_k_released += float(m_phase[k])
                m_phase[k] = 0.0
            continue
        frozen = bV is not None and v in bV
        if present_before:
            if frozen:
                continue  # frozen cells keep their mass (as before)
            p_k = p_snap[vid][k]
        else:
            p_k = _local_phase_pressure(v, k, p_snap, dvp_snap)
            if p_k is None:
                if not level_done:
                    level = _phase_level(k, p_snap, dvp_snap)
                    level_done = True
                p_k = level
            if p_k is None:
                continue  # phase k was nowhere: nothing to adopt
            adopted.setdefault(vid, {})[k] = float(p_k)
        rho_target = float(eos_k.density(p_k))
        rho_target = max(rho_target, 1e-30)
        m_target = rho_target * dvp[k]
        targets_k[vid] = m_target
        M_k_target += m_target
    return targets_k, M_k_target, M_k_released, adopted


def redistribute_mass_multiphase(
    HC,
    dim: int,
    mps,
    bV: set | None = None,
    pressure_snapshot: dict | None = None,
    ledger: str = 'volume',
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
    ledger : {'snapshot', 'volume'}
        What becomes of a phase that appears at or disappears from a
        vertex across the rebuild (method axis ``phase_ledger``).

        - ``'snapshot'`` (default, the historic behaviour): a phase is
          re-targeted only where the snapshot had it.  A phase that
          APPEARS at a vertex (a flip gives an interface vertex its
          first bulk neighbour of that phase) gets no mass, and
          ``MultiphaseSystem.compute_phase_pressures`` then publishes
          ``p_phase[k] = 0`` ABSOLUTE for it while the force operator
          reads the phase as present (sub-volume > 0): a pressure hole
          of ``P0`` on every face of that vertex.  Invisible at
          ``P0 = 0`` (the droplet cases); at ``P0 = 101325`` Pa it
          ejects the neighbouring cell (dam break, laneF 2026-10-05:
          439 N on a 1.1e-4 kg air cell).  A phase that DISAPPEARS keeps
          its mass without a sub-volume (stranded inertia).
        - ``'volume'``: the per-phase mass follows the per-phase
          sub-volume (:func:`_targets_by_volume`): a new phase is
          targeted at the local snapshot pressure and a lost phase
          releases its mass to the pool, both inside the exact per-phase
          conservation.  Needs the geometry-aware snapshot.
        - ``'adopt'``: as ``'volume'`` for a new phase; a lost phase
          keeps its mass without a sub-volume (inertia stays with the
          vertex, the force reads the phase as absent).

    Returns
    -------
    dict
        ``per_phase_diagnostics`` list (with the released mass and the
        number of adopted entries per phase) and ``adopted``
        (``{id(v): {k: p_local}}``, empty under ``'snapshot'``).
    """
    if ledger not in LEDGER_RULES:
        raise ValueError(f"ledger must be one of {LEDGER_RULES}, got "
                         f"{ledger!r}")
    n_phases = mps.n_phases

    if pressure_snapshot is None:
        pressure_snapshot = snapshot_geometry_multiphase(HC, n_phases)

    p_snap, dvp_snap = _extract_snapshot_views(pressure_snapshot, n_phases)
    if ledger != 'snapshot' and dvp_snap is None:
        raise ValueError(f"ledger={ledger!r} needs the geometry-aware "
                         "snapshot (snapshot_geometry_multiphase)")
    if ledger == 'snapshot':
        build_targets = _targets_by_snapshot
    else:
        build_targets = partial(_targets_by_volume,
                                release=(ledger == 'volume'))

    phase_diag = []
    adopted_all: dict[int, dict[int, float]] = {}

    for k in range(n_phases):
        eos_k = mps.phases[k].eos

        targets_k, M_k_target, M_k_released, adopted_k = build_targets(
            HC, k, eos_k, bV, pressure_snapshot, p_snap, dvp_snap)
        for vid, kp in adopted_k.items():
            adopted_all.setdefault(vid, {}).update(kp)

        # Conservation target: only the mass of vertices that WILL be
        # modified (in targets_k).  Vertices with p_phase[k]=0 or
        # dvp[k]=0 keep their mass unchanged and must not inflate the sum.
        M_k_total = 0.0
        for v in HC.V:
            if id(v) in targets_k:
                m_phase = getattr(v, 'm_phase', None)
                if m_phase is not None:
                    M_k_total += m_phase[k]
        M_k_total += M_k_released

        # Scale and assign
        if M_k_target < 1e-30 or M_k_total < 1e-30:
            phase_diag.append({
                'phase': k,
                'total_mass_before': M_k_total,
                'total_mass_after': M_k_total,
                'scale_factor': 1.0,
                'mass_released': M_k_released,
                'n_adopted': len(adopted_k),
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
            'mass_released': M_k_released,
            'n_adopted': len(adopted_k),
        })

    # Recompute total mass from per-phase sums
    for v in HC.V:
        m_phase = getattr(v, 'm_phase', None)
        if m_phase is not None:
            v.m = float(np.sum(m_phase))

    return {
        'per_phase_diagnostics': phase_diag,
        'adopted': adopted_all,
    }
