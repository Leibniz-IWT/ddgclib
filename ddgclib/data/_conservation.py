"""Conservation diagnostics for dynamic simulations.

Computes snapshot-time invariants (kinetic energy, momentum, mass, volume,
per-phase breakdowns, velocity/pressure/edge-length extrema) that every
dynamic benchmark can check for drift. Stateless: call once per snapshot.

Usage
-----
    from ddgclib.data import compute_conservation

    diag = compute_conservation(HC, dim=2)
    # -> {'ke': ..., 'momentum': array([...]), 'mass_total': ...,
    #     'mass_phase': [...], 'volume_total': ..., 'n_vertices': ...,
    #     'h_min': ..., 'h_max': ..., 'u_max': ..., 'p_min': ...,
    #     'p_max': ...}

When used through :class:`StateHistory` with ``conservation=True``, the
diagnostics are merged into each snapshot's ``diagnostics`` dict.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _infer_dim(HC, dim: Optional[int]) -> int:
    if dim is not None:
        return int(dim)
    for v in HC.V:
        return len(v.x_a)
    return 0


def _vertex_mass(v, fallback_vol: bool = True) -> float:
    """Return scalar mass for a vertex.

    Prefers ``v.m`` (set by mass-aware ICs / per-phase bookkeeping);
    falls back to ``v.dual_vol`` if ``v.m`` is absent or zero and
    ``fallback_vol=True`` (useful for pure-CFD cases where mass is
    implicit in the dual volume).
    """
    m = getattr(v, 'm', None)
    if m is None or (fallback_vol and float(m) == 0.0):
        dv = getattr(v, 'dual_vol', None)
        if dv is not None:
            return float(dv)
        return 0.0
    return float(m)


def _phase_masses(v, n_phases: int) -> Optional[np.ndarray]:
    mp = getattr(v, 'm_phase', None)
    if mp is None:
        return None
    arr = np.asarray(mp, dtype=float)
    if arr.shape[0] < n_phases:
        out = np.zeros(n_phases)
        out[: arr.shape[0]] = arr
        return out
    return arr[:n_phases].copy()


def _phase_volumes(v, n_phases: int) -> Optional[np.ndarray]:
    vp = getattr(v, 'dual_vol_phase', None)
    if vp is None:
        return None
    arr = np.asarray(vp, dtype=float)
    if arr.shape[0] < n_phases:
        out = np.zeros(n_phases)
        out[: arr.shape[0]] = arr
        return out
    return arr[:n_phases].copy()


def _detect_n_phases(HC) -> int:
    n = 0
    for v in HC.V:
        mp = getattr(v, 'm_phase', None)
        if mp is not None:
            n = max(n, len(mp))
    return n


def compute_conservation(HC, dim: Optional[int] = None) -> dict[str, Any]:
    """Compute snapshot-time conservation diagnostics.

    Parameters
    ----------
    HC : Complex
        Simplicial complex. Vertices must have ``v.u`` and (``v.m`` or
        ``v.dual_vol``). ``v.p``, ``v.m_phase``, ``v.dual_vol_phase`` are
        read opportunistically if present.
    dim : int, optional
        Spatial dimension. Inferred from ``v.x_a`` if not given.

    Returns
    -------
    dict
        Scalar and small-array diagnostics suitable for JSON serialisation
        after conversion via :func:`as_jsonable` — keys include ``ke``,
        ``momentum`` (length ``dim``), ``mass_total``, ``volume_total``,
        ``n_vertices``, ``h_min``, ``h_max``, ``u_max``, ``u_min``,
        ``p_max``, ``p_min``; per-phase variants ``ke_phase``,
        ``mass_phase``, ``volume_phase`` are present only when
        ``v.m_phase``/``v.dual_vol_phase`` exist.
    """
    dim = _infer_dim(HC, dim)
    n_phases = _detect_n_phases(HC)

    ke_total = 0.0
    momentum = np.zeros(dim) if dim > 0 else np.zeros(0)
    mass_total = 0.0
    volume_total = 0.0

    ke_phase = np.zeros(n_phases) if n_phases else None
    mass_phase = np.zeros(n_phases) if n_phases else None
    volume_phase = np.zeros(n_phases) if n_phases else None

    u_sq_max = 0.0
    u_sq_min = np.inf
    p_max = -np.inf
    p_min = np.inf
    n_vertices = 0
    any_p = False

    for v in HC.V:
        n_vertices += 1
        u = np.asarray(v.u, dtype=float)[:dim] if dim else np.zeros(0)
        m = _vertex_mass(v)
        mass_total += m
        vol = float(getattr(v, 'dual_vol', 0.0) or 0.0)
        volume_total += vol

        if dim:
            u2 = float(np.dot(u, u))
            ke_total += 0.5 * m * u2
            momentum += m * u
            if u2 > u_sq_max:
                u_sq_max = u2
            if u2 < u_sq_min:
                u_sq_min = u2

        p = getattr(v, 'p', None)
        if p is not None:
            any_p = True
            pf = float(p)
            if pf > p_max:
                p_max = pf
            if pf < p_min:
                p_min = pf

        if n_phases:
            mp = _phase_masses(v, n_phases)
            vp = _phase_volumes(v, n_phases)
            if mp is not None:
                mass_phase += mp
                if dim:
                    ke_phase += 0.5 * mp * u2
            if vp is not None:
                volume_phase += vp

    h_min, h_max = _edge_length_extrema(HC, dim)

    diag: dict[str, Any] = {
        'ke': ke_total,
        'momentum': momentum,
        'mass_total': mass_total,
        'volume_total': volume_total,
        'n_vertices': n_vertices,
        'h_min': h_min,
        'h_max': h_max,
        'u_max': float(np.sqrt(u_sq_max)) if n_vertices else 0.0,
        'u_min': float(np.sqrt(u_sq_min)) if (n_vertices and u_sq_min != np.inf) else 0.0,
    }
    if any_p:
        diag['p_min'] = p_min
        diag['p_max'] = p_max
    if n_phases:
        diag['ke_phase'] = ke_phase
        diag['mass_phase'] = mass_phase
        diag['volume_phase'] = volume_phase
    return diag


def _edge_length_extrema(HC, dim: int) -> tuple[float, float]:
    h_min = np.inf
    h_max = 0.0
    for v in HC.V:
        vid = id(v)
        x_i = np.asarray(v.x_a, dtype=float)[:dim]
        for v_j in v.nn:
            if id(v_j) <= vid:
                continue
            d = float(np.linalg.norm(np.asarray(v_j.x_a, dtype=float)[:dim] - x_i))
            if d < h_min:
                h_min = d
            if d > h_max:
                h_max = d
    if h_min == np.inf:
        h_min = 0.0
    return float(h_min), float(h_max)


def as_jsonable(diag: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in a diagnostics dict to lists for JSON dumping."""
    out: dict[str, Any] = {}
    for k, v in diag.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def drift_fractions(
    initial: dict[str, Any],
    current: dict[str, Any],
    keys: tuple[str, ...] = ('mass_total', 'volume_total'),
) -> dict[str, float]:
    """Per-key absolute fractional drift: ``|current - initial| / |initial|``.

    Missing or zero-magnitude initial values map to ``0.0`` to avoid
    divide-by-zero in early-step checks.
    """
    out: dict[str, float] = {}
    for k in keys:
        if k not in initial or k not in current:
            continue
        a = float(initial[k])
        b = float(current[k])
        if a == 0.0:
            out[k] = abs(b)
        else:
            out[k] = abs(b - a) / abs(a)
    return out
