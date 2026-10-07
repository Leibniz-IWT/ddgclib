#!/usr/bin/env python3
"""Detector for the air sliver-cell F/m ejection of the 2D dam break (laneF
follow-up, 2026-10-05).

Runs the case through ``PRESETS['dam_break_2D']`` (or ``preset.replace``
arms given as ``--replace axis=value``) and records, for every integrated
vertex at every step, the dual volume, the per-phase masses, the stress
force ``F = (a - g) m`` and the acceleration the preset's ``dudt_fn``
returns, together with the 1-ring.  The first vertex whose speed exceeds
``--u-eject`` (or that leaves the tank) is the ejected one; the script
prints its history over the last ``--history`` steps (volume, mass, |F|,
|a|, phase, neighbourhood, whether the 1-ring changed in that step) and
the per-face pressure contributions at the step before the ejection, then
writes everything to ``<out>/ejection_<label>.json``.

Usage (repo root)::

    python cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2
    python cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace sliver_mass_floor=0.1
    python cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --out /tmp/x

Without an ejection the run completes the horizon and the summary (KE
series, front position, mass drift, the smallest dual volume and the
largest |a| met along the run) is written all the same, so the same
script measures the candidate arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from ddgclib.methods import PRESETS, record_methods  # noqa: E402

_OUT = os.path.join(_HERE, 'results', 'sliver_ejection')


class _Ejected(Exception):
    pass


def _parse_value(s: str):
    if s in ('None', 'none'):
        return None
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def build(args):
    from cases_dynamic.dam_break.src import _params as p
    from cases_dynamic.dam_break.src._setup import (
        cfl_timestep, setup_dam_break_multiphase,
    )
    dim = args.dim
    alpha = args.alpha if args.alpha is not None else p.alpha_art
    t_end = args.t_end if args.t_end is not None else p.t_end
    refine = args.refine if args.refine is not None else (
        p.n_refine_2d if dim == 2 else p.n_refine_3d)
    rep = {k: _parse_value(v) for k, v in
           (s.split('=', 1) for s in args.replace)}
    preset = f'dam_break_{dim}D'
    methods = PRESETS[preset].replace(**rep) if rep else PRESETS[preset]
    HC, bV, mps, bc_set, dudt_fn, _r, params = setup_dam_break_multiphase(
        dim=dim, a=p.a, L=p.L, H=p.H, W=p.W, col_w=p.col_w, col_h=p.col_h,
        col_d=p.col_d, rho_l=p.rho_l, rho_g=p.rho_g, mu_l=p.mu_l,
        mu_g=p.mu_g, gamma=p.gamma, K_l=p.K_l, K_g=p.K_g, g=p.g,
        gravity_axis=p.gravity_axis, P_atm=p.P_atm, n_refine=refine,
        alpha_art=alpha, methods=methods, clamp_gap=args.gap)
    dt = cfl_timestep(HC, dim, float(np.sqrt(p.K_l / p.rho_l)), cfl=p.cfl)
    n_steps = args.steps if args.steps is not None else int(t_end / dt) + 1
    label = f"{dim}D_alpha{alpha:g}_r{refine}_n{n_steps}"
    if rep:
        label += '_' + '_'.join(f"{k}-{v}" for k, v in sorted(rep.items()))
    if args.gap is not None:
        label += f"_gap{args.gap:g}"
    box = [(0.0, p.L), (0.0, p.H)] + ([(0.0, p.W)] if dim == 3 else [])
    return dict(HC=HC, bV=bV, mps=mps, bc_set=bc_set, dudt_fn=dudt_fn,
                dt=dt, n_steps=n_steps, methods=methods, label=label,
                box=box, alpha=alpha, refine=refine, params=p, dim=dim)


def _holes(HC, n_phases):
    """(vertex, phase) pairs with a dual sub-volume but no mass: the force
    reads ``p_phase[k] = 0`` (absolute) there."""
    out = []
    for v in HC.V:
        dvp = getattr(v, 'dual_vol_phase', None)
        mp = getattr(v, 'm_phase', None)
        if dvp is None or mp is None:
            continue
        for k in range(n_phases):
            if dvp[k] > 1e-30 and mp[k] <= 1e-30:
                out.append((id(v), k))
    return out


def _stranded(HC, n_phases):
    """(vertex, phase) pairs with mass but no dual sub-volume."""
    out = []
    for v in HC.V:
        dvp = getattr(v, 'dual_vol_phase', None)
        mp = getattr(v, 'm_phase', None)
        if dvp is None or mp is None:
            continue
        for k in range(n_phases):
            if dvp[k] <= 1e-30 and mp[k] > 1e-30:
                out.append((id(v), k, float(mp[k])))
    return out


def _vrec(v, acc, dim=2):
    """One vertex record (floats only)."""
    m_phase = getattr(v, 'm_phase', None)
    dvp = getattr(v, 'dual_vol_phase', None)
    pp = getattr(v, 'p_phase', None)
    a_mag, F_mag = acc if acc is not None else (None, None)
    return dict(
        x=[float(c) for c in v.x_a[:dim]], u=[float(c) for c in v.u[:dim]],
        m=float(v.m), dual_vol=float(getattr(v, 'dual_vol', 0.0)),
        m_phase=[float(c) for c in m_phase] if m_phase is not None else None,
        dual_vol_phase=[float(c) for c in dvp] if dvp is not None else None,
        p_phase=[float(c) for c in pp] if pp is not None else None,
        p=float(getattr(v, 'p', 0.0)),
        phase=int(v.phase), is_interface=bool(getattr(v, 'is_interface', False)),
        boundary=bool(getattr(v, 'boundary', False)),
        n_nn=len(v.nn), a=a_mag, F=F_mag,
    )


def _face_terms(v, HC, mps, dim=2):
    """Per-face pressure and viscous contributions of the multiphase force
    at *v* (the terms ``multiphase_stress_force`` sums), for the report."""
    from ddgclib.geometry._dual_split_2d import edge_phase_area_fractions
    from ddgclib.operators.multiphase_stress import (
        _phase_present_at, _phase_pressure,
    )
    from ddgclib.operators.stress import (
        dual_area_vector, pressure_flux, viscous_flux,
    )
    rows = []
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        fr = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)
        for k, frac in fr.items():
            pi_ok = _phase_present_at(v, k)
            pj_ok = _phase_present_at(v_j, k)
            if not pi_ok and not pj_ok:
                continue
            if pi_ok:
                p_i = float(v.p_phase[k])
                p_j = _phase_pressure(v_j, k, fallback=p_i)
            else:
                p_i = p_j = float(v_j.p_phase[k])
            A_k = frac * A_ij
            Fp = pressure_flux(p_i, p_j, A_k)
            Fv = viscous_flux(float(mps.get_mu(k)), v_j.u[:dim] - v.u[:dim],
                              v_j.x_a[:dim] - v.x_a[:dim], A_k)
            rows.append(dict(
                nb=[float(c) for c in v_j.x_a[:dim]], nb_phase=int(v_j.phase),
                nb_interface=bool(getattr(v_j, 'is_interface', False)),
                k=int(k), frac=float(frac), A=[float(c) for c in A_ij],
                p_i=p_i, p_j=p_j, F_p=[float(c) for c in Fp],
                F_v=[float(c) for c in Fv]))
    return rows


def run(args) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c = build(args)
    HC, bV, mps, dudt_fn = c['HC'], c['bV'], c['mps'], c['dudt_fn']
    dim, dt, n_steps = c['dim'], c['dt'], c['n_steps']
    methods = c['methods']
    p = c['params']
    n_phases = mps.n_phases
    g_vec = np.asarray(getattr(dudt_fn, 'body_force', np.zeros(dim)), float)
    box = c['box']
    tol = 1e-12 * max(hi - lo for lo, hi in box)
    u_ref = float(np.sqrt(2.0 * p.g * p.col_h))

    # --- acceleration recorder (the integrator calls it per vertex) -----
    cur: dict[int, tuple[float, float]] = {}
    cur_terms: dict[int, dict] = {}   # face terms at FORCE time, large |a|

    def rec(v):
        a = dudt_fn(v)
        F = (np.asarray(a, float) - g_vec) * float(v.m)
        a_mag = float(np.linalg.norm(a))
        cur[id(v)] = (a_mag, float(np.linalg.norm(F)))
        if a_mag > args.a_report:
            rows = _face_terms(v, HC, mps, dim)
            cur_terms[id(v)] = dict(
                x=[float(c) for c in v.x_a[:dim]], a=a_mag,
                F=[float(c) for c in F], m=float(v.m),
                m_phase=[float(c) for c in v.m_phase],
                dual_vol=float(getattr(v, 'dual_vol', 0.0)),
                dual_vol_phase=[float(c) for c in v.dual_vol_phase],
                p_phase=[float(c) for c in v.p_phase],
                is_interface=bool(getattr(v, 'is_interface', False)),
                interface_phases=sorted(int(k) for k in getattr(
                    v, 'interface_phases', ())),
                boundary=bool(getattr(v, 'boundary', False)),
                closure=[float(c) for c in np.sum(
                    [np.asarray(r['A']) * r['frac'] for r in rows], axis=0)],
                F_p_sum=[float(c) for c in np.sum(
                    [r['F_p'] for r in rows], axis=0)],
                F_v_sum=[float(c) for c in np.sum(
                    [r['F_v'] for r in rows], axis=0)],
                rows=rows)
        return a

    hist: deque = deque(maxlen=max(2, args.history))
    series: list[dict] = []
    flips: list[int] = []
    prev_edges: set | None = None
    worst = {'a_max': 0.0, 'a_max_step': None, 'a_max_vertex': None,
             'vol_min': np.inf, 'vol_min_step': None, 'vol_min_vertex': None,
             'u_max': 0.0}
    face_terms_prev: dict = {}
    eject: dict = {}
    clamp: dict = {'n_events': 0, 'n_steps': 0, 'first': None}
    relabel: dict = {'n': 0, 'first': None, 'events': [],
                     'n_released': 0, 'm_released': [0.0] * n_phases,
                     'released': []}
    holes_log: list[dict] = []     # steps at which a hole exists
    holes_first: dict = {}
    hole_steps = 0
    every = max(1, n_steps // 100)
    M0 = sum(v.m for v in HC.V)
    M0_phase = np.sum([v.m_phase for v in HC.V], axis=0)

    def outside(v):
        return any(v.x_a[i] < lo - tol or v.x_a[i] > hi + tol
                   for i, (lo, hi) in enumerate(box))

    def liquid(v):
        return v.phase == 1 or getattr(v, 'is_interface', False)

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        nonlocal prev_edges, hole_steps
        # laneV: how often the wall clamp (axis wall_clamp) put a vertex back
        n_cl = sum(int(n) for k, n in (diagnostics or {}).items()
                   if 'WallClampBC' in k)
        if n_cl:
            clamp['n_events'] += n_cl
            clamp['n_steps'] += 1
            if clamp['first'] is None:
                clamp['first'] = step
        edges = {frozenset((id(v), id(w))) for v in HC_cb.V for w in v.nn}
        flipped = prev_edges is not None and edges != prev_edges
        if flipped:
            flips.append(step)
        prev_edges = edges
        holes = _holes(HC_cb, n_phases)
        stranded = _stranded(HC_cb, n_phases)
        if holes or stranded:
            hole_steps += 1
            if not holes_first:
                holes_first.update(step=step, t=float(t),
                                   holes=len(holes), stranded=len(stranded))
            if len(holes_log) < 400:
                byid = {id(v): v for v in HC_cb.V}
                holes_log.append(dict(
                    step=step, flipped=flipped,
                    holes=[dict(k=k, x=[float(c) for c in byid[i].x_a[:dim]],
                                interface=bool(getattr(byid[i], 'is_interface',
                                                       False)),
                                dvp=float(byid[i].dual_vol_phase[k]),
                                p_phase=float(byid[i].p_phase[k]),
                                nn_bulk_k=sum(1 for w in byid[i].nn
                                              if w.phase == k))
                           for i, k in holes],
                    stranded=[dict(k=k, m=m,
                                   x=[float(c) for c in byid[i].x_a[:dim]])
                              for i, k, m in stranded]))
        snap = {}
        prev_snap = hist[-1]['snap'] if hist else {}
        for v in HC_cb.V:
            r = _vrec(v, cur.get(id(v)), dim)
            r['nn'] = sorted(id(w) for w in v.nn)
            r['frozen'] = v in bV_cb
            snap[id(v)] = r
            # laneV census: a bulk vertex relabelled straight into the
            # other bulk phase by the simplex vote (its whole mass of the
            # old phase is stranded or released at once)
            q = prev_snap.get(id(v))
            if q is not None and q['m_phase'] and r['m_phase']:
                # ... and the ledger side of it: a phase whose whole mass
                # left the vertex in one step (released into the pool by
                # the volume ledger, or stranded under the snapshot rule)
                for k in range(n_phases):
                    if q['m_phase'][k] > 1e-30 and r['m_phase'][k] <= 1e-30:
                        relabel['n_released'] += 1
                        relabel['m_released'][k] += q['m_phase'][k]
                        if len(relabel['released']) < 200:
                            relabel['released'].append(dict(
                                step=step, k=k, x=r['x'], m=q['m_phase'][k],
                                phase_before=q['phase'], phase_after=r['phase'],
                                interface_before=q['is_interface']))
            if (q is not None and q['phase'] >= 0 and r['phase'] >= 0
                    and q['phase'] != r['phase']):
                relabel['n'] += 1
                if relabel['first'] is None:
                    relabel['first'] = step
                if len(relabel['events']) < 200:
                    relabel['events'].append(dict(
                        step=step, x=r['x'], old=q['phase'], new=r['phase'],
                        m_phase_before=q['m_phase'],
                        m_phase_after=r['m_phase']))
            a = r['a'] or 0.0
            if r['a'] is not None and a > worst['a_max']:
                worst.update(a_max=a, a_max_step=step,
                             a_max_vertex=dict(r, nn=None))
            if (not r['frozen'] and r['dual_vol'] > 0.0
                    and r['dual_vol'] < worst['vol_min']):
                worst.update(vol_min=r['dual_vol'], vol_min_step=step,
                             vol_min_vertex=dict(r, nn=None))
        hist.append(dict(step=step, t=float(t), flipped=flipped, snap=snap))
        u_max_v = max(HC_cb.V, key=lambda v: float(np.linalg.norm(v.u[:dim])))
        u_max = float(np.linalg.norm(u_max_v.u[:dim]))
        worst['u_max'] = max(worst['u_max'], u_max)
        # per-face terms recorded at FORCE time (before the move) for the
        # vertices whose |a| exceeded --a-report in this step
        if cur_terms:
            face_terms_prev.clear()
            face_terms_prev['step'] = step
            face_terms_prev['vertices'] = dict(cur_terms)
            cur_terms.clear()
        if step % every == 0 or step == n_steps - 1:
            sel = [v for v in HC_cb.V if liquid(v)]
            x_front = max(v.x_a[0] for v in sel)
            series.append(dict(
                step=step, t=float(t), n_vertices=len(HC_cb.V),
                n_frozen=len(bV_cb), n_interface=sum(
                    1 for v in HC_cb.V if getattr(v, 'is_interface', False)),
                KE_liq=float(sum(0.5 * v.m * np.dot(v.u[:dim], v.u[:dim])
                                 for v in HC_cb.V if v.phase == 1)),
                KE_liq_iface=float(sum(0.5 * v.m * np.dot(v.u[:dim], v.u[:dim])
                                       for v in sel)),
                x_front=float(x_front), u_max=u_max,
                n_flips=len(flips),
                mass_rel=float(sum(v.m for v in HC_cb.V) / M0 - 1.0),
                mass_phase_rel=[float(c) for c in
                                np.sum([v.m_phase for v in HC_cb.V], axis=0)
                                / M0_phase - 1.0],
                vol_min=float(min(getattr(v, 'dual_vol', 0.0)
                                  for v in HC_cb.V if v not in bV_cb)),
                a_max=float(max((r[0] for r in cur.values()), default=0.0)),
                n_holes=len(holes), n_stranded=len(stranded),
                # per-phase: measured volume and sub-volume weighted
                # pressure level (the EOS reads mass / volume)
                vol_phase=[float(sum(v.dual_vol_phase[k] for v in HC_cb.V))
                           for k in range(n_phases)],
                p_level=[float(sum(v.p_phase[k] * v.dual_vol_phase[k]
                                   for v in HC_cb.V if v.m_phase[k] > 1e-30)
                               / max(sum(v.dual_vol_phase[k] for v in HC_cb.V
                                         if v.m_phase[k] > 1e-30), 1e-300))
                         for k in range(n_phases)],
            ))
        n_out = sum(1 for v in HC_cb.V if outside(v))
        if u_max > args.u_eject * u_ref or n_out:
            eject.update(step=step, t=float(t), vertex=id(u_max_v),
                         u_max=u_max, n_outside=n_out)
            raise _Ejected()

    t0 = time.time()
    abort = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            methods.integrate(HC, bV, rec, dt=dt, n_steps=n_steps,
                              bc_set=c['bc_set'], callback=callback, mps=mps)
    except _Ejected:
        pass
    except Exception as e:  # noqa: BLE001 - an abort is a result here
        abort = f"{type(e).__name__}: {e}"[:300]
    wall = time.time() - t0

    # --- report ----------------------------------------------------------
    out = dict(
        label=c['label'], alpha=c['alpha'], refine=c['refine'], dt=dt,
        n_steps=n_steps, steps_done=(hist[-1]['step'] + 1) if hist else 0,
        u_ref=u_ref, abort=abort, wall_s=round(wall, 1),
        flips=flips, n_flip_steps=len(flips),
        holes_first=holes_first or None, n_hole_steps=hole_steps,
        holes_log=holes_log,
        relabel=relabel, clamp=clamp,
        worst={k: (None if (isinstance(v, float) and not np.isfinite(v))
                   else v) for k, v in worst.items()},
        series=series, eject=eject or None, trace=None,
        methods=methods.to_dict(),
        final=dict(
            KE_liq=series[-1]['KE_liq'] if series else None,
            KE_liq_peak=max((s['KE_liq'] for s in series), default=None),
            t_KE_peak=(max(series, key=lambda s: s['KE_liq'])['t']
                       if series else None),
            x_front_end=series[-1]['x_front'] if series else None,
            mass_rel=series[-1]['mass_rel'] if series else None,
        ),
    )
    if eject:
        vid = eject['vertex']
        trace = []
        for h in hist:
            r = h['snap'].get(vid)
            if r is None:
                continue
            row = dict(step=h['step'], t=h['t'], flipped_global=h['flipped'],
                       **{k: r[k] for k in ('x', 'u', 'm', 'dual_vol',
                                            'm_phase', 'dual_vol_phase',
                                            'p_phase', 'p', 'phase',
                                            'is_interface', 'boundary',
                                            'frozen', 'n_nn', 'a', 'F')})
            row['nn_phases'] = [h['snap'][w]['phase'] for w in r['nn']
                                if w in h['snap']]
            row['nn_interface'] = sum(h['snap'][w]['is_interface']
                                      for w in r['nn'] if w in h['snap'])
            row['nn_frozen'] = sum(h['snap'][w]['frozen']
                                   for w in r['nn'] if w in h['snap'])
            trace.append(row)
        for i in range(1, len(trace)):
            a, b = hist[i - 1]['snap'].get(vid), hist[i]['snap'].get(vid)
            trace[i]['ring_changed'] = (a is not None and b is not None
                                        and a['nn'] != b['nn'])
        if trace:
            trace[0]['ring_changed'] = None
        out['trace'] = trace
        out['face_terms'] = dict(step=face_terms_prev.get('step'),
                                 vertices=face_terms_prev.get('vertices', {}),
                                 ejected=vid)
        state = sorted((tuple(r['x']), tuple(r['u']), r['m'])
                       for r in hist[-1]['snap'].values())
        out['state_sha256'] = hashlib.sha256(repr(state).encode()).hexdigest()
    else:
        state = sorted((tuple(float(c) for c in v.x_a[:dim]),
                        tuple(float(c) for c in v.u[:dim]), float(v.m))
                       for v in HC.V)
        out['state_sha256'] = hashlib.sha256(repr(state).encode()).hexdigest()

    out_dir = args.out or _OUT
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"ejection_{c['label']}")
    with open(stem + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    record_methods(stem + '_methods.json', methods, HC,
                   extra={'dt': dt, 'n_steps': n_steps, 'alpha_art': c['alpha'],
                          'n_refine': c['refine'], 'label': c['label']})
    return out


def report(out: dict) -> None:
    f = out['final']
    print(f"[{out['label']}] steps {out['steps_done']}/{out['n_steps']}, "
          f"dt {out['dt']:.4e}, {out['wall_s']} s, flip steps "
          f"{out['n_flip_steps']}, abort {out['abort']}")
    print(f"  KE_liq peak {f['KE_liq_peak']} at t {f['t_KE_peak']}, "
          f"KE_liq end {f['KE_liq']}, x_front end {f['x_front_end']}, "
          f"mass drift {f['mass_rel']}, |u|max {out['worst']['u_max']:.4e}")
    w = out['worst']
    print(f"  smallest integrated dual volume {w['vol_min']} at step "
          f"{w['vol_min_step']}; largest |a| {w['a_max']:.4e} at step "
          f"{w['a_max_step']}")
    if w['a_max_vertex']:
        r = w['a_max_vertex']
        print(f"    at x {r['x']}, phase {r['phase']}, interface "
              f"{r['is_interface']}, boundary {r['boundary']}, m {r['m']:.3e}, "
              f"vol {r['dual_vol']:.3e}, |F| {r['F']:.3e}")
    print(f"  state {out['state_sha256'][:16]}")
    print(f"  (vertex, phase) holes (sub-volume without mass) or stranded "
          f"masses (mass without sub-volume) on {out['n_hole_steps']} steps; "
          f"first {out['holes_first']}")
    print(f"  bulk vertices relabelled straight into the other bulk phase "
          f"by the vote: {out['relabel']['n']} (first at step "
          f"{out['relabel']['first']})")
    for r in out['relabel']['events'][:6]:
        print(f"    step {r['step']} x {np.round(r['x'], 4).tolist()} "
              f"{r['old']} -> {r['new']} m_phase {r['m_phase_before']} -> "
              f"{r['m_phase_after']}")
    print(f"  wall clamp: {out['clamp']['n_events']} put-backs on "
          f"{out['clamp']['n_steps']} steps (first at step {out['clamp']['first']})")
    rl = out['relabel']
    print(f"  (vertex, phase) masses that left a vertex whole in one step: "
          f"{rl['n_released']} events, mass per phase {rl['m_released']}")
    for r in rl['released'][:8]:
        print(f"    step {r['step']} phase {r['k']} x "
              f"{np.round(r['x'], 4).tolist()} m {r['m']:.3e} label "
              f"{r['phase_before']}{'(if)' if r['interface_before'] else ''}"
              f" -> {r['phase_after']}")
    for h in out['holes_log'][:8]:
        print(f"    step {h['step']} flip {h['flipped']}: holes "
              f"{[(r['k'], np.round(r['x'], 4).tolist(), 'if' if r['interface'] else 'bulk', 'dvp %.2e' % r['dvp'], 'p %.1f' % r['p_phase'], 'nn_k %d' % r['nn_bulk_k']) for r in h['holes']]} "
              f"stranded {[(r['k'], np.round(r['x'], 4).tolist(), '%.2e' % r['m']) for r in h['stranded']]}")
    if out['eject']:
        e = out['eject']
        print(f"  EJECTION at step {e['step']} t {e['t']:.5f}: |u| "
              f"{e['u_max']:.4e} ({e['u_max'] / out['u_ref']:.1f} u_ref), "
              f"{e['n_outside']} outside")
        print("  trace of the ejected vertex (step, t, vol, m, m_phase, |F|, "
              "|a|, |u|, phase, iface, bnd, n_nn, nn phases, ring changed, "
              "global flip):")
        for r in out['trace']:
            print(f"    {r['step']:5d} {r['t']:.5f} vol {r['dual_vol']:.3e} "
                  f"m {r['m']:.3e} mph {r['m_phase']} |F| "
                  f"{(r['F'] if r['F'] is not None else float('nan')):.3e} "
                  f"|a| {(r['a'] if r['a'] is not None else float('nan')):.3e} "
                  f"|u| {np.linalg.norm(r['u']):.3e} ph {r['phase']} "
                  f"if {int(r['is_interface'])} bd {int(r['boundary'])} "
                  f"nn {r['n_nn']} {r['nn_phases']} ring "
                  f"{r['ring_changed']} flip {r['flipped_global']} "
                  f"x {np.round(r['x'], 5).tolist()}")
        ft = out.get('face_terms') or {}
        for vid, rec_v in (ft.get('vertices') or {}).items():
            tag = 'EJECTED' if vid == ft.get('ejected') else 'also'
            print(f"  [{tag}] force-time state at step {ft['step']}: x "
                  f"{np.round(rec_v['x'], 5).tolist()} |a| {rec_v['a']:.3e} "
                  f"F {np.round(rec_v['F'], 4).tolist()} m {rec_v['m']:.3e} "
                  f"m_phase {rec_v['m_phase']} vol {rec_v['dual_vol']:.3e} "
                  f"dvp {rec_v['dual_vol_phase']} p_phase {rec_v['p_phase']} "
                  f"iface {rec_v['is_interface']} {rec_v['interface_phases']} "
                  f"bnd {rec_v['boundary']}\n      closure sum frac*A "
                  f"{np.round(rec_v['closure'], 8).tolist()} F_p sum "
                  f"{np.round(rec_v['F_p_sum'], 4).tolist()} F_v sum "
                  f"{np.round(rec_v['F_v_sum'], 4).tolist()}")
            for r in rec_v['rows']:
                print(f"    nb {np.round(r['nb'], 5).tolist()} ph "
                      f"{r['nb_phase']} if {int(r['nb_interface'])} k {r['k']} "
                      f"frac {r['frac']:.2f} |A| "
                      f"{np.linalg.norm(r['A']):.3e} p_i {r['p_i']:.3f} "
                      f"p_j {r['p_j']:.3f} F_p {np.round(r['F_p'], 4).tolist()} "
                      f"F_v {np.round(r['F_v'], 4).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dim', type=int, default=2, choices=(2, 3),
                    help='2: PRESETS["dam_break_2D"], 3: ["dam_break_3D"]')
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--refine', type=int, default=None)
    ap.add_argument('--t-end', type=float, default=None, dest='t_end')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--replace', action='append', default=[],
                    metavar='AXIS=VALUE',
                    help='preset.replace(...) arm, repeatable')
    ap.add_argument('--u-eject', type=float, default=5.0, dest='u_eject',
                    help='ejection threshold in units of u_ref = sqrt(2 g a)')
    ap.add_argument('--history', type=int, default=12,
                    help='steps of per-vertex history kept for the trace')
    ap.add_argument('--a-report', type=float, default=1e3, dest='a_report',
                    help='record the per-face terms at force time for every '
                         'vertex whose |a| exceeds this (m/s^2)')
    ap.add_argument('--gap', type=float, default=None,
                    help='wall_clamp put-down gap of the setup (laneV)')
    ap.add_argument('--out', default=None,
                    help=f'output directory (default {_OUT})')
    args = ap.parse_args()
    report(run(args))


if __name__ == '__main__':
    main()
