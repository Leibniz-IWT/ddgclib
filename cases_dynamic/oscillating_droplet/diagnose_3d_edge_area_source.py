#!/usr/bin/env python3
"""laneJ: 3D edge-area source A/B (e_star cache vs DEC p_ij), audit F4.

MEASUREMENT ONLY.  No library code is changed: the p_ij path is forced
by wrapping a preset's retopology function so that it clears
``HC._edge_area_cache`` after every call (``stress_force`` /
``multiphase_stress_force`` then fall back to ``dual_area_vector`` ->
``_dual_area_vector_3d_p_ij``).  Every configuration is built from
``ddgclib.methods.PRESETS``; the p_ij arm is the same preset with
``connectivity='custom'`` and ``custom=no_cache(base)``.

Sub-commands (run from the repo root with the ddg env python)::

    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py static
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py a5b
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dynamic --arm pij
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dynamic --arm cache

Outputs (``results_3d/``): ``laneJ_static.json``, ``laneJ_a5b.json``,
``score_pij.json`` / ``methods_pij.json`` / ``diags_pij.json`` /
``snapshots_pij/`` (and ``*_cache_laneJ`` for the reference arm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (  # noqa: E402
    R0, epsilon as EPS_CASE, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, t_end_3d,
)
from cases_dynamic.oscillating_droplet.src._setup import (  # noqa: E402
    setup_oscillating_droplet,
)
from ddgclib.methods import PRESETS, effective_methods, record_methods  # noqa: E402
from ddgclib.operators.stress import dual_area_vector, stress_force  # noqa: E402

_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_CASE_DIR, 'results_3d')


# ---------------------------------------------------------------------------
# p_ij forcing wrapper
# ---------------------------------------------------------------------------
def no_cache(base):
    """Wrap a preset retopology callable; clear the e_star cache after it.

    Declares by name exactly the kwargs ``_do_retopologize`` forwards to
    the base partial (``skip_triangulation`` is deliberately NOT declared:
    the dual_only partial binds it and a call-time value would override
    the binding).
    """
    def _no_cache(HC, bV, dim, boundary_filter=None, merge_cdist=None,
                  backend=None, remesh_mode='delaunay', remesh_kwargs=None):
        base(HC, bV, dim, boundary_filter=boundary_filter,
             merge_cdist=merge_cdist, backend=backend,
             remesh_mode=remesh_mode, remesh_kwargs=remesh_kwargs)
        HC._edge_area_cache = None
    _no_cache.__name__ = 'no_cache'
    return _no_cache


def _simplex_pij_A(v_i, v_j, tets):
    """DEC p_ij area vector built from the explicit tets around edge ij.

    Same polygon as ``stress._dual_area_vector_3d_p_ij`` (tet barycentres
    interleaved with face barycentres (x_i + x_j + x_k)/3), but the ring
    order and the face vertex k are read from ``HC._simplices`` (the link
    cycle of the edge) instead of the dual-vertex ring walk + the
    nearest-midpoint face-barycentre heuristic.  Returns None for an
    open link (boundary edge) -> caller falls back to the library path.
    """
    if not tets:
        return None
    others = [tuple(w for w in T if w is not v_i and w is not v_j)
              for T in tets]
    if any(len(o) != 2 for o in others):
        return None
    cnt: dict[int, int] = {}
    for o in others:
        for w in o:
            cnt[id(w)] = cnt.get(id(w), 0) + 1
    if any(c != 2 for c in cnt.values()):
        return None  # open or non-manifold link
    n = len(tets)
    used = [False] * n
    order, ks = [0], [others[0][1]]
    used[0] = True
    cur = others[0][1]
    while len(order) < n:
        nxt = next((t for t in range(n)
                    if not used[t] and any(w is cur for w in others[t])), None)
        if nxt is None:
            return None
        used[nxt] = True
        order.append(nxt)
        cur = [w for w in others[nxt] if w is not cur][0]
        ks.append(cur)
    x_i, x_j = v_i.x_a[:3], v_j.x_a[:3]
    pts = []
    for a in range(n):
        pts.append(np.mean([w.x_a[:3] for w in tets[order[a]]], axis=0))
        pts.append((x_i + x_j + ks[a].x_a[:3]) / 3.0)
    P = np.array(pts)
    c = P.mean(axis=0)
    A = np.zeros(3)
    for k in range(len(P)):
        A += 0.5 * np.cross(P[k] - c, P[(k + 1) % len(P)] - c)
    if np.dot(A, x_j - x_i) < 0:
        A = -A
    return A


def simplex_pij_cache(HC) -> dict:
    """{id(v): {id(nb): A_ij}} for every edge with a closed tet link."""
    et: dict[frozenset, list] = {}
    for T in HC._simplices:
        T = tuple(T)
        for a in range(4):
            for b in range(a + 1, 4):
                et.setdefault(frozenset((id(T[a]), id(T[b]))), []).append(T)
    cache: dict = {}
    for v in HC.V:
        row = {}
        for nb in v.nn:
            A = _simplex_pij_A(v, nb, et.get(frozenset((id(v), id(nb))), []))
            if A is not None:
                row[id(nb)] = A
        if row:
            cache[id(v)] = row
    return cache


def simplex_cache(base):
    """Like :func:`no_cache` but fills the cache with simplex-driven p_ij."""
    def _simplex_cache(HC, bV, dim, boundary_filter=None, merge_cdist=None,
                       backend=None, remesh_mode='delaunay',
                       remesh_kwargs=None):
        base(HC, bV, dim, boundary_filter=boundary_filter,
             merge_cdist=merge_cdist, backend=backend,
             remesh_mode=remesh_mode, remesh_kwargs=remesh_kwargs)
        HC._edge_area_cache = simplex_pij_cache(HC)
        HC._laneJ_edge_area_source = 'p_ij_simplex'
    _simplex_cache.__name__ = 'simplex_cache'
    return _simplex_cache


def arm_methods(preset: str, arm: str, mps):
    """(methods, custom) for an arm in {'cache', 'pij', 'pij_simplex'}."""
    methods = PRESETS[preset]
    if arm == 'cache':
        return methods, None
    base = methods.retopologize_fn(mps=mps)
    if arm == 'pij':
        m = methods.replace(
            connectivity='custom',
            label=methods.label + ' + laneJ no_cache wrapper (p_ij forced)',
            notes=f'laneJ: preset {preset!r} retopology, then '
                  'HC._edge_area_cache = None',
        )
        return m, no_cache(base)
    m = methods.replace(
        connectivity='custom',
        label=methods.label + ' + laneJ simplex_cache wrapper',
        notes=f'laneJ: preset {preset!r} retopology, then '
              'HC._edge_area_cache = simplex-driven p_ij (scratch construction; '
              'effective_methods will misreport batch_e_star_cache)',
    )
    return m, simplex_cache(base)


def _eff(HC, dim, methods):
    e = effective_methods(HC, dim, methods)
    if getattr(HC, '_laneJ_edge_area_source', None):
        e['edge_area_source_laneJ'] = HC._laneJ_edge_area_source
    return e


# ---------------------------------------------------------------------------
# Static geometry probes
# ---------------------------------------------------------------------------
def _stats(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {'n': 0}
    return {'n': int(a.size), 'median': float(np.median(a)),
            'max': float(np.max(a)), 'mean': float(np.mean(a))}


def _A_cache(HC, v, nb):
    c = HC._edge_area_cache
    if c is not None and id(v) in c and id(nb) in c[id(v)]:
        return np.asarray(c[id(v)][id(nb)], dtype=float), True
    return dual_area_vector(v, nb, HC, 3), False  # stress.py fallback


def probe_mesh(HC, bV, label: str, groups: dict | None = None) -> dict:
    """Per-edge A_ij diff, linear precision and closure on both paths.

    *groups*: optional {name: predicate(v)} for extra vertex classes
    (interface, bulk droplet, ...).  Standard classes: 'interior'
    (not in bV, no bV neighbour), 'boundary_adjacent' (not in bV, >= 1
    bV neighbour), 'boundary' (in bV: open truncated cell, closure not
    expected; both paths use p_ij there because batch_e_star only
    caches interior vertices).
    """
    assert HC._edge_area_cache is not None, 'probe needs the cache path'
    g = np.array([1.0, 2.0, 3.0])
    p0 = 5.0
    saved = {}
    for v in HC.V:
        saved[id(v)] = (np.array(getattr(v, 'u', np.zeros(3)), copy=True),
                        getattr(v, 'p', 0.0))
        v.u = np.zeros(3)
        v.p = p0 + float(g @ v.x_a[:3])

    rel_diff, abs_diff = [], []
    n_cache_hit = n_cache_miss = 0
    per_v: dict[str, dict] = {'cache': {}, 'pij': {}}
    for v in HC.V:
        nbs = list(v.nn)
        Ac, Ap = [], []
        for nb in nbs:
            a_c, hit = _A_cache(HC, v, nb)
            a_p = dual_area_vector(v, nb, HC, 3)
            Ac.append(a_c)
            Ap.append(a_p)
            if v not in bV:
                n_cache_hit += hit
                n_cache_miss += (not hit)
                if np.linalg.norm(a_p) > 0:
                    rel_diff.append(np.linalg.norm(a_c - a_p)
                                    / np.linalg.norm(a_p))
                abs_diff.append(np.linalg.norm(a_c - a_p))
        for path, As in (('cache', Ac), ('pij', Ap)):
            As = np.array(As) if As else np.zeros((0, 3))
            s = As.sum(axis=0) if len(As) else np.zeros(3)
            ssum = float(np.sum(np.linalg.norm(As, axis=1))) if len(As) else 0.0
            M = np.zeros((3, 3))
            for nb, a in zip(nbs, As):
                M += 0.5 * np.outer(nb.x_a[:3] - v.x_a[:3], a)
            vol = float(getattr(v, 'dual_vol', 0.0) or 0.0)
            per_v[path][id(v)] = {
                'closure_rel': float(np.linalg.norm(s) / ssum) if ssum else 0.0,
                'offdiag_abs': float(np.max(np.abs(M - np.diag(np.diag(M))))),
                'tensor_rel': (float(np.max(np.abs(M - vol * np.eye(3))) / vol)
                               if vol > 0 else None),
                'vol': vol,
            }

    # Force residual with a linear point-valued pressure (mu=0, u=0):
    # exact integrated force is -g * Vol_i.
    cache_saved = HC._edge_area_cache
    for path in ('cache', 'pij'):
        HC._edge_area_cache = cache_saved if path == 'cache' else None
        for v in HC.V:
            rec = per_v[path][id(v)]
            if v in bV or rec['vol'] <= 0:
                rec['force_rel'] = None
                continue
            F = stress_force(v, dim=3, mu=0.0, HC=HC)
            rec['force_rel'] = float(np.linalg.norm(F + g * rec['vol'])
                                     / (np.linalg.norm(g) * rec['vol']))
    HC._edge_area_cache = cache_saved

    for v in HC.V:
        v.u, v.p = saved[id(v)]

    classes = {
        'interior': lambda v: v not in bV and not any(nb in bV for nb in v.nn),
        'boundary_adjacent': lambda v: v not in bV and any(nb in bV for nb in v.nn),
        'boundary': lambda v: v in bV,
    }
    if groups:
        classes.update(groups)
    out = {'label': label,
           'n_vertices': sum(1 for _ in HC.V), 'n_bV': len(bV),
           'edge_rel_diff': _stats(rel_diff),
           'edge_abs_diff': _stats(abs_diff),
           'n_directed_interior_edges_cache_hit': int(n_cache_hit),
           'n_directed_interior_edges_cache_miss': int(n_cache_miss),
           'classes': {}}
    for cname, pred in classes.items():
        vs = [v for v in HC.V if pred(v)]
        entry = {'n': len(vs)}
        for path in ('cache', 'pij'):
            recs = [per_v[path][id(v)] for v in vs]
            entry[path] = {
                'closure_rel': _stats([r['closure_rel'] for r in recs]),
                'offdiag_abs': _stats([r['offdiag_abs'] for r in recs]),
                'tensor_rel': _stats([r['tensor_rel'] for r in recs
                                      if r['tensor_rel'] is not None]),
                'force_rel': _stats([r['force_rel'] for r in recs
                                     if r.get('force_rel') is not None]),
            }
        out['classes'][cname] = entry
    return out


def run_static() -> dict:
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.geometry.domains import ball, box

    res = {}
    for name, builder in (('box_r2', box), ('ball_r2', ball)):
        r = builder(refinement=2)
        HC, bV = r.HC, set(r.bV)
        _retopologize(HC, bV, 3)
        res[name] = probe_mesh(HC, bV, name)
        res[name]['effective_methods'] = effective_methods(HC, 3)
        print(json.dumps({name: res[name]['edge_rel_diff']}))

    # Droplet meshes (task 2 / task 3 fixtures) after ONE preset retopology
    for tag, eps, preset in (
        ('droplet_eps0_static_floor_preset', 0.0, 'static_droplet_floor_3D'),
        ('droplet_eps0.05_dual_only_preset', EPS_CASE, 'oscillating_droplet_3D'),
    ):
        methods = PRESETS[preset]
        HC, bV, mps, *_ = setup_oscillating_droplet(
            dim=3, R0=R0, epsilon=eps, l=l, rho_d=rho_d, rho_o=rho_o,
            mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
            L_domain=L_domain, refinement_outer=2, refinement_droplet=2,
            split_method=methods.split_method,
            redistribute_mass=methods.redistribute_mass,
        )
        cache_at_setup = getattr(HC, '_edge_area_cache', None) is not None
        retopo = methods.retopologize_fn(mps=mps)
        retopo(HC, bV, 3)
        groups = {
            'interface': lambda v: bool(getattr(v, 'is_interface', False)),
            'bulk_droplet': lambda v: (not getattr(v, 'is_interface', False)
                                       and v not in bV
                                       and int(getattr(v, 'phase', 0)) == 1),
            'bulk_outer_interior': lambda v, bV=bV: (
                not getattr(v, 'is_interface', False) and v not in bV
                and int(getattr(v, 'phase', 0)) == 0
                and not any(nb in bV for nb in v.nn)),
        }
        res[tag] = probe_mesh(HC, bV, tag, groups)
        res[tag]['edge_area_cache_present_at_setup'] = cache_at_setup
        res[tag]['effective_methods'] = effective_methods(HC, 3, methods)
        print(json.dumps({tag: res[tag]['edge_rel_diff']}))
    return res


# ---------------------------------------------------------------------------
# A.5.b static droplet floor (mirror of diagnose_a5_bisection.run_a5b)
# ---------------------------------------------------------------------------
def run_a5b_arm(arm: str, n_steps: int = 20) -> dict:
    from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import (
        _compute_dt, _max_interface_force,
    )
    from ddgclib.data import compute_conservation

    dim = 3
    preset = 'static_droplet_floor_3D'
    methods0 = PRESETS[preset]
    HC, bV, mps, bc_set, dudt_fn, _setup_retopo, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=0.0, l=l, rho_d=rho_d, rho_o=rho_o,
            mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
            L_domain=L_domain, refinement_outer=2, refinement_droplet=2,
            split_method=methods0.split_method,
            redistribute_mass=methods0.redistribute_mass,
        )
    methods, custom = arm_methods(preset, arm, mps)
    c_s = float(np.sqrt(K_d / rho_d))
    dt, dx_min = _compute_dt(HC, dim, c_s)

    maxF = [_max_interface_force(HC, dim, mps)[0]]
    mass = [compute_conservation(HC, dim=dim)['mass_total']]
    eff: dict = {}
    cache_frames = []
    cb_time = [0.0]

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        t0 = time.perf_counter()
        if step == 0:
            eff['after_step1'] = _eff(HC_cb, dim, methods)
        cache_frames.append(getattr(HC_cb, '_edge_area_cache', None) is not None)
        maxF.append(_max_interface_force(HC_cb, dim, mps)[0])
        mass.append(compute_conservation(HC_cb, dim=dim)['mass_total'])
        for v in HC_cb.V:
            v.u[:] = 0.0
        cb_time[0] += time.perf_counter() - t0
        print(f"  [{arm}] step {step + 1:3d} max|F| = {maxF[-1]:.9e}",
              flush=True)

    t0 = time.perf_counter()
    methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                      bc_set=bc_set, callback=cb, mps=mps, custom=custom)
    wall = time.perf_counter() - t0
    eff['end'] = _eff(HC, dim, methods)
    return {
        'arm': arm, 'preset': preset, 'config': methods.to_dict(),
        'n_steps': n_steps, 'dt': dt,
        'max_abs_F_step0': maxF[0],
        'max_abs_F_peak': float(np.max(maxF)),
        'max_abs_F_end': float(maxF[-1]),
        'max_abs_F_history': [float(x) for x in maxF],
        'mass_rel_drift': abs(mass[-1] - mass[0]) / abs(mass[0]),
        'cache_present_frames': int(sum(cache_frames)),
        'n_frames': len(cache_frames),
        'effective_methods': eff,
        'wall_s': wall, 'callback_s': cb_time[0],
        'wall_per_step_excl_callback_s': (wall - cb_time[0]) / n_steps,
    }


def run_a5b(n_steps: int = 20) -> dict:
    from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import (
        run_a5b as harness,
    )
    out = {}
    t0 = time.perf_counter()
    h = harness(dim=3, refinement_outer=2, refinement_droplet=2,
                n_steps=n_steps, split_method='neighbour_count',
                redistribute_mass=True, curvature_path='integrated')
    out['harness_cache'] = {k: h[k] for k in (
        'max_abs_F_peak', 'max_abs_F_end', 'max_abs_F_history',
        'mass_rel_drift', 'dt')}
    out['harness_cache']['wall_s'] = time.perf_counter() - t0
    out['pin'] = 7.274172e-05
    for arm in ('cache', 'pij'):
        out[arm] = run_a5b_arm(arm, n_steps)
    return out


# ---------------------------------------------------------------------------
# Full 3D dynamic droplet (mirror of oscillating_droplet_3D.py main)
# ---------------------------------------------------------------------------
def run_dynamic(arm: str) -> dict:
    from cases_dynamic.oscillating_droplet.src._analytical import (
        damped_frequency, lamb_damping_rate, max_radius_envelope,
        rayleigh_frequency,
    )
    from cases_dynamic.oscillating_droplet.src._metrics import (
        oscillation_score_3d, save_score,
    )
    from cases_dynamic.oscillating_droplet.src._plot_helpers import (
        compute_diagnostics,
    )
    from ddgclib.data import StateHistory

    dim = 3
    preset = 'oscillating_droplet_3D'
    suffix = {'pij': '_pij', 'cache': '_cache_laneJ',
              'pij_simplex': '_pij_simplex'}[arm]
    methods0 = PRESETS[preset]
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    _ = damped_frequency(omega, beta)

    HC, bV, mps, bc_set, dudt_fn, _setup_retopo, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=EPS_CASE, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=2, refinement_droplet=2,
            split_method=methods0.split_method,
            redistribute_mass=methods0.redistribute_mass,
        )
    methods, custom = arm_methods(preset, arm, mps)
    print(methods.describe(), flush=True)

    c_s = np.sqrt(K_d / rho_d)
    dx_min = min(
        np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(rho_d * dx_min**3 / gamma) if gamma > 0 else 1.0)
    t_end = min(t_end_3d, 5.0 / beta if beta > 0 else 0.01)
    n_steps = int(t_end / dt) + 1
    record_every = max(1, n_steps // 100)
    print(f"[{arm}] dt={dt:.2e}, n_steps={n_steps}, t_end={t_end:.4f}",
          flush=True)

    snapshots_dir = os.path.join(_RESULTS, 'snapshots' + suffix)
    os.makedirs(snapshots_dir, exist_ok=True)
    history = StateHistory(fields=['u', 'p', 'phase', 'is_interface'],
                           record_every=record_every, save_dir=snapshots_dir)
    diag_list: list[dict] = []

    def record(t):
        d = compute_diagnostics(HC, dim=dim, polar_axis='z')
        d['t'] = float(t)
        d['total_dual_vol'] = float(sum(
            float(getattr(v, 'dual_vol', 0.0) or 0.0) for v in HC.V))
        diag_list.append(d)

    record(0.0)
    eff: dict = {}
    cache_frames = [0, 0]  # [present, total]
    t_wall0 = time.perf_counter()
    step_times: list[float] = []
    t_last = [time.perf_counter()]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        now = time.perf_counter()
        step_times.append(now - t_last[0])
        if step == 0:
            eff['after_step1'] = _eff(HC_cb, dim, methods)
        cache_frames[0] += getattr(HC_cb, '_edge_area_cache', None) is not None
        cache_frames[1] += 1
        history.callback(step, t, HC_cb, bV_cb, diagnostics)
        if step % record_every == 0:
            record(t)
            if step % (record_every * 10) == 0:
                d = diag_list[-1]
                print(f"  [{arm}] step {step}/{n_steps} t={t:.4e} "
                      f"R_max={d['R_max']:.6f} KE={d['KE']:.4e} "
                      f"mass={d['total_mass']:.6e} "
                      f"wall={now - t_wall0:.0f}s", flush=True)
        t_last[0] = time.perf_counter()

    t_final = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                                bc_set=bc_set, callback=callback, mps=mps,
                                custom=custom)
    wall = time.perf_counter() - t_wall0
    record(t_final)

    score = oscillation_score_3d(
        diag_list, R0=R0, epsilon=EPS_CASE, l=l, omega=omega, beta=beta,
        r_boundary=L_domain,
    )
    # Sign decomposition (laneG par 2): quarter-horizon mean of
    # (R_max - envelope) / (eps R0).
    t_arr = np.array([d['t'] for d in diag_list])
    R_arr = np.array([d['R_max'] for d in diag_list])
    err = (R_arr - np.asarray(max_radius_envelope(
        t_arr, R0, EPS_CASE, omega, beta, l=l))) / (EPS_CASE * R0)
    q = np.array_split(err, 4)
    score['refinement_outer'] = 2
    score['refinement_droplet'] = 2
    score['retopo_policy'] = 'dual_only'
    score['laneJ'] = {
        'arm': arm, 'edge_area_source_after_step1':
            eff['after_step1'].get('edge_area_source_laneJ',
                                   eff['after_step1']['edge_area_source']),
        'cache_present_frames': cache_frames[0],
        'n_callback_frames': cache_frames[1],
        'quarter_mean_err': [float(np.mean(x)) for x in q],
        'KE_max': float(max(d['KE'] for d in diag_list)),
        'wall_s': wall,
        'wall_per_step_s': wall / n_steps,
        'median_step_s': float(np.median(step_times)),
        'two_fluid_reference': 'not applicable: add_two_fluid_reference '
                               'uses the 2D dispersion only',
    }
    save_score(os.path.join(_RESULTS, f'score{suffix}.json'), score)
    record_methods(
        os.path.join(_RESULTS, f'methods{suffix}.json'), methods, HC,
        extra={'arm': arm, 'base_preset': preset,
               'effective_after_step1': eff['after_step1'],
               'retopo_policy': 'dual_only', 'dt': dt, 'n_steps': n_steps,
               't_end': t_end, 'refinement_outer': 2,
               'refinement_droplet': 2, 'K_d': K_d, 'K_o': K_o,
               'wall_s': wall},
    )
    with open(os.path.join(_RESULTS, f'diags{suffix}.json'), 'w') as f:
        json.dump(diag_list, f, default=lambda o: np.asarray(o).tolist())
    print(json.dumps({k: score[k] for k in (
        'l2_error_normalized', 'tail_growth', 'mass_drift', 'R_max_peak',
        'dual_vol_step0_jump', 'dual_vol_drift_post', 'summary')}),
        flush=True)
    print(json.dumps(score['laneJ']), flush=True)
    return score


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('task', choices=('static', 'a5b', 'dynamic'))
    ap.add_argument('--arm', choices=('cache', 'pij', 'pij_simplex', 'all'),
                    default='all')
    ap.add_argument('--n-steps', type=int, default=20)
    a = ap.parse_args()
    os.makedirs(_RESULTS, exist_ok=True)
    if a.task == 'static':
        out = run_static()
        path = os.path.join(_RESULTS, 'laneJ_static.json')
    elif a.task == 'a5b' and a.arm == 'all':
        out = run_a5b(a.n_steps)
        path = os.path.join(_RESULTS, 'laneJ_a5b.json')
    elif a.task == 'a5b':
        out = run_a5b_arm(a.arm, a.n_steps)
        path = os.path.join(_RESULTS, f'laneJ_a5b_{a.arm}.json')
    else:
        run_dynamic('pij' if a.arm == 'all' else a.arm)
        sys.exit(0)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'wrote {path}')
