#!/usr/bin/env python3
"""3D edge-area source A/B: the method axis ``edge_area_source`` (laneJ
measured it with case-side wrappers, laneQ made it a library axis).

MEASUREMENT ONLY.  Every arm is ``PRESETS[...]`` or
``preset.replace(...)``; the arm names map to ``replace`` keywords in
:data:`ARMS`.  ``cache`` is the preset itself without ``replace``: on the
droplet and dam-break presets ``edge_area_source=None`` = the legacy
``batch_e_star`` fan cache (the ``hydrostatic`` sub-command runs the
explicit sources instead, because ``hydrostatic_3D`` reads
``p_ij_simplex`` since laneQ); ``pij_simplex`` the exact cached faces of
``hyperct.ddg.simplex_dual_face_areas``, ``pij`` the same face built per
edge, ``pij_ring`` the legacy ring walk (laneJ's ``pij`` arm, the
pre-laneQ ``hydrostatic_3D``).  The ``*_noredis`` and ``*_p2`` arms are
laneG's lever (b): ``redistribute_mass=False`` and ``projection_every=2``.

Sub-commands (run from the repo root with the ddg env python)::

    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py static
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py a5b [--arm NAME]
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dynamic --arm NAME [--snapshots]
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py hydrostatic [--refine N] [--n-tac T]
    python cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dambreak [--n-steps N]

``--out DIR`` redirects every output (default ``results_3d/``, where the
kept arms of laneJ and laneQ live: ``laneQ_static.json``,
``laneQ_a5b.json``, ``score_laneQ_<arm>.json`` / ``methods_laneQ_<arm>.json``
/ ``diags_laneQ_<arm>.json``, ``laneQ_hydrostatic*.json``,
``laneQ_dambreak.json``).  The laneJ files (``laneJ_*.json``,
``score_pij*.json``, ...) were produced by the previous version of this
driver with ``connectivity='custom'`` wrappers; ``pij_ring`` reproduces
its ``pij`` arm through the axis.
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

# arm name -> SolverMethods.replace(...) keywords
ARMS: dict[str, dict] = {
    'cache': {},
    'pij_simplex': dict(edge_area_source='p_ij_simplex'),
    'pij': dict(edge_area_source='p_ij'),
    'pij_ring': dict(edge_area_source='p_ij_ring'),
    'cache_noredis': dict(redistribute_mass=False),
    'pij_simplex_noredis': dict(edge_area_source='p_ij_simplex',
                                redistribute_mass=False),
    'cache_p2': dict(projection_every=2),
    'pij_simplex_p2': dict(edge_area_source='p_ij_simplex',
                           projection_every=2),
}
SOURCE_ARMS = ('cache', 'pij_simplex', 'pij', 'pij_ring')
# hydrostatic_3D builds no fan cache: its sources are the explicit three
HYDRO_ARMS = ('pij_ring', 'pij_simplex', 'pij')


def arm_methods(preset: str, arm: str):
    methods = PRESETS[preset]
    kw = ARMS[arm]
    if not kw:
        return methods
    return methods.replace(label=f"{methods.label} [laneQ arm {arm}]",
                           notes=f"laneQ arm {arm!r} of preset {preset!r}: "
                                 f"{kw}", **kw)


def _setup(methods, eps, refine=2):
    return setup_oscillating_droplet(
        dim=3, R0=R0, epsilon=eps, l=l, rho_d=rho_d, rho_o=rho_o,
        mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
        L_domain=L_domain, refinement_outer=refine, refinement_droplet=refine,
        split_method=methods.split_method,
        redistribute_mass=methods.redistribute_mass,
    )


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


PATHS = ('cache', 'p_ij', 'p_ij_ring')


def probe_mesh(HC, bV, label: str, groups: dict | None = None) -> dict:
    """Per-edge A_ij diff, linear precision and closure on three paths:
    the e_star cache as the force reads it (``cache``), the exact per-edge
    construction (``p_ij``) and the legacy ring walk (``p_ij_ring``).

    *groups*: optional {name: predicate(v)} for extra vertex classes.
    Standard classes: 'interior' (not in bV, no bV neighbour),
    'boundary_adjacent' (not in bV, >= 1 bV neighbour), 'boundary' (in
    bV: open truncated cell, closure not expected; the cache path reads
    the ring walk there because batch_e_star only caches interior
    vertices).
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

    rel_diff = {'cache_vs_p_ij': [], 'p_ij_ring_vs_p_ij': []}
    n_cache_hit = n_cache_miss = 0
    per_v: dict[str, dict] = {p: {} for p in PATHS}
    for v in HC.V:
        nbs = list(v.nn)
        As = {p: [] for p in PATHS}
        for nb in nbs:
            a_c, hit = _A_cache(HC, v, nb)
            a_p = dual_area_vector(v, nb, HC, 3, source='p_ij')
            a_r = dual_area_vector(v, nb, HC, 3, source='p_ij_ring')
            As['cache'].append(a_c)
            As['p_ij'].append(a_p)
            As['p_ij_ring'].append(a_r)
            if v not in bV:
                n_cache_hit += hit
                n_cache_miss += (not hit)
                if np.linalg.norm(a_p) > 0:
                    rel_diff['cache_vs_p_ij'].append(
                        np.linalg.norm(a_c - a_p) / np.linalg.norm(a_p))
                    rel_diff['p_ij_ring_vs_p_ij'].append(
                        np.linalg.norm(a_r - a_p) / np.linalg.norm(a_p))
        for path in PATHS:
            A = np.array(As[path]) if As[path] else np.zeros((0, 3))
            s = A.sum(axis=0) if len(A) else np.zeros(3)
            ssum = float(np.sum(np.linalg.norm(A, axis=1))) if len(A) else 0.0
            M = np.zeros((3, 3))
            for nb, a in zip(nbs, A):
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
    # exact integrated force is -g * Vol_i.  The force reads the cache
    # when present, else dual_area_vector(source=HC._edge_area_source).
    cache_saved = HC._edge_area_cache
    source_saved = getattr(HC, '_edge_area_source', None)
    for path in PATHS:
        if path == 'cache':
            HC._edge_area_cache = cache_saved
            HC._edge_area_source = source_saved
        else:
            HC._edge_area_cache = None
            HC._edge_area_source = path
        for v in HC.V:
            rec = per_v[path][id(v)]
            if v in bV or rec['vol'] <= 0:
                rec['force_rel'] = None
                continue
            F = stress_force(v, dim=3, mu=0.0, HC=HC)
            rec['force_rel'] = float(np.linalg.norm(F + g * rec['vol'])
                                     / (np.linalg.norm(g) * rec['vol']))
    HC._edge_area_cache = cache_saved
    HC._edge_area_source = source_saved

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
           'edge_rel_diff': {k: _stats(d) for k, d in rel_diff.items()},
           'n_directed_interior_edges_cache_hit': int(n_cache_hit),
           'n_directed_interior_edges_cache_miss': int(n_cache_miss),
           'classes': {}}
    for cname, pred in classes.items():
        vs = [v for v in HC.V if pred(v)]
        entry = {'n': len(vs)}
        for path in PATHS:
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
        HC, bV, mps, *_ = _setup(methods, eps)
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
    methods = arm_methods(preset, arm)
    HC, bV, mps, bc_set, dudt_fn, _setup_retopo, params = _setup(methods, 0.0)
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
            eff['after_step1'] = effective_methods(HC_cb, dim, methods)
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
                      bc_set=bc_set, callback=cb, mps=mps)
    wall = time.perf_counter() - t0
    eff['end'] = effective_methods(HC, dim, methods)
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
                n_steps=n_steps, curvature_path='integrated',
                methods=PRESETS['static_droplet_floor_3D'])
    out['harness_cache'] = {k: h[k] for k in (
        'max_abs_F_peak', 'max_abs_F_end', 'max_abs_F_history',
        'mass_rel_drift', 'dt')}
    out['harness_cache']['wall_s'] = time.perf_counter() - t0
    out['pin'] = 7.274134e-05
    for arm in SOURCE_ARMS:
        out[arm] = run_a5b_arm(arm, n_steps)
    return out


# ---------------------------------------------------------------------------
# Full 3D dynamic droplet (mirror of oscillating_droplet_3D.py main)
# ---------------------------------------------------------------------------
def run_dynamic(arm: str, out_dir: str, snapshots: bool = False) -> dict:
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
    suffix = f'_laneQ_{arm}'
    methods = arm_methods(preset, arm)
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    _ = damped_frequency(omega, beta)

    HC, bV, mps, bc_set, dudt_fn, _setup_retopo, params = _setup(methods, EPS_CASE)
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

    snapshots_dir = None
    if snapshots:
        snapshots_dir = os.path.join(out_dir, 'snapshots' + suffix)
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
            eff['after_step1'] = effective_methods(HC_cb, dim, methods)
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
                                bc_set=bc_set, callback=callback, mps=mps)
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
    score['laneQ'] = {
        'arm': arm, 'replace': ARMS[arm],
        'edge_area_source_after_step1': eff['after_step1']['edge_area_source'],
        'cache_present_frames': cache_frames[0],
        'n_callback_frames': cache_frames[1],
        'quarter_mean_err': [float(np.mean(x)) for x in q],
        'R_max_end': float(R_arr[-1]),
        'KE_max': float(max(d['KE'] for d in diag_list)),
        'wall_s': wall,
        'wall_per_step_s': wall / n_steps,
        'median_step_s': float(np.median(step_times)),
        'two_fluid_reference': 'not applicable: add_two_fluid_reference '
                               'uses the 2D dispersion only',
    }
    save_score(os.path.join(out_dir, f'score{suffix}.json'), score,
               methods=methods)
    record_methods(
        os.path.join(out_dir, f'methods{suffix}.json'), methods, HC,
        extra={'arm': arm, 'base_preset': preset,
               'effective_after_step1': eff['after_step1'],
               'retopo_policy': 'dual_only', 'dt': dt, 'n_steps': n_steps,
               't_end': t_end, 'refinement_outer': 2,
               'refinement_droplet': 2, 'K_d': K_d, 'K_o': K_o,
               'box_shift': params.get('box_shift'), 'wall_s': wall},
    )
    with open(os.path.join(out_dir, f'diags{suffix}.json'), 'w') as f:
        json.dump(diag_list, f, default=lambda o: np.asarray(o).tolist())
    print(json.dumps({k: score[k] for k in (
        'l2_error_normalized', 'tail_growth', 'mass_drift', 'R_max_peak',
        'dual_vol_step0_jump', 'dual_vol_drift_post', 'summary')}),
        flush=True)
    print(json.dumps(score['laneQ']), flush=True)
    return score


# ---------------------------------------------------------------------------
# Hydrostatic 3D column (lane P / lane T case of the hull-edge tie)
# ---------------------------------------------------------------------------
def run_hydrostatic(refine: int, n_tac: float, arms=HYDRO_ARMS) -> dict:
    """``hydrostatic_3D`` (dual_only_bare, no fan cache; the preset reads
    ``p_ij_simplex`` since laneQ, ``pij_ring`` is the preset as it was
    before) and its remap arm (delaunay_material) under each explicit
    source.  Reports the peak and end |u|, the end KE and the integrated
    pressure error, with the wall time."""
    from cases_dynamic.Hydrostatic_column.src._column import (
        build_column, column_errors, remap_arm, run_column, CASES,
    )
    kw = CASES['hydrostatic_3D']
    out: dict = {'refine': refine, 'n_tac': n_tac, 'arms': {}}
    for base_name, base in (('preset', PRESETS['hydrostatic_3D']),
                            ('remap', remap_arm(PRESETS['hydrostatic_3D']))):
        for arm in arms:
            rep = dict(ARMS[arm])
            m = base.replace(**rep) if rep else base
            col = build_column(3, refine, H=kw['H'], side_walls=kw['side_walls'],
                               ic='drop')
            t0 = time.perf_counter()
            res = run_column(col, m, n_tac=n_tac)
            wall = time.perf_counter() - t0
            err = column_errors(col)
            eff = effective_methods(col.HC, 3, m)
            row = dict(
                config=m.to_dict(), edge_area_source=eff['edge_area_source'],
                n_steps=int(res['n_steps']), dt=float(res['dt']),
                umax_peak=float(res['umax'].max()),
                umax_end=float(res['umax'][-1]),
                umax_last4=float(res['umax'][int(len(res['umax']) * (1 - 4 / n_tac)):].max())
                if n_tac > 4 else float(res['umax'].max()),
                ke_end=float(res['ke'][-1]), ke_peak=float(res['ke'].max()),
                l2=float(err['l2']), l2_interior=float(err['l2_interior']),
                rho_g_H=float(err['rho_g_H']), mass_drift=float(err['mass_drift']),
                wall_s=wall, wall_per_step_s=wall / int(res['n_steps']),
            )
            out['arms'][f'{base_name}/{arm}'] = row
            print(f"  [{base_name}/{arm}] src {row['edge_area_source']:12s} "
                  f"umax peak {row['umax_peak']:.6e} end {row['umax_end']:.3e} "
                  f"KE end {row['ke_end']:.12e} l2 {row['l2']:.3e} "
                  f"{wall:.0f} s ({row['wall_per_step_s'] * 1e3:.0f} ms/step)",
                  flush=True)
    return out


# ---------------------------------------------------------------------------
# Dam break 3D smoke
# ---------------------------------------------------------------------------
def run_dambreak(n_steps: int, arms=('cache', 'pij_simplex', 'pij')) -> dict:
    from cases_dynamic.dam_break.src._params import (
        H, L, W, a, alpha_art, cfl, col_d, col_h, col_w, g, gamma as gam,
        gravity_axis, K_g, K_l, mu_g, mu_l, n_refine_3d, P_atm, rho_g, rho_l,
    )
    from cases_dynamic.dam_break.src._setup import (
        cfl_timestep, setup_dam_break_multiphase,
    )
    from ddgclib.data import compute_conservation

    out: dict = {'n_steps': n_steps, 'arms': {}}
    for arm in arms:
        m = PRESETS['dam_break_3D']
        rep = dict(ARMS[arm])
        if rep:
            m = m.replace(**rep)
        HC, bV, mps, bc_set, dudt_fn, _r, params = setup_dam_break_multiphase(
            dim=3, a=a, L=L, H=H, W=W, col_w=col_w, col_h=col_h, col_d=col_d,
            rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g, gamma=gam,
            K_l=K_l, K_g=K_g, g=g, gravity_axis=gravity_axis, P_atm=P_atm,
            n_refine=n_refine_3d, alpha_art=alpha_art,
            redistribute_mass=m.redistribute_mass)
        c_s = np.sqrt(K_l / rho_l)
        dt = cfl_timestep(HC, 3, c_s, cfl=cfl)
        m0 = compute_conservation(HC, dim=3)['mass_total']
        ke_hist = []

        def cb(step, t, HC_cb, bV_cb=None, diagnostics=None, ke_hist=ke_hist):
            ke_hist.append(sum(0.5 * v.m * float(np.dot(v.u[:3], v.u[:3]))
                               for v in HC_cb.V if v.phase == 1))

        t0 = time.perf_counter()
        status = 'ok'
        try:
            m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps, bc_set=bc_set,
                        callback=cb, mps=mps)
        except Exception as e:  # noqa: BLE001 - smoke report
            status = f'aborted: {type(e).__name__}: {e}'
        wall = time.perf_counter() - t0
        row = dict(
            config=m.to_dict(), status=status, dt=float(dt),
            steps_done=len(ke_hist),
            edge_area_source=effective_methods(HC, 3, m)['edge_area_source'],
            mass_rel_drift=float(abs(compute_conservation(HC, dim=3)['mass_total'] - m0) / m0),
            ke_liq_end=float(ke_hist[-1]) if ke_hist else None,
            ke_liq_max=float(max(ke_hist)) if ke_hist else None,
            umax_end=float(max(np.linalg.norm(v.u[:3]) for v in HC.V)),
            wall_s=wall, wall_per_step_s=wall / max(len(ke_hist), 1),
        )
        out['arms'][arm] = row
        print(f"  [{arm}] {status}; src {row['edge_area_source']}, steps "
              f"{row['steps_done']}, KE_liq end {row['ke_liq_end']:.6e} max "
              f"{row['ke_liq_max']:.6e}, |u|max {row['umax_end']:.4f}, mass "
              f"drift {row['mass_rel_drift']:.1e}, {wall:.0f} s "
              f"({row['wall_per_step_s'] * 1e3:.0f} ms/step)", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('task', choices=('static', 'a5b', 'dynamic', 'hydrostatic',
                                     'dambreak'))
    ap.add_argument('--arm', choices=tuple(ARMS) + ('all',), default='all')
    ap.add_argument('--n-steps', type=int, default=20)
    ap.add_argument('--refine', type=int, default=1)
    ap.add_argument('--n-tac', type=float, default=40.0)
    ap.add_argument('--snapshots', action='store_true',
                    help='dynamic: also write the StateHistory snapshots')
    ap.add_argument('--out', default=_RESULTS,
                    help='output directory (default results_3d/)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.task == 'static':
        out = run_static()
        path = os.path.join(a.out, 'laneQ_static.json')
    elif a.task == 'a5b' and a.arm == 'all':
        out = run_a5b(a.n_steps)
        path = os.path.join(a.out, 'laneQ_a5b.json')
    elif a.task == 'a5b':
        out = run_a5b_arm(a.arm, a.n_steps)
        path = os.path.join(a.out, f'laneQ_a5b_{a.arm}.json')
    elif a.task == 'hydrostatic':
        arms = HYDRO_ARMS if a.arm == 'all' else (a.arm,)
        out = run_hydrostatic(a.refine, a.n_tac, arms)
        path = os.path.join(a.out, f'laneQ_hydrostatic_r{a.refine}_t{a.n_tac:g}'
                            + ('' if a.arm == 'all' else f'_{a.arm}') + '.json')
    elif a.task == 'dambreak':
        arms = ('cache', 'pij_simplex', 'pij') if a.arm == 'all' else (a.arm,)
        out = run_dambreak(a.n_steps, arms)
        path = os.path.join(a.out, 'laneQ_dambreak'
                            + ('' if a.arm == 'all' else f'_{a.arm}')
                            + f'_{a.n_steps}.json')
    else:
        run_dynamic('cache' if a.arm == 'all' else a.arm, a.out,
                    snapshots=a.snapshots)
        sys.exit(0)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'wrote {path}')
