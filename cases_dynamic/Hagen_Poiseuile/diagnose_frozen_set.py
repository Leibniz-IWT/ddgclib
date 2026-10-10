#!/usr/bin/env python3
"""A/B of the method axis ``frozen_set`` (laneL): ``hull`` against
``membership`` on the shipped setups that freeze by hull membership.

Every arm is a preset or ``preset.replace(frozen_set=...)``; nothing here
builds a retopology function by hand.  For each arm the script records
what became of the wall vertices (the frozen set after the first
retopology), how many vertices left the domain box, the kinetic energy,
and a digest of the final state, so "neutral" can be read as "bit
identical" and "better" as "the walls did not move".

Usage (repo root)::

    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D --alpha 0.2
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D --alpha 0.5 --t-end 0.45
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_2D
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_3D --steps 300
    python cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py droplet_2D --steps 200

Output: ``results/frozen_set_ab/<case><label>_<arm>.json`` (+ the
``methods`` record of the arm) next to this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from ddgclib.methods import PRESETS, record_methods  # noqa: E402

_OUT = os.path.join(_HERE, 'results', 'frozen_set_ab')


# ---------------------------------------------------------------------------
# case builders: (preset name, HC, bV, integrate kwargs, box, liquid test)
# ---------------------------------------------------------------------------
def build_hp2d(args):
    from cases_dynamic.Hagen_Poiseuile.src._setup import (
        setup_poiseuille_2d_lagrangian,
    )
    # laneV: the preset's wall_clamp (and any --replace arm) is applied by
    # the setup; the arm's frozen_set is set by run_arm on the same config
    rep = dict(viscous_flux='two_point', **_replaces(args))
    HC, bV, bc_set, wall, params = setup_poiseuille_2d_lagrangian(
        L=args.L, methods=PRESETS['hagen_poiseuille_2D'].replace(**rep),
        clamp_gap=args.gap)
    dt = args.dt if args.dt is not None else 0.05
    n_steps = args.steps if args.steps is not None else 300
    kw = dict(dt=dt, n_steps=n_steps, bc_set=bc_set, boundary_filter=wall)
    # the outlet buffer reaches to L + 2
    box = [(0.0, args.L + 2.0), (0.0, params['D'])]
    # The reproducer is the configuration before laneH: this setup (hull
    # inlet, pressure advected with the vertices) with the two-point
    # viscous flux.  The preset is on 'simplex_gradient' since laneH.
    from cases_dynamic.Hagen_Poiseuile.src._metrics import profile_error
    # laneV: the velocity against the analytical profile on the downstream
    # half of the channel at the end (dual-volume weighted l2, laneH's
    # measure), so that a clamp arm is judged on the physics, not only on
    # the count of vertices outside
    return ('hagen_poiseuille_2D', HC, bV, kw, box, None,
            dict(mu=params['mu'], replace=rep,
                 profile=lambda HC_, bV_: profile_error(
                     HC_, bV_, dict(params, flow_axis=0), 0.5 * args.L,
                     args.L)),
            f"_L{args.L:g}_dt{dt:g}_n{n_steps}" + _rep_label(args))


def build_dam_break_2d(args):
    from cases_dynamic.dam_break.src import _params as p
    from cases_dynamic.dam_break.src._setup import (
        cfl_timestep, setup_dam_break_multiphase,
    )
    alpha = args.alpha if args.alpha is not None else p.alpha_art
    t_end = args.t_end if args.t_end is not None else p.t_end
    refine = args.refine if args.refine is not None else p.n_refine_2d
    methods = PRESETS['dam_break_2D']
    HC, bV, mps, bc_set, dudt_fn, _r, _params = setup_dam_break_multiphase(
        dim=2, a=p.a, L=p.L, H=p.H, W=p.W, col_w=p.col_w, col_h=p.col_h,
        col_d=p.col_d, rho_l=p.rho_l, rho_g=p.rho_g, mu_l=p.mu_l,
        mu_g=p.mu_g, gamma=p.gamma, K_l=p.K_l, K_g=p.K_g, g=p.g,
        gravity_axis=p.gravity_axis, P_atm=p.P_atm, n_refine=refine,
        alpha_art=alpha, methods=methods)
    dt = cfl_timestep(HC, 2, float(np.sqrt(p.K_l / p.rho_l)), cfl=p.cfl)
    n_steps = args.steps if args.steps is not None else int(t_end / dt) + 1
    kw = dict(dt=dt, n_steps=n_steps, bc_set=bc_set, mps=mps)
    box = [(0.0, p.L), (0.0, p.H)]
    return ('dam_break_2D', HC, bV, kw, box, lambda v: v.phase == 1,
            dict(dudt_fn=dudt_fn),
            f"_alpha{alpha:g}_r{refine}_n{n_steps}")


def _build_electrolysis(args, dim):
    """The shipped electrolysis_bubble_2D.py / _3D.py configuration."""
    from cases_dynamic.electrolysis_bubble.src import _params as p
    from cases_dynamic.electrolysis_bubble.src._reaction import (
        inject_gas_mass,
    )
    from cases_dynamic.electrolysis_bubble.src._setup import (
        setup_electrolysis_bubble,
    )
    preset = f'electrolysis_bubble_{dim}D'
    methods = PRESETS[preset]
    ro, rd, t_end, dm_dt, c_cfl, c_st = (
        (p.n_refine_outer_2d, p.n_refine_drop_2d, p.t_end_2d, p.dm_dt_2d,
         1.0, 0.4) if dim == 2 else
        (p.n_refine_outer_3d, p.n_refine_drop_3d, p.t_end_3d, p.dm_dt_3d,
         0.5, 0.2))
    HC, bV, mps, bc_set, dudt_fn, _r, _params = setup_electrolysis_bubble(
        dim=dim, R0=p.R0, L_domain=p.L_domain,
        nucleation_frac=p.nucleation_frac, rho_liq=p.rho_liq,
        rho_gas=p.rho_gas, mu_liq=p.mu_liq, mu_gas=p.mu_gas, gamma=p.gamma,
        K_liq=p.K_liq, K_gas=p.K_gas, g=p.g, P0=p.P0,
        refinement_outer=ro, refinement_droplet=rd,
        methods=methods, box_shift=args.box_shift)
    c_s = max(np.sqrt(p.K_liq / p.rho_liq), np.sqrt(p.K_gas / p.rho_gas))
    dx_min = min(d for d in (np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                             for v in HC.V for nb in v.nn) if d > 1e-15)
    dt = float(min(c_cfl * p.cfl_safety * dx_min / c_s,
                   c_st * np.sqrt(p.rho_liq * dx_min**3 / p.gamma)))
    n_steps = args.steps if args.steps is not None else int(t_end / dt) + 1

    def inject(step, t, HC_cb, bV_cb=None, diagnostics=None):
        inject_gas_mass(HC_cb, mps, dm_dt=dm_dt, dt=dt, gas_phase=1)

    kw = dict(dt=dt, n_steps=n_steps, bc_set=bc_set, mps=mps)
    box = [(-p.L_domain, p.L_domain)] * dim
    return (preset, HC, bV, kw, box, lambda v: v.phase == 0,
            dict(dudt_fn=dudt_fn, extra_callback=inject),
            f"_n{n_steps}{_bs_label(args)}")


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


def _replaces(args) -> dict:
    """``--replace axis=value`` arms (laneV), applied on top of the
    case's preset."""
    return {k: _parse_value(v) for k, v in
            (s.split('=', 1) for s in getattr(args, 'replace', []))}


def _rep_label(args) -> str:
    rep = _replaces(args)
    s = ''.join(f"_{k}-{v}" for k, v in sorted(rep.items()))
    if getattr(args, 'gap', None) is not None:
        s += f"_gap{args.gap:g}"
    return s


def _bs_label(args) -> str:
    """laneB: the outer box shift of the droplet builders is a setup
    choice; the lossy pre-laneB mesh (``--box-shift evict``) gets its own
    record so it never overwrites the default one."""
    return '' if args.box_shift == 'move_all' else f"_bs-{args.box_shift}"


def build_electrolysis_2d(args):
    return _build_electrolysis(args, 2)


def build_electrolysis_3d(args):
    return _build_electrolysis(args, 3)


def build_droplet_2d(args):
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    methods = PRESETS['oscillating_droplet_2D']
    R0, L = 0.01, 0.05
    HC, bV, mps, bc_set, dudt_fn, _r, params = setup_oscillating_droplet(
        dim=2, R0=R0, epsilon=0.05, l=2, L_domain=L, refinement_outer=2,
        refinement_droplet=2, methods=methods, box_shift=args.box_shift)
    n_steps = args.steps if args.steps is not None else 200
    kw = dict(dt=args.dt if args.dt is not None else 2e-5, n_steps=n_steps,
              bc_set=bc_set, mps=mps)
    return ('oscillating_droplet_2D', HC, bV, kw, [(-L, L), (-L, L)],
            lambda v: v.phase == 1, dict(dudt_fn=dudt_fn),
            f"_n{n_steps}{_bs_label(args)}")


BUILDERS = {
    'hp2d': build_hp2d,
    'dam_break_2D': build_dam_break_2d,
    'electrolysis_2D': build_electrolysis_2d,
    'electrolysis_3D': build_electrolysis_3d,
    'droplet_2D': build_droplet_2d,
}


# ---------------------------------------------------------------------------
def run_arm(case: str, arm: str, args) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        preset, HC, bV, kw, box, in_phase, extra, label = BUILDERS[case](args)
    methods = PRESETS[preset].replace(frozen_set=arm,
                                      **extra.get('replace', {}))
    if methods.workers:                 # serial: same numbers, no fork cost
        methods = methods.replace(workers=None)
    dim = methods.dim
    dudt_fn = extra.get('dudt_fn') or methods.dudt_fn(HC, mu=extra['mu'])
    extra_cb = extra.get('extra_callback')
    bfilter = kw.get('boundary_filter')
    tol = 1e-12 * max(hi - lo for lo, hi in box)

    def outside(v):
        return any(v.x_a[i] < lo - tol or v.x_a[i] > hi + tol
                   for i, (lo, hi) in enumerate(box))

    walls: dict = {}
    series: list[dict] = []
    first = {'frozen_changed': None, 'outside': None, 'wall_moved': None}
    clamp = {'n_events': 0, 'n_steps': 0, 'first': None}
    every = max(1, kw['n_steps'] // 60)
    n_done = [0]

    def callback(step, t, HC, bV=None, diagnostics=None):
        n_done[0] = step + 1
        n_cl = sum(int(n) for k, n in (diagnostics or {}).items()
                   if 'WallClampBC' in k)
        if n_cl:                              # laneV: the wall clamp fired
            clamp['n_events'] += n_cl
            clamp['n_steps'] += 1
            if clamp['first'] is None:
                clamp['first'] = step
        if extra_cb is not None:
            extra_cb(step, t, HC, bV, diagnostics)
        if step == 0:
            # the frozen set of the first retopology (plus what the BCs
            # added in the first pass) is "the walls"
            walls.update({id(v): (v, v.x_a.copy()) for v in bV})
        frozen = {id(v) for v in bV}
        n_wall_frozen = sum(k in frozen for k in walls)
        n_moved = sum(not np.array_equal(v.x_a, x0)
                      for v, x0 in walls.values())
        n_out = sum(1 for v in HC.V if outside(v))
        if first['frozen_changed'] is None and n_wall_frozen != len(walls):
            first['frozen_changed'] = step
        if first['wall_moved'] is None and n_moved:
            first['wall_moved'] = step
        if first['outside'] is None and n_out:
            first['outside'] = step
        if step % every == 0 or step == kw['n_steps'] - 1:
            sel = [v for v in HC.V if in_phase is None or in_phase(v)]
            series.append({
                'step': step, 't': float(t), 'n_vertices': len(HC.V),
                'n_frozen': len(bV), 'n_wall_frozen': n_wall_frozen,
                'n_wall_moved': n_moved, 'n_outside': n_out,
                'KE': float(sum(0.5 * v.m * np.dot(v.u[:dim], v.u[:dim])
                                for v in sel)),
                'u_max': float(max((np.linalg.norm(v.u[:dim])
                                    for v in HC.V), default=0.0)),
            })

    t0 = time.time()
    abort = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            methods.integrate(HC, bV, dudt_fn, callback=callback, **kw)
    except Exception as e:  # noqa: BLE001 - an abort is a result here
        abort = f"{type(e).__name__}: {e}"[:300]
    wall = time.time() - t0

    moved = [float(np.linalg.norm(v.x_a - x0)) for v, x0 in walls.values()]
    frozen = {id(v) for v in bV}
    state = sorted((tuple(float(c) for c in v.x_a[:dim]),
                    tuple(float(c) for c in v.u[:dim]), float(v.m))
                   for v in HC.V)
    out = {
        'case': case, 'arm': arm, 'preset': preset, 'label': label,
        'dt': kw['dt'], 'n_steps': kw['n_steps'],
        'steps_done': n_done[0],
        'abort': abort, 'wall_time_s': round(wall, 1),
        'n_walls': len(walls),
        'n_walls_still_frozen': sum(k in frozen for k in walls),
        'n_walls_moved': sum(d > 0.0 for d in moved),
        'max_wall_displacement': max(moved, default=0.0),
        'first_step': first,
        'final': series[-1] if series else None,
        # maxima over the sampled series (every n_steps // 60 steps)
        'KE_max': max((s['KE'] for s in series), default=0.0),
        'u_max_max': max((s['u_max'] for s in series), default=0.0),
        'n_outside_max': max((s['n_outside'] for s in series), default=0),
        'mass_total': float(sum(v.m for v in HC.V)),
        'state_sha256': hashlib.sha256(repr(state).encode()).hexdigest(),
        'clamp': clamp,
        'profile': (extra['profile'](HC, bV) if 'profile' in extra
                    and abort is None else None),
        'series': series,
    }
    out_dir = args.out or _OUT
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{case}{label}_{arm}")
    with open(stem + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    record_methods(stem + '_methods.json', methods, HC,
                   extra={'dt': kw['dt'], 'n_steps': kw['n_steps'],
                          'case': case, 'label': label,
                          'box_shift': args.box_shift})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('case', choices=sorted(BUILDERS))
    ap.add_argument('--arm', default='both',
                    choices=['both', 'hull', 'membership'])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--L', type=float, default=2.0, help='hp2d channel length')
    ap.add_argument('--alpha', type=float, default=None,
                    help='dam break alpha_art')
    ap.add_argument('--t-end', type=float, default=None, dest='t_end')
    ap.add_argument('--refine', type=int, default=None)
    ap.add_argument('--box-shift', default='move_all', dest='box_shift',
                    choices=['move_all', 'evict'],
                    help='droplet / electrolysis builders (laneB): evict = '
                         'the lossy pre-laneB outer mesh')
    ap.add_argument('--replace', action='append', default=[],
                    metavar='AXIS=VALUE',
                    help='preset.replace(axis=value) arm (laneV; hp2d only)')
    ap.add_argument('--gap', type=float, default=None,
                    help='wall_clamp put-down gap of the hp2d setup (laneV)')
    ap.add_argument('--out', default=None,
                    help=f'output directory (default {_OUT})')
    args = ap.parse_args()

    arms = ['hull', 'membership'] if args.arm == 'both' else [args.arm]
    res = {}
    for arm in arms:
        r = res[arm] = run_arm(args.case, arm, args)
        f = r['final'] or {}
        print(f"{args.case}{r['label']} [{arm:>10}] steps {r['steps_done']}"
              f"/{r['n_steps']}  walls {r['n_walls']}: frozen "
              f"{r['n_walls_still_frozen']}, moved {r['n_walls_moved']} "
              f"(max {r['max_wall_displacement']:.3e})  outside max "
              f"{r['n_outside_max']}  first: {r['first_step']}\n"
              f"    KE_max {r['KE_max']:.6e}  KE_end {f.get('KE', 0.0):.6e}  "
              f"|u|max {r['u_max_max']:.4e}  nV_end {f.get('n_vertices')}  "
              f"mass {r['mass_total']:.15e}  {r['wall_time_s']} s"
              + (f"\n    ABORT {r['abort']}" if r['abort'] else ''))
        print(f"    wall clamp: {r['clamp']['n_events']} put-backs on "
              f"{r['clamp']['n_steps']} steps (first {r['clamp']['first']})")
        if r.get('profile'):
            pr = r['profile']
            print(f"    profile on x in [L/2, L]: l2 {pr['l2']:.6e}  u_max "
                  f"{pr['u_max']:.6f}  u_cross {pr['u_cross']:.3e}  "
                  f"({pr['n']} vertices)")
    if len(res) == 2:
        same = res['hull']['state_sha256'] == res['membership']['state_sha256']
        print(f"final states bit-identical: {same}")


if __name__ == '__main__':
    main()
