#!/usr/bin/env python3
"""Census and A/B of the method axis ``phase_ledger`` (laneF, 2026-10-05).

For a shipped multiphase configuration (the builders of
``cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py``: dam_break_2D,
droplet_2D, electrolysis_2D / _3D) this runs the preset, or
``preset.replace(phase_ledger=...)``, and counts at every step the
(vertex, phase) pairs that have a dual sub-volume but no mass ("holes":
the force reads ``p_phase = 0`` absolute there) and the pairs that have
mass but no sub-volume ("stranded").  It reports the first step of each,
the number of steps affected, the per-phase mass drift, the kinetic
energy maximum and a digest of the final state, so an arm can be read
as bit-identical to another.

Usage (repo root)::

    python cases_dynamic/dam_break/diagnose_phase_ledger.py droplet_2D --steps 200
    python cases_dynamic/dam_break/diagnose_phase_ledger.py dam_break_2D --alpha 0.2 --arm both
    python cases_dynamic/dam_break/diagnose_phase_ledger.py electrolysis_2D --steps 1000

Output: ``<out>/<case><label>_<arm>.json`` (default
``results/phase_ledger/`` next to this script).
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
from cases_dynamic.Hagen_Poiseuile.diagnose_frozen_set import (  # noqa: E402
    BUILDERS,
)
from cases_dynamic.dam_break.diagnose_sliver_ejection import (  # noqa: E402
    _holes, _stranded,
)

_OUT = os.path.join(_HERE, 'results', 'phase_ledger')


def run_arm(case: str, arm: str, args) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        preset, HC, bV, kw, box, in_phase, extra, label = BUILDERS[case](args)
    methods = PRESETS[preset].replace(**extra.get('replace', {}))
    if methods.phases != 'multi':
        raise SystemExit(f"{case}: single phase, no ledger")
    methods = methods.replace(phase_ledger=arm)
    if methods.workers:
        methods = methods.replace(workers=None)
    dim = methods.dim
    dudt_fn = extra['dudt_fn']
    extra_cb = extra.get('extra_callback')
    mps = kw['mps']
    n_phases = mps.n_phases
    M0 = np.sum([v.m_phase for v in HC.V], axis=0)

    first = {'hole': None, 'stranded': None}
    counts = {'hole_steps': 0, 'stranded_steps': 0, 'holes': 0, 'stranded': 0}
    events: list[dict] = []
    ke_max = [0.0]
    n_done = [0]

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        n_done[0] = step + 1
        if extra_cb is not None:
            extra_cb(step, t, HC_cb, bV_cb, diagnostics)
        holes = _holes(HC_cb, n_phases)
        stranded = _stranded(HC_cb, n_phases)
        if holes:
            counts['hole_steps'] += 1
            counts['holes'] += len(holes)
            if first['hole'] is None:
                first['hole'] = step
        if stranded:
            counts['stranded_steps'] += 1
            counts['stranded'] += len(stranded)
            if first['stranded'] is None:
                first['stranded'] = step
        if (holes or stranded) and len(events) < 200:
            byid = {id(v): v for v in HC_cb.V}
            events.append(dict(
                step=step, t=float(t),
                holes=[dict(k=k, x=[float(c) for c in byid[i].x_a[:dim]],
                            p=float(byid[i].p_phase[k]))
                       for i, k in holes],
                stranded=[dict(k=k, m=m,
                               x=[float(c) for c in byid[i].x_a[:dim]])
                          for i, k, m in stranded]))
        ke_max[0] = max(ke_max[0], float(sum(
            0.5 * v.m * np.dot(v.u[:dim], v.u[:dim])
            for v in HC_cb.V if in_phase is None or in_phase(v))))

    t0 = time.time()
    abort = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            methods.integrate(HC, bV, dudt_fn, callback=callback, **kw)
    except Exception as e:  # noqa: BLE001 - an abort is a result here
        abort = f"{type(e).__name__}: {e}"[:300]
    wall = time.time() - t0
    state = sorted((tuple(float(c) for c in v.x_a[:dim]),
                    tuple(float(c) for c in v.u[:dim]), float(v.m))
                   for v in HC.V)
    M1 = np.sum([v.m_phase for v in HC.V], axis=0)
    out = dict(
        case=case, arm=arm, preset=preset, label=label,
        dt=kw['dt'], n_steps=kw['n_steps'], steps_done=n_done[0],
        abort=abort, wall_time_s=round(wall, 1),
        first=first, counts=counts, events=events,
        KE_max=ke_max[0],
        mass_phase_rel_drift=[float(c) for c in (M1 / M0 - 1.0)],
        n_vertices=len(HC.V),
        state_sha256=hashlib.sha256(repr(state).encode()).hexdigest(),
    )
    out_dir = args.out or _OUT
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{case}{label}_{arm}")
    with open(stem + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    record_methods(stem + '_methods.json', methods, HC,
                   extra={'dt': kw['dt'], 'n_steps': kw['n_steps'],
                          'case': case, 'label': label})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('case', choices=sorted(c for c in BUILDERS if c != 'hp2d'))
    ap.add_argument('--arm', default='snapshot',
                    choices=['both', 'snapshot', 'volume'])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--L', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--t-end', type=float, default=None, dest='t_end')
    ap.add_argument('--refine', type=int, default=None)
    ap.add_argument('--box-shift', default='move_all', dest='box_shift',
                    choices=['move_all', 'evict'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    arms = ['snapshot', 'volume'] if args.arm == 'both' else [args.arm]
    res = {}
    for arm in arms:
        r = res[arm] = run_arm(args.case, arm, args)
        print(f"{args.case}{r['label']} [{arm:>8}] steps {r['steps_done']}/"
              f"{r['n_steps']}  holes: first {r['first']['hole']}, "
              f"{r['counts']['hole_steps']} steps / {r['counts']['holes']} "
              f"pairs; stranded: first {r['first']['stranded']}, "
              f"{r['counts']['stranded_steps']} steps / "
              f"{r['counts']['stranded']} pairs; KE_max {r['KE_max']:.6e}; "
              f"mass drift {r['mass_phase_rel_drift']}; nV {r['n_vertices']}; "
              f"state {r['state_sha256'][:16]}; {r['wall_time_s']} s"
              + (f"\n    ABORT {r['abort']}" if r['abort'] else ''))
    if len(res) == 2:
        print("final states bit-identical: "
              f"{res['snapshot']['state_sha256'] == res['volume']['state_sha256']}")


if __name__ == '__main__':
    main()
