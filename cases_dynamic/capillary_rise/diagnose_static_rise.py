#!/usr/bin/env python3
"""Diagnostics of the static capillary rise on the library integrators
(laneI, 2026-10-06).  Every arm is a preset of ``ddgclib.methods.PRESETS``
or a ``.replace`` of it, run through ``src/_static.py``.

Modes (``--out DIR`` is required; nothing is written into the case
directory):

    convergence   refinement levels x initial conditions at one horizon:
                  the volume-averaged height against the Young-Laplace
                  reference (and Jurin's height), the surface RMS error,
                  the integrated pressure error, the settling numbers
    arms          the preset against its reconnecting arm
                  (delaunay_material + conservative remap)
    viscosity     artificial viscosity sweep (alpha_art) at one refinement
    series        envelope of h(t), max|u|(t) of a finished run
                  (results/<tag>/series.npz) per window of acoustic times

Examples
--------
    python cases_dynamic/capillary_rise/diagnose_static_rise.py convergence \
        --dim 2 --refinements 2 3 --n-tac 60 --out /tmp/laneI
    python cases_dynamic/capillary_rise/diagnose_static_rise.py arms \
        --dim 2 --n-refine 2 --n-tac 30 --out /tmp/laneI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.methods import PRESETS  # noqa: E402
from cases_dynamic.capillary_rise.src._static import (  # noqa: E402
    CASES, build_static_column, remap_arm, residual_acceleration,
    run_static, static_errors,
)

_PRESET = {2: 'capillary_rise_static_2D', 3: 'capillary_rise_static_3D'}
_KEYS = ('h_mean', 'h_ref', 'h_jurin', 'h_ref_poly', 'h_error_rel',
         'h_error_poly_rel', 'h_contact', 'h_contact_ref', 'h_apex',
         'h_apex_ref', 'shape_rms', 'p_l2', 'p_max_int', 'injected',
         'mass_drift', 'vol_min_rel', 'edge_min_rel')


def _envelope(res, t_ac: float, t0: float, t1: float, key: str = 'umax'):
    t = res['t'] / t_ac
    sel = (t >= t0) & (t < t1)
    return float(res[key][sel].max()) if sel.any() else float('nan')


def _one(dim: int, n_refine: int, methods, ic: str, n_tac: float,
         alpha_art: float, r: float, n_cells: int) -> dict:
    col = build_static_column(dim, n_refine, r=r, n_cells=n_cells, ic=ic)
    p = col.params
    err0 = static_errors(col)
    t0 = time.perf_counter()
    res = run_static(col, methods, n_tac=n_tac, alpha_art=alpha_art)
    wall = time.perf_counter() - t0
    err = static_errors(col)
    out = dict(
        dim=dim, n_refine=n_refine, ic=ic, n_tac=n_tac, alpha_art=alpha_art,
        r=r, n_cells=n_cells, label=methods.label, connectivity=methods.connectivity,
        n_vertices=p['n_vertices'], n_free=p['n_free'], dt=res['dt'],
        n_steps=res['n_steps'], mu=res['mu'], wall_s=wall,
        umax_peak=float(res['umax'].max()), umax_end=float(res['umax'][-1]),
        umax_env_last4=_envelope(res, p['t_ac'], n_tac - 4.0, n_tac + 1e-9),
        ke_end=float(res['ke'][-1]),
        h_env_last4=(_envelope(res, p['t_ac'], n_tac - 4.0, n_tac + 1e-9, 'h')
                     - float(res['h'][(res['t'] / p['t_ac']) >= n_tac - 4.0].min())),
        settled_max_a=residual_acceleration(col, res['dudt_fn']),
        start={k: err0[k] for k in _KEYS},
        **{k: err[k] for k in _KEYS},
    )
    print(f"  dim {dim} ref {n_refine} ic {ic:13s} {methods.connectivity:18s} "
          f"alpha {alpha_art:<5} | {p['n_vertices']:5d} v, {res['n_steps']:6d} "
          f"steps, {wall:7.1f} s | h {err['h_mean']:.6e} "
          f"({err['h_error_rel']:+.3e}; poly {err['h_error_poly_rel']:+.3e}) "
          f"shape {err['shape_rms']:.3e} p_l2 {err['p_l2']:.3e} | u peak "
          f"{out['umax_peak']:.2e} end {out['umax_end']:.2e} env4 "
          f"{out['umax_env_last4']:.2e} h-swing4 {out['h_env_last4']:.2e} "
          f"a {out['settled_max_a']:.2e} cell {err['vol_min_rel']:.3f} edge "
          f"{err['edge_min_rel']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('convergence', 'arms', 'viscosity', 'series'))
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--n-refine', type=int, default=None)
    ap.add_argument('--refinements', type=int, nargs='+', default=None)
    ap.add_argument('--ics', nargs='+', default=('young_laplace', 'flat'))
    ap.add_argument('--n-tac', type=float, default=None)
    ap.add_argument('--alpha-art', type=float, default=0.05)
    ap.add_argument('--alphas', type=float, nargs='+', default=(0.05, 0.1, 0.5))
    ap.add_argument('--r', type=float, default=None)
    ap.add_argument('--n-cells', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--series', default=None, help='series mode: results dir')
    ap.add_argument('--t-ac', type=float, default=None,
                    help='series mode: acoustic time of the run [s]')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.mode == 'series':
        d = np.load(os.path.join(args.series, 'series.npz'))
        t_ac = args.t_ac
        t = d['t'] / t_ac
        w = 4.0
        print("window [t_ac]   max|u|        h max        h min        h swing")
        start = 0.0
        while start < t[-1]:
            sel = (t >= start) & (t < start + w)
            if sel.any():
                print(f"{start:5.0f}-{start + w:<5.0f} {d['umax'][sel].max():.3e}  "
                      f"{d['h'][sel].max():.6e}  {d['h'][sel].min():.6e}  "
                      f"{d['h'][sel].max() - d['h'][sel].min():.3e}")
            start += w
        return

    defaults = CASES[_PRESET[args.dim]]
    n_tac = args.n_tac if args.n_tac is not None else defaults['n_tac']
    r = args.r if args.r is not None else defaults['r']
    n_cells = args.n_cells if args.n_cells is not None else defaults['n_cells']
    n_refine = args.n_refine if args.n_refine is not None else defaults['n_refine']
    preset = PRESETS[_PRESET[args.dim]]
    rows = []
    if args.mode == 'convergence':
        refs = args.refinements or [n_refine, n_refine + 1]
        for ic in args.ics:
            for n in refs:
                rows.append(_one(args.dim, n, preset, ic, n_tac, args.alpha_art,
                                 r, n_cells))
    elif args.mode == 'arms':
        for m in (preset, remap_arm(preset)):
            for ic in args.ics:
                rows.append(_one(args.dim, n_refine, m, ic, n_tac,
                                 args.alpha_art, r, n_cells))
    elif args.mode == 'viscosity':
        for a in args.alphas:
            for ic in args.ics:
                rows.append(_one(args.dim, n_refine, preset, ic, n_tac, a, r,
                                 n_cells))
    path = os.path.join(args.out, f'{args.mode}_{args.dim}d.json')
    with open(path, 'w') as fh:
        json.dump(rows, fh, indent=2)
        fh.write('\n')
    print(f"-> {path}")


if __name__ == '__main__':
    main()
