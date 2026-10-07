#!/usr/bin/env python3
"""Electrolysis bubble through its preset, with the integrated checks of
laneG (2026-10-06): the Laplace jump of the gas against the analytical
``gamma (dim - 1) / R0`` (``ddgclib.analytical.integrated_phase_pressure_jump``,
bulk cells, volume weighted), the gas volume against the circle / sphere,
the per-phase mass ledger (the gas mass against ``M0 + dm_dt t`` under
injection), the kinetic energy and the largest speed.

Default: a STATIC bubble (``g = 0``, no injection) of the case's
parameters, the configuration in which the analytical solution is the
preloaded state itself.  ``--inject`` adds the case's gas source
(``inject_gas_mass``) in the callback, ``--g 9.81`` the case's gravity.
Every arm is ``PRESETS['electrolysis_bubble_<dim>D']`` or
``preset.replace(...)`` (``--replace axis=value``, repeatable).

Usage (from the repository root)::

    python cases_dynamic/electrolysis_bubble/diagnose_static_bubble.py
    python ... --dim 3 --ro 2 --rd 2 --steps 2000
    python ... --replace remap=conservative
    python ... --inject --g 9.81 --steps 2300     # the shipped 3D horizon
    python ... --out /tmp/some/dir                 # JSON summary there

Writes nothing unless ``--out`` is given.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.electrolysis_bubble.src import _params as P  # noqa: E402
from cases_dynamic.electrolysis_bubble.src._analytical import (  # noqa: E402
    young_laplace_jump,
)
from cases_dynamic.electrolysis_bubble.src._reaction import (  # noqa: E402
    inject_gas_mass,
)
from cases_dynamic.electrolysis_bubble.src._setup import (  # noqa: E402
    setup_electrolysis_bubble,
)
from ddgclib.analytical import integrated_phase_pressure_jump  # noqa: E402
from ddgclib.methods import PRESETS  # noqa: E402


def _coerce(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {'None': None, 'True': True, 'False': False}.get(text, text)


def build(dim, ro, rd, g, methods, box_shift='move_all', clamp=True):
    return setup_electrolysis_bubble(
        dim=dim, R0=P.R0, L_domain=P.L_domain,
        nucleation_frac=P.nucleation_frac, rho_liq=P.rho_liq,
        rho_gas=P.rho_gas, mu_liq=P.mu_liq, mu_gas=P.mu_gas, gamma=P.gamma,
        K_liq=P.K_liq, K_gas=P.K_gas, g=g, P0=P.P0, refinement_outer=ro,
        refinement_droplet=rd, box_shift=box_shift,
        # laneV: the clamp is the axis wall_clamp of the methods
        methods=methods if clamp else methods.replace(wall_clamp=None))


def time_step(HC, dim):
    """The runners' CFL step (``electrolysis_bubble_2D.py`` / ``_3D.py``)."""
    c_s = max(np.sqrt(P.K_liq / P.rho_liq), np.sqrt(P.K_gas / P.rho_gas))
    dx_min = min(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                 for v in HC.V for nb in v.nn
                 if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    dt_cfl = (0.5 if dim == 3 else 1.0) * P.cfl_safety * dx_min / c_s
    dt_st = 0.2 * np.sqrt(P.rho_liq * dx_min ** 3 / P.gamma)
    return float(min(dt_cfl, dt_st))


def digest(HC, dim):
    state = sorted((tuple(v.x_a[:dim]), tuple(v.u[:dim]), float(v.m),
                    tuple(float(p) for p in v.p_phase)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


def measure(HC, dim):
    iface = np.array([v.x_a[:dim] for v in HC.V
                      if getattr(v, 'is_interface', False)])
    centre = iface.mean(axis=0) if len(iface) else np.zeros(dim)
    return dict(
        R_interface=float(np.linalg.norm(iface - centre, axis=1).mean())
        if len(iface) else 0.0,
        V_gas=sum(float(v.dual_vol_phase[1]) for v in HC.V),
        M_gas=sum(float(v.m_phase[1]) for v in HC.V),
        M_liq=sum(float(v.m_phase[0]) for v in HC.V),
        dp_bulk=integrated_phase_pressure_jump(HC, 1, 0),
        dp_all=integrated_phase_pressure_jump(HC, 1, 0, bulk_only=False),
        KE=sum(0.5 * float(v.m) * float(v.u[:dim] @ v.u[:dim]) for v in HC.V),
        u_max=max(float(np.linalg.norm(v.u[:dim])) for v in HC.V),
        n_interface=sum(1 for v in HC.V if getattr(v, 'is_interface', False)),
        n_gas_cells=sum(1 for v in HC.V if v.dual_vol_phase[1] > 1e-30),
        n_vertices=len(list(HC.V)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dim', type=int, default=3)
    ap.add_argument('--ro', type=int, default=1, help='refinement_outer')
    ap.add_argument('--rd', type=int, default=1, help='refinement_droplet')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--g', type=float, default=0.0)
    ap.add_argument('--inject', action='store_true')
    ap.add_argument('--no-clamp', action='store_true')
    ap.add_argument('--box-shift', default='move_all')
    ap.add_argument('--records', type=int, default=10)
    ap.add_argument('--replace', action='append', default=[],
                    metavar='AXIS=VALUE')
    ap.add_argument('--out', default=None, help='directory for the JSON')
    args = ap.parse_args(argv)

    methods = PRESETS[f'electrolysis_bubble_{args.dim}D']
    if args.replace:
        methods = methods.replace(**{k: _coerce(v) for k, v in
                                     (r.split('=', 1) for r in args.replace)})
    print(methods.describe())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = build(
            args.dim, args.ro, args.rd, args.g, methods,
            box_shift=args.box_shift, clamp=not args.no_clamp)
    dim = args.dim
    dt = time_step(HC, dim)
    dm_dt = P.dm_dt_3d if dim == 3 else P.dm_dt_2d
    dp_exact = young_laplace_jump(P.gamma, P.R0, dim=dim)
    V_exact = (4.0 / 3.0) * np.pi * P.R0 ** 3 if dim == 3 else np.pi * P.R0 ** 2
    m0 = measure(HC, dim)
    print(f"setup: {m0['n_vertices']} vertices, {m0['n_interface']} interface, "
          f"dp {m0['dp_bulk']:.6f} (exact {dp_exact}), V/V_exact "
          f"{m0['V_gas'] / V_exact:.6f}; dt {dt:.4e}, {args.steps} steps")
    hist = []
    every = max(1, args.steps // args.records)
    t0 = time.time()

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if args.inject:
            inject_gas_mass(HC_cb, mps, dm_dt=dm_dt, dt=dt, gas_phase=1)
        if (step + 1) % every == 0 or step + 1 == args.steps:
            m = measure(HC, dim)
            m['step'] = step + 1
            m['t'] = float(t)
            hist.append(m)
            print(f"step {step + 1:5d} t={t:.3e} dp={m['dp_bulk']:.3f} "
                  f"V/V_exact={m['V_gas'] / V_exact:.6f} "
                  f"R_if/R0={m['R_interface'] / P.R0:.6f} M_gas="
                  f"{m['M_gas']:.10e} KE={m['KE']:.3e} u_max={m['u_max']:.3e} "
                  f"n_if={m['n_interface']} gas_cells={m['n_gas_cells']}",
                  flush=True)

    status = 'ok'
    try:
        methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=args.steps,
                          bc_set=bc_set, callback=cb, mps=mps)
    except Exception as e:     # noqa: BLE001 - the abort is the result
        status = f'ABORT {type(e).__name__}: {e}'
    mf = measure(HC, dim)
    M_gas_expected = m0['M_gas'] + (dm_dt * dt * args.steps if args.inject
                                    else 0.0)
    out = dict(
        dim=dim, ro=args.ro, rd=args.rd, steps=args.steps, dt=dt, g=args.g,
        inject=args.inject, box_shift=args.box_shift, status=status,
        methods=methods.to_dict(), dp_exact=dp_exact, V_exact=V_exact,
        setup=m0, end=mf, history=hist, digest=digest(HC, dim),
        KE_max=max((h['KE'] for h in hist), default=mf['KE']),
        u_max=max((h['u_max'] for h in hist), default=mf['u_max']),
        M_gas_expected=M_gas_expected,
        M_gas_drift_rel=(mf['M_gas'] - M_gas_expected) / M_gas_expected,
        M_liq_drift_rel=(mf['M_liq'] - m0['M_liq']) / m0['M_liq'],
        wall_s=time.time() - t0,
    )
    print(f"end: {status}; dp {mf['dp_bulk']:.4f} Pa (exact {dp_exact}: "
          f"{mf['dp_bulk'] / dp_exact - 1.0:+.4f} relative), V/V_exact "
          f"{mf['V_gas'] / V_exact:.6f}, gas mass drift "
          f"{out['M_gas_drift_rel']:.2e}, liquid {out['M_liq_drift_rel']:.2e}, "
          f"KE_max {out['KE_max']:.4e}, u_max {out['u_max']:.4e}, "
          f"n_if {mf['n_interface']}, digest {out['digest']}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = (f"{dim}D_{args.ro}{args.rd}_{args.steps}"
               + ('_inject' if args.inject else '') + f"_g{args.g:g}"
               + ''.join('_' + r.replace('=', '-') for r in args.replace))
        path = os.path.join(args.out, f'static_bubble_{tag}.json')
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('written', path)
    return out


if __name__ == '__main__':
    main()
