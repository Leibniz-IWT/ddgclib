#!/usr/bin/env python3
"""Shearing-plate droplet through its preset, with the integrated checks
of laneG (2026-10-06): the Laplace jump against gamma (dim - 1) / R
(``ddgclib.analytical.integrated_phase_pressure_jump``), the droplet
volume, the interface vertex count, the deformation parameter, the sum
of the forces on the free vertices (a quiescent droplet, ``--U 0``, must
give round-off: no spurious seam force), the largest free-vertex speed
and where it is, and the final-state digest.

Every arm is ``PRESETS['shearing_plate_droplet_<dim>D']`` or
``preset.replace(...)`` (``--replace axis=value``, repeatable).

Usage (from the repository root)::

    python cases_dynamic/shearing_plate_droplet/diagnose_short_window.py
    python ... --U 0 --steps 200                  # quiescent droplet
    python ... --replace remap=conservative       # an arm
    python ... --dim 3 --ro 1 --rd 2 --steps 5    # the 3D setup + steps
    python ... --out /tmp/some/dir                # JSON summary there

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

from cases_dynamic.shearing_plate_droplet.src import _params as sp  # noqa: E402
from cases_dynamic.shearing_plate_droplet.src._analytical import (  # noqa: E402
    taylor_deformation,
)
from cases_dynamic.shearing_plate_droplet.src._plot_helpers import (  # noqa: E402
    compute_diagnostics,
)
from cases_dynamic.shearing_plate_droplet.src._setup import (  # noqa: E402
    setup_shearing_plate_droplet,
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


def build(dim, ro, rd, U, methods):
    kw = dict(dim=dim, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=U,
              rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
              gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
              refinement_outer=ro, refinement_droplet=rd, methods=methods)
    if dim == 3:
        kw['L_z'] = sp.L_z
    return setup_shearing_plate_droplet(**kw)


def time_step(HC, dim):
    c_s = float(np.sqrt(sp.K_o / sp.rho_o))
    dx_min = min(float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
                 for v in HC.V for nb in v.nn
                 if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    return 0.1 * dx_min / c_s


def is_plate(v):
    return abs(abs(v.x_a[1]) - sp.L_y) < 1e-9


def digest(HC, dim):
    state = sorted((tuple(v.x_a[:dim]), tuple(v.u[:dim]), float(v.m),
                    tuple(float(p) for p in v.p_phase)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


def measure(HC, dudt_fn, dim):
    """The integrated checks at the current state."""
    d = compute_diagnostics(HC, dim=dim)
    F = np.zeros(dim)
    F_total = np.zeros(dim)     # every vertex, the plates' half cells too
    f_max = 0.0
    u_free = 0.0
    at = None
    for v in HC.V:
        f = float(v.m) * dudt_fn(v)[:dim]
        F_total += f
        if is_plate(v):
            continue
        F += f
        f_max = max(f_max, float(np.linalg.norm(f)))
        u = float(np.linalg.norm(v.u[:dim]))
        if u > u_free:
            u_free = u
            at = [float(x) for x in v.x_a[:dim]] + [
                int(v.phase), bool(getattr(v, 'is_interface', False)),
                float(v.p), float(v.dual_vol)]
    return dict(
        KE=float(d['KE']), D=float(d['D']), n_interface=int(d['n_interface']),
        V_droplet=sum(float(v.dual_vol_phase[1]) for v in HC.V),
        M_droplet=sum(float(v.m_phase[1]) for v in HC.V),
        M_outer=sum(float(v.m_phase[0]) for v in HC.V),
        dp_bulk=integrated_phase_pressure_jump(HC, 1, 0),
        dp_all=integrated_phase_pressure_jump(HC, 1, 0, bulk_only=False),
        F_free=F.tolist(), F_total=F_total.tolist(), f_max=f_max,
        u_free=u_free, u_free_at=at,
        n_vertices=len(list(HC.V)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--ro', type=int, default=3, help='refinement_outer')
    ap.add_argument('--rd', type=int, default=3, help='refinement_droplet')
    ap.add_argument('--U', type=float, default=sp.U_wall)
    ap.add_argument('--t-end', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=None,
                    help='override the step count of --t-end')
    ap.add_argument('--records', type=int, default=20)
    ap.add_argument('--replace', action='append', default=[],
                    metavar='AXIS=VALUE')
    ap.add_argument('--out', default=None, help='directory for the JSON')
    args = ap.parse_args(argv)

    methods = PRESETS[f'shearing_plate_droplet_{args.dim}D']
    if args.replace:
        methods = methods.replace(**{k: _coerce(v) for k, v in
                                     (r.split('=', 1) for r in args.replace)})
    print(methods.describe())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, groups, params = build(
            args.dim, args.ro, args.rd, args.U, methods)
    dt = time_step(HC, args.dim)
    n_steps = args.steps if args.steps is not None else int(args.t_end / dt) + 1
    dp_exact = sp.gamma * (args.dim - 1) / sp.R0
    V_exact = np.pi * sp.R0 ** 2 if args.dim == 2 else 4.0 / 3.0 * np.pi * sp.R0 ** 3
    m0 = measure(HC, dudt_fn, args.dim)
    print(f"setup: {m0['n_vertices']} vertices, {m0['n_interface']} interface, "
          f"{len(bV)} plate vertices, dp {m0['dp_bulk']:.6f} (exact {dp_exact}), "
          f"V/V_exact {m0['V_droplet'] / V_exact:.6f}, |F_free| "
          f"{np.linalg.norm(m0['F_free']):.3e}, |F_total| "
          f"{np.linalg.norm(m0['F_total']):.3e} (f_max {m0['f_max']:.3e}); "
          f"dt {dt:.4e}, {n_steps} steps")
    hist = []
    every = max(1, n_steps // args.records)
    t0 = time.time()

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if (step + 1) % every == 0 or step + 1 == n_steps:
            m = measure(HC, dudt_fn, args.dim)
            m['step'] = step + 1
            m['t'] = float(t)
            hist.append(m)
            print(f"step {step + 1:5d} t={t:.4e} D={m['D']:.4f} "
                  f"n_if={m['n_interface']} V/V_exact="
                  f"{m['V_droplet'] / V_exact:.6f} dp={m['dp_bulk']:.3f} "
                  f"KE={m['KE']:.3e} u_free/U={m['u_free'] / max(args.U, 1e-30):.3f} "
                  f"|F_free|={np.linalg.norm(m['F_free']):.3e} "
                  f"|F_total|={np.linalg.norm(m['F_total']):.3e} at "
                  f"{m['u_free_at']}", flush=True)

    status = 'ok'
    try:
        methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                          bc_set=bc_set, callback=cb, mps=mps,
                          domain_bounds=params['domain_bounds'])
    except Exception as e:     # noqa: BLE001 - the abort is the result
        status = f'ABORT {type(e).__name__}: {e}'
    mf = measure(HC, dudt_fn, args.dim)
    out = dict(
        dim=args.dim, ro=args.ro, rd=args.rd, U=args.U, dt=dt, n_steps=n_steps,
        status=status, methods=methods.to_dict(), dp_exact=dp_exact,
        V_exact=V_exact, D_taylor=taylor_deformation(sp.Ca, sp.visc_ratio),
        setup=m0, end=mf, history=hist, digest=digest(HC, args.dim),
        KE_max=max((h['KE'] for h in hist), default=mf['KE']),
        u_free_max=max((h['u_free'] for h in hist), default=mf['u_free']),
        n_interface_min=min((h['n_interface'] for h in hist),
                            default=mf['n_interface']),
        wall_s=time.time() - t0,
    )
    print(f"end: {status}; dp {mf['dp_bulk']:.4f} Pa (exact {dp_exact}), "
          f"V/V_exact {mf['V_droplet'] / V_exact:.6f}, n_if {mf['n_interface']}, "
          f"D {mf['D']:.5f}, KE_max {out['KE_max']:.4e}, u_free_max "
          f"{out['u_free_max']:.4e}, digest {out['digest']}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"{args.dim}D_{args.ro}{args.rd}_U{args.U:g}" + ''.join(
            '_' + r.replace('=', '-') for r in args.replace)
        path = os.path.join(args.out, f'short_window_{tag}.json')
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('written', path)
    return out


if __name__ == '__main__':
    main()
