#!/usr/bin/env python3
"""laneW (2026-10-05): A/B digests of the setups and runners that bind
their force and retopology through ``SolverMethods`` but carry no pinned
number.

Each sub-command runs one short job and prints a digest of the final
state.  Run it once from this tree and once with ``--root`` pointing at
an export of another commit (``git archive <sha> ddgclib cases_dynamic |
tar -x -C DIR`` plus a ``hyperct`` link); equal digests mean the
conversion changed nothing.  The sub-commands detect the old setup /
runner signatures, so the same script runs on a tree from before laneW.

    python cases_dynamic/diagnose_setup_bindings.py setups --out /tmp/new
    python cases_dynamic/diagnose_setup_bindings.py setups --root /tmp/old --out /tmp/old_out
    python cases_dynamic/diagnose_setup_bindings.py fritz noair2d noair3d \
        meshconv adaptive massredist a5step1 --out /tmp/new

Sub-commands
------------
setups      every setup at a small refinement: the force partial behind
            the wrappers (function name, keyword names), the retopology
            binding, and a digest of the acceleration on twelve interior
            vertices; the shearing-plate setup also digests its state
            after the one periodic retopology it applies at setup
fritz       electrolysis_bubble_fritz_2D: the Fritz mesh, its IC and the
            80-step short dynamics (``run_short_dynamics``)
noair2d / noair3d
            setup_dam_break_single_phase + 150 (2D) / 20 (3D) steps with
            the walls frozen through ``boundary_filter``
meshconv    mesh_convergence_2D.run_single(1, 2, n_steps_max=30)
adaptive    oscillating_droplet_2D_adaptive.run_one_mode, both arms
            (Delaunay, adaptive), refinement 2/2, 30 steps
massredist  oscillating_droplet_2D_mass_redist._run_simulation, the three
            arms, refinement 2/2, t_end 1e-3 s (NO_RETOPO is expected to
            differ from a tree before laneW: its closure became
            connectivity='dual_only' without redistribution)
a5step1     diagnose_a5_step1_diff.py (its single retopology call at
            epsilon 0, refinement 2/2) in 2D and 3D: the digest of the
            per-vertex JSON the probe writes (the state before and after
            the call, per interface vertex) and its largest |F| jump

Every sub-command writes under ``--out`` (required: a scratch directory,
never the tree); the digests are printed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))


def _state_digest(HC, dim: int) -> str:
    state = sorted((tuple(float(c) for c in v.x_a[:dim]),
                    tuple(float(c) for c in v.u[:dim]), float(v.m))
                   for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


def _array_digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())
    return h.hexdigest()[:16]


def _force_digest(HC, dudt_fn, n: int = 12) -> str:
    vs = [v for v in HC.V if not v.boundary][:n]
    return hashlib.sha256(np.array([dudt_fn(v) for v in vs]).tobytes()
                          ).hexdigest()[:16]


def _binding(fn) -> str:
    """'func(kw, kw, ...)' of a partial, through the wrappers laneW added
    (``stress_fn``) and the plain closures before it."""
    from functools import partial
    inner = getattr(fn, 'stress_fn', fn)
    if isinstance(inner, partial):
        return f"{inner.func.__name__}({', '.join(sorted(inner.keywords))})"
    return type(inner).__name__


def _accepts(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


# ---------------------------------------------------------------------------
def cmd_setups(out: str) -> None:
    from ddgclib.methods import PRESETS
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    HC, bV, mps, bc, f, r, p = setup_oscillating_droplet(
        dim=2, refinement_outer=1, refinement_droplet=2)
    print('droplet   ', _binding(f), '|', _binding(r), '|', _force_digest(HC, f))

    from cases_dynamic.dam_break.src import _params as D
    from cases_dynamic.dam_break.src._setup import (
        setup_dam_break_multiphase, setup_dam_break_single_phase,
    )
    kw = dict(dim=2, a=D.a, L=D.L, H=D.H, W=D.W, col_w=D.col_w, col_h=D.col_h,
              col_d=D.col_d, rho_l=D.rho_l, rho_g=D.rho_g, mu_l=D.mu_l,
              mu_g=D.mu_g, gamma=D.gamma, K_l=D.K_l, K_g=D.K_g, g=D.g,
              gravity_axis=D.gravity_axis, P_atm=D.P_atm, n_refine=2,
              alpha_art=0.5)
    if _accepts(setup_dam_break_multiphase, 'methods'):
        kw['methods'] = PRESETS['dam_break_2D']
    HC, bV, mps, bc, f, r, p = setup_dam_break_multiphase(**kw)
    print('dam_break ', _binding(f), '|', _binding(r), '|', _force_digest(HC, f))

    kw = dict(dim=2, a=D.a, col_w=D.col_w, col_h=D.col_h, col_d=D.col_d,
              rho_l=D.rho_l, mu_l=D.mu_l, K_l=D.K_l, g=D.g,
              gravity_axis=D.gravity_axis, P_atm=D.P_atm, n_refine=2,
              alpha_art=0.5)
    if _accepts(setup_dam_break_single_phase, 'methods'):
        kw['methods'] = PRESETS['dam_break_2D_no_air']
    HC, bV, bc, f, p = setup_dam_break_single_phase(**kw)
    print('dam_1phase', _binding(f), '|', _force_digest(HC, f))

    from cases_dynamic.electrolysis_bubble.src._setup import (
        setup_electrolysis_bubble,
    )
    kw = dict(dim=2, refinement_outer=1, refinement_droplet=2)
    if _accepts(setup_electrolysis_bubble, 'methods'):
        kw['methods'] = PRESETS['electrolysis_bubble_2D']
    HC, bV, mps, bc, f, r, p = setup_electrolysis_bubble(**kw)
    print('electro   ', _binding(f), '|', _binding(r), '|', _force_digest(HC, f))

    from cases_dynamic.shearing_plate_droplet.src import _params as sp
    from cases_dynamic.shearing_plate_droplet.src._setup import (
        setup_shearing_plate_droplet,
    )
    kw = dict(dim=2, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=sp.U_wall,
              rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
              gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o, refinement_outer=2,
              refinement_droplet=2)
    if _accepts(setup_shearing_plate_droplet, 'methods'):
        kw['methods'] = PRESETS['shearing_plate_droplet_2D']
    HC, bV, mps, bc, f, r, g, p = setup_shearing_plate_droplet(**kw)
    name = getattr(r, '__name__', None) or _binding(r)
    print('shearing  ', _binding(f), '|', name, '|', len(HC.V),
          'setup state', _state_digest(HC, 2), '|', _force_digest(HC, f))


def cmd_fritz(out: str) -> None:
    mod = importlib.import_module(
        'cases_dynamic.electrolysis_bubble.electrolysis_bubble_fritz_2D')
    mod._CASE_DIR = out
    mod._FIG = os.path.join(out, 'fig')
    from cases_dynamic.electrolysis_bubble.src import _params as p
    from cases_dynamic.electrolysis_bubble.src._analytical import bond_number
    Bo0 = bond_number(p.R0, p.gamma, p.rho_liq - p.rho_gas, p.g)
    HC, bV, mps, meta = mod.build_fritz_bubble_in_box_2d(
        R_top=p.R0, L_domain=p.L_domain, Bo=Bo0, electrode_z=-p.L_domain,
        refinement_outer=p.n_refine_outer_2d,
        refinement_droplet=p.n_refine_drop_2d, contact_angle=0.5 * np.pi)
    mod._apply_fritz_ic(HC, mps, meta, electrode_z=-p.L_domain, P0=p.P0,
                        rho_liq=p.rho_liq, rho_gas=p.rho_gas, g=p.g,
                        gamma=p.gamma, R_top=p.R0)
    mod.mass_conserving_merge(HC, cdist=1e-12)
    mps.refresh(HC, dim=2, reset_mass=False, split_method='neighbour_count')
    info = mod.run_short_dynamics(HC, bV, mps, meta, n_steps=80)
    print('fritz', info, len(HC.V), _state_digest(HC, 2))


def _cmd_noair(out: str, dim: int) -> None:
    from cases_dynamic.dam_break.src import _params as D
    from cases_dynamic.dam_break.src._setup import (
        cfl_timestep, setup_dam_break_single_phase,
    )
    kw = dict(dim=dim, a=D.a, col_w=D.col_w, col_h=D.col_h, col_d=D.col_d,
              rho_l=D.rho_l, mu_l=D.mu_l, K_l=D.K_l, g=D.g,
              gravity_axis=D.gravity_axis, P_atm=D.P_atm,
              n_refine=D.n_refine_2d if dim == 2 else D.n_refine_3d,
              alpha_art=D.alpha_art)
    n_steps = 150 if dim == 2 else 20

    def wall(v):
        return bool(getattr(v, 'is_wall', False))

    new = _accepts(setup_dam_break_single_phase, 'methods')
    status = 'ok'
    if new:
        from ddgclib.methods import PRESETS
        m = PRESETS[f'dam_break_{dim}D_no_air']
        HC, bV, bc_set, dudt_fn, params = setup_dam_break_single_phase(
            methods=m, **kw)
        dt = cfl_timestep(HC, dim, float(np.sqrt(D.K_l / D.rho_l)), cfl=D.cfl)
        try:
            m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps, bc_set=bc_set,
                        boundary_filter=wall)
        except Exception as e:  # noqa: BLE001 - an abort is a result here
            status = f'abort {type(e).__name__}'
    else:   # the runner before laneW
        from ddgclib.dynamic_integrators import symplectic_euler
        HC, bV, bc_set, dudt_fn, params = setup_dam_break_single_phase(**kw)
        dt = cfl_timestep(HC, dim, float(np.sqrt(D.K_l / D.rho_l)), cfl=D.cfl)
        try:
            symplectic_euler(HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
                             bc_set=bc_set, boundary_filter=wall)
        except Exception as e:  # noqa: BLE001
            status = f'abort {type(e).__name__}'
    print(f'noair{dim}d', 'preset' if new else 'hand-written', n_steps, status,
          len(HC.V), _state_digest(HC, dim))


def cmd_meshconv(out: str) -> None:
    mod = importlib.import_module(
        'cases_dynamic.oscillating_droplet.mesh_convergence_2D')
    n_verts, t_arr, R = mod.run_single(1, 2, n_steps_max=30)
    print('meshconv', n_verts, len(t_arr), _array_digest(t_arr, R), R[-1])


def cmd_adaptive(out: str) -> None:
    mod = importlib.import_module(
        'cases_dynamic.oscillating_droplet.oscillating_droplet_2D_adaptive')
    from cases_dynamic.oscillating_droplet.src import _params as P
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    HC_tmp = setup_oscillating_droplet(
        dim=2, R0=P.R0, epsilon=P.epsilon, l=P.l, rho_d=P.rho_d, rho_o=P.rho_o,
        mu_d=P.mu_d, mu_o=P.mu_o, gamma=P.gamma, K_d=P.K_d, K_o=P.K_o,
        L_domain=P.L_domain, refinement_outer=2, refinement_droplet=2)[0]
    c_s = np.sqrt(P.K_d / P.rho_d)
    dx_min = min(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) for v in HC_tmp.V
                 for nb in v.nn if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15)
    dt = min(0.25 * dx_min / c_s, 0.5 * np.sqrt(P.rho_d * dx_min**3 / P.gamma))
    akw = {'alpha_min': 0.3, 'alpha_max': 2.5, 'quality_target_deg': 20.0,
           'max_iterations': 1, 'smooth_iterations': 0}
    common = dict(dim=2, dt=dt, n_steps=30, record_every=5,
                  refine_outer=2, refine_droplet=2)
    if _accepts(mod.run_one_mode, 'methods'):
        from ddgclib.methods import PRESETS
        base = PRESETS['oscillating_droplet_2D_bare_delaunay']
        runs = [mod.run_one_mode('delaunay', base, **common),
                mod.run_one_mode('adaptive', base.replace(
                    connectivity='adaptive', remesh_kwargs=akw), **common)]
    else:   # the runner before laneW
        runs = [mod.run_one_mode('delaunay', 'delaunay', None, **common),
                mod.run_one_mode('adaptive', 'adaptive', akw, **common)]
    for r in runs:
        d = r['diags']
        print('adaptive', r['label'], len(d), d[-1]['n_verts'],
              _array_digest([x['R_max'] for x in d], [x['KE'] for x in d]),
              d[-1]['R_max'])


def cmd_massredist(out: str) -> None:
    mod = importlib.import_module(
        'cases_dynamic.oscillating_droplet.oscillating_droplet_2D_mass_redist')
    mod.t_end_2d = 1.0e-3
    mod.n_refine_outer = 2
    mod.n_refine_droplet = 2
    mod._RESULTS = out
    for mode in ('no_redist', 'redist', 'no_retopo'):
        r = mod._run_simulation(mode.upper(), retopo_mode=mode)
        print('massredist', mode, len(r['t']), _array_digest(r['R_max'], r['mass']),
              r['R_max'][-1], r['mass'][-1])


def cmd_a5step1(out: str) -> None:
    import contextlib
    import json
    mod = importlib.import_module(
        'cases_dynamic.oscillating_droplet.diagnose_a5_step1_diff')
    mod._RESULTS = out
    os.makedirs(out, exist_ok=True)
    for dim in (2, 3):
        argv, sys.argv = sys.argv, ['diagnose_a5_step1_diff.py', '--dim', str(dim)]
        try:
            with open(os.path.join(out, f'a5step1_{dim}d.log'), 'w',
                      encoding='utf-8') as fh, contextlib.redirect_stdout(fh):
                mod.main()
        finally:
            sys.argv = argv
        with open(os.path.join(out, f'a5_step1_diff_{dim}d_neighbour_count.json'),
                  'rb') as fh:
            raw = fh.read()
        rec = json.loads(raw)
        print('a5step1', f'{dim}d', rec['n_iface_before'], rec['n_iface_after'],
              len(rec['deltas']), hashlib.sha256(raw).hexdigest()[:16],
              max(d['d_|F|'] for d in rec['deltas']))


COMMANDS = {
    'setups': cmd_setups, 'fritz': cmd_fritz,
    'noair2d': lambda out: _cmd_noair(out, 2),
    'noair3d': lambda out: _cmd_noair(out, 3),
    'meshconv': cmd_meshconv, 'adaptive': cmd_adaptive,
    'massredist': cmd_massredist, 'a5step1': cmd_a5step1,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('what', nargs='+', choices=sorted(COMMANDS))
    ap.add_argument('--root', default=_ROOT,
                    help='tree to import ddgclib / cases_dynamic from '
                         '(default: this one)')
    ap.add_argument('--out', required=True,
                    help='directory for the runners\' files and the probe '
                         'logs (a scratch directory, not the tree)')
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    os.makedirs(args.out, exist_ok=True)
    print(f'tree {root}')
    warnings.simplefilter('ignore')
    for what in args.what:
        COMMANDS[what](os.path.join(args.out, what))
    return 0


if __name__ == '__main__':
    sys.exit(main())
