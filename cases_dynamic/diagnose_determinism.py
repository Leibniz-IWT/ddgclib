#!/usr/bin/env python3
"""Determinism detector for the dynamic runs (lane T, 2026-10-02).

A dynamic run should be a pure function of its inputs: the same final
state in every fresh interpreter and after any other run in the same
interpreter.  This script runs a short case through its preset (or a
``.replace(...)`` arm) and prints a digest of the final state; ``sweep``
repeats that in fresh interpreters and reports how many distinct digests
came out.

Usage (repo root)::

    python cases_dynamic/diagnose_determinism.py list
    python cases_dynamic/diagnose_determinism.py run hp3d_centred --steps 300
    python cases_dynamic/diagnose_determinism.py run hp3d_centred \
        --pre droplet2d,hydro3d_remap          # other runs first, same process
    python cases_dynamic/diagnose_determinism.py run hp3d_centred --scramble 7
    python cases_dynamic/diagnose_determinism.py run hp3d_centred --perturb 1e-15
    python cases_dynamic/diagnose_determinism.py sweep hp3d_centred --procs 8
    python cases_dynamic/diagnose_determinism.py sweep all --procs 4

``--pre A,B``
    Run cases A and B (short) in the same interpreter first.
``--scramble SEED``
    Before anything is built, allocate a few hundred thousand small
    objects and free a random subset.  Later allocations then fill the
    holes, so the memory addresses of the vertex objects are no longer
    monotone in creation order: whatever iterates in ``id()`` order sees
    another order.  This turns a process dependence that shows up in one
    interpreter out of ten into one that shows up in every interpreter.
``--perturb EPS``
    Shift the initial coordinates of the interior vertices (not frozen,
    not on the hull) by ``EPS * scale * r``, ``r`` uniform in [-1, 1]
    (seeded), ``scale`` the largest coordinate of the mesh.
    The spread of the result under ``EPS = 1e-15`` is the round-off
    amplification of the run: the size a change of summation order may
    have, and no more.  (The shift also re-inserts the moved vertices at
    the end of ``HC.V``, i.e. it changes the summation orders as well.)
``--trace FILE``
    Write one digest per step (JSON list), to locate the first step at
    which two runs differ.
``--lib DIR``
    Import ``ddgclib`` and ``hyperct`` from DIR instead of this tree (the
    cases stay the ones of this tree), e.g. an export of an older commit
    of both repositories (``git archive <sha> ddgclib | tar -x -C DIR``,
    ``git -C ../hyperct archive <sha> hyperct | tar -x -C DIR``): the
    same sweep before and after a change.

``sweep`` runs ``--procs`` fresh interpreters: the first plain, the
second after ``--pre`` runs, the others with scramble seeds 1, 2, ...;
with ``--perturb EPS`` it adds ``--n-perturb`` shifted runs, which are
listed but not counted (``--procs 1`` = the plain run and the shifted
ones).  Exit status 1 when the digests differ.

Cases whose default is 0 steps (``pin_*``, ``hydro3d_remap_40tac``) run
the horizon of the pinned test or of the lane log they reproduce, given
in acoustic times; ``--steps`` overrides it.  ``sweep all --procs 6
--jobs 12`` takes about 20 minutes.

Nothing is written unless ``--trace`` or ``--out`` names a file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))


# ----------------------------------------------------------------------
# address scrambling (must run before the meshes are built)
# ----------------------------------------------------------------------
class _Filler:
    """Same allocation size class as a hyperct vertex (a plain instance
    with a ``__dict__``)."""

    def __init__(self, i):
        self.x = (float(i), 0.0, 0.0)
        self.nn = set()


_KEEP: list = []


def scramble(seed: int, n: int = 300_000) -> None:
    """Fragment the small-object heap: later objects get addresses that
    are not monotone in their creation order."""
    rng = random.Random(seed)
    objs: list = [_Filler(i) for i in range(n)]
    arrs: list = [[float(i)] * rng.randint(1, 12) for i in range(n // 4)]
    rng.shuffle(objs)
    rng.shuffle(arrs)
    cut = rng.randint(n // 4, 3 * n // 4)
    _KEEP.append(objs[cut:])            # stays alive: holes stay holes
    _KEEP.append(arrs[len(arrs) // 2:])
    del objs, arrs


# ----------------------------------------------------------------------
# state digest
# ----------------------------------------------------------------------
def state_rows(HC) -> list:
    """Sorted exact (repr) state of every vertex: position, velocity,
    mass, pressure, per-phase masses and pressures, dual volume."""
    rows = []
    for v in HC.V:
        row = [tuple(float(c) for c in v.x_a),
               tuple(float(c) for c in getattr(v, 'u', ())),
               float(getattr(v, 'm', 0.0)),
               ]
        p = getattr(v, 'p', None)
        row.append(None if p is None else float(p))
        for name in ('m_phase', 'p_phase'):
            arr = getattr(v, name, None)
            row.append(None if arr is None else tuple(float(c) for c in arr))
        dv = getattr(v, 'dual_vol', None)
        row.append(None if dv is None else float(dv))
        rows.append(tuple(row))
    rows.sort(key=repr)
    return rows


def digest(HC) -> str:
    return hashlib.sha256(repr(state_rows(HC)).encode()).hexdigest()[:16]


def _perturb(HC, bV, eps: float, seed: int = 0) -> int:
    """Shift the free vertices (not frozen, not on the hull: boundary
    conditions and wall criteria read those coordinates exactly) by
    ``eps * scale * r``.  Returns the number of vertices moved."""
    import numpy as np
    rng = np.random.default_rng(seed)
    verts = list(HC.V)
    scale = max(float(np.max(np.abs(v.x_a))) for v in verts)
    moves = []
    for v in verts:
        r = rng.uniform(-1, 1, len(v.x))
        if v in bV or getattr(v, 'boundary', False):
            continue
        moves.append((v, tuple(np.asarray(v.x_a, dtype=float)
                               + eps * scale * r)))
    HC.V.move_all(moves)
    return len(moves)


class _Tracer:
    """Integrator callback: one digest per step."""

    def __init__(self):
        self.digests: list[str] = []

    def __call__(self, step, t, HC, bV=None, diagnostics=None):
        self.digests.append(digest(HC))


# ----------------------------------------------------------------------
# cases: name -> fn(steps, hook, callback) -> (HC, scalars)
#   hook(HC, bV) is called after the setup and before the first step.
# ----------------------------------------------------------------------
def _hp(dim: int, changes: dict, steps: int, hook, callback, custom=None,
        methods=None, L: float | None = None):
    from ddgclib.methods import PRESETS
    from cases_dynamic.Hagen_Poiseuile.src import _run

    if methods is None:
        methods = PRESETS[f'hagen_poiseuille_{dim}D']
        if changes:
            methods = methods.replace(**changes)
    kw = (dict(dim=2, L=3.0, mu=0.1, n_refine=1, dt=0.02) if dim == 2
          else dict(dim=3, L=2.0, mu=0.1, n_refine=1, dt=0.01))
    if L is not None:
        kw['L'] = L
    setup = _run.setup_poiseuille_developing

    def setup_and_hook(**skw):
        out = setup(**skw)
        hook(out[0], out[1])
        return out

    _run.setup_poiseuille_developing = setup_and_hook
    try:
        r = _run.run_developing(methods, n_steps=steps, callback=callback,
                                custom=custom, **kw)
    finally:
        _run.setup_poiseuille_developing = setup
    return r['HC'], {'l2': r['profile']['l2'], 'u_max': r['profile']['u_max'],
                     'u_cross_max': float(max(r['u_cross'], default=0.0))}


def _hp3d_ring(steps, hook, callback, L: float | None = None):
    """Centred pressure flux on the p_ij ring (no edge-area cache)."""
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.methods import SolverMethods

    def ring(HC, bV, dim, **_kw):
        _retopologize(HC, bV, dim, frozen_set='membership')
        HC._edge_area_cache = None

    methods = SolverMethods(dim=3, connectivity='custom',
                            viscous_flux='simplex_gradient',
                            label='delaunay + membership, cache cleared')
    return _hp(3, {}, steps, hook, callback, custom=ring, methods=methods,
               L=L)


def _column(name: str, n_refine: int, arm: str, steps: int, hook, callback,
            changes: dict | None = None, n_tac: float | None = None):
    """``steps`` steps of the column; with ``steps == 0`` the horizon is
    ``n_tac`` acoustic times, exactly as the pinned test passes it."""
    import numpy as np
    from ddgclib.methods import PRESETS
    from cases_dynamic.Hydrostatic_column.src._column import (
        CASES, build_column, remap_arm, run_column,
    )
    kw = CASES[name]
    col = build_column(kw['dim'], n_refine, H=kw['H'],
                       side_walls=kw['side_walls'], ic='drop')
    hook(col.HC, col.bV)
    methods = PRESETS[name] if arm == 'preset' else remap_arm(PRESETS[name])
    if changes:
        methods = methods.replace(**changes)
    p = col.params
    dt = 0.25 * p['dx_min'] / p['c0']
    res = run_column(col, methods, callback=callback,
                     n_tac=steps * dt / p['t_ac'] if steps else n_tac)
    tail = max(1, len(res['umax']) // 10)        # 36 .. 40 of 40 t_ac
    return col.HC, {'umax_peak': float(np.max(res['umax'])),
                    'umax_tail': float(np.max(res['umax'][-tail:])),
                    'ke_end': float(res['ke'][-1]),
                    'n_steps': int(res['n_steps'])}


def _droplet(dim: int, preset: str, refine: tuple, steps: int, hook, callback,
             changes: dict | None = None):
    import numpy as np
    from ddgclib.methods import PRESETS
    from cases_dynamic.oscillating_droplet.src import _params as P
    from cases_dynamic.oscillating_droplet.src._plot_helpers import (
        compute_diagnostics,
    )
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    methods = PRESETS[preset]
    if changes:
        methods = methods.replace(**changes)
    HC, bV, mps, bc_set, dudt_fn, _fn, params = setup_oscillating_droplet(
        dim=dim, R0=P.R0, epsilon=P.epsilon, l=P.l, rho_d=P.rho_d,
        rho_o=P.rho_o, mu_d=P.mu_d, mu_o=P.mu_o, gamma=P.gamma, K_d=P.K_d,
        K_o=P.K_o, L_domain=P.L_domain, refinement_outer=refine[0],
        refinement_droplet=refine[1], split_method=methods.split_method,
        redistribute_mass=methods.redistribute_mass)
    hook(HC, bV)
    c_s = np.sqrt(P.K_d / P.rho_d)
    dx_min = min(d for d in (np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])
                             for v in HC.V for nb in v.nn) if d > 1e-15)
    dt = min(0.25 * dx_min / c_s,
             0.5 * np.sqrt(P.rho_d * dx_min ** 3 / P.gamma))
    methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=steps, bc_set=bc_set,
                      callback=callback, mps=mps)
    d = compute_diagnostics(HC, dim=dim)
    return HC, {'R_max': float(d['R_max']), 'KE': float(d['KE']),
                'mass': float(d['total_mass'])}


def _dam_break(dim: int, steps, hook, callback):
    import numpy as np
    from ddgclib.methods import PRESETS
    from cases_dynamic.dam_break.src import _params as P
    from cases_dynamic.dam_break.src._setup import (
        setup_dam_break_multiphase, cfl_timestep,
    )
    methods = PRESETS[f'dam_break_{dim}D']
    HC, bV, mps, bc_set, dudt_fn, _fn, params = setup_dam_break_multiphase(
        dim=dim, a=P.a, L=P.L, H=P.H, W=P.W, col_w=P.col_w, col_h=P.col_h,
        col_d=P.col_d, rho_l=P.rho_l, rho_g=P.rho_g, mu_l=P.mu_l, mu_g=P.mu_g,
        gamma=P.gamma, K_l=P.K_l, K_g=P.K_g, g=P.g,
        gravity_axis=P.gravity_axis, P_atm=P.P_atm,
        n_refine=3 if dim == 2 else P.n_refine_3d,
        alpha_art=P.alpha_art, redistribute_mass=methods.redistribute_mass)
    hook(HC, bV)
    dt = cfl_timestep(HC, dim, float(np.sqrt(P.K_l / P.rho_l)), cfl=P.cfl)
    methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=steps, bc_set=bc_set,
                      callback=callback, mps=mps)
    ke = sum(0.5 * float(v.m) * float(np.dot(v.u[:dim], v.u[:dim]))
             for v in HC.V)
    return HC, {'KE': ke}


def _electrolysis(dim: int, steps, hook, callback):
    import numpy as np
    from ddgclib.methods import PRESETS
    from cases_dynamic.electrolysis_bubble.src._setup import (
        setup_electrolysis_bubble,
    )
    methods = PRESETS[f'electrolysis_bubble_{dim}D']
    HC, bV, mps, bc_set, dudt_fn, _fn, params = setup_electrolysis_bubble(
        dim=dim, refinement_outer=1, refinement_droplet=2,
        redistribute_mass=methods.redistribute_mass)
    hook(HC, bV)
    methods.integrate(HC, bV, dudt_fn, dt=1e-7, n_steps=steps, bc_set=bc_set,
                      callback=callback, mps=mps)
    ke = sum(0.5 * float(v.m) * float(np.dot(v.u[:dim], v.u[:dim]))
             for v in HC.V)
    return HC, {'KE': ke}


def _shearing(dim: int, steps, hook, callback):
    """Periodic multiphase retopology (2D: the lane L probe; 3D: the
    setup of ``_run_short_3D.py``)."""
    import numpy as np
    from ddgclib.methods import PRESETS
    from cases_dynamic.shearing_plate_droplet.src import _params as sp
    from cases_dynamic.shearing_plate_droplet.src._setup import (
        setup_shearing_plate_droplet,
    )
    m = PRESETS[f'shearing_plate_droplet_{dim}D']
    kw = (dict(refinement_outer=3, refinement_droplet=3) if dim == 2
          else dict(L_z=sp.L_z, refinement_outer=1, refinement_droplet=2))
    (HC, bV, mps, bc_set, dudt_fn, _fn, _groups,
     params) = setup_shearing_plate_droplet(
        dim=dim, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=sp.U_wall,
        rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
        gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
        redistribute_mass=m.redistribute_mass, **kw)
    hook(HC, bV)
    m.integrate(HC, bV, dudt_fn, dt=1e-5 if dim == 2 else 1e-6,
                n_steps=steps, bc_set=bc_set, mps=mps, callback=callback,
                domain_bounds=params['domain_bounds'])
    ke = sum(0.5 * float(v.m) * float(np.dot(v.u[:dim], v.u[:dim]))
             for v in HC.V)
    return HC, {'KE': ke}


# name -> (default steps, function(steps, hook, callback))
CASES: dict[str, tuple[int, Callable]] = {
    # --- 3D, the arms rule 8 was written for ---
    'hp3d_centred': (300, lambda n, h, c: _hp(
        3, dict(pressure_flux='centred'), n, h, c)),
    'hp3d_ring': (150, _hp3d_ring),
    # the two arms of diagnose_poiseuille.py arms3d (L 3, 600 steps) whose
    # late l2 took two values over 9 processes (lane H log, section 5.3)
    'hp3d_centred_laneH': (600, lambda n, h, c: _hp(
        3, dict(pressure_flux='centred'), n, h, c, L=3.0)),
    'hp3d_ring_laneH': (600, lambda n, h, c: _hp3d_ring(n, h, c, L=3.0)),
    'hp3d': (300, lambda n, h, c: _hp(3, {}, n, h, c)),
    'hydro3d_remap': (150, lambda n, h, c: _column(
        'hydrostatic_3D', 2, 'remap', n, h, c)),
    'hydro3d': (150, lambda n, h, c: _column(
        'hydrostatic_3D', 2, 'preset', n, h, c)),
    # refinement 1 (35 vertices): the short case of test_determinism.py
    'hydro3d_remap_r1': (40, lambda n, h, c: _column(
        'hydrostatic_3D', 1, 'remap', n, h, c)),
    'droplet3d_delaunay': (40, lambda n, h, c: _droplet(
        3, 'oscillating_droplet_3D_delaunay', (2, 2), n, h, c)),
    'droplet3d_remap': (40, lambda n, h, c: _droplet(
        3, 'oscillating_droplet_3D_delaunay', (2, 2), n, h, c,
        changes=dict(remap='conservative'))),
    'droplet3d': (40, lambda n, h, c: _droplet(
        3, 'oscillating_droplet_3D', (2, 2), n, h, c)),
    # --- 1D and 2D ---
    'hydro1d': (300, lambda n, h, c: _column(
        'hydrostatic_1D', 3, 'preset', n, h, c)),
    'hp2d': (300, lambda n, h, c: _hp(2, {}, n, h, c)),
    'hp2d_centred_twopoint': (300, lambda n, h, c: _hp(
        2, dict(viscous_flux='two_point'), n, h, c)),
    'hydro2d_remap': (300, lambda n, h, c: _column(
        'hydrostatic_2D', 2, 'remap', n, h, c)),
    'droplet2d': (100, lambda n, h, c: _droplet(
        2, 'oscillating_droplet_2D', (2, 2), n, h, c)),
    'droplet2d_bare': (100, lambda n, h, c: _droplet(
        2, 'oscillating_droplet_2D_bare_delaunay', (2, 2), n, h, c)),
    'droplet2d_adaptive': (60, lambda n, h, c: _droplet(
        2, 'oscillating_droplet_2D_bare_delaunay', (2, 2), n, h, c,
        # kwargs of oscillating_droplet_2D_adaptive.py
        changes=dict(connectivity='adaptive', remesh_kwargs=dict(
            alpha_min=0.3, alpha_max=2.5, quality_target_deg=20.0,
            max_iterations=1, smooth_iterations=0)))),
    'dam_break2d': (200, lambda n, h, c: _dam_break(2, n, h, c)),
    'hydro2d_density_diffusion': (300, lambda n, h, c: _column(
        'hydrostatic_2D', 2, 'preset', n, h, c,
        changes=dict(density_diffusion=0.05))),
    'electrolysis2d': (100, lambda n, h, c: _electrolysis(2, n, h, c)),
    'shearing2d': (10, lambda n, h, c: _shearing(2, n, h, c)),
    # --- 3D, further presets ---
    'dam_break3d': (5, lambda n, h, c: _dam_break(3, n, h, c)),  # smoke only
    'electrolysis3d': (30, lambda n, h, c: _electrolysis(3, n, h, c)),
    'shearing3d': (5, lambda n, h, c: _shearing(3, n, h, c)),
    # --- the remaining connectivity values and the execution axes ---
    'droplet2d_dual_only': (100, lambda n, h, c: _droplet(
        2, 'oscillating_droplet_2D_dual_only', (2, 2), n, h, c)),
    'droplet3d_frozen': (40, lambda n, h, c: _droplet(
        3, 'oscillating_droplet_3D', (2, 2), n, h, c,
        changes=dict(connectivity='frozen'))),
    'droplet2d_workers': (40, lambda n, h, c: _droplet(
        2, 'oscillating_droplet_2D', (2, 2), n, h, c,
        changes=dict(workers=2))),
    'hp3d_centred_mp': (100, lambda n, h, c: _hp(
        3, dict(pressure_flux='centred', backend='multiprocessing'),
        n, h, c)),
    # --- the pinned runs of ddgclib/tests/test_case_hydrostatic.py
    #     (TestColumn3D), at their own horizon: for --perturb ---
    'pin_hydro3d': (0, lambda n, h, c: _column(
        'hydrostatic_3D', 1, 'preset', n, h, c, n_tac=40.0)),
    'pin_hydro3d_remap': (0, lambda n, h, c: _column(
        'hydrostatic_3D', 2, 'remap', n, h, c, n_tac=2.0)),
    # the run of the lane P log, section 11 (envelope of max|u| over the
    # last 4 of 40 acoustic times: 1.718e-03 or 1.728e-03); 6 minutes
    'hydro3d_remap_40tac': (0, lambda n, h, c: _column(
        'hydrostatic_3D', 2, 'remap', n, h, c, n_tac=40.0)),
}

# Short "unrelated" runs for --pre (steps kept small).
_PRE_STEPS = {'droplet2d': 20, 'hp3d': 20, 'hp3d_centred': 20,
              'hydro3d_remap': 20, 'droplet3d_delaunay': 5, 'hp2d': 50,
              'hydro2d_remap': 50, 'droplet3d': 5, 'dam_break2d': 20}


def run_case(name: str, steps: int | None = None, perturb: float = 0.0,
             perturb_seed: int = 0, trace: bool = False) -> dict[str, Any]:
    """Run one case; return ``{'case', 'steps', 'digest', 'scalars',
    'seconds'[, 'trace']}``."""
    default_steps, fn = CASES[name]
    steps = default_steps if steps is None else steps

    def hook(HC, bV):
        if perturb:
            _perturb(HC, bV, perturb, perturb_seed)

    tracer = _Tracer() if trace else None
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        HC, scalars = fn(steps, hook, tracer)
    out = {'case': name, 'steps': steps, 'digest': digest(HC),
           'scalars': scalars, 'seconds': round(time.perf_counter() - t0, 2)}
    if tracer is not None:
        out['trace'] = tracer.digests
    return out


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def _cmd_run(args) -> int:
    if args.scramble:
        scramble(args.scramble)
    sys.path.insert(0, _ROOT)
    if args.lib:
        # another tree's library with this tree's cases (before / after)
        sys.path.insert(0, os.path.abspath(args.lib))
    # imported here, not inside the timed call: 'seconds' is then
    # comparable between this tree and --lib
    import ddgclib
    import hyperct
    if args.lib:
        for mod in (ddgclib, hyperct):
            assert mod.__file__.startswith(os.path.abspath(args.lib)), mod
    for pre in filter(None, (args.pre or '').split(',')):
        r = run_case(pre, _PRE_STEPS.get(pre, 10))
        print(f"PRE {pre} {r['digest']}", flush=True)
    r = run_case(args.case, args.steps, perturb=args.perturb,
                 perturb_seed=args.perturb_seed, trace=bool(args.trace))
    if args.trace:
        with open(args.trace, 'w') as fh:
            json.dump(r.pop('trace'), fh)
    print('RESULT ' + json.dumps(r), flush=True)
    return 0


def spawn(case: str, steps: int | None = None, pre: str = '',
          scramble_seed: int = 0, perturb: float = 0.0, perturb_seed: int = 0,
          trace: str = '', lib: str = '',
          timeout: float = 3600.0) -> dict[str, Any]:
    """Run ``run`` in a fresh interpreter and return its RESULT dict."""
    cmd = [sys.executable, os.path.abspath(__file__), 'run', case]
    if steps is not None:
        cmd += ['--steps', str(steps)]
    if lib:
        cmd += ['--lib', lib]
    if pre:
        cmd += ['--pre', pre]
    if scramble_seed:
        cmd += ['--scramble', str(scramble_seed)]
    if perturb:
        cmd += ['--perturb', repr(perturb), '--perturb-seed',
                str(perturb_seed)]
    if trace:
        cmd += ['--trace', trace]
    out = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                         timeout=timeout)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith('RESULT ')]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr[-3000:]}")
    res = json.loads(lines[-1][len('RESULT '):])
    res['variant'] = (f"pre={pre}" if pre else
                      f"scramble={scramble_seed}" if scramble_seed else
                      f"perturb={perturb:g}/{perturb_seed}" if perturb else
                      'plain')
    return res


def _cmd_sweep(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    names = list(CASES) if args.case == 'all' else args.case.split(',')
    status = 0
    summary = []
    for name in names:
        variants: list[dict] = [dict()]
        if args.procs > 1:
            variants.append(dict(pre=args.pre))
        variants += [dict(scramble_seed=k)
                     for k in range(1, max(args.procs - 1, 1))]
        if args.perturb:
            variants += [dict(perturb=args.perturb, perturb_seed=k)
                         for k in range(args.n_perturb)]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(
                lambda kw: spawn(name, args.steps, lib=args.lib, **kw),
                variants))
        exact = [r for r in rows if not r['variant'].startswith('perturb')]
        digests = sorted({r['digest'] for r in exact})
        print(f"\n== {name}: {rows[0]['steps']} steps, {len(exact)} "
              f"interpreters, {len(digests)} distinct digest(s)")
        for r in rows:
            sc = ', '.join(f"{k}={v!r}" for k, v in r['scalars'].items())
            print(f"  {r['variant']:<16} {r['digest']}  {r['seconds']:7.1f} s"
                  f"  {sc}")
        if len(digests) > 1:
            status = 1
        summary.append({'case': name, 'steps': rows[0]['steps'],
                        'n_interpreters': len(exact),
                        'n_digests': len(digests), 'rows': rows})
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(summary, fh, indent=1)
            fh.write('\n')
        print(f"-> {args.out}")
    print('\nDETERMINISTIC' if status == 0 else '\nNOT DETERMINISTIC')
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    r = sub.add_parser('run')
    r.add_argument('case', choices=sorted(CASES))
    r.add_argument('--steps', type=int)
    r.add_argument('--pre', default='')
    r.add_argument('--scramble', type=int, default=0)
    r.add_argument('--perturb', type=float, default=0.0)
    r.add_argument('--perturb-seed', type=int, default=0)
    r.add_argument('--trace', default='')
    r.add_argument('--lib', default='')
    s = sub.add_parser('sweep')
    s.add_argument('case')
    s.add_argument('--steps', type=int)
    s.add_argument('--lib', default='')
    s.add_argument('--procs', type=int, default=6)
    s.add_argument('--jobs', type=int, default=4)
    s.add_argument('--pre', default='droplet2d,hp3d_centred')
    s.add_argument('--perturb', type=float, default=0.0)
    s.add_argument('--n-perturb', type=int, default=4)
    s.add_argument('--out', default='')
    args = ap.parse_args(argv)
    if args.cmd == 'list':
        for name, (steps, _fn) in CASES.items():
            print(f"{name:<24} {steps} steps")
        return 0
    return _cmd_run(args) if args.cmd == 'run' else _cmd_sweep(args)


if __name__ == '__main__':
    sys.exit(main())
