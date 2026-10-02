#!/usr/bin/env python3
"""Measurements behind the developing Poiseuille presets (laneH).

Every dynamic arm is ``PRESETS['hagen_poiseuille_2D' / '_3D']`` or a
``.replace(...)`` of it, run through ``src/_run.py:run_developing``; the
one exception is the 3D ``p_ij`` arm, which needs the sanctioned
``connectivity='custom'`` wrapper that clears the edge-area cache
(METHODS.md, axis ``edge_area_source``).

Usage (repo root)::

    python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py static
    python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms2d
    python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d
    python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py convergence
    python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py slivers

``static``
    Force residual of the exact Poiseuille field on meshes whose interior
    vertices were jittered and sheared and that were reconnected by the
    library Delaunay retopology: each flux against ``G Vol``.  For the
    viscous fluxes also the residual of a LINEAR velocity field with the
    wall shear rate of the Poiseuille profile (it should be zero); that
    is the discriminating number.  The nodal residual of the quadratic
    field is not: the simplex form is the Galerkin P1 discretisation,
    whose nodal truncation error on an irregular mesh is O(1) of
    ``G Vol`` while its solution error converges (``convergence``).
``arms2d`` / ``arms3d``
    The preset against its arms: profile error, transverse velocity,
    vertices outside the walls.  The two ``pressure_flux='centred'`` arms
    of ``arms3d`` are not reproducible from process to process (the
    preset and two-point arms are, to the bit): quote them as a range
    over at least two processes, ``arms3d --only centred --tag p2``.
``convergence``
    2D profile error against refinement, at a fixed time and at the
    steady state of the mesh.
``slivers``
    Simplices that the simplex-gradient viscous flux leaves out
    (``|T| <= 1e-3 l_min**dim``) during a run, and how many of them
    touch a free vertex inside the channel that is not on the hull (a
    simplex left out there breaks linear precision at its vertices; on
    the hull it only changes how the boundary is triangulated).

Output: a table on stdout and ``results/laneH/<mode>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from ddgclib.methods import PRESETS, SolverMethods  # noqa: E402
from cases_dynamic.Hagen_Poiseuile.src._run import run_developing  # noqa: E402

_OUT = os.path.join(_HERE, 'results', 'laneH')


def _save(mode: str, rows, tag: str = '') -> None:
    os.makedirs(_OUT, exist_ok=True)
    name = f'{mode}_{tag}.json' if tag else f'{mode}.json'
    with open(os.path.join(_OUT, name), 'w') as fh:
        json.dump(rows, fh, indent=2)
        fh.write('\n')
    print(f"-> {os.path.join(_OUT, name)}")


# ----------------------------------------------------------------------
# static: consistency of the fluxes with the exact Poiseuille field
# ----------------------------------------------------------------------
def static(args) -> None:
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
    from ddgclib.geometry.domains import cylinder_volume, rectangle
    from ddgclib.operators.stress import (
        dual_area_vector, pressure_force_simplex_gradient, stress_force,
        viscous_force_simplex_gradient,
    )

    mu, rows = 0.1, []
    for dim, refinement, jitter, shear in (
            (2, 2, 0.0, 0.0), (2, 3, 0.2, 0.3), (2, 4, 0.2, 0.3),
            (3, 2, 0.0, 0.0), (3, 2, 0.15, 0.0)):
        rng = np.random.default_rng(0)
        if dim == 2:
            L, D, G = 2.0, 1.0, 12 * mu * 0.1
            res = rectangle(L=L, h=D, refinement=refinement, flow_axis=0)
            axis = 0

            def u_exact(x):
                return np.array([G / (2 * mu) * x[1] * (D - x[1]), 0.0])
        else:
            L, R, G = 3.0, 0.5, 8 * mu * 0.1 / 0.25
            res = cylinder_volume(R=R, L=L, refinement=refinement, flow_axis=2)
            axis = 2

            def u_exact(x):
                return np.array([0.0, 0.0, G / (4 * mu) * max(
                    R**2 - x[0]**2 - x[1]**2, 0.0)])
        HC, bV = res.HC, res.bV
        n = sum(1 for _ in HC.V)
        h = (res.metadata['volume'] / n) ** (1.0 / dim)
        moves = []
        for v in list(HC.V):
            if v in bV:
                continue
            x = v.x_a + jitter * h * rng.uniform(-1, 1, dim)
            if dim == 2:
                x[0] += shear * 4 * v.x_a[1] * (1.0 - v.x_a[1])
            moves.append((v, tuple(x)))
        HC.V.move_all(moves)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _retopologize(HC, bV, dim)
        for v in HC.V:
            v.u = u_exact(v.x_a)
            v.p = -G * v.x_a[axis]
            v.m = 1.0
        e = np.zeros(dim)
        e[axis] = 1.0
        cache = HC._edge_area_cache
        r = {k: [] for k in ('p_centred', 'p_ring', 'p_simplex', 'v_two_point',
                             'v_simplex', 'v_two_point_linear',
                             'v_simplex_linear')}
        shear_rate = G * (D if dim == 2 else R) / (2 * mu)   # at the wall
        measured = []
        for v in HC.V:
            if v.boundary or any(nb.boundary for nb in v.nn):
                continue
            if not 0.6 < v.x_a[axis] < L - 0.6 + (shear if dim == 2 else 0.0):
                continue
            ref = G * v.dual_vol
            r['p_centred'].append(np.linalg.norm(
                stress_force(v, dim=dim, mu=0.0, HC=HC) - ref * e) / ref)
            if dim == 3:        # the p_ij ring, read without the cache
                HC._edge_area_cache = None
                F = np.zeros(dim)
                for nb in v.nn:
                    F += -0.5 * (v.p + nb.p) * dual_area_vector(v, nb, HC, dim)
                HC._edge_area_cache = cache
                r['p_ring'].append(np.linalg.norm(F - ref * e) / ref)
            r['p_simplex'].append(np.linalg.norm(
                pressure_force_simplex_gradient(v, HC, dim) - ref * e) / ref)
            visc = stress_force(v, dim=dim, mu=mu, HC=HC) - stress_force(
                v, dim=dim, mu=0.0, HC=HC)
            r['v_two_point'].append(np.linalg.norm(visc + ref * e) / ref)
            r['v_simplex'].append(np.linalg.norm(
                viscous_force_simplex_gradient(v, HC, dim, mu) + ref * e) / ref)
            measured.append(v)
        for v in HC.V:
            v.u = shear_rate * v.x_a[1 if dim == 2 else 0] * e
        for v in measured:
            ref = G * v.dual_vol
            r['v_two_point_linear'].append(np.linalg.norm(
                stress_force(v, dim=dim, mu=mu, HC=HC)
                - stress_force(v, dim=dim, mu=0.0, HC=HC)) / ref)
            r['v_simplex_linear'].append(np.linalg.norm(
                viscous_force_simplex_gradient(v, HC, dim, mu)) / ref)
        row = dict(dim=dim, refinement=refinement, jitter=jitter, shear=shear,
                   n_vertices=n, n_measured=len(r['p_simplex']))
        for k, vals in r.items():
            if vals:
                row[k] = dict(median=float(np.median(vals)),
                              max=float(np.max(vals)))
        rows.append(row)
        print(f"{dim}D refinement {refinement} jitter {jitter} shear {shear}: "
              f"{row['n_measured']} interior vertices; residual / (G Vol), "
              "median / max")
        for k in r:
            if k in row:
                print(f"    {k:<18} {row[k]['median']:.3e} / {row[k]['max']:.3e}")
    _save('static', rows)


# ----------------------------------------------------------------------
# dynamic arms
# ----------------------------------------------------------------------
def _row(label: str, methods, r: dict, wall_s: float) -> dict:
    tail = slice(len(r['t']) * 3 // 4, None)
    row = dict(
        arm=label, methods=methods.to_dict(), n_steps=r['n_steps'],
        dt=r['dt'], t_end=r['t_end'], window=list(r['window']),
        l2_end=r['profile']['l2'], l2_tail_mean=float(np.mean(r['l2'][tail])),
        l2_tail_max=float(np.max(r['l2'][tail])),
        u_max_end=r['profile']['u_max'],
        u_max_analytical=r['params']['U_max'],
        u_cross_max=float(np.max(r['u_cross'])),
        n_outside_max=int(np.max(r['n_outside'])),
        n_channel=[int(np.min(r['n_channel'])), int(np.max(r['n_channel']))],
        mass_flux_in=r['mass_flux_in'], mass_flux_out=r['mass_flux_out'],
        flux_periods=r['flux_periods'],
        volume_flux_window=float(np.mean(r['q_window'][tail])),
        walls=r['walls'], wall_s=round(wall_s, 1))
    print(f"  {label:<34} l2 end {row['l2_end']:.4e}  tail mean "
          f"{row['l2_tail_mean']:.4e} max {row['l2_tail_max']:.4e}  u_max "
          f"{row['u_max_end']:.5f}  u_cross {row['u_cross_max']:.1e}  outside "
          f"{row['n_outside_max']}  {wall_s:.0f} s", flush=True)
    return row


def _run(label, methods, n_steps, **kw):
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = run_developing(methods, n_steps=n_steps, **kw)
    return _row(label, methods, r, time.time() - t0)


def arms2d(args) -> None:
    preset = PRESETS['hagen_poiseuille_2D']
    kw = dict(dim=2, L=4.0, mu=0.1, n_refine=args.refine or 2, dt=0.01)
    n = args.steps or 3000
    print(f"2D, {kw}, {n} steps")
    rows = [
        _run('preset', preset, n, **kw),
        _run("viscous_flux='two_point'",
             preset.replace(viscous_flux='two_point'), n, **kw),
        _run("pressure_flux='simplex_gradient'",
             preset.replace(pressure_flux='simplex_gradient'), n, **kw),
        _run("frozen_set='hull'", preset.replace(frozen_set='hull'), n, **kw),
    ]
    _save('arms2d', rows)


def arms3d(args) -> None:
    from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize

    preset = PRESETS['hagen_poiseuille_3D']
    kw = dict(dim=3, L=3.0, mu=0.1, n_refine=args.refine or 1, dt=0.01)
    n = args.steps or 600
    print(f"3D, {kw}, {n} steps")

    def ring(HC, bV, dim, **_kw):
        # delaunay + membership, then force the p_ij ring (no area cache)
        _retopologize(HC, bV, dim, frozen_set='membership')
        HC._edge_area_cache = None

    arms = [
        ('preset', preset, {}),
        ("pressure_flux='centred' (e_star cache)",
         preset.replace(pressure_flux='centred'), {}),
        ("pressure_flux='centred' (p_ij ring)",
         SolverMethods(dim=3, connectivity='custom',
                       viscous_flux='simplex_gradient',
                       label='delaunay + membership, cache cleared'),
         dict(custom=ring)),
        ("viscous_flux='two_point'",
         preset.replace(viscous_flux='two_point'), {}),
    ]
    rows = [_run(label, methods, n, **extra, **kw)
            for label, methods, extra in arms if (args.only or '') in label]
    _save('arms3d', rows, args.tag)


def convergence(args) -> None:
    preset = PRESETS['hagen_poiseuille_2D']
    rows = []
    print("2D, L = 4, mu = 0.1: l2 at t = 10 (regular rows) and at t = 60 "
          "(sheared rows)")
    for refine, dt in ((1, 0.02), (2, 0.01), (3, 0.005)):
        for t_end in (10.0, 60.0):
            if refine == 3 and t_end > 30.0 and not args.long:
                continue
            rows.append(_run(f'refinement {refine}, t = {t_end:g}', preset,
                             int(round(t_end / dt)), dim=2, L=4.0, mu=0.1,
                             n_refine=refine, dt=dt))
            rows[-1]['n_refine'] = refine
    _save('convergence', rows)


def slivers(args) -> None:
    from ddgclib.operators.stress import _SIMPLEX_FLAT_TOL

    rows = []
    for name, kw, n in (
            ('hagen_poiseuille_2D',
             dict(dim=2, L=4.0, mu=0.1, n_refine=2, dt=0.01), 3000),
            ('hagen_poiseuille_3D',
             dict(dim=3, L=3.0, mu=0.1, n_refine=args.refine or 1, dt=0.01),
             args.steps or 600)):
        dim = kw['dim']
        stat = dict(case=name, steps=n, q_min_channel=np.inf,
                    skipped_max=0, skipped_touching_channel_max=0,
                    steps_with_skipped_in_channel=0)
        state = {}

        def callback(step, t, HC, bV=None, diagnostics=None):
            simplices = getattr(HC, '_simplices', None)
            if simplices is None or simplices is state.get('seen'):
                return
            state['seen'] = simplices
            pts = np.array([[w.x_a[:dim] for w in s] for s in simplices])
            vol = np.abs(np.linalg.det(pts[:, 1:] - pts[:, :1])) / (
                2.0 if dim == 2 else 6.0)
            el = np.linalg.norm(pts[:, :, None] - pts[:, None, :], axis=3)
            q = vol / np.where(el > 0, el, np.inf).min(axis=(1, 2)) ** dim
            L = kw['L']
            axis = 0 if dim == 2 else 2
            free = np.array([any(w not in bV and not w.boundary
                                 and 0.0 < w.x_a[axis] <= L
                                 for w in s) for s in simplices])
            skipped = q <= _SIMPLEX_FLAT_TOL
            stat['skipped_max'] = max(stat['skipped_max'], int(skipped.sum()))
            k = int((skipped & free).sum())
            stat['skipped_touching_channel_max'] = max(
                stat['skipped_touching_channel_max'], k)
            stat['steps_with_skipped_in_channel'] += bool(k)
            if free.any():
                stat['q_min_channel'] = min(stat['q_min_channel'],
                                            float(q[free].min()))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            run_developing(PRESETS[name], n_steps=n, callback=callback, **kw)
        rows.append(stat)
        print(f"  {name}: {n} steps; simplices left out per step at most "
              f"{stat['skipped_max']}, of them with a free interior channel "
              f"vertex at most {stat['skipped_touching_channel_max']} "
              f"({stat['steps_with_skipped_in_channel']} steps); smallest "
              f"|T| / l_min^dim of a simplex with such a vertex "
              f"{stat['q_min_channel']:.3e} (tolerance {_SIMPLEX_FLAT_TOL:g})")
    _save('slivers', rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('mode', choices=['static', 'arms2d', 'arms3d',
                                     'convergence', 'slivers'])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--refine', type=int, default=None)
    ap.add_argument('--long', action='store_true',
                    help='convergence: refinement 3 to t = 60 as well')
    ap.add_argument('--only', default=None,
                    help='arms3d: only the arms whose label contains this')
    ap.add_argument('--tag', default='',
                    help='arms3d: write results/laneH/arms3d_<tag>.json (the '
                         'two centred arms differ from process to process; '
                         'repeat them with --only centred --tag p2, p3, ...)')
    a = ap.parse_args()
    globals()[a.mode](a)
