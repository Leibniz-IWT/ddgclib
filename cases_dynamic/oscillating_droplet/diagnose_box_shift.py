#!/usr/bin/env python3
"""laneB (2026-10-05): the droplet-in-box builders lost outer vertices.

``droplet_in_box_2d`` / ``_3d`` build the outer box on ``[0, 2L]^dim``
and shift it by ``-L`` with a loop of ``HC.V.move``.  Some targets are
keys that other vertices still hold, and until laneB each such
collision dropped a vertex (the ``(L, ..., L)`` corner among them).
The builders now take ``box_shift='move_all'`` (default, every vertex
kept) or ``'evict'`` (the old loop).  This script measures both arms.

Subcommands
-----------
census    The raw shift: outer rectangle / box at refinement 1..3 (2D)
          and 1..2 (3D), vertices built / kept / lost, the lost
          coordinates, corner presence.
mesh      Both builder arms through ``setup_oscillating_droplet`` at
          the pinned refinements (2D 3/3, 3D 2/2) and at the fast-test
          fixtures: vertices, walls, corners in the wall set, total dual
          volume against ``(2L)^dim``, outer-phase mass against
          ``rho_o * V_outer``, cell sizes of the outer phase (largest,
          median, the cells at the lost positions), each at setup and
          after ONE retopology through the preset at frozen positions.
shearing  The shearing-plate 2D setup: collisions of the builder shift
          and of the anisotropic rescale, the latter classified as
          mover-onto-mover (an ordering problem that move_all removes)
          or mover-onto-stayer (two vertices wanted at one point).
fullrun   The shipped runner (``oscillating_droplet_2D.py``, ``_3D.py``
          or ``static_droplet_2D.py``) with its output directories
          redirected to ``--out/<runner>_<box_shift>[_<policy>]`` and,
          with ``--policy``, the retopology policy constant of
          ``src/_params.py`` overridden (2D: the ``retopo_policy_2d``
          key of the runner's preset map; 3D: the ``--retopo`` value).
          This is how the A/B scores of the laneB log were made.

Usage
-----
    PY=/home/endres/anaconda3/envs/ddg/bin/python
    $PY cases_dynamic/oscillating_droplet/diagnose_box_shift.py census --out <dir>
    $PY cases_dynamic/oscillating_droplet/diagnose_box_shift.py mesh --out <dir>
    $PY cases_dynamic/oscillating_droplet/diagnose_box_shift.py shearing --out <dir>
    $PY cases_dynamic/oscillating_droplet/diagnose_box_shift.py fullrun \
        --runner 2d --box-shift evict --out <dir> --no-anim
    $PY cases_dynamic/oscillating_droplet/diagnose_box_shift.py fullrun \
        --runner 2d --policy delaunay_remap_p2 --out <dir> --no-anim

Default ``--out`` is ``results/box_shift/`` next to this file (the kept
record); probe runs pass a scratch directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ddgclib.geometry.domains import rectangle, box  # noqa: E402
from ddgclib.geometry.domains._multiphase_droplet import (  # noqa: E402
    BOX_SHIFTS, _shift_outer_box,
)
from ddgclib.methods import PRESETS  # noqa: E402

_OUT_DEFAULT = os.path.join(_HERE, 'results', 'box_shift')


def _dump(out_dir: str, name: str, doc: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1, default=lambda o: repr(o))
    return path


def _corners(L: float, dim: int) -> list[tuple[float, ...]]:
    import itertools
    return [tuple(float(s * L) for s in signs)
            for signs in itertools.product((-1.0, 1.0), repeat=dim)]


def _has_key(HC, key, tol=1e-12) -> bool:
    key = np.asarray(key)
    for v in HC.V:
        if np.max(np.abs(v.x_a[:len(key)] - key)) < tol:
            return True
    return False


# ---------------------------------------------------------------------------
# census: the raw shift
# ---------------------------------------------------------------------------
def _shift_census(dim: int, refinement: int, L: float) -> dict:
    rows = {}
    for arm in BOX_SHIFTS:
        if dim == 2:
            res = rectangle(L=2 * L, h=2 * L, refinement=refinement,
                            flow_axis=0)
        else:
            res = box(Lx=2 * L, Ly=2 * L, Lz=2 * L, refinement=refinement)
        HC = res.HC
        before = {tuple(float(c) for c in v.x_a) for v in HC.V}
        n_before = len(before)
        _shift_outer_box(HC, (-L,) * dim, arm)
        after = [v for v in HC.V]
        keys_after = {tuple(float(c) for c in v.x_a) for v in after}
        # Which built positions are missing after the shift (shift the
        # built keys and look them up).
        shifted = {tuple(float(c) - L for c in k) for k in before}
        lost = sorted(k for k in shifted if k not in keys_after)
        corners = _corners(L, dim)
        rows[arm] = {
            'built': n_before,
            'after': len(after),
            'lost': len(lost),
            'lost_positions': lost,
            'corners_present': {repr(c): c in keys_after for c in corners},
            # a dropped vertex keeps its edges: dangling neighbours
            'dangling_nn': int(sum(
                1 for v in after for nb in v.nn
                if tuple(float(c) for c in nb.x_a) not in keys_after)),
        }
    return {'dim': dim, 'refinement': refinement, 'L': L, 'arms': rows}


def cmd_census(args) -> None:
    out = []
    for dim, refs in ((2, (1, 2, 3)), (3, (1, 2))):
        for r in refs:
            row = _shift_census(dim, r, args.L)
            out.append(row)
            a, b = row['arms']['evict'], row['arms']['move_all']
            print(f"{dim}D refinement {r}: built {a['built']}, evict keeps "
                  f"{a['after']} (lost {a['lost']}, dangling nn "
                  f"{a['dangling_nn']}), move_all keeps {b['after']} "
                  f"(lost {b['lost']}); corner (L,..,L) present: evict "
                  f"{a['corners_present'][repr(tuple([args.L] * dim))]}, "
                  f"move_all {b['corners_present'][repr(tuple([args.L] * dim))]}")
            if a['lost']:
                print("   lost (evict): " + ', '.join(
                    '(' + ', '.join(f'{c:+.4g}' for c in p) + ')'
                    for p in a['lost_positions']))
    path = _dump(args.out, 'census.json', {'rows': out})
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# mesh: the builder arms at setup and after one retopology
# ---------------------------------------------------------------------------
def _mesh_state(HC, bV, mps, dim, L, lost_positions) -> dict:
    from ddgclib.operators.stress import cache_dual_volumes  # noqa: F401
    verts = list(HC.V)
    vol = np.array([float(getattr(v, 'dual_vol', 0.0) or 0.0) for v in verts])
    outer = np.array([v.phase == 0 for v in verts])
    m0 = float(sum(float(v.m_phase[0]) for v in verts
                   if hasattr(v, 'm_phase')))
    m1 = float(sum(float(v.m_phase[1]) for v in verts
                   if hasattr(v, 'm_phase')))
    corners = _corners(L, dim)
    bkeys = {tuple(float(c) for c in v.x_a) for v in bV}
    keys = {tuple(float(c) for c in v.x_a) for v in verts}
    # cells whose vertex is a 1-ring neighbour of a lost position
    near = []
    for p in lost_positions:
        p = np.asarray(p)
        d = np.array([np.linalg.norm(v.x_a[:dim] - p) for v in verts])
        h = L / 2 ** 2  # placeholder; replaced by nearest spacing below
        h = float(np.sort(d)[1]) if len(d) > 1 else h
        idx = np.where(d < 1.5 * h + 1e-12)[0]
        near.append({'position': [float(c) for c in p],
                     'n_neighbour_cells': int(len(idx)),
                     'neighbour_cell_vol_max': float(vol[idx].max()) if len(idx) else None,
                     'neighbour_cell_vol_mean': float(vol[idx].mean()) if len(idx) else None})
    return {
        'n_vertices': len(verts),
        'n_walls': len(bV),
        'n_boundary_tag': int(sum(1 for v in verts if getattr(v, 'boundary', False))),
        'n_interface': int(sum(1 for v in verts if getattr(v, 'is_interface', False))),
        'corners_in_complex': {repr(c): (c in keys) for c in corners},
        'corners_in_walls': {repr(c): (c in bkeys) for c in corners},
        'total_dual_vol': float(vol.sum()),
        'box_volume': float((2 * L) ** dim),
        'outer_cell_vol_max': float(vol[outer].max()) if outer.any() else None,
        'outer_cell_vol_median': float(np.median(vol[outer])) if outer.any() else None,
        'outer_cell_vol_min_positive': float(vol[outer][vol[outer] > 0].min()) if (vol[outer] > 0).any() else None,
        'n_zero_vol': int(np.sum(vol <= 0.0)),
        'mass_phase0': m0,
        'mass_phase1': m1,
        'mass_total': float(sum(float(v.m) for v in verts)),
        'cells_at_lost_positions': near,
    }


def _run_mesh_arm(dim, ro, rd, arm, preset_name, L) -> dict:
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    from cases_dynamic.oscillating_droplet.src._params import (
        R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    )
    methods = PRESETS[preset_name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        HC, bV, mps, bc_set, dudt_fn, _rf, params = setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=0.0, l=l, rho_d=rho_d, rho_o=rho_o,
            mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
            L_domain=L, refinement_outer=ro, refinement_droplet=rd,
            split_method=methods.split_method,
            redistribute_mass=methods.redistribute_mass, box_shift=arm)
    lost = _shift_census(dim, ro, L)['arms']['evict']['lost_positions']
    setup_state = _mesh_state(HC, bV, mps, dim, L, lost)
    retopo = methods.retopologize_fn(mps=mps)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        retopo(HC, bV, dim)
    after_state = _mesh_state(HC, bV, mps, dim, L, lost)
    V_outer = (2 * L) ** dim - (np.pi * R0 ** 2 if dim == 2
                                else 4.0 / 3.0 * np.pi * R0 ** 3)
    return {
        'dim': dim, 'refinement_outer': ro, 'refinement_droplet': rd,
        'box_shift': arm, 'preset': preset_name,
        'methods': methods.to_dict(),
        'rho_o_times_V_outer': float(rho_o * V_outer),
        'setup': setup_state,
        'after_one_retopology': after_state,
    }


def cmd_mesh(args) -> None:
    from cases_dynamic.oscillating_droplet.src._params import L_domain
    fixtures = [
        (2, 3, 3, 'oscillating_droplet_2D'),
        (2, 2, 2, 'oscillating_droplet_2D'),
        (2, 1, 2, 'oscillating_droplet_2D'),
        (3, 2, 2, 'oscillating_droplet_3D'),
        (3, 2, 2, 'oscillating_droplet_3D_delaunay'),
        (3, 1, 1, 'oscillating_droplet_3D'),
    ]
    if args.only:
        fixtures = [f for f in fixtures
                    if f"{f[0]}d_{f[1]}{f[2]}_{f[3]}" in args.only]
    rows = []
    for dim, ro, rd, preset in fixtures:
        for arm in BOX_SHIFTS:
            row = _run_mesh_arm(dim, ro, rd, arm, preset, L_domain)
            rows.append(row)
            s, a = row['setup'], row['after_one_retopology']
            print(f"\n{dim}D refine {ro}/{rd} {preset} box_shift={arm}:")
            for tag, st in (('setup', s), ('after 1 retopo', a)):
                corners_w = sum(st['corners_in_walls'].values())
                print(f"  {tag:15s} V {st['n_vertices']:4d} walls {st['n_walls']:3d} "
                      f"(corners {corners_w}/{2 ** dim}) tag {st['n_boundary_tag']:3d} "
                      f"iface {st['n_interface']:3d} | sum dual_vol "
                      f"{st['total_dual_vol']:.10e} (box {st['box_volume']:.4e}) "
                      f"| m0 {st['mass_phase0']:.10e} (rho_o V_outer "
                      f"{row['rho_o_times_V_outer']:.6e}) m1 {st['mass_phase1']:.10e} "
                      f"| outer cell max/med {st['outer_cell_vol_max']:.4e}/"
                      f"{st['outer_cell_vol_median']:.4e} zero {st['n_zero_vol']}")
            if s['cells_at_lost_positions']:
                print("  cells around the lost positions (setup; max / mean dual_vol):")
                for c in s['cells_at_lost_positions']:
                    print(f"    {tuple(round(x, 5) for x in c['position'])}: "
                          f"{c['n_neighbour_cells']} cells, "
                          f"{c['neighbour_cell_vol_max']:.4e} / "
                          f"{c['neighbour_cell_vol_mean']:.4e}")
    path = _dump(args.out, 'mesh.json', {'rows': rows})
    print(f"\n-> {path}")


# ---------------------------------------------------------------------------
# shearing: the rescale collision
# ---------------------------------------------------------------------------
def cmd_shearing(args) -> None:
    from ddgclib.geometry.domains import droplet_in_box_2d
    from cases_dynamic.shearing_plate_droplet.src import _params as sp
    L_build = max(sp.L_x, sp.L_y)
    out = {}
    for arm in BOX_SHIFTS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = droplet_in_box_2d(R=sp.R0, L=L_build, refinement_outer=3,
                                    refinement_droplet=3, box_shift=arm)
        HC = res.HC
        n_built = len(list(HC.V))
        scale = np.array([sp.L_x / L_build, sp.L_y / L_build])
        movers = []
        for v in list(HC.V):
            r = float(np.linalg.norm(v.x_a[:2]))
            if r > sp.R0 + 1e-12:
                new_pos = v.x_a.copy()
                new_pos[:2] = v.x_a[:2] * scale
                movers.append((v, tuple(new_pos)))
        mover_ids = {id(v) for v, _ in movers}
        onto_stayer, onto_mover, duplicate_target = [], [], []
        targets = {}
        for v, key in movers:
            other = HC.V.cache.get(key)
            if other is not None and other is not v:
                (onto_mover if id(other) in mover_ids else onto_stayer).append(
                    {'from': [float(c) for c in v.x], 'to': [float(c) for c in key],
                     'occupant_phase': int(getattr(other, 'phase', -1)),
                     'occupant_r': float(np.linalg.norm(other.x_a[:2]))})
            if key in targets:
                duplicate_target.append([float(c) for c in key])
            targets[key] = v
        out[arm] = {
            'vertices_after_builder': n_built,
            'rescale_movers': len(movers),
            'rescale_onto_mover_key': len(onto_mover),
            'rescale_onto_stayer_key': len(onto_stayer),
            'rescale_two_movers_one_key': len(duplicate_target),
            'onto_stayer': onto_stayer, 'onto_mover': onto_mover[:5],
        }
        print(f"shearing 2D builder box_shift={arm}: {n_built} vertices; "
              f"rescale moves {len(movers)}: onto a mover's key "
              f"{len(onto_mover)}, onto a stayer's key {len(onto_stayer)}, "
              f"two movers on one key {len(duplicate_target)}")
        for c in onto_stayer:
            print(f"   stayer collision: {tuple(round(x, 6) for x in c['from'])} -> "
                  f"{tuple(round(x, 6) for x in c['to'])} held by phase "
                  f"{c['occupant_phase']} at r {c['occupant_r']:.6f} (R0 {sp.R0})")
    path = _dump(args.out, 'shearing.json', out)
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# floors: the pinned static floors (run_a5b harness) in both arms
# ---------------------------------------------------------------------------
def cmd_floors(args) -> None:
    from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import run_a5b
    rows = []
    for dim, ro, rd, preset in ((2, 3, 3, 'static_droplet_floor_2D'),
                                (3, 2, 2, 'static_droplet_floor_3D')):
        for arm in BOX_SHIFTS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = run_a5b(dim=dim, refinement_outer=ro, refinement_droplet=rd,
                            n_steps=args.steps or 20, methods=PRESETS[preset],
                            box_shift=arm)
            hist = np.asarray(r['max_abs_F_history'])
            plateau = hist[1:]
            row = {
                'dim': dim, 'refinement_outer': ro, 'refinement_droplet': rd,
                'preset': preset, 'box_shift': arm, 'n_steps': r['n_steps'],
                'step0_maxF': float(hist[0]), 'step1_maxF': float(hist[1]),
                'peak_maxF': r['max_abs_F_peak'], 'end_maxF': r['max_abs_F_end'],
                'plateau_rel_spread': float((plateau.max() - plateau.min()) / plateau[-1]),
                'mass_rel': r['mass_rel_drift'] if 'mass_rel_drift' in r else None,
                'n_verts': r['n_verts_history'][0], 'n_iface': r['n_iface_history'][0],
                'volume_history_0_1': [r['volume_history'][0], r['volume_history'][1]],
            }
            rows.append(row)
            print(f"{dim}D {ro}/{rd} {preset} box_shift={arm}: step0 "
                  f"{row['step0_maxF']:.10e} step1 {row['step1_maxF']:.10e} "
                  f"peak {row['peak_maxF']:.10e} end {row['end_maxF']:.10e} "
                  f"plateau spread {row['plateau_rel_spread']:.2e} "
                  f"V {row['n_verts']} iface {row['n_iface']}")
    path = _dump(args.out, 'floors.json', {'rows': rows})
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# envelope: the fast dynamic pins (refinement 2/2 mirror, 1/2 fixtures)
# ---------------------------------------------------------------------------
def _cfl_dt(HC, dim, K_d, rho_d, gamma):
    c_s = float(np.sqrt(K_d / rho_d))
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    return min(0.25 * dx_min / c_s,
               0.5 * float(np.sqrt(rho_d * dx_min ** 3 / gamma)))


def cmd_envelope(args) -> None:
    from cases_dynamic.oscillating_droplet.src._setup import (
        setup_oscillating_droplet,
    )
    from cases_dynamic.oscillating_droplet.src._params import (
        R0, epsilon, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
        L_domain, t_end_2d,
    )
    from cases_dynamic.oscillating_droplet.src._analytical import (
        rayleigh_frequency, lamb_damping_rate,
    )
    from cases_dynamic.oscillating_droplet.src._metrics import oscillation_score
    from cases_dynamic.oscillating_droplet.src._plot_helpers import (
        compute_diagnostics,
    )
    dim = 2
    omega = rayleigh_frequency(l, gamma, rho_d, R0, dim=dim, rho_outer=rho_o)
    beta = lamb_damping_rate(l, mu_d, rho_d, R0, dim=dim)
    rows = []
    for arm in BOX_SHIFTS:
        # --- the refinement 2/2 mirror of the full run (envelope test) ---
        methods = PRESETS['oscillating_droplet_2D']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, mps, bc_set, dudt_fn, _r, params = setup_oscillating_droplet(
                dim=dim, R0=R0, epsilon=epsilon, l=l, rho_d=rho_d, rho_o=rho_o,
                mu_d=mu_d, mu_o=mu_o, gamma=gamma, K_d=K_d, K_o=K_o,
                L_domain=L_domain, refinement_outer=2, refinement_droplet=2,
                split_method=methods.split_method,
                redistribute_mass=methods.redistribute_mass, box_shift=arm)
        dt = _cfl_dt(HC, dim, K_d, rho_d, gamma)
        t_end = min(t_end_2d, 5.0 / beta)
        n_steps = int(t_end / dt) + 1
        record_every = max(1, n_steps // 200)
        diags = []

        def record(t):
            d = compute_diagnostics(HC, dim=dim)
            d['t'] = float(t)
            diags.append(d)
        record(0.0)

        def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
            if step % record_every == 0:
                record(t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t_final = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                                        bc_set=bc_set, callback=cb, mps=mps)
        record(t_final)
        score = oscillation_score(diags, R0=R0, epsilon=epsilon, l=l,
                                  omega=omega, beta=beta)
        row = {'fixture': 'envelope_2/2', 'box_shift': arm, 'preset': methods.label,
               'n_verts': sum(1 for _ in HC.V), 'n_steps': n_steps, 'dt': dt,
               'l2': score['l2_error_normalized'], 'linf': score['linf_error_normalized'],
               'tail': score['tail_growth'], 'mass_drift': score['mass_drift'],
               'KE_max': float(max(d['KE'] for d in diags))}
        rows.append(row)
        print(f"envelope 2/2 box_shift={arm}: V {row['n_verts']} steps {n_steps} "
              f"l2 {row['l2']:.17g} tail {row['tail']:.17g} linf {row['linf']:.17g} "
              f"mass {row['mass_drift']:.3e} KE_max {row['KE_max']:.17g}")

        # --- the refinement 1/2 fixtures of the fast suite ---
        def build(preset, eps=0.05):
            m = PRESETS[preset]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return m, setup_oscillating_droplet(
                    dim=2, R0=0.01, epsilon=eps, l=2, rho_d=800.0, rho_o=1000.0,
                    mu_d=0.5, mu_o=0.1, gamma=0.05, L_domain=0.05,
                    refinement_outer=1, refinement_droplet=2,
                    split_method=m.split_method,
                    redistribute_mass=m.redistribute_mass, box_shift=arm)

        for preset, n, label, extra in (
                ('oscillating_droplet_2D', 200, 'endurance_200', {}),
                ('oscillating_droplet_2D', 40, 'remap_40', {}),
                ('oscillating_droplet_2D_dual_only', 40, 'dual_only_40', {}),
                ('oscillating_droplet_2D_dual_only', 40, 'dual_only_40_p4',
                 {'projection_every': 4}),
                ('oscillating_droplet_2D', 40, 'remap_40_p5',
                 {'projection_every': 5})):
            m, (HC, bV, mps, bc_set, dudt_fn, _r, params) = build(preset)
            if extra:
                m = m.replace(**extra)
            dt = _cfl_dt(HC, 2, params['K_d'], 800.0, 0.05)
            d0 = compute_diagnostics(HC, dim=2)
            ke = []

            def cb2(step, t, HC_cb, bV_cb=None, diagnostics=None):
                if step % 40 == 0:
                    ke.append(float(compute_diagnostics(HC_cb, dim=2)['KE']))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n, bc_set=bc_set,
                            callback=cb2, mps=mps)
            d1 = compute_diagnostics(HC, dim=2)
            row = {'fixture': label, 'box_shift': arm, 'preset': preset,
                   'replace': extra, 'n_verts': sum(1 for _ in HC.V), 'dt': dt,
                   'KE_end': float(d1['KE']), 'KE_trace_max': float(max(ke)) if ke else None,
                   'R_max_end': float(d1['R_max']),
                   'mass_drift': abs(d1['total_mass'] - d0['total_mass']) / d0['total_mass']}
            rows.append(row)
            print(f"{label:16s} box_shift={arm}: V {row['n_verts']} KE_end "
                  f"{row['KE_end']:.10e} KE_trace_max {row['KE_trace_max']} "
                  f"R_max_end {row['R_max_end']:.17g} mass {row['mass_drift']:.3e}")
    path = _dump(args.out, 'envelope.json', {'rows': rows})
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# shearrun: the shearing-plate 2D short window in both arms
# ---------------------------------------------------------------------------
def _digest(HC, dim=2) -> str:
    import hashlib
    state = sorted((tuple(v.x_a[:dim]), tuple(v.u[:dim]), float(v.m),
                    tuple(float(p) for p in v.p_phase)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()


def cmd_shearrun(args) -> None:
    from cases_dynamic.shearing_plate_droplet.src._setup import (
        setup_shearing_plate_droplet,
    )
    from cases_dynamic.shearing_plate_droplet.src import _params as sp
    m = PRESETS['shearing_plate_droplet_2D']
    out = {}
    for arm in BOX_SHIFTS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, mps, bc_set, dudt_fn, _rf, groups, params = \
                setup_shearing_plate_droplet(
                    dim=2, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=sp.U_wall,
                    rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
                    gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
                    refinement_outer=3, refinement_droplet=3,
                    redistribute_mass=m.redistribute_mass, box_shift=arm)
        n0 = sum(1 for _ in HC.V)
        iface0 = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        rec = {'box_shift': arm, 'setup_digest': _digest(HC), 'n_verts': n0,
               'n_iface': iface0, 'n_walls': len(bV),
               'mass_total': float(sum(v.m for v in HC.V))}
        # lane L's three-step probe (dt 1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.integrate(HC, bV, dudt_fn, dt=1e-5, n_steps=3, bc_set=bc_set,
                        mps=mps, domain_bounds=params['domain_bounds'])
        rec['three_step_digest'] = _digest(HC)
        rec['three_step_n_verts'] = sum(1 for _ in HC.V)
        # the short window of _run_short_2D.py (fresh setup)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, mps, bc_set, dudt_fn, _rf, groups, params = \
                setup_shearing_plate_droplet(
                    dim=2, R0=sp.R0, L_x=sp.L_x, L_y=sp.L_y, U_wall=sp.U_wall,
                    rho_d=sp.rho_d, rho_o=sp.rho_o, mu_d=sp.mu_d, mu_o=sp.mu_o,
                    gamma=sp.gamma, K_d=sp.K_d, K_o=sp.K_o,
                    refinement_outer=3, refinement_droplet=3,
                    redistribute_mass=m.redistribute_mass, box_shift=arm)
        c_s = float(np.sqrt(sp.K_o / sp.rho_o))
        dx_min = min(float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
                     for v in HC.V for nb in v.nn
                     if np.linalg.norm(v.x_a[:2] - nb.x_a[:2]) > 1e-15)
        dt = 0.1 * dx_min / c_s
        t_end = args.t_end if args.t_end else 0.05
        n_steps = args.steps or int(t_end / dt) + 1
        every = max(1, n_steps // 50)
        series = []
        first_loss = [None]

        def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
            n_if = sum(1 for v in HC_cb.V if getattr(v, 'is_interface', False))
            if first_loss[0] is None and n_if < iface0:
                first_loss[0] = (step, float(t), n_if)
            if step % every == 0 or step == n_steps - 1:
                ke = float(sum(0.5 * v.m * float(np.dot(v.u[:2], v.u[:2]))
                               for v in HC_cb.V))
                umax = float(max(np.linalg.norm(v.u[:2]) for v in HC_cb.V))
                series.append({'step': step, 't': float(t), 'n_verts': len(HC_cb.V),
                               'n_iface': n_if, 'KE': ke, 'umax_over_Uwall': umax / sp.U_wall})
        abort = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps, bc_set=bc_set,
                            callback=cb, mps=mps, domain_bounds=params['domain_bounds'])
        except Exception as e:  # noqa: BLE001 - an abort is a result here
            abort = f"{type(e).__name__}: {e}"[:300]
        rec.update({'dt': dt, 'n_steps': n_steps, 'abort': abort,
                    'first_interface_loss': first_loss[0],
                    'n_iface_end': series[-1]['n_iface'] if series else None,
                    'KE_max': max(s['KE'] for s in series) if series else None,
                    'umax_over_Uwall_max': max(s['umax_over_Uwall'] for s in series) if series else None,
                    'final_digest': _digest(HC), 'series': series})
        out[arm] = rec
        print(f"shearing 2D box_shift={arm}: setup V {n0} iface {iface0} walls "
              f"{len(bV)} digest {rec['setup_digest'][:16]}; 3 steps -> "
              f"{rec['three_step_digest'][:16]} ({rec['three_step_n_verts']} V); "
              f"short window {n_steps} steps dt {dt:.3e}: first interface loss "
              f"{first_loss[0]}, iface end {rec['n_iface_end']}, KE_max "
              f"{rec['KE_max']:.6e}, |u|max/U_wall {rec['umax_over_Uwall_max']:.3f}"
              + (f", ABORT {abort}" if abort else ''))
    path = _dump(args.out, 'shearrun.json', out)
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# fullrun: a shipped runner with redirected outputs
# ---------------------------------------------------------------------------
_RUNNERS = {'2d': 'oscillating_droplet_2D', '3d': 'oscillating_droplet_3D',
            'static': 'static_droplet_2D'}


def cmd_fullrun(args) -> None:
    import importlib
    name = _RUNNERS[args.runner]
    mod = importlib.import_module(f'cases_dynamic.oscillating_droplet.{name}')
    tag = f"{name}_{args.box_shift}" + (f"_{args.policy}" if args.policy else '')
    out = os.path.join(args.out, tag)
    mod._FIG = os.path.join(out, 'fig')
    mod._RESULTS = os.path.join(out, 'results')
    mod._SNAPSHOTS = os.path.join(mod._RESULTS, 'snapshots')
    if args.no_anim:
        def _skip(*_a, **_k):
            raise RuntimeError('animation skipped (--no-anim)')
        mod.dynamic_plot_fluid = _skip
    print(f"[fullrun] {name} box_shift={args.box_shift} policy={args.policy} "
          f"-> {out}")
    if args.runner == '3d':
        mod.main(retopo_policy=args.policy, box_shift=args.box_shift)
    else:
        if args.policy:
            if args.runner != '2d':
                raise SystemExit('--policy applies to the 2d and 3d runners')
            mod.retopo_policy_2d = args.policy
        mod.main(box_shift=args.box_shift)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('census', 'mesh', 'shearing', 'floors',
                                    'envelope', 'shearrun', 'fullrun'))
    ap.add_argument('--out', default=_OUT_DEFAULT)
    ap.add_argument('--L', type=float, default=0.05)
    ap.add_argument('--only', nargs='*', default=None,
                    help="mesh: fixture tags like 2d_33_oscillating_droplet_2D")
    ap.add_argument('--steps', type=int, default=None,
                    help='floors / shearrun: number of steps')
    ap.add_argument('--t-end', dest='t_end', type=float, default=None,
                    help='shearrun: window length (default 0.05 s)')
    ap.add_argument('--runner', choices=sorted(_RUNNERS), default='2d')
    ap.add_argument('--box-shift', dest='box_shift', default='move_all',
                    choices=BOX_SHIFTS)
    ap.add_argument('--policy', default=None,
                    help="fullrun: retopo policy (2d: delaunay_remap, "
                         "dual_only, delaunay, delaunay_remap_p2; 3d: "
                         "dual_only, delaunay)")
    ap.add_argument('--no-anim', dest='no_anim', action='store_true')
    args = ap.parse_args()
    {'census': cmd_census, 'mesh': cmd_mesh, 'shearing': cmd_shearing,
     'floors': cmd_floors, 'envelope': cmd_envelope, 'shearrun': cmd_shearrun,
     'fullrun': cmd_fullrun}[args.cmd](args)


if __name__ == '__main__':
    main()
