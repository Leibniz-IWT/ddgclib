#!/usr/bin/env python3
"""Phase 2 of the 2026-04-28 stabilisation roadmap — diff the single
A.5.b step-1 retopology call in 3D.

Phase 1 (`diagnose_a5_bisection.py --split-method exact`) ruled out the
``split_method='neighbour_count'`` hypothesis: with ``'exact'`` applied
end-to-end (setup + runtime), the 3D step-1 max |F| jump is
**1.75e-3 vs 1.44e-3** — i.e. ``'exact'`` is *slightly worse*, not the
fix.  2D is unaffected: A.5.b ``exact`` matches A.5.a exactly
(2.37e-3 vs 2.37e-3), confirming retopology is exactly neutral on a
static 2D mesh under ``'exact'`` and that the 2D residual is pure
curvature stencil.

This script answers Phase 2's question: at the single 3D retopology
call that bumps interface max |F| from 6.02e-5 to ~1.4-1.8e-3, **which
per-vertex per-phase quantity diverges first?**  Per the plan, the
first quantity to diverge identifies the next layer.

For every interface vertex (keyed by ``v.x`` coordinate tuple, which
survives Delaunay reconnection because vertex positions don't change
in this experiment):

  - ``v.is_interface`` (bool — does interface tagging flip?)
  - ``v.phase`` (int — does the bulk-phase label flip?)
  - ``v.dual_vol`` (float — does the total dual cell volume change?)
  - ``v.dual_vol_phase`` (per-phase array — split policy artefact?)
  - ``v.m_phase`` (per-phase array — should be Lagrangian-conserved
    by ``reset_mass=False`` inside ``mps.refresh``)
  - ``v.p_phase`` (per-phase array — derived from m/V via EOS)
  - 1-ring connectivity set ``{nb.x for nb in v.nn}`` (Delaunay churn?)
  - per-vertex multiphase stress force magnitude
    ``|multiphase_stress_force(v, dim, mps, HC)|`` (the actual
    quantity that drives the |F| jump)

Reports a ranked summary by largest absolute delta in each quantity,
plus the top-N vertices contributing to the |F| jump.

Usage
-----
    python cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py
    python cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py \
        --split-method exact
    python cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py \
        --refine-outer 2 --refine-droplet 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o, L_domain,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from ddgclib.operators.multiphase_stress import multiphase_stress_force


_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_CASE_DIR, 'results_a5_bisection')


def _snapshot(HC, dim, mps):
    """Snapshot per-vertex multiphase state.

    Keyed by ``v.x`` (coord tuple) so we can match across the retopo
    call.  Records every quantity called out by Phase 2 of the plan,
    plus the per-vertex stress force as the closing reduction.
    """
    snap = {}
    for v in HC.V:
        if not getattr(v, 'is_interface', False):
            # We only diff interface vertices — bulk vertices are
            # uninvolved in the |F| jump.  But we DO snapshot whether
            # is_interface flips — see check_interface_flip below.
            pass
        key = tuple(v.x)
        F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
        snap[key] = {
            'is_interface': bool(getattr(v, 'is_interface', False)),
            'phase': int(getattr(v, 'phase', -1)),
            'dual_vol': float(getattr(v, 'dual_vol', 0.0)),
            'dual_vol_phase': np.array(
                getattr(v, 'dual_vol_phase', np.zeros(mps.n_phases)),
                copy=True, dtype=float,
            ),
            'm_phase': np.array(
                getattr(v, 'm_phase', np.zeros(mps.n_phases)),
                copy=True, dtype=float,
            ),
            'p_phase': np.array(
                getattr(v, 'p_phase', np.zeros(mps.n_phases)),
                copy=True, dtype=float,
            ),
            'rho_phase': np.array(
                getattr(v, 'rho_phase', np.zeros(mps.n_phases)),
                copy=True, dtype=float,
            ),
            'nn': frozenset(tuple(nb.x) for nb in v.nn),
            'F': np.array(F, copy=True, dtype=float),
            '|F|': float(np.linalg.norm(F)),
        }
    return snap


def _l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _diff_summary(before, after, top_n=10):
    """Compare two snapshots; return a per-quantity ranked summary."""
    common_keys = [k for k in before if k in after]
    only_before = [k for k in before if k not in after]
    only_after = [k for k in after if k not in before]

    n_iface_before = sum(1 for k in before if before[k]['is_interface'])
    n_iface_after = sum(1 for k in after if after[k]['is_interface'])

    # Per-vertex delta records (interface vertices BEFORE the call only —
    # the question is "which interface vertex's |F| changed and why")
    deltas = []
    for key in common_keys:
        b = before[key]
        a = after[key]
        if not (b['is_interface'] or a['is_interface']):
            continue
        nn_added = a['nn'] - b['nn']
        nn_removed = b['nn'] - a['nn']
        deltas.append({
            'key': key,
            'is_iface_before': b['is_interface'],
            'is_iface_after': a['is_interface'],
            'phase_before': b['phase'],
            'phase_after': a['phase'],
            'phase_flipped': b['phase'] != a['phase'],
            'iface_flipped': b['is_interface'] != a['is_interface'],
            'd_dual_vol': abs(b['dual_vol'] - a['dual_vol']),
            'd_dual_vol_phase_l2': _l2(b['dual_vol_phase'], a['dual_vol_phase']),
            'd_dual_vol_phase_maxabs': _maxabs(
                b['dual_vol_phase'], a['dual_vol_phase']
            ),
            'd_m_phase_l2': _l2(b['m_phase'], a['m_phase']),
            'd_m_phase_maxabs': _maxabs(b['m_phase'], a['m_phase']),
            'd_p_phase_l2': _l2(b['p_phase'], a['p_phase']),
            'd_p_phase_maxabs': _maxabs(b['p_phase'], a['p_phase']),
            'd_rho_phase_maxabs': _maxabs(b['rho_phase'], a['rho_phase']),
            'd_F_l2': _l2(b['F'], a['F']),
            'd_|F|': abs(b['|F|'] - a['|F|']),
            'F_after': a['F'].tolist(),
            'F_before': b['F'].tolist(),
            'nn_n_added': len(nn_added),
            'nn_n_removed': len(nn_removed),
            'nn_n_before': len(b['nn']),
            'nn_n_after': len(a['nn']),
        })

    return {
        'n_keys_before': len(before),
        'n_keys_after': len(after),
        'n_keys_common': len(common_keys),
        'n_only_before': len(only_before),
        'n_only_after': len(only_after),
        'n_iface_before': n_iface_before,
        'n_iface_after': n_iface_after,
        'iface_count_changed': n_iface_before != n_iface_after,
        'deltas': deltas,
    }


def _print_top(deltas, key, top_n=10, label=None):
    label = label or key
    sorted_ = sorted(deltas, key=lambda d: d[key], reverse=True)
    print(f"  Top {top_n} by {label}:")
    for d in sorted_[:top_n]:
        x = d['key']
        if isinstance(x, tuple) and len(x) >= 3:
            xs = f"({x[0]:+.3e}, {x[1]:+.3e}, {x[2]:+.3e})"
        else:
            xs = str(x)
        print(f"    {label}={d[key]:.4e}  "
              f"x={xs}  "
              f"phase {d['phase_before']}->{d['phase_after']}  "
              f"iface {int(d['is_iface_before'])}->{int(d['is_iface_after'])}  "
              f"|F|: {np.linalg.norm(d['F_before']):.3e}->"
              f"{np.linalg.norm(d['F_after']):.3e}  "
              f"nn:{d['nn_n_before']}->{d['nn_n_after']} "
              f"(+{d['nn_n_added']}/-{d['nn_n_removed']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, default=3, choices=[2, 3])
    parser.add_argument(
        '--split-method', choices=['neighbour_count', 'exact'],
        default='neighbour_count',
    )
    parser.add_argument('--refine-outer', type=int, default=2)
    parser.add_argument('--refine-droplet', type=int, default=2)
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(_RESULTS, exist_ok=True)

    print('=' * 72)
    print(f"A.5 Phase 2 — single retopo step diff (dim={args.dim}, "
          f"split_method={args.split_method!r})")
    print('=' * 72)

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=args.dim, R0=R0, epsilon=0.0, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=args.refine_outer,
            refinement_droplet=args.refine_droplet,
            split_method=args.split_method,
        )

    n_iface_init = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    print(f"  Initial: nV={sum(1 for _ in HC.V)} nIface={n_iface_init}")

    before = _snapshot(HC, args.dim, mps)
    print(f"  Snapshotted {len(before)} vertices BEFORE retopo")

    # Run the SINGLE retopology call — no integration, no motion.
    retopo_fn(HC, bV, args.dim)

    after = _snapshot(HC, args.dim, mps)
    print(f"  Snapshotted {len(after)} vertices AFTER retopo")

    diff = _diff_summary(before, after)

    print()
    print('-' * 72)
    print(f"  vertex set:       common={diff['n_keys_common']}  "
          f"only_before={diff['n_only_before']}  "
          f"only_after={diff['n_only_after']}")
    print(f"  interface count:  before={diff['n_iface_before']}  "
          f"after={diff['n_iface_after']}  "
          f"changed={diff['iface_count_changed']}")

    deltas = diff['deltas']
    print(f"  interface vertices diffed: {len(deltas)}")
    print()

    # Aggregate stats per quantity
    print("  AGGREGATE max-of-per-vertex deltas across interface vertices:")
    for k, label in [
        ('d_dual_vol', 'd dual_vol'),
        ('d_dual_vol_phase_maxabs', 'd dual_vol_phase (maxabs)'),
        ('d_m_phase_maxabs', 'd m_phase (maxabs)'),
        ('d_rho_phase_maxabs', 'd rho_phase (maxabs)'),
        ('d_p_phase_maxabs', 'd p_phase (maxabs)'),
        ('d_F_l2', 'd |F_vec| (l2)'),
        ('d_|F|', 'd |F| (scalar)'),
    ]:
        if not deltas:
            continue
        vals = np.array([d[k] for d in deltas])
        mean = float(np.mean(vals))
        med = float(np.median(vals))
        mx = float(np.max(vals))
        print(f"    {label:32s}  max={mx:.4e}  mean={mean:.4e}  "
              f"median={med:.4e}")

    n_phase_flipped = sum(1 for d in deltas if d['phase_flipped'])
    n_iface_flipped = sum(1 for d in deltas if d['iface_flipped'])
    print(f"\n  vertices with phase flipped:        {n_phase_flipped}")
    print(f"  vertices with is_interface flipped: {n_iface_flipped}")

    # Connectivity churn
    nn_added = np.array([d['nn_n_added'] for d in deltas])
    nn_removed = np.array([d['nn_n_removed'] for d in deltas])
    print(f"\n  1-ring churn (interface verts):")
    print(f"    edges added:    max={int(nn_added.max() if nn_added.size else 0)}  "
          f"mean={float(nn_added.mean() if nn_added.size else 0):.3f}  "
          f"sum={int(nn_added.sum() if nn_added.size else 0)}")
    print(f"    edges removed:  max={int(nn_removed.max() if nn_removed.size else 0)}  "
          f"mean={float(nn_removed.mean() if nn_removed.size else 0):.3f}  "
          f"sum={int(nn_removed.sum() if nn_removed.size else 0)}")

    print()
    _print_top(deltas, 'd_|F|', top_n=args.top_n,
               label='Δ|F|')
    print()
    _print_top(deltas, 'd_p_phase_maxabs', top_n=args.top_n,
               label='Δp_phase')
    print()
    _print_top(deltas, 'd_dual_vol_phase_maxabs', top_n=args.top_n,
               label='Δdual_vol_phase')
    print()
    _print_top(deltas, 'd_m_phase_maxabs', top_n=args.top_n,
               label='Δm_phase')

    # Save full per-vertex diff to JSON for follow-up plots
    out_path = os.path.join(
        _RESULTS,
        f'a5_step1_diff_{args.dim}d_{args.split_method}.json'
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        # convert tuple keys to strings, numpy arrays to lists
        json_deltas = []
        for d in deltas:
            jd = dict(d)
            jd['key'] = list(d['key'])
            jd['F_before'] = list(d['F_before'])
            jd['F_after'] = list(d['F_after'])
            json_deltas.append(jd)
        json.dump({
            'dim': args.dim,
            'split_method': args.split_method,
            'n_iface_before': diff['n_iface_before'],
            'n_iface_after': diff['n_iface_after'],
            'iface_count_changed': diff['iface_count_changed'],
            'deltas': json_deltas,
        }, f, indent=2, default=float)
    print(f"\n  Saved per-vertex diff JSON: {out_path}")


if __name__ == '__main__':
    main()
