#!/usr/bin/env python3
"""A.5 static-droplet frozen-mesh bisection.

Per ``.claude/plans/please-do-some-deep-vectorized-tarjan.md`` (Phase A.5):
attribute the residual interface-vertex force on the static
circular/spherical droplet (~3.8e-3 in 2D, ~8.5e-5 in 3D) to either:

  A.5.a  Pure curvature-stencil residual on a fixed curved mesh
         (retopology disabled, mesh held fixed, evaluate
         ``multiphase_stress_force`` on every interface vertex once).

  A.5.b  Pure retopology-induced residual with no velocity-driven
         geometry change (retopology as normal, ``v.u = 0`` forced
         every step, record max |F| per step over ~100 steps).

Uses ``euler`` (forward Euler) for A.5.b so the position update reads
the OLD velocity — zeroing ``u`` in the callback then freezes geometry
exactly (``x_{n+1} = x_n + dt * 0 = x_n``) while retopology still
executes at the start of every step.

Records conservation diagnostics (``compute_conservation``) alongside
max |F| so any KE / mass / volume drift is visible at the same cadence.

Usage
-----
    python cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py
    python cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py --skip-3d
    python cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py --n-steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._params import (
    R0, l, rho_d, rho_o, mu_d, mu_o, gamma, K_d, K_o,
    L_domain, n_refine_outer, n_refine_droplet,
)
from cases_dynamic.oscillating_droplet.src._setup import (
    setup_oscillating_droplet,
)
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.dynamic_integrators import euler
from ddgclib.data import compute_conservation


_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_CASE_DIR, 'results_a5_bisection')


# Full-dynamic baselines quoted in the stabilisation roadmap (D4 /
# docs/3d_multiphase_interface_pressure_fix.md): post-April-2026
# own-phase pressure fix, these are the residual max |F| on interface
# vertices at static equilibrium with full retopology + integration.
BASELINE_2D = 3.8e-3
BASELINE_3D = 8.5e-5


def _interface_vertices(HC):
    return [v for v in HC.V if getattr(v, 'is_interface', False)]


def _max_interface_force(HC, dim, mps, curvature_path: str = 'integrated'):
    iface = _interface_vertices(HC)
    max_F = 0.0
    mean_F = 0.0
    n = 0
    for v in iface:
        F = multiphase_stress_force(
            v, dim=dim, mps=mps, HC=HC,
            curvature_path=curvature_path,
        )
        mag = float(np.linalg.norm(F))
        if mag > max_F:
            max_F = mag
        mean_F += mag
        n += 1
    mean_F = mean_F / n if n else 0.0
    return max_F, mean_F, n


def _compute_dt(HC, dim, c_s):
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15
    )
    dt_acoustic = 0.25 * dx_min / c_s
    dt_capillary = 0.5 * np.sqrt(rho_d * dx_min ** 3 / gamma) if gamma > 0 else 1.0
    return min(dt_acoustic, dt_capillary), dx_min


def run_a5a(
    dim: int, refinement_outer: int, refinement_droplet: int,
    split_method: str = 'neighbour_count',
    curvature_path: str = 'integrated',
) -> dict:
    """A.5.a — frozen mesh, no retopology, evaluate F once."""
    print(f"\n{'=' * 70}")
    print(f"A.5.a ({dim}D) — frozen mesh, retopology DISABLED, "
          f"split_method={split_method!r}, "
          f"curvature_path={curvature_path!r}")
    print('=' * 70)

    HC, bV, mps, bc_set, dudt_fn, _retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=0.0, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            split_method=split_method,
        )

    n_verts = sum(1 for _ in HC.V)
    iface = _interface_vertices(HC)
    n_iface = len(iface)
    print(f"  Mesh: {n_verts} vertices, {n_iface} interface")

    # Interface radii — sanity
    radii = np.array([np.linalg.norm(v.x_a[:dim]) for v in iface])
    r_dev = float(np.max(np.abs(radii - R0)))
    print(f"  Interface radius deviation from R0: {r_dev:.3e}")

    t0 = time.perf_counter()
    max_F, mean_F, n = _max_interface_force(
        HC, dim, mps, curvature_path=curvature_path,
    )
    dt_eval = time.perf_counter() - t0

    cons = compute_conservation(HC, dim=dim)
    baseline = BASELINE_2D if dim == 2 else BASELINE_3D
    ratio = max_F / baseline if baseline > 0 else float('inf')

    print(f"  Interface max |F|:   {max_F:.4e}")
    print(f"  Interface mean |F|:  {mean_F:.4e}")
    print(f"  Full-dynamic baseline: {baseline:.4e}")
    print(f"  ratio max|F| / baseline: {ratio:.3f}")
    print(f"  KE = {cons['ke']:.4e}  mass_total = {cons['mass_total']:.6e}"
          f"  volume_total = {cons['volume_total']:.6e}")
    print(f"  (eval time: {dt_eval * 1000:.1f} ms)")

    return {
        'flavour': 'A.5.a',
        'dim': dim,
        'n_verts': n_verts,
        'n_iface': n_iface,
        'interface_radius_deviation': r_dev,
        'max_abs_F': max_F,
        'mean_abs_F': mean_F,
        'baseline': baseline,
        'ratio_to_baseline': ratio,
        'ke': cons['ke'],
        'mass_total': cons['mass_total'],
        'volume_total': cons['volume_total'],
        'mass_phase': (
            cons['mass_phase'].tolist() if 'mass_phase' in cons else None
        ),
        'volume_phase': (
            cons['volume_phase'].tolist() if 'volume_phase' in cons else None
        ),
    }


def run_a5b(
    dim: int,
    refinement_outer: int,
    refinement_droplet: int,
    n_steps: int = 100,
    split_method: str = 'neighbour_count',
    redistribute_mass: bool = False,
    curvature_path: str = 'integrated',
    displacement_eps: float | None = None,
) -> dict:
    """A.5.b — retopology ON, velocity forced to zero every step.

    Parameters
    ----------
    split_method : {'neighbour_count', 'exact'}
        Per-phase dual-volume split policy used inside
        ``_retopologize_multiphase``.  ``'neighbour_count'`` matches the
        production default; ``'exact'`` swaps in the geometric split
        (Phase 1 of the 2026-04-28 stabilisation plan — diagnoses
        whether neighbour-count majority-vote on cross-phase reconnection
        is the dominant retopology bug in 3D).
    """
    print(f"\n{'=' * 70}")
    print(f"A.5.b ({dim}D) — retopology ON, u forced to 0 every step "
          f"({n_steps} steps), split_method={split_method!r}, "
          f"redistribute_mass={redistribute_mass}, "
          f"curvature_path={curvature_path!r}, "
          f"displacement_eps={displacement_eps!r}")
    print('=' * 70)

    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = \
        setup_oscillating_droplet(
            dim=dim, R0=R0, epsilon=0.0, l=l,
            rho_d=rho_d, rho_o=rho_o, mu_d=mu_d, mu_o=mu_o,
            gamma=gamma, K_d=K_d, K_o=K_o, L_domain=L_domain,
            refinement_outer=refinement_outer,
            refinement_droplet=refinement_droplet,
            split_method=split_method,
            redistribute_mass=redistribute_mass,
        )

    n_verts0 = sum(1 for _ in HC.V)
    n_iface0 = len(_interface_vertices(HC))
    print(f"  Initial mesh: {n_verts0} vertices, {n_iface0} interface")

    c_s = float(np.sqrt(K_d / rho_d))
    dt, dx_min = _compute_dt(HC, dim, c_s)
    print(f"  dt = {dt:.3e}   dx_min = {dx_min:.3e}")
    if displacement_eps is not None:
        print(f"  displacement_eps = {displacement_eps:.3e}  "
              f"(ratio eps/dx_min = {displacement_eps / dx_min:.3e})")

    # Records per step
    step_max_F: list[float] = []
    step_mean_F: list[float] = []
    step_ke_pre_zero: list[float] = []
    step_mass: list[float] = []
    step_volume: list[float] = []
    step_n_iface: list[int] = []
    step_n_verts: list[int] = []

    # t = 0 snapshot (before any step)
    maxF0, meanF0, _ = _max_interface_force(
        HC, dim, mps, curvature_path=curvature_path,
    )
    cons0 = compute_conservation(HC, dim=dim)
    step_max_F.append(maxF0)
    step_mean_F.append(meanF0)
    step_ke_pre_zero.append(cons0['ke'])
    step_mass.append(cons0['mass_total'])
    step_volume.append(cons0['volume_total'])
    step_n_iface.append(n_iface0)
    step_n_verts.append(n_verts0)
    print(f"  t=0.0e+00  max|F|={maxF0:.4e}  mean|F|={meanF0:.4e}  "
          f"KE={cons0['ke']:.3e}  nV={n_verts0}  nI={n_iface0}")

    # Zero-velocity enforcement callback. Runs AFTER the BC step in
    # euler(), so zeroing u here means the NEXT step starts with u=0
    # and the OLD-velocity-based position update freezes geometry.
    def zero_u_callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        # Record BEFORE zeroing u so KE reflects the intra-step impulse.
        maxF, meanF, _ = _max_interface_force(
            HC_cb, dim, mps, curvature_path=curvature_path,
        )
        cons = compute_conservation(HC_cb, dim=dim)
        step_max_F.append(maxF)
        step_mean_F.append(meanF)
        step_ke_pre_zero.append(cons['ke'])
        step_mass.append(cons['mass_total'])
        step_volume.append(cons['volume_total'])
        step_n_iface.append(
            sum(1 for v in HC_cb.V if getattr(v, 'is_interface', False))
        )
        step_n_verts.append(sum(1 for _ in HC_cb.V))

        # Zero every vertex's velocity so the next step's OLD-velocity
        # position update is trivial (x doesn't move).
        for v in HC_cb.V:
            v.u[:] = 0.0

        if (step + 1) % max(1, n_steps // 10) == 0 or step == 0:
            print(f"  step={step + 1:4d}  t={t:.3e}  max|F|={maxF:.4e}  "
                  f"mean|F|={meanF:.4e}  KE={cons['ke']:.3e}  "
                  f"nV={step_n_verts[-1]}  nI={step_n_iface[-1]}")

    t0 = time.perf_counter()
    euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, callback=zero_u_callback,
        retopologize_fn=retopo_fn,
        remesh_mode=params['remesh_mode'],
        remesh_kwargs=params['remesh_kwargs'],
        displacement_eps=displacement_eps,
    )
    wall = time.perf_counter() - t0

    max_F_arr = np.array(step_max_F)
    mean_F_arr = np.array(step_mean_F)

    maxF_overall = float(max_F_arr.max())
    maxF_end = float(max_F_arr[-1])
    meanF_end = float(mean_F_arr[-1])
    mass_rel = abs(step_mass[-1] - step_mass[0]) / abs(step_mass[0])
    vol_rel = abs(step_volume[-1] - step_volume[0]) / abs(step_volume[0])

    baseline = BASELINE_2D if dim == 2 else BASELINE_3D
    ratio_end = maxF_end / baseline
    ratio_peak = maxF_overall / baseline

    print(f"\n  Over {n_steps} steps (wall {wall:.1f}s):")
    print(f"    max|F| peak over run:   {maxF_overall:.4e}"
          f"  (×{ratio_peak:.3f} of baseline {baseline:.4e})")
    print(f"    max|F| at final step:   {maxF_end:.4e}"
          f"  (×{ratio_end:.3f} of baseline)")
    print(f"    mean|F| at final step:  {meanF_end:.4e}")
    print(f"    |dM/M0|:                {mass_rel:.3e}")
    print(f"    |dV/V0|:                {vol_rel:.3e}")
    print(f"    vertex count drift:     {step_n_verts[0]} -> "
          f"{step_n_verts[-1]}")
    print(f"    interface count drift:  {step_n_iface[0]} -> "
          f"{step_n_iface[-1]}")

    return {
        'flavour': 'A.5.b',
        'dim': dim,
        'n_steps': n_steps,
        'dt': dt,
        'split_method': split_method,
        'redistribute_mass': redistribute_mass,
        'displacement_eps': displacement_eps,
        'max_abs_F_peak': maxF_overall,
        'max_abs_F_end': maxF_end,
        'mean_abs_F_end': meanF_end,
        'max_abs_F_history': max_F_arr.tolist(),
        'mean_abs_F_history': mean_F_arr.tolist(),
        'ke_history': step_ke_pre_zero,
        'mass_history': step_mass,
        'volume_history': step_volume,
        'n_verts_history': step_n_verts,
        'n_iface_history': step_n_iface,
        'baseline': baseline,
        'ratio_peak_to_baseline': ratio_peak,
        'ratio_end_to_baseline': ratio_end,
        'mass_rel_drift': mass_rel,
        'volume_rel_drift': vol_rel,
    }


def attribute(a: dict, b: dict, baseline: float) -> dict:
    """Attribute the residual to curvature stencil, retopology, or both."""
    fa = a['max_abs_F']                  # A.5.a — curvature stencil alone
    fb = b['max_abs_F_peak']             # A.5.b — retopology alone (peak)
    # How much of baseline each bucket already explains
    share_a = fa / baseline
    share_b = fb / baseline

    # Classification heuristic
    #  - if share_a >= 0.5:  curvature stencil dominates
    #  - elif share_b >= 0.5: retopology dominates
    #  - if both >= 0.3: mixed
    #  - else: neither bucket alone accounts — instability is from
    #          the full dynamic loop (coupling) and can only be
    #          reached by the dynamic path itself
    if share_a >= 0.5 and share_b < share_a * 0.5:
        verdict = 'curvature-dominant'
    elif share_b >= 0.5 and share_a < share_b * 0.5:
        verdict = 'retopology-dominant'
    elif share_a >= 0.3 and share_b >= 0.3:
        verdict = 'mixed'
    elif max(share_a, share_b) < 0.3:
        verdict = 'neither-bucket-captures-baseline'
    else:
        verdict = 'mixed'

    return {
        'baseline': baseline,
        'curvature_alone_max_abs_F': fa,
        'retopology_alone_max_abs_F_peak': fb,
        'share_curvature': share_a,
        'share_retopology': share_b,
        'verdict': verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-3d', action='store_true',
                        help='Skip the 3D probes (useful for a quick 2D-only run)')
    parser.add_argument('--n-steps', type=int, default=100,
                        help='Number of steps for A.5.b (default 100)')
    parser.add_argument('--refine-3d-outer', type=int, default=2)
    parser.add_argument('--refine-3d-droplet', type=int, default=2)
    parser.add_argument(
        '--split-method', choices=['neighbour_count', 'exact'],
        default='neighbour_count',
        help='Per-phase dual-volume split policy used in '
             '_retopologize_multiphase for the A.5.b runs '
             '(default: neighbour_count, matches production setup).',
    )
    parser.add_argument(
        '--redistribute-mass', action='store_true', default=False,
        help='Enable per-phase pressure-preserving mass redistribution '
             'in _retopologize_multiphase (Phase 2 finding: Lagrangian '
             'm_phase frozen against shifting dual_vol_phase causes the '
             'rho/p_phase drift driving the 3D step-1 |F| jump). '
             'Default: False (matches production setup).',
    )
    parser.add_argument(
        '--curvature-path', choices=['integrated', 'csf_dual', 'stokes'],
        default='integrated',
        help='Surface-tension curvature stencil for the |F| evaluation: '
             "'integrated' (default) — FTC in 2D / cotangent-Heron in "
             '3D, exact for piecewise-linear / triangulated interfaces; '
             "'csf_dual' — Continuum-Surface-Force form aligned with "
             'the dual face-area vector S_inner used by the per-phase '
             "pressure flux; 'stokes' — Stokes-theorem boundary integral "
             'on the barycentric dual cell of v_i restricted to interface '
             'triangles (3D only; 2D delegates to the existing FTC form). '
             'Per Tier 2B step 1 audit (2026-05-06): the '
             "static-droplet residual under 'integrated' is provably "
             'first-order discretization error of the polygon mesh '
             'and converges as O(h) (32→2.37e-3, 64→1.11e-3, 128→5.4e-4 '
             "in 2D); 'stokes' is the Probe 2 (2026-05-27) integrated "
             'rewrite targeting the 3D pointwise truncation residual.',
    )
    parser.add_argument(
        '--displacement-eps', type=float, default=None,
        help='Probe 5 (2026-05-28): skip-retopology gate threshold. '
             'When set to a positive value, _do_retopologize short-circuits '
             'when every vertex moved less than EPS since the last call. '
             'On A.5.b with u=0 forced, eps small (e.g. 1e-10) skips every '
             'retopology call and the run collapses to A.5.a. On real '
             'dynamic runs use eps proportional to h_min '
             '(suggested first cut: 1e-4 * dx_min). Default: None (gate '
             'disabled, every-step retopology — matches production).',
    )
    parser.add_argument(
        '--results-suffix', default='',
        help='Suffix appended to the output JSON filename '
             '(useful for keeping baseline + variant probe runs side-by-side).',
    )
    args = parser.parse_args()

    os.makedirs(_RESULTS, exist_ok=True)

    all_results = {}

    # ---- 2D ----
    print('\n' + '#' * 70)
    print('# A.5 BISECTION — 2D')
    print('#' * 70)
    a2d = run_a5a(dim=2, refinement_outer=n_refine_outer,
                  refinement_droplet=n_refine_droplet,
                  split_method=args.split_method,
                  curvature_path=args.curvature_path)
    b2d = run_a5b(dim=2, refinement_outer=n_refine_outer,
                  refinement_droplet=n_refine_droplet,
                  n_steps=args.n_steps,
                  split_method=args.split_method,
                  redistribute_mass=args.redistribute_mass,
                  curvature_path=args.curvature_path,
                  displacement_eps=args.displacement_eps)
    attr2d = attribute(a2d, b2d, BASELINE_2D)
    all_results['2D'] = {'a': a2d, 'b': b2d, 'attribution': attr2d}

    # ---- 3D ----
    if not args.skip_3d:
        try:
            print('\n' + '#' * 70)
            print('# A.5 BISECTION — 3D')
            print('#' * 70)
            a3d = run_a5a(dim=3,
                          refinement_outer=args.refine_3d_outer,
                          refinement_droplet=args.refine_3d_droplet,
                          split_method=args.split_method,
                          curvature_path=args.curvature_path)
            b3d = run_a5b(dim=3,
                          refinement_outer=args.refine_3d_outer,
                          refinement_droplet=args.refine_3d_droplet,
                          n_steps=args.n_steps,
                          split_method=args.split_method,
                          redistribute_mass=args.redistribute_mass,
                          curvature_path=args.curvature_path,
                          displacement_eps=args.displacement_eps)
            attr3d = attribute(a3d, b3d, BASELINE_3D)
            all_results['3D'] = {'a': a3d, 'b': b3d, 'attribution': attr3d}
        except Exception as exc:
            import traceback
            print(f"\n[!!] 3D probe failed: {exc}")
            traceback.print_exc()
            all_results['3D'] = {'error': repr(exc)}
    else:
        print('\n(3D probes skipped via --skip-3d)')

    # ---- Summary ----
    print('\n' + '=' * 70)
    print('A.5 BISECTION SUMMARY')
    print('=' * 70)

    def _fmt_row(tag, attr):
        if attr is None:
            return f"  {tag:5s}  (skipped / failed)"
        return (
            f"  {tag:5s}  baseline={attr['baseline']:.4e}  "
            f"A.5.a={attr['curvature_alone_max_abs_F']:.4e} "
            f"(×{attr['share_curvature']:.3f})  "
            f"A.5.b={attr['retopology_alone_max_abs_F_peak']:.4e} "
            f"(×{attr['share_retopology']:.3f})  "
            f"verdict={attr['verdict']}"
        )

    print(_fmt_row('2D', all_results['2D']['attribution']))
    if '3D' in all_results and 'attribution' in all_results['3D']:
        print(_fmt_row('3D', all_results['3D']['attribution']))

    suffix = (f"_{args.results_suffix}" if args.results_suffix else '')
    out_path = os.path.join(_RESULTS, f'a5_bisection{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved full results to {out_path}")


if __name__ == '__main__':
    main()
