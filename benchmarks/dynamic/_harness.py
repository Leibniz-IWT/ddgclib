"""Shared harness for the A.3 single-phase retopology bisection (1A / 1B.i / 1B.ii).

All three rungs use the **same physics** — a viscously decaying shear wave in
an x-periodic channel — and vary **only the retopology mode**, so any
difference in conservation attributes cleanly to retopology:

    1A   frozen-mesh        retopologize_fn=False        (no topology mgmt)
    1B.i skip_triangulation duals recomputed, fixed connectivity
    1B.ii full Delaunay     global periodic Delaunay rebuild every step

Setup
-----
Domain: rectangle ``L x h``, periodic in x (axis 0), no-slip walls at
``y = 0`` and ``y = h`` (frozen, ``u = 0`` there).

Initial condition: a single divergence-free shear mode

    u_x(y, 0) = U0 * sin(k y),   k = pi / h,   u_y = 0,   p = 0

which is an exact eigenmode of the viscous diffusion operator
``u_t = nu u_yy`` with ``nu = mu / rho``.  It generates no pressure
gradient (``du_x/dx = 0``), so the stress force is purely viscous and the
kinetic energy decays analytically as

    KE(t) / KE(0) = exp(-2 nu k^2 t).

Vertices advect in +x with the (small, decaying) velocity, deforming the
mesh and so exercising retopology.  Mass is Lagrangian-fixed per vertex
(``v.m = rho * dual_vol`` at setup), so total mass is conserved by
construction; **total dual volume is the discriminating invariant** — it
is recomputed from the dual cells each snapshot and is only conserved to
machine precision when the duals are not redrawn.

A KNOWN FINDING (see ``run_a3_bisection.py`` report): the documented
``skip_triangulation=True`` integrator flag is **silently bypassed** when
``periodic_axes`` is set — :func:`ddgclib.dynamic_integrators._integrators_dynamic._retopologize`
dispatches to :func:`retopologize_periodic` and returns before consulting
the flag, so a flag-based 1B.i is bit-identical to 1B.ii.  To make 1B.i a
genuine "recompute duals on fixed connectivity" rung we install
:func:`periodic_skip_retopo` as a custom ``retopologize_fn`` instead.
"""
from __future__ import annotations

import json
import os
from functools import partial
from typing import Any, Callable

import numpy as np

from ddgclib.data import compute_conservation, drift_fractions
from ddgclib.dynamic_integrators import euler
from ddgclib.geometry.domains import periodic_rectangle
from ddgclib.geometry.periodic import retopologize_periodic, wrap_positions
from ddgclib.operators.stress import cache_dual_volumes, dudt_i
from hyperct.ddg import compute_vd


# --- shared physical / numerical parameters -------------------------------

DEFAULTS: dict[str, Any] = {
    'refinement': 3,
    'L': 2.0,
    'h': 1.0,
    'rho0': 1.0,
    'U0': 0.15,
    'mu': 0.1,
    'dt': 2.0e-4,
    'n_steps': 400,
    'record_every': 10,
}

# Pass/fail tolerances.  Mass is a Lagrangian invariant and volume is only
# machine-precision-conserved when the duals are never redrawn, so volume
# drift is the bisection discriminator.
MASS_TOL = 1e-12          # |dM/M0| — all modes must pass
VOL_TOL = 1e-10           # |dV/V0| — machine-precision dual-volume conservation
KE_ANALYTIC_TOL = 0.15    # |KE/KE0 - analytic| relative; spatial truncation


# --- genuine periodic skip-triangulation ----------------------------------

def periodic_skip_retopo(HC, bV, dim, periodic_axes, domain_bounds) -> None:
    """Recompute duals on the **existing connectivity** (no retriangulation).

    This is the periodic analogue of ``_retopologize(..., skip_triangulation=
    True)``: wrap positions into the fundamental domain, then rebuild the
    barycentric dual mesh and cache dual volumes — but keep the primal
    connectivity and boundary set fixed.  Mirrors the dual path of
    :func:`retopologize_periodic` (``compute_vd`` -> ``cache_dual_volumes``)
    minus the disconnect/Delaunay steps, so it isolates the dual-volume
    refresh from the connectivity churn.
    """
    wrap_positions(HC, periodic_axes, domain_bounds)
    compute_vd(HC, method="barycentric")
    HC._periodic_axes = periodic_axes
    HC._periodic_bounds = domain_bounds
    cache_dual_volumes(HC, dim)


# --- setup ----------------------------------------------------------------

def build_decaying_sinusoid(refinement: int, L: float, h: float,
                            rho0: float, U0: float) -> tuple:
    """Construct the x-periodic channel + decaying shear-wave IC.

    Returns ``(HC, bV, domain_bounds, k)``.
    """
    res = periodic_rectangle(L=L, h=h, refinement=refinement,
                             periodic_axes=[0])
    HC, bV = res.HC, res.bV
    domain_bounds = res.metadata['domain_bounds']

    # Build valid periodic duals once at setup (merges the x-seam, resolves
    # ghost connectivity, caches dual volumes).  compute_vd on the raw
    # structured periodic mesh is degenerate, so this step is required even
    # for the frozen-mesh rung.
    retopologize_periodic(HC, bV, 2, [0], domain_bounds)

    k = np.pi / h
    for v in HC.V:
        y = v.x_a[1]
        v.u = np.array([U0 * np.sin(k * y), 0.0])
        v.p = 0.0
        v.m = rho0 * float(getattr(v, 'dual_vol', 0.0) or 0.0)
    return HC, bV, domain_bounds, k


# --- retopology mode -> integrator kwargs ---------------------------------

def mode_kwargs(mode: str, domain_bounds) -> dict[str, Any]:
    """Map a bisection mode name to ``euler`` keyword arguments.

    Modes
    -----
    ``frozen``            : ``retopologize_fn=False`` — no topology
                            management at all (1A).
    ``skip_triangulation``: genuine dual-only refresh on fixed connectivity
                            via :func:`periodic_skip_retopo` (1B.i).
    ``full_delaunay``     : global periodic Delaunay rebuild every step
                            (1B.ii, the production default).
    ``skip_flag``         : the *documented* ``skip_triangulation=True`` flag
                            with ``periodic_axes`` set — used only to
                            demonstrate it is inert (bypassed) under
                            periodicity.
    """
    if mode == 'frozen':
        return {'dim': 2, 'retopologize_fn': False}
    if mode == 'skip_triangulation':
        return {'dim': 2, 'retopologize_fn': partial(
            periodic_skip_retopo, periodic_axes=[0],
            domain_bounds=domain_bounds)}
    if mode == 'full_delaunay':
        return {'dim': 2, 'periodic_axes': [0], 'domain_bounds': domain_bounds}
    if mode == 'skip_flag':
        return {'dim': 2, 'skip_triangulation': True, 'periodic_axes': [0],
                'domain_bounds': domain_bounds}
    raise ValueError(f"unknown mode {mode!r}")


# --- run ------------------------------------------------------------------

def run_benchmark(mode: str, params: dict[str, Any] | None = None,
                  progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run one retopology mode and return metrics + per-step history.

    The result dict is JSON-serialisable (via :func:`result_as_jsonable`).
    """
    p = {**DEFAULTS, **(params or {})}
    HC, bV, domain_bounds, k = build_decaying_sinusoid(
        p['refinement'], p['L'], p['h'], p['rho0'], p['U0'])
    nu = p['mu'] / p['rho0']
    dudt = partial(dudt_i, dim=2, mu=p['mu'], HC=HC)
    kw = mode_kwargs(mode, domain_bounds)

    init = compute_conservation(HC, dim=2)
    ke0 = float(init['ke'])
    history: list[dict[str, Any]] = []

    def _record(step: int) -> None:
        diag = compute_conservation(HC, dim=2)
        t = step * p['dt']
        dr = drift_fractions(init, diag)
        history.append({
            'step': step,
            't': t,
            'ke': float(diag['ke']),
            'ke_ratio': float(diag['ke']) / ke0 if ke0 else 0.0,
            'ke_analytic': float(np.exp(-2.0 * nu * k * k * t)),
            'mass_drift': dr['mass_total'],
            'vol_drift': dr['volume_total'],
            'n_vertices': diag['n_vertices'],
            'u_max': diag['u_max'],
        })

    _record(0)

    def cb(step, t, HC_, bV_, diag):
        s = step + 1
        if s % p['record_every'] == 0 or s == p['n_steps']:
            _record(s)

    euler(HC, bV, dudt, dt=p['dt'], n_steps=p['n_steps'], callback=cb, **kw)

    # --- summary metrics ---------------------------------------------------
    vol_drifts = [r['vol_drift'] for r in history]
    mass_drifts = [r['mass_drift'] for r in history]
    ke_ratios = [r['ke_ratio'] for r in history]
    final = history[-1]

    # First step where volume conservation breaks machine precision.
    first_vol_fail = next(
        (r['step'] for r in history if r['vol_drift'] > VOL_TOL), None)

    # KE must decay monotonically (allow a tiny roundoff slack).
    ke_monotonic = all(
        b <= a + 1e-12 for a, b in zip(ke_ratios, ke_ratios[1:]))
    ke_analytic_relerr = (
        abs(final['ke_ratio'] - final['ke_analytic'])
        / final['ke_analytic'] if final['ke_analytic'] else float('inf'))

    metrics = {
        'max_mass_drift': max(mass_drifts),
        'max_vol_drift': max(vol_drifts),
        'final_vol_drift': final['vol_drift'],
        'first_vol_fail_step': first_vol_fail,
        'final_ke_ratio': final['ke_ratio'],
        'final_ke_analytic': final['ke_analytic'],
        'ke_analytic_relerr': ke_analytic_relerr,
        'ke_monotonic': ke_monotonic,
        'n_vertices_initial': history[0]['n_vertices'],
        'n_vertices_final': final['n_vertices'],
    }

    checks = {
        'mass_conserved': metrics['max_mass_drift'] < MASS_TOL,
        'volume_conserved': metrics['max_vol_drift'] < VOL_TOL,
        'ke_monotonic': ke_monotonic,
        'ke_matches_analytic': ke_analytic_relerr < KE_ANALYTIC_TOL,
    }
    # The bisection gate: mass + machine-precision volume conservation.
    passed = checks['mass_conserved'] and checks['volume_conserved']

    if progress:
        progress(f"  [{mode}] passed={passed}  max|dV/V0|="
                 f"{metrics['max_vol_drift']:.3e}  "
                 f"first_vol_fail_step={first_vol_fail}")

    return {
        'mode': mode,
        'params': p,
        'passed': passed,
        'checks': checks,
        'metrics': metrics,
        'history': history,
    }


def result_as_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """No numpy in the result, so a shallow copy is JSON-ready already."""
    return result


# --- presentation ---------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def save_result(result: dict[str, Any], filename: str) -> str:
    """Write *result* as JSON under ``benchmarks/dynamic/results/``."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result_as_jsonable(result), f, indent=2)
    return path


def print_summary(result: dict[str, Any]) -> None:
    """Print a one-rung conservation table + PASS/FAIL verdict."""
    m, c = result['metrics'], result['checks']
    print(f"\n=== {result['mode']} "
          f"({'PASS' if result['passed'] else 'FAIL'}) ===")
    print(f"{'step':>6} {'t':>8} {'KE/KE0':>9} {'analytic':>9} "
          f"{'|dM/M0|':>10} {'|dV/V0|':>10} {'nV':>5}")
    for r in result['history']:
        print(f"{r['step']:>6} {r['t']:>8.4f} {r['ke_ratio']:>9.4f} "
              f"{r['ke_analytic']:>9.4f} {r['mass_drift']:>10.2e} "
              f"{r['vol_drift']:>10.2e} {r['n_vertices']:>5}")
    print(f"  max|dM/M0|={m['max_mass_drift']:.3e}  "
          f"max|dV/V0|={m['max_vol_drift']:.3e}  "
          f"first_vol_fail_step={m['first_vol_fail_step']}")
    print(f"  KE/KE0(final)={m['final_ke_ratio']:.4f} vs "
          f"analytic {m['final_ke_analytic']:.4f} "
          f"(rel err {m['ke_analytic_relerr']:.1%}, "
          f"monotonic={m['ke_monotonic']})")
    print(f"  checks: {c}")
