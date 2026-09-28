"""A.3 bisection — run 1A / 1B.i / 1B.ii and report which retopology mode
first fails the conservation gate on the periodic decaying-sinusoid setup.

Bisection ladder (each rung adds one source of complexity):

    frozen-mesh        -> dual-only refresh -> full Delaunay rebuild

Reports the first rung to break machine-precision volume conservation and
writes ``results/a3_bisection.json`` + ``DASHBOARD.md``.

Run::

    python benchmarks/dynamic/run_a3_bisection.py
"""
from __future__ import annotations

import os
import sys

from _harness import RESULTS_DIR, VOL_TOL, run_benchmark, save_result

# Ordered rungs: name -> (mode, label).
LADDER = [
    ('1A', 'frozen', 'frozen-mesh (retopologize_fn=False)'),
    ('1B.i', 'skip_triangulation', 'skip_triangulation (dual refresh, fixed connectivity)'),
    ('1B.ii', 'full_delaunay', 'full periodic Delaunay rebuild'),
]


def main() -> int:
    print("A.3 retopology bisection — periodic single-phase decaying sinusoid\n")
    results = {}
    for rung, mode, label in LADDER:
        print(f"running {rung}: {label} ...")
        results[rung] = run_benchmark(
            mode, progress=lambda s: print(s))

    # --- bisection table ---------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'rung':>6} {'mode':>20} {'pass':>5} {'max|dM/M0|':>12} "
          f"{'max|dV/V0|':>12} {'1st vol-fail':>12} {'KE relerr':>10}")
    for rung, _mode, _label in LADDER:
        r = results[rung]
        m = r['metrics']
        fvf = m['first_vol_fail_step']
        print(f"{rung:>6} {r['mode']:>20} {str(r['passed']):>5} "
              f"{m['max_mass_drift']:>12.2e} {m['max_vol_drift']:>12.2e} "
              f"{str(fvf):>12} {m['ke_analytic_relerr']:>10.1%}")

    # --- verdict -----------------------------------------------------------
    first_fail = next(
        (rung for rung, _m, _l in LADDER if not results[rung]['passed']), None)
    print("\n" + "=" * 78)
    if first_fail is None:
        verdict = "All rungs pass the conservation gate."
    else:
        r = results[first_fail]
        verdict = (
            f"FIRST FAILURE: {first_fail} "
            f"({r['mode']}) — max|dV/V0|={r['metrics']['max_vol_drift']:.3e} "
            f"exceeds VOL_TOL={VOL_TOL:.0e}.")
    print(verdict)

    # --- persist -----------------------------------------------------------
    summary = {
        'verdict': verdict,
        'first_fail_rung': first_fail,
        'vol_tol': VOL_TOL,
        'rungs': {rung: {
            'mode': results[rung]['mode'],
            'passed': results[rung]['passed'],
            'metrics': results[rung]['metrics'],
            'checks': results[rung]['checks'],
        } for rung, _m, _l in LADDER},
    }
    save_result(summary, 'a3_bisection.json')
    for rung, _m, _l in LADDER:
        save_result(results[rung], f'a3_{rung.replace(".", "_")}.json')
    _write_dashboard(summary)

    # Aggregator always exits 0 — the "failure" of 1B is the expected,
    # informative bisection result, not a harness error.
    return 0


def _write_dashboard(summary: dict) -> None:
    lines = [
        "# Dynamic Benchmark Dashboard",
        "",
        "## A.3 — single-phase retopology bisection (decaying sinusoid)",
        "",
        f"**Verdict:** {summary['verdict']}",
        "",
        f"Gate: mass + machine-precision volume conservation "
        f"(`VOL_TOL = {summary['vol_tol']:.0e}`).",
        "",
        "| rung | mode | pass | max\\|dM/M0\\| | max\\|dV/V0\\| | 1st vol-fail step | KE rel err |",
        "|------|------|------|------------|------------|-------------------|-----------|",
    ]
    for rung in ('1A', '1B.i', '1B.ii'):
        r = summary['rungs'][rung]
        m = r['metrics']
        lines.append(
            f"| {rung} | `{r['mode']}` | {'PASS' if r['passed'] else 'FAIL'} "
            f"| {m['max_mass_drift']:.2e} | {m['max_vol_drift']:.2e} "
            f"| {m['first_vol_fail_step']} | {m['ke_analytic_relerr']:.1%} |")
    lines += [
        "",
        "### Notes",
        "",
        "- **frozen-mesh** conserves volume to machine precision because the "
        "duals are never redrawn (mass is Lagrangian-fixed per vertex).",
        "- **skip_triangulation** (genuine dual-only refresh on fixed "
        "connectivity) is the first rung to break machine-precision volume "
        "conservation: rebuilding the barycentric duals on the deformed mesh "
        "shifts total dual volume by ~O(1%).",
        "- **full Delaunay** roughly triples that drift via connectivity "
        "churn on the near-structured cloud and additionally perturbs KE.",
        "- The documented `skip_triangulation=True` integrator flag is "
        "**silently bypassed** when `periodic_axes` is set "
        "(`_retopologize` returns early on the periodic dispatch), so a "
        "flag-based 1B.i is bit-identical to 1B.ii. The genuine rung here "
        "uses a custom `retopologize_fn` (`periodic_skip_retopo`).",
        "",
        "Regenerate: `python benchmarks/dynamic/run_a3_bisection.py`",
        "",
    ]
    path = os.path.join(os.path.dirname(__file__), 'DASHBOARD.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    sys.exit(main())
