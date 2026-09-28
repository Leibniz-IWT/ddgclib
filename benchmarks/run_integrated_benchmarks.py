#!/usr/bin/env python
"""
Benchmark runner for integrated gradient validation.

Compares DDG integrated gradient operators against analytically
integrated solutions over the exact same dual cell domains.

Tests all combinations of:
- Dual mesh method: barycentric, circumcentric
- Polygon formulation: barycentric (duals only), barycentric_dual_p_ij (+ edge midpoints)
- Mesh type: symmetric, jittered (seed=42)
- Field type: linear (expect machine precision), non-linear (expect convergence)

Usage::

    python benchmarks/run_integrated_benchmarks.py
    python benchmarks/run_integrated_benchmarks.py --dim 2 --refine 2
    python benchmarks/run_integrated_benchmarks.py --linear-only
    python benchmarks/run_integrated_benchmarks.py --convergence
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmarks._integrated_benchmark_cases import (
    LINEAR_BENCHMARKS,
    NONLINEAR_BENCHMARKS,
    ALL_GRADIENT_BENCHMARKS,
    LinearScalar1D, LinearVector1D,
    LinearScalar2D, LinearVector2D,
    LinearScalar3D, LinearVector3D,
    QuadraticScalar1D, QuadraticVector1D,
    QuadraticScalar2D, QuadraticVector2D,
    CubicScalar2D, CubicVector2D,
    TrigScalar2D,
    PoiseuilleVector2D,
    QuadraticScalar3D, QuadraticVector3D,
)


# ---------------------------------------------------------------------------
# Configuration: what to compare
# ---------------------------------------------------------------------------

# Dual mesh methods (passed to compute_vd)
DUAL_METHODS = ["barycentric", "circumcentric"]

# Polygon formulations (how the dual cell polygon is constructed)
POLYGON_METHODS = ["barycentric_dual_p_ij", "barycentric"]

# Mesh types
SEEDS = [None, 42]  # None = symmetric, 42 = jittered

# Refinement levels for convergence study
CONVERGENCE_REFINES = [1, 2, 3]


def _fmt_err(err: float) -> str:
    """Format error with color-coding for terminal output."""
    if err < 1e-13:
        return f"\033[92m{err:.2e}\033[0m"  # green = machine precision
    elif err < 1e-3:
        return f"\033[93m{err:.2e}\033[0m"  # yellow = good convergence
    else:
        return f"\033[91m{err:.2e}\033[0m"  # red = large error


def _seed_label(seed):
    return "symmetric" if seed is None else f"jittered(seed={seed})"


# ---------------------------------------------------------------------------
# Method comparison table
# ---------------------------------------------------------------------------

def run_method_comparison(dim: int = 2, n_refine: int = 1):
    """Compare all method combinations for each benchmark case."""
    print("=" * 90)
    print(f"  METHOD COMPARISON — {dim}D, refine={n_refine}")
    print("=" * 90)

    # Select benchmarks compatible with this dimension
    cases_to_run = []
    for BenchClass in ALL_GRADIENT_BENCHMARKS:
        try:
            b = BenchClass(dim=dim)
            if b.field_type == "scalar":
                b.f_callable(np.zeros(dim))
            else:
                b.u_callable(np.zeros(dim))
            cases_to_run.append(BenchClass)
        except (IndexError, ValueError, NotImplementedError):
            continue

    # Header
    combos = []
    for dm in DUAL_METHODS:
        for pm in POLYGON_METHODS:
            for seed in SEEDS:
                combos.append((dm, pm, seed))

    header = f"{'Case':<28}"
    for dm, pm, seed in combos:
        pm_short = "p_ij" if pm == "barycentric_dual_p_ij" else "bary"
        s_short = "sym" if seed is None else "jit"
        dm_short = "B" if dm == "barycentric" else "C"
        header += f" {dm_short}/{pm_short}/{s_short:>3}"
    print(header)
    print("-" * len(header))

    for BenchClass in cases_to_run:
        row = f"{BenchClass.__name__:<28}"
        for dm, pm, seed in combos:
            try:
                bench = BenchClass(
                    dim=dim,
                    dual_method=dm,
                    polygon_method=pm,
                    n_refine=n_refine,
                    seed=seed,
                )
                s = bench.run()
                row += f" {_fmt_err(s['max_abs_error']):>14}"
            except Exception as e:
                row += f" {'FAIL':>14}"
        print(row)

    print()
    print("Legend: B=barycentric, C=circumcentric, p_ij=with edge midpoints, bary=duals only, sym=symmetric, jit=jittered")
    print()


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def run_convergence_study(dim: int = 2, dual_method: str = "barycentric",
                          polygon_method: str = "barycentric_dual_p_ij"):
    """Show error convergence with mesh refinement for non-linear fields."""
    print("=" * 80)
    pm_short = "p_ij" if polygon_method == "barycentric_dual_p_ij" else "bary"
    print(f"  CONVERGENCE STUDY — {dim}D, {dual_method}/{pm_short}")
    print("=" * 80)

    cases_to_run = []
    for BenchClass in NONLINEAR_BENCHMARKS:
        try:
            b = BenchClass(dim=dim)
            if b.field_type == "scalar":
                b.f_callable(np.zeros(dim))
            else:
                b.u_callable(np.zeros(dim))
            cases_to_run.append(BenchClass)
        except (IndexError, ValueError, NotImplementedError):
            continue

    for seed in [None, 42]:
        print(f"\n  Mesh: {_seed_label(seed)}")
        header = f"  {'Case':<28}"
        for r in CONVERGENCE_REFINES:
            header += f"  refine={r:>1}"
        header += "   rate"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for BenchClass in cases_to_run:
            errors = []
            row = f"  {BenchClass.__name__:<28}"
            for r in CONVERGENCE_REFINES:
                try:
                    bench = BenchClass(
                        dim=dim,
                        dual_method=dual_method,
                        polygon_method=polygon_method,
                        n_refine=r,
                        seed=seed,
                    )
                    s = bench.run()
                    err = s["max_abs_error"]
                    errors.append(err)
                    row += f"  {_fmt_err(err):>14}"
                except Exception:
                    errors.append(np.inf)
                    row += f"  {'FAIL':>14}"

            # Compute convergence rate
            if len(errors) >= 2 and errors[0] > 1e-15 and errors[-1] > 1e-15:
                rate = np.log(errors[0] / errors[-1]) / np.log(
                    2 ** (CONVERGENCE_REFINES[-1] - CONVERGENCE_REFINES[0])
                )
                row += f"   {rate:.1f}"
            else:
                row += f"   —"
            print(row)

    print()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def run_linear_precision_check(dim: int = 2):
    """Verify all linear fields give machine precision on all methods."""
    print("=" * 80)
    print(f"  LINEAR PRECISION CHECK — {dim}D (expect all < 1e-13)")
    print("=" * 80)

    cases_to_run = []
    for BenchClass in LINEAR_BENCHMARKS:
        try:
            b = BenchClass(dim=dim)
            if b.field_type == "scalar":
                b.f_callable(np.zeros(dim))
            else:
                b.u_callable(np.zeros(dim))
            cases_to_run.append(BenchClass)
        except (IndexError, ValueError, NotImplementedError):
            continue

    all_pass = True
    for BenchClass in cases_to_run:
        for dm in DUAL_METHODS:
            for pm in POLYGON_METHODS:
                for seed in SEEDS:
                    try:
                        bench = BenchClass(
                            dim=dim,
                            dual_method=dm,
                            polygon_method=pm,
                            n_refine=1,
                            seed=seed,
                        )
                        s = bench.run()
                        err = s["max_abs_error"]
                        status = "PASS" if err < 1e-13 else "FAIL"
                        if err >= 1e-13:
                            all_pass = False
                        pm_s = "p_ij" if pm == "barycentric_dual_p_ij" else "bary"
                        print(f"  {status}  {BenchClass.__name__:<24} "
                              f"{dm:>13}/{pm_s:<4} {_seed_label(seed):>16}  "
                              f"err={_fmt_err(err)}")
                    except Exception as e:
                        all_pass = False
                        print(f"  FAIL  {BenchClass.__name__:<24} — {e}")

    print()
    if all_pass:
        print("  \033[92mAll linear precision checks PASSED\033[0m")
    else:
        print("  \033[91mSome linear precision checks FAILED\033[0m")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run integrated gradient validation benchmarks"
    )
    parser.add_argument("--dim", type=int, default=2,
                        help="Spatial dimension (default: 2)")
    parser.add_argument("--refine", type=int, default=1,
                        help="Refinement level for method comparison (default: 1)")
    parser.add_argument("--linear-only", action="store_true",
                        help="Only run linear precision checks")
    parser.add_argument("--convergence", action="store_true",
                        help="Only run convergence study")
    parser.add_argument("--comparison", action="store_true",
                        help="Only run method comparison table")
    args = parser.parse_args()

    run_all = not (args.linear_only or args.convergence or args.comparison)

    if run_all or args.linear_only:
        run_linear_precision_check(dim=args.dim)

    if run_all or args.comparison:
        run_method_comparison(dim=args.dim, n_refine=args.refine)

    if run_all or args.convergence:
        for dm in DUAL_METHODS:
            for pm in POLYGON_METHODS:
                run_convergence_study(
                    dim=args.dim, dual_method=dm, polygon_method=pm
                )


if __name__ == "__main__":
    main()
