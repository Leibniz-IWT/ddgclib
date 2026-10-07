#!/usr/bin/env python
"""
CLI runner for the int_bench validation suite.

Usage
-----

    python -m int_bench.run_benchmarks                          # full suite
    python -m int_bench.run_benchmarks --dims 1 2 3
    python -m int_bench.run_benchmarks --controls primal dual
    python -m int_bench.run_benchmarks --jitter 42
"""
from __future__ import annotations

import argparse
import sys

from .benchmarks import print_results, run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run integrated gradient/Hessian benchmarks."
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=[1, 2, 3],
        help="dimensions to test (default: 1 2 3)",
    )
    parser.add_argument(
        "--controls", nargs="+", default=["primal", "dual"],
        choices=["primal", "dual"],
        help="control volumes to test",
    )
    parser.add_argument(
        "--n", type=int, default=7,
        help="vertices per axis on the regular grid (default: 7)",
    )
    parser.add_argument(
        "--jitter", type=int, default=None,
        help="seed for vertex jittering (default: symmetric)",
    )
    parser.add_argument(
        "--n-gauss", type=int, default=10,
        help="Gauss quadrature order for analytical integration",
    )
    args = parser.parse_args(argv)

    print("=" * 110)
    print(f"  int_bench: integrated gradient / Hessian validation")
    print(f"  dims={args.dims}  controls={args.controls}  n={args.n}"
          f"  jitter={args.jitter}  n_gauss={args.n_gauss}")
    print("=" * 110)

    results = run_suite(
        dims=tuple(args.dims),
        controls=tuple(args.controls),
        n=args.n,
        jitter_seed=args.jitter,
        n_gauss=args.n_gauss,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
