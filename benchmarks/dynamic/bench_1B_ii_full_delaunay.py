"""Tier 1B.ii — retopology on, full periodic Delaunay rebuild every step.

Same decaying-shear-wave physics as 1A/1B.i, but with the production
default retopology: a global periodic Delaunay retriangulation
(``periodic_axes=[0]``) every step.  This adds connectivity churn on top
of the dual-volume refresh measured by 1B.i.

Run::

    python benchmarks/dynamic/bench_1B_ii_full_delaunay.py
"""
from __future__ import annotations

import sys

from _harness import print_summary, run_benchmark, save_result


def main() -> int:
    result = run_benchmark('full_delaunay')
    print_summary(result)
    save_result(result, 'bench_1B_ii_full_delaunay.json')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
