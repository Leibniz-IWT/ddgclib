"""Tier 1A — frozen-mesh transient decay (single-phase).

Decaying shear wave in an x-periodic channel with retopology disabled
(``retopologize_fn=False``).  Isolates the time integrator + stress
operator dynamics from all retopology effects.  Because the duals are
never redrawn, total dual volume is conserved to machine precision by
construction; this rung is the conservation baseline the 1B rungs are
measured against.

Run::

    python benchmarks/dynamic/bench_1A_frozen_mesh.py

Exits non-zero if the rung fails its conservation gate.
"""
from __future__ import annotations

import sys

from _harness import print_summary, run_benchmark, save_result


def main() -> int:
    result = run_benchmark('frozen')
    print_summary(result)
    save_result(result, 'bench_1A_frozen_mesh.json')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
