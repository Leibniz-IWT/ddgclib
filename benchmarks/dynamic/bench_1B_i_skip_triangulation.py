"""Tier 1B.i — retopology on, duals recomputed but connectivity fixed.

Same decaying-shear-wave physics as 1A, but the duals are rebuilt every
step on the **existing connectivity** (no Delaunay).  Compared against
1B.ii (full Delaunay), this isolates the dual-volume refresh from the
connectivity churn:

    1B.i conserves & 1B.ii doesn't  -> bug is Delaunay connectivity churn
    1B.i already fails               -> bug is the dual-volume refresh / mass

KNOWN ISSUE (demonstrated below).  The documented ``skip_triangulation=True``
integrator flag is **silently bypassed** when ``periodic_axes`` is set:
``_retopologize`` dispatches to ``retopologize_periodic`` and returns before
the flag is consulted, so the flag-based run is bit-identical to a full
Delaunay rebuild.  This script therefore measures the *genuine* skip rung
via a custom ``retopologize_fn`` (``periodic_skip_retopo``) and separately
shows the flag is inert.

Run::

    python benchmarks/dynamic/bench_1B_i_skip_triangulation.py
"""
from __future__ import annotations

import sys

from _harness import print_summary, run_benchmark, save_result


def demonstrate_flag_bypass() -> bool:
    """Confirm ``skip_triangulation=True`` is inert under ``periodic_axes``.

    Returns True if the flag-based run is bit-identical to full Delaunay
    (i.e. the flag had no effect), which is the bug we are flagging.
    """
    flag = run_benchmark('skip_flag')
    delaunay = run_benchmark('full_delaunay')
    same = (abs(flag['metrics']['max_vol_drift']
                - delaunay['metrics']['max_vol_drift']) < 1e-15
            and abs(flag['metrics']['final_ke_ratio']
                    - delaunay['metrics']['final_ke_ratio']) < 1e-15)
    print("\n--- skip_triangulation flag bypass check ---")
    print(f"  skip_flag    max|dV/V0|={flag['metrics']['max_vol_drift']:.6e}")
    print(f"  full_delaunay max|dV/V0|="
          f"{delaunay['metrics']['max_vol_drift']:.6e}")
    print(f"  flag is inert (bit-identical to full Delaunay): {same}")
    return same


def main() -> int:
    result = run_benchmark('skip_triangulation')
    print_summary(result)
    save_result(result, 'bench_1B_i_skip_triangulation.json')
    demonstrate_flag_bypass()
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
