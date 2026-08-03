"""A.5.b long-run regression — Probe 6 (Phase A, stabilisation roadmap).

Pins the steady-state max|F| from the A.5.b harness (retopology ON,
``v.u`` forced to 0 every step, ``redistribute_mass=True``) on the
oscillating-droplet static fixture.  The harness is described in
``cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py``.

Why this regression exists
--------------------------
After the 2026-04-29 ``redistribute_mass_multiphase`` guard fix (Phase 2c
of the 3D retopology blow-up isolation), the 3D static-droplet A.5.b
max|F| collapsed from 1.44e-3 to **7.3768e-05** — a ×195 improvement
that brought 3D within ×1.23 of the A.5.a curvature-stencil floor
(6.02e-05).  Probe 6 (2026-05-27) confirmed the floor is flat over
n_steps=2000 with no slow drift, so it can safely be encoded as a
regression guard.  Any future change that touches the
mass-redistribution / per-phase split / retopology / curvature-stencil
stack and shifts this value by more than 1% should trip this test.

The 2D variant pins the analogous 2.3749e-3 floor (curvature-stencil
bounded, retopology benign because Delaunay-of-static-cloud is
idempotent in 2D — see plan A.5.2D verdict).
"""
import os
import sys
import unittest

import pytest

# cases_dynamic.oscillating_droplet.diagnose_a5_bisection lives in the
# repo root's cases_dynamic/ tree.  The repo root is the parent of
# ddgclib/, which is the parent of ddgclib/tests/.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cases_dynamic.oscillating_droplet.diagnose_a5_bisection import (  # noqa: E402
    run_a5b,
)


# Measured A.5.b max|F| floors.  The harness records two distinct
# steady-state values per run:
#
# - ``max_abs_F_peak`` — the peak of the recorded max|F| over the run.
#   In 2D the peak equals the step-0 (frozen-mesh, pre-retopo) value
#   because Delaunay-of-static-cloud is idempotent in 2D and retopology
#   actually decreases |F| on the first step.  In 3D the peak equals
#   the step-1+ post-retopo value because 3D Delaunay churns ~48
#   cross-phase edges per static-cloud retopo step, pushing |F| UP
#   relative to the frozen mesh — even after the Phase-2c
#   ``dual_vol_phase``-gated mass-redistribution fix.
#
# - ``max_abs_F_end`` — the max|F| recorded at the final step.  Flat
#   after the step-0 -> step-1 transition.
#
# 3D: post-Probe-2c (2026-04-29 redistribute_mass guard fix),
# confirmed flat at 7.3768e-05 over n_steps=2000 by Probe 6
# (2026-05-27).  Refinement 2/2, 472 vertices / 98 interface.
A5B_3D_PEAK = 7.3768e-05
A5B_3D_END = 7.3768e-05

# 2D: post-Phase-2c, curvature-stencil bound on the polygon-vs-circle
# truncation.  Refinement 3/3, 311 vertices / 32 interface.  Peak is
# the step-0 A.5.a-equivalent value; end is the slightly lower
# post-retopo steady-state.
A5B_2D_PEAK = 2.3749e-03
A5B_2D_END = 2.2717e-03

# Tolerance per Probe 6 spec: ~1% of each pinned floor.
TOLERANCE_REL = 0.01


def _assert_within_tolerance(measured, pinned, label):
    rel = abs(measured - pinned) / pinned
    assert rel < TOLERANCE_REL, (
        f"{label}: measured={measured:.4e} drifted "
        f"{rel * 100:.2f}% from pinned floor {pinned:.4e} "
        f"(tolerance {TOLERANCE_REL * 100:.1f}%).  If this is intentional, "
        f"re-measure with diagnose_a5_bisection.py and update the floor."
    )


class TestA5BLongRunRegression2D(unittest.TestCase):
    """A.5.b 2D steady-state regression (fast: ~1-2 s)."""

    def test_a5b_2d_steady_state_floor(self):
        result = run_a5b(
            dim=2,
            refinement_outer=3,
            refinement_droplet=3,
            n_steps=20,
            split_method='neighbour_count',
            redistribute_mass=True,
            curvature_path='integrated',
        )
        _assert_within_tolerance(
            result['max_abs_F_peak'], A5B_2D_PEAK,
            'A.5.b 2D peak max|F|',
        )
        _assert_within_tolerance(
            result['max_abs_F_end'], A5B_2D_END,
            'A.5.b 2D final max|F|',
        )


@pytest.mark.slow
class TestA5BLongRunRegression3D(unittest.TestCase):
    """A.5.b 3D steady-state regression (slow: refinement 2/2 = 472 vertices).

    Marked slow because retopology + per-phase refresh on every step
    over a 472-vertex / 98-interface 3D mesh is heavier than the 2D
    fast-suite budget.  Use ``pytest -m "not slow"`` to skip in routine
    runs; CI should include slow tests on the main-branch sweep.
    """

    def test_a5b_3d_steady_state_floor(self):
        result = run_a5b(
            dim=3,
            refinement_outer=2,
            refinement_droplet=2,
            n_steps=20,
            split_method='neighbour_count',
            redistribute_mass=True,
            curvature_path='integrated',
        )
        _assert_within_tolerance(
            result['max_abs_F_peak'], A5B_3D_PEAK,
            'A.5.b 3D peak max|F|',
        )
        _assert_within_tolerance(
            result['max_abs_F_end'], A5B_3D_END,
            'A.5.b 3D final max|F|',
        )


if __name__ == '__main__':
    unittest.main()
