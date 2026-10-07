"""Regression for the A.3 single-phase retopology bisection (1A / 1B.i / 1B.ii).

Pins the bisection ordering on the periodic decaying-sinusoid setup:

    frozen-mesh   conserves volume to machine precision (PASS)
    1B.i (skip)   first to break it via the dual-volume refresh   (FAIL)
    1B.ii (full)  worse still via Delaunay connectivity churn      (FAIL)

Run::

    pytest benchmarks/dynamic/test_a3_bisection.py -v
"""
from __future__ import annotations

import pytest

from _harness import MASS_TOL, VOL_TOL, run_benchmark

# Reduced horizon for a fast regression (full benchmark uses n_steps=400).
_PARAMS = {'n_steps': 120, 'record_every': 10}


@pytest.fixture(scope='module')
def results() -> dict:
    return {mode: run_benchmark(mode, params=_PARAMS)
            for mode in ('frozen', 'skip_triangulation', 'full_delaunay')}


def test_mass_conserved_all_modes(results):
    for mode, r in results.items():
        assert r['metrics']['max_mass_drift'] < MASS_TOL, mode


def test_frozen_conserves_volume(results):
    assert results['frozen']['passed']
    assert results['frozen']['metrics']['max_vol_drift'] < VOL_TOL


def test_skip_triangulation_first_to_fail_volume(results):
    # The dual-volume refresh alone breaks machine-precision conservation.
    assert not results['skip_triangulation']['passed']
    assert results['skip_triangulation']['metrics']['max_vol_drift'] > VOL_TOL


def test_full_delaunay_worse_than_skip(results):
    # Connectivity churn amplifies the dual-refresh drift.
    skip = results['skip_triangulation']['metrics']['max_vol_drift']
    full = results['full_delaunay']['metrics']['max_vol_drift']
    assert full > skip


def test_ke_decays_monotonically(results):
    for mode, r in results.items():
        assert r['metrics']['ke_monotonic'], mode


def test_skip_flag_is_bypassed_under_periodic():
    # Documented skip_triangulation=True flag is inert when periodic_axes
    # is set: bit-identical to a full Delaunay rebuild.
    flag = run_benchmark('skip_flag', params=_PARAMS)
    full = run_benchmark('full_delaunay', params=_PARAMS)
    assert abs(flag['metrics']['max_vol_drift']
               - full['metrics']['max_vol_drift']) < 1e-15
    assert abs(flag['metrics']['final_ke_ratio']
               - full['metrics']['final_ke_ratio']) < 1e-15
