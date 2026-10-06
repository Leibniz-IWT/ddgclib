"""Static capillary rise on the library integrators (laneI, 2026-10-06).

Both runners go through ``PRESETS['capillary_rise_static_2D' / '_3D']``
(``cases_dynamic/capillary_rise/src/_static.py``): no time loop in the
case, gravity as ``body_force``, the EOS as ``pressure_model``, surface
tension and contact angle through ``contact_line='energy_gradient'``.
The pinned numbers are end-of-run values (protocol rule 8, lane T).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.methods import PRESETS, effective_methods  # noqa: E402


def _run(name: str, n_refine: int, n_tac: float, ic: str = 'young_laplace',
         alpha_art: float = 0.1, methods=None):
    from cases_dynamic.capillary_rise.src._static import (
        CASES, build_static_column, run_static, static_errors)
    defaults = CASES[name]
    col = build_static_column(defaults['dim'], n_refine, r=defaults['r'],
                              n_cells=defaults['n_cells'], ic=ic)
    err0 = static_errors(col)
    res = run_static(col, methods or PRESETS[name], n_tac=n_tac,
                     alpha_art=alpha_art)
    return col, res, err0, static_errors(col)


class TestSetup:
    def test_2d_column_is_the_documented_one(self):
        from cases_dynamic.capillary_rise.src._static import build_static_column
        col = build_static_column(2, 2)
        p = col.params
        assert p['n_vertices'] == 113 and p['n_frozen'] == 27
        assert p['n_free'] == 5 and p['n_contact'] == 2
        assert col.HC._simplices is not None
        # the band holds the column: reservoir level at y = 0
        assert min(v.x_a[1] for v in col.HC.V) == pytest.approx(-p['D'])
        assert p['D'] == pytest.approx(3 * p['w'] - p['h_jurin'])
        # the reference is the compressible Young-Laplace meniscus
        assert p['h_ref'] > p['h_jurin']
        assert p['h_ref'] / p['h_jurin'] == pytest.approx(
            1.0 + p['rho'] * p['g'] * p['h_jurin'] / (2 * p['K']), abs=1e-4)
        # no contact vertex is frozen, every wall vertex is
        assert not any(v in col.walls for v in col.contact)
        assert all(col.is_wall(v) for v in col.bV)

    def test_young_laplace_start_is_within_the_discretisation_error(self):
        from cases_dynamic.capillary_rise.src._static import (
            build_static_column, static_errors)
        for n, tol in ((2, 4e-2), (3, 1.2e-2)):
            err = static_errors(build_static_column(2, n, ic='young_laplace'))
            assert abs(err['h_error_rel']) < tol     # 3.4e-2, 1.06e-2
            assert err['h_contact'] == pytest.approx(err['h_contact_ref'],
                                                     rel=1e-9)
        err = static_errors(build_static_column(2, 2, ic='flat'))
        assert err['h_mean'] == pytest.approx(err['h_jurin'], rel=1e-12)
        assert err['shape_rms'] > 3e-4

    def test_3d_tube_is_the_inscribed_polygon(self):
        from cases_dynamic.capillary_rise.src._static import build_static_column
        col = build_static_column(3, 1)
        p = col.params
        assert p['n_vertices'] == 87 and p['n_contact'] == 8 and p['n_free'] == 9
        # octagon inscribed in the circle: perimeter / area = 8 sin(pi/8) / (sqrt(2) r)
        r = p['r']
        assert p['perimeter'] / p['area'] == pytest.approx(
            8 * 2 * r * np.sin(np.pi / 8) / (2 * np.sqrt(2) * r**2), rel=1e-9)
        assert p['h_jurin_poly'] / p['h_jurin'] == pytest.approx(
            (p['perimeter'] / p['area']) * r / 2, rel=1e-12)


class TestStatic2D:
    def test_short_run_pins(self):
        """Refinement 2 (113 vertices), 10 acoustic times from the
        Young-Laplace start, alpha_art 0.1."""
        col, res, err0, err = _run('capillary_rise_static_2D', 2, 10.0)
        eff = effective_methods(col.HC, 2, PRESETS['capillary_rise_static_2D'])
        assert eff['dual_volume'] == 'simplex_exact'
        assert len(col.bV) == 27
        assert all(v.x_a[0] in (0.0, col.params['w']) for v in col.contact)
        assert res['dt'] == res['dt_acoustic'] < res['dt_capillary']
        assert err['h_mean'] == pytest.approx(PIN_2D_H_10, rel=1e-9)
        assert float(res['umax'][-1]) == pytest.approx(PIN_2D_UMAX_10, rel=1e-9)
        assert float(res['umax'].max()) == pytest.approx(PIN_2D_UMAX_PEAK,
                                                         rel=1e-9)

    @pytest.mark.slow
    def test_settles_on_the_young_laplace_meniscus(self):
        """Refinement 2, 100 acoustic times, alpha_art 0.05."""
        col, res, err0, err = _run('capillary_rise_static_2D', 2, 100.0,
                                   alpha_art=0.05)
        assert abs(err['h_error_rel']) < 1e-2          # measured +6.1e-03
        assert err['shape_rms'] < 1e-4                  # measured 6.0e-05
        assert float(res['umax'][-1]) < 1e-3            # measured 5.0e-04
        assert abs(err['mass_drift']) < 2e-2           # measured -8.3e-03
        assert err['h_mean'] == pytest.approx(PIN_2D_H_100, rel=1e-9)
        assert float(res['umax'][-1]) == pytest.approx(PIN_2D_UMAX_100, rel=1e-9)


class TestStatic3D:
    def test_short_run_pins(self):
        """Refinement 1 (87 vertices, octagonal tube), 4 acoustic times."""
        col, res, err0, err = _run('capillary_rise_static_3D', 1, 4.0)
        eff = effective_methods(col.HC, 3, PRESETS['capillary_rise_static_3D'])
        assert eff['edge_area_source'] == 'p_ij_simplex'
        assert eff['boundary_dual_vol'] == 'half_cell'
        assert all(v.dual_vol > 0.0 for v in col.bV)
        assert err['h_mean'] == pytest.approx(PIN_3D_H_4, rel=1e-9)
        assert float(res['umax'][-1]) == pytest.approx(PIN_3D_UMAX_4, rel=1e-9)

    @pytest.mark.slow
    def test_settles_on_the_polygon_jurin_height(self):
        """Refinement 1, 40 acoustic times, alpha_art 0.05."""
        col, res, err0, err = _run('capillary_rise_static_3D', 1, 40.0,
                                   alpha_art=0.05)
        assert abs(err['h_error_poly_rel']) < 3e-2      # measured -1.43e-02
        assert float(res['umax'][-1]) < 2e-3            # measured 5.5e-04
        assert err['h_mean'] == pytest.approx(PIN_3D_H_40, rel=1e-9)
        assert float(res['umax'][-1]) == pytest.approx(PIN_3D_UMAX_40, rel=1e-9)


# ---------------------------------------------------------------------------
# pins (laneI, 2026-10-06; end-of-run values, one process suffices since
# lane T; the methods block of each run is in the lane log)
# ---------------------------------------------------------------------------
PIN_2D_H_10 = 0.00380100337536971
PIN_2D_UMAX_10 = 0.0015325773786546008
PIN_2D_UMAX_PEAK = 0.01683708352699222
PIN_2D_H_100 = 0.0037063081015925948
PIN_2D_UMAX_100 = 0.0004958823503840141
PIN_3D_H_4 = 0.00781529116541829
PIN_3D_UMAX_4 = 0.001228862275931968
PIN_3D_H_40 = 0.00786039962372492
PIN_3D_UMAX_40 = 0.0005474210261802552
