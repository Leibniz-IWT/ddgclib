"""Pinned smoke tests and physical checks of the electrolysis bubble case
(laneG, 2026-10-06).

The case runs through ``PRESETS['electrolysis_bubble_2D']`` / ``['_3D']``
on library paths; the gas injection of the callback is
``ddgclib.operators.mass_source.add_phase_mass`` (the case's
``inject_gas_mass`` wraps it).

Physical checks (integrated, ``ddgclib.analytical``): a STATIC bubble
(``g = 0``, no injection) whose preloaded state is the analytical
solution holds its Laplace pressure ``gamma (dim - 1) / R0`` to the
discretisation error of the polyhedral bubble over the window, with the
per-phase masses conserved to round-off; under the case's injection the
gas mass follows ``M0 + dm_dt t`` to round-off and the liquid mass is
conserved.  Pins: per-phase mass drift, bubble volume, interface vertex
count, final-state digest (2D and 3D smoke), the 3D jump at refinement
2/2 (slow).
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.analytical import integrated_phase_pressure_jump  # noqa: E402
from ddgclib.methods import PRESETS  # noqa: E402

from cases_dynamic.electrolysis_bubble.src import _params as P  # noqa: E402
from cases_dynamic.electrolysis_bubble.src._analytical import (  # noqa: E402
    young_laplace_jump,
)
from cases_dynamic.electrolysis_bubble.src._reaction import (  # noqa: E402
    inject_gas_mass,
)
from cases_dynamic.electrolysis_bubble.src._setup import (  # noqa: E402
    setup_electrolysis_bubble,
)
from cases_dynamic.electrolysis_bubble.diagnose_static_bubble import (  # noqa: E402
    time_step,
)


def _build(dim, ro, rd, g=0.0, methods=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return setup_electrolysis_bubble(
            dim=dim, R0=P.R0, L_domain=P.L_domain,
            nucleation_frac=P.nucleation_frac, rho_liq=P.rho_liq,
            rho_gas=P.rho_gas, mu_liq=P.mu_liq, mu_gas=P.mu_gas,
            gamma=P.gamma, K_liq=P.K_liq, K_gas=P.K_gas, g=g, P0=P.P0,
            refinement_outer=ro, refinement_droplet=rd,
            methods=methods or PRESETS[f'electrolysis_bubble_{dim}D'])


def _digest(HC, dim):
    state = sorted((tuple(v.x_a[:dim]), tuple(v.u[:dim]), float(v.m),
                    tuple(float(p) for p in v.p_phase)) for v in HC.V)
    return hashlib.sha256(repr(state).encode()).hexdigest()[:16]


def _masses(HC):
    return (sum(float(v.m_phase[0]) for v in HC.V),
            sum(float(v.m_phase[1]) for v in HC.V))


def _gas_volume(HC):
    return sum(float(v.dual_vol_phase[1]) for v in HC.V)


def _n_interface(HC):
    return sum(1 for v in HC.V if getattr(v, 'is_interface', False))


def _run(dim, ro, rd, n_steps, g=0.0, inject=False, methods=None):
    HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = _build(
        dim, ro, rd, g=g, methods=methods)
    methods = methods or PRESETS[f'electrolysis_bubble_{dim}D']
    dt = time_step(HC, dim)
    dm_dt = P.dm_dt_3d if dim == 3 else P.dm_dt_2d
    M0 = _masses(HC)
    V0 = _gas_volume(HC)
    dp0 = integrated_phase_pressure_jump(HC, 1, 0)

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        if inject:
            inject_gas_mass(HC_cb, mps, dm_dt=dm_dt, dt=dt, gas_phase=1)

    methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                      bc_set=bc_set, callback=cb, mps=mps)
    M = _masses(HC)
    return dict(
        HC=HC, dt=dt, M0=M0, M=M, V0=V0, V=_gas_volume(HC), dp0=dp0,
        dp=integrated_phase_pressure_jump(HC, 1, 0),
        n_interface=_n_interface(HC), digest=_digest(HC, dim),
        M_gas_expected=M0[1] + (dm_dt * dt * n_steps if inject else 0.0),
        KE=sum(0.5 * float(v.m) * float(v.u[:dim] @ v.u[:dim]) for v in HC.V),
    )


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

PIN_3D_STATIC = {       # laneG 2026-10-06; refinement 1/1, g = 0, 200 steps
    'n_vertices': 95, 'n_interface': 26, 'n_steps': 200,
    'V_ratio': 0.99998526773405, 'dp': 145.50716769098642,
    'digest': 'ed70a7c01e8a2f0a',
}

PIN_3D_INJECT = {       # laneG; refinement 1/1, g = 9.81, injection, 100 steps
    'n_steps': 100, 'V_ratio': 1.0000026474036243, 'dp': 410.47151474489704,
    'digest': 'eac3602aa359adb0',
}


class TestStaticBubble3D:
    def test_preload_is_the_analytical_state(self):
        HC, bV, mps, *_ = _build(3, 1, 1)
        assert len(list(HC.V)) == PIN_3D_STATIC['n_vertices']
        assert _n_interface(HC) == PIN_3D_STATIC['n_interface']
        assert integrated_phase_pressure_jump(HC, 1, 0) == pytest.approx(
            young_laplace_jump(P.gamma, P.R0, dim=3), abs=1e-9)
        assert young_laplace_jump(P.gamma, P.R0, dim=3) == 144.0

    def test_pinned_static_window(self):
        """200 steps (1.3e-5 s) of the static bubble through the preset:
        masses conserved to round-off, the jump within 10 % of 144 Pa,
        the 35 gas cells and 26 interface vertices kept."""
        r = _run(3, 1, 1, PIN_3D_STATIC['n_steps'])
        assert abs(r['M'][0] / r['M0'][0] - 1.0) < 1e-13
        assert abs(r['M'][1] / r['M0'][1] - 1.0) < 1e-13
        assert r['n_interface'] == PIN_3D_STATIC['n_interface']
        assert abs(r['dp'] / 144.0 - 1.0) < 0.10
        assert r['V'] / r['V0'] == pytest.approx(PIN_3D_STATIC['V_ratio'],
                                                 rel=1e-9)
        assert r['dp'] == pytest.approx(PIN_3D_STATIC['dp'], rel=1e-9)
        assert r['digest'] == PIN_3D_STATIC['digest']

    def test_pinned_injection_smoke(self):
        """The runner's loop (gravity, wall clamps, gas injection in the
        callback) for 100 steps: the gas mass follows M0 + dm_dt t and
        the liquid mass is conserved, both to round-off; the gas phase
        is kept."""
        r = _run(3, 1, 1, PIN_3D_INJECT['n_steps'], g=P.g, inject=True)
        assert abs(r['M'][1] / r['M_gas_expected'] - 1.0) < 1e-13
        assert abs(r['M'][0] / r['M0'][0] - 1.0) < 1e-13
        assert r['n_interface'] == PIN_3D_STATIC['n_interface']
        assert sum(1 for v in r['HC'].V if v.dual_vol_phase[1] > 1e-30) == 35
        assert r['V'] / r['V0'] == pytest.approx(PIN_3D_INJECT['V_ratio'],
                                                 rel=1e-9)
        assert r['dp'] == pytest.approx(PIN_3D_INJECT['dp'], rel=1e-9)
        assert r['digest'] == PIN_3D_INJECT['digest']


PIN_3D_STATIC_SLOW = {  # laneG 2026-10-06; refinement 2/2, g = 0, 2000 steps
    'n_vertices': 475, 'n_interface': 98, 'n_steps': 2000,
    'dp_tol': 0.20, 'dp': 165.4055950625294,
}


@pytest.mark.slow
class TestStaticBubble3DRefined:
    def test_pinned_static_window_refinement_2(self):
        r = _run(3, 2, 2, PIN_3D_STATIC_SLOW['n_steps'])
        assert len(list(r['HC'].V)) == PIN_3D_STATIC_SLOW['n_vertices']
        assert r['n_interface'] == PIN_3D_STATIC_SLOW['n_interface']
        assert abs(r['M'][1] / r['M0'][1] - 1.0) < 1e-13
        assert abs(r['dp'] / 144.0 - 1.0) < PIN_3D_STATIC_SLOW['dp_tol']
        assert r['dp'] == pytest.approx(PIN_3D_STATIC_SLOW['dp'], rel=1e-9)


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

PIN_2D_STATIC = {       # laneG 2026-10-06; refinement 1/2, g = 0, 200 steps
    'n_vertices': 69, 'n_interface': 16, 'n_steps': 200,
    'V_ratio': 0.9999987994161297, 'dp': 72.1266009076101,
    'digest': '22a178b7ef93e21f',
}

PIN_2D_INJECT = {       # laneG; refinement 1/2, g = 9.81, injection, 100 steps
    'n_steps': 100, 'V_ratio': 1.0000304343618025, 'dp': 585.9510701158798,
    'digest': '30d427443eae547e',
}


class TestStaticBubble2D:
    def test_pinned_static_window(self):
        r = _run(2, 1, 2, PIN_2D_STATIC['n_steps'])
        assert len(list(r['HC'].V)) == PIN_2D_STATIC['n_vertices']
        assert r['n_interface'] == PIN_2D_STATIC['n_interface']
        assert r['dp0'] == pytest.approx(young_laplace_jump(P.gamma, P.R0, 2),
                                         abs=1e-9)
        assert abs(r['M'][0] / r['M0'][0] - 1.0) < 1e-13
        assert abs(r['M'][1] / r['M0'][1] - 1.0) < 1e-13
        assert abs(r['dp'] / (P.gamma / P.R0) - 1.0) < 0.10
        assert r['V'] / r['V0'] == pytest.approx(PIN_2D_STATIC['V_ratio'],
                                                 rel=1e-9)
        assert r['dp'] == pytest.approx(PIN_2D_STATIC['dp'], rel=1e-9)
        assert r['digest'] == PIN_2D_STATIC['digest']

    def test_pinned_injection_smoke(self):
        r = _run(2, 1, 2, PIN_2D_INJECT['n_steps'], g=P.g, inject=True)
        assert abs(r['M'][1] / r['M_gas_expected'] - 1.0) < 1e-13
        assert abs(r['M'][0] / r['M0'][0] - 1.0) < 1e-13
        assert r['n_interface'] == PIN_2D_STATIC['n_interface']
        assert r['V'] / r['V0'] == pytest.approx(PIN_2D_INJECT['V_ratio'],
                                                 rel=1e-9)
        assert r['dp'] == pytest.approx(PIN_2D_INJECT['dp'], rel=1e-9)
        assert r['digest'] == PIN_2D_INJECT['digest']
