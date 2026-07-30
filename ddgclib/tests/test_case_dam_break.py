"""Regression tests for the 2D dam-break case (laneF, 2026-07-30).

Locks the two laneF case fixes that make the dam actually collapse:

1. Geometry: the water column must NOT fill the tank height
   (``col_h < H`` — a full-height slab pinned between the frozen floor
   and lid rows cannot collapse; the module docstring diagram shows the
   column top at ``y = a`` with air above).
2. Hydrostatic IC: under ``redistribute_mass=True`` the per-vertex
   pressure STRUCTURE is preserved across every step (only a uniform
   per-phase offset can evolve, laneD §1.1), so the collapse-driving
   hydrostatic head must be in the INITIAL state.  With the old
   flat-at-P_atm IC the liquid column measured a uniform +0.121 Pa
   gauge after 0.098 s under gravity instead of the ~490 Pa head, the
   dam face saw ~0 horizontal force, and the case stalled at
   |u| ~ 0.02 m/s forever.

Plus a short collapse smoke on the shipped configuration (per-step
Delaunay + ``retopo_remap='conservative'``): the laneF A/B measured
plain per-step Delaunay blowing up at its FIRST reconnection event
(KE x28 in one step) while the remap absorbs reconnection and survives
the full case horizon.
"""
import warnings
from functools import partial

import numpy as np
import pytest

from cases_dynamic.dam_break.src._params import (
    a, L, H, W, col_w, col_h, col_d,
    rho_l, rho_g, mu_l, mu_g, gamma, K_l, K_g,
    g, gravity_axis, P_atm, cfl, alpha_art,
)
from cases_dynamic.dam_break.src._setup import (
    setup_dam_break_multiphase, cfl_timestep,
)
from ddgclib.dynamic_integrators import symplectic_euler


def _build_2d(alpha=alpha_art):
    # Default: the shipped case configuration (params alpha_art).
    return setup_dam_break_multiphase(
        dim=2, a=a, L=L, H=H, W=W,
        col_w=col_w, col_h=col_h, col_d=col_d,
        rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
        gamma=gamma, K_l=K_l, K_g=K_g,
        g=g, gravity_axis=gravity_axis, P_atm=P_atm,
        n_refine=3, alpha_art=alpha,
    )


class TestDamBreakHydrostaticIC:
    """The dam must be loaded (geometry + hydrostatic head) at t=0."""

    def test_column_has_headspace(self):
        # laneF geometry fix: col_h == H (full-height slab) cannot
        # collapse — the top liquid row is frozen to the lid.
        assert col_h < H - 1e-12

    def test_hydrostatic_pressure_structure(self):
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = _build_2d(0.5)
        # Bulk liquid vertices carry the hydrostatic head (structure is
        # what redistribution preserves — it must be present at t=0).
        checked = 0
        for v in HC.V:
            if v.phase != 1 or getattr(v, 'is_interface', False):
                continue
            y = float(v.x_a[gravity_axis])
            p_target = (P_atm + rho_g * g * (H - col_h)
                        + rho_l * g * (col_h - y))
            assert v.p == pytest.approx(p_target, abs=1e-6), \
                f"liquid vertex at y={y}: p={v.p} != {p_target}"
            checked += 1
        assert checked >= 5

    def test_dam_face_is_released(self):
        # The mid-height dam-face interface vertex must feel an O(10)
        # horizontal stress acceleration at t=0 (measured +39 m/s^2)
        # and near-hydrostatic vertical support.  The stalled flat-IC
        # configuration measured |a_x| <= 0.7 m/s^2 at all times.
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = _build_2d(0.5)
        face = [v for v in HC.V
                if getattr(v, 'is_interface', False)
                and abs(v.x_a[0] - col_w) < 1e-9
                and 0.3 * col_h < v.x_a[1] < 0.7 * col_h]
        assert face, "no mid-height dam-face interface vertices found"
        for v in face:
            acc = dudt_fn(v)
            assert acc[0] > 10.0, \
                f"dam face not released: a_x={acc[0]} at {v.x_a}"
            assert abs(acc[1]) < 5.0, \
                f"vertical hydrostatic balance broken: a_y={acc[1]}"


class TestDamBreakCollapseSmoke:
    """Short smoke of the shipped config: collapse starts, stays clean."""

    def test_collapse_starts_and_conserves_mass(self):
        HC, bV, mps, bc_set, dudt_fn, retopo_fn, params = _build_2d(0.5)
        # Shipped runner configuration: per-step full Delaunay kept
        # thermodynamically neutral by the laneD conservative remap.
        retopo_fn = partial(retopo_fn, retopo_remap='conservative')
        c_s = float(np.sqrt(K_l / rho_l))
        dt = cfl_timestep(HC, 2, c_s, cfl=cfl)
        M0 = sum(v.m for v in HC.V)
        x_front_0 = max(v.x_a[0] for v in HC.V
                        if v.phase == 1 or getattr(v, 'is_interface', False))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            symplectic_euler(HC, bV, dudt_fn, dt=dt, n_steps=150, dim=2,
                             bc_set=bc_set, retopologize_fn=retopo_fn)
        # No NaN anywhere
        for v in HC.V:
            assert np.all(np.isfinite(v.u)), f"NaN velocity at {v.x_a}"
            assert np.isfinite(v.p), f"NaN pressure at {v.x_a}"
        # Mass conservation at machine precision
        M1 = sum(v.m for v in HC.V)
        assert abs(M1 / M0 - 1.0) < 1e-12
        # The collapse is under way: liquid KE built up and the front
        # moved right (measured at 150 steps / alpha_art=0.5:
        # KE ~ 5e-4 J, front +~1.4 mm; stalled flat-IC config: KE
        # plateau 1e-4 with front frozen at +0.1 mm over 2242 steps).
        ke = sum(0.5 * v.m * float(np.dot(v.u[:2], v.u[:2]))
                 for v in HC.V
                 if v.phase == 1 or getattr(v, 'is_interface', False))
        assert ke > 2e-4, f"collapse did not start: KE_liq={ke}"
        x_front = max(v.x_a[0] for v in HC.V
                      if v.phase == 1 or getattr(v, 'is_interface', False))
        assert x_front - x_front_0 > 5e-4, \
            f"dam face did not advance: {x_front_0} -> {x_front}"
