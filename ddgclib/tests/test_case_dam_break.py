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

laneF (2026-10-05) pins: the "sliver ejection" of the July lane was two
ledger defects at one flip (method axes ``phase_ledger`` and
``face_closure``, defaults since then), and the 3D case ran on the
``batch_e_star`` fan cache whose 1 % closure defect times the absolute
pressure ejected an air cell at step 4 (``edge_area_source='p_ij_simplex'``
in the preset since then) and on a non-hydrostatic preload (criterion
labels against the vote labels).  ``TestDamBreakPins`` locks the runs
through the presets: 2D at refinement 2 / alpha_art 0.1 (fast; the
old options eject it), 2D at the shipped refinement 3 / alpha_art 0.2
over the full horizon (slow), and 3D (fast smoke, slow full horizon).
"""
import warnings

import numpy as np
import pytest

from cases_dynamic.dam_break.src._params import (
    a, L, H, W, col_w, col_h, col_d,
    rho_l, rho_g, mu_l, mu_g, gamma, K_l, K_g,
    g, gravity_axis, P_atm, cfl, alpha_art, t_end,
)
from cases_dynamic.dam_break.src._setup import (
    setup_dam_break_multiphase, cfl_timestep,
)
from ddgclib.methods import PRESETS

# The shipped runner configuration (METHODS.md): per-step full Delaunay
# kept thermodynamically neutral by the laneD conservative remap.
METHODS = PRESETS['dam_break_2D']


def _build(dim=2, alpha=alpha_art, refine=3, methods=None):
    # Default: the shipped case configuration (params alpha_art).
    return setup_dam_break_multiphase(
        dim=dim, a=a, L=L, H=H, W=W,
        col_w=col_w, col_h=col_h, col_d=col_d,
        rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
        gamma=gamma, K_l=K_l, K_g=K_g,
        g=g, gravity_axis=gravity_axis, P_atm=P_atm,
        n_refine=refine, alpha_art=alpha,
        methods=methods if methods is not None else PRESETS[f'dam_break_{dim}D'],
    )


def _build_2d(alpha=alpha_art):
    return _build(2, alpha, 3, METHODS)


def run_pinned(dim, alpha, refine, n_steps=None, methods=None,
               t_samples=(0.1, 0.2)):
    """Run a dam break through its preset (or *methods*) and return the
    pinned quantities: liquid KE peak and its time, the KE at the end,
    the front position (largest x of a liquid or interface vertex) at
    *t_samples* and at the end, the per-phase mass drift, the largest
    speed, the number of vertices outside the tank and the step count.
    KE_liq counts the bulk liquid vertices (``v.phase == 1``), as the
    runner does."""
    methods = methods if methods is not None else PRESETS[f'dam_break_{dim}D']
    HC, bV, mps, bc_set, dudt_fn, _r, params = _build(dim, alpha, refine,
                                                       methods)
    dt = cfl_timestep(HC, dim, float(np.sqrt(K_l / rho_l)), cfl=cfl)
    if n_steps is None:
        n_steps = int(t_end / dt) + 1
    M0 = np.sum([v.m_phase for v in HC.V], axis=0)
    box = [(0.0, L), (0.0, H)] + ([(0.0, W)] if dim == 3 else [])
    tol = 1e-12 * L
    sample_steps = {int(round(ts / dt)) for ts in t_samples}
    out = dict(KE_peak=-1.0, t_peak=None, u_max=0.0, n_outside=0,
               front={}, steps=0)

    def front(HC_cb):
        return max(v.x_a[0] for v in HC_cb.V
                   if v.phase == 1 or getattr(v, 'is_interface', False))

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        out['steps'] = step + 1
        ke = sum(0.5 * v.m * float(np.dot(v.u[:dim], v.u[:dim]))
                 for v in HC_cb.V if v.phase == 1)
        if ke > out['KE_peak']:
            out['KE_peak'], out['t_peak'] = ke, float(t)
        out['u_max'] = max(out['u_max'], max(
            float(np.linalg.norm(v.u[:dim])) for v in HC_cb.V))
        out['n_outside'] = max(out['n_outside'], sum(
            1 for v in HC_cb.V if any(
                v.x_a[i] < lo - tol or v.x_a[i] > hi + tol
                for i, (lo, hi) in enumerate(box))))
        if step + 1 in sample_steps:
            out['front'][step + 1] = front(HC_cb)
        if step == n_steps - 1:
            out['KE_end'] = ke
            out['front_end'] = front(HC_cb)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                          bc_set=bc_set, callback=callback, mps=mps)
    M1 = np.sum([v.m_phase for v in HC.V], axis=0)
    out['mass_drift'] = [float(c) for c in (M1 / M0 - 1.0)]
    out['dt'] = dt
    out['n_steps'] = n_steps
    out['finite'] = all(np.all(np.isfinite(v.u)) and np.isfinite(v.p)
                        for v in HC.V)
    return out


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
        HC, bV, mps, bc_set, dudt_fn, _retopo_fn, params = _build_2d(0.5)
        c_s = float(np.sqrt(K_l / rho_l))
        dt = cfl_timestep(HC, 2, c_s, cfl=cfl)
        M0 = sum(v.m for v in HC.V)
        x_front_0 = max(v.x_a[0] for v in HC.V
                        if v.phase == 1 or getattr(v, 'is_interface', False))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            METHODS.integrate(HC, bV, dudt_fn, dt=dt, n_steps=150,
                              bc_set=bc_set, mps=mps)
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


# ---------------------------------------------------------------------------
# laneF 2026-10-05 pins.  Every run is deterministic on one machine (lane
# T), so the values are pinned tightly; a run that reconnects amplifies a
# last-bit change (rule 8 of the protocol), hence rel 1e-6 on the 2D
# reconnecting runs and 1e-9 on the fixed-connectivity 3D run.
# ---------------------------------------------------------------------------
PIN_2D_R2_A01 = dict(          # refinement 2, alpha_art 0.1, 793 steps, 12 s
    KE_peak=0.0012835121048480583, t_peak=0.15876550547536425,
    KE_end=0.0012792116948716176, front_0p1=0.06111878830584278,
    front_end=0.05)
PIN_2D_R3_A02 = dict(          # refinement 3, alpha_art 0.2, 1585 steps, 90 s
    KE_peak=0.030123940492238016, t_peak=0.18451126312002433,
    KE_end=0.002392063056936934, front_0p1=0.06269550806263897,
    front_end=0.06543644759233601)
PIN_3D_SMOKE = dict(           # refinement 2, 50 steps, 6 s
    KE_end=4.071767240848256e-06, front_end=0.05031922795171791)
PIN_3D_FULL = dict(            # refinement 2, 793 steps, 100 s
    KE_peak=4.079177649585958e-06, t_peak=0.014134925765692273,
    KE_end=3.769706715085531e-06, front_0p1=0.05252173565415043,
    front_end=0.05503666879631933)


def _check(out, pin, rel):
    assert out['finite']
    assert out['n_outside'] == 0
    assert out['steps'] == out['n_steps']
    for k, want in pin.items():
        key = {'front_0p1': None, 'front_end': 'front_end'}.get(k, k)
        if k == 'front_0p1':
            got = out['front'][int(round(0.1 / out['dt']))]
        else:
            got = out[key]
        assert got == pytest.approx(want, rel=rel), f"{k}: {got} != {want}"
    assert max(abs(c) for c in out['mass_drift']) < 1e-12


class TestDamBreakPins:
    """The dam break runs its horizon without an ejection through the
    presets (laneF 2026-10-05) and reproduces the pinned numbers."""

    def test_2d_refinement2_alpha01_fast(self):
        # Five reconnections, a toe event at step 778 where a one-cell
        # liquid tongue loses its last bulk neighbour: the volume ledger
        # releases it (the front measure returns to the dam face, 0.05)
        # and the renormalised faces keep the cells closed.  The old
        # options survive this coarse run too (the toe stays stranded as
        # inertia, |u|max 0.39 against 1.29, digest f309a551f5823e64
        # against 9b5c0fbfcf25b88c): it locks the ledger's behaviour at
        # a presence change, the refinement 3 slow pin locks the fix.
        out = run_pinned(2, 0.1, 2)
        _check(out, PIN_2D_R2_A01, rel=1e-6)

    @pytest.mark.slow
    def test_2d_shipped_refinement_alpha02_full_horizon(self):
        # The run lanes F and L lost at step 1427 (the first vertex
        # outside the tank at 1267 before laneW's digests).
        out = run_pinned(2, 0.2, 3)
        _check(out, PIN_2D_R3_A02, rel=1e-6)

    def test_3d_smoke(self):
        out = run_pinned(3, alpha_art, 2, n_steps=50, t_samples=())
        _check(out, PIN_3D_SMOKE, rel=1e-9)

    @pytest.mark.slow
    def test_3d_full_horizon(self):
        out = run_pinned(3, alpha_art, 2)
        _check(out, PIN_3D_FULL, rel=1e-9)
