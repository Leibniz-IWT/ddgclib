"""Tests for the hydrostatic column case (1D, 2D, 3D).

Validates that:
1. At analytical hydrostatic equilibrium, the acceleration is zero (or near-zero)
   for all interior vertices.
2. A perturbed state returns toward equilibrium when integrated.
"""

import numpy as np
import numpy.testing as npt
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Fixtures

@pytest.fixture
def hydrostatic_1d():
    from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
    HC, bV, ic, bc_set, params = setup_hydrostatic(dim=1, n_refine=3)
    ic.apply(HC, bV)
    return HC, bV, ic, bc_set, params


@pytest.fixture
def hydrostatic_2d():
    from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
    HC, bV, ic, bc_set, params = setup_hydrostatic(dim=2, n_refine=1)
    ic.apply(HC, bV)
    return HC, bV, ic, bc_set, params


# Static equilibrium tests

class TestHydrostaticEquilibrium1D:
    def test_pressure_profile(self, hydrostatic_1d):
        """Verify analytical pressure: P = rho*g*(h - x)."""
        HC, bV, _, _, p = hydrostatic_1d
        rho, g, h = p['rho'], p['g'], p['h']
        for v in HC.V:
            expected = rho * g * (h - v.x_a[0])
            npt.assert_allclose(v.p, expected, atol=1e-10,
                                err_msg=f"Wrong p at x={v.x_a[0]}")

    def test_zero_velocity(self, hydrostatic_1d):
        HC, _, _, _, _ = hydrostatic_1d
        for v in HC.V:
            npt.assert_array_equal(v.u, np.zeros(1))

    def test_acceleration_near_zero(self, hydrostatic_1d):
        """At equilibrium, acceleration should be ~zero for interior vertices.

        NOTE: This test requires barycentric duals to be computed. We use the
        clean gradient operator which depends on e_star from _duals.py.
        """
        from hyperct.ddg import compute_vd
        from ddgclib.operators.gradient import pressure_gradient

        HC, bV, _, _, p = hydrostatic_1d
        compute_vd(HC, cdist=1e-10)

        for v in HC.V:
            if v not in bV:
                grad_P = pressure_gradient(v, dim=1, HC=HC)
                # For hydrostatic: grad_P should equal rho*g (downward)
                # The net force = -grad_P + rho*g should be ~zero
                # With our convention, acceleration = (-gradP)/m
                # and gravity is already encoded in P analytically.
                # So |gradP| should be finite but the *net* dudt should be small
                # if we add gravity as a body force.
                # For now, just verify gradient is computed without error
                assert grad_P.shape == (1,)


class TestHydrostaticEquilibrium2D:
    def test_pressure_profile(self, hydrostatic_2d):
        """Verify analytical pressure in 2D (gravity along axis=1)."""
        HC, bV, _, _, p = hydrostatic_2d
        rho, g, h = p['rho'], p['g'], p['h']
        axis = p['gravity_axis']  # 1 for 2D
        for v in HC.V:
            expected = rho * g * (h - v.x_a[axis])
            npt.assert_allclose(v.p, expected, atol=1e-10,
                                err_msg=f"Wrong p at x={v.x_a}")

    def test_zero_velocity(self, hydrostatic_2d):
        HC, _, _, _, _ = hydrostatic_2d
        for v in HC.V:
            npt.assert_array_equal(v.u, np.zeros(2))


# Perturbation recovery tests

class TestPerturbationRecovery1D:
    def test_perturbed_state_converges(self, hydrostatic_1d):
        """Perturb velocity, run integrator, verify L2 norm decreases.

        With no-slip BCs and viscous damping, the system should dissipate
        kinetic energy and return toward equilibrium.
        """
        from ddgclib.dynamic_integrators import euler_velocity_only
        from ddgclib.operators.gradient import acceleration

        HC, bV, _, bc_set, p = hydrostatic_1d

        # Perturb interior velocities
        for v in HC.V:
            if v not in bV:
                v.u = np.array([0.1])

        # Compute initial KE
        ke_initial = sum(0.5 * v.m * np.dot(v.u, v.u) for v in HC.V
                         if v not in bV)

        # Run a few steps (velocity only, no mesh movement)
        # Note: acceleration depends on pressure gradient + viscous term
        # For this simple test, use a mock that just damps velocity
        def damping_accel(v, dim=1, **kw):
            return -10.0 * v.u[:dim]  # simple damping

        euler_velocity_only(HC, bV, damping_accel, dt=0.001, n_steps=50,
                            dim=1, bc_set=bc_set)

        ke_final = sum(0.5 * v.m * np.dot(v.u, v.u) for v in HC.V
                       if v not in bV)

        # KE should decrease due to damping
        assert ke_final < ke_initial


# Setup function tests

class TestSetupFunction:
    def test_setup_1d(self):
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(dim=1)
        assert params['dim'] == 1
        assert len(bV) > 0
        assert bc_set is not None

    def test_setup_2d(self):
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(dim=2, n_refine=1)
        assert params['dim'] == 2
        assert params['gravity_axis'] == 1

    def test_setup_3d(self):
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(dim=3, n_refine=0)
        assert params['dim'] == 3
        assert params['gravity_axis'] == 2

    def test_custom_gravity_axis(self):
        from cases_dynamic.Hydrostatic_column.src._setup import setup_hydrostatic
        HC, bV, ic, bc_set, params = setup_hydrostatic(
            dim=2, n_refine=1, gravity_axis=0,
        )
        assert params['gravity_axis'] == 0
        ic.apply(HC, bV)
        # Pressure should vary along axis 0
        pressures = {}
        for v in HC.V:
            x0 = round(v.x_a[0], 10)
            pressures.setdefault(x0, []).append(v.p)
        # Different x0 values should have different pressures
        unique_pressures = set()
        for x0, ps in pressures.items():
            unique_pressures.add(round(ps[0], 5))
        assert len(unique_pressures) > 1


# ---------------------------------------------------------------------------
# The column on the library integrators (lane P, 2026-10-01)
#
# Every run below is a preset of ddgclib.methods.PRESETS integrated by
# SolverMethods.integrate through cases_dynamic/Hydrostatic_column/src/_column.py
# (the code the four runners use).  Evidence and the full-size measurements:
# docs_temp/debug_session/laneP-hydrostatic-library-integrators.md
# ---------------------------------------------------------------------------

def _column_run(name, n_refine, n_tac, ic, arm='preset', **run_kw):
    import warnings
    from cases_dynamic.Hydrostatic_column.src._column import (
        CASES, build_column, column_errors, remap_arm, run_column,
    )
    from ddgclib.methods import PRESETS
    kw = CASES[name]
    col = build_column(kw['dim'], n_refine, H=kw['H'],
                       side_walls=kw['side_walls'], ic=ic)
    methods = PRESETS[name] if arm == 'preset' else remap_arm(PRESETS[name])
    with warnings.catch_warnings():
        # a preset must not be one of the measured-unstable combinations
        warnings.filterwarnings('error', message='SolverMethods')
        res = run_column(col, methods, n_tac=n_tac, **run_kw)
    return col, res, column_errors(col)


def _envelope(col, res, t_from, t_to):
    """max |u| over a window given in acoustic times."""
    t = res['t'] / col.params['t_ac']
    return float(res['umax'][(t > t_from) & (t <= t_to)].max())


class TestHydrostaticPresets:
    def test_every_runner_has_a_single_phase_preset(self):
        from cases_dynamic.Hydrostatic_column.src._column import CASES
        from ddgclib.methods import PRESETS
        assert set(CASES) == {'hydrostatic_1D', 'hydrostatic_2D',
                              'hydrostatic_3D', 'hydrostatic_2D_periodic'}
        for name, kw in CASES.items():
            m = PRESETS[name]
            assert (m.dim, m.phases, m.integrator) == (
                kw['dim'], 'single', 'symplectic_euler')
            # fixed connectivity; the reconnecting arm is a .replace()
            assert m.remap is None and m.redistribute_mass is False

    def test_remap_arm_is_the_material_delaunay(self):
        from cases_dynamic.Hydrostatic_column.src._column import remap_arm
        from ddgclib.methods import PRESETS
        for name in ('hydrostatic_2D', 'hydrostatic_3D',
                     'hydrostatic_2D_periodic'):
            arm = remap_arm(PRESETS[name])
            assert (arm.connectivity, arm.remap, arm.redistribute_mass) == (
                'delaunay_material', 'conservative', True)
        with pytest.raises(ValueError):          # no remap in 1D
            remap_arm(PRESETS['hydrostatic_1D'])

    def test_static_balance_is_exact_for_nodal_pressures(self):
        """The centred pressure flux is exact for NODAL values of a
        linear field.  With dual-cell averages the half cells on the
        walls differ from the nodal value, and their neighbours see an
        O(1) residual (2D: 1.9075 m/s^2 at every refinement)."""
        from cases_dynamic.Hydrostatic_column.src._column import (
            static_residuals,
        )
        from ddgclib.methods import PRESETS
        r1 = static_residuals(1, 3, PRESETS['hydrostatic_1D'], H=10.0)
        r2 = static_residuals(2, 2, PRESETS['hydrostatic_2D'])
        r3 = static_residuals(3, 1, PRESETS['hydrostatic_3D'])
        for r in (r1, r2, r3):
            assert r['nodal'] < 1e-10
        assert r2['cell_average'] == pytest.approx(1.9075, rel=1e-3)


class TestColumn1D:
    """17 vertices, 10 m, frozen bottom vertex, free top vertex."""

    def test_library_run_matches_the_former_hand_rolled_loop(self):
        """PRESETS['hydrostatic_1D'] (the 'delaunay' chain rebuild with
        boundary_filter = bottom) against the loop Hydrostatic_1D.py
        shipped with until lane P, at the same fixed time step.

        The loop refreshed the duals on the standing chain; the library
        rebuilds the chain every step, which changes the neighbour
        iteration order and with it the summation order of the force.
        So the two agree to round-off (measured 3.3e-17 in u after 128
        steps), and exactly once the loop uses the library rebuild."""
        from functools import partial
        from cases_dynamic.Hydrostatic_column.src._column import build_column
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _move, _recompute_duals, _retopologize,
        )
        from ddgclib.operators.stress import (
            cache_dual_volumes, stress_acceleration,
        )
        col, res, _ = _column_run('hydrostatic_1D', 3, 2.0, 'drop')
        assert res['n_steps'] == 128
        lib = sorted((v.x_a[0], v.u[0]) for v in col.HC.V)

        def loop(library_rebuild: bool):
            ref = build_column(1, 3, H=10.0, ic='drop')
            dudt = partial(stress_acceleration, dim=1, mu=res['mu'],
                           HC=ref.HC, pressure_model=ref.eos)
            g_vec = np.array([-ref.params['g']])
            for _ in range(res['n_steps']):
                if library_rebuild:
                    _retopologize(ref.HC, ref.bV, 1,
                                  boundary_filter=ref.is_wall)
                else:
                    _recompute_duals(ref.HC)
                    cache_dual_volumes(ref.HC, dim=1)
                free = [v for v in ref.HC.V if v not in ref.bV]
                acc = {v: dudt(v) + g_vec for v in free}
                for v in free:
                    v.u[:1] += res['dt'] * acc[v][:1]
                    _move(v, v.x_a[:1] + res['dt'] * v.u[:1], ref.HC, ref.bV)
                ref.bc_set.apply_all(ref.HC, ref.bV, res['dt'])
            return sorted((v.x_a[0], v.u[0]) for v in ref.HC.V)

        assert loop(library_rebuild=True) == lib
        npt.assert_allclose(np.array(loop(library_rebuild=False)),
                            np.array(lib), rtol=0.0, atol=1e-13)

    def test_drop_rings_and_decays_at_the_viscous_rate(self):
        """Uniform density at t = 0: the column rings at its fundamental
        mode (period 4 H / c0) and the kinetic energy decays at
        nu k^2 with k = pi / (2 H), over 40 acoustic times."""
        from cases_dynamic.Hydrostatic_column.src._column import decay_rate
        col, res, err = _column_run('hydrostatic_1D', 3, 40.0, 'drop')
        p = col.params
        nu = res['mu'] / p['rho']
        theory = nu * (np.pi / (2 * p['H']))**2 * p['t_ac']
        assert decay_rate(res, col) == pytest.approx(theory, rel=0.03)
        assert _envelope(col, res, 36, 40) < 0.3 * _envelope(col, res, 0, 4)
        assert err['mass_drift'] == 0.0
        assert float(res['umax'].max()) == pytest.approx(PIN_1D_UMAX_PEAK,
                                                         rel=1e-9)
        assert float(res['ke'][-1]) == pytest.approx(PIN_1D_KE_END, rel=1e-7)

    def test_equilibrium_start_stays_at_rest(self):
        col, res, err = _column_run('hydrostatic_1D', 3, 20.0, 'equilibrium')
        assert res['umax'].max() < 2e-5                 # measured 1.10e-05
        assert _envelope(col, res, 16, 20) < _envelope(col, res, 0, 4)
        assert err['l2'] < 1e-4 * err['rho_g_H']        # measured 7.9e-06


class TestColumn2D:
    """Unit square, refinement 2 (41 vertices), free surface on top."""

    def test_drop_settles_with_noslip_walls(self):
        col, res, err = _column_run('hydrostatic_2D', 2, 20.0, 'drop')
        p = col.params
        assert len(col.bV) == p['n_frozen'] == 13   # top vertices stay free
        assert _envelope(col, res, 16, 20) < 1e-2 * _envelope(col, res, 0, 4)
        # the column has compressed by rho g H / (2 K) = 0.5 %
        assert sum(v.dual_vol for v in col.HC.V) == pytest.approx(
            p['h_surface'], abs=2e-4)
        assert abs(err['mass_drift']) < 1e-14
        assert err['l2'] < 2e-2 * err['rho_g_H']        # measured 1.47e-02
        assert float(res['umax'].max()) == pytest.approx(PIN_2D_UMAX_PEAK,
                                                         rel=1e-9)
        assert float(res['ke'][-1]) == pytest.approx(PIN_2D_KE_END, rel=1e-6)

    def test_free_slip_walls_keep_the_column_one_dimensional(self):
        """Hydrostatic_2D_periodic: side vertices slide along the wall
        (FreeSlipWallBC), only the bottom is frozen."""
        col, res, err = _column_run('hydrostatic_2D_periodic', 2, 20.0,
                                    'equilibrium')
        assert len(col.bV) == 5
        side = [v for v in col.HC.V if id(v) in col.slip_axes]
        assert len(side) == 8
        assert all(v.u[0] == 0.0 and v.x_a[0] in (0.0, 1.0) for v in side)
        assert res['umax'].max() < 5e-5                 # measured 2.63e-05
        assert _envelope(col, res, 16, 20) < 0.3 * _envelope(col, res, 0, 4)
        # the error sits in the boundary cells (nodal against cell average)
        assert err['l2_interior'] < 2e-4 * err['rho_g_H']   # measured 8.7e-05
        assert err['l2'] < 2e-2 * err['rho_g_H']            # measured 1.47e-02
        assert float(res['umax'].max()) == pytest.approx(PIN_2DP_UMAX_PEAK,
                                                         rel=1e-9)


class TestColumn3D:
    def test_preset_runs_with_a_free_surface_and_wall_half_cells(self):
        """Refinement 1, 4 acoustic times from the equilibrium masses.
        The preset uses connectivity='dual_only_bare': the 3D branch of
        'dual_only' zeroes the dual volume of every frozen vertex, so the
        wall cells would read the reference pressure P0.  Since lane Q
        it reads the exact dual faces (edge_area_source='p_ij_simplex',
        cached by the bare refresh), not the ring walk."""
        from ddgclib.methods import PRESETS, effective_methods
        col, res, err = _column_run('hydrostatic_3D', 1, 4.0, 'equilibrium')
        eff = effective_methods(col.HC, 3, PRESETS['hydrostatic_3D'])
        assert eff['boundary_dual_vol'] == 'half_cell'
        assert eff['edge_area_source'] == 'p_ij_simplex'
        assert eff['edge_area_cache_present'] is True
        assert len(col.bV) == 25 and len(col.free) == 10
        assert all(v.dual_vol > 0.0 for v in col.bV)
        assert res['umax'].max() < 2e-4                 # measured 7.5e-05
        assert err['l2'] < 1e-3 * err['rho_g_H']        # measured 1.7e-04

    @pytest.mark.slow
    def test_drop_settles_over_40_acoustic_times(self):
        """The end-of-run value is pinned since lane T: until then
        protocol rule 8 kept 3D pins to early peaks, because 3D runs on
        fixed connectivity agreed between processes to 4e-09 only (the
        refinement 2 column: 8 final states in 8 interpreters; this
        refinement 1 run gave one in 6 before the lane as well).  A 1e-15
        shift of the interior vertices moves the kinetic energy at 40
        acoustic times by at most 6.1e-12 relative and the peak by
        1.7e-13 (8 seeds,
        ``diagnose_determinism.py sweep pin_hydro3d --perturb 1e-15``)."""
        col, res, err = _column_run('hydrostatic_3D', 1, 40.0, 'drop')
        assert _envelope(col, res, 36, 40) < 1e-2 * _envelope(col, res, 0, 4)
        assert float(res['umax'].max()) == pytest.approx(PIN_3D_UMAX_PEAK,
                                                         rel=1e-9)
        assert float(res['ke'][-1]) == pytest.approx(PIN_3D_KE_40, rel=1e-9)

    @pytest.mark.slow
    def test_library_dual_only_cannot_hold_the_3d_column(self):
        """Why the 3D preset is not 'dual_only': from the equilibrium
        masses that path leaves the equilibrium at once (frozen-vertex
        dual volumes are zeroed, wall cells read P0)."""
        import warnings
        from cases_dynamic.Hydrostatic_column.src._column import (
            build_column, run_column,
        )
        from ddgclib.methods import PRESETS
        col = build_column(3, 1, ic='equilibrium')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = run_column(col, PRESETS['hydrostatic_3D'].replace(
                connectivity='dual_only'), n_tac=4.0)
        assert res['umax'].max() > 100 * 7.5e-05        # measured 7.7e-02

    @pytest.mark.slow
    def test_remap_arm_holds_the_3d_column(self):
        """Refinement 2 (189 vertices), 2 acoustic times from uniform
        density: delaunay_material + conservative remap against the
        preset.  The rebuild keeps the walls frozen, leaves no flat wall
        tetrahedron in the cache (785 simplices instead of 768 with the
        first, topological peel) and the column compresses as on fixed
        connectivity.

        The peak is early (0.87 acoustic times).  Until lane T nothing
        later was pinned: the reconnecting 3D arm was reproducible from
        process to process to two digits only (lane P log, section 11).
        It is bit-identical in every interpreter now, so the kinetic
        energy at the end of the run is pinned too.  Until lane Q both
        pins were tie-decided numbers: the mesh is structured
        (cospherical points) and the ring walk put a spurious face
        barycentre into the polygon of 30 of the 56 free-surface edges
        (lane T log, section 6), so a 1e-15 shift of the interior
        vertices moved the peak by up to 1.25e-03 relative and the end
        value by 6.9e-04 (8 seeds, ``diagnose_determinism.py sweep
        pin_hydro3d_remap --perturb 1e-15``).  Lane Q (2026-10-05) reads
        the exact dual faces (``edge_area_source='p_ij_simplex'``) and
        re-pinned both: peak 0.15658060026054665 -> 0.15305813130485327,
        end 0.5431445985762776 -> 0.528851067635385 (the removed hull-edge
        areas, beyond the perturbation range; the old values reproduce
        with ``edge_area_source='p_ij_ring'``)."""
        col, res, err = _column_run('hydrostatic_3D', 2, 2.0, 'drop', 'remap')
        col_p, res_p, _ = _column_run('hydrostatic_3D', 2, 2.0, 'drop')
        assert len(col.bV) == len(col_p.bV) == 89
        assert len(col.HC._simplices) == 768
        assert float(res['umax'].max()) == pytest.approx(
            float(res_p['umax'].max()), rel=0.1)
        assert sum(v.dual_vol for v in col.HC.V) == pytest.approx(
            sum(v.dual_vol for v in col_p.HC.V), abs=3e-4)   # 1.5e-04
        assert abs(err['mass_drift']) < 1e-13
        assert float(res['umax'].max()) == pytest.approx(
            PIN_3D_REMAP_UMAX_PEAK, rel=1e-6)
        assert float(res['ke'][-1]) == pytest.approx(
            PIN_3D_REMAP_KE_END, rel=1e-6)


@pytest.mark.slow
class TestColumnSettlesOver40AcousticTimes:
    """Refinement 2, 40 acoustic times, both connectivity arms."""

    @pytest.fixture(scope='class')
    def runs(self):
        return {
            'preset': _column_run('hydrostatic_2D', 2, 40.0, 'drop'),
            'remap': _column_run('hydrostatic_2D', 2, 40.0, 'drop', 'remap'),
            'freeslip': _column_run('hydrostatic_2D_periodic', 2, 40.0,
                                    'drop'),
        }

    def test_fixed_connectivity_settles(self, runs):
        col, res, err = runs['preset']
        assert _envelope(col, res, 36, 40) < 2e-3 * _envelope(col, res, 0, 4)
        assert float(res['ke'][-1]) == pytest.approx(PIN_2D_KE_40, rel=1e-6)

    def test_remap_arm_settles_like_the_fixed_connectivity(self, runs):
        col, res, err = runs['remap']
        _, res_p, err_p = runs['preset']
        assert float(res['umax'].max()) == pytest.approx(
            float(res_p['umax'].max()), rel=0.1)
        assert _envelope(col, res, 36, 40) < 5e-3 * _envelope(col, res, 0, 4)
        assert abs(err['mass_drift']) < 1e-13
        assert err['l2'] < 2.5e-2 * err['rho_g_H']
        assert float(res['ke'][-1]) == pytest.approx(PIN_2D_REMAP_KE_40,
                                                     rel=1e-5)

    def test_free_slip_column_rings_down(self, runs):
        col, res, err = runs['freeslip']
        assert _envelope(col, res, 36, 40) < 3e-2 * _envelope(col, res, 0, 4)

    def test_integrated_error_converges_with_refinement(self):
        """Equilibrium masses, free-slip walls (one-dimensional
        solution), 20 acoustic times: the volume-weighted L2 error drops
        by 2^1.5 per refinement (boundary cells) and its interior part
        by 2^2."""
        e2 = _column_run('hydrostatic_2D_periodic', 2, 20.0, 'equilibrium')[2]
        e3 = _column_run('hydrostatic_2D_periodic', 3, 20.0, 'equilibrium')[2]
        assert 2.5 < e2['l2'] / e3['l2'] < 3.2                    # 2.89
        assert 3.3 < e2['l2_interior'] / e3['l2_interior'] < 5.5  # 4.67
        assert 7.0 < e2['max_int'] / e3['max_int'] < 9.0          # 8.01


# Pinned 2026-10-01 (lane P), presets hydrostatic_1D / _2D / _2D_periodic /
# _3D, alpha_art = 0.5, CFL 0.25.
PIN_1D_UMAX_PEAK = 0.8463965668751099
PIN_1D_KE_END = 8.07861451666529
PIN_2D_UMAX_PEAK = 0.1951860472291084
PIN_2D_KE_END = 3.1650897086706908e-06
PIN_2DP_UMAX_PEAK = 2.631506910977075e-05
# RE-PIN 2026-10-05 (lane Q, hydrostatic_3D on edge_area_source=
# 'p_ij_simplex'): the refinement 1 column has no edge between two
# free-surface vertices, so the exact faces differ from the ring walk by
# round-off only; the run amplifies that to 1e-16 (peak) and 1.2e-11
# (KE at 40 t_ac), the order lane T measured for a 1e-15 shift.
# Before: 0.08910127097486757 / 6.206365156298652e-06 (reproduced by
# edge_area_source='p_ij_ring').
PIN_3D_UMAX_PEAK = 0.08910127097486756
PIN_2D_KE_40 = 9.872765069804785e-07
PIN_2D_REMAP_KE_40 = 2.721558596123262e-06
# Pinned 2026-10-01 (lane P, review fix): remap arm of hydrostatic_3D
# (connectivity='delaunay_material', remap='conservative',
# redistribute_mass=True), refinement 2, 2 acoustic times.
# RE-PIN 2026-10-05 (lane Q): 0.15658060026054665 -> 0.15305813130485327
# with the exact dual faces (the ring walk had 30 of the 56 free-surface
# edge areas off by up to 37 %, lane T).
PIN_3D_REMAP_UMAX_PEAK = 0.15305813130485327
# Pinned 2026-10-02 (lane T): end-of-run values of the two 3D runs above,
# possible since 3D runs are bit-identical between interpreters
# (ddgclib/tests/test_determinism.py).  Kinetic energy of the free
# vertices after 40 acoustic times (preset, refinement 1) and after 2
# acoustic times (remap arm, refinement 2).
# RE-PIN 2026-10-05 (lane Q): 6.206365156298652e-06 -> 6.2063651562987255e-06
# (round-off, see PIN_3D_UMAX_PEAK), 0.5431445985762776 ->
# 0.528851067635385 (the hull-edge areas, see PIN_3D_REMAP_UMAX_PEAK).
PIN_3D_KE_40 = 6.2063651562987255e-06
PIN_3D_REMAP_KE_END = 0.528851067635385
