"""Tests for the Hagen-Poiseuille (planar 2D) case.

Validates:
1. Analytical Poiseuille velocity profile is correctly applied.
2. At equilibrium, the pressure gradient is non-zero but balanced.
3. Starting from uniform plug flow with BCs, velocity develops toward parabolic.
"""

import numpy as np
import numpy.testing as npt
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Fixtures

@pytest.fixture
def poiseuille_2d():
    from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
    HC, bV, ic, bc_set, params = setup_poiseuille_2d(
        G=1.0, mu=1.0, n_refine=2, L=1.0, h=1.0,
    )
    ic.apply(HC, bV)
    return HC, bV, ic, bc_set, params


# Analytical profile verification

class TestAnalyticalProfile:
    def test_velocity_profile(self, poiseuille_2d):
        """u_x(y) = (G/(2*mu)) * y * (h - y), u_y = 0."""
        HC, bV, _, _, p = poiseuille_2d
        G, mu, h = p['G'], p['mu'], p['h']

        for v in HC.V:
            y = v.x_a[1]
            u_anal = (G / (2 * mu)) * y * (h - y)
            npt.assert_allclose(v.u[0], u_anal, atol=1e-10,
                                err_msg=f"Wrong u_x at y={y}")
            npt.assert_allclose(v.u[1], 0.0, atol=1e-14)

    def test_pressure_gradient(self, poiseuille_2d):
        """P = -G * x, so vertices at different x should have different P."""
        HC, bV, _, _, p = poiseuille_2d
        G = p['G']

        for v in HC.V:
            expected = -G * v.x_a[0]
            npt.assert_allclose(v.p, expected, atol=1e-10)

    def test_wall_velocity_zero(self, poiseuille_2d):
        """Velocity at y=0 and y=h should be zero."""
        HC, bV, _, _, p = poiseuille_2d
        h = p['h']

        for v in HC.V:
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14:
                npt.assert_allclose(v.u[0], 0.0, atol=1e-14)

    def test_max_velocity_at_centerline(self, poiseuille_2d):
        """Maximum velocity should be at y = h/2."""
        HC, bV, _, _, p = poiseuille_2d
        G, mu, h = p['G'], p['mu'], p['h']
        U_max = G * h**2 / (8 * mu)

        # Find vertex closest to centerline
        center_v = min(
            (v for v in HC.V if abs(v.x_a[1] - h/2) < 0.1),
            key=lambda v: abs(v.x_a[1] - h/2)
        )
        npt.assert_allclose(center_v.u[0], U_max, atol=0.01)


# Developing flow test

class TestDevelopingFlow:
    def test_plug_flow_develops(self, poiseuille_2d):
        """Start with uniform plug flow, run integrator.

        With no-slip BCs, velocity near walls should decrease
        and center should begin to increase.
        """
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV, _, bc_set, p = poiseuille_2d

        # Override to uniform plug flow
        for v in HC.V:
            v.u = np.array([0.1, 0.0])

        # Simple viscous damping at walls (mock acceleration)
        def mock_accel(v, dim=2, **kw):
            # Laplacian-like: neighbors' average minus self
            if not v.nn:
                return np.zeros(dim)
            avg = np.mean([nb.u[:dim] for nb in v.nn], axis=0)
            return 5.0 * (avg - v.u[:dim])

        # Record initial state
        wall_u_initial = [abs(v.u[0]) for v in bV
                          if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14]

        euler_velocity_only(HC, bV, mock_accel, dt=0.001, n_steps=10,
                            dim=2, bc_set=bc_set)

        # Wall velocities should be zero (no-slip enforced)
        for v in bV:
            if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - 1.0) < 1e-14:
                npt.assert_array_equal(v.u, np.zeros(2))


# Setup function tests

class TestSetupFunction:
    def test_setup_creates_mesh(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        HC, bV, ic, bc_set, params = setup_poiseuille_2d(n_refine=1)
        assert sum(1 for _ in HC.V) > 0
        assert len(bV) > 0
        assert params['dim'] == 2

    def test_poiseuille_ic_object(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        _, _, _, _, params = setup_poiseuille_2d()
        pic = params['poiseuille_ic']
        # analytical_velocity returns a scalar (flow-axis component)
        u_flow = pic.analytical_velocity(np.array([0.5, 0.5]))
        assert u_flow > 0  # Flow in x-direction at y=0.5

    def test_custom_parameters(self):
        from cases_dynamic.Hagen_Poiseuile.src._setup import setup_poiseuille_2d
        _, _, _, _, params = setup_poiseuille_2d(G=2.0, mu=0.5)
        assert params['G'] == 2.0
        assert params['mu'] == 0.5
        # U_max = G*h^2/(8*mu) = 2*1/(8*0.5) = 0.5
        npt.assert_allclose(params['U_max'], 0.5)


# ---------------------------------------------------------------------------
# Developing Lagrangian flow through the presets (laneH, 2026-10-01)
# ---------------------------------------------------------------------------
# Plug flow enters, the pressure field G (L - x) is prescribed, the mesh
# moves with the fluid.  Every run is PRESETS['hagen_poiseuille_2D'] /
# ['hagen_poiseuille_3D'] (or a .replace(...) arm) integrated by the library
# on cases_dynamic/Hagen_Poiseuile/src/_setup.py:setup_poiseuille_developing.
# The velocity is compared with the developed profile by a dual-volume
# weighted l2 norm on the downstream half of the channel.

# 2D, refinement 1 (48 vertices), mu = 0.1 (t_dev 1.01 s), 500 steps of
# 0.02 = one inlet period = 9.9 t_dev.  2D runs are bit-identical from
# process to process (protocol rule 8).
PIN_2D_L2 = 0.013084885355719682
PIN_2D_UMAX = 0.1495783625570828
# 2D, refinement 2 (171 vertices), same horizon
PIN_2D_R2_L2 = 0.005388811850628623
# 3D, refinement 1 (91 vertices), mu = 0.1 (t_dev 0.43 s), 300 steps of 0.01
PIN_3D_L2 = 0.07252269818859532
PIN_3D_UMAX = 0.19558769279936808


def _developing(name, n_steps, **kw):
    from ddgclib.methods import PRESETS
    from cases_dynamic.Hagen_Poiseuile.src._run import run_developing
    changes = kw.pop('changes', {})
    methods = PRESETS[name]
    if changes:
        methods = methods.replace(**changes)
    return run_developing(methods, n_steps=n_steps, **kw)


_KW_2D = dict(dim=2, L=3.0, mu=0.1, n_refine=1, dt=0.02)
_KW_3D = dict(dim=3, L=2.0, mu=0.1, n_refine=1, dt=0.01)


class TestDeveloping2D:
    """The Lagrangian channel develops the Poiseuille profile from a plug
    through the preset."""

    @pytest.fixture(scope='class')
    def run(self):
        return _developing('hagen_poiseuille_2D', 500, **_KW_2D)

    def test_profile_is_pinned(self, run):
        npt.assert_allclose(run['profile']['l2'], PIN_2D_L2, rtol=1e-9)
        npt.assert_allclose(run['profile']['u_max'], PIN_2D_UMAX, rtol=1e-9)

    def test_develops_from_the_plug(self, run):
        """The plug is 0.4 away from the parabola in this norm; after ten
        time constants 1.3 % is left (the resolution of 3 fluid rows)."""
        assert run['l2'][0] > 0.2
        assert run['profile']['l2'] < 0.02
        assert np.max(run['l2'][len(run['l2']) * 3 // 4:]) < 0.03
        p = run['params']
        npt.assert_allclose(p['U_max'], 1.5 * p['U_avg'])
        assert abs(run['profile']['u_max'] / p['U_max'] - 1.0) < 0.01

    def test_no_transverse_motion_and_nothing_leaves(self, run):
        """The pressure force of the prescribed linear field is exact, so
        no vertex leaves its row."""
        assert np.max(run['u_cross']) < 1e-14
        assert np.max(run['n_outside']) == 0

    def test_walls_hold(self, run):
        w = run['walls']
        assert w['n_wall_start'] == 18
        assert w['n_frozen'] == 18 and w['n_moved'] == 0
        assert w['max_displacement'] == 0.0

    def test_mass_flux_in_equals_mass_flux_out(self, run):
        """Over one inlet period exactly one unit cell of fluid mass
        crosses the inlet plane.  The same mass crosses the outlet in
        THIS configuration: a count, not a conservation law.  Each of the
        three fluid rows sends two columns across ``x = L`` within the
        10 s, although the rows move at different speeds, and the count
        starts after the first step (the column that starts on the outlet
        plane is not in it).  With another horizon or step the two
        numbers differ by whole columns (shipped 2D run: 0.868 out
        against 0.833 in, because the horizon is shorter than the transit
        time of the slowest row; ``L = 4`` arm: 1.000 against 0.833).
        Kept as a pin of the marker bookkeeping at inlet and outlet."""
        fluid = run['params']['fluid_fraction']
        assert run['flux_periods'] == 1
        npt.assert_allclose(fluid, 2.0 / 3.0, rtol=1e-12)
        npt.assert_allclose(run['mass_flux_in'], fluid, rtol=1e-12)
        npt.assert_allclose(run['mass_flux_out'], fluid, rtol=1e-12)

    def test_no_pile_up_and_no_depletion(self, run):
        """18 free vertices are in the channel at the start.  The total
        grows from 43 by what the buffer behind the outlet holds."""
        assert 15 <= np.min(run['n_channel'])
        assert np.max(run['n_channel']) <= 20
        assert np.min(run['n_total']) == 43 and np.max(run['n_total']) <= 49
        census = run['census']
        assert census['inlet_buffer'] == 6 and census['channel'] == 18
        assert census['total'] == 18 + 6 + 18 + census['outlet_buffer']

    def test_two_point_viscous_flux_does_not_reach_the_profile(self):
        """The arm that attributes the result: the same run with
        ``viscous_flux='two_point'`` stays 30 % off (the two-point flux is
        not linearly precise on the sheared mesh)."""
        arm = _developing('hagen_poiseuille_2D', 500,
                          changes=dict(viscous_flux='two_point'), **_KW_2D)
        assert arm['profile']['l2'] > 0.25
        assert arm['profile']['u_max'] > 1.25 * arm['params']['U_max']


class TestDeveloping3DBackendAxis:
    """``--backend`` of the 3D runner and of ``run_cluster.py``: the
    preset with the ``backend`` axis replaced runs (it stopped at the
    first retopology before the fix round of laneH) and gives the serial
    numpy result.  ``'gpu'`` falls back to numpy when PyTorch is absent;
    the preset's forces read no dual face area, so the result does not
    depend on who computed the area cache."""

    @pytest.fixture(scope='class')
    def reference(self):
        return _developing('hagen_poiseuille_3D', 20, **_KW_3D)

    @pytest.mark.parametrize('backend', ['gpu', 'multiprocessing'])
    def test_backend_arm_runs_and_matches(self, reference, backend):
        from ddgclib.dynamic_integrators import _integrators_dynamic as mod
        try:
            r = _developing('hagen_poiseuille_3D', 20,
                            changes=dict(backend=backend), **_KW_3D)
        finally:
            pool = getattr(mod._BACKEND_INSTANCES.pop('multiprocessing', None),
                           'pool', None)
            if pool is not None:
                pool.terminate()
        assert r['profile']['l2'] == reference['profile']['l2']
        assert r['profile']['u_max'] == reference['profile']['u_max']
        assert r['census'] == reference['census']


@pytest.mark.slow
class TestDevelopingSlow:
    def test_2d_refinement_2(self):
        r = _developing('hagen_poiseuille_2D', 500,
                        **(_KW_2D | dict(n_refine=2)))
        npt.assert_allclose(r['profile']['l2'], PIN_2D_R2_L2, rtol=1e-9)
        assert r['profile']['l2'] < 0.5 * PIN_2D_L2      # it converges
        assert np.max(r['u_cross']) < 1e-14
        assert np.max(r['n_outside']) == 0

    def test_3d_pipe_develops_through_the_preset(self):
        """Octagonal pipe (refinement 1): 7 % from the profile of the
        circular pipe after 7 time constants, no radial motion.  The
        simplex cache is never stale: after the BCs of a step it is
        either dropped (a vertex entered or left) or made of live
        vertices only."""
        seen = {'cached': 0, 'stale': 0}

        def check_cache(step, t, HC, bV=None, diagnostics=None):
            simplices = HC._simplices
            if simplices is None:
                return
            seen['cached'] += 1
            cache = HC.V.cache
            seen['stale'] += any(cache.get(v.x) is not v
                                 for s in simplices for v in s)

        r = _developing('hagen_poiseuille_3D', 300, callback=check_cache,
                        **_KW_3D)
        assert seen['cached'] > 100 and seen['stale'] == 0
        npt.assert_allclose(r['profile']['l2'], PIN_3D_L2, rtol=1e-6)
        npt.assert_allclose(r['profile']['u_max'], PIN_3D_UMAX, rtol=1e-6)
        assert r['l2'][0] > 0.3
        assert np.max(r['u_cross']) < 1e-14
        assert np.max(r['n_outside']) == 0
        w = r['walls']
        assert w['n_frozen'] == w['n_wall_start'] == 56
        assert w['n_moved'] == 0
        assert 16 <= np.min(r['n_channel']) and np.max(r['n_channel']) <= 26

    def test_3d_centred_flux_on_the_area_cache_drifts_radially(self):
        """Why the 3D preset uses ``pressure_flux='simplex_gradient'``:
        with the centred flux the pressure force reads the batch_e_star
        area cache, which is not linearly precise, and the vertices pick
        up radial velocity."""
        r = _developing('hagen_poiseuille_3D', 300,
                        changes=dict(pressure_flux='centred'), **_KW_3D)
        assert np.max(r['u_cross']) > 1e-3
        assert r['profile']['l2'] > 1.2 * PIN_3D_L2
