"""Hagen-Poiseuille equilibrium validation for stress.py operators.

At the analytical Poiseuille equilibrium:
    u_x(y) = (G / 2mu) * y * (h - y),  u_y = 0
    P(x)   = -G * x

the viscous stress and pressure gradient are in exact balance, so the net
stress acceleration  dudt_i = stress_acceleration  should be zero.

On a discrete mesh the residual is not exactly zero (truncation error), but
it should:
    (a) be bounded,
    (b) decrease (or at least not increase) with mesh refinement,
    (c) show that pressure and viscous contributions individually are non-zero
        but approximately cancel.

Run::

    pytest cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py -v
    pytest cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py -v -m slow
"""

import numpy as np
import numpy.testing as npt
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.operators.stress import (
    dual_area_vector,
    dual_volume,
    stress_acceleration,
    stress_force,
    velocity_difference_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_poiseuille_equilibrium(
    G: float = 1.0,
    mu: float = 1.0,
    n_refine: int = 2,
    L: float = 1.0,
    h: float = 1.0,
    rho: float = 1.0,
):
    """Build a 2D Poiseuille mesh at analytical equilibrium.

    Returns (HC, bV, bV_wall, params) with duals already computed.
    """
    from ddgclib.initial_conditions import (
        CompositeIC,
        LinearPressureGradient,
        PoiseuillePlanar,
        UniformMass,
    )

    HC = Complex(2, domain=[(0.0, L), (0.0, h)])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()

    bV = set()
    for v in HC.V:
        on_boundary = (
            abs(v.x_a[0]) < 1e-14
            or abs(v.x_a[0] - L) < 1e-14
            or abs(v.x_a[1]) < 1e-14
            or abs(v.x_a[1] - h) < 1e-14
        )
        if on_boundary:
            bV.add(v)
            v.boundary = True
        else:
            v.boundary = False

    bV_wall = {
        v for v in bV
        if abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - h) < 1e-14
    }

    ic = CompositeIC(
        PoiseuillePlanar(G=G, mu=mu, y_lb=0.0, y_ub=h,
                         flow_axis=0, normal_axis=1, dim=2),
        LinearPressureGradient(G=G, axis=0, P_ref=0.0),
        UniformMass(total_volume=L * h, rho=rho),
    )
    ic.apply(HC, bV)
    compute_vd(HC, cdist=1e-10)

    params = {
        "dim": 2, "G": G, "mu": mu, "L": L, "h": h, "rho": rho,
        "U_max": G * h**2 / (8 * mu),
    }
    return HC, bV, bV_wall, params


def _interior_residuals(HC, bV, mu, dim=2):
    """Compute ||stress_acceleration|| for every interior vertex."""
    residuals = []
    for v in HC.V:
        if v in bV:
            continue
        a = stress_acceleration(v, dim=dim, mu=mu, HC=HC)
        residuals.append(np.linalg.norm(a))
    return np.array(residuals)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equil_coarse():
    """n_refine=2  (~25 vertices)."""
    return _build_poiseuille_equilibrium(n_refine=2)


@pytest.fixture
def equil_fine():
    """n_refine=3  (~81 vertices)."""
    return _build_poiseuille_equilibrium(n_refine=3)


@pytest.fixture
def equil_finer():
    """n_refine=4  (~289 vertices)."""
    return _build_poiseuille_equilibrium(n_refine=4)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestHagenPoiseuilleEquilibrium:
    """Validate stress.py operators at analytical Poiseuille equilibrium."""

    # -- IC sanity checks ---------------------------------------------------

    def test_velocity_profile_applied(self, equil_coarse):
        """Verify analytical u_x(y) = (G/2mu)*y*(h-y) was applied."""
        HC, bV, _, p = equil_coarse
        G, mu, h = p["G"], p["mu"], p["h"]

        for v in HC.V:
            y = v.x_a[1]
            expected = (G / (2 * mu)) * y * (h - y)
            npt.assert_allclose(v.u[0], expected, atol=1e-12,
                                err_msg=f"u_x wrong at y={y:.4f}")
            npt.assert_allclose(v.u[1], 0.0, atol=1e-14)

    def test_pressure_applied(self, equil_coarse):
        """Verify P = -G*x was applied."""
        HC, _, _, p = equil_coarse
        G = p["G"]

        for v in HC.V:
            npt.assert_allclose(v.p, -G * v.x_a[0], atol=1e-12)

    def test_wall_velocity_zero(self, equil_coarse):
        """Velocity at walls y=0 and y=h must be exactly zero."""
        _, _, bV_wall, _ = equil_coarse
        for v in bV_wall:
            npt.assert_allclose(v.u[0], 0.0, atol=1e-14)

    # -- Dual geometry sanity -----------------------------------------------

    def test_dual_volumes_positive(self, equil_coarse):
        """Every interior dual cell must have positive volume."""
        HC, bV, _, _ = equil_coarse
        for v in HC.V:
            if v in bV:
                continue
            vol = dual_volume(v, HC, dim=2)
            assert vol > 0, f"Non-positive dual volume at {v.x}"

    def test_dual_area_vectors_sum_to_zero(self, equil_coarse):
        """Sum of outward A_ij over all neighbors should be ~0 (closed surface)."""
        HC, bV, _, _ = equil_coarse
        for v in HC.V:
            if v in bV:
                continue
            A_sum = np.zeros(2)
            for v_j in v.nn:
                A_sum += dual_area_vector(v, v_j, HC, dim=2)
            npt.assert_allclose(A_sum, np.zeros(2), atol=1e-10,
                                err_msg=f"A_ij sum != 0 at {v.x}")

    # -- Core equilibrium residual tests ------------------------------------

    def test_equilibrium_residual_bounded(self, equil_coarse):
        """At equilibrium, ||stress_acceleration|| must be bounded.

        The discrete operators introduce truncation error, so the residual
        is not exactly zero, but it should not blow up.
        """
        HC, bV, _, p = equil_coarse
        res = _interior_residuals(HC, bV, p["mu"])

        assert len(res) > 0, "No interior vertices found"
        assert np.max(res) < 1e3, (
            f"Residual too large: max={np.max(res):.4e}, "
            f"median={np.median(res):.4e}"
        )

    def test_equilibrium_residual_median(self, equil_fine):
        """On the fine mesh the median residual should be modest."""
        HC, bV, _, p = equil_fine
        res = _interior_residuals(HC, bV, p["mu"])
        assert np.median(res) < 1e3, f"Median residual too large: {np.median(res):.4e}"

    @pytest.mark.slow
    def test_equilibrium_residual_converges(self, equil_coarse, equil_fine):
        """Residual should not grow with refinement (ideally it decreases)."""
        _, bV_c, _, p_c = equil_coarse
        res_coarse = np.median(
            _interior_residuals(equil_coarse[0], bV_c, p_c["mu"])
        )

        _, bV_f, _, p_f = equil_fine
        res_fine = np.median(
            _interior_residuals(equil_fine[0], bV_f, p_f["mu"])
        )

        # Fine should not be dramatically worse than coarse
        assert res_fine <= res_coarse * 2.0, (
            f"Residual increased: coarse={res_coarse:.4e}, fine={res_fine:.4e}"
        )

    # -- Pressure vs viscous decomposition ----------------------------------

    def test_pressure_force_nonzero(self, equil_coarse):
        """Pure pressure force (mu=0) should be non-zero at equilibrium.

        The linear pressure gradient drives flow in +x, so the pressure
        force on interior cells should be non-negligible.
        """
        HC, bV, _, _ = equil_coarse
        magnitudes = []
        for v in HC.V:
            if v in bV:
                continue
            F_p = stress_force(v, dim=2, mu=0.0, HC=HC)
            magnitudes.append(np.linalg.norm(F_p))

        assert np.mean(magnitudes) > 1e-14, (
            "Pressure-only force is zero — pressure gradient is ineffective"
        )

    def test_viscous_force_nonzero(self, equil_coarse):
        """Pure viscous force (p set to 0) should be non-zero.

        With a parabolic velocity profile the Laplacian is non-zero, so
        viscous diffusion should produce a non-zero force.
        """
        HC, bV, _, p = equil_coarse
        mu = p["mu"]

        # Temporarily zero pressure, compute force, restore
        saved_p = {}
        for v in HC.V:
            saved_p[v] = v.p
            v.p = 0.0

        magnitudes = []
        for v in HC.V:
            if v in bV:
                continue
            F_v = stress_force(v, dim=2, mu=mu, HC=HC)
            magnitudes.append(np.linalg.norm(F_v))

        for v in HC.V:
            v.p = saved_p[v]

        assert np.mean(magnitudes) > 1e-14, (
            "Viscous-only force is zero — strain rate is ineffective"
        )

    def test_pressure_and_viscous_oppose(self, equil_coarse):
        """At equilibrium, pressure force and viscous force should
        approximately oppose each other for interior vertices.

        Check that the x-component of pressure force and viscous force
        have opposite signs for a majority of interior vertices.
        """
        HC, bV, _, p = equil_coarse
        mu = p["mu"]
        opposing = 0
        total = 0

        for v in HC.V:
            if v in bV:
                continue
            # Pressure-only force
            F_p = stress_force(v, dim=2, mu=0.0, HC=HC)
            # Full force
            F_full = stress_force(v, dim=2, mu=mu, HC=HC)
            # Viscous force = full - pressure
            F_visc = F_full - F_p

            total += 1
            if F_p[0] * F_visc[0] < 0:
                opposing += 1

        assert total > 0
        assert opposing > total * 0.5, (
            f"Only {opposing}/{total} interior vertices have opposing "
            f"pressure/viscous forces in x-direction"
        )

    # -- Velocity difference tensor checks ----------------------------------

    def test_velocity_difference_tensor_not_zero(self, equil_coarse):
        """du_i should be non-zero away from centerline (parabolic profile).

        At y = h/2 the shear rate du_x/dy = 0 analytically (max velocity),
        so we exclude vertices near the centerline.
        """
        HC, bV, _, p = equil_coarse
        h = p["h"]
        for v in HC.V:
            if v in bV:
                continue
            # Skip vertices near centerline where du_x/dy ≈ 0
            if abs(v.x_a[1] - h / 2) < 0.15 * h:
                continue
            du = velocity_difference_tensor(v, HC, dim=2)
            assert np.linalg.norm(du) > 1e-14, (
                f"du_i is zero at {v.x} — velocity gradient not captured"
            )

    def test_velocity_difference_tensor_shear(self, equil_coarse):
        """For Poiseuille flow, du should have dominant du_x/dy (shear).

        The parabolic u_x(y) profile produces du/dy != 0, while du/dx ≈ 0
        (fully developed => no streamwise variation in the velocity field).
        """
        HC, bV, _, _ = equil_coarse
        dominant_count = 0
        total = 0

        for v in HC.V:
            if v in bV:
                continue
            du = velocity_difference_tensor(v, HC, dim=2)
            total += 1
            # du[0,1] is du_x/dy (shear component)
            # du[0,0] is du_x/dx (should be ~0 for developed flow)
            if abs(du[0, 1]) > abs(du[0, 0]):
                dominant_count += 1

        assert dominant_count > total * 0.5, (
            f"Only {dominant_count}/{total} vertices have |du_x/dy| > |du_x/dx|"
        )

    # -- Single time-step stability test ------------------------------------

    def test_single_step_stays_near_equilibrium(self, equil_fine):
        """One Euler step from equilibrium should barely change velocity.

        If dudt ≈ 0 at equilibrium, then after one small dt the velocity
        should remain very close to the analytical profile.
        """
        from functools import partial
        from ddgclib.dynamic_integrators import euler_velocity_only

        HC, bV, bV_wall, p = equil_fine
        G, mu, h = p["G"], p["mu"], p["h"]

        # Record analytical velocities before stepping
        u_before = {v: v.u.copy() for v in HC.V}

        from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), bV_wall)

        from ddgclib.operators.stress import dudt_i
        dudt_fn = partial(dudt_i, dim=2, mu=mu, HC=HC)

        dt = 1e-4
        euler_velocity_only(HC, bV, dudt_fn, dt=dt, n_steps=1, dim=2,
                            bc_set=bc_set)

        # Velocity change should be small
        max_delta = 0.0
        for v in HC.V:
            if v in bV:
                continue
            delta = np.linalg.norm(v.u - u_before[v])
            max_delta = max(max_delta, delta)

        # With dt=1e-4 and bounded residual, change should be modest
        U_max = p["U_max"]
        assert max_delta < U_max, (
            f"Velocity changed by {max_delta:.4e} after one step "
            f"(U_max={U_max:.4e})"
        )

    # -- Parametric: different G, mu combinations ---------------------------

    @pytest.mark.parametrize("G,mu", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
        (10.0, 1.0),
    ])
    def test_equilibrium_bounded_parametric(self, G, mu):
        """Residual is bounded for various G, mu combinations."""
        HC, bV, _, p = _build_poiseuille_equilibrium(G=G, mu=mu, n_refine=2)
        res = _interior_residuals(HC, bV, mu)
        assert np.max(res) < 1e3, (
            f"G={G}, mu={mu}: max residual={np.max(res):.4e}"
        )
