"""Tests for the switchable pressure flux (method axis ``pressure_flux``)
and the gradient-corrected density diffusion (axis ``density_diffusion``).

Added 2026-09-26 with the capillary-rise free-surface diagnosis
(cases_dynamic/capillary_rise_energy_grad).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from hyperct import Complex
from hyperct.ddg import connect_and_cache_simplices

from ddgclib.eos import TaitMurnaghan
from ddgclib.methods import SolverMethods
from ddgclib.operators.stabilisation import density_diffusion_step
from ddgclib.operators.stress import (
    cache_dual_volumes, dudt_i, pressure_flux, pressure_flux_methods,
    pressure_flux_riemann, stress_force,
)
from ddgclib.dynamic_integrators._integrators_dynamic import _recompute_duals


def _strip(nx=6, ny=8, dx=0.1):
    """Structured 2D triangle strip with a simplex cache and duals."""
    HC = Complex(2, domain=[(0.0, nx * dx), (0.0, ny * dx)])
    idx, verts = {}, []
    for j in range(ny + 1):
        for i in range(nx + 1):
            v = HC.V[(i * dx, j * dx)]
            idx[(i, j)] = len(verts)
            verts.append(v)
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    connect_and_cache_simplices(HC, verts, 2, simplices=np.array(tris))
    from hyperct.ddg import boundary_from_simplices
    bset = set(boundary_from_simplices(HC, 2))
    for v in HC.V:
        v.boundary = v in bset
    _recompute_duals(HC)
    cache_dual_volumes(HC, dim=2)
    return HC, verts, bset


EOS = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1000.0 * 4.0 ** 2, n=1.0, rho_clip=(0.3, 3.0))


class TestRiemannFlux:
    def test_registry(self):
        assert pressure_flux_methods["centred"] is pressure_flux
        assert pressure_flux_methods["acoustic-riemann"] is pressure_flux_riemann
        assert set(pressure_flux_methods.available()) == {
            "centred", "acoustic-riemann", "simplex_gradient"}

    def test_reduces_to_centred_without_velocity_jump(self):
        A = np.array([0.3, -0.1])
        u = np.array([0.5, -0.2])
        F_c = pressure_flux(10.0, 4.0, A)
        F_r = pressure_flux_riemann(10.0, 4.0, 1000.0, 1010.0, 4.0, 4.0, u, u.copy(), A)
        assert np.allclose(F_r, F_c, atol=0, rtol=1e-15)
        # tangential velocity jump: no effect
        t = np.array([A[1], -A[0]])
        F_t = pressure_flux_riemann(10.0, 4.0, 1000.0, 1010.0, 4.0, 4.0, u, u + 3.0 * t, A)
        assert np.allclose(F_t, F_c, atol=1e-12)

    def test_antisymmetric_and_damping(self):
        A = np.array([0.3, -0.1])
        n = A / np.linalg.norm(A)
        u_i, u_j = np.zeros(2), 0.2 * n           # j moves away from i along n
        F_ij = pressure_flux_riemann(0.0, 0.0, 1000.0, 1000.0, 4.0, 4.0, u_i, u_j, A)
        F_ji = pressure_flux_riemann(0.0, 0.0, 1000.0, 1000.0, 4.0, 4.0, u_j, u_i, -A)
        assert np.allclose(F_ij + F_ji, 0.0, atol=1e-15)      # momentum conservation
        assert F_ij @ n > 0.0 and F_ji @ n < 0.0               # pulls the pair together
        assert np.isclose(F_ij @ n, 0.5 * 1000.0 * 4.0 * 0.2 * np.linalg.norm(A))

    def test_stress_force_uniform_state_identical(self):
        HC, verts, bset = _strip()
        for v in HC.V:
            v.m = 1000.0 * v.dual_vol
            v.u = np.array([0.3, 0.1])           # rigid translation
        for v in verts:
            if v not in bset:
                F_c = stress_force(v, dim=2, mu=1e-3, HC=HC, pressure_model=EOS)
                F_r = stress_force(v, dim=2, mu=1e-3, HC=HC, pressure_model=EOS,
                                   pressure_flux="acoustic-riemann")
                assert np.allclose(F_c, F_r, atol=1e-14)
                assert np.allclose(F_c, 0.0, atol=1e-12)

    def test_stress_force_riemann_damps_checkerboard_velocity(self):
        HC, verts, bset = _strip()
        for k, v in enumerate(HC.V):
            v.m = 1000.0 * v.dual_vol
            i = round(v.x_a[0] / 0.1); j = round(v.x_a[1] / 0.1)
            v.u = np.array([0.0, 0.05 * (-1) ** (i + j)])   # checkerboard velocity
        # centred flux: no force at all (uniform pressure); riemann: force
        # opposes the local velocity (dissipative)
        power = 0.0
        for v in verts:
            if v in bset:
                continue
            assert np.allclose(stress_force(v, dim=2, mu=0.0, HC=HC, pressure_model=EOS), 0.0, atol=1e-12)
            F = stress_force(v, dim=2, mu=0.0, HC=HC, pressure_model=EOS,
                             pressure_flux="acoustic-riemann")
            power += float(F @ v.u)
        assert power < 0.0

    def test_needs_eos(self):
        HC, verts, bset = _strip()
        for v in HC.V:
            v.m = 1000.0 * v.dual_vol; v.u = np.zeros(2); v.p = 0.0
        with pytest.raises(ValueError):
            stress_force(verts[10], dim=2, mu=0.0, HC=HC, pressure_flux="acoustic-riemann")
        with pytest.raises(KeyError):
            stress_force(verts[10], dim=2, mu=0.0, HC=HC, pressure_flux="upwind")


class TestDensityDiffusion:
    def test_conservation_and_linear_field_exactness(self):
        HC, verts, bset = _strip()
        g = np.array([3.0, -2.0])                              # linear density field
        for v in HC.V:
            v.m = (1000.0 + g @ v.x_a[:2]) * v.dual_vol
        m0 = sum(v.m for v in HC.V)
        interior = [v for v in verts if v not in bset]
        m_int_before = np.array([v.m for v in interior])
        st = density_diffusion_step(HC, interior, 0.1, 4.0, 1e-3, dim=2)
        assert abs(sum(v.m for v in HC.V) - m0) < 1e-12 * m0
        # interior cells are closed: the corrected flux is zero on a linear field
        assert np.abs(np.array([v.m for v in interior]) - m_int_before).max() < 1e-12 * m0
        assert st["n_uncorrected"] == 0

    def test_damps_checkerboard(self):
        HC, verts, bset = _strip()
        for v in HC.V:
            i = round(v.x_a[0] / 0.1); j = round(v.x_a[1] / 0.1)
            v.m = 1000.0 * (1.0 + 0.05 * (-1) ** (i + j)) * v.dual_vol
        interior = [v for v in verts if v not in bset]

        def spread():
            return float(np.std([v.m / v.dual_vol for v in interior]))

        s0 = spread()
        m0 = sum(v.m for v in HC.V)
        # per step each cell exchanges ~ delta c0 |A| dt / V ~ 0.4 % of the
        # jump with each of its 6 neighbours: ~2.4 %/step, 60 steps -> 0.23
        for _ in range(60):
            density_diffusion_step(HC, interior, 0.1, 4.0, 1e-3, dim=2)
        assert spread() < 0.5 * s0
        assert abs(sum(v.m for v in HC.V) - m0) < 1e-12 * m0

    def test_uncorrected_reduces_to_plain_diffusion(self):
        HC, verts, bset = _strip()
        for v in HC.V:
            v.m = 1000.0 * v.dual_vol
        st = density_diffusion_step(HC, [v for v in verts if v not in bset], 0.1, 4.0,
                                    1e-3, dim=2, corrected=False)
        assert st["max_rel_dm"] == 0.0


class TestMethodAxes:
    def test_defaults_and_partial_identity(self):
        HC, verts, bset = _strip()
        m = SolverMethods(dim=2)
        assert m.pressure_flux == "centred" and m.density_diffusion is None
        ref = partial(dudt_i, dim=2, mu=0.1, HC=HC, pressure_model=None)
        got = m.dudt_fn(HC, mu=0.1)
        assert got.func is ref.func and got.keywords == ref.keywords

    def test_riemann_bound_only_when_requested(self):
        HC, verts, bset = _strip()
        m = SolverMethods(dim=2, pressure_flux="acoustic-riemann")
        fn = m.dudt_fn(HC, mu=0.1, pressure_model=EOS)
        assert fn.keywords["pressure_flux"] == "acoustic-riemann"
        with pytest.raises(ValueError):
            m.dudt_fn(HC, mu=0.1)                   # needs an EOS

    def test_density_diffusion_kwarg(self):
        m = SolverMethods(dim=2, density_diffusion=0.1)
        kw = m.integrator_kwargs(pressure_model=EOS)
        assert kw["density_diffusion"] == 0.1
        with pytest.raises(ValueError):
            m.integrator_kwargs()                   # needs an EOS
        assert "density_diffusion" not in SolverMethods(dim=2).integrator_kwargs()

    @pytest.mark.parametrize("kw", [
        dict(dim=2, pressure_flux="upwind"),
        dict(dim=2, density_diffusion=0.0),
        dict(dim=2, density_diffusion=-0.1),
        dict(dim=2, phases="multi", pressure_flux="acoustic-riemann"),
        dict(dim=2, phases="multi", density_diffusion=0.1),
        dict(dim=2, integrator="rk45", density_diffusion=0.1),
    ])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            SolverMethods(**kw)

    def test_integrator_runs_with_density_diffusion(self):
        from ddgclib.dynamic_integrators import symplectic_euler
        HC, verts, bset = _strip()
        for v in HC.V:
            i = round(v.x_a[0] / 0.1); j = round(v.x_a[1] / 0.1)
            v.m = 1000.0 * (1.0 + 0.02 * (-1) ** (i + j)) * v.dual_vol
            v.u = np.zeros(2); v.p = 0.0
        m = SolverMethods(dim=2, connectivity="dual_only", density_diffusion=0.1)
        fn = m.dudt_fn(HC, mu=1e-3, pressure_model=EOS)
        kw = m.integrator_kwargs(pressure_model=EOS)
        t = symplectic_euler(HC, bset, fn, dt=1e-3, n_steps=5, dim=2, **kw)
        assert t == pytest.approx(5e-3)
        assert all(np.isfinite(v.m) and v.m > 0 for v in HC.V)
