"""Free-surface tension and contact-angle force, the two BCs of the static
capillary rise and the Young-Laplace meniscus (laneI, 2026-10-06)."""
from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest
from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib._boundary_conditions import (
    AxialSlideBC, BoundaryConditionSet, HydrostaticReservoirBC,
)
from ddgclib.analytical import (
    hydrostatic_pressure_tait, jurin_height, young_laplace_meniscus,
)
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry import ensure_simplex_cache
from ddgclib.geometry.domains import cylinder_volume, rectangle
from ddgclib.initial_conditions import HydrostaticEOSMass
from ddgclib.methods import AXES, PRESETS, SolverMethods
from ddgclib.operators.free_surface import (
    FreeSurface, facet_area, facet_area_gradient,
)
from ddgclib.operators.stress import cache_dual_volumes


def _column(dim: int, jitter: float = 0.0, seed: int = 0):
    """Builder mesh with a free top: (HC, walls, free, contact)."""
    if dim == 2:
        HC = rectangle(L=1.0, h=2.0, refinement=2).HC
    else:
        HC = cylinder_volume(R=0.5, L=1.0, refinement=1).HC
    ax = dim - 1
    top = max(v.x_a[ax] for v in HC.V)
    hull = HC.boundary()
    free = [v for v in hull if abs(v.x_a[ax] - top) < 1e-12]
    walls = [v for v in hull if abs(v.x_a[ax] - top) >= 1e-12]
    if dim == 2:
        contact = [v for v in free if v.x_a[0] in (0.0, 1.0)]
    else:
        contact = [v for v in free
                   if abs(np.hypot(v.x_a[0], v.x_a[1]) - 0.5) < 1e-8]
    if jitter:
        rng = np.random.default_rng(seed)
        contact_ids = {id(v) for v in contact}
        for v in list(free):
            x = np.array(v.x_a[:dim], dtype=float)
            x[ax] += jitter * rng.standard_normal()
            if id(v) not in contact_ids:
                x[:ax] += jitter * rng.standard_normal(ax)
            HC.V.move(v, tuple(x))
    return HC, walls, free, contact


class TestFacetAreaGradient:
    @pytest.mark.parametrize('n', [2, 3])
    def test_matches_finite_differences(self, n):
        rng = np.random.default_rng(1)
        pts = rng.random((n, n))
        for i in range(n):
            g = facet_area_gradient(pts, i)
            fd = np.zeros(n)
            for k in range(n):
                d = np.zeros(n)
                d[k] = 1e-7
                pp, pm = pts.copy(), pts.copy()
                pp[i] += d
                pm[i] -= d
                fd[k] = (facet_area(pp) - facet_area(pm)) / 2e-7
            npt.assert_allclose(g, fd, atol=1e-8)

    def test_degenerate_facet_has_no_gradient(self):
        assert not facet_area_gradient(np.zeros((2, 2)), 0).any()
        assert not facet_area_gradient(np.zeros((3, 3)), 1).any()


class TestFreeSurfaceForce:
    @pytest.mark.parametrize('dim', [2, 3])
    def test_force_is_minus_the_energy_gradient(self, dim):
        HC, walls, free, contact = _column(dim, jitter=0.02)
        fs = FreeSurface(HC, dim, gamma=0.07, theta_deg=20.0, walls=walls,
                         free=free, contact=contact)
        assert len(fs.free_facets) == (4 if dim == 2 else 8)
        assert len(fs.wet_facets) == (2 if dim == 2 else 16)
        for v in free:
            F = fs.force(v)
            x0 = np.array(v.x_a[:dim], dtype=float)
            fd = np.zeros(dim)
            for k in range(dim):
                d = np.zeros(dim)
                d[k] = 1e-6
                HC.V.move(v, tuple(x0 + d))
                Ep = fs.energy()
                HC.V.move(v, tuple(x0 - d))
                Em = fs.energy()
                HC.V.move(v, tuple(x0))
                fd[k] = -(Ep - Em) / 2e-6
            npt.assert_allclose(F, fd, atol=1e-9)
        contact_ids = {id(v) for v in contact}
        for v in walls:
            # a wall vertex off the contact line carries no force (the one
            # below the line gets the reaction of its wetted facet)
            if not any(id(nb) in contact_ids for nb in v.nn):
                assert not fs.force(v).any()

    def test_2d_flat_surface_contact_force_is_young(self):
        HC, walls, free, contact = _column(2)
        gamma, theta = 0.07, 30.0
        fs = FreeSurface(HC, 2, gamma, theta, walls, free, contact)
        for v in free:
            if v in contact:
                continue
            npt.assert_allclose(fs.force(v), 0.0, atol=1e-15)
        left = next(v for v in contact if v.x_a[0] == 0.0)
        # tension along the flat surface (+x) and gamma cos(theta) up the wall
        npt.assert_allclose(fs.force(left),
                            [gamma, gamma * math.cos(math.radians(theta))],
                            rtol=1e-12)

    def test_3d_contact_force_is_young_per_contact_line_length(self):
        HC, walls, free, contact = _column(3)
        gamma, theta = 0.07, 30.0
        fs = FreeSurface(HC, 3, gamma, theta, walls, free, contact)
        for v in contact:
            F = fs.force(v)
            ring = [nb for nb in v.nn if any(nb is c for c in contact)]
            length = 0.5 * sum(np.linalg.norm(nb.x_a[:3] - v.x_a[:3])
                               for nb in ring)
            assert F[2] == pytest.approx(
                gamma * math.cos(math.radians(theta)) * length, rel=1e-12)

    def test_facets_follow_a_new_simplex_cache(self):
        HC, walls, free, contact = _column(2)
        fs = FreeSurface(HC, 2, 0.07, 20.0, walls, free, contact)
        key = fs.free_facets
        HC._simplices = list(HC._simplices)      # a rebuilt cache
        fs.force(free[0])
        assert fs.free_facets is not key
        assert len(fs.free_facets) == 4

    def test_needs_the_simplex_cache_and_consistent_sets(self):
        HC, walls, free, contact = _column(2)
        HC._simplices = None
        with pytest.raises(ValueError, match='_simplices'):
            FreeSurface(HC, 2, 0.07, 20.0, walls, free, contact)
        ensure_simplex_cache(HC, 2)
        with pytest.raises(ValueError, match='contact'):
            FreeSurface(HC, 2, 0.07, 20.0, walls, free, contact=walls[:1])


class TestAxialSlideBC:
    def test_keeps_the_axial_motion_and_restores_the_lateral(self):
        HC, walls, free, contact = _column(3)
        bc = AxialSlideBC(2, contact)
        anchors = {id(v): tuple(v.x_a[:2]) for v in contact}
        for v in contact:
            v.u = np.array([0.3, -0.2, 0.7])
            HC.V.move(v, (v.x_a[0] + 0.01, v.x_a[1] - 0.02, v.x_a[2] + 0.05))
        n = bc.apply(HC, dt=0.1, target_vertices=contact)
        assert n == len(contact) == 8
        for v in contact:
            npt.assert_array_equal(v.u, [0.0, 0.0, 0.7])
            assert tuple(v.x_a[:2]) == anchors[id(v)]
            assert v.x_a[2] == pytest.approx(1.05)
            assert v.x == tuple(v.x_a)              # cache key updated

    def test_target_narrows_the_anchored_set(self):
        HC, walls, free, contact = _column(2)
        bc = AxialSlideBC(1, contact)
        for v in contact:
            v.u = np.array([1.0, 1.0])
        assert bc.apply(HC, 0.1, target_vertices=contact[:1]) == 1
        assert contact[0].u[0] == 0.0 and contact[1].u[0] == 1.0


class TestHydrostaticReservoirBC:
    def test_band_vertices_get_the_profile_mass(self):
        HC = rectangle(L=1.0, h=2.0, refinement=2).HC
        hull = HC.boundary()
        for v in HC.V:
            v.boundary = v in hull
        compute_vd(HC, method='barycentric')
        cache_dual_volumes(HC, 2)
        eos = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1e5, n=1.0)
        ic = HydrostaticEOSMass(eos=eos, rho0=1000.0, g=9.81, gravity_axis=1,
                                h_ref=2.0, P_ref=0.0)
        for v in HC.V:
            v.m = 1.0
        bc = HydrostaticReservoirBC(ic, level=1.0)
        bc_set = BoundaryConditionSet().add(bc, HC.V)
        bc_set.apply_all(HC, set(), 0.1)
        n_band = 0
        for v in HC.V:
            if v.x_a[1] < 1.0:
                n_band += 1
                assert v.m == pytest.approx(
                    eos.density(ic._compressible_pressure(v.x_a[1])) * v.dual_vol)
            else:
                assert v.m == 1.0
        assert n_band == 18
        assert bc.injected == pytest.approx(
            sum(v.m for v in HC.V if v.x_a[1] < 1.0) - n_band)
        assert bc.axis == 1

    def test_assign_is_what_apply_does(self):
        eos = TaitMurnaghan(rho0=1000.0, P0=0.0, K=1e5, n=1.0)
        ic = HydrostaticEOSMass(eos=eos, rho0=1000.0, g=9.81, gravity_axis=0,
                                h_ref=1.0, P_ref=0.0)
        HC = Complex(1, domain=[(0.0, 1.0)])
        HC.triangulate()
        for v in HC.V:
            v.dual_vol = 0.5
        ic.apply(HC, set())
        m = {v.x: v.m for v in HC.V}
        for v in HC.V:
            v.m = 0.0
            ic.assign(v)
            assert v.m == m[v.x]


class TestContactLineAxis:
    def test_registered_and_single_phase_only(self):
        assert AXES['contact_line'].keys() == [None, 'energy_gradient']
        m = SolverMethods(dim=2, contact_line='energy_gradient')
        assert m.status_of('contact_line') == 'experimental'
        with pytest.raises(ValueError, match="single"):
            SolverMethods(dim=2, phases='multi', contact_line='energy_gradient')
        with pytest.raises(ValueError, match="not available in 1D"):
            SolverMethods(dim=1, contact_line='energy_gradient')

    def test_dudt_needs_and_records_the_surface(self):
        HC, walls, free, contact = _column(2)
        hull = HC.boundary()
        for v in HC.V:
            v.boundary = v in hull
        compute_vd(HC, method='barycentric')
        cache_dual_volumes(HC, 2)
        for v in HC.V:
            v.m = 1.0
            v.p = 0.0
            v.u = np.zeros(2)
        fs = FreeSurface(HC, 2, 0.07, 30.0, walls, free, contact)
        m = PRESETS['capillary_rise_static_2D']
        with pytest.raises(ValueError, match='free_surface'):
            m.dudt_fn(HC, mu=0.0)
        with pytest.raises(ValueError, match='contact_line'):
            SolverMethods(dim=2).dudt_fn(HC, mu=0.0, free_surface=fs)
        fn = m.dudt_fn(HC, mu=0.0, free_surface=fs)
        assert fn.free_surface is fs
        left = next(v for v in contact if v.x_a[0] == 0.0)
        npt.assert_allclose(fn(left), fs.force(left) / left.m)
        g = m.dudt_fn(HC, mu=0.0, free_surface=fs, body_force=[0.0, -9.81])
        npt.assert_allclose(g(left), fs.force(left) / left.m + [0.0, -9.81])
        assert g.free_surface is fs and g.body_force[1] == -9.81

    def test_presets_run_the_axis(self):
        for name in ('capillary_rise_static_2D', 'capillary_rise_static_3D'):
            assert PRESETS[name].contact_line == 'energy_gradient'
            assert PRESETS[name].phases == 'single'


class TestYoungLaplaceMeniscus:
    rho, g, gamma, theta = 997.0, 9.81, 0.0728, 9.99

    @pytest.mark.parametrize('dim', [2, 3])
    def test_mean_height_is_jurin_for_the_incompressible_profile(self, dim):
        r = 2e-3
        m = young_laplace_meniscus(r, self.gamma, self.theta,
                                   lambda y: -self.rho * self.g * y, dim)
        assert abs(m.residual) < 1e-10
        assert m.mean == pytest.approx(
            jurin_height(r, self.gamma, self.theta, self.rho, self.g, dim),
            rel=1e-5)
        assert m.apex < m.mean < m.contact
        assert m.height(r if dim == 2 else 0.0) == pytest.approx(m.apex)
        assert m.height(0.0 if dim == 2 else r) == pytest.approx(m.contact)

    @pytest.mark.parametrize('dim', [2, 3])
    def test_small_tube_is_a_spherical_cap(self, dim):
        r = 1e-4
        m = young_laplace_meniscus(r, self.gamma, self.theta,
                                   lambda y: -self.rho * self.g * y, dim)
        R = r / math.cos(math.radians(self.theta))
        assert m.contact - m.apex == pytest.approx(
            R * (1.0 - math.sin(math.radians(self.theta))), rel=1e-3)

    def test_compressible_column_stands_higher(self):
        r = 2e-3
        K = self.rho * (10.0 * math.sqrt(self.g * jurin_height(
            r, self.gamma, self.theta, self.rho, self.g, 2)))**2
        P = hydrostatic_pressure_tait(self.rho, self.g, K)
        assert P(0.0) == 0.0
        m = young_laplace_meniscus(r, self.gamma, self.theta, P, 2)
        h_J = jurin_height(r, self.gamma, self.theta, self.rho, self.g, 2)
        assert m.mean / h_J == pytest.approx(
            1.0 + self.rho * self.g * h_J / (2.0 * K), abs=1e-4)
