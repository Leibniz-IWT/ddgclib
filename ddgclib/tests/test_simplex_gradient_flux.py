"""Tests for the simplex-gradient fluxes (laneH, 2026-10-01): method axes
``viscous_flux`` and ``pressure_flux``, value ``'simplex_gradient'``.

``viscous_force_simplex_gradient`` is the flux through the barycentric
dual faces with the gradient of the piecewise-linear velocity on each
primal simplex; ``pressure_force_simplex_gradient`` is minus the integral
of the piecewise-linear pressure gradient over the dual cell.  Both are
linearly precise on any simplicial mesh, which the two-point viscous flux
(and, in 3D, the centred pressure flux on the cached ``batch_e_star``
areas) is not.
"""
from functools import partial

import numpy as np
import numpy.testing as npt
import pytest

from hyperct.ddg import invalidate_simplex_cache, simplex_dual_volumes

from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
from ddgclib.geometry.domains import box, cylinder_volume, rectangle
from ddgclib.methods import AXES, SolverMethods
from ddgclib.operators.stress import (
    dudt_i, pressure_flux_methods, pressure_force_simplex_gradient,
    stress_force, viscous_flux, viscous_flux_methods,
    viscous_force_simplex_gradient,
)


def _jittered(dim: int, seed: int = 1, amplitude: float = 0.03):
    """A builder mesh whose interior vertices were moved at random and
    that was reconnected by the library Delaunay retopology."""
    rng = np.random.default_rng(seed)
    res = (rectangle(L=2.0, h=1.0, refinement=3) if dim == 2
           else box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2))
    HC, bV = res.HC, res.bV
    HC.V.move_all([(v, tuple(v.x_a + amplitude * rng.uniform(-1, 1, dim)))
                   for v in list(HC.V) if v not in bV])
    _retopologize(HC, bV, dim)
    for v in HC.V:
        v.u = np.zeros(dim)
        v.p = 0.0
        v.m = 1.0
    return HC, bV, rng


def _interior(HC):
    return [v for v in HC.V if not v.boundary]


class TestRegistry:
    def test_keys(self):
        assert viscous_flux_methods["two_point"] is viscous_flux
        assert (viscous_flux_methods["simplex_gradient"]
                is viscous_force_simplex_gradient)
        assert (pressure_flux_methods["simplex_gradient"]
                is pressure_force_simplex_gradient)
        assert set(viscous_flux_methods.available()) == {
            "two_point", "simplex_gradient"}

    def test_axes(self):
        ax = AXES['viscous_flux']
        assert ax.default == 'two_point' and ax.applies_to == 'single'
        assert {o.key for o in ax.options} == set(
            viscous_flux_methods.available())
        assert {o.key for o in AXES['pressure_flux'].options} == set(
            pressure_flux_methods.available())

    def test_unknown_key_raises(self):
        HC, _, _ = _jittered(2)
        v = _interior(HC)[0]
        with pytest.raises(KeyError):
            stress_force(v, dim=2, mu=1.0, HC=HC, viscous_flux="cotan")


@pytest.mark.parametrize('dim', [2, 3])
class TestViscousSimplexGradient:
    def test_linear_precision(self, dim):
        """Zero for a linear velocity field at every interior vertex of a
        jittered Delaunay mesh; the two-point flux is not."""
        HC, _, rng = _jittered(dim)
        A = rng.normal(size=(dim, dim))
        b = rng.normal(size=dim)
        for v in HC.V:
            v.u = A @ v.x_a[:dim] + b
        interior = _interior(HC)
        assert len(interior) > 50
        simplex = max(np.abs(viscous_force_simplex_gradient(
            v, HC, dim, 1.0)).max() for v in interior)
        two_point = max(np.abs(stress_force(
            v, dim=dim, mu=1.0, HC=HC)).max() for v in interior)
        assert simplex < 1e-13
        assert two_point > 1e-2

    def test_momentum_and_dissipation(self, dim):
        """Pairwise antisymmetric (the forces sum to zero over the whole
        mesh) and negative semi-definite (u . F <= 0)."""
        HC, _, rng = _jittered(dim)
        for v in HC.V:
            v.u = rng.normal(size=dim)
        F = {v: viscous_force_simplex_gradient(v, HC, dim, 0.7)
             for v in HC.V}
        npt.assert_allclose(sum(F.values()), 0.0, atol=1e-12)
        assert sum(float(v.u @ f) for v, f in F.items()) < 0.0

    def test_stress_force_dispatch(self, dim):
        """``viscous_flux='simplex_gradient'`` swaps the viscous part and
        leaves the pressure part; the default is the two-point flux."""
        HC, _, rng = _jittered(dim)
        for v in HC.V:
            v.u = rng.normal(size=dim)
            v.p = float(rng.normal())
        v = _interior(HC)[3]
        pressure = stress_force(v, dim=dim, mu=0.0, HC=HC)
        mixed = stress_force(v, dim=dim, mu=0.3, HC=HC,
                             viscous_flux='simplex_gradient')
        npt.assert_allclose(
            mixed, pressure + viscous_force_simplex_gradient(v, HC, dim, 0.3),
            rtol=0, atol=1e-13)
        npt.assert_array_equal(
            stress_force(v, dim=dim, mu=0.3, HC=HC),
            stress_force(v, dim=dim, mu=0.3, HC=HC, viscous_flux='two_point'))


def test_2d_equals_the_cotangent_weights():
    """Per edge the 2D weights are 1/2 (cot alpha + cot beta)."""
    HC, _, rng = _jittered(2)
    for v in HC.V:
        v.u = rng.normal(size=2)
    for v in _interior(HC)[:20]:
        F = np.zeros(2)
        for s in HC._simplices:
            if v not in s:
                continue
            a, b = (w for w in s if w is not v)
            for near, far in ((a, b), (b, a)):
                # angle at `far`, opposite the edge (v, near)
                e1 = v.x_a - far.x_a
                e2 = near.x_a - far.x_a
                cot = (e1 @ e2) / abs(e1[0] * e2[1] - e1[1] * e2[0])
                F += 0.5 * cot * (near.u - v.u)
        npt.assert_allclose(viscous_force_simplex_gradient(v, HC, 2, 1.0), F,
                            rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('dim', [2, 3])
class TestPressureSimplexGradient:
    def test_linear_pressure_is_exact_on_every_vertex(self, dim):
        """``F = -Vol grad(p)`` with the simplex-exact dual volume, hull
        vertices included."""
        HC, _, rng = _jittered(dim)
        g = rng.normal(size=dim)
        for v in HC.V:
            v.p = 3.0 + g @ v.x_a[:dim]
        vols = simplex_dual_volumes(HC, dim)
        err = max(np.abs(pressure_force_simplex_gradient(v, HC, dim)
                         + vols[v] * g).max() for v in HC.V)
        assert err < 1e-14

    def test_uniform_pressure_gives_no_force_on_the_hull(self, dim):
        """The volume form has no ambient pressure: an open cell feels
        nothing at uniform pressure, the centred flux pushes it out."""
        HC, bV, _ = _jittered(dim)
        for v in HC.V:
            v.p = 5.0
        hull = next(v for v in HC.V if v.boundary)
        npt.assert_allclose(
            stress_force(hull, dim=dim, mu=0.0, HC=HC,
                         pressure_flux='simplex_gradient'), 0.0, atol=1e-14)
        assert np.abs(stress_force(hull, dim=dim, mu=0.0, HC=HC)).max() > 1e-3

    def test_callable_pressure_model(self, dim):
        HC, _, rng = _jittered(dim)
        g = rng.normal(size=dim)
        v = _interior(HC)[0]
        F = stress_force(v, dim=dim, mu=0.0, HC=HC,
                         pressure_model=lambda w: g @ w.x_a[:dim],
                         pressure_flux='simplex_gradient')
        npt.assert_allclose(F, -simplex_dual_volumes(HC, dim)[v] * g,
                            rtol=0, atol=1e-14)


def test_3d_centred_flux_on_the_cached_areas_is_not_linearly_precise():
    """Why the 3D pipe uses the simplex form: on a jittered Delaunay mesh
    the centred flux reads the ``batch_e_star`` area cache."""
    HC, _, rng = _jittered(3)
    g = np.array([0.0, 0.0, 1.0])
    for v in HC.V:
        v.p = g @ v.x_a
    vols = simplex_dual_volumes(HC, 3)
    rel = [np.linalg.norm(stress_force(v, dim=3, mu=0.0, HC=HC) + vols[v] * g)
           / vols[v] for v in _interior(HC)]
    assert max(rel) > 0.02


class TestFlatSimplices:
    def test_builder_cylinder_has_flat_tetrahedra_and_finite_forces(self):
        """qhull returns coplanar tetrahedra on the structured cylinder;
        they have no gradient and are left out of the viscous force."""
        res = cylinder_volume(R=0.5, L=3.0, refinement=1, flow_axis=2)
        HC, bV = res.HC, res.bV
        _retopologize(HC, bV, 3)
        pts = np.array([[w.x_a for w in s] for s in HC._simplices])
        vol = np.abs(np.linalg.det(pts[:, 1:] - pts[:, :1])) / 6.0
        assert (vol < 1e-14).sum() > 0
        for v in HC.V:
            r2 = v.x_a[0]**2 + v.x_a[1]**2
            v.u = np.array([0.0, 0.0, max(0.25 - r2, 0.0)])
            v.p = -v.x_a[2]
            v.m = 1.0
        for v in HC.V:
            F = stress_force(v, dim=3, mu=1.0, HC=HC,
                             pressure_flux='simplex_gradient',
                             viscous_flux='simplex_gradient')
            assert np.all(np.isfinite(F))

    def test_a_flat_triangle_is_skipped(self):
        """Three almost collinear vertices: the triangle is left out, a
        well shaped one with a short edge is not."""
        from hyperct import Complex
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.0)])
        a, b = HC.V[(0.0, 0.0)], HC.V[(1.0, 0.0)]
        c = HC.V[(0.5, 1e-6)]          # cap: height 1e-6 over a unit base
        d = HC.V[(0.0, 1e-3)]          # needle: short edge a-d
        for v in (a, b, c, d):
            v.u = np.zeros(2)
        c.u = np.array([1.0, 0.0])
        d.u = np.array([1.0, 0.0])
        HC._simplices = [(a, b, c)]
        npt.assert_array_equal(viscous_force_simplex_gradient(c, HC, 2, 1.0),
                               np.zeros(2))
        HC._simplices = [(a, d, b)]
        assert abs(viscous_force_simplex_gradient(d, HC, 2, 1.0)[0]) > 100.0


class TestSimplexCache:
    def test_needs_the_simplex_cache(self):
        HC, _, _ = _jittered(2)
        v = _interior(HC)[0]
        invalidate_simplex_cache(HC)
        with pytest.raises(ValueError, match='HC._simplices'):
            stress_force(v, dim=2, mu=1.0, HC=HC,
                         viscous_flux='simplex_gradient')

    def test_incidence_follows_a_retopology(self):
        """The vertex-to-simplex map is rebuilt when the retopology
        replaces ``HC._simplices``: linear precision holds on the new
        connectivity."""
        HC, bV, rng = _jittered(2)
        A = rng.normal(size=(2, 2))
        v0 = _interior(HC)[0]
        viscous_force_simplex_gradient(v0, HC, 2, 1.0)     # builds the map
        HC.V.move_all([(v, tuple(v.x_a + 0.05 * rng.uniform(-1, 1, 2)))
                       for v in list(HC.V) if v not in bV])
        _retopologize(HC, bV, 2)
        for v in HC.V:
            v.u = A @ v.x_a
        assert max(np.abs(viscous_force_simplex_gradient(
            v, HC, 2, 1.0)).max() for v in _interior(HC)) < 1e-13


class TestSolverMethods:
    def test_default_binds_nothing(self):
        HC, _, _ = _jittered(2)
        fn = SolverMethods(dim=2).dudt_fn(HC, mu=0.1)
        assert isinstance(fn, partial) and fn.func is dudt_i
        assert 'viscous_flux' not in fn.keywords
        assert 'pressure_flux' not in fn.keywords

    def test_simplex_gradient_is_bound_and_needs_no_eos(self):
        HC, _, _ = _jittered(2)
        m = SolverMethods(dim=2, viscous_flux='simplex_gradient',
                          pressure_flux='simplex_gradient')
        fn = m.dudt_fn(HC, mu=0.1)
        assert fn.keywords['viscous_flux'] == 'simplex_gradient'
        assert fn.keywords['pressure_flux'] == 'simplex_gradient'
        v = _interior(HC)[0]
        npt.assert_array_equal(fn(v), np.zeros(2))     # u = 0, p = 0
        assert SolverMethods.from_dict(m.to_dict()) == m

    @pytest.mark.parametrize('kw', [
        dict(dim=2, viscous_flux='cotan'),
        dict(dim=1, viscous_flux='simplex_gradient'),
        dict(dim=1, pressure_flux='simplex_gradient'),
        dict(dim=2, phases='multi', viscous_flux='simplex_gradient'),
        dict(dim=2, connectivity='periodic', periodic_axes=(0,),
             viscous_flux='simplex_gradient'),
        dict(dim=2, connectivity='periodic', periodic_axes=(0,),
             pressure_flux='simplex_gradient'),
    ])
    def test_combinations_that_would_not_work_raise(self, kw):
        with pytest.raises(ValueError):
            SolverMethods(**kw)

    def test_presets(self):
        from ddgclib.methods import PRESETS
        m2, m3 = PRESETS['hagen_poiseuille_2D'], PRESETS['hagen_poiseuille_3D']
        assert (m2.viscous_flux, m2.pressure_flux) == (
            'simplex_gradient', 'centred')
        assert (m3.viscous_flux, m3.pressure_flux) == (
            'simplex_gradient', 'simplex_gradient')
        for m in (m2, m3):
            assert (m.connectivity, m.frozen_set) == ('delaunay', 'membership')


# ---------------------------------------------------------------------------
# Finding of laneH outside its brief: orientation of the 2D dual area vector
# (a strict xfail until laneO, 2026-10-05, fixed it: axis area_orientation)
# ---------------------------------------------------------------------------

def test_2d_dual_area_vectors_close_on_a_sheared_jittered_mesh():
    """The dual cell of an interior vertex is closed: sum_j A_ij = 0, and
    every A_ij points along its edge.  Under the legacy rule
    ``area_orientation='dual_midpoint'`` 6 of 665 area vectors of this
    mesh point against their edge and the closure residual is 0.23
    (test_area_orientation.py keeps that count)."""
    from ddgclib.operators.stress import dual_area_vector
    rng = np.random.default_rng(0)
    res = rectangle(L=2.0, h=1.0, refinement=3, flow_axis=0)
    HC, bV = res.HC, res.bV
    h = (2.0 / sum(1 for _ in HC.V)) ** 0.5
    HC.V.move_all([
        (v, (v.x_a[0] + 0.3 * 4 * v.x_a[1] * (1 - v.x_a[1])
             + 0.2 * h * rng.uniform(-1, 1),
             v.x_a[1] + 0.2 * h * rng.uniform(-1, 1)))
        for v in list(HC.V) if v not in bV])
    _retopologize(HC, bV, 2)
    closure, against = 0.0, 0
    for v in HC.V:
        if v.boundary:
            continue
        total = np.zeros(2)
        for nb in v.nn:
            A = dual_area_vector(v, nb, HC, 2)
            total += A
            against += float(A @ (nb.x_a - v.x_a)) < 0.0
        closure = max(closure, float(np.linalg.norm(total)))
    assert against == 0
    assert closure < 1e-12
