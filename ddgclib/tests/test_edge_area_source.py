"""The explicit 3D method axis ``edge_area_source`` (lane Q, 2026-10-05).

Which construction gives the oriented dual face area vectors ``A_ij`` the
force operators read in 3D:

- ``None`` / ``'e_star_cache'``: the legacy ``batch_e_star`` fan cache of
  the interior vertices (not linearly precise, laneJ), ring walk on hull
  vertices;
- ``'p_ij_simplex'``: the exact face of every edge of every vertex from
  ``hyperct.ddg.simplex_dual_face_areas`` (cached at the retopology);
- ``'p_ij'``: the same face built per edge from the tetrahedra around it;
- ``'p_ij_ring'``: the legacy per-edge ring walk with its face heuristic
  (status broken; laneJ's ``p_ij`` arm).

Checked: the retopology functions fill the cache and tag the mesh for
every value, ``None`` is bit-identical to ``'e_star_cache'``, the exact
sources close every interior cell on a mesh with flat tetrahedra and give
the linear-pressure force to round-off where the fan cache is 7 % off,
``effective_methods`` reports what ran, ``SolverMethods`` rejects the
combinations that would be silent no-ops, the cache-less retopologies
honour the axis, and short droplet runs through the preset agree between
the two exact sources.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ddgclib.dynamic_integrators._integrators_dynamic import (  # noqa: E402
    _do_retopologize, _retopologize,
)
from ddgclib.geometry.domains import box  # noqa: E402
from ddgclib.methods import (  # noqa: E402
    AXES, PRESETS, SolverMethods, effective_methods, record_methods,
)
from ddgclib.methods._retopo import (  # noqa: E402
    bare_dual_refresh, retopologize_material_delaunay,
)
from ddgclib.operators.stress import (  # noqa: E402
    dual_area_vector, stress_force,
)

VALUES = ('e_star_cache', 'p_ij_simplex', 'p_ij', 'p_ij_ring')
G = np.array([1.0, 2.0, 3.0])


def _box(refinement=2):
    r = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=refinement)
    HC, bV = r.HC, set(r.bV)
    for v in HC.V:
        v.u = np.zeros(3)
        v.p = 5.0 + float(G @ v.x_a[:3])
        v.m = 1.0
    return HC, bV


def _cache_by_coords(HC):
    c = HC._edge_area_cache
    if c is None:
        return None
    by_id = {id(v): v.x for v in HC.V}
    return {(by_id[i], by_id[j]): tuple(a) for i, row in c.items()
            for j, a in row.items()}


def _flat_count(HC):
    P = np.array([[w.x_a[:3] for w in s] for s in HC._simplices])
    return int((np.linalg.det(P[:, 1:] - P[:, :1]) == 0).sum())


def _force_error(HC, bV):
    """Largest relative error of the centred force of the linear pressure
    5 + G . x against -G Vol_i over the interior vertices."""
    return max(np.linalg.norm(stress_force(v, dim=3, mu=0.0, HC=HC)
                              + G * v.dual_vol) / (np.linalg.norm(G) * v.dual_vol)
               for v in HC.V if v not in bV)


def _closure(HC, bV, area):
    return max(np.linalg.norm(sum(area(v, nb) for nb in v.nn))
               / sum(np.linalg.norm(area(v, nb)) for nb in v.nn)
               for v in HC.V if v not in bV)


class TestRetopologyFillsTheSource:
    def test_none_is_the_fan_cache_bit_identical(self):
        HC0, bV0 = _box()
        _retopologize(HC0, bV0, 3)
        HC1, bV1 = _box()
        _retopologize(HC1, bV1, 3, edge_area_source='e_star_cache')
        assert HC0._edge_area_source == HC1._edge_area_source == 'e_star_cache'
        assert _cache_by_coords(HC0) == _cache_by_coords(HC1)
        assert effective_methods(HC0, 3)['edge_area_source'] == 'e_star_cache'
        # interior vertices only (hull vertices read the ring walk)
        assert set(HC0._edge_area_cache) == {id(v) for v in HC0.V if v not in bV0}
        assert {v.x: v.dual_vol for v in HC0.V} == {v.x: v.dual_vol for v in HC1.V}

    def test_fan_cache_is_not_linearly_precise_but_the_exact_sources_are(self):
        errors = {}
        vols = {}
        for src in (None,) + VALUES:
            HC, bV = _box()
            _retopologize(HC, bV, 3, edge_area_source=src)
            errors[src] = _force_error(HC, bV)
            vols[src] = {v.x: v.dual_vol for v in HC.V}
        assert errors[None] == errors['e_star_cache'] > 1e-2      # 7.0e-02
        assert errors['p_ij_simplex'] < 1e-13                      # 3.1e-15
        assert errors['p_ij'] < 1e-13
        assert errors['p_ij_ring'] < 1e-13          # exact on the box (laneJ)
        # the dual volumes do not depend on the source
        assert all(vols[s] == vols[None] for s in VALUES)

    def test_p_ij_simplex_caches_every_edge_of_every_vertex(self):
        HC, bV = _box()
        _retopologize(HC, bV, 3, edge_area_source='p_ij_simplex')
        assert _flat_count(HC) >= 10                 # 13 flat tets after Delaunay
        cache = HC._edge_area_cache
        assert set(cache) == {id(v) for v in HC.V}
        assert all(set(cache[id(v)]) == {id(nb) for nb in v.nn} for v in HC.V)
        eff = effective_methods(HC, 3)
        assert eff['edge_area_source'] == 'p_ij_simplex'
        assert eff['edge_area_cache_present'] is True
        assert eff['dual_volume'] == 'simplex_exact'
        # closure of every interior cell, flat tetrahedra included
        assert _closure(HC, bV, lambda v, nb: cache[id(v)][id(nb)]) < 1e-14
        # antisymmetric to the bit
        for i, row in cache.items():
            for j, a in row.items():
                assert np.array_equal(cache[j][i], -a)
        # equal to the per-edge construction
        worst = 0.0
        for v in HC.V:
            for nb in v.nn:
                a = cache[id(v)][id(nb)]
                b = dual_area_vector(v, nb, HC, 3, source='p_ij')
                worst = max(worst, np.linalg.norm(a - b)
                            / max(np.linalg.norm(a), 1e-300))
        assert worst < 1e-13

    @pytest.mark.parametrize('src', ['p_ij', 'p_ij_ring'])
    def test_uncached_sources_tag_the_mesh(self, src):
        HC, bV = _box()
        _retopologize(HC, bV, 3, edge_area_source=src)
        assert HC._edge_area_cache is None
        assert HC._edge_area_source == src
        eff = effective_methods(HC, 3)
        assert eff['edge_area_source'] == src
        assert eff['edge_area_cache_present'] is False
        v = next(v for v in HC.V if v not in bV)
        nb = next(iter(v.nn))
        # dual_area_vector follows the tag
        npt.assert_array_equal(dual_area_vector(v, nb, HC, 3),
                               dual_area_vector(v, nb, HC, 3, source=src))

    def test_the_two_per_edge_constructions_differ(self):
        """The ring walk picks its face by a nearest-barycentre heuristic;
        the simplex construction reads the faces.  On the box they agree to
        round-off (laneJ: the heuristic fails on the droplet mesh, where
        test_determinism and the slow droplet tests look)."""
        HC, bV = _box()
        _retopologize(HC, bV, 3, edge_area_source='p_ij')
        v = next(v for v in HC.V if v not in bV)
        for nb in v.nn:
            npt.assert_allclose(dual_area_vector(v, nb, HC, 3, source='p_ij'),
                                dual_area_vector(v, nb, HC, 3, source='p_ij_ring'),
                                rtol=1e-12, atol=1e-18)

    def test_invalid_values_and_dimensions_raise(self):
        HC, bV = _box()
        with pytest.raises(ValueError, match='edge_area_source must be'):
            _retopologize(HC, bV, 3, edge_area_source='p_ij_ring_3d')
        from ddgclib.geometry.domains import rectangle
        r = rectangle(L=1.0, h=1.0, refinement=2)
        for v in r.HC.V:
            v.u = np.zeros(2)
            v.p = 0.0
            v.m = 1.0
        with pytest.raises(ValueError, match='3D axis'):
            _retopologize(r.HC, set(r.bV), 2, edge_area_source='p_ij')
        HC2, bV2 = _box()
        HC2._simplices = None
        v = next(v for v in HC2.V if v not in bV2)
        with pytest.raises(ValueError, match='_simplices'):
            dual_area_vector(v, next(iter(v.nn)), HC2, 3, source='p_ij')


class TestCacheLessRetopologies:
    @pytest.mark.parametrize('fn', [bare_dual_refresh,
                                    retopologize_material_delaunay])
    def test_honour_the_axis(self, fn):
        for src, cached, reported in ((None, False, 'p_ij_ring'),
                                      ('p_ij_ring', False, 'p_ij_ring'),
                                      ('p_ij', False, 'p_ij'),
                                      ('p_ij_simplex', True, 'p_ij_simplex')):
            HC, bV = _box()
            fn(HC, bV, 3, edge_area_source=src)
            assert (HC._edge_area_cache is not None) is cached
            assert effective_methods(HC, 3)['edge_area_source'] == reported
            assert effective_methods(HC, 3)['boundary_dual_vol'] == 'half_cell'
            if src == 'p_ij_simplex':
                assert set(HC._edge_area_cache) == {id(v) for v in HC.V}
        HC, bV = _box()
        with pytest.raises(ValueError, match='builds no batch_e_star cache'):
            fn(HC, bV, 3, edge_area_source='e_star_cache')


class TestSolverMethodsAxis:
    def test_registry(self):
        ax = AXES['edge_area_source']
        assert ax.explicit and ax.default is None
        assert set(VALUES) <= set(ax.keys())
        assert ax.option('p_ij_ring').status == 'broken'
        assert SolverMethods.__dataclass_fields__['edge_area_source'].default is None

    @pytest.mark.parametrize('kw', [
        dict(dim=2, edge_area_source='p_ij'),
        dict(dim=1, edge_area_source='e_star_cache'),
        dict(dim=3, edge_area_source='shared_vd_2d'),
        dict(dim=3, edge_area_source='p_ij_ring_3d'),
        dict(dim=3, connectivity='frozen', edge_area_source='p_ij_simplex'),
        dict(dim=3, connectivity='custom', edge_area_source='p_ij'),
        dict(dim=3, phases='multi', connectivity='periodic', periodic_axes=(0,),
             edge_area_source='p_ij'),
        dict(dim=3, connectivity='dual_only_bare', edge_area_source='e_star_cache'),
        dict(dim=3, connectivity='delaunay_material', edge_area_source='e_star_cache'),
        dict(dim=3, edge_area_source='p_ij_simplex', backend='gpu'),
        dict(dim=3, edge_area_source='p_ij', backend='multiprocessing'),
    ])
    def test_invalid_combinations_raise(self, kw):
        with pytest.raises(ValueError):
            SolverMethods(**kw)

    def test_valid_combinations(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for conn in ('delaunay', 'dual_only'):
                for src in ('e_star_cache', 'p_ij', 'p_ij_simplex'):
                    SolverMethods(dim=3, connectivity=conn, edge_area_source=src)
            SolverMethods(dim=3, connectivity='dual_only_bare',
                          edge_area_source='p_ij_simplex')
            SolverMethods(dim=3, connectivity='delaunay_material',
                          edge_area_source='p_ij')
            SolverMethods(dim=3, edge_area_source='e_star_cache', backend='gpu')
            SolverMethods(dim=2)        # None is fine in every dimension
        with pytest.warns(UserWarning, match="'broken'"):
            SolverMethods(dim=3, edge_area_source='p_ij_ring')

    def test_integrator_kwargs_carry_the_field(self):
        m = PRESETS['oscillating_droplet_3D']
        assert m.edge_area_source is None
        assert m.integrator_kwargs(mps=object())['edge_area_source'] is None
        m2 = m.replace(edge_area_source='p_ij_simplex')
        assert m2.integrator_kwargs(mps=object())['edge_area_source'] == 'p_ij_simplex'
        assert m2.status_of('edge_area_source') == 'validated'
        assert m.replace(edge_area_source='p_ij').status_of('edge_area_source') == 'opt-in'
        assert "edge_area_source   = 'p_ij_simplex'" in m2.describe()
        assert SolverMethods.from_dict(m2.to_dict()) == m2

    def test_forwarded_to_the_retopology_functions(self, tmp_path):
        """Single phase (library default function) and the multiphase
        partial both receive the integrator kwarg by name."""
        HC, bV = _box()
        _do_retopologize(HC, bV, 3, edge_area_source='p_ij_simplex')
        assert HC._edge_area_source == 'p_ij_simplex'
        m = SolverMethods(dim=3, connectivity='dual_only_bare',
                          edge_area_source='p_ij')
        HC, bV = _box()
        _do_retopologize(HC, bV, 3, retopologize_fn=m.retopologize_fn(),
                         edge_area_source=m.edge_area_source)
        assert HC._edge_area_source == 'p_ij' and HC._edge_area_cache is None
        doc = record_methods(tmp_path / 'methods.json', m, HC)
        assert doc['config']['edge_area_source'] == 'p_ij'
        assert doc['effective']['edge_area_source'] == 'p_ij'


class TestDropletRuns:
    """3D droplet, refinement 1/1, a few steps through the preset."""

    @staticmethod
    def _run(methods, n_steps=5):
        from cases_dynamic.oscillating_droplet.src._setup import (
            setup_oscillating_droplet,
        )
        HC, bV, mps, bc_set, dudt_fn, _r, params = setup_oscillating_droplet(
            dim=3, refinement_outer=1, refinement_droplet=1,
            split_method=methods.split_method,
            redistribute_mass=methods.redistribute_mass)
        m0 = sum(v.m for v in HC.V)
        x0 = {id(v): v.x for v in HC.V}
        methods.integrate(HC, bV, dudt_fn, dt=1e-5, n_steps=n_steps,
                          bc_set=bc_set, mps=mps)
        # keyed by the initial position: the exact sources move the
        # vertices by round-off, so the final coordinates are no key
        state = {x0[id(v)]: tuple(v.x_a) + tuple(v.u) for v in HC.V}
        return HC, bV, m0, sum(v.m for v in HC.V), state

    def test_every_value_runs_and_conserves_mass(self):
        base = PRESETS['oscillating_droplet_3D']
        states = {}
        for src in (None,) + VALUES:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                HC, bV, m0, m1, state = self._run(base.replace(edge_area_source=src))
            assert abs(m1 - m0) / m0 < 1e-12
            assert effective_methods(HC, 3, base)['edge_area_source'] == (
                src or 'e_star_cache')
            states[src] = state
        # the legacy value is the default to the bit
        assert states[None] == states['e_star_cache']
        # the two exact sources are the same face up to round-off
        keys = sorted(states['p_ij'])
        a = np.array([states['p_ij'][k] for k in keys])
        b = np.array([states['p_ij_simplex'][k] for k in keys])
        npt.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.abs(a).max())
        # and they differ from the fan cache
        c = np.array([states['e_star_cache'][k] for k in keys])
        assert np.abs(a - c).max() > 1e-6 * np.abs(a).max()

    def test_delaunay_preset_runs_with_the_exact_cache(self):
        m = PRESETS['oscillating_droplet_3D_delaunay'].replace(
            edge_area_source='p_ij_simplex')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            HC, bV, m0, m1, _ = self._run(m, n_steps=3)
        assert abs(m1 - m0) / m0 < 1e-12
        assert HC._edge_area_source == 'p_ij_simplex'
        assert all(v.dual_vol == 0.0 for v in bV)        # zeroed convention kept
