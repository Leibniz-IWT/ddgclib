"""Delaunay rebuild that keeps the fluid domain (lane P).

``_retopologize`` triangulates the convex hull of the vertex cloud.  When
a free surface has moved, that fills the gap between the surface and the
hull with simplices that are not fluid; on the hydrostatic column the
conservative remap then pins the total volume to the hull and the run is
unstable.  ``retopologize_material_delaunay``
(``SolverMethods(connectivity='delaunay_material')``) treats the boundary
of the previous connectivity as material and peels the simplices Delaunay
adds outside it.  The test is geometric (winding number of the simplex
centroid about the old boundary): the first version was topological and
removed fluid in 3D, where the wall facets change between rebuilds (lane P
review, classes ``TestPeel3D``).

Evidence: docs_temp/debug_session/laneP-hydrostatic-library-integrators.md
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from hyperct import Complex
from hyperct.ddg import (boundary_from_simplices, compute_vd,
                         connect_and_cache_simplices, simplex_dual_volumes)

from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry.domains import box, rectangle
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity
from ddgclib.methods import SolverMethods
from ddgclib.methods._retopo import (_facet_points, _measure, _points,
                                     enclosed_volume, oriented_boundary,
                                     peel_outside_boundary,
                                     retopologize_material_delaunay,
                                     winding_number)
from ddgclib.operators.mass_redistribution import snapshot_pressure_fresh
from ddgclib.operators.stress import cache_dual_volumes

RHO0, G = 1000.0, 9.81
K = RHO0 * (10.0 * np.sqrt(G))**2


def _eos() -> TaitMurnaghan:
    return TaitMurnaghan(rho0=RHO0, P0=0.0, K=K, n=1.0, rho_clip=(0.5, 2.0))


def _column_2d(refinement: int = 2):
    """Unit square, frozen bottom and sides, free top; uniform density."""
    result = rectangle(L=1.0, h=1.0, refinement=refinement, flow_axis=0)
    HC = result.HC
    walls = {v for v in HC.V
             if v.x_a[1] < 1e-12 or v.x_a[0] < 1e-12 or v.x_a[0] > 1 - 1e-12}
    bV = set(walls)
    for v in HC.V:
        v.boundary = v in result.bV
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, 2)
    ZeroVelocity(dim=2).apply(HC, bV)
    DualVolumeMass(rho=RHO0).apply(HC, bV)
    return HC, bV, frozenset(walls)


def _dent_surface(HC, walls, depth: float = 0.02) -> None:
    """Push the free-surface vertices down (more in the middle), so the
    fluid boundary is no longer convex."""
    for v in list(HC.V):
        if v not in walls and abs(v.x_a[1] - 1.0) < 1e-12:
            x = v.x_a[0]
            HC.V.move(v, (x, 1.0 - depth * np.sin(np.pi * x)))


def _volume(HC, dim: int) -> float:
    return float(sum(simplex_dual_volumes(HC, dim).values()))


def _edges(HC) -> set:
    return {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}


def _facet_ids(boundary) -> set:
    return {frozenset(id(v) for v in f) for f in boundary}


def _rebuild(HC, dim: int) -> None:
    """Plain Delaunay of the cloud (covers the convex hull)."""
    verts = list(HC.V)
    for v in verts:
        for nb in list(v.nn):
            v.disconnect(nb)
    connect_and_cache_simplices(
        HC, verts, dim, coords=np.array([v.x_a[:dim] for v in verts]))


class TestPeel:
    def test_oriented_boundary_of_the_builder_mesh(self):
        HC, _, _ = _column_2d()
        boundary = oriented_boundary(HC._simplices, 2)
        # refinement 2: four boundary edges per side of the unit square
        assert len(boundary) == 16
        on_hull = {id(v) for v in boundary_from_simplices(HC, 2)}
        assert set().union(*_facet_ids(boundary)) == on_hull
        fac = _facet_points(boundary, 2)
        assert enclosed_volume(fac) == pytest.approx(1.0, rel=1e-14)
        w = winding_number(np.array([[0.5, 0.5], [0.01, 0.99], [0.5, 1.01],
                                     [-0.2, 0.3], [3.0, 3.0]]), fac)
        assert w == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0], abs=1e-12)

    def test_oriented_boundary_3d(self):
        HC = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1, flow_axis=2).HC
        fac = _facet_points(oriented_boundary(HC._simplices, 3), 3)
        assert len(fac) == 6 * 8               # 2 x 2 squares per face
        assert enclosed_volume(fac) == pytest.approx(1.0, rel=1e-14)
        w = winding_number(np.array([[0.5, 0.5, 0.5], [0.01, 0.02, 0.99],
                                     [0.5, 0.5, 1.01], [2.0, 0.5, 0.5]]), fac)
        assert w == pytest.approx([1.0, 1.0, 0.0, 0.0], abs=1e-12)

    def test_convex_hull_fill_is_removed(self):
        HC, _, walls = _column_2d()
        boundary = oriented_boundary(HC._simplices, 2)
        _dent_surface(HC, walls)
        v_domain = _volume(HC, 2)             # old connectivity, new positions
        assert v_domain < 1.0 - 1e-3
        fac = _facet_points(boundary, 2)
        assert enclosed_volume(fac) == pytest.approx(v_domain, rel=1e-13)
        # a point in the dent is outside the fluid, one below it inside
        assert winding_number(np.array([[0.5, 0.99], [0.5, 0.97]]),
                              fac) == pytest.approx([0.0, 1.0], abs=1e-12)

        _rebuild(HC, 2)
        # Delaunay covers the hull: the dent is filled
        assert _volume(HC, 2) == pytest.approx(1.0, rel=1e-12)

        n_removed = peel_outside_boundary(HC, boundary, 2)
        assert n_removed > 0
        assert _volume(HC, 2) == pytest.approx(v_domain, rel=1e-12)
        assert _facet_ids(oriented_boundary(HC._simplices, 2)) == _facet_ids(
            boundary)
        # the connectivity is exactly the edge set of the kept simplices
        kept_edges = {frozenset((id(a), id(b))) for s in HC._simplices
                      for i, a in enumerate(s) for b in s[i + 1:]}
        assert _edges(HC) == kept_edges

    def test_nothing_to_peel_on_a_convex_domain(self):
        HC, _, _ = _column_2d()
        cache = HC._simplices
        assert peel_outside_boundary(HC, oriented_boundary(cache, 2), 2) == 0
        assert HC._simplices is cache

    def test_orientation_is_taken_when_the_simplices_are_built(self):
        """A thin boundary simplex may invert between two rebuilds.  The
        boundary oriented at build time is still a closed curve (integer
        winding numbers); one re-derived from the inverted simplex is
        not, which is why the retopology function caches it."""
        HC = Complex(2, domain=[(0.0, 1.0), (0.0, 1.1)])
        q, p1, p3 = HC.V[(0.5, 0.0)], HC.V[(0.0, 1.0)], HC.V[(1.0, 1.0)]
        p2 = HC.V[(0.5, 1.05)]
        simplices = [(p1, p3, q), (p1, p2, p3)]      # bulk + thin cap
        boundary = oriented_boundary(simplices, 2)
        probes = np.array([[0.5, 0.5], [0.5, 1.02], [0.5, 0.95], [0.5, 1.2]])
        assert winding_number(probes, _facet_points(boundary, 2)) == (
            pytest.approx([1.0, 1.0, 1.0, 0.0], abs=1e-12))

        HC.V.move(p2, (0.5, 0.9))                    # the cap inverts
        fac = _facet_points(boundary, 2)
        assert enclosed_volume(fac) == pytest.approx(0.45, rel=1e-14)
        assert winding_number(probes, fac) == (
            pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-12))
        w = winding_number(probes, _facet_points(
            oriented_boundary(simplices, 2), 2))
        assert np.abs(w - np.round(w)).max() > 0.2


def _lattice_cube(n: int):
    """Plain n^3 lattice of the unit cube, Delaunay connectivity: unlike
    the builder mesh it has tetrahedra whose four vertices are all on the
    boundary, and qhull returns flat tetrahedra in the wall planes."""
    HC = Complex(3, domain=[(0.0, 1.0)] * 3)
    g = np.linspace(0.0, 1.0, n)
    for idx in np.ndindex(n, n, n):
        HC.V[tuple(float(g[i]) for i in idx)]
    _rebuild(HC, 3)
    return HC


def _n_exposed_flat(HC, dim: int) -> int:
    """Flat simplices of the cache that own a boundary facet."""
    flat = _measure(_points(HC._simplices, dim))[1]
    n_owners: dict = {}
    for s in HC._simplices:
        for j in range(dim + 1):
            key = frozenset(id(v) for i, v in enumerate(s) if i != j)
            n_owners[key] = n_owners.get(key, 0) + 1
    return sum(
        bool(flat[si]) and any(
            n_owners[frozenset(id(v) for i, v in enumerate(s) if i != j)] == 1
            for j in range(dim + 1))
        for si, s in enumerate(HC._simplices))


def _is_interior(v) -> bool:
    x = np.array(v.x_a[:3])
    return bool(np.all((x > 1e-9) & (x < 1.0 - 1e-9)))


class TestPeel3D:
    """The defects of the first (topological) peel, lane P review."""

    def test_lattice_cube_keeps_its_fluid(self):
        """Convex domain, fixed planar walls, interior vertices jittered.
        The wall squares are cocircular, so their Delaunay diagonals
        change between rebuilds; the topological peel took every
        tetrahedron with four wall vertices for hull fill and the total
        volume fell to 0.790 (29 of 30 steps below 1)."""
        HC = _lattice_cube(4)
        rng = np.random.default_rng(0)
        bV: set = set()
        wall_ids = {id(v) for v in HC.V if not _is_interior(v)}
        n_all_boundary = 0
        for _ in range(10):
            for v in list(HC.V):
                if _is_interior(v):
                    HC.V.move(v, tuple(np.array(v.x_a[:3])
                                       + 1e-3 * rng.standard_normal(3)))
            change = retopologize_material_delaunay(HC, bV, 3)
            assert abs(change) < 1e-12
            assert sum(v.dual_vol for v in HC.V) == pytest.approx(1.0,
                                                                  rel=1e-12)
            assert {id(v) for v in bV} == wall_ids
            # no flat tetrahedron is left in a wall plane (qhull also
            # returns flat ones between the cospherical lattice cells;
            # those are enclosed and stay, as on the plain rebuild)
            assert _n_exposed_flat(HC, 3) == 0
            n_all_boundary += sum(all(id(v) in wall_ids for v in s)
                                  for s in HC._simplices)
        assert n_all_boundary > 0, "probe must have all-boundary tetrahedra"

    def test_one_call_on_the_coarsest_lattice(self):
        """n = 3: one interior vertex; the topological peel lost 41.7 %
        of the volume in one call."""
        HC = _lattice_cube(3)
        centre = next(v for v in HC.V if _is_interior(v))
        HC.V.move(centre, (0.501, 0.499, 0.5005))
        change = retopologize_material_delaunay(HC, set(), 3)
        assert abs(change) < 1e-12
        assert _volume(HC, 3) == pytest.approx(1.0, rel=1e-12)

    def test_closed_box_with_a_stirred_interior(self):
        """Builder mesh, closed box, interior random walk of 0.03 per
        step: 15 of 40 steps lost fluid before (minimum 0.979)."""
        result = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2, flow_axis=2)
        HC = result.HC
        walls = frozenset(result.bV)
        bV = set(walls)
        rng = np.random.default_rng(0)
        for _ in range(12):
            for v in list(HC.V):
                if v not in walls:
                    x = np.array(v.x_a[:3]) + 0.03 * rng.standard_normal(3)
                    HC.V.move(v, tuple(np.clip(x, 0.02, 0.98)))
            change = retopologize_material_delaunay(
                HC, bV, 3, boundary_filter=lambda v: v in walls)
            assert abs(change) < 1e-12
            assert sum(v.dual_vol for v in HC.V) == pytest.approx(1.0,
                                                                  rel=1e-12)
            assert bV == set(walls)

    def test_non_planar_free_surface(self):
        """Free surface pushed into a bowl (not planar).  The hull fill
        goes, the walls stay frozen, and the domain is kept up to the
        slivers between the old surface diagonals and the Delaunay ones:
        the rebuild is not constrained to the old surface facets.  The
        change is what the function returns."""
        result = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2, flow_axis=2)
        HC = result.HC
        walls = frozenset(v for v in HC.V if v.x_a[2] < 1.0 - 1e-12
                          and v in result.bV)
        surface = [v for v in result.bV if v not in walls]
        assert len(surface) == 25
        bV = set(walls)

        def push(depth: float) -> None:
            for v in surface:
                x, y, z = v.x_a[:3]
                HC.V.move(v, (x, y, z - depth * np.sin(np.pi * x)
                              * np.sin(np.pi * y)))

        push(0.004)
        v_domain = _volume(HC, 3)             # old connectivity, new positions
        assert v_domain < 1.0 - 1e-3
        change = retopologize_material_delaunay(
            HC, bV, 3, boundary_filter=lambda v: v in walls)
        v_new = sum(v.dual_vol for v in HC.V)
        assert v_new == pytest.approx(v_domain * (1.0 + change), rel=1e-12)
        assert v_new < 1.0 - 1e-3             # the bowl is not filled
        # at most one sliver per surface square, each below h^2 depth / 6;
        # measured +9.78e-05 (a known limit: no facet recovery)
        assert 1e-6 < abs(change) < 0.004 / 6.0
        assert bV == set(walls)
        assert all(v.boundary for v in surface)
        assert not _measure(_points(HC._simplices, 3))[1].any()

        # second call: the boundary cached by the first one is used and the
        # surface triangulation is now the Delaunay one, so nothing changes
        assert HC._material_boundary[0] is HC._simplices
        push(0.0)
        again = retopologize_material_delaunay(
            HC, bV, 3, boundary_filter=lambda v: v in walls)
        assert abs(again) < 1e-12
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(v_new, rel=1e-12)


class TestRetopologizeMaterialDelaunay:
    def test_domain_volume_and_frozen_set(self):
        HC, bV, walls = _column_2d()
        _dent_surface(HC, walls)
        v_domain = _volume(HC, 2)
        e_before = _edges(HC)
        retopologize_material_delaunay(HC, bV, 2,
                                       boundary_filter=lambda v: v in walls)
        assert _edges(HC) != e_before, "probe must reconnect"
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(v_domain,
                                                              rel=1e-12)
        assert bV == set(walls)
        tagged = {v for v in HC.V if v.boundary}
        assert tagged == boundary_from_simplices(HC, 2)
        assert len(tagged) == 16               # free surface is boundary
        assert HC._edge_area_cache is None

    def test_domain_change_is_returned_and_warned(self):
        HC, bV, walls = _column_2d()
        _dent_surface(HC, walls)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            change = retopologize_material_delaunay(
                HC, bV, 2, boundary_filter=lambda v: v in walls)
        assert abs(change) < 1e-12
        # an interior vertex leaves through the free surface: its simplices
        # are kept (it would be orphaned otherwise), so the domain grows
        v_domain = _volume(HC, 2)
        inner = min((v for v in HC.V if not v.boundary),
                    key=lambda v: abs(v.x_a[0] - 0.375) + abs(v.x_a[1] - 0.875))
        HC.V.move(inner, (inner.x_a[0], 1.1))
        with pytest.warns(UserWarning, match='domain volume changed'):
            change = retopologize_material_delaunay(
                HC, bV, 2, boundary_filter=lambda v: v in walls)
        assert change > 1e-3
        assert _volume(HC, 2) == pytest.approx(v_domain * (1.0 + change),
                                               rel=1e-12)

    def test_convex_rebuild_changes_the_domain(self):
        HC, bV, walls = _column_2d()
        _dent_surface(HC, walls)
        _retopologize(HC, bV, 2, boundary_filter=lambda v: v in walls)
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(1.0, rel=1e-12)

    def test_remap_keeps_the_pressure_and_the_mass(self):
        HC, bV, walls = _column_2d()
        _dent_surface(HC, walls)
        eos = _eos()
        p_before = snapshot_pressure_fresh(HC, 2, eos)
        m_before = sum(v.m for v in HC.V)
        retopologize_material_delaunay(
            HC, bV, 2, boundary_filter=lambda v: v in walls,
            pressure_model=eos, redistribute_mass=True,
            retopo_remap='conservative')
        assert sum(v.m for v in HC.V) == pytest.approx(m_before, rel=1e-14)
        ratio = np.array([v.m / v.dual_vol / float(eos.density(p_before[id(v)]))
                          for v in HC.V])
        assert len(ratio) == len(p_before)
        assert np.ptp(ratio) < 1e-12           # one scale for every vertex
        # The scale differs from 1 by (density contrast) x (volume that
        # changed cells); the 2 % dent is a strong probe (measured 3.0e-4,
        # on the settling column the offset K (s - 1) stays below 1 Pa).
        assert abs(ratio[0] - 1.0) < 1e-2

    def test_validation(self):
        HC, bV, _ = _column_2d()
        with pytest.raises(ValueError, match='go together'):
            retopologize_material_delaunay(HC, bV, 2, redistribute_mass=True)
        with pytest.raises(ValueError, match='go together'):
            retopologize_material_delaunay(HC, bV, 2, pressure_model=_eos(),
                                           retopo_remap='conservative')
        with pytest.raises(ValueError, match='EquationOfState'):
            retopologize_material_delaunay(HC, bV, 2, redistribute_mass=True,
                                           retopo_remap='conservative')
        with pytest.raises(ValueError, match='retopo_remap'):
            retopologize_material_delaunay(HC, bV, 2, retopo_remap='bogus')
        with pytest.raises(ValueError, match='dim 2'):
            retopologize_material_delaunay(HC, bV, 1)
        HC._simplices = None
        with pytest.raises(ValueError, match='_simplices'):
            retopologize_material_delaunay(HC, bV, 2)

    def test_3d_keeps_half_cells_and_the_domain(self):
        result = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=1, flow_axis=2)
        HC = result.HC
        walls = frozenset(v for v in HC.V if v.x_a[2] < 1.0 - 1e-12
                          and v in result.bV)
        bV = set(walls)
        for v in list(HC.V):
            if v not in walls and abs(v.x_a[2] - 1.0) < 1e-12:
                HC.V.move(v, (v.x_a[0], v.x_a[1], 0.97))
        v_domain = _volume(HC, 3)
        assert v_domain < 1.0 - 1e-3
        retopologize_material_delaunay(HC, bV, 3,
                                       boundary_filter=lambda v: v in walls)
        assert sum(v.dual_vol for v in HC.V) == pytest.approx(v_domain,
                                                              rel=1e-12)
        assert bV == set(walls)
        assert all(v.dual_vol > 0.0 for v in walls)     # not zeroed


class TestColumnThroughTheIntegrator:
    """Free-surface column under gravity, uniform density at t = 0, 6
    acoustic times, refinement 2 (41 vertices): every arm is a
    ``SolverMethods`` run through ``integrate``."""

    N_TAC = 6.0

    def _run(self, methods: SolverMethods):
        HC, bV, walls = _column_2d()
        eos = _eos()
        c0 = float(eos.sound_speed(RHO0))
        edges = [float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
                 for v in HC.V for nb in v.nn]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dudt_fn = methods.dudt_fn(
                HC, mu=0.5 * RHO0 * c0 * float(np.mean(edges)),
                pressure_model=eos, body_force=[0.0, -G])
        bc_set = BoundaryConditionSet()
        bc_set.add(NoSlipWallBC(dim=2), bV)
        dt = 0.25 * min(edges) / c0
        umax: list[float] = []

        def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
            umax.append(max(float(np.linalg.norm(v.u[:2])) for v in HC_cb.V))

        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            methods.integrate(HC, bV, dudt_fn, dt=dt,
                              n_steps=int(round(self.N_TAC / c0 / dt)),
                              bc_set=bc_set, callback=callback,
                              pressure_model=eos,
                              boundary_filter=lambda v: v in walls)
        volume = _volume(HC, 2)
        p_bottom = np.mean([float(eos.pressure(v.m / v.dual_vol))
                            for v in HC.V if v.x_a[1] < 1e-12])
        return np.array(umax), volume, p_bottom, len(bV)

    @pytest.fixture(scope='class')
    def runs(self):
        material = SolverMethods(dim=2, connectivity='delaunay_material',
                                 remap='conservative', redistribute_mass=True)
        return {
            'material': self._run(material),
            'convex': self._run(material.replace(connectivity='delaunay')),
            'dual_only': self._run(SolverMethods(dim=2,
                                                 connectivity='dual_only')),
        }

    def test_material_arm_follows_the_fixed_connectivity_run(self, runs):
        u_m, vol_m, p_m, n_frozen = runs['material']
        u_d, vol_d, p_d, _ = runs['dual_only']
        assert n_frozen == 13                       # top vertices stay free
        assert u_m.max() == pytest.approx(u_d.max(), rel=0.1)
        # the column compresses by rho g H / (2 K) = 0.5 % under its weight
        assert vol_d == pytest.approx(0.995, abs=3e-4)
        assert vol_m == pytest.approx(vol_d, abs=1e-4)
        # both carry the full head rho g H at the bottom (still ringing at
        # 6 acoustic times: measured -3.0 % and -0.8 %)
        assert p_m == pytest.approx(RHO0 * G, rel=0.05)
        assert p_d == pytest.approx(RHO0 * G, rel=0.05)

    def test_convex_arm_cannot_hold_the_column(self, runs):
        """The hull is pinned by the frozen top corners, so the convex
        rebuild keeps the total volume at 1: the column cannot compress
        and the hydrostatic head never develops.  Until laneO (2026-10-05)
        this arm also read flipped dual area vectors (9 in 5 force
        evaluations, ``area_orientation='dual_midpoint'``) and peaked at
        0.6219 m/s = 3.19 times the fixed-connectivity arm; with the
        correct orientation it peaks at 1.14 times (pinned below, the
        legacy value is kept in test_area_orientation.py)."""
        u_c, vol_c, p_c, _ = runs['convex']
        u_d, _, p_d, _ = runs['dual_only']
        assert vol_c == pytest.approx(1.0, abs=1e-9)
        assert u_c.max() == pytest.approx(PIN_CONVEX_UMAX, rel=1e-6)
        assert u_c.max() > 1.1 * u_d.max()
        assert p_c < 0.6 * RHO0 * G           # measured 0.4848 rho g H

    def test_material_arm_pinned(self, runs):
        u_m, vol_m, p_m, _ = runs['material']
        assert u_m.max() == pytest.approx(PIN_UMAX, rel=1e-6)
        assert vol_m == pytest.approx(PIN_VOLUME, rel=1e-9)


# Pinned 2026-10-01 (laneP): SolverMethods(dim=2,
# connectivity='delaunay_material', remap='conservative',
# redistribute_mass=True), refinement 2, 6 acoustic times.
PIN_UMAX = 0.20282519399577198
PIN_VOLUME = 0.995147730045464
# Pinned 2026-10-05 (laneO): the same with connectivity='delaunay' (the
# convex-hull arm).  Peak velocity; 0.6218859935218073 before laneO with
# the flipped area vectors (area_orientation='dual_midpoint').
PIN_CONVEX_UMAX = 0.22276943647536884
