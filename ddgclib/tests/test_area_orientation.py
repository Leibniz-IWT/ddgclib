"""Orientation of the 2D dual area vector (laneO, 2026-10-05): method axis
``area_orientation``.

``dual_area_vector`` returns, in 2D, the normal of the segment between the
two dual vertices the endpoints of an edge share.  Its sign must be
"outward from i": ``A_ij . d_ij > 0`` with ``d_ij = x_j - x_i``
(``'primal_edge'``, the default).  The rule before laneO
(``'dual_midpoint'``: away from ``x_i`` as seen from the midpoint of the
segment) flips the vector when the two triangles at the edge subtend more
than 180 degrees at ``x_i``; it is kept, registered as broken, so that the
numbers pinned before the fix can be reproduced.

The reference is ``simplex_area_vectors``: the exact barycentric dual face
vectors from the simplex cache, with no orientation choice.
"""
import warnings
from functools import partial

import numpy as np
import numpy.testing as npt
import pytest

from hyperct.ddg import compute_vd

from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
from ddgclib.geometry.domains import box, disk, rectangle
from ddgclib.methods import AXES, SolverMethods
from ddgclib.operators.stress import (
    AREA_ORIENTATIONS, cache_dual_volumes, dual_area_vector,
    simplex_area_vectors, stress_force,
)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

def _sheared_jittered(refinement=3, shear=0.3, jitter=0.2, seed=0, L=2.0,
                      rebuilds=1):
    """The laneH mesh: a channel whose interior vertices are sheared by
    ``shear * 4 y (1 - y)`` in x and jittered by ``jitter`` mean spacings,
    reconnected by the library Delaunay retopology."""
    rng = np.random.default_rng(seed)
    res = rectangle(L=L, h=1.0, refinement=refinement, flow_axis=0)
    HC, bV = res.HC, res.bV
    h = (L / sum(1 for _ in HC.V)) ** 0.5
    for _ in range(rebuilds):
        HC.V.move_all([
            (v, (v.x_a[0] + shear * 4 * v.x_a[1] * (1 - v.x_a[1]) / rebuilds
                 + jitter * h * rng.uniform(-1, 1),
                 v.x_a[1] + jitter * h * rng.uniform(-1, 1)))
            for v in list(HC.V) if v not in bV])
        _retopologize(HC, bV, 2)
    return HC


def _built(res):
    compute_vd(res.HC, method='barycentric')
    cache_dual_volumes(res.HC, 2)
    return res.HC


MESHES = {
    'rectangle': lambda: _built(rectangle(L=2.0, h=1.0, refinement=3)),
    'disk': lambda: _built(disk(R=1.0, refinement=3)),
    'laneH_sheared_jittered': lambda: _sheared_jittered(),
    'jittered': lambda: _sheared_jittered(shear=0.0),
    'strongly_sheared': lambda: _sheared_jittered(shear=0.6, jitter=0.3,
                                                  seed=1),
    'reconnected_twice': lambda: _sheared_jittered(shear=0.6, jitter=0.25,
                                                   seed=2, rebuilds=2),
}


def _edge_count(HC):
    """Number of top simplices at each undirected edge (2D)."""
    count: dict = {}
    for s in HC._simplices:
        for a in s:
            for b in s:
                if id(a) < id(b):
                    key = frozenset((id(a), id(b)))
                    count[key] = count.get(key, 0) + 1
    return count


def _hull_half_faces(v, HC):
    """Outward vectors of the two hull half faces of a 2D hull vertex *v*
    (from *v* to the midpoint of each of its hull edges), from the
    simplex cache."""
    faces = []
    for s in HC._simplices:
        if v not in s:
            continue
        others = [w for w in s if w is not v]
        for w in others:
            k = others[0] if others[1] is w else others[1]
            n_tri = sum(1 for t in HC._simplices if v in t and w in t)
            if n_tri != 1:
                continue
            e = w.x_a[:2] - v.x_a[:2]
            n = 0.5 * np.array([-e[1], e[0]])
            if n @ (k.x_a[:2] - v.x_a[:2]) > 0:
                n = -n
            faces.append(n)
    return faces


# ---------------------------------------------------------------------------
# the sign rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(MESHES))
class TestPrimalEdgeRule:
    def test_every_vector_points_along_its_edge_and_is_antisymmetric(
            self, name):
        HC = MESHES[name]()
        n = 0
        for v in HC.V:
            for nb in v.nn:
                A = dual_area_vector(v, nb, HC, 2)
                B = dual_area_vector(nb, v, HC, 2)
                assert np.linalg.norm(A) > 0.0
                assert float(A @ (nb.x_a[:2] - v.x_a[:2])) > 0.0
                npt.assert_array_equal(B, -A)
                n += 1
        assert n > 200

    def test_interior_cells_close(self, name):
        HC = MESHES[name]()
        n = 0
        for v in HC.V:
            if v.boundary:
                continue
            total = np.zeros(2)
            scale = 0.0
            for nb in v.nn:
                A = dual_area_vector(v, nb, HC, 2)
                total += A
                scale += float(np.linalg.norm(A))
            assert np.linalg.norm(total) < 1e-13 * scale
            n += 1
        assert n > 50

    def test_hull_half_cells_close_with_their_hull_faces(self, name):
        HC = MESHES[name]()
        n = 0
        for v in HC.V:
            if not v.boundary:
                continue
            faces = _hull_half_faces(v, HC)
            assert len(faces) == 2, name
            total = sum((dual_area_vector(v, nb, HC, 2) for nb in v.nn),
                        np.zeros(2)) + faces[0] + faces[1]
            scale = sum(float(np.linalg.norm(f)) for f in faces)
            assert np.linalg.norm(total) < 1e-13 * scale
            n += 1
        assert n > 20

    def test_equals_the_simplex_cache_reference_on_every_edge(self, name):
        HC = MESHES[name]()
        for v in HC.V:
            verts, A_ref = simplex_area_vectors(v, HC, 2)
            ref = {id(w): a for w, a in zip(verts, A_ref)}
            assert set(ref) == {id(nb) for nb in v.nn}
            for nb in v.nn:
                A = dual_area_vector(v, nb, HC, 2)
                # the segment is built from absolute barycentre coordinates
                # (|x| up to 2), the reference from offsets x_k - x_v
                npt.assert_allclose(A, ref[id(nb)], rtol=0, atol=1e-14)


class TestLegacyRule:
    """``'dual_midpoint'`` reproduces the defect on record."""

    def test_counts_of_the_laneH_mesh(self):
        HC = _sheared_jittered()
        against = closure = n_interior_edges = 0
        for v in HC.V:
            total = np.zeros(2)
            for nb in v.nn:
                A = dual_area_vector(v, nb, HC, 2, 'dual_midpoint')
                against += float(A @ (nb.x_a[:2] - v.x_a[:2])) < 0.0
                total += A
                n_interior_edges += not v.boundary
            if not v.boundary:
                closure = max(closure, float(np.linalg.norm(total)))
        assert against == 6 and n_interior_edges == 665
        assert closure == pytest.approx(0.2316, abs=5e-4)

    def test_disk_builder_mesh_has_flipped_vectors_too(self):
        HC = MESHES['disk']()
        against = sum(
            float(dual_area_vector(v, nb, HC, 2, 'dual_midpoint')
                  @ (nb.x_a[:2] - v.x_a[:2])) < 0.0
            for v in HC.V for nb in v.nn)
        assert against == 4

    def test_rules_agree_where_the_legacy_rule_is_right(self):
        HC = MESHES['rectangle']()
        for v in HC.V:
            for nb in v.nn:
                npt.assert_array_equal(
                    dual_area_vector(v, nb, HC, 2, 'dual_midpoint'),
                    dual_area_vector(v, nb, HC, 2, 'primal_edge'))

    def test_unknown_rule_raises(self):
        HC = MESHES['rectangle']()
        for w in HC.V:
            w.p, w.u = 0.0, np.zeros(2)
        v = next(iter(HC.V))
        nb = next(iter(v.nn))
        with pytest.raises(KeyError):
            dual_area_vector(v, nb, HC, 2, 'outward')
        with pytest.raises(KeyError):
            stress_force(v, dim=2, mu=0.0, HC=HC, area_orientation='outward')


def _linear_pressure_error(HC, orientation):
    """Largest |F + V g| / (V |g|) of the centred force of a linear nodal
    pressure over the interior cells."""
    g = np.array([0.7, -1.3])
    for v in HC.V:
        v.p = float(g @ v.x_a[:2])
        v.u = np.zeros(2)
    worst = 0.0
    for v in HC.V:
        if v.boundary:
            continue
        F = stress_force(v, dim=2, mu=0.0, HC=HC,
                         area_orientation=orientation)
        worst = max(worst, float(np.linalg.norm(F + v.dual_vol * g))
                    / (v.dual_vol * float(np.linalg.norm(g))))
    return worst


class TestForce:
    def test_centred_pressure_force_is_linearly_precise(self):
        HC = _sheared_jittered()
        assert _linear_pressure_error(HC, 'primal_edge') < 1e-12
        assert _linear_pressure_error(HC, 'dual_midpoint') > 10.0   # 17.6

    def test_two_point_edge_weights_are_positive(self):
        HC = _sheared_jittered()
        for v in HC.V:
            for nb in v.nn:
                d = nb.x_a[:2] - v.x_a[:2]
                A = dual_area_vector(v, nb, HC, 2)
                assert float(d @ A) / float(d @ d) > 0.0


# ---------------------------------------------------------------------------
# periodic branch (minimum image)
# ---------------------------------------------------------------------------

class TestPeriodicBranch:
    def _mesh(self):
        from ddgclib.geometry.domains import periodic_rectangle
        from ddgclib.geometry.periodic import retopologize_periodic
        res = periodic_rectangle(L=1.0, h=1.0, refinement=3,
                                 periodic_axes=[0])
        HC, bV = res.HC, res.bV
        rng = np.random.default_rng(4)
        HC.V.move_all([
            (v, (v.x_a[0] + 0.3 * 4 * v.x_a[1] * (1 - v.x_a[1])
                 + 0.03 * rng.uniform(-1, 1),
                 v.x_a[1] + 0.03 * rng.uniform(-1, 1)))
            for v in list(HC.V) if v not in bV])
        retopologize_periodic(HC, bV, dim=2, periodic_axes=[0],
                              domain_bounds=[(0.0, 1.0), (0.0, 1.0)])
        return HC

    @staticmethod
    def _d(v, nb, HC):
        d = nb.x_a[:2] - v.x_a[:2]
        lo, hi = HC._periodic_bounds[0]
        d[0] -= round(d[0] / (hi - lo)) * (hi - lo)
        return d

    def test_along_the_minimum_image_edge_and_antisymmetric(self):
        """Every non-zero vector points along the minimum-image edge and,
        away from the seam, is the negative of its partner.  Known limit
        of the periodic branch, not of the sign rule (laneO, measured on
        this mesh): on 62 of the 886 directed edges with a non-zero vector
        the two endpoints build different dual segments (61 of them cross
        the seam: the common neighbours are min-imaged about x_i, so the
        two sides can pick different periodic images) and on 2 more one
        side returns a zero vector.  Those pairs are counted, not
        asserted; the legacy rule flips 13 of the 886."""
        HC = self._mesh()
        assert HC._periodic_axes
        n = against_legacy = pairs_differ = 0
        for v in HC.V:
            for nb in v.nn:
                A = dual_area_vector(v, nb, HC, 2)
                B = dual_area_vector(nb, v, HC, 2)
                if np.linalg.norm(A) == 0.0:
                    continue
                d = self._d(v, nb, HC)
                assert float(A @ d) > 0.0
                pairs_differ += not np.array_equal(B, -A)
                against_legacy += float(
                    dual_area_vector(v, nb, HC, 2, 'dual_midpoint') @ d) < 0.0
                n += 1
        assert n > 500
        assert against_legacy > 0
        assert pairs_differ < 0.1 * n


# ---------------------------------------------------------------------------
# the method axis
# ---------------------------------------------------------------------------

class TestAxis:
    def test_registry(self):
        ax = AXES['area_orientation']
        assert ax.explicit and ax.default == 'primal_edge'
        assert tuple(o.key for o in ax.options) == AREA_ORIENTATIONS
        assert ax.option('dual_midpoint').status == 'broken'
        assert ax.option('dual_midpoint').dims == (2,)

    def test_legacy_is_2d_only_and_warns(self):
        with pytest.warns(UserWarning, match='broken'):
            m = SolverMethods(dim=2, area_orientation='dual_midpoint')
        assert m.area_orientation == 'dual_midpoint'
        with pytest.raises(ValueError):
            SolverMethods(dim=3, area_orientation='dual_midpoint')
        with pytest.raises(ValueError):
            SolverMethods(dim=2, area_orientation='outward')

    def test_default_binds_nothing(self):
        HC = MESHES['rectangle']()
        fn = SolverMethods(dim=2).dudt_fn(HC, mu=0.1)
        assert 'area_orientation' not in fn.keywords
        from ddgclib.multiphase import MultiphaseSystem
        m = SolverMethods(dim=2, phases='multi', redistribute_mass=True)
        fn = m.dudt_fn(HC, mps=MultiphaseSystem.__new__(MultiphaseSystem))
        assert 'area_orientation' not in fn.keywords

    def test_legacy_is_bound_into_both_partials(self):
        HC = MESHES['rectangle']()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = SolverMethods(dim=2, area_orientation='dual_midpoint')
            m = SolverMethods(dim=2, phases='multi', redistribute_mass=True,
                              area_orientation='dual_midpoint')
        fn = s.dudt_fn(HC, mu=0.1)
        assert isinstance(fn, partial)
        assert fn.keywords['area_orientation'] == 'dual_midpoint'
        from ddgclib.multiphase import MultiphaseSystem
        fn = m.dudt_fn(HC, mps=MultiphaseSystem.__new__(MultiphaseSystem))
        assert fn.keywords['area_orientation'] == 'dual_midpoint'

    def test_single_phase_partial_reproduces_the_legacy_force(self):
        HC = _sheared_jittered()
        g = np.array([0.7, -1.3])
        for v in HC.V:
            v.p = float(g @ v.x_a[:2])
            v.u = np.zeros(2)
            v.m = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            legacy = SolverMethods(dim=2, area_orientation='dual_midpoint')
        fn_new = SolverMethods(dim=2).dudt_fn(HC, mu=0.0)
        fn_old = legacy.dudt_fn(HC, mu=0.0)
        interior = [v for v in HC.V if not v.boundary]
        differ = sum(not np.array_equal(fn_new(v), fn_old(v))
                     for v in interior)
        assert differ > 0
        for v in interior:
            npt.assert_array_equal(
                fn_old(v), stress_force(v, dim=2, mu=0.0, HC=HC,
                                        area_orientation='dual_midpoint'))


# ---------------------------------------------------------------------------
# the legacy rule reproduces the numbers pinned before laneO
# ---------------------------------------------------------------------------

class TestLegacyReproducesThePreLaneOPins:
    """The three pinned 2D runs whose integrated vertices read flipped
    vectors before laneO (every other 2D pin never read one and is
    bit-identical).  Each is ``preset.replace(area_orientation=
    'dual_midpoint')`` and must give the number of its lane log."""

    def test_laneP_convex_hull_arm(self):
        from ddgclib.tests.test_material_delaunay import (
            PIN_CONVEX_UMAX, TestColumnThroughTheIntegrator,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            legacy = SolverMethods(dim=2, connectivity='delaunay',
                                   remap='conservative',
                                   redistribute_mass=True,
                                   area_orientation='dual_midpoint')
        umax, volume, p_bottom, _ = TestColumnThroughTheIntegrator()._run(
            legacy)
        assert umax.max() == pytest.approx(0.6218859935218073, rel=1e-6)
        assert umax.max() > 2.0 * PIN_CONVEX_UMAX
        assert volume == pytest.approx(1.0, abs=1e-9)
        assert p_bottom == pytest.approx(4900.947188722312, rel=1e-6)

    def test_laneR_bare_delaunay_box(self):
        from ddgclib.tests import test_single_phase_remap as t
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            legacy = t.BARE.replace(area_orientation='dual_midpoint')
        _, ke, umax = t._run(legacy)
        assert ke[-1] == pytest.approx(3253.490893153858, rel=1e-6)
        assert np.nanmax(umax) / t.U0 == pytest.approx(121.54072533271666,
                                                       rel=1e-6)
        # unstable under both rules: the KE doubles at the first step,
        # 32 steps before the first flipped vector
        assert ke[1] > 2.0 * ke[0]

    def test_laneL_hull_collapse_reproducer(self):
        from ddgclib.methods import PRESETS
        from ddgclib.tests.test_frozen_set import TestHagenPoiseuille2D as T
        name = 'hagen_poiseuille_2D'
        preset = PRESETS[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PRESETS[name] = preset.replace(area_orientation='dual_midpoint')
            try:
                report, n_on_wall = T._run('hull')
            finally:
                PRESETS[name] = preset
        assert report['max_displacement'] == pytest.approx(
            0.061843693307756, rel=1e-6)
        first_drop = next(i for i in range(1, len(n_on_wall))
                          if n_on_wall[i] < n_on_wall[i - 1])
        assert first_drop == 250                 # 249 since laneO


# ---------------------------------------------------------------------------
# the reference in 3D
# ---------------------------------------------------------------------------

class TestSimplexReference3D:
    def test_closed_antisymmetric_and_equal_to_the_p_ij_ring(self):
        """On a lightly jittered box the ``p_ij`` ring of an interior edge
        is exact (laneJ), so the reference must agree with it."""
        from ddgclib.operators.stress import _dual_area_vector_3d_p_ij
        rng = np.random.default_rng(1)
        res = box(Lx=1.0, Ly=1.0, Lz=1.0, refinement=2)
        HC, bV = res.HC, res.bV
        HC.V.move_all([(v, tuple(v.x_a + 0.03 * rng.uniform(-1, 1, 3)))
                       for v in list(HC.V) if v not in bV])
        _retopologize(HC, bV, 3)
        refs = {id(v): dict(zip(map(id, a), b))
                for v in HC.V for a, b in [simplex_area_vectors(v, HC, 3)]}
        n = 0
        for v in HC.V:
            ref = refs[id(v)]
            if not v.boundary:
                total = sum(ref.values(), np.zeros(3))
                assert np.linalg.norm(total) < 1e-14
            for nb in v.nn:
                npt.assert_allclose(
                    ref[id(nb)], -refs[id(nb)][id(v)], rtol=0,
                    atol=1e-14 * np.linalg.norm(ref[id(nb)]))
                if v.boundary or nb.boundary:
                    continue
                ring = _dual_area_vector_3d_p_ij(v, nb, HC)
                npt.assert_allclose(ring, ref[id(nb)], rtol=0, atol=1e-14)
                n += 1
        assert n > 500
