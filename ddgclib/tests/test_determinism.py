"""A dynamic run is a pure function of its inputs (lane T, 2026-10-02).

Until lane T, 3D runs that read boundary dual faces gave other digits in
another interpreter, and in one interpreter after other runs:
``hyperct.ddg.compute_vd`` summed each boundary face barycentre in
``id()`` order of the face's vertices, i.e. in the order of their memory
addresses; the last bit of the barycentre moved, with it the hash of the
dual vertex and the iteration order of every set that holds it
(debugging_plan.md, protocol rule 8; the lane log is
docs_temp/debug_session/laneT-deterministic-iteration-order.md).

The cases, the state digest and the address scrambling are those of
``cases_dynamic/diagnose_determinism.py`` (``sweep all`` is the full
battery: every preset that reconnects, in 2D and 3D, and every value of
the connectivity axis).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic import diagnose_determinism as dd  # noqa: E402

# Hydrostatic 3D column, refinement 2 (189 vertices), remap arm
# (connectivity='delaunay_material' + conservative remap): a Delaunay
# rebuild every step, wall and free-surface half cells that read the
# boundary dual faces.  Measured on the library before lane T
# (``sweep hydro3d_remap,hydro3d --steps 4 --procs 8 --lib <HEAD export>``):
# 8 of 8 interpreters gave a digest of their own after these 4 steps (7
# distinct digests in 8 on the fixed-connectivity preset), and three runs
# in one interpreter gave three digests.
_CASE, _STEPS = 'hydro3d_remap', 4


def _fresh(variants: list[dict], case: str = _CASE,
           steps: int = _STEPS) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        return list(pool.map(lambda kw: dd.spawn(case, steps, **kw),
                             variants))


@pytest.fixture(scope='module')
def fresh_interpreters():
    """One plain interpreter and three whose small-object heap was
    fragmented first, so that the vertex objects sit at addresses that
    are not monotone in creation order (``--scramble``)."""
    return _fresh([dict(), dict(scramble_seed=3), dict(scramble_seed=5),
                   dict(scramble_seed=7)])


class TestReconnecting3DRunIsDeterministic:
    def test_fresh_interpreters_agree_to_the_bit(self, fresh_interpreters):
        digests = {r['variant']: r['digest'] for r in fresh_interpreters}
        assert len(set(digests.values())) == 1, digests
        scalars = [r['scalars'] for r in fresh_interpreters]
        assert all(s == scalars[0] for s in scalars)

    def test_twice_in_one_interpreter_after_an_unrelated_run(
            self, fresh_interpreters):
        """This interpreter has run whatever the test session ran
        before; the case is run, then an unrelated reconnecting 2D case,
        then the case again.  Both runs equal the fresh interpreters."""
        first = dd.run_case(_CASE, _STEPS)
        dd.run_case('droplet2d', 10)
        second = dd.run_case(_CASE, _STEPS)
        assert first['digest'] == second['digest']
        assert first['scalars'] == second['scalars']
        assert first['digest'] == fresh_interpreters[0]['digest']

    def test_scramble_changes_the_address_order(self):
        """The detector detects: after ``scramble`` the addresses of
        newly built vertices are no longer monotone in creation order
        (measured: 99 of 188 consecutive pairs ascending, 159 without)."""
        from hyperct import Complex

        dd.scramble(3, n=50_000)
        try:
            HC = Complex(3, domain=[(0.0, 1.0)] * 3)
            HC.triangulate()
            HC.refine_all()
            HC.refine_all()
            ids = [id(v) for v in HC.V]
            ascending = sum(b > a for a, b in zip(ids, ids[1:]))
            assert ascending < 0.75 * (len(ids) - 1)
        finally:
            dd._KEEP.clear()


class TestDensityDiffusionPairOrder:
    def test_mass_fluxes_do_not_depend_on_addresses(self):
        """``density_diffusion_step`` visited each pair from its
        lower-``id()`` vertex, so the order of the sums into a cell
        followed the memory addresses (this test fails on the library
        before lane T: the returned ``sum_dm`` differs between the
        builds).  The same Delaunay mesh is built four times with the
        heap fragmented in between."""
        from hyperct import Complex
        from hyperct.ddg import (
            boundary_from_simplices, compute_vd, connect_and_cache_simplices,
        )
        from ddgclib.operators.stabilisation import density_diffusion_step
        from ddgclib.operators.stress import cache_dual_volumes

        coords = np.random.default_rng(3).uniform(0.0, 1.0, size=(120, 2))

        def build():
            HC = Complex(2)
            verts = [HC.V[tuple(p)] for p in coords]
            connect_and_cache_simplices(HC, verts, 2, coords=coords)
            dV = boundary_from_simplices(HC, 2)
            for v in HC.V:
                v.boundary = v in dV
            compute_vd(HC, method='barycentric')
            cache_dual_volumes(HC, 2)
            rng = np.random.default_rng(9)
            for v in HC.V:
                v.m = v.dual_vol * (1000.0 + 50.0 * rng.uniform(-1, 1))
            interior = [v for v in HC.V if v not in dV]
            sums = [density_diffusion_step(HC, interior, 0.1, 4.0, 1e-3,
                                           dim=2)['sum_dm'].hex()
                    for _ in range(5)]
            return sums, [float(v.m).hex() for v in HC.V]

        try:
            results = []
            for seed in (0, 1, 2, 3):
                if seed:
                    dd.scramble(seed, n=40_000)
                results.append(build())
        finally:
            dd._KEEP.clear()
        assert all(r == results[0] for r in results[1:])


def _per_tetrahedron_area(v_i, v_j, HC) -> np.ndarray:
    """The barycentric dual face of the primal edge (i, j), independent
    of ``ddgclib.operators.stress``: one quadrilateral (edge midpoint,
    face barycentre, tetrahedron barycentre, face barycentre) per
    tetrahedron on the edge; the vector area of a quadrilateral is half
    the cross product of its diagonals.  Oriented along ``x_j - x_i``."""
    x_i, x_j = v_i.x_a, v_j.x_a
    mid = 0.5 * (x_i + x_j)
    area = np.zeros(3)
    for s in HC._simplices:
        if v_i in s and v_j in s:
            a, b = (w for w in s if w is not v_i and w is not v_j)
            quad = 0.5 * np.cross((x_i + x_j + a.x_a + b.x_a) / 4.0 - mid,
                                  (b.x_a - a.x_a) / 3.0)
            area += quad if quad @ (x_j - x_i) > 0 else -quad
    return area


class TestFreeSurfaceEdgeAreaIsDecidedByATie:
    """What determinism does not give.  The run is the same in every
    interpreter, but on the 3D free surface it is still decided by
    round-off: ``_dual_area_vector_3d_p_ij`` puts 'the nearest face
    barycentre' between two consecutive dual vertices of the ring, and
    for a boundary edge the ring also holds the edge midpoint and the two
    boundary face barycentres.  Between the midpoint and a boundary face
    barycentre the nearest candidate is that barycentre itself (a
    duplicate point, harmless) or an interior face barycentre (a spurious
    polygon vertex); on the builder lattice the two are at the same
    distance, so the last bit of the positions decides, and the area
    vector of a free-surface edge changes by up to 37 %.  Found with
    ``diagnose_determinism.py sweep hydro3d --perturb 1e-15``: a 1e-15
    shift of the interior vertices moves the kinetic energy of the
    refinement 2 column by 2e-06 after 150 steps on FIXED connectivity
    (at refinement 1, where no edge joins two free-surface vertices, by
    6e-12 after 370 steps).  Not fixed in lane T: it changes the force on
    every 3D free surface and with it the pins of lane P.

    Lane Q (2026-10-05): the exact sources of the axis ``edge_area_source``
    (``'p_ij'`` per edge, ``'p_ij_simplex'`` cached) read the polygon from
    the tetrahedra and are exact on the hull edges too; the ring walk is
    the registered value ``'p_ij_ring'`` (status broken) and still the
    default of a mesh no retopology has tagged, which the first test
    keeps locking."""

    @pytest.fixture(scope='class')
    def column(self):
        from cases_dynamic.Hydrostatic_column.src._column import (
            CASES, build_column,
        )
        kw = CASES['hydrostatic_3D']
        return build_column(3, 2, H=kw['H'], side_walls=kw['side_walls'],
                            ic='drop')

    @staticmethod
    def _errors(col, on_surface: bool, area=None) -> list[float]:
        from ddgclib.operators.stress import dual_area_vector
        HC = col.HC
        if area is None:
            area = lambda v, nb: dual_area_vector(v, nb, HC, 3)  # noqa: E731
        errors = []
        for v in HC.V:
            if v in col.bV:
                continue
            for nb in v.nn:
                both = bool(v.boundary) and bool(nb.boundary)
                if both != on_surface:
                    continue
                ref = _per_tetrahedron_area(v, nb, HC)
                errors.append(float(np.linalg.norm(area(v, nb) - ref)
                                    / np.linalg.norm(ref)))
        return errors

    def test_edges_with_an_interior_endpoint_are_exact(self, column):
        errors = self._errors(column, on_surface=False)
        assert len(errors) == 1127
        assert max(errors) < 1e-14                    # measured 8.3e-16

    def test_ring_walk_is_off_on_the_boundary_edges(self, column):
        """The legacy construction (the default of an untagged mesh and
        the value 'p_ij_ring'): 30 of the 56 boundary edges at the 9
        free-surface vertices off by up to 0.373 (lane T, section 6)."""
        from ddgclib.operators.stress import dual_area_vector
        errors = self._errors(column, on_surface=True)
        ring = self._errors(column, on_surface=True, area=lambda v, nb:
                            dual_area_vector(v, nb, column.HC, 3,
                                             source='p_ij_ring'))
        assert errors == ring
        assert len(errors) == 56
        assert sum(e > 1e-12 for e in errors) == 30
        assert 0.37 < max(errors) < 0.38

    def test_exact_sources_are_exact_on_the_boundary_edges(self, column):
        """Lane Q: the per-edge polygon read from the tetrahedra
        ('p_ij') and the vectorised cache ('p_ij_simplex') against the
        per-tetrahedron sum, on the 56 free-surface edges."""
        from hyperct.ddg import simplex_dual_face_areas
        from ddgclib.operators.stress import dual_area_vector
        HC = column.HC
        per_edge = self._errors(column, on_surface=True, area=lambda v, nb:
                                dual_area_vector(v, nb, HC, 3, source='p_ij'))
        cache = simplex_dual_face_areas(HC, 3)
        cached = self._errors(column, on_surface=True,
                              area=lambda v, nb: cache[id(v)][id(nb)])
        assert len(per_edge) == len(cached) == 56
        assert max(per_edge) < 1e-14
        assert max(cached) < 1e-14


@pytest.mark.slow
class TestEveryProcessDependentArmIsDeterministic:
    """Every arm that gave more than one digest on the library before
    lane T (rule 8 was written for the first three), plus the bare 3D
    droplet rebuild: plain, after two other runs in the same
    interpreter, and scrambled."""

    @pytest.mark.parametrize('case, steps', [
        ('hp3d_centred', 300),          # lane H: l2 0.0811, 0.0821 or 0.1107
        ('hydro3d_remap', 40),          # lane P: two digits after 10 t_ac
        ('hydro3d', 40),                # lane P: 4e-09 on fixed connectivity
        ('shearing3d', 5),              # periodic 3D: 2 digests in 6
        ('droplet2d_adaptive', 60),     # adaptive 2D: 6 digests in 6
        ('droplet3d_delaunay', 20),
    ])
    def test_digest_is_the_same_in_every_interpreter(self, case, steps):
        rows = _fresh([dict(), dict(pre='droplet2d,hp3d_centred'),
                       dict(scramble_seed=3), dict(scramble_seed=5)],
                      case, steps)
        digests = {r['variant']: r['digest'] for r in rows}
        assert len(set(digests.values())) == 1, digests
