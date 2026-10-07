"""Domain builders return their mesh with the top-simplex cache (laneS).

Before 2026-10-01 a builder mesh had ``HC._simplices is None``, so the
first ``cache_dual_volumes`` call fell back to ``dual_cell_area_2d`` (2D)
or the ``v_star`` fan walk (3D).  Those volumes did not tile the domain
(rectangle total 0.96875 at refinement 2, box 0.9167 at refinement 1) and
in 2D a moving free-surface vertex was credited with a quarter of its own
volume change, which made Hydrostatic_2D grow exponentially from
round-off (docs_temp/debug_session/laneK-single-phase-eos-instability.md
section 4).  The builders now cache the simplices of the connectivity
they built, so setup volumes are the exact barycentric ones,
``Vol_i = sum_{T contains i} |T| / (dim + 1)``, the same source every
later retopology uses.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from hyperct.ddg import compute_vd, dual_cell_area_2d, simplex_dual_volumes

from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry.domains import (
    annulus, ball, box, cylinder_volume, disk, l_shape, periodic_box,
    periodic_rectangle, pipe, rectangle,
)
from ddgclib.initial_conditions import DualVolumeMass, ZeroVelocity
from ddgclib.methods import SolverMethods, effective_methods
from ddgclib.operators.stress import cache_dual_volumes

# name -> (builder call, exact measure or None for a curved boundary)
BUILDERS_2D = {
    'rectangle': (lambda: rectangle(L=2.0, h=1.0, refinement=3), 2.0),
    'l_shape': (lambda: l_shape(L=2.0, h=1.0, notch_L=1.0, notch_h=0.5,
                                refinement=3), 1.5),
    'disk': (lambda: disk(R=1.0, refinement=3), None),
    'annulus': (lambda: annulus(R_outer=1.0, R_inner=0.3, refinement=3),
                None),
}
BUILDERS_3D = {
    'box': (lambda: box(Lx=2.0, Ly=1.0, Lz=1.0, refinement=2), 2.0),
    'cylinder_volume': (lambda: cylinder_volume(R=0.5, L=2.0, refinement=2),
                        None),
    'pipe': (lambda: pipe(R=0.5, L=3.0, refinement=1), None),
    'ball': (lambda: ball(R=1.0, refinement=2), None),
}
BUILDERS = {**BUILDERS_2D, **BUILDERS_3D}
PERIODIC = {
    'periodic_rectangle': lambda: periodic_rectangle(
        L=1.0, h=1.0, refinement=2, periodic_axes=[0]),
    'periodic_box': lambda: periodic_box(
        Lx=1.0, Ly=1.0, Lz=1.0, refinement=1, periodic_axes=[0]),
}


def _mesh_edges(HC) -> set:
    return {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}


def _enclosed_measure(HC, dim: int) -> float:
    """Area / volume enclosed by the mesh boundary, from the divergence
    theorem over the faces that belong to exactly one top simplex.  It
    does not depend on how the interior is cut into simplices."""
    owners: dict = {}
    for s in HC._simplices:
        for face in combinations(s, dim):
            owners.setdefault(frozenset(id(v) for v in face), []).append(
                (face, s))
    total = 0.0
    for lst in owners.values():
        if len(lst) != 1:
            continue
        face, s = lst[0]
        opposite = next(v for v in s if all(v is not w for w in face))
        pts = np.array([v.x_a[:dim] for v in face], dtype=float)
        centre = pts.mean(axis=0)
        if dim == 2:
            t = pts[1] - pts[0]
            normal = np.array([t[1], -t[0]])            # |normal| = length
        else:
            normal = 0.5 * np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(normal, opposite.x_a[:dim] - centre) > 0.0:
            normal = -normal
        total += float(np.dot(centre, normal)) / dim
    return total


def _setup_volumes(result) -> None:
    """What every case setup does after calling a builder."""
    compute_vd(result.HC, method='barycentric')
    cache_dual_volumes(result.HC, result.dim)


@pytest.mark.parametrize('name', list(BUILDERS) + list(PERIODIC))
def test_builder_caches_the_simplices_of_its_own_connectivity(name):
    result = (BUILDERS[name][0] if name in BUILDERS else PERIODIC[name])()
    HC, dim = result.HC, result.dim
    edges_built = _mesh_edges(HC)
    assert HC._simplices, f"{name}: no simplex cache"
    in_mesh = {id(v) for v in HC.V}
    for s in HC._simplices:
        assert len(s) == dim + 1 == len({id(v) for v in s})
        assert all(id(v) in in_mesh for v in s)
    cached_edges = {frozenset((id(a), id(b)))
                    for s in HC._simplices for a, b in combinations(s, 2)}
    # nothing re-triangulated: the cache spans exactly the builder edges
    assert cached_edges == edges_built
    # every face belongs to one (boundary) or two (interior) simplices
    n_owners: dict = {}
    for s in HC._simplices:
        for face in combinations(s, dim):
            key = frozenset(id(v) for v in face)
            n_owners[key] = n_owners.get(key, 0) + 1
    assert set(n_owners.values()) == {1, 2}


@pytest.mark.parametrize('name', list(BUILDERS))
def test_setup_dual_volumes_tile_the_domain(name):
    build, exact = BUILDERS[name]
    result = build()
    HC, dim = result.HC, result.dim
    _setup_volumes(result)
    assert effective_methods(HC, dim)['dual_volume'] == 'simplex_exact'
    vols = np.array([v.dual_vol for v in HC.V])
    assert vols.min() > 0.0
    total = float(vols.sum())
    assert total == pytest.approx(_enclosed_measure(HC, dim), rel=1e-12)
    if exact is not None:
        assert total == pytest.approx(exact, rel=1e-12)
    else:
        # polygon / polyhedron inscribed in the curved boundary
        assert 0.85 * result.metadata['volume'] < total
        assert total < result.metadata['volume']
    ref = simplex_dual_volumes(HC, dim)
    assert all(v.dual_vol == ref[v] for v in HC.V)


@pytest.mark.parametrize('name', list(BUILDERS_2D))
def test_2d_volumes_agree_with_the_dual_polygon_of_every_vertex(name):
    """Two independent constructions of the same cell: the simplex rule
    and the shoelace area of the dual polygon (corner, re-entrant corner
    and curved-boundary vertices included)."""
    result = BUILDERS_2D[name][0]()
    _setup_volumes(result)
    for v in result.HC.V:
        assert v.dual_vol == pytest.approx(
            dual_cell_area_2d(v, include_edge_midpoints=True), rel=1e-11), \
            f"{name}: vertex {v.x}"


def test_rectangle_corner_cells_are_not_undercounted():
    """The four corner cells were 4x too small on the fallback path."""
    result = rectangle(L=1.0, h=1.0, refinement=2)
    _setup_volumes(result)
    corners = [v for v in result.HC.V
               if v.x_a[0] in (0.0, 1.0) and v.x_a[1] in (0.0, 1.0)]
    assert len(corners) == 4
    for v in corners:
        # one right triangle with legs 1/4, a third of it
        assert v.dual_vol == pytest.approx(0.25 * 0.25 / 2.0 / 3.0, rel=1e-12)


@pytest.mark.parametrize('skip_triangulation', [True, False])
def test_no_volume_jump_between_setup_and_first_retopology(skip_triangulation):
    """Setup and retopology read the same volume source: the total is
    unchanged by the first retopology (it was 0.96875 -> 1.0), and with
    the connectivity kept (dual_only) every single volume is."""
    result = rectangle(L=1.0, h=1.0, refinement=2)
    HC, bV = result.HC, result.bV
    _setup_volumes(result)
    before = {v: v.dual_vol for v in HC.V}
    _retopologize(HC, bV, 2, skip_triangulation=skip_triangulation)
    assert sum(v.dual_vol for v in HC.V) == pytest.approx(
        sum(before.values()), rel=1e-13)
    assert sum(before.values()) == pytest.approx(1.0, rel=1e-13)
    if skip_triangulation:
        assert all(v.dual_vol == before[v] for v in HC.V)


def test_already_cached_mesh_is_left_alone():
    """The Delaunay-built droplet meshes keep their own cache."""
    from ddgclib.geometry.domains import droplet_in_box_2d
    from ddgclib.geometry.domains._result import DomainResult
    result = droplet_in_box_2d(R=0.01, L=0.05, refinement_outer=1,
                               refinement_droplet=1)
    cache = result.HC._simplices
    assert cache is not None
    again = DomainResult(HC=result.HC, bV=result.bV, dim=2)
    assert again.HC._simplices is cache


def test_free_surface_box_is_stable_through_the_library_integrator():
    """Closed box with a free top, no gravity, 1e-6 m/s seed.

    This is the Hydrostatic_2D mechanism of laneK section 4 without the
    hand-rolled loop: fixed connectivity (``dual_only``), EOS pressure,
    free-surface vertices that move.  On the old fallback volumes the
    seed grew exponentially (blow-up after about 14 acoustic times); on
    a builder mesh without the cache the library path could not even
    run it (legacy compute_vd needs every hull vertex tagged).  Measured
    with the cache: 1.538e-06 -> 1.989e-09 m/s over 8 acoustic times.
    """
    rho0, c0 = 1000.0, 31.32
    result = rectangle(L=1.0, h=1.0, refinement=2)
    HC, groups = result.HC, result.boundary_groups
    bV = set(groups['bottom_wall']) | set(groups['inlet']) | set(groups['outlet'])
    _setup_volumes(result)
    ZeroVelocity(dim=2).apply(HC, bV)
    DualVolumeMass(rho=rho0).apply(HC, bV)
    rng = np.random.default_rng(0)
    for v in sorted(HC.V, key=lambda w: w.x):
        if v not in bV:
            v.u[:2] = 1e-6 * rng.standard_normal(2)
    eos = TaitMurnaghan(rho0=rho0, P0=0.0, K=rho0 * c0**2, n=1.0,
                        rho_clip=(0.5, 2.0))
    edges = [float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
             for v in HC.V for nb in v.nn]
    methods = SolverMethods(dim=2, connectivity='dual_only')
    dudt_fn = methods.dudt_fn(HC, mu=0.5 * rho0 * c0 * float(np.mean(edges)),
                              pressure_model=eos)
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=2), bV)
    dt = 0.25 * min(edges) / c0
    umax: list[float] = []

    def callback(step, t, HC_cb, bV_cb=None, diagnostics=None):
        umax.append(max(float(np.linalg.norm(v.u[:2])) for v in HC_cb.V))

    methods.integrate(HC, bV, dudt_fn, dt=dt,
                      n_steps=int(round(8.0 / c0 / dt)), bc_set=bc_set,
                      callback=callback, pressure_model=eos)
    assert len(bV) == 13                     # the nine top vertices are free
    assert max(umax) <= umax[0] * (1.0 + 1e-9)
    assert umax[-1] < 1e-2 * umax[0]
