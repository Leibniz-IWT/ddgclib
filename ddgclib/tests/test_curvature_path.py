"""Method axis ``curvature_path`` (laneM, 2026-10-05).

One moving-mesh guard per surviving value and dimension: the force bound
by ``preset.replace(curvature_path=value)`` is evaluated after three
integrator steps on a reconnecting mesh plus laneI's jitter + Delaunay
rebuild (the interface sub-complex changes), and must equal a fresh
evaluation with every interface cache dropped.  A stale cache of the
kind audit 2026-09-25 F1/F2 found (apex map, coordinate map) cannot
return unnoticed.

Also locked here: the value ``'stokes'`` is gone (it was the cotangent
form to round-off, see the registry notes), and the 3D stencil is the
gradient of the interface triangle areas on a non-manifold fan too
(every triangle at an edge counts, not the first two).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cases_dynamic.oscillating_droplet.src._setup import (  # noqa: E402
    setup_oscillating_droplet,
)
from ddgclib import _curvatures_heron as heron  # noqa: E402
from ddgclib.methods import AXES, PRESETS, SolverMethods  # noqa: E402
from ddgclib.operators.multiphase_stress import (  # noqa: E402
    _interface_surface_tension,
)

AXIS = AXES['curvature_path']
VALUES = tuple(AXIS.keys())
PRESET = {2: 'oscillating_droplet_2D_bare_delaunay',
          3: 'oscillating_droplet_3D_delaunay'}
REFINE = {2: (2, 2), 3: (1, 1)}
INTERFACE_CACHES = ('_interface_edge_to_apex', '_interface_tri_ids')


def _methods(dim: int, value: str):
    m = PRESETS[PRESET[dim]]
    return m if value == AXIS.default else m.replace(curvature_path=value)


def _setup(dim: int, methods):
    ro, rd = REFINE[dim]
    return setup_oscillating_droplet(dim=dim, refinement_outer=ro,
                                     refinement_droplet=rd, methods=methods)


def _iface(HC) -> list:
    return [v for v in HC.V if getattr(v, 'is_interface', False)]


def _sub_ids(HC, dim: int) -> frozenset:
    """Interface triangles (3D) / edges (2D) by vertex identity; only
    valid right after a refresh (the keys are coordinates)."""
    x2v = {v.x: v for v in HC.V}
    faces = HC.interface_triangles if dim == 3 else HC.interface_edges
    return frozenset(frozenset(id(x2v[k]) for k in f) for f in faces)


def _jitter(HC, bV, dim: int, amp: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for v in list(HC.V):
        if v in bV:
            continue
        x = v.x_a.copy()
        x[:dim] += amp * rng.standard_normal(dim)
        HC.V.move(v, tuple(x))


def _dt(HC, dim: int, params) -> float:
    c_s = float(np.sqrt(params['K_d'] / params['rho_d']))
    dx_min = min(float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
                 for v in HC.V for nb in v.nn
                 if np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]) > 1e-15)
    return min(0.25 * dx_min / c_s,
               0.5 * np.sqrt(params['rho_d'] * dx_min ** 3 / params['gamma']))


def _drop_interface_caches(HC) -> None:
    for name in INTERFACE_CACHES:
        if hasattr(HC, name):
            delattr(HC, name)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_surviving_values(self):
        assert VALUES == ('integrated', 'csf_dual')
        assert AXIS.default == 'integrated'
        assert AXIS.option('csf_dual').status == 'measured-worse'

    def test_stokes_removed(self):
        with pytest.raises(ValueError):
            SolverMethods(dim=3, phases='multi', curvature_path='stokes')
        assert 'stokes' not in AXIS.keys()
        assert 'stokes' in AXIS.notes          # the record of what it was
        assert not hasattr(heron, 'integrated_hndA_i_interface')

    def test_unknown_value_raises_in_the_force(self):
        HC, bV, mps, *_ = setup_oscillating_droplet(
            dim=2, refinement_outer=1, refinement_droplet=1)
        v = _iface(HC)[0]
        with pytest.raises(ValueError, match='stokes'):
            _interface_surface_tension(v, 2, mps, HC=HC, curvature_path='stokes')
        assert not hasattr(HC, '_interface_x_to_v')


# ---------------------------------------------------------------------------
# moving mesh: the caches follow the mesh, per value and dimension
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('value', VALUES)
def test_force_follows_moving_mesh(dim, value):
    methods = _methods(dim, value)
    HC, bV, mps, bc_set, dudt_fn, _r, params = _setup(dim, methods)
    # the preset reached the force
    assert dudt_fn.keywords.get('curvature_path', AXIS.default) == value
    # warm the caches at the setup state and remember the sub-complex
    iface0 = _iface(HC)
    for v in iface0:
        dudt_fn(v)
    sub0 = _sub_ids(HC, dim)
    x0 = {id(v): v.x_a.copy() for v in HC.V}

    # three steps of the preset (Delaunay reconnection every step), then
    # laneI's jitter and a Delaunay rebuild
    methods.integrate(HC, bV, dudt_fn, dt=_dt(HC, dim, params), n_steps=3,
                      bc_set=bc_set, mps=mps)
    stale = getattr(HC, '_interface_edge_to_apex', None)
    _jitter(HC, bV, dim, 1e-3)
    retopo = methods.retopologize_fn(mps=mps)
    retopo(HC, bV, dim)

    moved = max(float(np.linalg.norm(v.x_a - x0[id(v)]))
                for v in HC.V if id(v) in x0)
    assert moved > 1e-4, 'fixture must move the mesh'
    sub1 = _sub_ids(HC, dim)
    assert sub1 != sub0, 'fixture must change the interface sub-complex'

    iface = _iface(HC)
    a_bound = {id(v): np.asarray(dudt_fn(v), dtype=float) for v in iface}
    _drop_interface_caches(HC)
    a_fresh = {id(v): np.asarray(dudt_fn(v), dtype=float) for v in iface}
    scale = max(float(np.linalg.norm(a)) for a in a_fresh.values())
    assert scale > 0
    for k, a in a_fresh.items():
        npt.assert_allclose(a_bound[k], a, rtol=0, atol=1e-12 * scale)

    if dim == 3:
        # power check: the apex map of the pre-jitter sub-complex is wrong
        # here (both values read it: the force, and the csf_dual magnitude)
        assert stale is not None
        HC._interface_edge_to_apex = stale
        a_stale = {id(v): np.asarray(dudt_fn(v), dtype=float) for v in iface}
        diff = max(float(np.linalg.norm(a_stale[k] - a_fresh[k]))
                   for k in a_fresh if k in a_stale)
        assert diff > 1e-3 * scale


# ---------------------------------------------------------------------------
# the 3D stencil is the area gradient, every triangle at an edge included
# ---------------------------------------------------------------------------
class _V:
    def __init__(self, x):
        self.x = tuple(float(c) for c in x)
        self.x_a = np.array(x, dtype=float)
        self.nn: set = set()


def _fan(x_v: np.ndarray):
    """One vertex ``v`` with four neighbours; the triangles (v,j,k),
    (v,j,l) and (v,j,m) share the edge (v,j): a non-manifold interface
    edge (three triangles), the other edges carry one triangle."""
    v = _V(x_v)
    j = _V((1.0, 0.0, 0.0))
    k = _V((0.3, 1.0, 0.1))
    l_ = _V((0.2, -0.2, 1.0))
    m = _V((0.4, -0.9, -0.6))
    verts = [v, j, k, l_, m]
    tris = [(v, j, k), (v, j, l_), (v, j, m)]
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (a, c)):
            p.nn.add(q)
            q.nn.add(p)

    class _HC:
        V = verts
        interface_triangles = {frozenset((a.x, b.x, c.x)) for a, b, c in tris}

    return v, verts, tris, _HC


def _total_area(x_v: np.ndarray) -> float:
    v, verts, tris, _HC = _fan(x_v)
    return sum(0.5 * float(np.linalg.norm(np.cross(b.x_a - a.x_a, c.x_a - a.x_a)))
               for a, b, c in tris)


def test_integrated_is_area_gradient_on_nonmanifold_fan():
    x_v = np.array([0.1, 0.05, -0.02])
    v, verts, tris, HC = _fan(x_v)
    HNdA, _C = heron.hndA_i_interface(v, set(verts), HC=HC)
    # the apex map holds all three apexes of edge (v, j)
    j = verts[1]
    assert len(HC._interface_edge_to_apex[frozenset((id(v), id(j)))]) == 3
    # -gamma * HNdA is the surface-tension force = -gamma * dA/dx_v
    h = 1e-6
    grad = np.array([
        (_total_area(x_v + h * e) - _total_area(x_v - h * e)) / (2 * h)
        for e in np.eye(3)])
    npt.assert_allclose(-HNdA[:3], -grad, rtol=1e-7, atol=1e-12)
    assert np.linalg.norm(grad) > 0.1

    # the three-triangle edge contributes three cotangent terms: dropping
    # one (the pre-laneM truncation to two apexes) misses the gradient
    v2, verts2, tris2, HC2 = _fan(x_v)
    HC2.interface_triangles = {frozenset((a.x, b.x, c.x)) for a, b, c in tris2[:2]}
    HNdA2, _ = heron.hndA_i_interface(v2, set(verts2), HC=HC2)
    assert np.linalg.norm(HNdA2 - HNdA) > 0.05 * np.linalg.norm(grad)
