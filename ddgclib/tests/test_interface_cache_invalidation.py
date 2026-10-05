"""Interface curvature caches must follow the interface sub-complex.

Regression for audit 2026-09-25 F1 (multiphase T1):
``HC._interface_edge_to_apex`` (3D 'integrated' surface tension) was
built once and never cleared, while ``extract_interface`` rebuilds
``HC.interface_triangles`` on every ``mps.refresh``.  After a 3D Delaunay
retopology the stale apex map gave a surface-tension force wrong by
~100 % of its magnitude.

The fix lives in :func:`ddgclib.geometry._interface_subcomplex.extract_interface`:
the apex map is dropped when the interface triangle set changes by vertex
identity and KEPT under frozen connectivity (so the 3D dual_only pins
stay bit-identical).  The coordinate-keyed map of the former 'stokes'
path (audit F2) went with that path in laneM (2026-10-05); the per-value
moving-mesh guards are in ``test_curvature_path.py``.
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
from ddgclib.operators.multiphase_stress import (  # noqa: E402
    _interface_surface_tension,
)


def _tri_ids(HC) -> set[frozenset]:
    x2v = {v.x: v for v in HC.V}
    return {frozenset(id(x2v[k]) for k in t) for t in HC.interface_triangles}


def _fst(HC, mps) -> dict[int, np.ndarray]:
    return {id(v): np.asarray(_interface_surface_tension(v, 3, mps, HC=HC))
            for v in HC.V if getattr(v, 'is_interface', False)}


def _jitter(HC, bV, amp: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for v in list(HC.V):
        if v in bV:
            continue
        x = v.x_a.copy()
        x[:3] += amp * rng.standard_normal(3)
        HC.V.move(v, tuple(x))


@pytest.fixture
def droplet3d():
    return setup_oscillating_droplet(dim=3, refinement_outer=1,
                                     refinement_droplet=1)


def test_apex_cache_rebuilt_after_delaunay_retopology(droplet3d):
    HC, bV, mps, _bc, _dudt, retopo, _p = droplet3d
    _fst(HC, mps)
    stale = HC._interface_edge_to_apex
    tris0 = _tri_ids(HC)

    _jitter(HC, bV, 1e-3)
    retopo(HC, bV, 3)  # full Delaunay + mps.refresh -> extract_interface
    tris1 = _tri_ids(HC)
    assert tris0 != tris1, "fixture must change the interface topology"
    assert getattr(HC, '_interface_edge_to_apex', None) is None

    F_after = _fst(HC, mps)
    assert HC._interface_edge_to_apex is not stale
    # Fresh reference: force a rebuild from the current triangles.
    del HC._interface_edge_to_apex
    F_fresh = _fst(HC, mps)
    assert F_after.keys() == F_fresh.keys()
    scale = max(np.linalg.norm(f) for f in F_fresh.values())
    for k, f in F_fresh.items():
        npt.assert_allclose(F_after[k], f, rtol=0, atol=1e-12 * scale)

    # The pre-fix stale map really was wrong on this mesh (guards the
    # fixture's power to detect a regression).
    HC._interface_edge_to_apex = stale
    F_stale = _fst(HC, mps)
    diff = max(np.linalg.norm(F_stale[k] - F_fresh[k])
               for k in F_fresh if k in F_stale)
    assert diff > 0.1 * scale


def test_apex_cache_kept_under_frozen_connectivity(droplet3d):
    """dual_only path: same triangles by identity -> same cache object."""
    HC, bV, mps, _bc, _dudt, _retopo, _p = droplet3d
    _fst(HC, mps)
    cache = HC._interface_edge_to_apex
    tris0 = _tri_ids(HC)

    _jitter(HC, bV, 1e-4)
    mps.refresh(HC, 3, reset_mass=False)
    assert _tri_ids(HC) == tris0
    assert HC._interface_edge_to_apex is cache

    F_kept = _fst(HC, mps)
    del HC._interface_edge_to_apex
    F_fresh = _fst(HC, mps)
    scale = max(np.linalg.norm(f) for f in F_fresh.values())
    for k, f in F_fresh.items():
        npt.assert_allclose(F_kept[k], f, rtol=0, atol=1e-12 * scale)
