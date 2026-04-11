"""Integration tests for interface-preserving adaptive remeshing.

Covers the end-to-end pipeline: running ``_retopologize`` and the
dynamic integrators with ``remesh_mode='adaptive'`` on 2D meshes, with
a focus on preserving sharp ``v.phase`` interfaces.  Includes coverage
of:

- Single-phase ``_retopologize`` (backward compatibility and adaptive
  dispatch)
- The multiphase ``_retopologize_multiphase`` wrapper, which is the
  path actually exercised by Cube2droplet and other multiphase cases
- The ``retopologize_fn=...`` callable-injection path used by case
  studies that close over a ``MultiphaseSystem``
- 3D rejection surfacing a clear ``NotImplementedError``
"""
import numpy as np
import pytest

from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize,
    _retopologize_multiphase,
    symplectic_euler,
)
from hyperct._complex import Complex
from hyperct.remesh import adaptive_remesh, is_interface_edge


def _square_2d_grid(n=4, L=1.0):
    """Build a 2D grid on [0,L]x[0,L] with n*n vertices and triangulate
    each quad along the bottom-left / top-right diagonal.  Returns
    ``(HC, bV)``.
    """
    HC = Complex(2)
    verts = {}
    for i in range(n):
        for j in range(n):
            x = (i * L / (n - 1), j * L / (n - 1))
            verts[(i, j)] = HC.V[x]

    for i in range(n - 1):
        for j in range(n - 1):
            a = verts[(i, j)]
            b = verts[(i + 1, j)]
            c = verts[(i, j + 1)]
            d = verts[(i + 1, j + 1)]
            a.connect(b)
            a.connect(d)
            b.connect(d)
            a.connect(c)
            c.connect(d)

    bV = set()
    for (i, j), v in verts.items():
        v.boundary = (i == 0 or j == 0 or i == n - 1 or j == n - 1)
        if v.boundary:
            bV.add(v)
    return HC, bV, verts


class TestRetopologizeAdaptive2D:
    def test_adaptive_mode_preserves_vertex_count(self):
        HC, bV, verts = _square_2d_grid(n=4)
        n_before = len(list(HC.V))
        # Use explicit L thresholds wide enough to include the quad
        # diagonal edge (length sqrt(2)/3 ≈ 0.471) without splitting.
        _retopologize(
            HC, bV, dim=2,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 1, 'smooth_iterations': 0,
            },
        )
        assert len(list(HC.V)) == n_before

    def test_adaptive_mode_recomputes_duals(self):
        HC, bV, verts = _square_2d_grid(n=4)
        _retopologize(
            HC, bV, dim=2,
            remesh_mode='adaptive',
            remesh_kwargs={'max_iterations': 1, 'smooth_iterations': 0},
        )
        # After the call, interior vertices should have dual cells.
        for v in HC.V:
            if not v.boundary:
                assert hasattr(v, 'vd')
                assert v.dual_vol >= 0.0

    def test_adaptive_mode_does_not_cross_phase(self):
        """The canonical interface-preservation test.

        Checks three invariants on a vertical two-phase partition of a
        5x5 grid:

        1. Bulk-interior vertices (not on the original interface) stay
           bulk-interior — no new interface vertices are manufactured.
        2. At least one interface edge still exists after the remesh
           (the interface isn't trivially destroyed).
        3. No edge between same-phase vertices becomes a cross-phase
           edge (a stronger statement than #1, ruling out connectivity
           rewires that happen to leave vertex labels untouched).
        """
        HC, bV, verts = _square_2d_grid(n=5)
        for (i, j), v in verts.items():
            v.phase = 0 if i < 2 else 1

        def interface_vertex_ids(hc):
            ids = set()
            for v in hc.V:
                for nb in v.nn:
                    if getattr(v, 'phase', None) != getattr(nb, 'phase', None):
                        ids.add(id(v))
                        break
            return ids

        def same_phase_pair_ids(hc):
            """Frozensets of (id, id) for edges where both ends share a phase."""
            pairs = set()
            for v in hc.V:
                for nb in v.nn:
                    if getattr(v, 'phase', None) == getattr(nb, 'phase', None):
                        pairs.add(frozenset((id(v), id(nb))))
            return pairs

        def count_interface_edges(hc):
            n = 0
            seen = set()
            for v in hc.V:
                for nb in v.nn:
                    key = frozenset((id(v), id(nb)))
                    if key in seen:
                        continue
                    seen.add(key)
                    if is_interface_edge(v, nb):
                        n += 1
            return n

        iface_before = interface_vertex_ids(HC)
        bulk_before = {id(v) for v in HC.V} - iface_before
        same_phase_before = same_phase_pair_ids(HC)
        n_iface_before = count_interface_edges(HC)
        assert n_iface_before > 0, "test precondition: interface must exist"

        _retopologize(
            HC, bV, dim=2,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 2, 'smooth_iterations': 0,
            },
        )

        # 1. Bulk-interior vertices still bulk-interior
        iface_after = interface_vertex_ids(HC)
        still_exist = {id(v) for v in HC.V}
        newly_interfaced = (iface_after & bulk_before) & still_exist
        assert not newly_interfaced, (
            "Adaptive remesh turned bulk vertices into interface vertices: "
            f"{len(newly_interfaced)} such vertices"
        )

        # 2. Interface still present (wasn't wiped out by flips/collapses)
        assert count_interface_edges(HC) > 0, (
            "Adaptive remesh destroyed the entire phase interface"
        )

        # 3. No previously same-phase edge is now cross-phase.  Build a
        # dict {pair -> (phase_i, phase_j)} for edges that still exist
        # and whose endpoint-ids match a same-phase edge before remesh.
        for v in HC.V:
            for nb in v.nn:
                key = frozenset((id(v), id(nb)))
                if key in same_phase_before:
                    assert getattr(v, 'phase', None) == getattr(nb, 'phase', None), (
                        "A same-phase edge became a cross-phase edge under remesh"
                    )

    def test_adaptive_mode_smooths_interior(self):
        """A grid with a single perturbed interior vertex should see
        that vertex migrate back toward the grid centre under
        Laplacian smoothing.
        """
        HC, bV, verts = _square_2d_grid(n=5)
        # Perturb the centre vertex away from (0.5, 0.5)
        v_mid = verts[(2, 2)]
        HC.V.move(v_mid, (0.5 + 0.08, 0.5 + 0.08))
        initial_dist = np.hypot(0.08, 0.08)

        _retopologize(
            HC, bV, dim=2,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 3, 'smooth_iterations': 3,
                'smooth_relax': 0.5,
            },
        )

        # Find the closest vertex to (0.5, 0.5) — should be the
        # (now-relaxed) original centre vertex.
        best = min(
            (v for v in HC.V if not v.boundary),
            key=lambda v: (v.x[0] - 0.5) ** 2 + (v.x[1] - 0.5) ** 2,
        )
        best_dist = np.hypot(best.x[0] - 0.5, best.x[1] - 0.5)
        assert best_dist < initial_dist, (
            f"Smoothing did not reduce distance: {best_dist:.4f} >= "
            f"{initial_dist:.4f}"
        )


class TestIntegratorWithAdaptiveRemesh:
    def test_symplectic_euler_runs_with_adaptive(self):
        """End-to-end smoke test: run a few steps of symplectic_euler
        with remesh_mode='adaptive' and confirm the integrator finishes
        without error.
        """
        HC, bV, verts = _square_2d_grid(n=4)
        # Give interior vertices a small non-zero velocity.
        for v in HC.V:
            v.u = np.array([0.0, 0.0])
            v.p = 0.0
            v.m = 1.0
            v.phase = 0 if v.x[0] < 0.5 else 1

        def zero_dudt(v, **kwargs):
            return np.zeros(2)

        t = symplectic_euler(
            HC, bV, zero_dudt, dt=1e-3, n_steps=3, dim=2,
            remesh_mode='adaptive',
            remesh_kwargs={'max_iterations': 1, 'smooth_iterations': 0},
        )
        assert t == pytest.approx(3e-3)
        # Still should have vertices with valid duals after the run.
        assert any(not v.boundary for v in HC.V)

    def test_delaunay_mode_still_default(self):
        """Backward compatibility: the default remesh_mode should be
        'delaunay' and behave identically to the pre-adaptive code
        path (no crash, valid duals)."""
        HC, bV, verts = _square_2d_grid(n=4)
        _retopologize(HC, bV, dim=2)  # No remesh_mode specified.
        for v in HC.V:
            if not v.boundary:
                assert hasattr(v, 'vd')


class TestMultiphaseRetopologize:
    """Coverage for the multiphase path — the Cube2droplet scenario.

    These tests hit two call chains:
      1. ``_retopologize_multiphase`` called directly.
      2. A 3-arg / kwargs-accepting closure passed as ``retopologize_fn``
         to an integrator, exactly like ``cases_dynamic/Cube2droplet/src/_setup.py``.
    """

    def _two_phase_grid(self, n=5):
        HC, bV, verts = _square_2d_grid(n=n)
        for (i, j), v in verts.items():
            v.phase = 0 if i < n // 2 else 1
            v.u = np.array([0.0, 0.0])
            v.p = 0.0
            v.m = 1.0
        return HC, bV, verts

    def test_retopologize_multiphase_accepts_remesh_kwargs(self):
        """_retopologize_multiphase must accept and forward remesh_mode
        so the multiphase Cube2droplet pipeline can opt into adaptive."""
        HC, bV, verts = self._two_phase_grid()
        _retopologize_multiphase(
            HC, bV, dim=2, mps=None,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 1, 'smooth_iterations': 0,
            },
        )
        # Interior vertices still have valid duals
        for v in HC.V:
            if not v.boundary:
                assert hasattr(v, 'vd')

    def test_kwargs_closure_receives_forwarded_kwargs(self):
        """A user closure that accepts **kwargs (as in Cube2droplet's
        `retopo_fn`) should receive `remesh_mode` and `remesh_kwargs`
        when they are passed to the integrator."""
        HC, bV, verts = self._two_phase_grid()
        seen = {}

        def retopo_fn(HC_, bV_, dim_, **kwargs):
            seen.update(kwargs)
            _retopologize_multiphase(HC_, bV_, dim_, mps=None, **kwargs)

        def zero_dudt(v, **kw):
            return np.zeros(2)

        symplectic_euler(
            HC, bV, zero_dudt, dt=1e-4, n_steps=1, dim=2,
            retopologize_fn=retopo_fn,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 1, 'smooth_iterations': 0,
            },
        )

        assert seen.get('remesh_mode') == 'adaptive', (
            "remesh_mode was not forwarded to the user closure"
        )
        rk = seen.get('remesh_kwargs')
        assert isinstance(rk, dict) and rk.get('L_max') == 0.9, (
            "remesh_kwargs was not forwarded verbatim"
        )

    def test_legacy_3arg_closure_still_works(self):
        """A pre-existing 3-arg closure with no **kwargs must keep
        working — the dispatch uses inspect.signature to detect and
        skip the new kwargs, preserving backward compatibility with
        case studies that haven't been updated.
        """
        HC, bV, verts = self._two_phase_grid()
        called = {'n': 0}

        def legacy_retopo(HC_, bV_, dim_):
            called['n'] += 1
            _retopologize_multiphase(HC_, bV_, dim_, mps=None)

        def zero_dudt(v, **kw):
            return np.zeros(2)

        symplectic_euler(
            HC, bV, zero_dudt, dt=1e-4, n_steps=2, dim=2,
            retopologize_fn=legacy_retopo,
            # Note: user passes remesh_mode but the legacy closure
            # should NOT crash — it just ignores the kwargs.
            remesh_mode='adaptive',
            remesh_kwargs={'L_max': 0.9},
        )

        assert called['n'] == 2

    def test_adaptive_multiphase_preserves_interface(self):
        """End-to-end check on the multiphase path: running a few
        integrator steps with the Cube2droplet-style closure and
        ``remesh_mode='adaptive'`` should NOT turn bulk-phase vertices
        into interface vertices, and should NOT introduce cross-phase
        edges where none existed before.
        """
        HC, bV, verts = self._two_phase_grid(n=6)

        # Record same-phase edges before the run
        same_phase_before = set()
        for v in HC.V:
            for nb in v.nn:
                if getattr(v, 'phase', None) == getattr(nb, 'phase', None):
                    same_phase_before.add(frozenset((id(v), id(nb))))

        def retopo_fn(HC_, bV_, dim_, **kwargs):
            _retopologize_multiphase(HC_, bV_, dim_, mps=None, **kwargs)

        def zero_dudt(v, **kw):
            return np.zeros(2)

        symplectic_euler(
            HC, bV, zero_dudt, dt=1e-4, n_steps=3, dim=2,
            retopologize_fn=retopo_fn,
            remesh_mode='adaptive',
            remesh_kwargs={
                'L_min': 0.05, 'L_max': 0.9,
                'max_iterations': 2, 'smooth_iterations': 0,
            },
        )

        # No previously-same-phase edge may have become cross-phase
        for v in HC.V:
            for nb in v.nn:
                key = frozenset((id(v), id(nb)))
                if key in same_phase_before:
                    assert getattr(v, 'phase') == getattr(nb, 'phase'), (
                        "Adaptive multiphase pipeline broke a same-phase edge"
                    )


class TestAdaptive3DRejection:
    """Passing ``remesh_mode='adaptive'`` with ``dim=3`` should surface
    a clear NotImplementedError — not silently fall back to Delaunay."""

    def test_direct_driver_rejects_3d(self):
        HC = Complex(3)
        HC.V[(0.0, 0.0, 0.0)]
        with pytest.raises(NotImplementedError):
            adaptive_remesh(HC, dim=3)

    def test_retopologize_raises_on_3d_adaptive(self):
        """Even at the ddgclib wrapper level, a 3D adaptive request
        should propagate the NotImplementedError from the driver —
        no silent fallback to Delaunay."""
        HC = Complex(3)
        # Build the tiniest 3D mesh that has > dim+1 vertices
        for x in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                  (1.0, 1.0, 1.0)]:
            HC.V[x]
        for v in HC.V:
            v.boundary = True
        bV = set(HC.V)
        with pytest.raises(NotImplementedError):
            _retopologize(
                HC, bV, dim=3,
                remesh_mode='adaptive',
                remesh_kwargs={'max_iterations': 1},
            )
