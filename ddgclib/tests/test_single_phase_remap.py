"""Single-phase conservative retopology remap (lane R).

A Delaunay reconnection changes the barycentric dual volume of a vertex
by 33-100 % at fixed positions.  With masses held, the EOS reads that as
compression and a single-phase weakly-compressible run blows up at any
time step (lane K).  ``_retopologize(retopo_remap='conservative')``
(``SolverMethods(remap='conservative')``) makes the pressure field
invariant across the rebuild.

Evidence and the measured DO-NOT variants:
docs_temp/debug_session/laneK-single-phase-eos-instability.md
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from hyperct.ddg import simplex_dual_volumes

from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC
from ddgclib.dynamic_integrators._integrators_dynamic import _retopologize
from ddgclib.eos import TaitMurnaghan
from ddgclib.geometry.domains import rectangle
from ddgclib.initial_conditions import CompositeIC, CustomFieldIC, DualVolumeMass
from ddgclib.methods import SolverMethods
from ddgclib.operators.mass_redistribution import snapshot_pressure_fresh

RHO0, MU, U0, C_S = 1000.0, 50.0, 0.1, 10.0


def _eos() -> TaitMurnaghan:
    return TaitMurnaghan(rho0=RHO0, P0=0.0, K=RHO0 * C_S**2, n=1.0,
                         rho_clip=(0.5, 2.0))


def _swirl(x: np.ndarray) -> np.ndarray:
    """Divergence-free swirl, zero on all four walls."""
    sx, cx = np.sin(np.pi * x[0]), np.cos(np.pi * x[0])
    sy, cy = np.sin(np.pi * x[1]), np.cos(np.pi * x[1])
    return U0 * np.array([sx * sx * sy * cy, -sx * cx * sy * sy])


def _box(refinement: int = 2):
    """Closed unit box, all walls no-slip.  The connectivity is settled
    with one library retopology before the ICs, so every arm starts from
    the same Delaunay mesh with exact (simplex) dual volumes."""
    result = rectangle(L=1.0, h=1.0, refinement=refinement, flow_axis=0)
    HC, bV = result.HC, result.bV
    for v in HC.V:
        v.boundary = v in bV
    _retopologize(HC, bV, 2)
    CompositeIC(CustomFieldIC(_swirl, field_name='u'),
                DualVolumeMass(rho=RHO0)).apply(HC, bV)
    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=2), bV)
    dx_min = min(float(np.linalg.norm(v.x_a[:2] - nb.x_a[:2]))
                 for v in HC.V for nb in v.nn)
    return HC, bV, bc_set, dx_min


def _run(methods: SolverMethods, n_steps: int = 200, cfl: float = 0.25):
    HC, bV, bc_set, dx_min = _box()
    eos = _eos()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # the bare-Delaunay arm warns
        dudt_fn = methods.dudt_fn(HC, mu=MU, pressure_model=eos)
    hist: list[tuple[float, float]] = []

    def cb(step, t, HC_cb, bV_cb=None, diagnostics=None):
        ke = sum(0.5 * v.m * float(np.dot(v.u[:2], v.u[:2])) for v in HC_cb.V)
        umax = max(float(np.linalg.norm(v.u[:2])) for v in HC_cb.V)
        hist.append((ke, umax))

    with np.errstate(all='ignore'):
        methods.integrate(HC, bV, dudt_fn, dt=cfl * dx_min / C_S,
                          n_steps=n_steps, bc_set=bc_set, callback=cb,
                          pressure_model=eos)
    ke = np.array([h[0] for h in hist])
    umax = np.array([h[1] for h in hist])
    return HC, ke, umax


REMAP = SolverMethods(dim=2, connectivity='delaunay', remap='conservative',
                      redistribute_mass=True)
DUAL_ONLY = SolverMethods(dim=2, connectivity='dual_only')
BARE = SolverMethods(dim=2, connectivity='delaunay')


class TestRemapAcrossOneRebuild:
    """The rebuild must not change any vertex pressure, walls included."""

    def _moved_box(self):
        HC, bV, _, _ = _box()
        rng = np.random.default_rng(0)
        # Random interior displacement: real compression + forces flips
        # (the structured grid's squares are co-circular).
        for v in list(HC.V):
            if v in bV:
                continue
            HC.V.move(v, tuple(v.x_a[:2] + 0.03 * rng.standard_normal(2)))
        return HC, bV

    @staticmethod
    def _edges(HC) -> set:
        return {frozenset((id(v), id(nb))) for v in HC.V for nb in v.nn}

    def test_pressure_invariant_and_mass_conserved(self):
        HC, bV = self._moved_box()
        eos = _eos()
        p_before = snapshot_pressure_fresh(HC, 2, eos)
        m_before = sum(v.m for v in HC.V)
        edges_before = self._edges(HC)

        _retopologize(HC, bV, 2, pressure_model=eos, redistribute_mass=True,
                      retopo_remap='conservative')

        assert self._edges(HC) != edges_before, "probe must reconnect"
        assert sum(v.m for v in HC.V) == pytest.approx(m_before, rel=1e-14)
        # Density after = scale * density of the snapshot pressure, with
        # ONE scale for every vertex, walls included (the exact-mass
        # rescale; a uniform factor exerts no force on a closed fan).
        ratio = np.array([v.m / v.dual_vol / float(eos.density(p_before[id(v)]))
                          for v in HC.V])
        assert len(ratio) == len(p_before) == sum(1 for _ in HC.V)
        assert np.ptp(ratio) < 1e-12
        assert abs(ratio[0] - 1.0) < 1e-2
        assert any(v in bV for v in HC.V)

    def test_without_remap_the_rebuild_jumps_the_pressure(self):
        HC, bV = self._moved_box()
        eos = _eos()
        p_before = snapshot_pressure_fresh(HC, 2, eos)
        _retopologize(HC, bV, 2)
        p_after = {id(v): float(eos.pressure(v.m / v.dual_vol)) for v in HC.V}
        dp = max(abs(p_after[k] - p_before[k]) for k in p_before)
        assert dp > 0.05 * RHO0 * C_S**2

    def test_fresh_snapshot_is_not_the_stale_vertex_pressure(self):
        HC, bV = self._moved_box()
        for v in HC.V:
            v.p = 0.0                       # what the last force call left
        snap = snapshot_pressure_fresh(HC, 2, _eos())
        assert max(abs(p) for p in snap.values()) > 1.0
        vols = simplex_dual_volumes(HC, 2)
        assert sum(vols.values()) == pytest.approx(1.0, rel=1e-12)

    def test_requires_eos_and_redistribution(self):
        HC, bV, _, _ = _box()
        with pytest.raises(ValueError, match='redistribute_mass'):
            _retopologize(HC, bV, 2, pressure_model=_eos(),
                          retopo_remap='conservative')
        with pytest.raises(ValueError, match='EquationOfState'):
            _retopologize(HC, bV, 2, redistribute_mass=True,
                          retopo_remap='conservative')
        with pytest.raises(ValueError, match='retopo_remap'):
            _retopologize(HC, bV, 2, retopo_remap='bogus')

    def test_default_is_unchanged(self):
        """retopo_remap=None must be the previous behaviour to the bit."""
        eos = _eos()
        masses = []
        for kw in ({}, {'retopo_remap': None}):
            HC, bV = self._moved_box()
            for v in HC.V:
                v.p = 0.0
            _retopologize(HC, bV, 2, pressure_model=eos,
                          redistribute_mass=True, **kw)
            masses.append(sorted((v.x, v.m) for v in HC.V))
        assert masses[0] == masses[1]


class TestBoxDecayWithEOS:
    """Weakly-compressible viscous decay in a closed box (CFL 0.25,
    Mach 0.01, 200 steps): the lane K arms F5 / F9 / P10 through the
    library."""

    @pytest.fixture(scope='class')
    def runs(self):
        return {name: _run(m) for name, m in
                (('remap', REMAP), ('dual_only', DUAL_ONLY), ('bare', BARE))}

    def test_bare_delaunay_is_unstable(self, runs):
        _, ke, umax = runs['bare']
        assert umax.max() > 10 * U0
        assert ke.max() > 100 * ke[0]

    def test_remap_is_stable_and_dissipative(self, runs):
        HC, ke, umax = runs['remap']
        assert np.all(np.isfinite(ke))
        assert umax.max() <= 1.05 * umax[0]
        assert ke[-1] < 0.1 * ke[0]
        rho = np.array([v.m / v.dual_vol for v in HC.V])
        assert np.all(np.abs(rho / RHO0 - 1.0) < 0.01)

    def test_remap_matches_fixed_connectivity(self, runs):
        _, ke_r, _ = runs['remap']
        _, ke_d, _ = runs['dual_only']
        # Same mesh at t = 0; the fixed-connectivity mesh then shears
        # with the swirl, so the two discretisations differ by a few
        # per cent of KE0 (measured 2.7 %), not by the remap.
        assert ke_r[0] == ke_d[0]
        assert np.max(np.abs(ke_r - ke_d)) < 0.05 * ke_d[0]

    def test_remap_pinned(self, runs):
        _, ke, _ = runs['remap']
        assert ke[0] == pytest.approx(PIN_KE0, rel=1e-9)
        assert ke[-1] == pytest.approx(PIN_KE_END, rel=1e-7)


# Pinned 2026-10-01 (laneR), preset-equivalent config REMAP above.
PIN_KE0 = 0.43990437705748403
PIN_KE_END = 0.008987112540608227
