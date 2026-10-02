"""Registry of every method choice in the dynamic (Lagrangian) pipeline.

This module is DATA.  Each :class:`MethodAxis` names one decision the
solver makes (time integrator, connectivity policy, per-phase volume
split, ...).  Each :class:`MethodOption` names one allowed value, says
where in the code it lives, what its measured status is, and which lane
log / test is the evidence.

Two kinds of axis exist:

``explicit``
    A field of :class:`ddgclib.methods.SolverMethods`.  The value is
    chosen by the user and applied by the builders in ``_config.py``.

``reported`` (``explicit=False``)
    The code picks the value implicitly (by dimension, by whether a
    cache exists, by which function was called).  These are not
    controllable today; :func:`ddgclib.methods.effective_methods`
    resolves them on a concrete mesh so they are RECORDED next to every
    result.  Making one of them controllable is a code change to the
    operator layer, not to this registry.

Status vocabulary (``MethodOption.status``):

- ``validated``      default of a pinned case / regression-locked
- ``opt-in``         tested, works, deliberately not the default
- ``experimental``   exists, no regression net, use with care
- ``measured-worse`` A/B'd against the default and rejected (DO-NOT)
- ``broken``         known to give wrong results in some regime
- ``dead``           no production caller / superseded

Keep this file in sync with ``METHODS.md`` (regenerate the tables with
``python -m ddgclib.methods --markdown``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ['MethodOption', 'MethodAxis', 'AXES', 'AXIS_GROUPS', 'STATUSES']

STATUSES = (
    'validated', 'opt-in', 'experimental', 'measured-worse', 'broken', 'dead',
)


@dataclass(frozen=True)
class MethodOption:
    """One allowed value of a method axis."""

    key: Any
    summary: str
    status: str
    where: str
    evidence: str = ''
    dims: tuple[int, ...] = (1, 2, 3)
    phases: str = 'both'   # 'single' | 'multi' | 'both'

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"bad status {self.status!r} for option {self.key!r}; "
                f"allowed: {STATUSES}"
            )


@dataclass(frozen=True)
class MethodAxis:
    """One decision the dynamic pipeline makes."""

    name: str
    title: str
    group: str
    options: tuple[MethodOption, ...]
    default: Any = None
    kind: str = 'choice'      # 'choice' | 'int' | 'float' | 'bool' | 'object'
    explicit: bool = True
    applies_to: str = 'both'  # 'single' | 'multi' | 'both'
    control: str = ''         # where the value is applied at runtime
    notes: str = ''

    def keys(self) -> list[Any]:
        return [o.key for o in self.options]

    def option(self, key: Any) -> MethodOption:
        for o in self.options:
            if o.key == key:
                return o
        raise KeyError(
            f"unknown value {key!r} for axis {self.name!r}; "
            f"allowed: {self.keys()}"
        )


def _opt(key, summary, status, where, evidence='', dims=(1, 2, 3), phases='both'):
    return MethodOption(key, summary, status, where, evidence, dims, phases)


AXIS_GROUPS = (
    'problem', 'time', 'connectivity', 'thermodynamics', 'forces',
    'dual geometry (reported)', 'execution',
)

_LANE = 'docs_temp/debug_session/'

_AXES: list[MethodAxis] = [
    # ------------------------------------------------------------------
    # problem class
    # ------------------------------------------------------------------
    MethodAxis(
        name='mesh', title='Mesh representation', group='problem',
        default='complex', explicit=False,
        control='how the case builds HC (domain builders always build '
                'hyperct.Complex); reported from type(HC) and HC._SC',
        options=(
            _opt('complex', 'hyperct.Complex vertex-vertex flag complex '
                 '(v.nn sets) + raw top-simplex list HC._simplices', 'validated',
                 'hyperct/_complex.py:Complex',
                 'the only representation any ddgclib code path uses'),
            _opt('simplicial', 'Complex(simplicial=True) with the hyperct '
                 'SimplicialComplex / _ops index-array layer (branch '
                 'wip/simplicial-layer, not on master)', 'broken',
                 'hyperct (branch wip/simplicial-layer: _simplicial.py, _ops/, '
                 'Complex._simplices setter hooks)',
                 'audit 2026-09-25: unused by ddgclib; 4 verified sync defects '
                 '(retriangulation ignored, invalidate no-op, collapse holes, '
                 'read side effects); parked on the branch 2026-09-25'),
        ),
    ),
    MethodAxis(
        name='phases', title='Phase model', group='problem',
        default='single',
        control='selects dudt_i vs multiphase_dudt_i and _retopologize vs '
                '_retopologize_multiphase',
        options=(
            _opt('single', 'One fluid; v.p from pressure_model or held; '
                 'forces from operators.stress.stress_force', 'validated',
                 'ddgclib/operators/stress.py:stress_force',
                 'Hagen-Poiseuille / hydrostatic machine-precision equilibria '
                 '(for NODAL pressures: with dual-cell averages the wall half '
                 'cells break the linear precision, 2D static residual 1.9075 '
                 'm/s^2 at every refinement, laneP). laneP: with an EOS and '
                 'gravity the discrete hydrostatic equilibrium is a saddle of '
                 'the discrete energy: slow modes grow at 1.13 g/c0 (0.353 1/s '
                 'on the 2D column; the same in a closed box, where the stiffness '
                 'matrix is symmetric to 5e-10; 1.01 g/c0 at refinement 2, so it '
                 'does not refine away), followed in a real run (amplitude x4.816 '
                 'at 200 acoustic times, cosh(sigma t) 4.816). Viscosity turns the growth into '
                 'creep (rate ~ 1/mu: 1.2e-4 1/s at mu = 0.5 rho c0 dx); with '
                 'the viscosity of water the no-slip drop exceeds c0 at 64 t_ac. '
                 'FREE SURFACE: the open-fan force is not an energy gradient '
                 '(stiffness asymmetry 11 to 16 %, closed fans symmetric to '
                 '5e-10); without viscosity the no-slip column flutters at '
                 'refinement 2 (0.75 1/s, x61 in a real run against x59 '
                 'predicted) and 4 (1.22 1/s), not at 3 and never with a closed '
                 'lid; 0.05 rho c0 dx of viscosity removes it '
                 '(cases_dynamic/Hydrostatic_column/diagnose_column.py)'),
            _opt('multi', 'Sharp-interface n-phase model on MultiphaseSystem; '
                 'per-phase summed stress + surface tension', 'validated',
                 'ddgclib/operators/multiphase_stress.py:multiphase_stress_force',
                 'oscillating droplet 2D/3D pins (test_case_oscillating_droplet.py)'),
        ),
    ),
    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    MethodAxis(
        name='integrator', title='Time integrator', group='time',
        default='symplectic_euler',
        control='ddgclib.dynamic_integrators.<name>(HC, bV, dudt_fn, ...)',
        options=(
            _opt('symplectic_euler', 'u += dt a; x += dt u_new (Lagrangian)',
                 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:symplectic_euler',
                 'every pinned dynamic case'),
            _opt('euler', 'x += dt u_old; u += dt a (forward Euler, Lagrangian)',
                 'opt-in',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:euler',
                 'cube_flow demos and the u=0 static floor tests only; laneK: with '
                 'an EOS it GROWS at CFL 0.25 even on fixed connectivity '
                 '(symplectic_euler is stable to dt c_s/dx = 1.5)'),
            _opt('rk45', 'scipy RK45 per macro step; duals/edge cache frozen at '
                 'macro-step start (stale within stages)', 'experimental',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:rk45',
                 'audit 2026-09-25 §1.2: no per-stage dual rebuild; skips BCs '
                 'when no interior vertices'),
            _opt('euler_velocity_only', 'Eulerian fixed mesh, u += dt a only. '
                 'Validation/equilibrium checks ONLY (CLAUDE.md)', 'opt-in',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:euler_velocity_only',
                 'Poiseuille equilibrium tests'),
            _opt('euler_adaptive', 'Advective-CFL adaptive dt; velocity_only=True '
                 'by default (Eulerian), else forward Euler', 'experimental',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:euler_adaptive',
                 'audit §1.3: no sound-speed term in the CFL; not used by any case'),
        ),
    ),
    # ------------------------------------------------------------------
    # connectivity / retopology
    # ------------------------------------------------------------------
    MethodAxis(
        name='connectivity', title='Connectivity (retopology) policy',
        group='connectivity', default='delaunay',
        control='retopologize_fn / skip_triangulation / remesh_mode / '
                'periodic_axes on the integrator; bound into the multiphase '
                'retopo partial by SolverMethods.retopologize_fn()',
        notes='Formerly spread over five switches and named differently '
              'per case (retopo_policy_2d "delaunay_remap", 3D CLI '
              '"delaunay|dual_only", dam break partial).',
        options=(
            _opt('delaunay', 'Per-step global scipy Delaunay rebuild '
                 '(connect_and_cache_simplices), boundary_from_simplices, '
                 'compute_vd, dual volumes/edge-area cache', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize',
                 '2D droplet default WITH remap (laneE); bare (no remap) is '
                 'measured-worse for multiphase: 2D l2 0.490 / 3D 1.524 (lane5, laneB). '
                 'SINGLE-PHASE + EOS without remap: measured unstable in every '
                 'laneK arm (a flip changes a dual volume by 33-100 %, read as '
                 '3e4-5e4 Pa; blows up at CFL 0.01, c_s 10 or 100, n 1 or 7.15, '
                 'redistribution on or off); use remap=conservative (laneR). '
                 'FREE SURFACE (laneP): the rebuild triangulates the convex hull '
                 'of the cloud, so the gap between a moved free surface and the '
                 'hull is filled with near-degenerate simplices. Even WITH '
                 'remap=conservative the 2D hydrostatic column reaches 42 m/s at '
                 '3 t_ac, the total volume stays pinned to the hull and the '
                 'column carries half the hydrostatic head (integrated L2 4.9e3 '
                 'Pa = rho g H / 2); from the equilibrium masses it exceeds c0 '
                 'at 95 t_ac; 3D 7.1 m/s. Use delaunay_material. In 1D the '
                 'rebuild is the sorted chain and equals a non-reconnecting loop '
                 'to round-off (hydrostatic_1D preset)',
                 dims=(1, 2, 3)),
            _opt('delaunay_material', 'Per-step Delaunay rebuild that keeps the '
                 'fluid domain: the boundary of the previous connectivity is '
                 'material, the simplices Delaunay adds between a free surface '
                 'and the convex hull are removed again (geometric test: winding '
                 'number of the simplex centroid about the old boundary; exposed '
                 'flat wall simplices are dropped). 2D: domain kept to round-off. '
                 '3D: kept up to the slivers of free-surface diagonal flips (no '
                 'facet recovery); the relative volume change is returned and '
                 'warned above domain_tol. Half-cell boundary volumes, no '
                 'edge-area cache (2D shared dual vertices, 3D p_ij ring). Single '
                 'phase; with remap=conservative it runs the single-phase remap '
                 'around the rebuild', 'opt-in',
                 'ddgclib/methods/_retopo.py:retopologize_material_delaunay',
                 'laneP 2026-10-01, with remap=conservative on the hydrostatic '
                 'column (free surface). 2D: 200 t_ac from uniform density max|u| '
                 'envelope 1.24e-4 m/s (dual_only 1.14e-4), from the equilibrium '
                 'masses bounded at a reconnection noise floor of 2.3e-6 m/s '
                 '(dual_only decays to 4.2e-9); mass drift below 3e-14; largest '
                 'offset K (s - 1) of the mass rescale 0.99 Pa; domain volume '
                 'change 0 to round-off in every call. 3D (189 vertices): '
                 '100 t_ac envelope 5.6e-4 to 5.7e-4 m/s in two processes '
                 '(dual_only_bare 2.4e-4), from the equilibrium masses a floor '
                 'of 9.9e-7 (dual_only_bare 5.4e-7); '
                 'domain change at most 1.2e-6 per call (the four corner squares '
                 'at the first rebuild), 1.6e-6 summed over 185 calls, offset 2.4 '
                 'Pa; after about 10 t_ac the 3D drop run is reproducible between '
                 'processes to 2 digits only (cospherical mesh, Delaunay ties). '
                 'A 0.004 bowl pushed into the builder surface in one go changes '
                 'the 3D domain by 9.8e-5. Interior integrated pressure error '
                 '(2D) 36 to 42 Pa against 0.23 Pa on the builder mesh: that is '
                 'the offset between cell centroid and vertex on the Delaunay '
                 'cells (attribution run: 31.3 Pa integrated, rho g x rms offset '
                 '31.1 Pa, error against the nodal value 0.20 Pa, builder mesh '
                 '0.18). REVIEW FIX: the first peel was topological (a simplex '
                 'went when it exposed a facet that was not an old boundary '
                 'facet) and removed FLUID in 3D, where the diagonals of planar '
                 'wall squares change between rebuilds: lattice cube total '
                 'volume down to 0.790, 41.7 % lost in one call on the coarsest '
                 'lattice; with the geometric test the volume is 1 to 1e-15 and '
                 'every 1D / 2D number is bit-identical. Re-deriving the '
                 'boundary orientation at each call is a DO-NOT (a thin surface '
                 'simplex inverts during the step: winding numbers 0.5, 3D '
                 'domain flicker 8e-5 per call, peak 0.185 instead of 0.157 m/s). '
                 'WITHOUT the remap it is the laneK instability (128 m/s). '
                 'Needs HC._simplices before the first call; no merge step, no '
                 'inlet / outlet BCs, backend not applied. '
                 'test_material_delaunay.py (18), test_case_hydrostatic.py',
                 dims=(2, 3), phases='single'),
            _opt('dual_only', 'skip_triangulation=True: keep builder connectivity, '
                 'refresh v.boundary tags, duals, dual volumes (and per-phase '
                 'split / redistribution / EOS for multiphase) every step',
                 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize '
                 '(skip_triangulation branch)',
                 '3D droplet default (laneB l2 0.24811); 2D opt-in (l2 0.17857, lane5). '
                 'Cannot follow large deformation (dam break NaN-aborts, laneF). '
                 'laneK: single-phase + EOS is stable here without any remap '
                 '(pressure force = exact volume gradient to 1e-10; '
                 'symplectic_euler stable to dt c_s/dx = 1.5). laneS: on a builder '
                 'mesh it now reads simplex_exact volumes and runs with free '
                 '(untagged) surface vertices, which raised IndexError before: '
                 'free-surface box, g = 0, 1e-6 m/s seed decays 1.5e-6 -> 2.0e-9 '
                 'over 8 acoustic times (test_builder_simplex_cache.py). laneP: '
                 'hydrostatic_2D / hydrostatic_2D_periodic presets (200 t_ac: '
                 'max|u| envelope 1.1e-4 / 1.1e-6 m/s from uniform density, '
                 '4.2e-9 / 3.4e-11 from the equilibrium masses). 3D SINGLE '
                 'PHASE + EOS: do not use, the 3D branch zeroes the dual volume '
                 'of every frozen vertex, so wall cells read P0: hydrostatic 3D '
                 'column max|u| 3.9e-2 m/s and integrated L2 1.5e4 Pa (1.5 '
                 'rho g H) at 100 t_ac even from the equilibrium masses; use '
                 'dual_only_bare',
                 dims=(2, 3)),
            _opt('dual_only_bare', 'Frozen connectivity, boundary retagged from '
                 'HC.boundary(), compute_vd + cache_dual_volumes (half-cell '
                 'boundary volumes, no edge-area cache) + per-phase split; NO '
                 'mps.refresh, NO redistribution. Multiphase: no EOS update '
                 '(p_phase frozen at its setup value). Single phase: the EOS in '
                 'dudt_fn reads the refreshed dual volumes, so the pressure is '
                 'live', 'validated',
                 'ddgclib/methods/_retopo.py:bare_dual_refresh',
                 'static_droplet_2D pin 1.1847162859108737e-03 (was the '
                 'case-local _dual_only_retopo closure; audit F11: validates '
                 'surface tension against a FROZEN pressure field). laneP: the '
                 'integrator boundary_filter is honoured (default None = whole '
                 'hull frozen, unchanged). Single phase the EOS in dudt_fn is '
                 'live, so this is the fixed-connectivity path with wall half '
                 'cells and p_ij faces in 3D: hydrostatic_3D preset, 100 t_ac '
                 'max|u| envelope 2.4e-4 m/s (uniform density start) / 5.4e-7 '
                 '(equilibrium masses), integrated L2 0.99 Pa = 1.0e-4 rho g H',
                 dims=(2, 3)),
            _opt('frozen', 'retopologize_fn=False: NO topology or dual refresh at '
                 'all. Surface meshes / A.5.a static probes only', 'opt-in',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize',
                 'A.5.a frozen-mesh floors 2.3749e-3 (2D) / 6.0153e-05 (3D); '
                 'multiphase p_phase is NEVER updated on this path; laneK: with an '
                 'EOS the pressure is inert (dual volumes never refreshed), so a '
                 'stable frozen run is NOT evidence of EOS stability'),
            _opt('adaptive', 'hyperct.remesh.adaptive_remesh local split/collapse/flip '
                 'preserving the v.phase interface; 2D only', 'opt-in',
                 'hyperct/remesh/_driver.py:adaptive_remesh via '
                 '_retopologize(remesh_mode="adaptive")',
                 'lane4 upstream conservation fix; l2 0.326 at refine 3/3 '
                 '(second-best after dual_only); no pinned case uses it',
                 dims=(2,)),
            _opt('periodic', 'retopologize_periodic: wrap, ghost-cell Delaunay, '
                 'min-image duals (2D only in stress.py). Multiphase: '
                 'retopologize_multiphase_periodic adds mps.refresh + '
                 'redistribution + EOS (no remap, no cadence)', 'experimental',
                 'ddgclib/geometry/periodic.py:retopologize_periodic; '
                 'ddgclib/methods/_retopo.py:retopologize_multiphase_periodic',
                 'periodic path ignores skip_triangulation/remesh/backend; '
                 'shearing_plate_droplet 2D is unstable (interface lost by '
                 't=0.044 s), 3D crashes in setup; domain_bounds is a build-time '
                 'argument (geometry), periodic_axes the method field. laneP, '
                 'single phase + EOS: not usable. After ONE retopologize_periodic '
                 'of periodic_rectangle (unit square, refinement 3) the total '
                 'dual volume is 2.488 (exact 1.0; seam simplices are measured '
                 'with raw coordinates), the cache holds 287 simplices instead '
                 'of 256, 45 of 136 vertices fail dual-face closure (29 of them '
                 'interior) and 6 interior vertices are tagged boundary; the '
                 'hydrostatic column exceeds c0 at 0.66 t_ac (diagnose_column.py '
                 'periodic). The Hydrostatic_2D_periodic runner therefore uses '
                 'free-slip walls on dual_only',
                 dims=(2, 3)),
            _opt('custom', 'User-supplied retopologize_fn callable (e.g. '
                 'static_droplet_2D bare dual-only, Hagen_Poiseuile_3D cylinder)',
                 'experimental',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize '
                 '(callable branch)',
                 'recorded by label only; kwargs forwarded by declared name'),
        ),
    ),
    MethodAxis(
        name='remap', title='Conservative retopology remap', group='connectivity',
        default=None,
        control='retopo_remap= in _retopologize_multiphase (multiphase partial) '
                'or in _retopologize (single-phase partial built by '
                'SolverMethods.retopologize_fn)',
        notes='Same axis, two implementations. Multiphase: projection of the '
              'PRE-call field (cadence on projection_every). Single-phase: '
              'fresh snapshot, so the step\'s compression survives and there '
              'is no cadence to choose.',
        options=(
            _opt(None, 'No remap: reconnection changes dual volumes, EOS reads '
                 'them as compression', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize_multiphase',
                 'required value under dual_only (remap is a silent no-op there)'),
            _opt('conservative', 'Pressure field invariant across the rebuild. '
                 'MULTIPHASE: stage-1 dual refresh on OLD connectivity, rebuild, '
                 'per-phase redistribution, vol_corr gauge, '
                 'restore_pressure_multiphase, anchor_phase_pressure_levels. '
                 'SINGLE-PHASE: fresh snapshot eos(m / V_old-connectivity) at the '
                 'current positions, rebuild, re-target EVERY vertex with a dual '
                 'volume (frozen walls included), one exact mass rescale',
                 'validated',
                 'ddgclib/operators/mass_redistribution.py:restore_pressure_multiphase, '
                 'anchor_phase_pressure_levels (multi); snapshot_pressure_fresh, '
                 'redistribute_mass_single_phase(include_frozen=True) via '
                 '_retopologize(retopo_remap=) (single)',
                 '2D droplet default (laneD/E, l2 0.17479); dam break survives '
                 'reconnection (laneF); 3D multiphase measured-worse (laneE l2 1.873 '
                 'vs 0.248, DO-NOT; confirmed cache-free by laneI). SINGLE-PHASE '
                 '(laneK prototype, laneR library): box + EOS stable where bare '
                 'Delaunay blows up at any CFL / c_s / n; final KE within 0.2 % of '
                 'dual_only; pinned by test_single_phase_remap.py. Interior-only '
                 'or stale-v.p variants are measured DO-NOTs (laneK P2, P3). '
                 'laneP: the uniform offset K (s - 1) of the mass rescale '
                 '(laneR known limit) is NOT what breaks a free-surface column: '
                 'without the rescale the convex-hull arm is worse (59.4 against '
                 '42.1 m/s, mass drift +1.5 %, total volume 1.03; '
                 'diagnose_column.py remap), and once the hull fill is removed '
                 '(connectivity=delaunay_material) the offset stays below 1 Pa '
                 '(convex arm: up to 2.7e3 Pa per rebuild) and the run with and '
                 'without the rescale agree to 3 digits. No gauge was added',
                 dims=(2, 3)),
        ),
    ),
    MethodAxis(
        name='projection_every', title='Pressure-projection cadence',
        group='connectivity', default=1, kind='int', applies_to='multi',
        control='projection_every= in _retopologize_multiphase (partial); '
                'counter on mps._projection_call_idx',
        options=(
            _opt(1, 'Every call: per-phase masses re-targeted to the PRE-call '
                 'pressure snapshot', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize_multiphase',
                 'all pins; attributed cause of the 2D over-decay (laneH) and '
                 'half the 3D bump (laneG)'),
            _opt('N>1', 'Project every N-th call; off-cadence remap snapshots '
                 'are strain-advanced (evolve_snapshot_local_strain)', 'opt-in',
                 'ddgclib/operators/mass_redistribution.py:evolve_snapshot_local_strain',
                 'laneH: delaunay+remap N=2 gives l2 0.03796 (-78%) and matches '
                 'the two-fluid reference to ~2%; tail gate uncalibrated; '
                 '3D untested; forbidden under bare delaunay (ValueError)',
                 phases='multi'),
        ),
    ),
    MethodAxis(
        name='displacement_eps', title='Skip-retopology displacement gate',
        group='connectivity', default=None, kind='float',
        control='displacement_eps= integrator kwarg (_do_retopologize)',
        options=(
            _opt(None, 'Retopologize every step', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize'),
            _opt('eps>0', 'Skip when every vertex moved < eps since the last '
                 'call; first call always skips', 'measured-worse',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:'
                 '_displacement_gate_should_skip',
                 'lane5 sweep: eps in {0.01,0.05,0.2} h_min all worse than either '
                 'extreme (cases_dynamic/oscillating_droplet/src/_params.py)'),
        ),
    ),
    MethodAxis(
        name='merge_cdist', title='Pre-retopology vertex merge', group='connectivity',
        default=None, kind='float',
        control='merge_cdist= integrator kwarg (forwarded to _retopologize)',
        options=(
            _opt(None, 'No merge', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize'),
            _opt('cdist>0', 'HC.V.merge_all(cdist) before Delaunay; NOT '
                 'mass-conserving and does not merge m_phase', 'experimental',
                 'hyperct/_vertex.py:merge_all; ddgclib/multiphase.py:'
                 'mass_conserving_merge is the separate conserving path',
                 'DO-NOT wire into multiphase without per-phase ledger (laneF)'),
        ),
    ),
    MethodAxis(
        name='frozen_set', title='Frozen (wall) vertex set',
        group='connectivity', default='hull',
        control='frozen_set= in _retopologize / _retopologize_multiphase, '
                'bound into the retopology partial by '
                'SolverMethods.retopologize_fn()',
        notes='Which vertices the integrators do not move (bV). The tag '
              'v.boundary that compute_vd needs for half cells follows the '
              'topological boundary under both values. boundary_filter '
              '(build-time argument) narrows either set.',
        options=(
            _opt('hull', 'bV is rebuilt at every retopology from the '
                 'topological boundary of the new connectivity, narrowed by '
                 'boundary_filter. A vertex is frozen because it is on the '
                 'hull and released when it is not', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize '
                 '(step 6)',
                 'every pin (all droplet, hydrostatic and electrolysis presets '
                 'run on it). FAILS when one vertex steps past a straight wall: '
                 'the wall vertices next to it leave the hull, are released and '
                 'integrated, and nothing re-captures them (audit 2026-09-25 '
                 'F10 C1). Measured by laneL: Hagen_Poiseuile_2D, 3000 steps: '
                 'walls released at step 1262 (two outlet buffer vertices '
                 'drift past the wall lines), 2 of 62 wall vertices still '
                 'frozen, 60 moved, largest displacement 3.46; dam_break_2D '
                 'in its two ejection configurations (alpha_art 0.2; 0.5 to '
                 't = 0.45 s): one step after the first vertex leaves the '
                 'tank the walls are released and all 32 wall vertices move '
                 '(test_frozen_set.py, '
                 'cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py)'),
            _opt('membership', 'bV is persistent: a retopology keeps the '
                 'members that are still in the complex (and pass '
                 'boundary_filter), never adds a vertex because it is on the '
                 'hull and never drops one because it is not. A hull vertex '
                 'that is not a member (inlet, outlet, free surface, a vertex '
                 'that left through a wall) is tagged, gets a half cell and '
                 'is integrated; a member off the hull stays frozen with a '
                 'closed cell; capture at a wall is left to the BC that holds '
                 'bV (PositionalNoSlipWallBC)', 'opt-in',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize '
                 '(step 6)',
                 'laneL 2026-10-01. Hagen_Poiseuile_2D (preset), 3000 steps, past '
                 'the former collapse: 62 wall vertices, 62 still frozen, 0 '
                 'moved. dam_break_2D (preset): shipped run (1585 steps) final '
                 'state bit-identical to hull; in the two ejection '
                 'configurations 32 of 32 walls stay frozen and in place, but '
                 'the fluid still blows up and the run aborts with the same '
                 'QhullError 23 / 8 steps later than under hull (after 1303 '
                 'against 1280 steps; 3030 against 3022; the same in every '
                 'process since the tied simplex vote is deterministic, fix '
                 'round 1): the walls are not what fails there. Bit-identical '
                 'to hull while no vertex leaves the hull: 2D droplet with '
                 'remap (100 steps), electrolysis 2D (6330 steps, the shipped '
                 'horizon), electrolysis 3D (300 steps). CONNECTIVITY: '
                 'implemented for delaunay only. With adaptive it RAISES (in '
                 'SolverMethods and in both retopology functions): '
                 'hyperct.remesh protects vertices by v.boundary, not by bV. '
                 'Measured on a channel: one adaptive retopology left the 8 '
                 'wall vertices created by wall-edge splits unfrozen (10 of '
                 '18 against 18 of 18 under hull; integrated, they leave the '
                 'wall), and with 7 members off the hull adaptive_remesh '
                 'moved up to 7 of them (smoothing, up to 0.17) and removed '
                 'up to 5 (collapse). LIMITS: impenetrability is not '
                 'enforced (HP2D: one fluid vertex 5.1e-3 outside the top wall '
                 'at t = 30); a wall-row vertex injected by an inlet is frozen '
                 'only when PositionalNoSlipWallBC runs AFTER the inlet BC '
                 '(under hull the next retopology captured it); 3D: a vertex '
                 'whose dual fan fails is tagged and zero-volumed but not '
                 'frozen (0 occurrences in a jittered box and a 3D droplet); '
                 'merge_cdist can merge a member into a mobile vertex; not '
                 'implemented for periodic, delaunay_material and custom '
                 'retopology (SolverMethods raises), not needed for dual_only '
                 '/ dual_only_bare / frozen (their bV never changes). '
                 'test_frozen_set.py (31)',
                 dims=(2, 3)),
        ),
    ),
    # ------------------------------------------------------------------
    # thermodynamics / mass bookkeeping
    # ------------------------------------------------------------------
    MethodAxis(
        name='redistribute_mass', title='Pressure-preserving mass redistribution',
        group='thermodynamics', default=False, kind='bool',
        control='redistribute_mass= (integrator kwarg for single-phase, partial '
                'for multiphase); single-phase also needs pressure_model',
        options=(
            _opt(False, 'Lagrangian masses held against the new duals', 'opt-in',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py',
                 'multiphase noredist rings acoustically: 2D l2 0.495 (laneH); '
                 '3D l2 0.266 vs 0.248 but cleaner channels (laneG)'),
            _opt(True, 'After each rebuild rescale masses so the pre-rebuild '
                 'pressure field is reproduced, exact total per phase',
                 'validated',
                 'ddgclib/operators/mass_redistribution.py:'
                 'redistribute_mass_multiphase / redistribute_mass_single_phase',
                 'every shipped multiphase setup binds True (M1 rollout); '
                 '3D A.5.b 1.44e-3 -> 7.38e-05 (Phase 2c). SINGLE-PHASE: on its '
                 'own it is NOT a cure for Delaunay + EOS (laneK: stale v.p '
                 'snapshot, frozen walls skipped, so wall flip jumps survive); '
                 'combine it with remap=conservative (laneR)'),
        ),
    ),
    MethodAxis(
        name='split_method', title='Per-phase dual-volume split at interface vertices',
        group='thermodynamics', default='neighbour_count', applies_to='multi',
        control='split_method= in mps.refresh (setup) AND in the retopo partial; '
                'the two MUST match',
        notes='A typo silently falls back to neighbour_count (multiphase.py '
              'tests only == "exact"). SolverMethods validates the key.',
        options=(
            _opt('neighbour_count', 'Interface vertex: fraction of 1-ring bulk '
                 'neighbours per phase times v.dual_vol', 'validated',
                 'ddgclib/multiphase.py:MultiphaseSystem.split_dual_volumes',
                 'all pins', phases='multi'),
            _opt('exact', '2D: clip the barycentric dual polygon by the interface '
                 'polyline (NOT rescaled to v.dual_vol); 3D: PCA tangent plane '
                 'clip of the dual polyhedron, rescaled to v.dual_vol',
                 'measured-worse',
                 'ddgclib/geometry/_dual_split_2d.py:split_dual_polygon_2d / '
                 'split_dual_polyhedron_3d',
                 '3D end-to-end 1.75e-3 vs 1.44e-3 (worse); 2D retopo-neutral '
                 'but no metric gain (debugging_plan 2026-04-29)',
                 dims=(2, 3), phases='multi'),
        ),
    ),
    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    MethodAxis(
        name='curvature_path', title='Interface curvature / surface-tension stencil',
        group='forces', default='integrated', applies_to='multi',
        control='curvature_path= on multiphase_dudt_i (dudt partial)',
        options=(
            _opt('integrated', '2D: exact piecewise-linear FTC gamma*(t_next - t_prev) '
                 '(surface_tension_force_2d); 3D: cotangent/Heron '
                 'hndA_i_interface on the interface sub-mesh', 'validated',
                 'ddgclib/operators/multiphase_stress.py:_interface_surface_tension',
                 'all pins. The 3D apex cache HC._interface_edge_to_apex was '
                 'never invalidated (audit 2026-09-25 T1); FIXED in laneI '
                 '(cleared when the interface triangle set changes, '
                 'test_interface_cache_invalidation.py). laneI also showed the '
                 'droplet runs never flip interface triangles, so every pinned '
                 '3D score (dual_only, delaunay, delaunay+remap) is bit-identical '
                 'before/after the fix',
                 dims=(2, 3), phases='multi'),
            _opt('stokes', '3D conormal boundary integral on the barycentric dual '
                 '(integrated_hndA_i_interface); 2D aliases "integrated"',
                 'experimental',
                 'ddgclib/_curvatures_heron.py:integrated_hndA_i_interface',
                 'bit-identical to integrated on a STATIC mesh (Probe 2). The '
                 'coordinate-keyed cache HC._interface_x_to_v that made the force '
                 'vanish after the first vertex move (audit T2) is cleared on '
                 'every interface refresh since laneI (tested); no dynamic A/B '
                 'or regression pin yet',
                 dims=(2, 3), phases='multi'),
            _opt('csf_dual', 'Magnitude of the integrated stencil redirected along '
                 'the dual-face normal S_inner', 'experimental',
                 'ddgclib/operators/multiphase_stress.py:_csf_dual_surface_tension',
                 'A/B probe only, no tests', dims=(2, 3), phases='multi'),
        ),
    ),
    MethodAxis(
        name='pressure_flux', title='Pressure flux across the dual faces',
        group='forces', default='centred', applies_to='single',
        control='pressure_flux= on dudt_i / stress_force (dudt partial); '
                'registry operators.stress.pressure_flux_methods',
        notes='Added 2026-09-26 (capillary_rise_energy_grad). The multiphase '
              'force still hard-codes the centred flux.',
        options=(
            _opt('centred', 'Face-average -1/2 (p_i + p_j) A_ij: exact volume '
                 'gradient at uniform p (linear precision), no dissipation; '
                 'BLIND to the checkerboard density/pressure mode (the '
                 'face average of an alternating field is uniform)', 'validated',
                 'ddgclib/operators/stress.py:pressure_flux',
                 'every pinned case; capillary_rise static check: half-cell '
                 'closure 4e-15 (run_free_surface_static_check.py). laneH: the '
                 'linear precision is that of the face areas it reads (axis '
                 'edge_area_source): exact in 2D where the dual cell closes, '
                 'and with the 3D p_ij ring; with the 3D batch_e_star cache the '
                 'force of a linear pressure is off by 0.4 % on the builder '
                 'cylinder and by 1.3 % (median) / 21 % (max) on a jittered one, '
                 'and the developing pipe picks up radial velocity (6.3e-3 = '
                 '0.03 U_max in each of 9 processes; l2 0.0811 or 0.0821, '
                 'depending on the process, against 0.0566 with '
                 'simplex_gradient: diagnose_poiseuille.py arms3d, refinement '
                 '1, 600 steps, results/laneH/arms3d*.json)',
                 phases='single'),
            _opt('acoustic-riemann', 'Lagrangian Godunov contact pressure '
                 'p* = 1/2 (p_i + p_j) - 1/2 rho_f c_f (u_j - u_i).n: momentum '
                 'conserving, zero for rigid translation, damps normal velocity '
                 'jumps; numerical bulk viscosity ~ rho c |d| on compressive '
                 'modes (low-Mach caveat). Needs an EOS pressure_model',
                 'measured-worse',
                 'ddgclib/operators/stress.py:pressure_flux_riemann',
                 'capillary_rise dynCA A/B 2026-09-26 (energy_grad README 5.6b): '
                 'with c_s = 10 u_ref the numerical viscosity rho c dx ~ 0.5 Pa s '
                 'is 700x mu; the column barely flows (smoke L2 0.53 vs 0.075 '
                 'centred + density diffusion; quiescent column drains as with '
                 'centred but the driven rise is lost). Correct for acoustic '
                 'velocity noise, wrong tool at low Mach; the density-diffusion '
                 'axis is the one to use. Unit tests: antisymmetry, rigid '
                 'translation, dissipativity (test_pressure_flux_stabilisation.py)',
                 phases='single'),
            _opt('simplex_gradient', 'Volume form: minus the integral over the '
                 'dual cell of the gradient of the piecewise-linear pressure, '
                 'F_i = -sum_T |T| / (dim + 1) grad(p)_T. Exact for a linear '
                 'pressure on any simplicial mesh and independent of the dual '
                 'face areas; zero for a uniform pressure at every vertex, hull '
                 'included (an open cell feels NO ambient pressure). Needs '
                 'HC._simplices', 'opt-in',
                 'ddgclib/operators/stress.py:pressure_force_simplex_gradient',
                 'laneH 2026-10-01. Force of a linear pressure exact to 1e-15 at '
                 'every vertex, hull included, on jittered Delaunay meshes in 2D '
                 'and 3D (test_simplex_gradient_flux.py). Preset '
                 'hagen_poiseuille_3D (prescribed pressure): largest radial '
                 'velocity in the window 2.6e-18 against 6.3e-3 with centred '
                 'on the e_star cache and 1.3e-5 to 3.6e-4 with centred on the '
                 'p_ij ring (custom wrapper, 2.9x the wall time); l2 from the '
                 'developed profile 0.0566 / 0.0811 to 0.0821 / 0.0566 '
                 '(refinement 1, 600 steps, diagnose_poiseuille.py arms3d). '
                 'The preset arm is bit-identical from process to process; '
                 'the two centred arms are not (protocol rule 8): ranges over '
                 '9 processes, each arm takes one of two values (cache: l2 '
                 '0.08211 in 7, 0.08111 in 2; ring: radial velocity 1.26e-5 '
                 'in 6, 3.65e-4 in 3). In 2D it reproduces centred '
                 '(l2 1.1998e-2 in both arms). LIMITS: not run with an EOS; not '
                 'for a free surface that the pressure should push outwards; '
                 'not a sum of pairwise antisymmetric fluxes',
                 dims=(2, 3), phases='single'),
        ),
    ),
    MethodAxis(
        name='viscous_flux', title='Viscous flux across the dual faces',
        group='forces', default='two_point', applies_to='single',
        control='viscous_flux= on dudt_i / stress_force (dudt partial); '
                'registry operators.stress.viscous_flux_methods',
        notes='Added 2026-10-01 (laneH). Same barycentric dual faces in both '
              'values; they differ in the velocity gradient put on a face. The '
              'multiphase force hard-codes the two-point flux.',
        options=(
            _opt('two_point', 'Face gradient from the two cell values along the '
                 'edge: (mu / |d_ij|) (u_j - u_i) (d_hat . A_ij)', 'validated',
                 'ddgclib/operators/stress.py:viscous_flux',
                 'every pinned case; Poiseuille equilibrium residual 1e-13 on '
                 'the symmetric refined-square mesh (test_stress.py). laneH '
                 '2026-10-01: NOT linearly precise without a symmetric edge '
                 'stencil. Residual of a LINEAR velocity field, in units of '
                 'G Vol of the Poiseuille problem (median / max): 0.64 / 2.6 on '
                 'a jittered sheared 2D Delaunay mesh at refinement 3, 1.41 / '
                 '5.2 at refinement 4 (it grows like 1 / h), 0.20 on the '
                 'unjittered 3D builder cylinder, 0.28 / 0.84 on a jittered one '
                 '(diagnose_poiseuille.py static). Developing Poiseuille flow '
                 'on the moving mesh does not reach the profile: 2D l2 0.26 '
                 '(u_max 0.189 against 0.150; simplex_gradient 0.012), 3D 0.53 '
                 '(0.276 against 0.200; 0.057); at Re 10 the transverse '
                 'velocity grows from round-off (l2 1.0, 26 vertices outside '
                 'the walls). Its edge weight d_ij . A_ij also inherits the '
                 'orientation defect of the 2D area vector (edge_area_source '
                 'shared_vd_2d)',
                 phases='single'),
            _opt('simplex_gradient', 'Gradient of the piecewise-linear velocity '
                 'on each primal simplex, integrated over the dual faces: '
                 'F_i = mu sum_T G_T . a_iT, a_iT = -|T| grad(phi_i) (the '
                 'cotangent weights in 2D). Linearly precise on any simplicial '
                 'mesh, pairwise antisymmetric, negative semi-definite. Needs '
                 'HC._simplices', 'opt-in',
                 'ddgclib/operators/stress.py:viscous_force_simplex_gradient',
                 'laneH 2026-10-01. Residual of a linear velocity field 1e-15 '
                 'on the meshes where two_point has 0.2 to 5; equal to the '
                 'cotangent weights in 2D; momentum and dissipation tested '
                 '(test_simplex_gradient_flux.py). Presets hagen_poiseuille_2D / '
                 '_3D: the developing flow reaches the developed profile to l2 '
                 '1.09e-2 (2D shipped run: Re 10, refinement 2, 478 vertices, '
                 't = 120 s) and 1.99e-2 (3D: Re 2, refinement 2, 928 vertices, '
                 '16-sided pipe). The 2D error falls by about 3 per refinement: '
                 '1.08e-2, 3.6e-3, 1.34e-3 at refinement 1, 2, 3 on the regular '
                 'rows (t = 10 s); 3.3e-2, 1.1e-2, 3.4e-3 on the sheared rows '
                 '(t = 60 s). The nodal '
                 'residual of the exact quadratic profile is NOT small on an '
                 'irregular mesh (median 0.05 to 0.07 of G Vol: Galerkin P1, '
                 'the solution error is what converges). LIMITS: a simplex '
                 'with |T| <= 1e-3 l_min^dim is left out (flat simplices have '
                 'no gradient), which breaks linear precision at its vertices '
                 'if they are interior: none in the 2D run, in the 3D run a '
                 'flat tetrahedron of four free vertices of equal radius (a '
                 'planar rectangle between two cross-sections) in 2 of 600 '
                 'steps; a hull vertex is coupled to whatever the convex-hull '
                 'fill connects it to, so integrated vertices must not be on '
                 'the hull (PeriodicInletBufferedBC); explicit stability as '
                 'for two_point; not run with an EOS or multiphase',
                 dims=(2, 3), phases='single'),
        ),
    ),
    MethodAxis(
        name='density_diffusion', title='Gradient-corrected density diffusion',
        group='thermodynamics', default=None, kind='float', applies_to='single',
        control='density_diffusion= integrator kwarg (euler, symplectic_euler); '
                'needs pressure_model=EOS; case loops call '
                'operators.stabilisation.density_diffusion_step directly',
        notes='Added 2026-09-26. delta-SPH type mass flux on the dual faces; '
              'the cure for the checkerboard density mode that the centred '
              'pressure flux cannot see. Not available on rk45 / euler_adaptive.',
        options=(
            _opt(None, 'No density diffusion', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:symplectic_euler',
                 'every pinned case', phases='single'),
            _opt('delta>0', 'dm_i/dt = sum_j delta c0 |A_ij| [(rho_j - rho_i) - '
                 '1/2 (grad rho_i + grad rho_j).d_ij]: exactly mass conserving, '
                 'exactly zero on linear density fields (interior), no shear '
                 'viscosity; explicit stability delta < ~0.3 at acoustic CFL 0.4',
                 'opt-in',
                 'ddgclib/operators/stabilisation.py:density_diffusion_step',
                 'capillary_rise dynCA smoke (water R 0.5 mm, after the corner '
                 'fix): L2 0.26 -> 0.075 (delta 0.05) / 0.13 (0.1); 0.2-0.3 '
                 'over-smooth (capillary_rise_energy_grad README Section 5). '
                 'laneP: it does not cure the slow instability of the inviscid '
                 'hydrostatic column (no-slip 2D drop with the viscosity of '
                 'water exceeds c0 at 63 / 57 t_ac for delta 0.05 / 0.1, 64 '
                 'without)',
                 phases='single'),
        ),
    ),
    # ------------------------------------------------------------------
    # dual geometry: resolved from the mesh, reported only
    # ------------------------------------------------------------------
    MethodAxis(
        name='dual_method', title='Dual vertex construction', group='dual geometry (reported)',
        default='barycentric', explicit=False,
        control='hard-coded compute_vd(HC, method="barycentric") in _retopologize',
        options=(
            _opt('barycentric', 'Dual vertices at simplex barycentres', 'validated',
                 'hyperct/ddg/_compute_dual.py:compute_vd'),
            _opt('circumcentric', 'Dual vertices at circumcentres (benchmarks only; '
                 'linear precision lost on jittered meshes)', 'opt-in',
                 'hyperct/ddg/_compute_dual.py:compute_vd',
                 'INTEGRATED_BENCHMARKS.md'),
        ),
    ),
    MethodAxis(
        name='dual_path', title='Dual construction path', group='dual geometry (reported)',
        default='simplex_aware', explicit=False,
        control='presence of HC._simplices (connect_and_cache_simplices at every '
                'Delaunay retopology; rebuild_simplex_cache_2d / _3d of the built '
                'connectivity in every domain builder, DomainResult.__post_init__)',
        options=(
            _opt('simplex_aware', 'Top-simplex cache drives compute_vd, '
                 'boundary_from_simplices, exact volumes', 'validated',
                 'hyperct/ddg/_retriangulation.py:connect_and_cache_simplices',
                 'commit 8321c71; test_simplex_aware_duals.py'),
            _opt('nn_walk', 'Legacy 1-skeleton (v.nn intersection) walk; ghost '
                 'K_{d+1} cliques on Delaunay meshes', 'dead',
                 'hyperct/ddg/_compute_dual.py (legacy branch)',
                 'docs/3d_simplex_aware_dual_fix.md'),
        ),
    ),
    MethodAxis(
        name='dual_volume', title='Dual cell volume source', group='dual geometry (reported)',
        default='simplex_exact', explicit=False,
        control='dim + HC._simplices + HC._vd_method + whether batch_e_star runs '
                '(stress.py:_use_exact_barycentric_volume, _retopologize step 5b)',
        options=(
            _opt('simplex_exact', 'Vol_i = (1/(d+1)) sum_{T contains i} |T| '
                 '(hyperct.ddg.simplex_dual_volumes / vertex_dual_volume)',
                 'validated',
                 'hyperct/ddg/_dual_volume.py',
                 '3D switch ON 2026-07-29 (laneA), floor re-pinned 7.274172e-05. '
                 'laneS (2026-10-01): the domain builders cache the simplices of '
                 'the connectivity they build, so SETUP reads this source too '
                 '(rectangle total 0.96875 -> 1.0, box 0.9167 -> 1.0, no volume '
                 'jump at the first retopology; test_builder_simplex_cache.py). '
                 'Shipped Hydrostatic_2D then settles (100 t_ac, |u| 3.0e-4) '
                 'instead of reaching 10 c0 at 6.9 t_ac; every pinned number is '
                 'bit-identical (the droplet meshes already carried a Delaunay '
                 'cache)'),
            _opt('fan_walk_3d', 'batch_e_star / v_star tetra fan sum; undercounts '
                 '1-4% interior, ~20% boundary', 'measured-worse',
                 'hyperct/ddg/_operators.py:batch_e_star(compute_volumes=True)',
                 'docs_temp/audit/dual-volume-3d.md. laneS: no builder mesh '
                 'reaches it any more (it gave the box builder 0.9167 of its '
                 'volume at setup); left for hand-built 3D complexes without a '
                 'simplex cache and for circumcentric duals', dims=(3,)),
            _opt('dual_cell_area_2d', 'Shoelace area of the 2D dual polygon '
                 '(circumcentric duals, or a hand-built 2D complex without a '
                 'simplex cache; builder meshes no longer reach it)', 'opt-in',
                 'hyperct/ddg/_dual_cell.py:dual_cell_area_2d',
                 'Was broken until laneS (laneK: boundary polygon without the '
                 'vertex itself, so the four corner cells were 4x too small, '
                 'rectangle total 0.96875, and a moving free-surface vertex got '
                 '1/4 of its own volume change; Hydrostatic_2D blew up with 0 '
                 'flips). FIXED in hyperct 2026-10-01: the half cell of a '
                 'boundary vertex is walked as an open chain and closed through '
                 'the vertex; it now equals the simplex rule at every vertex of '
                 'a kinked boundary to 1e-11 (hyperct test_dual_volume.py) and '
                 'the Hydrostatic loop run on it (driver variant fallback) '
                 'matches the simplex_exact run. Degenerate fans still use the '
                 'angular sort (no vertex point); circumcentric boundary cells '
                 'are not validated (laneK P14/P15)', dims=(2,)),
            _opt('interval_1d', 'Distance between the two dual vertices', 'validated',
                 'ddgclib/operators/stress.py:dual_volume', dims=(1,)),
        ),
    ),
    MethodAxis(
        name='boundary_dual_vol', title='Boundary-vertex dual volume convention',
        group='dual geometry (reported)', default='half_cell', explicit=False,
        control='dim: 3D retopology zeroes boundary dual_vol (batch_e_star path); '
                '1D/2D/periodic/setup keep the truncated half cell, and so do '
                'connectivity=dual_only_bare and delaunay_material in 3D',
        options=(
            _opt('zeroed', 'v.dual_vol = 0 on every vertex in bV after retopology',
                 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize step 5b',
                 'measured 2026-09-25: 3D box boundary dual_vol 0.0. laneP: '
                 'with a single-phase EOS a zero-volume wall cell reads the '
                 'reference pressure P0, which breaks any case whose wall '
                 'pressure is not P0 (hydrostatic 3D)', dims=(3,)),
            _opt('half_cell', 'Boundary vertices keep the truncated dual cell '
                 '(cache_dual_volumes path)', 'validated',
                 'ddgclib/operators/stress.py:cache_dual_volumes',
                 'measured 2026-09-25: 2D rectangle max boundary dual_vol 0.0156, '
                 'total 1.0. Also 3D under connectivity=dual_only_bare and '
                 'delaunay_material (laneP)', dims=(1, 2, 3)),
        ),
    ),
    MethodAxis(
        name='edge_area_source', title='Oriented dual face area A_ij source',
        group='dual geometry (reported)', default='shared_vd_2d', explicit=False,
        control='dim + HC._edge_area_cache + HC._periodic_axes '
                '(stress.py:stress_force, dual_area_vector)',
        notes='laneJ (2026-09-25) recommends making this an EXPLICIT 3D axis '
              '(keys e_star_cache | p_ij | p_ij_simplex) once p_ij_simplex has '
              'a vectorised hyperct kernel; until then it stays reported. '
              'Forcing p_ij today = connectivity="custom" wrapper that clears '
              'HC._edge_area_cache after each retopology '
              '(cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py).',
        options=(
            _opt('batch_e_star_cache', '3D: cached e_star fan areas from '
                 'batch_e_star(orient=True) at the last retopology', 'validated',
                 'hyperct/ddg/_operators.py:batch_e_star',
                 'all 3D pins. NOT linearly precise: laneJ measured per-edge '
                 'difference to p_ij median 0.125 / max 0.625 (box), closure '
                 'residual ~1 % at every droplet interface vertex, linear-precision '
                 'error 3-25 %; it carries ~79 % of the 3D static floor '
                 'retopology excess (7.274172e-05 vs 6.2839e-05 on p_ij, frozen '
                 'floor 6.0153e-05) and part of the dynamic outward bump '
                 '(final inflation 1.87 % -> 0.82 % R0 on p_ij) - but p_ij alone '
                 'scores l2 0.28713 vs 0.24811 (laneG cancellation exposed), so '
                 'no flip', dims=(3,)),
            _opt('p_ij_ring_3d', '3D DEC p_ij dual polygon ring walk (linearly '
                 'precise on box/ball, 1e-18); used only when no cache exists',
                 'opt-in',
                 'ddgclib/operators/stress.py:_dual_area_vector_3d_p_ij',
                 'test_stress.py p_ij linear-precision tests. laneJ: the face '
                 'vertex is chosen as the common neighbour nearest the midpoint '
                 'of two tet barycentres; on 743 of 5193 directed droplet edges '
                 'that picks a non-face vertex, giving the 2.6 % closure '
                 'residuals laneG attributed to the pressure side. 4.1x wall '
                 'cost (2.05 vs 0.50 s/step)', dims=(3,)),
            _opt('p_ij_simplex', 'p_ij polygon with ring order and face vertices '
                 'read from HC._simplices (exact faces): closure 2.3e-16, linear '
                 'precision 1.9e-15 at every interior vertex', 'experimental',
                 'cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py '
                 '(driver only, not in the library yet)',
                 'laneJ: static floor 6.28386e-05, dynamic l2 0.28653 / tail '
                 '0.08818, 1.89 s/step. Target default after a vectorised '
                 'hyperct kernel, 3D re-pin and co-evaluation with the '
                 'redistribution-pump rework (laneG lever b)', dims=(3,)),
            _opt('shared_vd_2d', '2D: segment between the two dual vertices shared '
                 'by v_i and v_j, oriented outward', 'validated',
                 'ddgclib/operators/stress.py:dual_area_vector (2D branch)',
                 'all 2D pins (batch_e_star raises for dim != 3). DEFECT found '
                 'by laneH 2026-10-01, NOT fixed: the vector is oriented away '
                 'from x_i as seen from the midpoint of the dual segment; when '
                 'the barycentres of the two triangles at the edge subtend more '
                 'than 180 degrees at x_i that points AGAINST the edge. On a '
                 'jittered sheared Delaunay mesh 6 of 665 vectors are flipped, '
                 'the closure residual of an interior cell is 0.23 and the '
                 'centred force of a linear pressure is off by up to 17.6 V |g|; '
                 'oriented by A_ij . d_ij > 0 both are exact (1e-14). Census of '
                 'the test suite (counting probe): 2055 flipped vectors of 1.13e6 '
                 '2D calls, in the laneL HP2D reproducer (737 / 431), the '
                 'bare-Delaunay + EOS instability test of laneR (272), '
                 'test_material_delaunay (9) and the buffer zones of the '
                 'Poiseuille runs (no integrated vertex); none in a droplet, '
                 'hydrostatic or dam-break pin. Strict xfail reproducer: '
                 'test_simplex_gradient_flux.py::'
                 'test_2d_dual_area_vectors_close_on_a_sheared_jittered_mesh',
                 dims=(2,)),
            _opt('min_image_2d', '2D periodic: minimum-image rebuild of the dual '
                 'segment', 'experimental',
                 'ddgclib/operators/stress.py:dual_area_vector (periodic branch)',
                 'd_ij is NOT min-imaged (06_known_issues)', dims=(2,)),
        ),
    ),
    MethodAxis(
        name='boundary_rule', title='Topological boundary detection',
        group='dual geometry (reported)', default='boundary_from_simplices',
        explicit=False,
        control='HC._simplices present -> boundary_from_simplices, else HC.boundary(); '
                'dual_only carries the previous bV',
        options=(
            _opt('boundary_from_simplices', 'Faces belonging to exactly one top '
                 'simplex', 'validated', 'hyperct/ddg/_boundary.py:boundary_from_simplices'),
            _opt('HC.boundary', 'Legacy hyperct vertex-hull test', 'opt-in',
                 'hyperct/_complex.py:Complex.boundary'),
            _opt('carried_bV', 'skip_triangulation: reuse the previous (possibly '
                 'filtered) bV; 3D failed fans are promoted and stay boundary',
                 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize'),
        ),
    ),
    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    MethodAxis(
        name='backend', title='batch_e_star compute backend', group='execution',
        default=None,
        control='backend= integrator kwarg, a NAME that _retopologize resolves '
                'to a hyperct backend instance (_resolve_backend, one instance '
                'per name). Only reaches the 3D batch_e_star (dual face areas '
                'of the edge-area cache); not read in 1D / 2D, and compute_vd '
                'always runs numpy',
        notes='laneH fix round 2026-10-02: until then every value but None '
              'stopped the first 3D retopology with "\'str\' object has no '
              'attribute \'batch_cross_areas\'" (the name was handed to '
              'batch_e_star, which calls methods of an instance). The axis '
              'changes who computes the cached areas, not the method: a force '
              'that does not read the cache (both fluxes simplex_gradient) is '
              'unaffected to the bit.',
        options=(
            _opt(None, 'numpy', 'validated', 'hyperct/_backend.py'),
            _opt('torch', 'PyTorch tensors, on CUDA when available, else on the '
                 'CPU. ImportError without PyTorch (no silent fallback)',
                 'opt-in', 'hyperct/_backend.py',
                 'test_gpu_backend.py (compute_vd). laneH fix round: the ddg '
                 'environment has no PyTorch, so the fast suite covers this '
                 'value by its ImportError only; run in environments with '
                 'PyTorch 2.14 / 2.10 + CUDA (RTX 4090): the three backend '
                 'tests of test_methods.py pass (edge-area cache equal to the '
                 'numpy one to rtol 1e-12), run_cluster.py --backend torch '
                 'reproduces the numpy l2 after 50 steps; with PyTorch 2.8 + '
                 'CUDA and pressure_flux=centred the GPU areas move l2 in the '
                 '16th digit after 40 steps (0.23195448845688516 against '
                 '0.2319544884568851)'),
            _opt('gpu', 'Auto-detect: PyTorch on CUDA, else PyTorch on the CPU, '
                 'else numpy', 'opt-in', 'hyperct/_backend.py',
                 'laneH fix round: test_methods.py::TestEffectiveMethods::'
                 'test_3d_backend_axis_fills_the_same_edge_area_cache; preset '
                 'hagen_poiseuille_3D with backend replaced gives the serial '
                 'result to the bit (test_case_hagen_poiseuille.py::'
                 'TestDeveloping3DBackendAxis); default of '
                 'Hagen_Poiseuile_3D/run_cluster.py'),
            _opt('multiprocessing', 'hyperct MultiprocessingBackend (owns a '
                 'pool of 2 processes; its batch_cross_areas is the numpy one, '
                 'so nothing is gained on this path)', 'experimental',
                 'hyperct/_backend.py',
                 'laneH fix round: same two tests as gpu'),
        ),
    ),
    MethodAxis(
        name='workers', title='dudt evaluation workers', group='execution',
        default=None, kind='int',
        control='workers= integrator kwarg (_compute_accel fork pool)',
        options=(
            _opt(None, 'Sequential', 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_compute_accel'),
            _opt('n>1', 'fork pool over dudt_fn (Linux only). Safe when the force '
                 'has no side effects (pressure_model=None); with an EOS bound '
                 'into dudt_fn the v.p / v.rho writes of _resolve_pressure happen '
                 'in the children and are LOST in the parent', 'experimental',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_compute_accel',
                 'audit 2026-09-25 §0.5; used by Hagen_Poiseuile 2D (20) / 3D (8), '
                 'where pressure_model is None, until laneH; both presets are '
                 'serial since (2D at about 200 vertices: 38.2 ms per step '
                 'serial against 85.5 with 20 workers) and --workers is an arm '
                 '(default 8 in Hagen_Poiseuile_3D/run_cluster.py)'),
        ),
    ),
]

AXES: dict[str, MethodAxis] = {a.name: a for a in _AXES}

# sanity: every group used is declared
for _a in _AXES:
    if _a.group not in AXIS_GROUPS:
        raise RuntimeError(f"axis {_a.name} uses undeclared group {_a.group!r}")
