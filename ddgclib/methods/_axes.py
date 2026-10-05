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
    'dual geometry', 'execution',
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
        notes='SETUP CHOICE, NOT AN AXIS (laneB 2026-10-05): the droplet-in-box '
              'builders ddgclib.geometry.domains.droplet_in_box_2d / _3d take '
              'box_shift="move_all" (default: the outer box is translated onto '
              'the droplet by one HC.V.move_all, every vertex kept) or '
              '"evict" (the loop of single moves used until 2026-10-05, which '
              'lost one outer vertex per key collision: 2D refinement 3 6 of '
              '145, 3D refinement 2 3 of 189, the (L, ..., L) corner always '
              'among them, so the convex hull was cut at that corner and the '
              'builder Delaunay papered over the holes with larger cells). The '
              'setups of oscillating_droplet, electrolysis_bubble (its '
              'off-centre 2D builder uses the same helper) and '
              'shearing_plate_droplet pass it through and record it in '
              'params["box_shift"]; every runner writes it into the extra '
              'block of methods.json; diagnose_box_shift.py measures both '
              'arms. Every droplet, electrolysis and shearing-plate number '
              'pinned before laneB was produced on the "evict" mesh and is '
              'reproduced by that value (static_droplet_2D 1.1847162859108737e-03, '
              '3D plateau 7.274172e-05, 2D baseline l2 0.17479361640597058 and '
              '3D 0.24811340819647862 to the bit).',
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
                 '3 t_ac (laneO 2026-10-05, with the correct orientation of the '
                 '2D area vectors: 111.7 m/s at 2.2 t_ac, volume 1.037; the '
                 'legacy rule area_orientation=dual_midpoint gives the 42.10 '
                 'again), the total volume stays pinned to the hull and the '
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
                 'Pa. laneT 2026-10-02: until then the 3D drop run was '
                 'reproducible between processes to 2 digits only after about '
                 '10 t_ac (40 t_ac envelope 1.718e-3 to 1.731e-3 in 4 '
                 'interpreters; the 100 t_ac range above is from before); it is '
                 'bit-identical in every interpreter now (40 t_ac envelope '
                 '1.7288e-3), but still decided by ties: a 1e-15 shift of the '
                 'interior vertices moves the peak at 0.87 t_ac by 1.25e-3 '
                 'relative and the kinetic energy at 8 t_ac by 2.2 % '
                 '(cospherical mesh, free-surface edge areas). '
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
                 '(equilibrium masses), integrated L2 0.99 Pa = 1.0e-4 rho g H. '
                 'laneQ 2026-10-05: the 3D dual faces follow the axis '
                 'edge_area_source (None = the legacy ring walk; hydrostatic_3D '
                 'reads the exact p_ij_simplex cache since laneQ)',
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
                 '(second-best after dual_only); no pinned case uses it. laneT '
                 '2026-10-02: until then the remeshed complex depended on memory '
                 'addresses (the driver oriented each edge by id(), and a '
                 'collapse keeps the first vertex of the pair): 6 final states '
                 'in 6 interpreters on the 2D droplet after 60 steps (KE '
                 '1.1467e-5 to 1.1483e-5); one since (edges oriented in HC.V '
                 'order, test_determinism.py). Every number quoted for this '
                 'value before laneT is one realisation of that spread',
                 dims=(2,)),
            _opt('periodic', 'retopologize_periodic: wrap, ghost-cell Delaunay, '
                 'min-image duals (2D only in stress.py). Multiphase: '
                 'retopologize_multiphase_periodic adds mps.refresh + '
                 'redistribution + EOS (no remap, no cadence)', 'experimental',
                 'ddgclib/geometry/periodic.py:retopologize_periodic; '
                 'ddgclib/methods/_retopo.py:retopologize_multiphase_periodic',
                 'periodic path ignores skip_triangulation/remesh/backend; '
                 'laneT: the 3D rebuild is the same in every interpreter since '
                 '2026-10-02 (5 steps of the shearing-plate 3D short setup: 2 '
                 'final states in 6 interpreters before, KE 6.258e-6 or '
                 '6.276e-6); '
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
                 'diagnose_column.py remap; both measured before laneO, with '
                 'the correct 2D area orientation the convex arm with the '
                 'rescale reaches 111.7 m/s), and once the hull fill is removed '
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
                 '0.03 U_max; l2 0.0821 against 0.0566 with simplex_gradient: '
                 'diagnose_poiseuille.py arms3d, refinement 1, 600 steps, '
                 'results/laneH/arms3d*.json). Until laneT that l2 was 0.0811, '
                 '0.0821 or 0.1107 depending on the process; it is '
                 '0.08211494566330206 in every interpreter since, and a 1e-15 '
                 'shift of the interior vertices moves it between 0.079 and '
                 '0.169 (8 seeds): the arm is chaotic, quote it with that '
                 'spread. laneQ 2026-10-05: on the exact faces '
                 '(edge_area_source=p_ij_simplex) the same arm gives l2 '
                 '0.056658 and radial velocity 2.5e-17, i.e. the result of the '
                 'simplex_gradient volume form, at 44 s against 58 s',
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
                 'on the e_star cache and 1.26e-5 with centred on the '
                 'p_ij ring (custom wrapper, 2.9x the wall time); l2 from the '
                 'developed profile 0.0566 / 0.0821 / 0.0566 '
                 '(refinement 1, 600 steps, diagnose_poiseuille.py arms3d). '
                 'Every arm is bit-identical from process to process since '
                 'laneT (2026-10-02). Before, the two centred arms were not: '
                 'over 9 processes the cache arm gave l2 0.08211 in 7 and '
                 '0.08111 in 2, the ring arm radial velocity 1.26e-5 in 6 and '
                 '3.65e-4 in 3. In 2D it reproduces centred '
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
                 'the walls). Its edge weight d_ij . A_ij inherited the '
                 'orientation defect of the 2D area vector until laneO '
                 '(axis area_orientation)',
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
        name='area_orientation', title='Sign rule of the 2D dual face vector',
        group='forces', default='primal_edge',
        control='area_orientation= on dudt_i / stress_force and '
                'multiphase_dudt_i / multiphase_stress_force (dudt partial, '
                'bound only when not the default) -> '
                'stress.dual_area_vector(orientation=); 2D only, the 3D '
                'sources and the 1D sign ignore it; operators that read '
                'dual_area_vector outside the force (density_diffusion_step, '
                'scalar_gradient_integrated, the csf_dual curvature path) '
                'always use the default',
        notes='Added 2026-10-05 (laneO). The 2D dual face of an edge is the '
              'segment between the two dual vertices (barycentres, or '
              'barycentre and edge midpoint on the hull) the endpoints share; '
              'its normal has magnitude |segment| and the axis fixes its sign. '
              'The defect was found by laneH (2026-10-01) and left in place '
              'because the fix moves the pinned numbers of lanes L, R and P; '
              'laneO fixed it and re-pinned them.',
        options=(
            _opt('primal_edge', 'A_ij . (x_j - x_i) > 0: the normal of the dual '
                 'segment on the side of x_j. Exact for any valid pair of '
                 'triangles (A_ij . d_ij = (2/3) (|T_left| + |T_right|)) and '
                 'for a hull edge ((2/3) |T|); A_ij = -A_ji and the cell of an '
                 'interior vertex closes on every mesh',
                 'validated',
                 'ddgclib/operators/stress.py:_orient_2d',
                 'laneO 2026-10-05: on the laneH mesh (sheared 0.3, jitter 0.2, '
                 'refinement 3, reconnected) 0 of 806 vectors against their '
                 'edge, closure of every interior cell 2.8e-17, antisymmetry '
                 'exact, equal to the simplex-cache reference '
                 'simplex_area_vectors to 1e-16 (dual_midpoint: 6 flipped, '
                 'closure 0.23, linear-pressure force off by 17.6 V |g|); '
                 'also on the disk builder mesh (4 flipped at setup under '
                 'dual_midpoint) and in the periodic branch. Every 2D pin that '
                 'had a flipped vector was re-measured and re-pinned '
                 '(test_frozen_set.py, test_single_phase_remap.py, '
                 'test_material_delaunay.py, test_case_hagen_poiseuille.py '
                 'two-point arm; see the laneO log for old -> new); the droplet, '
                 'dam-break and hydrostatic pins never read a flipped vector '
                 'and are bit-identical. Tests: test_area_orientation.py, '
                 'test_simplex_gradient_flux.py (the former strict xfail)',
                 dims=(1, 2, 3)),
            _opt('dual_midpoint', 'The vector points away from x_i as seen from '
                 'the midpoint of the dual segment (the rule before laneO). '
                 'Flipped when the two triangles at the edge subtend more than '
                 '180 degrees at x_i (x_i inside the triangle of the three '
                 'other vertices): the vector then points AGAINST its edge, the '
                 'cell does not close and the pressure and viscous fluxes of '
                 'that face act backwards. Kept so that the numbers pinned '
                 'before 2026-10-05 can be reproduced',
                 'broken',
                 'ddgclib/operators/stress.py:_orient_2d',
                 'laneH 2026-10-01 (found), laneO 2026-10-05 (measured): census '
                 'of the fast suite before the fix 2341 flipped vectors of '
                 '1.17e6 2D calls in 8 tests (laneL HP2D reproducer 737 / 431, '
                 'HP2D two-point arm 354, density-diffusion pair-order test 280, '
                 'laneR bare-Delaunay instability test 272, HP2D preset pin 252 '
                 'at inlet-buffer vertices only, test_material_delaunay convex '
                 'arm 9, the xfail 6); along the runs: 2D droplet with remap, '
                 'dual_only, dam break, electrolysis, hydrostatic 2D (fixed '
                 'connectivity, remap, density diffusion): 0 flipped vectors; '
                 'bare-Delaunay 2D droplet (refinement 2, 300 steps): 454 '
                 'flipped vectors in 109 of 300 evaluations from step 168 on; '
                 'laneR bare box: 272 from step 33, the KE had doubled at step '
                 '1 already (the laneK instability does not come from it). '
                 'Reproduces every pre-laneO pin through '
                 'preset.replace(area_orientation="dual_midpoint") (tested in '
                 'test_area_orientation.py)',
                 dims=(2,)),
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
        name='dual_method', title='Dual vertex construction', group='dual geometry',
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
        name='dual_path', title='Dual construction path', group='dual geometry',
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
        name='dual_volume', title='Dual cell volume source', group='dual geometry',
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
        group='dual geometry', default='half_cell', explicit=False,
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
        group='dual geometry', default=None,
        control='edge_area_source= integrator kwarg, forwarded by name to '
                '_retopologize / _retopologize_multiphase / bare_dual_refresh / '
                'retopologize_material_delaunay, which fill HC._edge_area_cache '
                'from the chosen source and record the value on '
                'HC._edge_area_source; every reader of a dual face (stress_force, '
                'multiphase_stress_force, velocity_difference_tensor, '
                'scalar_gradient_integrated, velocity_laplacian, '
                'density_diffusion_step, the csf_dual stencil) takes the cache '
                'entry of the edge, else dual_area_vector(source=HC._edge_area_source). '
                '3D only; 1D and 2D keep their reported sources',
        notes='EXPLICIT since laneQ (2026-10-05); the 2D values stay reported. '
              'A flat tetrahedron (qhull\'s triangulated output puts 44 of them '
              'into the 2540 of the droplet mesh and 13 into a Delaunay rebuild '
              'of the refinement 2 box) has a dual face piece in its own plane, '
              'so no geometric rule can orient it: the exact sources orient it '
              'by its neighbours (combinatorial orientation of the complex), '
              'which is what closes the cells there. DEFAULT DECISION (laneQ, '
              'protocol rule 5): the exact faces are the default of '
              'hydrostatic_3D (re-pinned) and NOT of the 3D droplet presets: on '
              'the main benchmark every exact-area arm fails the better-l2-AND-'
              'tail rule because the pinned score is laneG\'s bump / over-decay '
              'cancellation (full table in the laneQ log, section 4). '
              'cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py '
              'runs every arm through the axis.',
        options=(
            _opt(None, 'The legacy source of the path: e_star_cache where the '
                 'retopology builds the fan cache (connectivity delaunay, '
                 'dual_only), p_ij_ring on the cache-less 3D paths '
                 '(dual_only_bare, delaunay_material, periodic, frozen, a '
                 'setup mesh); 2D shared_vd_2d / min_image_2d, 1D the interval',
                 'validated',
                 'ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize step 5b',
                 'every pin before laneQ; bit-identical to the explicit value '
                 'it resolves to'),
            _opt('e_star_cache', '3D: cached e_star fan areas of every edge at '
                 'an interior vertex from batch_e_star(orient=True) at the last '
                 'retopology (hull vertices read the ring walk); built by '
                 'connectivity delaunay / dual_only only', 'validated',
                 'hyperct/ddg/_operators.py:batch_e_star',
                 'all 3D pins. NOT linearly precise: laneJ measured per-edge '
                 'difference to p_ij median 0.125 / max 0.625 (box), closure '
                 'residual ~1 % at every droplet interface vertex, linear-precision '
                 'error 3-25 %; it carries ~79 % of the 3D static floor '
                 'retopology excess (7.274172e-05 vs 6.2839e-05 on p_ij, frozen '
                 'floor 6.0153e-05) and part of the dynamic outward bump '
                 '(final inflation 1.87 % -> 0.82 % R0 on p_ij) - but p_ij alone '
                 'scores l2 0.28713 vs 0.24811 (laneG cancellation exposed), so '
                 'no flip. laneQ 2026-10-05 (full outer mesh, every arm a '
                 'preset.replace through this axis): the droplet keeps this '
                 'value; the A.5.b floor 7.2741338970e-05 and the full run (l2 '
                 '0.24811443136179492 / tail 0.0841737962816189) reproduce the '
                 'pins to the bit; the arms p_ij_simplex / p_ij_simplex + '
                 'redistribute_mass=False / + projection_every=2 score l2 '
                 '0.28653 / 0.32625 / 0.34581 against the cache arms 0.24811 / '
                 '0.26658 / 0.28898 (sign decomposition in the log): the exact '
                 'faces cut the early bump (q2 +0.245 -> +0.093) and halve the '
                 'final inflation (1.87 % -> 0.83 % R0), which exposes the '
                 'genuine over-decay, so no arm passes the flip rule. Retopology '
                 '407 ms against 231 ms with the exact cache on the 2/2 droplet '
                 '(batch_e_star 187 ms, the kernel 11 ms)', dims=(3,)),
            _opt('p_ij_simplex', '3D: the exact barycentric dual face of EVERY '
                 'directed edge of every vertex (hull included), cached at the '
                 'retopology by hyperct.ddg.simplex_dual_face_areas in one '
                 'vectorised pass over HC._simplices: per simplex '
                 '|T| (grad phi_j - grad phi_i) / 4 = the two '
                 'barycentric-subdivision triangles at the edge, flat '
                 'tetrahedra oriented by their neighbours; no fan walk, so no '
                 'fan-failure promotion', 'validated',
                 'hyperct/ddg/_dual_volume.py:simplex_dual_face_areas',
                 'laneQ 2026-10-05: antisymmetric to the bit, closure of every '
                 'interior cell 1.6e-16 and linear precision 2e-15 on the '
                 'droplet mesh with its 44 flat tetrahedra, hull half cells '
                 'closed against the box-face normals to 1.7e-16 (82 of the 98 '
                 'hull vertices touch a flat tetrahedron), equal to laneJ\'s '
                 'per-edge polygon to 2.1e-15 on every link and to laneT\'s '
                 'per-tetrahedron quads to 1.3e-15 on every link without a flat '
                 'tetrahedron (on the 179 of 6220 directed hull-hull links with '
                 '1 or 2 flat tetrahedra the quads\' per-piece sign rule '
                 'quad . d_ij > 0 is undefined and differs by up to 163 %); '
                 '11 ms per call on 2540 tetrahedra against 187 ms for '
                 'batch_e_star (hyperct test_dual_face_areas.py, 14 tests incl. '
                 'the lattice hull beside flat tetrahedra; '
                 'test_edge_area_source.py). Linear-pressure force on the '
                 'droplet interface cells 1e-13 V|g| (cache: up to 62, ring '
                 'walk: up to 47). DEFAULT of hydrostatic_3D (pins re-measured '
                 'and re-pinned: refinement 1 peak / KE at 40 t_ac move in '
                 'round-off only, the refinement 2 remap arm by -2.2 % / -2.6 % '
                 '= the removed hull-edge tie of laneT; 22 against 55 ms per '
                 'step). Hagen-Poiseuille 3D with pressure_flux=centred '
                 '(refinement 1, 600 steps): l2 0.056658 and radial velocity '
                 '2.5e-17 against 0.082115 / 6.3e-3 on the fan cache and '
                 '0.056561 / 1.3e-5 on the ring walk, 44 s against 58 / 152 s. '
                 'Droplet: A.5.b floor 6.2838104071e-05 (-13.6 % against the '
                 'cache, 0.36 against 0.56 s/step); full run l2 0.28653 / tail '
                 '0.08825 / R_max_end 0.010083 / mass 8.6e-14 at 0.34 s/step '
                 '(cache 0.554), NOT adopted there (flip rule, see the axis '
                 'notes). dam_break_3D smoke: the shipped preset aborts with '
                 'NaN after 17 steps on the fan cache (HEAD library too), '
                 'after 156 with the exact cache, after 91 per edge: the case '
                 'blows up on every source', dims=(3,)),
            _opt('p_ij', '3D: no cache; every edge built on demand from the '
                 'tetrahedra around it (stress._dual_area_vector_3d_simplex: '
                 'the DEC p_ij polygon with ring order and face vertices read '
                 'from HC._simplices, open chain through the edge midpoint on '
                 'a hull edge). The same face as p_ij_simplex, uncached',
                 'opt-in',
                 'ddgclib/operators/stress.py:_dual_area_vector_3d_simplex',
                 'laneQ 2026-10-05: equal to the p_ij_simplex cache to '
                 'round-off on every mesh tried (test_edge_area_source.py); '
                 'replaces the heuristic face selection of p_ij_ring (laneJ '
                 'F4b) and the hull-edge tie (laneT). Same numbers as '
                 'p_ij_simplex up to the run\'s amplification of round-off '
                 '(A.5.b floor 6.2838104071e-05 in both; hydrostatic 3D '
                 'refinement 1 KE at 40 t_ac 6.2063651563e-06 in both; '
                 'Poiseuille centred arm l2 0.056612 against 0.056658; full 3D '
                 'droplet l2 0.28653087622018630 / tail 0.08825446501866843 '
                 'against 0.28653087631979274 / 0.08825446509925354), at the '
                 'cost of a per-edge Python construction: force evaluation of '
                 'the 2/2 droplet 603 ms against 70 ms (1.07 against 0.36 '
                 's/step on the A.5.b floor, 1.07 against 0.34 on the full '
                 'droplet)', dims=(3,)),
            _opt('p_ij_ring', '3D: no cache; the legacy ring walk over the '
                 'shared dual vertices with the nearest-barycentre face '
                 'heuristic (dual_area_vector before laneQ; what every '
                 'cache-less 3D path read)', 'broken',
                 'ddgclib/operators/stress.py:_dual_area_vector_3d_p_ij',
                 'test_stress.py p_ij linear-precision tests (box/ball: 1e-18). '
                 'laneJ: the face vertex is chosen as the common neighbour '
                 'nearest the midpoint of two tet barycentres; on 743 of 5193 '
                 'directed droplet edges that picks a non-face vertex, giving '
                 'the 2.6 % closure residuals laneG attributed to the pressure '
                 'side. 4.1x wall cost (2.05 vs 0.50 s/step). laneT 2026-10-02, '
                 'BOUNDARY edges: there the ring also holds the edge midpoint '
                 'and the two boundary-face barycentres, and the nearest-'
                 'barycentre rule puts an interior face barycentre next to the '
                 'midpoint whenever it is not farther than the boundary one; '
                 'on the builder lattice the two are equidistant, so round-off '
                 'decides: 30 of the 56 boundary edges at the 9 free-surface '
                 'vertices of the refinement 2 hydrostatic column are off by '
                 'up to 37 % against the per-tetrahedron sum (edges with an '
                 'interior endpoint: 1127, exact to 8e-16). A 1e-15 shift of '
                 'the interior vertices moves the KE of that column by 2e-6 '
                 'after 150 steps on FIXED connectivity (6e-12 at refinement '
                 '1, which has no such edge). Kept selectable so that laneJ\'s '
                 'p_ij arm and the pre-laneQ numbers of the cache-less paths '
                 'reproduce: laneQ measured the Poiseuille centred arm '
                 '0.05656136676066496 / radial 1.258e-05 (= laneH / laneT), '
                 'the hydrostatic_3D pins before laneQ to the bit, the A.5.b '
                 'floor 6.2838085762e-05 (laneJ 6.283858835e-05 on the lossy '
                 'mesh); 2.2 s/step on the A.5.b floor (force evaluation 1.8 s '
                 'against 0.07 s with a cache)', dims=(3,)),
            _opt('shared_vd_2d', '2D (reported, not selectable): segment between '
                 'the two dual vertices shared by v_i and v_j, oriented outward '
                 'by the axis area_orientation (A_ij . d_ij > 0 since laneO)',
                 'validated',
                 'ddgclib/operators/stress.py:dual_area_vector (2D branch)',
                 'all 2D pins (batch_e_star raises for dim != 3). DEFECT found '
                 'by laneH 2026-10-01, FIXED by laneO 2026-10-05 (axis '
                 'area_orientation; the old rule is its broken value '
                 'dual_midpoint): the vector was oriented away from x_i as seen '
                 'from the midpoint of the dual segment; when the barycentres of '
                 'the two triangles at the edge subtend more than 180 degrees at '
                 'x_i that points AGAINST the edge. On a jittered sheared '
                 'Delaunay mesh 6 of 806 vectors were flipped, the closure '
                 'residual of an interior cell 0.23 and the centred force of a '
                 'linear pressure off by up to 17.6 V |g|; oriented by '
                 'A_ij . d_ij > 0 the vector equals the simplex-cache reference '
                 '(stress.simplex_area_vectors) to 1e-14 on every mesh tried, '
                 'cells close to round-off and A_ij = -A_ji exactly '
                 '(test_area_orientation.py). Census of the fast suite before '
                 'the fix: 2341 flipped vectors of 1.17e6 2D calls in 8 tests; '
                 'the three pins that read one were re-measured and re-pinned '
                 '(laneO log, section 3), every other 2D pin and the full 2D '
                 'droplet baseline are bit-identical',
                 dims=(2,)),
            _opt('min_image_2d', '2D periodic (reported, not selectable): '
                 'minimum-image rebuild of the dual segment, sign by the axis '
                 'area_orientation', 'experimental',
                 'ddgclib/operators/stress.py:dual_area_vector (periodic branch)',
                 'd_ij is NOT min-imaged (06_known_issues). laneO 2026-10-05: '
                 'the sign defect of shared_vd_2d was here too (64 vectors '
                 'against the minimum-image edge in 10 steps of the 2D '
                 'shearing plate; fixed by the same rule), and a defect of its '
                 'own is NOT fixed: on a sheared, jittered periodic_rectangle '
                 'after retopologize_periodic the two endpoints of 62 of 886 '
                 'directed edges (61 crossing the seam) build DIFFERENT dual '
                 'segments (the common neighbours are min-imaged about x_i, so '
                 'the two sides can pick different periodic images) and 2 '
                 'return a zero vector from one side only '
                 '(test_area_orientation.py::TestPeriodicBranch)', dims=(2,)),
        ),
    ),
    MethodAxis(
        name='boundary_rule', title='Topological boundary detection',
        group='dual geometry', default='boundary_from_simplices',
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
