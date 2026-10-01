"""Named configurations = what the shipped dynamic cases actually run.

Every preset here is consumed by its case runner (so the runner and the
registry cannot drift) and, where the case has a pinned number, is proven
bit-identical to the pre-wrapper hand-written partials by
``ddgclib/tests/test_methods.py``.  The ``label`` says which runner /
policy string it corresponds to; the ``notes`` carry the pinned numbers
and the lane log or audit entry that justify it.

Cases whose runners are hand-rolled loops (Hydrostatic_column,
capillary_rise, dynamic_caprise_tube, liquid_bridge_approach) have NO
preset on purpose: a preset would describe the nearest library-equivalent
configuration, not what runs.  They are listed in ``METHODS.md`` §4 with
status "hand-rolled".
"""
from __future__ import annotations

from ddgclib.methods._config import SolverMethods

__all__ = ['PRESETS', 'preset']

_OD = 'cases_dynamic/oscillating_droplet/'
_DB = 'cases_dynamic/dam_break/'
_EB = 'cases_dynamic/electrolysis_bubble/'
_SP = 'cases_dynamic/shearing_plate_droplet/'
_HP = 'cases_dynamic/Hagen_Poiseuile'

PRESETS: dict[str, SolverMethods] = {
    # ------------------------------------------------------------------
    # oscillating droplet 2D
    # ------------------------------------------------------------------
    'oscillating_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap='conservative',
        redistribute_mass=True, split_method='neighbour_count',
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap')",
        notes='Default since laneE (2026-07-29). Full run refine 3/3: l2 '
              '0.17479361640597058 / tail 0.9998967874595965 '
              '(baselines/baseline_oscillation.json; reproduced through the '
              'wrapper 2026-09-25). Fast mirror refine 2/2: l2 0.054514 < '
              '0.0600 (TestOscillationEnvelopeRegression2D).',
    ),
    'oscillating_droplet_2D_dual_only': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='dual_only')",
        notes='lane5 default before laneE: l2 0.17857 / tail 0.99925. '
              'Guard: TestDualOnlyRetopoPolicy2D.',
    ),
    'oscillating_droplet_2D_bare_delaunay': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='delaunay')",
        notes='MEASURED WORSE for dynamics (per-step Delaunay KE pump: l2 '
              '0.48992 / tail 1.72505, lane5). Kept because the static '
              'floor tests exercise exactly this path with u=0.',
    ),
    'oscillating_droplet_2D_projection2': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap='conservative', projection_every=2,
        redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap_p2')",
        notes='laneH opt-in: l2 0.03795682994323827 (-78%), l2_two_fluid '
              '0.02193, tail 1.3973 (tail gate uncalibrated). NOT the default.',
    ),
    'static_droplet_floor_2D': SolverMethods(
        dim=2, phases='multi', integrator='euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label='ddgclib/tests/test_case_oscillating_droplet.py::'
              'TestStaticDroplet2DRetopologyFloor (u=0 every step)',
        notes='Pinned floors 2.3748568e-03 (step 0) / 2.2716938e-03 (post-retopo).',
    ),
    'static_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only_bare', redistribute_mass=False,
        label=_OD + 'static_droplet_2D.py',
        notes='Bare dual-only refresh (ddgclib.methods._retopo.bare_dual_refresh, '
              'formerly the case-local _dual_only_retopo closure): no '
              'mps.refresh, no redistribution, no EOS update, so the pressure '
              'field is frozen at its setup value (audit F11). Pinned summary '
              '1.1847162859108737e-03, mass 0.0.',
    ),
    # ------------------------------------------------------------------
    # oscillating droplet 3D
    # ------------------------------------------------------------------
    'oscillating_droplet_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        label=_OD + "oscillating_droplet_3D.py (retopo_policy_3d='dual_only')",
        notes='laneB default: l2 0.24811340819647862 / tail 0.08410 '
              '(baselines/baseline_oscillation_3d.json). laneG: the score '
              'is a bump/over-decay cancellation; do not read as inflation.',
    ),
    'oscillating_droplet_3D_delaunay': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_OD + "oscillating_droplet_3D.py --retopo delaunay",
        notes='MEASURED WORSE: l2 1.5244561707801316 / tail 0.47067 (laneB). '
              'laneI (2026-09-25) fixed the never-invalidated 3D interface apex '
              'cache and re-ran this config: bit-identical (the run never flips '
              'interface triangles), so the rejection is confirmed free of the '
              'stale-cache effect. delaunay+remap likewise stays 1.8734809234034775.',
    ),
    'static_droplet_floor_3D': SolverMethods(
        dim=3, phases='multi', integrator='euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label='ddgclib/tests/test_case_oscillating_droplet.py::'
              'TestStaticDroplet3DRetopologyFloor (u=0 every step)',
        notes='Pinned floors 6.0153e-05 (step 0) / 7.274172e-05 (plateau, laneA).',
    ),
    # ------------------------------------------------------------------
    # dam break
    # ------------------------------------------------------------------
    'dam_break_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap='conservative',
        redistribute_mass=True,
        label=_DB + 'dam_break_2D.py',
        notes='laneF: remap ON survives reconnection (plain Delaunay blows up '
              'at the first flip, KE x28). Hydrostatic per-phase mass '
              'preload IC, alpha_art 0.3 baked into PhaseProperties.mu, '
              'gravity via the setup closure. Open blocker: air sliver-cell '
              'F/m ejection.',
    ),
    'dam_break_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        label=_DB + 'dam_break_3D.py',
        notes='Frozen connectivity (was skip_triangulation=True at integrator '
              'level). Not re-run since laneF; April outputs only.',
    ),
    # ------------------------------------------------------------------
    # electrolysis bubble
    # ------------------------------------------------------------------
    'electrolysis_bubble_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_EB + 'electrolysis_bubble_2D.py',
        notes='Per-step Delaunay without remap (the configuration measured '
              'worse on the droplet). Gravity + NaN guard in the setup dudt '
              'wrapper, WallClampBC, gas mass injection in the callback. '
              'Unvalidated; only pin is 5-step per-phase mass drift <= 1.94e-15.',
    ),
    'electrolysis_bubble_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_EB + 'electrolysis_bubble_3D.py',
        notes='UNSTABLE: gas phase lost entirely by t~1.1e-4 s (audit). Same '
              'wiring as 2D. (The stale 3D interface apex cache, audit T1, is '
              'fixed since laneI; this case has not been re-run.)',
    ),
    'electrolysis_bubble_fritz_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=False,
        label=_EB + 'electrolysis_bubble_fritz_2D.py (run_short_dynamics)',
        notes='80-step smoke on the Fritz-shaped bubble. redistribute_mass was '
              'left unbound in the case partial, so the integrator default '
              '(False) applied; recorded explicitly here.',
    ),
    # ------------------------------------------------------------------
    # shearing plate droplet (periodic multiphase)
    # ------------------------------------------------------------------
    'shearing_plate_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='periodic', periodic_axes=(0,), redistribute_mass=True,
        label=_SP + 'shearing_plate_droplet_2D.py, _run_short_2D.py',
        notes='retopologize_multiphase_periodic (formerly the case-local '
              '_make_periodic_multiphase_retopo closure): ghost Delaunay + '
              'refresh + redistribution, no remap/cadence. domain_bounds from '
              'the setup params. UNSTABLE: interface lost by t~0.044 s, '
              '|u|max 65x U_wall (audit).',
    ),
    'shearing_plate_droplet_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='periodic', periodic_axes=(0, 2), redistribute_mass=True,
        label=_SP + 'shearing_plate_droplet_3D.py, _run_short_3D.py',
        notes='Main 3D runner crashes in setup (outer-vertex rescale collides '
              'with a droplet vertex key); short runner stalled (audit).',
    ),
    # ------------------------------------------------------------------
    # Hagen-Poiseuille (single phase)
    # ------------------------------------------------------------------
    'hagen_poiseuille_2D': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='delaunay', workers=20,
        label=_HP + '/Hagen_Poiseuile_2D.py',
        notes='Lagrangian channel with boundary_filter=walls (passed at build '
              'time), PositionalNoSlipWallBC + OutletBufferedDeleteBC + '
              'PeriodicInletBC. pressure_model=None so workers>1 is safe. '
              'UNSTABLE: wall collapse by hull re-tagging (audit C1/M4).',
    ),
    'hagen_poiseuille_2D_eulerian': SolverMethods(
        dim=2, phases='single', integrator='euler_velocity_only',
        connectivity='delaunay',
        label=_HP + '_2D_Eulerian/Hagen_Poiseuile_2D_Eulerian.py',
        notes='Fixed mesh (velocity-only); default per-step Delaunay on the '
              'unmoved mesh; whole hull frozen. Developing Poiseuille profile; '
              'no pinned number (figures 2026-02-17).',
    ),
    'hagen_poiseuille_3D': SolverMethods(
        dim=3, phases='single', integrator='symplectic_euler',
        connectivity='custom', workers=8,
        label=_HP + '_3D/Hagen_Poiseuile_3D.py (retopologize_cylinder)',
        notes='Case-local filtered Delaunay for the cylinder (never updates '
              'HC._simplices; freezes every hull vertex incl. the inlet cap, '
              'audit M2). STALLED: no interior vertices near mid-tube after '
              '3000 steps.',
    ),
}


def preset(name: str) -> SolverMethods:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: "
                       f"{sorted(PRESETS)}") from None
