"""Named configurations = what the shipped dynamic cases actually run.

Every preset here is consumed by its case runner (so the runner and the
registry cannot drift) and, where the case has a pinned number, is proven
bit-identical to the pre-wrapper hand-written partials by
``ddgclib/tests/test_methods.py``.  The ``label`` says which runner /
policy string it corresponds to; the ``notes`` carry the pinned numbers
and the lane log or audit entry that justify it.

Cases whose runners are hand-rolled loops (capillary_rise,
dynamic_caprise_tube, liquid_bridge_approach) have NO preset on purpose: a
preset would describe the nearest library-equivalent configuration, not
what runs.  They are listed in ``METHODS.md`` §4 with status "hand-rolled".
"""
from __future__ import annotations

from ddgclib.methods._config import SolverMethods

__all__ = ['PRESETS', 'preset']

_OD = 'cases_dynamic/oscillating_droplet/'
_DB = 'cases_dynamic/dam_break/'
_EB = 'cases_dynamic/electrolysis_bubble/'
_SP = 'cases_dynamic/shearing_plate_droplet/'
_HP = 'cases_dynamic/Hagen_Poiseuile'
_HY = 'cases_dynamic/Hydrostatic_column/'

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
        frozen_set='membership', redistribute_mass=True,
        label=_DB + 'dam_break_2D.py',
        notes='laneF: remap ON survives reconnection (plain Delaunay blows up '
              'at the first flip, KE x28). Hydrostatic per-phase mass '
              'preload IC, alpha_art 0.3 baked into PhaseProperties.mu, '
              'gravity via the setup closure. Open blocker: air sliver-cell '
              'F/m ejection. laneL: frozen_set=membership. On the shipped '
              'run (1585 steps) the final state is bit-identical to '
              '.replace(frozen_set="hull") (no vertex leaves the tank). In '
              'the two ejection configurations (alpha_art 0.2; alpha_art '
              '0.5 to t = 0.45 s) the arms are bit-identical until the first '
              'vertex leaves the tank (step 1267 / 3002). Then hull releases '
              'the walls in the next step (all 32 move) and aborts after '
              '1280 / 3022 steps; membership keeps the 32 wall vertices '
              'frozen and in place, but the fluid still blows up and the '
              'run aborts with the same QhullError after 1303 / 3030 steps '
              '(numbers of fix round 1, where the tied simplex vote became '
              'deterministic; equal in every process). The walls are not '
              'what fails there.',
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
        connectivity='delaunay', frozen_set='membership',
        viscous_flux='simplex_gradient',
        label=_HP + '/Hagen_Poiseuile_2D.py',
        notes='Developing Lagrangian channel flow on src/_setup.py:'
              'setup_poiseuille_developing: pressure G (L - x) prescribed and '
              're-imposed every step (DirichletPressureBC over HC.V, nodal), '
              'OutletBufferedDeleteBC, PeriodicInletBufferedBC (upstream buffer '
              'of prescribed plug motion), PositionalNoSlipWallBC; bV = walls. '
              'laneH 2026-10-01, shipped run (Re_D 10, L 12, refinement 2, dt '
              '0.05, 2400 steps = 11.8 t_dev): dual-volume weighted l2 from the '
              'developed profile on 6 <= x <= 12 1.086e-2 (last quarter mean '
              '1.089e-2, max 1.167e-2), u_max 0.15085 (0.15), largest '
              'transverse velocity 5.9e-17 in the window and 1.1e-16 over all '
              'free vertices at every step, 0 vertices outside the walls, 106 '
              'of 106 wall vertices frozen and unmoved, 473 -> 478 vertices, '
              'mass flux in / out 0.8333 / 0.8681 rho U D over 6 inlet periods. '
              'What is left is resolution (7 fluid rows; the error falls by '
              'about 3 per refinement). Arm '
              'viscous_flux="two_point": l2 1.04, 26 vertices outside. '
              'workers=None: serial is 2.2 times faster than 20 workers at 200 '
              'vertices (38 against 86 ms per step). Pin: '
              'test_case_hagen_poiseuille.py PIN_2D_L2 (L 3, refinement 1, 500 '
              'steps). The laneL wall-collapse reproducer (test_frozen_set.py) '
              'is this preset with viscous_flux="two_point" on the setup '
              'before laneH (setup_poiseuille_2d_lagrangian).',
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
        connectivity='delaunay', frozen_set='membership',
        pressure_flux='simplex_gradient', viscous_flux='simplex_gradient',
        label=_HP + '_3D/Hagen_Poiseuile_3D.py',
        notes='Developing Lagrangian pipe flow, same construction as the 2D '
              'channel (cases_dynamic/Hagen_Poiseuile/src/_setup.py:'
              'setup_poiseuille_developing with dim=3). Library per-step '
              'Delaunay with walls frozen by membership replaces the case-local '
              'retopologize_cylinder (laneH: it froze the whole hull, the inlet '
              'cap included, and kept no simplex cache; audit M2). '
              'pressure_flux=simplex_gradient because the centred flux reads '
              'the batch_e_star area cache in 3D, which is not linearly '
              'precise: radial velocity 6.3e-3 against 2.6e-18 (refinement 1, '
              '600 steps; l2 of that arm 0.0821, the same in every process '
              'since laneT, 0.0811 to 0.1107 before). laneH '
              '2026-10-01, shipped run (Re_D 2, L 4, refinement 2 = 16-sided '
              'pipe, dt 0.01, 1000 steps = 11.6 t_dev): l2 from the developed '
              'profile of the circular pipe on 2 <= z <= 4 1.99e-2 (last '
              'quarter mean 1.91e-2, max 2.11e-2), u_max 0.1977 (0.2), largest '
              'transverse velocity 3.3e-17 in the window and 1.6e-15 over all '
              'free vertices at every step, 0 vertices outside, 336 of 336 '
              'wall vertices frozen and unmoved, 845 -> 928 vertices. Left: '
              'resolution and the polygonal wall (cross-section 2.8 % below '
              'pi R^2). Pin (slow): test_case_hagen_poiseuille.py PIN_3D_L2 '
              '(L 2, refinement 1, 300 steps).',
    ),
    # ------------------------------------------------------------------
    # Hydrostatic column (single phase, EOS, free surface)
    # ------------------------------------------------------------------
    'hydrostatic_1D': SolverMethods(
        dim=1, phases='single', integrator='symplectic_euler',
        connectivity='delaunay',
        label=_HY + 'Hydrostatic_1D.py',
        notes='1D rebuild = sorted chain (no flips; equal to round-off to the '
              'hand-rolled loop it replaces). boundary_filter = bottom vertex, '
              'top vertex free, Tait n = 1, c0 = 10 sqrt(g H), gravity as '
              'body_force, mu = 0.5 rho c0 dx. 33 vertices, 200 t_ac: from '
              'uniform density the column rings at its fundamental mode, KE '
              'decay 0.0389 / t_ac (viscous theory 0.0386), max|u| 0.89 -> '
              '1.7e-2 (envelope of the last 4 t_ac; last sample 1.44e-2); '
              'from the equilibrium masses 2.8e-6 -> 7.0e-8, '
              'integrated L2 0.28 Pa. Error against refinement 3..6: 1.04 / '
              '0.220 / 0.0448 / 0.0094 Pa (laneP).',
    ),
    'hydrostatic_2D': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='dual_only',
        label=_HY + 'Hydrostatic_2D.py',
        notes='No-slip bottom and side walls (boundary_filter), free surface. '
              '145 vertices, 200 t_ac: max|u| 0.248 -> 1.1e-4 from uniform '
              'density (integrated L2 49.9 Pa = 5.1e-3 rho g H, interior 5.8 '
              'Pa), 7.0e-6 -> 4.2e-9 from the equilibrium masses (L2 48.6, '
              'interior 0.23). L2 against refinement 2..4: 136 / 48.6 / 17.2 '
              'Pa. Reconnecting arm = .replace(connectivity=delaunay_material, '
              'remap=conservative, redistribute_mass=True): stable, 1.2e-4 / '
              'noise floor 2.3e-6. Needs the artificial viscosity: with the '
              'viscosity of water the drop exceeds c0 at 64 t_ac (laneP).',
    ),
    'hydrostatic_2D_periodic': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='dual_only',
        label=_HY + 'Hydrostatic_2D_periodic.py (free-slip side walls)',
        notes='NOT periodic connectivity (measured unusable for a single-phase '
              'EOS column, see the periodic option): the side vertices slide '
              'on FreeSlipWallBC, only the bottom is frozen, so the solution '
              'is one-dimensional. 145 vertices, 200 t_ac: max|u| 0.256 -> '
              '1.1e-6 from uniform density (KE decay 0.126 / t_ac, viscous '
              'theory 0.125), 6.6e-6 -> 3.4e-11 from the equilibrium masses; '
              'L2 against refinement 2..4: 144 / 49.8 / 17.4 Pa, interior '
              '0.90 / 0.22 / 0.051 (laneP).',
    ),
    'hydrostatic_3D': SolverMethods(
        dim=3, phases='single', integrator='symplectic_euler',
        connectivity='dual_only_bare',
        label=_HY + 'Hydrostatic_3D.py',
        notes='dual_only_bare, not dual_only: the 3D branch of dual_only '
              'zeroes the dual volume of frozen vertices, so wall cells would '
              'read P0 (measured L2 1.5e4 Pa). Here: wall half cells, p_ij '
              'dual faces, boundary_filter = walls, free top. 189 vertices, '
              '100 t_ac: max|u| 0.144 -> 2.4e-4 from uniform density, 3.2e-5 '
              '-> 5.4e-7 from the equilibrium masses (integrated L2 0.99 Pa; '
              'refinement 1: 2.15 Pa). 3D cell integrals of ddgclib.analytical '
              'are point value x volume (laneP).',
    ),
}


def preset(name: str) -> SolverMethods:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: "
                       f"{sorted(PRESETS)}") from None
