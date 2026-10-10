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
_CR = 'cases_dynamic/capillary_rise/'
_C2D = 'cases_dynamic/cube2droplet/'
_LB = 'cases_dynamic/liquid_bridge_'
_BD = 'cases_dynamic/bc_demo/'

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
              '0.17439096487276182 / tail 0.9998871416222597 / mass drift '
              '1.48e-14 (baselines/baseline_oscillation.json, laneB '
              '2026-10-05 on the full outer mesh, 317 vertices; the setup '
              'choice box_shift="evict" reproduces the pre-laneB baseline l2 '
              '0.17479361640597058 / tail 0.9998967874595965 on the lossy '
              '311-vertex mesh to the bit). Fast mirror refine 2/2: l2 '
              '0.054618 < 0.0600 (TestOscillationEnvelopeRegression2D; '
              '0.054514 before laneB).',
    ),
    'oscillating_droplet_2D_dual_only': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='dual_only')",
        notes='lane5 default before laneE: l2 0.17857 / tail 0.99925 (lossy '
              'pre-laneB mesh); laneB 2026-10-05, full outer mesh: l2 '
              '0.17946703687459944 / tail 0.9998387858074947. '
              'Guard: TestDualOnlyRetopoPolicy2D.',
    ),
    'oscillating_droplet_2D_bare_delaunay': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='delaunay')",
        notes='MEASURED WORSE for dynamics (per-step Delaunay KE pump: l2 '
              '0.48992 / tail 1.72505, lane5). Kept because the static '
              'floor tests exercise exactly this path with u=0. laneO '
              '(2026-10-05): this is the one droplet configuration whose '
              'integrated vertices read flipped 2D area vectors (454 in 300 '
              'steps at refinement 2); with the correct orientation the full '
              'run gives l2 0.5038096226631333 / tail 1.2281376515706395 '
              '(the pre-laneO library reproduces 0.48991833470391266 / '
              '1.7250489596305962); still measured worse.',
    ),
    'oscillating_droplet_2D_projection2': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap='conservative', projection_every=2,
        redistribute_mass=True,
        label=_OD + "oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap_p2')",
        notes='laneH opt-in: l2 0.03795682994323827 (-78%), l2_two_fluid '
              '0.02193, tail 1.3973 (tail gate uncalibrated). NOT the default. '
              'laneB 2026-10-05, full outer mesh: l2 0.038841363169171125 '
              '(-78 % against the default 0.17439), l2_two_fluid 0.02306, '
              'tail 1.3936743069461104: the missing box corner was not the '
              'over-decay.',
    ),
    'static_droplet_floor_2D': SolverMethods(
        dim=2, phases='multi', integrator='euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label='ddgclib/tests/test_case_oscillating_droplet.py::'
              'TestStaticDroplet2DRetopologyFloor (u=0 every step)',
        notes='Pinned floors 2.3748568e-03 (step 0) / 2.2716938e-03 (post-retopo). '
              'laneB 2026-10-05 (full outer mesh): step 0 bit-identical, '
              'post-retopo 2.2716937806e-03 against 2.2716937802e-03 before '
              '(1.8e-10 relative), pinned digits unchanged.',
    ),
    'static_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only_bare', redistribute_mass=False,
        label=_OD + 'static_droplet_2D.py',
        notes='Bare dual-only refresh (ddgclib.methods._retopo.bare_dual_refresh, '
              'formerly the case-local _dual_only_retopo closure): no '
              'mps.refresh, no redistribution, no EOS update, so the pressure '
              'field is frozen at its setup value (audit F11). Pinned summary '
              '1.1672989414885857e-03 (interface radius drift over 100 '
              'steps), max KE normalised 8.79e-09, mass 0.0 (laneB '
              '2026-10-05, full outer mesh; the pre-laneB value '
              '1.1847162859108737e-03 is reproduced by the setup choice '
              'box_shift="evict").',
    ),
    # ------------------------------------------------------------------
    # oscillating droplet 3D
    # ------------------------------------------------------------------
    'oscillating_droplet_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        label=_OD + "oscillating_droplet_3D.py (retopo_policy_3d='dual_only')",
        notes='laneB (July) default: l2 0.24811340819647862 / tail 0.08410 '
              'on the lossy pre-2026-10-05 mesh (472 vertices, 96 walls, the '
              'box corner missing). laneB (2026-10-05, full outer mesh, 475 '
              'vertices, 98 walls): l2 0.24811443136179492 / tail '
              '0.0841737962816189 / R_max_peak 0.010790237633926668 / mass '
              'drift 4.8e-14 (baselines/baseline_oscillation_3d.json); the '
              'setup choice box_shift="evict" reproduces the old baseline in '
              'every key. laneG: the score is a bump/over-decay '
              'cancellation; do not read as inflation (unchanged by the '
              'corner: 4e-6 relative). laneQ 2026-10-05: edge_area_source '
              'stays None (= the batch_e_star fan cache, not linearly '
              'precise); the exact faces (.replace(edge_area_source='
              '"p_ij_simplex")) score l2 0.28653087631979274 / tail '
              '0.08825446509925354 with the final inflation halved (R_max at '
              't_end 0.010083 against 0.010187) and the early bump cut (q2 '
              '+0.245 -> +0.093), i.e. the cancellation exposed; with '
              'redistribute_mass=False as well l2 0.32625 / tail 0.00955 '
              '(one-signed over-decay, no overshoot, mass drift 1e-16); with '
              'projection_every=2 l2 0.34581 / tail 0.14696 (inflates). No '
              'arm passes the flip rule; measured through '
              'diagnose_3d_edge_area_source.py dynamic.',
    ),
    'oscillating_droplet_3D_delaunay': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label=_OD + "oscillating_droplet_3D.py --retopo delaunay",
        notes='MEASURED WORSE: l2 1.5244561707801316 / tail 0.47067 (laneB, '
              'July). laneI (2026-09-25) fixed the never-invalidated 3D interface apex '
              'cache and re-ran this config: bit-identical (the run never flips '
              'interface triangles), so the rejection is confirmed free of the '
              'stale-cache effect. delaunay+remap likewise stays 1.8734809234034775. '
              'laneB 2026-10-05 (full outer mesh): l2 1.5291100090540053 / '
              'tail 0.4594532578392181 / R_max_peak 0.011400164376344007, '
              'still measured worse.',
    ),
    'static_droplet_floor_3D': SolverMethods(
        dim=3, phases='multi', integrator='euler',
        connectivity='delaunay', remap=None, redistribute_mass=True,
        label='ddgclib/tests/test_case_oscillating_droplet.py::'
              'TestStaticDroplet3DRetopologyFloor (u=0 every step)',
        notes='Pinned floors 6.0153e-05 (step 0) / 7.274134e-05 (plateau; '
              'laneB 2026-10-05 on the full outer mesh, 475 vertices; laneA '
              'pinned 7.274172e-05 on the lossy 472-vertex mesh, reproduced by '
              'box_shift="evict"). Step 0 is bit-identical between the meshes. '
              'laneQ 2026-10-05: the plateau on the exact dual faces '
              '(.replace(edge_area_source="p_ij_simplex") or "p_ij") is '
              '6.2838104071e-05 (-13.6 %; the fan cache carries 79 % of the '
              'excess over the frozen-mesh floor 6.0153e-05), on the legacy '
              'ring walk ("p_ij_ring") 6.2838085762e-05; not flipped with the '
              'droplet default.',
    ),
    # ------------------------------------------------------------------
    # dam break
    # ------------------------------------------------------------------
    'dam_break_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap='conservative',
        frozen_set='membership', redistribute_mass=True,
        wall_clamp='project',
        label=_DB + 'dam_break_2D.py',
        notes='laneV 2026-10-07: wall_clamp="project" (the library WallClampBC '
              'on the four tank walls, gap 0.1 of the wall spacing): identity '
              'on every pinned run (no vertex leaves: refinement 3 at alpha '
              '0.3 / 0.2 / 0.1 bit-identical, 0 put-backs), and refinement 4 '
              'at alpha 0.3 completes its 3170 steps instead of ending with '
              'an air vertex 1.2e-6 m under the floor at step 2892 '
              '(diagnose_sliver_ejection.py --refine 4). laneF: remap ON survives reconnection (plain Delaunay blows up '
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
              'what fails there. laneF 2026-10-05: the ejection was not a '
              'sliver cell but two ledger defects at one flip (axes '
              'phase_ledger and face_closure, both defaults now): the shipped '
              'run is bit-identical (952d4544676ca366, no presence change '
              'along it), alpha_art 0.2 and 0.1 complete the horizon (1585 '
              'steps) with no vertex outside and |u|max 1.00 / 1.40 m/s; the '
              'toe event (the liquid tongue goes one cell thick at t = 0.180 '
              '/ 0.097 s) costs a transient: the released toe vertices are '
              'kicked by the interface pressure jump, KE of liquid plus '
              'interface 3.1e-3 -> 3.5e-2 J for ~0.01 s at alpha 0.2 '
              '(diagnose_sliver_ejection.py --alpha 0.2 / 0.1). '
              'split_method="simplex" keeps the tongue (opt-in, changes '
              'every run).',
    ),
    'dam_break_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        edge_area_source='p_ij_simplex', wall_clamp='project',
        label=_DB + 'dam_break_3D.py',
        notes='laneV 2026-10-07: wall_clamp="project" on the six box walls '
              '(identity on the pinned run: no vertex leaves). Frozen connectivity (was skip_triangulation=True at integrator '
              'level). laneF 2026-10-05: edge_area_source="p_ij_simplex" '
              '(the exact dual faces of laneQ). On the batch_e_star fan cache '
              'the shipped run blew up at step 4 (laneQ: NaN after 17): its '
              '1 % closure defect times the ABSOLUTE pressure 101325 Pa is '
              '0.26 N on a 2e-5 kg air cell, 1000x the body force on it '
              '(invisible at P0 = 0). With the exact faces the run still '
              'ejected at step 96 until the setup preloaded the hydrostatic '
              'masses on the vote labels (the criterion labels differ in 3D: '
              'p_liq 101203 to 120945 Pa at t = 0, the EOS clip) and zeroed '
              'the phases the vote gives no sub-volume. Full horizon (793 '
              'steps, refinement 2): KE_liq peak 4.079e-06 J at 0.0144 s, '
              '|u|max 0.062 m/s, |a|max 96 m/s^2 at step 0 decaying to 0.03, '
              'front +5.0 mm, mass drift -2.6e-15, liquid level 101547 -> '
              '101429 Pa, final-state digest 639c87c7700c2c71 '
              '(diagnose_sliver_ejection.py --dim 3). The column creeps '
              '(mu_l_eff 105.8 Pa s). split_method="simplex" is as clean '
              '(adc368cdb99b3f4a, |u|max 0.057) and not adopted. The 3D '
              'wall cells are zeroed (dual_only), so the measured liquid '
              'volume is 6.06e-5 of the 1.25e-4 m^3 column.',
    ),
    'dam_break_2D_no_air': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='delaunay',
        label=_DB + 'dam_break_2D_no_air.py',
        notes='Single-phase liquid column with a free surface (no air '
              'phase), setup_dam_break_single_phase: EOS pressure, gravity '
              'as body_force, boundary_filter = the tank walls (v.is_wall; '
              'the free-surface vertices are topological boundary vertices '
              'that advect), hull-frozen per-step Delaunay without a remap, '
              'i.e. the configuration laneK measured unstable with an EOS '
              '(the force builder warns). laneW 2026-10-05: wired to the '
              'preset (the runner built its partial and integrator kwargs by '
              'hand before); unvalidated, no pin, never scored.',
    ),
    'dam_break_3D_no_air': SolverMethods(
        dim=3, phases='single', integrator='symplectic_euler',
        connectivity='delaunay',
        label=_DB + 'dam_break_3D_no_air.py',
        notes='3D version of dam_break_2D_no_air (same wiring). The runner '
              'catches the integrator abort (3D free-surface corners produce '
              'large spurious forces) to keep the snapshots. Unvalidated, no '
              'pin.',
    ),
    # ------------------------------------------------------------------
    # electrolysis bubble
    # ------------------------------------------------------------------
    'electrolysis_bubble_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        wall_clamp='project',
        label=_EB + 'electrolysis_bubble_2D.py',
        notes='laneV 2026-10-07: wall_clamp="project" records the clamp the '
              'setup has applied since 2026-07 (top and bottom wall, gap 0.02 '
              'R0; the library WallClampBC now, bit-identical). laneG 2026-10-06: connectivity="dual_only" replaces the '
              'per-step Delaunay without remap. Static bubble (g = 0, no '
              'injection; the preload is the analytical state, jump gamma / '
              'R0 = 72 Pa; diagnose_static_bubble.py --dim 2): refinement '
              '2/3 (214 vertices), 1500 steps (4.7e-5 s): Delaunay 8065.6 Pa '
              'at the end (|u|max 2.47 m/s, KE_max 4.7e-3 J), dual_only '
              '72.83 Pa (KE_max 3.5e-9, |u|max 0.013), '
              '.replace(connectivity="delaunay", remap="conservative") 72.82 '
              'Pa (3.4e-9); refinement 1/2: Delaunay 75.84 Pa (KE_max 4.7e-8), '
              'dual_only 75.84 (bit-identical: no flip in the window), remap 76.02. '
              'Shipped horizon (6330 steps, '
              'gravity, injection): Delaunay: KE_max 4.3e-2 J, |u|max 3.24 m/s, the jump swinging between +17221 and -1302 Pa, gas volume 1.148 V_exact at the end; dual_only: KE_max 2.5e-4 J, |u|max 1.03 m/s, the jump 1964 / -783 / 1050 Pa at t = 4e-5 / 1.2e-4 / 2e-4 s (the bubble breathing under the injected mass, +12.2 % of volume at the end), gas mass at M0 + dm_dt t to 2.9e-14, liquid 2.9e-14; the 32 interface vertices kept in both. The remap is not used because it '
              'erases the liquid\'s compression response under the injection '
              '(the 3D preset notes). Gravity + NaN guard in the setup dudt '
              'wrapper, WallClampBC, gas mass injection in the callback '
              '(add_phase_mass). Pins: test_case_electrolysis_bubble.py. '
              'History (Delaunay preset): unvalidated; only pin 5-step '
              'per-phase mass drift <= 1.94e-15. '
              'laneB 2026-10-05: the setup mesh lacked 8 of 41 outer vertices '
              '(L 0.004, refinement 2) until the builder fix (setup choice '
              'box_shift, recorded in methods.json); shipped horizon (6330 '
              'steps) A/B digest hull = membership a8301121c7bf44ab on the full '
              'mesh (214 vertices, 16 walls, KE at the end 4.2625e-02 J, '
              '|u|max 3.32 m/s) against 38530636a343cf7a (206, 15, 6.212278e-03 '
              'J, 1.74 m/s) on the pre-laneB mesh with the current library '
              '(lane L recorded 9ed4c69378ac129a on the library of '
              '2026-10-01); unvalidated case, no reference ranks the two.',
    ),
    'electrolysis_bubble_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=True,
        wall_clamp='project',
        label=_EB + 'electrolysis_bubble_3D.py',
        notes='laneV 2026-10-07: wall_clamp="project" records the clamp the '
              'setup has applied since 2026-07 (bit-identical). laneG 2026-10-06: connectivity="dual_only" (the 3D droplet '
              'default) replaces the per-step Delaunay without remap. The '
              'audit\'s "gas phase lost by t~1.1e-4 s" does not reproduce on '
              'the library of lanes B and F: the shipped horizon (2292 steps, '
              'refinement 1/1, gravity, injection) keeps its 35 gas cells and '
              '26 interface vertices with the gas mass at M0 + dm_dt t to '
              'round-off. What the Delaunay preset did: at every flip the '
              'measured gas volume jumped by 1.5 % (4.7355e-9 <-> 4.8087e-9 '
              'm^3 at fixed positions) and the redistribution turned it into '
              'a uniform gas pressure jolt of +-1500 Pa on a 144 Pa Laplace '
              'jump (-4026 to +1104 Pa along the horizon). Static bubble '
              '(g = 0, no injection, diagnose_static_bubble.py; the preload '
              'is the analytical state, jump 2 gamma / R0 = 144 Pa): '
              'refinement 1/1 (95 vertices), 2000 steps (1.3e-4 s): Delaunay '
              '-4872 Pa at the end (swinging -7675 to +1893), KE_max 7.8e-7 J, '
              '|u|max 1.13 m/s; dual_only 224.1 Pa, 4.2e-9 J, 0.115 m/s; '
              '.replace(connectivity="delaunay", remap="conservative") 241.9 '
              'Pa, 4.6e-9 J, 0.113 m/s. Refinement 2/2 (475 vertices), 2000 '
              'steps (4.5e-5 s): Delaunay 1511.6 Pa (6.7e-9 J, 0.234 m/s), '
              'remap 167.5 Pa (2.9e-10 J), dual_only 165.4 Pa (2.6e-10 J, |u|max '
              '0.042 m/s, gas mass drift 1.1e-14). The drift of '
              'the kept arms is the relaxation of the polyhedral bubble (26 / '
              '98 interface vertices, 13 % / 6 % larger than the sphere) '
              'toward its discrete equilibrium jump, not a flip artifact. The '
              'remap arm is a measured DO-NOT with the injection: the '
              'projection of every call erases the liquid\'s compression '
              'response (laneH), so the bubble did not grow while its gas '
              'pressure followed K dm / m to 6493 Pa at the end of the horizon '
              '(dual_only: 535.1 Pa with the bubble +5.8 % in volume, the '
              'gas mass at M0 + dm_dt t to -2.6e-16, the liquid to -1.9e-15). '
              'Pins: test_case_electrolysis_bubble.py. '
              'History: laneB 2026-10-05, the setup mesh lacked 2 of 35 outer '
              'vertices (refinement 1, the box corner among them); 300-step '
              'A/B digest hull = membership 6bdbc9c542ffd6fc on the full mesh '
              '(Delaunay preset), d4e464d4974dbf5d (lane L\'s record, '
              'reproduced to the bit by box_shift="evict").',
    ),
    'electrolysis_bubble_fritz_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', remap=None, redistribute_mass=False,
        wall_clamp='project',
        label=_EB + 'electrolysis_bubble_fritz_2D.py (run_short_dynamics)',
        notes='80-step smoke on the Fritz-shaped bubble. redistribute_mass was '
              'left unbound in the case partial, so the integrator default '
              '(False) applied; recorded explicitly here. laneV: '
              'wall_clamp="project" records the two clamps of '
              'setup_fritz_dynamics (the library class now).',
    ),
    # ------------------------------------------------------------------
    # shearing plate droplet (periodic multiphase)
    # ------------------------------------------------------------------
    'shearing_plate_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='periodic', periodic_axes=(0,), remap='conservative',
        redistribute_mass=True,
        label=_SP + 'shearing_plate_droplet_2D.py, _run_short_2D.py',
        notes='retopologize_multiphase_periodic (formerly the case-local '
              '_make_periodic_multiphase_retopo closure): ghost Delaunay + '
              'refresh + redistribution; since laneG (2026-10-06) with the '
              'conservative remap (the same ledger closure as the Delaunay '
              'path, multiphase_rebuild_with_ledger). domain_bounds from '
              'the setup params. History: UNSTABLE, interface lost by '
              't~0.044 s, |u|max 65x U_wall (audit); laneB 2026-10-05: the '
              'setup mesh lacked 22 of 145 outer vertices until the builder '
              'fix (box_shift); on the full mesh the short window '
              '(refinement 3/3, t = 0.05 s, 1649 steps) blew up by t = 0.05 '
              '(|u| 295 U_wall), the rescale having deleted both droplet '
              'poles. laneG: the rescale is rescale_droplet_box (no vertex '
              'lost), the periodic rebuild measures the seam simplices with '
              'minimum-image coordinates and keeps one image per simplex '
              '(total dual volume 1.94 x the box before), the setup resets '
              'the outer masses on the periodic duals (the outer phase sat '
              'at -100 Pa: measured jump 106 Pa against gamma / R = 6). With '
              'remap=None the repaired case still goes unstable at t = '
              '0.042 s in the row next to the plates (|u| 2.4 U_wall at t = '
              '0.05 s, jump 12.4 Pa, droplet volume -0.9 %); with the remap '
              'the short window completes with the 32 interface vertices, '
              'jump 6.018 Pa, volume ratio 0.983992 (0.984008 at setup), '
              'first-row speed 0.59 U_wall, digest 3c66efcb929b9e1c '
              '(test_case_shearing_plate.py). A quiescent droplet holds the '
              'jump with the sum of forces on the free vertices at round-off '
              '(no seam force).',
    ),
    'shearing_plate_droplet_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='periodic', periodic_axes=(0, 2), remap='conservative',
        redistribute_mass=True,
        label=_SP + 'shearing_plate_droplet_3D.py, _run_short_3D.py',
        notes='Same wiring as 2D (laneG 2026-10-06). History: the main 3D '
              'runner crashed in setup (the uniform rescale put 11 outer '
              'vertices inside the shell and collided with droplet keys), '
              'the short runner stalled (audit). laneG: the setup builds '
              '(refinement 1/2: 306 vertices, 98 interface, 8 plate '
              'vertices) and the first steps run; known limit: the 3D dual '
              'faces of seam edges are built from unwrapped coordinates '
              '(stress.py has the minimum-image rebuild in 2D only), so the '
              '3D case is a setup + smoke pin, not a physical run.',
    ),
    # ------------------------------------------------------------------
    # Hagen-Poiseuille (single phase)
    # ------------------------------------------------------------------
    'hagen_poiseuille_2D': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='delaunay', frozen_set='membership',
        viscous_flux='simplex_gradient', wall_clamp='project',
        label=_HP + '/Hagen_Poiseuile_2D.py',
        notes='laneV 2026-10-07: wall_clamp="project" on the wall lines y = 0 '
              'and y = D (gap 0.1 of the wall spacing): identity on the pinned '
              'developing run (nothing leaves, l2 0.013084885355719682 in both '
              'arms); on the pre-laneH configuration of laneL (hull inlet, '
              'two-point flux, L 15, 3000 steps of 0.01) the vertices outside '
              'the walls go from 54 to 0 and the profile l2 on the downstream '
              'half from 0.554 to the value of the laneV log '
              '(diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000). '
              'Developing Lagrangian channel flow on src/_setup.py:'
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
              'before laneH (setup_poiseuille_2d_lagrangian). laneO '
              '(2026-10-05): the 2D area-vector orientation fix leaves the '
              'preset, its pins and the pressure_flux="simplex_gradient" arm '
              'bit-identical (the flipped vectors were at inlet-buffer '
              'vertices only); pressure_flux stays "centred": the '
              'simplex_gradient arm differs by 2e-8 in l2 and tail (better '
              'l2, worse tail mean), no measured gain.',
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
              'viscosity of water the drop exceeds c0 at 64 t_ac (laneP). '
              'laneO (2026-10-05): the preset and the remap arm are '
              'bit-identical under the 2D area-vector orientation fix; the '
              'convex-hull arm (connectivity=delaunay + remap) reaches 111.7 '
              'm/s at 2.2 t_ac instead of 42.1 at 3.0.',
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
        connectivity='dual_only_bare', edge_area_source='p_ij_simplex',
        label=_HY + 'Hydrostatic_3D.py',
        notes='dual_only_bare, not dual_only: the 3D branch of dual_only '
              'zeroes the dual volume of frozen vertices, so wall cells would '
              'read P0 (measured L2 1.5e4 Pa). Here: wall half cells, exact '
              'p_ij dual faces, boundary_filter = walls, free top. 189 '
              'vertices, 100 t_ac: max|u| 0.144 -> 2.4e-4 from uniform '
              'density, 3.2e-5 -> 5.4e-7 from the equilibrium masses '
              '(integrated L2 0.99 Pa; refinement 1: 2.15 Pa). 3D cell '
              'integrals of ddgclib.analytical are point value x volume '
              '(laneP). laneQ 2026-10-05: edge_area_source="p_ij_simplex" '
              '(the exact dual face of every edge, hull edges included, from '
              'hyperct.ddg.simplex_dual_face_areas) replaces the ring walk, '
              'whose hull-edge tie moved this column by 2e-6 under a 1e-15 '
              'shift (laneT). Refinement 1 pins move in round-off only (peak '
              '0.08910127097486757 -> 0.08910127097486756, KE at 40 t_ac '
              '6.206365156298652e-06 -> 6.2063651562987255e-06); the remap '
              'arm at refinement 2 (2 t_ac) moves from peak '
              '0.15658060026054665 / KE end 0.5431445985762776 to '
              '0.15305813130485327 / 0.528851067635385 (-2.2 % / -2.6 %, '
              'beyond the 1.25e-3 perturbation range: the 37 % hull-edge '
              'areas are gone). 2.5x faster per step (22 against 55 ms at '
              'refinement 1, 159 against 550 at refinement 2). '
              '.replace(edge_area_source="p_ij_ring") reproduces every '
              'pre-laneQ number of this preset to the bit.',
    ),
    # ------------------------------------------------------------------
    # static capillary rise (single phase, EOS, free surface with tension)
    # ------------------------------------------------------------------
    'capillary_rise_static_2D': SolverMethods(
        dim=2, phases='single', integrator='symplectic_euler',
        connectivity='dual_only', contact_line='energy_gradient',
        label=_CR + 'capillary_rise_2D.py',
        notes='laneI 2026-10-06. Slit of width 4 mm (r = 2 mm), water '
              '(theta 9.99 deg), 3 extruded unit cells (reservoir band '
              '8.3 mm below y = 0 + Jurin 3.67 mm), walls frozen by '
              'membership (boundary_filter), band on HydrostaticReservoirBC, '
              'contact vertices on AxialSlideBC, Tait n = 1 with c0 = 10 '
              'sqrt(g h_J), gravity as body_force, mu = alpha rho c0 dx. '
              'Measurements: laneI log.',
    ),
    'capillary_rise_static_3D': SolverMethods(
        dim=3, phases='single', integrator='symplectic_euler',
        connectivity='dual_only_bare', edge_area_source='p_ij_simplex',
        contact_line='energy_gradient',
        label=_CR + 'capillary_rise_3D.py',
        notes='laneI 2026-10-06. Round tube r = 2 mm (cylinder_volume '
              'cross-section, extruded), water, 3 unit cells (band 4.7 mm '
              '+ Jurin 7.33 mm); dual_only_bare with the exact p_ij faces '
              'as hydrostatic_3D (dual_only zeroes the wall half cells). '
              'Measurements: laneI log.',
    ),
    # ------------------------------------------------------------------
    # cube-to-droplet relaxation (laneX 2026-10-06 / 07: per-step Delaunay
    # + per-phase redistribution + the conservative remap; the historic
    # no-remap configuration is the runner arm 'bare' = .replace(remap=None))
    # ------------------------------------------------------------------
    'cube_to_droplet_2D': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', redistribute_mass=True,
        remap='conservative',
        label=_C2D + 'cube_to_droplet_2D.py',
        notes='laneX 2026-10-06. Square droplet (half side R = 0.01 m) '
              'relaxing in a 3R box under gamma = 0.01 N/m; Tait n = 1 '
              'with K_d 100 / K_o 125 Pa, NoSlipWallBC + '
              'AtmosphericPressureBC on the wall-adjacent outer vertices. '
              'The runners imported a module path that did not exist '
              '(cases_dynamic.Cube2droplet) since 2026-09-25; the setup '
              'bound multiphase_dudt_i and a **kwargs retopo closure by '
              'hand, without remap: that configuration (the runner arm '
              "'bare' = .replace(remap=None)) loses the droplet's bulk by "
              't = 0.4 s of the 1 s run (circularity 0, no phase-1 '
              'sub-volume left), the conservative remap keeps it and ends '
              'at +6.99 % of the Laplace jump; the dual_only refresh at '
              '-357 %. Measurements: laneX log.',
    ),
    'cube_to_droplet_2D_dual_only': SolverMethods(
        dim=2, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=False,
        label=_C2D + 'cube_to_droplet_2D_bc_comparison.py',
        notes='laneX 2026-10-06. Fixed connectivity, duals and per-phase '
              'pressures refreshed every step (the case-local '
              'dual_only_retopo_multiphase closure of the BC comparison '
              'and mass-redistribution runners until laneX).',
    ),
    'cube_to_droplet_3D': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='delaunay', redistribute_mass=True,
        remap='conservative',
        label=_C2D + 'cube_to_droplet_3D.py',
        notes='laneX 2026-10-06. Cube droplet, refinement 2; as the 2D '
              'preset (edge_area_source None = the fan cache of the '
              'Delaunay path). 2000 steps of 5e-5 s: with the conservative '
              'remap the integrated jump is +1.7710 Pa against 2 gamma / '
              'R_eq = 1.6120 (+9.86 %), sphericity 0.6124 -> 0.8677; without '
              "it (runner arm 'bare') the droplet loses its bulk by step "
              '1000; dual_only +41.2 % with the cube not relaxing '
              '(sphericity 0.6174). Measurements: laneX log.',
    ),
    'cube_to_droplet_3D_dual_only': SolverMethods(
        dim=3, phases='multi', integrator='symplectic_euler',
        connectivity='dual_only', redistribute_mass=False,
        label=_C2D + 'cube_to_droplet_3D.py --arm dual_only',
        notes='laneX 2026-10-07. The 3D fixed-connectivity arm of the A/B '
              '(as cube_to_droplet_2D_dual_only, dim 3): 2000 steps of '
              '5e-5 s at refinement 2 end at +2.2763 Pa against 2 gamma / '
              'R_eq = 1.6120 (+41.2 %) with the cube not relaxing '
              '(sphericity 0.6124 -> 0.6174); measured worse than the '
              'Delaunay + remap preset. Measurements: laneX log.',
    ),
    # ------------------------------------------------------------------
    # thin-film surface meshes (laneX 2026-10-06)
    # ------------------------------------------------------------------
    'liquid_bridge_film_3D': SolverMethods(
        dim=3, phases='film', integrator='symplectic_euler',
        connectivity='frozen',
        label=_LB + 'equilibrium/Case_1_equilibrium_particle_particle_bridge_benchmark.py',
        notes='laneX 2026-10-06. Exact catenoid surface mesh (a = 1, '
              'v in [-1.5, 1.5]) held on the Heron surface-tension force '
              'with velocity damping 20 1/s, dt 2e-6, 100 steps; rims '
              'frozen. Until laneX the case called the volumetric '
              'stress_force on the surface mesh (AttributeError vd). '
              'Measurements: laneX log.',
    ),
    'liquid_bridge_cfd_dem_3D': SolverMethods(
        dim=3, phases='film', integrator='symplectic_euler',
        connectivity='custom',
        label=_LB + 'cfd_dem/liquid_bridge_cfd_dem_case.py',
        notes='laneX 2026-10-06. Two spherical-cap films (refinement 2, '
              'thickness 10 um) on the Heron force with damping 1e-3, 10 '
              'fluid sub-steps of 1e-7 s per DEM step; custom = the '
              'case-local retopologize_surface (Unverdi-Tryggvason edge '
              'remesh, rim / particle-attached vertex update, Heron '
              'masses). Smoke only, no pin.',
    ),
    'liquid_bridge_volume_3D': SolverMethods(
        dim=3, phases='single', integrator='symplectic_euler',
        connectivity='frozen',
        label=_LB + 'equilibrium/Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark.py',
        notes='laneX 2026-10-06. Structured tetrahedral catenoid volume '
              '(prism / hex fill) at p = 0, mu = 0, u = 0 on the '
              'volumetric stress force, duals frozen: the force is '
              'identically zero and the hold is inert (the case '
              'short-circuits it when the static force norm is 0).',
    ),
    # ------------------------------------------------------------------
    # boundary-condition kinematics demo (laneX 2026-10-06)
    # ------------------------------------------------------------------
    'bc_demo_2D': SolverMethods(
        dim=2, phases='single', integrator='euler', connectivity='frozen',
        label=_BD + 'bc_demo.py',
        notes='laneX 2026-10-06. Prescribed advection (zero acceleration, '
              'u = U on every non-wall vertex) through PositionalNoSlipWallBC, '
              'OutletDeleteBC and PeriodicInletBC; no force, no dual mesh '
              '(the demo advected the vertices in its own loop until laneX).',
    ),
}


def preset(name: str) -> SolverMethods:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: "
                       f"{sorted(PRESETS)}") from None
