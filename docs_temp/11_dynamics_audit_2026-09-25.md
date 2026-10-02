# 11 - Dynamics audit (2026-09-25): state of the code, method inventory, corner-vertex verdict

> Sources: four line-anchored audit reports written on 2026-09-25 and saved
> under [audit_2026-09-25/](audit_2026-09-25/) (`integrators.md`,
> `multiphase.md`, `hyperct.md`, `bcs_partA.md`) plus per-case sub-reports
> summarised in section 3; repo state ddgclib `72e8cf6` (dirty) / hyperct
> `5655d31` (dirty). Written by the audit-and-abstract session that also
> shipped `ddgclib/methods/` and `METHODS.md`.

## 0. One paragraph

The dynamics code in ddgclib is committed and green (fast suite 881 passed /
0 failed / 2 xfailed; hyperct 336 passed with the 39 known benchmark-fixture
errors). hyperct's working tree carries ~650 uncommitted lines, ONE of which
ddgclib already depends on (`batch_e_star(orient=True)`); the rest is an
unused, defective `SimplicialComplex`/`_ops` layer that must not be enabled.
The solver makes about twenty method decisions per run, formerly scattered
across integrator kwargs, `functools.partial` bindings in setup helpers,
policy strings in `_params.py` files and dimension-gated branches, with
different names in every case. They are now named, validated and recorded
by `ddgclib.methods.SolverMethods` (registry in `METHODS.md`), with the
shipped droplet and dam-break runners proven bit-identical through it. The
audit found two latent bugs that matter for the unsolved 3D/large-deformation
regime (a never-invalidated 3D interface-curvature cache; a `'stokes'` path
that silently returns zero after the first vertex move), two implicit
dimension-gated choices nobody had written down (2D never builds an
edge-area cache and keeps boundary half-cells; 3D forces read a
non-linearly-precise cache instead of the documented p_ij construction), and
a reproducible mechanism behind the "vertices spawning in corners" symptom
(convex-hull re-tagging unfreezes a whole wall the moment one vertex crosses
it, so only the corners stay pinned).

## 1. State of the code

| item | state |
|---|---|
| ddgclib dynamics (`dynamic_integrators/`, `operators/`, `multiphase.py`, BCs/ICs, cases) | committed in `72e8cf6` (2026-08-11); working-tree changes are only benchmarks + `cases_mean_flow/equil_bubble` |
| ddgclib fast suite | 881 P / 12 S / 2 xfail / 0 F in 77 s (`pytest ddgclib/tests -m "not slow"`), identical to the 2026-07-30 campaign close |
| hyperct HEAD `5655d31` (2026-07-29) | contains laneA canonical ordering, lane4 remesh conservation, simplex-aware duals |
| hyperct load-bearing hunks | **COMMITTED on master 2026-09-25 (`0e6294b`)**: `ddg/_operators.py` `batch_e_star(orient=)` (called by `_retopologize` step 5b; against the previous HEAD every 3D retopology raised `TypeError`), `ddg/_dual_cell.py` walk-ordered 2D dual polygon, `ddg/_curvature.py` `HC=` kwarg, `_backend.py` heron kernel only + `tests/test_gpu.py`; plus a follow-up initialising `Complex._simplices = None` (a ddgclib test read it directly and only passed because the parked branch had made it a property). hyperct suite on master: 290 passed / 0 failed |
| hyperct SimplicialComplex layer | **PARKED on branch `wip/simplicial-layer` (`6cd09b1`)**: `_simplicial.py`, `_ops/`, `_complex.py`/`_vertex.py` hooks, circumcenter/sparse backend kernels, 4 test files, agent-written docs claiming "Complete". Off by default; zero ddgclib references; four reproduced defects (retriangulation silently ignored after the first, `invalidate_simplex_cache` no-op, `edge_collapse_2d` leaves holes, reading `HC._simplices` re-runs Delaunay without disconnecting). Do not wire into ddgclib core until `audit_2026-09-25/hyperct.md` §3.4 items 1-4 are fixed and tested |
| environment | `ddg` env has a stale non-editable hyperct 0.3.5 wheel; the live tree is picked up only when cwd is the ddgclib root (any script run from elsewhere imports the wheel and fails on `connect_and_cache_simplices`). torch not installed, so no GPU/torch path has ever executed here |
| broken imports in cases | `cube2droplet/*` (`cases_dynamic.Cube2droplet`, capital C, all 6 runners); `liquid_bridge_approach/*` (ImportError on a function that exists only on an unmerged branch); `shearing_plate_droplet_3D.py` crashes in setup (rescale moves an outer vertex onto a droplet vertex key, `HC.V.move` evicts silently); `oscillating_droplet_p_ref/run_github_twophase_oscillating_preview.py` (sys.modules stub) |

## 2. Findings, ranked by impact on the unsolved dynamics

Each item names the file, what is wrong, what it affects, and the fix
site. None of these was changed by this session (scope: audit + wrapper).

### F1. 3D interface curvature apex cache is never invalidated (multiphase T1) — FIXED (lane I, 2026-09-25); NOT the cause of the 3D rejections
Lane I result: fix in `extract_interface` (conditional apex-map clear, unconditional coordinate-map clear), 3 regression tests, every pin bit-identical, and the 3D per-step Delaunay / delaunay+remap scores bit-identical to laneB/laneE because the droplet runs never flip an interface triangle (0 of 872 steps). Original finding follows.
`ddgclib/_curvatures_heron.py:76-89` builds `HC._interface_edge_to_apex`
once; `extract_interface` (`geometry/_interface_subcomplex.py:112-114`)
rebuilds `HC.interface_triangles` on every `mps.refresh` but nothing clears
the apex cache (grep: only writer is line 89). Probe: one 3D Delaunay
retopology after a 1e-3 jitter changed 22 of 48 interface triangles; the
surface-tension force from the stale cache differed from the fresh value by
1.15e-3 N against max|F| 1.05e-3 N, i.e. 100 %. Harmless under the 3D
default `dual_only` (connectivity frozen, cache stays valid). Latent in every
3D per-step Delaunay run, which are exactly the configurations laneB/laneE
measured as much worse (l2 1.52 bare, 1.87 with remap) and rejected. Not yet
shown to be the cause of those scores. Fix: clear `HC._interface_edge_to_apex`
(and `_interface_x_to_v`) at the end of `extract_interface`, then re-run the
3D remap A/B (laneE §1d) before trusting the 3D DO-NOT.

### F2. `curvature_path='stokes'` is unusable on a moving mesh (multiphase T2) — FIXED (lane I): the coordinate map is now cleared on every interface refresh (tested); status downgraded from `broken` to `experimental` pending a dynamic A/B
`_curvatures_heron.py:466-469` caches a coordinate-keyed `HC._interface_x_to_v`
once; `HC.V.move` re-keys `v.x`, so after the first move + refresh the lookups
miss and triangles are skipped. Probe: stale-vs-fresh difference equals the
full force (6.0e-4 N) with unchanged topology. Static-mesh equivalence with
`'integrated'` (Probe 2, 2026-05-27) still holds. Status in the registry:
`broken`. Fix: key by `id(v)` and rebuild with the interface, or drop the path.

### F3. 2D and 3D take different dual-geometry paths, undocumented (integrators §0.1)
`hyperct/ddg/_operators.py:403-404` raises `NotImplementedError` for
`dim != 3`, so `_retopologize` step 5b (`_integrators_dynamic.py:231-280`)
always falls into the `cache_dual_volumes` branch in 1D/2D: no
`HC._edge_area_cache`, boundary vertices keep their truncated half-cell
`dual_vol` (2D rectangle: max boundary dual_vol 0.0156, total 1.0), and forces
rebuild `A_ij` from `v.vd` every call. In 3D the cache is set and boundary
`dual_vol` is zeroed. The NOTE at `_integrators_dynamic.py:257-261` ("2D keeps
batch_e_star's volumes ... zeroing preserved in both paths") and the docs'
"all boundary vertices get dual_vol=0" are wrong for 2D. Consequences:
`DualVolumeMass` and the mass-relaxation BCs (`dual_vol < 1e-30` skip) behave
differently per dimension. Now reported per run by
`ddgclib.methods.effective_methods` (`boundary_dual_vol`, `edge_area_source`).

### F4. 3D forces read the e_star cache, not the documented p_ij area vector (integrators §0.2, hyperct F14) — MEASURED by lane J (2026-09-25): the cache carries ~79 % of the 3D static-floor retopology excess (7.274172e-05 -> 6.2839e-05 on p_ij; frozen floor 6.0153e-05) and part of the dynamic outward bump (final inflation 1.87 % -> 0.82 % R0), but p_ij alone raises l2 to 0.28713 because it exposes the over-decay the bump cancelled; no flip. Sub-finding F4b: the library p_ij picks face vertices as "common neighbour nearest the midpoint of two tet barycentres" and is wrong on 743 of 5193 directed droplet edges (laneG's 2.6 % closure residuals); reading faces from `HC._simplices` closes to 2e-16. Registry option `p_ij_simplex` (experimental) records it; lane Q in the plan is the follow-up. Original finding follows.
`stress_force` (`operators/stress.py:826-834`) and `multiphase_stress_force`
(`multiphase_stress.py:172-181`) prefer `HC._edge_area_cache`, filled in 3D
from `batch_e_star(orient=True)` (tet-barycentre fan, no face barycentres).
`stress.py:280-291` documents that construction as "NOT linearly precise";
`test_stress.py:2098-2123` asserts the cache equals it. The linearly precise
`_dual_area_vector_3d_p_ij` (`stress.py:182-277`) runs only before the first
retopology. Measured on `box(refinement=2)` / `ball(refinement=2)`: per-edge
relative difference median 0.125, max 0.625 / 0.25; linear-precision residual
1.3e-3 / 9.0e-4 on the cache path vs 1e-18 on p_ij. Every pinned 3D number
was produced on the cache path, so switching is a re-pin. Candidate lever for
laneG's "pressure-side dual-measure rework" (dual-face closure residuals up to
2.6 % at interface/outer vertices).

### F5. Inverted kwarg precedence for `remesh_mode` / `remesh_kwargs` (integrators §0.3, multiphase T5)
`_do_retopologize:427-430` forwards these two whenever the callable declares
them, so the integrator value overrides a `functools.partial` binding; the
laneF rule (partial wins) applies to the other eight kwargs (`:436-461`).
Documented only as a workaround comment in the old 2D runner. `SolverMethods`
always passes them at integrator level, which sidesteps the trap.

### F6. `DynamicSimulation` is unsafe for the canonical `partial(dudt_i, mu=...)` pattern
`_simulation.py:61-65,170`: `SimulationParams.dudt_kwargs` always injects
`mu` (default 8.9e-4) as a call-time kwarg, which beats the partial's binding;
a `multiphase_dudt_i` partial raises `TypeError`. It forwards no retopology
switch except `skip_triangulation`, leaks `extra` into integrator kwargs,
ignores `t_end` for fixed-step integrators and never uses `rho`. No case uses
it. `SolverMethods.integrate` is the replacement; consider deprecating.

### F7. `workers > 1` loses the EOS side effects (integrators §0.5)
`_resolve_pressure` writes `v.p`/`v.rho` (`stress.py:738-739`) inside forked
children (`_integrators_dynamic.py:833-849`); the parent's fields go stale for
BCs, callbacks and snapshots. Forces are still right. Registry status `broken`.

### F8. Docstring promises that the code does not keep
Harmonic-mean viscosity at cross-phase faces (`multiphase_stress.py:21-23`,
DEVELOPMENT.md) is NOT implemented; the code and its test use the phase's own
`mu_k` (`multiphase_stress.py:116-126`, `test_multiphase_stress_per_phase.py:108-155`).
`curvature_path` docstring lists 2 of 3 values; closure-warning, tie-rule and
edge-fraction docstrings are stale (multiphase §4).

### F9. Silent no-ops and plumbing holes (integrators §4, multiphase §3)
`retopo_remap='conservative'` with `skip_triangulation=True` or `mps=None`;
`redistribute_mass=True` without `pressure_model` (single) or `mps` (multi);
`split_method` typos fall back to `neighbour_count`; `**kwargs` retopo sinks
receive only `remesh_*` (cube2droplet); `_retopologize_multiphase` does not
declare `periodic_axes`/`domain_bounds`/`pressure_model` (periodic multiphase
needs the copied case wrapper, no remap/cadence); periodic path ignores five
switches; `backend` means two different things; single-phase `pressure_model`
is specified twice with no consistency check; `mps` run-state
(`vol_corr`, `_remap_*`, `_projection_call_idx`) is never reset between runs.
`SolverMethods.__post_init__` turns the first group into `ValueError`s.

### F10. "Too many vertices spawning in corners": verdict (bcs_partA §A.3). C1, C2 and the key collision of C3 FIXED by lane L (2026-10-01)
Lane L result: opt-in method axis `frozen_set='membership'` (persistent wall set, `v.boundary` still topological). `Hagen_Poiseuile_2D.py` through the preset: 3000 steps, 62 of 62 wall vertices frozen and unmoved; the hull arm releases the walls at step 1262 (60 of 62 move, largest displacement 3.46). hyperct `HC.V.move` now refuses a move onto the key of another vertex (`on_collision='evict'` = old behaviour, `move_all` for shifts); the inlet ghost keeps its 13 vertices and no longer overwrites a resident vertex. Open: C3b (two wall-row duplicates), the doubled seam column of the inlet, impenetrability, C5 to C9. Side finding: the droplet builders' own box shift loses 6 (2D) / 3 (3D) outer vertices to the same key collisions; pinned, so kept explicitly. Fix round 1 (same day): `'membership'` is refused under `connectivity='adaptive'` (split vertices of wall edges are not members; members off the hull are collapsed and smoothed), and the process dependence seen in the shearing-plate setup and in the dam break after an ejection was a tied simplex phase vote decided by `id()` order, now deterministic. Log: `docs_temp/debug_session/laneL-frozen-set-membership.md`. Original finding follows.
| # | mechanism | verified | where |
|---|---|---|---|
| C1 | **Convex-hull re-tagging.** `bV` is rebuilt from the hull every step (`_integrators_dynamic.py:209-215,285-288`); nothing enforces impenetrability. Once any vertex steps past a straight wall, every collinear wall vertex stops being a hull vertex, leaves `bV`, gets integrated and is never re-captured by `PositionalNoSlipWallBC` (tolerance 1e-10). Only the corners stay frozen, so fluid piles up there. In HP2D the trigger is `OutletBufferedDeleteBC`'s frozen velocity keeping its wall-normal component (`_boundary_conditions.py:512-534`) | reproduced (step 1247: two buffer vertices cross; step 1248: wall set 23 -> 2) | HP2D/HP3D, `test_outlet_*`; latent in every box case once a vertex penetrates a wall (dam break, capillary rise, droplet-in-box) |
| C2 | `PeriodicInletBC._reset_ghost` moves the unit mesh's far face onto the near face's keys; `HC.V.move` evicts the occupant silently, so the ghost loses vertices (13 -> 12) and one wall never receives injections | reproduced | HP2D/HP3D, bc_demo, cube_flow |
| C3 | Plug-speed ghost injects every column at the same float key; `mesh.V[key]` overwrites the frozen inlet-corner wall vertex's `u` to `U_avg` every period (kick), and interior near-wall inflow exceeds the no-slip outflow, so vertices accumulate at the inlet-wall corners (the HP2D header itself records ~2900 extra vertices in 3000 steps for the layer path) | key collision reproduced; accumulation reasoned + header evidence | same |
| C3b | **Duplicate frozen wall vertices one advection step from the inlet corner.** The ghost's wall-row vertices enter at x = U dt = 1e-3, 1 mm from the frozen corner at x = 0; `cdist = 1e-10` (and HP3D's retopo merge at 1e-9) can never merge them, and `PositionalNoSlipWallBC` freezes them at once. Bounded per cycle (later cycles hit the same key) but doubles the wall density at the corner. In HP3D the custom `retopologize_cylinder` additionally freezes EVERY hull vertex (no filter), so the inlet cap is a plug and injected vertices pile up behind it | HP3D final state: 9 wall vertices at z = 0 plus 7 at z = 0.001; 305 of 407 vertices frozen, 48 of them strictly inside the tube; `run.log`: no interior vertex near mid-tube after 3000 steps | HP2D, HP3D, `run_cluster.py` (`cases_poiseuille_hydrostatic_bridges.md` §13) |
| C5 | `OutletDeleteBC` uses `>=` (`_boundary_conditions.py:426`) so `outlet_pos == L` deletes the outlet plane including both wall corners on step 1 | code | cube_flow, bc_demo |
| C6 | Adaptive remesh ratchet: `can_collapse` refuses any edge with a `v.boundary` endpoint (`hyperct/remesh/_interface.py:84-89`) while splits on boundary-adjacent edges are allowed, so near walls/corners the count can only grow. In capillary_rise dynCA the kinematically driven contact vertex stretches the wall edge below it, the split midpoint lands exactly on the wall line with `boundary=True`, and `tag_groups_2d` (`capillary_rise/src/_setup_dynca.py:118-156`) re-tags it as a frozen wall vertex: one new wall vertex per 0.8 dx0 of contact-line travel per side, and the impenetrability clamp converts interior vertices into wall vertices as well | **measured** on `results/dynca_2d_water_R0.5mm_prod2_final.json`: minimum wall spacing exactly 0.30 dx0 (= the merge threshold), 12/13 wall pairs closer than 0.5 dx0, wall line density 12.0-14.5 vertices/mm vs 8.0/mm initially; vertex count 260 -> 339 at abort in the smoke run | capillary_rise dynCA (adaptive every 5 steps); any `remesh_mode='adaptive'` run |
| C7 | Truncated/zeroed corner duals and sliver cells: `a = F/m` spikes on near-zero-volume air cells after reconnection (laneF's dam-break blocker, vertex ejected 6.7x outside the domain) | laneF | dam break, electrolysis 3D, every box case |

Also found in code: electrolysis_bubble's `WallClampBC` stacks every
clamped vertex onto one line near the wall (slivers there); the three
mass-relaxation pressure BCs skip `dual_vol < 1e-30`, i.e. every wall vertex
after a 3D retopology, so they are no-ops on the walls they target.

The "spawning" is therefore not a single BC bug: C1 (freezing by hull
membership) is the structural cause of wall collapse, C2/C3/C3b the source of
extra inlet-corner vertices, C6 the capillary-rise ratchet, C7 the corner
blow-up. A further inlet artefact, not a vertex count: the unit-mesh IC gives
the ghost `p = -G x_unit`, so each cycle's first injected column arrives with
`p = -G` next to `p ~ 0` neighbours and the field is never re-evaluated
(`pressure_model=None`), a pressure sawtooth of height G at the inlet corners. Note also that the three headline numbers (2D l2 0.17479, 3D l2
0.24811, static 1.1847e-03) are pinned only by baseline JSONs and
comments, not by any pytest: the floor tests exercise setup's bare-Delaunay
retopology, not the runners' policies (the fast envelope test pins a
refine-2/2 mirror at l2 < 0.0600). The principled fix for C1
is the `prescribed_V` / wall-membership freeze already specified in
DEVELOPMENT.md ("Fixed Inlet/Outlet Boundary Conditions") instead of hull
membership; C2/C3 need a collision check in `HC.V.move` (hyperct) and a
non-plug ghost advance.

### F11. The pinned static-droplet score validates a frozen pressure field
`static_droplet_2D.py:114-123` runs a bare closure (`HC.boundary` +
`compute_vd` + `cache_dual_volumes` + `split_dual_volumes`) that never calls
`mps.refresh`, `compute_phase_pressures` or redistribution, so `p_phase` keeps
its setup value for all 100 steps. The pin 1.1847162859108737e-03 measures
surface tension against a frozen pressure, not the EOS response. Same in
`cube2droplet/diagnostic_no_retopo.py`. Now recorded as
`connectivity='custom'` in the run's `methods.json`.

### F12. Cases that bypass the library integrators — MECHANISM CLOSED by lane K (2026-09-25): single-phase + EOS blows up under Delaunay because a flip changes a dual volume by 33-100 % (read as 3e4-5e4 Pa); stable under `dual_only`; the Hydrostatic column fails separately because the setup-time `dual_cell_area_2d` fallback undercounts corner / free-surface volumes (0 flips). Prototype single-phase conservative remap is stable; lanes R (library remap) and S (exact setup volumes) are the follow-ups, then P (port the hand-rolled loops). Lane S SHIPPED 2026-10-01: builders cache their simplices, the fallback is fixed, and the unmodified `Hydrostatic_2D.py` settles over 100 acoustic times (laneS log). Lane P SHIPPED 2026-10-01: the four Hydrostatic_column runners run through presets (`hydrostatic_1D` / `_2D` / `_2D_periodic` / `_3D`, fixed connectivity) on the library integrators and settle (2D 100 t_ac: max|u| 0.248 -> 3.0e-4, integrated L2 5.1e-3 rho g H; 3D 40 t_ac: 2.9e-4); `delaunay` + remap is unstable on the free surface because Delaunay fills the convex hull, fixed by the new `connectivity='delaunay_material'` (geometric peel since the review fix of the same day: the first, topological peel removed fluid in 3D; domain kept to round-off in 2D and to 1.2e-6 per call in 3D); the "periodic" runner uses free-slip walls because single-phase periodic connectivity is unusable with an EOS; the settled column needs artificial viscosity (discrete equilibrium is a saddle, growth 1.13 g/c0) (laneP log). capillary_rise, dynamic_caprise_tube and liquid_bridge_approach are still hand-rolled. Original finding follows.
Hydrostatic_column (all four), capillary_rise (all four, incl. dynCA),
dynamic_caprise_tube and liquid_bridge_approach use hand-rolled loops around
`_recompute_duals`/`compute_vd` (no `_retopologize`, no `batch_e_star`, no
boundary re-tagging). They cannot take a `SolverMethods` preset and are listed
as "hand-rolled" in METHODS.md. Hydrostatic 2D is unstable (|u| reaches 10 c0),
2D_periodic aborts, cube_flow stalls after step 0 (outlet column frozen by the
hull rule), shearing_plate 2D loses its interface by t = 0.044 s. The
static-theta capillary-rise scaffold binds `mu = mu_art` ALONE (104x the
physical viscosity in 2D, 559x in 3D; the physical value is dropped), leaves
the wall-top corners out of every no-slip target, and aborts at 1.3 % of its
horizon with 30 of 145 vertices pushed out through the slit walls; the 3D
column sinks. Full detail: `audit_2026-09-25/cases_caprise_dam_bubble_shear.md`.

### F13. Dead / duplicated code (not removed)
`operators/stress_pointwise.py` (0 importers), `geometry/_dual_split_3d.py`
(re-export shim, 0 importers), `MultiphaseEOS.__call__` as `pressure_model`
(dead in production, every case still builds it), `snapshot_pressure_multiphase`
(re-arms the P0=0 guard bug if used), IC classes `PhaseAssignment`/`MultiphaseMass`/
`MultiphasePressure` (test-only, interface-index wrap bug), `MovingWallBC`,
`MeshAdvancer`, `NeumannBC` (no case), `_recompute_duals` (library-unused),
`_diagnostic_plot.py` (orphan), the `oscillating_droplet/_params.py` dead
constants, `hyperct/_vertex.py:787` NameError, five copies of the same
`_do_retopologize` call in the integrators, `euler` update duplicated inside
`euler_adaptive`, three different interface-neighbour rules (2D FTC angular
heuristic vs exact `interface_edges` vs flag-only).

## 3. Case status matrix (2026-09-25)

Preset = name in `ddgclib.methods.PRESETS` (consumed by the runner, proven
bit-identical). "hand-rolled" = the runner does not call a library integrator.

| case | dim | preset / how it runs | status | pinned number |
|---|---|---|---|---|
| oscillating_droplet_2D | 2 | `oscillating_droplet_2D` (delaunay + conservative remap) | validated | l2 0.17479361640597058 / tail 0.9998967874595965 |
| oscillating_droplet_2D (`dual_only`) | 2 | `oscillating_droplet_2D_dual_only` | validated opt-in | l2 0.17857 / tail 0.99925 |
| oscillating_droplet_2D (`delaunay_remap_p2`) | 2 | `oscillating_droplet_2D_projection2` | opt-in, not adopted | l2 0.03796 / l2_2f 0.02193 / tail 1.397 |
| static_droplet_2D | 2 | `static_droplet_2D_bare_dual_only` (custom closure, frozen pressure, F11) | validated | summary 1.1847162859108737e-03, mass 0.0 (reproduced through the wrapper) |
| floor tests 2D / 3D | 2/3 | `static_droplet_floor_2D/3D` (euler, u=0, bare delaunay) | regression-locked | 2.3748568e-03 / 2.2716938e-03; 6.0153e-05 / 7.274172e-05 |
| oscillating_droplet_3D | 3 | `oscillating_droplet_3D` (dual_only) | validated, physics gap open (laneG cancellation) | l2 0.24811340819647862 |
| oscillating_droplet_3D `--retopo delaunay` | 3 | `oscillating_droplet_3D_delaunay` | measured-worse (+F1 stale apex cache) | l2 1.52446 |
| dam_break_2D | 2 | `dam_break_2D` (delaunay + remap, gravity closure, alpha_art 0.3) | runs (laneF); sliver F/m ejection blocker | KE_liq peak 1.04e-3 J, mass 5e-15 |
| dam_break_3D | 3 | `dam_break_3D` (dual_only) | not re-run since laneF (snapshots from 2026-04-10, old geometry, per-step-Delaunay era) | - |
| dam_break_2D/3D_no_air | 2/3 | single-phase default retopo + `boundary_filter=is_wall`, uniform p (docstring's hydrostatic IC not applied) | unstable (2D 100-280 m/s by t=6e-4 s; 3D vertices at 1e54 by t=0.0065 s), stale April outputs | - |
| oscillating_droplet_2D_adaptive / _mass_redist / mesh_convergence | 2 | bare setup partial (do NOT honour `retopo_policy_2d`), adaptive kwargs, custom closures | runs, poor (adaptive l2 2.73) / April snapshots / dt 16x CFL | - |
| capillary_rise 2D/3D (static theta) | 2/3 | hand-rolled (`_recompute_duals`), `mu = mu_art` only | 2D aborts at t=0.0044 s (|u| 38 m/s ≈ 10 c0, 30/145 vertices outside the slit); 3D column sinks 0.880 -> 0.833 cm | - |
| capillary_rise 2D/3D dynCA | 2/3 | hand-rolled (`mass_conserving_merge` + `adaptive_remesh` every 5 steps + `_recompute_duals` + ledger repair), data-driven CA(t) | 2D: -5.5 % at t=0.013 s then stalls (free-surface compression; wall-vertex ratchet C6); 3D prod: -14.8 % at t=0.04 s | 2D L2 0.14 window / 0.686 full (h_final -84 %); 3D L2 0.483 |
| electrolysis_bubble 3D | 3 | `partial(_retopologize_multiphase, ...)`, per-step Delaunay | unstable: gas phase lost entirely by t≈1.1e-4 s | - |
| Hydrostatic_column 1D/2D/3D/2D_periodic | 1-3 | AT THE AUDIT: hand-rolled symplectic loop, `_recompute_duals`. Since lane P (2026-10-01): presets `hydrostatic_1D` (`delaunay` chain) / `_2D`, `_2D_periodic` (`dual_only`; free-slip walls, not periodic) / `_3D` (`dual_only_bare`) | at the audit: 1D decaying, 2D unstable (aborts), 3D stalls, periodic aborts. Lane P: all four run and settle | fast and slow pins in `test_case_hydrostatic.py` |
| Hagen_Poiseuile_2D | 2 | AT THE AUDIT: `symplectic_euler` + `boundary_filter` + `PeriodicInletBC` + `OutletBufferedDeleteBC` (no preset yet). Since lane L (2026-10-01): preset `hagen_poiseuille_2D` with `frozen_set='membership'` | at the audit: wall collapse (C1). Lane L: 3000 steps, walls intact; profile not validated | wall positions in `test_frozen_set.py` |
| Hagen_Poiseuile_3D | 3 | custom `retopologize_cylinder` (never updates `HC._simplices`) | runs | - |
| Hagen_Poiseuile_2D_Eulerian / _equilibrium | 2 | `euler_velocity_only` | validated equilibria | machine precision |
| cube_flow 1D/2D/3D, bc_demo*, template | 1-3 | mock dudt, default retopo | demos; cube_flow stalls after step 0 | - |
| electrolysis_bubble 2D/3D/fritz | 2/3 | `partial(_retopologize_multiphase, split_method, redistribute_mass)`; fritz leaves redistribute unbound (=False) | unvalidated | - |
| shearing_plate_droplet 2D/3D | 2/3 | case-local periodic multiphase wrapper (no remap/cadence) | 2D unstable, 3D crashes in setup | - |
| cube2droplet (6 runners) | 2/3 | ModuleNotFoundError (capital C import) | broken | - |
| liquid_bridge_equilibrium / cfd_dem | 3 | `retopologize_fn=False` surface meshes | runs | - |
| liquid_bridge_approach | 3 | own semi-implicit loop; ImportError at load | broken | - |

## 4. What this session built

- `ddgclib/methods/` (registry `_axes.py`, `SolverMethods` `_config.py`,
  resolver/recorder `_effective.py`, markdown `_report.py`, `PRESETS`
  `_presets.py`, CLI `__main__.py`), 62 tests in
  `ddgclib/tests/test_methods.py` including a bit-identical 8-step A/B against
  the hand-written runner path and locks on the F3 findings.
- `METHODS.md` at the repo root: generated axis tables + case matrix.
- Runners switched to presets and now write `methods.json` next to their
  scores: `oscillating_droplet_2D.py` (policy strings incl. the new
  `delaunay_remap_p2`), `oscillating_droplet_3D.py`, `static_droplet_2D.py`,
  `dam_break_2D.py`, `dam_break_3D.py`.
- Phase 2 (same day, after the hyperct commit): the case-local retopology
  closures became library connectivity values in `ddgclib/methods/_retopo.py`
  (`dual_only_bare` = the static-droplet closure; `periodic` for multiphase =
  the shearing-plate closure), each with a bit-identity test against the
  closure it replaces; presets and `methods.json` wiring for
  `Hagen_Poiseuile_2D` / `_2D_Eulerian` / `_3D`, `electrolysis_bubble`
  2D / 3D / fritz and `shearing_plate_droplet` 2D / 3D; the template
  showcase rebuilt around `SolverMethods` (a probe showed every single-phase
  variant with an EOS in the loop blows up, so the template holds pressure
  and decays viscously, KE 0.36 -> 2e-11 J).

## 5. Recommended next steps (ordered; status as of the end of 2026-09-25)

1. DONE: hyperct load-bearing hunks on master (`0e6294b`, `4bb966d`), the
   SimplicialComplex layer on branch `wip/simplicial-layer`. Still open:
   `pip install -e` hyperct into `ddg` (or remove the 0.3.5 wheel); commit
   ddgclib (its tree also holds unrelated user edits).
2. IN PROGRESS as lane I: fix F1 (cache invalidation in `extract_interface`)
   and re-run the 3D per-step-Delaunay and delaunay+remap A/Bs; if the 3D
   DO-NOT flips, laneE's rejection was an artefact.
3. Fix F2 or delete `'stokes'` (lane M in the plan).
4. Replace hull-membership freezing with wall membership (`prescribed_V`
   spec) and add a collision check to `HC.V.move`: closes C1/C2/C3 and is a
   prerequisite for any inlet/outlet or capillary-rise case (lane L).
5. IN PROGRESS as lane J: F4 (p_ij vs e_star cache in 3D) as the
   pressure-side lever for laneG.
6. IN PROGRESS as lane K: the single-phase EOS instability (F12): mechanism,
   single-phase counterpart of the conservative remap; gates capillary rise.
7. DONE: presets + `methods.json` for HP2D/3D/Eulerian, electrolysis,
   shearing plate; hand-rolled cases stay marked as such (port them only
   after lane K).
8. Capillary-rise wall ratchet (F10 C6): allow boundary-boundary collapses
   on straight walls or defer the wall re-tag of split midpoints (lane O).

## 6. Detailed reports (audit_2026-09-25/)

| file | covers |
|---|---|
| `integrators.md` | integrator kwargs and step order, retopology decision table, 19 method axes, kwarg plumbing holes |
| `multiphase.md` | multiphase data model, 16 method axes, 16 consistency traps (T1 apex cache, T2 stokes), dead code |
| `hyperct.md` | uncommitted hyperct inventory, 13 mesh/dual axes, SimplicialComplex defects, invariants re-verified, commit plan |
| `bcs_and_cases.md` (+ `bcs_partA.md` draft) | BC/IC class tables, corner-spawning verdict C1-C9, regression-locked configs |
| `cases_caprise_dam_bubble_shear.md` | per-runner configs and status for capillary_rise, dam_break, electrolysis_bubble, shearing_plate_droplet; capillary-rise ratchet evidence |
| `cases_poiseuille_hydrostatic_bridges.md` | per-runner configs for Hagen_Poiseuile*, Hydrostatic_column, bc_demo, cube_flow, template, liquid_bridge_*; inlet-corner mechanisms M1-M8 |
