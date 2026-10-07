# DEVELOPMENT.md — Feature Status Distillation
> Sources: /home/endres/projects/ddgclib/DEVELOPMENT.md (1272 lines, read fully) | Written: 2026-07-02 by understand-and-document workflow

## What is actively being worked on (as of the file's last state)

1. **Per-phase summed interface stress ("Phase 6")** — DEVELOPMENT.md:5-70. In Progress. 2D+3D implementation landed; 2D full-length oscillation validation done (improvement but misses spec target); 3D validation blocked on missing 3D metric harness. Remaining blocker: upstream **hyperct `hyperct/remesh/_operations_2d.py:198` mass-averaging bug** which prevents adaptive remesh from closing the residual gap.
2. **Multiphase Fluid Simulation** — DEVELOPMENT.md:737-789. In Progress: Phases 1–5 done; Phase 6 open items = full dynamic validation of overdamped envelope, underdamped Rayleigh frequency match, 3D convergence study.
3. **Cube Flow Case Study** — DEVELOPMENT.md:181-190. In Progress: 3D video generation and unit tests pending.
4. **Pylint Standards Enforcement** — DEVELOPMENT.md:226-268. In Progress: `.pylintrc` done (score 1.23/10 → ~7.8/10, 15,948 → 4,310 messages); all cleanup subtasks (wildcard imports, E-level errors, docstrings, formatting) still open.

Driver for #1: "catastrophic KE growth in `cases_dynamic/oscillating_droplet/`" (DEVELOPMENT.md:9-11); baselines + metric harness in `cases_dynamic/oscillating_droplet/baselines/` and `src/_metrics.py`.

## Full feature status table

| Feature | Status | Lines |
|---|---|---|
| Per-phase summed interface stress (Phase 6) | **In Progress** (2D/3D landed, validation pending) | 5-70 |
| Domain Builder Module | Complete (open: `channel_with_obstacle()` DFG benchmark, surface-of-revolution builder) | 72-85 |
| Initial Conditions Module | Complete (20 tests) | 87-96 |
| Boundary Conditions Module | Complete (15 tests) | 98-106 |
| Operators Package | Complete (11 tests) | 108-116 |
| Enhanced Dynamic Integrators | Complete (24 tests; `bc_set`, `euler_adaptive` CFL, `DynamicSimulation`) | 118-126 |
| Data Handling Package | Complete (JSON save/load, `StateHistory`, 16 tests) | 128-137 |
| Visualization Package | Complete (53 tests, 1 skipped) | 139-162 |
| Hydrostatic Column Case Study | Complete (1D/2D/3D, 10 tests) | 164-171 |
| Hagen-Poiseuille Case Study | Complete (8 tests) | 173-179 |
| Cube Flow Case Study | **In Progress** (3D video + tests pending) | 181-190 |
| Dynamic Capillary Rise Case Study | **Not Started (deferred)** — needs surface tension on meniscus + curvature ops | 192-196 |
| Documentation (DEVELOPMENT/ARCHITECTURE/FEATURES.md) | Complete | 198-202 |
| DDG Migration to hyperct.ddg | Complete (4 phases) | 204-224 |
| Pylint Standards Enforcement | **In Progress** | 226-268 |
| Integrated Analytical Validation Framework | Complete (47 tests; listed **twice**, also at 708-735) | 270-292, 708-735 |
| Cauchy Stress Tensor Operators | Complete (open: Template case update, dam break test case) | 294-332 |
| Lagrangian Mesh Retopologize | Complete (`_retopologize` per-step Delaunay) | 334-349 |
| Fix PeriodicInletBC Ghost Vertex Accumulation | Complete (bugfix) | 351-362 |
| HDF5 Data Handling + hyperct I/O | **Not Started** (plan: `.claude/plans/jiggly-tumbling-lampson.md`) | 364-379 |
| Outlet Buffer Ghost Zone (`OutletBufferedDeleteBC`) | Complete | 381-401 |
| Fixed Inlet/Outlet BCs (`prescribed_V` 3-tier vertex classification) | **Not Started** (detailed spec incl. `FixedVelocityInletBC`/`FixedPressureOutletBC` code sketches) | 403-548 |
| DEM Submodule (`ddgclib/dem/`) | Complete (3 phases, 124 tests + 1 slow) | 550-608 |
| Parametric Surface Geometry Module | Complete (3 phases incl. CFD-DEM fluid film) | 610-653 |
| Vectorized Stress Pipeline / Per-Edge Dual Area Caching | Complete (open: HP3D numpy/gpu backend verification) | 655-688 |
| Migrate Geometry (dual_area_vector, dual_volume) to hyperct.ddg | **Not Started** | 690-695 |
| Periodic Boundary Conditions | Complete Phase 1-3 (open: Phase 4 `Hydrostatic_2D_periodic.py` case; future CGAL periodic Delaunay) | 697-706 |
| Multiphase Fluid Simulation | **In Progress** (Phases 1-5 done, Phase 6 validation open) | 737-789 |
| Interface-Preserving Adaptive Refinement / Remeshing | **Phase 1–2 Complete (2D only)**; Phase 3 (3D ops) and Phase 4 (error indicators) not started | 791-992 |
| Pressure-Preserving Mass Redistribution After Retriangulation | Complete (15 tests) | 994-1018 |
| Rules-Based Mesh Quality Maintenance | Not Started (overlaps/duplicates remesh feature above) | 1020-1030 |
| Intrinsic Delaunay Triangulation (iDT) | Not Started (ref: Sharp/Soliman/Crane 2019, `navigating_intrinsic_triangulations.pdf` in repo) | 1032-1045 |
| 3D p_ij Dual Area Vector (linear precision) | Complete for interior edges; boundary edges O(h), "blocked by compute_vd bug" | 1047-1061 |
| FIX: 3D boundary sliver tets break `_compute_vd_3d` | **Fixed** (simplex-aware path via `HC._simplices`) | 1063-1185 |
| Adaptive Mesh Refinement (error-indicator driven) | Not Started | 1187-1196 |
| Multiphase dam_break interface-conforming refactor | Not Started (needs `rectangle_with_interface`/`box_with_interface` builders) | 1198-1215 |
| Backlog from 2026-06-08 library audit (LIBRARY_AUDIT.md) | Not Started: curated `ddgclib/__init__.py` API, legacy quarantine, docs consolidation, 3D oscillation metric harness, `int_bench/` home | 1217-1272 |

## Multiphase flow (Feature: Multiphase Fluid Simulation, lines 737-789)

Extends single-phase Lagrangian FVM with interface tracking, per-phase material properties, surface tension.
- **Phase 1 (done)**: `ddgclib/multiphase.py` — `PhaseProperties` dataclass, `MultiphaseSystem` (phase assignment, interface identification, mass fractions, harmonic-mean viscosity at cross-phase edges, surface tension lookup per phase pair). ICs: `PhaseAssignment`, `MultiphaseMass`, `MultiphasePressure`. 24 tests (`test_multiphase.py`).
- **Phase 2 (done)**: EOS — `ddgclib/eos/_ideal_gas.py`, `ddgclib/eos/_multiphase_eos.py` (`MultiphaseEOS` callable protocol for `stress_force(..., pressure_model=meos)`; mass-fraction-weighted pressure at interface vertices).
- **Phase 3 (done)**: `ddgclib/operators/multiphase_stress.py` — `multiphase_stress_force`, `multiphase_dudt_i`; harmonic-mean viscosity at interface; surface tension from Heron curvature (`hndA_i_interface()` in `ddgclib/_curvatures_heron.py` operating on interface sub-mesh).
- **Phase 4 (done)**: `ddgclib/geometry/domains/_multiphase_droplet.py` — `droplet_in_box_2d`, `droplet_in_box_3d`; `_retopologize_multiphase` in `dynamic_integrators/_integrators_dynamic.py`.
- **Phase 5 (done)**: Oscillating droplet case (see below).
- **Phase 6 (open)**: full dynamic validation (overdamped R_max(t) envelope, underdamped Rayleigh frequency, 3D convergence study).

### Per-phase summed interface stress rewrite (lines 5-70) — the current fix for multiphase instability
Spec: `docs/interface_stress_rewrite.md`. Key completed pieces:
- Exact geometric dual-volume splits: `split_dual_polygon_2d` and `split_dual_polyhedron_3d` in `ddgclib/geometry/_dual_split_2d.py` (3D: PCA-fit interface plane on 1-ring interface neighbours, Sutherland-Hodgman tet-fan clipping of the DEC p_ij dual polyhedron, sub-volumes rescaled to match `v.dual_vol`; falls back to own-phase-full-volume when <3 interface neighbours).
- `MultiphaseSystem.split_dual_volumes(method='exact'|'neighbour_count')` — default remains `'neighbour_count'`, `'exact'` is opt-in; plumbed through `refresh(..., split_method=...)` and `_retopologize_multiphase(..., split_method=...)`.
- `multiphase_stress_force` rewritten to sum F_k per present phase — own-phase pressure on both ends of each sub-face (fixes old bug where `v_j.p_phase[own_phase]` read phase-k pressure at a bulk neighbour that never stores phase k).
- `edge_phase_area_fractions(dim=...)`: 2D 50/50 only at curve neighbours (two-polyline-neighbour rule; chord edges stay own-phase); 3D every interface neighbour is 50/50.
- Harmonic-mean viscosity at cross-phase faces: μ_f = 2 μ_i μ_j / (μ_i + μ_j).
- Tests: `ddgclib/tests/test_dual_split_2d.py` (24) + `ddgclib/tests/test_multiphase_stress_per_phase.py` (7); 733 fast tests passed at that point.

### Multiphase dam_break refactor (Not Started, lines 1198-1215)
dam_break 2D/3D still uses legacy phase-field model (spatial phase criterion on plain rectangle/box). After the "primal-subcomplex interface refactor", `MultiphaseSystem.refresh` requires explicit simplex-phase labels and a closed interface subcomplex — needs `rectangle_with_interface` / `box_with_interface` builders. Transient interface evolution needs interface-preserving remesh (2D exists in `hyperct.remesh`, 3D TODO).

## Surface tension

- Surface tension force computed at interface vertices from **Heron curvature** on the interface sub-mesh: `hndA_i_interface()` (`ddgclib/_curvatures_heron.py`), used inside `multiphase_stress_force` (lines 757-762).
- Surface tension coefficients are per-phase-pair lookups in `MultiphaseSystem` (line 746).
- Dynamic Capillary Rise case (lines 192-196) is **deferred** precisely because it "Requires surface tension forces on meniscus" and depends on curvature operators from the mean-flow pipeline.
- CFD-DEM fluid film (lines 641-653): capillary force from catenoid neck tension F = 2πaγ; analytical curvature stored on film vertices (H = 1/R sphere, H = 0 catenoid). DEM liquid bridges use toroidal-approximation capillary force with Lian et al. 1993 rupture criterion (lines 584-588).

## Remeshing / retopologization

- **Baseline (Complete, lines 334-349)**: `_retopologize(HC, bV, dim)` in `dynamic_integrators/_integrators_dynamic.py` runs at the top of **every** step of all 5 integrators (`euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`): scipy Delaunay retriangulation (1D: sorted chain), boundary recompute, `compute_vd(method="barycentric")`.
- **Problem (lines 794-815)**: global Delaunay destroys the sharp phase interface — reconnection creates spurious interface vertices receiving surface tension with wrong curvature → instability growing from corners of the initial shape. Diagnostic (`cases_dynamic/Cube2droplet/diagnostic_no_retopo.py`): with full retopo circularity 0.47→0.72→collapse, dual volume ratio degrades 2:1→11:1; without retopo (fixed connectivity + dual recompute) stable 1.0 s, 36/36 interface vertices preserved, but circularity slowly degrades 0.57→0.47 as triangulation goes degenerate.
- **Solution — `hyperct/remesh/` module (Phase 1–2 Complete, 2D only, lines 817-977)**: local ops edge split / collapse / flip with interface constraints (interface edges never flipped; splits create interface vertices; collapses restricted to same-phase pairs). Thresholds: L_max = 1.4·h_local, L_min = 0.5·h_local, h_local = median 1-ring edge length; quality target min angle > 20°. Driver `adaptive_remesh(HC, dim, mps=None, L_min, L_max, quality_target=20.0, max_iterations=5)` incl. Laplacian smoothing (interface vertices smoothed along interface only). Exposed via `remesh_mode='adaptive'` / `remesh_kwargs=` in `_retopologize` and all 5 integrators (default `'delaunay'`). Tests: `hyperct/tests/test_remesh.py` (36), `ddgclib/tests/test_adaptive_remesh.py` (6). **Phase 3 (3D ops: split/collapse/2-3 face swap/3-2 edge swap, targets aspect ratio < 10, min dihedral > 10°) and Phase 4 (curvature/velocity-gradient/pressure-jump error indicators) not started.** Refs: Persson & Strang 2004; Freitag & Ollivier-Gooch 1997.
- **Known upstream blocker**: `hyperct/remesh/_operations_2d.py:198` mass-averaging bug makes adaptive remesh unusable for the oscillating-droplet fix (line 69-70).
- **Pressure-preserving mass redistribution (Complete, lines 994-1018)**: after retriangulation, dual volumes change → spurious pressure jumps via `P = eos.pressure(m/Vol)`. Fix: snapshot `v.p` before retopo, set `m_target = eos.density(p_before) * Vol_new`, global scale to conserve total mass exactly. Module `ddgclib/operators/mass_redistribution.py` (`snapshot_pressure`, `redistribute_mass_single_phase`, `redistribute_mass_multiphase`); integrator params `pressure_model=None, redistribute_mass=False`. BC-aware (walls excluded, injected vertices excluded, periodic supported). 15 tests.
- **3D sliver-tet dual bug (Fixed, lines 1063-1185)**: re-Delaunay of advected-interior + fixed-boundary vertices creates sliver tets; `_compute_vd_3d`'s nn-intersection `v1.nn ∩ v2.nn ∩ v3.nn` then returns "ghost tets" (4+ candidates), breaking `vd.nn` dual connectivity and the p_ij ring-walk → O(h) instead of machine precision near walls, every timestep. Ghost faces grow with refinement: 0 (refine=1), 90 (refine=2), 868 (refine=3). Fix: simplex-aware dual construction via cached `HC._simplices` (from `tri.simplices` in `_retopologize`), face→tet map / `Delaunay.neighbors`. Workaround for remaining boundary edges: angular-sort fallback (valid polygon, O(h) accuracy).
- Not-started remesh-adjacent features: Rules-Based Mesh Quality Maintenance (1020-1030, duplicates the remesh spec; refs MMG, Geometry Central, CGAL), Intrinsic Delaunay Triangulation (1032-1045), error-indicator Adaptive Mesh Refinement (1187-1196).

## Oscillating droplet case (lines 771-789 and 5-70)

- Files: `cases_dynamic/oscillating_droplet/src/{_params,_analytical,_setup,_plot_helpers,_boundary_conditions,_metrics}.py`, `oscillating_droplet_2D.py`, `oscillating_droplet_3D.py`, `mesh_convergence_2D.py`, `baselines/baseline_oscillation.json`. Analytical: Lamb/Rayleigh solution (Rayleigh frequency 2D/3D, Lamb damping, damped frequency, radius perturbation). Parameter sets: overdamped (oil) + underdamped (water). Tests: `ddgclib/tests/test_case_oscillating_droplet.py` (8 pass, 1 skip, 1 slow).
- **2D validation with per-phase stress (done, lines 48-58)**: 1839 steps, t_end = 0.114 s. Metrics vs baseline: summary 4.29 vs 5.38 (−20%), tail_growth 1.85 vs 2.69 (−31%), l2_error 4.29 vs 5.38 (−20%), mass_drift machine precision. **Still above spec target tail_growth < 1.0.** Static-droplet diagnostic: per-phase scheme has 38% lower initial interface |F| than old own-phase scheme; residuals dispersed, not coherent shrinkage. Closing the gap requires adaptive remesh (blocked by hyperct bug).
- **3D validation (incomplete, lines 59-68)**: run completes (872 steps, t_end = 0.16 s, 472 vertices / 98 interface, mass drift 1e-14) but `oscillating_droplet_3D.py` never emitted an `oscillation_score` → **no checked-in 3D baseline exists**; R_max saturates at ~5·R0 (far mesh boundary). A separate "3D oscillating-droplet metric harness" backlog feature (lines 1262-1266) tracks adding scoring + baseline.
- Note: DEVELOPMENT.md tracks **no oscillating-bubble case**; the only bubble-related code (`cases_mean_flow/equil_bubble/`, `_bubble.py`) is not in this tracker (legacy mean-flow pipeline).

## Solver stability — issues and fixes

1. **Catastrophic KE growth** in oscillating droplet (multiphase surface tension) → per-phase summed interface stress rewrite (In Progress, above).
2. **Interface destruction by global Delaunay** → interface-preserving adaptive remesh (2D done, 3D not started; blocked by hyperct mass-averaging bug for production use).
3. **Spurious discrete compressibility in viscous flux** (Fixed, lines 326-330): symmetric transpose term `d_hat*(du.A)` discretizes μ∇(∇·u) with nonzero trace on diagonal edges even for div-free fields; dropped. Diffusion form F_v = (μ/|d|)·du·(d̂·A) gives exact zero for Poiseuille at machine precision.
4. **Dimensional consistency of stress formulation** (Fixed, lines 320-325): refactored to face-centered integrated formulation via Stokes' theorem on dual flux planes — no vertex-centered gradients, no /Vol division. Pressure force half-difference F_p = −0.5·(p_j − p_i)·A_ij. Old pointwise code archived in `stress_pointwise.py` / `Fundamentals_pointwise.md`; new in `Fundamentals.md`.
5. **Outlet backflow** (Fixed via `OutletBufferedDeleteBC`, lines 381-401): boundary vertices have truncated dual cells (`dual_area_vector()` returns zeros for boundary edges, stress.py:98-100) → imbalanced stress pushes vertices backward. Buffer ghost zone `[outlet_pos, outlet_pos + buffer_width]` with frozen entry velocity + position correction `correct_pos += frozen_u * dt`; `id(v)`-keyed buffer (vertex hash changes on `mesh.V.move()`). The more fundamental `prescribed_V` fixed inlet/outlet architecture (3-tier vertex classification) is fully spec'd but Not Started (lines 403-548).
6. **PeriodicInletBC ghost accumulation** (Fixed, lines 351-362): injected ghost vertices were re-injected every step, creating duplicate chains along walls/corners; now removed from ghost after injection, ghost re-cloned from `unit_mesh` when depleted.
7. **Spurious pressure jumps after retriangulation** → mass redistribution (Complete, item above).
8. **3D boundary sliver tets → ghost tets → broken duals** (Fixed via `HC._simplices` simplex-aware `_compute_vd_3d`; boundary p_ij edges still O(h) per lines 1049/1061 — see contradiction note).
9. **Performance**: per-edge dual area caching via `batch_e_star(orient=True)` → `HC._edge_area_cache`, O(1) lookups in `stress_force` (was ~100M sequential Python calls / 5000 steps; expected 10–50x speedup). GPU/numpy backend physics-equivalence verification for HP3D still unchecked (lines 687-688).

## Contradictions / stale claims noticed

- **"3D p_ij Dual Area Vector" says boundary edges are "Blocked by compute_vd bug (boundary edges)"** (lines 1049, 1061) while the referenced bug section says "Status: Fixed (simplex-aware path enabled via `HC._simplices`)" (line 1065). The unchecked "boundary edges still degrade to O(h)" item was likely not revisited after the fix — unclear whether boundary p_ij precision is now resolved.
- **Duplicate feature entries**: "Integrated Analytical Validation Framework" appears twice (lines 270 and 708), both Complete; "Rules-Based Mesh Quality Maintenance" (1020) duplicates the adaptive remesh spec and is marked Not Started even though remesh Phases 1–2 are complete.
- **Documentation consolidation checkbox inconsistency** (line 1250): "Add core-features overview + quick start to README.md" is `[ ]` unchecked but annotated "*(done 2026-06-08)*".
- **Stale test counts**: feature sections quote snapshots of 279, 450, 470, 532, 558, 604, 652, 733 passing tests; the audit section (line 1251-1252) says actual is **808 passed**, vs ARCHITECTURE.md (~604) and CLAUDE.md (~415) — all hard-coded counts are stale.
- The "Interface-Preserving Adaptive Refinement" architecture text (lines 929-946) describes `remesh_mode='adaptive'` as the "After" state; it IS implemented (Phase 2 checked) — the before/after framing reads as stale planning prose.
- `ddgclib/__init__.py` is empty of exports (only commented-out imports) — `import ddgclib` exposes nothing (line 1223-1226); `__version__` only in `setup.py` (0.4.3).
- `geometry/_dual_split_3d.py` is a dead stub raising NotImplementedError; the real 3D split lives in the misleadingly named `geometry/_dual_split_2d.py` (lines 1233-1235).
