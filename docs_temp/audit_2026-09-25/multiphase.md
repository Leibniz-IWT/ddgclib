# Multiphase / interface method stack: inventory for a method registry

Audit date 2026-09-25, read-only. All anchors are `file:line` against the current working tree.
Abbreviations: `MP` = `ddgclib/multiphase.py`, `MS` = `ddgclib/operators/multiphase_stress.py`,
`MR` = `ddgclib/operators/mass_redistribution.py`, `DS` = `ddgclib/geometry/_dual_split_2d.py`,
`IS` = `ddgclib/geometry/_interface_subcomplex.py`, `CH` = `ddgclib/_curvatures_heron.py`,
`C2` = `ddgclib/operators/curvature_2d.py`, `INT` = `ddgclib/dynamic_integrators/_integrators_dynamic.py`,
`ST` = `ddgclib/operators/stress.py`, `EOS` = `ddgclib/eos/`, `IC` = `ddgclib/initial_conditions.py`,
`DB` = `ddgclib/geometry/domains/_multiphase_droplet.py`, `OD` = `cases_dynamic/oscillating_droplet/src/_setup.py`,
`DAM` = `cases_dynamic/dam_break/src/_setup.py`.

Two probes were run (scripts in this scratchpad: `probe_apex_cache.py`, `probe_stokes_cache.py`; no repo files touched). Results in section 3 (T1, T2).

---

## 1. Data model

### 1.1 Per-vertex attributes

| attr | writers | readers | stale when |
|---|---|---|---|
| `v.phase` (int, `-1` = `INTERFACE_PHASE`, MP:67) | `assign_vertex_phases_from_simplices` MP:336-339; `identify_interface_from_subcomplex` MP:403 (sets -1); legacy `assign_phases` MP:189-190; `PhaseAssignment` IC:431-433; builder `_build_combined_mesh` DB:99-101; dam-break per-vertex labelling DAM:123-124; isolated-vertex default 0 MP:332-333 | simplex vote MP:280-281; neighbour_count split MP:486,500; 2D exact split DS:126,204,212-228; edge fractions DS:592,596,599,617; `MultiphaseEOS.__call__` EOS/_multiphase_eos.py:112,133; `get_gamma` MP:596-598; IC `MultiphaseMass`/`MultiphasePressure` IC:453,486,496; BC `PressureReservoirBC` `_boundary_conditions.py:846` | after vertex motion until next `refresh`; legacy/IC writers bypass the simplex model entirely |
| `v.is_interface` (bool) | MP:400-405 only | nearly everything: MP:481,572; MS:158,274; DS:41,125,203,427,489,542-545,588-589; C2:47; `_apply_perturbation` OD:258; `restore_pressure_multiphase` MR:305; `anchor_phase_pressure_levels` MR:482 | same as `phase` |
| `v.interface_phases` (frozenset) | MP:334,340 | MS:77-80 (`_phases_present`), MS:278,362; neighbour_count MP:494-497; DS:141,216,436,451,464,476,604-605,624; `interface_mean_pressure` EOS/_multiphase_eos.py:51; simplex vote fallback MP:293-296 | same |
| `v.m_phase[k]` | `init_phase_fields` MP:421 (zeros); `compute_phase_masses` MP:529-534 (`rho0_k*dvp_k`); `_reinit_geometry_fields` MP:707-708 (only if missing); `redistribute_mass_multiphase` MR:590,604; YL preload OD:196-199; hydrostatic preload DAM:185-192, `electrolysis_bubble/src/_setup.py:389-417`; BCs with `update_m_phase` `_boundary_conditions.py:853-857,932-936,1025-1029`; hyperct `edge_split_2d` (per OD:220-222) | `compute_phase_pressures` MP:559; presence gates `interface_mean_pressure` EOS:55-61, `restore` MR:298, `anchor` MR:455,472; `MultiphaseEOS` EOS:98-103; redistribution MR:570-572 | NOT touched by `mass_conserving_merge` MP:729-806 (only `v.m` lumped, MP:785-795) |
| `v.m` | `compute_phase_masses` MP:535; MR:614-617 (sum of m_phase); preloads OD:199, DAM:192; `mass_conserving_merge` MP:792; `MultiphaseMass` IC:456-458; BCs | `multiphase_stress_acceleration` MS:397-399 (a = F/m); `_is_redistributable`/single-phase MR:97,165,185 | inconsistent with `sum(m_phase)` after a merge or `MultiphaseMass` IC |
| `v.dual_vol_phase[k]` | `split_dual_volumes` MP:464,474,485-487,514; zeroed by `_reinit_geometry_fields` MP:705 and `init_phase_fields` MP:424 | EOS read MP:558; **presence key** for force MS:98-100, `interface_mean_pressure` EOS:61, restore MR:294-301, anchor MR:454, redistribution MR:544-560, strain advance MR:362-375, `phase_volume_totals` MR:380-387 | whenever `v.dual_vol`/connectivity changes without a `refresh` (e.g. YL preload after `_apply_perturbation` OD:170-199 uses pre-perturbation volumes, see laneD log :226-231) |
| `v.rho_phase[k]` | MP:562,565; EOS/_multiphase_eos.py:105,108,130 | diagnostics only; `MultiphaseEOS` v.rho EOS:136 | |
| `v.p_phase[k]` | MP:563,566; `restore_pressure_multiphase` MR:302; `anchor_phase_pressure_levels` MR:474; `MultiphaseEOS.__call__` EOS:106,109,131 | **the force reads only this** MS:164,211-221; snapshots MR:46-50,66-73 | between retopo calls: never recomputed inside `dudt_fn` (MS:151-154) |
| `v.p` | MP:579-581; MR:305-308, MR:482-485; `MultiphaseEOS` EOS:146; `MultiphasePressure` IC:488,497; `mass_conserving_merge` MP:795 | only fallback in force (MS:165-167 via `_resolve_pressure` ST:701); single-phase `snapshot_pressure` MR:39; visualisation | diagnostic in the multiphase pipeline |
| `v.rho` | EOS:136-145 | diagnostics | |
| `v.dual_vol` | `_retopologize` INT:270-275 (boundary zeroed); `cache_dual_volumes` ST:433-466 (NOT boundary zeroed); cube2droplet case direct `dual_volume` loop | neighbour_count MP:480; 3D exact rescale DS:393-397,516; redistribution gate MR:85; `MultiphaseMass` IC:454 | source differs by dim and by call site (axis A2) |
| `v.u`, `v.x_a`, `v.nn`, `v.vd`, `v.boundary` | integrator / hyperct | everywhere | |

### 1.2 HC-level and mps-level caches

| cache | writer | readers | invalidation |
|---|---|---|---|
| `mps.simplex_phase` (dict frozenset(v.x) -> int; "authoritative", MP:156-161) | `assign_simplex_phases` MP:239-246; `assign_simplex_phases_from_vertices` MP:278-298; copied from builder OD:156, `electrolysis/_setup.py:366` | `extract_interface` IS:77-84; `assign_vertex_phases_from_simplices` MP:319-325 | keyed on `v.x`: **stale after any vertex move** (keys no longer match); every `refresh` rebuilds it (MP:660-670) |
| `mps._simplex_criterion_fn` | MP:240; copied OD:157 | `refresh` MP:660 | only used when `reset_mass=True` |
| `mps.vol_corr` (ndarray, ones) | MP:176; `_retopologize_multiphase` INT:708-709 | `compute_phase_pressures` MP:561 only (NOT `MultiphaseEOS` EOS:104) | never reset (persists across `refresh(reset_mass=True)`) |
| `mps._remap_vol_tar`, `mps._remap_rho_ref` | MR:435-448,466 | MR:467 | never reset; first call is decided by attribute absence MR:435 |
| `mps._projection_call_idx` | INT:621-623 | INT:622 | never reset |
| `HC.interface_vertices / interface_edges / interface_triangles` | `extract_interface` IS:112-114 (2D: `interface_triangles = set()`) | IS:122-166, 174-241; DS:132-136, 547-549; CH:73, 461 | rebuilt on every `refresh` |
| `HC._simplices` | hyperct `connect_and_cache_simplices` (`hyperct/ddg/_retriangulation.py:143-147`); auto-populated by MP:231-237, 270-276; `rebuild_simplex_cache_2d` INT:180 | `iter_top_simplices` MP:90-97; exact volumes ST:311-324; `boundary_from_simplices` | `invalidate_simplex_cache` (hyperct :150-171) after merges/BC injection/adaptive 3D |
| `HC._edge_to_apex` | hyperct `get_edge_apex_map` :279-302 | CH:53 (`hndA_i`, not the interface path) | cleared by invalidate/rebuild only; **not cleared by `connect_and_cache_simplices`** (hyperct :143-147) |
| `HC._interface_edge_to_apex` (id-keyed) | CH:76-89 (built once) | CH:349 (`hndA_i_interface`, 3D 'integrated' + 'csf_dual') | **never invalidated anywhere** (only writer CH:89). See T1 |
| `HC._interface_x_to_v` (v.x-keyed) | CH:466-469 (built once) | CH:482-485 (`integrated_hndA_i_interface`, 'stokes') | **never invalidated; stale after the first vertex move**. See T2 |
| `HC._edge_area_cache` (id-keyed, interior only) | INT:276 / INT:280 (None) | MS:172-181; ST:512,585,826 | rebuilt each retopo call; stale when retopo is skipped (`retopologize_fn=False`, displacement gate INT:410-412) |
| `HC._vd_method` | hyperct `compute_vd` | ST:323 | |
| `HC._retopo_prev_positions` | INT:334-336 | INT:317-329 | |

---

## 2. Method axes (config schema candidates)

Legend: **[explicit]** kwarg exists; **[implicit]** chosen by dim / cache presence / another flag; status tags: VALIDATED (default, pinned), OPT-IN (tested, not default), DEAD (measured worse / do-not-retry), UNTESTED.

### A1. Simplex phase-label source (interface identification)
- Values: `criterion` (centroid test `criterion_fn(centroid)`, MP:241-246) | `vertex_vote` (majority of bulk `v.phase` per simplex, MP:279-298) | legacy `per_vertex` (`assign_phases` MP:181-190, `PhaseAssignment` IC:417-433; bypasses the subcomplex).
- Where: **[implicit]** in `refresh` MP:660-670: `criterion` iff `reset_mass and fn is not None`, else `vertex_vote`. An explicit `criterion_fn=` passed with `reset_mass=False` is **silently ignored** (MP:660-661).
- Math: interface face = primal face shared by top-simplices of >= 2 phases (IS:90-93); in 3D interface edges = edges of interface triangles (IS:105-110). Vertex is interface iff incident simplices carry >1 phase (MP:336-339).
- Tie rule: docstring says lower phase id (MP:259-261) but code is `Counter.most_common(1)` (MP:286-287) = first-encountered on ties, i.e. vertex-order dependent. All-interface simplex -> `min(interface_phases)` (MP:297).
- Case usage: droplet cases `criterion` at setup (DB:184, OD:164), `vertex_vote` at runtime (INT:654,686). Dam break never registers a criterion (DAM:169,173) so even its `reset_mass=True` setup uses `vertex_vote` from per-vertex labels DAM:123-124.
- Tests: `test_interface_subcomplex.py`, `test_multiphase.py`; no direct test of `assign_simplex_phases_from_vertices` or the tie rule.

### A2. Closure validation strictness
- Values: strict (raise) | lenient (swallow). **[implicit]** `strict_closure=reset_mass` (MP:682-684). Lenient path is `except ValueError: pass` (MP:390-393), no warning despite docstring MP:365-367.
- Boundary contact vertices are excluded from the degree check (IS:197-204, 228-233).
- Tests: `validate_closure` in `test_interface_subcomplex.py`, `test_simplex_aware_curvature.py`; `strict_closure` untested.

### A3. Total dual volume source (`v.dual_vol`) feeding the split
- Values: `simplex_exact` (`Vol_i = (1/(d+1)) sum_T |T|`, `hyperct.ddg.simplex_dual_volumes`) | `batch_e_star` fan walk | `v_star` fan walk (3D legacy) | `dual_cell_area_2d` polygon.
- Where: **[implicit]** by dim, call site and cache presence:
  - runtime retopo 2D: `batch_e_star` volumes (INT:273-275); 3D: `simplex_exact` when `_use_exact_barycentric_volume(HC)` (INT:262-272, ST:311-324).
  - setup `cache_dual_volumes`: `simplex_exact` in 2D AND 3D when `HC._simplices` exists (ST:448-459), else per-vertex `dual_volume` (ST:379-423).
  - boundary vertices zeroed only in the retopo path (INT:272,275), not at setup (ST:457-458) -> first-retopo boundary volume jump (06_known_issues footgun 3).
- Status: 3D exact VALIDATED 2026-07-29 (ST:386-403, INT:241-261; laneA). 2D kept on fan walk for bit-identity.

### A4. Per-phase dual volume split (`split_method`)
- Values: `'neighbour_count'` (default) | `'exact'`. **[explicit]** `split_dual_volumes(method=)` MP:428-430, `refresh(split_method=)` MP:615, `_retopologize_multiphase(split_method=)` INT:486, `setup_oscillating_droplet(split_method=)` OD:41. Any other string (typo) silently falls through to neighbour_count (MP:454 only tests `== 'exact'`; no validation).
- `neighbour_count` (MP:479-514): bulk -> all `v.dual_vol` to own phase; interface -> fraction of 1-ring bulk neighbours per active phase times `v.dual_vol`; all-interface ring -> equal split.
- `exact`, **dimension picks the algorithm [implicit]**:
  - 2D `split_dual_polygon_2d` DS:101-232: barycentric dual polygon with edge midpoints, clipped by the polyline m(v,v_prev) -> v -> m(v,v_next); curve neighbours from `HC.interface_edges` (DS:132-136). Areas come from `hyperct dual_cell_area_2d` / shoelace (DS:96-98, 195-196) and are **not rescaled to `v.dual_vol`**; bulk vertices get `dual_cell_area_2d` even when `v.dual_vol` was zeroed (boundary) (DS:125-128). Two-phase hardcoding `1 - phase` (DS:210-223).
  - 3D `split_dual_polyhedron_3d` DS:400-528: PCA tangent plane from interface neighbours (DS:239-263), `dual_cell_faces_3d` fan-tets clipped by plane (DS:306-375), then **rescaled to `v.dual_vol`** (DS:513-523). Fallbacks to equal split at DS:433-444, 450-459, 463-472.
- Status: default neighbour_count VALIDATED (all pins). `exact`: DEAD as default flip, 3D end-to-end 1.75e-3 vs 1.44e-3 worse, 2D retopo-neutral (docs_temp/06_known_issues:85). Cases all hardcode `neighbour_count` (electrolysis `_setup.py:371,437,475`; shearing plate `_setup.py:226`).
- Tests: `test_dual_split_2d.py` (both 2D and 3D unit tests), a5b regression (`test_a5b_longrun_regression.py:105,135`, neighbour_count only).

### A5. Interface-edge dual-face phase fraction (force side)
- Values: only one rule set, no kwarg. `edge_phase_area_fractions` DS:561-628, called at MS:189-191 and MS:370.
- Rules: bulk-bulk -> `{v_i.phase: 1}` (DS:591-593); iface-bulk(k) -> `{k: 1}` (DS:595-599); iface-iface on `HC.interface_edges` -> **50/50** (DS:601-609); iface-iface chord -> majority bulk phase of `v_i.nn & v_j.nn` (DS:616-622), fallback equal split over `v_i.interface_phases` (DS:624-628, asymmetric in i/j). Docstring rule 4 (DS:580-581) is stale vs code.
- `dim` switches the legacy (no-HC) curve-adjacency rule only (DS:551-558); with `interface=HC` (always passed by MS:190) adjacency is exact set membership in both dims.
- **Not coupled to A4**: even with `split_method='exact'` the face split stays 50/50 (inconsistent with the volume clip on asymmetric geometry). Flagged by laneG §6.5 and closure §"open" item 4 as a pressure-side hygiene target (bias flips sign with refinement).
- Tests: `test_dual_split_2d.py`.

### A6. Phase-presence gate (force and remap)
- Values: geometric (`dual_vol_phase[k] > 1e-30`, MS:83-101) | legacy pressure sentinel (removed from force; still selectable in redistribution by snapshot format, A11).
- Symmetric one-sided extrapolation: absent-at-i -> face pressure `p_j` at both ends (MS:214-221), absent-at-j -> `p_i` (MS:213). Status VALIDATED (lane1 zero-gauge fix). Tests: `test_multiphase_gauge_invariance.py` (gauge, net momentum, absent-phase, zero-gauge average).

### A7. Interface representative pressure `v.p`
- Values: arithmetic mean of `p_phase[k]` over geometrically present phases (`interface_mean_pressure` EOS/_multiphase_eos.py:37-63). Single convention shared by MP:579, EOS:142, MR:306, MR:483.
- Not read by the multiphase force (it reads `p_phase`), so it only matters for diagnostics, `snapshot_pressure` (single-phase), `mass_conserving_merge` averaging (MP:790) and the fallback `pressure_model` path.
- Tests: `test_eos_consistency.py`, `test_multiphase_gauge_invariance.py:201-259`.

### A8. Face viscosity at cross-phase faces
- Values: only "exact per-phase `mu_k` on the phase-k sub-face" (`_face_viscosity_for_phase` MS:116-126). **Harmonic mean is NOT implemented** although promised in MS:21-23 (stale docstring; also DEVELOPMENT.md per 06_known_issues:8).
- Viscous flux form `viscous_flux(mu_k, du, d_ij, A_k)` MS:225 (ST diffusion form). Artificial viscosity is case-level only: `mu_eff = mu + alpha_art*rho*c_s*dx_mean` baked into `PhaseProperties.mu` at setup (DAM:103-114; `alpha_art` in `cases_dynamic/dam_break/src/_params.py:112`, = 0.3).
- Tests: `test_multiphase_stress_per_phase.py:108-155` (asserts exact mu_k, "not harmonic mean"), `test_multiphase.py:201`.

### A9. Surface tension / curvature path (`curvature_path`)
- **[explicit]** kwarg on `multiphase_stress_force`/`multiphase_stress_acceleration` (MS:135, MS:391); default `'integrated'`. No case binds it (production value is always the default; only the a5b test passes it explicitly `test_a5b_longrun_regression.py:107,137`). Docstring MS:248 lists only `{'integrated','csf_dual'}`, stale.
- Values x dim **[implicit dim branch]**:
  | value | 2D | 3D | status |
  |---|---|---|---|
  | `'integrated'` | FTC `gamma (t_next - t_prev)` via `surface_tension_force_2d` (MS:315-318, C2:152-181) | `-gamma * hndA_i_interface` cotan-Heron on interface sub-mesh (MS:311-314, CH:310-393) | VALIDATED default |
  | `'stokes'` | identical to 'integrated' (MS:301-303) | `integrated_hndA_i_interface` conormal boundary integral (MS:290-300, CH:396-522) | OPT-IN, bit-identical to 'integrated' on the static 3D mesh (laneG §1.1 table); **broken in dynamics (T2)**; no MS-level test |
  | `'csf_dual'` | FTC magnitude redirected along `S_inner` (MS:321-382) | Heron magnitude, same redirect | A/B probe only, "not a replacement" (MS:270-272); inner phase = highest id (MS:365); UNTESTED |
  | other | ValueError (MS:305-309) | | |
- Interface neighbour set passed to all stencils = `{nb in v.nn : nb.is_interface}` (MS:274), NOT `interface_nn` (edge membership, IS:122-142).
- **Curve-neighbour / apex source [implicit]**:
  - 2D FTC always uses the angular-gap heuristic `_select_curve_neighbours` (C2:50-88, C2:134); it never receives HC. The 2D exact split uses exact `curve_neighbours` from `HC.interface_edges` (DS:132-136). Heuristic measured correct on all real droplet 1-rings but wrong on a chord-contaminated 1-ring (docs_temp/audit/curvature-2d-consistency.md:42-61).
  - 3D Heron apex: `HC._interface_edge_to_apex` from `HC.interface_triangles` if the attribute exists (always true after `extract_interface`), else legacy `vi.nn & vj.nn & interface_set` (CH:349-357). Legacy vs simplex apex measured bit-identical on A.5 numbers (06_known_issues:87).
- gamma = `get_gamma_pair(sorted(phases)[0], sorted(phases)[1])` (MS:282-283): two-phase only; triple junctions ignored. Surface tension requires >= 2 interface neighbours (MS:274-276).
- Unused alternatives in the codebase: full-mesh `surface_tension_force` / `hndA_i` (`operators/surface_tension.py:28-61`, thin-film only, no duals); `reconstruct_arc_length_and_bulge_area` (C2:184-231, exported but unused; laneH §1.4 names it the lever for the l=0 bump); `Curvature_i` registry (`operators/curvature.py:12-66`, legacy mean-flow, not wired to multiphase).
- Tests: `test_curvature_2d.py` (FTC), `test_simplex_aware_curvature.py` (`hndA_i_interface`, `integrated_hndA_i_interface`, apex cache on a static mesh), `test_multiphase_flat_interface.py` (gamma>0, kappa=0 cancellation 2D/3D).

### A10. Phase EOS and clipping
- Per phase: `PhaseProperties.eos` (MP:126), any `EquationOfState` (EOS/_base.py:9-28): `TaitMurnaghan(rho0,P0,K,n,rho_clip)` (EOS/_tait_murnaghan.py:65-83; `P = P0 + K/n((rho/rho0)^n - 1)`), `IdealGas(rho0,T,R,P0)` isothermal (EOS/_ideal_gas.py:33-60; `density` floors P at 0).
- `rho_clip`: default `(0.9, 1.1)` (EOS/_tait_murnaghan.py:71), saturating in `pressure`, `density`, `sound_speed` coherently (:87-147), `clip_count` + warn-once (:80-106). Case values: droplet (0.8,1.2) OD:116-119; dam break gas (0.2,5.0) / liquid (0.8,1.2) DAM:133-138; cube2droplet (0.1,10). Note the builder's placeholder mps uses defaults `TaitMurnaghan(rho0=1000)` with P0=101325 and clip (0.9,1.1) (DB:176-183) and gamma unset; cases replace it (OD:121-157).
- EOS read: `rho_k = m_k / (vol_corr[k]*dvp_k)` in `compute_phase_pressures` MP:556-566. `MultiphaseEOS` (EOS/_multiphase_eos.py:66-151) duplicates the read WITHOUT `vol_corr` (EOS:104) and is only reached when `v` lacks `p_phase` (MS:163-167): effectively dead in the production pipeline (laneD §2.6.4).
- Clip telemetry: ~5.7e4 transient outer-phase clips per 2D delaunay_remap run on pre-restore evaluations (laneE §1 probes, `_params.py:111-116`). Health tripwire needs pre-restore vs genuine split.
- Tests: `test_eos_consistency.py` (round trip, clip_count, interface mean), `test_multiphase.py`, `test_mass_redistribution.py` (IdealGas).

### A11. Mass redistribution (projection) policy
- Values and where:
  - `redistribute_mass` bool **[explicit]** INT:484 (default False), bound True by all shipped multiphase setups (OD:42,232; DAM:64,216; electrolysis :240,476; cube2droplet `_setup.py:166`).
  - `projection_every: int >= 1` **[explicit]** INT:488 (default 1 = every call).
  - off-cadence snapshot source **[implicit]**: under remap -> `evolve_snapshot_local_strain` (INT:656-669, MR:312-377: `p_new = eos(eos^-1(p_snap) * dvp_snap/dvp_now)`); under dual_only -> redistribution block skipped (INT:631-632, 689). The raw `eos(m/dual_vol)` recompute is not selectable (good: DEAD).
  - snapshot format **[implicit]**: `_retopologize_multiphase` always uses geometry-aware `snapshot_geometry_multiphase` (INT:633-636). Legacy `snapshot_pressure_multiphase` (MR:42-51) flips the presence guard to `p > 1e-30` (MR:552-557), the pre-lane1 P0=0 bug; only reachable by calling MR directly (used only in tests).
- Math: `m_k,i = scale_k * eos_k^-1(p_snap_k,i) * dvp_new_k,i`, `scale_k = M_k / sum(...)` + residual fixup (MR:536-611). Exclusions: `bV`, `dual_vol < 1e-30`, vertices not in snapshot (MR:81-89).
- Guards (code-enforced): `projection_every > 1` requires mps + redistribute_mass (INT:608-613) and, if reconnecting, the remap (INT:614-620).
- Status: every-call projection VALIDATED (pins) but it is the attributed cause of 2D over-decay (laneH verdict) and ~half the 3D bump under dual_only (laneG §1.2b). `projection_every=2..20` under delaunay_remap OPT-IN: 2D l2 0.0380 vs 0.1748 (laneH table), not adopted (tail score calibration). `redistribute_mass=False`: 2D rings (l2 0.495, laneH §1.3), 3D l2 0.266 vs 0.248 but every physics channel cleaner (laneG §2), rejected by rule. 3D cadence UNTESTED (laneH §6.6).
- Tests: `test_mass_redistribution.py`, `test_case_oscillating_droplet.py::TestProjectionCadence2D` (validation, bit-identity, frozen-position neutrality, dual_only and remap cadence).

### A12. Retopology remap closure
- `retopo_remap in {None, 'conservative'}` **[explicit]** INT:487, validated INT:592-602. `'conservative'` bundles three pieces that are not individually selectable: volume gauge `vol_corr = scale_k` (INT:708-709), structure restore `restore_pressure_multiphase` (MR:245-309, INT:725), level anchor `anchor_phase_pressure_levels` (MR:390-486, INT:734-737), plus a stage-1 measurement pass on old connectivity (INT:652-655).
- **[implicit]** no-op when `skip_triangulation=True` or `mps is None` (INT:597-598).
- Status: 2D default via runner (`oscillating_droplet_2D.py:98-99`, `_params.py:139`), dam break runner (`dam_break_2D.py:121`). 3D REJECTED (l2 1.873 vs 0.248, laneE §1d; `_params.py:159-169`). Sub-variants DEAD: restore-only v1, restore+gauge without anchor v2 (laneD §2.2).
- Tests: `test_case_oscillating_droplet.py` (`TestConservativeRetopoRemap2D`, `TestDelaunayRemapEndurance2D`), `test_case_dam_break.py`. No unit tests of `restore_pressure_multiphase`, `anchor_phase_pressure_levels`, `evolve_snapshot_local_strain`, `vol_corr` in isolation.

### A13. Connectivity update policy (retopology mode)
- Values: per-step full Delaunay (default) | `dual_only` (`skip_triangulation=True`) | `adaptive` (`remesh_mode='adaptive'`, 2D only, INT:156-189) | displacement gate (`displacement_eps`, INT:300-336, 410-412) | periodic (`periodic_axes`, INT:125-132, **not declared by `_retopologize_multiphase`**, so periodic multiphase needs the case-local copy `shearing_plate_droplet/src/_setup.py:330-374`, which lacks remap/cadence/vol_corr).
- Where: `skip_triangulation` bound in runner partials (`oscillating_droplet_3D.py:106-107`, `oscillating_droplet_2D.py:96-97`) or forwarded by name from the integrator (INT:448-461); `remesh_mode` always forwarded from the integrator (INT:427-430, see T5).
- Status: 2D delaunay(+remap) default; 3D dual_only default (`_params.py:169`); plain Delaunay without remap DEAD (2D l2 0.490, 3D 1.524); displacement-gated / hybrid DEAD (`_params.py:87-89`); adaptive OPT-IN (`oscillating_droplet_2D_adaptive.py`, upstream fixes OD:217-228). Tests: `test_adaptive_remesh.py`, `test_retopo_displacement_gate.py`, floor battery.

### A14. EOS update cadence (implicit)
- Pressures are recomputed ONLY inside the retopo function (`mps.refresh` INT:654/686 and `compute_phase_pressures` INT:713); `multiphase_stress_force` never calls the EOS (MS:151-154). So with `retopologize_fn=False`, a skipped displacement-gate step, or the intra-step stages of `rk45`/`euler_adaptive`, forces read stale pressures at moved positions. **[implicit]**, no kwarg.

### A15. Mass preload ICs (setup)
- Library ICs: `MultiphaseMass` (IC:436-458, `rho0_phase * dual_vol`), `MultiphasePressure` (IC:461-497, bulk `eos(rho0)` + `gamma*kappa` on inner phase, assumes outer = phase 0 IC:492), `PhaseAssignment` (IC:417-433). All three index by `v.phase` and wrap to the LAST phase on interface vertices (`v.phase=-1`, IC:453,486); they predate the per-phase data model and are used only in `test_multiphase.py`. Single-phase `HydrostaticEOSMass` (IC:329-412) writes `v.m` only.
- Production preloads are hand-written per case, each a variant of "`m_phase[k] = eos_k^-1(P_target_k(x)) * dvp_k` then `refresh(reset_mass=False)`":
  - YL (analytic jump `gamma (dim-1)/R0`, bulk-uniform) OD:187-203; shearing plate `_setup.py:259-276`; electrolysis gas `_setup.py:401-417`.
  - Hydrostatic per phase DAM:175-196 (done AFTER one `_retopologize` so volumes match step 0, DAM:171-173); electrolysis liquid `_setup.py:389-399`.
- Status: analytic YL VALIDATED; scalar discrete-consistent 3D YL DEAD (no-op, laneG §1.2, OD:178-186); flat-pressure IC for gravity under redistribution DEAD (DAM:149-163, laneF). Not tested as a unit.

### A16. Other choices found
- Body force (gravity): case-level closure wrapping `dudt_fn` (DAM:209-213; electrolysis `_setup.py:461-472`), not part of `multiphase_dudt_i`.
- Degenerate-cell handling: none in library; case-level NaN sweeps (electrolysis `_setup.py:373-387,419-427`) and `dudt_fn` zeroing on non-finite mass. Sliver-cell F/m ejection at reconnection is the open dam-break blocker (closure doc §F, item 2).
- Vertex merge: `mass_conserving_merge` (MP:729-806) lumps `v.m`, averages `p`, ignores per-phase arrays; called after the perturbation OD:173 and in electrolysis `_setup.py:436`. DO-NOT without per-phase ledger (closure doc :140).
- Pressure BCs with `update_m_phase` (`_boundary_conditions.py:784-1030`) act on bulk gas vertices only (phase test :846) and overwrite `m_phase[gas] = v.m` (valid only for bulk).
- Remesh mass transfer lives in hyperct `edge_split_2d` (OD:220-222).

---

## 3. Consistency traps

| # | choices that must match | enforced? |
|---|---|---|
| T1 | 3D `'integrated'` apex map `HC._interface_edge_to_apex` vs current `HC.interface_triangles` | **No.** Built once (CH:76-89), never cleared. Probe (3D refine 1/1, one Delaunay retopo after a 1e-3 jitter that changed 22 of 48 interface triangles): stale vs fresh F_st differ by 1.15e-3 N against max|F| 1.05e-3 N (100 %). Harmless under 3D dual_only (connectivity frozen; small jitter kept 48/48 triangles and diff 1e-19). Latent in any 3D per-step Delaunay run, i.e. the configs measured worse in laneB/laneE (plain Delaunay l2 1.52, remap 1.87); not yet isolated as a cause. |
| T2 | 3D `'stokes'` coordinate map `HC._interface_x_to_v` vs current `v.x` | **No.** Built once (CH:466-469); `HC.V.move` re-keys `v.x`, so after the first move + refresh triangles are skipped (CH:482-485). Probe: stale-vs-fresh diff equals the full force magnitude (6.0e-4 N) even with unchanged topology. `'stokes'` is unusable for dynamics as shipped. |
| T3 | `split_method` at setup vs runtime partial | Only by convention: threaded in `setup_oscillating_droplet` (OD:72-80, 164, 203, 231); other cases hardcode `'neighbour_count'` in both places. Mismatch reproduces rho x147 (2D) / x4.7 (3D) (06_known_issues:47). Dam break relies on both defaults (DAM:169-196, 215-216). |
| T4 | 2D `'exact'` split vs `v.dual_vol` | **No.** 2D exact never reads `v.dual_vol` (DS:96-98,127) while 3D rescales to it (DS:513-523) and neighbour_count multiplies it (MP:480,514). After a retopo, 2D boundary vertices have `dual_vol=0` (INT:275) but nonzero exact `dual_vol_phase`. |
| T5 | `remesh_mode`/`remesh_kwargs` bound in a partial vs integrator value | **Inverted precedence.** The laneF rule "partial binding wins" (INT:431-461) is applied to 8 kwargs but NOT to `remesh_mode`/`remesh_kwargs` (INT:427-430), which are always forwarded and so override a partial binding (call kwargs beat partial keywords). Documented only in a runner comment (`oscillating_droplet_2D.py:148-151`). |
| T6 | integrator-level `redistribute_mass`/`skip_triangulation` vs retopo partial | Forwarded by name unless partial-bound (INT:448-461). A multiphase partial that does not bind `redistribute_mass` inherits the integrator's value (default False). With `retopologize_fn=None` and `redistribute_mass=True, pressure_model=meos`, the SINGLE-phase `redistribute_mass_single_phase` runs on a multiphase mesh (INT:141-144, 291-297) and writes `v.m` without `m_phase`. `pressure_model` is not declared by `_retopologize_multiphase`, so never forwarded there. |
| T7 | `retopo_remap` / `projection_every` vs `redistribute_mass`, `skip_triangulation` | Enforced by ValueError (INT:599-620). Silent no-op of the remap when `skip_triangulation=True` (INT:597-598). |
| T8 | snapshot format vs redistribution presence guard | Implicit (MR:222-242, 552-557). Mixed-format snapshots would crash (`dual_vol_view` set to None then indexed, MR:238-241). The integrator always uses the geometry format. |
| T9 | `vol_corr` gauge vs every EOS read | Only `compute_phase_pressures` applies it (MP:561). `MultiphaseEOS.__call__` (EOS:104) and `evolve_snapshot_local_strain` (MR:372-375, uses dvp ratios, fine) do not; any direct EOS consumer during a remap run sees ungauged densities (laneD §2.6.4). |
| T10 | mps run-state (`vol_corr`, `_remap_*`, `_projection_call_idx`) vs a fresh run | Never reset (MP:176; MR:435; INT:621). Reusing an mps object across A/B runs in one process carries remap targets and cadence phase. |
| T11 | `simplex_phase` (v.x keys) vs moved vertices; interface tags vs positions | Any consumer between a move and the next `refresh` (e.g. `_apply_perturbation` OD:170 followed by the YL preload OD:192-199 on pre-move `dual_vol_phase`; laneD §2.2 "v3 init fix") reads stale geometry. |
| T12 | `_edge_area_cache` vs positions | Stale when retopo is skipped (see A14). Interior-only (INT:233-237); others fall back to `dual_area_vector` (MS:178-181). |
| T13 | 2D FTC curve neighbours vs 2D exact-split curve neighbours vs `interface_nn` | Three different neighbour rules (C2:50-88 heuristic; IS:145-166 exact; MS:274 flag-only). Agree on real droplet meshes (audit curvature-2d-consistency probe 2) but not by construction. |
| T14 | `strict_closure` and label source vs `reset_mass` | Both tied implicitly to `reset_mass` (MP:661, 682-684). A setup-time `refresh(reset_mass=False)` (preload finalisation OD:203, DAM:196) swallows closure errors. |
| T15 | `mass_conserving_merge` vs per-phase ledger | Not enforced; must be followed by `refresh(reset_mass=False)` and even then `m_phase` of merged vertices is lost (MP:784-795). |
| T16 | `HC._edge_to_apex` vs `HC._simplices` (hyperct) | `connect_and_cache_simplices` replaces `_simplices` without clearing `_edge_to_apex` (hyperct `_retriangulation.py:143-147` vs :166-168). Not on the multiphase force path (only `hndA_i`, `_bubble.py`, periodic, hyperct curvature), but a hyperct hygiene bug. |

---

## 4. Dead / duplicated / suspicious

- `ddgclib/geometry/_dual_split_3d.py` (17 lines): pure re-export shim, zero importers (only cited in MP:448 docstring). Dead.
- `MultiphaseEOS.__call__` as `pressure_model`: dead in production (MS:163-167), duplicates `compute_phase_pressures` minus `vol_corr`. Every case still builds and passes it (OD:210-214, DAM:203-207).
- `MultiphaseSystem.assign_phases` (MP:181-190), `get_gamma` (MP:591-599), `phase_vertices` (MP:712-714), `interface_vertices()` (MP:716-718): no production callers.
- IC classes `PhaseAssignment`, `MultiphaseMass`, `MultiphasePressure` (IC:417-497): legacy per-vertex model, interface-vertex wrap bug, test-only.
- `snapshot_pressure_multiphase` (MR:42-51): legacy format, re-arms the P0=0 guard bug if used; exported (`operators/__init__.py:55`).
- `_csf_dual_surface_tension` (MS:321-382): A/B probe, untested, arbitrary inner-phase convention.
- `'stokes'` path: 3D bit-identical to 'integrated' statically and broken dynamically (T2).
- `reconstruct_arc_length_and_bulge_area` (C2:184-231): exported, unused.
- Duplicated curvature: `hndA_i` / `hndA_i_interface` / `int_hndA_i` (CH:213, 310, 557) plus torch copy `_curvatures_heron_torch_vectorized.py`; `integrated_hndA_i_interface` (CH:396). Inherited `e_ij = -e_ij  # WHY???` sign (CH:158,241,585; mirrored CH:361).
- Duplicated `_interface_neighbours` (C2:44-47 and DS:39-41) and three curve-neighbour rules (T13).
- Duplicated retopo wrapper: periodic multiphase copy in `shearing_plate_droplet/src/_setup.py:330-374` (no remap/cadence); cube2droplet `retopo_fn(**kwargs)` closure (`cube2droplet/src/_setup.py:165-167`).
- Single-phase vs multiphase redistribution: `redistribute_mass_single_phase` (MR:105-215) is reached from `_retopologize` (INT:291-297); multiphase from `_retopologize_multiphase` (INT:689-713). Shared helpers `_is_redistributable`; `_compute_conserved_mass` (MR:92-98) is unused.
- Stale docstrings: harmonic mean (MS:21-23), curvature_path options (MS:248), closure warning (MP:365-367), tie rule (MP:259-261), edge-fraction rule 4 (DS:580-581), `MultiphaseEOS` usage promoting `pressure_model` (EOS:27-28).
- Builder placeholder mps (DB:176-183) with default Tait P0=101325, clip (0.9,1.1), no gamma: easy to use by accident.

---

## 5. What a case author must decide today, and where

| decision | where it lives today |
|---|---|
| Geometry + initial phase labels | domain builder (`droplet_in_box_2d/3d` DB:190/313 with centroid criterion DB:172-174) or per-vertex labels + `vertex_vote` (DAM:123-124); must copy `simplex_phase` and `_simplex_criterion_fn` from builder mps into the case mps by hand (OD:155-157) |
| Phases: EOS class, rho0, P0, K, n, rho_clip, mu, name | `PhaseProperties`/`TaitMurnaghan` in case setup (OD:116-127, DAM:133-145); `_params.py` for values |
| gamma per pair | `MultiphaseSystem(gamma={(0,1): g})` (OD:126) |
| split_method (setup AND runtime, must match) | `mps.refresh(split_method=)` at setup + `partial(_retopologize_multiphase, split_method=)` (OD:164,203,231) |
| Mass preload (YL / hydrostatic) | hand-written loops in case setup (OD:187-203, DAM:175-196) |
| Artificial viscosity | baked into `PhaseProperties.mu` at setup (DAM:103-114), value in `_params.py:112` |
| Gravity / body force | closure around `multiphase_dudt_i` (DAM:209-213) |
| dudt: curvature_path, pressure_model | `partial(multiphase_dudt_i, dim, mps, HC, pressure_model=meos[, curvature_path])` (OD:211-214) |
| redistribute_mass | setup kwarg bound into the retopo partial (OD:42,232); else integrator kwarg forwarded (T6) |
| retopo policy (delaunay / dual_only / remap / cadence) | runner, from `_params.py` string (`retopo_policy_2d/3d`, `_params.py:139,169`) mapped to `partial(retopo_fn, skip_triangulation=True)` or `retopo_remap='conservative'` (`oscillating_droplet_2D.py:96-99`, `_3D.py:106-107`, `dam_break_2D.py:121`); `projection_every` only via partial (never set by a shipped runner) |
| remesh_mode / remesh_kwargs | integrator kwarg ONLY (T5) (`oscillating_droplet_2D.py:152-157`) |
| displacement_eps, boundary_filter, merge_cdist, backend, periodic | integrator kwargs (INT:1018-1027), forwarded by name if declared (periodic is not declared: needs a custom wrapper) |
| BCs (walls, pressure reservoirs with `update_m_phase`) | `BoundaryConditionSet` in setup |
| dt / CFL | case (`_params.py:179`, `dam_break/src/_setup.py:381-391`) |

Implicit (no knob today, value fixed by dim, flags or cache presence): label source and closure strictness (`reset_mass`), dual-volume source (dim + `_simplices` + call site), exact-split algorithm and whether it is rescaled (dim), 2D curve-neighbour rule (heuristic), 3D apex source (presence of `interface_triangles`), edge phase fraction (50/50), face viscosity (mu_k), interface `v.p` convention (mean), EOS update cadence (tied to retopo calls), off-cadence snapshot construction (policy-dependent), remap sub-stages (bundled).

Suggested registry axes (one field each): `label_source`, `closure_strict`, `dual_volume_source`, `phase_split`, `edge_face_fraction`, `presence_gate`, `interface_pressure`, `face_viscosity`, `artificial_viscosity`, `curvature_path`, `curve_neighbour_rule`, `apex_source`, `eos[k]` (+`rho_clip`), `eos_update_cadence`, `redistribute`, `projection_every`, `offcadence_snapshot`, `retopo_remap`, `connectivity_policy` (delaunay/dual_only/adaptive/gate/periodic), `mass_preload`, `body_force`, `merge_policy`. Validate at construction: T3, T5, T6, T7, T10 and the dim restrictions (adaptive 2D only; 'stokes'/'csf_dual' flagged experimental until T1/T2 are fixed).
