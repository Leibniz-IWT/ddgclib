# 06 — Known Issues, Debug History, and Open Problems
> Sources: sources/debugging_plan_distilled.md, sources/library_audit_and_features.md, sources/development_status.md, code_map/* (all) | Written: 2026-07-02 by understand-and-document workflow

**This is the launch pad for the debugging workflow.** Section 1 is the ranked open-problem list; Section 2 the DO-NOTs (lanes already closed); Section 3 the full bug ledger with status; Section 4 the hypothesis/experiment history that produced those statuses; Section 5 footguns and stale claims. Primary underlying log: `debugging_plan.md` (repo root, reverse-chronological, 2026-04-24 → 2026-06-02), distilled at [sources/debugging_plan_distilled.md](sources/debugging_plan_distilled.md).

**Framing:** the spatial discretisation is machine-precision at equilibrium; **every dynamic case is unstable or imperfect**. The whole debugging effort is closing that gap by isolation (Tier 0–3 benchmark ladder + targeted probes). Static lane is DONE and regression-locked; dynamic lane is open.

---

## 1. OPEN PROBLEMS, RANKED (priority order as of the last debugging_plan entry, 2026-06-02)

1. **Tier 2B step 1 — 2D integrated γ-flux curvature rewrite** (est. 4–8 h). The ONLY lever on the 2D static-droplet floor 2.2717e-3, which the plan attributes 100% to pointwise-curvature-stencil O(h) truncation. Plan action: route the dim=2 path of `_interface_surface_tension` (`ddgclib/operators/multiphase_stress.py`) through the exact PL operator `surface_tension_force_2d` (`ddgclib/operators/curvature_2d.py:91-181`); re-run `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py`; target 2.37e-3/2.27e-3 → machine precision; then **re-pin `TestStaticDroplet2DRetopologyFloor`** at the new floor. **⚠ VERIFIED STALE AS WRITTEN (2026-07-02): the dim=2 `'integrated'` path ALREADY calls `surface_tension_force_2d` (`multiphase_stress.py:262` and `:277`), yet the pinned floor is unchanged at 2.2717e-3 and the in-code comment (`multiphase_stress.py:216-219`, plan note 2026-05-06) attributes the residual to first-order convergence of the *polygonal geometry* vs a smooth circle — i.e. the routing landed but did NOT collapse the floor. Before spending the 4–8 h, check git history / re-run `diagnose_a5_bisection.py` to determine whether this item should be closed as "done, machine precision unattainable via routing alone" or whether a genuinely higher-order lever (e.g. `reconstruct_arc_length_and_bulge_area` constant-curvature reconstruction) is the intended remaining work.**
2. **Probe 5 — skip-retopology displacement gate** (est. 1–2 h). Skip `_retopologize` when max vertex displacement < eps (first cut eps = 1e-4·h_min; must scale ∝ h_local on dynamic runs). Site: `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize`. Would collapse the 3D A.5.b floor (7.3768e-05) to the frozen floor (6.0153e-05) by eliminating residual 3D Delaunay non-uniqueness churn. Both floor tests will instantly catch breakage. NOTE: `test_retopo_displacement_gate.py` (13 tests) already exists for a `displacement_eps` gate — verify what is implemented vs what the plan still lists as open before starting (the gate at `_integrators_dynamic.py:250-279` skips on first call by design).
3. **M4 / Tier 3A — the actual failing dynamic case: 2D oscillating droplet (Rayleigh–Lamb).** Current `results/score.json`: tail_growth **1.68** (target < 1.0), l2_error **≈ 5.6** (target < 0.2); mass drift machine precision. This is the ~37% of the 2D full-dynamic baseline (3.80e-3 vs 2.37e-3 static) attributable to dynamic coupling (motion → retopology → force noise) — **never probed**. Gated on #2 and/or #5. Related open reproducer: `static_droplet_2D.py` KE grows 0 → 4.4e-3 in 100 steps under full Delaunay retopo (while fixed-connectivity `cube2droplet/diagnostic_no_retopo.py` is stable 5000 steps). The p_ref benchmark's diagnosis: Delaunay-flip dual-volume jumps are read by the EOS as physical compression.
4. **A.3 single-phase probes (Tier 1A + 1B.i/1B.ii) — NEVER RUN.** Frozen-mesh transient decay (KE ~ e^{−2νk²t}); rigid-body advection with retopo on, μ=0 (1B.i `skip_triangulation=True` vs 1B.ii full Delaunay). Bisects integrator vs dual-volume-refresh vs Delaunay churn. **Explicitly gates all BC work (Tier 1E).** Related stalled S-lane finding: single-phase retopology leaks **~2–4% volume** with BOTH skip_triangulation and full Delaunay; root cause localized to the *dual-volume refresh*, not Delaunay churn (`benchmarks/dynamic/`); frozen meshes conserve to machine precision.
5. **A.4 upstream hyperct remesh fixes — blocks adaptive remesh entirely.** (a) `edge_split_2d` arithmetic-mean mass assignment inflates total mass **9.7 → 187 over 100 steps** (`hyperct/remesh/_operations_2d.py:198`); audit measured ~164% mass loss + KE explosion by step ~80 on the oscillating droplet under `remesh_mode='adaptive'`. Fix: length-weighted/conservative split + Σm-invariance unit test. (b) global `h_local` threshold triggers unbounded splits in coarse regions → make per-vertex/per-edge. Gate: full ddgclib fast suite passes after the hyperct change. The audit calls this "the gate on closing the dynamic-multiphase stability gap".
6. **Tier 3B — 3D oscillating droplet metric harness.** No `oscillation_score`/baseline exists for 3D (run completes, mass drift 1e-14, but R_max saturates at the mesh boundary ~5·R0 — physically suspect). Backlog item in DEVELOPMENT.md:1262-1266.
7. **Minor open checks:** does `hndA_i_interface[dim=2]` alias the 2D FTC form? (flagged "worth checking, not the blocker"); audit setup-time volume reporting never feeds IC mass ratios (currently safe); D2 BC pathologies (OutletDeleteBC backflow/KE destruction, PeriodicInletBC ghost-field discontinuities, PressureReservoirBC un-calibrated relaxation) — all un-quantified, gated behind #4.

### Latent code-level defects found by the code-map pass (unranked; verify before building on the affected paths)
- `MultiphaseEOS.__call__` (`ddgclib/eos/_multiphase_eos.py:81,86`) indexes `p_phase[v.phase]`/`eos_list[v.phase]` **without an interface guard** — interface vertices have `v.phase = -1`, so numpy negative indexing silently reads the LAST phase.
- `_phase_pressure`/`compute_phase_pressures` treat pressure **exactly 0.0 as "missing"** — wrong forces whenever gauge pressure crosses zero (P0=0 is common in cases).
- `edge_phase_area_fractions` hardcodes a **50/50 interface-face split even under `split_method='exact'`** (`ddgclib/geometry/_dual_split_2d.py:585-593`).
- `TaitMurnaghan` default `rho_clip=(0.9,1.1)` applies only in `pressure()` — silently flat-lines compressibility outside the band and is inconsistent with `density()`/`sound_speed()`.
- Empirical sign flip `e_ij = -e_ij` marked "WHY???" in `_curvatures_heron.py:241`, load-bearing for surface-tension direction (propagated to `hndA_i_interface:361`).
- 3D `dual_volume` silently skips edges raising KeyError/IndexError/ValueError → undercounted cell volume, no warning (`ddgclib/operators/stress.py:367-368`).
- rk45 evaluates RK-stage forces on duals/edge-area caches frozen at macro-step start (no mid-step dual rebuild).
- Momentum/action-reaction not guaranteed in `multiphase_stress_force` when a phase is absent on one side (fallback `p_j_k = p_i_k` breaks pairwise symmetry).
- `_vertex.py:787` (hyperct) NameError: `proc_minimisers` references undefined `v2` for a vertex with `f` but no neighbours.
- `cases_dynamic/cube2droplet/` scripts import `cases_dynamic.Cube2droplet` (capital C) → ModuleNotFoundError, verified.

## 2. DO-NOTs — lanes already closed (do not reopen)

- **Do NOT chase the 3D ×1.23 A.5.b/A.5.a gap with curvature-stencil variants.** Probe 2 closed the lane *mathematically*: on a PL surface, ∫_{Γ_i} 2H N dA is a vertex-supported distribution — identical for ANY dual cell containing v_i; the Stokes-form `integrated_hndA_i_interface` is bit-identical (max|diff| 1.14e-19) to the cotangent path.
- **Do NOT change the redistribute_mass guard again** — the 2026-04-29 fix (geometry-snapshot gate) is fully wired and A/B-confirmed (Probe 4).
- **Do NOT touch BC isolation (Tier 1E / OutletDeleteBC) until A.3 single-phase probes run.**
- **Do NOT enable `remesh_mode='adaptive'` in cases until the A.4 hyperct fixes land.**
- **Do NOT switch to Eulerian integrators** (`euler_velocity_only` is validation-only).
- **Do NOT delete the `'stokes'` curvature path in `multiphase_stress.py`** — deliberate regression-tested A/B equivalence probe, not dead code.

## 3. Bug/instability ledger (status of everything known)

### FIXED (with regression guards)
| Bug | Root cause | Fix | Evidence |
|---|---|---|---|
| **3D retopology blow-up** (A.5.b 1.44e-3, ×24 over frozen floor, from ONE retopo on an unmoved mesh) | 3D Delaunay non-uniqueness on near-cospherical interface cloud flips ~48 cross-phase edges → `dual_vol_phase` Δ~1e-8 on tiny off-side phase volumes → ρ=m/V swings ~700 kg/m³ → EOS p swings ~375 Pa → per-vertex \|F\| ×21–60; made fatal by `redistribute_mass_multiphase` guard `p_phase<1e-30` skipping the whole outer phase at P0=0 | `snapshot_geometry_multiphase` + guard gated on snapshotted `dual_vol_phase[k]>1e-30` (`ddgclib/operators/mass_redistribution.py:54-74,:308-310`); `redistribute_mass=True` now default in all multiphase setups (M1) | 3D A.5.b **1.44e-3 → 7.3768e-05 (×195)**; pinned by `TestStaticDroplet3DRetopologyFloor` |
| Setup/runtime `split_method` mismatch | setup `mps.refresh` + YL mass pre-load used a different dual-split than the runtime retopo partial | `setup_oscillating_droplet` takes `split_method`+`redistribute_mass`, threads both; docstring warns they MUST match | mismatch reproduces ρ ×147 (2D) / ×4.7 (3D) jumps |
| 3D boundary NaN dual volumes (95/472 vertices) | degenerate dual base triangle (duplicate dual vertex from boundary fan walk at box corners) → zero cross product → NaN | `if norm_sq == 0.0: return 0.0` in BOTH copies: `hyperct/ddg/_geometry.py:79-105` AND `hyperct/ddg/barycentric/_duals.py:328-365` | 3D conservation diagnostics now functional; `TestDegenerateGeometry` in hyperct tests |
| 3D sliver tets → ghost K_5 cliques → broken p_ij ring-walk | flag-complex `v.nn`-intersection apex enumeration on Delaunay meshes | simplex-aware duals via cached `HC._simplices` (`connect_and_cache_simplices`/`boundary_from_simplices`/`invalidate_simplex_cache`, commit 8321c71); 12/22 `vi.nn∩vj.nn` sites migrated | `test_simplex_aware_duals.py` (13), `test_simplex_aware_curvature.py` (16) |
| O(1) equilibrium residual (~0.34) of the full symmetric viscous stress | rank-1 face gradient has spurious discrete compressibility (nonzero trace on diagonal edges for div-free fields) → picks up μ∇(∇·u) discretely | transpose term dropped; diffusion form F_v=(μ/\|d\|)Δu(d̂·A) | Poiseuille equilibrium machine precision; old form archived in `stress_pointwise.py`/`Fundamentals_pointwise.md` |
| Spurious pressure jumps after retriangulation (single-phase) | dual volumes change under Delaunay while masses are frozen | pressure-preserving mass redistribution (`snapshot_pressure`, `redistribute_mass_single_phase`), 15 tests | `test_mass_redistribution.py` |
| Outlet backflow | truncated boundary dual cells (A_ij=0 on boundary edges) → imbalanced stress pushes vertices backward | `OutletBufferedDeleteBC` ghost buffer (stopgap; principled `prescribed_V` BCs fully spec'd, Not Started, DEVELOPMENT.md:403-548) | — |
| PeriodicInletBC ghost accumulation | ghosts re-injected every step → duplicate chains | ghosts removed after injection, re-cloned when depleted | — |
| Dimensional inconsistency of vertex-centered stress | — | face-centered Stokes-theorem formulation (no /Vol in force loop) | current `stress.py` |

### OPEN / SUSPECT
- **2D oscillating droplet dynamic validation FAILS** (l2 5.62, tail_growth 1.68) — see §1.3. The per-phase summed interface-stress rewrite ("Phase 6", DEVELOPMENT.md:5-70) improved metrics 20–31% (summary 4.29 vs baseline 5.38) but misses spec; full closure declared blocked on adaptive remesh (i.e., §1.5).
- **static_droplet_2D KE growth under full Delaunay retopo** (0 → 4.4e-3 / 100 steps) — open; suspects listed in `INVESTIGATION_PROMPT.md`: YL pre-load imbalance, retopo destroying interface, ST sign, phase-fraction error. Note the contradictory retopo guidance: dual-only retopo is stable for droplets but *less* stable for electrolysis_bubble with gravity.
- **Adaptive remesh unusable in production** (§1.5, upstream hyperct).
- **Single-phase retopology volume leak ~2–4%** (S-lane, stalled; dual-volume refresh, §1.4).
- **3D oscillating droplet unvalidated** (no metric harness/baseline; boundary saturation, §1.6).
- **Boundary p_ij dual-area precision ambiguous**: DEVELOPMENT.md marks "3D p_ij boundary edges O(h), blocked by compute_vd bug" (lines 1049,1061) while the referenced bug is marked Fixed (line 1065) — never revisited post-fix; unclear whether boundary precision is now resolved.
- **Corner/boundary spurious accelerations** (dam_break truncated dual cells; electrolysis_bubble 3D corner NaNs needing per-step sanitisation) — mitigated (alpha_art, short horizons), not fixed. No boundary-flux closure term exists in `stress.py`.
- **Body-force (gravity) term not in `stress_acceleration`** — hydrostatic equilibrium verified without an in-operator gravity term; cases add gravity in their own dudt wrappers.
- **DEM liquid bridge xfail** — bridge-formation timing tuning (`test_dem_liquid_bridge.py`).
- **Periodic 2D**: hyperct `compute_vd` gives wrong dual positions near periodic faces; worked around in `stress.py:97-156` (min-image dual rebuild) but `d_ij` in `stress_force` is NOT min-imaged; upstream unfixed.
- **hyperct batch vs sequential 3D dual connectivity differ** (`_compute_vd_3d_batch` omits `vd_mid.connect(vd_face)`); `edge_collapse_2d` returns True even when the final move aborts.

## 4. Hypothesis / experiment history (what was tried, what it showed)

The A.5 bisection harness (`diagnose_a5_bisection.py`, static Young–Laplace droplet, max|F| on interface vertices): **A.5.a** = frozen mesh (pure stencil residual); **A.5.b** = retopo on, u zeroed (pure retopology residual).

| Baselines | 2D | 3D |
|---|---|---|
| Full-dynamic | 3.80e-3 | 8.50e-5 |
| A.5.a frozen floor | **2.3748568e-3** | **6.0153e-05** |
| A.5.b pre-fix | 2.37e-3 (benign) | **1.44e-3** (the blow-up) |
| A.5.b post-fix floor | **2.2716938e-3** | **7.3768e-05** |
| 2000-step stability | bit-identical steps 1–2000 | bit-identical steps 2–2000 |

Meshes: 2D 311 verts/32 interface (refine 3/3); 3D 472/98 (refine 2/2). KE plateaus 3.7e-11 / 2.4e-10; mass drift ≤ 8.8e-15.

**Hypotheses REFUTED (do not re-test):**
- Flat-interface per-phase pressure-cancellation bug (D4) — A.2: machine precision all 4 variants (γ=0 and γ>0/κ=0, 2D+3D, ≤1.05e-15).
- `split_method='exact'` fixes the 3D blow-up — Phase 1: end-to-end 1.75e-3, *worse* than neighbour_count's 1.44e-3 (2D: exactly retopo-neutral). M2 (flip default to 'exact') effectively dead.
- `redistribute_mass=True` as shipped fixes it — Phase 2b: no-op; the guard itself was the bug (fixed in Phase 2c, see ledger).
- Apex-enumeration ghost cliques drive the residual — Probe 1: bit-identical A.5 numbers before/after the simplex-aware migration (hygiene only).
- ANY integrated 3D curvature-stencil variant reduces the residual — Probe 2: mathematically closed (bit-identical; dual-cell-independence of the integrated mean-curvature normal on PL surfaces).

**Mechanism CONFIRMED (Phase 2, single-retopo state diff `diagnose_a5_step1_diff.py`):** vertex set and `m_phase` bit-preserved; first-to-diverge chain `dual_vol_phase` (Δmax 1.03e-8) → `rho_phase` (Δmax 7.10e+02 kg/m³) → `p_phase` (Δmax 3.75e+02 Pa) → per-vertex |F| ×21–60; 48 cross-phase edges flip. Root cause: 3D Delaunay non-uniqueness on near-cospherical clouds (2D Delaunay unique ⇒ benign).

**Residual attribution (final):** 2D floor = 100% pointwise curvature-stencil O(h) truncation (lever: §1.1). 3D frozen floor = irreducible O(h²) polygon-vs-sphere truncation. 3D ×1.23 retopo gap = residual Delaunay churn (lever: §1.2). Remaining ~37% of 2D dynamic baseline = unprobed dynamic coupling (lever: §1.3).

**Benchmark ladder status (Tier 0–3):** Tier 0 conservation diagnostics DONE (A.1, 18 tests). Tier 1 (single-phase dynamic 1A–1E) ALL PENDING (= §1.4). Tier 2A flat interface PASSED machine precision; 2B static droplet floors pinned (further progress = §1.1 + §1.2); 2C sessile droplet not started. Tier 3A failing (§1.3); 3B needs harness (§1.6); 3C dam break regression-only.

Test-suite trajectory across the campaign: 761 → 794 fast passed, 0 failures (2026-06-02); 802 full (2026-05-27); audit 2026-06-08: 808 fast passed.

## 5. Footguns, stale claims, contradictions (verify before trusting any doc)

1. **Harness flag default:** `diagnose_a5_bisection.py` without `--redistribute-mass` overrides the production True default → reproduces pre-fix 1.44e-3. Expected.
2. **split_method must match** between setup and runtime for ANY new multiphase setup (enforced only in `setup_oscillating_droplet`).
3. **3D |dV/V0| = 0.305 one-shot at step 0→1** = boundary dual-shell zeroing artefact (boundary verts get `dual_vol=0.0` at first retopo, `_integrators_dynamic.py:211/:225`). Assert volume from step ≥ 2 only. Corollary: `DualVolumeMass` applied after an integrator retopo gives boundary verts m≈0; mass-relaxation BCs silently skip them.
4. **All debugging_plan fixes were left working-tree only** at the time of writing — verify presence/commits before relying on quoted line numbers.
5. **Test counts are stale everywhere**: CLAUDE.md ~415, ARCHITECTURE.md ~604 (self-contradicts to ~415), Fundamentals 488, DEVELOPMENT.md snapshots 279–733; audit measured 808. Re-run pytest.
6. **hyperct symlink**: actual chain `ddgclib/hyperct → /home/endres/projects/bilevel_param/hyperct → /home/endres/projects/hyperct/hyperct`; every doc states something else.
7. **Duelling docs**: both Fundamentals.md and Fundamentals_v1.md claim "single source of truth" (Fundamentals.md wins); FEATURES.md puts completed items under "Planned" (adaptive remesh Phase 1–2, 2D integrated curvature, exact dual split) and overstates oscillating-droplet validation; internal ARCHITECTURE.md module graph omits multiphase/eos/surface_tension/curvature_2d/periodic.
8. **Stale docstrings that contradict working code**: `test_multiphase_stress_per_phase.py` module doc claims harmonic-mean cross-phase viscosity but implementation asserts exact per-phase μ_k (`multiphase_stress.py:94-104`) — while DEVELOPMENT.md still lists harmonic mean as the design; `multiphase_stress.py:21-23` promises harmonic mean too; `dudt_i` docstring (stress.py:838-840) and `_integrators_dynamic.py:24-36` examples show the broken kwargs-forwarding pattern; `multiphase.py:536-537` (interface v.p claimed inner-phase, actually averaged) and `:353-356` (closure-failure warning claimed, exception silently swallowed at :379-382).
9. **3D flat/aligned interface fixtures** must use an explicit Kuhn-decomposed cube; default `Complex(3).triangulate()+refine_all()` does not align z=0.
10. **Two copies of `volume_of_geometric_object`** (hyperct `_geometry.py` canonical + `barycentric/_duals.py` legacy) — geometry fixes must be applied to both.
11. **Any topology mutation outside `connect_and_cache_simplices` must call `invalidate_simplex_cache(HC)`** (clears `_simplices` AND `_edge_to_apex`); `HC._edge_area_cache` is `id()`-keyed → stale after vertex replacement.
12. **`scalar_gradient_integrated`** (stress.py:486) is not exported by `operators/__init__.py` — import from `ddgclib.operators.stress`.
13. Superseded number: old static-droplet-2D target 7.41e-3 → live `summary = 1.20e-3`.
14. hyperct pytest "errors" (9–39) = missing `pytest-benchmark` fixture, environmental noise.

## 6. Key artifacts for the next debugging session

| Artifact | Role |
|---|---|
| `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` | main static harness (floors) |
| `cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py` | single-retopo per-vertex/per-phase state diff |
| `cases_dynamic/oscillating_droplet/static_droplet_2D.py` | fastest failing reproducer (100 steps) |
| `cases_dynamic/oscillating_droplet/results_a5_bisection/*.json` | raw probe data |
| `ddgclib/tests/test_case_oscillating_droplet.py` | pinned 2D (fast) + 3D (slow) retopology floors |
| `ddgclib/tests/test_a5b_longrun_regression.py` | long-run floor guards, 1% trip wire |
| `ddgclib/operators/mass_redistribution.py` | the Phase-2c fix lives here |
| `ddgclib/dynamic_integrators/_integrators_dynamic.py` | `_retopologize` (:48-247), `_retopologize_multiphase` (:391-475, verified 2026-07-02; debugging_plan's :325-406 is stale), displacement gate (:250-279) |
| `ddgclib/operators/curvature_2d.py:91-181` | `surface_tension_force_2d` — target of open problem #1 |
| `hyperct/remesh/_operations_2d.py:198` | open upstream mass bug — open problem #5 |
| `docs/{3d_multiphase_interface_pressure_fix.md, 3d_simplex_aware_dual_fix.md, interface_stress_rewrite.md}` | background fix write-ups |
| `debugging_plan.md`, `DEVELOPMENT.md`, `LIBRARY_AUDIT.md` | underlying trackers (trust the debugging_plan status table over per-entry TODO lists) |
