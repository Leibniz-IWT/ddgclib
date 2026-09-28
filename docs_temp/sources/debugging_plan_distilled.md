# ddgclib Debugging Plan (Stabilisation Roadmap) — Distilled
> Sources: /home/endres/projects/ddgclib/debugging_plan.md (587 lines; status-log entries 2026-04-24 → 2026-06-02, reverse-chronological) | Written: 2026-07-02 by understand-and-document workflow

Original title: "Stabilisation Roadmap for `ddgclib` — From Equilibrium Benchmarks to Stable Dynamic Multiphase".

---

## 0. Problem statement

The **spatial discretisation** (integrated FVM Cauchy-stress operator on hyperct's barycentric DDG dual mesh) is validated to machine precision on equilibrium benchmarks (linear fields < 1e-13; hydrostatic; quadratic viscous flux; combined Poiseuille — refs: `benchmarks/_integrated_benchmark_cases.py:43-123`, `:618-664`, `:667-756`, `:709-756`; `benchmarks/run_integrated_benchmarks.py:210-260`). Despite this, **every dynamic case in `cases_dynamic/` is unstable or imperfect**: energy drifts, vertex counts grow, retopology corrupts fields, outlet BCs cause backflow, multiphase cases (oscillating droplet, dam break) drift or blow up. **The gap between machine-precision equilibrium and dynamic stability is the problem to close.** Strategy: a hierarchy of isolation benchmarks (Tier 0–3, Section 4) that add one concern per rung, plus targeted probes (Section 5) to attribute each failure mode.

Primary quantified failing case: **2D oscillating droplet (Rayleigh–Lamb, Tier 3A)** — `tail_growth = 1.68` (target < 1.0), `l2_error ≈ 5.6` (target < 0.2), from `cases_dynamic/oscillating_droplet/results/score.json`. The static droplet 2D live score is `summary = 1.20e-3`, `mass_drift = 1.8e-16`, `max_KE_normalized = 1.02e-8` (the plan's old 7.41e-3 target for it is **stale**).

---

## 1. Master status table (all probes / hypotheses / fixes)

| Item | Status | Result / key number |
|---|---|---|
| A.1 conservation diagnostics (`compute_conservation`, `StateHistory(conservation=True)`) | ✅ done 2026-04-24, re-validated 2026-04-26 | 18/18 tests, `ddgclib/data/_conservation.py`, `ddgclib/tests/test_conservation.py` |
| A.2 flat-interface pytest (γ=0 and γ>0/κ=0, 2D+3D) | ✅ done 2026-04-24 | Machine precision all 4 variants (2.22e-16 2D; ~1.05e-15 3D) → **D4 per-phase cancellation bug REFUTED** |
| A.3 single-phase probes (Tier 1A frozen-mesh + 1B.i/1B.ii skip_triangulation vs full Delaunay) | ⬜ **NEVER RUN — still pending** | Gates Tier 1E BC work and lane-S fixes |
| A.4 upstream hyperct fixes (edge_split_2d mass averaging; h_local global threshold) | 🟨 partial / **pending** | Simplex-container refactor (2026-04-26) independently fixed the 3D linear-precision concern; the two remesh bugs remain → adaptive remesh still blocked |
| A.5 2D static-droplet bisection (curvature vs retopology) | ✅ done, confirmed | 2D residual = **100% curvature stencil, 0% retopology** (A.5.b `'exact'` == A.5.a exactly, 2.37e-3) |
| A.5 3D Phase 1 — hypothesis `split_method='exact'` fixes 3D blow-up | ✅ done 2026-04-29 — **REFUTED** | `'exact'` end-to-end gives 1.75e-3, slightly WORSE than `neighbour_count` 1.44e-3 |
| A.5 3D Phase 2 — single-retopo-step state diff | ✅ done 2026-04-29 — **mechanism CONFIRMED** | First-to-diverge: `dual_vol_phase` (Δ 1.03e-8) → `rho_phase` (Δ 7.10e+02 kg/m³) → `p_phase` (Δ 3.75e+02 Pa); `m_phase` bit-preserved; 48 cross-phase edges flip per static-cloud retopo (3D Delaunay non-uniqueness on near-cospherical cloud) |
| A.5 3D Phase 2b — hypothesis `redistribute_mass=True` fixes it | ✅ done 2026-04-29 — **REFUTED as-is** | No-op (identical 1.44e-3): guard at `mass_redistribution.py:250` (`p_phase < 1e-30`) skips the whole outer phase at P0=0 |
| A.5 3D Phase 2c — guard fix (`dual_vol_phase`-gated) | ✅ done 2026-04-29 — **POSITIVE, the key fix** | 3D A.5.b max\|F\|: **1.44e-3 → 7.3768e-05** (×195), ×1.23 over A.5.a floor 6.0153e-05 |
| Setup/runtime `split_method` mismatch bug | ✅ found + fixed 2026-04-29 | Mismatch → rho jumps ×147 (2D) / ×4.7 (3D); catastrophic A.5.b 3.50e-1 (2D) / 6.73e-3 (3D). `setup_oscillating_droplet` now takes `split_method` + `redistribute_mass` and threads both through setup refresh and runtime retopo_fn |
| M1 rollout — `redistribute_mass=True` default in all multiphase setups | ✅ done 2026-04-29 | dam_break / cube_to_droplet / shearing_plate / electrolysis_bubble; per-phase mass drift ≤ 1.94e-15 in 5-step smokes |
| Simplex-aware curvature apex enumeration (kill ghost K_{dim+1} cliques) | ✅ done 2026-05-27 | 12 of 22 audited `vi.nn ∩ vj.nn` sites migrated; 11 new tests; 802 full-suite pass |
| Probe 1 — did the apex migration move the A.5 numbers? | ✅ done 2026-05-27 — **non-driver** | Bit-identical to 2026-04-29 baseline (2D 2.3749e-3, 3D 7.3768e-05); wiring verified (`HC._interface_edge_to_apex` = 1152 edges × 2 apexes) |
| Probe 2 — integrated Stokes-form 3D curvature (`integrated_hndA_i_interface`) reduces residual? | ✅ done 2026-05-27 — **REFUTED mathematically** | Bit-identical to cotangent path (max\|diff\| 1.14e-19 over 98 vertices; 2.62e-16 on irregular fixture). Integrated mean-curvature normal is dual-cell-independent on PL surfaces → **no integrated stencil variant can reduce the 3D residual** |
| Probe 3 — `_compute_vd_3d` boundary NaN (`hyperct/ddg/_geometry.py:91`) | ✅ fixed 2026-05-27 | 746/30272 calls hit degenerate base (`points[0]==points[2]`) → 95/472 vertices NaN dual_vol. One-line `if norm_sq == 0.0: return 0.0` in 2 places. 3D conservation diagnostics now functional (were NaN) |
| Probe 4 — redistribute_mass guard fix "still pending?" | ✅ A/B-confirmed 2026-05-27: **already shipped 2026-04-29** | Without flag: 1.4415e-3 (legacy guard path); with: 7.3768e-05. Doc self-contradiction resolved |
| Tier 2B 2D long-run regression | ✅ done 2026-05-28 | A.5.b 2D bit-stable over 2000 steps (std 4.34e-19, ptp 0.0); `TestStaticDroplet2DRetopologyFloor` pins 2.3748568e-3 (frozen) / 2.2716938e-3 (post-retopo) |
| Probe 6 — A.5.b 3D long-run regression | ✅ done 2026-06-02 | Bit-stable steps 2–2000 (1 unique value, rel spread 0.0); `TestStaticDroplet3DRetopologyFloor` pins 6.0153e-05 / 7.3768e-05 |
| **Tier 2B step 1 (2D) — integrated γ-flux curvature rewrite** | ⬜ **OPEN — top open lever** | Only lever on the 2D 2.27e-3 O(h) floor; est. 4–8 h; use existing `surface_tension_force_2d` (`ddgclib/operators/curvature_2d.py:91-181`) |
| **Probe 5 — skip-retopology gate (`max-vertex-displacement < eps`)** | ⬜ **OPEN** | Would collapse 3D A.5.b (7.38e-05) to A.5.a (6.02e-05); est. 1–2 h; first cut `eps = 1e-4 * h_min`; site: `_integrators_dynamic.py:_retopologize`. Now safe (both floors regression-locked) |
| **M4 / Tier 3A — dynamic 2D oscillating droplet (Rayleigh–Lamb)** | ⬜ **OPEN — the actual failing dynamic case** | tail_growth 1.68 → <1.0; l2 ≈5.6 → <0.2; = the ~37% of 2D baseline that static probes do not capture; gated on Probe 5 and/or adaptive remesh |
| Tier 1E — BC isolation (`OutletDeleteBC` etc.) | ⬜ blocked | Explicitly gated behind A.3 |
| Adaptive remesh (`remesh_mode='adaptive'`) in dynamic cases | ⬜ blocked | Gated on A.4 upstream hyperct fixes |
| M2 — flip `split_method` default to `'exact'` | effectively dead | Conditional on Phase A showing improvement; 3D showed `'exact'` slightly worse, 2D neutral |
| Tier 3B — 3D oscillating droplet | ⬜ not started | Needs a 3D metric harness first (DEVELOPMENT.md:59-68) |
| D1 sub-item "3D fallback loses linear precision" | ✅ resolved 2026-04-26 | Core multiphase refactor (commit 8321c71): `connect_and_cache_simplices` / `boundary_from_simplices` / `invalidate_simplex_cache` now canonical; guarded by `test_simplex_aware_duals.py` (13 tests) |

Test-suite trajectory (fast suite, `pytest ddgclib/tests/ -m "not slow"`): 761 → 765 → 778 → 780 → 789 → **794 passed, 0 failures** (2026-06-02). Full suite incl. slow: 802 passed (2026-05-27). hyperct: 275 passed (39 apparent errors are missing `pytest-benchmark` fixture only — pre-existing, environmental).

---

## 2. Quantitative baselines and floors (static Young–Laplace droplet, max |F| on interface vertices)

Harness: `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` (n=100 or 2000 steps), setup `setup_oscillating_droplet(epsilon=0)`.
- **A.5.a** = retopology disabled (`retopologize_fn=False`), frozen mesh → pure curvature-stencil-vs-pressure-jump residual.
- **A.5.b** = retopology enabled, u forced to 0 each step → pure retopology-induced residual.

| quantity | 2D | 3D |
|---|---|---|
| Full-dynamic baseline | 3.80e-3 | 8.50e-5 |
| A.5.a frozen floor | **2.3748568012e-03** (×0.625 of baseline) | **6.0153e-05** (×0.708) |
| A.5.b pre-fix (retopo blow-up) | 2.37e-3 (benign) | **1.44e-3** (×24 over A.5.a, ×17 over baseline — the "3D retopology blow-up") |
| A.5.b post-Phase-2c floor | **2.2716937802e-03** (settles at step 1; −4.4% one-shot) | **7.3768e-05** (settles at step 2; ×1.23 over A.5.a) |
| Long-run stability (2000 steps) | bit-identical steps 1–2000 (std 4.34e-19, slope 5.93e-19/step) | bit-identical steps 2–2000 (1 unique value, rel spread 0.0) |
| KE plateau (impulse residual, non-growing) | 3.688e-11 | 2.371e-10 |
| \|dM/M0\| | 1.643e-15 | 8.78e-15 |
| \|dV/V0\| | 3.542e-16 | 3.055e-01 one-shot step 0→1 (boundary-shell zeroing artefact, NOT drift); rel spread 0.0 from step 2 |
| Mesh | 311 verts / 32 interface (refinement_outer=3, refinement_droplet=3) | 472 verts / 98 interface (refine 2/2) |
| Wall time / 2000 steps | 125.7 s | 686.9 s |

**Residual attribution (final):**
- 2D floor 2.27e-3 = 100% pointwise curvature-stencil truncation — `hndA_i_interface` has O(h) error on a polygon approximation of a circle. Retopology exactly neutral (2D Delaunay of a static cloud is unique/idempotent).
- 3D floor 6.02e-05 (A.5.a) = irreducible O(h²) polygon-vs-smooth-sphere truncation; per-vertex F_st error vs analytical F_st = −(2γ/R)·A·N. **No integrated-stencil variant can reduce it** (Probe 2, mathematically closed: Δ_S X is a vertex-supported distribution; the integrated ∫2H N dA over any dual cell containing v_i is dual-partition-independent).
- 3D ×1.23 A.5.b/A.5.a gap = residual 3D Delaunay non-uniqueness churn (~48 cross-phase edge flips per retopo on the near-cospherical interface cloud). Only lever: Probe 5 skip-retopo gate.
- The remaining ~37% of the 2D full-dynamic baseline (3.80e-3 vs 2.37e-3 static) = dynamic coupling (motion → retopology → force noise); target of M4, not yet probed.

---

## 3. Failure-entry diagnosis (D1–D5)

Since spatial operators are exact at equilibrium, instability must enter via:

- **D1 — Retopologization every step injects energy/noise.** `_retopologize` at `ddgclib/dynamic_integrators/_integrators_dynamic.py:48-237` runs at the top of every step. (a) Delaunay redraw changes `Vol_i` even when vertices barely move; without `redistribute_mass=True` + EOS, `v.p` is stale masses over new volumes → spurious pressure gradients (`mass_redistribution.py:82-150`); most dynamic cases historically did NOT enable it (now fixed for multiphase setups via M1). (b) ~~3D fallback loses linear precision~~ — RESOLVED 2026-04-26. (c) Boundary misclassification: vertices with `dual_vol < 1e-30` auto-marked boundary and frozen (`_integrators_dynamic.py:211`; also cited as `:225` — boundary `dual_vol` zeroing) — can freeze interior vertices during tangle. (d) Adaptive remesh is 2D-only and blocked by upstream hyperct bugs: `edge_split_2d` arithmetic-mean mass inflates total mass **9.7 → 187 over 100 steps** in the static-droplet stress test (`hyperct/remesh/_operations_2d.py:198`); global `h_local` threshold triggers unbounded splits in coarse regions (DEVELOPMENT.md:69-70).
- **D2 — Boundary conditions destroy energy conservation** (`ddgclib/_boundary_conditions.py`). `OutletDeleteBC` (:395-440): truncated dual cells at outlet → imbalanced stress forces pushing vertices backward; `backflow_clamp` destroys KE non-physically. `OutletBufferedDeleteBC` (:443-522): freezes velocity in ghost buffer, overwriting stress update. `PeriodicInletBC` (:532-680): injected ghost vertices carry independently-evolved fields → discontinuities; merge tolerance is an unprincipled knob. `PressureReservoirBC` relaxation timescale un-calibrated. **All un-quantified — gated behind A.3/Tier 1E.**
- **D3 — No built-in conservation diagnostics** — RESOLVED by A.1 (Tier 0): `StateHistory` + `compute_conservation` now record KE, momentum, mass/volume (per phase), h_min/h_max, extrema per snapshot.
- **D4 — Multiphase interface-vertex residual force** — NARROWED by A.2: the per-phase summed pressure-flux cancellation and planar `hndA_i_interface` are both exact (machine precision on flat interfaces). Residual on curved interfaces comes from curvature stencil (dominant) + retopology (3D pre-fix) — fully attributed by A.5 (Section 2). Ref: `docs/3d_multiphase_interface_pressure_fix.md:147-177`.
- **D5 — Interface fragility during remesh.** Default `split_method='neighbour_count'` (`ddgclib/multiphase.py:417-503`) is a 1-ring majority vote; `'exact'` geometric split exists in `ddgclib/geometry/_dual_split_2d.py` (2D + 3D PCA-plane clipping) but is not default (and per Phase 1 is NOT an improvement in 3D). Global Delaunay creates cross-phase edges; `assign_simplex_phases_from_vertices` (`multiphase.py:237-287`) re-labels by majority vote (not exact recovery); `strict_closure=False` at runtime masks conformity violations. Interface-preserving adaptive remesh is the right tool but blocked by D1(d).

---

## 4. Benchmark ladder (hierarchical isolation; each rung = one added concern, pytest + Tier-0 harness)

- **Tier 0** — conservation diagnostics. ✅ DONE (A.1). Records per step: KE = 0.5 Σ m_i|u_i|² (total + per-phase), Σ m_i u_i, mass/volume totals + per-phase, min/max |u| and v.p, vertex count, h_min/h_max, step max |F|.
- **Tier 1 (single-phase dynamic)** — ALL PENDING except as noted:
  - 1A frozen-mesh transient decay (no retopo, damped sinusoid in periodic box; expect KE ~ e^{−2νk²t} to O(dt²), machine mass/volume conservation). Failure ⇒ time-integrator bug.
  - 1B rigid-body advection, retopo on, zero stress (uniform u, mu=0, no pressure). **1B.i** `skip_triangulation=True` (`_integrators_dynamic.py:188` — duals recomputed, connectivity kept) vs **1B.ii** full Delaunay. i-pass/ii-fail ⇒ connectivity churn; i-fail ⇒ dual-volume refresh / mass bookkeeping.
  - 1C transient Poiseuille from rest (existing `Hagen_Poiseuile` setup): O(h²) centerline convergence, monotone KE rise, mass drift < 1e-12.
  - 1D small-perturbation hydrostatic: KE must damp at rate νk².
  - 1E `PeriodicInletBC + OutletDeleteBC` vs native periodic — quantifies D2. Gated behind A.3.
- **Tier 2 (multiphase static)**:
  - 2A flat interface (γ=0; γ>0/κ=0). ✅ PASSED at machine precision (2026-04-24), both dims. 3D prerequisite discovered: default `Complex(3).triangulate()+refine_all()` does NOT give a planar z=0 interface (64/817 tets cross z=0); tests must build an explicit **Kuhn-decomposed cube** (6 tets/unit cube, axis-monotone diagonals) — required for any mesh-aligned 3D flat/spherical reference benchmark.
  - 2B static Young–Laplace droplet. ✅ Static floors reached and regression-locked (Section 2); further progress = Tier 2B step 1 (2D integrated curvature rewrite) + Probe 5.
  - 2C static sessile droplet (contact line). Not started; after 2B.
- **Tier 3 (multiphase dynamic)**:
  - 3A 2D oscillating droplet (Rayleigh–Lamb). ❌ FAILING: tail_growth 1.68 (target <1.0), l2_error ≈5.6 (target <0.2). Cannot pass without 2B at machine precision (per plan; static floors are now pinned but NOT at machine precision in 2D).
  - 3B 3D oscillating droplet — needs 3D metric harness (missing, DEVELOPMENT.md:59-68).
  - 3C dam break — regression-only, no analytical reference.
- **Gate for "stable solver" claim**: every rung 1A–3C passes checked-in tolerance; dashboard `benchmarks/dynamic/DASHBOARD.md` (planned) shows no regressions.

---

## 5. Experiment log by theme (what was run, what it showed)

### 5.1 A.5 bisection and the 3D retopology blow-up (2026-04-24 → 04-29)
1. **2026-04-24 initial bisection** (`diagnose_a5_bisection.py`, n=100): 2D A.5.a ≈ A.5.b (2.37e-3; retopo even improves it to 2.27e-3) ⇒ 2D = curvature stencil. 3D A.5.a 6.02e-5 but **A.5.b jumps to 1.44e-3 after exactly one retopo call on an unmoved mesh** (flat from step 1). Conservation totals NaN in 3D (95 boundary vertices with NaN dual_vol — separate bug).
2. **2026-04-28 post-refactor re-run**: bit-identical pre/post refactor (verify probe confirmed simplex-aware path exercised: `HC._simplices = 2515`). Localised the |F| jump: NaN boundary duals are orthogonal (retopo actually cleans them to 0, yet |F| jumps ×24 across the same call) ⇒ cause is in code mutating interface state during retopo, suspected `mps.refresh(..., split_method='neighbour_count')` in `_retopologize_multiphase` (`_integrators_dynamic.py:325-406`).
3. **2026-04-29 Phase 1** — `--split-method exact`: 2D exact = perfectly retopo-neutral (A.5.b == A.5.a exactly); 3D exact = **1.75e-3, worse**. Hypothesis "majority-vote split is the 3D bug" **killed**. Side discovery: setup/runtime split-method mismatch is its own bug — setup `mps.refresh(reset_mass=True)` (`_setup.py:144`) and YL mass adjustment (`_setup.py:169`: `v.m_phase[1] = rho_d_eq * vol_d`) MUST use the same split_method as the runtime retopo partial, else rho = m/V jumps ×147 (2D) / ×4.7 (3D) on first retopo (catastrophic 3.50e-1 / 6.73e-3 runs). Fixed: `setup_oscillating_droplet` (`_setup.py:24`) takes `split_method` + `redistribute_mass`, threads both; docstring warns they MUST match.
4. **2026-04-29 Phase 2** — new diagnostic `diagnose_a5_step1_diff.py` (single retopo call, per-vertex per-phase diff, 98 interface vertices, neighbour_count): vertex set unchanged (472→472, no phase/is_interface flips); `m_phase` bit-preserved (Δ=0.0, Lagrangian mass exact); `dual_vol_phase` Δmax 1.03e-8 / mean 5.02e-9 / median 6.32e-17; **`rho_phase` Δmax 7.10e+02 kg/m³ (70% of rho_o); `p_phase` Δmax 3.75e+02 Pa; per-vertex |F| ×21–×60 (e.g. 2.30e-5→4.88e-4, 2.48e-5→1.49e-3); 48 cross-phase edges added + 48 removed** in the interface 1-ring. Arithmetic mechanism: `m_phase[k] = rho0_k · dual_vol_phase[k]_setup`, so tiny Δvol on a small off-side phase volume (~1e-9–1e-8) is O(1) relative → rho = m/V off by 10× → EOS pressure swings hundreds of Pa → interface stress ×20–60. **Root cause: 3D Delaunay non-uniqueness on near-cospherical clouds** (interface vertices were placed ON a sphere): re-running Delaunay on identical coordinates can return a different tessellation. 2D Delaunay is unique ⇒ benign.
5. **2026-04-29 Phase 2b** — `--redistribute-mass`: no-op (identical 1.44e-3). Cause: guard at `mass_redistribution.py:250` skips any (vertex, phase) with snapshotted `p_phase[k] < 1e-30`; case uses P0=0 ⇒ entire outer phase skipped — exactly the phase whose rho collapses 1000→~91 kg/m³. Guard conflates "phase absent at v" with "phase at reference pressure".
6. **2026-04-29 Phase 2c (THE FIX)** — new `snapshot_geometry_multiphase` capturing pre-retopo `dual_vol_phase`; `redistribute_mass_multiphase` gates phase presence on snapshotted `dual_vol_phase[k] > 1e-30` (legacy pressure snapshot still accepted). `_retopologize_multiphase` uses the geometry-aware snapshot. Also fixed a harness bug (3D `run_a5b` not forwarding `--redistribute-mass`). Production: `setup_oscillating_droplet` defaults `redistribute_mass=True`. **Result: 3D A.5.b 1.44e-3 → 7.3768e-05 (×195).** Live code: `mass_redistribution.py:54-74`, `:308-310`; `_integrators_dynamic.py:374-379`; `_setup.py:42`.

### 5.2 Curvature-stencil lane closure (2026-05-27, three entries)
- **Apex-enumeration migration**: 12/22 audited `vi.nn.intersection(vj.nn)` sites moved to explicit top-dim simplex cache, removing K_{dim+1} ghost-clique apex contamination on Delaunay meshes. New hyperct API: `get_edge_apex_map(HC)` (lazy `HC._edge_to_apex`: `dict[frozenset[id(vi),id(vj)], list[apex]]`); `invalidate_simplex_cache` also clears `_edge_to_apex`. ddgclib: `_apex_via_simplex_cache` (triangle caches only; tet caches → fallback) and `_apex_via_interface_triangles` (reads `HC.interface_triangles`, caches `HC._interface_edge_to_apex`) at top of `_curvatures_heron.py`; `A_i`, `hndA_i`, `int_hndA_i`, `hndA_i_interface` (+ torch-vectorized) gained optional `HC=None` kwarg with verbatim legacy fallback; callers threaded (`multiphase_stress._interface_surface_tension`, `surface_tension.*`, 4 sites in `_bubble.py` via `_common_neigh`, `interface_nn` prefers `HC.interface_edges`, periodic `_fixup_periodic_duals`). Plotting deliberately NOT migrated (operates on dual graph, exact by construction). 11 tests in `test_simplex_aware_curvature.py`.
- **Probe 1** (does the migration move A.5?): **No** — bit-identical both dims. Apex enumeration was a non-driver on this fixture; defensive hygiene only.
- **Probe 2** (integrated Stokes-form curvature): new `integrated_hndA_i_interface(v, interface_set, HC, gamma)` in `_curvatures_heron.py` — integrates in-surface conormal ν over dual-boundary segments midpoint→centroid→midpoint per interface triangle; wired as `curvature_path='stokes'` in `_interface_surface_tension` (3D only; default unchanged; CLI flag on the harness). **Bit-identical to the cotangent/'integrated' path** (max|diff| 1.14e-19 on droplet; 2.62e-16 on a deliberately irregular non-coplanar fixture). Math: on a PL surface, ∫_{Γ_i} 2H N dA is a vertex-supported distribution — identical for ANY dual cell containing v_i; dual-cell choice only affects averaged κ = (∫HN)/A_dual, not the integrated force. **Closes the entire 3D integrated-stencil lane.** 5 tests (`TestIntegratedHndAIInterface`): flat Kuhn interface F_st = 7.36e-18 via genuine conormal cancellation; spherical interface inward + sign-consistent 26/26; closed-sphere force sum 2.7e-19. Open sub-question flagged: whether the 2D FTC form and `hndA_i_interface[dim=2]` alias — "worth checking but not the active blocker".

### 5.3 Probe 3 — 3D boundary NaN fix (2026-05-27)
Traceback: `cache_dual_volumes` → `dual_volume(v)` → `_v_star(v, v_j, HC, dim=3)` → `volume_of_geometric_object(verts, v_i.x_a)`. 746/30272 calls (2.5%) had a degenerate base triangle (`points[0] == points[2]`, duplicate dual vertex from the boundary fan walk at outer-box corners) → zero cross product → NaN division → 95/472 vertices NaN dual_vol (matches memory `project_3d_boundary_dual_bug.md`). Fix: `if norm_sq == 0.0: return 0.0` in BOTH copies — `hyperct/ddg/_geometry.py:79-105` (canonical) and `hyperct/ddg/barycentric/_duals.py:328-365` (legacy shim). 3 regression tests (`TestDegenerateGeometry` in `hyperct/tests/test_ddg.py`): collinear base, duplicated-base-point, unit corner tet = 1/6. Post-fix: 3D mass_total 0.91035, volume_total 9.11e-04 (were NaN); max|F| bit-identical (NaN was masked from |F| because retopo zeroes boundary dual_vol anyway). **Surfaced artefact**: |dV/V0| = 0.305 at step 0→1 — boundary vertices now carry real "outer shell" volume at setup which `_retopologize` zeroes on first call (`_integrators_dynamic.py:211`) — correct semantics, NOT a drift; volume stable thereafter. Flagged audit: fine as long as setup-time volume reporting never feeds IC mass-to-volume ratios (currently it does not — setup only touches inner-phase mass where `v.dual_vol_phase[1] > 1e-30`).

### 5.4 Long-run floors + regression guards (2026-05-28, 2026-06-02)
- 2D 2000-step: single unique max|F| value steps 1–2000 (std 4.34e-19 = roundoff floor; linear slope 5.93e-19/step). Guard: `TestStaticDroplet2DRetopologyFloor` (`ddgclib/tests/test_case_oscillating_droplet.py`, ~125 lines, 30 A.5.b steps, ~2.6 s): step-0 within 1% of 2.3748568e-3; step-1 within 1% of 2.2716938e-3; (max−min)/step1 < 1e-10; mass and volume drift < 1e-10; 311→311 verts / 32→32 interface.
- 3D 2000-step (`@pytest.mark.slow`): plateau at step 2 (step 1 = 7.0325e-05 one-step transient); single unique value steps 2–2000. Guard: `TestStaticDroplet3DRetopologyFloor` (10 steps, ~6 s, `SETTLE_STEPS=2`): step-0 within 1% of 6.0153e-05; plateau within 1% of 7.3768e-05; plateau rel spread < 1e-10; mass drift < 1e-10; volume spread checked POST-settle only (boundary-shell artefact); 472→472 / 98→98.
- Raw JSONs: `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection{,_pre_refactor,_2d_longrun,_3d_longrun,_with_redistmass,_without_redistmass}.json`.

### 5.5 M1 rollout (2026-04-29)
`redistribute_mass: bool = True` added + forwarded in: `setup_dam_break_multiphase` (partial), `setup_cube_to_droplet` (closure setdefault), `setup_shearing_plate_droplet` (custom `_make_periodic_multiphase_retopo` extended to snapshot_geometry_multiphase → retopologize_periodic → mps.refresh → redistribute_mass_multiphase → compute_phase_pressures), `setup_electrolysis_bubble`. 5-step 2D smokes: per-phase mass drift ≤ 1.94e-15 (dam_break/shearing_plate/electrolysis); cube_to_droplet phase-1 drift 1.32e-16 (phase-0 non-zero BY DESIGN: `AtmosphericPressureBC` is a deliberate mass source/sink). Electrolysis EOS sanity: `eos_gas.density(200) = 10.02` (rho0=10, K=1e5, n=1, clip (0.5, 2.0)).

### 5.6 A.1 / A.2 (2026-04-24) and refactor re-validation (2026-04-26)
- A.1: `ddgclib/data/_conservation.py` (`compute_conservation`, `as_jsonable`, `drift_fractions`); `StateHistory` opt-in `conservation=True, dim=...`; 18 tests.
- A.2: `test_multiphase_flat_interface.py` — 2D γ=0 bulk 3.14e-16 / interface 2.22e-16; 2D γ=0.05 same; 3D γ=0 bulk 1.19e-15 / interface 1.05e-15; 3D γ=0.05 1.04e-15. Conclusion: per-phase pressure-flux sum cancels exactly; planar `hndA_i_interface` exactly zero.
- 2026-04-26 refactor (commit 8321c71): replaced manual `HC._simplices = ...` workaround with `hyperct.ddg.connect_and_cache_simplices` / `boundary_from_simplices` / `invalidate_simplex_cache` as the primary integrator path (`_integrators_dynamic.py:188-196`); `test_simplex_aware_duals.py` 13 tests; A.1 18/18 and A.2 4/4 re-validated bit-comparable.

---

## 6. Explicit next steps / TODO (priority order as of last entry, 2026-06-02)

1. **Tier 2B step 1 (2D integrated γ-flux curvature rewrite)** — the ONLY lever on the 2D 2.27e-3 floor; ~4–8 h. Replace `hndA_i_interface` on the 2D path with existing `surface_tension_force_2d` (`ddgclib/operators/curvature_2d.py:91-181`, exact for piecewise-linear interfaces), called from `_interface_surface_tension` for dim=2; re-run `diagnose_a5_bisection.py`; target: 2.37e-3 / 2.27e-3 → machine precision; then **re-pin `TestStaticDroplet2DRetopologyFloor` at the new floor**. Audit note reference: `diagnose_a5_bisection.py:402-409` (O(h) → O(h²) → exact convergence rationale). *[Completeness-critic note 2026-07-02: the dim=2 `'integrated'` path already calls `surface_tension_force_2d` in current code (`multiphase_stress.py:262,:277`) while the floor remains 2.2717e-3 — see 06_known_issues §1.1 before acting on this item.]*
2. **Probe 5 — skip-retopology gate** (`max ||x_i − x_i_prev|| < eps` ⇒ skip `_retopologize`); ~1–2 h; first cut `eps = 1e-4 * h_min`; on dynamic runs eps must be ∝ h_local (staleness-vs-churn tradeoff). Would collapse 3D A.5.b → A.5.a (6.02e-05). Any breakage caught immediately by the two pinned floor tests.
3. **Pivot to dynamic validation — M4 / Tier 3A** (2D oscillating droplet Rayleigh–Lamb): the ~37% of 2D baseline not captured statically; gated on Probe 5 and/or adaptive remesh. Targets: tail_growth < 1.0, l2_error < 0.2.
4. **A.3 single-phase probes (Tier 1A + 1B.i/ii)** — still never run; required before any BC work; determines whether D1 or D2 dominates single-phase; bisects integrator vs dual-refresh vs Delaunay churn.
5. **A.4 upstream hyperct**: fix `edge_split_2d` mass averaging (`hyperct/remesh/_operations_2d.py:198` — use length-weighted or explicitly conservative split, add Σm-invariance unit test) and refactor global `h_local` to per-vertex/per-edge local scale. Gate: full ddgclib fast suite passes after symlink update. Unblocks adaptive remesh for M3.3 / M4.
6. Minor open checks: does `hndA_i_interface[dim=2]` alias the 2D FTC form?; audit setup-time volume reporting vs IC mass ratios (currently safe); 3D metric harness for Tier 3B.

**Explicit DO-NOTs (from the plan's latest guidance):**
- Do NOT chase the 3D ×1.23 gap with further curvature-stencil variants (Probe 2 closed that lane mathematically).
- Do NOT add further redistribute_mass guard changes (Probe 4 fully wired, A/B-confirmed).
- Do NOT touch BC isolation (Tier 1E / `OutletDeleteBC`) until A.3 single-phase probes run.
- Do NOT enable `remesh_mode='adaptive'` in cases until the A.4 hyperct fixes land.
- Do NOT switch to Eulerian integrators (library-wide Lagrangian formalism; `euler_velocity_only` is validation-only).

---

## 7. Stale claims, contradictions, footguns (for future agents)

- **Doc self-contradiction (resolved)**: the 2026-05-27 "(later)" entry listed Probe 4 (guard fix) as pending; it had shipped 2026-04-29 (Phase 2c status row). A/B on 2026-05-27 confirmed it live. Trust the Phase-A status table, not the per-entry "next probe options" lists.
- **Harness flag footgun**: `diagnose_a5_bisection.py --redistribute-mass` uses `action='store_true'` ⇒ harness default False **overrides** `setup_oscillating_droplet`'s production default True. Running WITHOUT the flag exercises the legacy guard path and reproduces the pre-fix 1.44e-3 — this is expected, not a regression.
- **Setup/runtime `split_method` must match** (setup refresh + YL mass adjustment vs runtime retopo partial) — mismatch produces ×147 (2D) rho jumps; enforced by parameters + docstring in `setup_oscillating_droplet`, but any NEW multiphase setup must replicate this.
- **Stale number**: original plan's static-droplet 2D target 7.41e-3 superseded by live `summary = 1.20e-3`.
- **3D volume metric**: never assert |dV/V0| from step 0 in 3D — the 0.305 step-0→1 one-shot is boundary dual-shell zeroing, by design. Assert on the post-settle window (step ≥ 2).
- **3D flat/aligned interface meshes**: default `Complex(3).triangulate()+refine_all()` does NOT align z=0; use an explicit Kuhn-decomposed cube.
- **hyperct test noise**: 9–39 "errors" = missing `pytest-benchmark` fixture, pre-existing/environmental.
- **Two copies of `volume_of_geometric_object`** exist (canonical `hyperct/ddg/_geometry.py` + legacy `hyperct/ddg/barycentric/_duals.py`) — fixes must be applied to both.
- Any code mutating topology outside `connect_and_cache_simplices` MUST call `invalidate_simplex_cache(HC)` (clears `_simplices` AND `_edge_to_apex`); read-only consumers should `getattr(HC, '_simplices', None)`-check with legacy `vi.nn ∩ vj.nn` fallback (template: `_apex_via_simplex_cache` in `_curvatures_heron.py`).
- Symlink path note: doc refers to hyperct at `/home/stefan_endres/projects/hyperct/`; current env user is `endres` — verify symlink target validity before running.
- All fixes described were left **working-tree only** ("user stages/commits as preferred") — verify they are actually present/committed before relying on line numbers.

---

## 8. Key file / artifact index

| Artifact | Role |
|---|---|
| `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` | Main A.5 harness (flags: `--skip-3d --n-steps --split-method --redistribute-mass --results-suffix`, curvature-path CLI); audit note :402-409 |
| `cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py` | Single-retopo-call per-vertex per-phase state diff |
| `cases_dynamic/oscillating_droplet/diagnose_static.py` | 7-section static smoke diagnostic |
| `cases_dynamic/oscillating_droplet/src/_setup.py` | `setup_oscillating_droplet` (:24 signature, :42 redistribute default, :144 setup refresh, :169 YL mass adjust) |
| `cases_dynamic/oscillating_droplet/results_a5_bisection/*.json` | Raw probe data (baseline, pre_refactor, 2d/3d_longrun, with/without_redistmass) |
| `cases_dynamic/oscillating_droplet/{static_droplet_2D.py, oscillating_droplet_2D.py}` | Tier 2B / 3A case scripts; scores in `results*/score.json` |
| `ddgclib/tests/test_case_oscillating_droplet.py` | `TestStaticDroplet2DRetopologyFloor` (fast) + `TestStaticDroplet3DRetopologyFloor` (slow) — pinned floors |
| `ddgclib/operators/mass_redistribution.py` | `snapshot_pressure_multiphase` (:42-51), `snapshot_geometry_multiphase`, `redistribute_mass_multiphase` (:199; fixed guard :54-74, :308-310; legacy guard was :250) |
| `ddgclib/dynamic_integrators/_integrators_dynamic.py` | `_retopologize` (:48-237; boundary zeroing :211/:225; `skip_triangulation` :188; simplex path :188-196), `_retopologize_multiphase` (plan cites :325-406; **current code: :391-475**, verified 2026-07-02) |
| `ddgclib/operators/multiphase_stress.py` | `multiphase_stress_force`, `_interface_surface_tension` (`curvature_path='integrated'|'stokes'`) |
| `ddgclib/_curvatures_heron.py` | `hndA_i_interface`, `integrated_hndA_i_interface`, `_apex_via_simplex_cache`, `_apex_via_interface_triangles` |
| `ddgclib/operators/curvature_2d.py:91-181` | `surface_tension_force_2d` — exact PL 2D operator, target of Tier 2B step 1 |
| `ddgclib/geometry/_dual_split_2d.py` | `split_dual_polygon_2d` / `split_dual_polyhedron_3d` (`split_method='exact'`) |
| `ddgclib/multiphase.py` | `split_method` default (:417-503), `assign_simplex_phases_from_vertices` (:237-287) |
| `ddgclib/_boundary_conditions.py` | `OutletDeleteBC` :395-440, `OutletBufferedDeleteBC` :443-522, `PeriodicInletBC` :532-680 (D2, un-probed) |
| `ddgclib/data/{_history.py,_conservation.py}` | Tier-0 diagnostics (`StateHistory` callback :71-115) |
| `ddgclib/analytical/_integrated_comparison.py` | `integrated_pressure_error`, `integrated_l2_norm`, `compare_stress_force`, `volume_averaged_scalar` — ONLY sanctioned error metrics (volume-averaged, never point-wise) |
| `hyperct/ddg/_geometry.py:79-105` (+ `hyperct/ddg/barycentric/_duals.py:328-365`) | `volume_of_geometric_object` degenerate-base fix |
| `hyperct/ddg/_retriangulation.py` | `connect_and_cache_simplices`, `get_edge_apex_map`, `invalidate_simplex_cache` |
| `hyperct/remesh/_operations_2d.py:198` | OPEN BUG: `edge_split_2d` mass inflation (9.7→187/100 steps) |
| `ddgclib/tests/test_{conservation,multiphase_flat_interface,simplex_aware_duals,simplex_aware_curvature,integrated_validation,stress}.py` | Regression suites referenced by the plan |
| `docs/{3d_multiphase_interface_pressure_fix.md,3d_simplex_aware_dual_fix.md,interface_stress_rewrite.md}` | Background fix write-ups (interface residual numbers at `3d_multiphase_interface_pressure_fix.md:147-177`) |
| Memory files | `project_3d_boundary_dual_bug.md`, `feedback_multiphase_mass_redistribution.md` |
