# 08 — Debug Session 2026-07-02/03: Sequential Lane Report

> Sources: lane logs in [`debug_session/`](debug_session/) (lanes 1–5 and 7; there is no
> lane 6), audit synthesis [`07_physics_audit.md`](07_physics_audit.md) + per-item probes in
> [`audit/`](audit/). Written 2026-07-03 by the session-recorder workflow.
> **All fixes are working-tree only — no git operations were performed, in EITHER repo**
> (ddgclib and the hyperct sibling at `/home/endres/projects/hyperct/hyperct`, edited via
> the `./hyperct` symlink). Every number below is sourced from a lane log; none are invented.

## 1. Executive summary

Six sequential lanes executed the fix order recommended by the 2026-07-02 physics audit
(07 §5). All six **landed**; one production switch (3D exact dual volumes) was built,
measured, and intentionally backed out. The primary failing case — the 2D oscillating
droplet vs the Rayleigh–Lamb analytical envelope — improved from l2 2.95 to **0.179**
(−94.0 %) and KE tail_growth 2.35 to **0.999** (−57.5 %), meeting both lane targets
(l2 < 0.2, tail < 1.0) for the first time. The fast suite went from
796 passed + 1 pre-existing failure to **833 passed, 0 failed** — the first fully green
fast suite of the session. All four pinned static floors are unchanged
(2D 2.3748568e-3 / 2.2716938e-3; 3D 6.0153e-5 / 7.3768e-5).

### Metrics before → after

| metric | baseline (session start) | final (post-lane-7) | Δ |
|---|---|---|---|
| osc l2_error_normalized | 2.9519 | **0.1785660454150319** | **−94.0 %** |
| osc tail_growth | 2.3530 | **0.9992507831101141** | **−57.5 %** |
| osc linf_error_normalized | 5.9145 | **0.32842491275938274** | **−94.4 %** |
| osc KE_max | ~4.1e-02 J, still growing at t_end (pre-lane-5) | **8.317e-07 J, physical overdamped decay** | ~5e4× lower; envelope matches analytical (decay 6.79 vs 7.18 1/s) |
| equil summary (static_droplet_2D) | 0.0011636 | 0.0011847162859108737 | +1.81 % (inside the 5 % band; drift-metric sensitivity, lane 3) |
| fast suite (`-m "not slow"`) | 796 passed, 1 failed (pre-existing `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`) | **833 passed, 0 failed**, 12 skipped, 17 deselected, 3 xfailed in 56.60 s | +37 tests, stale expectation fixed, green |
| pinned floor tests | 12 passed | **14 passed** (12 pinned + envelope regression + companion) | floors bit-compatible, no re-pin |
| 2D floor (frozen / post-retopo) | 2.3748568e-3 / 2.2716938e-3 | unchanged | — |
| 3D floor (step0 / plateau) | 6.0153e-5 / 7.3768e-5 | unchanged | — |
| hyperct suite | 301 passed (+ 39 pre-existing pytest-benchmark fixture errors) | **334 passed**, 40 skipped, 6 xfailed, same 39 errors (post-lane-4; lane 7 did not touch hyperct) | +33 tests |
| osc mass_drift | — | 2.676988154993443e-14 | machine precision |

### Per-lane metric trail

| lane | status | osc l2 | osc tail | equil | fast suite |
|---|---|---|---|---|---|
| (baseline) | — | 2.9519 | 2.3530 | 0.0011636 | 796 P / 1 F |
| 1 zero-gauge-pressure | landed | 0.7090740204580823 | 2.2052688800146534 | 0.0011351 | 804 P / 1 F |
| 2 eos-consistency | landed | bit-identical to L1 | bit-identical | bit-identical | 818 P / 1 F |
| 3 exact-dual-volumes | landed (2D; 3D backed out) | 0.48991833470391266 | 1.7250489596305962 | 0.0011847162859108737 | 824 P / 1 F |
| 4 remesh-upstream | landed | bit-identical to L3 | bit-identical | bit-identical | 824 P / 1 F |
| 5 dynamic-config-sweep | landed | 0.1785660454150319 | 0.9992507831101141 | bit-identical | 825 P / 1 F |
| 7 cleanup-regression-lockin | landed | bit-identical to L5 | bit-identical | bit-identical | **833 P / 0 F** |

(The 1 F through lane 5 is always the same pre-existing
`test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`
— a stale expectation vs newer hyperct, fixed in lane 7. No lane introduced a NEW failure.)

linf trail: 5.9145 → 1.05384362884825 (L1) → 0.8503543927096853 (L3) →
0.32842491275938274 (L5, final).

---

## 2. What landed, per lane

### Lane 1 — zero-gauge-pressure sentinel + multiphase momentum fix
Log: [debug_session/lane1-zero-gauge-pressure.md](debug_session/lane1-zero-gauge-pressure.md).
Audits: zero-gauge-pressure (CONFIRMED high), multiphase-momentum (CONFIRMED high).

Root cause: multiphase code keyed "phase present at vertex" on the stored pressure float
being exactly `0.0`; at the case default gauge P0=0 this misread deleted the
pressure-difference flux one-sidedly, breaking gauge invariance and Newton's third law.

- `ddgclib/operators/multiphase_stress.py` — geometric presence test
  (`_phase_present_at`: `dual_vol_phase[k] > 1e-30`) replaces the `val == 0.0` sentinel in
  `_phase_pressure`; per-phase sub-face loop of `multiphase_stress_force` rewritten so
  absent-phase handling is symmetric on both face ends (one-sided skip branch removed;
  neighbour-only presence now mirrors the one-sided extrapolation so F_ij = −F_ji holds).
- `ddgclib/multiphase.py` — interface `v.p` average filter changed from `p_phase[k] != 0.0`
  to volume/mass-keyed presence.
- NEW `ddgclib/tests/test_multiphase_gauge_invariance.py` (8 tests): gauge-offset
  invariance (max |F(p+1000)−F(p)| measured 8.9e-15 N, pre-fix 1.17 N), free-vertex
  momentum closure (|ΣF|/max|F| 1.5e-13, pre-fix 0.70), fallback semantics, interface-mean
  inclusion of bitwise-0.0 present phases.

Effect: osc l2 −76 %, linf −82 %, tail −6.3 %; equil −2.4 %; floors bitwise unchanged
(at static equilibrium the fallback coincides with truth, as predicted).

### Lane 2 — EOS consistency
Log: [debug_session/lane2-eos-consistency.md](debug_session/lane2-eos-consistency.md).
Audits: eos-formulas (medium), multiphase-eos-interface (low/latent).

- `ddgclib/eos/_tait_murnaghan.py` — `rho_clip` now applied consistently in
  `pressure()`/`density()`/`sound_speed()` via shared `_clip_rho()` (coherent saturating
  model); saturation never silent: per-method `clip_count` counters + warn-once
  `RuntimeWarning`.
- `ddgclib/eos/_multiphase_eos.py` — NEW shared `interface_mean_pressure(v, n_phases)`
  (THE interface `v.p` convention, single source of truth); `MultiphaseEOS.__call__`
  guarded with `v.phase >= 0` (interface sentinel −1 gets the mean convention instead of
  the silent numpy `p_phase[-1]` wrap); fallback path B raises `ValueError` for
  `v.phase < 0` (tripwire).
- `ddgclib/multiphase.py` — `compute_phase_pressures` delegates to the shared helper
  (verified behaviourally identical); stale docstring fixed.
- `ddgclib/eos/_ideal_gas.py` — `density()` floored at 0; defaults documented.
  `ddgclib/eos/_base.py` — sound-speed docstrings corrected.
- NEW `ddgclib/tests/test_eos_consistency.py` (14 tests).

Effect: all production metrics **bit-identical** to post-lane-1 (latent-trap closure, not a
behaviour change). Measured: clip engagement is a frozen-wall-vertex phenomenon (0 interface
engagements in 1839 steps); the (0.8, 1.2) clip band is load-bearing (widening to (0.5, 2.0)
made l2 4.4× WORSE) — kept.

### Lane 3 — exact simplex-container dual volumes (2D landed; 3D built + backed out)
Log: [debug_session/lane3-exact-dual-volumes.md](debug_session/lane3-exact-dual-volumes.md).
Audit: dual-volume-3d (CONFIRMED high).

- hyperct: NEW `hyperct/ddg/_dual_volume.py` — `simplex_dual_volumes(HC, dim)` /
  `vertex_dual_volume(HC, v, dim)`, the exact barycentric closed form
  Vol_i = (1/(dim+1))·Σ_{T∋i}|T|, partition-of-unity exact to < 1e-12 in 2D AND 3D;
  exported; `compute_vd` now records `HC._vd_method`. NEW
  `hyperct/tests/test_dual_volume.py` (11 tests).
- ddgclib: `ddgclib/operators/stress.py` — `dual_volume` / `cache_dual_volumes` dim==2 use
  the exact routine when `HC._simplices` present and duals are barycentric
  (`_use_exact_barycentric_volume`); dim==3 behaviour unchanged, carries
  `NOTE(lane3-dual-volume)` explaining the gated switch.
  `ddgclib/tests/test_stress.py` — `TestDualVolumeExactSimplex` (6 pass + 1 strict xfail
  documenting that production `dual_volume(dim=3)` still does not tile: 0.86–0.94).

Effect: osc l2 −30.9 %, tail −21.8 %, linf −19.3 % (wall/corner dual cells now get their
true measure at every dynamic retopo); equil +4.37 % (drift-metric sensitivity to bit-level
trajectory changes, within the 5 % band); 2D Σ dual_vol == domain area to 1e-12.

### Lane 4 — upstream hyperct remesh fixes (open problem A.4)
Log: [debug_session/lane4-remesh-upstream.md](debug_session/lane4-remesh-upstream.md).

- hyperct `remesh/_operations_2d.py` — **mass-conservative `edge_split_2d`** (endpoints
  cede the exact dual-area mass fraction; sum(m)/sum(m_phase) invariant; midpoint `u` is
  the mass-weighted mix → momentum conserved, KE non-increasing); `edge_collapse_2d` —
  upfront abort (no partial mutation), `m` AND `m_phase` additive (was: half the merged
  per-phase mass silently destroyed by averaging), momentum-conserving merged `u`;
  `m_phase` added to `_SKIP_ATTRS`.
- hyperct `remesh/_driver.py` — per-edge local length scale (`length_scale='local'`
  default; `alpha_min`/`alpha_max` × `_edge_h_local`), `smooth_skip_interface` kwarg.
- hyperct `ddg/_retriangulation.py` — NEW `rebuild_simplex_cache_2d(HC)` (ghost-K3-filtered
  rebuild of `HC._simplices` after local ops).
- hyperct tests: `test_remesh.py` (2 expectations re-pinned to the conservative behaviour,
  documented old → new), NEW `test_remesh_conservation.py` (22 tests).
- ddgclib `dynamic_integrators/_integrators_dynamic.py` (adaptive branch of
  `_retopologize` only) — calls `rebuild_simplex_cache_2d` instead of
  `invalidate_simplex_cache` and uses `boundary_from_simplices`; the delaunay path is
  untouched. `cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py` —
  local-scale kwargs (`alpha_min=0.3, alpha_max=2.5`), `smooth_iterations=0`.

Effect: production (delaunay) metrics **bit-identical** to post-lane-3. A.4 win condition
met: adaptive-mode mass blow-up (documented 9.7 → 187) now machine-precision flat
(drift 2.95e-15); KE explosion fixed (tail 4.99 → 1.37 on the 200-step smoke). Key
diagnosis: the adaptive branch's `invalidate_simplex_cache` was silently downgrading ALL
geometry (dual construction, boundary, lane-3 exact volumes) to 1-skeleton fallbacks —
any path that drops the simplex cache without rebuilding it reintroduces pre-lane-3 errors.

### Lane 5 — dynamic-config sweep → `dual_only` retopo policy is the new 2D default
Log: [debug_session/lane5-dynamic-config-sweep.md](debug_session/lane5-dynamic-config-sweep.md).

- `cases_dynamic/oscillating_droplet/src/_params.py` — NEW `retopo_policy_2d = 'dual_only'`
  (sweep table + rationale in comment). c_s stays 1 m/s, K_d=800/K_o=1000 unchanged.
- `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` — under `dual_only`, rebinds
  `retopo_fn = partial(retopo_fn, skip_triangulation=True)`: builder connectivity is kept
  for the whole run while duals, per-phase splits, mass redistribution, and EOS pressures
  still refresh every step.
- `ddgclib/tests/test_case_oscillating_droplet.py` — NEW `TestDualOnlyRetopoPolicy2D`
  (fast, ~1 s).
- `cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json` — re-pinned to
  the new official score (old pre-lane-1-era values documented in the lane log).

Effect: l2 0.48992 → **0.17857**, tail 1.72505 → **0.99925** — both lane targets met for
the first time. KE(t) quantitatively reproduces the analytical overdamped Rayleigh–Lamb
envelope (peak t 0.0549 vs 0.0598 s; decay 6.79 vs 7.183 1/s; KE_max 8.3e-07 J vs 4.1e-02 J
still-growing under per-step Delaunay — i.e. ~100 % of the old KE was retopology-injected
noise). Sweep also established: displacement gate has no good eps for this case; hybrid
gate+refresh is strictly worse; adaptive (lane-4-fixed) is second-best (l2 0.326);
dual_only is c_s-insensitive while per-step Delaunay + c_s=5 is catastrophic (interface
destroyed); the l2 metric is anti-convergent under refinement (see §4/§5 caveats).

### Lane 7 — cleanup and regression lock-in (session-closing)
Log: [debug_session/lane7-cleanup-regression-lockin.md](debug_session/lane7-cleanup-regression-lockin.md).

- `ddgclib/tests/test_simplex_aware_duals.py` — the session-long pre-existing failure
  (`test_raises_unsupported_dim`, stale expectation vs newer hyperct's generalized
  `boundary_from_simplices`) rewritten to the NEW contract + companion test locking the
  silent-skip branch.
- `ddgclib/analytical/_integrated_comparison.py` — **factor-2 validation bug fixed**
  (audit gravity-bodyforce §3e, CONFIRMED high): spurious `2.0 *` removed from the
  production `_dual_cell_pressure_integral_2d_simple` (Dunavant weights sum to 1) and the
  erroneous `ti+tj>1` skip removed from the Duffy variant. Pre-fix,
  `volume_averaged_scalar(1) == 2.0`, `HydrostaticPressure`/`LinearPressureGradient`
  assigned 2× pressure whenever duals existed, and `integrated_pressure_error` inverted
  good and bad fields. `_divergence_theorem.py` docstring-only correction.
  NEW `TestDualCellVolumeAverage2D` (5 tests) in `test_integrated_validation.py`.
  Grep-verified: NO pre-existing test had the factor 2 baked in.
- `ddgclib/tests/test_case_oscillating_droplet.py` — NEW fast (~7 s)
  `TestOscillationEnvelopeRegression2D`: full production decay envelope at refinement 2/2,
  pinned l2 0.050006441230559064 / tail 0.9130549583972877 with ~10 % headroom
  (L2_MAX 0.0550, TAIL_MAX 1.004); documented A/B: reverting the lane-5 policy scores
  l2 1.384569 on this fixture — the pin catches that regression class by >25×.
- `docs_temp/05_test_cases_and_validation.md:101` — corrected: the <1e-13 linear-precision
  gate is scoped to barycentric/p_ij (the production method, PASS at 2.2e-16..7.6e-16);
  the `bary` integration-polygon and circumcentric variants have measured PRE-EXISTING
  floors (5.6e-2..1.2e-1 / 2.7e-3..5.7e-3) — the benchmark runner's FAIL banner for them
  is expected and not a session regression.

Effect: fast suite green (833 P / 0 F); all four battery metrics bit-identical to
post-lane-5 (validation utilities + tests + docs only; `volume_averaged_scalar` is not on
the droplet path).

---

## 3. What was reverted, backed out, or rejected (and why)

1. **3D exact-dual-volume production switch (lane 3) — built, measured, backed out.**
   Wiring `simplex_dual_volumes` into the 3D production path moves the pinned 3D
   static-droplet retopo floor UP: 7.376786e-05 → 7.616854e-05 (+3.26 %), and the lane
   decision rule only allows re-pinning a *lower* floor. Fully diagnosed: NOT a bug in
   the exact volumes — the exact measure honestly reports a larger real settle-step
   volume jump (order-dependent qhull tie-breaking on the cospherical droplet cloud at
   retopo #1–2) that `redistribute_mass_multiphase`'s mass-conserving global rescale
   converts into a uniform ~1.3 Pa droplet pressure offset. The switch is one conditional
   in two places (`NOTE(lane3-dual-volume)` markers in `stress.py` and
   `_integrators_dynamic.py`), the upstream routine is tested and ready, and the strict
   xfail `test_partition_of_unity_3d_jittered_production` documents the gap.
2. **EOS clip-band widening (lane 2 A/B) — rejected.** Widening (0.8, 1.2) → (0.5, 2.0)
   made l2 4.4× worse / linf 5.6× worse (the wider window lets the frozen-wall bookkeeping
   artifact inject ~20 kPa instead of capped ~375 Pa near-wall pressure). Band kept; the
   real target is the frozen-wall-mass artifact, now permanently visible via `clip_count`.
3. **Displacement-gate and hybrid retopo configs (lane 5 sweep) — closed for this case.**
   No good eps exists (small eps degenerates to per-step Delaunay with stale-dual
   episodes, tail 2.31; large eps accumulates displacement so each rewire is a bigger
   shock, l2 0.83/tail 1.92); hybrid gate+dual-refresh is strictly worse (hybrid_02
   exploded: l2 2.59, tail 7.45). Do not resurrect for the 2D droplet; the gate remains
   valid for its original 3D static purpose.
4. **Laplacian smoothing in adaptive remesh (lane 4) — shipped config is smoothing-off.**
   Smoothing teleports parcels without mass remap; the resulting op churn + retagging
   eroded the droplet (interface 60 → 29 edges in 45 steps; bulk-only smoothing trades
   l2 for a returning KE pump). Needs ALE-style mass/momentum-remapped smoothing — a
   future lane, not a remesh-op bug.
5. **Raising c_s (lane 5 §4) — rejected; c_s = 1 m/s stays.** Under dual_only it buys
   nothing (R_max matches to ~1e-6) and costs 5–10× the steps; under per-step Delaunay
   c_s=5 is catastrophic (KE 18.7 J, interface ring destroyed mid-run) because every
   rewiring volume-jolt maps through K=20 000 Pa instead of 800 Pa.
6. **Lane-1 one-sided skip branch (removed, not reverted).** Under corrupted/stale
   interface tags the operator now books a symmetric neighbour-extrapolated flux instead
   of silently dropping one side.

---

## 4. Fix → effect attribution chain

The l2 error decomposed into three separable mechanisms, each killed by a different lane:

1. **Spurious momentum injection during the wave-launch transient** (lane 1). The `==0.0`
   phase-presence sentinel misfired at gauge P0=0, injecting a non-conservative force
   (|ΣF| up to 86 % of max|F|) that contaminated the Rayleigh–Lamb fit first-order.
   Fix → gauge invariance and pairwise antisymmetry restored to machine precision
   → **l2 2.95 → 0.71**. tail barely moved (2.35 → 2.21), proving the KE tail was a
   different mechanism.
2. **Wall/corner dual-volume misestimation refreshed at every retopo** (lane 3). Legacy
   2D polygon reconstruction was short by h²/8 at the 4 box corners and had
   ValueError→0.0 degenerate-vertex zeroing; `rho = m/dual_vol` at wall vertices was
   overestimated, feeding the EOS. Exact simplex-container volumes →
   **l2 0.71 → 0.49, tail 2.21 → 1.73** — consistent with lane 2's finding that wall-vertex
   EOS saturation/near-wall flux was the dominant KE-tail driver.
3. **Per-step global Delaunay reconnection jolts** (lane 5, enabled by lanes 1–4 making
   the effect measurable). 2D Delaunay of the moving droplet cloud is non-unique near the
   symmetric interface ring; every reconnection changes dual volumes discontinuously and
   the mass-redistribution rescale converts that into pressure/mass jolts. Freezing
   connectivity (dual_only) while refreshing duals/pressures every step →
   **l2 0.49 → 0.179, tail 1.73 → 0.999**, and KE(t) collapses onto the analytical
   overdamped envelope — ~100 % of the residual KE was retopology noise.

Lanes 2, 4, 7 were deliberately production-neutral (bit-identical metrics): lane 2 closed
latent EOS traps and made saturation visible; lane 4 made adaptive mode conservative and
scoreable (unblocking the lane-5 sweep's adaptive arm and future large-deformation work);
lane 7 fixed the sanctioned-validation factor-2 (globally high severity, zero droplet
impact) and locked everything behind regression tests.

Cross-lane dependency worth remembering: lane 4's diagnosis that
`invalidate_simplex_cache` without a rebuild silently reverts lane 3's exact geometry —
the lanes compose only because the adaptive branch now rebuilds the cache.

---

## 5. Where the next session should start

1. **Commit the working tree.** Every fix in BOTH repos (ddgclib + hyperct) is
   uncommitted. Suggested split mirrors the lanes (each lane log has the complete file
   list); test counts to verify after commit: ddgclib fast 833 P / 0 F, hyperct
   334 P + 39 pre-existing benchmark-fixture errors, floor battery 14 P.
2. **3D exact-dual-volume switch (highest-value deferred item).** Recipe in lane 3 §"What
   the next lane must know": flip all THREE switch points together (`stress.py` dim==3
   `dual_volume`, `cache_dual_volumes` dim in (2,3), `_integrators_dynamic.py` step 5b
   batch_e_star preference), re-pin `TestStaticDroplet3DRetopologyFloor.EXPECTED_PLATEAU_MAXF`
   7.3768e-05 → 7.616854e-05 (measured bit-stable), flip the strict xfail. Candidate
   companion fix: canonicalize Delaunay input order in `connect_and_cache_simplices` to
   kill the settle-step artifact (re-pins every 3D dynamic number).
3. **Residual l2 = 0.179 is a smooth ~5–10 % over-decay, and part of it is probably the
   REFERENCE's error** (lane 5 §5/§7): the single-fluid Lamb β=25.0 ignores outer-fluid
   damping (ρ_o=1000 bath, no-slip box), and the metric is anti-convergent under
   refinement. Next validation step: a two-fluid (Prosperetti-type) reference or a
   viscous-corrected β — do NOT chase l2 → 0 against the current reference.
4. **Large-deformation retopo remains open** (dam break, detaching bubbles cannot freeze
   connectivity). Candidate real fixes: conservative old-dual→new-dual remap (p_ref
   benchmark pattern) or ALE-style mass/momentum-remapped smoothing (lane 4 §6.1,
   lane 5 §8).
5. **3D dynamic score harness still missing** (06 §1.6; DEVELOPMENT.md backlog item) —
   without it the 3D fan-volume churn mechanism and any 3D policy sweep are unmeasurable.
6. Smaller open items: tag-consistency flag 19 (swallowed closure validation,
   `multiphase.py:379-382`, lane 1); `dual_only_noredist`'s cleanest-of-all KE decay
   (tail 0.111) — if the wave-launch transient is fixed, redistribution could plausibly
   be turned OFF for a more physical benchmark (lane 5 §8); `mps=` parameter of
   `adaptive_remesh` reserved/unused; phase-tag hazard for direct `edge_split_2d` users
   (refresh before force evaluation, lane 4 §6.4).
7. **Re-baseline caution:** any pre-2026-07-02 logged numbers from 2D cases that assign
   ICs after `compute_vd` or report `integrated_pressure_error`/`integrated_l2_norm` are
   2× off on the analytical side (lane 7 factor-2 fix). Also, interface `v.p` diagnostics
   at gauge P0=0 changed (correctly) after lane 1.
8. **Traps to not re-trip:** don't silence the TaitMurnaghan `RuntimeWarning`
   (read `eos.clip_count`); don't drop `HC._simplices` without
   `rebuild_simplex_cache_2d`; don't blind-apply `dual_only` to the 3D droplet (different
   boundary-volume bookkeeping); judge KE health by the envelope, not the fragile binary
   tail_growth (peak sits 2 ms before the half-run split).

## 6. Artifact index

- Lane logs: `docs_temp/debug_session/lane{1,2,3,4,5,7}-*.md` (no lane 6).
- Audit basis: `docs_temp/07_physics_audit.md` + `docs_temp/audit/*.md` (15 items).
- Score JSON copies per lane: scratchpad `wf3/lane*_{osc,equil}_score.json`
  (session-scratchpad, ephemeral — the numbers are preserved in the lane logs).
- Re-pinned case baseline: `cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json`.
- New test files: `test_multiphase_gauge_invariance.py`, `test_eos_consistency.py`,
  `hyperct/tests/test_dual_volume.py`, `hyperct/tests/test_remesh_conservation.py`,
  plus new classes in `test_stress.py`, `test_case_oscillating_droplet.py`,
  `test_integrated_validation.py`, `test_simplex_aware_duals.py`.
- Status-log entry: `debugging_plan.md` (2026-07-02/03 entry, top of the Status log).
