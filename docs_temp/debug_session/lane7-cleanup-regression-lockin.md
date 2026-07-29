# Lane 7 — cleanup and regression lock-in (session-closing lane)

Date: 2026-07-02/03.  Verdict: **LANDED** — fast suite fully green for the
first time this session (833 passed, 0 failed), all four battery metrics
bit-identical to post-lane-5, floor constants untouched.

Provenance note: an interrupted first run of this lane applied the code/test
edits on 2026-07-02 18:33–18:54 (probes `scratchpad/wf3/lane7_probe_factor2.py`,
`lane7_probe_shortrun.py`) but died before the measurement battery and this
log.  The closing session re-verified every edit from first principles, added
the `05_test_cases_and_validation.md` correction, ran the full battery, and
wrote this log.  All changes below are lane-7 changes regardless of which run
typed them.

## 1. What changed (file:line)

### (T1) Stale test expectation — `test_raises_unsupported_dim`
- `ddgclib/tests/test_simplex_aware_duals.py:159-168` — rewritten to the NEW
  hyperct contract.  Judged from the code
  (`hyperct/ddg/_boundary.py:18-97`, symlink → sibling repo): newer hyperct
  intentionally generalizes `boundary_from_simplices` to any `dim >= 1`
  (raises `ValueError("... dim >= 1 ...")` only for `dim < 1`,
  `_boundary.py:57-60`) and **silently skips** simplex entries whose length
  != dim+1 (`_boundary.py:81-82`, ghost-dedup leftovers).  The old
  `dim ∈ {2, 3}` whitelist is gone and the generalization is coherent with
  the face-counting algorithm (works for any dim), so the OLD contract was
  not "actually correct" — the test was updated, no upstream guard re-added.
- `ddgclib/tests/test_simplex_aware_duals.py:170-176` — NEW companion
  `test_mismatched_simplex_entries_skipped` locks the silent-skip branch
  (dummy `(None,)` entry at dim=4 → empty boundary set, no exception).

### (T2) Validation-utility factor-2 bug (audit: docs_temp/audit/gravity-bodyforce.md §3e, CONFIRMED high)
- `ddgclib/analytical/_integrated_comparison.py:68-120`
  (`_dual_cell_pressure_integral_2d`, Duffy variant): removed the erroneous
  `if ti + tj > 1.0: continue` — the Duffy (collapsed-square) map
  `l1 = ti, l2 = tj*(1-ti)` already keeps every point of the unit square
  inside the triangle, so the skip under-integrated by ~1/3.  The
  `2.0 * tri_area` factor in THIS variant is **correct** (it is the
  reference-triangle→physical Jacobian for the Duffy map) and was kept, with
  a comment distinguishing it from the weights-sum-to-1 rule (`:101-118`).
- `ddgclib/analytical/_integrated_comparison.py:157-175`
  (`_dual_cell_pressure_integral_2d_simple`, the production path used by
  `volume_averaged_scalar` / `integrated_pressure_error` /
  `integrated_l2_norm`): removed the spurious `2.0 *` — the Dunavant weights
  from `_triangle_quadrature_points` sum to 1, so `∫_T P dA ≈ Area * Σ w P`.
  Pre-fix, every 2D dual-cell integral was exactly doubled:
  `volume_averaged_scalar(1) == 2.0`, the `HydrostaticPressure` /
  `LinearPressureGradient` ICs assigned 2× pressure whenever duals were
  present, and `integrated_pressure_error` compared against a doubled
  `∫P dV` (inverting good and bad fields).
- `ddgclib/analytical/_divergence_theorem.py:279-281` — stale docstring
  fixed: `∫_T f dA ≈ Area * Σ w_i f(x_i)` (was claiming a `2*Area` rule).
  **Docstring-only**: the code at `:353-358` / `:376-382` already used
  `w * f * n_vec * 0.5` with `|n_vec| = 2*Area`, i.e. the correct
  weights-sum-to-1 convention.  No behavioural change in this file.
- Regression tests: `ddgclib/tests/test_integrated_validation.py:983-1087`,
  NEW class `TestDualCellVolumeAverage2D` (5 tests):
  - `test_volume_averaged_constant_is_one` — the requested
    `volume_averaged_scalar(f≡1) == 1.0` on a mesh WITH duals
    (`compute_vd` + `cache_dual_volumes`), atol 1e-12 (was exactly 2.0).
  - `test_volume_averaged_constant_is_one_jittered` — same on a jittered
    re-Delaunay mesh (no symmetry cancellation).
  - `test_duffy_variant_matches_shoelace_area` — both integral variants
    reproduce the exact shoelace polygon area (catches BOTH the factor 2
    and the `ti+tj>1` skip, rtol 1e-12).
  - `test_hydrostatic_ic_after_duals_not_doubled` — the production-reachable
    IC path (`initial_conditions.py:104-106,142-144`) assigns `<P>`, not
    `2<P>` (this was the mean `a_y = +g` spurious-acceleration mechanism in
    the audit).
  - `test_integrated_pressure_error_zero_for_consistent_field` — the
    sanctioned metric scores ~0 (< 1e-12) for a self-consistent
    volume-averaged linear field (pre-fix it scored O(∫P dV)).

#### Existing tests with the factor 2 baked in: NONE
Grep across `ddgclib/tests/` shows no pre-existing test imports
`volume_averaged_scalar`, `integrated_pressure_error`, `integrated_l2_norm`
or the private `_dual_cell_pressure_integral_2d*` helpers (only the new
regression class does).  Tests using `HydrostaticPressure`
(`test_initial_conditions.py`, `test_stress.py`) apply it on meshes whose
dual volumes are absent at IC time, hitting the unaffected point-wise
fallback.  Consistently, the fast suite is green after the fix with **zero**
expectation changes outside the new class — nothing had baked the doubling
in, so no test "corrections to the correct physics" were needed.

#### Integrated benchmarks linear check (required probe)
`/home/endres/anaconda3/envs/ddg/bin/python benchmarks/run_integrated_benchmarks.py --linear-only`:
- **PASS at machine precision (2.2e-16..7.6e-16) for barycentric/p_ij — the
  production method — on every row**: {LinearScalar, LinearVector} ×
  {1D, 2D} × {symmetric, jittered(seed=42)}.
- FAIL rows (runner exits "Some linear precision checks FAILED"):
  `bary` integration-polygon variant err 5.56e-2..1.19e-1 **even on
  symmetric meshes**, and circumcentric duals err 2.71e-3..5.74e-3 on
  jittered meshes.  These are PRE-EXISTING method floors, not lane-7 (or
  session) regressions.  Proof: (a) nothing in `benchmarks/` imports
  `_integrated_comparison.py`, and the `_divergence_theorem.py` edit was
  docstring-only, so lane 7 cannot move any benchmark number; (b) symmetric
  meshes never build a simplex cache (`_integrated_benchmark_classes.py:137-161`;
  cache only in `_jitter_mesh` `:214-216`), and circumcentric paths are
  gated out of lane-3's exact-volume rewire (`_vd_method == 'barycentric'`
  required), so every FAIL row runs code untouched by lanes 1–5 as well;
  (c) the `bary` failure is jitter-independent (1.08e-1 symmetric vs
  1.09e-1 jittered) — a property of the method, not the mesh pipeline.
- Cleanup: `docs_temp/05_test_cases_and_validation.md:101` claimed the
  <1e-13 gate held "for ALL linear fields" — corrected to scope the gate to
  barycentric/p_ij and record the measured pre-existing floors of the
  non-production variants.

### (T3) Fast oscillation regression pytest
- `ddgclib/tests/test_case_oscillating_droplet.py:506-629`, NEW class
  `TestOscillationEnvelopeRegression2D` (NOT slow-marked; ~7 s measured).
  Mirrors `oscillating_droplet_2D.py` exactly — same `setup_oscillating_droplet`
  params from `src/_params.py`, same lane-5 `dual_only` rebinding
  (`partial(retopo_fn, skip_triangulation=True)`), same CFL dt formula, same
  `oscillation_score` — at reduced refinement 2/2 (95 verts / 16 iface, 267
  steps) over the FULL production duration `t_end = min(t_end_2d, 5/beta)
  = 0.1143 s`.  Note on "≥ 1 full period": the l=2 mode is overdamped
  (beta=25 1/s > omega=12.91 rad/s), so no literal oscillation period
  exists; the test integrates the entire production decay envelope (KE peak
  at t=0.0386 s falls safely in the first half, making the half-split
  `tail_growth` metric robust on this fixture — unlike the refine-3/3
  fragility flagged in the lane-5 log).
  Pinned post-lane-5 metrics (measured on this fixture, deterministic):
  `l2_error_normalized = 0.050006441230559064`,
  `tail_growth = 0.9130549583972877`, linf 0.08490720480839112,
  mass_drift 4.6e-15.  Thresholds with ~10% headroom:
  `L2_MAX = 0.0550`, `TAIL_MAX = 1.004`, mass_drift < 1e-10.
  A/B sensitivity recorded in the docstring: reverting only the retopo
  policy to per-step Delaunay (pre-lane-5 state) scores l2 1.384569 /
  tail 2.110904 / KE_max 9.93e-2 J on this exact fixture, so the pins catch
  that regression class by >25x.

### (T4) Pinned floor tests re-verified — no re-pin needed
`pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`:
**14 passed in 18.92 s** (12 pre-lane-7 tests + envelope regression +
companion; includes both slow floors).  Constants confirmed against reality:
- 2D `TestStaticDroplet2DRetopologyFloor`: `EXPECTED_STEP0_MAXF =
  2.3748568e-03`, `EXPECTED_STEP1_MAXF = 2.2716938e-03` — identical to the
  session baselines (floor_frozen_2d 0.0023749, floor_postretopo_2d
  0.0022717).  No lane moved them.
- 3D `TestStaticDroplet3DRetopologyFloor`: 6.0153e-05 / 7.3768e-05 — lane 3
  measured that the exact-3D-volume switch would RAISE the plateau to
  7.616854e-05 (+3.26%) and correctly backed it out per the
  re-pin-only-if-lower rule; the comment trail (docstring + lane-3
  NOTE(lane3-dual-volume) markers in `stress.py` /
  `_integrators_dynamic.py`) is accurate and self-consistent: constants
  unchanged, conditional re-pin value documented for a future
  all-three-switch-points lane.
- Lane 5's re-pin was `cases_dynamic/oscillating_droplet/baselines/
  baseline_oscillation.json` (case baseline, old values documented in the
  lane-5 log), not a floor-test constant — trail checked, consistent.
- The lane-2 `TaitMurnaghan rho_clip` RuntimeWarning fires once in the 3D
  floor test — intentional (see lane-2 log §6), do not silence.

## 2. Measurement battery (session-closing measurement)

cwd `/home/endres/projects/ddgclib`, `/home/endres/anaconda3/envs/ddg/bin/python`:

1. Pinned floor tests (`-m ""`): **14 passed, 1 warning in 18.92 s** (0 failed).
2. Fast suite (`-m "not slow" -q`): **833 passed, 12 skipped, 17 deselected,
   3 xfailed, 5 warnings in 56.60 s** — ZERO failures.  Reconciliation:
   825 post-lane-5 passed + 1 (stale `test_raises_unsupported_dim` fixed:
   fail→pass) + 1 (`test_mismatched_simplex_entries_skipped`) +
   5 (`TestDualCellVolumeAverage2D`) + 1 (`TestOscillationEnvelopeRegression2D`)
   = 833.  First fully-green fast suite of the session (baseline: 796 passed
   + 1 pre-existing failure).
3. Equilibrium: `summary = 0.0011847162859108737`
   (interface_radius_drift 1.1847e-03, max_KE_normalized
   7.673878163142909e-09, mass_drift 0.0) — bit-identical to post-lane-3/4/5;
   vs 0.0011636 pre-session baseline: +1.81 %, inside the 5 % band (lane-3
   documented drift-metric sensitivity; unchanged since).
4. Oscillation: `l2_error_normalized = 0.1785660454150319`,
   `tail_growth = 0.9992507831101141`,
   `linf_error_normalized = 0.32842491275938274`,
   `summary = 0.1785660454150319`, mass_drift 2.676988154993443e-14,
   n_frames 207, t_end 0.11432433142499636 — ALL bit-identical to
   post-lane-5.  Lane 7 changed no production behaviour, as intended
   (validation utilities + tests + docs only; `volume_averaged_scalar` is
   not on the droplet path).

Score copies:
`scratchpad/wf3/lane7-cleanup-regression-lockin_osc_score.json`,
`scratchpad/wf3/lane7-cleanup-regression-lockin_equil_score.json`.

hyperct: NOT modified by lane 7; suite not re-run (last verified post-lane-4:
334 passed, 40 skipped, 6 xfailed + pre-existing pytest-benchmark fixture
errors).

## 3. Session-closing metric trail (baseline → final)

| metric | baseline | L1 | L3 | L5 | L7 (final) | Δ |
|---|---|---|---|---|---|---|
| osc l2 | 2.9519 | 0.70907 | 0.48992 | 0.1785660454150319 | 0.1785660454150319 | −94.0 % |
| osc tail | 2.3530 | 2.20527 | 1.72505 | 0.9992507831101141 | 0.9992507831101141 | −57.5 % |
| osc linf | 5.9145 | 1.05384 | 0.85035 | 0.32842491275938274 | 0.32842491275938274 | −94.4 % |
| equil | 0.0011636 | 0.0011351 | 0.0011847 | 0.0011847162859108737 | 0.0011847162859108737 | +1.81 % |
| fast suite | 796 P / 1 F | 804 P / 1 F | 824 P / 1 F | 825 P / 1 F | **833 P / 0 F** | green |

(L2 = bit-identical to L1; L4 = bit-identical to L3.)

## 4. Notes for future sessions

- The factor-2 fix changes reported numbers of any 2D case that assigns ICs
  AFTER computing duals (`HydrostaticPressure`/`LinearPressureGradient` via
  `volume_averaged_scalar`) or that reports `integrated_pressure_error` /
  `integrated_l2_norm` — old logged values from such runs are 2× off on the
  analytical side.  Re-baseline hydrostatic-style case scores before
  comparing against pre-2026-07-02 artifacts.
- The `--linear-only` runner's exit banner ("Some linear precision checks
  FAILED") is expected: only barycentric/p_ij carries the <1e-13 gate; see
  the corrected `05_test_cases_and_validation.md:101` for the measured
  pre-existing floors of `bary`/circumcentric variants.
- `TestOscillationEnvelopeRegression2D` pins post-lane-5 dynamics on a fast
  fixture; update `L2_MAX`/`TAIL_MAX` ONLY for genuine improvements
  (document old → new in the class docstring, keeping the ~10 % headroom
  rule).
- Open items this lane deliberately did NOT touch: tag-consistency flag 19
  (lane 1 §6), the 3D exact-volume three-point switch (lane 3 §2), the
  missing 3D dynamic score harness (06 §1.6), ALE/remap for
  large-deformation retopo (lane 5 §8).
