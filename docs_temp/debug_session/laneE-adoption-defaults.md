# laneE-adoption-defaults — long-run proofs + production default adoption

Date: 2026-07-29.  Lane E of the wf5 session (continuation of the wf4
four-lane session; prerequisite state: laneA-D all landed and committed).
Task: turn the validated retopo-policy candidates into production
defaults with long-run proof — 2D `dual_only` vs lane-D
`delaunay+remap`, 3D `dual_only` vs `delaunay+remap` — then apply the
decided defaults and re-pin affected regressions.

Drivers + raw JSONs preserved in scratchpad `wf5/`
(`static_endurance.py`, `dyn2d.py`, `dyn3d.py`, `probe_clip2d.py`,
`probe_clip3d.py`, `envelope_fixture_remap.py`; results
`static_*.json`, `dyn2d_*.json`, `dyn3d_*.json`, `run_*.log`).
All drivers were validated by bit-exact reproduction of pinned numbers
before trusting new measurements (2D dual_only l2 0.1785660454150319 /
tail 0.9992507831101141 / mass 2.676988154993443e-14; 2D remap l2
0.17479361640597058 / tail 0.9998967874595965 / KE_max
8.31727774219008e-07 = laneD; 3D dual_only l2 0.24811340819647862 /
mass 1.905002320272536e-14 = laneB baseline).

## Verdict up front

**2D default FLIPPED `dual_only` -> `delaunay_remap`** (per-step full
Delaunay + lane-D `retopo_remap='conservative'`): every long-run proof
clean, metrics at least as good (l2 0.17479 vs 0.17857), and it keeps
retopology honestly ACTIVE — the library's stated goal is
complex/changing topologies, which `dual_only` structurally cannot
serve.  Cost 1.61x wall.  Re-pins:
`baselines/baseline_oscillation.json`, envelope-regression fixture
pins; new 200-step endurance guard test.

**3D default KEPT `dual_only`**: `delaunay_remap` measured WORSE on the
trajectory metric (l2 1.87348 vs 0.24811, fails the better-l2-AND-tail
flip rule) despite suppressing the KE pump; the laneD §2.6.5 3D
bookkeeping caveat is real (droplet `vol_corr` gauge 0.9696 in 20
steps; EOS pressure clips on the Delaunay path).  No file changes
needed (laneB already flipped 3D to dual_only); A/B evidence recorded
in the `_params.py` comment.

## 1. Long-run proofs (task 1)

### 1a. 2D delaunay+remap static endurance — PASS

A.5.b harness (u=0 forced every step, per-step FULL Delaunay, remap ON,
redistribute_mass=True, refine 3/3, 311 verts / 32 iface), 2000 steps
(`static_remap2d.json`, wall 231.0 s):

| requirement | measured |
|---|---|
| step0 max\|F\| | 2.374856801157633e-03 — pinned 2D floor bit-identical |
| plateau (steps 1..2000) | mean 2.271694172414664e-03 = pinned post-retopo floor 2.2716938e-03 (+1.6e-7 rel) |
| no drift: rel spread < 1e-8 | **1.3363445308012518e-15** |
| mass machine-precision | **2.3514085145212675e-15** |
| interface count constant | **32/32**; verts 311/311; total dual-vol spread 0.0 |

### 1b. 2D delaunay+remap dynamic endurance (2x horizon) — PASS

Runner-replica at 2x the standard horizon: 3678 steps, t_end 0.228649 s
= 10 damping times (`dyn2d_remap_x2.json`, wall 410.5 s):

| requirement | measured |
|---|---|
| KE on/below envelope tail | KE_max 8.31727774219008e-07 J @ t 0.056012 (bit-identical to the standard-horizon run — the peak is in the first half); KE_final 1.2976060041052706e-07, KE_final/KE_max 0.15601; tail decay-rate fit over the second half 13.52 1/s vs analytical 2·lambda_slow = 7.183 — the sim decays FASTER than (i.e. stays below) the envelope; this is the known laneC solver-side over-decay, not growth |
| no late-time growth | **late_growth_ratio = 1.0** (max KE over last quarter / KE at the 3/4 point); tail_growth 0.7063635532645631 |
| mass clean | **4.521939451002437e-14** |
| bookkeeping | n_iface 32 constant |

l2 over the DOUBLED horizon is 0.300394277446834 (vs 0.17479 standard):
the known smooth monotone over-decay error (lane5 §7 / laneC) integrates
with time, so a longer horizon inflates l2 by construction.  This is a
reference-tracking property, not instability — every stability channel
above is clean.  Do not compare 2x-horizon l2 against the pinned
standard-horizon baseline.

### 1c. 3D dual_only static endurance — PASS

Same harness, dim=3 refine 2/2 (472 verts / 98 iface),
skip_triangulation=True, 2000 steps (`static_dual_only3d.json`, wall
1030.6 s):

| requirement | measured |
|---|---|
| step0 max\|F\| | 6.0153201139652905e-05 — pinned 3D step0 bit-identical |
| own plateau | **6.836475840821090e-05** (steps 1..2000) — 6.0% BELOW the pinned per-step-Delaunay plateau 7.274172e-05 |
| stability comparable to pinned plateau | rel spread **8.960380197595157e-14** over 2000 steps (laneA measured 6.05e-12 over 100 steps on the Delaunay path) |
| mass / interface | **8.912291556830577e-16** / 98 constant, verts 472 constant, dual-vol spread 0.0 |

### 1d. 3D delaunay+remap — dim-agnostic YES; measured and REJECTED

**Dim-agnosticism (code reading)**: `_retopologize_multiphase`'s remap
stages and all three helpers (`restore_pressure_multiphase`,
`phase_volume_totals`, `anchor_phase_pressure_levels`,
`redistribute_mass_multiphase`) operate on per-vertex
`p_phase`/`m_phase`/`dual_vol_phase` with no dim branching; `dim` is
only forwarded to `_retopologize`.  The stage-1 measurement pass is the
3D skip-triangulation path laneB verified bookkeeping-clean.  So the
scored 3D A/B and static endurance were run.

**3D static endurance (2000 steps, remap ON)**: clean — plateau
6.829963052584747e-05 (rel spread 5.00e-14), mass 4.121934845034142e-14,
iface 98 constant (`static_remap3d.json`, wall 1858.8 s).  laneA's
canonical-order fix makes 3D static-cloud Delaunay idempotent, so
statically the remap has almost nothing to remap.

**3D scored oscillation A/B (872 steps, refine 2/2)**:

| metric | dual_only (default, rerun) | delaunay_remap | plain delaunay (laneB) |
|---|---|---|---|
| l2_error_normalized | **0.24811340819647862** | 1.8734809234034775 | 1.5244561707801316 |
| linf | 0.5992701359956998 | 2.6033119321282814 | 2.265475988110048 |
| tail_growth | 0.08409976059818802 | 0.2532690179972299 | 0.47066568050584034 |
| mass_drift | 1.905002320272536e-14 | 4.2333384894945245e-14 | 3.87e-14 |
| R_max_peak | 0.010790236105250779 | 0.01155883605820377 | 0.011397 |
| KE_max [J] / t@peak | 1.7431572435348644e-06 / 0.00606 | **6.136872043516315e-08** / 0.02074 | — |
| dual_vol_drift_post | 1.0684e-04 | 5.0761e-03 | 5.08e-03 |
| n_interface / saturation | 98 constant / False | 98 constant / False | 98 constant / False |
| wall [s] | 426.3 | 773.5 | — |

Reading: the remap DOES kill the 3D Delaunay KE pump (KE_max 28x below
dual_only's), but the R_max trajectory inflates MORE than plain
Delaunay — the remap pins the per-phase pressure levels while the
droplet slowly inflates through the known curvature-side physics gap
(laneB: ~0.8% of R0 under dual_only), so the inflation runs less
opposed and l2 lands at 1.87.  Fails the flip rule (better l2 AND
tail).  **Keep `dual_only` in 3D.**  Do NOT re-try without first
closing the 3D inflation gap (debugging_plan next-lane item 3).

### Side-channel probes (recorded, no action this lane)

EOS `TaitMurnaghan.pressure()` rho_clip engagement (band = ±20% around
the reference density; clips counted per phase over the run):

| run | outer clips | droplet clips | vol_corr end |
|---|---|---|---|
| 2D dual_only refine 2/2 (267 steps) | 0 | 0 | [1.0, 1.0] |
| 2D remap refine 2/2 (267 steps) | 2410 | 10 | [1.00750, 0.99967] |
| 2D remap refine 3/3 (1839 steps) | 57197 | 3 | [1.01113, 0.99962] |
| 3D delaunay (20 steps) | 112 | 32 | [1.0, 1.0] |
| 3D dual_only (20 steps) | 0 | 8 | [1.0, 1.0] |
| 3D remap (20 steps) | 118 | 40 | [0.99970, 0.96956] |

The clips fire inside `compute_phase_pressures` on churned
corner/boundary dual cells right after a Delaunay rebuild — i.e. on the
transient pre-restore evaluation; `restore_pressure_multiphase` then
overwrites those values for every persistent (vertex, phase) entry, and
all outcome channels (KE envelope, l2, mass, floors) are clean.  It is
a pre-existing property of the per-step-Delaunay path (3D plain
Delaunay clips too; the old 3D default shipped with it), made visible
in 2D by the flip.  Cosmetic cost: the warn-once RuntimeWarning now
appears in default 2D runs.  Flag for a future lane if clip telemetry
is wanted as a health channel.

## 2. Decision rationale (task 2)

**2D — flip to `delaunay_remap`.**

- Long-run cleanliness (criterion 1): 1a and 1b both pass with
  machine-precision mass and zero drift/growth.
- Metrics (criterion 2): l2 0.17479361640597058 vs 0.1785660454150319
  (marginally better), tail 0.9998967874595965 vs 0.9992507831101141
  (both healthy; the lane-5 log already flagged tail's ±0.002
  metric-fragility around the half-split), KE envelope physical in
  both (peak t 0.0560 vs analytical 0.0598).
- Wall-time (criterion 3): 205.8 s vs 127.5 s full run = **1.61x** —
  acceptable for a case runner.
- GENERALITY (criterion 4, decisive): `delaunay_remap` keeps global
  reconnection active every step, so the benchmark now exercises the
  code path that large-deformation / changing-topology cases (dam
  break, detaching bubbles) MUST use; `dual_only` freezes builder
  connectivity and cannot generalize.  Given (1)-(3) are at worst
  neutral, the more honest and more general configuration wins the
  default.  `dual_only` stays available as an opt-in policy value.

**3D — keep `dual_only`.**  The flip rule required better l2 AND tail,
mass <= 1e-12, constant interface, floors unaffected.  Measured: l2
1.87348 vs 0.24811 — immediate fail (mass 4.23e-14 and interface 98/98
would have passed).  The pinned 3D floors exercise setup's own
per-step-Delaunay retopo_fn and are policy-independent (re-verified
green in the battery below).

## 3. Applied changes (task 3)

1. `cases_dynamic/oscillating_droplet/src/_params.py` —
   `retopo_policy_2d = 'dual_only'` -> `'delaunay_remap'` with the
   lane-E measurement table + rationale + clip caveat in the comment;
   `retopo_policy_3d` comment records the 3D A/B rejection evidence
   (value unchanged `'dual_only'`).
2. `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` —
   policy dispatch gains the `'delaunay_remap'` branch
   (`retopo_fn = partial(retopo_fn, retopo_remap='conservative')`);
   comment updated.
3. `ddgclib/tests/test_case_oscillating_droplet.py` —
   - `TestOscillationEnvelopeRegression2D` re-pinned to mirror the new
     default (rebinding dual_only -> remap).  Fixture measurements
     (95 verts / 16 iface, 267 steps) old -> new:
     l2 0.050006441230559064 -> 0.05451425932968201,
     tail 0.9130549583972877 -> 0.9368903708926931,
     linf 0.08490720480839112 -> 0.08367647560330455,
     mass 4.6e-15 -> 4.1e-15, KE_max 1.4609567009656418e-06 ->
     1.4214765248811987e-06.  Thresholds: `L2_MAX` 0.0550 -> 0.0600
     (~10% headroom), `TAIL_MAX` kept 1.004 (~7% headroom).
   - NEW `TestDelaunayRemapEndurance2D` (200-step remap-ON
     full-Delaunay endurance guard, ~6 s): bounded KE on every
     recorded sample, machine mass, preserved interface/vertex sets,
     Delaunay actually rewiring.  Mirrors `TestDualOnlyRetopoPolicy2D`
     at 5x its horizon.
   - `TestDualOnlyRetopoPolicy2D` / `TestConservativeRetopoRemap2D` /
     all pinned floor tests untouched.
4. `cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json`
   — re-pinned to the new-default official run.  Old -> new:
   l2 0.1785660454150319 -> 0.17479361640597058,
   linf 0.32842491275938274 -> 0.32364245955409165,
   tail 0.9992507831101141 -> 0.9998967874595965,
   mass 2.676988154993443e-14 -> 2.4056717879332966e-14,
   summary 0.1785660454150319 -> 0.17479361640597058
   (inputs beta 25.0 / omega 12.909944487358056 unchanged).
5. No 3D file changes (decision = keep); no solver/operator changes at
   all this lane — the remap machinery shipped in laneD.

## 4. Dam break (task 4, optional) — cited, not re-run

The solver is bit-identical to the laneD code state (this lane touched
only params/runner/tests/baseline), and `dam_break_2D.py` does not read
`retopo_policy_2d` — so laneD §2.3's smoke A/B remains the current
measurement: at the shipped smoke horizon the collapse is stalled and
Delaunay never actually rewires, so remap ON/OFF cannot be
discriminated there; alpha_art 2.0 -> 0.5 is stable both ways (KE ~15x
higher, bounded, zero clips) with **no evidence yet that the remap
allows shrinking the artificial-viscosity crutch, and none against** —
the crutch addresses the corner-vertex force defect, orthogonal to
retopo churn.  Re-test once the dead `skip_triangulation` flag is fixed
and the collapse actually runs (debugging_plan next-lane item 2).  No
dam-break defaults changed.

## 5. Final measurement battery (task 5; ddg env, repo root)

1. Floor battery `pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`:
   **18 passed in 24.63s** (17 + 1 new `TestDelaunayRemapEndurance2D`);
   all four pinned floors untouched (2.3748568e-03 / 2.2716938e-03 /
   6.0153e-05 / 7.274172e-05).
2. Fast suite `pytest ddgclib/tests/ -m "not slow" -q`:
   **866 passed, 12 skipped, 17 deselected, 2 xfailed in 62.04s — 0
   failures** (= laneD 865 + 1 new).
3. `static_droplet_2D.py`: summary **1.1847162859108737e-03**
   (interface_radius_drift 1.1847162859108737e-03, mass_drift 0.0) —
   pinned channels bit-identical; max_KE_normalized 7.673878163142909e-09
   (= the 7.6739e-09 quoted by lanes B/D at their 4-digit precision).
4. `oscillating_droplet_2D.py` (NEW default delaunay_remap):
   l2 **0.17479361640597058** / linf **0.32364245955409165** / tail
   **0.9998967874595965** / mass **2.4056717879332966e-14** / summary
   0.17479361640597058 — bit-identical to the laneD full-run and to
   this lane's driver replica; score re-pinned as the baseline.
5. `oscillating_droplet_3D.py` (default dual_only, scored):
   l2 **0.24811340819647862** / linf 0.5992701359956998 / tail
   0.08409976059818802 / mass 1.905002320272536e-14 / R_max_peak
   0.010790236105250779 / boundary_saturation False — bit-identical to
   `baselines/baseline_oscillation_3d.json` (no re-pin needed).
6. hyperct suite (`pytest hyperct/tests -q`, sibling repo): **336
   passed, 40 skipped, 6 xfailed, 39 errors in 3.78s** — identical to
   laneA (the 39 are the pre-existing benchmark-fixture errors).
7. a5b long-run regression
   `pytest ddgclib/tests/test_a5b_longrun_regression.py -v -m ""`:
   **2 passed in 13.94s** (2D + slow 3D; pinned floors 2.3749e-03 /
   2.2717e-03 / 7.274172e-05 all held).

## 6. Caveats / guidance for future lanes

1. **2x-horizon l2 is not a regression channel**: 0.3004 at 10 damping
   times vs 0.17479 at 5 is the known over-decay integrating; pin
   baselines at the standard horizon only.
2. **EOS clip telemetry under the new 2D default** (§1 probes): ~5.7e4
   transient outer-phase pressure clips per full run, overwritten by
   the restore; outcome channels clean.  If a future lane wants
   clip_count as a health tripwire, it must first split pre-restore
   (bookkeeping) clips from genuine ones.
3. **Do not apply the remap in 3D** until the droplet-inflation gap is
   closed: it un-opposes the inflation (l2 1.87) even though it kills
   the KE pump (6.1e-8 J).  After the curvature-side fix lands, re-run
   the 1d A/B — the KE result suggests the remap may then win.
4. The 3D dual_only static plateau **6.836475840821e-05** (this lane,
   2000 steps, machine-flat) is a NEW reference number for the frozen-
   connectivity path; the pinned 7.274172e-05 floor remains the
   per-step-Delaunay setup-path number and is unaffected.
5. `dual_only` remains fully supported in 2D (`retopo_policy_2d =
   'dual_only'`); `TestDualOnlyRetopoPolicy2D` still guards that path.
