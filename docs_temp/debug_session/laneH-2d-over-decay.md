# laneH-2d-over-decay — attribute and reduce the 2D over-decay residual (l2 0.17479)

Date: 2026-07-30.  Lane H of the wf6 session (next-prompt option 4; the
reference-error escape hatch was CLOSED by laneC — solver-side lane).
Drivers + raw JSONs + logs in scratchpad `wf6/laneH/` (`driver.py`,
`analyze_*.py`, `run_*.json`, `run_*.log`).  The instrumented driver
was validated by bit-exact reproduction of BOTH pinned configs before
any new measurement was trusted (`std_remap` l2
**0.17479361640597058** / tail 0.9998967874595965 / mass
2.4056717879332966e-14; `dual_N1` l2 **0.1785660454150319** / tail
0.99925 / mass 2.677e-14 — all to 16 digits), and its energy budget
closes to 9e-22 J/step with force-decomposition residual < 9e-18 N.

## Verdict up front

**ATTRIBUTED AND FIXED (opt-in); NO DEFAULT FLIP.**  The ~2.5x
amplitude over-decay behind the pinned l2 0.17479 is the **every-call
pressure-preserving mass redistribution erasing each step's local EOS
compression response** (lane-5/laneC suspect (i)) — NOT the O(h)
32-gon curvature bias (suspect (ii)), and NOT an energy leak in the
projection's KE bookkeeping.  With the new opt-in
`projection_every=N` cadence (`_retopologize_multiphase`), the SAME
default configuration (per-step full Delaunay + conservative remap,
reconnection active every step) scores:

| config (full 1839-step run) | l2 (pinned single-fluid metric) | l2 vs exact two-fluid (laneC) | tail | KE_max [J] | mass |
|---|---|---|---|---|---|
| delaunay_remap (pinned default) | **0.17479361640597058** | 0.18461 | 0.99990 | 8.32e-07 @ t=0.056 | 2.4e-14 |
| delaunay_remap + projection_every=2 | **0.03795682994323827** (−78%) | **0.02193** | 1.3973 | 1.15e-06 @ t_end | 1.6e-14 |
| + projection_every=3 / 5 / 20 | 0.04002 / 0.04108 / 0.04194 | 0.02410 / 0.02523 / 0.02614 | 1.39 / 1.38 / 1.39 | 1.15e-06 @ t_end | ≤ 3.7e-14 |

The fixed-cadence solver lands on laneC's **exact two-fluid
reference to ~2% of the perturbation** (l2_2f 0.0219, linf_2f 0.0288)
and its KE(t) has correlation **0.994** with the exact two-fluid
started-from-rest shape `e^{-2βt} sin²(ω_d t)` (s = −6.8326 ±
5.4707i).  The WIN condition "l2 below 0.17479 with tail <= 1.0" is
met on l2 by 4.2-4.6x and **measured unattainable on tail for any
faithful run**: the exact two-fluid KE peaks at t* = arctan(ω_d/β)/ω_d
= 0.1234 s, BEYOND t_end = 0.1143 s, so the true physics has KE rising
through the second half (analytic tail on this horizon = 1.66; sim
1.38-1.40).  The pinned default passes tail <= 1.0 only because the
every-call projection reshapes KE onto the single-fluid overdamped
envelope.  By the standard adoption rule (better l2 AND tail, clean
channels) no candidate qualifies → default, baselines, and every pin
unchanged; the fix ships as a tested opt-in.

## 1. Attribution (task step 1 — all measured before any fix)

### 1.0 Post-hoc structure of the pinned run (free, from `results/diag_series.json`)

The apex error is TWO superimposed artifacts, not one (the 2D sibling
of laneG's 3D cancellation):

- **l=2 over-decay**: amplitude `A_l2 = (R_max−R_min)/(2εR0)` decays
  SUPRA-exponentially — instantaneous rate 2.5 → 16.1 1/s across the
  run vs the single-fluid reference's 1.6 → 3.5 (ratio grows to ~4.3x)
  — ending at A_l2 = 0.286 vs reference 0.718.
- **l=0 bump**: the mean interface radius rises to +0.59% R0 by
  t≈0.06 then drifts slowly down (+0.54% at t_end) — at the apex this
  adds +0.117 in ε-units, CANCELLING part of the l=2 deficit (apex
  error −0.324 instead of −0.43).

### 1.1 (a) Energy budget per step (instrumented std_remap replica)

Per-step decomposition dKE = W_pressure + W_viscous + W_st + T_trunc
+ dKE_redist, closure ≤ 8e-22 J:

| cumulative term (full run) | value [J] | share |
|---|---|---|
| E_surf release γ·(L0−L_end) | +6.8533e-06 | source |
| W_st (surface-tension work) | +6.8494e-06 | = −ΔE_surf to **0.06%** (FTC force is the exact PL-perimeter gradient) |
| W_viscous | −6.4869e-06 | 95% sink |
| W_pressure | +2.2355e-07 | net source +3% |
| **dKE_redist (projection KE jump)** | **−2.2299e-09** | **0.03% — NOT the sink** |
| T_trunc (integrator O(dt²)) | +3.0066e-09 | 0.04% |

The projection does NOT dissipate through its mass/KE bookkeeping
(the task's "restore per-vertex velocity-weighted KE across
redistribution" candidate is measured a non-lever, 3 orders too
small).  The non-physical loss is in the FORCING: with the pressure
structure erased every call (laneD §1.1: under `redistribute_mass=True`
the structure cannot evolve; only the level moves), the interface
relaxes without local elastic push-back, releasing surface energy
~1.6x faster than the fixed-cadence run (6.85e-6 vs 4.30e-6 J) with
W_viscous draining the excess.

### 1.2 (b) Redistribution-frequency probe — suspect (i) CONFIRMED by dose-response

Under `delaunay_remap` the probe is **structurally incoherent as
posed**: the conservative remap requires `redistribute_mass=True`
(code-enforced ValueError), and skipping the redistribution while
reconnection fires is exactly the laneD "restore-only" measured dead
end (bounce corr +0.9998, DO-NOT).  The coherent probe is (1) under
`dual_only` (no reconnection): skip the redistribution on off-cadence
steps; and (2) under the remap: keep the remap neutrality every call
but move the erasure to a cadence (§2).  dual_only results (driver
toggling ≡ shipped `projection_every` verified bit-identical at full
horizon, l2 equal to 16 digits):

| dual_only cadence | l2 | tail | KE_max [J] | A_l2 windowed rates vs ref [1/s] |
|---|---|---|---|---|
| N=1 (pinned dual_only) | 0.17857 | 0.99925 | 8.32e-07 @ 0.055 | 5.4 / 10.7 / 12.5 / 15.2 vs 1.6 / 3.0 / 3.4 / 3.5 |
| N=2 | 0.03107 | 0.02504 | 7.40e-05 @ 0.044 | 1.68 / 4.01 / 5.02 / 4.94 |
| N=3 | 0.03094 | 0.09796 | 1.91e-05 @ 0.045 | 1.59 / 3.89 / 5.48 / 6.27 |
| N=5 | 0.036322506487598424 | 1.44875 | 1.16e-06 @ t_end | 1.56 / 3.79 / 5.31 / 6.63 |
| N=20 | 0.03642 | 1.45281 | 1.17e-06 @ t_end | 1.53 / 3.75 / 5.26 / 6.71 |
| noredist (N=∞) | 0.49503 | 0.11132 | 7.69e-04 @ 0.039 | ringing (+33 / −58 / −4.6 / +1.7) |

Over-decay collapses as soon as the erasure is not every-step
(saturated by N=5); the late-window rate lands at 6.6-6.7 1/s — the
exact two-fluid amplitude rate is 6.83 (laneC).  Suspect (i)
confirmed with a clean dose-response.

**KE-channel caveat**: N=2/3 pass tail <= 1.0 only via a marginally
damped ACOUSTIC channel — cumulative T_trunc 7.2e-5 / 3.9e-5 J
(vs 4.2e-9 at N=5), KE_max 64x/16x the mode level, decaying.  The
compressible DOF is live between projections and the symplectic-Euler
truncation pumps it; 1-in-2/1-in-3 projection is the only damper.
Not "clean channels".

### 1.3 (c) The end-members

- `dual_only_noredist` reproduces lane-5 exactly (l2 0.49503, tail
  0.11132, mass 0.0 exact): the launch acoustic transient rings at
  KE 7.7e-4 J (~660x mode level), pumped by T_trunc (+8.4e-4 J
  cumulative) with W_p +1.5e-3 J feeding it — the reason "just turn
  redistribution off" was never the fix.
- `delaunay_remap_noredist` is **structurally forbidden**:
  `retopo_remap='conservative'` raises ValueError without
  `redistribute_mass=True`, and its only workaround is the laneD
  restore-only dead end.  Said so; not run.

### 1.4 Suspect (ii) — 32-gon curvature bias: NOT the l=2 driver, bounded

With the cadence fix in place the trajectory matches the exact
two-fluid reference to l2_2f 0.0219 — there is no room for an O(h)
force bias to be driving the amplitude channel.  The surviving (ii)-
shaped signal is the **l=0 bump** (+0.58-0.66% R0 across ALL
redistribution-bearing configs, insensitive to cadence; a regular
32-gon holding the initial enclosed area sits at vertex radius
+0.32% R0, same order), which contributes ±0.1 ε-units to the apex
metric through cancellation and is the dominant residual inside
l2 ≈ 0.04.  Wiring `reconstruct_arc_length_and_bulge_area`
(curvature_2d.py:184) into the interface balance is deferred as
beyond lane scope: it targets a ≤0.6%-R0 l=0 channel, would re-pin
the 2D static floors, and the A.5 static harness shows the discrete
equilibrium is only ~0.12% R0 from round (static_droplet summary
1.1847e-3) — the bump is mostly the YL-preload l=0 self-correction
riding the polygon-area offset, a mesh-family property (laneG's
recorded 2D sibling).

## 2. The fix (opt-in, task step 2): `projection_every` cadence

`_retopologize_multiphase(..., projection_every: int = 1)`
(`_integrators_dynamic.py`) — cadence of the pressure-structure
projection.  Default 1 = previous behaviour bit-exactly.  Semantics:

- `dual_only` path: off-cadence calls skip the redistribution block
  (mass stays Lagrangian; duals/splits/EOS pressures still refresh).
- `delaunay_remap` path: the remap machinery runs EVERY call
  (reconnection neutrality is not optional), but on off-cadence calls
  the snapshot that redistribution/restore reproduce across the
  rebuild is the pre-call field **advanced by the step's local
  Lagrangian strain** — new helper
  `evolve_snapshot_local_strain(HC, mps, snapshot)`
  (`operators/mass_redistribution.py`):
  `p_new_k = eos_k.pressure(eos_k.density(p_snap_k) * dvp_snap_k /
  dvp_now_k)` per parcel — so only the connectivity artifact is
  projected out, not the compression response.
- bare per-step Delaunay without the remap: `projection_every > 1`
  raises (skipping redistribution under live reconnection re-opens
  the lane-5 KE pump).
- counter on `mps._projection_call_idx`; first call always projects.

**Measured dead end (do not retry)**: building the off-cadence
snapshot from the raw stage-1 `eos(m/dual_vol)` recompute instead of
the strain advance loses the restore/anchor LEVEL corrections (they
live in `p_phase`, not the mass ledger): one off-cadence call at
frozen positions jolts the field by 8.8 Pa on the coarse fixture, and
the full-horizon runs score l2 1.31-1.34 (7.5x WORSE than the pin;
archived `run_remap_proj{5,20}_RAWRECOMPUTE.json`).  The
frozen-positions neutrality unit test now guards this invariant.

Why the remap+cadence variant is the right shape of the fix: unlike
the dual_only cadence it shows **no acoustic contamination at any N**
(T_trunc cumulative 3.6e-9 J for N=2..20, KE smooth on the two-fluid
shape) because the per-call redistribution against the
strain-advanced field keeps m/p consistent every step; and it keeps
global reconnection honestly active (laneE's generality rationale
intact).  Energy ledger of remap_proj2: W_st +4.40e-6 = −ΔE_surf to
0.07%, W_v −3.75e-6, W_p +4.9e-7, dKE_redist +5.3e-9.

## 3. Adoption decision (task step 3) — NO FLIP, rule applied

- remap_proj2 (best): l2 0.03796 ✓ (−78%) but tail 1.3973 ✗.  All
  remap cadences fail better-tail — and §1.2/§Verdict show tail <= 1.0
  is not achievable by faithful physics on this horizon (analytic
  two-fluid tail 1.66; KE peak t* = 0.1234 > t_end).
- dual_L2/L3 pass both numbers (0.031 / tail 0.03-0.10) but fail
  clean-channels (T_trunc-pumped acoustic KE 16-64x mode level) and
  would re-freeze connectivity (reverses laneE's generality
  decision).  Rejected.
- Therefore: default `retopo_policy_2d='delaunay_remap'` unchanged,
  `baseline_oscillation.json` unchanged, every pin bit-identical
  (verified below).  The fix ships as the tested opt-in;
  **recommended next lane**: recalibrate the 2D score (two-fluid
  reference + a tail window aware of the two-fluid KE peak — laneC's
  successor, validation-side), then revisit adoption of
  delaunay_remap + projection_every=2 with the endurance battery.

## 4. Measurement battery (ddg env, repo root)

| item | result |
|---|---|
| floor battery (`test_case_oscillating_droplet.py -m ""`) | **23 passed** in 28.1s (18 + 5 new `TestProjectionCadence2D`); all pinned floors untouched |
| fast suite (`-m "not slow" -q`) | **881 passed, 0 failed**, 12 skipped, 17 deselected, 2 xfailed in 71.8s (= 876 + 5 new) |
| a5b regression (`test_a5b_longrun_regression.py -m ""`) | **2 passed** in 14.1s |
| static_droplet_2D | summary **1.1847e-03** (interface_radius_drift 1.1847e-03, mass 0.0) — bit-identical |
| oscillating_droplet_2D (default) | l2 **0.17479361640597058** / linf 0.32364245955409165 / tail **0.9998967874595965** / mass 2.4056717879332966e-14 — **bit-identical to the pin** |
| oscillating_droplet_3D (dual_only default) | l2 **0.24811340819647862** / R_max_peak 0.010790236105250779 / mass 1.905002320272536e-14 — bit-identical to `baseline_oscillation_3d.json` |
| hyperct (`pytest hyperct/tests -q`) | **336 passed**, 40 skipped, 6 xfailed, 39 pre-existing benchmark-fixture errors — identical (no hyperct edits) |

## 5. Changed files

- `ddgclib/dynamic_integrators/_integrators_dynamic.py` —
  `_retopologize_multiphase` gains `projection_every` (validation,
  cadence counter, strain-advanced off-cadence snapshot for the remap
  path; default path bit-identical).
- `ddgclib/operators/mass_redistribution.py` — new
  `evolve_snapshot_local_strain` helper.
- `ddgclib/tests/test_case_oscillating_droplet.py` — new
  `TestProjectionCadence2D` (5 tests: validation errors, default
  bit-identity, frozen-positions off-cadence neutrality, dual_only
  cadence, remap cadence under full Delaunay).
- `cases_dynamic/oscillating_droplet/src/_params.py` — comment block
  recording the attribution + opt-in (no policy change).
- This log; dated entry at the top of `debugging_plan.md`.
- No baseline, runner-behaviour, or hyperct changes.

## 6. Guidance for future lanes (measured, binding)

1. **Do not chase tail_growth <= 1.0 on the current 2D score with a
   physically faithful solver** — the exact two-fluid KE peaks beyond
   t_end (analytic tail 1.66).  Configs that pass it do so via
   projection erasure (the pin) or acoustic head-loading (dual N=2/3).
2. **Do not build off-cadence remap snapshots from the raw
   `eos(m/dual_vol)` recompute** — measured 8.8 Pa/call jolt and
   full-horizon l2 1.31; use `evolve_snapshot_local_strain`.
3. **Do not skip redistribution under live Delaunay reconnection**
   (now code-enforced for `projection_every > 1`).
4. The projection KE-jump channel is a measured non-lever (−2.2e-9 J
   over the run); do not spend a lane on velocity-rescaling the
   redistribution.
5. The l=0 bump (+0.6% R0, cadence-insensitive) is the dominant
   residual inside l2 ≈ 0.04 — chord-vs-arc/polygon-area territory
   (suspect (ii)), shared with laneG's 3D shape-mode findings; a
   dedicated lane would wire the arc/bulge reconstruction and re-pin
   the 2D floors.
6. 3D: `projection_every` is dim-agnostic and untested in 3D beyond
   the default path (bit-identical); laneG measured redistribution =
   ~half the 3D bump under dual_only, so a 3D cadence probe is a
   natural follow-up AFTER the 3D score recalibration (laneG DO-NOTs
   apply).
