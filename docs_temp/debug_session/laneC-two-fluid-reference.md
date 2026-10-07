# laneC-two-fluid-reference — two-fluid reference for the 2D oscillating-droplet score

Date: 2026-07-29. Validation lane (NO solver changes). Follow-up to
lane5-dynamic-config-sweep.md §5/§7: the winner's l2 = 0.179 is a smooth
~5-10 % over-decay, ANTI-convergent under refinement, and lane 5
suspected the single-fluid Lamb reference (beta = 25, ignores the outer
bath rho_o = 1000 / mu_o = 0.1 and the no-slip walls at 5 R0)
under-counts dissipation, i.e. part of the "error" is the reference's.

## Verdict

**LANDED — and the lane-5 hypothesis is REFUTED.** The exact two-fluid
reference explains essentially NONE of l2 = 0.179; it slightly
*increases* the mismatch:

| reference for r_apex(t) | l2 | linf |
|---|---|---|
| single-fluid Lamb oscillator (pinned metric) | **0.1785660454150319** | 0.32842491275938274 |
| **two-fluid normal-mode dispersion (exact, new)** | **0.1885496661389148** (+5.6 %) | 0.3159570966402604 (−3.8 %) |
| two-fluid energy-method oscillator (closed form, diagnostic) | 0.140723 | 0.252863 |

Mechanism: adding the outer bath does NOT produce a faster-decaying
version of the single-fluid envelope — it changes the mode *structure*.
The exact least-damped two-fluid mode is weakly **oscillatory**,
s = **−6.8325510 ± 5.4707015i** (beta 6.83, omega_d 5.47,
omega_eff = |s| = 8.7528), not overdamped-biexponential, and its
started-from-rest temporal factor stays slightly ABOVE the single-fluid
biexponential through the window (0.924 vs 0.898 at t = 0.05 s) and
lands almost on top of it at t_end (0.706 vs 0.719 — coincidence of
this parameter set). The simulation under-shoots BOTH references
(temporal 0.390 at t_end; final-frame apex error −0.328 (1f) / −0.316
(2f) in eps·R0 units). A confined-wall 6×6 probe (rigid no-slip circle
at r_w) shifts the exact rate by only ~0.1 % at r_w = 5 R0
(−6.8326 → −6.8401): the wall is negligible too. **The over-decay is
solver-side** (lane-5 candidates (i) per-step mass-redistribution
projection and (ii) O(h) 32-gon curvature bias remain; candidate (iii)
"the reference is wrong" is closed).

The energy-method closed form fits the trajectory better (0.141) only
by coincidence: it decays faster (lam_slow 5.556 vs 3.591) in the same
direction as the solver's over-decay, but it is NOT the defensible
physics — its potential-flow bulk-dissipation assumption misses the
interfacial vortical layer, which the dispersion solution shows
dominates at weak viscosity (damping ~ sqrt(mu), 2–7× the energy-method
value; the Miller–Scriven mechanism, verified numerically). Do not
promote it to a target.

## Decay-rate comparison (the lane question, item 4)

Measured on the fresh official-runner series (207 frames, log-linear
fit on the second half; lane 5's own fit of the same config gave
KE 6.79):

| quantity | value [1/s] |
|---|---|
| measured KE tail decay rate | **6.53** (lane-5 fit: 6.79) |
| measured apex-amplitude tail decay rate | 11.40 |
| measured whole-window amplitude decay ln(a0/aT)/T | 8.23 |
| single-fluid Lamb: 2·lam_slow (KE) / lam_slow (amp) | 7.183 / 3.591 |
| two-fluid exact dispersion: 2·beta (KE env) / beta (amp env) | 13.665 / 6.833 |
| two-fluid energy-method osc: 2·lam_slow / lam_slow | 11.111 / 5.556 |

The measured KE decay is *closest to the single-fluid* prediction; the
corrected references predict FASTER energy decay, not slower — the
opposite of what "reference under-counts dissipation" needs. Note the
sim's KE (6.5) and apex-amplitude (11.4) tail rates disagree by ~2×:
the simulated decay is not single-mode, so any single-mode reference
comparison saturates around this level.

## 1. What changed (file:line)

1. **`cases_dynamic/oscillating_droplet/src/_analytical.py`** (:216-527,
   NEW functions only — pinned single-fluid functions untouched):
   - `lamb_damping_rate_two_fluid` (:225) — closed-form Lamb
     energy-method beta, 2D: `beta = 2l[(l-1)mu + (l+1)mu_o] /
     ((rho+rho_o) R0²)`; recovers `lamb_damping_rate` exactly at
     `mu_outer = rho_outer = 0`; docstring carries the derivation and a
     warning that it underestimates two-fluid damping (interfacial
     layer).
   - `mode_temporal_ivp` (:269) — exact started-from-rest damped-mode
     temporal factor in all regimes (the pinned `radius_perturbation`
     omits the `(beta/w_d) sin` term in its underdamped branch; here
     `beta/w_d ≈ 1.25`, so the exact term matters).
   - `radius_perturbation_two_fluid` (:308) — same IC construction as
     the pinned metric, exact temporal factor, labelled single-mode
     IVP-projection approximation (Prosperetti-type transient
     neglected).
   - `two_fluid_dispersion_det_2d` (:352) — the 4×4 determinant of the
     linearised two-fluid Stokes/NS normal-mode problem
     (streamfunction; inner `r^l`, `I_l(q_i r)`; outer `r^-l`,
     `K_l(q_o r)`; u_r/u_theta continuity, tangential-stress
     continuity, normal-stress–curvature jump; overflow-safe scaled
     Bessel ratios). Full formulation + verified limits in the
     docstring.
   - `two_fluid_dispersion_roots_2d` (:443) — multi-seed complex
     Newton root finder, spurious-|s|→0 filter, least-damped-first.
   - `two_fluid_omega_beta_2d` (:487) — maps the dominant root(s) to
     the `(omega, beta)` pair consumed by the reference (complex pair
     and real-pair branches).
2. **`cases_dynamic/oscillating_droplet/src/_metrics.py`** (:128
   `add_two_fluid_reference` + import) — attaches
   `l2_error_normalized_two_fluid`, `linf_error_normalized_two_fluid`,
   `inputs_two_fluid` to an oscillation score. Pinned fields and
   `summary` untouched; `diff_baselines` unaffected (iterates baseline
   keys only).
3. **`cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py`**
   (:30-33, :44, :166-208) — computes both scores side by side, prints
   them, writes both into `results/score.json`, and now also dumps the
   raw scalar diagnostic series to `results/diag_series.json` so
   future reference changes can re-score without re-running (lane 5's
   scratchpad series had been tmp-cleaned; this run had to be repeated
   once for exactly that reason).
4. **`ddgclib/tests/test_case_oscillating_droplet_two_fluid.py`**
   (NEW, 15 tests, 0.4 s) — deliberately a separate file so the pinned
   floor battery keeps its exact 14-test count. Limiting cases:
   `mu_outer = rho_outer = 0` recovers `lamb_damping_rate` to 1e-14;
   dispersion inviscid limit → ±i·omega(rho+rho_o); vanishing-outer +
   weak-mu limit → Re s → −lamb_damping_rate (2D single-fluid);
   case-parameter root pinned (−6.8325510 ± 5.4707015i, rel 1e-4);
   `mode_temporal_ivp` IC/ODE-residual checks and bit-match of its
   overdamped branch to the pinned `radius_perturbation`; metric
   attachment leaves pinned fields bit-unchanged and scores a perfect
   two-fluid trajectory at < 1e-12.

NOT changed: `src/_params.py`, `_setup.py`, any solver/operator/
integrator code, any baseline JSON (pinned metric = single-fluid,
unchanged; `baselines/baseline_oscillation.json` still matches the
official run bit-for-bit).

## 2. Dispersion-relation validation (before trusting the root)

- Near-inviscid (mu ×1e-4): s → −0.0380 + 12.8723i vs inviscid
  omega = 12.9099 with (rho+rho_o) inertia (0.3 %; Re → 0). ✓
- Damping scaling at weak viscosity: Re s ∝ sqrt(mu) (0.122 → 0.409
  for mu ×10), i.e. the interfacial vortical layer dominates the O(mu)
  bulk terms — Miller–Scriven mechanism; the energy-method closed form
  is confirmed as a weak-viscosity *underestimate*. ✓
- Vanishing outer phase (rho_o → 1, mu_o → 1e-5) at weak mu_i = 0.005:
  s = −0.2437 + 19.3368i vs single-fluid Lamb beta 0.25 /
  omega 19.3649 (2.5 % / 0.15 %); independent inner-only 2×2
  free-surface determinant agrees (−0.2398 + 19.3539i). ✓
- Viscosity ramp 0.01× → 1×: the root moves continuously
  (−0.41+12.53i → −6.83+5.47i), no branch jump — the target root is
  the analytic continuation of the validated weak-viscosity mode. ✓
- Both target roots have Re(q) > 0 in both phases → genuinely decaying
  eigenfunctions (discrete normal modes; the analytic-continuation
  caveat applies only to purely real negative s, not used here). ✓
- Confined 6×6 probe (scratchpad `proto_confined.py`, rigid no-slip
  circle): r_w = 50 R0 reproduces the unbounded root to 6 digits;
  r_w = 5 R0 → −6.8401 + 5.4631i (+0.11 % on beta); even r_w = 2.5 R0
  only reaches −7.55. Wall dissipation is negligible at this
  confinement — kept as a probe, not shipped.

Aside (single-fluid, for perspective): the exact single-fluid 2D
free-surface dispersion at the full mu_d = 0.5 (Oh ≈ 0.79) gives
s = −17.44 ± 6.55i — the weak-damping oscillator mapping
(beta 25 > omega 12.91, "overdamped", lam_slow 3.59) is itself far
from the exact single-fluid mode at this viscosity. Both the pinned
reference and the sim sit in a regime where all these envelope
references are approximations; judge changes by the pinned score's
*relative* movement, not by chasing l2 → 0.

## 3. Measurement battery (cwd repo root, ddg env)

1. Floor battery (`pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`):
   **14 passed** (new tests live in a separate file by design).
2. Fast suite (`pytest ddgclib/tests/ -m "not slow" -q`):
   **862 passed, 12 skipped, 17 deselected, 2 xfailed** = 847
   (post-lane-A/B) + 15 new; **0 failures, 0 new failures**.
3. `static_droplet_2D.py`: summary **1.1847e-03**
   (interface_radius_drift 1.1847162859108737e-03) — unchanged.
4. `oscillating_droplet_2D.py`: l2 **0.1785660454150319** / linf
   0.32842491275938274 / tail **0.9992507831101141** / mass
   2.676988154993443e-14 — **bit-identical to 16 digits** (run twice:
   the first run died at the new diag-series JSON dump on the ndarray
   'com' field — fixed by dumping scalar fields only — and was
   repeated end-to-end cleanly; identical scores both runs).
   New fields: l2_two_fluid 0.1885496661389148, linf_two_fluid
   0.3159570966402604, inputs_two_fluid {omega 8.752846836064656,
   beta 6.832550976708139, beta_energy 17.777777777777775}.

No hyperct edits this lane.

## 4. Caveats / guidance for future lanes

- **The single-fluid pinned metric stays the regression gate.** The
  two-fluid numbers are reported side by side in `score.json`; do not
  re-pin baselines to them (the l2 targets in the plan were calibrated
  against the single-fluid reference).
- The two-fluid reference is a **single-normal-mode, started-from-rest
  projection** (2nd-order-ODE weights). The full viscous IVP adds a
  vorticity-diffusion transient (Prosperetti 1980, 3D analogue). Given
  the sim's own KE/amplitude rate disagreement (6.5 vs 11.4 1/s), a
  full IVP treatment would still not license chasing l2 → 0.
- Residual-attribution update for 06 §4 / lane-5 §7: candidate (iii)
  ("beta = 25 reference ignores outer damping → error is partly the
  reference's") is **closed by measurement**; candidates (i)
  per-step mass-redistribution projection and (ii) 32-gon curvature
  bias remain the live suspects for the ~5-10 % over-decay, now
  measured as: sim temporal 0.390 at t_end vs 0.706-0.719 for every
  defensible reference.
- 3D was NOT touched (Miller & Scriven 1968 has the 3D two-fluid
  results if laneB's 0.248 baseline ever needs the same treatment;
  `lamb_damping_rate_two_fluid` deliberately raises
  NotImplementedError for dim=3 pointing there).
- Scratchpad artifacts (prototypes, confined-wall probe, analysis,
  logs): `scratchpad/wf4/laneC-two-fluid-reference/` (proto2.py,
  proto_confined.py, analyze_series.py, analysis_output.txt,
  reference_rates.txt, score_with_two_fluid.json, diag_series.json,
  run2d_fixed.log).
