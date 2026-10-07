# Lane B — 3D oscillating-droplet score harness (Tier 3B, 06 §1.6)

> Session 2026-07-29 (wf4).  Builds the missing 3D dynamic score harness
> (06 §1.6; DEVELOPMENT.md "3D oscillating-droplet metric harness" backlog),
> produces the FIRST 3D baseline score on the default policy, and runs the
> first informed 3D retopo A/B (per-step Delaunay vs dual_only) with the
> lane-5 boundary-bookkeeping caveat explicitly verified first.

## Verdict up front

**LANDED.**  `oscillation_score_3d` scores R_max(t) against the Rayleigh–Lamb
`max_radius_envelope` (Miller–Scriven-corrected omega = 16.5145 rad/s with
rho_outer = 1000, Lamb beta_3d = 31.25 1/s — overdamped) exactly like the 2D
score, plus a secondary apex score, machine-checkable `boundary_saturation`
flag for the documented R_max-at-~5·R0 mesh-boundary artefact, and dual-volume
/ interface-count bookkeeping fields.  First 3D baseline (per-step Delaunay,
refine 2/2, 872 steps): **l2 1.5245, linf 2.2655, tail_growth 0.4707,
mass_drift 3.87e-14, boundary_saturation FALSE** — the historical
saturation-at-the-far-boundary artefact is GONE in the current code state
(R_max_peak 0.011397 = 1.14·R0, far below the 0.045 threshold); what remains
is a genuine physics gap (slow ~1.4 % droplet inflation instead of decay to
R0).  A/B: dual_only was verified bookkeeping-clean BEFORE scoring
(step-granular probe: identical boundary count 96, identical |dV/V0| step-0
artefact 0.357050, mass drift ~2e-15), then measured **clearly better on the
full run: l2 0.24811 (−83.7 %), tail 0.08410, mass 1.91e-14, interface 98
constant, no saturation** — the decision rule (clearly better AND clean
mass/interface) is met, so **`retopo_policy_3d = 'dual_only'` is the new 3D
default** (mirroring the 2D lane-5 conclusion; per-step Delaunay reconnection
of the moving cospherical interface cloud was again the dominant error
source).  2D untouched: every 2D battery number bit-identical to 16 digits.

## What changed (file:line)

1. **`cases_dynamic/oscillating_droplet/src/_metrics.py`** — NEW
   `oscillation_score_3d(diags, R0, epsilon, l, omega, beta, r_boundary=None,
   saturation_frac=0.9)`:
   - Primary: L2/Linf of `R_max(t)` vs `max_radius_envelope`, normalized by
     `epsilon*R0` (same construction as the 2D score's apex L2).
   - Secondary: `apex_l2_error_normalized`/`apex_linf_error_normalized` vs
     `radius_perturbation` at the tracked `theta_apex` when apex fields are
     recorded (None otherwise).
   - KE `tail_growth` and `mass_drift`: bitwise the same math as 2D.
   - **`boundary_saturation`**: any frame with
     `R_max >= saturation_frac * r_boundary` flags the documented artefact
     (DEVELOPMENT.md Tier 3B note: R_max saturating at the mesh boundary
     ~5·R0 is non-physical) instead of silently scoring garbage; also
     reports `n_saturated_frames` and `R_max_peak`.
   - Optional bookkeeping when recorded: `dual_vol_step0_jump` /
     `dual_vol_drift_post` (from per-frame `total_dual_vol`) and
     `n_interface_start/end/min/max` — the dual_only A/B verification
     channels.
   - `summary = max(l2, mass_drift, tail_growth-1)`, `kind='oscillation_3d'`;
     `diff_baselines` works unchanged on it.
2. **`cases_dynamic/oscillating_droplet/src/_plot_helpers.py`**
   (`compute_diagnostics`) — new kwarg `polar_axis='x'` (3D only): `'z'`
   measures theta from the +z perturbation axis of
   `_setup._apply_perturbation`, making `theta_apex` consistent with
   `radius_perturbation`.  Default `'x'` keeps every existing consumer
   (2D and 3D) bit-identical; all call sites checked (max 2 positional args).
3. **`cases_dynamic/oscillating_droplet/src/_params.py`** — NEW
   `retopo_policy_3d = 'dual_only'` (flipped from the implicit Delaunay
   default by the A/B below; sweep table + lane-5-caveat verification
   summary in the comment; 2D `retopo_policy_2d` untouched).
4. **`cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py`** —
   outputs move to `results_3d/` (score.json + snapshots; the old runner
   wrote snapshots into the 2D `results/snapshots`); 2D-style `diag_list`
   recording with `polar_axis='z'` and per-frame `total_dual_vol`; scores
   and saves `results_3d/score.json` after the run with
   `r_boundary=L_domain`; `--retopo {delaunay,dual_only}` CLI flag
   (default = `retopo_policy_3d`); non-default policies write suffixed
   artifacts (`score_dual_only.json`, `snapshots_dual_only/`, suffixed
   figs) so an A/B run can never clobber the default baseline.
5. **`ddgclib/tests/test_oscillation_score_3d.py` (NEW, 13 tests, ~1 s)** —
   synthetic-data guards: perfect trajectory scores ~0; known-offset l2
   exact; saturation flagged at the threshold and not without `r_boundary`;
   healthy amplitude unflagged; tail-growth penalty; mass-drift detection;
   apex fields present/absent; bookkeeping fields incl. the frame-0→1
   zeroing pattern; JSON round-trip; PLUS `TestDualOnlyRetopoPolicy3D`
   (default-policy pin + refine-1/1 skip_triangulation bookkeeping net).
6. **`cases_dynamic/oscillating_droplet/baselines/baseline_oscillation_3d.json`
   (NEW)** — the first checked-in 3D baseline (= the new-default dual_only
   score, summary 0.24811340819647862).
7. **`DEVELOPMENT.md`** — "3D oscillating-droplet metric harness" feature
   marked Complete with pointers.

Raw artifacts preserved in scratchpad
`wf4/laneB-3d-score-harness/` (`score_3d_delaunay_baseline.json`,
`run3d_delaunay.log`, `run3d_dual_only.log`,
`probe_dual_only_bookkeeping.py` + output in this log).

## First 3D baseline (default per-step Delaunay)

Run: `/home/endres/anaconda3/envs/ddg/bin/python
cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py` (refine 2/2,
472 verts / 98 interface, dt=1.84e-04, 872 steps, t_end 0.1601 s).

| metric | value |
|---|---|
| l2_error_normalized (R_max vs envelope) | **1.5244561707801316** |
| linf_error_normalized | 2.265475988110048 |
| apex_l2 / apex_linf | 1.5244561164327057 / 2.265475988110048 |
| tail_growth | 0.47066568050584034 |
| mass_drift | 3.865706462775263e-14 |
| R_max_peak | 0.011397245128957192 (= 1.14·R0) |
| boundary_saturation | **False** (0 frames ≥ 0.9·L_domain = 0.045) |
| n_interface start/end/min/max | 98 / 98 / 98 / 98 |
| dual_vol_step0_jump / drift_post | 0.3570496083550926 / 0.0050761421317108086 |
| summary | 1.5244561707801316 |

Reading: mass and interface bookkeeping are machine-clean and KE decays
(tail 0.47), but R_max does NOT follow the overdamped decay envelope — it
drifts up from 1.05·R0 to ~1.14·R0 over 5 damping times (analytical:
monotonic decay to R0).  The pre-harness DEVELOPMENT.md observation
"R_max saturates at ~5·R0" no longer reproduces post-2026-07-02/03 fixes +
laneA exact 3D dual volumes; the artefact detector stays in as a tripwire.
The 35.7 % step-0 dual-volume jump is the known boundary-cell-zeroing +
mps.refresh transition (laneA: "residual step-0→1 jump"), identical in both
retopo policies (probe below).

## Lane-5 caveat verification (before trusting the dual_only score)

Lane 5 explicitly warned: "3D skip-triangulation path has different
boundary-volume bookkeeping, batch_e_star zeroing; do not blind-apply."
Step-granular probe (`probe_dual_only_bookkeeping.py`, 5 steps each policy,
identical setup):

| check | delaunay | dual_only |
|---|---|---|
| n_boundary after retopo | 96 | 96 (dV = set(bV) coincides with `boundary_from_simplices`) |
| exact simplex volumes engaged | True | True (`HC._simplices` from setup survives skip) |
| \|dV/V0\| step-0 artefact | 0.357050 | 0.357050 (identical to 6 dp) |
| post-transition \|dV/V1\| (5 steps) | 0.000000 | 6.6e-9/step (volumes track motion under frozen connectivity — expected) |
| mass drift | 1.8e-15 | 2.7e-15 |
| n_interface / n_verts | 98 / 472 | 98 / 472 |

Boundary dual_vol handling is equivalent between the paths on this case;
the dual_only score is trustworthy subject to its own bookkeeping fields.

## A/B: per-step Delaunay vs dual_only (3D)

Both full 872-step runs at identical parameters (refine 2/2, dt 1.84e-04,
t_end 0.1601 s); dual_only = `--retopo dual_only` =
`partial(retopo_fn, skip_triangulation=True)` (builder connectivity frozen;
duals, per-phase splits, mass redistribution, EOS pressures refreshed every
step).

| metric | per-step Delaunay (old default) | dual_only (**new default**) | delta |
|---|---|---|---|
| l2_error_normalized | 1.5244561707801316 | **0.24811340819647862** | **−83.7 %** |
| linf_error_normalized | 2.265475988110048 | 0.5992701359956998 | −73.5 % |
| apex_l2 | 1.5244561164327057 | 0.24811340819647862 | −83.7 % |
| tail_growth | 0.47066568050584034 | 0.08409976059818802 | −82.1 % |
| mass_drift | 3.865706462775263e-14 | 1.905002320272536e-14 | both machine precision |
| R_max_peak | 0.011397245128957192 | 0.010790236105250779 | inflation 1.40 %→0.79 % of R0... closer to the 1.05·R0 analytical start |
| boundary_saturation | False | False | — |
| n_interface (start/end/min/max) | 98/98/98/98 | 98/98/98/98 | clean both |
| dual_vol_step0_jump | 0.3570496083550926 | 0.35704960835509636 | identical boundary-zeroing transition |
| dual_vol_drift_post | 5.0761e-03 | 1.0684e-04 | dual_only's dual volumes are 47× steadier |
| summary | 1.5244561707801316 | **0.24811340819647862** | |

Decision (per lane brief): dual_only is clearly better on every physics
channel AND mass/interface stay clean AND the boundary bookkeeping was
pre-verified equivalent → **3D default flipped to `dual_only`**
(`src/_params.py: retopo_policy_3d`, sweep table in the comment).  The
mechanism mirrors 2D lane-5: per-step Delaunay of the moving near-cospherical
interface cloud reconnects non-uniquely, the dual-volume churn
(drift_post 5.1e-3 vs 1.1e-4) maps through redistribution + EOS into
spurious interface forcing.  Checked-in artifacts follow the new default:
`results_3d/score.json` + `snapshots/` + unsuffixed figs = dual_only run;
the Delaunay A/B leg is preserved as `score_delaunay.json` /
`snapshots_delaunay/` / `*_delaunay.*` figs;
`baselines/baseline_oscillation_3d.json` = the new-default score
(summary 0.24811340819647862).  Both raw score JSONs additionally preserved
in scratchpad `wf4/laneB-3d-score-harness/`
(`score_3d_delaunay_baseline.json`, `score_3d_dual_only.json`).
NEW regression net `TestDualOnlyRetopoPolicy3D` (2 tests, ~1 s, refine 1/1):
default-policy pin + frozen 1-skeleton / interface-set / boundary-zeroing /
exact-volume / mass checks under skip_triangulation.

## Measurement battery (full)

| item | baseline (post-laneA) | this lane |
|---|---|---|
| floor battery (`test_case_oscillating_droplet.py -m ""`) | 14 P | **14 P** (15.5 s, no re-pins; re-run green after the policy flip — the floor tests use setup's per-step-Delaunay `retopo_fn`, not `retopo_policy_3d`) |
| fast suite (`-m "not slow" -q`) | 834 P / 0 F / 2 xfail | **847 P / 0 F**, 12 skipped, 17 deselected, 2 xfailed (51.4 s; +13 = new harness + policy tests) |
| equil summary (static_droplet_2D) | 1.1847162859108737e-03 | 1.1847e-03; max_KE_norm 7.6739e-09, mass_drift 0.0 — **bit-identical** |
| osc 2D l2 / linf / tail / mass | 0.1785660454150319 / 0.32842491275938274 / 0.9992507831101141 / 2.676988154993443e-14 | **identical to 16 digits** (results/score.json re-generated by the battery run) |
| osc 3D (NEW baseline, dual_only default) | — (no harness) | summary 0.24811340819647862, no saturation |
| hyperct | not touched this lane | not run (no hyperct edits) |

## What the next lane / future sessions must know

- The 3D case now has a scored baseline:
  `baselines/baseline_oscillation_3d.json` (summary 1.5245).  Regressions
  can be caught with `diff_baselines(baseline, current)` exactly like 2D.
- **Do not chase l2 → 0 against this reference yet**: like 2D pre-lane-5,
  the 3D l2 is dominated by a physical-model gap (slow droplet inflation,
  +1.4 % R over 5 tau), not noise.  Mass/interface/KE channels are clean.
  Suspects (untested): 3D interface curvature operator `hndA_i_interface`
  truncation on the refine-2 icosphere-like ring, YL mass-preload
  discretisation at refine 2/2.  The 2D analog (chord-vs-arc pressure-side
  floor, 07 §3) suggests a curvature-side mechanism.
- `boundary_saturation` FALSE is a meaningful health signal now; if a future
  change reintroduces interface escape, the score flags it instead of
  producing a silently-garbage l2.
- The apex score in 3D requires `compute_diagnostics(..., polar_axis='z')`
  (perturbation axis).  The legacy `polar_axis='x'` default measures theta
  from +x and CANNOT be fed to `radius_perturbation` for the 3D mode shape.
- **The 3D default is now `dual_only`** (like 2D).  The pinned 3D
  static-droplet floor tests (7.274172e-05 plateau etc.) still exercise the
  per-step Delaunay path via setup's default `retopo_fn` — they are
  unaffected by the policy constant and were re-verified green.  Large
  deformation / detaching-interface 3D cases still cannot freeze
  connectivity (08 §5.4) — this flip is for the oscillating-droplet runner,
  not a global integrator change.
- Lane brief said "keep Delaunay unless clearly better": dual_only won by
  6.1× on l2 with every cleanliness channel verified, which is the same
  magnitude of win that justified the 2D lane-5 flip.
- Updated numbers stale-list: DEVELOPMENT.md's "R_max saturates at ~5·R0
  (872 steps ...)" observation predates the 2026-07-02/03 fixes and laneA;
  the current runs (both policies) peak at 1.08–1.14·R0 with 98/98
  interface vertices retained.  The `boundary_saturation` tripwire keeps
  guarding against a regression to that state.
