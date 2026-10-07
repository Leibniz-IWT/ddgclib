# lane5-dynamic-config-sweep — retopo policy / c_s / refinement sweep on the 2D oscillating droplet

Date: 2026-07-02. Sequential lane 5 (after lane1-zero-gauge-pressure,
lane2-eos-consistency, lane3-exact-dual-volumes, lane4-remesh-upstream).
All edits are working-tree only (no git operations).

## Verdict

**LANDED.** Winner: **`dual_only` retopology policy at c_s = 1 m/s,
refinement 3/3** — keep the interface-conforming builder connectivity for
the whole run (`skip_triangulation=True` bound into setup's
`_retopologize_multiphase` partial) while still refreshing duals,
per-phase dual splits, per-phase mass redistribution and EOS pressures
every step.  Applied as the new default of `oscillating_droplet_2D.py`
via a new `retopo_policy_2d = 'dual_only'` knob in `src/_params.py`.

Official battery scores (full 1839-step run):

| metric | post-lane-4 default (per-step Delaunay) | lane-5 default (dual_only, official battery run) | target |
|---|---|---|---|
| l2_error_normalized | 0.48991833470391266 | **0.1785660454150319** | < 0.2 ✓ |
| tail_growth | 1.7250489596305962 | **0.9992507831101141** | < 1.0 ✓ |
| linf_error_normalized | 0.8503543927096853 | **0.32842491275938274** | — |
| mass_drift | 1.42893286651677e-14 | 2.676988154993443e-14 | — |
| summary | 0.7250489596305962 | **0.1785660454150319** | — |
| KE_max [J] | 4.0716e-02 (still growing at t_end) | **8.317e-07 (physical overdamped decay)** | KE must decay ✓ |
| equilibrium summary (static_droplet_2D) | 1.1847162859108737e-03 | 1.1847162859108737e-03 (bit-identical) | no regression ✓ |

(The sweep-driver replica of the same config measured l2
0.1785656763929343 / tail 0.9992520610059813 — a 4e-7 relative
run-to-run delta vs the official runner; the driver's `full_delaunay`
control reproduced the old production score bit-exactly.)

**Both lane targets are met for the first time** (l2 < 0.2, tail < 1.0).
The KE(t) of the winner is not merely "lower noise" — it quantitatively
reproduces the analytical overdamped Rayleigh–Lamb energy envelope
(β=25.0, ω=12.9099: λ_slow = β − √(β²−ω²) = 3.5913 1/s):
KE peaks at t = 0.0549 s (analytical t* = 0.0598 s), tail decay rate
6.79 1/s vs analytical 2λ_slow = 7.183 1/s (−5.5 %), KE_final/KE_peak
0.6955 vs 0.6758.  Under per-step Delaunay the KE is ~5·10⁴× larger and
still growing at t_end — i.e. essentially 100 % of the old KE was
retopology-injected noise, not fluid motion.

## 1. What changed (file:line)

1. **`cases_dynamic/oscillating_droplet/src/_params.py`** (~:71): new
   `retopo_policy_2d = 'dual_only'` with the sweep table and rationale
   in a comment.  No other parameter changed — **c_s stays at the 1 m/s
   floor, K_d=800 / K_o=1000 unchanged** (see §4), so the pinned floor
   tests and every other consumer of `_params` see identical physics.
2. **`cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py`**
   (~:24 import, ~:79-90): when `retopo_policy_2d == 'dual_only'`, rebind
   `retopo_fn = partial(retopo_fn, skip_triangulation=True)`.  Setup's
   returned partial (`_retopologize_multiphase` with `mps`,
   `split_method='neighbour_count'`, `redistribute_mass=True`) is
   otherwise untouched, so `mps.refresh` + `redistribute_mass_multiphase`
   + `compute_phase_pressures` still run every step — only the global
   Delaunay disconnect/reconnect is skipped.
3. **`ddgclib/tests/test_case_oscillating_droplet.py`** (appended,
   `TestDualOnlyRetopoPolicy2D`): fast regression net (1 s) for the new
   default path — frozen 1-skeleton across a 40-step symplectic run,
   interface/vertex sets preserved, mass drift < 1e-10, KE < 1e-5 J.
4. **`cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json`**:
   re-pinned to the new official score.  OLD values (pre-lane-1 era,
   never updated by lanes 1–4): l2 5.380444352276059,
   linf 16.433190183556857, tail_growth 2.6858286134208984,
   mass_drift 1.8261960666002507e-16, summary 5.380444352276059,
   inputs beta=43.75 / omega=19.364916731037084 (the old runner used the
   `_params` beta_2d; the current runner uses `lamb_damping_rate` beta=25.0,
   omega=12.909944487358056).  NEW values = the battery step-4 score
   below.  `baseline_equilibrium.json` left untouched (its summary
   0.0074108 is still an upper bound; equilibrium unchanged this lane).

NOT changed (deliberately): `src/_setup.py` (setup's default retopo_fn
still per-step Delaunay — the pinned A.5.b floor tests exercise exactly
that path and its 2.2716938e-3 / 7.3768e-5 plateaus remain valid);
`static_droplet_2D.py` (already dual-only via its own `_dual_only_retopo`);
`oscillating_droplet_3D.py` (3D not swept this lane — the 3D
skip-triangulation path has different boundary-volume bookkeeping,
batch_e_star zeroing; do not blind-apply);
`mesh_convergence_2D.py` and the `diagnose_*.py` forensic scripts.

## 2. The sweep (driver: scratchpad wf3/lane5/driver.py, faithful runner replica)

Control reproduced the production score **bit-exactly** (l2
0.489918…, tail 1.725049…), validating the driver.  All runs: full
1839 steps, dt = 6.2167e-05, refine 3/3, 311 verts / 32 interface,
c_s=1 unless stated.  `full` = full Delaunay retopo calls, `light` =
dual-only refresh calls.

| config | l2 | linf | tail_growth | summary | KE_max [J] | t@KE_max | KE_fin/KE_max | full | light |
|---|---|---|---|---|---|---|---|---|---|
| **dual_only** (winner) | **0.17857** | **0.32842** | **0.99925** | **0.17857** | 8.317e-07 | 0.0549 | 0.6955 | 0 | 1839 |
| adaptive (lane-4 kwargs) | 0.32641 | 0.66634 | 1.09313 | 0.32641 | 1.311e-02 | 0.0605 | 0.8009 | 1839 | 0 |
| dual_only_noredist | 0.49503 | 1.38640 | **0.11132** | 0.49503 | 7.694e-04 | 0.0387 | 0.0423 | 0 | 1839 |
| full_delaunay (old default) | 0.48992 | 0.85035 | 1.72505 | 0.72505 | 4.072e-02 | 0.1136 | 0.9682 | 1839 | 0 |
| gate_005 (eps=0.05·h_min) | 0.46692 | 0.72034 | 1.83154 | 0.83154 | 4.248e-02 | 0.1064 | 0.9446 | 1508 | 0 |
| gate_02 (eps=0.2·h_min) | 0.82821 | 1.51371 | 1.92345 | 0.92345 | 5.043e-02 | 0.1041 | 0.9916 | 634 | 0 |
| gate_001 (eps=0.01·h_min) | 0.49984 | 0.88773 | 2.31041 | 1.31041 | 5.543e-02 | 0.1143 | 1.0000 | 1773 | 0 |
| hybrid_005 (gate+dual refresh) | 0.48634 | 0.83203 | 2.39467 | 1.39467 | 5.102e-02 | 0.1142 | 0.9936 | 1439 | 400 |
| hybrid_02 | 2.59080 | 6.21257 | 7.45263 | 6.45263 | 1.066e-01 | 0.1086 | 0.8249 | 713 | 1126 |

mass_drift ≤ 2.7e-14 in every config (0.0 exactly for noredist).

### Reading of the table

- **Per-step global Delaunay reconnection is the dominant l2/tail error
  source.**  Within 60 steps it pumps KE to 6e-4 J while the physical
  level is ~1e-7 J; over the run it reaches 4.1e-2 J, ~5e4× physical,
  still growing at t_end (KE_fin/KE_max 0.97).  r_apex oscillates around
  the analytical envelope (overshooting below R0) instead of decaying
  monotonically.
- **The displacement gate (existing `displacement_eps` in
  `_do_retopologize`) is wired correctly but does not help this case.**
  How to enable it (validated here): pass `displacement_eps=frac*dx_min`
  to `symplectic_euler` AND run one manual `retopo_fn(HC, bV, dim)`
  before integrating (the gate's first call snapshots-and-skips, and
  setup's duals are stale w.r.t. the perturbation, so without the manual
  call the first steps run on stale geometry).  Results: small eps
  (0.01·h_min) degenerates to per-step Delaunay once motion builds
  (1773/1839 full calls) with occasional stale-dual episodes → tail
  WORSE (2.31); large eps (0.2·h_min) accumulates displacement between
  events so each Delaunay rewire is a bigger shock → both metrics worse
  (0.83/1.92).  There is no good eps: the noise the gate is meant to
  suppress is itself what keeps vertex displacement above any useful
  threshold (chicken-and-egg).
- **Hybrid (dual-only refresh on gated steps, full Delaunay past eps) is
  strictly worse than either extreme** — rare-but-large rewiring shocks
  on a mesh whose pressures were kept consistent between events:
  hybrid_02 exploded (l2 2.59, tail 7.45).  Do not resurrect.
- **adaptive (lane-4 fixed) is now second-best** and beats per-step
  Delaunay on BOTH l2 (0.326 vs 0.490) and tail (1.093 vs 1.725) at the
  full refine-3/3 run — better than the lane-4 refine-2 smoke suggested
  (2.73 vs 1.48).  Still 1.8× dual_only on l2; consistent with lane 4's
  frozen-connectivity-distortion diagnosis + local-op cost.
- **redistribute_mass matters for l2, not for stability, under dual_only**:
  noredist keeps the fully compressible response — KE decays beautifully
  (tail 0.111, KE_fin/max 0.042) but the wave-launch acoustic transient
  is bigger (KE_max 7.7e-4, linf 1.386 early excursion) and the decay
  overshoots → l2 0.495.  With redistribution the pressure field is
  quasi-projected each step and both metrics land (0.179/0.999).

## 3. Why dual_only wins (mechanism)

2D Delaunay of the moving droplet cloud is non-unique/churning near the
symmetric interface ring; every reconnection changes dual volumes
discontinuously and `redistribute_mass_multiphase`'s global rescale
converts that into pressure/mass jolts (the p_ref benchmark documented
the same mechanism: "the code can read a topological connectivity change
as a physical compression/expansion event").  The lane-3 exact dual
volumes reduced but could not eliminate this (l2 0.49 floor).  With the
connectivity frozen to the interface-conforming builder mesh, the only
per-step changes are smooth geometric ones (duals from moved positions),
and the measured KE(t) collapses onto the analytical overdamped
envelope (§ verdict).  For the small-strain overdamped droplet
(ε=0.05, monotone decay) frozen connectivity costs nothing: interface
count stays 32/32, mass drift 2.7e-14, no mesh tangling (KE and R(t)
smooth throughout).  This mirrors the two case-level precedents:
`static_droplet_2D` (equilibrium; dual-only was already its fix) and
`cube2droplet/diagnostic_no_retopo` (the documented stable reference).

## 4. c_s floor sensitivity — (b) re-test, answer: KEEP c_s = 1

The `_params.py` comment said raising c_s 1→50 was "inconclusive because
real KE-growth bugs dominate".  Post lanes 1–4 the answer is now sharp,
and it is policy-dependent (K_d = ρ_d·c_s², K_o = ρ_o·c_s², dt
auto-shrinks ∝ 1/c_s: c_s=1 → dt 6.2167e-5/1839 steps; c_s=5 →
1.2433e-5/9192; c_s=10 → 6.2167e-6/18384):

| config | l2 | linf | tail | KE_max [J] | wall [s] | notes |
|---|---|---|---|---|---|---|
| dual_only c_s=1 | 0.178566 | 0.328425 | 0.999251 | 8.32e-07 | 129 | winner |
| dual_only c_s=5 | 0.178492 | 0.328151 | 0.979736 | 8.75e-07 | 645 | R_max matches c_s=1 to ~1e-6 at equal t |
| dual_only c_s=10 | 0.178730 | 0.327927 | 0.308235 | 3.39e-06 | 1234 | same trajectory; tail "improves" only because a larger early acoustic KE peak shifts the head/tail ratio — l2 unchanged |
| delaunay c_s=5 | 12.960052 | 20.831807 | 3.730859 | 1.95e+01 | 660 | DIVERGED: KE 2.46 J by t=0.034; R_max 0.0126 at t=0.056; interface ring destroyed (n_iface→0, R_max→0) by t≈0.078 |

- **Per-step Delaunay + stiffer EOS is catastrophic**: every rewiring
  volume-jolt now maps through K=20 000 Pa instead of 800 Pa; KE reaches
  2.46 J by t=0.034 and 18.7 J by t=0.078, and the interface ring is
  completely destroyed mid-run.  The old "inconclusive" verdict is
  superseded: with the retopo-noise mechanism now understood, c_s
  amplifies it linearly in K.
- **dual_only is c_s-insensitive** (R_max matches to ~1e-6 between c_s=1
  and c_s=5 at equal times; KE envelope unchanged): with per-step
  redistribution the pressure field is re-projected each step, so K only
  scales the (tiny) residual compressibility.  Raising c_s buys nothing
  and costs 5–10× the steps.  **Default stays c_s = 1 m/s** — also keeps
  K_d/K_o and every pinned floor bit-compatible.

## 5. Refinement direction — (c) 3/3 vs 4/4 (dual_only)

| config | mesh | l2 | linf | tail | KE_max [J] | KE_fin/max |
|---|---|---|---|---|---|---|
| dual_only refine 3/3 | 311 verts / 32 iface, dx_min 2.4867e-4, dt 6.2167e-5, 1839 steps | 0.178566 | 0.328425 | 0.999251 | 8.32e-07 | 0.6955 |
| dual_only refine 4/4 | 1106 verts / 64 iface, dx_min 2.5439e-4, dt 6.3598e-5, 1798 steps | 0.270630 | 0.453846 | 1.006449 | 8.40e-07 | 0.6887 |

**The l2 metric is ANTI-convergent under refinement** (+52 %) while the
KE envelope is refinement-stable (peak magnitude, peak time and decay
ratio essentially unchanged).  The finer 64-gon interface decays
*faster still* relative to the Lamb reference — i.e. refining does not
pull the simulation toward the reference curve.  Combined with §7 this
strengthens the suspicion that a chunk of the residual "error" is in
the single-fluid Lamb β=25.0 reference (no outer-fluid damping; the
outer bath here is ρ_o=1000, μ_o=0.1, in a 5R0 box with no-slip walls
— all of which physically ADD dissipation the reference ignores), not
in the solver.  Do NOT spend a lane chasing l2 to 0 against this
reference on this configuration; a two-fluid (Prosperetti-type)
reference or a viscous-corrected β would be the right next validation
step.  Refinement 3/3 stays the default.

## 6. Measurement battery (after all changes; cwd repo root, ddg env)

1. Pinned floor tests (`pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`):
   **13 passed** (12 pinned + 1 new `TestDualOnlyRetopoPolicy2D`), floors
   untouched (2.3748568e-3 / 2.2716938e-3 / 6.0153e-5 / 7.3768e-5 —
   setup's Delaunay path is unchanged).
2. Fast suite (`pytest ddgclib/tests/ -m "not slow" -q`):
   **1 failed, 825 passed, 12 skipped, 17 deselected, 3 xfailed in 50.74s**
   — the 1 failure is the pre-existing
   `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`;
   825 = 824 (post-lane-4) + 1 new test.  **No NEW failures.**
3. `static_droplet_2D.py`: summary **1.1847e-03**
   (interface_radius_drift 1.1847162859108737e-03, max_KE_normalized
   7.6739e-09, mass_drift 0.0) — bit-identical to post-lane-4.
4. `oscillating_droplet_2D.py` (new default): see verdict table.
   Score copies: `scratchpad/wf3/lane5-dynamic-config-sweep_osc_score.json`,
   `_equil_score.json`.

## 7. Residual error characterisation (the next l2 target)

The winner's remaining l2 = 0.179 is a **smooth, monotone over-decay**,
not noise: the θ=0 apex error vs `radius_perturbation` grows steadily to
−0.328·(εR0) at t_end (the linf is AT the final frame; RMS 0.1786
reproduces the l2 to 4 digits).  KE decay rate is 6.79 vs analytical
7.18 1/s, so the interface amplitude decays ~5–10 % too fast per unit
time and the error integrates.  Candidate mechanisms for a future lane:
(i) the per-step mass-redistribution projection acts as extra
dissipation; (ii) O(h) polygon curvature bias (32-gon) in the ST-vs-
pressure balance; (iii) `lamb_damping_rate` 2D β=25.0 is itself an
approximation for a two-fluid system (ρ_o=1000 outer bath is NOT a
vacuum; the analytical form ignores outer-fluid damping, which would
make the TRUE decay *faster* than the reference — i.e. part of the
"error" may be the reference's, not the solver's).

## 8. Caveats / guidance for future lanes

- **Scope of the win**: this fixes the *benchmark configuration*, not the
  Delaunay-churn defect itself.  Cases with large deformation / topology
  change (dam break, detaching bubbles) cannot freeze connectivity; for
  them the per-retopo jolt mechanism remains open (candidate real fixes:
  conservative old-dual→new-dual remap as in the p_ref benchmark, or the
  ALE-style remapped smoothing flagged in lane 4 §6.3).
- **tail_growth 0.99925 is metric-fragile**: KE peaks at t=0.0549 s,
  just before the half-run boundary t=0.0572 the metric splits on.  Any
  change that shifts the peak later by ~2 ms flips tail past 1.0 without
  physics regressing.  Judge KE health by the envelope match (peak time,
  decay rate), not the binary tail number.
- The gate/hybrid results above close the "displacement gate for the 2D
  droplet" idea; the gate remains correct and useful for its original 3D
  static purpose (`--displacement-eps` collapses A.5.b to A.5.a).
- `dual_only_noredist`'s KE tail (0.111) is the cleanest decay of all
  configs — if a future lane fixes the wave-launch transient (linf 1.39),
  redistribution could plausibly be turned OFF for an even more physical
  benchmark.
- The `results/score.json` `inputs.beta=25.0/omega=12.9099` come from
  `lamb_damping_rate`/`rayleigh_frequency` (rho_outer-corrected), NOT the
  `_params.beta_2d=43.75`; the old baseline JSON recorded the latter era.
