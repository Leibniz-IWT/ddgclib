# 09 — Continuation Session (wf4): The Four 2026-07-03 Next-Lane Options, All Landed
> Sources: debug_session/laneA–laneD logs, debugging_plan.md status entries 2026-07-29, cross-checked against the measurement batteries in each lane log | Written: 2026-07-29 by the wf4 wrap-up agent | Executed session date: 2026-07-29 (continuation of the 2026-07-03 plan in [08](08_debug_session_2026-07-02.md) §5)

The 2026-07-03 "Recommended next prompt" listed four orthogonal lane options.
This session ran all four in parallel; **all four landed with major progress**.
All edits remain **working-tree only in BOTH repos** (ddgclib + hyperct sibling
checkout) — commit first; the lane logs are the authoritative changed-file lists.

Final battery state: fast suite **865 passed / 0 failed** (833 → 834 A → 847 B
→ 862 C → 865 D; +32 new tests), floor battery **17 P** (14 pinned + 3 new
remap tests), a5b long-run regression 2 P, hyperct **336 P** (+2), every pinned
2D number **bit-identical to 16 digits** across all four lanes (equil
1.1847162859108737e-03; osc l2 0.1785660454150319 / linf 0.32842491275938274 /
tail 0.9992507831101141 / mass_drift 2.676988154993443e-14; 2D floors
2.3748568e-03 / 2.2716938e-03).

## Lane A — 3D exact-dual-volume switch + canonical Delaunay order ([log](debug_session/laneA-3d-exact-dual-volume-switch.md))

The last confirmed-high audit item (dual-volume-3d) is closed in production.
The three `NOTE(lane3-dual-volume)` switch points were flipped together
(`dual_volume` dim==3, `cache_dual_volumes` dim in (2,3), integrator step-5b
preference), and hyperct's `connect_and_cache_simplices` now canonicalizes the
3D qhull input order (lexsort + index remap) so static-cloud retopo is
idempotent.

- 3D floor plateau: 7.3768e-05 (fan) → 7.616853911101026e-05 (switch only,
  reproduces lane 3's measured target exactly — the settle artifact) →
  **7.274172178727318e-05 shipped (−1.4%, plateau from step 1, settle step
  eliminated)**. Step0 6.0153201139652905e-05 bit-unchanged. F rel spread
  6.05e-12 over steps 1..100; volume spread 0.0.
- Re-pins: `EXPECTED_PLATEAU_MAXF` 7.274172e-05, `SETTLE_STEPS` 2 → 1,
  `A5B_3D_PEAK`/`END`. Strict xfail `test_partition_of_unity_3d_jittered_production`
  flipped to passing (3D duals tile to 1e-12).
- Open footgun (documented, small): `assign_simplex_phases` auto-Delaunay on
  structured-connectivity meshes leaves a mixed mesh state.

## Lane B — 3D score harness + first 3D baseline + 3D retopo A/B ([log](debug_session/laneB-3d-score-harness.md))

The last unmeasurable dynamic case (06 §1.6) is now scored and
regression-locked. NEW `oscillation_score_3d` (R_max vs Rayleigh–Lamb
envelope, apex score via `polar_axis='z'`, `boundary_saturation` tripwire,
dual-vol/interface bookkeeping fields), runner writes `results_3d/score.json`,
first baseline checked in.

- FIRST 3D baseline (per-step Delaunay, 872 steps, refine 2/2): l2
  **1.5244561707801316**, linf 2.2655, tail 0.47067, mass 3.87e-14,
  R_max_peak 0.011397, **boundary_saturation False** — the historical
  R_max-at-~5·R0 artefact no longer reproduces.
- A/B → **`retopo_policy_3d = 'dual_only'` is the new 3D runner default**:
  l2 **0.24811340819647862 (−83.7%)**, linf 0.5993, tail 0.08410, mass
  1.91e-14, dual_vol_drift_post 1.07e-4 vs 5.08e-3 (47x steadier),
  n_interface 98 constant both. Lane-5's bookkeeping caveat was verified
  clean BEFORE trusting the A/B (n_bV 96 both, identical step-0 jump
  0.357050, exact volumes engaged both, mass ~2e-15).
- Remaining 0.248 is a smooth physics gap (droplet inflation ~0.8% of R0,
  curvature-side suspects) — do not chase l2 → 0.

## Lane C — exact two-fluid reference: lane-5 hypothesis REFUTED ([log](debug_session/laneC-two-fluid-reference.md))

Validation lane, no solver changes. Full 2D two-fluid viscous normal-mode
dispersion relation + energy-method closed form + exact IVP temporal factor in
`src/_analytical.py`; both scores now reported side by side in `score.json`;
`diag_series.json` dumped for rerun-free rescoring. 15 new tests (limits
recover single-fluid Lamb to 1e-14).

- Exact least-damped mode (rho 800/1000, mu 0.5/0.1, gamma 0.05, R0 0.01,
  l=2): s = **−6.8325510 ± 5.4707015i** (weakly oscillatory, not overdamped;
  omega_eff 8.7528, beta 6.8326). Energy-method beta_2f 17.778 (diagnostic
  only — underestimates interfacial-layer damping, do not promote).
- l2 vs two-fluid reference **0.1885496661389148 (+5.6%)** vs pinned
  single-fluid 0.1785660454150319 (bit-identical) — the corrected reference
  explains essentially NONE of the residual. Confined-wall probe at
  r_w = 5R0 shifts beta ~0.1% — negligible.
- Decay rates [1/s]: measured KE tail 6.53 vs 1f 7.183 / 2f-exact 13.665 /
  2f-energy 11.111; sim temporal at t_end 0.390 vs 0.706–0.719 for every
  defensible reference. **The ~5-10% over-decay is solver-side**; live
  suspects: per-step mass-redistribution projection, O(h) 32-gon curvature
  bias. Single-fluid metric stays the regression gate.

## Lane D — conservative retopology remap: the large-deformation structural fix ([log](debug_session/laneD-conservative-retopo-remap.md))

Opt-in `retopo_remap='conservative'` in `_retopologize_multiphase` (default
OFF everywhere — every baseline bit-identical). Three-part closure: bit-exact
pressure-structure restore across each rebuild, per-phase EOS volume gauge
(`MultiphaseSystem.vol_corr`) so redistribution cannot bounce the artifact
back, and a level anchor pinning per-phase pressure to strain vs
connectivity-artifact-corrected volume targets (p_ref pattern).

- Full 1839-step 2D droplet with per-step FULL Delaunay active: KE_max
  4.0716e-2 (growing) → **8.31727774219008e-7 J (48,954x, onto the physical
  envelope)**; l2 0.48992 → **0.17479361640597058** (beats dual_only's
  0.17857); tail 1.72505 → **0.9998967874595965** (< 1.2 win condition);
  t@KE_max 0.0560 (analytical 0.0598); mass 2.41e-14. One-call pressure
  neutrality 3.8e-13 Pa (vs 1.7e2 Pa without).
- Mechanism probe-confirmed first: outer-phase redistribution scale noise
  1.43e-2 under Delaunay vs 1.44e-5 dual_only (~1000x; K_o=1000 → up to
  14 Pa jolts vs the 5 Pa Laplace jump). Dead ends measured, do not retry:
  restore-only (bounce corr +0.9998), incremental level (far-field runaway
  p_far −0.06 → −9.56 Pa, 100% of residual KE far-field).
- Discovery: `dam_break_2D.py`'s `skip_triangulation=True` is silently
  ignored — the shipped case runs per-step Delaunay (harmless today only
  because the collapse is stalled at smoke parameters).

## Next

In priority order (full detail in debugging_plan.md's 2026-07-29 recommended
prompt): (1) **commit both working trees**, then verify 865/17/2/336; (2) flip
the 2D droplet default to delaunay + remap after long-run/refine-4/c_s=5
proving (would re-pin `baseline_oscillation.json`); (3) fix the dam-break dead
flag and unstick the collapse, then A/B the remap where reconnection fires;
(4) 3D curvature-side inflation gap behind summary 0.248; (5) 2D over-decay
suspects (redistribution projection, 32-gon curvature bias) — the
reference-error candidate is closed.
