# 10 - Campaign Closure (wf6, 2026-07-30): Lanes F, G, H

> Sources: debug_session/laneF-H logs, debugging_plan.md status entries 2026-07-30, wrap-up verifier report | Written: 2026-07-30 by the wf6 wrap-up agent | Covers: the three next-lane options from the laneE guidance (dam break, 3D inflation gap, 2D over-decay); laneE (wf5, 2026-07-29) is summarized in the index and in debugging_plan.md

Three lanes ran: **F landed (major)**, **G partial (major)**, **H landed
(major)**. No baseline moved; every pinned number is bit-identical.
Session edits are still **working-tree only in BOTH repos** (ddgclib +
the hyperct sibling checkout; wf6 itself made no hyperct edits) -
**commit first**, then verify: fast suite **881 P / 0 F**, floor
battery **23 P**, a5b **2 P**, hyperct **336 P**.

Final battery at close (ddg env, repo root): fast suite 881 passed / 0
failed (12 skipped, 17 deselected, 2 xfailed); floor battery 23 passed;
a5b 2 passed; hyperct 336 passed (same 39 pre-existing benchmark-fixture
errors); static_droplet_2D summary 1.1847162859108737e-03 / mass 0.0;
oscillating_droplet_2D l2 0.17479361640597058 / tail 0.9998967874595965 /
mass 2.4056717879332966e-14; oscillating_droplet_3D l2
0.24811340819647862 / tail 0.08409976059818802 / mass
1.905002320272536e-14 / R_max_peak 0.010790236105250779.

## Lane F - dam break unstuck + remap proven where reconnection fires ([log](debug_session/laneF-dam-break-unstick.md)) [landed]

The collapse actually runs for the first time, and the laneD
conservative remap got its first proof on a case that NEEDS
reconnection.

- Root solver fix: the callable branch of `_do_retopologize` silently
  dropped 8 retopology kwargs (incl. the dam break's dead
  `skip_triangulation=True`); now forwarded by declared name, with
  explicit partial bindings taking precedence (6 new forwarding tests).
- Case fixes: hydrostatic per-phase mass preload IC (flat-at-P_atm IC
  can never develop a head under structure-frozen redistribution; dam
  face now released at a_x = +39.0 m/s^2), `col_h = a` headspace
  geometry, alpha_art 2.0 -> 0.3, t_end 0.02 -> 0.2 s.
- A/B WIN: at alpha 0.3/0.5 plain per-step Delaunay blows up at its
  FIRST reconnection event (KE x28 in one step) while remap ON absorbs
  6/2 reconnection steps and completes the horizon; at alpha 0.1 ON
  absorbs 28 flip-steps vs OFF dying at its first (1.74x survival).
  Case default shipped: per-step Delaunay + `retopo_remap='conservative'`.
- Shipped runner (reproduced exactly at verification): KE_liq peak
  1.0369e-3 J @ t=0.0506 decaying to 6.3164e-4, |u|max 0.107 m/s,
  145 verts / 9 interface at setup, 1585 steps, 159 snapshots, no
  aborts, mp4 regenerated. Probe-side (incl. interface vertices):
  KEpk 2.20e-3, front +18.1 mm, mass 6.2e-15, 0 genuine clips.
- Remaining blocker (localized, own lane): air sliver-cell F/m
  ejection at reconnection - bounds alpha_art <= 0.2 (refine 3), all
  refine-4 configs, horizons >= 0.38 s at alpha 0.5.

## Lane G - 3D inflation gap: diagnosis closed, no flip ([log](debug_session/laneG-3d-inflation-gap.md)) [partial]

- The "smooth ~0.8% inflation" is a volume-neutral cube-symmetry SHAPE
  mode of the cube-sphere interface (the 6 valence-8 face centers ARE
  R_max; droplet volume constant to 3e-4) plus a per-step
  mass-redistribution pump under dual_only.
- YL-preload suspect: real in sign (discrete LSQ jump 9.1313 vs
  analytic 10.0; max|F_net| = the pinned 3D floor 6.0153e-05) but a
  measured NO-OP as a fix (trajectory unchanged to 0.03% R0).
- O(h^2)-truncation prediction FAILS: bump +9.1% / +2.0% / +4.1% R0 at
  droplet refine 1/2/3 - non-convergent; the deficit is carried by the
  18 scale-invariant cube-sphere special vertices.
- The pinned l2 0.24811 is a CANCELLATION: redistribution OFF removes
  the R_max overshoot and cleans every physics channel (tail 0.01009,
  mass 0.0 exact) but scores l2 0.26636 (+7.4%) - rejected by the flip
  rules. WIN target (summary < 0.1) measured unattainable at refine
  2/2 by any configuration/preload/stencil lever.
- Remap A/B re-decision: precondition "gap closed" unmet, so the laneE
  DO-NOT stands untouched; `retopo_policy_3d='dual_only'` and
  `baseline_oscillation_3d.json` unchanged (score.json now
  self-describing: refinement_outer=2 / refinement_droplet=2 /
  retopo_policy='dual_only').
- Why [partial]: the gap itself is NOT fixed (real levers recorded:
  icosphere-class interface triangulation; pressure-side dual-measure
  rework); the 2/3 remap leg is detached to scratchpad
  (`wf6/laneG/dyn_full_rd3_remap.json`); the log carries a leftover
  duplicate "(to be completed)" heading after the completed
  re-decision section.

## Lane H - 2D over-decay attributed + reduced (opt-in) ([log](debug_session/laneH-2d-over-decay.md)) [landed]

- Attribution (measured before fixing; energy budget closes to 9e-22
  J/step): the ~2.5x amplitude over-decay is the every-call
  pressure-preserving mass redistribution ERASING each step's local
  EOS compression response - not the 32-gon curvature bias, not the
  projection's KE bookkeeping (non-lever, -2.2e-9 J). Dose-response:
  dual_only cadence N=5 collapses l2 0.17857 -> 0.0363 with the late
  amplitude rate landing at 6.6-6.7 1/s vs the exact two-fluid 6.83.
- Fix shipped as tested opt-in: `projection_every=N` in
  `_retopologize_multiphase` with strain-advanced off-cadence
  snapshots (`evolve_snapshot_local_strain`); the raw eos(m/dual_vol)
  variant is a measured dead end (l2 1.31-1.34, guarded by test).
- delaunay_remap + projection_every=2 scores l2 **0.03795682994323827**
  (-78%) and lands on laneC's exact two-fluid reference to ~2%
  (l2_2f 0.02193, KE-shape correlation 0.994).
- NO default flip: tail <= 1.0 is measured unattainable for faithful
  physics on this horizon (the exact two-fluid KE peaks at t* =
  0.1234 s > t_end = 0.1143 s; analytic tail 1.66) - the pinned
  default passes it only via the every-call erasure. Defaults,
  baselines, and every pin unchanged; 5 new tests.
- Residual inside l2 ~ 0.04: the cadence-insensitive l=0 bump
  (+0.6% R0, polygon-area/chord-vs-arc territory).

## Verification caveats (wrap-up verifier, confirmed)

- The independent 3D droplet rerun (laneG check) was still executing
  at report time (no abort, EOS warnings only). The pin verification
  rests on the on-disk `results_3d/score.json` being bit-identical to
  `baselines/baseline_oscillation_3d.json` on every numeric key and
  carrying the new self-describing keys - not on that rerun completing.
- laneF cosmetic only: the log's shipped-runner sentence says "mass
  ~5e-15" where its own A/B table pins 6.2e-15 (probe-side, incl.
  interface vertices); "front +18.1 mm", "n_iface 9 constant" and "0
  genuine clips" are probe-driver instrumentation numbers, not printed
  by the shipped runner. Every number the runner does print reproduced
  exactly.
- debugging_plan.md's stale 2026-07-29 top prompt (its options 2/3/4
  ARE lanes F/G/H) was superseded during closure by the 2026-07-30
  block at the head of the status log.

## What remains open (next-lane options, full detail in debugging_plan.md's 2026-07-30 prompt)

1. **2D score recalibration** (two-fluid reference + peak-aware tail
   window), THEN revisit adoption of delaunay_remap +
   projection_every=2 with the endurance battery.
2. **Air sliver-cell F/m ejection at reconnection** - the dam break's
   remaining blocker (laneF §4).
3. **Icosphere-class isotropic 3D interface triangulation** - removes
   the 18 special vertices behind the 3D bump and floor (re-pins the
   3D floors; own lane).
4. **Redistribution-projection + pressure-side dual-measure rework**
   (shared 2D/3D suspect: 50/50 `edge_phase_area_fractions` heuristic,
   dual-face closure residuals up to 2.6% at interface/outer vertices).

Plus: commit both working trees FIRST (verify 881/23/2/336 after
staging), and observe the consolidated wf6 DO-NOTs in
debugging_plan.md (no tail<=1.0 chasing on the current 2D score, no
raw-recompute off-cadence snapshots, no redistribution skipping under
live reconnection, no 3D refinement flips or scalar YL preload
retries, no reading 3D l2 0.24811 without sign decomposition, no flat
pressure ICs for gravity cases under redistribute_mass, no
`mass_conserving_merge` without per-phase ledger handling).
