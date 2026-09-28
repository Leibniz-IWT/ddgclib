# Stabilisation Roadmap for `ddgclib` — From Equilibrium Benchmarks to Stable Dynamic Multiphase

## Status log

- **2026-07-30 — Lane H shipped (MAJOR PROGRESS: the 2D over-decay is ATTRIBUTED and fixed as an opt-in; no default flip, every pin bit-identical): the ~2.5x l=2 amplitude over-decay behind the pinned l2 0.17479 is the every-call pressure-preserving mass redistribution ERASING each step's local EOS compression response (suspect (i)) — with the new opt-in `projection_every=N` cadence in `_retopologize_multiphase`, the SAME delaunay_remap default configuration scores l2 0.03796 (−78%) and lands on laneC's exact two-fluid reference to ~2% (l2_two_fluid 0.02193, KE-shape corr 0.994 with e^(−2βt)sin²(ω_d t)) with reconnection still active every step.**

  Attribution per the brief, all measured before fixing (instrumented driver bit-exactly reproduces both pins; per-step energy budget closes to 9e-22 J): (a) the projection's own KE jump is a NON-LEVER (−2.2e-9 J cumulative vs 6.85e-6 J surface-energy turnover, 0.03%; T_trunc +3.0e-9 J) — the loss is in the FORCING: with pressure STRUCTURE frozen (laneD §1.1) the interface relaxes without local elastic push-back, releasing surface energy 1.6x too fast (6.85e-6 vs 4.30e-6 J) with W_visc draining the excess (W_st = −ΔE_surf to 0.06% — the FTC force is the exact PL-perimeter gradient). (b) dose-response: dual_only with the redistribution fired every N steps collapses l2 0.17857 → 0.0363 (N=5, saturated by N=5 vs N=20; late amplitude rate 6.6-6.7 1/s = the exact two-fluid 6.83); the pinned run's supra-exponential rate ramp 2.5→16.1 1/s is the erasure integrating. (c) end-members: dual_only_noredist reproduces lane5 exactly (l2 0.49503; acoustic launch KE 7.7e-4 J pumped by integrator truncation +8.4e-4 J — why "just turn it off" never worked); delaunay_remap_noredist is structurally forbidden (code ValueError; = laneD restore-only dead end). Suspect (ii) 32-gon bias is NOT the amplitude driver (no room at l2_2f 0.022); its residual fingerprint is the cadence-insensitive l=0 bump (+0.6% R0 ≈ the 32-gon polygon-area offset +0.32%, the 2D sibling of laneG's shape mode) which dominates the remaining l2 ≈ 0.04 via apex cancellation — arc/bulge wiring deferred (own lane, would re-pin 2D floors).

  | full-run config | l2 (pinned metric) | l2 two-fluid | tail | KE_max [J] |
  |---|---|---|---|---|
  | delaunay_remap (pinned default) | **0.17479361640597058** | 0.18461 | 0.99990 | 8.32e-07 @ t=0.056 |
  | + projection_every=2 (opt-in fix) | **0.03795682994323827** | **0.02193** | 1.3973 | 1.15e-06 @ t_end (two-fluid shape) |
  | dual_only + N=5 (probe) | 0.03632 | 0.02025 | 1.44875 | 1.16e-06 @ t_end |

  **No adoption (rule applied):** every clean candidate fails better-tail — MEASURED unattainable for faithful physics on this horizon: the exact two-fluid KE peaks at t*=0.1234 s > t_end=0.1143 (analytic tail 1.66; sim 1.38-1.40); the pinned default passes tail<=1.0 only because the every-call projection reshapes KE onto the single-fluid envelope. dual_only N=2/3 pass both numbers (l2 0.031, tail 0.03-0.10) but via a T_trunc-pumped acoustic channel (KE 16-64x mode level) + connectivity freeze — fails clean-channels and laneE generality. Recommended next lane: recalibrate the 2D score (two-fluid reference + tail window; laneC successor), then revisit adoption of delaunay_remap+projection_every=2 with endurance. Implementation note (measured dead end, guarded by a frozen-positions neutrality test): off-cadence remap snapshots MUST be the pre-call field advanced by local Lagrangian strain (`evolve_snapshot_local_strain`), NOT the raw eos(m/dual_vol) recompute — the raw form drops the restore/anchor level (8.8 Pa/call jolt; full-horizon l2 1.31-1.34). New DO-NOTs: no tail<=1.0 chasing on the current 2D score with a faithful solver; no raw-recompute off-cadence snapshots; no redistribution skipping under live reconnection (now code-enforced); no velocity-rescaling "fix" of the redistribution KE jump (non-lever).

  Battery: floor **23 P** (18 + 5 new `TestProjectionCadence2D`, 28.1s, no re-pins); fast suite **881 P / 0 F** (+5, 71.8s); a5b **2 P**; equil **1.1847162859108737e-03** / mass 0.0; osc 2D **bit-identical to the pin** (l2 0.17479361640597058 / tail 0.9998967874595965 / mass 2.4056717879332966e-14); osc 3D **bit-identical** (l2 0.24811340819647862); hyperct **336 P** (no edits). Changed files: `_integrators_dynamic.py` (`projection_every`), `operators/mass_redistribution.py` (`evolve_snapshot_local_strain`), `tests/test_case_oscillating_droplet.py` (+5), `src/_params.py` (comment only). Full evidence: [`docs_temp/debug_session/laneH-2d-over-decay.md`](docs_temp/debug_session/laneH-2d-over-decay.md).

  **Recommended next prompt** (updated 2026-07-30, supersedes the 2026-07-29 prompt below — its options 2, 3 and 4 were executed this session as lanes F, G and H):

  > The 2026-07-30 wf6 session landed lanes F (dam break unstuck — hydrostatic mass-preload IC, `col_h = a` headspace, alpha_art 0.3, t_end 0.2; `_do_retopologize` kwarg-forwarding fixed for callable retopo_fns; the laneD conservative remap proven load-bearing where reconnection fires — case default now delaunay + `retopo_remap='conservative'`), G (3D inflation gap DIAGNOSIS CLOSED, no flip: the "inflation" is the cube-sphere discrete-equilibrium SHAPE mode + a redistribution pump, and the pinned 3D l2 0.24811 is a bump/over-decay CANCELLATION; `retopo_policy_3d='dual_only'` and every pin stand) and H (2D over-decay ATTRIBUTED to the every-call pressure-projection erasing the local EOS compression response; opt-in `projection_every=N` lands the default config on the exact two-fluid reference — l2 0.03796 vs pin 0.17479 — with NO default flip because tail<=1.0 is measured unattainable for faithful physics on this horizon). Read the three lane logs in `docs_temp/debug_session/lane{F,G,H}-*.md` first. Everything is STILL working-tree only in BOTH repos: **commit first** (verify after staging: fast suite 881 P / 0 F, floor battery 23 P, a5b 2 P, hyperct 336 P).
  >
  > **Next lane options (orthogonal, pick by user preference)**:
  >
  > 1. **Recalibrate the 2D score** (validation-side, laneC successor): score against the exact two-fluid reference with a tail window aware of its KE peak (t* = 0.1234 s > t_end = 0.1143 s), THEN revisit adoption of `delaunay_remap + projection_every=2` with the endurance battery (laneH §3 — it already scores l2 0.038 / l2_2f 0.022 but fails the current tail gate for physics reasons).
  > 2. **Air sliver-cell F/m ejection at reconnection** — the dam break's remaining blocker (laneF §4): sliver-aware mass/force handling, or interface-preserving adaptive remesh + per-phase-conserving vertex merge. Bounds today: alpha_art <= 0.2 (refine 3), all refine-4 configs, horizons >= 0.38 s at alpha 0.5.
  > 3. **Icosphere-class isotropic 3D interface triangulation** (laneG lever (a)): removes the 18 scale-invariant cube-sphere special vertices that pin the 3D floor and carry the bump; re-pins the 3D floors — a full lane of its own.
  > 4. **Redistribution-projection + pressure-side dual-measure rework** (laneG lever (b), shared with the 2D residual): the 50/50 `edge_phase_area_fractions` heuristic (sign-flipping bias under refinement) and the dual-face closure residuals (up to 2.6% at interface/outer vertices); co-evaluate against the 3D l2 cancellation.
  >
  > Do NOT (new this session, all measured): chase tail<=1.0 on the current 2D score with a faithful solver; build off-cadence remap snapshots from the raw `eos(m/dual_vol)` recompute (use `evolve_snapshot_local_strain`); skip redistribution under live Delaunay reconnection (code-enforced); velocity-rescale the redistribution KE jump (non-lever); flip the 3D droplet refinement (2/3 and 3/3 measured worse); re-try scalar YL preloads for the 3D shape drift; read the 3D l2 0.24811 as a pure inflation gap without sign decomposition; initialise gravity-driven multiphase cases with flat pressure under `redistribute_mass=True` (preload the IC); wire `mass_conserving_merge` into multiphase paths without per-phase `m_phase` ledger handling.

- **2026-07-30 — Lane G shipped (diagnosis lane, no solver or default changes; every pin bit-identical): the 3D droplet-inflation gap behind dual_only summary 0.24811 is CLOSED as a diagnosis and every candidate fix in the brief was measured and rejected by the flip rules — the "inflation" is the cube-symmetry discrete-equilibrium SHAPE mode of the cube-sphere interface (the 6 valence-8 face-center vertices ARE R_max; droplet volume constant to 3e-4), and the pinned l2 0.24811 is a CANCELLATION of that outward bump against a genuine early over-decay (the 3D sibling of laneC's 2D over-decay).**

  Discrimination per the brief, all measured on the frozen eps=0 droplet first: (ii) YL-preload mismatch is REAL in sign — discrete-consistent jump 9.1313 Pa (area-weighted LSQ; net-radial-neutral 9.7574) vs the analytic 10.0 preload, net outward residual, and max|F_net| = **6.0153201139652905e-05 = exactly the pinned 3D floor step0** (the floor force IS this imbalance) — but a scalar discrete preload is a measured NO-OP (trajectory unchanged to 0.03% R0; the l=0 error self-corrects through the EOS volume constraint). (i) O(h^2)-truncation FAILS its refinement prediction: eps=0 bump at matched t=0.0563 is +9.1% R0 (droplet refine 1) -> +2.0% (2) -> **+4.1% and growing (3)** — NON-convergent; outer refinement is a no-op at the interface (3/3 static balance identical to 2/3 to 4 digits); stencil variants are closed ('stokes' == 'integrated' to 1e-14) and the cotan operator is already EXACTLY anti-parallel to the PL volume gradient (cos = -1.0 at every vertex), so no consistency pairing puts the coarse sphere in equilibrium — the per-vertex imbalance is scale-invariant at the 18 cube-sphere special vertices (val-4 edge-midpoints dp* 23.28/23.85 at 2/2 and 2/3, val-8 face centers 7.95/6.07 vs 10). "Look deeper" branch (fired, per the brief): the eps=0 relaxation runs along a marginal energy valley (gamma*dA = dp*dV to 0.4% at 2/2) and at 2/3 the discrete energy is PUMPED monotonically (dE +8.11e-8 J/0.095 s with redistribution, **+5.66e-8 J without** — at refine 3 the dominant pump is the volumetric-dual pressure side itself: the 50/50 interface-edge `edge_phase_area_fractions` heuristic biases sign-flippingly with refinement, val-6 dp* 9.22 -> 10.61 while the exact-PL pairing converges 10.15 -> 10.05, plus up-to-2.6% dual-face closure residuals at interface/outer vertices vs machine-clean bulk).

  | full-horizon scored config (dual_only unless noted) | l2 | tail | mass | R_max_peak |
  |---|---|---|---|---|
  | 2/2 redistribution ON (pinned) | **0.24811340819647862** | 0.08410 | 1.9e-14 | 0.010790 (overshoot) |
  | 2/2 redistribution OFF | 0.26636475750102395 | **0.01009** | **0.0 exact** | **0.0105 (no overshoot)** |
  | 2/3 redistribution ON | 0.46645348342458526 | 0.01266 | 6.9e-14 | 0.010775 |
  | 2/3 redistribution OFF | 0.37204702775924850 | 0.00177 | 0.0 | 0.010544 (ends +4.2% R0) |

  Redistribution OFF under dual_only kills the R_max overshoot entirely and cleans every physics channel (it carries ~half the 2/2 bump; there is no reconnection for it to fix under frozen connectivity) but RAISES the pinned l2 by 7.4% because the bump was cancelling the over-decay — rejected by the better-l2 rule and the <=5% regression rule. Task WIN (summary < 0.1) NOT met — measured impossible at refine 2/2 by any configuration/preload/stencil lever: the static artifact forces (~6e-5 N/vertex) are ~5x the physical l=2 restoring force at this resolution and R_max rides a special vertex. Remap A/B re-decision: precondition "inflation gap closed" unmet -> laneE's rejection STANDS untouched (`retopo_policy_3d='dual_only'`, no re-pin; the 2/3 remap leg is preserved in scratchpad for the record). **New DO-NOTs (measured):** do not flip the 3D droplet runner refinement (2/3 and 3/3 are worse; the bump does not O(h^2)-shrink on this mesh family); do not re-try scalar YL preloads for shape drift; do not read the 3D l2 0.24811 as a pure inflation gap (sign-decompose first — removing the bump RAISES it). **Real levers recorded for future lanes:** (a) icosphere-class isotropic interface triangulation (removes the 18 scale-invariant special vertices; re-pins the 3D floors — own lane); (b) redistribution-projection + pressure-side dual-measure rework (50/50 fraction heuristic, p_ij closure at distorted tets) — shared suspects with the 2D over-decay.

  Battery: floor **18 P** (25.4s, no re-pins); fast suite **876 P / 0 F** (70.1s, identical pre/post); equil **1.1847162859108737e-03** / mass 0.0; osc 2D l2 **0.17479361640597058** / tail 0.9998967874595965 / mass 2.4056717879332966e-14; osc 3D official runner **bit-identical on every numeric key** to `baseline_oscillation_3d.json` (l2 0.24811340819647862, R_max_peak 0.010790236105250779, mass 1.905002320272536e-14) with new self-describing `refinement_*`/`retopo_policy` keys in score.json; hyperct **336 P** (same 39 pre-existing benchmark errors). Changed files: comments + score self-description only (`oscillating_droplet_3D.py`, `src/_setup.py`); no solver/operator/params/baseline/test/hyperct edits. Full evidence: [`docs_temp/debug_session/laneG-3d-inflation-gap.md`](docs_temp/debug_session/laneG-3d-inflation-gap.md).

- **2026-07-30 — Lane F shipped (MAJOR PROGRESS): dam break unstuck — the collapse actually runs for the first time — and the laneD conservative remap proven load-bearing on a case that NEEDS reconnection: at alpha_art 0.3/0.5 plain per-step Delaunay BLOWS UP at its FIRST reconnection event (KE x28 in one step) while remap ON absorbs 6/2 reconnection events and survives the full 0.2 s horizon; at alpha 0.1 remap absorbs 28 flip-steps vs OFF dying at its first (1.74x survival). Case default now delaunay + `retopo_remap='conservative'`.**

  Three-part diagnosis of the laneD stall (|u|max <= 0.03 at t_end x5), all measured: (1) fundamental — under `redistribute_mass=True` the pressure STRUCTURE is re-imposed every step (laneD §1.1), so the case's flat-at-P_atm IC could NEVER develop the hydrostatic head: after 0.098 s under gravity the liquid column read a uniform +0.121 Pa gauge instead of ~981 Pa, dam-face |a_x| <= 0.7 m/s^2 vs ~70 needed — fixed with a hydrostatic per-phase mass preload (droplet YL-preload pattern; face now released at a_x = +39 m/s^2 with vertical balance 0.025); (2) geometry — `col_h = 2a == H`, the "column" filled the tank lid-to-floor with no headspace (docstring diagram violated), top liquid row frozen INTO the lid → `col_h = a`; (3) alpha_art 2.0 = 390 Pa s effective viscosity (Re 0.18, creeping) + t_end 0.02 = 1/8 of the gravity ramp → alpha_art 0.3, t_end 0.2 (~2.8 t_ref) from the sweep (0.3 = smallest surviving value; 0.2 aborts t=0.16, 0.1 aborts t=0.094). Also fixed at the root: `_do_retopologize` silently DROPPED `skip_triangulation` (+7 more kwargs: boundary_filter, merge_cdist, backend, periodic_axes, domain_bounds, pressure_model, redistribute_mass) for CALLABLE retopo_fns — now forwarded when declared by name, never into `**kw` sinks, and never overriding partial bindings (droplet policy partials protected; 6 new forwarding tests). Shipped runner: KE_liq rises to 1.04e-3 @ t=0.051 then decays, |u|max 0.107 m/s, toe runs out to x≈0.069 (classic slump profile, interface ring 9 constant), mass ~5e-15, 159 snapshots, `fig/dam_break_2D.mp4` regenerated.

  | battery item | result |
  |---|---|
  | floor battery (-m "") | **18 P** — all pins untouched |
  | fast suite | **876 P / 0 F**, 12 skipped, 17 deselected, 2 xfailed (= 866 + 6 forwarding + 4 dam-break) |
  | static_droplet_2D | **1.1847162859108737e-03**, mass 0.0 — bit-identical |
  | oscillating_droplet_2D | l2 **0.17479361640597058** / tail **0.9998967874595965** / mass 2.4056717879332966e-14 — bit-identical |
  | oscillating_droplet_3D | l2 **0.24811340819647862** — bit-identical |
  | hyperct | untouched (no run required) |

  **Remaining blocker, precisely localized (next lane):** air sliver-cell F/m ejection — reconnection during large deformation leaves an air vertex on a near-zero dual volume with finite dual-face areas; m = rho_g·dvp ~ 1e-4 kg ⇒ a = F/m spikes ⇒ ballistic ejection (~km/s in one step, measured a gas vertex at y=0.676 = 6.7x outside the domain) ⇒ coordinate overflow ⇒ QhullError. Zero non-ballistic jumps detected — pure F/m integration, the corner-vertex defect the alpha_art crutch papers over, caught at its root. Bounds alpha_art <= 0.2 (refine 3), ALL refine-4 configs, and horizons >= 0.38 s at alpha 0.5 (remap always last-man-standing). Because of it the |u|max = O(u_ref) gate is met only fractionally (0.11 of u_ref 0.99 at the shipped default; 0.36 u_ref at alpha 0.1 pre-abort). Candidate fixes: sliver-aware mass/force floor, or adaptive interface-preserving remesh + per-phase-conserving merge (`mass_conserving_merge` does NOT merge `m_phase` — do not wire it in as-is, it breaks the per-phase ledger). Bonus finding: laneD §2.3's unexplained 2.3x KE-plateau does not reproduce on the fixed case (5.85e-5 vs 5.76e-5 at alpha 2.0) — it was the level anchor holding the phase mean in the structurally-frozen flat-pressure state; resolved. New DO-NOTs from this lane: do not initialise gravity-driven multiphase cases with flat pressure under `redistribute_mass=True` (structure cannot develop — preload the IC); do not wire `mass_conserving_merge` into multiphase paths without per-phase ledger handling. Full evidence: [`docs_temp/debug_session/laneF-dam-break-unstick.md`](docs_temp/debug_session/laneF-dam-break-unstick.md).

- **2026-07-29 — Lane E shipped (adoption): 2D droplet runner default FLIPPED `dual_only` -> `delaunay_remap` (per-step full Delaunay + lane-D conservative remap) on long-run proof; 3D default STAYS `dual_only` — the 3D remap A/B measured WORSE on the trajectory metric (l2 1.87348 vs 0.24811) and is rejected by the better-l2-AND-tail rule. `baseline_oscillation.json` re-pinned l2 0.1785660454150319 -> 0.17479361640597058 / tail 0.9992507831101141 -> 0.9998967874595965; new 200-step remap endurance guard test.**

  Long-run proofs (drivers validated by bit-exact reproduction of every pinned number first): (1a) 2D remap STATIC endurance, 2000 steps u=0 under per-step full Delaunay — plateau = the pinned 2.2716938e-03 floor with rel spread **1.34e-15**, mass 2.35e-15, interface 32 constant; (1b) 2D remap DYNAMIC endurance at 2x horizon (3678 steps, 10 damping times) — KE_max 8.31727774219008e-07 unchanged, late-growth ratio exactly 1.0, KE decays BELOW the analytical envelope (tail fit 13.5 vs 7.18 1/s = the known laneC over-decay), mass 4.52e-14 (the 2x-horizon l2 0.3004 is that over-decay integrating — never compare it to the standard-horizon pin); (1c) 3D dual_only static endurance, 2000 steps — own plateau **6.836475840821e-05** (6.0% below the pinned Delaunay plateau 7.274172e-05, which is setup-path and unaffected), rel spread 8.96e-14, mass 8.9e-16; (1d) lane-D remap verified dim-agnostic by code reading, then measured in 3D: static clean (plateau 6.82996e-05, spread 5.0e-14) but the scored 872-step A/B REJECTS it — it kills the KE pump (KE_max 6.14e-8 vs dual_only 1.74e-6 J) yet un-opposes the known droplet-inflation physics gap (R_max_peak 0.011559, l2 1.87348, dual-vol churn = plain Delaunay 5.08e-3), and the laneD §2.6.5 3D bookkeeping caveat is real (droplet vol_corr gauge 0.9696 in 20 steps).

  2D flip rationale: metrics at least as good (l2 0.17479 vs 0.17857; tail 0.99990 vs 0.99925, both healthy), long-run channels machine-clean, wall cost 1.61x (205.8 vs 127.5 s), and GENERALITY decisive — the default now keeps global reconnection honestly active, the code path that changing-topology cases must use; `dual_only` stays as opt-in. Recorded caveat: the outer phase's TaitMurnaghan pressure() saturates transiently on churned corner cells pre-restore (~5.7e4 clips/full run, values overwritten by the restore; all outcome channels clean; 3D per-step Delaunay — the OLD 3D default — clips too, 2D dual_only never does). Dam break: not re-run — solver bit-identical to laneD and the case ignores `retopo_policy_2d`, so laneD §2.3 stands (collapse stalled at smoke horizon; no evidence yet the alpha_art crutch can shrink under remap, none against).

  | battery item | result |
  |---|---|
  | floor battery (-m "") | **18 P** in 24.63s (17 + new `TestDelaunayRemapEndurance2D`); all four pinned floors untouched |
  | fast suite | **866 P / 0 F**, 12 skipped, 17 deselected, 2 xfailed in 62.04s (+1) |
  | a5b regression (-m "") | **2 P** in 13.94s |
  | static_droplet_2D | summary **1.1847162859108737e-03**, mass 0.0 — bit-identical |
  | oscillating_droplet_2D (NEW default) | l2 **0.17479361640597058** / linf 0.32364245955409165 / tail **0.9998967874595965** / mass 2.4056717879332966e-14 — bit-identical to the laneD run; = new baseline pin |
  | oscillating_droplet_3D (dual_only) | l2 **0.24811340819647862** — bit-identical to `baseline_oscillation_3d.json`, no re-pin |
  | hyperct | **336 P**, 40 skip, 6 xfail, same 39 pre-existing benchmark errors |

  Changed files: `src/_params.py` (policy flip + A/B evidence comments), `oscillating_droplet_2D.py` (delaunay_remap dispatch), `ddgclib/tests/test_case_oscillating_droplet.py` (envelope test re-pinned to the remap mirror: fixture l2 0.050006441230559064 -> 0.05451425932968201, tail 0.9130549583972877 -> 0.9368903708926931, L2_MAX 0.0550 -> 0.0600; + new 200-step endurance guard), `baselines/baseline_oscillation.json` (re-pin above). No solver/operator/hyperct edits. Full evidence: [`docs_temp/debug_session/laneE-adoption-defaults.md`](docs_temp/debug_session/laneE-adoption-defaults.md).

  **Next-lane guidance (updates the laneD prompt below):** option 1 (2D default flip) is DONE this lane. Remaining, in priority order: (2) fix the dam-break dead `skip_triangulation` flag and unstick the collapse, then A/B the remap where reconnection actually fires; (3) close the 3D droplet-inflation gap (curvature-side suspects) — THEN re-run the 3D remap A/B, whose KE result (6.1e-8 J) suggests it may win once the inflation is fixed; (4) 2D over-decay residual (redistribution projection, 32-gon curvature bias). New DO-NOTs from this lane: do not compare multi-horizon l2 against standard-horizon pins (over-decay integrates); do not adopt the remap in 3D before the inflation gap closes; do not read the pre-restore EOS clip_count as a health signal without splitting bookkeeping clips from genuine ones.

- **2026-07-29 — Lane D shipped (MAJOR PROGRESS): opt-in conservative retopology remap (`retopo_remap='conservative'` in `_retopologize_multiphase`) makes per-step FULL Delaunay reconnection thermodynamically neutral — the structural fix lane 5 left open for large-deformation cases that cannot freeze connectivity. On the full 1839-step 2D droplet with Delaunay active every step: KE_max 4.0716e-2 → 8.3173e-7 J (48,954x, onto the physical envelope), l2 0.48992 → 0.17479 (better than dual_only's 0.17857), tail 1.72505 → 0.99990. Defaults OFF everywhere — every baseline bit-identical.**

  Three-part closure (all in `mass_redistribution.py` / `multiphase.py` / `_integrators_dynamic.py`, threaded as an opt-in kwarg): (1) `restore_pressure_multiphase` — pressure STRUCTURE restored bit-exactly across each rebuild (one-call neutrality 3.8e-13 Pa vs 1.7e2 Pa without); (2) `MultiphaseSystem.vol_corr` per-phase EOS volume gauge (= stage-2 redistribution scale) so the next redistribution cannot bounce the artifact back (restore-only bounces: corr(stage1[n+1]−1, stage2[n]−1) = +0.9998, a measured dead end); (3) `anchor_phase_pressure_levels` — the per-phase pressure LEVEL pinned to volume strain vs connectivity-artifact-corrected volume targets (`V_tar *= V_new/V_mid` at frozen positions; p_ref pattern), because an incrementally-integrated level rectifies reconnection noise into runaway phase tension (measured: p_far −0.06 → −9.6 Pa accelerating by step 275, 100% of residual KE in the far field; gauge+restore alone plateau at KE 2.7e-2).

  | metric (full run, per-step Delaunay) | remap OFF | remap ON | dual_only ref |
  |---|---|---|---|
  | l2 / linf | 0.48992 / 0.85035 | **0.17479 / 0.32364** | 0.17857 / 0.32842 |
  | tail_growth | 1.72505 | **0.99990** | 0.99925 |
  | KE_max [J] / t@peak | 4.07e-2 growing / 0.114 | **8.3173e-7 / 0.0560** | 8.317e-7 / 0.0549 |
  | mass_drift | 1.4e-14 | 2.4e-14 | 2.7e-14 |

  **Verification.** Floor battery **17 P** (14 pinned + 3 new `TestConservativeRetopoRemap2D`; no re-pins); fast suite **865 P / 0 F** (+3); equil **1.1847e-03**; official 2D oscillation **bit-identical to 16 digits** (l2 0.1785660454150319 / tail 0.9992507831101141); no-remap Delaunay driver replica bit-identical pre/post edit; hyperct untouched. Mechanism was probe-confirmed before design: outer-phase redistribution scale noise 1.43e-2 max under Delaunay vs 1.44e-5 dual_only (~1000x), i.e. up to 14 Pa uniform jolts vs the 5 Pa Laplace jump.

  **Dam break smoke** (no case files changed): discovery — the shipped `dam_break_2D.py` `skip_triangulation=True` is silently IGNORED (`_do_retopologize` doesn't forward it to callable retopo_fns): the shipped case runs per-step full Delaunay. Harmless-by-accident today: the collapse is stalled at smoke parameters (|u|max ≤ 0.03 m/s, displacement << dx even at t_end×5), Delaunay never actually flips, so remap ON/OFF cannot be discriminated there yet (all variants: no aborts, mass ≤ 5e-15, zero EOS clips, interface stable; alpha_art 2.0→0.5 stable both ways — the crutch is orthogonal to retopo churn; re-test when the collapse actually runs).

  **Proposed future lanes (not applied):** flip the 2D droplet default to delaunay+remap once long-run/refine-4/c_s=5 stability is proven (it already scores better and keeps retopology honest; would re-pin `baseline_oscillation.json`); fix the dam-break dead flag. Full design/evidence/dead-ends: [`docs_temp/debug_session/laneD-conservative-retopo-remap.md`](docs_temp/debug_session/laneD-conservative-retopo-remap.md).

  **Recommended next prompt** (updated 2026-07-29, supersedes the 2026-07-03 prompt below — all four of its next-lane options were executed this session):

  > The 2026-07-29 four-lane continuation (wf4) landed ALL FOUR options from the 2026-07-03 prompt: (A) 3D exact-dual-volume switch ON + canonical 3D qhull input order — 3D floor re-pinned DOWN to 7.274172e-05, settle step eliminated, last confirmed-high audit item (dual-volume-3d) closed; (B) 3D score harness built + FIRST 3D baseline — `retopo_policy_3d='dual_only'` new 3D runner default (summary 0.24811), boundary-saturation tripwire in place, 06 §1.6 closed; (C) exact two-fluid reference — the "single-fluid Lamb reference under-counts dissipation" hypothesis REFUTED, the ~5-10% 2D over-decay is solver-side; (D) opt-in `retopo_remap='conservative'` makes per-step FULL Delaunay reconnection thermodynamically neutral (KE_max 4.07e-2 → 8.3173e-7 J, l2 0.17479 — beats dual_only 0.17857) — the structural fix for large-deformation cases. Read `docs_temp/09_continuation_2026-07-03.md` first, then the four lane logs. Everything is STILL working-tree only in BOTH repos: **commit first** (per-lane changed-file lists in `docs_temp/debug_session/lane{A,B,C,D}-*.md`; verify after staging: fast suite 865 P / 0 F, floor battery 17 P, a5b regression 2 P, hyperct 336 P).
  >
  > **Next lane options (orthogonal, pick by user preference)**:
  >
  > 1. **Flip the 2D droplet default to delaunay + `retopo_remap='conservative'`** once long-run stability is proven (multi-t_end horizons, refine 4/4, c_s=5): it already scores better AND keeps retopology honestly active; would re-pin `baseline_oscillation.json` (laneD §2.6.1).
  > 2. **Fix the dam-break dead flag and unstick the collapse** — `dam_break_2D.py`'s `skip_triangulation=True` is silently ignored (`_do_retopologize` doesn't forward it to callable retopo_fns; the case actually runs per-step Delaunay). Then re-run at horizons where reconnection actually fires (larger t_end / weaker alpha_art / finer mesh) and A/B the remap there, where it should be load-bearing (laneD §2.3).
  > 3. **3D physics gap behind summary 0.248** — a smooth ~0.8%-of-R0 droplet inflation instead of decay to R0, not noise (mass/interface/KE channels machine-clean). First suspects: 3D interface-curvature operator truncation (`hndA_i_interface` on the refine-2 ring) and YL mass-preload discretisation — same class as the 2D chord-vs-arc floor (laneB). Do NOT chase l2 → 0 against the envelope reference.
  > 4. **2D over-decay residual (l2 0.179, anti-convergent)** — the reference-error candidate is CLOSED by laneC; live suspects are (i) the per-step mass-redistribution projection and (ii) the O(h) 32-gon curvature bias.
  >
  > Do NOT: re-pin any baseline to the two-fluid numbers (single-fluid Lamb stays the regression gate, laneC); retry restore-only or incremental-level remap variants (both measured dead ends — bounce corr +0.9998, far-field pressure runaway, laneD §2.2); apply the remap in 3D without re-verifying boundary-volume bookkeeping (laneD §2.6.5); reintroduce mixed 3D dual-volume sources or pin against the intermediate switch-only value 7.616854e-05 (laneA); rely on `SETTLE_STEPS=2` reasoning — 3D static-cloud retopo is now idempotent (laneA). Open footgun (small): `assign_simplex_phases` auto-Delaunay-populates `HC._simplices` on structured-connectivity meshes (laneA §3).

- **2026-07-29 — Lane C shipped (validation lane, no solver changes): exact two-fluid reference built for the 2D droplet score; lane-5's "the single-fluid Lamb reference under-counts dissipation" hypothesis REFUTED — the corrected reference explains essentially NONE of l2 = 0.179 (l2 vs two-fluid = 0.1885, +5.6%); the ~5-10% over-decay is solver-side.**

  NEW in `src/_analytical.py` (pinned single-fluid functions untouched): full 2D two-fluid viscous normal-mode dispersion relation (4×4 streamfunction determinant, `two_fluid_dispersion_roots_2d` / `two_fluid_omega_beta_2d`), closed-form energy-method `lamb_damping_rate_two_fluid` (= 2l[(l−1)μ+(l+1)μ_o]/((ρ+ρ_o)R0²), recovers single-fluid Lamb exactly at μ_o=ρ_o=0), and the exact started-from-rest `mode_temporal_ivp` / `radius_perturbation_two_fluid`. `_metrics.py:add_two_fluid_reference` + the 2D runner now report BOTH scores side by side in `score.json` (pinned metric unchanged) and dump `results/diag_series.json` for rerun-free rescoring.

  | quantity | value |
  |---|---|
  | exact two-fluid least-damped mode (ρ 800/1000, μ 0.5/0.1) | s = **−6.8325510 ± 5.4707015i** (weakly oscillatory, NOT overdamped) |
  | l2 vs single-fluid Lamb (pinned) / vs two-fluid exact | **0.1785660454150319** (bit-identical) / **0.1885496661389148** |
  | measured KE tail decay vs 1f / 2f-exact / 2f-energy predictions | **6.53** (lane-5 fit 6.79) vs 7.183 / 13.665 / 11.111 1/s |
  | sim temporal factor at t_end vs any defensible reference | 0.390 vs 0.706–0.719 (sim under-shoots BOTH) |
  | no-slip wall probe (confined 6×6, r_w = 5 R0) | −6.8401+5.4631i — wall shifts β by ~0.1%, negligible |

  **Verification.** Dispersion relation validated in 5 limits before use (inviscid → ±iω with ρ+ρ_o inertia 0.3%; √μ weak-viscosity damping = Miller–Scriven interfacial-layer scaling; vanishing-outer → single-fluid Lamb 2.5%; continuous μ-ramp, no branch jump; Re(q)>0 both phases). Battery: floor **14 P** (new tests in a separate file by design); fast suite **862 P / 0 F** (+15 new `test_case_oscillating_droplet_two_fluid.py`, incl. μ_o→0/ρ_o→0 limit tests to 1e-14); equil **1.1847e-03**; oscillation l2/tail/linf/mass **bit-identical to 16 digits**. No solver, `_params`, `_setup`, baseline, or hyperct edits.

  **What it means.** The residual l2 = 0.179 and its ANTI-convergence cannot be blamed on the reference: the outer bath changes the mode *structure* (oscillatory, β 6.83) but its envelope tracks the single-fluid biexponential to ~2% at t_end, and the wall at 5R0 is negligible. The live suspects for the over-decay are lane-5's (i) per-step mass-redistribution projection and (ii) O(h) 32-gon curvature bias. Do NOT re-pin baselines to the two-fluid numbers. Full detail: [`docs_temp/debug_session/laneC-two-fluid-reference.md`](docs_temp/debug_session/laneC-two-fluid-reference.md).

- **2026-07-29 — Lane B shipped: 3D oscillating-droplet score harness built (Tier 3B, 06 §1.6); FIRST 3D baseline scored; first 3D retopo A/B → `retopo_policy_3d = 'dual_only'` is the new 3D runner default (l2 1.52446 → 0.24811, −83.7%); historical R_max-at-~5·R0 boundary-saturation artefact no longer reproduces (tripwire added); every 2D number bit-identical.**

  NEW `src/_metrics.py:oscillation_score_3d` scores R_max(t) against the Rayleigh–Lamb `max_radius_envelope` (Miller–Scriven omega 16.5145 with rho_o=1000; Lamb beta_3d 31.25 — overdamped) exactly like the 2D score, plus a secondary apex score (`compute_diagnostics(..., polar_axis='z')`, new kwarg, default 'x' bit-identical), a machine-checkable `boundary_saturation` flag (any frame with R_max ≥ 0.9·L_domain flags the documented mesh-boundary artefact instead of silently scoring garbage) and `dual_vol_*`/`n_interface_*` bookkeeping fields. Runner `oscillating_droplet_3D.py` now writes `results_3d/score.json` (+ `--retopo` A/B flag with suffixed artifacts); first checked-in baseline `baselines/baseline_oscillation_3d.json`. Lane-5's 3D skip-triangulation caveat was verified BEFORE trusting the A/B (step-granular probe: identical boundary count 96, identical |dV/V0| step-0 boundary-zeroing artefact 0.357050, exact simplex volumes engaged in both paths, mass ~2e-15).

  | metric (872 steps, refine 2/2) | per-step Delaunay | dual_only (new default) |
  |---|---|---|
  | l2 / linf (R_max vs envelope) | 1.5244561707801316 / 2.2655 | **0.24811340819647862 / 0.5993** |
  | tail_growth / mass_drift | 0.47067 / 3.87e-14 | **0.08410** / 1.91e-14 |
  | R_max_peak / boundary_saturation | 0.011397 / False | 0.010790 / False |
  | n_interface / dual_vol_drift_post | 98 constant / 5.08e-3 | 98 constant / **1.07e-4** |

  **Verification.** Fast suite **847 P / 0 F** (+13 new tests: `test_oscillation_score_3d.py` synthetic scoring-math guards + `TestDualOnlyRetopoPolicy3D` policy/bookkeeping net); floor battery **14 P** no re-pins (floor tests exercise setup's per-step-Delaunay retopo_fn, unaffected by the policy constant); equil 1.1847e-03 and 2D oscillation l2 0.1785660454150319 / tail 0.9992507831101141 / mass 2.676988154993443e-14 **bit-identical to 16 digits**; hyperct untouched.

  **What this means.** The last unmeasurable dynamic case is now scored and regression-locked: 3D validation can be asserted (and diffed via `diff_baselines`) for the first time. Both 3D policies conserve mass/interface to machine precision and neither saturates at the boundary any more — the remaining summary 0.248 is a smooth physics-side gap (slow droplet inflation ~0.8% of R0 instead of decay to R0; curvature-side suspects, same class as the 2D chord-vs-arc floor) — do NOT chase l2 → 0 against this reference before checking the 3D interface-curvature operator. Full detail: [`docs_temp/debug_session/laneB-3d-score-harness.md`](docs_temp/debug_session/laneB-3d-score-harness.md).

- **2026-07-29 — Lane A shipped: 3D exact-dual-volume switch ON + canonical 3D Delaunay input order; 3D retopology floor re-pinned DOWN 7.3768e-05 → 7.274172e-05 (−1.4%), settle step eliminated (plateau now from step 1); every 2D number bit-identical.**

  Flipped the three `NOTE(lane3-dual-volume)` switch points TOGETHER (`stress.py` `dual_volume` dim==3, `cache_dual_volumes` dim in (2,3), `_integrators_dynamic.py` step-5b batch_e_star preference — 3D-gated so 2D keeps its bit-identical baseline) and, per the optional probe, canonicalized the 3D qhull input order in hyperct `connect_and_cache_simplices` (lexicographic sort + index remap, `NOTE(laneA-canonical-order)`). The switch alone reproduced lane 3's measured plateau 7.616853911101026e-05 (+3.26%, the honest settle-step artifact); the canonical order makes static-cloud retopo idempotent, kills the step-1→2 settle entirely, and lands the plateau at **7.274172178727318e-05 — below the old fan floor**. Verified via `diagnose_a5_bisection.py --redistribute-mass --n-steps 100` before pinning (step0 6.0153201139652905e-05 bit-unchanged; F rel spread 6.05e-12 steps 1..100; volume spread 0.0). Strict xfail `test_partition_of_unity_3d_jittered_production` flipped to passing (production 3D `dual_volume` tiles to 1e-12). Companion fix: `_dual_split_2d.py::_dual_volume_3d` now reads the cached authoritative `v.dual_vol` (the switch had desynced it from the `assign_simplex_phases` auto-Delaunay cache on structured meshes — 3 `test_dual_split_2d.py` failures, now green; the auto-Delaunay-on-structured-mesh mixed state itself remains a documented open footgun).

  | metric | before | after |
  |--------|--------|-------|
  | 3D floor plateau | 7.3768e-05 (from step 2) | **7.274172e-05 (from step 1, SETTLE_STEPS 2→1)** |
  | 3D floor step0 | 6.0153e-05 | 6.0153e-05 (bit-unchanged) |
  | 2D floors / equil / osc | 2.3748568e-03 / 2.2716938e-03; 1.1847162859108737e-03; l2 0.1785660454150319, tail 0.9992507831101141 | **all bit-identical** |
  | fast suite | 833 P / 0 F / 3 xfail | **834 P / 0 F / 2 xfail** (+1 = flipped xfail) |
  | floor battery + a5b | 14 P + 2 P | 14 P + 2 P (re-pins: plateau, SETTLE_STEPS, `A5B_3D_PEAK`/`END` 7.3768e-05 → 7.274172e-05) |
  | hyperct | 334 P (+39 pre-existing benchmark errors) | **336 P** (+2 order-invariance tests, same 39 errors) |

  **What this means.** The last confirmed-high audit item (dual-volume-3d) is closed in production: 3D `rho = m/dual_vol` no longer reads the 1–4% interior / ~20% boundary fan undercount, 3D dual volumes tile domains to machine precision, and the order-dependent Delaunay tie-breaking artifact that both created the settle step and blocked lane 3's switch is gone at the source. Full detail + changed-file list: [`docs_temp/debug_session/laneA-3d-exact-dual-volume-switch.md`](docs_temp/debug_session/laneA-3d-exact-dual-volume-switch.md). Still open: 3D dynamic score harness (06 §1.6) — this lane validates the static floor only.

- **2026-07-02/03 — Six-lane physics-audit fix session shipped: 2D oscillating-droplet l2 2.95 → 0.179 (−94%), tail_growth 2.35 → 0.999 (−57.5%), both dynamic-validation targets (l2 < 0.2, tail < 1.0) met for the first time; fast suite fully green (833 passed, 0 failed). All edits working-tree only in BOTH repos (ddgclib + hyperct).**

  Executed the fix order from [`docs_temp/07_physics_audit.md`](docs_temp/07_physics_audit.md) §5 as six sequential lanes; full session report with per-lane file lists, probe evidence, and the fix→effect attribution chain at [`docs_temp/08_debug_session_2026-07-02.md`](docs_temp/08_debug_session_2026-07-02.md), individual lane logs in [`docs_temp/debug_session/`](docs_temp/debug_session/). Lanes: (1) zero-gauge-pressure sentinel + multiphase momentum antisymmetry (`multiphase_stress.py` geometric `dual_vol_phase[k] > 1e-30` presence test replaces the `== 0.0` misread; gauge invariance restored 1.17 N → 8.9e-15 N, |ΣF|/max|F| 0.70 → 1.5e-13); (2) EOS consistency (`rho_clip` coherent across pressure/density/sound_speed with warn-once + `clip_count` visibility, shared `interface_mean_pressure` convention, `MultiphaseEOS` phase≥0 guard — production metrics bit-identical, latent-trap closure); (3) exact simplex-container dual volumes (`hyperct.ddg.simplex_dual_volumes`, Vol_i = (1/(dim+1))·Σ|T∋i|, wired into the 2D production path; the 3D switch was built, measured to raise the pinned 3D floor 7.3768e-05 → 7.6169e-05 (+3.3%), and **backed out** per the re-pin-only-if-lower rule — mechanism fully diagnosed as order-dependent Delaunay tie-breaking at the settle steps, switch points marked `NOTE(lane3-dual-volume)`); (4) upstream hyperct remesh conservation (mass/momentum-conserving `edge_split_2d`/`edge_collapse_2d`, per-edge local length scale, `rebuild_simplex_cache_2d` in the adaptive branch — A.4's documented 9.7 → 187 mass blow-up now machine-precision flat, KE explosion fixed; delaunay production path bit-identical); (5) dynamic-config sweep → new 2D default `retopo_policy_2d='dual_only'` (freeze builder connectivity, refresh duals/splits/redistribution/EOS every step); (6=lane 7) cleanup: stale `test_raises_unsupported_dim` fixed to the new hyperct contract, the factor-2 bug in `analytical/_integrated_comparison.py` fixed (pre-fix `volume_averaged_scalar(1) == 2.0`, `HydrostaticPressure` assigned 2× pressure after duals, `integrated_pressure_error` inverted good/bad fields), fast oscillation-envelope regression test pinned.

  | metric | baseline (session start) | final (post-lane-7) | Δ |
  |--------|--------------------------|---------------------|---|
  | osc l2_error_normalized | 2.9519 | **0.1785660454150319** | −94.0% |
  | osc tail_growth | 2.3530 | **0.9992507831101141** | −57.5% |
  | osc linf_error_normalized | 5.9145 | **0.32842491275938274** | −94.4% |
  | osc KE_max | ~4.1e-02 J, still growing at t_end | **8.317e-07 J** (physical overdamped decay; rate 6.79 vs analytical 7.18 1/s) | ~5e4× lower |
  | osc mass_drift | — | 2.676988154993443e-14 | machine precision |
  | equil summary (static_droplet_2D) | 1.1636e-03 | 1.1847162859108737e-03 | +1.81% (inside 5% band; lane-3 drift-metric sensitivity) |
  | 2D floor (frozen / post-retopo) | 2.3748568e-03 / 2.2716938e-03 | unchanged (bit-compatible) | no re-pin |
  | 3D floor (step0 / plateau) | 6.0153e-05 / 7.3768e-05 | unchanged (3D exact-volume switch backed out) | no re-pin |

  Per-lane trail (l2 / tail): baseline 2.9519/2.3530 → L1 0.70907/2.20527 → L2 bit-identical → L3 0.48992/1.72505 → L4 bit-identical → L5 0.17857/0.99925 → L7 bit-identical. The l2 decomposed into three separable mechanisms: transient momentum injection (L1, −76% l2), wall/corner dual-volume misestimation refreshed each retopo (L3, −31% l2 / −22% tail), and per-step global Delaunay reconnection jolts (L5 — under the old default, KE was ~5·10⁴× the physical level and still growing; with `dual_only` the KE(t) quantitatively reproduces the analytical overdamped Rayleigh–Lamb envelope, i.e. essentially 100% of the old KE was retopology-injected noise). The lane-5 sweep also *closed* the displacement-gate and hybrid retopo ideas for this case (no good eps exists; hybrids strictly worse) and re-answered the c_s question: keep c_s = 1 m/s (dual_only is c_s-insensitive; per-step Delaunay at c_s=5 destroys the interface ring).

  **Verification.**
  - `pytest ddgclib/tests/ -m "not slow" -q` — **833 passed, 12 skipped, 17 deselected, 3 xfailed, 5 warnings in 56.60s — ZERO failures** (baseline: 796 passed + 1 pre-existing `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`, fixed in lane 7; +37 new regression tests across the lanes; the +1 xfail vs baseline is the intentional strict xfail documenting the un-switched 3D dual-volume path).
  - `pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""` — **14 passed in 18.92s** (12 pinned floor tests + new `TestDualOnlyRetopoPolicy2D` + new `TestOscillationEnvelopeRegression2D`; all four floor constants untouched). The envelope test pins l2 0.050006441230559064 / tail 0.9130549583972877 on a fast refine-2/2 fixture with ~10% headroom; reverting the lane-5 policy scores l2 1.384569 there, so the pin catches that regression class by >25×.
  - hyperct suite (post-lane-4, last hyperct-touching lane): **334 passed, 40 skipped, 6 xfailed, 39 errors** — the 39 errors are the pre-existing pytest-benchmark fixture errors, bit-identical to the pre-session baseline run (301 passed); +33 new upstream tests (`test_dual_volume.py` 11, `test_remesh_conservation.py` 22).
  - Case baseline re-pinned: `cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json` (old pre-lane-1-era values documented in the lane-5 log). `baseline_equilibrium.json` untouched.
  - **All edits are working-tree only — no git operations were performed, in either ddgclib or the hyperct sibling checkout** (`/home/endres/projects/hyperct/hyperct`, edited via the `./hyperct` symlink). User stages/commits as preferred; each lane log carries its complete changed-file list.

  **What this means for the attack plan.** The M4 dynamic-validation milestone (2D oscillating droplet vs Rayleigh–Lamb) is met on the shipped default configuration, and the win is regression-locked at three levels (case baseline JSON, fast envelope pytest, pinned floors). The retopology-noise mechanism suspected since 2026-04-29 is now measured and quantified: per-step global Delaunay reconnection was the dominant l2/tail driver, and for the small-strain droplet the cure is freezing connectivity (`dual_only`), not gating it (the displacement-gate lane is closed — no eps works). What this does NOT fix: large-deformation cases (dam break, detaching bubbles) cannot freeze connectivity, so the per-rewire jolt mechanism remains open there — candidate real fixes are a conservative old-dual→new-dual remap or ALE-style mass/momentum-remapped smoothing (lane-4/5 logs). The residual l2 = 0.179 is a smooth ~5–10% over-decay that is anti-convergent under refinement and plausibly partly the *reference's* error (single-fluid Lamb β ignores the ρ_o=1000 outer bath) — a two-fluid Prosperetti-type reference is the right next validation step, not further solver chasing. The exact 3D dual-volume switch is built, tested upstream, and one gated conditional away (re-pin plateau to 7.616854e-05 when switching all three points together). Open: 3D dynamic score harness (06 §1.6), tag-consistency flag 19, `dual_only_noredist`'s cleanest KE decay as a possible more-physical benchmark once the wave-launch transient is addressed.

  **Recommended next prompt** (updated 2026-07-03):

  > The 2026-07-02/03 six-lane session met both 2D dynamic-validation targets (l2 0.179 < 0.2, tail 0.999 < 1.0; fast suite 833 passed / 0 failed) — read `docs_temp/08_debug_session_2026-07-02.md` first. Everything is working-tree only in BOTH repos: **commit first** (per-lane file lists are in `docs_temp/debug_session/lane*.md`; verify 833 P / 0 F fast, 14 P floor battery, hyperct 334 P after staging).
  >
  > **Next lane options (orthogonal, pick by user preference)**:
  >
  > 1. **3D exact-dual-volume switch (~2–4 h)** — flip the three `NOTE(lane3-dual-volume)` switch points TOGETHER (`stress.py` dim==3 `dual_volume`, `cache_dual_volumes` dim in (2,3), `_integrators_dynamic.py` batch_e_star preference), re-pin `TestStaticDroplet3DRetopologyFloor.EXPECTED_PLATEAU_MAXF` 7.3768e-05 → 7.616854e-05 (measured bit-stable), flip the strict xfail `test_partition_of_unity_3d_jittered_production`. Optionally canonicalize Delaunay input order in `connect_and_cache_simplices` to kill the settle-step artifact (re-pins every 3D dynamic number).
  > 2. **3D oscillating-droplet score harness (~half day)** — the 3D case still emits no `oscillation_score` (06 §1.6, DEVELOPMENT.md backlog); without it neither the 3D fan-volume churn nor a 3D `dual_only` policy sweep is measurable. Do NOT blind-apply `dual_only` to 3D (different boundary-volume bookkeeping).
  > 3. **Two-fluid analytical reference (validation lane)** — the residual 2D l2 0.179 is a smooth over-decay, anti-convergent under refinement; replace/augment the single-fluid Lamb β=25.0 with a Prosperetti-type two-fluid rate (or viscous-corrected β) before spending any solver effort on it.
  > 4. **Conservative retopo remap for large deformation** — dam break/detaching bubbles can't freeze connectivity; prototype the old-dual→new-dual conservative remap (p_ref benchmark pattern) or ALE-remapped smoothing (lane-4 §6.1).
  >
  > Do NOT: re-open the displacement-gate/hybrid retopo for the 2D droplet (lane-5 sweep closed it), raise c_s (catastrophic under Delaunay, useless under dual_only), widen the EOS clip band (4.4× worse l2, lane-2 A/B), silence the TaitMurnaghan clip RuntimeWarning (read `eos.clip_count` instead), or drop `HC._simplices` without `rebuild_simplex_cache_2d` (silently reverts the lane-3 geometry). Old logged numbers from 2D cases using `volume_averaged_scalar`-based ICs or `integrated_pressure_error` predate the lane-7 factor-2 fix — re-baseline before comparing.

- **2026-06-02 — Probe 6 shipped: A.5.b 3D long-run bit-stable over 2000 steps, regression test pinned symmetrically with the 2D floor. Both static-residual floors now regression-locked.**

  Picked Probe 6 (the cheap symmetric closure) from the 2026-05-28 options list. Ran [`diagnose_a5_bisection.py`](cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py) `--n-steps 2000 --redistribute-mass --results-suffix 3d_longrun` against the production-default code path (`redistribute_mass=True`, `split_method='neighbour_count'`, `curvature_path='integrated'`), 3D refine 2/2 (472-vertex, 98-interface mesh). Wall time 686.9s for 2000 steps. Raw JSON preserved at `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection_3d_longrun.json`.

  | metric | value |
  |--------|-------|
  | A.5.a (frozen, pre-retopo) max\|F\| | **6.0153e-05** (×0.708 of full-dynamic baseline 8.5e-5) |
  | A.5.b step 0 max\|F\| | 6.0153e-05 (bit-identical to A.5.a) |
  | A.5.b step 1 max\|F\| | 7.0325e-05 (one-step retopo transient) |
  | A.5.b step 2 max\|F\| | **7.3768e-05** (×0.868 of baseline; post-retopo settle) |
  | A.5.b step 2..2000 unique values | **1** (bit-identical across 1999 retopo calls) |
  | A.5.b step 2..2000 rel spread | **0.0** (exact) |
  | KE plateau | 2.371e-10 (impulse residual, non-growing) |
  | \|dM/M0\| | **8.78e-15** (machine-precision conservation) |
  | \|dV/V0\| step 0→1 | 3.055e-01 (one-shot boundary-shell zeroing; **not** a drift) |
  | volume rel spread step 2..2000 | **0.0** (bit-stable post-settle) |
  | vertex count drift | 472 → 472 (unchanged) |
  | interface count drift | 98 → 98 (unchanged) |

  **Interpretation.** The post-Phase-2c production code path is retopology-neutral in 3D to machine precision under the static-droplet harness, confirming the 2026-05-27 Probe 1 100-step trace survives at 20× the horizon. Two 3D-specific differences from the 2D floor: (1) the plateau is reached at step 2 rather than step 1 — one extra retopo settle step on the near-cospherical interface cloud (step 1 = 7.0325e-05, step 2+ = 7.3768e-05, then literally bit-identical for 1999 calls); (2) the one-shot `|dV/V0| = 0.305` at step 0→1 is the documented boundary dual-cell zeroing artefact (outer-box boundary vertices lose their "shell" dual volume on the first retopo), **not** a drift — total volume is bit-stable from step 2 onward (spread 0.0). The ×1.23 gap between A.5.a (6.02e-05) and the A.5.b plateau (7.38e-05) is the residual 3D Delaunay non-uniqueness churn already characterised in the 2026-04-29 Phase 2 entry, and is mathematically capped by Probe 2's no-op verdict on the integrated curvature stencil.

  **Regression guard shipped.** New `@pytest.mark.slow` class [`TestStaticDroplet3DRetopologyFloor`](ddgclib/tests/test_case_oscillating_droplet.py), mirroring `TestStaticDroplet2DRetopologyFloor`. Runs 10 A.5.b steps (~6s wall; the plateau is established by step 2 so 10 is sufficient to detect any future regression) and asserts:
  - Step-0 max\|F\| within 1% of 6.0153e-05 (A.5.a frozen floor)
  - Plateau (step ≥ `SETTLE_STEPS`=2) max\|F\| within 1% of 7.3768e-05 (A.5.b post-retopo steady state)
  - Plateau max\|F\| rel spread < 1e-10 (bit-stability proxy)
  - Mass drift step0→final < 1e-10
  - Volume rel spread across the plateau window < 1e-10 (checked post-settle, NOT from step 0, because of the boundary-shell artefact)
  - 472 → 472 vertices, 98 → 98 interface (no topology flips)

  **Verification.**
  - `pytest ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet3DRetopologyFloor -v -m slow` — **PASSED** in 6.11s.
  - `pytest ddgclib/tests/ -m "not slow"` — **794 passed, 15 skipped, 17 deselected, 2 xfailed, 0 failures** (+1 deselected = the new slow 3D test correctly marked and deselected here).
  - Working-tree only; user stages/commits as preferred.

  **What this means for the attack plan.** Both static-residual floors (2D 2.27e-3, 3D 7.38e-05) are now regression-locked at machine-precision conservation + 1% steady-state floor tolerance. The static-residual lane is fully closed pending higher-order curvature work. The remaining open levers are exactly the three orthogonal probes from the 2026-05-28 list: (2) higher-order 2D curvature stencil (research push, the only lever on the 2D O(h) floor), (5) the max-vertex-displacement skip-retopo gate (defensive cleanup, now safe with both floors locked), and the pivot to dynamic validation (M4 / 2D oscillating-droplet Rayleigh-Lamb), which is the ~37% of 2D baseline that A.5 does not capture and is gated on Probe 5 and/or adaptive remesh.

- **2026-05-28 — Tier 2B 2D long-run shipped: A.5.b 2D bit-stable over 2000 steps, regression test pinned at the post-retopo floor.**

  Picked the "Tier 2B 2D long-run" branch from the 2026-05-27 (latest) recommended-next-prompt options. Ran [`diagnose_a5_bisection.py`](cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py) `--skip-3d --n-steps 2000 --redistribute-mass --results-suffix 2d_longrun` against the production-default code path (`redistribute_mass=True`, `split_method='neighbour_count'`, `curvature_path='integrated'`). Wall time 125.7s for 2000 steps over a 311-vertex, 32-interface mesh (`refinement_outer=3, refinement_droplet=3`). Raw JSON preserved at `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection_2d_longrun.json`.

  | metric | value |
  |--------|-------|
  | A.5.a (frozen, pre-retopo) max\|F\| | **2.3748568012e-03** (×0.625 of full-dynamic baseline) |
  | A.5.b step 0 max\|F\| | 2.3748568012e-03 (bit-identical to A.5.a) |
  | A.5.b step 1 max\|F\| | **2.2716937802e-03** (×0.598 of baseline; post-retopo settle) |
  | A.5.b step 1..2000 unique values | **1** (literal bit-identical across 1999 retopo calls) |
  | A.5.b step 1..2000 std | **4.34e-19** (float roundoff floor) |
  | A.5.b step 1..2000 ptp | 0.0 |
  | linear fit slope per step | 5.93e-19 (zero to roundoff) |
  | KE plateau | 3.688e-11 (impulse residual, non-growing) |
  | \|dM/M0\| | **1.643e-15** (machine-precision conservation) |
  | \|dV/V0\| | **3.542e-16** (machine-precision conservation) |
  | vertex count drift | 311 → 311 (unchanged) |
  | interface count drift | 32 → 32 (unchanged) |

  **Interpretation.** The post-Phase-2c production code path is *exactly* retopology-neutral in 2D under the static-droplet harness: every retopo call from step 1 onwards produces the same `max|F|` to within floating-point roundoff (single unique value across 1999 calls), and the mass/volume invariants stay at machine precision throughout. The settle from step-0 (2.3749e-3) to step-1 (2.2717e-3) is a one-shot 4.4% adjustment as the first retopo re-evaluates the interface from the symmetric initial cloud onto its Delaunay-of-static-cloud equivalent; that delta is not a drift, just the difference between A.5.a's frozen evaluation and the A.5.b post-retopo equilibrium. Confirms the 2026-04-29 Phase-2c claim — "2D A.5.b matches A.5.a exactly" — survives at 20× the original 100-step horizon, and the 2D residual is **provably 100% curvature stencil** (Tier 2B step 1 target), 0% retopology-induced drift.

  **Regression guard shipped.** New `@pytest.mark.slow` class [`TestStaticDroplet2DRetopologyFloor`](ddgclib/tests/test_case_oscillating_droplet.py) (~125 lines). Runs 30 A.5.b steps (~2.6s wall; bit-stability is established at step 1 so 30 is sufficient to detect any future regression that breaks step-1 stability or introduces drift) and asserts:
  - Step-0 max\|F\| within 1% of 2.3748568e-03 (A.5.a frozen floor)
  - Step-1 max\|F\| within 1% of 2.2716938e-03 (A.5.b post-retopo steady state)
  - `(max - min) / step1 < 1e-10` across the 30-step window (bit-stability proxy)
  - Mass drift < 1e-10, volume drift < 1e-10
  - 311 → 311 vertices, 32 → 32 interface (no topology flips)

  Tolerance picked per the 2026-05-27 Probe 6 spec ("within ~1% of the measured value"). The 1e-10 spread tolerance is generous — the actual 2000-step floor is bit-identical (std=4.3e-19) — but leaves room for double-vs-single-precision changes in downstream code without false failures.

  **Verification.**
  - `pytest ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet2DRetopologyFloor -v` — **PASSED** in 2.59s.
  - `pytest ddgclib/tests/ -m "not slow"` — **781 passed, 15 skipped, 16 deselected, 2 xfailed, 0 failures** (up from 780 post-Probe-3, +1 = the new test correctly marked slow and deselected here).
  - Working-tree only; user stages/commits as preferred.

  **What this means for the Tier 2B attack plan.** Both Probes 5 and 6 (3D) and the 2D long-run are now closed. The 2D static-droplet residual is locked in as a regression baseline; the 1.23× 3D gap is the only remaining static-residual lever and is mathematically capped by Probe 2's no-op verdict on the integrated curvature stencil. The active frontier is now genuinely Tier 2B step 1 (the *2D-specific* integrated γ-flux curvature rewrite) — which, per the 2026-05-06 audit note in `diagnose_a5_bisection.py:402-409`, is what would drive the 2.37e-3 / 2.27e-3 floors toward machine precision under O(h) → O(h²) → exact convergence on the polygon approximation of the circle.

  **Recommended next prompt** (updated 2026-05-28):

  > The 2D A.5.b floor is now a pinned regression baseline (2.2717e-3 post-retopo, bit-stable over 2000 steps, locked in `TestStaticDroplet2DRetopologyFloor`). Both 3D (Probe 6 — implicitly already verified by the bit-identical 100-step trace) and 2D long-runs confirm the post-Phase-2c code path is retopology-neutral at machine precision. No drift lurking.
  >
  > **Next probe options (orthogonal, pick by user preference)**:
  >
  > 1. **Tier 2B step 1 for 2D — integrated γ-flux curvature rewrite (~4–8 hours)**. The 2D residual is now provably 100% pointwise-curvature truncation: `hndA_i_interface` on a polygon approximation of the circle has O(h) error. Replace it with the existing `surface_tension_force_2d` ([`curvature_2d.py:91-181`](ddgclib/operators/curvature_2d.py#L91-L181)) — already exact for piecewise-linear interfaces — verified called from `_interface_surface_tension` for the 2D dim path, then re-run the diagnose harness. Target: drop 2.37e-3 / 2.27e-3 toward machine precision in 2D. After this lands, re-pin the regression test at the new floor.
  >
  > 2. **Probe 5 — `max-vertex-displacement < eps` skip-retopology gate (~1–2 hours)**. Still pending from the 2026-05-27 (latest) options list. Now that both 2D and 3D static floors are regression-locked, this gate becomes safer to add — any breakage shows up immediately in `TestStaticDroplet2DRetopologyFloor`. On the A.5.b harness with `u=0`, would collapse A.5.b to A.5.a (6.02e-05 in 3D) by skipping every retopo call; on dynamic runs needs eps proportional to `h_local`.
  >
  > 3. **Probe 6 — explicit 3D long-run regression (~30 min)**. Not strictly required (3D 100-step trace was already bit-flat per the 2026-05-27 Probe 1 entry), but encoding it as a `@pytest.mark.slow` test matching `TestStaticDroplet2DRetopologyFloor` would lock in the 7.38e-05 3D floor symmetrically with the 2D one. Cheap and symmetric.
  >
  > Do NOT touch BC isolation (Tier 1E / `OutletDeleteBC`) yet — still gated behind A.3 single-phase probes. Adaptive remesh (`remesh_mode='adaptive'`) still depends on the upstream hyperct mass-averaging + h_local fixes (A.4 partial).

- **2026-05-27 (latest) — A/B confirmation that Probe 4 was already shipped on 2026-04-29, and Probe 3 (`_compute_vd_3d` boundary NaN) fixed. 3D conservation diagnostics now functional; A.5.b 3D max|F| unchanged at 7.3768e-05.**

  **A/B confirmation that Probe 4 is live (not pending).** The 2026-05-27 (later) entry's "next probe options" sub-list framed Probe 4 (redistribute_mass guard fix) as still pending — that framing contradicts the plan's own Phase-2c status row (line 359), the `feedback_multiphase_mass_redistribution.md` memory, and the live code at [`mass_redistribution.py:54-74`](ddgclib/operators/mass_redistribution.py#L54-L74) / [`:308-310`](ddgclib/operators/mass_redistribution.py#L308-L310) / [`_integrators_dynamic.py:374-379`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L374-L379) / [`oscillating_droplet/src/_setup.py:42`](cases_dynamic/oscillating_droplet/src/_setup.py#L42). A two-run A/B verified the wiring end-to-end:

  | run | 3D A.5.b max\|F\| | path exercised |
  |-----|-------------------|----------------|
  | `diagnose_a5_bisection.py --n-steps 100` (no `--redistribute-mass`) | **1.4415e-3** | legacy guard (`p_phase < 1e-30`) — outer phase at P0=0 unconditionally skipped |
  | `diagnose_a5_bisection.py --redistribute-mass --n-steps 100` | **7.3768e-05** | post-fix guard (`dual_vol_phase[k] < 1e-30`) — outer phase redistributed |

  The pre-fix value (1.44e-3) only surfaces when `setup_oscillating_droplet`'s production default `redistribute_mass=True` is explicitly overridden by the harness's `--redistribute-mass=False` (the flag's `action='store_true'` makes `False` the harness default; without the CLI flag, the harness forwards `False` and overrides the setup's `True` default). Probe 1's 7.3768e-05 from earlier today **is** the post-fix steady value — there is no residual Probe-4 work to do. JSONs preserved at `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection_{with,without}_redistmass.json`.

  **Probe 3 — `volume_of_geometric_object` NaN fix (~30 min, shipped).** Localised the `hyperct/ddg/_geometry.py:91 RuntimeWarning: invalid value encountered in scalar divide` via traceback at setup time: `cache_dual_volumes` → `dual_volume(v)` → `_v_star(v, v_j, HC, dim=3)` → `volume_of_geometric_object(verts, v_i.x_a)`. Instrumented count: 746 of 30272 calls (2.5%) hit a literally degenerate base triangle where `points[0] == points[2]` (duplicate dual vertex from the boundary fan walk at outer-box corners); cross product is exactly zero → division returns NaN → 95 of 472 vertices end up with `dual_vol = NaN` (matches `project_3d_boundary_dual_bug.md`).

  The "pyramid" over a zero-area base has zero volume regardless of apex, so the fix is a one-line short-circuit before the division:

  ```python
  if norm_sq == 0.0:
      return 0.0
  ```

  Applied in two places (both copies of the function): [`hyperct/ddg/_geometry.py:79-105`](../hyperct/hyperct/ddg/_geometry.py#L79-L105) (canonical) and [`hyperct/ddg/barycentric/_duals.py:328-365`](../hyperct/hyperct/ddg/barycentric/_duals.py#L328-L365) (legacy duplicate kept for the deprecated `ddgclib.barycentric` shim). Regression coverage: new [`TestDegenerateGeometry`](../hyperct/hyperct/tests/test_ddg.py) class with 3 tests — collinear base, duplicated-base-point (the exact failure mode from the diagnostic), and a non-degenerate sanity check (unit corner tet → 1/6).

  **Post-fix verification.** Re-ran `diagnose_a5_bisection.py --redistribute-mass --n-steps 100`:

  | metric | pre-Probe-3 | post-Probe-3 |
  |--------|-------------|--------------|
  | 3D A.5.a `mass_total` | NaN | **0.91035** |
  | 3D A.5.a `volume_total` | NaN | **9.11e-04** |
  | 3D A.5.b max\|F\| (peak) | 7.3768e-05 | **7.3768e-05** (bit-identical) |
  | 3D A.5.b `|dM/M0|` | NaN | **8.66e-15** (machine-precision conservation) |
  | 3D A.5.b `|dV/V0|` | NaN | 3.05e-01 (one-shot at step 0→1, then stable to 1e-3) |
  | NaN dual_vol vertex count | 95 / 472 | **0 / 472** |
  | RuntimeWarning at setup | fires every run | **gone** |

  Max\|F\| being bit-identical confirms the fix is non-perturbative on the force calculation (boundary NaN was masked from |F| because retopology overwrites boundary `dual_vol` with 0 anyway).

  **New finding surfaced by the fix.** The `|dV/V0| = 3.05e-01` step 0→1 drop is **not** a regression — it's a previously-hidden artefact of boundary semantics: at setup, the 95 outer-box boundary vertices used to carry NaN `dual_vol` (silently dropped from sums); now they carry real geometric "outer shell" volume. The integrator's `_retopologize` boundary-handling at [`_integrators_dynamic.py:211`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L211) then zeros boundary `dual_vol` on the first retopo call (correct semantics — wall vertices should not carry interior fluid volume). After step 1, volume is stable to 1e-3 across the remaining 99 steps. No fix needed here; the previous NaN-masking just made the discontinuity invisible. Worth a separate audit if the setup-time volume reporting feeds into any IC mass-to-volume ratio (currently it does not — `setup_oscillating_droplet` only touches inner-phase mass for `v.dual_vol_phase[1] > 1e-30`, so the inner phase IC is unaffected).

  **Verification.**
  - `pytest hyperct/tests/test_ddg.py` — **27 passed, 9 pre-existing benchmark-fixture errors** (no `pytest-benchmark` installed, documented in the 2026-04-26 entry).
  - `pytest ddgclib/tests/ -m "not slow"` — **780 passed, 15 skipped, 14 deselected, 2 xfailed, 0 failures, 0 regressions** (matches the post-Probe-2 baseline).
  - All edits working-tree only; user stages/commits as preferred.

  **What this means for the Tier 2B attack plan.** The two probes orthogonal to the curvature-stencil lane are now closed:
  - Probe 3 ✅ — 3D conservation diagnostics finally functional in `compute_conservation` for the static-droplet (and every other 3D case). Downstream regressions: 3D conservation metrics (mass drift, volume drift, KE evolution) can now be encoded as pytest tolerances without the NaN-skip pattern that's currently shimmed in some case studies.
  - Probe 4 ✅ — was already done 28 days ago; just needed the misreading flagged and verified end-to-end.

  The 1.23× residual gap above the A.5.a floor (7.38e-05 / 6.02e-05) remains, and per the 2026-05-27 (later) Probe 2 analysis, is **mathematically irreducible** by any further integrated curvature stencil variant (the cotangent form, the Stokes-barycentric form, and the CSF-dual form are all the same vertex-supported integral on a piecewise-linear surface). The next-most-leveraged remaining attack is the 3D Delaunay non-uniqueness mitigation from the 2026-04-29 "Parallel orthogonal investigation" entry: **skip retopology when `max-vertex-displacement < eps`**. On the A.5.b harness with `u=0`, that would make the retopology a no-op (max-displacement = 0), collapsing A.5.b 3D to A.5.a 3D = 6.02e-05. On real dynamic runs the threshold has to balance staleness against churn.

  **Recommended next prompt** (updated 2026-05-27 latest):

  > Probes 3 and 4 are now both closed (Probe 4 was already live, A/B-verified; Probe 3 NaN fix shipped with 3 regression tests). 3D conservation diagnostics work, 780 fast tests pass, 0 regressions.
  >
  > **Next probe options (orthogonal, pick by user preference)**:
  >
  > 1. **Probe 5 — `max-vertex-displacement < eps` skip-retopology gate (~1-2 hours)**. Per the 2026-04-29 "Parallel orthogonal investigation" entry: the deeper driver of the 7.4e-05 / 6.02e-05 (×1.23) static-droplet gap is 3D Delaunay non-uniqueness churning ~48 cross-phase edges per retopo step on a near-cospherical interface cloud. On A.5.b 3D with `u=0`, gating retopology on `max ||x_i - x_i_prev|| > eps` would skip every retopo call and collapse A.5.b to A.5.a (6.02e-05). On real dynamic runs the gate needs an eps proportional to `h_local`; cheapest first cut is `eps = 1e-4 * h_min`. Implementation site: [`_integrators_dynamic.py:_retopologize`](ddgclib/dynamic_integrators/_integrators_dynamic.py).
  >
  > 2. **Probe 6 — A.5.b 3D long-run regression (~30 min)**. Re-run with `--n-steps 2000` to confirm the 7.3768e-05 floor has no slow drift; encode as a pytest in `ddgclib/tests/` pinning the steady-state max|F| within ~1% of the measured value. Locks in the current state as a regression guard. Now safe to add because conservation diagnostics are functional.
  >
  > 3. **Tier 2B 2D long-run** — same as Probe 6 but for 2D (2.37e-3 floor) — confirms it's truly stable before any 2D-specific 2B step-1 work.
  >
  > Do NOT chase the 1.23× gap with further curvature-stencil variants (Probe 2 closed that lane mathematically), and do NOT add further redistribute_mass guard changes (Probe 4 is fully wired and the A/B confirms it).

- **2026-05-27 (later) — Probes 1 & 2 against the simplex-aware migration: apex enumeration is a non-driver; integrated-Stokes curvature variant is mathematically identical to the existing cotangent form. The 1.23x A.5.b residual gap is the irreducible O(h^2) polygon-vs-smooth-sphere truncation, not a stencil-choice artefact.**

  **Probe 1 — measure migration delta.** Re-ran `diagnose_a5_bisection.py --redistribute-mass --n-steps 100` against the post-migration code. Numbers (interface max |F|, raw at `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection.json`):

  | dim | A.5.a (frozen) | A.5.b (retopo + u=0, peak) | 2026-04-29 A.5.b |
  |-----|----------------|----------------------------|-------------------|
  | 2D | 2.3749e-03 | 2.3749e-03 | 2.3749e-03 (unchanged) |
  | 3D | 6.0153e-05 | **7.3768e-05** | 7.38e-05 (bit-identical) |

  Both dimensions bit-identical to the 2026-04-29 baseline. Outcome per the decision tree in the prior status entry's "Recommended next prompt": the "stays at ~7.38e-05" branch — apex enumeration was a non-driver on this fixture (and the wiring is exercised end-to-end; `HC._interface_edge_to_apex` populates to 1152 edges with 2 apex each after the first stress call, verified independently). The apex migration closed one confounder without moving the headline number — good defensive hygiene, no residual leverage.

  **Probe 2 — integrated 3D `integrated_hndA_i_interface` via Stokes' theorem on the barycentric dual cell.** Implemented as recommended:
  - New function [`integrated_hndA_i_interface(v, interface_set, HC, gamma)`](ddgclib/_curvatures_heron.py) next to `hndA_i_interface`. Algorithm: for each interface triangle `(v_i, v_j, v_k)` containing v_i, integrate the in-surface conormal `nu` (oriented outward from v_i) over the two boundary segments `midpoint(v_i,v_j) -> centroid -> midpoint(v_i,v_k)`. Returns `gamma * boundary-integral nu dl`, the exact Stokes image of `gamma * 2H N dA` on a piecewise-linear interface.
  - Wired into [`_interface_surface_tension`](ddgclib/operators/multiphase_stress.py) as a new `curvature_path='stokes'` value (3D only; 2D delegates to the existing FTC form). Default unchanged. CLI flag added to `diagnose_a5_bisection.py`.
  - 5 new regression tests in [`test_simplex_aware_curvature.py`](ddgclib/tests/test_simplex_aware_curvature.py): (a) flat z=0 Kuhn-cube interface gives `max|F_st| = 7.36e-18` (machine precision, *not* via gamma=0 early return — verified via direct cancellation of opposite conormals); (b) on a closed spherical interface every vertex's F_st points inward AND has the same sign as the cotangent form (26/26); (c) sum of F_st over the closed sphere balances to 2.7e-19 (Newton's third law symmetry).

  **Result on A.5.b 3D: bit-identical to the 'integrated' (cotangent) path.** max|F| = 7.3768e-05, exactly the same as Probe 1, to within 1e-19 absolute on every interface vertex (relative ~1e-15, pure float roundoff). A direct diff check via `_interface_surface_tension(v, curvature_path='stokes')` vs `'integrated'` on every interface vertex of the oscillating droplet gave `max|diff| = 1.14e-19` over 98 vertices. To rule out mesh-symmetry coincidence, also ran on a deliberately irregular fixture (3 non-coplanar triangles around a vertex with deliberately unequal edge lengths): again bit-identical, `max|diff| = 2.62e-16`.

  **Mathematical interpretation.** The cotangent formula and the barycentric Stokes formula both compute the same integrated mean curvature normal `integral_{Gamma_i} 2H N dA` at vertex i. On a piecewise-linear triangulated surface, this integral is determined entirely by the surface geometry, not by the dual partition: `Delta_S X` is a vertex-supported distribution (Dirac mass at each vertex plus dihedral-angle line distributions on edges), and integrating it over *any* dual cell that contains v_i in its interior picks up exactly the same vertex contribution. The dual cell choice (Voronoi vs barycentric vs anything else) only matters for the *averaged* curvature `kappa = (integrated H N) / A_dual` — the *integrated* quantity used in F_st is dual-cell-independent.

  **Implication for the 1.23x A.5.b residual gap.** No integrated curvature stencil variant can reduce it — the gap is the irreducible O(h^2) polygon-vs-smooth-sphere truncation. On a perfect smooth sphere, F_st_analytical = -(2 gamma / R) A N exactly; on a triangulated polyhedron approximating that sphere, every per-vertex F_st has an O(h^2) error from the polygon-vs-sphere mismatch, and that residual is what A.5.a measures (6.02e-05). The further 1.23x bump to A.5.b is the retopology-on-static-cloud component (Delaunay non-uniqueness on the near-cospherical interface vertex cloud — already characterised in the 2026-04-29 Phase 2 entry, not a stencil concern).

  **Verification** — 780 passed (`pytest ddgclib/tests/ -m "not slow"`), 0 regressions, including 5 new `TestIntegratedHndAIInterface` tests. Working-tree only; not committed.

  **What this means for the Tier 2B attack plan.** Steps 1–3 of the 2026-04-29 attack order (redistribute_mass guard fix, 3D Delaunay non-uniqueness mitigation, `_compute_vd_3d` NaN) are now the only remaining levers on the 3D static-droplet residual. The "step 1 — replace pointwise curvature with integrated form" line of attack is *closed* by this probe: there is no pointwise/integrated distinction in 3D when both forms collapse to the same vertex-supported integral. The 2D variant remains genuinely different (the 2D FTC IS the exact piecewise-linear integral, vs `hndA_i_interface[dim=2]` which is the same thing routed through the cotangent code path, modulo whether they alias — worth checking but not the active blocker).

  **Recommended next prompt** (updated 2026-05-27 late):

  > Probe 2 (integrated 3D curvature rewrite) is closed as a no-op: the Stokes barycentric-dual form is mathematically identical to the cotangent form on a piecewise-linear surface (verified bit-identical on both the osc-droplet icosphere and a deliberately irregular fixture). The 1.23x A.5.b residual is the irreducible polygon truncation, not a stencil-choice artefact.
  >
  > **Next probe options (orthogonal, pick by user preference)**:
  >
  > 1. **Probe 3 — `_compute_vd_3d` boundary-NaN fix (~30 min)**. Localise `hyperct/ddg/_geometry.py:91` `RuntimeWarning: invalid value encountered in scalar divide` via traceback; likely degenerate ghost-tet on outer box. Unblocks 3D conservation totals (currently NaN), separate from Tier 2B residual lane. Memory: `project_3d_boundary_dual_bug.md`.
  >
  > 2. **Probe 4 — `redistribute_mass_multiphase` guard fix** (still pending from 2026-04-29 step 1). Test whether the per-phase pressure-guard at `mass_redistribution.py:250` (skips outer phase at reference pressure P0=0) is what makes `redistribute_mass=True` a no-op in the oscillating-droplet case. Tier 2B residual lane, narrowed scope.
  >
  > 3. **Park curvature lane, move to A.5.b 3D long-run regression test (~30 min)**. Confirm the 7.4e-05 max|F| is a stable steady state (no slow drift) by extending `--n-steps 2000`, then encode the floor as a pytest in `ddgclib/tests/`. Locks in the post-migration state as a regression guard.
  >
  > Do NOT add any further curvature-stencil variants. Probe 2 closed the integrated-form line of attack mathematically.

- **2026-05-27 — Simplex-aware curvature apex enumeration shipped across the curvature / bubble / interface / periodic call sites.** Three logically separable batches of working-tree edits (no git operations performed at user request) migrated 12 of the 22 audited `vi.nn.intersection(vj.nn)` sites from the flag-complex 1-skeleton enumeration to the explicit top-dim simplex cache. This unblocks Tier 2B step 1 by removing one of the two confounders in the pointwise curvature stencil: the K_{dim+1} ghost-clique apex contamination on Delaunay-derived meshes (the other being the O(h²) truncation of the pointwise stencil itself, which still requires the integrated rewrite).

  **What landed (working-tree edits — user will stage/commit as they prefer):**

  - **hyperct side** ([`hyperct/ddg/_retriangulation.py`](../hyperct/hyperct/ddg/_retriangulation.py), [`__init__.py`](../hyperct/hyperct/ddg/__init__.py)):
    - New `get_edge_apex_map(HC) -> dict | None` — lazily builds and caches `HC._edge_to_apex` (a `dict[frozenset[id(v_i), id(v_j)], list[apex_vertex]]`) from `HC._simplices`. Returns `None` when no simplex cache is populated, so the caller can fall back to the legacy 1-skeleton path.
    - `invalidate_simplex_cache(HC)` extended to also clear `HC._edge_to_apex` (previously only cleared `_simplices`).
    - Exported via the public `hyperct.ddg` API.

  - **ddgclib curvature** ([`ddgclib/_curvatures_heron.py`](ddgclib/_curvatures_heron.py), [`_curvatures_heron_torch_vectorized.py`](ddgclib/_curvatures_heron_torch_vectorized.py)):
    - Two private helpers added at the top of `_curvatures_heron.py`:
      - `_apex_via_simplex_cache(HC, v_i, v_j)` — returns `None` unless `HC._simplices` holds 2-simplex *triangles* (3-tuples); ignores volumetric tet caches (4-tuples) so `hndA_i` correctly falls back when called on a volumetric mesh.
      - `_apex_via_interface_triangles(HC, v_i, v_j)` — reads `HC.interface_triangles` (the primal-subcomplex sharp interface model from [`ddgclib/geometry/_interface_subcomplex.py`](ddgclib/geometry/_interface_subcomplex.py)), caches a per-edge apex lookup on `HC._interface_edge_to_apex`.
    - `A_i`, `hndA_i`, `int_hndA_i`, `hndA_i_interface` (both modules) now take an optional `HC=None` kwarg. When `HC` is supplied and a usable cache exists, apex enumeration goes through the simplex-aware path; when `HC` is `None` or the cache is absent, the legacy `vi.nn.intersection(vj.nn)` path runs verbatim. No signature break for any existing caller.

  - **ddgclib callers** ([`ddgclib/operators/multiphase_stress.py`](ddgclib/operators/multiphase_stress.py), [`surface_tension.py`](ddgclib/operators/surface_tension.py)):
    - `_interface_surface_tension` now accepts and forwards `HC` so `hndA_i_interface(v, interface_nbs | {v}, HC=HC)` gets the cache (`multiphase_stress_force` already had `HC` in scope).
    - `surface_tension_force`, `surface_tension_acceleration`, `dual_area_heron` now accept `HC=None` and forward to `hndA_i`. Case-study callers in `cases_dynamic/liquid_bridge_equilibrium/` continue to work unchanged.

  - **Bubble / interface helpers** ([`ddgclib/_bubble.py`](ddgclib/_bubble.py), [`ddgclib/geometry/_interface_subcomplex.py`](ddgclib/geometry/_interface_subcomplex.py)):
    - `_bubble.py` — added private `_common_neigh(HC, v1, v2)` at the top of the module; replaced the four `vi.nn.intersection(vj.nn)` sites in `refine_edges`, `refine_boundaries`, `reconnect_long_diagonals` (×2), `get_forces`. `hndA_i(v)` in `get_forces` now passes `HC=HC`. HC was already available in each enclosing function signature, so no signature changes needed.
    - `_interface_subcomplex.py` — `interface_nn(v, HC)` now prefers `HC.interface_edges` membership (a primal subcomplex edge between two interface vertices), falling back to `v.nn & interface_vertices` when `interface_edges` is not populated. The legacy form is correct on closed manifolds but susceptible to spurious K_3 cliques near contact lines.

  - **Periodic** ([`ddgclib/geometry/periodic.py`](ddgclib/geometry/periodic.py) — `_fixup_periodic_duals`): replaced the primal triangle-apex enumeration with the simplex-cache lookup (triangle caches only; falls back when absent or holding tets). The neighbouring `v1.vd.intersection(v2.vd)` on the *dual* graph is left unchanged.

  - **Plotting deliberately not migrated.** The four sites in [`ddgclib/_plotting.py`](ddgclib/_plotting.py) (`vd_i.nn ∩ dset`) operate on the *dual* vertex graph, which is constructed by `compute_vd` and already uses the simplex-aware path; the dual-graph `nn ∩` is exact by construction and not flag-clique-prone. Documented in the conversation transcript so a future reader doesn't read the skip as an oversight.

  **Regression coverage added**: [`ddgclib/tests/test_simplex_aware_curvature.py`](ddgclib/tests/test_simplex_aware_curvature.py) — 11 tests covering (a) `get_edge_apex_map` cache primitives, lazy build, and invalidation; (b) `_apex_via_*` fall-back semantics; (c) machine-precision equivalence between legacy and simplex-aware `hndA_i_interface` on `droplet_in_box_3d` (clean fixture, the two paths *must* agree); (d) tet-mesh fall-back for the vectorized `hndA_i`.

  **Verification — 0 regressions.**

  | suite | result |
  |-------|--------|
  | `pytest ddgclib/tests/ -m "not slow"` (after each batch × 3) | 789 passed |
  | `pytest ddgclib/tests/test_stress.py -m slow` | 8 passed |
  | `pytest ddgclib/tests/` (full, incl. slow) | **802 passed, 0 failures** |
  | `pytest hyperct/tests/` (excl. pre-existing missing `pytest-benchmark` fixture) | **275 passed, 0 failures** |
  | `cases_dynamic/oscillating_droplet/diagnose_static.py` smoke | clean, all 7 diagnostic sections ran |

  (The 39 hyperct errors that initially appeared are all `fixture 'benchmark' not found` — pre-existing environmental issue, no `pytest-benchmark` installed — confirmed unrelated.)

  **What this changes for the plan**: Tier 2B step 1 (curvature-stencil rewrite for the static Young–Laplace droplet) is now *narrowed*. The previously-conflated "interface apex enumeration is wrong" and "interface curvature is pointwise instead of integrated" failure modes are now separable: the simplex-aware path closes the first one without touching the second. The remaining ×1.23 gap above the A.5.a floor (post-Phase-2c) is now provably the pointwise-stencil truncation alone, since apex enumeration is exact. The integrated curvature rewrite is therefore the next probe — but the next prompt should *first* re-run A.5 with the simplex-aware path enabled to measure the residual delta from the migration in isolation, before deciding whether the integrated rewrite is the highest-leverage next move or whether the apex fix alone closed enough of the gap.

  **Recommended next prompt — see end of file (updated 2026-05-27)**.

- **2026-04-29 — Phase 1 + Phase 2 of the 3D retopology blow-up landed. HYPOTHESES KILLED, MECHANISM IDENTIFIED, NEXT FIX SCOPED.**

  All four runs use [`diagnose_a5_bisection.py`](cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py) (now extended with `--split-method` and `--redistribute-mass` CLI flags) and a new focused diagnostic [`diagnose_a5_step1_diff.py`](cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py) that fires the single retopology call once and diffs every interface vertex's per-phase state before vs after.

  | run | 2D A.5.b | 3D A.5.b | reference |
  |--------------------------------------------------------------|----------|----------|-----------|
  | A.5.a (frozen mesh, no retopo) — both dims | 2.37e-3 | 6.02e-5 | (target) |
  | A.5.b baseline `neighbour_count` | 2.37e-3 | 1.44e-3 | (×17 of baseline 8.5e-5 in 3D) |
  | A.5.b `--split-method exact` (setup mismatch — see note) | 3.50e-1 | 6.73e-3 | (catastrophic; setup/runtime split mismatch artefact) |
  | A.5.b `--split-method exact` (consistent end-to-end) | 2.37e-3 | 1.75e-3 | 2D matches A.5.a (perfect retopo neutrality); 3D *slightly worse* than NC |
  | A.5.b `neighbour_count` + `--redistribute-mass` | 2.37e-3 | 1.44e-3 | identical to baseline — redistribute_mass is a no-op here, see below |

  **Phase 1 verdict — `split_method='exact'` is NOT the fix.** In 2D, with `'exact'` consistent end-to-end, A.5.b drops to *exactly* A.5.a (2.37e-3 vs 2.37e-3) — `'exact'` is perfectly retopology-neutral on a static 2D mesh, which proves the 2D residual is purely the curvature stencil on a curved interface (and 2D Delaunay is unique on this point cloud, so no churn). In 3D, `'exact'` is **slightly worse** (1.75e-3 vs `'neighbour_count'`'s 1.44e-3). The user's hypothesis that the neighbour-count majority-vote was the dominant 3D bug is killed.

  - **Setup/runtime split-method mismatch is itself a real bug** — was hidden until this experiment. Setup-time `mps.refresh(reset_mass=True)` (line [`_setup.py:144`](cases_dynamic/oscillating_droplet/src/_setup.py#L144)) and the Young-Laplace mass adjustment ([line 169](cases_dynamic/oscillating_droplet/src/_setup.py#L169): `v.m_phase[1] = rho_d_eq * vol_d`) MUST use the same `split_method` as the runtime `_retopologize_multiphase` partial — otherwise the first retopology step recomputes `dual_vol_phase` under a different policy on an unmoved vertex and the `rho = m/V` ratio jumps by orders of magnitude (in our case ×147 in 2D, ×4.7 in 3D). Fix landed: [`setup_oscillating_droplet`](cases_dynamic/oscillating_droplet/src/_setup.py#L24) now takes `split_method` and `redistribute_mass` parameters that flow through both the initial refreshes and the runtime `retopo_fn` partial, with a docstring callout warning that the two MUST match. Default behaviour is unchanged.

  **Phase 2 verdict — first-to-diverge quantity is `dual_vol_phase` → `rho_phase` → `p_phase`, root cause is Delaunay non-uniqueness on a static cospherical cloud.** Per-vertex diff at the single 3D step-1 retopo call (98 interface vertices, `'neighbour_count'`):

  - **Vertex set is unchanged**: 472 → 472, no `is_interface` flips, no `phase` flips.
  - **`m_phase` is bit-for-bit preserved**: `max | Δm_phase | = 0.0e+00` across all interface vertices. `reset_mass=False` works correctly — Lagrangian mass is conserved exactly.
  - **`dual_vol_phase` shifts are tiny but nonzero**: `max | Δdual_vol_phase | = 1.03e-8`, `mean = 5.02e-9`, `median = 6.32e-17`. Microscopic geometric perturbation.
  - **`rho_phase` swings massively**: `max | Δρ_phase | = 7.10e+02` kg/m³ (i.e. 70% of `rho_o`).
  - **`p_phase` swings**: `max | Δp_phase | = 3.75e+02` Pa.
  - **|F| jumps**: top interface vertices' max |F| goes 2.30e-5 → 4.88e-4 to 2.48e-5 → 1.49e-3 (×21 to ×60).
  - **48 cross-phase edges added + 48 removed** in the 1-ring of interface vertices. This is the smoking gun: the mesh hasn't moved, but Delaunay reconnects ~48 cross-phase edges per retopo step.

  The mechanism is now arithmetic: at setup, `compute_phase_masses` sets `m_phase[k] = rho0_k * dual_vol_phase[k]_setup`, so `rho_setup = rho0` exactly. After retopo with `reset_mass=False`, `m_phase` is Lagrangian-frozen but `dual_vol_phase[k]` shifts because (a) Delaunay reconnects ~48 cross-phase edges on a static cloud (the `'neighbour_count'` fraction shifts), or (b) the `'exact'` geometric split itself is not invariant under cross-phase edge flips of the same point cloud. Even though `Δvol_phase ~ 1e-8` is tiny, when `vol_phase[k]` itself is small near the interface (~1e-9 to 1e-8 for the off-side phase), the relative perturbation is order 1 → `rho = m/V` deviates from `rho0` by factors of 10x → `p = eos.pressure(rho)` swings by hundreds of Pa → interface stress force jumps by ×20-60.

  **The deeper root cause is 3D Delaunay non-uniqueness on a near-cospherical static cloud.** In 2D the Delaunay tessellation on a generic point cloud is unique (3 points on a circle is the boundary case). In 3D, the Delaunay tessellation is non-unique whenever 4 or more points are cospherical, which is approximately the case for our droplet interface vertices (they were placed *on* a sphere by the domain builder). Even an idempotent re-run of Delaunay on the same coordinates can return a different tessellation, and the two flip ~48 cross-phase edges apart per step. This is why 2D is benign (Delaunay-of-static-cloud is idempotent) and 3D is not.

  **Why `redistribute_mass=True` did not help**: the per-phase pressure-preserving redistribution in [`mass_redistribution.py:250`](ddgclib/operators/mass_redistribution.py#L250) skips any vertex/phase pair where the snapshotted `p_phase[k] < 1e-30`. The oscillating-droplet case uses `P0=0`, so the outer phase pressure at equilibrium is exactly 0 — the entire outer phase is unconditionally skipped by the redistribution. Yet the outer phase is precisely the one whose `rho` collapses from 1000 to ~91 kg/m³. The guard conflates "phase is absent at v" (legitimate skip) with "phase is at reference pressure" (illegitimate skip). This is a third bug: the guard should test `dual_vol_phase[k]_before > 1e-30` against a pre-retopo geometry snapshot, not the pressure value. Fix would require also snapshotting `dual_vol_phase` in [`snapshot_pressure_multiphase`](ddgclib/operators/mass_redistribution.py#L42-L51) (or adding a new `snapshot_geometry_multiphase`) and consulting it in the guard.

  **Tier 2B attack order — REVISED 2026-04-29:**

  1. **Fix `redistribute_mass_multiphase` guard at [line 250](ddgclib/operators/mass_redistribution.py#L250)** to use a pre-retopo `dual_vol_phase` snapshot (existence test) instead of a `p_phase` magnitude test. Requires a tiny extension to `snapshot_pressure_multiphase` (or a new sibling fn) and a one-line guard change. Re-run A.5.b 3D `--redistribute-mass` and check whether interface |F| collapses toward A.5.a's 6.02e-5. This is THE next probe — cheap, targeted, and tests the now-narrowed hypothesis.
  2. **3D Delaunay non-uniqueness on cospherical clouds** — the deeper root cause. Possible mitigations: (a) symbolic perturbation in hyperct's 3D Delaunay (so a static cloud has a unique tessellation), (b) skip retopology when no vertex has moved more than O(eps), (c) lock interface-edge connectivity across reconnection (interface-preserving 3D adaptive remesh — currently a roadmap item, blocked by hyperct). Option (b) is the cheapest probe.
  3. **3D `_compute_vd_3d` setup-time NaN at outer-box boundary** ([`hyperct/ddg/_geometry.py:91`](../hyperct/hyperct/ddg/_geometry.py#L91)) — separate priority, surfaces as NaN mass/volume in conservation diagnostics (Phase 3 in the 2026-04-28 entry). Currently masked because retopology overwrites boundary `dual_vol = 0`.
  4. **2D — integrated curvature operator (Tier 2B step 1 for 2D).** Unchanged. 2D residual is now provably 100% curvature stencil (A.5.b 2D `'exact'` matches A.5.a 2D exactly), needs the integrated γ-flux form.
  5. **3D curvature stencil.** Re-evaluated after 1–2 land.

  **Recommended next prompt (Tier 2B, M-lane, 3D first — REVISED 2026-04-29)**:

  > Phase 2 of the 3D retopology blow-up identified the mechanism: per-phase `rho = m/V` blows up because Delaunay reconnects ~48 cross-phase edges per static-cloud retopo step, shifting `dual_vol_phase` while `m_phase` is Lagrangian-frozen. `redistribute_mass=True` does not help in this setup because `mass_redistribution.py` guards on `p_phase_before > 1e-30`, which skips the outer phase at reference pressure (`P0=0`).
  >
  > **Next probe (~1 hour)**: Fix the guard in [`redistribute_mass_multiphase`](ddgclib/operators/mass_redistribution.py#L199) to test pre-retopo `dual_vol_phase[k] > 1e-30` instead of `p_phase[k] > 1e-30`. This requires (a) extending `snapshot_pressure_multiphase` (or adding `snapshot_geometry_multiphase`) to record per-phase `dual_vol_phase`, (b) threading the new snapshot through `_retopologize_multiphase` to `redistribute_mass_multiphase`, (c) the guard change. Then re-run `python cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py --redistribute-mass --n-steps 100` and check whether 3D A.5.b drops from 1.44e-3 toward A.5.a's 6.02e-5. If yes, the production fix is to default `redistribute_mass=True` for multiphase setups (plan item M1) AND fix the guard.
  >
  > **Parallel orthogonal investigation**: 3D Delaunay non-uniqueness on the (near-cospherical) interface point cloud. Even with the redistribute_mass fix, the underlying retopology churn (48 cross-phase edges flipping per step on a static cloud) is the deeper instability driver and should be characterised in hyperct. Cheapest probe: skip retopology entirely when the maximum vertex displacement since the last retopo is below a threshold.
  >
  > Do NOT touch interface-curvature code (Tier 2B step 1 / 2D operator) until the redistribute_mass-guard fix lands — it's still cheap-test territory.

- **2026-04-28 — A.5.3D re-run post-refactor. RETOPOLOGY BLOW-UP NOT CLEARED — CAUSE LOCALISED.** Re-ran [`diagnose_a5_bisection.py`](cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py) (n=100) against the post-refactor codebase. Numbers are bit-for-bit identical to the pre-refactor probe (raw JSON: `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection.json`; pre-refactor backup at `…/a5_bisection_pre_refactor.json`):

  | dim | A.5.a (frozen, no retopo) — pre / post | A.5.b (retopo + u≡0, peak) — pre / post | full-dynamic baseline |
  |-----|----------------------------------------|------------------------------------------|-----------------------|
  | 2D | 2.37e-3 (×0.625) / 2.37e-3 (×0.625) | 2.37e-3 (×0.625) / 2.37e-3 (×0.625) | 3.80e-3 |
  | 3D | 6.02e-5 (×0.708) / 6.02e-5 (×0.708) | **1.44e-3 (×16.96)** / **1.44e-3 (×16.96)** | 8.50e-5 |

  Verification probe `/tmp/verify_a5_simplex_path.py` confirmed the post-refactor simplex-aware path is being exercised end-to-end (`HC._simplices = 2515` populated immediately after `setup_oscillating_droplet`, repopulated after the first `retopo_fn` call; `boundary_from_simplices` taken; `compute_vd` reads the cache). A.5.b therefore is exercising the canonical post-refactor code, not a legacy fallback.

  - **Interface-|F| trajectory in A.5.b is sharply discontinuous at step 1.** From the JSON history:
    - **t=0 (post-setup, pre-retopo)**: max |F| = **6.02e-5** — identical to A.5.a, clean curvature-stencil residual.
    - **step 1 (after exactly one `retopo_fn` call)**: max |F| = **1.44e-3** — ×24 jump.
    - **steps 2–100**: 1.44e-3, completely flat.
    The 24× jump is therefore caused by exactly one retopology call on a mesh nobody moved.
  - **The NaN boundary-dual signal is a SEPARATE orthogonal bug, not the cause of the |F| jump.** The verify probe shows:
    - Immediately after `setup_oscillating_droplet(dim=3)`: 95 outer-box boundary vertices already carry `dual_vol = NaN` (`hyperct/ddg/_geometry.py:91 RuntimeWarning: invalid value encountered in scalar divide`). Interface |F| at this state is the clean 6.02e-5.
    - After one `retopo_fn` call: 0 NaN dual_vols, 96 zero dual_vols (boundary vertices get `dual_vol = 0` per [`_integrators_dynamic.py:225`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L225)). Interface |F| is 1.44e-3.
    Retopology actually **cleans up** the NaN boundary duals (zeroing them is correct for boundary vertices) — yet the interface |F| jumps 24× across the same call. So the boundary-dual NaN at setup is real and needs fixing, but it's NOT the 3D retopology blow-up's cause. The cause is downstream of the boundary-dual handling, in code that mutates **interface state** during retopology.
  - **Most likely cause** (to confirm in the next prompt): the multiphase refresh inside [`_retopologize_multiphase`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L325-L406), specifically `mps.refresh(HC, dim, reset_mass=False, split_method='neighbour_count')`. Delaunay re-triangulation perturbs cross-phase connectivity around the interface; the default `'neighbour_count'` split is a 1-ring majority-vote approximation that does not exactly preserve per-phase dual volumes (`v.dual_vol_phase`) under reconnection. `multiphase_stress_force` reads `v.p_phase` (computed from `v.m_phase / v.dual_vol_phase` via `mps.compute_phase_pressures`), and a small bias in `dual_vol_phase` becomes a large bias in `p_phase` — which the per-phase stress then surfaces as an interface force imbalance. The cheap test is to flip `split_method='exact'` (already implemented in [`_dual_split_2d.py`](ddgclib/geometry/_dual_split_2d.py) for 2D and 3D) and re-run A.5.b 3D.
  - **3D does NOT match the 2D pattern.** 2D verdict: `mixed` (curvature ≈ retopology ≈ 0.625× baseline, retopology benign because Delaunay-of-static-cloud is idempotent). 3D verdict: `retopology-dominant` (×17 of baseline, ×24 of A.5.a) — driven by multiphase refresh, not the simplex-aware dual recomputation itself.
  - **Tier 2B attack order — REVISED:**
    1. **3D — `split_method='exact'` for 3D multiphase refresh.** Cheap probe (one-line setup change in `_retopologize_multiphase`). If it drops A.5.b 3D from 1.44e-3 toward 6e-5, the 3D retopology blow-up is solved with one default flip. If it doesn't, the bug is elsewhere in `mps.refresh` and we instrument deeper.
    2. **3D — `_compute_vd_3d` setup-time NaN at boundary** (separate priority). 95/96 outer-box boundary vertices come out of the initial `compute_vd` with `dual_vol = NaN`. Currently masked because retopology overwrites them with zeros, but it's a real bug in the geometric primitive at `hyperct/ddg/_geometry.py:91` and will surface again as soon as anyone tries to use boundary dual volumes.
    3. **2D — integrated curvature operator (Tier 2B step 1 for 2D).** Unchanged: 63% of 2D baseline is the static curvature-stencil residual, requires replacing pointwise `hndA_i_interface` / `surface_tension_force_2d` with an integrated γ-flux form.
    4. **3D curvature stencil.** Re-evaluated after 1 lands.
    5. **Dynamic coupling residual (~37% of 2D baseline).** Same as before — only chase after 1–4 are done.

- **2026-04-26 — Core multiphase refactor merged & re-validated.** Major refactor (commit 8321c71 "ENH: Core multiphase refactoring" + follow-ups) replaced the manual `HC._simplices = ...` workaround with a proper hyperct API (`hyperct.ddg.connect_and_cache_simplices`, `boundary_from_simplices`, `invalidate_simplex_cache`). The integrator [`_integrators_dynamic.py:188-196`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L188-L196) now uses these as the **primary path**, not a workaround. New regression guard: 13-test suite [`test_simplex_aware_duals.py`](ddgclib/tests/test_simplex_aware_duals.py) covers 2D simplex caching, simplex-aware `compute_vd`, `boundary_from_simplices` (2D + 3D), invalidation semantics, and domain-builder integration. Effect on this plan: D1's "3D fallback loses linear precision" sub-bullet is now **resolved** — the simplex-aware path is canonical. Re-validation results:
  - A.1 conservation: 18/18 pass.
  - A.2 flat-interface: 4/4 pass with identical magnitudes (2D 2.22e-16 / 2.22e-16; 3D 1.05e-15 / 1.04e-15).
  - Simplex-aware regression guard: 13/13 pass.
  - Full fast suite: **778 passed, 1 skipped, 14 deselected, 2 xfailed, 0 regressions** (up from 765 pre-refactor).
- **2026-04-24 — A.1 complete.** Added [`ddgclib/data/_conservation.py`](ddgclib/data/_conservation.py) (`compute_conservation`, `as_jsonable`, `drift_fractions`) and extended `StateHistory` with opt-in `conservation=True, dim=...` that merges KE, momentum, mass/volume totals + per-phase, `h_min`/`h_max`, velocity/pressure extrema into each snapshot's diagnostics. 18 new tests in [`ddgclib/tests/test_conservation.py`](ddgclib/tests/test_conservation.py), full suite 761 passed / 0 regressions.
- **2026-04-24 — A.2 complete. D4 FLAT-INTERFACE BUG RULED OUT.** New test file [`ddgclib/tests/test_multiphase_flat_interface.py`](ddgclib/tests/test_multiphase_flat_interface.py), both variants pass at machine precision in both dimensions:
  - 2A.i 2D (γ=0, flat y=0): bulk `|F|` = 3.14e-16, interface `|F|` = 2.22e-16
  - 2A.ii 2D (γ=0.05, flat y=0, κ=0 stencil): same, 2.22e-16
  - 2A.i 3D (γ=0, flat z=0 Kuhn-decomposed cube): bulk 1.19e-15, interface 1.05e-15
  - 2A.ii 3D (γ=0.05, flat z=0): same, 1.04e-15
  Full suite 765 passed / 0 regressions. **Interpretation**: the per-phase summed pressure-flux formula cancels exactly, and `hndA_i_interface` on a planar interface triangulation returns exactly zero. The residual `~3.8e-3` (2D) / `~8.5e-5` (3D) on the static droplet (see D4) therefore does NOT come from the per-phase stress sum itself — it comes from the **curvature stencil on a curved interface**, and/or from **retopology**. M-lane scope is narrowed accordingly (see updated M3).
  - **3D mesh-construction prerequisite flagged**: default `Complex(3).triangulate() + refine_all()` does NOT produce a planar z=0 interface (64/817 tets cross z=0, vertices land at z=-0.25). The test built an explicit Kuhn-decomposed cube (6 tets per unit cube via axis-monotone diagonals) so z=0 lies on a cube-face layer and no tet crosses it. This is a prerequisite for any flat-interface-in-3D benchmark going forward, including Tier 2B's spherical droplet diagnostic if we ever want a mesh-aligned reference.
- **2026-04-24 — A.5 complete. CURVATURE vs RETOPOLOGY BISECTED. 2D and 3D DIVERGE.** New diagnostic [`cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py`](cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py) runs both flavours in both dimensions, reuses `compute_conservation` (A.1) and the existing `setup_oscillating_droplet(epsilon=0)`. JSON dump at `cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection.json`. Numbers (max |F| on interface vertices, n=100 steps):

  | dim | A.5.a (frozen, no retopo) | A.5.b (retopo + u≡0, peak over 100 steps) | full-dynamic baseline |
  |-----|---------------------------|-------------------------------------------|-----------------------|
  | 2D | 2.37e-3 (×0.625) | 2.37e-3 (×0.625, flat after step 1) | 3.80e-3 |
  | 3D | 6.02e-5 (×0.708) | **1.44e-3 (×16.96)**, flat after step 1 | 8.50e-5 |

  - 2D: A.5.a ≈ A.5.b. The first retopology step actually *decreases* max |F| from 2.37e-3 to 2.27e-3 and stays there (Delaunay of a static cloud is idempotent). Retopology is benign on a static 2D mesh; the ~63% of the 3.80e-3 baseline captured by A.5.a is **pure curvature-stencil error on the curved interface**. The remaining ~37% comes from the dynamic coupling (motion feeding retopology feeding force noise).
  - 3D: A.5.a already captures 71% of the baseline at 6e-5 — the spherical-interface curvature stencil is the dominant static residual. But a **single retopology step** on a mesh nobody moved takes max |F| to 1.44e-3, a 24× jump over A.5.a and 17× over the quoted baseline. The step-1 jump is consistent with the `_compute_vd_3d` boundary-dual / ghost-tet bug already flagged in memory ([project_3d_boundary_dual_bug.md](../.claude/projects/-home-stefan-endres-projects-ddgclib/memory/project_3d_boundary_dual_bug.md)). Conservation totals return NaN in 3D because 95 outer-box boundary vertices carry `dual_vol = NaN` — same root cause, orthogonal to the interface force but a smoking gun that 3D retopology is actively corrupting dual data.
  - **Attribution & Tier 2B attack order (recommended):**
    1. **3D first — fix `_compute_vd_3d` boundary / ghost-tet handling.** The 24× retopology blow-up + NaN boundary dual volumes is the single largest fixable signal anywhere in this bisection; every 3D dynamic number downstream is suspect until it lands. This is Tier 2B step 2 (retopology) applied to 3D only.
    2. **2D — integrated curvature operator (Tier 2B step 1).** Retopology is not the 2D bottleneck; the 63% of baseline captured on a frozen mesh must be driven down by replacing the pointwise `hndA_i_interface`/`surface_tension_force_2d` path with an integrated γ-flux form that is exactly consistent with the same-phase pressure flux on interface edges.
    3. **3D curvature stencil (Tier 2B step 1 for 3D)** comes next — it already sits at 71% of baseline and will re-dominate once the retopology bug is gone.
    4. **Dynamic coupling residual** (the ~37% of 2D baseline not captured by either A.5 flavour) is a property of motion × retopology and will only be probed once 1 and 2 are done and the static residual is near machine precision. Don't chase it yet.
  - M3 sub-items are re-prioritised accordingly: M3.2 (3D adaptive/retopology fix) moves ahead of M3.1 (2D integrated surface-tension) for 3D cases; for 2D cases M3.1 stays first.

## Context

`ddgclib` has reached a notable milestone: the **spatial discretisation** (integrated FVM stress operator built on hyperct's DDG dual mesh) is validated to machine precision on a battery of equilibrium benchmarks — Hagen–Poiseuille, hydrostatic column, linear-precision gradient tests, and mixed pressure–viscous Poiseuille equilibrium. Concretely:

- Linear scalar/vector fields: `< 1e-13` error on every dual method and seed ([`_integrated_benchmark_cases.py:43-123`](benchmarks/_integrated_benchmark_cases.py#L43-L123), [`run_integrated_benchmarks.py:210-260`](benchmarks/run_integrated_benchmarks.py#L210-L260))
- Constant pressure gradient (hydrostatic) force: machine precision ([`_integrated_benchmark_cases.py:618-664`](benchmarks/_integrated_benchmark_cases.py#L618-L664))
- Quadratic velocity viscous flux on symmetric meshes: machine precision ([`_integrated_benchmark_cases.py:667-756`](benchmarks/_integrated_benchmark_cases.py#L667-L756))
- Combined Poiseuille equilibrium (pressure + viscous): machine precision on symmetric meshes, O(h²) on jittered ([`_integrated_benchmark_cases.py:709-756`](benchmarks/_integrated_benchmark_cases.py#L709-L756))
- 3D DEC p_ij dual with simplex-aware caching (April 2026 fix in [`docs/3d_simplex_aware_dual_fix.md`](docs/3d_simplex_aware_dual_fix.md))
- Multiphase bulk-phase stress (April 2026 own-phase pressure fix): machine precision on **bulk** vertices, improved 13 orders of magnitude ([`docs/3d_multiphase_interface_pressure_fix.md`](docs/3d_multiphase_interface_pressure_fix.md))

Despite this, **every dynamic case in `cases_dynamic/` is unstable or imperfect** in one way or another: energy drifts, vertex counts grow, retopology corrupts fields, outlet BCs cause backflow, and multiphase cases (oscillating droplet, dam break) drift or blow up. The **gap between machine-precision equilibrium and dynamic stability** is the problem to close.

This plan identifies where noise/instability enters the dynamic pipeline, then prescribes a **hierarchy of isolation benchmarks** that strips concerns one at a time so we can attribute and fix each failure mode in turn.

## Where the instability enters (diagnosis)

Since the spatial operators are exact on equilibrium, failure must live in one of five places. Ordered by expected impact:

### D1. Retopologization every step injects energy/noise

[`_integrators_dynamic.py:48-237`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L48-L237) (`_retopologize`) runs at the top of every integrator step. Problems:

- **Volume-change pressure noise**: Delaunay retriangulation (default `remesh_mode='delaunay'`) redraws the dual — `Vol_i` can change substantially even when vertices barely moved. Unless `redistribute_mass=True` with a real EOS, `v.p` is effectively overwritten by stale masses on new volumes → spurious pressure gradients → spurious stress forces. Most dynamic cases do **not** enable redistribution ([`mass_redistribution.py:82-150`](ddgclib/operators/mass_redistribution.py#L82-L150)).
- ~~**3D fallback loses linear precision**~~: **RESOLVED 2026-04-26** by the core multiphase refactor. The previous `HC._simplices = ...` workaround that triggered `_compute_vd_3d_simplex_aware` is now the canonical path via [`hyperct.ddg.connect_and_cache_simplices`](hyperct/hyperct/ddg/_retriangulation.py) and [`boundary_from_simplices`](hyperct/hyperct/ddg/_boundary.py). Linear precision is recovered to machine epsilon on jittered 3D meshes; regression-guarded by [`test_simplex_aware_duals.py`](ddgclib/tests/test_simplex_aware_duals.py) (13 tests) and `test_p_ij_linear_precision_jittered_3d` in [`test_stress.py`](ddgclib/tests/test_stress.py).
- **Boundary misclassification**: Vertices with `dual_vol < 1e-30` are auto-marked boundary and frozen ([`_integrators_dynamic.py:211`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L211)), which can freeze interior fluid vertices during tangle.
- **Adaptive remesh blocked**: `remesh_mode='adaptive'` is 2D-only and is currently **blocked by upstream hyperct bugs** ([`DEVELOPMENT.md:69-70`](DEVELOPMENT.md#L69-L70)): `edge_split_2d` mass-averaging inflates total mass (9.7 → 187 over 100 steps in the static-droplet stress test), and the global `h_local` threshold triggers unbounded splits in coarse regions.

### D2. Boundary conditions destroy energy conservation

[`ddgclib/_boundary_conditions.py`](ddgclib/_boundary_conditions.py):

- `OutletDeleteBC` ([lines 395-440](ddgclib/_boundary_conditions.py#L395-L440)): truncated dual cells at the outlet produce imbalanced stress forces that push vertices backward; `backflow_clamp` destroys kinetic energy non-physically.
- `OutletBufferedDeleteBC` ([lines 443-522](ddgclib/_boundary_conditions.py#L443-L522)): freezes velocity inside the ghost buffer, overwriting the integrator's stress update → breaks energy conservation.
- `PeriodicInletBC` ([lines 532-680](ddgclib/_boundary_conditions.py#L532-L680)): injected ghost-mesh vertices carry fields that have evolved independently → field discontinuities at the inlet merge; merge tolerance is an unprincipled tuning knob.
- `PressureReservoirBC` relaxation timescale is user-provided and un-calibrated.

### D3. No built-in conservation diagnostics

`StateHistory` ([`_history.py:27-200`](ddgclib/data/_history.py#L27-L200)) records snapshots but does **not** track kinetic/potential energy, momentum, mass (per phase), volume (per phase), or divergence. Every case ships its own ad-hoc diagnostic script (`diagnose_*.py` in `cases_dynamic/oscillating_droplet/`). Instability is silent until blowup, and cases cannot be compared.

### D4. Multiphase — residual interface-vertex force (NARROWED by A.2)

Per [`docs/3d_multiphase_interface_pressure_fix.md:147-177`](docs/3d_multiphase_interface_pressure_fix.md#L147-L177), after the April 2026 own-phase pressure fix:

- Bulk-phase vertices: machine precision ✅
- Interface vertices: **~3.8e-3 in 2D, ~8.5e-5 in 3D** (static-equilibrium per-vertex |F|) ❌

**A.2 result (2026-04-24)**: On a mesh-aligned FLAT interface with both γ=0 and γ>0 κ=0 variants, per-vertex `|F|` is machine precision in 2D and 3D ([`test_multiphase_flat_interface.py`](ddgclib/tests/test_multiphase_flat_interface.py)). The per-phase pressure-flux cancellation and the planar `hndA_i_interface` are BOTH exact. The residual ~3.8e-3 / ~8.5e-5 therefore MUST come from one (or both) of:

1. **Curvature stencil error on a curved interface** — `hndA_i_interface` on a polygonal approximation of a circle/sphere has O(h²) truncation, not zero; it does not exactly cancel the Laplace pressure jump on the discrete geometry.
2. **Retopology artefacts** — Delaunay retriangulation redraws the interface sub-polyline each step, changing the discrete curvature samples even at "static equilibrium" runs.

A.5 (newly added below) is a targeted bisection that attributes the residual to (1), (2), or both.

**Current live metrics** (from `cases_dynamic/oscillating_droplet/results/score.json` and `results_equilibrium/score.json` in-tree):

- Static droplet 2D: `summary = 1.20e-3`, `mass_drift = 1.8e-16`, `max_KE_normalized = 1.02e-8` — **already close to the old 7.41e-3 target**; the original plan's 7.41e-3 number is stale.
- Oscillating droplet 2D: `tail_growth = 1.68`, `l2_error ≈ 5.6` — the **live gap** is the oscillation score, not the static equilibrium score.

### D5. Multiphase — interface fragility during remesh

- Default `split_method='neighbour_count'` ([`multiphase.py:417-503`](ddgclib/multiphase.py#L417-L503)) is a 1-ring majority-vote approximation. The `'exact'` geometric split exists in [`_dual_split_2d.py`](ddgclib/geometry/_dual_split_2d.py) (2D and 3D via PCA-plane clipping) but is **not opt-in by default**.
- Global Delaunay retriangulation creates cross-phase edges at the interface each step; `assign_simplex_phases_from_vertices` re-labels via vertex majority vote ([`multiphase.py:237-287`](ddgclib/multiphase.py#L237-L287)), which is not exact recovery. `strict_closure=False` during runtime masks conformity violations.
- The interface-preserving adaptive remesh (`hyperct.remesh`) is the correct tool but is blocked by D1's upstream bugs.

## Proposed direction: hierarchical isolation benchmarks

The strategy is to **build a ladder of dynamic benchmarks** where each rung adds exactly one source of complexity. When a rung fails, the newly added concern is provably the culprit. Each rung uses the same conservation-diagnostics harness (see Tier 0) and ships as a `pytest` regression.

### Tier 0 — Conservation diagnostics (ENABLER, do first)

Without this, every benchmark below is subjective. Add a `ConservationDiagnostics` callback (or extend `StateHistory`) that records **every step**:

- Total kinetic energy `0.5 Σ m_i |u_i|²` (overall and per phase)
- Total momentum `Σ m_i u_i`
- Total mass (overall and per phase)
- Total volume (overall and per phase, from dual cells)
- min/max `|u|`, min/max `v.p`
- Vertex count, `h_min`, `h_max`
- Step-to-step max force magnitude

Produce, per case, a standard conservation-plot PDF (`fig/conservation.pdf`) with KE(t), mass drift, vertex-count drift. Fail a pytest if drift exceeds a per-case tolerance.

**Critical files**: [`ddgclib/data/_history.py`](ddgclib/data/_history.py), [`ddgclib/dynamic_integrators/_integrators_dynamic.py`](ddgclib/dynamic_integrators/_integrators_dynamic.py) (to call the diagnostic hook inside the loop).

**Effort**: small (few hundred lines). **Payoff**: every failure below becomes quantitatively diagnosable.

### Tier 1 — Single-phase dynamic benchmarks

Each one strips one concern from the full dynamic pipeline.

- **1A. Frozen-mesh transient decay.** Disable retopology (`retopologize_every=None` or `retopologize_fn=False`), initialise a damped sinusoid velocity field in a periodic box, integrate to steady state. Validates **pure time integration + stress operator dynamics** in isolation. Expect machine-precision mass/volume conservation, KE decay matching analytical `e^{-2νk²t}` to O(dt²). If this fails: time integrator bug.
- **1B. Rigid-body advection with retopology on, zero stress — TWO FLAVOURS.** Uniform velocity on all vertices, `mu=0`, no pressure. The two flavours bisect retopology:
  - **1B.i** `skip_triangulation=True` — retopology recomputes duals each step but does NOT rebuild connectivity. Uses the existing flag at [`_integrators_dynamic.py:188`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L188).
  - **1B.ii** Full Delaunay rebuild (default).
  If 1B.i conserves and 1B.ii doesn't, the bug is connectivity churn (Delaunay on moving clouds), not dual recomputation. If 1B.i already fails, it's the dual-volume refresh / mass bookkeeping. This bisection makes the fix target concrete for free.
- **1C. Transient Poiseuille from rest.** Existing `Hagen_Poiseuile` setup, start from `u=0`, apply pressure gradient, watch it develop to steady state. Validates **full single-phase pipeline**. Expect O(h²) convergence of centerline velocity to analytical, monotone KE rise to steady-state plateau, mass drift < 1e-12.
- **1D. Small-perturbation hydrostatic.** Existing `Hydrostatic_column` with a tiny initial velocity perturbation. Must damp to zero KE at rate `ν k²`. Validates **pressure–gravity–viscous balance dynamically**.
- **1E. Periodic inlet/outlet vs true periodic.** Same domain, run once with `PeriodicInletBC + OutletDeleteBC`, once with native periodic BCs. Field discrepancy isolates the **inlet/outlet BC artefacts** (D2). This is what justifies fixing the BCs.

### Tier 2 — Multiphase static benchmarks

- **2A. Flat-interface zero-velocity — TWO VARIANTS.** The canonical check from [`docs/interface_stress_rewrite.md:93-95`](docs/interface_stress_rewrite.md#L93-L95). Two phases, horizontal interface y=0, uniform pressure per side, u=0.
  - **2A.i** `γ=0` (no surface tension): isolates per-phase summed pressure-flux cancellation. `F_i=0` machine precision expected.
  - **2A.ii** `γ>0` with exact κ=0: isolates whether the surface-tension code path leaks noise even when the curvature stencil evaluates to zero. This catches whether `F_st` numerically returns exactly zero when the interface is straight, or whether its geometric implementation introduces floor noise.
  If either fails, the per-phase stress ([`multiphase_stress_force`](ddgclib/operators/multiphase_stress.py)) has a cancellation bug and D4 cannot be addressed until it passes. This is the highest-leverage single test in the ladder.
- **2B. Static circular / spherical Young–Laplace droplet.** Existing [`static_droplet_2D.py`](cases_dynamic/oscillating_droplet/static_droplet_2D.py). Current live scores: `summary = 1.20e-3`, `max_KE_normalized = 1.02e-8`, `mass_drift = 1.8e-16`. Drive per-vertex `|F|` on interface vertices from current ~3.8e-3 (2D) / ~8.5e-5 (3D) toward machine precision. **Per A.2, the fix is no longer "make the per-phase stress sum cancel" — it's "make the curvature-vs-pressure-jump discretisation consistent on a curved interface under retopology".** Attack plan driven by A.5 probe results:
  1. **If A.5 shows residual is dominated by curvature stencil on a curved interface**: replace the pointwise `hndA_i_interface` with an integrated form that is by-construction exactly consistent with the same-phase pressure flux on interface-to-interface edges. Candidates: integrate γ around each interface vertex's dual sub-edge (2D has `surface_tension_force_2d` already in [`curvature_2d.py:91-181`](ddgclib/operators/curvature_2d.py#L91-L181) — verify it's called); in 3D, sum γ·(edge tangent outer product) on the interface sub-surface loop.
  2. **If A.5 shows residual is dominated by retopology**: switch the static-droplet case to `remesh_mode='adaptive'` (requires A.4) and/or reduce retopology frequency. Test whether `split_method='exact'` on interface vertices ([`_dual_split_2d.py`](ddgclib/geometry/_dual_split_2d.py)) moves the needle.
  3. Both likely matter to different degrees in 2D vs 3D — A.5 numbers decide the weighting.
- **2C. Static sessile droplet.** Contact line equilibrium. Add-on after 2B passes.

### Tier 3 — Multiphase dynamic benchmarks

- **3A. 2D oscillating droplet (Rayleigh–Lamb).** Existing [`oscillating_droplet_2D.py`](cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py). Current `tail_growth = 1.68`, `l2_error ≈ 5.6`; target `tail_growth < 1.0`, `l2_error < 0.2`. Cannot pass without 2B at machine precision.
- **3B. 3D oscillating droplet.** Needs 3D metric harness and baseline first ([`DEVELOPMENT.md:59-68`](DEVELOPMENT.md#L59-L68) notes it is missing).
- **3C. Dam break.** Gross-scale validation of multiphase dynamics; regression only, no analytical reference.

## Recommended execution order — ruthlessly prioritised

Execution is in two phases: a **three-probe diagnostic pass** that identifies which failure bucket dominates in ~1 week of work, then **parallel single-phase + multiphase attack lanes** driven by the probe results. Upstream hyperct fixes run concurrent to Phase A as a third lane (they're in a sibling repo, do not block Phase A probes, and unblock adaptive remesh for Tier 2B/3A later).

### Phase A — Three probes + upstream fixes, all concurrent (~1 week)

These four workstreams have no dependencies among each other and can be done in parallel.

**A.1 Tier 0 diagnostics, MINIMAL form** (~100 lines). Extend `StateHistory.callback` at [`_history.py:71`](ddgclib/data/_history.py#L71) to record KE, mass-per-phase, vertex count, max `|F|` each step. Skip PDF plots initially — dump JSON. One harness reused by every benchmark below. This is the enabler for every probe.

**A.2 Tier 2A flat-interface pytest** (~50 lines using `multiphase_stress_force` directly, no integrator). Both variants (2A.i γ=0, 2A.ii γ>0 with κ=0). If it fails, D4 is real and the multiphase lane is fully justified. If it passes at machine precision, D4 is misdiagnosed and we save weeks of interface-curvature work.

**A.3 Tier 1A (frozen-mesh) + 1B.i/1B.ii (skip_triangulation vs full Delaunay)** (~200 lines total, using existing integrator flags). Bisects "is it the integrator, the dual recompute, or the Delaunay rebuild" in hours. Result determines whether D1 or D2 dominates in single-phase.

**A.5 (NEW) Static-droplet frozen-mesh bisection** (~150 lines). Added after A.2 ruled out the flat-interface bug. Question: is the ~3.8e-3 (2D) / ~8.5e-5 (3D) interface residual on the static circular/spherical droplet driven by the curvature stencil on a curved interface, or by retopology redrawing the interface sub-polyline each step?

Two flavours, both starting from the existing [`static_droplet_2D.py`](cases_dynamic/oscillating_droplet/static_droplet_2D.py) (and a 3D spherical analogue):

- **A.5.a** Disable retopology entirely (`retopologize_fn=False`), hold the initial circular/spherical mesh fixed, evaluate `multiphase_stress_force` + surface tension on every interface vertex. Record max `|F|` per step over a short window. This value is the **pure curvature-stencil-vs-pressure-jump residual** on a curved but fixed mesh.
- **A.5.b** Enable retopology as normal, but freeze all velocities to zero every step (`u=0` forced via a callback). Record max `|F|` per step. This value is the **pure retopology-induced residual** with no velocity-driven geometry change.

Compare the two residuals to the full-dynamic baseline (~3.8e-3 / ~8.5e-5). If A.5.a matches the baseline → curvature stencil dominates, prioritise the 2B attack plan step 1 (integrated curvature operator). If A.5.b matches → retopology dominates, prioritise step 2 (adaptive remesh + `split_method='exact'`). If both contribute → both fixes are needed and we prioritise by magnitude.

This probe leverages the A.1 diagnostics harness directly. Result determines whether M3 starts with a curvature rewrite or a retopology freeze.

**A.4 Upstream hyperct fixes** (in the sibling repo [`/home/stefan_endres/projects/hyperct/`](../hyperct/)). Two bounded fixes:
- [`hyperct/remesh/_operations_2d.py:198`](../hyperct/hyperct/remesh/_operations_2d.py#L198) — `edge_split_2d` currently sets midpoint mass to the arithmetic mean of endpoints, inflating total mass on every split. Fix: weight by sub-edge length so `m_mid = (L_a·m_a + L_b·m_b) / (L_a + L_b)` for the collapsed two-edge decomposition, or conserve explicitly by splitting the endpoint masses. Add a unit test that total `Σ m` is invariant across repeated `edge_split_2d` calls on a static mesh.
- Adaptive-driver `h_local` global threshold — refactor to per-vertex/per-edge local length scale (e.g. local `h_i = mean(edge lengths at v_i)`), so a fine droplet region and coarse outer region don't cross-contaminate `L_min`/`L_max`.
Unit tests live in the hyperct repo's `tests/` dir; add one dedicated to the mass-invariance and one to the mixed-scale mesh. **Gate**: full ddgclib `pytest -m "not slow"` passes after updating the hyperct symlink.

**Gate**: Each probe produces a one-page report (pass/fail, magnitudes, short diagnosis). Aggregate into a single probe-results table before Phase B.

**Current Phase A status (2026-04-29, post-Phase-2c)**:

| probe | status | result |
|-------|--------|--------|
| A.1 diagnostics | ✅ done & re-validated | 18/18 pass post-refactor |
| A.2 flat-interface | ✅ done & re-validated | machine precision both dims, both γ variants (2D 2.22e-16; 3D ~1.05e-15); D4 flat-cancellation ruled out |
| A.3 frozen-mesh + skip_triangulation | ⬜ pending | — |
| A.4 upstream hyperct | 🟨 partially superseded | mass-averaging + h_local still pending; simplex-container refactor independently fixed the 3D linear-precision concern |
| A.5.2D static-droplet bisection | ✅ done & confirmed | A.5.b `'exact'` consistent matches A.5.a exactly (2.37e-3 == 2.37e-3) — 2D residual is 100% curvature stencil, 0% retopology |
| A.5.3D Phase 1 — `split_method='exact'` | ✅ done (2026-04-29) | NEGATIVE: `'exact'` consistent end-to-end gives **1.75e-3** (vs `neighbour_count`'s 1.44e-3) — slightly worse, hypothesis killed |
| A.5.3D Phase 2 — single retopo step diff | ✅ done (2026-04-29) | first-to-diverge: `dual_vol_phase` (Δmax 1.03e-8) → `rho_phase` (Δmax 7.10e+02 kg/m³) → `p_phase` (Δmax 3.75e+02 Pa). `m_phase` exactly preserved. 48 cross-phase edges flip per static-cloud retopo step → 3D Delaunay non-uniqueness on near-cospherical interface points. Setup/runtime split-method mismatch identified as a separate bug, fixed in `setup_oscillating_droplet`. |
| A.5.3D Phase 2b — `redistribute_mass=True` | ✅ done (2026-04-29) | NEGATIVE: identical 1.44e-3. Cause: `mass_redistribution.py:250` guards on `p_phase < 1e-30`, which skips the entire outer phase at `P0=0`. Guard fix scoped — see next prompt. |
| A.5.3D Phase 2c — guard fix (`dual_vol_phase`-gated) | ✅ done (2026-04-29) | **POSITIVE**: with guard fix + `redistribute_mass=True`, 3D A.5.b max\|F\| collapses **1.44e-3 → 7.38e-05**, ×195 improvement, ×1.23 over A.5.a's 6.02e-05 floor. Implementation: new `snapshot_geometry_multiphase` capturing pre-retopo `dual_vol_phase`; `redistribute_mass_multiphase` now gates phase-presence on snapshotted `dual_vol_phase[k] > 1e-30` instead of `p_phase < 1e-30` (legacy snapshot still accepted for backwards compat). `_retopologize_multiphase` now uses the geometry-aware snapshot. Bisection harness pre-existing bug (3D `run_a5b` not forwarding `--redistribute-mass`) fixed inline. Production fix shipped: `setup_oscillating_droplet` now defaults `redistribute_mass=True`. Full fast suite 789/0 post-fix. |
| M1 rollout — promote `redistribute_mass=True` in remaining multiphase setups | ✅ done (2026-04-29) | Four setups now accept `redistribute_mass: bool = True` and forward it through their `retopo_fn`: `setup_dam_break_multiphase` (`partial(_retopologize_multiphase, mps=mps, redistribute_mass=…)`), `setup_cube_to_droplet` (closure `setdefault`), `setup_shearing_plate_droplet` (custom `_make_periodic_multiphase_retopo` extended to mirror the snapshot-then-redistribute pattern: `snapshot_geometry_multiphase` → `retopologize_periodic` → `mps.refresh` → `redistribute_mass_multiphase` → `compute_phase_pressures`), `setup_electrolysis_bubble` (added next to `split_method='neighbour_count'`). Per-case smoke (5 symplectic-Euler steps each, 2D, with retopo): per-phase mass drift ≤ 1.94e-15 on dam_break / shearing_plate / electrolysis_bubble; cube_to_droplet phase-1 (droplet) drift = 1.32e-16 (the existing `AtmosphericPressureBC` is a deliberate phase-0 mass source/sink, so phase-0 drift is non-zero by design — present with `redistribute_mass=False` too). Electrolysis-bubble EOS sanity: `eos_gas.density(200)` = 10.02 (rho0=10, K=1e5, n=1, clip (0.5, 2.0)) — well-conditioned at expected gauge pressures. Full fast suite 789/0 post-rollout. |
| Simplex-aware curvature apex enumeration | ✅ done (2026-05-27) | Three batches: PR1 curvature + helper + caller threading; PR2 bubble + interface_subcomplex; PR3 periodic. Added `hyperct.ddg.get_edge_apex_map` and `_edge_to_apex`/`_interface_edge_to_apex` caches; threaded `HC=None` kwarg through `_curvatures_heron.{A_i,hndA_i,int_hndA_i,hndA_i_interface}`, vectorized variants, `surface_tension.*`, `multiphase_stress._interface_surface_tension`; replaced 4 sites in `_bubble.py` via private `_common_neigh` helper; `interface_nn` prefers `HC.interface_edges`; periodic `_fixup_periodic_duals` uses simplex cache for primal triangle apex. Plotting deliberately not migrated (dual-graph, not flag-clique-prone). 11 new regression tests in `test_simplex_aware_curvature.py`. 802 full pass / 275 hyperct pass / 0 regressions / clean static-droplet smoke. Closes the "apex enumeration is contaminated by ghost K_3 cliques" confounder for Tier 2B step 1; the remaining residual is now provably pointwise-stencil truncation. |
| Refactor regression guard | ✅ done | `test_simplex_aware_duals.py` 13/13 pass; `test_simplex_aware_curvature.py` 11/11 pass; full fast suite 789/0 |
| Tier 2B 2D long-run regression | ✅ done (2026-05-28) | A.5.b 2D bit-stable over 2000 steps (std 4.3e-19); `TestStaticDroplet2DRetopologyFloor` pinned at 2.3749e-3 (frozen) / 2.2717e-3 (post-retopo) |
| Probe 6 — A.5.b 3D long-run regression | ✅ done (2026-06-02) | A.5.b 3D bit-stable over 2000 steps (step 2..2000 single unique value, rel spread 0.0, \|dM/M0\|=8.78e-15); `TestStaticDroplet3DRetopologyFloor` pinned at 6.0153e-05 (frozen) / 7.3768e-05 (post-retopo plateau, reached at step 2); volume bit-stable from step 2 (step 0→1 \|dV/V0\|=0.305 is the documented boundary-shell zeroing, not drift). Both static floors now regression-locked. Fast suite 794/0. |

**Recommended next prompt (post-Phase-2c, REVISED 2026-04-29)**:

> The 3D retopology blow-up isolated by A.5.3D is now resolved at the M-lane root layer: with the new `snapshot_geometry_multiphase` + `dual_vol_phase`-gated guard in `redistribute_mass_multiphase`, A.5.b 3D max\|F\| sits at 7.38e-05 vs A.5.a's 6.02e-05 floor (×1.23 over). The remaining ×1.23 gap is plausibly residual 3D Delaunay non-uniqueness churn (~48 cross-phase edges flip per step on a near-cospherical static cloud) but is now small enough to be tackled alongside the broader curvature-stencil rewrite rather than as a fire.
>
> **Next probes, in priority order**:
>
> 1. **M1 broader rollout** (~2–4 hours): Promote `redistribute_mass=True` in remaining multiphase `setup_*` helpers — `setup_dam_break_multiphase`, `setup_shearing_plate_droplet`, `setup_cube_to_droplet`, `setup_electrolysis_bubble`. Each currently builds its `retopo_fn` via `partial(_retopologize_multiphase, mps=mps, ...)` without exposing or setting `redistribute_mass`. Add the parameter (default True) and forward through. Per-case validation: run any existing smoke test or short driver script after each flip; expect mass conservation invariants to remain ≤1e-10 since the redistributor now scales each phase's mass to a per-phase total before applying targets. Watch for any setup whose EOS doesn't have a sensible `.density(p)` at the case's reference pressure (electrolysis bubble is the most likely to need attention).
>
> 2. **Phase 3 (parallel, orthogonal)**: The `_compute_vd_3d` NaN at outer-box boundary (`hyperct/ddg/_geometry.py:91 RuntimeWarning: invalid value encountered in scalar divide`) still surfaces — A.5.a 3D reports `KE = nan mass_total = nan volume_total = nan` at frozen-mesh evaluation, and A.5.b shows `|dM/M0|: nan`. Currently masked from the |F| computation because `_retopologize` overwrites boundary `dual_vol` with 0, but conservation diagnostics are non-functional in 3D until this is fixed. Localise the NaN to the offending tetrahedron via the warning's traceback and fix the divide-by-near-zero (likely a degenerate ghost-tet on the outer box — see `project_3d_boundary_dual_bug.md` memory).
>
> 3. **Tier 2B step 1 (now unblocked)**: The 2D static-droplet residual is provably 100% curvature stencil (per A.5.2D); the 3D residual after the guard fix is dominantly curvature too (the ×1.23 gap above the floor is ≪ the previous ×24). Both dimensions are now ready for the integrated curvature operator rewrite — `surface_tension_force_2d` already exists and is exact for piecewise-linear interfaces; `_curvatures_heron.py` needs the `hndA_i_interface` 3D variant that sums only over interface triangles. Driving 2D residual toward machine precision is the next ladder rung (M3 step 1).
>
> 4. **A.5.b 3D long-run regression** (~30 min): Re-run with the new defaults but `n_steps=2000` to confirm no slow drift. Current 100-step run is flat at 7.38e-05 from step 10 onward, so a long run should stay flat — but a regression guard test would lock this in.
>
> Do NOT touch the BC isolation work (Tier 1E / `OutletDeleteBC`) yet — it remains gated behind the A.3 single-phase probe. Adaptive remesh (`remesh_mode='adaptive'`) also still depends on the upstream hyperct mass-averaging + h_local fixes (A.4 partial).

### Phase B — Parallel single-phase and multiphase lanes, driven by probe results

The user's priorities are **multiphase first** AND **single-phase first** — so the two lanes run in parallel, with shared instrumentation (A.1) and the upstream fix (A.4) already in place.

#### Multiphase lane (lane M)

- **M1.** Promote `redistribute_mass=True` in multiphase case-template `setup_*` helpers (wire in an isothermal EOS). **Not** as a global integrator default (unsafe for pure-CFD cases without an EOS), but on every multiphase setup.
- **M2.** Flip `split_method` default to `'exact'` in [`multiphase.py:417-503`](ddgclib/multiphase.py#L417-L503) if Phase A shows `'exact'` improves the 2A or 2B residual.
- **M3. Tier 2B** (static Young–Laplace droplet) — drive per-vertex interface `|F|` toward machine precision:
  1. Verify `surface_tension_force_2d` ([`curvature_2d.py:91-181`](ddgclib/operators/curvature_2d.py#L91-L181)) is exercised in all 2D multiphase paths.
  2. ~~Add / verify `hndA_i_interface` in [`_curvatures_heron.py`](ddgclib/_curvatures_heron.py) that sums only over interface triangles (3D).~~ **Closed 2026-05-27**: `hndA_i_interface(v, interface_set, HC=HC)` now consults `HC.interface_triangles` via the simplex-aware path when present, so apex enumeration restricts to the interface sub-complex exactly (no ghost K_3 cliques from the bulk). Remaining work in this sub-item is the *integrated* curvature operator (replace pointwise Heron's stencil with a γ-flux form summed around each interface vertex's dual sub-edge); the apex-enumeration prerequisite is now satisfied.
  3. Switch to `remesh_mode='adaptive'` now that A.4 is done — interface preservation during retopology.
- **M4. Tier 3A** (2D oscillating droplet) — target `tail_growth < 1.0`, `l2_error < 0.2`. Only meaningful after M3 passes.
- **M5. Tier 3B/3C** (3D oscillating droplet, dam break) — 3B needs a 3D metric harness first ([`DEVELOPMENT.md:59-68`](DEVELOPMENT.md#L59-L68)); 3C is a regression-only sanity check.

#### Single-phase lane (lane S)

- **S1.** Fix whichever of (time integrator | dual recompute | Delaunay rebuild) the A.3 probes identified. Likely candidates:
  - If 1B.i also fails: dual-volume refresh / mass redistribution — promote `redistribute_mass=True` in setup helpers here too.
  - If 1B.ii fails but 1B.i passes: Delaunay-on-moving-cloud connectivity churn — switch setup helpers to adaptive remesh (now available post-A.4) or investigate partial-retopology strategies.
  - If 1A fails: time-integrator bug — likely a missing symplectic-Euler invariant or an order-of-application issue between BC and stress update.
- **S2. Tier 1C** (transient Poiseuille from rest) — should now pass.
- **S3. Tier 1D** (perturbation hydrostatic damping) — should now pass.
- **S4. Tier 1E** (periodic inlet/outlet vs true periodic) — isolates D2 (BC artefacts). Fix the specific BC that leaks: `OutletDeleteBC` truncated-dual backflow is the most-likely culprit; a principled fix is to use the buffered variant but NOT overwrite velocity inside the buffer, instead letting stress act normally while only marking vertices for deletion at the boundary.

### Why this ordering works for your priorities

You asked for **both** multiphase and single-phase attacked promptly. The probe pass (Phase A) is deliberately cheap (~1 week) and covers **both** lanes simultaneously — 2A probes multiphase, 1A/1B probes single-phase. After Phase A, lanes M and S run in parallel with shared instrumentation, so neither priority delays the other. The upstream hyperct fix (A.4) is done during Phase A so adaptive remesh is available when M3/M4 need it, and is not on the critical path for any Phase A probe.

## Critical files to modify (execution guide)

- **Diagnostics**: [`ddgclib/data/_history.py`](ddgclib/data/_history.py), [`ddgclib/dynamic_integrators/_integrators_dynamic.py`](ddgclib/dynamic_integrators/_integrators_dynamic.py) (hook call sites), new [`ddgclib/data/_conservation.py`](ddgclib/data/_conservation.py).
- **Benchmarks (new)**: `benchmarks/dynamic/` directory with one script per rung (1A, 1B, 1C, 1D, 1E, 2A, 2B, 2C, 3A, 3B, 3C); each as a `pytest` regression hitting the Tier-0 harness.
- **Retopology defaults**: [`ddgclib/dynamic_integrators/_integrators_dynamic.py:230-236`](ddgclib/dynamic_integrators/_integrators_dynamic.py#L230-L236) — flip `redistribute_mass` default to `True` once 1B passes.
- **Multiphase defaults**: [`ddgclib/multiphase.py:417-503`](ddgclib/multiphase.py#L417-L503) — flip `split_method` default to `'exact'` once 2A passes.
- **Upstream bugs (sibling repo)**: [`/home/stefan_endres/projects/hyperct/hyperct/remesh/_operations_2d.py:198`](../hyperct/hyperct/remesh/_operations_2d.py#L198) (mass averaging) and the adaptive driver `h_local` threshold.
- **Interface curvature (Tier 2B)**: [`ddgclib/_curvatures_heron.py`](ddgclib/_curvatures_heron.py) — `hndA_i_interface(v, interface_set, HC=HC)` now consults `HC.interface_triangles` for apex enumeration (post-2026-05-27). The remaining work is the integrated γ-flux rewrite of the pointwise Heron stencil; the apex enumeration is now exact. Helpers `_apex_via_simplex_cache` and `_apex_via_interface_triangles` at the top of the module are the canonical place to consult the simplex / interface cache from other curvature operators.
- **Simplex-aware infrastructure**: [`hyperct/ddg/_retriangulation.py`](../hyperct/hyperct/ddg/_retriangulation.py) — `get_edge_apex_map(HC)` is the public lazy builder; `invalidate_simplex_cache(HC)` clears `_simplices` AND `_edge_to_apex`. Any new code path that mutates topology outside `connect_and_cache_simplices` must call `invalidate_simplex_cache` to keep `_edge_to_apex` consistent. Read-only consumers should `getattr(HC, '_simplices', None)`-check and fall back to legacy `vi.nn ∩ vj.nn` — see the helper template in [`ddgclib/_curvatures_heron.py`](ddgclib/_curvatures_heron.py) (`_apex_via_simplex_cache`).
- **BC fixes (Tier 1E)**: [`ddgclib/_boundary_conditions.py:395-680`](ddgclib/_boundary_conditions.py#L395-L680) — principled outlet/inlet treatment once the isolation benchmark has quantified the artefact.

## Existing utilities to reuse (don't re-write)

- Integrated comparison utilities already exist and should be the only way Tier 0 reports errors: [`integrated_pressure_error`, `integrated_l2_norm`, `compare_stress_force`, `volume_averaged_scalar` in `ddgclib/analytical/_integrated_comparison.py`](ddgclib/analytical/_integrated_comparison.py) (memory: volume-averaged only, never point-wise).
- `StateHistory` already has a callback interface for integrators ([`_history.py:71-115`](ddgclib/data/_history.py#L71-L115)) — extend it, don't replace it.
- The per-phase summed stress operator ([`multiphase_stress_force` in `ddgclib/operators/multiphase_stress.py`](ddgclib/operators/multiphase_stress.py)) is the right target — test 2A exercises it directly, no rewrite needed.
- Exact dual split `split_dual_polygon_2d` / `split_dual_polyhedron_3d` exists in [`_dual_split_2d.py`](ddgclib/geometry/_dual_split_2d.py) — just flip default.
- Integrated 2D surface-tension operator `surface_tension_force_2d` in [`curvature_2d.py:91-181`](ddgclib/operators/curvature_2d.py#L91-L181) is exact for piecewise-linear interfaces — verify it is called from the per-phase stress, do not re-derive.

## Verification

Each rung is a pytest under `benchmarks/dynamic/` or `ddgclib/tests/` and produces:

- A PDF conservation plot (`fig/conservation.pdf`)
- A pass/fail comparison to an analytical reference or a checked-in baseline
- An entry in a top-level dashboard (`benchmarks/dynamic/DASHBOARD.md`) that is re-run on each refactor and shows the status of every rung

The gate for claiming "stable solver" is: **every rung 1A–3C passes its checked-in tolerance, and the dashboard has no regressions.** At that point the library is ready for cases beyond the ladder (capillary rise, contact-line dynamics, Rayleigh–Taylor, etc.).