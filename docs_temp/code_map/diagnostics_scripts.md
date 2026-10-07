# Code Map: Oscillating-Droplet Diagnostic Scripts (Forensic Instruments)

> Sources checked | Written 2026-07-02 by physics-audit workflow
>
> Covers all `diagnose_*.py` scripts in `cases_dynamic/oscillating_droplet/`,
> plus `mesh_convergence_2D.py`, `INTERFACE_STRESS.md`, and
> `INVESTIGATION_PROMPT.md`. Baseline numbers are cross-referenced against
> `debugging_plan.md` (status log 2026-04-24 → 2026-06-02) and against fresh
> runs executed 2026-07-02 (env `ddg`, cwd = repo root). All scripts must be
> run as `python cases_dynamic/oscillating_droplet/<script>.py` from the repo
> root with `/home/endres/anaconda3/envs/ddg/bin/python` (they `sys.path.insert`
> the repo root themselves, but the hyperct symlink must be importable).

## The shared fixture

Every script builds the same static (epsilon=0) two-phase droplet via
`setup_oscillating_droplet` (`cases_dynamic/oscillating_droplet/src/_setup.py:24`)
with parameters from `src/_params.py`: R0=0.01 m, gamma=0.05 N/m, rho_d=800,
rho_o=1000, mu_d=0.5, mu_o=0.1, K from c_s floor 1 m/s, L_domain=5*R0,
`n_refine_outer=3, n_refine_droplet=3`. The 2D mesh is 311 vertices with a
32-vertex interface polyline; the standard 3D mesh (refine 2/2) is 472
vertices with a 98-vertex interface. Analytical Young–Laplace jump:
2D ΔP = γ/R = 5.0 Pa, 3D ΔP = 2γ/R = 10.0 Pa.

Key pinned baselines (regression-locked in
`ddgclib/tests/test_case_oscillating_droplet.py`, see `debugging_plan.md`
2026-05-28 / 2026-06-02 entries):

| quantity | 2D | 3D |
|---|---|---|
| A.5.a frozen-mesh interface max\|F\| | 2.3748568e-03 | 6.0153e-05 |
| A.5.b post-retopo plateau max\|F\| | 2.2716938e-03 (step 1+) | 7.3768e-05 (step 2+) |
| full-dynamic baseline (pre-A.5) | 3.8e-3 | 8.5e-5 |
| bulk-vertex max\|F\| | ~1e-16 (machine) | ~1e-15 (machine) |

The 2D residual is **provably 100% curvature-stencil O(h) truncation** of the
polygonal interface (32→2.37e-3, 64→1.11e-3, 128→5.4e-4); the 3D ×1.23 gap
between A.5.a and A.5.b is irreducible Delaunay non-uniqueness churn on the
near-cospherical interface cloud. These floors are the reference points every
script below is measured against.

---

## 1. `diagnose_static.py` — the first-look force-balance panel

- **File**: `cases_dynamic/oscillating_droplet/diagnose_static.py` (340 lines)
- **CLI flags**: none (edit constants in `main()`; `epsilon=0.0` hardwired).
- **What it measures**: seven numbered sections on the 2D static droplet at
  t=0, no time integration:
  1. Mesh conformity — interface vertex radii vs R0.
  2. IC consistency — bulk p_phase means/stds, Laplace jump vs γκ, sample
     interface-vertex per-phase state (p_phase, dual_vol_phase, m_phase).
  3. Per-component force decomposition (F_pressure, F_viscous, F_st) on
     interface + bulk samples, via an inline re-implementation of the
     `multiphase_stress_force` loop (`pressure_flux`/`viscous_flux` per
     phase-fraction of `dual_area_vector`).
  4. Surface-tension direction census (inward/outward/zero counts).
  5. `edge_phase_area_fractions` samples on interface edges.
  6. Predicted KE after one symplectic step (CFL dt from acoustic+capillary).
  7. Interface closure — each interface vertex has exactly 2 interface edges.
- **Baseline output (fresh run 2026-07-02, matches pinned floors)**:
  - conformity: max radius deviation 1.73e-18, all on circle.
  - Laplace jump 5.000000, error vs expected 2.49e-14; bulk stds ~9e-14.
  - Interface max|F| = 2.374857e-03 (mean 8.148e-04, std 7.18e-04);
    bulk droplet max|F| = 9.78e-17; bulk outer 1.03e-15; F_visc exactly 0.
  - Surface tension: 32/32 INWARD, 0 outward.
  - Edge fractions: iface–iface edges {0: 0.5, 1: 0.5}, bulk edges pure.
  - Predicted 1-step KE = 3.688e-11 (this is the KE plateau seen in A.5.b);
    max |a| = 1.675 at an interface vertex; dt = 6.22e-05.
  - Closure: 32 edges / 32 vertices, 2 iface edges per vertex.
- **Runtime**: ~1–2 s.
- **Reach for it when**: anything changed in the multiphase stress pipeline,
  EOS, IC, dual split, or interface tagging and you want a one-shot "is the
  static equilibrium still healthy" panel. It is the broadest single-shot
  instrument; the debugging plan used it as the post-refactor smoke test
  (2026-05-27: "clean, all 7 diagnostic sections ran"). Red flags: any
  OUTWARD surface-tension vertex, non-zero F_visc at u=0, bulk |F| above
  ~1e-14, Laplace-jump error above ~1e-12, or interface max|F| drifting off
  the 2.3749e-3 floor by more than ~1%.

## 2. `diagnose_a5_bisection.py` — THE canonical curvature-vs-retopology bisection harness

- **File**: `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` (516 lines)
- **CLI flags** (the richest of all the scripts):
  - `--skip-3d` — 2D only (fast).
  - `--n-steps N` — A.5.b step count (default 100; long-runs used 2000).
  - `--refine-3d-outer / --refine-3d-droplet` (default 2/2).
  - `--split-method {neighbour_count,exact}` — dual-volume split policy,
    applied end-to-end (setup + runtime; the 2026-04-29 entry documents that
    a setup/runtime mismatch is itself a bug that produces catastrophic
    3.5e-1 / 6.7e-3 artefacts).
  - `--redistribute-mass` — **IMPORTANT GOTCHA**: `action='store_true'` with
    default False, and the harness *forwards its value into
    `setup_oscillating_droplet`, overriding the production default
    `redistribute_mass=True`*. Omitting the flag therefore exercises the
    LEGACY path and yields 3D A.5.b = 1.44e-3; passing it yields the
    production 7.3768e-05 (A/B documented 2026-05-27). Always pass
    `--redistribute-mass` when you want production behaviour.
  - `--curvature-path {integrated,csf_dual,stokes}` — surface-tension stencil
    used in the |F| evaluation (all three verified bit-identical in 3D,
    Probe 2, 2026-05-27).
  - `--displacement-eps EPS` — Probe 5 skip-retopology gate; with u=0 forced,
    any small eps (e.g. 1e-10) collapses A.5.b to A.5.a.
  - `--results-suffix STR` — suffix for the JSON dump.
- **What it measures**: two flavours per dimension:
  - **A.5.a** — frozen mesh, retopology disabled: evaluate
    `multiphase_stress_force` once on every interface vertex → pure
    curvature-stencil residual. Also prints KE/mass/volume via
    `compute_conservation`.
  - **A.5.b** — forward `euler` with retopology ON and `v.u=0` forced in the
    callback every step (position update reads OLD velocity, so geometry is
    exactly frozen while retopology fires every step) → pure
    retopology-induced residual. Records per-step max|F|, mean|F|, KE, mass,
    volume, vertex/interface counts.
  - `attribute()` classifies the run: curvature-dominant /
    retopology-dominant / mixed / neither.
  - Dumps everything to `results_a5_bisection/a5_bisection<suffix>.json`.
- **Baseline outputs** (debugging_plan.md logged runs):
  - 2D: A.5.a = 2.3748568012e-03; A.5.b step-0 bit-identical, step 1..2000 a
    single unique value 2.2716937802e-03 (std 4.34e-19); KE plateau
    3.688e-11; |dM/M0| = 1.64e-15, |dV/V0| = 3.54e-16; 311→311 vertices,
    32→32 interface.
  - 3D (refine 2/2, `--redistribute-mass`): A.5.a = 6.0153e-05; A.5.b step 1
    = 7.0325e-05, step 2..2000 bit-identical 7.3768e-05; KE plateau
    2.371e-10; |dM/M0| = 8.78e-15; one-shot |dV/V0| = 0.305 at step 0→1
    (boundary dual-shell zeroing artefact, NOT a drift — volume bit-stable
    from step 2); 472→472 vertices, 98→98 interface.
  - 3D WITHOUT `--redistribute-mass` (legacy guard): 1.44e-3 (×17 baseline).
  - 3D `--split-method exact`: 1.75e-3 (slightly WORSE than neighbour_count —
    the exact-split hypothesis was killed 2026-04-29).
- **Runtime**: 2D 100 steps ~ seconds–tens of seconds; 2D 2000 steps 125.7 s;
  3D 2000 steps 686.9 s; a default 2D+3D 100-step run is a few minutes.
- **Reach for it when**: any change touches retopology
  (`_retopologize_multiphase`), the dual split, mass redistribution,
  curvature stencils, or hyperct Delaunay/dual code. This is the harness
  whose floors are pinned by `TestStaticDroplet2DRetopologyFloor` /
  `TestStaticDroplet3DRetopologyFloor`; if those tests fail, re-run this
  script with matching flags to see *which* step and which flavour moved.

## 3. `diagnose_a5_step1_diff.py` — single-retopo-call state diff (Phase 2 instrument)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py` (332 lines)
- **CLI flags**: `--dim {2,3}` (default 3), `--split-method
  {neighbour_count,exact}`, `--refine-outer` / `--refine-droplet` (default
  2/2), `--top-n` (default 10).
- **What it measures**: snapshots every vertex's multiphase state
  (`is_interface`, `phase`, `dual_vol`, `dual_vol_phase`, `m_phase`,
  `p_phase`, `rho_phase`, 1-ring `nn` set, stress force F), fires exactly ONE
  `retopo_fn(HC, bV, dim)` call with zero vertex motion, snapshots again, and
  ranks per-interface-vertex deltas for each quantity. Answers "which
  per-vertex per-phase quantity diverges FIRST across a retopo call". Also
  reports 1-ring connectivity churn (edges added/removed) and dumps JSON to
  `results_a5_bisection/a5_step1_diff_<dim>d_<split>.json`.
- **Baseline output** (2026-04-29 Phase 2, 3D neighbour_count, legacy path):
  vertex set unchanged 472→472, no phase/is_interface flips; Δm_phase = 0.0
  exactly (Lagrangian mass preserved); Δdual_vol_phase max 1.03e-8 (tiny) →
  Δrho_phase max 7.10e+02 kg/m3 → Δp_phase max 3.75e+02 Pa → per-vertex |F|
  jumps ×21–×60; 48 cross-phase edges added + 48 removed (3D Delaunay
  non-uniqueness smoking gun). In 2D the same experiment shows essentially
  zero churn (Delaunay of a static generic 2D cloud is unique).
  Post-guard-fix (redistribute_mass geometry-gated), the p_phase/rho_phase
  swings collapse.
- **Runtime**: tens of seconds (two full force sweeps + one retopo; 3D
  refine 2/2).
- **Reach for it when**: `diagnose_a5_bisection.py` shows a step-1 (or
  step-2) jump in A.5.b and you need to localise the mechanism to a
  quantity/ordering: mass bookkeeping vs volume split vs EOS vs connectivity
  churn. This is the "microscope" that produced the
  dual_vol_phase→rho→p causal chain and the redistribute_mass guard fix.

## 4. `diagnose_a5_dissect_2d.py` — single-vertex F_p vs F_st anatomy (2D)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_a5_dissect_2d.py` (121 lines)
- **CLI flags**: none.
- **What it measures**: for the first 3 interface vertices in polar order:
  interface 1-ring identification (`_interface_neighbours`,
  `_select_curve_neighbours` prev/next), 1-ring composition
  (iface/inner-bulk/outer-bulk counts), per-phase state, then the exact
  decomposition F_p = F_total − F_st, the element-wise ratio −F_p/F_st
  (should be 1 at discrete equilibrium), the inner-phase dual area vector
  S_inner = Σ frac_1 · A_ij, the prediction F_p = −ΔP·S_inner, and the
  required-vs-actual surface tension ratio |F_st| / |ΔP·S_inner|.
- **Expected output**: on the 32-gon, F_p matches −ΔP·S_inner to machine
  precision (bulk pressure is exactly uniform), so the entire residual is
  the mismatch ratio |F_st|/|F_st_required| ≠ 1 — the discrete curvature
  γ(t_next − t_prev) of the polygon vs the dual-face-consistent ΔP·S_inner.
  Per `diagnose_discrete_dp` numbers this ratio varies vertex-by-vertex
  (effective ΔP 4.42–7.52 across the ring; see below).
- **Runtime**: ~1–2 s.
- **Reach for it when**: working on the Tier 2B step-1 2D curvature rewrite
  (integrated γ-flux form) and you need to see, per vertex, exactly how far
  the FTC tangent-difference F_st is from the value that would cancel the
  pressure flux on the SAME dual faces. It is the smallest-granularity 2D
  instrument.

## 5. `diagnose_discrete_dp.py` — discrete-consistent ΔP vs analytical γκ

- **File**: `cases_dynamic/oscillating_droplet/diagnose_discrete_dp.py` (132 lines)
- **CLI flags**: none.
- **What it measures**: builds the case WITHOUT Young–Laplace preloading,
  then for every interface vertex computes the per-vertex effective pressure
  jump ΔP_eff = (F_st·S_1)/|S_1|^2 that would zero the normal force, plus
  the global least-squares optimal ΔP = Σ(F_st·S_1)/Σ|S_1|^2. Then it
  re-loads the droplet mass at the analytical ΔP and at the optimal ΔP and
  compares interface force residuals.
- **Baseline output (fresh run 2026-07-02)**: analytical ΔP = 5.0; optimal
  ΔP = 4.891952 (ratio 0.9784); per-vertex ΔP_eff mean 5.17, std 0.91, range
  [4.4174, 7.5189] — the 45-degree "corner" vertices of the projected-square
  ring need ΔP_eff = 7.52 while the axis vertices need 4.42. Residuals:
  analytical ΔP → max|F| 2.374856e-03; optimal ΔP → max|F| 2.476725e-03
  ("Improvement: 1.0x"). **Key negative result**: no single scalar ΔP can
  cancel the per-vertex mismatch — the residual is angular (stencil-shape)
  error, not a wrong jump magnitude. This justified abandoning
  "calibrate ΔP" approaches in favour of the curvature-stencil rewrite.
- **Runtime**: ~1–2 s.
- **Reach for it when**: someone proposes fixing the static residual by
  tweaking the Young–Laplace pre-load (mass/pressure IC). This script is the
  evidence that IC tuning cannot help; re-run it after any curvature-stencil
  change to see whether the per-vertex ΔP_eff spread (std 0.91 today)
  tightens toward zero — that spread is the honest scalar measure of
  curvature/pressure-flux inconsistency.

## 6. `diagnose_balance.py` — the (removed) interface force-balance pre-load, replayed

- **File**: `cases_dynamic/oscillating_droplet/diagnose_balance.py` (130 lines)
- **CLI flags**: none.
- **What it measures**: manually replicates setup steps 1–5 (no balance),
  prints interface forces, then applies an experimental
  "`_balance_interface_forces`" step inline: per interface vertex, compute
  ΔP_eff = −(F_st·S_1)/|S_1|^2 and reset `m_phase[1] = rho(p_outer +
  ΔP_eff)·vol_d` (skipping ΔP_eff < 0), refresh, and print forces after.
- **Status note**: `_balance_interface_forces` NO LONGER EXISTS in
  `src/_setup.py` (verified 2026-07-02: no match in the file) — this script
  is a frozen replay of an experiment that was trialled and rejected (the
  per-vertex mass balance is the same dead end that `diagnose_discrete_dp`
  quantifies: it changes the vertex's own pressure, which leaks into
  neighbours' fluxes, so it cannot converge per-vertex). Historical
  instrument; keep for provenance, do not expect it to match current setup
  behaviour (it also predates the `split_method`/`redistribute_mass` setup
  parameters).
- **Runtime**: ~1–2 s.
- **Reach for it when**: essentially never for new debugging; only to
  understand why per-vertex mass pre-balancing was abandoned. Prefer
  `diagnose_discrete_dp.py`.

## 7. `diagnose_dual_only.py` — dual-only vs full-Delaunay retopo A/B (2D, dynamic)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_dual_only.py` (171 lines)
- **CLI flags**: none (100 steps, symplectic_euler, hardwired).
- **What it measures**: two full dynamic 100-step runs of the static droplet
  with REAL (not zeroed) velocities:
  1. `dual_only_retopo` — recompute barycentric duals + cache volumes +
     `mps.split_dual_volumes` on FIXED connectivity (the Cube2droplet
     `diagnostic_no_retopo` recipe).
  2. The production full-Delaunay `retopo_fn` (with remesh_mode/kwargs from
     setup).
  Records KE / R_max / R_min / interface count every 10 steps and prints a
  Max-KE ratio (Delaunay/dual-only).
- **Expected output**: this was the INVESTIGATION_PROMPT.md item 3
  instrument ("does static_droplet_2D stabilise with dual-only retopo?").
  Both runs start from the same IC; KE stays at the impulse floor (~1e-11
  scale, cf. predicted 1-step KE 3.688e-11) if the interface residual is not
  amplified. Historically (April 2026) the Delaunay run showed KE growth to
  ~4.4e-3 in 100 steps; after the own-phase-pressure and redistribute-mass
  fixes the two paths should be comparable — a large ratio (≫1) flags
  retopology re-injecting energy under real dynamics (this is the piece the
  u=0 A.5.b harness deliberately cannot see).
- **Runtime**: ~1–3 min (two 100-step dynamic runs with retopo).
- **Reach for it when**: A.5.b (u=0) is clean but the real dynamic case
  still gains KE — this script isolates motion×retopology coupling, the
  "~37% of 2D baseline" bucket that the frozen probes do not capture.

## 8. `diagnose_eos_ic.py` — EOS / IC consistency audit (2D + 3D)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_eos_ic.py` (192 lines)
- **CLI flags**: none (`main()` runs dim=2 then dim=3 with
  `neighbour_count`; the `split_method` argument of `verify_eos_ic` exists
  but the 'exact' variant is noted as not wired through setup in this
  script — use `diagnose_split_methods.py` for that comparison).
- **What it measures**: five checks per dimension:
  1. EOS round-trip: `pressure(density(p)) == p` for the droplet EOS at
     p_outer + ΔP.
  2. Bulk droplet: p_phase[1] uniform == p_outer + γκ; max
     |EOS(m/V) − p_stored|.
  3. Bulk outer: p_phase[0] uniform == p_outer.
  4. Interface vertices: both-phase pressures, Laplace jump mean/std,
     phase-1 volume fraction stats, EOS(m/V) vs stored p for both phases.
  5. Force balance: max/mean |F| on interface, bulk-drop, bulk-outer
     (bulk-outer limited to 50 vertices for 3D speed).
- **Expected output**: 2D — jump 5.000000 with std ~1e-13, EOS errors
  ~1e-14, interface max|F| at the 2.37e-3 floor, bulk at machine precision.
  3D (refine 1/1 for speed) — jump 10.0, interface max|F| at the
  corresponding coarse-mesh floor. Any mismatch between stored p_phase and
  EOS(m_phase/dual_vol_phase) indicates a stale-pressure bug (a refresh
  ordering problem), which is exactly what this script exists to catch.
- **Runtime**: ~10–60 s (3D setup dominates).
- **Reach for it when**: after touching `MultiphaseEOS`, `mps.refresh`,
  `compute_phase_masses/pressures`, or the Young–Laplace pre-load in
  `_setup.py` — it verifies the *data-model invariants* (p = EOS(m/V) per
  phase, jump = γκ) independently of the force operators.

## 9. `diagnose_retopo_effect.py` — single Delaunay retopo, field-level diff (2D)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_retopo_effect.py` (221 lines)
- **CLI flags**: none.
- **What it measures**: the 2D-only, field-oriented predecessor of
  `diagnose_a5_step1_diff.py`: snapshot per-vertex state keyed by `v.x`,
  apply ONE production `retopo_fn` with no vertex motion, diff: interface
  vertex/edge sets gained/lost, Δdual_vol, Δdual_vol_phase[k], Δp_phase[k]
  per interface vertex (sample table by angle), interface max|F|
  before/after and ratio, and bulk pressure changes.
- **Baseline output (fresh run 2026-07-02)**: interface vertices 32→32,
  edges 32→32, gained/lost 0/0; max |Δdual_vol| = 9.9e-16, max
  |Δdual_vol_phase| ~ 5.0e-16; max |Δp_phase[0]| = 1.87e-08, |Δp_phase[1]| =
  4.07e-07; interface max|F| 2.374857e-03 → 2.374856e-03 (ratio 1.00);
  bulk Δp ≤ 4.07e-07. I.e. 2D retopology on the static cloud is neutral to
  roundoff — the healthy reference signature.
- **Runtime**: ~1–2 s.
- **Reach for it when**: quick 2D sanity check that a retopology-side change
  (hyperct Delaunay, `_retopologize_multiphase`, split, redistribution) is
  still exactly neutral on a static conforming mesh. If interface edges are
  gained/lost or Δp jumps above ~1e-6, the change broke 2D retopo
  neutrality. For 3D or for ranked per-quantity attribution use
  `diagnose_a5_step1_diff.py` instead.

## 10. `diagnose_split_methods.py` — neighbour_count vs exact split A/B (2D, dynamic)

- **File**: `cases_dynamic/oscillating_droplet/diagnose_split_methods.py` (146 lines)
- **CLI flags**: none (runs both methods, 100 steps each).
- **What it measures**: builds the case manually (bypassing
  `setup_oscillating_droplet`, so it controls `split_method` end-to-end,
  including the Young–Laplace mass pre-load), reports per method: interface
  phase-1 volume-fraction stats, p_phase means/stds, Laplace jump, t=0
  max/mean |F|; then runs 100 symplectic-Euler steps with `dual_only_retopo`
  (fixed connectivity) and reports final/max KE.
- **Expected output**: `neighbour_count` gives interface volume fraction
  exactly 0.5 (by construction of the 1-ring vote on the conforming mesh);
  `exact` gives the geometric fraction. Both should reproduce the 5.0
  Laplace jump; the t=0 max|F| stays at the curvature floor for both. The
  significant historical result is in the *setup consistency*: mixing split
  methods between setup and runtime is catastrophic (2026-04-29:
  2D 3.50e-1), while consistent 'exact' makes 2D retopology exactly neutral
  (A.5.b == A.5.a).
- **Runtime**: ~1–3 min (two 100-step dynamic runs).
- **Reach for it when**: evaluating a change to
  `MultiphaseSystem.split_dual_volumes` / `_dual_split_2d.py`, or deciding
  the `split_method` default (plan item M2). Note the caveat: uses dual-only
  retopo, so it tests the split policy itself, not its interaction with
  Delaunay churn (use `diagnose_a5_bisection.py --split-method exact` for
  that).

## 11. `mesh_convergence_2D.py` — dynamic mesh-independence study (Tier 3A instrument)

- **File**: `cases_dynamic/oscillating_droplet/mesh_convergence_2D.py` (193 lines)
- **CLI flags**: none (refinement levels hardwired: (1,2), (2,3), (3,4)).
- **What it measures**: full dynamic perturbed-droplet (epsilon=0.05) runs at
  three refinements, ≤500 steps each, comparing the R_max(t) envelope to the
  Rayleigh/Lamb analytical `max_radius_envelope`; reports L2/Linf error per
  level, an approximate convergence order from consecutive levels, and a
  mesh-independence metric max|R_coarse − R_fine| on a common time grid.
  Saves `fig/oscillating_droplet_2D_convergence.png` (NOTE: to CWD-relative
  `fig/`, not the case dir — run from repo root and expect the figure at
  `<repo>/fig/`).
- **Expected/baseline output**: no pinned baseline in debugging_plan.md — the
  dynamic oscillating-droplet lane (M4/Tier 3A) is still open with live
  scores `tail_growth = 1.68`, `l2_error ≈ 5.6` (target <1.0 / <0.2), so this
  script currently documents *lack* of convergence rather than a pass. It is
  the acceptance instrument for M4 once the static floors are driven down.
- **Runtime**: several minutes (three dynamic runs, the (3,4) level
  dominates).
- **Reach for it when**: after any fix that plausibly improves the dynamic
  oscillation (curvature rewrite, retopo gating, adaptive remesh) — this is
  the ladder rung that says whether the *dynamic* solution is becoming
  mesh-independent. Do not bother while static max|F| floors are unchanged;
  it cannot pass before Tier 2B does (debugging_plan.md: "3A cannot pass
  without 2B at machine precision").

## 12. `INTERFACE_STRESS.md` — the interface data-model contract

- **File**: `cases_dynamic/oscillating_droplet/INTERFACE_STRESS.md` (156 lines)
- Not a script: the authoritative prose spec of the primal-subcomplex sharp
  interface model that all the diagnostics assume. Key contracts:
  - Vertex roles: bulk droplet (phase 1), bulk gas (phase 0), sharp
    interface (`v.phase = INTERFACE_PHASE = -1`, `v.is_interface = True`,
    `v.interface_phases = {0,1}`).
  - Per-phase arrays `m_phase / p_phase / rho_phase / dual_vol_phase`;
    scalar shortcuts `v.m = sum(m_phase)`, `v.p = p_phase[v.phase]`.
  - Flux formula (all vertices): F_i = Σ_j [−0.5(p_i+p_j)A_ij +
    (mu/|d|)Δu(d̂·A_ij)] with own-phase pressure and own-phase viscosity
    (NO harmonic mean); interface vertices additionally get
    F_st = γκn dA integrated (2D FTC `surface_tension_force_2d`, 3D
    cotangent-Heron `hndA_i_interface`).
  - Per-phase EOS with no blending; Lagrangian m_phase fixed while
    dual sub-volumes and densities move.
  - Initialisation sequence steps 1–7 including the Young–Laplace **mass
    pre-load** (rho_d_eq = eos.density(p_outer + γκ)) — explicitly NOT a
    pressure bump and NOT force double-counting (surface tension still acts
    every step and the compressed-droplet pressure balances it).
  - File-location table mapping each concept to its implementation module.
- **Reach for it when**: onboarding to any interface diagnostic; whenever a
  probe result seems to contradict "what the code should do", this file is
  the intended-behaviour reference (it was corrected once already — the note
  at the bottom records that a previously-documented additive pressure bump
  never existed in code).

## 13. `INVESTIGATION_PROMPT.md` — the original forensic brief (historical)

- **File**: `cases_dynamic/oscillating_droplet/INVESTIGATION_PROMPT.md` (106 lines)
- The April 2026 hand-off prompt that spawned the whole `diagnose_*` family:
  puzzle statement (Cube2droplet no-retopo works, static_droplet_2D with
  Delaunay retopo gains KE 0 → 4.4e-3 in 100 steps at epsilon=0), reading
  list, and 5 numbered investigations — (1) t=0 force balance decomposition
  → became `diagnose_static.py`; (2) IC consistency → `diagnose_eos_ic.py`;
  (3) dual-only vs Delaunay → `diagnose_dual_only.py`; (4) mesh conformity /
  closure → `diagnose_static.py` sections 1 & 7; (5) surface-tension
  direction → `diagnose_static.py` section 4. Expected-outcome list (IC
  imbalance / retopo corruption / sign error / phase-fraction error) is the
  original hypothesis set; the eventual answers are recorded in
  `debugging_plan.md` (curvature-stencil O(h) in 2D; retopo +
  redistribute-mass guard in 3D; sign and fractions were fine).
- **Reach for it when**: writing a new investigation prompt for a fresh
  session — it is the template — or reconstructing why each diagnostic
  exists.

---

## Quick selection table for a solver-debugging workflow

| symptom | first instrument | escalation |
|---|---|---|
| pinned floor test fails / static droplet KE grows | `diagnose_static.py` (~2 s) | `diagnose_a5_bisection.py --skip-3d` |
| A.5.b jumps above A.5.a at step 1–2 | `diagnose_a5_step1_diff.py --dim {2,3}` | `diagnose_retopo_effect.py` (2D quick check) |
| suspect EOS / IC / refresh ordering | `diagnose_eos_ic.py` | `diagnose_split_methods.py` |
| suspect the curvature stencil itself (2D) | `diagnose_a5_dissect_2d.py` | `diagnose_discrete_dp.py` (ΔP_eff spread) |
| clean u=0 probes but dynamic KE growth | `diagnose_dual_only.py` | full case + `mesh_convergence_2D.py` |
| proposal to "fix" via IC/ΔP tuning | `diagnose_discrete_dp.py` (shows it can't work) | — |
| dual-volume split policy change | `diagnose_split_methods.py` | `diagnose_a5_bisection.py --split-method exact` |
| dynamic-lane acceptance (M4) | `mesh_convergence_2D.py` | — |

Reproducibility notes: (a) always run from the repo root with the `ddg` env
python; (b) for `diagnose_a5_bisection.py`, pass `--redistribute-mass` to get
the production path (the store_true default silently reverts to the legacy
guard); (c) JSON artifacts land in
`cases_dynamic/oscillating_droplet/results_a5_bisection/` and the historical
run archive there (14 JSONs, e.g. `a5_bisection_2d_longrun.json`,
`a5_step1_diff_3d_neighbour_count.json`) is itself a baseline library.
