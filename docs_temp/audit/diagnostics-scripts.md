# Audit: diagnostics-scripts (documentation task)

> Sources checked | Written 2026-07-02 by physics-audit workflow

## Task

Documentation item, no bug verdict required: read every `diagnose_*.py`
script in `cases_dynamic/oscillating_droplet/` plus `INTERFACE_STRESS.md`,
`INVESTIGATION_PROMPT.md`, and `mesh_convergence_2D.py`, and produce the
code_map reference `docs_temp/code_map/diagnostics_scripts.md` describing
purpose, CLI flags, measured quantities, expected/baseline outputs, runtime,
and when a debugging workflow should reach for each.

## What was read (all quoted with line evidence in the code_map file)

- `cases_dynamic/oscillating_droplet/diagnose_static.py` (340 lines)
- `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` (516 lines)
- `cases_dynamic/oscillating_droplet/diagnose_a5_step1_diff.py` (332 lines)
- `cases_dynamic/oscillating_droplet/diagnose_a5_dissect_2d.py` (121 lines)
- `cases_dynamic/oscillating_droplet/diagnose_discrete_dp.py` (132 lines)
- `cases_dynamic/oscillating_droplet/diagnose_balance.py` (130 lines)
- `cases_dynamic/oscillating_droplet/diagnose_dual_only.py` (171 lines)
- `cases_dynamic/oscillating_droplet/diagnose_eos_ic.py` (192 lines)
- `cases_dynamic/oscillating_droplet/diagnose_retopo_effect.py` (221 lines)
- `cases_dynamic/oscillating_droplet/diagnose_split_methods.py` (146 lines)
- `cases_dynamic/oscillating_droplet/mesh_convergence_2D.py` (193 lines)
- `cases_dynamic/oscillating_droplet/INTERFACE_STRESS.md` (156 lines)
- `cases_dynamic/oscillating_droplet/INVESTIGATION_PROMPT.md` (106 lines)
- `cases_dynamic/oscillating_droplet/src/_params.py` (fixture parameters)
- `debugging_plan.md` (full 588 lines — the logged-run baseline source)
- Cross-checks: `src/_setup.py` (setup signature: `split_method:str =
  "neighbour_count"`, `redistribute_mass: bool = True` at lines 41–42; NO
  `_balance_interface_forces` remains — `diagnose_balance.py` replays a
  removed experiment), `ddgclib/operators/multiphase_stress.py`
  (`curvature_path` values `integrated|csf_dual|stokes` at lines 107–267).

## Probe design and OUTPUT (fresh verification runs, 2026-07-02)

Three cheap read-only scripts were executed with
`/home/endres/anaconda3/envs/ddg/bin/python` from the repo root (outputs
archived in the session scratchpad `audit/diagnostics-scripts/*.out`):

1. `diagnose_static.py` (wall ~1 s):
   - Interface max|F| = **2.374857e-03** — matches the pinned 2D A.5.a floor
     2.3748568012e-03 (debugging_plan.md 2026-05-28 table) to all printed
     digits.
   - Mesh conformity max deviation 1.73e-18; Laplace jump 5.000000 (error
     2.49e-14); F_visc exactly 0; bulk droplet max|F| 9.78e-17, bulk outer
     1.03e-15; surface tension 32/32 INWARD; predicted 1-step KE 3.688e-11
     (identical to the logged 2D A.5.b KE plateau 3.688e-11); closure 32
     edges / 32 vertices.
2. `diagnose_discrete_dp.py` (wall <1 s):
   - Analytical ΔP = 5.0, least-squares optimal ΔP = 4.891952 (ratio
     0.9784); per-vertex ΔP_eff spread [4.4174, 7.5189], std 0.9129; max|F|
     analytical 2.374856e-03 vs optimal 2.476725e-03 ("Improvement: 1.0x") —
     confirms the documented negative result that no scalar ΔP calibration
     can remove the angular stencil error.
3. `diagnose_retopo_effect.py` (wall ~1 s):
   - Single Delaunay retopo on the static 2D mesh: 32→32 interface
     vertices/edges, 0 gained/lost; max |Δdual_vol| 9.9e-16; max
     |Δp_phase[1]| 4.07e-07; interface max|F| 2.374857e-03 → 2.374856e-03
     (ratio 1.00) — 2D retopology neutrality holds, matching the 2026-05-28
     "bit-stable" long-run claim.

All fresh numbers agree with the debugging_plan.md logged baselines, so the
baselines quoted in the code_map document are current, not stale.

## Notable findings recorded in the code_map (not bugs, but workflow traps)

- **`--redistribute-mass` flag inversion trap** in
  `diagnose_a5_bisection.py:394-399`: the harness default (`store_true` →
  False) OVERRIDES the production setup default (`redistribute_mass=True` in
  `_setup.py:42`). Running without the flag exercises the legacy guard and
  reproduces the historical 3D 1.44e-3 blow-up, not production behaviour.
  Documented (this A/B is itself logged in debugging_plan.md 2026-05-27).
- **`diagnose_balance.py` is a historical replay**: the
  `_balance_interface_forces` step it debugs no longer exists in
  `src/_setup.py` (grep confirms no match); the script predates the
  `split_method`/`redistribute_mass` setup parameters.
- **`mesh_convergence_2D.py` writes its figure to CWD-relative `fig/`**, not
  the case directory (contrary to the case-study output convention in
  CLAUDE.md) — flagged in the doc so the workflow knows where to look.
- `diagnose_eos_ic.py`'s `split_method` parameter is accepted but the
  'exact' variant is not wired through setup within that script (comment at
  lines 179-182); `diagnose_split_methods.py` is the correct instrument for
  that comparison.

## Verdict

CORRECT_AS_INTENDED / severity none — documentation task. The diagnostic
suite is coherent, its logged baselines reproduce exactly on the current
tree, and the code_map file
`docs_temp/code_map/diagnostics_scripts.md` now documents all 13 artifacts
with a symptom→instrument selection table for the solver-debugging workflow.

## Droplet impact

These scripts ARE the oscillating-droplet forensic toolchain; nothing in
this audit changes case behaviour. The reproduced floors (2D interface
max|F| 2.3749e-3; 2D retopo neutrality to 4e-7 in p_phase) confirm the
static-droplet lane is in the state the debugging plan claims, so the
upcoming solver-debugging workflow can trust the pinned baselines.
