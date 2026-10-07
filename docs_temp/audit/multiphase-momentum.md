# Audit: multiphase-momentum — momentum conservation / Newton's 3rd law in `multiphase_stress_force`
> Sources checked: docs_temp/02_physics_foundations.md (§2, §3.1, §3.4–3.5, §6), docs_temp/code_map/multiphase_surface_tension.md (§4, §7 flags 5–6), docs_temp/audit/zero-gauge-pressure.md, docs_temp/audit/dual-closure-antisymmetry.md, ddgclib/operators/multiphase_stress.py, ddgclib/geometry/_dual_split_2d.py, ddgclib/multiphase.py, cases_dynamic/oscillating_droplet/src/_setup.py, ddgclib/dynamic_integrators/_integrators_dynamic.py | Written 2026-07-02 by physics-audit workflow

## Item

Suspects (distillation flags 5–6 of `code_map/multiphase_surface_tension.md` §7):

1. **Skip branch** — `ddgclib/operators/multiphase_stress.py:171-177`: phase-k sub-face
   flux skipped when `k not in phases_present(v_i)`; the neighbour may still book its
   side of the same face → pairwise action–reaction broken.
2. **Fallback** — `multiphase_stress.py:181` `p_j_k = _phase_pressure(v_j, k, fallback=p_i_k)`
   with `_phase_pressure` (`:83-91`) treating a stored `p_phase[k] == 0.0` as "missing";
   the two ends of a face can then use different face pressures.

## What the physics requires

`02_physics_foundations.md` §2: dual faces satisfy `A_ji = -A_ij` exactly → "Newton's
3rd law pairwise"; §6 invariant table: momentum antisymmetry `F_p_ij = -F_p_ji` at
atol 1e-12. For internal stresses (pressure + viscous + surface tension on a CLOSED
interface) the exact discrete sum `Σ_i F_i` over all parcels must vanish: the
pressure/viscous flux is pairwise antisymmetric by construction when both endpoints
use the same face state, and the 2D FTC surface tension `γ(t_next − t_prev)`
telescopes to zero around a closed polyline (the 2026-05-27 probe measured 2.7e-19
for the 3D stokes path).

## What the code does

- `multiphase_stress_force` (`multiphase_stress.py:107-193`): per neighbour j,
  `fractions = edge_phase_area_fractions(v, v_j, dim, interface=HC)` (`:167-169`),
  then per phase k: **skip** if `k not in phases_present` (`:171-177`, comment claims
  this "happens only for bulk-bulk cross-phase edges"); own-side pressure read RAW
  (`p_i_by_phase = {k: float(v.p_phase[k])}`, `:141-142`) but neighbour-side pressure
  via `_phase_pressure(v_j, k, fallback=p_i_k)` (`:181`) which substitutes the
  fallback when the stored value is exactly `0.0` (`:88-91`).
- `edge_phase_area_fractions` (`_dual_split_2d.py:545-612`): bulk–bulk → `{v_i.phase: 1.0}`
  (`:575-577`); interface–bulk → `{bulk.phase: 1.0}` (`:579-583`); curve-adjacent
  interface–interface → 50/50 over the UNION of both endpoints' `interface_phases`
  (`:585-593`); interior chord → majority bulk phase of shared 1-ring (`:600-606`),
  fallback split over `v_i.interface_phases` (`:607-612`).

### Where antisymmetry CAN break (exact lines)

| # | Mechanism | Lines | Fires on clean fixtures? |
|---|---|---|---|
| a | zero-sentinel fallback: j stores `p_phase[k]==0.0` (legit gauge zero), i substitutes `p_i_k`; j's own side reads its raw `0.0` → face pressures differ by `p_i_k/2` | `multiphase_stress.py:88-91`, `:181` vs `:141-142` | **YES** — 509–1572 firings per force sweep at the case-default `P0=0` |
| b | skip branch: face flux dropped one-sided | `multiphase_stress.py:171-177` | **NO** — 0 firings in all 9 clean scenarios (2D/3D × static/perturbed × pre/post-retopo); requires inconsistent `interface_phases` tags |
| c | fraction asymmetry: union-pair (`_dual_split_2d.py:586-593`) or chord tie-break (`:600-606`) differing between the (i,j) and (j,i) calls | `_dual_split_2d.py:585-612` | **NO** — scenario C/G2 pairwise defects are exactly 0.0/3.5e-18, so fractions are bitwise symmetric on these meshes |
| d | one-sided fallback for a genuinely absent phase (the documented rationale, module docstring `:14-18,35-46`) | `:83-91` | **NO** — `fallback[missing] = 0` in every scenario; on consistently-tagged meshes the neighbour always stores the requested phase |

Note the comment at `:172-176` misidentifies the skip trigger: for a **bulk** `v_i`,
`edge_phase_area_fractions` always returns `{v_i.phase: 1.0}` (`_dual_split_2d.py:575-583`),
which is always in `phases_present(v_i)` — the skip can NEVER fire on a bulk–bulk
cross-phase edge from either side. It can only fire when `v_i` is an interface vertex
whose `interface_phases` is missing a phase that the fraction rules produce (stale /
corrupted tagging).

## Probe design

Scripts in
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/multiphase-momentum/`
(run with `/home/endres/anaconda3/envs/ddg/bin/python`, cwd = repo root):

- `instrument.py` — monkeypatches the SHIPPED `ddgclib.operators.multiphase_stress`
  module (NOT a replica): wraps `edge_phase_area_fractions` (records current edge +
  counts `k ∉ phases_present` skips against an independent `_phases_present` call),
  `pressure_flux`/`viscous_flux` (per-directed-edge flux ledger), `_phase_pressure`
  (classifies each fallback firing as `missing` vs `zero-sentinel`), and
  `_interface_surface_tension` (per-vertex ST ledger). Then runs
  `multiphase_stress_force` on EVERY vertex and reports: per-unordered-edge pairwise
  defect `|f_ij + f_ji|`, `|Σ F|` over all and over free (non-`bV`) vertices, ratios
  to `max|F|` and `Σ|F|`, and `|Σ F_st|` over the closed interface.
- `probe1_droplet_2d.py` — 2D oscillating-droplet fixture
  (`setup_oscillating_droplet`, refinement 3/3, 311 vertices / 32 interface, the
  A.5.b regression mesh). Scenarios: **A** static circle ε=0 P0=0 as-built; **A′**
  static, duals recomputed (`compute_vd` + `cache_dual_volumes` + `mps.refresh(reset_mass=False)`);
  **B** perturbed ε=0.05 mode-2, P0=0, duals recomputed (real Young–Laplace +
  perturbation pressures); **C** = B at P0=1000 Pa (gauge shift: no exact-0.0
  stored pressures); **D** = C with ONE interface vertex's `interface_phases`
  corrupted `{0,1}→{1}` to force the skip branch.
- `probe2_droplet_3d.py` — 3D fixture (refinement 2/2, 472 vertices / 98 interface):
  **E** static as-built; **E′** static after one production
  `_retopologize_multiphase` step (Delaunay + `batch_e_star` edge cache +
  `redistribute_mass=True`); **F/G** perturbed ε=0.05 post-retopo at P0=0/1000.
- `probe3_droplet_3d_nonuniform.py` — **F2/G2**: same as F/G but
  `redistribute_mass=False` (the `run_a5b` default), which leaves a spatially
  varying per-phase pressure field (F/G came out per-phase-uniform, hiding the
  interior mechanism).

## Probe OUTPUT (key numbers)

2D (`probe1_droplet_2d.py`; 311 vertices, 899 edges, 32 interface):

| scenario | fallback zero/missing | skip | max pairwise defect (edge scale) | \|ΣF\| free | /max\|F\| | /Σ\|F\| |
|---|---|---|---|---|---|---|
| A static, P0=0, as-built | 740 / 0 | 0 | 7.2e-16 (6.96e-3) | 5.6e-15 | 2.4e-12 | 2.2e-13 |
| A′ static, duals recomputed | 522 / 0 | 0 | 1.3e-9 | 5.3e-9 | 2.2e-6 | 2.0e-7 |
| **B perturbed ε=0.05, P0=0** | 509 / 0 | 0 | **2.03 N** (7.55 N) | **9.09 N** | **7.9e-1** | **2.9e-2** |
| C perturbed, P0=1000 | 0 / 0 | 0 | **exactly 0.0** | 9.1e-14 | 7.9e-15 | 3.0e-16 |
| D = C + 1 corrupted tag | 0 / 0 | **3** | 9.5e-1 N | 9.6e-1 N | 8.4e-2 | 3.1e-3 |

- B worst edges are ALL bulk phase-0 pairs with one end stored `p_phase[0] = 0.0`
  bitwise (quiescent far field at gauge zero) and the other at −2547…+1908 Pa;
  `fallback_on_edge=True` on every one. 27/899 edges violate.
- C proves the per-phase fraction machinery + `dual_area_vector` + viscous flux are
  **exactly** pairwise-conservative (defect 0.0 bitwise) once no fallback fires.
- D: 3 skip events at the corrupted vertex → the phase-0 sub-face flux on its
  interface–bulk edge is dropped one-sided (one-sided edge count 1, defect 0.947 N
  = the neighbour's entire unmatched face force); ΣF jumps 9.1e-14 → 0.96 N.
- ST closed-loop sum (2D FTC path): |Σ F_st| = 1.9e-18–3.6e-18 in every scenario.

3D (`probe2/probe3`; 472 vertices, ~3000 edges, 98 interface):

| scenario | fallback zero | skip | max defect | \|ΣF\| free | /max\|F\| free |
|---|---|---|---|---|---|
| E static, P0=0, as-built | 3281 | 0 | 2.6e-17 | 2.8e-16 | 4.7e-12 |
| E′ static, post-retopo (redistrib=True) | 969 | 0 | 1.3e-6 | 1.8e-16 | 2.4e-12 |
| **F2 perturbed, P0=0, post-retopo (redistrib=False)** | 1572 | 0 | **2.5e-2 N** (1.3e-1) | **1.61e-2 N** | **6.9e-2** (1.4e-3 of Σ\|F\|) |
| G2 perturbed, P0=1000, same state | 969 | 0 | 3.5e-18 (free–free) | 1.1e-14 | 4.7e-14 |

- F2: 133 free–free edges violate; worst edges again `p_phase[0]=0.0` bitwise
  (39/285 present phase-0 entries exact zeros post-retopo) against ±2787 Pa
  neighbours. |Σ pairwise defect over free–free edges| = 1.61e-2 N accounts for the
  entire free-vertex momentum sum.
- G2: the 341 remaining defect edges (max 0.215 N) all touch FROZEN wall vertices
  (`dual_vol=0` post-retopo ⇒ stored `p_phase=[0,0]` ⇒ sentinel) — no momentum
  injected into the integrated system (free–free defects ≤ 3.5e-18).
- ST sum (3D `'integrated'` cotan-Heron path, closed sphere): 5.7e-19–9.2e-19 —
  consistent with the 2.7e-19 stokes-path figure.

## Verdict: CONFIRMED_BUG (fallback), latent hazard only (skip)

1. **Fallback (`multiphase_stress.py:88-91` + `:181`) — CONFIRMED momentum bug in the
   shipped operator.** On the default oscillating-droplet configuration (`P0=0`,
   `_setup.py:39`) with any non-trivial pressure state, quiescent vertices store
   `p_phase[k] = 0.0` bitwise; the vertex on the other end of the face substitutes
   its own pressure while the zero vertex uses its raw 0.0 → the two ends book
   different face pressures → `F_ij ≠ -F_ji`. Net momentum source: 9.09 N = 79 % of
   max|F| (2D perturbed ε=0.05 mode-2), 1.6e-2 N = 6.9 % of max|F| (3D perturbed
   post-retopo). Gauge-shifting to P0=1000 (physically identical state) restores
   exact conservation (defect bitwise 0.0 in 2D, ≤3.5e-18 in 3D) — i.e. the violation
   is purely the `val == 0.0` sentinel, confirming and extending the
   `zero-gauge-pressure` audit **in the shipped code path** (that audit's probe 3c
   replicated the loop; here the real function is instrumented) and on the
   asymmetric perturbed interface, 2D and 3D.
2. **Skip branch (`:171-177`) — latent, never exercised.** 0 firings in all 9 clean
   scenarios (instrumented counter), including after a real 3D Delaunay retopo step
   with cross-phase churn: on consistently-tagged meshes every phase produced by
   `edge_phase_area_fractions(v_i, ·)` is provably in `phases_present(v_i)`
   (bulk v_i always gets its own phase; interface v_i gets phases of incident
   simplices, which define `interface_phases`). When forced (scenario D, one stale
   tag) it drops an entire face flux one-sided — defect 0.95 N on a single edge —
   so it IS an antisymmetry breaker, but only downstream of a tag-consistency bug
   elsewhere (e.g. the swallowed closure validation, multiphase.py:379-382, flag 19).
   Its comment ("bulk-bulk cross-phase edges") misdescribes the only trigger.
3. **Everything else is exactly conservative**: per-edge fractions are call-order
   symmetric on real meshes, `A_ji = -A_ij` holds (corroborates the
   dual-closure-antisymmetry audit), viscous flux antisymmetric, and the surface
   tension on the closed interface sums to ≤ 9.2e-19 N (2D FTC and 3D
   integrated/Heron paths) — machine zero.

## Droplet / bubble impact

Direct and first-order for the shipped default. The oscillating-droplet case runs
both TaitMurnaghan phases at `P0 = 0` (`cases_dynamic/oscillating_droplet/src/_setup.py:39,116-119`),
so every run enters the F2/B regime as soon as pressures deviate from equilibrium:
a spurious net momentum source of order several % of the driving force appears at
the quiescent/active pressure front (79 % of max|F| on the ε=0.05 mode-2 fixture).
Oscillation frequency/damping measurements at P0=0 are contaminated; identical runs
at P0 ≫ amplitude are exactly conservative — a silent P0-dependence that physics
forbids. The A.5 static regressions (2.37e-3 / 7.38e-5 floors) are NOT affected
(static state: fallback value coincides with the true 0 pressure; probe scenarios
A/E conserve to ~1e-15). The skip branch does not affect current droplet/bubble
runs unless interface tagging goes inconsistent (non-manifold interface after
retopo — which `identify_interface_from_subcomplex` currently swallows silently).

## Suggested fix

1. Key phase presence on geometry/mass, not the pressure value — replace
   `_phase_pressure`'s `val == 0.0` test (`multiphase_stress.py:88-91`) with
   `v.dual_vol_phase[k] > 1e-30` (and/or store `NaN` for absent phases in
   `compute_phase_pressures`, `multiphase.py:547-549`). Verified by the
   zero-gauge-pressure audit's fixed-variant probes and by scenario C/G2 here
   (removing sentinel firings restores exact conservation).
2. For a genuinely absent phase at the neighbour, make the treatment symmetric:
   either drop the sub-face from BOTH sides or use one-sided extrapolation on both
   — currently i books `-p_i A_k` while j skips or books a different value.
3. Make the skip branch (`:171-177`) loud: increment a diagnostic counter / warn —
   it only fires when interface tags are inconsistent, which is itself a bug
   (flag 19); fix its comment (bulk–bulk cross-phase edges cannot trigger it).
4. Add a regression: instrumented pairwise-antisymmetry sweep (max `|f_ij + f_ji|`
   < 1e-12·scale over all free–free edges) on the perturbed ε=0.05 droplet at
   P0=0, plus gauge-offset invariance of `Σ F`.

## Skeptic review

Adversarial re-review (2026-07-02, independent skeptic pass). Default position was
that the claim is wrong; it survived every refutation attempt. **Verdict upheld:
CONFIRMED_BUG (fallback), severity high; skip branch latent-only as stated.**

Refutation angles attempted and their outcomes:

1. **"The probe manufactured the state" — refuted.** The auditor's scratchpad
   contains a probe NOT cited in this doc, `probe3_dynamic_run.py`, which runs the
   SHIPPED production pipeline (`symplectic_euler` + `_retopologize_multiphase`,
   `setup_oscillating_droplet` at case defaults, CFL-limited dt) with only a
   counting wrapper on `_phase_pressure`. Re-run result: during the first ~9 steps
   of a bona fide dynamic run, `|Σ F|` over all vertices reaches **13.5 N with
   ratio |ΣF|/max|F| up to 0.86**, with 11–65 zero-sentinel substitutions per force
   sweep and substituted pressures up to 9.38e3 Pa. Once the exact-0.0 population
   at the pressure front is consumed (step 9), conservation returns to ~1e-14
   (with recurring ~0.85–0.96 N spikes at steps 15/18, n_sub=0 — likely retopo /
   wall-vertex related, worth a follow-up but not needed for this verdict). The
   defect therefore fires in real production time-stepping, during exactly the
   transient that determines the case's frequency/damping fits — the strongest
   single piece of evidence, and it should have been in the main doc.
2. **"P0=0 is not the shipped configuration" — refuted.** `setup_oscillating_droplet`
   defaults `P0: float = 0.0` (signature, `_setup.py`), and grep finds no `P0`
   override in `oscillating_droplet_2D.py`, `oscillating_droplet_3D.py`, or
   `src/_params.py` — the drivers run at the gauge-zero default.
3. **"The fallback is intentional design" — refuted.** The module docstring
   (`multiphase_stress.py:13-18, 35-46`) documents the fallback exclusively for a
   neighbour that *does not store* a phase-k pressure (bulk in a different phase).
   On every violating edge both endpoints are bulk **same-phase** vertices that DO
   store phase-k pressure; `compute_phase_pressures` (`multiphase.py:543-549`)
   wrote a legitimate EOS value that happens to be bitwise 0.0 at gauge zero. The
   sentinel misfires against the code's own documented intent — not a design choice
   for this path. Hand-check of the arithmetic confirms the mechanism: i books
   `-p·A_ij` (fallback promotes the face to pressure p), j books `+0.5·p·A_ij`
   (raw 0.0 own side, true p neighbour side); net `-0.5·p·A_ij` per edge.
4. **"Some other mechanism (fractions / A_ij / viscous) is responsible" — refuted.**
   Reproduced scenario B/C: at P0=0 the max pairwise defect is 2.03 N (27/899
   edges, every one with a bitwise-0.0 end and `fallback_on_edge=True`); at
   P0=1000 — a physically identical state, max|F| = 11.47 N in both — the defect
   is **bitwise 0.0** and |ΣF|/max|F| = 7.9e-15. Gauge dependence of a momentum
   sum is impossible in exact arithmetic for this discretisation; the isolation to
   the `val == 0.0` test (`multiphase_stress.py:89`) is airtight.
5. **Skip-branch claims — verified as stated.** `edge_phase_area_fractions`
   (`_dual_split_2d.py:575-583`) provably returns `{v_i.phase: 1.0}` for bulk
   `v_i` and only phases from `interface_phases`-consistent rules otherwise, so
   the comment at `:172-176` ("bulk-bulk cross-phase edges") cannot be the
   trigger; 0 firings in clean scenarios reproduced; forced corruption (scenario D)
   reproduced the one-sided 0.947 N drop. Latent hazard, correctly not counted as
   a shipped-run bug. (Side observation: on a genuine bulk–bulk *cross-phase* edge
   — a mesh artefact — the two sides would book different phases with different
   pressures via the missing-phase fallback, another latent asymmetry in the same
   family; none exist on the clean fixtures, `fallback[missing]=0` everywhere.)

Severity kept at **high**: the violation is silent, hits the shipped default
configuration of the flagship multiphase case, injects percent-level (transiently
O(1)-of-max|F|) spurious momentum precisely during the amplitude-establishing
transient, makes results P0-gauge-dependent (physics forbids), and recurs whenever
fresh bitwise-zero pressures appear (large quiescent domains, post-retopo
`dual_vol=0` vertices). Mitigating factors (static A.5 floors unaffected; one-line
workaround `P0≠0`) do not offset a conservation violation in a library whose
purpose is validation-quality conservation. The suggested fix (key presence on
`dual_vol_phase[k]`, not the pressure value) is correct and minimal.
