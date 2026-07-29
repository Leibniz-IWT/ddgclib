# Audit: zero-gauge-pressure — multiphase code treats pressure exactly 0.0 as "phase missing"
> Sources checked | Written 2026-07-02 by physics-audit workflow

## Item

`_phase_pressure` (`ddgclib/operators/multiphase_stress.py:83-91`) and
`compute_phase_pressures` (`ddgclib/multiphase.py:528-563`) use the stored float
value `0.0` as a sentinel for "phase not present at this vertex". A legitimate
zero **gauge** pressure (P0 = 0, ρ = ρ0) is indistinguishable from the sentinel.
The oscillating-droplet case runs **both** phases at gauge `P0 = 0.0` by default
(`cases_dynamic/oscillating_droplet/src/_setup.py:39,116-119`), so exact zeros
are pervasive.

## What the physics requires

1. **Gauge invariance.** The integrated pressure force on a dual cell is
   `F_p_i = Σ_j -½(p_i + p_j) A_ij` (`stress.py:680-682`, doc
   `docs_temp/02_physics_foundations.md` §3.1). For a closed (interior) dual
   cell `Σ_j A_ij = 0`, so adding a constant C to all phase pressures must leave
   `F` unchanged: `F(p+C) = F(p) - C Σ_j A_ij = F(p)`.
2. **Face-value consistency / Newton's third law.** Both endpoints of an edge
   must use the same face pressure `½(p_i + p_j)` so that pairwise pressure
   fluxes satisfy `F_ij = -F_ji` (momentum conservation of the FVM scheme).

## What the code does

- `multiphase_stress.py:88-91`:
  ```python
  val = float(p_phase[k])
  if val == 0.0:
      return fallback
  ```
  Called at line 181 as `p_j_k = _phase_pressure(v_j, k, fallback=p_i_k)`.
  When neighbour j's phase-k pressure is stored as exactly `0.0` — which is the
  *correct physical value* for an equilibrium-density vertex under
  `TaitMurnaghan(P0=0)` — the code substitutes `p_i_k`. The face pressure
  becomes `½(p_i + p_i) = p_i` instead of `½(p_i + 0)`, i.e. the entire
  pressure-difference flux on that face is silently zeroed **for vertex i
  only** (vertex j, whose own `p_i_k = 0`, still resolves i's nonzero pressure
  correctly and books `½(0 + p_i)` — the two sides of the same face disagree).
- The sentinel is *written* by `compute_phase_pressures`
  (`multiphase.py:547-549`): absent phases get `v.p_phase[k] = 0.0`. So 0.0
  genuinely is overloaded to mean "absent" — the ambiguity is baked into the
  data model, not just the reader.
- Same pattern in the interface `v.p` average (`multiphase.py:555-561`): the
  filter `v.p_phase[k] != 0.0` (line 559) drops a phase whose gauge pressure is
  exactly zero from the mean.
- `_phase_pressure` has exactly one call site (`multiphase_stress.py:181`) and
  no test coverage.

## Probe design

Scripts in
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/zero-gauge-pressure/`,
run with `/home/endres/anaconda3/envs/ddg/bin/python` (`PYTHONPATH` = repo root).

1. **probe1_unit.py** — direct semantics of `_phase_pressure` on a mock vertex
   with phase 0 present (`dual_vol_phase[0] = 1e-6`, `m_phase[0] = 1e-3`) but
   `p_phase[0] = 0.0`.
2. **probe2_gauge.py** — build the real 2D oscillating-droplet fixture twice via
   `setup_oscillating_droplet(dim=2)` (defaults: ε = 0.05, R0 = 0.01,
   γ = 0.05), identical except `P0 = 0` vs `P0 = 1000`. TaitMurnaghan is
   additive in P0 and densities are P0-independent, so
   `p_phase(B) = p_phase(A) + 1000` exactly on present entries (verified
   bitwise). Compare `multiphase_stress_force` per vertex, shipped vs a
   monkeypatched "fixed" `_phase_pressure` that keys presence on
   `dual_vol_phase[k] > 1e-30` instead of `val == 0.0`. Count fallback firings.
3. **probe3_wavefront.py** — same fixture at P0 = 0, then reproduce a
   mid-simulation state: outer-phase mass scaled by +1 % for r < 2.2 R0
   (compression front, p ≈ 258 Pa with the case's softened K_o = 25 000 Pa),
   far field untouched → `p_phase[0] == 0.0` bitwise (as in a real run:
   unmoved vertices keep ρ ≡ ρ0). Recompute pressures via
   `mps.compute_phase_pressures(HC)`, then (a) force error shipped-vs-fixed,
   (b) gauge invariance under +1000 Pa offset, (c) pairwise action–reaction of
   the phase-0 pressure flux over every edge (replicating exactly the loop at
   `multiphase_stress.py:155-185`, incl. `edge_phase_area_fractions` and the
   `phases_present` skip).

## Probe output

**Probe 1** (unit):
```
stored p_phase[0] = 0.0 (phase PRESENT, vol=1e-06)
_phase_pressure(v, 0, fallback=123.456) = 123.456   <-- expected 0.0, got fallback: True
after +1000 gauge shift: _phase_pressure(v, 0, fallback=1123.456) = 1000.0
gauge-shift consistency (should be got0 + 1000 = 1123.456): VIOLATED
face pressure: code = 50.0, physics = 25.0  (error = 25.0 Pa = 100% of the pressure difference)
```

**Probe 2** (real fixture, t = 0):
```
present (vol,m > 1e-30) phase entries: 247
max |p_phase(B) - p_phase(A) - 1000.0| over present entries: 0.000e+00
present entries stored as EXACTLY 0.0 in run A (P0=0): 72
_phase_pressure ==0.0 fallback fired: run A: 349/1318 calls, run B: 0/1318 calls
interior vertices with closed dual cells: 200 / 215
--- gauge invariance, interior vertices ---
|F_B_orig - F_A_orig| (SHIPPED code): max=5.075834e-14
|F_B_fix  - F_A_fix | (fixed code) : max=3.607123e-15
--- interface v.p averaging with != 0.0 filter ---
interface v.p, run A (P0=0):    range [2.5, 5]
interface v.p, run B (P0=1000): range [1002.5, 1002.5]
gauge-shifted difference (v.p_B - v.p_A - 1000): range [-2.5, 0]
```
At t = 0 the fixture's pressure field is still ~1e-12 Pa (setup does not
recompute duals after the perturbation), so force errors are fp-dust — but the
mechanism fires 349 times and 72/247 present entries are exact zeros. The
interface `v.p` is already wrong: at P0 = 0 some interface vertices report the
full droplet pressure 5 Pa (outer phase excluded by the `!= 0.0` filter) instead
of the two-phase mean 2.5 Pa — a **gauge-dependent 50 %-of-Laplace-jump error**
in `v.p`.

**Probe 3** (compression front at gauge zero — the state every real P0 = 0 run
reaches as soon as pressures deviate):
```
K_o = 25000.0 Pa, gamma = 0.05, Laplace jump = 5.0 Pa
outer p_phase[0] over present entries: min=-2.7e-12, max=257.8 Pa; EXACT zeros: 26/102
--- (a) shipped-code force error at gauge P0=0 ---
|F_orig - F_fix|: max=2.4847e+00 N, mean=4.4614e-02 N   (|F| scale: max 2.74 N)
worst vertex: r=1.77 R0, phase=0, p_phase=[257.8, 0.0]
  |F_fix| there = 1.9221e+00 N  ->  relative error = 1.293   (129 %)
vertices with |dF| > 1e-12 N: 19/200
--- (b) gauge invariance (P0=0 field vs +1000 Pa offset) ---
SHIPPED: max |F(p+1000) - F(p)| = 2.4847e+00 N
FIXED:   max |F(p+1000) - F(p)| = 3.7228e-15 N
--- (c) action-reaction of the phase-0 pressure flux ---
sum over all edges of (F_ij + F_ji): SHIPPED |.| = 3.8733e+00 N,  FIXED = 0.0
worst edge: p_i=0 (r=3.54 R0) vs p_j=257.8 (r=1.77 R0):
  shipped F_ij+F_ji = [-1.074, -1.074] N (nonzero -> momentum created)
```

## Verdict: CONFIRMED_BUG

- Unit level: a present phase with a legitimate 0.0 gauge pressure is replaced
  by the fallback (probe 1) — 100 % error in the face pressure difference.
- System level: the shipped force operator is **not gauge invariant** (2.48 N
  violation where the fixed operator is invariant to 3.7e-15 N), and the
  pairwise pressure flux **violates Newton's third law** (3.87 N net momentum
  source over the mesh vs exactly 0 when fixed). Both are hard physics
  requirements; nothing in `docs_temp/02_physics_foundations.md` or the module
  docstrings sanctions the 0.0 sentinel (the docstring says "if populated" —
  0.0 *is* populated). This confirms audit flag #2 in
  `docs_temp/code_map/multiphase_surface_tension.md` with numbers.
- Error magnitude scales with the local pressure difference across faces that
  touch an exact-zero vertex: negligible at t = 0, up to ~100 % of the local
  force once O(100 Pa) perturbations develop — i.e. exactly during droplet
  oscillation.

## Impact on the oscillating droplet / bubble cases

Direct. The case defaults to `P0 = 0.0` for both TaitMurnaghan phases
(`_setup.py:39,116-119`), so every quiescent vertex stores `p_phase[k] = 0.0`
bitwise (72/247 present entries at t = 0; zeros persist through
`split_dual_volumes`/`compute_phase_pressures` recomputation on unmoved
vertices). As the oscillation develops, every dual face between the perturbed
near field and the quiescent far field has one side at exact zero → the
pressure-difference (restoring/radiating) flux on the perturbed side is zeroed
and a spurious net momentum source appears at the front. Secondary: interface
`v.p` (diagnostics/visualization, single-phase consumers) reports the full inner
pressure instead of the two-phase mean, gauge-dependently. Any published
oscillation-frequency/damping numbers from P0 = 0 runs carry this contamination;
running at P0 ≫ pressure amplitude (e.g. 101325) masks the bug entirely, which
also makes results P0-dependent when they must not be.

## Suggested fix

Presence must be keyed on geometry/mass, not on the pressure value:

1. `multiphase_stress.py:_phase_pressure` — replace the `val == 0.0` test with a
   presence test, e.g.
   ```python
   dvp = getattr(v, 'dual_vol_phase', None)
   if dvp is not None and k < len(dvp) and dvp[k] <= 1e-30:
       return fallback
   return float(p_phase[k])
   ```
   (verified by probes 2–3: restores gauge invariance and pairwise
   antisymmetry to machine precision). Alternatively store `NaN` for absent
   phases in `compute_phase_pressures` and test `isnan`.
2. `multiphase.py:559` — change the interface-average filter from
   `v.p_phase[k] != 0.0` to `v.dual_vol_phase[k] > 1e-30 and v.m_phase[k] > 1e-30`.
3. Add a regression test: gauge-offset invariance of `multiphase_stress_force`
   on the droplet fixture (assert `max|F(p+C) − F(p)| < 1e-12` on interior
   vertices) and the pairwise-flux antisymmetry sum.

## Skeptic review

> Adversarial re-verification 2026-07-02. Default position: claim wrong.
> Result: **claim survives — CONFIRMED, severity high upheld**, with one
> correction to the impact narrative (self-quenching, below).

**Code re-read (independent).** All cited lines verified verbatim:
`multiphase_stress.py:88-90` (`val == 0.0 → return fallback`, sole call site
line 181, neighbour side only — vertex i's own `p_phase[k]` at line 142 is read
*without* the zero check, which is what makes the flux asymmetric),
`multiphase.py:547-549` (0.0 written for absent phases) and `:559`
(`!= 0.0` interface-average filter), `_setup.py` default `P0=0.0`, and
`TaitMurnaghan.pressure(rho0) = P0` exactly.  The production run script
`oscillating_droplet_2D.py` does **not** override `P0`, so the flagship case
runs at gauge zero; `electrolysis_bubble` and `shearing_plate_droplet` setups
also default `P0=0.0`.  `_phase_pressure` has zero test coverage (no hit for
`_phase_pressure` or `gauge` under `ddgclib/tests/`).

**Refutation attempts and outcomes.**

1. *"Intentional design"* — refuted as a defense.  The module docstring scopes
   the fallback to neighbours that "do not store `p_phase[k]`"; 0.0 **is**
   stored for present quiescent phases at P0=0.  Decisively:
   `operators/mass_redistribution.py` (docstring of
   `redistribute_mass_multiphase`) already gates phase presence on the
   pre-retopo `dual_vol_phase[k]` *explicitly because* "pressure can
   legitimately be zero … required for cases at reference pressure P0=0".
   The codebase itself recognizes this exact trap and fixed it elsewhere —
   `_phase_pressure` and `multiphase.py:559` are the stragglers.
2. *"Probe artefact — the hand-injected probe3 front never arises in a real
   run"* — this attempt **partially landed and then failed**.  A 600-step real
   run (shipped integrator + retopo, default config) shows the exact-zero
   population collapses from 72/247 at t=0 to 0/247 by step 100, and shipped
   vs presence-keyed forces agree bitwise at every 100-step checkpoint:
   ulp-level noise from the per-step Delaunay retopo + per-phase
   mass-redistribution rescaling de-zeros the far field, so the auditor's
   claim that faces lose flux "for the whole oscillation" overstates
   persistence.  **However**, per-call instrumentation of `_phase_pressure`
   *during* every dudt evaluation (not just at checkpoints) shows the
   misfires are real in production: with defaults, **84 misfires within the
   first ~25 steps with substituted pressures up to 840 Pa** (face-pressure
   error ≈ 420 Pa = 100 % of the pressure-difference flux on those faces,
   comparable to the local |F| scale); with the public option
   `redistribute_mass=False`, **117 misfires with substitutions up to
   9379 Pa** (the ρ-clip band maximum).  The bug therefore fires in every
   default P0=0 run exactly during the wave-launch transient that sets the
   oscillation's effective initial conditions, injecting non-conservative
   momentum (probe3c: pairwise flux sum 3.87 N vs exactly 0 when fixed),
   then self-quenches.  How long zeros persist depends on fp accidents of
   retopo/redistribution — i.e. the error is also non-convergent under
   refinement and configuration-dependent, which is worse for validation,
   not better.
3. *"Static equilibrium runs are corrupted too"* — partially refuted, in the
   library's favour: at exact equilibrium (static droplet, ε=0) misfires
   substitute 0.0-for-0.0 in the outer phase, so the **force** error is nil
   there.  The interface `v.p` diagnostic (multiphase.py:559) is still wrong
   at that state (reports 5.0 Pa instead of the 2.5 Pa two-phase mean,
   gauge-dependently) — confirmed by probe 2 on the real fixture.

**Probe re-runs.** Probes 1–3 reproduced bit-identically (72/247 exact zeros,
349/1318 fallback firings, 2.48 N gauge violation / 129 % relative force
error / 3.87 N momentum source, all → machine precision under the
presence-keyed fix; control at P0=1000 shows shipped ≡ fixed, isolating the
`== 0.0` branch as the sole cause).

**Verdict:** CONFIRMED_BUG, severity **high** upheld.  Strongest evidence:
(a) in-run misfires with 840 Pa substitutions in the unmodified default
production pipeline; (b) gauge invariance and pairwise antisymmetry are hard
requirements, both violated by the shipped code and both restored to machine
precision by the one-line presence-keyed fix; (c) the codebase already
documents this exact failure mode in `mass_redistribution.py` and fixed it
there, so no "intentional design" reading survives.  One narrative
correction: the corruption is concentrated in the initial transient of each
P0=0 run (self-quenching via ulp noise), not sustained over the entire
oscillation — this does not change the severity, because the transient is
what launches the oscillation whose frequency/damping the case measures, and
the injected momentum persists in the Lagrangian velocity field.
The suggested fix (presence keyed on `dual_vol_phase`/`m_phase`, matching
`compute_phase_pressures`' own write-side gate at multiphase.py:543) is
correct and verified.
