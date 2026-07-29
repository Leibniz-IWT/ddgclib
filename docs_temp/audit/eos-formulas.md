# Audit: EOS correctness (TaitMurnaghan, IdealGas, MultiphaseEOS, compute_phase_pressures)
> Sources checked | Written 2026-07-02 by physics-audit workflow (re-verified same day: all probes re-run, all numbers reproduced; §2.2/§2.3 refined)

Sources: `docs_temp/02_physics_foundations.md` §5, `docs_temp/sources/fundamentals.md` (EOS update step),
`docs_temp/code_map/multiphase_surface_tension.md` §3, §6, §7.
Note: the task named `ddgclib/eos.py` — that file does not exist; the EOS lives in the package
`ddgclib/eos/` (`_base.py`, `_tait_murnaghan.py`, `_ideal_gas.py`, `_multiphase_eos.py`, `_update.py`).
`ddgclib/_eos.py` is 9 lines of dead code (truncated CoolProp wrapper, import commented out; confirmed by inspection).

Probe scripts (run with `/home/endres/anaconda3/envs/ddg/bin/python`, `PYTHONPATH=/home/endres/projects/ddgclib`):
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/eos-formulas/`
- `probe_eos_math.py` — formulas, round trips, clip, monotonicity (no mesh)
- `probe_meos_dispatch.py` — MultiphaseEOS dispatch incl. interface sentinel
- `probe_mesh_split.py`, `probe_mesh_split_circle.py`, `probe_refresh_diag.py` — mesh-level
  `m_phase/dual_vol_phase → rho → p` pipeline, exact-vs-neighbour_count split, clip-hit during a
  runtime-style `refresh(reset_mass=False)`

---

## 1. What the physics requires

- Tait–Murnaghan (Dymond & Malhotra 1988 / Cole form): `P(ρ) = P0 + B[(ρ/ρ0)^n − 1]` with
  `B = K/n`; water `K ≈ 2.15e9 Pa`, `n = 7.15`, `c(ρ0) = √(K/ρ0) ≈ 1466 m/s` (lit. ~1480).
- Ideal gas (isothermal): `P = ρ R T`, `ρ = P/(RT)`, `c_iso = √(RT)`.
- `pressure` and `density` must be mutual inverses (bijection) over the operating range, and
  `dP/dρ > 0` (real sound speed) wherever the EOS is active — this is the entire compressibility
  closure of the library (physics doc §5: "pressure responds to dual-cell volume change";
  `02_physics_foundations.md:24` — "No continuity equation is solved … Compressibility enters
  only through the EOS").
- Per the multiphase data model (`multiphase.py:59–65`), `v.phase == -1` (INTERFACE_PHASE) must
  never be used as an array index; "Callers MUST check `v.is_interface` (or equivalently
  `v.phase >= 0`) before indexing per-phase arrays by `v.phase` directly" (verbatim comment).

## 2. Findings

### 2.1 Core formulas: CORRECT (verified to machine precision)

`probe_eos_math.py` A1: `TaitMurnaghan.pressure` (`_tait_murnaghan.py:59-67`) matches an
independently coded reference at 6 densities with **|diff| = 0.0 exactly**; `B = K/n =
3.00699e8 Pa` (lit. ~3.0e8); `c(ρ0) = 1466.29 m/s` (= √(K/ρ0); lit. water ~1480 — parameters,
not formula); FD check `dP/dρ` at ρ0 = `2.150000e6` vs `sound_speed()²` rel-diff `5.6e-11`.
Sign/gauge conventions are consistent: `pressure()` (line 67) and `density()` (line 73,
`ratio = (n/K)(P−P0)+1`) both use `(P−P0)`; `P0` is a pure gauge offset, so absolute
(`P0=101325`) and gauge (`P0=0`) usage in the cases are both handled correctly.
IdealGas `P=ρRT` / `ρ=P/RT` (`_ideal_gas.py:42-48`) round-trips to `3.4e-16` rel.
Existing tests corroborate: `pytest ddgclib/tests/test_multiphase.py -k "Tait or IdealGas or
MultiphaseEOS or Pressure or Density"` → **8 passed** (re-run 2026-07-02, 0.46 s).

### 2.2 Round trip p == pressure(density(p)) over p ∈ [−1e4, 1e5]  (probe A2, 2001 pts)

| EOS | max abs / rel round-trip error | broken range |
|---|---|---|
| water Tait, default clip (0.9,1.1) | 3.8e-7 Pa / 2.9e-9 | none physical (3/2001 pts near P≈0 exceed 1e-9 *relative* purely by fp cancellation of (P−P0)/K against K=2.15e9; clip band spans [−1.59e8, +2.94e8] Pa, never engaged in this range) |
| water Tait, clip=None | identical (3.8e-7 Pa) | none |
| n=1 "linear Tait" (hydro/electrolysis style, K=1e5, clip (0.5,2.0)) | 4.4e-11 Pa / 1.1e-13 | none |
| IdealGas | 2.9e-11 Pa / 3.4e-16 | none (but `density(−1e4) = −0.1188` — negative density, no guard) |
| **soft droplet Tait K=800, n=7.15, P0=0, clip (0.8,1.2)** | **9.97e4 Pa / 0.997** | **[−1e4, 1e5] almost entirely (1994/2001 pts)** |
| soft droplet, clip=None | 9.89e3 Pa / 0.989 | P ∈ [−1e4, −155] |

The soft-EOS failure has two components:
1. **Intrinsic Tait domain limit**: no real density exists below `P0 − K/n = −111.888 Pa`;
   `density()` floors the ratio at 1e-30 (`_tait_murnaghan.py:74`) giving `ρ = 6.37e-2 kg/m³`
   for ALL P ≤ −112 Pa (probe A5: P=−112, −1000, −1e4 all → same ρ, round-trip back to
   −89.196 Pa) — a silent cavitation clamp, no warning/NaN.
2. **The clip**: the representable pressure window of the clipped soft EOS is only
   `[P(0.8ρ0), P(1.2ρ0)] = [−89.196, +300.143] Pa` (probe A5). Any state outside saturates.

### 2.3 Density clipping breaks the bijection and thermodynamic consistency (probe A3/A4)

`rho_clip` is applied **inside `pressure()` only** (`_tait_murnaghan.py:61-66`);
`density()` (71-75) and `sound_speed()` (79-83) ignore it:

- `ρ=850 → P = −1.5903e8 → density(P) = 900.0` (error 50 kg/m³); `ρ=1200 → density(pressure(ρ)) = 1100.0`
  (error 100 kg/m³). Bijection broken exactly at the band edges (probe A3 table).
- `dP/dρ` (numerical, ρ∈[800,1300], 5001 pts): inside band `[1.13e6, 3.85e6] > 0` (monotone,
  real sound speed); **outside band exactly 0.0** → effective sound speed 0, **while
  `sound_speed(1300)` reports 3285.45 m/s** and `sound_speed(800)` reports 738.28 m/s.
  Three mutually inconsistent answers to "what is the fluid stiffness here".
- Unclipped Tait over ρ∈[1,2000] (4000 pts): **0 negative diffs**; strictly increasing except
  a 10-point floating-point plateau at ρ∈[1, 6.5] kg/m³ where P saturates at `P0 − B =
  −3.006e8` (analytic dP/dρ there ≈ 6.6e-10, below fp resolution of |P|≈3e8). Monotone
  non-decreasing everywhere; genuinely fine for ρ > ~10 kg/m³.

**The clip is actually hit during dynamics** — see §2.5. And the inverse is used in the dynamics
loop: `redistribute_mass_multiphase` (`ddgclib/operators/mass_redistribution.py:314`,
`rho_target = float(eos_k.density(p_k_before))`; single-phase variant at line 156) inverts
*clipped* snapshot pressures. A vertex genuinely at ρ = 1.5ρ0 snapshots the saturated
p = 300.14 Pa; redistribution targets ρ = 1.2ρ0, i.e. it silently rewrites that vertex's phase
mass by −20% (globally rebalanced by the scale factor, so total mass is conserved but the
spatial distribution is corrupted toward the clip boundary).

### 2.4 MultiphaseEOS dispatch: bulk correct; interface sentinel violated (probe B)

- B3/B4: bulk dispatch exact — `phase 0, ρ=1020 → 2000.0 Pa` (= `eos0.pressure`, diff 0.0),
  `phase 1, ρ=1.05 → 50.0 Pa`; `pressure_for_phase` exact.
- **B1 (CONFIRMED BUG)**: interface vertex `v.phase = −1`, `p_phase = [5000, 100]` →
  `__call__` returns `p_phase[-1] = 100.0` and sets `v.p = 100.0`, `v.rho = rho_phase[-1] = 1.1`
  (`_multiphase_eos.py:86-88`, no `v.phase >= 0` guard). `compute_phase_pressures`
  (`multiphase.py:555-561`) would set `v.p = mean = 2550.0` — the two conventions **differ by
  2450 Pa (96%)** on the same state. Violates the documented sentinel contract
  (`multiphase.py:59-65`).
- **B2 (CONFIRMED BUG)**: fallback path (no `dual_vol_phase`) uses `self.eos_list[v.phase]`
  (`_multiphase_eos.py:81`) → for `v.phase=−1` the *last* phase's EOS is applied to the
  *total* density: returned **1.049e6 Pa** where the correct own-phase value is 5000 Pa.
- Mitigation in the canonical pipeline: `multiphase_stress_force` reads `v.p_phase` directly and
  only calls `pressure_model` when `p_phase` is absent (`multiphase_stress.py:135-142`; docstring
  128-132: "Not called inside the per-phase loop"), and `mps.refresh` repopulates `p_phase` every
  step — so the wrap path is dormant in the shipped oscillating-droplet flow
  (`cases_dynamic/oscillating_droplet/src/_setup.py:201-205` binds `pressure_model=meos`).
  It fires for any vertex reaching the force pass without per-phase arrays (e.g. freshly
  inserted vertices before a refresh).

### 2.5 compute_phase_pressures and exact vs neighbour_count split (probes C)

`compute_phase_pressures` (`multiphase.py:528-563`) maps `ρ_k = m_phase[k]/dual_vol_phase[k]`,
`p_k = eos_k.pressure(ρ_k)` with `1e-30` guards on both `vol_k` and `m_k` (543). Verified
**self-consistent**: with simplex labels held fixed, re-running `split_dual_volumes('exact')` +
`compute_phase_pressures` on a circular-droplet mesh gives **0 (vertex,phase) pairs with
ρ_k ≠ ρ0** and `p_phase ∈ [−8.7e-14, 0] Pa` (probe_refresh_diag control). Same at init for BOTH
split methods (probe_mesh_split C1: `max|ρ_k/ρ0−1| = 0.0` for `neighbour_count` and `exact`),
because `compute_phase_masses` (`multiphase.py:507-524`) derives masses from the *same* volumes.
So the m/V → ρ → p pipeline is internally consistent per split method. On a **straight**
axis-aligned interface the two split methods even agree exactly (probe C2: mixed nc→exact
re-split gives max ratio error 1e-15, split disagreement 0.0 — symmetric geometry).

**But** on a curved (droplet-like) interface the two splits disagree on the volumes themselves
(circular interface, refine=4): `|vol_exact − vol_nc|/vol_total` up to **0.25** (mean 0.063,
median 0.025) per interface vertex. Densities stay consistent only while masses and volumes come
from the same interface geometry. The runtime path breaks that: the integrator calls
`mps.refresh(HC, dim, reset_mass=False, split_method=...)` after every retopologization
(`_integrators_dynamic.py:462`), which relabels simplices by vertex majority vote
(`multiphase.py:647-652`) and re-splits volumes against **frozen Lagrangian mass**
(`_reinit_geometry_fields`, `multiphase.py:676-690`). On a droplet-like circular interface with
NO mesh motion at all (probe_refresh_diag / probe_mesh_split_circle):

- 8/1024 simplex labels flip, 8 vertices change `is_interface` status;
- 32 (vertex,phase) pairs get ρ ratios in **[0.5, 1.5]** (e.g. vertex (−0.5,0): init
  `dual_vol_phase = [0.0026, 0.0078]`, after refresh `[0.0052, 0.0052]`, `m_phase` frozen at
  `[2.604, 7.813]` → ratios 0.5 and 1.5);
- **24/64 interface pairs land OUTSIDE the (0.8, 1.2) clip band** (exact→exact control; 8/64
  for nc→nc, 17/64 for nc→exact, 24/64 for exact→nc); their `p_phase` pins at exactly the band
  edges **−89.196 / +300.143 Pa** (= `P(0.8ρ0)`, `P(1.2ρ0)`) — flat-lined EOS, `dP/dρ = 0`,
  zero restoring force;
- with the stiff water EOS unclipped, the same ρ error would produce **−2.99e8 … +5.16e9 Pa**
  spurious pressure (this is the physics-doc §5 "dominant spurious-force generator" quantified).

Mixing split methods across init/runtime (possible because the case setup and the integrator take
independent `split_method` parameters) produces the same 0.5–1.5 ρ ratios (probe C2b, "nc→exact"
and "exact→nc" mixed runs above).

### 2.6 Interface v.p average drops legitimate zero-gauge pressures (probe C3)

`compute_phase_pressures` filters `v.p_phase[k] != 0.0` (`multiphase.py:559`). Hand-crafted
interface vertex with phase 0 exactly at reference (gauge `p = 0.0`) and phase 1 compressed 10%:
`p_phase = [0.0, 109.29]`, physical mean of present phases = **54.64 Pa**, code sets
`v.p = 109.29 Pa`. A legitimately zero gauge pressure is treated as "missing" — same conflation
as `_phase_pressure` (`multiphase_stress.py:83-91`, `if val == 0.0: return fallback`). All
`P0=0` cases (oscillating droplet, electrolysis, hydrostatic gauge variants) sit at exactly this
reference state. Also the docstring (`multiphase.py:536-537`, "inner-phase pressure") does not
match the code (averages). Note: `mass_redistribution.py:304-313` deliberately uses the
volume-based presence guard instead, with a comment explaining exactly this pitfall.

### 2.7 IdealGas minor issues

- `sound_speed = √(RT) = 290.09 m/s` — isothermal, self-consistent with `P = ρRT`, but the ABC
  docstring (`_base.py:13-14, 27`) promises the *isentropic* speed (adiabatic air: 343.24 m/s).
  Stale doc, not a physics error given the isothermal law.
- Default `P0 = ρ0RT = 103085.04 Pa`, 1.74% above 1 atm (ρ0 = 1.225 is the 15 °C ISA density
  combined with T = 293.15 K = 20 °C). `P0` is unused by `pressure()`, so this only misleads
  code reading `eos.P0` as "1 atm".
- `density(P<0)` returns negative density with no guard (probe A4: `density(−1e4) = −0.1188`).

## 3. Verdict and reasoning

**CONFIRMED_BUG (severity high)** for the item as a whole, with this decomposition:

| Sub-item | Verdict |
|---|---|
| Tait formula, signs, gauge (P0) convention | CORRECT_AS_INTENDED (diff 0.0 vs reference) |
| IdealGas law + round trip | CORRECT_AS_INTENDED (minor doc/guard caveats) |
| Round trip over [−1e4, 1e5], stiff/linear EOS | CORRECT_AS_INTENDED (≤3.8e-7 Pa abs) |
| Monotonicity dP/dρ inside operating band | CORRECT_AS_INTENDED (0 negative diffs unclipped; fp plateau only at ρ<6.5 kg/m³) |
| `rho_clip` one-sided application (pressure clipped; density/sound_speed not) | **CONFIRMED_BUG** — breaks bijection at band edges (50–100 kg/m³ round-trip error); `dP/dρ = 0` outside band while `sound_speed()` reports up to 3285 m/s; inverse used on clipped pressures by `mass_redistribution.py:156,314` rewrites mass toward clip boundaries |
| Clip actually hit during dynamics | **CONFIRMED** — 8–24 of 64 interface pairs saturate after one runtime-style refresh on a static droplet-like interface (split-method dependent) |
| `MultiphaseEOS.__call__` unguarded `v.phase = −1` | **CONFIRMED_BUG** — last-phase wrap (96% / 200× errors in probes), violates documented sentinel contract (`multiphase.py:59-65`); dormant in canonical flow |
| `compute_phase_pressures` m/V → ρ → p pipeline | CORRECT_AS_INTENDED per split method (0 bad pairs, ≤1e-13 Pa) |
| `!= 0.0` zero-gauge filter in interface `v.p` | **CONFIRMED_BUG** (109.29 vs 54.64 Pa) |
| exact vs neighbour_count density consistency | DESIGN_LIMITATION at init (consistent by construction); **breaks O(1) under runtime relabelling/mixed methods** (ρ ratios 0.5–1.5) — the relabelling itself is outside this item's scope but is what pushes states into the clip |

Note: the one-sided clip and the sentinel wrap are listed as "known defects" in
`02_physics_foundations.md:83,108-110` — i.e. the audit docs describe the flawed behaviour —
but they contradict the intended physics (bijective EOS closure, sentinel contract), so they
remain CONFIRMED_BUG rather than documented design choices.

## 4. Suggested fixes

1. `TaitMurnaghan`: apply `rho_clip` consistently — either clip in all three of
   `pressure/density/sound_speed` (making the clipped EOS a coherent saturating model and the
   round trip idempotent), or better: don't silently clip; emit a (rate-limited) warning /
   counter when the clip engages so runs that have switched off compressibility physics are
   detectable. Expose `clip_engaged` diagnostics.
2. `MultiphaseEOS.__call__`: guard the tail (`_multiphase_eos.py:81,86-88`) with
   `if v.phase >= 0: ... else:` — for interface vertices return the same convention as
   `compute_phase_pressures` (or explicitly the mean/own-side value) instead of `p_phase[-1]`,
   and never index `eos_list[v.phase]` with the sentinel.
3. Replace the `p_phase[k] != 0.0` presence tests (`multiphase.py:559`,
   `multiphase_stress.py:88-90`) with volume-based presence: `dual_vol_phase[k] > 1e-30`
   (the geometry-aware snapshot in `mass_redistribution.py:304-313` already does exactly this
   for exactly this reason, per its own comment).
4. `IdealGas`: floor `density(P)` at 0 (or raise), fix ABC docstring (isothermal), and either
   set `rho0=1.204` for T=293.15 or document the ISA-density default.
5. Fix stale docstring `multiphase.py:536-537` ("inner-phase" → "mean of active phases").

## 5. Droplet/bubble case impact

Direct. The oscillating droplet (`cases_dynamic/oscillating_droplet/src/_setup.py:110-119`) uses
the soft Tait (`K = ρc²`, `c ~ 1 m/s` → K ≈ 800–1000 Pa, `P0` gauge, `rho_clip=(0.8,1.2)`; same
in `shearing_plate_droplet/src/_setup.py:144-146`) whose entire representable pressure window is
≈ **[−89, +300] Pa**; every integrator step calls `refresh(reset_mass=False)`
(`_integrators_dynamic.py:462`) and any interface-label churn or split-method inconsistency
pushes interface densities outside the band (probe: 24/64 pairs on a static droplet-like mesh),
pinning `p_phase` at the band edges with zero compressibility stiffness precisely at the
interface vertices that must carry the Laplace jump — and `redistribute_mass_multiphase` then
inverts those saturated pressures through the unclipped `density()`, silently rewriting
per-phase mass. The electrolysis bubble (`electrolysis_bubble_fritz_2D.py:409-412`) uses
`rho_clip=(0.5,2.0)` with `n=1` (window `[P0−K/2, P0+K]`, much wider relative to K, so
saturation is less likely but the same one-sided-clip inconsistency applies; the n=1 linear
Tait round-trips to 1e-13 inside the band). The `MultiphaseEOS` wrap-around affects any
droplet/bubble vertex that reaches the force pass without populated `p_phase` (e.g. inserted
vertices before a refresh).

## Skeptic review (adversarial re-verification, 2026-07-02)

Re-read all cited code lines and re-ran the probes plus three new counter-probes
(`probe_skeptic_full_pipeline.py`, `probe_skeptic_real_run.py`,
`probe_skeptic_pinned_ident.py` in the same scratchpad directory). **Verdict: the code-level
defects are real (not refuted), but the production harm story in §2.5/§5 is substantially
overstated for the shipped configurations. Severity downgraded high → medium.**

### Confirmed by independent re-reading / re-run

- `rho_clip` applied only inside `pressure()` (`_tait_murnaghan.py:61-66`); `density()` (71-75)
  and `sound_speed()` (79-83) ignore it — factually correct, and already listed as a known
  defect in `02_physics_foundations.md:108`.
- `MultiphaseEOS.__call__` has no `v.phase >= 0` guard (`_multiphase_eos.py:81,86-88`); the
  sentinel comment at `multiphase.py:59-66` explicitly makes any wrapping caller a bug.
- The `p_phase[k] != 0.0` filter (`multiphase.py:559`) and `_phase_pressure`'s `val == 0.0`
  fallback (`multiphase_stress.py:88-90`) exist as described.
- The clip **is** engaged during real shipped-default dynamics (see below), so "clip actually
  hit during dynamics" survives — but not where the auditor said.

### Refuted / overstated

1. **"Flat-lining dP/dρ exactly at the vertices carrying the Laplace jump" — refuted for the
   shipped configuration.** The auditor's mesh probes stop after `refresh(reset_mass=False)`,
   but the shipped pipeline (`_retopologize_multiphase`, `_integrators_dynamic.py:441-475`,
   with `redistribute_mass=True` — the default in both droplet case setups, `_setup.py:42/100`)
   always follows refresh with `redistribute_mass_multiphase` + `compute_phase_pressures`.
   Re-running the auditor's static-circle scenario through the FULL pipeline: 24-32 saturated
   pairs at the auditor's stop point → **0 out-of-band, 0 pinned pairs after the redistribution
   step**, every cycle, both split methods; the majority-vote churn is a one-time
   init-labelling→vote transition that is stable from cycle 2 onward. In a real instrumented
   45-step `setup_oscillating_droplet` run at shipped refinement (2/3), the pinned pairs
   (~9/247, mean 8.1/step) were **all frozen wall-boundary vertices (0 on interface vertices at
   every inspected step)** — wall mass is frozen and `bV` vertices are excluded from
   redistribution (`_is_redistributable`, `mass_redistribution.py:83-84`), so their densities
   drift while interface densities are healed each retopo. The Laplace-jump-carrying vertices
   never flat-line under shipped defaults. (The scenario does hold for `redistribute_mass=False`
   variants, which exist as comparison/diagnostic scripts.)
2. **The "[−89, +300] Pa representable window" does not describe the shipped droplet cases.**
   Oscillating-droplet defaults (`R0=0.01`, `epsilon=0.05`) give `c_s = max(10·ε·R0·1000, 1) =
   5 m/s` → `K_d = 2.0e4`, `K_o = 2.5e4 Pa`, windows `[−2230, +7504]` / `[−2787, +9379] Pa` —
   25× wider than claimed, and ~50× the case's dynamic pressure scale by the Ma≈0.1 design
   (`c_s = 10·u_scale` in `shearing_plate` too, `_setup.py:134-140`). K ≈ 800 Pa only occurs
   when the `c_s = 1 m/s` floor engages (small `ε·R0` / small `U_wall`).
3. **The `MultiphaseEOS` sentinel wrap is dead code in the canonical flow, not merely
   "dormant".** With `MultiphaseEOS.__call__` instrumented at class level, a 60-step shipped
   oscillating-droplet run produced **0 invocations** (any phase, let alone −1) and 0 vertices
   ever missing `p_phase`: setup calls `mps.refresh` before binding `dudt_fn`, `_reinit_geometry_fields`
   gives every vertex `p_phase` on every refresh, and `multiphase_stress_force` only calls
   `pressure_model` when `hasattr(v,'p_phase')` is False (`multiphase_stress.py:135,141-144`).
   Delaunay retopo inserts no vertices; only insertion BCs (not used in the droplet cases)
   could expose the path. Latent trap, real, but zero production traffic observed.
4. **IdealGas negative density has no production impact**: the only runtime inversion sites
   clamp it (`rho_target = max(rho_target, 1e-30)`, `mass_redistribution.py:157,315`);
   `initial_conditions.py:394,405` inverts analytic init pressures. Likewise `sound_speed`
   has **zero production call sites** (cases compute `c_s = sqrt(K/ρ)` manually), so the
   "3285 m/s vs dP/dρ=0" inconsistency is a latent API wart, not an active error source.
5. **The `!= 0.0` zero-gauge filter is diagnostic-level.** It affects only the scalar `v.p`
   on interface vertices (forces read `v.p_phase[k]` per phase); the `_phase_pressure`
   fallback coincides with the correct value at exact-reference states (all pressures are 0
   there), and exactly-0.0 float pressures are measure-zero during dynamics. Worth fixing
   (it can distort Laplace-jump diagnostics/plots), but it does not corrupt the dynamics.

### What survives, and residual risk

- The one-sided clip is a genuine, silent thermodynamic inconsistency and it *is* engaged
  persistently in shipped runs — at frozen wall vertices, whose pinned band-edge pressures
  (−2787 Pa in the probe) do enter neighbouring interior vertices' pressure fluxes via
  `_phase_pressure`. That is a real spurious near-wall force contribution; note however the
  clip *caps* (mitigates) what would otherwise be a far larger spurious pressure from the
  frozen-wall-mass bookkeeping artifact — removing the clip without fixing wall mass handling
  would make the shipped cases worse.
- `redistribute_mass=False` runs (exposed parameter; used in `oscillating_droplet_2D_mass_redist.py`
  and several diagnose scripts) DO experience the auditor's interface saturation at the first
  retopo.
- The suggested fixes in §4 all remain sensible (consistent/warned clipping, sentinel guard,
  volume-based presence tests, IdealGas floor, docstring fixes).

**Adjusted verdict: CONFIRMED_BUG, severity medium** — real, documented-as-known API/contract
defects with production exposure limited to (a) wall-vertex pressure pinning entering near-wall
forces under shipped defaults, (b) interface flat-lining only in non-default
`redistribute_mass=False` configurations, and (c) latent traps (`MultiphaseEOS` wrap, zero-gauge
filter, unclipped `density`/`sound_speed`) awaiting a non-canonical caller.
