# Audit: MultiphaseEOS.__call__ unguarded `v.phase` indexing on interface vertices
> Sources checked | Written 2026-07-02 by physics-audit workflow

Item key: `multiphase-eos-interface`
Verdict: **CONFIRMED_BUG (latent — dead path in production droplet/bubble configurations; severity low there, high if ever reached)**

Note: the task named `ddgclib/eos.py`; that file does not exist. The class
lives in `ddgclib/eos/_multiphase_eos.py` (package `ddgclib/eos/`), per
`docs_temp/code_map/multiphase_surface_tension.md:4`. Legacy `ddgclib/_eos.py`
is dead code.

Sources: `docs_temp/code_map/multiphase_surface_tension.md` (audit flags 1, 18),
`ddgclib/eos/_multiphase_eos.py`, `ddgclib/multiphase.py`,
`ddgclib/operators/multiphase_stress.py`, `ddgclib/operators/stress.py`,
`ddgclib/dynamic_integrators/_integrators_dynamic.py`,
`cases_dynamic/oscillating_droplet/src/_setup.py`,
`cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py`,
`ddgclib/tests/test_multiphase.py`.

## 1. What the physics / the code's own contract requires

The sharp-interface model assigns `v.phase = INTERFACE_PHASE = -1` to interface
vertices (`ddgclib/multiphase.py:66`, applied in
`identify_interface_from_subcomplex`, multiphase.py:389–394). The sentinel is
explicitly documented as a tripwire (multiphase.py:59–65):

> "Callers MUST check ``v.is_interface`` (or equivalently ``v.phase >= 0``)
> before indexing per-phase arrays by ``v.phase`` directly."

Per-phase thermodynamics require `rho_k = m_phase[k] / dual_vol_phase[k]`,
`p_k = eos_k.pressure(rho_k)` with no cross-phase EOS evaluation. An interface
vertex has no single "own phase"; the production convention for its
representative scalar `v.p` is `MultiphaseSystem.compute_phase_pressures`
(multiphase.py:555–561): arithmetic mean of the *nonzero* active phase
pressures.

## 2. What the code does

`MultiphaseEOS.__call__` (`ddgclib/eos/_multiphase_eos.py:46–89`) has **no
`v.phase >= 0` / `is_interface` guard anywhere**:

- **Path A** (per-phase arrays present, lines 65–74): loops over all k and
  fills `p_phase[k]`, `rho_phase[k]` **correctly**. But then lines 86–88:
  ```python
  own_p = v.p_phase[v.phase]      # v.phase == -1 -> numpy wraps to LAST phase
  v.p = own_p
  v.rho = v.rho_phase[v.phase] ...
  ```
  On an interface vertex this silently returns/stores the **last phase's**
  pressure and density (the droplet phase in every case setup, since the
  droplet/gas is phase 1 of 2).
- **Path B** (fallback, no per-phase arrays, lines 75–84):
  `rho = v.m / v.dual_vol` is a **mixture density** (the dual cell straddles
  both phases), fed into `self.eos_list[v.phase]` = `eos_list[-1]` = the
  **last phase's EOS** (line 81–82), and the result is written into slot
  `p_phase[-1]` (lines 83–84). Wrong density, wrong (wrapped) EOS, wrong
  storage slot — silent.

## 3. Probe design

Scripts under
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/multiphase-eos-interface/`,
run with `/home/endres/anaconda3/envs/ddg/bin/python`.

1. **`probe1_direct_call.py`** — builds the production oscillating-droplet
   setup (`setup_oscillating_droplet(dim=2, refinement_outer=1,
   refinement_droplet=2)`; R0=0.01 m, gamma=0.05 N/m → Laplace jump 5 Pa;
   TaitMurnaghan pair rho0=1000/800), verifies what phase interface vertices
   actually carry, then calls the very `meos` instance bound into the
   production `dudt_fn` (`dudt_fn.keywords['pressure_model']`) on an interface
   vertex. Path B exercised by removing `dual_vol_phase`. Run at `P0=1000` (to
   separate the two conventions) and at the case default `P0=0`.
2. **`probe2_reachability.py`** — monkeypatches `MultiphaseEOS.__call__` with
   a counter (split bulk / interface) and wraps `multiphase_stress_force` to
   count vertices missing `p_phase` at force entry, then runs the exact
   production loop of `oscillating_droplet_2D.py`: `symplectic_euler` with
   `retopologize_fn=_retopologize_multiphase` partial, script's CFL dt,
   60 steps, Delaunay retopo.

## 4. Probe OUTPUT

Probe 1, `P0 = 1000` (68 vertices, 16 interface):

```
(a) n_interface = 16, interface v.phase values = {-1}          <- sentinel confirmed
    v.p_phase           = [1000. 1005.]   (0=outer, 1=droplet)
(c) v.p after compute_phase_pressures  = 1.002500000e+03       (mean; production convention)
(b) meos(v) returned                   = 1.005000000e+03
    v.p_phase[-1] (last phase, drop)   = 1.005000000e+03
    -> returned == p_phase[LAST]: True;  == p_phase[outer]: False
    -> v.p changed from mean-convention by 2.5 Pa (0.25% rel); v.rho set to 800.2 (droplet rho)
(d) fallback branch (dual_vol_phase removed):
    mixture rho = m/dual_vol = 900.099923 kg/m^3
    meos(v2) returned                     = 4.701238919e+03
    eos_list[-1](drop).pressure(rho_mix)  = 4.701238919e+03    <- exact match
    spurious pressure error vs correct droplet p: 3.696239e+03 Pa  (vs 5 Pa Laplace jump)
    wrote into p_phase slot [-1] (phase 1); droplet EOS near clip edge (rho/rho0 = 1.125)
(e) no IndexError / warning raised anywhere -> silent wraparound confirmed
```

Probe 1, `P0 = 0` (the actual case default): `p_phase = [0. 5.]`;
`compute_phase_pressures` gives `v.p = 5.0` (its `!= 0.0` filter drops the
exactly-zero outer gauge pressure) and `meos(v)` also returns 5.0 — the two
conventions **coincide only by accident** of the zero filter. Path B fallback:
3701.2 Pa where the correct value is 5 Pa.

Probe 2 (production loop, 60 steps, Delaunay retopo every step):

```
mesh: 68 vertices, 16 interface;  dt = 8.571e-05, n_steps = 60
force evaluations: 3600
vertices missing p_phase at force entry: 0
MultiphaseEOS.__call__ total: 0   (setup: 0, integration: 0, interface: 0, bulk: 0)
interface vertices whose v.p deviates from compute_phase_pressures convention after run: 0
```

## 5. Reachability analysis (why the count is zero)

- `multiphase_stress_force` reads `v.p_phase` directly
  (`multiphase_stress.py:135,141–142`) and calls
  `_resolve_pressure(v, pressure_model, ...)` (→ `meos(v)` via
  `stress.py:659–661`) **only** when `hasattr(v, 'p_phase')` is False
  (`multiphase_stress.py:143–145`); the docstring at 130–132 states
  `pressure_model` "is not called inside the per-phase loop".
- `mps.refresh` (setup `_setup.py:164,194`) → `init_phase_fields` /
  `compute_phase_pressures` populates `p_phase` on **every** vertex, and
  `_retopologize_multiphase` re-refreshes after every Delaunay reconnection
  (`_integrators_dynamic.py:462`, plus `compute_phase_pressures` at :475 on
  the mass-redistribution path). Delaunay retopo reconnects but never creates
  vertices, so no vertex ever lacks `p_phase`.
- The integrator-level `pressure_model` kwarg
  (`_integrators_dynamic.py:141,241–247`, single-phase mass redistribution)
  is **not passed** by `oscillating_droplet_2D.py:129–133`; the multiphase
  wrapper uses `redistribute_mass_multiphase` + `mps.compute_phase_pressures`
  instead.
- All sibling cases bind `meos` identically as a dead fallback
  (`cube2droplet/src/_setup.py:158–159`,
  `electrolysis_bubble/src/_setup.py:458–459`,
  `shearing_plate_droplet/src/_setup.py:306–307`,
  `dam_break/src/_setup.py:176`).
- Interface pressures in production therefore always come from
  `compute_phase_pressures`, which **has** the guard (multiphase.py:555).

## 6. Verdict and reasoning

**CONFIRMED_BUG, severity low (latent).** The code demonstrably violates its
own INTERFACE_PHASE contract (multiphase.py:59–65) and its own class docstring
(the promised "own-phase pressure" is undefined for interface vertices):
probe 1 shows silent wraparound to the last phase's pressure/EOS with no error
and no warning — precisely the failure mode the sentinel comment says must be
gated. Path B additionally produces a thermodynamically inconsistent pressure
(mixture density × wrong-phase EOS: 3701 Pa where the physical value is 5 Pa,
a ~740× error) written into the wrong `p_phase` slot.

It is *latent* rather than active: probe 2 proves the vulnerable path executes
zero times in 3600 production force evaluations with per-step retopology,
because `mps.refresh` guarantees `p_phase` exists everywhere. The bug fires
only if (a) a vertex lacks `p_phase` — e.g. a vertex inserted by
`adaptive_remesh`'s `edge_split_2d` before any `mps.refresh`, or a
user-constructed mesh — or (b) a user follows the class docstring literally
and passes `meos` as `pressure_model` to the single-phase `stress_force`
(stress.py:753,769 call `_resolve_pressure` per vertex *and* per neighbour;
interface vertices then get droplet-phase pressure attributed to outer-phase
faces). The test suite never exercises an interface vertex through
`MultiphaseEOS` (`test_multiphase.py:237–276` uses bulk `FakeVertex` only), so
nothing would catch a regression here.

## 7. Droplet / bubble impact

**None in current production runs** (oscillating_droplet 2D/3D, cube2droplet,
electrolysis_bubble, shearing_plate_droplet, dam_break): interface pressures
and stress forces come from `compute_phase_pressures` + `p_phase`, verified
bit-identical to the convention after 60 integrated steps (probe 2, last
line). Coincidentally, in the default `P0=0` gauge the wraparound answer even
equals the production `v.p` (both 5.0 Pa) because the zero-pressure filter in
`compute_phase_pressures` discards the outer phase. The hazard is confined to
future/alternate wiring (adaptive remesh inserting vertices before a refresh,
single-phase `stress_force` + `meos`, diagnostics calling `meos(v)` directly),
where interface vertices would silently acquire kPa-scale spurious pressures
against a 5 Pa Laplace jump.

## 8. Suggested fix

In `ddgclib/eos/_multiphase_eos.py` `__call__`:

1. Guard the representative-value block (lines 86–88):
   ```python
   own = int(getattr(v, 'phase', 0))
   if own >= 0:
       own_p = float(v.p_phase[own])
   else:  # INTERFACE_PHASE: match compute_phase_pressures convention
       active = [k for k in getattr(v, 'interface_phases', range(n))
                 if 0 <= k < n and v.p_phase[k] != 0.0]
       own_p = float(np.mean([v.p_phase[k] for k in active])) if active else 0.0
   v.p = own_p
   ```
   Ideally factor a single shared helper used by both `MultiphaseEOS.__call__`
   and `MultiphaseSystem.compute_phase_pressures` so the two conventions can
   never diverge (they already differ today whenever both phase pressures are
   nonzero — 1005 vs 1002.5 Pa in probe 1).
2. In the fallback branch (75–84), raise `ValueError` (or at minimum
   `warnings.warn`) when `v.phase < 0` — a phase pressure cannot be computed
   from a mixture density, and `eos_list[-1]` is never the right EOS.
3. Add a `TestMultiphaseEOS` case with an interface `FakeVertex`
   (`phase=-1`, `interface_phases={0,1}`, per-phase arrays populated)
   asserting the returned `v.p` matches the `compute_phase_pressures`
   convention, and that the fallback branch raises/warns for `phase=-1`.
