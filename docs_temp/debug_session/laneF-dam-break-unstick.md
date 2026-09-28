# laneF-dam-break-unstick — make the dam break actually run, prove the remap where reconnection fires

Date: 2026-07-30.  Lane F of the wf6 session, executing next-prompt
option 2 (laneD §2.3 / laneE next-lane guidance).  Scratch evidence:
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/wf6/laneF/`
(probe driver `probe_dam.py`, run JSONs in `runs/` = broken case,
`runs2/` = fixed case, `summarize.py`).

## 1. Solver fix: `_do_retopologize` silently dropped kwargs for callable retopo_fns

LaneD's discovery confirmed and fixed at the root.  The callable branch
of `_do_retopologize` (`_integrators_dynamic.py`) forwarded ONLY
`remesh_mode`/`remesh_kwargs`; every other retopology kwarg the
integrators accept was silently dropped — `skip_triangulation` (the
dam-break dead flag: the shipped case ran per-step full Delaunay while
its comment claimed frozen connectivity), `boundary_filter`,
`merge_cdist`, `backend`, `periodic_axes`, `domain_bounds`,
`pressure_model`, `redistribute_mass`.

**Fix** (`NOTE(laneF-forward)`): those eight kwargs are now forwarded
when the callable declares them BY NAME — not into `**kwargs` sinks
(legacy dual-only closures use `**kw` to ignore unknown keys), and
never when a `functools.partial` chain already binds the name
(**explicit partial bindings win**; the droplet policy wrappers bind
`skip_triangulation=True` / `retopo_remap='conservative'` in partials
and must not be clobbered by integrator-level defaults).
`remesh_mode`/`remesh_kwargs` forwarding is bit-unchanged.

Audit of the same path: `retopo_remap` and `split_method` are NOT
integrator kwargs — the supported route is partial-binding (as the
droplet runner does); nothing is dropped for them, and passing
`retopo_remap` to an integrator fails loudly via `**dudt_kwargs` at
the first acceleration call.  Behaviour change is limited to callers
that pass the fixed kwargs at integrator level with a callable
retopo_fn: exactly `dam_break_2D.py` / `dam_break_3D.py` (their
`skip_triangulation=True` becomes live, i.e. the 3D case now honestly
runs the frozen connectivity its comment documents).

Regression: 6 new tests `TestRetopoKwargForwarding`
(`ddgclib/tests/test_dynamic_integrators.py`): by-name forwarding,
partial-binding precedence, var-kw sink exclusion, legacy 3-arg
compatibility, end-to-end through `symplectic_euler` (the exact
dam-break call shape).

## 2. Why the collapse stalled — three interlocking case defects (all measured)

LaneD measured |u|max <= 0.03 m/s at t_end x5.  Reproduced
(`runs/frozen_a2.0_t0.2.json`: |u|max 0.0155, front +0.0001 m, zero
flips over 2242 steps), then diagnosed:

1. **Structure-frozen pressure + flat IC (the fundamental one).**
   Under `redistribute_mass=True` the per-vertex pressure STRUCTURE is
   re-imposed from the pre-step snapshot every step; only a uniform
   per-phase offset can evolve (laneD §1.1 — "the entire compressible
   physics flows through scale_k").  The case's flat-at-P_atm IC
   ("the hydrostatic profile then develops dynamically" — impossible)
   therefore could NEVER develop the hydrostatic head.  Measured
   directly: after 0.098 s under gravity the liquid column reads a
   UNIFORM +0.121 Pa gauge (pure offset channel) instead of the ~981
   Pa head; horizontal stress acceleration on the dam face stays
   |a_x| <= 0.7 m/s^2 at all times (a real ~900 Pa jump across one
   cell is ~70 m/s^2).  No drive, no collapse — and vertically the
   bulk liquid is in permanent free-fall against the artificial
   viscosity only (at alpha_art=0.1 the unopposed settling crushes
   the bottom cells: NaN at t=0.157 with ZERO flips).
2. **Geometry: `col_h = 2a == H`.**  The "column" filled the tank
   lid-to-floor — no air above it, contradicting the module docstring
   diagram (column top at y=a); the top liquid row is frozen INTO the
   lid by NoSlipWallBC.  Not a dam break (at best a lock exchange).
3. **alpha_art=2.0 overdamping + horizon.**  mu_l_eff = alpha*rho_l*
   c_s*dx = 390 Pa s (390,000x water), Re(u_ref) = 0.18, creeping
   terminal velocity O(0.05) m/s = exactly the measured stall band;
   and t_end = 0.02 s is 1/8 of the gravity ramp time u_ref/g.

**Case fixes** (all in `cases_dynamic/dam_break/`):
- `src/_params.py`: `col_h = a` (documented square Martin–Moyce column
  with headspace, `NOTE(laneF-geometry)`); `alpha_art = 0.3`
  (`NOTE(laneF-alpha)`, sweep below); `t_end = 0.2` (~2.8 t_ref,
  `NOTE(laneF-horizon)`).
- `src/_setup.py`: hydrostatic per-phase mass preload
  (`NOTE(laneF-hydrostatic-ic)`, the droplet Young–Laplace preload
  pattern): `m_phase[k] = eos_k.density(P_hydro_k(y)) * dvp_k`, gas
  column atmospheric, liquid continuous with the gas at the column
  top.  Verified: bulk column reproduces the analytic head to 1e-6 Pa
  and the mid-height dam-face vertex feels a_x = +39.0 m/s^2 with
  vertical balance |a_tot_y| = 0.025 m/s^2 at t=0 — the dam is
  loaded AND released.
- `dam_break_2D.py`: dead `skip_triangulation=True` replaced by
  `retopo_fn = partial(retopo_fn, retopo_remap='conservative')` (the
  laneD remap; evidence below).  3D runner untouched (3D remap
  adoption is a standing DO-NOT).

## 3. A/B: remap ON vs OFF where reconnection actually fires — WIN

All runs: fixed case, refine 3 (145 verts, dx_min 0.0125), c_s 9.905,
dt = 1.262e-4 (CFL 0.1), t_end 0.2 (1585 steps) unless noted.  Flips =
edge-set diff across each retopo call at frozen positions (counted
every step); "bookkeeping clips" = pre-restore transient clips
overwritten by the remap's restore (laneE convention — split from
genuine clips; gas clips here are genuine, they appear only in
blow-ups).

| alpha | frozen (true dual-only) | delaunay (remap OFF) | delaunay + remap ON |
|---|---|---|---|
| 2.0 | ok, creep: KEpk 5.76e-5 @0.005, \|u\| 0.015, front +2.5mm, 0 flips | ok, identical to frozen (0 flips fire) | ok, KEpk 5.85e-5, \|u\| 0.015, 0 flips |
| 1.0 | ok, \|u\| 0.026, front +5.0mm, 0 flips | ok, identical | ok, \|u\| 0.027, 0 flips |
| 0.5 | ok, \|u\| 0.056, front +10.2mm, 0 flips | **BLOWS UP at its FIRST reconnection** t=0.199 (2 flips → KE 6.7e-4 → 6.6e+6) | **survives full horizon**, \|u\| 0.059, front +10.5mm, 2 flips absorbed, KE rise@0.036→fall 0.72, 472 bookkeeping clips |
| 0.3 | — | **dies t=0.125** (15 flip-steps, KE→4.3e12, 3318 genuine gas clips) | **survives full horizon**: KEpk 2.20e-3 @0.051 → 0.67, \|u\| 0.112, front +18.1mm (+36% col_w), **6 reconnection steps / 12 flips absorbed**, 0 genuine clips, mass 6.2e-15, n_iface 9 constant |
| 0.2 | — | — | dies t=0.162 (24 flip-steps absorbed first) |
| 0.1 | NaN t=0.092 (ZERO flips — frozen connectivity cannot follow deformation ~ dx) | **dies at FIRST flip** t=0.053 (KE x28 in ONE step at flip onset: 1.09e-2→3.08e-1) | dies t=0.094 — **1.74x survival, 28 flip-steps absorbed** (vs 1) |

Also: refine 4 — delaunay a0.1 dies t=0.033 / a0.5 t=0.068; remap a0.1
t=0.038 / a0.5 t=0.113 (remap always last-man-standing; finer mesh =
earlier flips + lighter air slivers).  Endurance remap a0.5 t_end 0.4:
survives to t=0.381 (5.3 t_ref) absorbing 23 reconnection steps.

**WIN condition met**: remap ON survives horizons/viscosities where
OFF blows up — at alpha 0.3 and 0.5 OFF dies at reconnection onset
while ON completes the horizon with physical KE shape; at alpha 0.1
ON absorbs 28 reconnection steps vs OFF's death at its first.  This
is the first demonstration on a case that NEEDS reconnection ⇒ case
default `retopo_remap='conservative'` shipped (+ smoke regression).
Bonus: laneD §2.3's unexplained 2.3x KE-plateau (remap vs shipped) on
the stalled smoke does NOT reproduce on the fixed case (KEpk 5.85e-5
vs 5.76e-5 at alpha 2.0, 1.5%): it was the level anchor holding the
phase mean in the structurally-frozen flat-pressure state — resolved.

## 4. Remaining blocker (localized, next lane): air sliver-cell F/m ejection

Every reconnection-active config eventually dies of ONE mechanism,
now precisely localized (ejection-instrumented probe, `--eject-debug`
analysis): during large deformation a reconnection event leaves an
AIR vertex on a sliver dual cell (volume → ~0 while dual-face areas
stay finite); its redistributed mass m = rho_g*dvp is O(1e-4) kg, the
stress force stays finite ⇒ a = F/m spikes ⇒ the vertex goes
ballistic (~km/s in one step, e.g. a gas vertex at (0.0727, 0.676) —
6.7x outside the domain — one step after a 2-flip event at t=0.0915),
coordinates overflow within ~100 steps ⇒ `QhullError` (coordinates
O(1e77-1e87) in the qhull message).  The level anchor then correctly
reads the exploded air dual volume as expansion (uniform gas pressure
−51 Pa) — a symptom, not the cause.  No non-ballistic position jumps
were detected (the position-jump detector found zero events: nothing
teleports; it is pure F/m integration).  This is the corner-vertex
force defect the `alpha_art` crutch papers over, caught at its root.
It bounds: alpha_art <= 0.2 (refine 3), refine 4 at any tested alpha,
and horizons >= 0.38 s at alpha 0.5.  A proper fix (sliver-aware
mass/force handling, or interface-preserving adaptive remeshing with
quality targets + per-phase-conserving vertex merge —
`mass_conserving_merge` currently does NOT merge `m_phase` ledgers)
is a lane of its own.  Because of it, the |u|max = O(u_ref) gate is
met only fractionally: \|u\|max 0.112 m/s = 11% of u_ref = 0.990 at
the shipped default (0.36 u_ref reached at alpha 0.1 before its
abort); front advance, KE rise-then-fall, mass <= 6.2e-15, no
NaN/abort at the configured horizon are all met.

## 5. Final shipped configuration + animation

`dam_break_2D.py` defaults: refine 3, alpha_art 0.3, t_end 0.2,
per-step Delaunay + conservative remap.  Full runner (159 snapshots,
no aborts): KE_liq (phase-1 only, runner diagnostic) rises to
1.0369e-3 J @ t=0.051 then decays to 6.3e-4; |u|max 0.007 → 0.107
m/s; interface toe runs out to x ≈ 0.069 along the floor with the
face rotated into the classic slump profile (phases snapshot);
probe-side numbers incl. interface vertices: KEpk 2.20e-3 @ 0.051,
front +18.1 mm, mass 6.2e-15.  `fig/dam_break_2D.mp4` regenerated via
`dynamic_plot_fluid` (phase + interface overlay), interface ring
intact (9 vertices throughout).

## 5b. Side effects on sibling runners (recorded, not battery-gated)

`dam_break_3D.py` and the `*_no_air.py` variants share `_params.py`
and therefore inherit the square column, alpha_art 0.3 and t_end 0.2;
the 3D runner additionally becomes TRUE frozen-connectivity via the
§1 forwarding fix (its documented intent — 3D remap adoption remains
a standing DO-NOT).  At the new horizon the 3D frozen run may abort
mid-collapse like 2D frozen does (its integrator call is
try/except-guarded); re-tune these runners when the sliver-ejection
lane lands.

## 6. Measurement battery (ddg env, repo root)

1. Floor battery `pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`:
   **18 passed** — all pins untouched.
2. Fast suite `pytest ddgclib/tests/ -m "not slow" -q`:
   **876 passed, 0 failures**, 12 skipped, 17 deselected, 2 xfailed
   (= 866 baseline + 6 `TestRetopoKwargForwarding` + 4
   `test_case_dam_break.py`).
3. `static_droplet_2D.py`: summary **1.1847162859108737e-03**
   (mass 0.0) — bit-identical.
4. `oscillating_droplet_2D.py`: l2 **0.17479361640597058** / tail
   **0.9998967874595965** / linf 0.32364245955409165 / mass
   2.4056717879332966e-14 — bit-identical to the laneE pin.
5. `oscillating_droplet_3D.py`: l2 **0.24811340819647862** —
   bit-identical to `baseline_oscillation_3d.json`.
6. hyperct: untouched this lane (no suite run required).

## 7. Changed files

- `ddgclib/dynamic_integrators/_integrators_dynamic.py` — callable
  retopo_fn kwarg forwarding (`NOTE(laneF-forward)`) + docstrings.
- `ddgclib/tests/test_dynamic_integrators.py` — NEW
  `TestRetopoKwargForwarding` (6 tests).
- `ddgclib/tests/test_case_dam_break.py` — NEW (4 tests: headspace
  geometry, hydrostatic-IC structure pin, dam-face release force,
  150-step collapse smoke on the shipped remap config).
- `cases_dynamic/dam_break/src/_params.py` — col_h, alpha_art, t_end
  (+ evidence comments).
- `cases_dynamic/dam_break/src/_setup.py` — hydrostatic per-phase
  mass preload IC.
- `cases_dynamic/dam_break/dam_break_2D.py` — retopo policy
  delaunay+remap (dead flag removed).
- `docs_temp/debug_session/laneF-dam-break-unstick.md` (this log),
  `debugging_plan.md` (status entry).
