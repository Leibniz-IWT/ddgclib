# Audit: stability-critical per-step operation order in the dynamic integrators
> Sources checked | Written 2026-07-02 by physics-audit workflow

Item key: `integrator-ordering`
Audited file: `/home/endres/projects/ddgclib/ddgclib/dynamic_integrators/_integrators_dynamic.py` (1212 L)
Docs checked: `docs_temp/04_solver_pipeline.md`, `docs_temp/code_map/integrators_and_bcs.md`, `docs_temp/02_physics_foundations.md` (§1–5)
Cross-checked case: `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` + `cases_dynamic/oscillating_droplet/src/_setup.py`
Probe scripts (run with `/home/endres/anaconda3/envs/ddg/bin/python`, outputs re-verified 2026-07-02):
- `scratchpad/audit/integrator-ordering/probe_trace.py` — monkeypatched trace of `_do_retopologize`, `_retopologize`, `hyperct.ddg.compute_vd`, `hyperct.ddg.batch_e_star`, `_move`, `_compute_accel`, `_apply_bc_set` over 2–3 steps of every integrator, plus a **geometric drift metric** `max|x_now − x_at_last_compute_vd|` at every force pass
- `scratchpad/audit/integrator-ordering/probe_cfl_and_droplet.py` — numerical CFL-formula verification + trace of the production droplet pipeline

---

## 1. What the physics requires

The Lagrangian FVM (docs_temp/02_physics_foundations.md §1–2, §5) integrates
`m_i du_i/dt = F_stress_i` where `F_stress_i = Σ_j σ_f · A_ij` and, for weakly
compressible flow, `p = EOS(m_i / Vol_i^dual)` (resolved inside the force pass,
`stress.py:663-673`). Stability requires per step:

1. Dual geometry (`v.vd`, `v.dual_vol`, `A_ij` = `HC._edge_area_cache`) must be
   **consistent with the current vertex positions when forces are evaluated** —
   otherwise the EOS misreads geometric staleness as physical compression (the
   documented 3D blow-up mechanism, 04_solver_pipeline.md §7).
2. All accelerations must be computed **before any vertex moves** within the
   force pass (no mid-pass geometry contamination).
3. BC mutations (velocity zeroing, mass relaxation) must not inject energy by
   racing the position update; pressure must not be updated from stale volumes.

## 2. What the code does — exact ordered lists (verified by reading + trace)

### `euler` (loop :720-751) and `symplectic_euler` (loop :810-841) — identical skeleton

Actual loop body (symplectic, `_integrators_dynamic.py:811-839`; euler differs
only in the update rule at :736-740):

```python
for step in range(n_steps):
    _do_retopologize(HC, bV, dim, ...)              # :812 (euler :722)
    verts = _interior_verts(HC, bV)                 # :821 (:731)
    accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)  # :823 (:733)
    updates = {}
    for v in verts:                                 # buffered — no moves yet
        u_new = v.u[:dim] + dt * accel[v][:dim]     # :828  (euler: x from OLD u :738)
        x_new = v.x_a[:dim] + dt * u_new            # :829  (euler: u_new :739)
        updates[v] = (x_new, u_new)
    for v, (x_new, u_new) in updates.items():
        v.u[:dim] = u_new
        _move(v, x_new, HC, bV)                     # :832-834 (:742-744)
    diagnostics = _apply_bc_set(bc_set, HC, bV, dt) # :836 (:746) — AFTER the move
    t += dt                                         # :837
    _invoke_callback(...); _maybe_save_state(...)   # :838-839
```

Ordered list per step:
1. `_do_retopologize` → inside `_retopologize`: Delaunay reconnect (:174-198) →
   tag `v.boundary` (:206-207) → `compute_vd` (:210) → `batch_e_star` dual_vol +
   `HC._edge_area_cache` (:214-226; boundary verts `dual_vol=0.0` at :225) →
   `bV.clear()/update(dV)` (:237-238) → optional `redistribute_mass_single_phase` (:241-247).
2. `verts = _interior_verts(HC, bV)` (:531-533).
3. **Force pass** — all accelerations before any move; EOS/pressure resolved
   INSIDE `dudt_fn` (`stress.py:663-673`: `rho = v.m / dual_vol` with the
   dual_vol cached in step 1; writes `v.p`, `v.rho` in place).
4. Buffered state update, then applied via `_move` (:521-528).
5. `_apply_bc_set` — post-move (velocity/mass mutations seen by NEXT step's forces).
6. `t += dt`; callback; save.

**Probe OUTPUT (probe_trace.py, symplectic_euler, 3 steps, dt=1e-4, TaitMurnaghan
EOS bound into `dudt_i`; euler identical):**
```
_do_retopologize BEGIN
  compute_vd #1
  batch_e_star (dual_vol+edge_area cache) #1
_do_retopologize END
FORCE PASS (25 verts) | moves_since_compute_vd=0 | max_pos_drift_since_vd=0.000e+00 (fresh duals)
[first vertex MOVE after last compute_vd]
_apply_bc_set | moves_since_compute_vd=25
callback step=0 t=1.000e-04
... (identical pattern steps 1, 2)
-- totals: compute_vd=3, batch_e_star=3, moves=75
```
→ **Forces always see fresh duals** (0 moves AND 0.0 geometric drift since the
dual build at every force pass); BCs run post-move with `dual_vol` one advection
stale (25 moves since duals built) — matches 04_solver_pipeline.md §1 exactly.
Requirement (2) holds because both loops buffer the full `updates` dict before
the first `_move`.

### `rk45` (loop :928-986)

```python
for step in range(n_steps):
    _do_retopologize(...)                      # :930 — ONCE per macro step
    verts = _interior_verts(HC, bV)            # :939
    y0 = _pack_state(verts, dim)               # :945
    def rhs(_t, y):                            # :947-959
        _sync_mesh(verts, x_flat, u_flat, dim, HC, bV)   # :950 — moves ALL verts
        dydt[:n*dim] = u_flat
        accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)  # :956
        ...
    sol = solve_ivp(rhs, (t, t+dt), y0, method='RK45', rtol, atol)      # :961-969
    _sync_mesh(verts, x_final, u_final, ...)   # :979 — accepted state
    diagnostics = _apply_bc_set(bc_set, HC, bV, dt)                     # :981
    t += dt; callback; save                    # :982-984
```

`compute_vd`/`batch_e_star` are **never** called inside `rhs` — duals and
`HC._edge_area_cache` are frozen at macro-step start.

**Probe OUTPUT (rk45, 2 macro steps, dt=1e-4):**
```
_do_retopologize BEGIN
  compute_vd #1
  batch_e_star #1
_do_retopologize END
[first vertex MOVE after last compute_vd]
FORCE PASS (25 verts) | moves_since_compute_vd=25  | max_pos_drift_since_vd=0.000e+00 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=50  | max_pos_drift_since_vd=8.904e-07 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=75  | max_pos_drift_since_vd=2.346e-06 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=100 | max_pos_drift_since_vd=3.519e-06 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=125 | max_pos_drift_since_vd=9.383e-06 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=150 | max_pos_drift_since_vd=1.043e-05 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=175 | max_pos_drift_since_vd=1.173e-05 (STALE DUALS)
FORCE PASS (25 verts) | moves_since_compute_vd=200 | max_pos_drift_since_vd=1.173e-05 (STALE DUALS)
_apply_bc_set | moves_since_compute_vd=225
-- totals: compute_vd=2, batch_e_star=2, moves=450
```
→ **Distillation claim CONFIRMED with geometric evidence**: 1 `compute_vd` +
1 `batch_e_star` per macro step vs 8 RHS force passes; the first stage is
evaluated exactly at `y0` (drift 0.000e+00 — scipy RK45's first eval is at the
initial state), stages 2–8 evaluate forces with vertices displaced up to
1.17e-05 (≈ |u|·dt) from the geometry the duals encode.

**Physical consequence**: within a macro step, `A_ij` and `v.dual_vol` (hence
EOS density `m/dual_vol`, `stress.py:665-668`) are frozen while positions and
velocities advance — **pressure IS updated from stale volumes at every
intermediate stage**. The acoustic restoring force (the EOS spring, the
stiffest term for this weakly-compressible method) is sampled zero-order in
geometry: the RK4(5) stages integrate a *different* ODE (frozen-geometry) than
the true one, so (a) the formal high order collapses to first order in the
geometric coupling, and (b) scipy's embedded error control is blind to the
geometric error — `rtol/atol` guarantee nothing about it. Pressure–volume work
accumulated along the stage path is inconsistent with the actual volume change,
so per-macro-step energy errors of O(dt·drift) are possible in either sign.
**The in-code docstring overclaims**: `:919-922` "Each RHS evaluation moves all
interior vertices via `HC.V.move()` so that discrete operators (pressure
gradient, viscous Laplacian, …) are evaluated at the correct intermediate
geometry" — true only for positions/velocities entering `d_ij` and `Δu`, false
for the dual-based inputs (`A_ij`, `dual_vol`), which the probe shows are never
rebuilt mid-step. `docs_temp/04_solver_pipeline.md:34` documents the staleness
correctly; the docstring does not.

### `euler_velocity_only` (loop :1041-1063)

```
1. _do_retopologize (:1043)   2. accel (:1053)   3. v.u += dt*a only (:1055-1056)
4. _apply_bc_set (:1058)      5. t += dt; callback; save (:1059-1061)
```
Probe: `moves=0`, `compute_vd=2` for 2 steps, force passes fresh
(drift 0.000e+00) every step. Eulerian, validation only per project convention.

### `euler_adaptive` (loop :1152-1210)

```python
while t < t_end - 1e-15:                            # :1152
    dt = min(dt, t_end - t)                         # :1154 anti-overshoot
    _do_retopologize(...)                           # :1156
    verts = _interior_verts(HC, bV)                 # :1165
    accel = _compute_accel(...)                     # :1166
    if velocity_only:                               # True by DEFAULT (:1070)
        v.u[:dim] += dt * a[:dim]                   # :1168-1170
    else:                                           # forward Euler, x from OLD u
        x_new = v.x_a[:dim] + dt * v.u[:dim]        # :1174  (NOT symplectic)
        u_new = v.u[:dim] + dt * accel[v][:dim]     # :1175
        ... _move ...                               # :1177-1179
    diagnostics = _apply_bc_set(bc_set, HC, bV, dt) # :1181
    diagnostics['dt'] = dt                          # :1182
    t += dt; callback; save; step += 1              # :1184-1187
    # CFL — AFTER the step (new dt applies to the NEXT step)
    u_max = max_i |v.u[:dim]|                       # :1190-1194
    if u_max > 1e-30:
        h_min = min edge length over interior 1-rings   # :1198-1203
        dt_cfl = cfl_target * h_min / u_max         # :1205
        dt = np.clip(dt_cfl, dt_min, dt_max)        # :1206
    else:
        dt = dt_max                                 # :1210
```
Defaults: `cfl_target=0.5`, `dt_min=1e-12`, `dt_max=dt_initial` (:1069, :1145-1146).

**Probe OUTPUT (probe_cfl_and_droplet.py Part A)** — exact numerical match:
```
h_min = 0.176777, u_max = 2.0, expected dt_next = clip(0.5*0.1768/2.0) = 4.419417e-02
observed (step, dt): [(0, 1.0), (1, 0.04419417382415922), (2, 0.04419...), (3, 0.04419...)]
step-0 dt == dt_initial (CFL runs AFTER step): True
step-1 dt == expected CFL dt: True (observed 4.419417e-02)
u=0 everywhere -> dt sequence: [0.7, 0.7, 0.6]   (== dt_max; final 0.6 is the t_end clamp)
```
Also probe_trace.py (velocity_only=False): retopo → fresh-dual force pass
(drift 0.0) → forward-Euler move → BC, identical skeleton to `euler`.

Assessment of the CFL signal: `max|u|` over interior verts and global `h_min`
over interior 1-rings is a sensible **advective** CFL, but the formula contains
**no acoustic term** (`c_s = sqrt(dP/drho)` of the EOS — the binding stability
constraint for this weakly-compressible method; cases pick dt from the acoustic
CFL manually, e.g. `oscillating_droplet_2D.py:80-87`) and **no capillary
timescale** (`dt_cap ~ sqrt(rho h^3/gamma)`, used at
`oscillating_droplet_2D.py:87`). Starting from rest (`u_max <= 1e-30`) it
returns `dt = dt_max` regardless of stiffness. Mitigation: `dt_max` defaults to
`dt_initial`, so dt can never exceed the user's initial (presumably acoustically
chosen) step — the scheme only *refines below* a user-supplied cap. Acceptable
but under-documented: the docstring (:1079-1086) presents this CFL as *the*
stability criterion, which it is not for stiff-EOS or surface-tension runs.

### `displacement_eps` gate (opt-in, default None)

`_displacement_gate_should_skip` (:267-270): **the very first call always
snapshots and SKIPS** (documented rationale :262-265, :339-343); later calls
skip when every vertex moved `< eps` and the vertex id-set is unchanged.

**Probe OUTPUT (symplectic_euler, displacement_eps=1e9, 2 steps, duals built at setup):**
```
_do_retopologize BEGIN / END        (no compute_vd, no batch_e_star inside)
FORCE PASS (25 verts) | moves_since_compute_vd=0  (setup duals, positions unmoved — fresh)
[moves...]
FORCE PASS (25 verts) | moves_since_compute_vd=25 (STALE DUALS)
-- totals during run: compute_vd=0, batch_e_star=0, moves=50
```
(The drift number printed for this run is a probe artifact — the position
snapshot belongs to the previous integrator's mesh; the `moves_since_compute_vd`
counter is the valid signal.) Step-1 forces run on duals one full step stale →
**stale by design**, bounded by eps; this is the deliberate Probe-5 trade
(:329-346). With the default `displacement_eps=None` the gate is off.

## 3. Energy-injection check on the ordering

- **No BC runs between force evaluation and position update** in any integrator
  (BC call sites :746, :836, :981, :1058, :1181 are all after the state update).
  A BC velocity overwrite post-move is the standard end-of-step BC pattern: it
  affects the next step's forces and positions (one-step weak lag), not an
  energy source. For the droplet pipeline, `NoSlipWallBC` targets outer-wall
  vertices (`_setup.py:197-198`) which are in `bV` (full topological boundary,
  `boundary_filter=None`, :237-238) and are excluded from integration
  (`_interior_verts` :531-533) — they never move, so re-zeroing u injects nothing.
- **Pressure is never updated from stale volumes in the fixed-step
  integrators**: the EOS resolves `rho = m/dual_vol` inside `dudt_fn`
  (`stress.py:663-673`) using the dual_vol cached by the *same* step's retopo,
  with zero moves and zero geometric drift in between (trace §2).
- The two places where pressure IS computed from stale geometry are `rk45`
  intermediate stages (§2, flagged) and steps skipped by `displacement_eps`
  (opt-in, bounded). Mass-relaxation BCs (`PressureReservoirBC` etc.) read
  `v.dual_vol` one advection stale and skip `dual_vol=0` boundary verts — the
  documented trap (04_solver_pipeline.md §4), not an ordering defect.

## 4. Cross-check: oscillating droplet / bubble production pipelines

`oscillating_droplet_2D.py:129-134` runs **`symplectic_euler`** with fixed dt
(acoustic + capillary CFL chosen manually at :80-88), `retopologize_fn =
partial(_retopologize_multiphase, mps=mps, split_method=...,
redistribute_mass=True)` (`_setup.py:218-220`; default True at `_setup.py:42`),
no `displacement_eps`, `remesh_mode='delaunay'`. Same for
`oscillating_droplet_3D.py:116`, `static_droplet_2D.py:126`,
`electrolysis_bubble_2D.py:167`. **Neither `rk45` nor `euler_adaptive` is used
by any droplet/bubble case** (only `cases_dynamic/template/example_features_demo.py`
references them).

**Probe OUTPUT (probe_cfl_and_droplet.py Part B, setup_oscillating_droplet dim=2,
symplectic_euler 2 steps — per step):**
```
_retopologize_multiphase BEGIN
  snapshot_geometry_multiphase (pre-retopo p,vol)
  _retopologize (Delaunay + duals) BEGIN
    compute_vd
  _retopologize END
  mps.refresh (dual_vol_phase + pressures, mass kept)
  mps.compute_phase_pressures (v.p_phase <- EOS(m/vol))
  redistribute_mass_multiphase (mutates v.m_phase)
  mps.compute_phase_pressures (v.p_phase <- EOS(m/vol))
_retopologize_multiphase END
FORCE PASS (60 verts) | moves_since_compute_vd=0 (fresh)
[first vertex MOVE after last compute_vd]
_apply_bc_set (NoSlipWallBC) | moves_since_compute_vd=60
callback step=0
```
→ Per-phase pressures (`v.p_phase`, read by `multiphase_stress_force`) are
recomputed twice inside the retopo (after `mps.refresh` and again after mass
redistribution — the second call at `_integrators_dynamic.py:475` ensures the
force pass sees redistributed masses, not stale densities), and the force pass
runs with zero moves since `compute_vd`. **Ordering is correct for the
droplet/bubble cases; the known 2D oscillating-droplet validation failure
(04_solver_pipeline.md §7: tail_growth 1.68–1.85 vs <1.0) is NOT an ordering
bug** — geometry is rebuilt fresh every step, so the failure mechanism lives
inside the Delaunay rebuild itself (dual-volume churn; separate audit items
`retopo-edge-cases`, `dual-volume-3d`), not in stale-force sequencing.

## 5. Verdict and reasoning

**DESIGN_LIMITATION (severity: low)** for the item as a whole:

- `euler`, `symplectic_euler`, `euler_velocity_only`, `euler_adaptive`:
  per-step order **CORRECT_AS_INTENDED** — retopo → fresh-dual force pass
  (all accelerations before any move, updates buffered) → move → BC → callback.
  Probe-verified with 0 moves AND 0.0 geometric drift at every force pass;
  no energy-injecting ordering found.
- `rk45`: distillation claim **CONFIRMED** — all RK stages reuse duals +
  `HC._edge_area_cache` frozen at macro-step start (probe: 1 `compute_vd` vs 8
  force passes per macro step, stage drift up to 1.17e-05). This is a known,
  documented design limitation (04_solver_pipeline.md :34, :76) — but the
  in-code Notes docstring (:917-926) falsely claims "correct intermediate
  geometry": a documentation defect worth fixing. Not used by any production case.
- `euler_adaptive`: CFL formula `dt = clip(cfl_target*h_min/u_max, dt_min, dt_max)`
  verified to machine precision; runs AFTER the step; signal is advective-only
  (no acoustic `c_s`, no capillary timescale), returns `dt_max` from rest —
  safe only because `dt_max=dt_initial` caps dt at the user's choice.
  Acceptable, under-documented. Also note `velocity_only=False` gives forward
  Euler (old-u positions), not symplectic — matches its docstring.
- `displacement_eps` gate: intentional, documented stale-dual window (first
  call always skips; staleness bounded by eps thereafter). Off by default.

## 6. Suggested fixes (non-blocking)

1. `rk45` docstring (:917-926): replace the Notes claim with an explicit
   statement that dual cells, dual volumes and oriented face areas
   (`HC._edge_area_cache`) are frozen at macro-step start, so intermediate
   stages evaluate pressure (EOS) and fluxes on stale geometry and scipy's
   error control does not see that error; recommend `symplectic_euler` with
   smaller dt for weakly-compressible runs (the docstring currently suggests
   this for performance reasons only).
2. `euler_adaptive`: when `pressure_model` is an EOS, include the sound speed
   in the CFL signal, e.g. `dt = clip(cfl_target * h_min / (u_max + c_s), ...)`
   with `c_s = sqrt(dP/drho)|rho0`; optionally a capillary limit
   `dt_cap = sqrt(rho h_min^3 / (2*pi*gamma))` when surface tension is bound.
   At minimum, document that the built-in CFL covers only advection and that
   `dt_initial` must satisfy the acoustic/capillary limits.
