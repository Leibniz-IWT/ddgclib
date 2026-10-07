# Time Integration, Retopologization, Boundary & Initial Conditions
> Sources: ddgclib/dynamic_integrators/_integrators_dynamic.py (1212 L), ddgclib/dynamic_integrators/_simulation.py (183 L), ddgclib/dynamic_integrators/__init__.py, ddgclib/_boundary_conditions.py (1030 L), ddgclib/initial_conditions.py (497 L) | Written: 2026-07-02 by understand-and-document workflow

All paths below relative to `/home/endres/projects/ddgclib/`.

Public API (`ddgclib/dynamic_integrators/__init__.py`): `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`, `DynamicSimulation`, `SimulationParams`. Helpers `_retopologize`, `_do_retopologize`, `_retopologize_multiphase`, `_recompute_duals` are module-private but imported directly by cases.

## Universal per-step ordering (all 5 integrators)

Every integrator step executes, **in this exact order** (stability-critical):

1. **`_do_retopologize(...)`** — retriangulate, rebuild barycentric duals, cache `v.dual_vol` + edge-area vectors, repopulate `bV`. (e.g. `euler` at `_integrators_dynamic.py:722`)
2. **`verts = _interior_verts(HC, bV)`** (`:531-533`) — `[v for v in HC.V if v not in bV]`. Frozen set = `bV` at this instant.
3. **`accel = _compute_accel(dudt_fn, verts, workers, **dudt_kwargs)`** (`:543-588`) — ALL accelerations computed before ANY vertex moves, so duals from step 1 are geometrically consistent for the whole force pass. Pressure/EOS evaluation happens *inside* `dudt_fn`: `ddgclib/operators/stress.py:634` `_resolve_pressure(v, pressure_model, HC, dim)` — `pressure_model=None` reads `v.p` as-is (incompressible); an `EquationOfState` computes `P = eos.pressure(m/dual_vol)` and **writes `v.p` and `v.rho` in place** (`stress.py:669-672`); a plain callable returns `pressure_model(v)`.
4. **State update** (scheme-specific, see below). Position moves via `_move(v, pos, HC, bV)` (`:521-528`), which pops/re-adds `v` from `bV` around `HC.V.move` because moving changes the vertex hash.
5. **`_apply_bc_set(bc_set, HC, bV, dt)`** (`:536-540`) — BCs applied **after** the update, on post-move geometry but with `v.dual_vol` still from step 1 (stale by one advection). Returns diagnostics dict.
6. `t += dt`; `_invoke_callback(callback, step, t, HC, bV, diagnostics)` (`:602-619`, auto-detects 3-arg `callback(step,t,HC)` vs ≥5-arg `callback(step,t,HC,bV,diagnostics)`); `_maybe_save_state(...)` (`:490-518`, JSON `state_{step:06d}_t{t:.6f}.json` via `ddgclib.data._io.save_state`, default fields `('u','p','m')`, only if both `save_every` and `save_dir` set).

Consequence: fields mutated by BCs (velocity zeroing, mass relaxation, vertex injection/deletion) take effect on the **next** step's force evaluation, after the next retopologize rebuilds duals.

## Integrator update schemes

### `euler` (`:652-751`) — explicit forward Euler, Lagrangian
```
a^n     = dudt_fn(v)
x^{n+1} = x^n + dt * u^n        # OLD velocity (:738)
u^{n+1} = u^n + dt * a^n        # (:739)
```
Buffered: `updates[v] = (x_new, u_new)` computed for all verts first, then applied (`:742-744`). Returns final `t = n_steps*dt`.

### `symplectic_euler` (`:756-841`) — semi-implicit, Lagrangian (primary integrator)
```
a^n     = dudt_fn(v)
u^{n+1} = u^n + dt * a^n        # velocity FIRST (:828)
x^{n+1} = x^n + dt * u^{n+1}    # NEW velocity   (:829)
```
Conserves a modified Hamiltonian; preferred for long-time energy behaviour.

### `rk45` (`:846-986`) — scipy `solve_ivp(method='RK45')` per macro step
- State vector `y = [x_0..x_{N-1}, u_0..u_{N-1}]`, length `2*N*dim` (`_pack_state` `:622-632`).
- Per macro step of length `dt`: retopologize **once** at start; `solve_ivp(rhs, t_span=(t, t+dt), rtol=1e-6, atol=1e-9)` (`:961-969`). Defaults `rtol=1e-6, atol=1e-9`; scipy does the internal adaptive sub-stepping/error control — there is no ddgclib-side CFL logic here.
- RHS (`:947-959`): `_sync_mesh` moves every interior vertex to the intermediate stage positions (`HC.V.move` per stage — expensive), sets `dydt[:n*dim] = u_flat`, `dydt[n*dim:] = accel`. **Duals/edge-area caches are NOT recomputed at intermediate stages** — stress operators inside the RHS use duals frozen at start-of-macro-step geometry (stale within the macro step).
- `sol.success == False` → `RuntimeError` (`:971-974`). Final stage synced back (`:977-979`), then BC/callback/save. If 0 interior verts the step is skipped with `t += dt` (`:941-943`).

### `euler_velocity_only` (`:991-1063`) — Eulerian, validation only
```
u^{n+1} = u^n + dt * dudt_fn(v)     # no position update (:1055-1056)
```
Still calls `_do_retopologize` every step (pass `skip_triangulation=True` to avoid pointless Delaunay churn on the fixed mesh; duals are still refreshed). Per project convention this integrator is for validation/equilibrium checks only — do not suggest it as the main scheme.

### `euler_adaptive` (`:1068-1212`) — CFL-adaptive explicit Euler
- Signature differs: `dt_initial`, `t_end` (no `n_steps`). `dt_max` defaults to `dt_initial` (`:1145-1146`). Loop `while t < t_end - 1e-15`, with `dt = min(dt, t_end - t)` anti-overshoot (`:1152-1154`).
- `velocity_only=True` **by default** (`:1070`, `:1168-1170`): only `u += dt*a`. With `velocity_only=False` it uses the **forward-Euler** update (position with OLD velocity, `:1172-1179`), *not* symplectic.
- `diagnostics['dt'] = dt` injected (`:1182`).
- **CFL control runs AFTER the step** (`:1189-1210`), so the new dt applies to the *next* step:
  ```
  u_max = max |v.u| over interior verts
  h_min = min edge length over interior verts' 1-rings   # O(E) scan each step
  dt = clip(cfl_target * h_min / u_max, dt_min, dt_max)
  ```
  Defaults: `cfl_target=0.5`, `dt_min=1e-12`, `dt_max=dt_initial`. If `u_max <= 1e-30` or no finite `h_min` → `dt = dt_max`.

### Shared kwargs (all integrators)
`workers` (fork-based multiprocessing in `_compute_accel` `:571-588`, Linux only, falls back to sequential elsewhere; uses module globals `_mp_dudt_fn/_mp_dudt_kwargs/_mp_verts` `:591-594`), `boundary_filter`, `retopologize_fn`, `merge_cdist`, `backend`, `periodic_axes`, `domain_bounds`, `skip_triangulation`, `pressure_model`, `redistribute_mass`, `remesh_mode`, `remesh_kwargs`, `displacement_eps`, `save_every`, `save_dir`, `**dudt_kwargs` (forwarded verbatim to `dudt_fn`).

**Stale docstring / trap**: module docstring examples at `_integrators_dynamic.py:24-25` and `:35-36` pass `HC` positionally AND `HC=HC` as keyword — this raises `TypeError: multiple values for argument 'HC'`. CLAUDE.md is right: bind `dim`, `mu`, `HC` into `dudt_fn` with `functools.partial`, never via `**dudt_kwargs`. Similarly `dim=` passed to an integrator is consumed by the integrator's own `dim` parameter and is NOT forwarded to `dudt_fn`.

## `_retopologize` (`_integrators_dynamic.py:48-247`) — the per-step topology/dual rebuild

Numbered steps (as executed):

0. `periodic_axes` set → delegates entirely to `ddgclib.geometry.periodic.retopologize_periodic` (`:124-131`, def at `ddgclib/geometry/periodic.py:344`) and returns.
   Early return if `len(HC.V) < dim+1` (`:136-137`) — **duals left stale** in that case.
1. If `redistribute_mass` and `pressure_model`: snapshot pressure field (`snapshot_pressure`, `:139-143`).
2. Unless `skip_triangulation`:
   - `merge_cdist > 0` → `HC.V.merge_all(cdist=merge_cdist)`; `bV.intersection_update(set(HC.V))` drops stale refs (`:146-153`).
   - **`remesh_mode='adaptive'`** (`:155-172`): `hyperct.remesh.adaptive_remesh(HC, dim=dim, **remesh_kwargs)` — local edge split/collapse/flip preserving sharp `v.phase` interfaces; **2D only** (raises `NotImplementedError` for dim≠2, deliberately not caught). Then `invalidate_simplex_cache(HC)` (adaptive ops bypass the simplex cache) and boundary via `HC.boundary()`.
   - **`remesh_mode='delaunay'`** (default, `:173-198`): disconnect ALL edges (`v.disconnect(nb)` for every pair); dim==1 → sort by x and connect as chain (`:180-184`); dim≥2 → `hyperct.ddg.connect_and_cache_simplices(HC, verts, dim, coords)` (global scipy Delaunay, populates `HC._simplices`); boundary via `boundary_from_simplices(HC, dim)` when the cache exists, else `HC.boundary()` (`:192-198`). Delaunay creates cross-phase edges at sharp interfaces — the motivation for adaptive mode.
   - `skip_triangulation=True` → keep connectivity, `dV = set(bV)` (`:199-202`). **Caveat**: if `bV` was previously filtered by `boundary_filter`, non-wall open-boundary vertices are then tagged `v.boundary=False` and `compute_vd` treats them as interior.
3. Tag `v.boundary = v in dV` on **all** vertices (`:204-207`) — the full topological boundary, before filtering.
4. `compute_vd(HC, method="barycentric")` (`:210`) — rebuild dual cells `v.vd`.
5. Cache dual volumes + oriented dual-face areas (`:212-230`): preferred path `hyperct.ddg.batch_e_star(interior, HC, dim, backend, orient=True, compute_volumes=True)` → `HC._edge_area_cache = edge_areas`; vertices in `failed` are force-promoted to boundary (`v.boundary=True; dV.add(v)`); **`v.dual_vol = 0.0` for all boundary vertices** (`:225`). Fallback (ImportError/NotImplementedError): `ddgclib.operators.stress.cache_dual_volumes(HC, dim)` (def at `stress.py:379`) and `HC._edge_area_cache = None`.
6. `boundary_filter` semantics (`:232-238`): `dV = {v for v in dV if boundary_filter(v)}`, then `bV.clear(); bV.update(dV)`. So `bV` is **wholly rewritten every step**: only filtered (typically wall) vertices are frozen; inlet/outlet topological-boundary vertices stay interior and advect. Any hand-curated `bV` from setup is discarded — use `boundary_filter` or `PositionalNoSlipWallBC` to re-tag walls each step. Note filtering happens *after* step 3 tagging, so `v.boundary=True` on filtered-out vertices persists until the next retopologize.
7. `redistribute_mass` + `pressure_model` → `redistribute_mass_single_phase(HC, dim, pressure_model, bV, pressure_snapshot)` (`:240-247`) — **mutates `v.m`** so the pre-retriangulation pressure field is preserved across topology changes (requires `pressure_model.density(P)` inverse).

### `_do_retopologize` dispatch (`:289-388`)
- `retopologize_fn=False` → skip topology management entirely (`:348-349`).
- **Displacement gate** `displacement_eps` (`:351-353`, gate at `:250-279`): skip retopo when every vertex moved `< eps` since last snapshot AND vertex id-set unchanged. **On the very first call it snapshots and SKIPS** (`:267-270`) — assumes setup already built valid duals (`compute_vd` must have been run at setup or step 1 crashes on missing `v.vd`). Rationale documented `:329-346`: avoids 3D Delaunay non-uniqueness churning ~48 cross-phase edges/step on near-cospherical static interfaces (Phase 2 finding 2026-04-29). Suggested value `1e-4 * h_min`. `None` (default) disables.
- `retopologize_fn` callable → called as `retopologize_fn(HC, bV, dim)`, with `remesh_mode`/`remesh_kwargs` forwarded only when its signature accepts them (inspect-based, `:355-374`); other kwargs (merge_cdist, boundary_filter, …) are NOT forwarded to custom callables. Use for surface meshes or `_retopologize_multiphase` wrappers.
- Else → default `_retopologize` with all kwargs. Snapshot positions after (`:387-388`).

### `_retopologize_multiphase` (`:391-475`) — not exported; bind via `retopologize_fn=partial(...)`
1. `redistribute_mass` + `mps` → `snapshot_geometry_multiphase(HC, mps.n_phases)` (pre-retopo `dual_vol_phase` needed to distinguish "phase at P0=0" from "phase absent", `:436-445`).
2. `_retopologize(...)` (no single-phase redistribution).
3. `mps.refresh(HC, dim, reset_mass=False, split_method=split_method)` — geometry (`dual_vol_phase`) + pressure recomputed; Lagrangian `v.m`/`v.m_phase` preserved. `split_method` default `'neighbour_count'` (legacy); `'exact'` = geometric 2D dual split consistent with per-phase stress force.
4. `redistribute_mass_multiphase(...)` then `mps.compute_phase_pressures(HC)` (`:465-475`) so `v.p_phase` reflects redistributed densities, not stale pre-redistribution values. This is the one place in the step cycle where pressures are explicitly recomputed *outside* `dudt_fn`.

`_recompute_duals(HC)` (`:478-487`): bare `compute_vd` only — positions changed, topology unchanged; does NOT refresh `dual_vol` or edge-area cache.

## `DynamicSimulation` / `SimulationParams` (`_simulation.py`)

`SimulationParams` (`:27-65`): `dt=1e-4, n_steps=100, t_end=None, dim=3, mu=8.9e-4, rho=1.0, skip_triangulation=False, extra={}`. `dudt_kwargs` property → `{'dim': dim, 'mu': mu, **extra}`. `rho` is stored but never forwarded anywhere.

`DynamicSimulation(HC, bV, params)` (`:68-183`): chainable `set_initial_conditions / set_boundary_conditions / set_integrator / set_acceleration_fn`. Default integrator = `euler_velocity_only` (`:87`). `run()` (`:126-183`): (1) `ic.apply(HC, bV)`; (2) build kwargs `{HC, bV, dudt_fn, dt, dim, callback, bc_set, skip_triangulation}` + `dudt_kwargs`; if integrator `is euler_adaptive` → replace `dt` with `dt_initial=p.dt` and `t_end = p.t_end or p.n_steps*p.dt` (`:173-178`), else `n_steps=p.n_steps`. Raises `ValueError` if no `dudt_fn`.

**Trap**: because `dim` and `HC` are named integrator parameters, `dudt_kwargs`' `dim` is swallowed by the integrator and `dudt_fn` receives only `mu` + `extra`. A raw `stress.dudt_i` used here would fall back to its own `dim=3`/`HC=None` defaults — pre-bind with `partial(dudt_i, dim=..., HC=HC)` and keep `extra` free of colliding names.

## Boundary conditions (`ddgclib/_boundary_conditions.py`)

Helpers: `identify_boundary_vertices(HC, criterion_fn)` (`:29-45`); `identify_cube_boundaries(HC, lb, ub, dim)` (`:48-75`, tol `1e-14` on each axis).

`BoundaryCondition` ABC (`:80-104`): `__init__(axis=2)`, abstract `apply(mesh, dt, target_vertices=None)`.

`BoundaryConditionSet` (`:109-156`): `add(bc, vertices=None)` (None → BC targets the full `bV` at apply time); `apply_all(mesh, bV, dt)` applies in **insertion order**, returns diagnostics keyed `f"bc_{i}_{ClassName}"`. Because `_retopologize` rewrites `bV` every step, a BC added with `vertices=None` tracks the current (filtered) `bV`; a BC added with an explicit set holds **stale vertex references** if those vertices are deleted/merged.

Per-class actions (all `apply(mesh, dt, target_vertices)`; `target_vertices=None` → all `mesh.V` unless noted):

| Class (line) | Action per step |
|---|---|
| `NoSlipWallBC(dim)` (`:161`) | `v.u = zeros(dim)` on targets. |
| `MovingWallBC(wall_velocity, dim)` (`:183`) | `v.u = wall_velocity` (constant vector or `fn(v)`); vertices do NOT translate — plate velocity is a BC, not mesh motion. |
| `ShearingPlateBC(plate_velocity, plate_axis, plate_coord, wrap_axes, dim=2)` (`:227`) | Translates plate vertices by `plate_velocity*dt` (`mesh.V.move`), **clamps** normal coord to `plate_coord`, wraps periodic axes into `[lb,ub]`, sets `v.u = plate_velocity`. Plate verts need NOT be in `bV` (clamped normal, free tangential). |
| `DirichletVelocityBC(value, dim)` (`:295`) | `v.u = value` (const or `fn(v)`). |
| `DirichletPressureBC(value)` (`:323`) | `v.p = value` (const or `fn(v)`). |
| `NeumannBC(field_name='u', flux_value=0.0)` (`:350`) | Copies field from nearest interior neighbour (nb not in target set); nonzero flux: vector fields get `+ flux_value*dist*normal_dir`, scalars `+ flux_value*dist`. Skips vertices with no interior neighbour. |
| `OutletDeleteBC(outlet_pos, axis=2, bV=None, backflow_clamp=None)` (`:395`) | Deletes every `v` with `x[axis] >= outlet_pos` (`mesh.V.remove`, `bV.discard`); on any deletion calls `hyperct.ddg.invalidate_simplex_cache(mesh)` (`:435-436`). `backflow_clamp` width w: for `x[axis] >= outlet_pos - w`, clamp `u[axis] = max(u[axis], 0)` — counters spurious backward push from truncated outlet duals. |
| `OutletBufferedDeleteBC(outlet_pos, buffer_width, axis=0, bV=None)` (`:449`) | Ghost buffer `[outlet_pos, outlet_pos+buffer_width]`: vertices crossing `outlet_pos` get velocity+position frozen at entry (`:510-514`); each step the BC **overrides the integrator's stress-contaminated update** by moving buffer verts to `correct_pos += frozen_u*dt` and resetting `v.u[:] = frozen_u` (`:517-536`); deleted at buffer end. Buffer keyed by `id(v)` because `move()` changes vertex hash (`:484-486`). Keeps outlet-adjacent duals complete → balanced stress, no backflow. |
| `PeriodicInletBC(unit_mesh, velocity, axis=2, inlet_pos=0.0, cdist=None, fields=['u','p','m'], period=1.0)` (`:542`) | Ghost clone of unit mesh one `period` upstream (`_reset_ghost` `:635-641`). Each apply: advance ghost by `velocity*dt`; inject ghost verts with `x[axis] > inlet_pos` (strict >) as `mesh.V[tuple(x)]`, copying `fields`; copy connectivity only among co-injected verts (`:669-678`); remove from ghost; re-clone when ghost depleted (`:684-687`); **`mesh.V.merge_all(cdist)` every apply** (`:689`, `cdist` auto = 0.5×min unit-mesh edge, `:579-595`); `invalidate_simplex_cache` when anything entered. Returns injection count. |
| `PositionalNoSlipWallBC(criterion_fn, dim=2, bV=None)` (`:700`) | Scans **all** mesh vertices each step; matching verts get `u=0`, `v.boundary=True`, added to `bV`. Catches freshly injected inlet verts at wall positions; the canonical way to keep walls frozen given `bV` is rewritten by retopologize. |
| `PressureReservoirBC(rho_target, tau_inv, gas_phase=0, update_m_phase=True)` (`:784`) | **Mutates mass**: `alpha = min(1, tau_inv*dt)`; `v.m += alpha*(rho_target*dual_vol - v.m)` on gas-phase verts with `dual_vol >= 1e-30`; mirrors into `v.m_phase[gas_phase]`. First-order mass sink/source; total mass NOT conserved (open boundary, intended). `tau_inv ≈ c_s/L_domain`. Docstring (`:820-824`): must run *after* the integrator step; mass change acts on next step's forces. |
| `AbsorbingPressureBC(tau_inv, phase=None, update_m_phase=True)` (`:861`) | Sponge/Sommerfeld: target density = mean of same-phase neighbour densities `nb.m/nb.dual_vol`; same relaxation `v.m += alpha*(rho_nb_avg*vol - v.m)`. `tau_inv ≈ c_s/buffer_width`. |
| `ExpandingDomainBC(initial_rho_target, tau_inv, tau_inv_target, gas_phase=0)` (`:941`) | Floating reservoir: drifts internal `rho_target += min(1,tau_inv_target*dt)*(rho_interior_mean - rho_target)` (interior mean excludes `v.boundary` verts, `:993-1010`), then PressureReservoir-style mass relaxation on targets. `tau_inv_target << tau_inv`. |

`MeshAdvancer(mesh, inlet_bc, outlet_bc, velocity)` (`:740-777`): standalone stepper (not a BC) — advects the whole mesh by `velocity*dt` along `inlet_bc.axis`, then `outlet.apply`, then `inlet.apply`. Used for pure-advection tests, not the stress pipeline.

**Stability note on mass-relaxation BCs**: they read `v.dual_vol`, but the `batch_e_star` path in `_retopologize` sets `dual_vol = 0.0` on all boundary vertices (`:225`) — such vertices are silently skipped by the `vol < 1e-30` guard. So these BCs only act on vertices that are *interior* per the current dual build; target them via explicit vertex sets near (not on) the frozen wall, or ensure the fallback `cache_dual_volumes` half-cell path is in effect.

## Initial conditions (`ddgclib/initial_conditions.py`)

Base: `InitialCondition.apply(HC, bV)` ABC (`:27-36`); `CompositeIC(*ics)` applies in sequence (`:39-47`).

Scalar (pressure) — these implement the FVM volume-averaged convention:
- `UniformPressure(P0=0.0)` (`:52`): `v.p = P0`.
- `HydrostaticPressure(rho, g=9.81, axis=2, h_ref=0.0, P_ref=0.0)` (`:63`): field `P(x) = P_ref + rho*g*(h_ref - x[axis])`. If `v.vd` and `v.dual_vol` exist → `v.p = volume_averaged_scalar(P, v, dim)` (`ddgclib.analytical._integrated_comparison`), i.e. `(1/Vol_i)∫P dV`, machine-precision for linear P; else point-wise fallback `P(x_vertex)` (`:99-108`). **Apply after `compute_vd`** to get the averaged branch.
- `LinearPressureGradient(G, axis=0, P_ref=0.0)` (`:111`): `P(x) = P_ref - G*x[axis]`, same dual-averaged/point-wise branching (`:137-146`).

Vector (velocity) — all **point-wise** (no volume averaging; the FVM convention in CLAUDE.md is stated for scalars):
- `ZeroVelocity(dim=3)` (`:151`): `v.u = zeros(dim)`.
- `UniformVelocity(u_vec)` (`:162`): `v.u = u_vec.copy()`.
- `PoiseuillePlanar(G, mu, y_lb=0, y_ub=1, flow_axis=0, normal_axis=1, dim=2)` (`:173`): `u_flow(y) = (G/(2 mu))(y - y_lb)(y_ub - y)` (`:206-209`).
- `HagenPoiseuille3D(U_max, R, flow_axis=2, dim=3)` (`:217`): `u_z(r) = U_max*max(0, 1-(r/R)^2)`, r over non-flow axes.

Generic / mass:
- `CustomFieldIC(fn, field_name='u')` (`:255`): `setattr(v, field_name, fn(v.x_a))`.
- `UniformMass(total_volume, rho=1.0)` (`:276`): `v.m = rho*total_volume/n_verts` — equal mass per vertex; local density `m/dual_vol` is NOT uniform on non-uniform meshes.
- `DualVolumeMass(rho=1000.0)` (`:300`): `v.m = rho*dual_vol` — exact uniform density incl. boundary half-cells; **requires `compute_vd` + `cache_dual_volumes` first**. Missing/tiny `dual_vol` → `v.m = rho*1e-30`. Beware: after an integrator-driven `_retopologize` with `batch_e_star`, boundary verts have `dual_vol=0.0` and would get near-zero mass — apply this IC after a setup using the `cache_dual_volumes` half-cell path.
- `HydrostaticEOSMass(eos, rho0=1000.0, g=9.81, gravity_axis=1, h_ref=1.0, P_ref=0.0)` (`:329`): compressible hydrostatic. Linear EOS (`eos.n == 1`, has `K`): closed form `P(z) = P_ref + K*(exp(rho0*g*(h_ref-z)/K) - 1)` (`:386-388`); general EOS: `solve_ivp` of `dP/dz = -eos.density(P)*g`, rtol=atol=1e-12 (`:390-399`); `depth <= 0` → `P_ref`. Sets `v.m = rho_eq*dual_vol` (1e-30 fallback), plus `v.rho` and `v.p` (`:401-412`) — thermodynamically consistent with `pressure_model=eos` in the stress pipeline.

Multiphase:
- `PhaseAssignment(criterion_fn)` (`:417`): `v.phase = int(criterion_fn(v.x_a))`. Phase labels survive retriangulation (vertex attributes).
- `MultiphaseMass(multiphase_system)` (`:436`): `v.m = mps.phases[v.phase].rho0 * dual_vol` (1e-30 fallback); requires duals + phases set.
- `MultiphasePressure(mps, inner_phase=1, curvature=None)` (`:461`): `v.p = eos.pressure(rho0)` per phase; if `curvature` given, adds Young–Laplace `Δp = gamma*curvature` to inner-phase verts, with `gamma = mps._gamma[(min(0,inner), max(0,inner))]` (`:491-497`) — key construction assumes the outer phase is 0.

## Stability-relevant summary (where meshes/duals/mass are rebuilt vs stale)

- **Duals fresh** at force evaluation in euler / symplectic_euler / euler_velocity_only / euler_adaptive: retopo → accel → move (no move before force pass).
- **Duals stale** inside `rk45` RK stages (mesh synced per stage; duals + `HC._edge_area_cache` from macro-step start). Also stale: after `_retopologize` early-return with `< dim+1` vertices; whole steps skipped by the `displacement_eps` gate (including the very first step — setup MUST have built duals).
- **BCs run post-move with pre-move `dual_vol`** (one step stale for the mass-relaxation BCs).
- **`bV` is rewritten every retopologize** (step 6); `boundary_filter` decides what stays frozen. `v.boundary` remains True on filtered-out topological-boundary vertices until next retopo (they are integrated but tagged boundary — intended, `compute_vd` needs the full boundary).
- **Mass mutation sites**: `redistribute_mass_single_phase`/`redistribute_mass_multiphase` (inside retopologize, pressure-preserving), `PressureReservoirBC`/`AbsorbingPressureBC`/`ExpandingDomainBC` (relaxation sinks/sources, intentionally non-conservative), `HC.V.merge_all` in `PeriodicInletBC.apply` and via `merge_cdist` (merge is NOT mass-conserving unless `mass_conserving_merge` is used first — noted at `_integrators_dynamic.py:413-417`).
- **Topology mutation outside retopologize**: `OutletDeleteBC`, `OutletBufferedDeleteBC`, `PeriodicInletBC` (all invalidate the simplex cache so the next retopo rebuilds cleanly; duals stay stale until then).
- **Stale docs**: `_integrators_dynamic.py:24-25` and `:35-36` examples raise `TypeError` (double `HC`); CLAUDE.md's `partial` binding is the correct pattern. `DynamicSimulation` silently drops `dim`/`HC` from `dudt_fn` kwargs (swallowed by integrator parameters) and never uses `params.rho`.
