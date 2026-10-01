# Audit: dynamic-integrator layer (method inventory for the solver-config registry)

Date: 2026-09-25. Read-only audit. Line numbers are current as of this date.
- `INT` = `ddgclib/dynamic_integrators/_integrators_dynamic.py` (1474 L)
- `SIM` = `ddgclib/dynamic_integrators/_simulation.py` (183 L)
- `ST` = `ddgclib/operators/stress.py` (913 L)

All other paths are relative to `/home/endres/projects/ddgclib/`.

Two runtime probes were run from the scratchpad with `PYTHONDONTWRITEBYTECODE=1`, so no repo files were written: `scratchpad/audit/probe.py` and `probe2.py`. Their results are marked **[measured]**. Everything else comes from reading the code.

---

## 0. Headline findings (read these first)

1. **In 2D, `batch_e_star` never runs.** hyperct raises `NotImplementedError("batch_e_star only supports dim=3")` (`hyperct/ddg/_operators.py:403-404`). So `_retopologize` step 5b always drops into the `except` fallback in 2D (`INT:277-280`). That fallback is `cache_dual_volumes`, and it sets `HC._edge_area_cache = None`. Consequences:
   - 2D boundary vertices keep real (half-cell) `dual_vol`. Only 3D zeroes them.
   - 2D forces always rebuild `A_ij` from `v.vd` (`ST:159-173`).
   - No "failed vertex" promotion ever happens in 2D.
   - The `NOTE(lane3-dual-volume)` comment at `INT:257-261` ("2D keeps batch_e_star's volumes here... Boundary zeroing convention preserved in both paths") is wrong.
   - **[measured]** on `rectangle(refinement=2)` after one `_retopologize`: `_edge_area_cache is None`, max boundary `dual_vol` 0.0156, total dual volume 1.0 (the full domain). On `box(refinement=1)`: cache set, boundary `dual_vol` 0.0, total 0.818.
   - The older docs repeat the "all boundary verts get dual_vol=0" claim as universal (`docs_temp/04_solver_pipeline.md` §3.5 and §4 BC trap; `code_map/integrators_and_bcs.md:82,136`). It is only true in 3D.
2. **After retopology, 3D forces do not use the documented p_ij area vector.** `stress_force` (`ST:826-834`) and `multiphase_stress_force` (`ddgclib/operators/multiphase_stress.py:172-181`) read `HC._edge_area_cache` first. In 3D `_retopologize` fills that cache from `batch_e_star(orient=True)` (`INT:234-237,276`). `ddgclib/tests/test_stress.py:2098-2123` asserts that this cache equals the **legacy e_star** path, which `ST:280-291` documents as "NOT linearly precise". The p_ij construction that `ST:52-71,182-277` calls primary is only used when there is no cache: before the first retopo, or with a custom retopo_fn that never sets the cache.
   - **[measured]** on `box(refinement=2)` and `ball(refinement=2)`: per-edge relative difference cache vs p_ij has median 0.125 and max 0.625 / 0.25. The linear-precision off-diagonal residual is 1.3e-3 / 9.0e-4 on the cache path vs 1e-18 on the p_ij path.
   - This is an implicit, dim-gated method choice. I am not assessing its physics impact here.
3. **`remesh_mode`/`remesh_kwargs` use the opposite precedence to every other forwarded kwarg.** The laneF by-name kwargs respect `functools.partial` bindings (`INT:436-461`). `remesh_mode`/`remesh_kwargs` are forwarded whenever the callable declares them (`INT:427-430`), with no bound-keyword check. So the integrator value **overrides** a partial binding. The 2D droplet runner documents this as a workaround (`cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py:146-149`).
4. **`DynamicSimulation.run` overrides a partial-bound `mu`.** `SimulationParams.dudt_kwargs` always contains `mu` (default 8.9e-4) (`SIM:61-65`). It is merged into the integrator kwargs (`SIM:170`) and then passed to `dudt_fn` at call time (`INT:995`). `functools.partial` lets call-time kwargs win, so `partial(dudt_i, mu=0.1, ...)` silently runs with 8.9e-4. A `multiphase_dudt_i` partial would raise `TypeError`, because it has no `mu` parameter. `DynamicSimulation` also forwards **no** retopology switch except `skip_triangulation` (`SIM:160-169`).
5. **`workers > 1` loses the EOS side effects.** `_resolve_pressure` writes `v.p` and `v.rho` in place (`ST:738-739`). Under fork-based workers (`INT:833-849`) those writes happen in child processes and are lost. The parent's `v.p` stays stale for BCs, callbacks and snapshots. Forces are still right.
6. **Retopology is chosen per case in three different ways, with different spellings:**
   - `retopo_policy_2d` = `'dual_only' | 'delaunay_remap' | <anything else → plain Delaunay>` (`cases_dynamic/oscillating_droplet/src/_params.py:139`, dispatch `oscillating_droplet_2D.py:96-99`, no else branch).
   - The 3D runner accepts only `'delaunay' | 'dual_only'` (`oscillating_droplet_3D.py:106-107,262-263`).
   - The dam break hard-codes a partial (`dam_break_2D.py:121`), or an integrator kwarg in 3D (`dam_break_3D.py:112`).

   None of this lives in the library.

---

## 1. Integrator functions

### 1.1 Shared kwargs (identical names and defaults in all five integrators)

The signatures are `INT:914-922`, `1018-1027`, `1108-1116`, `1253-1262` and `1330-1340`.

| kwarg | default | controls | where consumed |
|---|---|---|---|
| `dim` | `3` | spatial dim. Integrator-owned, **not** forwarded to `dudt_fn` | everywhere |
| `callback` | `None` | `callback(step,t,HC)` or `(step,t,HC,bV,diag)`, picked by arity ≥5 | `_invoke_callback` `INT:864-881` |
| `bc_set` | `None` | `BoundaryConditionSet.apply_all(HC,bV,dt)` after the move | `_apply_bc_set` `INT:798-802` |
| `save_every`, `save_dir` | `None, None` | JSON dump every N steps. Fields are hard-coded `('u','p','m')` | `_maybe_save_state` `INT:752-780` |
| `workers` | `None` | fork `multiprocessing` pool for `dudt_fn`. Linux only; sequential elsewhere | `_compute_accel` `INT:805-850` |
| `boundary_filter` | `None` | which topological-boundary verts get frozen into `bV` | `_retopologize` `INT:285-288` |
| `retopologize_fn` | `None` | `None` = default `_retopologize`; `False` = no topology management at all; callable = custom | `_do_retopologize` `INT:407-475` |
| `merge_cdist` | `None` | `HC.V.merge_all` before retriangulation. Not mass-conserving | `INT:148-154` |
| `backend` | `None` | **only** the `batch_e_star` backend (`INT:235`). Not passed to `compute_vd` (`INT:227`). Its meaning differs from `retopologize_periodic(backend='ghost')` | `INT:232-237` |
| `periodic_axes`, `domain_bounds` | `None, None` | switches to `retopologize_periodic` | `INT:125-132` |
| `skip_triangulation` | `False` | keep connectivity, refresh duals (the "dual_only" policy) | `INT:146,216-219` |
| `pressure_model` | `None` | **only** used for single-phase mass redistribution. Independent of the `pressure_model` bound into `dudt_fn` | `INT:142,291` |
| `redistribute_mass` | `False` | single-phase pressure-preserving mass redistribution (needs `pressure_model`) | `INT:142-144,291-297` |
| `remesh_mode` | `'delaunay'` | `'delaunay'` or `'adaptive'` (adaptive is 2D only) | `INT:156` |
| `remesh_kwargs` | `None` | forwarded to `hyperct.remesh.adaptive_remesh` | `INT:169` |
| `displacement_eps` | `None` | skip-retopology gate | `INT:410-412,477-478` |
| `**dudt_kwargs` | — | forwarded verbatim to `dudt_fn(v, **kw)` | `INT:995` etc. |

Integrator-specific kwargs:
- `rk45`: `rtol=1e-6`, `atol=1e-9` (`INT:1109`).
- `euler_adaptive`: `dt_initial` and `t_end` instead of `dt`/`n_steps`, plus `cfl_target=0.5`, `dt_min=1e-12`, `dt_max=None` (→ `dt_initial`, `INT:1407-1408`) and `velocity_only=True` (`INT:1331-1332`).

Every integrator calls `_do_retopologize` with the same 13-argument list (`INT:984-992`, `1074-1082`, `1192-1200`, `1305-1313`, `1418-1426`). No kwarg is threaded into one integrator and not another.

### 1.2 Per-step order

| step | `euler` | `symplectic_euler` | `rk45` | `euler_velocity_only` | `euler_adaptive` |
|---|---|---|---|---|---|
| 0 | — | — | — | — | `dt=min(dt,t_end-t)` `:1416` |
| 1 retopo | `:984` | `:1074` | `:1192` (once per macro step) | `:1305` | `:1418` |
| 2 interior verts | `:993` | `:1083` | `:1201`. If n==0: `t+=dt; continue`, **skipping BC, callback and save** `:1203-1205` | `:1314` | `:1427` |
| 3 accel (all before any move) | `:995` | `:1085` | inside `rhs` `:1218`, after `_sync_mesh` moves vertices to each stage state `:1212`. **Duals and cache are not rebuilt per stage** | `:1315` | `:1428` |
| 4 update | `x+=dt*u_old`, `u+=dt*a` (buffered) `:998-1006` | `u+=dt*a`, then `x+=dt*u_new` `:1089-1096` | `solve_ivp(RK45)` `:1223-1231`; RuntimeError on failure `:1233-1236`; final sync `:1239-1241` | `u+=dt*a` only `:1317-1318` | `velocity_only=True`: `u+=dt*a` `:1430-1432`; else **forward Euler**, a copy of the euler block `:1434-1441` |
| 5 BCs | `:1008` | `:1098` | `:1243` | `:1320` | `:1443`, then `diag['dt']=dt` `:1444` |
| 6 t, callback, save | `:1009-1011` | `:1099-1101` | `:1244-1246` | `:1321-1323` | `:1446-1449` |
| 7 dt control | — | — | scipy internal | — | after the step: `dt=clip(cfl*h_min/u_max, dt_min, dt_max)` over the **pre-BC** `verts` list `:1452-1472`. Advective CFL only, no sound speed |

`DynamicSimulation.run` (`SIM:126-183`):
1. Raise if there is no `dudt_fn` (`:148-152`).
2. `ic.apply(HC,bV)` (`:155-156`).
3. Build kwargs `{HC,bV,dudt_fn,dt,dim,callback,bc_set,skip_triangulation}` (`:160-169`), then `update(p.dudt_kwargs)` = `{dim, mu, **extra}` (`:170`).
4. If `integrator is euler_adaptive` (identity check), swap `dt` for `dt_initial` and set `t_end = p.t_end or n_steps*dt` (`:173-178`). Otherwise use `n_steps` (`:180`).
5. Call the integrator (`:182`).

Default integrator: `euler_velocity_only` (`SIM:87`).

### 1.3 Differences that look accidental
- `euler_adaptive` defaults to `velocity_only=True`, i.e. Eulerian. That contradicts the Lagrangian convention in CLAUDE.md. Its Lagrangian branch is forward Euler, not symplectic (`INT:1434-1441`).
- `rk45` skips BCs, callback and save when there are no interior vertices (`INT:1203-1205`). The other integrators still apply them.
- `rk45` + `workers>1` forks a new Pool on **every RHS evaluation** (`INT:1218` → `:847`).
- `DynamicSimulation`:
  - Only `skip_triangulation` is forwarded. There is no route for `retopologize_fn`, `remesh_*`, `boundary_filter`, `pressure_model`, `redistribute_mass`, `displacement_eps`, `workers` or `save_*`, except by accident through `params.extra`: because `update(p.dudt_kwargs)` merges `extra` into the **integrator** kwargs, any `extra` key that matches an integrator parameter name is taken by the integrator (`SIM:170`).
  - `SimulationParams.rho` is never used.
  - The `t_end` docstring says it "overrides n_steps", but for fixed-step integrators it is ignored (`SIM:37-38` vs `:180`).
- The module docstring examples (`INT:24-25`, `:35-36`) pass `HC` twice and raise `TypeError`.

---

## 2. Retopology layer

### 2.1 `_retopologize(HC, bV, dim, boundary_filter=None, merge_cdist=None, periodic_axes=None, domain_bounds=None, backend=None, skip_triangulation=False, pressure_model=None, redistribute_mass=False, remesh_mode='delaunay', remesh_kwargs=None)` — `INT:49-297`

Body in execution order:
0. **Periodic dispatch** (`:125-132`): if `periodic_axes` is set, call `retopologize_periodic(HC,bV,dim,periodic_axes,domain_bounds, boundary_filter, merge_cdist)` and **return**. `skip_triangulation`, `remesh_*`, `pressure_model`, `redistribute_mass` and `backend` are all ignored on this path.
1. `if len(HC.V) < dim+1: return`. Duals are left stale (`:136-138`).
2. Single-phase snapshot, if `redistribute_mass and pressure_model is not None` (`:141-144`).
3. If `not skip_triangulation` (`:146`):
   - a. Merge, if `merge_cdist>0` (`:148-154`): `merge_all`, `bV.intersection_update`, and a second `< dim+1` early return.
   - b. `remesh_mode=='adaptive'` (`:156-189`): `adaptive_remesh(HC, dim, **remesh_kwargs)` (`:169`; raises for dim≠2). Then 2D `rebuild_simplex_cache_2d` (`:179-180`) or else `invalidate_simplex_cache` (`:181-182`). Boundary from `boundary_from_simplices` if `HC._simplices` exists, else `HC.boundary()` (`:185-189`).
   - c. Otherwise, Delaunay (`:190-215`): disconnect every edge (`:192-194`). In 1D, connect a sorted chain (`:197-201`). In 2D/3D, `connect_and_cache_simplices(HC, verts, dim, coords)` (`:203-207`; the canonical 3D qhull order lives in hyperct). Boundary from `boundary_from_simplices` if the cache exists, else `HC.boundary()` (`:211-215`).
4. Else (`skip_triangulation`): `dV = set(bV)`, i.e. the previous, possibly **filtered**, `bV` (`:216-219`).
5. Tag `v.boundary = v in dV` on all vertices (`:223-224`).
6. `compute_vd(HC, method="barycentric")` (`:227`). The method is hard-coded and there is no backend. It uses the simplex-aware path when `HC._simplices` exists (`hyperct/ddg/_compute_dual.py:115,159,320`).
7. **Step 5b, volume and area cache** (`:231-280`):
   - `try: batch_e_star(interior, HC, dim, backend, orient=True, compute_volumes=True)` (`:232-237`). **This raises NotImplementedError unless dim==3.**
   - 3D: promote failed verts, `v.boundary=True; dV.add(v)` (`:238-240`).
   - 3D: if `_use_exact_barycentric_volume(HC)` (`ST:311-324`: `_simplices` exists and `_vd_method=='barycentric'`), use `simplex_dual_volumes` (`:262-269`). Otherwise use the batch fan-walk volumes (`:273-275`). **Boundary verts get 0.0** (`:272,275`). `HC._edge_area_cache = edge_areas` (`:276`).
   - `except (ImportError, NotImplementedError)` (1D, 2D, old hyperct): `cache_dual_volumes(HC, dim)`, then `HC._edge_area_cache = None` (`:277-280`). `cache_dual_volumes` (`ST:433-466`) uses exact simplex volumes for dim 2/3 when available, **including boundary verts**. Otherwise it uses per-vertex `dual_volume` (`ST:327-426`).
8. Freeze set: `dV = {v in dV if boundary_filter(v)}`, then `bV.clear(); bV.update(dV)` (`:285-288`).
9. Single-phase redistribution: `redistribute_mass_single_phase(HC,dim,pressure_model,bV,snapshot)` (`:291-297`).

### 2.2 `_do_retopologize(HC, bV, dim, boundary_filter=None, retopologize_fn=None, merge_cdist=None, periodic_axes=None, domain_bounds=None, backend=None, skip_triangulation=False, pressure_model=None, redistribute_mass=False, remesh_mode='delaunay', remesh_kwargs=None, displacement_eps=None)` — `INT:339-478`

1. `retopologize_fn is False` → return (`:407-408`). The gate is not consulted and nothing is refreshed.
2. Displacement gate, if `displacement_eps>0` (`:410-412`; helper `:300-329`). **The first call snapshots and skips.** Later calls skip if the vertex id-set is unchanged and every displacement is `< eps`.
3. Callable path (`:414-464`):
   - `remesh_mode`/`remesh_kwargs` are forwarded if declared by name **or** if the callable has `**kwargs`. There is no partial-binding check (`:427-430`).
   - `bound_kw` collects the keywords from the whole partial chain (`:436-440`).
   - `skip_triangulation, boundary_filter, merge_cdist, backend, periodic_axes, domain_bounds, pressure_model, redistribute_mass` are forwarded only when **declared by name** and not partial-bound (`:448-461`). The `VAR_KEYWORD` kind check at `:459-460` can never be true for a named parameter, so it is dead.
   - Signature failure → forward nothing (`:462-463`). Then call `retopologize_fn(HC,bV,dim,**extra)` (`:464`).
   - **Never forwarded:** `displacement_eps` (handled here), `retopo_remap`, `projection_every`, `split_method`, `mps`.
4. Default path: `_retopologize(...)` with every kwarg (`:465-475`).
5. Gate snapshot after the call (`:477-478`).

### 2.3 `_retopologize_multiphase(HC, bV, dim, mps=None, boundary_filter=None, merge_cdist=None, backend=None, skip_triangulation=False, redistribute_mass=False, remesh_mode='delaunay', remesh_kwargs=None, split_method='neighbour_count', retopo_remap=None, projection_every=1)` — `INT:481-737`

It does **not** accept `periodic_axes`, `domain_bounds` or `pressure_model`. Those integrator-level values are silently not forwarded (they are not declared).

Body:
1. Validation:
   - `retopo_remap ∈ {None,'conservative'}`, else ValueError (`:592-596`).
   - `remap_active = remap=='conservative' and not skip and mps is not None` (`:597-598`). With `mps=None` the remap is **silently off**.
   - Remap active without `redistribute_mass` → ValueError (`:599-602`).
   - `projection_every` must be an int ≥1 (`:603-606`).
2. Cadence (`:607-623`): if N>1, it needs mps and redistribute (`:609-613`), and it needs `skip or remap_active` (`:614-620`). `project_now = idx % N == 0`, and the counter lives on `mps._projection_call_idx` (`:621-623`).
3. Snapshot, if `redistribute_mass and mps and (project_now or remap_active)`: `snapshot_geometry_multiphase` (`:630-636`).
4. Remap stage 1 (`:638-669`):
   - `_retopologize(..., boundary_filter, backend, skip_triangulation=True)`. No `merge_cdist` and no remesh (`:652-653`).
   - `mps.refresh(reset_mass=False, split_method)` (`:654`). `_vol_mid = phase_volume_totals` (`:655`).
   - If off-cadence: `_p_snap = evolve_snapshot_local_strain(...)` (`:656-669`).
5. Main `_retopologize(..., boundary_filter, merge_cdist, backend, skip_triangulation, remesh_mode, remesh_kwargs)` (`:672-676`). No single-phase redistribution.
6. If mps (`:679`):
   - `mps.refresh(HC, dim, reset_mass=False, split_method)` (`:686`). This includes `compute_phase_pressures` (`ddgclib/multiphase.py:692`) and `strict_closure=reset_mass=False` (`multiphase.py:683`).
   - If `redistribute_mass and _p_snap`: `redistribute_mass_multiphase` (`:689-695`).
     - If remap: `mps.vol_corr[k] = scale_factor` (`:696-709`).
     - `mps.compute_phase_pressures` (`:713`).
     - If remap: `restore_pressure_multiphase`, then `anchor_phase_pressure_levels(_vol_mid, _vol_new)` (`:714-737`).

**Decision table.** S = `skip_triangulation`, R = `retopo_remap`, M = `redistribute_mass`, P = mps given, N = `projection_every`.

| S | R | M | P | N | path executed |
|---|---|---|---|---|---|
| any | bad value | any | any | any | ValueError |
| F | cons | F | T | any | ValueError |
| any | any | any | any | <1 or non-int | ValueError |
| any | any | F or no P | any | >1 | ValueError |
| F | None | T | T | >1 | ValueError (lane-5 KE pump) |
| F | None | F | T | 1 | plain Delaunay + refresh. No redistribution. Lagrangian mass held against the new duals |
| F | None | T | T | 1 | **plain Delaunay + refresh + per-phase redistribution + pressure recompute** (library-default shape; setups bind M=T) |
| T | None/cons | F | T | 1 | dual_only, refresh only. The remap is a silent no-op under S=T (`:597-598`) |
| T | None/cons | T | T | 1 | **dual_only + redistribution every call** (3D default policy) |
| T | None/cons | T | T | N>1 | dual_only; redistribution only when `idx%N==0` |
| F | cons | T | T | 1 | **stage 1 (dual refresh at old connectivity) + Delaunay + redistribution + vol_corr + restore + anchor** (2D default policy; dam break 2D) |
| F | cons | T | T | N>1 | as above, with off-cadence snapshot = strain-advanced (laneH opt-in) |
| any | cons | any | F (None) | 1 | remap silently off; plain `_retopologize` only, no multiphase refresh |
| F | any | any | any | any, with `remesh_mode='adaptive'` | adaptive replaces Delaunay in step 5. Remap/cadence combined with adaptive is **unvalidated** (no guard, no test found). Adaptive split/collapse changes the vertex set between snapshot and restore |
| F | cons | T | T | any, with `merge_cdist>0` | the merge happens only in stage 2, after `_vol_mid`. **Unvalidated.** `merge_all` does not merge `m_phase` ledgers (laneF §4) |

Inside `_retopologize` (§2.1), the sub-path is set by: `periodic_axes` (periodic) > `skip_triangulation` (carry bV) > `remesh_mode` (adaptive/Delaunay) > dim (1D chain / 2D-3D qhull). The volume/area sub-path is set by dim (3 → batch; else fallback) and by `HC._simplices` (exact volumes or not).

### 2.4 Other retopology helpers
- `_displacement_gate_should_skip` / `_snapshot_retopo_positions` (`INT:300-336`). Keyed by `id(v)`. First call → skip.
- `_recompute_duals(HC)` (`INT:740-749`): bare `compute_vd`. It does not refresh `dual_vol` or the edge cache. Unused inside the library; the hand-rolled loops call it (`cases_dynamic/Hydrostatic_column/*.py`, `capillary_rise/capillary_rise_3D_dynCA.py:103,155,175`, `liquid_bridge_approach/_ddgclib_case_core.py:1611`).
- `retopologize_periodic` (`ddgclib/geometry/periodic.py:344-540`): wraps positions, optionally merges, drops ub-face duplicates, runs ghost Delaunay and `connect_and_cache_simplices`, applies a periodic boundary rule, runs `compute_vd`, sets `HC._periodic_axes/_bounds`, calls **`cache_dual_volumes`** (boundary half-cells kept) and filters `bV`.
  - It never sets or clears `HC._edge_area_cache`.
  - Its own `backend` means `'ghost'|'cgal'`.
  - `ST` only has a periodic min-image path in 2D (`ST:102-156`). The 3D `dual_area_vector` has none (`ST:175-176`).
- Case-local retopo functions (not in the library, but they act as method values):
  - `_make_periodic_multiphase_retopo` (`cases_dynamic/shearing_plate_droplet/src/_setup.py:330-381`): periodic + refresh + optional redistribution. No remap, cadence or merge. Accepts and ignores `remesh_*`.
  - `retopologize_cylinder` (`cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py:176-280`): filtered Delaunay with quality 1e-4, connecting edges manually. It **does not touch `HC._simplices`**, so any stale cache still drives `compute_vd` and exact-volume selection.
  - `static_droplet_2D._dual_only_retopo` (`static_droplet_2D.py:114-123`) and the `dual_only_retopo*` copies in the diagnose scripts: `HC.boundary()` + `compute_vd` + `cache_dual_volumes` + `split_dual_volumes`. No refresh, no redistribution.
  - `cube2droplet/src/_setup.py:165-167`: a `**kwargs` wrapper that receives only `remesh_*`.
  - `retopologize_fn=False` for surface meshes (`ddgclib/operators/surface_tension.py:20`, `liquid_bridge_*`).

---

## 3. Method axes (proposed config schema)

**Implicit** means the value is currently chosen by something other than an explicit switch.

| # | axis | allowed values | default | status (lane logs / code) | how it is selected today |
|---|---|---|---|---|---|
| A1 | time integrator | `symplectic_euler`, `euler`, `rk45`, `euler_velocity_only`, `euler_adaptive(velocity_only=T/F)`, hand-rolled loops in cases | none in the library; `DynamicSimulation` uses `euler_velocity_only` | symplectic is used by every validated case. `euler_velocity_only` is for validation only. rk45 evaluates stages on stale duals | the function called |
| A2 | dt control | fixed (case-computed), adaptive advective CFL, scipy RK45 internal | fixed | the droplet runners compute acoustic+capillary dt (`oscillating_droplet_2D.py:103-110`) | case code / choice of integrator |
| A3 | connectivity policy | `delaunay` (per-step global), `dual_only` (`skip_triangulation=True`), `frozen` (`retopologize_fn=False`, no dual refresh), `adaptive` (2D), `gated_delaunay` (`displacement_eps`), `periodic_ghost`, `filtered_delaunay` (cylinder), custom | `delaunay` | 2D: `delaunay`+remap is the droplet default (laneE §2). 3D: `dual_only` is the default; 3D `delaunay_remap` was measured and rejected (laneE §1d, l2 1.87 vs 0.248). Plain Delaunay without remap is measured worse (lane5: l2 0.490). The gate at eps ∈ {0.01,0.05,0.2}·h_min was measured worse than either extreme (`_params.py:87-89`). Adaptive: upstream mass bug fixed (`INT:171-178`, `_setup.py:216-227`); opt-in, not scored as a default | spread across 3 switches: `retopologize_fn` (False/callable), `skip_triangulation` (integrator **or** partial), `remesh_mode` (integrator only, see §4.3), `displacement_eps`, `periodic_axes` |
| A4 | retopo remap | `None`, `'conservative'` | `None` | 2D validated (laneD §2; adopted laneE); dam break WIN (laneF §3); 3D DO-NOT (laneE §6.3); requires M=T and mps | partial on `_retopologize_multiphase` only |
| A5 | projection cadence | int N ≥ 1 | 1 | N>1 opt-in: 2D l2 0.0380-0.0419 but tail 1.38-1.40 (laneH §Verdict); not adopted. dual_only N=2/3 rejected (acoustic, laneH §1.2). 3D untested (laneH §6.6). Raw-recompute snapshot is a DO-NOT (laneH §6.2) | partial only; counter on `mps` |
| A6 | mass redistribution | off, single-phase (`pressure_model`+`redistribute_mass`), multiphase per-phase | library `False` (`INT:919,484`); droplet setup `True` (`src/_setup.py:42`); dam break / electrolysis via setup arg; **electrolysis fritz leaves it unbound → False** (`electrolysis_bubble_fritz_2D.py:769-772`) | noredist rings (lane5; laneH §1.3 l2 0.495) | integrator kwarg (single-phase) or partial/integrator (multiphase). Implicit: single vs multi is chosen by **which retopo fn** runs |
| A7 | per-phase dual split | `neighbour_count`, `exact` (2D polygon clip; 3D polyhedron `multiphase.py:468-475`) | `neighbour_count` | must match between setup and runtime (docs ×147 rho jump) | partial + setup arg, set separately in two places |
| A8 | dual construction | barycentric (hard-coded `INT:227`); simplex-aware vs 1-skeleton | barycentric / simplex-aware | — | **implicit**: `HC._simplices` present (`_compute_dual.py:115,159,320`) |
| A9 | dual-volume method | exact simplex (`simplex_dual_volumes` / `vertex_dual_volume`), batch fan-walk (3D), `dual_cell_area_2d` (2D), `v_star` sum (3D legacy), 1D interval | exact when available | 3D exact enabled 2026-07-29 (`INT:241-261`, `ST:387-403`); fan-walk undercounts 1-4% (same) | **implicit**: dim, `HC._simplices`, `HC._vd_method`, and whether `batch_e_star` raises. Three switch points that must stay in sync (`INT:263-269`, `ST:380,404`, `ST:448`) |
| A10 | boundary `dual_vol` convention | zeroed, half-cell | 3D zeroed; 2D/1D/periodic/fallback half-cell | **[measured]** §0.1 | **implicit**: dim (batch success). Affects `DualVolumeMass` and the mass-relaxation BCs (`dual_vol<1e-30` skip) |
| A11 | edge-area `A_ij` source | 3D `batch_e_star` cache (e_star), 3D p_ij (uncached), 3D e_star fallback per edge, 2D `v.vd` shared dual verts, 2D periodic min-image | 3D post-retopo: e_star cache; 2D: vd | **[measured]** §0.2: the 3D cache is not linearly precise | **implicit**: dim + existence of `HC._edge_area_cache` + `HC._periodic_axes`. `gradient.velocity_laplacian` always bypasses the cache (`gradient.py:91`) |
| A12 | boundary classification | `boundary_from_simplices`, `HC.boundary()`, carried `bV` (skip), periodic rule, custom; plus freeze filter `boundary_filter`; plus 3D failed-fan promotion | simplex-aware when cached | skip path + `boundary_filter` → filtered verts tagged interior for `compute_vd` (docs caveat) | **implicit**: `HC._simplices`, `skip_triangulation`, dim (promotion is 3D only). Under S=T a promoted vertex is carried in `bV` indefinitely (`INT:219,238-240,288`; inference) |
| A13 | pressure / EOS update site | single-phase: inside `dudt_fn` via `_resolve_pressure` (`None`/callable/EOS, `ST:701-740`); multiphase: inside retopo (`mps.refresh`→`compute_phase_pressures`) with the `vol_corr` gauge; the force reads `v.p_phase` (`multiphase_stress.py:163-164`) | — | with `retopologize_fn=False` or a gated skip, multiphase `p_phase` is **never updated**, while single-phase EOS still re-reads (stale) `dual_vol` | **implicit**: which `dudt_fn` + which retopo fn. The EOS is specified twice for single-phase (integrator `pressure_model` for redistribution vs `dudt` partial for force) |
| A14 | interface curvature path | `integrated`, `stokes`, `csf_dual` (`multiphase_stress.py:287-309`) | `integrated` | `stokes`==`integrated` to 1e-14 in 3D (laneG, cited in `oscillating_droplet_3D.py:83-84`); `csf_dual` experimental | `dudt_fn` partial. The docstring lists 2 of the 3 values (`:248`) |
| A15 | viscous form | diffusion form (transpose omitted) | fixed | — | hard-coded (`ST:752-766`) |
| A16 | merge | `merge_cdist` float | None | not mass-conserving; `mass_conserving_merge` is the separate path | integrator kwarg |
| A17 | parallel / backend | `workers` int; `backend` (batch_e_star only) | None | workers loses EOS side effects (§0.5) | integrator kwargs |
| A18 | periodicity | `periodic_axes`, `domain_bounds` | None | multiphase needs the case wrapper (no remap/cadence) | integrator kwarg (single-phase) / case wrapper (multiphase) |
| A19 | skip gate | `displacement_eps` | None | measured worse (A3); first call always skips | integrator kwarg |

---

## 4. Kwarg plumbing problems

1. **The laneF fix is complete for the 8 kwargs it names** (`INT:448-457`), for callables that declare them by name. Remaining holes:
   1. **`remesh_mode`/`remesh_kwargs` precedence is inverted.** See §0.3 (`INT:427-430`).
   2. **`**kwargs` sinks receive only `remesh_*`.** `cube2droplet/src/_setup.py:165-167` forwards its `**kwargs` into `_retopologize_multiphase`, but an integrator-level `skip_triangulation`, `boundary_filter`, `merge_cdist`, `backend` or `redistribute_mass` never reaches it. Silent.
   3. **`_retopologize_multiphase` does not declare `periodic_axes`, `domain_bounds` or `pressure_model`.** Integrator-level periodicity is silently ignored for multiphase.
   4. **`retopo_remap`, `projection_every` and `split_method` are partial-only.** Passing one to an integrator lands in `**dudt_kwargs`. That raises TypeError for `stress`/`multiphase_dudt_i` (laneF §1), but it is **silently swallowed** by `dudt_fn(v, **_kw)` wrappers (`electrolysis_bubble/src/_setup.py:464`, `electrolysis_bubble_fritz_2D.py:761`).
   5. An **unbound** declared kwarg now picks up the integrator's value, including its default. This is harmless today because the defaults are identical (`INT:919` vs `:484`, etc.), but the effective M for `electrolysis_bubble_fritz_2D.py:769-772` is whatever the integrator passes (default False).
2. **Periodic branch drops switches** (`INT:125-132`): `skip_triangulation`, `remesh_*`, `pressure_model`, `redistribute_mass` and `backend` are ignored. The periodic path never clears `HC._edge_area_cache`.
3. **`backend` names two different things**: the batch_e_star compute backend (`INT:235`) and the periodic Delaunay backend (`periodic.py:353`). `compute_vd` gets neither (`INT:227`).
4. **`pressure_model` is specified twice** for single-phase (integrator vs `dudt_fn` partial). Nothing checks that they agree.
5. **Silent no-ops instead of errors**:
   - `redistribute_mass=True` with `pressure_model=None` (single-phase, `INT:142,291`).
   - `redistribute_mass=True` with `mps=None`.
   - `retopo_remap='conservative'` with `mps=None` (`INT:597-598`).
   - `retopo_remap='conservative'` with `skip_triangulation=True` (documented, but silent).
6. **DynamicSimulation holes** (§0.4, §1.3): it overrides the partial-bound `mu`, forwards only `skip_triangulation`, leaks `extra` into integrator kwargs, ignores `t_end` for fixed-step integrators and never uses `rho`.
7. **Case policy names diverge**:
   - 2D uses `'dual_only' | 'delaunay_remap'`, with a silent fall-through.
   - 3D CLI uses `'delaunay' | 'dual_only'` (no remap option).
   - Dam break 3D uses the integrator kwarg `skip_triangulation=True` (`dam_break_3D.py:112`).
   - Dam break 2D uses the partial remap (`dam_break_2D.py:121`).
   - The droplet runners get the `split_method`/`redistribute_mass` partial from setup (`src/_setup.py:230-232`).
   - `params['remesh_mode']` is hard-coded to `'delaunay'` in setup (`src/_setup.py:242`) and must be re-passed at integrator level (`oscillating_droplet_2D.py:152-156`).

---

## 5. Dead, duplicated or suspicious code

- `INT:257-261` NOTE(lane3) text about 2D batch_e_star volumes and "zeroing in both paths" is incorrect (§0.1). The `dim == 3` guard at `INT:263` is redundant, because batch_e_star only succeeds in 3D.
- `INT:459-460` VAR_KEYWORD check on a named parameter can never be true.
- `INT:115-122` `_retopologize` "Steps" docstring is stale (no 5b, filter, redistribution or adaptive). `euler` docstring (`:932-975`) omits `workers`, `merge_cdist`, `backend`, `periodic_*`, `pressure_model`, `redistribute_mass` and `displacement_eps`.
- Five copies of the same `_do_retopologize` call (§1.1). `euler` update logic is duplicated in `euler_adaptive` (`:998-1006` vs `:1434-1441`).
- `_recompute_duals` (`INT:740-749`) is not used by the library. It is a partial refresh (no `dual_vol` or cache).
- `ddgclib/operators/stress_pointwise.py` (391 L) is marked "ARCHIVED... not used in the production pipeline" (lines 1-5). It has **zero importers** in `.py`/`.ipynb` and is referenced only by `Fundamentals_pointwise.md`, `Fundamentals_old.md` and `docs_temp/*`. It is a removal candidate; I have not deleted it.
- `gradient.py`: `velocity_laplacian` bypasses the cache (`:91`). `acceleration` has no `pressure_model` passthrough (`:104-135`). `pressure_gradient` returns a force, not a gradient (`:59`).
- `ST` TODO markers:
  - constitutive TODOs (`ST:32-41`);
  - "move dual_area_vector/dual_volume to hyperct" (`ST:49,89`);
  - the periodic `compute_vd` bug workaround (`ST:98-101`), with no 3D counterpart.
- NOTE(lane…) markers and what they gate:
  - `NOTE(lane3-dual-volume)` `INT:241`, `ST:387`: the 3D exact-volume switch (dim==3 plus `_use_exact_barycentric_volume`).
  - `NOTE(laneF-forward)` `INT:431,441`: partial-precedence and by-name forwarding.
  - `NOTE(laneH)` `INT:558`, `:657`: `projection_every` semantics and the strain-advanced snapshot.
  - The lane4 comment `INT:171-178`: the 2D simplex-cache rebuild after adaptive remesh.

---

## 6. What a case author has to decide today

**(a) Single-phase.**
1. **Mesh / builder** (`geometry.domains`). Setup must run `compute_vd` + `cache_dual_volumes` if `retopologize_fn=False` or `displacement_eps` is used.
2. **Mass/pressure IC** (`initial_conditions`).
3. **`dudt_fn`** = `partial(dudt_i, dim, mu, HC, pressure_model=EOS|None|callable)`. This fixes the force operator and the EOS read.
4. **Integrator + dt** (case code).
5. **`bc_set`**, plus `boundary_filter` (integrator kwarg).
6. **Connectivity**: `retopologize_fn` (None/False/custom), `skip_triangulation`, `merge_cdist`, `periodic_axes`/`domain_bounds`, `displacement_eps`, `remesh_mode` (all integrator kwargs).
7. **Redistribution**: `redistribute_mass` + a *second* `pressure_model` (integrator kwargs).

**(b) Multiphase 2D** (droplet pattern).
1. **Setup helper** (`cases_dynamic/*/src/_setup.py`) decides:
   - the EOS (TaitMurnaghan with `rho_clip`);
   - `split_method` (setup + partial);
   - the `redistribute_mass` partial binding;
   - the YL / hydrostatic mass preload;
   - `dudt_fn = partial(multiphase_dudt_i, dim, mps, HC, pressure_model=MultiphaseEOS[, curvature_path])`, plus a gravity wrapper in the case.
2. **Case `_params.py`** decides the retopology policy (`retopo_policy_2d`).
3. **The runner** turns that policy into a partial: `skip_triangulation=True` or `retopo_remap='conservative'` (`projection_every` has to be added by hand; no runner exposes it).
4. **Integrator kwargs** decide `remesh_mode`/`remesh_kwargs` (they must go at integrator level) and dt. `displacement_eps` is available but measured worse.

**(c) Multiphase 3D.** Same places as (b), with these differences:
- Policy `retopo_policy_3d='dual_only'`; the remap is a DO-NOT.
- Adaptive remesh is unavailable (raises).
- The dual-volume method (exact) and the edge-area source (e_star cache, §0.2) are chosen **implicitly** by dim + `HC._simplices`.
- Boundary cells are zeroed and failed fans are promoted automatically.
- The dam break 3D sets frozen connectivity through the integrator kwarg instead of the partial.

**Decisions no switch controls today** (candidates for explicit registry fields): A8–A12 (dual construction, volume method, boundary volume convention, `A_ij` source, boundary rule) and A13 (EOS update site).
