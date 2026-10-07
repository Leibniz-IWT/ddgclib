# Solver Pipeline: Per-Timestep Loop, Retopologization, BCs, Timestep Control
> Sources: code_map/integrators_and_bcs.md, code_map/operators_stress.md, sources/debugging_plan_distilled.md, sources/development_status.md, code_map/cases_dynamic_inventory.md, sources/library_audit_and_features.md | Written: 2026-07-02 by understand-and-document workflow

Code: `ddgclib/dynamic_integrators/_integrators_dynamic.py` (1212 L) + `_simulation.py`. Line-level map: [code_map/integrators_and_bcs.md](code_map/integrators_and_bcs.md). All five integrators (`euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`) share the same per-step skeleton; only the state-update rule differs.

---

## 1. Universal per-step order (stability-critical, all integrators)

```
1. _do_retopologize(...)          # rebuild topology + duals + caches + bV   (:722 in euler)
2. verts = _interior_verts(HC,bV) # [v for v in HC.V if v not in bV]        (:531-533)
3. accel = _compute_accel(...)    # ALL accelerations before ANY move       (:543-588)
   └─ pressure/EOS resolved INSIDE dudt_fn (stress.py:634 _resolve_pressure;
      EOS writes v.p, v.rho in place from rho = m/dual_vol)
4. state update (scheme-specific, §2); position via _move(v,pos,HC,bV) (:521-528)
   which pops/re-adds v from bV around HC.V.move (hash changes on move)
5. _apply_bc_set(bc_set, HC, bV, dt)   # AFTER the move, insertion order    (:536-540)
6. t += dt; callback(step,t,HC[,bV,diagnostics]); _maybe_save_state(...)
```

Consequences (the load-bearing ordering facts):
- **Forces always see fresh duals** (retopo happens immediately before the force pass; nothing moves mid-pass) — except inside `rk45` RK stages (§2).
- **BC mutations take effect one step late**: BCs run post-move; fields they set (velocity zeroing, mass relaxation, injected/deleted vertices) are only seen by the next step's force pass, after the next retopo rebuilds duals.
- **BCs run with `v.dual_vol` one advection stale** (dual volumes are from step 1, positions from step 4).
- Topology-mutating BCs (`OutletDeleteBC`, `OutletBufferedDeleteBC`, `PeriodicInletBC`) call `invalidate_simplex_cache`; duals stay stale until the next step's retopo.

## 2. Update schemes per integrator

| Integrator | Update rule | Notes |
|---|---|---|
| `euler` (:652-751) | `x += dt*u_OLD; u += dt*a` (:738-739) | buffered: all updates computed, then applied |
| `symplectic_euler` (:756-841) | `u += dt*a` FIRST, then `x += dt*u_NEW` (:828-829) | **primary integrator**; conserves modified Hamiltonian |
| `rk45` (:846-986) | scipy `solve_ivp(RK45)` per macro-dt on `y=[x…,u…]`, `rtol=1e-6, atol=1e-9` | retopo ONCE at macro-step start; `_sync_mesh` moves vertices per RK stage but **duals + `HC._edge_area_cache` are NOT rebuilt mid-step** → intermediate stages evaluate forces on stale duals. `sol.success==False` → RuntimeError |
| `euler_velocity_only` (:991-1063) | `u += dt*a` only, no position update | Eulerian, **validation only**; pass `skip_triangulation=True` on fixed meshes |
| `euler_adaptive` (:1068-1212) | `velocity_only=True` by DEFAULT; with False it is forward-Euler (old-u positions), NOT symplectic | signature: `dt_initial`, `t_end` (no `n_steps`) |

## 3. `_retopologize` (:48-247) — the per-step topology/dual rebuild

Executed order:

0. `periodic_axes` set → delegate entirely to `ddgclib.geometry.periodic.retopologize_periodic` and return. Early return (duals left STALE) if `len(HC.V) < dim+1` (:136-137).
1. If `redistribute_mass` + `pressure_model`: snapshot pressure field.
2. Connectivity, unless `skip_triangulation=True`:
   - `merge_cdist > 0` → `HC.V.merge_all(cdist)` (NOT mass-conserving; use `mass_conserving_merge` first if needed) + `bV.intersection_update(set(HC.V))`.
   - `remesh_mode='delaunay'` (default): disconnect ALL edges → `hyperct.ddg.connect_and_cache_simplices` (global scipy Delaunay, populates `HC._simplices`) → boundary via `boundary_from_simplices`. **Delaunay creates cross-phase edges at sharp interfaces** — the documented interface-destroying instability.
   - `remesh_mode='adaptive'`: `hyperct.remesh.adaptive_remesh(HC, dim=dim, **remesh_kwargs)` — interface-preserving local split/collapse/flip + smoothing, **2D only**; then `invalidate_simplex_cache`, boundary via `HC.boundary()`. **Currently blocked in production** by the upstream mass-averaging bug (`hyperct/remesh/_operations_2d.py:198`, ~164% mass loss by step 80 on the oscillating droplet).
   - `skip_triangulation=True`: keep connectivity, `dV = set(bV)`. **Caveat**: if `bV` was previously narrowed by `boundary_filter`, open-boundary (inlet/outlet) vertices get `v.boundary=False` and `compute_vd` treats them as interior → possibly malformed half-cells.
3. Tag `v.boundary = v in dV` on ALL vertices (full topological boundary, before filtering).
4. `compute_vd(HC, method="barycentric")` — rebuild all `v.vd`.
5. Cache geometry: preferred `hyperct.ddg.batch_e_star(interior, HC, dim, backend, orient=True, compute_volumes=True)` → `v.dual_vol` + `HC._edge_area_cache` (keyed by `id(v)` — stale if vertex objects are replaced without a rebuild); vertices whose fan-walk fails are promoted to boundary; **all boundary vertices get `v.dual_vol = 0.0`** (:225). Fallback: `stress.cache_dual_volumes` (half-cell path, boundary verts get real half-volumes) and `HC._edge_area_cache = None`.
6. `boundary_filter` semantics (:232-238): `dV = {v for v in dV if filter(v)}`; then **`bV.clear(); bV.update(dV)` — `bV` is wholly rewritten every step**. Hand-curated `bV` from setup is discarded; only filtered (typically wall) vertices stay frozen; inlet/outlet topological-boundary vertices advect. Use `boundary_filter` or `PositionalNoSlipWallBC` to keep walls frozen. Note `v.boundary=True` persists on filtered-out vertices until the next retopo (intended: `compute_vd` needs the full boundary).
7. `redistribute_mass` + `pressure_model` → `redistribute_mass_single_phase(...)` — **mutates `v.m`** so the pre-retriangulation pressure field is preserved (needs `pressure_model.density(P)` inverse).

### `_do_retopologize` dispatch (:289-388)
- `retopologize_fn=False` → skip all topology management (frozen-connectivity runs; duals must have been built at setup).
- **Displacement gate** `displacement_eps` (:250-279): skip retopo when every vertex moved < eps and the vertex id-set is unchanged. **The very first call always snapshots and SKIPS** — setup MUST have run `compute_vd` or step 1 crashes. Suggested `eps = 1e-4 * h_min`. Rationale: 3D Delaunay non-uniqueness on near-cospherical interface clouds flips ~48 cross-phase edges per static retopo, shifting `dual_vol` against frozen mass → EOS misreads it as compression (the historical 3D blow-up; see [sources/debugging_plan_distilled.md](sources/debugging_plan_distilled.md)).
- Custom callable → `retopologize_fn(HC, bV, dim)`; `remesh_mode/remesh_kwargs` forwarded only if the signature accepts them; other kwargs are NOT forwarded.

### `_retopologize_multiphase` (:391-475) — bind via `retopologize_fn=partial(...)`
1. `snapshot_geometry_multiphase` (pre-retopo `dual_vol_phase`; the guard on **snapshotted** volume — not `p_phase < 1e-30` — was the 2026-04-29 fix that cut the 3D static floor ×195). 2. `_retopologize(...)`. 3. `mps.refresh(HC, dim, reset_mass=False, split_method=...)` — geometry + per-phase pressures recomputed; Lagrangian `v.m`/`v.m_phase` preserved. **Setup and runtime `split_method` MUST match** (mismatch → ×147 rho jump). 4. `redistribute_mass_multiphase` then `mps.compute_phase_pressures` — the only place pressures are recomputed outside `dudt_fn`. `redistribute_mass=True` is the production default for multiphase.

## 4. Boundary conditions (application order & semantics)

`BoundaryConditionSet.apply_all(mesh, bV, dt)` applies BCs in **insertion order**, once per step, post-move. A BC added with `vertices=None` targets the live (filtered, rewritten-each-step) `bV`; a BC added with an explicit vertex set holds stale references if those vertices are deleted/merged.

Key classes (`ddgclib/_boundary_conditions.py`; full table in [code_map/integrators_and_bcs.md](code_map/integrators_and_bcs.md)):
- Walls: `NoSlipWallBC` (u=0), `MovingWallBC`, `ShearingPlateBC` (translates + clamps normal coord), `PositionalNoSlipWallBC` (**canonical wall re-tagging** — rescans all vertices each step, sets u=0, `v.boundary=True`, adds to `bV`; catches freshly injected vertices).
- Flow-through: `PeriodicInletBC` (ghost-mesh injection + `merge_all` every apply), `OutletDeleteBC` (delete past `outlet_pos`, optional `backflow_clamp`), `OutletBufferedDeleteBC` (ghost buffer freezes u and **overrides the integrator's position update** for buffer vertices — keeps outlet duals complete, no backflow).
- Mass-relaxation (intentionally non-conservative, first-order relaxation on `v.m`): `PressureReservoirBC`, `AbsorbingPressureBC`, `ExpandingDomainBC`. **Trap**: they skip vertices with `dual_vol < 1e-30` — after a `batch_e_star` retopo ALL boundary vertices have `dual_vol=0.0`, so these BCs only act on interior vertices; target explicit sets near (not on) walls.
- Dirichlet/Neumann: `DirichletVelocityBC`, `DirichletPressureBC`, `NeumannBC` (copies from nearest interior neighbour).

## 5. Timestep control

- `euler_adaptive`: CFL control runs **AFTER** each step (new dt applies to the next step): `dt = clip(cfl_target * h_min / u_max, dt_min, dt_max)`; defaults `cfl_target=0.5, dt_min=1e-12, dt_max=dt_initial`; `u_max<=1e-30` → `dt=dt_max`. O(E) edge scan per step.
- `rk45`: scipy-internal adaptive substepping within each macro `dt` (`rtol=1e-6, atol=1e-9`); no ddgclib CFL logic; duals stale within the macro step.
- Fixed-dt integrators: cases pick dt from the acoustic CFL of the (softened) EOS sound speed — e.g. electrolysis_bubble uses `cfl_safety=0.05` with matched `K=1e5` on both phases; the oscillating droplet softens water to c_s≈1 m/s. See [code_map/cases_dynamic_inventory.md](code_map/cases_dynamic_inventory.md).
- `DynamicSimulation.run()` swaps `dt` → `dt_initial` + `t_end = p.t_end or p.n_steps*p.dt` when the integrator **is** `euler_adaptive` (identity check).

## 6. Dual/cache freshness table (what is fresh vs stale, where)

| Moment | Duals / caches |
|---|---|
| Force pass (euler, symplectic, velocity_only, adaptive) | FRESH (rebuilt this step) |
| Inside rk45 RK stages | STALE (macro-step-start geometry) |
| BC application | positions fresh, `dual_vol` one step stale |
| Steps skipped by `displacement_eps` (incl. ALWAYS the first call) | reused from setup/last rebuild |
| After `_retopologize` early-return (< dim+1 verts) | STALE, silently |
| After BC vertex deletion/injection | STALE until next step's retopo |
| `HC._edge_area_cache` | keyed by `id(v)` — invalid after any vertex-object replacement without rebuild |
| `_recompute_duals(HC)` helper | rebuilds `v.vd` ONLY — does NOT refresh `dual_vol` or edge-area cache |

## 7. Stability-critical constraints & known instabilities (checklist)

Ordering/setup constraints:
1. **Always `functools.partial(dudt_i, dim=, mu=, HC=)`** — never via `**dudt_kwargs` ("multiple values for HC"); the module docstring examples at `_integrators_dynamic.py:24-25, :35-36` are broken (TypeError). `DynamicSimulation` additionally swallows `dim`/`HC` (they collide with integrator params) and never forwards `params.rho`.
2. Setup must run `compute_vd` (+ volume cache) before integrating whenever `displacement_eps` or `retopologize_fn=False` is used — the first step will not build duals.
3. `DualVolumeMass` IC must be applied after a setup using the `cache_dual_volumes` half-cell path — after an integrator retopo, boundary verts have `dual_vol=0.0` → `m = rho*1e-30`.
4. Multiphase: `split_method` consistent between setup and runtime; `redistribute_mass=True`; per-phase state must be rebuilt (`mps.refresh(reset_mass=False)`) after any `mass_conserving_merge`.
5. Surface meshes (thin-film surface tension): `retopologize_fn=False`.

Known open instabilities (status as of 2026-07-02; details in [sources/debugging_plan_distilled.md](sources/debugging_plan_distilled.md) and [sources/development_status.md](sources/development_status.md)):
- **Static droplet under full Delaunay retopo**: KE grows 0 → 4.4e-3 in 100 steps (`cases_dynamic/oscillating_droplet/static_droplet_2D.py`, fastest failing repro) while the fixed-connectivity diagnostic is stable over 5000 steps. Diagnosis: Delaunay-flip dual-volume jumps misread by the EOS as physical compression. Static max|F| floors are regression-pinned (2D peak 2.3749e-3 / steady 2.2717e-3; 3D steady 7.3768e-05; `test_a5b_longrun_regression.py`).
- **2D oscillating droplet dynamic validation FAILS spec**: tail_growth 1.68–1.85 vs target <1.0, l2_error ~5.6 vs <0.2. Gated on the Probe-5 skip-retopology gate (`displacement_eps`) and/or adaptive remesh.
- **Adaptive remesh mass bug** (upstream `hyperct/remesh/_operations_2d.py:198`, split averages mass): blocks `remesh_mode='adaptive'` in production.
- **Single-phase retopology volume leak** ~2-4% via the dual-volume refresh (both `skip_triangulation` and full Delaunay); frozen meshes conserve to machine precision.
- **Contradictory retopo guidance across cases**: dual-only retopo (skip_triangulation) is stable for droplets but LESS stable for electrolysis_bubble with gravity; dam_break survives only with `alpha_art=2.0` + frozen connectivity + 0.02 s horizon.
- One-shot `|dV/V0|≈0.3` after the first 3D retopo is a boundary-shell `dual_vol` zeroing artefact — never assert 3D volume conservation from step 0.

## 8. Recording / outputs

`StateHistory(fields=('u','p'[,'phase','is_interface']), record_every=N | record_every_t=..., save_dir=..., conservation=True)` → pass `history.callback` as integrator `callback`. Snapshots auto-saved as `state_{idx:06d}_t{t:.6f}.json`; `load_state` does NOT persist duals (recompute `compute_vd`) and infers domain bounds from the vertex bbox. Animations: `dynamic_plot_fluid(history, HC, save_path=..., phase_field='phase', interface_field='is_interface')`; 3D replay `python -m ddgclib.scripts.view_polyscope --snapshots <dir>`. Outputs always in `cases_dynamic/<name>/{fig,results/snapshots}`.
