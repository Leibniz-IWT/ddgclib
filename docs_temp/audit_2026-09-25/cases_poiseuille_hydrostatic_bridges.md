# Solver configuration inventory: Hagen_Poiseuile*, Hydrostatic_column, bc_demo, cube_flow, template, liquid_bridge_*

Audit date 2026-09-25 (sub-report of `bcs_and_cases.md`). Paths relative to
`/home/endres/projects/ddgclib/`; `_int.py` = `ddgclib/dynamic_integrators/_integrators_dynamic.py`,
`_bc.py` = `ddgclib/_boundary_conditions.py`, `_ic.py` = `ddgclib/initial_conditions.py`.
No repo files were modified; probes ran in the scratchpad only.

## 0. Shared library semantics

- Step order in `euler`/`symplectic_euler`/`euler_velocity_only`: `_do_retopologize` first (`_int.py:984,1074,1305`), then accelerations on `HC.V minus bV` (`:793-795`), then `bc_set.apply_all` (`:1008,1098,1320`).
- Default `retopologize_fn=None` runs `_retopologize`: full disconnect + Delaunay every step (`:191-207`), boundary from the simplex cache (`:209-215`), `compute_vd` (`:227`), `batch_e_star` volumes (`:232-276`), then **bV is cleared and rebuilt** as the topological hull, narrowed by `boundary_filter` if given (`:285-288`).
- A callable `retopologize_fn` receives `remesh_mode/remesh_kwargs` only if it declares them; other retopo kwargs only if declared by name and not partial-bound (`:414-464`). `False` skips retopology entirely (`:407-408`).
- `dudt_i` = `stress_acceleration`: `pressure_model=None` means `v.p` is read as-is and never re-evaluated (`stress.py:805-813`); no gravity term.
- `LinearPressureGradient`/`HydrostaticPressure` use the volume average only if `v.vd` **and** `v.dual_vol` exist, else `v.p = P(x_vertex)` (`_ic.py:140-146,102-108`). `compute_vd` never sets `dual_vol` (only `cache_dual_volumes`/`_retopologize` do), so every runner below that applies the IC after a bare `compute_vd` takes the point-value branch.
- `PositionalNoSlipWallBC`: scans all vertices each step, sets `u=0`, `v.boundary=True`, adds to bV (`_bc.py:726-735`). `OutletDeleteBC`: deletes all vertices with `x[axis] >= outlet_pos` incl. walls (`:426-430`). `OutletBufferedDeleteBC`: entry at `x > outlet_pos` freezes `u` (`:512-514`), re-moves the vertex at frozen `u` every step (`:524-532`), deletes at `outlet_pos+buffer_width` (`:497-504`). `PeriodicInletBC`: ghost shifted to `[inlet-period, inlet]` (`:635-641`), advanced `U*dt` (`:647-650`), vertices with `x > inlet` created via `mesh.V[tuple]` (a cache hit overwrites the fields of an existing vertex at that exact key, `:658-667`), ghost re-cloned when empty (`:685-687`), then global `merge_all(cdist)` **without updating bV** (`:689`; mass of the removed twin is dropped, `hyperct/_vertex.py:454-466`).

## 1. Hagen_Poiseuile/

### 1.1 `Hagen_Poiseuile_2D.py` (Lagrangian HP2D)
| Row | Value |
|---|---|
| domain | manual `Complex(2,[(0,1),(0,D=1)])`, `refine_all` x1 (`n_refine=1`, `:108-114`), then `extrude(HC_unit, L=15, axis=0, cdist=1e-10)` (`:120`); `bV = HC.boundary(HC.V)` (`:134`), `compute_vd` (`:140`). Blocking `print(help(HC.plot_complex))` + `HC.plot_complex()` at `:121-122` and `:276` |
| integrator | `symplectic_euler(dt=0.01, n_steps=3000, dim=2, bc_set, boundary_filter=wall_criterion, callback, save_every=500, workers=20)` (`:277-286`); default Delaunay retopo, no merge |
| dudt | `partial(dudt_i, dim=2, mu=1e-3, HC=HC)` (`:262`), no pressure_model, no gravity |
| BCs (order) | `PositionalNoSlipWallBC(wall_criterion |y|<1e-10 or |y-1|<1e-10, bV)`; `OutletBufferedDeleteBC(outlet_pos=15, buffer_width=2.0, axis=0, bV)` (hard-coded 2.0, not `outlet_buffer=0.5`); `PeriodicInletBC(unit_mesh, velocity=0.1, axis=0, inlet_pos=0, cdist=1e-10, period=1.0)` (`:201-224`) |
| ICs | `UniformVelocity([0.1,0]) + LinearPressureGradient(G=3.2e-3) + UniformMass(15)` (`:236-241`); unit mesh same with `unit_bV` computed from the MAIN mesh (`:190-194`). Point-value `v.p` (no `dual_vol` at IC time) |
| status | **unstable / not validated**. Docstring is a pasted TODO calling the inlet accumulation "the core bug" (`:22-51`). `results/hp2d_final_state.json` (2026-02-22, pre-buffered-outlet): 158 vertices, 14 at y<0, 45 at y>1, 20 at x>15, only 2 frozen wall vertices left. No post-switch artefact exists |

### 1.2/1.3 `test_outlet_old_bc.py` / `test_outlet_new_bc.py`
Same mesh; `symplectic_euler(dt=0.01, n_steps=200, boundary_filter=wall_criterion)`; `PositionalNoSlipWallBC` + `OutletDeleteBC(15.5, backflow_clamp=2.0)` (old) / `OutletBufferedDeleteBC(15, 2.0)` (new); **no inlet**; point-value p; diagnostic prints only. Run as scripts, not collected by pytest.

### 1.4 `src/_setup.py::setup_poiseuille_2d`
Used by `ddgclib/tests/test_stress.py` fixtures only: `Complex(2)` refine x2; `PoiseuillePlanar(G=1, mu=1) + LinearPressureGradient(G=1) + UniformMass`; `NoSlipWallBC` on walls only (`DirichletVelocityBC` imported, never added, contrary to the docstring `:91`). **validated** (median residual < 1e-13).

### 1.5 dead helpers
`src/_params.py` `refinements=3`, `inlet_layer_thickness`, `outlet_buffer`, `add_inlet_every`, `CFL_target` are unused (runner hard-codes `n_refine=1`); `src/_boundary_conditions.py` legacy 3D helpers have no importer; `src/_mass.py:34` calls `set_mass_3d` without the required `r, L` (would raise).

## 2. `Hagen_Poiseuile_2D_Eulerian.py`
`Complex(2,[(0,2),(0,1)])` refine x3; `euler_velocity_only(dt=0.005, n_steps=2000, bc_set)` with default retopo and `boundary_filter=None` (inlet/outlet columns frozen at u=0 the whole run); `partial(dudt_i, mu=0.1)`, G=1; `NoSlipWallBC` on walls; point-value `LinearPressureGradient`; point-wise velocity comparison at x=L/2. Runs (figures 2026-02-17); no pins.

## 3. `Hagen_Poiseuile_3D/`
| Row | Value |
|---|---|
| domain | `cylinder_volume(R=0.5, L=15, refinement=2, flow_axis=2)` (`:297`); `compute_vd` only after ICs (`:674`) |
| integrator | `symplectic_euler(dt=0.01, n_steps=3000, dim=3, bc_set, retopologize_fn=retopologize_cylinder, save_every=500, workers=8)` (`:444-453`); `boundary_filter`, `merge_cdist`, `skip_triangulation`, `remesh_mode`, `backend` not passed |
| dudt | `partial(dudt_i, dim=3, mu=0.01, HC)`; G = 0.032 |
| retopo_fn | `retopologize_cylinder(HC, bV, dim)` (`:176-283`): `merge_all(1e-9)`, disconnect + scipy Delaunay, sliver filter `vol/max_edge^3 > 1e-4`, **`HC.boundary()` with no filter (every hull vertex frozen incl. inlet/outlet caps and flag-complex false positives)**, `compute_vd(backend)`, `batch_e_star` + failed-fan promotion; never updates `HC._simplices`. The geometric `boundary_criterion` (`:160-173`) is dead; the comment at `:440-443` claiming geometric detection is wrong |
| BCs | `PositionalNoSlipWallBC(r>=R-1e-8)`; `OutletBufferedDeleteBC(15, 2.0, axis=2)`; `PeriodicInletBC(unit cylinder, 0.1, axis=2, cdist=1e-10, period=1.0)` (`:354-379`) |
| ICs | `UniformVelocity + LinearPressureGradient(G, axis=2) + UniformMass` applied BEFORE `compute_vd`: point-value p |
| status | **stalled**. `run.log` (2026-03-25): 2485 vertices, 994 boundary, 3000 steps, then "No interior vertices near midpoint". `results/hp3d_final_state.json` (t=1.0, 407 vertices): 305 frozen, 48 of them strictly inside the tube (spurious hull freezing); 9 wall vertices at z=0 plus 7 at z=0.001 (duplicate ring, M1) |

`run_cluster.py` wraps the same physics (`--backend gpu` default); no cluster log in the repo.

## 4. `Hagen_Poiseuile_equilibrium_2D.py`
Static residual evaluation (no integrator) at refinements 1..5; point-value ICs applied before `compute_vd`; **validated** (median residual < 1e-13 in `test_stress.py:968-1022`). Companion `test_equilibrium.py` is outside `ddgclib/tests/` and not collected.

## 5. Hydrostatic_column/ (see also `bcs_and_cases.md`)
All four runners: hand-written symplectic loop, `_recompute_duals` + `cache_dual_volumes` each step, no retopology, `mu_art = 0.5 rho c0 dx`, `TaitMurnaghan(n=1, rho_clip=(0.5,2))`; `HydrostaticPressure` volume-averaged after `compute_vd` but Section 2b `HydrostaticEOSMass` writes point `v.p`; 3D "integrated" error is point x vol. Status: 1D still decaying at 200 t_ac; **2D unstable** (KE grows from ~2.5 t_ac, |u| ~ 310 m/s ~ 10 c0 at ~6.3 t_ac); 3D plateaus; **2D_periodic aborted** (|u| > 2 c0 near 6.8 t_ac; it is not periodic but reflecting, side vertices drift by `dt u_x` before `u_x` is zeroed).

## 6. bc_demo/ (kinematic only)
`bc_demo.py`: `Complex(2,[(0,3),(0,1)])` refine x1, non-wall `x += U dt` (U=0.5, dt=0.05, 120 steps), `PositionalNoSlipWallBC -> OutletDeleteBC(3.0) -> PeriodicInletBC(cdist auto 0.177)`. Probe: outlet corners deleted on step 1 (wall set 6 -> 4), interior 7 -> 19 (unit-mesh density fills the coarse main mesh). v1/v2/parametric use case-local BC functions (purge injected wall vertices, raise on wall-count drop); counts constant in the probe. All three overwrite the same `fig/bc_demo.gif`.

## 7. cube_flow/ 1D/2D/3D
`Complex(dim)` refine x1 for main and unit mesh; `euler(dt=0.01, n_steps=400, mu=0.1, G=0.001)` with a mock dudt (`G/v.m + mu (mean(u_nn) - u)`, dimensionally wrong); `OutletDeleteBC(1, axis 0)` **without bV**, `PeriodicInletBC(0.05, cdist auto)`; `static_walls=False` but every hull vertex is frozen by the default retopo; point-value p. Probe: **stalled** after step 0 (1D: 1 deleted, 1 injected, xmax stuck 0.7505; 2D: 3 deleted, n=10 thereafter, 7-8 of 10 frozen): the next column becomes hull and is frozen, ghost vertices merge into the frozen inlet column at cdist 0.177, the next ghost column needs 500 steps > 400. DEVELOPMENT.md "1D and 2D videos verified" is stale.

## 8. template/
`template.py`: mock diffusion dudt with no pressure drive ("Poiseuille" that never moves); `example_features_demo.py`: point-value ICs, Example 5 "Adaptive Euler with CFL" runs velocity-only. Demos only.

## 9. liquid_bridge_approach/
Every solver runner fails at import (`ImportError` on `multiphase_sparse_compressible_eos_pressure_correction`, `_ddgclib_case_core.py:156-161`; the function exists only on branch `origin/songyideng/liquid-bridge-separation-12cases`). When runnable: custom semi-implicit loop (`core:9532-9675`); the `symplectic_euler(..., retopologize_fn=False)` block at `core:14696-14736` is dead code; Chorin tet projection in the limit branch so K = 9.7e8 is never used; acceleration clipped to 5e-3 m/s^2 and displacement to 2 um per step; inertial density effectively 1 kg/m^3 vs hydrostatic 965; Cox term never affects dynamics; eight functions defined twice. README force formula, gorge pressure (frozen at t0), undocumented D/R blend and smooth-curve claims do not match the code.

## 10. liquid_bridge_dem/, liquid_bridge_cfd_dem/
`liquid_bridge_dem`: pure DEM (`dem_step` with `HertzContact(E=70e9, nu=0.22)` + `LiquidBridgeManager(gamma=0.072)`), dt 1e-6 x 100000; **runs** (~7 s): bridge forms at step 30001, max|F_cap| = 3.887e-4 N (README says "~0.39 uN"), undamped oscillation, contact never engages, `check_ruptures()` never called.
`liquid_bridge_cfd_dem`: two spherical-cap films merged into `Complex(3)`; `symplectic_euler(dt=1e-7, n_steps=10, retopologize_fn=partial(retopologize_surface, ...))` per DEM step with `partial(surface_tension_acceleration, gamma, damping, dim=3)` (HC unbound); **crashes at step 0** (remeshed vertices lack `u`; `_film_forces_fn` called with one arg). README describes a `FluidParticleCoupler`/Stokes-drag design that does not exist.

## 11. liquid_bridge_equilibrium/ Case_1..5
Case 1: exact catenoid mesh, `symplectic_euler(dt=2e-6, n_steps=100, retopologize_fn=False)` with `stress_force(mu=0) + surface_tension_force(gamma=0.0728)`, damping 20: **crashes** (`stress_force` needs `v.vd`). Cases 2-4: custom loop with an undocumented spring toward `x_exact` (stiffness 1000); Case 4 README reports t=2e-4 s but runs 1e-4 s. Case 5: catenoid a=1.03 tet fill, `symplectic_euler(..., workers=8, retopologize_fn=False)` **short-circuited** when the static max force is 0.0, then **crashes**. README tables (2026-04-21) cannot be regenerated.

## 12. Case-local BC equivalents
- `Hagen_Poiseuile_3D.py::retopologize_cylinder` acts as a BC: bV = full `HC.boundary()` each step, failed fans promoted into bV, `dual_vol` zeroed on bV; inlet cap, outlet cap and buffer vertices are frozen; injected vertices that form a new hull face are frozen on the next step.
- bc_demo v1/v2/parametric `apply_bcs_measured`: zero u on walls, delete non-wall `x>=3`, inject, purge injected wall vertices, raise on wall-count drop.
- Hydrostatic_2D_periodic: reflecting x-clamp and post-move `u_x=0` on the original side vertices.
- liquid_bridge_cfd_dem: `retopologize_surface` remeshes and **clears bV** (all vertices become free); `sync_film_to_particles` overwrites `v.u` with the particle velocity every DEM step.

## 13. Corner spawning candidates (Poiseuille family)

Ghost arithmetic (U dt = 1e-3, period 1, unit x in {0, 0.5, 1}): injections occur at x-offsets 1e-3 (steps 0, 1000, 2000), 4.4e-16 (steps 499, 1499, ...), 8.8e-16 (steps 999, ...); offsets and y/r positions repeat exactly each cycle.

| # | Mechanism | Evidence | Cases |
|---|---|---|---|
| M1 | **Duplicate frozen wall vertices 1e-3 from the inlet corner.** Ghost wall-row vertices enter at x=1e-3, 1 mm from the frozen corner vertex at x=0. `cdist=1e-10` (`_params.py:23`, `_bc.py:689`) and the retopo merge (HP2D none; HP3D 1e-9) cannot merge them; `PositionalNoSlipWallBC` freezes them at once and `boundary_filter`/`HC.boundary()` keep them frozen. A sliver column/ring at the inlet/wall corner; bounded (later cycles hit the same key) but doubles the wall density there | HP3D state: 9 wall vertices at z=0 plus 7 at z=0.001 | HP2D `:213-224`, HP3D `:368-379` |
| M2 | **Frozen inlet cap plus spurious hull freezing (3D).** `retopologize_cylinder` freezes every `HC.boundary()` vertex with no filter, on a flag complex hyperct itself warns is unreliable on Delaunay slivers (`hyperct/_complex.py:2289-2307`). The inlet cap never advects, injected vertices pile up behind it, and any injected vertex that becomes hull is frozen before it moves | `run.log`: no interior vertices near mid-tube after 3000 steps; 305/407 frozen, 48 at strictly interior positions | HP3D, `run_cluster.py` |
| M3 | **Pressure sawtooth at the inlet.** The unit IC assigns `p = -G x_unit`; the ghost is shifted by minus one period so the first vertex to enter (x_unit=1) arrives at x~0 with `p=-G` next to main-mesh vertices with `p~0`; fields are copied verbatim and never re-evaluated, so each cycle injects a pressure step of G at the inlet, strongest at the frozen inlet corners | code path | HP2D, HP3D |
| M4 | **Wall dissolution (2D).** Nothing prevents wall crossing; once any vertex is outside [0,D] the original wall vertices stop being hull, `boundary_filter` unfreezes them and they are accelerated off the wall (the C1 mechanism of `bcs_and_cases.md`) | `hp2d_final_state.json`: 14 vertices y<0, 45 y>1, 2 frozen wall vertices left | HP2D, `test_outlet_*` |
| M5 | Stale bV after `PeriodicInletBC.merge_all` (`_bc.py:689`): harmless where bV is rebuilt every step, but a merge can remove the original corner vertex and keep the injected twin, dropping its mass | code path | all `PeriodicInletBC` users |
| M6 | bc_demo: `OutletDeleteBC` deletes the wall corners at exactly x=3; injected wall vertices at x in (0,0.025] never advect and survive only through the auto `cdist=0.177` merge | probe: wall set 6 -> 4, interior 7 -> 19 | `bc_demo.py` |
| M7 | cube_flow: no `boundary_filter`, so the inlet column is frozen and ghost vertices merge into it | probe: injection stops after step 0 | cube_flow 1D/2D/3D |
| M8 | Hydrostatic_2D_periodic: reflection plus post-move `u_x=0` crowds the top corners | saved mesh-evolution figure | `Hydrostatic_2D_periodic.py` |

Ruled out: `PeriodicInletBC._clone_unit` does not add hypercube corner vertices (`Complex.__init__` creates none, `hyperct/_complex.py:252-264`); the HP2D docstring's "145 vs 145 vertices" density argument is stale since the extruded main mesh now has the same axial spacing as the unit mesh.

**DEVELOPMENT.md "Fixed Inlet/Outlet Boundary Conditions"** (Status: Not Started) is the principled fix: a third vertex tier `prescribed_V` excluded from integration but present in Delaunay and stress, `FixedVelocityInletBC`, `FixedPressureOutletBC` with zero-gradient velocity copy, `BoundaryConditionSet.prescribed_V` aggregation, `_retopologize` keeping prescribed vertices out of bV. All subtasks unchecked.

## 14. Selected discrepancies (runtime vs docs)
1. HP2D prints "periodic inlet (period=0.15)" and "outlet at L+0.5"; actual `period=1.0`, `OutletBufferedDeleteBC(15, 2.0)`; `_params.refinements=3` vs `n_refine=1`.
2. HP3D docstring claims geometric boundary detection; it uses `HC.boundary()` topologically.
3. `src/_setup.py:91` promises Dirichlet inlet/outlet velocity; none is added.
4. Every Poiseuille runner ends in the point-value branch of `LinearPressureGradient` (no `dual_vol` at IC time); the Eulerian runner compares velocity point-wise.
5. DEVELOPMENT.md / docs_temp/05 call the Hagen-Poiseuille case complete/validated; that covers `src/_setup.py` and the equilibrium residual only, not the Lagrangian HP2D/HP3D runners (no pins; artefacts show wall loss in 2D and total freezing in 3D).
6. DEVELOPMENT.md `:231-237` describes `setup_hydrostatic` + `UniformMass`; runners use `setup_hydrostatic_column` + `DualVolumeMass`. Hydrostatic 2D figures predate the factor-2 fix in `volume_averaged_scalar`.
7. `cube_flow/src/_setup.py:60-64` "static_walls=False: wall vertices flow dynamically" is false at runtime; "1D and 2D videos verified" vs probe stall.
8. liquid_bridge_cfd_dem README and DEVELOPMENT.md `:708-722` describe `FluidParticleCoupler`/`FluidFilmManager` components that do not exist; liquid_bridge_dem README force 0.39 uN vs 3.89e-4 N; liquid_bridge_equilibrium README tables cannot be regenerated (Cases 1 and 5 crash).
9. No README in Hagen_Poiseuile*, Hydrostatic_column, bc_demo, cube_flow, template; liquid_bridge_equilibrium/_approach write to `out/` without `StateHistory`/snapshots/`view_polyscope.py`.
