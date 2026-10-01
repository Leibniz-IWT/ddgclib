# Solver configuration inventory: capillary_rise, dynamic_caprise_tube, dam_break, electrolysis_bubble, shearing_plate_droplet

Audit date 2026-09-25 (sub-report of `bcs_and_cases.md`). Paths relative to
`/home/endres/projects/ddgclib`; `hyperct/...` resolves to the sibling
checkout. Numbers marked "probe" come from throwaway scripts run with the
`ddg` python that only build meshes / read result JSONs (no repo files
modified).

## 0. Library facts every table relies on

| Item | Evidence |
|---|---|
| `symplectic_euler` per-step order: `_do_retopologize` -> interior verts (`HC.V - bV`) -> accel -> u,x update -> `bc_set.apply_all` -> callback | `ddgclib/dynamic_integrators/_integrators_dynamic.py:1073-1101` |
| Default `_retopologize` (retopologize_fn=None): per-step disconnect + global Delaunay (`:191-207`), `bV` rebuilt from the topological hull (`:282-288`), `boundary_filter` applied only at that final step (`:285-286`), `merge_cdist` default None (`:148`), `skip_triangulation=False` (`:146`), `redistribute_mass=False` (`:52`), `remesh_mode='delaunay'` (`:53`) | `_integrators_dynamic.py:49-297` |
| Vertices whose `batch_e_star` dual walk fails are silently promoted to boundary and added to `dV` (frozen unless a `boundary_filter` rejects them), and get `dual_vol=0` | `_integrators_dynamic.py:233-240, 271-275` |
| Callable `retopologize_fn`: integrator forwards `remesh_mode`/`remesh_kwargs` (`:427-430`) plus `skip_triangulation, boundary_filter, merge_cdist, backend, periodic_axes, domain_bounds, pressure_model, redistribute_mass` ONLY if declared by name and not partial-bound (`:448-461`); before laneF (2026-07-30) all eight were dropped | `NOTE(laneF-forward)` `:441-447`; `docs_temp/debug_session/laneF-dam-break-unstick.md:9-37` |
| `_retopologize_multiphase` signature: no `periodic_axes/domain_bounds/pressure_model`; defaults `split_method='neighbour_count'`, `retopo_remap=None`, `projection_every=1` | `_integrators_dynamic.py:481-488` |
| Adaptive remesh: split edges > L_max with no boundary check (`hyperct/remesh/_driver.py:107-121`); collapse forbidden if EITHER endpoint has `v.boundary` (`hyperct/remesh/_interface.py:82-90`); flip forbidden if both endpoints boundary (`:61-65`); a split midpoint of a boundary edge inherits `boundary=True` (`hyperct/remesh/_operations_2d.py:294-300`) | |
| `mass_conserving_merge(HC, cdist)` merges any pair within cdist, no boundary/frozen check, survivor = first member, position unchanged, does not merge `m_phase` | `ddgclib/multiphase.py:729-806` |
| `HC.V.move` pops the cache entry at `v.x` and re-inserts at the new key; a second vertex at the same key is overwritten | `hyperct/_vertex.py:270-306` |

## 1. cases_dynamic/capillary_rise

### 1a. `capillary_rise_2D.py` (static theta, "original scaffold")

| Row | Value (file:line) |
|---|---|
| dim | 2, gravity_axis=1 (`capillary_rise_2D.py:60-63`) |
| domain builder | `rectangle(L=2r=1e-3, h=h_init=0.3*h_jurin=4.40e-3 m, refinement=3, flow_axis=0)` (`src/_setup.py:236-237`, `:216-217`); groups by `identify_face_groups` (`:240-245`). Probe: 145 verts, bV 14, free surface 9 |
| integrator | NONE from the library. Hand-written CFL loop: `_recompute_duals`, `cache_dual_volumes`, `iv=_interior_verts(HC,bV)`, `dt=min(0.25*dx_min/(c0+u_max), t_end-t)`, `u += dt*a`, `_move(x+dt*u)`, then `bc_set.apply_all` (`capillary_rise_2D.py:185-207`). No retopology of any kind (comment `:159`) |
| dt / t_end | adaptive dt (CFL 0.25); `t_end = 300 * h_init/c0` = 0.348 s (`:160-163`); abort if `u_max > 10*c0` (`:225-227`) |
| dudt | `make_capillary_dudt` = `partial(stress_acceleration, dim, mu=mu_art, HC, pressure_model=eos)` + g_vec + `P_cap/(rho*h_ref())` on the gravity axis for EVERY interior vertex (`src/_setup.py:79-94`, `:305-309`). `mu = mu_art = 0.1*rho*c0*dx_mean` = 0.114 Pa s (probe; 104x water); the physical `mu=0.0011` is NOT added (`:293, 306`); P_cap = 143.4 Pa (2D) |
| retopo | none; duals recomputed every step via `compute_vd` only (`:186`) |
| BCs | `NoSlipWallBC` on `wall_verts - free_surface` and on `bottom_verts - free_surface` (`src/_setup.py:322-325`); `bV = wall - free - bottom` (`:265`), so the bottom row and wall/bottom corners are integrated then velocity-zeroed each step (drift `dt^2 a` per step), and wall/top corners are fully free. `PeriodicInletBC` imported but unused (`:39`, TODO `:327-329`). `src/_boundary_conditions.py::make_capillary_bc` is dead code (never imported) |
| ICs | `ZeroVelocity`; `HydrostaticEOSMass(eos, rho0=rho, g, gravity_axis, h_ref=h_init, P_ref=0)` (`:277-285`) |
| EOS | `TaitMurnaghan(rho0=997, P0=0, K=rho*(10*sqrt(g*h_jurin))^2, n=1, rho_clip=(0.5,2.0))` (`:221-230`); c0 = 3.79 m/s, K = 1.43e4 Pa (probe). `src/_params.py::eos_params` (`:222-254`) is never used |
| StateHistory | `fields=['u','p'], record_every=1`, appended manually every `rec` steps (`:171, 211-218`) |
| validation | h_mean of free-surface vertices vs Washburn ODE + Jurin line (`:253-269`) |
| status | **unstable / aborted**. `results/caprise_2d_final.json` (2026-04-09): t_final = 0.0044 s = 1.3 % of t_end, max u 38.5 m/s ~ 10 c0 (the abort threshold); 30 of 145 vertices lie outside the slit (x from -1.7 to +2.7 mm for a 1 mm wide slit), all in the top rows (probe) |

### 1b. `capillary_rise_3D.py` (static theta)

| Row | Value |
|---|---|
| dim / builder | 3, gravity_axis=2; `cylinder_volume(R=r, L=h_init=8.80e-3, refinement=2, flow_axis=2)` (`src/_setup.py:252-259`). Probe: 189 verts, bV 48, free 25 |
| integrator | manual CFL loop as 2D but duals recomputed only every `DUAL_EVERY=50` steps (`capillary_rise_3D.py:110, 121-123`) |
| dt / t_end | `dt = 0.1*dx_mean/c0` only sizes `t_end = dt*2000` = 0.0429 s (`:81-82, 113`); step CFL 0.25 adaptive |
| dudt | same wrapper; `mu = mu_art = 0.615 Pa s` (559x water, probe); P_cap = 286.8 Pa |
| BCs / ICs / EOS | as 2D (`bV = walls - outlet - inlet`, `:265`); c0 = 5.36 m/s, K = 2.87e4 |
| status | **unstable, completed window**. `results/caprise_3d_diagnostics.npz`: h 0.880 -> 0.833 cm (column SANK); 31 of 189 vertices outside r=R (max r = 3.18 mm for R = 0.5 mm), all in the top layers; max u 0.60 m/s (probe) |

### 1c. `capillary_rise_2D_dynCA.py` (data-driven CA)

| Row | Value |
|---|---|
| dim | 2, gravity_axis=1 (`capillary_rise_2D_dynCA.py:76`) |
| domain builder | custom `build_strip_2d(width=2a=R, y_bot=-H_res=-4*width, y_top=h0=h_exp(t0), nx)` structured triangles + `connect_and_cache_simplices` (`src/_setup_dynca.py:43-71`). Defaults nx=6 (`:519`); README prod uses nx=4. Probe nx=4: 260 verts, dx0 = 125 um, 104 wall verts, 80 band verts |
| integrator | manual CFL loop (`:216-389`); `_move(v, new_x, HC, frozen)` (`:339`) |
| dt / t_end | `dt = min(cfl*dx_min/(c0+u_max), 0.5*sqrt(rho*dx_min^3/gamma), T_win - t)`, cfl=0.4 (`:310-311`); window t in [t0=0.01, t_end], README prod t_end 0.06 s; `--smoke`: nx=4, t_end=t0+0.02 |
| dudt | `make_dudt_dynca`: `partial(stress_acceleration, dim, mu=mu2, HC, pressure_model=eos)` + g + `F_surf[id(v)]/v.m` (polyline line tension) + in body mode `p_cap(t)/(rho0*h)` for y>0 (`src/_setup_dynca.py:398-444`). `mu = mu_2d = (2/3)*0.0011` (physical, matched slit); gamma = 0.0728 only via F_surf; no alpha_art; drive_mode default `'surface'` (`:528`), README recommends `body` |
| retopo | none via library. Case-local `maintain_mesh()` every `remesh_every=5` steps or when `dx_min < 0.25 dx0` (`:223`): `mass_conserving_merge(cdist=0.3*dx0)` -> `adaptive_remesh(dim=2, L_min=0.45 dx0, L_max=1.6 dx0, quality_target_deg=15, max_iterations=1, smooth_iterations=0)` -> merge again -> `_cleanup_orphans()` -> `rebuild_simplex_cache_2d` (`:185-201`); then density-continuity repair `m = rho_old * vol_new` booked to `injected` (`:231-244`) |
| BCs | no `BoundaryCondition` classes. Per step: (1) `tag_groups_2d` positional re-tagging of wall/bottom/contact/surface and `v.boundary` (`src/_setup_dynca.py:118-156`); (2) `band_mass_reset` for y<0 (`:217-236`); (3) `boundary_mass_reset` on `frozen|contact` (`:239-261`); (4) `u=0` on frozen; (5) one-sided free-surface shave `m -> 1.10*rho*vol` if exceeded (`:277-282`); (6) wall impenetrability clamp x into [0,width], u_x=0 (`:336-338`); (7) contact vertices excluded from integration and slaved kinematically to `theta_exp(t)` with `v_cl = max(3|hdot_exp|, 0.02)` (`:308, 344-362`) |
| ICs / EOS | `apply_ics` (rho0 above datum / hydrostatic below, Poiseuille profile) (`_setup_dynca.py:264-296`); `TaitMurnaghan(rho0, P0=0, K=rho0*c0^2, n=1, rho_clip=(0.3,3.0))`, c0 = 5.36 m/s (`:197-208`) |
| validation | normalized L2 of h_sim vs Heshmati-Piri/Lunowa h_exp, final rel. error, mass closure, ODE references (`:396-411`) |
| status | **stalled** (README `:80-94`). `results/dynca_2d_water_R0.5mm_prod2.json` (nx=4, cfl 0.4, t 0.01-0.06, 21,889 steps, 25 min): L2 0.686, h_final 0.282 vs 1.801 cm (-84 %), peak 0.573 cm, mass_err 3.8e-15, 4 velocity-cap events. `_smoke` (debug on): aborted "debug invariant violation" at step 770, 339 verts (from 260) |

### 1d. `capillary_rise_3D_dynCA.py`

| Row | Value |
|---|---|
| builder | custom `build_tube_3d(R, z_bot=-H_res, z_top=h0, n_rings)` hex-ring disk layers, one global Delaunay via `connect_and_cache_simplices`, boundary from `boundary_from_simplices` (`src/_setup_dynca.py:74-111`); defaults n_rings=3, t0=0.03, t_end=0.06 |
| integrator / dudt | manual CFL loop, `dt = min(0.4*dx_min/(c0+u_max), T_win-t)`; `make_dudt_dynca(mu physical 0.0011, drive_mode='body')`; F_surf always empty in 3D |
| retopo | every 10 steps or `dx_min < 0.12 dx0`: `mass_conserving_merge(cdist=0.15 dx0)`; if anything merged, full disconnect + global re-Delaunay + `boundary_from_simplices` (`:160-182`); no adaptive remesh in 3D |
| BCs | positional `tag_groups_3d` but `v.boundary` then overwritten with the TOPOLOGICAL hull (`:152-153`); band + boundary mass reset; surface shave at exactly `rho*vol` (`:195-200`, NOT the 1.10 band used in 2D); contact ring u_x=u_y=0; radial clamp to r<=R (`:221-227`) |
| status | **diagnostic / runs**. `results/dynca_3d_water_R0.5mm_prod.json` (2026-07-29; n_rings=2, t 0.03-0.04, 3760 steps, 59.5 min, 966 final verts): L2 0.483, h_final 1.182 vs 1.386 cm (-14.8 %), mass_err 1.5e-15 |

## 2. cases_dynamic/dynamic_caprise_tube/Dynamic_caprise_3D_tube.py

Purely kinematic relaxation of a legacy `cube_to_tube` mesh (wall vertices `z += 0.2e-3*(h_jurin - z)`), `HC.V.merge_all(1e-12)` + `compute_vd(cdist=1e-7)` per step, polyscope screenshots. No integrator, dudt, EOS, BCs, ICs or validation. All module imports resolve but `plot_complex_3d_mat` (`:457`) is undefined -> `NameError` if `cap_rise_init_dyn` is called; no `__main__`. **Stalled / legacy.**

## 3. cases_dynamic/dam_break

Shared `src/_params.py`: a=0.05, L=4a, H=2a, col_w=col_h=col_d=a (`NOTE(laneF-geometry)`), rho_l 1000, rho_g 1.225, mu_l 1e-3, mu_g 1.81e-5, gamma 0.072, P_atm 101325, c_s = max(10*sqrt(2 g col_h), 5) = 9.905, K_l = 9.81e4, K_g = 120.2 (probe), alpha_art 0.3, t_end 0.2, cfl 0.1, refine 3 (2D) / 2 (3D).

### 3a. `dam_break_2D.py` (multiphase) — preset `dam_break_2D`
| Row | Value |
|---|---|
| builder | `rectangle(L=0.2, h=0.1, refinement=3)`; phase by position; probe 145 verts, bV 32, 16 liquid, 9 interface |
| dt / n_steps | `dt = cfl*dx_min/c_s` = 1.262e-4, n_steps 1585 |
| dudt | `partial(multiphase_dudt_i, ...)` + g_vec; per-phase mu: liquid `mu_l + alpha_art*rho_l*c_s*dx_mean` = 46.6 Pa s, gas 0.0571 Pa s (probe) |
| retopo | `partial(_retopologize_multiphase, mps, redistribute_mass=True)` + `retopo_remap='conservative'` (now via the preset); per-step FULL Delaunay with stage-1 refresh, redistribution, restore, level anchor |
| ICs | hydrostatic per-phase mass preload `m_phase[k] = eos_k.density(P_hydro_k(y)) * dual_vol_phase[k]` after one `_retopologize` (`src/_setup.py:168-196`) |
| EOS | gas `TaitMurnaghan(rho0=1.225, P0=P_atm, K=120.2, n=1, rho_clip=(0.2,5.0))`; liquid `(rho0=1000, P0=P_atm, K=9.81e4, n=1, rho_clip=(0.8,1.2))` |
| status | **runs** (laneF): KE_liq peak 1.0369e-3 J @ t=0.051 -> 6.3e-4, max u 0.107 m/s, no abort, interface ring 9 constant. Blocker: air sliver-cell F/m ejection. `results/snapshots_2D` mixes the laneF run with an April run (456 files) |

### 3b. `dam_break_3D.py` — preset `dam_break_3D`
`box(0.2, 0.1, 0.1, refinement=2)`; probe 189 verts, bV 98, 7 liquid, 17 interface; dt 2.524e-4, n_steps 793; mu_l_eff 105.8 Pa s; frozen connectivity (`skip_triangulation=True`, forwarded since laneF; before that silently dropped -> per-step Delaunay). **Not re-run since the laneF fix**: `results/snapshots_3D` latest is 2026-04-10 (old col_h=2a, alpha 2.0, t_end 0.02).

### 3c/3d. `dam_break_2D_no_air.py`, `dam_break_3D_no_air.py` (single phase)
Default `_retopologize` with `boundary_filter=v.is_wall`; `partial(stress_acceleration, mu=mu_l+alpha_art*rho_l*c_s*dx_mean = 15.09 / 38.23 Pa s, pressure_model=eos_liq)` + g; uniform `m = rho_l*dual_vol`, `p = P_atm` (the docstring's hydrostatic profile is NOT applied). **Unstable (stale outputs, 2026-04-10, old geometry)**: 2D max u 9.3 m/s at step 0, 100-280 m/s by t=5.8e-4 s, fastest vertices from the top row and the top of the free face; 3D: 7 vertices below the floor at t=0.0031 s, coordinates ~1e54 at t=0.0065 s, aborted.

## 4. cases_dynamic/electrolysis_bubble

| Runner | Config | Status |
|---|---|---|
| `electrolysis_bubble_2D.py` | off-centre bubble box (`_build_offcenter_bubble_box_2d`, raw `HC.V.move` shift loses the (+L,+L) corner: 41 -> 33 rectangle verts); `symplectic_euler(..., retopologize_fn=partial(_retopologize_multiphase, mps, split_method='neighbour_count', redistribute_mass=True), remesh_mode='delaunay')`; per-step Delaunay, no remap; dt 3.16e-8, n_steps 6330; `NoSlipWallBC` + case-local `WallClampBC` (parks non-bV vertices 0.02 R0 = 20 um from y=+/-L, closer than dx_min 63 um -> slivers); both phases `TaitMurnaghan(n=1, P0=0, K=1e5, rho_clip=(0.5,2.0))`, liquid rho0 1000 (c_s 10), gas rho0 10 (c_s 100); `inject_gas_mass(dm_dt=2e-2)` in the callback | runs, not validated; only pinned number: 5-step per-phase mass drift <= 1.94e-15 |
| `electrolysis_bubble_3D.py` | `droplet_in_box_3d(R=1e-3, L=4e-3, refinement 1/1)`, 93 verts, 26 interface, 9 bulk gas; same wiring; dt 6.55e-8, n_steps 2292 | **unstable**: R_eq -> 0 from ~1.1e-4 s (gas phase lost entirely) |
| `electrolysis_bubble_fritz_2D.py` | Fritz-shaped bubble builder (604 verts); `partial(_retopologize_multiphase, mps, split_method='neighbour_count')` WITHOUT `redistribute_mass` -> integrator default False forwarded (no per-phase redistribution, unlike 2D/3D); 80-step smoke | runs (smoke), no pins |

Other electrolysis discrepancies: injection budget over t_end is 13 % (2D) / 6 % (3D) of the initial gas mass (<= +6 % / +2 % radius), so the README's R_eq 1.4-1.5 mm cannot come from mass growth; detachment unreachable in t_end (5 % of the capillary time); R_det uses the 3D pinned-contact-line formula on a 2D planar bubble; `src/_params.py:39-41` "EOS never produces negative pressure" is false with n=1, P0=0; 2D and 3D share `results/snapshots`.

## 5. cases_dynamic/shearing_plate_droplet

Shared setup: `droplet_in_box_2d/3d(R=5e-3, L=0.015)` then anisotropic rescale of r>R0 vertices via `HC.V.move` (pushes 30 outer vertices inside R0 in 2D; in 3D one lands exactly on a droplet vertex -> KeyError crash in setup); plates classified positionally (tol 1e-10), `bV = top|bottom`; periodic axes [0] (2D) / [0,2] (3D); EOS `TaitMurnaghan(n=7.15, rho_clip=(0.8,1.2))` with c_s floor 1.0 m/s (K_o 1000, K_d 900); YL preload; `partial(multiphase_dudt_i, ...)`, mu 0.05 both phases, gamma 0.03, no gravity/alpha_art; `redistribute_mass=True`.

retopo_fn for all runners: case-local closure `_retopo(HC, bV, dim, remesh_mode='delaunay', remesh_kwargs=None)` that IGNORES `remesh_mode` and declares no other retopo kwarg (so `skip_triangulation` etc. are never forwarded): `snapshot -> retopologize_periodic(merge_cdist=None, boundary_filter=None, backend='ghost') -> mps.refresh('neighbour_count') -> redistribute_mass_multiphase -> compute_phase_pressures`. No remap, no cadence.

BCs: `ShearingPlateBC(+/-U_wall e_x, plate_axis=1, plate_coord=+/-L_y, wrap_axes=[(0,(-L_x,L_x))])` on static plate sets captured at setup; plates are both frozen (in bV) and moved by the BC; `HC.V.move` re-keys vertices so `v in bV` checks in callbacks are wrong (snapshot `boundary_coords` holds 2 of 13).

| Runner | refine | dt | t_end | status |
|---|---|---|---|---|
| `shearing_plate_droplet_2D.py` | 3/3 | acoustic/capillary/viscous min = 4.815e-5 | 1.6 s (33,227 steps) | never run in this form |
| `_run_short_2D.py` | 3/3 | 1.926e-5 | 0.05 s | **unstable**: max u 3.2 m/s = 65x U_wall > c_s; interface count 31 -> 0 by t≈0.044 s; 91 of 287 vertices beyond the plates |
| `shearing_plate_droplet_3D.py` | 2/2 | CFL min | 0.8 s | **crashes in setup** (`multiphase.py:325` KeyError) |
| `_run_short_3D.py` | 1/2 | 4.504e-5 | 0.015 s | stalled (1 snapshot on disk) |

Validation: Taylor `D = Ca (19 lambda + 16)/(16 lambda + 16)` = 0.0456 (Ca 0.0417, Re 2.5); initial state is already D0 = 0.075 (2D) / 0.279 (3D) because of the rescale.

## 6. Per-case BC behaviour per step

| BC | writes | adds/deletes vertices | changes bV / tags | overrides integrator velocity | cite |
|---|---|---|---|---|---|
| `NoSlipWallBC` | `v.u = 0` on target set | no | no | yes (post-move zeroing; capillary_rise static targets include non-bV bottom/corner vertices, so they creep by `dt^2 a`) | `_boundary_conditions.py:161-180` |
| `ShearingPlateBC` | `v.x` via `HC.V.move`, `v.u` | no (but `V.move` onto an occupied key drops the occupant) | no | yes | `:266-292` |
| `WallClampBC` (electrolysis) | `v.x` (clamped plane), into-wall `v.u` | no | no; skips bV; ignores targets | partial | `electrolysis_bubble/src/_setup.py:92-118` |
| dynCA `tag_groups_2d/3d` | `v.boundary` on ALL vertices, positional groups | no | yes, every step | n/a | `_setup_dynca.py:118-182` |
| dynCA `band_mass_reset` / `boundary_mass_reset` / surface shave | `v.m`, `v.p` (source/sink booked to `injected`) | no | no | no | `_setup_dynca.py:217-261`; `2D_dynCA.py:277-282` |
| dynCA wall clamp + contact slaving | `v.x`, `v.u` | no | clamped vertices become wall-tagged next step | yes | `2D_dynCA.py:308, 336-362` |
| dynCA `maintain_mesh` | topology, `v.m`, `v.u`, `v.boundary` | YES: splits create, collapse/merge/orphan cleanup delete | split midpoints inherit boundary | no | `2D_dynCA.py:158-201` |

## 7. Corner / contact-line / wall-junction vertex spawning (evidence)

### Capillary rise (dynCA 2D is the runner that actually SPAWNS vertices)
1. **Contact-line rise splits the wall edge and spawns frozen wall vertices at the corner.** The contact vertex is moved up the wall kinematically at up to `v_cl = max(3|hdot|, 0.02)` m/s while the wall vertex below it is frozen; the edge stretches; `adaptive_remesh(L_max=1.6 dx0)` splits it (no boundary exemption); the midpoint has x=0 exactly and `boundary=True`; the next `tag_groups_2d` tags it wall -> frozen, mass reset. One new wall vertex per 0.8 dx0 of contact-line travel per side.
2. **Wall vertices can never be removed by the remesh** (`can_collapse` refuses boundary endpoints); the only removal is `mass_conserving_merge(cdist=0.3 dx0)`. Probe on `results/dynca_2d_water_R0.5mm_prod2_final.json`: minimum wall spacing is exactly 0.30 dx0 (the merge threshold); 12 / 13 wall pairs closer than 0.5 dx0; wall line density 12.0-14.5 vertices/mm in y in [0, 2] mm vs 8.0/mm initially. Vertex count 260 -> 339 at abort in `_smoke`.
3. **Interior vertices are converted into wall vertices by the impenetrability clamp** (`new_x[0]` clamped to 0/width, `u_x=0` -> tagged wall next step -> frozen forever, mass overwritten). README calls this "absorbed into the film". This is what produces sub-0.8 dx0 wall spacings.
4. **Positional free-surface tagging** makes surface vertices non-collapsible and surface-edge splits create more boundary vertices; the contact corner accumulates on both legs.
5. 3D dynCA: no splitting; the radial clamp + `tag_groups_3d` converts near-wall interior vertices into frozen wall vertices; count only decreases (1007 -> 966).
6. Static-theta 2D/3D (no spawning): wall-top corners are excluded from bV and every NoSlip target, there is no impenetrability, and the truncated corner duals push the top rows out through the walls: 30/145 (2D) and 31/189 (3D) vertices end outside the tube.

### Dam break
7. Silent promotion of degenerate-dual vertices to frozen boundary with `dual_vol=0` (multiphase runners, no filter); in the no_air runners the filter keeps them unfrozen but `dual_vol=0` remains, which the EOS reads as maximal compression.
8. Frozen corners pinning advecting free faces (no_air): (0,col_h) and (col_w,0) are frozen while adjacent face vertices advect -> slivers at both wall/free-surface junctions; fastest early vertices originate there.
9. Air sliver cells at reconnection (multiphase): `a=F/m` ejection -> QhullError (laneF §4); `mass_conserving_merge` cannot cure it (drops `m_phase`).
10. No dam-break runner injects or splits vertices; `bV` is refilled from the hull every step, so any vertex reaching the hull becomes a permanent frozen "wall" vertex.

### Other
11. Shearing droplet: plate vertices are simultaneously frozen and translated; the periodic ub-face merge removes vertices at x=+L_x that match an x=-L_x twin, but BC target sets are static, so a removed plate vertex keeps being `V.move`d, popping whatever vertex sits at that key. Candidate for the observed plate-vertex loss 5/8 -> 0/0.
12. Electrolysis: `WallClampBC` parks vertices 20 um from the wall (dx_min 63 um) -> slivers; coincident clamps silently drop a vertex via `V.move`.

## 8. Runtime config vs README / docs discrepancies (selected)
- capillary_rise README `:115-117`: "3D exact simplex dual-volume switch intentionally not enabled" is stale (enabled 2026-07-29; `build_tube_3d` populates `HC._simplices`). README prod flags (`--nx 4 --drive-mode body`) differ from script defaults (nx=6, `surface`). Undocumented 0.02 m/s contact-line speed floor. `src/_setup.py:8` "Bottom: periodic inlet" vs closed bottom. Static-theta dudt binds `mu=mu_art` ALONE (104x / 559x physical). DEVELOPMENT.md / LIBRARY_AUDIT / FEATURES still say "Not started".
- dam_break README `:109-121` claims `alpha_art=2.0`, `t_end=0.02`, `skip_triangulation=True` for the multiphase variants; actual 0.3, 0.2, 2D per-step Delaunay + remap. README `:93-95` "column a x 2a" vs `col_h = a`. Single-phase docstring claims a hydrostatic IC that is not applied. All outputs except `snapshots_2D` predate laneF.
- electrolysis: missing box corner from raw `V.move` translation (2D and 3D); Fritz runs without redistribution while its docstring says it mirrors setup; README bubble growth / detachment claims unreachable at the injection budget.
- shearing plate: README says `MovingWallBC` / plates do not translate (code: `ShearingPlateBC` translates them); `_params.py:33` "Ca ~ 0.1, Re ~ 1" vs actual 0.0417 / 2.5; README lists 3D outputs that do not exist; c_s floor 1.0 m/s is only 20x U_wall and the run exceeds it.
