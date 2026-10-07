# PART A - BC / IC layer (library)

All line numbers refer to the working tree on 2026-09-25. `_bc` = `ddgclib/_boundary_conditions.py`, `_int` = `ddgclib/dynamic_integrators/_integrators_dynamic.py`, `_ic` = `ddgclib/initial_conditions.py`.

## A.0 How BCs and bV interact with the integrator (applies to every case)

Every Lagrangian integrator runs this sequence each step (e.g. `symplectic_euler`, `_int:1073-1101`):

1. `_do_retopologize(...)` (`_int:1074-1082`). With the default `retopologize_fn=None` it calls `_retopologize`. That function **rebuilds bV from scratch every step** from the topological (convex-hull / simplex) boundary: `dV = boundary_from_simplices(...)` (`_int:209-215`), then `dV = {v in dV if boundary_filter(v)}`, then `bV.clear(); bV.update(dV)` (`_int:285-288`).
2. `verts = _interior_verts(HC, bV)` (`_int:793-795`). Vertices in bV are frozen: no acceleration, no velocity update, no position update.
3. Velocity and position update for the interior vertices only (`_int:1087-1096`).
4. `bc_set.apply_all(HC, bV, dt)` runs **after** the move (`_int:1098`, `_bc:134-156`). BCs are applied in insertion order. A BC added with `vertices=None` receives the live `bV` as its target.

Consequences that every BC inherits:
- **All topological boundary vertices** get `v.boundary=True` and `v.dual_vol = 0.0`, including the ones `boundary_filter` leaves unfrozen (`_int:221-224`, `_int:271-275`). Vertices where `batch_e_star` fails are also pushed into `dV` and tagged boundary (`_int:238-240`). Inlet and outlet vertices in HP2D are therefore unfrozen but carry zero dual volume and truncated duals.
- **The freeze decision comes from the convex hull, not from the BC.** If any vertex moves across the line of a straight wall, the whole wall stops being hull boundary. It then drops out of bV and is integrated (see A.3, candidate C1, which I reproduced).
- A BC's `target_vertices` set is captured when `bc_set.add` is called (`_bc:126-132`). After that it is static. It is never refreshed for merged, deleted or injected vertices. The exceptions are BCs that take `vertices=None` (they receive live bV) and BCs that scan `mesh.V` themselves (`PositionalNoSlipWallBC`, the outlet BCs, `PeriodicInletBC`).
- The periodic path returns early (`_int:124-132`). It **silently ignores** `skip_triangulation`, `backend`, `pressure_model`, `redistribute_mass`, `remesh_mode` and `remesh_kwargs`. `retopologize_periodic` has no parameters for them (`ddgclib/geometry/periodic.py:344-354`).
- `HC.V.move` (hyperct `_vertex.py:270-306`) does `cache.pop(v.x)` followed by `cache[x] = v`, with **no collision check**. If a vertex moves onto a key held by another vertex, that vertex is silently evicted from `HC.V`, but its `nn` links stay alive. This affects every BC that moves vertices (ShearingPlateBC, OutletBufferedDeleteBC, PeriodicInletBC ghost). I reproduced it for PeriodicInletBC in A.3 (C2).
- `merge_all` / `merge_pair` (hyperct `_vertex.py:394-466`) keep `vp[0]`, which is whichever vertex the grid iteration reaches first. They **transfer no mass, momentum or boundary tag** from `vp[1]`. Every merge loses mass, and a frozen wall vertex can be replaced by a mobile one.

## A.1 BC class inventory

| Class (file:line) | Per-step action | Attrs written | Injects / deletes vertices | Touches bV | Overrides integrator update | kwargs (defaults) | Used by (cases / tests) | Known problems |
|---|---|---|---|---|---|---|---|---|
| `BoundaryConditionSet` `_bc:109-156` | Runs each BC in insertion order. Target is `verts if verts is not None else bV` | - | - | passes live bV to BCs registered with `None` | - | - | everywhere | Static target sets are never refreshed (A.0). The same vertex can sit in several groups, and the last BC in order wins. |
| `NoSlipWallBC` `_bc:161-180` | `v.u = zeros(dim)` on target (all `mesh.V` if target None) | `u` | no | no | resets `u` after the move. Position is frozen only if v is in bV | `dim=3` | HP2D `src/_setup.py`, HP_Eulerian, HP_equilibrium test, Hydrostatic, capillary_rise `src/_setup.py:322-325`, cube2droplet, cube_flow, dam_break, electrolysis, oscillating_droplet `src/_setup.py`, template; tests `test_dynamic_integrators.py`, `test_stress.py` | Static target. A wall vertex that leaves the hull (and so is integrated) still has `u` zeroed each step. Its position still integrates one step of acceleration (`u_new = 0 + dt*a`), so it creeps. Default `dim=3` gives a length-3 `u` in 2D if the caller forgets `dim`. |
| `MovingWallBC` `_bc:183-224` | `v.u = wall_velocity` (constant or callable) | `u` | no | no | resets `u` | `wall_velocity`, `dim=3` | **unused** (no case, no test by grep) | Dead code in cases |
| `ShearingPlateBC` `_bc:227-292` | Moves target by `plate_velocity*dt` with `mesh.V.move`, clamps the normal coordinate, wraps `wrap_axes`, sets `u` | `x`, `u` | no | no (docstring says plates need not be in bV) | **yes**: moves vertices itself. If the plate vertices are not in bV the integrator has already moved them, so they get integrator + BC motion. The normal clamp removes only the normal part | `plate_velocity`, `plate_axis`, `plate_coord`, `wrap_axes=None`, `dim=2` | shearing_plate_droplet `src/_setup.py` | Wrapping can land a plate vertex on another plate vertex's key, and `HC.V.move` then evicts that vertex silently (A.0). Wrapping bypasses the periodic retopo. |
| `DirichletVelocityBC` `_bc:295-320` | `v.u = value` or `value(v)` | `u` | no | no | resets `u` | `value`, `dim=3` | HP `src/_setup.py`, template | Static target. Corner vertices sit in both the `inlet` and `walls` groups (A.2), so insertion order decides which BC wins. |
| `DirichletPressureBC` `_bc:323-345` | `v.p = value` | `p` | no | no | no (but for EOS cases `p` is recomputed from `m/V` elsewhere, which can overwrite it) | `value` | template, `example_features_demo.py` | Point value. It breaks the volume-averaged rule when `value` is a function of `x`. |
| `NeumannBC` `_bc:350-390` | Copies the field from the nearest neighbour outside the target set, plus `flux*dist*normal` | `field_name` | no | no | resets the field | `field_name='u'`, `flux_value=0.0` | **unused** in cases | If target is None it uses `set()` (`_bc:371`), a silent no-op when called directly. The "normal" is the vertex-to-neighbour chord, not the face normal. |
| `OutletDeleteBC` `_bc:395-446` | Deletes **every** vertex with `x[axis] >= outlet_pos` (`_bc:426`), including frozen wall vertices lying exactly on the outlet plane. Optional backflow clamp `u[axis]=max(u,0)` in the zone | `u[axis]` (clamp) | **deletes** | `bV.discard` if `bV` passed | clamp overrides `u` | `outlet_pos`, `axis=2`, `bV=None`, `backflow_clamp=None` | HP2D (legacy), `test_outlet_old_bc.py`, `test_outlet_new_bc.py`, bc_demo (all 4), cube_flow 1D/2D/3D + `src/_setup.py:113` | Uses `>=`: when `outlet_pos == L`, the outlet-face vertices, **including the two wall/outlet corner vertices**, are deleted on the first call. The default `axis=2` is wrong for 2D channels if the caller omits it. `bV=None` by default leaves stale bV references until the next retopo. The clamp destroys KE (D2, `docs_temp/sources/debugging_plan_distilled.md:83`). Deleting vertices loses mass. |
| `OutletBufferedDeleteBC` `_bc:449-537` | (1) delete `x >= outlet_pos+buffer_width`. (2) Register any vertex with `x > outlet_pos` and freeze its **full** velocity vector. (3) Every step, move each buffer vertex to `correct_pos + frozen_u*dt` and reset `u` | `x`, `u` | **deletes** | `bV.discard` | **yes**: position and velocity override (the integrator's update is discarded) | `outlet_pos`, `buffer_width`, `axis=0`, `bV=None` | HP2D (`Hagen_Poiseuile_2D.py:206-211`), HP3D, `test_outlet_*` scripts | The frozen velocity **keeps its wall-normal component**, so buffer vertices travel in straight lines and can leave the channel sideways. I reproduced this causing the wall collapse in C1. The buffer is keyed by `id(v)` (`_bc:486`). `mesh.V.move` onto an occupied key is not checked. |
| `PeriodicInletBC` `_bc:542-697` | Moves the ghost clone by `velocity*dt` **(uniform plug speed, independent of the fields)**. Injects every ghost vertex with `x > inlet_pos` via `mesh.V[tuple(x)]` (`_bc:658`), which **returns an existing vertex on an exact key hit and overwrites its u/p/m**. Copies the ghost's internal edges. Removes the injected vertices from the ghost. Re-clones when the ghost is empty. Then calls `mesh.V.merge_all(cdist)` **over the whole mesh** every step (`_bc:689`) | `u`, `p`, `m` (per `fields`) | **injects**, and merges (deletes) anywhere in the mesh | no (new vertices only enter bV at the next retopo or through `PositionalNoSlipWallBC`) | writes fields of new or hit vertices | `unit_mesh`, `velocity`, `axis=2`, `inlet_pos=0.0`, `cdist=None` (auto: `0.5*min_edge` of the unit mesh, `_bc:579-595`), `fields=['u','p','m']`, `period=1.0` | HP2D (`Hagen_Poiseuile_2D.py:213-224`, `cdist=1e-10` from `src/_params.py:23`), HP3D (`cdist=1e-10`, `Hagen_Poiseuile_3D.py:105,369-376`), bc_demo (all 4), cube_flow 1D/2D/3D (`src/_setup.py:115-117`, `cdist` auto). Imported but **not used** in capillary_rise `src/_setup.py:39` (TODO at `:327`) | See candidates C2 to C4. `period` must equal the unit-mesh span. Nothing checks this (`period=1.0` default). The ghost `p` is not volume-averaged on the main-mesh duals. The global `merge_all` drops mass. Ghost-field discontinuities (D2). |
| `PositionalNoSlipWallBC` `_bc:700-735` | Scans `mesh.V`. For each vertex that matches `criterion_fn`: `u=0`, `v.boundary=True`, `bV.add(v)` | `u`, `boundary` | no | **adds only, never removes** | resets `u` | `criterion_fn`, `dim=2`, `bV=None` | HP2D, HP3D, `test_outlet_*`, bc_demo (all 4). Imported, not used, in capillary_rise | Position test with a tight tolerance (HP2D `wall_tol=1e-10`, `Hagen_Poiseuile_2D.py:149-152`). A wall vertex that moves by more than 1e-10 is **never re-captured**. Its `bV.add` only lasts until the next `_retopologize` rebuilds bV from the hull (A.0). |
| `MeshAdvancer` `_bc:740-777` | Rigid advection of the whole mesh + outlet + inlet | `x` | via its BCs | - | replaces the integrator | - | **unused** | Legacy |
| `PressureReservoirBC` `_bc:784-858` | `m += min(1,tau_inv*dt)*(rho_target*dual_vol - m)` on gas-phase targets. `m_phase[gas]=m` | `m`, `m_phase` | no | no | changes mass (open system) | `rho_target`, `tau_inv`, `gas_phase=0`, `update_m_phase=True` | cube2droplet `cube_to_droplet_2D_bc_comparison.py` only | Skips vertices with `dual_vol < 1e-30`, and that is **every topological-boundary vertex** after a retopo (`_int:271-275`). So when it targets `walls`, this BC is a **no-op on the wall vertices themselves** (footgun 3, `docs_temp/06_known_issues_and_debug_history.md:102`). Relaxation rate is not calibrated (D2). It overwrites `m_phase[gas]` with the total `m`, which is wrong for mixed vertices. |
| `AbsorbingPressureBC` `_bc:861-938` | Relaxes `m` toward the neighbour-average density | `m`, `m_phase` | no | no | changes mass | `tau_inv`, `phase=None`, `update_m_phase=True` | cube2droplet `bc_comparison` only | Same `dual_vol=0` skip. With `phase=None` it writes `m_phase[0]` (`_bc:934`) |
| `ExpandingDomainBC` `_bc:941-1030` | Drifts `rho_target` toward the interior mean (skipping `v.boundary`), then relaxes like PressureReservoir | `m`, `m_phase`, internal target | no | no | changes mass | `initial_rho_target`, `tau_inv`, `tau_inv_target`, `gas_phase=0`, `update_m_phase=True` | cube2droplet `bc_comparison` only | Same skip. The interior estimate excludes `v.boundary` vertices, which already have `dual_vol=0`. |
| Helpers `identify_boundary_vertices` `_bc:29-45`, `identify_cube_boundaries` `_bc:48-75` | Position predicates | - | - | - | - | cube tolerance `1e-14` | HP, Hydrostatic, cube_flow, template | `identify_cube_boundaries` assumes the same `[lb,ub]` on every axis. |

## A.2 Domain-builder / boundary-group layer

- Groups are **static sets computed once at build time**: `identify_face_groups` (`ddgclib/geometry/domains/_boundary_groups.py:8-39`), with tolerance `1e-14`. A corner vertex belongs to several groups at once. In `rectangle`, (0,0) is in `inlet`, `bottom_wall` and `walls` (`_rectangles.py:70-76`). In `cylinder_volume`, the rim is in `walls` and `inlet`/`outlet` (`_cylinders.py:78-100`, and `walls` is not exclusive). A case that applies `DirichletVelocityBC` to `inlet` and `NoSlipWallBC` to `walls` gets order-dependent corner values.
- `DomainResult.bV` = all box faces (`_rectangles.py:67`). The integrator's first `_retopologize` discards it in favour of the hull ∩ `boundary_filter`.
- `periodic_rectangle` / `periodic_box` → `_strip_periodic_faces` (`_periodic.py:10-35`). This removes **every** periodic-face vertex from bV and from all groups, **including the wall∩periodic corner vertices** (so `walls` loses its corners). `retopologize_periodic`, in contrast, keeps them as boundary (`periodic.py:489-503`). Build-time and runtime disagree about the corners.
- `retopologize_periodic` (`periodic.py:344-523`) wraps positions and merges exact (1e-14) ub/lb-face duplicates (`:429-447`). It builds a ghost band `2*max NN distance` wide (`:450-458`), runs Delaunay on real+ghost points and resolves ghost indices to real ones (`:166-206`). Simplices that differ between the ghost-side and real-side triangulation of the same points are **both kept** (dedup only removes identical index tuples). This can add overlapping, non-manifold simplices near the seam, and most likely at the wall∩seam corners. **I infer this and did not measure it.** It uses `cache_dual_volumes` (not `batch_e_star`) and ignores `backend`.
- `droplet_in_box_2d/3d` (`_multiphase_droplet.py:190-310`, `:313-424`):
  - It keeps outer box vertices with `r > R` and adds all disk vertices plus a ring at `R + h_drop` (`:265-279`). Outer vertices with `R < r < R+h_drop` are **not removed**, which can leave near-coincident pairs and sliver triangles next to the ring.
  - Interface ring selection uses `np.linalg.norm(v.x_a)`, **ignoring `center`** (`:267-268`, `:382-383`). This is a latent bug for `center != 0`.
  - `_estimate_edge_length` samples only the first ~50 edges (`:42-43`).
  - Its built-in `mps` uses placeholder `TaitMurnaghan(rho0=1000)`, `mu=0.1` (`:176-183`). Cases must replace it.
  - `bV_walls` = hull ∩ `r > R+1e-10` (`:146-155`).

## A.3 "Too many vertices spawning in corners": candidate mechanisms

I ran two throw-away probes in the scratchpad, **not in the repo**: `inlet_corner_probe.py` and `hp2d_corner_probe*.py`. The second replicates the `Hagen_Poiseuile_2D.py` BC stack with L shortened to 5 and the same `_params` (`U_avg=0.1`, `dt=0.01`, `cdist=1e-10`, `n_refine=1`).

| # | Mechanism | Evidence | Verified? | Cases that exhibit it |
|---|---|---|---|---|
| **C1** | **Convex-hull re-tagging collapses a whole straight wall.** bV is rebuilt from the hull every step (`_int:209-215,285-288`), and nothing enforces impenetrability. When any vertex steps past a wall line, every collinear wall vertex between it and the far corner stops being a hull vertex. They leave bV, get integrated, drift off `|y|<1e-10`, and `PositionalNoSlipWallBC` never re-captures them (`_bc:726-735`). Only the hull extreme points (the corners) stay frozen. In HP2D the trigger is `OutletBufferedDeleteBC`, whose frozen velocity keeps its wall-normal component (`_bc:512-534`). | **Reproduced.** Step 1246: 23 wall vertices frozen. Step 1247: two buffer vertices at x=5.179, y=∓0.00029, with frozen `u=(0.122, ∓0.0466)`. Step 1248: **nWall = 2, nbV = 2** (only the corners). They never recover (`hp2d_corner_probe3.py`). | **Yes** | HP2D, HP3D, `test_outlet_*` (all use `boundary_filter` + buffer). The same hull logic applies to every case whose fluid vertices can cross a wall: dam_break, capillary_rise and all the droplet-in-box cases, whenever a vertex penetrates a wall. |
| **C2** | **PeriodicInletBC ghost loses vertices on reset through key collision.** `_reset_ghost` moves each vertex by `-period` (`_bc:635-641`). The unit mesh's x=span face lands on the x=0 face's keys, and `HC.V.move` evicts the vertex already there (hyperct `_vertex.py:283-298`). | **Reproduced.** A 13-vertex unit mesh gave a 12-vertex ghost: (0,1.0) was lost, so the top wall never receives an injected vertex. | **Yes** | Every PeriodicInletBC user: HP2D, HP3D, bc_demo*, cube_flow* |
| **C3** | **Injection at exact repeated keys and plug-speed ghost.** The ghost advances at the constant `velocity` (`_bc:644-650`), so every column crosses the inlet at the **same float position** `inlet + velocity*dt` (here x=0.001). `mesh.V[key]` then either (a) hits the vertex injected earlier and overwrites its `u/p/m` (`_bc:658-667`), or (b) creates a new vertex if the earlier one moved. For **frozen wall vertices** (a) happens: the wall vertex at (0.001, 0) is reset to `u=U_avg` every column period, in HP2D every 250 steps. `PositionalNoSlipWallBC` re-zeroes it only on the next BC pass, so neighbours see a periodic velocity kick at the inlet corner. For near-wall interior vertices, (b) plus the no-slip slowdown means inflow at `U_avg` exceeds the local outflow `u(y) << U_avg`, and **vertices build up near the inlet-wall corners**. The script's own header documents the density mismatch: a 145-vertex unit mesh gets dumped into a 0.15-wide strip, about 2900 extra vertices in 3000 steps (`Hagen_Poiseuile_2D.py:33-45`). That was for the `inlet_layer_thickness=0.15` path, which is now disabled behind `if 0:` (`:156-160`). | (a) reproduced: bit-identical injection key (0.001, 0.0) every 250 steps (`inlet_corner_probe.py`). (b) reasoned, and consistent with the script header. | Partly | HP2D/HP3D, bc_demo*, cube_flow* |
| **C4** | **`merge_cdist` / `cdist` either too tight or too global.** HP2D/HP3D pass `cdist=1e-10` (`src/_params.py:23`, `Hagen_Poiseuile_3D.py:105`), so near-duplicates are never merged. The auto value (`0.5*min_edge`, `_bc:579-595`) is applied to **the whole mesh** every step (`_bc:689`). `merge_pair` keeps an arbitrary survivor and drops `m` (hyperct `_vertex.py:454-466`), so a frozen corner vertex can be swapped for a mobile one. The integrator-level `merge_cdist` is `None` in every case I found. | Reading | Yes (code) | Same as C3. cube_flow uses the auto cdist. |
| **C5** | **`OutletDeleteBC` `>=` deletes the outlet plane.** With `outlet_pos == L`, the outlet-face vertices, including both wall/outlet corners, are removed on the first call (`_bc:426`). The next hull then leaves a notch at the outlet corners and re-tags the neighbours. | Reading | Yes (code) | cube_flow (`outlet_pos=L`, `src/_setup.py:113`), bc_demo (`outlet_pos=L_domain`, `bc_demo.py:127`) |
| **C6** | **Adaptive remesh ratchet at walls and corners.** `can_collapse` refuses **any** edge with a `v.boundary` endpoint (hyperct `remesh/_interface.py:84-89`). Splits are allowed on boundary and boundary-adjacent edges (`remesh/_driver.py:107-121`), with threshold `alpha_max*mean incident edge length` in the default `length_scale='local'` mode (`_driver.py:66-84`). Smoothing skips boundary vertices (`_driver.py:193`). **Near walls and corners the vertex count can only grow.** A 2D corner vertex has short and long incident edges, so its long diagonal edges get split repeatedly while the resulting short wall-adjacent edges can never be collapsed. `v.boundary` here is the full hull set, not only the filtered bV (`_int:221-224`). | Reading | Yes (code, not measured) | Only runs with `remesh_mode='adaptive'`: `oscillating_droplet_2D_adaptive.py:137,252` and `cube_to_droplet_2D_adaptive.py:155`. Shearing plate and electrolysis forward `params['remesh_mode']`, default delaunay. The DO-NOT in `06_known_issues:37` forbids adaptive in production. |
| **C7** | **Truncated corner duals, zero boundary volume.** Hull vertices get `dual_vol=0` (`_int:271-275`). Corners give degenerate dual base triangles, which produced NaN before the fix (`06_known_issues:48`), and `batch_e_star` failures are re-tagged boundary (`_int:238-240`). Accelerations here are spurious rather than extra vertices, but they drive C1 penetration. The dam_break README names "corner vertices where the barycentric dual cell is truncated" (`docs_temp/code_map/cases_dynamic_inventory.md:52`, `06_known_issues:63`). laneF localises the terminal dam-break failure to the same class of defect: a reconnection leaves an air vertex on a sliver dual cell with volume near 0, so `m = rho_g*dvp` is O(1e-4). The force stays finite, `a = F/m` spikes, and the vertex goes ballistic to about 6.7x outside the domain. `QhullError` follows within about 100 steps (`docs_temp/debug_session/laneF-dam-break-unstick.md:128-154`). `mass_conserving_merge` does not merge `m_phase` (`:149`). | Docs + code | Yes | dam_break, electrolysis_bubble 3D (NaN sanitisation), every box case |
| **C8** | **Periodic seam.** `_strip_periodic_faces` un-freezes the wall∩seam corner vertices at build (`_periodic.py:10-35`), while runtime re-freezes them (`periodic.py:489-503`). Ghost-band Delaunay can add overlapping simplices at the seam corners (A.2). | Reading, simplices inferred | Partly | Hydrostatic_2D_periodic, shearing_plate (plate wrap) |
| C9 | **Multiphase builder near-duplicates** at the interface ring (A.2). Not a corner effect, but it also adds sliver cells. | Reading | Yes (code) | oscillating_droplet, cube2droplet, shearing_plate, electrolysis (any user of `droplet_in_box_*`) |

The case agents' findings for capillary_rise contact-line BCs and dam-break slivers are in Part B (§B.2).

## A.4 IC class inventory

"Volume-averaged OK?" means: does the class obey the rule "v.p is the dual-cell average, never P(x_vertex)" (CLAUDE.md FVM conventions)?

| Class (`_ic` line) | Sets | Volume-averaged OK? | Prereqs | Used by | Problems |
|---|---|---|---|---|---|
| `CompositeIC` 39-47 | runs its children in order | - | - | HP*, Hydrostatic, bc_demo*, cube_flow, template, `test_stress.py` | Order matters: a mass IC that reads `dual_vol` must run after the duals exist. |
| `UniformPressure` 52-60 | `p = P0` | yes (constant) | - | **unused** | - |
| `HydrostaticPressure` 63-108 | `p` = volume average of `P_ref + rho*g*(h_ref - x)` | **Conditional**: only when `v.vd` and `dual_vol` exist (`:102-106`). Otherwise it **silently falls back to the point value** (`:107-108`). `volume_averaged_scalar` also falls back to the point value when `vol<1e-30` (`analytical/_integrated_comparison.py:214-216`), which is every boundary vertex after a retopo | duals before apply | Hydrostatic `src/_setup.py`, template demo, `test_integrated_validation.py` | Silent rule violation on boundary vertices and whenever it is applied before `compute_vd`. |
| `LinearPressureGradient` 111-146 | `p = P_ref - G*x[axis]`, volume-averaged | Same conditional fallback | duals | HP* (all), cube_flow, template, tests | HP2D applies it to the **unit mesh**, whose duals were never computed. So the ghost carries **point values** (`Hagen_Poiseuile_2D.py:183-194`: no `compute_vd` on `unit_mesh`), and those are injected as-is. |
| `ZeroVelocity` 151-159 | `u = 0` | n/a | - | most cases | - |
| `UniformVelocity` 162-170 | `u = u_vec` | n/a | - | HP*, bc_demo, cube_flow | Plug flow at no-slip walls until the BC runs (HP2D calls `bc_set.apply_all(dt=0)` first, `:244`). |
| `PoiseuillePlanar` 173-214 | point `u` profile | n/a (velocity is nodal) | - | HP2D analytic, HP_Eulerian, HP_equilibrium, template | - |
| `HagenPoiseuille3D` 217-250 | point `u_z(r)` | n/a | - | HP3D, `test_stress.py` | - |
| `CustomFieldIC` 255-273 | any attr = `fn(x)` | **Violates** the rule if used for `p` | - | **unused** | - |
| `UniformMass` 276-297 | `m = rho*V_total/N` (same for every vertex) | n/a, but **density `m/dual_vol` is non-uniform**, which is wrong for EOS cases | - | HP*, Hydrostatic, bc_demo, cube_flow, template | Only safe when `p` is not derived from `m/V`. |
| `DualVolumeMass` 300-326 | `m = rho*dual_vol` (`1e-30*rho` if `vol<1e-30`) | n/a | duals + cached `dual_vol` | Hydrostatic (all 4) | After a retopo, boundary vertices have `dual_vol=0` and get **m≈0** (footgun 3). |
| `HydrostaticEOSMass` 329-412 | `m = rho(P(z_vertex))*dual_vol`, `rho = rho(P)`, **`p = P(z_vertex)`** | **Violates**: pointwise `p` and pointwise density (`:401-412`) | duals | Hydrostatic 1D/2D/3D, capillary_rise `src/_setup.py`, dam_break `src/_setup.py` | Point-value `p` against the volume-averaged convention. Boundary `m≈0` as above. |
| `PhaseAssignment` 417-433 | `v.phase = criterion(x)` | n/a | - | cube2droplet `bc_comparison`, `diagnostic_no_retopo`, `test_multiphase.py` | Vertex-based. The droplet builders use simplex-based phases (`mps.assign_simplex_phases`), so the two models can disagree at the interface. |
| `MultiphaseMass` 436-458 | `m = rho0[phase]*dual_vol` | n/a | duals, `v.phase` | **tests only** (`test_multiphase.py`) | No `m_phase` set. Interface vertices (`phase=-1`) index `phases[-1]`, the last phase (`:453`). |
| `MultiphasePressure` 461-497 | `p = eos.pressure(rho0)` + `gamma*curvature` on the inner phase | constant per phase, OK | `mps._gamma` | **tests only** | Assumes the outer phase is 0 (`:492`). Does not set `p_phase`. The caller must pass curvature as 1/R (2D) or 2/R (3D). Interface `phase=-1` has the same indexing hazard (`:486-487`). |

The multiphase cases (oscillating_droplet, cube2droplet, shearing_plate, electrolysis, dam_break) **do not use these IC classes** for mass and pressure. They set `m`, `m_phase` and `p` inside each case's `src/_setup.py`, including the Young-Laplace pre-load. Those per-case ICs are listed in Part B.

# PART B - per-case configuration matrix

**Scope.** Rows marked **(direct)** I read and verified myself with file:line. Rows marked **(summary)** come from a grep-level read only. The per-case subagent reports went to the coordinator, not to me, so use those reports for the full per-runner tables.

**Working tree is moving.** An untracked package `ddgclib/methods/` (`SolverMethods`, `PRESETS`, `record_methods`, `METHODS.md` referenced) plus `ddgclib/tests/test_methods.py` appeared on 2026-09-25 13:00-13:06. `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` was modified at 13:07:18, while this audit was running. It now imports `PRESETS` (`oscillating_droplet_2D.py:47,58,73`). Line numbers for that file may shift. The presets are in `ddgclib/methods/_presets.py:24-121`.

Shared default. When a case passes a `partial(_retopologize_multiphase, ...)` without binding a kwarg, the defaults at `_int:481-488` apply: `merge_cdist=None`, `skip_triangulation=False`, `redistribute_mass=False`, `remesh_mode='delaunay'`, `split_method='neighbour_count'`, `retopo_remap=None`, `projection_every=1`.

Forwarding rule (`_int:414-464`, NOTE laneF-forward). Integrator-level kwargs reach a callable retopo_fn **only if the callable declares them by name and they are not already bound in the partial**. `remesh_mode` and `remesh_kwargs` are always forwarded when declared or accepted by `**kwargs`. This fixes the laneD finding that dam_break's integrator-level `skip_triangulation=True` was silently dropped. The current `dam_break_3D` preset note says connectivity was "was skip_triangulation=True at integrator level" (`_presets.py:117-120`).

## B.1 oscillating_droplet (direct)

| Item | 2D runner `oscillating_droplet_2D.py` | 3D runner `oscillating_droplet_3D.py` |
|---|---|---|
| builder | `setup_oscillating_droplet` (`src/_setup.py:24-43`) → `droplet_in_box_2d/3d`. Params: `L_domain=5*R0`, refine 3/3 (`src/_params.py:172-176`) | same, 3D |
| integrator | `symplectic_euler` (`:152-153`), forwards `remesh_mode`/`remesh_kwargs` from params | `symplectic_euler` (`:168-169`) |
| dt / t_end | `dt = min(0.25*dx_min/c_s, 0.5*sqrt(rho_d*dx_min^3/gamma))`. `t_end = min(t_end_2d, 5/beta)` (`:109-112`). Full run 1839 steps (`_params.py:74`) | `dt` same form (`:122`). `t_end_3d`. 872 steps (`_params.py:144`) |
| dudt | `partial(multiphase_dudt_i, dim, mps, HC, pressure_model=MultiphaseEOS([outer, drop]))` (`src/_setup.py:210-214`) | same |
| EOS | `TaitMurnaghan(n=7.15, rho_clip=(0.8,1.2), P0=0)`, `K = rho*c_s^2`, `c_s = max(10*u_scale, 1)` (`src/_setup.py:116-119`, `_params.py:65-68`) | same |
| retopo_fn | `partial(_retopologize_multiphase, mps, split_method='neighbour_count', redistribute_mass=True)` (`src/_setup.py:230-232`). Plus policy `retopo_policy_2d='delaunay_remap'` → `+ retopo_remap='conservative'` (`_params.py:139`, runner `:96-99`) | policy `retopo_policy_3d='dual_only'` → `+ skip_triangulation=True` (`_params.py:169`, runner `:106-107`) |
| BCs | `NoSlipWallBC` on the static `boundary_groups['walls']` (`src/_setup.py:207`) | same |
| ICs | Hand-rolled in setup: YL mass pre-load, `mps.refresh(split_method=...)` (`src/_setup.py:164,203`). No library IC classes | same |
| StateHistory | `['u','p','phase','is_interface']` (`:117-121`) | same |
| metric / status | Full-run l2 **0.17479361640597058**, tail 0.99990 (`_params.py:106-110`; not test-pinned, see Part C). **Runs; over-damped** (laneH attribution) | l2 **0.24811340819647862**, tail 0.08410 (`_params.py:147-149`). laneG: bump / over-decay cancellation. **Runs, not validated** |

Discrepancies:
1. The pinned floor tests do **not** exercise the runner policy. They use setup's bare-Delaunay `retopo_fn` (`_params.py:115-117`, `:156-158`).
2. The split_method trap (`06_known_issues:47,101`): setup and runtime must match. Only `setup_oscillating_droplet` threads it through.
3. `dual_only` is the pre-laneE 2D default, still selectable.
4. `oscillating_droplet_2D_adaptive.py` and `cube_to_droplet_2D_adaptive.py` are the only `remesh_mode='adaptive'` users (`:137,252`; `:155`). That path goes through corner candidate C6.
5. `static_droplet_2D.py` uses a custom `_dual_only_retopo` closure: no `mps.refresh`, no redistribution, no EOS update. Pinned summary **1.1847162859108737e-03** in the preset notes (`_presets.py:63-70`).

## B.2 dam_break (direct)

| Item | `dam_break_2D.py` |
|---|---|
| builder | `rectangle(L=4a, h=2a, refinement=n_refine_2d=3)` (`src/_setup.py:87-90`, `_params.py:31-35,141`). Phases tagged **by vertex** (`v.phase = in_column`, `src/_setup.py:116-125`) |
| integrator | `symplectic_euler(..., retopologize_fn=retopo_fn)`. No `boundary_filter`, so every hull vertex is frozen (`dam_break_2D.py:123-126`) |
| dt / t_end | `cfl_timestep(HC, dim, c_s=sqrt(K_l/rho_l), cfl=0.1)`, `t_end=0.2` (`dam_break_2D.py:71-73`, `_params.py:131,134`) |
| dudt | `multiphase_dudt_i(dim, mps, HC, pressure_model=MultiphaseEOS([gas, liq]))` + constant `g_vec` (`src/_setup.py:203-213`). `alpha_art=0.3` is baked into the phase `mu` as `mu + alpha*rho*c_s_l*dx_mean` (`src/_setup.py:111-114`, `_params.py:112`) |
| EOS | gas `TaitMurnaghan(rho0=1.225, P0=P_atm, K_g, n=1, clip (0.2,5.0))`. liquid `(rho0=1000, K_l, n=1, clip (0.8,1.2))`. `c_s = max(10*sqrt(2 g col_h), 5)` (`src/_setup.py:133-138`, `_params.py:81-84`) |
| retopo_fn | `partial(_retopologize_multiphase, mps, redistribute_mass=True)` (`src/_setup.py:215-216`) **+ `retopo_remap='conservative'` bound in the runner** (`dam_break_2D.py:120-121`). `split_method` defaults to neighbour_count |
| BCs | `NoSlipWallBC(dim)` on `bV_walls`. This is the **same set object** as the integrator's bV, so it is refreshed live by `_retopologize` (`src/_setup.py:199-200`, returned as bV at `:233`) |
| ICs | `ZeroVelocity`. One `_retopologize` at setup. Per-phase hydrostatic mass preload, which is **pointwise** `rho(P(y_vertex))*dual_vol_phase` (`src/_setup.py:160-197`) |
| StateHistory | `['u','p','phase','is_interface']`, record every n/150 (`dam_break_2D.py:81-85`) |
| status | laneF: **runs to horizon; not validated**. KE_liq peak 1.0369e-3 J at t=0.051, front +18.1 mm, mass 6.2e-15, \|u\|max 0.107 = 11% of u_ref. Open blocker: air sliver-cell F/m ejection (`laneF-dam-break-unstick.md:128-167`) |

Sibling runners: `dam_break_2D_no_air.py` / `_3D_no_air.py` use `boundary_filter=_wall_filter` on `v.is_wall` (`dam_break_2D_no_air.py:115-123`). `dam_break_3D` is dual_only per the preset (`_presets.py:117-120`). Smoke test: `test_case_dam_break.py:96-135` (150 steps, alpha 0.5, remap ON).

## B.3 capillary_rise (direct)

| Item | static θ: `capillary_rise_2D.py` (+`_3D`) | dynCA: `capillary_rise_2D_dynCA.py` (+`_3D_dynCA`) |
|---|---|---|
| builder | `setup_capillary_rise`: `rectangle(L=2r, h=h_init)` (2D) or `cylinder_volume(R=r, L=h_init)` (3D). `h_init = 0.3*h_jurin` (`src/_setup.py:215-259`) | `build_strip_2d(width, y_bot=-H_res, y_top, nx)` / `build_tube_3d` (`src/_setup_dynca.py:43,74`) |
| integrator | **Hand-rolled loop, no library integrator, no retopology**: `_recompute_duals` + `cache_dual_volumes` + explicit symplectic update + `bc_set` (`capillary_rise_2D.py:159-207`) | Hand-rolled loop (`:216-365`). Mesh maintenance every `remesh_every=5` steps or when `dx_min<0.25*dx0`: `mass_conserving_merge(0.3*dx0)` → `adaptive_remesh(L_min=0.45dx0, L_max=1.6dx0, q=15°, max_iter=1, smooth=0)` → merge → `_cleanup_orphans` → `rebuild_simplex_cache_2d` (`:173-190`, `:222-248`) |
| dt | `dt = min(0.25*dx_min/(c0+u_max), ...)`. `t_end = 300*h_init/c0` (`:160-163,197`) | `dt = min(cfl*dx_min/(c0+u_max), 0.5*sqrt(rho dx^3/gamma), ...)`, `cfl=0.4` (`:73,309-311`) |
| dudt | `make_capillary_dudt`: stress(mu=mu_art) + gravity + **Washburn body force** `P_cap/(rho*h)` (`src/_setup.py:51-94,305`) | `make_dudt_dynca` (`src/_setup_dynca.py:398`). Surface tension on the extracted surface chain. Contact vertices are **kinematically slaved** to the measured CA(t) (`:350-362`) |
| EOS | `TaitMurnaghan(rho0, P0=P_atm, K=rho*(10*sqrt(g h_J))^2, n=1, clip (0.5,2.0))` (`src/_setup.py:221-230`) | `make_eos` (`src/_setup_dynca.py:197`) |
| BCs | `NoSlipWallBC` on `wall − free_surface` and on `bottom − free_surface` (`src/_setup.py:322-325`). `bV = wall − free_surface − bottom` (`:265`). So the **bottom and wall∩bottom corners are not frozen**: `u` is zeroed but the position still integrates `dt*a`. The contact-line corners (wall∩top) are neither frozen nor no-slip. PeriodicInletBC is imported but unused (`:39`, TODO `:327`) | Positional regroup each step via `tag_groups_2d`, which **rewrites `v.boundary` for all vertices** (`src/_setup_dynca.py:118-156`). `frozen = wall ∪ bottom − surf` gets `u=0` (`:268-274`). Impenetrable clamp for `x∉[0,width]` (`:336-338`). `band_mass_reset` reservoir and `boundary_mass_reset` hydrostatic density on frozen and contact vertices (`:271-272`). One-sided surface shave at `1.10*rho0` |
| ICs | `ZeroVelocity` + `HydrostaticEOSMass` (**pointwise p**, A.4) (`src/_setup.py:277-285`) | `apply_ics` (`src/_setup_dynca.py:264`) |
| StateHistory | `['u','p']`, record_every=1 (`:171`) | `['u','p']` with save_dir (`:137`) |
| status | README: original scaffold, **superseded** | Best 2D prod: healthy window L2 **0.14**, h 0.514 vs 0.544 cm (−5.5%). Then **stalls**: peaks 0.573 cm at t=0.017 s, full-window L2 **0.69**, mass ledger ~4e-15 (`README.md:80-93`). 3D is diagnostic: −14.8%, L2 0.48 (memory note) |

**Capillary-rise corner verdict (dynCA).** The dynCA memory note states it directly: "Reservoir = Eulerian mass-source band … + **adaptive-remesh edge splits = vertex injection without ghost mesh**" and "crowding sink". The mechanism:
1. The Lagrangian shear slides the interior past the frozen wall columns. The contact vertex is dragged up the wall (`:350-362`). Wall-adjacent and wall-along edges stretch past `1.6*dx0`, so `edge_split_2d` inserts midpoints (`remesh/_driver.py:107-121`).
2. A split of a wall edge lands at `x=0`/`width`. `tag_groups_2d` re-classifies it as `wall` and therefore frozen (`src/_setup_dynca.py:133-146`).
3. `can_collapse` refuses every edge with a `v.boundary` endpoint (`remesh/_interface.py:84-89`), and `tag_groups_2d` marks walls, bottom and the whole surface band as boundary (`:152-154`). So those vertices can only be removed by `mass_conserving_merge` at `0.3*dx0` or by `_cleanup_orphans`.

Net effect: vertex count ratchets up along the walls, and most at the moving **wall∩surface contact corners**. This is C6, in the one production case that runs adaptive remesh. The static-θ variant spawns nothing (no remesh), but its unfrozen, un-clamped corners drift (see BCs row).

## B.4 electrolysis_bubble (direct)

| Item | 2D `electrolysis_bubble_2D.py` / 3D `_3D.py` |
|---|---|
| builder | `setup_electrolysis_bubble` (`src/_setup.py:221-240`). 2D uses an off-centre build. 3D uses `droplet_in_box_3d`, centred (`:318-350`). Refine outer/drop 2/3 (2D), 1/1 (3D) (`_params.py:72-76`) |
| integrator | `symplectic_euler(retopologize_fn=retopo_fn, remesh_mode=params['remesh_mode']='delaunay')` (2D `:167-171`, 3D `:153-156`, `src/_setup.py:499`) |
| dt / t_end | `min(dt_cfl, dt_st)`, `cfl_safety=0.05`. `t_end_2d=2e-4`, `t_end_3d=1.5e-4` (`_params.py:100-104`, runner `:105-106`) |
| dudt | `multiphase_dudt_i(... pressure_model=MultiphaseEOS([liq, gas]))` + gravity. **Returns zero** for `m<1e-30`, non-finite `m`, or non-finite `a`, which is the NaN guard for 3D corners (`src/_setup.py:456-472`). Mass source: `inject_gas_mass` in the runner callback (`electrolysis_bubble_2D.py:37`) |
| EOS | `TaitMurnaghan(n=1, clip (0.5,2.0))` for both phases (`src/_setup.py:299-304`) |
| retopo_fn | `partial(_retopologize_multiphase, mps, split_method='neighbour_count', redistribute_mass=True)` (`:474-477`, default `:240`) |
| BCs | `NoSlipWallBC` on bV (live set) + case-local **`WallClampBC`** top and bottom (`use_wall_clamp=True`, `:238,441-453`). WallClampBC moves any non-bV vertex that crosses the plane back to `level ± 0.02 R0` via `mesh.V.move` and zeroes the normal inward `u` (`src/_setup.py:66-115`). Setup also runs `mass_conserving_merge(cdist=1e-12)` (`:436`) |
| StateHistory | `['u','p','phase','is_interface']` |
| status | README: runs over the short horizon. 3D loses interface vertices on necking. Mass slowly lost. No contact-angle BC (`README.md:140-160`). **Runs, not validated** |

Corner note. `WallClampBC` stacks every clamped vertex on a single line at `level ± 0.02R0`. That line is parallel to the frozen wall row, so each clamp adds sliver triangles next to the wall. At the corners it meets the side walls. Collisions on that line are silently evicted by `HC.V.move` (A.0). This is a C1/C7-class hazard, inferred from code.

## B.5 Hagen_Poiseuile family (direct for HP2D, grep-level for the rest)

| Case | Integrator & retopo | BCs | ICs | dt / n | Status / notes |
|---|---|---|---|---|---|
| `Hagen_Poiseuile/Hagen_Poiseuile_2D.py` | `symplectic_euler(..., boundary_filter=wall_criterion, workers=20)`. Default `_retopologize`, `merge_cdist=None` (`:277-286`). `dudt = partial(dudt_i, dim=2, mu, HC)` (`:262`). Mesh: `extrude(Complex [0,1]x[0,D] refine 1, L=15)` (`:111-120`) | `PositionalNoSlipWallBC(wall_tol=1e-10, bV)`, `OutletBufferedDeleteBC(outlet_pos=L, buffer_width=2.0)`, `PeriodicInletBC(unit=[0,1]x[0,D], velocity=U_avg=0.1, cdist=1e-10, period=1.0)` (`:200-224`). Printed "outlet at L+outlet_buffer" is **wrong**: the real value is L+2.0 (`:205,208,227`) | plug `UniformVelocity(U_avg)` + `LinearPressureGradient` (unit mesh: **pointwise**, no duals) + `UniformMass` (`:183-194,236-241`). `unit_bV` is built from **HC** (the main mesh), not the unit mesh (`:190-192`), which is a bug | dt 0.01, n 3000. StateHistory `['u','p']` | **Reproduced wall collapse C1** at step ~1247 of a shortened replica. Ghost reset drops the top-wall vertex (C2). Header documents vertex crowding (`:33-51`). **Unstable / not validated** |
| `Hagen_Poiseuile_3D.py` | `symplectic_euler(retopologize_fn=retopologize_cylinder)`: custom `merge_all(1e-9)` + Delaunay + **drops sliver tets** (`quality 1e-4`) + `HC.boundary()` (`:186-230,444-452`) | `PositionalNoSlipWallBC` (r=R), `OutletBufferedDeleteBC(outlet_pos=L, buffer=outlet_buffer)`, `PeriodicInletBC(U_avg, cdist=1e-10, period=1.0)` (`:349-376`) | `HagenPoiseuille3D` / `UniformVelocity` / `LinearPressureGradient` / `UniformMass` | dt 0.01, n 3000 (`:111-112`) | Same C1-C4 exposure. Dropping sliver tets leaves holes, which become spurious boundary |
| `Hagen_Poiseuile_2D_Eulerian` | `euler_velocity_only(...)`, fixed mesh (`:156`) | `NoSlipWallBC` | `PoiseuillePlanar`, `LinearPressureGradient`, `UniformMass` | dt 0.005, n 2000 (`:147-148`) | Validation-only (Eulerian) |
| `Hagen_Poiseuile_equilibrium` | No time integration: evaluates `stress_force` / `stress_acceleration` on the analytic field (`:141-143`) | - | `PoiseuillePlanar` | - | Machine-precision equilibrium (`06_known_issues:50`) |

## B.6 Other cases: compact rows (summary; see the subagent reports for full tables)

| Case | dim | Integrator / retopo | BCs | Status (evidence) |
|---|---|---|---|---|
| Hydrostatic_column (1D/2D/3D/2D_periodic) | 1-3 | Hand-rolled CFL loop `while t < t_end` (`Hydrostatic_2D.py:97-104`), `make_gravity_dudt(mu_art)`. The periodic variant uses the manual loop + periodic retopo | `NoSlipWallBC` (`src/_setup.py`) | Hydrostatic equilibrium verified. `test_case_hydrostatic.py` (TestHydrostaticEquilibrium1D/2D, PerturbationRecovery1D). Periodic variant exposed to C8 |
| bc_demo (+v1, v2, parametric) | 2 | Kinematic demo, `dt=0.05`, n=120 (`bc_demo.py:49-50`) | `PeriodicInletBC(period=L_period=1)`, `OutletDeleteBC(outlet_pos=L_domain)` (C5), `PositionalNoSlipWallBC` | Demo |
| cube_flow 1D/2D/3D | 1-3 | `euler(...)`, custom `dudt_fn(mu=0.1, G=0.001)`, dt 0.01, t_end 4 (`cube_flow_2D.py:100-155`) | `OutletDeleteBC(outlet_pos=L, axis=0)` (C5) + `PeriodicInletBC(u_inlet, period=L, auto cdist)` (`src/_setup.py:113-117`) | Demo |
| template (`template.py`, `example_features_demo.py`) | 2 | `euler_velocity_only`, dt 1e-3, n 500. `mu` passed as `**dudt_kwargs` (`template.py:147-170`), the pattern CLAUDE.md warns about | `NoSlipWallBC`, `DirichletVelocityBC`, `DirichletPressureBC` | Template |
| cube2droplet | 2/3 | `symplectic_euler(retopologize_fn=retopo_fn)` (`cube_to_droplet_2D.py:168-170`). The adaptive variant uses `remesh_mode='adaptive'` (C6) | `NoSlipWallBC`. `bc_comparison`: `PressureReservoirBC` / `AbsorbingPressureBC` / `ExpandingDomainBC` (no-op on walls, A.1) | Import bug `cases_dynamic.Cube2droplet` (`06_known_issues:30`, verify) |
| shearing_plate_droplet (2D/3D, `_run_short_*`) | 2/3 | `symplectic_euler(retopologize_fn=retopo_fn, remesh_mode=params[...])`. Custom `_retopo(HC, bV, dim, remesh_mode, remesh_kwargs)` (`src/_setup.py:349`) | `ShearingPlateBC` with wrap (A.1 collision hazard) | See subagent report |
| dynamic_caprise_tube | 3 | Hand-rolled `while t < t_final`, dt 0.05 (`Dynamic_caprise_3D_tube.py:112-132`) | case-local | Legacy / diagnostic |
| liquid_bridge_cfd_dem | 3 | `symplectic_euler` fluid substeps with `retopologize_fn` + `dem_step` (`liquid_bridge_cfd_dem_case.py:73-140`) | - | See subagent report |
| liquid_bridge_dem | - | `dem_step` only (`liquid_bridge_case.py:72`) | - | DEM. xfail bridge timing (`06_known_issues:65`) |
| liquid_bridge_approach, liquid_bridge_equilibrium | 3 | `_ddgclib_case_core.py` / Case_1-5 scripts | - | See subagent report |
| oscillating_droplet_p_ref | 2/3 | Standalone scripts (PR33 operators, F-Heron benchmarks) | - | Out of the main pipeline. See subagent report |

**diagnose_*.py classification.** Grep-level only. The subagent report has the full table. Referenced in docs: `diagnose_a5_bisection.py` and `diagnose_a5_step1_diff.py` (`06_known_issues:71,90,119-120`). The rest are flagged as scratch by `LIBRARY_AUDIT.md:178-180`.

# PART C - regression-locked configurations (direct grep)

| Test (file:line) | Config | Pin / tolerance | Marker |
|---|---|---|---|
| `TestStaticDroplet2DRetopologyFloor` `test_case_oscillating_droplet.py:147-262` | `setup_oscillating_droplet` 2D, default retopo_fn (bare Delaunay, redistribute ON, neighbour_count), u zeroed each step (A.5.b) | step0 max\|F\| **2.3748568e-03**, post-retopo **2.2716938e-03**, `REL_TOL=0.01`. Mass and volume drift < 1e-10 (`:166-168,205-262`) | slow |
| `TestStaticDroplet3DRetopologyFloor` `:271-460` | same, 3D, exact simplex volumes | step0 **6.0153e-05**, plateau **7.274172e-05**, rel 0.01 (`:335-342`) | slow |
| `TestDualOnlyRetopoPolicy2D` `:467-540` | 2D + `skip_triangulation=True` | mass < 1e-10, KE < 1e-5 | fast |
| `TestOscillationEnvelopeRegression2D` `:544-674` | **refine 2/2**, full t_end, runner default policy (delaunay_remap) | `L2_MAX=0.0600` (measured 0.05451425932968201), `TAIL_MAX=1.004` (measured 0.93689), mass < 1e-10 (`:556-590,662-674`) | fast (long) |
| `TestConservativeRetopoRemap2D` `:677-779` | remap='conservative' | dp_max < 1e-9 across rebuild. KE bounded. Arg validation | fast |
| `TestDelaunayRemapEndurance2D` `:781-860` | remap, 200 steps | mass < 1e-10, max KE < 1e-5 | fast |
| `TestProjectionCadence2D` `:863-1035` | projection_every validation. `projection_every=1` **bit-identical (`assertEqual` on sorted x, u, m, p_phase)** to default (`:924-946`). Off-call neutral at frozen positions (dp < 1e-9). dual_only + cadence. remap + cadence | exact equality / 1e-9 / 1e-10 | fast |
| `test_a5b_longrun_regression.py:33-44,96-120` | long-run floors | 2D peak 2.3749e-03 / end 2.2717e-03. 3D 7.274172e-05. rel 0.01 | 3D slow |
| `test_oscillation_score_3d.py` (~`:209`) | 3D score harness | see file (laneB 0.24811 is in the docstring) | - |
| `test_case_dam_break.py:53,96-135` | hydrostatic IC. 150-step smoke, alpha 0.5, remap ON | finite u, p; `|M1/M0-1| < 1e-12`; KE_liq > 2e-4; front advance > 5e-4 | - |
| `test_case_hagen_poiseuille.py`, `test_case_hydrostatic.py` | analytic profile / setup / equilibrium | see files | - |
| `test_case_oscillating_droplet_two_fluid.py:117-131` | two-fluid reference IVP | rtol 1e-12 to 1e-13 | fast |
| `test_retopo_displacement_gate.py` | `displacement_eps` gate (13 tests per `06_known_issues:13`) | - | - |
| `ddgclib/tests/test_methods.py` (**new, untracked**) | preset ↔ hand-written partial bit-identity (`TestMultiphaseBuilders` `:297-305`, `TestEffectiveMethods` `:455`) | bit-identity | - |

**Not test-pinned.** Several headline numbers exist only in `_params.py` comments, `baselines/*.json`, lane logs and `_presets.py` notes:
- full-run 2D l2 **0.17479**
- 3D l2 **0.24811**
- static_droplet_2D summary **1.1847e-03**
- the dam-break laneF numbers

The wrapper must reproduce them against the baseline JSONs, not against the tests.

**Coverage gaps.** Of the BC classes, `test_boundary_conditions.py` (`:43-269`) covers the helpers, NoSlip, DirichletVelocity, DirichletPressure, Neumann, OutletDelete, OutletBufferedDelete and BoundaryConditionSet. **No unit tests** cover PeriodicInletBC, PositionalNoSlipWallBC, ShearingPlateBC, MovingWallBC, the three pressure BCs, or MeshAdvancer. Of the IC classes, `test_initial_conditions.py` (`:70-247`) has no tests for DualVolumeMass, HydrostaticEOSMass, PhaseAssignment, MultiphaseMass or MultiphasePressure (the last three are exercised only indirectly in `test_multiphase.py`). `test_periodic.py` covers the periodic retopo.

**Implications for the wrapper.** It must preserve, bit-exactly:
- the `_retopologize_multiphase` defaults (`_int:481-488`)
- partial-bound kwargs taking precedence over integrator kwargs (`_int:436-461`)
- `split_method` identical at setup and runtime
- the floor tests' use of setup's bare-Delaunay retopo_fn (not the runner policy)
- `projection_every=1` ≡ absent
- 3D exact-volume vs 2D `batch_e_star` volume sourcing (`_int:262-275`)
