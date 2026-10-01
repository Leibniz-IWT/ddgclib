# METHODS.md — every method the dynamic solver can use, and which one each case uses

Central registry for the Lagrangian dynamics pipeline. Source of truth is
the code registry `ddgclib/methods/_axes.py`; sections 2 and 3 below are
GENERATED from it (`python -m ddgclib.methods --markdown`), and
`ddgclib/tests/test_methods.py` fails if this file drifts from the registry.
Section 4 (case matrix) is maintained by hand. Findings behind the status
column: `docs_temp/11_dynamics_audit_2026-09-25.md`.

## 1. How to use it

```python
from ddgclib.methods import SolverMethods, PRESETS, record_methods, effective_methods

# A shipped configuration (bit-identical to the runner it names):
methods = PRESETS['oscillating_droplet_2D']
print(methods.describe())                      # one line per axis with status

# Or spell one out; invalid / silently-ignored combinations raise ValueError:
methods = SolverMethods(dim=2, phases='multi', integrator='symplectic_euler',
                        connectivity='delaunay', remap='conservative',
                        redistribute_mass=True, projection_every=2)

# Build the runtime objects the integrators consume (same partials the
# cases used to build by hand):
dudt_fn  = methods.dudt_fn(HC, mps=mps, pressure_model=meos)         # or mu=... for single-phase
t_final  = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                             bc_set=bc_set, callback=cb, mps=mps)    # custom=fn for connectivity='custom'

# Record what ran, next to the score: requested config + the implicit
# (dimension / cache gated) choices resolved on the final mesh + git SHAs.
record_methods('results/methods.json', methods, HC, extra={'dt': dt, 'n_steps': n_steps})
```

Rules:

- **Setup and runtime must agree.** Pass `split_method=methods.split_method`
  and `redistribute_mass=methods.redistribute_mass` into the case setup
  helper (they are bound into `mps.refresh` at setup and into the retopology
  partial at runtime; a mismatch reproduces the rho x147 jump).
- **Explicit vs reported axes.** Explicit axes are `SolverMethods` fields.
  Reported axes are chosen by the code from dimension / cache presence and
  cannot be set today; `effective_methods(HC, dim)` tells you which value
  ran. Making one controllable is an operator-layer change, not a registry
  edit.
- **Status vocabulary.** `validated` = default of a pinned case;
  `opt-in` = tested, deliberately not default; `experimental` = no
  regression net; `measured-worse` = A/B'd and rejected (a DO-NOT unless the
  surrounding physics changed, which is exactly when to re-try it);
  `broken` = wrong in some regime (construction warns); `dead` = no caller.
- **Adding a method.** Register the value in `_axes.py` (summary, status,
  code anchor, evidence), wire it in the builder if it needs a new kwarg,
  add the preset if a case adopts it, regenerate this file.
- **Talking about a result.** Quote the preset name or the run's
  `methods.json`, never a bare policy string.

## 2. Method axes (generated; group headings follow)

## problem

### `mesh` — Mesh representation (reported)

Default: `complex`. Applied via: how the case builds HC (domain builders always build hyperct.Complex); reported from type(HC) and HC._SC

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `complex` | validated | hyperct.Complex vertex-vertex flag complex (v.nn sets) + raw top-simplex list HC._simplices | `hyperct/_complex.py:Complex` | the only representation any ddgclib code path uses |
| `simplicial` | broken | Complex(simplicial=True) with the hyperct SimplicialComplex / _ops index-array layer (branch wip/simplicial-layer, not on master) | `hyperct (branch wip/simplicial-layer: _simplicial.py, _ops/, Complex._simplices setter hooks)` | audit 2026-09-25: unused by ddgclib; 4 verified sync defects (retriangulation ignored, invalidate no-op, collapse holes, read side effects); parked on the branch 2026-09-25 |

### `phases` — Phase model (explicit)

Default: `single`. Applied via: selects dudt_i vs multiphase_dudt_i and _retopologize vs _retopologize_multiphase

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `single` | validated | One fluid; v.p from pressure_model or held; forces from operators.stress.stress_force | `ddgclib/operators/stress.py:stress_force` | Hagen-Poiseuille / hydrostatic machine-precision equilibria |
| `multi` | validated | Sharp-interface n-phase model on MultiphaseSystem; per-phase summed stress + surface tension | `ddgclib/operators/multiphase_stress.py:multiphase_stress_force` | oscillating droplet 2D/3D pins (test_case_oscillating_droplet.py) |

## time

### `integrator` — Time integrator (explicit)

Default: `symplectic_euler`. Applied via: ddgclib.dynamic_integrators.<name>(HC, bV, dudt_fn, ...)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `symplectic_euler` | validated | u += dt a; x += dt u_new (Lagrangian) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:symplectic_euler` | every pinned dynamic case |
| `euler` | opt-in | x += dt u_old; u += dt a (forward Euler, Lagrangian) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:euler` | cube_flow demos and the u=0 static floor tests only; laneK: with an EOS it GROWS at CFL 0.25 even on fixed connectivity (symplectic_euler is stable to dt c_s/dx = 1.5) |
| `rk45` | experimental | scipy RK45 per macro step; duals/edge cache frozen at macro-step start (stale within stages) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:rk45` | audit 2026-09-25 §1.2: no per-stage dual rebuild; skips BCs when no interior vertices |
| `euler_velocity_only` | opt-in | Eulerian fixed mesh, u += dt a only. Validation/equilibrium checks ONLY (CLAUDE.md) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:euler_velocity_only` | Poiseuille equilibrium tests |
| `euler_adaptive` | experimental | Advective-CFL adaptive dt; velocity_only=True by default (Eulerian), else forward Euler | `ddgclib/dynamic_integrators/_integrators_dynamic.py:euler_adaptive` | audit §1.3: no sound-speed term in the CFL; not used by any case |

## connectivity

### `connectivity` — Connectivity (retopology) policy (explicit)

Default: `delaunay`. Applied via: retopologize_fn / skip_triangulation / remesh_mode / periodic_axes on the integrator; bound into the multiphase retopo partial by SolverMethods.retopologize_fn()

Formerly spread over five switches and named differently per case (retopo_policy_2d "delaunay_remap", 3D CLI "delaunay|dual_only", dam break partial).

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `delaunay` | validated | Per-step global scipy Delaunay rebuild (connect_and_cache_simplices), boundary_from_simplices, compute_vd, dual volumes/edge-area cache | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize` | 2D droplet default WITH remap (laneE); bare (no remap) is measured-worse for multiphase: 2D l2 0.490 / 3D 1.524 (lane5, laneB). SINGLE-PHASE + EOS without remap: measured unstable in every laneK arm (a flip changes a dual volume by 33-100 %, read as 3e4-5e4 Pa; blows up at CFL 0.01, c_s 10 or 100, n 1 or 7.15, redistribution on or off); use remap=conservative (laneR) |
| `dual_only` (dims 2,3) | validated | skip_triangulation=True: keep builder connectivity, refresh v.boundary tags, duals, dual volumes (and per-phase split / redistribution / EOS for multiphase) every step | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize (skip_triangulation branch)` | 3D droplet default (laneB l2 0.24811); 2D opt-in (l2 0.17857, lane5). Cannot follow large deformation (dam break NaN-aborts, laneF). laneK: single-phase + EOS is stable here without any remap (pressure force = exact volume gradient to 1e-10; symplectic_euler stable to dt c_s/dx = 1.5) |
| `dual_only_bare` (dims 2,3) | validated | Frozen connectivity, boundary retagged from HC.boundary(), compute_vd + cache_dual_volumes (half-cell boundary volumes, no edge-area cache) + per-phase split; NO mps.refresh, NO redistribution, NO EOS update (pressure frozen at its setup value) | `ddgclib/methods/_retopo.py:bare_dual_refresh` | static_droplet_2D pin 1.1847162859108737e-03 (was the case-local _dual_only_retopo closure; audit F11: validates surface tension against a FROZEN pressure field) |
| `frozen` | opt-in | retopologize_fn=False: NO topology or dual refresh at all. Surface meshes / A.5.a static probes only | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize` | A.5.a frozen-mesh floors 2.3749e-3 (2D) / 6.0153e-05 (3D); multiphase p_phase is NEVER updated on this path; laneK: with an EOS the pressure is inert (dual volumes never refreshed), so a stable frozen run is NOT evidence of EOS stability |
| `adaptive` (dims 2) | opt-in | hyperct.remesh.adaptive_remesh local split/collapse/flip preserving the v.phase interface; 2D only | `hyperct/remesh/_driver.py:adaptive_remesh via _retopologize(remesh_mode="adaptive")` | lane4 upstream conservation fix; l2 0.326 at refine 3/3 (second-best after dual_only); no pinned case uses it |
| `periodic` (dims 2,3) | experimental | retopologize_periodic: wrap, ghost-cell Delaunay, min-image duals (2D only in stress.py). Multiphase: retopologize_multiphase_periodic adds mps.refresh + redistribution + EOS (no remap, no cadence) | `ddgclib/geometry/periodic.py:retopologize_periodic; ddgclib/methods/_retopo.py:retopologize_multiphase_periodic` | periodic path ignores skip_triangulation/remesh/backend; shearing_plate_droplet 2D is unstable (interface lost by t=0.044 s), 3D crashes in setup; domain_bounds is a build-time argument (geometry), periodic_axes the method field |
| `custom` | experimental | User-supplied retopologize_fn callable (e.g. static_droplet_2D bare dual-only, Hagen_Poiseuile_3D cylinder) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize (callable branch)` | recorded by label only; kwargs forwarded by declared name |

### `remap` — Conservative retopology remap (explicit)

Default: `None`. Applied via: retopo_remap= in _retopologize_multiphase (multiphase partial) or in _retopologize (single-phase partial built by SolverMethods.retopologize_fn)

Same axis, two implementations. Multiphase: projection of the PRE-call field (cadence on projection_every). Single-phase: fresh snapshot, so the step's compression survives and there is no cadence to choose.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | No remap: reconnection changes dual volumes, EOS reads them as compression | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize_multiphase` | required value under dual_only (remap is a silent no-op there) |
| `conservative` (dims 2,3) | validated | Pressure field invariant across the rebuild. MULTIPHASE: stage-1 dual refresh on OLD connectivity, rebuild, per-phase redistribution, vol_corr gauge, restore_pressure_multiphase, anchor_phase_pressure_levels. SINGLE-PHASE: fresh snapshot eos(m / V_old-connectivity) at the current positions, rebuild, re-target EVERY vertex with a dual volume (frozen walls included), one exact mass rescale | `ddgclib/operators/mass_redistribution.py:restore_pressure_multiphase, anchor_phase_pressure_levels (multi); snapshot_pressure_fresh, redistribute_mass_single_phase(include_frozen=True) via _retopologize(retopo_remap=) (single)` | 2D droplet default (laneD/E, l2 0.17479); dam break survives reconnection (laneF); 3D multiphase measured-worse (laneE l2 1.873 vs 0.248, DO-NOT; confirmed cache-free by laneI). SINGLE-PHASE (laneK prototype, laneR library): box + EOS stable where bare Delaunay blows up at any CFL / c_s / n; final KE within 0.2 % of dual_only; pinned by test_single_phase_remap.py. Interior-only or stale-v.p variants are measured DO-NOTs (laneK P2, P3) |

### `projection_every` — Pressure-projection cadence (explicit, multiphase only)

Default: `1`. Applied via: projection_every= in _retopologize_multiphase (partial); counter on mps._projection_call_idx

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `1` | validated | Every call: per-phase masses re-targeted to the PRE-call pressure snapshot | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize_multiphase` | all pins; attributed cause of the 2D over-decay (laneH) and half the 3D bump (laneG) |
| `N>1` | opt-in | Project every N-th call; off-cadence remap snapshots are strain-advanced (evolve_snapshot_local_strain) | `ddgclib/operators/mass_redistribution.py:evolve_snapshot_local_strain` | laneH: delaunay+remap N=2 gives l2 0.03796 (-78%) and matches the two-fluid reference to ~2%; tail gate uncalibrated; 3D untested; forbidden under bare delaunay (ValueError) |

### `displacement_eps` — Skip-retopology displacement gate (explicit)

Default: `None`. Applied via: displacement_eps= integrator kwarg (_do_retopologize)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | Retopologize every step | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize` |  |
| `eps>0` | measured-worse | Skip when every vertex moved < eps since the last call; first call always skips | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_displacement_gate_should_skip` | lane5 sweep: eps in {0.01,0.05,0.2} h_min all worse than either extreme (cases_dynamic/oscillating_droplet/src/_params.py) |

### `merge_cdist` — Pre-retopology vertex merge (explicit)

Default: `None`. Applied via: merge_cdist= integrator kwarg (forwarded to _retopologize)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | No merge | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize` |  |
| `cdist>0` | experimental | HC.V.merge_all(cdist) before Delaunay; NOT mass-conserving and does not merge m_phase | `hyperct/_vertex.py:merge_all; ddgclib/multiphase.py:mass_conserving_merge is the separate conserving path` | DO-NOT wire into multiphase without per-phase ledger (laneF) |

## thermodynamics

### `redistribute_mass` — Pressure-preserving mass redistribution (explicit)

Default: `False`. Applied via: redistribute_mass= (integrator kwarg for single-phase, partial for multiphase); single-phase also needs pressure_model

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `False` | opt-in | Lagrangian masses held against the new duals | `ddgclib/dynamic_integrators/_integrators_dynamic.py` | multiphase noredist rings acoustically: 2D l2 0.495 (laneH); 3D l2 0.266 vs 0.248 but cleaner channels (laneG) |
| `True` | validated | After each rebuild rescale masses so the pre-rebuild pressure field is reproduced, exact total per phase | `ddgclib/operators/mass_redistribution.py:redistribute_mass_multiphase / redistribute_mass_single_phase` | every shipped multiphase setup binds True (M1 rollout); 3D A.5.b 1.44e-3 -> 7.38e-05 (Phase 2c). SINGLE-PHASE: on its own it is NOT a cure for Delaunay + EOS (laneK: stale v.p snapshot, frozen walls skipped, so wall flip jumps survive); combine it with remap=conservative (laneR) |

### `split_method` — Per-phase dual-volume split at interface vertices (explicit, multiphase only)

Default: `neighbour_count`. Applied via: split_method= in mps.refresh (setup) AND in the retopo partial; the two MUST match

A typo silently falls back to neighbour_count (multiphase.py tests only == "exact"). SolverMethods validates the key.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `neighbour_count` | validated | Interface vertex: fraction of 1-ring bulk neighbours per phase times v.dual_vol | `ddgclib/multiphase.py:MultiphaseSystem.split_dual_volumes` | all pins |
| `exact` (dims 2,3) | measured-worse | 2D: clip the barycentric dual polygon by the interface polyline (NOT rescaled to v.dual_vol); 3D: PCA tangent plane clip of the dual polyhedron, rescaled to v.dual_vol | `ddgclib/geometry/_dual_split_2d.py:split_dual_polygon_2d / split_dual_polyhedron_3d` | 3D end-to-end 1.75e-3 vs 1.44e-3 (worse); 2D retopo-neutral but no metric gain (debugging_plan 2026-04-29) |

### `density_diffusion` — Gradient-corrected density diffusion (explicit, singlephase only)

Default: `None`. Applied via: density_diffusion= integrator kwarg (euler, symplectic_euler); needs pressure_model=EOS; case loops call operators.stabilisation.density_diffusion_step directly

Added 2026-09-26. delta-SPH type mass flux on the dual faces; the cure for the checkerboard density mode that the centred pressure flux cannot see. Not available on rk45 / euler_adaptive.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | No density diffusion | `ddgclib/dynamic_integrators/_integrators_dynamic.py:symplectic_euler` | every pinned case |
| `delta>0` | opt-in | dm_i/dt = sum_j delta c0 |A_ij| [(rho_j - rho_i) - 1/2 (grad rho_i + grad rho_j).d_ij]: exactly mass conserving, exactly zero on linear density fields (interior), no shear viscosity; explicit stability delta < ~0.3 at acoustic CFL 0.4 | `ddgclib/operators/stabilisation.py:density_diffusion_step` | capillary_rise dynCA smoke (water R 0.5 mm, after the corner fix): L2 0.26 -> 0.075 (delta 0.05) / 0.13 (0.1); 0.2-0.3 over-smooth (capillary_rise_energy_grad README Section 5) |

## forces

### `curvature_path` — Interface curvature / surface-tension stencil (explicit, multiphase only)

Default: `integrated`. Applied via: curvature_path= on multiphase_dudt_i (dudt partial)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `integrated` (dims 2,3) | validated | 2D: exact piecewise-linear FTC gamma*(t_next - t_prev) (surface_tension_force_2d); 3D: cotangent/Heron hndA_i_interface on the interface sub-mesh | `ddgclib/operators/multiphase_stress.py:_interface_surface_tension` | all pins. The 3D apex cache HC._interface_edge_to_apex was never invalidated (audit 2026-09-25 T1); FIXED in laneI (cleared when the interface triangle set changes, test_interface_cache_invalidation.py). laneI also showed the droplet runs never flip interface triangles, so every pinned 3D score (dual_only, delaunay, delaunay+remap) is bit-identical before/after the fix |
| `stokes` (dims 2,3) | experimental | 3D conormal boundary integral on the barycentric dual (integrated_hndA_i_interface); 2D aliases "integrated" | `ddgclib/_curvatures_heron.py:integrated_hndA_i_interface` | bit-identical to integrated on a STATIC mesh (Probe 2). The coordinate-keyed cache HC._interface_x_to_v that made the force vanish after the first vertex move (audit T2) is cleared on every interface refresh since laneI (tested); no dynamic A/B or regression pin yet |
| `csf_dual` (dims 2,3) | experimental | Magnitude of the integrated stencil redirected along the dual-face normal S_inner | `ddgclib/operators/multiphase_stress.py:_csf_dual_surface_tension` | A/B probe only, no tests |

### `pressure_flux` — Pressure flux across the dual faces (explicit, singlephase only)

Default: `centred`. Applied via: pressure_flux= on dudt_i / stress_force (dudt partial); registry operators.stress.pressure_flux_methods

Added 2026-09-26 (capillary_rise_energy_grad). The multiphase force still hard-codes the centred flux.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `centred` | validated | Face-average -1/2 (p_i + p_j) A_ij: exact volume gradient at uniform p (linear precision), no dissipation; BLIND to the checkerboard density/pressure mode (the face average of an alternating field is uniform) | `ddgclib/operators/stress.py:pressure_flux` | every pinned case; capillary_rise static check: half-cell closure 4e-15 (run_free_surface_static_check.py) |
| `acoustic-riemann` | measured-worse | Lagrangian Godunov contact pressure p* = 1/2 (p_i + p_j) - 1/2 rho_f c_f (u_j - u_i).n: momentum conserving, zero for rigid translation, damps normal velocity jumps; numerical bulk viscosity ~ rho c |d| on compressive modes (low-Mach caveat). Needs an EOS pressure_model | `ddgclib/operators/stress.py:pressure_flux_riemann` | capillary_rise dynCA A/B 2026-09-26 (energy_grad README 5.6b): with c_s = 10 u_ref the numerical viscosity rho c dx ~ 0.5 Pa s is 700x mu; the column barely flows (smoke L2 0.53 vs 0.075 centred + density diffusion; quiescent column drains as with centred but the driven rise is lost). Correct for acoustic velocity noise, wrong tool at low Mach; the density-diffusion axis is the one to use. Unit tests: antisymmetry, rigid translation, dissipativity (test_pressure_flux_stabilisation.py) |

## dual geometry (reported)

### `dual_method` — Dual vertex construction (reported)

Default: `barycentric`. Applied via: hard-coded compute_vd(HC, method="barycentric") in _retopologize

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `barycentric` | validated | Dual vertices at simplex barycentres | `hyperct/ddg/_compute_dual.py:compute_vd` |  |
| `circumcentric` | opt-in | Dual vertices at circumcentres (benchmarks only; linear precision lost on jittered meshes) | `hyperct/ddg/_compute_dual.py:compute_vd` | INTEGRATED_BENCHMARKS.md |

### `dual_path` — Dual construction path (reported)

Default: `simplex_aware`. Applied via: presence of HC._simplices (connect_and_cache_simplices)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `simplex_aware` | validated | Top-simplex cache drives compute_vd, boundary_from_simplices, exact volumes | `hyperct/ddg/_retriangulation.py:connect_and_cache_simplices` | commit 8321c71; test_simplex_aware_duals.py |
| `nn_walk` | dead | Legacy 1-skeleton (v.nn intersection) walk; ghost K_{d+1} cliques on Delaunay meshes | `hyperct/ddg/_compute_dual.py (legacy branch)` | docs/3d_simplex_aware_dual_fix.md |

### `dual_volume` — Dual cell volume source (reported)

Default: `simplex_exact`. Applied via: dim + HC._simplices + HC._vd_method + whether batch_e_star runs (stress.py:_use_exact_barycentric_volume, _retopologize step 5b)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `simplex_exact` | validated | Vol_i = (1/(d+1)) sum_{T contains i} |T| (hyperct.ddg.simplex_dual_volumes / vertex_dual_volume) | `hyperct/ddg/_dual_volume.py` | 3D switch ON 2026-07-29 (laneA), floor re-pinned 7.274172e-05 |
| `fan_walk_3d` (dims 3) | measured-worse | batch_e_star / v_star tetra fan sum; undercounts 1-4% interior, ~20% boundary | `hyperct/ddg/_operators.py:batch_e_star(compute_volumes=True)` | docs_temp/audit/dual-volume-3d.md |
| `dual_cell_area_2d` (dims 2) | broken | Shoelace area of the 2D dual polygon (circumcentric or no simplex cache, i.e. every 2D mesh at SETUP before the first retopology) | `hyperct/ddg/_dual_cell.py:dual_cell_area_2d` | laneK: undercounts the four corner cells 4x (rectangle total 0.96875 instead of 1.0 = the old "2-4 % single-phase volume leak") and credits a moving free-surface vertex with 1/4 of its own volume change (0.0104 vs exact 0.0417): Hydrostatic_2D grows exponentially from roundoff with 0 flips, and decays 8 orders with exact simplex volumes. Populate HC._simplices at setup (lane S) |
| `interval_1d` (dims 1) | validated | Distance between the two dual vertices | `ddgclib/operators/stress.py:dual_volume` |  |

### `boundary_dual_vol` — Boundary-vertex dual volume convention (reported)

Default: `half_cell`. Applied via: dim: 3D retopology zeroes boundary dual_vol (batch_e_star path); 1D/2D/periodic/setup keep the truncated half cell

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `zeroed` (dims 3) | validated | v.dual_vol = 0 on every vertex in bV after retopology | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize step 5b` | measured 2026-09-25: 3D box boundary dual_vol 0.0 |
| `half_cell` (dims 1,2) | validated | Boundary vertices keep the truncated dual cell (cache_dual_volumes path) | `ddgclib/operators/stress.py:cache_dual_volumes` | measured 2026-09-25: 2D rectangle max boundary dual_vol 0.0156, total 1.0 |

### `edge_area_source` — Oriented dual face area A_ij source (reported)

Default: `shared_vd_2d`. Applied via: dim + HC._edge_area_cache + HC._periodic_axes (stress.py:stress_force, dual_area_vector)

laneJ (2026-09-25) recommends making this an EXPLICIT 3D axis (keys e_star_cache | p_ij | p_ij_simplex) once p_ij_simplex has a vectorised hyperct kernel; until then it stays reported. Forcing p_ij today = connectivity="custom" wrapper that clears HC._edge_area_cache after each retopology (cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py).

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `batch_e_star_cache` (dims 3) | validated | 3D: cached e_star fan areas from batch_e_star(orient=True) at the last retopology | `hyperct/ddg/_operators.py:batch_e_star` | all 3D pins. NOT linearly precise: laneJ measured per-edge difference to p_ij median 0.125 / max 0.625 (box), closure residual ~1 % at every droplet interface vertex, linear-precision error 3-25 %; it carries ~79 % of the 3D static floor retopology excess (7.274172e-05 vs 6.2839e-05 on p_ij, frozen floor 6.0153e-05) and part of the dynamic outward bump (final inflation 1.87 % -> 0.82 % R0 on p_ij) - but p_ij alone scores l2 0.28713 vs 0.24811 (laneG cancellation exposed), so no flip |
| `p_ij_ring_3d` (dims 3) | opt-in | 3D DEC p_ij dual polygon ring walk (linearly precise on box/ball, 1e-18); used only when no cache exists | `ddgclib/operators/stress.py:_dual_area_vector_3d_p_ij` | test_stress.py p_ij linear-precision tests. laneJ: the face vertex is chosen as the common neighbour nearest the midpoint of two tet barycentres; on 743 of 5193 directed droplet edges that picks a non-face vertex, giving the 2.6 % closure residuals laneG attributed to the pressure side. 4.1x wall cost (2.05 vs 0.50 s/step) |
| `p_ij_simplex` (dims 3) | experimental | p_ij polygon with ring order and face vertices read from HC._simplices (exact faces): closure 2.3e-16, linear precision 1.9e-15 at every interior vertex | `cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py (driver only, not in the library yet)` | laneJ: static floor 6.28386e-05, dynamic l2 0.28653 / tail 0.08818, 1.89 s/step. Target default after a vectorised hyperct kernel, 3D re-pin and co-evaluation with the redistribution-pump rework (laneG lever b) |
| `shared_vd_2d` (dims 2) | validated | 2D: segment between the two dual vertices shared by v_i and v_j, oriented outward | `ddgclib/operators/stress.py:dual_area_vector (2D branch)` | all 2D pins (batch_e_star raises for dim != 3) |
| `min_image_2d` (dims 2) | experimental | 2D periodic: minimum-image rebuild of the dual segment | `ddgclib/operators/stress.py:dual_area_vector (periodic branch)` | d_ij is NOT min-imaged (06_known_issues) |

### `boundary_rule` — Topological boundary detection (reported)

Default: `boundary_from_simplices`. Applied via: HC._simplices present -> boundary_from_simplices, else HC.boundary(); dual_only carries the previous bV

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `boundary_from_simplices` | validated | Faces belonging to exactly one top simplex | `hyperct/ddg/_boundary.py:boundary_from_simplices` |  |
| `HC.boundary` | opt-in | Legacy hyperct vertex-hull test | `hyperct/_complex.py:Complex.boundary` |  |
| `carried_bV` | validated | skip_triangulation: reuse the previous (possibly filtered) bV; 3D failed fans are promoted and stay boundary | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize` |  |

## execution

### `backend` — batch_e_star compute backend (explicit)

Default: `None`. Applied via: backend= integrator kwarg (only reaches batch_e_star; compute_vd always runs numpy)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | numpy | `hyperct/_backend.py` |  |
| `torch` | opt-in | PyTorch CPU tensors | `hyperct/_backend.py` | test_gpu_backend.py |
| `gpu` | opt-in | PyTorch CUDA (auto-detect) | `hyperct/_backend.py` |  |
| `multiprocessing` | experimental | parallel CPU | `hyperct/_backend.py` |  |

### `workers` — dudt evaluation workers (explicit)

Default: `None`. Applied via: workers= integrator kwarg (_compute_accel fork pool)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | Sequential | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_compute_accel` |  |
| `n>1` | experimental | fork pool over dudt_fn (Linux only). Safe when the force has no side effects (pressure_model=None); with an EOS bound into dudt_fn the v.p / v.rho writes of _resolve_pressure happen in the children and are LOST in the parent | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_compute_accel` | audit 2026-09-25 §0.5; used by Hagen_Poiseuile 2D (20) / 3D (8) where pressure_model is None |

## 3. Presets (generated)

Same in every preset: `displacement_eps=None`, `merge_cdist=None`, `split_method='neighbour_count'`, `curvature_path='integrated'`, `pressure_flux='centred'`, `density_diffusion=None`, `backend=None`

| preset | `dim` | `phases` | `integrator` | `connectivity` | `remap` | `projection_every` | `redistribute_mass` | `workers` | source |
|---|---|---|---|---|---|---|---|---|---|
| `oscillating_droplet_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `1` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap') |
| `oscillating_droplet_2D_dual_only` | `2` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='dual_only') |
| `oscillating_droplet_2D_bare_delaunay` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay') |
| `oscillating_droplet_2D_projection2` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `2` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap_p2') |
| `static_droplet_floor_2D` | `2` | `multi` | `euler` | `delaunay` | `None` | `1` | `True` | `None` | ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet2DRetopologyFloor (u=0 every step) |
| `static_droplet_2D` | `2` | `multi` | `symplectic_euler` | `dual_only_bare` | `None` | `1` | `False` | `None` | cases_dynamic/oscillating_droplet/static_droplet_2D.py |
| `oscillating_droplet_3D` | `3` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py (retopo_policy_3d='dual_only') |
| `oscillating_droplet_3D_delaunay` | `3` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py --retopo delaunay |
| `static_droplet_floor_3D` | `3` | `multi` | `euler` | `delaunay` | `None` | `1` | `True` | `None` | ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet3DRetopologyFloor (u=0 every step) |
| `dam_break_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `1` | `True` | `None` | cases_dynamic/dam_break/dam_break_2D.py |
| `dam_break_3D` | `3` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `True` | `None` | cases_dynamic/dam_break/dam_break_3D.py |
| `electrolysis_bubble_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `True` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py |
| `electrolysis_bubble_3D` | `3` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `True` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_3D.py |
| `electrolysis_bubble_fritz_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `False` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_fritz_2D.py (run_short_dynamics) |
| `shearing_plate_droplet_2D` | `2` | `multi` | `symplectic_euler` | `periodic` | `None` | `1` | `True` | `None` | cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_2D.py, _run_short_2D.py |
| `shearing_plate_droplet_3D` | `3` | `multi` | `symplectic_euler` | `periodic` | `None` | `1` | `True` | `None` | cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_3D.py, _run_short_3D.py |
| `hagen_poiseuille_2D` | `2` | `single` | `symplectic_euler` | `delaunay` | `None` | `1` | `False` | `20` | cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py |
| `hagen_poiseuille_2D_eulerian` | `2` | `single` | `euler_velocity_only` | `delaunay` | `None` | `1` | `False` | `None` | cases_dynamic/Hagen_Poiseuile_2D_Eulerian/Hagen_Poiseuile_2D_Eulerian.py |
| `hagen_poiseuille_3D` | `3` | `single` | `symplectic_euler` | `custom` | `None` | `1` | `False` | `8` | cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py (retopologize_cylinder) |

## 4. Case → configuration matrix (hand-maintained, 2026-09-25)

Preset = consumed by the runner and proven bit-identical in
`test_methods.py`. "hand-rolled" = the runner never calls a library
integrator, so no preset can describe it faithfully. Numbers are the
pinned values at the campaign close (2026-07-30) unless noted. Per-case
detail: `docs_temp/audit_2026-09-25/bcs_and_cases.md` and
`docs_temp/11_dynamics_audit_2026-09-25.md` §3.

| case (runner) | preset / configuration | extra (non-method) choices that matter | status | pinned |
|---|---|---|---|---|
| `oscillating_droplet/oscillating_droplet_2D.py` | `retopo_policy_2d` → `oscillating_droplet_2D` (delaunay + conservative remap); `..._dual_only`; `..._bare_delaunay`; `..._projection2` | refine 3/3, Tait n=7.15 clip (0.8,1.2), c_s 1 m/s (K_d 800 / K_o 1000), analytic YL preload, `mass_conserving_merge(1e-10)` at setup, CFL dt `min(0.25 dx/c_s, 0.5 sqrt(rho dx^3/gamma))` | validated | l2 0.17479361640597058 / tail 0.9998967874595965 (`baselines/baseline_oscillation.json`); p2 opt-in l2 0.03796 |
| `oscillating_droplet/oscillating_droplet_3D.py` | `retopo_policy_3d` → `oscillating_droplet_3D` (dual_only); `--retopo delaunay` → `..._3D_delaunay` | refine 2/2 hard-coded; same EOS/preload | validated (score is a bump/over-decay cancellation, laneG) | l2 0.24811340819647862 (`baselines/baseline_oscillation_3d.json`); delaunay 1.52446 |
| `oscillating_droplet/static_droplet_2D.py` | `static_droplet_2D` (connectivity=`dual_only_bare`: library `bare_dual_refresh`, pressure never updated) | eps 0, 100 steps | validated | summary 1.1847162859108737e-03, mass 0.0 (reproduced through the library path 2026-09-25) |
| `tests/test_case_oscillating_droplet.py` floors | `static_droplet_floor_2D` / `_3D` (euler, u=0, bare delaunay) | refine 3/3 (2D) / 2/2 (3D) | regression-locked (1 %) | 2.3748568e-03 / 2.2716938e-03; 6.0153e-05 / 7.274172e-05 |
| `tests/…::TestOscillationEnvelopeRegression2D` | mirror of `oscillating_droplet_2D` at refine 2/2 | full t_end 0.1143 s, 267 steps | fast pin (~9 s) | l2 < 0.0600 (measured 0.054514), tail < 1.004 |
| `oscillating_droplet/oscillating_droplet_2D_adaptive.py` | bare setup partial (ignores `retopo_policy_2d`) + `remesh_mode='adaptive'` {alpha 0.3/2.5, q20, 1 iter, no smoothing} | refine 2/2, 200 steps | runs, poor | delaunay l2 1.48 / adaptive 2.73 |
| `oscillating_droplet/oscillating_droplet_2D_mass_redist.py`, `mesh_convergence_2D.py` | own partials (no remap) / dt 1e-3 = 16x CFL | — | stale (April snapshots) | — |
| `dam_break/dam_break_2D.py` | `dam_break_2D` (delaunay + conservative remap, redistribute) | hydrostatic per-phase mass preload, `col_h = a` headspace, alpha_art 0.3 baked into `PhaseProperties.mu`, gravity closure, linear Tait, cfl 0.1 | runs full horizon (laneF); blocker: air sliver-cell F/m ejection | KE_liq peak 1.0369e-3 J @ 0.0506 s, mass 6.2e-15 |
| `dam_break/dam_break_3D.py` | `dam_break_3D` (dual_only) | as above | smoke only | — |
| `Hagen_Poiseuile/Hagen_Poiseuile_2D.py` | `hagen_poiseuille_2D` (single, symplectic, delaunay, workers 20) + `boundary_filter=walls` at build time; `PeriodicInletBC(cdist=1e-10)` + `OutletBufferedDeleteBC` + `PositionalNoSlipWallBC` | plug IC, point-value pressure on the ghost | wall collapse at step ~1248 (hull re-tagging, audit C1); not re-run through the preset (long, blocking plot calls) | — |
| `Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py` | `hagen_poiseuille_3D` (single, `custom` = `retopologize_cylinder`, never updates `HC._simplices`; workers from CLI) | — | stalled (inlet cap frozen, audit M2); not re-run | — |
| `Hagen_Poiseuile_2D_Eulerian/` | `hagen_poiseuille_2D_eulerian` (`euler_velocity_only`, default retopo) | point-value pressure IC | runs, no pin | — |
| `Hagen_Poiseuile_equilibrium/` | static residual evaluation, no integrator | — | validated equilibrium | median residual < 1e-13 |
| `Hydrostatic_column/*` (1D/2D/3D/2D_periodic) | hand-rolled symplectic loop, `_recompute_duals` + `cache_dual_volumes`, no retagging, `HydrostaticEOSMass` (point-value p) | mu_art 0.5 rho c0 dx | 1D decaying, 2D unstable (|u| → 10 c0), 3D stalls, periodic aborts | — |
| `capillary_rise/capillary_rise_2D.py`, `_3D.py` | hand-rolled, `_recompute_duals`, static-angle Washburn body force | — | scaffold | — |
| `capillary_rise/capillary_rise_2D_dynCA.py`, `_3D_dynCA.py` | hand-rolled: `mass_conserving_merge` + `adaptive_remesh` every few steps + `_recompute_duals` + density-continuity repair; data-driven CA(t) forcing or `--ca-model` (energy-gradient contact line, `capillary_rise_energy_grad`); opt-in `--surface-bc yl`, `--contact-mode mobility`, `--delta-diff` (= `density_diffusion`), `--pressure-flux` | wall film, contact line, free-surface line tension | 2026-09-25/26: the stall was the orphan cleanup deleting the strip's top corners (fixed); with YL-BC + mobility + `delta` 0.05 the smoke window scores L2 0.04 (was 0.72), 50 ms window within 2 % at 40 ms then −23 % (contact-line attachment) | smoke L2 0.040, 50 ms L2 0.12 |
| `electrolysis_bubble/electrolysis_bubble_2D.py`, `_3D.py` | `electrolysis_bubble_2D` / `_3D` (multi, delaunay, no remap, redistribute) | gravity + NaN guard in the setup dudt wrapper, `WallClampBC`, gas mass injection in the callback | 2D unvalidated; 3D unstable (gas phase lost by t≈1.1e-4 s) | 5-step mass drift ≤ 1.94e-15 |
| `electrolysis_bubble/electrolysis_bubble_fritz_2D.py` | `electrolysis_bubble_fritz_2D` (multi, delaunay, redistribute **False**: the historic partial left it unbound) | 80-step smoke | runs | — |
| `shearing_plate_droplet/*` (2D, 3D, `_run_short_*`) | `shearing_plate_droplet_2D` / `_3D` (multi, `periodic` = library `retopologize_multiphase_periodic`, no remap / cadence; `domain_bounds` from the setup params at build time), `ShearingPlateBC` | anisotropic outer rescale pushes vertices into the droplet | 2D unstable (interface lost t≈0.044 s), 3D crashes in setup | — |
| `cube2droplet/*` | `**kwargs` retopo closure | — | broken import (`Cube2droplet`) | — |
| `cube_flow/*`, `bc_demo*`, `template/*` | mock dudt, default retopo | — | demos; cube_flow stalls after step 0 | — |
| `liquid_bridge_equilibrium/*`, `liquid_bridge_cfd_dem/*` | `retopologize_fn=False` (frozen surface mesh) | — | runs | — |
| `liquid_bridge_approach/*` | own semi-implicit loop | — | broken import | — |

## 5. Where each explicit axis lands at runtime

| axis | single-phase | multiphase |
|---|---|---|
| `integrator` | `ddgclib.dynamic_integrators.<name>` | same |
| `connectivity` | `retopologize_fn` (None / False / custom) + integrator `skip_triangulation` / `remesh_mode` / `periodic_axes` | bound into `partial(_retopologize_multiphase, ...)`: `skip_triangulation=True` for dual_only; `remesh_mode` stays integrator-level (inverted precedence, audit F5) |
| `remap`, `projection_every`, `split_method`, `redistribute_mass` | `redistribute_mass` + `pressure_model` at integrator level | bound into the multiphase partial |
| `curvature_path` | n/a | bound into `partial(multiphase_dudt_i, ...)` |
| `pressure_flux` | bound into `partial(dudt_i, ..., pressure_flux=)` only when not `centred` (needs an EOS); registry `operators.stress.pressure_flux_methods` | n/a (multiphase force hard-codes the centred flux) |
| `density_diffusion` | integrator kwarg (`euler`, `symplectic_euler`; needs `pressure_model=EOS`); operator `operators.stabilisation.density_diffusion_step` | n/a |
| `displacement_eps`, `merge_cdist`, `backend`, `workers` | integrator kwargs | integrator kwargs (forwarded by name into the partial) |

## 6. Regenerating

```bash
python -m ddgclib.methods --update METHODS.md                       # rewrite sections 2-3 in place
python -m ddgclib.methods --markdown                                # print them instead
python -m ddgclib.methods --preset oscillating_droplet_2D           # describe one preset
pytest ddgclib/tests/test_methods.py -q                             # incl. the drift check
```
