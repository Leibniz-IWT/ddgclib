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
| `single` | validated | One fluid; v.p from pressure_model or held; forces from operators.stress.stress_force | `ddgclib/operators/stress.py:stress_force` | Hagen-Poiseuille / hydrostatic machine-precision equilibria (for NODAL pressures: with dual-cell averages the wall half cells break the linear precision, 2D static residual 1.9075 m/s^2 at every refinement, laneP). laneP: with an EOS and gravity the discrete hydrostatic equilibrium is a saddle of the discrete energy: slow modes grow at 1.13 g/c0 (0.353 1/s on the 2D column; the same in a closed box, where the stiffness matrix is symmetric to 5e-10; 1.01 g/c0 at refinement 2, so it does not refine away), followed in a real run (amplitude x4.816 at 200 acoustic times, cosh(sigma t) 4.816). Viscosity turns the growth into creep (rate ~ 1/mu: 1.2e-4 1/s at mu = 0.5 rho c0 dx); with the viscosity of water the no-slip drop exceeds c0 at 64 t_ac. FREE SURFACE: the open-fan force is not an energy gradient (stiffness asymmetry 11 to 16 %, closed fans symmetric to 5e-10); without viscosity the no-slip column flutters at refinement 2 (0.75 1/s, x61 in a real run against x59 predicted) and 4 (1.22 1/s), not at 3 and never with a closed lid; 0.05 rho c0 dx of viscosity removes it (cases_dynamic/Hydrostatic_column/diagnose_column.py) |
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
| `delaunay` | validated | Per-step global scipy Delaunay rebuild (connect_and_cache_simplices), boundary_from_simplices, compute_vd, dual volumes/edge-area cache | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize` | 2D droplet default WITH remap (laneE); bare (no remap) is measured-worse for multiphase: 2D l2 0.490 / 3D 1.524 (lane5, laneB). SINGLE-PHASE + EOS without remap: measured unstable in every laneK arm (a flip changes a dual volume by 33-100 %, read as 3e4-5e4 Pa; blows up at CFL 0.01, c_s 10 or 100, n 1 or 7.15, redistribution on or off); use remap=conservative (laneR). FREE SURFACE (laneP): the rebuild triangulates the convex hull of the cloud, so the gap between a moved free surface and the hull is filled with near-degenerate simplices. Even WITH remap=conservative the 2D hydrostatic column reaches 42 m/s at 3 t_ac, the total volume stays pinned to the hull and the column carries half the hydrostatic head (integrated L2 4.9e3 Pa = rho g H / 2); from the equilibrium masses it exceeds c0 at 95 t_ac; 3D 7.1 m/s. Use delaunay_material. In 1D the rebuild is the sorted chain and equals a non-reconnecting loop to round-off (hydrostatic_1D preset) |
| `delaunay_material` (dims 2,3) | opt-in | Per-step Delaunay rebuild that keeps the fluid domain: the boundary of the previous connectivity is material, the simplices Delaunay adds between a free surface and the convex hull are removed again (geometric test: winding number of the simplex centroid about the old boundary; exposed flat wall simplices are dropped). 2D: domain kept to round-off. 3D: kept up to the slivers of free-surface diagonal flips (no facet recovery); the relative volume change is returned and warned above domain_tol. Half-cell boundary volumes, no edge-area cache (2D shared dual vertices, 3D p_ij ring). Single phase; with remap=conservative it runs the single-phase remap around the rebuild | `ddgclib/methods/_retopo.py:retopologize_material_delaunay` | laneP 2026-10-01, with remap=conservative on the hydrostatic column (free surface). 2D: 200 t_ac from uniform density max|u| envelope 1.24e-4 m/s (dual_only 1.14e-4), from the equilibrium masses bounded at a reconnection noise floor of 2.3e-6 m/s (dual_only decays to 4.2e-9); mass drift below 3e-14; largest offset K (s - 1) of the mass rescale 0.99 Pa; domain volume change 0 to round-off in every call. 3D (189 vertices): 100 t_ac envelope 5.6e-4 to 5.7e-4 m/s in two processes (dual_only_bare 2.4e-4), from the equilibrium masses a floor of 9.9e-7 (dual_only_bare 5.4e-7); domain change at most 1.2e-6 per call (the four corner squares at the first rebuild), 1.6e-6 summed over 185 calls, offset 2.4 Pa; after about 10 t_ac the 3D drop run is reproducible between processes to 2 digits only (cospherical mesh, Delaunay ties). A 0.004 bowl pushed into the builder surface in one go changes the 3D domain by 9.8e-5. Interior integrated pressure error (2D) 36 to 42 Pa against 0.23 Pa on the builder mesh: that is the offset between cell centroid and vertex on the Delaunay cells (attribution run: 31.3 Pa integrated, rho g x rms offset 31.1 Pa, error against the nodal value 0.20 Pa, builder mesh 0.18). REVIEW FIX: the first peel was topological (a simplex went when it exposed a facet that was not an old boundary facet) and removed FLUID in 3D, where the diagonals of planar wall squares change between rebuilds: lattice cube total volume down to 0.790, 41.7 % lost in one call on the coarsest lattice; with the geometric test the volume is 1 to 1e-15 and every 1D / 2D number is bit-identical. Re-deriving the boundary orientation at each call is a DO-NOT (a thin surface simplex inverts during the step: winding numbers 0.5, 3D domain flicker 8e-5 per call, peak 0.185 instead of 0.157 m/s). WITHOUT the remap it is the laneK instability (128 m/s). Needs HC._simplices before the first call; no merge step, no inlet / outlet BCs, backend not applied. test_material_delaunay.py (18), test_case_hydrostatic.py |
| `dual_only` (dims 2,3) | validated | skip_triangulation=True: keep builder connectivity, refresh v.boundary tags, duals, dual volumes (and per-phase split / redistribution / EOS for multiphase) every step | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize (skip_triangulation branch)` | 3D droplet default (laneB l2 0.24811); 2D opt-in (l2 0.17857, lane5). Cannot follow large deformation (dam break NaN-aborts, laneF). laneK: single-phase + EOS is stable here without any remap (pressure force = exact volume gradient to 1e-10; symplectic_euler stable to dt c_s/dx = 1.5). laneS: on a builder mesh it now reads simplex_exact volumes and runs with free (untagged) surface vertices, which raised IndexError before: free-surface box, g = 0, 1e-6 m/s seed decays 1.5e-6 -> 2.0e-9 over 8 acoustic times (test_builder_simplex_cache.py). laneP: hydrostatic_2D / hydrostatic_2D_periodic presets (200 t_ac: max|u| envelope 1.1e-4 / 1.1e-6 m/s from uniform density, 4.2e-9 / 3.4e-11 from the equilibrium masses). 3D SINGLE PHASE + EOS: do not use, the 3D branch zeroes the dual volume of every frozen vertex, so wall cells read P0: hydrostatic 3D column max|u| 3.9e-2 m/s and integrated L2 1.5e4 Pa (1.5 rho g H) at 100 t_ac even from the equilibrium masses; use dual_only_bare |
| `dual_only_bare` (dims 2,3) | validated | Frozen connectivity, boundary retagged from HC.boundary(), compute_vd + cache_dual_volumes (half-cell boundary volumes, no edge-area cache) + per-phase split; NO mps.refresh, NO redistribution. Multiphase: no EOS update (p_phase frozen at its setup value). Single phase: the EOS in dudt_fn reads the refreshed dual volumes, so the pressure is live | `ddgclib/methods/_retopo.py:bare_dual_refresh` | static_droplet_2D pin 1.1847162859108737e-03 (was the case-local _dual_only_retopo closure; audit F11: validates surface tension against a FROZEN pressure field). laneP: the integrator boundary_filter is honoured (default None = whole hull frozen, unchanged). Single phase the EOS in dudt_fn is live, so this is the fixed-connectivity path with wall half cells and p_ij faces in 3D: hydrostatic_3D preset, 100 t_ac max|u| envelope 2.4e-4 m/s (uniform density start) / 5.4e-7 (equilibrium masses), integrated L2 0.99 Pa = 1.0e-4 rho g H |
| `frozen` | opt-in | retopologize_fn=False: NO topology or dual refresh at all. Surface meshes / A.5.a static probes only | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize` | A.5.a frozen-mesh floors 2.3749e-3 (2D) / 6.0153e-05 (3D); multiphase p_phase is NEVER updated on this path; laneK: with an EOS the pressure is inert (dual volumes never refreshed), so a stable frozen run is NOT evidence of EOS stability |
| `adaptive` (dims 2) | opt-in | hyperct.remesh.adaptive_remesh local split/collapse/flip preserving the v.phase interface; 2D only | `hyperct/remesh/_driver.py:adaptive_remesh via _retopologize(remesh_mode="adaptive")` | lane4 upstream conservation fix; l2 0.326 at refine 3/3 (second-best after dual_only); no pinned case uses it |
| `periodic` (dims 2,3) | experimental | retopologize_periodic: wrap, ghost-cell Delaunay, min-image duals (2D only in stress.py). Multiphase: retopologize_multiphase_periodic adds mps.refresh + redistribution + EOS (no remap, no cadence) | `ddgclib/geometry/periodic.py:retopologize_periodic; ddgclib/methods/_retopo.py:retopologize_multiphase_periodic` | periodic path ignores skip_triangulation/remesh/backend; shearing_plate_droplet 2D is unstable (interface lost by t=0.044 s), 3D crashes in setup; domain_bounds is a build-time argument (geometry), periodic_axes the method field. laneP, single phase + EOS: not usable. After ONE retopologize_periodic of periodic_rectangle (unit square, refinement 3) the total dual volume is 2.488 (exact 1.0; seam simplices are measured with raw coordinates), the cache holds 287 simplices instead of 256, 45 of 136 vertices fail dual-face closure (29 of them interior) and 6 interior vertices are tagged boundary; the hydrostatic column exceeds c0 at 0.66 t_ac (diagnose_column.py periodic). The Hydrostatic_2D_periodic runner therefore uses free-slip walls on dual_only |
| `custom` | experimental | User-supplied retopologize_fn callable (e.g. static_droplet_2D bare dual-only, Hagen_Poiseuile_3D cylinder) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_do_retopologize (callable branch)` | recorded by label only; kwargs forwarded by declared name |

### `remap` — Conservative retopology remap (explicit)

Default: `None`. Applied via: retopo_remap= in _retopologize_multiphase (multiphase partial) or in _retopologize (single-phase partial built by SolverMethods.retopologize_fn)

Same axis, two implementations. Multiphase: projection of the PRE-call field (cadence on projection_every). Single-phase: fresh snapshot, so the step's compression survives and there is no cadence to choose.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `None` | validated | No remap: reconnection changes dual volumes, EOS reads them as compression | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize_multiphase` | required value under dual_only (remap is a silent no-op there) |
| `conservative` (dims 2,3) | validated | Pressure field invariant across the rebuild. MULTIPHASE: stage-1 dual refresh on OLD connectivity, rebuild, per-phase redistribution, vol_corr gauge, restore_pressure_multiphase, anchor_phase_pressure_levels. SINGLE-PHASE: fresh snapshot eos(m / V_old-connectivity) at the current positions, rebuild, re-target EVERY vertex with a dual volume (frozen walls included), one exact mass rescale | `ddgclib/operators/mass_redistribution.py:restore_pressure_multiphase, anchor_phase_pressure_levels (multi); snapshot_pressure_fresh, redistribute_mass_single_phase(include_frozen=True) via _retopologize(retopo_remap=) (single)` | 2D droplet default (laneD/E, l2 0.17479); dam break survives reconnection (laneF); 3D multiphase measured-worse (laneE l2 1.873 vs 0.248, DO-NOT; confirmed cache-free by laneI). SINGLE-PHASE (laneK prototype, laneR library): box + EOS stable where bare Delaunay blows up at any CFL / c_s / n; final KE within 0.2 % of dual_only; pinned by test_single_phase_remap.py. Interior-only or stale-v.p variants are measured DO-NOTs (laneK P2, P3). laneP: the uniform offset K (s - 1) of the mass rescale (laneR known limit) is NOT what breaks a free-surface column: without the rescale the convex-hull arm is worse (59.4 against 42.1 m/s, mass drift +1.5 %, total volume 1.03; diagnose_column.py remap), and once the hull fill is removed (connectivity=delaunay_material) the offset stays below 1 Pa (convex arm: up to 2.7e3 Pa per rebuild) and the run with and without the rescale agree to 3 digits. No gauge was added |

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

### `frozen_set` — Frozen (wall) vertex set (explicit)

Default: `hull`. Applied via: frozen_set= in _retopologize / _retopologize_multiphase, bound into the retopology partial by SolverMethods.retopologize_fn()

Which vertices the integrators do not move (bV). The tag v.boundary that compute_vd needs for half cells follows the topological boundary under both values. boundary_filter (build-time argument) narrows either set.

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `hull` | validated | bV is rebuilt at every retopology from the topological boundary of the new connectivity, narrowed by boundary_filter. A vertex is frozen because it is on the hull and released when it is not | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize (step 6)` | every pin (all droplet, hydrostatic and electrolysis presets run on it). FAILS when one vertex steps past a straight wall: the wall vertices next to it leave the hull, are released and integrated, and nothing re-captures them (audit 2026-09-25 F10 C1). Measured by laneL: Hagen_Poiseuile_2D, 3000 steps: walls released at step 1262 (two outlet buffer vertices drift past the wall lines), 2 of 62 wall vertices still frozen, 60 moved, largest displacement 3.46; dam_break_2D in its two ejection configurations (alpha_art 0.2; 0.5 to t = 0.45 s): one step after the first vertex leaves the tank the walls are released and all 32 wall vertices move (test_frozen_set.py, cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py) |
| `membership` (dims 2,3) | opt-in | bV is persistent: a retopology keeps the members that are still in the complex (and pass boundary_filter), never adds a vertex because it is on the hull and never drops one because it is not. A hull vertex that is not a member (inlet, outlet, free surface, a vertex that left through a wall) is tagged, gets a half cell and is integrated; a member off the hull stays frozen with a closed cell; capture at a wall is left to the BC that holds bV (PositionalNoSlipWallBC) | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize (step 6)` | laneL 2026-10-01. Hagen_Poiseuile_2D (preset), 3000 steps, past the former collapse: 62 wall vertices, 62 still frozen, 0 moved. dam_break_2D (preset): shipped run (1585 steps) final state bit-identical to hull; in the two ejection configurations 32 of 32 walls stay frozen and in place, but the fluid still blows up and the run aborts with the same QhullError 23 / 8 steps later than under hull (after 1303 against 1280 steps; 3030 against 3022; the same in every process since the tied simplex vote is deterministic, fix round 1): the walls are not what fails there. Bit-identical to hull while no vertex leaves the hull: 2D droplet with remap (100 steps), electrolysis 2D (6330 steps, the shipped horizon), electrolysis 3D (300 steps). CONNECTIVITY: implemented for delaunay only. With adaptive it RAISES (in SolverMethods and in both retopology functions): hyperct.remesh protects vertices by v.boundary, not by bV. Measured on a channel: one adaptive retopology left the 8 wall vertices created by wall-edge splits unfrozen (10 of 18 against 18 of 18 under hull; integrated, they leave the wall), and with 7 members off the hull adaptive_remesh moved up to 7 of them (smoothing, up to 0.17) and removed up to 5 (collapse). LIMITS: impenetrability is not enforced (HP2D: one fluid vertex 5.1e-3 outside the top wall at t = 30); a wall-row vertex injected by an inlet is frozen only when PositionalNoSlipWallBC runs AFTER the inlet BC (under hull the next retopology captured it); 3D: a vertex whose dual fan fails is tagged and zero-volumed but not frozen (0 occurrences in a jittered box and a 3D droplet); merge_cdist can merge a member into a mobile vertex; not implemented for periodic, delaunay_material and custom retopology (SolverMethods raises), not needed for dual_only / dual_only_bare / frozen (their bV never changes). test_frozen_set.py (31) |

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
| `delta>0` | opt-in | dm_i/dt = sum_j delta c0 |A_ij| [(rho_j - rho_i) - 1/2 (grad rho_i + grad rho_j).d_ij]: exactly mass conserving, exactly zero on linear density fields (interior), no shear viscosity; explicit stability delta < ~0.3 at acoustic CFL 0.4 | `ddgclib/operators/stabilisation.py:density_diffusion_step` | capillary_rise dynCA smoke (water R 0.5 mm, after the corner fix): L2 0.26 -> 0.075 (delta 0.05) / 0.13 (0.1); 0.2-0.3 over-smooth (capillary_rise_energy_grad README Section 5). laneP: it does not cure the slow instability of the inviscid hydrostatic column (no-slip 2D drop with the viscosity of water exceeds c0 at 63 / 57 t_ac for delta 0.05 / 0.1, 64 without) |

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

Default: `simplex_aware`. Applied via: presence of HC._simplices (connect_and_cache_simplices at every Delaunay retopology; rebuild_simplex_cache_2d / _3d of the built connectivity in every domain builder, DomainResult.__post_init__)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `simplex_aware` | validated | Top-simplex cache drives compute_vd, boundary_from_simplices, exact volumes | `hyperct/ddg/_retriangulation.py:connect_and_cache_simplices` | commit 8321c71; test_simplex_aware_duals.py |
| `nn_walk` | dead | Legacy 1-skeleton (v.nn intersection) walk; ghost K_{d+1} cliques on Delaunay meshes | `hyperct/ddg/_compute_dual.py (legacy branch)` | docs/3d_simplex_aware_dual_fix.md |

### `dual_volume` — Dual cell volume source (reported)

Default: `simplex_exact`. Applied via: dim + HC._simplices + HC._vd_method + whether batch_e_star runs (stress.py:_use_exact_barycentric_volume, _retopologize step 5b)

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `simplex_exact` | validated | Vol_i = (1/(d+1)) sum_{T contains i} |T| (hyperct.ddg.simplex_dual_volumes / vertex_dual_volume) | `hyperct/ddg/_dual_volume.py` | 3D switch ON 2026-07-29 (laneA), floor re-pinned 7.274172e-05. laneS (2026-10-01): the domain builders cache the simplices of the connectivity they build, so SETUP reads this source too (rectangle total 0.96875 -> 1.0, box 0.9167 -> 1.0, no volume jump at the first retopology; test_builder_simplex_cache.py). Shipped Hydrostatic_2D then settles (100 t_ac, |u| 3.0e-4) instead of reaching 10 c0 at 6.9 t_ac; every pinned number is bit-identical (the droplet meshes already carried a Delaunay cache) |
| `fan_walk_3d` (dims 3) | measured-worse | batch_e_star / v_star tetra fan sum; undercounts 1-4% interior, ~20% boundary | `hyperct/ddg/_operators.py:batch_e_star(compute_volumes=True)` | docs_temp/audit/dual-volume-3d.md. laneS: no builder mesh reaches it any more (it gave the box builder 0.9167 of its volume at setup); left for hand-built 3D complexes without a simplex cache and for circumcentric duals |
| `dual_cell_area_2d` (dims 2) | opt-in | Shoelace area of the 2D dual polygon (circumcentric duals, or a hand-built 2D complex without a simplex cache; builder meshes no longer reach it) | `hyperct/ddg/_dual_cell.py:dual_cell_area_2d` | Was broken until laneS (laneK: boundary polygon without the vertex itself, so the four corner cells were 4x too small, rectangle total 0.96875, and a moving free-surface vertex got 1/4 of its own volume change; Hydrostatic_2D blew up with 0 flips). FIXED in hyperct 2026-10-01: the half cell of a boundary vertex is walked as an open chain and closed through the vertex; it now equals the simplex rule at every vertex of a kinked boundary to 1e-11 (hyperct test_dual_volume.py) and the Hydrostatic loop run on it (driver variant fallback) matches the simplex_exact run. Degenerate fans still use the angular sort (no vertex point); circumcentric boundary cells are not validated (laneK P14/P15) |
| `interval_1d` (dims 1) | validated | Distance between the two dual vertices | `ddgclib/operators/stress.py:dual_volume` |  |

### `boundary_dual_vol` — Boundary-vertex dual volume convention (reported)

Default: `half_cell`. Applied via: dim: 3D retopology zeroes boundary dual_vol (batch_e_star path); 1D/2D/periodic/setup keep the truncated half cell, and so do connectivity=dual_only_bare and delaunay_material in 3D

| value | status | what it does | code | evidence |
|---|---|---|---|---|
| `zeroed` (dims 3) | validated | v.dual_vol = 0 on every vertex in bV after retopology | `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize step 5b` | measured 2026-09-25: 3D box boundary dual_vol 0.0. laneP: with a single-phase EOS a zero-volume wall cell reads the reference pressure P0, which breaks any case whose wall pressure is not P0 (hydrostatic 3D) |
| `half_cell` | validated | Boundary vertices keep the truncated dual cell (cache_dual_volumes path) | `ddgclib/operators/stress.py:cache_dual_volumes` | measured 2026-09-25: 2D rectangle max boundary dual_vol 0.0156, total 1.0. Also 3D under connectivity=dual_only_bare and delaunay_material (laneP) |

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

| preset | `dim` | `phases` | `integrator` | `connectivity` | `remap` | `projection_every` | `frozen_set` | `redistribute_mass` | `workers` | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `oscillating_droplet_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `1` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap') |
| `oscillating_droplet_2D_dual_only` | `2` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='dual_only') |
| `oscillating_droplet_2D_bare_delaunay` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay') |
| `oscillating_droplet_2D_projection2` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `2` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py (retopo_policy_2d='delaunay_remap_p2') |
| `static_droplet_floor_2D` | `2` | `multi` | `euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet2DRetopologyFloor (u=0 every step) |
| `static_droplet_2D` | `2` | `multi` | `symplectic_euler` | `dual_only_bare` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/oscillating_droplet/static_droplet_2D.py |
| `oscillating_droplet_3D` | `3` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py (retopo_policy_3d='dual_only') |
| `oscillating_droplet_3D_delaunay` | `3` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py --retopo delaunay |
| `static_droplet_floor_3D` | `3` | `multi` | `euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | ddgclib/tests/test_case_oscillating_droplet.py::TestStaticDroplet3DRetopologyFloor (u=0 every step) |
| `dam_break_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `conservative` | `1` | `membership` | `True` | `None` | cases_dynamic/dam_break/dam_break_2D.py |
| `dam_break_3D` | `3` | `multi` | `symplectic_euler` | `dual_only` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/dam_break/dam_break_3D.py |
| `electrolysis_bubble_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py |
| `electrolysis_bubble_3D` | `3` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_3D.py |
| `electrolysis_bubble_fritz_2D` | `2` | `multi` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/electrolysis_bubble/electrolysis_bubble_fritz_2D.py (run_short_dynamics) |
| `shearing_plate_droplet_2D` | `2` | `multi` | `symplectic_euler` | `periodic` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_2D.py, _run_short_2D.py |
| `shearing_plate_droplet_3D` | `3` | `multi` | `symplectic_euler` | `periodic` | `None` | `1` | `hull` | `True` | `None` | cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_3D.py, _run_short_3D.py |
| `hagen_poiseuille_2D` | `2` | `single` | `symplectic_euler` | `delaunay` | `None` | `1` | `membership` | `False` | `20` | cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py |
| `hagen_poiseuille_2D_eulerian` | `2` | `single` | `euler_velocity_only` | `delaunay` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/Hagen_Poiseuile_2D_Eulerian/Hagen_Poiseuile_2D_Eulerian.py |
| `hagen_poiseuille_3D` | `3` | `single` | `symplectic_euler` | `custom` | `None` | `1` | `hull` | `False` | `8` | cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py (retopologize_cylinder) |
| `hydrostatic_1D` | `1` | `single` | `symplectic_euler` | `delaunay` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/Hydrostatic_column/Hydrostatic_1D.py |
| `hydrostatic_2D` | `2` | `single` | `symplectic_euler` | `dual_only` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py |
| `hydrostatic_2D_periodic` | `2` | `single` | `symplectic_euler` | `dual_only` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/Hydrostatic_column/Hydrostatic_2D_periodic.py (free-slip side walls) |
| `hydrostatic_3D` | `3` | `single` | `symplectic_euler` | `dual_only_bare` | `None` | `1` | `hull` | `False` | `None` | cases_dynamic/Hydrostatic_column/Hydrostatic_3D.py |

## 4. Case → configuration matrix (hand-maintained, 2026-09-25; Hydrostatic_column, Hagen_Poiseuile_2D and dam_break_2D rows 2026-10-01)

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
| `dam_break/dam_break_2D.py` | `dam_break_2D` (delaunay + conservative remap, redistribute, `frozen_set='membership'` since laneL) | hydrostatic per-phase mass preload, `col_h = a` headspace, alpha_art 0.3 baked into `PhaseProperties.mu`, gravity closure, linear Tait, cfl 0.1 | runs full horizon (laneF); blocker: air sliver-cell F/m ejection. laneL 2026-10-01: final state of the shipped run bit-identical to `.replace(frozen_set='hull')` (no vertex leaves the tank); in the ejection configurations (alpha_art 0.2; 0.5 to t = 0.45 s) the 32 wall vertices stay frozen and in place (hull: all 32 move one step after the first vertex leaves the tank), but the fluid blows up in both arms and both abort with a QhullError (hull after 1280 / 3022 steps, membership after 1303 / 3030; the same in every process since the tied simplex vote is deterministic, laneL fix round 1) | KE_liq peak 1.0369e-3 J @ 0.0506 s, mass 6.2e-15 |
| `dam_break/dam_break_3D.py` | `dam_break_3D` (dual_only) | as above | smoke only. laneS 2026-10-01: the setup now votes the simplex phases on the builder tetrahedra (it used to run a Delaunay on top of the builder edges), 9 of 189 vertices change label, setup mass 0.0622 → 0.0847; not re-run | — |
| `Hagen_Poiseuile/Hagen_Poiseuile_2D.py` | `hagen_poiseuille_2D` (single, symplectic, delaunay, `frozen_set='membership'`, workers 20) + `boundary_filter=walls` at build time; setup in `src/_setup.py:setup_poiseuille_2d_lagrangian`: `OutletBufferedDeleteBC` + `PeriodicInletBC(cdist=1e-10)` + `PositionalNoSlipWallBC` (wall BC last); `--frozen-set hull --tag hull` = the old rule as an A/B arm; `--headless --steps N` for scripted runs; the runner puts the repository root on `sys.path` itself | plug IC, point-value pressure on the ghost | laneL 2026-10-01: runs the 3000 steps through the preset, walls intact (62 wall vertices, 62 still frozen, 0 moved; `results/wall_report.json`). Hull arm: walls released at step 1262, 60 of 62 moved, largest displacement 3.46. NOT validated: U_max 0.29 against 0.20 at t = 30, one fluid vertex 5.1e-3 outside the top wall (no impenetrability), two wall-row duplicates one advection step from the inlet corners (audit C3b) | wall positions: `test_frozen_set.py::TestHagenPoiseuille2D` (L = 2, dt 0.05, 300 steps) |
| `Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py` | `hagen_poiseuille_3D` (single, `custom` = `retopologize_cylinder`, drops the builder simplex cache and never re-populates `HC._simplices`, laneS; workers from CLI) | — | stalled (inlet cap frozen, audit M2); not re-run; 6-step smoke bit-identical across laneS | — |
| `Hagen_Poiseuile_2D_Eulerian/` | `hagen_poiseuille_2D_eulerian` (`euler_velocity_only`, default retopo) | point-value pressure IC | runs, no pin | — |
| `Hagen_Poiseuile_equilibrium/` | static residual evaluation, no integrator | — | validated equilibrium | median residual < 1e-13 |
| `Hydrostatic_column/Hydrostatic_1D.py` | `hydrostatic_1D` (single, symplectic, `delaunay` = the sorted chain in 1D) + `boundary_filter` = bottom vertex at build time; gravity via `dudt_fn(body_force=)` | 33 vertices, H 10 m, Tait n = 1, c0 = 10 sqrt(g H), mu = 0.5 rho c0 dx, CFL 0.25, uniform-density start (`--ic equilibrium` for the hydrostatic masses) | laneP 2026-10-01: runs through the preset (equal to the former hand-rolled loop to round-off). 200 t_ac: rings at the fundamental mode, KE decay 0.0389 / t_ac (viscous theory 0.0386), max\|u\| 0.890 → 1.4e-2; equilibrium start 2.8e-6 → 7.0e-8 | fast: `PIN_1D_UMAX_PEAK` 0.8463965668751099, `PIN_1D_KE_END` 8.07861451666529 (refinement 3, 40 t_ac) |
| `Hydrostatic_column/Hydrostatic_2D.py` | `hydrostatic_2D` (single, symplectic, `dual_only`) + `boundary_filter` = no-slip walls; `--arm remap` = `.replace(connectivity='delaunay_material', remap='conservative', redistribute_mass=True)` | 145 vertices, free surface, same EOS / viscosity rule | laneP: settles. 100 t_ac max\|u\| 0.248 → 3.0e-4, integrated L2 50.2 Pa (5.1e-3 rho g H; boundary cells, interior 6.8 Pa); 200 t_ac 1.1e-4; equilibrium start 7.0e-6 → 4.2e-9. Remap arm stable (1.2e-4 at 200 t_ac, noise floor 2.3e-6; domain volume kept to round-off in every call). `delaunay` + remap is NOT (42 m/s at 3 t_ac: convex-hull fill). Needs mu_art: with the viscosity of water the drop exceeds c0 at 64 t_ac (discrete equilibrium is a saddle, growth 1.13 g/c0) | fast: `PIN_2D_UMAX_PEAK` 0.1951860472291084, `PIN_2D_KE_END` 3.1650897086706908e-06 (refinement 2, 20 t_ac); slow: `PIN_2D_KE_40` 9.872765069804785e-07, `PIN_2D_REMAP_KE_40` 2.721558596123262e-06 |
| `Hydrostatic_column/Hydrostatic_2D_periodic.py` | `hydrostatic_2D_periodic` (single, symplectic, `dual_only`); NOT periodic connectivity: free-slip side walls (`FreeSlipWallBC`), frozen bottom | as 2D | laneP: runs and rings down as a 1D column (KE decay 0.126 / t_ac, theory 0.125): 100 t_ac max\|u\| 0.256 → 2.6e-4, L2 50.6 Pa; 200 t_ac 1.1e-6; equilibrium start 6.6e-6 → 3.4e-11. `connectivity='periodic'` is unusable for this column (dual volume 2.488 for a unit domain, 45 of 136 vertices fail closure, exceeds c0 at 0.66 t_ac) | fast: `PIN_2DP_UMAX_PEAK` 2.631506910977075e-05 (refinement 2, equilibrium start, 20 t_ac) |
| `Hydrostatic_column/Hydrostatic_3D.py` | `hydrostatic_3D` (single, symplectic, `dual_only_bare` + `boundary_filter` = walls: wall half cells, p_ij faces, free top); `--arm remap` as 2D | 189 vertices | laneP: settles. 40 t_ac max\|u\| 0.144 → 2.9e-4, L2 174 Pa (1.8e-2 rho g H); 100 t_ac 2.4e-4; equilibrium start 3.2e-5 → 5.4e-7, L2 0.99 Pa. Remap arm stable: 100 t_ac 5.6e-4 to 5.7e-4 in two processes, floor 9.9e-7 from the equilibrium masses; the 3D rebuild keeps the domain volume to 1.2e-6 per call (free-surface diagonal slivers, 2D: round-off); after about 10 t_ac the 3D remap run is reproducible between processes to 2 digits only. `dual_only` (zeroed wall volumes) cannot hold it: L2 1.5e4 Pa | slow: `PIN_3D_UMAX_PEAK` 0.08910127097486757 (refinement 1, 40 t_ac); remap arm `PIN_3D_REMAP_UMAX_PEAK` 0.15658060026054665 (refinement 2, 2 t_ac) |
| `capillary_rise/capillary_rise_2D.py`, `_3D.py` | hand-rolled, `_recompute_duals`, static-angle Washburn body force | — | scaffold; laneS moved the setup volumes (2D total 4.2608e-06 → 4.3983e-06, 3D 5.727e-09 → 6.220e-09), not re-run | — |
| `capillary_rise/capillary_rise_2D_dynCA.py`, `_3D_dynCA.py` | hand-rolled: `mass_conserving_merge` + `adaptive_remesh` every few steps + `_recompute_duals` + density-continuity repair; data-driven CA(t) forcing or `--ca-model` (energy-gradient contact line, `capillary_rise_energy_grad`); opt-in `--surface-bc yl`, `--contact-mode mobility`, `--delta-diff` (= `density_diffusion`), `--pressure-flux` | wall film, contact line, free-surface line tension | 2026-09-25/26: the stall was the orphan cleanup deleting the strip's top corners (fixed); with YL-BC + mobility + `delta` 0.05 the smoke window scores L2 0.04 (was 0.72), 50 ms window within 2 % at 40 ms then −23 % (contact-line attachment) | smoke L2 0.040, 50 ms L2 0.12 |
| `electrolysis_bubble/electrolysis_bubble_2D.py`, `_3D.py` | `electrolysis_bubble_2D` / `_3D` (multi, delaunay, no remap, redistribute) | gravity + NaN guard in the setup dudt wrapper, `WallClampBC`, gas mass injection in the callback | 2D unvalidated; 3D unstable (gas phase lost by t≈1.1e-4 s). laneL: `frozen_set='membership'` is bit-identical on the shipped 2D horizon (6330 steps) and on 300 steps in 3D (no vertex leaves the box); not switched | 5-step mass drift ≤ 1.94e-15 |
| `electrolysis_bubble/electrolysis_bubble_fritz_2D.py` | `electrolysis_bubble_fritz_2D` (multi, delaunay, redistribute **False**: the historic partial left it unbound) | 80-step smoke | runs | — |
| `shearing_plate_droplet/*` (2D, 3D, `_run_short_*`) | `shearing_plate_droplet_2D` / `_3D` (multi, `periodic` = library `retopologize_multiphase_periodic`, no remap / cadence; `domain_bounds` from the setup params at build time), `ShearingPlateBC` | anisotropic outer rescale pushes vertices into the droplet | 2D unstable (interface lost t≈0.044 s), 3D crashes in setup. laneL fix round 1: the 2D setup was process-dependent (10 tied triangles in the simplex phase vote, decided by `id()` order); a tie now goes to the lower phase ID and the three-step digest `17e9a78408066fea` is the same in 120 of 120 interpreters | — |
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
| `frozen_set` | `'hull'`: nothing bound (`retopologize_fn=None`). `'membership'`: `partial(_retopologize, frozen_set='membership')` (plus `retopo_remap=` when the remap is on); the integrator forwards its retopology kwargs to it by name. `connectivity='delaunay'` only; `'adaptive'` raises | `'membership'` adds `frozen_set=` to `partial(_retopologize_multiphase, ...)`, which passes it to both of its `_retopologize` calls. `connectivity='delaunay'` only; `'adaptive'` raises |

## 6. Regenerating

```bash
python -m ddgclib.methods --update METHODS.md                       # rewrite sections 2-3 in place
python -m ddgclib.methods --markdown                                # print them instead
python -m ddgclib.methods --preset oscillating_droplet_2D           # describe one preset
pytest ddgclib/tests/test_methods.py -q                             # incl. the drift check
```
