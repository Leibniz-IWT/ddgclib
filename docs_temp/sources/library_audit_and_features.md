# Library Audit & Feature Matrix (distilled)
> Sources: /home/endres/projects/ddgclib/LIBRARY_AUDIT.md (dated 2026-06-08), /home/endres/projects/ddgclib/FEATURES.md | Written: 2026-07-02 by understand-and-document workflow

## 0. Identity and headline verdict

ddgclib v0.4.3 (Alpha), Python 3.9+ (dev on 3.13, conda env `ddg`). Lagrangian DDG fluid simulator: mesh IS the fluid; vertices carry u, p, m; physics = integrated operators on barycentric dual cells of a `hyperct.Complex` (hyperct symlinked into repo).

Two coexisting pipelines:

| Pipeline | Status | Entry points |
|---|---|---|
| Dynamic continuum (Cauchy stress) | **Active core** | `operators/stress.py`, `dynamic_integrators/`, `operators/multiphase_stress.py` |
| Mean curvature flow | **Legacy** | `mean_flow_integrators/`, `_curvatures.py`, `_bubble.py`, `cases_mean_flow/` |

**Core proven result:** at *equilibrium*, the integrated DDG stress/force operators reproduce Hagen-Poiseuille shear flow and hydrostatic column analytically to machine precision. **Central open problem:** carrying that accuracy into *dynamic multiphase* runs — retopologization and interface handling inject energy/instability. Tracked in `debugging_plan.md` (the canonical stability-investigation log per audit §6.7).

Test suite at audit time: `pytest -m "not slow"` → **808 passed, 1 skipped, 2 xfailed, 17 slow deselected** (~31 s). Known xfail: `test_dem_liquid_bridge.py` bridge-formation timing.

## 1. Per-module verdicts

### 1a. CORRECT / production-usable (tested, exercised by cases)
- `operators/stress.py` — Cauchy stress operators. `F_stress_i = Σ_j σ_f · A_ij`, `σ = -p I + 2 μ ε` (Newtonian). Chain: `dual_area_vector` → `velocity_difference_tensor` → `strain_rate` → `cauchy_stress` → `stress_force` → `stress_acceleration`; `dudt_i` = canonical integrator alias. 2D+3D. Validated vs Hagen-Poiseuille (equilibrium residual, developing flow, parabolic profile) and hydrostatic column (pressure force direction, zero viscous stress at rest, viscous damping). Audit: operator layer "already well-factored... no large rewrites warranted".
- `dynamic_integrators/` — `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only` (Eulerian, validation only), `euler_adaptive` (CFL); `DynamicSimulation` runner; all take `bc_set`, `remesh_mode`, `remesh_kwargs`.
- `initial_conditions.py` / `_boundary_conditions.py` — composable IC/BC classes; `BoundaryConditionSet` applies in insertion order; Dirichlet, Neumann, no-slip wall, outlet-delete, periodic-inlet. Multiphase ICs: `PhaseAssignment`, `MultiphaseMass`, `MultiphasePressure` (optional Young-Laplace jump).
- `geometry/domains/` — `rectangle`, `l_shape`, `disk`, `annulus`, `box`, `cylinder_volume`, `pipe`, `ball`, periodic variants, multiphase droplet builders (`droplet_in_box_2d/_3d` in `_multiphase_droplet.py`); 56 tests. `DomainResult` is the intended canonical mesh-construction schema (audit §5.3).
- `geometry/_parametric_surfaces.py` — sphere, catenoid, cylinder, hyperboloid, torus, plane + transforms; 37 tests.
- Multiphase framework — `multiphase.py` (`PhaseProperties`, `MultiphaseSystem`: `assign_phases`, `identify_interface`, `init_phase_fields`, `split_dual_volumes`, `refresh`, `get_mu`, `get_gamma`; per-phase fields `v.m_phase`, `v.p_phase`, `v.rho_phase`, `v.dual_vol_phase`), `operators/multiphase_stress.py` (own-phase pressure flux, phase-specific viscosity, surface tension only at interface vertices; `multiphase_dudt_i` alias), `operators/mass_redistribution.py` (pressure-preserving mass redistribution after retriangulation: `snapshot_pressure_multiphase`, `redistribute_mass_multiphase`), `operators/curvature_2d.py`, `operators/surface_tension.py`, `geometry/_dual_split_2d.py` (exact geometric dual-volume split, **covers both 2D and 3D** despite the name), `geometry/_interface_subcomplex.py`.
- 2D surface tension is *exact*, not approximate: `F_st_i = ∫_{Γ_i} γ κ N ds = γ (t_next − t_prev)` (FTC on the tangent along a piecewise-linear interface). Reconstructs circle arc length/enclosed area to machine precision. Impl: `curvature_2d.py` — `integrated_curvature_normal_2d(v)`, `surface_tension_force_2d(v, gamma)`, `reconstruct_arc_length_and_bulge_area(v_i, v_j, Delta_T)`; 18 tests in `ddgclib/tests/test_curvature_2d.py`. 3D uses cotangent-weight Heron curvature (`surface_tension_force`); `surface_tension_acceleration` is a drop-in `dudt_fn` for surface meshes without volume duals. `multiphase_stress._interface_surface_tension` delegates to `curvature_2d` in 2D.
- `eos/` — `EquationOfState` base; `TaitMurnaghan`, `IdealGas`, `MultiphaseEOS` dispatcher, `eos_pressure_update`.
- `dem/` — full self-contained pipeline, 124 tests: `Particle.sphere()`, `ParticleSystem`, `ContactDetector` (cell-linked spatial hash), `HertzContact` (F_n ∝ δ^{3/2}) / `LinearSpringDashpot` via `ContactForceModel` ABC + `contact_force_registry`, `dem_velocity_verlet` / `dem_symplectic_euler` / `dem_step` (sub-stepping), `SinterBond` + Frenkel/Kuczynski `grow_neck(dt,T,D)` + `BondManager` (cluster IDs), `LiquidBridge` (toroidal approx., Lian et al. 1993 rupture) + `LiquidBridgeManager` (frozenset pair keys), `FluidParticleCoupler` (IDW interpolation; Stokes & Schiller-Naumann drag; two-way `fluid_to_particle`/`particle_to_fluid`), JSON I/O (`ddgclib_dem_state_v1`), `import_particle_cloud`, plotting.
- `analytical/` — divergence-theorem + sympy integration; integrated comparisons; volume-averaged-field conventions. Plus `int_bench/` n-D integrated gradient/Hessian harness (self-described "throwaway staging" — home undecided, audit §4.12).
- `data/` — JSON `save_state`/`load_state`, `StateHistory`, conservation diagnostics (`data/_conservation.py`: mass/momentum/volume drift).
- `visualization/` — unified `plot_primal()`/`plot_dual()` (auto-detect `HC.dim`), `plot_fluid()`, `dynamic_plot_fluid()`, multiphase (`visualization/multiphase.py`: `record_multiphase_frame`, `dynamic_plot_multiphase`), polyscope optional; base mesh delegates to `hyperct._plotting`.
- `geometry/periodic.py` — ghost-cell merge, minimum-image duals; 26 tests.
- GPU/torch + multiprocessing backends via `hyperct._backend` and `compute_vd(..., backend="torch"|"gpu")`.
- Retopologization: `_retopologize` and `_retopologize_multiphase` (preserve phase labels + interface ID), `remesh_mode={'delaunay'|'adaptive'}`; default `'delaunay'`.

### 1b. SUSPECT / physically questionable / untested — FLAGGED
- **Adaptive remesh (`remesh_mode='adaptive'`) — built but UNSTABLE.** Blows up on the oscillating-droplet case: KE explodes, **~164% mass loss by ~step 80**. Root cause: upstream **mass-averaging bug in `hyperct/remesh/_operations_2d.py`** (mass averaged on split / summed on collapse is the intended contract). 3D remesh ops never started. Do not trust adaptive-mode dynamic multiphase results. Audit §7.3 calls fixing this "the gate on closing the dynamic-multiphase stability gap".
- **Dynamic multiphase stability — OPEN.** 2D & 3D *static-droplet* residual floors are regression-locked at machine-precision conservation, but full dynamic Rayleigh-Lamb validation is still pending. The oscillating-droplet case (FEATURES claims it "validates against Rayleigh-Lamb") should be read as *aspires to*; the audit lists it as unfinished.
- **Single-phase retopology volume conservation (S-lane / A.3) — STALLED.** `benchmarks/dynamic/`: frozen mesh conserves volume to machine precision, but both `skip_triangulation` and full Delaunay **leak ~2–4% volume**. Root cause localized to the *dual-volume refresh*, not Delaunay churn. Any long Lagrangian run with retopology has this systematic leak.
- **Boundary-dual semantics — cosmetic artefact.** First retopo zeros boundary `dual_vol` → one-shot `|dV/V0| ≈ 0.3` step in diagnostics that is *not* a drift. Don't misread it as a conservation bug.
- **3D oscillating-droplet validation — incomplete.** Case runs and conserves mass, but there is **no checked-in 3D `oscillation_score` baseline** to diff against; R_max saturates at the mesh boundary (physically suspect boundary interaction).
- **`MultiphaseSystem.split_dual_volumes` uses a neighbour-count approximation** (fraction of 1-ring neighbours per phase ≈ volume fraction). Exact geometric split exists in `geometry/_dual_split_2d.py` but the FEATURES "Planned" section still lists exact splitting + refinement-convergence test as future work — check which path a given case actually uses. Contradiction noted in §3 below.
- **3D circumcentric branches of `operators/area.py` and `operators/curvature.py`** raise `NotImplementedError` (fallback only).
- **Constitutive models beyond Newtonian** — viscoelastic / non-Newtonian / elastic are TODO stubs in `stress.py`; only σ = -pI + 2με is real.
- **`'stokes'` curvature path in `multiphase_stress.py`** — NOT dead code; deliberately kept as a regression-tested A/B equivalence probe (proven mathematically identical to the cotangent form). Do not "clean up".
- **DEM liquid bridge** — 1 xfail: bridge-formation timing needs tuning (`test_dem_liquid_bridge.py`).

### 1c. BROKEN / DEAD / NOT STARTED
- `ddgclib/geometry/_dual_split_3d.py` — **dead stub**, only `raise NotImplementedError`. The working 3D split lives in `_dual_split_2d.py`. **The filename actively misleads** (audit §4.2: delete stub, rename `_dual_split_2d.py` → `_dual_split.py`).
- `ddgclib/__init__.py` — entirely commented-out imports; `import ddgclib` exposes nothing. Audit §5.1 proposes a curated ~30-symbol public API + `__version__ = "0.4.3"`.
- `ddgclib/barycentric/_duals.py` — empty (0-line) migration artefact.
- `ddgclib/_eos.py` — 1-line shim (`from ddgclib.eos._base import *`); fold callers onto `ddgclib.eos`.
- `ddgclib/legacy/plots.py` — legacy and (believed) unimported; confirm and remove.
- Not started: dynamic capillary rise case (deferred, needs surface tension on meniscus + curvature operators); HDF5 data handling (JSON only; planned ~13x smaller, ~25x faster writes); hyperct geometry I/O module (`save_complex`/`load_complex` are incomplete stubs upstream); `channel_with_obstacle()` + surface-of-revolution builders; N-phase contact lines; fixed inlet/outlet BCs (`prescribed_V` — fully specced in DEVELOPMENT; `OutletBufferedDeleteBC` is the stopgap); migration of `dual_area_vector`/`dual_volume` into `hyperct.ddg._operators`; body-force (gravity) term in `stress_acceleration()` (FEATURES "Planned" — hydrostatic equilibrium is currently verified without an in-operator gravity term).
- Pylint pass in progress: baseline ~1.23/10 → ~7.8/10 via `.pylintrc`; wildcard imports / dead code / docstrings cleanup mostly unstarted.

## 2. Duplication & API inconsistencies (audit §4–§5)

- **Curvature triplication:** `_curvatures.py` (1717 lines, legacy) vs `_curvatures_heron.py` (active, imported by operators) vs `_curvatures_heron_torch_vectorized.py` (GPU, tests only). Target: one module under `operators/` with numpy + torch backends. `_curvatures.py` cannot be retired until `geometry/_volume.py` stops importing it.
- **`stress.py` vs `stress_pointwise.py`:** pointwise formulation is archived (documented in `Fundamentals_pointwise.md`); keep only if a test/benchmark references it, else move to `docs/archive/`.
- **Deprecation shims with no sunset:** `_plotting.py`, `_method_wrappers.py`, `barycentric/`, `circumcentric/` — emit `DeprecationWarning`; audit says set removal at ~v0.5. Canonical import is always `hyperct.ddg`.
- **Legacy layer at top level** (proposed move to `ddgclib/legacy/`): `_bubble.py`, `_capillary_rise.py`, `_capillary_rise_flow.py`, `_sessile.py`, `_flow.py`, `_case1.py`, `_case2.py`, `_cube_droplet.py`, `_catenoid.py`, `_ellipsoid.py`, `_hyperboloid.py`, `_sphere.py`, `_gauss_bonnet.py`, `_curved_volume.py`, `mean_flow_integrators/`.
- **Doc quadruplication:** `Fundamentals_old.md` / `Fundamentals_pointwise.md` / `Fundamentals_v1.md` superseded by `Fundamentals.md`; `ARCHITECTURE.md` vs `ARCHITECTURE_public.md` overlap heavily — one should be canonical.
- **Canonical usage patterns** (currently only in CLAUDE.md/ARCHITECTURE, should be README-level): `functools.partial(dudt_i, dim=, mu=, HC=)` is *the* recipe (never pass via `**dudt_kwargs` — "multiple values for HC" error); `DomainResult` (`HC`, `bV`, `boundary_groups`, `metadata`, `tag_boundaries()`, `summary()`) is the mesh-construction contract; extension ABCs are `InitialCondition.apply(HC, bV)` and `BoundaryCondition.apply(mesh, dt, target_vertices)`.
- `gradient.py` (`pressure_gradient`, `velocity_laplacian`, `acceleration`) are thin wrappers over the stress pipeline, not independent implementations.

## 3. Stale claims / contradictions noticed

1. **Test counts drift:** CLAUDE.md says "~415 tests"; ARCHITECTURE.md says "~604"; audit measured **808 passed** (2026-06-08). All hard-coded counts stale by design; audit recommends CI-generated counts.
2. **FEATURES.md internal contradiction on dual-volume splitting:** the "Multiphase Framework" section says `split_dual_volumes` is a neighbour-count *approximation* and lists exact geometric splitting under "Planned", but LIBRARY_AUDIT §3a states exact 2D **and** 3D geometric dual-volume splitting *landed* in `_dual_split_2d.py`. FEATURES "Planned" section is stale here.
3. **FEATURES.md structural oddity:** "Adaptive Mesh Refinement / Remeshing" sits under "## Planned" but is marked "Phase 1–2 Complete (2D only)"; similarly "2D Integrated Interface Curvature (Implemented)" lives under Planned. FEATURES' Planned/Implemented split cannot be trusted at section-heading level — read the per-item status lines.
4. **FEATURES.md redundant "Dam Break Test Case" under Planned** while "Dam Break Case (`cases_dynamic/dam_break/`)" is listed under Implemented (2D+3D, single-phase and two-phase variants).
5. **Oscillating droplet:** FEATURES says it "validates against Rayleigh-Lamb"; audit says dynamic Rayleigh-Lamb validation is *pending* and 3D has no baseline. Treat FEATURES' claim as overstated.
6. **README (pre-audit) described none of the core features** — audit's companion edit fixed this; verify current README before citing it.
7. Undocumented-but-working (README-invisible at audit time): domain builders, parametric surfaces, DEM, periodic BCs, EOS, GPU backends, conservation diagnostics, analytical framework, `int_bench/`, `tutorials/Case study 4 sphere.ipynb`, orphaned tutorials (`tutorials/domain_builder_tutorial.py`, `parametric_surfaces_tutorial.py`, `visualize_domains.py`, `visualize_parametric_surfaces.py`), high-quality fix write-ups `docs/3d_simplex_aware_*.md` and `docs/3d_multiphase_interface_pressure_fix.md`.
8. Case-study docs gap: 7/21 `cases_dynamic/` and 14/15 `cases_mean_flow/` lacked READMEs at audit time.

## 4. Reference cases & equations (from FEATURES)

- Hydrostatic column (`cases_dynamic/Hydrostatic_column/`): 1D/2D/3D pressure-operator validation; equilibrium (zero acceleration) + perturbation recovery.
- Hagen-Poiseuille (`cases_dynamic/Hagen_Poiseuile/` — note misspelled dir name): 2D planar + 3D channel; equilibrium residual convergence, developing flow from plug IC.
- Oscillating droplet (`cases_dynamic/oscillating_droplet/`): 2-phase 2D/3D; targets Rayleigh-Lamb frequency, viscous damping, Young-Laplace jump (see §3.5 caveat). Planned: analytical Lamb/Prosperetti initial velocity field.
- Dam break (`cases_dynamic/dam_break/`): 2D/3D, single- and two-phase; hydrostatic init, surface tension at liquid-air interface, no-slip walls.
- Adaptive remesh usage (when fixed): `symplectic_euler(HC, bV, dudt_fn, dt=1e-4, n_steps=500, dim=2, remesh_mode='adaptive', remesh_kwargs={'L_min': 0.5*h, 'L_max': 1.4*h, 'quality_target_deg': 20.0})`; tests `hyperct/tests/test_remesh.py` (36) + `ddgclib/tests/test_adaptive_remesh.py` (6).
- Remesh references: Persson & Strang 2004; Freitag & Ollivier-Gooch 1997; Jiao & Heath 2004; Compere et al. 2008; Quan & Schmidt 2007.

## 5. Audit-proposed backlog (DEVELOPMENT.md additions, §7)

1. Top-level public API in `__init__.py` + `__version__`. 2. Hygiene: legacy quarantine, delete `_dual_split_3d.py` stub + empty `barycentric/_duals.py`, honest module names. 3. Fix `hyperct/remesh/_operations_2d.py` mass-averaging (gate on multiphase stability). 4. Fix dual-volume-refresh leak (S-lane / A.3). 5. 3D `oscillation_score` harness + baseline. 6. README/index per case dir, tagged stable vs experimental. 7. Fundamentals/Architecture consolidation (4→1, 2→1). 8. Decide `int_bench/` home. 9. CI-checked test counts. 10. Constitutive-model hooks (demand-driven only).
