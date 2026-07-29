# 01 — Project Overview: ddgclib
> Sources: sources/architecture_and_conventions.md, sources/fundamentals.md, sources/library_audit_and_features.md, sources/development_status.md, code_map/* (all) | Written: 2026-07-02 by understand-and-document workflow

**Read this file first.** It orients you in the project and in `docs_temp/`. Detail lives in [sources/](sources/) and [code_map/](code_map/); the ranked open-problem list is in [06_known_issues_and_debug_history.md](06_known_issues_and_debug_history.md); case/test recipes are in [05_test_cases_and_validation.md](05_test_cases_and_validation.md).

---

## 1. What ddgclib is

**ddgclib** v0.4.3 (Alpha) is an experimental research library for **Lagrangian CFD via discrete differential geometry (DDG)**. The core idea: **the mesh IS the fluid**. Each primal vertex of a simplicial complex is a material parcel carrying velocity `v.u`, pressure `v.p` (volume-averaged over its dual cell, never a point value), and **fixed mass** `v.m`; vertices are advected by the flow. Forces are the integrated Cauchy stress over barycentric dual cells:

$$m_i\frac{d\mathbf{u}_i}{dt} = \sum_{j\in N(i)}\Big[\underbrace{-\tfrac12(p_i+p_j)\,\mathbf{A}_{ij}}_{\text{pressure flux}} + \underbrace{\tfrac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij}\!\cdot\!\mathbf{A}_{ij})}_{\text{viscous diffusion flux}}\Big] + \mathbf{F}_{\gamma,i}$$

where $\mathbf{A}_{ij}$ is the exact oriented dual-face area vector (outward from $i$, $\mathbf{A}_{ji}=-\mathbf{A}_{ij}$ so pairwise momentum is exact). No continuity equation is solved — mass conservation is by construction ($m_i$ const); density/pressure follow from an EOS on $\rho_i = m_i/\mathrm{Vol}_i^{\mathrm{dual}}$. The viscous term is **deliberately the diffusion form only** (transpose term dropped: the rank-1 face gradient has spurious discrete compressibility that produced O(1) equilibrium residuals — see [sources/fundamentals.md](sources/fundamentals.md) §2). Core implementation: `ddgclib/operators/stress.py` (`stress_force` :702, `stress_acceleration` :778, `dudt_i` alias :829). Full code map: [code_map/operators_stress.md](code_map/operators_stress.md).

Per-timestep pipeline (all dynamic integrators): retopologize (Delaunay or adaptive) → `compute_vd` barycentric duals + `batch_e_star` edge-area cache → compute all accelerations (EOS pressure resolved inside `dudt_fn`) → update u/x → apply BC set → callback/snapshot. See [code_map/integrators_and_bcs.md](code_map/integrators_and_bcs.md).

**Do NOT suggest switching to Eulerian fixed-mesh integration.** The library is Lagrangian by design; `euler_velocity_only` is Eulerian and exists only for validation/equilibrium checks.

## 2. Project status in one paragraph

The **equilibrium/static core is machine-precision-validated** (~1e-13): dual-face closure and antisymmetry, linear-field integrated gradients (all dual methods, jittered meshes), 2D Poiseuille and hydrostatic equilibrium, flat-interface multiphase zero force (2D+3D). The **central open problem** is carrying that accuracy into *dynamic* runs: every dynamic case is unstable or imperfect to some degree — retopology dual-volume churn is misread by the EOS as physical compression, the flagship 2D oscillating-droplet Rayleigh–Lamb validation FAILS (l2_error ≈ 5.6 vs target < 0.2; tail_growth 1.68 vs < 1.0), adaptive remesh is blocked by an upstream hyperct mass-averaging bug, and single-phase retopology leaks ~2–4% volume. Static Young–Laplace droplet residual floors are fully attributed and regression-locked (2D 2.2717e-3 = 100% curvature-stencil O(h) truncation; 3D 7.3768e-05 after a ×195 fix). Launch pad for debugging: [06_known_issues_and_debug_history.md](06_known_issues_and_debug_history.md).

## 3. Active vs legacy pipelines

| Pipeline | Status | Entry points |
|---|---|---|
| **Dynamic continuum (Cauchy stress)** | **Active core** | `ddgclib/operators/stress.py`, `ddgclib/dynamic_integrators/`, `ddgclib/operators/multiphase_stress.py`, `ddgclib/multiphase.py`, `ddgclib/eos/`, `cases_dynamic/` |
| Mean curvature flow | Legacy | `mean_flow_integrators/`, `_curvatures.py` (1717-line legacy), `_bubble.py`, `_capillary_rise*.py`, `_sessile.py`, `cases_mean_flow/` |
| DEM (particles) | Active, self-contained | `ddgclib/dem/` (124 tests; separate `ParticleSystem` data structure, two-way `FluidParticleCoupler`) |

Also active: domain builders (`ddgclib/geometry/domains/`, `DomainResult` contract), IC/BC classes (`initial_conditions.py`, `_boundary_conditions.py`), analytical validation (`ddgclib/analytical/`), data/recording (`ddgclib/data/`), visualization (`ddgclib/visualization/`, use `plot_primal` from `unified.py`). Deprecated shims (re-export `hyperct.ddg`, no sunset date): `barycentric/`, `circumcentric/`, `_plotting.py`, `_method_wrappers.py`. Dead code (per audit): `geometry/_dual_split_3d.py` (stub — the real 3D split lives in the misleadingly-named `geometry/_dual_split_2d.py`), empty `barycentric/_duals.py`, `_eos.py` shim, `legacy/plots.py`. **`import ddgclib` exposes nothing** — `ddgclib/__init__.py` is all commented out; import from submodules directly.

`cases_mean_flow/equil_bubble/` is a separate, currently-active *thermodynamics toy* (e-NRTL → Butler σ → Young–Laplace → Fritz detachment → VLE); it supplies the Fritz detachment physics reused by `cases_dynamic/electrolysis_bubble/` but is not on the dynamic stress pipeline.

## 4. The hyperct backend

All mesh data structures and DDG dual computation come from the external **hyperct** package, symlinked into the repo. **Verified symlink chain (2026-07-02):**

```
/home/endres/projects/ddgclib/hyperct
  -> /home/endres/projects/bilevel_param/hyperct
    -> /home/endres/projects/hyperct/hyperct   (final resolution)
```

(Both CLAUDE.md — `/home/stefan_endres/...` — and ARCHITECTURE.md — `../hyperct/hyperct` — state stale/imprecise targets.) Edits to "hyperct" files inside the repo edit the external checkout.

Key hyperct facts (full map: [code_map/hyperct_upstream.md](code_map/hyperct_upstream.md)):
- Vertices are identity-keyed by coordinate tuple `v.x`; reposition only via `HC.V.move()`; `v.vd` duals are only rebuilt by `compute_vd(HC, method="barycentric")` (requires `v.boundary` tagged on ALL vertices first).
- **`HC._simplices` cache invariant**: populated by `connect_and_cache_simplices`; MUST be invalidated via `invalidate_simplex_cache(HC)` after any unrouted topology change, else simplex-aware code silently falls back to flag-complex `v.nn` paths that produce ghost $K_{dim+1}$ cliques on Delaunay meshes.
- `hyperct.remesh` (interface-preserving adaptive remesh) is **2D-only** and has an open mass-inflation bug at `hyperct/remesh/_operations_2d.py:198` that blocks production use.

## 5. Environment facts (this machine)

- **Run everything with the `ddg` conda env python: `/home/endres/anaconda3/envs/ddg/bin/python`** (Python 3.13, numpy 1.26.3, matplotlib, pytest, ddgclib + hyperct importable). The sandbox `base`/system python imports ddgclib but **lacks matplotlib** (figures silently skipped) and **lacks pytest**.
- **Run from the repo root** `/home/endres/projects/ddgclib` (case scripts do `sys.path.insert(0, <repo_root>)` and expect `python cases_dynamic/<case>/<script>.py` paths). Exception: `cases_dynamic/oscillating_droplet_p_ref/` scripts run from inside that folder.
- Sandbox is disabled for this workflow; shell state does not persist between calls — use absolute paths.
- Fast test suite: `cd /home/endres/projects/ddgclib && /home/endres/anaconda3/envs/ddg/bin/python -m pytest ddgclib/tests/ -m "not slow"` (~18–31 s; **808 passed, 1 skipped, 2 xfailed** at the 2026-06-08 audit; ~813 test functions in 37 files). ALL hard-coded test counts in CLAUDE.md (~415), ARCHITECTURE.md (~604), Fundamentals (488) are stale — re-run pytest for truth.
- Canonical integrator recipe (the ONLY correct way): `dudt_fn = functools.partial(dudt_i, dim=2, mu=0.1, HC=HC)` — passing `HC`/`dim` via `**dudt_kwargs` raises "multiple values for HC" (some module docstrings show the broken pattern; they are stale).
- Commit prefixes: `ENH:`, `BUG:`, `MAINT:`.

## 6. Repo layout (orientation)

- `ddgclib/` — package: `operators/` (stress = core physics, multiphase_stress, curvature_2d, surface_tension, mass_redistribution), `dynamic_integrators/`, `multiphase.py`, `eos/`, `geometry/` (domains/, _dual_split_2d.py, periodic), `analytical/`, `data/`, `visualization/`, `dem/`, `tests/`
- `hyperct` — symlink (see §4)
- `cases_dynamic/` — dynamic case studies (oscillating_droplet is the flagship; inventory in [05_test_cases_and_validation.md](05_test_cases_and_validation.md))
- `cases_mean_flow/` — legacy mean-flow cases + the equil_bubble thermodynamics toy
- `benchmarks/` — integrated-operator benchmark runner (`run_integrated_benchmarks.py`) + `benchmarks/dynamic/`
- Top-level docs (all distilled here): `Fundamentals.md` (math, authoritative of the 4 Fundamentals versions), `ARCHITECTURE.md` (+ untracked `ARCHITECTURE_public.md`), `DEVELOPMENT.md` (feature tracker), `FEATURES.md` (heading-level status untrustworthy), `LIBRARY_AUDIT.md` (2026-06-08), `debugging_plan.md` (canonical stability-investigation log, reverse-chronological)

## 7. How docs_temp/ is organized

```
docs_temp/
├── 00_INDEX.md                         <- entry point: annotated map of this KB + fast paths
├── 01_project_overview.md              <- this file (orientation)
├── 02_physics_foundations.md           <- the discrete physics: equations, invariants, EOS, floors
├── 03_architecture_code_map.md         <- package layout, hyperct boundary, vertex model, APIs
├── 04_solver_pipeline.md               <- per-step loop, _retopologize, BC semantics, dt control
├── 05_test_cases_and_validation.md     <- case inventory, run commands, analytical refs, pytest recipes
├── 06_known_issues_and_debug_history.md<- ranked open problems, bug status, experiment history
├── sources/                            <- distilled top-level project documents
│   ├── fundamentals.md                 <- the math (current face-centered FVM + abandoned pointwise form)
│   ├── architecture_and_conventions.md <- module graph, 5-step workflow, vertex model, conventions
│   ├── development_status.md           <- DEVELOPMENT.md feature tracker distilled
│   ├── library_audit_and_features.md   <- per-module correct/suspect/dead verdicts (LIBRARY_AUDIT + FEATURES)
│   └── debugging_plan_distilled.md     <- full stabilisation-roadmap log: probes, refuted hypotheses, floors
└── code_map/                           <- distilled from the code itself
    ├── operators_stress.md             <- stress.py: dual_area_vector, fluxes, caches, EOS hook
    ├── multiphase_surface_tension.md   <- multiphase layer, interface subcomplex, EOS classes, ST forces
    ├── integrators_and_bcs.md          <- per-step order, retopologize machinery, all BC/IC classes
    ├── hyperct_upstream.md             <- mesh backend: caches, compute_vd, e_star/v_star, remesh
    ├── cases_dynamic_inventory.md      <- every case dir: run commands, params, success criteria
    └── validation_and_tests.md         <- analytical pkg, StateHistory/conservation, test-suite map
```

Reading order for a new debugging agent: this file → [06_known_issues_and_debug_history.md](06_known_issues_and_debug_history.md) → [05_test_cases_and_validation.md](05_test_cases_and_validation.md) → the specific code_map file for the subsystem you touch. For physics work start at [02_physics_foundations.md](02_physics_foundations.md); for integrator/BC work at [04_solver_pipeline.md](04_solver_pipeline.md); for architecture/API questions at [03_architecture_code_map.md](03_architecture_code_map.md). Full annotated map + fast paths: [00_INDEX.md](00_INDEX.md).

## 8. Top cross-cutting warnings (repeated because they bite)

1. `v.p` is a **volume average** $(1/\mathrm{Vol}_i)\int P\,dV$ — never assign or compare point values; validate only with `ddgclib.analytical` integrated comparisons (but note: their **3D branches silently fall back to point-wise** $P(x)\cdot Vol$).
2. Setup and runtime `split_method` (multiphase dual-volume split) MUST match, or ρ jumps ×147 on first retopo.
3. Never assert 3D volume conservation from step 0 — the one-shot |dV/V0| ≈ 0.305 at step 0→1 is boundary-shell zeroing, not drift; assert from step ≥ 2.
4. `diagnose_a5_bisection.py` without `--redistribute-mass` overrides the production default (True) and reproduces the pre-fix 1.44e-3 blow-up — expected, not a regression.
5. FEATURES.md section headings (Planned vs Implemented) cannot be trusted; read per-item status lines. DEVELOPMENT.md has duplicate/contradictory entries (see [sources/development_status.md](sources/development_status.md)).
6. Boundary edges get $\mathbf{A}_{ij}=\mathbf{0}$ (zero flux) in `dual_area_vector` — boundary vertices have incomplete surface integrals; there is no boundary-flux closure term. This drives outlet backflow and corner spurious accelerations.
7. `cases_dynamic/cube2droplet/` scripts are currently unrunnable (import `cases_dynamic.Cube2droplet`, capital C, verified ModuleNotFoundError).
