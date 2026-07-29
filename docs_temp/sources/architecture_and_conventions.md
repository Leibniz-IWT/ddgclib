# Architecture and Conventions (ddgclib)
> Sources: ARCHITECTURE.md, ARCHITECTURE_public.md, README.md, CLAUDE.md (project) | Written: 2026-07-02 by understand-and-document workflow

Project: **ddgclib** v0.4.3 (Alpha), Lagrangian DDG CFD library. The mesh IS the fluid: vertices carry `u`, `p`, `m` and are advected; forces are integrated Cauchy stress over dual cells. Python 3.9+ (dev on 3.13, conda env `ddg`, `environment.yml`). Install: `pip install -e .`. Required deps: `numpy`, `scipy`, `hyperct`. Optional extras: `[vis]` (matplotlib, polyscope), `[data]` (pandas), `[gpu]` (torch), `[dev]` (pytest, pytest-cov). Zenodo DOI 10.5281/zenodo.8010952.

Two pipelines: the **dynamic continuum (Cauchy stress) pipeline is the active core**; the **mean curvature flow** pipeline (`mean_flow_integrators/`, `cases_mean_flow/`, `_curvatures.py`, `_bubble.py`, `_capillary_rise*.py`, `_sessile.py`) is legacy (README.md:8). Never suggest switching to Eulerian fixed-mesh integration; `euler_velocity_only` is Eulerian and exists only for validation/equilibrium checks (CLAUDE.md).

## Module Dependency Graph (from ARCHITECTURE.md:5-85)

```
hyperct (SYMLINK — see "Stale claims" below for actual target)
    Complex                       simplicial complex backend (vertices/edges/triangles)
    ddg/                          compute_vd(HC, method="barycentric"|"circumcentric", backend=...),
                                  e_star(v,dim,HC), v_star(v,dim,HC), d_area(v,dim,HC),
                                  _geometry.py (normalized, area_of_polygon, volume_of_geometric_object),
                                  plot_dual.py (plot_dual_mesh_1D/2D/3D)
    remesh/                       interface-preserving adaptive remesh (2D only; 3D on roadmap):
                                  _driver.adaptive_remesh(HC, dim=2, L_min, L_max, quality_target_deg, max_iterations),
                                  _operations_2d (edge_split_2d, edge_collapse_2d, edge_flip_2d),
                                  _quality (triangle_min_angle, triangle_aspect_ratio, mesh_quality_histogram),
                                  _interface (is_interface_edge, can_flip, can_collapse)
    _plotting.py                  plot_complex, animate_complex
    _backend.py                   get_backend("numpy"|"torch"|"gpu"|"multiprocessing")
        |
        v
ddgclib/
    initial_conditions.py         IC classes (apply to mesh)
    _boundary_conditions.py       BC classes + BoundaryConditionSet
    _compat.py                    utilities not in hyperct (triang_dual, etc.)
    operators/
        _registry.py              MethodRegistry (pluggable methods)
        curvature.py              Curvature_i, Curvature_ijk
        area.py                   Area_i, Area_ijk, Area, DualArea_i
        volume.py                 Volume, Volume_i
        stress.py                 CORE physics: dual_area_vector, dual_volume,
                                  velocity_difference_tensor, strain_rate, cauchy_stress,
                                  stress_force, stress_acceleration, dudt_i,
                                  scalar_gradient_integrated
        gradient.py               thin wrappers: pressure_gradient(v,dim,HC) -> stress_force(v,dim,mu=0,HC),
                                  acceleration(v,dim,mu,HC) -> stress_acceleration; velocity_laplacian
    analytical/
        _divergence_theorem.py    integrated_gradient_1d/2d/3d (Gauss-Legendre)
        _sympy_integration.py     integrated_gradient_sympy_1d/2d/3d (exact)
        _integrated_comparison.py integrated_pressure_error, integrated_l2_norm,
                                  compare_stress_force, volume_averaged_scalar
    _method_wrappers.py           backward-compat shim -> operators/ (registries
                                  _curvature_i_methods, _area_i_methods, ...; toy methods
                                  loaded from benchmarks/_benchmark_toy_methods.py)
    dynamic_integrators/
        _integrators_dynamic.py   euler, symplectic_euler, rk45, euler_velocity_only, euler_adaptive
        _simulation.py            DynamicSimulation, SimulationParams
    data/
        _io.py                    save_state, load_state (JSON)
        _history.py               StateHistory (time-series recording)
    visualization/
        unified.py                plot_primal, plot_dual, plot_fluid, dynamic_plot_fluid
                                  (dynamic_plot_fluid takes phase_field, interface_field, reference_R)
        multiphase.py             record_multiphase_frame, dynamic_plot_multiphase
        matplotlib_1d/2d/3d.py    field overlays; extract_slice_profile (3D)
        polyscope_3d.py           optional polyscope
        animation.py              animate_scalar_1d, animate_scalar_2d
    scripts/view_polyscope.py     generic snapshot viewer:
                                  python -m ddgclib.scripts.view_polyscope --snapshots <dir>
    geometry/domains/             DomainResult builders — 2D: rectangle, l_shape, disk, annulus;
                                  3D: box, cylinder_volume, pipe, ball; projections cube_to_disk,
                                  cube_to_sphere (laws 'sinusoidal' default, 'linear', 'power', 'log')
    geometry/periodic             periodic BCs: ghost-cell merge, minimum-image duals (README only)
    dem/                          _particle (Particle, ParticleSystem), _contact (ContactDetector,
                                  Contact), _force_models (HertzContact, LinearSpringDashpot,
                                  ContactForceModel ABC), _integrators (dem_velocity_verlet,
                                  dem_symplectic_euler, dem_step), _bonds (SinterBond, BondManager;
                                  Frenkel neck growth), _liquid_bridge (LiquidBridge,
                                  LiquidBridgeManager; Lian et al. 1993), _coupling
                                  (FluidParticleCoupler: drag, IDW interp, feedback), _io
                                  (save_particles, load_particles, import_particle_cloud),
                                  _visualization (plot_particles, plot_bridges, plot_bonds)
    barycentric/, circumcentric/  DEPRECATED shims re-exporting hyperct.ddg
```

Modules in ARCHITECTURE_public.md / README but ABSENT from internal ARCHITECTURE.md graph (internal doc is stale here): `multiphase.py` (MultiphaseSystem sharp-interface framework), `eos/` (`TaitMurnaghan`, `IdealGas`, `MultiphaseEOS` dispatcher), `operators/surface_tension.py`, `operators/curvature_2d.py`, `operators/multiphase_stress`, parametric surfaces in `ddgclib.geometry` (`sphere`, `catenoid`, `cylinder`, `hyperboloid`, `torus`, `plane` with translate/scale/rotate), `ddgclib.geometry.periodic`.

## Physics (operators/stress.py — core)

Integrated Cauchy momentum on Lagrangian parcels:

$$m_i \frac{dv_i}{dt} = F^{stress}_i = \sum_j \sigma_f \cdot A_{ij}, \qquad \sigma_f = \tfrac12(\sigma_i + \sigma_j)$$

Newtonian constitutive: $\sigma = -p I + 2\mu\varepsilon$, $\varepsilon = \tfrac12(\nabla u + \nabla u^T)$ (from `strain_rate(du)` on `velocity_difference_tensor` output, a dim×dim DDG-integrated tensor). $A_{ij}$ = oriented dual face area vector, outward from parcel i (`dual_area_vector(v_i, v_j, HC, dim)`). Bottom-up call chain: dual_area_vector / dual_volume → velocity_difference_tensor → strain_rate → cauchy_stress(p, du, mu, dim) → stress_force(v, dim, mu, HC) → stress_acceleration = F/m_i → `dudt_i` (alias). Claimed validated to machine precision at equilibrium for Hagen-Poiseuille and hydrostatic column (README.md:6); public doc adds oscillating droplet (Rayleigh-Lamb) validation (ARCHITECTURE_public.md:182).

## Simulation Workflow

Internal ARCHITECTURE.md:87-113 defines the canonical **5-step workflow**:

1. **DOMAIN** — `HC = Complex(d)`, `HC.triangulate()`, `HC.refine_all()`, `compute_vd(HC)`; or domain builders: `result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)`; `HC, bV = result.HC, result.bV`; `result.boundary_groups` (e.g. {'walls','inlet','outlet','bottom_wall','top_wall'}), `result.metadata['volume']`. Builders call `tag_boundaries()` automatically. 3D projected domains (cylinder, ball) need `_retopologize()` before `compute_vd()` (integrators do this automatically).
2. **BOUNDARY CONDITIONS** — `bV = identify_cube_boundaries(HC, lb, ub)`; `bc_set = BoundaryConditionSet().add(NoSlipWallBC(...), verts).add(DirichletPressureBC(...), verts)`; `diagnostics = bc_set.apply_all(HC, bV, dt)`. ABC: `BoundaryCondition.apply(self, mesh, dt, target_vertices=None) -> int`. Concrete: `NoSlipWallBC`, `DirichletVelocityBC`, `DirichletPressureBC`, `NeumannBC`, `OutletDeleteBC`, `PeriodicInletBC` (+ `PositionalNoSlipWallBC` per CLAUDE.md). Lagrangian flow-through: `PeriodicInletBC` injects vertices from a ghost mesh at inlet; `OutletDeleteBC` deletes exited vertices; `boundary_filter` in integrators controls which topological boundary vertices are frozen (typically only walls, not inlet/outlet).
3. **INITIAL CONDITIONS** — `ic = CompositeIC(ZeroVelocity, HydrostaticPressure, UniformMass); ic.apply(HC, bV)`. ABC: `InitialCondition.apply(self, HC, bV: set) -> None`. Concrete: `UniformPressure`, `HydrostaticPressure`, `LinearPressureGradient`, `ZeroVelocity`, `UniformVelocity`, `PoiseuillePlanar`, `HagenPoiseuille3D`, `CustomFieldIC`, `UniformMass(total_volume=..., rho=...)`, `CompositeIC`.
4. **INTEGRATOR** — `dudt_fn = partial(dudt_i, dim=d, mu=mu, HC=HC)` then `t = euler(HC, bV, dudt_fn, dt, n_steps, bc_set=bc_set)`; OR high-level runner: `sim = DynamicSimulation(HC, bV, params); sim.set_initial_conditions(ic); sim.set_boundary_conditions(bc_set); sim.set_acceleration_fn(dudt_i); sim.run(callback=history.callback)`.
5. **POST-PROCESS** — `save_state(HC, bV)`, `history.query_*()`, `plot_fluid(HC, bV)`.

(ARCHITECTURE_public.md:50-58 compresses this to 4 steps: domain → dual mesh → physics → integrate, making `compute_vd` an explicit step. CLAUDE.md expands to 8 steps adding boundary tagging, visualization, animation. Same pipeline, different granularity.)

**Integrator interface** (all 5 integrators): `integrator(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, bc_set=None, **dudt_kwargs) -> float` (returns time t). `bc_set` is applied after each step; `callback` auto-detects old 3-arg vs new 5-arg signatures. **Critical pattern**: `HC` is the integrator's first positional arg, so bind `dim`, `mu`, `HC` into `dudt_i` via `functools.partial` — passing them as `**dudt_kwargs` raises "multiple values for argument 'HC'". Primary Lagrangian integrators: `symplectic_euler`, `euler`, `rk45` (update velocity AND position); `euler_velocity_only` is Eulerian fixed-mesh, validation only; `euler_adaptive` adaptive stepping. Remeshing: pass `remesh_mode='adaptive'` (+ optional `remesh_kwargs={...}`) to any dynamic integrator or to `_retopologize`; default `remesh_mode='delaunay'` (backward compatible). Example remesh call: `adaptive_remesh(HC, dim=2, L_min=0.5*h, L_max=1.4*h, quality_target_deg=20.0, max_iterations=3)`.

**Backends**: `compute_vd(HC, method="barycentric", backend="torch"|"gpu")`; NumpyBackend (default), MultiprocessingBackend, TorchBackend (CPU), CudaBackend (auto-detect via `backend="gpu"`).

**DEM coupled loop** (ARCHITECTURE.md:144-161): per step — `symplectic_euler(HC, bV, dudt_fn, dt, n_steps=1)` → `coupler.fluid_to_particle(dt)` → `dem_step(ps, detector, model, dt, dim, n_sub=1, bridge_manager=bridge_mgr)` → `coupler.particle_to_fluid(dt)`. Setup: `ps = import_particle_cloud(positions, radii, rho_s)`, `ContactDetector(ps)` (spatial hash broad-phase + sphere-sphere narrow-phase), `HertzContact()`, `FluidParticleCoupler(HC, ps, dim, mu)`, `LiquidBridgeManager(gamma=0.072)`, `BondManager()`. `Particle.sphere(x, radius, rho_s, dim)` auto-computes mass/inertia. DEM particles are a separate data structure from HC.

## Vertex Data Model

Every vertex `v` in `HC.V` (iterate `for v in HC.V`) carries (ARCHITECTURE.md:165-174):
- `v.x` — coordinate tuple, used as cache key / vertex identity
- `v.x_a` — numpy array position
- `v.nn` — set of 1-ring neighbor vertices
- `v.u` — velocity vector (`np.ndarray`, dim-sized)
- `v.p` — pressure, scalar float, **volume-averaged over dual cell** (see FVM convention)
- `v.m` — mass, scalar float
- `v.vd` — dual vertices, populated by `compute_vd()` (listed only in internal doc; public doc omits it)
- `v.boundary` — bool, MUST be set on all vertices before `compute_vd()`; tag via `for v in HC.V: v.boundary = v in bV`

Multiphase vertices additionally carry `v.phase` (sharp interface preserved by remeshing) and `is_interface` is a recordable field. `bV` = set of boundary vertex objects, threaded through most computation functions. Prefix `b_` = barycentric-dual variant (e.g. `b_curvatures`). Color convention: `coldict` in `ddgclib._misc`, `'db'` dark blue points/edges, `'lb'` light blue faces.

## FVM Volume-Averaged Field Convention

All scalar vertex fields are dual-cell volume averages, NOT point samples:

$$v.p = \frac{1}{\mathrm{Vol}_i}\int_{V_i} P(x)\, dV \quad\Rightarrow\quad v.p \cdot \mathrm{Vol}_i = \int P\, dV \text{ to machine precision (polynomial } P\text{)}$$

- NEVER assign `v.p = P(x_vertex)`. Use `volume_averaged_scalar` or IC classes (`HydrostaticPressure`, `LinearPressureGradient`) which do this automatically when duals exist.
- Validation must use integrated comparisons from `ddgclib.analytical`: `integrated_pressure_error` (|p_i·Vol_i − ∫P dV| per vertex), `integrated_l2_norm` (volume-weighted L2), `compare_stress_force` (force-balance diagnostic). NEVER point-wise `abs(v.p - P(x_vertex))` — conflates discretization error with point-vs-average mismatch.
- Benchmarks: `python benchmarks/run_integrated_benchmarks.py` (flags: `--linear-only` machine-precision check, `--comparison --dim 2`, `--convergence --dim 2`); tests `pytest ddgclib/tests/test_integrated_validation.py -v -m "not slow"`; visual notebooks `benchmarks/notebooks/01-03`.

## Dynamic Case Study File Convention (ARCHITECTURE.md:115-142 + CLAUDE.md)

```
cases_dynamic/<name>/
    <name>_2D.py, <name>_3D.py   # main scripts
    view_polyscope.py            # interactive replay (or point to generic viewer)
    README.md                    # run instructions + output listing (required)
    src/_setup.py, _params.py, _analytical.py, _plot_helpers.py
    fig/                         # plots + .mp4 animations
    results/snapshots/           # StateHistory JSON snapshots
```
Outputs go in the case dir, never project root. Animation workflow: (1) `history = StateHistory(fields=['u','p',...], record_every=N, save_dir=_SNAPSHOTS)`; (2) pass `history.callback` to integrator; (3) `dynamic_plot_fluid(history, HC, save_path=os.path.join(_FIG,'anim.mp4'))`; (4) multiphase: record `'phase'`, `'is_interface'` and pass `phase_field='phase'`, `interface_field='is_interface'`; (5) 3D replay: `python -m ddgclib.scripts.view_polyscope --snapshots <dir>`.

## Test Architecture

All tests in `ddgclib/tests/`, pytest + unittest, slow 3D cases marked `@pytest.mark.slow`. Fast suite ~18s. Commands: `pytest ddgclib/tests/ -v -m "not slow"` (fast), `pytest ddgclib/tests/ -v` (all), `pytest ddgclib/tests/test_dem_*.py -v` (DEM), `pytest ddgclib/tests/test_stress.py -v -m "slow"` (3D validation), `pytest ddgclib/tests/test_manuscript_tutorials.py -v`. Test data: `test_data/` (JSON). Notebooks in `tutorials/` and `cases_mean_flow/` double as integration tests. PyCharm run configs in `.idea/runConfigurations/`.

Per-file counts (ARCHITECTURE.md:283-314): test_initial_conditions 20, test_boundary_conditions 15, test_operators 11, test_integrated_validation 47, test_stress 58 (50 fast + 8 slow: 2D/3D dual geometry closure/antisymmetry/magnitude/volume; strain rate; Cauchy stress; uniform-field zero forces; gradient-wrapper back-compat; dudt_i with all 5 integrators; Hagen-Poiseuille 2D/3D equilibrium/developing-flow/profile; hydrostatic 2D/3D force direction + viscous damping), test_dynamic_integrators 24, test_data 16, test_visualization 54 (1 skip: polyscope), test_case_hydrostatic 10, test_case_hagen_poiseuille 8, test_gpu_backend 11 (5 skip w/o torch/CUDA), test_manuscript_tutorials 60 (4 skip). DEM: test_dem_particle 27, test_dem_contact 15, test_dem_forces 16, test_dem_integrators 13, test_dem_bonds 17, test_dem_liquid_bridge 16 (1 slow), test_dem_coupling 22 — CLAUDE.md says "124 fast + 1 slow" DEM total.

**Count inconsistency**: ARCHITECTURE.md heading says "~604 tests" (line 283) but its own total line says "~415 tests (407 passed, 8 skipped)" (line 314); the per-file numbers sum to ~526. ARCHITECTURE_public.md says "~600 tests". CLAUDE.md says "~415 tests, 407 pass, 8 skipped". Treat exact counts as unreliable; re-run pytest for truth.

Global testing rule (user CLAUDE.md): always run the full test suite after changes; confirm 0 failures before reporting completion.

## Commit Conventions

Prefixes: `ENH:` (enhancement), `BUG:` (bugfix), `MAINT:` (maintenance/refactoring). Recent history conforms (e.g. "ENH: commit new case files", "MAINT: Update md files").

## Internal vs Public Architecture Doc Differences

| Aspect | ARCHITECTURE.md (internal) | ARCHITECTURE_public.md (public, untracked) |
|---|---|---|
| Detail level | Full ASCII dependency graph w/ file-level entries, function lists | Condensed tree; module-level only |
| Workflow | 5 steps (domain/BC/IC/integrate/post-process), includes DynamicSimulation alternative + DEM coupled-loop diagram | 4 steps (domain/dual-mesh/physics/integrate); no DEM loop diagram |
| Vertex model | Includes `v.vd` | Omits `v.vd`; annotates `v.p` as "volume-averaged over dual cell" inline |
| Multiphase/EOS | Absent from module graph (stale) | Present: `multiphase.py`, `eos/`, `operators/surface_tension.py`, `operators/curvature_2d.py`, MultiphaseSystem section |
| Case-study conventions | Full `cases_dynamic/<name>/` layout + animation workflow | Absent |
| Test detail | Per-file counts, "~604" heading vs "~415" total (self-contradictory) | "~600 tests", one-line commands, names validation cases incl. oscillating droplet (Rayleigh-Lamb) |
| ICs/BCs | Full concrete class lists + ABC signatures | Two-line usage example only |
| hyperct | "symlink -> ../hyperct/hyperct" | GitHub link only, no symlink claim |
| σ_f definition | Explicit face average σ_f = ½(σ_i + σ_j) | Just σ_f · A_ij |
| Analytical module internals | Lists _divergence_theorem/_sympy_integration/_integrated_comparison functions | "analytical/ Analytical solutions for validation" |

## Stale / Contradictory Claims Noticed

1. **Test counts** self-contradictory (see above): 604 vs 415 vs ~600 vs per-file sum ~526.
2. **hyperct symlink target**: ARCHITECTURE.md:8 says `../hyperct/hyperct`; CLAUDE.md says `/home/stefan_endres/projects/hyperct/hyperct`; the actual symlink (verified 2026-07-02) is `/home/endres/projects/ddgclib/hyperct -> /home/endres/projects/bilevel_param/hyperct`. Both docs stale.
3. **Internal module graph omits multiphase/eos/surface_tension/curvature_2d/periodic**, which README (lines 27-37) and public doc describe as core capabilities — internal ARCHITECTURE.md predates those features.
4. README.md:16-17 claims all dynamic integrators "retopologize the moving mesh each step", but `euler_velocity_only` is fixed-mesh Eulerian (CLAUDE.md) — imprecise for that integrator.
5. README.md:138 references `./manuscript_figures`; the working tree contains `manuscript_figures_dfg/` (untracked artifact suggests possible rename/divergence). Original manuscript code tagged `v0.3.1-alpha`.
6. README links FEATURES.md (README.md:44) — existence not verified here.
7. ARCHITECTURE_public.md is untracked (git status), i.e. a draft not yet committed.
