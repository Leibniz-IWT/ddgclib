# DEVELOPMENT.md

Feature tracker for the dynamic continuum simulation framework in ddgclib.

## Feature: Domain Builder Module
Status: Complete
- [x] `DomainResult` dataclass with `HC`, `bV`, `boundary_groups`, `metadata`, `tag_boundaries()`, `summary()`
- [x] Projection engine: `cube_to_disk()`, `cube_to_sphere()`, `DISTRIBUTION_LAWS` registry
- [x] Boundary group helpers: `identify_face_groups()`, `identify_radial_boundary()`, `identify_all_boundary()`
- [x] 2D builders: `rectangle()`, `l_shape()`, `disk()`, `annulus()`
- [x] 3D builders: `box()`, `cylinder_volume()`, `pipe()`, `ball()`
- [x] Public API: `ddgclib.geometry.domains` and `ddgclib.geometry` re-exports
- [x] Integration with `compute_vd()`, `BoundaryConditionSet`, `CompositeIC`
- [x] Tutorial: `tutorials/domain_builder_tutorial.py`
- [x] 56 unit tests passing
- [ ] Add `channel_with_obstacle()` (DFG benchmark cylinder-in-channel)
- [ ] Add general surface-of-revolution builder
- [x] Polyscope viewer: `tutorials/visualize_domains.py` with boundary group coloring

## Feature: Initial Conditions Module
Status: Complete
- [x] `InitialCondition` ABC with `apply(HC, bV)` interface
- [x] `CompositeIC` for combining multiple ICs
- [x] `UniformPressure`, `HydrostaticPressure`, `LinearPressureGradient` (scalar `v.P`)
- [x] `ZeroVelocity`, `UniformVelocity`
- [x] `PoiseuillePlanar` with `analytical_velocity()` method
- [x] `HagenPoiseuille3D` with `analytical_velocity()` method
- [x] `CustomFieldIC`, `UniformMass`
- [x] 20 unit tests passing

## Feature: Boundary Conditions Module
Status: Complete
- [x] `identify_boundary_vertices(HC, criterion_fn)` helper
- [x] `identify_cube_boundaries(HC, lb, ub, dim)` helper
- [x] `BoundaryConditionSet` container with `add()` and `apply_all()`
- [x] `NoSlipWallBC`, `DirichletVelocityBC`, `DirichletPressureBC`, `NeumannBC`
- [x] Backward-compatible `BoundaryCondition.apply()` with optional `target_vertices`
- [x] Existing `OutletDeleteBC`, `PeriodicInletBC`, `MeshAdvancer` preserved
- [x] 15 unit tests passing

## Feature: Operators Package
Status: Complete
- [x] `MethodRegistry` class in `operators/_registry.py`
- [x] `Curvature_i`, `Curvature_ijk` moved to `operators/curvature.py`
- [x] `Area_i`, `Area_ijk`, `Area`, `DualArea_i` in `operators/area.py`
- [x] `Volume`, `Volume_i` in `operators/volume.py` (fixed `_volume_curved` forward-ref bug)
- [x] `pressure_gradient`, `velocity_laplacian`, `acceleration` in `operators/gradient.py`
- [x] `_method_wrappers.py` converted to backward-compat shim
- [x] 11 unit tests passing

## Feature: Enhanced Dynamic Integrators
Status: Complete
- [x] `bc_set` parameter added to `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`
- [x] `_invoke_callback()` auto-detects old (3-arg) vs new (5-arg) callback signatures
- [x] `euler_adaptive` with CFL-based adaptive time stepping
- [x] `DynamicSimulation` convenience runner class
- [x] `SimulationParams` dataclass
- [x] Backward compatibility: all integrators work without `bc_set`
- [x] 24 unit tests passing

## Feature: Data Handling Package
Status: Complete
- [x] `save_state(HC, bV, t, fields, path)` to JSON
- [x] `load_state(path)` reconstructs HC, bV, fields
- [x] Round-trip save/load verified
- [x] `StateHistory` class for time-series recording
- [x] `callback` method for integrator integration
- [x] `query_vertex()`, `query_field_at_time()`, `query_diagnostics()`
- [x] Configurable `record_every` for memory control
- [x] 16 unit tests passing

## Feature: Visualization Package
Status: Complete
- [x] `plot_scalar_field_1d`, `plot_velocity_profile_1d` (1D)
- [x] `plot_scalar_field_2d`, `plot_vector_field_2d`, `plot_mesh_2d` (2D)
- [x] `plot_scalar_field_3d` (3D scatter)
- [x] `extract_slice_profile(HC, axis, position, tol)` utility
- [x] `polyscope_3d` module (optional dependency, import-guarded)
- [x] `animate_scalar_1d`, `animate_scalar_2d` from `StateHistory`
- [x] All functions accept optional `ax` parameter for subplot composition
- [x] Unified `plot_primal(HC, ...)` wrapper — auto-detects `HC.dim` (1D/2D/3D)
- [x] Unified `plot_dual(HC, ...)` wrapper — auto-detects `HC.dim` (1D/2D/3D, 3D requires `vertex=`)
- [x] Abstract `scalar_field` parameter on primal/dual (e.g. `'P'`, any vertex attribute)
- [x] Abstract `vector_field` parameter on primal/dual (e.g. `'u'`, colored arrows at vertices)
- [x] 3D vector field support (matplotlib quiver + polyscope quantities)
- [x] Polyscope unified wrappers: `plot_primal_polyscope`, `plot_dual_polyscope`
- [x] Dual field interpolation from primal vertices to dual vertices (averaging)
- [x] Optional `save_path` and `dpi` on all plotting wrappers (default `results/fig/`)
- [x] `plot_fluid(HC, bV, t=0.0, ...)` — multi-panel snapshot (pressure + velocity), timestamp label
- [x] `dynamic_plot_fluid(history, HC, ...)` — FuncAnimation video from StateHistory snapshots
- [x] `dynamic_plot_fluid_polyscope(...)` — per-frame screenshots + ffmpeg/pillow compile
- [x] `_restore_snapshot()` helper to write StateHistory data back onto HC vertices per frame
- [x] `dynamic_plot_fluid` supports Lagrangian meshes (changing topology between frames)
- [x] `_compile_video_from_frames()` — ffmpeg concat for mp4, PIL for gif
- [x] 53 unit tests passing (1 skipped: polyscope not installed)

## Feature: Hydrostatic Column Case Study
Status: Complete
- [x] `setup_hydrostatic(dim, ...)` in `cases_dynamic/Hydrostatic_column/src/_setup.py`
- [x] Supports 1D, 2D, 3D with configurable gravity axis
- [x] Uses `CompositeIC(ZeroVelocity, HydrostaticPressure, UniformMass)`
- [x] Pressure profile verification test
- [x] Perturbation recovery test (KE dissipation)
- [x] 10 unit tests passing

## Feature: Hagen-Poiseuille Case Study
Status: Complete
- [x] `setup_poiseuille_2d(G, mu, ...)` in `cases_dynamic/Hagen_Poiseuile/src/_setup.py`
- [x] Uses new IC/BC classes (replaces manual loops)
- [x] Analytical profile verification (velocity, pressure, wall, centerline)
- [x] Developing flow test (plug flow with no-slip BCs)
- [x] 8 unit tests passing

## Feature: Cube Flow Case Study
Status: In Progress
- [x] PeriodicInletBC enhanced (field copying, proper clone, configurable period)
- [x] StateHistory time-based recording (`record_every_t`) + disk persistence (`save_dir`)
- [x] `setup_cube_flow(dim)` in `cases_dynamic/cube_flow/src/_setup.py`
- [x] 1D/2D/3D case scripts with visualization
- [x] `dynamic_plot_fluid` fixed for Lagrangian meshes (changing topology)
- [x] 1D and 2D videos generated and verified
- [ ] 3D video generation (slow, in progress)
- [ ] Unit tests (pending visual confirmation)

## Feature: Dynamic Capillary Rise Case Study
Status: Not Started (deferred)
- [ ] Requires surface tension forces on meniscus
- [ ] Depends on curvature operators from mean flow pipeline
- [ ] Will be planned after 7a and 7b are validated

## Feature: Documentation
Status: Complete
- [x] DEVELOPMENT.md (this file)
- [x] ARCHITECTURE.md
- [x] FEATURES.md

## Feature: DDG Migration to hyperct.ddg
Status: Complete
- [x] Phase 1: Remove ddgclib.barycentric/_duals.py and circumcentric code
  - [x] Update operators/gradient.py imports to `hyperct.ddg.e_star`
  - [x] Update operators/area.py imports to `hyperct.ddg.d_area`
  - [x] Update _plotting.py import (`du` from operators/gradient)
  - [x] Update test imports (test_operators, test_case_hydrostatic)
  - [x] Update case study imports (Hagen-Poiseuille, Dynamic caprise tube)
  - [x] Create `ddgclib/_compat.py` for orphan utility functions (triang_dual, etc.)
  - [x] Replace barycentric/ and circumcentric/ with deprecation shims
  - [x] Add GPU/backend tests (`test_gpu_backend.py`, 11 tests)
  - [x] Update environment.yml (torch) and setup.py (gpu extras)
- [x] Phase 2: Update examples and notebooks
  - [x] Update 4 Jupyter notebooks (Hagen-Poiseuille, Poiseuille, geometry_dev, caprise)
  - [x] Verify no remaining direct barycentric/circumcentric imports
- [x] Phase 3: Refactor visualization
  - [x] `plot_primal` delegates to `hyperct._plotting.plot_complex` with coldict colors
  - [x] `plot_dual` delegates to `hyperct.ddg.plot_dual` (2D/3D)
  - [x] Add deprecation warning to `ddgclib/_plotting.py`
  - [x] 53 visualization tests passing (1 skipped: polyscope)
- [x] Phase 4: Documentation updates (CLAUDE.md, ARCHITECTURE.md, DEVELOPMENT.md, FEATURES.md)

## Feature: Pylint Standards Enforcement
Status: In Progress
Scope: `ddgclib/`, `cases_mean_flow/`, `cases_dynamic/` (all `.py` files, no notebooks)
Baseline: 15,948 messages, score ~1.23/10 → After .pylintrc: 4,310 messages, score ~7.8/10
Target: score >= 6-7/10 with all critical issues resolved

- [x] Create `.pylintrc` with domain-appropriate configuration
  - Domain-standard naming (variable/argument/const/function = any style)
  - good-names whitelist for single-letter math variables
  - Disabled noisy checks (complexity, duplicate-code, fixme, etc.)
  - `max-line-length=120`, `generated-members` for hyperct/numpy
- [ ] Replace wildcard imports with explicit imports in ddgclib core (204 wildcard-import, 209 unused-wildcard-import)
  - `ddgclib/_sphere.py`, `_flow.py`, `_ellipsoid.py`, `_capillary_rise.py`
  - `_capillary_rise_flow.py`, `_cube_droplet.py`, `_catenoid.py`, `_hyperboloid.py`
  - `_curvatures.py`, `_plotting.py`, `_case1.py`, `_case2.py`, `_sessile.py`
  - cases_mean_flow/ and cases_dynamic/ wildcard imports
- [ ] Remove unused imports (238 unused-import, 62 reimported)
- [ ] Fix dangerous patterns
  - W0102 dangerous-default-value (mutable default arguments like `bV=set()`)
  - W1401 anomalous-backslash-in-string (60 occurrences — add `r` prefix)
  - W0640 cell-var-from-loop (22 occurrences)
  - W0104 pointless-statement (57 occurrences)
- [ ] Fix E-level errors (377 total)
  - E0602 undefined-variable (150 — many from wildcard imports)
  - E0102 function-redefined (60 — dead code/copy-paste duplicates)
  - E1101 no-member (72 — mostly false positives from hyperct)
  - E0401 import-error (22 — optional dependencies)
  - E1123 unexpected-keyword-arg (20 — review for actual bugs)
  - E0601 used-before-assignment (17)
  - E0611 no-name-in-module (18)
- [ ] Add minimal module docstrings (69 modules)
- [ ] Add minimal function/method docstrings (409 functions)
- [ ] Add minimal class docstrings (55 classes)
- [ ] Formatting fixes
  - W0311 bad-indentation (1,146 — normalize to 4-space indent)
  - C0303 trailing-whitespace (120)
  - C0304 missing-final-newline (23)
  - C0411 wrong-import-order (107)
  - C0413 wrong-import-position (150)
- [ ] Remove unused variables and dead code
  - W0612 unused-variable (500)
  - W0107 unnecessary-pass (26)
  - R1711 useless-return (22)

## Feature: Integrated Analytical Validation Framework
Status: Complete
- [x] `hyperct/ddg/_dual_cell.py`: Dual cell geometry extraction (1D, 2D, 3D)
  - Two 2D polygon formulations: `barycentric_dual_p_ij` (with edge midpoints) and `barycentric` (duals only)
  - `dual_cell_area_2d` (exact shoelace formula, replaces approximate `d_area`)
- [x] `ddgclib/analytical/`: Divergence theorem integration + sympy exact path
  - `integrated_gradient_1d/2d/3d` (Gauss-Legendre quadrature)
  - `integrated_gradient_sympy_1d/2d/3d` (exact symbolic, optional dependency)
  - `integrated_gradient_2d_vector` / `3d_vector` (tensor gradient)
  - `integrated_pressure_error`, `integrated_l2_norm`, `compare_stress_force`
  - `volume_averaged_scalar` — computes `(1/Vol) ∫ f dV` for FVM field assignment
- [x] `scalar_gradient_integrated` added to `ddgclib/operators/stress.py`
- [x] Benchmark framework: `benchmarks/_integrated_benchmark_classes.py`
  - `IntegratedGradientBenchmark` base class with `polygon_method` parameter
  - 16 gradient cases + 4 stress cases + curvature case
  - CLI runner: `benchmarks/run_integrated_benchmarks.py`
- [x] Jupyter notebooks: `benchmarks/notebooks/01-03` (gradient), `05` (stress/curvature)
- [x] 47 pytest tests in `test_integrated_validation.py` (all passing)
- [x] `dual_volume()` updated to use exact `dual_cell_area_2d` (was approximate `d_area`)
- [x] IC classes (`HydrostaticPressure`, `LinearPressureGradient`) assign volume-averaged pressure
- [x] All dynamic cases (Hydrostatic 1D/2D/3D, Poiseuille 2D/3D, equilibrium) use integrated comparisons exclusively
- [x] Point-wise pressure comparisons removed from all production code
- [x] See `benchmarks/ADDING_FORMULATIONS.md` for extending to new dual/operator formulations

## Feature: Cauchy Stress Tensor Operators
Status: Complete
- [x] `dual_area_vector(v_i, v_j, HC, dim)` — oriented A_ij (2D)
- [x] `dual_area_vector` — 3D support
- [x] `dual_volume(v, HC, dim)` — dual cell volume (2D via exact `dual_cell_area_2d`)
- [x] `dual_volume` — 3D support via v_star
- [x] `velocity_difference_tensor(v, HC, dim)` — integrated Du_i (no /Vol)
- [x] `velocity_difference_tensor_pointwise(v, HC, dim)` — Du_i / Vol_i for analytics
- [x] `strain_rate(du)` — symmetric part
- [x] `cauchy_stress(p, du, mu, dim)` — pointwise Newtonian fluid
- [x] `integrated_cauchy_stress(p, Du, mu, Vol_i, dim)` — volume-integrated stress
- [x] `stress_force(v, dim, mu, HC)` — face-centered flux formulation (Stokes' theorem)
  - Pressure: half-difference `F_p = -0.5*(p_j-p_i)*A_ij`
  - Viscous: diffusion form `F_v = (mu/|d|)*du*(d_hat.A)` (no transpose term)
- [x] `stress_acceleration(v, dim, mu, HC)` — a_i = F_stress / m
- [x] `cache_dual_volumes(HC, dim)` — cache v.dual_vol on all vertices
- [x] Replace old gradient.py functions with stress-tensor wrappers
- [x] Unit tests — 2D (closure, antisymmetry, magnitude, equilibrium, Poiseuille)
- [x] Unit tests — 3D (closure, antisymmetry, volume conservation, equilibrium)
- [x] `dudt_i` alias for `stress_acceleration` with integrator tests
- [x] Hagen-Poiseuille 2D validation (equilibrium residual, developing flow, profile shape)
- [x] Hydrostatic 2D validation (pressure force direction, zero viscous stress, perturbation damping)
- [x] Hagen-Poiseuille 3D validation (pressure force direction, developing flow, profile shape, bounded residual)
- [x] Hydrostatic 3D validation (pressure force direction, zero viscous stress, acceleration nonzero, viscous damping)
- [x] Hagen-Poiseuille equilibrium case (`cases_dynamic/Hagen_Poiseuile_equilibrium/`)
  - Analytical IC, convergence table, 6-panel field visualization, 18 pytest tests
- [x] Fix dimensional consistency in stress tensor formulation
  - Refactored to face-centered integrated formulation: forces computed via Stokes' theorem
    surface integrals on dual flux planes. No vertex-centered gradients, no /Vol division.
  - Old pointwise code archived in `stress_pointwise.py` and `Fundamentals_pointwise.md`
  - New formulation documented in `Fundamentals.md`
  - 62 tests passing (50 original + 12 new for caching, face-centered fluxes, integrated stress)
- [x] Fix spurious discrete compressibility in viscous flux
  - Dropped symmetric transpose term `d_hat*(du.A)` which discretizes `mu*grad(div u)` —
    the rank-1 face gradient has nonzero trace on diagonal edges even for div-free fields
  - Diffusion form `F_v = (mu/|d|)*du*(d_hat.A)` gives exact zero for Poiseuille at machine precision
  - All 470 unit tests + 18 equilibrium tests passing
- [ ] Update Template case study
- [ ] Dam break test case

## Feature: Lagrangian Mesh Retopologize
Status: Complete
Every integrator timestep must retopologize the mesh before evaluating `dudt_fn`,
to handle vertex injection/deletion by BCs and position changes from advection.

- [x] Add `_retopologize(HC, bV, dim)` hidden method to `_integrators_dynamic.py`
  - Retriangulate via `scipy.spatial.Delaunay` (disconnect all edges, reconnect from simplices)
  - 1D special case: sort vertices and connect as chain
  - Recompute boundary via `HC.boundary()` → update `bV` in-place
  - Tag `v.boundary` on all vertices
  - Recompute duals via `compute_vd(HC, method="barycentric")`
- [x] Call `_retopologize` at the top of every step in all integrators:
  `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`
- [x] Revert `dudt_safe` hack in `Hagen_Poiseuile_2D.py` — use `partial(dudt_i, ...)` directly
- [x] Remove dual recomputation from `combined_callback` (now in `_retopologize`)
- [x] Run all tests (279 passed, 3 skipped)

## Feature: Fix PeriodicInletBC Ghost Vertex Accumulation
Status: Complete
Bug: Ghost vertices that cross the inlet are NOT removed from the ghost mesh.
They advance every step and keep being re-injected, creating chains of duplicate
vertices along the walls (especially at corners y=0, y=H).

- [x] In `PeriodicInletBC.apply()`: remove injected ghost vertices from `self.ghost`
  after injection (each vertex injected exactly once)
- [x] Change periodic reset: re-clone ghost from `self.unit_mesh` when ghost is
  depleted (empty), rather than when all vertices have crossed
- [x] Update `bc_demo` case to verify fix
- [x] Run all tests (279 passed, 3 skipped)

## Feature: HDF5 Data Handling + hyperct I/O
Status: Not Started
Plan: `.claude/plans/jiggly-tumbling-lampson.md`

### Part A: ddgclib — HDF5 Data Handling
- [ ] Create `ddgclib/data/_hdf5.py` (HDF5Writer, HDF5Reader, self-contained snapshots)
- [ ] Update `ddgclib/data/_io.py` (HDF5 path, keep legacy JSON)
- [ ] Refactor StateHistory (list-of-arrays, hdf5_path streaming, _snapshots compat)
- [ ] Update `__init__.py`, `_integrators_dynamic.py`, `setup.py`
- [ ] Add HDF5 round-trip + structured StateHistory tests

### Part B: hyperct — Parallel Geometry I/O Module
- [ ] Create `hyperct/io.py` (save_complex, load_complex, JSON + HDF5)
- [ ] Add `Complex.to_arrays()` / `Complex.from_arrays()`
- [ ] Deprecate existing `Complex.save_complex()` / `load_complex()` stubs
- [ ] Create `hyperct/tests/test_io.py`

## Feature: Outlet Buffer Ghost Zone (OutletBufferedDeleteBC)
Status: Complete

At the outlet boundary, vertices on the topological boundary have truncated dual
cells — `dual_area_vector()` returns zeros for boundary edges (stress.py:98-100).
This creates imbalanced stress that pushes vertices backward (backflow).

Solution: buffer ghost zone `[outlet_pos, outlet_pos + buffer_width]` keeps buffer
vertices beyond the physical outlet so domain vertices are never on the boundary.
Buffer vertices have frozen velocity and corrected position (no stress drift).

- [x] `OutletBufferedDeleteBC` class in `_boundary_conditions.py`
  - `id(v)`-keyed internal buffer (vertex hash changes on `mesh.V.move()`)
  - Entry velocity frozen on first detection after crossing `outlet_pos`
  - Position correction each step: `correct_pos += frozen_u * dt`
  - Deletion at `outlet_pos + buffer_width`
  - `buffer_vertices` property for inspection
- [x] 6 unit tests (entry detection, velocity freeze, position correction,
  deletion, bV cleanup, domain vertices untouched)
- [x] Comparison scripts: `test_outlet_old_bc.py`, `test_outlet_new_bc.py`
- [x] HP2D case updated to use `OutletBufferedDeleteBC`

## Feature: Fixed Inlet/Outlet Boundary Conditions
Status: Not Started

### Problem Statement
In Lagrangian simulations, mesh vertices move with the fluid. At domain boundaries
(inlet/outlet), vertices on the topological boundary have truncated dual cells
because `compute_vd` builds half-cells for boundary vertices. `dual_area_vector()`
returns `np.zeros(2)` for boundary edges with only one shared dual vertex
(stress.py:98-100), causing imbalanced stress forces.

The `OutletBufferedDeleteBC` addresses this with a buffer zone, but a more
fundamental solution is to have **fixed (non-moving) boundary vertices** with
prescribed velocity/pressure values. These vertices stay at domain edges during
integration but have non-zero field values that participate in stress computation
for interior neighbours, ensuring the domain always has a well-defined boundary.

### Proposed Architecture Changes

**New vertex classification (3-tier instead of 2-tier):**

Current:
- `v in bV` → frozen: zero velocity, no position update, no acceleration computed
- `v not in bV` → interior: full integration (acceleration, velocity, position)

Proposed:
- `v in bV` → frozen walls: zero velocity, no position update
- `v in prescribed_V` → fixed boundary: prescribed velocity/pressure, position
  frozen, field values participate in stress computation for interior neighbours
- `v not in bV and v not in prescribed_V` → interior: full integration

**Implementation approach: new set `prescribed_V`**

- Add `prescribed_V: set` parameter to integrator functions alongside `bV`
- `_interior_verts(HC, bV, prescribed_V)` excludes both sets from integration
- Stress computation for interior vertex `v_i` iterates over `v_i.nn` which
  includes prescribed neighbours — their `v.u` and `v.p` are set by the BC, so
  `stress_force` automatically picks up the correct values
- Position update skips prescribed vertices (they stay fixed)

Alternative approaches (rejected):
- Vertex attribute `v.prescribed = True`: requires checking in every integrator
  loop, pollutes vertex namespace
- Modify bV semantics: breaking change to all existing BCs and integrators

### Implementation Subtasks

- [ ] Add `prescribed_V: set` parameter to all integrators
  - `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`
  - Default `None` (backward compatible, treated as empty set)
  - `_interior_verts(HC, bV, prescribed_V)` excludes both sets
  - Position update loop skips `prescribed_V` vertices

- [ ] Modify `_retopologize` to handle `prescribed_V`
  - Prescribed vertices participate in Delaunay triangulation
  - Tagged `v.boundary = True` if on topological boundary
  - NOT added to `bV` by `boundary_filter`
  - Need separate handling to keep them out of `bV`

- [ ] Implement `FixedVelocityInletBC`
  ```python
  class FixedVelocityInletBC(BoundaryCondition):
      def __init__(self, criterion_fn, velocity_fn, pressure_fn=None, dim=2):
          """
          criterion_fn : callable(v) -> bool  (identifies inlet vertices)
          velocity_fn  : callable(v) -> ndarray  OR  constant ndarray
          pressure_fn  : callable(v) -> float  OR  constant float
          """
          ...
          self.prescribed_V = set()

      def apply(self, mesh, dt, target_vertices=None):
          self.prescribed_V.clear()
          for v in mesh.V:
              if self.criterion_fn(v):
                  self.prescribed_V.add(v)
                  v.u = (self.velocity_fn(v) if callable(self.velocity_fn)
                         else self.velocity_fn.copy())
                  if self.pressure_fn is not None:
                      v.p = (self.pressure_fn(v) if callable(self.pressure_fn)
                             else self.pressure_fn)
          return len(self.prescribed_V)
  ```

- [ ] Implement `FixedPressureOutletBC`
  ```python
  class FixedPressureOutletBC(BoundaryCondition):
      def __init__(self, criterion_fn, pressure, velocity_gradient=0.0,
                   dim=2, axis=0):
          """
          criterion_fn     : callable(v) -> bool  (identifies outlet vertices)
          pressure         : float  (fixed outlet pressure, e.g. 0.0)
          velocity_gradient: float  (du/dn at outlet; 0.0 = zero-gradient/Neumann)
          """
          ...
          self.prescribed_V = set()

      def apply(self, mesh, dt, target_vertices=None):
          self.prescribed_V.clear()
          for v in mesh.V:
              if self.criterion_fn(v):
                  self.prescribed_V.add(v)
                  v.p = self.pressure
                  if self.velocity_gradient == 0.0:
                      # Zero-gradient: copy velocity from nearest interior neighbour
                      interior_nbs = [nb for nb in v.nn
                                      if not self.criterion_fn(nb)]
                      if interior_nbs:
                          nb = min(interior_nbs,
                                   key=lambda nb: np.linalg.norm(v.x_a - nb.x_a))
                          v.u = nb.u.copy()
          return len(self.prescribed_V)
  ```

- [ ] Update `BoundaryConditionSet` to aggregate `prescribed_V`
  - After `apply_all()`, collect `bc.prescribed_V` from each BC that has it
  - Expose as `bc_set.prescribed_V` property for the integrator

- [ ] Update integrator functions to use `prescribed_V`
  - Skip prescribed vertices in position update
  - Ensure prescribed vertices are included in Delaunay but excluded from
    acceleration computation

- [ ] Handle `_retopologize` interaction
  - Prescribed vertices at inlet/outlet are topological boundary
  - `boundary_filter` must NOT add them to `bV`
  - Option: extend `boundary_filter` to exclude prescribed vertices
  - Option: add `prescribed_V` parameter to `_retopologize`

- [ ] Unit tests
  - Prescribed velocity at inlet with analytical Poiseuille profile
  - Prescribed pressure at outlet with zero gradient
  - Interior neighbours of prescribed vertices get correct stress
  - Prescribed vertices do not move during integration
  - Backward compatibility (no `prescribed_V` parameter)

- [ ] Integration test: HP2D with `FixedVelocityInletBC` + `FixedPressureOutletBC`
  - Compare convergence to analytical profile
  - Verify no backflow at outlet
  - Verify velocity profile develops correctly from inlet

### Testing Strategy
1. Unit tests for each new BC class (apply, field setting, criterion detection)
2. Integration tests with `symplectic_euler` (prescribed vertices stay fixed)
3. Stress force tests (interior vertices near prescribed boundary get balanced forces)
4. Regression tests (existing BCs and integrators unchanged when `prescribed_V=None`)
5. HP2D comparison (new BCs vs `OutletBufferedDeleteBC` vs `OutletDeleteBC`)

## Feature: DEM Submodule (`ddgclib/dem/`)
Status: Complete
Self-contained Discrete Element Method for spherical particle simulation,
contact mechanics, sintered bonds, capillary liquid bridges, fluid-particle
coupling, and particle cloud I/O.

### Phase 1: Core DEM
- [x] `_particle.py` — `Particle` dataclass + `ParticleSystem` container
  - `Particle.sphere()` classmethod (auto mass/inertia)
  - Position (`x_a`), velocity (`u`), angular velocity (`omega`), force/torque accumulators
  - Wetting attributes (`wetted`, `wetting_angle`, `liquid_volume`)
  - `ParticleSystem`: add/remove, batch accessors (`positions()`, `velocities()`, `radii()`)
  - Gravity, cluster tracking, kinetic energy, momentum, center of mass
- [x] `_contact.py` — `ContactDetector` + `Contact` dataclass
  - Broad phase: cell-linked list spatial hashing (cell size = `2 * R_max`)
  - Narrow phase: sphere-sphere overlap test with contact geometry
  - Normal/tangential velocity decomposition, contact point
- [x] `_force_models.py` — Pluggable contact force models
  - `ContactForceModel` ABC + `ContactForceResult` dataclass
  - `HertzContact` (non-linear elastic `F_n ∝ δ^{3/2}` + viscous damping)
  - `LinearSpringDashpot` (Cundall & Strack)
  - `contact_force_registry` via `MethodRegistry`
- [x] `_integrators.py` — DEM time integration
  - `dem_velocity_verlet` (symplectic, 2nd-order)
  - `dem_symplectic_euler` (matches fluid integrator)
  - `dem_step` entry point with sub-stepping, callback
  - `_accumulate_all_forces` (gravity + contacts + bonds + bridges + external)
- [x] 71 Phase 1 tests passing (particle, contact, forces, integrators)

### Phase 2: Bridges and Bonds
- [x] `_bonds.py` — `SinterBond` + `BondManager`
  - Spring-dashpot along bond axis, fracture criterion
  - `grow_neck()` Frenkel/Kuczynski sintering model
  - Cluster ID assignment for bonded particles
- [x] `_liquid_bridge.py` — `LiquidBridge` + `LiquidBridgeManager`
  - Toroidal approximation capillary force with linear decay
  - Lian et al. 1993 rupture criterion
  - Formation/rupture tracking via `frozenset` pair keys
- [x] 31 Phase 2 tests passing (bonds, liquid bridges)
- [x] Integration test: two wetted particles approach, bridge forms, agglomerate

### Phase 3: Coupling, I/O, Visualization
- [x] `_coupling.py` — `FluidParticleCoupler`
  - IDW interpolation of fluid velocity/pressure at particle centres
  - Stokes and Schiller-Naumann drag correlations
  - Two-way coupling: `fluid_to_particle()` + `particle_to_fluid()` (Newton III)
  - `get_external_forces_fn()` for `dem_step` integration
- [x] `_io.py` — Particle cloud I/O
  - `save_particles()` / `load_particles()` JSON format (`ddgclib_dem_state_v1`)
  - `import_particle_cloud()` from NumPy arrays (mono/polydisperse, velocities, wetting)
- [x] `_visualization.py` — Particle rendering
  - `plot_particles()` matplotlib (2D circles / 3D scatter)
  - `plot_bridges()`, `plot_bonds()` for bridge/bond visualization
  - `plot_particles_polyscope()` optional polyscope point cloud
- [x] 22 Phase 3 tests passing (coupling, I/O, drag, interpolation)

### Test Summary
- 124 DEM tests total (+ 1 slow integration test)
- 0 regressions on existing 283 tests

## Feature: Parametric Surface Geometry Module
Status: Complete
Plan: `.claude/plans/vivid-chasing-sunrise.md`

Abstract the duplicated parametric surface pipeline (shared by `_sphere.py`,
`_catenoid.py`, `_hyperboloid.py`) into a reusable geometry module, then use
surface-conforming meshes for the CFD-DEM fluid film.

### Phase 1: Core Geometry Module (`ddgclib/geometry/_parametric_surfaces.py`)
- [x] `parametric_surface(f, domain, refinement, cdist, boundary_fn)` — core abstraction
  - Complex(2, domain) → triangulate → refine → project via f(u,v)→(x,y,z) → copy connectivity → merge_all → rebuild bV
- [x] `sphere(R, refinement, theta_range, phi_range)` — convenience wrapper
- [x] `catenoid(a, v_range, refinement)` — convenience wrapper
- [x] `cylinder(R, h_range, refinement)` — convenience wrapper
- [x] `hyperboloid(a, b, c, v_range, refinement)` — convenience wrapper
- [x] `torus(R, r, refinement)` — convenience wrapper (closed surface, no boundary)
- [x] `plane(x_range, y_range, refinement)` — flat reference surface
- [x] `translate_surface(HC, offset)`, `scale_surface(HC, factor, center)` — in-place transforms
- [x] `rotate_surface(HC, R, center)`, `rotation_matrix_align(from, to)` — rotation helpers
- [x] Update `geometry/__init__.py` exports
- [x] Refactor `_sphere.py`, `_catenoid.py`, `_hyperboloid.py` to delegate to `parametric_surface()`
- Note: `merge_surfaces` not needed — hyperct handles changing topologies natively

### Phase 2: Tutorial and Validation
- [x] `tutorials/parametric_surfaces_tutorial.py` — generate + visualize all surface types
  - Uses `plot_complex` from hyperct as base, overlays boundary vertices
- [x] `tutorials/visualize_parametric_surfaces.py` — polyscope 3D viewer with triangle extraction
- [x] Geometry validation (vertex positions at machine precision for all surfaces)
- [x] Convergence table (refinement 1→3, vertex counts + geometric error)
- [x] `ddgclib/tests/test_parametric_surfaces.py` — 37 unit tests passing (core + rotate)

### Phase 3: Fluid Film Projection for CFD-DEM
- [x] `cases_dynamic/liquid_bridge_cfd_dem/src/_fluid_film.py` — film mesh construction
  - `create_particle_film(particle, film_thickness, refinement, facing)` — spherical shell (hemisphere facing partner)
  - `create_bridge_film(p1, p2, bridge, refinement)` — catenoid aligned to particle axis
  - `FluidFilmManager` — manages particle + bridge films, syncs with DEM state
  - Capillary force from catenoid neck tension: `F = 2*pi*a*gamma`
  - Analytical curvature stored on vertices (H = 1/R for sphere, H = 0 for catenoid)
- [x] `_setup.py` returns `film_mgr` alongside existing outputs
- [x] Case loop syncs film meshes with particle positions / bridge events
- [x] History records film snapshots (vertices, edges, curvature per frame)
- [x] Visualizer renders film surfaces as polyscope triangle meshes (semi-transparent)
- [x] 7 fluid film tests + 6 rotation tests passing
- [x] 450 total tests passing (0 regressions)

## Feature: Vectorized Stress Pipeline with Per-Edge Dual Area Caching
Status: Complete
Plan: `.claude/plans/swirling-wondering-marshmallow.md`

Cache oriented dual area vectors from `batch_e_star` on HC during retopologization,
so `stress_force` uses O(1) dict lookups instead of recomputing `e_star()` per edge
per step. Eliminates ~100M sequential Python calls across 5000 steps. Expected
10-50x speedup on stress pipeline.

### hyperct changes (`hyperct/ddg/_operators.py`)
- [x] Add `orient: bool = False` parameter to `batch_e_star`
- [x] Collect primal vertex positions per edge in Phase 1 (`compute_volumes or orient`)
- [x] Orient + sum triangle areas in Phase 3 scatter (dot-product flip, matches `dual_area_vector`)

### ddgclib changes
- [x] `stress_force` — cache lookup via `HC._edge_area_cache` with fallback (`operators/stress.py`)
- [x] `velocity_difference_tensor` — same cache lookup pattern (`operators/stress.py`)
- [x] `_retopologize` — add `backend=None` parameter (`dynamic_integrators/_integrators_dynamic.py`)
- [x] `_retopologize` — replace `cache_dual_volumes` with `batch_e_star(orient=True, compute_volumes=True)`, set `HC._edge_area_cache`
- [x] `_do_retopologize` — thread `backend` parameter
- [x] Thread `backend=None` through all integrator signatures (`euler`, `symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`)
- [x] `retopologize_cylinder` — capture `edge_areas` and set `HC._edge_area_cache` (`cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py`)

### Tests
- [x] Cache consistency test: `dual_area_vector` vs `batch_e_star(orient=True)` for all edges
- [x] `stress_force` equivalence: with cache vs without cache
- [x] Fallback test: `HC._edge_area_cache = None` still works
- [x] Full test suite passes (532 passed, 3 skipped, 0 failures)

### Verification
- [x] `pytest ddgclib/tests/test_stress.py -v -m "not slow"` passes (65 passed)
- [x] `pytest ddgclib/tests/ -v -m "not slow"` passes (532 passed)
- [ ] HP3D case with `--backend numpy` gives identical physics output
- [ ] HP3D case with `--backend gpu` shows speedup

## Feature: Migrate Geometry to hyperct.ddg
Status: Not Started
- [ ] Move `dual_area_vector` to hyperct.ddg._operators
- [ ] Move `dual_volume` to hyperct.ddg._operators
- [ ] Update ddgclib imports to use hyperct.ddg
- [ ] Add hyperct unit tests for moved functions

## Feature: Periodic Boundary Conditions
Status: Complete (Phase 1-3)
- [x] Core ghost-cell utilities (`ddgclib/geometry/periodic.py`)
- [x] `retopologize_periodic` with opposite-face merge + ghost Delaunay
- [x] Minimum-image `dual_area_vector` in `stress.py` for periodic meshes
- [x] Wire `periodic_axes`/`domain_bounds` through all 5 integrators
- [x] Domain builders: `periodic_rectangle`, `periodic_box`
- [x] Test suite: 26 tests all passing (558 total suite passing)
- [ ] Phase 4: Update `Hydrostatic_2D_periodic.py` validation case
- [ ] Future: CGAL backend for true periodic Delaunay (near-linear scaling)

## Feature: Integrated Analytical Validation Framework
Status: Complete (Steps 1-8)
Plan: `.claude/plans/prancy-whistling-brooks.md`

Compares DDG integrated operators against analytically integrated solutions
over the exact same dual cell domains using the divergence theorem.

- [x] Step 1: Dual cell geometry extraction (`hyperct/ddg/_dual_cell.py`)
- [x] Step 2: Analytical integration module (`ddgclib/analytical/`)
- [x] Step 3: `scalar_gradient_integrated()` in `operators/stress.py`
- [x] Step 4: Benchmark framework (`benchmarks/_integrated_benchmark_classes.py`)
- [x] Step 5: Pytest suite (`tests/test_integrated_validation.py`, 47 tests)
- [x] Step 6: Jupyter notebooks (`benchmarks/notebooks/01-03`)
- [x] Step 7: Stress/curvature operator benchmarks
  - [x] `StressGradientBenchmark` base class
  - [x] `PressureGradientBenchmark` — hydrostatic P vs stress_force(mu=0)
  - [x] `ViscousFluxBenchmark` — linear (zero) and quadratic viscous force
  - [x] `PoiseuilleBenchmark` — equilibrium force cancellation
  - [x] `CurvatureBenchmark` — sphere/cylinder curvature validation
  - [x] 6 new stress tests in `test_integrated_validation.py`
  - [x] `benchmarks/notebooks/05_stress_curvature_validation.ipynb`
- [x] Step 8: Updated dynamic cases with integrated comparison
  - [x] `Hydrostatic_1D.py` — Section 4: integrated P error + L2 norm
  - [x] `Hydrostatic_2D.py` — Section 4: integrated P error + force balance
  - [x] `Hagen_Poiseuile_equilibrium_2D.py` — Du error + P L2 in convergence table
  - [x] `Hagen_Poiseuile_2D.py` — integrated Du comparison at final mesh
  - [x] `Hagen_Poiseuile_3D.py` — force balance diagnostic
  - [x] `ddgclib/analytical/_integrated_comparison.py` — reusable utilities

## Feature: Multiphase Fluid Simulation
Status: In Progress
Extends the single-phase Lagrangian FVM to handle multiple fluid phases with
interface tracking, per-phase material properties, and surface tension.

### Phase 1: Multiphase Data Model
- [x] `ddgclib/multiphase.py` — `PhaseProperties` dataclass, `MultiphaseSystem` class
  - Phase assignment, interface identification, mass fraction computation
  - Harmonic mean viscosity for cross-phase edges
  - Surface tension lookup per phase pair
- [x] Extend `ddgclib/initial_conditions.py` — `PhaseAssignment`, `MultiphaseMass`, `MultiphasePressure`
- [x] 24 unit tests passing (`test_multiphase.py`)

### Phase 2: EOS Infrastructure
- [x] `ddgclib/eos/_ideal_gas.py` — Ideal gas EOS for gas phases
- [x] `ddgclib/eos/_multiphase_eos.py` — `MultiphaseEOS` dispatching to per-phase EOS
  - Implements callable protocol for `stress_force(..., pressure_model=meos)`
  - Mass-fraction-weighted pressure at interface vertices
- [x] Updated `ddgclib/eos/__init__.py` exports

### Phase 3: Phase-Aware Stress & Surface Tension
- [x] `ddgclib/operators/multiphase_stress.py` — `multiphase_stress_force`, `multiphase_dudt_i`
  - Per-phase viscosity via harmonic mean at interface
  - Surface tension force from Heron curvature at interface vertices
- [x] `ddgclib/_curvatures_heron.py` — `hndA_i_interface()` for interface sub-mesh curvature
- [x] Updated `ddgclib/operators/__init__.py` exports

### Phase 4: Multiphase Mesh Construction
- [x] `ddgclib/geometry/domains/_multiphase_droplet.py` — `droplet_in_box_2d`, `droplet_in_box_3d`
  - Combines disk/ball droplet with rectangle/box outer domain
  - Automatic phase assignment and interface identification
- [x] `_retopologize_multiphase` in `dynamic_integrators/_integrators_dynamic.py`
- [x] Updated `ddgclib/geometry/domains/__init__.py` exports

### Phase 5: Oscillating Droplet Case Study
- [x] `cases_dynamic/oscillating_droplet/src/_params.py` — Overdamped (oil) + underdamped (water) parameter sets
- [x] `cases_dynamic/oscillating_droplet/src/_analytical.py` — Lamb/Rayleigh analytical solution
  - Rayleigh frequency (2D/3D), Lamb damping, damped frequency, radius perturbation
- [x] `cases_dynamic/oscillating_droplet/src/_setup.py` — `setup_oscillating_droplet(dim, ...)`
- [x] `cases_dynamic/oscillating_droplet/src/_plot_helpers.py` — Diagnostics and visualization
- [x] `cases_dynamic/oscillating_droplet/src/_boundary_conditions.py` — No-slip and open BCs
- [x] `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` — Main 2D simulation
- [x] `cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py` — Main 3D simulation

### Phase 6: Mesh Independence Study & Tests
- [x] `cases_dynamic/oscillating_droplet/mesh_convergence_2D.py` — Convergence study at multiple refinements
- [x] `ddgclib/tests/test_case_oscillating_droplet.py` — Integration tests (8 pass, 1 skip, 1 slow)
- [ ] Full dynamic validation: overdamped case R_max(t) matches analytical envelope
- [ ] Underdamped case: oscillation frequency matches Rayleigh prediction
- [ ] 3D convergence study

### Test Summary
- 32 new tests (24 multiphase unit + 8 integration), 0 regressions on existing 604 tests

## Feature: Interface-Preserving Adaptive Refinement / Remeshing
Status: Phase 1–2 Complete (2D only)

### Motivation

The current Lagrangian retopologization (`_retopologize` in
`dynamic_integrators/_integrators_dynamic.py`) performs a **global Delaunay
retriangulation** every time step.  This destroys the sharp phase interface:
Delaunay reconnection creates edges between droplet-interior and outer-phase
vertices that were not previously adjacent, turning interior vertices into
spurious interface vertices.  These receive surface tension forces with
incorrect curvature estimates, causing an instability that grows from the
corners of the initial shape.

Diagnostic evidence (from `cases_dynamic/Cube2droplet/diagnostic_no_retopo.py`):
- **With full retopo**: circularity 0.47 → 0.72 → collapse (interface band
  widens, dual volume ratio degrades 2:1 → 11:1)
- **Without retopo** (dual-only recompute on fixed connectivity): perfectly
  stable for 1.0 s, 36/36 interface vertices preserved, smooth pressure
  dynamics — but circularity slowly degrades (0.57 → 0.47) because the fixed
  triangulation becomes degenerate as vertices move

The solution is to replace the global Delaunay with **local mesh operations**
that maintain mesh quality while respecting the sharp interface as a
constrained boundary.

### Architecture: `hyperct/remesh/` module

A new module in hyperct (not ddgclib) providing mesh operations that work
on `Complex` objects.  ddgclib's `_retopologize` will call these instead of
`scipy.spatial.Delaunay`.

```
hyperct/remesh/
    __init__.py          Public API exports
    _operations_2d.py    2D local mesh operations
    _operations_3d.py    3D local mesh operations
    _quality.py          Element quality metrics
    _interface.py        Interface-aware constraints
    _driver.py           Adaptive remeshing driver
```

### 2D Local Mesh Operations (`_operations_2d.py`)

All operations preserve the simplicial complex structure and update
vertex connectivity, dual vertices, and edge caches incrementally.

| Operation | When | Effect |
|-----------|------|--------|
| **Edge split** | Edge too long (L > L_max) | Insert midpoint, split 2 adjacent triangles into 4 |
| **Edge collapse** | Edge too short (L < L_min) | Merge two vertices, remove 2 triangles |
| **Edge flip** | Non-Delaunay diagonal in quadrilateral | Swap diagonal of 2 adjacent triangles |

**Interface constraint**: edges that cross the phase boundary (one endpoint
phase=0, one phase=1) are **never flipped**.  Splits on interface edges
create new interface vertices.  Collapses near the interface are restricted
to same-phase pairs.

Length thresholds from local mesh spacing:
- `L_max = alpha_max * h_local` (default alpha_max = 1.4)
- `L_min = alpha_min * h_local` (default alpha_min = 0.5)
- `h_local` = median edge length of 1-ring neighbourhood

Quality metric: minimum angle of triangle (Delaunay-optimal = maximise
minimum angle).  Target: all triangles have min angle > 20°.

Reference: Persson & Strang (2004), "A Simple Mesh Generator in MATLAB",
SIAM Review 46(2), 329-345.  doi:10.1137/S0036144503429121

### 3D Local Mesh Operations (`_operations_3d.py`)

Extends the 2D operations to tetrahedral meshes.  The operation set follows
Freitag & Ollivier-Gooch (1997):

| Operation | When | Effect |
|-----------|------|--------|
| **Edge split** | Edge too long | Insert midpoint, subdivide adjacent tetrahedra |
| **Edge collapse** | Edge too short | Merge vertices, remove adjacent tetrahedra |
| **Face swap** (2-3) | Two tetrahedra sharing a face → three tetrahedra sharing an edge |
| **Edge swap** (3-2) | Three tetrahedra sharing an edge → two sharing a face |

**Interface constraint**: operations that would create a tetrahedron
straddling the phase boundary are rejected.  Interface triangles (faces
between phase=0 and phase=1 tetrahedra) are constrained.

Quality metric: aspect ratio (circumradius / inradius) or minimum
dihedral angle.  Target: aspect ratio < 10, min dihedral > 10°.

Reference: Freitag & Ollivier-Gooch (1997), "Tetrahedral mesh improvement
using swapping and smoothing", Int. J. Numer. Methods Eng. 40(21),
3979-4002.  doi:10.1002/(SICI)1097-0207(19971115)40:21<3979::AID-NME251>3.0.CO;2-9

### Element Quality Metrics (`_quality.py`)

```python
def triangle_min_angle(v0, v1, v2) -> float: ...
def triangle_aspect_ratio(v0, v1, v2) -> float: ...
def tet_min_dihedral(v0, v1, v2, v3) -> float: ...
def tet_aspect_ratio(v0, v1, v2, v3) -> float: ...
def mesh_quality_histogram(HC, dim) -> dict: ...
```

### Interface-Aware Constraints (`_interface.py`)

```python
def is_interface_edge(v_i, v_j) -> bool:
    """True if the edge crosses the phase boundary."""

def can_flip(v_i, v_j, HC) -> bool:
    """True if the edge can be flipped without violating interface."""

def can_collapse(v_i, v_j, HC) -> bool:
    """True if collapsing the edge preserves interface topology."""

def split_interface_edge(v_i, v_j, HC, mps) -> vertex:
    """Split an interface edge: new vertex inherits interface status."""
```

### Adaptive Remeshing Driver (`_driver.py`)

Called by `_retopologize` instead of `Delaunay`:

```python
def adaptive_remesh(HC, dim, mps=None, L_min=None, L_max=None,
                    quality_target=20.0, max_iterations=5):
    """Perform adaptive remeshing with interface constraints.

    1. Edge splits: subdivide edges longer than L_max
    2. Edge collapses: merge edges shorter than L_min
    3. Edge flips (2D) / face+edge swaps (3D): optimise quality
    4. Laplacian smoothing: relax interior vertex positions
       (interface vertices smoothed along the interface only)
    5. Recompute duals: compute_vd + cache_dual_volumes
    """
```

### Integration with ddgclib

Replace `_retopologize` internals:

```python
# Before (current):
def _retopologize(HC, bV, dim, ...):
    # Disconnect all edges
    # Delaunay retriangulation
    # Recompute boundary, duals, volumes

# After:
def _retopologize(HC, bV, dim, ..., remesh_mode='adaptive'):
    if remesh_mode == 'delaunay':
        # ... existing code (backward compatible)
    elif remesh_mode == 'adaptive':
        from hyperct.remesh import adaptive_remesh
        adaptive_remesh(HC, dim, mps=mps, ...)
        # Recompute boundary, duals, volumes
```

### Implementation Subtasks

Phase 1 — Quality metrics and 2D operations (complete):
- [x] `_quality.py`: triangle quality metrics (min angle, aspect ratio, area, mesh histogram)
- [x] `_operations_2d.py`: edge split, edge collapse, edge flip with field carry-over
- [x] `_interface.py`: interface constraint checks (`is_interface_edge`, `can_flip`, `can_collapse`)
- [x] Unit tests for each operation on simple meshes (`hyperct/tests/test_remesh.py`, 36 tests)
- [x] Validate: split+collapse round-trip preserves vertex count

Phase 2 — 2D adaptive driver (complete):
- [x] `_driver.py`: `adaptive_remesh` for dim=2 with split/collapse/flip/smooth sweeps
- [x] Laplacian smoothing with interface tangent constraint
- [x] `remesh_mode='adaptive'` / `remesh_kwargs=` plumbed through
      `_retopologize`, `_do_retopologize`, and all five dynamic integrators
- [x] Integration tests: `ddgclib/tests/test_adaptive_remesh.py` (6 tests —
      vertex count preservation, dual recomputation, interface preservation,
      Laplacian relaxation, end-to-end symplectic_euler run, delaunay default)
- [x] Validate: bulk-interior vertices stay out of the interface vertex set
      across a remesh sweep

Phase 3 — 3D operations (not started):
- [ ] `_operations_3d.py`: edge split, edge collapse, face swap, edge swap
- [ ] 3D quality metrics (dihedral angle, aspect ratio)
- [ ] Unit tests for each 3D operation
- [ ] Integration test: cube-to-droplet 3D with adaptive remesh

Phase 4 — Error indicators (not started):
- [ ] Curvature-based refinement near interface (smaller cells where kappa is large)
- [ ] Velocity gradient refinement (smaller cells in shear layers)
- [ ] Pressure jump refinement (smaller cells across phase boundaries)

### References

- Persson, P.-O. & Strang, G. (2004). "A Simple Mesh Generator in MATLAB".
  SIAM Review 46(2), 329-345.
- Freitag, L.A. & Ollivier-Gooch, C. (1997). "Tetrahedral mesh improvement
  using swapping and smoothing". Int. J. Numer. Methods Eng. 40(21), 3979-4002.
- Jiao, X. & Heath, M.T. (2004). "Common-refinement-based data transfer
  between non-matching meshes". Int. J. Numer. Methods Eng. 61(14), 2402-2427.
- Compere, G. et al. (2008). "Transient adaptivity applied to two-phase
  incompressible flows". J. Comput. Phys. 227(3), 1923-1942.
- Quan, S. & Schmidt, D.P. (2007). "A moving mesh interface tracking method
  for 3D incompressible two-phase flows". J. Comput. Phys. 221(2), 761-780.
- Alauzet, F. & Loseille, A. (2016). "A decade of progress on anisotropic
  mesh adaptation for computational fluid dynamics". Comput. Aided Des. 72, 13-39.

## Feature: Pressure-Preserving Mass Redistribution After Retriangulation
Status: Complete
Plan: `.claude/plans/drifting-meandering-allen.md`

When Lagrangian meshes are retriangulated (Delaunay reconnection), dual cell
volumes change even though vertex positions barely moved.  Since the EOS
computes `P = eos.pressure(m / Vol)`, this causes spurious pressure
discontinuities.  This feature redistributes vertex masses after every
retriangulation to preserve the pre-retriangulation pressure field.

Algorithm: snapshot `v.p` before retopo, compute `m_target = eos.density(p_before) * Vol_new`,
globally scale to conserve total mass exactly.

- [x] `ddgclib/operators/mass_redistribution.py` — core module
  - `snapshot_pressure(HC)`, `snapshot_pressure_multiphase(HC, n_phases)`
  - `redistribute_mass_single_phase(HC, dim, eos, bV, pressure_snapshot)`
  - `redistribute_mass_multiphase(HC, dim, mps, bV, pressure_snapshot)`
  - BC-aware: wall vertices excluded, newly injected vertices excluded, periodic domains supported
- [x] `_retopologize` — snapshot before disconnect, redistribute after dual caching
- [x] `_retopologize_multiphase` — per-phase snapshot + per-phase redistribution
- [x] `_do_retopologize` — forward `pressure_model`, `redistribute_mass`
- [x] All 5 integrators — `pressure_model=None, redistribute_mass=False` params
- [x] `ddgclib/operators/__init__.py` — exports updated
- [x] 15 unit tests in `test_mass_redistribution.py` (all passing)
- [x] 652 existing tests still passing (0 regressions)

## Feature: Rules-Based Mesh Quality Maintenance
Status: Not Started

Local mesh operations (split/collapse/flip) instead of global Delaunay
retriangulation.  Target edge lengths from curvature/strain-rate scale.
Convergence loop per macro timestep.

Reference implementations: MMG (mmgtools.org), Geometry Central, CGAL.
Implementation target: `hyperct/remesh/` module.

Already partially spec'd above under "Interface-Preserving Adaptive Refinement / Remeshing".

## Feature: Intrinsic Delaunay Triangulation (iDT)
Status: Not Started

Lawson's flip algorithm applied intrinsically (flip edges where opposite
angles sum > pi).  Connectivity-only changes — no geometry modification.
Dramatically improves discrete operator conditioning (cotangent Laplacian,
integrated gradients).

Variants: iDT (Delaunay flips), iDR (Delaunay refinement with minimum
angle guarantee ~30 deg), iODT (optimal Delaunay for vertex distribution).

Reference: Sharp, Soliman & Crane (2019) "Navigating Intrinsic Triangulations",
ACM Trans. Graph. 38(4). PDF in repo: `navigating_intrinsic_triangulations.pdf`.
Library: Geometry Central's `remesh()`.

## Feature: Adaptive Mesh Refinement
Status: Not Started

Error-indicator-driven refinement: refine near interfaces, high gradients,
high curvature.  Coarsen in smooth regions via vertex merging.  Combines
with mass redistribution (above) for mass-conservative refinement/coarsening.

When a new vertex is inserted (not in pressure snapshot), its mass is set
from neighbor-averaged pressure — the redistribution module already handles
this case.
