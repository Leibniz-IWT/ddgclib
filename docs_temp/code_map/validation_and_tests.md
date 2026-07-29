# Validation Infrastructure & Test Coverage Map
> Sources: ddgclib/analytical/{__init__.py,_integrated_comparison.py,_divergence_theorem.py,_sympy_integration.py}, ddgclib/data/{__init__.py,_io.py,_history.py,_conservation.py}, benchmarks/run_integrated_benchmarks.py (skim), ddgclib/tests/* (names + docstrings; test_stress.py, test_multiphase*.py, test_integrated_validation.py in detail) | Written: 2026-07-02 by understand-and-document workflow

## 1. Analytical integration package — `ddgclib/analytical/`

Core identity used everywhere: **divergence theorem** `∫_V ∇f dV = ∮_{∂V} f n dA` — reduces dual-cell volume integrals to boundary integrals. Two paths: numeric Gauss-Legendre (`_divergence_theorem.py`) and exact symbolic sympy (`_sympy_integration.py`, optional dep; `ddgclib.analytical.HAS_SYMPY` flag at `__init__.py:59`).

### 1.1 Integrated-comparison utilities — `ddgclib/analytical/_integrated_comparison.py`

FVM doctrine: `v.p` is a **volume average** `(1/Vol_i)∫_{V_i} P dV`, never a point value. Validation must compare `|p_i·Vol_i − ∫_{V_i} P dV|`, NOT `|v.p − P(x_vertex)|`.

Public signatures (all re-exported from `ddgclib.analytical`):

```python
volume_averaged_scalar(f_analytical: Callable, v, dim=2,
    polygon_method="barycentric_dual_p_ij", n_gauss=7) -> float   # L166
    # (1/Vol_i) ∫_{V_i} f dV — the correct value to ASSIGN to v.p
integrated_pressure_error(HC, interior_vertices: list, P_analytical, dim=2,
    polygon_method="barycentric_dual_p_ij", n_gauss=7) -> list[float]  # L222
    # per-vertex |p_i*Vol_i − ∫ P dV|
integrated_l2_norm(HC, interior_vertices, P_analytical, dim=2, ...) -> float  # L282
    # sqrt(Σ_i (p_i − <P>_i)^2 Vol_i / Σ_i Vol_i), <P>_i = ∫P dV / Vol_i
compare_stress_force(HC, interior_vertices, dim=2, mu=1e-3) -> dict  # L350
    # {'max_F','mean_F','median_F','F_norms'} of ||stress_force(v)|| — equilibrium check
```

Internals: `_dual_cell_pressure_integral_1d` (L39, uses `hyperct.ddg.dual_cell_vertices_1d`), `_dual_cell_pressure_integral_2d_simple` (L116, uses `hyperct.ddg.dual_cell_polygon_2d(v, include_edge_midpoints=...)` + centroid fan-triangulation + symmetric triangle quadrature). Dual volume comes from `ddgclib.operators.stress._get_dual_vol` (stress.py:402). Degenerate/boundary cells fall back to point-wise `f(x)·dual_vol`.

**IMPORTANT LIMITATION**: for `dim == 3` all three comparison functions use the **point-wise approximation** `P(x_vertex)·Vol` ("exact 3D dual cell integration via faces is deferred" — L214, L271-273, L334-336). So "integrated comparison" is only truly integrated in 1D/2D; 3D pressure comparisons re-introduce the point-vs-average mismatch these utilities were built to eliminate. (Note `integrated_gradient_3d` in `_divergence_theorem.py` DOES exist for polyhedra given explicit face lists — the gap is dual-cell **face extraction** in 3D, not the quadrature.)

### 1.2 Divergence-theorem quadrature — `ddgclib/analytical/_divergence_theorem.py`

```python
integrated_gradient_1d(f, a, b) -> (1,)                    # L41, exact: f(b)-f(a)
integrated_gradient_2d(f, polygon(N,2) CCW, n_gauss=10) -> (2,)   # L67
integrated_gradient_2d_vector(u, polygon, n_gauss=10) -> (dim_u,2) # L119, ∮ u⊗n dA
integrated_gradient_3d(f, faces: list[(M,3) outward-oriented], n_gauss=7) -> (3,)  # L170
integrated_gradient_3d_vector(u, faces, n_gauss=7) -> (3,3)  # L227
_gauss_legendre_01(n)                                       # L21, GL nodes on [0,1]
_triangle_quadrature_points(n)                              # L271: n<=1 centroid (deg 1),
    # n<=3 midpoint (deg 2), n<=4 4-pt (deg 3), else 7-pt (deg 5)
```
Outward normal for CCW polygon edge: `n_out = [edge_y, -edge_x]` (unnormalized, |n|=edge length). Exact for polynomial degree d with `(d+1)//2 + 1` GL points/edge.

### 1.3 Sympy path — `ddgclib/analytical/_sympy_integration.py`
`integrated_gradient_sympy_1d(f_expr, x_sym, a, b)` (L18), `integrated_gradient_sympy_2d(f_expr, x_sym, y_sym, polygon)` (L47), `integrated_gradient_sympy_2d_vector(u_exprs, ...)` (L105, NOT re-exported in `__init__`), `integrated_gradient_sympy_3d(f_expr, x,y,z syms, faces)` (L151). Edge/triangle parameterizations integrated exactly with `sympy.integrate`. Cross-validated against the Gauss path in `tests/test_integrated_validation.py::TestSympyCrossValidation`.

## 2. Data / recording package — `ddgclib/data/`

Exports (`__init__.py`): `save_state, load_state, StateHistory, compute_conservation, as_jsonable, drift_fractions`.

### 2.1 StateHistory — `ddgclib/data/_history.py` (class at L27)

```python
StateHistory(fields=('u','p'), record_every=1, record_every_t=None,
             save_dir=None, conservation=False, dim=None)
```
- `.callback(step, t, HC, bV=None, diagnostics=None)` (L85) — pass directly as integrator `callback=`; supports both old 3-arg and new 5-arg integrator callback signatures. Records when `step % record_every == 0`, OR (priority) when `t - last >= record_every_t - 1e-15`.
- Snapshot storage: `list[(t, {coord_tuple_key: {field: value}}, diagnostics_dict)]`; ndarray fields are `.copy()`-ed (deep, verified by `test_data.py::test_vector_field_snapshot`).
- `conservation=True` → merges `compute_conservation(HC, dim=self.dim)` into each snapshot's diagnostics (opt-in, off by default).
- `save_dir` set → each snapshot auto-written via `save_state()` to `{save_dir}/state_{idx:06d}_t{time:.6f}.json`.
- Query API: `.append(t, HC, diagnostics=None)` (manual record, L134), `.query_vertex(vertex_key: tuple, field) -> (times, values)` (L156), `.query_field_at_time(t, field) -> {key: value}` (nearest snapshot, L185), `.query_diagnostics() -> list[(t, diag)]` (L214), `.times`, `.n_snapshots`, `.clear()`.
- Multiphase convention (CLAUDE.md): also record `'phase'` and `'is_interface'` fields for `dynamic_plot_fluid` overlays.

### 2.2 Conservation diagnostics — `ddgclib/data/_conservation.py`

```python
compute_conservation(HC, dim=None) -> dict   # L86, stateless, once per snapshot
```
Returned keys: `ke` (Σ ½m|u|²), `momentum` (len-dim array Σ m·u), `mass_total`, `volume_total` (Σ dual_vol), `n_vertices`, `h_min`/`h_max` (edge-length extrema over `v.nn`, L187), `u_max`/`u_min` (speed extrema), `p_min`/`p_max` (only if any `v.p` present). Per-phase (only when `v.m_phase`/`v.dual_vol_phase` exist): `ke_phase`, `mass_phase`, `volume_phase` (n_phases auto-detected, L77).
- Mass fallback (`_vertex_mass`, L36): prefers `v.m`; if absent **or zero**, falls back to `v.dual_vol` (pure-CFD proxy).
- `as_jsonable(diag)` (L206): ndarray→list, np scalar→python for JSON dumps.
- `drift_fractions(initial, current, keys=('mass_total','volume_total')) -> dict[str,float]` (L219): `|cur−init|/|init|`; zero initial → returns `abs(cur)`.

### 2.3 State I/O — `ddgclib/data/_io.py`

```python
save_state(HC, bV: set, t=0.0, fields=('u','p','m'), path='state.json',
           extra_meta=None) -> str          # L29
load_state(path) -> (HC, bV, meta)          # L118
```
Format `'ddgclib_state_v1'`: vertices (coords + scalar/array fields), edges as coord-pair list, `boundary_coords`, `time`, `dim`, `fields`, optional `meta`. `load_state` reconstructs `Complex(dim, domain=bbox-of-coords)` — domain bounds are **inferred from vertex min/max**, not stored; vertices re-created via `HC.V[coord_tuple]`, connectivity via `v1.connect(v2)`. Dual quantities (`dual_vol`, `vd`) are NOT saved — must recompute `compute_vd` after load.

## 3. Benchmark runner — `benchmarks/run_integrated_benchmarks.py` (skim)

Compares DDG integrated gradient operators against analytically integrated solutions over the *same dual cells*. Sweeps: `DUAL_METHODS=["barycentric","circumcentric"]` (L53, passed to `compute_vd`), `POLYGON_METHODS=["barycentric_dual_p_ij","barycentric"]` (L56; p_ij = polygon includes edge midpoints), `SEEDS=[None, 42]` (symmetric vs jittered mesh), `CONVERGENCE_REFINES=[1,2,3]`. Machine-precision threshold in output color-coding: `err < 1e-13` = green/PASS.

Modes: `run_linear_precision_check(dim)` (L210, expect ALL < 1e-13), `run_method_comparison(dim, n_refine)` (L83), `run_convergence_study(dim, dual_method, polygon_method)` (L144, prints observed rate). CLI: `--dim`, `--refine`, `--linear-only`, `--comparison`, `--convergence`; no flags = all.

Benchmark case classes live in `benchmarks/_integrated_benchmark_cases.py`: `IntegratedGradientBenchmark` subclasses `Linear{Scalar,Vector}{1D,2D,3D}`, `Quadratic{Scalar,Vector}{1D,2D,3D}`, `Cubic{Scalar,Vector}2D`, `TrigScalar2D`, `PoiseuilleVector2D`; stress-level: `StressGradientBenchmark` (L297) → `PressureGradientBenchmark` (L618), `ViscousFluxBenchmark` (L667), `PoiseuilleBenchmark` (L709); surface: `CurvatureBenchmark` (L759). Each has `.run() -> {'max_abs_error', ...}` and `.build_mesh()/.compute_numerical()`.

## 4. Test suite map — `ddgclib/tests/` (~813 test functions across 37 files)

### 4.1 Stress / momentum operators — `test_stress.py` (2075 lines, 75 tests; only 2 classes slow)

Module doc: "Tests for ddgclib.operators.stress — Cauchy stress tensor operators." Key classes (all fast unless noted):
- `TestDualAreaVector2D` / `TestDualAreaVector3D`: **machine-precision structural invariants** — closure `Σ_j A_ij = 0` on interior dual cells (atol 1e-12), antisymmetry `A_ij = −A_ji` (atol 1e-12), |A_ij| matches `e_star` (rtol 1e-10).
- `TestDualVolume2D/3D`: positivity + partition-of-unity vs domain measure.
- `TestVelocityDifferenceTensor2D`: uniform u → du=0 (1e-10); linear u exact; `Du = du_pointwise · Vol_i` (rtol 1e-10); refinement convergence.
- `TestStrainRate`, `TestCauchyStress`: algebraic identities `ε = ½(∇u+∇uᵀ)`, `σ = −pI + 2με` (pointwise, exact).
- `TestStressForce2D/3D`, `TestGradientWrappers2D`: uniform p / uniform u → zero force/acceleration (atol 1e-10).
- `TestHagenPoiseuilleStress2D` (L905, FAST): equilibrium residual of `stress_acceleration` at analytical Poiseuille (u=(G/2μ)y(h−y), P=−Gx) — **median residual < 1e-13 (machine precision) on both refine=2 and refine=3 meshes** (`test_equilibrium_residual_converges`, L1022); pressure-force direction; developing flow with `dudt_i` incl. adaptive stepping. Fixture uses `cases_dynamic/Hagen_Poiseuile/src/_setup.setup_poiseuille_2d`.
- `TestHydrostaticStress2D`: force direction opposes gravity, zero viscous stress at u=0, `a = F_p/m`, KE decay under viscous perturbation.
- `TestHagenPoiseuilleStress3D` (L1357, **@slow**), `TestHydrostaticStress3D` (L1578, **@slow**): 3D analogs — directional/bounded/qualitative-profile assertions only, NOT machine precision (3D hydrostatic acceleration explicitly "nonzero").
- `TestDualVolCaching` (`cache_dual_volumes` stress.py:379, `_get_dual_vol` :402), `TestFaceCenteredViscousFlux` (diffusion form: linear shear→0, quadratic shear→nonzero; μ=0 stress_force == pressure_gradient), `TestIntegratedCauchyStress` (integrated σ / Vol == pointwise σ; pressure part −p·Vol·I), `TestEdgeAreaCache3D` (`batch_e_star` vs legacy `e_star`; **p_ij dual-area-vector linear precision in 3D incl. jittered mesh** with simplex-aware `compute_vd`), `TestDudtAlias`/`TestDudtIntegrators` (`dudt_i` is `stress_acceleration` at stress.py:829; works with all 5 integrators + `functools.partial` + `BoundaryConditionSet`), `TestPoiseuille2DIntegration`.

### 4.2 Integrated validation framework — `test_integrated_validation.py` (980 lines, 50 tests, 3 @slow)

- `TestDualCellGeometry`: 1D endpoints, 2D polygon CCW orientation / ≥6 vertices / contains primal vertex / areas < domain area.
- `TestIntegratedGradient{1D,Scalar2D,Vector2D}`: **machine precision** for linear fields on barycentric AND circumcentric duals AND jittered meshes; quadratic exact in 1D and on symmetric 2D mesh (superconvergence); quadratic/cubic/trig convergence on jittered meshes. Poiseuille profile u=[y(1−y),0] machine precision on symmetric mesh.
- `TestIntegratedGradient3D::test_linear_scalar_3d` — **@slow**; the ONLY 3D integrated-gradient test.
- `TestSympyCrossValidation`: sympy vs Gauss-Legendre agreement.
- `TestBenchmarkSuite`: regression wrappers around benchmark classes (linear = machine precision, quadratic/cubic/trig = converging).
- `TestKnownSolutionIntegrals`: hand-computed integrals over squares/triangles/rectangles/pentagon (atol 1e-13..1e-14) validating the quadrature framework itself.
- `TestIntegratedStress2D` (L806): hydrostatic pressure force < 1e-12 (linear P exact even jittered < 1e-10); linear velocity → zero viscous force < 1e-12; quadratic viscous force exact on symmetric mesh < 1e-10; **Poiseuille equilibrium net force < 1e-10 on symmetric mesh**; jittered Poiseuille only bounded < 0.1 (diffusion form has O(h) truncation on non-symmetric meshes).
- `TestIntegratedCurvature` (**@slow**, 2 tests): sphere H=1/R — only checks "runs, finite, doesn't blow up ×1.5"; accuracy limited by seam-degenerate triangles, not the operator.

### 4.3 Multiphase — 3 dedicated files + parts of others

- `test_multiphase.py` (413 lines, 24 tests, fast): unit tests for `MultiphaseSystem` (`assign_phases`, `identify_interface`, `split_dual_volumes`, `get_mu`, `get_gamma`), `IdealGas`/`MultiphaseEOS`, `MultiphaseMass` IC, and the Lamb/Rayleigh oscillating-droplet analytical solution (3D l=2: ω² = 8γ/(ρR₀³), β = 5μ/(ρR₀²); Young-Laplace ΔP = γ/R (2D), 2γ/R (3D); over/underdamped cases).
- `test_multiphase_flat_interface.py` (302 lines, 4 tests, fast): **"Tier 2A gatekeeper"** — flat-interface zero-force invariant for `multiphase_stress_force`, `ATOL = 1e-12` (machine precision), 2D and 3D. Variants: 2A.i γ=0 uniform per-phase pressure; 2A.ii γ=0.05 with geometrically exact κ=0. 3D uses a **Kuhn-decomposed structured cube mesh** (6 tets/cube sharing the (0,0,0)–(1,1,1) diagonal) because default `Complex(3).triangulate()` produces tets crossing z=0 → jagged interface with κ≠0. If these fail, residual interface-vertex force "D4" (see `docs/3d_multiphase_interface_pressure_fix.md`) is real.
- `test_multiphase_stress_per_phase.py` (243 lines, 6 tests, fast): per-phase summed scheme (Phase 6 rewrite of `ddgclib/operators/multiphase_stress.py`) — bulk-only mesh collapses to single-phase `stress_force`; uniform-pressure two-phase → zero net force (2D & 3D); `_face_viscosity_for_phase` returns exact μ_k.
- `test_a5b_longrun_regression.py` (2 tests, 1 @slow): **pinned regression floors** for steady-state max|F| on oscillating-droplet static fixture with retopology ON + `redistribute_mass=True`: 3D = 7.3768e-05 (after 2026-04-29 `redistribute_mass_multiphase` guard fix; ×195 improvement from 1.44e-3), 2D = 2.3749e-3. Trips on >1% shift from any change to mass-redistribution / per-phase split / retopology / curvature-stencil stack. Harness: `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py`.
- `test_case_oscillating_droplet.py` (12 tests, 3 @slow): dynamic multiphase droplet integration case.
- Related infrastructure: `test_interface_subcomplex.py` (11, primal-subcomplex interface model), `test_dual_split_2d.py` (24, exact 2D dual-volume split), `test_mass_redistribution.py` (15, pressure-preserving mass redistribution after retriangulation), `test_curvature_2d.py` (9, integrated 2D curvature operator), `test_simplex_aware_curvature.py` (16) / `test_simplex_aware_duals.py` (13, regression for stencil/dual refactors).

### 4.4 Integrators & dynamics

- `test_dynamic_integrators.py` (401 lines, 24 tests, fast): backward-compat (no bc_set, old 3-arg callback), `BoundaryConditionSet` applied each step, new 5-arg callback (`step, t, HC, bV, diagnostics`), `euler_velocity_only` constant-accel linearity, full `euler` position updates, `euler_adaptive` (reaches t_end; CFL reduces dt at high velocity), `rk45`, `SimulationParams`/`DynamicSimulation` runner incl. method chaining.
- `test_boundary_conditions.py` (21), `test_initial_conditions.py` (20), `test_periodic.py` (26, ghost-cell Delaunay periodic BCs), `test_retopo_displacement_gate.py` (13, `displacement_eps` skip-retopology gate), `test_adaptive_remesh.py` (12, interface-preserving remeshing), `test_mass_redistribution.py` (15).
- Case-level: `test_case_hagen_poiseuille.py` (8), `test_case_hydrostatic.py` (10, 1D/2D/3D column), `test_case_oscillating_droplet.py` (12/3 slow), `test_conservation.py` (18, exact-value checks on `compute_conservation`, per-phase splits, StateHistory integration, `drift_fractions`), `test_data.py` (16, save/load round-trips + StateHistory).
- **Stub**: `test_hagen_poiseuille_equilibrium.py` contains only the comment `# Moved to cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py` — 0 tests.

### 4.5 Machine-precision-verified physics vs weaker/untested

Machine precision (≤1e-12..1e-13):
- Dual-face closure & antisymmetry of `dual_area_vector` (2D & 3D).
- Integrated gradient of **linear** scalar/vector fields — all dual methods, symmetric & jittered meshes, 1D/2D (3D linear only in one @slow test; p_ij 3D linear precision incl. jittered covered fast in `TestEdgeAreaCache3D`).
- Quadratic fields on symmetric meshes (superconvergence) and in 1D.
- 2D Poiseuille equilibrium residual (median < 1e-13) and net stress force (< 1e-10); hydrostatic linear-pressure force (< 1e-12, incl. jittered).
- Flat-interface multiphase zero-force, 2D & 3D (1e-12), with and without γ.

Bounded / directional / qualitative only:
- Jittered-mesh Poiseuille residual (< 0.1 — diffusion-form O(h) truncation).
- All 3D channel/hydrostatic stress cases (@slow, direction & profile shape only).
- Sphere curvature (finite + non-divergence only; mesh-quality-limited).
- Multiphase long-run stability: pinned empirical floors (7.38e-5 / 2.37e-3), not zero.

Untested / gaps:
- 3D **integrated pressure comparison** is point-wise (deferred in `_integrated_comparison.py`) — no test exercises true 3D dual-cell pressure integration.
- `integrated_gradient_3d_vector` and `integrated_gradient_sympy_2d_vector` have no direct dedicated tests spotted (sympy vector path not exported).
- No convergence-rate assertions for 3D operators; curved-interface multiphase curvature accuracy (the flat-interface gatekeeper deliberately excludes it — "remaining multiphase instability lives elsewhere").
- `load_state` does not persist duals or original domain bounds (bbox inferred) — untested against domains larger than the vertex hull.

## 5. Fast-feedback pytest commands

Run from repo root, conda env `ddg` (dev Python 3.13). Note: base env here has **no pytest** — activate `ddg` first.

```bash
pytest ddgclib/tests/ -v -m "not slow"                       # full fast suite (~18 s)
pytest ddgclib/tests/test_stress.py -v -m "not slow"          # stress ops (2D machine precision)
pytest ddgclib/tests/test_stress.py -v -m "slow"              # 3D channel/hydrostatic validation
pytest ddgclib/tests/test_integrated_validation.py -v -m "not slow"  # integrated-gradient framework
pytest ddgclib/tests/test_multiphase.py ddgclib/tests/test_multiphase_flat_interface.py \
       ddgclib/tests/test_multiphase_stress_per_phase.py -v   # multiphase (all fast)
pytest ddgclib/tests/test_a5b_longrun_regression.py -v -m "not slow"  # 2D pinned floor only
pytest ddgclib/tests/test_dynamic_integrators.py ddgclib/tests/test_conservation.py \
       ddgclib/tests/test_data.py -v                          # integrators + diagnostics + I/O
pytest ddgclib/tests/test_dem_*.py -v -m "not slow"           # DEM (124 fast + 1 slow)
pytest ddgclib/tests/test_manuscript_tutorials.py -v          # manuscript tutorial regressions
python benchmarks/run_integrated_benchmarks.py --linear-only  # all-methods machine-precision gate
python benchmarks/run_integrated_benchmarks.py --comparison --dim 2
python benchmarks/run_integrated_benchmarks.py --convergence --dim 2
```

Slow markers live in: test_stress.py (L1357, L1578), test_integrated_validation.py (L535, L945, L959), test_a5b_longrun_regression.py (L109), test_case_oscillating_droplet.py (L122, L146, L270), test_manuscript_tutorials.py (L817), test_dem_liquid_bridge.py (1).

## 6. Contradictions / stale claims noticed

1. **Stale module docstring** in `ddgclib/tests/test_multiphase_stress_per_phase.py` (~L10-13): claims cross-phase-face viscosity is the **harmonic mean** `2μ_iμ_j/(μ_i+μ_j)`, but the implementation (`ddgclib/operators/multiphase_stress.py:94-104 _face_viscosity_for_phase`) and the test class docstrings in the same file assert **exact per-phase μ_k, no blending** ("not a blend"; `test_multiphase.py::TestPhaseViscosity` doc: "per-phase, no harmonic mean"). Module docstring predates the Phase 6 rewrite.
2. CLAUDE.md says test suite is "~415 tests, 407 pass, 8 skipped" — the directory now holds **~813 test functions** across 37 files; the count is stale.
3. CLAUDE.md FVM section says to "always use integrated comparisons" — true in 1D/2D, but the 3D branches of `integrated_pressure_error`/`integrated_l2_norm`/`volume_averaged_scalar` silently degrade to point-wise × Vol (comments: "exact 3D dual cell integration via faces is deferred").
4. `test_hagen_poiseuille_equilibrium.py` is an empty tombstone pointing at `cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py` (outside `ddgclib/tests/`, so not collected by the documented commands).
5. Environment quirk: sandbox `base` python lacks pytest; test commands require the `ddg` conda env.
