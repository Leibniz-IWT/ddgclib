# Architecture & Code Map
> Sources: sources/architecture_and_conventions.md, code_map/hyperct_upstream.md, code_map/operators_stress.md, code_map/integrators_and_bcs.md, code_map/multiphase_surface_tension.md, code_map/validation_and_tests.md, sources/library_audit_and_features.md, sources/development_status.md | Written: 2026-07-02 by understand-and-document workflow

ddgclib v0.4.3 (Alpha), Lagrangian DDG CFD library. **The mesh IS the fluid**: vertices carry `u`, `p`, `m` and are advected; forces are integrated Cauchy stress over barycentric dual cells. Python 3.9+ (dev: 3.13, conda env `ddg`). Two pipelines: the **dynamic continuum (Cauchy stress) pipeline is the active core**; the mean-curvature-flow pipeline (`mean_flow_integrators/`, `cases_mean_flow/`, `_curvatures.py`, `_bubble.py`, `_capillary_rise*.py`, `_sessile.py`) is legacy. Never suggest Eulerian fixed-mesh integration; `euler_velocity_only` is validation-only.

---

## 1. Package layout & dependency graph

```
hyperct  (EXTERNAL, symlinked — mesh backend, §2)
    │
    ▼
ddgclib/
  operators/
    stress.py              ★ CORE physics: dual_area_vector, dual_volume, pressure_flux,
                             viscous_flux, stress_force, stress_acceleration, dudt_i
    multiphase_stress.py   per-phase force + surface-tension assembly
    surface_tension.py     thin-film (no-dual) -γ·HNdA force
    curvature_2d.py        exact 2D FTC curvature (t_next - t_prev)
    gradient.py            legacy wrappers (pressure_gradient = stress_force(mu=0), ...)
    curvature.py/area.py/volume.py + _registry.py   MethodRegistry-based legacy operators
    stress_pointwise.py    ARCHIVED failed formulation (kept as documentation of failure mode)
    mass_redistribution.py pressure-preserving mass redistribution across retopo
  multiphase.py            MultiphaseSystem, sharp-interface primal subcomplex, phase fields
  eos/                     TaitMurnaghan, IdealGas, MultiphaseEOS, eos_pressure_update
                           (ddgclib/_eos.py = 9-line dead code, ignore)
  dynamic_integrators/     euler, symplectic_euler, rk45, euler_velocity_only, euler_adaptive;
                           _retopologize machinery; DynamicSimulation/SimulationParams
  _boundary_conditions.py  BC classes + BoundaryConditionSet (1030 L)
  initial_conditions.py    IC classes (497 L)
  geometry/
    domains/               DomainResult builders: rectangle, l_shape, disk, annulus,
                           box, cylinder_volume, pipe, ball; cube_to_disk/sphere
    _dual_split_2d.py      exact interface dual-volume splitting (ALSO contains the 3D split;
                           _dual_split_3d.py is a NotImplementedError stub — misleading names)
    _interface_subcomplex.py  extract_interface, curve_neighbours
    periodic.py            ghost-cell Delaunay periodic BCs, retopologize_periodic
  analytical/              _divergence_theorem (Gauss-Legendre), _sympy_integration (exact),
                           _integrated_comparison (volume_averaged_scalar, integrated_pressure_error,
                           integrated_l2_norm, compare_stress_force)
  data/                    save_state/load_state (JSON), StateHistory, compute_conservation
  visualization/           unified.py (plot_primal, plot_fluid, dynamic_plot_fluid), multiphase.py,
                           polyscope_3d.py; scripts/view_polyscope.py
  dem/                     self-contained DEM (ParticleSystem, ContactDetector, HertzContact,
                           dem_step, bonds, liquid bridges, FluidParticleCoupler) — separate from HC
  barycentric/ circumcentric/  DEPRECATED shims re-exporting hyperct.ddg
  tests/                   ~813 test functions / 37 files (all doc'd counts stale)
benchmarks/                run_integrated_benchmarks.py + benchmark case classes
cases_dynamic/             dynamic case studies (see code_map/cases_dynamic_inventory.md)
cases_mean_flow/           legacy mean-flow cases (incl. equil_bubble thermodynamics toy)
```

Call-graph spine (single-phase): `compute_vd` → `dual_area_vector`/`batch_e_star` → `pressure_flux + viscous_flux` inside `stress_force` → `stress_acceleration` (=`dudt_i`) → integrator. Multiphase: `MultiphaseSystem.refresh` → `split_dual_volumes`/`compute_phase_pressures` → `multiphase_stress_force` → `multiphase_dudt_i`.

**Note**: `ddgclib/__init__.py` is entirely commented out — `import ddgclib` exposes nothing; import from submodules (`ddgclib.operators.stress`, `ddgclib.dynamic_integrators`, ...). `scalar_gradient_integrated` (`stress.py:486`) is missing from `operators/__init__.py` — import from `ddgclib.operators.stress` directly.

## 2. The hyperct boundary (what lives upstream vs here)

`./hyperct` is a **symlink** to the external package source. Actual target on this machine (verified 2026-07-02): `/home/endres/projects/bilevel_param/hyperct` — **both CLAUDE.md and ARCHITECTURE.md state stale targets**. Treat as read-only unless a task explicitly targets upstream. Full upstream map: [code_map/hyperct_upstream.md](code_map/hyperct_upstream.md).

| Lives in hyperct (upstream) | Lives in ddgclib |
|---|---|
| `Complex`, vertex caches (`VertexCacheIndex/Field`), `HC.V.move/remove/merge_all` | All physics: stress, multiphase, surface tension, EOS |
| `compute_vd` (dual mesh, barycentric/circumcentric strategies) | `dual_area_vector`, `dual_volume` (marked TODO to move upstream, `stress.py:49`) |
| `e_star`, `v_star`, `batch_e_star`, `d_area` (approximate) | Time integrators, retopologization drivers, BCs, ICs |
| Exact dual-cell extraction: `dual_cell_polygon_2d`, `dual_cell_area_2d`, `dual_cell_faces_3d` | Analytical validation, conservation diagnostics, state I/O |
| `connect_and_cache_simplices` (Delaunay), `invalidate_simplex_cache`, `boundary_from_simplices`, `apex_vertices` | Domain builders, interface subcomplex extraction, dual splitting |
| `SimplicialComplex` (modern container), `_ops` registries | DEM, visualization, cases |
| `remesh/` (2D interface-preserving adaptive remesh) | Multiphase mass redistribution, periodic geometry |
| `_backend.py` (`numpy`/`torch`/`gpu`/`multiprocessing`) | — |

Load-bearing upstream invariants (violating these silently degrades ddgclib):
1. **Coordinate tuple = identity**: `v.x` is the cache key and hash; reposition only via `HC.V.move()`.
2. **`v.nn` is the flag complex**, not true simplices — K_{dim+1} cliques ≠ simplices on Delaunay meshes. Simplex-aware code needs `HC._simplices` (populated by `connect_and_cache_simplices`, invalidated by `invalidate_simplex_cache` after ANY unrouted topology change) or `HC.SC`.
3. **`v.boundary` must be tagged before `compute_vd`** (missing attribute is silently swallowed).
4. **Duals are throwaway**: `compute_vd` rebuilds `HC.Vd` and every `v.vd` from scratch; any move/remove/remesh invalidates them; nothing auto-refreshes.
5. `Complex.boundary()` is documented-unreliable on Delaunay meshes; use `boundary_from_simplices(HC, dim)` or `HC.SC.boundary()`.
6. Known upstream bugs: `compute_vd` dual positions wrong near periodic faces (worked around in `stress.py:97-156`); `remesh/_operations_2d.py:198` edge-split mass averaging is non-conservative (blocks adaptive remesh in production); `_vertex.py:787` NameError; 3D batch dual path omits `vd_mid.connect(vd_face)` wiring.

## 3. Vertex data model

Every `v` in `HC.V` (iterate `for v in HC.V`):

| Attribute | Meaning |
|---|---|
| `v.x` | coordinate tuple — identity/cache key |
| `v.x_a` | position as ndarray (lazy) |
| `v.nn` | 1-ring neighbour set (flag complex) |
| `v.u` | velocity ndarray, dim-sized |
| `v.p` | pressure — **volume-averaged over dual cell** (FVM convention) |
| `v.m` | mass (fixed Lagrangian parcel mass) |
| `v.rho` | density, written by EOS resolution as side effect |
| `v.vd` | set of dual vertex objects (after `compute_vd`) |
| `v.dual_vol` | cached dual measure (`cache_dual_volumes` / `batch_e_star`) — **0.0 on boundary verts after integrator retopo** |
| `v.boundary` | bool, REQUIRED before `compute_vd` |
| multiphase: `v.phase` | int ≥0 bulk, `-1` = INTERFACE_PHASE (beware numpy negative-index wrap) |
| `v.is_interface`, `v.interface_phases` | interface tagging |
| `v.m_phase/p_phase/rho_phase/dual_vol_phase` | per-phase arrays (length n_phases) |

`bV` = set of boundary vertex objects, threaded through most functions; **rewritten wholesale every retopologization** via `boundary_filter` (see [04_solver_pipeline.md](04_solver_pipeline.md)).

## 4. Key APIs (signatures)

```python
# Dual mesh (hyperct)
compute_vd(HC, method="barycentric"|"circumcentric"|DualStrategy, cdist=1e-10, backend=None)
batch_e_star(vertices, HC, dim=3, backend=None, compute_volumes=False, orient=False)
    # -> (edge_areas {id(v): {id(nb): A_ij}}, failed_vertices[, vertex_volumes])

# Stress operators (ddgclib/operators/stress.py)
dual_area_vector(v_i, v_j, HC, dim=3) -> np.ndarray          # :52
dual_volume(v, HC, dim=3) -> float                            # :311
stress_force(v, dim=3, mu=8.9e-4, HC=None, pressure_model=None) -> np.ndarray  # :702
stress_acceleration(...) -> np.ndarray   # :778;  dudt_i = stress_acceleration  # :829
# canonical binding — NEVER via **dudt_kwargs ("multiple values for HC"):
dudt_fn = functools.partial(dudt_i, dim=2, mu=0.1, HC=HC)

# Integrators (ddgclib/dynamic_integrators/) — uniform interface, returns final t
integrator(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, bc_set=None,
           boundary_filter=None, retopologize_fn=None, remesh_mode='delaunay',
           remesh_kwargs=None, skip_triangulation=False, merge_cdist=...,
           displacement_eps=None, save_every=None, save_dir=None, workers=None,
           **dudt_kwargs) -> float
euler_adaptive(HC, bV, dudt_fn, dt_initial, t_end, cfl_target=0.5, dt_min=1e-12,
               dt_max=None, velocity_only=True, ...)          # different signature

# Domains (ddgclib/geometry/domains/)
result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)  # DomainResult
result.HC, result.bV, result.boundary_groups, result.metadata['volume']

# BCs / ICs
BoundaryConditionSet().add(bc, vertices=None)   # None -> tracks live bV
bc_set.apply_all(mesh, bV, dt) -> diagnostics   # insertion order
InitialCondition.apply(HC, bV); CompositeIC(*ics)

# Multiphase
mps = MultiphaseSystem(phases=[PhaseProperties(eos, mu, rho0, name), ...], gamma={(0,1): 0.05})
mps.refresh(HC, dim, reset_mass=False, split_method='neighbour_count'|'exact')
dudt_fn = partial(multiphase_dudt_i, dim=2, mps=mps, HC=HC)

# Validation / recording
volume_averaged_scalar(P, v, dim=2); integrated_pressure_error(HC, verts, P, dim=2)
compare_stress_force(HC, verts, dim=2, mu=1e-3) -> {'max_F', 'median_F', ...}
history = StateHistory(fields=('u','p'), record_every=1, save_dir=None, conservation=False)
save_state(HC, bV, t, fields, path); load_state(path)  # duals NOT persisted — recompute compute_vd
```

## 5. Extension points

1. **Constitutive models**: replace/augment `stress_force` — the dual-geometry layer (`dual_area_vector`, `dual_volume`) is physics-agnostic. TODO markers for viscoelastic/non-Newtonian/elastic at `stress.py:32-41`.
2. **`pressure_model` protocol**: `None` | callable `fn(v)->float` | object with `.pressure(rho)` (EOS) — dispatched in `_resolve_pressure` (`stress.py:634`).
3. **Custom `DualStrategy`**: any `Callable[[(n,dim) array], (dim,) array]` passed to `compute_vd(method=...)`.
4. **`retopologize_fn`**: integrators accept `False` (frozen topology), a custom callable `(HC, bV, dim)`, or default `_retopologize`; multiphase binds `partial(_retopologize_multiphase, mps=mps, ...)`.
5. **BC/IC ABCs**: subclass `BoundaryCondition.apply(mesh, dt, target_vertices)` / `InitialCondition.apply(HC, bV)`.
6. **hyperct `_ops` registries**: `register_builder/register_refiner/register_local_op` (Protocol + `OpContext` for incremental `SimplicialComplex` sync); `SimplicialComplex` mutation API (`add_simplex/remove_simplex/remove_vertex`, `faces(k)/cofaces(k)/boundary_operator(k)`); `_sc_hook` event channel.
7. **Backends**: `get_backend("numpy"|"torch"|"gpu"|"multiprocessing")`; batch hooks `batch_dual_positions`, `batch_cross_areas`, `build_sparse_boundary`.
8. **MethodRegistry** (`operators/_registry.py`): pluggable curvature/area/volume methods (legacy pipeline only — stress functions are called directly).

## 6. Dead code / API inconsistencies (short list; full audit in [sources/library_audit_and_features.md](sources/library_audit_and_features.md))

- `geometry/_dual_split_3d.py` = NotImplementedError stub; the real 3D split lives in `_dual_split_2d.py`.
- `ddgclib/_eos.py` (dead CoolProp shim), `barycentric/_duals.py` (empty), `legacy/plots.py`.
- Curvature module triplication: `_curvatures.py` (1717-line legacy) vs `_curvatures_heron.py` (active) vs `_curvatures_heron_torch_vectorized.py`.
- Four Fundamentals docs, two ARCHITECTURE docs; two claim "single source of truth" — the docs_temp set you are reading supersedes for orientation.
- Do NOT remove the `'stokes'` curvature path in `multiphase_stress.py` — deliberate regression-tested A/B probe.
- `hyperct/_simplex.py` is vestigial/broken (list-valued hash) — never build on it.

## 7. Conventions

- Commits: `ENH:` / `BUG:` / `MAINT:` prefixes.
- Case studies: `cases_dynamic/<name>/` with `src/`, `fig/`, `results/snapshots/`, `README.md`, `view_polyscope.py`; outputs never in project root.
- Tests: `pytest ddgclib/tests/ -v -m "not slow"` (~18 s) from conda env `ddg` (base python lacks pytest); slow 3D cases `@pytest.mark.slow`.
- Prefix `b_` = barycentric-dual variant. Colors via `coldict` in `ddgclib._misc` (`'db'` points/edges, `'lb'` faces).
- Run case scripts with `/home/endres/anaconda3/envs/ddg/bin/python` from repo root.
