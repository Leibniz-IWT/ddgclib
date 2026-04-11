# ARCHITECTURE.md

High-level architecture of ddgclib's dynamic continuum simulation framework.

## Module Dependency Graph

```
hyperct (symlink -> ../hyperct/hyperct)
    Complex                          simplicial complex data structure
    ddg/
        compute_vd(HC, method=...)   barycentric/circumcentric dual vertices
        e_star, v_star, d_area       discrete DDG operators
        _geometry.py                 normalized, area_of_polygon, etc.
        plot_dual.py                 plot_dual_mesh_1D/2D/3D
    remesh/                          Interface-preserving adaptive remeshing
        _quality.py                  triangle_min_angle, aspect_ratio, mesh_quality_histogram
        _interface.py                is_interface_edge, can_flip, can_collapse
        _operations_2d.py            edge_split_2d, edge_collapse_2d, edge_flip_2d
        _driver.py                   adaptive_remesh(HC, dim=2, L_min, L_max, ...)
    _plotting.py                     plot_complex, animate_complex
    _backend.py                      get_backend("numpy"|"torch"|"gpu"|"multiprocessing")
    |
    v
ddgclib/
    initial_conditions.py ---------> IC classes (apply to mesh)
    _boundary_conditions.py -------> BC classes + BoundaryConditionSet
    _compat.py --------------------> Utility functions not in hyperct (triang_dual, etc.)
    operators/
        _registry.py                 MethodRegistry (pluggable methods)
        curvature.py                 Curvature_i, Curvature_ijk
        area.py                      Area_i, Area_ijk, Area, DualArea_i
        volume.py                    Volume, Volume_i
        stress.py -------+--------> Cauchy stress tensor pipeline (core)
                         |           dual_area_vector, dual_volume,
                         |           velocity_difference_tensor, strain_rate,
                         |           cauchy_stress, stress_force,
                         |           stress_acceleration, dudt_i,
                         |           scalar_gradient_integrated
        gradient.py -----+--------> thin wrappers (pressure_gradient, velocity_laplacian, acceleration)
    analytical/
        _divergence_theorem.py ----> integrated_gradient_1d/2d/3d (Gauss-Legendre)
        _sympy_integration.py -----> integrated_gradient_sympy_1d/2d/3d (exact)
        _integrated_comparison.py -> integrated_pressure_error, integrated_l2_norm,
                                     compare_stress_force, volume_averaged_scalar
    _method_wrappers.py              (backward-compat shim -> operators/)
                         |
    (imports e_star etc. from hyperct.ddg)
                         |
    dynamic_integrators/ |
        _integrators_dynamic.py ---> euler, symplectic_euler, rk45, euler_velocity_only, euler_adaptive
        _simulation.py ------------> DynamicSimulation, SimulationParams
    data/
        _io.py --------------------> save_state, load_state (JSON)
        _history.py ---------------> StateHistory (time-series recording)
    visualization/
        unified.py ----------------> plot_primal, plot_dual, plot_fluid, dynamic_plot_fluid
                                     (delegates base mesh to hyperct._plotting.plot_complex)
                                     dynamic_plot_fluid supports phase_field, interface_field,
                                     reference_R for multiphase overlays (1D/2D/3D)
        multiphase.py -------------> record_multiphase_frame, dynamic_plot_multiphase
                                     (convenience wrappers for multiphase cases)
        matplotlib_1d.py ----------> 1D scalar/velocity field overlays
        matplotlib_2d.py ----------> 2D scalar, vector field overlays
        matplotlib_3d.py ----------> 3D scatter, extract_slice_profile
        polyscope_3d.py -----------> Optional polyscope integration
        animation.py --------------> animate_scalar_1d, animate_scalar_2d
    scripts/
        view_polyscope.py ---------> Generic polyscope viewer for any StateHistory snapshots
                                     Usage: python -m ddgclib.scripts.view_polyscope --snapshots <dir>

    dem/
        __init__.py ------------------> Public API exports
        _particle.py -----------------> Particle dataclass + ParticleSystem container
        _contact.py ------------------> ContactDetector (spatial hash) + Contact
        _force_models.py -------------> HertzContact, LinearSpringDashpot, ContactForceModel ABC
        _integrators.py --------------> dem_velocity_verlet, dem_symplectic_euler, dem_step
        _bonds.py --------------------> SinterBond + BondManager (sintering aggregation)
        _liquid_bridge.py ------------> LiquidBridge + LiquidBridgeManager (capillary bridges)
        _coupling.py -----------------> FluidParticleCoupler (drag, IDW interpolation, feedback)
        _io.py -----------------------> save_particles, load_particles, import_particle_cloud
        _visualization.py ------------> plot_particles, plot_bridges, plot_bonds

    barycentric/ ------------------> DEPRECATED shim (re-exports from hyperct.ddg)
    circumcentric/ ----------------> DEPRECATED shim (re-exports from hyperct.ddg)
```

## Data Flow: 5-Step Dynamic Simulation Workflow

```
1. DOMAIN        2. BOUNDARY       3. INITIAL         4. INTEGRATOR       5. POST-PROCESS
                    CONDITIONS        CONDITIONS

HC = Complex(d)  bV = identify_    ic = CompositeIC(  from functools      save_state(HC, bV)
HC.triangulate()   cube_boundaries   ZeroVelocity,     import partial     history.query_*()
HC.refine_all()    (HC, lb, ub)      HydrostaticP,                        plot_fluid(HC, bV)
compute_vd(HC)                       UniformMass)     dudt_fn = partial(
                 bc_set =                               dudt_i,
                   BoundaryCondition ic.apply(HC, bV)    dim=d, mu=mu,
                   Set()                                 HC=HC)
                   .add(NoSlipWall)
                   .add(DirichletP)                    t = euler(
                                                         HC, bV, dudt_fn,
                                                         dt, n_steps,
                                                         bc_set=bc_set)

                                                       OR

                                                       sim = DynamicSimulation(HC, bV, params)
                                                       sim.set_initial_conditions(ic)
                                                       sim.set_boundary_conditions(bc_set)
                                                       sim.set_acceleration_fn(dudt_i)
                                                       sim.run(callback=history.callback)
```

## Dynamic Case Study File Convention

All dynamic case studies in ``cases_dynamic/`` save outputs **within their
own directory**, not in the project root:

```
cases_dynamic/<name>/
    <name>_2D.py              # Main simulation script
    <name>_3D.py              # 3D version
    view_polyscope.py         # Interactive polyscope viewer
    README.md                 # Run instructions and output listing
    src/
        _setup.py             # Setup function
        _params.py            # Physical parameters
        _analytical.py        # Analytical solution (if applicable)
        _plot_helpers.py      # Case-specific plot utilities
    fig/                      # Generated plots and animations (.mp4)
    results/
        snapshots/            # StateHistory JSON snapshots for replay
```

**Animation workflow** (use in every dynamic case):

1. Record with ``StateHistory(fields=['u', 'p', ...], save_dir=_SNAPSHOTS)``
2. Pass ``history.callback`` as the integrator callback
3. After simulation: ``dynamic_plot_fluid(history, HC, save_path=...)``
4. For multiphase: add ``phase_field='phase'``, ``interface_field='is_interface'``
5. For polyscope: ``python -m ddgclib.scripts.view_polyscope --snapshots <dir>``

## Data Flow: DEM-Fluid Coupled Simulation

```
1. PARTICLE CLOUD    2. DEM CONFIG        3. COUPLED LOOP              4. POST-PROCESS

ps = import_         detector =           for step in range(n_steps):  save_particles(ps, t)
  particle_cloud(      ContactDetector      symplectic_euler(           plot_particles(ps)
    positions,           (ps)                  HC, bV, dudt_fn,         plot_bridges(
    radii, rho_s)      model =                dt, n_steps=1)             ps, bridge_mgr)
                         HertzContact()      coupler.fluid_to_
bridge_mgr =          coupler =                particle(dt)
  LiquidBridge          FluidParticle       dem_step(ps, detector,
    Manager(              Coupler(HC, ps,     model, dt, dim,
    gamma=0.072)          mu=mu)              bridge_manager=
bond_mgr =                                     bridge_mgr)
  BondManager()                             coupler.particle_to_
                                               fluid(dt)
```

## Key Abstractions

### Vertex Data Model
Every vertex `v` in `HC.V` carries:
- `v.x` — coordinate tuple (cache key)
- `v.x_a` — numpy array position
- `v.nn` — set of neighbor vertices (1-ring)
- `v.u` — velocity vector (`np.ndarray`, dim-sized)
- `v.p` — pressure (scalar `float`)
- `v.m` — mass (scalar `float`)
- `v.vd` — dual vertices (after `compute_vd()`)
- `v.boundary` — `True`/`False` (required before `compute_vd()`)

### InitialCondition ABC
```python
class InitialCondition(ABC):
    def apply(self, HC, bV: set) -> None: ...
```
Concrete: `UniformPressure`, `HydrostaticPressure`, `LinearPressureGradient`,
`ZeroVelocity`, `UniformVelocity`, `PoiseuillePlanar`, `HagenPoiseuille3D`,
`CustomFieldIC`, `UniformMass`, `CompositeIC`.

### BoundaryCondition ABC
```python
class BoundaryCondition(ABC):
    def apply(self, mesh, dt, target_vertices=None) -> int: ...
```
Concrete: `NoSlipWallBC`, `DirichletVelocityBC`, `DirichletPressureBC`,
`NeumannBC`, `OutletDeleteBC`, `PeriodicInletBC`.

### BoundaryConditionSet
Container applying multiple BCs in order:
```python
bc_set = BoundaryConditionSet()
bc_set.add(NoSlipWallBC(dim=3), wall_vertices)
bc_set.add(DirichletPressureBC(0.0), outlet_vertices)
diagnostics = bc_set.apply_all(HC, bV, dt)
```

### Cauchy Stress Tensor Operators (operators/stress.py)
Core operator module implementing the integrated Cauchy momentum equation on
Lagrangian parcels:

```
m_i * dv_i/dt = F_stress_i = sum_j sigma_f @ A_ij
```

where `sigma_f = 0.5 * (sigma_i + sigma_j)` is the face-averaged Cauchy
stress and `A_ij` is the oriented dual area vector (outward from parcel i).

Functions (build on each other bottom-up):

| Function | Purpose |
|----------|---------|
| `dual_area_vector(v_i, v_j, HC, dim)` | Oriented area vector A_ij of dual face (2D/3D) |
| `dual_volume(v, HC, dim)` | Dual cell measure (area in 2D, volume in 3D) |
| `velocity_difference_tensor(v, HC, dim)` | DDG integrated du_i tensor (dim x dim) |
| `strain_rate(du)` | Symmetric part: epsilon = 0.5 * (du + du^T) |
| `cauchy_stress(p, du, mu, dim)` | Newtonian: sigma = -p*I + 2*mu*epsilon |
| `stress_force(v, dim, mu, HC)` | Total F_stress_i on dual cell |
| `stress_acceleration(v, dim, mu, HC)` | a_i = F_stress_i / m_i |
| `dudt_i` | Alias for `stress_acceleration` |

The old `gradient.py` functions are now thin wrappers:
- `pressure_gradient(v, dim, HC)` -> `stress_force(v, dim, mu=0, HC=HC)`
- `acceleration(v, dim, mu, HC)` -> `stress_acceleration(v, dim, mu, HC)`

### Integrator Interface with dudt_i
```python
from functools import partial
from ddgclib.operators.stress import dudt_i

# Bind physics parameters via partial (avoids HC keyword conflict)
dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)

# Pass to any integrator
t = euler_velocity_only(HC, bV, dudt_fn, dt=1e-4, n_steps=100, dim=2, bc_set=bc_set)
```
All integrators share the signature:
```python
def integrator(HC, bV, dudt_fn, dt, n_steps, dim=3, callback=None, bc_set=None, **dudt_kwargs) -> float
```
The `bc_set` is applied after each step. The `callback` auto-detects old
(3-arg) vs new (5-arg) signatures.

**Important**: Since integrators have `HC` as their first positional arg,
`dudt_i` must be used via `functools.partial` to pre-bind `dim`, `mu`, `HC`.
Passing them as `**dudt_kwargs` causes a "multiple values for argument 'HC'"
error.

### DDG Backend Architecture
Dual mesh computations support multiple backends via `hyperct._backend`:
- **NumpyBackend** — default, pure NumPy
- **MultiprocessingBackend** — parallel CPU via `multiprocessing`
- **TorchBackend** — PyTorch CPU tensors
- **CudaBackend** — PyTorch CUDA (auto-detected when `backend="gpu"`)

Usage: `compute_vd(HC, method="barycentric", backend="torch")`

### Validation Conventions

**All scalar fields on vertices represent volume-averaged values:**
`v.p = (1/Vol_i) * ∫_{V_i} P dV`, not `P(x_vertex)`. The IC classes
(`HydrostaticPressure`, `LinearPressureGradient`) handle this automatically
when duals are available. Use `volume_averaged_scalar()` for custom fields.

**Always use integrated comparisons for validation:**
```python
from ddgclib.analytical import integrated_pressure_error, integrated_l2_norm
errs = integrated_pressure_error(HC, interior, P_analytical, dim=2)
```
Never use point-wise `abs(v.p - P(x_vertex))` — it conflates discretization
error with the point-vs-average mismatch.

**Benchmarks:**
```bash
python benchmarks/run_integrated_benchmarks.py --linear-only  # machine precision check
python benchmarks/run_integrated_benchmarks.py --convergence  # convergence study
```

## Test Architecture (~604 tests)

All tests in `ddgclib/tests/`:
- `test_initial_conditions.py` — 20 tests
- `test_boundary_conditions.py` — 15 tests
- `test_operators.py` — 11 tests
- `test_integrated_validation.py` — 47 tests (gradient, stress, curvature, known-solution)
- `test_stress.py` — 58 tests (50 fast + 8 slow)
  - 2D/3D dual geometry (closure, antisymmetry, magnitude, volume)
  - Strain rate, Cauchy stress, stress force (uniform field zeros)
  - Gradient wrapper backward compatibility
  - `dudt_i` alias and integrator integration (all 5 integrators)
  - Hagen-Poiseuille 2D/3D validation (equilibrium, developing flow, profile shape)
  - Hydrostatic 2D/3D validation (pressure force direction, viscous damping)
- `test_dynamic_integrators.py` — 24 tests
- `test_data.py` — 16 tests
- `test_visualization.py` — 54 tests (1 skipped: polyscope)
- `test_case_hydrostatic.py` — 10 tests
- `test_case_hagen_poiseuille.py` — 8 tests
- `test_gpu_backend.py` — 11 tests (5 skipped without torch/CUDA)
- `test_manuscript_tutorials.py` — 60 tests (4 skipped, pre-existing)

DEM tests in `ddgclib/tests/`:
- `test_dem_particle.py` — 27 tests (Particle, ParticleSystem)
- `test_dem_contact.py` — 15 tests (sphere-sphere detection)
- `test_dem_forces.py` — 16 tests (Hertz, LSD, registry)
- `test_dem_integrators.py` — 13 tests (free fall, elastic collision, dem_step)
- `test_dem_bonds.py` — 17 tests (SinterBond, BondManager)
- `test_dem_liquid_bridge.py` — 16 tests (LiquidBridge, LiquidBridgeManager, 1 slow integration)
- `test_dem_coupling.py` — 22 tests (interpolation, drag, feedback, I/O)

Total: ~415 tests (407 passed, 8 skipped)

Run fast tests only: `pytest ddgclib/tests/ -m "not slow"`
Run all including slow 3D: `pytest ddgclib/tests/`
Run DEM tests only: `pytest ddgclib/tests/test_dem_*.py -v`
