# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ddgclib** (Discrete Differential Geometry Curvature Library) — an experimental Python library for discrete differential geometry curvature in fluid simulations. Used for mean curvature flow, finite volume methods, and problems with complex/changing topologies. Currently in Alpha (v0.4.3), under active refactoring.

## Setup and Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `hyperct`. Optional: `matplotlib`, `polyscope` (3D visualization), `pandas`, `torch` (GPU acceleration).

Python 3.9+ required. Development uses Python 3.13 (conda env `ddg`, see `environment.yml`).

## Running Tests

Tests use `unittest` and `pytest`. Some 3D case-study tests are marked `@pytest.mark.slow`.

```bash
# Fast tests only (~18s for full suite including DEM)
pytest ddgclib/tests/ -v -m "not slow"

# All tests including slow 3D case studies
pytest ddgclib/tests/ -v

# DEM tests only
pytest ddgclib/tests/test_dem_*.py -v -m "not slow"

# Stress tensor tests only
pytest ddgclib/tests/test_stress.py -v -m "not slow"

# Slow 3D validation tests only
pytest ddgclib/tests/test_stress.py -v -m "slow"

# Run manuscript tutorial tests
pytest ddgclib/tests/test_manuscript_tutorials.py -v

# Run a single test
pytest ddgclib/tests/test_manuscript_tutorials.py::TestClassName::test_method -v
```

Test data files are loaded from `test_data/` directory (JSON format) by the test suite.

### Integrated Validation Benchmarks

Benchmarks comparing DDG integrated operators against analytically integrated solutions over dual cells:

```bash
# Full benchmark suite (linear precision + method comparison + convergence)
python benchmarks/run_integrated_benchmarks.py

# Linear precision check only (all methods, expect machine precision)
python benchmarks/run_integrated_benchmarks.py --linear-only

# Method comparison table
python benchmarks/run_integrated_benchmarks.py --comparison --dim 2

# Convergence study
python benchmarks/run_integrated_benchmarks.py --convergence --dim 2

# Integrated validation tests
pytest ddgclib/tests/test_integrated_validation.py -v -m "not slow"
```

Jupyter notebooks in `benchmarks/notebooks/` provide visual validation (01–03).

Jupyter notebooks in `tutorials/` and `cases_mean_flow/` serve as integration tests and examples.

## Architecture

### Hyperct (External Dependency — Symlink)

The primary mesh data structure and DDG computations are provided by the external **[hyperct](https://github.com/stefan-endres/hyperct)** package. The source is symlinked into this repo at `./hyperct` -> `/home/stefan_endres/projects/hyperct/hyperct`. The `Complex` class is used throughout as the simplicial complex backend:

```python
from hyperct import Complex

HC = Complex(n, domain=bounds)  # Create n-dimensional simplicial complex
for v in HC.V:                  # Iterate over all vertices
    print(v.x)                  # Vertex coordinates as tuple key
```

Key hyperct classes used: `Complex`, `VertexCacheField`, `VertexCacheIndex`.

#### Remesh Module (`hyperct.remesh`)

Interface-preserving adaptive remeshing (2D).  Drop-in alternative to
global Delaunay retopologization for Lagrangian multiphase flow where
a sharp `v.phase` interface must be preserved.

```python
from hyperct.remesh import adaptive_remesh
stats = adaptive_remesh(HC, dim=2, L_min=0.5 * h, L_max=1.4 * h,
                        quality_target_deg=20.0, max_iterations=3)
```

Exposed via ddgclib: pass `remesh_mode='adaptive'` (and optional
`remesh_kwargs={...}`) to any dynamic integrator (`euler`,
`symplectic_euler`, `rk45`, `euler_velocity_only`, `euler_adaptive`)
or directly to `_retopologize`.  Default remains `'delaunay'`
(backward compatible).  Local ops: `edge_split_2d`,
`edge_collapse_2d`, `edge_flip_2d`; quality metrics:
`triangle_min_angle`, `triangle_aspect_ratio`, `mesh_quality_histogram`;
interface constraints: `is_interface_edge`, `can_flip`, `can_collapse`.
3D operations are still on the roadmap (see DEVELOPMENT.md Phase 3).

#### DDG Module (`hyperct.ddg`)

All discrete differential geometry dual mesh computation lives in `hyperct.ddg`:

```python
from hyperct.ddg import compute_vd, e_star, v_star, d_area

compute_vd(HC, method="barycentric")   # or method="circumcentric"
compute_vd(HC, method="barycentric", backend="torch")  # GPU acceleration
```

Operators: `e_star(v, dim, HC)`, `v_star(v, dim, HC)`, `d_area(v, dim, HC)`.
Geometry helpers in `hyperct.ddg._geometry`: `normalized`, `area_of_polygon`, `volume_of_geometric_object`, etc.

#### Backends (`hyperct._backend`)

```python
from hyperct._backend import get_backend
backend = get_backend("numpy")          # default
backend = get_backend("torch")          # PyTorch CPU
backend = get_backend("gpu")            # PyTorch CUDA (auto-detect)
backend = get_backend("multiprocessing") # parallel CPU
```

#### Plotting (`hyperct._plotting`)

```python
from hyperct._plotting import plot_complex, animate_complex
plot_complex(HC, show=True)             # static mesh plot
animate_complex(HC, update_state, ...)  # animation
```

Reference examples in `hyperct_examples/`.

### Barycentric / Circumcentric Modules (Deprecated)

**`ddgclib/barycentric/`** and **`ddgclib/circumcentric/`** are now deprecation shims that re-export from `hyperct.ddg`. All new code should use `hyperct.ddg` directly.

### Domain Builder Module (`ddgclib/geometry/domains/`)

One-liner functions for constructing common CFD simulation domains.  Each returns a `DomainResult` with the mesh, boundary vertices, and named boundary groups:

```python
from ddgclib.geometry.domains import rectangle, cylinder_volume, ball

# 2D channel
result = rectangle(L=10.0, h=1.0, refinement=3, flow_axis=0)
HC, bV = result.HC, result.bV
result.boundary_groups  # {'walls', 'inlet', 'outlet', 'bottom_wall', 'top_wall'}
result.metadata['volume']  # 10.0

# 3D cylinder
result = cylinder_volume(R=0.5, L=2.0, refinement=2, flow_axis=2)
result.boundary_groups  # {'walls', 'inlet', 'outlet'}
```

**Available builders:**
- 2D: `rectangle()`, `l_shape()`, `disk()`, `annulus()`
- 3D: `box()`, `cylinder_volume()`, `pipe()`, `ball()`

**Projection utilities:** `cube_to_disk(HC, R)` and `cube_to_sphere(HC, R)` generalize the cube-to-cylinder pattern. Distribution laws: `'sinusoidal'` (default, recommended), `'linear'`, `'power'`, `'log'`.

**Agent pattern** (standard recipe for AI-generated geometry):
```python
from ddgclib.geometry.domains import rectangle
result = rectangle(L=10.0, h=1.0, refinement=3)
bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups['walls'])
ic = CompositeIC(..., UniformMass(total_volume=result.metadata['volume'], rho=1.0))
```

**Note:** 3D projected domains (cylinder, ball) need `_retopologize()` before `compute_vd()` — the dynamic integrators handle this automatically.

### Core Computational Pipeline (Lagrangian Formalism)

This library follows a **Lagrangian formalism** — the mesh moves with the fluid. Vertices carry velocity, pressure, and mass as attributes and are advected by the flow. Do NOT suggest switching to Eulerian (fixed-mesh) integrators; `euler_velocity_only` exists for validation/equilibrium checks only.

1. **Mesh construction**: Use domain builders (`from ddgclib.geometry.domains import rectangle`) for standard geometries, or `from hyperct import Complex; HC = Complex(n, domain=bounds)` for custom domains
2. **Boundary tagging**: Set `v.boundary = True/False` for all vertices (required before `compute_vd`). Domain builders call `tag_boundaries()` automatically.
3. **Dual mesh**: `hyperct.ddg.compute_vd(HC, method="barycentric"|"circumcentric")` computes dual vertices
4. **Stress tensor operators**: `ddgclib/operators/stress.py` — the core physics module implementing the integrated Cauchy momentum equation:
   - `dual_area_vector(v_i, v_j, HC, dim)` — oriented dual face area vector A_ij
   - `stress_force(v, dim, mu, HC)` — total stress force F_stress_i = sum_j sigma_f @ A_ij
   - `stress_acceleration(v, dim, mu, HC)` — acceleration a_i = F_stress_i / m_i
   - `dudt_i` — alias for `stress_acceleration`, used as `dudt_fn` in integrators
5. **Integration**: `dynamic_integrators/` provides time-stepping (euler, symplectic_euler, rk45, euler_velocity_only, euler_adaptive). The primary integrators are Lagrangian (`symplectic_euler`, `euler`, `rk45`) which update both velocity and position. `euler_velocity_only` is Eulerian (fixed mesh) and used only for validation. Use `functools.partial` to bind `dudt_i` parameters:
   ```python
   from functools import partial
   from ddgclib.operators.stress import dudt_i
   dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)
   symplectic_euler(HC, bV, dudt_fn, dt=1e-4, n_steps=100, dim=2, bc_set=bc_set)
   ```
6. **Boundary conditions for Lagrangian flow**: `PeriodicInletBC` injects vertices from a ghost mesh at the inlet, `OutletDeleteBC` removes vertices that exit the domain, and `PositionalNoSlipWallBC` enforces no-slip at walls. Use `boundary_filter` in integrators to control which topological boundary vertices are frozen (excluded from integration) — typically only wall vertices should be frozen, not inlet/outlet boundary vertices.
7. **Visualization**: `ddgclib.visualization.plot_fluid(HC, bV)` for pressure + velocity snapshots
8. **Animation**: Use `StateHistory` to record snapshots during simulation, then `dynamic_plot_fluid(history, HC)` to generate `.mp4` videos.  For multiphase, pass `phase_field='phase'` and `interface_field='is_interface'` to overlay interface markers.  For interactive 3D replay, use `python -m ddgclib.scripts.view_polyscope --snapshots <dir>`.

#### Dynamic Case Study Convention

All dynamic case studies in `cases_dynamic/` follow this pattern:

1. **Outputs go in the case directory**, not the project root:
   - `cases_dynamic/<name>/fig/` — plots and animations
   - `cases_dynamic/<name>/results/snapshots/` — JSON state snapshots
2. **Use `StateHistory`** with `save_dir` for snapshot persistence:
   ```python
   history = StateHistory(fields=['u', 'p'], record_every=N, save_dir=_SNAPSHOTS)
   ```
   For multiphase cases, also record `'phase'` and `'is_interface'`.
3. **Generate animations** with `dynamic_plot_fluid`:
   ```python
   from ddgclib.visualization import dynamic_plot_fluid
   dynamic_plot_fluid(history, HC, save_path=os.path.join(_FIG, 'anim.mp4'),
                      phase_field='phase', interface_field='is_interface')
   ```
4. **Include a README.md** in each case directory with run instructions.
5. **Include a view_polyscope.py** (or point to the generic one) for interactive replay.

#### FVM Conventions — Volume-Averaged Fields

All scalar fields on vertices (``v.p``, etc.) represent **volume-averaged**
values over the dual cell, not point values at the vertex position:

    v.p = (1/Vol_i) * ∫_{V_i} P(x) dV

This ensures ``v.p * Vol_i = ∫ P dV`` to machine precision for polynomial
pressure fields.  **Never assign ``v.p = P(x_vertex)``** — use the
``volume_averaged_scalar`` utility or the IC classes (``HydrostaticPressure``,
``LinearPressureGradient``), which handle this automatically when duals are
available.

For validation, **always use integrated comparisons** from
``ddgclib.analytical``:

```python
from ddgclib.analytical import (
    integrated_pressure_error,  # |p_i * Vol_i - ∫ P dV| per vertex
    integrated_l2_norm,         # volume-weighted L2 norm
    compare_stress_force,       # force balance diagnostic
    volume_averaged_scalar,     # (1/Vol) ∫ f dV for assigning fields
)
```

Never use point-wise comparisons like ``abs(v.p - P(x_vertex))`` — these
conflate discretization error with the point-vs-average mismatch.

#### Mean Curvature Flow Pipeline (Legacy)
- **Curvature computation**: `_curvatures.py` — `construct_HC()`, `HC_curvatures()`, `curvatures()`, `b_curvatures()`
- **Mean flow integration**: `mean_flow_integrators/` — Euler, Adams-Bashforth, Newton-Raphson, line search
- **Volume/area**: `geometry/_volume.py` for volume conservation; `geometry/curved_volume/` for curved surface calculations

### Method Wrappers (`_method_wrappers.py`)

Pluggable computation methods via registry dictionaries and wrapper classes:
- `Curvature_i`, `Curvature_ijk` — curvature method selection
- `Area_i`, `Area_ijk`, `Area` — area computation methods
- `Volume`, `Volume_i` — volume computation methods

Methods are registered in `_curvature_i_methods`, `_area_i_methods`, etc. and loaded from `benchmarks/_benchmark_toy_methods.py`.

### Physical Simulation Modules

- `_bubble.py` — Bubble interface shape/dynamics, force computation, remeshing, energy calculation
- `_capillary_rise.py` / `_capillary_rise_flow.py` — Capillary rise simulations
- `_sessile.py` — Sessile droplet computations
- `_eos.py` — Equation of state

### DEM Submodule (`ddgclib/dem/`)

Self-contained Discrete Element Method for spherical particle dynamics. Separate
from the fluid mesh (`HC`) — particles are their own data structure (`ParticleSystem`).

```python
from ddgclib.dem import (
    Particle, ParticleSystem, ContactDetector, HertzContact,
    dem_step, LiquidBridgeManager, BondManager,
    FluidParticleCoupler, import_particle_cloud,
    save_particles, load_particles, plot_particles,
)
```

Key classes:
- `Particle.sphere(x, radius, rho_s, dim)` — auto mass/inertia
- `ParticleSystem` — container with `add()`, `positions()`, `velocities()`, `radii()`
- `ContactDetector(ps)` — spatial hash broad-phase + sphere-sphere narrow-phase
- `HertzContact` / `LinearSpringDashpot` — pluggable contact force models
- `dem_step(ps, detector, model, dt, dim, n_sub=1)` — main DEM time integration
- `SinterBond` / `BondManager` — sintered bonds with Frenkel neck growth
- `LiquidBridge` / `LiquidBridgeManager` — capillary bridges (Lian et al. 1993)
- `FluidParticleCoupler(HC, ps, dim, mu)` — two-way drag coupling
- `import_particle_cloud(positions, radii, rho_s, dim)` — NumPy array import

DEM tests: `pytest ddgclib/tests/test_dem_*.py -v` (124 fast + 1 slow)

### Key Patterns

- **Vertex `x` as tuple key**: Vertices are identified by their coordinate tuple `v.x`; positions stored as array in `v.x_a`
- **Boundary vertices**: `bV` (a set of boundary vertex objects) is passed through most computation functions
- **`v.boundary` attribute**: Must be set to `True`/`False` on all vertices before calling `compute_vd()`. Tag with: `for v in HC.V: v.boundary = v in bV`
- **HC convention**: `HC` refers to the simplicial `Complex` object throughout the codebase (imported from `hyperct`)
- **`nn` (nearest neighbors)**: `v.nn` provides the 1-ring neighborhood of a vertex
- **`for v in HC.V`**: Standard pattern for iterating over all vertices in the complex
- **`dudt_i` with `functools.partial`**: The canonical way to use the stress tensor pipeline with integrators. Always bind `dim`, `mu`, `HC` via `partial()` — do NOT pass them as `**dudt_kwargs` (causes "multiple values for HC" error)
- **Prefix `b_`**: Functions prefixed with `b_` use barycentric dual computations (e.g., `b_curvatures`)
- **Color convention**: Visualization uses `coldict` from `ddgclib._misc` — `'db'` (dark blue) for points/edges, `'lb'` (light blue) for triangle faces

## Directory Layout

- `ddgclib/` — Main package
  - `barycentric/` — Deprecated shim (re-exports from `hyperct.ddg`)
  - `circumcentric/` — Deprecated shim (re-exports from `hyperct.ddg`)
  - `_compat.py` — Compatibility utilities for functions not in hyperct
  - `operators/` — Stress tensor (core), gradient wrappers, curvature, area, volume operators
  - `geometry/` — Geometric operations, volume/area computations, curved volume pipeline
    - `domains/` — Domain builders: `rectangle`, `disk`, `cylinder_volume`, `ball`, etc. (56 tests)
  - `dynamic_integrators/` — Time-stepping for dynamic PDE simulations
  - `mean_flow_integrators/` — Mean curvature flow time-stepping
  - `visualization/` — Plotting (delegates base mesh to `hyperct._plotting`)
  - `dem/` — DEM submodule (particles, contacts, forces, integrators, bonds, bridges, coupling, I/O, viz)
  - `tests/` — Test suite (~415 tests, 407 pass, 8 skipped)
- `hyperct` — **Symlink** to `../hyperct/hyperct` (external dependency source)
- `hyperct_examples/` — Example scripts demonstrating hyperct API
- `benchmarks/` — Benchmark methods and toy implementations (imported by `_method_wrappers.py`)
- `cases_mean_flow/` — Mean curvature flow case studies
- `cases_dynamic/` — Dynamic flow case studies (Poiseuille, hydrostatic, capillary rise)
- `tutorials/` — Jupyter notebook tutorials (Case studies 1–4)
- `test_cases/` — Standalone test case scripts and mesh files (`.msh`)

## Commit Convention

Prefixes: `ENH:` (enhancement), `BUG:` (bugfix), `MAINT:` (maintenance/refactoring)
