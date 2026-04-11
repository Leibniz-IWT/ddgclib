# FEATURES.md

Feature roadmap for ddgclib's dynamic continuum simulation framework.

## Implemented

### Initial Conditions (`ddgclib/initial_conditions.py`)
Composable IC classes with `apply(HC, bV)` interface. Supports scalar pressure
(P), vector velocity (u), and mass fields. Includes analytical Poiseuille
profiles with `analytical_velocity()` methods for validation.

### Boundary Conditions (`ddgclib/_boundary_conditions.py`)
Dirichlet (fixed value), Neumann (fixed gradient), no-slip wall, outlet
deletion, and periodic inlet BCs. Managed via `BoundaryConditionSet` container
that applies BCs in insertion order. Includes boundary identification helpers.

### Cauchy Stress Tensor Operators (`ddgclib/operators/stress.py`)
Core physics module implementing the integrated Cauchy momentum equation on
Lagrangian parcels. Computes the total stress force on each dual cell via
face-averaged Cauchy stress tensors contracted with oriented dual area vectors:

    F_stress_i = sum_j sigma_f @ A_ij
    sigma = -p * I + 2 * mu * epsilon  (Newtonian fluid)

Functions: `dual_area_vector`, `dual_volume`, `velocity_difference_tensor`,
`strain_rate`, `cauchy_stress`, `stress_force`, `stress_acceleration`.
The `dudt_i` alias provides the canonical interface for dynamic integrators.

Works in 2D and 3D. Validated against Hagen-Poiseuille (equilibrium residual,
developing flow, parabolic profile shape) and hydrostatic column (pressure
force direction, zero viscous stress, viscous damping).

### Operators Package (`ddgclib/operators/`)
Pluggable computation methods via `MethodRegistry`. The stress tensor module
(`stress.py`) is the core operator for dynamic simulations. The old
`gradient.py` functions (`pressure_gradient`, `velocity_laplacian`,
`acceleration`) are now thin wrappers around the stress tensor pipeline.
Curvature/area/volume wrappers in `curvature.py`, `area.py`, `volume.py`.

### Enhanced Dynamic Integrators (`ddgclib/dynamic_integrators/`)
Five integrators: Euler, symplectic Euler, RK45, velocity-only Euler, and
adaptive Euler with CFL-based time stepping. All support `bc_set` parameter
for automatic BC enforcement. Use `functools.partial` to bind `dudt_i`
parameters before passing to integrators. `DynamicSimulation` runner class
bundles mesh, ICs, BCs, and integrator into a convenient interface.

### Data Handling (`ddgclib/data/`)
JSON-based save/load for simulation state (vertices, connectivity, fields,
boundary membership, time). `StateHistory` class records snapshots during
simulation for post-processing queries.

### Visualization (`ddgclib/visualization/`)
Matplotlib plotting for 1D/2D/3D scalar fields, vector fields (quiver), and
meshes. Base mesh rendering delegates to `hyperct._plotting.plot_complex` with
ddgclib's color scheme (`coldict['db']` dark blue for points/edges,
`coldict['lb']` light blue for faces). Dual mesh rendering delegates to
`hyperct.ddg.plot_dual`. Slice profile extraction. Optional Polyscope 3D
integration. Animation utilities from `StateHistory` data. Unified
`plot_primal()` and `plot_dual()` wrappers auto-detect `HC.dim` and dispatch
to the appropriate backend. Both accept abstract `scalar_field` and
`vector_field` parameters for field overlays (e.g. pressure coloring, velocity
arrows). `plot_fluid()` creates multi-panel pressure + velocity snapshots.
`dynamic_plot_fluid()` creates video animations from `StateHistory`.

### GPU Acceleration (`hyperct._backend`)
Dual mesh computations support multiple backends: NumPy (default),
multiprocessing (parallel CPU), PyTorch CPU, and PyTorch CUDA (GPU). Enabled
via `compute_vd(HC, method="barycentric", backend="torch")` or
`backend="gpu"` for CUDA. Install torch via `pip install ddgclib[gpu]` or
use the conda environment (`environment.yml`).

### Hydrostatic Column Case (`cases_dynamic/Hydrostatic_column/`)
Reference case for pressure operator validation. 1D/2D/3D with analytical
hydrostatic pressure profile. Tests equilibrium (zero acceleration) and
perturbation recovery. 2D/3D stress tensor validation tests verify pressure
force direction, zero viscous stress at rest, and viscous damping of
perturbations.

### Hagen-Poiseuille Case (`cases_dynamic/Hagen_Poiseuile/`)
Reference case for viscous shearing operator validation. 2D planar Poiseuille
and 3D channel flow with analytical velocity profiles. Tests equilibrium
residual convergence, developing flow from plug flow initial conditions,
and qualitative parabolic profile shape. Both 2D and 3D validated using
the Cauchy stress tensor pipeline.

### DEM Submodule (`ddgclib/dem/`)
Self-contained Discrete Element Method for spherical particle dynamics with
fluid coupling. Supports contact mechanics, sintered bond aggregation, and
capillary liquid bridge formation.

**Core DEM**: `Particle` dataclass with `Particle.sphere()` classmethod (auto
mass/inertia from density and radius). `ParticleSystem` container with batch
accessors for positions, velocities, radii. `ContactDetector` with cell-linked
list spatial hashing for broad phase and exact sphere-sphere narrow phase.
Pluggable force models via `ContactForceModel` ABC: `HertzContact` (non-linear
elastic, `F_n ∝ δ^{3/2}`) and `LinearSpringDashpot` (Cundall & Strack).
`contact_force_registry` for runtime model selection. Time integration via
`dem_velocity_verlet` (symplectic 2nd-order) and `dem_symplectic_euler`.
`dem_step` entry point with sub-stepping for stiff contact dynamics.

**Sintered Bonds**: `SinterBond` spring-dashpot along bond axis with fracture
criterion. Frenkel/Kuczynski neck growth model (`grow_neck(dt, T, D)`).
`BondManager` with cluster ID tracking for aggregate identification.

**Capillary Bridges**: `LiquidBridge` with toroidal approximation capillary
force and linear decay. Lian et al. 1993 rupture criterion. `LiquidBridgeManager`
tracks formation/rupture via `frozenset` pair keys.

**Fluid Coupling**: `FluidParticleCoupler` with IDW interpolation of fluid
velocity at particle centres. Stokes and Schiller-Naumann drag models.
Two-way coupling: `fluid_to_particle()` computes drag, `particle_to_fluid()`
distributes reaction forces (Newton III).

**I/O**: `save_particles`/`load_particles` JSON format (`ddgclib_dem_state_v1`).
`import_particle_cloud()` from NumPy arrays for external particle generation
codes. Supports monodisperse/polydisperse radii and wetting parameters.

**Visualization**: `plot_particles()` (matplotlib 2D/3D), `plot_bridges()`,
`plot_bonds()`, optional `plot_particles_polyscope()`.

## Planned

### Dynamic Capillary Rise Case
3D case combining pressure, viscous, and curvature forces. Requires surface
tension on meniscus interface. Depends on curvature operators from mean flow
pipeline. Deferred pending physics input.

### Body Force Integration
Add gravity/body force terms to `stress_acceleration()` so that hydrostatic
equilibrium can be verified with actual DDG operators giving zero net
acceleration (gravity balancing pressure gradient).

### Dam Break Test Case
2D dam break scenario — rectangular domain with pressure discontinuity and
gravity. Smoke test for the full tensor pipeline on transient problems.

### Adaptive Mesh Refinement / Remeshing

**Status:** Phase 1–2 Complete (2D only); 3D and error indicators pending.

Interface-preserving adaptive mesh refinement and remeshing for 2D
Lagrangian multiphase simulations.  Implemented in the new
`hyperct.remesh` module (with a shim in `ddgclib/dynamic_integrators`)
to replace the global Delaunay retopologization with **local** mesh
operations that never produce cross-phase edges, keeping interior
droplet vertices away from the discrete interface.

**2D operations** (``hyperct/remesh/_operations_2d.py``): edge split,
edge collapse, edge flip with Delaunay-quality gain check.  All
operations update ``v.nn``, the vertex cache, and carry field values
(``u``, ``p``, ``phase``, ``m``) across — mass is averaged on split
and summed on collapse.

**Interface-aware constraints** (``hyperct/remesh/_interface.py``):
`is_interface_edge`, `can_flip`, `can_collapse` forbid operations
that would destroy a sharp `v.phase` boundary or mutate a wall
vertex.

**Adaptive driver** (``hyperct/remesh/_driver.py``): `adaptive_remesh`
runs split / collapse / flip / Laplacian-smoothing sweeps until the
minimum-angle target is met.  Interface vertices are smoothed
tangentially (along their same-phase interface neighbours only), so
the shape of the interface is preserved.

**Integration**: ddgclib's `_retopologize` now takes
``remesh_mode={'delaunay'|'adaptive'}`` and ``remesh_kwargs``.  All
five dynamic integrators (``euler``, ``symplectic_euler``,
``euler_velocity_only``, ``euler_adaptive``, ``rk45``) forward these
to the retopologize stage.  The default remains ``'delaunay'`` for
backward compatibility.

```python
from ddgclib.dynamic_integrators import symplectic_euler
t = symplectic_euler(
    HC, bV, dudt_fn, dt=1e-4, n_steps=500, dim=2,
    remesh_mode='adaptive',
    remesh_kwargs={'L_min': 0.5 * h, 'L_max': 1.4 * h,
                   'quality_target_deg': 20.0},
)
```

**Tests**: `hyperct/tests/test_remesh.py` (36 unit tests) and
`ddgclib/tests/test_adaptive_remesh.py` (6 integration tests) cover
quality metrics, each local operation, interface constraints, and
end-to-end runs through the integrators.

**3D operations** and **error-indicator-driven refinement** remain
on the roadmap — see DEVELOPMENT.md.

References:
- Persson & Strang (2004). "A Simple Mesh Generator in MATLAB". SIAM Review 46(2).
- Freitag & Ollivier-Gooch (1997). "Tetrahedral mesh improvement using swapping
  and smoothing". Int. J. Numer. Methods Eng. 40(21).
- Jiao & Heath (2004). "Common-refinement-based data transfer between
  non-matching meshes". Int. J. Numer. Methods Eng. 61(14).
- Compere et al. (2008). "Transient adaptivity applied to two-phase
  incompressible flows". J. Comput. Phys. 227(3).
- Quan & Schmidt (2007). "A moving mesh interface tracking method for 3D
  incompressible two-phase flows". J. Comput. Phys. 221(2).

### HDF5 Data Handling
Replace JSON text serialization with HDF5 binary format (h5py). Each snapshot
is self-contained (full mesh + fields) to support Lagrangian topology changes
every step. ~13x smaller files, ~25x faster writes, random-access reads, crash
recovery. Legacy JSON preserved for backward compatibility. StateHistory
refactored from dict-of-dicts to list-of-numpy-arrays for in-memory efficiency.

### hyperct Geometry I/O
New `hyperct/io` module replacing incomplete `save_complex`/`load_complex`
stubs. JSON (`hyperct_v1`) and HDF5 formats with index-based connectivity,
format versioning, and `Complex.to_arrays()`/`from_arrays()` intermediate
representation. ddgclib delegates topology serialization to hyperct.

### Migrate Geometry to hyperct.ddg
Move `dual_area_vector` and `dual_volume` (pure geometry functions) from
`ddgclib.operators.stress` to `hyperct.ddg._operators`. Update ddgclib
imports to use hyperct.ddg.

### Multiphase: 2D Integrated Interface Curvature (Implemented)
The 2D surface tension operator now uses the exact integrated dual curvature:

    F_st_i = integral_{Gamma_i} gamma * kappa * N ds = gamma * (t_next - t_prev)

This identity follows from the Fundamental Theorem of Calculus applied to
the tangent vector along a piecewise linear interface curve — it is not an
approximation. For constant curvature (the circle) it reconstructs the arc
length and enclosed area to machine precision (see
`dev_notebooks/2D_machine_precision_area_from_curvatures`).

Implementation lives in `ddgclib/operators/curvature_2d.py`:
- `integrated_curvature_normal_2d(v)` — vector form of ∫κN ds
- `surface_tension_force_2d(v, gamma)` — gamma * ∫κN ds
- `reconstruct_arc_length_and_bulge_area(v_i, v_j, Delta_T)` —
  closed-form L, A_bulge, r, c from the notebook

`multiphase_stress._interface_surface_tension` delegates to this module
for 2D, so all existing 2D multiphase cases pick it up transparently.
Tests: `ddgclib/tests/test_curvature_2d.py` (18 tests — magnitude matches
`2 sin(pi/N)`, inward direction, closed-curve closure, machine-precision
arc-length reconstruction).

### Multiphase: Exact Dual Volume Splitting at Interface
The current interface dual volume split (`MultiphaseSystem.split_dual_volumes`)
uses a neighbour-count approximation: the fraction of 1-ring neighbours in each
phase estimates the volume fraction.  Planned improvements:

1. **Geometric split (2D)**: Intersect the barycentric dual polygon with the
   interface curve (line segment per edge) to compute exact sub-polygon areas
   for each phase.
2. **Geometric split (3D)**: Intersect the barycentric dual polyhedron with
   the interface surface (plane per face) to compute exact sub-volumes.
3. **Convergence test**: Verify that the approximate split converges to the
   exact split under mesh refinement for a circle/sphere interface.

### Multiphase: N-Phase Contact Lines
Extend `MultiphaseSystem` to handle three-phase contact lines (e.g.
fluid-fluid-solid) by computing boundaries of the interface surface mesh.
Triple-point vertices would carry mass/pressure from 3 phases.

### Oscillating Droplet: Analytical Initial Conditions
Set initial velocity field from the Lamb/Prosperetti analytical solution
so that the simulation starts with the correct mode shape and can be
compared against the analytical decay/oscillation from t=0.
