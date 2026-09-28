[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010952.svg)](https://doi.org/10.5281/zenodo.8010952)

# ddgclib
Experimental library for discrete differential geometry curvature in fluid simulations, especially useful for problems with complex and changing topologies.

ddgclib is a **Lagrangian** fluid simulator built on **discrete differential geometry (DDG)**: the mesh *is* the fluid. Every vertex carries velocity, pressure and mass and is advected by the flow, and the physics is expressed as *integrated* operators over the dual cells of a simplicial complex (the `Complex` data structure from the [`hyperct`](https://github.com/stefan-endres/hyperct) package). At equilibrium the integrated DDG stress operators reproduce analytical solutions (Hagen-Poiseuille flow, hydrostatic column) to machine precision; active work extends this to dynamic multiphase flows.

> Status: Alpha (v0.4.3), under active refactoring. The **dynamic continuum (Cauchy stress)** pipeline is the active core; the older **mean curvature flow** pipeline (`mean_flow_integrators/`, `cases_mean_flow/`) is legacy.

## Core capabilities

- **Cauchy stress tensor operators** (`ddgclib.operators.stress`) — the integrated momentum
  equation on Lagrangian parcels: `F_stress_i = Σ_j σ_f · A_ij` with `σ = −pI + 2με`.
  Validated against Hagen-Poiseuille and hydrostatic equilibrium.
- **Dynamic integrators** (`ddgclib.dynamic_integrators`) — `euler`, `symplectic_euler`,
  `rk45`, `euler_adaptive`, `euler_velocity_only`, plus a `DynamicSimulation` runner. All
  retopologize the moving mesh each step and apply boundary conditions automatically.
- **Initial & boundary conditions** — composable `InitialCondition` / `BoundaryCondition`
  classes (`CompositeIC`, `HydrostaticPressure`, `NoSlipWallBC`, `PeriodicInletBC`,
  `OutletDeleteBC`, …) managed via `BoundaryConditionSet`.
- **Domain builders** (`ddgclib.geometry.domains`) — one-liners for standard CFD meshes:
  `rectangle`, `disk`, `annulus`, `l_shape`, `box`, `ball`, `cylinder_volume`, `pipe`,
  plus periodic and multiphase-droplet variants. Each returns a `DomainResult` with the
  mesh, boundary vertices, named boundary groups, and metadata.
- **Parametric surfaces** (`ddgclib.geometry`) — `sphere`, `catenoid`, `cylinder`,
  `hyperboloid`, `torus`, `plane` with translate/scale/rotate transforms.
- **Multiphase framework** — sharp-interface model (`multiphase.py`), phase-aware stress and
  surface tension (`operators.multiphase_stress`, `operators.surface_tension`,
  `operators.curvature_2d`), exact dual-volume splitting, and pressure-preserving mass
  redistribution.
- **Equation of state** (`ddgclib.eos`) — `TaitMurnaghan` (weakly compressible liquid),
  `IdealGas`, and a `MultiphaseEOS` dispatcher.
- **DEM submodule** (`ddgclib.dem`) — self-contained Discrete Element Method: particles,
  contact detection, Hertz / linear-spring-dashpot force models, sintered bonds, capillary
  liquid bridges, and two-way fluid-particle coupling.
- **Periodic boundary conditions** (`ddgclib.geometry.periodic`) — ghost-cell merge with
  minimum-image duals, wired through all integrators.
- **Backends** — dual-mesh computation runs on NumPy (default), multiprocessing, PyTorch
  CPU, or CUDA via `compute_vd(HC, method="barycentric", backend="gpu")`.
- **Validation & data** — analytical integrated comparisons (`ddgclib.analytical`),
  conservation diagnostics (`ddgclib.data`), JSON state I/O, `StateHistory` recording, and
  matplotlib/Polyscope visualization and animation (`ddgclib.visualization`).

See [ARCHITECTURE.md](ARCHITECTURE.md) and [FEATURES.md](FEATURES.md) for the full module
map and feature roadmap, and the Sphinx docs (below) for the API reference and user guides.

## Quick start: a dynamic simulation

The canonical workflow is **domain → dual mesh → physics → integrate**. Physics parameters
are bound to the acceleration function `dudt_i` with `functools.partial` (this avoids an
`HC` keyword clash with the integrator's first positional argument):

```python
from functools import partial
from hyperct.ddg import compute_vd
from ddgclib.geometry.domains import rectangle
from ddgclib.operators.stress import dudt_i
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.initial_conditions import CompositeIC, ZeroVelocity, UniformMass
from ddgclib._boundary_conditions import BoundaryConditionSet, NoSlipWallBC

# 1. Domain (mesh + boundary groups)
result = rectangle(L=10.0, h=1.0, refinement=3)
HC, bV = result.HC, result.bV

# 2. Dual mesh
compute_vd(HC, method="barycentric")

# 3. Initial & boundary conditions
CompositeIC(ZeroVelocity(dim=2), UniformMass(total_volume=result.metadata['volume'], rho=1.0)).apply(HC, bV)
bc_set = BoundaryConditionSet()
bc_set.add(NoSlipWallBC(dim=2), result.boundary_groups['walls'])

# 4. Integrate (mesh moves with the fluid)
dudt_fn = partial(dudt_i, dim=2, mu=0.1, HC=HC)
t = symplectic_euler(HC, bV, dudt_fn, dt=1e-4, n_steps=1000, dim=2, bc_set=bc_set)
```

# Installation

## Conda (recommended)

Create a conda environment called `ddg` with all dependencies:

```bash
conda env create -f environment.yml
conda activate ddg
pip install -e .
```

To update an existing `ddg` environment after pulling new changes:

```bash
conda env update -f environment.yml --prune
```

## pip only

```bash
pip install -e .
```

## Dependencies

**Required:** `numpy`, `scipy`, [`hyperct`](https://github.com/stefan-endres/hyperct)

**Optional:**
- Visualization: `polyscope`, `matplotlib` — install with `pip install -e ".[vis]"`
- Data: `pandas` — install with `pip install -e ".[data]"`
- GPU acceleration: `torch` — install with `pip install -e ".[gpu]"`
- Development: `pytest`, `pytest-cov` — install with `pip install -e ".[dev]"`

Python 3.9+ required (development uses Python 3.13).

# Running Tests

```bash
# Run all tests
pytest ddgclib/tests/ -v

# Run fast tests only (skip slow convergence studies)
pytest ddgclib/tests/ -v -m "not slow"

# Run manuscript tutorial tests only
pytest ddgclib/tests/test_manuscript_tutorials.py -v
```

PyCharm run configurations are included in `.idea/runConfigurations/`.

# Tutorials and Manuscript figures

The tutorials progressively introduce the main functionality of the library and the most important functions in the source code, building up to show how the test cases from the manuscript can be reproduced:

- [Tutorial 1: Capillary rise](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%201%20Capillary%20rise.ipynb)
- [Tutorial 2: Particle-particle bridge](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%202%20particle-particle%20bridge.ipynb)
- [Tutorial 3: Sessile droplet comparison](https://github.com/Leibniz-IWT/ddgclib/blob/master/tutorials/Case%20study%203%20Sessile%20droplet.ipynb)

The original code for the manuscripts are contained in the `v0.3.1-alpha` tag. Currently we are refactoring the library so that some code might be broken. The published figures can be generated using the code inside `./manuscript_figures`

# Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org/) with the Read the Docs theme and includes user guides, API reference, and tutorials.

**Contents:**

- **Getting Started** — installation, quick start
- **User Guide** — simulation workflow, multiphase flows, DEM, mean curvature flow
- **API Reference** — operators, multiphase, EOS, BCs, ICs, integrators, geometry, visualization, DEM

## Build the docs

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
cd docs
make html
```

The built HTML is in `docs/build/html/`. Open with:

```bash
xdg-open docs/build/html/index.html   # Linux
open docs/build/html/index.html        # macOS
```

For live-reload during editing:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/build/html
# visit http://127.0.0.1:8000
```
