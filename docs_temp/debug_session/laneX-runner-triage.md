# laneX: every runner under `cases_dynamic/` runs its short mode or is retired, and the case matrix says which

Date: 2026-10-06 / 07 (cloud session, branch `claude/relaxed-davinci-tkh95t`,
started from `2019349`; hyperct `48d163a`, untouched).  The first
implementer was killed by a container restart on 2026-10-07 at about
00:22 with the 2D full runs done and the 3D A/B half done; the
continuation (same day) verified its work, finished the 3D A/B, moved
the cube2droplet presets to the conservative remap (section 2.2), fixed
three defects it had not seen (the Case 5 `global`, the Case 5 figure
shape, the Case 5 `methods.json` never written when the hold is inert),
ran Cases 2, 3 and 4, and wrote sections 4.1, 4.5, 6 and the case matrix.
Machine: 4 cores, 15 GB, Python 3.13.16, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2.  Scratch (probe logs, scratch copies, the full runs):
`/tmp/claude-0/-home-user-ddgclib/091c4829-271f-5fa0-a0eb-eaae3d5b54e0/scratchpad/laneX/`.
Everything a number below needs is in the repository: the runners, the
presets, `ddgclib/tests/test_case_runners_smoke.py`.

## 0. Verdict

1. **Every runner under `cases_dynamic/` (capillary_rise* and the
   dynCA runners excluded, rule 3) either starts headlessly in a short
   mode and writes `methods.json`, or carries a `RETIRED` header
   docstring that says why and what replaces it.**  Section 1 is the
   probe table (one row per runner, exit code, time, traceback head);
   section 2 the fixes; section 3 the retirements; section 4 the
   measurements; section 5 the known limits.
2. **One library change: `phases='film'` on the registry**, plus the
   cube2droplet presets carrying `remap='conservative'` (section 2.2:
   the no-remap configuration loses the droplet; registered evidence on
   both remap options).  A thin-film
   surface mesh (a 2-manifold in 3D) has no dual mesh, so neither
   `dudt_i` nor `multiphase_dudt_i` applies; the force that does is the
   Heron surface-tension acceleration that already existed in
   `ddgclib/operators/surface_tension.py`.  `SolverMethods.dudt_fn(HC,
   gamma=, damping=)` binds it; the validation restricts the film to
   `connectivity='frozen'` or `'custom'` and refuses every bulk-only
   axis.  Presets: `liquid_bridge_film_3D` (Case 1 hold),
   `liquid_bridge_cfd_dem_3D` (custom surface remesh),
   `liquid_bridge_volume_3D` (Case 5, single phase, frozen),
   `cube_to_droplet_2D` / `_2D_dual_only` / `_3D`, `bc_demo_2D`.
3. **No duplicate solver code remains in a maintained demo**: the
   cube2droplet setup builds its force and retopology from
   `SolverMethods`; its five runners pass presets and run through
   `methods.integrate`; the BC-comparison runner's second copy of the
   setup and the two dual-only closures are gone; `bc_demo.py` advects
   through the library `euler` integrator; the catenoid cases bind the
   film force through the preset.  The files that still carry hand-rolled
   loops (`liquid_bridge_approach`, `dynamic_caprise_tube`, the bc_demo
   variants, cube_flow, `diagnostic_no_retopo.py`,
   `diagnose_split_methods.py`, one p_ref preview) are retired with the
   reason in their header (section 3); nothing was deleted.
4. **Measured, not assumed**: the cube2droplet runs are measured against
   the Laplace jump `gamma (dim - 1) / R_eq` through
   `ddgclib.analytical.integrated_phase_pressure_jump` (section 4.1, with
   an A/B of the shipped configuration against the conservative remap and
   the fixed-connectivity refresh); the catenoid hold against the minimal
   surface (zero axial force; section 4.2, with a measured DO-NOT on the
   local curvature of the hyperct grid mesh); the film of the CFD-DEM case
   against the sphere curvature `2 / R` (section 4.3, measured worse than
   the mesh deserves: a known limit of its quad grid).
5. The fast and slow suites are green up to the three environment
   failures of the baseline (section 6).

## 1. Probe: every runner, shortest mode, scratch copy, live hyperct

`MPLBACKEND=Agg PYTHONPATH=/home/user/ddgclib timeout 600 python <scratch copy>`
from the repo root, HEAD = `2019349` (before any change of this lane).
"rc" is the exit code, "t" the wall time.

| runner | rc | t | traceback head / result |
|---|---|---|---|
| `cube2droplet/cube_to_droplet_2D.py`, `_2D_adaptive.py`, `_2D_bc_comparison.py`, `_2D_mass_redist.py`, `_3D.py`, `diagnostic_no_retopo.py` | 1 | 0 s | `ModuleNotFoundError: No module named 'cases_dynamic.Cube2droplet'` (all six) |
| `cube_flow/cube_flow_1D.py` / `_2D.py` / `_3D.py` | 0 | 18 / 19 / 32 s | run to t = 4 s and write their figures; the flow is frozen: 2D "step 0: vertices=10", final 10 vertices, 5 boundary (13 at setup); audit M7 |
| `bc_demo/bc_demo.py` | 0 | 27 s | 120 steps, 23 vertices remaining, gif written |
| `bc_demo/bc_demo_v1.py` | 0 | 28 s | runs (case-local BC loop) |
| `bc_demo/bc_demo_v2.py` | 1 | 1 s | `ImportError: cannot import name '_rebuild_nn_from_delaunay' from 'ddgclib.visualization.unified'` (never shipped: `git log -S` finds only this file) |
| `bc_demo/bc_demo_parametric_study.py` | killed | > 600 s | the sweep points complete (`nr0_ns120 ... OK`, `nr1_ns120 ... OK`), each with an ffmpeg MP4; the whole sweep exceeds the timeout |
| `template/template.py` | 0 | 6 s | already on `SolverMethods` (laneK), state round-trip OK |
| `template/example_features_demo.py` | 0 | 8 s | writes `fig/example4_methods.json` |
| `liquid_bridge_dem/liquid_bridge_case.py` | 0 | 10 s | pure DEM (no fluid solver), results + figures |
| `liquid_bridge_cfd_dem/liquid_bridge_cfd_dem_case.py` | 1 | 2 s | `AttributeError: 'VertexCube' object has no attribute 'u'` in `surface_tension_acceleration` at step 0: `retopologize_surface` creates vertices without fields; the DEM hook `_film_forces_fn(particles, dim_)` also had the wrong arity (`dem_step` calls `external_forces_fn(ps)`) |
| `liquid_bridge_equilibrium/Case_1_...py` | 1 | 1 s | `AttributeError: 'VertexCube' object has no attribute 'vd'` in `stress_force -> _dual_area_vector_3d_p_ij` (the volumetric stress operator on the surface mesh) |
| `liquid_bridge_equilibrium/Case_5_...py` | 1 | 2 s | the same, on the extracted boundary surface |
| `liquid_bridge_equilibrium/Case_2`, `_3`, `_4` | not probed at HEAD | | they import Case 1 as `base` and run their own gradient-descent loops (no library integrator); run after the fix: section 2.5 |
| `liquid_bridge_approach/case_1_finial.py` (and 2 to 5, `run_fig6_approach_case12.py`) | 1 | 1 s | `ImportError: cannot import name 'multiphase_sparse_compressible_eos_pressure_correction' from '...pr33_operators'` |
| `Hagen_Poiseuile_2D_Eulerian/...py` | 0 | 50 s | runs; point-wise profile error at x = L/2 max 2.22e-1, L2 1.31e-1 (the runner's own Eulerian comparison, not a campaign metric) |
| `Hagen_Poiseuile_equilibrium/...py` | 0 | 32 s | residual table 1e-17 to 1e-13 as pinned |
| `dynamic_caprise_tube/Dynamic_caprise_3D_tube.py` | 1 | 0 s | `ModuleNotFoundError: No module named 'polyscope'` (module-level import), hand-rolled loop |
| `oscillating_droplet_p_ref/scripts/sphere_fheron_eos_projection_benchmark.py --max-subdivision 1 ...` | 0 | ~2 min | in place (the scripts locate the repo root from their own path, a scratch copy raises `StopIteration`); outputs under the git-ignored `scripts/out/` |
| `oscillating_droplet_p_ref/scripts/sphere_fheron_dynamic_integrator_p_ref.py --case-id 5 --t-final 2e-4` | 0 | 933 s CPU | `a2=0.03395631, theory=0.03240990, abs err 1.546e-03` (its own Rayleigh comparison) |
| `oscillating_droplet_p_ref/scripts/run_github_twophase_oscillating_preview.py` | 1 | 1 s | `ImportError: cannot import name 'dual_cell_area_2d' from 'hyperct.ddg._dual_cell'`: the script's own stub of that module shadows the real one |
| other `oscillating_droplet_p_ref/scripts/*` (render_*, plot_*, patch_*, crop_*) | not run | | figure / pptx post-processing of recorded outputs, not solver runners |
| `oscillating_droplet/diagnose_split_methods.py` | not run | | stale 2026-04 driver (laneW section 9) |

## 2. Fixes (what changed and where)

### 2.1 Library: `phases='film'` (`ddgclib/methods/`)

- `_axes.py`: option `film` on the `phases` axis (`experimental`, dims
  `(3,)`, anchor `operators/surface_tension.py:surface_tension_acceleration`).
- `_config.py`: validation (film needs `connectivity in ('frozen',
  'custom')`; `redistribute_mass`, `pressure_flux`, `viscous_flux`,
  `density_diffusion`, `contact_line` refused; the multi-only fields are
  refused by the existing single-phase rule); `dudt_fn(HC, gamma=,
  damping=0.0)` returns `partial(surface_tension_acceleration, gamma,
  damping, dim, HC)` (`body_force=` wraps it as for the bulk models);
  `gamma=` on a bulk config and `mu=` / `mps=` / `pressure_model=` on a
  film raise.  `retopologize_fn` and `integrator_kwargs` needed no change
  (`frozen -> False`, `custom -> the callable`).
- `_presets.py`: the seven presets of the verdict.
- `METHODS.md` regenerated (`python -m ddgclib.methods --update`), case
  matrix rows by hand (section 4 of METHODS.md).
- Tests: `test_methods.py::TestValidation` (eight film combinations that
  must raise), `test_case_runners_smoke.py::TestFilmForce` (the builder
  is the canonical partial; the catenoid axial force).

### 2.2 cube2droplet (6 runners, 1 setup)

- `src/_setup.py`: `setup_cube_to_droplet(..., methods=None)`; `dudt_fn
  = methods.dudt_fn(HC, mps=mps, pressure_model=meos)`, `retopo_fn =
  methods.retopologize_fn(mps=mps)` (the hand-bound
  `partial(multiphase_dudt_i, ...)` and the `**kwargs` closure around
  `_retopologize_multiphase` are gone); `methods=None` builds
  `SolverMethods(dim, 'multi', connectivity='delaunay',
  redistribute_mass=redistribute_mass)`, the configuration it bound before;
  `params['atm_verts']` exposes the AtmosphericPressureBC targets.
- `cube_to_droplet_2D.py`: import path fixed (`cases_dynamic.cube2droplet`),
  `PRESETS['cube_to_droplet_2D']` through `methods.integrate`, `--n-steps`,
  `--n-refine`, `--no-anim`, `--arm {base, bare, dual_only}` (A/B through
  the preset: `bare` = `preset.replace(remap=None)`, the historic setup
  configuration; `dual_only` = the dual-only preset), the integrated
  Laplace jump printed and recorded, `results/methods.json`.
- **Preset default (2026-10-07, continuation): `cube_to_droplet_2D` carries
  `remap='conservative'`.**  The historic configuration (per-step Delaunay
  + per-phase redistribution, no remap) loses the droplet (section 4.1:
  circularity 0 and no phase-1 bulk sub-volume by t = 0.4 s of the 1 s
  run; in the first 20 steps the phase-1 bulk count goes 41 -> 25 -> 21
  -> 13 while the droplet pressure climbs to 249 Pa against the 0.886 Pa
  Laplace jump).  The old behaviour is a defect, so by rule 5 the default
  moved; it stays reachable as the registered value `remap=None` (runner
  arm `bare`, no new registry option needed) and the smoke test
  `test_2d_arms` runs it.
- `cube_to_droplet_3D.py`: the same on `cube_to_droplet_3D`, which also
  carries `remap='conservative'` (section 4.1: the no-remap arm loses the
  bulk by step 1000, dual_only does not relax), arms `{base, bare,
  dual_only}`.
- `cube_to_droplet_2D_adaptive.py`: `base.replace(connectivity='adaptive',
  remesh_kwargs=...)`, `results/methods_adaptive.json`.
- `cube_to_droplet_2D_mass_redist.py`: `_MODES` = preset /
  `.replace(redistribute_mass=False)` / `cube_to_droplet_2D_dual_only`;
  the case-local `dual_only_retopo_multiphase` closure is gone (the
  library `dual_only` path re-identifies the interface from the phase
  labels where the closure kept it frozen: section 5).
- `cube_to_droplet_2D_bc_comparison.py`: `build_case()` is the setup call
  on `cube_to_droplet_2D_dual_only` (its 60-line copy of the setup and
  the closure are gone); `results/methods_bc_<mode>.json`.
- `diagnostic_no_retopo.py`: retired (section 3).

### 2.3 bc_demo

`bc_demo.py`: the advection loop is `PRESETS['bc_demo_2D'].integrate(HC,
bV, dudt_zero, ...)` (library `euler`, `connectivity='frozen'`, bV = the
two walls so the inlet and outlet columns advect as before; the BCs run
inside the integrator; the per-step report and the frames move to the
callback); `--n-steps`; `results/methods.json`.  v1, v2, parametric:
retired.

### 2.4 liquid_bridge_cfd_dem

- `liquid_bridge_cfd_dem_case.py`: `PRESETS['liquid_bridge_cfd_dem_3D']`,
  `dudt_fn = methods.dudt_fn(HC_film, gamma=gamma, damping=film_damping)`,
  `methods.integrate(..., custom=retopo_fn)` per DEM step (the custom
  retopology is rebound with the new particle centres as before);
  `_film_forces_fn(ps)` adds the film force to `p.force` (the
  `dem_step` contract); `--n-steps`; `results/methods.json`.
- `src/_fluid_film.py::retopologize_surface`: a vertex the edge split
  created gets `u = 0` (it had no fields, which was the step-0 crash).

### 2.5 liquid_bridge_equilibrium Cases 1 and 5 (2, 3, 4 untouched)

- Case 1: `METHODS = PRESETS['liquid_bridge_film_3D']`;
  `_surface_stress_force` is the film force (surface tension on the
  interface vertices; the volumetric `stress_force` term it added was
  identically zero at mu = 0, p = 0 and crashed on the missing `v.vd`);
  `compute_dynamic_case` runs `METHODS.integrate` with
  `METHODS.dudt_fn(HC, gamma=GAMMA, damping=damping)` and writes
  `out/Case_1/methods.json`; the "multiphase stress" row is NaN (the
  volumetric multiphase operator has no dual cells on a film; the column
  is kept so the historic tables and figures keep their shape);
  `--refinements`.
- Case 5: `METHODS = PRESETS['liquid_bridge_volume_3D']` for the
  volumetric hold (`methods.replace(workers=...)` keeps the
  `DDGCLIB_CASE5_WORKERS` override; `dudt_fn(HC, mu=0.0)` is the historic
  `partial(stress_acceleration, dim=3, mu=0.0, HC=HC)`, `dudt_i is
  stress_acceleration`); the surface rows use the Case 1 film force; the
  multiphase row is NaN; `--refinements`; `out/Case_5/methods.json`.
- Cases 2, 3, 4: their dynamic part is a case-local gradient descent with
  a spring toward `x_exact` (case physics, no library integrator, no
  preset by rule 7).  They import Case 1 and run after the fix
  (continuation, 2026-10-07, scratch copies, three solver processes on
  the 4 cores): Case 2 and Case 3 exit 0 after 22 min wall each (rows and
  figures in 8 min, the hold-time sampling at refinement 5 is the rest);
  DDG capillary error 0 / 1.0e-15 / 1.9e-15 / 3.2e-15 % (Case 2) and
  4.0e-10 / 1.5e-10 / 1.8e-11 / 1.1e-11 % (Case 3) at refinements 2 / 3 /
  4 / 5, FD 1e-10 in both.  Case 4: section 4.5.  No smoke test: they have
  no short mode and are the owner's benchmark scripts (reproduce: section 7).
- Case 5, continuation: two further defects fixed. `main` declared
  `global REFINEMENTS` after using the name (SyntaxError at import); the
  reproduced figure sized the live series by the reference's four
  boundary counts (shape mismatch for any subset of refinements; the same
  fix the first implementer made in Case 1).  And the inert short-circuit
  (static force norm 0) returned before `record_methods`, so no
  `methods.json` existed at any refinement where the hold is inert (every
  one: p = 0, mu = 0, u = 0); the configuration is now recorded with
  `n_steps 0, inert true`.

### 2.6 Smoke tests (`ddgclib/tests/test_case_runners_smoke.py`)

Each revived runner runs from a scratch copy of its directory under
`tmp_path` (code only; `fig/`, `results/`, `out/` land there) through
`subprocess` with its short arguments, must exit 0 and must write a
`methods.json` whose `config` round-trips to the preset.  Fast: 2D
(20 steps, refinement 4 = the shipped resolution, 7 s wall; at
refinement 3 the droplet is 5 bulk vertices and even the remap arm
erodes it to nothing within two rebuilds, so a coarser smoke cannot
assert a droplet), 2D `bare` arm (5 steps, refinement 4; asserts that
the preset carries `remap='conservative'`), 3D (5 steps, refinement 2),
adaptive (3 steps), `bc_demo.py` (10 steps), cfd_dem (3 DEM steps);
slow: mass_redist + bc_comparison (7 short runs), Case 1 (refinement
2), Case 5 (refinement 0).  In-process: the film builder, the catenoid
axial force (refinements 2 and 3).

## 3. Retired (header docstring added, nothing deleted; the owner may delete)

| file(s) | reason | replacement |
|---|---|---|
| `cube_flow/cube_flow_1D.py`, `_2D.py`, `_3D.py` | duplicate of the Hagen-Poiseuille family with a mock force; stalls after step 0 (M7) | `Hagen_Poiseuile/Hagen_Poiseuile_2D.py` (`hagen_poiseuille_2D`), `Hagen_Poiseuile_3D/...` (`hagen_poiseuille_3D`) |
| `bc_demo/bc_demo_v1.py`, `bc_demo_parametric_study.py` | case-local BC loops duplicating the library classes, same `fig/bc_demo.gif` | `bc_demo/bc_demo.py` (`bc_demo_2D`) |
| `bc_demo/bc_demo_v2.py` | imports four helpers that never existed in any commit | `bc_demo/bc_demo.py` |
| `cube2droplet/diagnostic_no_retopo.py` | second copy of the setup + frozen-interface dual-only closure; import path never existed | `cube_to_droplet_2D_mass_redist.py` (`no_retopo` arm), `cube_to_droplet_2D.py --arm dual_only` |
| `liquid_bridge_approach/case_[1-5]_finial.py`, `run_fig6_approach_case12.py` | the 15 000-line core imports a PR33 operator that exists only on the unmerged branch `songyideng/liquid-bridge-separation-12cases`; the dense `multiphase_compressible_eos_pressure_correction` in the tree does not take the sparse call's `mobility_matrix` / `stiffness_matrix` / `diagonal_regularization` / `pressure_delta_limit`; the loop is case physics (rule 7) | merge that branch or port the sparse operator into `pr33_operators.py` |
| `dynamic_caprise_tube/Dynamic_caprise_3D_tube.py` | hard `import polyscope`, hand-rolled loop | `capillary_rise/capillary_rise_2D.py` / `_3D.py` (laneI) and the dynCA runners |
| `oscillating_droplet/diagnose_split_methods.py` | stale driver binding the combination the registry refuses (laneW section 9) | `diagnose_a5_bisection.py --split-method` |
| `oscillating_droplet_p_ref/scripts/run_github_twophase_oscillating_preview.py` | its own `hyperct.ddg._dual_cell` stub shadows the real module | `oscillating_droplet/oscillating_droplet_3D.py` |

Not retired, not converted, listed honestly: the other
`oscillating_droplet_p_ref/scripts/*` are figure / benchmark scripts
outside the case pipeline (two of them run in place, section 1); they
pass `retopologize_fn=` by hand (laneW section 9) and get no preset.
`liquid_bridge_dem/liquid_bridge_case.py` is pure DEM (no fluid solver,
no preset).  `Hagen_Poiseuile_2D_Eulerian` and `_equilibrium` run as
before (their rows are unchanged).

## 4. Measurements

Every number below is from one fresh process on this machine
(deterministic since laneT; the 3D runs are quoted as single
realisations, no perturbation sweep was run).

### 4.1 cube2droplet against the Laplace jump

Reference: `gamma (dim - 1) / R_eq` with `R_eq` the radius of the circle
(sphere) of the square's (cube's) area (volume): 2D 0.8862 Pa, 3D 1.6120
Pa.  Measured: `integrated_phase_pressure_jump(HC, 1, 0)` (bulk,
volume-weighted; `ddgclib.analytical`), the nodal mean jump the runner
always printed in brackets.

| run | configuration | n_steps / dt / refine | circularity (sphericity) final | integrated jump [Pa] | error vs Laplace |
|---|---|---|---|---|---|
| 2D full, shipped (arm `base`) | `cube_to_droplet_2D` (Delaunay + redistribution + `remap='conservative'`) | 5000 / 2e-4 / 4 | 0.9079 (max 0.9079) | +0.9481 (nodal +0.95) | +6.99 % |
| 2D full, arm `bare` (the historic setup) | `.replace(remap=None)` | 5000 / 2e-4 / 4 | 0.0000 (max 0.7211; 0 from t = 0.4 s) | n/a: phase 1 has no bulk sub-volume (nodal -0.00) | droplet lost |
| 2D full, arm `dual_only` | `cube_to_droplet_2D_dual_only` | 5000 / 2e-4 / 4 | 0.8669 | -2.2761 (nodal -1.02) | -357 % |
| 3D full, arm `bare` (Delaunay + redistribution, no remap) | `cube_to_droplet_3D.replace(remap=None)` | 2000 / 5e-5 / 2 | 0.0000 (0.8955 at step 500, 0 by step 1000) | n/a: phase 1 has no bulk sub-volume | droplet lost |
| 3D full, shipped (arm `base`) | `cube_to_droplet_3D` (Delaunay + redistribution + `remap='conservative'`) | 2000 / 5e-5 / 2 | 0.8677 (0.8661 / 0.8663 / 0.8668 at steps 500 / 1000 / 1500) | +1.7710 (nodal +1.42 at step 1500) | +9.86 % (802 s wall) |
| 3D full, arm `dual_only` | `.replace(connectivity='dual_only', redistribute_mass=False, remap=None)` | 2000 / 5e-5 / 2 | 0.6174 (0.6124 at t = 0: the cube does not relax) | +2.2763 | +41.2 % |
| 2D smoke, arm `bare` | `.replace(remap=None)` | 20 / 2e-4 / 3 | 0.7173 | +169.36 (nodal +174.16) | +1.9e4 % (the first acoustic transient of the square; not an equilibrium number) |
| 2D smoke, arm `base` | `cube_to_droplet_2D` | 20 / 2e-4 / 4 | 0.72 (max 0.7216) | +0.0210 | -97.6 % (t = 4e-3 s, the droplet has not started to relax; the smoke asserts that it exists) |

The 2D rows were run by the first implementer (2026-10-07 00:00 to 00:17,
`full_remap`, `full_base`, `full_dual` under the scratch directory; the
`base` of that time was the no-remap configuration and `remap` the
`.replace(remap='conservative')` arm, so they are the `bare` and `base`
rows of today's naming with identical SolverMethods objects); the 3D
`bare` and `dual_only` rows likewise (`full_3d`, `full3d_dual`); the 3D
`remap` row was re-run by the continuation (`r3d_remap`).  Per-step
probe of the first 20 steps at refinement 4 (`probe_remap.py` in the
scratch directory, counts of phase-1 bulk vertices / interface vertices):
`base` 41 / 40 at t = 0, 41 / 20 from step 0 on (the first Delaunay
rebuild halves the interface count with the bulk count unchanged; which
vertices lose the flag was not traced), droplet pressure 0.019 Pa at
step 19; `bare` 41 / 40 ->
25 / 32 (step 1) -> 21 / 16 (step 4) -> 13 / 12 (step 19), pressure 25
-> 249 Pa; `dual_only` 41 / 40 throughout, pressure -0.094 Pa.  The
erosion mechanism is the primal-subcomplex relabelling after a
reconnection (`assign_simplex_phases_from_vertices`: an all-interface
simplex falls to the lower phase id 0 and every bulk neighbour of such a
simplex becomes interface), which the remap's per-phase pressure
restoration keeps from running away but which still costs the 2D run 20
of its 40 interface vertices at the first rebuild.

A/B verdict: on this case the conservative remap is the only
reconnecting configuration that keeps the droplet (2D and 3D), so both
`cube_to_droplet` presets carry `remap='conservative'`; the historic
no-remap configuration stays reachable as `remap=None` (runner arm
`bare`, the `redist_bare` / `no_redist` modes of the mass-redistribution
demo).  The registry's laneE DO-NOT for the 3D multiphase remap (the
oscillating droplet, l2 1.873 against 0.248) does not transfer: here the
alternative loses the phase.  Remaining error of the shipped
configurations against Laplace: +6.99 % (2D, refinement 4) and +9.86 %
(3D, refinement 2); single realisations, no refinement study, the
droplets still creeping (2D max|u| 1.1e-3 m/s at step 4000, 3D 1.9e-3 at
step 1500).  `dual_only` is measured worse in both dimensions (2D -357 %
with the interface intact; 3D +41 % with the cube not relaxing).

### 4.2 Catenoid (Case 1) against the minimal surface

Analytical: a catenoid has H = 0, so the film force is zero at every
vertex and the integrated axial force of the interior vertices is zero.
Mesh: Case 1's `_build_live_endres_catenoid` (a hyperct `Complex(2)`
refined `r` times and mapped onto `a = 1`, `v in [-1.5, 1.5]`; vertex
degrees {4, 8}).  Film force = `liquid_bridge_film_3D.dudt_fn(HC,
gamma=0.0728)` times `v.m` (Heron dual area).

| refinement | vertices (rims) | integrated axial force / (2 pi gamma a) | max per-vertex 2 H a | mean per-vertex 2 H a | h_min .. h_max |
|---|---|---|---|---|---|
| 2 | 36 (8) | -5.7e-17 | 1.114 | 0.513 | 0.806 .. 1.264 |
| 3 | 136 (16) | -7.7e-17 | 1.215 | 0.556 | 0.382 .. 0.797 |
| 4 | 528 (32) | +7.0e-17 | 1.246 | 0.557 | 0.188 .. 0.451 |
| 5 | 2080 (64) | (not run) | 1.255 | 0.553 | 0.094 .. 0.240 |

The axial sum vanishes to round-off at every refinement: it is the
z-symmetry of the mesh, not consistency.  The LOCAL curvature does not
converge (mean 0.55 / a independent of h; the legacy
`b_curvatures_hn_ij_c_ij` operator the historic benchmark used gives the
identical numbers), so the film force on this mesh is a spurious force
of order 0.5 gamma / a per unit area, and the damped hold of Case 1
relaxes the mesh away from the catenoid rather than holding it.
Fix round 1 (section 8) measured what that local force IS: on every
interior vertex `dudt_fn(v) * v.m` equals the discrete area gradient
`-gamma dA/dx_i` (central differences of the one-ring triangle areas,
eps 1e-6) to 3.5e-10 (refinement 2) and 1.0e-9 (refinement 3) of the
largest vertex force.  The operator is therefore exactly the surface
tension force of the discrete surface; what does not vanish is the area
gradient of this particular grid, i.e. the degree-{4, 8} mesh of the
catenoid is not a discrete minimal surface at any refinement (the
centre vertices of the cells sit off the stationary configuration).
The smoke test asserts this area-gradient identity (section 8).  Cause
by the numbers: the hyperct refinement of the parameter square yields the
degree-{4, 8} "centre vertex per cell" pattern (section 5), on which the
cotangent mean-curvature vector is not consistent.  Measured DO-NOT: do
not read Case 1's local force rows as a curvature error of the operator
on a triangulated catenoid; the catenoid needs a proper triangulation
(the `sphere` / parametric builders of `ddgclib.geometry` with an
icosphere-type grid) before its hold means anything.  Not done in this
lane (the mesh builder is the case's reference, "Endres 2024", and
replacing it changes the benchmark).

Case 1 at `--refinements 2 3` after the fix (continuation, scratch
copy): static film-force rows 2.9e-15 / 2.7e-15 %, hold rows (100 steps
of 2e-6 s, damping 20) 1.8e-15 / 8.4e-15 %, DDG capillary 5.6e-15 /
3.5e-16 %, FD 1.3e-10 / 7.4e-10 %; `out/Case_1/methods.json` records
`liquid_bridge_film_3D` with `refinement 3, dt 2e-6, n_steps 100,
damping 20` (the last refinement overwrites).  Case 5 at `--refinements
0 1` (81 / 289 vertices): static and hold-time film rows 3.4e-15 /
6.3e-15 %, DDG capillary 5.6e-15 %, FD 5.4e-10 / 3.4e-11 %; the hold is
inert at both (`methods.json`: `n_steps 0, inert true`).  All of these
are the z-symmetric cancellation of section 4.2, not a curvature
validation.

### 4.5 Case 4 (continuation probe)

Case 4 (`Case_4_off_equilibrium_relaxation_particle_particle_bridge_benchmark.py`,
no arguments, scratch copy, one of two solver processes): exit 0 after
1190 s wall.  Its own table (the relaxed bridge against the exact
catenoid; case-local gradient descent on the Case 1 film force):

| refinement | n_boundary | n_total | DDG capillary error [%] | FD capillary error [%] |
|---|---|---|---|---|
| 2 | 8 | 36 | 1.136e-03 | 3.787e-04 |
| 3 | 16 | 136 | 2.741e-04 | 9.137e-05 |
| 4 | 32 | 528 | 1.216e-06 | 4.047e-07 |
| 5 | 64 | 2080 | 5.163e-07 | 1.720e-07 |

Recorded as the runner's own metric (section 4.2 applies to what the
local force means on this mesh); not a campaign pin.

### 4.3 Film of the CFD-DEM case against the sphere

The film meshes come from `ddgclib.geometry.sphere(R, refinement=2,
phi_range=(0.01, 0.55 pi))`: 72 vertices, degrees 4, 5 and 8 (a
latitude-longitude grid with a centre vertex per cell), rims
`HC.boundary()`-tagged: none (the open fans are not detected on this
connectivity), 48 particle-attached vertices frozen by the setup's angle
test.  Heron mean curvature against `2 / R_film` (`R_film` = 1.01 mm):
relative error -0.957 to +0.153, mean -0.36; by vertex degree: 4 ->
+0.001 to +0.153, 8 -> -0.957 to -0.703, 5 (pole, rim) -> -0.65 / -0.22.
So the capillary force the case reports (`stokes_integral`, 243.96 uN
after 3 DEM steps) has no analytical standing on this mesh; the case is
a smoke (it runs, it records its configuration), not a validation.  The
fix is a triangulated film (icosphere), a change of the case's mesh
builder, not done here.

### 4.4 bc_demo kinematics

10 steps of dt 0.05 at U 0.5 on the `bc_demo_2D` preset: 10 vertices
remaining (the integrator-driven run; the loop version at 120 steps
ended with 23).  Deterministic, no analytical content (prescribed
advection); recorded for the smoke only.

## 5. Known limits, not done

- The 3D cube2droplet keeps its droplet only on the remap arm (now the
  preset); the registry's laneE DO-NOT for the 3D remap (oscillating
  droplet l2 1.873 against 0.248) does not transfer to this case, where
  the alternative is no droplet at all.  +9.86 % at refinement 2 (189
  vertices, 26 interface) is a single realisation, no refinement study.
  The square's / cube's first Delaunay rebuild drops half of the
  interface flags (2D: 40 -> 20 with the bulk count unchanged).
- `cube_to_droplet_2D_mass_redist.py` `no_retopo` arm and the BC
  comparison run on the library `dual_only` path, which re-identifies
  the interface from the phase labels at every refresh; the retired
  closure kept the t = 0 interface frozen.  The arms are therefore not
  bit-identical to the April runs (which had been unrunnable since
  2026-09-25 anyway); no pin existed.
- The adaptive demo reports 31 eroded interface edges in 3 steps at
  refinement 3 (its own warning); it runs and records, nothing more.
- Case 1 / Case 5 "multiphase stress" column is NaN (section 2.5).
- The catenoid and film meshes are the degree-{4, 8} hyperct grids
  (sections 4.2, 4.3): the film force is consistent on neither; the
  revived runners are smokes with recorded configurations until their
  mesh builders triangulate.
- `liquid_bridge_approach`: retired, not revived (section 3).
- `Hagen_Poiseuile_2D_Eulerian` still compares point-wise (rule 8 of the
  brief); it is Eulerian validation code and was not touched.
- The `liquid_bridge_cfd_dem` full horizon (100 000 DEM steps x 10 fluid
  sub-steps) was not run; 3 DEM steps take 2 s, so the full run is of
  the order of 20 hours and belongs on the owner's machine (reproduce
  command below).
- hyperct untouched; its suite was not re-run (baseline applies).

## 6. Tests

Continuation, 2026-10-07, this machine (Python 3.13.16, numpy 2.5.3,
scipy 1.18.1, matplotlib 3.11.2):

- Smoke tests of this lane: `test_case_runners_smoke.py -m ""`: 12 passed
  (fast 9: 2D at refinement 4, 2D `bare` arm, 3D, adaptive, bc_demo,
  cfd_dem, film builder, catenoid axial force x2, 23 s; slow 3:
  mass_redist + bc_comparison 11 s, Case 1 refinement 2 about 5 s, Case 5
  refinement 0 10 s).
- Fast suite (`-m "not slow"`): 1306 passed, 1 failed, 11 skipped, 2
  xfailed in 368 s (run while two solver processes shared the cores).
  The failure is baseline (a), `test_single_phase_remap.py::TestBoxDecayWithEOS::test_remap_matches_fixed_connectivity`,
  0.4399043770574842 against 0.43990437705748414, unchanged.
- Slow battery (`-m slow`): 41 passed, 1 failed, 1 xfailed in 2828 s
  (47 min: the first 20 min shared the 4 cores with the Case 4 probe and
  the fast suite's tail; the battery is about 9 min alone).  The failure
  is baseline (b), `test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column`,
  peak 0.15269571220608844 against the pin 0.15305813130485327, unchanged.
- Fix round 1 (section 8), after every edit of that round, one process
  at a time alongside at most two solver runs: smoke file (`-m ""`) 12
  passed in 50 s; fast suite 1306 passed, 1 failed, 11 skipped, 2
  xfailed in 378 s (the failure is baseline (a), same two values);
  `test_methods.py` alone after the registry evidence edit and the
  `METHODS.md` regeneration: 94 passed in 6.2 s; slow battery: see the
  line below.
- Fix round 1 slow battery (`-m slow`): 41 passed, 1 failed, 1 xfailed
  in 2911 s (48.5 min with the first 7 min alongside the fast suite;
  the battery alone is not 9 min on this machine either: 43 tests,
  single process, 98 % CPU throughout).  The failure is baseline (b),
  `test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column`,
  peak 0.15269571220608844 against the pin 0.15305813130485327, unchanged.
- hyperct untouched; its suite was not re-run (baseline (c) applies as
  fixed by laneG).
- `-m slow` prints `PytestUnknownMarkWarning` for the new file as it does
  for every other slow-marked test file (no `pytest.ini` registers the
  mark); the deselection works.
- Untracked before this lane and left alone (rule 3):
  `cases_dynamic/capillary_rise/results/capillary_rise_static_2D_flat/`
  and `.../capillary_rise_static_3D_young_laplace/` (created 2026-10-06
  22:14 / 22:36, before the lane started at 23:43).

## 7. Reproduce

```
cd /home/user/ddgclib
PYTHONPATH=/home/user/ddgclib /usr/bin/python -m pytest ddgclib/tests/test_case_runners_smoke.py -q -p no:cacheprovider -m ""      # all smokes incl. slow
# full cube2droplet runs (from a scratch copy laid out as <root>/cases_dynamic/cube2droplet,
# since the runners insert ../.. into sys.path; 13 to 18 min each in 2D, 13 min in 3D):
python cube_to_droplet_2D.py --arm base --no-anim      # 5000 steps, refinement 4 (preset: remap)
python cube_to_droplet_2D.py --arm bare --no-anim      # the historic no-remap configuration
python cube_to_droplet_2D.py --arm dual_only --no-anim
python cube_to_droplet_3D.py --arm base --no-anim      # 2000 steps, refinement 2
python cube_to_droplet_3D.py --arm bare --no-anim
python cube_to_droplet_3D.py --arm dual_only --no-anim
# catenoid numbers of section 4.2 (in process):
python - <<'EOF'
import numpy as np, sys; sys.path.insert(0, '.')
from cases_dynamic.liquid_bridge_equilibrium import Case_1_equilibrium_particle_particle_bridge_benchmark as c1
from ddgclib._curvatures_heron import hndA_i
for r in (2, 3, 4):
    HC, bV = c1._build_live_endres_catenoid(r); c1._prepare_surface_benchmark_state(HC, bV)
    fn = c1.METHODS.dudt_fn(HC, gamma=c1.GAMMA)
    Fz = sum(float((fn(v) * v.m)[2]) for v in HC.V if v not in bV)
    e = [np.linalg.norm(hndA_i(v)[0]) / hndA_i(v)[1] for v in HC.V if v not in bV]
    print(r, Fz / (2 * np.pi * c1.GAMMA), max(e), np.mean(e))
EOF
# liquid bridges: Case 1 full (refinements 2 3 4 5), Case 5 full (0 1 2 3), cfd_dem full (no --n-steps)
# Cases 2, 3, 4 (no arguments; ~22 min each on this machine): from a scratch copy of the directory
MPLBACKEND=Agg PYTHONPATH=/home/user/ddgclib python Case_2_perturbed_mesh_equilibrium_particle_particle_bridge_benchmark.py
```

## 8. Fix round 1 (after independent review, 2026-10-07 01:49 to 02:40)

The reviewer could not finish the suites and re-measurements within its
budget, so every accuracy number stood unverified; it also listed six
non-blocking items.  Everything below ran in fresh processes on this
machine (same environment as section 6), from scratch copies under
`.../scratchpad/laneX/fix1/<arm><dim>/cases_dynamic/cube2droplet/`
(`run.log`, `results/methods.json`, `fig/` there), three solver
processes at a time.

### 8.1 Re-measurement of every cube2droplet A/B arm

| run (fresh process) | configuration | result | section 4.1 value | wall |
|---|---|---|---|---|
| 2D `--arm base` | `cube_to_droplet_2D` | `+0.9481 Pa (error +6.987 %)`, circularity 0.9079 (step 2000: 0.8163, dP +0.92; step 4000: 0.8836, dP +0.94), `dp_integrated_final 0.9481434251199523` | identical | 23 min (3 processes sharing) |
| 2D `--arm bare` | `.replace(remap=None)` | circularity 0.0000 at step 2000 (t = 0.4002) and 4000, final 0.0000, max 0.7211, `n/a (phase 1 has no bulk sub-volume)`, rc 0 | identical (rc 0 now: the runner catches the `ValueError` of `integrated_phase_pressure_jump`, the first implementer's run had rc 1) | 15 min |
| 2D `--arm dual_only` | `cube_to_droplet_2D_dual_only` | `-2.2761 Pa (error -356.830 %)`, circularity 0.8669 (0.7805 at step 2000) | identical | 7 min |
| 3D `--arm base` | `cube_to_droplet_3D` | `+1.7710 Pa (error +9.864 %)`, sphericity 0.6124 -> 0.8677 (0.8661 / 0.8663 / 0.8668 at steps 500 / 1000 / 1500) | identical | 17 min |
| 3D `--arm bare` | `.replace(remap=None)` | sphericity 0.8955 at step 500 (dP +81.16, max u 0.141 m/s), 0.0000 from step 1000, `n/a (phase 1 has no bulk sub-volume)` | identical | 11 min |
| 3D `--arm dual_only` | `cube_to_droplet_3D_dual_only` (the named preset of this round, section 8.3) | `+2.2763 Pa (error +41.209 %)`, sphericity 0.6174 (0.6130 / 0.6141 / 0.6156), `dp_integrated_final 2.2762826625290185` | identical | 10 min |

Every digit the lane log, the preset notes and the `_axes.py` evidence
quote reproduces; the runs are deterministic (laneT) so these are single
fresh processes, not ranges.

### 8.2 Catenoid block of section 7, fresh process

`2 -5.688663793254058e-17 1.1141627361681328 0.5128927554203873`,
`3 -7.665133141582247e-17 1.214687874318703 0.5556595672147279`,
`4 7.031643541565476e-17 1.2463741963146564 0.5568594537025569`
(refinement, F_z / (2 pi gamma a), max 2Ha, mean 2Ha): section 4.2
reproduced digit for digit.  New measurement: the per-vertex film force
is the discrete area gradient to 3.5e-10 / 1.0e-9 of the largest force
at refinements 2 / 3 (section 4.2, last paragraph).

### 8.3 Non-blocking review items, resolved

1. `debugging_plan.md`: the entry named the arms `{base, remap,
   dual_only}` (2D) and `{base, dual_only}` (3D) from the first
   implementer's naming; now `{base, bare, dual_only}` for both, as the
   runners register them.
2. `ddgclib/methods/_presets.py`: the section comment above
   `cube_to_droplet_2D` still described the historic no-remap setup; it
   now states the conservative remap and the `bare` arm.
3. `cube_to_droplet_2D.py` docstring: the dead `cases_dynamic/Cube2droplet`
   usage path replaced by the real path and the flags.
   `setup_cube_to_droplet(redistribute_mass=...)`: the parameter defaults
   to `None`; passing it together with a `methods=` whose
   `redistribute_mass` differs raises `ValueError` (it was silently
   ignored); `None` plus `methods=None` builds the historic `True`.  No
   caller passes both (the mass-redistribution demo uses
   `preset.replace`).
4. `test_case_runners_smoke.py::TestFilmForce::test_catenoid_axial_force_is_zero`:
   the tautological third assertion (`_stress_capillary_force_error ==
   _surface_tension_capillary_force_error`, both the same operator since
   the lane) is replaced by the per-vertex area-gradient identity of
   section 8.2 (`d_max / g_max < 1e-7`, measured 3.5e-10 / 1.0e-9), which
   has local content and does not pass by z-symmetry.  The first two
   assertions stay.
5. 3D `dual_only` arm: a named preset `cube_to_droplet_3D_dual_only`
   (identical `SolverMethods` to the former `.replace(connectivity=
   'dual_only', redistribute_mass=False, remap=None)`; the re-run of
   8.1 confirms the number), symmetric with 2D, listed in the preset
   table of METHODS.md and the 3D case-matrix row.
6. The git-ignored p_ref probe outputs (`scripts/out/`, 25 files, and
   `scripts/.mplconfig/fontlist-v3.11.0.json`) are the scripts' own
   convention and were left for the owner (listed in section 3's
   "owner may delete" sense; `git status` does not show them).

### 8.4 Tests of the fix round

See section 6 (updated): the smoke file, the fast suite, the slow
battery and the `test_methods.py` drift subset were all re-run after
the edits above.  Fix round 2 (section 9.4) re-ran all of them once
more in fresh processes.

## 9. Fix round 2 (after the second independent review, 2026-10-07 03:20)

The second reviewer found no defect; its only blocking item was that
its own re-measurements (the fast suite at 13 %, the 2D and 3D base
arms mid-run, the other four arms, the catenoid block and the slow
battery not started) were cut off by the harness, so every accuracy
number of the lane and the state of the suites stood unverified by the
reviewer.  This round re-runs all of it in fresh processes on this
machine, from fresh scratch copies under
`.../scratchpad/laneX/fix2/<arm><dim>/cases_dynamic/cube2droplet/`
(three queued slots: suites + catenoid block; the three 2D arms; the
three 3D arms; `slotA.sh`, `slotB.sh`, `slotC.sh` there), and resolves
the non-blocking items.

### 9.1 Re-measurement of every cube2droplet A/B arm (third fresh process each)

| run (fresh process, scratch copy) | configuration | result | `results/methods.json` | wall (3 processes sharing) |
|---|---|---|---|---|
| 2D `--arm base` | `cube_to_droplet_2D` | `+0.9481 Pa (error +6.987 %)`, circularity 0.9079 (step 2000: 0.8163, dP +0.92; step 4000: 0.8836, dP +0.94) | `dp_integrated_final 0.9481434251199523`, `circularity_final 0.9079317268458665` | 23 min 56 s |
| 2D `--arm bare` | `.replace(remap=None)` | circularity 0.0000 at step 2000 (t = 0.4002) and 4000, final 0.0000, max 0.7211, `n/a (phase 1 has no bulk sub-volume)`, rc 0 | `NaN`, `0.0` | 16 min 13 s |
| 2D `--arm dual_only` | `cube_to_droplet_2D_dual_only` | `-2.2761 Pa (error -356.830 %)`, circularity 0.8669 (0.7805 at step 2000, dP -1.94) | `-2.2760928392544915`, `0.8669471064270978` | 14 min 57 s |
| 3D `--arm base` | `cube_to_droplet_3D` | `+1.7710 Pa (error +9.864 %)`, sphericity 0.6124 -> 0.8677 (0.8661 / 0.8663 / 0.8668 at steps 500 / 1000 / 1500) | `1.7710037145333803`, `sphericity_final 0.867680652831543` | 13 min 49 s |
| 3D `--arm bare` | `.replace(remap=None)` | sphericity 0.8955 at step 500 (dP +81.16, max u 0.141 m/s), 0.0000 from step 1000, `n/a (phase 1 has no bulk sub-volume)` | `NaN`, `0.0` | 7 min 31 s |
| 3D `--arm dual_only` | `cube_to_droplet_3D_dual_only` | `+2.2763 Pa (error +41.209 %)`, sphericity 0.6174 (0.6130 / 0.6141 / 0.6156) | `2.2762826625290185`, `0.6174193326634658` | 6 min 52 s |

Every value of the `results/*.json` of this round is byte-identical to
the fix-round-1 file (`fix1/<arm>/.../results/`), and to sections 4.1
and 8.1: three fresh processes (continuation, fix 1, fix 2) agree to
the last digit, so these runs are deterministic on this machine and the
numbers are stated as single values, not ranges (protocol rule 8).

### 9.2 Catenoid block of section 7, third fresh process

`2 -5.688663793254058e-17 1.1141627361681328 0.5128927554203873`,
`3 -7.665133141582247e-17 1.214687874318703 0.5556595672147279`,
`4 7.031643541565476e-17 1.2463741963146564 0.5568594537025569`:
identical to section 8.2 (`fix2/catenoid.log`).  The area-gradient
identity (3.5e-10 / 1.0e-9) is asserted by the smoke file, which
passed (9.4).

### 9.3 Non-blocking review items, resolved before the runs finished

1. Rule 6: the git-ignored p_ref probe outputs the first implementer's
   probe created (`oscillating_droplet_p_ref/scripts/out/`,
   `scripts/.mplconfig/`, `scripts/__pycache__/`, all dated 2026-10-06
   23:53 to 23:54, after the lane start) are removed; the tree now
   carries nothing the lane created outside its files.  Section 8.3
   item 6 is superseded.
2. `cube2droplet/diagnostic_no_retopo.py` retired header: the
   fixed-connectivity question is answered by the `no_retopo` mode of
   `cube_to_droplet_2D_mass_redist.py`, which is
   `PRESETS['cube_to_droplet_2D_dual_only']` (the header said
   `.replace(connectivity='frozen')`, which that runner never builds).
3. `cube_to_droplet_2D_mass_redist.py` docstring: "four-way" and "all
   four modes" (it said 3-way / three cases with four modes).
4. METHODS.md case matrix: the nine em-dash placeholder cells of the
   lane's own rows are now `none` (the pre-existing rows keep the
   table's em-dash convention; the generated headings of sections 1 to 3
   are written by `python -m ddgclib.methods --update` and not touched).
5. `test_methods.py::TestValidation`: the eight film refusal rows also
   pass against the HEAD registry (phases='film' was unregistered, so
   `SolverMethods` raised anyway).  Added
   `test_film_preset_validates_and_binds_the_film_force`: the
   `liquid_bridge_film_3D` preset validates (`dim 3, film, frozen`,
   `status_of('phases') == 'experimental'`), `phases='film'` with
   `connectivity='custom'` validates, `dudt_fn(HC, gamma=0.0728,
   damping=20.0)` is `partial(surface_tension_acceleration, gamma=,
   damping=, dim=3, HC=)`, and the three refusals (`gamma=` missing,
   `mu=` on a film, `gamma=` on a bulk preset) raise.  This fails at
   HEAD (KeyError on the preset).
6. The `slow` mark warning: not addressed (a `pytest.ini` is the
   owner's call; every slow-marked file carries it).

### 9.4 Tests of fix round 2 (logs under `fix2/`)

| suite | command (from the repo root, `PYTHONPATH=/home/user/ddgclib`) | result |
|---|---|---|
| fast, before the edits of 9.3 | `python -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider` | 1306 passed, 1 failed, 11 skipped, 43 deselected, 2 xfailed in 383.92 s (`fast.log`) |
| fast, after every edit of this round | same | 1307 passed, 1 failed, 11 skipped, 43 deselected, 2 xfailed in 387.73 s (`fast_after_edits.log`); the +1 is `test_film_preset_validates_and_binds_the_film_force` |
| the one fast failure | | baseline (a) `test_single_phase_remap.py::TestBoxDecayWithEOS::test_remap_matches_fixed_connectivity`, `0.4399043770574842 == 0.43990437705748414` (one ulp), unchanged |
| smoke file | `python -m pytest ddgclib/tests/test_case_runners_smoke.py -q -p no:cacheprovider -m ""` | 12 passed in 50.42 s (`smoke.log`) |
| `test_methods.py` after the new test | `python -m pytest ddgclib/tests/test_methods.py -q -p no:cacheprovider` | 95 passed in 7.05 s (94 before) |
| slow battery | `python -m pytest ddgclib/tests -q -m slow -p no:cacheprovider` | 41 passed, 1 failed, 1321 deselected, 1 xfailed in 2967.57 s (49 min 27 s, the first 25 min alongside two cube2droplet arms; `slow.log`) |
| the one slow failure | | baseline (b) `test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column`, `0.15269571220608844 == 0.15305813130485327 +- 1.5e-07`, unchanged |
| hyperct | not touched (`git status --short` of /home/user/hyperct empty); baseline applies | |
| `python -m ddgclib.methods --update METHODS.md` | | "already current": the hand-edited matrix survives a regeneration |

No pin moved, no tolerance loosened, no assertion deleted.  Both trees
after this round: ddgclib `git status --short` lists the lane's files
only (plus the two untracked `capillary_rise/results/` directories that
predate the lane); the p_ref probe outputs are gone; hyperct clean.
