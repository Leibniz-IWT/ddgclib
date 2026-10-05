# laneW: every setup builds its force and its retopology from `SolverMethods`, so a recorded axis is an applied axis

Date 2026-10-05. Case-plumbing lane (ddgclib `cases_dynamic/`, the
registry and presets of `ddgclib/methods/`, tests; hyperct untouched).
Repo state at the start: ddgclib `79ecaad`, hyperct `48d163a`, both on
master, env `ddg`, cwd repo root, Python
`/home/endres/anaconda3/envs/ddg/bin/python`. Evidence in: laneM log
(the droplet setup bound to `methods.dudt_fn`, the other three setups
not), laneO log known limit 1 (`area_orientation` on those presets
recorded but not applied), METHODS.md section 5. Scratch (logs and the
HEAD export only):
`/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad/laneW/`.
Everything a number below needs lives in the repository: the shipped
runners, `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py`,
`cases_dynamic/oscillating_droplet/diagnose_box_shift.py`,
`cases_dynamic/diagnose_determinism.py` and the new
`cases_dynamic/diagnose_setup_bindings.py` (section 7).

Fix round 1 (2026-10-05, after independent review): section 10. The
review found one silent method change (`diagnose_a5_step1_diff.py` had
lost the setup default `redistribute_mass=True`); it is restored and the
A/B is the new `diagnose_setup_bindings.py a5step1` sub-command. The
registry sentence about hand-bound partials is qualified, the
`diagnose_area_orientation.py` docstring is current, `--out` of the
bindings driver is required (no default inside the tree), and the nine
determinism cases the review did not re-run were re-run on both trees.

## Verdict up front

1. **No setup, shipped runner or maintained driver builds a force
   partial, a retopology closure or integrator kwargs by hand any more**
   (the inventory of section 2; the four unconverted files, among them
   the stale `diagnose_split_methods.py`, which still binds
   `multiphase_dudt_i` and `_retopologize_multiphase`, are listed in
   section 9). Every multiphase setup
   (`setup_oscillating_droplet`, `setup_dam_break_multiphase`,
   `setup_electrolysis_bubble`, the Fritz runner's
   `setup_fritz_dynamics`, `setup_shearing_plate_droplet`) and the
   single-phase `setup_dam_break_single_phase` take `methods=`, read
   `split_method` / `redistribute_mass` (and, for the shearing plate,
   `periodic_axes`) from it, build `dudt_fn` with `methods.dudt_fn`
   (gravity through the library's `body_force=` wrapper) and
   `retopo_fn` with `methods.retopologize_fn`. So `curvature_path`,
   `area_orientation` and every other force or retopology axis of a
   preset is applied wherever it is recorded; laneM's open item and
   laneO's known limit 1 are closed.
2. **Bit-identity at the defaults, measured, not assumed.** Every fast
   and slow pin; the full 2D and 3D droplet runs (every key of both
   baselines); lane L's and lane B's smoke digests (dam break 2D shipped
   run `952d4544676ca366`, electrolysis 2D 6330 steps `a8301121c7bf44ab`,
   electrolysis 3D 300 steps `6bdbc9c542ffd6fc`, 2D droplet 100 steps
   `443c01a815acc6c3`, the shearing-plate setup / three-step / short
   window numbers); and A/B digests against an export of the HEAD tree
   for fifteen determinism cases, the Fritz smoke, the two no-air dam
   breaks, the three stale droplet runners and the outlet-BC scripts.
   One documented exception: the NO_RETOPO arm of
   `oscillating_droplet_2D_mass_redist.py` (section 3.6).
3. **Without a config the setups build one.** `methods=None` builds
   `SolverMethods(dim, phases='multi', split_method=..., redistribute_mass=...)`
   (per-step Delaunay, no remap; `connectivity='periodic'` for the
   shearing plate; the single-phase default for the no-air dam break)
   from the explicit kwargs, whose partials are the ones the setups used
   to build by hand (same callable, same keywords; proven in
   `test_methods.py::TestSetupsBuildFromMethods`). The explicit kwargs
   therefore keep working for the tests and drivers that pass them, and
   no setup carries a second code path.
4. **Case physics stays in the case, as the thinnest wrapper.** Gravity
   is the library's `body_force=` (generic: dam break, both phases;
   electrolysis; already the hydrostatic column). The electrolysis NaN
   guard (zero acceleration for a vertex with degenerate mass or a
   non-finite stress, a workaround for NaN dual volumes at 3D domain
   corners) is the case-local `electrolysis_dudt(methods, HC, mps, meos,
   g)` around `methods.dudt_fn(..., body_force=g)`, shared with the
   Fritz runner (one copy instead of two). The gas injection stays in
   the runner callback and the wall clamp in `bc_set` (boundary
   conditions, not force). The shearing-plate periodic closure is gone
   from the case: `methods.retopologize_fn` (the library
   `retopologize_multiphase_periodic` of 2026-09-25) is also what the
   setup applies once at setup; the closure survives verbatim in the
   `test_methods.py` probe so the bit-identity proof stays.
5. **Runners pass the preset and nothing else.** The redundant
   `split_method=` / `redistribute_mass=` kwargs beside `methods=` (lane
   M review item) are gone from the droplet runners; the dam-break,
   electrolysis and shearing-plate runners pass `methods=` instead of
   `redistribute_mass=`; the two no-air dam-break runners,
   `mesh_convergence_2D.py`, `oscillating_droplet_2D_adaptive.py`,
   `oscillating_droplet_2D_mass_redist.py`, `diagnose_dual_only.py`,
   the outlet-BC scripts and the equilibrium test run through a config
   (new presets `dam_break_2D_no_air` / `_3D_no_air`); every diagnose
   driver that passed the two kwargs passes `methods=`;
   `diagnose_a5_bisection.py` builds a config from its CLI kwargs
   instead of a hand-written `euler(...)` call. `methods.json` is the
   truth for every runner that writes one, and the dam-break records
   carry the gravity vector instead of a string.
6. The HP2D post-processing `_retopologize` call named in the brief is
   not in the tree (lane H's runner rewrite, commit `2f47210`, removed
   it); no public refresh method was added, because nothing needs one.
   The one remaining private call in a setup is the dam-break setup's
   one-shot geometry rebuild before the hydrostatic preload (section 9).

## 1. What changed and where

Library (`ddgclib/`):
- `methods/_presets.py`: `dam_break_2D_no_air`, `dam_break_3D_no_air`
  (single phase, symplectic Euler, per-step Delaunay, hull-frozen
  through `boundary_filter`; laneK's measured-unstable configuration,
  recorded as such; no pin).
- `methods/_axes.py`: evidence / control text of `phases='multi'`,
  `curvature_path`, `area_orientation` (where the value lands now).
- `methods/_config.py`: module docstring (the runner recipe passes
  `methods=` to the setup). No code change: `dudt_fn(body_force=)` and
  `retopologize_fn` already existed.

Cases (`cases_dynamic/`), setups:
- `oscillating_droplet/src/_setup.py`: `methods=None` builds the config
  from the explicit kwargs; the hand-built `partial(multiphase_dudt_i,
  ...)` and `partial(_retopologize_multiphase, ...)` are gone (imports
  with them).
- `dam_break/src/_setup.py`: both setups take `methods=`;
  `setup_dam_break_multiphase` reads `split_method` from it (passed to
  its three `mps.refresh` calls; the default value is what ran before),
  `dudt_fn = methods.dudt_fn(HC, mps=mps, pressure_model=meos,
  body_force=g_vec)`, `retopo_fn = methods.retopologize_fn(mps=mps)`;
  `setup_dam_break_single_phase`: `methods.dudt_fn(HC, mu=mu_eff,
  pressure_model=eos_liq, body_force=g_vec)`. The gravity closures and
  the two hand-built partials are gone.
- `electrolysis_bubble/src/_setup.py`: new module-level
  `electrolysis_dudt(methods, HC, mps, meos, gravity_vec)` (the guard,
  with `.stress_fn` / `.body_force` exposed); `setup_electrolysis_bubble(
  methods=)` reads `split_method` from the config (it hard-coded
  `'neighbour_count'`), `retopo_fn = methods.retopologize_fn(mps=mps)`.
- `electrolysis_bubble/electrolysis_bubble_fritz_2D.py`:
  `setup_fritz_dynamics(HC, bV, mps, meta, methods=None)` uses
  `electrolysis_dudt` and `methods.retopologize_fn`; `run_short_dynamics`
  passes the preset to it.
- `shearing_plate_droplet/src/_setup.py`: `methods=` (must be
  `connectivity='periodic'`; `periodic_axes`, `split_method`,
  `redistribute_mass` from it), `retopo_fn = methods.retopologize_fn(
  mps=mps, domain_bounds=...)` applied once at setup, `dudt_fn =
  methods.dudt_fn(...)`; `_make_periodic_multiphase_retopo` removed.

Cases, runners and drivers (every change is `methods=` into the setup,
or a config in place of a hand-written integrator call):
`oscillating_droplet_2D.py`, `_3D.py`, `static_droplet_2D.py` (the
redundant kwargs dropped), `mesh_convergence_2D.py`,
`oscillating_droplet_2D_adaptive.py` (`run_one_mode(methods=)`, the
arms are `oscillating_droplet_2D_bare_delaunay` and its
`.replace(connectivity='adaptive', remesh_kwargs=...)`; the JSON record
carries `methods.to_dict()`), `oscillating_droplet_2D_mass_redist.py`
(`_MODES`), `diagnose_dual_only.py`, `diagnose_a5_bisection.py`
(`run_a5a` and `run_a5b` without `methods` build one),
`diagnose_a5_step1_diff.py`, `diagnose_3d_edge_area_source.py`,
`diagnose_3d_remap_afterfix.py`, `diagnose_box_shift.py`;
`dam_break_2D.py`, `_3D.py` (`extra['body_force']` is the vector),
`dam_break_2D_no_air.py`, `_3D_no_air.py` (preset, `methods.integrate(
..., boundary_filter=)`, `record_methods`); `electrolysis_bubble_2D.py`,
`_3D.py`; `shearing_plate_droplet_2D.py`, `_3D.py`, `_run_short_2D.py`,
`_run_short_3D.py`; `Hagen_Poiseuile/diagnose_frozen_set.py`,
`Hagen_Poiseuile/test_outlet_old_bc.py`, `test_outlet_new_bc.py`
(`SolverMethods(dim=2, workers=1)`), `Hagen_Poiseuile_equilibrium/
test_equilibrium.py` (`SolverMethods(dim=2, integrator=
'euler_velocity_only')`), `diagnose_determinism.py`. New:
`cases_dynamic/diagnose_setup_bindings.py` (section 7).

Tests (`ddgclib/tests/`):
- `test_methods.py`: new `TestSetupsBuildFromMethods` (5 tests: per
  setup, the partial behind the wrappers has the function and the
  keyword set of the historic hand-built partial, the retopology partial
  the historic keywords, the body-force wrapper reproduces the historic
  gravity closure to the bit on a vertex, a non-default force axis on
  the config is bound into the partial, the guard of the electrolysis
  closure returns zero for a degenerate mass, the shearing setup refuses
  a non-periodic config, the Fritz wiring matches its preset);
  `test_electrolysis_presets_match_setup_partials` compares against the
  historic partial spelled out (it compared the setup's partial with
  itself otherwise); the shearing probe builds the removed closure
  itself and passes `methods=` to the setup.
- `test_case_dam_break.py`: `_build_2d` passes `methods=METHODS`.

Docs: this log, `debugging_plan.md` (status entry), `DEVELOPMENT.md`,
`METHODS.md` (sections 1, 3 regenerated, 4 rows, 5), `CLAUDE.md`
(registry paragraph: setups take `methods=`).

## 2. Inventory (task 1)

Every call site in `cases_dynamic/` (not `capillary_rise*`) that built
`partial(multiphase_dudt_i | dudt_i | stress_acceleration, ...)`, passed
`split_method` / `redistribute_mass` / `curvature_path` explicitly, or
built retopology kwargs by hand, and what happened to it:

| file | what it built | now |
|---|---|---|
| `oscillating_droplet/src/_setup.py` | fallback `partial(multiphase_dudt_i)` + `partial(_retopologize_multiphase)` when `methods` was None | config built from the kwargs, `methods.dudt_fn` / `retopologize_fn` always |
| `dam_break/src/_setup.py` (both setups) | `partial(multiphase_dudt_i)` + gravity closure + `partial(_retopologize_multiphase, redistribute_mass)`; `partial(stress_acceleration)` + gravity closure | `methods=`, `methods.dudt_fn(body_force=)`, `methods.retopologize_fn` |
| `electrolysis_bubble/src/_setup.py` | `partial(multiphase_dudt_i)` + gravity + NaN-guard closure, `partial(_retopologize_multiphase, split_method='neighbour_count', redistribute_mass)` | `electrolysis_dudt` around `methods.dudt_fn(body_force=)`, `methods.retopologize_fn` |
| `electrolysis_bubble/electrolysis_bubble_fritz_2D.py` (`setup_fritz_dynamics`) | the same two, duplicated | the shared `electrolysis_dudt`, `methods.retopologize_fn`, `methods=` from `run_short_dynamics` |
| `shearing_plate_droplet/src/_setup.py` | `partial(multiphase_dudt_i)`, `_make_periodic_multiphase_retopo` closure (applied once at setup and returned) | `methods.dudt_fn`, `methods.retopologize_fn(mps, domain_bounds)`; closure removed (kept in the test probe) |
| runners `oscillating_droplet_2D.py`, `_3D.py`, `static_droplet_2D.py` | `split_method=methods.split_method, redistribute_mass=...` beside (or instead of) `methods=` | `methods=` only |
| runners `dam_break_2D/3D.py`, `electrolysis_bubble_2D/3D.py`, `shearing_plate_droplet_2D/3D.py`, `_run_short_2D/3D.py` | `redistribute_mass=methods.redistribute_mass` | `methods=methods` |
| `dam_break_2D_no_air.py`, `_3D_no_air.py` | `symplectic_euler(..., boundary_filter=)` by hand, no preset, no `methods.json` | presets `dam_break_2D_no_air` / `_3D_no_air`, `methods.integrate`, `record_methods` |
| `oscillating_droplet/mesh_convergence_2D.py` | `symplectic_euler(retopologize_fn=retopo_fn)` on the setup default | `oscillating_droplet_2D_bare_delaunay`, `methods.integrate` |
| `oscillating_droplet/oscillating_droplet_2D_adaptive.py` | `symplectic_euler(retopologize_fn, remesh_mode, remesh_kwargs)` | `run_one_mode(methods=)`, arms = preset and `.replace(connectivity='adaptive', remesh_kwargs=)` |
| `oscillating_droplet/oscillating_droplet_2D_mass_redist.py` | two `partial(_retopologize_multiphase, redistribute_mass=...)` + a dual-only closure, `symplectic_euler` | `_MODES` (section 3.6), `methods.integrate` |
| `oscillating_droplet/diagnose_dual_only.py` | a `bare_dual_refresh` closure + `symplectic_euler` twice | `static_droplet_2D` and `oscillating_droplet_2D_bare_delaunay` through `methods.integrate` |
| `oscillating_droplet/diagnose_a5_bisection.py` | `run_a5a(split_method=)`; `run_a5b` without `methods`: `euler(retopologize_fn, remesh_mode, remesh_kwargs, displacement_eps)` | both build `SolverMethods` from the kwargs (the stencil is applied to the force as well as to the measurement) |
| `oscillating_droplet/diagnose_a5_step1_diff.py` | `split_method=args.split_method` (so the setup default `redistribute_mass=True` was bound) | `methods=SolverMethods(dim, phases='multi', split_method=, redistribute_mass=True)` (fix round 1: the first version of the lane dropped the `redistribute_mass=True`, since the field default is False, and changed the probe's single retopology call; section 10) |
| `diagnose_determinism.py`, `oscillating_droplet/diagnose_box_shift.py`, `diagnose_3d_edge_area_source.py` (droplet + dam break 3D), `diagnose_3d_remap_afterfix.py`, `Hagen_Poiseuile/diagnose_frozen_set.py` | `split_method=m.split_method, redistribute_mass=m.redistribute_mass` | `methods=m` |
| `Hagen_Poiseuile/test_outlet_old_bc.py`, `test_outlet_new_bc.py` | `partial(dudt_i, dim, mu, HC)`, `symplectic_euler(boundary_filter, workers=1)` | `SolverMethods(dim=2, workers=1).dudt_fn(HC, mu=mu)`, `methods.integrate(boundary_filter=)` |
| `Hagen_Poiseuile_equilibrium/test_equilibrium.py` | `partial(dudt_i, dim=2, mu, HC)`, `euler_velocity_only(...)` | `SolverMethods(dim=2, integrator='euler_velocity_only')` |
| `template/diagnose_single_phase_eos.py`, `Hydrostatic_column/*`, `Hagen_Poiseuile/src/_run.py`, `diagnose_curvature_path.py`, `diagnose_area_orientation.py`, `template/template.py` | already on `SolverMethods` | unchanged |
| NOT converted (section 9): `cube2droplet/*`, `liquid_bridge_cfd_dem/*`, `oscillating_droplet_p_ref/scripts/*`, `oscillating_droplet/diagnose_split_methods.py` | | |

Tests under `ddgclib/tests/` that pass `split_method=methods.split_method,
redistribute_mass=methods.redistribute_mass` to the droplet setup
(`test_case_oscillating_droplet.py`, `test_oscillation_score_3d.py`,
`test_edge_area_source.py`, the `droplet_2d` fixture of `test_methods.py`)
are left as they are: they exercise the explicit-kwarg path, which the
setup keeps (point 3 of the verdict), and they are not case files.

## 3. Measurements (every arm a preset or `preset.replace`, every number from one process; rule 8 of the protocol makes one process enough)

### 3.1 The pinned suites

Fast suite 1232 passed, 0 failed, 12 skipped, 2 xfailed (1227 before the
lane + 5 new tests; the first run of the lane showed the METHODS.md
drift test red because the registry text had changed and the file was
regenerated afterwards; green on the re-run). Slow battery 32 passed,
1 xfailed (238 s). hyperct untouched. After fix round 1 (section 10):
fast 1232 passed, 0 failed, 12 skipped, 2 xfailed, 130.7 s; slow 32
passed, 1 xfailed, 236.6 s (the METHODS.md drift test green on the
regenerated file).

### 3.2 The two droplet baselines (full runs through the shipped runners, `diagnose_box_shift.py fullrun --runner 2d|3d --no-anim --out <scratch>`)

| run | baseline | this lane |
|---|---|---|
| 2D (`oscillating_droplet_2D`, 3/3, 1839 steps): l2 / tail / linf / mass / frames / t_end | 0.17439096487276182 / 0.9998871416222597 / 0.323163334437819 / 1.4835808907017442e-14 / 207 / 0.11432433142499636 | the same in every key (`diff_baselines` delta 0 on every metric, no `SOLVER METHODS DIFFER`) |
| 3D (`oscillating_droplet_3D`, 2/2, 872 steps): l2 / tail / R_max_peak / mass / frames / t_end | 0.24811443136179492 / 0.0841737962816189 / 0.010790237633926668 / 4.8000942590870176e-14 / 111 / 0.16006110499010753 | the same in every one of the 25 keys |

### 3.3 The smoke digests of lanes L and B (`diagnose_frozen_set.py <case> --out <scratch>`; sha256 of sorted (x, u, m), both `frozen_set` arms)

| case, horizon | record (lane L 2026-10-01 / lane B 2026-10-05) | this lane (hull / membership) |
|---|---|---|
| `dam_break_2D` shipped (alpha_art 0.3, refinement 3), 1585 steps | `952d4544676ca366`, KE_max 1.036626e-03, 32 / 32 walls frozen (lane L) | `952d4544676ca366` / `952d4544676ca366`, KE_max 1.0366264395665179e-03, 145 vertices, mass 2.657834915771476, 32 walls frozen, 0 moved; 80.5 / 82.5 s |
| `electrolysis_bubble_2D`, 6330 steps | `a8301121c7bf44ab`, KE_max 4.2625019757386466e-02, 214 V, 16 walls, mass 0.060879178237541014 (lane B) | `a8301121c7bf44ab` / `a8301121c7bf44ab`, the same numbers; 377.6 / 373.9 s |
| `electrolysis_bubble_3D` (1/1), 300 steps | `6bdbc9c542ffd6fc`, KE_max 1.9456521671316604e-12, 95 V, 26 walls, mass 5.075113940820681e-04 (lane B) | `6bdbc9c542ffd6fc` / `6bdbc9c542ffd6fc`, the same numbers; 30.0 / 27.9 s |
| `oscillating_droplet_2D` (2/2), 100 steps of dt 2e-5 | `443c01a815acc6c3`, KE 2.8299870377596923e-10, 97 V, 16 walls, mass 9.933666299900592 (lane B) | `443c01a815acc6c3` / `443c01a815acc6c3`, the same numbers; 3.9 s |

Shearing plate (`diagnose_box_shift.py shearrun --out <scratch>`, both
`box_shift` arms, refinement 3/3):

| arm | setup digest / vertices / interface / plate vertices | three-step digest (vertices) | short window to t = 0.05 s (2596 steps, dt 1.926e-05): first interface loss (step, t, n_iface) / iface at the end / KE_max / max u over U_wall |
|---|---|---|---|
| `move_all` (shipped) | `855f8563c9733fb1` / 302 / 32 / 22 | `0ca6deaa1a7f766a` (302) | (183, 3.544148066386867e-03, 30) / 36 / 7.755663e-01 / 295.366 |
| `evict` | `b60215b8ee996368` / 287 / 31 / 10 | `285af826be7ff536` (287) | (78, 1.5216722676334947e-03, 30) / 28 / 2.688184e+00 / 347.639 |

Every entry is lane B's row of METHODS.md to the digit (lane B quoted
295 and 348 for the last column, 183 and 78 for the loss).

### 3.4 Determinism driver, this tree against the HEAD export (`diagnose_determinism.py run <case>`, one process each)

| case (preset), steps | digest, both trees | scalars |
|---|---|---|
| `dam_break2d` (`dam_break_2D`), 200 | `017fcfe2532cb8f7` (= lane T's sweep record) | KE 0.0019282963235294083 |
| `dam_break3d` (`dam_break_3D`), 5 | `12e7b83ef28be68a` (= lane T) | KE 0.0341749026085199 |
| `electrolysis2d` (`electrolysis_bubble_2D`, 1/2), 100 | `878f62b37958ead5` | KE 7.517571337744696e-10 |
| `electrolysis3d` (`electrolysis_bubble_3D`, 1/2), 30 | `66a594f1ef6d8353` | KE 8.648606339914853e-11 |
| `shearing2d` (`shearing_plate_droplet_2D`), 10 | `162d35ef40a6a4b2` | KE 8.844971423829144e-05 |
| `shearing3d` (`shearing_plate_droplet_3D`, 1/2), 5 | `1babe8c581caa132` | KE 6.629676723921086e-06 |
| `droplet2d` (`oscillating_droplet_2D`, 2/2), 100 | `066e53487662c7ba` | R_max 0.010500054457108332, KE 1.4245968791946972e-06 |
| `droplet2d_bare`, 100 | `c4a179e8b9aa9b71` | R_max 0.010396501819613024, KE 0.07209083415756856 |
| `droplet2d_dual_only`, 100 | `6e94a93ce0ab0023` | R_max 0.010495943906366493, KE 1.4529653896606493e-06 |
| `droplet2d_adaptive` (bare + `connectivity='adaptive'`), 60 | `4e30cfd65b1a2458` | R_max 0.010403364985995879, KE 1.1404221921463739e-05 |
| `droplet2d_workers` (`workers=2`), 40 | `18842d1b74a11489` | R_max 0.010505523004456379, KE 9.775532573087995e-07 |
| `droplet3d` (`oscillating_droplet_3D`, 2/2), 40 | `79f4d610c9df5c79` | R_max 0.01067437217313824, KE 1.657637002117674e-06 |
| `droplet3d_delaunay`, 40 | `f1c4bbc05aad5d02` | R_max 0.010347222859444952, KE 2.5013641196438643e-06 |
| `droplet3d_remap` (delaunay + `remap='conservative'`), 40 | `cc5b5e1bc552570d` | R_max 0.010533773930410053, KE 3.238923866658407e-08 |
| `droplet3d_frozen` (`connectivity='frozen'`), 40 | `0265f6067f345c4d` | R_max 0.01053268594949055, KE 8.097632277126949e-08 |

(The droplet, electrolysis and shearing digests of lane T's
`laneT-sweep-all.json` are from the lossy pre-laneB meshes and are not
comparable; the dam-break ones are, and match.)

### 3.5 The setups and the runners without a record, this tree against the HEAD export (`diagnose_setup_bindings.py <what> [--root <export>]`)

| what | this tree | HEAD export |
|---|---|---|
| `setups`: force partial behind the wrapper, retopology binding, acceleration digest of twelve interior vertices | droplet `multiphase_stress_acceleration(HC, dim, mps, pressure_model)` / `_retopologize_multiphase(mps, redistribute_mass, split_method)` / `2adf2d6a63bd2af7`; dam break the same partial / `_retopologize_multiphase(frozen_set, mps, redistribute_mass, retopo_remap, split_method)` (the preset) / `702c7c6aa9e1a5ec`; single-phase dam break `stress_acceleration(HC, dim, mu, pressure_model)` / `c961314d4d05714d`; electrolysis the multiphase partial / `_retopologize_multiphase(mps, redistribute_mass, split_method)` / `5649d29fd2c7968d`; shearing the multiphase partial / `retopologize_multiphase_periodic(domain_bounds, mps, periodic_axes, redistribute_mass, split_method)`, 90 vertices, setup state `767f7b58e19c2844` / `86cf958f89c51a90` | the same digests; the dam-break, single-phase and electrolysis forces were plain closures (`function`), the dam-break retopology lacked the explicit `split_method`, the shearing retopology was the closure `_retopo` |
| `fritz` (80 steps, 116 vertices) | `6a674f16ae632d8b`, t_final 3.1603372845846723e-06 | the same |
| `noair2d` (150 steps, 145 vertices) / `noair3d` (20 steps, 189 vertices) | `26e57ef151597ff9` / `b1800b37ddae43a3` | the same |
| `meshconv` (`run_single(1, 2, n_steps_max=30)`, 69 vertices) | `3278f19cbc4014ce`, R_max 0.010400833792239004 | the same |
| `adaptive` (2/2, 30 steps): Delaunay / adaptive arm | `0f0ab8d761fd7aad` (R_max 0.008845128671240522) / `d8f21859b53eef90` (0.010387963199648938) | the same |
| `massredist` (2/2, t_end 1e-3): NO_REDIST / REDIST / NO_RETOPO | `3ecb42de3cc5c299` (R_max 0.01048646501004752) / `15800af7dfe7ea07` (0.010483634685945249) / `a7b25b455e150148` (0.010486451036818589) | the same / the same / `8dc49d3307eedc85` (0.010486527419216681): section 3.6 |
| `a5step1` (fix round 1; `diagnose_a5_step1_diff.py`, epsilon 0, 2/2, sha256 of the probe's per-vertex JSON): 2D (16 interface vertices) / 3D (98) | `5deb668e5a8505be` (largest interface jump in abs F 1.5969882916078149e-09) / `fdf1b14c8388b5f1` (2.0397633556728416e-05) | the same, both JSON files byte-identical (`cmp`); section 10 |

Outlet-BC scripts (200 steps, L 15, refinement 1, both trees): the
same console output (`Final: 153 verts, t=2.0000, backward=0, buffer=1`
for the new BC, `backward=0` for the old one, the same reported
velocity columns). `Hagen_Poiseuile_equilibrium/test_equilibrium.py`:
18 passed. `diagnose_a5_bisection.run_a5b` without `methods` (the
config built from the CLI kwargs) against the HEAD tree's hand-written
`euler(...)` path: section 3.7.

### 3.6 The one arm that moved: NO_RETOPO of `oscillating_droplet_2D_mass_redist.py`

The closure the runner carried refreshed the duals, the per-phase split
and the EOS pressures on the frozen connectivity without `mps.refresh`.
The registry's `dual_only` (`_retopologize_multiphase(skip_triangulation=
True)`) runs the refresh (interface re-identification, per-phase arrays
re-derived with the masses kept) before the split and the pressures;
`dual_only_bare` (the static-droplet closure) refreshes no pressure at
all. Neither is the old closure, and a third value that is would be a
duplicate of `dual_only` up to the re-identification on a connectivity
that cannot change. The arm is expressed as
`oscillating_droplet_2D_dual_only.replace(redistribute_mass=False)` and
the difference is recorded: after 1e-3 s at refinement 2/2 R_max
0.010486451036818589 against 0.010486527419216681 (7.3e-6 relative),
mass 9.935228905597928 in both. The runner is a stale comparison (April
snapshots, no pin, METHODS.md section 4); the two Delaunay arms are
bit-identical.

### 3.7 `run_a5b` without a config

`run_a5b(dim=2, refinement_outer=2, refinement_droplet=2, n_steps=5,
redistribute_mass=False)` and `run_a5a(dim=2, 2, 2)` on this tree
(config built from the kwargs: `integrator='euler'`, per-step Delaunay)
and on the HEAD export (the hand-written `euler(...)` call): the
max |F| history is 0.005477298675717941 (step 0), 0.00503211417003938
(steps 1 to 5) and the A.5.a value 0.005477298675717941 on both trees,
to the bit (the floors of the test suite, which call
`run_a5b(methods=)`, are the pinned battery).

## 4. Decisions (task 3: wrappers; protocol rule 5)

- Gravity: generic, the library's `SolverMethods.dudt_fn(body_force=)`
  (it existed since 2026-09-25 and reproduces the dam-break closure to
  the bit, `test_dudt_partial_and_body_force`); used by the dam break
  (both phases) and the electrolysis case; the hydrostatic column used
  it already.
- NaN guard: case physics (a workaround for NaN dual volumes at 3D
  domain corners of the electrolysis box), kept as the thinnest
  case-local wrapper `electrolysis_dudt` in the electrolysis setup
  module and shared with the Fritz runner. Not a method axis, not in
  the library.
- Gas injection: a callback (mass source), unchanged. Wall clamp: a
  boundary condition, unchanged.
- The shearing-plate periodic closure: the library function
  `retopologize_multiphase_periodic` is the registered method
  (`connectivity='periodic'`, 2026-09-25); the case copy is deleted, the
  test keeps the historic closure for the proof.
- No preset changed. No pin moved. Two presets added for runners that
  had none (`dam_break_2D_no_air`, `_3D_no_air`), status recorded as
  unvalidated / laneK's measured-unstable configuration.

## 5. Pins and battery

Every pin bit-identical (sections 3.1 to 3.3): the fast suite, the slow
battery, the two droplet baselines in every key, lane L's and lane B's
smoke digests. Battery: ddgclib fast 1232 passed, 0 failed, 12 skipped,
2 xfailed; slow 32 passed, 1 xfailed; hyperct not touched
(`lane_diff.sh --stat` shows no hyperct change).

## 6. Measured DO-NOTs

- Do not pass `split_method=` / `redistribute_mass=` beside `methods=`
  to a setup: the config wins and the kwargs are ignored (documented in
  every setup docstring); a runner that passes both reads as if the
  kwargs mattered.
- Do not give the shearing-plate setup a non-periodic config: it raises
  (the side faces are periodic, not walls; a non-periodic retopology
  would freeze them).
- Do not read the NO_RETOPO arm of `_mass_redist.py` as the old closure
  (section 3.6).
- Do not route the dam-break setup's one-shot geometry rebuild through
  the preset's retopology without measuring: the preset runs the remap
  and the redistribution, the setup wants connectivity, boundary and
  duals only before it assigns the hydrostatic masses (section 9).

## 7. How to reproduce (repo root, ddg env; `S` = any scratch directory)

```bash
P=/home/endres/anaconda3/envs/ddg/bin/python
$P -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider            # 3.1, 1232 P
$P -m pytest ddgclib/tests -q -m slow -p no:cacheprovider                  # 3.1, 32 P
$P -m pytest ddgclib/tests/test_methods.py -q -p no:cacheprovider -k SetupsBuild
$P cases_dynamic/oscillating_droplet/diagnose_box_shift.py fullrun --runner 2d --no-anim --out $S   # 3.2, ~4 min
$P cases_dynamic/oscillating_droplet/diagnose_box_shift.py fullrun --runner 3d --no-anim --out $S   # 3.2, ~8 min
$P cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D --out $S/frozen       # 3.3, 2 x 82 s
$P cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_2D --out $S/frozen    # 2 x 375 s
$P cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_3D --steps 300 --out $S/frozen
$P cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py droplet_2D --steps 100 --out $S/frozen
$P cases_dynamic/oscillating_droplet/diagnose_box_shift.py shearrun --out $S/boxshift      # 3.3 shearing
for c in dam_break2d dam_break3d electrolysis2d electrolysis3d shearing2d shearing3d \
         droplet2d droplet2d_bare droplet2d_dual_only droplet2d_adaptive droplet2d_workers \
         droplet3d droplet3d_delaunay droplet3d_remap droplet3d_frozen; do
  $P cases_dynamic/diagnose_determinism.py run $c; done                                    # 3.4
$P cases_dynamic/diagnose_setup_bindings.py setups fritz noair2d noair3d meshconv adaptive massredist a5step1 --out $S/bind   # 3.5, 10
# the HEAD-tree side of 3.4 / 3.5 / 3.7 / 10: export the pre-lane commit and point --root (3.5) or
# run the exported driver copy (3.4) at it
mkdir -p $S/head && git archive 79ecaad ddgclib cases_dynamic | tar -x -C $S/head && ln -s $PWD/hyperct $S/head/hyperct
$P cases_dynamic/diagnose_setup_bindings.py setups fritz noair2d noair3d meshconv adaptive massredist a5step1 --root $S/head --out $S/bind_head
(cd $S/head && $P cases_dynamic/diagnose_determinism.py run dam_break2d)
PYTHONPATH=. $P cases_dynamic/Hagen_Poiseuile/test_outlet_new_bc.py                        # 3.5 outlet
$P -m pytest cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py -q
```

The pre-lane state is ddgclib `79ecaad` + hyperct `48d163a`; the
`diagnose_setup_bindings.py` sub-commands detect the old signatures, so
one script serves both trees. `--out` is required (a scratch directory;
a bare invocation no longer writes a `results_setup_bindings/` directory
into the tree).

## 8. Files

Repo, changed: `ddgclib/methods/_axes.py`, `_config.py`, `_presets.py`;
`ddgclib/tests/test_methods.py`, `test_case_dam_break.py`;
`cases_dynamic/oscillating_droplet/src/_setup.py`,
`oscillating_droplet_2D.py`, `_3D.py`, `static_droplet_2D.py`,
`mesh_convergence_2D.py`, `oscillating_droplet_2D_adaptive.py`,
`oscillating_droplet_2D_mass_redist.py`, `diagnose_dual_only.py`,
`diagnose_a5_bisection.py`, `diagnose_a5_step1_diff.py`,
`diagnose_3d_edge_area_source.py`, `diagnose_3d_remap_afterfix.py`,
`diagnose_box_shift.py`; `cases_dynamic/dam_break/src/_setup.py`,
`dam_break_2D.py`, `_3D.py`, `_2D_no_air.py`, `_3D_no_air.py`;
`cases_dynamic/electrolysis_bubble/src/_setup.py`,
`electrolysis_bubble_2D.py`, `_3D.py`, `electrolysis_bubble_fritz_2D.py`;
`cases_dynamic/shearing_plate_droplet/src/_setup.py`,
`shearing_plate_droplet_2D.py`, `_3D.py`, `_run_short_2D.py`,
`_run_short_3D.py`; `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py`,
`test_outlet_old_bc.py`, `test_outlet_new_bc.py`;
`cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py`;
`cases_dynamic/diagnose_determinism.py`; `METHODS.md`, `DEVELOPMENT.md`,
`debugging_plan.md`, `CLAUDE.md`.
Repo, new: `cases_dynamic/diagnose_setup_bindings.py`, this log.
Fix round 1 also touched `cases_dynamic/diagnose_area_orientation.py`
(docstring only).
Scratch (logs and the HEAD export only): `.../scratchpad/laneW/runs/logs/*.log`,
`.../scratchpad/laneW/head_tree/`, `.../scratchpad/laneW/fix1/` (fix
round 1: `logs/`, the pre-fix probe copy `prefix/`, `bind/`, `bind_head/`).
hyperct: untouched. The dynCA runners and `src/_setup_dynca.py` are
untouched and no library code changed, so their numbers cannot move.

## 9. Known limits, not done

- Not converted, with the reason: `cube2droplet/*` (every runner imports
  `cases_dynamic.Cube2droplet`, a path that does not exist; the case has
  been "broken import" in METHODS.md since 2026-09-25, and converting
  its setup without a runner that can exercise it proves nothing);
  `liquid_bridge_cfd_dem/*` (its force is the surface-tension film
  model `surface_tension_acceleration` and its retopology
  `retopologize_surface`, neither on the registry's force or
  connectivity axes; a `connectivity='custom'` config would describe
  only half of it); `oscillating_droplet_p_ref/scripts/*` (figure
  scripts; they build nothing themselves and consume the setup's
  partials, which are preset-built now, but still pass
  `retopologize_fn=retopo_fn` to `symplectic_euler` by hand);
  `oscillating_droplet/diagnose_split_methods.py` (stale 2026-04
  driver; its dual-only closure with a non-default `split_method` is the
  combination `dual_only_bare` + `split_method='exact'` that the registry
  refuses because the bare refresh ignores the split at runtime, which
  is exactly what that driver silently did; superseded by
  `diagnose_a5_bisection.py --split-method`).
- The dam-break setup still calls the private `_retopologize(HC, bV,
  dim)` once before the hydrostatic preload (a geometry rebuild: new
  connectivity, boundary, duals, no multiphase refresh). It is not a
  retopology closure or an integrator kwarg, and routing it through the
  preset's per-step retopology (which runs the conservative remap and
  the redistribution) would change the setup sequence; left as is and
  commented in the setup.
- The NO_RETOPO arm of `_mass_redist.py` is `dual_only` without
  redistribution, not the old closure (section 3.6; 7.3e-6 relative in
  R_max after 1e-3 s).
- The explicit-kwarg path of the setups (`methods=None`) is kept for
  the tests and drivers that use it; a caller that passes both a
  config and the kwargs gets the config.
- The two new presets describe runners that have never been validated
  (audit 2026-09-25: both no-air runs blow up); they record what runs,
  nothing more.

## 10. Fix round 1 (2026-10-05, after independent review)

Blocking finding, resolved at its cause. `diagnose_a5_step1_diff.py`
used to call `setup_oscillating_droplet(..., split_method=
args.split_method)`, so its single retopology call was bound with the
setup default `redistribute_mass=True`. The first version of this lane
replaced that with `SolverMethods(dim, phases='multi',
split_method=...)`, whose field default is `redistribute_mass=False`,
and its comment, the inventory row and the report claimed "no
redistribution: what this probe always ran". That was a silent method
change of the probe. The call now binds `redistribute_mass=True`
explicitly and the comment says why (the setup default the probe always
ran with; the field default differs). No pin involved (a diagnose
driver); no preset, no library code, no runner touched by the fix.

Measured with the new repo-resident A/B `diagnose_setup_bindings.py
a5step1 --out S` (the probe's `main()` in 2D and 3D at epsilon 0,
refinement 2/2; sha256 of the per-vertex JSON it writes, the number of
interface vertices before / after / diffed, and the largest interface
jump in abs F):

| arm | 2D | 3D |
|---|---|---|
| this tree, fixed (`redistribute_mass=True`) | `5deb668e5a8505be`, 16 / 16 / 16, 1.5969882916078149e-09 | `fdf1b14c8388b5f1`, 98 / 98 / 98, 2.0397633556728416e-05 |
| HEAD export (`split_method=` only) | `5deb668e5a8505be` (JSON byte-identical, `cmp`) | `fdf1b14c8388b5f1` (byte-identical) |
| the pre-fix probe (`redistribute_mass` unbound, i.e. False), copied to scratch and run against this tree | `3c248d27eb6ed5db`, 16 / 16 / 16, 2.7962652263124887e-09 | `dc42dd0674d020ce`, 98 / 98 / 98, 1.4673601641988932e-03 |

So the review's finding is reproduced (the pre-fix binding moves the
probe; in 3D the largest interface jump goes from 2.04e-5 to 1.47e-3,
the number the probe's April docstring quotes for a different library
state) and the fix restores bit-identity with the pre-lane call form.
The measurement is in the registry of this log's section 3.5 as well.

Non-blocking findings, addressed:

- The `phases='multi'` evidence in `ddgclib/methods/_axes.py` (and the
  regenerated METHODS.md row) said "no case file binds multiphase_dudt_i
  or _retopologize_multiphase by hand" without an exception; the stale,
  unconverted `diagnose_split_methods.py` does (section 9). The sentence
  now reads "no setup, shipped runner or maintained driver" and points
  at section 9; the verdict of this log and the `debugging_plan.md`
  entry say the same.
- The `diagnose_area_orientation.py` docstring (census paragraph) still
  said the multiphase setups build their own partial and ignore the
  preset's `area_orientation`; its census cases go through
  `diagnose_determinism.py`, which passes `methods=` to every setup
  since this lane, so `--orientation` reaches them now. Docstring
  corrected (no code change; the driver is not re-measured).
- `diagnose_setup_bindings.py --out` defaulted to
  `cases_dynamic/results_setup_bindings` inside the tree; it is required
  now and the docstring says it must be a scratch directory.
- The nine determinism cases the review did not re-run were re-run on
  this tree and on the HEAD export (`diagnose_determinism.py run`, one
  process each): `dam_break3d` `12e7b83ef28be68a` (KE
  0.0341749026085199), `electrolysis3d` `66a594f1ef6d8353`
  (8.648606339914853e-11), `shearing3d` `1babe8c581caa132`
  (6.629676723921086e-06), `droplet2d_bare` `c4a179e8b9aa9b71` (R_max
  0.010396501819613024), `droplet2d_adaptive` `4e30cfd65b1a2458`
  (0.010403364985995879), `droplet2d_workers` `18842d1b74a11489`
  (0.010505523004456379), `droplet3d_delaunay` `f1c4bbc05aad5d02`
  (0.010347222859444952), `droplet3d_remap` `cc5b5e1bc552570d`
  (0.010533773930410053), `droplet3d_frozen` `0265f6067f345c4d`
  (0.01053268594949055): every digest and scalar equal on both trees
  and equal to the table of section 3.4. With the review's six, all
  fifteen are now verified by a second party.
- Not changed: the empty-cell marker of the METHODS.md case matrix (the
  table's existing convention, not prose); the full 3D droplet run
  (about 8 minutes) was not re-run in this round, since the fix touches
  a diagnose driver only and no library code; the review confirmed the
  full 2D run.

Suites after the fix (one process each): fast and slow re-run, numbers
in section 3.1 (updated there); hyperct untouched.
