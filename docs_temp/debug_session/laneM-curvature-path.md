# laneM: the `curvature_path` axis measured on moving meshes; `'stokes'` removed as a duplicate, `'csf_dual'` measured worse

Date 2026-10-05. Library lane (ddgclib only, hyperct untouched). Repo state
at the start: ddgclib `386a308`, hyperct `48d163a`, both on master, env
`ddg`, cwd repo root, Python `/home/endres/anaconda3/envs/ddg/bin/python`.
Evidence in: audit 2026-09-25 F2, laneI (`laneI-3d-apex-cache.md`),
debugging_plan 2026-05-27 Probe 2. Scratch (logs only):
`/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad/laneM/`.
Everything a number below needs lives in the repository:
`cases_dynamic/oscillating_droplet/diagnose_curvature_path.py` and
`cases_dynamic/oscillating_droplet/results/laneM/`.

## Verdict up front

1. **`'stokes'` is gone.** The 3D conormal boundary integral of the
   interface over the barycentric dual cell (`integrated_hndA_i_interface`,
   Probe 2 of 2026-05-27; 2D aliased `'integrated'` outright) is the
   cotangent form, not a variant of it: on a piecewise-linear surface the
   integral of the conormal along the two dual segments inside a triangle
   is `n_T x (x_k - x_j) / 2` whatever the interior point of the path, i.e.
   the gradient of that triangle's area at `x_i`, and the cotangent stencil
   sums exactly those gradients. Measured equal to `'integrated'` to
   6.3e-16 on the static 3D mesh, 1.0e-15 at every force evaluation of 20
   moving steps under `dual_only` and under per-step Delaunay, 1.1e-15
   after a 2e-4 jitter and Delaunay rebuild, exactly 0 in 2D (same
   function); the full 2D run is bit-identical to the baseline and the
   full 3D run differs from the baseline by 4.7e-10 relative in l2 and
   1.8e-9 in the tail, inside the 1.3e-9 / 3.8e-9 that a 1e-15 shift of
   the free vertices moves the same run by (protocol rule 8, two seeds:
   that IS round-off). It cost 3.5x per vertex (0.39
   against 0.11 ms, it walked every interface triangle per vertex) and
   carried the coordinate-keyed cache of audit F2. The code path, the
   axis value, the cache deletion in `extract_interface` and the tests of
   the function are removed; the integral is kept as a measurement
   reference (`stokes_reference` in the diagnose driver) so the equality
   can be re-measured at any time, and the registry notes record what it
   was and why it went.
2. **One real difference was found and closed in the surviving path.**
   After laneI's 1e-3 jitter (10 % R0) and a Delaunay rebuild of the 2/2
   droplet the relabelled interface has 138 vertices and ONE edge with
   three (up to four) interface triangles. There the two stencils differed
   by 22 % of the force at the two vertices of that edge, because
   `hndA_i_interface` summed the first two apexes of an edge
   (`list(...)[:2]`) and the conormal integral every triangle. Every
   triangle counts now (one cotangent term per apex, in apex order): the
   stencil is the gradient of the discrete interface area on any triangle
   set, equal to the reference to 3.2e-16 there as well. Manifold edges
   have at most two apexes and keep their arithmetic and summation order,
   so every pin is bit-identical (verified: fast and slow suites, the 2D
   and 3D full runs, the floors).
3. **`'csf_dual'` is measured worse, status `measured-worse`.** Same
   magnitude as the default by construction (ratio 1.0 to 1e-16), direction
   off by up to 2.3 deg (2D static), 7.5 deg (2D perturbed), 6.9 / 10.0 deg
   (3D static / perturbed), 4.3 deg (2D) and 7.9 deg (3D) along 20 moving
   steps, 180 deg on the jittered non-manifold interface. Floors: 2D end
   2.2711535e-03 against 2.2716938e-03, 3D plateau 7.2744148e-05 against
   7.2741339e-05. Full runs through `preset.replace(curvature_path='csf_dual')`:
   2D l2 0.2041112129331814 / tail 1.0252046890273294 against
   0.17439096487276182 / 0.9998871416222597 (+17 %, and the KE grows in
   the second half), 3D l2 0.2784903890745515 / tail 0.08567541803104081
   against 0.24811443136179492 / 0.0841737962816189 (+12 % / +1.8 %). It
   costs 6x (2D) to 41x (3D) per vertex. Kept registered with its
   evidence (protocol rule 5).
4. **The axis reaches the force now.** `setup_oscillating_droplet(methods=)`
   builds `dudt_fn` through `methods.dudt_fn`, so `curvature_path` (and
   `area_orientation`) of a droplet preset is applied, not only recorded
   (laneO's open item for this case); for a preset at the default force
   axes the partial is the same callable with the same keywords, and the
   2D and 3D runners reproduce their baselines to the bit through it.
   `diagnose_a5_bisection.run_a5b(methods=)` takes the measurement stencil
   from the config as well.
5. **Guards.** `ddgclib/tests/test_curvature_path.py`: one moving-mesh
   test per surviving value and dimension (the preset-bound force after 3
   reconnecting steps + jitter + Delaunay rebuild equals a fresh evaluation
   with the apex cache dropped; in 3D a re-installed stale map is shown to
   be wrong), the removal of `'stokes'` (registry, config, force), and the
   area-gradient identity of the 3D stencil on a non-manifold fan (three
   triangles at one edge, against a finite-difference gradient of the
   total area, and the pre-laneM truncation shown to miss it).

## 1. What changed and where

Library (`ddgclib/`):
- `operators/multiphase_stress.py`: the `'stokes'` branch of
  `_interface_surface_tension` is gone; unknown values raise
  `ValueError("... expected 'integrated' or 'csf_dual'")`; docstring.
- `_curvatures_heron.py`: `integrated_hndA_i_interface` removed (129 lines,
  the only caller was the branch above); `hndA_i_interface` sums one
  cotangent term per apex of an edge instead of the first two (bit-identical
  wherever an edge has at most two interface triangles).
- `geometry/_interface_subcomplex.py`: the `del HC._interface_x_to_v` of
  laneI is gone with the cache; the apex-map invalidation is unchanged.
- `methods/_axes.py`: axis `curvature_path` has two values, `integrated`
  (validated) and `csf_dual` (measured-worse), notes record the removed
  value, the control line names the setup binding.
- No change to `methods/_config.py` (the validation is generic) or to the
  presets.

Case (`cases_dynamic/oscillating_droplet/`):
- `src/_setup.py`: `setup_oscillating_droplet(methods=None)`; with a
  config, `split_method` / `redistribute_mass` are taken from it and
  `dudt_fn = methods.dudt_fn(HC, mps=mps, pressure_model=meos)`.
- `oscillating_droplet_2D.py`, `_3D.py`: pass `methods=methods` to the
  setup (the only runner change; a stale comment on `'stokes'` updated).
- `diagnose_a5_bisection.py`: `run_a5b(methods=)` passes the config to the
  setup and measures with `methods.curvature_path`; the CLI choices of
  `--curvature-path` drop `'stokes'`.
- New `diagnose_curvature_path.py` (sub-commands `static`, `moving`,
  `nonmanifold`, `floors`, `dynamic`, see section 7).
- New `results/laneM/` (every JSON quoted below).

Tests (`ddgclib/tests/`):
- New `test_curvature_path.py` (8 tests).
- `test_interface_cache_invalidation.py`: the stokes-map test removed (its
  function is gone), docstring.
- `test_simplex_aware_curvature.py`: `TestIntegratedHndAIInterface`
  (4 tests of the removed function) removed with the import.
- `test_methods.py::test_broken_status_warns`: the synthetic broken option
  is patched onto `csf_dual` (there is no `stokes` value to patch).

Docs: this log, `debugging_plan.md` (status entry), `DEVELOPMENT.md`,
`METHODS.md` (regenerated sections 2 and 3, case matrix rows by hand).

## 2. Why the two 3D stencils are one operator

For an interface vertex `v_i` the surface-tension force is minus the
gradient of the interface energy, `F_i = -gamma dA/dx_i`. On a
piecewise-linear surface `A` is the sum of the triangle areas and for a
triangle `T = (i, j, k)`, `dA_T/dx_i = (1/2) n_T x (x_k - x_j)` (in plane,
normal to the opposite edge, pointing towards `x_i`), which is the
cotangent expression `(1/2)[cot(theta_k) (x_i - x_j) + cot(theta_j)
(x_i - x_k)]` restricted to that triangle. `hndA_i_interface` builds
exactly these terms edge by edge (one `HNdC_ijk` term per (edge, apex)
pair). The conormal form integrates `nu = n_T x t` along the path
`m_ij -> c -> m_ik` inside each triangle: `int nu dl = n_T x (m_ik - m_ij)
= (1/2) n_T x (x_k - x_j)`, independent of the interior point `c`
(barycentre, circumcentre or any other), and the outward orientation
of each segment with respect to the convex quadrilateral `(x_i, m_ij, c,
m_ik)` is exactly the sign of the area gradient. So the Stokes form IS
`-gamma dA/dx_i` triangle by triangle, as is the cotangent sum, and the
only thing that could make them differ is which triangles each visits.
That is what the non-manifold edge exposed (section 3.3): the conormal
integral visited all three triangles at the edge, the cotangent sum the
first two apexes. Both now visit all.

## 3. Measurements

Every arm is `PRESETS[...]` or `preset.replace(curvature_path=...)`, the
force bound by `methods.dudt_fn` through the setup. The 2D mesh is
refinement 3/3 (317 vertices, 32 interface), the 3D mesh 2/2 (475
vertices, 98 interface), `box_shift='move_all'`.

### 3.1 Static (`static`, `results/laneM/laneM_static.json`)

Surface-tension force of each stencil against `'integrated'` over the
interface vertices; "rel" is the largest `|F_arm - F_int|` over
`max |F_int|`; the magnitude ratio is 1.0 to 1e-16 for every arm (csf_dual
redirects the same magnitude). `stokes` is the library path of the
pre-lane state, `stokes_reference` the driver's copy of it; they are
identical to the bit.

| mesh | `stokes` rel / angle | `csf_dual` rel / angle max (median) | full-force floor A.5.a: integrated / stokes / csf_dual |
|---|---|---|---|
| 2D eps 0 (static floor preset) | 0 / 0 (same function) | 2.81e-2 / 2.26 deg (1.38) | 2.374856801157627e-03 / same / same |
| 2D eps 0.05 (run preset) | 0 / 8.5e-7 deg | 8.39e-2 / 7.53 deg (3.19) | 4.102855311416444e-03 / same / same |
| 3D eps 0 (static floor preset) | 6.27e-16 / 1.2e-6 deg | 7.55e-2 / 6.90 deg (1.66) | 6.015320113965291e-05 / ...288e-05 / ...292e-05 |
| 3D eps 0.05 (run preset) | 5.74e-16 / 1.5e-6 deg | 1.02e-1 / 9.97 deg (3.96) | 7.001397620489216e-05 / ...210e-05 / ...216e-05 |

The csf_dual floor equals the default's because the largest force sits
on a symmetry vertex where `S_inner` and `t_next - t_prev` are collinear.

Cost of one evaluation of the stencil over the interface (best of 5,
caches warm, ms per vertex): 2D integrated 0.015, stokes 0.015 (alias),
csf_dual 0.092 (eps 0) / 0.063 (eps 0.05); 3D integrated 0.113, stokes
0.398 (3.5x), stokes_reference 0.390, csf_dual 4.67 (41x: it evaluates
`edge_phase_area_fractions` and the dual face of every neighbour on top of
the cotangent magnitude).

### 3.2 Moving (`moving`, `results/laneM/laneM_moving.json`)

20 steps of the preset; at every force evaluation of an interface vertex
(right after the step's retopology, when the coordinate-keyed interface
sub-complex and the positions agree; the integrator callback runs after
the move and cannot be used) every stencil is evaluated on that vertex.
Largest deviation from `'integrated'` over all vertices and steps:

| run (20 steps) | interface moved (max, R0) | `stokes` rel / angle | `csf_dual` rel / angle | interface-triangle changes |
|---|---|---|---|---|
| 2D `oscillating_droplet_2D` (delaunay + remap) | 1.1e-4 | 0 / 1.2e-6 deg | 4.66e-2 / 4.25 deg | n/a |
| 2D `oscillating_droplet_2D_bare_delaunay` | 2.0e-3 | 0 / 1.2e-6 deg | 4.68e-2 / 4.27 deg | n/a |
| 3D `oscillating_droplet_3D` (dual_only) | 6.1e-3 | 1.02e-15 / 1.5e-6 deg | 8.13e-2 / 7.91 deg | 0 |
| 3D `oscillating_droplet_3D_delaunay` | 8.8e-3 | 9.09e-16 / 1.7e-6 deg | 5.90e-2 / 7.35 deg | 0 of 20 (laneI: 0 in 872) |

After the 3D delaunay run: laneI's 1e-3 jitter + the preset's Delaunay
rebuild changes the interface triangles (98 -> 138 interface vertices)
and the apex map is dropped by the refresh (as laneI's fix says); on that
mesh, with the library of the pre-lane state, `stokes` and the reference
differ from `'integrated'` by 2.44e-1 (21 deg) and `csf_dual` by 1.55
(179 deg). Section 3.3 locates the 2.44e-1; on the shipped library the
reference agrees there to 3.97e-16. The kept `laneM_static.json`,
`laneM_moving.json` and `laneM_floors.json` are from the shipped library
(no `stokes` arm, the reference only); the pre-lane runs of the same
three sub-commands, with the library `stokes` arm next to the reference
(identical to the bit) and the 2.44e-1 above, are kept under
`results/laneM/prelane/`.

### 3.3 Non-manifold edges (`nonmanifold`, `results/laneM/laneM_nonmanifold.json`)

Same jitter + rebuild on the 3D delaunay preset at 2/2, per vertex, split
by whether the vertex touches an interface edge with three or more
interface triangles. Measured on the pre-lane `hndA_i_interface` (scratch
probe, same numbers as the `moving` table) and on the fixed one (kept
JSON):

| jitter | interface vertices / triangles | edges with >= 3 triangles (max apexes) | vertices at them | rel diff, manifold vertices | rel diff, those vertices: before / after the fix |
|---|---|---|---|---|---|
| 1e-3 (10 % R0) | 138 / 274 | 1 (4) | 2 | 4.5e-16 | 2.21e-1 / 3.2e-16 |
| 2e-4 (2 % R0) | 98 / 192 | 0 | 0 | 1.1e-15 | n/a |

The whole gap sat on the one non-manifold edge, where the first-two-apex
truncation dropped triangles. The relabelled interface after a 10 % R0
jitter is not a configuration any shipped run reaches (0 interface-
triangle changes along the 2D and 3D runs, laneI), but merges, large
deformation and 3D remeshing can produce such edges, and the energy
gradient of the triangle set is the right force there.

### 3.4 Floors (`floors`, `results/laneM/laneM_floors.json`)

`run_a5b(methods=static_droplet_floor_{2D,3D}.replace(curvature_path=arm))`,
u = 0 every step, 2D 30 steps at 3/3, 3D 20 steps at 2/2:

| arm | 2D step 0 / step 1.. (pin 2.3748568e-03 / 2.2716938e-03) | 3D step 0 / plateau (pin 6.0153e-05 / 7.274134e-05) | mass drift |
|---|---|---|---|
| `integrated` | 2.374856801157627e-03 / 2.271693780620923e-03 | 6.015320113965291e-05 / 7.274133897024673e-05 | 2.7e-15 / 1.7e-15 |
| `stokes` (pre-lane library) | same to the bit | 6.015320113965288e-05 / 7.27413389702467e-05 | same |
| `csf_dual` | 2.374856801157627e-03 / 2.271153524599534e-03 (-2.4e-4) | 6.015320113965292e-05 / 7.274414790306941e-05 (+3.9e-5) | same |

### 3.5 Full runs (`dynamic`, `results/laneM/score_laneM_*.json` with the `methods` block, `methods_laneM_*.json`, `diags_laneM_*.json`)

2D: `oscillating_droplet_2D` (delaunay + conservative remap), refinement
3/3, dt 6.22e-5, 1839 steps, t_end 0.1143. 3D: `oscillating_droplet_3D`
(dual_only, fan cache), 2/2, dt 1.84e-4, 872 steps, t_end 0.1600. Wall
times are one process each on a 32-core machine, with one to four other
lane processes running at the same time (indicative; the per-vertex cost
in 3.1 is the clean measure).

| arm | 2D l2 / tail / mass / wall | 3D l2 / tail / mass / R_max peak / wall |
|---|---|---|
| pinned baseline (laneB) | 0.17439096487276182 / 0.9998871416222597 / 1.48e-14 | 0.24811443136179492 / 0.0841737962816189 / 4.80e-14 / 0.010790237633926668 |
| `integrated` through `setup(methods=)` | 0.17439096487276182 / 0.9998871416222597 / 1.48e-14 / 216.3 s (0.115 s per step) | 0.24811443136179492 / 0.0841737962816189 / 4.80e-14 / 0.010790237633926668 / 442.3 s (0.507 s per step): every key of the baseline |
| `stokes` (pre-lane library `386a308` + the setup change) | 0.17439096487276182 / 0.9998871416222597 / 1.48e-14 / 216.3 s | 0.2481144314795641 / 0.08417379643261974 / 1.82e-14 / 0.010790237633926666 / 469.8 s (0.538 s per step) |
| `integrated`, free vertices shifted by 1e-15 (seeds 0, 1; the round-off yardstick of rule 8) | not run | seed 0: 0.24811443168242495 / 0.08417379660376723 / 2.62e-14 / 0.01079023763392488; seed 1: 0.2481144313559126 / 0.08417379632864816 / 2.62e-14 / 0.010790237633926891 (446 / 449 s) |
| `csf_dual` | 0.2041112129331814 / 1.0252046890273294 / 1.00e-14 / 224.9 s (0.119 s per step); KE_max 1.07e-6 against 8.36e-7 | 0.2784903890745515 / 0.08567541803104081 / 6.58e-14 / 0.010788610398292235 / 458.3 s (0.526 s per step); apex l2 0.344 against 0.248, R_max at t_end 0.010163 against 0.010187 |

Reading: the 3D `stokes` run moves l2 by 4.7e-10 relative and the tail by
1.8e-9 (mass drift 1.8e-14 against 4.8e-14); a 1e-15 shift of the 377
free vertices moves the same run by up to 1.3e-9 in l2 and 3.8e-9 in the
tail (seed 0; seed 1: 2.4e-11 and 5.6e-10), so the `stokes` run lies
inside the round-off range of the default. In 2D the two values share
the function and the runs are bit-identical, as are the runs of the
default before and after the lane (the 2D and 3D runners now bind the
force through the preset). `csf_dual` in 2D: l2 +17 %, and the tail
reads 1.025 (the kinetic energy grows in the second half of the run,
KE_max 1.07e-6 J against 8.36e-7 J).

## 4. Decisions (protocol rule 5)

- `curvature_path='stokes'`: REMOVED (duplicate: same operator, same
  numbers to round-off on static and moving meshes, 3.5x the cost, one
  cache more). Not kept as a `broken` or `dead` option: the brief asks
  for removal of values that add nothing, and the reference integral in
  the driver keeps the equality measurable. Every number ever produced
  with it is reproduced by `'integrated'` to the digits quoted above.
- `curvature_path='csf_dual'`: `experimental` -> `measured-worse`. By
  the flip rule (l2 AND tail at least as good as the pin) it fails on
  both channels in both dimensions: 2D l2 0.20411 against 0.17439 and
  tail 1.0252 against 0.99989; 3D l2 0.27849 against 0.24811 and tail
  0.08568 against 0.08417. Kept registered with its evidence.
- `curvature_path='integrated'`: stays the default, `validated`; it now
  sums every triangle at an edge (bit-identical on every pinned mesh).
- No preset changed. No baseline re-pinned.

## 5. Pins and battery

- Every pin on the path is bit-identical: the 2D floors (2.374856801157627e-03
  / 2.271693780620923e-03), the 3D floor (7.274133897024673e-05), the 2D
  full run (every key of `baselines/baseline_oscillation.json`), the 3D
  full run (l2 0.24811443136179492, tail 0.0841737962816189, mass
  4.8000942590870176e-14, R_max_peak 0.010790237633926668, every key of
  `baselines/baseline_oscillation_3d.json`), both through the preset-bound
  force of the runners.
- Battery: ddgclib fast 1227 passed, 0 failed, 12 skipped, 2 xfailed (130 s; 1225 before
  the lane: 6 tests of the removed function dropped, 8 added); slow 32
  passed, 1 xfailed (242 s; unchanged counts, every slow pin bit-identical,
  the 3D floor included); hyperct not touched (`lane_diff.sh --stat`
  shows no hyperct change).

## 6. Measured DO-NOTs

- Do not add a "Stokes", "conormal" or "boundary-integral" variant of the
  3D interface curvature as a method value again: on a piecewise-linear
  surface it is the cotangent sum (section 2), and the only way to make it
  differ is to visit other triangles.
- Do not sum the first two apexes of an interface edge: a non-manifold
  edge (three or more interface triangles) loses 22 % of the force at its
  vertices.
- Do not compare stencils in the integrator callback on a coordinate-keyed
  sub-complex: after the move the keys of `HC.interface_triangles` are
  stale until the next retopology (a probe that does so sees a KeyError,
  or, with `.get`, a vanishing force: audit F2's symptom).
- Do not use `csf_dual` for accuracy or for cost (direction off by up to
  10 deg on the droplet, 180 deg on a non-manifold interface; 41x the
  cost per vertex in 3D).

## 7. How to reproduce (repo root, ddg env)

```bash
P=/home/endres/anaconda3/envs/ddg/bin/python
D=cases_dynamic/oscillating_droplet/diagnose_curvature_path.py
$P $D static                      # 3.1 (laneM_static.json), ~1 min
$P $D moving                      # 3.2 (laneM_moving.json), ~1 min
$P $D nonmanifold                 # 3.3 (laneM_nonmanifold.json), ~30 s
$P $D floors                      # 3.4 (laneM_floors.json), ~1 min
$P $D dynamic --dim 2 --arm integrated     # 3.5, ~4 min each
$P $D dynamic --dim 2 --arm csf_dual
$P $D dynamic --dim 3 --arm integrated     # ~8 min
$P $D dynamic --dim 3 --arm csf_dual
$P $D dynamic --dim 3 --arm integrated --perturb 1e-15 --perturb-seed 0
$P $D dynamic --dim 3 --arm integrated --perturb 1e-15 --perturb-seed 1
$P -m pytest ddgclib/tests/test_curvature_path.py ddgclib/tests/test_interface_cache_invalidation.py -q
```

The `'stokes'` rows were produced on the pre-lane library (ddgclib
`386a308` with the setup change of this lane, hyperct `48d163a`; the
`methods_laneM_*_stokes.json` files carry the SHAs); after the removal
the arm is refused at `SolverMethods` construction. The per-vertex
equality is reproducible on the shipped library through the reference
in the driver (`static`, `moving`, `nonmanifold`: the `stokes_reference`
arm); the full-run row follows from it and is kept as data
(`score_laneM_3d_stokes.json`). To re-run the arm itself, export
`386a308` (`git archive 386a308 ddgclib | tar -x -C DIR`) and run the
driver with that tree first on `sys.path`.

## 8. Files

Repo, changed: `ddgclib/operators/multiphase_stress.py`,
`ddgclib/_curvatures_heron.py`, `ddgclib/geometry/_interface_subcomplex.py`,
`ddgclib/methods/_axes.py`, `ddgclib/tests/test_interface_cache_invalidation.py`,
`ddgclib/tests/test_simplex_aware_curvature.py`, `ddgclib/tests/test_methods.py`,
`cases_dynamic/oscillating_droplet/src/_setup.py`,
`cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py`, `_3D.py`,
`cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py`, `METHODS.md`,
`DEVELOPMENT.md`, `debugging_plan.md`.
Repo, new: `ddgclib/tests/test_curvature_path.py`,
`cases_dynamic/oscillating_droplet/diagnose_curvature_path.py`, this log,
`cases_dynamic/oscillating_droplet/results/laneM/` (`laneM_static.json`,
`laneM_moving.json`, `laneM_nonmanifold.json`, `laneM_floors.json`,
`score_laneM_{2d,3d}_{integrated,stokes,csf_dual}.json` + `methods_` +
`diags_`, `score_laneM_3d_integrated_perturb1e-15_s{0,1}.json` + `methods_`
+ `diags_`, `prelane/` with the pre-lane `static`, `moving`, `floors`).
Scratch (logs only): `.../scratchpad/laneM/*.log`, `probe_nonmanifold.py`
(the first version of the `nonmanifold` sub-command).

## 9. Known limits, open

- The equality holds for the surface-tension stencil; the csf_dual
  direction and the two-apex truncation were the only differences found.
  The physics gaps of the droplet (laneG's 3D over-decay, laneH's 2D
  over-decay) are untouched: no stencil variant on a piecewise-linear
  surface changes them (Probe 2's verdict stands, now also on the moving
  mesh).
- `methods.dudt_fn` reaches the droplet setup only; the dam-break,
  electrolysis and shearing-plate setups still build their own partial
  (laneO's open item for those cases).
- Wall times in 3.5 were measured with other lane processes running.
