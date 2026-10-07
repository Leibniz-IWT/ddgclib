# laneS: exact setup dual volumes (builder simplex cache, 2D fallback fix)

Date: 2026-10-01. Closes laneK mechanism 2 (the setup dual volume) in the
library, so no case has to remember a workaround.

## 0. Verdict

Every domain builder now returns its mesh with the top-simplex cache
`HC._simplices` populated from the connectivity it built. The first
`cache_dual_volumes` call therefore gives the exact barycentric volumes
`Vol_i = sum_{T contains i} |T| / (dim + 1)`, in 2D and in 3D, the same
source every later retopology uses. The 2D fallback
`hyperct.ddg.dual_cell_area_2d` is fixed as well (the half cell of a
boundary vertex is closed through the vertex itself), so a hand-built 2D
complex without a cache also tiles its domain.

The shipped, unmodified `Hydrostatic_2D.py` settles over 100 acoustic
times (|u| 3.0e-4 m/s) instead of reaching 10 c0 at 6.9 t_ac. No pinned
number moved: every pinned setup already carried a Delaunay cache, and the
full 2D and 3D droplet runs, the static droplet, the dam break and the
template reproduce their numbers to the bit.

## 1. What changed and where

hyperct (working tree, uncommitted):

| file | change |
|---|---|
| `hyperct/ddg/_retriangulation.py` | new `rebuild_simplex_cache_3d(HC)`: the tetrahedra of the EXISTING connectivity as K_4 cliques, nothing re-triangulated. `rebuild_simplex_cache_2d` now enumerates in `HC.V` order instead of `id()` order |
| `hyperct/ddg/__init__.py` | exports `rebuild_simplex_cache_3d` |
| `hyperct/ddg/_dual_cell.py` | `_dual_cell_polygon_2d_walk`: a boundary vertex is walked as an open chain between its two boundary-edge midpoint duals and closed through the vertex. Interior vertices take the unchanged closed-cycle branch |
| `hyperct/remesh/_quality.py` | `iter_triangles_2d` skips neighbours that are no longer in `HC.V` (section 5) |
| `hyperct/tests/test_dual_volume.py` | +10 tests (`TestDualCellArea2dBoundary`, `TestRebuildSimplexCache`) |
| `hyperct/tests/test_remesh.py` | +1 test (`iter_triangles_2d` and a vertex dropped from the cache) |

ddgclib (working tree, uncommitted):

| file | change |
|---|---|
| `ddgclib/geometry/domains/_result.py` | `DomainResult.__post_init__`: if `HC._simplices` is `None`, call `rebuild_simplex_cache_2d` (dim 2) or `rebuild_simplex_cache_3d` (dim 3). This is the ONE place; all ten `DomainResult(...)` constructions in the builders go through it. A mesh that already has a cache (the Delaunay-built droplet meshes) is left alone |
| `ddgclib/geometry/domains/_rectangles.py` | docstring of `retopologize=` only |
| `ddgclib/tests/test_builder_simplex_cache.py` | new, 27 tests |
| `ddgclib/tests/test_simplex_aware_duals.py` | `test_rectangle_retopologize_populates_simplices` asserted the old default (`_simplices is None`); it now asserts that the default cache spans exactly the builder edges |
| `ddgclib/methods/_axes.py` | `dual_volume` axis: `simplex_exact`, `fan_walk_3d`, `dual_cell_area_2d` (status `broken` -> `opt-in`) evidence; `dual_path` control text |
| `ddgclib/methods/_presets.py` | note of `hagen_poiseuille_3D` |
| `cases_dynamic/template/template.py` | the explicit `rebuild_simplex_cache_2d(HC)` workaround is gone |
| `cases_dynamic/template/diagnose_single_phase_eos.py` | `hydro`: variant `case` keeps the builder cache (it used to reset `HC._simplices = None`); new variants with `fallback` in the name drop it, so the loop runs on `dual_cell_area_2d` |
| `cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py` | `retopologize_cylinder` calls `invalidate_simplex_cache(HC)` (section 4.4) |

No operator, integrator, IC or `SolverMethods` code changed.
`cache_dual_volumes` and `dual_volume` in `ddgclib/operators/stress.py`
are untouched: they already prefer the simplex rule when the cache exists.

## 2. Design decisions

**Where the cache is populated.** `DomainResult.__post_init__`, not each
builder and not `cache_dual_volumes`. Every builder ends by constructing a
`DomainResult` (after its vertex removals and merges), so one hook covers
`rectangle`, `l_shape`, `disk`, `annulus`, `box`, `cylinder_volume`,
`pipe`, `ball`, `periodic_rectangle`, `periodic_box`. Populating inside
`cache_dual_volumes` was rejected: it would hide a topology side effect in
a volume routine and change the dual path of meshes that are deliberately
cache-free.

**3D enumeration is feasible.** On every 3D builder mesh the K_4 cliques
of the 1-skeleton are exactly the tetrahedra: each triangle belongs to one
or two cliques, every edge is covered, no clique is degenerate, and the
clique volumes sum to the box volume (1.0 and 2.0 exactly) or to the
inscribed polyhedron (cylinder refinement 1: 0.7071067811865476 = octagonal
prism). `rebuild_simplex_cache_3d` has two safeguards, both tested:

- a clique split by a common neighbour strictly inside it is skipped (the
  3D form of the ghost-K_3 filter of the 2D routine);
- if any triangle is shared by more than two cliques the cache is left
  `None` and 0 is returned, so callers keep their fallbacks.

It is not meant for general Delaunay meshes (four mutually connected
vertices need not span a cell there); those get their cache from
`connect_and_cache_simplices`.

**Deterministic order.** `rebuild_simplex_cache_2d` ordered the triangles
and the vertices inside each triangle by `id()`. Five runs of the same
script produced four different orders (md5 of the coordinate list:
`bed58b77`, `18150560`, `816eb769` twice, `9c0fb0e7`). A builder cache
with a per-process order would make every 2D setup depend on memory
addresses, so both rebuild routines now enumerate by rank in `HC.V`
(four runs: `703f5224` each time). Rounding of anything summed over the
cache can differ from an `id()`-ordered run in the last bits. The template
is unaffected: `final_state.json` has md5 `404fa8b532ca6d61715193ba4f84833d`
on the pre-lane code (4 runs) and on the new code.

**The fallback.** A boundary vertex has an open chain of dual vertices:
boundary-edge midpoint, triangle barycentres, boundary-edge midpoint. The
old code fell back to an angular sort of those points, which leaves the
vertex out. That drops the triangle (midpoint, vertex, midpoint): nothing
on a straight boundary, 3/4 of a right-angle corner cell, and 3/4 of the
response of a free-surface vertex to its own normal motion. The walk now
handles the open chain and appends the vertex. `dual_cell_polygon_2d` is
also what `ddgclib.analytical` integrates pressure over, so the polygon
(not only the area) had to change: volume and reference integral must be
the same cell. The angular sort remains for degenerate fans only.

## 3. Pin safety (measured, not assumed)

Method: `scratchpad/laneS/fingerprint.py` builds each setup in its own
interpreter and digests the sorted per-vertex `(x, dual_vol, m, p)`. It was
run before any edit, after the hyperct fallback fix alone, and at the end.

| setup | total dual volume before | fallback fix only | final | state digest |
|---|---|---|---|---|
| `rectangle` 1x1 r2 | 0.9687500000000022 | 1.0000000000000013 | 1.0000000000000004 | moved |
| `rectangle` 2x1 r3 | 1.9843749999999956 | 1.9999999999999991 | 2.000000000000001 | moved |
| `l_shape` r3 (exact 1.5) | 1.4843749999999998 | 1.5000000000000038 | 1.5000000000000007 | moved |
| `disk` R1 r3 (polygon) | 3.0863382338065533 | 3.1190915091691283 | 3.1190915091691274 | moved |
| `annulus` r3 | compute_vd IndexError | IndexError | 2.579367717099514 | now works |
| `box` r1 | 0.9166666666666667 | same | 1.0 | moved |
| `box` 2x1x1 r2 | 1.8749999999999998 | same | 2.0000000000000018 | moved |
| `cylinder_volume` R0.5 L1 r1 | 0.651057271712679 | same | 0.7071067811865476 | moved |
| `pipe` R0.5 L3 r1 | 1.9603619542005633 | same | 2.121320343559643 | moved |
| `ball` R1 r1 | 3.0262316577131787 | same | 3.265986323710905 | moved |
| `periodic_rectangle`, `periodic_box` | compute_vd IndexError | IndexError | 1.0000000000000004, 1.0 | now works |
| `setup_hydrostatic_column` 2D | 0.9921874999999991 | 1.0000000000000004 | 1.0000000000000002 | moved |
| `setup_hydrostatic_column` 3D | 0.9166666666666667 | same | 1.0 | moved |
| `setup_dam_break_single_phase` 2D | 0.0024804687500000026 | 0.002500000000000004 | 0.0024999999999999996 | moved |
| `setup_dam_break_single_phase` 3D | 0.00011718750000000004 | same | 0.00012500000000000003 | moved |
| `setup_capillary_rise` 2D | 4.260837751627386e-06 | 4.398284130712138e-06 | 4.39828413071214e-06 | moved |
| `setup_capillary_rise` 3D | 5.727069732717234e-09 | same | 6.220113068823459e-09 | moved |
| `setup_dam_break_multiphase` 2D | 0.020000000000000007 | same | same | BIT-IDENTICAL |
| `setup_dam_break_multiphase` 3D | 0.0012630208333333332 | same | same volume, masses moved | moved (section 4.4) |
| `setup_electrolysis_bubble` 2D, 3D | | | | BIT-IDENTICAL |
| `setup_oscillating_droplet` 2D, 3D | | | | BIT-IDENTICAL |
| `test_single_phase_remap._box` | 1.0000000000000004 | same | same | BIT-IDENTICAL |

Every setup that moved was reading the broken fallback (2D) or the fan
walk (3D); none of them carries a pinned number. Every pinned setup was
bit-identical because it runs a library retopology (or builds from a
Delaunay mesh) before its ICs. This was verified, not assumed:

| pin | configuration | result |
|---|---|---|
| `test_single_phase_remap.py` `PIN_KE0` 0.43990437705748403, `PIN_KE_END` 0.008987112540608227 | `SolverMethods(dim=2, connectivity='delaunay', remap='conservative', redistribute_mass=True)` | test passes unchanged |
| `oscillating_droplet_2D.py` full run | `PRESETS['oscillating_droplet_2D']` | l2 0.17479361640597058, linf 0.32364245955409165, tail 0.9998967874595965, mass 2.4056717879332966e-14: identical to `baseline_oscillation.json` on every scalar key |
| `static_droplet_2D.py` | `PRESETS['static_droplet_2D']` | summary 0.0011847162859108737, mass 0.0: identical |
| `oscillating_droplet_3D.py` full run | `PRESETS['oscillating_droplet_3D']` | l2 0.24811340819647862, tail 0.08409976059818802, mass 1.905002320272536e-14, R_max_peak 0.010790236105250779: identical to `baseline_oscillation_3d.json` on all 18 numeric keys |
| `dam_break_2D.py` full run | `PRESETS['dam_break_2D']` | every printed step line identical to the pre-lane run; KE_liq peak 1.0369e-03 J at t = 0.0506 s |
| template | `SolverMethods(dim=2, connectivity='delaunay', remap='conservative', redistribute_mass=True)` | `final_state.json` md5 `404fa8b5...` pre-lane, with the workaround, without it |
| slow pinned battery | presets | 16 passed, 1 xfailed, as at baseline |

Except for the template (run in place), the scripts were run from scratch
copies so the case `fig/` and `results/` directories were not overwritten.
"Pre-lane" runs used the source mirror `scratchpad/snap_S` on `PYTHONPATH`.

## 4. Measurements

### 4.1 Hydrostatic column, lane K driver (task 5)

`cases_dynamic/template/diagnose_single_phase_eos.py hydro`: the loop of
`Hydrostatic_2D.py` section 3 copied verbatim (hand-rolled, so there is no
`SolverMethods` for it: symplectic update, `_recompute_duals` +
`cache_dual_volumes` each step, fixed connectivity, 0 flips). Rectangle
refinement 3, 145 vertices, 25 frozen, 9 free top vertices, Tait n = 1,
K = 9.81e5 Pa, c0 = 31.32 m/s, mu_art = 1590.6 Pa s, CFL 0.25.

| variant | volume source | pre-lane | after laneS |
|---|---|---|---|
| `case`, 8 t_ac | builder mesh | setup total 0.99219; \|u\| 313.2 m/s (10 c0) at 6.93 t_ac, abort | setup total 1.00000; \|u\| 2.125e-02, KE 5.530e-02 J at 8 t_ac |
| `case`, 40 t_ac | builder cache | | KE 1.003e-05 J, \|u\| 5.868e-04, interior p 598 to 9.18e3 Pa (= laneK `exact`: 1e-5 J, 6e-4, 598 to 9180) |
| `nogravity_seed`, 15 t_ac | builder mesh | \|u\| 316.4 m/s at 14.08 t_ac, abort | 1.753e-06 -> 4.162e-10 m/s (decays) |
| `fallback`, 8 t_ac | cache dropped, fixed `dual_cell_area_2d` every step | n/a | \|u\| 2.125e-02, KE 5.530e-02 J: equal to `case` in every printed digit |
| `nogravity_fallback_seed`, 15 t_ac | same | n/a | 1.753e-06 -> 4.155e-10 m/s |
| `exact`, `nogravity_exact_seed` | explicit rebuild | | same as `case` and `nogravity_seed` (the variants are now redundant) |

So the builder path and the repaired fallback both behave like the exact
variant of laneK.

### 4.2 Shipped Hydrostatic_column runners, unmodified

Run from scratch copies, pre-lane code against the new code. Porting them
is lane P; nothing in the runners was edited.

| runner | pre-lane | after laneS |
|---|---|---|
| `Hydrostatic_2D.py` static residual (section 2) | max\|a\| 332.3002, mean 11.6067 m/s^2 | max\|a\| 1.9075, mean 0.3292 m/s^2 |
| `Hydrostatic_2D.py` dynamic (100 t_ac) | ABORT at step 5859, 7 t_ac, \|u\| 312.98 m/s, KE 8.775e+05 J | completes, 4607 steps, KE 2.253973e-06 J, \|u\| 3.013386e-04, settled max\|a\| 1.006e-04, max\|p V - int P dV\| 1.5430, integrated L2 80.79 Pa |
| `Hydrostatic_3D.py` static residual | max\|a\| 1.4014, mean 1.1015; EOS residual 1.401431 | max\|a\| 0.0000 (printed), EOS residual 8.302737e-06 |
| `Hydrostatic_3D.py` dynamic (15 t_ac) | KE 2.776847e-04, \|u\| 2.984550e-03, settled max\|a\| 0.8815, max\|p V - int P dV\| 6.4630 | KE 4.556016e-05, \|u\| 8.337640e-04, settled max\|a\| 2.509e-03, 4.0671 |
| `Hydrostatic_2D_periodic.py` | ABORTED at step 840 (6.8 t_ac), \|u\| 64.7, P err 9995 % | completes 100 traversals, KE 2.013e-02, \|u\| 3.233e-02, P err 23 %, settled max\|a\| 193.8: runs, NOT settled |

The 3D improvement comes from two things the cache switches on together:
exact volumes (the box total was 0.9167) and the simplex-aware dual
construction (the legacy walk treats every face with three boundary-tagged
vertices as a boundary face).

### 4.3 Free surface through the library integrator

`SolverMethods(dim=2, connectivity='dual_only')` (single phase,
`symplectic_euler`, no remap, no redistribution), rectangle refinement 2
(41 vertices, 13 frozen, 9 free top vertices), Tait n = 1, c0 = 31.32,
mu_art = 0.5 rho c0 mean(dx), CFL 0.25, g = 0, 1e-6 m/s seed, 181 steps
(8 t_ac): max\|u\| 1.538e-06 -> 1.989e-09, never above its initial value.
Refinement 3 (145 vertices, 362 steps): 1.753e-06 -> 2.457e-09.

On the pre-lane code the same call raises `IndexError` in the legacy
`compute_vd`: `dual_only` tags only `bV` as boundary, and the 1-skeleton
walk needs every hull vertex tagged. With the builder cache the
simplex-aware path does not read the tags. This is the Hydrostatic loop
without a hand-rolled loop, i.e. the starting point of lane P, and it is
pinned as `test_free_surface_box_is_stable_through_the_library_integrator`.

### 4.4 Setups that moved and why

- `setup_dam_break_multiphase(dim=3)` (smoke only, no pin): the first
  `mps.refresh` used to find no cache and ran a Delaunay on top of the
  builder edges (the laneA footgun in `assign_simplex_phases*`). It now
  votes the simplex phases on the builder tetrahedra. 9 of 189 vertices
  change label (bulk air / liquid / interface 165 / 7 / 17 -> 162 / 8 /
  19), total mass 0.062192331881568544 -> 0.08470214097307874. The 2D
  setup is bit-identical (2D enumerates triangles from the 1-skeleton
  either way). Not re-pinned, nothing to re-pin; flagged for the dam break
  lane.
- `Hagen_Poiseuile_3D.py`: its case-local `retopologize_cylinder` rewires
  the connectivity by hand and keeps no simplex list. With the builder
  cache left in place `compute_vd` would build duals from the OLD
  tetrahedra. The function now calls `invalidate_simplex_cache(HC)`, which
  keeps the case exactly where it was: 6-step smoke
  (`--n-refine 1 --n-steps 6 --workers 1`, `PRESETS['hagen_poiseuille_3D']`)
  `hp3d_final_state.json` md5 `c99c6d52cef8d81be9298b4c16e5c9f8` pre-lane
  and after; without the invalidation `d588a321...` (stale cache).

## 5. A latent flake this lane exposed and fixed

`test_methods.py::TestMultiphaseBuilders::test_periodic_multiphase_is_bit_identical_to_shearing_wrapper`
failed in the baseline fast-suite run of this lane, before any edit
(`KeyError` in `assign_vertex_phases_from_simplices`). Cause, measured:

1. `setup_shearing_plate_droplet` rescales the outer vertices with
   `HC.V.move`; 23 moves land on an occupied coordinate, and each such
   move drops the displaced vertex from `HC.V` while its edges stay.
2. `hyperct.remesh._quality.iter_triangles_2d` yields each triangle from
   its lowest-`id()` vertex. A triangle that contains a dropped vertex was
   therefore yielded, and crashed the phase assignment, unless the dropped
   vertex happened to have the smallest address of the three.

Failure rate of the single test: pre-lane code 1 of 20. With the builder
cache 17 of 20: the sub-builders inside `droplet_in_box_2d` now allocate a
simplex list, which shifts the heap layout the test was depending on.
Fix: `iter_triangles_2d` skips neighbours that are not in `HC.V`. After
it 20 of 20 pass, and the probe state digest is
`17e9a78408066feab8ab38e43c1fdb5e...` (287 vertices) in 24 of 24 runs, the
same digest every passing pre-lane run produced. On a mesh without
dropped vertices the function yields exactly what it did (droplet 2D full
run bit-identical, section 3).

The 23 collisions themselves are untouched: the shearing-plate setup still
loses 23 vertices. That belongs to lane L (`HC.V.move` collision check)
and the shearing-plate lane.

## 6. DO-NOTs (measured)

- Do not rewire the connectivity of a builder mesh by hand (disconnect /
  connect, vertex removal, merge) and then call `compute_vd` or
  `cache_dual_volumes` without `invalidate_simplex_cache(HC)` or a rebuild:
  the cache describes the old simplices (HP3D smoke: different result with
  the stale cache). The library BCs that inject or delete vertices already
  invalidate.
- Do not order anything by `id()` when the result feeds a number
  (`rebuild_simplex_cache_2d` gave four orders in five runs).
- Do not read the `fallback` driver variants as a reason to drop the cache:
  the fallback is correct only for barycentric duals on a valid fan, the
  cache also switches on the simplex-aware `compute_vd`.
- Do not compare a setup volume from before 2026-10-01 with one after it
  on a builder mesh: the totals differ by the corner defect (2D, 0.8 to
  3.1 %) or the fan-walk defect (3D, 6 to 8 %).

## 7. Known limits

- A hand-built 3D complex without a simplex cache still gets the `v_star`
  fan walk (box total 0.9167). Call
  `hyperct.ddg.rebuild_simplex_cache_3d(HC)` (structured connectivity) or
  run one retopology before the ICs. In 2D the repaired fallback covers it.
- 3D setup and 3D retopology still differ by CONVENTION on boundary
  vertices: `cache_dual_volumes` at setup gives them their truncated cell,
  the 3D retopology zeroes them (reported axis `boundary_dual_vol`). The
  interior volumes agree. Not changed here.
- The repaired polygon of a re-entrant corner (L-shape notch) is not
  convex. `ddgclib.analytical._integrated_comparison` fans a polygon from
  its mean point with absolute triangle areas, which is exact only for
  star-shaped cells seen from that point. Not hit by any test; check
  before using integrated comparisons at re-entrant corners.
- Circumcentric boundary cells are not validated (laneK P14, P15).
- `rebuild_simplex_cache_3d` costs about 20 microseconds per tetrahedron
  (box refinement 4: 9009 vertices, 49152 tetrahedra, 0.95 s), paid once
  per builder call.
- `capillary_rise_2D_dynCA.py` calls `rebuild_simplex_cache_2d` after its
  remesh batches. Its triangle order is now reproducible; whether that
  changes dynCA output in the last bits against earlier (per-process
  ordered) runs was not measured. The runner itself was not touched.
- `iter_triangles_2d` is still `id()`-ordered for the triangles it yields.
  The pins are reproducible with it, so it was left alone.
- The methods block of a fresh droplet score carries two fields the
  pinned baselines do not have (`pressure_flux='centred'`,
  `density_diffusion=None`, both defaults, added 2026-09-26 by another
  lane), so `diff_baselines` will report a configuration difference on
  those keys although every number is identical. Not touched here.

## 8. Tests

- hyperct: 290 -> 301 passed (`pytest hyperct/tests -k "not benchmark"`).
  The command in the lane brief, `pytest -q` at the hyperct root, stops at
  collection with 4 "import file mismatch" errors from `archives/` and
  `hyperc_rl_quick_figs_delete/`; that is pre-existing and unrelated.
- ddgclib fast: 983 -> 1010 passed, 12 skipped, 2 xfailed.
- ddgclib slow: 16 passed, 1 xfailed.
- `benchmarks/test_integrated_suite.py`, `test_integrated_hessian.py`,
  `boundary_integrals/tests`: 273 passed, 3 skipped, before and after.
- The new tests fail on the pre-lane code: all 10 new hyperct tests
  in `test_dual_volume.py` (rectangle total 1.984375 instead of 2.0, corner
  cell 0.000658 instead of 0.002721, free-surface gain 1.0417e-05 instead
  of 4.1667e-05), the `iter_triangles_2d` test, and 23 of the 27 tests of
  `test_builder_simplex_cache.py`.

## 9. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
S=<scratchpad>/laneS
$PY -m pytest ddgclib/tests/test_builder_simplex_cache.py ddgclib/tests/test_single_phase_remap.py -q
(cd ../hyperct && $PY -m pytest hyperct/tests/test_dual_volume.py hyperct/tests/test_remesh.py -q)
$PY cases_dynamic/template/diagnose_single_phase_eos.py hydro --out $S/h --variant case --n-tac 8
$PY cases_dynamic/template/diagnose_single_phase_eos.py hydro --out $S/h --variant nogravity_seed --n-tac 15
$PY cases_dynamic/template/diagnose_single_phase_eos.py hydro --out $S/h --variant fallback --n-tac 8
$PY cases_dynamic/template/diagnose_single_phase_eos.py hydro --out $S/h --variant nogravity_fallback_seed --n-tac 15
$PY cases_dynamic/template/template.py && md5sum cases_dynamic/template/results/final_state.json
$PY $S/fingerprint.py $S/fp.json          # setup digests, one interpreter per setup
# pre-lane reference: run the same script from a scratch copy with
# PYTHONPATH=<scratchpad>/snap_S/hyperct_repo:<scratchpad>/snap_S/ddgclib_repo
```

Note for the reviewer: `lane_diff.sh S` also lists run artefacts that the
pre-lane reference runs left INSIDE `snap_S` (`__pycache__`,
`.pytest_cache`, `cases_dynamic/template/results`, `fig`), and the
concurrent edits of the capillary_rise_energy_grad agent
(`capillary_rise_energy_grad/*`, `capillary_rise/capillary_rise_2D_dynCA.py`,
one `DEVELOPMENT.md` hunk). None of those are lane S changes.
