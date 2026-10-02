# laneT: deterministic iteration order (a run is a pure function of its inputs)

Date: 2026-10-02. Replaces protocol rule 8 of `debugging_plan.md` ("3D
runs are not reproducible between processes, cause not traced").

A first attempt at this lane was interrupted by a machine restart. Its
edits were in the working trees, its measurements were lost. Everything in
this log was measured again from the start; section 1 says what was kept,
rewritten and reverted.

## 0. Verdict

A dynamic run is now the same to the bit in every fresh interpreter and
after any other runs in the same interpreter, in 1D, 2D and 3D, on every
value of the connectivity axis: 32 detector cases, 6 interpreters each
(plain, after two other runs, four with a fragmented heap), one final
state per case (section 3). Before the lane 10 of those cases gave
between 2 and 8 final states.

The cause was not the iteration order of `v.nn`. A hyperct vertex hashes
by its coordinate tuple (`VertexBase.__hash__`), so every set of primal or
dual vertices iterates in an order that is a function of the coordinates
and of the order of the insertions. What followed the memory addresses
were four explicit `id()` orders:

1. `hyperct.ddg.compute_vd` (simplex-aware 3D path) summed the barycentre
   of each BOUNDARY face in `id()` order of its three vertices. A sum of
   three floats depends on its order in the last bit; the barycentre is
   the coordinate key of the dual vertex, so its hash and the iteration
   order of every set that holds it (`v.vd`, the fan walk, the `p_ij`
   ring) followed the addresses. This alone is the 3D process dependence
   of lanes P and H.
2. `hyperct.remesh._driver._edge_list` oriented each edge by `id()`. The
   sweeps are not symmetric in the pair (a split copies the carried
   attributes of the first vertex to the midpoint, a collapse keeps the
   first vertex object and removes the second), so `adaptive_remesh`
   produced another mesh in every process (2D droplet with
   `connectivity='adaptive'`, the dynCA runner).
3. `hyperct.remesh._quality.iter_triangles_2d` listed each triangle from
   its lowest-`id()` vertex, vertices in `id()` order (lane L had met the
   consequence in the tied phase vote).
4. `ddgclib.operators.stabilisation.density_diffusion_step` visited each
   pair from its lower-`id()` endpoint, so the sums into a cell were
   ordered by addresses.

No pinned number moved: every case that was deterministic before the lane
has the same digest after it, both suites pass on the unchanged pins, and
the full 2D and 3D droplet runs reproduce their baselines in every key.
Two end-of-run pins were added to the hydrostatic 3D tests.

One finding reaches beyond the lane and is NOT fixed (section 6): the 3D
`p_ij` area vector of a boundary edge contains a spurious polygon vertex
that a geometric tie on the builder lattice switches on and off; 30 of
the 56 boundary edges at the free-surface vertices of the refinement 2
hydrostatic column are off by up to 37 %. That tie is what a 1e-15
perturbation flips on fixed connectivity. A strict xfail test holds it.

## 1. What changed and where

hyperct (working tree, on `476c289`):

| file | change |
|---|---|
| `hyperct/ddg/_compute_dual.py` | `_compute_vd_3d_simplex_aware`: the three vertices of a boundary face are taken in the order of the simplex that owns the face (`(v for v in simplices[simp_ids[0]] if id(v) in face)`), not in `id()` order. The sorted ids remain the KEY of the face. Same in `_compute_vd_2d_simplex_aware` (boundary edge; the midpoint is commutative, only the local merge neighbourhood depended on the order) and in `_compute_vd_nd_simplex_aware`. The `id_to_v` dictionaries are gone, so the routine does less work than before |
| `hyperct/remesh/_driver.py` | `_edge_list`: each edge once, in `HC.V` order, oriented from the endpoint that comes first in `HC.V` |
| `hyperct/remesh/_quality.py` | `iter_triangles_2d`: each triangle from its first vertex in `HC.V` order, vertices in that order; ranks of the later neighbours are looked up once per vertex (0.87 -> 0.59 ms on 311 vertices) |
| `hyperct/tests/test_deterministic_order.py` | new, 10 tests: the same complex built three times with the heap fragmented in between; dual vertex positions, dual cache order, `v.nn` and `v.vd` iteration order, edge list, triangle list, `adaptive_remesh` result. 7 of them fail on `476c289` |

ddgclib (working tree, on `5683e78`):

| file | change |
|---|---|
| `ddgclib/operators/stabilisation.py` | `density_diffusion_step`: each pair once, from its first endpoint in the order of the vertex list |
| `cases_dynamic/diagnose_determinism.py` | new: the detector (section 2) |
| `ddgclib/tests/test_determinism.py` | new: 5 fast + 1 strict xfail + 6 slow (section 8) |
| `ddgclib/tests/test_case_hydrostatic.py` | `PIN_3D_KE_40`, `PIN_3D_REMAP_KE_END` (section 5.3) |
| `ddgclib/multiphase.py`, `ddgclib/tests/test_multiphase.py` | comments only (the triangle vertex order is no longer `id()` order) |
| `ddgclib/methods/_axes.py`, `_presets.py`, `METHODS.md` | evidence: `delaunay_material`, `adaptive`, `periodic`, `pressure_flux` centred and simplex_gradient, `edge_area_source` p_ij ring; preset note of `hagen_poiseuille_3D`; case matrix rows |
| `cases_dynamic/Hagen_Poiseuile_3D/README.md`, `Hydrostatic_column/README.md`, `Hagen_Poiseuile/diagnose_poiseuille.py` | the statements about process dependence |
| `debugging_plan.md`, `DEVELOPMENT.md` | status entry, rule 8 rewritten, checklist |

No operator, integrator or `SolverMethods` code changed, and no axis
gained a value: the old behaviour was a function of memory addresses and
cannot be kept as a registered option (rule 5; there is nothing to
reproduce it with).

Handoff. Kept from the interrupted attempt: the diagnosis of item 1, the
`_edge_list`, `iter_triangles_2d` and `density_diffusion_step` fixes, the
detector, the two test files, the two hydrostatic pins (the values were
re-measured and are the same). Rewritten: the `compute_vd` fix (the
attempt stored a face-to-vertices dictionary for every face; the version
above reads the owning simplex for boundary faces only, same results,
no extra cost), the inner loop of `iter_triangles_2d`. Reverted: an edit
of `ddgclib/geometry/periodic.py:_fixup_periodic_duals`; the function has
no caller, so the edit could not be tested and changed nothing. Added:
detector cases for every connectivity value, the lane H and lane P runs,
the pinned runs; the `v.nn` order test; the tie finding and its test;
attribution, perturbation and timing measurements.

## 2. The detector

`cases_dynamic/diagnose_determinism.py` runs a short case through its
preset (or a `.replace(...)` arm) and prints a SHA-256 digest of the
sorted exact state of every vertex (position, velocity, mass, pressure,
per-phase masses and pressures, dual volume).

- `run CASE [--steps N]` one run; `--pre A,B` runs other cases first in
  the same interpreter; `--scramble SEED` fragments the small-object heap
  before anything is built, so that vertex addresses are not monotone in
  creation order (99 of 188 consecutive pairs ascending instead of 159);
  `--perturb EPS` shifts the interior vertices by `EPS * scale * r`;
  `--trace FILE` one digest per step; `--lib DIR` imports ddgclib and
  hyperct from another tree (an export of an older commit).
- `sweep CASES --procs N` runs N fresh interpreters (plain, after
  `--pre`, then scramble seeds) and reports the distinct digests; exit
  status 1 when they differ. `sweep all` is the battery.

The fragmentation matters: plain interpreters lay most vertices out in
creation order and often agree with each other (lane H: 7 of 9 processes,
lane L: 108 of 120); the fragmented ones sample other address orders.

## 3. Measurements

"Before" is the library at ddgclib `5683e78` / hyperct `476c289`,
exported with `git archive` and run through `--lib`, with the cases of
this tree. Six or more interpreters per case; a digest is 16 hex digits
of the SHA-256.

### 3.1 Every case, before and after

| case | configuration (`SolverMethods`) | steps | before: final states / interpreters | after | digest after | equal to before |
|---|---|---|---|---|---|---|
| `hp3d` | `PRESETS['hagen_poiseuille_3D']`, L 2, mu 0.1, refinement 1, dt 0.01 | 300 | 1 / 6 | 1 / 6 | `9c9a2ae4f85d60e2` | yes |
| `hp3d_centred` | same, `.replace(pressure_flux='centred')` | 300 | 6 / 8 | 1 / 6 | `b566aa57d87618d4` | one of them |
| `hp3d_ring` | `SolverMethods(dim=3, connectivity='custom', viscous_flux='simplex_gradient')`, wrapper clears the area cache | 150 | 1 / 6 | 1 / 6 | `2d97ae1ef84bb03e` | yes |
| `hp3d_centred_laneH` | as `hp3d_centred`, L 3 (the arm of `diagnose_poiseuille.py arms3d`) | 600 | 3 / 9 | 1 / 6 | `42f3d9a1a4a106ce` | one of them |
| `hp3d_ring_laneH` | as `hp3d_ring`, L 3 | 600 | 3 / 9 | 1 / 6 | `2be9494bc84204b6` | one of them |
| `hp3d_centred_mp` | `hp3d_centred` + `backend='multiprocessing'` | 100 | 4 / 6 | 1 / 6 | `607127079316a088` | one of them |
| `hydro3d` | `PRESETS['hydrostatic_3D']` (`dual_only_bare`), refinement 2 | 150 | 8 / 8 | 1 / 6 | `001806ce51045a1b` | no |
| `hydro3d_remap` | `remap_arm(PRESETS['hydrostatic_3D'])` (`delaunay_material` + conservative remap), refinement 2 | 150 | 8 / 8 | 1 / 6 | `9491f8ccf2e84bb4` | no |
| `hydro3d_remap_r1` | same, refinement 1 | 40 | 1 / 6 | 1 / 6 | `8bee59780589bda2` | yes |
| `hydro3d_remap_40tac` | same, refinement 2, 40 acoustic times | 739 | 4 / 4 | 1 / 6 | `e4aac91704f23645` | no |
| `droplet3d` | `PRESETS['oscillating_droplet_3D']` (`dual_only`), refinement 2/2 | 40 | 1 / 6 | 1 / 6 | `a1c3963c02137001` | yes |
| `droplet3d_delaunay` | `PRESETS['oscillating_droplet_3D_delaunay']` | 40 | 1 / 8 | 1 / 6 | `1aefa5622d4d8c7f` | yes |
| `droplet3d_remap` | same, `.replace(remap='conservative')` | 40 | 1 / 6 | 1 / 6 | `243a9c350af60878` | yes |
| `droplet3d_frozen` | `PRESETS['oscillating_droplet_3D'].replace(connectivity='frozen')` | 40 | 1 / 6 | 1 / 6 | `861df7ec916a714c` | yes |
| `dam_break3d` | `PRESETS['dam_break_3D']` | 5 | 1 / 6 | 1 / 6 | `12e7b83ef28be68a` | yes |
| `electrolysis3d` | `PRESETS['electrolysis_bubble_3D']` | 30 | 1 / 6 | 1 / 6 | `cdcd9fc6b7cbb4b1` | yes |
| `shearing3d` | `PRESETS['shearing_plate_droplet_3D']` (`periodic`), setup of `_run_short_3D.py` | 5 | 2 / 6 | 1 / 6 | `bfd138324d4dd24d` | one of them |
| `hydro1d` | `PRESETS['hydrostatic_1D']`, refinement 3 | 300 | 1 / 6 | 1 / 6 | `1a59dab4975b5dee` | yes |
| `hp2d` | `PRESETS['hagen_poiseuille_2D']`, L 3, refinement 1 | 300 | 1 / 6 | 1 / 6 | `e2a6a3ea73a23ce4` | yes |
| `hp2d_centred_twopoint` | same, `.replace(viscous_flux='two_point')` | 300 | 1 / 6 | 1 / 6 | `2f7f27c28e057795` | yes |
| `hydro2d_remap` | `remap_arm(PRESETS['hydrostatic_2D'])`, refinement 2 | 300 | 1 / 6 | 1 / 6 | `13b753f62143d602` | yes |
| `hydro2d_density_diffusion` | `PRESETS['hydrostatic_2D'].replace(density_diffusion=0.05)` | 300 | 1 / 6 | 1 / 6 | `d532cf9aaaaaf9a7` | yes |
| `droplet2d` | `PRESETS['oscillating_droplet_2D']`, refinement 2/2 | 100 | 1 / 6 | 1 / 6 | `9baa00d24e0bee92` | yes |
| `droplet2d_bare` | `PRESETS['oscillating_droplet_2D_bare_delaunay']` | 100 | 1 / 6 | 1 / 6 | `dfe04ff02640558a` | yes |
| `droplet2d_dual_only` | `PRESETS['oscillating_droplet_2D_dual_only']` | 100 | 1 / 6 | 1 / 6 | `a4505e1bc1932eed` | yes |
| `droplet2d_adaptive` | bare preset `.replace(connectivity='adaptive', remesh_kwargs=...)` (the kwargs of `oscillating_droplet_2D_adaptive.py`) | 60 | 6 / 6 | 1 / 6 | `0956407aef0625e0` | one of them |
| `droplet2d_workers` | `PRESETS['oscillating_droplet_2D'].replace(workers=2)` | 40 | 1 / 6 | 1 / 6 | `52a42ad941f29aed` | yes |
| `dam_break2d` | `PRESETS['dam_break_2D']` | 200 | 1 / 6 | 1 / 6 | `017fcfe2532cb8f7` | yes |
| `electrolysis2d` | `PRESETS['electrolysis_bubble_2D']` | 100 | 1 / 6 | 1 / 6 | `ee7703599b998e19` | yes |
| `shearing2d` | `PRESETS['shearing_plate_droplet_2D']` (`periodic`) | 10 | 1 / 6 | 1 / 6 | `738ca6522782702b` | yes |
| `pin_hydro3d` | the run of `TestColumn3D.test_drop_settles_over_40_acoustic_times` (preset, refinement 1, 40 t_ac) | 370 | 1 / 6 | 1 / 6 | `aeba50c43b6931a2` | yes |
| `pin_hydro3d_remap` | the run of `test_remap_arm_holds_the_3d_column` (remap arm, refinement 2, 2 t_ac) | 37 | 6 / 6 | 1 / 6 | `071d30c33e5249f6` | no (kinetic energy 0.5431445985762777 in all six before, ...776 after) |

The "after" column is the last `sweep all --procs 6 --jobs 12` on the
final tree: 32 cases, 192 runs, one digest per case, exit status 0. Its
output is `docs_temp/debug_session/laneT-sweep-all.json` (a local file:
`*.json` is git-ignored; the digests are the ones in the table).

`workers=2` and `backend='multiprocessing'` give the digest of the serial
run (`52a42ad941f29aed` for `droplet2d` at 40 steps, `607127079316a088`
for `hp3d_centred` at 100 steps).

### 3.2 The documented spreads, reproduced on the library before the lane

| run | documented | before, measured here | after |
|---|---|---|---|
| lane H, 3D centred on the `batch_e_star` cache, L 3, 600 steps: l2 end | 0.0811, 0.0821, and 0.1107 after other runs | 0.08211494566330206 (4 of 9), 0.08110864974622763 (2), 0.11070850411907882 (3: the run after two other runs and two with a fragmented heap) | 0.08211494566330206 (6 of 6) |
| same: largest radial velocity | 6.296e-03 in every process | 0.00629618754380668 or 0.006318123280976656 | 0.00629618754380668 |
| lane H, centred on the `p_ij` ring: l2 end / radial velocity | 0.056566 or 0.056561 / 3.65e-04 or 1.26e-05 | 3 digests in 9; l2 0.05656136676066495 or ...496, radial velocity 1.2583750357323547e-05 or ...349e-05 | 0.05656136676066496 / 1.2583750357323349e-05 (6 of 6) |
| lane P, 3D remap arm, refinement 2, 40 t_ac: largest max\|u\| of the last tenth of the run (3.95 t_ac; lane P took 36 to 40 t_ac) | 1.718e-03 / 1.728e-03 / 1.728e-03 | 1.7178e-03, 1.7311e-03, 1.7276e-03, 1.7219e-03 (4 of 4 differ) | 1.7288121037225635e-03 (6 of 6) |
| same: kinetic energy at 40 t_ac | | 2.0212e-05 to 2.0725e-05 | 2.054884654840061e-05 |
| lane P, 3D fixed connectivity, refinement 2: agreement between processes | 4e-09 relative | kinetic energy after 150 steps 4.39454523986159e-04 to ...415e-04 (6e-13; 8 of 8 differ) | 4.394545239861678e-04 |
| 3D remap arm, 150 steps: kinetic energy | two digits after 10 t_ac | 2.770e-04 to 2.828e-04 (8 of 8 differ) | 2.804148489815872e-04 |

The lane H ring arm did not show its second value (3.65e-04) in these 9
interpreters; it differed in the last digits only.

### 3.3 Attribution: one change at a time

Libraries built from the "before" export plus ONE changed file:

| library | `hp3d_centred` | `hydro3d_remap` | `hydro3d` | `droplet2d_adaptive` |
|---|---|---|---|---|
| before | 6 / 8 | 8 / 8 | 8 / 8 | 6 / 6 |
| + `_compute_dual.py` | 1 / 6 | 1 / 6 | 1 / 6 | 6 / 6 |
| + `remesh/_driver.py` (`_edge_list`) | | | | 1 / 6, digest `0956407aef0625e0` (that of the full fix) |
| + `remesh/_quality.py` (`iter_triangles_2d`) | | | | 6 / 6 |
| all | 1 / 6 | 1 / 6 | 1 / 6 | 1 / 6 |

(The row of `_compute_dual.py` was measured with the first version of
the fix and again with the final one: same counts, same digests.)

`iter_triangles_2d` and `density_diffusion_step` do not change the digest
of any detector case. Their address dependence is real at the level of
the function (`test_triangles_are_listed_in_cache_order` and
`TestDensityDiffusionPairOrder` fail on the library before the lane: the
triangle order, and the last bits of the returned `sum_dm`); in a run the
mass increment of a diffusion step is about 1e-6 of the mass, so its last
bit almost never reaches the mass.

### 3.4 One interpreter

`hydro3d_remap`, 4 steps, run, then `droplet2d` (10 steps), then run
again twice: before `d2217de26fbecfe5`, `6726e53229ed65c2`,
`d67ffbe66ecc1cff`; after `ace32922489c3134` three times.

### 3.5 Runs outside the detector

| run (from a scratch copy of the runner, final tree) | result |
|---|---|
| full `oscillating_droplet_2D.py` | every key of `score.json` equals `baselines/baseline_oscillation.json` (l2 0.17479361640597058, tail 0.9998967874595965), `methods` block equal |
| full `oscillating_droplet_3D.py` | every key equals `baselines/baseline_oscillation_3d.json` (l2 0.24811340819647862, tail 0.08409976059818802) |
| full `oscillating_droplet_3D.py --retopo delaunay` | l2 1.5244561707801316, tail 0.47066568050584034 in two processes before and three after (the documented value of the preset note) |
| `capillary_rise_2D_dynCA.py --smoke` (section 9) | before: 3 final states in 3 interpreters; after: 1 in 3 |

## 4. The audit: every place on the path

"Address-free" below means: the order is a function of coordinates and of
the insertion history, both of which are the same in every process.

| place | order it uses | verdict |
|---|---|---|
| `v.nn`, `v.vd`, their intersections; `HC.interface_vertices` and the other vertex sets | set of vertices; hash = `hash(v.x)` | address-free (test `test_neighbour_sets_iterate_in_the_same_order`, also on the old library) |
| `HC.V`, `HC.Vd` | `OrderedDict` by coordinate key, insertion order; a moved vertex is re-inserted at the end | address-free |
| `compute_vd` 3D simplex-aware: boundary face barycentre | was `id()` order of the face | **fixed** (the root of the 3D spread) |
| `compute_vd` 2D simplex-aware: boundary edge midpoint | midpoint commutative; merge neighbourhood of the lower-`id()` endpoint | **fixed** (no effect on any run measured) |
| `compute_vd` n-D simplex-aware: boundary face | `id()` order | **fixed** (no ddgclib caller) |
| `compute_vd` batch paths (`backend=`), legacy 1-skeleton paths | simplex cache order; `id()` frozensets are dictionary keys only | address-free |
| `rebuild_simplex_cache_2d/3d`, `connect_and_cache_simplices` | `HC.V` order (lanes S, A) | address-free |
| `boundary_from_simplices` | dictionary keyed by sorted ids, insertion order = simplex order; returns a vertex set | address-free |
| fan walk (`_walk_fan_3d`, `e_star`, `batch_e_star`) | starts at `next(iter(v_i.vd & v_j.vd))` | address-free once the dual positions are |
| `_dual_area_vector_3d_p_ij` | ring from `list(shared_vd)[0]`, common neighbours in set order | address-free; decided by a geometric tie at boundary edges (section 6) |
| `hyperct.remesh._driver._edge_list` | was oriented by `id()` | **fixed** |
| `hyperct.remesh._quality.iter_triangles_2d` | was `id()` order | **fixed** |
| `hyperct.remesh._operations_2d` (`_one_ring_area`, collapse check) | `id()` pairs as de-duplication keys | address-free |
| `_dual_cell_polygon_2d_walk`, `get_edge_apex_map` | `id()` as dictionary keys | address-free |
| `MultiphaseSystem` (phase vote, incidence, refresh), `extract_interface` | dictionaries keyed by coordinate frozensets or `id()`; tie rule of lane L | address-free |
| `mass_redistribution` snapshots | dictionaries keyed by `id()`, read in `HC.V` order | address-free; see the limit on recycled ids in section 10 |
| `density_diffusion_step` | was `id()` order of the pair | **fixed** |
| `retopologize_material_delaunay`, `bare_dual_refresh` | `id()` frozensets as keys and membership | address-free |
| `PeriodicInletBC`, buffered inlet and outlet | dictionaries keyed by `id()` that hold the vertex | address-free |
| `delaunay_with_ghosts` | set of integer tuples | address-free |
| scipy `Delaunay` | Qhull without joggle (`QJ` is not used anywhere) | deterministic |
| `workers > 1`, `backend='multiprocessing'` | `Pool.map`, ordered | equal to serial (section 3.1) |
| global random state | `np.random` appears in plotting and docstrings only | not on the path |

Left as they are, not on a dynamic path:
`ddgclib/geometry/periodic.py:_fixup_periodic_duals` (orders pairs by
`id()`; no caller), `ddgclib/data/_conservation.py:_edge_length_extrema`
(`id()` to visit each edge once; min and max do not depend on the order),
`hyperct/_vis_disc.py` (disconnectivity graph), the plotting helper
`cases_dynamic/oscillating_droplet/_diagnostic_plot.py`.

## 5. Pins

### 5.1 No existing pin moved

- Fast suite and slow battery pass on the pins as they were (section 8).
- Every detector case with one final state before the lane has the same
  digest after it (22 cases, section 3.1).
- Full 2D and 3D droplet runs: every key of the baselines (section 3.5).
- The peak pins of the two 3D hydrostatic runs are unchanged to the last
  digit (`PIN_3D_UMAX_PEAK` 0.08910127097486757, `PIN_3D_REMAP_UMAX_PEAK`
  0.15658060026054665). The remap run was process-dependent before (6
  final states in 6 interpreters after its 37 steps), but only below the
  printed digits: its peak was the same in all six and its final kinetic
  energy 0.5431445985762777 against ...776 now.

### 5.2 Numbers that were a spread: old spread, new value, perturbation

`sweep CASE --procs 1 --perturb 1e-15 --n-perturb 8`: the interior
vertices are shifted by `1e-15 * scale * r`, 8 seeds.

| run, quantity | before (range over interpreters) | after | range under the 1e-15 shift |
|---|---|---|---|
| `hydro3d` 150 steps, kinetic energy | 4.39454523986159e-04 to ...415e-04 (rel 6e-13) | 4.394545239861678e-04 | 4.39454774e-04 to 4.39455477e-04 (rel 2.2e-06) |
| `hydro3d_remap` 150 steps, kinetic energy | 2.770e-04 to 2.828e-04 | 2.804e-04 | 2.760e-04 to 2.865e-04 (2.2 %) |
| `hp3d_centred` 300 steps, l2 | 0.10146 to 0.10441 | 0.10413 | 0.10144 to 0.10925 (4.9 %) |
| `hp3d_centred_laneH` 600 steps, l2 | 0.0811, 0.0821, 0.1107 | 0.0821 | 0.0793 to 0.1693 |
| `droplet2d_adaptive` 60 steps, KE | 1.14667e-05 to 1.14826e-05 | 1.14702e-05 | 1.0045e-05 to 1.1473e-05 (12 %) |
| `shearing3d` 5 steps, KE | 6.2579e-06 or 6.2763e-06 | 6.2579e-06 | 6.319e-06 to 6.504e-06 (1 to 4 %) |

In every row the new value lies inside the old range or within the
perturbation range of it: the old spread was round-off amplified by the
run, nothing else. The perturbation range is far wider than the old
spread on fixed connectivity (2.2e-06 against 6e-13): a 1e-15 shift flips
the tie of section 6, the address order only moved the last bit of a few
boundary face barycentres.

### 5.3 New end-of-run pins (task 4)

Rule 8 had restricted the 3D pins of `test_case_hydrostatic.py` to early
peaks. Both runs are bit-identical in every interpreter now, so their
end values are pinned as well (the peak pins stay):

| pin | run | value | 1e-15 shift, 8 seeds (max relative deviation) | tolerance |
|---|---|---|---|---|
| `PIN_3D_KE_40` | `PRESETS['hydrostatic_3D']`, refinement 1, 40 t_ac (370 steps), kinetic energy of the free vertices at the end | 6.206365156298652e-06 | 6.1e-12 (peak 1.7e-13) | rel 1e-9 |
| `PIN_3D_REMAP_KE_END` | `remap_arm(PRESETS['hydrostatic_3D'])`, refinement 2, 2 t_ac (37 steps), same quantity | 0.5431445985762776 | 6.9e-04 (peak 1.25e-03) | rel 1e-6 |

The first is a robust number (and this refinement 1 run had one final
state in 6 interpreters before the lane too; rule 8 excluded it as a late
3D number). The second is tie-decided, as the peak pin
next to it already was: any change of a summation order upstream may move
both by up to the perturbation range; that is a re-pin with this table as
the yardstick, not a regression. A later reconnecting 3D value is not
pinned because of its cost (40 t_ac at refinement 2 takes 6 minutes); the
slow determinism tests assert equal digests instead, and the values are in
section 3.2. The 3D Hagen-Poiseuille pin (`PIN_3D_L2`, 300 steps) and the
3D droplet pins were end-of-run values of deterministic runs already.

## 6. Finding beyond the lane: the `p_ij` area of a boundary edge (NOT fixed)

Found through the perturbation runs: on FIXED connectivity a 1e-15 shift
moved the refinement 2 column by 2e-06 after 150 steps, and by 6e-12 at
refinement 1. One step after the shift the dual area vectors of 16 of
the 1183 (vertex, neighbour) pairs differed between the two runs, the
twelve largest by 18 to 37 %, all of them edges between two free-surface
vertices.

Mechanism (`ddgclib/operators/stress.py:_dual_area_vector_3d_p_ij`). The
ring of dual vertices shared by the two endpoints is walked, and between
two consecutive ring vertices the routine inserts "the face barycentre
`(x_i + x_j + x_k) / 3` nearest to their midpoint". For an interior edge
the ring holds tetrahedron barycentres only. For a boundary edge it also
holds the edge midpoint and the two boundary-face barycentres. Between
the midpoint `M` and a boundary-face barycentre `F` the right insertion
is nothing (or `F` itself, a duplicate point); the rule inserts the
barycentre of an interior face instead whenever the apex `x_c` of that
face is not farther, which is the case when `x_c` lies inside the sphere
with diameter (`M`, apex of the boundary face). On the builder lattice
`x_c` lies exactly on that sphere: both candidates are at distance
0.046585, and the last bit decides. The edge from (0.5, 0.5, 1) to
(0.75, 0.5, 1) after one step, in the run and in the run shifted by
1e-15: `[1.0417e-02, 0, 0]` against `[6.94e-03, 0, 1.74e-03]`, with one
and with two spurious vertices in the polygon (an edge gets 0, 1 or 2).

Measured against the per-tetrahedron sum (one quadrilateral edge
midpoint, face barycentre, tetrahedron barycentre, face barycentre per
tetrahedron on the edge; `test_determinism.py:_per_tetrahedron_area`) on
the builder mesh of the 3D column:

| edges seen from the free vertices | refinement 1 | refinement 2 |
|---|---|---|
| at least one interior endpoint | 95, exact (3.6e-16) | 1127, exact (8.3e-16) |
| both endpoints on the boundary | 8 at the single free-surface vertex, exact (5.0e-16) | 56 at the 9 free-surface vertices, 30 of them off, by 0.183 to 0.373 |

Who is affected: every 3D run that evaluates `dual_area_vector` on an
edge between two boundary vertices of which at least one is integrated:
`hydrostatic_3D` (free surface), and the integrated hull vertices of the
3D Hagen-Poiseuille ring arm (plausibly the origin of the two sizes of
its radial velocity in lane H; not verified). The 3D droplet and every
case with a frozen hull are not reached by this mechanism: their boundary
vertices are not integrated, and they read the `batch_e_star` cache.

Not fixed here: it changes the force on every 3D free surface, the lane P
pins and every 3D column number of lane P, and the registry already names
the fix (`edge_area_source='p_ij_simplex'`, lane J: faces read from
`HC._simplices`; the per-tetrahedron sum of the test is that
construction). The strict xfail
`TestFreeSurfaceEdgeAreaIsDecidedByATie.test_edges_on_the_boundary_are_exact`
turns into an unexpected pass when it lands.

## 7. Cost

Sequential runs on a machine with other load, alternating the two
libraries, 3 repeats, wall time of `run` (setup included), minimum:

| case | steps | before: min (all three), s | after: min (all three), s | after / before |
|---|---|---|---|---|
| `droplet2d` | 100 | 3.26 (3.32, 3.31, 3.26) | 3.21 (3.24, 3.23, 3.21) | 0.985 |
| `droplet3d` | 40 | 17.76 (17.86, 17.76, 17.86) | 17.59 (17.59, 17.59, 17.77) | 0.990 |
| `droplet3d_delaunay` | 40 | 18.19 (18.19, 18.19, 18.39) | 18.14 (18.37, 18.14, 18.34) | 0.997 |
| `hp3d` | 300 | 16.48 (16.62, 16.48, 16.71) | 16.44 (16.69, 16.70, 16.44) | 0.998 |
| `hydro3d_remap` | 40 | 19.54 (19.64, 19.63, 19.54) | 19.52 (19.84, 19.59, 19.52) | 0.999 |

No cost: the differences are inside the scatter of the repeats, with a
gain of 1.5 % on the 2D droplet. (A first round of this table showed the
new library 1.5 to 5.5 % slower. That was the measurement: the old
library was imported before the timer started, through `--lib`, the new
one inside the timed call. The detector now imports both before it
starts the timer; the table is with `--lib` on both sides.)

Kernels (minimum over repeats, two processes each): `iter_triangles_2d`
on the 2D droplet mesh of 311 vertices 0.85 to 0.88 ms before, 0.58 to
0.59 ms after; `compute_vd` in 2D 19.4 to 20.1 ms before, 19.7 to 20.0
after; in 3D (472 vertices, 2520 tetrahedra) 142.9 to 146.4 ms before,
142.8 to 144.6 after. Nothing is sorted that was not sorted before; the
`rank` dictionary of `iter_triangles_2d` replaces a set of the same size.

## 8. Tests

- `hyperct/tests/test_deterministic_order.py` (10): 3D and 2D Delaunay
  complexes built three times with the heap fragmented in between
  (asserted: the builds sit at other address orders). 7 fail on hyperct
  `476c289`.
- `ddgclib/tests/test_determinism.py`:
  - `TestReconnecting3DRunIsDeterministic` (fast, 3): `hydro3d_remap`, 4
    steps, in 4 fresh interpreters (one plain, three fragmented): equal
    digests and scalars; the same case twice in the test interpreter
    around an unrelated 2D run: both equal to the fresh interpreters; the
    fragmentation does change the address order. On the library before
    the lane the 4 steps gave 8 digests in 8 interpreters.
  - `TestDensityDiffusionPairOrder` (fast, 1), fails on the old library.
  - `TestFreeSurfaceEdgeAreaIsDecidedByATie` (fast, 1 + 1 strict xfail):
    section 6.
  - `TestEveryProcessDependentArmIsDeterministic` (slow, 6):
    `hp3d_centred` 300 steps, `hydro3d_remap` 40, `hydro3d` 40,
    `shearing3d` 5, `droplet2d_adaptive` 60, `droplet3d_delaunay` 20,
    each in 4 interpreters (plain, after two other runs, two fragmented).
- `test_case_hydrostatic.py`: the two pins of section 5.3.
- Suites on the final tree: ddgclib fast 1141 passed, 12 skipped, 4
  xfailed (1136 / 12 / 3 at the start: +5 and the strict xfail of section
  6), 111 s; slow 32 passed, 1 xfailed (26 at the start, +6), 286 s with
  other jobs on the machine (the six determinism tests add about 90 s to
  the 190 s of the battery); hyperct 326 passed (316 + 10).

No linter is installed in the `ddg` environment; the changed files were
compiled and checked for unused names by hand.

## 9. Note for the dynCA lane (`cases_dynamic/capillary_rise*`)

Nothing in those directories was edited. These library functions, which
`capillary_rise_2D_dynCA.py` and `src/_setup_dynca.py` call, changed
their order:

| function | change | effect on a dynCA run |
|---|---|---|
| `hyperct.remesh.adaptive_remesh` (through `_edge_list`) | the edge `(v_i, v_j)` is oriented by `HC.V` order, not by `id()`. Split, collapse and flip sweeps visit the same edges in the same sequence, but a split copies the carried attributes of `v_i` to the midpoint and a collapse keeps the object `v_i` (moved to the midpoint, so re-inserted at the end of `HC.V`) and removes `v_j` | the remeshed complex is now one function of the mesh; before, it differed from process to process. Numbers move by more than round-off |
| `hyperct.remesh._quality.iter_triangles_2d` (also behind `ddgclib.multiphase.iter_top_simplices` in 2D and `mesh_quality_histogram`) | triangle order and vertex order within a triangle follow `HC.V` | order of dictionary insertions; a mean over triangles in the last bits |
| `hyperct.ddg.compute_vd` (2D simplex-aware path) | the two vertices of a boundary edge in triangle order | none unless the local dual merge triggers |
| `ddgclib.operators.stabilisation.density_diffusion_step` (`--delta-diff`) | each pair from its first endpoint in the order of the vertex list | last bits of the mass increments and of `sum_dm` |
| `hyperct.ddg.rebuild_simplex_cache_2d`, `connect_and_cache_simplices`, `boundary_from_simplices` | unchanged | |

Measured with scratch copies of the runner as it stood in the working
tree on 2026-10-02 (`--smoke`, no other option; water, R 0.5 mm):

| library | interpreters | steps | normalised L2 of the height | final state file |
|---|---|---|---|---|
| before | 3 | 8607 / 8450 / 8668 | 0.5792 / 0.6441 / 0.5674 | 3 different |
| after | 3 | 8036 / 8036 / 8036 | 0.4860282924783144 in each | identical (sha256 `5cbea1669c6f...`) |
| before + `_edge_list` only | 2 | 8036 / 8036 | 0.4860282924783144 in each | identical, and equal to the "after" state |

The orientation of the remesh edges is the whole dependence of this run:
with that one change the old library gives the "after" state. So every
dynCA number measured before this lane is one realisation of a spread of
that size (this closes the open item of lane L, "the dynCA
smoke run is not reproducible between processes, not traced"), and the
runs must be re-measured once on the new library; after that a single
process is enough. The shipped option sets (`--surface-bc yl`,
`--contact-mode mobility`, `--delta-diff 0.05`) were not run.

## 10. Known limits

- The guarantee is for one machine and one software environment. Another
  numpy, scipy (Qhull) or BLAS build, another libm, or `backend='gpu'`
  (16th digit, lane H) can change last bits, and a run amplifies them.
- Deterministic is not robust. Runs that reconnect a structured mesh, or
  have a 3D free surface on the `p_ij` ring, are decided by ties; quote
  their late numbers with the range of a 1e-15 shift (section 5.2).
- `iter_triangles_2d` and `HC.V` order: a vertex that is moved is
  re-inserted at the end of the cache, so the enumeration order changes
  from step to step with the motion. It is a function of the run, not of
  the geometry alone.
- Snapshots keyed by `id()` (`mass_redistribution`, the displacement gate
  `_retopo_prev_positions`): a vertex removed inside a retopology call
  could hand its address to a vertex created later in the same call, and
  the new vertex would then read the stale entry. Inside `_retopologize`
  the list of vertices taken at entry keeps every removed vertex alive
  until the call returns, so collapses and splits of one `adaptive_remesh`
  cannot collide; the exception is `merge_cdist` together with
  `connectivity='adaptive'` on a complex without a simplex cache (the
  merged vertices are released before the remesh). No preset uses that
  combination; not fixed, not reproduced.
- `_fixup_periodic_duals` is dead and still ordered by `id()`.
- The `--perturb` shift also re-inserts the moved vertices at the end of
  `HC.V`, so it perturbs the summation orders together with the
  coordinates. Both are round-off sized; they were not separated.
- `electrolysis`, `dam_break_3D` and `shearing3d` are covered by short
  runs only (30, 5 and 5 steps).

## 11. DO-NOTs (measured)

- Do not order anything on the path by `id()`, and do not iterate a set
  of objects that hash by address. Use `HC.V` order, simplex order, or
  the coordinate-hashed vertex sets.
- Do not sum the vertices of a face in the order of a de-duplication key.
  The key may be sorted ids; the sum must take the vertices from the
  simplex.
- Do not read "identical in N processes" as "robust": the 3D remap peak
  was identical in six processes before the lane and moves by 1.25e-03
  under a 1e-15 shift.
- Do not compare an `adaptive` or dynCA number from before 2026-10-02
  with one from after it as a regression: the old one is one draw.
- Do not fix the boundary-edge `p_ij` polygon without re-measuring the
  3D column of lane P and re-pinning its 3D pins.
- Do not use `sweep` without the fragmented interpreters to declare a
  path address-free: plain interpreters agree far too often.

## 12. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
D=cases_dynamic/diagnose_determinism.py

$PY $D list
$PY $D sweep all --procs 6 --jobs 12              # about 20 min
$PY $D sweep hp3d_centred_laneH,hp3d_ring_laneH --procs 6
$PY $D sweep hydro3d_remap_40tac --procs 3        # 6 min each

# before / after: export the old commits and point --lib at them
mkdir -p /tmp/old && git archive 5683e78 ddgclib | tar -x -C /tmp/old
git -C ../hyperct archive 476c289 hyperct | tar -x -C /tmp/old
$PY $D sweep hp3d_centred,hydro3d_remap,hydro3d --procs 8 --lib /tmp/old
$PY $D sweep hydro3d_remap,hydro3d --steps 4 --procs 8 --lib /tmp/old

# round-off amplification of a run
$PY $D sweep pin_hydro3d,pin_hydro3d_remap,hydro3d --procs 1 \
    --perturb 1e-15 --n-perturb 8

# tests
$PY -m pytest ddgclib/tests/test_determinism.py -q -p no:cacheprovider
$PY -m pytest ddgclib/tests/test_determinism.py -q -m slow -p no:cacheprovider
(cd ../hyperct && $PY -m pytest hyperct/tests/test_deterministic_order.py -q)
```

The full droplet and dynCA runs of sections 3.5 and 9 were made from
scratch copies of the runners (the runners write into their case
directory): copy `cases_dynamic/__init__.py`, the case package and its
`src/` into an empty directory, set `PYTHONPATH` to the repository root
(or to the export of the old commits) and run the copy. The dynCA copy
also needs a link `data` to the repository's `data/` next to its
`cases_dynamic/`.
