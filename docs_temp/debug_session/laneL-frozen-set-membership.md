# laneL: frozen vertices by wall membership, not by hull membership

Date: 2026-10-01. Closes audit finding F10 C1 (wall collapse by hull
re-tagging), C2 (inlet ghost loses a vertex on reset) and the key-collision
part of C3 (injection on an occupied key). Evidence base:
`docs_temp/11_dynamics_audit_2026-09-25.md` F10,
`docs_temp/audit_2026-09-25/bcs_partA.md` A.0 and A.3.

This log describes the state after fix round 1 (independent review, same
day). What the round changed is listed in section 11; numbers that the
round replaced are marked there with their old values.

## 0. Verdict

- New explicit method axis `frozen_set` (`'hull'` default, `'membership'`
  opt-in). Under `'membership'` the set of frozen vertices `bV` is
  persistent and the tag `v.boundary` alone follows the topology.
- `Hagen_Poiseuile_2D.py` runs its 3000 steps through the preset
  `hagen_poiseuille_2D` with the walls intact: 62 wall vertices at the
  start, 62 still frozen, 0 moved. The same run with
  `.replace(frozen_set='hull')` releases the walls at step 1262: 2 of 62
  still frozen, 60 moved, largest displacement 3.457.
- hyperct `HC.V.move` refuses a move onto the key of another vertex
  (`VertexCollisionError`); `on_collision='evict'` is the old behaviour;
  new `HC.V.move_all` for shifts and rescales.
- `PeriodicInletBC`: the ghost keeps all its vertices on reset (13, was
  12), and an injection on an occupied key leaves the resident vertex
  alone.
- Every pinned number is bit-identical (fast and slow battery green). The
  default path of `_retopologize` is unchanged.
- Two presets switched to `'membership'`: `hagen_poiseuille_2D` (the old
  rule collapses) and `dam_break_2D` (bit-identical on the shipped run;
  in the ejection configurations the walls stay in place, the run is
  lost in both arms).
- `'membership'` is implemented for `connectivity='delaunay'` only. With
  `'adaptive'` it raises (fix round 1; it was accepted and wrong before).
- Fix round 1 also traced and removed a process dependence that predates
  the lane: a tied simplex phase vote was decided by memory addresses.
  The flaky shearing-plate bit-identity test and the non-reproducible
  dam-break numbers after the ejection both came from it.
- A finding that reaches beyond the lane: the pinned droplet meshes are
  missing outer vertices, the box corner included (section 5).

Not solved here: impenetrability (section 7), the Poiseuille profile
(lane H), the sliver ejection of the dam break.

## 1. What changed and where

| file | change |
|---|---|
| `hyperct/_vertex.py` (hyperct working tree, uncommitted) | `VertexCollisionError`; `move(v, x, on_collision='raise')`; `move_all(moves)`, which checks everything before it changes anything; the re-keying body is now `_rekey` |
| `hyperct/tests/test_vertex_move.py` (new) | 15 tests |
| `hyperct/tests/test_remesh.py` | the dropped-vertex test asks for `on_collision='evict'` |
| `ddgclib/dynamic_integrators/_integrators_dynamic.py` | `_retopologize(..., frozen_set='hull')` and `_retopologize_multiphase(..., frozen_set='hull')`; three branches, default path untouched; `'membership'` with `remesh_mode='adaptive'` raises in both |
| `ddgclib/multiphase.py` | `assign_simplex_phases_from_vertices`: a tied vote goes to the lower phase ID (fix round 1, section 11.2) |
| `ddgclib/geometry/domains/_disks.py`, `_spheres.py`, `ddgclib/geometry/_complex_operations.py`, `ddgclib/geometry/_parametric_surfaces.py` | whole-mesh shifts, rotations and rescales use one `move_all` (`disk`, `annulus`, `ball` with a centre, `translate`, `translate_surface`, `rotate_surface`, `scale_surface`) |
| `ddgclib/methods/_axes.py` | axis `frozen_set` (group connectivity) with evidence |
| `ddgclib/methods/_config.py` | field `frozen_set`, validation, builder plumbing |
| `ddgclib/methods/_presets.py` | `hagen_poiseuille_2D` and `dam_break_2D` on `'membership'`, notes |
| `ddgclib/_boundary_conditions.py` | `PeriodicInletBC._shift_ghost` (one `move_all`), occupied-key injection; `MeshAdvancer.step` advects with one `move_all` |
| `ddgclib/geometry/domains/_multiphase_droplet.py` | the two box shifts ask for `on_collision='evict'` (pin identity, section 5) |
| `cases_dynamic/electrolysis_bubble/src/_setup.py`, `cases_dynamic/shearing_plate_droplet/src/_setup.py` | same, one loop each |
| `cases_dynamic/Hagen_Poiseuile/src/_setup.py` | `setup_poiseuille_2d_lagrangian`, `wall_snapshot`, `wall_report` |
| `cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py` | thin runner on that setup; `--headless`, `--steps`, `--frozen-set`, `--workers`, `--tag`; writes `wall_report.json`; refreshes the duals before the post-processing; puts the repository root on `sys.path` like the other runners |
| `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py` (new) | A/B driver: hp2d, dam_break_2D, electrolysis_2D / _3D, droplet_2D |
| `cases_dynamic/Hagen_Poiseuile/README.md` (new) | run instructions, outputs, status |
| `cases_dynamic/oscillating_droplet/src/_metrics.py` | `diff_baselines` reads an axis missing from an old score as that axis' default |
| `ddgclib/tests/test_frozen_set.py` (new) | 31 tests |
| `ddgclib/tests/test_boundary_conditions.py` | `TestPeriodicInletBC`, 4 tests |
| `ddgclib/tests/test_domains.py` | `TestCentreShiftKeepsEveryVertex`, 4 tests |
| `ddgclib/tests/test_multiphase.py` | `test_tied_simplex_vote_goes_to_the_lower_phase` |
| `ddgclib/tests/test_methods.py` | `test_dam_break_bindings`: the hand-written partial is the `frozen_set='hull'` arm |
| `METHODS.md`, `DEVELOPMENT.md`, `debugging_plan.md`, audit doc F10 | documentation |

No integrator signature changed. As with the single-phase remap of lane R,
the value rides on a `functools.partial` of the library retopology
function, and the integrator forwards its retopology kwargs to it by name.

Exception to protocol rule 2 (new code paths go to `ddgclib/methods/_retopo.py`,
not to `_integrators_dynamic.py`): the brief named `_retopologize` as the
place of the defect and asked for the minimal change, and the policy is
three branches inside the step that rebuilds `bV`, not a retopology
function of its own. Lane R made the same exception for the single-phase
remap.

## 2. The policy and the three decisions

`_retopologize` did two things with one set. The topological boundary `dV`
of the new connectivity gave the tag `v.boundary` (which `compute_vd`
needs to build half cells), and the same set, narrowed by
`boundary_filter`, replaced `bV`, the vertices the integrators skip. A
vertex was frozen because it was on the hull.

Under `frozen_set='membership'` step 6 of `_retopologize` is

    bV <- { v in bV : v is still in HC.V and (no filter or filter(v)) }

and nothing else changes: the tag still comes from `dV`. With
`skip_triangulation=True` (the first stage of the multiphase remap) the
boundary is read from the kept connectivity (`boundary_from_simplices`,
else `HC.boundary()`) and no longer carried in `bV`, because `bV` is not
the boundary any more.

The three questions of the brief:

1. A hull vertex that is not a wall (inlet, outlet, free surface, a vertex
   that left through a wall): not a member, so it is integrated. It keeps
   `v.boundary = True` and a half cell. `boundary_filter` is how a runner
   that starts from "the whole hull" (HP2D) drops the inlet and outlet
   columns on the first call.
2. A vertex that reaches a wall: the retopology does not capture it. Under
   `'hull'` a vertex on the hull that met the filter was frozen at the next
   retopology; under `'membership'` that is the decision of a BC that holds
   `bV`. `PositionalNoSlipWallBC(bV=bV)` adds what meets its criterion, and
   the addition now persists (it used to last until the next rebuild).
   Consequence: the wall BC must run AFTER the inlet BC, else a wall-row
   vertex injected by the inlet is integrated for one step, leaves the
   wall line and is never captured (measured: one such vertex 0.05 below
   the wall inside the channel after 300 steps of the short case). The
   setup function orders the BCs accordingly.
3. A wall vertex that the hull no longer contains: stays in `bV`, does not
   move, and gets `v.boundary = False` and a closed dual cell, which is
   what its connectivity is.

Rejected variant (not implemented, not measured): capture every hull
vertex that passes the filter, i.e. the old capture without the release.
It would make the BC order irrelevant, but with no filter it freezes
every vertex that ever touches the hull, so an escaped vertex becomes a
wall for good. One rule (members only) is easier to state.

Where it applies. `SolverMethods` accepts `'membership'` for
`connectivity='delaunay'` and raises for the rest: `dual_only`,
`dual_only_bare` and `frozen` never rebuild the hull, so their `bV` is
persistent already; `periodic`, `delaunay_material` and `custom` have
their own boundary code; `adaptive` is refused for the reason below. 1D is
excluded.

`adaptive` (fix round 1). The first version accepted it without a
measurement. `hyperct.remesh` protects vertices by the topological tag
`v.boundary` and knows nothing of `bV`, which breaks the policy twice:

| measured on `rectangle(L=2, h=1)`, walls y = 0 and y = 1 | hull | membership |
|---|---|---|
| one `_retopologize(remesh_mode='adaptive', L_min=0.05, L_max=0.3, 2 iterations)`, refinement 2: wall-line vertices frozen | 18 of 18 | 10 of 18 (the 8 created by wall-edge splits are not members) |
| `SolverMethods(connectivity='adaptive').integrate`, uniform acceleration (0, 1), 5 steps of 1e-2: vertices left on y = 0 / y = 1 | 9 / 9 | 5 / 5 (the split vertices are integrated and leave the wall) |
| `adaptive_remesh` on a mesh whose 7 inner bottom-wall members are off the hull (refinement 3, one vertex 1e-3 below the wall): members removed / moved / largest displacement | not applicable | 1 / 6 / 7.3e-02 (`L_min=0.05, L_max=0.3`); 0 / 7 / 8.6e-02 (defaults); 5 / 2 / 1.7e-01 (`L_min=0.2, L_max=0.6`) |

The first two rows are the reviewer's probes, reproduced; the third is
`adaptive_offhull.py` of this round (scratch). Letting a split vertex
inherit membership would repair the first mechanism only: a member that
is off the hull is collapsed and Laplacian-smoothed like an interior
vertex. A correct version needs a constrained-vertex set inside
`hyperct.remesh`, which is its own piece of work. So the combination
raises, in `SolverMethods` and at the top of `_retopologize` and
`_retopologize_multiphase` (before the first remap stage touches
anything); `test_adaptive_remesh_is_refused_before_anything_changes`.
With `skip_triangulation=True` the remesh mode is not used and nothing
is refused.

## 3. Measurements

Every arm is a preset or `preset.replace(...)`.

### 3.1 The reproducer (task 1)

`setup_poiseuille_2d_lagrangian(L=2)`, the BC stack and parameters of the
shipped runner, `PRESETS['hagen_poiseuille_2D'].replace(workers=None,
frozen_set=...)`, `dt = 0.05`, 300 steps, 1.4 s per arm
(`test_frozen_set.py::TestHagenPoiseuille2D`,
`diagnose_frozen_set.py hp2d`).

| | hull | membership |
|---|---|---|
| wall vertices at the start (builder) | 10 | 10 |
| still frozen at the end | 2 (the inlet corners) | 10 |
| moved | 8 | 0 |
| vertices on the wall lines, minimum over the run | 2 | 12 |
| first step with a vertex outside the channel box | 249 | 249 |
| first step the frozen walls change | 250 | never |
| largest wall displacement (walls incl. the two inlet duplicates) | 6.184e-02 | 0.0 |

Mechanism, as the audit described it: two outlet buffer vertices keep the
wall-normal part of their frozen velocity, drift past the wall lines at
`x > L`, and from the next retopology the collinear wall vertices are
inside the hull.

Before any edit the scratch replica of the old runner (old BC order, old
inlet) reproduced the audit number exactly: L = 5, dt 0.01: 23 wall
vertices until step 1247, 2 at step 1248. The collapse step barely
depends on L (L = 2: 1246 to 1249, L = 3: 1226).

### 3.2 The shipped case (task 4)

`python Hagen_Poiseuile_2D.py --headless` (preset, workers 20, L = 15,
dt 0.01, 3000 steps, 3 min 47 s) and
`--headless --frozen-set hull --tag hull`. Numbers from
`results/wall_report.json` and `results/hull/wall_report.json`; both runs
were made twice and gave the same numbers.

| | membership (preset) | hull |
|---|---|---|
| wall vertices at the start | 62 | 62 |
| still in the complex | 62 | 61 |
| still frozen | 62 | 2 |
| moved | 0 | 60 |
| largest displacement | 0.0 | 3.456657916620642 |
| vertices on the wall lines: first / min / end | 64 / 64 / 64 | 64 / 2 / 2 |
| first step the count drops | never | 1262 |
| vertices at the end / frozen | 157 / 64 | 161 / 2 |
| vertices outside 0 <= y <= D at t = 30 | 3 (1 inside x <= L) | 36 (27 inside x <= L) |
| U_max at x = L / 2 (analytical 0.2) | 0.291212 | 0.337017 |

The collapse step is 1262 here, not the audit's 1248, because this run
already has the lane's BC order and inlet fixes.

Not validated, and not claimed: the profile (lane H). The integrated
pressure L2 of the membership run is 0.1227, the force balance max |F|
0.305.

### 3.3 Other cases that freeze by hull membership (task 5)

`cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py <case>`; the final
state digest is a sha256 over sorted (x, u, m).

| case, preset | horizon | hull | membership | verdict |
|---|---|---|---|---|
| `dam_break_2D`, shipped (alpha_art 0.3, refine 3) | 1585 steps, dt 1.262e-04 | KE_max (sampled) 1.036626e-03, walls 32 / 32 frozen | same, final state bit-identical | neutral |
| `dam_break_2D`, alpha_art 0.2 | 1585 steps planned; first vertex outside the tank at step 1267 in both arms, states bit-identical up to and including that step | walls released at step 1268, 1 / 32 frozen at the end, 32 moved; QhullError after 1280 steps (5 of 5 processes, digest `f36e005a21d74efd`) | walls 32 / 32 frozen, 0 moved; fluid blows up all the same; QhullError after 1303 steps (9 of 9 processes, digest `86002265325138f6`) | walls kept, run lost in both |
| `dam_break_2D`, alpha_art 0.5 to t = 0.45 s | 3566 steps planned; first vertex outside at step 3002 | walls released at step 3003, 0 / 32 frozen, 32 moved; QhullError after 3022 steps (3 of 3, `64e72018a75868db`) | 32 / 32 frozen, 0 moved; QhullError after 3030 steps (5 of 5, `7abc17e4e28c1b07`) | walls kept, run lost in both |
| `electrolysis_bubble_2D` | 6330 steps (shipped horizon) | KE_max 6.213015e-03 | bit-identical | neutral |
| `electrolysis_bubble_3D` | 300 steps | KE_max 1.874249e-12 | bit-identical | neutral |
| `oscillating_droplet_2D` (refine 2 / 2, remap) | 100 steps | KE 2.831557e-10 | bit-identical | neutral |

Reading. While no vertex leaves the hull the two values are the same
computation (same sets, so the same floats), also through the multiphase
remap whose first stage now reads the boundary from the old simplex
cache. In the dam break the first event is the ejected sliver vertex
(first vertex outside at step 1267 / 3002, at 5.5e3 to 5.7e3 m/s seven
steps later). Under `'hull'` the walls go in the next step. Under
`'membership'` the walls hold, but the fluid blows up as well and the
run ends in the same QhullError 23 / 8 steps later than the hull arm,
when the cloud is stretched until qhull's initial simplex is flat. The
walls are not what fails there; laneF's sliver F/m ejection is.

Reproducibility. Up to the ejection every arm is bit-identical in every
process. After it the first version of this lane found the membership
arm NOT reproducible between runs (abort after 1301 or 1303 steps; the
reviewer got 1300 and a third digest), cause not traced. Fix round 1
traced it (section 11.2): once the blown-up cloud reconnects, triangles
with one bulk vertex of each phase appear, the majority vote of
`assign_simplex_phases_from_vertices` is tied, and the tie was decided
by the `id()` order of the triangle's vertices, that is by memory
addresses. With the tie going to the lower phase ID the counts in the
table are the same in every process measured (22 runs in all). The
post-ejection numbers are therefore NOT the ones of the first version
(hull 1281 / 3025 steps, membership 1301 to 1303 / 3029). The first
vertex still leaves at step 1267 / 3002, and the shipped run and every
neutral case have the digests they had (section 11.2). The numbers after
the ejection still describe a run that is lost; they are quoted to show
that the walls stay.

Decisions:

- `hagen_poiseuille_2D`: switched (the old rule collapses).
- `dam_break_2D`: switched. Neutral where the case runs (bit-identical
  shipped run); where it fails it fails in both arms, with the walls in
  place under `'membership'`. No number of the case improves. The reason
  to switch is that the old rule adds a second failure (the wall
  collapse) on top of the first, which hides what a later sliver fix
  does. `test_dam_break_bindings` now proves the hand-written partial
  equal to the `frozen_set='hull'` arm and the preset equal to that plus
  `frozen_set='membership'`.
- `electrolysis_bubble_2D` / `_3D`, all droplet presets: not switched.
  Neutral, no vertex leaves the box, and the droplet presets carry pins.
- `hydrostatic_*`: not applicable. They run on `dual_only`,
  `dual_only_bare` or the 1D chain, where `bV` never changes.
- `shearing_plate_droplet_*`: not applicable (`periodic` path rebuilds
  `bV` itself; no membership there yet).
- capillary_rise static runners, dynCA: hand-rolled loops with a static
  frozen set, no retopology; nothing to switch.
- `dam_break_2D_no_air` / `_3D_no_air`, `cube_flow`, `Hagen_Poiseuile_3D`
  (`retopologize_cylinder`): not preset driven or custom retopology; not
  measured.

## 4. Key collisions (task 3)

### 4.1 `HC.V.move`

The cache is keyed by coordinate tuple. `move(v, x)` did
`cache.pop(v.x); cache[x] = v` with no check. Moving `v` onto the key of
`w` dropped `w` from the cache with its edges in place, and when `w` was
moved later `cache.pop(w.x)` removed `v`. A loop that shifts a structured
mesh by half its width therefore loses one vertex per collision.

Now: `move(v, x, on_collision='raise')` raises `VertexCollisionError`
(a `ValueError`) and changes nothing; `'evict'` is the old code path, bit
for bit; `move_all(moves)` releases every key first, so only the final
positions must be distinct, refuses two movers on one key or a mover on a
vertex that stays, and leaves the cache in the order a `move` loop gives.
It checks everything before it changes anything: a vertex listed twice or
a vertex that is not in the cache raises `ValueError` with the cache
untouched (fix round 1; before, such a call failed with `KeyError` after
some keys were already released).

Census before the change (a pytest plugin that counts, behaviour
unchanged): 68 moves onto an occupied key in 42 ddgclib tests, all from
the box shift of `droplet_in_box_2d` / `_3d` (and the copy of that loop
in the electrolysis setup); 1 in the hyperct suite (a test that drops a
vertex on purpose). After the change a scan of 61 runner scripts under
`cases_dynamic` (scratch copy, 100 s each, counting probe) found
collisions at four loops only, all four now explicit `'evict'`:
`_multiphase_droplet.py` 2D and 3D shift, `electrolysis_bubble/src/_setup.py`,
`shearing_plate_droplet/src/_setup.py`. The dynCA smoke run
(`capillary_rise_2D_dynCA.py --smoke`): 1 757 068 moves, 0 collisions, so
the refusing default does not change it (the reviewer's run: 1 928 173
moves, 0 collisions; the move counts differ because the dynCA smoke run
itself is not reproducible between processes on unchanged code, 8621
against 8689 steps, so its METHODS.md numbers are single-process values).

Library loops outside the scan (fix round 1). The scan covered runner
scripts, so a library loop that no runner exercises with a colliding
argument was missed: `disk(R=1, center=(1, 0), refinement=2)`,
`annulus(R_outer=1, R_inner=0.5, center=(1, 0))` and
`ball(R=1, center=(1, 0, 0), refinement=1)` raised `VertexCollisionError`
(the reviewer's finding; before the lane they returned 40 of 41, 36 and
34 of 35 vertices). The whole-mesh transforms of the library now use
`move_all`: the centre shift of `disk`, `annulus` and `ball`, `translate`,
`translate_surface`, `rotate_surface`, `scale_surface` and
`MeshAdvancer.step`. With no collision the result is bit-identical to the
loop (same cache order); with one, every vertex is kept
(`test_domains.py::TestCentreShiftKeepsEveryVertex`: 41 vertices and 104
edges for the shifted disk, as for the centred one). Loops left on single
moves on purpose: `cube_to_disk` / `cube_to_sphere` and the cylinder
projections (not a rigid transform, no collision in any test or scan),
`wrap_positions` of the periodic path (shearing-plate digest pinned by a
test), and the mean-flow modules (`_flow.py`, `_bubble.py`,
`_capillary_rise_flow.py`, `_sessile.py`, `_cube_droplet.py`,
`_ellipsoid.py`, `_hyperboloid.py`, `_capillary_rise.py`, `_volume.py`),
which move single vertices by a flow displacement. For those the new
default means: a move onto another vertex's key, which used to drop that
vertex silently, now raises. Coverage of that risk is thin and stated as
such: the fast suite (tutorial tests included) is green, and a scan of
106 scripts and notebooks under `cases_mean_flow`, `benchmarks`,
`tutorials` and `test_cases` counted 5786 moves and 0 collisions, but
only 23 of the 106 ran to the end in 100 s (75 failed for reasons of
their own: missing data files, stale absolute paths, old APIs; 8 timed
out). That scan must not be repeated the way it was run; see section
11.4.

### 4.2 C2, the inlet ghost

`PeriodicInletBC._reset_ghost` shifted the ghost by one period with a
loop of moves, which puts its downstream face on the keys of its upstream
face. Reproduced on the pre-lane code: 13 vertices in the unit mesh, 12
in the ghost. Now one `move_all` (`_shift_ghost`, also used for the
per-step advance): 13. A time step with `velocity * dt` equal to the
column spacing, which made every ghost vertex collide, is tested.

### 4.3 C3, injection on an occupied key

Every ghost column enters at the same position `inlet_pos + velocity *
dt`. `mesh.V[key]` returned the vertex already there and the BC
overwrote its `u`, `p`, `m`: a frozen wall vertex was reset to the inlet
velocity once per column. Now the resident vertex keeps its state (edges
are still copied). Audited, not changed:

- C3b: the first column leaves one wall-row vertex per wall one advection
  step from the corner (`x = U dt`). HP2D: 64 vertices on the wall lines
  against 62 walls, for the whole run. Bounded, because later columns hit
  that key.
- The unit mesh carries both periodic faces, so the seam column is
  injected twice per period, one step apart (the reset no longer loses
  one of them by accident). In HP2D the end state has 8 vertices with a
  neighbour closer than 0.05 and a smallest pair distance of 1.0e-03.
  The cure is to drop the leading face from the ghost; it changes every
  inlet case and belongs with the inlet work of lane H.
- Plug-speed ghost against a no-slip profile, and the point-value ghost
  pressure: unchanged.

## 5. Finding: the pinned droplet meshes miss outer vertices

`droplet_in_box_2d` builds the outer box on `[0, 2L]^2` and shifts it by
`-L` with a loop of moves. Measured with the loop as it is against
`move_all`:

| outer box | vertices built | after the loop | lost | corner `(L, L)` present |
|---|---|---|---|---|
| 2D refinement 1 | 13 | 12 | 1 | no |
| 2D refinement 2 | 41 | 39 | 2 | no |
| 2D refinement 3 (pinned droplet) | 145 | 139 | 6 | no |
| 3D refinement 1 | 35 | 33 | 2 | no |
| 3D refinement 2 (pinned droplet) | 189 | 186 | 3 | no |

With `move_all` nothing is lost. Every pinned droplet, electrolysis and
shearing-plate number was produced on these meshes, so the loops ask for
`on_collision='evict'` and carry a note; the pins did not move. The
repair is one line per loop plus a re-pin of the 2D and 3D droplet
baselines and floors. The shearing-plate 2D setup has 22 such collisions
in the shift and 1 in its rescale (the 23 of lane S).

## 6. Pin safety

- Fast suite and slow battery green (numbers in section 9); the pinned
  droplet, hydrostatic, remap and dam-break tests ran through the edited
  `_retopologize`.
- Default path: `frozen_set='hull'` executes the statements it executed
  before; `SolverMethods` binds nothing for the default
  (`test_default_binds_nothing`).
- Shearing-plate setup: state digest after 3 steps `17e9a78408066fea`,
  287 vertices, on the pre-lane copy and on the lane code. Since fix
  round 1 this is the digest in every interpreter (120 of 120), not only
  the most frequent one.
- Tied simplex vote (fix round 1): fast suite and slow battery green
  after the change; full `oscillating_droplet_2D.py` run bit-identical
  to `baseline_oscillation.json` (l2 0.17479361640597058, tail
  0.9998967874595965, mass drift 2.4056717879332966e-14) and full
  `oscillating_droplet_3D.py` run bit-identical to
  `baseline_oscillation_3d.json` (l2 0.24811340819647862, tail
  0.08409976059818802), both from scratch copies of the runners
  (protocol rule 6); the A/B digests
  of section 3.3 for the shipped dam break, the 2D droplet, electrolysis
  2D and 3D and the HP2D reproducer are unchanged.
- Library shifts on `move_all`: bit-identical where no key collides
  (`test_domains.py`, 60 tests, the pinned suites).
- `PeriodicInletBC` behaviour changed on purpose (sections 4.2, 4.3). It
  had no test and no pinned case.

## 7. Known limits

1. Impenetrability is not enforced. Walls are vertices; a fluid vertex
   can pass between two of them. HP2D at t = 30: one fluid vertex at
   `(8.354, 1.0051)`, 5.1e-3 above the top wall, moving outward. Under
   `'hull'` that event releases the wall; under `'membership'` the wall
   stays and the vertex is outside. A library wall clamp (planar wall,
   put back and zero the normal velocity) would replace the case-local
   `WallClampBC` (electrolysis) and the dynCA clamp. Note that a clamp
   onto a key that a wall vertex holds is now refused by `HC.V.move`.
2. BC order under `'membership'` (section 2, decision 2).
3. 3D: a vertex whose dual fan fails is tagged and zero-volumed as before
   but not frozen. 0 failed fans in a jittered 3D box (15 rebuilds) and in
   6 Delaunay steps of the 3D droplet.
4. `merge_cdist`: `merge_pair` keeps whichever vertex it meets first, so
   a member can be merged into a mobile vertex and the wall loses it. No
   preset sets `merge_cdist`.
5. A dead member is pruned by identity (`HC.V.cache.get(v.x) is v`); a
   vertex dropped by an `'evict'` collision is pruned the same way.
6. No membership on the periodic path, in `delaunay_material`, in
   `bare_dual_refresh` or in a custom retopology function, and none with
   `connectivity='adaptive'` (raises; section 2). An adaptive version
   needs a constrained-vertex set in `hyperct.remesh` (split vertices of
   wall edges become members, members are exempt from collapse and
   smoothing).
7. `_retopologize(frozen_set='membership', skip_triangulation=True)`
   works and is tested, but `SolverMethods` does not offer it for
   `dual_only`: there it would change the tag of free-surface vertices
   (they keep `v.boundary` instead of losing it after the first call),
   which is a different method and was not measured.

## 8. DO-NOTs (measured)

- Do not put `PositionalNoSlipWallBC` before the inlet BC under
  `frozen_set='membership'`.
- Do not shift or rescale a structured mesh with a loop of `HC.V.move`.
  Use `HC.V.move_all`. The loop now raises instead of losing vertices.
- Do not replace the `'evict'` loops in the droplet builders without a
  re-pin: 6 (2D) and 3 (3D) outer vertices come back.
- Do not read the dam-break abort as a wall problem: with the walls held
  the run is lost 8 to 23 steps later.
- Do not break a tie in a vote by the order in which a simplex lists its
  vertices. In 2D that order is `id()` order (`iter_triangles_2d`), which
  differs between interpreters. Until fix round 1 this made a blown-up
  dam break and the shearing-plate setup process-dependent.
- Do not select `frozen_set='membership'` with `connectivity='adaptive'`
  (raises): split vertices of wall edges are not members, and members off
  the hull are collapsed and smoothed.
- Do not run scripts of `cases_mean_flow` (manuscript generators above
  all) to probe library behaviour, not even from a scratch copy: several
  `os.chdir` to an absolute repository path and overwrite their outputs
  there (section 11.4).
- Do not expect `'membership'` to keep fluid inside. It keeps walls in
  place.
- Do not use the old tree symlinked into a snapshot for test runs
  (bytecode caches end up in the snapshot); copy it.

## 9. Tests

New: `ddgclib/tests/test_frozen_set.py` (31: mechanism on `_retopologize`
in 2D and 3D, tag against frozen set, pruning, BC additions, the refusal
of adaptive remeshing, the integrator, `SolverMethods` plumbing and
rejected combinations, the multiphase remap path, the HP2D reproducer in
both arms); `test_boundary_conditions.py::TestPeriodicInletBC` (4; the
first three fail on the pre-lane code: `12 == 13`, no `move_all`,
`2 == 3` new vertices; the fourth is `MeshAdvancer.step` by one column
spacing); `test_domains.py::TestCentreShiftKeepsEveryVertex` (4; raise
`VertexCollisionError` on the first version of the lane);
`test_multiphase.py::...::test_tied_simplex_vote_goes_to_the_lower_phase`
(fails on the vote as it was: winners `{0, 1}` in 2D);
`hyperct/tests/test_vertex_move.py` (15).

Battery at the end of fix round 1:

| suite | before the lane | first version | after fix round 1 |
|---|---|---|---|
| ddgclib fast (`-m "not slow"`) | 1047 passed, 12 skipped, 2 xfailed | 1078 passed | 1087 passed, 12 skipped, 2 xfailed (1078 + 3 frozen-set + 1 inlet + 4 domains + 1 multiphase) |
| ddgclib slow (`-m slow`) | 23 passed, 1 xfailed | 23 passed, 1 xfailed | 23 passed, 1 xfailed |
| hyperct (`pytest hyperct/tests -k "not benchmark"`) | 301 passed | 314 passed | 316 passed |

The brief's bare `pytest -q` at the hyperct root stops at 4 collection
errors (duplicate test module names in `hyperc_rl_quick_figs_delete/tests`
and `hyperct/tests`), before and after the lane; the command above is the
one that runs.

`test_periodic_multiphase_is_bit_identical_to_shearing_wrapper` was flaky
before this lane and is fixed in fix round 1 (section 11.2). It runs the
shearing-plate setup in two fresh interpreters and compares a state
digest. The reviewer measured the deviation rate with a three-step probe:
23 of 120 interpreters on the first version of the lane, 12 of 120 on
the pre-lane copy (the rate depends on incidental process state; the
first version of this log quoted 0 of 42 and 2 of 42). After the fix the
same probe gives one digest in 120 of 120 interpreters (8 in parallel),
and the setup-only probe one digest in 96 of 96.

## 10. Reproduce

    cd /home/endres/projects/ddgclib
    PY=/home/endres/anaconda3/envs/ddg/bin/python
    $PY -m pytest ddgclib/tests/test_frozen_set.py ddgclib/tests/test_boundary_conditions.py -q -p no:cacheprovider
    (cd ../hyperct && $PY -m pytest hyperct/tests/test_vertex_move.py -q -p no:cacheprovider)
    # section 3.1 and 3.3
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D --alpha 0.2
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py dam_break_2D --alpha 0.5 --t-end 0.45
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_2D
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py electrolysis_3D --steps 300
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py droplet_2D --steps 100
    # section 3.2 (about 4 minutes each)
    # NOTE (lane H, 2026-10-02): lane H rewrote this runner. The two
    # commands below now run the lane H configuration (buffered inlet,
    # viscous_flux='simplex_gradient', Re_D 10, L 12), NOT the L 15 run
    # of section 3.2 (walls released at step 1262), which can no longer
    # be reproduced from the runner. What is kept and still reproduces
    # the release of the walls is the short reproducer:
    # diagnose_frozen_set.py hp2d above (digests 62106841d3f841a9 hull,
    # 47e12338835537f3 membership) and
    # test_frozen_set.py::TestHagenPoiseuille2D, both pinned to the
    # lane L configuration (setup_poiseuille_2d_lagrangian,
    # viscous_flux='two_point').
    cd cases_dynamic/Hagen_Poiseuile
    $PY Hagen_Poiseuile_2D.py --headless
    $PY Hagen_Poiseuile_2D.py --headless --frozen-set hull --tag hull

A/B records: `cases_dynamic/Hagen_Poiseuile/results/frozen_set_ab/*.json`
with the `methods` record of each arm next to it.

## 11. Fix round 1 (after independent review, 2026-10-01)

Verdict of the review: fail, on one blocking issue; tree green apart from
a flaky test that predates the lane. Every claim the reviewer re-measured
held except two: the quoted dam-break abort steps (the reviewer's run
ended after 1300 steps, outside "1301 to 1303") and the statement that
`'membership'` works with `connectivity='adaptive'`.

### 11.1 Blocking: `'membership'` with adaptive remeshing

Accepted by `SolverMethods`, listed as supported in section 2, never
measured, and wrong (table in section 2). Resolution: the combination
raises, in `SolverMethods` (`_HULL_REBUILDING = ('delaunay',)`) and in
`_retopologize` / `_retopologize_multiphase`. Inheritance of membership
by split vertices was considered and not done, because it repairs one of
two mechanisms: the remesh driver also collapses and smooths members that
are off the hull (measured: up to 5 of 18 removed, up to 7 moved by up to
0.17). No preset uses the combination.

### 11.2 The process dependence, traced

Cause. `MultiphaseSystem.assign_simplex_phases_from_vertices` labels a
simplex by majority vote of its bulk vertices. Its docstring promises
that a tie goes to the lower phase ID. The code took
`Counter.most_common(1)`, which on a tie returns the phase met first,
i.e. the phase of the first bulk vertex of the simplex. In 2D the
simplices come from `hyperct.remesh._quality.iter_triangles_2d`, which
orders the three vertices by `id()`. Memory addresses differ between
interpreters, so a tied triangle could get either phase.

Evidence (`shear_tie_probe.py`, scratch): the shearing-plate setup has
10 tied triangles in its first vote. 47 of 48 interpreters gave all ten
to phase 0 (state digest `b60215b8`, total mass 0.8181555448190194); one
gave `1111001111` (digest `4302b3bb`, mass 0.8279967083971509), which is
one of the deviating masses the reviewer reported. Positions, edges and
dual volumes were equal, as the reviewer found.

Fix: the tie goes to the lower phase ID, as documented
(`ddgclib/multiphase.py`, three lines). For a vote that is not tied
nothing changes. The usual outcome so far was the lower phase, so the
shearing-plate digest every later lane compared against is unchanged.

| measurement after the fix | result |
|---|---|
| shearing-plate setup digest, 96 interpreters, 8 in parallel | `b60215b8` in 96 of 96 |
| reviewer's three-step probe (library wrapper and case closure), 120 interpreters, 8 in parallel | setup `b60215b8ee996368`, after 3 steps `17e9a78408066fea`, 287 vertices in 120 of 120 (first version of the lane: 23 of 120 deviate; pre-lane: 12 of 120) |
| `dam_break_2D` alpha_art 0.2, membership arm | 1303 steps, digest `86002265325138f6`, 32 / 32 walls frozen, 0 moved in 9 of 9 processes (before: 1300, 1301 and 1303 steps, three digests) |
| same, hull arm | 1280 steps, `f36e005a21d74efd`, 1 / 32 frozen, 32 moved in 5 of 5 (before: 1281 steps, `96d18343fac6bfe1`, 0 / 32 frozen) |
| `dam_break_2D` alpha_art 0.5 to t = 0.45 s | hull 3022 steps (`64e72018a75868db`, 3 of 3), membership 3030 (`7abc17e4e28c1b07`, 5 of 5); before: 3025 and 3029 |
| `dam_break_2D` shipped, both arms | `952d4544676ca366`, KE_max 1.036626e-03: unchanged |
| `oscillating_droplet_2D` 100 steps, both arms | `4f20087f8c189dfe`: unchanged |
| `electrolysis_bubble_2D` 6330 steps, both arms | `9ed4c69378ac129a`, KE_max 6.213015e-03: unchanged (this also closes the reviewer's "not re-run" for that case) |
| `electrolysis_bubble_3D` 300 steps, both arms | `d4e464d4974dbf5d`: unchanged |
| HP2D reproducer, hull / membership | `62106841d3f841a9` / `47e12338835537f3`: unchanged (single phase) |
| full `oscillating_droplet_2D.py` against `baseline_oscillation.json` | 8 of 8 numeric fields equal to the bit |
| full `oscillating_droplet_3D.py` against `baseline_oscillation_3d.json` | 13 of 13 numeric fields equal to the bit |
| fast suite, slow battery | green, no pin moved |

So the change is neutral for everything that ran and decides only cases
that were undecided before: the blown-up dam break after the ejection and
the shearing-plate setup. The dam-break records in
`results/frozen_set_ab/` were regenerated; the alpha 0.2 and alpha 0.5
files of the first version are superseded.

What is NOT fixed: `iter_triangles_2d` still lists triangles and their
vertices in `id()` order, and `hyperct.remesh._driver._edge_list` orients
edges by `id()`. Any consumer that depends on that order (a sum over
triangles in the last bits, which endpoint survives an edge collapse) can
still differ between processes. Not measured here. The dynCA smoke run is
not reproducible between processes either (single phase, so the vote is
not its cause); not traced.

### 11.3 Non-blocking findings

| finding | resolution |
|---|---|
| flaky shearing-plate bit-identity test | fixed at the cause, 11.2 |
| `disk` / `annulus` / `ball` with a centre, `translate`, `MeshAdvancer.step` raise instead of losing a vertex | converted to `move_all`, with `translate_surface`, `rotate_surface`, `scale_surface`; tests (section 4.1) |
| the raising default of `HC.V.move` was not scanned for `cases_mean_flow`, `tutorials`, `benchmarks` | scanned with thin coverage (5786 moves, 0 collisions, 23 of 106 scripts completed); stays a stated risk (section 4.1). The scan itself caused the incident of 11.4 |
| dam-break abort steps quoted as "1301 to 1303" | cause removed; exact numbers with the process counts behind them (section 3.3) |
| `dam_break_2D` switched with no number improving | unchanged decision, stated as such in section 3.3 |
| protocol rule 2 (new paths in `_retopo.py`) | exception stated in section 1 |
| `Hagen_Poiseuile_2D.py` needs `PYTHONPATH` | inserts the repository root itself now (checked: imports the live `hyperct` with `PYTHONPATH` unset). It still calls the private `_retopologize` once, in its post-processing block, to give vertices injected in the last BC pass a dual cell; a public refresh on `SolverMethods` would be a new feature and was not added |
| `move_all` not atomic for a vertex listed twice or missing from the cache | validated up front, 2 hyperct tests |
| stray probe outputs of the first version | still there, see below |
| unverified claims | electrolysis 2D over 6330 steps and dam break alpha 0.5 are re-run here (11.2); shipped HP2D re-run on the final tree: `hp2d_final_state.json` byte-identical to the stored file, 62 / 62 / 0 / 0.0. Still unverified by a second party: the pre-edit replica collapsing at step 1248, the census of 68 colliding moves, the scan of 61 runner scripts, zero failed dual fans in the 3D checks, `visualize_hp2d.py` against the new outputs |

Stray outputs that are still in the tree (file deletion in the repository
was denied to the first version, and this round did not retry it):
`cases_dynamic/Hagen_Poiseuile/results/laneL_smoke/` (5 files) and ten
files matching `*laneLprobe_smoke*` in `cases_dynamic/capillary_rise/fig/`
and `cases_dynamic/capillary_rise/results/`.

### 11.4 Incident: the collision scan overwrote four files in `cases_mean_flow`

What happened. To cover the reviewer's "not scanned" finding, this round
copied `cases_mean_flow`, `benchmarks`, `tutorials` and `test_cases` to
the scratchpad and ran 106 scripts there under a counting hook. The
selection was a text match on `ddgclib|hyperct`, which also matched
manuscript generator scripts that merely contain the repository path.
Three copies of `manuscript/validation/sigma_finalize.py` (the current
one and two older ones kept in backup directories) and two copies of
`tornado.py` start with `os.chdir("/home/endres/projects/ddgclib/cases_mean_flow/equil_bubble")`,
so they wrote into the real tree, which is off limits for this campaign:

| file (under `cases_mean_flow/equil_bubble/`) | overwritten at | state now |
|---|---|---|
| `manuscript/validation/sigma_results.json` | 22:47:04 | restored from the copy made before the scan: byte-identical, md5 `fd05b3baf8075be7486b39daf8783128`, mtime 2026-07-28 17:47:35 |
| `manuscript/tex/figures/sigma_vs_measured.pdf` | 22:47:04 | restored from `manuscript/_fig9_backup_2026-09-17/manuscript/tex_langmuir/figures/` (md5 `d0aa10107fa03530d7c18eec22cdfdb7`) |
| `manuscript/tex/figures/sigma_vs_measured.png` | 22:47:04 | restored from the same directory (md5 `27b8101c7d8acba3912fe3005e7f68ea`) |
| `fig/validation/uncertainty_tornado.png` | 22:47:18 | restored from `manuscript/_figfont_backup_2026-09-17/fig/validation/` (md5 `dd07f80ab8bcde4865ae308f8359c5c2`) |

How sure the restores are. The JSON is exact: the scratch copy was taken
before the scan. For the three figures no copy from before the scan
exists (the scratch copy excluded images), the files are git-ignored,
and the filesystem has no snapshots. They were restored to the md5 that
`manuscript/_figlabel_backup_2026-09-16/_tmp/baseline_md5.json` records
for exactly these paths on 2026-09-16, taken from backup copies with
those md5 sums. Evidence on whether anything changed them after that
date:

- `sigma_vs_measured` in `manuscript/tex/figures/`: the pre-scan JSON
  was dated 2026-07-28 17:47:35.47 and the two restored figure files
  carry 17:47:35.80 and 17:47:36.05 of the same day, the write order of
  the generator, so the generator had not been run in place since; the
  two backup manifests of 2026-09-17 list the `tex_langmuir` and slide
  copies of this figure as changed, not the `tex/` copy; nothing else in
  `manuscript/tex/figures/` is newer than 2026-08-17. Strong.
- `fig/validation/uncertainty_tornado.png`: weaker. The font pass of
  2026-09-17 backed this file up as one it might replace. That its
  replacement did not happen is suggested by: the redirected `tornado`
  run of that pass failed and left no output; the PDF next to it and its
  `tex_langmuir` copy still have the 2026-09-16 md5
  (`0d822f2aa6640244522e12bf8495dc76`); the change log of that pass says
  the tornado figure "remains DejaVu SANS". A regenerated PNG of that day
  exists in the backup's `_tmp/runs/valid/` (55122 bytes, md5
  `9d1d4ad536f5f87583664efe118c4610`) and would be the other candidate.

This is inference, not a before-copy. If one of the three files was
changed between 2026-09-16 and today, the restore is wrong for it.

What to check. `/home/endres/projects` is a Syncthing folder. The other
device held the pre-scan files until the overwrites synced; if file
versioning is on there, the exact pre-scan versions are in its
`.stversions`. The versions the scan wrote are kept in
`scratchpad/fixL/written_by_scan/` (md5: json `d5c38f48...`, pdf
`4c51fbae...`, png `09f5062d...`, tornado png `f92ad42f...`) in case
they are wanted for comparison.

Nothing else was touched: no other file or directory in either
repository, in the home directory or in another session's scratchpad is
newer than the start of the scan, apart from the edits listed in
section 1. One other generator (`trend.py`) was stopped by the timeout
before its first write.

Rule taken from it (section 8): scripts from `cases_mean_flow` are not
run to probe the library, not even from a copy. A static read of the
call sites is the safe way to do what the scan tried to do.
