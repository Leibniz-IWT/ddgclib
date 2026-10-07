# laneB (October 2026): the droplet-in-box builders keep every outer vertex

Date: 2026-10-05 (not the lane B of 2026-07-29, the 3D score harness,
whose log is `laneB-3d-score-harness.md`). Closes the side finding of lane L (section 5 of
`laneL-frozen-set-membership.md`): `droplet_in_box_2d` / `_3d` lost outer
vertices, the box corner included, to key collisions in their box shift.
Every oscillating-droplet, static-droplet, electrolysis and
shearing-plate number of the campaign was computed on those meshes.

## 0. Verdict

- The builders return the full mesh. The shift of the outer box onto the
  droplet centre is one `HC.V.move_all` (`box_shift='move_all'`, the
  default). The lossy loop of single moves is the explicit builder
  argument `box_shift='evict'`, a SETUP choice (not a solver axis),
  passed through by the three setups, recorded in `params['box_shift']`
  and in the `extra` block of every runner's `methods.json`, and
  documented under the `mesh` axis of the registry.
- What was lost (section 2): at the pinned box `L = 0.05` 6 of 145 outer
  vertices at 2D refinement 3 and 3 of 189 at 3D refinement 2, the
  `(L, ..., L)` corner always among them, so the convex hull was cut at
  that corner (2D total dual volume 9.8828e-3 instead of 1e-2, 1.17 % of
  the domain missing, outer-phase mass 1.2 % short, largest outer cell
  1.30e-4 instead of 1.04e-4). The count depends on the floating-point
  value of `L`: 22 of 145 at the shearing plate's `L = 0.015` (the whole
  positive quadrant's interior rows), 8 of 41 at the electrolysis box
  `L = 0.004` (refinement 2).
- Every pin on that path was re-measured in both arms. `'evict'`
  reproduces every pre-laneB number to the bit (2D baseline in every
  key, 3D baseline in every key, static droplet 1.1847162859108737e-03,
  3D plateau 7.2741721787e-05, lane L's A/B digests of the 2D droplet
  and the 3D electrolysis; the 2D electrolysis record of lane L is not
  reproduced by any library since lanes T and O, section 3.3). On the
  full mesh the main benchmark moves little: 2D l2 0.17479361640597058
  -> 0.17439096487276182, tail 0.9998967874595965 -> 0.9998871416222597;
  3D l2 0.24811340819647862 -> 0.24811443136179492 (4e-6 relative),
  tail 0.08409976059818802 -> 0.0841737962816189. Re-pinned:
  both baselines (with their methods block), the 3D plateau
  7.274172e-05 -> 7.274134e-05 (floor test and a5b), the static droplet
  summary, the preset notes. The 2D floors and every bound of the fast
  dynamic tests hold unchanged.
- The open physics items are not the corner (section 5): the 2D
  over-decay is still removed by `projection_every=2` (l2 0.0388 against
  0.1744, the same -78 %); the 3D bump / over-decay cancellation moves
  by 4e-6 relative; the shearing-plate setup still deletes interface
  vertices, now visibly two instead of one, because its anisotropic
  rescale maps two outer vertices exactly onto the droplet poles, and
  the pre-laneB shearing mesh also lacked 4 of the 9 top-plate vertices
  (section 4.3).

## 1. What changed and where

| file | change |
|---|---|
| `ddgclib/geometry/domains/_multiphase_droplet.py` | `BOX_SHIFTS`, `_shift_outer_box(HC, offset, box_shift)` (one `move_all`, or the old loop with `on_collision='evict'`); `droplet_in_box_2d(..., box_shift='move_all')`, `droplet_in_box_3d(...)` likewise; `metadata['box_shift']` |
| `ddgclib/geometry/domains/__init__.py` | exports `BOX_SHIFTS` |
| `cases_dynamic/oscillating_droplet/src/_setup.py` | `setup_oscillating_droplet(box_shift=)`, `params['box_shift']` |
| `cases_dynamic/electrolysis_bubble/src/_setup.py` | the off-centre 2D builder's own copy of the loop is gone, it calls `_shift_outer_box`; `setup_electrolysis_bubble(box_shift=)`, `params['box_shift']` |
| `cases_dynamic/shearing_plate_droplet/src/_setup.py` | `setup_shearing_plate_droplet(box_shift=)`, `params['box_shift']`; the anisotropic rescale loop is unchanged (section 4.3) |
| `cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py`, `_3D.py`, `static_droplet_2D.py` | `--box-shift {move_all,evict}`; the non-default arm writes suffixed artifacts (`score_evict.json`, `snapshots_evict/`, ...); `box_shift` in the score and in the `extra` block of `methods.json` |
| `cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py`, `_3D.py`, `cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_2D.py`, `_3D.py`, `_run_short_2D.py`, `_run_short_3D.py` | `box_shift` in the `extra` block of `methods.json` |
| `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py` | `run_a5b(box_shift=)`, recorded in the result |
| `cases_dynamic/oscillating_droplet/diagnose_box_shift.py` (new) | `census`, `mesh`, `shearing`, `floors`, `envelope`, `shearrun`, `fullrun` (section 7) |
| `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py` | `--box-shift`, `--out`; the evict arm gets its own record name (`_bs-evict`) |
| `cases_dynamic/oscillating_droplet/baselines/baseline_oscillation.json`, `baseline_oscillation_3d.json` | re-pinned with their methods block, `box_shift` and refinement recorded |
| `ddgclib/tests/test_box_shift.py` (new) | 15 tests (section 6) |
| `ddgclib/tests/test_case_oscillating_droplet.py`, `test_a5b_longrun_regression.py` | 3D plateau 7.274172e-05 -> 7.274134e-05; measured values in the docstrings |
| `ddgclib/methods/_axes.py` | note on the `mesh` axis: where the switch lives |
| `ddgclib/methods/_presets.py` | notes of the droplet, static, electrolysis and shearing presets |
| `METHODS.md`, `DEVELOPMENT.md`, `debugging_plan.md` | documentation |

No solver, integrator, operator or hyperct code changed. `hyperct.move_all`
is lane L's.

Why a builder argument and not an axis: the shift is geometry of the
setup, `SolverMethods` describes the solver; a solver axis would make
every preset carry a mesh defect switch. The value is recorded next to
the refinement in `methods.json` (`extra.box_shift`), which is where
`diff_baselines` readers find the refinement too.

## 2. The loss, reproduced (task 1)

`diagnose_box_shift.py census` (records in
`cases_dynamic/oscillating_droplet/results/box_shift/census.json`). The
outer box is built on `[0, 2L]^dim` and every vertex moved by `-L`; in a
loop of single moves a target that is still the key of another vertex
evicts it (its edges stay, so the neighbours keep dangling references
and the evicted vertex is simply absent from `HC.V`).

| box `L = 0.05` | built | kept (evict) | lost | corner `(L, ..., L)` | dangling `nn` entries | kept (move_all) |
|---|---|---|---|---|---|---|
| 2D refinement 1 | 13 | 12 | 1 | missing | 3 | 13 |
| 2D refinement 2 | 41 | 39 | 2 | missing | 5 | 41 |
| 2D refinement 3 (pinned 2D droplet) | 145 | 139 | 6 | missing | 32 | 145 |
| 3D refinement 1 | 35 | 33 | 2 | missing | 18 | 35 |
| 3D refinement 2 (pinned 3D droplet) | 189 | 186 | 3 | missing | 30 | 189 |

Lost at 2D refinement 3: `(0, 0.0375)`, `(0.01875, 0.01875)`,
`(0.0375, 0)`, `(0.0375, 0.0375)`, `(0.0375, 0.05)` (a wall vertex),
`(0.05, 0.05)` (the corner). At 3D refinement 2: `(0, 0, 0.05)` (the
centre of the top face), `(0.0375, 0.0375, 0.0375)`, `(0.05, 0.05, 0.05)`.

A collision needs `x - L` to equal another grid key exactly in floating
point, so the count is a property of the value of `L`:

| box | 2D r1 / r2 / r3 lost | 3D r1 / r2 lost |
|---|---|---|
| `L = 0.05` (droplet) | 1 / 2 / 6 | 2 / 3 |
| `L = 0.015` (shearing plate, `L_build`) | 2 / 10 / 22 | 2 / 11 |
| `L = 0.004` (electrolysis) | 2 / 8 / 16 | 2 / 11 |

The shearing plate's 22 (2D refinement 3) are every interior vertex of
the positive quadrant plus three wall vertices and the corner; lane L
counted the same 22.

What the mesh looks like with the hole (`diagnose_box_shift.py mesh`,
both arms through `setup_oscillating_droplet`, each at setup and after
ONE retopology through the preset at frozen positions;
`results/box_shift/mesh.json`):

| fixture, arm | vertices | walls (corners) | sum of dual volumes (box) | outer-phase mass (`rho_o V_outer` = 9.685841) | largest / median outer cell |
|---|---|---|---|---|---|
| 2D 3/3 `oscillating_droplet_2D`, move_all | 317 | 32 (4 / 4) | 1.0000000000e-02 (1e-2) | 9.6819786782 | 1.0417e-04 / 5.2083e-05 |
| 2D 3/3, evict | 311 | 31 (3 / 4) | 9.8828125000e-03 | 9.5647911782 | 1.3021e-04 / 5.2083e-05 |
| 2D 2/2 (envelope fixture), move_all | 97 | 16 (4 / 4) | 1.0000000000e-02 | 9.6679997542 | 3.6458e-04 / 1.5625e-04 |
| 2D 2/2, evict | 95 | 15 (3 / 4) | 9.6875000000e-03 | 9.3554997542 | 3.6458e-04 / 1.5625e-04 |
| 2D 1/2 (fast fixtures), move_all | 69 | 8 (4 / 4) | 1.0000000000e-02 | 9.6679997542 | 7.4484e-04 / 1.9348e-04 |
| 2D 1/2, evict | 68 | 8 (3 / 4) | 8.7500000000e-03 | 8.4179997542 | 7.4484e-04 / 1.9348e-04 |
| 3D 2/2 `oscillating_droplet_3D`, move_all | 475 | 98 (8 / 8) | 1.0000000000e-03 (1e-3) | 0.99569825403 (0.9958112) | 8.0339e-06 / 3.9063e-06 |
| 3D 2/2, evict | 472 | 96 (7 / 8) | 9.9739583333e-04 | 0.99309408736 | 8.1380e-06 / 3.9063e-06 |
| 3D 1/1, move_all | 95 | 26 (8 / 8) | 1.0000000000e-03 | 0.99487181826 | 3.9496e-05 / 1.4323e-05 |
| 3D 1/1, evict | 93 | 24 (7 / 8) | 9.7916666667e-04 | 0.97403848493 | 3.9496e-05 / 1.3021e-05 |

Reading. The hole is not papered over by the first retopology; it is
papered over by the builder's own Delaunay in `_build_combined_mesh`
(the convex hull of the point cloud lacks the corner, so the domain is
the box minus a corner triangle / tetrahedron, and the cells next to
the lost positions are 25 % larger at 2D refinement 3: around
`(0, 0.0375)` the 8 remaining cells have mean dual volume 8.14e-05
against 7.09e-05 for the 9 cells with the vertex). The first retopology
at frozen positions changes nothing in 2D (idempotent) and in 3D only
zeroes the wall cells as it always does (sum 6.48e-4 with 98 walls
against 6.41e-4 with 96). The missing domain is 1.17 % of the box at 2D
refinement 3, 3.1 % at 2/2 and 12.5 % at the 1/2 fast fixture (one of
the four corner squares of a 4 x 4 lattice); the outer-phase mass is
short by the same fraction. The droplet at 5 R0 from the corner is
barely affected, which the numbers of section 3 confirm.

## 3. Re-measurement and re-pin (task 3)

Every arm is a preset, the two builder arms differ in the setup choice
only. "old" = `box_shift='evict'`, measured in this lane and equal to
the recorded pin to the bit wherever a bit-level pin exists; "new" =
`box_shift='move_all'`, the default now.

### 3.1 2D oscillating droplet

| measurement (preset) | old (evict) | new (move_all) |
|---|---|---|
| full run `oscillating_droplet_2D` (refinement 3/3, 1839 steps): l2 | 0.17479361640597058 (= `baseline_oscillation.json` before this lane, every numeric key equal) | 0.17439096487276182 |
| same: linf / tail / mass drift | 0.32364245955409165 / 0.9998967874595965 / 2.4056717879332966e-14 | 0.323163334437819 / 0.9998871416222597 / 1.4835808907017442e-14 |
| same: two-fluid reference l2 / linf | 0.18461475641216668 / 0.3111746434345773 | 0.18420032366674188 / 0.31069551831830466 |
| full run `oscillating_droplet_2D_dual_only`: l2 / tail | 0.17857 / 0.99925 (lane 5 record) | 0.17946703687459944 / 0.9998387858074947 |
| full run `oscillating_droplet_2D_projection2`: l2 / tail / two-fluid l2 | 0.03795682994323827 / 1.3973 / 0.02193 (lane H record) | 0.038841363169171125 / 1.3936743069461104 / 0.023062128950581105 |
| `static_droplet_2D` (100 steps): summary = interface radius drift / max KE normalised / mass | 1.1847162859108737e-03 (= pin) / 7.673878163142909e-09 / 0.0 | 1.1672989414885857e-03 / 8.792414018747378e-09 / 0.0 |
| `static_droplet_floor_2D` (3/3, 20 steps): step 0 / step 1 / plateau spread | 2.3748568012e-03 / 2.2716937802e-03 / 0.0 | 2.3748568012e-03 (identical) / 2.2716937806e-03 / 1.44e-13 |
| a5b 2D (`run_a5b`, same preset): peak / end | 2.3748568012e-03 / 2.2716937802e-03 | 2.3748568012e-03 / 2.2716937806e-03 |
| envelope mirror 2/2 (`TestOscillationEnvelopeRegression2D`, 267 steps): l2 / tail / linf / KE_max | 0.054514259329682013 / 0.93689037089269311 / 0.083676475603304545 / 1.4214765248811987e-06 (= the test docstring) | 0.054618059950224354 / 0.93731955721225069 / 0.083920319769496821 / 1.4248519830969824e-06 |
| endurance 1/2, 200 steps: KE end / trace max | 9.6791934625e-07 / 8.0400e-07 | 9.6792333344e-07 / 8.0401e-07 |
| remap 1/2, 40 steps: KE end | 1.2955767335e-07 | 1.2956049457e-07 |
| dual_only 1/2, 40 steps: KE end; with `projection_every=4` | 1.4662942037e-07; 2.1771967317e-07 (R_max 0.010495469963 / 0.010494728707) | 1.4977509792e-07; 2.1534792369e-07 (0.010495484252 / 0.010494777382) |
| remap 1/2 with `projection_every=5`, 40 steps: KE end | 6.3440659532e-08 | 6.3450279857e-08 |

Pins: `baseline_oscillation.json` re-pinned to the new row (methods
block, `box_shift`, refinement and the two-fluid keys recorded); the 2D
floors keep their digits (step 1 moves by 1.8e-10 relative); the
envelope bounds `l2 < 0.0600`, `tail < 1.004` hold with the same
headroom; the KE bounds (1e-5) of the endurance, remap, dual-only and
cadence tests hold with the same margins (the largest KE is 9.7e-7);
the cadence A/B still differs (the test's `assertNotEqual`). The static
droplet summary 1.1847162859108737e-03 -> 1.1672989414885857e-03
(preset note, METHODS.md; the runner's `baseline_equilibrium.json` of
April 2026, summary 7.41e-03, predates the library path and is left as
it is).

### 3.2 3D oscillating droplet

| measurement (preset) | old (evict) | new (move_all) |
|---|---|---|
| full run `oscillating_droplet_3D` (refinement 2/2, `dual_only`, 872 steps): l2 = apex l2 | 0.24811340819647862 (= `baseline_oscillation_3d.json` before this lane, every one of its 19 numeric and boolean keys equal) | 0.24811443136179492 |
| same: linf / tail / mass drift | 0.5992701359956998 / 0.08409976059818802 / 1.905002320272536e-14 | 0.5992731933474778 / 0.0841737962816189 / 4.8000942590870176e-14 |
| same: R_max_peak / step-0 dual-volume jump / post drift | 0.010790236105250779 / 0.35704960835509636 / 1.0684e-04 | 0.010790237633926668 / 0.3515625000000047 / 1.0567e-04 |
| same: interface count start / min / max / end, saturation | 98 / 98 / 98 / 98, none | 98 / 98 / 98 / 98, none |
| full run `oscillating_droplet_3D_delaunay`: l2 / tail / R_max_peak | 1.5244561707801316 / 0.47067 (lane B of July) | 1.5291100090540053 / 0.4594532578392181 / 0.011400164376344007 |
| `static_droplet_floor_3D` (2/2, 20 steps): step 0 / plateau (step 1) / end / plateau spread | 6.0153201140e-05 / 7.2741721787e-05 / 7.2741721787e-05 / 6.05e-12 | 6.0153201140e-05 (identical) / 7.2741338970e-05 / 7.2741338968e-05 / 2.72e-11 |
| a5b 3D (`run_a5b`, same preset): peak / end | 7.2741721787e-05 | 7.2741338970e-05 / 7.2741338968e-05 |

The step-0 dual-volume jump of the 3D runner is the fraction of the box
held by the zeroed wall cells; with all 98 walls it is the exact lattice
fraction 0.3515625 (= 45 / 128), with 96 it was 0.35705 because the
hull itself was cut. Pins: `baseline_oscillation_3d.json` re-pinned to
the new row; `EXPECTED_PLATEAU_MAXF` and `A5B_3D_PEAK` / `_END`
7.274172e-05 -> 7.274134e-05 (the old value is 5.3e-6 relative away and
would have passed the 1 % tolerance; the pin is moved so the quoted
digits are the measured ones); step 0 is bit-identical because the
interface stencil does not reach the box.

### 3.3 Electrolysis bubble and shearing plate (smoke numbers)

Lane L's A/B driver (`cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py`,
`frozen_set` hull against membership; its records are the smoke numbers
of these cases) re-run in both builder arms. The final-state digest is
a sha256 over sorted `(x, u, m)`.

| case (preset), horizon | old (evict) | lane L's record (2026-10-01) | new (move_all) |
|---|---|---|---|
| `oscillating_droplet_2D` refinement 2/2, 100 steps of dt 2e-5: digest / KE / vertices / walls / mass | `4f20087f8c189dfe` / 2.831557066728673e-10 / 95 / 15 / 9.62116629990062 | `4f20087f8c189dfe` / 2.831557e-10 (reproduced) | `443c01a815acc6c3` / 2.8299870377596923e-10 / 97 / 16 / 9.933666299900592 |
| `electrolysis_bubble_3D` (refinement 1/1), 300 steps: digest / KE_max / vertices / walls / mass | `d4e464d4974dbf5d` / 1.874248949188324e-12 / 93 / 24 / 4.968436810154017e-04 | `d4e464d4974dbf5d` / 1.874249e-12 (reproduced) | `6bdbc9c542ffd6fc` / 1.9456521671316604e-12 / 95 / 26 / 5.075113940820681e-04 |
| `electrolysis_bubble_2D` (refinement 2/3), 6330 steps (the shipped horizon): digest / KE_max / vertices / walls / mass | `38530636a343cf7a` / 6.212277546530253e-03 / 206 / 15 / 0.05887904743753862 | `9ed4c69378ac129a` / 6.213015246914107e-03 / 206 (NOT reproduced; see below) | `a8301121c7bf44ab` / 4.2625019757386466e-02 / 214 / 16 / 0.060879178237541014 |
| hull = membership in every arm | yes | yes | yes |

The electrolysis 2D record of lane L is not reproduced by the evict arm
while the 3D and the droplet records are. Lanes T and O changed the
library after lane L and measured this case over 100 steps only (both
found it identical); the 6330-step horizon was never re-measured.
Attribution run on the HEAD library (this lane's changes absent,
`git archive HEAD` into scratch with the `hyperct` symlink added, the
same driver, hull arm): digest `38530636a343cf7a`, KE_max
6.212277546530253e-03, 206 vertices, mass 0.05887904743753862, 375 s,
i.e. exactly this lane's evict arm. So the deviation from lane L's
record is the library of 2026-10-02 to 2026-10-05 (lanes T and O; lane
O's census of this case was 100 steps), not this lane, and the evict
arm is faithful to HEAD for all three records. The builder helper
itself is exact for this case: the off-centre builder's loop did
`pos -= L`, the helper does `pos += (-L)`, the same IEEE operation.
Which of the two lanes moved the 6330-step digest was not separated
(each would need a run on its own predecessor tree).

Meshes: the 2D electrolysis setup (`L = 0.004`, outer refinement 2)
lacked 8 of 41 outer vertices, the 3D setup (refinement 1) 2 of 35; the
vertex and wall counts in the table show them back (2D: 206 -> 214
vertices at the end of the run, 15 -> 16 frozen walls; 3D: 93 -> 95,
24 -> 26). The 2D end state changes substantially on the full mesh: the
kinetic energy of the liquid at the end of the shipped horizon is
4.26e-02 J against 6.21e-03 (6.9 x; in both arms the maximum of the
sampled series is the final sample, the case is still accelerating),
the largest speed 3.32 against 1.74 m/s. The case is unvalidated and
has no reference to say which is better; the eight missing outer
vertices were 20 % of the outer mesh at this refinement, so a change of
this size is not surprising. The masses in the table are the totals at
the end of the run (gas mass is injected every step by the callback, so
they are not conservation statements); the M1 smoke pin "5-step
per-phase mass drift <= 1.94e-15" was not re-measured here.

Shearing plate: section 4.3.

### 3.4 Dam break

Not on the path: `setup_dam_break_multiphase` builds its tank with
`rectangle` / `box` at the origin and never shifts it. Its pins
(`test_case_dam_break.py`) pass unchanged in the fast suite.

## 4. What the fix does to the open physics items (task 4)

### 4.1 The 2D over-decay (lane H of July, `projection_every`)

Not the corner. The default's l2 moves from 0.17479 to 0.17439 (-0.23 %)
and `projection_every=2` still takes it to 0.0388 (-78 %, two-fluid l2
0.0231), with the same uncalibrated tail (1.394 against 1.397). Lane H's
attribution to the every-call pressure projection stands; the missing
corner at 5 R0 was a 1.2 % deficit of outer-phase mass far from the
droplet.

### 4.2 The 3D bump / over-decay cancellation (lane G of July)

Not the corner. The `dual_only` score moves by 4e-6 relative (l2
0.248113 -> 0.248114), `R_max_peak` by 1.5e-7 relative, the tail from
0.08410 to 0.08417; the `delaunay` arm stays measured worse (1.5245 ->
1.5291). Lane G located the bump at the six valence-8 face-centre
vertices of the cube-sphere interface; the three missing outer vertices
were at the top face centre, on the diagonal and at the corner of the
box, 4 to 5 R0 away, and the numbers say they did not reach the
interface.

### 4.3 The shearing-plate instability and its 23 setup collisions

Lane L counted 23 collisions in the 2D setup: 22 in the builder's shift
and 1 in the case's anisotropic rescale. The 22 are gone with the
builder (the shearing builder mesh has 313 vertices instead of 293).
The rescale collision is of another kind (`diagnose_box_shift.py
shearing`, `results/box_shift/shearing.json`): the rescale maps the
outer vertices by `(L_x / L_build, L_y / L_build) = (1, 2/3)`, and the
two outer vertices at `(0, +-0.0075)` land on `(0, +-0.005)`, which are
the droplet's pole vertices on the interface (`R0 = 0.005`). That is a
mover onto a STAYER's key, not an ordering problem: `move_all` would
refuse it, and the loop's `on_collision='evict'` drops the interface
vertex and puts an outer-phase vertex in its place. On the pre-laneB
mesh only one of the two outer vertices existed (the other was among
the 22 lost in the shift), so one pole was damaged; on the full mesh
both poles are. The 22 lost vertices also included four of the nine
vertices of the top plate `y = +L_build` (`(0.00375, 0.015)`,
`(0.0075, 0.015)`, `(0.01125, 0.015)` and the corner) and three of the
periodic face `x = +L_build`, so the pre-laneB shearing mesh had a top
plate with 5 of 9 vertices: 13 frozen plate vertices after the periodic
merge against 16 on the full mesh.

Measured on the short window of `_run_short_2D.py` (refinement 3/3,
`shearing_plate_droplet_2D`, dt 1.926e-05, 2596 steps to t = 0.05 s;
`diagnose_box_shift.py shearrun`, `results/box_shift/shearrun.json`):

| | old (evict) | new (move_all) |
|---|---|---|
| setup: vertices / interface / frozen plate vertices / total mass | 287 / 31 / 13 / 0.8181555448190211 | 302 / 32 / 16 / 0.833041936016908 |
| setup state digest | `b60215b8ee996368` (= lane L's setup digest) | `855f8563c9733fb1` |
| lane L's three-step probe (dt 1e-5): digest / vertices | `285af826be7ff536` / 287 | `0ca6deaa1a7f766a` / 302 |
| first step with fewer interface vertices than at setup | 78 (t = 1.5e-3 s, 31 -> 30) | 183 (t = 3.5e-3 s, 32 -> 30) |
| `max u / U_wall` at t = 0.0128 / 0.0256 / 0.0383 / 0.050 s | 348 / 303 / 300 / 296 | 11.0 / 10.9 / 8.3 / 295 |
| KE [J] at the same times (KE at step 0: 6.1e-05 / 6.8e-05) | 0.279 / 0.148 / 0.449 / 2.35 | 0.0069 / 0.0034 / 0.0042 / 0.776 |
| interface vertices at t = 0.05 s | 28 | 36 |

Lane L's three-step digest `17e9a78408066fea` is not reproduced by the
evict arm although its setup digest is; lane O changed the sign of the
2D periodic area vector and counted 64 flipped vectors in 10 steps of
this case, which is the attribution (not re-run on the pre-laneO
library here). Reading: the full mesh is better for the first 0.04 s
(the velocity stays at 8 to 11 `U_wall` instead of 300) but the run is
lost all the same by t = 0.05 s, the interface loses vertices from step
183 on and reconnects into 36 "interface" vertices by the end. The
rescale needs a geometric fix in the case (scale only vertices outside
the ring, or push a rescaled vertex that lands inside `R0 + h` radially
out), which is case design, not this lane's library change; left open
with the numbers above. The 3D shearing setup is not run (it crashes on
the same rescale collision, audit 2026-09-25; `move_all` would refuse
it as well).

## 5. Known limits

1. `'evict'` reproduces the old meshes only for the builders and for
   the electrolysis off-centre builder; the shearing-plate rescale keeps
   its own `on_collision='evict'` loop in both arms (section 4.3).
2. The static-droplet runner's `baseline_equilibrium.json` (April
   2026) is not re-pinned: it predates the library path (summary
   7.41e-03 against the runner's 1.18e-03 since 2026-09-25) and nothing
   reads it.
3. `diagnose_frozen_set.py` records of lane L for `droplet_2D`,
   `electrolysis_2D` / `_3D` are regenerated on the full mesh under
   their old names; the pre-laneB records are reproduced by
   `--box-shift evict` (written as `*_bs-evict_*.json`).
4. The count of lost vertices depends on the floating-point value of
   `L`; the census above covers the three box sizes the cases use.

## 6. Tests

New `ddgclib/tests/test_box_shift.py` (15): the shift keeps every vertex
and the evict arm loses exactly the census count with the corner
missing and dangling neighbour entries, no-collision shifts are
bit-identical between the arms (keys and cache order), bad values
raise, the 2D / 3D builders hold all corners in `walls` (317 / 32 and
95 / 26 vertices / walls against 311 / 31 and 93 / 24), the total dual
volume is the box (1e-2, 1e-3) against 9.6875e-3 for the lossy 2D
refinement 2 mesh, `metadata['box_shift']`, the droplet and
electrolysis setups record the choice and hold the corners.

Re-pinned: `TestStaticDroplet3DRetopologyFloor.EXPECTED_PLATEAU_MAXF`
and `A5B_3D_PEAK` / `A5B_3D_END` 7.274172e-05 -> 7.274134e-05 (1 %
tolerance; the old value is 5.3e-6 away and would still pass, the pin
is moved so that the quoted digits are the measured ones).

Battery (ddg environment, repository root):

| suite | before the lane (lane O) | after |
|---|---|---|
| ddgclib fast (`-m "not slow"`) | 1182 passed, 12 skipped, 3 xfailed | 1197 passed, 12 skipped, 3 xfailed (1182 + 15 new; 131 s) |
| ddgclib slow (`-m slow`) | 32 passed, 1 xfailed | 32 passed, 1 xfailed (316 s under the parallel load of the full runs) |
| hyperct | 326 passed | not touched |

Every other pin is bit-identical: the dam break, hydrostatic, Hagen-
Poiseuille, frozen-set, remap and determinism tests do not build a
droplet; the droplet tests that do (fast fixtures at refinement 1/2 and
2/2) assert bounds, and the bounds hold with the margins of section
3.1.

## 7. DO-NOTs (measured)

- Do not shift a structured box with a loop of single moves; the loss
  depends on the floating-point value of the box size (6, 16 or 22 of
  145 vertices at `L` = 0.05, 0.004, 0.015).
- Do not compare a droplet, electrolysis or shearing-plate number from
  before this lane with one from after it without the `box_shift`
  entry of its `methods.json`: the meshes differ (317 against 311
  vertices in 2D, 475 against 472 in 3D).
- Do not read the 2D over-decay or the 3D bump as a consequence of the
  missing corner (section 4).
- Do not rescale outer vertices onto the interface (shearing plate):
  `move_all` refuses it, the loop deletes interface vertices.

## 8. Reproduce

    cd /home/endres/projects/ddgclib
    PY=/home/endres/anaconda3/envs/ddg/bin/python
    D=cases_dynamic/oscillating_droplet/diagnose_box_shift.py
    $PY -m pytest ddgclib/tests/test_box_shift.py -q -p no:cacheprovider
    # section 2 (seconds)
    $PY $D census --out <dir>; $PY $D census --L 0.015 --out <dir2>; $PY $D census --L 0.004 --out <dir3>
    $PY $D mesh --out <dir>
    # section 3.1 / 3.2 pins (floors about 2 min, envelope about 1 min)
    $PY $D floors --out <dir>
    $PY $D envelope --out <dir>
    # full runs (2D about 4 min each, 3D about 10 min each; --box-shift evict reproduces the pre-laneB baselines)
    $PY $D fullrun --runner 2d --box-shift move_all --out <dir> --no-anim
    $PY $D fullrun --runner 2d --box-shift evict --out <dir> --no-anim
    $PY $D fullrun --runner 2d --policy delaunay_remap_p2 --out <dir> --no-anim
    $PY $D fullrun --runner 2d --policy dual_only --out <dir> --no-anim
    $PY $D fullrun --runner 3d --box-shift move_all --out <dir> --no-anim
    $PY $D fullrun --runner 3d --box-shift evict --out <dir> --no-anim
    $PY $D fullrun --runner 3d --policy delaunay --out <dir> --no-anim
    $PY $D fullrun --runner static --box-shift move_all --out <dir> --no-anim
    $PY $D fullrun --runner static --box-shift evict --out <dir> --no-anim
    # or the shipped runners in place: oscillating_droplet_2D.py [--box-shift evict], _3D.py, static_droplet_2D.py
    # section 3.3 / 4.3
    $PY $D shearing --out <dir>
    $PY $D shearrun --out <dir>
    F=cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py
    for bs in move_all evict; do
      $PY $F droplet_2D --steps 100 --box-shift $bs --out <dir>
      $PY $F electrolysis_3D --steps 300 --box-shift $bs --out <dir>
      $PY $F electrolysis_2D --box-shift $bs --out <dir>
    done
