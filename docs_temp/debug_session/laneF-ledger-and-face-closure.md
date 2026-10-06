# laneF (October): the dam break through its horizons

Date: 2026-10-05. Brief: the dam break through the whole 0.2 s horizon in
2D at the shipped `alpha_art` and at least one lower value, a working 3D
run, pinned, the fix a registered library method. Evidence base: the July
lane F log (`laneF-dam-break-unstick.md`, blocker "air sliver-cell F/m
ejection"), lane L (walls held, run still lost 8 to 23 steps later), lane
Q (3D aborts on every edge-area source), lane S (3D setup labels).

Every arm below is `PRESETS['dam_break_2D']` / `['dam_break_3D']` or
`preset.replace(...)`, run through
`cases_dynamic/dam_break/diagnose_sliver_ejection.py` (records, per step
and per integrated vertex, the dual volume, the per-phase masses, the
force and the acceleration the preset's `dudt_fn` returns, the 1-ring,
and a census of (vertex, phase) pairs with a sub-volume but no mass
("holes") or mass but no sub-volume ("stranded")). Numbers of one process;
every run is deterministic on this machine (lane T), and the
configuration behind each is the preset named plus the `--replace` arms
given.

## 0. Verdict

- The ejection was not a sliver cell. At the flip of step 1427 (alpha
  0.2, refinement 3) two ledger defects fire at once, both invisible at
  the gauge reference `P0 = 0` of every droplet, electrolysis and shearing
  preset and lethal at `P_atm = 101325` Pa:
  1. a phase that APPEARS at a vertex across the rebuild got no mass and
     published `p_phase = 0` absolute for a sub-volume the force reads as
     present: a 1 atm hole, |F| = 439 N on a 1.1e-4 kg air cell whose
     dual volume (9.0e-5 m^2) is 64 % of the mean cell;
  2. a 50/50 interface-edge face between two interface vertices that
     both lost their last bulk liquid neighbour was dropped from both
     sides, opening each cell by A/2 to the absolute pressure: 493.6 N on
     1.58e-4 kg.
- Two new method axes, defaults flipped by protocol rule 5 with the old
  behaviour kept as `broken`: `phase_ledger` (`volume`) and
  `face_closure` (`renormalise`). Neither alone carries the run.
- `dam_break_2D`: shipped run bit-identical (`952d4544676ca366`); alpha
  0.2 and 0.1 complete the 1585-step horizon with no vertex outside.
- `dam_break_3D`: the fan cache's 1 % closure defect times the absolute
  pressure ejected an air cell at step 4; the preset reads the exact
  faces now (`edge_area_source='p_ij_simplex'`), and the setup preloads
  the hydrostatic masses on the vote labels (the criterion labels gave
  p_liq up to the EOS clip at t = 0 in 3D). The run completes its 793
  steps.
- Every other pin unchanged (fast and slow suites green, both full
  droplet runs reproduce their baselines in every key).
- Cost, stated: at the toe event the released toe vertices are kicked by
  the interface pressure jump (a transient of about 0.01 s); the front
  measure retreats because the one-cell tongue is no longer liquid in
  the ledger. Refinement 4 stays a known limit.

## 1. Task 1: the ejection, reproduced and attributed

`python cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2
--replace phase_ledger=snapshot --replace face_closure=skip` (the
pre-lane library; on the pre-lane tree the plain `--alpha 0.2` call gives
the same run).

Trace of the ejected vertex (air, bulk, interior, (0.0744, 0.0028), one
cell above the floor next to the liquid toe):

| step | t | dual vol | m | |F| | |a| | |u| | ring |
|---|---|---|---|---|---|---|---|
| 1416 .. 1425 | 0.1788 .. 0.1800 | 2.27e-5 .. 2.23e-5 | 2.82e-5 .. 2.78e-5 | 2.95e-4 .. 2.91e-4 | 1.70 .. 1.77 m/s^2 | 0.089 .. 0.087 | 3 neighbours, unchanged |
| 1426 | 0.1801 | 2.23e-5 | 2.77e-5 | 2.91e-4 | 1.78 | 0.0865 | a flip elsewhere |
| 1427 | 0.1802 | 9.02e-5 | 1.11e-4 | 439 | 3.94e6 | 497 m/s = 502 u_ref | ring changed: 4 neighbours, the new one the interface vertex (0.0473, 0.0154) |

So: a bulk vertex, not created by an inlet or a merge (the case has
neither), not a corner, and its cell did not collapse: it grew 4x in the
reconnection. The per-face terms at force time: the face to the new
neighbour reads `p_i = 101326.18`, `p_j = 0.000`, |A| = 8.67e-3 m, i.e.
`F_p = (430, -90)` N on that face alone; the other three faces carry
0.07 to 0.15 Pa differences. The mass ledger is at fault, not the
geometry: the hole census finds exactly one hole in the whole run, at
step 1427, phase 0 at (0.0473, 0.0154), `dual_vol_phase[0]` 6.5e-5,
`p_phase[0]` 0.0, one bulk air neighbour (the ejected one), and two
stranded liquid masses (0.0415 kg at (0.0714, 0.0215), 0.0333 kg at
(0.0754, 0.0053)) at the same step.

Mechanism, in the code as it was:

- `redistribute_mass_multiphase` skipped every (vertex, phase) whose
  snapshot sub-volume was zero ("newly present: keep EOS value");
  `restore_pressure_multiphase` skipped it too; `compute_phase_pressures`
  published `p_phase[k] = 0` because `m_phase[k] = 0`; the force's
  presence test (`_phase_present_at`: sub-volume > 1e-30) read the phase
  as present and booked the 0 Pa face. The audit of 2026-07-02
  (`docs_temp/audit/multiphase-momentum.md`) keyed presence on the
  sub-volume to cure the zero-gauge sentinel at `P0 = 0`; at `P_atm` the
  same test exposes the hole.
- The same flip takes the last bulk liquid neighbour from two interface
  vertices of the toe (the liquid tongue is one cell thick there). Under
  `split_method='neighbour_count'` their liquid sub-volume is then zero
  although their incident triangles are voted liquid; their liquid mass
  is stranded (snapshot ledger) and the 50/50 interface-edge face between
  them (`edge_phase_area_fractions`, by the interface TAGS) has its
  liquid half dropped at both ends ("no phase-k material at either end:
  skip from both sides"). With the hole repaired this is what ejects next:
  `--replace phase_ledger=volume --replace face_closure=skip` ejects the
  interface vertex (0.0755, 0.0053) at step 1427 with |F| = 493.6 N on
  1.58e-4 kg, closure sum frac*A = (-9.7e-5, -4.87e-3) m, 101326 Pa *
  4.87e-3 m = 493.6 N.

The shipped alpha 0.3 run has 0 holes and 0 strandings over its 1585
steps, which is why it survives, and why lane L found the walls innocent
(32 of 32 frozen in every arm here too).

## 2. Task 2: the candidate methods

### 2.1 Implemented and registered

| axis / value | where | what |
|---|---|---|
| `phase_ledger='volume'` (default) | `mass_redistribution.py:_targets_by_volume`, `restore_pressure_multiphase(adopted=)`, `_retopologize_multiphase(phase_ledger=)`, `retopologize_multiphase_periodic(phase_ledger=)` | a phase that appeared at a vertex is targeted at the sub-volume weighted snapshot pressure of the 1-ring neighbours that had it (else the phase level) and joins the conserving rescale; a phase that disappeared releases its mass into the phase pool; the remap's restore keeps the adopted pressure; frozen vertices take part in the two presence changes only. Per-phase mass conserved to 1e-12 (unit test) |
| `phase_ledger='snapshot'` (broken) | `_targets_by_snapshot` (the historic loop, verbatim) | the old rule |
| `phase_ledger='adopt'` (opt-in) | `_targets_by_volume(release=False)` | new phases adopted, lost mass kept as inertia |
| `face_closure='renormalise'` (default) | `multiphase_stress.py:_close_fractions` | fractions renormalised over the phases present at either end; returns the dict unchanged when nothing is dropped (bit-identical wherever `skip` never fired) |
| `face_closure='skip'` (broken) | the `continue` of the per-phase loop | the old rule |
| `split_method='simplex'` (opt-in) | `multiphase.py:split_dual_volumes` | each incident top simplex gives |T|/(dim+1) to its voted phase, rescaled to `v.dual_vol` (zero on a 3D hull vertex); sub-volume presence == `interface_phases` |
| `edge_area_source='p_ij_simplex'` on `dam_break_3D` | preset | lane Q's validated value |

`SolverMethods`: fields `phase_ledger`, `face_closure` (multi only; a
non-default ledger needs `redistribute_mass=True`), bound into the
partials only when not the default; the function defaults equal the axis
defaults (the first measurement of this lane ran the old ledger through a
preset that recorded the new one, because the builder binds nothing for
a default: recorded must equal applied).

### 2.2 Measured one at a time (refinement 3, 1585 steps, dt 1.262e-4)

| arm | alpha 0.3 (shipped) | alpha 0.2 | alpha 0.1 |
|---|---|---|---|
| snapshot + skip (pre-lane) | `952d4544676ca366`, KE_liq peak 1.0368602514251731e-03 J at 0.0512 s, front +18.1 mm | ejected step 1427 (497 m/s) | ejected step 765 (437 m/s) |
| volume + skip | | ejected 1427 (394 m/s, open face) | |
| snapshot + renormalise | | ejected 1427 (497 m/s, hole) | |
| **volume + renormalise (defaults)** | **identical to pre-lane** | completes: |u|max 1.00 m/s, 0 outside, 14 flip steps, KE_liq peak 3.01e-2 J at 0.1845 s, end 2.39e-3, front 0.0627 (0.1 s) / 0.0654 (0.2 s), mass 4e-15, `e98a3ce60a279df9` | completes: |u|max 1.40, 37 flip steps, `217114c814658816` |
| defaults + `split_method='simplex'` | changes every run | completes, |u|max 1.12, `2e483c47a3271a45`, KE_liq peak 3.75e-2 at 0.155 s (a lone liquid vertex voted into the air at step 1182) | completes, |u|max 1.59, `b6f00767f3851509` |
| `adopt` + renormalise | identical to pre-lane | completes, |u|max 1.003, `68300516b06ea197`, 2 stranded masses on 158 steps, |a|max 899 m/s^2 at step 1428 (35 N on a 0.039 kg interface vertex), KE_liq end 1.98e-3, front 0.0652 | |

Adoption by the protocol: the defaults arm. It survives both lower
viscosities, leaves the shipped run bit-identical, and its state is
self-consistent (no mass without volume, no volume without mass: the
census is empty on every step). `simplex` keeps the toe but moves every
interface split and every multiphase pin, and the vote still erases a
lone liquid vertex (tie to the lower phase ID = air; under the snapshot
ledger that vertex carried 0.11 kg as air and free-fell through the
floor at 0.33 m/s by step 1341). `adopt` keeps mass outside the ledger
(the level anchor and the pressure field see less liquid than exists).

What the defaults cost. At the toe event the tongue's vertices have no
liquid sub-volume under `neighbour_count`; the volume ledger releases
their 0.07 kg into the pool (conserved) and the now light interface
vertices are kicked by the liquid-air pressure jump: KE of liquid plus
interface 3.1e-3 -> 3.5e-2 J at t = 0.189 s, back to 5.4e-3 by 0.199
(alpha 0.2); at alpha 0.1 the event is at 0.097 s (7.0e-2 J, |u| 0.86
at the sample). The front measure (largest x of a liquid or interface
vertex) retreats from 0.0754 to 0.0645 m because the toe is no longer
liquid in the ledger. At refinement 2 / alpha 0.1 the whole one-cell
toe evaporates (front back to 0.05); the snapshot rule survives that
coarse run with the toe stranded as inertia (|u|max 0.39 against 1.29).

### 2.3 Candidates of the brief that were not implemented

- Mass-conserving sliver merge (`mass_conserving_merge`, a merge
  threshold): not the mechanism. The ejected cell was 64 % of the mean
  cell; the smallest integrated cell along the alpha 0.2 run is 2.2e-5
  (16 % of the mean) and accelerates at 1.7 m/s^2. A merge would not
  have closed a 1 atm hole.
- Minimum dual-volume floor in the acceleration: not the mechanism (the
  mass 1.1e-4 kg is `rho_g` times a normal cell; a floor at the mean
  cell would turn 3.9e6 into 2.5e6 m/s^2).
- Material Delaunay (lane P): single phase only, and the tank is convex
  and full, so there is no hull fill to remove.
- Both would address the refinement 4 event (section 4), which IS a
  small-cell F/m spike.
- The frozen-set policy: `membership` is the preset; 32 of 32 walls
  frozen and unmoved in every arm.

## 3. Task 3: the reference

The shipped run is unchanged: KE_liq (bulk liquid, as the runner counts
it) peaks at 1.0368602514251731e-03 J at t = 0.0512 s on the per-15-step
sample of the detector (the July log's 1.0369e-3 at 0.0506 s is the same
run on the runner's 10-step sample), decays to 5.91e-4 at 0.2 s, front
0.0681 m (+18.1 mm) at 0.2 s, mass drift -3.6e-15. The alpha 0.2 run
peaks at 2.10e-3 J at 0.066 s before its toe event.

Classical front law for the resolution used: Ritter's inviscid solution
gives the front speed 2 sqrt(g a) = 1.40 m/s, i.e. x_front = a + 2 t
sqrt(g a) = 0.33 m at 0.2 s (the far wall, x = 0.2, would be reached at
0.107 s); the Martin and Moyce surge is slower than Ritter at early
times. The simulation's front is +18.1 mm (alpha 0.3) to +25 mm (alpha
0.2, before the toe event) at 0.2 s: the column creeps under the
artificial viscosity (mu_l_eff = 46.6 Pa s, 46 600 times water; Re of
order 0.1). This is a stability case, not a validation of the dam-break
law; the artificial viscosity is the lever, and lowering it now meets the
toe event instead of an ejection.

## 4. Known limits

1. Refinement 4 (alpha 0.3, 3170 steps): the defaults carry the run to
   step 2892 (t = 0.183 s) with 0 holes; an air vertex of 4.0e-6 m^2 (11
   % of the mean cell) takes |a| 6.8e4 m/s^2 from |F| 0.34 N at step 1725
   (a genuine small-cell F/m spike, |u|max 2.6 m/s) and passes through the
   floor at 0.2 m/s at step 2892 (impenetrability is not enforced, lane
   L). The simplex arm ejects at step 1343 (|a| 1.6e5 on a 3.9e-6 cell).
2. The toe event (section 2.2): a one-cell liquid tongue is not
   representable under `neighbour_count`; the vote erases a lone liquid
   vertex. An interface-aware vote or an interface-preserving remesh is
   the next step.
3. 3D: the wall cells are zeroed under `dual_only`, so the measured
   liquid volume is 6.06e-5 of the 1.25e-4 m^3 column and the liquid mass
   0.0607 kg; the column creeps (mu_l_eff 105.8 Pa s; |u|max 0.062 m/s,
   front +5 mm). The simplex split gives 6.71e-5.
4. The `electrolysis_bubble_3D` setup carries 26 stranded pairs from step
   0 (interface vertices with mass of a phase they have no sub-volume of,
   `diagnose_phase_ledger.py electrolysis_3D`); the volume ledger
   releases them at the first rebuild. The case is unpinned and
   unvalidated; not measured further.
5. `adopt` is registered and measured on two runs only.

## 5. The 3D case

`diagnose_sliver_ejection.py --dim 3` (refinement 2, 189 vertices, 793
steps, dt 2.524e-4):

| arm | result |
|---|---|
| shipped preset before the lane (fan cache) | air vertex (0.025, 0.0625, 0.0625) at |a| 82 m/s^2 at step 0, 1.3e4 at step 4, ejected (|u| 5.9); its 15 faces carry about 20 N of absolute-pressure flux each (P_atm times |A| about 2.5e-4) that should cancel; the residual 0.26 N is the fan cache's 1 % closure defect (lane Q), 1000x the body force on the cell |
| `edge_area_source='p_ij_simplex'`, setup before the lane | ejected at step 96 (|u| 2.35, 1 outside): the liquid level had fallen to 91.7 kPa; the setup state was p_liq 101203 to 120945 Pa (the clip), because the preload used the sub-volumes of the criterion labels and the final refresh relabels by the vote (9 of 189 differ, lane S) |
| same, setup on the vote labels (one vote refresh before the preload, no mass where the vote sub-volume is zero; 2D bit-identical) | setup hydrostatic to 2.9e-11 Pa; completes 793 steps, |u|max 0.0623 m/s, |a|max 96 at step 0 decaying to 0.03, KE_liq peak 4.079e-6 J at 0.0141 s, end 3.77e-6, front 0.05252 (0.1 s) / 0.05504 (0.2 s), liquid level 101547 -> 101429 Pa, mass -2.6e-15, `639c87c7700c2c71` |
| plus `split_method='simplex'` | completes, |u|max 0.057, `adc368cdb99b3f4a` (before the preload zeroing: a 9 kPa transient at step 0 from 1 stranded pair released) |
| fan cache after the setup fix (`--replace edge_area_source=e_star_cache`) | still ejects, at step 14 (`b34233d9ec1f7c3d`) |

The preset carries `edge_area_source='p_ij_simplex'` now;
`diagnose_3d_edge_area_source.py dambreak` names the fan cache
explicitly in its 'cache' arm.

## 6. Pins (`ddgclib/tests/test_case_dam_break.py::TestDamBreakPins`)

`run_pinned(dim, alpha, refine)` runs the preset and returns KE_liq peak
and time, KE at the end, the front at 0.1 s and at the end, the per-phase
mass drift, |u|max, vertices outside, steps done.

| test | configuration | pins | time |
|---|---|---|---|
| `test_2d_refinement2_alpha01_fast` | refinement 2, alpha 0.1, 793 steps | KE_liq peak 1.2835121048480583e-03 J at 0.15876550547536425 s, end 1.2792116948716176e-03, front 0.06111878830584278 at 0.1 s, 0.05 at the end | 12 s |
| `test_2d_shipped_refinement_alpha02_full_horizon` (slow) | refinement 3, alpha 0.2, 1585 steps | 3.0123940492238016e-02 J at 0.18451126312002433 s, end 2.392063056936934e-03, front 0.06269550806263897 / 0.06543644759233601 | 90 s |
| `test_3d_smoke` | refinement 2, 50 steps | KE_liq end 4.071767240848256e-06, front 0.05031922795171791 | 6 s |
| `test_3d_full_horizon` (slow) | refinement 2, 793 steps | peak 4.079177649585958e-06 J at 0.014134925765692273 s, end 3.769706715085531e-06, front 0.05252173565415043 / 0.05503666879631933 | 100 s |

All at rel 1e-6 (2D, reconnecting) / 1e-9 (3D, fixed connectivity),
mass drift < 1e-12 per phase, no vertex outside, every value finite.
The fast 2D pin is honest about what it locks: the old options survive
that coarse run too (section 2.2); the slow refinement 3 pin is the one
the old options eject.

## 7. Pin safety and battery

- Census of presence changes on the pinned multiphase runs
  (`diagnose_phase_ledger.py`): droplet 2D (refinement 2/2, 400 steps) 0
  holes / 0 stranded; electrolysis 2D (6330 steps, `a8301121c7bf44ab`)
  0 / 0; the shipped dam break 0 / 0. So the ledger default is
  bit-identical on them, and `renormalise` returns the fractions
  unchanged wherever `skip` never fired.
- Fast suite (`pytest ddgclib/tests -q -m "not slow"`): 1236 passed with
  the new defaults before the pins were added (1232 + 5 ledger tests, the
  stale METHODS.md test deselected); final: **1240 passed**, 12 skipped,
  2 xfailed, 0 failed (1232 + 6 ledger tests + 2 fast pins), 150 s.
- Slow battery (`-m slow`): 32 passed, 1 xfailed with the new defaults
  before the pins; final: **34 passed**, 1 xfailed, 0 failed (32 + the
  two slow pins), 417 s (the two pins add about 190 s).
- Full `oscillating_droplet_2D.py` (scratch copy, 1839 steps): every key
  equal to `baselines/baseline_oscillation.json` (l2 1.7439e-01, tail
  9.9989e-01, mass 1.4836e-14, delta 0 on every key).
- Full `oscillating_droplet_3D.py` (scratch copy, 872 steps): every one
  of the 21 numeric keys equal to `baselines/baseline_oscillation_3d.json`
  (l2 2.4811e-01, tail 8.4174e-02, R_max_peak 1.0790e-02, mass
  4.8001e-14, delta 0).
- hyperct: untouched.

## 8. DO-NOTs (measured)

- Do not attribute the dam-break ejection to a sliver, to the walls or
  to the geometry: it is the ledger (one hole, two strandings, one open
  face at one flip).
- Do not key phase presence on the sub-volume in the force while the
  ledger can leave a sub-volume massless: at `P0 = 0` the hole is the
  gauge reference, at `P_atm` it is 1 atm.
- Do not drop a sub-face from both sides: the cell opens by frac * A and
  the absolute pressure acts on the gap.
- Do not run the 3D dam break (or any case at absolute pressure) on the
  `batch_e_star` fan cache: its closure defect times P0 is the force.
- Do not preload a hydrostatic state on sub-volumes of labels the
  integrator will not use (3D: criterion against vote).
- Do not read the fast 2D pin as the regression of the fix (the old
  options survive it); the slow one is.
- Do not take `adopt` or `simplex` as defaults without the measurements
  of section 2.2.

## 9. Files

`ddgclib/operators/mass_redistribution.py`, `operators/multiphase_stress.py`,
`multiphase.py`, `dynamic_integrators/_integrators_dynamic.py`,
`methods/_retopo.py`, `methods/_axes.py`, `methods/_config.py`,
`methods/_presets.py`; `tests/test_mass_redistribution.py` (+6),
`tests/test_case_dam_break.py` (+4, helper `run_pinned`);
`cases_dynamic/dam_break/src/_setup.py`, `diagnose_sliver_ejection.py`
(new), `diagnose_phase_ledger.py` (new), `README.md`;
`cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py`;
`METHODS.md`, `DEVELOPMENT.md`, `debugging_plan.md`. Kept records:
`cases_dynamic/dam_break/results/sliver_ejection/ejection_*.json` (the
shipped run, alpha 0.2 and 0.1, alpha 0.2 on the old options, 3D on the
exact faces and on the fan cache) with their `methods.json`.

## 10. Reproduce

    cd /home/endres/projects/ddgclib
    PY=/home/endres/anaconda3/envs/ddg/bin/python
    $PY -m pytest ddgclib/tests/test_case_dam_break.py ddgclib/tests/test_mass_redistribution.py -q -m "" -p no:cacheprovider
    # section 1 (the ejection on the old options) and 2.2
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace phase_ledger=snapshot --replace face_closure=skip
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace face_closure=skip
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace phase_ledger=snapshot
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.1
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace split_method=simplex
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace phase_ledger=adopt
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4     # about 12 min
    # section 5
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --dim 3
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --dim 3 --replace edge_area_source=e_star_cache
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --dim 3 --replace split_method=simplex
    # section 7
    $PY cases_dynamic/dam_break/diagnose_phase_ledger.py droplet_2D --steps 400
    $PY cases_dynamic/dam_break/diagnose_phase_ledger.py electrolysis_2D --steps 6330
    $PY cases_dynamic/dam_break/diagnose_phase_ledger.py electrolysis_3D --steps 300
