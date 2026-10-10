# laneV: impenetrability (a library wall clamp) and the interface vote

Date: 2026-10-07 (cloud session, branch `claude/relaxed-davinci-tkh95t`,
Python 3.13.16, numpy 2.5.3, scipy 1.18.1; every 2D number below is the
value of one process, process-independent on this machine by protocol
rule 8; digests are this machine's and differ from the owner's machine
for the reconnecting runs, see section 8).
Brief: the two blockers every free-surface case shares. (1) Walls are
vertices and a fluid vertex can pass between two of them (laneL known
limit 1: HP2D with a vertex 5.1e-3 outside the top wall at t = 30, the
dam break at refinement 4 and its `split_method='simplex'` arm ending
with a vertex through the floor). (2) The simplex vote erases a lone
liquid vertex (laneF follow-up: a tie goes to the lower phase ID), and a
mass-conserving sliver merge for the refinement 4 small-cell spike.
Goal: a library wall clamp behind a registered BC and a method axis,
replacing the case-local clamps; an interface-aware vote as a registered
option; the sliver merge extended to `m_phase` behind an axis; each
measured one at a time through `preset.replace(...)` on the dam break
(alpha_art 0.1 and refinement 4), on HP2D (profile error against the
analytical solution) and on the electrolysis case (Laplace jump against
the analytical value, gas mass drift); adoption by the flip rule.

## 0. Verdict

- Three new method axes: `wall_clamp` (None / `'project'`),
  `simplex_vote` (`'bulk_majority'` / `'mass_fraction'`) and
  `merge_method` (`'merge_all'` / `'mass_conserving'`). The library
  class `ddgclib._boundary_conditions.WallClampBC` (planes or a box, a
  put-down gap, the frozen set excluded) replaces the case-local
  `WallClampBC` of the electrolysis case bit-identically (every
  electrolysis digest unchanged); it is built by
  `SolverMethods.wall_clamp_bc(...)` in the dam break, Hagen-Poiseuille
  2D and electrolysis setups, so the policy is recorded and applied
  together. The dynCA clamp of `capillary_rise` is untouched (section 7).
- ADOPTED: `wall_clamp='project'` on `dam_break_2D`, `dam_break_3D`
  and `hagen_poiseuille_2D` (and recorded on the three electrolysis
  presets whose setups already applied it). The clamp is one-sided and
  the identity wherever impenetrability holds: no pin moves (refinement
  3 dam break at alpha 0.3 / 0.2 / 0.1 bit-identical with 0 put-backs,
  the HP2D developing pin equal to the digit, the 3D dam break and the
  electrolysis pins unchanged). The dam break at refinement 4 (alpha 0.3,
  3170 steps) completes its horizon instead of ending at step 2892 with
  an air vertex 1.2e-6 m under the floor; the pre-laneH HP2D
  configuration of laneL goes from 54 vertices outside the walls to 0
  and its profile error against the analytical Poiseuille profile from
  l2 0.554 to 0.399 (transverse velocity 0.556 to 8.4e-4).
- The put-down gap matters: ON the wall line (gap 0) the put-back vertex
  is a hull vertex whose half cell is open to the absolute pressure
  (633 N on 2e-6 kg at refinement 4, |a| 3e8 m/s^2 absorbed by the clamp
  every step); the default gap of 0.1 wall vertex spacing keeps it
  strictly inside with a closed cell (section 2.1).
- NOT adopted, registered with their evidence: `simplex_vote=
  'mass_fraction'` keeps the one-cell liquid tongue of the dam break
  (front 0.0772 instead of 0.0654 at alpha 0.2, the toe kick gone, KE
  peak 4.2e-3 instead of 2.8e-2 J) but only with `split_method=
  'simplex'` AND the merge; with the preset's `neighbour_count` split it
  blows up at step 0 (the half-fractions of the neighbour count make
  every (liquid, air, interface) triangle a tie that the preloaded
  density tips, section 3); on the electrolysis bubble the vote alone is
  bit-identical and simplex + vote collapses the bubble.
  `merge_method='mass_conserving'` (`merge_cdist` 0.1 h) removes the
  refinement 4 small-cell spike (|u|max 2.59 to 0.51 m/s, |a|max 6.8e4 to
  8.4e3 m/s^2) but does not carry the run alone (the floor crossing
  remains); with the clamp it completes the horizon. It is opt-in: it
  never fires at refinement 3 alpha 0.3 / 0.2 and changes the alpha 0.1
  run by one merge, and the preset cannot carry a mesh-dependent
  `merge_cdist`.
- Fast suite, slow battery and hyperct: green by the session's definition
  (section 6); three new test files, one fast pin per adopted method.

## 1. Task 1: the blockers reproduced through the presets

| blocker | command | result on the pre-lane library |
|---|---|---|
| HP2D, laneL's configuration (hull inlet, two-point flux, L 15, dt 0.01, 3000 steps) | `diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000 --arm membership` | 64 walls frozen and unmoved; first vertex outside 0 <= y <= D at step 1248, up to 54 outside, 15 at the end; KE_max 0.1532, |u|max 0.708; profile on x in [L/2, L] against the analytical Poiseuille profile: l2 0.5537, u_max 0.2737 (analytical 0.2), transverse 0.556 |
| dam break refinement 4, alpha 0.3 (3170 steps, dt 6.31e-5) | `diagnose_sliver_ejection.py --alpha 0.3 --refine 4` | ends at step 2892 (t = 0.1826 s) with one air vertex outside: (0.0653, -1.2e-6), dual volume 2.35e-6 m^2 (6.5 % of the mean cell), speed 0.2 m/s; the small-cell F/m spike of laneF known limit 1 at step 1725 (|a| 6.8e4 m/s^2 on a 4.0e-6 m^2 air cell at (0.0626, 0.0010), |u|max 2.59 m/s) is survived; 0 holes, 0 strandings, 86 flip steps, 949 s |
| the same with `split_method='simplex'` | `... --refine 4 --replace split_method=simplex` | ejected at step 1343 (|u| 6.9 m/s = 7 u_ref): an air sliver between two floor vertices, (0.0620, 0.0010), dual volume 3.9e-6 m^2, 3 neighbours, 0.62 N of VISCOUS force on 4.8e-6 kg (|a| 1.6e5) |
| the lone liquid vertex erased by the vote (laneF: the simplex arm, step 1182) | `--alpha 0.2 --replace split_method=simplex` with the new census of the detector (section 1.1) | 2 events: at step 1182 an interface vertex at (0.0697, 0.0053) carrying 0.1097 kg of liquid is relabelled bulk air by the vote and its whole liquid mass leaves the vertex (released into the pool by the volume ledger), at step 1206 another at (0.0672, 0.0196) with 0.0503 kg: 0.160 kg of liquid, the toe, gone in two steps. Under the preset's `neighbour_count` split the same toe event releases 0.0748 kg at step 1427 (two interface vertices keep their label but lose their liquid sub-volume: laneF's cost) |

### 1.1 Detector additions (`cases_dynamic/dam_break/diagnose_sliver_ejection.py`)

`--gap` (the clamp's put-down distance, default the setup's), a census
of bulk vertices relabelled straight into the other bulk phase between
two steps (0 in every run of this lane: the erasure goes through the
interface state), a census of (vertex, phase) masses that leave a vertex
whole in one step (count, mass per phase, the first 200 events with the
label before and after), and the clamp activity (put-backs, steps,
first step). `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py`:
`--replace axis=value` and `--gap` for the hp2d case, the clamp
activity, and the profile error of `src/_metrics.py:profile_error` on
the downstream half at the end.

## 2. Task 2: the three methods, measured one at a time

### 2.1 `wall_clamp='project'` (the library `WallClampBC`)

Each step every vertex not in the excluded set (`bV`) that is past a
plane is put back to `level + direction * gap` on that axis and loses its
velocity component into the wall (the component away from it is kept).
Target vertices of the `BoundaryConditionSet` are ignored (the set's
default target is `bV`, the excluded set), as the electrolysis clamp did.
`SolverMethods.wall_clamp_bc(planes= | box=, axes=, min_gap=, exclude=)`
returns the BC or `None`. Setups: the dam break (`clamp_gap=None` = 0.1
of the wall vertex spacing along the floor, from the new helper
`wall_vertex_spacing`), `setup_poiseuille_2d_lagrangian` and
`setup_poiseuille_developing` (2D wall lines; the round 3D pipe wall
raises), the electrolysis setup (bottom electrode and top wall, gap 0.02
R0 as before; `use_wall_clamp` is honoured only when `methods=None`) and
`setup_fritz_dynamics`. The resolved gap is in `params['clamp_gap']`.

Dam break, refinement 3, 1585 steps (the preset = the clamp now;
`.replace(wall_clamp=None)` = before the lane):

| alpha | preset digest (this machine) | `wall_clamp=None` | put-backs |
|---|---|---|---|
| 0.3 (shipped) | `b69bcb3d7fce85df` | identical | 0 |
| 0.2 | `cc2a87e93c46209b` | identical | 0 |
| 0.1 | `1f7e46cb66f4fadb` | identical | 0 |

No vertex leaves the tank at refinement 3, so the BC never fires: the
fast pin (refinement 2 / alpha 0.1), the slow pin (refinement 3 / alpha
0.2) and both 3D pins pass unchanged (section 6).

Dam break, refinement 4, alpha 0.3, 3170 steps, KE_liq on the detector's
31-step sample:

| arm | steps | outside | KE_liq peak | KE_liq at 0.1 s | front at 0.1 s / end | |u|max | |a|max | vertices |
|---|---|---|---|---|---|---|---|---|
| `wall_clamp=None` (before the lane) | 2893 (ends, 1 outside) | 1 at step 2892 | 7.91e-3 at 0.141 s | 3.6465e-3 | 0.06595 / 0.07206 | 2.592 (step 1725) | 6.8e4 | 545 |
| clamp, gap 0 (on the wall line) | 3170 | 0 | 1.03e-2 at 0.198 s | 3.6465e-3 | 0.06595 / 0.06918 | 2.592 | 3.0e8 (section 2.1, below) | 545 |
| clamp, gap 1.25e-3 (= 0.1 x 0.0125, the setup default) | 3170 | 0 | 9.31e-3 at 0.200 s | 3.5892e-3 | 0.06593 / 0.06923 | 2.530 | 7.5e4 (an interior 3.3e-6 m^2 cell at (0.0737, 0.0025), step 3079) | 545 |
| merge only (section 2.3) | 2764 (ends, 1 outside) | 1 at step 2763 | 8.10e-3 at 0.143 s | 3.9371e-3 | 0.06713 / 0.07088 | 0.504 | 8.4e3 | 544 |
| clamp gap 0 + merge | 3170 | 0 | 9.37e-3 at 0.200 s (still rising) | 3.9371e-3 | 0.06713 / 0.06872 | 0.510 | 4.2e7 | 541 |
| clamp gap 1.25e-3 + merge | 3170 | 0 | 1.06e-2 at 0.200 s (still rising) | 3.9371e-3 | 0.06713 / 0.06904 | 3.663 | 1.05e5 (interior 2.3e-6 m^2 cell at (0.0740, 0.0025), step 3059) | 544 |
| simplex + mass_fraction + merge + clamp (section 3) | 3170 | 0 | 7.15e-3 at 0.047 s | 1.7402e-3 | 0.06931 / 0.08146 | 0.430 | 8.6e2 | 541 |

Until the first event the arms are the base run (KE and front equal at
0.1 s). The gap: with the vertex put down ON the wall line (gap 0) the
Delaunay makes it a hull vertex between the two wall vertices (tagged
`boundary`, half cell): the absolute pressure acts on its open half
cell, 633 N on an air cell of 1.7e-6 m^2 and 2.1e-6 kg (|a| 3.0e8 m/s^2
at (0.0659, 0.0), recorded on 10 of the last 12 samples), and only the
clamp's zeroing of the inward velocity every step keeps the state
finite (|u|max stays 2.59, the step-1725 spike). With the default gap of
0.1 wall spacing the vertex is interior with a closed cell, and the
clamp is an exclusion band: it fires from step 1327 on (2457 put-backs
on 1790 steps, about one vertex per step held at y = gap while it slides
along the floor with its inward velocity discarded, a free-slip contact
on a virtual wall 1.25 mm above the floor) instead of once at the
crossing, so the run departs from the base run earlier (KE_liq at 0.1 s
3.5892e-3 against 3.6465e-3) and its largest accelerations are the
usual interior small-cell spikes (7.5e4 m/s^2 on a 3.3e-6 m^2 cell)
instead of the open half cell; |u|max 2.53 (the step-1725 spike of the
base run, before the clamp ever fires). Both gaps complete the horizon;
the default is the closed-cell one. With the merge the gap arm reaches
|u|max 3.66 (a 2.3e-6 m^2 cell at the toe, step 3059) against 0.51 at
gap 0: the merge and the band interact at the toe, so the merge stays
opt-in (section 2.3).
`wall_vertex_spacing(bV, axis=gravity_axis, level=0, along=0)` is 0.05 /
0.025 / 0.0125 m at refinement 2 / 3 / 4.

HP2D, laneL's configuration (`diagnose_frozen_set.py hp2d --L 15 --dt
0.01 --steps 3000 --arm membership`, the preset's `wall_clamp` applied
by the setup; `--replace wall_clamp=None` = before the lane):

| arm | outside (max / end) | first outside | put-backs | KE_max / KE_end | |u|max | profile l2 on [L/2, L] | u_max (analytical 0.2) | transverse |
|---|---|---|---|---|---|---|---|---|
| `wall_clamp=None` | 54 / 15 | step 1248 | 0 | 0.1532 / 0.1440 | 0.708 | 0.5537 | 0.2737 | 0.556 |
| `--gap 0` | 1 / 0 (one vertex past the outlet buffer end, not a wall) | 1796 | 40056 on 1749 steps | 0.0870 / 0.0828 | 0.293 | 0.4096 | 0.2584 | 0.0577 |
| gap 0.05 (= 0.1 x 0.5, the setup default) | 0 / 0 | never | 109591 on 1849 steps | 0.1123 / 0.1123 | 0.253 | 0.3990 | 0.2438 | 8.4e-4 |

The profile error does not grow, it falls (0.554 to 0.399); the
configuration is still laneH's "not validated" one (hull inlet, pressure
advected with the vertices, two-point flux, 45 vertices in the window),
so 0.399 is not a validation of the channel. The shipped developing
preset (`run_developing`, the fast pin: L 3, mu 0.1, refinement 1, 500
steps) is bit-identical with and without the clamp: l2
0.013084885355719682, u_max 0.1495783625570828, nothing outside, 0
put-backs (`test_case_hagen_poiseuille.py` unchanged). 3D
(`hagen_poiseuille_3D`): the pipe wall is round, no clamp; the setup
raises on a request.

Electrolysis: the presets `electrolysis_bubble_2D` / `_3D` /
`_fritz_2D` carry `wall_clamp='project'` as the record of the clamp
their setups have applied since 2026-07 (gap 0.02 R0 = 2e-5 m); the
library class is arithmetically the case class, every digest of
`test_case_electrolysis_bubble.py` is unchanged (3D static
`ed70a7c01e8a2f0a`, injection `eac3602aa359adb0`, 2D `22a178b7ef93e21f` /
`30d427443eae547e`, the slow 2/2 static), the static bubble of laneG
reproduces (224.0641 Pa at 2000 steps, digest `7f61e7da0348ecf0`).

### 2.2 `simplex_vote='mass_fraction'`

Every vertex votes with the weight of phase k equal to its
volume-equivalent mass share `(m_phase[k] / rho0_k) / sum_j (m_phase[j] /
rho0_j)`; a vertex without a ledger votes by its label; the simplex goes
to the largest total, a tie to the lower phase ID. Under the bulk
majority a triangle (liquid bulk, air bulk, interface) is always a tie
(-> air), and an interface row that still carries liquid mass cannot
vote at all. Wired: `mps.refresh(simplex_vote=)` at the setups' runtime
refreshes and in `multiphase_rebuild_with_ledger` (both multiphase
retopology paths); bound into the partial only when not the default.

Dam break, refinement 3, 1585 steps, one axis at a time
(`diagnose_sliver_ejection.py --alpha ... --replace ...`):

| arm | alpha | result |
|---|---|---|
| preset + `simplex_vote='mass_fraction'` | 0.2 and 0.1 | EJECTED at step 0 (|u| 250 m/s at the column corner (0.05, 0.05)): the setup's final refresh re-votes the preloaded state; under `neighbour_count` the interface vertices carry exact half fractions, every (liquid, air, interface) triangle of the column face is a tie between the two votes, and the preloaded liquid density (rho(P) > rho0) tips it to liquid, so the interface vertex (0.05, 0.0375) keeps its liquid mass on a sub-volume one third as large: p_liq at the EOS clip 120945 Pa. DO-NOT combine with `neighbour_count` |
| `split_method='simplex'` + vote | 0.2 | the tongue is kept (front 0.0760 at the end against 0.0657 for simplex alone, KE_liq peak 6.5e-3 against 3.75e-2 J: no toe kick, 0 whole-mass releases), but EJECTED at step 1407: a 9.1e-6 m^2 air cell next to the floor at (0.0737, 0.0010) takes 0.75 N of viscous force (|a| 6.7e4, |u| 6.5 m/s): the small-cell F/m spike |
| simplex + vote + merge (cdist 2.5e-3) | 0.2 | completes: front 0.0772, KE_liq peak 4.19e-3 at 0.087 s, end 1.4e-4, |u|max 0.491, |a|max 693, 14 flip steps, 2 whole-mass events (0.0159 kg of liquid at a free-surface vertex (0.041, 0.043) at step 1040, 6.4e-5 kg of air at the floor corner), 143 vertices (2 merges), digest `73082212bde3fd2f` |
| the same | 0.1 | completes: front 0.0854, KE_liq peak 3.43e-2 at 0.127 s (preset 2.32e-2 at 0.102), end 6.7e-4 (5.7e-3), |u|max 2.85 (1.40), |a|max 3406 (2729), 180 flip steps (37), 7 whole-mass events / 0.134 kg of liquid (preset: 3 / 0.110), 143 vertices, digest `e91ed94b8f0c1dd6` |
| the same + clamp | refinement 4, alpha 0.3 | completes (table of 2.1): front 0.0815 at the end, |u|max 0.43, but 110 whole-mass events moving 1.213 kg of liquid through the pool (the column holds 2.5 kg): the ledger reshuffles the toe continuously |

Electrolysis static bubble (3D, refinement 1/1, 2000 steps, g = 0, no
injection; `diagnose_static_bubble.py --dim 3 --ro 1 --rd 1 --steps
2000`, jump against 2 gamma / R0 = 144 Pa):

| arm | jump at the end | V / V_exact (setup -> end) | gas cells / interface | gas mass drift | digest |
|---|---|---|---|---|---|
| preset (`dual_only`, neighbour_count) | 224.064 Pa (+0.556) | 1.1305 -> 1.1296 | 35 / 26 | 2.8e-15 | `7f61e7da0348ecf0` |
| `.replace(simplex_vote='mass_fraction')` | 224.064, bit-identical | | | | `7f61e7da0348ecf0` |
| `.replace(split_method='simplex')` | 289.091 (+1.008) | 0.7797 -> 0.7786 | 35 / 26 | 8.1e-15 | `f99c65e872210b2f` |
| simplex + vote | 101172.6 (+701.6): the setup already relabels 8 gas cells (V / V_exact 0.5755, jump 186.2 at t = 0), the bubble collapses to 15 cells / 14 interface vertices | 0.5755 -> 0.0664 | 15 / 14 | -1.8e-15 | `16e6e6631729d306` |

Verdict: the vote reads the ledger as intended (the tongue survives, the
kick vanishes) but it needs the simplex split, the merge and a setup
whose preload is made on its own labels; measured-worse at alpha 0.1
(speeds doubled) and on the bubble. Registered `experimental` with
these DO-NOTs; not on any preset.

### 2.3 `merge_method='mass_conserving'`

`mass_conserving_merge(HC, cdist, prefer=bV)`: the survivor takes the
sum of `m`, of `m_phase` per phase and of the momentum (`u`
mass-weighted), `p` averaged, a member of `bV` preferred as the survivor
(laneL known limit 4: `merge_all` could merge a wall vertex into a mobile
one); the label and `p_phase` of the survivor are left to the next
refresh. Applied in `_retopologize` step 0 in place of `HC.V.merge_all`
when `merge_cdist > 0` and the Delaunay / adaptive rebuild runs;
`SolverMethods` refuses the value without `merge_cdist` or under another
connectivity. Per-phase ledger exact (`test_merge_method.py`:
mass, `m_phase` and momentum to 1e-14; `merge_all` loses the merged
vertex's mass).

| run | cdist (0.1 wall spacing) | result |
|---|---|---|
| refinement 3, alpha 0.3 / 0.2 | 2.5e-3 | never fires: bit-identical (`b69bcb3d7fce85df`, `cc2a87e93c46209b`) |
| refinement 3, alpha 0.1 | 2.5e-3 | one merge late in the run (145 -> 144 vertices): KE_liq peak equal (2.3158e-2 at 0.1024 s), end 5.15e-3 against 5.72e-3, front 0.06743 against 0.06750, |u|max equal, digest `0ee49ba2a4550aa7` against `1f7e46cb66f4fadb` |
| refinement 4, alpha 0.3 | 1.25e-3 | the step-1725 spike is gone (|a|max 6.8e4 -> 8.4e3, |u|max 2.59 -> 0.50 m/s, 545 -> 544 vertices) but the run still ends with an air vertex under the floor at step 2763 (0.18 m/s): impenetrability is the clamp's job; with the clamp (gap 0): completes, 541 vertices (4 merges), |u|max 0.51 |

Opt-in (status `opt-in`): the gain is real at refinement 4 and the
ledger is exact, but a preset cannot carry a mesh-dependent
`merge_cdist`, and at refinement 3 the one merge of the alpha 0.1 run
would move that pin for nothing. The old `merge_all` stays the default
(validated: no preset sets `merge_cdist`).

## 3. Task 3: adoption

By protocol rule 5: the clamp is a preset change on `dam_break_2D`,
`dam_break_3D` and `hagen_poiseuille_2D` (and the record on the three
electrolysis presets). The flip rule: every pinned run is bit-identical
(no vertex leaves in any pinned run, so the one-sided BC is the
identity), and the two runs that the brief names as blocked complete.
No pin moves; nothing to re-pin. The old behaviour is the registered
value `None` (`.replace(wall_clamp=None)`, status `validated` on every
pin until this lane). `simplex_vote` and `merge_method` stay at their
old defaults (sections 2.2 and 2.3).

Known fragility that the adoption carries: the clamp handles the planes
the setup names (the four tank walls, the two channel wall lines, the
electrode and the top wall); a vertex that steps past the outlet end of
the HP2D buffer (the one "outside" of the gap-0 arm) is not a wall and
is left to the outlet BC.

## 4. Task 4: pins

- `test_wall_clamp.py` (fast, 8 tests): the BC (planes, box, put-back,
  inward velocity zeroed, outward kept, gap, excluded walls, target
  ignored), the builder and validation, the electrolysis / HP2D / dam
  break wiring, `test_identity_on_a_run_that_stays_inside` (40 steps of
  the refinement 2 dam break: preset == `.replace(wall_clamp=None)` to
  the bit) and the pin `test_pushed_through_the_floor`: an air vertex
  displaced 4 mm under the floor with 0.3 m/s downward is put back at the
  gap 0.1 x 0.05 m after one step with no downward velocity, stays inside
  for 30 steps, digest `39710661593753f5` (`PIN_PUSHED_DIGEST`); without
  the clamp it stays outside.
- `test_simplex_vote.py` (fast, 8 tests): the hand-labelled grid (bulk
  majority unchanged, the ledger read, the (liquid, air, interface) tie
  decided by the interface share, a vertex without a ledger votes by its
  label), the binding, the dam break setup on the arm (labels of every
  interface-free simplex equal, 20 steps finite).
- `test_merge_method.py` (fast, 4 tests): the ledger sums, the wall
  survives, the `_retopologize` axis against `merge_all`, the builder
  and refusals.
- The preset flips leave every existing pin unchanged (section 6).

## 5. Known limits

1. The clamp is planar: a curved wall (the 3D pipe) has none; a general
   wall (the polyline / surface of the frozen wall vertices) would need a
   projection onto the nearest wall facet and is not implemented.
2. Gap 0 puts the vertex on the hull with an open half cell at the
   absolute pressure (section 2.1): use the default (0.1 wall spacing) or
   a positive gap.
3. The clamp does not stop a vertex from passing between two wall
   vertices of a wall the setup did not name (the HP2D outlet buffer
   end).
4. `simplex_vote='mass_fraction'` is fragile under `neighbour_count`
   (exact half fractions) and needs the simplex split, the merge and a
   setup whose preload is made on its own labels; it reshuffles the
   ledger at refinement 4 (1.2 kg through the pool).
5. `merge_method` is applied on the Delaunay / adaptive path only (the
   periodic rebuild keeps `HC.V.merge_all`), and `merge_cdist` is a
   fixed length: a preset cannot scale it with the mesh.
6. Not measured: the electrolysis injection horizon with the vote (the
   static bubble is bit-identical, the injection was not re-run), the
   3D dam break with the vote or the merge (dual_only: the merge does not
   apply), the 2D electrolysis static bubble with the vote.
7. The r4 arm of section 2.1 with an explicit `--gap 1.25e-3` and the
   run on the preset default (`diagnose_sliver_ejection.py --alpha 0.3
   --refine 4`, gap `0.1 * 0.012499999999999983`) differ in the last bits
   of the put-down position: the preset run completes its 3170 steps with
   the same summary to the printed digits (KE_liq peak 9.3064e-3 at
   0.1996 s, end 8.351556832e-3 against 8.351556833e-3, front 0.06923,
   |u|max 2.530, |a|max 7.49e4, 2457 put-backs on 1790 steps from step
   1327, 40 whole-mass events / 0.0918 kg) and a different digest
   (`81cd39f4c5a203bd` against `f9e1c33376598f3c`): the reconnecting run
   amplifies the 1e-16 difference to 1e-10 relative at the end (rule 8).
   Quote the preset run.
8. Two vertices clamped onto the same corner key in one step raise
   hyperct's `VertexCollisionError` (seen only in the step-0 blow-up of
   the `mass_fraction` + `neighbour_count` arm, where the clamp meets
   vertices flying out of the tank at 250 m/s): a lost run, not a clamp
   defect, but the exception is the failure mode.

## 6. Battery

The implementer was cut off (model usage limit) before running the
suites; the session owner ran them on the final tree of this lane
(uncommitted state, 2026-10-07, this machine), from the repo root with
`PYTHONPATH=<ddgclib root>`:

| suite | result |
|---|---|
| fast, `-m "not slow"` | 1 failed, 1327 passed, 11 skipped, 2 xfailed in 577 s; the failure is the environment baseline (a) `test_single_phase_remap.py::TestBoxDecayWithEOS::test_remap_matches_fixed_connectivity` (one ulp) |
| slow, `-m slow` | 1 failed, 41 passed, 1 xfailed in 4438 s; the failure is the environment baseline (b) `test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column` (0.15269571220608844 against the pin 0.15305813130485327) |
| hyperct | untouched by this lane; not re-run |

Green by the session's definition (docs_temp/HANDOFF_2026-10-07.md).
This lane has NOT been independently reviewed: see the hand-off.

## 7. What the dynCA clamp would take

`cases_dynamic/capillary_rise/src/_setup_dynca.py` and the dynCA runners
are read-only for this lane. Their wall treatment is a contact-line
model (the dynamic contact angle sets the slip of the contact vertex),
not a put-back clamp: the library class would replace only the part
that keeps interior vertices off the tube wall, by
`methods.wall_clamp_bc(planes=[(0, 0, +1), (0, 2 half_width, -1)],
exclude=bV)` in 2D (the slit walls) and a cylindrical wall for the 3D
tube, which `WallClampBC` does not have (limit 1). Nothing of this lane
moves the dynCA numbers: the capillary-rise setups do not take
`methods` for a clamp and bind no vote or merge.

## 8. DO-NOTs (measured)

- Do not put a clamped vertex ON the wall line (gap 0): hull vertex, open
  half cell, 633 N at atmospheric pressure (section 2.1).
- Do not use `simplex_vote='mass_fraction'` with `split_method=
  'neighbour_count'` (step-0 ejection) or without the merge (the
  simplex + vote arm ejects at step 1407 on a 9e-6 m^2 cell), or on the
  electrolysis bubble with the simplex split (collapse at setup).
- Do not read the refinement 4 clamp-only run as clean: its survival at
  gap 0 is the clamp absorbing 3e8 m/s^2 every step.
- Do not compare digests across machines: the alpha 0.2 defaults run is
  `e98a3ce60a279df9` on the owner's machine (laneF) and
  `cc2a87e93c46209b` here with KE_liq peak 3.0123940492238016e-02 on the
  every-step sample of `run_pinned` (the slow pin passes here) and
  2.765e-2 on the detector's 15-step sample.

## 9. Files

Library: `ddgclib/_boundary_conditions.py` (`WallClampBC`,
`wall_vertex_spacing`), `ddgclib/multiphase.py`
(`assign_simplex_phases_from_vertices(vote=)`, `refresh(simplex_vote=)`,
`mass_conserving_merge(prefer=)` with `m_phase`),
`ddgclib/dynamic_integrators/_integrators_dynamic.py`
(`_retopologize(merge_method=)`, `_retopologize_multiphase(simplex_vote=,
merge_method=)`), `ddgclib/methods/_retopo.py`
(`retopologize_multiphase_periodic(simplex_vote=)`,
`multiphase_rebuild_with_ledger(simplex_vote=)`), `ddgclib/methods/
_axes.py` (three axes), `_config.py` (fields `wall_clamp`,
`simplex_vote`, `merge_method`, builder `wall_clamp_bc`, validation,
partial bindings), `_presets.py` (the flips and records). Cases:
`cases_dynamic/dam_break/src/_setup.py` (`clamp_gap`, the vote at the
runtime refreshes), `diagnose_sliver_ejection.py` (section 1.1),
`cases_dynamic/Hagen_Poiseuile/src/_setup.py` (`methods=`, `clamp_gap`,
`_wall_clamp`), `src/_run.py` (passes `methods` to the setup),
`diagnose_frozen_set.py` (section 1.1),
`cases_dynamic/electrolysis_bubble/src/_setup.py` (the library clamp
through the axis), `electrolysis_bubble_fritz_2D.py`,
`diagnose_static_bubble.py` (`--no-clamp` = `.replace(wall_clamp=None)`).
Tests: `test_wall_clamp.py`, `test_simplex_vote.py`,
`test_merge_method.py` (new). Docs: `METHODS.md`, `debugging_plan.md`,
this log. hyperct: untouched.

## 10. Reproduce

    cd /home/user/ddgclib
    PY="env PYTHONPATH=/home/user/ddgclib /usr/bin/python"
    $PY -m pytest ddgclib/tests/test_wall_clamp.py ddgclib/tests/test_simplex_vote.py ddgclib/tests/test_merge_method.py -q -p no:cacheprovider
    # section 1 (the blockers; --replace wall_clamp=None is the pre-lane preset)
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000 --arm membership --replace wall_clamp=None --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --replace wall_clamp=None --out /tmp/laneV     # 16 min
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --replace split_method=simplex --replace wall_clamp=None --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace split_method=simplex --out /tmp/laneV
    # section 2.1 (the preset = the clamp with the default gap)
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --gap 0 --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.3 --refine 4 --replace merge_cdist=1.25e-3 --replace merge_method=mass_conserving --out /tmp/laneV
    for a in 0.3 0.2 0.1; do $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha $a --out /tmp/laneV; $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha $a --replace wall_clamp=None --out /tmp/laneV; done
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000 --arm membership --out /tmp/laneV
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000 --arm membership --gap 0 --out /tmp/laneV
    # section 2.2
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace simplex_vote=mass_fraction --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace split_method=simplex --replace simplex_vote=mass_fraction --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.2 --replace split_method=simplex --replace simplex_vote=mass_fraction --replace merge_cdist=2.5e-3 --replace merge_method=mass_conserving --out /tmp/laneV
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.1 --replace split_method=simplex --replace simplex_vote=mass_fraction --replace merge_cdist=2.5e-3 --replace merge_method=mass_conserving --out /tmp/laneV
    $PY cases_dynamic/electrolysis_bubble/diagnose_static_bubble.py --dim 3 --ro 1 --rd 1 --steps 2000 --out /tmp/laneV
    $PY cases_dynamic/electrolysis_bubble/diagnose_static_bubble.py --dim 3 --ro 1 --rd 1 --steps 2000 --replace simplex_vote=mass_fraction --out /tmp/laneV
    $PY cases_dynamic/electrolysis_bubble/diagnose_static_bubble.py --dim 3 --ro 1 --rd 1 --steps 2000 --replace split_method=simplex --replace simplex_vote=mass_fraction --out /tmp/laneV
    # section 2.3
    $PY cases_dynamic/dam_break/diagnose_sliver_ejection.py --alpha 0.1 --replace merge_cdist=2.5e-3 --replace merge_method=mass_conserving --out /tmp/laneV
    # suites: rule 7 of the brief
