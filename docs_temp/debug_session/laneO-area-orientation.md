# laneO: orientation of the 2D dual area vector

Date: 2026-10-05. Closes the finding of laneH (section 7 of its log) and
the strict xfail `test_2d_dual_area_vectors_close_on_a_sheared_jittered_mesh`.

A first attempt at this lane stopped early (usage limit). It left the
reference helper `simplex_area_vectors` in `stress.py` and the probe
`cases_dynamic/diagnose_area_orientation.py` in the working tree, both
unreviewed and unused. Both were reviewed (section 1) and kept; every
measurement in this log is from this attempt.

## 0. Verdict

`dual_area_vector` (2D) returned, on some edges of sheared or jittered
meshes, a vector that points AGAINST its edge. The 2D dual face of an
edge is the segment between the two dual vertices its endpoints share
(two barycentres, or a barycentre and the edge midpoint on the hull); the
magnitude of the vector was right, its sign was taken from the midpoint
of the segment as seen from `x_i`. That rule is wrong whenever the two
triangles at the edge subtend more than 180 degrees at `x_i`, i.e. when
`x_i` lies inside the triangle of the three other vertices. The exact
rule is the sign of `A_ij . d_ij`, which for the barycentric segment is
`(2/3) (|T_left| + |T_right|) > 0` on any valid pair of triangles and
`(2/3) |T| > 0` on a hull edge. With it `A_ij = -A_ji` exactly, the cell
of every interior vertex closes to round-off, the hull half cells close
with their hull faces, and the vector equals the per-simplex reference
`simplex_area_vectors` to 1e-14 on every mesh tried.

The fix is a method axis, `area_orientation`, default `'primal_edge'`
(the fix) with the legacy rule reachable as `'dual_midpoint'` (status
`broken`), bound into the force partials by `SolverMethods.dudt_fn`.

What depended on the defect (section 3): three pinned tests of lanes L,
R and P and the measured-worse bare-Delaunay droplet. Nothing else: every
2D droplet pin (remap, dual_only, floors, static), the dam break, the
hydrostatic 2D pins, the Hagen-Poiseuille 2D pins, every slow pin and the
full 2D droplet baseline are bit-identical, because no integrated vertex
of those runs ever read a flipped vector (the Hagen-Poiseuille runs read
them at inlet-buffer vertices only, whose motion is prescribed). The
instability of single-phase bare Delaunay + EOS (lanes K, R) does NOT come
from the defect: the kinetic energy doubles at the first step, 32 steps
before the first flipped vector, and the run still blows up with the
correct orientation (KE x 22504 instead of x 24086).

Beyond the lane, measured and NOT fixed: the 3D `batch_e_star` cache
orients each fan triangle on its own and so overstates the area of a
folded fan (section 5); the 2D periodic branch builds different dual
segments from the two ends of a seam edge (section 5).

## 1. What changed and where

| file | change |
|---|---|
| `ddgclib/operators/stress.py` | `AREA_ORIENTATIONS`, `_orient_2d` (the two sign rules); `dual_area_vector(..., orientation='primal_edge')`, applied in the periodic branch (hull and interior edges, with the minimum-image `x_j`) and in the standard branch; `stress_force` / `stress_acceleration` take `area_orientation=` and hand it to `dual_area_vector`. `simplex_area_vectors(v, HC, dim)` (from the first attempt, reviewed: `A_ij = 1/(dim+1) sum_T |T| (grad phi_j - grad phi_i)`, verified in 2D by hand on a right triangle and numerically against the segment on every mesh and against the 3D `p_ij` ring on a jittered box, section 2) is kept as the reference; no force reads it. 1D and 3D ignore the orientation (their sign rules already follow `d_ij`) |
| `ddgclib/operators/multiphase_stress.py` | `multiphase_stress_force` / `multiphase_stress_acceleration` take `area_orientation=`; the `csf_dual` curvature path keeps the default |
| `ddgclib/methods/_axes.py` | new explicit axis `area_orientation` (group forces, both phase models): `primal_edge` (validated, default), `dual_midpoint` (broken, 2D only); evidence of `edge_area_source.shared_vd_2d`, `.min_image_2d` and `viscous_flux.two_point` updated |
| `ddgclib/methods/_config.py` | field `area_orientation`; validated like the flux axes; `dudt_fn` binds it into the single-phase and the multiphase partial only when it is not the default (the default partials are unchanged to the keyword) |
| `ddgclib/tests/test_area_orientation.py` | new (41 tests): sign, antisymmetry, interior closure, hull half-cell closure with the hull faces from the simplex cache, equality with the reference on six meshes (builder rectangle and disk, the laneH mesh, jitter only, strongly sheared, reconnected twice); the legacy counts (6 of 665 on the laneH mesh, closure 0.2316; 4 on the disk builder); the centred force of a linear pressure (1e-12 against 17.6); positive two-point weights; the periodic branch; the axis (registry, 2D only, warns, binding, legacy force); the legacy rule reproduces the three pre-laneO pins; the 3D reference |
| `ddgclib/tests/test_simplex_gradient_flux.py` | the strict xfail is a passing test |
| `ddgclib/tests/test_material_delaunay.py` | the convex-hull arm statement `u_c > 2 u_d` is replaced by the measured pin `PIN_CONVEX_UMAX = 0.22276943647536884` (1.14 `u_d`); the old value 0.6218859935218073 is kept in the comment and locked in `test_area_orientation.py` under the legacy rule |
| `cases_dynamic/diagnose_area_orientation.py` | the probe of the first attempt, reviewed and kept: `static` (meshes), `three_d` (both 3D sources, fan census), `census CASE...` (a run with a scan before every force evaluation; `--orientation` = `preset.replace(area_orientation=...)`), pytest plugin (per-test count of the calls on which the rules differ). Changed: the scan reads the vectors with the run's orientation; docstring says which runs the option reaches |
| `cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py` | `arms2d` honours `--only` and `--tag` like `arms3d` (record `results/laneH/arms2d_laneO.json`) |
| `METHODS.md`, `debugging_plan.md`, `DEVELOPMENT.md` | registry tables regenerated, case matrix rows, status entry, checklist |

Not changed: `hyperct`; the integrators; any case setup or runner. The
multiphase setups (`oscillating_droplet`, `dam_break`, `electrolysis_bubble`,
`shearing_plate_droplet`) build their own `partial(multiphase_dudt_i, ...)`,
so for them `area_orientation` (like `curvature_path`) on the preset is
not applied: they always run the library default (section 7).

## 2. The defect, measured

### 2.1 Static meshes (`diagnose_area_orientation.py static`, library before the fix)

| mesh | directed edges | flipped | not antisymmetric | closure max (interior cell) | against the simplex reference | centred force of a linear pressure, max over interior cells, in `V |g|` |
|---|---|---|---|---|---|---|
| rectangle L 2, refinement 3 (builder) | 800 | 0 | 0 | 0 | 1.5e-16 | 2.6e-15 |
| disk R 1, refinement 3 (builder) | 800 | 4 | 8 | 1.89e-01 | 1.89e-01 | 19.2 |
| sheared 0.3 + jitter 0.2, refinement 3 (the laneH mesh) | 806 | 6 | 12 | 2.32e-01 | 2.32e-01 | 17.6 |
| sheared 0.3 + jitter 0.2, refinement 4 | 3150 | 1 | 2 | 1.25e-01 | 1.25e-01 | 38.5 |
| jitter 0.2 only, refinement 3 | 800 | 0 | 0 | 1.4e-17 | 5.2e-16 | 2.5e-15 |
| sheared 0.6 + jitter 0.3, refinement 3, seed 1 | 806 | 8 | 16 | 3.39e-01 | 3.39e-01 | 66.9 |
| oscillating droplet 2D setup, refinement (2, 2) / (3, 3) | 534 / 1798 | 0 / 0 | 0 | 2e-18 | 1e-17 | 1e-15 |
| dam break 2D setup, refinement 3 | 800 | 0 | 0 | 2e-18 | 5e-17 | 4e-15 |
| hydrostatic column 2D, refinement 3 | 800 | 0 | 0 | 0 | 8e-17 | 3e-15 |
| closed box of lanes K / R, refinement 2 | 208 | 0 | 0 | 2e-17 | 8e-17 | 1e-15 |

The disk builder mesh is new: four of its edges are flipped at setup
(nothing pinned uses `disk`). With the fix every row is 0 flipped, every
interior cell closes below 1e-13 relative, every hull half cell closes
with its two hull faces below 1e-13, and the vector equals the simplex
reference to 1e-14 (`test_area_orientation.py::TestPrimalEdgeRule`, all
six meshes; linear-pressure force 1e-12 on the laneH mesh).

### 2.2 Along the runs (`census all2d`, scan of the integrated vertices before every force evaluation)

"flipped" = calls on which the two rules give different vectors (the same
count whichever rule the library runs). Scalars: before = library before
the fix (= `area_orientation='dual_midpoint'`, reproduced to the bit by
the registered option), after = the fix.

| run (steps) | flipped vectors / evaluations with one / first | scalars before | scalars after |
|---|---|---|---|
| `droplet2d` = `oscillating_droplet_2D` refinement 2/2 (300) | 0 | R_max 0.010283319830164903, KE 6.03498463764473e-07 | identical |
| `droplet2d_dual_only` (300) | 0 | R_max 0.010277554325672004, KE 5.923162338193777e-07 | identical |
| `droplet2d_bare` = `oscillating_droplet_2D_bare_delaunay` (300) | 454 / 109 / step 168 | R_max 0.009445421084245422, KE 0.03582948198940213 | R_max 0.00873836527771301, KE 0.06302934048755672 |
| `dam_break2d` = `dam_break_2D` (400) | 0 | KE 0.0022001911803609547 | identical |
| `electrolysis2d` (100) | 0 | KE 7.516829450276713e-10 | identical |
| `shearing2d` = `shearing_plate_droplet_2D`, periodic branch (10) | 64 against the min-image edge, 144 not antisymmetric, closure 8.4e-3 | KE 7.563487158340381e-05 | 0 against, 27 not antisymmetric (seam, section 5), closure 7.1e-3; KE 7.542555131008922e-05 |
| `hp2d` = `hagen_poiseuille_2D` fast-pin configuration (500) | 252 / 126 / step 125, all at inlet-buffer vertices (x < 0, prescribed motion) | l2 0.013084885355719682, u_max 0.1495783625570828 | identical |
| `hp2d_centred_twopoint` = `.replace(viscous_flux='two_point')` (500) | 354 / 166 / step 125, buffer vertices | l2 0.304160829554134 | identical |
| `hydro2d_remap` (`hydrostatic_2D` + `delaunay_material` remap arm, 300) | 0 | umax_peak 0.20282519399577198, ke_end 2.0944622398202863e-05 | identical |
| `hydro2d_density_diffusion` (300) | 0 | umax_peak 0.19543064599418145, ke_end 0.0008408474475954827 | identical |
| `laneL_hull` (`test_frozen_set.py::TestHagenPoiseuille2D`, hull, 300) | 737 / 222 / step 2 | n_moved 8, max displacement 0.061843693307756, walls released at step 250, 2 on the wall at the end | n_moved 8, 0.02126192366233863, step 249, 2 |
| `laneL_membership` (300) | 431 / 222 / step 2 | n_moved 0, max displacement 0.0, 12 on the wall | identical |
| `laneR_bare` (`test_single_phase_remap.py` BARE, 200) | 272 / 54 / step 33 | ke_end 3253.490893153858, KE max 24085.5 KE0, max u 121.54 U0, KE above 2 KE0 at step 1 | ke_end 9899.545497833766, 22503.9, 84.91, step 1 |
| `laneR_remap` / `laneR_dual_only` (200) | 0 / 0 | ke_end 0.008987112540608227 / 0.011227120983125657 | identical |
| `laneP_material` (`test_material_delaunay.py` material arm, 6 t_ac) | 0 | umax 0.20282519399577198, volume 0.995147730045464 | identical |
| `laneP_convex` (convex-hull arm) | 9 / 5 / evaluation 16 | umax 0.6218859935218073, volume 1.0, p_bottom 4900.947188722312 | umax 0.22276943647536884, 1.0, 4755.980881475467 |
| `laneP_dual_only` | 0 | umax 0.1951860472291084 | identical |

The closure column of the census reports 2.5e-01 for `hydro2d_density_diffusion`
and `laneP_dual_only`: those are the free-surface vertices, which carry
`v.boundary = False` under `dual_only` and have open cells; their vectors
equal the reference to 3e-16, so that residual is the open cell, not an
orientation error.

### 2.3 The test suite (pytest plugin, fast suite before the fix)

2341 calls on which the rules differ, of 1 169 810 2D non-periodic calls
in 110 tests, in 8 tests: `test_frozen_set.py::TestHagenPoiseuille2D`
hull 737 / membership 431; `test_case_hagen_poiseuille.py::TestDeveloping2D`
two-point arm 354 and preset pin 252 (buffer vertices);
`test_determinism.py::TestDensityDiffusionPairOrder` 280 (random point
cloud; its assertion is an equality between builds, unchanged);
`test_single_phase_remap.py::TestBoxDecayWithEOS::test_bare_delaunay_is_unstable`
272; `test_material_delaunay.py::TestColumnThroughTheIntegrator` 9 (the
convex arm); the xfail 6. `v_i` on the hull in 0 of them. (laneH counted
2055 in 6 tests; the two-point arm and the pair-order test are from lanes
H's fix round and T.)

## 3. Pins: what moved, A/B through the registered option

Every number below is `preset` against `preset.replace(area_orientation='dual_midpoint')`
(or the test's `SolverMethods` against its `.replace(...)`). The legacy
value of each is the pre-laneO value to the bit (section 2.2, column
"before"; locked by `test_area_orientation.py::TestLegacyReproducesThePreLaneOPins`).

| pin | configuration | old (`dual_midpoint`) | new (`primal_edge`) | where |
|---|---|---|---|---|
| convex-hull arm peak velocity | `SolverMethods(dim=2, connectivity='delaunay', remap='conservative', redistribute_mass=True)`, refinement 2, 6 t_ac | 0.6218859935218073 (3.19 x the `dual_only` arm; the test asserted `> 2 u_d`) | `PIN_CONVEX_UMAX` 0.22276943647536884 (1.14 x; asserted `> 1.1 u_d` and pinned at rel 1e-6) | `test_material_delaunay.py::test_convex_arm_cannot_hold_the_column` |
| same, bottom pressure | | 4900.947188722312 (0.4996 rho g H) | 4755.980881475467 (0.4848 rho g H; the assertion `< 0.6 rho g H` stands) | same |
| laneL hull-collapse reproducer | `hagen_poiseuille_2D.replace(frozen_set='hull', viscous_flux='two_point', workers=None)` on `setup_poiseuille_2d_lagrangian(L=2)`, dt 0.05, 300 steps | walls released at step 250, largest wall displacement 6.184e-02 | step 249, 2.126e-02 (the test's ranges `200 <= first_drop <= 299`, `n_moved == 8`, `> 1e-3` hold unchanged) | `test_frozen_set.py::TestHagenPoiseuille2D` |
| laneR bare Delaunay + EOS | `SolverMethods(dim=2, connectivity='delaunay')` box, CFL 0.25, 200 steps | KE end 3253.49, max 24085 KE0, max u 121.5 U0 | KE end 9899.55, max 22504 KE0, max u 84.9 U0; the assertions `max u > 10 U0`, `KE max > 100 KE0` hold | `test_single_phase_remap.py::test_bare_delaunay_is_unstable` |

Bit-identical (measured, not assumed): `PIN_KE0` / `PIN_KE_END` of the
single-phase remap (0.43990437705748403 / 0.008987112540608227);
`PIN_UMAX` / `PIN_VOLUME` of the material arm; `PIN_2D_L2` / `PIN_2D_UMAX`
and the slow `PIN_2D_R2_L2` of Hagen-Poiseuille 2D; every hydrostatic 2D
pin (`PIN_2D_UMAX_PEAK`, `PIN_2D_KE_END`, `PIN_2DP_UMAX_PEAK`, slow
`PIN_2D_KE_40`, `PIN_2D_REMAP_KE_40`); the 1D and 3D pins (the axis does
not reach them); the droplet floors, the static droplet, the dual_only
and remap pins, the envelope mirror, the a5b long run, the dam break (fast
and slow suites green on the unchanged pins, section 8).

Full 2D droplet run (`oscillating_droplet_2D.py`, preset
`oscillating_droplet_2D`, refinement 3/3, from a scratch copy of the case
package):

| key | `baselines/baseline_oscillation.json` | this lane |
|---|---|---|
| l2_error_normalized (summary) | 0.17479361640597058 | 0.17479361640597058 |
| tail_growth | 0.9998967874595965 | 0.9998967874595965 |
| linf_error_normalized | 0.32364245955409165 | 0.32364245955409165 |
| mass_drift | 2.4056717879332966e-14 | 2.4056717879332966e-14 |
| n_frames / t_end | 207 / 0.11432433142499636 | the same |

`diff_baselines`: every key equal, delta 0 (the run never reads a flipped
vector: 0 along the refinement 2/2 mirror, section 2.2, and the full run
reproduces the baseline to the bit). Two-fluid reference of the run:
l2 0.18461, linf 0.31117.

The measured-worse bare-Delaunay droplet (`oscillating_droplet_2D_bare_delaunay`,
refinement 3/3, full horizon; registry note l2 0.48992 / tail 1.72505 from
lane 5) is the one droplet configuration that did read flipped vectors
(454 in 300 steps of the refinement 2/2 mirror). A/B with the same scratch
copy of the runner (`retopo_policy_2d = 'delaunay'`) on the library of
HEAD exported with `git archive` (= `dual_midpoint`) and on this lane:

| key | HEAD library (legacy orientation) | this lane |
|---|---|---|
| l2_error_normalized | 0.48991833470391266 | 0.5038096226631333 |
| tail_growth | 1.7250489596305962 | 1.2281376515706395 |
| linf_error_normalized | 0.8503543927096853 | 0.918520751330535 |
| l2 against the two-fluid reference | 0.49871002928721586 | 0.5128602948528049 |
| mass_drift | 1.4e-14 | 2.0e-14 |

The HEAD library reproduces the lane 5 note (0.48992 / 1.72505). With the
correct orientation l2 is 2.8 % worse and the tail growth 29 % smaller;
the value stays measured-worse against the default (0.17479) and its
registry status does not change. The preset's note carries both numbers.

Decision for the droplet: the preset is unchanged; the fix is a
correctness fix adopted regardless, and both l2 and tail of the shipped
configuration are the pinned values to the bit.

## 4. Hagen-Poiseuille 2D: `pressure_flux` reconsidered

`diagnose_poiseuille.py arms2d --tag laneO` (L 4, mu 0.1, refinement 2,
dt 0.01, 3000 steps, window 2 <= x <= 4; record
`cases_dynamic/Hagen_Poiseuile/results/laneH/arms2d_laneO.json`, laneH's
record `arms2d.json` for comparison):

| arm | l2 end (laneH) | l2 end (laneO) | tail mean / max (laneO) | u_max | transverse velocity | wall time |
|---|---|---|---|---|---|---|
| preset (`pressure_flux='centred'`, `viscous_flux='simplex_gradient'`) | 0.011998303695501984 | 0.011998303695501984 (identical in every key) | 0.0104512748544629 / 0.012961826331668107 | 0.15106006829667012 | 2.6e-17 | 128 s |
| `.replace(pressure_flux='simplex_gradient')` | 0.011998073082707693 | 0.011998073082707693 (identical in every key) | 0.010451466133566279 / 0.012961578882656965 | 0.1510600078003142 | 6.3e-18 | 106 s |
| `.replace(viscous_flux='two_point')` | 0.26048858437285916 | 0.2626249742454225 | 0.22747877612887393 / 0.2626249742454225 (laneH 0.2274 / 0.3181) | 0.18893638877994834 (laneH 0.18862) | 3.5e-17 (laneH 0.19) | 103 s |
| `.replace(frozen_set='hull')` | = preset | = preset | | | | 129 s |

Decision: `pressure_flux` of `hagen_poiseuille_2D` stays `'centred'`. The
orientation fix does not move either arm (the flipped vectors of these
runs are at inlet-buffer vertices, whose force is discarded), so the
comparison is the one laneH made: the volume form is better in the end
value by 2.3e-7 (2e-5 relative) and worse in the tail mean by 1.9e-7;
by the flip rule (l2 AND tail at least as good) there is no flip, and
the difference is below anything the case resolves. In 2D the centred
flux on closed, correctly oriented cells IS linearly precise, which was
the reason for the 3D choice. What the fix does move is the two-point
arm: its transverse velocity no longer grows (0.19 -> 3.5e-17 at Re 1 on
this channel) and its tail max falls from 0.318 to 0.263; the arm stays
measured-worse (l2 0.263 against 0.012) for the reason laneH gave (not
linearly precise on sheared rows), now without the orientation defect on
top.

## 5. The 3D paths and the periodic branch (measured, not fixed)

`diagnose_area_orientation.py three_d` (both 3D sources at the vertices
off the hull, the fan census on the `batch_e_star` triangles):

| mesh | source | against `d_ij` | closure, relative, max | against the simplex reference, relative, max | fans with mixed triangle orientation / forced sum above the signed sum |
|---|---|---|---|---|---|
| box refinement 2, builder | cache | 0 | 0 | 0.25 | 0 of 1178 |
| | `p_ij` ring | 0 | 7e-17 | 8e-16 | |
| box, jitter 0.03 | cache | 0 | 6e-17 | 0.25 | 0 of 1178 |
| | ring | 0 | 1.6e-16 | 1.9e-15 | |
| box, jitter 0.08 | cache | 0 | 7.5e-02 | 0.84 | 56 of 1216 / up to 49 % |
| | ring | 0 | 1.9e-02 | 0.50 | |
| box, jitter 0.05 + shear 0.3 | cache | 0 | 0.46 | 1.16 | 149 of 1292 / up to 61 % |
| | ring | 0 | 7.5e-02 | 1.0 | |
| oscillating droplet 3D setup (2, 2), one Delaunay retopology | cache | 0 | 1e-16 | 0.625 | 1 of 5001 / 1e-15 |
| | ring | 0 (2 zero vectors) | 2.6e-02 | 1.0 | |

- No 3D vector points against its edge: `_dual_area_vector_3d_p_ij`
  orients the polygon by `A . d_ij`, and `batch_e_star(orient=True)` and
  `_dual_area_vector_3d_e_star` orient each fan triangle by `d_ij`. The
  3D analogue of the 2D defect is the per-triangle sign: when the fan
  from the edge midpoint through the shared dual vertices folds back (a
  triangle whose normal projects negatively on `d_ij` along a consistent
  walk), forcing every triangle positive adds its area instead of
  subtracting it. On the jittered and sheared boxes 5 to 12 % of the
  fans are such, and the cache deviates from the exact face by up to
  116 % of its norm (the known lane J finding, 25 % even on the builder
  lattice, is the non-planar fan without the face barycentres). The ring
  is exact on the regular and lightly jittered box and loses closure on
  stronger jitter through its nearest-barycentre face rule (lane J, lane
  T). The exact 3D vector is `simplex_area_vectors` (equal to the ring
  where the ring is exact, `test_area_orientation.py::TestSimplexReference3D`);
  making it the source is lane Q (`edge_area_source='p_ij_simplex'`),
  which moves every 3D pin. Not touched here.
- Periodic 2D branch (`shearing2d` census and `TestPeriodicBranch` on a
  sheared, jittered `periodic_rectangle` after `retopologize_periodic`):
  the sign defect was there too (64 vectors against the minimum-image
  edge in 10 steps of the shearing plate; 13 of 886 on the test mesh) and
  is fixed by the same rule. What remains is not a sign: on 62 of the 886
  directed edges with a non-zero vector (61 of them crossing the seam)
  the two endpoints build DIFFERENT dual segments, because the common
  neighbours are min-imaged about `x_i` and the two sides can pick
  different periodic images, and on 2 edges one side returns a zero
  vector. `edge_area_source='min_image_2d'` stays experimental; the
  shearing-plate preset is unstable for other reasons (registry).

## 6. Capillary-rise smoke runner (report only)

`capillary_rise_2D_dynCA.py --smoke` (water, R 0.5 mm, no other option)
from one scratch copy of the runner as it stands in the working tree
(another agent's uncommitted edits included, 326 lines beyond HEAD), run
once on the library of HEAD (exported with `git archive`) and once on this
lane; nothing in `cases_dynamic/capillary_rise*` was edited:

| library | steps | normalised L2 of the height | final height error | mass closure | cap events |
|---|---|---|---|---|---|
| HEAD (= `dual_midpoint`) | 8036 | 0.4860282924783144 | -58.8 % | 9.0e-16 | 0 |
| this lane | 8052 | 0.5070964538354998 | -60.3 % | 3.7e-15 | 1 |

The HEAD number is laneT's to the bit (8036 / 0.4860282924783144), so the
working-tree edits of the runner do not change the smoke run and the whole
move is the orientation fix: the dynCA run reads flipped vectors (its
adaptively remeshed, sheared strip has the skewed configurations), and
with the correct ones the height is 4 % further from the experiment over
the smoke window. The runner builds its force by hand, so there is no
`area_orientation` on record for it; the run with the legacy rule is the
HEAD library. Owner of the interpretation: the dynCA lane (the smoke
number of METHODS.md section 4 is updated to this lane's).

## 7. Known limits

1. The multiphase setups build their own `partial(multiphase_dudt_i, ...)`
   (`cases_dynamic/oscillating_droplet/src/_setup.py:211`, dam break,
   electrolysis, shearing plate), so `area_orientation` on their presets
   is recorded but not applied; they run the library default, like
   `curvature_path`. The `--orientation` option of the census therefore
   reaches the single-phase runs and the lane L / R / P arms, not
   `droplet2d*` (the legacy census of `droplet2d_bare` is the after-fix
   run). The pre-laneO bare-droplet numbers are reproducible only on a
   library before this lane. Follow-up: give the multiphase setups a
   `methods=` argument and build the force with `methods.dudt_fn`.
2. Operators that read `dual_area_vector` outside the force
   (`density_diffusion_step`, `scalar_gradient_integrated`,
   `velocity_difference_tensor`, the `csf_dual` curvature path, the
   diagnostics) always use the default rule. No pin with
   `density_diffusion` had a flipped vector (0 along `hydro2d_density_diffusion`).
3. The periodic branch (section 5) and the 3D cache (section 5) are not
   fixed.
4. The `disk` builder mesh has 4 flipped edges under the legacy rule at
   setup (not used by any pinned case).
5. When both triangles at an edge are exactly flat the sign is undefined
   (`A . d_ij = 0`); the vector is then perpendicular to a degenerate
   edge and is returned as computed.

## 8. Tests

New: `ddgclib/tests/test_area_orientation.py` (41 tests, 12 s):
`TestPrimalEdgeRule` on six meshes (every vector along its edge and
antisymmetric to the bit; interior cells close below 1e-13 relative; hull
half cells close with their two hull faces from the simplex cache;
equality with `simplex_area_vectors` to 1e-14), `TestLegacyRule` (6 of
665 interior directed edges flipped on the laneH mesh with closure 0.2316;
4 on the disk builder; both rules agree on the builder rectangle; unknown
rule raises), `TestForce` (centred force of a linear pressure 1e-12
against > 10 `V |g|` under the legacy rule; positive two-point weights),
`TestPeriodicBranch` (sign rule and the counted seam defects),
`TestAxis` (registry, 2D only, warns as broken, default binds nothing,
legacy bound into both partials, the single-phase partial reproduces the
legacy force), `TestLegacyReproducesThePreLaneOPins` (laneP convex arm
0.6218859935218073 and 4900.947188722312, laneR bare box 3253.490893153858
and 121.54 U0 with the KE doubling at step 1, laneL hull reproducer
6.1843693307756e-02 and step 250), `TestSimplexReference3D` (closed,
antisymmetric, equal to the `p_ij` ring on a lightly jittered box).
Changed: `test_simplex_gradient_flux.py` (the strict xfail passes),
`test_material_delaunay.py` (`PIN_CONVEX_UMAX`).

Battery (ddg environment, repo root): fast **1182 passed, 12 skipped, 3
xfailed, 0 failed** in 126 s (before the lane: 1141 passed, 12 skipped,
4 xfailed; +41 new tests, the laneH xfail now passes); slow **32 passed,
1 xfailed** in 311 s (unchanged counts; every slow pin bit-identical);
hyperct not touched (316 at the last run of lane T, 326 with its new
tests). The fast suite with the counting plugin
(`-p cases_dynamic.diagnose_area_orientation`) before the fix: 2341
flipped calls in 8 tests (section 2.3).

## 9. DO-NOTs (measured)

- Do not orient a dual face by where the vertex lies relative to the
  face's midpoint: on a Delaunay mesh with shear 0.3 and jitter 0.2 that
  flips 6 of 806 vectors and the centred force of a LINEAR pressure is
  off by 17.6 `V |g|` at the worst cell (66.9 at shear 0.6).
- Do not read `test_convex_arm_cannot_hold_the_column` as "the convex
  hull fill triples the velocity": 2.6 of the 3.19 were flipped vectors;
  the convex arm is 1.14 x the fixed-connectivity arm, and what it cannot
  do is compress (volume 1.0, head 0.485 rho g H).
- Do not attribute the single-phase bare-Delaunay + EOS blow-up to the
  orientation: KE doubles at step 1, the first flipped vector is at step
  33, and the fixed run blows up to 22504 KE0.
- Do not expect `area_orientation` on a multiphase preset to reach the
  droplet, dam-break, electrolysis or shearing-plate force (section 7).
- Do not sum per-triangle forced signs in 3D and call it the area of the
  fan: 5 to 12 % of the fans of a jittered box fold back (section 5).

## 10. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
D=cases_dynamic/diagnose_area_orientation.py

$PY $D static                                  # 11 meshes, both rules
$PY $D three_d                                 # 3D sources, fan census
$PY $D census all2d --out /tmp/census.json     # every 2D run, fixed library
$PY $D census laneL_hull laneR_bare laneP_convex --orientation dual_midpoint   # the legacy arms
$PY -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider -p cases_dynamic.diagnose_area_orientation   # per-test flipped counts

$PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms2d --tag laneO   # section 4
$PY -m pytest ddgclib/tests/test_area_orientation.py ddgclib/tests/test_simplex_gradient_flux.py -q -p no:cacheprovider
```

Full droplet and dynCA runs: scratch copies of the case packages
(`cases_dynamic/__init__.py`, the case directory's `.py` files, `src/`,
`baselines/`; a `data` link next to `cases_dynamic/` for dynCA), run with
`PYTHONPATH` = repository root, as in the laneT log, section 12.
