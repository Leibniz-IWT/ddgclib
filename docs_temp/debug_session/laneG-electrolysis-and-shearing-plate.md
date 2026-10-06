# laneG: the 3D electrolysis bubble keeps its gas phase; the shearing-plate droplet survives its setup and its shear

Date: 2026-10-06 (cloud session, branch `claude/relaxed-davinci-tkh95t`).
Brief: reproduce the loss of the gas phase of `electrolysis_bubble_3D`
through the preset and fix it in the library behind the registry, with
the static bubble's Laplace pressure against the analytical value at two
refinement levels and the injected gas mass conserved to round-off; move
the shearing-plate rescale into a builder with `move_all`, make the
periodic path forward the method axes like the non-periodic path (or
refuse clearly), fix the seam asymmetry of lane O if it is on the path,
check a quiescent droplet (Laplace jump, no spurious seam force) and a
weakly sheared one (volume and interface kept over the short window);
pins for both cases; `setup_shearing_plate_droplet` re-entrant.

Every arm is `PRESETS[name]` or `preset.replace(...)`; every number
below comes from one process on this machine (every run is deterministic
here, lane T), through the two diagnose scripts of section 9.  Machine:
Python 3.13.16, numpy 2.5.3, scipy 1.18.1.

## 0. Verdict

- **Electrolysis 3D.** The loss of the gas phase does NOT reproduce on
  the current library: the shipped 3D horizon (2292 steps, refinement
  1/1, gravity, injection) keeps its 35 gas cells and 26 interface
  vertices, gas mass 4.7437e-8 -> 5.0438e-8 kg (= M0 + dm_dt t to the
  printed digits).  What is left is the defect the preset itself carries
  (`connectivity='delaunay'`, no remap, the configuration measured worse
  on the droplet): at every Delaunay flip the measured gas volume jumps
  by 1.5 % (4.7355e-9 <-> 4.8087e-9 m^3 at fixed positions) and the
  per-phase redistribution turns it into a uniform gas pressure jolt of
  K (V_old / V_new - 1) = +-1500 Pa; over the horizon the gas pressure
  swings between -4026 and +1104 Pa around a Laplace value of 144.
  A STATIC bubble (g = 0, no injection) measures it cleanly: see the
  table of section 1.  Fix (protocol rule 5, a preset change): both
  electrolysis presets run on `connectivity='dual_only'` (the 3D droplet
  default); `remap='conservative'` is as clean on the static bubble but
  a measured DO-NOT with the injection (the bubble does not grow,
  section 1.3).
- **Shearing plate.** Four defects fixed, all in the library or the
  setup: (1) the uniform anisotropic rescale deleted both droplet poles
  (`rescale_droplet_box`, one `move_all`, identity on the band
  `|x_a| <= shell_R`); (2) the periodic rebuild measured the seam
  simplices unwrapped (total dual volume 1.94 x the box) and kept
  overlapping images of seam simplices (8 facets with 3 owners); (3) the
  setup's one-time periodic pass left the outer phase at -100 Pa (jump
  106 Pa against gamma / R = 6); (4) without the conservative remap the
  repaired case still blows up at t = 0.042 s in the row next to the
  plates.  The remap, `frozen_set`, `edge_area_source` and
  `projection_every` now reach the periodic path through the ledger
  closure shared with `_retopologize_multiphase`
  (`multiphase_rebuild_with_ledger`).  With `remap='conservative'` on
  the presets the short window (1649 steps to t = 0.05 s) completes
  with its 32 interface vertices, jump 6.018 Pa, droplet volume ratio
  0.983992 (0.984008 at setup); the quiescent droplet holds the jump
  with the sum of forces on the free vertices at round-off.  The 3D
  setup builds and runs its first steps (known limit: the 3D seam faces
  are unwrapped).
- The lane O seam asymmetry is not on the path (0 of 1792 directed
  edges on the shearing mesh); its 62 pairs are triangles of the test's
  jittered mesh that span more than half the period.

## 1. Electrolysis 3D

### 1.1 Reproduction through the preset

`PRESETS['electrolysis_bubble_3D']`, `setup_electrolysis_bubble(dim=3,
refinement 1/1)`, the runner's dt 6.5457e-8, injection `dm_dt_3d` in the
callback (the runner's loop, reproduced by
`diagnose_static_bubble.py --inject --g 9.81 --steps 2300`).

| step / t | gas mass | gas volume | R_eq | gas pressure (min, max over the gas cells) | KE | u_max |
|---|---|---|---|---|---|---|
| 0 | 4.743704e-8 | 4.7355e-9 | 1.0417 mm | 173.4, 173.6 | 0 | 0 |
| 101 / 6.6e-6 | 4.756926e-8 | 4.7354e-9 | 1.0417 | 454.5, 454.7 | 5.7e-11 | 1.4e-2 |
| 126 / 8.2e-6 | 4.760199e-8 | 4.8087e-9 | 1.0471 | -1007.9, -1007.7 | 9.6e-11 | 1.8e-2 |
| 151 / 9.9e-6 | 4.763472e-8 | 4.7353e-9 | 1.0417 | 595.7, 595.9 | 1.4e-10 | 2.2e-2 |
| 1326 / 8.7e-5 | 4.917297e-8 | 5.1236e-9 | 1.0695 | -4026.2, -4026.0 | 9.3e-9 | 0.19 |
| 2300 / 1.5e-4 | 5.044808e-8 | 5.0088e-9 | 1.0614 | 719.2, 719.4 | 3.5e-7 | 0.79 |

35 gas cells, 26 interface vertices, 9 bulk gas vertices, 95 vertices,
26 walls at every record; no stranded (mass without volume) pair.  The
gas mass at the end equals 4.743704e-8 + 2e-5 x 2300 x 6.5457e-8 =
5.044808e-8 (round-off; section 1.3).  So the gas phase is kept; lane
F's `phase_ledger='volume'` (the 26 stranded pairs of the setup state
are released at the first rebuild) and lane B's full outer mesh are the
library state this was measured on.  The audit's "R_eq -> 0 from
1.1e-4 s" was measured on the pre-laneB mesh with the snapshot ledger
(the arms `--box-shift evict --replace phase_ledger=snapshot` reproduce
that state if the owner wants to re-run it; not re-measured here, the
horizon costs 15 min per arm and the defect that remains is visible in
the table above: the pressure of a 1.5 % volume flip, +-1500 Pa, on a
144 Pa jump).

### 1.2 Static bubble against the analytical Laplace pressure

`diagnose_static_bubble.py` (g = 0, no injection, the preloaded state
IS the analytical solution: p_gas - p_liq = 2 gamma / R0 = 144 Pa, V =
4/3 pi R0^3 = 4.18879e-9 m^3).  The jump is the volume-weighted mean
gas pressure over the bulk gas cells minus the same for the liquid
(`ddgclib.analytical.integrated_phase_pressure_jump`).

| refinement (outer / droplet), vertices | arm | steps / t | jump at the end [Pa] (exact 144) | V_gas / V_exact (setup -> end) | KE_max [J] | u_max [m/s] | gas mass drift |
|---|---|---|---|---|---|---|---|
| 1/1, 95 | preset (delaunay, no remap) | 2000 / 1.31e-4 | -4872.3 (swinging -7675 .. +1893 along the run) | 1.1305 -> 1.1887 | 7.8e-7 | 1.13 | 5.6e-15 |
| 1/1 | `.replace(remap='conservative')` | 2000 | 241.9 (145.5 at 200, 157.6 at 600, 192.3 at 1200) | 1.1305 -> 1.2231 (1.2242 after the first rebuild) | 4.6e-9 | 0.113 | -8.1e-15 |
| 1/1 | `.replace(connectivity='dual_only')` | 2000 | 224.1 (147.4 at 300, 166.1 at 800) | 1.1305 -> 1.1296 | 4.2e-9 | 0.115 | 2.8e-15 |
| 1/1 | Delaunay (pre-lane preset) | 688 / 4.50e-5 | 278.4 | 1.1305 -> 1.1290 | 5.5e-9 | 0.123 | 1.1e-15 |
| 1/1 | `dual_only` (the preset now) | 688 | 160.7 | 1.1305 -> 1.1303 | 5.8e-10 | 0.049 | -1.5e-15 |
| 1/1 | remap | 688 | 161.7 | 1.1305 -> 1.2242 -> 1.2241 | 6.1e-10 | 0.049 | -7.0e-15 |
| 2/2, 475 | Delaunay (pre-lane preset) | 2000 / 4.50e-5 | 1511.6 (-1450 at 1000, +1440 at 1800) | 1.0605 -> 1.0466 | 6.7e-9 | 0.234 | -9.5e-15 |
| 2/2 | remap | 2000 | 167.5 (145.2 at 400, 151.3 at 1000, 160.7 at 1600) | 1.0605 -> 1.0782 (after the first rebuild) -> 1.0781 | 2.9e-10 | 0.041 | -3.3e-14 |
| 2/2 | `dual_only` (the preset now) | 2000 | 165.4 (144.3 at 200, 150.8 at 1000, 159.3 at 1600) | 1.0605 -> 1.0604 | 2.6e-10 | 0.042 | 1.1e-14 |

At equal physical time (4.5e-5 s, 0.06 of the liquid acoustic time of
the box) the preset's jump is 160.7 Pa at refinement 1/1 and 165.4 Pa at
2/2 against 144: the error is +11.6 % / +14.9 %.  Both kept arms drift
monotonically upward (the polyhedral bubble of 26 / 98 interface
vertices relaxes toward its discrete equilibrium jump; the volume of the
gas cells is 13 % / 6 % above the sphere) and the pre-lane Delaunay
preset swings by +-1500 Pa at every flip on top of it.  The 2D case
(`--dim 2`, jump gamma / R0 = 72 Pa, 1500 steps = 4.7e-5 s): refinement
2/3 (214 vertices): Delaunay 8065.6 Pa (KE_max 4.7e-3 J, |u| 2.47 m/s),
`dual_only` 72.83 (3.5e-9 J, 0.013 m/s), remap 72.82 (3.4e-9); refinement
1/2 (89 vertices): Delaunay 75.84, `dual_only` 75.84 (bit-identical: no
flip in the window), remap 76.02.  2D shipped horizon (6330 steps, gravity, injection, refinement 2/3): Delaunay: KE_max 4.3e-2 J, |u|max 3.24 m/s, the jump swinging between +17221 and -1302 Pa, gas volume 1.148 V_exact at the end; dual_only: KE_max 2.5e-4 J, |u|max 1.03 m/s, the jump 1964 / -783 / 1050 Pa at t = 4e-5 / 1.2e-4 / 2e-4 s (the bubble breathing under the injected mass, +12.2 % of volume at the end), gas mass at M0 + dm_dt t to 2.9e-14, liquid 2.9e-14; the 32 interface vertices kept in both.  Both electrolysis presets are
`connectivity='dual_only'` now (rule 5; the Delaunay value stays
registered, status unchanged, reachable by
`.replace(connectivity='delaunay')`).

### 1.3 Mass ledger under injection

The runner's loop (gravity, wall clamps, `dm_dt_3d` injected in the
callback), refinement 1/1, 2300 steps (1.5e-4 s, the shipped horizon;
`diagnose_static_bubble.py --inject --g 9.81 --steps 2300`):

| arm | gas mass at the end against M0 + dm_dt t | liquid mass drift | gas volume (setup 4.7355e-9 m^3) | jump at the end [Pa] | KE_max [J] | u_max |
|---|---|---|---|---|---|---|
| Delaunay, no remap (the preset until this lane) | 5.044808e-8 (to the printed digits) | | 5.0088e-9 (+5.8 %) | gas pressure 719 (swinging -4026 .. +1104 along the run) | 3.5e-7 | 0.79 |
| `dual_only` (the preset now) | -2.6e-16 relative | -1.9e-15 | 5.0098e-9 (+5.8 %) | 535.1 | 6.5e-7 | 0.68 |
| `.replace(connectivity='delaunay', remap='conservative')` | +1.7e-14 | -1.1e-14 | 5.2792e-9 raw at the end, 1.14 to 1.29 x V_exact at the records (the rebuild moves the raw measure; the gauge absorbs it) | 6492.8 = K dm / m with no relief | 1.1e-6 | 0.63 |

Every arm conserves the ledger to round-off (the injected mass is spread
over the gas sub-volumes and the per-phase redistribution conserves per
phase).  The remap arm's jump follows K dm / m exactly (6493 Pa for
dm / m = 6.3 %): the projection of every call erases the liquid's
compression response (lane H), so the liquid never yields and the
bubble does not grow; the `dual_only` arm relieves the injected mass to
535 Pa by growing 5.8 % within 0.19 of a liquid acoustic time of the
box.  The remap is therefore a measured DO-NOT for this case with its
injection (section 7).

### 1.4 The mass source and the remap

`add_phase_mass` (new, `ddgclib/operators/mass_source.py`; the case's
`inject_gas_mass` is a thin wrapper) spreads dm over the sub-volumes of
the phase as before AND scales the level anchor's reference density of
that phase (`mps._remap_rho_ref[k] *= (M + dm) / M`) when the remap is
running.  Measured DO-NOT without it: on the injected 3D run with
`remap='conservative'` the gas pressure the EOS reads climbed to 4807 Pa
at t = 1.1e-4 s while the gas volume stayed at 5.1236e-9 m^3 at every
record (the anchor pins the phase level to the volume strain, so the
force never saw the injected mass); the run without the remap grew the
bubble by +5.8 % of volume over the same horizon.

## 2. Shearing plate

### 2.1 The rescale (`ddgclib.geometry.domains.rescale_droplet_box`)

The builders make a cube of half-side L_build = max(L_x, L_y, L_z); the
channel wants (L_x, L_y[, L_z]) = (0.015, 0.01[, 0.01]).  The uniform
scale `x_a * L_a / L_build` of the setup until this lane mapped the outer
vertices at (0, +-0.0075) onto the droplet poles (0, +-0.005) (2D,
refinement 3/3), and its `on_collision='evict'` loop deleted both
interface poles (lane B); at refinement 2/2 it put 8 of 52 outer vertices
inside the shell ring (R0 + h = 0.0053), one at r = 0.00468 inside the
droplet; in 3D (1/2 and 2/2) 11 outer vertices landed inside the shell
and 4 collided with droplet keys (the setup crash of the audit).  The
new map is piecewise linear per rescaled axis: identity for
`|x_a| <= shell_R` (the builder records `metadata['shell_R']` now),
`[shell_R, L_build]` onto `[shell_R, L_a]`; strictly monotone, so no
collision is possible and no moved vertex lands inside the shell; one
`HC.V.move_all`.  2D refinement 3/3: 88 of 304 vertices moved, 32
interface, 16 plate vertices (8 + 8), setup digest `fe107bac36e7b9da`
(the pre-lane setup: 302 vertices, 22 plate vertices before the periodic
merge).  3D refinement 1/2: 36 of 306 moved, 98 interface, 8 plates.  A
second call in one process gives the same digest: the setup is
re-entrant (the audit's crash was the rescale collision, not the
function).

### 2.2 The periodic rebuild (`ddgclib/geometry/periodic.py`, hyperct `simplex_dual_volumes(periods=)`)

Measured on the 2D shearing mesh after setup (`probe` in the log of
this lane; the test `TestSetup2D::test_periodic_duals_tile_the_box`
guards it): total dual volume 1.1641e-3 against the box 6.0e-4 (ratio
1.940), 21 seam simplices measured unwrapped (sum of the raw simplex
measures 1.1641e-3, with minimum-image coordinates 6.1406e-4 = box +
2.3 %), facet owner counts {2: 874, 1: 16, 3: 8}: eight facets with three
owners, i.e. overlapping images of seam simplices.  Two changes:

1. `delaunay_with_ghosts` keeps only the simplices whose centroid (in
   the padded coordinates) lies in the fundamental domain on every
   periodic axis: one image per periodic simplex.  Cocircular seam
   squares of a structured mesh are triangulated by qhull with a
   diagonal that may differ between the real square and its ghost
   image (unit square, refinement 3, with the filter alone: 261
   simplices, total 1.0195, 10 facets with 3 owners), so every real
   vertex and all its ghosts get the same deterministic offset (1e-7 of
   the shortest period, along the periodic axes only so that a wall row
   stays collinear; `numpy.random.default_rng(0)` over the vertex
   order) and the perturbed coordinates decide the connectivity only.
2. `cache_dual_volumes` passes the periods of `HC._periodic_axes` /
   `_periodic_bounds` to `hyperct.ddg.simplex_dual_volumes(periods=)`,
   which brings every simplex to the minimum image of its first vertex
   before the determinant.

After: unit square refinement 3 / 4 and the periodic box refinement 2
after one rebuild: 256 / 1024 / 768 simplices, total dual volume 1.0 to
round-off, facets {2, 1 (walls only)}, 0 interior vertices tagged
boundary (lane P: 2.488, 287 simplices, 6 tagged); shearing mesh: 592
simplices, 18 seam simplices, total 6.0e-4 = the box, {2: 880, 1: 16}.

### 2.3 The setup pressure

After the one-time periodic pass the setup kept the masses the
redistribution had conserved from the pre-periodic mesh on duals whose
total differs (the ub-face vertices are merged away), so the outer phase
sat at a uniform -100.3 Pa and the measured jump was 106.3 Pa against
gamma / R = 6.  The setup now resets the masses on the periodic duals
(`mps.compute_phase_masses`, not a `refresh(reset_mass=True)`, which
would relabel the seam simplices by the builder's radial criterion on
their unwrapped centroids and break the interface closure) and then
preloads the droplet: jump 6.000000 Pa at setup.  The one-time pass runs
with `remap=None`: the remap's level anchor fixes its reference at its
first call, and with the pass under the remap the first integrator step
anchored the droplet level back to 0 Pa (jump 6 -> 0.000 at step 1,
measured).

### 2.4 The periodic path forwards the axes

`retopologize_periodic(frozen_set=, edge_area_source=,
skip_triangulation=)`, `retopologize_multiphase_periodic(frozen_set=,
boundary_filter=, merge_cdist=, edge_area_source=, retopo_remap=,
projection_every=)`; `SolverMethods` accepts `frozen_set='membership'`,
`edge_area_source` (not `'e_star_cache'`: no fan cache) and
`remap='conservative'` / `projection_every > 1` with
`connectivity='periodic'` (multiphase); the single-phase periodic
rebuild has no remap (refused with that message).  The ledger closure of
`_retopologize_multiphase` (snapshot, remap stage 1, rebuild, refresh,
redistribution, gauge, restore, anchor) moved verbatim to
`ddgclib/methods/_retopo.py:multiphase_rebuild_with_ledger`, which both
functions call with their own rebuild closures: every droplet and dam
break pin is unchanged (fast suite), the lane W bit-identity probe of
the pre-laneW closure now runs on `preset.replace(remap=None)`.

### 2.5 The short window and the quiescent droplet

`diagnose_short_window.py` (refinement 3/3, dt 3.033e-5, 1649 steps to
t = 0.05 s; `--U 0` for the quiescent droplet).

| arm | end state | first-row speed / U_wall | jump [Pa] (exact 6) | V / V_exact (setup 0.984008) | interface vertices | KE_max [J] | digest |
|---|---|---|---|---|---|---|---|
| before this lane (lane B / W record) | blows up | 295 at t = 0.05 | | | first loss at step 183 | 0.776 | `0ca6deaa1a7f766a` (3 steps) |
| rescale + rebuild + setup fixes, `remap=None` | goes unstable at step 1394 (t = 0.0423 s) in the row next to the plates near the seam (vertex (-0.01366, -0.00921), dual area 1.54e-6 against 2.55e-6 of its row) | 0.52 at 0.040 s, 1.42 at 0.042, 2.43 at 0.05 | 12.44 | 0.975171 | 32 | 1.81e-4 | `2f621bf47fe85c18` |
| `remap='conservative'` (the presets now) | completes | 0.59 at 0.05 s (monotone) | 6.018 | 0.983992 | 32 | 6.33e-5 | `3c66efcb929b9e1c` |
| quiescent, `U_wall = 0`, remap | completes | largest speed 5.0e-3 m/s (the interface relaxing) | 6.018 (5.953 to 6.118 along the run) | 0.983991 (0.98390 to 0.98405) | 32 | 5.39e-8 | `d12ffe97bd621b81` |

Forces (the force the integrator reads, m a per cell): the sum over
EVERY cell, plates included, is 1.9e-18 N at setup and stays at round-off
along the quiescent run (the pairwise fluxes across the seam are
antisymmetric: no spurious seam force); the sum over the free vertices
alone is 8.7e-19 N at setup against the largest single |m a| of 1.43e-3
N and at most 4.2e-11 N along the run (9e-9 of the largest cell force,
4.5e-3 N): that residual is the plates' reaction on the fluid, not the
seam.  `TestQuiescentDroplet2D` asserts both bounds (200 steps).

### 2.6 3D

`diagnose_short_window.py --dim 3 --ro 1 --rd 2 --steps 5` (the short
runner's refinement): setup 306 vertices, 98 interface, 8 plate
vertices, jump 12.000000 Pa (2 gamma / R0 = 12), V / V_exact 1.042902,
sum of the forces over every cell 1.9e-18 N; five steps (dt 4.5045e-5)
complete in 21 s: jump 11.9993, V / V_exact 1.061937 (the raw measured
volume moves with the 3D rebuild; the remap gauge absorbs it), 98
interface vertices kept, KE_max 3.73e-6 J, digest `303c19a7f707bb2a`
(`TestSetupAndSteps3D`, slow).  The 3D faces of the seam edges are
unwrapped (section 4), so this is a smoke pin: the main 3D runner
(refinement 2/2, t = 0.8 s) was not run.

## 3. The seam asymmetry of lane O (not on the path)

The 62 of 886 directed edges of `TestPeriodicBranch` are re-attributed:
that test jitters a corner vertex to (0.0064, -0.0074), below the wall
row y = 0, and qhull closes the hull with slivers from it to wall
vertices up to 0.5 away, i.e. triangles longer than half the period,
which no minimum image can represent (a vertex seen from one end is
imaged to the other side).  Reading the apexes from the simplex cache
instead of `v_i.nn & v_j.nn` changes nothing there (64 = 64 pairs on the
current rebuild) and the shearing mesh has 0 asymmetric pairs of 1792
directed edges at setup, so `dual_area_vector` was left as it is.

## 4. Known limits

- 3D periodic: the dual faces of seam edges are built from unwrapped
  coordinates (the minimum-image rebuild of `dual_area_vector` is 2D
  only); the volumes are exact now, the faces are not.  The 3D shearing
  case is therefore a setup + smoke pin.
- The electrolysis bubble at refinement 1/1 is 26 interface vertices; the
  discrete bubble is 13 % (22 % after the first rebuild) larger than the
  sphere and its jump drifts from 144 Pa by the numbers of section 1.2
  within 0.16 of a liquid acoustic time of the box; the comparison at
  two refinement levels is the measurement of that error, not a
  validation of the case.
- `rescale_droplet_box` compresses the rows between `shell_R` and the
  plates by the ratio (L_a - shell_R) / (L_build - shell_R) (2D: 0.44).

## 5. Pins

`ddgclib/tests/test_case_electrolysis_bubble.py` (fast: 3D refinement
1/1 static 200 steps and injection 100 steps, 2D refinement 1/2 the
same; slow: 3D refinement 2/2 static 2000 steps): per-phase mass drift
at round-off (< 1e-13 relative), the gas mass against M0 + dm_dt t,
interface vertex count, gas volume ratio, the jump and the final-state
digest; `ddgclib/tests/test_case_shearing_plate.py` (fast: the 2D setup
invariants, the periodic duals tiling the box, the preload, the
re-entrant setup, the quiescent droplet; slow: the 2D short window
(volume, interface count, D, digest) and the 3D setup plus five steps).
Values in the two files; nothing pinned before this lane for either
case moved except the shearing-plate digests, which the lane's setup
and rebuild changes replace (the lane W probe of the pre-laneW closure
now runs on `.replace(remap=None)` and still agrees to the bit).

## 6. Battery

On this machine (Python 3.13.16, numpy 2.5.3, scipy 1.18.1): fast 1250 passed / 1 failed / 11 skipped / 2 xfailed (the failure is the environment baseline (a), test_single_phase_remap.py one-ulp KE equality, unchanged; +10 tests of this lane), slow 36 passed / 1 failed / 1 xfailed (the failure is baseline (b), test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column, peak 0.15269571220608844 against the pin 0.15305813130485327 as before the lane; +3 slow tests), hyperct 340 passed / 38 skipped / 6 xfailed (baseline (c), the matplotlib 3.11 contour colour, fixed in hyperct/_plotting.py).  Green by the session's definition: no failure beyond the three baseline ones, (a) and (b) unchanged, (c) fixed.

## 7. DO-NOTs (measured)

- A mass source under `remap='conservative'` without scaling the anchor
  reference (section 1.4).
- Running the remap before a setup's pressure preload (section 2.3).
- `refresh(reset_mass=True)` after a periodic rebuild (section 2.3).
- The uniform anisotropic rescale (section 2.1).

## 8. Files

Library: `ddgclib/geometry/periodic.py` (centroid filter + consistent
tie-break in `delaunay_with_ghosts`, `retopologize_periodic(frozen_set=,
edge_area_source=, skip_triangulation=)`), `ddgclib/methods/_retopo.py`
(`multiphase_rebuild_with_ledger`, the periodic function's new
keywords), `ddgclib/dynamic_integrators/_integrators_dynamic.py`
(`_retopologize_multiphase` calls the shared closure; the periodic
refusals of `frozen_set` / `edge_area_source` lifted),
`ddgclib/methods/_config.py` (`_HULL_REBUILDING`, `_RECONNECTING`,
`_EDGE_AREA_CONNECTIVITIES` include `periodic`; the periodic partial
binds `retopo_remap`, `projection_every`, `frozen_set`),
`ddgclib/methods/_presets.py` (shearing presets `remap='conservative'`,
`electrolysis_bubble_3D` `connectivity='dual_only'`),
`ddgclib/methods/_axes.py` (evidence), `ddgclib/operators/stress.py`
(`cache_dual_volumes` passes the periods),
`ddgclib/operators/mass_source.py` (new, `add_phase_mass`),
`ddgclib/analytical/_integrated_comparison.py`
(`integrated_phase_pressure_jump`),
`ddgclib/geometry/domains/_multiphase_droplet.py` (`rescale_droplet_box`,
`metadata['shell_R']`).  hyperct: `hyperct/ddg/_dual_volume.py`
(`simplex_dual_volumes(periods=)`), `hyperct/_plotting.py` (the
matplotlib 3.11 `colors=` fix of baseline failure (c)).  Cases:
`cases_dynamic/shearing_plate_droplet/src/_setup.py` (rescale, mass
reset, the one-time pass without the remap),
`cases_dynamic/electrolysis_bubble/src/_reaction.py` (wraps
`add_phase_mass`), `src/_setup.py` (docstring), the two diagnose
scripts.  Tests: `test_case_electrolysis_bubble.py`,
`test_case_shearing_plate.py` (new), `test_frozen_set.py`,
`test_edge_area_source.py`, `test_methods.py` (the periodic refusals
turned into positive tests).  The dynCA runners of `capillary_rise` are
not on the periodic path and bind no remap: nothing of this lane moves
their numbers.

## 9. Reproduce

From the repository root, `PYTHONPATH=. python ...`:

- `cases_dynamic/electrolysis_bubble/diagnose_static_bubble.py --dim 3
  --ro 1 --rd 1 --steps 2000` (static bubble, the preset), `--replace
  connectivity=delaunay` (the pre-lane preset), `--replace
  connectivity=delaunay --replace remap=conservative`; `--ro 2 --rd 2`
  for refinement 2/2 (12 min per arm); `--inject --g 9.81 --steps
  2300` for the shipped horizon (4 to 7 min per arm); `--dim 2 --ro 2
  --rd 3 --steps 1500` for the 2D rows.
- `cases_dynamic/shearing_plate_droplet/diagnose_short_window.py` (the
  short window, 4 min), `--replace remap=None` (the unstable arm),
  `--U 0` (quiescent), `--dim 3 --ro 1 --rd 2 --steps 5`.
- The periodic mesh numbers of section 2.2:
  `test_case_shearing_plate.py::TestSetup2D::test_periodic_duals_tile_the_box`
  and, for the unit square, `test_periodic.py`.
- Suites: rule 7 of the brief.
