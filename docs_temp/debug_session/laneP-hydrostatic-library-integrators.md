# laneP: Hydrostatic_column on the library integrators

Date: 2026-10-01. Ports the four hand-rolled Hydrostatic_column runners onto
`SolverMethods` presets, measures both connectivity arms for each, and adds
the one library method the reconnecting arm needed.

Revised the same day after independent review (fix round 1): the peel of
`delaunay_material` removed fluid in 3D and was replaced by a geometric
test (section 4b); the statements and numbers that depended on it are
corrected in place. Sections 3, 4, 9, 10 and 11 describe the final state.

## 0. Verdict

- All four runners (`Hydrostatic_1D.py`, `_2D.py`, `_3D.py`,
  `_2D_periodic.py`) run through `ddgclib.methods.PRESETS` on the library
  `symplectic_euler`, with gravity as the `body_force` of
  `SolverMethods.dudt_fn`. No time loop is left in the case. They write
  `methods.json`, `StateHistory` snapshots, figures and an mp4, headless.
- The presets use FIXED connectivity (`delaunay` chain in 1D, `dual_only` in
  2D, `dual_only_bare` in 3D). They settle over 40 to 200 acoustic times and
  the integrated pressure error converges with refinement (section 5).
- The reconnecting arm of the brief, `delaunay` + `remap='conservative'`, is
  UNSTABLE on a free surface (42 m/s at 3 t_ac). The suspect named in the
  lane R log, the uniform offset `K (s - 1)` of the mass rescale, is not the
  cause: without the rescale it is worse. The cause is that Delaunay
  triangulates the convex hull of the cloud, so the gap between a moved free
  surface and the hull is filled with simplices that are not fluid. New
  library method `connectivity='delaunay_material'` (the boundary of the
  previous connectivity is material, the hull fill is peeled by a geometric
  test): with the remap it is stable in 2D and 3D and tracks the
  fixed-connectivity arm (section 3, 4). It keeps the domain volume to
  round-off in 2D and to 1.2e-06 per call in 3D (free-surface diagonal
  slivers, section 10).
- Review finding, fixed: the first peel was topological (remove a simplex
  that exposes a facet that is not an old boundary facet). In 3D the
  diagonals of planar wall squares change between rebuilds, so it took
  fluid tetrahedra with four wall vertices for hull fill: a lattice cube
  fell to 0.790 of its volume. The geometric peel keeps it at 1 to 1e-15,
  and every 1D and 2D number of this log is bit-identical (section 4b).
- `Hydrostatic_2D_periodic` is NOT periodic. The library periodic path is
  unusable for a single-phase EOS column (dual volume 2.488 for a unit
  domain, 45 of 136 vertices fail dual-face closure, section 8). The preset
  runs free-slip side walls (new `FreeSlipWallBC`), which is what the runner
  did before with case-local code.
- Three findings that reach beyond this case (sections 5 to 7): (a) the
  discrete hydrostatic equilibrium of an EOS column under gravity is a
  saddle of the discrete energy, slow modes grow at 1.1 to 1.2 g/c0, with
  or without a free surface. (b) The free-surface force is not an energy
  gradient, and without viscosity that gives a flutter instability on some
  meshes (no-slip column: 0.75 1/s at refinement 2, none at 3, 1.22 1/s at
  4; never with a closed lid). The artificial viscosity suppresses the
  flutter and slows the saddle to a creep; it is what keeps the column
  settled. (c) The centred pressure flux is exact for NODAL pressures, so
  boundary cells settle on the nodal value, not on the cell average.

## 1. What changed and where

ddgclib (working tree, uncommitted). hyperct was not touched.

| file | change |
|---|---|
| `ddgclib/methods/_retopo.py` | new `retopologize_material_delaunay` (returns the relative domain-volume change, warns above `domain_tol`), `peel_outside_boundary`, `oriented_boundary`, `winding_number`, `enclosed_volume`; `bare_dual_refresh(..., boundary_filter=None)` (default unchanged) |
| `ddgclib/methods/_axes.py` | `connectivity` option `delaunay_material` (status `opt-in`); evidence on `single`, `delaunay`, `dual_only`, `dual_only_bare`, `periodic`, `remap=conservative`, `density_diffusion`, `boundary_dual_vol` |
| `ddgclib/methods/_config.py` | builder + validation for `delaunay_material` (single phase, no `merge_cdist`, no `backend`, redistribution only with the remap); `_RECONNECTING` includes it; the lane K warning is skipped in 1D (the chain cannot flip) |
| `ddgclib/methods/_presets.py` | presets `hydrostatic_1D`, `hydrostatic_2D`, `hydrostatic_2D_periodic`, `hydrostatic_3D` |
| `ddgclib/_boundary_conditions.py` | new `FreeSlipWallBC(wall_axis, wall_coord)` |
| `cases_dynamic/Hydrostatic_column/src/_column.py` | new: `build_column`, `run_column`, `column_errors`, `static_residuals`, `remap_arm`, `run_case` (the runner body) |
| `cases_dynamic/Hydrostatic_column/Hydrostatic_{1D,2D,3D,2D_periodic}.py` | rewritten as thin runners (`run_case(<preset name>)`) |
| `cases_dynamic/Hydrostatic_column/diagnose_column.py` | new: modes `arms`, `convergence`, `viscosity` (with `density_diffusion` rows), `free_surface`, `growth`, `periodic`, `remap` |
| `cases_dynamic/Hydrostatic_column/README.md` | new |
| `ddgclib/tests/test_material_delaunay.py` | new, 18 tests |
| `ddgclib/tests/test_case_hydrostatic.py` | +9 fast, +7 slow preset tests |
| `ddgclib/tests/test_methods.py` | +2 tests, +5 invalid combinations, warning cases |
| `ddgclib/tests/test_boundary_conditions.py` | +3 tests (`FreeSlipWallBC`) |
| `METHODS.md`, `debugging_plan.md`, `DEVELOPMENT.md`, audit F12 line | documentation |

Not changed: `_integrators_dynamic.py`, `stress.py`,
`mass_redistribution.py`, `initial_conditions.py`, `src/_setup.py` (still
used by the lane K driver and by older tests). No pinned number moved.

One existing default changed, a warning only: `SolverMethods(dim=1)
.dudt_fn(pressure_model=eos)` no longer emits the lane K "UNSTABLE"
warning (reconnecting connectivity + EOS without remap). In 1D the
rebuild is the sorted chain and cannot flip; the test in
`TestColumn1D` shows the equality with a non-reconnecting loop, and
`test_methods.py` covers the changed warning. Everything else is opt-in.

## 2. What the hand-rolled loops did that the library did not

| item in the old loops | where it is now |
|---|---|
| symplectic update, `_recompute_duals` + `cache_dual_volumes`, no reconnection, no retagging | 1D: `connectivity='delaunay'` (sorted chain). 2D: `dual_only`. 3D: `dual_only_bare`. All with `boundary_filter = walls` at build time so free-surface vertices stay free |
| gravity (`make_gravity_dudt`) | `methods.dudt_fn(..., body_force=g_vec)` |
| artificial viscosity `mu_art = 0.5 rho c0 dx` | the `mu` passed to `dudt_fn` (physics, recorded in `methods.json` extra) |
| adaptive time step `CFL dx_min / (c0 + u_max)` | fixed `dt = CFL dx_min / c0` at t = 0 (u_max is below 1 % of c0); no library change |
| reflecting clamp and `u_x = 0` on side vertices (`_2D_periodic`) | library `FreeSlipWallBC` |
| abort at `u > 10 c0` | not needed; `diagnose_column.py` has its own guard |
| point-value pressure errors (`_2D_periodic`) | integrated comparisons only (`column_errors`) |

1D: the preset equals the old loop to round-off, not bit for bit. The
library rebuilds the chain every step, which changes the neighbour
iteration order and so the summation order of the force: max difference
3.3e-17 in `u` after 128 steps; exactly equal once the old loop uses the
library rebuild (`TestColumn1D::test_library_run_matches_the_former_hand_rolled_loop`).

Needed in the library and added: `boundary_filter` on `bare_dual_refresh`
(3D), `FreeSlipWallBC`, `delaunay_material` (remap arm).

## 3. Both connectivity arms, measured

`diagnose_column.py arms`. Every arm is `PRESETS[case]` or a `.replace` of
it. Shipped refinement (33 / 145 / 145 / 189 vertices), Tait n = 1,
c0 = 10 sqrt(g H), mu = 0.5 rho c0 dx, CFL 0.25. "env(T)" = max|u| over
the last 4 t_ac before T (one period of the fundamental mode, so it is not
a phase sample). "drop" = uniform density at t = 0, "eq" = equilibrium
masses (`HydrostaticEOSMass`).

Arms: preset; remap = `.replace(connectivity='delaunay_material',
remap='conservative', redistribute_mass=True)`; convex =
`.replace(connectivity='delaunay', remap='conservative',
redistribute_mass=True)`; 3D only: `.replace(connectivity='dual_only')`.

2D and 1D, 200 t_ac (`arms --n-tac 200`):

| case | arm | ic | peak | env(10) | env(40) | env(100) | env(200) | L2 [Pa] | interior | mass drift |
|---|---|---|---|---|---|---|---|---|---|---|
| 1D | preset `delaunay` | drop | 8.897e-01 | 7.277e-01 | 3.929e-01 | 1.235e-01 | 1.727e-02 | 5.544e+02 | 5.589e+02 | 0 |
| 1D | preset | eq | 2.757e-06 | 2.418e-06 | 1.512e-06 | 4.785e-07 | 7.023e-08 | 2.750e-01 | 2.756e-01 | 0 |
| 2D | preset `dual_only` | drop | 2.482e-01 | 3.671e-02 | 6.166e-04 | 3.145e-04 | 1.138e-04 | 4.985e+01 | 5.751e+00 | 1.1e-16 |
| 2D | preset | eq | 6.979e-06 | 1.159e-06 | 2.241e-08 | 1.156e-08 | 4.226e-09 | 4.857e+01 | 2.256e-01 | 1.1e-16 |
| 2D | remap | drop | 2.541e-01 | 4.093e-02 | 6.238e-04 | 3.626e-04 | 1.244e-04 | 5.661e+01 | 3.932e+01 | -1.7e-14 |
| 2D | remap | eq | 5.302e-06 | 2.309e-06 | 2.452e-06 | 2.145e-06 | 2.289e-06 | 4.756e+01 | 3.613e+01 | -9.3e-15 |
| 2D | convex | drop | 4.210e+01 | 3.566e-01 | 3.277e-01 | 1.638e-04 | 2.776e-06 | 4.882e+03 | 4.882e+03 | -1.7e-15 |
| 2D | convex | eq | BLOW-UP, max\|u\| > c0 at 95.04 t_ac | | | | | | | |
| 2D free-slip | preset `dual_only` | drop | 2.561e-01 | 1.644e-01 | 2.507e-02 | 5.851e-04 | 1.115e-06 | 4.980e+01 | 2.823e-01 | 1.1e-16 |
| 2D free-slip | preset | eq | 6.591e-06 | 5.000e-06 | 7.682e-07 | 1.796e-08 | 3.432e-11 | 4.977e+01 | 2.258e-01 | 0 |
| 2D free-slip | remap | drop | 2.481e-01 | 1.443e-01 | 1.268e-02 | 9.936e-05 | 1.763e-06 | 5.924e+01 | 3.450e+01 | -2.3e-14 |
| 2D free-slip | remap | eq | 6.944e-06 | 6.294e-06 | 2.054e-06 | 2.863e-06 | 2.852e-06 | 5.513e+01 | 4.241e+01 | -2.0e-14 |
| 2D free-slip | convex | drop | 2.562e-01 | 1.445e-01 | 1.237e-02 | 9.322e-05 | 1.488e-05 | 7.549e+01 | 5.721e+01 | -2.6e-14 |
| 2D free-slip | convex | eq | 5.887e-04 | 5.634e-05 | 6.510e-05 | 2.804e-04 | 1.495e-04 | 5.888e+01 | 4.083e+01 | 1.9e-14 |

3D, 100 t_ac (`arms --n-tac 100 --only hydrostatic_3D`):

| arm | ic | peak | env(10) | env(40) | env(100) | L2 [Pa] | settled max\|a\| |
|---|---|---|---|---|---|---|---|
| preset `dual_only_bare` | drop | 1.439e-01 | 6.207e-03 | 2.965e-04 | 2.362e-04 | 1.581e+02 | 1.60e-05 |
| preset | eq | 3.207e-05 | 3.312e-06 | 9.465e-07 | 5.412e-07 | 9.864e-01 | 9.33e-08 |
| remap | drop | 1.566e-01 | 5.385e-03 | 1.718e-03 | 5.651e-04 | 1.498e+02 | 1.08e-02 |
| remap, second process | drop | 1.566e-01 | 5.389e-03 | 1.718e-03 | 5.575e-04 | 1.497e+02 | 1.15e-02 |
| remap | eq | 2.777e-05 | 1.469e-06 | 7.017e-07 | 9.856e-07 | 7.901e-01 | 1.14e-04 |
| convex | drop | 7.054e+00 | 2.630e-01 | 2.762e-01 | 1.756e-01 | 6.647e+03 | 1.10e+02 |
| convex | eq | 7.031e+00 | 2.678e-01 | 2.659e-01 | 1.668e+00 | 7.699e+03 | 1.26e+03 |
| `dual_only` | drop | 1.468e-01 | 7.414e-02 | 2.762e-02 | 3.875e-02 | 1.465e+04 | 3.95e-03 |
| `dual_only` | eq | 9.759e-02 | 6.039e-02 | 2.736e-02 | 3.900e-02 | 1.478e+04 | 3.98e-03 |

The 3D remap rows are those of the final (geometric) peel. With the first
peel they read 5.627e-04 / 1.566e-06 at 100 t_ac. Reproducibility of the
3D rows between processes: section 11.

Reading:

- The presets settle. From the equilibrium masses max|u| never exceeds
  3.2e-05 m/s and decays; from uniform density the envelope decays over the
  whole horizon.
- The remap arm is stable and follows the preset from uniform density
  (2D env(200) 1.24e-04 against 1.14e-04). From the equilibrium masses it
  does not decay below a reconnection noise floor of 2.3e-06 m/s (2D),
  2.9e-06 (free-slip), about 1e-06 in 3D (7.0e-07 at 40 t_ac, 9.9e-07 at
  100 t_ac). The floor is flat over 200 t_ac in 2D; 3D beyond 100 t_ac was
  not run.
- 3D remap arm from uniform density: the settled max|a| of 1.1e-02 m/s^2
  belongs to a column that still moves at 5.6e-04 m/s. It has the size of
  the viscous term (nu u / dx^2 = 3e-02 with nu = 4 m^2/s); that reading
  was not checked separately. From the equilibrium masses it is 1.1e-04.
- Its interior integrated error (36 to 42 Pa against 0.23 Pa) is not a
  pressure error: on the Delaunay cells the cell centroid is not the vertex.
  Attribution run (free-slip, eq, 40 t_ac): integrated 31.3 Pa,
  rho g x rms(centroid - vertex) = 31.1 Pa, error against the nodal value
  0.204 Pa (preset on the builder mesh: 0.213 integrated, 0.179 nodal).
  This nodal number is a diagnostic for the attribution only.
- Convex-hull Delaunay + remap is unstable on the free surface in every
  case with frozen top corners, and noisy and slowly growing without them
  (free-slip eq: KE fit grows, 5.9e-04 peak).
- 3D `dual_only` cannot hold the column even from the equilibrium masses:
  the 3D branch of `_retopologize` zeroes the dual volume of frozen
  vertices, `_resolve_pressure` then returns the reference pressure for the
  wall cells. That is why the 3D preset is `dual_only_bare`.
- Choice: fixed connectivity is the preset for every case (lower noise
  floor, lower settled residual, same cost in 3D, cheaper in 2D). The remap
  arm is the tested alternative for cases that must reconnect.

Wall times were taken on a loaded machine and are only comparable within
one table: 2D 200 t_ac preset 340 to 410 s, remap 440 to 460 s; 3D 100 t_ac
preset 1084 s, remap 1082 s.

## 4. The remap arm: diagnosis and fix

`diagnose_column.py remap`: 2D column, uniform density, 145 vertices,
8 t_ac. Each arm is a rebuild (library `_retopologize` = convex hull, or
`retopologize_material_delaunay`) inside the single-phase remap, with the
global mass rescale (`redistribute_mass_single_phase`, the library remap)
or without it (every vertex re-targeted to the snapshot pressure, total
mass free), run as `SolverMethods(connectivity='custom')` through
`run_column`. The two "+ rescale" rows are the library arms of section 3
(same peak values).

| arm | max\|u\| over 8 t_ac | at 8 t_ac | total volume | mass drift | max K(s-1) per rebuild |
|---|---|---|---|---|---|
| convex + rescale (library `delaunay` + remap) | 42.10 m/s (2.98 t_ac) | 7.687e-02 | 1.000000000 | -3e-15 | 2713 Pa |
| convex, NO rescale | 59.42 m/s (6.56 t_ac) | 9.167 | 1.029500477 | +1.514e-02 | |
| material + rescale (library `delaunay_material` + remap) | 0.2541 (1.06 t_ac) | 2.270e-02 | 0.994997934 | 1e-15 | 0.9920 Pa |
| material, no rescale | 0.2541 | 2.277e-02 | 0.995003052 | +5.617e-06 | |
| preset `dual_only` (reference) | 0.2482 (1.04 t_ac) | 2.126e-02 | 0.995025755 | 1e-16 | |
| control, closed lid: convex + rescale | 0.1192 | 6.267e-04 | 1.000000000 | 4e-15 | 2.981 Pa |
| control, g = 0, seed 1e-6: convex + rescale | 5.076e-06 (2.81 t_ac) | 1.649e-07 | 1.000000000 | | 1.1e-03 Pa |
| control, g = 0, seed 1e-6: material + rescale | 1.951e-06 (start) | 8.499e-09 | 1.000000000 | | 1.5e-04 Pa |

So the rescale is not the cause: removing it makes things worse (mass
drift +1.5 %, total volume 1.03), and with the hull fill removed it
changes the result in the third digit only. The offset is large in the
convex arm (2.7e+03 Pa per rebuild) because the hull pins the volume; it is
a symptom there. Two controls confirm the location: with a closed lid and
gravity the library remap is stable (max|u| 0.119, max K(s-1) 2.98 Pa,
volume exactly 1); with a free surface, g = 0 and a 1e-6 m/s seed the
convex arm does not decay at first (max 5.08e-06, 2.6 times the largest
seed velocity) while the material arm decays 1.95e-06 -> 8.5e-09.

The lane's original scratch prototype (`scratchpad/laneP/proto_remap.py`:
scipy Delaunay, then `_retopologize(skip_triangulation=True)` for the
duals) gave 69.61 m/s at 7.98 t_ac, mass drift +2.6 % and volume 1.050 for
the no-rescale arm, and the same numbers as above for the other three. The
unstable arm depends on the details of the dual refresh; the conclusion
does not.

Mechanism: the frozen top corners keep the hull at y = H. As soon as the
surface drops, Delaunay covers the gap with slivers between surface
vertices. They flip every step, they change the dual faces of surface
vertices by O(dx), and the total dual volume cannot fall below the hull, so
the column cannot compress: it ends with half the hydrostatic head
(integrated L2 4.9e+03 Pa = rho g H / 2; mean bottom pressure 4901 Pa in
the refinement 2 test, against 9730 on `dual_only`).

Fix (`ddgclib/methods/_retopo.py:retopologize_material_delaunay`):

1. fresh pressure snapshot on the old connectivity (remap only);
2. the material boundary: the oriented boundary facets of the old
   `HC._simplices`, cached by the previous call (`HC._material_boundary`),
   evaluated at the current positions;
3. disconnect, Delaunay through hyperct `connect_and_cache_simplices`;
4. `peel_outside_boundary` (section 4b);
5. `boundary_from_simplices` tags, `compute_vd`, `cache_dual_volumes` (wall
   half cells in 2D and 3D), no edge-area cache;
6. `bV` = boundary narrowed by `boundary_filter`;
7. re-target every vertex to the snapshot, one exact mass rescale (remap).

It returns the relative change of the domain volume across the rebuild and
warns when it exceeds `domain_tol` (default 1e-3).

It lives in `_retopo.py`, not in `_integrators_dynamic.py` (protocol rule
2), and is selected by `SolverMethods(connectivity='delaunay_material')`.
It is registered on the `connectivity` axis, not on `remap`, because the
defect is in the rebuild, not in the remap.

## 4b. The peel (rewritten after review)

First version (topological): from the hull inwards, remove a simplex when
all its vertices are old boundary vertices and it exposes a facet that is
not an old boundary facet. The review showed that this removes FLUID in
3D. The squares of a planar wall are cocircular, so both diagonals are
Delaunay and qhull picks either from one rebuild to the next. The new wall
facet is then not an old boundary facet, a tetrahedron with four wall
vertices that owns it is removed, and the removal cascades. Reproduced
here with the reviewer's probes before the fix
(`scratchpad/reviewP/peel_lattice2.py`, `peel_box.py`):

| mesh (walls planar and fixed, interior vertices moved) | total volume, first peel | geometric peel |
|---|---|---|
| lattice cube n = 4, jitter 1e-3, 30 steps | min 0.790, 29 steps below 1 | 1 to 4e-16, 0 steps |
| lattice cube n = 5, 30 steps | min 0.917, 30 steps below 1 | 1 to 4e-16 |
| lattice cube n = 6, 15 steps | min 0.955, 15 steps below 1 | 1 to 4e-16 |
| `box()` refinement 2, closed, random walk 0.03, 40 steps | min 0.979, 15 steps below 1 | 1 to 9e-16 |
| lattice cube n = 3, one call (reviewer's number) | 41.7 % lost | change below 1e-12 (test) |

The shipped column was protected only by its mesh: the builder puts a
vertex in every cell centre, so no tetrahedron has four wall vertices.

Final version (geometric), three ingredients:

- **Inside or outside is decided by the winding number.** Only a simplex
  whose vertices are ALL old boundary vertices can be hull fill (one with
  an interior vertex is never removed, so no interior vertex is orphaned).
  Such a simplex goes when its centroid is outside the old boundary:
  `winding_number` (2D signed angles, 3D signed solid angles, Van Oosterom
  and Strackee) below 1/2. Whether the old facets are facets of the new
  triangulation is not asked.
- **Flat simplices are dropped when exposed.** qhull returns flat
  tetrahedra in the wall planes of a structured mesh (the reviewer counted
  785 simplices against 768 on the column). A flat simplex has no inside;
  it is removed when it has a facet no other live simplex shares
  (`_drop_exposed_flat`, tolerance 1e-12 of the longest edge cubed). The
  column cache now holds 768 at 2 t_ac. Flat simplices enclosed between
  cospherical lattice cells stay, as on the plain rebuild (removing one
  would open a crack).
- **The boundary is oriented when its simplices are built.**
  `oriented_boundary` orients each facet by the vertex of its owner that
  is not on it. That is only valid while the owner is not inverted, so the
  result is cached at the end of the call (`HC._material_boundary`) and
  reused, with the new coordinates, by the next one. The first prototype
  of the geometric peel re-derived the orientation at each call. A thin
  simplex at the 3D surface inverts within one step, the facets then do
  not form a closed surface, the winding numbers come out as 0.497 to
  0.519, and the test misfires: 3D column domain changes of up to
  +8.3e-05 per call (13 of 185 calls), peak 0.1848 m/s instead of 0.1566,
  env(40) 1.956e-03 instead of 1.72e-03. With the cached orientation the
  winding numbers are integers to 1.5e-10 and the numbers are those of
  section 3. `test_orientation_is_taken_when_the_simplices_are_built` is
  the 2D miniature of this.

Domain volume across the rebuild (`diagnose_column.py remap`, the value the
function returns, remap arm):

| run | calls | max \|dV / V\| per call | sum | calls with a change | max K(s-1) |
|---|---|---|---|---|---|
| 2D no-slip, refinement 3, 20 t_ac | 905 | 2.2e-16 | 1e-14 | 0 | 0.99 Pa |
| 2D free-slip, refinement 3, 20 t_ac | 905 | 2.2e-16 | 0 | 0 | 0.31 Pa |
| 3D refinement 1, 20 t_ac | 185 | 1.953e-05 | 1.953e-05 | 1 | 37.2 Pa |
| 3D refinement 2, 10 t_ac | 185 | 1.221e-06 | 1.617e-06 | 8 | 2.43 Pa |
| 3D refinement 2, equilibrium start, 10 t_ac | 185 | 6.6e-09 | 7.8e-09 | 9 | 0.048 Pa |

2D results are bit-identical to those of the first peel: every 1D and 2D
row of `arms_40.json` is equal in every stored field, and the pins
`PIN_UMAX`, `PIN_VOLUME`, `PIN_2D_REMAP_KE_40` did not move. The 3D remap
numbers moved in the third digit (section 3).

The 3D changes are real and are a limit of the method (section 10): the
largest one is the first rebuild, where the Delaunay diagonals of the four
corner squares of the free surface differ from the builder's
(4 x h^2 delta / 6 with delta = 3e-05, the first step's drop).

## 5. Validation

Shipped runs (uniform-density start, default horizon):

| runner | horizon | max\|u\| peak -> end | settled max\|a\| | integrated L2 | max\|p V - int P dV\| |
|---|---|---|---|---|---|
| `Hydrostatic_1D.py` | 200 t_ac | 0.8897 -> 1.441e-02 | 1.304e-01 | 554.4 Pa (5.65e-03 rho g H) | 245.3 |
| `Hydrostatic_2D.py` | 100 t_ac | 0.2482 -> 3.014e-04 | 1.006e-04 | 50.19 Pa (5.12e-03) | 1.289 |
| `Hydrostatic_2D_periodic.py` | 100 t_ac | 0.2561 -> 2.590e-04 | 1.883e-02 | 50.55 Pa (5.15e-03) | 1.241 |
| `Hydrostatic_3D.py` | 40 t_ac | 0.1439 -> 2.858e-04 | 2.684e-04 | 174.4 Pa (1.78e-02) | 3.708 |
| `Hydrostatic_2D.py --arm remap` | 100 t_ac | 0.2541 -> 3.470e-04 | 1.228e-04 | 57.20 Pa (5.83e-03) | 1.291 |
| `Hydrostatic_3D.py --arm remap` | 40 t_ac | 0.1566 -> 1.597e-03 | 1.811e-02 | 173.4 Pa (1.77e-02) | 4.246 |

(The 1D end value 1.441e-02 is the last sample; the envelope over the last
4 t_ac is 1.727e-02, which is what the preset notes quote. The remap rows
were re-run with the final peel; the 2D one is unchanged in every digit,
the 3D one is a single realisation, see section 11.)

`Hydrostatic_2D.py` matches the unmodified runner of lane S (KE 2.2552e-06
against 2.2540e-06 J, max|u| 3.014e-04 against 3.013e-04, settled max|a|
1.006e-04 in both); the remaining difference is the fixed time step.

Convergence (`diagnose_column.py convergence`: equilibrium masses, 60 t_ac,
presets):

| case | refinement | vertices | L2 [Pa] | L2 / rho g H | interior L2 | max\|p V - int P dV\| |
|---|---|---|---|---|---|---|
| 1D | 3 / 4 / 5 / 6 | 17 / 33 / 65 / 129 | 1.037 / 0.2204 / 0.04484 / 0.009369 | 1.06e-05 ... 9.6e-08 | same | 0.687 / 0.0854 / 0.0107 / 0.00135 |
| 2D | 2 / 3 / 4 | 41 / 145 / 545 | 136.1 / 48.57 / 17.20 | 1.39e-02 / 4.95e-03 / 1.75e-03 | 0.8945 / 0.2255 / 0.05659 | 9.954 / 1.243 / 0.1553 |
| 2D free-slip | 2 / 3 / 4 | 41 / 145 / 545 | 144.1 / 49.77 / 17.40 | 1.47e-02 / 5.07e-03 / 1.77e-03 | 0.8963 / 0.2222 / 0.05070 | 9.954 / 1.243 / 0.1553 |
| 3D | 1 / 2 | 35 / 189 | 2.154 / 0.9770 | 2.20e-04 / 9.96e-05 | 2.027 / 0.9791 | 0.2534 / 0.01514 |

Ratios per refinement: 1D 4.70, 4.91, 4.79; 2D total 2.80, 2.82, interior
3.97, 3.99, max 8.01, 8.00; free-slip total 2.89, 2.86, interior 4.03, 4.38;
3D 2.20.

Why the 2D total converges at order 1.5 (section 7): the pressure of a
boundary half cell settles on the nodal value, which differs from the cell
average by O(dx) on a volume fraction O(dx). `ddgclib.analytical` compares
1D boundary cells and all 3D cells with point value times volume, so the
offset does not show there. The 3D "integrated" error is therefore not a
cell integral.

## 6. Viscosity: how much, and why

`diagnose_column.py viscosity` (100 t_ac):

| case | ic | alpha_art (mu) | status | env(10) | env(100) | KE decay / t_ac | viscous theory |
|---|---|---|---|---|---|---|---|
| 1D | drop | water (1e-3 Pa s) | bounded | 9.247e-01 | 8.182e-01 | 1.2e-04 | 2.5e-09 |
| 1D | drop | 0.05 (1548) | decays | 8.912e-01 | 6.558e-01 | 3.994e-03 | 3.855e-03 |
| 1D | drop | 0.5 (15476) | decays | 7.277e-01 | 1.235e-01 | 3.884e-02 | 3.855e-02 |
| 2D free-slip | drop | water | bounded | 2.714e-01 | 2.893e-01 | 8.9e-04 | 7.9e-08 |
| 2D free-slip | drop | 0.05 (159) | decays | 2.554e-01 | 1.387e-01 | 1.263e-02 | 1.253e-02 |
| 2D free-slip | drop | 0.5 (1591) | decays | 1.644e-01 | 5.851e-04 | 1.257e-01 | 1.253e-01 |
| 2D no-slip | drop | water | BLOW-UP at 64.10 t_ac | | | | |
| 2D no-slip | drop | 0.05 | settles | 1.854e-01 | 6.099e-04 | | |
| 2D no-slip | drop | 0.5 | settles | 3.671e-02 | 3.145e-04 | | |
| 2D no-slip | eq | water | flat | 1.635e-05 | 1.652e-05 | -4.5e-04 (grows) | |
| 2D no-slip | drop | water + `density_diffusion` 0.05 | BLOW-UP at 63.22 t_ac | | | | |
| 2D no-slip | drop | water + `density_diffusion` 0.1 | BLOW-UP at 57.03 t_ac | | | | |

Two separate roles:

1. Damping of the acoustic ringing of the uniform-density start. Theory:
   KE rate `nu k^2` with `k = pi / (2 H)`. Measured to 1 % in 1D and in the
   free-slip column. The damping time grows like H / dx (52 t_ac for the 1D
   amplitude at 33 vertices), which is why the 1D default horizon is 200
   t_ac.
2. The settled state is not stable without it (next section): saddle
   modes on every mesh, flutter of the free surface on some. With the
   viscosity of water the no-slip drop exceeds c0 at 64 t_ac; alpha_art
   0.05 is enough to settle.

So the column does need artificial viscosity, and not only for the
acoustics. The shipped 0.5 is kept. It is not hiding a fast instability:
at alpha_art 0.05 every case still settles.

## 7. Free surface, and the modes of the settled column

`diagnose_column.py free_surface` and `growth` (2D, settled from the
equilibrium masses on mu_art, then linearised by central differences,
h = 1e-7, `SolverMethods(dim=2, connectivity='dual_only')`; the asymmetry
is identical for h = 1e-6 and 1e-8). "closed lid" is the control: top
frozen, every fan closed.

Shipped mesh (refinement 3, 145 vertices):

| | no-slip | free-slip | closed lid |
|---|---|---|---|
| degrees of freedom (on open fans) | 240 (14) | 256 (30) | 226 (0) |
| settled max\|a\| [m/s^2] | 5.4e-06 | 2.6e-06 | 5.0e-09 |
| surface pressure [Pa] | -0.192 | -0.192 | |
| \|F - (-dE/dx)\| / weight, surface vertices | 5.4e-07 | 4.5e-07 | |
| same, interior vertices | 3.7e-07 | 3.1e-07 | 4.8e-07 |
| \|K - K^T\| / \|K\|, all | 1.115e-01 | 1.152e-01 | 5.0e-10 |
| closed-fan block | 4.7e-10 | 3.5e-10 | 5.0e-10 |
| open-fan rows | 1.115e-01 | 1.152e-01 | 0 |
| largest real eigenvalue of d a / d x [s^-2] | +0.1246 | +0.1284 | +0.1242 |
| complex eigenvalues | none | none | none |

Growth rates against refinement (`growth`). A real positive eigenvalue is
a saddle direction (monotone growth); a complex pair is flutter
(oscillation at omega with growth Re sqrt(lambda)):

| kind | refinement | dof | saddle [1/s] (x g/c0) | saddle modes | flutter [1/s] | omega [rad/s] | complex eigenvalues |
|---|---|---|---|---|---|---|---|
| no-slip | 2 | 56 | 0.3176 (1.014) | 15 | 0.7518 | 141.0 | 2 |
| free-slip | 2 | 64 | 0.3438 (1.098) | 20 | 0 | | 0 |
| closed lid | 2 | 50 | 0.3155 (1.007) | 13 | 0 | | 0 |
| no-slip | 3 | 240 | 0.3530 (1.127) | 91 | 0 | | 0 |
| free-slip | 3 | 256 | 0.3584 (1.144) | 104 | 0 | | 0 |
| closed lid | 3 | 226 | 0.3524 (1.125) | 85 | 0 | | 0 |
| no-slip | 4 | 992 | 0.3675 (1.173) | 435 | 1.219 | 507.7 | 8 |
| free-slip | 4 | 1024 | 0.3687 (1.177) | 464 | 0.0524 | 664.8 | 2 |
| closed lid | 4 | 962 | 0.3673 (1.173) | 421 | 0 | | 0 |

Largest growth rate against viscosity (largest real part of the damped
linear system, 1/s; `mu_art` = 0.5 rho c0 dx):

| mu [Pa s] | 0 | 1e-3 (water) | 0.1 | 1 | 10 | 0.1 mu_art | mu_art |
|---|---|---|---|---|---|---|---|
| no-slip, refinement 3 (saddle) | 0.353 | 0.3526 | 0.320 | 0.151 | 0.0194 | 1.22e-03 | 1.22e-04 |
| no-slip, refinement 2 (flutter) | 0.752 | 0.7517 | 0.746 | 0.693 | 0.170 | 2.35e-03 | 2.35e-04 |
| closed lid, refinement 2 (saddle) | 0.3155 | 0.3155 | 0.309 | 0.256 | 0.0713 | 2.35e-03 | 2.35e-04 |

(refinement 3: closed lid the same as no-slip to 0.5 %, free-slip up to
8.4 % higher. At refinement 2 the residual growth under mu_art is the
saddle creep in all three: the flutter is gone.)

The fastest mode followed in a real run (seed 1e-7 m, mu = 0), amplitude
over seed against the linear prediction `|cosh(rate t)|`:

| mesh | mode | 50 t_ac | 100 t_ac | 150 t_ac | 200 t_ac |
|---|---|---|---|---|---|
| no-slip r3 | saddle 0.3530 1/s | 1.1614 / 1.1613 | 1.6970 / 1.6968 | 2.8120 / 2.8117 | 4.8161 / 4.8155 |
| closed lid r3 | saddle 0.3524 1/s | 1.1608 / 1.1607 | 1.6944 / 1.6943 | 2.8046 / 2.8043 | 4.7984 / 4.7978 |
| no-slip r2 | flutter 0.7518 1/s, 141 rad/s | 1.856 / 1.813 | 5.561 / 5.401 | 18.88 / 18.43 | 60.98 / 59.22 |
| closed lid r2 | saddle 0.3155 1/s | 1.134 / 1.134 | 1.546 / 1.546 | 2.383 / 2.383 | 3.777 / 3.777 |

(sample times 50.9, 99.5, 150.3, 198.9 t_ac at refinement 2.) A scratch
run of the no-slip r3 saddle to 300 t_ac gave 14.627 against 14.625.

Sloshing seed (`u = a exp(k (y - H)) (-sin kx, cos kx)`, k = pi,
a = 1e-3 c0 = 0.0313 m/s, 100 t_ac, refinement 3), kinetic energy maximum
per 20 t_ac window over the initial value:

| case | arm | mu | windows | end |
|---|---|---|---|---|
| 2D no-slip | preset | water | 1.000, 0.977, 0.980, 0.983, 0.977 | does not grow |
| 2D no-slip | remap | water | 1.010, 0.915, 0.835, 0.853, 0.830 | does not grow |
| 2D free-slip | preset | water | 1.000, 0.9985, 0.9985, 0.9985, 0.9986 | does not grow |
| 2D free-slip | remap | water | 1.006, 1.006, 1.006, 1.007, 1.006 | does not grow |
| all four | | mu_art | decays to max\|u\| 7e-05 | |

Answers to task 5:

- For the settled column the open-fan mismatch that lane K measured (30 to
  45 % of the pressure force) is not visible as a force: it is proportional
  to the surface pressure, and the surface cells settle at -0.19 Pa. The
  force on a surface vertex equals the energy gradient to 5e-07 of its
  weight, the same as in the interior.
- It is visible in the linearisation, and it does matter without
  viscosity. The open-fan rows make the stiffness matrix non-symmetric (11
  to 16 % of its norm; the closed-fan block is symmetric to 5e-10). On the
  shipped refinement 3 mesh every eigenvalue is still real, but that is
  specific to that mesh: the no-slip column has a flutter pair at
  refinement 2 (growth 0.752 1/s, e-fold 42 t_ac, confirmed in a real run:
  amplitude x61 after 199 t_ac against x59 predicted) and at refinement 4
  (1.219 1/s, e-fold 26 t_ac; eigenvalues only, no run), the free-slip
  column at refinement 4 (0.052 1/s). The closed lid never has one. The
  flutter is faster than the saddle below.
- The artificial viscosity removes it: at refinement 2 the largest growth
  rate drops from 0.752 (water) to 2.35e-03 1/s at 0.1 mu_art, which is the
  saddle creep (the closed lid has the same 2.35e-03). Refinement 4 against
  viscosity was not computed.
- A sloshing seed of 1e-3 c0 with the viscosity of water does not grow over
  100 t_ac on the refinement 3 mesh, in either arm.
- Independent of the free surface, the discrete hydrostatic equilibrium is
  a saddle of the discrete energy: the closed lid, with a symmetric
  stiffness matrix, has the same positive eigenvalues. The rate does not
  refine away: 1.01, 1.13, 1.17 g/c0 at refinement 2, 3, 4 (closed lid).

Probable mechanism of the saddle (reasoned, not verified by a targeted
test): in the continuum a barotropic column is neutrally stratified because
the second variation of the energy is a perfect square,
`(rho c^2 / V) (dV - g V dy / c^2)^2`. That needs `sum_i p(y_i) V_i` to be
constant to second order, which holds for the nodal quadrature only when p
is linear. The compressible profile is not, and the second derivative of
the interpolation error with respect to the vertex positions does not
shrink with dx. The residue is of order `g^2 / c^2`, the measured size. A
stiffer EOS lowers the rate like 1 / c0.

Not explained: the no-slip drop with the viscosity of water blows up at
64 t_ac on the refinement 3 mesh, which has no flutter pair around the
SETTLED state and whose saddle rate only doubles an amplitude in that time.
During the drop the surface pressure is far from zero, so the open-fan
mismatch is large; a finite-amplitude flutter is the likely cause, not
shown. The free-slip drop at the same amplitude stays bounded.

## 8. Why the "periodic" column is not periodic

`diagnose_column.py periodic`: `periodic_rectangle(L=1, h=1, refinement=3,
periodic_axes=[0])`, then one library `_retopologize(periodic_axes=[0])`
(what the integrator does at the start of every step), then
`SolverMethods(dim=2, connectivity='periodic', periodic_axes=(0,))`.

| quantity | value | expected |
|---|---|---|
| total dual volume from the builder | 1.0000000000000016 | 1 |
| total dual volume after the periodic retopology | 2.488281250000001 | 1 |
| simplices in the cache | 287 | 256 |
| vertices whose dual faces do not close | 45 of 136 (29 interior) | 0 |
| interior vertices tagged boundary | 6 | 0 |
| column run | max\|u\| > c0 at 0.66 t_ac, peak 540 m/s, frozen set 8 -> 0 | |

Three separate defects: seam simplices are measured with raw coordinates
(`simplex_dual_volumes` knows nothing about the period), the ghost
resolution leaves a non-manifold simplex set on a structured mesh
(cocircular squares get different diagonals in the ghost copies), and the
min-image dual faces then do not close. `d_ij` in the viscous flux is not
min-imaged either (known). The existing periodic test tolerates a force of
up to 200 at uniform pressure 1000 on seam vertices. On top of that the
path reconnects every step without a remap. None of this was fixed here;
it belongs to a periodic single-phase lane (and probably to the shearing
plate).

## 9. DO-NOTs (measured)

- Do not run a free surface on `connectivity='delaunay'`, with or without
  `remap='conservative'` (42 m/s at 3 t_ac, half the hydrostatic head, blow
  up at 95 t_ac from the equilibrium masses). Use `delaunay_material`.
- Do not look for the cure in the mass rescale of the remap: without it the
  run is worse (59.4 m/s, mass drift +1.5 %; 69.6 m/s and 2.6 % in the
  scratch prototype).
- Do not decide "outside the fluid" from the facets (the first peel): wall
  diagonals change between 3D rebuilds and fluid is removed (lattice cube
  down to 0.790 of its volume). Decide it geometrically.
- Do not re-derive the orientation of the old boundary at the next call: a
  thin surface simplex inverts within a step, winding numbers come out as
  0.5, the 3D domain flickers by 8e-05 per call. Orient at build time.
- Do not use `delaunay_material` with an EOS and without the remap (128
  m/s: the lane K instability; `dudt_fn` warns).
- Do not use `dual_only` for a 3D single-phase EOS case whose wall pressure
  is not P0 (L2 1.5e+04 Pa on the column). Use `dual_only_bare` with a
  `boundary_filter`.
- Do not use `connectivity='periodic'` for a single-phase EOS case (dual
  volume 2.488 for a unit domain).
- Do not initialise the pressure of boundary cells with the dual-cell
  average when the force reads `v.p` (no EOS): static residual 1.9075 m/s^2
  in 2D at every refinement, 2.5e-14 with nodal values. The same holds for
  masses: `HydrostaticEOSMass` (point values) gives max|a| 4.6e-03 at
  refinement 3 and converges, volume-averaged masses give 2.86 m/s^2 at
  every refinement.
- Do not run the column without artificial viscosity and expect it to stay
  settled (saddle 0.35 1/s on every mesh, free-surface flutter up to 1.22
  1/s on some); `density_diffusion` does not help (blow-up at 63.22 / 57.03
  t_ac for delta 0.05 / 0.1, 64.10 without; `diagnose_column.py viscosity`).
- Do not conclude from one mesh that a free surface is free of flutter: the
  no-slip column has none at refinement 3 and has it at 2 and 4.
- Do not read the settled max|a| of a free-slip run without removing the
  wall-normal component on the wall vertices: it is the wall reaction (190
  to 270 m/s^2 raw, 1.5e-05 after projection). `residual_acceleration` does
  this.
- Do not sample a ringing column at multiples of half its period: the
  values look like growth. Use a window maximum.

## 10. Known limits

- `delaunay_material` does not constrain the new triangulation to the old
  boundary facets and does not recover a missing one. 2D: a boundary edge
  is missing only when an interior vertex encroaches on it; never on the
  column (0 to round-off over 905 calls). 3D: a free-surface square whose
  Delaunay diagonal differs from the old one is hit on the column (8 of
  185 calls at refinement 2), and the domain then gains or loses the
  sliver between the two triangulations: at most 1.2e-06 of the volume per
  call, 1.6e-06 in total; 1.95e-05 once at refinement 1; 9.8e-05 when a
  0.004 bowl is pushed into the builder surface in one go (test). With the
  remap a change c shifts the pressure everywhere by about -K c (2.4 Pa at
  refinement 2, 37 Pa at refinement 1). The change is returned and warned
  above `domain_tol`; a facet recovery by local flips was not implemented.
  A simplex with an interior vertex that reaches across the old boundary
  is kept whole. Topology changes of the free surface are not supported
  (as with `dual_only`).
- 3D remap arm at refinement 1 (35 vertices, ONE free surface vertex): from
  uniform density it settles at a total volume of 0.99713 against 0.99545
  for the preset (analytic 0.99503) and an integrated L2 of 0.216 rho g H
  against 0.042 at 80 t_ac; from the equilibrium masses it is fine
  (3.1e-04 against 2.2e-04). It was the same with the first peel (volume
  0.99740 at 20 t_ac in both). Not explained; probable cause, not tested:
  the Delaunay diagonals of the four corner squares change how much volume
  a drop of the single free vertex displaces. At refinement 2 there is no
  such offset (L2 173.4 against 174.4 Pa at 40 t_ac).
- It needs `HC._simplices` at the first call and has no merge step; inlet
  or outlet BCs that add or delete vertices are not supported.
- Remap arm: noise floor 2e-06 m/s (2D) and 1e-06 (3D) from the
  equilibrium masses; 3D checked to 100 t_ac only, and after about 10 t_ac
  the 3D run from uniform density is reproducible between processes to two
  digits only (section 11). The interior integrated error reads 36 to 42 Pa because
  of the centroid offset (section 3).
- The saddle modes of section 7 are damped, not removed. At alpha_art 0.5
  the e-fold time is 2.6e+05 t_ac. The free-surface flutter is removed by
  the artificial viscosity at refinement 2; refinement 4 against viscosity
  and a real run of the refinement 4 flutter were not done. No fix of
  either was attempted (an energy-consistent free-surface force would
  remove the flutter).
- The 3D integrated comparison is point value times volume.
- No 2D or 3D run is bit-identical to the old loops: they used an adaptive
  time step. Only the 1D loop was compared (round-off).
- The old figures in `fig/` (`hydrostatic_*d_*`) and the three one-cell
  notebooks are stale and were left in place.
- `connect_and_cache_simplices` does not reset `HC._edge_to_apex` when it
  replaces the cache (`peel_outside_boundary` does). Not on the single
  phase force path; `_fixup_periodic_duals` and the curvature routines read
  that map.

## 11. Tests

- `ddgclib/tests/test_material_delaunay.py` (18): oriented boundary and
  winding number in 2D and 3D, peel (hull fill removed, domain volume and
  boundary facets preserved, nothing peeled on a convex domain, boundary
  orientation survives an inverted thin simplex), `TestPeel3D` (the review
  defects: lattice cube with all-boundary tetrahedra over 10 jittered
  rebuilds, the coarsest lattice in one call, a stirred closed builder box,
  a non-planar free surface with the cached boundary on the second call),
  remap invariance, the returned domain change and its warning,
  validation, 3D half cells, and the column through `integrate` for the
  material, convex and `dual_only` arms. Pins `PIN_UMAX`
  0.20282519399577198, `PIN_VOLUME` 0.995147730045464 (unchanged by the
  review fix).
  The first version's `test_fluid_simplices_are_never_peeled` was vacuous
  (an empty facet set makes the boundary vertex set empty, so nothing can
  be removed on any mesh) and is gone; `TestPeel3D` fails on the first
  peel.
- `ddgclib/tests/test_case_hydrostatic.py`: 9 fast (presets, static
  balance, 1D against the old loop, 1D decay rate against theory, 1D
  equilibrium, 2D drop, 2D free-slip, 3D smoke) and 7 slow (3D 40 t_ac, 3D
  `dual_only` fails, 3D remap arm 2 t_ac at refinement 2, 2D both arms
  40 t_ac, free-slip, convergence).
  Pins: `PIN_1D_UMAX_PEAK` 0.8463965668751099, `PIN_1D_KE_END`
  8.07861451666529, `PIN_2D_UMAX_PEAK` 0.1951860472291084, `PIN_2D_KE_END`
  3.1650897086706908e-06, `PIN_2DP_UMAX_PEAK` 2.631506910977075e-05,
  `PIN_3D_UMAX_PEAK` 0.08910127097486757, `PIN_2D_KE_40`
  9.872765069804785e-07, `PIN_2D_REMAP_KE_40` 2.721558596123262e-06, and
  new in the fix round `PIN_3D_REMAP_UMAX_PEAK` 0.15658060026054665
  (rel 1e-6; the peak at 0.87 t_ac was identical in six processes).
- `test_methods.py`, `test_boundary_conditions.py`: builder, validation
  (also `backend` under `delaunay_material`), warnings, `boundary_filter`
  on the bare refresh, `FreeSlipWallBC`.
- Repeatability, as measured (the first version of this log claimed
  "agrees in every printed digit" for all of `arms` at 40 t_ac; that holds
  for 1D and 2D only):
  - 1D and 2D: every row of `diagnose_column.py arms` at 40 t_ac (all
    arms, both initial conditions) is equal in every stored field to the
    JSON of the first run, before and after the review fix. The two 2D
    fast pins are identical to the last digit in four fresh interpreters.
  - 3D, fixed connectivity (`dual_only_bare`, `dual_only`): three
    processes agree to 4e-09 relative from uniform density and to 2e-06
    from the equilibrium masses (quantities at the 1e-07 level); the peak
    values are identical. This is not from this lane (it is also on the
    library `dual_only` path). The 3D pins are peak values at rel 1e-9.
  - 3D, `delaunay_material` + remap, uniform density: reproducible to two
    digits after about 10 t_ac. env(40) 1.718e-03 / 1.728e-03 / 1.728e-03
    and settled max|a| 2.56e-02 / 2.31e-02 / 2.31e-02 in three processes;
    env(100) 5.651e-04 / 5.575e-04 in two. The mesh is cospherical, so
    round-off decides Delaunay ties. From the equilibrium masses three
    processes agreed to 13 digits (env(40) 7.0165e-07). Every 3D remap
    number in this log, in `_axes.py` and in METHODS.md is therefore one
    realisation of a spread of that size.
- Suites at the end of the fix round (each run twice): ddgclib fast 1047
  passed, 12 skipped, 2 xfailed (1010 at the start of the lane, 1040
  before the fix round); slow 23 passed, 1 xfailed (16, 22); hyperct 301
  passed (`pytest hyperct/tests -k "not benchmark"`, not touched by this
  lane). All ten pins of the first round were recomputed after the fix in
  a fresh process and are identical to the last digit.
- No linter is installed in the `ddg` environment (`ruff` missing); the
  changed files were compiled and checked for unused imports only.

## 12. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
C=cases_dynamic/Hydrostatic_column
$PY $C/Hydrostatic_1D.py                 # 200 t_ac, about 1.5 min
$PY $C/Hydrostatic_2D.py                 # 100 t_ac, about 2.5 min
$PY $C/Hydrostatic_2D.py --arm remap
$PY $C/Hydrostatic_2D_periodic.py
$PY $C/Hydrostatic_3D.py                 # 40 t_ac, about 8 min
$PY $C/diagnose_column.py arms --n-tac 200 --only hydrostatic_1D hydrostatic_2D hydrostatic_2D_periodic --procs 14
$PY $C/diagnose_column.py arms --n-tac 100 --only hydrostatic_3D --procs 8     # about 18 min
$PY $C/diagnose_column.py convergence    # about 12 min
$PY $C/diagnose_column.py viscosity
$PY $C/diagnose_column.py free_surface   # refinement 3, about 10 min
$PY $C/diagnose_column.py free_surface --n-refine 2   # the flutter mesh, about 2 min
$PY $C/diagnose_column.py growth         # refinement 2, 3, 4: about 30 min
$PY $C/diagnose_column.py periodic
$PY $C/diagnose_column.py remap          # section 4 and 4b, about 2 min
$PY -m pytest ddgclib/tests/test_case_hydrostatic.py ddgclib/tests/test_material_delaunay.py -q
$PY -m pytest ddgclib/tests/test_case_hydrostatic.py -m slow -q
```

JSON of every table: `cases_dynamic/Hydrostatic_column/results/diagnose/`
(`arms_40`, `arms_100`, `arms_200`, `convergence`, `viscosity`,
`free_surface`, `free_surface_r2`, `growth`, `periodic`, `remap`; the
directory is git-ignored). `arms_40`, `arms_100`, `viscosity` and `remap`
are from the fix round (final peel); `arms_200` (1D, 2D) was not re-run
because its 2D rows are bit-identical at 40 t_ac.

Notes for the reviewer: `lane_diff.sh P` also lists the concurrent edits of
the capillary_rise_energy_grad agent, which are not part of this lane. Both
repositories were committed by someone else during this lane (ddgclib
`465c61f`, hyperct `c82bc80`, the lane S work, 16:50); this lane ran no git
write command and its changes are uncommitted in the ddgclib working tree.
Scratch scripts behind the early probes: `scratchpad/laneP/`
(`proto_remap.py`, `diag_remap.py`, `probe*.py`, `spectrum*.py`,
`ic_residual.py`). Section 4 no longer depends on them
(`diagnose_column.py remap`). Fix round: the reviewer's probes are in
`scratchpad/reviewP/`, the before and after runs and the per-call trace of
the 3D arm in `scratchpad/fixP/` (`before_lattice2.txt`, `before_box.txt`,
`trace3d.py`, `arms40_v2_{a,b,c}.txt`, `arms100_v2_{a,b}.txt`).
