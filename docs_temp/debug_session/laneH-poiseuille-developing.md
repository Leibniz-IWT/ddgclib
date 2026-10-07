# laneH: Hagen-Poiseuille 2D and 3D develop the analytical profile

Date: 2026-10-01 / 02; fix round 1 after independent review on
2026-10-02 (section 14; the sections before it describe the final state).
Closes the open inlet items of lane L and audit finding M2 (3D stall). Evidence base:
`docs_temp/audit_2026-09-25/cases_poiseuille_hydrostatic_bridges.md`
(sections 1, 3, 13), `docs_temp/debug_session/laneL-frozen-set-membership.md`
(sections 4.3 and 7), METHODS.md case matrix.

(An older `laneH-2d-over-decay.md` in this directory belongs to the
droplet campaign of July; it is not related.)

## 0. Verdict

- The Lagrangian 2D channel and the 3D pipe develop the Poiseuille
  profile from a plug, through the presets `hagen_poiseuille_2D` and
  `hagen_poiseuille_3D` on the library integrator. Shipped runs: 2D l2
  1.086e-02 from the developed profile, 3D 1.99e-02 (dual-volume weighted,
  downstream half of the channel). Transverse velocity 6e-17 and 3e-17, no
  vertex outside the walls, walls frozen and unmoved, vertex count steady.
- Four things stopped the 2D run, all of them fixed in the library or in
  the case setup (section 2). The decisive one is a method: the two-point
  viscous flux is not linearly precise on a sheared mesh. New value
  `viscous_flux='simplex_gradient'` (new axis `viscous_flux`).
- The 3D stall was the case-local `retopologize_cylinder` (it froze 121
  of 126 vertices within 300 steps on a short pipe). It is replaced by the
  library connectivity `delaunay` with `frozen_set='membership'`; the
  closure is deleted. The 3D preset also needs
  `pressure_flux='simplex_gradient'` (new value), because the centred flux
  reads the `batch_e_star` area cache in 3D.
- New BC `PeriodicInletBufferedBC`: an upstream buffer of prescribed plug
  motion, the inlet counterpart of `OutletBufferedDeleteBC`.
- Every existing pin is bit-identical; defaults are unchanged. The lane L
  wall-collapse reproducer keeps its numbers and digests (it is pinned to
  the configuration before this lane).
- What remains between the runs and the analytical profile is resolution
  (2D: the error falls by 2.7 to 3.3 per refinement, measured) and, in 3D,
  the polygonal wall (section 6).
- One finding outside the brief, NOT fixed: the 2D branch of
  `dual_area_vector` flips the area vector of strongly skewed edges
  (section 7). No integrated vertex of this lane is affected.

- Fix round 1 (section 14): the `backend` axis now runs in 3D (a backend
  NAME used to stop the first 3D retopology, so `run_cluster.py` and the
  `--backend` arm of the 3D runner crashed); the two 3D
  `pressure_flux='centred'` arms are quoted as ranges over 9 processes;
  `PeriodicInletBufferedBC` injects a frozen row once, whatever the time
  step.

Not solved: continuity is not enforced (there is no pressure solve; the
pressure is prescribed), so this is not an entrance-length benchmark
(section 3).

## 1. What changed and where

| file | change |
|---|---|
| `ddgclib/operators/stress.py` | `viscous_force_simplex_gradient`, `pressure_force_simplex_gradient`, helpers `_vertex_simplices`, `_simplex_fan`; registry `viscous_flux_methods`; `pressure_flux_methods['simplex_gradient']`; `stress_force` / `stress_acceleration` take `viscous_flux=` (default `'two_point'`: the statements of the default path are the ones that ran before) |
| `ddgclib/_boundary_conditions.py` | new class `PeriodicInletBufferedBC(PeriodicInletBC)`; the base class is untouched. Fix round: a row whose injected vertex was frozen is dropped from the ghost (`_row`, `_frozen_rows`) |
| `ddgclib/dynamic_integrators/_integrators_dynamic.py` | fix round only: `_resolve_backend` (backend name to hyperct instance, one per name), called before the 3D `batch_e_star`; `backend` documented in the `_retopologize` docstring |
| `ddgclib/methods/_axes.py` | axis `viscous_flux` (`two_point`, `simplex_gradient`); value `simplex_gradient` on `pressure_flux`; evidence on `pressure_flux.centred`, `edge_area_source.shared_vd_2d`. Fix round: control, notes and evidence of the axis `backend`; `workers` evidence; process ranges in the `pressure_flux` evidence |
| `ddgclib/methods/_config.py` | field `viscous_flux`; validation (single phase only, not under `periodic`); `dudt_fn` binds it when not default; `pressure_flux` needs an EOS only for `acoustic-riemann` |
| `ddgclib/methods/_presets.py` | `hagen_poiseuille_2D`: `viscous_flux='simplex_gradient'`, `workers=None` (was 20); `hagen_poiseuille_3D`: `connectivity='delaunay'` (was `custom`), `frozen_set='membership'`, both fluxes `simplex_gradient`, `workers=None` (was 8) |
| `cases_dynamic/Hagen_Poiseuile/src/_setup.py` | new `setup_poiseuille_developing(dim, ...)` (2D and 3D); `setup_poiseuille_2d_lagrangian` kept as it was (lane L reproducer) |
| `cases_dynamic/Hagen_Poiseuile/src/_run.py` (new) | `run_developing`, `run_case`, `CASES` (shipped parameters of both cases): the body of both runners. `src/_params.py` (pipe formulas, imported by the older notebooks) is not touched and not used |
| `cases_dynamic/Hagen_Poiseuile/src/_metrics.py` (new) | `profile_error`, `fluxes`, `census`, `PlaneFlux` |
| `cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py` | thin runner (the pasted inlet TODO of the old docstring is gone) |
| `cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py` | thin runner; `retopologize_cylinder` and the hand-built BC / IC functions are gone; the parameter names `visualize_hp3d.py` imports are kept |
| `cases_dynamic/Hagen_Poiseuile_3D/run_cluster.py` | forwards to `run_case` with `backend` / `workers` replaced on the preset (runs since the fix round; docstring says what each backend needs) |
| `cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py` (new) | `static`, `arms2d`, `arms3d`, `convergence`, `slivers`; fix round: `arms3d --only LABEL --tag NAME` to repeat arms in fresh processes |
| `cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py` | the `hp2d` arm replaces `viscous_flux='two_point'` (lane L configuration) |
| `cases_dynamic/Hagen_Poiseuile/visualize_hp2d.py`, `Hagen_Poiseuile_3D/visualize_hp3d.py` | read the new result directories |
| `cases_dynamic/Hagen_Poiseuile/README.md`, `Hagen_Poiseuile_3D/README.md` (new) | run instructions, results |
| `ddgclib/tests/test_simplex_gradient_flux.py` (new) | 30 tests + 1 strict xfail (section 7) |
| `ddgclib/tests/test_boundary_conditions.py` | `TestPeriodicInletBufferedBC`, 6 tests + 1 in the fix round (incommensurate step) |
| `ddgclib/tests/test_case_hagen_poiseuille.py` | `TestDeveloping2D` (7, fast), `TestDevelopingSlow` (3, slow); fix round: `TestDeveloping3DBackendAxis` (2, fast) |
| `ddgclib/tests/test_methods.py` | fix round: `TestEffectiveMethods::test_3d_backend_axis_fills_the_same_edge_area_cache` (3 values of the axis) |
| `ddgclib/tests/test_frozen_set.py` | the HP2D reproducer replaces `viscous_flux='two_point'`; numbers unchanged |
| `ddgclib/tests/test_pressure_flux_stabilisation.py` | the registry key set gains `simplex_gradient` |
| `METHODS.md`, `DEVELOPMENT.md`, `debugging_plan.md` | documentation |

hyperct: not touched. `_integrators_dynamic.py`: not touched by the lane;
the fix round added the backend resolution (section 14.1). With
`backend=None` the function executes the statements it executed before.

Deviations from protocol rule 2, declared. (a) New code paths belong
next to `_retopo.py`, not in `stress.py`: the two fluxes are options of
`stress_force`, selected by a key exactly like
`pressure_flux='acoustic-riemann'`, which already lives there with its
registry. (b) The new values are registered with status `opt-in`, not
`experimental`: they were registered after their measurement, are pinned
and are what two presets run. (c) Their `SolverMethods` plumbing tests are
in `test_simplex_gradient_flux.py`, next to the operator tests, not in
`test_methods.py` (the test of the `backend` axis added in the fix round
is in `test_methods.py`).

## 2. Diagnosis: what stopped the runs

### 2.1 2D (`Hagen_Poiseuile_2D.py` as lane L left it)

1. The pressure was advected. `pressure_model=None` reads `v.p`, the
   vertices carry it, and the inlet copies the ghost's point values, so
   the fluid behind the inlet has no pressure gradient (and a sawtooth of
   `G` per period, audit M3) while the initial fluid carries its old
   `-G x0`. Sheared rows then produce `dp/dy`, which is what pushed
   vertices through the walls in lane L's run. Fix: the pressure is a
   field of position, re-imposed on every vertex after every step
   (`DirichletPressureBC` over `HC.V`; no library change).
2. The first fluid column was the hull. Its cells are open (an open cell
   sees zero ambient pressure), and the Delaunay rebuild fills the gap
   between the column and the frozen inlet corners with simplices that
   couple fluid to the corners. With a linearly precise viscous flux that
   coupling is taken seriously: the inlet column went from 0.1 to
   -0.83 m/s in one step in the prototype. Fix: `PeriodicInletBufferedBC`
   (section 4.3): every hull vertex that is not a wall is a buffer vertex
   of prescribed motion.
3. The seam column of the unit cell was injected twice per period, one
   step apart (lane L, C3): pairs `U dt` apart, stiff. Fix: the buffered
   inlet drops the leading face of the ghost and carries the lead of the
   last injected column across the reset.
4. The two-point viscous flux is not linearly precise on a sheared mesh
   (section 5.1). With 1 to 3 fixed the profile is still 20 to 34 % off
   and fluctuates. Fix: `viscous_flux='simplex_gradient'`.

Also: `G` was the pipe formula (`8 mu U / r^2`), whose planar profile has
`U_max = 4 U_avg`; README and runner compared with 0.2. Now
`G = 12 mu U_avg / D^2`, `U_max = 1.5 U_avg`. And at `Re_D = 100` the
time constant of the development is 101 s, so nothing could have been
developed at `t = 30 s`; the shipped case is `Re_D = 10`.

### 2.2 3D (audit M2)

Reproduced with the functions of the old runner on a short pipe
(`L = 4`, refinement 1, `dt = 0.01`; scratch `stall3d.py`, which loads
`retopologize_cylinder` from the lane snapshot):

| step | vertices | frozen | frozen and not on the wall (strictly inside `0 < z < L`) | free |
|---|---|---|---|---|
| 1 | 122 | 74 | 2 (0) | 48 |
| 100 | 122 | 85 | 5 (3) | 37 |
| 200 | 122 | 114 | 34 (32) | 8 |
| 300 | 126 | 121 | 41 (35) | 5 |
| 600 | 126 | 121 | 41 (35) | 5 |

`retopologize_cylinder` rebuilt `bV` as `HC.boundary()` of its hand-made
connectivity at every step, without a filter and without a simplex cache
(`HC._simplices` is `None` throughout), and promoted every vertex whose
dual fan failed into `bV`. That is a ratchet: after 300 steps 121 of 126
vertices are frozen, 35 of them strictly inside the tube, and the 5 free
ones sit at `z` between 0.1 and 0.4. No library connectivity was missing:
the default `_retopologize` handles the cylinder (the coplanar
tetrahedra at the wall are in the simplex cache and the simplex-aware
duals cope with them), and `frozen_set='membership'` keeps the walls. So
the closure was deleted, not moved into `_retopo.py`. `HC._simplices` is
rebuilt by `connect_and_cache_simplices` at every step.

## 3. The model, and what it is not

There is no pressure solve and no EOS: the pressure `P = G (L - x)` is
prescribed (zero at the outlet plane). A fluid vertex therefore obeys
`m du/dt = G Vol + viscous force` along its path line and relaxes to the
developed profile with the time constant of the slowest viscous mode,
`t_dev = rho D^2 / (pi^2 mu)` in 2D and `rho R^2 / (2.405^2 mu)` in 3D.
The development length is about `5 t_dev U_max`. Consequences:

- Continuity is not enforced. No vertex leaves its row (the transverse
  force is zero to round-off), the rows move at their own speed, and the
  cell density `m_i / Vol_i` follows: it is not a constant-density flow.
  The mass flux of the markers is conserved along the channel; the volume
  flux is not (it goes from the plug value of the mesh to the developed
  one, section 5.4).
- The steady state does not depend on the masses (`G Vol + F_visc = 0`);
  the rate of the transient does (`Vol_i / m_i`).
- It is not an entrance-length benchmark of the incompressible equations.
  It validates the viscous operator, the walls, the inlet and the outlet
  on a moving, reconnecting mesh against an exact steady solution.

Pressure values are NODAL (`P(x_i)`), not dual-cell averages: the centred
flux and the simplex form are linearly precise for nodal values (lane P
measured 1.9 m/s^2 of residual with cell averages). The integrated
pressure error of `ddgclib.analytical` is therefore not zero by
construction and is not what this case validates; the velocity is, by a
dual-volume weighted norm.

Masses: `rho` times the dual volume the vertex has in the periodic tiling
of unit cells, for the mesh and for the ghost alike (read from the second
unit cell of the mesh).

## 4. The methods

### 4.1 `viscous_flux='simplex_gradient'`

`F_i = mu sum_{T contains i} G_T . a_iT`, with `G_T` the gradient of the
piecewise-linear velocity on the primal simplex `T` and
`a_iT = -|T| grad(phi_i)` the area vector of the barycentric dual face of
cell `i` inside `T` (the face of `T` opposite `i`, divided by `dim`). Per
edge: `sum_j w_ij (u_j - u_i)`, `w_ij = -mu sum_T |T| grad(phi_i) .
grad(phi_j)`, the cotangent weights in 2D. Same dual faces as the
two-point flux; the difference is the gradient put on a face: the full
simplex gradient instead of `(u_j - u_i) / |d|` along the edge.

Properties (tested): zero for a linear velocity at interior vertices of
any simplicial mesh; `w_ij = w_ji` (forces sum to zero); negative
semi-definite; natural (zero-gradient) condition on the hull. Diffusion
form, like the two-point flux. Cost 52 us per vertex in 2D, 134 us in 3D
(pressure form 33 / 96 us).

The geometry is read from `HC._simplices` at the current positions; only
the vertex-to-simplex incidence is cached, keyed by the identity of the
simplex list (hyperct replaces the list, never edits it; tested across a
retopology).

Flat simplices: `|T| <= 1e-3 l_min^dim` are left out. `a_iT` is built
from the adjugate, so nothing is divided by a small volume before the
filter. Measured (section 5.5): only hull simplices in 2D; in 3D also a
flat tetrahedron of four free vertices of equal radius in 2 of 600 steps.

### 4.2 `pressure_flux='simplex_gradient'`

`F_i = -sum_T |T| / (dim + 1) grad(p)_T`: minus the integral of the
piecewise-linear pressure gradient over the dual cell (volume form).
Exact for a linear pressure at every vertex, hull included, in 2D and 3D,
whatever the dual face areas are. Zero for a uniform pressure on an open
cell: no ambient pressure, which is right for a prescribed field and
wrong for a free surface under an EOS. With both fluxes on
`simplex_gradient` `stress_force` reads no dual face at all.

### 4.3 `PeriodicInletBufferedBC`

The zone `[inlet_pos - buffer_width, inlet_pos]` of the main mesh is a
buffer: every free vertex in it is put on its kinematic position with the
inlet velocity after every step (as the outlet buffer does), released at
`inlet_pos`, and fed at the upstream end from the ghost. The ghost has no
leading face (it is the periodic image of the trailing face of the copy
before it), and a new copy starts with the lead of the last injected
column, so the columns stay one spacing apart across the seam (tested:
all spacings 0.25 over two periods at `U dt` = half a spacing; without
the lead one gap was 0.375). Members of `bV` are never moved; a wall-row
vertex that the ghost injects is frozen by the wall BC (after the inlet
BC) at most one advection step from the upstream corner, or merged into
the corner when it lands within `cdist` of it (the 2D shipped run: 106
walls, 108 frozen at the end; 3D: 336, 352). Since the fix round the row
of such a vertex is dropped from the ghost (the BC sees on its next call
that a vertex it injected is in `bV`): a frozen vertex never leaves the
injection plane, so the row needs no further vertices. Before that, the
later columns of a wall row were injected too. They landed on the key of
the first one when `U dt` divides the column spacing (the shipped and
pinned steps) and otherwise each left one more frozen vertex within
`U dt` of the corner, without bound (section 14.3).

## 5. Measurements

Every dynamic arm is a preset or `preset.replace(...)`, built by
`run_developing`; setup `setup_poiseuille_developing`, `U_avg = 0.1`,
`rho = 1`, `D = 1` (`R = 0.5`), both buffers 1.0. "l2" is
`sqrt(sum V_i |u_i - u_ana(x_i)|^2 / sum V_i |u_ana(x_i)|^2)` over the
free vertices of the downstream half of the channel, all velocity
components included.

### 5.1 Static consistency (`diagnose_poiseuille.py static`)

Exact Poiseuille field on meshes whose interior vertices were jittered
(fraction of the mean spacing) and, in 2D, sheared by `0.3 * 4 y (1 - y)`
in x, then reconnected by the library `_retopologize`. Residual of each
flux against `G Vol`, median / max over interior vertices. "linear": a
linear velocity field with the wall shear rate of the profile (exact
answer: zero).

| mesh | centred pressure | simplex pressure | two-point, linear | simplex, linear | two-point, quadratic | simplex, quadratic |
|---|---|---|---|---|---|---|
| 2D refinement 3, jitter 0.2, shear 0.3 (41 vertices measured) | 1.1e-15 / 4.0e-15 | 2.9e-16 / 6.1e-16 | 0.64 / 2.6 | 2.4e-15 / 6.6e-15 | 0.16 / 1.04 | 0.054 / 0.40 |
| 2D refinement 4, same (231) | 2.3e-15 / 1.2e-14 | 5.9e-16 / 2.4e-15 | 1.41 / 5.2 | 8.2e-15 / 5.6e-14 | 0.43 / 3.8 | 0.068 / 0.69 |
| 3D cylinder refinement 2, builder (39) | 3.9e-03 / 4.8e-03 (p_ij ring 1.5e-15 / 4.7e-15) | 3.7e-16 / 8.0e-16 | 0.20 / 0.20 | 3.0e-16 / 8.0e-16 | 0.092 / 0.092 | 0.019 / 0.039 |
| 3D cylinder refinement 2, jitter 0.15 (39) | 1.3e-02 / 0.21 (p_ij ring 3.7e-15 / 1.5e-14) | 2.7e-16 / 9.4e-16 | 0.28 / 0.84 | 2.1e-16 / 7.1e-16 | 0.099 / 0.36 | 0.030 / 0.26 |

Reading. The two-point flux fails the linear field by O(1) of the driving
force and the error grows like `1 / h` (0.64 to 1.41 from refinement 3 to
4): it is inconsistent on such meshes, not just inaccurate. The nodal
residual of the exact QUADRATIC profile is not a consistency measure for
the simplex form (it is the Galerkin P1 discretisation: its nodal
truncation error on an irregular mesh is O(1) of `G Vol` while the
solution error converges, section 5.2); it is listed so that nobody
reads the 0.05 as a defect. On the unjittered 2D builder mesh the
two-point flux is linearly precise (1e-16) but 6.7 % off for the
quadratic profile.

### 5.2 2D

Shipped run, `PRESETS['hagen_poiseuille_2D']`, `Re_D = 10` (`mu = 0.01`,
`t_dev = 10.13 s`), `L = 12`, refinement 2, `dt = 0.05`, 2400 steps
(`results/hagen_poiseuille_2D/summary.json`; two processes of the lane,
the reviewer's and one after the fix round gave identical values in every
key):

| quantity | value |
|---|---|
| l2 on `6 <= x <= 12` at the end | 0.01085880248519068 |
| mean / max over the last quarter | 1.0888e-02 / 1.1672e-02 |
| `u_max` in the window (analytical 0.15) | 0.15085007 |
| largest transverse velocity in the window, any sample | 5.9e-17 |
| largest transverse velocity over ALL free vertices, every step (fix round) | 1.1e-16 |
| volume flux in the window, in `U_avg D` | 0.9644 (nodal quadrature of the parabola on the 7 rows: 0.96875) |
| vertices start / end | 473 / 478 |
| free vertices in the channel, min / max / end | 311 / 339 / 318 |
| vertices outside the walls | 0 |
| walls frozen / moved | 106 of 106 / 0 |
| wall time | 3 min 50 s (serial) |

Arms on the short channel (`diagnose_poiseuille.py arms2d`: `L = 4`,
`mu = 0.1`, refinement 2, `dt = 0.01`, 3000 steps, window `2 <= x <= 4`):

| arm | l2 end | tail mean / max | `u_max` | transverse velocity |
|---|---|---|---|---|
| preset | 1.1998e-02 | 1.045e-02 / 1.296e-02 | 0.15106 | 2.6e-17 |
| `.replace(viscous_flux='two_point')` | 0.2605 | 0.227 / 0.318 | 0.18862 | 0.19 |
| `.replace(pressure_flux='simplex_gradient')` | 1.1998e-02 (differs in the 6th digit) | 1.045e-02 / 1.296e-02 | 0.15106 | 6.3e-18 |
| `.replace(frozen_set='hull')` | identical to the preset in every digit | | | |

("transverse velocity" in these tables is the largest one among the free
vertices of the comparison window, over the samples of the run.)

Shipped parameters with `--viscous-flux two_point`: l2 1.040, `u_max`
1.71, transverse velocity 59 m/s, 26 vertices outside the walls, 322
vertices left of 473. The transverse velocity grows from round-off; a
flipped area vector (section 7) makes the two-point edge weight negative,
which is the probable seed (flips were counted in this arm, the causal
link was not isolated).

Convergence (`diagnose_poiseuille.py convergence --long`, `L = 4`,
`mu = 0.1`):

| refinement (fluid rows) | dt | l2 at `t = 10 s`, rows still regular | l2 at `t = 60 s`, rows sheared (tail mean) |
|---|---|---|---|
| 1 (3) | 0.02 | 1.083e-02 | 3.27e-02 (4.55e-02) |
| 2 (7) | 0.01 | 3.60e-03 | 1.14e-02 (1.22e-02) |
| 3 (15) | 0.005 | 1.34e-03 | 3.40e-03 (3.36e-03) |

Ratios 3.0 and 2.7 at `t = 10 s`, 2.9 and 3.3 at `t = 60 s` (a
second-order scheme would give 4). The error of the sheared mesh is three times that of the regular
one at the same refinement and it fluctuates as the rows slide past each
other (the discrete steady state depends on the current triangulation).

### 5.3 3D

Shipped run, `PRESETS['hagen_poiseuille_3D']`, `Re_D = 2` (`mu = 0.05`,
`t_dev = 0.865 s`), `L = 4`, refinement 2 (16-sided pipe), `dt = 0.01`,
1000 steps (`results/hagen_poiseuille_3D/summary.json`; two processes of
the lane, the reviewer's and one after the fix round gave identical values
in every key):

| quantity | value |
|---|---|
| l2 on `2 <= z <= 4` at the end | 0.019943600646088133 |
| mean / max over the last quarter | 1.9145e-02 / 2.1112e-02 |
| `u_max` in the window (analytical 0.2) | 0.19767 |
| largest transverse velocity in the window, any sample | 3.3e-17 |
| largest transverse velocity over ALL free vertices, every step (fix round; behind the release plane, section 14.4) | 1.6e-15 |
| volume flux in the window, in `U_avg A` | 0.940 |
| vertices start / end | 845 / 928 |
| free vertices in the channel, min / max / end | 380 / 412 / 389 |
| vertices outside the wall | 0 |
| walls frozen / moved | 336 of 336 / 0 |
| wall time | 14.5 min (serial, 0.87 s per step) |

Arms (`diagnose_poiseuille.py arms3d`: `L = 3`, `mu = 0.1`, refinement 1
= octagon, `dt = 0.01`, 600 steps, window `1.5 <= z <= 3`):

| arm | l2 end | tail mean / max | `u_max` | radial velocity | wall time |
|---|---|---|---|---|---|
| preset | 0.05661 | 0.0596 / 0.0675 | 0.19855 | 2.6e-18 | 47 s |
| `.replace(pressure_flux='centred')`: `batch_e_star` cache | 0.08111 to 0.08211 | 0.0887 to 0.0893 / 0.1657 to 0.1694 | 0.20434 to 0.20607 | 6.3e-03 | 52 s |
| centred on the `p_ij` ring (`connectivity='custom'` wrapper: `_retopologize(frozen_set='membership')`, then the cache cleared) | 0.056561 to 0.056566 | 0.05943 to 0.05946 / 0.06737 to 0.06739 | 0.19857 to 0.19858 | 1.26e-05 to 3.65e-04 | 138 s |
| `.replace(viscous_flux='two_point')` | 0.5296 | 0.510 / 0.534 | 0.27637 | 3.5e-18 | 54 s |

Process dependence (protocol rule 8; fix round). The preset and the
two-point arm are bit-identical from process to process (the lane's
process, the reviewer's, `p2` and `p5` of the fix round). The two centred
arms are not: their pressure force reads dual face areas (the cache, or
the `p_ij` ring), and their rows above are ranges over 9 fresh processes
(`results/laneH/arms3d.json` = p1 and `arms3d_p2.json` ...
`arms3d_p9.json`). Each arm takes one of two values, equal to 15 digits
within a group:

| arm | group | l2 end | tail mean | tail max | `u_max` | radial velocity | processes |
|---|---|---|---|---|---|---|---|
| cache | A | 0.08211494566330206 | 0.08872 | 0.16573 | 0.20434 | 0.00629618754380668 | 7 of 9 (p1, p3, p4, p6, p7, p8, p9) |
| cache | B | 0.08110864974622763 | 0.08927 | 0.16936 | 0.20607 | 0.00629618754380668 | 2 of 9 (p2, p5); also the reviewer's |
| ring | A | 0.0565663464877044 | 0.059428 | 0.067373 | 0.198569 | 3.648e-04 | 3 of 9 (p1, p5, p6) |
| ring | B | 0.0565613667606650 | 0.059458 | 0.067386 | 0.198577 | 1.258e-05 | 6 of 9 (p2, p3, p4, p7, p8, p9); also the reviewer's |

The groups of the two arms are not correlated (p5: cache B, ring A; p6:
cache A, ring A; p2: cache B, ring B). What does not depend on the
process: the radial velocity of the cache arm (its maximum is reached
before the runs part), that the cache arm is 43 to 45 % above the preset
in l2, and that the ring arm is within 1e-4 of the preset in l2 with a
radial velocity of 1e-05 to 4e-04. The 300-step configuration of the slow
test (`test_3d_centred_flux_on_the_area_cache_drifts_radially`) is still
bit-identical between processes (four concurrent processes of the
reviewer: l2 0.1041293294517488, radial velocity 0.004687758802582311).
The cause of the process dependence was not traced (rule 8 says the same
of lane P); a force that reads no dual face area is not affected.

At refinement 2 the cache arm reached radial velocities of 1.5e-02,
2.2e-02 and 0.17 m/s within 400 steps and the ring arm ran at 2.2 s per
step against 0.62 (scratch runs, stopped early; not in the JSON).

### 5.4 Inlet, outlet, vertex count

`PlaneFlux` counts the mass of the vertices that cross a plane, over
whole inlet periods (the inlet feeds one unit cell per `period / U_avg`;
a shorter count depends on which columns happen to cross, and a sum over
a slab is phase-locked to the columns: both were tried first and gave
misleading numbers).

| run | periods | mass flux in | mass flux out | share of the mass not in wall cells |
|---|---|---|---|---|
| 2D shipped | last 6 | 0.8333 | 0.8681 | 0.8333 |
| 2D fast test (`L = 3`, refinement 1) | 1 (the whole run) | 0.6667 | 0.6667 | 0.6667 |
| 2D short channel, `t = 20..60 s` (scratch probe, planes 0 / 0.5 / 1 / 2 / 4) | 4 | 0.8333 at 0, 0.5, 1; 0.82 at 2 | 0.8333 | 0.8333 |
| 3D shipped | 1 (the whole run) | 0.7004 | 0.8348 | 0.7004 |

In units of `rho U_avg A`. The inlet flux is exactly the fluid share of
the tiling: the wall cells hold mass that never moves, so the plug of the
mesh carries `1 - wall share` of the nominal flux, and the developed
volume flux (0.964 in 2D, 0.940 in 3D) is the nodal quadrature of the
profile. The outlet flux equals the inlet flux once the slowest row has
crossed the channel (`L / u(y_1)`: 183 s in the 2D shipped run, so 0.868
at 120 s still contains the initial spacing of the near-wall rows; the
3D number contains the whole start-up).

No pile-up and no depletion: 2D shipped 311 to 339 free vertices in the
channel (318 at the end); rows near the wall get denser and central rows
thinner, in proportion to `U_avg / u(y)`. On the short channel (`L = 4`,
refinement 2, 3000 steps, sampled every 10 steps; scratch `mindist.py`)
the inlet buffer holds 28 free vertices at every sample and the smallest
distance from a free channel vertex to any other vertex is 0.125, the row
spacing; the only closer pair of the mesh is the frozen wall vertex next
to the upstream corner (5.4e-04).

### 5.5 Simplices left out by the viscous flux (`diagnose_poiseuille.py slivers`)

| run | left out per step, at most | of them with a free interior channel vertex | smallest `|T| / l_min^dim` of a simplex with such a vertex |
|---|---|---|---|
| 2D, `L = 4`, refinement 2, 3000 steps | 2 | 0 | 0.329 |
| 3D, `L = 3`, refinement 1, 600 steps | 53 | 2, in 2 steps (256 and 257) | 0 |

The 3D ones are exactly flat: four free vertices of radius 0.3232, two
in each of two cross-sections (a planar rectangle), which qhull returns
as a tetrahedron when the two cross-sections line up. Leaving it out
breaks linear precision at those four vertices for that step. All other
simplices that are left out are coplanar wall or cap tetrahedra.

### 5.6 Workers

2D, refinement 2 (about 200 vertices), 150 steps: serial 38.2 ms per
step, `workers=20` 85.5 ms. The presets are serial; `--workers` is an arm.

## 6. What remains, and which axis it belongs to

| gap | size | attributed to |
|---|---|---|
| 2D shipped: l2 1.09e-02, `u_max` +0.6 % | falls by about 3 per refinement (3.3e-02, 1.1e-02, 3.4e-03 at refinement 1, 2, 3 on the sheared rows; 1.08e-02, 3.6e-03, 1.34e-03 on regular rows) | resolution: 7 rows, and the sheared triangulation (three times the error of the regular one). Not a method axis: with `viscous_flux='simplex_gradient'` the error converges |
| 2D with `viscous_flux='two_point'` | 0.26 (Re 1), 1.04 with ejected vertices (Re 10) | axis `viscous_flux` |
| 3D shipped: l2 1.99e-02, `u_max` -1.2 % | 5.7e-02 at refinement 1 | mesh: radial resolution (free vertices at 6 radii) and the polygonal wall (16 flat faces, cross-section 2.8 % below `pi R^2`; the comparison is with the circular pipe). Not separated from each other |
| 3D with `pressure_flux='centred'` | l2 0.081 to 0.082 (9 processes) against 0.057, radial velocity 6.3e-03 | axis `edge_area_source`: the `batch_e_star` cache (lane J, lane Q). On the `p_ij` ring the centred flux gives the preset's result at 2.9 times the cost |
| 3D with `viscous_flux='two_point'` | 0.53 | axis `viscous_flux` |
| volume flux 0.964 (2D) / 0.940 (3D) of `U_avg A`, mass flux of the markers 0.833 / 0.700 | | the model (no continuity, section 3) and the wall cells of the mesh |
| outlet mass flux above the inlet value at the end of the shipped horizons | 0.868 against 0.833 (2D) | horizon shorter than the transit time of the slowest row |

## 7. Finding outside the brief: orientation of the 2D dual area vector

`dual_area_vector` (2D, non-periodic branch) orients `A_ij` so that it
points away from `x_i` as seen from the midpoint of the dual segment.
That is right as long as the barycentres of the two triangles at the edge
subtend less than 180 degrees at `x_i`. When they subtend more (two large
angles at `x_i`), the straight segment between the barycentres passes
behind `x_i` and the rule flips the vector: it then points AGAINST the
edge. The exact rule is `A_ij . d_ij > 0` (`d_ij . A_ij` is a third of
the cross product of the two diagonals of the quadrilateral, positive for
any valid pair of triangles; the same holds for a boundary edge).

Measured (`rectangle(L=2, h=1, refinement=3)`, shear 0.3, jitter 0.2,
seed 0; scratch `orient_fix_check.py`, pinned as a strict xfail test):

| | library rule | oriented by `A . d > 0` |
|---|---|---|
| area vectors against their edge | 6 of 665 | 0 |
| closure `|sum_j A_ij|` of an interior cell, max | 0.2316 | 2.8e-17 |
| centred force of a linear pressure, `|F + V g| / (V |g|)`, max | 17.6 | 1.0e-14 |
| two-point weight `d . A / |d|^2`, min | -4.12 | positive |

Census with a counting probe (pytest plugin, no behaviour change) over
the suites: 2055 flipped vectors of 1 134 409 2D calls in the fast suite,
in six tests:

| test | flipped |
|---|---|
| `test_frozen_set.py::TestHagenPoiseuille2D::test_hull_policy_collapses_the_wall` | 737 |
| same, `test_membership_runs_past_it...` | 431 |
| `test_case_hagen_poiseuille.py::TestDeveloping2D`, two-point arm | 354 |
| `test_single_phase_remap.py::TestBoxDecayWithEOS::test_bare_delaunay_is_unstable` | 272 |
| `test_case_hagen_poiseuille.py::TestDeveloping2D`, preset | 252 (all at buffer vertices) |
| `test_material_delaunay.py::...::test_material_arm_follows_the_fixed_connectivity_run` | 9 |

Slow battery: 359, all in this lane's 2D refinement 2 test (buffer
vertices). In the preset runs every flipped vector belongs to a vertex of
the inlet or outlet buffer, whose motion is prescribed (located with
`orient_where.py`: 252 of 252 and 951 of 951), so no number of this lane
depends on it. No droplet, hydrostatic or dam-break pin contains one.

Not fixed, on purpose: the fix is two lines, but it changes forces in the
lane L reproducer, in lane R's "bare Delaunay + EOS is unstable" test and
in a lane P pin, i.e. measured statements of three earlier lanes. Whether
the instability of lane K / R (finding F12) or the dam-break ejection
depend on it is NOT known; that is the question of a lane of its own.

## 8. Pin safety

- Default paths: `stress_force` with `viscous_flux='two_point'` and
  `pressure_flux='centred'` executes the statements it executed before
  (the diff removes no line of the loop; the two new branches are behind
  key tests). `PeriodicInletBC` is untouched.
- Fast suite, slow battery: green, no pin moved (section 11).
- Lane L reproducer: `diagnose_frozen_set.py hp2d` gives the digests of
  the lane L log (`62106841d3f841a9` hull, `47e12338835537f3`
  membership) and the same table (walls released at step 250, 6.184e-02).
- 2D runs of this lane are bit-identical between processes (fast test
  twice, shipped run twice: every key of `summary.json` equal). 3D: the
  PRESET runs are too (the slow test configuration and the shipped run,
  two processes of the lane, the reviewer's and one of the fix round,
  equal in every key). Rule 8 of the protocol (3D reconnecting runs
  agree to two digits) does bite for the two 3D arms with
  `pressure_flux='centred'`, whose force reads dual face areas: they are
  reported as ranges over 9 processes (section 5.3). The first version
  of this log quoted single values for them and said the rule did not
  bite; the review showed that it does (section 14.2).

## 9. Known limits

1. No continuity (section 3). A case that needs the incompressible
   entrance flow needs a pressure solve or an EOS with the single-phase
   remap; neither is here.
2. `simplex_gradient` viscous: interior simplices below the flat
   tolerance break linear precision at their vertices for that step
   (3D: 2 of 600 steps). A reconnecting 3D mesh with many near-flat
   interior tetrahedra would be stiff: the weight of a thin simplex grows
   like `1 / thickness` until the tolerance removes it. (On the builder
   cylinder the smallest quality of a tetrahedron that is not flat is
   0.117; the moving mesh was not scanned for thin, non-flat ones.)
3. `simplex_gradient` couples a hull vertex to whatever the convex-hull
   fill connects it to. Integrated vertices must not be on the hull; the
   buffers take care of that here. It was not tried on a free surface.
4. `pressure_flux='simplex_gradient'`: no ambient pressure on open cells,
   not pairwise antisymmetric, not run with an EOS.
5. Neither flux exists for multiphase or under `connectivity='periodic'`
   (both raise).
6. `PeriodicInletBufferedBC`: the main mesh must reach to
   `inlet_pos - buffer_width` with a column on that plane; unit cells of
   length `period`; one extra wall vertex per wall row within one
   advection step of the upstream corner unless it lands within `cdist`
   (for any time step since the fix round; if `U dt` exceeds the column
   spacing, two columns can enter before the row is known as frozen,
   which was not run). A vertex that flows back across the inlet plane
   is captured. A buffer vertex that some other BC freezes takes its row
   out of the ghost, which is the intended behaviour for a wall row.
7. The buffers are integrated and then overridden, so their force
   evaluations are wasted (52 of the 370 free vertices at the end of the
   shipped 2D run).
8. Outlet: `OutletBufferedDeleteBC` as before. Its buffer has no wall
   vertices beyond `L`; the hull fill there touches buffer vertices only.
9. 3D comparison is with the circular pipe; the exact solution of the
   polygonal pipe was not computed.
10. Companion scripts. `visualize_hp2d.py` ran to the end on the shipped
    2D outputs and `visualize_hp3d.py --no-polyscope` on the 3D ones
    (both overwrote the older `hp2d_*` / `hp3d_*` figures of the runs
    before this lane in `fig/`). After the 2D run the parameter import of
    `visualize_hp2d.py` was changed once more (it reads `CASES` now); only
    its header was re-executed after that. The per-bin curves of
    `visualize_hp3d.py` alternate between rings and its `L_e` line is the
    incompressible correlation, which does not apply here. Not run:
    `view_polyscope` on the 3D snapshots, the polyscope viewer of
    `visualize_hp3d.py`. `run_cluster.py` was run in the fix round with
    every backend (section 14.1); not on a SLURM node and not at
    refinement 3.
11. 3D shipped run: outside the comparison window the transverse
    velocity is not flat at 1e-17; behind the release plane it reaches
    5e-16 to 2e-15 from the second quarter of the run on (section 14.4).
    Round-off level over the shipped horizon; a longer run was not made.
12. The `ddg` environment has no PyTorch, so the test suite covers
    `backend='torch'` only through its ImportError; the value itself was
    run by hand in two other environments (section 14.1).

## 10. DO-NOTs (measured)

- Do not use `viscous_flux='two_point'` on a mesh that shears or
  reconnects and expect a profile: its residual for a LINEAR field is 0.6
  to 5 of the driving force and grows with refinement.
- Do not put an integrated vertex on the hull when the viscous flux is
  `simplex_gradient`: the hull fill couples it to distant frozen vertices
  (0.1 to -0.83 m/s in one step at the old inlet).
- Do not advect a prescribed pressure with the vertices; re-impose it.
- Do not use `pressure_flux='centred'` in 3D on a reconnecting mesh for a
  flow that must stay parallel: the `batch_e_star` cache gives radial
  velocity (6.3e-03 at refinement 1, up to 0.17 at refinement 2).
- Do not measure a mass flux over less than a whole inlet period, or by
  summing `m u` over a slab (phase-locked to the columns: 0.70 was read
  where 0.8333 crossed).
- Do not read the nodal residual of the exact quadratic profile as the
  accuracy of `simplex_gradient`; use the linear field or the solution
  error.
- Do not freeze by `HC.boundary()` of a hand-built connectivity and
  promote failed fans (the old `retopologize_cylinder`): 121 of 126
  vertices frozen after 300 steps.
- Do not fix the orientation of the 2D area vector without re-measuring
  lanes K, L, P and R (section 7).
- Do not quote a late number of a 3D reconnecting run whose force reads
  dual face areas from one process: the cache arm ends at l2 0.0811 or
  0.0821 and the ring arm at radial velocity 1.3e-05 or 3.6e-04,
  depending on the process (section 5.3).
- Do not hand a backend NAME to hyperct's `batch_e_star`; it needs the
  instance (`_resolve_backend`). And do not resolve it inside the `try`
  of step 5b of `_retopologize`: an ImportError there would silently
  switch the dual volume and edge-area source.

## 11. Tests

New: `test_simplex_gradient_flux.py` (30 + 1 strict xfail: registry and
axes, linear precision in 2D and 3D against the two-point flux, momentum,
dissipation, cotangent weights, linear pressure on every vertex, no force
on the hull at uniform pressure, the 3D cache is not linearly precise,
flat simplices, simplex cache required and followed across a retopology,
`SolverMethods` plumbing and rejected combinations, presets);
`test_boundary_conditions.py::TestPeriodicInletBufferedBC` (6, and 1 in
the fix round: one extra wall vertex per wall whatever the step);
`test_case_hagen_poiseuille.py::TestDeveloping2D` (7 fast, one run of
4 s plus the two-point arm) and `TestDevelopingSlow` (3 slow: 2D
refinement 2, 3D preset, 3D centred arm). Fix round:
`test_case_hagen_poiseuille.py::TestDeveloping3DBackendAxis` (2 fast: the
3D preset with `backend='gpu'` / `'multiprocessing'` runs and equals the
serial run) and
`test_methods.py::TestEffectiveMethods::test_3d_backend_axis_fills_the_same_edge_area_cache`
(3: every value of the axis; `'torch'` asserts the ImportError when
PyTorch is absent).

Pins (`test_case_hagen_poiseuille.py`):

| pin | value | configuration |
|---|---|---|
| `PIN_2D_L2` | 0.013084885355719682 | preset 2D, `L = 3`, `mu = 0.1`, refinement 1, `dt = 0.02`, 500 steps; rel 1e-9 |
| `PIN_2D_UMAX` | 0.1495783625570828 | same |
| `PIN_2D_R2_L2` (slow) | 0.005388811850628623 | same at refinement 2 |
| `PIN_3D_L2` (slow) | 0.07252269818859532 | preset 3D, `L = 2`, `mu = 0.1`, refinement 1, `dt = 0.01`, 300 steps; rel 1e-6 |
| `PIN_3D_UMAX` (slow) | 0.19558769279936808 | same |

Changed tests: `test_frozen_set.py::TestHagenPoiseuille2D` builds its
methods with `viscous_flux='two_point'` (assertions untouched);
`test_pressure_flux_stabilisation.py::test_registry` expects the third
key.

Battery at the end:

| suite | before the lane | after |
|---|---|---|
| ddgclib fast (`-m "not slow"`) | 1087 passed, 12 skipped, 2 xfailed | lane: 1130 passed, 12 skipped, 3 xfailed (1087 + 30 simplex fluxes + 6 buffered inlet + 7 developing flow; the third xfail is the strict reproducer of section 7). After the fix round: 1136 passed, 12 skipped, 3 xfailed (+ 3 backend axis, + 2 3D backend arms, + 1 buffered inlet) |
| ddgclib slow (`-m slow`) | 23 passed, 1 xfailed | 26 passed, 1 xfailed (23 + 3); unchanged by the fix round |
| hyperct (`pytest hyperct/tests -k "not benchmark"`) | 316 passed | 316 passed (not touched) |

The bare `pytest -q` at the hyperct root still stops at 4 collection
errors (duplicate test module names, as lane L recorded); the command in
the table is the one that runs (re-checked in the fix round: 4 errors
during collection, in `hyperc_rl_quick_figs_delete/tests` and
`hyperct/tests`). Wall time: fast 95 to 100 s, slow 182 to 186 s (the
three new slow tests take 15 to 18 s each, 51 s together).

## 12. Reproduce

    cd /home/endres/projects/ddgclib
    PY=/home/endres/anaconda3/envs/ddg/bin/python
    $PY -m pytest ddgclib/tests/test_simplex_gradient_flux.py ddgclib/tests/test_boundary_conditions.py ddgclib/tests/test_case_hagen_poiseuille.py -q -p no:cacheprovider
    $PY -m pytest ddgclib/tests/test_case_hagen_poiseuille.py -q -m slow -p no:cacheprovider     # about 55 s
    # sections 5.1, 5.2, 5.3, 5.5
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py static        # 10 s
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms2d        # 8 min
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d        # 5 min
    # the two centred arms again, each in a fresh process (ranges, section 5.3)
    for k in 2 3 4; do $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d --only centred --tag p$k & done; wait
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py convergence   # 12 min (--long: refinement 3 to t = 60 s, about an hour)
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py slivers       # 3 min
    # shipped runs
    $PY cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --headless                                   # 4 min
    $PY cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --headless --viscous-flux two_point --tag two_point --no-anim
    $PY cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --headless                                # 15 min
    # lane L reproducer, unchanged digests
    $PY cases_dynamic/Hagen_Poiseuile/diagnose_frozen_set.py hp2d
    # fix round: the backend axis in 3D (gpu falls back to numpy without PyTorch)
    $PY -m pytest ddgclib/tests/test_methods.py ddgclib/tests/test_case_hagen_poiseuille.py -k "backend or Backend" -q -p no:cacheprovider
    (cd cases_dynamic/Hagen_Poiseuile_3D && $PY run_cluster.py --n-refine 1 --n-steps 50 --dt 0.01)

Records: `cases_dynamic/Hagen_Poiseuile/results/laneH/*.json` (each row
with the `methods` dict of its arm), `results/hagen_poiseuille_2D*/`,
`Hagen_Poiseuile_3D/results/hagen_poiseuille_3D/`.

Scratch only (not in the tree): the prototypes, `stall3d.py`,
`orient_fix_check.py`, `orient_where.py`, the counting plugin
`plug/orient_probe.py`, the plane-crossing probe; in
`scratchpad/laneH/` of the session. Fix round, in `scratchpad/fixH/`:
`backend_check.py`, `incommensurate_fix.py`, `shipped_check.py`,
`ucross_where.py`, `ucross_sliver.py`, `diag_redirect.py` and their
logs.

## 13. Stray outputs

File deletion in the repository was denied to this lane as well. Left
behind by smoke runs: `cases_dynamic/Hagen_Poiseuile/results/hagen_poiseuille_2D_smoke/`,
`cases_dynamic/Hagen_Poiseuile/fig/hagen_poiseuille_2D_smoke*` (4 files),
`cases_dynamic/Hagen_Poiseuile_3D/results/hagen_poiseuille_3D_cluster/`
and `fig/hagen_poiseuille_3D_cluster_*` (a 20-step check of
`run_cluster.py`; overwritten by the 50-step runs of the fix round). The older files in `Hagen_Poiseuile/results/` (top
level, `hull/`, `laneL_smoke/`) and in `Hagen_Poiseuile_3D/results/` (top
level), `results_full/`, `archive/`, `run.log` are outputs of runs before
this lane.

## 14. Fix round 1 (after independent review, 2026-10-02)

Verdict of the review: fail, on two blocking issues; all three suites
green. The reviewer reproduced the shipped runs, the pins, the static
table, the 2D arms, four of the six convergence rows, the sliver census
and the lane L digests digit for digit. Not reproduced: the recorded
digits of the two 3D centred arms, the claim that `run_cluster.py` runs,
and the statement about the extra wall vertex of the buffered inlet.

### 14.1 Blocking: the `backend` axis stopped every 3D run that set it

`run_cluster.py` in its default configuration (`--backend gpu`), both
examples of its docstring, and `Hagen_Poiseuile_3D.py --backend ...`
aborted at the first retopology with `AttributeError: 'str' object has no
attribute 'batch_cross_areas'`.

Cause. `SolverMethods.backend` is a NAME and `_retopologize` passed it on
to hyperct's `batch_e_star(backend=...)`, which calls methods of a backend
INSTANCE. A library defect older than the lane (it fails with `hull`,
`centred` and `two_point` as well); the `run_cluster.py` before the lane
built the instance itself, the lane's rewrite routed the name through the
axis and reported the runner as running on a check with `--backend numpy`,
which does not set the axis.

Fix, in the library (`ddgclib/dynamic_integrators/_integrators_dynamic.py`):
`_resolve_backend(backend)` turns a name into the instance of
`hyperct._backend.get_backend`, one instance per name (the multiprocessing
backend owns a pool; one per step would leak processes); `None` and
instances pass through. `_retopologize` calls it for `dim == 3`, before
the `try` of step 5b: inside it, the ImportError of a missing PyTorch
would be caught by `except (ImportError, NotImplementedError)` and the
run would silently continue on the other dual volume and edge-area
source. 1D and 2D do not read the backend and are untouched; with
`backend=None` nothing changes.

Measured (preset `hagen_poiseuille_3D` and `.replace(...)`, `L = 2`,
`mu = 0.1`, refinement 1, `dt = 0.01`, 40 steps; `backend_check.py`):

| arm | `ddg` environment (no PyTorch) | environment with PyTorch 2.8 + CUDA (RTX 4090) |
|---|---|---|
| preset | l2 0.23124619366781088 | the same |
| `backend='gpu'` | the same (falls back to numpy) | the same (the forces read no dual face area) |
| `backend='multiprocessing'` | the same | the same |
| `backend='torch'` | `ModuleNotFoundError: No module named 'torch'` | the same as the preset |
| `pressure_flux='centred'` | l2 0.2319544884568851, radial velocity 1.5218472228449773e-03 | the same |
| `pressure_flux='centred', backend='gpu'` | the same | 0.23195448845688516, 1.5218472228449777e-03 (areas computed on the GPU) |
| `pressure_flux='centred', backend='multiprocessing'` | the same | the same as numpy |

Before the fix (resolution disabled in a scratch call): the
`AttributeError` of the review.

`run_cluster.py --n-refine 1 --n-steps 50 --dt 0.01`: default backend
`gpu` with 8 workers, `--backend multiprocessing`, and (PyTorch 2.14 +
CUDA) `--backend torch` and `--backend gpu` all run and print the same
l2 (2.7691e-01 after 0.6 time constants); 6.7 steps per second without
PyTorch, 4.9 with it. `--backend torch` in the `ddg` environment stops
with the ModuleNotFoundError, as the docstring now says; the local
example of the docstring uses `gpu`. `Hagen_Poiseuile_3D.py --backend gpu
--n-refine 1 --steps 50 --tag cluster` runs.

Tests: `test_methods.py::TestEffectiveMethods::test_3d_backend_axis_fills_the_same_edge_area_cache`
(every value of the axis on a box: same keys and areas as numpy to rtol
1e-12, same dual volumes, one instance per name, instances pass through;
`'torch'` without PyTorch must raise ImportError) and
`test_case_hagen_poiseuille.py::TestDeveloping3DBackendAxis` (the preset
with `backend` replaced, 20 steps, equal to the serial run). The five
tests also pass in two environments with PyTorch 2.14 / 2.10 and CUDA,
where the `'torch'` branch computes.

### 14.2 Blocking: protocol rule 8 for the two 3D centred arms

The first version of this log quoted one process for the arms
`pressure_flux='centred'` on the cache and on the `p_ij` ring and said
that rule 8 did not bite. The reviewer's process gave other digits (cache
l2 0.08111 against 0.08211; ring radial velocity 1.26e-05 against
3.65e-04). `diagnose_poiseuille.py arms3d` got `--only` and `--tag`, and
the arms were repeated in 8 more fresh processes (p2 and p5 all four
arms, the others the two centred ones). Result in section 5.3: each arm
takes one of two values; the preset and two-point arms are bit-identical
in every process. The ranges replace the single values in
`ddgclib/methods/_axes.py` (`pressure_flux`: `centred` and
`simplex_gradient`), the preset notes, METHODS.md, both READMEs,
`debugging_plan.md` and sections 5.3, 6, 8 and 10 here. The conclusions
did not change.

Records: `results/laneH/arms3d.json` (p1, the lane's first process),
`arms3d_p2.json` to `arms3d_p9.json`. Three records of the lane
(`arms3d.json`, `arms2d.json`, `convergence.json`) carry a placeholder in
`methods.notes` (`NOTES_3D` / `NOTES_2D`): the preset notes were not
written yet when they ran. `arms2d.json` and `convergence.json` were
replaced by the re-runs of the fix round (every number identical, only
`methods.notes` and the wall times differ); `arms3d.json` keeps the
placeholder, its `methods` are otherwise the presets.

### 14.3 Non-blocking, fixed: the buffered inlet and an incommensurate step

The review found that the claim "later columns land on its key" holds
only when `U dt` divides the column spacing; otherwise every wall-row
column of the ghost left one more frozen vertex within `U dt` of the
upstream corner. `--dt` is on the runner CLI. Fixed in
`PeriodicInletBufferedBC` (section 4.3): a row whose injected vertex was
frozen is dropped from the ghost.

Fast-test configuration (`L = 3`, `mu = 0.1`, refinement 1) to `t = 30 s`
(`incommensurate_fix.py`; "before" = the same code with the row rule
disabled):

| `dt` | frozen vertices at the end, before / after (18 walls at the start) | vertices | l2, before = after |
|---|---|---|---|
| 0.02 (divides) | 18 / 18 (the injected vertex merges into the corner) | 43 -> 44 | 0.03783139571568113 |
| 0.0237 | 30 / 20 | 43 -> 53 / 43 -> 45 | 0.03794619980371149 |
| 0.017 | 30 / 20 | 43 -> 53 / 43 -> 45 | 0.0378429986994438 |

The profile was not affected by the extra vertices and is not affected
by the fix (they sit in the buffer zone, where only prescribed vertices
are). Test:
`TestPeriodicInletBufferedBC::test_one_extra_wall_vertex_whatever_the_step`.

The shipped steps take the new path too (the first wall-row vertex of
the 2D shipped run enters one step from the corner, so its rows are then
dropped: 108 frozen vertices at the end, as before). Re-measured after
the fix round, all bit-identical to the records of the lane: the 2D
shipped run and the 3D shipped run (every key of `summary.json`), the
five pins, `arms2d` (four arms), `slivers`, `static`, the preset and
two-point arms of `arms3d`, and all six convergence rows, the two at
refinement 3 included (0.0013422520413264984 at `t = 10 s`,
0.0033988273879066323 at `t = 60 s`; the review had left these two out).

### 14.4 Other non-blocking findings

- `test_mass_flux_in_equals_mass_flux_out`: the assertions are kept; the
  docstring now says that the equality of inlet and outlet count is a
  property of this configuration and horizon (each of the three rows
  sends two columns across `x = L` in 10 s), not a conservation law.
- "largest transverse velocity of the run" was the maximum over the free
  vertices of the comparison window. The texts say so now, and the
  shipped configurations were re-run with a callback over ALL free
  vertices at every step (`shipped_check.py`, `ucross_where.py`):
  1.1e-16 in 2D, 1.6e-15 in 3D. Per zone and quarter of the run:

  | run | zone | maximum in quarter 1, 2, 3, 4 |
  |---|---|---|
  | 2D | inlet buffer | 0 (prescribed) |
  | 2D | channel upstream of the window | 1.06e-16, 1.11e-16, 9.4e-17, 1.10e-16 |
  | 2D | window, every step | 5.7e-17, 5.9e-17, 5.9e-17, 5.3e-17 |
  | 2D | outlet buffer | 3.8e-17, 3.8e-17, 4.4e-17, 4.4e-17 |
  | 3D | inlet buffer | 0 (prescribed) |
  | 3D | channel upstream of the window | 3.6e-17, 6.6e-16, 5.2e-16, 1.6e-15 |
  | 3D | window, every step | 1.4e-17, 5.4e-17, 1.5e-17, 1.5e-17 |
  | 3D | outlet buffer | 1.6e-19, 2.5e-19, 4.6e-19, 4.6e-19 |

  2D is flat. In 3D the largest value (step 993 of 1000) belongs to a
  vertex of the outermost free ring (r = 0.414) at z = 0.20, just
  behind the release plane; the upstream zone shows values of 5e-16 to
  2e-15 from the second quarter on, 10 to 100 times the window. That is
  still round-off (1.6e-14 of `U_avg`), and the window stays at 1e-17,
  but it is not flat: whether it keeps growing over a longer horizon
  was not run, and the cause was not located. The flat tetrahedra of
  section 5.5 were tested as the candidate at refinement 1 and are not
  it there (`ucross_sliver.py`, `L = 3`, 600 steps): the two steps
  with a flat simplex at free channel vertices (256, 257) show no jump
  (2e-18 to 5e-18 from step 252 to 262) and the maximum over all free
  vertices of that run is 1.0e-17.
- The reproduce commands of the lane L log pointed at the runner this
  lane rewrote; a note there says which reproducer is kept.
- Registry: the `backend` axis says what each value does and where it
  was run; the `workers` evidence no longer says that the Poiseuille
  presets use it.
- Left as they are, declared: nodal pressure values (section 3), the
  three deviations from protocol rule 2 (section 1), the orientation
  defect of the 2D area vector (section 7, a lane of its own), the
  convergence order in 2D and the unseparated 3D residual (section 6),
  the stray smoke outputs (section 13; deletion in the repository is
  denied).

### 14.5 Battery after the fix round

ddgclib fast 1136 passed, 12 skipped, 3 xfailed (1130 + 6); slow 26
passed, 1 xfailed; hyperct 316 passed (not touched). No pin moved.
