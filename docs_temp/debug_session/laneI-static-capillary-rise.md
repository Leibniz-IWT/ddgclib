# laneI: static capillary rise on presets, hand-built 3D setups on exact volumes

Date: 2026-10-06 (cloud session, branch `claude/relaxed-davinci-tkh95t`).
Machine: 4 cores, Python 3.13.16, numpy 2.5.3, scipy 1.18.1.

## 0. Verdict

- `capillary_rise_2D.py` and `capillary_rise_3D.py` are thin runners on
  `PRESETS['capillary_rise_static_2D' / '_3D']` through
  `SolverMethods.integrate` (`src/_static.py`, no time loop in the case).
  Gravity is the `body_force`, the EOS the `pressure_model`, surface
  tension and the contact angle enter through the new method axis
  `contact_line='energy_gradient'` (`ddgclib.operators.free_surface
  .FreeSurface`, bound by `dudt_fn(free_surface=...)`), the reservoir
  through the new `HydrostaticReservoirBC`, the contact-line vertices
  through the new `AxialSlideBC`, the walls are frozen by membership
  (`boundary_filter`).
- The static answer is measured against the Young-Laplace meniscus of the
  same compressible hydrostatic profile (new `ddgclib.analytical
  .young_laplace_meniscus`, arc-length shooting, 2D and axisymmetric),
  whose volume-averaged height is Jurin's height up to the model's
  compressibility (+0.5 % at `c0 = 10 sqrt(g h_J)`). 2D slit, water,
  r = 2 mm, refinement 2 (113 vertices), 300 acoustic times, alpha_art
  0.05: from the flat meniscus the volume-averaged height is
  3.684283e-03 m against the reference 3.683887e-03 (+1.1e-4; Jurin
  3.665237e-03), surface RMS error 5.5e-05 m, max|u| 2.3e-06 m/s, max|a|
  6.9e-08 m/s^2: settled to round-off of the discrete equilibrium.
  Refinement 3 (417 vertices): section 4.
- 3D (octagonal tube at refinement 1, 87 vertices; 16-gon at refinement
  2, 517): the discrete tube is the polygon inscribed in the circle, whose
  force balance (perimeter / area) stands 8.2 % (2.1 %) above the round
  tube; against that reference the column settles at -1.1e-2 (refinement
  1, 100 t_ac, pre-shaped start). Section 4.
- Connectivity by measurement: `dual_only` (2D) / `dual_only_bare` (3D)
  are the presets. The reconnecting arm (`delaunay_material` + conservative
  remap) settles in 40 t_ac without creep but 5.1 % too high: measured
  worse for this case (section 5).
- Every hand-built 3D complex under `cases_dynamic/` (cube_flow,
  cube2droplet, `Hagen_Poiseuile/src/_geometry.py`, `setup_hydrostatic`)
  and the hand-built 2D ones (`setup_poiseuille_2d`, `_lagrangian`,
  `_developing`) get their simplex cache through one library call,
  `ddgclib.geometry.ensure_simplex_cache` (the same helper
  `DomainResult` now uses); their dual volumes tile the domain
  (`test_hand_built_simplex_cache.py`, 13 tests).
- One library correction found on the way: `HydrostaticEOSMass` used the
  closed form `P_ref + K expm1(alpha depth)`, exact only for `P_ref = P0`;
  it is now `P_ref + (K + P_ref - P0) expm1(alpha depth)` (the exact
  solution of `dP/dz = -rho(P) g` for a linear Tait fluid). Every
  existing caller has `P_ref = P0 = 0`, so nothing pinned moved (section 7).

## 1. What changed and where

ddgclib (hyperct untouched):

| file | change |
|---|---|
| `ddgclib/geometry/_retriangulation.py`, `geometry/__init__.py` | new `ensure_simplex_cache(HC, dim)`: rebuild the 2D / 3D cache of the existing connectivity if `HC._simplices is None` |
| `ddgclib/geometry/domains/_result.py` | `DomainResult.__post_init__` calls it (one place for builders and hand-built meshes) |
| `cases_dynamic/cube_flow/src/_setup.py`, `cube2droplet/src/_setup.py`, `Hagen_Poiseuile/src/_geometry.py` (`unit_cylinder`, `cube_to_tube`), `Hagen_Poiseuile/src/_setup.py` (three setups), `Hydrostatic_column/src/_setup.py` (`setup_hydrostatic`) | call `ensure_simplex_cache` after the last connectivity edit |
| `ddgclib/operators/free_surface.py` | new: `FreeSurface` (facets of the simplex cache classified by vertex membership: free, wetted wall; `force(v) = -dE/dx_v`, `energy`, `area_free`, `area_wet`), `facet_area`, `facet_area_gradient` |
| `ddgclib/methods/_axes.py` | new explicit axis `contact_line` (group forces, single phase, dims 2 and 3): `None` (validated, every single-phase pin) and `'energy_gradient'` (experimental, evidence from this lane) |
| `ddgclib/methods/_config.py` | field `contact_line`; validation (single phase, dims); `dudt_fn(..., free_surface=)` adds `F / m`; the wrapper carries `.free_surface`; raises when the axis and the object disagree |
| `ddgclib/methods/_presets.py` | presets `capillary_rise_static_2D` (`dual_only`), `capillary_rise_static_3D` (`dual_only_bare`, `edge_area_source='p_ij_simplex'`) |
| `ddgclib/_boundary_conditions.py` | new `AxialSlideBC(axis, vertices)` (lateral velocity zeroed, lateral position restored to anchors held by identity), `HydrostaticReservoirBC(ic, level)` (band below `level` reset to `ic.assign(v)` each step, `injected` running total) |
| `ddgclib/initial_conditions.py` | `HydrostaticEOSMass.assign(v)` (the per-vertex rule, shared with the BC); the closed form exact for `P_ref != P0` |
| `ddgclib/analytical/_meniscus.py`, `analytical/__init__.py` | new: `young_laplace_meniscus`, `MeniscusProfile`, `hydrostatic_pressure_tait`, `jurin_height` |
| `cases_dynamic/capillary_rise/src/_static.py` | new: `build_static_column`, `run_static`, `static_errors`, `mean_height`, `shape_error`, `cross_section`, `remap_arm`, `run_case` |
| `cases_dynamic/capillary_rise/capillary_rise_2D.py`, `_3D.py` | rewritten as thin runners (`run_case(<preset>)`) |
| `cases_dynamic/capillary_rise/diagnose_static_rise.py`, `README.md` | new |
| `ddgclib/tests/test_hand_built_simplex_cache.py` (13), `test_free_surface.py` (21), `test_case_capillary_rise_static.py` | new |
| `METHODS.md`, `debugging_plan.md` | documentation |

Not changed: `_integrators_dynamic.py`, `stress.py`, the dynCA runners
and `src/_setup_dynca.py`, `src/_setup.py` (the former scaffold setup;
nothing imports it any more, the dynCA runners import `_setup_dynca.py`,
`_params.py` and `_dynamic_ca.py`; deleting it is the owner's call since
the other agent owns the neighbouring files), `src/_params.py` (its
`jurin_height` duplicates the library one for the dynCA runners; left).

## 2. The model

Slit of width `2 r` (2D) or round tube of radius `r` (3D), `r = 2 mm`,
water (`rho` 997, `gamma` 0.0728, `theta` 9.99 deg): Jurin's height
3.665 mm (2D) / 7.330 mm (3D), Bond number `(r / l_c)^2 = 0.54` so the
meniscus is not a circular arc and the Young-Laplace solution with
gravity is a real test. The mesh is the liquid in the tube from `y = -D`
up to the meniscus, `D = 3 (2 r) - h_J` (8.3 mm in 2D, 4.7 mm in 3D):
three extruded unit cells (structured square / `cylinder_volume`
cross-section), `ensure_simplex_cache` on the extrusion.

| item | method |
|---|---|
| walls | frozen by membership: `bV` = hull minus the top, `boundary_filter = col.is_wall`; the contact vertices are NOT members |
| reservoir | `HydrostaticReservoirBC`: every step `m = rho(P(y)) V_i` for `y < 0` with `P(y) = K expm1(-rho0 g y / K)` (the IC's own rule); the band supplies / absorbs the mass the column needs |
| contact line | `AxialSlideBC(axis, contact)`: slide along the wall; the lateral force is the wall reaction |
| tension + angle | `FreeSurface`: `F_i = -d/dx_i [gamma A_free - gamma cos(theta) A_wet]` over the boundary facets of `HC._simplices` |
| EOS | `TaitMurnaghan(n = 1, P0 = 0, K = rho (10 sqrt(g h_J))^2)`, `c0` 1.896 (2D) / 2.682 (3D) m/s |
| time step | `0.25 dx_min / c0` (the capillary limit `0.5 sqrt(rho dx^3 / 2 pi gamma)` is 4 to 7 times larger at these sizes) |
| viscosity | `mu = alpha_art rho c0 dx_mean`; runner default and measurements 0.05 (section 5); the fast pins use 0.1 |
| ICs | `'flat'` (2D default): flat meniscus at Jurin's height; `'young_laplace'` (3D default): the column above `y = 0` stretched onto the reference profile |

Why the energy gradient: on a single-phase mesh the free surface has no
second phase to carry a curvature stencil. With `E = gamma A_free - gamma
cos(theta) A_wet`, the force on a 2D surface vertex is `gamma (t_next -
t_prev)` (the integrated curvature normal of `curvature_2d.py`), in 3D
the cotangent mean-curvature normal, and on a contact vertex `gamma
cos(theta)` per unit contact-line length along the wall (Young). The
pressure force of the open dual fan (`p A_open`, the free-surface force
of laneK / laneP) balances it at `p = -gamma kappa`: the discrete
Young-Laplace condition. Verified: the force is the negative energy
gradient to 8e-12 (2D) / 1.2e-11 (3D) by central differences on jittered
builder meshes (`test_free_surface.py`); the interior hydrostatic
residual of the built column is 0.13 m/s^2 (1.3 % of g, the quadratic
term of the profile on 1 mm cells) and 0.10 on the flat start.

The analytical reference: `young_laplace_meniscus(r, gamma, theta, P,
dim)` integrates `dphi/ds = kappa(y) = -P(y) / gamma` (3D: minus
`sin(phi) / r`) from the apex and shoots on the apex height until the
wall angle is `pi/2 - theta`. Checks: residual below 1e-10; the mean
height of the incompressible profile is Jurin's height to 7e-7 (2D) and
8e-7 (3D) (the force balance on the column); a 0.1 mm tube gives the
spherical cap to 3e-4; the compressible profile stands higher by
`rho g h_J / (2 K)` (measured 1.00509 against 1.00500).

In 3D the discrete tube is the polygon inscribed in the circle
(`cube_to_disk`): octagon at refinement 1, 16-gon at refinement 2. The
exact static mean height of THAT model is `gamma cos(theta) (perimeter /
area) / (rho g)`: 1.0824 (octagon) and 1.0206 (16-gon) times the round
tube's, reported as `h_ref_poly` (with the compressibility factor of the
round reference) next to the round `h_ref`.

## 3. Scores

All through `static_errors` (`src/_static.py`), integrated comparisons:

- `h_mean`: the integral of the piecewise-linear free surface over the
  cross-section divided by its area (2D: the polyline; 3D: the triangles
  over the inscribed polygon), against `h_ref` (round) and `h_ref_poly`;
- `shape_rms`: area-weighted RMS distance of the surface from the
  profile (21 samples per edge in 2D, the three edge midpoints per
  triangle in 3D);
- `p_l2`: `ddgclib.analytical.integrated_l2_norm` of the EOS pressure
  against `P(y)` over the mobile vertices above the band;
- `settled_max_a`: max residual acceleration over the mobile vertices
  (lateral components of contact vertices excluded);
- `vol_min_rel`, `edge_min_rel`: the smallest mobile cell / edge.

## 4. Measurements

Every row: preset (or `.replace`) through `run_static`, `cfl = 0.25`.
"t_ac" = `L_dom / c0` = 6.33 ms (2D), 4.47 ms (3D).

### 4.1 2D slit (preset `capillary_rise_static_2D`, `dual_only`)

| refinement | vertices | ic | alpha | t_ac | h_mean [m] | h_mean / h_ref - 1 | shape rms [m] | p_l2 [Pa] | contact / ref | apex / ref | max\|u\| end | max\|a\| end | wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 113 | young_laplace | 0.05 | 300 | 3.694613e-03 | +2.9e-03 | 5.65e-05 | 2.99 | 4.522e-03 / 4.831e-03 | 3.278e-03 / 3.311e-03 | 1.09e-04 | 1.4e-01 (burst, section 5) | 669 s |
| 2 | 113 | young_laplace | 0.1 | 300 | 3.698816e-03 | +4.1e-03 | 5.76e-05 | 2.70 | 4.526e-03 | 3.282e-03 | 1.09e-04 | 1.6e-04 | 11 min |
| 2 | 113 | young_laplace | 0.02 | 300 | 3.693684e-03 | +2.7e-03 | 5.61e-05 | 3.13 | | | 1.39e-04 | 1.1e-01 | 672 s |
| 2 | 113 | flat | 0.05 | 300 | 3.684283e-03 | +1.1e-04 | 5.49e-05 | 0.65 | 4.517e-03 | 3.269e-03 | 2.32e-06 | 6.9e-08 | 487 s |
| 3 | 417 | young_laplace | 0.05 | 300 | BLOW-UP at 177 t_ac (3.6891e-03 = +1.4e-03 at 120 t_ac, max\|u\| 3.6e-04 falling; interior velocities grow from 132 t_ac, 1.4e-02 at 156, two symmetric interior vertices at (1.0, 2.62) / (3.0, 2.62) mm reach 1 m/s at 178 t_ac) | | | | | | | | 97 min |
| 3 | 417 | flat | 0.05 | 300 | 3.683998e-03 | +3.0e-05 | 1.56e-05 | 0.76 | 4.707e-03 | 3.299e-03 | 3.69e-06 | 5.4e-08 | 81 min (3 jobs on 4 cores) |

Reference: `h_ref` 3.683887e-03 (Jurin 3.665237e-03), contact
4.830854e-03, apex 3.311151e-03, `P_cap` 35.85 Pa.

Reading:

- The flat start settles completely at refinement 2 (max|a| 6.9e-08,
  max|u| 2.3e-06 after 300 t_ac) and the volume-averaged height is the
  reference to 1.1e-04. Its surface RMS error (5.5e-05 m = 1.4 % of
  h_J) is the resolution of a 5-vertex polyline: the discrete contact
  point sits at 4.52 mm against 4.83 (the first edge takes the whole
  contact angle, so the discrete meniscus is flatter near the wall and
  deeper at the apex).
- The pre-shaped start reaches the same shape (same contact and apex to
  1e-3) but its mean height creeps: 3.819 (0 to 4 t_ac), 3.7245 (100),
  3.7063 (200), 3.6946e-03 (300 t_ac), the decrement per 20 t_ac
  falling by 0.85 per window, extrapolating to about 3.6945e-03
  (+2.9e-03). The creep is the contact-line vertex sliding against the
  viscous flux to its frozen wall neighbour: twice slower at alpha 0.1
  (3.6988 at 300 t_ac, same extrapolation). The two starts are two
  discrete equilibria of the same mesh family, 3e-03 apart: a plateau of
  the discrete energy along the contact-line position (section 5).
- Refinement 3: the flat start settles by 100 t_ac (max|u| 4e-06, h
  constant to 1e-11 per window) at +3.0e-05; the pre-shaped start creeps
  the same way as at refinement 2 (+1.06e-02 -> +1.4e-03 at 120 t_ac) and
  then the fixed connectivity fails: the slow circulation of the creep
  drives two symmetric interior vertices into a squeezed cell, the EOS
  reads the squeeze as compression and the run blows up at 177 t_ac
  (section 5.2). The flat start is therefore the 2D runner default
  (`CASES`), the pre-shaped start stays an option.

Convergence (flat start, 300 t_ac): refinement 2 to 3: h_mean error
+1.1e-04 to +3.0e-05, shape RMS 5.49e-05 to 1.56e-05, p_l2 0.65 to
0.76 Pa, contact 4.517 to 4.707 (error 0.314 to 0.124 mm). Orders: mean height
1.8, shape 1.8, contact height 1.3; the integrated pressure error does
not fall (the boundary half cells settle on the nodal value, lane P
section 7).

### 4.2 3D round tube (preset `capillary_rise_static_3D`, `dual_only_bare`, `p_ij_simplex`)

| refinement | vertices (contact) | ic | alpha | t_ac | h_mean [m] | / h_ref_poly - 1 | / h_ref - 1 | shape rms | p_l2 [Pa] | max\|u\| end | max\|a\| end | wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (octagon) | 87 (8) | young_laplace | 0.05 | 100 | 7.887545e-03 | -1.09e-02 | +7.06e-02 | 5.90e-04 | 1.72 | 1.70e-04 | 7.4e-04 | 476 s |
| 1 | 87 | flat | 0.05 | 100 | 7.237770e-03 | -9.24e-02 | -1.76e-02 | 2.26e-04 | 19.4 | 1.42e-04 | 7.6e-04 | 287 s |
| 1 | 87 | flat | 0.05 | 300 | 7.229637e-03 | -9.34e-02 | -1.87e-02 | 2.25e-04 | 18.9 | 1.18e-06 | 6.4e-06 | 15 min |
| 2 (16-gon) | 517 (16) | young_laplace | 0.05 | 40 | 7.541105e-03 | +2.00e-03 | +2.36e-02 | 2.03e-04 | 3.79 | 1.22e-03 | 3.3e-03 | about 80 min (3 jobs) |

References: round `h_ref` 7.367491e-03 (Jurin 7.330474e-03), contact
8.326160e-03, apex 6.778111e-03; polygon `h_ref_poly` 7.974514e-03
(octagon), 7.526036e-03 (16-gon); `P_cap` 71.70 Pa.

Reading: at refinement 1 the tube is an octagon with 8 contact vertices
and one apex vertex; the pre-shaped start is 1.1 % below the force
balance of the octagon (contact 8.316 against 8.326 mm, apex 6.814
against 6.778), the flat start settles (max|a| 6.4e-06) into a second
discrete equilibrium 9 % lower with a 19 Pa pressure error: at this
resolution the discrete energy has two minima and the flat start finds
the wrong one, so the 3D runner default is the pre-shaped start.
Refinement 2 (16-gon, 16 contact vertices, 25 surface vertices) after
40 t_ac, still settling (max|u| 1.2e-03 falling): +2.0e-03 against the
force balance of the 16-gon (+2.4e-02 against the round tube, of which
2.1e-02 is the polygon), surface RMS 2.0e-04 against 5.9e-04 at
refinement 1 (ratio 2.9), contact 8.243 against 8.326 mm, apex 6.837
against 6.778. The 3D column follows the same balance as the 2D one once
the discrete cross-section is accounted for; a converged 3D number needs
the 100 t_ac run on the owner's hardware (section 11).

### 4.3 Pins (`test_case_capillary_rise_static.py`)

| pin | configuration | value |
|---|---|---|
| `PIN_2D_H_10`, `PIN_2D_UMAX_10`, `PIN_2D_UMAX_PEAK` | 2D refinement 2, young_laplace, alpha 0.1, 10 t_ac (fast) | 3.80100337536971e-03 / 1.5325773786546008e-03 / 1.683708352699222e-02 (bit-identical in a second process) |
| `PIN_2D_H_100`, `PIN_2D_UMAX_100` | 2D refinement 2, young_laplace, alpha 0.05, 100 t_ac (slow) | 3.7063081015925948e-03 / 4.958823503840141e-04 (h +6.1e-03, shape 6.0e-05, p_l2 2.50, mass -8.3e-03) |
| `PIN_3D_H_4`, `PIN_3D_UMAX_4` | 3D refinement 1, young_laplace, alpha 0.1, 4 t_ac (fast) | 7.81529116541829e-03 / 1.228862275931968e-03 |
| `PIN_3D_H_40`, `PIN_3D_UMAX_40` | 3D refinement 1, young_laplace, alpha 0.05, 40 t_ac (slow) | 7.86039962372492e-03 / 5.474210261802552e-04 (-1.43e-02 against the octagon reference) |

End-of-run values (rule 8); fixed connectivity, no reconnection, so one
process suffices. The shipped runs into the case directory
(`results/capillary_rise_static_2D_flat`, 100 t_ac, alpha 0.05: h_mean
3.684247e-03 = +9.8e-05, max|u| 2.5e-06, max|a| 2.1e-05, shape RMS
5.49e-05, p_l2 0.65 Pa; `results/capillary_rise_static_3D_young_laplace`,
40 t_ac: 7.860400e-03 = -1.43e-02 against the octagon, +6.7e-02 against
the round tube, max|u| 5.5e-04) carry `methods.json`, `summary.json`,
`series.npz` and the figures; results and figures are git-ignored except
`series.npz` (two untracked files the owner may keep or delete). The methods block of each is the preset's
`to_dict()` (`results/<tag>/methods.json` of the shipped runs).

## 5. Connectivity, viscosity, and what the static case says about the lane P items

### 5.1 The reconnecting arm

`remap_arm(preset)` = `.replace(connectivity='delaunay_material',
remap='conservative', redistribute_mass=True)`, 2D refinement 2,
young_laplace, alpha 0.05, 300 t_ac: max|u| 1.87e-02 -> 4.6e-06 (settled
by 40 t_ac, no creep; flips give bursts of 5e-04 every 80 t_ac),
h_mean 3.872564e-03 (+5.1e-02), contact 4.790e-03, apex 3.409e-03, shape
RMS 2.03e-04, p_l2 3.20 Pa, injected mass +5.2e-03 of the mesh. The
reconnecting column stands 5 % too high: the band reset and the remap's
mass rescale (every rebuild re-targets every vertex to the snapshot
pressure, then rescales the total) do not commute, so the mass budget of
the column is not the force balance. Measured worse for this case;
`dual_only` is the preset. Not pursued further here.

### 5.2 Viscosity and the free-surface items of lane P

- alpha 0.05 is enough: both ICs settle, the flat start to round-off.
  alpha 0.02 shows the free-surface flutter of lane P (max|u| envelope
  8.7e-04 over the last 4 t_ac against an end value 1.4e-04, max|a| 0.11),
  alpha 0.1 doubles the creep time. The runner default (`CASES` in
  `src/_static.py`) and the measurements use 0.05; lane P's 0.1 for the
  column is kept only by the two fast pins, which pass it explicitly.
- The saddle / slow drift of lane P is visible here as the slow
  circulation that the creeping meniscus drives (centre up, walls down,
  1e-03 m/s at 20 t_ac, 5e-04 at 100): on the fixed connectivity it
  carries the interior vertex under the apex upward until the cell is
  squeezed (alpha 0.05, pre-shaped start: the vertex 9 um under the apex
  at 250 t_ac, `vol_min_rel` falling), and the squeezed cell rings
  (bursts of max|u| up to 1.5e-02 at 246, 254, 262, 269 t_ac that die
  within one window). The flat start does not show it in 300 t_ac. A
  fixed connectivity cannot repair this; the reconnecting arm does not
  have it but carries the 5 % bias. The artificial viscosity does not
  remove the drift, it only slows it (lane P's finding, confirmed).
- The free-surface force balance holds: the discrete Young-Laplace
  state is reached with max|a| 6.9e-08 and the integrated force balance
  of the column (the mean height) to 1.1e-04. The open-fan pressure
  force and the energy-gradient tension are consistent at the
  equilibrium; the linearised non-symmetry of lane P (open-fan rows)
  only shows as the flutter at alpha 0.02.

### 5.3 What the dynCA work would need from the library next

(Read-only look at `capillary_rise_2D_dynCA.py` / `src/_setup_dynca.py`.)
The dynCA runner hand-rolls what this lane registered: a polyline line
tension (`F_surf`) -> `FreeSurface`; `band_mass_reset` ->
`HydrostaticReservoirBC`; the wall clamp and the slaved contact
vertices -> `AxialSlideBC` plus a prescribed contact-line velocity.
What is still missing in the library for the DYNAMIC rise: (a) a
dynamic contact angle (`theta(Ca)`) as a `contact_line` option that
replaces `cos(theta)` per contact vertex by a function of its sliding
speed; (b) contact-line slip: the frozen wall vertex next to the
contact vertex applies the full two-point viscous flux, which is the
creep of section 4 and the "contact-line attachment" of the dynCA
README; a Navier-slip wall flux (`viscous_flux` option or a BC that
scales the wall flux) is the library piece; (c) wall vertex
creation / removal as the contact line travels more than a cell
(`adaptive` connectivity with `frozen_set='membership'`, the lane L
open item), since the static case only needs the contact vertex to
slide within its cell.

## 6. Hand-built 3D setups (task 1)

`ensure_simplex_cache` applied; tiling measured by
`test_hand_built_simplex_cache.py`:

| setup | before (fan walk / fallback) | after | exact |
|---|---|---|---|
| `Complex(3).triangulate().refine_all()` unit cube | 0.9166666666666667 | 1.0 | 1 |
| `setup_cube_flow(dim=3, n_refine=1, L=2)` (mesh and inlet unit mesh) | fan walk | 8.0 | 8 |
| `setup_cube_to_droplet(dim=3, n_refine=1)` | fan walk | 0.06^3 | exact |
| `unit_cylinder(0.5, 1, 2.0)`, `cube_to_tube` | fan walk | equal to the enclosed polyhedron to 1e-12 | |
| `setup_hydrostatic(dim=3, n_refine=1)` | fan walk | 1.0 | 1 |
| 2D: `setup_cube_flow`, `setup_cube_to_droplet`, `setup_hydrostatic`, `setup_poiseuille_2d`, `_lagrangian` | repaired fallback (exact since laneS) | exact, now from the cache | |

No pinned number moved: `test_case_hydrostatic.py`,
`test_case_hagen_poiseuille.py`, `test_frozen_set.py`, `test_stress.py`,
`test_builder_simplex_cache.py` pass unchanged (168 tests). The
`liquid_bridge_cfd_dem` film is a surface mesh (no volume cache); the
lane X runners (`bc_demo`, `liquid_bridge_*`, `oscillating_droplet_p_ref`
scripts, `dynamic_caprise_tube`) build their own complexes and are left to
lane X.

## 7. Pin safety

- `HydrostaticEOSMass` closed form: for `P_ref = P0` the new factor
  `(K + P_ref - P0)` is `K` exactly, the same floating-point expression:
  `test_case_hydrostatic.py` (all pins), the hydrostatic presets and
  `test_integrated_validation.py` are unchanged. The fast suite:
  section 8.
- `SolverMethods` has a new field `contact_line` (default `None`): old
  `methods.json` blocks read as the default (`diff_baselines`, laneL
  rule), every preset round-trips (`test_methods.py`).
- `DomainResult.__post_init__` calls the helper with the same two
  branches: every builder cache is the same object graph as before.

## 8. Tests

- fast suite (`pytest ddgclib/tests -q -m "not slow"`): 1289 passed, 1
  failed, 11 skipped, 2 xfailed (4.5 min). The failure is the
  environment baseline (a), `test_single_phase_remap.py::TestBoxDecayWithEOS
  ::test_remap_matches_fixed_connectivity`, with the same two values
  (0.4399043770574842 against 0.43990437705748414). Baseline 1237 + 3
  (sympy) + 13 + 21 + 5 new + the 10 hand-built / case tests moved.
- slow battery (`-m slow`): 38 passed, 1 failed, 1 xfailed (38 min with
  three jobs on four cores). The failure is baseline (b),
  `test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column`,
  with the same value (0.15269571220608844 against the pin
  0.15305813130485327). Baseline 33 + 3 (sympy) + 2 new.
- hyperct (not touched): 340 passed, 38 skipped, 6 xfailed (baseline (c)
  was fixed by lane G).
- New tests fail on the pre-lane code: `test_hand_built_simplex_cache.py`
  (no helper), `test_free_surface.py` (no module / axis),
  `test_case_capillary_rise_static.py` (no preset).

## 9. DO-NOTs (measured)

- Do not build a membership set of vertices before the last `move_all`
  of the setup: a vertex hashes by its coordinates, the set loses it
  (the first pre-shaped run had 6 unfrozen wall vertices with open cells
  and collapsed: h_mean 3.81 -> 1.84 mm in 5 t_ac, max|u| 1.3 m/s).
- Do not take the reconnecting arm for this case: +5.1e-02 on the mean
  height (section 5.1).
- Do not read the round-tube reference against a refinement 1 tube: the
  octagon's force balance is 8.2 % higher; use `h_ref_poly`.
- Do not use the old `HydrostaticEOSMass` closed form with `P_ref != P0`:
  it is low by `-P_ref expm1(alpha depth)` below the reference level,
  at the bottom of the band 1.19 Pa for the flat start (`P_ref` =
  `P(h_J)` = -35.85 Pa, depth 12.0 mm) and 1.74 Pa for the pre-shaped
  start (`P_ref` = `P(4.83 mm)`, the first runs: 5 % of `P_cap`), and
  the column stood 1.7 % too high.
- Do not run below alpha_art 0.05 on this free surface (flutter at 0.02).

## 10. Known limits

- The contact-line creep: the discrete equilibrium reached from the
  pre-shaped start differs from the flat one by 3e-03 in the mean
  height at refinement 2 (a plateau of the discrete energy along the
  contact position); the flat start is the measurement of record.
- The fixed connectivity cannot repair the slow squeeze of the cell
  under the apex (section 5.2).
- 3D refinement 2 was run for 40 t_ac only (about 80 min with three
  jobs on four cores; the owner's command for 100 t_ac is in section
  11); refinement 3 in 3D (about 4000 vertices) was not run.
- 2D refinement 3 from the pre-shaped start blows up at 177 t_ac on the
  fixed connectivity (section 4.1); the shipped 2D default is the flat
  start, which settles. A reconnecting arm without the 5 % bias (a band
  reset that commutes with the remap) would remove the limit.
- `FreeSurface` holds its vertex sets by `id`: a retopology that creates
  or deletes vertices (`adaptive`, inlets) is not covered.
- The shipped 3D horizon (40 t_ac, refinement 1) shows the settling of
  the pre-shaped start, not a converged height (section 4.2).

## 11. Reproduce

```bash
cd /home/user/ddgclib
PY="env PYTHONPATH=/home/user/ddgclib /usr/bin/python"
$PY -m pytest ddgclib/tests/test_free_surface.py ddgclib/tests/test_hand_built_simplex_cache.py ddgclib/tests/test_case_capillary_rise_static.py -q -p no:cacheprovider
$PY cases_dynamic/capillary_rise/capillary_rise_2D.py --out /tmp/laneI/2d               # shipped: flat, 100 t_ac, alpha 0.05
$PY cases_dynamic/capillary_rise/capillary_rise_3D.py --out /tmp/laneI/3d               # shipped: young_laplace, 40 t_ac
# measurements of section 4
$PY cases_dynamic/capillary_rise/capillary_rise_2D.py --n-tac 300 --no-anim --out /tmp/laneI/r2flat
$PY cases_dynamic/capillary_rise/capillary_rise_2D.py --n-refine 3 --n-tac 300 --no-anim --out /tmp/laneI/r3flat
$PY cases_dynamic/capillary_rise/capillary_rise_2D.py --n-refine 3 --ic young_laplace --n-tac 300 --no-anim --out /tmp/laneI/r3yl   # blows up at 177 t_ac
$PY cases_dynamic/capillary_rise/capillary_rise_2D.py --arm remap --ic young_laplace --n-tac 300 --no-anim --out /tmp/laneI/remap
$PY cases_dynamic/capillary_rise/diagnose_static_rise.py convergence --dim 3 --refinements 1 --n-tac 100 --alpha-art 0.05 --out /tmp/laneI/c3
$PY cases_dynamic/capillary_rise/capillary_rise_3D.py --n-refine 2 --alpha-art 0.05 --n-tac 100 --no-anim --out /tmp/laneI/r2_3d   # owner's hardware, about 2 h
$PY cases_dynamic/capillary_rise/diagnose_static_rise.py series --series /tmp/laneI/r2flat/results/capillary_rise_static_2D_flat --t-ac 6.3284e-03 --out /tmp/laneI
# the refinement 2 rows of section 4.1 were run with --ic young_laplace / --alpha-art 0.1 and 0.02 as listed; the 2D fast pin uses alpha 0.1
```
