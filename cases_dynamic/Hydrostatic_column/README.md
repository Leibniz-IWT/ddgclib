# Hydrostatic column

A column of weakly compressible fluid under gravity with a free surface on
top: the simplest dynamic test of the pressure force, the EOS coupling and
the free surface. Since lane P (2026-10-01) the four runners have no time
loop of their own. Each one runs a preset of `ddgclib.methods.PRESETS` on
the library `symplectic_euler` integrator, with gravity as the `body_force`
of `SolverMethods.dudt_fn`. The shared code is `src/_column.py`.

| runner | preset | connectivity | geometry |
|---|---|---|---|
| `Hydrostatic_1D.py` | `hydrostatic_1D` | `delaunay` (in 1D: the sorted chain) | 10 m, 33 vertices, frozen bottom vertex |
| `Hydrostatic_2D.py` | `hydrostatic_2D` | `dual_only` | unit square, 145 vertices, no-slip bottom and sides |
| `Hydrostatic_2D_periodic.py` | `hydrostatic_2D_periodic` | `dual_only` | unit square, free-slip sides (`FreeSlipWallBC`), frozen bottom |
| `Hydrostatic_3D.py` | `hydrostatic_3D` | `dual_only_bare` | unit cube, 189 vertices, no-slip bottom and sides |

Common physics: gauge pressure, linear Tait EOS with `c0 = 10 sqrt(g H)`
(1 % compression at the bottom), `rho0 = 1000`, `g = 9.81`, CFL 0.25 on the
sound speed, viscosity `mu = alpha_art rho c0 dx` with `alpha_art = 0.5`.

## Run

From the repository root, in the `ddg` environment:

```bash
python cases_dynamic/Hydrostatic_column/Hydrostatic_1D.py            # 200 t_ac, about 1.5 min
python cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py            # 100 t_ac, about 2.5 min
python cases_dynamic/Hydrostatic_column/Hydrostatic_2D_periodic.py   # 100 t_ac, about 2.5 min
python cases_dynamic/Hydrostatic_column/Hydrostatic_3D.py            #  40 t_ac, about 8 min
```

`t_ac = H / c0` is one acoustic traversal of the column. Options (all four
runners):

| option | meaning |
|---|---|
| `--ic drop` (default) | uniform density at t = 0: the column falls, compresses and rings until viscosity has damped it |
| `--ic equilibrium` | masses of the compressible hydrostatic profile: starts within the discretisation error of the discrete equilibrium |
| `--arm remap` | the reconnecting arm: `preset.replace(connectivity='delaunay_material', remap='conservative', redistribute_mass=True)` (2D and 3D) |
| `--n-refine N`, `--n-tac T`, `--alpha-art A` | refinement, horizon in acoustic times, artificial viscosity factor |
| `--no-anim` | skip the mp4 |

The runners are headless (matplotlib `Agg`).

## Outputs

- `results/<preset>[_remap]/snapshots/`: `StateHistory` snapshots (`u`, `p`)
- `results/<preset>[_remap]/methods.json`: the `SolverMethods` that ran, the
  implicit choices resolved on the mesh, both git SHAs, run parameters
- `results/<preset>[_remap]/summary.json`: the numbers printed at the end
- `fig/<preset>[_remap]_settling.png`, `_final_profile.png`,
  `_mesh_pressure.png` (2D, 3D), `<preset>[_remap].mp4`

Interactive replay of the snapshots:

```bash
python -m ddgclib.scripts.view_polyscope --snapshots cases_dynamic/Hydrostatic_column/results/hydrostatic_3D/snapshots
```

## What the shipped runs give

Reference: `P(y) = K (exp(rho0 g (h - y) / K) - 1)` with `h` the surface
height that conserves the mass (0.995033 H for the uniform-density start).
Comparisons are the integrated ones of `ddgclib.analytical`.

| runner (default run) | max\|u\| peak | max\|u\| end | settled max\|a\| | integrated L2 | L2 / rho g H |
|---|---|---|---|---|---|
| 1D, 200 t_ac | 0.8897 | 1.44e-02 (last sample; envelope of the last 4 t_ac 1.73e-02) | 1.30e-01 | 554.4 Pa | 5.7e-03 |
| 2D, 100 t_ac | 0.2482 | 3.01e-04 | 1.01e-04 | 50.19 Pa | 5.1e-03 |
| 2D free-slip, 100 t_ac | 0.2561 | 2.59e-04 | 1.88e-02 | 50.55 Pa | 5.2e-03 |
| 3D, 40 t_ac | 0.1439 | 2.86e-04 | 2.68e-04 | 174.4 Pa | 1.8e-02 |
| 2D `--arm remap`, 100 t_ac | 0.2541 | 3.47e-04 | 1.23e-04 | 57.20 Pa | 5.8e-03 |

The 1D and free-slip columns are still ringing at the end of the default
run: their motion is one-dimensional, so the only damping is the viscous
decay of the fundamental acoustic mode (kinetic energy rate `nu k^2` with
`k = pi / (2 H)`; measured 0.0389 and 0.126 per `t_ac`, theory 0.0386 and
0.125). With `--ic equilibrium` the same presets stay at rest: over 200
`t_ac` max|u| goes 2.8e-06 -> 7.0e-08 (1D), 7.0e-06 -> 4.2e-09 (2D),
6.6e-06 -> 3.4e-11 (free-slip), and over 100 `t_ac` 3.2e-05 -> 5.4e-07 (3D).

Convergence of the settled column (`--ic equilibrium`, 60 `t_ac`,
`diagnose_column.py convergence`):

| case | refinement | integrated L2 [Pa] | interior cells [Pa] |
|---|---|---|---|
| 1D | 3, 4, 5, 6 | 1.04, 0.220, 0.0448, 0.0094 | same |
| 2D | 2, 3, 4 | 136.1, 48.57, 17.20 | 0.895, 0.226, 0.0566 |
| 2D free-slip | 2, 3, 4 | 144.1, 49.77, 17.40 | 0.896, 0.222, 0.0507 |
| 3D | 1, 2 | 2.154, 0.977 | 2.03, 0.979 |

The 2D error drops by 2^1.5 per refinement and sits in the boundary cells:
the centred pressure flux is exact for nodal values of a linear field, so
the pressure of a boundary half cell settles on the nodal value, which
differs from the cell average by O(dx). The interior error is second
order. In 1D and 3D `ddgclib.analytical` compares boundary cells (1D) or
all cells (3D) with the point value times the volume, so that offset does
not appear there.

## Things to know before changing this case

- **Artificial viscosity is required, for three reasons.** It damps the
  acoustic ringing of the uniform-density start (above). The discrete
  hydrostatic equilibrium is a saddle of the discrete energy: its slowest
  modes grow at 1.13 g / c0 (0.353 1/s, e-fold 89 `t_ac`), also in a closed
  box without any free surface; viscosity turns that into creep (1.2e-04
  1/s at `alpha_art = 0.5`, 1.2e-03 at 0.05). And the free-surface force is
  not an energy gradient, which without viscosity gives a flutter
  instability on some meshes (no-slip column: 0.75 1/s at refinement 2,
  none at the shipped refinement 3, 1.22 1/s at refinement 4); the
  artificial viscosity removes it. With the viscosity of water the no-slip
  2D uniform-density run exceeds c0 at 64 `t_ac`.
  `diagnose_column.py free_surface`, `growth` and `viscosity` reproduce
  this.
- **`Hydrostatic_2D_periodic` is not periodic.** The library periodic
  connectivity cannot carry a single-phase EOS column: after one
  `retopologize_periodic` the total dual volume of the unit square is 2.488
  and 45 of 136 vertices fail dual-face closure
  (`diagnose_column.py periodic`). The runner uses free-slip side walls,
  which is what its former case-local code did.
- **Do not run a free surface on `connectivity='delaunay'`**, with or
  without the remap: Delaunay fills the gap between the surface and the
  convex hull (42 m/s at 3 `t_ac`, half the hydrostatic head). Use
  `delaunay_material`.
- **`delaunay_material` keeps the domain exactly in 2D, not in 3D.** The
  rebuild is not constrained to the old surface facets. In 2D the domain
  volume changes by round-off only. In 3D a free-surface square whose
  Delaunay diagonal differs from the old one gains or loses the sliver
  between the two triangulations: at most 1.2e-06 of the volume per call on
  this column (1.6e-06 summed over 185 calls), 9.8e-05 when a 0.004 bowl is
  pushed into the builder surface in one go. The function returns the
  relative change and warns above `domain_tol = 1e-3`
  (`diagnose_column.py remap`). The 3D `--arm remap` run from uniform
  density is bit-identical in every process since lane T (2026-10-02;
  before, to about two digits after 10 `t_ac`). It is still decided by
  ties (cospherical mesh, free-surface edge areas): a 1e-15 shift of the
  interior vertices moves its peak by 1.25e-03 relative
  (`cases_dynamic/diagnose_determinism.py sweep pin_hydro3d_remap
  --perturb 1e-15`).
- **Do not use `dual_only` for the 3D column**: its 3D branch zeroes the
  dual volume of frozen vertices, so wall cells read the reference
  pressure.
- **Pressures are nodal, not cell averages, on boundary cells**: an initial
  pressure set to the dual-cell average gives a static residual of 1.9
  m/s^2 in 2D at every refinement; nodal values give round-off.

## Diagnose

```bash
python cases_dynamic/Hydrostatic_column/diagnose_column.py arms          # every case, both arms and the measured-bad ones, 40 t_ac
python cases_dynamic/Hydrostatic_column/diagnose_column.py convergence   # error against refinement
python cases_dynamic/Hydrostatic_column/diagnose_column.py viscosity     # how much viscosity, and what happens without
python cases_dynamic/Hydrostatic_column/diagnose_column.py free_surface  # linearised column, fastest mode in a real run, sloshing seed
python cases_dynamic/Hydrostatic_column/diagnose_column.py growth        # saddle and flutter growth rates at refinement 2, 3, 4
python cases_dynamic/Hydrostatic_column/diagnose_column.py periodic      # the library periodic path on this column
python cases_dynamic/Hydrostatic_column/diagnose_column.py remap         # convex hull against material rebuild, with and without the mass rescale; domain volume kept
```

Tests: `pytest ddgclib/tests/test_case_hydrostatic.py ddgclib/tests/test_material_delaunay.py`
(add `-m slow` for the 40 `t_ac` pins). Measurements and method notes:
`docs_temp/debug_session/laneP-hydrostatic-library-integrators.md`.
