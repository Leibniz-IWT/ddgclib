# Hagen-Poiseuille 2D (Lagrangian developing channel flow)

Plug flow `U_avg` enters a channel `[0, L] x [0, D]` with no-slip walls at
`y = 0` and `y = D` and develops under the pressure field `P = G (L - x)`
towards the Poiseuille profile `u_x(y) = G / (2 mu) y (D - y)`, with
`G = 12 mu U_avg / D^2`, so that `U_max = 1.5 U_avg`. The mesh moves with
the fluid.

The runner goes through the preset `hagen_poiseuille_2D` of
`ddgclib.methods` (symplectic Euler, per-step Delaunay,
`frozen_set='membership'`, `viscous_flux='simplex_gradient'`, serial) on
the library integrator. There is no time loop and no retopology code in
this directory.

| file | content |
|---|---|
| `Hagen_Poiseuile_2D.py` | runner: `run_case('hagen_poiseuille_2D')` |
| `src/_run.py` | runner body shared with the 3D pipe (`run_developing`, `run_case`) and the shipped parameters (`CASES`) |
| `src/_setup.py` | `setup_poiseuille_developing`: mesh, BCs, ICs (2D and 3D) |
| `src/_metrics.py` | profile error, mass flux through planes, vertex census |
| `src/_params.py` | parameters of the older notebooks (pipe formulas); not used by the runner |
| `diagnose_poiseuille.py` | A/B arms, static consistency, convergence, sliver census |
| `diagnose_frozen_set.py` | lane L: `hull` against `membership` on the configuration before lane H |
| `visualize_hp2d.py` | extra figures and animations from the saved run |

## What is solved

There is no pressure solve. The pressure is a function of position,
re-imposed on every vertex after every step (`DirichletPressureBC` over
`HC.V`, nodal values). Each fluid vertex therefore relaxes to the developed
profile along its path line, with the time constant
`t_dev = rho D^2 / (pi^2 mu)` of the slowest viscous mode; the development
length is about `5 t_dev U_max`. Continuity is not enforced: the vertices
stay in their rows (no transverse force), the rows move at different
speeds, and the cell density `m_i / Vol_i` follows.

Boundary conditions, in this order:

1. `OutletBufferedDeleteBC`: vertices that cross `x = L` keep their
   velocity for one buffer width and are removed.
2. `PeriodicInletBufferedBC`: the zone `[-inlet_buffer, 0]` of the mesh is
   a buffer of prescribed plug motion, fed at its upstream end by a ghost
   copy of the unit cell; a vertex is released at `x = 0`. A wall row of
   the ghost is injected once: the wall BC freezes the vertex where it
   entered (at most `U_avg dt` from the upstream corner) and the row is
   then dropped from the ghost, so `--dt` need not divide the column
   spacing (one extra wall vertex per wall, 106 -> 108 frozen in the
   shipped run).
3. `PositionalNoSlipWallBC`: walls, frozen by membership.
4. `DirichletPressureBC`: the prescribed pressure.

Every hull vertex that is not a wall is a buffer vertex, so no integrated
vertex has an open dual cell.

## Run

In the `ddg` environment, from anywhere:

```bash
python cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --headless          # shipped run, about 4 min
python cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --L 4 --mu 0.1 --dt 0.02 --steps 500 --n-refine 1 --no-anim --tag quick   # 4 s
python cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py --viscous-flux two_point --tag two_point --no-anim
python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py static
cd cases_dynamic/Hagen_Poiseuile && PYTHONPATH=../.. python visualize_hp2d.py
```

| option | meaning |
|---|---|
| `--L`, `--mu`, `--n-refine`, `--dt`, `--t-end`, `--steps` | case parameters (defaults: `CASES` in `src/_run.py`) |
| `--viscous-flux`, `--pressure-flux`, `--frozen-set`, `--backend`, `--workers` | A/B arm: the preset with that axis replaced (`--backend` is only read by the 3D retopology; in 2D it changes nothing) |
| `--tag NAME` | write to `results/hagen_poiseuille_2D_NAME/` |
| `--headless` | accepted for scripted runs (nothing blocks) |
| `--no-anim` | skip the mp4 |

## Outputs

- `results/hagen_poiseuille_2D[_tag]/summary.json`: profile error, fluxes,
  vertex counts, walls
- `results/hagen_poiseuille_2D[_tag]/methods.json`: the `SolverMethods` that
  ran and the implicit choices resolved on the mesh
- `results/hagen_poiseuille_2D[_tag]/snapshots/`, `hp2d_final_state.json`,
  `hp2d_history.pkl`
- `fig/hagen_poiseuille_2D[_tag]_profile.png`, `_development.png`,
  `_mesh.png`, `.mp4`
- `results/laneH/*.json`: tables of `diagnose_poiseuille.py`

## Result (lane H, 2026-10-01; re-run unchanged after the fix round of 2026-10-02)

Shipped run: `Re_D = 10` (`mu = 0.01`, `t_dev = 10.1 s`), `L = 12`,
refinement 2 (473 vertices), `dt = 0.05`, 2400 steps to `t = 120 s`
(11.8 `t_dev`). The comparison is the dual-volume weighted l2 norm of
`u - u_analytical` over the free vertices of `6 <= x <= 12`, relative to
the norm of the analytical profile.

| quantity | value |
|---|---|
| profile error l2 at the end / mean and maximum over the last quarter | 1.086e-02 / 1.089e-02, 1.167e-02 |
| `u_max` in the window (analytical 0.15) | 0.15085 |
| largest transverse velocity in the window, any sample (over all free vertices at every step: 1.1e-16) | 5.9e-17 |
| mass flux through `x = 0` / `x = L` over the last 6 inlet periods, in `rho U_avg D` | 0.8333 / 0.8681 |
| share of the mass that is not in wall cells (the plug flux of the mesh) | 0.8333 |
| volume flux in the window, in `U_avg D` (nodal quadrature of the parabola: 0.9688) | 0.9644 |
| vertices: start / end; free vertices in the channel: min, max | 473 / 478; 311, 339 |
| vertices outside the walls, any time | 0 |
| walls: frozen / moved | 106 of 106 / 0 |

What is left is the resolution: 7 fluid rows across the channel. On the
short channel (`L = 4`, `mu = 0.1`; `results/laneH/convergence.json`) the
error falls by about 3 per refinement: 1.1e-02, 3.6e-03, 1.3e-03 at
refinement 1, 2, 3 while the rows are still the regular initial pattern
(`t = 10 s`), and 3.3e-02, 1.1e-02, 3.4e-03 once they are sheared
(`t = 60 s`).

The same run with `--viscous-flux two_point` does not reach the profile:
l2 0.26 on the short channel, and on the shipped one the transverse
velocity grows from round-off (l2 1.0, 26 vertices outside the walls). The
two-point flux is not linearly precise on a sheared mesh (residual of a
linear velocity field 0.64 to 1.4 of `G Vol`, growing with refinement;
`diagnose_poiseuille.py static`).

The outlet mass flux reaches the inlet value only when the slowest row has
crossed the channel (`L / u(y_1)`, 183 s here); until then the near-wall
rows still carry their initial spacing.

## Lane L reproducer

`diagnose_frozen_set.py hp2d` and
`ddgclib/tests/test_frozen_set.py::TestHagenPoiseuille2D` keep the
configuration before lane H (`setup_poiseuille_2d_lagrangian`: hull inlet,
pressure advected with the vertices, two-point viscous flux) as the
reproducer of the wall collapse under `frozen_set='hull'`. Evidence:
`docs_temp/debug_session/laneL-frozen-set-membership.md`.

`test_outlet_old_bc.py` and `test_outlet_new_bc.py` are older stand-alone
outlet probes on that configuration.

Evidence for this page: `docs_temp/debug_session/laneH-poiseuille-developing.md`.
