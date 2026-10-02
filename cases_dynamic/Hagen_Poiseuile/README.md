# Hagen-Poiseuille 2D (Lagrangian developing channel flow)

Plug flow enters a channel `[0, L] x [0, D]` with no-slip walls at `y = 0`
and `y = D` and develops under a constant pressure gradient. The mesh moves
with the fluid: a ghost copy of the unit cell injects vertices at the inlet
(`PeriodicInletBC`), a buffer zone behind the outlet removes them
(`OutletBufferedDeleteBC`), the walls are held by
`PositionalNoSlipWallBC`.

The runner uses the preset `hagen_poiseuille_2D` of `ddgclib.methods`
(symplectic Euler, per-step Delaunay, `frozen_set='membership'`, 20 dudt
workers). Mesh, BCs and ICs are built by
`src/_setup.py:setup_poiseuille_2d_lagrangian`.

## Run

From this directory, in the `ddg` environment:

```bash
PYTHONPATH=../.. python Hagen_Poiseuile_2D.py                 # 3000 steps, opens two mesh plots
PYTHONPATH=../.. python Hagen_Poiseuile_2D.py --headless      # no plot windows, about 4 min
PYTHONPATH=../.. python Hagen_Poiseuile_2D.py --headless --steps 1400 --workers 1
PYTHONPATH=../.. python Hagen_Poiseuile_2D.py --headless --frozen-set hull --tag hull
PYTHONPATH=../.. python visualize_hp2d.py                     # figures and animations
```

`PYTHONPATH=../..` puts the repository root first, so that the live
`hyperct` tree (symlink in the root) is imported and not an installed wheel.
`Hagen_Poiseuile_2D.py` and `diagnose_frozen_set.py` insert the root
themselves and also run without it; `visualize_hp2d.py` needs the variable.

| option | meaning |
|---|---|
| `--steps N` | number of time steps (`dt = 0.01`, default 3000) |
| `--headless` | skip the blocking `HC.plot_complex()` calls |
| `--frozen-set hull\|membership` | A/B arm: the preset with `frozen_set` replaced |
| `--workers N` | dudt worker processes (default: the preset's 20; 1 = serial) |
| `--tag NAME` | write to `results/NAME/` instead of `results/` |

## Outputs

- `results/[tag/]methods.json`: the `SolverMethods` that ran and the implicit
  choices resolved on the mesh
- `results/[tag/]wall_report.json`: the wall vertices at the start and what
  became of them (still frozen, moved, largest displacement, first step the
  number of vertices on the wall lines drops)
- `results/[tag/]hp2d_final_state.json`, `hp2d_history.pkl`,
  `state_<step>_t<time>.json` every 500 steps
- `fig/`: written by `visualize_hp2d.py`

## Status (lane L, 2026-10-01)

- The walls hold. Shipped run: 62 wall vertices, 62 still frozen, 0 moved.
  With `--frozen-set hull` (the rule before lane L) the walls are released
  at step 1262, when two outlet buffer vertices drift past the wall lines:
  2 of 62 still frozen, 60 moved.
- Not validated against the Poiseuille profile: `U_max` 0.29 at `x = L / 2`
  against the analytical 0.20 at `t = 30`.
- Nothing keeps a fluid vertex inside the channel: at `t = 30` one vertex is
  5.1e-3 above the top wall.
- The first inlet column leaves one extra wall vertex per wall one advection
  step from the inlet corner, and the seam column of the unit cell is
  injected twice per period.

`diagnose_frozen_set.py` runs the `hull` against `membership` comparison on
a short channel and on the dam break, electrolysis and droplet setups (see
its docstring). Evidence:
`docs_temp/debug_session/laneL-frozen-set-membership.md`.

`test_outlet_old_bc.py` and `test_outlet_new_bc.py` are older stand-alone
outlet probes on the default (hull) rule.
