# Hagen-Poiseuille 3D (Lagrangian developing pipe flow)

Plug flow `U_avg` enters a pipe of radius `R` along z and develops under
the pressure field `P = G (L - z)` towards
`u_z(r) = U_max (1 - (r / R)^2)`, with `G = 8 mu U_avg / R^2`, so that
`U_max = 2 U_avg`. The mesh moves with the fluid.

The runner goes through the preset `hagen_poiseuille_3D` of
`ddgclib.methods` (symplectic Euler, per-step Delaunay,
`frozen_set='membership'`, `pressure_flux='simplex_gradient'`,
`viscous_flux='simplex_gradient'`, serial) on the library integrator. The
case has no retopology code of its own: the former `retopologize_cylinder`
closure is gone (it froze every hull vertex, the inlet cap included, and
kept no simplex cache, which is what stalled the run; audit 2026-09-25 M2).

Everything is shared with the 2D channel, see
`cases_dynamic/Hagen_Poiseuile/README.md` for the model (prescribed
pressure, no pressure solve, upstream buffer of prescribed plug motion,
buffer behind the outlet, walls frozen by membership):

| file | content |
|---|---|
| `Hagen_Poiseuile_3D.py` | runner: `run_case('hagen_poiseuille_3D')` |
| `../Hagen_Poiseuile/src/_run.py` | runner body, shipped parameters (`CASES`) |
| `../Hagen_Poiseuile/src/_setup.py` | `setup_poiseuille_developing(dim=3, ...)` |
| `../Hagen_Poiseuile/src/_metrics.py` | profile error, mass flux, census |
| `../Hagen_Poiseuile/diagnose_poiseuille.py` | `arms3d`, `static`, `slivers` |
| `run_cluster.py` | the same run with the `backend` / `workers` axes replaced (SLURM); default `--backend gpu --workers 8` |
| `visualize_hp3d.py` | figures and polyscope viewer from the saved run |

## Run

In the `ddg` environment, from anywhere:

```bash
python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --headless     # shipped run, about 15 min
python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --n-refine 1 --L 2 --mu 0.1 --steps 300 --tag quick   # 20 s
python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py --pressure-flux centred --tag centred
python cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d
python -m ddgclib.scripts.view_polyscope --snapshots cases_dynamic/Hagen_Poiseuile_3D/results/hagen_poiseuille_3D/snapshots
cd cases_dynamic/Hagen_Poiseuile_3D && python visualize_hp3d.py --no-polyscope
```

Options as for the 2D runner (`--L`, `--mu`, `--n-refine`, `--dt`,
`--t-end`, `--steps`, the A/B arms `--viscous-flux`, `--pressure-flux`,
`--frozen-set`, `--backend`, `--workers`, and `--tag`).

`--backend` (and `run_cluster.py --backend`) selects who computes the
dual face areas of the 3D retopology: `torch` (needs PyTorch, uses CUDA
when available; ImportError otherwise), `gpu` (PyTorch on CUDA, else
PyTorch on the CPU, else numpy), `multiprocessing`. The preset's forces
do not read those areas, so the result is the numpy one to the bit; only
an arm with `--pressure-flux centred` sees the backend (16th digit).
Before 2026-10-02 every value stopped the run at the first retopology
(a backend name reached hyperct where an instance is needed).

```bash
cd cases_dynamic/Hagen_Poiseuile_3D && python run_cluster.py --n-refine 1 --n-steps 50 --dt 0.01   # smoke run, 8 s
```

## Outputs

- `results/hagen_poiseuille_3D[_tag]/summary.json`, `methods.json`,
  `snapshots/`, `hp3d_final_state.json`, `hp3d_history.pkl`
- `fig/hagen_poiseuille_3D[_tag]_profile.png`, `_development.png`

The directories `results/` (top level files), `results_full/`, `archive/`
and `run.log` are outputs of the stalled runs before lane H.

## Result (lane H, 2026-10-01; re-run unchanged after the fix round of 2026-10-02)

Shipped run: `Re_D = 2` (`mu = 0.05`, `t_dev = rho R^2 / (2.405^2 mu) =
0.86 s`), `L = 4`, refinement 2 (845 vertices, the wall is a 16-sided
prism), `dt = 0.01`, 1000 steps to `t = 10 s` (11.6 `t_dev`). The
comparison is the dual-volume weighted l2 norm of `u - u_analytical` (the
profile of the CIRCULAR pipe) over the free vertices of `2 <= z <= 4`.

| quantity | value |
|---|---|
| profile error l2 at the end / mean and maximum over the last quarter | 1.99e-02 / 1.91e-02, 2.11e-02 |
| `u_max` in the window (analytical 0.2) | 0.1977 |
| largest transverse velocity in the window, any sample (over all free vertices at every step: 1.6e-15) | 3.3e-17 |
| mass flux through `z = 0` over one inlet period, in `rho U_avg A` (share of the mass not in wall cells: 0.7004) | 0.7004 |
| volume flux in the window, in `U_avg A` | 0.940 |
| vertices: start / end; free vertices in the channel: min, max | 845 / 928; 380, 412 |
| vertices outside the wall, any time | 0 |
| walls: frozen / moved | 336 of 336 / 0 |

What is left between the run and the analytical profile is the mesh: the
radial resolution (free vertices at 6 radii, the axis included) and the
polygonal wall, whose cross-section is 2.8 % smaller than `pi R^2`. At
refinement 1 (an octagon, free vertices at 2 radii; `L = 3`, `mu = 0.1`)
the measure is 5.7e-02.

Attribution of the method axes (`diagnose_poiseuille.py arms3d`,
refinement 1, 600 steps):

| arm | l2 end | largest radial velocity |
|---|---|---|
| preset | 0.0566 | 2.6e-18 |
| `pressure_flux='centred'` on the `batch_e_star` area cache (what `delaunay` builds) | 0.0821 | 6.3e-03 |
| `pressure_flux='centred'` on the `p_ij` ring (cache cleared by a custom wrapper, 2.9 times the wall time) | 0.0566 | 1.26e-05 |
| `viscous_flux='two_point'` | 0.530 | 3.5e-18 |

Every arm is bit-identical from process to process since lane T
(2026-10-02; `cases_dynamic/diagnose_determinism.py sweep
hp3d_centred_laneH,hp3d_ring_laneH`). Before, the two centred arms were
not (their force reads dual face areas, and a boundary face barycentre
was summed in the order of memory addresses): over 9 fresh processes the
cache arm ended at l2 0.0811 or 0.0821 (0.1107 after other runs in the
same process or on a fragmented heap) and the ring arm at radial
velocity 1.3e-05 or 3.6e-04
(`results/laneH/arms3d*.json` in the 2D case directory). The cache arm is
chaotic all the same: a 1e-15 shift of the interior vertices moves its l2
between 0.079 and 0.169 (8 seeds).

Known limits: a flat tetrahedron between four free vertices of equal
radius appears in a few steps (2 of 600 at refinement 1) and is left out of
the viscous force for that step; the outlet mass flux over the single inlet
period of the shipped horizon (0.83) still contains the start-up of the
slow outer rings.

Evidence: `docs_temp/debug_session/laneH-poiseuille-developing.md`.
