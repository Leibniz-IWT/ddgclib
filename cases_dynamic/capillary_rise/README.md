# Capillary rise

Two families of runners share this directory.

| runner | preset | what runs |
|---|---|---|
| `capillary_rise_2D.py` | `capillary_rise_static_2D` | static rise in a 4 mm slit on the library `symplectic_euler` (laneI, 2026-10-06) |
| `capillary_rise_3D.py` | `capillary_rise_static_3D` | static rise in a 4 mm round tube, same path |
| `capillary_rise_2D_dynCA.py`, `_3D_dynCA.py` | none (hand-rolled loop) | dynamic rise with a data-driven or modelled contact angle, remeshing; owned by another agent, see `METHODS.md` section 4 |

## The static case (laneI)

A water column stands in a tube dipped into a reservoir whose free surface is
at `y = 0`. The mesh is the liquid in the tube from a reservoir band
(`y < 0`, about two tube widths deep) up to the meniscus. The static answer is
the Young-Laplace meniscus of the contact angle: its volume-averaged height is
Jurin's height `gamma cos(theta) / (rho g r)` (slit) or twice that (round
tube), up to the compressibility of the linear Tait model (about
`rho g h / (2 K)` higher, 0.5 % at `c0 = 10 sqrt(g h_J)`).

Everything runs through `SolverMethods` (`src/_static.py`, no time loop of
its own):

| item | library method |
|---|---|
| connectivity | `dual_only` (2D), `dual_only_bare` + `edge_area_source='p_ij_simplex'` (3D), walls frozen by membership (`boundary_filter`) |
| gravity | `SolverMethods.dudt_fn(body_force=...)` |
| EOS | `TaitMurnaghan(n=1, P0=0, K=rho (10 sqrt(g h_J))^2)` as `pressure_model` |
| surface tension and contact angle | method axis `contact_line='energy_gradient'`: `ddgclib.operators.free_surface.FreeSurface`, the gradient of `gamma A_free - gamma cos(theta) A_wet` over the boundary facets, bound through `dudt_fn(free_surface=...)` |
| contact-line vertices | `AxialSlideBC`: slide along the wall, lateral velocity and position are the wall reaction |
| reservoir | `HydrostaticReservoirBC`: the band below `y = 0` is held on the compressible hydrostatic profile every step (it supplies or absorbs the mass the column needs) |
| reference | `ddgclib.analytical.young_laplace_meniscus` (arc-length shooting, 2D and axisymmetric), `jurin_height`, `hydrostatic_pressure_tait`; errors by `ddgclib.analytical.integrated_l2_norm` |

The mesh is `n_cells` unit cells of width `2 r` stacked along gravity
(`extrude` of the structured square or of the `cylinder_volume`
cross-section; `ensure_simplex_cache` gives the extrusion its exact simplex
volumes). In 3D the tube is the polygon inscribed in the circle, so the exact
static mean height of the discrete model is `gamma cos(theta) (perimeter /
area) / (rho g)`; the runner prints both the round-tube reference and the
polygon one (`h_ref_poly`).

## Run

```bash
python cases_dynamic/capillary_rise/capillary_rise_2D.py            # refinement 2, 100 t_ac, about 4 min
python cases_dynamic/capillary_rise/capillary_rise_3D.py            # refinement 1, 40 t_ac, about 3 min
```

| option | meaning |
|---|---|
| `--ic flat` (2D default) | flat meniscus at Jurin's height with the hydrostatic masses: the surface has to curve and the column to take in the missing volume through the band; settles to round-off (the measurement of record) |
| `--ic young_laplace` (3D default) | the column above `y = 0` is stretched onto the reference profile; it creeps towards the equilibrium (contact-line creep) and at 2D refinement 3 the fixed connectivity drifts into a collapsed cell at 177 t_ac; the coarse octagonal 3D tube needs it (its flat start settles into a second, 9 % lower equilibrium) |
| `--arm remap` | the reconnecting arm `preset.replace(connectivity='delaunay_material', remap='conservative', redistribute_mass=True)` |
| `--n-refine N`, `--n-tac T`, `--alpha-art A`, `--r R`, `--n-cells N` | refinement, horizon in acoustic times `L_dom / c0`, artificial viscosity `mu = A rho c0 dx` (default 0.05: enough to settle, 0.02 flutters, 0.1 doubles the creep time), tube half-width, cells |
| `--out DIR` | write `results/` and `fig/` under `DIR` instead of the case directory |

Outputs: `results/<tag>/methods.json` (the configuration that ran, with the
effective implicit choices and both git SHAs), `summary.json`,
`series.npz` (`t`, `ke`, `umax`, `h`), `snapshots/`; `fig/<tag>_settling.png`,
`<tag>_meniscus.png`, `<tag>_mesh_pressure.png`, `<tag>.mp4`.

`diagnose_static_rise.py` runs the refinement study (`convergence`), the
connectivity A/B (`arms`), the viscosity sweep (`viscosity`) and prints the
settling envelope of a finished run (`series`); it writes only under
`--out`. The measurements are in
`docs_temp/debug_session/laneI-static-capillary-rise.md`; the pins in
`ddgclib/tests/test_case_capillary_rise_static.py`.

The former scaffold (`src/_setup.py`: Washburn body force on a closed column,
hand-rolled loop) is not used by any runner any more (the dynCA runners
import `src/_setup_dynca.py`, `src/_params.py` and `src/_dynamic_ca.py`); it
is left in place for the owner to delete.
