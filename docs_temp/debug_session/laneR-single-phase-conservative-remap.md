# laneR: single-phase conservative retopology remap (library)

Date: 2026-10-01. Lifts the lane K prototype
(`cases_dynamic/template/diagnose_single_phase_eos.py:remap_retopo`) into
the library behind the existing `remap` method axis.

## 0. What changed

| file | change |
|---|---|
| `ddgclib/operators/mass_redistribution.py` | new `snapshot_pressure_fresh(HC, dim, eos)` and its helper `_connectivity_dual_volumes`; `redistribute_mass_single_phase(..., include_frozen=False)` |
| `ddgclib/dynamic_integrators/_integrators_dynamic.py` | `_retopologize(..., retopo_remap=None)`; default path bit-identical |
| `ddgclib/methods/_axes.py` | `remap` axis applies to both phase models; evidence on `delaunay`, `dual_only`, `redistribute_mass` |
| `ddgclib/methods/_config.py` | `remap` removed from `_MULTI_ONLY`; single-phase `retopologize_fn()` returns `partial(_retopologize, retopo_remap='conservative')`; the lane K warning no longer fires when the remap is on |
| `ddgclib/methods/__main__.py` | `python -m ddgclib.methods --update METHODS.md` rewrites the generated sections 2-3 in place |
| `ddgclib/tests/test_single_phase_remap.py` | 9 new tests (below) |
| `ddgclib/tests/test_methods.py` | builder / validation / no-warning tests for the single-phase remap |
| `cases_dynamic/template/template.py` | showcase is now weakly compressible (Tait EOS, Mach 0.01) with `remap='conservative'` |

No integrator signature changed. The integrator already forwards
`pressure_model`, `redistribute_mass`, `boundary_filter`, `merge_cdist`,
`backend`, `remesh_mode` and `remesh_kwargs` by name to a callable
`retopologize_fn`, so the remap rides on a `functools.partial` of the
library `_retopologize`.

## 1. Algorithm

Inside one `_retopologize` call the vertex positions are fixed.

1. Snapshot `p_i = eos(m_i / V_i)` with `V_i` re-measured on the OLD
   connectivity at the current positions (`simplex_dual_volumes` on the
   `HC._simplices` of the previous retopology). This is the pressure the
   fluid has now. The stale `v.p` (last force evaluation, before the move)
   is not used: re-targeting to it erases the step's compression (lane K
   P3/P12).
2. Rebuild connectivity, boundary tags and duals (unchanged code).
3. Set `m_i = rho(p_i) V_i_new` on every vertex with a dual volume that is
   in the snapshot, frozen `bV` vertices included, then one global rescale
   so the total mass of those vertices is conserved exactly.

Snapshot fallbacks:

- No simplex cache, or a cache that references vertices no longer in
  `HC.V` (outlet deletion, merges): in 2D the cache is rebuilt from the
  1-skeleton (`rebuild_simplex_cache_2d`); in 3D the cached `v.dual_vol`
  is used, which is exact only while no vertex has moved since it was
  cached (true on the first call).
- Vertices without a measurable volume are not in the snapshot and keep
  their mass (newly injected inlet vertices, 3D boundary vertices under
  the zeroed-boundary-volume convention).

Validation: `retopo_remap='conservative'` raises without
`redistribute_mass=True` and an EOS `pressure_model`, and on the periodic
path. It is a no-op under `skip_triangulation=True`.

## 2. Measurements

Box of lane K with all four walls no-slip, mesh settled by one library
retopology before the ICs, Tait `n = 1`, `c_s = 10` (Mach 0.01), CFL 0.25,
200 steps, 41 vertices. All three arms are `SolverMethods` configs
(`ddgclib/tests/test_single_phase_remap.py`).

| arm | config | KE at step 1 | KE at step 200 | max \|u\| / u at step 1 | density range / rho0 |
|---|---|---|---|---|---|
| remap | `connectivity='delaunay', remap='conservative', redistribute_mass=True` | 0.43990437705748403 | 0.008987112540608227 | 1.00 | 0.9996 to 1.0004 |
| dual_only | `connectivity='dual_only'` | 0.43990437705748403 | 0.011227120983125657 | 1.00 | 0.9999 to 1.0001 |
| bare | `connectivity='delaunay'` | 0.43990437705748403 | 3253.49 (max 1.06e4) | 246 | 0.40 to 2.08 |

The remap and the fixed-connectivity run differ by at most 2.7 % of the
initial KE over the horizon. They are different discretisations after the
first step (the fixed mesh shears with the swirl), so this is not a remap
error bound.

One rebuild after a 3 % random interior displacement (forces flips):
without the remap the largest pressure jump exceeds 5 % of K; with it the
density after the rebuild equals the snapshot density times ONE factor for
all 41 vertices (spread below 1e-12), and total mass is conserved to
1e-14.

Template (`cases_dynamic/template/template.py`, builder mesh with
`rebuild_simplex_cache_2d`, 400 steps): KE 4.399e-01 to 1.415e-04 J,
density within 4.6e-04 of rho0.

## 3. Known limits

- The global rescale leaves a uniform pressure offset `K (s - 1)` per
  rebuild, where `s - 1` is of the order of (density variation) times
  (fraction of volume that changed cells). It is 1.4e-3 K in the 3 %
  displacement probe and negligible at Mach 0.01 (lane K P11 = P9). A
  uniform offset exerts no force on a closed fan. It is not force-free on
  an open (free-surface) fan. If the hydrostatic column or capillary rise
  shows a per-rebuild jolt at the free surface, the next step is an
  overlap-based conservative remap or a gauge like the multiphase
  `vol_corr`.
- Masses are re-targeted at fixed velocities, so linear momentum and KE are
  not conserved across a rebuild that moves volume between vertices of
  different velocity. A first rebuild from a non-Delaunay builder mesh
  changed KE by 4 % in a probe; settle the mesh at setup (one
  `_retopologize` before the ICs) when that matters.
- Setup volumes must be the simplex-exact ones. On a 2D builder mesh
  without a simplex cache the IC masses come from `dual_cell_area_2d`
  (corners undercounted 4x) and the first fresh snapshot then records a
  wrong corner density. The template calls `rebuild_simplex_cache_2d`
  explicitly; lane S moves that into the builders.
- 3D is covered by the algorithm (interior vertices; boundary vertices have
  zero dual volume by convention and are never read through the EOS) but
  has no pinned run yet.

## 4. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
$PY -m pytest ddgclib/tests/test_single_phase_remap.py -q
$PY cases_dynamic/template/template.py
```
