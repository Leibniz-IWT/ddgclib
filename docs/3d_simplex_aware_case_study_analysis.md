# 3D Simplex-Aware Dual Fix — Case Study Analysis

**Date:** 2026-04-14
**Scope:** `cube_flow`, `oscillating_droplet`, and adjacent setup/periodic paths.

## Summary per case

### `cases_dynamic/cube_flow/` — status: uses fix automatically, no action needed

- `cube_flow_3D.py` (line 144) drives `ddgclib.dynamic_integrators.euler`,
  which calls `_retopologize` every step
  (`ddgclib/dynamic_integrators/_integrators_dynamic.py:49`).
- That function already caches `HC._simplices` in 3D
  (`_integrators_dynamic.py:196-200`), so the Lagrangian loop uses the
  simplex-aware dual path automatically.
- Setup (`cases_dynamic/cube_flow/src/_setup.py`) uses `HC.triangulate()`
  + `HC.refine_all()`. The initial dual is built lazily by the first
  integrator step, so the fix kicks in before any stress computation.
- **No changes required.**

### `cases_dynamic/oscillating_droplet/` — status: needed fix (applied)

- `oscillating_droplet_3D.py` (line 116) drives `symplectic_euler` with
  `retopo_fn = partial(_retopologize_multiphase, ...)`
  (`src/_setup.py:184-186`).
- `_retopologize_multiphase`
  (`_integrators_dynamic.py:330`) delegates to `_retopologize` at line
  382, so every dynamic retopology step benefits from the fix.
- **Setup gap:** The mesh is first built by `droplet_in_box_3d` -->
  `_build_combined_mesh` in
  `ddgclib/geometry/domains/_multiphase_droplet.py`. That function runs
  its own `scipy.spatial.Delaunay` and `_setup_phases_and_duals`
  immediately calls `compute_vd` (line 128) — before any integrator
  runs. On the initial 3D mesh `HC._simplices` was unset, so the
  pre-simulation dual used the legacy nn-intersection path.
  The sharp droplet interface (fixed interface vertices + denser bulk
  mesh) is exactly the fixed-vs-advected geometry pattern that produces
  slivers.
- **Fix applied:** `ddgclib/geometry/domains/_multiphase_droplet.py`
  lines 80-99 — cache `HC._simplices` after the initial Delaunay when
  `dim == 3`, and invalidate the cache if `_build_combined_mesh` removes
  isolated vertices (rare but possible after deduplication).

### Related path also fixed

- `ddgclib/geometry/periodic.py` `retopologize_periodic` (lines 443-463)
  also ran Delaunay + `compute_vd` without caching simplices. No 3D case
  under `cases_dynamic/` currently uses it, but to keep the abstraction
  consistent I added the same simplex cache — only 4-vertex resolved
  simplices are retained (3D tets).

## Files modified

- `ddgclib/geometry/domains/_multiphase_droplet.py:80-99` — cache
  `HC._simplices` after initial 3D Delaunay; clear on isolated-vertex
  removal.
- `ddgclib/geometry/periodic.py:443-463` — cache `HC._simplices` after
  ghost-resolved 3D Delaunay.

## Verification

- Full fast suite: **702 passed, 3 skipped** (identical to pre-fix
  baseline; no regressions).
- Targeted `test_periodic.py`, `test_domains.py`, `test_stress.py`:
  **157/157 pass**, including
  `test_p_ij_linear_precision_jittered_3d`
  (`ddgclib/tests/test_stress.py`) which is the jittered-boundary
  regression test.
- The 3D oscillating droplet script is not part of the fast suite and
  typical runtime is ~5 min; I did not execute it, but the setup-time
  fix is exercised by `test_domains.py::TestMultiphaseDroplet3D` (now
  green with the cache in place).

## Abstraction (applied after this report)

The three+ call sites that duplicated the
`Delaunay(coords); connect; HC._simplices = [...]` pattern have been
refactored to use a single helper:

```python
from ddgclib.geometry import connect_and_cache_simplices

connect_and_cache_simplices(HC, verts, dim, coords=coords)
# or, if you already have the simplex list (e.g. after ghost resolution):
connect_and_cache_simplices(HC, verts, dim, simplices=simplices)
```

Location: `ddgclib/geometry/_retriangulation.py`.  Also exposes
`invalidate_simplex_cache(HC)` for callers that mutate topology outside
of this helper (e.g. isolated-vertex removal in
`_multiphase_droplet.py`).

**Refactored call sites (4):**

- `ddgclib/dynamic_integrators/_integrators_dynamic.py:181-185`
- `ddgclib/geometry/domains/_multiphase_droplet.py:80-95`
- `ddgclib/geometry/periodic.py:450-456`
- `benchmarks/_integrated_benchmark_classes.py:211-215` and
  `benchmarks/_integrated_benchmark_cases.py:555-559`

After refactor: 702 fast tests pass, 0 regressions.  Jittered linear
precision remains at machine epsilon.

## Remaining follow-ups

- `_compute_vd_3d_batch` (`hyperct/ddg/_compute_dual.py:485`) still
  uses nn-intersection. Worth routing through the simplex-aware path
  as part of the follow-ups listed in
  `docs/3d_simplex_aware_dual_fix.md` section "Follow-up Opportunities".
- `ddgclib/_compat.py:44` and `ddgclib/legacy/plots.py:234` also run
  standalone Delaunay but are plotting/legacy utilities, not FVM
  pipelines — not in scope.
