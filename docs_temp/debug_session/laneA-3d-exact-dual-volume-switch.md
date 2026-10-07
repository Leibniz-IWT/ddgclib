# Lane A — 3D exact-dual-volume switch + canonical Delaunay input order

> Session 2026-07-29 (wf4).  Executes the highest-value deferred item from the
> 2026-07-02/03 session (08 §5.2): flip the three `NOTE(lane3-dual-volume)`
> switch points TOGETHER and re-pin the 3D floor.  Basis:
> [lane3-exact-dual-volumes.md](lane3-exact-dual-volumes.md) (mechanism +
> measured 7.616854e-05 target), [audit/dual-volume-3d.md](../audit/dual-volume-3d.md).

## Verdict up front

**LANDED, including the optional canonical-order probe — the 3D floor is now
LOWER than the old fan-walk floor.**  The 3D production pipeline uses the exact
simplex-container dual volumes (`Vol_i = (1/4)·Σ_{T∋i}|T|`) at all three switch
points, and hyperct `connect_and_cache_simplices` canonicalizes the 3D qhull
input order (lexicographic coordinate sort), which makes retopology of a static
cloud idempotent and **eliminates the step-1→2 settle artifact** diagnosed by
lane 3.  Net: 3D plateau floor **7.3768e-05 → 7.274172e-05 (−1.4 %)** with the
plateau reached at **step 1** (was step 2); `dual_volume(dim=3)` now tiles
jittered Delaunay domains to 1e-12 (strict xfail flipped to a passing test).
Every 2D number is **bit-identical** (both changes are 3D-gated by design).
Fast suite 834 P / 0 F; hyperct 336 P; floor battery 14 P + a5b 2 P.

Composition of the floor move (both measured on `diagnose_a5_bisection.py
--redistribute-mass --n-steps 100`, refine 2/2, the exact
`TestStaticDroplet3DRetopologyFloor` configuration):

| configuration | step-0 | step-1 | plateau | plateau start |
|---|---|---|---|---|
| fan volumes (pre-lane, pinned) | 6.015320113965291e-05 | 7.0325e-05 | **7.3768e-05** | step 2 |
| exact volumes only (switch, no canon order) | 6.015320113965291e-05 | 7.274170420612488e-05 | **7.616853911101026e-05** (+3.26 %) | step 2 (bit-stable ≥ 2, matches lane 3's measured 7.616854e-05 exactly) |
| exact volumes + canonical 3D qhull order (**shipped**) | 6.0153201139652905e-05 (bit-unchanged) | 7.274172178727318e-05 | **7.274172178727318e-05** (−1.4 % vs fan) | **step 1** (settle step gone) |

The +3.26 % of the switch alone is lane 3's diagnosed settle artifact (the
exact measure honestly reports the larger real settle-step volume jump from
order-dependent qhull tie-breaking, which `redistribute_mass_multiphase`'s
global rescale converts into a uniform ~1.3 Pa droplet pressure offset).  The
canonical input order removes the artifact's cause — retopo #1/#2 now rebuild
the identical triangulation — so the honest exact measure lands BELOW the old
fan floor.  Plateau stability: F rel spread 6.05e-12 over steps 1..100
(inside the test's 1e-10 assertion; the old fan plateau was exactly
bit-stable), volume rel spread 0.0 from step 1, mass drift 5.6e-16.

## What changed (file:line)

ddgclib:

1. **`ddgclib/operators/stress.py`**
   - `dual_volume` dim==3 branch: exact `vertex_dual_volume` when
     `_use_exact_barycentric_volume(HC)` (mirrors dim==2); legacy `v_star` fan
     walk kept as circumcentric / no-simplex-cache fallback.  Docstring +
     `NOTE(lane3-dual-volume)` updated to the enabled/final state.
   - `cache_dual_volumes`: `dim == 2` → `dim in (2, 3)` for the vectorized
     `simplex_dual_volumes` pass.
2. **`ddgclib/dynamic_integrators/_integrators_dynamic.py`** (step 5b of
   `_retopologize`): in 3D, prefer `hyperct.ddg.simplex_dual_volumes` over
   `batch_e_star`'s fan-walk volumes when the exact helper applies; boundary
   zeroing convention preserved; 2D intentionally keeps the batch_e_star
   volumes (bit-identical 2D baseline).  `edge_areas` still come from
   `batch_e_star`.
3. **`ddgclib/geometry/_dual_split_2d.py`** — `_dual_volume_3d` now prefers
   the cached `v.dual_vol` (the authoritative pipeline value, per its own
   design comment) over recomputing.  Required because
   `assign_simplex_phases` (multiphase.py:221-227) auto-populates
   `HC._simplices` via Delaunay on structured-connectivity meshes: post-switch
   the recompute measured the mismatched auto-cache triangulation and desynced
   from the cached volumes (3 `test_dual_split_2d.py` failures on the box
   fixture: split partition 0.0625 vs cached 0.1328125 at the origin vertex).
   Pre-switch behaviourally neutral (cached == recomputed fan).  This
   **auto-Delaunay-on-structured-mesh mixed state remains a pre-existing
   footgun** (adds Delaunay edges to a structured 1-skeleton) — out of lane
   scope, now documented in the helper docstring.
4. **`ddgclib/tests/test_stress.py`** — strict xfail
   `test_partition_of_unity_3d_jittered_production` flipped to a passing test
   (production 3D `dual_volume` tiles to 1e-12); class docstring updated.
5. **`ddgclib/tests/test_case_oscillating_droplet.py`**
   (`TestStaticDroplet3DRetopologyFloor`) — RE-PIN `EXPECTED_PLATEAU_MAXF`
   7.3768e-05 → **7.274172e-05** and `SETTLE_STEPS` 2 → **1**; docstring
   documents the old→new chain (incl. the intermediate 7.616854e-05
   switch-only value) and marks the 2026-06-02 long-run numbers HISTORICAL.
6. **`ddgclib/tests/test_a5b_longrun_regression.py`** — RE-PIN
   `A5B_3D_PEAK`/`A5B_3D_END` 7.3768e-05 → **7.274172e-05** (peak == end now:
   the step-1 transient IS the plateau); 2D pins untouched.

hyperct (sibling working tree `/home/endres/projects/hyperct`):

7. **`hyperct/ddg/_retriangulation.py`** (`connect_and_cache_simplices`) —
   `NOTE(laneA-canonical-order)`: when triangulating from `coords` with
   dim==3, sort the qhull input lexicographically (`np.lexsort(coords.T[::-1])`)
   and map `tri.simplices` back through the permutation
   (`order[tri.simplices]`), so the triangulation is a function of the point
   SET only.  Gated to dim==3 — 2D pinned baselines stay bit-identical.
   Callers passing precomputed `simplices` are untouched; the vertex
   correspondence invariant is preserved by the index remap.
8. **`hyperct/tests/test_retriangulation_order.py` (NEW, 2 tests)** —
   cospherical 26-point sphere cloud + centre + jittered interior:
   (a) any input permutation yields the identical simplex set;
   (b) the index remap keeps vertex correspondence (cached tets tile the
   convex hull volume to 1e-12).

Result artifacts (probe JSONs, kept):
`cases_dynamic/oscillating_droplet/results_a5_bisection/a5_bisection_laneA_exact3d.json`
(switch only) and `a5_bisection_laneA_canon_probe.json` (switch + canonical
order).  Raw logs in scratchpad `wf4/laneA-3d-exact-dual-volume-switch/`.

## Measurement battery (final state)

Baselines: 2026-07-03 committed state — fast 833 P / 0 F / 3 xfail; floor
battery 14 P; equil 1.1847162859108737e-03; osc l2 0.1785660454150319 /
tail 0.9992507831101141; hyperct 334 P + 39 pre-existing benchmark errors;
3D floors 6.0153e-05 / 7.3768e-05.

| metric | baseline | this lane | delta |
|---|---|---|---|
| fast suite | 833 P, 0 F, 3 xfailed | **834 P, 0 F, 12 skipped, 17 deselected, 2 xfailed** (51.4 s) | +1 P / −1 xfail = the flipped 3D partition-of-unity test; NO new failures |
| floor battery (`test_case_oscillating_droplet.py -m ""`) | 14 P | **14 P** (15.6 s; re-runs green after every edit incl. the split fix) | 2 documented re-pins (plateau, SETTLE_STEPS) |
| a5b long-run regression (`-m ""`, incl. slow 3D) | 2 P | **2 P** | 1 documented re-pin (3D peak/end) |
| slow battery (`-m slow`) | 3 P | **3 P** | — |
| hyperct suite | 334 P, 40 skip, 6 xfail, 39 pre-existing benchmark errors | **336 P**, 40 skip, 6 xfail, same 39 errors | +2 new order-invariance tests |
| 2D floor step0 / step1 | 2.3748568e-03 / 2.2716938e-03 | 0.002374856801157633 / 0.002271693780237238 | **bit-identical** |
| 3D floor step0 | 6.0153e-05 (pin) | 6.0153201139652905e-05 | bit-compatible, pin unchanged |
| 3D floor plateau | 7.3768e-05 (pin) | **7.274172178727318e-05, from step 1** | **re-pinned 7.274172e-05 (−1.4 %), settle step eliminated** |
| equil summary | 1.1847162859108737e-03 | 1.1847162859108737e-03 (max_KE_norm 7.673878163142909e-09, mass_drift 0.0) | **bit-identical** |
| osc l2 / linf / tail | 0.1785660454150319 / 0.32842491275938274 / 0.9992507831101141 | identical to 16 digits | **bit-identical** (mass_drift 2.676988154993443e-14) |

2D-leak check demanded by the lane brief: 2D floors, equil, oscillation, and
the 2D half of the A.5 probe are all bit-identical — nothing leaked (the
`dual_volume`/`cache_dual_volumes` 3D branches, the step-5b preference, and
the canonical qhull order are all dim==3-gated; the `_dual_volume_3d` cache
preference is only reachable from the 3D split).

## What the next lane / future sessions must know

- **The 3D floor pins are now**: step0 6.0153e-05 (unchanged),
  plateau 7.274172e-05 from step 1 (`SETTLE_STEPS = 1`), rel-spread assertion
  unchanged at 1e-10 (measured 6.05e-12 over 100 steps — the exact path is no
  longer exactly bit-stable like the fan was, just stable to ~1e-11).  The
  switch-only intermediate value 7.616854e-05 appears in comments as history;
  do not pin against it.
- **3D retopology of a static cloud is now idempotent** (canonical qhull
  input order).  Any future probe that relied on the settle step (e.g.
  `SETTLE_STEPS = 2` reasoning, step-1 transient 7.0325e-05) is obsolete.
  The residual step-0→1 jump (6.0153e-05 → 7.2742e-05) is the remaining
  boundary dual-cell zeroing + `mps.refresh` transition, NOT order dependence.
- **`dual_volume(dim=3)` / `cache_dual_volumes(dim=3)` / integrator step 5b
  are consistent exact sources**; the fan walk survives only for
  circumcentric duals or no `HC._simplices`.  Do not reintroduce mixed
  sources — lane 3 measured the first-retopo pressure jump they cause.
- **Footgun surfaced (open, small)**: `assign_simplex_phases`
  auto-Delaunay-populates `HC._simplices` on structured-connectivity meshes
  (multiphase.py:221-227), adding Delaunay edges to a structured 1-skeleton
  and caching a triangulation that does not match the structured duals.  The
  split helper now reads cached `v.dual_vol` so the multiphase pipeline is
  internally consistent, but the mixed mesh state itself remains; a future
  lane could make the auto-population rebuild duals/volumes or refuse on
  structured meshes.
- The 2D droplet-case conclusions of lanes 1–5/7 are untouched (all 2D
  numbers bit-identical).  The 3D dynamic score harness is STILL missing
  (06 §1.6) — this lane validated the static floor only.
