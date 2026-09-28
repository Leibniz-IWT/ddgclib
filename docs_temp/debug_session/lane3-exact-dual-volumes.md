# Lane 3 — exact simplex-container dual volumes

> Session 2026-07-02, third lane (after lane1-zero-gauge-pressure, lane2-eos-consistency).
> Audit basis: [audit/dual-volume-3d.md](../audit/dual-volume-3d.md) (CONFIRMED high),
> [audit/dual-closure-antisymmetry.md](../audit/dual-closure-antisymmetry.md) (side finding).

## Verdict up front

**LANDED (2D + hyperct scope; the 3D production switch was built, measured,
and intentionally gated off).**  Net effect on the primary failing case:
oscillating-droplet 2D **l2 0.70907 → 0.48992 (−31%)**, **tail_growth
2.20527 → 1.72505 (−22%)**; all pinned floors bit-compatible; no new test
failures.  The exact barycentric dual-volume routine
(`Vol_i = (1/(dim+1)) * Σ_{T∋i} |T|`) now lives in hyperct
(`hyperct.ddg.simplex_dual_volumes` / `vertex_dual_volume`), is
partition-of-unity-exact to < 1e-12 in 2D **and** 3D (tests), and is wired
into the ddgclib **2D** production path (`dual_volume` dim==2,
`cache_dual_volumes` dim==2).  The **3D production rewire was implemented,
measured, and backed out**: it moves the pinned 3D static-droplet retopology
floor **UP** 7.3768e-05 → 7.6169e-05 (+3.3%), and the lane decision rule only
allows re-pinning a *lower* floor.  The mechanism is fully diagnosed below —
it is NOT a bug in the exact volumes; the exact measure honestly reports a
larger real settle-step volume jump that the mass-conserving redistribution
rescale converts into a uniform droplet pressure offset (~1.3 Pa).  The 3D
switch point is one conditional in two places, marked
`NOTE(lane3-dual-volume)` in `ddgclib/operators/stress.py`.

## What changed (file:line)

hyperct (symlinked sibling working tree — uncommitted, like the rest of it):

1. **`hyperct/ddg/_dual_volume.py` (NEW)** — `simplex_dual_volumes(HC, dim)`
   (vectorized one-pass over `HC._simplices`, returns `{vertex: Vol_i}` for
   every vertex in `HC.V`, 0.0 for simplex-less vertices) and
   `vertex_dual_volume(HC, v, dim)` (single-vertex scan).  Both raise
   `ValueError` when `HC._simplices is None`.  No exception-swallowing, no
   fan walks, exact for barycentric duals in any dim.
2. **`hyperct/ddg/__init__.py`** — exports both (`__all__` updated).
3. **`hyperct/ddg/_compute_dual.py:88-92`** — `compute_vd` now records
   `HC._vd_method = method` (`"custom"` for callables) so downstream geometry
   code can distinguish barycentric from circumcentric duals.  Previously
   nothing recorded the method anywhere.
4. **`hyperct/tests/test_dual_volume.py` (NEW, 11 tests)** — partition of
   unity on refined structured AND jittered Delaunay meshes, 2D (refine 3)
   and 3D (refine 2), rtol 1e-12; agreement with `dual_cell_area_2d` on
   interior 2D vertices (rtol 1e-12 structured / 1e-11 jittered);
   vertex-vs-batch consistency; boundary/corner inclusion; no-cache
   ValueError.

ddgclib:

5. **`ddgclib/operators/stress.py`**
   - new helper `_use_exact_barycentric_volume(HC)` (`stress.py:311-324`):
     requires `HC._simplices` present AND `HC._vd_method == 'barycentric'`
     (missing attribute defaults to barycentric = pipeline default).
   - `dual_volume` dim==2 branch (`stress.py:378-383`): exact
     `vertex_dual_volume` when the helper is True, else the legacy
     `dual_cell_area_2d` (kept for circumcentric / no cache).
   - `dual_volume` dim==3 branch: **unchanged behaviour**, now carries
     `NOTE(lane3-dual-volume)` (`stress.py:385-397`) documenting the
     measured +3.3% floor shift that blocks the switch, and the docstring
     Notes explain both paths.
   - `cache_dual_volumes` (`stress.py:436-445`): dim==2 + helper → one
     vectorized `simplex_dual_volumes` pass (avoids O(N·M) per-vertex scans);
     dim==3 / fallback loop unchanged.
6. **`ddgclib/dynamic_integrators/_integrators_dynamic.py:221-233`** —
   comment-only `NOTE(lane3-dual-volume)` at the batch_e_star volume
   assignment (the 3D production volume source); behaviour unchanged
   (the exact override was implemented here, measured, and reverted).
7. **`ddgclib/tests/test_stress.py`** — new `_build_jittered_delaunay_mesh`
   helper + `TestDualVolumeExactSimplex` (6 pass + 1 strict xfail):
   - `test_partition_of_unity_2d_jittered` — production `dual_volume` dim=2
     tiles the unit square to rtol 1e-12 (pre-fix: 0.9922, corner deficit).
   - `test_partition_of_unity_3d_jittered_hyperct` — exact 3D truth via
     `hyperct.ddg.simplex_dual_volumes`, rtol 1e-12.
   - `test_partition_of_unity_3d_jittered_production` — **strict xfail**
     documenting that production `dual_volume(dim=3)` still does NOT tile
     (0.86–0.94), with the reason pointing here.
   - agreement with `dual_cell_area_2d` on interior vertices; exact
     `cache_dual_volumes`; circumcentric fallback (must NOT use the
     1/3-rule); `_retopologize` 2D assigns exact volumes that tile.
   (The old `TestDualVolume3D::test_partition_of_unity` on the symmetric
   9-vertex no-cache fixture is untouched — it exercises the legacy path.)

## Why

`dual_volume` reconstructed barycentric dual cells geometrically: 2D shoelace
on the dual polygon (exact except the 4 domain-corner cells, O(h²) total
deficit) and a 3D `v_star` tet-barycenter-only fan walk with silent per-edge
exception skipping (1–4% interior / ~20% boundary undercount, non-converging,
partition-of-unity deficit 6–14%).  The EOS reads `rho = m/dual_vol`, so any
volume error/jump is misread as compression.  For barycentric duals the exact
closed form is `Vol_i = (1/(dim+1))·Σ_{T∋i}|T|` (barycentric subdivision
argument; validated numerically to 3.4e-15 in the audit).

## Probe evidence (all scripts in
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/wf3/`)

### The 3D rewire works geometrically…

With the (since backed-out) 3D wiring, `dual_volume`/`cache_dual_volumes`/
`_retopologize` all produced exact volumes: partition of unity 1.0 to
< 1e-12 on jittered 3D Delaunay meshes (pre-fix 0.86–0.92), interior
`_retopologize` volumes exact, boundary zeroing convention preserved.

### …but the pinned 3D floor moves UP (probe_ab_fan_vs_exact.py)

3D A.5.b scenario (retopo on, u forced 0, redistribute_mass=True, refine 2/2,
the exact configuration of `TestStaticDroplet3DRetopologyFloor`):

| | step-0 (frozen) | plateau (step ≥ 2) | plateau mean interface F | max p_phase drift vs t=0 |
|---|---|---|---|---|
| fan (pre-change, reproduced in-process) | 6.015320e-05 | **7.376786e-05** (= pinned, bitwise) | 2.529028e-05 | 1.044664 Pa |
| exact 3D (candidate) | 6.015320e-05 | **7.616854e-05 (+3.26%)** | 2.795886e-05 (+10.6%) | 1.321879 Pa (+26.5%) |

Both plateaus bit-stable from step 2.  Step-0 identical (t=0 pressures are
volume-independent because mass ICs use the same volumes — the bias cancels).

### Root cause of the shift (probe_step1_diff.py, probe_redistrib_diag.py)

- Across retopo #1, **272 interior vertices change dual_vol in BOTH schemes**
  with **0 phase/interface tag changes** — the *triangulation itself* changes
  (e.g. interface-vertex cells 3.86e-08 → 5.00e-08, +30%, in the exact
  measure).  Cause: qhull tie-breaking on the deliberately symmetric
  (cospherical) droplet cloud is input-ORDER dependent; setup's builder order
  differs from `list(HC.V)` order at retopo #1 (perturbation/merge re-keying
  moves vertices to the end of the cache), and the step-1 position writes
  re-key every vertex once more, which is why the floor test's documented
  `SETTLE_STEPS = 2` exists at all.
- The droplet-phase dual volume total grows across retopo #1 by rel 1.526e-03
  (exact) vs 8.804e-04 (fan): the fan measure *under-reports the real
  geometric jump*.  `redistribute_mass_multiphase` preserves pressure per
  vertex but conserves phase mass via a global scale factor
  (`mass_redistribution.py:330-345`): scale_droplet = 0.9984761 (exact) vs
  0.9991204 (fan) at retopo #1 → uniform droplet pressure offset
  ≈ K_d·|1−scale| ≈ 1.2 Pa (exact) vs 0.7 Pa (fan) → higher plateau residual.
- So the +3.3% is an *honest measurement of a pre-existing settle artifact*
  (order-driven Delaunay non-uniqueness), not an error in the exact volumes.
  Fixing it properly means canonicalizing Delaunay input order (changes every
  pinned 3D number) or changing the redistribution rescale design (guarded by
  the 06 §2 DO-NOT / mass-conservation invariant) — both out of lane scope.

## Measurement battery (after final state: 2D exact, 3D unchanged)

Baselines: post-lane-2 suite "1 failed, 818 passed, 12 skipped, 17
deselected, 2 xfailed"; equil 1.1351e-03; osc_l2 0.7090740204580823;
osc_tail 2.2052688800146534; floors 2.3749e-3 / 2.2717e-3 (2D),
6.0153e-05 / 7.3768e-05 (3D).

1. **Pinned floor tests** (`test_case_oscillating_droplet.py -m ""`):
   **12 passed** (identical to the pre-change baseline run this session).
   3D floor restored bitwise by the back-out; 2D floors pass with the exact
   2D volumes.
2. **Fast suite**: **1 failed, 824 passed, 12 skipped, 17 deselected,
   3 xfailed in 49.96s** — the only failure is the pre-existing
   `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`.
   824 = 818 (post-lane-2) + 6 new tests; +1 xfail is the intentional strict
   xfail documenting the 3D gap.  No NEW failures.
3. **hyperct suite**: 312 passed, 40 skipped, 6 xfailed, 39 errors —
   312 = 301 baseline + 11 new; the 39 errors are the pre-existing
   pytest-benchmark fixture errors (bit-identical to the baseline run).
4. **Equilibrium score** (`static_droplet_2D.py`): summary = **1.1847e-03**
   (post-lane-2 1.1351e-03, +4.37%; pre-lane baseline 1.1636e-03, +1.81%).
   max_KE_normalized 7.6739e-09, mass_drift 0.0.  Within the ≤5% band.
   Attribution: the 2D exact path changes dual_vol bit-level everywhere
   (same value, different arithmetic) and by +h²/8 at the 4 frozen box
   corners; the marginally-stable full-Delaunay-retopo trajectory amplifies
   bit-level differences to a few percent of this drift metric.
5. **Oscillation score** (`oscillating_droplet_2D.py`): see table below.
6. **A.5 bisection** (`diagnose_a5_bisection.py --redistribute-mass
   --n-steps 100`): see table below.

### Final-state metric table (verbatim)

| metric | baseline (post-lane-2) | this lane | delta |
|---|---|---|---|
| fast suite | 818 passed, 1 pre-existing fail, 2 xfailed | 824 passed, 1 pre-existing fail, 3 xfailed | +6 tests, +1 intentional xfail, no NEW failures |
| hyperct suite | 301 passed, 39 pre-existing benchmark errors | 312 passed, 39 same errors | +11 tests |
| 2D floor step0 / step1 | 2.3748568e-03 / 2.2716938e-03 (pins) | 2.374856801157633e-03 / 2.271693780237238e-03 | bit-compatible, tests pass |
| 3D floor step0 / plateau | 6.0153e-05 / 7.3768e-05 (pins) | 6.0153201139652905e-05 / 7.376786303662525e-05 | bit-compatible, tests pass |
| A.5 bisection 3D A.5.a / A.5.b peak | 6.0153e-05 / 7.3768e-05 | 6.0153201139652905e-05 / 7.376786303662525e-05 | unchanged (3D switch backed out) |
| equil summary | 1.1351e-03 | **1.1847162859108737e-03** | **+4.37% (within the 5% band)** |
| osc l2_error_normalized | 0.7090740204580823 | **0.48991833470391266** | **−30.9% (improved)** |
| osc tail_growth | 2.2052688800146534 | **1.7250489596305962** | **−21.8% (improved)** |
| osc linf_error_normalized | 1.05384362884825 | **0.8503543927096853** | **−19.3% (improved)** |
| osc summary | 1.20527 | 0.7250489596305962 | improved |
| osc mass_drift | ~1e-14 | 1.42893286651677e-14 | machine precision |

The oscillating-droplet improvement is attributable to the 2D exact
`cache_dual_volumes` at every dynamic retopo: wall/corner boundary cells now
get their true measure instead of the truncated polygon (corners −h²/8) and
the occasional ValueError→0.0 degenerate-vertex zeroing is gone, so
`rho = m/dual_vol` at wall vertices is less overestimated — consistent with
lane 2's finding that wall-vertex EOS clip saturation / near-wall flux of
pinned band-edge pressures is the dominant KE-tail driver.  The equilibrium
score's +4.4% is drift-metric sensitivity to bit-level trajectory changes
(same mechanism class, opposite sign, small).

## What the next lane / future sessions must know

- **The exact 3D dual volume is ready and tested upstream** —
  `hyperct.ddg.simplex_dual_volumes(HC, dim)` / `vertex_dual_volume`.
  Turning it on in production is: (a) `stress.py` dim==3 branch of
  `dual_volume` (mirror the dim==2 branch), (b) `cache_dual_volumes`
  `dim == 2` → `dim in (2, 3)`, (c) `_integrators_dynamic.py` step 5b —
  prefer `simplex_dual_volumes` over the `batch_e_star` fan volumes when
  `HC._simplices` is present (keep boundary zeroing).  All three must switch
  TOGETHER (mixed volume sources across setup/retopo create a first-retopo
  pressure jump much larger than either consistent choice).  Then re-pin
  `TestStaticDroplet3DRetopologyFloor.EXPECTED_PLATEAU_MAXF` 7.3768e-05 →
  7.6169e-05 (measured bit-stable) and flip the strict xfail
  `test_partition_of_unity_3d_jittered_production`.
- **Why the 3D floor rises with exact volumes**: the settle-step (retopo #1
  and #2) triangulation change is real and its droplet-volume jump is
  ~1.7× larger than the fan measure reports; the redistribution's
  mass-conserving global rescale turns Σ-volume changes into uniform
  per-phase pressure offsets.  The floor is therefore *partly an artifact
  of order-dependent Delaunay tie-breaking at setup→step-2*, not of volume
  accuracy.  Candidate real fixes (new lanes): canonicalize Delaunay input
  order (sort coords) inside `connect_and_cache_simplices` — makes
  triangulation order-invariant and kills the settle steps, but re-pins
  every 3D dynamic number; or make setup end in the retopo fixed point.
- **`HC._vd_method`** is now recorded by every `compute_vd` call — use it
  instead of guessing the dual type.
- The lane's original framing ("EOS reads the fan error as compression,
  refreshed at every retopo") is **only half-right under
  redistribute_mass=True**: on the static pinned harness the triangulation
  is bit-stable from step 2, so the fan error is *constant*, cancels against
  the mass ICs, and does not churn.  The fan-error churn mechanism needs
  *positions to move* (real dynamic runs) — there is still no 3D dynamic
  score harness to measure that (06 §1.6).
- 2D production dual volumes are now exact including the box corners;
  Σ dual_vol == domain area to 1e-12 (was −h²·/8·4 corners short).  2D
  conservation diagnostics based on Σ dual_vol are now exact.
