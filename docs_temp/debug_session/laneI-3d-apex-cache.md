# laneI: 3D interface apex-cache invalidation (audit 2026-09-25 F1/F2)

Date 2026-09-25. Python `/home/endres/anaconda3/envs/ddg/bin/python`, run from
the repo root. Nothing committed. Scratch:
`/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad/laneI/`.

## TL;DR

- **The fix is in and tested.** Both caches are now invalidated when
  `extract_interface` rebuilds the interface. The regression test fails 2 of
  3 on the pre-fix code and passes 3 of 3 with the fix.
- **All pins are bit-identical**: the 3D default (dual_only), the floor
  battery, the fast suite, and the A.5 bisection floors (2D 2.3749e-03 /
  2.2717e-03, 3D 7.2742e-05).
- **The 3D per-step Delaunay scores are also bit-identical** after the fix:
  bare Delaunay 1.5244561707801316 / 0.47066568050584034, and Delaunay+remap
  1.8734809234034775 / 0.2532690179972299. The reason: in the real 872-step
  runs the interface triangle set **never changes** by vertex identity
  (0 changes in 875 / 1747 `extract_interface` calls). Delaunay reconnects
  the bulk tets on 863 / 870 steps, but the 192 interface triangles stay the
  same. The stale apex cache was therefore never stale in these runs.
- **Verdict:** F1 is a real latent bug and is now fixed. It is **not** the
  cause of laneB's 1.52 or laneE's 1.87. The 3D DO-NOT (keep `dual_only`,
  do not adopt delaunay or delaunay+remap) **stands unchanged**. I flipped
  no defaults.

## 1. The fix (exact diff)

Only `ddgclib/geometry/_interface_subcomplex.py` changed. `_curvatures_heron.py`
is untouched. `extract_interface` is the only writer of `HC.interface_triangles`.
The grep also found a hand-built fixture in `test_interface_subcomplex.py:324`,
which is a test, not a runtime writer.

```diff
@@ -113,6 +113,27 @@ def extract_interface(
     HC.interface_edges = iface_edges
     HC.interface_triangles = iface_tris
 
+    # Invalidate the curvature caches derived from interface_triangles
+    # (audit 2026-09-25 F1/F2, _curvatures_heron.py).  The coordinate-keyed
+    # ``_interface_x_to_v`` ('stokes') is stale after any vertex move, so
+    # it is always dropped.  The id-keyed apex map ('integrated') is
+    # dropped only when the triangle set changed by vertex identity: under
+    # frozen connectivity (3D dual_only) it stays valid and is kept, which
+    # keeps its apex order (and hence the summation order) bit-identical.
+    # While the apex map exists it holds references to every vertex in the
+    # signature, so the ids cannot be recycled behind a matching signature.
+    if hasattr(HC, '_interface_x_to_v'):
+        del HC._interface_x_to_v
+    tri_ids = frozenset(
+        frozenset(id(v) for v in entries[0][1])
+        for fkey, entries in incident_phases.items()
+        if fkey in iface_tris
+    )
+    if tri_ids != getattr(HC, '_interface_tri_ids', None):
+        if hasattr(HC, '_interface_edge_to_apex'):
+            del HC._interface_edge_to_apex
+        HC._interface_tri_ids = tri_ids
```

**Why the apex map is dropped only on change, instead of on every refresh
(the "minimal" fix proposed in the brief).**
- The apex lists are filled by iterating `HC.interface_triangles`, a set of
  frozensets of coordinate tuples. Its iteration order changes whenever
  vertices move.
- `hndA_i_interface` then accumulates `HNdA_i += hnda_ijk; HNdA_i += hnda_ijl`
  in apex order.
- Rebuilding on every refresh could therefore swap the two apexes of an edge
  and change the last bits of every dual_only force. That would break the
  pinned 3D default.
- Rebuilding only when the triangles change by vertex identity is exactly as
  correct, and it keeps dual_only bit-identical (verified in §3a).
- `_interface_x_to_v` is keyed by coordinates, so any move makes it stale.
  It is always dropped (F2). `'stokes'` is used by no pin.

## 2. Regression test

New file `ddgclib/tests/test_interface_cache_invalidation.py`. It uses the 3D
droplet at refine 1/1 and runs in about 0.8 s.

| test | what it asserts |
|---|---|
| `test_apex_cache_rebuilt_after_delaunay_retopology` | Build the cache, add 1e-3 jitter, then run the setup's full Delaunay retopo (mps.refresh). Checks: the interface triangle set changed; `_interface_edge_to_apex` was dropped; the next force equals a forced-fresh rebuild (atol 1e-12 * max\|F\|). Also checks that re-installing the stale map differs by more than 10 % of max\|F\|, so the fixture can detect a regression. |
| `test_apex_cache_kept_under_frozen_connectivity` | Add 1e-4 jitter and run `mps.refresh` (dual_only-like). Checks: same triangles by identity; the cache object is the **same** object (`is`); the force equals a fresh rebuild. |
| `test_stokes_coordinate_map_dropped_on_refresh` | After a move and refresh, `_interface_x_to_v` is gone and `integrated_hndA_i_interface` equals a fresh rebuild. |

- **With the fix:** 3 passed. `test_simplex_aware_curvature.py` and
  `test_interface_subcomplex.py` also pass (30 passed in total).
- **Negative control:** I loaded the git-HEAD `extract_interface` through a
  pytest plugin in scratch (`conftest_prefix.py`), without editing the repo.
  Result: 2 failed (apex-cache test and stokes test), 1 passed (the
  frozen-connectivity test, as expected). The test detects the bug.

## 3. Battery (pins untouched)

**Blocker, not mine.** `ddgclib/tests/test_case_oscillating_droplet.py` fails
at collection with `NameError: PRESETS`. The campaign lead is editing that
file right now (it uses `PRESETS` at module level, line 689, without the
import). I did not edit it. Workarounds:
- I ran it with a scratch pytest plugin (`shim_presets.py`) that puts
  `ddgclib.methods.PRESETS` (and friends) into builtins.
- I ran the fast suite with that one file `--ignore`d.
- Re-run both without the shim once the lead adds the import.

| check | result |
|---|---|
| Floor battery `test_case_oscillating_droplet.py test_a5b_longrun_regression.py test_oscillation_score_3d.py -m ""` (with shim) | **38 passed**, 0 failed (44 s) |
| Fast suite `pytest ddgclib/tests -m "not slow"` (oscillating file ignored) | **933 passed**, 12 skipped, 2 xfailed, 0 failed |
| `diagnose_a5_bisection.py --redistribute-mass --n-steps 20`, 2D | A.5.b peak **2.3749e-03**, end **2.2717e-03**: unchanged |
| same, 3D | A.5.a 6.0153e-05; A.5.b peak **7.2742e-05**, end **7.2742e-05** (= 7.274172e-05 at the printed 5 digits): unchanged |

## 4. Scored 3D runs (872 steps, refine 2/2, dt 1.84e-04, t_end 0.1600)

Every configuration is built from `ddgclib.methods.PRESETS` and has a
`methods.json` next to its score.

| run | config | l2 | tail | mass | R_max_peak | wall (s) |
|---|---|---|---|---|---|---|
| pin: baseline_oscillation_3d.json | `PRESETS['oscillating_droplet_3D']` (dual_only) | 0.24811340819647862 | 0.08409976059818802 | 1.905002320272536e-14 | 0.010790236105250779 | – |
| **a** after fix | same, `oscillating_droplet_3D.py` | **0.24811340819647862** | **0.08409976059818802** | **1.905002320272536e-14** | **0.010790236105250779** | 540 |
| pin laneB (pre-fix) | `PRESETS['oscillating_droplet_3D_delaunay']` | 1.5244561707801316 | 0.47066568050584034 | 3.865706462775263e-14 | 0.011397245128957192 | – |
| **b-prefix** (fix disabled, this machine) | same, runner via scratch wrapper with the HEAD `extract_interface` | 1.5244561707801316 | 0.47066568050584034 | 3.865706462775263e-14 | 0.011397245128957192 | 555 |
| **b** after fix | same, `oscillating_droplet_3D.py --retopo delaunay` | **1.5244561707801316** | **0.47066568050584034** | **3.865706462775263e-14** | **0.011397245128957192** | 545 |
| pin laneE (pre-fix) | delaunay + conservative remap | 1.8734809234034775 | 0.2532690179972299 | 4.23e-14 | – | – |
| **c** after fix | `PRESETS['oscillating_droplet_3D_delaunay'].replace(remap='conservative')`, `diagnose_3d_remap_afterfix.py` | **1.8734809234034775** | **0.2532690179972299** | **4.2333384894945245e-14** | **0.01155883605820377** | 812 |

- All runs finished with EXIT 0.
- Every run: boundary_saturation False, n_interface 98 / 98, n_frames 111.
- The four jobs ran concurrently on a 32-core machine, so the wall times are
  indicative only.

**Bit-identity summary**
- **(a)** equals the baseline on every numeric key (l2, linf, apex_l2/linf,
  tail, mass, R_max_peak, dual_vol_drift_post, dual_vol_step0_jump,
  n_interface_*, t_end, inputs). The rewritten `score.json` is also identical
  to the pre-run copy.
- **(b-prefix)** equals laneB's pre-existing `score_delaunay.json` on every
  key. This validates the harness on this machine.
- **(b)** equals b-prefix on every key.
- **(c)** equals laneE's l2 and tail to all 16 digits. mass 4.2333e-14 is
  consistent with laneE's 4.23e-14.
- **Nothing moved.**

## 5. Why nothing moved (probe)

**Probe.** `scratch/laneI/probe_invalidation_count.py` wraps
`extract_interface` and `_apex_via_interface_triangles` and runs the full
872-step 3D run for each configuration.

| config | extract_interface calls | interface-triangle set changes (by id) | apex-cache builds | steps where the bulk tet set changed |
|---|---|---|---|---|
| delaunay | 875 | **0** | 1 | 863 / 872 |
| delaunay + remap | 1747 | **0** | 1 | 870 / 872 |

**Reading.**
- Per-step Delaunay does reconnect the bulk on almost every step.
- The 192 interface triangles never change, because the interface vertices
  move smoothly and the simplex phase relabelling keeps the same faces.
- So the apex cache built at setup stayed correct for the whole run. The
  fixed and unfixed code compute identical forces.
- The audit probe's 100 % force error needed a 1e-3 jitter on a refine-1/1
  mesh (10 % of R0). That flips 22 of 48 interface triangles, and the real
  dynamics never do this.
- F1 remains a correctness bug for any run where interface triangles *do*
  flip: larger deformation, adaptive remesh in 3D, merges, and so on. Hence
  the fix and the test.

## 6. Verdict and recommendation

- **Does the fix change the 3D per-step-Delaunay / remap scores?** No. They
  are bit-identical (l2 1.5244561707801316 / tail 0.47066568050584034; l2
  1.8734809234034775 / tail 0.2532690179972299).
- **Does it change the adoption decision?** No.
  - Both Delaunay variants still fail the rule "l2 AND tail at least as good
    as the pin".
  - Pin: 0.24811 / 0.08410. Delaunay: 1.524 / 0.471. Delaunay+remap:
    1.873 / 0.253.
  - Recommend: keep `dual_only` as the 3D default. The laneB/laneE 3D DO-NOTs
    stand, and their evidence is now confirmed to be free of the stale-cache
    artifact.
- **Registry note for the lead** (I did not edit `ddgclib/methods/` or
  METHODS.md):
  - The `curvature_path='integrated'` evidence ("apex cache ... never
    invalidated ... stale under 3D per-step Delaunay") and the
    `oscillating_droplet_3D_delaunay` preset note can now say: fixed (laneI);
    the pre-fix Delaunay scores were unaffected, because the interface
    triangles never changed.
  - `'stokes'` (F2): the coordinate-map staleness is fixed by the same change
    and covered by the test. Its `broken` status could be re-evaluated, but
    it has no dynamic A/B yet.
- The 3D inflation gap (laneG) remains the open lever for a future 3D remap
  retry.

## 7. Files

Repo (new or changed):
- `ddgclib/geometry/_interface_subcomplex.py` (the fix)
- `ddgclib/tests/test_interface_cache_invalidation.py` (new)
- `cases_dynamic/oscillating_droplet/diagnose_3d_remap_afterfix.py` (new driver)
- `cases_dynamic/oscillating_droplet/results_3d/score_delaunay_remap_afterfix.json`,
  `methods_delaunay_remap_afterfix.json`, `snapshots_delaunay_remap_afterfix/`
- `cases_dynamic/oscillating_droplet/results_3d/score.json`, `methods.json`,
  `score_delaunay.json`, `methods_delaunay.json`: rewritten by runs a/b with
  bit-identical scores. `score_delaunay.json` gained the runner's
  `refinement_*` / `retopo_policy` keys.
- This report.

Scratch (`.../scratchpad/laneI/`):
- `prefix_results_backup/`: every `results_3d/*.json` before the runs
- `afterfix/`: copies of the a/b/c score + methods JSONs
- `prefix_b/`: b-prefix score + methods + snapshots
- `probe_invalidation_{delaunay,remap}.json` (+ `_methods.json`)
- `fix.diff`
- logs: `run_{a,b,c,b_prefix}.log`, `battery.log`, `fast.log`,
  `bisection.log`, `probe_*.log`
- helpers: `run_b_prefix.py`, `conftest_prefix.py`,
  `_interface_subcomplex_prefix.py`, `shim_presets.py`,
  `probe_invalidation_count.py`
