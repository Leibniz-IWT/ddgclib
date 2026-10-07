# lane4-remesh-upstream — upstream hyperct remesh fixes (open problem A.4 / 06 §1.5)

Date: 2026-07-02. Sequential lane 4 (after lane1-zero-gauge-pressure,
lane2-eos-consistency, lane3-exact-dual-volumes). All edits are working-tree
only (no git operations), in BOTH repos: the hyperct symlink at
`/home/endres/projects/ddgclib/hyperct` resolves to
`/home/endres/projects/hyperct/hyperct`.

## Verdict

**LANDED.** Fast ddgclib suite has no NEW failures; all production
(delaunay-mode) metrics are BIT-IDENTICAL to the post-lane-3 baseline (the
production pipeline never touches the changed code paths). The A.4 win
condition is met on its primary axes: adaptive mode **no longer
mass-explodes** (documented 9.7 → 187 blow-up now machine-precision flat,
drift 2.95e-15) and **no longer KE-explodes** (tail_growth 4.99 → 1.37 vs
delaunay 1.21). The residual adaptive-vs-delaunay gap is l2 2.73 vs 1.48 on
the 200-step smoke, diagnosed (with probe evidence) as frozen-connectivity
mesh distortion — NOT a remesh-operation defect; see §6.

## 1. What changed (file:line)

### hyperct (upstream)

1. **`hyperct/remesh/_operations_2d.py`**
   - `_SKIP_ATTRS` (~:41): added `"m_phase"` next to `"m"` — extensive
     quantities must never be averaged by `_carry_attrs`.
   - NEW `_EXTENSIVE_ATTRS = ("m", "m_phase")`, `_one_ring_area(v)`,
     `_transfer_extensive(v_m, v_i, v_j, f_i, f_j)` (~:137-215).
   - `edge_split_2d` (~:224): **mass-conservative split.** The midpoint no
     longer gets `0.5*(m_i+m_j)` assigned out of thin air. With barycentric
     duals the midpoint's new dual cell (area `(1/3)(|T_k|+|T_l|)`) is carved
     half from each endpoint's cell, so each endpoint cedes the mass fraction
     `f = (|T_k|+|T_l|)/(2*ring)` (`ring` = endpoint one-ring triangle area,
     measured pre-split). `sum(m)` and `sum(m_phase)` are invariant to
     round-off and a uniform density field stays *exactly* uniform w.r.t. the
     post-split dual areas (test-verified). Midpoint `u` is the
     **mass-weighted** mix of the transferred parcels → `sum(m*u)` invariant,
     KE non-increasing.
   - `edge_collapse_2d` (~:352): (a) upfront abort (return False, zero
     mutation) when the merged midpoint is occupied by a third vertex or
     `v_j` is stale in the cache — fixes the documented "returns True even
     when the final move aborts" defect (old :339-343; code_map upstream bug
     list) and the latent partial-mutation-then-False `HC.V.remove` KeyError
     path; (b) `m` AND `m_phase` are now additive (m_phase previously fell
     through `_carry_attrs` averaging — half the merged per-phase mass was
     silently destroyed); mass kept even when only the removed vertex has it;
     (c) merged `u` is the momentum-conserving `(m_i*u_i+m_j*u_j)/(m_i+m_j)`.
2. **`hyperct/remesh/_driver.py`**
   - NEW `_edge_h_local(v_i, v_j)` (~:67): mean length of all edges incident
     to the two endpoints — per-edge local length scale.
   - `_split_long_edges` / `_collapse_short_edges`: threshold is absolute
     `L_max`/`L_min` when given, else per-edge `alpha_* * _edge_h_local`.
   - `adaptive_remesh`: NEW kwargs `length_scale: str = "local"`
     (`"global"` restores the legacy mesh-wide-median behaviour) and
     `smooth_skip_interface: bool = False` (True pins interface vertices
     during Laplacian smoothing). Explicit `L_min`/`L_max` remain absolute
     regardless of `length_scale` (public API preserved; all existing tests
     pass unchanged).
3. **`hyperct/ddg/_retriangulation.py`**: NEW `rebuild_simplex_cache_2d(HC)`
   (~:155) — rebuilds `HC._simplices` from the post-remesh 1-skeleton with a
   strict point-in-triangle ghost-K3 filter (subdivided-triangle flag-complex
   ambiguity), clearing derived caches exactly like
   `invalidate_simplex_cache`. Exported from `hyperct/ddg/__init__.py`.
4. **`hyperct/tests/test_remesh.py`**: re-pinned 2 assertions that pinned the
   old non-conservative behaviour, with comments documenting old → new:
   `test_split_field_averaging` u `[2,0]` → `[7/3,0]` (+ new endpoint-mass
   assertions `a.m 2→1`, `b.m 4→2`), `test_split_carries_all_fields` u
   `[0,3]` → `[0,3.5]`. The midpoint-mass expectations (3.0, 2.0) held
   because in `_single_quad` both endpoints cede exactly half their mass.
5. **`hyperct/tests/test_remesh_conservation.py`** (NEW, 22 tests): sum(m) /
   sum(m_phase) invariance across single/repeated splits, collapses, and
   split+collapse cycles; uniform-density-stays-uniform exactness;
   momentum conservation + KE dissipation for both ops; collapse
   collision-abort leaves the mesh untouched; mixed-scale mesh (fine disc in
   coarse box, scipy Delaunay) stays within a 1.5x vertex budget over 10
   `adaptive_remesh` calls with driver-level mass conservation;
   local-beats-global comparative; explicit-threshold API invariance;
   uniform-grid no-op; `smooth_skip_interface` pins interface positions;
   `rebuild_simplex_cache_2d` triangle count, ghost-K3 filtering, and
   partition-of-unity of `simplex_dual_volumes` after an op batch.

### ddgclib

6. **`ddgclib/dynamic_integrators/_integrators_dynamic.py`** (~:155-180,
   adaptive branch of `_retopologize` only — delaunay path untouched): after
   `adaptive_remesh`, 2D now calls `rebuild_simplex_cache_2d(HC)` instead of
   `invalidate_simplex_cache(HC)`, and boundary uses
   `boundary_from_simplices` when the cache is present (parity with the
   Delaunay branch). Rationale in §5.
7. **`cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py`**:
   adaptive kwargs switched from absolute global-mean `L_min/L_max` (the
   cross-contamination trap) to local-scale `alpha_min=0.3, alpha_max=2.5`,
   and `smooth_iterations=0` (see §6 smoothing findings). Removed the
   throwaway mesh build previously used to size `h_mean`.
8. **`cases_dynamic/oscillating_droplet/src/_setup.py`**: comment-only update
   (~:207) — the two upstream blockers are fixed; default stays `'delaunay'`.

## 2. Probe / test evidence

- **Mass fix, op level**: new tests pass at rel 1e-12/1e-13 over 25+ splits,
  collapses, and cycles (masses + per-phase masses + m==sum(m_phase)).
- **Mass fix, production**: probe run of the real oscillating droplet under
  `remesh_mode='adaptive'`: total mass 9.6227 bit-stable over 50 steps with
  ops firing (was the 9.7 → 187 blow-up); A/B smoke mass_drift
  2.953601801446362e-15 (identical to delaunay).
- **Local length scale**: mixed-scale probe (95-vertex fine-disc-in-coarse-box,
  10 driver calls): `local` 95 → 106 vertices, `global` (legacy) 95 → **554**;
  mass machine-precision flat in both (probe:
  `scratchpad/wf3/probe_lane4_mixed.py`).
- **Collapse abort**: direct construction test (midpoint occupied) returns
  False with vertex count, connectivity, and masses unchanged.

## 3. Measurement battery (all after final code state; cwd repo root, ddg env)

| Step | Result | Baseline (post-lane-3) | Delta |
|---|---|---|---|
| 1. pinned floor tests | 12 passed | 12 passed | none |
| 2. fast suite | 1 failed, 824 passed, 12 skipped, 17 deselected, 3 xfailed in 49.71s | identical counts | no NEW failures (the 1 failure is the pre-existing `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`) |
| 3. equil summary | 1.1847162859108737e-03 | 1.1847162859108737e-03 | bit-identical |
| 4. osc l2 / tail / linf / summary | 0.48991833470391266 / 1.7250489596305962 / 0.8503543927096853 / 0.7250489596305962 | identical | bit-identical |

hyperct suite: 334 passed, 40 skipped, 6 xfailed, 39 errors (= 312 baseline
+ 22 new tests; the 39 errors are the pre-existing pytest-benchmark fixture
errors). Score copies: `scratchpad/wf3/lane4-remesh-upstream_osc_score.json`,
`_equil_score.json`.

Bit-identical production metrics are expected and verified: the delaunay
retopo path, stress operators, and EOS are untouched; all behavioural
changes are gated behind `remesh_mode='adaptive'` / direct remesh-op calls.

## 4. Adaptive A/B (oscillating_droplet_2D_adaptive.py smoke, 200 steps, refine 2)

Pre-lane, adaptive mode was un-scoreable (documented mass 9.7 → 187 blow-up,
audit: ~164% mass error + KE explosion by step ~80). The middle column below
is the first scoreable state measured mid-lane (mass fix + local scale
landed, legacy smoothing still on, simplex cache still invalidated):

| Metric | Delaunay | Adaptive mid-lane (smoothing on, cache dropped) | Adaptive FINAL |
|---|---|---|---|
| mass_drift | 2.95e-15 | 3.6920e-15 (already fixed) | 2.953601801446362e-15 |
| l2_error_normalized | 1.4803925777850173 | 19.708 (droplet phase destroyed by smoothing churn) | 2.7315517424752116 |
| linf | 2.961825724719834 | 20.993 | 6.723045065982862 |
| tail_growth | 1.2147266483665555 | 2.9253 (4.99 with smoothing off but cache still dropped) | 1.3722156720182621 |
| summary | 1.4803925777850173 | 19.708 | 2.7315517424752116 |
| interface edges | 60 → 62 | 60 → 0 (destroyed) | 60 → 78 (preserved, band thickens late) |
| vertices | 95 → 95 | 95 → 84 | 95 → 88 |

Win condition scorecard: mass-explosion **fixed**; KE-explosion **fixed**;
interface now survives the whole run; l2/summary still 1.8x delaunay — the
remaining gap is NOT a remesh-op defect (§6.3).

## 5. Key diagnosis: the adaptive branch silently downgraded ALL geometry

The single biggest adaptive-mode defect found this lane was not in
`hyperct.remesh` at all: `_retopologize`'s adaptive branch called
`invalidate_simplex_cache(HC)`, which wipes `HC._simplices` — and
`compute_vd`, `boundary_from_simplices`, the edge-apex map, AND lane 3's
exact 2D dual volumes all silently fall back to 1-skeleton flag-complex
paths when the cache is gone. Probe (monkeypatched rebuild, zero remesh ops
firing): KE tail pump 0.29 J (monotone) with the cache dropped vs 0.05 J
(oscillating, delaunay-like) with the cache rebuilt — on the SAME
trajectories otherwise. Fix: `rebuild_simplex_cache_2d` (ghost-K3-filtered
flag enumeration) + simplex-aware boundary in the adaptive branch.
**Lesson for future lanes: any code path that invalidates the simplex cache
without rebuilding it before force evaluation reintroduces the pre-lane-3
geometry errors.**

## 6. Findings for the remaining adaptive-vs-delaunay gap (open, documented)

1. **Laplacian smoothing is incompatible with this Lagrangian solver as-is.**
   With `smooth_iterations=1, relax=0.2` per retopo, parcels are teleported
   without mass remap and the resulting edge-length churn fires ops whose
   retagging (majority-vote `assign_simplex_phases_from_vertices` +
   `min(iph)=0` all-interface fallback in `assign_simplex_phases`,
   multiphase.py:270-288) erodes the droplet: probe showed phase-1 bulk
   25 → 2 vertices and interface 60 → 29 edges in 45 steps; with smoothing
   off the same run is perfectly stable (phases, interface, mass all
   invariant for 50 steps). Bulk-only smoothing (`smooth_skip_interface=True`,
   relax 0.05) trades l2 1.21 for tail 3.02 — the KE pump returns. The
   shipped case config is smoothing-off.
2. **Local ops are NOT the l2 driver**: with `alpha_min=0.05` (zero ops fire,
   n stays 95) l2 = 2.84 vs 2.78 with ops — statistically the same.
3. **The residual l2 ~2.7 vs 1.48 is frozen-connectivity mesh distortion**:
   adaptive mode (correctly) refuses to rewire interface edges and only
   flips for min-angle gain, so late in the run the connectivity lags the
   deformation and the oscillation amplitude overshoots the analytical
   envelope (R_max grows to 0.0141 by step 190; delaunay oscillates
   0.009-0.0115). `max_iterations=3` (more flip sweeps) does not help
   (KE 0.28 at 190). Closing this needs either mass/momentum-remapped
   smoothing (ALE-style) or interface-aware rewiring — a new lane, not a
   remesh-op bug.
4. **Phase-tag hazard for op-inserted vertices**: split midpoints inherit
   `phase` from `src_i` verbatim (can be `INTERFACE_PHASE=-1`); production
   is safe because `_retopologize_multiphase` runs `mps.refresh` before any
   EOS call (lane 2's phase=-1 tripwire never fired in any adaptive run this
   lane), but direct users of `edge_split_2d` on ddgclib meshes must refresh
   before evaluating forces.
5. The `mps=` parameter of `adaptive_remesh` remains reserved/unused.

## 7. Changed files (complete)

hyperct working tree (`/home/endres/projects/hyperct/hyperct/...`, edited via
the ddgclib symlink):
- `remesh/_operations_2d.py`
- `remesh/_driver.py`
- `ddg/_retriangulation.py`
- `ddg/__init__.py`
- `tests/test_remesh.py` (2 re-pinned expectations, documented)
- `tests/test_remesh_conservation.py` (NEW, 22 tests)

ddgclib working tree:
- `ddgclib/dynamic_integrators/_integrators_dynamic.py` (adaptive branch only)
- `cases_dynamic/oscillating_droplet/oscillating_droplet_2D_adaptive.py`
- `cases_dynamic/oscillating_droplet/src/_setup.py` (comment-only)
- `docs_temp/debug_session/lane4-remesh-upstream.md` (this log)

Probes kept in scratchpad `wf3/`: `probe_lane4_mixed.py`,
`probe_lane4_interface.py`, `probe_lane4_ke.py`, `probe_lane4_rebuild.py`,
`probe_lane4_score.py`.
