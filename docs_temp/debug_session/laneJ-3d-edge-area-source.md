# laneJ: 3D edge-area source, e_star cache vs DEC p_ij (audit F4)

Date: 2026-09-25. Measurement lane. No library, test, baseline, preset,
`METHODS.md` or `debugging_plan.md` edits. Repo state: ddgclib `72e8cf6`
(dirty), hyperct `4bb966d` (clean), env `ddg`, cwd repo root.

New files:
- `cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py`: one driver,
  sub-commands `static`, `a5b [--arm]`, `dynamic --arm {cache,pij,pij_simplex}`.
- JSONs in `cases_dynamic/oscillating_droplet/results_3d/`: `laneJ_static.json`,
  `laneJ_a5b.json`, `laneJ_a5b_pij_simplex.json`,
  `score_{pij,pij_simplex,cache_laneJ}.json`, `methods_*.json` and `diags_*.json`
  with the same suffixes, plus `snapshots_{pij,pij_simplex,cache_laneJ}/`.
- Scratchpad `.../scratchpad/laneJ/`: logs, `probe_pij_fallback.{py,json}`,
  `probe_pij_simplex.{py,json}`, `score_cache_laneJ_run1.json` (first cache run;
  its diags dump crashed after the score was saved; that run was re-done and
  is bit-identical).

## Verdict up front

1. **Audit F4 reproduced exactly.** On `box(refinement=2)` and `ball(refinement=2)`,
   the per-edge relative difference between the cache and p_ij has median 0.125 / 0.127
   and max 0.625 / 0.250. The linear-precision off-diagonal residual is
   1.34e-3 / 9.05e-4 on the cache and 1.3e-18 / 4.3e-18 on p_ij.
2. **The cache contributes to the 3D static floor.** The A.5.b floor (refine 2/2,
   20 steps) is 7.274172e-05 on the cache path, reproducing the pin bit-for-bit. On
   p_ij it is 6.283859e-05 (-13.6 %). The frozen-mesh floor (A.5.a, step 0, no cache
   yet) is 6.0153e-05. The retopology-induced excess therefore drops from 1.259e-05
   to 2.69e-06 (-79 %). Most of the "3D retopology bump" of the static floor is the
   non-linearly-precise cache.
3. **Dynamic effect: real, but not a flip.** The full 872-step droplet on p_ij
   scores l2 0.28713 against the pinned 0.24811 (+15.7 %) and tail 0.08832 against
   0.08410 (+5.0 %). It fails the better-l2-AND-tail rule. The sign decomposition
   shows why. p_ij shrinks the early outward bump (q2 +0.245 -> +0.093) and lowers
   the final inflation (R_max(t_end) 1.87 % -> 0.82 % of R0). This exposes more of the
   genuine over-decay (q3 -0.061 -> -0.258, q4 -0.140 -> -0.347). That is exactly
   laneG's cancellation mechanism. The cache is **one contributor to laneG's bump**,
   not the source of the gap.
4. **laneG's 2.6 % closure defect is on the p_ij path, not the cache.** Setup
   meshes have no cache, and laneG probed the setup mesh. The residual comes from
   the library p_ij's face-barycentre heuristic, not from the construction itself.
   `_dual_area_vector_3d_p_ij` picks the face vertex k by "nearest to the midpoint
   of two tet barycentres". On the droplet mesh, 743 of 5193 directed interior
   edges have more common 1-ring neighbours than incident tets (§1c). A p_ij built
   from `HC._simplices` (edge link cycle, the "p_ij_simplex" scratch arm) closes to
   2.3e-16 and is linearly precise to 1.9e-15 at every interior vertex. Dynamically,
   the heuristic's defects are negligible: l2 0.28653 simplex vs 0.28713 library.
5. **Recommendation:** make `edge_area_source` an explicit 3D axis with keys
   `e_star_cache` (default, all pins), `p_ij` (library ring walk, uncached) and
   `p_ij_simplex` (new, link-cycle construction, cached at retopology).
   **Do not change the default now.** Switching re-pins every 3D number and, by
   itself, worsens the pinned l2. Adopt `p_ij_simplex` only together with laneG
   lever (b), the redistribution-pump rework, and judge it on the sign-decomposed
   channels. Details in §5.

## 0. How the arms were built (reproducibility)

All arms use `ddgclib.methods.PRESETS`:

- Task 2 uses `static_droplet_floor_3D`: euler, delaunay, `redistribute_mass=True`.
  It is the same partial as the A.5.b harness.
- Task 3 uses `oscillating_droplet_3D`: symplectic_euler, dual_only, redistribute.
- The p_ij arm is `preset.replace(connectivity='custom')` with
  `custom=no_cache(preset.retopologize_fn(mps=mps))`. The wrapper declares and
  forwards `boundary_filter, merge_cdist, backend, remesh_mode, remesh_kwargs`, the
  exact set `_do_retopologize:414-464` forwards to the base partial. It deliberately
  does NOT declare `skip_triangulation`. The dual_only partial binds it as True, and
  a call-time value would override that binding. After the base call it sets
  `HC._edge_area_cache = None`.
- The `pij_simplex` arm is the same wrapper but fills the cache with the link-cycle
  p_ij (`simplex_pij_cache` in the driver). Edges with an open link, i.e. boundary
  edges, are left out of the cache and fall back to the library path.
- Driver validation (cache arm):
  - A.5.b: the driver arm's 21-value max|F| history is `==` the harness history
    (`run_a5b` called unmodified).
  - Dynamic: `score_cache_laneJ.json` is bit-identical to
    `baselines/baseline_oscillation_3d.json` on every numeric key: l2
    0.24811340819647862, linf 0.5992701359956998, tail 0.08409976059818802, mass
    1.905002320272536e-14, R_max_peak 0.010790236105250779, dual_vol_step0_jump
    0.35704960835509636, dual_vol_drift_post 0.00010683841612299112.

### effective_methods proof (recorded after the first step, in `methods_*.json` extra / `laneJ_a5b*.json`)

| run | `edge_area_source` | `edge_area_cache_present` frames | config.connectivity |
|---|---|---|---|
| A.5.b cache | `batch_e_star_cache` | 20 / 20 | delaunay |
| A.5.b p_ij | `p_ij_ring_3d` | 0 / 20 | custom |
| A.5.b p_ij_simplex | `batch_e_star_cache` (misreport); `edge_area_source_laneJ = p_ij_simplex` | 20 / 20 (simplex cache) | custom |
| dynamic cache | `batch_e_star_cache` | 872 / 872 | dual_only |
| dynamic p_ij | `p_ij_ring_3d` | 0 / 872 | custom |
| dynamic p_ij_simplex | `batch_e_star_cache` (misreport); laneJ tag `p_ij_simplex` | 872 / 872 | custom |

Two reporting artefacts in `effective_methods` show up on custom arms. Neither
changes behaviour.

- `boundary_dual_vol` is keyed on cache presence, so it reports `half_cell` for the
  p_ij arm. The boundary cells are still zeroed: `dual_vol_step0_jump` is identical,
  0.35704960835509636, in all three arms.
- `boundary_rule` is keyed on `connectivity == 'dual_only'`, so it reports
  `boundary_from_simplices` for the custom dual_only arm, where the actual rule is
  carried bV.

## 1. Static accuracy (task 1), `laneJ_static.json`

Metrics per vertex:

- closure = |sum_j A_ij| / sum_j |A_ij|
- off-diag = max |offdiag(M_i)|, where M_i = 0.5 sum_j d_ij (x) A_ij (the
  `test_stress.py` definition)
- tensor = max|M_i - Vol_i I| / Vol_i, with Vol_i = `v.dual_vol`, the exact simplex
  volume
- force = |F_i + g Vol_i| / (|g| Vol_i), from `stress_force` with mu=0, u=0 and
  point-valued p = 5 + g.x, g = (1,2,3)

The force metric mixes closure (times the p offset) with linear precision. On the
droplet it is dominated by closure times offset, so read the tensor column there.

Vertex classes:

- interior = not in bV and no bV neighbour
- boundary-adjacent = not in bV and at least one bV neighbour
- boundary = in bV. These cells are open, so closure is not expected. Both paths
  are identical here because `batch_e_star` only caches interior vertices.

### 1a. box / ball refine 2 after one `_retopologize`

| mesh | per-edge rel diff median / max | class (n) | path | closure med / max | off-diag med / max | tensor med / max | force med / max |
|---|---|---|---|---|---|---|---|
| box | 0.125 / 0.625 | interior (11) | cache | 0 / 0 | 3.7e-4 / 7.3e-4 | 4.6e-2 / 7.1e-2 | 3.0e-2 / 3.9e-2 |
| | | | p_ij | 2.4e-17 / 5.1e-17 | 3.3e-19 / 8.7e-19 | 1.2e-15 / 1.3e-15 | 1.4e-15 / 2.9e-15 |
| | | bnd-adj (80) | cache | 0 / 0 | 4.9e-4 / **1.34e-3** | 6.3e-2 / 1.2e-1 | 3.8e-2 / 1.6e-1 |
| | | | p_ij | 2.7e-17 / 7.9e-17 | 4.3e-19 / **1.3e-18** | 1.1e-15 / 1.4e-15 | 1.7e-15 / 3.4e-15 |
| ball | 0.127 / 0.250 | interior (9) | cache | 2.6e-17 / 4.4e-17 | 7.2e-5 / 7.2e-5 | 1.5e-3 / 1.5e-3 | 2.3e-3 / 2.6e-3 |
| | | | p_ij | 2.9e-17 / 7.0e-17 | 1.7e-18 / 3.5e-18 | 1.4e-16 / 2.9e-16 | 4.7e-16 / 1.5e-15 |
| | | bnd-adj (82) | cache | 3.1e-17 / 8.1e-17 | 7.2e-4 / **9.05e-4** | 3.2e-2 / 5.8e-2 | 3.1e-2 / 9.6e-2 |
| | | | p_ij | 5.1e-17 / 1.3e-16 | 2.4e-18 / **4.3e-18** | 2.0e-16 / 7.4e-16 | 1.0e-15 / 4.4e-15 |

What this shows:

- The cache cells close to machine precision on box and ball but are not linearly
  precise. The tensor error is up to 12 %, and the linear-pressure force error is
  up to 16 % of g Vol_i.
- p_ij is at machine precision on every metric.
- Cache hits: all 1191 / 1166 directed interior edges.

### 1b. Droplet meshes, refine 2/2, after one preset retopology

Mesh: 472 vertices, 96 boundary, 98 interface. The setup mesh carries NO cache
(`edge_area_cache_present_at_setup = False`).

Per-edge rel diff, cache vs p_ij:

- eps=0 (static-floor preset): median 0.159, p90 0.257.
- eps=0.05 (dual_only preset): median 0.162.
- The max is 6.5e16 because 3 of 5193 edges have |A_p| < 1e-12, i.e. degenerate
  co-spherical faces. On those the cache assigns up to 10x the median |A|, which is
  5.5e-6.

| mesh | class (n) | path | closure med / max | tensor med / max |
|---|---|---|---|---|
| eps=0 | interface (98) | cache | **1.08e-2 / 1.74e-2** | 8.4e-2 / 2.5e-1 |
| | | p_ij | 6.7e-17 / 9.1e-3 | 5.4e-16 / 1.3e-2 |
| | bulk droplet (91) | cache | 3.9e-17 / 8.0e-17 | 3.2e-2 / 5.8e-2 |
| | | p_ij | 5.7e-17 / 1.3e-16 | 5.7e-16 / 1.5e-15 |
| | bulk outer interior (107) | cache | 3.6e-3 / 8.0e-3 | 1.5e-1 / 2.0e-1 |
| | | p_ij | 4.4e-3 / 1.5e-2 | 2.1e-3 / 1.5e-2 |
| | boundary-adjacent (80) | cache | 2.4e-17 / 5.7e-17 | 2.8e-2 / 8.9e-2 |
| | | p_ij | 3.1e-17 / **2.6e-2** | 9.1e-16 / 6.0e-2 |
| eps=0.05 | interface (98) | cache | **1.48e-2 / 3.44e-2** | 9.6e-2 / 2.5e-1 |
| | | p_ij | 5.8e-17 / 2.3e-2 | 4.7e-16 / 8.3e-2 |
| | bulk droplet (91) | cache | 3.1e-17 / 8.6e-17 | 2.9e-2 / 6.0e-2 |
| | | p_ij | 4.5e-17 / 2.4e-16 | 4.5e-16 / 1.0e-15 |
| | bulk outer interior (107) | cache | 4.2e-3 / 9.9e-3 | 1.4e-1 / 2.3e-1 |
| | | p_ij | 6.3e-3 / 1.7e-2 | 1.1e-2 / 1.1e-1 |
| | boundary-adjacent (80) | cache | 2.3e-17 / 5.7e-17 | 2.8e-2 / 8.9e-2 |
| | | p_ij | 3.1e-17 / 2.6e-2 | 9.1e-16 / 6.0e-2 |

Reading:

- On the cache path, every interface vertex fails closure at about 1 % (the median
  is 1e-2), and every class is 3-25 % off linear precision.
- On p_ij, most vertices are machine-clean. A minority of outer and interface
  vertices do not close, up to 2.6 %. That is laneG §1.5's number: laneG measured
  the no-cache (p_ij) path.

### 1c. Why library p_ij fails closure on the droplet (scratch probes)

Source: `probe_pij_fallback.json`.

- At eps=0, 100 non-boundary vertices do not close: 72 bulk outer, 24 interface,
  4 outer that are boundary-adjacent.
- The ring walk itself never falls back. What fails is the face-barycentre
  matching. On 743 of 5193 directed edges, the number of common 1-ring neighbours
  exceeds the number of incident tets. The extra common neighbours come from
  non-face 3-cycles of the graph.
- So the "nearest face barycentre to the midpoint of two tet barycentres" rule can
  pick a vertex that does not span a face.

Source: `probe_pij_simplex.json`.

- Take the same interleaved polygon but read the ring order and the face vertex k
  from the edge's link cycle in `HC._simplices`.
- At the 296 interior (not boundary-adjacent) vertices this gives closure max
  1.7e-16 / 2.3e-16 and tensor max 1.9e-15 / 1.9e-15 (eps 0 / 0.05).
- Library p_ij gives closure max 1.5e-2 / 2.3e-2 and tensor max 1.5e-2 / 1.1e-1.
- The construction is sound. The heuristic is the defect.

## 2. 3D static droplet floor A.5.b (task 2)

Setup: refine 2/2, eps=0, 20 steps, u := 0 every step, dt = 7.9e-5 (harness
formula). Step 0 is the setup mesh, so every arm starts on p_ij.

| arm | step 0 | peak | end (step 20) | excess over A.5.a 6.0153e-05 | mass drift | wall / step (excl. callback) |
|---|---|---|---|---|---|---|
| harness `run_a5b` (cache) | 6.015320e-05 | **7.274172178727e-05** | 7.274172178684e-05 | 1.259e-05 | 5.6e-16 | (11.9 s total) |
| driver, cache | 6.015320e-05 | 7.274172178727e-05 (== harness) | 7.274172178684e-05 | 1.259e-05 | 5.6e-16 | 0.530 s |
| driver, p_ij | 6.015320e-05 | **6.283858835e-05** | 6.283858835e-05 | **2.69e-06 (-79 %)** | 5.6e-16 | 2.063 s (3.9x) |
| driver, p_ij_simplex | 6.015320e-05 | 6.283860687e-05 | 6.283860687e-05 | 2.69e-06 | 5.6e-16 | 1.862 s (3.5x) |

All arms are flat from step 1. p_ij reaches its plateau at step 1, like the cache.
The p_ij and p_ij_simplex floors agree to 3e-7 relative, so the heuristic defects
of §1c do not touch the interface-force maximum.

## 3. Full 3D droplet (task 3): refine 2/2, dual_only, 872 steps

dt = 1.8356e-4. Record cadence, score, `total_dual_vol` and seed are as in the
runner. The 3D score has no two-fluid reference: `add_two_fluid_reference` uses the
2D dispersion only.

| channel | pin (baseline_oscillation_3d.json) | cache (driver) | **p_ij** | p_ij_simplex |
|---|---|---|---|---|
| l2 (= summary) | 0.24811340819647862 | 0.24811340819647862 | **0.2871269636993966** (+15.7 %) | 0.2865275192675471 (+15.5 %) |
| linf | 0.5992701359956998 | same | 0.5625157534128178 | 0.5624675954878644 |
| tail_growth | 0.08409976059818802 | same | **0.08831881396937855** (+5.0 %) | 0.08818323801034324 |
| mass_drift | 1.905e-14 | same | 5.68e-15 | 1.80e-14 |
| R_max_peak | 0.010790236105250779 | same | 0.010771858913959338 | 0.01077183483499686 |
| R_max(t_end) | (0.010187) | 0.010187 | **0.010082** | 0.010083 |
| dual_vol_step0_jump | 0.35704960835509636 | same | 0.35704960835509636 | same |
| dual_vol_drift_post | 1.068e-4 | same | 5.57e-5 | 5.56e-5 |
| KE_max [J] | 1.7432e-06 | 1.7432e-06 | 1.7052e-06 | 1.7049e-06 |
| n_interface | 98 / 98 | same | 98 / 98 | 98 / 98 |
| boundary_saturation | false | false | false | false |
| quarter mean err q1 / q2 / q3 / q4 | (laneG: +0.319 / +0.262 / -0.056 / -0.139) | +0.325 / +0.245 / -0.061 / -0.140 | **+0.279 / +0.093 / -0.258 / -0.347** | +0.279 / +0.093 / -0.257 / -0.346 |
| wall total / per step | - | 432 s / **0.496 s** | 1788 s / **2.050 s (4.1x)** | 1648 s / 1.890 s (3.8x) |

(The first, crashed cache run was 457 s at 0.524 s/step. All three arms of the
re-run ran concurrently on a 32-core box, so the per-step ratios are the
comparable numbers.)

Reading:

- The accurate A_ij moves the whole trajectory down by about 1e-4 m (about 1 % R0)
  after the first quarter. The early bump shrinks, most visibly q2, and the final
  inflation more than halves.
- Because the pinned l2 is a bump(+) / over-decay(-) cancellation (laneG §2),
  removing part of the bump makes l2 worse. The same happened with laneG's
  redistribution-OFF arm: l2 0.26636, better physics channels.
- p_ij does not remove the R_max overshoot: the peak barely moves, -0.17 %.
  Removing the overshoot is what redistribution OFF did. The two levers act on
  different parts of the bump.
- Library vs simplex p_ij differs by 0.2 % in l2. The heuristic is irrelevant to
  the dynamics at this resolution.

## 4. Cost

Retopology plus force evaluation per step on the 472-vertex mesh:

| path | A.5.b s/step | dynamic s/step | ratio vs cache |
|---|---|---|---|
| e_star cache (`batch_e_star`, vectorised) | 0.530 | 0.496 | 1x |
| p_ij library (per-edge Python ring walk, uncached, computed twice per edge: once from each endpoint) | 2.063 | 2.050 | 3.9-4.1x |
| p_ij_simplex (scratch Python cache builder, once per retopology) | 1.862 | 1.890 | 3.5-3.8x |

A vectorised `p_ij_simplex` in hyperct would be a batch over the `_simplices`
array, the same shape of computation as `batch_e_star`. It should cost about the
same as the cache. The scratch builder's cost is pure-Python overhead, not an
inherent cost of the construction.

## 5. Verdict and recommendation

- **Static floor:** the e_star cache IS a contributor. It carries 79 % of the
  retopology excess of the pinned 3D A.5.b floor (7.274e-5 -> 6.284e-5 on p_ij).
  What remains, 6.28e-5 vs 6.02e-5, is the curvature stencil plus redistribution,
  as laneG found.
- **Dynamic gap:** the cache is a secondary contributor to laneG's outward bump.
  It accounts for about 60 % of the q2 bump and about half of the final inflation.
  It is not a contributor to the over-decay. Switching alone fails the flip rules:
  l2 +15.7 %, tail +5 %. That is the laneG cancellation, not a regression in
  physics; mass, dual_vol drift and the final-radius error all improve.
- **Closure defect (laneG §1.5):** this is the library p_ij face-matching
  heuristic, not the cache. The `p_ij_simplex` construction removes it
  (2e-16 / 2e-15).

Recommended axis (`edge_area_source`, 3D; promote from reported to explicit):

| key | meaning | status | default? |
|---|---|---|---|
| `e_star_cache` | today's `batch_e_star(orient=True)` cache filled at retopology | validated (all 3D pins); NOT linearly precise | **yes, keep** (no default change this lane) |
| `p_ij` | today's uncached `_dual_area_vector_3d_p_ij` (ring walk + nearest-midpoint face matching) | opt-in; linearly precise on box/ball, up to 2.6 % non-closure on the droplet mesh; 4x cost | no |
| `p_ij_simplex` | p_ij polygon from the edge link cycle in `HC._simplices`, cached per retopology (needs a hyperct batch implementation) | experimental; machine-precise closure and linear precision on all interior cells here | **target default** once implemented in hyperct, re-pinned, and co-evaluated with laneG lever (b) |

2D keys stay as reported (`shared_vd_2d`, `min_image_2d`). Also:

- `velocity_laplacian` bypasses the cache (`gradient.py:91`, audit A11). It should
  follow the same axis.
- `effective_methods` should read the explicit field instead of inferring from
  cache presence. That also fixes the two misreports in §0.

Do-not list for the next lane:

- Do not flip 3D to p_ij in isolation on the l2 score. Judge it on the
  sign-decomposed channels, together with the redistribution rework.
- Do not re-pin from these runs.
- Do not treat laneG's 2.6 % closure number as a cache defect.

## 6. Commands

```
PY=/home/endres/anaconda3/envs/ddg/bin/python
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py static
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py a5b                    # harness + cache + pij
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py a5b --arm pij_simplex
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dynamic --arm {cache,pij,pij_simplex}
$PY <scratchpad>/laneJ/probe_pij_fallback.py ; $PY <scratchpad>/laneJ/probe_pij_simplex.py
```
