# Audit: dual-closure-antisymmetry — fundamental dual-geometry invariants of the single-phase stress operator
> Sources checked: docs_temp/02_physics_foundations.md (§2, §3.1, §6, §7), docs_temp/code_map/operators_stress.md, ddgclib/operators/stress.py, ddgclib/tests/test_stress.py, ddgclib/tests/test_simplex_aware_duals.py, ddgclib/tests/test_integrated_validation.py, hyperct working tree (uncommitted edits) | Written 2026-07-02 by physics-audit workflow

## Scope

Item: for the single-phase stress operator (`ddgclib/operators/stress.py`), on 2D/3D
structured and jittered barycentric-dual meshes:

- (a) closure `sum_j A_ij = 0` for interior vertices,
- (b) antisymmetry `A_ij = -A_ji`,
- (c) linear pressure field ⇒ `stress_force` = `-grad(p) * Vol_i` exactly,
- plus quantification of the boundary-vertex `A_ij` semantics.

Audit ran against the CURRENT working tree, which includes **uncommitted upstream
hyperct edits** (modified: `hyperct/ddg/_compute_dual.py`, `hyperct/ddg/_operators.py`,
`hyperct/ddg/barycentric/_duals.py`, `hyperct/_complex.py`, `hyperct/_vertex.py`, …;
untracked new: `hyperct/ddg/_dual_cell.py`, `hyperct/ddg/_boundary.py`,
`hyperct/_simplicial.py`; last commit `5c708e7 Release v0.3.6`).

## What the physics requires

02_physics_foundations.md §2: the dual-cell boundary "tiles exactly into one dual face
per primal edge ij, with oriented area vector A_ij (outward from i). A_ji = -A_ij exactly
→ Newton's 3rd law pairwise." §3.1: the pressure force "recovers −∫ ∇p dV exactly for
linear p". §6 invariant table: closure and antisymmetry at atol 1e-12
(`test_stress.py::TestDualAreaVector{2D,3D}`), linear-field exactness < 1e-13.
§7 acknowledges: "Boundary edges contribute zero flux (A_ij = zeros) — boundary vertices
have incomplete surface integrals; no boundary-flux closure exists in stress.py."

## What the code does

- 2D standard path `dual_area_vector` (stress.py:158-173): 90°-rotated shared dual edge;
  `< 2` shared dual vertices → `np.zeros(2)` (stress.py:161-163).
- 3D primary path `_dual_area_vector_3d_p_ij` (stress.py:182-277): ring-walk over
  `v_i.vd ∩ v_j.vd`, interleaved with primal-face barycenters; `_angular_sort_3d` retry
  (stress.py:220-229); `_dual_area_vector_3d_e_star` fallback (stress.py:280-308),
  which returns `np.zeros(3)` on `e_star` failure (stress.py:294-299).
- `stress_force` (stress.py:702-775) assembles `pressure_flux` (stress.py:680-682) +
  `viscous_flux` (stress.py:685-699) over `v.nn`.
- `dual_volume` dim=3 (stress.py:355-369): per-edge `hyperct.ddg.v_star` wedge volumes,
  with per-edge `KeyError/IndexError/ValueError` silently skipped (stress.py:367-368).

## Probe design

Scripts under
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/dual-closure-antisymmetry/`
(run with `/home/endres/anaconda3/envs/ddg/bin/python`, `PYTHONPATH=/home/endres/projects/ddgclib`
so the symlinked hyperct is imported):

1. `probe_invariants.py` — four meshes (2D/3D × structured/jittered; jittered = interior
   vertices displaced ±5% of min edge, re-Delaunay, `HC._simplices` cached — the exact
   pattern of `test_stress.py::test_p_ij_linear_precision_jittered_3d` and
   `test_simplex_aware_duals.py::_build_jittered_2d`; structured = raw hyperct
   `triangulate` + `refine_all`×2 connectivity, `compute_vd(method='barycentric')`).
   Measures (a), (b) over ALL edges, (c) with point-valued linear p
   (2D: `p = 3x − 2y + 1`, 3D: `p = 3x − 2y + 5z + 1`, `mu=0`, expected
   `-grad_p * dual_volume(v)`), boundary closure defects, zero-`A_ij` edge census,
   3D e_star-fallback census, uniform-p boundary force.
2. `probe_localize_3d.py` — localizes 3D violations: per-edge path instrumentation
   (monkeypatched counters), linear-precision tensor `T = 0.5 Σ_j d_ij ⊗ A_ij` vs
   `Vol·I` per interior vertex, partition of unity of `dual_volume`.
3. `probe_dual_volume_3d.py` — exact barycentric dual volume from `HC._simplices`
   (each tet contributes `Vol_T/4` per corner; partition of unity exact by construction)
   vs `stress.dual_volume` vs `trace(T)/3`; re-checks invariant (c) against the exact
   volume.
4. `probe_boundary_semantics.py` — `v.vd` content census; 2D wall-edge `A_ij` vs the
   complete geometric face (edge midpoint ↔ triangle barycenter); 2D defect-vs-wall-cap
   identity; 3D face-interior boundary vertices: defect vs exact wall cap
   (barycentric share `A_tri/3` of wall triangles).

## Probe output (key numbers)

`probe_invariants.py` summary (max over interior vertices / all edges):

| mesh | closure (a) | antisym (b) | linear-p (c) vs `dual_volume` |
|---|---|---|---|
| 2D structured (41 v, 104 e) | 0.0 | 0.0 | 4.280e-15 (rel 2.9e-14) |
| 2D jittered | 3.103e-17 | 0.0 | 2.556e-15 (rel 1.7e-14) |
| 3D structured, native refine_all (189 v, 1052 e) | 2.112e-17 | **3.882e-03** | **6.020e-03 (rel 1.43e-01)** |
| 3D jittered, re-Delaunay + `_simplices` (1075 e) | 2.671e-17 | 3.005e-18 | **1.561e-03 (rel 4.5e-02)** |

`probe_localize_3d.py` — the two apparent 3D failures dissected:

- Antisymmetry: **2 of 1052 edges** violate, both **bnd-bnd** (e.g.
  `(0.5,0,0.75)↔(0.75,0,0.75)`, |A_ij+A_ji| = 3.882e-3, both directions on the p_ij
  path), ONLY on the hyperct-native `refine_all` connectivity. After re-Delaunay +
  `HC._simplices` (the `_retopologize` pattern): **0 of 1057** and **0 of 1075** edges
  violate. All edges touching an interior vertex are exactly antisymmetric everywhere.
- Linear-p: the tensor `T = 0.5 Σ_j d_ij ⊗ A_ij` is diagonal and isotropic to
  ≤ 5.2e-18 for EVERY interior vertex on all meshes — the A_ij geometry is exact.
  The (c) residual is entirely `trace(T)/3 ≠ dual_volume(v)`:
  native connectivity 9.766e-04 (= 2⁻¹⁰) even at the deepest interior vertex
  `(0.5,0.5,0.5)` (vol 0.0166 → 5.9% rel); re-Delaunay meshes: deep interior exact
  (2.6e-18) but boundary-adjacent interior off by 2.441e-04 (= 2⁻¹², 4.3% rel).

`probe_dual_volume_3d.py` — `dual_volume` vs exact barycentric volume
(`Σ_{T∋i} Vol_T/4`, cross-checked by `trace(T)/3` to 3.5e-18):

```
3D structured re-Delaunay: sum dual_volume = 0.927992  (deficit 7.20e-02 of 1.0)
   rel err interior max 4.167e-02 median 1.136e-02 | boundary max 5.556e-01 median 1.854e-01
3D jittered  re-Delaunay: sum dual_volume = 0.922163  (deficit 7.78e-02)
   rel err interior max 4.316e-02 median 1.121e-02 | boundary max 6.381e-01 median 1.903e-01
3D native refine_all:     sum dual_volume = 0.937500  (deficit 6.25e-02)
(c') linear-p vs EXACT volume: max abs 6.2e-17 / 1.6e-16, max REL 1.5e-15 / 3.6e-15
2D control: sum dual_volume = 0.968750 both meshes (deficit 3.125e-02 = 1/32,
   all of it in boundary cells; 2D interior areas are exact per the (c) 2D result)
```

**So invariant (c) holds to machine precision (rel ≤ 3.6e-15 in 3D, ≤ 2.9e-14 in 2D)
once measured against the true dual volume.** The `stress_force` operator is exact; the
companion `dual_volume(dim=3)` (stress.py:355-369) systematically under-counts.

`probe_boundary_semantics.py` — boundary `A_ij` semantics:

```
2D: interior v.vd = 8 triangle barycenters only; boundary v.vd = 4 barycenters
    + 2 edge midpoints.  bnd-bnd edges with <2 shared vd: 0 of 16
    → the zeros(2) branch (stress.py:161-163) NEVER fires on standard meshes.
    Q2 max ||A_code − A_full_face|| over wall edges: 0.0 (structured AND jittered)
    Q3 max ||defect + wall_cap||: 6.9e-18 / 1.4e-17
3D: interior v.vd = 24 tet barycenters; boundary v.vd = 19 tet barys + 6 edge
    midpoints (midpoints of its boundary edges).  bnd-bnd edges with <2 shared
    vd: 0 of 311.  e_star fallback: 3 (structured) / 6 (jittered) bnd-bnd edges;
    A_ij == 0 exactly: 0 (structured) / 19 (jittered), all bnd-bnd.
    Face-interior boundary vertices (54): ||defect|| max 8.3e-02 median 6.2e-02;
    ||defect + exact_cap|| median 6.3e-03–1.3e-02, max 7.8e-02–8.2e-02
    (median 9–20% of the defect, worst ~190%).
Uniform p=5 (probe 1): |F| ≤ 2.2e-16 interior, but up to 1.25 (2D) / 0.42 (3D)
    on boundary vertices — the unbalanced −p·(missing cap) force.
```

Corroborating tests in the current tree:
`pytest ddgclib/tests/test_stress.py -k "TestDualAreaVector or p_ij_linear or TestDualVolume"`
→ 13 passed; `test_simplex_aware_duals.py + test_integrated_validation.py -m "not slow"`
→ 59 passed, 1 failed (`TestBoundaryFromSimplices::test_raises_unsupported_dim` —
`boundary_from_simplices` no longer raises `ValueError` for unsupported dim; unrelated
to this item but indicates an uncommitted hyperct behaviour change).

## Verdict and reasoning

**The three audited invariants are CORRECT_AS_INTENDED in the current working tree,
including on jittered meshes and including the uncommitted hyperct edits:**

- (a) closure ≤ 3.1e-17 for every interior vertex, all four meshes;
- (b) antisymmetry exact (≤ 3.9e-18) for every edge with at least one interior endpoint,
  all four meshes;
- (c) `stress_force(mu=0)` = `−grad(p)·Vol_i` to rel ≤ 3.6e-15 for every interior vertex
  when Vol_i is the true barycentric dual volume.

Two side findings (documented here, not the item's headline):

1. **`dual_volume(dim=3)` (stress.py:355-369) under-counts** — a real defect in the
   volume diagnostic, NOT in the force operator: interior rel err up to 4.3%
   (median 1.1%), boundary median ~19%, partition-of-unity deficit 6.3–7.8% of the
   domain. The deficits are exact dyadic rationals (2⁻¹⁰, 2⁻¹²) → wedges are being
   dropped/skipped systematically (silent `continue` at stress.py:367-368 and/or
   incomplete `v_star` wedge enumeration upstream), with a connectivity-dependent
   pattern (deep interior exact on Delaunay+`_simplices`, 2⁻¹⁰ off on hyperct-native
   `refine_all` connectivity). The existing gate
   `TestDualVolume3D::test_partition_of_unity` (test_stress.py:686, rtol 1%) passes only
   because it runs on the unrefined 9-vertex cube. This matches — and provides a
   mechanism for — the documented "single-phase retopology dual-volume-refresh leaks
   ~2-4% volume" open issue (02 §6 caveats).
2. **bnd-bnd edge antisymmetry can break (3.9e-3) on hyperct-native 3D connectivity**
   (no `HC._simplices`): boundary `v.vd` mixes edge midpoints in with tet barycenters,
   so the p_ij ring-walk/`_angular_sort_3d` closes open boundary fans differently from
   the two edge ends. Vanishes entirely under the re-Delaunay + `_simplices` pattern
   that `_retopologize` uses; never affects edges with an interior endpoint. The
   existing regression `TestDualAreaVector3D::test_antisymmetry` (test_stress.py:642)
   skips `v_i in bV`, so this regime is untested by design.

**Boundary-vertex A_ij semantics (exact statement, corrects the code_map):**

- 2D: because barycentric `compute_vd` puts the boundary-edge midpoints into `v.vd`,
  wall edges have 2 shared dual vertices and get the COMPLETE dual face
  (midpoint ↔ barycenter, verified to 0.0). The documented "boundary edge → zeros(2)"
  branch (stress.py:161-163, code_map special-casing table row 2) is **dormant** on
  standard meshes. The single missing surface piece of a boundary parcel is the
  domain-boundary cap through the vertex itself: `defect = −A_cap` to 1.4e-17. A
  boundary parcel's surface integral omits exactly `∮_cap σ·n dS` (no wall-pressure
  reaction, no wall-shear closure) — consistent with the 02 §7 note, and why boundary
  vertices must be frozen/BC-handled.
- 3D: boundary-edge dual faces are neither zero nor exact — the included faces close
  the cell to within only ~80–90% (median) of the true wall cap; a minority of bnd-bnd
  edges silently return `zeros(3)` (19/311 jittered) or use the non-linearly-precise
  e_star fallback (3–6 edges). Incomplete AND (unlike 2D) not cleanly separable into
  "complete interior faces + missing cap".

## Suggested fix (no code changed by this audit)

1. `dual_volume(dim=3)`: when `HC._simplices` is available, compute the exact
   barycentric volume `Σ_{T∋i} |T|/4` (or equivalently `trace(0.5 Σ_j d_ij⊗A_ij)/3`
   from the already-exact area vectors) instead of the `v_star` wedge sum; at minimum,
   log rather than silently `continue` on per-edge exceptions (stress.py:367-368).
2. Add a refined-mesh partition-of-unity regression (refine_all×2 and jittered
   re-Delaunay; current gate covers only the unrefined cube).
3. Either extend `test_antisymmetry` to bnd-bnd edges after documenting the intended
   boundary semantics, or assert/tag that bnd-bnd `A_ij` values are only used behind
   frozen-boundary filters.
4. Update `code_map/operators_stress.md` special-casing table and 02 §7: the 2D
   "boundary edge → zero flux" claim should read "wall-edge dual faces are complete;
   only the wall cap through the vertex is missing (2D); 3D boundary faces are
   approximate/inconsistent (angular-sort closure, e_star fallback, occasional zeros)".

## Droplet / bubble impact

- 2D oscillating droplet: **no impact** from these findings — 2D interior closure,
  antisymmetry, linear-p exactness and interior dual areas are all machine precision;
  interface volumes use the separate multiphase dual-split machinery (out of scope).
- 3D droplet/bubble (EOS-driven): the force operator itself is exact, but EOS density
  uses the under-counting `dual_volume` via `cache_dual_volumes` at retopo
  (`ddgclib/dynamic_integrators/_integrators_dynamic.py:228-229`), `_get_dual_vol` in
  `_resolve_pressure` (stress.py:664) and `ddgclib/eos/_update.py:32`. Because the
  undercount pattern is connectivity-dependent (0 vs 2⁻¹² vs 2⁻¹⁰ per vertex depending
  on triangulation), any retopo edge flip shifts `dual_vol` by O(1e-4–1e-3) against
  bit-frozen mass, which the EOS reads as spurious compression — a plausible
  contributor (larger than the ~1e-8 cospherical-flip jumps cited in 02 §5) to the
  documented #1 open problem "retopology-induced EOS noise". Where masses are
  initialized as `rho0 * dual_vol` with the SAME biased volumes, the bias cancels at
  t=0 and only the retopo-induced pattern change matters.

## Re-verification (same day, later session)

All four probes were re-run against the identical working tree
(hyperct `5c708e7 Release v0.3.6` + uncommitted edits, ddgclib `master` at
`8a41873`; `stress.py` unmodified since `8321c71`). Every headline number
reproduced **bit-identically**:

- (a)/(b)/(c) summary table unchanged (2D closure 0.0 / 3.103e-17, 2D antisym 0.0,
  2D linear-p 4.280e-15 / 2.556e-15; 3D closure 2.112e-17 / 2.671e-17;
  3D bnd-bnd antisym outlier 3.882e-03 on native `refine_all` connectivity only,
  0 violations of 1057/1075 edges after re-Delaunay + `_simplices`).
- (c') vs exact barycentric volume: max abs 6.160e-17 (structured) / 1.601e-16
  (jittered), rel ≤ 3.576e-15.
- `dual_volume(dim=3)` undercount unchanged: partition-of-unity sums 0.9375 /
  0.927992 / 0.921694–0.922163 (jittered sum varies in the last probes only via
  RNG-identical seeds — both runs reproduced); `trace(T)/3` vs exact ≤ 3.469e-18.
- Boundary semantics: 2D wall-face completeness residual 0.0, wall-cap identity
  ≤ 1.388e-17; 3D face-interior defect median 6.2e-02, cap-residual median
  6.26e-03–1.27e-02 (structured/jittered), max ratio 1.66–1.94.
- Pytest: `test_stress.py -k "TestDualAreaVector or p_ij_linear or TestDualVolume"`
  → 13 passed; `test_simplex_aware_duals.py + test_integrated_validation.py
  -m "not slow"` → 59 passed, 1 failed (same unrelated
  `TestBoundaryFromSimplices::test_raises_unsupported_dim`).

Verdict unchanged: CORRECT_AS_INTENDED for the three audited invariants; the
`dual_volume(dim=3)` undercount and the bnd-bnd-only antisymmetry break on
hyperct-native 3D connectivity remain the two side findings.
