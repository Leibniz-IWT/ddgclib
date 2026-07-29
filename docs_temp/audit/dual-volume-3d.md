# Audit: dual_volume 3D — silent per-edge exception skipping + domain tiling
> Sources checked | Written 2026-07-02 by physics-audit workflow

## Item
(a) `dual_volume` (`ddgclib/operators/stress.py:311-372`) dim==3 branch silently
catches `KeyError/IndexError/ValueError` per edge (`stress.py:367-368`) and skips —
suspected silent volume undercount in 3D.
(b) Volume partition: sum of dual volumes over all vertices should tile the domain
(unit cube → 1.0).

## What the physics requires
- `docs_temp/02_physics_foundations.md:29`: the barycentric dual cells tile the domain;
  parcel mass `m_i = rho_i * Vol_i` with Vol_i the dual cell measure.
- `docs_temp/sources/fundamentals.md:96`: "all geometric quantities (**A_ij, Vol_i**)
  **exact** for barycentric dual".
- EOS pressure is `P = eos.pressure(m / dual_vol)` (`stress.py:663-665`), so Vol_i errors
  are directly misread as compression (`02_physics_foundations.md:114`).
- Exact ground truth for barycentric duals (proved below): the barycentric subdivision of a
  d-simplex has (d+1)! equal-volume flag simplices, and vertex i owns d! of them per
  incident top simplex, hence **Vol_i = (1/(d+1)) · Σ_{T∋i} |T|** exactly.
  Verified numerically: 200 random tets, max rel |6-flag-subtet sum − V_T/4| = **3.4e-15**
  (`probe_subdivision_check.py`); in 2D the library's own `dual_cell_area_2d` matches the
  1/3-rule to **6.6e-16** on interior vertices, independently validating the rule.

## What the code does
- **dim==2** (`stress.py:351-353`): delegates to `hyperct.ddg.dual_cell_area_2d`
  (shoelace on duals + edge midpoints, `hyperct/ddg/_dual_cell.py:132-156`).
- **dim==3** (`stress.py:355-369`): per neighbor edge, calls
  `hyperct.ddg.v_star(v, v_j, HC, dim=3)` (`hyperct/ddg/_operators.py:153-236`), which
  fan-walks the shared dual vertices (tet barycenters only — **no face barycenters, no
  edge midpoint hub**) and sums pyramid volumes `(vc_12, vd_i, vd_j; apex v_i)` via
  `volume_of_geometric_object` (`hyperct/ddg/_geometry.py:79-104`). Exceptions are
  swallowed per edge with `continue` (`stress.py:367-368`).
- `cache_dual_volumes` (`stress.py:379-399`) zeroes degenerate vertices; the 3D dynamic
  retopo path refreshes `v.dual_vol` every step
  (`ddgclib/dynamic_integrators/_integrators_dynamic.py:225-229`).

## Probe design
Scripts in `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/dual-volume-3d/`
(run with `/home/endres/anaconda3/envs/ddg/bin/python`, cwd project root):
1. `probe_dual_volume.py` — structured (hyperct native, legacy nn compute_vd) and
   jittered-Delaunay (simplex-aware path, same recipe as
   `test_stress.py::test_p_ij_linear_precision_jittered_3d`) unit cubes, 2D and 3D.
   Instrumented replica of the dim==3 loop (verified bit-identical to library,
   max diff 0.0) counting caught exceptions; monkeypatched
   `volume_of_geometric_object` counting `norm_sq==0` triggers and classifying the base
   triangle (coincident points vs collinear-but-distinct).
2. `probe_ground_truth.py` — exact per-vertex truth `Vol_i = (1/(d+1))Σ|T∋i|` from
   `HC._simplices`; also divergence-theorem volume of the p_ij polyhedron from
   `hyperct.ddg._dual_cell.dual_cell_faces_3d`.
3. `probe_subdivision_check.py` — validates the 1/4-rule against explicit flag subdivision.
4. `probe_keyerror_site.py` — traceback of the swallowed exception + error on affected vertices.

## Probe OUTPUT (decisive numbers)

### (b) Tiling of the unit cube/square, sum over ALL vertices (target 1.0)
| mesh | sum dual_volume | error |
|---|---|---|
| 2D structured ref=2 | 0.968750000000002 | −3.125e-02 |
| 2D structured ref=3 | 0.992187500000003 | −7.813e-03 |
| 2D jittered ref=3 | 0.992187500000003 | −7.813e-03 |
| 3D structured ref=1 (legacy nn) | 0.916666666666667 | **−8.33%** |
| 3D structured ref=2 (legacy nn) | 0.937499999999999 | **−6.25%** |
| 3D near-structured Delaunay ref=2 | 0.925021687344211 | **−7.50%** |
| 3D jittered Delaunay ref=1 | 0.857463254178603 | **−14.25%** |
| 3D jittered Delaunay ref=2 | 0.922162997434080 | **−7.78%** |

2D deficit is EXACTLY the 4 domain-corner cells: signed corner error −7.8125e-03 =
4 × h²/8 (ref=3); interior 2D cells match truth to 6.6e-16; straight-boundary cells exact.
(`dual_cell_polygon_2d` omits the primal vertex from the boundary polygon → corner wedge
lost; O(h²) total, converging.)

### 3D per-vertex vs exact truth (1/4-rule)
| mesh | interior v_star err | boundary v_star err | interior p_ij-polyhedron err |
|---|---|---|---|
| near-structured ref=1 | max rel 4.17%, mean 3.70% | max 38.2%, mean 16.6% | max 11.57%, mean 10.29% |
| near-structured ref=2 | max rel **4.17%**, mean 1.33% | max 44.4%, mean 20.7% | max **11.57%**, mean 3.67% |
| jittered ref=1 | max rel 4.24%, mean 3.70% | max 43.2%, mean 25.7% | max 11.60%, mean 10.27% |
| jittered ref=2 | max rel 4.32%, mean 1.34% | max **63.8%**, mean 21.7% | max 11.60%, mean 3.68% |

**Max relative interior error does NOT converge under refinement** (4.17% at both ref=1
and ref=2): the tb-only fan volume is inconsistent, not merely low-order.

### (a) Silent exception skipping
- Structured meshes (legacy nn compute_vd): **0** skipped edges (308 / 2104 directed edges visited).
- Jittered Delaunay ref=2: **46 / 2150 directed edges skipped**, all `KeyError`, all on
  boundary–boundary edges, raised at `hyperct/ddg/_operators.py:229`
  (`dsetnn_k.remove(vd_i)` in the boundary fan walk). 33 boundary vertices affected.
- Affected vertices: mean rel error 22.20% (max 63.8%) vs unaffected boundary vertices
  21.64% (max 46.8%) — i.e. the skip is a **secondary symptom**; boundary wedges that
  complete the walk are already ~20% wrong (hardcoded `iter_len = 3` at
  `_operators.py:198` truncates rings; the true boundary cell cap through the primal
  vertex is never built).

### norm_sq==0 short-circuit (`hyperct/ddg/_geometry.py:96-97`)
- Structured legacy meshes: 0 triggers in 18 432 calls.
- Jittered Delaunay ref=2: 1488 triggers in 19 284 calls — **all 1488 with exactly
  coincident base points (min pairwise distance < 1e-14), zero collinear-but-distinct
  cases**. It returns 0.0 for genuinely zero-area bases (vc_12 ∈ dset on boundary
  edges); it does **not** hide real geometric degeneracy. The fix is benign.

### Production-path cross-check (skeptic_probe_production.py; re-verified 2026-07-02)
The production 3D dual-volume path used during retopo (`batch_e_star`-based) agrees with
`stress.dual_volume` to max |diff| **1.4e-17** on all interior vertices (ref=1/2,
structured + jittered) — i.e. the errors above are in the real pipeline, not a probe
artifact. Against the exact 1/4-rule truth: max rel **4.167%** (ref=1) and **4.316%**
(jittered ref=2) interior; interior sums low by −3.57% (ref=1) / −1.21% (ref=2).
All decisive numbers in this file were re-run and reproduced bit-identically on
re-verification (tiling table, 46 KeyError skips at `_operators.py:229`, 1488
coincident-point norm_sq==0 triggers, 3.4e-15 flag-subdivision check).

### Existing test coverage gap
`test_stress.py::TestDualVolume3D::test_partition_of_unity` (line 686, rtol=0.01) passes
**only** because the `mesh_3d` fixture (line 597) is the unrefined 9-vertex cube, where
symmetry makes the total exactly 1.0 (probe: err −2.2e-16). One `refine_all()` or any
jitter breaks it by 6–14×  the tolerance. `hyperct`'s legacy `d_area` cross-check in 2D:
total 1.0787 (+7.9%) — worse than `dual_cell_area_2d`, consistent with its documented
deprecation (`_dual_cell.py:138-140`).

## Verdict: CONFIRMED_BUG (3D); 2D is a minor corner-only design limitation
1. **3D `dual_volume` is structurally wrong**, independent of the exception skipping:
   dual cells do not tile the domain (−6% to −14%), boundary cells carry ~20% mean
   (up to 64%) error, and interior cells carry up to ~4.2% error that does not converge.
   Root cause: `v_star`'s volume fan uses tet barycenters only, skipping the primal-face
   barycenters that the true barycentric dual face passes through (the same omission the
   module itself documents for areas at `stress.py:285-291`), plus truncated/unclosed
   boundary cells. Even the linear-precision `dual_cell_faces_3d` p_ij polyhedron gives
   volumes up to 11.6% off, because the ring-spanned polygon is a *different surface*
   than the true dual face (a cone of quads through the edge midpoint) — identical
   boundary ring ⇒ identical area vector A_ij (forces stay exact), but different
   enclosed volume.
2. **(a) confirmed as real but secondary**: the silent `continue` at `stress.py:367-368`
   fires only on Delaunay/simplex-aware meshes (46 edges, KeyError at
   `_operators.py:229`, boundary edges), silently dropping wedges; it is masked by the
   larger structural boundary error.
3. The `norm_sq==0` short-circuit is **NOT_A_BUG** (all triggers are exactly-coincident
   points; returning 0 volume is correct).
4. This contradicts `fundamentals.md:96` ("Vol_i exact for barycentric dual") — the doc
   claim holds for A_ij and for 2D interior cells, not for 3D volumes.

## Droplet / bubble impact
- **2D oscillating droplet / bubble cases: essentially unaffected.** `dual_volume` dim=2
  is machine-exact on interior and straight-boundary cells; only the 4 frozen outer-box
  corner cells are undercounted (75% of a corner cell, O(h²) total).
- **3D oscillating droplet (`oscillating_droplet_3D.py`): directly affected.**
  `rho = m/dual_vol` → EOS pressure (`stress.py:663-665`) is built on volumes with
  1–4% interior / ~20% boundary configuration-dependent errors, refreshed every retopo
  (`_integrators_dynamic.py:225-229`). At t=0 the error cancels (mass initialized from
  the same wrong volume), but every re-Delaunay changes the fan configuration and hence
  the *error*, producing dual_vol jumps of O(1%) against bit-frozen mass — misread by the
  EOS as compression: spurious Δp ≈ K·δV/V ≈ 8 Pa at the softened K_d = 800 Pa, MPa-scale
  at stiff water K. This is a quantitatively much larger version of the documented
  1e-8-scale retopo jump named as the root cause of the historical 3D retopology blow-up
  (`02_physics_foundations.md:114`); the documented one-shot |dV/V|≈0.3 "boundary-shell
  dual_vol zeroing artefact" (`:138`) is consistent with the ~20% boundary errors +
  KeyError skips + `_integrators_dynamic.py:225` zeroing measured here. 3D volume
  conservation diagnostics based on Σ dual_vol under-report by 6–14%.

## Suggested fix
For barycentric duals (pipeline default), replace the dim==2/3 branches of
`dual_volume` with the exact closed form
`Vol_i = (1/(dim+1)) * Σ_{T ∋ i} |T|`
over incident top simplices (from `HC._simplices` when present; else enumerate incident
simplices from `v.nn`). This is exact per vertex to machine eps, tiles the domain
including boundary/corner cells by construction, removes every exception path, and is
cheaper than the fan walk. Keep the v_star path (or a real Voronoi construction) only for
circumcentric duals, and tighten
`test_partition_of_unity` to a refined + jittered mesh with rtol ~1e-12. Optionally fix
the 2D corner cells by inserting the primal vertex into `dual_cell_polygon_2d`'s
boundary polygon.

## Skeptic review
> Adversarial re-verification 2026-07-02 (independent probe, written from scratch:
> `scratchpad/skeptic_dual_volume.py`). **Verdict: claim CONFIRMED, severity HIGH upheld.**

Attempted refutations, all failed:

1. **"Ground truth might be wrong."** The 1/4-rule is exact mathematics (the region of a
   tet where vertex i's barycentric coordinate is maximal has volume |T|/4 by permutation
   symmetry; the 4 regions partition T). Independently cross-checked with a 2M-point
   Monte Carlo on the worst interior vertex (jittered ref=1): 1/4-rule truth 0.046751 vs
   MC 0.046816 (within sampling noise), while `dual_volume` returns 0.044768 (−4.2%).
   Moreover Σ dual_vol ≠ domain volume is truth-independent: any valid dual partition
   must tile, and Delaunay tets themselves tile to 1.000000000000000 on the same mesh.
2. **"Probe misconfigured boundary tagging / compute_vd."** Reproduced with the library's
   own canonical recipes (the exact jittered-Delaunay recipe from
   `test_stress.py::test_p_ij_linear_precision_jittered_3d` incl. `HC._simplices` and
   simplex-aware `compute_vd`, and the standard structured pipeline). Numbers reproduce:
   structured ref=1/2 sums 0.9167/0.9375; jittered ref=1/2 sums 0.8556/0.9217 (different
   RNG draw than auditor, same magnitudes); interior max rel err 4.24%/4.32%.
3. **"Interior error might converge with more refinement."** Extended to jittered ref=3
   (1241 verts): interior max rel err **4.19%** — flat across ref=1/2/3. Non-convergence
   confirmed. (The *total* tiling error does converge ~O(h): 14.4% → 7.8% → 4.2%,
   boundary-dominated — a mild correction to any reading of the tiling table as
   non-converging; the per-vertex interior defect is the non-converging part.)
4. **"Production doesn't use this path."** Refuted: `batch_e_star(compute_volumes=True)`
   (`hyperct/ddg/_operators.py:465-476`, used by `_retopologize` at
   `_integrators_dynamic.py:217-225`) uses the same tet-barycenter-only fan
   (`_walk_fan_3d`) — matches `dual_volume` to max|diff| 2.6e-18 at ref=2/3. The
   `cache_dual_volumes` fallback is also called directly in production/case code:
   `cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py:517`,
   `ddgclib/geometry/domains/_multiphase_droplet.py:166`, `ddgclib/geometry/periodic.py:515`.
   Multiphase treats `dual_volume` as the *authoritative* per-vertex total when rescaling
   phase splits (`ddgclib/geometry/_dual_split_2d.py:493-500`).
5. **"Mass redistribution compensates the EOS impact."** Refuted: `redistribute_mass`
   defaults to `False` in all dynamic integrators, and
   `cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py:116-121` calls
   `symplectic_euler` without `redistribute_mass`/`pressure_model` — retopo dual_vol
   jumps feed the (multiphase) EOS uncompensated.
6. **"Boundary errors moot because retopo zeroes boundary dual_vol."** Partially true for
   the dynamic retopo path (`_integrators_dynamic.py:225` sets boundary dual_vol = 0 by
   design), but the `cache_dual_volumes` path and the direct case/IC usages above assign
   the ~20–64%-wrong boundary values, and the 1–4% non-converging interior error remains
   in every path.
7. **Exception skipping grows with refinement**: 0 (structured), 46/2150 directed edges
   (jittered ref=2), **452/15988 = 2.8%** (jittered ref=3) — all silently swallowed at
   `stress.py:367-368` (KeyError from `dsetnn_k.remove(vd_i)`,
   `hyperct/ddg/_operators.py:229`; same pattern at `:348` in `_walk_fan_3d`).
8. **Coverage gap confirmed**: `TestDualVolume3D::test_partition_of_unity` passes (ran
   it) but only on the unrefined symmetric 9-vertex `mesh_3d` fixture
   (`test_stress.py:597-614`); a single `refine_all()` already yields 0.9167 — 8× the
   rtol=0.01 tolerance.

Strongest evidence: the non-converging ~4.2% interior per-vertex error against an
analytically exact, MC-validated ground truth, produced bit-identically by both the
`dual_volume` and production `batch_e_star` paths, on meshes built with the library's own
test recipe. Severity **high** is calibrated correctly: not critical because the 2D
pipeline (primary production use) is machine-exact away from the 4 domain corners and the
t=0 EOS error cancels (mass initialized from the same wrong volumes), but every 3D
quantitative result depending on Vol_i (EOS density, mass ICs, Σ-volume conservation
diagnostics, multiphase splits, retopo stability) is corrupted, contradicting the
documented "Vol_i exact for barycentric dual" contract.
