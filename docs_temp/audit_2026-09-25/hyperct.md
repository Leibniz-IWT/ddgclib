# hyperct audit (as used by ddgclib), 2026-09-25

Scope: `/home/endres/projects/hyperct` working tree (HEAD `5655d31`, 2026-07-29) including all uncommitted and untracked work, cross-checked against ddgclib (`/home/endres/projects/ddgclib/ddgclib`, excluding `cases_mean_flow/` and `benchmarks/`). Read-only: nothing in either repo was modified. Paths below are relative to `/home/endres/projects/hyperct/hyperct/` unless prefixed with `ddgclib/` (= `/home/endres/projects/ddgclib/ddgclib/`). Probe scripts used for empirical checks live next to this report (`probe_sc.py`, `probe_dbuilder.py`, `probe_clique.py`).

---

## 0. TL;DR

1. **Test status.** hyperct: 336 passed, 40 skipped, 6 xfailed, 39 errors (all 39 = known `benchmark` fixture missing). With `-k "not benchmark"`: 325 passed, 40 skipped, 50 deselected, 6 xfailed, **0 failures**. New-layer tests: `test_ops` 11 pass, `test_simplicial` 13 pass, `test_simplicial_sync` 5 pass, `test_simplicial_gpu` 6 pass + 2 skip, `test_gpu` 17 pass + 23 skip. **torch is not installed in the `ddg` env, so every torch/GPU path (old and new) is untested here.** ddgclib fast suite against the current working tree: **881 passed, 12 skipped, 2 xfailed, 0 failed** (the 2026-07-02 baseline failure `test_raises_unsupported_dim` no longer fails).
2. **ddgclib already depends on uncommitted hyperct code.** `ddgclib/dynamic_integrators/_integrators_dynamic.py:234-237` calls `batch_e_star(..., orient=True, compute_volumes=True)`; `orient` exists only in the uncommitted `ddg/_operators.py` (HEAD signature: `def batch_e_star(vertices, HC, dim=3, backend=None, compute_volumes=False)`). Against committed hyperct, `_retopologize` raises an uncaught `TypeError` in 2D and 3D (only `ImportError/NotImplementedError` are caught, `:277`). **This hunk must be committed.**
3. **Nobody in ddgclib uses the new `SimplicialComplex` / `_ops` layer** (0 hits for `SimplicialComplex`, `hyperct._ops`, `_simplicial`, `simplicial=`, `.SC`, `get_builder/get_refiner/get_local_op` in `ddgclib/`). With `simplicial=False` (the default) it is behaviour-neutral (verified by both test suites).
4. **The SC layer is NOT safe to turn on under ddgclib's pipeline.** Verified empirically: once an SC exists, (a) every later `connect_and_cache_simplices` is silently ignored and `HC._simplices` keeps returning the OLD triangulation, (b) `invalidate_simplex_cache` and `rebuild_simplex_cache_2d` do not take effect, (c) an `edge_collapse_2d` leaves holes in the SC (boundary 20 vertices vs true 16), (d) with the `DelaunayBuilder`, merely *reading* `HC._simplices` re-runs Delaunay and adds edges without disconnecting (53 -> 56 edges, non-planar). The hyperct docs (DEVELOPMENT/FEATURES/ARCHITECTURE) claim "Complete" and "kept in sync under interleaved operations"; that overclaims.
5. **Remesh:** 2D only (`remesh/_driver.py:314-315`). Lane-4 conservation fixes (split, collapse, and also flip) are committed in `59f2941` and verified in code; ddgclib's `remesh_mode='adaptive'` is wired correctly (`adaptive_remesh` then `rebuild_simplex_cache_2d` then `boundary_from_simplices`). 3D remesh is entirely unimplemented.
6. **Env footgun:** the `ddg` env has a stale non-editable hyperct **0.3.5** wheel in `site-packages` (Feb 2026, no `remesh/`, no `_retriangulation`, no `_simplicial`). The live source is only picked up when cwd (or `sys.path[0]`) is the ddgclib root (symlink shadowing). A script run from another directory, e.g. `python cases_dynamic/<name>/run.py`, silently imports 0.3.5 (reproduced: `ImportError: cannot import name 'connect_and_cache_simplices'` when run from the scratchpad).

Recommendation (section 7): commit the load-bearing, backward-compatible hunks (`_operators.py` orient, `_dual_cell.py` walk, `_curvature.py` `HC=`, `_backend.py` heron + `test_gpu.py`) to master now; park the SC/_ops layer (+ `_complex.py`/`_vertex.py` hooks, circumcenter/sparse backend kernels, docs) on a feature branch until the four sync bugs in section 3.4 are fixed.

---

## 1. Uncommitted change inventory

### 1.1 Modified tracked files (`git diff --stat`: 649+, 32-)

| File | What it does | Complete? | Tests | ddgclib uses it? |
|---|---|---|---|---|
| `ddg/_operators.py` (+34) | `batch_e_star(..., orient=False)` new kwarg (`:358-359`, doc `:381-394`). When `orient=True`, per-directed-edge fan triangles are sign-flipped so they point away from `v_i` (flip if `A_tri . (x_i - mid) > 0`) and summed to one `(3,)` vector (`:486-498`); primal positions now collected for `orient` too (`:439-440`). Empty edge returns `np.zeros(3)` (`:490`). | Yes (3D only; `dim != 3` raises `NotImplementedError` `:403-404`) | No direct hyperct test of `orient=True` found; exercised indirectly by ddgclib 3D tests | **Yes, load-bearing** (`_integrators_dynamic.py:234-237`). HEAD lacks it -> TypeError. |
| `ddg/_dual_cell.py` (+101) | `dual_cell_polygon_2d` now orders by walking primal-edge adjacency (`_dual_cell_polygon_2d_walk`, `:118-179`), dedups coincident points, orients CCW by shoelace; old angular sort kept as fallback (`:182-216`) when the walk cannot close (boundary / degenerate). Fixes wrong areas for circumcentric cells whose circumcenter lies outside an obtuse triangle (non-star-convex). | Yes | Covered by existing `test_ddg.py` dual-cell tests (pass); no new dedicated test for the obtuse case seen | Yes: `ddgclib/analytical/_integrated_comparison.py:79,134`, `ddgclib/operators/stress.py:383-384` (2D fallback when no simplex cache or circumcentric), `ddgclib/geometry/_dual_split_2d.py:97`. Barycentric interior cells are star-convex, so results unchanged there; circumcentric results change (correctly). |
| `ddg/_curvature.py` (+22/-?) | `normal_area`, `mean_curvature`, `integrated_curvature` gain `HC=None` (`:57, :125, :205`) and use committed `apex_vertices(HC, vi, vj)` (`:84, :157, :238`). `HC=None` == legacy `vi.nn & vj.nn`. | Yes, backward compatible | Existing curvature tests pass | No (ddgclib has its own `_curvatures_heron.py` using `get_edge_apex_map`). |
| `_backend.py` (+301) | Protocol + NumPy/MP/Torch impls of: `batch_heron_curvature` (`:146-172`, numpy `:348-368`, mp `:447-469`, torch `:665-695`), `batch_circumcenters` (`:174-188`, CPU kernel `:224-243`, torch `:697-716`), `build_sparse_boundary` / `batch_boundary_apply` / `batch_coboundary_apply` (`:190-220`, CPU CSR kernels `:246-289`, torch `torch.sparse` `:718-750`). | Code complete | `test_gpu.py` (+72, heron numpy + torch parity), `test_simplicial_gpu.py` (circumcenters, sparse). Torch variants **skipped** (no torch). | `batch_heron_curvature`: yes, opportunistically via `hasattr` (`ddgclib/_curvatures_heron.py:778-779`). `batch_circumcenters` and sparse ops: **no library caller at all** (only `_simplicial.py:287` uses `build_sparse_boundary`); `compute_vd` batch path still uses `batch_dual_positions`. |
| `_complex.py` (+101) | `Complex(..., simplicial=False)` (`:109`); `_simplicial/_simplices_raw/_SC` init + hook install (`:268-288`); `_simplices` becomes a **property** (getter `:298-308`, setter `:310-317`), `SC` property (`:319-322`), `_sc_notify` (`:324-339`); events in `split_edge` (`:1498`) and `connect_vertex_non_symm` (`:2065`); `boundary()` warning docstring + dispatch to `SC.boundary()` when `V is None and _SC is not None` (`:2317-2318`). Imports `SimplicialComplex` at module load (`:66`). | Partially (see 3.4) | `test_simplicial*.py`, `test_ops.py` | Only the property itself (transparent when off). |
| `_vertex.py` (+50) | `VertexCacheBase._sc_hook` (`:256`); `move` emits `vertex_moved` (`:303-304`); `remove` emits `vertex_removed` (`:325-326`); `merge_nn` / `merge_all` suspend the hook during batch and emit one `merge` (`:379-391`, `:441-452`). | Yes | sync tests | No (hook is `None` when off). |
| `tests/test_gpu.py` (+72) | Heron kernel tests. | Yes | 2 numpy pass, 2 torch skipped | n/a |

### 1.2 Untracked new files

| File | Lines | Content |
|---|---|---|
| `_simplicial.py` | 389 | `SimplicialComplex`: `(N, dim+1)` int32 `_top` into private `_vtable` snapshot; `from_object_tuples` (`:65-84`), `from_index_array` (`:86-100`); lazy `faces(k)` (`:161-184`), `cofaces(k)` (`:186-208`, Python loop), exact `boundary()` (`:210-227`), `object_view()` (`:229-245`), graded `boundary_operator(k)` (`:258-291`), `apply_boundary/apply_coboundary` (`:299-307`); mutations `add_simplex/remove_simplex/remove_vertex` (`:312-349`); `on_event` (`:351-370`), `mark_dirty` (`:372-376`), `_resolve` (`:121-134`), `rebuild_from_nn` (`:378-389`). |
| `_ops/__init__.py` | 50 | Re-exports. |
| `_ops/_builder.py` | 207 | `enumerate_top_cliques` (`:27-62`), `_clique_rebuild` (`:65-78`), `_attach_sc` (`:81-87`); `HypercubeBuilder` (`:102-111`), `DelaunayBuilder` (`:114-151`), `ManualBuilder` (`:154-165`); registry/factory (`:183-207`). |
| `_ops/_refiner.py` | 129 | `GenerationRefiner`, `SplitGenerationRefiner`, `StarRefiner`, `LocalSpaceRefiner` (all `_resync` = `mark_dirty`), `AdaptiveRemesher` (`:89-101`); registry (`:104-129`). |
| `_ops/_local.py` | 129 | `OpContext` dataclass, `LocalOp` protocol, `_BaseLocalOp.sc_update`, wrappers for `split_edge`, `connect_vertex_non_symm`, `edge_split_2d/collapse/flip`; registry. **Note:** the wrappers' `sc_update` is never called by `apply` (e.g. `:66-68`, `:81-83`), so `OpContext` is dead scaffolding; sync relies solely on the events fired inside the wrapped functions. |
| `tests/test_ops.py`, `test_simplicial.py`, `test_simplicial_sync.py`, `test_simplicial_gpu.py` | 113/231/132/104 | See 3.5 for coverage gaps. |
| `ARCHITECTURE.md`, `DEVELOPMENT.md`, `FEATURES.md`, `READTHEDOCS_SETUP.md`, `.readthedocs.yaml`, `docs/` | | Agent-written docs; Sphinx scaffold (`docs/_build/` is already gitignored). |
| `pytest-of-stefan_endres/` | | Junk pytest tmp dirs; delete, do not commit. |

### 1.3 Committed-but-anticipating code (important context)

`59f2941` and `5655d31` already contain code that talks to the uncommitted SC layer, guarded so HEAD is safe:
- `remesh/_operations_2d.py:304-311, 479-480, 650-654` call `HC._sc_notify(...)` only `if getattr(HC, "_SC", None) is not None`.
- `ddg/_retriangulation.py:169-171, 239-241` mark `HC._SC` dirty via `getattr`.
- `ddg/_compute_dual.py:115-118, 928-998` (`_compute_vd_nd_simplex_aware`), `_retriangulation.py:305-325` (`apex_vertices`), `:174-242` (`rebuild_simplex_cache_2d`) are committed.

So HEAD is internally consistent (no `AttributeError`), but the SC design was partly baked into committed code.

### 1.4 Doc claims vs code (overclaims)

| Claim | Reality |
|---|---|
| DEVELOPMENT.md:4 "Status: Complete"; FEATURES.md:17-18 "Kept in sync under interleaved vertex operations (insert / remove / move / merge / remesh)" | False for any SC created through the `_simplices` setter (i.e. via `connect_and_cache_simplices`, the ddgclib path): no `_rebuild_fn`, so `mark_dirty` just clears the flag and keeps stale rows; collapse leaves holes; later retriangulations are ignored (section 3.4). |
| FEATURES.md:19-20 "Populated by ... Delaunay (`connect_and_cache_simplices`)"; ARCHITECTURE.md:325-327 "so `connect_and_cache_simplices` and every simplex-aware ddg path work unchanged" | Only the FIRST call populates; subsequent calls are silently dropped (setter `:316` requires `self._SC is None`). |
| DEVELOPMENT.md:14 "extend `invalidate_simplex_cache` to notify SC" | It marks dirty but `HC._simplices` still returns the stale view (verified: 32 stale triangles after invalidate). |
| `_ops/_refiner.py:92-95` "remesh local ops keep HC.SC in sync incrementally (split/flip) or mark it dirty (collapse)" | Event-wise true; outcome-wise collapse corrupts a setter-built SC (holes). |
| ARCHITECTURE.md:346-360 GPU table all ticks | Torch column never executed in this env; `batch_circumcenters` / `batch_heron_curvature` unused by hyperct itself. |
| FEATURES.md:3 "v0.3.4"; "BatchBackend Four operations" (`:93`); curvature "Known bugs" (`:154-159`) | Stale: package is 0.3.6, protocol has 11 methods, curvature bugs were fixed per DEVELOPMENT.md Phase 4. |
| ARCHITECTURE.md line counts (`_compute_dual.py ~259`, `_backend.py ~402`, `_operators.py ~228`) | Actual 997 / 809 / 512. |
| `test_simplicial.py:200-231` `TestFlagComplexBug` "documents the legacy bug" | Never asserts that legacy `Complex.boundary` differs; only checks SC. `test_ops.py:16-26` compares `H.boundary()` against `boundary_from_simplices(H)`, both derived from the same SC list (tautological). |

---

## 2. Mesh / dual method axes (config-wrapper schema)

Legend: **C** = committed at HEAD, **U** = uncommitted, **T** = has hyperct tests, **D** = actually exercised by ddgclib (non-test code).

### A1. Mesh representation
| Value | Where | Status | ddgclib |
|---|---|---|---|
| `Complex` flag complex (vertex cache + `v.nn`), optional raw `HC._simplices` list (default) | `_complex.py:106`, `_vertex.py:230-470` | C, T | **D (only value used)**: every domain builder does `Complex(dim, domain=...)` (`ddgclib/geometry/domains/_rectangles.py`, `_boxes.py:49`, `_disks.py:42,115`, `_cylinders.py:66,163`, ...). |
| `Complex(dim, simplicial=True)` + `SimplicialComplex` | `_complex.py:109, 268-339`, `_simplicial.py` | U, T | Not used. Note: `simplicial=True` alone does nothing for hypercube meshes; SC is attached only by `_ops` builders or the `_simplices` setter. |

### A2. Initial construction (builder)
| Value | Where | Status | ddgclib |
|---|---|---|---|
| Hypercube cyclic-product `triangulate()` + `refine_all()` (Kuhn-type; flag complex is exact here: verified clique emission tiles the unit square/cube to 1.000000 for 0-2 refinements, `probe_clique.py`) | `_complex.py:554, 774-818` | C, T | **D** (all domain builders; `_simplices` stays `None` until first retopology). |
| Delaunay point cloud via `connect_and_cache_simplices` | `ddg/_retriangulation.py:52-147` | C, T | **D** (`_integrators_dynamic.py:205-207`, `periodic.py:470-475`, `multiphase.py:233-237, 272-276`, `_multiphase_droplet.py:117`, `DomainResult.retopologize` `_result.py:46-89`). |
| `vf_to_vv` (explicit vertices/faces) | `_complex.py:2759` | C | Not in ddgclib core. |
| `_ops` `get_builder("hypercube"|"delaunay"|"manual")` | `_ops/_builder.py:183-200` | U, T | Not used. |

### A3. Primal (re)triangulation per step (`remesh_mode`)
| Value | Where | Default | Status | ddgclib |
|---|---|---|---|---|
| `'delaunay'`: disconnect all edges, scipy Delaunay, cache simplices. 3D input lexsorted (canonical order), 2D not (`_retriangulation.py:116-130`); qhull fallback `"Qbb Qt Qz"` (`:127-129`); dims 2,3 only (`:94-97`); 1D chain in ddgclib (`_integrators_dynamic.py:197-201`) | ddgclib `_retopologize` `:190-215` | **default** (`:53`) | C, T | D |
| `'adaptive'`: `adaptive_remesh` (2D) then `rebuild_simplex_cache_2d` (2D) / `invalidate_simplex_cache` (else) then boundary | ddgclib `:156-189`; hyperct `remesh/_driver.py:234-402` | opt-in; forwarded through all 5 integrators (`:470-475, 673-675, 986-990, 1076-1080, 1194-1198, 1307-1311, 1420-1424`) | C, T | D |
| frozen (`skip_triangulation=True`): keep connectivity, reuse `bV`, recompute duals | ddgclib `:216-219` | used by `euler_velocity_only`, multiphase measurement pass (`:652-653`) | C | D |
| periodic ghost Delaunay (`periodic_axes`) | `ddgclib/geometry/periodic.py:455-506` | when `periodic_axes` set (`_integrators_dynamic.py:125-132`) | C | D |
| pre-step `merge_cdist` merge (`HC.V.merge_all`) | `_vertex.py:394-452`; ddgclib `:148-154` | `None` | C | D (see footgun F12) |

### A4. Simplex cache source / freshness
| Value | Where | Status | ddgclib |
|---|---|---|---|
| none (`HC._simplices is None`): all consumers fall back to flag-complex paths | | C | D (structured initial mesh; after BC inject/delete) |
| populated by `connect_and_cache_simplices` | `_retriangulation.py:143-147` | C, T | D |
| rebuilt from 1-skeleton, 2D, ghost-K3 filtered | `rebuild_simplex_cache_2d` `_retriangulation.py:174-242` | C (tested indirectly via ddgclib adaptive tests) | D (`_integrators_dynamic.py:179-180`) |
| invalidated | `invalidate_simplex_cache` `_retriangulation.py:150-171` (clears `_simplices`, `_edge_to_apex`, marks SC dirty) | C | D (`_boundary_conditions.py:435, 507, 694`; `_multiphase_droplet.py:125`; adaptive 3D branch `:182`) |
| SC object view | `_simplicial.py:229-245` | U | not used |

Consumers that silently switch path on this axis: `compute_vd` (2D `:159`, 3D `:320`, ND `:115`, batch `:522, :631`), `boundary_from_simplices` (raises if None, `_boundary.py:62-70`), `simplex_dual_volumes`/`vertex_dual_volume` (raise, `_dual_volume.py:29-38`), `get_edge_apex_map`/`apex_vertices` (fallback), ddgclib `_use_exact_barycentric_volume` (`stress.py:311-324`).

### A5. Dual vertex placement (`compute_vd(method=)`)
| Value | Where | Default | Status | ddgclib |
|---|---|---|---|---|
| `"barycentric"` (mean) | `_strategies.py:16-22` | **default** (`_compute_dual.py:46`) | C, T | **D, hard-coded** everywhere: `_integrators_dynamic.py:227, 749`, `periodic.py:506`, `_multiphase_droplet.py:162`, `operators/area.py:111`. |
| `"circumcentric"` (det formula 2D; `np.linalg.solve` kD; QR embedded; barycenter fallback only when exactly singular or 2D `|D|<1e-12`) | `_strategies.py:25-128` | | C, T | Only via deprecated `ddgclib/circumcentric/circumcentric_duals.py:31-33`. |
| custom `DualStrategy` callable | `_compute_dual.py:87-88` | | C, T | No. |
Records `HC._vd_method` (`_compute_dual.py:93`); ddgclib reads it (`stress.py:323`).

### A6. Dual computation path (inside `compute_vd`)
| Value | Trigger | Where | Status | ddgclib |
|---|---|---|---|---|
| legacy nn-walk 2D / 3D (uses `v.boundary`, `nn` intersections) | `_simplices` falsy, `backend=None` | `_compute_dual.py:147-231`, `:308-401` | C, T | D (structured initial mesh, `skip_triangulation` after BC invalidation) |
| simplex-aware 2D / 3D (face counts; ignores `v.boundary`) | `_simplices` truthy, `backend=None` | `:233-306`, `:403-501` | C, T | **D (main path after every Delaunay retopo)** |
| batch 2D / 3D (backend `batch_dual_positions`, no local merge; simplex-aware if cache) | `backend is not None` | `:503-610`, `:612-760` | C, T | **Never** (ddgclib never passes `backend` to `compute_vd`) |
| N-D recursive / N-D simplex-aware | `dim > 3` | `:771-926`, `:928-998` | C, T (4D) | No |
| global merge `HC.Vd.merge_all(cdist)` | `global_merge=True` default, `cdist=1e-10` | `:120-124` | C | D (defaults) |

### A7. Dual face / edge area vector A_ij
| Value | Where | Status | ddgclib |
|---|---|---|---|
| `e_star` 2D scalar `|vd1 - vd2|` from `v_i.vd & v_j.vd` | `_operators.py:45-50` | C, T | via `stress_pointwise.py` |
| ddgclib 2D `dual_area_vector` (rotated vd-pair, oriented away from `v_i`; periodic min-image variant) | `ddgclib/operators/stress.py:97-173` | ddgclib | D (2D, always; no 2D edge cache) |
| `e_star` 3D fan walk (tet-bary fan through edge midpoint; `n`-oriented) | `_operators.py:52-100` | C, T | D as fallback (`stress.py:280-308`) |
| ddgclib 3D `_dual_area_vector_3d_p_ij` (tet barycentres interleaved with face barycentres; linear-precise) | `stress.py:182-277` | ddgclib | D only when no edge cache |
| `batch_e_star(orient=True)` cache `HC._edge_area_cache` (tet-bary fan, **no face barycentres**) | `_operators.py:358-512` | **U** (orient) | **D and takes priority in 3D dynamic runs** (`stress.py:512-520, 585-594, 826-834`; set at `_integrators_dynamic.py:276`). See footgun F14. |
| `dual_cell_faces_3d` (p_ij polygons) | `_dual_cell.py:274-375` | C, T | `ddgclib/geometry/_dual_split_2d.py` |

### A8. Dual volume Vol_i
| Value | Where | Status | ddgclib |
|---|---|---|---|
| exact barycentric `(1/(d+1)) sum |T|` batched | `simplex_dual_volumes` `_dual_volume.py:41-88` | C, T | D: 3D dynamic (`_integrators_dynamic.py:262-272`), `cache_dual_volumes` 2D+3D (`stress.py:447-457`) |
| same, per vertex (O(N_simplices) scan per call) | `vertex_dual_volume` `_dual_volume.py:91-128` | C, T | D: `stress.dual_volume` 2D/3D (`stress.py:380-382, 404-406`) |
| 2D exact polygon (shoelace over walk-ordered p_ij polygon) | `dual_cell_area_2d` `_dual_cell.py:223-247` (+ walk U) | C (+U) | D fallback (circumcentric / no cache) |
| 3D `v_star` fan tets (undercounts 1-4% interior) | `_operators.py:180-236` | C, T | D fallback (`stress.py:410-424`) |
| 3D `batch_e_star(compute_volumes=True)` | `_operators.py:465-508` | C | computed but **discarded** in 3D when exact applies (`_integrators_dynamic.py:270-275`); unreachable in 2D (raises) |
| `d_area` (0.5*b*h approximation) | `_operators.py:276-298` | C | D in `visualization/matplotlib_2d.py`, `operators/area.py`, `operators/stress_pointwise.py` |
Boundary convention differs: 3D dynamic path zeroes `dual_vol` on `dV` (`_integrators_dynamic.py:272, 275`); 2D goes through `cache_dual_volumes` which does **not** zero boundary cells (`stress.py:455-457`). The comment at `_integrators_dynamic.py:257-260` ("2D keeps batch_e_star's volumes ... Boundary zeroing convention preserved in both paths") is inaccurate: in 2D `batch_e_star` raises `NotImplementedError` and the except branch runs.

### A9. Boundary detection
| Value | Where | Status | ddgclib |
|---|---|---|---|
| `HC.boundary()` legacy clique enumeration (unreliable on Delaunay flag complexes) | `_complex.py:2281-2348` | C | D fallback when no cache (`_integrators_dynamic.py:189, 215`, `periodic.py:481`, `_multiphase_droplet.py:150`) |
| `HC.boundary()` dispatch to `SC.boundary()` | `_complex.py:2317-2318` | U | no |
| `boundary_from_simplices` (face count) | `_boundary.py:18-97` | C, T | **D (primary)** |
| geometric bounding-box classification | ddgclib `identify_all_boundary` (`_rectangles.py:67`) | ddgclib | D (initial meshes) |
| `boundary_filter` (which topological boundary vertices are frozen) | ddgclib `_integrators_dynamic.py:285-288` | ddgclib | D |
| `batch_e_star` failed fan -> promote to boundary | `_integrators_dynamic.py:238-240` | | D (3D) |

### A10. Apex enumeration (curvature / interface)
`apex_vertices(HC, vi, vj)` / `get_edge_apex_map` (C, `_retriangulation.py:245-325`) vs legacy `vi.nn & vj.nn`. hyperct curvature `HC=None` default = legacy (U). ddgclib uses `get_edge_apex_map` directly (`_curvatures_heron.py`, `_bubble.py`, `periodic.py`).

### A11. 2D dual-cell polygon ordering
walk (U, `_dual_cell.py:118-179`) with angular fallback (`:182-216`); `include_edge_midpoints` True (p_ij, default) / False.

### A12. Backend
| Value | Where | Status | ddgclib |
|---|---|---|---|
| `None` / `"numpy"` | `_backend.py:292-380, 799-800` | C (+U methods) | default |
| `"multiprocessing"` (`workers=`) | `:386-497` | C (+U) | possible via `backend=` |
| `"torch"` / `"gpu"` (auto CUDA) | `:541-750, 763-780` | C (+U), **untested here** | possible |
ddgclib passes `backend` only to `batch_e_star` (cross-product kernel, `_integrators_dynamic.py:235`) and DEM (`dem/_particle.py:197-199`); never to `compute_vd` or `Complex(backend=)`. Parity footgun: CPU/torch `batch_circumcenters` fall back on `|det| <= 1e-12` (`_backend.py:236-241`, `:706-713`) whereas sequential `circumcenter` 3D uses unguarded `np.linalg.solve` (`_strategies.py:88-97`), so near-degenerate slivers get different duals per path.

### A13. Remesh parameters (only meaningful for `remesh_mode='adaptive'`)
`adaptive_remesh(HC, dim=2, mps=None, L_min=None, L_max=None, alpha_min=0.5, alpha_max=1.4, quality_target_deg=20.0, max_iterations=3, smooth_iterations=1, smooth_relax=0.3, preserve_interface=True, length_scale="local", smooth_skip_interface=False)` (`remesh/_driver.py:234-249`); `length_scale in {"local","global"}` (`:316-318`); `mps` unused (`:270-271`); auto-fixes `L_min >= L_max` and `alpha_min >= alpha_max` (`:347-350`).

---

## 3. The new SimplicialComplex / `_ops` layer

### 3.1 API
- `Complex(dim, ..., simplicial=True)`; `HC.SC` (`SimplicialComplex` or `None`); `HC._simplices` property returns `SC.object_view()` when SC exists.
- `SimplicialComplex`: `simplices` (int array), `vtable`, `len()`, `faces(k)`, `cofaces(k)`, `boundary()`, `object_view()`, `boundary_operator(k)`, `coboundary_operator(k)`, `apply_boundary(k, x)`, `apply_coboundary(k, y)`, `add_simplex`, `remove_simplex`, `remove_vertex`, `on_event`, `mark_dirty`, `rebuild_from_nn`.
- `_ops`: `get_builder/register_builder`, `get_refiner/register_refiner`, `get_local_op/register_local_op`, `enumerate_top_cliques`, `OpContext`.

### 3.2 What it offers over `Complex` + raw `_simplices`
- Integer index array (`(N, dim+1)`) decoupled from volatile `v.index`: groundwork for vectorised/GPU operators.
- Exact boundary for any dim (the raw path already has this via `boundary_from_simplices`).
- `faces(k)`/`cofaces(k)` generalising the edge-apex map; graded `d_k` with `d d = 0` (tested).
- Incremental sync for `V.remove`, remesh split/flip; hooks on move/merge.
- Clique emission for hypercube meshes (verified exact).
For ddgclib today the practical gain is small: ddgclib already keeps a raw simplex list through every retopology and uses the exact consumers. The genuinely new capabilities (graded operators, index arrays) have no ddgclib consumer.

### 3.3 Is ddgclib using it?
No. Zero references in `ddgclib/` (package and tests).

### 3.4 Verified defects (block enabling it under ddgclib)
Reproduced with `probe_sc.py` / `probe_dbuilder.py` against the working tree:
1. **Retriangulation silently ignored.** Setter `_complex.py:310-317` only promotes when `self._SC is None`; afterwards it writes `_simplices_raw`, but the getter (`:305-307`) returns the old `SC.object_view()`. After the ddgclib pattern (move, disconnect all, `connect_and_cache_simplices`): `simplicial=False` -> cache == new Delaunay; `simplicial=True` -> cache == OLD triangulation and cached triangles are no longer edges of `v.nn`. `compute_vd`, `boundary_from_simplices`, exact volumes would all run on a dead triangulation.
2. **`invalidate_simplex_cache` / `rebuild_simplex_cache_2d` ineffective.** `HC._simplices = None` does not drop the SC; `mark_dirty` with `_rebuild_fn=None` (`_simplicial.py:131-134`) clears the flag and keeps rows. Probe: 32 stale triangles after invalidate; `rebuild_simplex_cache_2d` returned 30 while `HC._simplices` still had 24.
3. **Collapse creates holes.** `V.remove(v_j)` fires `vertex_removed` -> incident rows dropped (`_simplicial.py:339-349`); `edge_collapse_2d` then fires `collapse` -> `mark_dirty` which, without a rebuild fn, never re-adds the rewired fan. Probe: 32 -> 24 triangles, `HC.boundary()` 20 vertices vs true 16.
4. **Read with side effects (DelaunayBuilder).** Its `_rebuild_fn` (`_ops/_builder.py:137-151`) re-runs `connect_and_cache_simplices` on current positions **without disconnecting** existing edges, and is triggered lazily by any read of `HC._simplices`/`SC`. Probe: after 3 flips + 1 collapse, reading `HC._simplices` changed `v.nn` from 53 edges (planar, Euler-consistent) to 56.
5. Unhandled events `edge_split` (`_complex.py:1498`), `vertex_connected` (`:2065`), `merge`, `collapse` all degrade to `mark_dirty` (`_simplicial.py:369-370`); only the hypercube clique rebuild resolves them correctly, and clique rebuild is only valid for generative (non-Delaunay) meshes.
6. `vertex_moved` invalidates derived caches but a Lagrangian move can invert simplices; nothing flags that (same as raw path).
7. `_ops/_local.py` `OpContext`/`sc_update` is never invoked by `apply` (dead path).
8. Performance: `cofaces` and `object_view` are pure-Python loops; `faces` uses `np.unique(axis=0)`; `object_view` is rematerialised after every invalidation.

### 3.5 Test gaps
No test re-triangulates an SC-backed complex twice; no test calls `invalidate_simplex_cache` with SC; `test_collapse_marks_dirty` (`test_simplicial_sync.py:82-102`) only checks the dirty flag, not validity; `test_move_preserves_topology` compares identity sets (would pass with stale objects); `TestFlagComplexBug` never exercises legacy `boundary()`; builder boundary test is tautological; torch tests skipped.

### 3.6 What ddgclib would need to run integrators on SC
1. hyperct: setter must replace (or rebuild rows of) an existing SC when a new list is assigned, and drop it on `None`; `invalidate_simplex_cache`/`rebuild_simplex_cache_2d` must then work unchanged.
2. hyperct: setter-created SCs need a correct rebuild strategy, or `edge_collapse_2d` must emit exact `remove_simplices`/`add_simplices` (the 2D collapse knows both sets locally).
3. hyperct: `DelaunayBuilder` rebuild must disconnect first and should not fire implicitly on read.
4. ddgclib: create complexes with `simplicial=True` in domain builders (or attach via builder after construction); `_retopologize` then works through the fixed setter; BC injection paths already call `invalidate_simplex_cache`.
5. Re-run the pinned regression baselines (2D oscillating droplet, 3D static droplet floor 7.274172e-5), since `object_view` row order differs from the raw list order and duals are order-sensitive through local merging.

### 3.7 Safe to leave uncommitted?
Functionally yes (off by default; both suites green), but it is at risk (untracked files, no branch) and it shares `_complex.py` / `_backend.py` hunks with load-bearing changes. Do not commit as-is to master with the current "Complete" docs. Commit it on a feature branch (e.g. `simplicial-layer`), fix 3.4 items 1-4, add the missing tests, then merge.

---

## 4. Load-bearing invariants and footguns

### 4.1 The 10 invariants from `docs_temp/code_map/hyperct_upstream.md` section 8, re-verified
| # | Invariant | Status now |
|---|---|---|
| 1 | Coordinate tuple = identity; mutate only via `HC.V.move` | Holds (`_vertex.py:67-81, 270-306`). Addendum: `move` to an already-occupied key silently overwrites the other vertex in the cache (`:298`); only the remesh smoother guards this (`remesh/_driver.py:228-230`). `move` also reorders the OrderedDict (vertex goes to the end), so `list(HC.V)` order changes every step. |
| 2 | `v.nn` is the only always-present topology (flag complex) | Holds. For hypercube-generated meshes the flag complex is exact (verified). |
| 3 | `HC._simplices` = vertex objects, top-dim tuples; invalidate on unrouted changes | Holds for `simplicial=False`. **Broken for `simplicial=True`** (section 3.4 items 1-2). |
| 4 | `v.boundary` must be tagged before `compute_vd` | Holds for legacy paths (`_compute_dual.py:175, 352-356, 602`). Simplex-aware paths ignore `v.boundary` (`:233-306, :403-501`), but `e_star/v_star/_walk_fan_3d/batch_e_star` still read it (`_operators.py:64, 192, 329, 420`), so tags must agree with the face-count boundary. |
| 5 | Duals are throwaway | Holds (`_compute_dual.py:96-100`). Addendum: global `Vd.merge_all` removes merged dual vertices from `HC.Vd` but leaves them in primal `v.vd` sets (`_vertex.py:454-466`), i.e. dangling references (rare at `cdist=1e-10`). |
| 6 | Dual identity tuple-keyed; must merge | Holds (`_geometry.py:24-64`). Batch paths skip local merge (`_compute_dual.py:508-510`). |
| 7 | 3D fan walk get-or-create on midpoint | Holds; verified: interior `e_star(dim=3)` grew `HC.Vd` 170 -> 179. Section 5 of the code map ("KeyError if the midpoint dual doesn't exist") is wrong: it never raises, it pollutes. ddgclib's `_dual_area_vector_3d_e_star` does the same (`stress.py:300-301`). |
| 8 | `_top` indexes `_vtable`, never `v.index` | Holds (`_simplicial.py:49-52, 105-112`). |
| 9 | Remesh preconditions; ops sync `v.nn`+SC, not `v.vd` / raw cache | Holds (`_operations_2d.py:304-311, 479-480, 650-654`); ddgclib rebuilds the raw cache (`_integrators_dynamic.py:179-180`). |
| 10 | Split averages mass (non-conservative) | **Stale.** Split transfers area-fraction mass (`_operations_2d.py:262-278`) with momentum-weighted `u` (`:286-292`); collapse sums `m` and `m_phase` (`:424-443`) with momentum-conserving `u` (`:451-461`); flip also transfers mass/momentum (`:560-645`). `phase` still copied from `src_i` (`:130-134`). |

### 4.2 The section-9 bug list, re-verified
| Bug | Status |
|---|---|
| `_vertex.py:787` `v2.check_max` NameError in `proc_minimisers` | **Still present** (`_vertex.py:787`). |
| `_simplex.py` hash is a list | Still present (`_simplex.py:24`); module unused. |
| `d_area` approximate | Still (`_operators.py:276-298`); ddgclib still uses it in 3 modules. |
| N-D `e_star/v_star` approximations | Still (`_operators.py:102-150, 238-273`). |
| `_compute_vd_3d_batch` omits `vd_mid.connect(vd_face)` | Still (`_compute_dual.py:754-760` vs `:499-500`); moot for ddgclib (never uses batch `compute_vd`). |
| `edge_collapse_2d` returns True after aborted move | **Fixed**: upfront zero-mutation abort (`_operations_2d.py:408-412`); residual guard `:484-489` is unreachable. |
| `Complex.boundary` unreliable on Delaunay | Still (`_complex.py:2281-2348`), now documented. |
| circumcenter silent barycenter fallback | Still (`_strategies.py:74-75, 96-97, 124-125`); plus 3D has no conditioning threshold (see A12). |
| ddgclib CLAUDE.md stale symlink path | Still (`/home/endres/projects/ddgclib/CLAUDE.md:85` says `/home/stefan_endres/...`). |
| `adaptive_remesh(mps=)` unused; no 3D ops | Still (`_driver.py:270-271, 314-315`). |
Also stale in the code map: remesh line numbers (driver now `:234`, split `:218`, collapse `:375`, flip `:499`), `invalidate_simplex_cache :150`, `get_edge_apex_map :245`, `apex_vertices :305`, and `dual_cell_polygon_2d` "angular sort" (now walk-ordered, U).

### 4.3 New footguns (from uncommitted work and this audit)
- **F11 (critical, commit hazard):** ddgclib requires uncommitted `batch_e_star(orient=)` (section 0 item 2).
- **F12:** `merge_cdist` merge runs before `remesh_mode='adaptive'` (`_integrators_dynamic.py:148-154`); `merge_pair` unions neighbour sets (`_vertex.py:463-466`), producing a non-conforming mesh that the adaptive ops (which infer triangles from `v.nn & v.nn`) then operate on. Safe in Delaunay mode (full retriangulation follows), not in adaptive mode.
- **F13:** remesh ops and `iter_triangles_2d` infer triangles from the flag complex (`_operations_2d.py:202-211`, `_quality.py:123-144`), never from the simplex cache, even when one exists.
- **F14:** In 3D dynamic runs, `HC._edge_area_cache` from `batch_e_star(orient=True)` (tet-barycentre fan, no face barycentres) overrides ddgclib's linear-precise `_dual_area_vector_3d_p_ij` in `stress_force`, `velocity_difference_tensor`, `scalar_gradient_integrated` (`stress.py:512-520, 585-594, 826-834`). Different A_ij in dynamic vs static 3D paths; worth a linear-precision check of the cached vectors. The cache is also not refreshed when the displacement gate skips retopology (`_integrators_dynamic.py:300-329`).
- **F15:** SC layer defects 3.4 items 1-4 (setter, invalidate, collapse holes, read side effect).
- **F16:** `Complex.boundary()` dispatches to `SC.boundary()` even when the SC is stale (`_complex.py:2317-2318`).
- **F17:** Stale hyperct 0.3.5 in `site-packages` shadows the live tree outside the ddgclib root (section 0 item 6). ddgclib itself is not pip-installed in `ddg` either.
- **F18:** Torch/GPU code paths (old and new) have zero executed coverage in the `ddg` env.
- **F19:** `vertex_dual_volume` is O(N_simplices) per call; `stress.dual_volume` calls it per vertex (O(N^2) if used in a loop instead of `cache_dual_volumes`).

---

## 5. Remesh package

- **Status:** 2D only; `adaptive_remesh` raises `NotImplementedError` for `dim != 2` (`remesh/_driver.py:314-315`). Sweep = split (`:87-122`) -> collapse (`:125-150`) -> quality flips (`:153-165`, `min_quality_gain=1e-6`) -> Laplacian smoothing (`:168-231`); stops on no-ops or min angle >= target (`:387-390`). Committed in `59f2941` with `test_remesh.py` + `test_remesh_conservation.py` (22 tests); all pass. No uncommitted remesh changes (`git diff HEAD -- hyperct/remesh` empty).
- **Lane-4 conservation fixes (verified in code):**
  - `edge_split_2d`: mass fraction `f = 0.5*area_T/ring` taken FROM each endpoint (`_operations_2d.py:276-278`), per-phase too (`_transfer_extensive` `:163-195`); momentum-weighted midpoint `u` (`:286-292`). Exact "uniform density stays uniform" only for barycentric duals (the derivation assumes dual area = ring/3, `:262-275`); total mass is conserved for any dual.
  - `edge_collapse_2d`: zero-mutation abort on collision / stale vertex (`:408-412`); additive `m`, `m_phase` (`:429-443`); momentum-conserving `u` (`:451-461`). The survivor is then moved to the midpoint (`:483-490`), which changes neighbouring dual volumes without mass remap (density perturbation, mass still conserved).
  - `edge_flip_2d`: barycentric-share mass/momentum transfer from losers `v_i, v_j` to gainers `v_k, v_l` (`:560-645`). Not listed in ddgclib DEVELOPMENT.md Phase 2b but present.
  - Driver: per-edge local length scale (`_edge_h_local` `:66-84`, `length_scale='local'` default) and `smooth_skip_interface`.
  - Remaining non-conservative element: Laplacian smoothing moves vertices without any field remap (`:220-231`); ddgclib's adaptive case disables smoothing for this reason (ddgclib DEVELOPMENT.md:1058-1060). Boundary-edge splits place the midpoint on the chord (no projection back to curved geometry).
- **Missing for 3D (nothing exists):** `_operations_3d.py` (edge split with tet star, edge collapse with inversion test, 2-3/3-2 face/edge swaps), 3D quality metrics (dihedral angles), 3D barycentric mass fractions (1/4 of incident tet volume), face-level interface constraints, a 3D triangle/tet iterator, and a `rebuild_simplex_cache_3d` (K4-clique ghost filter) so the adaptive branch can keep the simplex cache in 3D (currently it can only invalidate, `_integrators_dynamic.py:181-182`). Tracked as ddgclib DEVELOPMENT.md:1062-1066 "Phase 3 (not started)".
- **ddgclib wiring (`remesh_mode='adaptive'`):** correct. `_retopologize` calls `adaptive_remesh(HC, dim=dim, **remesh_kwargs)` on existing connectivity (`:169`), rebuilds the 2D simplex cache (`:179-180`), recomputes the boundary with `boundary_from_simplices` (`:185-187`), retags `v.boundary`, recomputes barycentric duals, then volumes via `cache_dual_volumes` (2D). `remesh_mode/remesh_kwargs` are threaded through all five integrators and `_retopologize_multiphase`. `dim != 2` intentionally propagates `NotImplementedError` (`:157-163`). Caveats: F12 (merge before adaptive), F13 (flag-complex triangle inference), and the ops use `v.boundary` from the previous step for `can_collapse/can_flip`/smoothing (fine, since new midpoints inherit boundary at `_operations_2d.py:297-300`).

---

## 6. Environment / how tests were run
- hyperct: `cd /home/endres/projects/hyperct && /home/endres/anaconda3/envs/ddg/bin/python -m pytest hyperct/tests -q --no-header -p no:cacheprovider [-k "not benchmark"]`.
- ddgclib: `cd /home/endres/projects/ddgclib && ... -m pytest ddgclib/tests -q -x -m "not slow"` -> 881 passed, 12 skipped, 17 deselected, 2 xfailed, 77 s.
- Probes need `PYTHONPATH=/home/endres/projects/hyperct` (F17).

---

## 7. Recommended commit plan (owner decides; nothing was committed)
1. Save the full working tree on a branch first (e.g. `git switch -c wip/simplicial-layer && git add -A ':!pytest-of-stefan_endres' && git commit`), so the untracked SC work is no longer at risk.
2. On master, commit only the backward-compatible, load-bearing hunks: `ddg/_operators.py` (orient, required by ddgclib), `ddg/_dual_cell.py` (walk ordering, bug fix), `ddg/_curvature.py` (`HC=`), `_backend.py` `batch_heron_curvature` hunks + `tests/test_gpu.py`. `_backend.py` mixes heron with circumcenter/sparse hunks, so this needs hunk-level staging, or accept the extra unused kernels (they are additive and tested on numpy).
3. Keep `_simplicial.py`, `_ops/`, the `_complex.py` / `_vertex.py` hooks, the circumcenter/sparse kernels, `test_simplicial*.py`, `test_ops.py` and the agent docs on the branch until 3.4 items 1-4 are fixed and tested (retriangulate-twice, invalidate-with-SC, collapse-validity, read-without-side-effects), and the docs are corrected.
4. Separately: fix `_vertex.py:787`; delete `pytest-of-stefan_endres/`; refresh `docs_temp/code_map/hyperct_upstream.md` (invariant 10, e_star KeyError claim, line numbers); fix the ddgclib CLAUDE.md symlink path; consider `pip install -e /home/endres/projects/hyperct` into `ddg` (or uninstall the 0.3.5 wheel) to remove F17.
