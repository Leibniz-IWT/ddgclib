# Hyperct Upstream Package Map (mesh backend for ddgclib)
> Sources: hyperct/ddg/{__init__,_compute_dual,_strategies,_operators,_dual_cell,_geometry,_curvature,_retriangulation,_boundary}.py, hyperct/{_complex,_vertex,_simplex,_simplicial}.py, hyperct/_ops/{__init__,_builder,_refiner,_local}.py, hyperct/remesh/{__init__,_driver,_interface,_operations_2d,_quality}.py | Written: 2026-07-02 by understand-and-document workflow

All paths below relative to `/home/endres/projects/ddgclib/hyperct/` (a **symlink**; verified chain 2026-07-02: `→ /home/endres/projects/bilevel_param/hyperct → /home/endres/projects/hyperct/hyperct`; treat as read-only source of the external package). Top-level `hyperct/__init__.py` does `from hyperct._complex import *`.

---

## 1. Vertex layer (`_vertex.py`)

### Classes
- `VertexBase` (ABC, `_vertex.py:12`): `self.x` = coordinate **tuple** (identity key), `self.hash = hash(self.x)` precomputed at `__init__` (`:69`), `self.nn` = set of neighbour vertex objects (1-ring), `self.index`, `self.dtype`. `x_a` is **lazy**: `__getattr__` (`:88-94`) builds `np.array(self.x, dtype=self.dtype)` on first access and caches it as an instance attribute. `__hash__` returns precomputed hash; **no `__eq__`** → set/dict membership is by object identity (custom hash bucket + identity compare). `star()` (`:116`) mutates `self.nn` by adding self — side-effecting, avoid.
- `VertexCube(VertexBase)` (`:127`): pure-geometry vertex. `connect(v)` (`:133`): symmetric `nn.add` both ways, no-op for self/duplicates. `disconnect(v)` (`:138`) symmetric remove.
- `VertexScalarField(VertexBase)` (`:144`): adds `check_min/check_max` flags; `connect/disconnect` also reset those flags on both endpoints. `minimiser()/maximiser()` compare `v.f` against neighbours.
- `VertexVectorField` (`:216`): raises `NotImplementedError` (WIP).

### Caches (get-or-create keyed by coordinate tuple)
- `VertexCacheBase` (`:230`): `self.cache = collections.OrderedDict()` mapping coord-tuple → vertex object. Iteration snapshots values (`list(self.cache.values())`, `:263`) so mutation during iteration is safe. `self._sc_hook` (`:256`) — optional callback `hook(event, **kw)` installed by a Complex with an active simplicial representation.
  - `move(v, x)` (`:270`): pops old key, **disconnects all neighbours, re-keys (`v.x`, `v.hash`, `v.x_a`), reconnects** — required because the hash changes. Emits `_sc_hook("vertex_moved", vertex=v)`. NOTE: does **not** touch `v.vd` — duals are stale after any move; recompute via `compute_vd`.
  - `remove(v)` (`:308`): disconnects all neighbours, pops from cache, sets `_indices_dirty=True` (indices rebuilt lazily by `_rebuild_indices` `:330`), emits `_sc_hook("vertex_removed", vertex=v)`.
  - `merge_nn(cdist)` (`:347`), `merge_all(cdist)` (`:394`): pairwise/grid-hashed (3^dim neighbour cells) merging of vertices within `cdist`; `merge_pair(vp)` (`:454`) reconnects `vp[1]`'s neighbours to `vp[0]` then removes `vp[1]`. Both suspend `_sc_hook` during the batch and emit a single `"merge"` event afterwards.
- `VertexCacheIndex(VertexCacheBase)` (`:480`): `Vertex = VertexCube`; `__getitem__(x)` = get-or-create. Used for **dual caches** (`HC.Vd`).
- `VertexCacheField(VertexCacheBase)` (`:498`): `Vertex = VertexScalarField`; deferred field/constraint pools (`fpool`/`gpool`) processed by `process_pools()`; dispatch to serial / multiprocessing / backend batch paths.
- **BUG** `proc_minimisers` (`:761-787`): final line `v2.check_max = False` (`:787`) references undefined `v2` → `NameError` whenever a vertex has `f` but empty `nn`.

### `_simplex.py` (78 lines, vestigial)
`SimplexBase`/`SimplexOrdered` are **stubs**: unordered hash key is a *list* (`self.hash = hkey`, `_simplex.py:24`) so `__hash__` would raise `TypeError`; dead `if 0:` blocks. Not used anywhere in the live code paths — the real simplex containers are `HC._simplices` (list of tuples) and `SimplicialComplex` (§3). New simplex-container code should NOT build on `_simplex.py`.

---

## 2. Complex lifecycle (`_complex.py`, 2987 lines)

`Complex.__init__(dim, domain=None, sfield=None, ..., symmetry=None, constraints=None, workers=None, backend=None, simplicial=False)` (`_complex.py:106`):
- `self.bounds` defaults to `[(0.0,1.0)]*dim`; domain must be a convex hyperrectangle (non-convexity via `g_cons` inequality constraints, which set `v.feasible`; `cut_g()` `:1501` removes infeasible vertices).
- `self.V` = `VertexCacheField` if `sfield` or constraints given, else `VertexCacheIndex` (`:247-264`).
- `self.H` = generation storage (list of vertex groups; legacy), `self.gen`.
- `backend` string → `get_backend(backend)` stored as `self._backend`.
- `simplicial=True` sets `self._simplicial`, installs `self.V._sc_hook = self._sc_notify` (`:288`).

### Triangulation / refinement (all operate on the 1-skeleton only)
- `triangulate(n=None, symmetry=None, centroid=True)` (`:554`): cyclic-product C2^dim hypercube triangulation via generator `cyclic_product` (`:342`); optionally stops at `n` total vertices; records `self.triangulated_vectors = [(origin, supremum), ...]`.
- `refine(n=1)` (`:714`): adds ~n vertices via `refine_local_space` generator; triangulates first if needed. `refine_all()` (`:774`): refines every triangulated vector region.
- `split_edge(v1, v2)` (`:1482`): **takes coordinate tuples** (does `self.V[v1]`), disconnects the edge, creates midpoint vertex `self.V[tuple(vct)]`, connects it to both endpoints, emits `_sc_notify("edge_split", v1=, v2=, vc=)`, returns the midpoint vertex. Does NOT connect midpoint to the opposite apices (that's `refine_star`'s job, or the remesh `edge_split_2d` which does).
- `refine_star(v)` (`:2163`) / `refine_all_star(exclude=set())` (`:2243`): split all star edges and cross-connect midpoints.
- `connect_vertex_non_symm(v_x, near=None)` (`:1987`): insert an arbitrary point into the existing triangulation (locate containing simplex via `in_simplex` `:2068`, connect to its vertices).
- `vf_to_vv(vertices, simplices)` (`:2759`): build complex from explicit vertex/face lists (OBJ-style import).
- `save_complex/load_complex` (`:2913/:2957`): JSON persistence.

### Boundary
- `Complex.boundary(V=None)` (`:2281`): enumerates `dim`-combinations of each `v.nn`, checks mutual connectivity (clique test), boundary iff common-neighbour set is empty. **Documented as unreliable on Delaunay-derived meshes** (flag-complex spurious K_{dim+1} cliques, warning at `:2289-2307`). If `V is None and self._SC is not None` dispatches to exact `SimplicialComplex.boundary()` (`:2317`).
- Preferred exact alternative: `hyperct.ddg.boundary_from_simplices(HC, dim)` (`ddg/_boundary.py:18`): counts (dim-1)-face occurrences over `HC._simplices` (face key = `tuple(sorted(id(v) for v in face))`); count==1 → boundary face; returns set of vertex objects. Raises `ValueError` if `HC._simplices is None`.

### `_simplices` property (`:298-317`)
- Getter: returns `self._SC.object_view()` when the simplicial representation is active, else `self._simplices_raw` (`None` for any complex built via `triangulate`/`refine_all`/manual `connect` — only Delaunay helpers assign it).
- Setter: stores into `_simplices_raw`; when `simplicial=True` and no SC yet, **promotes** the list to a `SimplicialComplex` via `from_object_tuples`.
- `HC.SC` property (`:320`) exposes the active `SimplicialComplex` or `None`.
- `_sc_notify(event, **kw)` (`:324`): no-op when `_SC is None`; otherwise dispatch to `SC.on_event` (falls back to `mark_dirty`).

---

## 3. Simplex containers

### 3a. Legacy raw cache: `HC._simplices`
A plain Python list of `(dim+1)`-tuples of **vertex objects** (object identity preserved so field data stays attached). Populated by `hyperct.ddg.connect_and_cache_simplices(HC, verts, dim, simplices=None, coords=None, qhull_options=None)` (`ddg/_retriangulation.py:52`):
- Runs `scipy.spatial.Delaunay(coords)` (fallback `qhull_options="Qbb Qt Qz"` on cospherical failure), connects all pairwise edges of each simplex, then caches `HC._simplices = [tuple(verts[s[i]]...) for s in simplices if len(s)==dim+1]`. Supports `dim ∈ {2,3}` only (`:94`).
- **Vertex-correspondence invariant** (docstring `:34-44`): `verts` must be `list(HC.V)` taken AFTER any disconnect/merge/ghost-resolution but BEFORE the Delaunay call; integer indices are immediately translated to objects so later cache-order changes are harmless.
- **Cache MUST be invalidated after any topology change not routed through this helper**: `invalidate_simplex_cache(HC)` (`:131`) sets `HC._simplices = None`, clears `HC._edge_to_apex`, and marks `HC.SC` dirty. Forgetting the cache step silently reverts dual/boundary code to the buggy flag-complex paths.
- `get_edge_apex_map(HC)` (`:155`): lazily builds `HC._edge_to_apex: dict[frozenset[id,id] -> list[vertex]]` (apices of every top simplex containing each edge). Returns `None` if no simplex cache.
- `apex_vertices(HC, vi, vj)` (`:215`): exact apices from the map when available, else legacy `vi.nn ∩ vj.nn` (spurious on skinny Delaunay meshes). `HC=None` forces legacy.

### 3b. `SimplicialComplex` (`_simplicial.py:36`) — the modern container
Source of truth = `self._top`, an `(N, dim+1)` int32 numpy array of rows into a **private frozen `_vtable`** (`list(HC.V)` snapshot at build time; `_id_to_local: id(v) -> row`). Rationale: `v.index` is rebuilt lazily and NOT stable across mutations, so indices must never point at `v.index` (`:20-27`).
- Constructors: `from_object_tuples(HC, simplices_obj, dim)` (`:66`), `from_index_array(HC, simplices_idx, vtable, dim)` (`:87`).
- Derived, lazy+cached (all cleared by `_invalidate_derived` `:114`): `faces(k)` (`:161`, unique sorted `(M,k+1)` int array), `cofaces(k)` (`:186`, `dict[sorted-tuple -> [top rows]]` — generalises the edge-apex map), `object_view()` (`:229`, legacy `(dim+1)`-tuple-of-objects list; drops rows referencing tombstoned vertices), `boundary()` (`:210`, exact face-count rule for any dim), `boundary_operator(k)` (`:258`, sparse oriented `d_k: C_k -> C_{k-1}` via backend `build_sparse_boundary`, sign `(-1)^i`), `apply_boundary/apply_coboundary` (`:299/:303`).
- Mutation primitives (public extension points, `:312-349`): `add_simplex(simplices)`, `remove_simplex(simplices)`, `remove_vertex(v)` (drops incident rows, tombstones `_vtable[li]=None`).
- Sync: `on_event(event, **kw)` (`:351`) handles `"vertex_removed"` (incremental), `"vertex_moved"` (invalidate derived only — topology unchanged), `"add_simplices"`, `"remove_simplices"`; anything else (`"merge"`, `"edge_split"`, `"collapse"`) → `mark_dirty()` (`:372`) which clears derived caches and defers `_top` regeneration to `_rebuild_fn` on next read (`_resolve` `:121`). `rebuild_from_nn()` (`:378`) delegates to the registered rebuild strategy — reconstructing from `v.nn` alone is exactly the flag-complex problem this class avoids.

---

## 4. Dual mesh computation (`ddg/_compute_dual.py`)

### Entry point
`compute_vd(HC, method="barycentric"|"circumcentric"|DualStrategy, cdist=1e-10, global_merge=True, backend=None)` (`_compute_dual.py:42`):
1. Resolves strategy (`ddg/_strategies.py`): `barycenter(verts) = np.mean(verts, axis=0)` (`:16`); `circumcenter(verts)` (`:25`) handles full-dim (determinant formula k=2, linear solve k>=3, `:56`) and embedded k<dim simplices (QR project–solve–lift, `:100`); **falls back to barycenter for degenerate configs** (|D|<1e-12 or `LinAlgError`). A custom `DualStrategy = Callable[[(n_verts,dim) array], (dim,) array]` may be passed directly (extension point).
2. `HC.Vd = VertexCacheIndex()` — fresh dual cache keyed by dual coordinate tuple; `v.vd = set()` initialised on every primal vertex (`:91-95`).
3. Dim dispatch: 1D `_compute_vd_1d` (`:124`, edge midpoints); 2D `_compute_vd_2d` (`:142`) or `_compute_vd_2d_batch` (`:498`) if `backend` given; 3D `_compute_vd_3d` (`:303`) / `_compute_vd_3d_batch` (`:607`); N-D `_compute_vd_nd_simplex_aware` (`:923`) if `HC._simplices` else `_compute_vd_nd` (`:766`, recursive face extension from `v.nn` intersections).
4. Optional `HC.Vd.merge_all(cdist)` global spatial-hash dedup (`:118`).

**Precondition:** `v.boundary` must be set (True/False) on all vertices BEFORE calling (the legacy paths use it to classify boundary edges/faces; missing attribute is swallowed by try/except AttributeError → boundary edges silently mis-handled, e.g. `_compute_vd_2d:169-190` and `_has_boundary` `:758`).

### 2D algorithm (simplex-aware, `_compute_vd_2d_simplex_aware:228`)
1. Per triangle in `HC._simplices`: dual vertex at `strategy(3x dim coords)`; associate with all 3 primal vertices (`v.vd.add(vd)`).
2. Build `edge -> [simplex indices]` map (edge key = `tuple(sorted((id(a),id(b))))`).
3. Interior edge (2 tris): connect the two duals (`vd_a.connect(vd_b)` — dual connectivity lives in `vd.nn`, same VertexCube machinery). Boundary edge (1 tri): create **edge-midpoint dual**, add to both endpoints' `.vd`, connect it to the triangle dual.
Legacy nn-path (`_compute_vd_2d:142`) does the same via `v1.nn ∩ v2.nn` and is wrong on flag-complex K_3 cliques.

### 3D algorithm (simplex-aware, `_compute_vd_3d_simplex_aware:398`)
1. Per tet: dual at `strategy(4x3)`; associate with all 4 primal vertices.
2. `face -> [tets]` map. Interior face: connect the two tet duals. Boundary face: create **face dual** at `strategy(face 3x3)` (barycenter/circumcenter of the boundary triangle), associate with the 3 face vertices, connect tet-dual↔face-dual; plus **edge-midpoint duals** for all 3 boundary-face edges, each connected to the face dual (`:495`, "ring-walk support").
- Batch variants collect all simplices, call `backend.batch_dual_positions(simplex_arr, strategy)` once, then wire connectivity; local merge skipped, relies on global merge. **Inconsistency:** `_compute_vd_3d_batch` creates boundary edge-midpoint duals but does NOT connect them to the face dual (`:749-755`), unlike the sequential simplex-aware path (`:495`) — fan walks over batch-computed boundary duals may terminate differently.

### `v.vd` contents (data structure for duals)
`v.vd` = Python set of dual vertex objects (`VertexCube` in `HC.Vd`). Interior vertex: duals of all incident top simplices. Boundary vertex additionally holds edge-midpoint duals (2D/3D) and boundary-face duals (3D). Dual-dual adjacency in `vd.nn`. The dual polygon/face around a primal edge `(v_i, v_j)` is recovered as `v_i.vd ∩ v_j.vd`; the connectivity walk over that shared set is the core "fan walk" primitive.

### Dedup/merge
`_merge_local_duals_vector(x_a_l, Vd_cache, cdist=1e-10)` (`ddg/_geometry.py:24`): snaps proposed dual positions onto existing duals within `cdist` (vectorized pairwise distances) — prevents floating-point duplicate dual vertices since `HC.Vd` is exact-tuple-keyed.

---

## 5. DDG operators (`ddg/_operators.py`)

- `e_star(v_i, v_j, HC, n=None, dim=2)` (`:16`) — Hodge star of primal edge:
  - dim=1: scalar distance between the (1 or 2) shared duals.
  - dim=2: scalar `|vd1 - vd2|` from `v_i.vd ∩ v_j.vd` (assumes exactly ≥2 shared duals; boundary 2D edges have midpoint dual + triangle dual so 2 exist — but **order in the set is arbitrary**, callers must not assume orientation).
  - dim=3: returns `np.ndarray (N_fan, 3)` of dual-triangle **vector areas** `0.5 * (vc_12 - vd_i) × (vd_j - vd_i)`, oriented by optional direction `n` (flipped if `dot(normalized(w), n) < 0`; default `n = [0,0,0]` → no flip). Fan walk: `vc_12 = HC.Vd[tuple(midpoint)]` (**KeyError if the midpoint dual doesn't exist**, i.e. interior edges have no midpoint dual — 3D `e_star` is therefore only valid where boundary duals were created, or callers must catch; `batch_e_star` catches `(KeyError, IndexError)`). Walk over `dset = v_i.vd ∩ v_j.vd` via `vd.nn` adjacency; boundary edge start chosen as the dual with exactly 1 in-set neighbour, `iter_len=3` on boundary else `len(dset)`.
  - dim>3: approximate (dim-2)-volume via centroid fan + Gram determinants (`:102-150`) — **approximation, not exact DEC**.
- `v_star(v_i, v_j, HC, n=None, dim=2)` (`:153`): dim=2 same as e_star; dim=3 returns `(A_ij, V_ij)` — vector-area array plus **signed tetrahedron volumes** `volume_of_geometric_object([vc_12, vd_i, vd_j], apex=v_i.x_a)` per fan triangle (`(1/3)*base*height`, `ddg/_geometry.py:79`; degenerate-base → 0.0 short-circuit at `:96` — previously NaN-poisoned `dual_vol` on 95 boundary vertices of the 3D droplet box). dim>3: crude distance-power approximations (`dual_dist**(dim-1)`, `norm**dim`, `:261-271`) — **placeholder quality only**.
- `d_area(v)` (`:276`): scalar dual area as Σ over neighbours/shared duals of `0.5 * b * h` with `h = |mp - v|/1`, `b = |vd - mp|` — **approximate** (right-triangle assumption). Exact replacement: `dual_cell_area_2d` below.
- `batch_e_star(vertices, HC, dim=3, backend=None, compute_volumes=False, orient=False)` (`:358`): 3D only. Phase 1 CPU fan walk (`_walk_fan_3d` `:313`) collecting `(mid, vdi, vdj)` triples; skips `v.boundary` vertices; vertices whose fan walk raises `KeyError/IndexError` (broken duals from degenerate tets) go into `failed_vertices` — **caller contract: promote those to boundary**. Phase 2 vectorized `np.cross/2` (or `backend.batch_cross_areas`). Returns `edge_areas: {id(v): {id(nb): (N,3) or (3,)}}` (`orient=True` sums outward-oriented triangles per directed edge — matches `ddgclib.operators.stress.dual_area_vector`), `failed_vertices`, optionally `vertex_volumes: {id(v): float}` (Σ |det|/6 tet volumes).

### Dual cell extraction (`ddg/_dual_cell.py`) — the exact-integration layer
- `dual_cell_vertices_1d(v)` (`:36`): `(a, b)` interval; raises `ValueError` for boundary vertices.
- `dual_cell_polygon_2d(v, include_edge_midpoints=True)` (`:69`): ordered CCW `(N,2)` polygon. Two formulations: `include_edge_midpoints=True` = standard DEC barycentric dual cell `p_ij` (alternating edge midpoints and triangle barycenters — **linear precision**); `False` = barycenters only. Dedup by 12-decimal rounding; angular sort around the primal vertex (star-convexity assumption). Interior vertices only.
- `dual_cell_area_2d(v, include_edge_midpoints=True)` (`:132`): exact shoelace area — the documented replacement for approximate `d_area`.
- `dual_cell_faces_3d(v, HC, include_face_barycenters=True)` (`:183`): list of `(M,3)` ordered face polygons, one per primal edge from `v`. Ring-walk over `v.vd ∩ v_j.vd` using dual `nn` adjacency; fallback `_angular_sort_3d` (`:287`). With `include_face_barycenters=True` interleaves primal face barycenters `(x_i+x_j+x_k)/3` between consecutive tet duals (nearest-midpoint matching) — required for linear precision of barycentric duals. Faces oriented outward from `v` via total-area-vector check (`:270-281`). Skips edges with `<3` shared duals (boundary/degenerate).

### Curvature (`ddg/_curvature.py`, legacy mean-curvature-flow support)
- `HNdC_ijk(e_ij, l_ij, l_jk, l_ik)` (`:16`): stable Heron area `A = 0.25*sqrt((a+(b+c))(c-(a-b))(c+(a-b))(a+(b-c)))` with sorted a>=b>=c; cotan weight `w_ij = (l_jk² + l_ik² − l_ij²)/(8A)`; returns `(w_ij * e_ij, 0.5 * |w_ij| l_ij * 0.5 l_ij)` = per-triangle curvature vector + dual-area contribution.
- `mean_curvature(v, n_i=None, HC=None)` (`:125`) → `(HNdA_i (3,), C_i float)`; handles boundary (1-apex) edges. `normal_area` (`:57`) ignores boundary edges. `integrated_curvature` (`:205`) same recursion on `v.hnda_i` differences. Sign convention: `e_ij = -(vj − vi)` (edge points j→i). All use `apex_vertices(HC, vi, vj)` — pass `HC` to get exact simplex-aware apices.

---

## 6. Remesh package (`remesh/`) — interface-preserving 2D adaptive remeshing

**Driver** `adaptive_remesh(HC, dim=2, mps=None, L_min=None, L_max=None, alpha_min=0.5, alpha_max=1.4, quality_target_deg=20.0, max_iterations=3, smooth_iterations=1, smooth_relax=0.3, preserve_interface=True)` (`remesh/_driver.py:195`), **dim=2 only** (raises otherwise). Per iteration: (1) split edges > `L_max` (`_split_long_edges:66`, phase-topology guard), (2) collapse edges < `L_min` subject to `can_collapse`, (3) Delaunay-quality flips (`min_quality_gain=1e-6`), (4) tangential Laplacian smoothing (`_laplacian_smooth:129` — boundary vertices frozen; interface vertices averaged over interface neighbours only). Stops when no ops or min angle ≥ target. `L_min/L_max` default to `alpha_min/alpha_max × median edge length`; forced `L_min = 0.5*L_max` if inverted. Returns stats dict `{n_splits, n_collapses, n_flips, min_angle_deg, iterations, n_triangles}`. **Does NOT recompute duals — caller must run `compute_vd` afterwards** (`:12-13`); also does not maintain `HC._simplices` — call `invalidate_simplex_cache` after a remesh batch (per `_retriangulation.py:131` docstring).

**Interface constraints** (`remesh/_interface.py`): phase read via `getattr(v, "phase", None)`; `None` = single-phase = unconstrained. `is_interface_edge` (`:27`) = differing non-None phases. `can_flip` (`:52`): never flip interface edges nor edges with both endpoints boundary. `can_collapse` (`:68`): never across interface; never if **either** endpoint is boundary. `split_preserves_phase_topology` (`:93`): interface edges always splittable; bulk edges only if all opposite vertices share `v_i.phase` (prevents new cross-phase edges).

**Operations** (`remesh/_operations_2d.py`); triangles inferred from mutual connectivity (`triangles_around_edge(v_i,v_j) = v_i.nn & v_j.nn`, `:138`; 2 interior / 1 boundary):
- `edge_split_2d(HC, v_i, v_j)` (`:154`): midpoint vertex via `HC.V[x_m]` (aborts returning `None` if `x_m` already in cache); connects midpoint to endpoints + opposites, disconnects original edge; `_carry_attrs` averages fields; mass `v_m.m = 0.5*(m_i+m_j)` (NOT conservative — noted in-code); boundary inherited only when both endpoints boundary AND ≤1 opposite; incremental SC update replaces each `(v_i,v_j,v_k)` with `(v_i,v_m,v_k)+(v_j,v_m,v_k)` via `_sc_notify` (`:212-219`).
- `edge_collapse_2d(HC, v_i, v_j)` (`:283`): survivor `v_i` moved to midpoint; rejects if `_would_invert` (`:228`, signed-area flip test) in either direction; mass additive `v_i.m += v_j.m`; rewires neighbours, `HC.V.remove(v_j)`; emits `_sc_notify("collapse")` → SC mark-dirty (non-local). **Returns True even when the final `move` aborts due to a position collision** (`:339-343`) — connectivity merged but position unchanged.
- `edge_flip_2d(HC, v_i, v_j, min_quality_gain=0.0)` (`:353`): requires exactly 2 opposites and new edge `(v_k,v_l)` absent; min-angle quality gain + bowtie (signed-area-sum) + degeneracy checks; incremental SC simplex swap.
- `_carry_attrs(dst, src_i, src_j)` (`:51`): averages numeric/ndarray attrs, ANDs bools, copies rest from `src_i`; `_SKIP_ATTRS` (`:35`) excludes `x, x_a, nn, hash, index, vd, dual_vol, dual_vol_phase, is_interface, interface_phases, f, feasible, m, phase, ...`; `phase` copied verbatim from `src_i` (categorical; interface splits get the `src_i` side).

**Quality** (`remesh/_quality.py`): `triangle_min_angle` (radians, law of cosines, `:47`), `triangle_aspect_ratio = longest/(2*inradius)` (`:80`), `triangle_area` (signed 2D / unsigned 3D, `:28`), `edge_length` (`:115`), `iter_triangles_2d(HC)` (`:123`, canonical-id-ordered clique iteration — flag-complex based, same caveat), `mesh_quality_histogram` (`:147`, 2D only, slivers = min angle < 20°).

---

## 7. Extension points for new simplex-container geometry code

1. **`_ops` registry layer** (`_ops/__init__.py`) — Protocol + factory pattern mirroring `hyperct._backend.get_backend`, explicitly designed for plugging new operations while keeping `HC.SC` in sync:
   - Builders (`_ops/_builder.py`): `Builder` protocol with `build(HC, **kw)`; registered names `{"hypercube", "delaunay", "manual"}` (`:105/:123/:157`, registry `_BUILDERS:183`); `register_builder(builder)` (`:203`). `enumerate_top_cliques(HC)` (`:28`) emits real top simplices for *generative* (hypercube) complexes where every clique IS a simplex; `_clique_rebuild` (`:66`) is registered as `SimplicialComplex._rebuild_fn` so `mark_dirty` self-heals.
   - Refiners (`_ops/_refiner.py`): names `{"generation", "split_generation", "star", "local", "adaptive"}`; `register_refiner`; each calls `_resync(HC)` = `HC._SC.mark_dirty()` after bulk changes.
   - Local ops (`_ops/_local.py`): `LocalOp` protocol (`apply(HC, ...)` + `sc_update(HC, ctx: OpContext)`); `OpContext` dataclass declares `new_vertices / removed_vertices / touched_edges / added_simplices / removed_simplices` for incremental SC edits; names `{"split_edge", "connect_vertex", "edge_split_2d", "edge_collapse_2d", "edge_flip_2d"}`; `register_local_op`.
2. **`SimplicialComplex` mutation + derived API** (§3b): `add_simplex/remove_simplex/remove_vertex`, `faces(k)/cofaces(k)/boundary_operator(k)`, custom `_rebuild_fn`, `on_event` — the natural home for new exact-topology geometry (graded operators, 3D remesh, exact 3D boundary dual cells).
3. **`_sc_hook` event channel**: events currently emitted — `"vertex_removed"`, `"vertex_moved"`, `"merge"` (VertexCacheBase), `"edge_split"` (Complex.split_edge), `"collapse"`, `"add_simplices"`, `"remove_simplices"` (remesh ops). New topology-mutating code should emit through `HC._sc_notify` or the vertex-cache hook, or at minimum call `invalidate_simplex_cache(HC)`.
4. **Custom `DualStrategy`**: any `Callable[[(n,dim) array], (dim,) array]` passed as `compute_vd(HC, method=my_strategy)`; `_batch_strategy` (`_compute_dual.py:19`) vectorizes only `barycenter`, loops otherwise.
5. **Backends** (`_backend.py`, `get_backend("numpy"|"torch"|"gpu"|"multiprocessing")`): batch hooks consumed here — `batch_dual_positions`, `batch_cross_areas`, `build_sparse_boundary`, `batch_boundary_apply/coboundary_apply`, `batch_field_eval`, `batch_feasibility`.
6. **Deprecated shims**: `ddg/barycentric/`, `ddg/circumcentric/` re-export; do not extend.

---

## 8. Internal invariants (load-bearing for new code)

1. **Coordinate tuple = identity.** `v.x` is the cache key and hash source. Never mutate `v.x`/`v.hash` directly — always `HC.V.move(v, x_new)` (handles disconnect/re-key/reconnect). Two vertices at the same tuple are the same object (get-or-create).
2. **`v.nn` is the only always-present topology.** It is the flag complex; K_{dim+1} cliques ≠ simplices on Delaunay meshes. Anything needing true simplices must consult `HC._simplices` / `HC.SC` and handle the `None` fallback.
3. **`HC._simplices` holds vertex *objects*, top-dim only, `dim+1`-tuples.** Invalidate on any unrouted topology change; setter promotes to SC only in `simplicial=True` mode.
4. **`v.boundary` must be tagged before `compute_vd`** (and before `e_star`/`v_star` boundary fan logic). Missing attribute silently degrades boundary handling.
5. **Duals are throwaway**: `compute_vd` rebuilds `HC.Vd` and `v.vd` from scratch each call; any vertex move/remove/remesh invalidates them; nothing auto-refreshes.
6. **Dual identity is also tuple-keyed** (`HC.Vd = VertexCacheIndex`); dedup relies on `_merge_local_duals_vector` (local, cdist=1e-10) + `Vd.merge_all` (global). Geometry code creating duals must merge or positions differing at ~1e-16 create duplicate dual vertices and break fan walks.
7. **3D fan walk contract**: edge-midpoint duals exist only on boundary edges/faces; `e_star`/`v_star` dim=3 do `HC.Vd[tuple(midpoint)]` which *creates* a disconnected dual if absent (get-or-create!) and then the `vd_i.nn ∩ dset` walk can `IndexError`/`KeyError` — `batch_e_star` treats those as `failed_vertices` to be promoted to boundary.
8. **`SimplicialComplex._top` indexes `_vtable`, never `v.index`.**
9. **Remesh preconditions**: callers must check `can_collapse`/`can_flip` first; ops maintain `v.nn` + SC sync but not `v.vd` nor `HC._simplices_raw`.
10. **Mass/phase conventions in remesh**: split averages mass (non-conservative), collapse sums; `phase` inherited from `src_i` verbatim.

## 9. Bugs / stale claims / inconsistencies noticed

- `_vertex.py:787` — `v2.check_max = False` with undefined `v2` → `NameError` for a vertex with `f` and no neighbours (dead-branch bug).
- `_simplex.py` — `SimplexBase.hash` is a list; `__hash__` unusable; entire module vestigial.
- `d_area` (`ddg/_operators.py:276`) is geometrically approximate; `dual_cell_area_2d` docstring explicitly calls it incorrect for non-right triangles. ddgclib code should prefer the dual-cell versions.
- N-D (`dim>3`) `e_star`/`v_star` branches are crude approximations (distance powers, centroid fans) — not DEC-exact.
- `_compute_vd_3d_batch` omits the `vd_mid.connect(vd_face)` wiring that the sequential simplex-aware path performs (`_compute_dual.py:495` vs `:749-755`) — batch vs sequential duals differ in boundary dual connectivity.
- `edge_collapse_2d` returns `True` even when the position move is aborted by a cache collision (`remesh/_operations_2d.py:339-343`).
- `Complex.boundary` is documented-unreliable on Delaunay meshes; `boundary_from_simplices` or `SC.boundary()` are the exact replacements.
- `circumcenter` silently falls back to barycenter on degeneracy — circumcentric duals can locally be barycentric without warning (`ddg/_strategies.py:74-75, :96-97, :124-125`).
- ddgclib CLAUDE.md documents the symlink target as `/home/stefan_endres/projects/hyperct/hyperct`; on this machine the working tree is under `/home/endres/` (stale path in docs; the in-repo symlink resolves correctly).
- `adaptive_remesh(mps=...)` parameter is reserved/unused; 3D remesh ops not implemented (roadmap).
