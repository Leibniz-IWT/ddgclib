# Audit: batch vs sequential 3D dual computation disagreement
> Sources checked | Written 2026-07-02 by physics-audit workflow
> (docs_temp/00_INDEX.md, 02_physics_foundations.md §2/§6, code_map/hyperct_upstream.md §4/§9,
> hyperct/ddg/_compute_dual.py, hyperct/ddg/_operators.py, ddgclib/operators/stress.py,
> ddgclib/dynamic_integrators/_integrators_dynamic.py, cases_dynamic/oscillating_droplet/*,
> cases_dynamic/Hagen_Poiseuile_3D/*. Independent re-verification of an earlier audit pass;
> all numbers below re-measured from fresh probes.)

**Item key:** `batch-vs-sequential-3d`
**Verdict:** CONFIRMED_BUG (upstream hyperct; severity **low** for the ddgclib physics
pipeline — a latent dual-connectivity inconsistency with zero effect on production interior
FVM operators and zero effect on the oscillating-droplet/bubble cases)

## 1. What the physics requires

The dual mesh produced by `compute_vd` is the FVM control-volume tessellation: every force
is a flux through dual faces (02_physics_foundations.md §2,
`F_i = Σ_j [−½(p_i+p_j)A_ij + …]`), with the machine-precision invariant "Dual-face closure
Σ_j A_ij = 0 on every interior dual cell, atol 1e-12" (02 §6 invariant table). Two
implementations of the same operation — `_compute_vd_3d_batch` (backend path) and
`_compute_vd_3d[_simplex_aware]` (sequential path) — must produce the *same* dual mesh
(positions AND connectivity), because downstream fan walks (`e_star`, `v_star`,
`_walk_fan_3d`) traverse dual adjacency (`vd.nn`) to order the ring of duals around each
primal edge (code_map/hyperct_upstream.md §5 :114, §8 invariant 7). `compute_vd`'s
docstring (`_compute_dual.py:66-71`) presents `backend` as a pure acceleration option, so
any result difference between the paths is a deviation from intent.

## 2. What the code does

`/home/endres/projects/ddgclib/hyperct/ddg/_compute_dual.py`:

- Path selection (`compute_vd`, **:104-108**): `backend is not None` → `_compute_vd_3d_batch`;
  else `_compute_vd_3d` (→ `_compute_vd_3d_simplex_aware` when `HC._simplices` is cached,
  **:315-317**). There is no size threshold — only the `backend` kwarg selects the path.
- Sequential simplex-aware path, boundary faces (**:466-495**): creates the face-barycenter
  dual `vd_face` (:480), connects tet dual → face dual (:483), creates the 3 edge-midpoint
  duals `vd_mid` (:491) and **connects each to the face dual**: `vd_mid.connect(vd_face)`
  (**:495**, comment "ring-walk support").
- Batch path, boundary faces (**:741-755**): creates `vd_face` (:743), connects tet dual →
  face dual (:748), creates the 3 edge-midpoint duals (:751-755) — **but never connects
  `vd_mid` to anything**. No `connect` call exists for `vd_mid` in the Phase-4 loop.
- Incidentally, the *legacy* sequential nn-path (`_compute_vd_3d`, **:356-363**) also leaves
  its midpoint dual `vd12` unconnected — so the simplex-aware sequential path is the odd one
  out with the extra wiring; legacy-nn and batch agree with each other.

Consumers of that connectivity: `e_star`/`v_star` dim=3 boundary branch
(`/home/endres/projects/ddgclib/hyperct/ddg/_operators.py` **:64-73, :192-200, :329-337**)
select the fan-walk start as "the dual in `dset` with exactly 1 in-set neighbour" and then
walk a hardcoded `iter_len = 3` triangles.

Production wiring: `_retopologize`
(`/home/endres/projects/ddgclib/ddgclib/dynamic_integrators/_integrators_dynamic.py`) calls
`compute_vd(HC, method="barycentric")` **with no backend** (**:210**); its `backend` kwarg
is forwarded only to `batch_e_star` (**:218**), which **skips boundary vertices**
(`_operators.py:420-421`). Boundary vertices get `dual_vol = 0`
(`_integrators_dynamic.py:225`). The only repo call sites that pass a backend to
`compute_vd` are `cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py:255`,
`cases_dynamic/Hagen_Poiseuile_3D/run_cluster.py:239`, and
`ddgclib/tests/test_gpu_backend.py`.

## 3. Probe design

Scripts in
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/batch-vs-sequential-3d/`
(run with `/home/endres/anaconda3/envs/ddg/bin/python`, cwd = repo root):

1. `probe_parity.py` — same primal mesh (jittered 5×5×5 grid in the unit cube: 126 verts;
   `_retopologize` gives 484 Delaunay tets, 192 boundary faces, 288 boundary-surface edges,
   28 interior verts). Run `compute_vd` sequentially (via `_retopologize`, exactly the
   production path) then with `backend=get_backend("numpy")` on the same `HC`; rebuild the
   `batch_e_star(orient=True, compute_volumes=True)` caches identically to `_retopologize`
   :214-230 in each state; diff dual counts/positions, per-interior-vertex `dual_vol`,
   oriented `A_ij`, closure, and `stress_force` on linear `p = 1000 + (5,3,−2)·x` with
   mu=0; direct `e_star()` on all boundary-surface edges.
2. `probe_counts_and_closure.py` — clean dual counts immediately after each fresh
   `compute_vd` (probe 1's sequential count was polluted by midpoint duals get-or-created
   by `batch_e_star` inside `_retopologize`); closure re-measured with the production
   `dual_area_vector` ring-walk; fan-walk stall detection.
3. `probe_droplet_path.py` — spy wrappers on `_compute_vd_3d` / `_compute_vd_3d_batch`
   (module-level monkeypatch, no repo edits); real `setup_oscillating_droplet(dim=3,
   refinement_outer=1, refinement_droplet=1)` + 2 steps of `symplectic_euler` with the
   case's own `retopo_fn`.
4. `probe_boundary_reference.py` — independent ground-truth dual-face vector area per
   boundary edge (closed polygon: edge midpoint → face1 barycenter → adjacency-ordered tet
   barycenters → face2 barycenter, built directly from `HC._simplices` without using dual
   connectivity); compare `Σ e_star` under both connectivities.
5. `probe_link_census.py` — direct census of `vd_mid ↔ vd_face` links per boundary face in
   each state.

## 4. Probe OUTPUT (verbatim numbers)

Mesh: `126 verts, 484 tets, 192 boundary faces, 288 boundary edges, 28 interior verts`.

**Dual mesh diff (probe 2, clean counts):**
```
SEQ   fresh: n_dual=923  n_conn=1558
BATCH fresh: n_dual=923  n_conn=1064          # diff = -494 connections
```
Dual positions identical as sets (probe 1: `|B\A| = 0`; count-clean comparison shows equal
vertex sets).

**Link census (probe 5)** — reconciles the −494 exactly:
```
192 boundary faces -> expected mid-face links = 576
[SEQ  ] mid-face links present=576 absent=0 merged(mid==face)=0
[BATCH] mid-face links present=82  absent=494 merged(mid==face)=0
```
(the 82 "present" under batch are aliasing coincidences where a midpoint position equals
another, connected, dual — degenerate coplanar-surface slivers; 576 − 82 = 494 = the whole
connection deficit.)

**Production-pipeline parity (probe 1, interior vertices, caches built exactly as
`_retopologize` does):**
```
dual_vol: common interior verts=28, max|diff| = 0.000e+00        (bit-identical)
edge areas A_ij (oriented, batch_e_star): max|diff| = 7.758e-18  (machine eps)
stress_force (linear p, mu=0): max|F_seq - F_batch| = 1.465e-14  (max|F| = 6.804)
sum(dual_vol) identical in both states: 0.430863662003
```

**Where the paths actually disagree — boundary-surface-edge `e_star` (probes 1 & 4):**
```
boundary-edge e_star: 240 edges compared, 239 differ (>1e-12),
                      max|diff| = 3.620e-02  (typical |sumA| ≈ 2.1e-02 → O(100%) disagreement)
vd_mid in-set nbr count:              SEQ min=2 max=2 mean=2.00 | BATCH min=0 max=2 mean=0.28
# duals with exactly-1 in-set nbr
  (the documented walk-start contract) SEQ min=0 max=2 mean=0.28 | BATCH min=2 max=2 mean=2.00
e_star walk failures (of 288):        SEQ 41 | BATCH 48
```
vs independent ground truth (probe 4; tets-around-edge histogram
`{1:48, 2:94, 3:35, 4:54, 5:34, 6:19, 7:4}`):
```
[SEQ  ] 247 edges (41 failures): median rel err=1.009, max=2.028, exact(<1e-12)=14
[BATCH] 240 edges (48 failures): median rel err=1.455, max=2.053, exact(<1e-12)=43
```
i.e. boundary-edge `e_star` is ~100% wrong under **both** connectivities. The
`vd_mid.connect` extra wiring in the sequential path actually *breaks* the walk-start
contract ("boundary start = dual with exactly 1 in-set neighbour"): it closes the dual ring
so **no** valid start exists (mean 0.28 candidates) and the walk starts at an arbitrary
set-iteration element; the batch path leaves an open chain with exactly 2 deterministic
start candidates (and is machine-exact on nearly all 1-tet boundary edges: 43 of 48). On
multi-tet edges both paths are truncated by the hardcoded `iter_len = 3`
(`_operators.py:71/:198/:335`) — a separate, deeper `e_star` boundary defect that dominates
the error under either connectivity.

**Which path the oscillating-droplet 3D case uses (probe 3, runtime spy):**
```
droplet 3D mesh: 93 vertices
after setup:               {'seq_dispatch': 1, 'seq_simplex_aware': 1, 'seq_legacy_nn': 0, 'batch': 0}
after 2 integrator steps:  {'seq_dispatch': 3, 'seq_simplex_aware': 3, 'seq_legacy_nn': 0, 'batch': 0}
VERDICT: droplet 3D uses SEQUENTIAL (simplex-aware)
```

**Incidental observation (out of scope, path-INDEPENDENT):** on this jittered Delaunay box
mesh the interior dual-cell closure is violated at ~7e-3 under *both* paths and *both* area
methods (`batch_e_star` areas: 6.773e-03; production `dual_area_vector`: 6.984e-03; worst
vertex is boundary-adjacent, 6/14 nbrs on the hull; no fan-walk stalls detected;
`len(tris) == len(dset)` on all 446 interior edges). This contradicts the documented 1e-12
closure invariant (02_physics_foundations.md §6), which is pinned only on pristine
symmetric `Complex.triangulate()` meshes (`ddgclib/tests/test_stress.py:596-640`). Since it
is identical across both code paths it is NOT caused by this item — recommend a separate
audit item (likely degenerate coplanar-hull slivers / merged duals corrupting interior
rings, cf. the dual-closure-antisymmetry audit file).

## 5. Verdict and reasoning

**CONFIRMED_BUG, severity low.** The two code paths demonstrably produce *different dual
meshes*: identical dual-vertex sets (923 = 923, positions bit-equal) but 494 missing
`vd_mid ↔ vd_face` connections in the batch path (100% of non-aliased mid–face links), and
O(100%) disagreement in direct boundary-edge `e_star` results (239/240 edges differ, max
3.6e-2 on |A| ≈ 2e-2). This is a genuine implementation inconsistency, exactly as flagged
in code_map/hyperct_upstream.md §9 (":495 vs :749-755").

Severity is low because the production physics pipeline is unaffected: `_retopologize`
builds `dual_vol` and `_edge_area_cache` via `batch_e_star`, which skips boundary vertices
and walks only interior edges whose dual rings are closed rings of tet duals that never
touch `vd_mid`/`vd_face` adjacency — probes show **bit-identical** dual volumes (diff 0.0),
edge areas (7.8e-18) and stress forces (1.5e-14, pure fp noise) between the paths.
Boundary vertices are frozen with `dual_vol = 0` and "boundary edges contribute zero flux"
by design (02_physics_foundations.md, special-casing note). The only consumers of the
differing connectivity are direct `e_star`/`v_star` calls on boundary–boundary surface
edges (e.g. the `cache_dual_volumes`/`dual_volume` fallback, `stress.py:356-370`, which in
this environment never fires because `batch_e_star` imports fine) — and those calls are
~100% wrong under *both* connectivities anyway due to the `iter_len=3` hardcode and the
start-selection contract that the sequential path's ring closure makes unsatisfiable.

## 6. Droplet / bubble impact

**None.** The runtime spy proves `oscillating_droplet_3D.py` (via `_retopologize_multiphase`
→ `_retopologize` → `compute_vd` with no backend, `_integrators_dynamic.py:210`) executes
only `_compute_vd_3d_simplex_aware` — at setup and at every integrator step. No
droplet/bubble case passes `backend=` to `compute_vd` (only `Hagen_Poiseuile_3D` and the
GPU tests do). Even if a backend were used, the interior force/volume pipeline is
bit-identical between paths, so droplet dynamics would not change; multiphase interface
vertices are interior to the domain and never hit the boundary branch of the fan walk.

## 7. Suggested fix

1. In `_compute_vd_3d_batch` Phase 4 (`_compute_dual.py:749-755`), wire the midpoint duals
   like the simplex-aware path does at :495:
   ```python
   for va, vb in ((v1, v2), (v1, v3), (v2, v3)):
       cd_mid = va.x_a + 0.5 * (vb.x_a - va.x_a)
       vd_mid = HC.Vd[tuple(cd_mid)]
       va.vd.add(vd_mid)
       vb.vd.add(vd_mid)
       vd_mid.connect(vd_face)      # <-- missing line
   ```
   and add the same wiring to the legacy nn path (`:356-363`) so all three paths agree.
2. Independently (bigger payoff): fix the 3D boundary fan walk itself —
   `e_star`/`v_star`/`_walk_fan_3d` hardcode `iter_len = 3`
   (`_operators.py:71/:198/:335`), which truncates the fan on any boundary edge with >1
   incident tet (83% of boundary edges on the probe mesh), and the "exactly-1 in-set
   neighbour" start selection is unsatisfiable once `vd_mid` closes the ring. Correct
   behaviour: walk the full open chain (length `len(dset) − 1` excluding `vd_mid`, or walk
   until termination), starting deterministically from a face dual. Note the fix order
   matters: applying fix 1 alone makes the *batch* path inherit the sequential path's
   ambiguous-start behaviour on boundary edges, so fix 2 should land with (or before) fix 1.
3. Add a regression test comparing batch vs sequential duals (dual-vertex count,
   connection count, and boundary `e_star`) on a jittered Delaunay mesh — nothing in the
   current suite covers this parity.
