# 3D Simplex-Aware Dual Construction Fix

**Date:** 2026-04-13
**Scope:** `hyperct/ddg/_compute_dual.py`, `ddgclib/operators/stress.py`,
`ddgclib/dynamic_integrators/_integrators_dynamic.py`,
`hyperct/ddg/_dual_cell.py`, `ddgclib/tests/test_stress.py`

## Problem Diagnosed

The 3D barycentric DDG integrated gradient operator

$$Df_i = \tfrac{1}{2} \sum_j (f_j - f_i)\, \mathbf{A}_{ij}$$

was **not** satisfying the linear precision identity

$$\tfrac{1}{2} \sum_j \mathbf{d}_{ij} \otimes \mathbf{A}_{ij} = \text{Vol}_i \cdot \mathbf{I}$$

on realistic Lagrangian meshes.  Interior-to-interior edges on symmetric
meshes achieved machine precision, but **interior-to-boundary edges on
jittered / advected meshes had ~O(h) relative error (~14%)**.

### Root cause chain

1. **Lagrangian advection** moves interior vertices each time step; wall
   BCs keep boundary vertices **fixed** on the domain surface.

2. **`_retopologize`** disconnects all edges and runs
   `scipy.spatial.Delaunay` on the perturbed vertex cloud.  The
   fixed-boundary + jittered-interior geometry produces many **sliver
   tetrahedra** at the boundary layer.

3. **Slivers give boundary vertices high mutual edge connectivity.**  A
   boundary vertex ends up connected to many interior vertices via
   different simplices.

4. **`_compute_vd_3d` used `v1.nn ∩ v2.nn ∩ v3.nn`** to find the
   tetrahedra sharing face `(v1, v2, v3)`.  On sliver-rich boundary
   meshes, this returns **"ghost tets"** — groups of 4 mutually
   connected vertices that are *not* a single simplex.

5. When the code picked a ghost-tet vertex (`list(...)[0]` or `[1]`),
   the dual vertex was placed at the wrong position, and the
   connection `vd1.connect(vd2)` linked the wrong pair, leaving the
   correct pair unconnected.  The dual vertex ring
   (`vd.nn`) was broken.

6. A broken ring caused the p_ij area vector algorithm to fall back
   to angular sorting, losing machine precision.

### Evidence (jittered [0,1]³ mesh, seed=42)

| Refine | Ghost faces | p_ij max \|off-diag\| (before) |
|--------|-------------|-------------------------------|
| 1      | 0           | 1.04e-17 (ok — too few vertices to hit the issue) |
| 2      | 90          | **7.37e-04** (broken) |
| 3      | 868         | **1.62e-04** (broken) |

Symmetric mesh at refine=2: **0 ghost faces, 5.4e-19** (machine eps).
Conclusion: the breakdown was specific to the re-Delaunay step on
advected meshes.

## Fix Implemented

The fix is **simplex-aware dual construction**: instead of inferring
tetrahedra from edge connectivity via nn-intersection, use the explicit
simplex list produced by the Delaunay triangulation.

### 1. New function `_compute_vd_3d_simplex_aware`

Added to [`hyperct/ddg/_compute_dual.py`](../../hyperct/hyperct/ddg/_compute_dual.py).
Takes the same signature as `_compute_vd_3d` and produces the same output,
but reads simplices from `HC._simplices` instead of using nn-intersection.

Algorithm:

```
1. For each simplex (v1, v2, v3, v4):
   - Compute its dual vertex vd at strategy(simplex)
   - Associate vd with all 4 primal vertices (v.vd.add(vd))

2. Build face → simplices map:
   For each simplex, for each of its 4 faces, add to map.
   Interior face: maps to 2 simplices
   Boundary face: maps to 1 simplex

3. For each face in the map:
   - 2 simplices → connect their duals (vd1.connect(vd2))
   - 1 simplex → create face barycenter dual and edge midpoint duals;
                 connect vd_tet → vd_face and vd_mid → vd_face
```

This algorithm never uses nn-intersection.  It enumerates exactly the
tets that exist in the triangulation, so ghost tets cannot appear.

### 2. Dispatcher in `_compute_vd_3d`

The existing `_compute_vd_3d` was modified to check for
`HC._simplices` at entry and dispatch to the new path:

```python
def _compute_vd_3d(HC, strategy, cdist):
    if getattr(HC, '_simplices', None):
        _compute_vd_3d_simplex_aware(HC, strategy, cdist)
        return
    # ... legacy nn-intersection path (unchanged) ...
```

This is **fully backward compatible**: code that doesn't set
`HC._simplices` gets the legacy behavior.

### 3. Simplex caching in `_retopologize`

Modified [`ddgclib/dynamic_integrators/_integrators_dynamic.py`](../ddgclib/dynamic_integrators/_integrators_dynamic.py).
After the Delaunay call and edge-connect loop, cache the simplex list:

```python
if dim == 3:
    HC._simplices = [
        tuple(verts[s[i]] for i in range(4))
        for s in tri.simplices
    ]
```

This is the single critical threading: `_retopologize` already has the
simplex list in scope (`tri.simplices`), so the cost is one list
comprehension per retopologization.

### 4. p_ij dual area vector (pre-existing)

The p_ij construction in
[`ddgclib/operators/stress.py`](../ddgclib/operators/stress.py)
(`_dual_area_vector_3d_p_ij`) interleaves tet barycenters with face
barycenters `(x_i + x_j + x_k)/3` to produce the correct 3D dual face
polygon.  It relies on a valid `vd.nn` ring from `compute_vd` — which
is what the simplex-aware fix restores.

### 5. `dual_cell_faces_3d` consistency

Updated [`hyperct/ddg/_dual_cell.py`](../../hyperct/hyperct/ddg/_dual_cell.py)
to use connectivity-based ring walking (same as the operator) and
interleave face barycenters, so the analytical reference faces match
the DDG operator's geometry.  An `include_face_barycenters=True`
parameter controls this (default `True`).

### 6. Regression test

Added `test_p_ij_linear_precision_jittered_3d` in
[`ddgclib/tests/test_stress.py`](../ddgclib/tests/test_stress.py)
which asserts machine precision (`atol=1e-14`) on a jittered [0,1]³
mesh at refine=2 — the exact case that was broken before.

## Results

### Linear precision identity

| Mesh / method | Before fix | After fix |
|---|---|---|
| Symmetric refine=2 | 5.4e-19 (ok) | **5.4e-19** |
| Jittered refine=1 | 1.0e-17 | **9.5e-18** |
| Jittered refine=2 | **7.4e-04** (broken) | **4.0e-18** |
| Jittered refine=3 | **1.6e-04** (broken) | **1.1e-18** |

### Ghost face count (unchanged by the fix)

The nn-intersection still finds ghost tets (the mesh geometry is
unchanged), but `_compute_vd_3d_simplex_aware` no longer uses
nn-intersection, so ghost tets no longer cause errors.

### Multi-seed robustness

Tested seeds {1, 7, 42, 100, 2026}: all give `max|off-diag| < 1e-17`.

### Test suite

- `pytest ddgclib/tests/ -m "not slow"`: **702 passed**, 0 failed
- `pytest ddgclib/tests/ -m "slow" -k "3d"`: **8 passed**, 0 failed
  (Hagen-Poiseuille 3D, hydrostatic 3D)
- `pytest hyperct/tests/`: **269 passed**, 0 failed

## Backward Compatibility

- Code that does not set `HC._simplices` gets the legacy nn-intersection
  path unchanged.  This includes direct use of `Complex` + `compute_vd`
  outside the dynamic integrators.
- All existing tests pass with no modifications.
- `dual_cell_faces_3d` gains an optional `include_face_barycenters=True`
  parameter; default behavior matches new expectations.  Callers passing
  `include_face_barycenters=False` get the pre-fix behavior.

## Files Modified

- `hyperct/ddg/_compute_dual.py` — added `_compute_vd_3d_simplex_aware`, dispatch
- `hyperct/ddg/_dual_cell.py` — ring-walk + face-barycenter interleaving
- `ddgclib/operators/stress.py` — `_dual_area_vector_3d_p_ij` (pre-existing p_ij fix)
- `ddgclib/dynamic_integrators/_integrators_dynamic.py` — `HC._simplices` cache
- `ddgclib/tests/test_stress.py` — regression test for jittered 3D

## Follow-up Opportunities

1. **Apply the simplex-aware pattern elsewhere.**  Any code path that
   manually re-triangulates a 3D mesh and then calls `compute_vd`
   should also cache `HC._simplices`.  Candidates to check:
   - Cube flow, oscillating droplet, and other dynamic case studies
   - Any multiphase case that does its own retriangulation
   - Case-study setup scripts that build a mesh from scratch

2. **Apply to `_compute_vd_3d_batch`.**  The batch path in
   `hyperct/ddg/_compute_dual.py` also uses nn-intersection; should be
   updated to the simplex-aware path when `HC._simplices` is available.

3. **Delaunay.neighbors optimization.**  Instead of building a
   face→simplex map, use `scipy.spatial.Delaunay.neighbors` (tet
   adjacency graph) for O(N) tet-to-tet connection lookup.  Cleaner
   and slightly faster.

4. **Interface sliver tets in multiphase.**  The same fixed-vs-advected
   geometry occurs at sharp multiphase interfaces.  The fix should
   already apply there when the multiphase integrator calls
   `_retopologize_multiphase`, but this needs verification on the
   oscillating droplet / dam break cases.
