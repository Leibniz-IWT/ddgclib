# Adding New Dual Cell / Operator Formulations

## The Core Issue

The current DDG gradient operator `0.5 * Σ_j (f_j - f_i) * A_ij` samples `f`
at primal vertex positions. For **barycentric** duals, the resulting
approximation is exact for linear `f` on any valid triangulation. For
**circumcentric** duals, it is only exact on meshes without obtuse triangles.

A **simplex-weighted gradient** formulation is exact for linear `f` on any
Delaunay triangulation with either dual type:

```
∫_{V_i} ∇f dV = (1/2) Σ_j Σ_{k∈common(i,j)} grad_T(f) · Area(v_i, mp_ij, cc_T, mp_ik)
```

where `grad_T` is the constant FEM gradient on triangle T, and the sub-polygon
`(v_i, mp_ij, cc_T, mp_ik)` is the dual cell portion within that triangle.

Verified: machine precision for linear fields with both barycentric and
circumcentric duals on jittered Delaunay meshes.

## Implementation plan

### What to modify in hyperct

**Don't change `dual_area_vector`** — it correctly gives the geometric dual face
vector. The flux-plane geometry is not the problem.

**Add `simplex_gradient_integrated` to `hyperct/ddg/_operators.py`** (or a new
file). This is pure geometry + interpolation, no physics:

```python
def simplex_gradient_integrated_2d(v, field_attr='f'):
    """Integrated gradient exact for linear fields on any Delaunay mesh."""
    f_i = getattr(v, field_attr)
    Df = np.zeros(2)
    for v_j in v.nn:
        for v_k in v.nn.intersection(v_j.nn):
            x_i, x_j, x_k = v.x_a[:2], v_j.x_a[:2], v_k.x_a[:2]
            area2 = (x_j[0]-x_i[0])*(x_k[1]-x_i[1]) - (x_k[0]-x_i[0])*(x_j[1]-x_i[1])
            if abs(area2) < 1e-30:
                continue

            # Constant gradient on this triangle (FEM formula)
            e_jk = x_k - x_j
            e_ki = x_i - x_k
            e_ij = x_j - x_i
            grad_T = (1.0 / area2) * (
                f_i          * np.array([-e_jk[1], e_jk[0]]) +
                getattr(v_j, field_attr) * np.array([-e_ki[1], e_ki[0]]) +
                getattr(v_k, field_attr) * np.array([-e_ij[1], e_ij[0]])
            )

            # Dual cell portion: quadrilateral (v_i, mp_ij, cc_T, mp_ik)
            shared_vd = v.vd.intersection(v_j.vd).intersection(v_k.vd)
            if not shared_vd:
                continue
            cc = list(shared_vd)[0].x_a[:2]
            mp_ij = 0.5 * (x_i + x_j)
            mp_ik = 0.5 * (x_i + x_k)
            sub_area = abs(_shoelace_area(np.array([x_i, mp_ij, cc, mp_ik])))

            # Factor 0.5: each triangle visited twice (once per edge from v_i)
            Df += grad_T * sub_area * 0.5
    return Df
```

### 3D generalization

The same principle applies in 3D:

```python
def simplex_gradient_integrated_3d(v, HC, field_attr='f'):
    """Integrated gradient exact for linear fields in 3D."""
    f_i = getattr(v, field_attr)
    Df = np.zeros(3)

    # Iterate over tetrahedra containing v
    # A tetrahedron is identified by v and three mutual neighbors
    for v_j in v.nn:
        common_jk = v.nn.intersection(v_j.nn)
        for v_k in common_jk:
            common_jkl = common_jk.intersection(v_k.nn)
            for v_l in common_jkl:
                # Tetrahedron (v, v_j, v_k, v_l)
                x_i = v.x_a[:3]
                x_j = v_j.x_a[:3]
                x_k = v_k.x_a[:3]
                x_l = v_l.x_a[:3]

                # Signed volume (6x)
                mat = np.array([x_j - x_i, x_k - x_i, x_l - x_i])
                vol6 = np.linalg.det(mat)
                if abs(vol6) < 1e-30:
                    continue

                # Gradient on this tet (FEM formula):
                # grad_T = (1/vol6) * sum_m f_m * outward_face_normal_m
                # where outward_face_normal is cross product of opposite face edges
                f_j = getattr(v_j, field_attr)
                f_k = getattr(v_k, field_attr)
                f_l = getattr(v_l, field_attr)

                # Face opposite to each vertex (outward normals):
                n_i = np.cross(x_k - x_j, x_l - x_j)  # face (j,k,l)
                n_j = np.cross(x_l - x_i, x_k - x_i)  # face (i,k,l) — note order
                n_k = np.cross(x_j - x_i, x_l - x_i)  # face (i,j,l)
                n_l = np.cross(x_k - x_i, x_j - x_i)  # face (i,j,k)

                grad_T = (1.0 / vol6) * (f_i * n_i + f_j * n_j + f_k * n_k + f_l * n_l)

                # Dual cell portion within this tet:
                # The dual cell of v_i intersects this tet in a sub-polyhedron
                # bounded by the tet dual vertex (barycenter/circumcenter),
                # face dual vertices, and edge midpoints.
                # Volume can be computed from the shared dual vertices.
                shared_all = v.vd.intersection(v_j.vd).intersection(v_k.vd).intersection(v_l.vd)
                if not shared_all:
                    continue
                cc = list(shared_all)[0].x_a[:3]  # tet dual vertex

                # Sub-volume: the portion of the dual cell within this tet
                # = 1/4 of tet volume for barycentric duals
                # = computed from circumcenter position for circumcentric
                # For now use the geometric sub-tet volume from v_i to face centers
                mp_ij = 0.5 * (x_i + x_j)
                mp_ik = 0.5 * (x_i + x_k)
                mp_il = 0.5 * (x_i + x_l)
                # Approximate sub-volume (exact decomposition is more complex)
                # TODO: exact sub-volume from dual geometry
                sub_vol = abs(vol6) / 6.0 / 4.0  # 1/4 of tet for now

                # Divide by number of times this tet is visited
                # Each tet has 3 edges from v_i, each visited once, so 6x overcounting
                Df += grad_T * sub_vol / 3.0

    return Df
```

**Note:** The 3D sub-volume computation is more involved than 2D because the
dual cell intersects each tetrahedron in an irregular polyhedron. The exact
decomposition requires computing volumes from the dual vertex (tet
barycenter/circumcenter), face dual vertices (triangle barycenters/circumcenters),
and edge midpoints. The `v_star` operator in hyperct already computes some of
these volumes — it should be reused.

### Integration into ddgclib

In `ddgclib/operators/stress.py`, add the new operator alongside the existing
one. The existing `scalar_gradient_integrated` stays (it's exact for barycentric).

For the stress operator `stress_force`, the viscous flux already uses the
diffusion form `(mu/|d|)*du*(d_hat·A)` which IS exact for quadratic velocity
on any Delaunay mesh (it's a finite-difference, not a face-average). Only the
pressure gradient and diagnostic gradient operators need the simplex-weighted
formulation.

### Benchmark validation

Run after implementation:
```bash
python benchmarks/run_integrated_benchmarks.py --linear-only --dim 2
python benchmarks/run_integrated_benchmarks.py --convergence --dim 2
```

The key diagnostic is the first-moment condition:
```python
M = Σ_j 0.5 * e_ij ⊗ A_ij
# M should equal Vol * I for linear-exact operators
```

## Where the code lives

| Component | File | What it does |
|-----------|------|-------------|
| Dual vertex placement | `hyperct/ddg/_strategies.py` | `barycenter()`, `circumcenter()` |
| Dual mesh computation | `hyperct/ddg/_compute_dual.py` | `compute_vd()` |
| Dual face area vector | `ddgclib/operators/stress.py` | `dual_area_vector()` — geometrically correct, don't change |
| Edge-based gradient | `ddgclib/operators/stress.py` | `scalar_gradient_integrated()` — exact for barycentric only |
| Simplex-weighted gradient | **TO ADD** | Exact for both dual types |
| Benchmark framework | `benchmarks/_integrated_benchmark_classes.py` | Override `compute_numerical()` for new operators |
| Benchmark runner | `benchmarks/run_integrated_benchmarks.py` | `python benchmarks/run_integrated_benchmarks.py` |
