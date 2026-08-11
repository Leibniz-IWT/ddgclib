# 3D Simplex-Aware Fix: Benchmark Results

**Date:** 2026-04-13
**Scope:** Verification of `_compute_vd_3d_simplex_aware` fix across integrated benchmarks.

## Summary

| Test | Metric | Result |
|------|--------|--------|
| Linear precision tensor identity (3D barycentric, jittered) | max \|off-diag\| | **4e-18** (machine eps) ✓ |
| DDG integrated gradient vs `a * Vol` (3D barycentric, jittered) | max \|Df - a·Vol\| | **2e-17** (machine eps) ✓ |
| `pytest ddgclib/tests/` (not slow) | pass/fail | **702 passed, 0 failed** ✓ |
| `pytest ddgclib/tests/ -m "slow" -k "3d"` | pass/fail | **8 passed, 0 failed** ✓ |
| `pytest hyperct/tests/` | pass/fail | **269 passed, 0 failed** ✓ |

## Detailed Benchmark Results

### 2D Linear Precision (unchanged by fix — was already working)

All barycentric/p_ij combinations:
- Symmetric: `3e-16` ✓ machine precision
- Jittered: `3e-16` ✓ machine precision

### 3D Linear Precision (the fix target)

**True linear precision** measured via tensor identity
`0.5 * sum d_ij ⊗ A_ij - Vol*I`:

| Method | Mesh | Refine | Before fix | **After fix** |
|--------|------|--------|------------|---------------|
| p_ij barycentric | symmetric | 2 | 5.4e-19 | 5.4e-19 |
| p_ij barycentric | jittered | 1 | 1.0e-17 | **9.5e-18** |
| p_ij barycentric | jittered | 2 | **7.4e-04** ✗ | **4.0e-18** ✓ |
| p_ij barycentric | jittered | 3 | **1.6e-04** ✗ | **1.1e-18** ✓ |

Verified across seeds {1, 7, 42, 100, 2026}: all produce machine precision
after the fix.

### 3D DDG Integrated Gradient vs `a * Vol`

For a linear field `f = x - 2y + 3z`, expected `Df_i = a * Vol_i`:

| Mesh | Refine | Before fix | **After fix** |
|------|--------|------------|---------------|
| Jittered (seed=42) | 2 | 3.09e-03 | **1.74e-17** ✓ |
| Jittered (seed=42) | 3 | 5.84e-04 | **5.93e-18** ✓ |

### 3D Nonlinear Convergence (unchanged by fix)

Quadratic, cubic, trig fields show standard O(h) to O(h²) convergence —
the fix doesn't affect nonlinear accuracy because discretization error
dominates.

### Poiseuille 3D Equilibrium (full stress tensor)

| Mesh | Before fix | After fix |
|------|-----------|-----------|
| Symmetric | 1.7e-03 | 1.7e-03 |
| Jittered (seed=42) | 7.4e-03 | 7.4e-03 |
| Jittered (seed=7) | 8.2e-03 | 8.1e-03 |

The Poiseuille residual has other error sources (face-average pressure
flux, diffusion form for viscosity) that are not addressed by the p_ij
fix.  The fix restores the correctness of `A_ij`, which is necessary
but not sufficient for full machine-precision equilibrium on jittered
meshes.

## Understanding the Benchmark Framework Metric

The integrated benchmark framework compares the DDG operator against an
**analytical surface integral** over the p_ij face polygons
(`integrated_gradient_3d(f, dual_cell_faces_3d(v, HC))`).  For 3D
**barycentric** duals with non-planar p_ij faces, these two quantities
**are not equal** even for linear fields:

- DDG operator: `Df = 0.5 * sum (f_j - f_i) * A_ij` — this is a
  face-average approximation of the flux
- Analytical: `∫ f * n dA` over the non-planar polygon — this depends
  on the exact surface, which for p_ij barycentric faces has
  area-weighted centroid ≠ 0.5*(x_i + x_j)

The benchmark therefore shows ~1e-3 residual errors on 3D barycentric
even though the **true linear precision** (tensor identity) holds at
machine precision.

**For circumcentric duals on Delaunay meshes**, the dual face *is*
perpendicular to the primal edge (centroid *is* at the edge midpoint),
so DDG and analytical agree to machine precision.  That's why
circumcentric shows ~1e-16 in the benchmark while barycentric shows ~1e-3.

**This is not a failure of the DDG operator** — the volume integral
`∫_{V_i} ∇f dV = a * Vol_i` is still exact for linear fields after
the fix.  What's non-exact is the per-face decomposition `∫_{face_ij}
f * n dA`, which is not the quantity the DDG operator computes.

## Conclusion

**Fix verified.**  The simplex-aware `_compute_vd_3d` achieves
machine-precision linear precision for 3D barycentric duals on any
tetrahedral mesh (symmetric, jittered, boundary-adjacent).  This
restores the property that was lost on Lagrangian advected meshes,
where fixed-boundary + perturbed-interior geometry previously
produced ghost tets.

The benchmark framework's surface-integral comparison continues to
show ~1e-3 residuals for 3D barycentric p_ij — this is a **known
property** of the non-planar p_ij face geometry, not a regression.

For applications that require both linear precision **and** face-level
exactness (e.g. sharp-interface multiphase), use **circumcentric duals**
on Delaunay meshes.  For applications that only need volume-level linear
precision (most stress/flux operators), barycentric p_ij with the
simplex-aware fix is sufficient.
