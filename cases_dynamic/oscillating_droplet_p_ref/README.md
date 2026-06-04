# Oscillating droplet pressure-reference benchmark

This folder contains the benchmark material used to compare EOS pressure,
local incompressible projection, internal ALE flux, pairwise pressure trials,
and Delaunay retopology for a Rayleigh-Lamb oscillating droplet.

The presentation is included as:

```text
20260604_Oscillating Droplet Benchmark_twelve_cases.pptx
```

## What Was Tested

The benchmark starts from a small `l=2` shape perturbation of a spherical
droplet and compares the fitted amplitude `a2(t)` against Rayleigh-Lamb theory.
The same geometry is used to test:

- weak-compressible EOS pressure closure,
- local incompressible volume projection,
- Lagrangian material-face ALE flux,
- deliberately nonzero cell-average/Rusanov flux stress tests,
- pairwise `A_ij` pressure trials,
- active Delaunay retopology with a tet-volume pressure closure,
- GitHub `ddgclib` two-phase triangulation visual comparison.

The stable physical cases use a moving Lagrangian tet mesh. For a material
face, the mesh velocity equals the material face velocity, so the internal mass
flux is zero. The vertex positions are still updated dynamically from nodal
forces.

## Achieved Result

The corrected cases reproduce the Rayleigh-Lamb `a2(t)` response for the
selected mesh and time horizon. The ddgclib-backed repeats match the previous
successful benchmark rows to numerical roundoff:

| case | closure | key setting | result |
|---|---|---|---|
| #5_ddgclib | compressible EOS | Lagrangian full flux | same `a2(t)` as old #5 |
| #6_ddgclib | incompressible projection | Lagrangian full flux | same `a2(t)` as old #6 |
| #11_ddgclib | compressible EOS | active retopology | same `a2(t)` as old #11, nonzero tet flips |
| #12_ddgclib | incompressible projection | active retopology | same `a2(t)` as old #12, nonzero tet flips |

For #11/#12 the solver tet list is actively rebuilt. The final comparison run
showed hundreds of changed tets, confirming that retopology happened in the
solver path, not only in the visualization.

## Problem Addressed

The upstream oscillating-droplet setup can suffer from a retopology artifact:
after a Delaunay flip, the HC local dual cell around a vertex can change even
when the physical liquid has not locally compressed by that amount. If density
and pressure are computed from that changing HC dual volume, this can create an
artificial density/pressure jump.

The benchmark workaround is:

1. use tet volumes directly for pressure and continuity,
2. rebuild the tet-volume gradient matrix after retopology,
3. remap tet mass and target volume from the new tet volumes,
4. lump tet quantities to vertices barycentrically only for nodal mass/volume
   diagnostics.

This is a controlled benchmark closure, not a full conservative old-dual to
new-dual overlap remap.

## Main Equations

Rayleigh-Lamb frequency for shape mode `l`:

```text
omega_l^2 = l (l - 1) (l + 2) gamma / (rho R^3)
```

HC/FHeron surface force:

```text
F_i^Heron = -gamma (HN dA)_i
```

Tet-volume continuity operator:

```text
B_{t,i} = partial V_t / partial x_i
Vdot_t = sum_i B_{t,i} . u_i
S = B M^{-1} B^T
```

Weak-compressible EOS correction:

```text
(I + dt^2 D_K S) delta p
  = -D_K (V - V^tar + dt B u*)

D_K = diag(K / V_t^0)
V_t^tar = m_t / rho_0
```

Local incompressible projection:

```text
S p = (r - B u) / dt - B M^{-1} F_np
r = -(V - V^tar) / dt
```

Lagrangian ALE material-face flux:

```text
Phi_m,f = rho^up ((u_f - w_f) . A_f)

for a material face:
u_f = w_f, so Phi_m,f = 0
```

Tet-volume lumping:

```text
V_i = sum_{t contains i} V_t / 4
m_i = sum_{t contains i} m_t / 4
```

Active retopology remap:

```text
new Delaunay tets -> recompute V_t and B
m_t = rho V_t
V_t^tar = scaled V_t
```

## ddgclib Operator Additions

The benchmark adds new functions to `ddgclib/operators/stress.py` and
`ddgclib/operators/multiphase_stress.py`. Existing functions are not changed;
the new operators are added in a separate benchmark add-on block near the top
of each file.

Important one-phase functions:

```text
heron_surface_force_from_faces
heron_forces_for_points
tet_volume_matrix
tet_volume_lumped_nodal_values
compressible_eos_pressure_correction
incompressible_volume_projection
tet_face_ale_fluxes
active_retopology_tet_remap
```

Important multiphase wrappers:

```text
multiphase_tet_volume_matrix
multiphase_tet_volume_lumped_masses
multiphase_compressible_eos_pressure_correction
multiphase_incompressible_volume_projection
multiphase_active_retopology_remap
```

## Included Scripts

The `scripts/` folder contains the Python code used for the benchmark and deck
figures:

```text
sphere_fheron_eos_projection_benchmark.py
sphere_fheron_flux_fv_benchmark.py
sphere_fheron_flux_fv_ddgclib_benchmark.py
sphere_fheron_twophase_gasmesh_benchmark.py
run_github_twophase_oscillating_preview.py
render_github_twophase_triangulation_active_retopology.py
render_twophase_gasmesh_active_retopology.py
render_free_surface_air_mesh_preview.py
plot_github_case_a2_vs_t.py
plot_force_components_vs_t.py
crop_fheron_gifs.py
patch_iwt_office_math_slide3.py
patch_office_math_slide2.py
four_tet_retriangulation_dual_volume_demo.py
four_tet_retriangulation_dual_volume/four_tet_retriangulation_dual_volume_demo.py
```

## Example Commands

Run #5_ddgclib:

```bash
python3 scripts/sphere_fheron_flux_fv_ddgclib_benchmark.py \
  --closure compressible \
  --subdivision 1 \
  --steps 81 \
  --substeps-per-sample 20 \
  --t-final 0.016 \
  --shape-mode-ar 1.05 \
  --damping-ratio 0 \
  --inertia-scale 1.08 \
  --mass-model star-volume \
  --mass-flux-coupling 1 \
  --momentum-flux-coupling 1 \
  --face-flux-mode lagrangian \
  --density-change-limit 0 \
  --out-dir out/case5_ddgclib
```

Run #6_ddgclib:

```bash
python3 scripts/sphere_fheron_flux_fv_ddgclib_benchmark.py \
  --closure incompressible \
  --subdivision 1 \
  --steps 81 \
  --substeps-per-sample 20 \
  --t-final 0.016 \
  --shape-mode-ar 1.05 \
  --damping-ratio 0 \
  --inertia-scale 1.08 \
  --mass-model star-volume \
  --mass-flux-coupling 1 \
  --momentum-flux-coupling 1 \
  --face-flux-mode lagrangian \
  --density-change-limit 0 \
  --out-dir out/case6_ddgclib
```

Run #11/#12 ddgclib active-retopology source rows:

```bash
python3 scripts/sphere_fheron_flux_fv_ddgclib_benchmark.py \
  --closure both \
  --subdivision 1 \
  --steps 81 \
  --substeps-per-sample 5 \
  --t-final 0.016 \
  --shape-mode-ar 1.05 \
  --damping-ratio 0 \
  --inertia-scale 1.08 \
  --mass-model star-volume \
  --mass-flux-coupling 1 \
  --momentum-flux-coupling 1 \
  --face-flux-mode lagrangian \
  --density-change-limit 0 \
  --retriangulation-mode active \
  --out-dir out/case11_12_ddgclib_active_retopology
```

Render the GitHub two-phase triangulation view for #11/#12:

```bash
python3 scripts/render_github_twophase_triangulation_active_retopology.py \
  --source-dir out/case11_12_ddgclib_active_retopology \
  --out-dir out/case11_12_ddgclib_github_twophase_triangulation
```

## Literature References

- Rayleigh, L. (1879). On the capillary phenomena of jets. Proceedings of the
  Royal Society of London, 29, 71-97.
- Lamb, H. (1932). Hydrodynamics, 6th ed. Cambridge University Press.
- Chorin, A. J. (1968). Numerical solution of the Navier-Stokes equations.
  Mathematics of Computation, 22(104), 745-762.
- Temam, R. (1969). Sur un schema d'approximation de la solution des equations
  de Navier-Stokes. Bulletin de la Societe Mathematique de France, 96, 115-152.
- Hirt, C. W., Amsden, A. A., & Cook, J. L. (1974). An arbitrary
  Lagrangian-Eulerian computing method for all flow speeds. Journal of
  Computational Physics, 14, 227-253.
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid
  Dynamics, 3rd ed. Springer.
- Chen, Z., & Przekwas, A. (2010). A coupled pressure-based computational
  method for incompressible/compressible flows. Journal of Computational
  Physics, 229(24), 9150-9165.
