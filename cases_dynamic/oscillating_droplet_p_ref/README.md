

# Oscillating droplet pressure-reference benchmark

This folder contains the benchmark material used to compare EOS pressure,
local incompressible projection, internal ALE flux, pairwise pressure trials,
and Delaunay retopology for a Rayleigh-Lamb oscillating droplet. The main
addition is a tet-volume pressure-reference operator that can replace the
face-average pressure flux in an oscillating-droplet test while still using the
standard dynamic integrator update `du_i/dt = F_i/m_i`, `dx_i/dt = u_i`.

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
- `ddgclib` two-phase liquid/gas triangulation with a multiphase EOS/projection
  pressure-reference solve.

The stable physical cases use a moving Lagrangian tet mesh. For a material
face, the mesh velocity equals the material face velocity, so the internal mass
flux is zero. The vertex positions are still updated dynamically from nodal
forces.

## Achieved Result

The corrected cases reproduce the Rayleigh-Lamb `a2(t)` response for the
selected mesh and time horizon. Cases #5/#6 are ddgclib-backed repeats of the
successful one-phase benchmark. Cases #11/#12 keep a two-phase liquid/gas mesh
and verify that active Delaunay retopology occurs while the liquid pressure
operator remains stable:

| case | closure | key setting | result |
|---|---|---|---|
| #5_ddgclib | compressible EOS | Lagrangian full flux | same `a2(t)` as old #5 |
| #6_ddgclib | incompressible projection | Lagrangian full flux | same `a2(t)` as old #6 |
| #11_ddgclib | compressible EOS | liquid + gas mesh, active retopology | Rayleigh match with nonzero tet flips |
| #12_ddgclib | incompressible projection | liquid + gas mesh, active retopology | Rayleigh match with nonzero tet flips |


For #11/#12 the full liquid-plus-gas vertex cloud is actively rebuilt by
Delaunay retopology. New tetrahedra are reclassified by their persistent
Lagrangian vertex phase, and the liquid EOS/projection pressure operator is
rebuilt from the liquid tet subset. The validation rows report both liquid and
gas tetrahedra, plus more than one thousand changed combined tets, confirming
that retopology happened in the solver path, not only in the visualization.

Representative dynamic-integrator validation at `t = 0.016 s`, `AR = 1.05`,
`R = 1 mm`, `gamma = 0.072 N/m`, and `rho = 1000 kg/m^3`:

| case | final `a2` | Rayleigh `a2` | changed tets | note |
|---|---:|---:|---:|---|
| #5_ddgclib | 0.02810 | 0.02989 | 0 | one-phase compressible EOS |
| #6_ddgclib | 0.02810 | 0.02989 | 0 | one-phase incompressible projection |
| #11_ddgclib | 0.02830 | 0.02989 | 1084 | two-phase compressible, active retopology |
| #12_ddgclib | 0.02830 | 0.02989 | 1060 | two-phase incompressible, active retopology |



### Dev note 

USE ONLY case 11 and 12 for dev purposes.

## Problem Addressed

The upstream oscillating-droplet setup can suffer from a retopology artifact:
after a Delaunay flip, the HC local dual cell around a vertex can change even
when the physical liquid has not locally compressed by that amount. If density
and pressure are computed from that changing HC dual volume, the code can read a
topological connectivity change as a physical compression/expansion event. That
creates an artificial density/pressure jump and changes the droplet frequency.

The benchmark workaround is:

1. use tet volumes directly for pressure and continuity,
2. rebuild the tet-volume gradient matrix after retopology,
3. remap tet mass and target volume from the new tet volumes,
4. lump tet quantities to vertices barycentrically only for nodal mass/volume
   diagnostics.

This works for the benchmark because the pressure equation measures and
corrects exactly the same quantity after every retopology step: tet volume.
The operator `B = dV_t/dx_i`, the target tet volumes, and the nodal masses are
all rebuilt from the same new tet list. The pressure force therefore pushes in
the direction that reduces the measured tet-volume error instead of reacting to
an unrelated HC dual-volume jump.

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

## Benchmark Operator Wrapper

The benchmark-specific operators now live in `scripts/pr33_operators.py`.
The core `ddgclib/operators/stress.py` and
`ddgclib/operators/multiphase_stress.py` files are left on their existing API
surface.  The benchmark runners use local wrapper imports to bind the PR #33
helpers into the unchanged base benchmark scripts.

Important one-phase helpers:

```text
heron_surface_force_from_faces
heron_forces_for_points
tet_volume_matrix
tet_volume_lumped_nodal_values
compressible_eos_pressure_correction
incompressible_volume_projection
tet_face_ale_fluxes
active_retopology_tet_remap
VolumeGradientPressureState
volume_gradient_pressure_acceleration
```

Important multiphase helpers kept out of the core multiphase operator file:

```text
MultiphaseVolumeGradientPressureState
multiphase_volume_gradient_pressure_acceleration
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
sphere_fheron_dynamic_integrator_p_ref.py
sphere_fheron_twophase_gasmesh_benchmark.py
run_github_twophase_oscillating_preview.py
render_github_twophase_triangulation_active_retopology.py
render_dynamic_integrator_twophase.py
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

Run #11/#12 ddgclib dynamic-integrator two-phase active-retopology rows:

```bash
python3 scripts/sphere_fheron_dynamic_integrator_p_ref.py \
  --case-id 11 \
  --subdivision 1 \
  --steps 81 \
  --substeps-per-sample 5 \
  --t-final 0.016 \
  --shape-mode-ar 1.05 \
  --inertia-scale 1.08 \
  --out-dir out/dynamic_integrator_p_ref_case11_multiphase
```

Use `--case-id 12 --out-dir out/dynamic_integrator_p_ref_case12_multiphase`
for the incompressible projection repeat.

Run the built-in dynamic-integrator pressure-reference validation:

```bash
python3 scripts/sphere_fheron_dynamic_integrator_p_ref.py \
  --case-id 5 \
  --subdivision 1 \
  --steps 81 \
  --substeps-per-sample 20 \
  --t-final 0.016 \
  --shape-mode-ar 1.05 \
  --damping-ratio 0 \
  --inertia-scale 1.08 \
  --mass-flux-coupling 1 \
  --momentum-flux-coupling 1 \
  --face-flux-mode lagrangian \
  --density-change-limit 0 \
  --out-dir out/dynamic_integrator_p_ref_case5
```

Use `--case-id 6`, `--case-id 11`, or `--case-id 12` for the other
validation runs. Cases #11/#12 use the integrator `retopologize_fn` callback,
so the pressure state rebuilds the tet list, `B=dV/dx`, masses, and target
volumes before the next `dudt_fn` call.

Render the dynamic-integrator two-phase GIFs:

```bash
python3 scripts/render_dynamic_integrator_twophase.py \
  --case-id 11 \
  --source-dir out/dynamic_integrator_p_ref_case11_multiphase
```

Use `--case-id 12 --source-dir out/dynamic_integrator_p_ref_case12_multiphase`
for the incompressible projection GIF.

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
