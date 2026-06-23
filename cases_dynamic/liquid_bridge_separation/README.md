# Liquid bridge separation benchmark cases 1 to 12

This folder contains a reproducible archive for the Pitois liquid bridge
separation comparison used to test ddgclib based force, pressure, volume, and
contact line handling.

The committed files are the source files needed to rerun the twelve cases and
to regenerate the GIFs. Generated histories, mesh time series, PNG frame
folders, and GIF outputs are not committed because they are large and can be
regenerated from the scripts. The PowerPoint file is included as the compact
visual report of the completed runs.

Included presentation:

```text
20260612_liquid_bridge_solver_comparison_12cases.pptx
```

## What was done

Twelve liquid bridge separation cases were collected into one package. The
cases compare the same Pitois Fig. 5 style separation benchmark while changing
one or two numerical ingredients at a time:

```text
initial mesh radius
pressure closure
volume projection
viscous force path
hydrostatic pressure force
PR33 pressure force
contact line outward growth
pressure gauge
```

The main target is to identify which parts of the ddgclib liquid bridge
workflow must be robust before the benchmark can be matched.

## Main conclusion

The benchmark is matched only when the bridge is effectively incompressible and
the initial mesh matches the measured time zero geometry.

Effective incompressibility can come from the incompressible solver or from a
compressible labelled case with the incompressible limit pressure projection.
Pure compressible pressure correction can add unwanted force near the contact
line, which changes contact line motion, geometry, and axial force.

The initial mesh must match the measured time zero state, especially
contact line radius, neck radius, volume, and initial force. If the initial mesh
is wrong, a stable solver can still produce the wrong force curve.

## Initial meshes

Two initial meshes are used.

```text
initial_mesh/rcl_1p540_mm/mesh/rcl_1p540_mm.msh
initial_mesh/rcl_1p486_mm/mesh/rcl_1p486_mm.msh
```

The first mesh uses the nominal contact line radius near 1.540 mm. It is used
by cases 1, 2, 5, 6, 7, 8, 9, 10, and 11.

The second mesh is force calibrated at time zero and has contact line radius
near 1.486 mm. It is used by cases 3, 4, and 12.

Each mesh folder also contains three display PNGs:

```text
display_pngs/mesh_xz.png
display_pngs/mesh_xyz.png
display_pngs/mesh_top.png
```

Regenerate the initial meshes:

```bash
cd cases_dynamic/liquid_bridge_separation
python initial_mesh/rcl_1p540_mm.py
python initial_mesh/rcl_1p486_mm.py
```

The generator scripts document the mesh equations in code comments. The
1.540 mm mesh uses the Pitois geometry, contact plane relation, smooth
volume matched surface profile, Pappus volume estimate, and Gmsh tetrahedral
volume correction. The 1.486 mm mesh solves for the contact line radius that
matches the time zero Pitois force using the ddgclib/FHeron gorge force path.

## Case map

The numbered case files are intentionally thin wrappers. Each wrapper prints
the selected process steps, then calls the common runner.

```text
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case1.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case2.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case3.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case4.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case5.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case6.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case7.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case8.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case9.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case10.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case11.py
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12.py
```

Exact CLI options for all cases are in:

```text
scripts/_ddgclib_case_runner.py
```

Summary of the tested changes:

| Case | Initial mesh | Closure | Viscous force | Pressure add-ons | rCL handling |
| --- | --- | --- | --- | --- | --- |
| 1 | rCL 1.540 mm | compressible | edge flux | none | Pitois Eq. 6 rCL; outward growth allowed |
| 2 | rCL 1.540 mm | incompressible | edge flux | none | Pitois Eq. 6 rCL; outward growth constrained |
| 3 | rCL 1.486 mm | compressible label plus limit projection | edge flux | none | loaded rCL; contact-angle update; outward growth allowed |
| 4 | rCL 1.486 mm | incompressible | edge flux | none | loaded rCL; outward growth constrained |
| 5 | rCL 1.540 mm | compressible | tet Cauchy | none | Pitois Eq. 6 rCL; outward growth allowed |
| 6 | rCL 1.540 mm | incompressible | tet Cauchy | none | Pitois Eq. 6 rCL; outward growth constrained |
| 7 | rCL 1.540 mm | compressible | edge flux | hydrostatic top CL reference | Pitois Eq. 6 rCL; outward growth constrained |
| 8 | rCL 1.540 mm | incompressible | edge flux | hydrostatic top CL reference | Pitois Eq. 6 rCL; outward growth constrained |
| 9 | rCL 1.540 mm | compressible label plus limit projection | tet Cauchy | PR33 Fp, p_ref = 0; hydrostatic top CL reference | Pitois Eq. 6 rCL; outward growth constrained; PR33 CL mobility on |
| 10 | rCL 1.540 mm | incompressible | tet Cauchy | PR33 Fp, p_ref = 0; hydrostatic top CL reference | Pitois Eq. 6 rCL; outward growth constrained; PR33 CL mobility on |
| 11 | rCL 1.540 mm | compressible label plus limit projection | tet Cauchy | PR33 Fp, p_ref = p_Heron; hydrostatic top CL reference | Pitois Eq. 6 rCL; outward growth constrained; PR33 CL mobility on |
| 12 | rCL 1.486 mm | compressible label plus limit projection | tet Cauchy | PR33 Fp, p_ref = 0; hydrostatic top CL reference | Pitois Eq. 6 rCL target; outward growth constrained; PR33 CL mobility on |

Here "outward growth" means the solver `allow_contact_line_growth` flag.
Cases 1, 3, and 5 allow outward rCL growth. This is separate from the Pitois
Eq. 6 effective rCL choice and from PR33 contact-line mobility.

## What worked and why

Cases with good benchmark agreement use an effectively incompressible pressure
update and a time zero mesh consistent with the measured geometry and initial
force. This is why the force calibrated 1.486 mm mesh is important for cases
3, 4, and 12.

Case 3 keeps the compressible label, but uses the incompressible limit
projection and allows the contact line to grow outward. That prevents the bad
collapse seen when compressible pressure correction was combined with a fixed
outward contact line constraint.

Case 12 keeps the compressible label, uses the incompressible limit projection,
uses PR33 pressure force with fixed zero gauge, and uses the force calibrated
mesh. That combination matches the benchmark better than using the 1.540 mm
initial mesh.

Case 11 is numerically stable with p_ref = p_Heron, but it changes the final
force level and distorts the benchmark force comparison. For this benchmark,
p_ref = 0 is the better PR33 pressure gauge.

## ddgclib connection

The capillary force path is the ddgclib/FHeron path in all twelve cases.
The solver files keep this explicit because the purpose is to improve the
ddgclib liquid bridge workflow, not to build an unrelated standalone solver.

The tetrahedral Cauchy viscous term is included in cases 5, 6, 9, 10, 11, and
12. The edge flux viscous term is included in cases 1, 2, 3, 4, 7, and 8.

PR33 means the pressure force from the volume gradient operator:

```text
B = dV/dx
F_p = B^T p
```

For cases 9 to 12, PR33 is on, so F_p is added to the nodal solver force. For
cases 1 to 8, this pressure force term is off.

The projection form is:

```text
B M^{-1} B^T p = volume residual
```

In code, this is assembled through the local arrays named volume matrix,
mobility, pressure matrix, and stiffness. The README uses B only as the
mathematical operator notation.

The optional bulk viscosity term zeta div(u) I is implemented as a command line
option in the solver, but it is not selected by these twelve canonical cases.

## Important functions

Case selection and process printing:

```text
scripts/_ddgclib_case_runner.py
  CASE_ARGS
  show_case_process()
  run_case()
```

Base solver used by cases 1 to 8:

```text
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib.py
```

Core solver used by cases 9 to 12:

```text
scripts/_ddgclib_case_core.py
```

Key solver functions:

```text
_gmsh_surface_heron_force_map()
_surface_tension_force_heron()
_gmsh_tet_cauchy_viscous_force_vector_cache()
_apply_gmsh_incompressible_projection()
_gmsh_solver_hydrostatic_force_map()
_gmsh_heron_pressure_reference()
_update_moving_contact_line()
_write_pitois_fig5_comparison()
```

Local PR33 operator copy:

```text
scripts/pr33_operators.py
  multiphase_sparse_compressible_eos_pressure_correction()
  multiphase_tet_volume_lumped_masses()
  heron_forces_for_points()
  pressure_equivalent_from_forces()
```

## How to run one case

Run from this folder inside a ddgclib checkout:

```bash
cd cases_dynamic/liquid_bridge_separation/scripts
python v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12.py
```

The case creates an output folder beside the Python file:

```text
scripts/v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12/
```

## How to run all 12 cases

```bash
cd cases_dynamic/liquid_bridge_separation/scripts
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  python "v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case${n}.py"
done
```

## How to regenerate charts and GIFs

After the case output folders exist:

```bash
cd cases_dynamic/liquid_bridge_separation/scripts
python plot_case_chart_vs_t.py --only case12
python refresh_case_fig5_compare.py --only case12
python render_meshbatch_fixed_top_cap.py --only v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12
python render_case_diagnostic_gif.py --only case12 --ppt-optimized
```

Render all diagnostic GIFs after all cases finish:

```bash
cd cases_dynamic/liquid_bridge_separation/scripts
python render_meshbatch_fixed_top_cap.py
python render_case_diagnostic_gif.py --all --ppt-optimized
```

If fixed top PNG frames already exist and only the mesh GIF needs to be rebuilt:

```bash
python render_meshbatch_fixed_top_cap.py --gif-only --only v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12
```

## Why Python files are needed

The `.msh` files only store geometry at one state. The output PNGs and GIFs
only show saved results. The Python files are needed because they define:

```text
pressure closure
volume projection
contact line outward growth rule
PR33 pressure gauge
hydrostatic pressure reference
viscous force path
ddgclib/FHeron force path
recording and GIF generation settings
```

Without the Python files, the same initial mesh can produce a different result.

## Literature references

Pitois, Moucheront, and Chateau, Journal of Colloid and Interface Science,
231, 26 to 31, 2000. Used for the liquid bridge benchmark geometry, Fig. 5
force curve, and the contact radius relation.

Chorin, Mathematics of Computation, 22, 745 to 762, 1968. Used as the
projection method reference for the incompressible pressure correction idea.

Meyer, Desbrun, Schroder, and Barr, Discrete Differential Geometry Operators
for Triangulated 2 Manifolds, 2003. Used as the DDG geometry background for
the Heron and dual area style geometry used by ddgclib.

Batchelor, An Introduction to Fluid Dynamics, 1967. Used as the continuum
mechanics reference for the Newtonian Cauchy stress form used in the
tetrahedral viscous force tests.

## Quick source check

```bash
cd cases_dynamic/liquid_bridge_separation
python -m py_compile scripts/*.py initial_mesh/*.py initial_mesh/rcl_1p540_mm/scripts/*.py initial_mesh/rcl_1p486_mm/scripts/*.py
```
