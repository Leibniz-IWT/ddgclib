# Figure 3-8 Volume-Operator Comparison Reproducibility README

This folder reproduces Figure 3-8 from local source scripts, local
geometry generators, and local saved surface states.  The goal is to make the
comparison auditable: a reader should be able to see which literature model is
implemented, which equation is used, which Python file contains it, and which
CSV/PNG/PDF files are regenerated.

## Quick Reproduction

Run from this folder:

```bash
python3 recompute_all_cases.py
```

On Linux/Marie, run the same command from `/home/deng/Figure_3-8_VolumeOperatorComparison`.
The script installs Python packages into the local `_python_deps/` folder if
needed.  The Evrard-type paraboloid panel also needs the C++ helper
`common/irl_paraboloid_clip_volume`; if the executable is missing or belongs
to another platform, `evrard_2023.py` rebuilds it from the bundled source using
`common/build_irl_paraboloid_clip_volume.sh`.  This build step requires a C++17
compiler and CMake.  If CMake is not available system-wide, place a private
CMake executable at `.tools/cmake/bin/cmake`.

On macOS, the same run can be launched by double-clicking:

```bash
RUN_RECOMPUTE_FIGURE_3_8.command
```

The recompute script removes old generated CSV/TXT/PNG/PDF files, recomputes
the eight case CSV files, redraws Figure 3-8a through Figure 3-8h, rebuilds the
eight-page `Figure_3-8.pdf`, regenerates the mesh-preview PNGs, validates the
row counts, validates the PDF page count with `pypdf`, validates the figure
outputs, and copies `Figure_3-8.pdf` to the rebuttal package when that package
folder exists.

For plot-only regeneration from already computed CSV files, run:

```bash
python3 Figure_3-8_VolumeOperatorComparison.py
```

For mesh-preview regeneration only, run:

```bash
python3 mesh_plot.py
```

## Expected Outputs

After a successful full run, these top-level files should exist:

```text
Figure_3-8a.png
Figure_3-8b.png
Figure_3-8c.png
Figure_3-8d.png
Figure_3-8e.png
Figure_3-8f.png
Figure_3-8g.png
Figure_3-8h.png
Figure_3-8.pdf
Figure_3-8_VolumeOperatorComparison_result.txt
```

The PDF should have 8 pages.  The generated CSV row counts should be:

| Case | Panel | CSV rows |
|---|---:|---:|
| `sphere` | 3-8a | 4 |
| `cylinder` | 3-8b | 4 |
| `paraboloid` | 3-8c | 4 |
| `hyperboloid` | 3-8d | 4 |
| `parabolic_cylinder` | 3-8e | 4 |
| `hyperbolic_cylinder` | 3-8f | 4 |
| `cube2sphere` | 3-8g | 1000 |
| `droplet_oscillation` | 3-8h | 2000 |

The `mesh/` folder should contain 130 method-labeled PNG previews and
`mesh/manifest.csv` with 130 data rows.  Static panels 3-8a through 3-8f have
four refinement levels and five method previews per level.  Dynamic panels 3-8g
and 3-8h show only the initial saved state `t0` for each method.

Static panels plot relative volume error against the number of surface
triangles.  Dynamic panels plot relative volume error against iteration.

The Linux/Marie plots may not be byte-identical to macOS plots because
matplotlib uses the available system font when Arial is absent.  The numerical
CSV values are expected to match the macOS results up to floating-point
roundoff.

## What Is Being Compared

All curves are reported as volume error relative to a common PL baseline.
The comparison isolates the volume-operator behavior, not complete VOF
transport solvers.  This is important because several cited methods were
developed for different numerical pipelines.  In this folder, they are recast
as volume operators so the same case geometry can be compared panel by panel.

The present-method line uses the manuscript curved-volume implementation.
The literature lines are reference operators matched to their published
geometric idea:

| Method column | Main file | Literature reference | What is implemented here |
|---|---|---|---|
| PLIC / PL | `plic_pl_1998_1999.py` | Rider and Kothe (1998); Scardovelli and Zaleski (1999) | Closed triangular-surface volume baseline. |
| Evrard-type paraboloid | `evrard_2023.py` | Evrard et al. (2023) | Paraboloid clipping for the model-matched paraboloid case; local paraboloid approximation for other surface meshes. |
| THINC/QQ | `xie_xiao_2017.py` | Xie and Xiao (2017) | Quadratic surface volume evaluated by Gaussian quadrature. |
| Strobl sphere overlap | `strobl_2016.py` | Strobl et al. (2016) | Least-squares sphere fit followed by sphere/hexahedron overlap volume. |
| Present quadric patch | `present_quadric_patch_2026.py` | Present manuscript | Class-aware closed-form quadric-patch volume operator. |

## Third-party Code

This folder contains author-created reproducibility scripts and a few bundled
third-party source trees needed to rebuild comparison helpers. The third-party
source trees keep their original license files:

- `common/_external_irl_quadratic_cutting/`: Interface Reconstruction Library
  source used to build the Evrard-type paraboloid-clipping helper; see its
  `LICENSE.txt` and README.
- `common/_external_eigen/`: Eigen headers used when building the IRL helper;
  see its `LICENSE` and `COPYING.*` files.
- `neatplot-main/`: local plotting style package; see its MIT `LICENSE`.

Python packages are installed from `requirements.txt` into `_python_deps/`
when the scripts run; `_python_deps/` is not part of the Zenodo package.

The author-year files in this folder are comparison wrappers for Figure 3-8.
They are not full redistributed releases of the original literature codes.

For the five static geometries shared with manuscript Table 3-1 (`cylinder`,
`paraboloid`, `hyperboloid`, `parabolic_cylinder`, and
`hyperbolic_cylinder`), the Table 3-1 PL and present-method entries are
anchored to the same mesh generators and per-triangle correction CSVs used for
Table 3-1.  For 3-8e, the sweep starts from the Table 3-1 exterior mesh and
then uses three midpoint refinements of that mesh.  The copied Table 3-1
correction files are stored in `source_data/table_3_1/`, and
`table31_reference.py` recomputes the anchor-row errors from those files before
the case CSVs are written.

## Equations And Functions By File

`plic_pl_1998_1999.py`

```text
V_PL = abs((1/6) * sum_faces x_a dot (x_b cross x_c))
```

Functions used:

```text
signed_closed_surface_volume(points, faces)
closed_surface_volume(points, faces)
```

This is the common piecewise-linear closed-surface baseline.

`evrard_2023.py`

Model-matched paraboloid clipping:

```text
z = a*x^2 + b*y^2
V = zeroth moment of tetrahedra clipped by the prescribed paraboloid
```

Local paraboloid approximation for other panels:

```text
x(u,v) = p0 + u*e1 + v*e2 + h(u,v)*n
h(u,v) = a*u^2 + b*u*v + c*v^2 + d*u + e*v + f
V_face = (1/3) * int_Omega [
    x dot n - (x dot e1) * dh/du - (x dot e2) * dh/dv
] du dv
```

Functions used:

```text
run_irl_paraboloid_clip(...)
paraboloid_forward_volume(...)
paraboloid_taylor_volume(...)
surface_ppic_volume(...)
```

The model-matched paraboloid panel uses a separate background tetrahedral mesh
for the prescribed-paraboloid clipping operator.  Other panels use the same
surface mesh through a local paraboloid approximation.

`xie_xiao_2017.py`

Quadratic interface model:

```text
F(x,y,z) =
    A*x^2 + B*y^2 + C*z^2 + D*x*y + E*x*z + F*y*z
    + G*x + H*y + I*z + J = 0
```

Volume integral:

```text
V_face = (1/3) * int_S x dot n_S dA
```

For a local graph this becomes:

```text
V_face = (1/3) * int_Omega [
    x dot n - (x dot e1) * dh/du - (x dot e2) * dh/dv
] du dv
```

Functions used:

```text
gaussian_quadric_volume(...)
local_quadratic_gq_volume(...)
implicit_quadric_graph_volume_gq(...)
graph_volume_gq(...)
```

This comparison uses the quadratic-surface / Gaussian-quadrature part of the
THINC/QQ idea.  It is not a full THINC/QQ advection solver.

`strobl_2016.py`

Sphere fit:

```text
|x - center|^2 = R^2
```

Volume:

```text
V = sum_cells overlap_volume(Sphere(center, R), Hexahedron(cell))
```

Functions used:

```text
fit_sphere(points, vertex_ids)
sphere_hexgrid_volume(points, hexgrid_n=4)
```

This is a sphere-special reference.  It is model-matched for spherical
geometry, but it intentionally degrades for non-spherical quadrics because a
single fitted sphere cannot represent cylinders, paraboloids, hyperboloids, or
saddle-type patches.

`present_quadric_patch_2026.py`

The present method calls the manuscript implementation in `_curved_volume.py`:

```text
manuscript_curved_volume(points, faces, workdir, ...)
```

Conceptually, the operator performs:

```text
fit local quadric
classify canonical quadric type
compute V_patch by class-specific closed-form subtraction
add/subtract V_patch relative to the PL closure
```

This is the line used for the present method in Figure 3-8.

## Geometry And Mesh Files

Static geometry generators:

```text
sphere_mesh.py
cylinder_mesh.py
paraboloid_mesh.py
hyperboloid_mesh.py
parabolic_cylinder_mesh.py
hyperbolic_cylinder_mesh.py
```

Each file exposes `generate_mesh(level)`.  The six static case folders call
these functions before computing the method columns.

Dynamic source states are stored locally in:

```text
source_data/figure_3_3/cube_present_surface_states/
source_data/figure_3_7/droplet_present_surface_states/
```

For dynamic panels, PL and present-method values are read from saved full-solver
summary files.  The Evrard-type, THINC/QQ-type, and Strobl-type comparison
columns are recomputed from the saved surface meshes.

The mesh previews are generated by:

```bash
python3 mesh_plot.py
```

Most methods use the same current triangular surface mesh.  The previews are
still method-labeled to make this explicit.  The Evrard-type model-matched
paraboloid case displays a 3D wireframe cutaway of the background tetrahedral
mesh used by the clipping operator.  The Strobl preview displays the fitted
sphere and the 4^3 hexahedral overlap grid used for the sphere/mesh-element
overlap.

## Current Reproduced Results

The latest reproduced finest-level/static and final-iteration/dynamic relative
volume errors are:

| Case | PLIC / PL | Evrard type | THINC/QQ | Strobl type | Present method |
|---|---:|---:|---:|---:|---:|
| Sphere | 0.861% | 0.384% | 0.379% | 5.51e-13% | 1.91e-13% |
| Cylinder | 0.0168% | 0.00963% | 0.00761% | 15.6% | 9.98e-10% |
| Paraboloid | 0.0919% | 4.38e-11% | 0.0370% | 15.0% | 1.59e-5% |
| Hyperboloid | 0.0142% | 0.0220% | 0.00926% | 27.1% | 4.65e-4% |
| Parabolic cylinder | 0.0769% | 0.0326% | 0.0382% | 20.6% | 4.12e-7% |
| Hyperbolic cylinder | 0.00816% | 0.00195% | 0.00371% | 70.2% | 7.95e-9% |
| Cube-to-sphere final | 7.14% | 4.46% | 1.38% | 0.0736% | 0.0443% |
| Oscillating droplet final | 0.757% | 0.584% | 0.334% | 0.155% | 6.53e-4% |

These numbers should match `Table 3-3` in the manuscript after the CSV files are
regenerated.

## What Works

`recompute_all_cases.py` has been run successfully in this folder.  It
regenerates the eight case CSV files, the eight PNG panels, the eight-page PDF,
and the 130 mesh-preview PNGs.  It also validates expected row counts and output
existence before finishing.

The top-level method files contain the references, equations, and callable
functions used by the case scripts.  A reader should not need to search through
hidden helper files to understand which equation a literature line represents.

The folder is self-contained for this figure: it includes local method files,
local mesh generators, local saved source data, local helper libraries, and
dependency bootstrap code.

## Known Issues And Limits

The literature curves are comparison operators, not complete reproductions of
the original papers' full simulation pipelines.  This is intentional because
Section 3.4 asks a specific question: what happens when different curved-volume
operators are compared against the same PL baseline?

The Evrard-type line is exact and model-matched only in the prescribed
paraboloid clipping panel.  In other panels it is a local paraboloid
approximation applied to the same surface-mesh setting, so it should not be
read as a full general Evrard solver.

The Strobl-type line uses a fitted sphere and a hexahedral overlap grid.  It is
therefore expected to perform best for the sphere and poorly for non-spherical
quadric classes.

The THINC/QQ line uses a quadratic surface and Gaussian quadrature.  It
represents the volume-integration behavior of a THINC/QQ-style quadratic model,
not the full transport/advection algorithm.

Very new Python versions can have package-wheel issues for the `overlap`
dependency.  Python 3.12 or 3.13 is the safest choice.  The bootstrap script
tries to use a dependency-ready runtime first and otherwise installs packages
locally into `_python_deps`.

The local dependency list is `requirements.txt`: `numpy`, `pandas`,
`matplotlib`, `meshio`, `overlap`, `pypdf`, `reportlab`, and `Pillow`.

## File Map

Top-level recomputation and plotting:

```text
recompute_all_cases.py          full recomputation and validation
RUN_RECOMPUTE_FIGURE_3_8.py      small Python launcher
RUN_RECOMPUTE_FIGURE_3_8.command macOS double-click launcher
Figure_3-8_VolumeOperatorComparison.py          plot from generated CSV files
mesh_plot.py                    regenerate mesh-preview PNGs
```

Method definitions:

```text
plic_pl_1998_1999.py
evrard_2023.py
xie_xiao_2017.py
strobl_2016.py
present_quadric_patch_2026.py
```

Case folders:

```text
sphere/
cylinder/
paraboloid/
hyperboloid/
parabolic_cylinder/
hyperbolic_cylinder/
cube2sphere/
droplet_oscillation/
```

Generated outputs:

```text
*_all_methods_result.csv        case-level numerical data
Figure_3-8*.png                 plotted panels
Figure_3-8.pdf                  combined eight-page PDF
mesh/*.png                      method-labeled mesh previews
mesh/manifest.csv               mesh-preview metadata
```
