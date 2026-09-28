# Dam Break Case Study

Classic multiphase dam break — a rectangular water column collapses
under gravity inside a rectangular tank.  Used to test the full
dynamic pipeline end-to-end: gravity, pressure, viscous shear,
retopologisation, and (in the multiphase variant) surface tension
at the water–air interface.

## Geometry

```
y=H     +---------------+
        |               |
        | air           |
y=col_h +----+          |
        |    |          |
        |water          |
y=0     +----+----------+
        x=0  col_w     L
```

In 3D the same layout is extruded a depth ``W`` in z with the column
occupying ``col_d`` in that direction.  Parameters live in
``src/_params.py``.

## Running the simulations

```bash
# 2D multiphase (liquid + air, with surface tension)
python cases_dynamic/dam_break/dam_break_2D.py

# 2D single-phase (liquid only, free surface at atmospheric)
python cases_dynamic/dam_break/dam_break_2D_no_air.py

# 3D multiphase
python cases_dynamic/dam_break/dam_break_3D.py

# 3D single-phase
python cases_dynamic/dam_break/dam_break_3D_no_air.py
```

## Variants

- ``dam_break_2D.py`` / ``dam_break_3D.py`` — **multiphase**
  (phase 0 = air, phase 1 = water).  Uses
  ``ddgclib.operators.multiphase_stress.multiphase_dudt_i`` which
  includes interface curvature → surface tension at the sharp
  liquid/air interface.  This is the primary test for the surface
  tension operator.

- ``dam_break_2D_no_air.py`` / ``dam_break_3D_no_air.py`` —
  **single phase**.  The mesh covers only the water column; the
  bottom and "upstream" sides are frozen no-slip walls while the
  top and "downstream" sides are free surfaces.  The absolute
  pressure is tracked by the Tait–Murnaghan EOS, initialised from
  ``P(y) = P_atm + rho_l * g * (col_h - y)``.  No explicit
  boundary condition is applied at the free surface — the EOS
  drives the pressure at those vertices toward the atmospheric
  reference.

## Outputs

All outputs are saved **within this case directory**:

```
cases_dynamic/dam_break/
    fig/
        dam_break_2D_fluid.png
        dam_break_2D_phases.png
        dam_break_2D.mp4
        dam_break_2D_no_air_fluid.png
        dam_break_2D_no_air.mp4
        dam_break_3D.mp4
        dam_break_3D_no_air.mp4
        ...
    results/
        snapshots_2D/          # JSON snapshots for polyscope replay
        snapshots_2D_no_air/
        snapshots_3D/
        snapshots_3D_no_air/
```

## Viewing results in polyscope

```bash
python -m ddgclib.scripts.view_polyscope \
    --snapshots cases_dynamic/dam_break/results/snapshots_2D/ \
    --scalars p --vectors u
```

## Parameters

See ``src/_params.py``.  The default column is ``a × 2a`` with
``a = 0.05 m`` (water, gamma = 0.072 N/m); the tank is ``4a × 2a``
in 2D and ``4a × 2a × 2a`` in 3D.

``t_end`` defaults to a **short** run (``0.02 s``, i.e. ``~0.2
t_ref``) so the smoke test passes out of the box.  See the
stability note below before extending it.

## Stability note

The current DDG FVM stress operator develops large spurious
accelerations at free-surface and multiphase-interface corner
vertices where the barycentric dual cell is truncated.  The dam
break is a demanding test in this regard because it has two
adjacent free surfaces meeting at a (possibly moving) corner.

To keep the out-of-the-box run tractable this case uses:

- artificial viscosity ``alpha_art = 2.0`` (SPH-style,
  ``mu_art = alpha * rho * c_s * dx``);
- a short ``t_end = 0.02 s`` and a small CFL factor ``cfl = 0.1``;
- ``skip_triangulation=True`` in the multiphase variants to keep
  the initial Delaunay connectivity frozen (avoids cross-phase
  edges from re-triangulation — see
  ``FEATURES.md: AMR remeshing``);
- a ``boundary_filter`` in the single-phase variants so only the
  tank walls are frozen while the free surface advects;
- a ``try/except`` around the integration loop so partial output
  is still saved if the simulation diverges.

For longer / higher-fidelity runs the case needs the interface
curvature and dual-split improvements listed under ``Planned`` in
``FEATURES.md``.
