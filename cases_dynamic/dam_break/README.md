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

See ``src/_params.py``.  The default column is the square ``a × a``
(Martin and Moyce column with headspace, laneF 2026-07-30) with
``a = 0.05 m`` (water, gamma = 0.072 N/m); the tank is ``4a × 2a``
in 2D and ``4a × 2a × 2a`` in 3D.  ``t_end = 0.2 s`` (about 2.8
``t_ref = sqrt(a / g)``), ``cfl = 0.1``, ``alpha_art = 0.3``.

## Methods and status

The multiphase runners consume the presets ``dam_break_2D``
(per-step Delaunay + conservative remap, ``frozen_set='membership'``)
and ``dam_break_3D`` (``dual_only``, exact dual faces
``edge_area_source='p_ij_simplex'``) from ``ddgclib.methods`` and
write ``results/methods_2D.json`` / ``methods_3D.json``.  The axes
``phase_ledger='volume'`` and ``face_closure='renormalise'`` (defaults
since laneF, 2026-10-05) are what let the 2D run survive its
reconnections at ``alpha_art`` 0.2 and 0.1; the old values are kept as
``'snapshot'`` / ``'skip'`` (status broken) for reproduction.  The
hydrostatic per-phase mass preload is made on the vote labels the
integrator uses (3D: the spatial-criterion labels differ).

Regression pins (``ddgclib/tests/test_case_dam_break.py``): 2D at
refinement 2 / alpha 0.1 (fast) and at the shipped refinement 3 /
alpha 0.2 over the full horizon (slow); 3D 50-step smoke (fast) and
full horizon (slow).  Diagnostics: ``diagnose_sliver_ejection.py``
(per-vertex force trace, hole census, ``--replace axis=value`` arms,
``--dim 3``) and ``diagnose_phase_ledger.py`` (presence-change census
of the shipped multiphase cases).  Lane log:
``docs_temp/debug_session/laneF-ledger-and-face-closure.md``.

## Known limits

- The liquid toe becomes one cell thick during the collapse at
  ``alpha_art <= 0.2`` (refinement 3).  Under the ``neighbour_count``
  split its vertices then lose their liquid sub-volume, the volume
  ledger releases their mass into the liquid pool and the interface
  pressure jump kicks the light vertices: a transient of about 0.01 s
  (KE of liquid plus interface 3.1e-3 -> 3.5e-2 J at alpha 0.2) and a
  front measure that retreats.  ``split_method='simplex'`` keeps the
  tongue (opt-in; changes every run).
- Refinement 4 is not carried through the horizon (laneF known
  limit, see the lane log).
- The 3D wall cells are zeroed under ``dual_only``, so the measured
  liquid volume is about half the column; the 3D column creeps
  (effective viscosity 105.8 Pa s).
- The single-phase ``*_no_air`` runners are laneK's measured-unstable
  configuration and are unvalidated.
