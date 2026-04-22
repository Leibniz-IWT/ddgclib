# Shearing-plate droplet

An oil droplet of radius `R0` is placed at the centre of a water-filled
channel between two counter-moving plates.

```
         -----> U_wall              (top plate, no-slip, y = +L_y)
         =============
            |  _   |
            | (oil)|                  Ca ~ mu_o * U_wall * R0 / (gamma * L_y)
            |  -   |                  lambda = mu_d / mu_o
         =============
         <----- U_wall              (bottom plate, no-slip, y = -L_y)
```

- x direction (and z in 3D) is **periodic**, so the channel represents
  an infinite array of oil droplets in a simple-shear background.
- Top / bottom plates are no-slip moving walls applied via the
  `MovingWallBC` boundary condition that was added to
  `ddgclib/_boundary_conditions.py` as part of this case.
- Initial state is the **equilibrium droplet** from the oscillating
  droplet setup (no perturbation); deformation arises purely from
  the imposed shear.

## Running

```bash
# 2D
python cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_2D.py

# 3D (coarser mesh, shorter window)
python cases_dynamic/shearing_plate_droplet/shearing_plate_droplet_3D.py
```

## Outputs

All outputs are written under this directory:

```
cases_dynamic/shearing_plate_droplet/
    fig/
        shearing_plate_droplet_2D_fluid.png
        shearing_plate_droplet_2D_profile.png      # mean u_x(y) vs Couette
        shearing_plate_droplet_2D_deformation.png  # D(t), tilt(t)
        shearing_plate_droplet_2D.mp4              # animation
        ...                                         (3D equivalents)
    results/snapshots/        # 2D StateHistory
    results_3d/snapshots/     # 3D StateHistory
```

## Viewing results in polyscope

```bash
python -m ddgclib.scripts.view_polyscope \
    --snapshots cases_dynamic/shearing_plate_droplet/results/snapshots/ \
    --scalars p --vectors u
```

## Physics

The governing equations follow the standard multiphase pipeline
(see `Fundamentals.md` and
`cases_dynamic/oscillating_droplet/INTERFACE_STRESS.md`):

- Per-parcel Cauchy momentum:  `m_i dv_i/dt = F_stress + F_surface_tension`
- Newtonian pressure + viscous stress with own-phase viscosity
- Sharp-interface surface tension on vertices with `v.is_interface = True`
- Weakly-compressible Tait–Murnaghan EOS per phase

Boundary conditions:

- `MovingWallBC(+U_wall e_x)` on top plate, `MovingWallBC(-U_wall e_x)` on
  bottom plate — these just impose a fixed tangential velocity on the
  frozen wall vertices; the walls themselves do not translate.
- Periodic in x (and z): handled by
  `ddgclib.geometry.periodic.retopologize_periodic`, wrapped in a
  multiphase refresh composed inside `src/_setup.py`.

## Validation reference

For `Ca << 1` and Stokes flow, Taylor (1934) predicts the steady-state
deformation parameter:

    D = (L - B) / (L + B) = Ca * (19*lambda + 16) / (16*lambda + 16)

The driver prints this prediction alongside the measured `D` from a
principal-axis fit of the interface vertices.  At the default
parameters, the simulation is mildly inertial (`Re ~ O(1)`) so the
measured `D` will generally exceed the Stokes prediction somewhat —
this test case is a qualitative shape-response benchmark rather than
a quantitative Stokes verification.
