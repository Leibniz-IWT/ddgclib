# CFD-DEM Two-Particle Liquid Bridge Case Study

Coupled CFD-DEM simulation of capillary bridge formation between two wetted
silica glass spheres surrounded by a quiescent water film.

## What This Demonstrates

- **Two-way fluid-particle coupling** via `FluidParticleCoupler` (Stokes drag
  + Newton III feedback onto the fluid mesh)
- **Operator-splitting loop**: fluid step -> coupling -> DEM step -> feedback
- **Comparison with pure DEM**: results should be nearly identical since
  Stokes drag (~0.01 uN) is negligible compared to capillary force (~0.4 uN)

## Coupling Workflow

```
For each timestep:
  1. Fluid velocity update     (explicit Euler on fixed 3D mesh)
  2. Fluid -> particle         (IDW interpolation + Stokes drag)
  3. DEM step                  (Hertz contact + capillary bridge + drag)
  4. Particle -> fluid         (reaction force distributed to mesh)
```

## Running

First run the pure-DEM case to generate reference data for comparison plots:

```bash
cd cases_dynamic/liquid_bridge_dem
python liquid_bridge_case.py
```

Then run the coupled case:

```bash
cd cases_dynamic/liquid_bridge_cfd_dem
python liquid_bridge_cfd_dem_case.py
```

## Output

- `fig/separation.png` — Separation vs time (with DEM overlay)
- `fig/velocity.png` — Particle velocities vs time
- `fig/forces.png` — Capillary + drag forces vs time
- `fig/summary.png` — Combined multi-panel figure
- `results/history.json` — Time-series data
- `results/final_state.json` — Final DEM state

## Expected Results

With default parameters (1 mm silica glass, water film):

- **Bridge forms** at ~30 ms (same as pure DEM)
- **Final separation** matches DEM within < 1%
- **Stokes drag** peaks at ~0.01 uN (3 orders of magnitude below capillary)
- **CoM drift** ~ 0 (momentum conservation)

## Parameters

All DEM parameters are inherited from `liquid_bridge_dem/src/_params.py`.
Fluid-specific parameters in `src/_params.py`:

```python
mu_f = 8.9e-4       # water viscosity [Pa s]
rho_f = 1000.0      # water density [kg/m^3]
drag_model = "stokes"
```

## References

- Same as the pure-DEM case (Lian 1993, Willett 2000)
- Stokes drag: F = 6 pi mu R v_rel
